# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
"""benchdnn matmul results as (shape, per-call time) pairs.

Two inputs: the CSV file benchdnn writes (a header row, then one row per call:
"[Layer Number, Model Name, BS,] M, K, N, Iterations, Data type, Bias Enabled, Bias
Data type, Post Operation, PostOp Data type, ..., Weight Group Size, Weight Scale Data
type, ..., Src Dynamic Quant, ..., Src Scale Data type, Total time (ms) (all iters), Avg
time (ms), GFLOPS, ..."), read by column name; or the result lines it prints, as the
native BRGEMM tuner reads them: the fields of a matmul input line (benchdnn/doc/
matmul.md) with the total ms last.
"""

from __future__ import annotations

import csv
import io
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from op_perf_utility.matmul.work import INT_WEIGHTS, TYPE_BYTES, MatmulShape

SCALE_BYTES = {"f32": 4, "bf16": 2, "f16": 2}


@dataclass(frozen=True)
class Measured:
    shape: MatmulShape
    time_ms: float
    iters: int
    source: str


def _truthy(v: Optional[str]) -> bool:
    return (v or "").strip().lower() in ("1", "true", "yes", "on")


def _int(v: Optional[str]) -> int:
    v = (v or "").strip()
    return int(v) if v.isdigit() else 0


def _num(v: Optional[str]) -> float:
    try:
        return float((v or "").strip())
    except ValueError:
        return 0.0


def parse_types(text: str) -> Optional[Sequence[str]]:
    """(src, wei, dst) of "src:wei:dst", or None."""
    t = [x.strip() for x in text.split(":")]
    if len(t) != 3 or not all(x in TYPE_BYTES for x in t):
        return None
    return t


def parse_post_ops(post_ops: str, dt: str) -> Tuple[int, str]:
    """The binary post-ops of a "binary_add:relu" list, each reading an M x N tensor, and the
    type of those tensors (f32 unless dt names another)."""
    n = sum(1 for op in post_ops.split(":") if op.strip().lower().startswith("binary"))
    dt = dt.split(":")[0].strip()
    return n, dt if dt in TYPE_BYTES else "f32"


def _group(granularity: str, size: int, k: int) -> Tuple[bool, int]:
    """(one scale for the tensor, the K extent of a scale group or 0) of a scale granularity
    (per-tensor, per-channel or per-token, per-group with its size)."""
    g = granularity.strip().lower()
    group = size if ("group" in g or not g) and 0 < size < k and k % size == 0 else 0
    return "tensor" in g, group


def _shape(m: int, n: int, k: int, batch: int, types: Sequence[str], bias: str, threads: int,
           post: Tuple[int, str] = (0, "f32"), wei_scales: Tuple[str, int, str] = ("", 0, ""),
           src_quant: bool = False, src_scales: Tuple[str, int, str] = ("", 0, "")) -> MatmulShape:
    """A shape of benchdnn's fields; *_scales are (granularity, group size, scale type).  A
    source has scales when the call quantizes it, or when it is integer and names a scale
    type (static scales)."""
    src, wei, dst = types
    int_wei = wei in INT_WEIGHTS
    wei_tensor, wei_group = _group(wei_scales[0], wei_scales[1], k) if int_wei else (False, 0)
    scaled = src_quant or (src in ("u8", "s8") and bool(src_scales[2]))
    src_tensor, src_group = _group(src_scales[0], src_scales[1], k) if scaled else (False, 0)
    return MatmulShape(m=m, n=n, k=k, batch=max(1, batch), src=src, wei=wei, dst=dst, bias=bias,
                       wei_group=wei_group, wei_scale_tensor=wei_tensor,
                       wei_scale_bytes=SCALE_BYTES.get(wei_scales[2], 4) if int_wei else 0,
                       src_scale_bytes=SCALE_BYTES.get(src_scales[2], 4) if scaled else 0,
                       src_group=src_group, src_scale_tensor=src_tensor,
                       binary=post[0], binary_dt=post[1], threads=threads)


def _from_csv(text: str, threads: int, source: str) -> List[Measured]:
    rows = list(csv.reader(io.StringIO(text), skipinitialspace=True))
    head = next((i for i, r in enumerate(rows) if {"M", "K", "N"} <= {c.strip() for c in r}), None)
    if head is None:
        return []
    col: Dict[str, int] = {c.strip(): i for i, c in enumerate(rows[head])}

    def get(r: List[str], name: str) -> str:
        i = col.get(name)
        return r[i].strip() if i is not None and i < len(r) else ""

    out = []
    for n_row, r in enumerate(rows[head + 1:], head + 2):
        # A pipeline's Summary row (its N the layers' joined) aggregates the layers that follow.
        if get(r, "Layer Number").lower() == "summary" or not all(get(r, c).isdigit() for c in ("M", "N", "K")):
            continue
        types = parse_types(get(r, "Data type"))
        avg, total, iters = _num(get(r, "Avg time (ms)")), _num(get(r, "Total time (ms) (all iters)")), _int(
            get(r, "Iterations"))
        # Both times have two decimals: a call under 5 us averages 0.00, its total over the
        # iterations does not.
        if total > 0 and iters > 0:
            avg = total / iters
        if types is None or not avg > 0:
            continue
        bias = get(r, "Bias Data type") if _truthy(get(r, "Bias Enabled")) else ""
        out.append(Measured(_shape(int(get(r, "M")), int(get(r, "N")), int(get(r, "K")), _int(get(r, "BS")) or 1,
                                   types, bias if bias in TYPE_BYTES else ("f32" if bias else ""), threads,
                                   parse_post_ops(get(r, "Post Operation"), get(r, "PostOp Data type")),
                                   (get(r, "Weight Scale Granularity"), _int(get(r, "Weight Group Size")),
                                    get(r, "Weight Scale Data type")),
                                   _truthy(get(r, "Src Dynamic Quant")),
                                   (get(r, "Src Scale Granularity"), _int(get(r, "Src Group Size")),
                                    get(r, "Src Scale Data type"))),
                            avg, _int(get(r, "Iterations")) or 1, "%s:%d" % (source, n_row)))
    return out


# The fields of a matmul input line (benchdnn/doc/matmul.md) a printed result line repeats.
LINE_FIELDS = ("m", "k", "n", "iters", "types", "bias", "bias_dt", "post_ops", "post_op_dt", "kernel",
               "weights_const", "trans_a", "trans_b", "alpha", "beta", "wei_scale_granularity", "wei_group",
               "wei_scale_dt", "warmup", "src_dynamic_quant", "src_scale_granularity", "src_group", "src_scale_dt")


def _from_lines(text: str, threads: int, source: str) -> List[Measured]:
    out = []
    for n_line, line in enumerate(text.splitlines(), 1):
        f = [x.strip() for x in line.split(",")]
        if len(f) < 8 or not all(x.isdigit() for x in f[:4]):
            continue
        # A batched call (ndims > 2) prints its batch size first.
        batch = 1
        if parse_types(f[4]) is None and len(f) > 8 and f[4].isdigit() and parse_types(f[5]) is not None:
            batch, f = int(f[0]), f[1:]
        types = parse_types(f[4])
        try:
            total = float(f[-1])
        except ValueError:
            continue
        m, k, n, iters = (int(x) for x in f[:4])
        if types is None or iters < 1 or not total > 0:
            continue
        # The input fields before the total; a short line leaves the later ones unset.
        g = dict(zip(LINE_FIELDS, f[:-1]))
        bias_dt = g.get("bias_dt", "")
        bias = (bias_dt if bias_dt in TYPE_BYTES else "f32") if _truthy(g.get("bias")) else ""
        quant = (_truthy(g["src_dynamic_quant"]) if "src_dynamic_quant" in g
                 else types[1] in INT_WEIGHTS and types[0] in ("bf16", "f32"))
        out.append(Measured(_shape(m, n, k, batch, types, bias, threads,
                                   parse_post_ops(g.get("post_ops", ""), g.get("post_op_dt", "")),
                                   (g.get("wei_scale_granularity", ""), _int(g.get("wei_group")),
                                    g.get("wei_scale_dt") or "f32"),
                                   quant,
                                   (g.get("src_scale_granularity", ""), _int(g.get("src_group")),
                                    g.get("src_scale_dt", ""))),
                            total / iters, iters, "%s:%d" % (source, n_line)))
    return out


def read_benchdnn(path_or_text: str, threads: int = 32) -> List[Measured]:
    """The matmul results of a benchdnn output file (or its text)."""
    if os.path.exists(path_or_text):
        with open(path_or_text, encoding="utf-8", errors="replace") as f:
            text, source = f.read(), path_or_text
    else:
        text, source = path_or_text, "<text>"
    return _from_csv(text, threads, source) or _from_lines(text, threads, source)
