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
"""benchdnn backend: one native input line per problem.

The lines are benchdnn's matmul input lines (BF16 weights, or DA8W8: a bf16
source quantized per token with s8 weights and per-channel scales; no other
weight type or scheme) with the kernel left to the library
(none), so the job's ZENDNNL_MATMUL_ALGO=10 selects the native BRGEMM as it does
for a model's calls.
Each run sets ZENDNNL_MATMUL_NATIVE_CONFIG for its slot (jobs.Session) and runs
benchdnn --op=matmul on its problems' lines; the per-call time is the result
line's last column (total ms over iters) divided by iters.  By default every
line runs in its own process: in one process a freed weight buffer can come
back at an address inside an earlier line's in-place weight image, which the
native weight cache refuses.  Timing is benchdnn's (weights stay warm).
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..problems import Problem
from ..results import median
from .common import Backend

BF16_LINE = ("{m}, {k}, {n}, {iters}, bf16:bf16:{dst}, {bias}, f32, , , none, true, {ta}, "
             "false, 1.0, 0.0, per-channel, , f32, {warmup}")
S8_LINE = ("{m}, {k}, {n}, {iters}, {src}:s8:{dst}, {bias}, f32, , , none, true, {ta}, "
           "false, 1.0, 0.0, per-channel, , f32, {warmup}, true, per-token, 0, f32")


def default_iters(m: int) -> Tuple[int, int]:
    """(iters, warmup) of the native input files for M."""
    if m <= 64:
        return 100, 20
    if m <= 128:
        return 50, 10
    if m <= 512:
        return 20, 5
    return 10, 3


def bd_line(p: Problem, iters: int, warmup: int) -> str:
    fmt = BF16_LINE if p.bf16 else S8_LINE
    return fmt.format(m=p.m, k=p.k, n=p.n, iters=iters, warmup=warmup, dst=p.dst, src=p.src,
                      bias="true" if p.has_bias else "false", ta="true" if p.trans_a else "false")


def parse_result_line(line: str) -> Optional[Dict[str, Any]]:
    """m, n, k, weight type, iters and per-call ms of one benchdnn result line, or None."""
    f = [x.strip() for x in line.split(",")]
    if len(f) < 20 or not f[0].isdigit():
        return None
    try:
        m, k, n, iters = int(f[0]), int(f[1]), int(f[2]), int(f[3])
        total = float(f[-1])
    except ValueError:
        return None
    dt = f[4].split(":")
    if len(dt) != 3 or iters < 1 or not total > 0:
        return None
    return {"m": m, "n": n, "k": k, "wei": dt[1], "iters": iters, "time_ms": total / iters, "total_ms": total}


class BenchdnnBackend(Backend):
    name = "benchdnn"

    def __init__(self, binary: str = "benchdnn", threads: int = 32, per_line: bool = True,
                 iters_scale: float = 1.0):
        super().__init__(threads)
        self.binary = binary
        self.per_line = per_line
        self.iters_scale = iters_scale

    def supports(self, p: Problem) -> Optional[str]:
        why = super().supports(p)
        if why:
            return why
        if p.dtype not in ("bf16", "s8") or p.int8_scheme != "per_channel":
            return "benchdnn lines here take bf16 weights or s8 weights with per-channel scales"
        if p.src != "bf16":
            return "benchdnn lines here take a bf16 source"
        if not p.bf16 and not p.dynamic:
            return "s8 lines here quantize the source in the call (dynamic)"
        if p.epilogue not in ("store", "bias"):
            return "no post-op lines"
        return None

    @staticmethod
    def input_name(p: Problem, probe: bool) -> str:
        h = hashlib.sha1(p.key().encode()).hexdigest()[:8]
        return "inputs/%s_m%d_n%d_k%d_%s%s.txt" % (p.dtype, p.m, p.n, p.k, h, ".probe" if probe else "")

    def write_inputs(self, sdir: str, gid: int, problems: Sequence[Problem]) -> None:
        for p in problems:
            iters, warmup = default_iters(p.m)
            iters = max(1, int(round(iters * self.iters_scale)))
            warmup = max(1, int(round(warmup * self.iters_scale)))
            for probe, (it, wu) in ((False, (iters, warmup)), (True, (1, 1))):
                with open(os.path.join(sdir, self.input_name(p, probe)), "w", encoding="utf-8") as f:
                    f.write(bd_line(p, it, wu) + "\n")

    def preamble(self) -> List[str]:
        return [
            "for v in $(compgen -e | grep -E '^(ZENDNNL_|KMP_)'); do unset \"$v\"; done",
            "unset OMP_WAIT_POLICY",
            "export OMP_NUM_THREADS=%d OMP_PROC_BIND=close OMP_PLACES=cores" % self.threads,
            "BD=%s" % self.binary,
            'mkdir -p "$S/work" && cd "$S/work"',
            "# bdl: one benchdnn process per line; bdw: all lines in one process",
            'bdl() { local f; for f in "$@"; do "$BD" --op=matmul --input_file="$S/$f" || '
            + 'echo "benchdnn rc=$? input=$f"; done; }',
            'bdw() { local f; for f in "$@"; do cat "$S/$f"; done > "$S/work/lines.txt"; '
            + '"$BD" --op=matmul --input_file="$S/work/lines.txt"; }',
        ]

    def command(self, gid: int, problems: Sequence[Problem], probe: bool) -> str:
        files = " ".join(self.input_name(p, probe) for p in problems)
        return ("bdl " if self.per_line else "bdw ") + files

    def parse_timed(self, text: str, problems: Sequence[Problem]) -> Dict[str, Dict[str, Any]]:
        found: Dict[Tuple[str, int, int, int], List[float]] = {}
        for line in text.splitlines():
            r = parse_result_line(line)
            if r is not None:
                found.setdefault((r["wei"], r["m"], r["n"], r["k"]), []).append(r["time_ms"])
        out = {}
        for p in problems:
            ts = found.get((p.dtype, p.m, p.n, p.k))
            if ts:
                out[p.key()] = {"time_ms": median(ts), "stats": {"lines": len(ts)}}
        return out

    def describe(self) -> Dict[str, Any]:
        return {"backend": self.name, "binary": self.binary, "threads": self.threads,
                "per_line": self.per_line, "iters_scale": self.iters_scale}
