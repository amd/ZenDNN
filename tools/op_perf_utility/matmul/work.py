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
"""The work of one matmul call: its FLOPs and the bytes it must move.

The bytes are compulsory traffic, each operand once: the weights and their scales,
the source and the per-token scales of a source the call quantizes, the output, the
bias and the tensor each binary post-op (add, mul, ...) reads, the output's size.
Re-reads that a blocking adds are the kernel's cost, not the call's.  A batched call
has its own weights per batch.  A gated call (a gated activation over the gate and up
halves of N) writes N / 2 columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, Optional

TYPE_BYTES = {"f32": 4.0, "bf16": 2.0, "f16": 2.0, "s32": 4.0, "s16": 2.0, "s8": 1.0, "u8": 1.0,
              "s4": 0.5, "u4": 0.5}
INT_WEIGHTS = ("s8", "u8", "s4", "u4")


def _bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "on", "yes"):
        return True
    if s in ("0", "false", "off", "no", ""):
        return False
    raise ValueError("not a bool: %r" % (v,))


@dataclass(frozen=True)
class MatmulShape:
    """One matmul call: dst[M, N'] = src[M, K] x wei[K, N] (per batch), N' = N or N / 2 gated.

    wei_group is the K rows one weight scale covers (0: one scale per column;
    wei_scale_tensor: one for the tensor); wei_scale_bytes the bytes of one weight scale
    (0: weights without scales); src_scale_bytes the bytes of one source scale (0: none),
    one per row (src_group: per group of K columns; src_scale_tensor: one for the tensor),
    of a source the call quantizes or of an integer one; bias the bias type ("" for none);
    binary the binary post-ops, each reading an M x N tensor of binary_dt (the
    pre-activation of a gated call).  The bias and the binary operands are one copy the
    batch shares, as BenchDNN gives them, unless per_batch_operands."""

    m: int
    n: int
    k: int
    batch: int = 1
    src: str = "bf16"
    wei: str = "bf16"
    dst: str = "bf16"
    bias: str = ""
    wei_group: int = 0
    wei_scale_bytes: int = 0
    src_scale_bytes: int = 0
    gated: bool = False
    binary: int = 0
    binary_dt: str = "f32"
    threads: int = 32
    name: str = field(default="", compare=False)
    wei_scale_tensor: bool = False
    src_group: int = 0
    src_scale_tensor: bool = False
    per_batch_operands: bool = False

    def __post_init__(self) -> None:
        for f in ("m", "n", "k", "batch", "threads"):
            if int(getattr(self, f)) < 1:
                raise ValueError("%s must be positive: %r" % (f, self))
        for f in ("src", "wei", "dst"):
            if getattr(self, f) not in TYPE_BYTES:
                raise ValueError("%s type %r is not one of %s" % (f, getattr(self, f), ", ".join(TYPE_BYTES)))
        if self.bias and self.bias not in TYPE_BYTES:
            raise ValueError("bias type %r is not one of %s" % (self.bias, ", ".join(TYPE_BYTES)))
        if self.wei_group < 0 or (self.wei_group and self.k % self.wei_group):
            raise ValueError("a weight scale group of %d rows does not divide K=%d" % (self.wei_group, self.k))
        if self.src_group < 0 or (self.src_group and self.k % self.src_group):
            raise ValueError("a source scale group of %d columns does not divide K=%d" % (self.src_group, self.k))
        if (self.wei_scale_tensor and self.wei_group) or (self.src_scale_tensor and self.src_group):
            raise ValueError("a tensor's scales are either one for the tensor or per group: %r" % (self,))
        if self.wei_scale_bytes < 0 or self.src_scale_bytes < 0 or self.binary < 0:
            raise ValueError("scale bytes and binary post-ops must not be negative: %r" % (self,))
        if self.binary_dt not in TYPE_BYTES:
            raise ValueError("binary post-op type %r is not one of %s" % (self.binary_dt, ", ".join(TYPE_BYTES)))
        if self.gated and self.n % 2:
            raise ValueError("a gated call needs an even N, not %d" % self.n)

    @classmethod
    def parse(cls, text: str, threads: Optional[int] = None) -> "MatmulShape":
        """"m=4,n=4096,k=4096,wei=s8,..." (',' separated); threads fills an unset thread count."""
        names = {f.name: f for f in fields(cls)}
        d: Dict[str, Any] = {}
        for item in filter(None, (t.strip() for t in text.split(","))):
            k, eq, v = item.partition("=")
            k = k.strip().lower()
            if not eq or k not in names:
                raise ValueError("bad shape field %r (fields: %s)" % (item, ", ".join(names)))
            d[k] = v.strip()
        for k in ("m", "n", "k"):
            if k not in d:
                raise ValueError("a shape needs m, n and k: %r" % text)
        for k in ("m", "n", "k", "batch", "threads", "wei_group", "wei_scale_bytes", "src_scale_bytes", "binary",
                  "src_group"):
            if k in d:
                d[k] = int(d[k])
        for k in ("gated", "wei_scale_tensor", "src_scale_tensor", "per_batch_operands"):
            if k in d:
                d[k] = _bool(d[k])
        if "threads" not in d and threads:
            d["threads"] = threads
        if d.get("wei") in INT_WEIGHTS and "wei_scale_bytes" not in d:
            d["wei_scale_bytes"] = 4
        return cls(**d)

    def label(self) -> str:
        wei = "%s g%d" % (self.wei, self.wei_group) if self.wei_group else self.wei
        b = "%dx" % self.batch if self.batch > 1 else ""
        post = " +%d %s binary" % (self.binary, self.binary_dt) if self.binary else ""
        return "%s%s:%s:%s %sM=%d N=%d K=%d%s%s" % (self.name + " " if self.name else "", self.src, wei, self.dst,
                                                    b, self.m, self.n, self.k, " gated" if self.gated else "", post)


@dataclass(frozen=True)
class MatmulWork:
    """FLOPs and compulsory bytes of one call."""

    flops: float
    wei_bytes: float
    wei_scale_bytes: float
    src_bytes: float
    src_scale_bytes: float
    dst_bytes: float
    bias_bytes: float
    post_bytes: float = 0.0
    # The u8 copy of a source the call quantizes: written once, then read by the tiles.
    src_quant_bytes: float = 0.0

    @property
    def weight_bytes(self) -> float:
        return self.wei_bytes + self.wei_scale_bytes

    @property
    def write_bytes(self) -> float:
        """The bytes the call stores: the output and a quantized source's copy."""
        return self.dst_bytes + self.src_quant_bytes

    @property
    def footprint_bytes(self) -> float:
        """The bytes of the call's distinct buffers, which decide where it lives."""
        return (self.wei_bytes + self.wei_scale_bytes + self.src_bytes + self.src_scale_bytes
                + self.dst_bytes + self.bias_bytes + self.post_bytes + self.src_quant_bytes)

    @property
    def total_bytes(self) -> float:
        """The bytes the call moves: a quantized source's copy is written and read back."""
        return self.footprint_bytes + self.src_quant_bytes


def work(s: MatmulShape) -> MatmulWork:
    b, m, n, k = s.batch, s.m, s.n, s.k
    groups = k // s.wei_group if s.wei_group else 1
    out_cols = n // 2 if s.gated else n
    quantized = s.src_scale_bytes > 0 and s.wei in INT_WEIGHTS and s.src not in ("u8", "s8")
    wei_scales = 1 if s.wei_scale_tensor else b * n * groups
    src_scales = 1 if s.src_scale_tensor else b * m * (k // s.src_group if s.src_group else 1)
    ob = b if s.per_batch_operands else 1  # copies of the bias and the binary operands
    return MatmulWork(
        flops=2.0 * b * m * n * k,
        wei_bytes=b * n * k * TYPE_BYTES[s.wei],
        wei_scale_bytes=float(wei_scales * s.wei_scale_bytes),
        src_bytes=b * m * k * TYPE_BYTES[s.src],
        src_scale_bytes=float(src_scales * s.src_scale_bytes),
        dst_bytes=b * m * out_cols * TYPE_BYTES[s.dst],
        bias_bytes=ob * n * TYPE_BYTES[s.bias] if s.bias else 0.0,
        post_bytes=ob * m * n * s.binary * TYPE_BYTES[s.binary_dt],
        src_quant_bytes=b * m * k * TYPE_BYTES["u8"] if quantized else 0.0)


def from_problem(p: Any) -> MatmulShape:
    """The shape of a native BRGEMM tuner problem (duck-typed: dtype, m, n, k, threads, src,
    dst, has_bias, epilogue, dynamic, group, scale_bytes, name).  An s8 weight carries an
    f32 scale per column (per channel), or one of scale_bytes per group of K rows for a
    per-group problem that gives its group; an s4 weight one scale of scale_bytes per group
    of K rows; a source quantized in the call (dynamic) an f32 scale per row; the bias is
    f32, as the tuner's benchdnn lines give it.  A postops epilogue names no post-op, so it
    reads no tensor here."""
    wei = p.dtype
    group, scale = 0, 0
    if wei == "s8":
        g = getattr(p, "group", 0) or 0
        group, scale = (g if 0 < g < p.k else 0), getattr(p, "scale_bytes", 0) or 4
    elif wei == "s4":
        group, scale = (p.group if p.group and p.group < p.k else 0), p.scale_bytes or 2
    return MatmulShape(m=p.m, n=p.n, k=p.k, src=p.src, wei=wei, dst=p.dst, bias="f32" if p.has_bias else "",
                       wei_group=group, wei_scale_bytes=scale, src_scale_bytes=4 if p.dynamic else 0,
                       gated=p.epilogue == "gated", threads=p.threads, name=getattr(p, "name", ""))
