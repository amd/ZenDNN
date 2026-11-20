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
"""The bound of one matmul call on a machine: the larger of its FLOPs over the compute
ceiling of its type and its compulsory bytes over the bandwidth of the memory level its
working set lives in; which of the two it is; and a measured time's efficiency.

Compute ceiling by weight type: bf16 weights the bf16 dot product (vdpbf16ps); s8 and u8
weights the INT8 dot product (vpdpbusd); s4 and u4 weights the INT4 tile's inner loop
(nibble unpacking, two VPDPBUSD a K octet) where the probe measured it, else the INT8 dot
product; f16 weights FP16 FMA where the CPU has AVX512-FP16, else the F16 kernels' loop
(widening to f32, then FP32 FMA) where measured, else FP32 FMA; f32 weights FP32 FMA.

Memory level: DRAM for cold weights (read once per call, as the tuner's replay backend
times them); else L2 when the working set fits in half the team's L2, L3 when it fits in
80 % of the L3 of the domains the team spans, DRAM otherwise.  A call's reads are priced
at that level's read bandwidth and its stores (the output, a quantized source's copy) at
its write bandwidth where the probe measured one: a store reads its line first.  A source
the call quantizes is written as u8 and read back; the quantizing arithmetic is not
counted.

Threads: the team's, at most one per 16 columns of one row of one batch item.  No plan
splits a call finer (the native tiles are at least 16 columns wide, the GEMV path's
units of M = 1 calls 32, and no plan splits K over threads), so a smaller call leaves the
other threads idle: its compute, bandwidth and caches are those of the threads it uses.

Instances: with n copies of the call running at once over the same memory, one copy's
DRAM bandwidth is the DRAM bandwidth of n x its threads, over n.  Above the largest
thread count the probe measured, DRAM keeps that point's bandwidth; probe the copies'
CPUs together for a measured share.  L2 and L3 belong to the cores and domains of each
copy and are not shared.

SMT: a team with two threads on the siblings of a core (run.team_cores) takes half as
many cores, and its compute, L1 and L2 are those of the cores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from op_perf_utility.common.ceilings import Ceilings
from op_perf_utility.matmul.work import INT_WEIGHTS, MatmulShape, MatmulWork, work

L2_FILL = 0.5
L3_FILL = 0.8
# The narrowest native tile: the finest unit of work a plan gives one thread.
UNIT_COLS = 16


def usable_threads(s: MatmulShape, threads: int) -> int:
    """The threads of a team of threads that can work on s: one per unit of work."""
    return max(1, min(threads, s.batch * s.m * -(-s.n // UNIT_COLS)))


def compute_path(s: MatmulShape, c: Ceilings) -> str:
    if s.wei in INT_WEIGHTS:
        return "int4" if s.wei in ("s4", "u4") and c.has_compute("int4") else "int8"
    if s.wei == "bf16":
        return "bf16"
    if s.wei == "f16":
        for path in ("fp16", "f16_widen"):
            if c.has_compute(path):
                return path
    return "fp32"


def memory_level(w: MatmulWork, c: Ceilings, threads: int, cold: bool = False) -> str:
    if not cold:
        ws = w.footprint_bytes
        if c.has_level("l2") and ws <= L2_FILL * c.team_l2_bytes(threads):
            return "l2"
        l3 = c.team_l3_bytes(threads)
        if l3 > 0 and c.has_level("l3") and ws <= L3_FILL * l3:
            return "l3"
    return "dram"


@dataclass(frozen=True)
class MatmulBound:
    shape: MatmulShape
    work: MatmulWork
    path: str
    level: str
    threads: int
    compute_gflops: float
    bandwidth_gbps: float
    team: int = 0
    instances: int = 1
    dram_measured: bool = True
    write_gbps: float = 0.0  # 0: stores priced at bandwidth_gbps

    @property
    def compute_ms(self) -> float:
        return self.work.flops / (self.compute_gflops * 1e6)

    @property
    def memory_ms(self) -> float:
        stored = self.work.write_bytes if self.write_gbps > 0 else 0.0
        ms = (self.work.total_bytes - stored) / (self.bandwidth_gbps * 1e6)
        return ms + (stored / (self.write_gbps * 1e6) if stored else 0.0)

    @property
    def attainable_ms(self) -> float:
        """The best time the call could take: compute and memory overlapped."""
        return max(self.compute_ms, self.memory_ms)

    @property
    def serial_ms(self) -> float:
        """Compute and memory one after the other, for context."""
        return self.compute_ms + self.memory_ms

    @property
    def bound(self) -> str:
        """compute, or the memory level whose bandwidth bounds the call."""
        return "compute" if self.compute_ms >= self.memory_ms else self.level

    def efficiency(self, measured_ms: Optional[float]) -> Optional[float]:
        """attainable / measured (1.0 at the bound); None without a measured time."""
        if measured_ms is None or not measured_ms > 0:
            return None
        return self.attainable_ms / measured_ms

    def row(self, measured_ms: Optional[float] = None) -> Dict[str, Any]:
        """One report row."""
        s, w = self.shape, self.work
        eff = self.efficiency(measured_ms)
        return {
            "shape": s.label(), "threads": self.threads, "team": self.team or self.threads,
            "instances": self.instances, "gflop": w.flops / 1e9, "mbytes": w.total_bytes / 1e6,
            "path": self.path, "level": self.level, "bound": self.bound,
            "compute_ms": self.compute_ms, "memory_ms": self.memory_ms, "attainable_ms": self.attainable_ms,
            "measured_ms": measured_ms,
            "gflops": w.flops / (measured_ms * 1e6) if eff is not None else None,
            "gbps": w.total_bytes / (measured_ms * 1e6) if eff is not None else None,
            "efficiency_pct": 100.0 * eff if eff is not None else None,
        }


def matmul_bound(s: MatmulShape, c: Ceilings, cold: bool = False, threads: Optional[int] = None,
                 instances: int = 1) -> MatmulBound:
    """The bound of s on the machine of c for a team of threads (default: the shape's), with
    instances copies of the call sharing the machine's DRAM."""
    if instances < 1:
        raise ValueError("instances must be positive, not %r" % (instances,))
    team = threads or s.threads
    t = usable_threads(s, team)
    w = work(s)
    path = compute_path(s, c)
    level = memory_level(w, c, t, cold)
    stores = c.has_level(level, write=True)
    if level == "dram":
        bw = c.bandwidth_gbps("dram", instances * t) / instances
        wbw = c.bandwidth_gbps("dram", instances * t, write=True) / instances if stores else 0.0
        measured = instances * t <= c.measured_threads("dram")
    else:
        bw, measured = c.bandwidth_gbps(level, t), True
        wbw = c.bandwidth_gbps(level, t, write=True) if stores else 0.0
    return MatmulBound(s, w, path, level, t, c.compute_gflops(path, t), bw, team=team, instances=instances,
                       dram_measured=measured, write_gbps=wbw)
