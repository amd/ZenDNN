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
"""The native BRGEMM planner's cost-model inputs (the cm_* keys) from measured ceilings.

The cost model (native_cost_model.cpp, mirrored by tools/native_brgemm_tuning/cost_model.py)
prices a plan in cycles of its busiest worker from hardware inputs it otherwise assumes:

  cm_fma_pipes      dot-product instructions a core issues per cycle: a K step of a call
                    takes rows x vectors / pipes cycles (0: the CPU's fma_ports)
  cm_fma_lat        cycles of one dot-product instruction: a K step takes at least this,
                    each accumulator being a chain
  cm_l2_bytes       bytes a core reads from its L2 per cycle: the later calls of a K pass
                    re-read the slab from L2
  cm_l3_cycles      cycles per KiB a core reads from L3: source rows that do not fit L2 with
                    the slab, and the second reads of a weight tile in one domain
  cm_mem_cycles     cycles per KiB a thread streams from memory while its L3 domain streams:
                    the weights' memory time (0: 2400 / sqrt(domain threads))
  cm_stream_cycles  cycles per KiB one core streams from memory alone: the domain's link
                    bounds the stream where cm_mem_cycles is above it

calibrate() maps the probe's measurements onto them at the team the ceilings describe, per
compute path (the dot product of a weight type: bf16, int8, fp16 or fp32) for the first two:

  cm_fma_pipes      the path's instructions per cycle on one core, rounded
  cm_fma_lat        the path's latency in cycles, rounded
  cm_l2_bytes       L2 read bytes per cycle per thread, the whole team reading
  cm_l3_cycles      1024 / L3 read bytes per cycle per thread, one domain's threads reading
  cm_mem_cycles     1024 / DRAM read bytes per cycle per thread, the whole team reading
  cm_stream_cycles  1024 / DRAM read bytes per cycle of one thread alone

each an integer within the key's range.  The probe streams with every hardware prefetcher
helping and no compute; the planner's calls stream under their kernels, so the values are
inputs to try, which tune.py calibrate keeps only when an A/B against the built-in inputs
says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..common.ceilings import PATHS, Ceilings

KEYS = ("cm_fma_pipes", "cm_fma_lat", "cm_l2_bytes", "cm_l3_cycles", "cm_mem_cycles", "cm_stream_cycles")
PATH_KEYS = ("cm_fma_pipes", "cm_fma_lat")
RANGES = {"cm_fma_pipes": (1, 16), "cm_fma_lat": (1, 64), "cm_l2_bytes": (1, 4096),
          "cm_l3_cycles": (1, 1 << 20), "cm_mem_cycles": (1, 1 << 20), "cm_stream_cycles": (1, 1 << 20)}
# The planner's value of an unset key (heuristic_params_t{}); None where it derives one.
BUILTIN: Dict[str, Optional[int]] = {"cm_fma_pipes": None, "cm_fma_lat": 6, "cm_l2_bytes": 16, "cm_l3_cycles": 10,
                                     "cm_mem_cycles": None, "cm_stream_cycles": 600}
# The dot-product instruction of each weight type's calls, which cm_fma_pipes and cm_fma_lat
# describe; the bound's int4 and f16_widen paths time the tiles' other instructions too.
DTYPE_PATHS = {"bf16": "bf16", "s8": "int8", "u8": "int8", "s4": "int8", "u4": "int8", "f16": "fp16", "f32": "fp32"}


def _clamp(key: str, v: float) -> int:
    lo, hi = RANGES[key]
    return int(min(hi, max(lo, round(v))))


@dataclass
class Calibration:
    """cm_* values of one machine and team: common holds the memory keys, paths the pipes and
    latency of each compute path the probe measured; sources says what each came from."""

    threads: int
    domain_threads: int
    common: Dict[str, int] = field(default_factory=dict)
    paths: Dict[str, Dict[str, int]] = field(default_factory=dict)
    sources: Dict[str, str] = field(default_factory=dict)

    def values(self, path: str) -> Dict[str, int]:
        """Every key's value for the calls of one compute path."""
        return dict(self.common, **self.paths.get(path, {}))

    def path_of(self, wei: str) -> str:
        """The compute path of a weight type's calls: an f16 weight's FP16 FMA where the probe
        measured one, else FP32 FMA (the F16 kernels widen the source to f32)."""
        path = DTYPE_PATHS.get(wei, "fp32")
        return "fp32" if path == "fp16" and "fp16" not in self.paths else path

    def for_dtype(self, wei: str) -> Dict[str, int]:
        return self.values(self.path_of(wei))

    def settings(self, path: str) -> str:
        """"key=value,..." of values(path), in KEYS order."""
        v = self.values(path)
        return ",".join("%s=%d" % (k, v[k]) for k in KEYS if k in v)

    def override(self) -> str:
        """A ZENDNNL_MATMUL_NATIVE_CONFIG of one clause per weight type the native kernels take."""
        return ";".join("dtype=%s:%s" % (w, self.settings(self.path_of(w))) for w in ("bf16", "s8", "s4", "f16"))

    def rows(self) -> List[Dict[str, Any]]:
        out = []
        for k in KEYS:
            if k in self.common:
                out.append({"key": k, "path": "all", "value": self.common[k], "builtin": BUILTIN[k],
                            "source": self.sources.get(k, "")})
            for path in PATHS:
                if k in self.paths.get(path, {}):
                    out.append({"key": k, "path": path, "value": self.paths[path][k], "builtin": BUILTIN[k],
                                "source": self.sources.get("%s@%s" % (k, path), "")})
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {"threads": self.threads, "domain_threads": self.domain_threads, "common": dict(self.common),
                "paths": {p: dict(v) for p, v in self.paths.items()}}


def calibrate(c: Ceilings, threads: Optional[int] = None, domain_threads: Optional[int] = None) -> Calibration:
    """The cm_* values of the machine c describes for a team of threads (default: the probe's)
    in L3 domains of domain_threads (default: the probe's)."""
    t = max(1, threads or c.threads)
    dt = max(1, min(t, domain_threads or c.domain_threads))
    cal = Calibration(threads=t, domain_threads=dt)
    for path in PATHS:
        if not c.has_compute(path):
            continue
        ipc = c.instr_per_cycle(path)
        vals = {"cm_fma_pipes": _clamp("cm_fma_pipes", ipc)}
        cal.sources["cm_fma_pipes@%s" % path] = "%.2f instructions per cycle on one core" % ipc
        lat = c.latency_cycles(path)
        if lat is not None:
            vals["cm_fma_lat"] = _clamp("cm_fma_lat", lat)
            cal.sources["cm_fma_lat@%s" % path] = "latency %.2f cycles" % lat
        cal.paths[path] = vals
    if c.has_level("l2"):
        b = c.bytes_per_cycle_per_thread("l2", t)
        cal.common["cm_l2_bytes"] = _clamp("cm_l2_bytes", b)
        cal.sources["cm_l2_bytes"] = "L2 %.1f B/cycle per thread at %d threads" % (b, t)
    if c.has_level("l3"):
        b = c.bytes_per_cycle_per_thread("l3", dt)
        cal.common["cm_l3_cycles"] = _clamp("cm_l3_cycles", 1024.0 / b)
        cal.sources["cm_l3_cycles"] = "L3 %.1f B/cycle per thread at %d threads (one domain)" % (b, dt)
    if c.has_level("dram"):
        b = c.bytes_per_cycle_per_thread("dram", t)
        cal.common["cm_mem_cycles"] = _clamp("cm_mem_cycles", 1024.0 / b)
        cal.sources["cm_mem_cycles"] = "DRAM %.2f B/cycle per thread at %d threads" % (b, t)
        b1 = c.bytes_per_cycle_per_thread("dram", 1)
        cal.common["cm_stream_cycles"] = _clamp("cm_stream_cycles", 1024.0 / b1)
        cal.sources["cm_stream_cycles"] = "DRAM %.2f B/cycle of one thread alone" % b1
    return cal
