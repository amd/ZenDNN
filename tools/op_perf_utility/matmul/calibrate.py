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
                    takes rows x vectors / pipes cycles (0: the library's model of the
                    kernels' compute path, uarch.dot, where uarch.priced, else 2)
  cm_fma_lat        cycles a K step takes at least, whatever the kernel's accumulator sets
                    (built in 6; 0: the path's latency over the sets of the call's kernel)
  cm_l2_bytes       bytes a core reads from its L2 per cycle: the later calls of a K pass
                    re-read the slab from L2
  cm_l3_cycles      cycles per KiB a core reads from L3: source rows that do not fit L2 with
                    the slab, and the second reads of a weight tile in one domain
  cm_mem_cycles     cycles per KiB a thread streams from memory while its L3 domain streams:
                    the weights' memory time (0: the machine's uarch.mem_cycles, 2400 where
                    not measured, over sqrt(domain threads))
  cm_stream_cycles  cycles per KiB one core streams from memory alone: the domain's link
                    bounds the stream where cm_mem_cycles is above it (-1: the machine's
                    uarch.stream_cycles, 600 where not measured)

calibrate() maps the probe's measurements onto them at the team the ceilings describe, per
compute path (the inner loop of a weight type's kernels: bf16, int8, the s4 tile's int4,
fp16, the F16 kernels' f16_widen, fp32) for the first, only where the measurement is more
than MODEL_TOLERANCE away from what the library prices with (the model given, by default the
probe's uarch.dot, at 2 a cycle where uarch.priced is false; every path for a probe before
1.2):

  cm_fma_pipes      the path's instructions per cycle on one core, rounded
  cm_l2_bytes       L2 read bytes per cycle per thread, the whole team reading
  cm_l3_cycles      1024 / L3 read bytes per cycle per thread, one domain's threads reading
  cm_mem_cycles     1024 / DRAM read bytes per cycle per thread, the whole team reading
  cm_stream_cycles  1024 / DRAM read bytes per cycle of one thread alone

each an integer within the key's range.  cm_fma_lat is not derived: its built-in 6 is a floor
fitted to timed plans, not a latency, and the measured latency in its place made f16 calls of
16 rows slower; a path's measured latency goes into Calibration.uarch_override() instead
(<path>_lat), which corrects the model.  The probe streams with every hardware prefetcher
helping and no compute; the planner's calls stream under their kernels, so the values are
inputs to try, which tune.py calibrate keeps only when an A/B against the built-in inputs
says so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..common.ceilings import PATHS, Ceilings

KEYS = ("cm_fma_pipes", "cm_l2_bytes", "cm_l3_cycles", "cm_mem_cycles", "cm_stream_cycles")
PATH_KEYS = ("cm_fma_pipes",)
RANGES = {"cm_fma_pipes": (1, 16), "cm_l2_bytes": (1, 4096), "cm_l3_cycles": (1, 1 << 20),
          "cm_mem_cycles": (1, 1 << 20), "cm_stream_cycles": (1, 1 << 20)}
# The planner's value of an unset key (heuristic_params_t{}); None where it derives one from the
# machine model.
BUILTIN: Dict[str, Optional[int]] = {"cm_fma_pipes": None, "cm_l2_bytes": 16, "cm_l3_cycles": 10,
                                     "cm_mem_cycles": None, "cm_stream_cycles": None}
# The compute path of each weight type's kernels (the families' paths), which cm_fma_pipes
# describes; f16 takes fp16 where the probe measured it (the F16_PH kernels), else f16_widen.
DTYPE_PATHS = {"bf16": "bf16", "s8": "int8", "u8": "int8", "s4": "int4", "u4": "int4", "f16": "fp16", "f32": "fp32"}
# A path whose measured instructions per cycle are within this share of the library's model is
# calibrated already.
MODEL_TOLERANCE = 0.2
# The values ZENDNNL_MATMUL_NATIVE_UARCH takes (apply_native_uarch): <path>_ipc above 0 and at
# most 64, at the two decimals uarch_override() prints; <path>_lat; mem_cycles and stream_cycles.
UARCH_RANGES = {"ipc": (0.01, 64.0), "lat": (1, 64), "cycles": (1, 1 << 20)}


def _clamp(key: str, v: float) -> int:
    lo, hi = RANGES[key]
    return int(min(hi, max(lo, round(v))))


@dataclass
class Calibration:
    """cm_* values of one machine and team: common holds the memory keys, paths the pipes of each
    compute path the probe measured away from the model; sources says what each came from."""

    threads: int
    domain_threads: int
    # The threads of an L3 domain the library divides UarchParams::mem_cycles over the square
    # root of (native_plan_domain_threads); domain_threads where unknown.
    plan_domain: int = 0
    common: Dict[str, int] = field(default_factory=dict)
    paths: Dict[str, Dict[str, int]] = field(default_factory=dict)
    sources: Dict[str, str] = field(default_factory=dict)
    # Per path the probe measured: its instructions per cycle and latency, and the library's
    # model of it ({ipc, latency}) where the probe printed one.
    measured: Dict[str, Dict[str, float]] = field(default_factory=dict)
    model: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def values(self, path: str) -> Dict[str, int]:
        """Every key's value for the calls of one compute path."""
        return dict(self.common, **self.paths.get(path, {}))

    def path_of(self, wei: str) -> str:
        """The compute path of a weight type's calls: an f16 weight's FP16 FMA where the probe
        measured one, else the F16 kernels' loop (f16_widen, or FP32 FMA from a probe without
        it); an s4 weight's tile loop (int4, or the INT8 dot product without it)."""
        path = DTYPE_PATHS.get(wei, "fp32")
        if path == "fp16" and "fp16" not in self.measured:
            path = "f16_widen" if "f16_widen" in self.measured else "fp32"
        if path == "int4" and "int4" not in self.measured:
            path = "int8"
        return path

    def uarch_override(self) -> str:
        """ZENDNNL_MATMUL_NATIVE_UARCH of the measured model, which the library prices every call
        with: the measured paths ("<path>_ipc=v,<path>_lat=n") and, where the probe measured
        DRAM, the memory terms in the library's form (mem_cycles = cm_mem_cycles times the
        square root of plan_domain, stream_cycles = cm_stream_cycles), so that the model is
        complete; each value within the range the library takes."""

        def clamp(kind: str, v: float) -> float:
            lo, hi = UARCH_RANGES[kind]
            return min(hi, max(lo, v))

        items = []
        for path in PATHS:
            m = self.measured.get(path)
            if m:
                items.append("%s_ipc=%g" % (path, clamp("ipc", round(m["ipc"], 2))))
                if m.get("latency"):
                    items.append("%s_lat=%d" % (path, clamp("lat", int(round(m["latency"])))))
        if "cm_mem_cycles" in self.common and "cm_stream_cycles" in self.common:
            mem = self.common["cm_mem_cycles"] * math.sqrt(self.plan_domain or self.domain_threads)
            items.append("mem_cycles=%d" % clamp("cycles", int(round(mem))))
            items.append("stream_cycles=%d" % clamp("cycles", self.common["cm_stream_cycles"]))
        return ",".join(items)

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
                "paths": {p: dict(v) for p, v in self.paths.items()},
                "measured": {p: dict(v) for p, v in self.measured.items()},
                "model": {p: dict(v) for p, v in self.model.items()}}


def calibrate(c: Ceilings, threads: Optional[int] = None, domain_threads: Optional[int] = None,
              model: Optional[Dict[str, Tuple[float, int]]] = None) -> Calibration:
    """The cm_* values of the machine c describes for a team of threads (default: the probe's)
    in L3 domains of domain_threads (default: the probe's), against model, path -> (ipc,
    latency) the library prices with (default: the probe's, c.model; a tuning campaign passes
    its schema's)."""
    t = max(1, threads or c.threads)
    dt = max(1, min(t, domain_threads or c.domain_threads))
    plan_domain = domain_threads or int(c.uarch.get("l3_domain_cores") or 0) or c.domain_threads
    cal = Calibration(threads=t, domain_threads=dt, plan_domain=max(1, plan_domain))
    model = c.model if model is None else model
    for path in PATHS:
        if not c.has_compute(path):
            continue
        ipc = c.instr_per_cycle(path)
        lat = c.latency_cycles(path)
        cal.measured[path] = dict({"ipc": round(ipc, 4)}, **({"latency": round(lat, 3)} if lat is not None else {}))
        m = model.get(path)
        if m is not None:
            cal.model[path] = {"ipc": m[0], "latency": m[1]}
        if m is None or abs(m[0] - ipc) > MODEL_TOLERANCE * ipc:
            cal.paths[path] = {"cm_fma_pipes": _clamp("cm_fma_pipes", ipc)}
            cal.sources["cm_fma_pipes@%s" % path] = "%.2f instructions per cycle on one core%s" % (
                ipc, ", model %g" % m[0] if m else "")
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
