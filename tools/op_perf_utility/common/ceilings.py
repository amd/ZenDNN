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
"""The machine's ceilings as probe/probe.cpp measured them (format op-perf-ceilings,
version 1), and the lookups the per-op bounds read.

    {"format": "op-perf-ceilings", "version": 1, "probe_version": "...", "created": "...",
     "cpu":   {vendor, family, model, stepping, isa, l1d_kib, l2_kib, l3_kib, logical_cpus,
               smt, physical_cores, l3_domain_threads},        # the library's native_hw()
     "uarch": {l1d_bytes, l2_bytes, l3_domain_bytes, l3_domain_cores, family_l3_cores,
               fma_ports, vec_width_bits},                     # native_plan_uarch()
     "run":   {threads, domain_threads, cpuset, team_cpus, team_cores, omp_proc_bind, omp_places,
               repeats, quick, hugepages},
     "clock": {ghz_1core, ghz_team, ...},                      # sustained clock, 1 and all threads
     "compute": {fp32 | bf16 | int8 | fp16 | int4 | f16_widen:
                     {flops_per_instr, instr_per_cycle, latency_cycles, gflops_core, gflops_team, ...}},
     "bandwidth": [{level: l1 | l2 | l3 | dram, threads, bytes_per_thread, gbps, ...}],
     "write_bandwidth": [...]}                                  # as bandwidth, for stores

int4 and f16_widen are the inner loops of the INT4 tile and of the F16 kernels (where the
CPU has no AVX512-FP16), the others one dot-product instruction.  team_cores and
write_bandwidth come with probe 1.1; without them a team takes one core a thread and
stores are priced as reads.

The file belongs to the run that measured it: it describes one machine at one
thread count and pinning, and is never committed.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

FORMAT = "op-perf-ceilings"
VERSION = 1
LEVELS = ("l1", "l2", "l3", "dram")
PATHS = ("fp32", "bf16", "int8", "fp16", "int4", "f16_widen")
# The cpu fields a tuning campaign's schema reports too (its "hw" block); a ceilings
# file applies to a campaign only when every one both report agrees.
MATCH_KEYS = ("vendor", "family", "model", "l1d_kib", "l2_kib", "l3_kib", "l3_domain_threads")


class CeilingsError(ValueError):
    pass


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _pos(v: Any) -> bool:
    return _num(v) and v > 0


def validate(doc: Any) -> List[str]:
    """The errors of a ceilings document; empty when it is usable."""
    if not isinstance(doc, dict):
        return ["the ceilings file is not a JSON object"]
    errs: List[str] = []
    if doc.get("format") != FORMAT:
        errs.append("format must be %r" % FORMAT)
    if doc.get("version") != VERSION:
        errs.append("version must be %d" % VERSION)
    cpu = doc.get("cpu")
    if not isinstance(cpu, dict) or not isinstance(cpu.get("isa"), list):
        errs.append("cpu must be an object with an isa list")
    uarch = doc.get("uarch")
    if not isinstance(uarch, dict) or not all(_pos(uarch.get(k)) for k in ("l1d_bytes", "l2_bytes")):
        errs.append("uarch must give positive l1d_bytes and l2_bytes")
    run = doc.get("run")
    if not isinstance(run, dict) or not isinstance(run.get("threads"), int) or run["threads"] < 1:
        errs.append("run.threads must be a positive integer")
    elif "team_cores" in run and not (isinstance(run["team_cores"], int) and run["team_cores"] >= 0):
        errs.append("run.team_cores must be an integer >= 0")
    clock = doc.get("clock")
    if not isinstance(clock, dict) or not (_pos(clock.get("ghz_1core")) and _pos(clock.get("ghz_team"))):
        errs.append("clock must give positive ghz_1core and ghz_team")
    comp = doc.get("compute")
    if not isinstance(comp, dict) or not comp:
        errs.append("compute must be a non-empty object")
    else:
        for path, c in comp.items():
            if path not in PATHS:
                errs.append("compute: unknown path %r (one of %s)" % (path, ", ".join(PATHS)))
            elif not isinstance(c, dict) or not all(
                    _pos(c.get(k)) for k in ("flops_per_instr", "instr_per_cycle", "gflops_core", "gflops_team")):
                errs.append("compute.%s needs positive flops_per_instr, instr_per_cycle, gflops_core and "
                            "gflops_team" % path)
            elif "latency_cycles" in c and not _pos(c["latency_cycles"]):
                errs.append("compute.%s.latency_cycles must be positive" % path)
    bw = doc.get("bandwidth")
    if not isinstance(bw, list) or not bw:
        errs.append("bandwidth must be a non-empty list")
    for name in ("bandwidth", "write_bandwidth"):
        entries = doc.get(name, [])
        if not isinstance(entries, list):
            errs.append("%s must be a list" % name)
            continue
        for i, b in enumerate(entries):
            if not isinstance(b, dict) or b.get("level") not in LEVELS:
                errs.append("%s[%d]: level must be one of %s" % (name, i, ", ".join(LEVELS)))
            elif not isinstance(b.get("threads"), int) or b["threads"] < 1 or not _pos(b.get("gbps")):
                errs.append("%s[%d]: needs integer threads >= 1 and positive gbps" % (name, i))
    return errs


class Ceilings:
    """One validated ceilings document and its lookups."""

    def __init__(self, doc: Dict[str, Any], source: str = "<dict>"):
        errs = validate(doc)
        if errs:
            raise CeilingsError("%s: %s" % (source, "; ".join(errs)))
        self.doc = doc
        self.source = source
        self.cpu: Dict[str, Any] = doc["cpu"]
        self.uarch: Dict[str, Any] = doc["uarch"]
        self.run: Dict[str, Any] = doc["run"]
        self.clock: Dict[str, Any] = doc["clock"]
        self.compute: Dict[str, Dict[str, Any]] = doc["compute"]
        self._bw: Dict[str, List[Tuple[int, float]]] = {}
        self._wbw: Dict[str, List[Tuple[int, float]]] = {}
        for name, table in (("bandwidth", self._bw), ("write_bandwidth", self._wbw)):
            for b in doc.get(name, []):
                table.setdefault(b["level"], []).append((int(b["threads"]), float(b["gbps"])))
            for pts in table.values():
                pts.sort()

    @classmethod
    def load(cls, path: str) -> "Ceilings":
        try:
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise CeilingsError("%s: %s" % (path, e))
        return cls(doc, path)

    # ------------------------------------------------------------ machine

    @property
    def threads(self) -> int:
        """The team the probe measured: the run's thread count."""
        return int(self.run["threads"])

    @property
    def domain_threads(self) -> int:
        """Threads of one L3 domain: the run's, else the plan's domain, else the CPU's."""
        for v in (self.run.get("domain_threads"), self.uarch.get("l3_domain_cores")):
            if isinstance(v, int) and v > 0:
                return v
        ldt, smt = self.cpu.get("l3_domain_threads") or 0, self.cpu.get("smt") or 1
        return max(1, int(ldt) // max(1, int(smt))) if ldt else max(1, self.threads)

    @property
    def team_cores(self) -> int:
        """The physical cores of the probe's team (run.team_cores); its threads when unknown."""
        v = self.run.get("team_cores")
        return int(v) if isinstance(v, int) and v > 0 else self.threads

    def cores(self, threads: int) -> int:
        """The cores a team of threads takes, as many threads to a core as the probe's team:
        two threads on the siblings of one core share its pipes, L1 and L2."""
        return max(1, -(-max(1, threads) * self.team_cores // self.threads))

    def clock_ghz(self, threads: int) -> float:
        """The sustained clock of a call on threads threads: one core's alone, else the team's."""
        return float(self.clock["ghz_1core"] if threads <= 1 else self.clock["ghz_team"])

    def team_l2_bytes(self, threads: int) -> float:
        """The L2 of the cores a team of threads holds."""
        return float(self.uarch["l2_bytes"]) * self.cores(threads)

    @property
    def domain_cores(self) -> int:
        """Physical cores of one L3 domain: the plan's (uarch.l3_domain_cores), else those the
        run's domain threads take."""
        v = self.uarch.get("l3_domain_cores")
        return v if isinstance(v, int) and v > 0 else self.cores(self.domain_threads)

    def domains(self, threads: int) -> int:
        """The L3 domains the cores of a team of threads span."""
        return -(-self.cores(threads) // self.domain_cores)

    def team_l3_bytes(self, threads: int) -> float:
        """The L3 of the domains a team of threads spans; 0 on a machine without L3."""
        return float(self.uarch.get("l3_domain_bytes") or 0) * self.domains(threads)

    # ------------------------------------------------------------ compute

    def has_compute(self, path: str) -> bool:
        return path in self.compute

    def compute_gflops(self, path: str, threads: int) -> float:
        """The sustained FLOP rate of threads threads on path's instruction, from the team's
        measurement (which runs at the all-core clock) scaled by the cores they take."""
        c = self.compute.get(path)
        if c is None:
            raise CeilingsError("%s: no %s compute ceiling" % (self.source, path))
        if threads == self.threads:
            return float(c["gflops_team"])
        return float(c["gflops_team"]) / self.team_cores * self.cores(threads)

    def instr_per_cycle(self, path: str) -> float:
        return float(self.compute[path]["instr_per_cycle"])

    def latency_cycles(self, path: str) -> Optional[float]:
        v = self.compute.get(path, {}).get("latency_cycles")
        return float(v) if _pos(v) else None

    # ------------------------------------------------------------ memory

    def has_level(self, level: str, write: bool = False) -> bool:
        return level in (self._wbw if write else self._bw)

    def measured_threads(self, level: str, write: bool = False) -> int:
        """The largest thread count the probe measured level's bandwidth at (0: none)."""
        return max((n for n, _ in (self._wbw if write else self._bw).get(level, [])), default=0)

    def bandwidth_gbps(self, level: str, threads: int, write: bool = False) -> float:
        """The read (or write) bandwidth of threads threads streaming from level: the
        measured value at that thread count, else linear between the measured counts; below
        the first, that point's per-thread share; above the last, its per-core share for the
        private L1 and L2, its per-domain share for L3, and the last value for DRAM, which
        saturates."""
        pts = (self._wbw if write else self._bw).get(level)
        if not pts:
            raise CeilingsError("%s: no %s %s bandwidth" % (self.source, level, "write" if write else "read"))
        t = max(1, threads)
        for n, g in pts:
            if n == t:
                return g
        if t < pts[0][0]:
            return pts[0][1] * t / pts[0][0]
        for (n0, g0), (n1, g1) in zip(pts, pts[1:]):
            if n0 < t < n1:
                return g0 + (g1 - g0) * (t - n0) / (n1 - n0)
        n, g = pts[-1]
        if level in ("l1", "l2"):
            return g * self.cores(t) / self.cores(n)
        if level == "l3":
            return g * self.domains(t) / self.domains(n)
        return g

    def bytes_per_cycle_per_thread(self, level: str, threads: int) -> float:
        t = max(1, threads)
        return self.bandwidth_gbps(level, t) / t / self.clock_ghz(t)

    def cycles_per_kib_per_thread(self, level: str, threads: int) -> float:
        return 1024.0 / self.bytes_per_cycle_per_thread(level, threads)

    # ------------------------------------------------------------ matching and output

    def match(self, hw: Dict[str, Any]) -> List[str]:
        """The cpu fields that differ from a schema's hw block (every key both report, and
        the ISA features), as "key: probe X, schema Y"."""
        out = []
        for k in MATCH_KEYS:
            a, b = self.cpu.get(k), hw.get(k)
            if a in (None, "", -1, 0) or b in (None, "", -1, 0):
                continue
            if a != b:
                out.append("%s: probe %s, schema %s" % (k, a, b))
        want = set(hw.get("isa") or [])
        have = set(self.cpu.get("isa") or [])
        if want and have and want != have:
            out.append("isa: probe %s, schema %s" % ("+".join(sorted(have)), "+".join(sorted(want))))
        return out

    def brief(self) -> Dict[str, Any]:
        """What a committed file may say about the measurement: the probe and the team, no
        measured value."""
        return {"probe_version": self.doc.get("probe_version", ""), "created": self.doc.get("created", ""),
                "threads": self.threads}

    def summary(self, levels: Sequence[str] = LEVELS) -> str:
        cpu = self.cpu
        lines = ["%s: %s family %s model %s, %d threads (CPUs %s), L3 domain %d threads" % (
            self.source, cpu.get("vendor", "?"), cpu.get("family", "?"), cpu.get("model", "?"), self.threads,
            self.run.get("team_cpus") or self.run.get("cpuset", "?"), self.domain_threads)
            + (", on %d cores" % self.team_cores if self.team_cores != self.threads else "")
            + (", buffers advised huge pages" if self.run.get("hugepages") else ""),
            "clock: %.2f GHz one core, %.2f GHz all threads" % (self.clock["ghz_1core"], self.clock["ghz_team"])]
        for path in PATHS:
            c = self.compute.get(path)
            if c:
                lat = c.get("latency_cycles")
                lines.append("compute %-4s: %.2f instr/cycle/core, %.1f GFLOP/s per core, %.0f GFLOP/s team%s" % (
                    path, c["instr_per_cycle"], c["gflops_core"], c["gflops_team"],
                    ", latency %.1f cycles" % lat if _pos(lat) else ""))
        for name, table in (("bandwidth", self._bw), ("write", self._wbw)):
            for lvl in levels:
                for n, g in table.get(lvl, []):
                    lines.append("%-9s %-4s %3d threads: %8.1f GB/s (%.1f B/cycle/thread)" % (
                        name, lvl, n, g, g / n / self.clock_ghz(n)))
        return "\n".join(lines)
