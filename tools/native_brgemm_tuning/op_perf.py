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
"""A campaign through the op performance utility (tools/op_perf_utility): the ceilings
the probe measured on the campaign's machine with its team (workdir/ceilings.json), and
per problem the bound they give, its attainable time and the efficiency (attainable /
measured) of the built-in plan and of the best candidate.

What a tuning file may carry (meta()) is the probe's version, the bound kinds and the
efficiencies in percent: no measured ceiling, so a committed file says nothing about the
machine beyond the hardware block it already has."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from op_perf_utility.common.ceilings import Ceilings, CeilingsError
from op_perf_utility.matmul.bounds import MatmulBound, matmul_bound
from op_perf_utility.matmul.work import from_problem

from .cost_model import UNPRICED_IPC
from .problems import Problem
from .results import ProblemSummary

CEILINGS = "ceilings.json"

__all__ = ["CEILINGS", "Ceilings", "CeilingsError", "attach", "best_plan", "bounds", "cold_weights",
           "efficiency_rows", "geomean_pct", "library_model", "load", "machine_mismatch", "meta", "model_check"]


def load(path: str) -> Ceilings:
    return Ceilings.load(path)


def _hw_dict(hw: Any) -> Dict[str, Any]:
    return {k: getattr(hw, k, None) for k in ("vendor", "family", "model", "stepping", "isa", "l1d_kib",
                                             "l2_kib", "l3_kib", "logical_cpus", "smt", "physical_cores",
                                             "l3_domain_threads")}


def machine_mismatch(ceil: Ceilings, schema: Any, threads: int) -> List[str]:
    """Why ceilings do not describe the campaign's machine and team: the CPU fields both
    the probe and the schema's hw block report, AVX512-FP16, the plan's L3 domain and the
    thread count."""
    out = ceil.match(_hw_dict(schema.hw)) if getattr(schema, "hw", None) is not None else []
    # AVX512-FP16 picks the f16 kernels and their ceiling (F16_PH and FP16 FMA, or FP32 FMA).
    fp16 = ((getattr(schema, "doc", None) or {}).get("uarch") or {}).get("avx512fp16")
    probed = ceil.cpu.get("avx512fp16")
    if isinstance(fp16, bool) and isinstance(probed, bool) and fp16 != probed:
        out.append("avx512fp16: probe %s, schema %s" % (probed, fp16))
    domain = schema.uarch.plan_domain if getattr(schema, "uarch", None) is not None else None
    probed = ceil.uarch.get("l3_domain_cores")
    if domain and isinstance(probed, int) and probed > 0 and probed != domain:
        out.append("L3 domain: probe %d cores, campaign %d" % (probed, domain))
    if ceil.threads != threads:
        out.append("threads: probe %d, campaign %d" % (ceil.threads, threads))
    return out


# A path whose measured dot products a cycle differ from the library's model by more than this
# share is flagged.
MODEL_TOLERANCE = 0.2


def library_model(schema: Any) -> Dict[str, Tuple[float, int]]:
    """path -> (ipc, latency) the library prices the schema's machine with: its uarch.dot, at 2 a
    cycle where the model does not price; empty without a uarch."""
    uarch = getattr(schema, "uarch", None)
    if uarch is None:
        return {}
    priced = getattr(uarch, "priced", True)
    return {path: (d.ipc if priced else UNPRICED_IPC, d.latency) for path, d in uarch.dot.items()}


def model_check(ceil: Ceilings, schema: Any) -> List[str]:
    """The paths whose price in the library (library_model) the probe measured more than
    MODEL_TOLERANCE away, as "int8: model 2, measured 3.50"; a flag, not a refusal: the probe
    measured the campaign's machine, the model is what the library assumes of it."""
    out = []
    for path, (ipc, _) in library_model(schema).items():
        if not ceil.has_compute(path):
            continue
        measured = ceil.instr_per_cycle(path)
        if measured > 0 and abs(ipc - measured) > MODEL_TOLERANCE * measured:
            out.append("%s: model %g, measured %.2f" % (path, ipc, measured))
    return out


def cold_weights(backend: Dict[str, Any]) -> bool:
    """Whether a backend's timed calls find their weights in DRAM: the replay driver cycles
    through cold_mb of distinct weights; benchdnn and the synthetic backend reuse one set."""
    return backend.get("backend") == "replay" and float(backend.get("cold_mb") or 0) > 0


def bounds(problems: Sequence[Problem], ceil: Ceilings, cold: bool) -> Dict[str, MatmulBound]:
    """pkey -> the problem's bound on the measured machine."""
    out: Dict[str, MatmulBound] = {}
    for p in problems:
        try:
            out[p.key()] = matmul_bound(from_problem(p), ceil, cold=cold)
        except (CeilingsError, ValueError):
            continue
    return out


def attach(summ: Dict[str, ProblemSummary], bnds: Dict[str, MatmulBound]) -> None:
    """Gives every summary with a bound its bounding resource and attainable time."""
    for k, ps in summ.items():
        b = bnds.get(k)
        if b is not None:
            ps.bound, ps.attainable_ms = b.bound, b.attainable_ms


def _pct(x: float) -> Optional[float]:
    return round(100.0 * x, 1) if x > 0 and math.isfinite(x) else None


def best_plan(ps: ProblemSummary, confirmed: Optional[ProblemSummary],
              min_gain: float) -> Tuple[str, float, str, Optional[float]]:
    """(text, ratio to the built-in plan, status, the built-in time of the summary the ratio is
    of) of a problem's best plan.  A candidate that beats the built-in plan in a confirmation
    is confirmed; the search's best is unconfirmed while no confirmation timed it, and
    rejected (the built-in plan stays the best) when one did and it did not hold there.
    ("", 1, "", the search's built-in time) without a significant candidate."""
    if confirmed is not None:
        held = confirmed.best(min_gain, significant_only=True)
        if held is not None:
            return held.text, held.ratio, "confirmed", confirmed.default_ms or ps.default_ms
    best = ps.best(min_gain, significant_only=True)
    if best is None:
        return "", 1.0, "", ps.default_ms
    timed = confirmed.cands.get(best.text) if confirmed is not None else None
    if timed is not None and timed.ratios:
        return best.text, 1.0, "rejected", ps.default_ms
    return best.text, best.ratio, "unconfirmed", ps.default_ms


def efficiency_rows(summ: Dict[str, ProblemSummary], problems: Sequence[Problem],
                    bnds: Dict[str, MatmulBound], min_gain: float,
                    confirmed: Optional[Dict[str, ProblemSummary]] = None) -> List[Dict[str, Any]]:
    """One row per problem with a bound: the bound, its attainable time, and the built-in
    plan's and the best plan's (best_plan(), from the confirmation summaries when given) time
    and efficiency (percent of the bound)."""
    rows = []
    for p in problems:
        b = bnds.get(p.key())
        if b is None:
            continue
        ps = summ.get(p.key())
        row: Dict[str, Any] = {"problem": p.key(), "bound": b.bound, "level": b.level, "path": b.path,
                               "attainable_ms": b.attainable_ms, "builtin_ms": None, "builtin_pct": None,
                               "best_ms": None, "best_pct": None, "best": "", "best_status": ""}
        if ps is not None and ps.default_ms > 0:
            row.update(builtin_ms=ps.default_ms, builtin_pct=_pct(b.efficiency(ps.default_ms)))
            text, ratio, status, builtin_ms = best_plan(ps, (confirmed or {}).get(p.key()), min_gain)
            ms = builtin_ms * ratio
            row.update(best_ms=ms, best_pct=_pct(b.efficiency(ms)), best=text, best_status=status)
        rows.append(row)
    return rows


def geomean_pct(rows: Sequence[Dict[str, Any]], col: str) -> Optional[float]:
    vals = [r[col] for r in rows if isinstance(r.get(col), (int, float)) and r[col] > 0]
    if not vals:
        return None
    return round(math.exp(sum(math.log(v) for v in vals) / len(vals)), 1)


def meta(rows: Sequence[Dict[str, Any]], ceil: Ceilings, cold: bool,
         calibration: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """meta.op_perf of a tuning file: the probe's version and team, and per problem its bound
    and the efficiencies in percent."""
    out: Dict[str, Any] = {
        "probe": ceil.brief(), "cold_weights": cold,
        "builtin_pct_geomean": geomean_pct(rows, "builtin_pct"), "best_pct_geomean": geomean_pct(rows, "best_pct"),
        "problems": [{"problem": r["problem"], "bound": r["bound"], "builtin_pct": r["builtin_pct"],
                      "best_pct": r["best_pct"], "best_status": r["best_status"]}
                     for r in rows if r["builtin_pct"] is not None]}
    if calibration is not None:
        out["calibration"] = calibration
    return out
