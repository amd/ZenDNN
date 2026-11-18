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
"""synthetic backend: a host-side stand-in for a pod session.

It times nothing.  A deterministic cost model of the resolved plan (thread
load, rows per kernel call, K passes, weight traffic, team overhead) with
log-normal noise produces the same probe and measure records as
jobs.collect(), in the same A B .. B A order, so the search, the results
store and the export run end to end on the host (tests, dry runs).  Its
numbers mean nothing about any machine.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence

from ..candidates import Candidate
from ..plan import Plan, ceil_div, int8_widths, plan_log_fields
from ..problems import Problem
from .common import Backend


def model_cost_ms(p: Problem, q: Plan, l2_bytes: int = 1 << 20) -> float:
    """Arbitrary but smooth cost of plan q for p, in ms."""
    flop_rate = 4.0e10 if p.bf16 else 8.0e10  # per thread
    bw = 6.0e9  # weight bytes per second per thread
    rows = min(q.mb, p.m)
    if p.bf16:
        call_rows = [min(q.mr, rows)]
    else:
        call_rows = [min(q.rows(w // 16), rows) for w in int8_widths(q)]
    eff = sum(r / (r + 2.5) for r in call_rows) / len(call_rows)
    passes = ceil_div(p.k, max(1, min(q.kc, p.k))) if p.bf16 else 1
    slab = q.nr * p.k * p.esz
    l2_fit = 1.0 if slab <= l2_bytes // 2 or passes > 1 or rows <= q.mr else 1.25
    tile_compute = 2.0 * rows * q.nr * p.k / (flop_rate * eff) * l2_fit
    reuse = q.m_blocks if q.traversal == "m_inner" else max(1.0, q.m_blocks / 2.0)
    tile_mem = slab / bw / max(1.0, reuse ** 0.5)
    tile_pass = passes * rows * q.nr * 8 / bw
    per_thread = ceil_div(q.tiles, q.threads)
    team = 1.5e-6 * math.log2(q.threads + 1) + (0.4e-6 * q.col_groups if q.col_groups else 0.0)
    return 1e3 * (per_thread * (tile_compute + tile_mem + tile_pass) + team)


class SyntheticBackend(Backend):
    name = "synthetic"
    runs_on_pod = False

    def __init__(self, threads: int = 32, noise: float = 0.02, seed: int = 0, l2_bytes: int = 1 << 20):
        super().__init__(threads)
        self.noise = noise
        self.seed = seed
        self.l2_bytes = l2_bytes

    def run_session(self, problems: Sequence[Problem], assignments: Dict[str, List[Candidate]],
                    defaults: Dict[str, Candidate], session_id: str, phase: str = "search",
                    repeats: int = 2) -> List[Dict[str, Any]]:
        """Probe and measure records, as jobs.collect() returns for a pod session."""
        rng = random.Random("%s/%s" % (self.seed, session_id))
        recs: List[Dict[str, Any]] = []
        base = {"session": session_id, "phase": phase, "backend": self.name}
        pos = 0
        for p in problems:
            k = p.key()
            slots: List[Optional[Candidate]] = [None] + list(assignments.get(k, []))
            for j, c in enumerate(slots):
                plan = (c or defaults[k]).plan
                recs.append(dict(base, kind="probe", pkey=k, problem=p.to_dict(), text=c.text if c else "",
                                 status="applied", reason="", slot=j, model_diff=[],
                                 logged=dict(plan_log_fields(plan, p), source="override" if c else "heuristic",
                                             cfg=c.text if c else "-"),
                                 hash="0", bits_same=True, warnings=[]))
            for rep in range(max(1, repeats)):
                order = list(range(1, len(slots)))
                if rep % 2:
                    order.reverse()
                for j in [0] + order + [0]:
                    c = slots[j]
                    t = model_cost_ms(p, (c or defaults[k]).plan, self.l2_bytes)
                    t *= math.exp(rng.gauss(0.0, self.noise))
                    recs.append(dict(base, kind="measure", pkey=k, problem=p.to_dict(), text=c.text if c else "",
                                     time_ms=t, stats={}, repeat=rep, pos=pos, slot=j, status="ok"))
                    pos += 1
        return recs

    def evaluator(self, problems: Dict[str, Problem], defaults: Dict[str, Candidate], repeats: int = 2):
        """evaluate(round, phase, batch) for search.Searcher."""

        def evaluate(round_id: int, phase: str, batch: Dict[str, List[Candidate]]) -> List[Dict[str, Any]]:
            keys = [k for k in batch if batch[k]]
            return self.run_session([problems[k] for k in keys], batch, defaults,
                                    "%s%03d_0" % (phase[0], round_id), phase, repeats)

        return evaluate

    def describe(self) -> Dict[str, Any]:
        return {"backend": self.name, "threads": self.threads, "noise": self.noise, "seed": self.seed,
                "l2_bytes": self.l2_bytes}
