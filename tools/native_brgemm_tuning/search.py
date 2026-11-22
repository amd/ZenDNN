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
"""Search: grid within a budget, random, and model-guided.

A round proposes up to `batch` new candidates per problem (never one whose
plan was already measured or equals the built-in plan), hands them to an
evaluator (evaluate(round, phase, batch) -> records: one pod session per
part, or the synthetic backend) and appends the records to the results store.
Rounds go on until every problem has `budget` candidates measured or its
space is exhausted; a new Searcher on the same store resumes.

- grid: Space.iter_grid in order (every one-key change, then pairs, ...).
- random: each key changed with probability p_change.
- guided: `init` random candidates per problem first; then each round fits a
  surrogate of log(time / built-in time) on every problem measured so far
  (problem features plus resolved-plan features) and proposes, per problem,
  the lowest lower confidence bounds (mean - kappa * spread) among a random
  pool and the neighbours of its best candidates, plus an `explore` fraction
  drawn at random from that pool.
- local: per problem, the neighbours (Space.neighbors, in key order) of its
  `seeds` best candidates that beat the built-in plan beyond the noise; a
  problem without such a candidate gets none.

With the problems' attainable times on the measured machine (tune.py probe),
plan_round() can skip a problem whose built-in plan is near its bound, stop
one whose best candidate is, and split the budget by headroom (see there).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from .candidates import Candidate, Deduper, Space
from .plan import plan_features
from .problems import Problem
from .results import ResultsDB, measured_texts, probe_status, summarize

Evaluator = Callable[[int, str, Dict[str, List[Candidate]]], List[Dict[str, Any]]]


@dataclass
class SearchSettings:
    strategy: str = "guided"  # grid | random | guided | local
    budget: int = 32  # candidates measured per problem, besides the built-in plan
    batch: int = 8  # candidates per problem per round
    max_keys: Optional[int] = None  # grid: most keys changed at once
    p_change: Optional[float] = None  # random / guided pool: chance to change each key
    init: int = 8  # guided: random candidates per problem in its first round
    pool: int = 400  # guided: random candidates scored per problem per round
    explore: float = 0.25  # guided: share of a batch drawn at random from the pool
    kappa: float = 1.0  # guided: weight of the model's spread
    surrogate: str = "auto"
    seed: int = 0
    z: float = 2.0
    seeds: int = 2  # local: best significant candidates whose neighbours are proposed
    min_gain: float = 0.02
    # Headroom on the measured machine (tune.py probe); off unless one of the three is set.
    attainable: Optional[Dict[str, float]] = None  # pkey -> the problem's attainable time (ms)
    weights: Optional[Dict[str, float]] = None  # pkey -> calls of the problem's weight (default 1)
    prioritize: bool = False  # budget x problems split by headroom
    # calls: calls x best time x (1 - efficiency); relative: (1 - efficiency) alone, for flat calls
    prioritize_by: str = "calls"
    skip_at: float = 0.0  # no candidates for a problem whose built-in plan reaches this efficiency
    stop_at: float = 0.0  # no more candidates once a problem's best candidate reaches it
    scout: int = 1  # candidates of a problem without a built-in time yet, with headroom on

    @property
    def headroom(self) -> bool:
        return bool(self.attainable) and (self.prioritize or self.skip_at > 0 or self.stop_at > 0)


def featurize(p: Problem, c: Candidate) -> List[float]:
    """Problem features and resolved-plan features of one candidate."""
    f = [math.log2(p.m), math.log2(p.n), math.log2(p.k), float(p.threads), 1.0 if p.bf16 else 0.0]
    for name, v in plan_features(c.plan, p).items():
        logged = name in ("mb", "m_blocks", "n_tiles", "tiles_per_thread", "k_passes", "copy_block", "row_unit")
        f.append(math.log2(v) if v > 0 and logged else v)
    return f


class Searcher:
    def __init__(self, problems: Sequence[Problem], spaces: Dict[str, Space], evaluate: Evaluator,
                 db: ResultsDB, settings: SearchSettings, log: Callable[[str], None] = print):
        if settings.strategy not in ("grid", "random", "guided", "local"):
            raise ValueError("unknown strategy %r" % settings.strategy)
        self.problems = {p.key(): p for p in problems}
        self.spaces = spaces
        self.evaluate = evaluate
        self.db = db
        self.s = settings
        self.log = log
        self.dedup = {k: Deduper(spaces[k]) for k in self.problems}
        self.count = {k: 0 for k in self.problems}
        self.exhausted: set = set()
        self.grid: Dict[str, Iterator[Candidate]] = {}
        self.round = 1
        # With headroom on, per round: (budget, batch) of the problems it changes, and why a
        # problem gets no candidates.
        self.limits: Dict[str, Tuple[int, int]] = {}
        self.held: Dict[str, str] = {}
        self._load()

    def _cand(self, k: str, text: str) -> Optional[Candidate]:
        try:
            return self.spaces[k].from_text(text)
        except ValueError:
            return None

    def _load(self) -> None:
        recs = [r for r in self.db.load() if r.get("phase") == "search"]
        rounds = [int(r.get("round", 0)) for r in recs]
        self.round = max(rounds) + 1 if rounds else 1
        for k, texts in measured_texts(recs).items():
            if k not in self.problems:
                continue
            for t in texts:
                if not t:
                    continue
                c = self._cand(k, t)
                if c is None or self.dedup[k].add(c):
                    self.count[k] += 1

    def want(self, k: str) -> int:
        if k in self.exhausted or k in self.held:
            return 0
        budget, batch = self.limits.get(k, (self.s.budget, self.s.batch))
        return max(0, min(batch, budget - self.count[k]))

    def plan_round(self) -> None:
        """With headroom on, each problem's place in the next round from its built-in and best
        times against its attainable time: a problem not timed yet gets `scout` candidates; one
        whose built-in plan reaches skip_at, or whose best plan (the built-in one when no
        candidate beats it) reaches stop_at, gets none;
        with prioritize, budget x problems is split by headroom (calls x best time x
        (1 - efficiency)), each problem between a quarter and four times the budget.  A
        round still gives a problem at most batch candidates: a session's slots, each a
        process of its group, follow the largest batch of the group, whatever the others.
        A problem whose built-in run is a mismatch gets none, headroom or not: its
        candidates are not judged (results.summarize)."""
        self.limits, self.held = {}, {}
        for (k, text), (st, why) in probe_status(self.db.load()).items():
            if not text and st == "mismatch" and k in self.problems:
                self.held[k] = "its built-in run: %s" % why
        if not self.s.headroom:
            return
        att = self.s.attainable or {}
        weights = self.s.weights or {}
        summ = summarize(self.db.load(), phases=["search"], z=self.s.z)
        head: Dict[str, float] = {}
        for k in self.problems:
            a = att.get(k)
            if not a or k in self.held:
                continue
            ps = summ.get(k)
            if ps is None or not ps.default_ms > 0:
                self.limits[k] = (self.s.budget, max(1, self.s.scout))
                continue
            best = ps.best(self.s.min_gain, significant_only=True)
            t_best = ps.default_ms * best.ratio if best is not None else ps.default_ms
            eff, eff_best = a / ps.default_ms, a / t_best
            if self.s.skip_at > 0 and eff >= self.s.skip_at:
                self.held[k] = "built-in plan at %.0f%% of its bound" % (100 * eff)
            elif self.s.stop_at > 0 and eff_best >= self.s.stop_at:
                self.held[k] = "best plan at %.0f%% of its bound" % (100 * eff_best)
            elif self.s.prioritize_by == "relative":
                head[k] = max(0.0, 1.0 - eff_best)
            else:
                head[k] = float(weights.get(k, 1.0)) * t_best * max(0.0, 1.0 - eff_best)
        total = sum(head.values())
        if self.s.prioritize and total > 0:
            pool = self.s.budget * len(head)
            lo, hi = max(1, self.s.budget // 4), 4 * self.s.budget
            for k, h in head.items():
                self.limits[k] = (min(hi, max(lo, int(round(pool * h / total)))), self.s.batch)

    def _take(self, k: str, it: Iterator[Candidate], n: int, exhaust: bool = True) -> List[Candidate]:
        out: List[Candidate] = []
        for c in it:
            if self.dedup[k].add(c):
                out.append(c)
                if len(out) == n:
                    return out
        if exhaust or not out:
            self.exhausted.add(k)
        return out

    def _random_iter(self, k: str, tries: int) -> Iterator[Candidate]:
        rng = random.Random("%d/%d/%s" % (self.s.seed, self.round, k))
        sp = self.spaces[k]
        if getattr(sp, "base_candidate", None) is not None:
            yield sp.base_candidate
        for _ in range(tries):
            c = sp.sample(rng, self.s.p_change)
            if c is not None:
                yield c

    def propose(self) -> Dict[str, List[Candidate]]:
        if self.s.strategy == "guided":
            return self._propose_guided()
        if self.s.strategy == "local":
            return self._propose_local()
        out = {}
        for k in self.problems:
            n = self.want(k)
            if n == 0:
                continue
            if self.s.strategy == "grid":
                it = self.grid.setdefault(k, self.spaces[k].iter_grid(self.s.max_keys))
                out[k] = self._take(k, it, n)
            else:
                out[k] = self._take(k, self._random_iter(k, 200 * n), n, exhaust=False)
        return out

    def _propose_local(self) -> Dict[str, List[Candidate]]:
        summary = summarize(self.db.load(), phases=["search"], z=self.s.z)
        out: Dict[str, List[Candidate]] = {}
        for k in self.problems:
            n = self.want(k)
            ps = summary.get(k)
            if n == 0 or ps is None:
                continue
            good = sorted((c for c in ps.cands.values() if c.ratios and c.significant(self.s.min_gain)),
                          key=lambda c: c.ratio)
            seeds = [c for c in (self._cand(k, cs.text) for cs in good[:max(1, self.s.seeds)]) if c is not None]
            if not seeds:
                continue
            sp = self.spaces[k]
            it = (c for b in seeds for c in sp.neighbors(b))
            out[k] = self._take(k, it, n, exhaust=False)
        return out

    def _training(self) -> Any:
        import numpy as np

        rows, ys = [], []
        for k, ps in summarize(self.db.load(), phases=["search"], z=self.s.z).items():
            if k not in self.problems:
                continue
            p = self.problems[k]
            rows.append(featurize(p, self.spaces[k].default))
            ys.append(0.0)
            for text, cs in ps.cands.items():
                c = self._cand(k, text)
                if c is not None and cs.ratios:
                    rows.append(featurize(p, c))
                    ys.append(math.log(cs.ratio))
        return np.array(rows, dtype=float), np.array(ys, dtype=float)

    def _propose_guided(self) -> Dict[str, List[Candidate]]:
        import numpy as np

        from .surrogate import make_surrogate

        X, y = self._training()
        model = None
        if len(y) >= 8 and float(np.ptp(y)) > 0:
            model = make_surrogate(self.s.surrogate, seed=self.s.seed + self.round).fit(X, y)
        summary = summarize(self.db.load(), phases=["search"], z=self.s.z) if model is not None else {}
        out: Dict[str, List[Candidate]] = {}
        for k, p in self.problems.items():
            n = self.want(k)
            if n == 0:
                continue
            if model is None or self.count[k] == 0:
                out[k] = self._take(k, self._random_iter(k, 200 * n), min(n, max(1, self.s.init)), exhaust=False)
                continue
            sp = self.spaces[k]
            pool: Dict[tuple, Candidate] = {}
            for c in self._random_iter(k, 3 * self.s.pool):
                if c not in self.dedup[k]:
                    pool.setdefault(c.sig, c)
                if len(pool) >= self.s.pool:
                    break
            ps = summary.get(k)
            if ps is not None:
                for cs in sorted(ps.cands.values(), key=lambda c: c.ratio)[:3]:
                    best = self._cand(k, cs.text)
                    if best is not None:
                        for c in sp.neighbors(best):
                            if c not in self.dedup[k]:
                                pool.setdefault(c.sig, c)
            cands = list(pool.values())
            if not cands:
                self.exhausted.add(k)
                continue
            mu, sd = model.predict(np.array([featurize(p, c) for c in cands], dtype=float))
            order = list(np.argsort(mu - self.s.kappa * sd, kind="stable"))
            n_exp = min(len(cands), int(round(n * self.s.explore)))
            pick = order[:max(0, min(n, len(cands)) - n_exp)]
            rest = order[len(pick):]
            rng = random.Random("%d/%d/%s/x" % (self.s.seed, self.round, k))
            pick += rng.sample(rest, min(n_exp, len(rest)))
            out[k] = [cands[i] for i in pick]
            for c in out[k]:
                self.dedup[k].add(c)
        return out

    def run(self, max_rounds: Optional[int] = None) -> int:
        """Rounds until the budget is spent; returns the rounds done."""
        done = 0
        logged: set = set()
        while max_rounds is None or done < max_rounds:
            self.plan_round()
            for k, why in sorted(self.held.items()):
                if k not in logged:
                    logged.add(k)
                    self.log("no more candidates for %s: %s" % (self.problems[k].label(), why))
            batch = {k: v for k, v in self.propose().items() if v}
            if not batch:
                break
            n = sum(len(v) for v in batch.values())
            self.log("round %d (%s): %d candidates over %d problems" % (self.round, self.s.strategy, n, len(batch)))
            recs = self.evaluate(self.round, "search", batch)
            for r in recs:
                r["round"] = self.round
            self.db.append(recs)
            for k, v in batch.items():
                self.count[k] += len(v)
            self.round += 1
            done += 1
        return done
