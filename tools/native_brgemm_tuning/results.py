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
"""Results database: one JSON record per line, appended only.

Records (all carry kind, session, phase, backend, pkey, problem, text, ts):

- kind "probe": what one run with the API log on reported for a problem and a
  candidate text: status (applied, dropped, mismatch, no_log, gemv, failed,
  bits_changed), source, cfg, the logged plan fields, differences from the
  plan model, the output hash (replay) and native warnings.
- kind "measure": one timed result: time_ms (per call), stats, repeat, pos.

text is the candidate's settings ("" for the built-in plan).  A candidate is
judged per session against that session's built-in runs, so sessions on
different pods compare fairly: ratio = median(candidate) / median(built-in),
noise = z * sqrt(rel_mad(built-in)^2 + rel_mad(candidate)^2), and a gain counts
only when 1 - ratio exceeds max(min_gain, noise) in every session.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

APPLIED = "applied"
BAD_STATUSES = ("dropped", "mismatch", "no_log", "gemv", "failed", "bits_changed")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def median(xs: List[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if not n:
        return float("nan")
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


class Stat:
    """Median, spread and count of positive samples."""

    __slots__ = ("values", "n", "median", "min", "max", "rel_mad")

    def __init__(self, values: Iterable[float]):
        self.values = sorted(float(v) for v in values)
        self.n = len(self.values)
        self.median = median(self.values)
        self.min = self.values[0] if self.n else float("nan")
        self.max = self.values[-1] if self.n else float("nan")
        if self.n >= 2 and self.median > 0:
            self.rel_mad = 1.4826 * median([abs(v - self.median) for v in self.values]) / self.median
        else:
            self.rel_mad = 0.0


class CandStat:
    """One candidate of one problem across sessions."""

    def __init__(self, text: str):
        self.text = text
        self.ratios: List[float] = []
        self.noises: List[float] = []
        self.sessions: List[str] = []
        self.times: List[float] = []

    @property
    def ratio(self) -> float:
        """Geometric mean of the per-session ratios to the built-in plan."""
        if not self.ratios:
            return float("nan")
        return math.exp(sum(math.log(r) for r in self.ratios) / len(self.ratios))

    @property
    def noise(self) -> float:
        return max(self.noises) if self.noises else float("inf")

    def significant(self, min_gain: float) -> bool:
        """Faster than the built-in plan beyond the noise in every session."""
        return bool(self.ratios) and all(
            1.0 - r > max(min_gain, nz) for r, nz in zip(self.ratios, self.noises))


class ProblemSummary:
    def __init__(self, pkey: str, problem: Dict[str, Any]):
        self.pkey = pkey
        self.problem = problem
        self.default_ms = float("nan")
        self.default_noise = 0.0
        self.cands: Dict[str, CandStat] = {}
        self.excluded: Dict[str, str] = {}  # text -> status / reason
        # The machine's bound on the problem (op_perf.attach): its resource and time.
        self.bound = ""
        self.attainable_ms = float("nan")

    def best(self, min_gain: float = 0.0, significant_only: bool = False) -> Optional[CandStat]:
        pool = [c for c in self.cands.values() if c.ratios and (not significant_only or c.significant(min_gain))]
        return min(pool, key=lambda c: c.ratio) if pool else None

    def efficiency(self, ms: float) -> float:
        """The attainable time over ms (nan without a bound or a time)."""
        if self.attainable_ms > 0 and ms > 0:
            return self.attainable_ms / ms
        return float("nan")

    @property
    def builtin_efficiency(self) -> float:
        return self.efficiency(self.default_ms)

    def best_efficiency(self, min_gain: float = 0.0) -> float:
        """The best significant candidate's efficiency, else the built-in plan's."""
        b = self.best(min_gain, significant_only=True)
        return self.efficiency(self.default_ms * b.ratio) if b is not None else self.builtin_efficiency


class ResultsDB:
    """An append-only JSONL file."""

    def __init__(self, path: str):
        self.path = path
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)

    def append(self, records: Iterable[Dict[str, Any]]) -> int:
        n = 0
        with open(self.path, "a", encoding="utf-8") as f:
            for r in records:
                r.setdefault("ts", utc_now())
                f.write(json.dumps(r, sort_keys=True) + "\n")
                n += 1
        return n

    def load(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        out = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # a line cut by an interrupted writer
        return out


def probe_verdict(r: Dict[str, Any]) -> Tuple[str, str]:
    """(status, reason) of one probe record.  A candidate the library applied with another
    plan than the plan model's (model_diff) is a mismatch: an entry would name the model's
    plan, which is not the plan that was timed."""
    st, why = r.get("status", "no_log"), r.get("reason", "")
    if st == APPLIED and r.get("text") and r.get("model_diff"):
        return "mismatch", "the library ran another plan than the plan model's: " + "; ".join(r["model_diff"])
    return st, why


def probe_status(records: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str], Tuple[str, str]]:
    """(pkey, text) -> (status, reason): the first rejection among the text's probe verdicts,
    else applied.  A probe whose run left no output (failed) says nothing about the candidate:
    it stands only until another probe of the text answers, in either order."""
    out: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for r in records:
        if r.get("kind") != "probe":
            continue
        key = (r["pkey"], r.get("text", ""))
        st, why = probe_verdict(r)
        prev = out.get(key)
        if prev is None or prev[0] == "failed" or (prev[0] == APPLIED and st != "failed"):
            out[key] = (st, why)
    return out


def summarize(records: List[Dict[str, Any]], phases: Optional[Iterable[str]] = None,
              z: float = 2.0) -> Dict[str, ProblemSummary]:
    """Per problem: built-in time and noise, and every applied candidate's ratios."""
    want = set(phases) if phases else None
    status = probe_status(records)
    by: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    problems: Dict[str, Dict[str, Any]] = {}
    for r in records:
        if r.get("kind") != "measure" or (want and r.get("phase") not in want):
            continue
        t = r.get("time_ms")
        if not isinstance(t, (int, float)) or not t > 0:
            continue
        by[r["pkey"]][r.get("session", "")][r.get("text", "")].append(float(t))
        problems.setdefault(r["pkey"], r.get("problem", {}))
    out: Dict[str, ProblemSummary] = {}
    for pkey, sessions in by.items():
        ps = ProblemSummary(pkey, problems[pkey])
        defaults = []
        for sid, texts in sessions.items():
            if "" not in texts:
                continue
            d = Stat(texts[""])
            defaults.append(d)
            for text, vals in texts.items():
                if not text:
                    continue
                st, why = status.get((pkey, text), (APPLIED, ""))
                if st != APPLIED:
                    ps.excluded[text] = "%s%s" % (st, ": " + why if why else "")
                    continue
                c = Stat(vals)
                cs = ps.cands.setdefault(text, CandStat(text))
                cs.ratios.append(c.median / d.median)
                cs.noises.append(z * math.sqrt(d.rel_mad ** 2 + c.rel_mad ** 2))
                cs.sessions.append(sid)
                cs.times.extend(vals)
        if defaults:
            ps.default_ms = median([d.median for d in defaults])
            ps.default_noise = max(d.rel_mad for d in defaults)
        out[pkey] = ps
    return out


def measured_texts(records: Iterable[Dict[str, Any]]) -> Dict[str, set]:
    """pkey -> candidate texts with a result: a time, or a probe the library answered
    (a candidate whose runs left no output can be proposed again)."""
    out: Dict[str, set] = defaultdict(set)
    for r in records:
        kind = r.get("kind")
        if (kind == "measure" and isinstance(r.get("time_ms"), (int, float))) or (
                kind == "probe" and r.get("status") != "failed"):
            out[r["pkey"]].add(r.get("text", ""))
    return out
