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
"""What every pod backend shares: the API-log line (DESIGN section 10), the
applied / dropped decision, and the interface jobs.Session drives."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..candidates import MAX_CLAUSES, problem_filters
from ..plan import LOG_PLAN_FIELDS, sets_gemv_keys
from ..problems import LAYOUT_PACK_NR, Problem

LOG_TAG = "[MATMUL.NATIVE_BRGEMM]"
# The fields of an M = 1 call's line that give its schedule on the GEMV kernels: the kernels, their
# threads and the most blocks one of their calls reads.
GEMV_SCHEDULE_FIELDS = ("gemv", "gemv_threads", "gemv_blocks")
WARN_RE = re.compile(
    r"native BRGEMM: (?P<label>.*?) (?:gives an illegal plan|conflicts with a higher layer) for M=(?P<M>\d+) "
    r"N=(?P<N>\d+) K=(?P<K>\d+) threads=(?P<threads>\d+) (?P<dtype>\S+): (?P<reason>.*?); dropped")
WARN_HINTS = ("native brgemm", "native_config", "native_tuning")


def parse_apilog_line(line: str) -> Optional[Dict[str, str]]:
    """key -> value of one "[MATMUL.NATIVE_BRGEMM] M=.. N=.. ..." line, or None."""
    i = line.find(LOG_TAG)
    if i < 0:
        return None
    out: Dict[str, str] = {}
    for tok in line[i + len(LOG_TAG):].split():
        k, eq, v = tok.partition("=")
        if eq:
            out[k] = v
    return out if {"M", "N", "K"} <= set(out) else None


def parse_apilog(text: str) -> Dict[Tuple[int, int, int], List[Dict[str, str]]]:
    """(M, N, K) -> the log lines of those calls, in order."""
    out: Dict[Tuple[int, int, int], List[Dict[str, str]]] = defaultdict(list)
    for line in text.splitlines():
        f = parse_apilog_line(line)
        if f is not None:
            try:
                out[(int(f["M"]), int(f["N"]), int(f["K"]))].append(f)
            except ValueError:
                continue
    return out


def parse_warnings(text: str) -> Tuple[Dict[Tuple[int, int, int], str], List[str]]:
    """Illegal-plan reasons by (M, N, K), and every line that mentions the planner."""
    reasons: Dict[Tuple[int, int, int], str] = {}
    lines: List[str] = []
    for line in text.splitlines():
        if LOG_TAG in line:
            continue
        m = WARN_RE.search(line)
        if m:
            reasons.setdefault((int(m["M"]), int(m["N"]), int(m["K"])), m["reason"])
        if m or any(h in line.lower() for h in WARN_HINTS):
            lines.append(line.strip()[:400])
    return reasons, lines


def moves_gemv(expected: str, want: Optional[Dict[str, str]] = None,
               builtin: Optional[Dict[str, str]] = None) -> bool:
    """Whether the candidate of the settings expected gives an M = 1 call on the GEMV kernels another
    schedule than the built-in plan's: want and builtin, the expected log fields of its plan and of
    the built-in plan (plan_log_fields), differ in GEMV_SCHEDULE_FIELDS (a key of the tiles moves it
    through the plan's threads, as threads_cap does); without both, it sets a key of the GEMV
    kernels (GEMV_KEYS)."""
    if want and builtin:
        return any(want.get(f) != builtin.get(f) for f in GEMV_SCHEDULE_FIELDS)
    return sets_gemv_keys(expected)


def applied_status(fields: Optional[Dict[str, str]], expected: str, want: Optional[Dict[str, str]] = None,
                   builtin: Optional[Dict[str, str]] = None) -> Tuple[str, str]:
    """(status, reason) of a call whose candidate settings were expected ("" = built-in), want and
    builtin the expected log fields of its plan and of the built-in plan, when known.

    applied: the override was the source and cfg equals the candidate; dropped:
    the override was not used (an illegal plan, or no clause matched); mismatch:
    another config was applied; gemv: the call ran on the GEMV kernels (an M = 1
    call does unless gemv is off), which take none of the plan's tiles, M blocks
    or K passes, and the candidate leaves their schedule as built in (moves_gemv);
    no_log: the call printed no line with source=.
    """
    if fields is None:
        return "no_log", "no %s line for this problem" % LOG_TAG
    if fields.get("gemv", "none") != "none" and expected and not moves_gemv(expected, want, builtin):
        return "gemv", "the call ran on the %s GEMV kernels" % fields["gemv"]
    src = fields.get("source")
    if src is None:
        return "no_log", "the log line has no source= field"
    layers = src.split("+")
    cfg = fields.get("cfg", "-")
    if not expected:
        if "override" in layers:
            return "mismatch", "an override applied to a built-in run (cfg=%s)" % cfg
        return "applied", ""
    if "override" not in layers:
        return "dropped", "source=%s" % src
    if cfg != expected:
        return "mismatch", "cfg=%s" % cfg
    return "applied", ""


def filter_parts(problems: Sequence[Problem]) -> List[List[Problem]]:
    """problems in order, cut into parts of at most MAX_CLAUSES whose override filters
    (problem_filters) differ: of two clauses that select one call, as for two Ns of one
    padded width, the library applies the first to both."""
    out: List[List[Problem]] = []
    cur: List[Problem] = []
    seen = set()
    for p in problems:
        f = problem_filters(p)
        if f in seen or len(cur) == MAX_CLAUSES:
            out.append(cur)
            cur, seen = [], set()
        cur.append(p)
        seen.add(f)
    if cur:
        out.append(cur)
    return out


def consistent(lines: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    """The first line when every line of one problem agrees on the plan, else None."""
    if not lines:
        return None
    first = {k: v for k, v in lines[0].items()}
    for other in lines[1:]:
        for k in LOG_PLAN_FIELDS + ("source", "cfg", "pairing"):
            if other.get(k) != first.get(k):
                return None
    return first


class Backend:
    """A way to time problems on a pod.

    groups() splits problems into processes; write_inputs() and command() give
    each run its files and shell command; parse_timed() / parse_probe() read a
    run's output.  Problems of one process must differ in their override
    filters (dtype, M, the padded N, K, threads and an s4 group size).
    """

    name = "base"
    runs_on_pod = True

    def __init__(self, threads: int = 32):
        self.threads = threads

    def supports(self, p: Problem) -> Optional[str]:
        """Why this backend cannot time p, or None."""
        if p.threads != self.threads:
            return "threads %d != backend threads %d" % (p.threads, self.threads)
        # Probes compare plan fields, not pack_nr: another width would be timed on the
        # library's calls and exported for calls that never occur.
        if self.runs_on_pod and p.pack_nr != LAYOUT_PACK_NR:
            return "pack_nr %d: the library's calls pack every weight in blocks of %d columns" % (
                p.pack_nr, LAYOUT_PACK_NR)
        return None

    def cost(self, p: Problem, layers: int = 0) -> float:
        """Relative pod time one run spends on p (balances the sessions of a round), over
        layers weight sets for a backend that cycles through them (0: its own count)."""
        return 1.0

    def groups(self, problems: Sequence[Problem]) -> List[List[Problem]]:
        """Problems per process: per dtype, filter_parts() of its problems by N, K and M."""
        out: List[List[Problem]] = []
        by_dtype: Dict[str, List[Problem]] = defaultdict(list)
        for p in problems:
            by_dtype[p.dtype].append(p)
        for dtype in sorted(by_dtype):
            out += filter_parts(sorted(by_dtype[dtype], key=lambda q: (q.n, q.k, q.m)))
        return out

    def preamble(self) -> List[str]:
        return []

    def probe_env(self) -> List[str]:
        """Shell lines that give an OpenMP program of GCC's runtime (the op performance
        utility's probe) the team and pinning of this backend's timed runs."""
        return [
            "for v in $(compgen -e | grep -E '^(ZENDNNL_|KMP_|GOMP_)'); do unset \"$v\"; done",
            "unset OMP_WAIT_POLICY",
            "export OMP_NUM_THREADS=%d OMP_PROC_BIND=close OMP_PLACES=cores" % self.threads,
        ]

    def write_inputs(self, sdir: str, gid: int, problems: Sequence[Problem]) -> None:
        pass

    def command(self, gid: int, problems: Sequence[Problem], probe: bool) -> str:
        raise NotImplementedError

    def parse_timed(self, text: str, problems: Sequence[Problem]) -> Dict[str, Dict[str, Any]]:
        """pkey -> {"time_ms": per-call time, "stats": {...}} for the problems found."""
        raise NotImplementedError

    def parse_probe(self, text: str, problems: Sequence[Problem]) -> Dict[str, Dict[str, Any]]:
        """pkey -> {"logged": fields or None, "hash": str or None}."""
        log = parse_apilog(text)
        out = {}
        for p in problems:
            lines = log.get((p.m, p.n, p.k), [])
            out[p.key()] = {"logged": consistent(lines) if lines else None,
                            "inconsistent": bool(lines) and consistent(lines) is None, "hash": None}
        return out

    def describe(self) -> Dict[str, Any]:
        return {"backend": self.name, "threads": self.threads}
