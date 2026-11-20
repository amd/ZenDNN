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
"""replay backend: the cold-weight replay driver (replay/replay.cpp) on a pod.

The driver makes the framework plugin's matmul_direct calls (a bf16 source
over bf16 weights, or over s8 or s4 weights whose source it quantizes per
token, s4 in groups along K; an f16 source over f16 weights) and visits
--layers weight sets per shape in layer order, so every weight is cold when it
comes back, as in a model.  A process whose shapes are small takes more
layers, so that it cycles through at least cold_mb of distinct weights
whichever problems share it.  The job preamble reproduces the model server's
CPU worker: the preload libraries (tcmalloc and the worker's libiomp5), a
KMP_AFFINITY proclist of the first `threads` CPUs of the pod's cpuset,
KMP_BLOCKTIME=5, KMP_TPAUSE=0 and 32 threads.  One process times its
problems as shapes x Ms; probe runs use one layer, few iterations and
--check 1, whose output hash shows whether a candidate kept the built-in
plan's bits.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..candidates import MAX_CLAUSES
from ..problems import Problem
from .common import Backend, filter_parts

RESULT_RE = re.compile(
    r"^RESULT shape=(?P<shape>\S+) N=(?P<N>\d+) K=(?P<K>\d+) M=(?P<M>\d+) calls=(?P<calls>\d+) "
    r"median_ms=(?P<median>[\d.eE+-]+) mean_ms=(?P<mean>[\d.eE+-]+) min_ms=(?P<min>[\d.eE+-]+) "
    r"p90_ms=(?P<p90>[\d.eE+-]+)")
CHECK_RE = re.compile(r"^CHECK shape=\S+ N=(?P<N>\d+) K=(?P<K>\d+) M=(?P<M>\d+) hash=(?P<hash>[0-9a-fA-F]+)")
CPU_LIST = ("LIST=$(awk '/Cpus_allowed_list/{print $2}' /proc/self/status | tr ',' '\\n' | "
            "awk -F- '{ if (NF == 2) for (i = $1; i <= $2; i++) printf \"%d,\", i; "
            "else printf \"%d,\", $1 }' | sed 's/,$//')")


def replay_group(p: Problem) -> int:
    """The driver's --group of an s4 problem: its group size, -1 for one group over K; 0 for
    other weights."""
    if not p.int4:
        return 0
    return -1 if p.group == p.k else p.group


class ReplayBackend(Backend):
    name = "replay"

    def __init__(self, binary: str = "replay", threads: int = 32, preload: str = "", layers: int = 4,
                 iters: int = 30, warmup: int = 2, budget_ms: int = 1500, scale_dt: str = "bf16",
                 group_cost: float = 0.0, group_gb: float = 0.0, cold_mb: float = 0.0):
        super().__init__(threads)
        self.binary = binary
        self.preload = preload
        self.layers = layers
        self.iters = iters
        self.warmup = warmup
        self.budget_ms = budget_ms
        self.scale_dt = scale_dt
        # 0: no cap.  A cap keeps one process short (the built-in runs that bracket a
        # candidate stay close in time) and bounds the weights it allocates.
        self.group_cost = group_cost
        self.group_gb = group_gb
        # 0: --layers as given.  Above the caches' size, a weight is evicted before it comes back.
        self.cold_mb = cold_mb
        if group_gb > 0 and cold_mb > group_gb * 1e3:
            raise ValueError("a timed process cycles through at least cold_mb=%g MB of weights, more than "
                             "group_gb=%g GB lets it allocate" % (cold_mb, group_gb))

    def supports(self, p: Problem) -> Optional[str]:
        why = super().supports(p)
        if why:
            return why
        own = "f16" if p.f16 else "bf16"
        if p.f32 or p.int8_scheme != ("per_group" if p.int4 else "per_channel"):
            return "the driver makes bf16, s8 (per-channel scales), s4 and f16 calls"
        if p.src != own or p.dst != own or p.trans_a or p.has_bias or p.epilogue != "store":
            return "the driver makes %s -> %s calls without trans_a, bias or post-ops" % (own, own)
        if p.int8 and not p.dynamic:
            return "the driver quantizes the s8 and s4 problems' source per token (dynamic)"
        if p.int4 and (p.scale_bytes == 4) != (self.scale_dt == "f32"):
            return "the s4 problem's scales take %d bytes, the backend's are %s" % (p.scale_bytes, self.scale_dt)
        if self.group_gb > 0 and self.process_gb([p]) > self.group_gb:
            return ("a timed process of this problem alone allocates %.3f GB of weights (%d layers), above "
                    "group_gb %g" % (self.process_gb([p]), self.timed_layers([p]), self.group_gb))
        return None

    def cost(self, p: Problem, layers: int = 0) -> float:
        """Seconds of one timed run on p over layers weight sets (default --layers): the
        warm-up, sizing and timed passes at a roofline estimate of one call."""
        layers = layers or self.layers
        call = max(2.0 * p.m * p.n * p.k / (8e12 if p.int8 else 4e12), float(p.n) * p.k * p.wei_esz / 2e11)
        iters = min(self.iters, max(3, int(self.budget_ms / 1000.0 / (layers * call))))
        return layers * call * (self.warmup + 1 + iters)

    @staticmethod
    def _set_mb(problems: Sequence[Problem]) -> float:
        """MB of one weight set of a process over problems: each shape once."""
        shapes = {(p.n, p.k): p.wei_esz for p in problems}
        return sum(n * k * esz for (n, k), esz in shapes.items()) / 1e6

    def timed_layers(self, problems: Sequence[Problem]) -> int:
        """--layers of a timed process over problems: at least cold_mb MB of distinct weights."""
        if self.cold_mb <= 0:
            return self.layers
        return max(self.layers, int(math.ceil(self.cold_mb / self._set_mb(problems))))

    def process_gb(self, problems: Sequence[Problem]) -> float:
        """GB of the weights a timed process over problems allocates: every layer it times."""
        return self.timed_layers(problems) * self._set_mb(problems) / 1e3

    def groups(self, problems: Sequence[Problem]) -> List[List[Problem]]:
        """Per dtype, the Ms in order, each in filter_parts() by N and K; parts of consecutive Ms
        with the same shapes share a process (<= 64 clauses).
        A group above group_cost seconds (cost()) or group_gb weight GB (process_gb()), each
        at the layers the process times, is cut into single-M parts below the caps; a
        problem whose process alone is above group_gb is one supports() refuses."""
        out: List[List[Problem]] = []
        for g in self._clause_groups(problems):
            out += self._split(g)
        return out

    def _over(self, g: Sequence[Problem]) -> bool:
        layers = self.timed_layers(g)
        if self.group_cost > 0 and sum(self.cost(q, layers) for q in g) > self.group_cost:
            return True
        return self.group_gb > 0 and self.process_gb(g) > self.group_gb

    def _split(self, g: List[Problem]) -> List[List[Problem]]:
        if len(g) < 2 or not self._over(g):
            return [g]
        parts: List[List[Problem]] = []
        cur: List[Problem] = []
        for q in g:
            if cur and (q.m != cur[-1].m or self._over(cur + [q])):
                parts.append(cur)
                cur = []
            cur.append(q)
        if cur:
            parts.append(cur)
        return parts

    def _clause_groups(self, problems: Sequence[Problem]) -> List[List[Problem]]:
        out: List[List[Problem]] = []
        # One process takes one weight type, and one --group of s4 weights.
        by: Dict[Tuple[str, int], Dict[int, List[Problem]]] = defaultdict(lambda: defaultdict(list))
        for p in problems:
            by[(p.dtype, replay_group(p))][p.m].append(p)
        for dtype in sorted(by):
            cur: List[Problem] = []
            cur_shapes: Optional[Tuple] = None
            for m in sorted(by[dtype]):
                for chunk in filter_parts(sorted(by[dtype][m], key=lambda q: (q.n, q.k))):
                    shapes = tuple((q.n, q.k) for q in chunk)
                    if cur and (shapes != cur_shapes or len(cur) + len(chunk) > MAX_CLAUSES):
                        out.append(cur)
                        cur = []
                    cur += chunk
                    cur_shapes = shapes
            if cur:
                out.append(cur)
        return out

    def preamble(self) -> List[str]:
        # On a pod with more CPUs than the team, the team takes the first L3 domains.
        return [
            "unset OMP_PROC_BIND OMP_PLACES OMP_WAIT_POLICY",
            "for v in $(compgen -e | grep -E '^(ZENDNNL_|KMP_)'); do unset \"$v\"; done",
            CPU_LIST,
            "LIST=$(echo \"$LIST\" | tr ',' '\\n' | head -n %d | paste -sd, -)" % self.threads,
            'export OMP_NUM_THREADS=%d KMP_AFFINITY="granularity=fine,explicit,proclist=[$LIST]"' % self.threads,
            "export KMP_BLOCKTIME=5 KMP_TPAUSE=0",
            "REPLAY=%s" % self.binary,
            "PRE=%s" % self.preload,
            'rp() { if [ -n "$PRE" ]; then env LD_PRELOAD="$PRE" "$REPLAY" "$@"; else "$REPLAY" "$@"; fi; }',
        ]

    def probe_env(self) -> List[str]:
        # preamble()'s proclist as one place per CPU: GCC's runtime does not read KMP_AFFINITY.
        return [
            "for v in $(compgen -e | grep -E '^(ZENDNNL_|KMP_|GOMP_)'); do unset \"$v\"; done",
            "unset OMP_WAIT_POLICY",
            CPU_LIST,
            "LIST=$(echo \"$LIST\" | tr ',' '\\n' | head -n %d | paste -sd, -)" % self.threads,
            "export OMP_NUM_THREADS=%d OMP_PROC_BIND=close "
            "OMP_PLACES=\"$(echo \"$LIST\" | sed -E 's/[0-9]+/{&}/g')\"" % self.threads,
        ]

    def command(self, gid: int, problems: Sequence[Problem], probe: bool) -> str:
        kinds = {(p.dtype, replay_group(p)) for p in problems}
        if len(kinds) != 1:
            raise ValueError("one replay process takes one weight type (and one s4 --group)")
        dtype, group = kinds.pop()
        shapes = sorted({(p.n, p.k) for p in problems})
        ms = sorted({p.m for p in problems})
        args = ["--wei", dtype, "--shapes", ",".join("n%dk%d:%d:%d" % (n, k, n, k) for n, k in shapes),
                "--ms", ",".join(map(str, ms)), "--scale-dt", self.scale_dt]
        if dtype == "s4":
            args += ["--group", str(group)]
        if probe:
            args += ["--layers", "1", "--iters", "1", "--warmup", "0", "--budget-ms", "0", "--check", "1"]
        else:
            args += ["--layers", str(self.timed_layers(problems)), "--iters", str(self.iters),
                     "--warmup", str(self.warmup), "--budget-ms", str(self.budget_ms)]
        return "rp " + " ".join(args)

    def parse_timed(self, text: str, problems: Sequence[Problem]) -> Dict[str, Dict[str, Any]]:
        found: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        for line in text.splitlines():
            m = RESULT_RE.match(line.strip())
            if m:
                found[(int(m["M"]), int(m["N"]), int(m["K"]))] = {
                    "time_ms": float(m["median"]),
                    "stats": {"calls": int(m["calls"]), "mean_ms": float(m["mean"]),
                              "min_ms": float(m["min"]), "p90_ms": float(m["p90"])}}
        return {p.key(): found[(p.m, p.n, p.k)] for p in problems if (p.m, p.n, p.k) in found}

    def parse_probe(self, text: str, problems: Sequence[Problem]) -> Dict[str, Dict[str, Any]]:
        out = super().parse_probe(text, problems)
        hashes = {}
        for line in text.splitlines():
            m = CHECK_RE.match(line.strip())
            if m:
                hashes[(int(m["M"]), int(m["N"]), int(m["K"]))] = m["hash"].lower()
        for p in problems:
            out[p.key()]["hash"] = hashes.get((p.m, p.n, p.k))
        return out

    def describe(self) -> Dict[str, Any]:
        return {"backend": self.name, "binary": self.binary, "threads": self.threads, "preload": self.preload,
                "layers": self.layers, "iters": self.iters, "warmup": self.warmup,
                "budget_ms": self.budget_ms, "scale_dt": self.scale_dt, "group_cost": self.group_cost,
                "group_gb": self.group_gb, "cold_mb": self.cold_mb}
