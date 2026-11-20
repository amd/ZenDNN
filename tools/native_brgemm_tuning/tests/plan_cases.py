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
"""Random problems, machines and configs for checking plan.py against the
library's planner, and the digests of the library's plans for them.

cases(seed, count) draws the same cases on every Python version.  A plan dump
of the library reads one case per line, "cfg l1d l2 l3 family_l3 dom wei scheme
N K M threads pack_nr group scale_bytes" (cfg: an override setting list, or -;
l3 0: no L3; wei: bf16, s8, s4, f16 or f32; scheme: the INT8 scheme of an s8
weight; group and scale_bytes: those of an s4 weight, 0 otherwise), builds the
problem of that call with make_native_problem (N not a multiple of 32 is
planned at its padded width) and prints, per line, the problem's N and the
plan of heuristic_plan(p, uarch, apply_native_config({},
parse_native_config(cfg))) with validate_native_plan's verdict, as plan_line()
writes it, or "BADCFG ..." for a config the parser rejects.  A case without a
config plans with the library's defaults, those of its schema (--schema); a
fraction of the cases sets every rule switch to 0, and another every cost-model
input.  --machine fixes the machine of every case.  The kernels of an f16 call
follow the registry, so the dump and the schema come from machines of one ISA.
fixtures/plan_check.json holds the SHA-256 of each chunk of those lines and the
counts of the kinds of case; tuner_line() is the same line from plan.py.

  plan_cases.py cases   [--seed S] [--count N] [--machine M]            the dump's input
  plan_cases.py compare DUMP [--seed S] [--count N] [--machine M] [--schema]
                                                    plan.py against a dump, field by field
  plan_cases.py fixture DUMP [--seed S] [--count N] [--schema]          the digest fixture of a dump
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from native_brgemm_tuning import schema as schema_mod  # noqa: E402
from native_brgemm_tuning.candidates import parse_settings  # noqa: E402
from native_brgemm_tuning.plan import (  # noqa: E402
    RULE_SWITCHES, Plan, heuristic_plan, params_from_config, validate_plan)
from native_brgemm_tuning.problems import Problem  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(schema_mod.FIXTURE), "plan_check.json")
SEED = 20261023
COUNT = 50000
CHUNK = 1000
MASK64 = (1 << 64) - 1

# Each key is set with the given percent chance; values in the override grammar.
RULE_KEYS = (
    ("makespan", ["0", "1"]), ("l1_pass", ["0", "1"]), ("traffic_groups", ["0", "1"]),
    ("min_gain_pct", ["0", "1", "2", "5", "10", "20", "100"]), ("idle_pct", ["1", "5", "10", "25", "50", "100"]),
    ("l1_pct", ["1", "50", "75", "100", "133", "150", "200", "1000"]),
    ("l1_pass_max_mb", ["1", "24", "32", "64", "128", "192", "256", "8192"]),
    ("min_pass_k", ["2", "64", "128", "256", "512", "1024"]),
    ("bf16_alias", ["0", "1"]), ("bf16_alias_mr", ["1", "4", "6", "8", "12"]),
    ("bf16_narrow_n", ["0", "1"]), ("bf16_narrow_n_min_m", ["1", "64", "1024", "4096"]),
    ("f16_wide_min_m", ["1", "64", "192", "1024", "1073741824"]),
    ("blocks", ["0", "1"]), ("bf16_blocks_min_m", ["1", "256", "1024", "4096"]),
    ("int8_blocks_min_m", ["1", "1024", "3072", "8192"]),
    ("int4_l1_pct", ["0", "50", "125", "400"]),
    ("cost_model", ["0", "1"]), ("cm_gain_pct", ["0", "2", "5", "20", "100"]),
)
# The cost model's inputs, each set with its own chance in the cases that vary them.
COST_KEYS = (
    ("cm_short_rows", ["1", "4", "6", "8", "10", "12"]), ("cm_fma_pipes", ["0", "1", "2", "4"]),
    ("cm_fma_lat", ["1", "4", "6", "12"]), ("cm_call_cycles", ["0", "200", "800", "4000"]),
    ("cm_tall_pct", ["-1", "0", "25", "50", "100"]), ("cm_l2_bytes", ["1", "8", "16", "64"]),
    ("cm_l3_cycles", ["0", "10", "40"]), ("cm_mem_cycles", ["0", "100", "424", "600", "849", "2400"]),
    ("cm_prefetch_kib", ["0", "4", "64"]), ("cm_overlap_pct", ["0", "20", "100"]),
    ("cm_reuse_pct", ["1", "50", "200", "1000"]), ("cm_stream_cycles", ["0", "424", "600", "849", "100000"]),
    ("cm_sync_tiles", ["0", "1", "5", "100"]), ("cm_group_slab_pct", ["1", "75", "100000"]),
    ("cm_group_l3_pct", ["1", "75", "100000"]), ("cm_group_skew_pct", ["100", "110", "200"]),
    ("cm_group_rows_pct", ["100", "115", "200"]), ("cm_drift_kib", ["0", "64", "2048", "4096", "65536"]),
    ("cm_pass_kib", ["0", "16", "128", "1024"]), ("cm_pass_calls", ["1", "2", "4", "16"]),
    ("cm_block_kib", ["0", "1024", "16384", "1073741824"]),
    ("cm_block_k", ["0", "1024", "2048", "4096", "1073741824"]), ("cm_int8_l1_pct", ["0", "50", "100", "200"]),
    ("cm_int4_min_m", ["0", "1", "16", "48", "1024"]),
)
PINS = (
    ("nr", ["16", "32", "64", "128"]), ("mr", ["1", "4", "6", "7", "8", "12"]),
    ("mb", ["1", "7", "24", "100", "192", "768"]), ("kc", ["2", "64", "256", "1000"]),
    ("col_groups", ["0", "1", "2", "4", "8"]), ("threads_cap", ["1", "7", "24"]),
    ("min_tiles_per_thread", ["1", "2", "8"]), ("row_unit", ["1", "5", "8", "12", "48"]),
    ("tile_rows", ["1", "4", "6", "7"]), ("tall_rows", ["1", "8", "12"]),
    ("block_rows", ["24", "100", "192", "768"]), ("min_tile_cols", ["16", "32"]),
    ("row_split", ["0", "1"]), ("bf16_l2_pct", ["1", "50", "200"]), ("kc_split", ["budget", "equal"]),
    ("l3_pct", ["1", "100", "400"]), ("domain_threads", ["1", "2", "8", "16", "32"]),
    ("int8_mr", ["1", "5", "12"]),
    ("int8_cw", ["16", "32/16"]), ("int8_l2_pct", ["1", "50", "1000"]),
    ("int8_kc", ["4", "64", "1024"]), ("unit_tiles", ["1", "2", "3", "8", "4096"]),
)
# Weight types of the drawn cases and their percent shares; s4 group sizes (0: one group over K).
DTYPE_SHARES = (("bf16", 34), ("s8", 26), ("s4", 22), ("f16", 12), ("f32", 6))
S4_GROUPS = (0, 0, 32, 64, 128, 128, 256)
SCHEDULES = ("slice", "row_groups", "blocks")
# (N, K) of dense layers and lm_heads, and the Ms of the default M classes.
LAYERS = ((4096, 4096), (6144, 4096), (14336, 4096), (4096, 14336), (11008, 4096), (4096, 11008),
          (2048, 2048), (5120, 5120), (13824, 5120), (5120, 13824), (1536, 1536), (3072, 8192),
          (8192, 3072), (28672, 8192), (8192, 28672), (896, 896), (4864, 896), (896, 4864),
          (151936, 2048), (262144, 1536), (128256, 4096))
LAYER_MS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


class Rng:
    """SplitMix64: the same draws on every Python version."""

    def __init__(self, seed: int):
        self.state = seed & MASK64

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def below(self, n: int) -> int:
        return self.next() % n

    def randint(self, lo: int, hi: int) -> int:
        return lo + self.below(hi - lo + 1)

    def choice(self, seq: Sequence[Any]) -> Any:
        return seq[self.below(len(seq))]

    def percent(self, p: int) -> bool:
        return self.below(100) < p


class Case(NamedTuple):
    cfg: str
    l1d: int
    l2: int
    l3: int
    family_l3: int
    dom: int
    dtype: str
    scheme: str
    n: int
    k: int
    m: int
    threads: int
    pack_nr: int
    group: int
    scale_bytes: int

    def settings(self) -> Dict[str, str]:
        return dict(item.partition("=")[::2] for item in self.cfg.split(","))

    def rules_off(self) -> bool:
        """The case sets every rule switch to 0."""
        cfg = self.settings()
        return all(cfg.get(s) == "0" for s in RULE_SWITCHES)

    def costs_varied(self) -> bool:
        """The case sets a cost-model input."""
        return any(k in self.settings() for k, _ in COST_KEYS)

    def line(self) -> str:
        return "%s %d %d %d %d %d %s %s %d %d %d %d %d %d %d" % self

    def problem(self) -> Problem:
        if self.dtype == "s4":
            return Problem("s4", self.m, self.n, self.k, threads=self.threads, pack_nr=self.pack_nr,
                           group=self.group, scale_bytes=self.scale_bytes)
        return Problem(self.dtype, self.m, self.n, self.k, threads=self.threads, pack_nr=self.pack_nr,
                       int8_scheme=self.scheme if self.dtype == "s8" else "per_channel")

    def uarch(self) -> schema_mod.Uarch:
        return schema_mod.Uarch(l1d_bytes=self.l1d, l2_bytes=self.l2, l3_bytes=self.l3,
                                family_l3_cores=self.family_l3, l3_domain_cores=self.dom)


def random_config(rng: Rng, m: int) -> str:
    roll = rng.below(100)
    if roll < 15:
        return "-"
    cfg: Dict[str, str] = {}
    if roll < 45:
        cfg.update((s, "0") for s in RULE_SWITCHES)
    for key, values in RULE_KEYS:
        if rng.percent(15):
            cfg[key] = rng.choice(values)
    if rng.percent(10):
        cfg["decode_m"] = rng.choice(["0", "8", "128", "4096", str(m), str(m - 1)])
    if rng.percent(40):
        for key, values in COST_KEYS:
            if rng.percent(20):
                cfg[key] = rng.choice(values)
    for key, values in PINS:
        if rng.percent(5):
            cfg[key] = rng.choice(values)
    if "col_groups" in cfg and rng.percent(50):
        cfg["schedule"] = "row_groups" if cfg["col_groups"] != "0" else "slice"
    elif rng.percent(6):
        cfg["schedule"] = rng.choice(SCHEDULES)
        cfg.pop("col_groups", None)
    return ",".join("%s=%s" % kv for kv in cfg.items()) or "-"


def random_dtype(rng: Rng) -> str:
    roll = rng.below(100)
    for dtype, share in DTYPE_SHARES:
        if roll < share:
            return dtype
        roll -= share
    return DTYPE_SHARES[0][0]


def random_case(rng: Rng, machine: Optional[Tuple[int, int, int, int, int]] = None) -> Case:
    if rng.percent(30):
        n, k = rng.choice(LAYERS)
        m = rng.choice(LAYER_MS) if rng.percent(70) else rng.randint(1, 4096)
        threads = rng.choice([24, 32, 48, 64, 96, 128, 192, 256])
        pack = rng.choice([32, 32, 64])
    else:
        pack = rng.choice([32, 32, 32, 64])
        m = rng.choice([rng.randint(1, 16), rng.randint(17, 160), rng.randint(161, 2000), rng.randint(2001, 8192)])
        n = pack * rng.choice([1, 2, 3, rng.randint(1, 64), rng.randint(64, 2048)])
        k = rng.choice([1, 7, 64, rng.randint(1, 1024), rng.randint(1024, 32768)])
        threads = rng.choice([1, 8, 24, 32, 33, 64, rng.randint(1, 256)])
    # Any N: a share of the weights are not whole 32-column blocks.
    if rng.percent(15):
        n = rng.choice([rng.randint(1, 31), rng.randint(33, 4095), rng.randint(4097, 65536)])
    dtype = random_dtype(rng)
    scheme = "per_group" if dtype == "s4" or (dtype == "s8" and rng.percent(25)) else "per_channel"
    group = scale_bytes = 0
    if dtype == "s4":
        # The group size divides K; one group takes all of K.
        group = rng.choice(S4_GROUPS)
        if group:
            k = group * max(1, k // group)
        else:
            group = k
        scale_bytes = rng.choice([2, 2, 4])
    l1d = rng.choice([32768, 49152])
    l2 = rng.choice([512 << 10, 1 << 20, 2 << 20])
    l3 = rng.choice([16 << 20, 32 << 20, 128 << 20]) if rng.percent(95) else 0
    family_l3 = rng.choice([1, 8])
    dom = rng.choice([0, 8, 16, 32])
    if machine is not None:
        l1d, l2, l3, family_l3, dom = machine
    return Case(random_config(rng, m), l1d, l2, l3, family_l3, dom, dtype, scheme, n, k, m, threads, pack,
                group, scale_bytes)


def cases(seed: int = SEED, count: int = COUNT,
          machine: Optional[Tuple[int, int, int, int, int]] = None) -> List[Case]:
    rng = Rng(seed)
    return [random_case(rng, machine) for _ in range(count)]


def parse_machine(text: str) -> Tuple[int, int, int, int, int]:
    """"l1d,l2,l3,family_l3,dom" in bytes and cores, e.g. 49152,1048576,33554432,8,8."""
    v = [int(x) for x in text.split(",")]
    if len(v) != 5:
        raise ValueError("a machine is l1d,l2,l3,family_l3,dom: %r" % text)
    return v[0], v[1], v[2], v[3], v[4]


def plan_line(p: Problem, q: Plan, valid: bool) -> str:
    return ("n=%d nr=%d mr=%d mb=%d m_blocks=%d n_tiles=%d kc=%d kind=%s threads=%d col_groups=%d unit_tiles=%d "
            "domain_threads=%d row_unit=%d traversal=%s scratch=%d rows=%d/%d cw=%d budget=%d quads=%d "
            "direct=%d dequant_bias=%d scheme=%s copy_block=%d copy_par_min=%d valid=%d" % (
                p.plan_n, q.nr, q.mr, q.mb, q.m_blocks, q.n_tiles, q.kc, q.kind, q.threads, q.col_groups,
                q.unit_tiles, q.domain_threads, q.row_unit, q.traversal, q.scratch_floats, q.int8_rows_nv1,
                q.int8_rows_nv2, q.int8_cw_mask, q.int8_pass_budget, q.int8_pass_quads, q.int8_direct_store,
                q.int8_bias_in_dequant, q.int8_scheme, q.copy_block, q.copy_par_min, valid))


def tuner_line(c: Case, schema: Any) -> str:
    """The dump's line for c, from plan.py and the tuner's config parser."""
    try:
        cfg = parse_settings(c.cfg, schema) if c.cfg != "-" else {}
    except ValueError:
        return "BADCFG"
    p = c.problem()
    q = heuristic_plan(p, c.uarch(), params_from_config(cfg, schema), schema)
    return plan_line(p, q, validate_plan(q, p, schema) is None)


def normalized(line: str) -> str:
    """A dump line as the digests take it: the parser's reason dropped."""
    line = line.strip()
    return "BADCFG" if line.startswith("BADCFG") else line


def digests(lines: Sequence[str], chunk: int = CHUNK) -> List[str]:
    out = []
    for i in range(0, len(lines), chunk):
        text = "".join(normalized(x) + "\n" for x in lines[i:i + chunk])
        out.append(hashlib.sha256(text.encode("ascii")).hexdigest())
    return out


def case_counts(cs: Sequence[Case]) -> Dict[str, int]:
    """Cases without a config, cases that set every rule switch to 0, cases that set a cost-model
    input, and the cases of each weight type."""
    out = {"no_config": sum(c.cfg == "-" for c in cs), "rules_off": sum(c.rules_off() for c in cs),
           "cost_inputs": sum(c.costs_varied() for c in cs)}
    for dtype, _ in DTYPE_SHARES:
        out[dtype] = sum(c.dtype == dtype for c in cs)
    return out


def load_fixture(path: str = FIXTURE) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _fields(line: str) -> Dict[str, str]:
    return dict(t.partition("=")[::2] for t in normalized(line).split())


def compare(cs: Sequence[Case], dump: Sequence[str], schema: Any) -> Tuple[int, List[str]]:
    """(mismatched cases, the first few as text) of plan.py against a dump's lines."""
    if len(dump) != len(cs):
        raise ValueError("%d dump lines for %d cases" % (len(dump), len(cs)))
    bad = 0
    shown: List[str] = []
    for c, lib in zip(cs, dump):
        mine = tuner_line(c, schema)
        if normalized(lib) == mine:
            continue
        bad += 1
        if len(shown) < 10:
            a, b = _fields(lib), _fields(mine)
            diff = ["%s lib %s tuner %s" % (k, a.get(k), b.get(k))
                    for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
            shown.append("%s: %s" % (c.line(), ", ".join(diff)))
    return bad, shown


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("cases", "compare", "fixture"))
    ap.add_argument("dump", nargs="?", help="the plan dump's output for the cases")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--count", type=int, default=COUNT)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--schema", help="schema of the library (default: the fixture)")
    ap.add_argument("--machine", help="l1d,l2,l3,family_l3,dom of every case (default: drawn per case)")
    a = ap.parse_args(argv)
    if a.machine and a.command == "fixture":
        ap.error("the fixture's cases draw their machines")
    cs = cases(a.seed, a.count, parse_machine(a.machine) if a.machine else None)
    if a.command == "cases":
        sys.stdout.write("".join(c.line() + "\n" for c in cs))
        return 0
    if not a.dump:
        ap.error("%s needs the dump's output" % a.command)
    with open(a.dump, encoding="utf-8") as f:
        dump = [x for x in f.read().splitlines() if x.strip()]
    schema = schema_mod.load_schema(a.schema)
    counts = case_counts(cs)
    if a.command == "compare":
        bad, shown = compare(cs, dump, schema)
        for s in shown:
            print(s)
        blocks = sum(" unit_tiles=0 " not in normalized(x) and normalized(x) != "BADCFG" for x in dump)
        print("cases %d: %d without a config, %d with every rule off, %d plans of the blocks schedule; "
              "mismatches %d" % (len(cs), counts["no_config"], counts["rules_off"], blocks, bad))
        return 1 if bad else 0
    doc = {"origin": "plans of the library's planner for the cases of plan_cases.py",
           "seed": a.seed, "count": a.count, "chunk": a.chunk, "cases": counts, "sha256": digests(dump, a.chunk)}
    sys.stdout.write(json.dumps(doc, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
