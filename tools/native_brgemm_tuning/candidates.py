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
"""Candidates: legal, de-duplicated knob sets per problem, and the
ZENDNNL_MATMUL_NATIVE_CONFIG text that applies them (DESIGN section 6.1).

A candidate is a partial config (keys left out stay at their built-in value or
auto).  Space(problem) holds the values each searched key may take for that
problem; Space.make() keeps a config only when it passes the static checks
(section 11.1), uses no bit-changing key unless allowed (approval decision 6),
and resolves to a legal plan (section 11.2).  Its effective-plan signature
identifies it: two configs with one signature run the same work, so only the
first is measured.
"""

from __future__ import annotations

import itertools
import random
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from . import cost_model as cm
from .plan import Plan, builtin_plan, ceil_div, effective_signature, mirrored, resolve, slab_tile_width
from .problems import Problem
from .schema import Schema

MAX_CLAUSES = 64
FILTER_KEYS = ("m", "n", "k", "threads", "pack_nr", "group", "dtype", "src", "dst", "epilogue",
               "trans_a", "dynamic", "has_bias")


# ---------------------------------------------------------------- config text

def format_settings(config: Dict[str, Any], schema: Schema) -> str:
    """format_native_config: "key=value,..." in schema order; "" when empty."""
    return ",".join("%s=%s" % (k.name, k.format(config[k.name])) for k in schema.keys if k.name in config)


def check_config(config: Dict[str, Any], schema: Schema) -> Optional[str]:
    """check_native_config: the first static violation (section 11.1), or None."""
    for name, v in config.items():
        if name not in schema.key_map:
            return "unknown key %s" % name
        why = schema.key_map[name].check(v)
        if why:
            return why
    sched, cg = config.get("schedule"), config.get("col_groups")
    if sched == "slice" and cg is not None and cg > 0:
        return "schedule=slice conflicts with col_groups=%d" % cg
    if sched == "row_groups" and cg == 0:
        return "schedule=row_groups conflicts with col_groups=0"
    if sched == "blocks" and cg is not None:
        return "schedule=blocks conflicts with col_groups=%d" % cg
    return None


def parse_settings(text: str, schema: Schema) -> Dict[str, Any]:
    """The config of "key=value,..."; ValueError on a bad token or a static violation."""
    cfg: Dict[str, Any] = {}
    for item in filter(None, (t.strip() for t in text.split(","))):
        k, eq, v = item.partition("=")
        name = k.strip().lower()
        if not eq or name not in schema.key_map:
            raise ValueError("bad setting %r" % item)
        cfg[name] = schema.key_map[name].parse(v)
    why = check_config(cfg, schema)
    if why:
        raise ValueError(why)
    return {k.name: cfg[k.name] for k in schema.keys if k.name in cfg}


def builtin_value(spec: Any) -> Any:
    """The value heuristic_params_t{} holds for a key: its default, or its auto value where
    that is a legal setting (row_unit 12, cm_tall_pct -1, cm_mem_cycles 0, cm_fma_pipes 0):
    setting a key to it is the same as leaving it unset."""
    if spec.default is not None:
        return spec.default
    if spec.auto is not None and spec.check(spec.auto) is None:
        return spec.auto
    return None


def canonical(config: Dict[str, Any], schema: Schema, dtype: str, keep: Sequence[str] = ()) -> Dict[str, Any]:
    """config without keys that do not apply to dtype or equal their built-in value (but those
    of keep)."""
    out = {}
    for spec in schema.keys:
        if spec.name not in config or not spec.applies_to(dtype):
            continue
        v = config[spec.name]
        if spec.type == "list":
            v = sorted(set(v), reverse=True)
        b = builtin_value(spec)
        if b is not None and v == b and spec.name not in keep:
            continue
        out[spec.name] = v
    return out


def problem_filters(p: Problem, m_lo: Optional[int] = None, m_hi: Optional[int] = None) -> str:
    """Override filters that select p (or p's shape over M m_lo .. m_hi); n is the
    width the planner matches (plan_n)."""
    if m_lo is None:
        m = str(p.m)
    else:
        m = "%d-%s" % (m_lo, "" if m_hi is None else m_hi)
    text = "dtype=%s,m=%s,n=%d,k=%d,threads=%d" % (p.dtype, m, p.plan_n, p.k, p.threads)
    return text + (",group=%d" % p.group if p.int4 else "")


def format_override(clauses: Sequence[Tuple[str, str]]) -> str:
    """ZENDNNL_MATMUL_NATIVE_CONFIG of (filters, settings) clauses, in order."""
    if len(clauses) > MAX_CLAUSES:
        raise ValueError("%d clauses: the library reads at most %d" % (len(clauses), MAX_CLAUSES))
    return ";".join("%s:%s" % (f, s) if f else s for f, s in clauses)


def parse_override(text: str, schema: Schema) -> List[Tuple[Dict[str, str], Dict[str, Any]]]:
    """(filters, config) per clause; ValueError on a bad clause."""
    out = []
    for clause in filter(None, (c.strip() for c in text.split(";"))):
        filt_text, colon, settings = clause.partition(":")
        if not colon:
            filt_text, settings = "", clause
        filters: Dict[str, str] = {}
        for item in filter(None, (t.strip() for t in filt_text.split(","))):
            k, eq, v = item.partition("=")
            if not eq or k.strip().lower() not in FILTER_KEYS:
                raise ValueError("bad filter %r" % item)
            filters[k.strip().lower()] = v.strip()
        out.append((filters, parse_settings(settings, schema)))
    return out


# ---------------------------------------------------------------- search space

def search_keys(schema: Schema, dtype: str, allow_bit_changing: bool = False,
                include: Optional[Iterable[str]] = None, exclude: Iterable[str] = ()) -> List[str]:
    """Keys searched for dtype: those that apply to it and keep its bits
    (decision 6).  By default only the keys the plan mirror reads: a candidate
    of any other key resolves as dropped; and the unit of the blocks schedule
    as nb, its columns whatever nr the plan takes, where the schema has nb,
    not as unit_tiles."""
    names = list(include) if include is not None else [k.name for k in schema.keys if mirrored(k.name)]
    skip = set(exclude)
    if include is None and "nb" in schema.key_map:
        skip.add("unit_tiles")
    out = []
    for n in names:
        spec = schema.key(n)
        if not spec.applies_to(dtype) or n in skip:
            continue
        if not spec.same_bits(dtype) and not allow_bit_changing:
            continue
        out.append(n)
    return out


def _value_rule(key: str, p: Problem, schema: Schema) -> List[Any]:
    route = schema.route(p)
    kind = route.family
    t = p.threads
    if key == "nr":
        # The widths of the route's kernels: 16 and 32 within a block, tiles of slabs where the
        # build has them (every kind, or BF16 only in an older one), and a slab of two blocks
        # (the F16_PH tile of 64).
        slab = cm.SLAB_COLS.get(kind, 32)
        return [w for w in schema.nr_widths(kind) if (w in (16, 32) and w <= p.pack_nr and p.pack_nr % w == 0)
                or slab_tile_width(kind, w, p.pack_nr) or (w == slab > 32 and p.pack_nr == 32)]
    if key == "mr":
        # The rows of the route's kernels; the rows of an INT8 call are int8_mr's.
        return [r for r in (2, 4, 6, 8, 12)
                if route.cls != "rows" or any(schema.covers(kind, r, w) for w in schema.nr_widths(kind))]
    if key == "mb":
        vals = {p.m, 6, 12, 24, 48, 96, 192, 384, 768, 1536}
        vals |= {ceil_div(ceil_div(p.m, d), 6) * 6 for d in (2, 4, 8)}
        return sorted(v for v in vals if 1 <= v <= p.m)
    if key == "kc":
        return [p.k] + [v for v in (256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192) if v < p.k]
    if key == "col_groups":
        return [0] + [g for g in (1, 2, 4, 8, 16, 32) if g <= t and t % g == 0]
    if key == "threads_cap":
        return sorted({v for v in (t // 4, t // 2, 3 * t // 4, 8, 16, 24) if 1 <= v < t})
    if key == "domain_threads":
        return [v for v in (4, 8, 16, 32) if v <= t]
    if key == "kc_split":
        return ["equal"]
    if key == "int8_kc":
        return [v for v in (256, 512, 1024, 2048) if v < p.k]
    if key == "decode_m":
        # Only the side of M it falls on counts: multiply-adds at every M, or bytes up to this one.
        return [0, p.m]
    fixed = {
        "nb": [64, 128, 256],
        "schedule": ["slice", "row_groups", "blocks"], "traversal": ["n_inner"], "unit_tiles": [2, 4],
        "min_tiles_per_thread": [2, 4], "row_unit": [6, 24, 48], "tile_rows": [4, 8, 12],
        "tall_rows": [6, 8], "block_rows": [96, 384, 768], "min_tile_cols": [32],
        "row_split": [False], "min_pass_k": [128, 512, 1024], "bf16_l2_pct": [25, 75, 100],
        "l3_pct": [50, 200, 400], "int8_mr": [4, 6, 8],
        "int8_cw": [[32, 16], [16]], "int8_l2_pct": [25, 100, 200],
        "direct_store": [False], "bias_in_dequant": [False], "copy_block": [32, 128, 256],
        "bf16_copy_par_min": [1 << 16, 1 << 40], "int8_copy_par_min": [0, 1 << 40],
        "makespan": [0, 1], "l1_pass": [0, 1], "traffic_groups": [0, 1],
        "min_gain_pct": [0, 2, 10, 20], "idle_pct": [5, 25, 50],
        "l1_pct": [50, 75, 150, 200], "l1_pass_max_mb": [32, 128, 192],
        "bf16_alias": [0, 1], "bf16_alias_mr": [4, 6, 12], "bf16_narrow_n": [0, 1],
        # The narrow-N, F16_PH wide-tile and blocks rules take calls from these Ms: 1 turns
        # them on at every M, 2^30 off.
        "bf16_narrow_n_min_m": [1], "f16_wide_min_m": [1, 64, 1 << 30], "blocks": [0, 1],
        "bf16_blocks_min_m": [1], "int8_blocks_min_m": [1], "int4_l1_pct": [0, 50, 250],
        # The cost model: its switch, margin and inputs.
        "cost_model": [0, 1], "cm_gain_pct": [0, 2, 10], "cm_short_rows": [4, 6, 10],
        "cm_fma_pipes": [1, 4], "cm_fma_lat": [4, 8], "cm_call_cycles": [400, 1600],
        "cm_tall_pct": [0, 25, 50], "cm_l2_bytes": [8, 32], "cm_l3_cycles": [5, 20],
        "cm_mem_cycles": [300, 600, 1200], "cm_prefetch_kib": [0, 16], "cm_overlap_pct": [0, 50],
        "cm_reuse_pct": [25, 100], "cm_stream_cycles": [400, 900], "cm_sync_tiles": [1, 20],
        "cm_group_slab_pct": [50, 150], "cm_group_l3_pct": [50, 150], "cm_group_skew_pct": [100, 150],
        "cm_group_rows_pct": [100, 150], "cm_drift_kib": [1024, 16384], "cm_pass_kib": [0, 64, 256],
        "cm_pass_calls": [2, 8], "cm_block_kib": [4096, 65536], "cm_block_k": [0, 1024, 4096],
        "cm_int8_l1_pct": [50, 100], "cm_int4_min_m": [0, 16, 128],
    }
    if key in fixed:
        return fixed[key]
    spec = schema.key(key)
    return list(spec.values or [])


def value_choices(key: str, p: Problem, schema: Schema) -> List[Any]:
    """Values (other than the built-in one) the search tries for key on p."""
    spec = schema.key(key)
    builtin = builtin_value(spec)
    out = []
    for v in _value_rule(key, p, schema):
        if spec.type == "list":
            v = sorted(set(v), reverse=True)
        if spec.check(v) is None and v != builtin and v not in out:
            out.append(v)
    return out


def parse_values(items: Iterable[str], schema: Schema) -> Dict[str, List[Any]]:
    """"key=a|b|..." items (one key each) as value lists."""
    out: Dict[str, List[Any]] = {}
    for item in items:
        k, eq, vals = item.partition("=")
        name = k.strip().lower()
        if not eq:
            raise ValueError("bad --values item %r (key=a|b)" % item)
        spec = schema.key(name)
        parsed = [spec.parse(v) for v in vals.split("|") if v.strip()]
        for v in parsed:
            why = spec.check(v)
            if why:
                raise ValueError(why)
        out[name] = parsed
    return out


class Candidate:
    """A canonical config, its override text, resolved plan and plan signature."""

    __slots__ = ("config", "text", "plan", "sig")

    def __init__(self, config: Dict[str, Any], text: str, plan: Plan, sig: Tuple):
        self.config = config
        self.text = text
        self.plan = plan
        self.sig = sig

    def __repr__(self) -> str:
        return "Candidate(%s)" % (self.text or "<built-in>")


class Space:
    """The candidates of one problem."""

    def __init__(self, problem: Problem, schema: Schema, uarch: Any,
                 keys: Optional[Sequence[str]] = None, values: Optional[Dict[str, List[Any]]] = None,
                 allow_bit_changing: bool = False, keep_threads: bool = False,
                 base: Optional[Dict[str, Any]] = None):
        self.p = problem
        self.schema = schema
        self.uarch = uarch
        self.allow_bit_changing = allow_bit_changing
        # Settings under every candidate's own (tune.py calibrate): the built-in plan stays the
        # reference, and the base alone is a candidate the search proposes first.
        self.base = canonical(base or {}, schema, problem.dtype)
        # Leave out plans with a smaller team than the built-in plan's: a call on
        # fewer threads changes the thread team the calls before and after it see.
        self.keep_threads = keep_threads
        keys = list(keys) if keys is not None else search_keys(schema, problem.dtype, allow_bit_changing)
        self.choices: Dict[str, List[Any]] = {}
        for k in keys:
            spec = schema.key(k)
            if not spec.applies_to(problem.dtype):
                continue
            # A key the INT8 scheme of the problem does not apply is dropped by the library.
            if problem.int8 and problem.int8_scheme not in spec.int8_schemes:
                continue
            if not spec.same_bits(problem.dtype) and not allow_bit_changing:
                raise ValueError("key %s changes %s output bits; it needs --allow-bit-changing" % (k, problem.dtype))
            vals = list(values[k]) if values and k in values else value_choices(k, problem, schema)
            if vals:
                self.choices[k] = vals
        self.keys = list(self.choices)
        plan = builtin_plan(problem, uarch, schema)
        self.default = Candidate({}, "", plan, effective_signature(plan, problem))
        made = self.make({}) if self.base else None
        self.base_candidate = made if made is not None and made.sig != self.default.sig else None

    def make(self, config: Dict[str, Any]) -> Optional[Candidate]:
        """The candidate of config over the base, or None when it is illegal or bit-changing."""
        # A base key config sets back to its built-in value stays in the text: from_text()
        # reads a text over the base.
        cfg = canonical(dict(self.base, **config), self.schema, self.p.dtype,
                        keep=[k for k in self.base if k in config])
        if check_config(cfg, self.schema):
            return None
        if not self.allow_bit_changing and any(not self.schema.key_map[k].same_bits(self.p.dtype) for k in cfg):
            return None
        if not cfg:
            return self.default
        r = resolve(self.p, cfg, self.uarch, self.schema)
        if not r.applied or (self.keep_threads and r.plan.threads < self.default.plan.threads):
            return None
        return Candidate(cfg, format_settings(cfg, self.schema), r.plan, effective_signature(r.plan, self.p))

    def from_text(self, text: str) -> Optional[Candidate]:
        return self.make(parse_settings(text, self.schema)) if text else self.default

    def size(self) -> int:
        """Configs of the grid before legality and de-duplication."""
        n = 1
        for v in self.choices.values():
            n *= len(v) + 1
        return n

    def iter_grid(self, max_keys: Optional[int] = None) -> Iterator[Candidate]:
        """Legal candidates with 1, then 2, ... changed keys (lazily, in key order), after the
        base candidate."""
        if self.base_candidate is not None:
            yield self.base_candidate
        top = len(self.keys) if max_keys is None else min(max_keys, len(self.keys))
        for level in range(1, top + 1):
            for combo in itertools.combinations(self.keys, level):
                for vals in itertools.product(*(self.choices[k] for k in combo)):
                    c = self.make(dict(zip(combo, vals)))
                    if c is not None:
                        yield c

    def neighbors(self, cand: Candidate) -> Iterator[Candidate]:
        """Legal candidates one key away from cand (a key changed, set or reset)."""
        for k in self.keys:
            options: List[Any] = list(self.choices[k])
            if k in cand.config:
                options = [v for v in options if v != cand.config[k]] + [None]
            for v in options:
                cfg = dict(cand.config)
                if v is None:
                    cfg.pop(k, None)
                else:
                    cfg[k] = v
                c = self.make(cfg)
                if c is not None:
                    yield c

    def sample(self, rng: random.Random, p_change: Optional[float] = None) -> Optional[Candidate]:
        """A random config (each key changed with probability p_change), or None if illegal."""
        if not self.keys:
            return None
        pc = p_change if p_change is not None else min(0.5, 2.5 / len(self.keys))
        cfg = {k: rng.choice(self.choices[k]) for k in self.keys if rng.random() < pc}
        if not cfg:
            k = rng.choice(self.keys)
            cfg = {k: rng.choice(self.choices[k])}
        return self.make(cfg)


class Deduper:
    """Signatures seen for one problem; the built-in plan counts as seen."""

    def __init__(self, space: Space):
        self.seen: Dict[Tuple, Candidate] = {space.default.sig: space.default}

    def add(self, cand: Candidate) -> bool:
        if cand.sig in self.seen:
            return False
        self.seen[cand.sig] = cand
        return True

    def __contains__(self, cand: Candidate) -> bool:
        return cand.sig in self.seen


def unique(cands: Iterable[Candidate], space: Space) -> List[Candidate]:
    """cands without duplicates of each other or of the built-in plan, in order."""
    d = Deduper(space)
    return [c for c in cands if d.add(c)]
