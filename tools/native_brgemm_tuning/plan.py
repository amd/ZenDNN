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
"""The planner's built-in plan, plan validation and plan identity.

heuristic_plan() is the library's heuristic_plan (native_planner.cpp) line for
line, integer arithmetic as in C++: rules_plan() (the tile grid, the rules
behind makespan, l1_pass, traffic_groups, bf16_alias, bf16_narrow_n and blocks,
the F16_PH tiles and blocks of K, the INT8 and INT4 tile fields), then with
cost_model the model's choice among the rules' variants (model_plan for BF16
and F16, int8_model_plan for per-channel INT8, int4_model_plan and
int4_group_passes for s4) priced by cost_model.py.  validate_plan() is
validate_native_plan, and resolve() applies one override the way
native_planner_t does with a single source: the merged plan when it is legal,
else the built-in plan.  The built-in plan is heuristic_plan with every key at
the schema's default (builtin_params); a rule whose switch the schema lacks
is one its build does not have.  The kernel registry comes from the schema.
The tuner uses them to drop illegal candidates before a pod sees them and to
merge candidates that resolve to the same plan; every job also reads the
plan the library logged and records any difference (compare_with_log).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import gcd
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from . import cost_model as cm
from .cost_model import ceil_div, int4_pass_quads, int8_call_rows, int8_col_width, int8_pass_quads
from .problems import FORMATS, Problem
from .schema import route_scheme

# heuristic_params_t{}: every key of the planner at its built-in value; a pin at 0
# (-1 for col_groups) is auto, a rule switch at 0 leaves its rule out.
BUILTIN_PARAMS: Dict[str, Any] = {
    "nr": 0, "mr": 0, "mb": 0, "nb": 0, "kc": 0, "col_groups": -1,
    "schedule": "auto", "traversal": "m_inner", "unit_tiles": 0, "threads_cap": 0,
    "min_tiles_per_thread": 1, "row_unit": 12, "tile_rows": 6, "tall_rows": 12,
    "block_rows": 192, "min_tile_cols": 16, "row_split": True, "min_pass_k": 256,
    "bf16_l2_pct": 50, "kc_split": "budget", "l3_pct": 100, "domain_threads": 0,
    "int8_mr": 12, "int8_cw": [32, 16], "int8_l2_pct": 50, "int8_kc": 0, "int4_l1_pct": 125,
    "direct_store": True, "bias_in_dequant": True, "copy_block": 64,
    "bf16_copy_par_min": 0, "int8_copy_par_min": 65536,
    "makespan": 1, "min_gain_pct": 5, "idle_pct": 10, "decode_m": 128,
    "l1_pass": 1, "l1_pct": 100, "l1_pass_max_mb": 64, "traffic_groups": 1,
    "bf16_alias": 1, "bf16_alias_mr": 8, "bf16_narrow_n": 1, "bf16_narrow_n_min_m": 1024,
    "f16_wide_min_m": 192, "blocks": 1, "bf16_blocks_min_m": 1024, "int8_blocks_min_m": 3072,
    "cost_model": 1, "cm_gain_pct": 5, "cm_short_rows": 8, "cm_fma_pipes": 0, "cm_fma_lat": 6,
    "cm_call_cycles": 800, "cm_tall_pct": -1, "cm_l2_bytes": 16, "cm_l3_cycles": 10,
    "cm_mem_cycles": 0, "cm_prefetch_kib": 4, "cm_overlap_pct": 20, "cm_reuse_pct": 50,
    "cm_stream_cycles": -1, "cm_sync_tiles": 5, "cm_group_slab_pct": 75, "cm_group_l3_pct": 75,
    "cm_group_skew_pct": 110, "cm_group_rows_pct": 115, "cm_drift_kib": 4096, "cm_pass_kib": 128,
    "cm_pass_calls": 4, "cm_block_kib": 16384, "cm_block_k": 2048, "cm_int8_l1_pct": 0, "cm_int4_min_m": 48,
}
AUTO_PINS = ("nr", "mr", "mb", "nb", "kc", "col_groups", "unit_tiles", "threads_cap", "domain_threads", "int8_kc")
# Keys whose built-in value a rule or the hardware decides (an auto value in the schema).
AUTO_KEYS = AUTO_PINS + ("row_unit", "cm_fma_pipes", "cm_tall_pct", "cm_mem_cycles", "cm_stream_cycles")
RULE_SWITCHES = ("makespan", "l1_pass", "traffic_groups", "bf16_alias", "bf16_narrow_n", "blocks", "cost_model")
# Keys whose rule a build without them does not have: their value that leaves it out.
RULE_OFF_VALUES = {"int4_l1_pct": 0, "cm_int4_min_m": 0, "f16_wide_min_m": (1 << 31) - 1, "cm_block_k": 0}
# row_unit has no auto value in the planner: a unit other than this one is a pin.
BUILTIN_ROW_UNIT = 12
# Column tiles per unit of the blocks schedule without a pin.
BLOCKS_TILES = 1
CW_BITS = {16: 1, 32: 2}
# Rows per call of the tallest kernels.
MAX_CALL_ROWS = 12
# The widest tile a kernel computes: a wider pack takes tiles of this width.
MAX_TILE_COLS = 32
# Steps of K an F16_PH kernel accumulates in f16 before it adds the block in f32.
F16_ACCUM_BLOCK_K = 256
# Source rows this many bytes apart share an L1 set.
ALIAS_BYTES = 4096
# Row splits per tile width the makespan rule weighs.
MAX_SPLITS = 6
# Row blocks the cost model weighs for a single s4 row block: 2 to this.
INT4_ROW_BLOCKS = 4
INT64_MAX = (1 << 63) - 1


def cw_mask(widths: List[int]) -> int:
    m = 0
    for w in widths:
        m |= CW_BITS[w]
    return m


def cw_widths(mask: int) -> List[int]:
    return [w for w in (32, 16) if mask & CW_BITS[w]]


def mirrored(name: str) -> bool:
    """A key heuristic_plan reads.  A key of the schema it does not read (one of a
    newer build) has plans the tuner cannot vouch for."""
    return name in BUILTIN_PARAMS


def builtin_params(schema: Any) -> Dict[str, Any]:
    """heuristic_params_t{} of the schema's build: each key at the schema's default
    or auto value (BUILTIN_PARAMS for a key the schema lacks)."""
    hp = dict(BUILTIN_PARAMS)
    for k in schema.keys:
        v = k.default if k.default is not None else k.auto
        if v is not None:
            hp[k.name] = v
    # A rule whose switch or bound the schema lacks is one the build does not have.
    for s in RULE_SWITCHES:
        if s not in schema.key_map:
            hp[s] = 0
    for s, off in RULE_OFF_VALUES.items():
        if s not in schema.key_map:
            hp[s] = off
    return hp


def params_from_config(config: Dict[str, Any], schema: Any) -> Dict[str, Any]:
    """apply_native_config(heuristic_params_t{}, config) of the schema's build."""
    hp = builtin_params(schema)
    for k, v in config.items():
        if k not in hp:
            raise KeyError("unknown config key %r" % (k,))
        hp[k] = v
    return hp


@dataclass
class Plan:
    """native_plan_t without provenance."""

    nr: int = 0
    mr: int = 0
    mb: int = 0
    m_blocks: int = 0
    n_tiles: int = 0
    kc: int = 0
    # The kernels of the tiles: bf16, int8, int4, f32, f16 or f16_ph.
    kind: str = "bf16"
    threads: int = 1
    col_groups: int = 0
    # > 0: the blocks schedule (col_groups 0), units of a row block by unit_tiles
    # column tiles dealt per L3 domain of domain_threads workers.
    unit_tiles: int = 0
    domain_threads: int = 1
    row_unit: int = 12
    traversal: str = "m_inner"
    scratch_floats: int = 0
    # The tiles pair a gated call's halves at a width above the plan's (paired_plan): the plan's
    # INT8 K pass is one call's.
    paired_wider: bool = False
    int8_rows_nv1: int = 12
    int8_rows_nv2: int = 12
    int8_cw_mask: int = 3
    int8_pass_budget: int = 0
    int8_pass_quads: int = 0
    int8_direct_store: bool = True
    int8_bias_in_dequant: bool = True
    int8_scheme: str = "per_channel"
    copy_block: int = 64
    copy_par_min: int = 0

    @property
    def tiles(self) -> int:
        return self.m_blocks * self.n_tiles

    def rows(self, nv: int) -> int:
        """native_int8_call_rows."""
        return self.int8_rows_nv2 if nv == 2 else self.int8_rows_nv1

    def col_width(self, remaining: int) -> int:
        """native_int8_col_width."""
        return int8_col_width(self, remaining)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Grid(NamedTuple):
    """A tile grid of the makespan rule."""

    nr: int
    mr: int
    mb: int
    m_blocks: int
    n_tiles: int


def grid_of(q: Plan) -> Grid:
    return Grid(q.nr, q.mr, q.mb, q.m_blocks, q.n_tiles)


class Rules(NamedTuple):
    """The rules a parameter set turns on."""

    makespan: bool
    l1_pass: bool
    traffic: bool
    bf16_alias: bool
    bf16_narrow_n: bool
    blocks: bool


def rule_switches(hp: Dict[str, Any]) -> Rules:
    """The rules hp turns on: each one whose switch is not 0."""
    return Rules(*(hp[s] != 0 for s in RULE_SWITCHES[:6]))


def call_rows(hp: Dict[str, Any], int8: bool, nr: int) -> int:
    """Rows per call: the pin, else tall_rows for a non-INT8 tile of 32 columns or more, else tile_rows."""
    if hp["mr"] > 0:
        return hp["mr"]
    return hp["tall_rows"] if not int8 and nr >= 32 else hp["tile_rows"]


def kernel_kind(p: Problem, schema: Any) -> str:
    """The family of p's route, whose kernels its tiles run (the planner's kernel_kind)."""
    return schema.route(p).family


def slab_tile_width(kind: str, nr: int, pack_nr: int) -> bool:
    """A tile of 64 or 128 columns of several slabs of kind's kernels (native_slab_cols):
    32-column slabs, each inside one pack block (the kernel reads a slab with the block's
    stride), of whole pack blocks or of one block; F16_PH slabs of 64 columns, each two pack
    blocks of 32 as the F16_PH tile of 64."""
    slab = cm.SLAB_COLS.get(kind, 32)
    if nr not in (64, cm.MAX_SLAB_TILE_COLS) or nr <= slab:
        return False
    if kind == "f16_ph":
        return pack_nr == MAX_TILE_COLS
    return pack_nr > 0 and pack_nr % slab == 0 and (nr % pack_nr == 0 or pack_nr % nr == 0)


def widest_tile(p: Problem, kind: str, hp: Dict[str, Any]) -> int:
    """The widest tile without a pinned nr: 32 columns, or 64 on the F16_PH kernels from
    f16_wide_min_m rows where N is whole tiles of 64."""
    if kind == "f16_ph" and p.m >= hp["f16_wide_min_m"] and p.pack_nr == MAX_TILE_COLS and p.plan_n % 64 == 0:
        return 2 * MAX_TILE_COLS
    return min(p.pack_nr, MAX_TILE_COLS)


def gain_bound(work: int, gain_pct: int) -> int:
    """What a candidate's work must stay below to replace a grid of this work:
    keep percent of it, rounded up, with keep = 100 - gain_pct."""
    keep = 100 - min(max(gain_pct, 0), 100)
    return work // 100 * keep + (work % 100 * keep + 99) // 100


def tile_work(nr: int, rows: int, nbytes: int) -> int:
    """A tile's work per element of K: with bytes its bytes (B slab and A rows, a row of A
    nbytes units to a column of B), else its multiply-adds."""
    return nr + rows * nbytes if nbytes else nr * rows


def slice_makespan(g: Grid, m: int, workers: int, nbytes: int, bound: int = INT64_MAX) -> int:
    """The work of the busiest worker when workers threads take the tiles of g
    in slices of the m_inner order; past bound only whether it stays below
    bound counts."""
    tiles = g.m_blocks * g.n_tiles
    if tiles <= 0:
        return 0
    full = tile_work(g.nr, g.mb, nbytes)
    per, extra = tiles // workers, tiles % workers
    most = per + (1 if extra > 0 else 0)
    last = m - (g.m_blocks - 1) * g.mb
    if last >= g.mb:
        return most * full
    mm = g.m_blocks
    lighter = full - tile_work(g.nr, last, nbytes)
    if most * full - (most + mm - 1) // mm * lighter >= bound:
        return bound
    q0, r0, q1, r1 = per // mm, per % mm, (per + 1) // mm, (per + 1) % mm
    busiest = err = phase = 0
    for _ in range(workers):
        err += extra
        longer = err >= workers
        if longer:
            err -= workers
        phase += r1 if longer else r0
        lasts = q1 if longer else q0
        if phase >= mm:
            phase -= mm
            lasts += 1
        busiest = max(busiest, (per + 1 if longer else per) * full - lasts * lighter)
    return busiest


def makespan_grid(p: Problem, hp: Dict[str, Any], threads: int, base: Grid, schema: Any) -> Grid:
    """The grid of the makespan rule, from base, the grid of the narrowing and
    the row split.  Per tile width, the pinned nr, else the widest tile down by
    halves to the width of base: the M block before the row split and, with the
    row split, the most row blocks that take each further round of the team.  A
    candidate replaces base only below min_gain_pct percent less than base's
    busiest worker's work; the earlier candidate wins a tie."""
    m, n = p.m, p.plan_n
    int8 = p.int8
    nbytes = p.row_byte_units if m <= hp["decode_m"] else 0
    workers = max(1, threads)
    if base.n_tiles <= 0 or base.m_blocks <= 0:
        return base
    best = base
    best_work = gain_bound(slice_makespan(base, m, workers, nbytes), hp["min_gain_pct"])
    kind = kernel_kind(p, schema)
    widest = hp["nr"] if hp["nr"] > 0 else widest_tile(p, kind, hp)
    if widest <= 0:
        return base
    tiles = n // widest
    least = tiles * (widest + m * nbytes) if nbytes else tiles * widest * m
    if (least + workers - 1) // workers >= best_work:
        return base
    nr = widest
    while True:
        n_tiles = n // nr if nr > 0 else 0
        mr = call_rows(hp, int8, nr)
        # A BF16, FP32 or F16 width counts only with kernels for its rows per call.
        served = int8 or schema.covers(kind, mr, nr)
        mb0 = max(1, min(m, hp["mb"] if hp["mb"] > 0 else hp["block_rows"]))
        mbs = [mb0]
        if hp["mb"] == 0 and hp["row_split"] and n_tiles > 0 and mr > 0 and mb0 > mr:
            blocks = ceil_div(m, mb0)
            prev = mb0
            splits = i = 0
            while i < 4 * MAX_SPLITS and splits < MAX_SPLITS and prev > mr and blocks < m:
                rounds = ((blocks + 1) * n_tiles + workers - 1) // workers
                blocks = min(m, rounds * workers // n_tiles)
                mb = max(mr, ceil_div(ceil_div(m, blocks), mr) * mr)
                if mb < prev:
                    mbs.append(mb)
                    prev = mb
                    splits += 1
                i += 1
        for mb in mbs:
            g = Grid(nr, mr, mb, ceil_div(m, mb), n_tiles)
            if not served or g.n_tiles <= 0 or g.m_blocks <= 0:
                continue
            # A pinned col_groups must still divide the column tiles and the team.
            if hp["col_groups"] > 0:
                team = min(workers, g.m_blocks * g.n_tiles // max(1, hp["min_tiles_per_thread"]))
                if g.n_tiles % hp["col_groups"] or max(1, team) % hp["col_groups"]:
                    continue
            work = slice_makespan(g, m, workers, nbytes, best_work)
            if work < best_work:
                best, best_work = g, work
        if hp["nr"] > 0 or nr <= hp["min_tile_cols"] or nr <= base.nr:
            break
        nr //= 2
    return best


def grid_work(q: Plan, m: int, nbytes: int) -> int:
    """The work of all tiles of the plan's grid, per element of K."""
    last = m - (q.m_blocks - 1) * q.mb
    return q.n_tiles * ((q.m_blocks - 1) * tile_work(q.nr, q.mb, nbytes) + tile_work(q.nr, last, nbytes))


def row_group_work(q: Plan, g: int, row_units: int, m: int, nbytes: int) -> int:
    """The work of the busiest worker of g column groups: the column tiles of its row group
    over the most rows a group takes."""
    unit = max(1, q.row_unit)
    groups = q.threads // g
    units = (row_units + groups - 1) // groups
    rows = min(m, units * unit)
    return (q.n_tiles // g) * tile_work(q.nr, rows, nbytes)


def idle_col_groups(q: Plan, m: int, dt: int, nbytes: int, bound: int) -> int:
    """The row groups whose busiest worker has less work than bound, the most
    column groups on a tie: col_groups, or 0 when none.  Column groups divide
    the column tiles, the team and the L3 domain."""
    top = gcd(gcd(q.n_tiles, q.threads), dt)
    unit = max(1, q.row_unit)
    row_units = (m + unit - 1) // unit
    best, best_work = 0, bound
    for g in range(top, 0, -1):
        if top % g:
            continue
        work = row_group_work(q, g, row_units, m, nbytes)
        if work < best_work:
            best, best_work = g, work
    return best


def busiest_work(q: Plan, m: int, nbytes: int) -> int:
    """The work of the plan's busiest worker, per element of K: its slice, or the
    column tiles of its row group over the most rows a group takes."""
    g = q.col_groups
    if g <= 0 or q.threads % g or q.n_tiles % g:
        return slice_makespan(grid_of(q), m, max(1, q.threads), nbytes)
    unit = max(1, q.row_unit)
    return row_group_work(q, g, (m + unit - 1) // unit, m, nbytes)


def traffic_col_groups(q: Plan, m: int, n: int, k: int, elem: int, wei_elem: float, dt: int,
                       l3_share: int) -> int:
    """col_groups of the traffic rule, or 0 for the slice: when the source and
    the B slabs of a domain's threads exceed the domain's share of L3, the
    schedule whose L3 domains read fewer source and weight bytes from memory.
    Floating point in the library's order of operations."""
    workers = q.threads
    g = gcd(gcd(q.n_tiles, workers), dt)
    if q.m_blocks <= 1 or workers <= 1 or g < 1:
        return 0
    de = min(dt, workers)
    esz = float(elem)
    a = float(m) * k * esz
    b = float(n) * k * wei_elem
    slabs = float(de) * q.nr * k * wei_elem
    share = float(l3_share)
    if a + slabs <= share:
        return 0
    tiles = q.m_blocks * q.n_tiles
    sweeps = float(tiles) / workers / q.m_blocks
    # Distinct row blocks the slices of the first domain start at; past 64
    # threads per domain, as many as there can be.
    starts = min(de, q.m_blocks)
    if de <= 64:
        starts = len({tiles * w // workers % q.m_blocks for w in range(de)})
    slice_bytes = a * sweeps * starts + b * de / workers
    rows = a * de / workers
    rg_a = rows if rows + slabs <= share else rows * (q.n_tiles // g)
    return g if rg_a + b * de / g < slice_bytes else 0


def rules_plan(p: Problem, uarch: Any, hp: Dict[str, Any], schema: Any) -> Plan:
    """rules_plan(p, uarch, params): the plan of the rules."""
    if uarch.plan_domain is None:
        raise ValueError("the L3 domain is unknown: pass it (e.g. --uarch family_l3_cores=8)")
    # The tiles of a weight of N not a multiple of its blocks run on its padded width.
    m, n, k = p.m, p.plan_n, p.k
    # The float routes take the rules of BF16 with their own element size, but not the BF16
    # tiles of several slabs.
    route = schema.route(p)
    int8 = route.cls != "rows"
    int4 = route.family == "int4"
    esz, wei_esz = p.esz, p.wei_esz
    threads = min(p.threads, hp["threads_cap"]) if hp["threads_cap"] > 0 else p.threads
    on = rule_switches(hp)
    nbytes = p.row_byte_units if m <= hp["decode_m"] else 0
    q = Plan()
    kind = route.family
    q.kind = kind
    f16_ph = q.kind == "f16_ph"
    q.nr = hp["nr"] if hp["nr"] > 0 else widest_tile(p, q.kind, hp)
    q.mb = max(1, min(m, hp["mb"] if hp["mb"] > 0 else hp["block_rows"]))

    def count() -> int:
        q.m_blocks = ceil_div(m, q.mb)
        q.n_tiles = n // q.nr if q.nr > 0 else 0
        return q.tiles

    if hp["nr"] > 0:
        count()
    else:
        # A pinned BF16, FP32 or F16 mr stops at the narrowest width with its kernels.
        while (count() < threads and q.m_blocks == 1 and q.nr > hp["min_tile_cols"]
               and (hp["mr"] == 0 or int8 or schema.covers(kind, hp["mr"], q.nr // 2))):
            q.nr //= 2
    q.mr = call_rows(hp, int8, q.nr)
    if (hp["mb"] == 0 and hp["row_split"] and count() < threads and q.n_tiles > 0 and q.mr > 0
            and q.mb > q.mr):
        blocks = ceil_div(threads, q.n_tiles)
        q.mb = max(q.mr, ceil_div(ceil_div(m, blocks), q.mr) * q.mr)
        count()
    if on.makespan:
        g = makespan_grid(p, hp, threads, grid_of(q), schema)
        q.nr, q.mr, q.mb = g.nr, g.mr, g.mb
        count()
    # Source rows a multiple of 4 KiB apart map the rows of one call to one L1
    # set: a tile of several calls takes fewer rows per call.  The widened rows
    # of an f16 source the F16 kernels read are never a multiple of 4 KiB apart.
    if (not int8 and q.kind != "f16" and on.bf16_alias and hp["mr"] == 0 and k % (ALIAS_BYTES // esz) == 0
            and m > hp["decode_m"] and q.nr >= 32 and q.mb > hp["bf16_alias_mr"]):
        q.mr = min(q.mr, hp["bf16_alias_mr"])
    q.threads = max(1, min(threads, q.tiles // max(1, hp["min_tiles_per_thread"])))

    q.kc = k
    if hp["kc"] > 0:
        q.kc = min(hp["kc"], k)
    elif not int8 and q.nr > 0 and q.mb > q.mr:
        kc = uarch.l2_bytes * hp["bf16_l2_pct"] // 100 // (q.nr * esz)
        if kc < k:
            q.kc = max(max(2, hp["min_pass_k"]), kc & ~1)
        if hp["kc_split"] == "equal" and q.kc < k:
            q.kc = (ceil_div(k, ceil_div(k, q.kc)) + 1) & ~1
        if on.l1_pass and q.mr > 0 and 2 * q.mr <= q.mb <= hp["l1_pass_max_mb"]:
            l1 = uarch.l1d_bytes * hp["l1_pct"] // 100 // ((q.nr + q.mr) * esz)
            k_pass = max(max(2, hp["min_pass_k"]), min(kc, l1)) & ~1
            if k_pass < k:
                q.kc = (ceil_div(k, ceil_div(k, k_pass)) + 1) & ~1
    # The F16_PH kernels accumulate blocks of K from the start of each call's K: K
    # passes of whole blocks keep every block at the same K whatever the plan.
    if f16_ph and q.kc < k:
        q.kc = max(F16_ACCUM_BLOCK_K, q.kc // F16_ACCUM_BLOCK_K * F16_ACCUM_BLOCK_K)

    dt = cm.domain_threads(uarch, hp)
    q.row_unit = hp["row_unit"]

    def take_blocks() -> None:
        q.col_groups = 0
        if hp["nb"] > 0:
            tiles = max(1, hp["nb"] // max(1, q.nr))
        else:
            tiles = hp["unit_tiles"] if hp["unit_tiles"] > 0 else BLOCKS_TILES
        q.unit_tiles = min(max(1, q.n_tiles), tiles)
        # Non-INT8 K passes keep the accumulators of every tile of a unit in the worker's
        # scratch: at most the tiles whose accumulators fit the bf16_l2_pct share of L2
        # that sizes the K pass, and at least one.
        if not int8 and q.kc < k and q.unit_tiles > 1:
            share = uarch.l2_bytes * hp["bf16_l2_pct"] // 100
            q.unit_tiles = max(1, min(q.unit_tiles, share // max(1, q.mb * q.nr * 4)))
        q.domain_threads = max(1, dt)

    narrow_blocks = False
    if hp["col_groups"] >= 0:
        q.col_groups = hp["col_groups"]
    elif hp["schedule"] == "row_groups":
        q.col_groups = gcd(gcd(q.n_tiles, q.threads), dt)
    elif hp["schedule"] == "blocks":
        take_blocks()
    elif hp["schedule"] == "auto":
        l3_share = uarch.l3_bytes * hp["l3_pct"] // 100
        if on.makespan and q.n_tiles > 0:
            work = slice_makespan(grid_of(q), m, q.threads, nbytes)
            team = float(threads) * float(work)
            used = float(grid_work(q, m, nbytes))
            if work > 0 and 100.0 * (team - used) > hp["idle_pct"] * team:
                q.col_groups = idle_col_groups(q, m, max(1, dt), nbytes, gain_bound(work, hp["min_gain_pct"]))
        # The L3 rules below need an L3 (l3_share above 0).
        if q.col_groups == 0 and on.traffic:
            if dt > 1 and l3_share > 0:
                q.col_groups = traffic_col_groups(q, m, n, k, esz, wei_esz, dt, l3_share)
        elif q.col_groups == 0:
            domain_bytes = m * k * esz + int(float(dt) * q.nr * k * wei_esz)
            if (q.m_blocks > 1 and q.n_tiles % q.threads != 0 and dt > 1 and l3_share > 0
                    and domain_bytes > l3_share):
                q.col_groups = gcd(gcd(q.n_tiles, q.threads), dt)
        # Each L3 domain's workers take the units of the domain's share of the row
        # blocks: where that share of the source fits the domain's L3 share, the
        # blocks schedule replaces the schedule of a non-INT8 plan whose B slab over
        # all of K exceeds the bf16_l2_pct share of L2, and the slice of an INT8 plan.
        domains = ceil_div(q.threads, max(1, dt))
        de = min(max(1, dt), q.threads)
        share_bytes = (m * de + q.threads - 1) // q.threads * k * esz
        slab_bytes = int(float(q.nr) * k * wei_esz)
        if (on.blocks and domains > 1 and m >= (hp["int8_blocks_min_m"] if int8 else hp["bf16_blocks_min_m"])
                and (q.col_groups == 0 if int8 else slab_bytes * 100 > uarch.l2_bytes * hp["bf16_l2_pct"])
                and share_bytes <= l3_share):
            take_blocks()
        # A non-INT8 slice of 32-column tiles with at most two column tiles per domain
        # takes the blocks schedule under the same bounds, unless bf16_narrow_n widens
        # the tiles of a BF16 plan (below).
        narrow_blocks = (on.blocks and not int8 and domains > 1 and m >= hp["bf16_blocks_min_m"]
                         and q.unit_tiles == 0 and q.col_groups == 0 and q.nr == MAX_TILE_COLS
                         and q.n_tiles <= 2 * domains and share_bytes <= l3_share)
    q.traversal = hp["traversal"]
    q.scratch_floats = scratch_floats(q, p)
    # An INT8 call takes the rows per call of its route's kernels.
    group_kernel = route.cls == "group"
    q.int8_rows_nv1 = min(hp["int8_mr"], schema.route_max_mr(route, 1))
    q.int8_rows_nv2 = min(hp["int8_mr"], schema.route_max_mr(route, 2))
    q.int8_cw_mask = cw_mask(hp["int8_cw"])
    q.int8_pass_budget = uarch.l2_bytes * hp["int8_l2_pct"] // 100
    # The later row calls of an s4 tile read each batch of its groups again: in a
    # plan of at most decode_m rows, batches of the int4_l1_pct share of L1.
    if int4 and hp["int4_l1_pct"] > 0 and m <= hp["decode_m"]:
        q.int8_pass_budget = min(q.int8_pass_budget, uarch.l1d_bytes * hp["int4_l1_pct"] // 100)
    q.int8_pass_quads = min(hp["int8_kc"] // 4, ceil_div(k, 4)) if hp["int8_kc"] > 0 else 0
    # The INT8 form of the L1 pass, at the tile's pass width and rows per call;
    # equal passes in whole quads.  Only the per-channel tile reads it.
    if int8 and not group_kernel and hp["int8_kc"] == 0 and on.l1_pass and q.nr > 0:
        cw = int8_col_width(q, q.nr)
        rows = int8_call_rows(q, cw // 16)
        if rows > 0 and 2 * rows <= q.mb <= hp["l1_pass_max_mb"]:
            per_k = cm.int8_pass_cols(q) + rows
            l1 = uarch.l1d_bytes * hp["l1_pct"] // 100 // per_k
            quads = max(hp["min_pass_k"], min(l1, q.int8_pass_budget // per_k)) // 4
            k_quads = (k + 3) // 4
            if 1 <= quads < k_quads:
                q.int8_pass_quads = ceil_div(k_quads, ceil_div(k_quads, quads))
    q.int8_direct_store = bool(hp["direct_store"])
    q.int8_bias_in_dequant = bool(hp["bias_in_dequant"])
    q.int8_scheme = p.int8_scheme
    q.copy_block = hp["copy_block"]
    q.copy_par_min = hp["int8_copy_par_min"] if int8 else hp["bf16_copy_par_min"]
    # Fewer than two 32-column tiles per L3 domain in the slice: the widest BF16 tile
    # of whole slabs that is fewer than the domains, whose plan at that width gives the
    # busiest worker at most min_gain_pct percent more work, replaces the plan; it may
    # take the blocks schedule.  The domains are those of the planned team.
    team_domains = ceil_div(q.threads, max(1, dt))
    if (route.family == "bf16" and on.bf16_narrow_n and hp["nr"] == 0 and m >= hp["bf16_narrow_n_min_m"]
            and q.nr == MAX_TILE_COLS and q.col_groups == 0 and q.unit_tiles == 0
            and q.n_tiles < 2 * team_domains):
        work = busiest_work(q, m, nbytes)
        most = work + work * min(max(hp["min_gain_pct"], 0), 100) // 100
        nr = cm.MAX_SLAB_TILE_COLS
        while nr > q.nr:
            if n % nr == 0 and n // nr < team_domains and slab_tile_width("bf16", nr, p.pack_nr):
                w = rules_plan(p, uarch, dict(hp, nr=nr), schema)
                if busiest_work(w, m, nbytes) <= most and validate_plan(w, p, schema) is None:
                    return w
            nr //= 2
    if narrow_blocks:
        take_blocks()
        q.scratch_floats = scratch_floats(q, p)
    return q


def row_groups_split(m: int, unit: int, hp: Dict[str, Any]) -> bool:
    """Whether two row groups of whole units of unit rows, the first of ceil(M / unit) / 2
    units, each take more than cm_short_rows rows and the larger at most cm_group_rows_pct
    percent of half of M."""
    unit = max(1, unit)
    first = min(m, ceil_div(m, unit) // 2 * unit)
    return (min(first, m - first) > hp["cm_short_rows"]
            and max(first, m - first) * 200 <= m * hp["cm_group_rows_pct"])


def _gain_bound_cycles(base: float, hp: Dict[str, Any]) -> float:
    return base * (100 - min(max(hp["cm_gain_pct"], 0), 100)) / 100.0


def model_plan(p: Problem, uarch: Any, hp: Dict[str, Any], q: Plan, schema: Any) -> Plan:
    """The cost model's plan for the rules' plan q of a priced route of the rows class
    (model_plan)."""
    route = schema.route(p)
    if route.cls != "rows" or not route.priced or q.nr < MAX_TILE_COLS or q.mr < 1 or q.n_tiles < q.threads:
        return q
    m, k = p.m, p.k
    wei_esz = schema.formats[p.dtype].elem_bits // 8
    short_rows = (hp["mr"] == 0 and hp["cm_short_rows"] < hp["tall_rows"] and q.mr > hp["cm_short_rows"]
                  and q.mb > hp["cm_short_rows"] and (q.m_blocks == 1 or cm.tall_pct(uarch, hp) > 0))
    g = q.threads // 2
    split_rows = row_groups_split(m, q.row_unit, hp)
    short_unit = hp["row_unit"] if hp["row_unit"] != BUILTIN_ROW_UNIT else hp["cm_short_rows"]
    split_short = short_rows and row_groups_split(m, short_unit, hp)
    row_groups = (hp["col_groups"] < 0 and hp["schedule"] == "auto" and q.col_groups == 0
                  and q.unit_tiles == 0 and q.m_blocks == 1 and g >= 1 and q.threads % 2 == 0
                  and q.n_tiles % g == 0 and (split_rows or split_short) and cm.slice_skewed(q, hp)
                  and q.nr * k * wei_esz * 100 <= uarch.l2_bytes * hp["cm_group_slab_pct"]
                  and cm.groups_fit_l3(p, q, uarch, hp, schema))
    pass_k = (hp["cm_pass_kib"] * 1024 // (q.nr * wei_esz)) & ~1
    long_pass = (hp["kc"] == 0 and q.m_blocks == 1 and ceil_div(q.mb, q.mr) >= hp["cm_pass_calls"]
                 and pass_k >= 2 and pass_k > q.kc and pass_k < k)
    # Where one core streams slower than its share of the domain's link, the short
    # M blocks of a BF16 plan whose team fills each L3 domain it spans take only a K
    # above cm_block_k.
    block_k_bound = (route.family == "bf16" and cm.core_streams(uarch, hp)
                     and q.threads % cm.domain_threads(uarch, hp) == 0)
    short_blocks = (hp["mb"] == 0 and hp["kc"] == 0 and q.m_blocks == 1 and m > 2 * q.mr
                    and hp["tile_rows"] < q.mr
                    and (m + q.nr) * k * wei_esz * 100 <= uarch.l2_bytes * hp["bf16_l2_pct"]
                    and (k > hp["cm_block_k"] or not block_k_bound))
    if not short_rows and not row_groups and not long_pass and not short_blocks:
        return q
    base = cm.plan_cost(q, p, uarch, hp, schema).cycles
    if not base > 0:
        return q
    state = {"best": q, "bound": _gain_bound_cycles(base, hp)}

    def consider(c: Plan) -> bool:
        if validate_plan(c, p, schema) is not None:
            return False
        cycles = cm.plan_cost(c, p, uarch, hp, schema).cycles
        if not cycles < state["bound"]:
            return False
        state["best"] = c
        state["bound"] = cycles
        return True

    if short_rows:
        consider(rules_plan(p, uarch, dict(hp, tall_rows=hp["cm_short_rows"]), schema))
    took_groups = False
    if row_groups:
        groups = dict(hp, col_groups=g)
        if split_rows:
            took_groups = consider(rules_plan(p, uarch, groups, schema))
        if split_short:
            groups = dict(groups, tall_rows=hp["cm_short_rows"], row_unit=short_unit)
            if consider(rules_plan(p, uarch, groups, schema)):
                took_groups = True
    best = state["best"]
    drift = long_pass and cm.core_streams(uarch, hp)
    if (drift or short_blocks) and (not took_groups or split_rows):
        c = cm.plan_cost(best, p, uarch, hp, schema)
        r = dict(hp)
        if best.unit_tiles > 0:
            r["schedule"] = "blocks"
            r["unit_tiles"] = best.unit_tiles
            r["nb"] = 0
        else:
            r["col_groups"] = best.col_groups
        if drift and c.core > c.stream and c.stream_kib > hp["cm_drift_kib"]:
            r["kc"] = pass_k
        elif (short_blocks and best.col_groups == 0 and best.unit_tiles == 0 and c.core <= c.stream
              and c.stream_kib > hp["cm_block_kib"]):
            r["mb"] = hp["tile_rows"]
        else:
            return best
        s = rules_plan(p, uarch, r, schema)
        if validate_plan(s, p, schema) is None:
            return s
    return best


def int8_model_plan(p: Problem, uarch: Any, hp: Dict[str, Any], q: Plan, schema: Any) -> Plan:
    """The cost model's plan for the rules' INT8 plan q of the channel class (int8_model_plan)."""
    if not cm.cost_int8(p, schema) or q.nr < MAX_TILE_COLS or q.n_tiles < q.threads:
        return q
    cw = int8_col_width(q, q.nr)
    rows = int8_call_rows(q, cw // 16)
    if rows < 1 or hp["int8_mr"] < MAX_CALL_ROWS or q.m_blocks != 1:
        return q
    k = p.k
    g = q.threads // 2
    row_groups = (hp["col_groups"] < 0 and hp["schedule"] == "auto" and q.col_groups == 0
                  and q.unit_tiles == 0 and g >= 1 and q.threads % 2 == 0 and q.n_tiles % g == 0
                  and row_groups_split(p.m, q.row_unit, hp) and cm.slice_skewed(q, hp)
                  and q.nr * k * 100 <= uarch.l2_bytes * hp["cm_group_slab_pct"]
                  and cm.groups_fit_l3(p, q, uarch, hp, schema))
    if row_groups:
        base = cm.plan_cost(q, p, uarch, hp, schema).cycles
        c = rules_plan(p, uarch, dict(hp, col_groups=g), schema)
        if (base > 0 and validate_plan(c, p, schema) is None
                and cm.plan_cost(c, p, uarch, hp, schema).cycles < _gain_bound_cycles(base, hp)):
            return c
    rows_short = min(hp["tile_rows"], MAX_CALL_ROWS)
    if (hp["cm_int8_l1_pct"] > 0 and q.col_groups == 0 and q.unit_tiles == 0 and hp["int8_kc"] == 0
            and hp["mb"] == 0 and ceil_div(q.mb, rows) == 3 and rows_short < rows
            and cm.core_streams(uarch, hp)
            and (q.nr + rows_short) * k * 100 <= uarch.l2_bytes * hp["int8_l2_pct"]):
        v = dict(hp, int8_mr=rows_short, l1_pct=hp["cm_int8_l1_pct"], col_groups=0)
        r = rules_plan(p, uarch, v, schema)
        if r.int8_pass_quads > 0 and validate_plan(r, p, schema) is None:
            return r
    return q


def int4_model_plan(p: Problem, uarch: Any, hp: Dict[str, Any], q: Plan, schema: Any) -> Plan:
    """The cost model's plan for the rules' plan q of the INT4 kernels (int4_model_plan): M
    blocks of ceil(M / b) rows, b 2 to 4, for a single row block whose slice does not divide
    its column tiles over the threads.  The model's tile is that of the format's per-group
    route: the plans of a weight of one group are weighed at its rows per call and keep their
    own."""
    if (hp["cm_int4_min_m"] <= 0 or p.m < hp["cm_int4_min_m"] or hp["mb"] != 0 or hp["col_groups"] >= 0
            or hp["schedule"] != "auto" or q.nr < MAX_TILE_COLS or q.m_blocks != 1 or q.col_groups != 0
            or q.unit_tiles != 0 or q.n_tiles < q.threads or q.n_tiles % q.threads == 0):
        return q

    group = schema.route_of(p.dtype, "per_group")

    def cost(c: Plan) -> float:
        c = replace(c, int8_rows_nv1=min(c.int8_rows_nv1, schema.route_max_mr(group, 1)),
                    int8_rows_nv2=min(c.int8_rows_nv2, schema.route_max_mr(group, 2)))
        return cm.plan_cost(c, p, uarch, hp, schema).cycles

    base = cost(q)
    if not base > 0:
        return q
    best, bound = q, _gain_bound_cycles(base, hp)
    for b in range(2, INT4_ROW_BLOCKS + 1):
        mb = ceil_div(p.m, b)
        if mb < hp["tile_rows"]:
            break
        c = rules_plan(p, uarch, dict(hp, mb=mb), schema)
        if c.m_blocks < 2 or validate_plan(c, p, schema) is not None:
            continue
        cycles = cost(c)
        if cycles < bound:
            best, bound = c, cycles
    return best


def int4_group_passes(p: Problem, uarch: Any, hp: Dict[str, Any], q: Plan) -> Plan:
    """The cost model's pass budget for the s4 plan q of one group over K whose row blocks
    take K passes at its budget: the int8_l2_pct share of L2."""
    if p.group < p.k:
        return q
    cw = int8_col_width(q, q.nr)
    rows = int8_call_rows(q, cw // 16)
    quads = ceil_div(p.k, 4)
    if rows < 1 or int4_pass_quads(q, quads, cm.int8_pass_cols(q), q.mb, rows) >= quads:
        return q
    return replace(q, int8_pass_budget=max(q.int8_pass_budget, uarch.l2_bytes * hp["int8_l2_pct"] // 100))


def heuristic_plan(p: Problem, uarch: Any, hp: Dict[str, Any], schema: Any) -> Plan:
    """heuristic_plan(p, uarch, params): the rules' plan and, with cost_model, the model's choice."""
    q = rules_plan(p, uarch, hp, schema)
    if hp["cost_model"] == 0:
        return q
    route = schema.route(p)
    if route.family == "int4":
        return int4_group_passes(p, uarch, hp, int4_model_plan(p, uarch, hp, q, schema))
    if route.cls != "rows":
        return int8_model_plan(p, uarch, hp, q, schema)
    return model_plan(p, uarch, hp, q, schema)


def validate_plan(q: Plan, p: Problem, schema: Any) -> Optional[str]:
    """validate_native_plan: the first rule q breaks for p, or None."""
    narrow = q.nr in (16, 32) and q.nr <= p.pack_nr and p.pack_nr % q.nr == 0
    f16_ph_wide = q.kind == "f16_ph" and p.pack_nr == MAX_TILE_COLS and q.nr == 2 * MAX_TILE_COLS
    # A build whose registry has no tiles of slabs of a kind lists no such width.
    slabs = slab_tile_width(q.kind, q.nr, p.pack_nr) and schema.caps(q.kind, q.nr) is not None
    if not narrow and not f16_ph_wide and not slabs:
        return ("nr=%d is not 16 or 32 dividing pack_nr=%d, for F16_PH two blocks, or 64 or 128 of slabs of "
                "the kernels' widest call" % (q.nr, p.pack_nr))
    n = p.plan_n
    if (q.mb < 1 or q.mb > p.m or q.m_blocks != ceil_div(p.m, q.mb) or q.n_tiles < 1 or n % q.nr
            or q.n_tiles != n // q.nr):
        return "tile grid mb=%d n_tiles=%d does not cover M=%d N=%d" % (q.mb, q.n_tiles, p.m, n)
    if not 1 <= q.mr <= MAX_CALL_ROWS:
        return "mr=%d is outside 1 .. 12" % q.mr
    if p.bf16 and not schema.covers("bf16", q.mr, q.nr):
        return "no BF16 kernel for every row count 1 .. %d at nr %d" % (q.mr, q.nr)
    if p.bf16 and q.kc < p.k and (q.kc < 2 or q.kc % 2):
        return "kc=%d is below K and not an even number of at least 2" % q.kc
    if p.f32 and not schema.covers("f32", q.mr, q.nr):
        return "no FP32 kernel for every row count 1 .. %d at nr %d" % (q.mr, q.nr)
    f16_ph = q.kind == "f16_ph"
    if p.f16 and not f16_ph and not schema.covers("f16", q.mr, q.nr):
        return "no F16 kernel for every row count 1 .. %d at nr %d" % (q.mr, q.nr)
    if f16_ph and (not p.f16 or not schema.covers("f16_ph", q.mr, q.nr)):
        return "no F16_PH kernel for every row count 1 .. %d at nr %d" % (q.mr, q.nr)
    if (p.f32 or p.f16) and q.kc < 1:
        return "kc is below 1"
    if f16_ph and q.kc < p.k and q.kc % F16_ACCUM_BLOCK_K:
        return "kc=%d is below K and not a whole number of F16_PH blocks" % q.kc
    if not 1 <= q.threads <= p.threads:
        return "threads=%d is outside 1 .. %d" % (q.threads, p.threads)
    if q.col_groups < 0 or (q.col_groups > 0 and (q.threads % q.col_groups or q.n_tiles % q.col_groups)):
        return "col_groups=%d does not divide threads=%d and n_tiles=%d" % (q.col_groups, q.threads, q.n_tiles)
    if q.unit_tiles < 0 or q.unit_tiles > q.n_tiles or q.domain_threads < 1 or (q.unit_tiles > 0 and q.col_groups):
        return ("unit_tiles=%d is outside 0 .. n_tiles=%d, domain_threads=%d is below 1, or the blocks "
                "schedule has col_groups=%d" % (q.unit_tiles, q.n_tiles, q.domain_threads, q.col_groups))
    if q.row_unit < 1 or q.copy_block < 1 or q.copy_par_min < 0:
        return "row_unit, copy_block or copy_par_min out of range"
    if p.int8:
        route = schema.route(p)
        if not q.int8_cw_mask & 1:
            return "int8_cw does not allow 16"
        for nv in (1, 2):
            width = 16 * nv
            if width > q.nr or not q.int8_cw_mask & CW_BITS[width]:
                continue
            r = q.rows(nv)
            # A format of group scales only takes per-group calls.
            served = ((not schema.formats[p.dtype].grouped or p.int8_scheme == "per_group")
                      and schema.covers(route.family, r, width, route_scheme(route)))
            if not 1 <= r <= MAX_CALL_ROWS or not served:
                if route.family == "int4":
                    return ("no INT4 kernel for %d rows per call at width %d (the kernels of one group "
                            "over K, or the group kernels)" % (r, width))
                return "no INT8 kernel of the call's scheme for %d rows per call at width %d" % (r, width)
        if q.int8_pass_quads < 0 or q.int8_pass_budget < 0:
            return "int8_pass_quads or int8_pass_budget is negative"
    if q.scratch_floats != scratch_floats(q, p):
        return "scratch size mismatch"
    return None


def scratch_floats(q: Plan, p: Problem) -> int:
    """native_plan_scratch_floats: one row call's accumulators for a non-INT8 plan of one K
    pass, else a row block's for each tile a unit computes between K passes."""
    passes = not p.int8
    if passes and q.kc >= p.k:
        return q.mr * q.nr
    return q.mb * q.nr * (max(1, q.unit_tiles) if passes else 1)


def paired_plan(planned: Plan, p: Problem, width: int) -> Plan:
    """The plan the paired tiles of a gated call run at width (native_paired_plan)."""
    q = replace(planned)
    if width > planned.nr:
        q.nr = width
        q.paired_wider = True
        q.n_tiles = p.plan_n // q.nr
        if q.unit_tiles > 0:
            q.unit_tiles = max(1, q.unit_tiles * planned.nr // width)
        units = planned.n_tiles * planned.m_blocks
        if (q.unit_tiles == 0 and q.col_groups == 0 and q.n_tiles * q.m_blocks < q.threads
                and units > q.n_tiles * q.m_blocks):
            want = min(units, q.threads)
            m_blocks = min(p.m, ceil_div(want, q.n_tiles))
            q.mb = ceil_div(p.m, m_blocks)
            q.m_blocks = ceil_div(p.m, q.mb)
        q.scratch_floats = scratch_floats(q, p)
    q.scratch_floats = max(q.scratch_floats, q.mb * q.nr)
    return q


def paired_width(q: Plan, p: Problem) -> int:
    """native_paired_width: the width at which the tiles of a gated call pair its [gate | up] halves,
    from the plan's tiles, row blocks and team; 0 for another call, or for one that stages its rows
    instead (N not whole 32-column blocks, or halves not whole blocks of an M = 1 f32 or f16 call,
    whose GEMV kernels read a block's second 16 columns right after its first)."""
    if p.epilogue != "gated" or q.nr <= 0 or q.nr % 16 or p.n % 32:
        return 0
    if (p.f16 or p.f32) and gemv_kind(q, p) != "none" and p.n // 2 % 32:
        return 0
    if q.nr >= 64:
        return q.nr
    if p.n % 64 == 0:
        team = max(1, q.threads)
        tiles = p.n // 64 * q.m_blocks
        if tiles % team == 0 or tiles >= 8 * team:
            return 64
    return 32


def paired_log_fields(planned: Plan, p: Problem) -> Dict[str, Dict[str, str]]:
    """A gated call's log fields at the width its tiles pair its halves at (the line's pairing=<width>)
    when that is wider than the plan's; {} for other calls."""
    w = paired_width(planned, p)
    return {str(w): plan_log_fields(paired_plan(planned, p, w), p)} if w > planned.nr else {}


def log_pairing(planned: Plan, p: Problem) -> Optional[str]:
    """The pairing= of a gated call's line: the width its tiles pair its halves at, or staged where they
    stage its rows; None for another call, whose line names none."""
    if p.epilogue != "gated":
        return None
    w = paired_width(planned, p)
    return str(w) if w > 0 else "staged"


def compare_pairing(want: Optional[str], logged: Dict[str, str]) -> List[str]:
    """The line's pairing= against the expected one (log_pairing); a side without one is not compared."""
    got = logged.get("pairing")
    return ["pairing: model %s, log %s" % (want, got)] if want and got and got != want else []


@dataclass
class Resolution:
    plan: Plan
    applied: bool  # False: the override was dropped for this problem (built-in plan)
    reason: Optional[str] = None


def builtin_plan(p: Problem, uarch: Any, schema: Any) -> Plan:
    return heuristic_plan(p, uarch, builtin_params(schema), schema)


def scheme_config(config: Dict[str, Any], p: Problem, schema: Any) -> Dict[str, Any]:
    """native_config_for_scheme: config without the keys the problem's INT8 scheme does
    not apply (a problem of another weight type, or of per-channel scales, applies all)."""
    if not p.int8 or p.int8_scheme == "per_channel":
        return config
    out = {}
    for k, v in config.items():
        spec = schema.key_map.get(k)
        # The schema lists no INT8 scheme for a key of the other tiles; every scheme keeps it.
        if spec is None or spec.applies == "bf16" or p.int8_scheme in spec.int8_schemes:
            out[k] = v
    return out


def resolve(p: Problem, config: Dict[str, Any], uarch: Any, schema: Any) -> Resolution:
    """The plan of p with config as the only source (native_planner_t's fallback).
    A config that sets a key heuristic_plan does not read resolves to no plan the
    tuner can vouch for: it counts as dropped."""
    if not config:
        return Resolution(builtin_plan(p, uarch, schema), True)
    unread = [k for k in config if not mirrored(k)]
    if unread:
        return Resolution(builtin_plan(p, uarch, schema), False, "%s: not mirrored" % ", ".join(unread))
    scoped = scheme_config(config, p, schema)
    if not scoped:
        return Resolution(builtin_plan(p, uarch, schema), False,
                          "%s: not applied by %s calls" % (", ".join(config), p.int8_scheme))
    config = scoped
    q = heuristic_plan(p, uarch, params_from_config(config, schema), schema)
    why = validate_plan(q, p, schema)
    if why:
        return Resolution(builtin_plan(p, uarch, schema), False, why)
    return Resolution(q, True)


def int8_widths(q: Plan) -> List[int]:
    """The INT8 call widths of one nr-wide tile, left to right."""
    out, rem = [], q.nr
    while rem > 0:
        w = q.col_width(rem)
        out.append(w)
        rem -= w
    return out


def int8_turn_runs(q: Plan, p: Problem, paired: bool) -> List[Tuple[int, int, int]]:
    """(call width, calls, K pass columns) of each run of calls of the INT8 tiles of q that take each
    K pass (or batch of groups) and row call in turn (int8_block, native_int8_calls_in_turn): every
    call of a tile of several slabs, else one, in a tile the pairing widened (paired_wider), a tile of
    one slab and the tail tile (compute_int8_tail_tile: the columns past the last whole tile, rounded
    up to 16).  A 32-column call whose second 16 columns are not the next 16 of its pack block (on
    tiles paired at 32, the up run) takes the cache sets of twice its width (native_int8_channel_tile).
    Distinct runs, sorted."""
    def col(n0: int, nr: int, t: int) -> int:
        # native_weight_view_t::tile_col: a paired tile holds runs of 16 columns of each half.
        if not paired:
            return n0 + t
        half = nr // 2
        return n0 // 2 + t if t < half else p.n // 2 + n0 // 2 + t - half

    def runs(n0: int, nr: int, tail: bool) -> List[Tuple[int, int, int]]:
        calls = 1 if tail or q.paired_wider or nr <= cm.SLAB_COLS["int8"] else nr // q.col_width(nr)
        out, c = [], 0
        while c < nr:
            cw = q.col_width(nr - c)
            cols = 0
            for t in range(c, c + calls * cw, cw):
                p0 = col(n0, nr, t)
                p1 = col(n0, nr, t + 16) if paired and cw == 32 else p0 + 16
                cols += 2 * cw if cw == 32 and (p1 != p0 + 16 or p0 % p.pack_nr) else cw
            out.append((cw, calls, cols))
            c += calls * cw
        return out

    whole = p.n // q.nr
    # Which calls read apart repeats every 2 * pack_nr columns of the tiles.
    period = 2 * p.pack_nr // gcd(q.nr, 2 * p.pack_nr)
    found = {r for t in range(min(whole, period)) for r in runs(t * q.nr, q.nr, False)}
    if p.n % q.nr:
        found.update(runs(whole * q.nr, ceil_div(p.n % q.nr, 16) * 16, True))
    return sorted(found)


def effective_signature(q: Plan, p: Problem) -> Tuple:
    """What the call executes under q (a gated call: the plan its tiles pair its halves at): equal
    signatures run the same work.  The K passes of the per-channel INT8 tiles are sized over the
    columns of the calls that take them in turn, the tail tile's one call at a time
    (int8_turn_runs); the calls of a tile run in that order once a call takes more than one row
    call or K pass.

    Fields the call cannot observe for p are left out (traversal of a one-row or
    one-column grid or of the blocks schedule, row_unit of a slice or blocks
    schedule, call rows above the tile rows, the source copy without trans_a,
    the INT8 store / bias gates the epilogue closes, pass budgets of
    single-call tiles).  Unknown cases keep the raw field, so two different
    plans are never merged.
    """
    width = paired_width(q, p)
    if width > q.nr:
        q = paired_plan(q, p, width)
    row_groups = q.col_groups > 0
    sig: List[Any] = [q.kind, q.nr, q.mb, q.threads]
    if q.unit_tiles > 0:
        sig += ["blocks", q.unit_tiles, q.domain_threads]
    elif row_groups:
        sig += ["row_groups", q.col_groups, q.row_unit, q.traversal]
    else:
        sig += ["slice", q.traversal if q.m_blocks > 1 and q.n_tiles > 1 else "-"]
    if not p.int8:
        sig += [q.mr if row_groups else min(q.mr, q.mb), min(q.kc, p.k)]
        if p.trans_a:
            sig += [q.copy_block, p.m * p.k >= q.copy_par_min]
        return tuple(sig)
    widths = int8_widths(q)
    runs = int8_turn_runs(q, p, width > 0)
    used = sorted({w for w, _, _ in runs})
    rows_max = p.m if row_groups else q.mb
    sig += [tuple(widths), tuple(min(q.rows(w // 16), rows_max) for w in used)]
    k_quads = ceil_div(p.k, 4)
    multi_call = any(rows_max > q.rows(w // 16) for w in used)
    if p.int8_scheme == "per_channel":
        if q.int8_pass_quads > 0:
            sig.append(("quads", min(q.int8_pass_quads, k_quads)))
        elif row_groups:
            sig.append(("budget", q.int8_pass_budget) if multi_call else ("whole",))
        else:
            last = p.m - (q.m_blocks - 1) * q.mb
            sig.append(("passes",) + tuple(
                int8_pass_quads(q, k_quads, cols, r, q.rows(w // 16))
                for w, _, cols in runs for r in sorted({q.mb, last})))
    else:
        # The group tiles batch their groups by the pass budget only past one row call.
        sig.append(("budget", q.int8_pass_budget) if multi_call else ("whole",))
    if multi_call or (p.int8_scheme == "per_channel" and 0 < q.int8_pass_quads < k_quads):
        sig.append(("in_turn",) + tuple(calls for _, calls, _ in runs))
    if p.epilogue == "store" and p.dst in ("bf16", "f32", "f16"):
        sig.append(("direct", q.int8_direct_store))
    if p.has_bias and p.epilogue == "store" and p.int8_scheme == "per_channel":
        sig.append(("dequant_bias", q.int8_bias_in_dequant))
    sig += [q.copy_block, p.m * p.k >= q.copy_par_min]
    return tuple(sig)


def plan_features(q: Plan, p: Problem) -> Dict[str, float]:
    """Numeric description of a resolved plan for the surrogate model."""
    if p.int8:
        k_passes = ceil_div(ceil_div(p.k, 4), q.int8_pass_quads) if q.int8_pass_quads > 0 else 1
        widths = int8_widths(q)
    else:
        k_passes = ceil_div(p.k, max(1, min(q.kc, p.k)))
        widths = [q.nr]
    calls_per_tile = sum(ceil_div(min(q.mb, p.m), max(1, q.rows(w // 16) if p.int8 else q.mr)) for w in widths)
    per_thread = q.tiles / float(q.threads)
    return {
        "nr": float(q.nr), "mr": float(q.mr), "mb": float(q.mb), "m_blocks": float(q.m_blocks),
        "n_tiles": float(q.n_tiles), "threads_used": float(q.threads),
        "thread_frac": q.threads / float(p.threads),
        "tiles_per_thread": per_thread,
        "imbalance": ceil_div(q.tiles, q.threads) / per_thread if per_thread else 1.0,
        "k_passes": float(k_passes), "row_groups": float(q.col_groups > 0),
        "col_groups": float(q.col_groups), "unit_tiles": float(q.unit_tiles),
        "n_inner": float(q.traversal == "n_inner"),
        "row_unit": float(q.row_unit), "calls_per_tile": float(calls_per_tile),
        "int8_cw_mask": float(q.int8_cw_mask), "direct": float(q.int8_direct_store),
        "pass_budget": float(q.int8_pass_budget) if p.int8 else 0.0,
        "copy_block": float(q.copy_block), "copy_parallel": float(p.m * p.k >= q.copy_par_min),
    }


# The fields of the [MATMUL.NATIVE_BRGEMM] line the tuner compares with its plan: the
# plan's, and the GEMV kernels of the call (gemv).
LOG_PLAN_FIELDS = ("nr", "mb", "kc", "kernels", "tiles", "threads", "col_groups", "unit_tiles", "nb",
                   "domain_threads", "gemv", "mr", "row_unit", "traversal", "int8_rows", "int8_cw", "int8_scheme",
                   "gs", "int8_pass", "pass_budget", "copy_block")
# Fields the line prints only for some plans, and their value when it does not.
LOG_OPTIONAL_FIELDS = {"unit_tiles": "0"}
# Every field the line prints: the call, the plan, the call's choices that are not
# plan fields (gemv, direct, dequant_bias, wei, order, team, padded_n, pairing),
# its provenance (source, cfg) and transA.
LOG_LINE_FIELDS = ("M", "N", "K", "pack_nr", "nr", "mb", "kc", "kernels", "tiles", "threads", "col_groups",
                   "unit_tiles", "nb", "domain_threads", "transA", "gemv", "mr", "row_unit", "traversal",
                   "int8_rows", "int8_cw", "int8_scheme", "direct", "wei", "gs", "order", "pass_budget", "int8_pass",
                   "dequant_bias", "copy_block", "source", "cfg", "team", "padded_n", "pairing")


def _gemv_kinds() -> Tuple[str, ...]:
    """none, then the GEMV kernels' names of the routes, those of the float routes first, each
    once (the schema dump's plan_line_gemv)."""
    routes = [r for f in FORMATS.values() for r in f["routes"]]
    names = ["none"]
    for r in [r for r in routes if r["class"] == "rows"] + [r for r in routes if r["class"] != "rows"]:
        if r["gemv"] not in names:
            names.append(r["gemv"])
    return tuple(names)


# The values of the line's gemv field: the GEMV kernels an M = 1 call runs on, those of its
# route, and none for a call on the tiles.
GEMV_KINDS = _gemv_kinds()


def gemv_kind(q: Plan, p: Problem) -> str:
    """The GEMV kernels the call runs on (gemv_kind_of): every M = 1 call runs on those of
    its route, the route of its format on its plan's kernels that serves its INT8 scheme
    (the group and the one-group routes of a format of group scales only run kernels of one
    name); none for any other M.  A GEMV kernel takes all of K for each block of columns:
    the plan's tiles, M blocks and K passes do not apply to the call."""
    if p.m != 1:
        return "none"
    routes = [r for r in FORMATS[p.dtype]["routes"] if r["family"] == q.kind]
    return next((r for r in routes if p.int8_scheme in r["int8_schemes"]), routes[0])["gemv"]


def plan_log_fields(q: Plan, p: Problem) -> Dict[str, str]:
    """The fields of the API-log line this plan should print (log_plan_line) that the tuner
    compares: the plan's and the call's GEMV kernels."""
    out = {
        "nr": str(q.nr), "mb": str(q.mb), "tiles": str(q.tiles),
        "threads": str(q.threads), "col_groups": str(q.col_groups), "gemv": gemv_kind(q, p), "mr": str(q.mr),
        "row_unit": str(q.row_unit), "traversal": q.traversal, "copy_block": str(q.copy_block),
    }
    if not p.int8:
        out["kc"] = str(q.kc)
    if p.f16:
        out["kernels"] = "f16_ph" if q.kind == "f16_ph" else "f16"
    # The blocks schedule names the fields its deal reads: the unit width, in tiles and in
    # columns, and the threads of a domain.
    if q.unit_tiles > 0:
        out["unit_tiles"] = str(q.unit_tiles)
        out["nb"] = str(q.nr * q.unit_tiles)
        out["domain_threads"] = str(q.domain_threads)
    if p.int8:
        out["int8_rows"] = "%d/%d" % (q.int8_rows_nv1, q.int8_rows_nv2)
        out["int8_cw"] = "/".join(str(w) for w in cw_widths(q.int8_cw_mask))
        if p.int8_scheme != "per_channel":
            out["int8_scheme"] = p.int8_scheme
        if p.int4:
            out["gs"] = str(p.group)
            out["pass_budget"] = str(q.int8_pass_budget)
        elif p.int8_scheme == "per_channel":
            out["int8_pass"] = str(q.int8_pass_quads) if q.int8_pass_quads > 0 else "rule"
    return out


def compare_fields(want: Dict[str, str], logged: Dict[str, str], bf16: bool = False) -> List[str]:
    """Fields where a logged plan differs from the expected log fields; an optional field
    the line leaves out has its value when absent, and a field one side lacks is not
    compared (bf16 is kept for older callers)."""
    out = []
    for f in LOG_PLAN_FIELDS:
        absent = LOG_OPTIONAL_FIELDS.get(f)
        a, b = want.get(f, absent), logged.get(f, absent)
        if a is not None and b is not None and a != b:
            out.append("%s: model %s, log %s" % (f, a, b))
    return out


def compare_with_log(q: Plan, p: Problem, logged: Dict[str, str]) -> List[str]:
    """Fields where the library's logged plan differs from this model's plan.  A gated call
    whose tiles paired its halves logs the paired plan (pairing=<width>), at the model's width."""
    out = compare_pairing(log_pairing(q, p), logged)
    pairing = logged.get("pairing")
    if pairing is not None and pairing.isdigit():
        q = paired_plan(q, p, int(pairing))
    return out + compare_fields(plan_log_fields(q, p), logged)
