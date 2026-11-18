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
"""The planner's cost model (native_cost_model.cpp) line for line.

native_plan_cost() is the modeled time of a plan in cycles of its busiest
worker: kernel calls priced per K step, the weight stream of each L3 domain,
the idle of a stream behind the later calls of a K pass, and the second reads
of row groups from L3.  Every floating-point expression keeps the library's
order of operations (the library builds the model without contracted
multiply-adds), so the planner's comparisons of modeled times come out the
same here.
"""

from __future__ import annotations

import math
from typing import Any, Dict, NamedTuple

BF16_BYTES = 2.0
F16_WIDENED_BYTES = 4.0
VECTOR_COLS = 16
F16_PH_VECTOR_COLS = 32
MAX_DOMAINS = 64
INT8_BYTES = 1.0
INT8_STEP_K = 4
INT4_BYTES = 0.5
# Columns of one 2-vector BF16 kernel call of a wider tile.
BF16_SLAB_COLS = 32


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def bf16_slab_cols(nr: int) -> int:
    """native_bf16_slab_cols."""
    return nr if nr < BF16_SLAB_COLS else BF16_SLAB_COLS


def int8_col_width(q: Any, remaining: int) -> int:
    """native_int8_col_width: the widest allowed INT8 call width within remaining columns."""
    if q.int8_cw_mask & 2 and remaining >= 32:
        return 32
    return 16


def int8_call_rows(q: Any, nv: int) -> int:
    """native_int8_call_rows."""
    return q.int8_rows_nv2 if nv == 2 else q.int8_rows_nv1


def int8_pass_quads(q: Any, k_quads: int, cw: int, rows: int, max_mr: int) -> int:
    """native_int8_pass_quads: quads per K pass of a tile cw wide and rows tall."""
    if q.int8_pass_quads > 0:
        return min(q.int8_pass_quads, k_quads)
    quad_bytes = 4 * (cw + max_mr)
    budget = q.int8_pass_budget
    if rows <= max_mr or k_quads * quad_bytes <= budget:
        return k_quads
    per = max(1, budget // quad_bytes)
    passes = ceil_div(k_quads, per)
    return ceil_div(k_quads, passes)


def int4_pass_quads(q: Any, group_quads: int, cw: int, rows: int, max_mr: int) -> int:
    """native_int4_pass_quads: quads per K pass of the one group of an s4 tile."""
    quad_bytes = 2 * cw + 4 * max_mr
    budget = q.int8_pass_budget
    if rows <= max_mr or group_quads * quad_bytes <= budget:
        return group_quads
    per = max(2, budget // quad_bytes)
    passes = ceil_div(group_quads, per)
    quads = ceil_div(group_quads, passes)
    return min(group_quads, (quads + 1) & ~1)


def domain_threads(uarch: Any, hp: Dict[str, Any]) -> int:
    """native_plan_domain_threads: domain_threads, else the L3 domain's cores, else the family's."""
    if hp["domain_threads"] > 0:
        dt = hp["domain_threads"]
    elif uarch.l3_domain_cores > 0:
        dt = uarch.l3_domain_cores
    else:
        dt = uarch.family_l3_cores or 0
    return max(1, dt)


def mem_cycles(uarch: Any, hp: Dict[str, Any]) -> float:
    """native_plan_mem_cycles."""
    if hp["cm_mem_cycles"] > 0:
        return float(hp["cm_mem_cycles"])
    return 2400.0 / math.sqrt(domain_threads(uarch, hp))


def tall_pct(uarch: Any, hp: Dict[str, Any]) -> float:
    """native_plan_tall_pct."""
    if hp["cm_tall_pct"] >= 0:
        return float(hp["cm_tall_pct"])
    doublings = math.log2(domain_threads(uarch, hp) / 8.0)
    return min(max(12.5 * doublings, 0.0), 50.0)


def core_streams(uarch: Any, hp: Dict[str, Any]) -> bool:
    """native_plan_core_streams: one core streams slower than its share of the domain's link."""
    return mem_cycles(uarch, hp) < hp["cm_stream_cycles"]


def cost_int8(p: Any) -> bool:
    """native_plan_cost_int8: the per-channel INT8 tiles the model prices."""
    return p.int8 and p.int8_scheme == "per_channel"


def int4_weight_bytes(p: Any) -> float:
    """native_int4_weight_bytes: half a byte plus the group's scale per weight element."""
    gs = p.group if p.group > 0 else p.k
    return INT4_BYTES + float(p.scale_bytes) / max(1, gs)


def groups_fit_l3(p: Any, q: Any, uarch: Any, hp: Dict[str, Any]) -> bool:
    """native_plan_groups_fit_l3."""
    domains = ceil_div(max(1, q.threads), domain_threads(uarch, hp))
    nbytes = float(p.plan_n) * p.k * (INT8_BYTES if p.int8 else BF16_BYTES)
    return nbytes * 100.0 <= float(uarch.l3_bytes) * domains * hp["cm_group_l3_pct"]


def slice_skewed(q: Any, hp: Dict[str, Any]) -> bool:
    """native_plan_slice_skewed."""
    threads = max(1, q.threads)
    tiles = max(1, q.n_tiles)
    return ceil_div(tiles, threads) * threads * 100 >= tiles * hp["cm_group_skew_pct"]


class PlanCost(NamedTuple):
    core: float = 0.0
    memory: float = 0.0
    cycles: float = 0.0
    stream: float = 0.0
    stream_kib: float = 0.0


class TileModel:
    """tile_model_t: the kernel calls of a tile priced per K step of its kernel."""

    def __init__(self, q: Any, p: Any, uarch: Any, hp: Dict[str, Any]):
        self.q, self.p, self.hp = q, p, hp
        self.int8, self.int4 = p.int8, p.int4
        self.pipes = hp["cm_fma_pipes"] if hp["cm_fma_pipes"] > 0 else max(1, uarch.fma_ports)
        self.l1_bytes = uarch.l1d_bytes * hp["l1_pct"]
        self.kc = 1
        self.passes = 1
        self.groups = 0
        self.l1_fit = False
        f16_ph = q.kind == "f16_ph"
        if self.int8:
            self.slab_cols = int8_col_width(q, q.nr)
            self.mr = int8_call_rows(q, self.slab_cols // VECTOR_COLS)
            self.k_steps = float(ceil_div(p.k, INT8_STEP_K))
            self.wei_bytes = int4_weight_bytes(p) if self.int4 else INT8_BYTES
            self.step_bytes = INT8_STEP_K * (INT4_BYTES if self.int4 else INT8_BYTES)
            self.src_bytes = INT8_BYTES
            self.l2_share = float(uarch.l2_bytes) * hp["int8_l2_pct"] / 100.0
            if self.int4:
                gs = p.group if p.group > 0 else p.k
                self.groups = ceil_div(p.k, max(1, gs))
                self.l1_fit = ((self.slab_cols * self.wei_bytes + self.mr * INT8_BYTES) * p.k * 100
                               <= float(self.l1_bytes))
        else:
            self.slab_cols = q.nr if f16_ph else bf16_slab_cols(q.nr)
            self.mr = q.mr
            self.kc = min(max(1, q.kc), p.k)
            self.passes = ceil_div(p.k, self.kc)
            self.k_steps = float(ceil_div(p.k, 2))
            self.wei_bytes = BF16_BYTES
            self.step_bytes = 2 * BF16_BYTES
            self.src_bytes = BF16_BYTES
            self.l1_fit = (q.nr + q.mr) * self.kc * 2 * 100 <= uarch.l1d_bytes * hp["l1_pct"]
            self.l2_share = float(uarch.l2_bytes) * hp["bf16_l2_pct"] / 100.0
            if p.f16:
                self.k_steps = float(p.k)
                self.step_bytes = BF16_BYTES
            if p.f16 and not f16_ph:
                self.src_bytes = F16_WIDENED_BYTES
                self.l1_fit = ((q.nr * BF16_BYTES + q.mr * F16_WIDENED_BYTES) * self.kc * 100
                               <= float(uarch.l1d_bytes) * hp["l1_pct"])
        self.slabs = max(1, q.nr // max(1, self.slab_cols))
        self.vectors = float(self.slab_cols) / (F16_PH_VECTOR_COLS if f16_ph else VECTOR_COLS)
        self.mem_cycles = mem_cycles(uarch, hp)
        self.tall = 1.0 + tall_pct(uarch, hp) / 100.0
        self.link_bound = self.mem_cycles > hp["cm_stream_cycles"]

    def idle_tiles(self, tiles: int) -> float:
        return float(min(tiles, self.hp["cm_sync_tiles"])) if self.link_bound else float(tiles)

    def call(self, rows: int, from_l2: bool) -> float:
        hp = self.hp
        fma = rows * self.vectors / self.pipes
        if rows > hp["cm_short_rows"]:
            fma *= self.tall
        per = max(fma, float(hp["cm_fma_lat"]))
        if from_l2:
            per = max(per, self.slab_cols * self.step_bytes / max(1, hp["cm_l2_bytes"]))
        return per * self.k_steps + float(self.groups) * rows * self.vectors / self.pipes

    def tile(self, rows: int) -> "tuple[float, float]":
        """(core, idle) of a tile of rows rows."""
        if rows <= 0:
            return 0.0, 0.0
        p, q, hp = self.p, self.q, self.hp
        passes, l1_fit = self.passes, self.l1_fit
        if self.int8 and not self.int4:
            k_quads = ceil_div(p.k, INT8_STEP_K)
            pss = max(1, int8_pass_quads(q, k_quads, self.slab_cols, rows, self.mr))
            passes = ceil_div(k_quads, pss)
            l1_fit = (self.slab_cols + self.mr) * pss * INT8_STEP_K * 100 <= self.l1_bytes
        full, rest = rows // self.mr, rows % self.mr
        calls = full + (1 if rest > 0 else 0)
        first = self.call(self.mr if full > 0 else rest, False)
        if full > 0:
            later = (full - 1) * self.call(self.mr, not l1_fit) + (self.call(rest, not l1_fit) if rest > 0 else 0.0)
        else:
            later = 0.0
        core = self.slabs * (first + later + float(passes) * calls * hp["cm_call_cycles"])
        if (rows * self.src_bytes + q.nr * self.wei_bytes) * float(p.k) > self.l2_share:
            core += rows * float(p.k) * self.src_bytes / 1024.0 * hp["cm_l3_cycles"]
        per_pass = self.slabs * later / passes
        idle = passes * max(0.0, per_pass - float(hp["cm_prefetch_kib"]) * self.mem_cycles)
        return core, idle

    def combine(self, core: float, memory: float) -> float:
        return max(core, memory) + self.hp["cm_overlap_pct"] / 100.0 * min(core, memory)


def _row_group_rows(m: int, unit: int, rg: int, row_groups: int) -> int:
    units = (m + unit - 1) // unit
    r0 = min(m, units * rg // row_groups * unit)
    r1 = min(m, units * (rg + 1) // row_groups * unit)
    return r1 - r0


def plan_cost(q: Any, p: Any, uarch: Any, hp: Dict[str, Any]) -> PlanCost:
    """native_plan_cost: the modeled time of plan q of p; zero for a plan the model does not price."""
    int8 = cost_int8(p) or p.int4
    if ((not p.bf16 and not p.f16 and not int8) or q.nr < 1 or q.mr < 1 or q.mb < 1 or q.threads < 1
            or q.n_tiles < 1 or q.m_blocks < 1):
        return PlanCost()
    if int8 and int8_call_rows(q, int8_col_width(q, q.nr) // VECTOR_COLS) < 1:
        return PlanCost()
    model = TileModel(q, p, uarch, hp)
    m = p.m
    tile_kib = float(q.nr) * p.k * model.wei_bytes / 1024.0
    workers = q.threads
    g = q.col_groups
    dt = domain_threads(uarch, hp)
    domains = min(ceil_div(workers, dt), MAX_DOMAINS)

    def first_of(d: int) -> int:
        return d * dt

    def end_of(d: int) -> int:
        return workers if d == domains - 1 else min(workers, (d + 1) * dt)

    domain_kib = [0.0] * MAX_DOMAINS

    def domain_cycles(d: int) -> float:
        return domain_kib[d] * model.mem_cycles / max(1, end_of(d) - first_of(d))

    def share_kib(d: int) -> float:
        return domain_kib[d] / max(1, end_of(d) - first_of(d))

    out = [0.0, 0.0, 0.0, 0.0, 0.0]

    def keep(core: float, memory: float, stream: float, d: int) -> None:
        cycles = model.combine(core, memory)
        if cycles > out[2]:
            out[0], out[1], out[2], out[3], out[4] = core, memory, cycles, stream, share_kib(d)

    if q.unit_tiles == 0 and g > 0 and workers % g == 0 and q.n_tiles % g == 0:
        row_groups = workers // g
        cols = q.n_tiles // g
        unit = max(1, q.row_unit)
        weight_kib = float(q.n_tiles) * tile_kib
        l3_share_kib = float(uarch.l3_bytes) * hp["l3_pct"] / 100.0 / 1024.0
        reuse = weight_kib / ceil_div(workers, dt) * hp["cm_reuse_pct"] / 100.0 <= l3_share_kib
        rereads = [False] * MAX_DOMAINS
        for d in range(domains):
            a, b = first_of(d), end_of(d)
            reads = 0
            rg = a // g
            while rg * g < b:
                if _row_group_rows(m, unit, rg, row_groups) > 0:
                    reads += min(b, (rg + 1) * g) - max(a, rg * g)
                rg += 1
            domain_kib[d] = ((min(g, reads) if reuse else reads) * cols) * tile_kib
            rereads[d] = reuse and reads > g
        for rg in range(row_groups):
            rows = _row_group_rows(m, unit, rg, row_groups)
            if rows <= 0:
                continue
            core = idle = 0.0
            m0 = 0
            while m0 < rows:
                tc, ti = model.tile(min(q.mb, rows - m0))
                core += tc
                idle += ti
                m0 += q.mb
            l3 = cols * tile_kib * hp["cm_l3_cycles"]
            last_domain = min(domains - 1, ((rg + 1) * g - 1) // dt)
            for d in range(min(domains - 1, rg * g // dt), last_domain + 1):
                reread = l3 if rereads[d] else 0.0
                keep(cols * core, domain_cycles(d) + model.idle_tiles(cols) * idle + reread,
                     domain_cycles(d) + reread, d)
        return PlanCost(*out)
    last = m - (q.m_blocks - 1) * q.mb
    full_core, full_idle = model.tile(q.mb)
    last_core, last_idle = model.tile(last)
    mbs = q.m_blocks
    tiles = mbs * q.n_tiles
    n_inner = q.traversal == "n_inner"
    per, extra = tiles // workers, tiles % workers
    first_last = (mbs - 1) * q.n_tiles
    col_step = (per // mbs, (per + 1) // mbs)
    row_step = (per % mbs, (per + 1) % mbs)

    class Walk:
        __slots__ = ("t0", "col", "row", "err")

        def __init__(self) -> None:
            self.t0 = self.col = self.row = self.err = 0

    def step(c: Walk) -> "tuple[int, int, int]":
        """(count, lasts, cols) of the next worker's share."""
        c.err += extra
        longer = 1 if c.err >= workers else 0
        if longer:
            c.err -= workers
        count = per + longer
        col = c.col + col_step[longer]
        row = c.row + row_step[longer]
        if row >= mbs:
            row -= mbs
            col += 1
        lasts = cols = 0
        if count > 0:
            if not n_inner:
                lasts = col - c.col
                cols = (col - 1 if row == 0 else col) - c.col + 1
            else:
                t1 = c.t0 + count
                lasts = max(0, t1 - max(c.t0, first_last))
                cols = min(q.n_tiles, count)
        c.t0 += count
        c.col = col
        c.row = row
        return count, lasts, cols

    c = Walk()
    for d in range(domains):
        for _ in range(first_of(d), end_of(d)):
            domain_kib[d] += step(c)[2] * tile_kib
    c = Walk()
    for d in range(domains):
        memory = domain_cycles(d)
        for _ in range(first_of(d), end_of(d)):
            count, lasts, cols = step(c)
            if count == 0:
                continue
            fulls = count - lasts
            core = fulls * full_core + lasts * last_core
            idle = model.idle_tiles(cols) * (full_idle if mbs > 1 or lasts == 0 else last_idle)
            keep(core, memory + idle, memory, d)
    return PlanCost(*out)
