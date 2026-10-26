/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_planner.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <numeric>
#include <utility>

#include <omp.h>

#include "common/hash_utils.hpp"
#include "common/zendnnl_global.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/int8/native_int8_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/native_kernel_registry.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_int8_tile.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_tuning_file.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_hw.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::common::hash_combine;
using zendnnl::common::post_op_type_t;
using zendnnl::error_handling::apilog_verbose;
using zendnnl::error_handling::apilog_verbose_enabled;
using zendnnl::error_handling::apilog_warning;

namespace {

/// Rows per call of the tallest kernels; also the INT8 tile's per-row stack
/// arrays.
constexpr int kMaxCallRows = 12;

/// Columns of the widest tile the kernels serve.
constexpr int kMaxTileCols = 32;

/// Slots a lookup or an insert of the plan cache probes.
constexpr size_t kCacheWindow = 16;

/// Entries a planner warns about when it drops them; the drops after these
/// get one line, then none.
constexpr uint32_t kMaxDropWarnings = 16;
static_assert((kNativePlanCacheSlots & (kNativePlanCacheSlots - 1)) == 0,
        "the slot index is a mask of the hash");

/// The first slot of p's probe window.
size_t cache_slot(const native_problem_t &p) {
    uint64_t h = hash_native_problem(p);
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdull;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ull;
    h ^= h >> 33;
    return static_cast<size_t>(h) & (kNativePlanCacheSlots - 1);
}

/// The source layers, lowest precedence first.
constexpr native_plan_source_t kLayers[] = {native_plan_source_t::rules,
        native_plan_source_t::table, native_plan_source_t::override};
constexpr int kLayerCount = 3;

int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

const char *source_name(native_plan_source_t s) {
    switch (s) {
        case native_plan_source_t::rules: return "rules";
        case native_plan_source_t::table: return "table";
        case native_plan_source_t::override: return "override";
        default: return "heuristic";
    }
}

bool has_layer(const native_plan_t &plan, native_plan_source_t layer) {
    return (plan.sources & (1u << static_cast<unsigned>(layer))) != 0;
}

void keep_layer(native_plan_t &plan, native_plan_source_t layer,
        const native_config_entry_t *entry) {
    plan.sources = static_cast<uint8_t>(
            plan.sources | (1u << static_cast<unsigned>(layer)));
    switch (layer) {
        case native_plan_source_t::rules: plan.rules_entry = entry; break;
        case native_plan_source_t::table: plan.table_entry = entry; break;
        default: plan.override_entry = entry; break;
    }
}

/// The v2 rules a parameter set turns on.
struct v2_rules_t {
    bool makespan = false;
    bool l1_pass = false;
    bool traffic = false;
};

v2_rules_t v2_rules(const heuristic_params_t &hp) {
    const bool v2 = hp.preset == native_preset_t::v2;
    const auto on = [v2](int s) { return s < 0 ? v2 : s != 0; };
    v2_rules_t r;
    r.makespan = on(hp.makespan);
    r.l1_pass = on(hp.l1_pass);
    r.traffic = on(hp.traffic_groups);
    return r;
}

/// A tile grid of the makespan rule.
struct grid_t {
    int nr = 0;
    int mr = 0;
    int mb = 0;
    int m_blocks = 0;
    int n_tiles = 0;
};

grid_t grid_of(const native_plan_t &q) {
    grid_t g;
    g.nr = q.nr;
    g.mr = q.mr;
    g.mb = q.mb;
    g.m_blocks = q.m_blocks;
    g.n_tiles = q.n_tiles;
    return g;
}

/// Row splits per tile width the makespan rule weighs.
constexpr int kMaxSplits = 6;

/// The work a candidate must stay below to replace one of work work: keep
/// percent of it, rounded up, with keep = 100 - gain_pct.
int64_t gain_bound(int64_t work, int gain_pct) {
    const int64_t keep = 100 - std::clamp(gain_pct, 0, 100);
    return work / 100 * keep + (work % 100 * keep + 99) / 100;
}

/// A tile's work per element of K: its bytes (B slab and A rows) or its
/// multiply-adds.
int64_t tile_work(int nr, int rows, bool bytes) {
    return bytes ? static_cast<int64_t>(nr) + rows
                 : static_cast<int64_t>(nr) * rows;
}

/// The work of the busiest of workers workers (at least 1) taking the tiles
/// of g in slices of the m_inner order.  Above bound only whether it is
/// below bound counts.
int64_t slice_makespan(const grid_t &g, int M, int workers, bool bytes,
        int64_t bound = std::numeric_limits<int64_t>::max()) {
    const int64_t tiles = static_cast<int64_t>(g.m_blocks) * g.n_tiles;
    if (tiles <= 0) return 0;
    const int64_t full = tile_work(g.nr, g.mb, bytes);
    const int64_t per = tiles / workers, extra = tiles % workers;
    const int64_t most = per + (extra > 0 ? 1 : 0);
    const int last = M - (g.m_blocks - 1) * g.mb;
    if (last >= g.mb) return most * full;
    // Slice [t0, t1) holds floor(t1 / m) - floor(t0 / m) tiles of the last
    // row block; t0 mod m carries from one slice to the next.
    const int64_t m = g.m_blocks;
    const int64_t lighter = full - tile_work(g.nr, last, bytes);
    if (most * full - (most + m - 1) / m * lighter >= bound) return bound;
    const int64_t q0 = per / m, r0 = per % m;
    const int64_t q1 = (per + 1) / m, r1 = (per + 1) % m;
    int64_t busiest = 0, err = 0, phase = 0;
    for (int w = 0; w < workers; ++w) {
        err += extra;
        const bool longer = err >= workers;
        if (longer) err -= workers;
        phase += longer ? r1 : r0;
        int64_t lasts = longer ? q1 : q0;
        if (phase >= m) {
            phase -= m;
            ++lasts;
        }
        busiest = std::max(
                busiest, (longer ? per + 1 : per) * full - lasts * lighter);
    }
    return busiest;
}

/// The grid of the makespan rule, from the legacy grid base.  Per tile
/// width, the pinned nr, else the widest tile down by halves to the width of
/// base (a narrower tile would leave the threads alternate cache lines of
/// each weight block): the M block before the row split and, with the row
/// split, the most row blocks that take each further round of the team.  The
/// busiest worker's work decides: a candidate replaces base only below
/// min_gain_pct percent less than base's work, and the earlier candidate
/// wins a tie.
grid_t makespan_grid(const native_problem_t &p, const heuristic_params_t &hp,
        int threads, const grid_t &base) {
    const int M = p.M, N = p.N;
    const bool bf16 = p.bf16();
    const bool bytes = M <= hp.decode_m;
    const int workers = std::max(1, threads);
    if (base.n_tiles <= 0 || base.m_blocks <= 0) return base;
    grid_t best = base;
    int64_t best_work = gain_bound(
            slice_makespan(base, M, workers, bytes), hp.min_gain_pct);
    // No grid gives its busiest worker less than an even share of the least
    // total work: the widest tiles in one row block for bytes, every column
    // of the widest tiles times M for flops.
    const int widest = hp.nr > 0 ? hp.nr : std::min(p.pack_nr, kMaxTileCols);
    if (widest <= 0) return base;
    const int64_t tiles = N / widest;
    const int64_t least = bytes ? tiles * (widest + static_cast<int64_t>(M))
                                : tiles * widest * M;
    if ((least + workers - 1) / workers >= best_work) return base;
    for (int nr = widest;;) {
        grid_t g;
        g.nr = nr;
        g.n_tiles = nr > 0 ? N / nr : 0;
        g.mr = hp.mr > 0 ? hp.mr
                         : (bf16 && nr == 32 ? hp.tall_rows : hp.tile_rows);
        // A BF16 width counts only with kernels for its rows per call.
        const bool rows_served = !bf16
                || native_kernel_registry().covers(
                        native_kernel_kind_t::bf16, g.mr, nr);
        const auto consider = [&](int mb) {
            g.mb = mb;
            g.m_blocks = ceil_div(M, mb);
            if (!rows_served || g.n_tiles <= 0 || g.m_blocks <= 0) return;
            // A pinned col_groups must still divide the column tiles and the
            // team.
            if (hp.col_groups > 0) {
                const int64_t team = std::min<int64_t>(workers,
                        static_cast<int64_t>(g.m_blocks) * g.n_tiles
                                / std::max(1, hp.min_tiles_per_thread));
                if (g.n_tiles % hp.col_groups != 0
                        || std::max<int64_t>(1, team) % hp.col_groups != 0)
                    return;
            }
            const int64_t work
                    = slice_makespan(g, M, workers, bytes, best_work);
            if (work < best_work) {
                best = g;
                best_work = work;
            }
        };
        const int mb0
                = std::max(1, std::min(M, hp.mb > 0 ? hp.mb : hp.block_rows));
        consider(mb0);
        if (hp.mb == 0 && hp.row_split && g.n_tiles > 0 && g.mr > 0
                && mb0 > g.mr) {
            const int64_t n = g.n_tiles;
            int64_t blocks = ceil_div(M, mb0);
            int prev = mb0;
            int splits = 0;
            for (int i = 0; i < 4 * kMaxSplits && splits < kMaxSplits
                    && prev > g.mr && blocks < M;
                    ++i) {
                const int64_t rounds
                        = ((blocks + 1) * n + workers - 1) / workers;
                blocks = std::min<int64_t>(M, rounds * workers / n);
                const int mb = std::max(g.mr,
                        ceil_div(ceil_div(M, static_cast<int>(blocks)), g.mr)
                                * g.mr);
                if (mb < prev) {
                    consider(mb);
                    prev = mb;
                    ++splits;
                }
            }
        }
        if (hp.nr > 0 || nr <= hp.min_tile_cols || nr <= base.nr) break;
        nr /= 2;
    }
    return best;
}

/// The work of all tiles of the plan's grid, per element of K.
int64_t grid_work(const native_plan_t &q, int M, bool bytes) {
    const int last = M - (q.m_blocks - 1) * q.mb;
    return static_cast<int64_t>(q.n_tiles)
            * ((q.m_blocks - 1) * tile_work(q.nr, q.mb, bytes)
                    + tile_work(q.nr, last, bytes));
}

/// The row groups whose busiest worker has less work than bound, the most
/// column groups on a tie: col_groups, or 0 when none.  Column groups divide
/// the column tiles, the team and the L3 domain.
int idle_col_groups(
        const native_plan_t &q, int M, int dt, bool bytes, int64_t bound) {
    const int top = std::gcd(std::gcd(q.n_tiles, q.threads), dt);
    const int unit = std::max(1, q.row_unit);
    const int64_t row_units = (static_cast<int64_t>(M) + unit - 1) / unit;
    int best = 0;
    int64_t best_work = bound;
    for (int g = top; g >= 1; --g) {
        if (top % g != 0) continue;
        const int64_t groups = q.threads / g;
        const int64_t units = (row_units + groups - 1) / groups;
        const int rows = static_cast<int>(std::min<int64_t>(M, units * unit));
        const int64_t work = (q.n_tiles / g) * tile_work(q.nr, rows, bytes);
        if (work < best_work) {
            best = g;
            best_work = work;
        }
    }
    return best;
}

/// col_groups of the traffic rule, or 0 for the slice: when the source and
/// the B slabs of a domain's threads exceed the domain's share of L3, the
/// schedule whose L3 domains read fewer source and weight bytes from memory.
/// In the slice the threads of a domain read the source once per sweep and
/// per distinct row block their slices start at, and their share of the
/// weight.  With g column groups a domain reads only its rows of the source
/// (once, or once per column tile of a thread when they do not fit either)
/// and the whole weight once per row group.
int traffic_col_groups(const native_plan_t &q, int M, int N, int K, bool bf16,
        int dt, int64_t l3_share) {
    const int workers = q.threads;
    const int g = std::gcd(std::gcd(q.n_tiles, workers), dt);
    if (q.m_blocks <= 1 || workers <= 1 || g < 1) return 0;
    const int de = std::min(dt, workers);
    const double esz = bf16 ? 2.0 : 1.0;
    const double a = static_cast<double>(M) * K * esz;
    const double b = static_cast<double>(N) * K * esz;
    const double slabs = static_cast<double>(de) * q.nr * K * esz;
    const double share = static_cast<double>(l3_share);
    if (a + slabs <= share) return 0;
    const int64_t tiles = static_cast<int64_t>(q.m_blocks) * q.n_tiles;
    const double sweeps = static_cast<double>(tiles) / workers / q.m_blocks;
    // Distinct row blocks the slices of the first domain start at; past 64
    // threads per domain, as many as there can be.
    int starts = std::min(de, q.m_blocks);
    if (de <= 64) {
        starts = 0;
        for (int w = 0; w < de; ++w) {
            const int64_t phase = tiles * w / workers % q.m_blocks;
            bool seen = false;
            for (int v = 0; v < w && !seen; ++v)
                seen = tiles * v / workers % q.m_blocks == phase;
            starts += seen ? 0 : 1;
        }
    }
    const double slice = a * sweeps * starts + b * de / workers;
    const double rows = a * de / workers;
    const double rg_a = rows + slabs <= share ? rows : rows * (q.n_tiles / g);
    const double rg = rg_a + b * de / g;
    return rg < slice ? g : 0;
}

} // namespace

bool native_problem_t::operator==(const native_problem_t &o) const {
    return M == o.M && N == o.N && K == o.K && pack_nr == o.pack_nr
            && threads == o.threads && src_dt == o.src_dt && wei_dt == o.wei_dt
            && dst_dt == o.dst_dt && trans_a == o.trans_a
            && dynamic_quant == o.dynamic_quant && has_bias == o.has_bias
            && epilogue == o.epilogue && int8_scheme == o.int8_scheme;
}

size_t hash_native_problem(const native_problem_t &p) noexcept {
    size_t h = 0;
    h = hash_combine(h, p.M);
    h = hash_combine(h, p.N);
    h = hash_combine(h, p.K);
    h = hash_combine(h, p.pack_nr);
    h = hash_combine(h, p.threads);
    h = hash_combine(h, static_cast<int>(p.src_dt));
    h = hash_combine(h, static_cast<int>(p.wei_dt));
    h = hash_combine(h, static_cast<int>(p.dst_dt));
    const unsigned flags = (p.trans_a ? 1u : 0u) | (p.dynamic_quant ? 2u : 0u)
            | (p.has_bias ? 4u : 0u) | (static_cast<unsigned>(p.epilogue) << 3)
            | (static_cast<unsigned>(p.int8_scheme) << 5);
    return hash_combine(h, flags);
}

native_problem_t make_native_problem(const GemmDescriptor &desc,
        const matmul_params &params, const void *bias, int pack_nr) {
    native_problem_t p;
    p.M = desc.M;
    p.N = desc.N;
    p.K = desc.K;
    p.pack_nr = pack_nr;
    p.threads = omp_in_parallel() ? 1 : std::max(1, desc.num_threads);
    p.src_dt = desc.src_dt;
    p.wei_dt = desc.wei_dt;
    p.dst_dt = desc.dst_dt;
    p.trans_a = desc.transA;
    const bool int8 = desc.wei_dt == data_type_t::s8;
    p.dynamic_quant = int8 && params.dynamic_quant
            && (desc.src_dt == data_type_t::bf16
                    || desc.src_dt == data_type_t::f32);
    p.has_bias = bias != nullptr;
    const native_int8_decision_t int8_decision = int8
            ? decide_int8_call(desc, params, bias)
            : native_int8_decision_t {};
    p.int8_scheme = int8_decision.scheme;

    bool postops = desc.alpha != 1.0f || desc.beta != 0.0f
            || (int8 && params.quant_params.dst_scale.buff != nullptr);
    for (const auto &po : params.postop_) {
        if (po.po_type != post_op_type_t::none) postops = true;
    }
    if (desc.gated_act != matmul_gated_act_t::none)
        p.epilogue = native_epilogue_class_t::gated;
    else if (postops)
        p.epilogue = native_epilogue_class_t::postops;
    else if (bias != nullptr && !int8_decision.bias_in_dequant)
        p.epilogue = native_epilogue_class_t::bias;
    else
        p.epilogue = native_epilogue_class_t::store;
    return p;
}

native_plan_t heuristic_plan(const native_problem_t &p,
        const UarchParams &uarch, const heuristic_params_t &hp) {
    const int M = p.M, N = p.N, K = p.K;
    const bool bf16 = p.bf16();
    const int threads = hp.threads_cap > 0 ? std::min(p.threads, hp.threads_cap)
                                           : p.threads;
    const v2_rules_t v2 = v2_rules(hp);
    const bool bytes = M <= hp.decode_m;
    // Every division below has a positive divisor for any input, so a pin
    // that cannot serve the problem reaches validate_native_plan.
    native_plan_t q;
    q.nr = hp.nr > 0 ? hp.nr : std::min(p.pack_nr, kMaxTileCols);
    q.mb = std::max(1, std::min(M, hp.mb > 0 ? hp.mb : hp.block_rows));
    auto count = [&q, M, N]() {
        q.m_blocks = ceil_div(M, q.mb);
        q.n_tiles = q.nr > 0 ? N / q.nr : 0;
        return q.tiles();
    };
    // Every thread needs a tile: narrow the tiles first (one row block streams
    // each B slab once), then split the rows.
    // A pinned BF16 mr stops at the narrowest width with its kernels.
    const auto narrower_has_mr = [&]() {
        return hp.mr == 0 || !bf16
                || native_kernel_registry().covers(
                        native_kernel_kind_t::bf16, hp.mr, q.nr / 2);
    };
    if (hp.nr > 0)
        count();
    else
        while (count() < threads && q.m_blocks == 1 && q.nr > hp.min_tile_cols
                && narrower_has_mr())
            q.nr /= 2;
    q.mr = hp.mr > 0 ? hp.mr
                     : (bf16 && q.nr == 32 ? hp.tall_rows : hp.tile_rows);
    if (hp.mb == 0 && hp.row_split && count() < threads && q.n_tiles > 0
            && q.mr > 0 && q.mb > q.mr) {
        const int blocks = ceil_div(threads, q.n_tiles);
        q.mb = std::max(q.mr, ceil_div(ceil_div(M, blocks), q.mr) * q.mr);
        count();
    }
    if (v2.makespan) {
        const grid_t g = makespan_grid(p, hp, threads, grid_of(q));
        q.nr = g.nr;
        q.mr = g.mr;
        q.mb = g.mb;
        count();
    }
    q.threads = std::max(1,
            std::min(
                    threads, q.tiles() / std::max(1, hp.min_tiles_per_thread)));

    // Row tiles of a block stream the same B slab; a slab over the
    // bf16_l2_pct share of L2 is split along K.
    q.kc = K;
    if (hp.kc > 0) {
        q.kc = std::min(hp.kc, K);
    } else if (bf16 && q.nr > 0 && q.mb > q.mr) {
        const int kc = static_cast<int>(static_cast<int64_t>(uarch.l2_bytes)
                * hp.bf16_l2_pct / 100
                / (q.nr * static_cast<int>(sizeof(uint16_t))));
        if (kc < K) q.kc = std::max(std::max(2, hp.min_pass_k), kc & ~1);
        if (hp.kc_split == native_kc_split_t::equal && q.kc < K)
            q.kc = (ceil_div(K, ceil_div(K, q.kc)) + 1) & ~1;
        // A block of a few row calls reads each pass's B slab from L1 after
        // its first call when the slab and one call's A rows fit the l1_pct
        // share of L1; the passes are equal.
        if (v2.l1_pass && q.mr > 0 && 2 * q.mr <= q.mb
                && q.mb <= hp.l1_pass_max_mb) {
            const int64_t l1 = static_cast<int64_t>(uarch.l1d_bytes) * hp.l1_pct
                    / 100
                    / ((q.nr + q.mr) * static_cast<int64_t>(sizeof(uint16_t)));
            const int64_t pass = std::max<int64_t>(std::max(2, hp.min_pass_k),
                                         std::min<int64_t>(kc, l1))
                    & ~int64_t {1};
            if (pass < K) {
                const int64_t passes = (K + pass - 1) / pass;
                q.kc = static_cast<int>(
                        ((K + passes - 1) / passes + 1) & ~int64_t {1});
            }
        }
    }

    // Unless every thread takes whole columns, the slices of the tile order
    // start at different row blocks.  When the source and the B slabs of one
    // L3 domain's threads (domain_threads adjacent threads) exceed the l3_pct
    // share of its L3, the source is then read from memory once per starting
    // row block.  There the threads of each domain take the same rows
    // instead: col_groups equal column ranges times threads / col_groups row
    // ranges.
    const int dt = hp.domain_threads > 0 ? hp.domain_threads
            : uarch.l3_domain_cores > 0  ? uarch.l3_domain_cores
                                         : uarch.ccx_cores;
    q.row_unit = hp.row_unit;
    if (hp.col_groups >= 0) {
        q.col_groups = hp.col_groups;
    } else if (hp.schedule == native_schedule_t::row_groups) {
        q.col_groups = std::gcd(std::gcd(q.n_tiles, q.threads), dt);
    } else if (hp.schedule == native_schedule_t::automatic) {
        const int64_t l3_share = static_cast<int64_t>(uarch.l3_bytes_per_ccd)
                * hp.l3_pct / 100;
        if (v2.makespan && q.n_tiles > 0) {
            const int64_t slice
                    = slice_makespan(grid_of(q), M, q.threads, bytes);
            const double team
                    = static_cast<double>(threads) * static_cast<double>(slice);
            const double used = static_cast<double>(grid_work(q, M, bytes));
            if (slice > 0 && 100.0 * (team - used) > hp.idle_pct * team)
                q.col_groups = idle_col_groups(q, M, std::max(1, dt), bytes,
                        gain_bound(slice, hp.min_gain_pct));
        }
        if (q.col_groups == 0 && v2.traffic) {
            if (dt > 1)
                q.col_groups
                        = traffic_col_groups(q, M, N, K, bf16, dt, l3_share);
        } else if (q.col_groups == 0) {
            const int64_t domain_bytes
                    = (static_cast<int64_t>(M)
                              + static_cast<int64_t>(dt) * q.nr)
                    * K * (bf16 ? 2 : 1);
            if (q.m_blocks > 1 && q.n_tiles % q.threads != 0 && dt > 1
                    && domain_bytes > l3_share)
                q.col_groups = std::gcd(std::gcd(q.n_tiles, q.threads), dt);
        }
    }
    q.traversal = hp.traversal;
    q.scratch_floats
            = static_cast<size_t>(bf16 && q.kc >= K ? q.mr : q.mb) * q.nr;
    q.int8_rows_nv1 = std::min(hp.int8_mr, native_int8_ukernel_max_mr(1));
    q.int8_rows_nv2 = std::min(hp.int8_mr, native_int8_ukernel_max_mr(2));
    q.int8_cw_mask = hp.int8_cw_mask;
    q.int8_pass_budget
            = static_cast<int64_t>(uarch.l2_bytes) * hp.int8_l2_pct / 100;
    q.int8_pass_quads
            = hp.int8_kc > 0 ? std::min(hp.int8_kc / 4, ceil_div(K, 4)) : 0;
    // The INT8 form of the L1 pass, at the tile's call width and rows per
    // call; equal passes in whole quads.
    if (!bf16 && hp.int8_kc == 0 && v2.l1_pass && q.nr > 0) {
        const int cw = native_int8_col_width(q, q.nr);
        const int rows = native_int8_call_rows(q, cw / 16);
        if (rows > 0 && 2 * rows <= q.mb && q.mb <= hp.l1_pass_max_mb) {
            const int64_t per_k = static_cast<int64_t>(cw) + rows;
            const int64_t l1 = static_cast<int64_t>(uarch.l1d_bytes) * hp.l1_pct
                    / 100 / per_k;
            const int64_t pass
                    = std::max<int64_t>(hp.min_pass_k,
                              std::min(l1, q.int8_pass_budget / per_k))
                    / 4;
            const int64_t k_quads = (static_cast<int64_t>(K) + 3) / 4;
            if (pass >= 1 && pass < k_quads) {
                const int64_t passes = (k_quads + pass - 1) / pass;
                q.int8_pass_quads
                        = static_cast<int>((k_quads + passes - 1) / passes);
            }
        }
    }
    q.int8_direct_store = hp.direct_store;
    q.int8_bias_in_dequant = hp.bias_in_dequant;
    q.int8_scheme = p.int8_scheme;
    q.copy_block = hp.copy_block;
    q.copy_par_min = bf16 ? hp.bf16_copy_par_min : hp.int8_copy_par_min;
    return q;
}

const char *validate_native_plan(
        const native_plan_t &plan, const native_problem_t &p) {
    const native_plan_t &q = plan;
    const bool bf16 = p.bf16();
    if ((q.nr != 16 && q.nr != 32) || q.nr > p.pack_nr || p.pack_nr % q.nr != 0)
        return "nr is not 16 or 32 dividing pack_nr";
    if (q.mb < 1 || q.mb > p.M || q.m_blocks != ceil_div(p.M, q.mb)
            || q.n_tiles < 1 || p.N % q.nr != 0 || q.n_tiles != p.N / q.nr)
        return "the tile grid does not match M, N, mb and nr";
    const native_kernel_registry_t &kernels = native_kernel_registry();
    if (q.mr < 1 || q.mr > kMaxCallRows) return "mr is outside 1 .. 12";
    if (bf16 && !kernels.covers(native_kernel_kind_t::bf16, q.mr, q.nr))
        return "no BF16 kernel for every row count 1 .. mr at nr";
    if (bf16 && q.kc < p.K && (q.kc < 2 || q.kc % 2 != 0))
        return "kc is below K and not an even number of at least 2";
    if (q.threads < 1 || q.threads > p.threads)
        return "threads is outside 1 .. the requested team";
    if (q.col_groups < 0
            || (q.col_groups > 0
                    && (q.threads % q.col_groups != 0
                            || q.n_tiles % q.col_groups != 0)))
        return "col_groups does not divide threads and n_tiles";
    if (q.row_unit < 1 || q.copy_block < 1 || q.copy_par_min < 0)
        return "row_unit, copy_block or copy_par_min is out of range";
    if (!bf16) {
        if ((q.int8_cw_mask & kNativeInt8Cw16) == 0)
            return "int8_cw does not allow 16";
        for (const int nv : {1, 2}) {
            const int width = 16 * nv;
            const uint8_t bit = nv == 2 ? kNativeInt8Cw32 : kNativeInt8Cw16;
            if (width > q.nr || (q.int8_cw_mask & bit) == 0) continue;
            const int rows = native_int8_call_rows(q, nv);
            if (rows < 1 || rows > kMaxCallRows
                    || !kernels.covers_int8(rows, width, p.int8_scheme))
                return "no INT8 kernel of the call's scheme for the rows per "
                       "call at an allowed width";
        }
        if (q.int8_pass_quads < 0 || q.int8_pass_budget < 0)
            return "int8_pass_quads or int8_pass_budget is negative";
    }
    if (q.scratch_floats
            != static_cast<size_t>(bf16 && q.kc >= p.K ? q.mr : q.mb) * q.nr)
        return "scratch_floats does not match the tile";
    return nullptr;
}

int native_int8_pass_quads(
        const native_plan_t &p, int k_quads, int cw, int rows, int max_mr) {
    if (p.int8_pass_quads > 0) return std::min(p.int8_pass_quads, k_quads);
    // All of K, unless the tile takes more than one row call and the B slab
    // plus one row call's A rows exceed the budget; then equal passes that
    // fit, so every row call reads a pass's slab from L2.
    const int64_t quad_bytes = static_cast<int64_t>(4) * (cw + max_mr);
    const int64_t budget = p.int8_pass_budget;
    if (rows <= max_mr || k_quads * quad_bytes <= budget) return k_quads;
    const int64_t per = std::max<int64_t>(1, budget / quad_bytes);
    const int64_t passes = (k_quads + per - 1) / per;
    return static_cast<int>((k_quads + passes - 1) / passes);
}

struct native_planner_t::cache_entry_t {
    native_problem_t problem;
    native_plan_t plan;
};

native_planner_t::native_planner_t(const UarchParams &uarch,
        std::unique_ptr<const native_config_source_t> override_source,
        std::unique_ptr<const native_config_source_t> table_source,
        std::unique_ptr<const native_config_source_t> rules_source)
    : uarch_(uarch)
    , override_(std::move(override_source))
    , table_(std::move(table_source))
    , rules_(std::move(rules_source)) {
    if (!has_sources()) return;
    cache_ = std::make_unique<std::atomic<const cache_entry_t *>[]>(
            kNativePlanCacheSlots);
    for (size_t i = 0; i < kNativePlanCacheSlots; ++i)
        cache_[i].store(nullptr, std::memory_order_relaxed);
}

native_planner_t::~native_planner_t() {
    if (cache_ == nullptr) return;
    for (size_t i = 0; i < kNativePlanCacheSlots; ++i)
        delete cache_[i].load(std::memory_order_relaxed);
}

native_plan_t native_planner_t::plan(const native_problem_t &p) const noexcept {
    if (!has_sources()) return heuristic_plan(p, uarch_, heuristic_params_t {});
    if (p.M > kNativePlanCacheMaxM) return resolve(p);
    // Slots only ever go from empty to an immutable entry, and an insert takes
    // the first empty slot of the window: p is before any empty slot.
    const size_t first = cache_slot(p);
    for (size_t j = 0; j < kCacheWindow; ++j) {
        const cache_entry_t *e
                = cache_[(first + j) & (kNativePlanCacheSlots - 1)].load(
                        std::memory_order_acquire);
        if (e == nullptr) break;
        if (e->problem == p) return e->plan;
    }
    const native_plan_t q = resolve(p);
    remember(p, q);
    return q;
}

void native_planner_t::remember(
        const native_problem_t &p, const native_plan_t &plan) const noexcept {
    // Allocated only once an empty slot is seen: a full window costs no
    // allocation.
    cache_entry_t *mine = nullptr;
    const size_t first = cache_slot(p);
    for (size_t j = 0; j < kCacheWindow; ++j) {
        std::atomic<const cache_entry_t *> &slot
                = cache_[(first + j) & (kNativePlanCacheSlots - 1)];
        const cache_entry_t *cur = slot.load(std::memory_order_acquire);
        if (cur == nullptr) {
            if (mine == nullptr) {
                mine = new (std::nothrow) cache_entry_t {p, plan};
                if (mine == nullptr) return;
            }
            if (slot.compare_exchange_strong(cur, mine,
                        std::memory_order_acq_rel, std::memory_order_acquire))
                return;
        }
        // Taken, by now or before: by p (a racing insert) or another problem.
        if (cur->problem == p) break;
    }
    delete mine;
}

bool native_planner_t::has_sources() const noexcept {
    return override_ != nullptr || table_ != nullptr || rules_ != nullptr;
}

native_plan_t native_planner_t::resolve(
        const native_problem_t &p) const noexcept {
    // The checks and the warnings of the layers allocate; a call that finds
    // no memory for them gets the built-in plan.
    try {
        return resolve_layers(p);
    } catch (...) { return heuristic_plan(p, uarch_, heuristic_params_t {}); }
}

native_plan_t native_planner_t::resolve_layers(
        const native_problem_t &p) const {
    const native_config_entry_t *const found[kLayerCount]
            = {rules_ != nullptr ? rules_->lookup(p) : nullptr,
                    table_ != nullptr ? table_->lookup(p) : nullptr,
                    override_ != nullptr ? override_->lookup(p) : nullptr};
    // Each layer without the keys the problem's INT8 scheme does not apply;
    // a layer left with no key is not applied.
    const native_config_entry_t *entries[kLayerCount] = {};
    native_config_t configs[kLayerCount];
    const bool scoped
            = !p.bf16() && p.int8_scheme != native_int8_scheme_t::per_channel;
    for (int i = 0; i < kLayerCount; ++i) {
        const native_config_entry_t *e = found[i];
        if (e == nullptr) continue;
        configs[i] = scoped
                ? native_config_for_scheme(e->config, p.int8_scheme, nullptr)
                : e->config;
        const bool dropped = scoped && !(configs[i] == e->config);
        if (dropped && !e->scope_noted.exchange(true, std::memory_order_relaxed)
                && apilog_verbose_enabled()) {
            std::string names;
            native_config_for_scheme(e->config, p.int8_scheme, &names);
            apilog_verbose("native BRGEMM: ", e->label, " sets ", names,
                    ", which an INT8 call of scheme ",
                    native_int8_scheme_name(p.int8_scheme), " does not apply");
        }
        if (!dropped || !configs[i].empty()) entries[i] = e;
    }
    bool keep[kLayerCount] = {};
    native_config_t merged;
    for (int i = 0; i < kLayerCount; ++i) {
        if (entries[i] == nullptr) continue;
        merge_native_config(merged, configs[i]);
        keep[i] = true;
    }
    if (!keep[0] && !keep[1] && !keep[2])
        return heuristic_plan(p, uarch_, heuristic_params_t {});

    // The plan of a merged config, or nullptr with why set.
    std::string bad;
    const auto legal = [&](const native_config_t &cfg, native_plan_t &out) {
        if (!check_native_config(cfg, bad)) return bad.c_str();
        out = heuristic_plan(
                p, uarch_, apply_native_config(heuristic_params_t {}, cfg));
        return validate_native_plan(out, p);
    };
    native_plan_t plan;
    if (legal(merged, plan) != nullptr) {
        // Rebuild from the highest layer down: each layer stays only if the
        // plan with it under the layers kept above it is legal, so a lower
        // layer never removes a higher one.  No layer kept: the heuristic
        // plan, which is always legal.
        plan = heuristic_plan(p, uarch_, heuristic_params_t {});
        native_config_t kept;
        for (int i = kLayerCount - 1; i >= 0; --i) {
            keep[i] = false;
            const native_config_entry_t *e = entries[i];
            if (e == nullptr) continue;
            native_config_t trial = configs[i];
            merge_native_config(trial, kept);
            native_plan_t next;
            const char *why = legal(trial, next);
            if (why == nullptr) {
                kept = trial;
                plan = next;
                keep[i] = true;
            } else if (!e->warned.exchange(true, std::memory_order_relaxed)) {
                const uint32_t n = drop_warnings_.fetch_add(
                        1, std::memory_order_relaxed);
                if (n < kMaxDropWarnings) {
                    // A layer legal alone conflicts with the layers kept
                    // above it.
                    const std::string reason = why;
                    native_plan_t alone;
                    const bool conflict = !kept.empty()
                            && legal(configs[i], alone) == nullptr;
                    apilog_warning("native BRGEMM: ", e->label,
                            conflict ? " conflicts with a higher layer"
                                     : " gives an illegal plan",
                            " for M=", p.M, " N=", p.N, " K=", p.K,
                            " threads=", p.threads, " ",
                            p.bf16() ? "bf16" : "s8", ": ", reason,
                            "; dropped");
                } else if (n == kMaxDropWarnings) {
                    apilog_warning(
                            "native BRGEMM: more entries are dropped "
                            "for some calls; the drops are no longer "
                            "logged");
                }
            }
        }
    }
    for (int i = 0; i < kLayerCount; ++i) {
        if (keep[i]) keep_layer(plan, kLayers[i], entries[i]);
    }
    return plan;
}

const native_planner_t &native_planner() {
    // Never destroyed: calls may still plan while static objects are
    // destroyed.
    static const native_planner_t *const planner
            = make_native_planner(std::getenv("ZENDNNL_MATMUL_NATIVE_CONFIG"),
                    std::getenv("ZENDNNL_MATMUL_NATIVE_TUNING_FILE"))
                      .release();
    return *planner;
}

std::unique_ptr<native_planner_t> make_native_planner(
        const char *config, const char *tuning_file) {
    std::unique_ptr<native_config_source_t> override_source;
    if (config != nullptr && *config != '\0')
        override_source = make_native_override_source(config);
    native_tuning_sources_t file;
    if (tuning_file != nullptr && *tuning_file != '\0')
        file = load_native_tuning_file(tuning_file, native_hw());
    return std::make_unique<native_planner_t>(native_plan_uarch(),
            std::move(override_source), std::move(file.table),
            std::move(file.rules));
}

std::string native_plan_source_names(const native_plan_t &plan) {
    std::string names;
    for (int i = kLayerCount - 1; i >= 0; --i) {
        if (!has_layer(plan, kLayers[i])) continue;
        if (!names.empty()) names += '+';
        names += source_name(kLayers[i]);
    }
    return names.empty() ? source_name(native_plan_source_t::heuristic) : names;
}

std::string native_plan_config_text(const native_plan_t &plan) {
    native_config_t cfg;
    for (const native_config_entry_t *e :
            {plan.rules_entry, plan.table_entry, plan.override_entry}) {
        if (e != nullptr) merge_native_config(cfg, e->config);
    }
    if (plan.int8_scheme != native_int8_scheme_t::per_channel)
        cfg = native_config_for_scheme(cfg, plan.int8_scheme, nullptr);
    const std::string text = format_native_config(cfg);
    return text.empty() ? "-" : text;
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
