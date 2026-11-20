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
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/int8/native_int4_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/int8/native_int8_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/native_kernel_registry.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_int8_tile.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_cost_model.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_tuning_file.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_hw.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_pack.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_utils.hpp"

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

/// Columns of one kernel call: the widest tile of the narrowing and the
/// makespan grid.
constexpr int kMaxTileCols = 32;

/// Column tiles per unit of the blocks schedule without a pin.
constexpr int kBlocksTiles = 1;

/// The parameters of the built-in plan.
constexpr heuristic_params_t kBuiltInParams {};

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

/// A BF16 tile of 64 or 128 columns: 32-column slabs, each inside one pack
/// block (the kernel reads a slab with the block's stride), of whole blocks
/// or of one block.
bool bf16_wide_width(int nr, int pack_nr) {
    return (nr == 64 || nr == kNativeBf16MaxTileCols) && pack_nr > 0
            && pack_nr % native_bf16_slab_cols(nr) == 0
            && (nr % pack_nr == 0 || pack_nr % nr == 0);
}

/// BF16 and FP32 rows per call by tile width: tall_rows from 32 columns,
/// else tile_rows; INT8 tile_rows.
int call_rows(const heuristic_params_t &hp, bool int8, int nr) {
    if (hp.mr > 0) return hp.mr;
    return !int8 && nr >= 32 ? hp.tall_rows : hp.tile_rows;
}

/// The kernels of p's tile: BF16, FP32, F16 and F16_PH run one kernel per
/// row call and K pass, INT8 its own tile.
native_kernel_kind_t kernel_kind(const native_problem_t &p) {
    if (p.int8()) return native_kernel_kind_t::int8;
    if (p.f16()) return native_kernel_registry().f16_kind();
    return p.f32() ? native_kernel_kind_t::f32 : native_kernel_kind_t::bf16;
}

/// The widest tile of p's plans without a pinned nr: kMaxTileCols columns,
/// or on the F16_PH kernels from f16_wide_min_m rows, whose tile of two
/// blocks shares each source broadcast, two blocks of kMaxTileCols where N
/// is whole tiles of them.
int widest_tile(const native_problem_t &p, native_kernel_kind_t kind,
        const heuristic_params_t &hp) {
    if (kind == native_kernel_kind_t::f16_ph && p.M >= hp.f16_wide_min_m
            && p.pack_nr == kMaxTileCols && p.N % (2 * kMaxTileCols) == 0)
        return 2 * kMaxTileCols;
    return std::min(p.pack_nr, kMaxTileCols);
}

/// Bytes of a source element of the call (of a weight element too, but for
/// an s4 weight): the L1, L2 and L3 rules weigh rows and slabs by it.
int elem_bytes(const native_problem_t &p) {
    return p.f32() ? 4 : (p.bf16() || p.f16() ? 2 : 1);
}

/// Bytes of a weight element: elem_bytes, half a byte for an s4 weight.
double wei_elem_bytes(const native_problem_t &p) {
    return p.int4() ? 0.5 : elem_bytes(p);
}

/// The units of the bytes the makespan weighs per element of K: a column of
/// weight and a row of source count one each, but over an s4 weight, whose
/// column takes half the bytes of a source row, two for a row.
int row_byte_units(const native_problem_t &p) {
    return p.int4() ? 2 : 1;
}

const char *dtype_name(const native_problem_t &p) {
    if (p.f16()) return "f16";
    return p.f32() ? "f32" : (p.bf16() ? "bf16" : (p.int4() ? "s4" : "s8"));
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

/// The rules a parameter set turns on.
struct rule_switches_t {
    bool makespan = false;
    bool l1_pass = false;
    bool traffic = false;
    bool bf16_alias = false;
    bool bf16_narrow_n = false;
    bool blocks = false;
};

rule_switches_t rule_switches(const heuristic_params_t &hp) {
    rule_switches_t r;
    r.makespan = hp.makespan != 0;
    r.l1_pass = hp.l1_pass != 0;
    r.traffic = hp.traffic_groups != 0;
    r.bf16_alias = hp.bf16_alias != 0;
    r.bf16_narrow_n = hp.bf16_narrow_n != 0;
    r.blocks = hp.blocks != 0;
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

/// A tile's work per element of K: with bytes its bytes (B slab and A rows,
/// a row of A bytes units to a column of B, row_byte_units), else its
/// multiply-adds.
int64_t tile_work(int nr, int rows, int bytes) {
    return bytes != 0
            ? static_cast<int64_t>(nr) + static_cast<int64_t>(rows) * bytes
            : static_cast<int64_t>(nr) * rows;
}

/// The work of the busiest of workers workers (at least 1) taking the tiles
/// of g in slices of the m_inner order.  Above bound only whether it is
/// below bound counts.
int64_t slice_makespan(const grid_t &g, int M, int workers, int bytes,
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

/// The grid of the makespan rule, from base, the grid of the narrowing and
/// the row split.  Per tile width, the pinned nr, else the widest tile down
/// by halves to the width of base (a narrower tile would leave the threads
/// alternate cache lines of each weight block): the M block before the row
/// split and, with the row split, the most row blocks that take each further
/// round of the team.  The busiest worker's work decides: a candidate
/// replaces base only below min_gain_pct percent less than base's work, and
/// the earlier candidate wins a tie.
grid_t makespan_grid(const native_problem_t &p, const heuristic_params_t &hp,
        int threads, const grid_t &base) {
    const int M = p.M, N = p.N;
    const bool int8 = p.int8();
    const int bytes = M <= hp.decode_m ? row_byte_units(p) : 0;
    const int workers = std::max(1, threads);
    if (base.n_tiles <= 0 || base.m_blocks <= 0) return base;
    grid_t best = base;
    int64_t best_work = gain_bound(
            slice_makespan(base, M, workers, bytes), hp.min_gain_pct);
    // No grid gives its busiest worker less than an even share of the least
    // total work: the widest tiles in one row block for bytes, every column
    // of the widest tiles times M for flops.
    const int widest = hp.nr > 0 ? hp.nr : widest_tile(p, kernel_kind(p), hp);
    if (widest <= 0) return base;
    const int64_t tiles = N / widest;
    const int64_t least = bytes != 0
            ? tiles * (widest + static_cast<int64_t>(M) * bytes)
            : tiles * widest * M;
    if ((least + workers - 1) / workers >= best_work) return base;
    for (int nr = widest;;) {
        grid_t g;
        g.nr = nr;
        g.n_tiles = nr > 0 ? N / nr : 0;
        g.mr = call_rows(hp, int8, nr);
        // A BF16 or FP32 width counts only with kernels for its rows per
        // call.
        const bool rows_served = int8
                || native_kernel_registry().covers(kernel_kind(p), g.mr, nr);
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
int64_t grid_work(const native_plan_t &q, int M, int bytes) {
    const int last = M - (q.m_blocks - 1) * q.mb;
    return static_cast<int64_t>(q.n_tiles)
            * ((q.m_blocks - 1) * tile_work(q.nr, q.mb, bytes)
                    + tile_work(q.nr, last, bytes));
}

/// The work, per element of K, of the busiest worker of g column groups of
/// the plan: the column tiles of its row group over the most rows a group
/// takes, of the row_units row units of M.  The idle rule picks row groups
/// by it and the narrow-N rule weighs them by it.
int64_t row_group_work(
        const native_plan_t &q, int g, int64_t row_units, int M, int bytes) {
    const int unit = std::max(1, q.row_unit);
    const int64_t groups = q.threads / g;
    const int64_t units = (row_units + groups - 1) / groups;
    const int rows = static_cast<int>(std::min<int64_t>(M, units * unit));
    return (q.n_tiles / g) * tile_work(q.nr, rows, bytes);
}

/// The row groups whose busiest worker has less work than bound, the most
/// column groups on a tie: col_groups, or 0 when none.  Column groups divide
/// the column tiles, the team and the L3 domain.
int idle_col_groups(
        const native_plan_t &q, int M, int dt, int bytes, int64_t bound) {
    const int top = std::gcd(std::gcd(q.n_tiles, q.threads), dt);
    const int unit = std::max(1, q.row_unit);
    const int64_t row_units = (static_cast<int64_t>(M) + unit - 1) / unit;
    int best = 0;
    int64_t best_work = bound;
    for (int g = top; g >= 1; --g) {
        if (top % g != 0) continue;
        const int64_t work = row_group_work(q, g, row_units, M, bytes);
        if (work < best_work) {
            best = g;
            best_work = work;
        }
    }
    return best;
}

/// The work of the busiest worker of the plan, per element of K: its slice,
/// or the column tiles of its row group over the most rows a group takes.
/// A plan of the blocks schedule is weighed as the slice of its grid.
int64_t busiest_work(const native_plan_t &q, int M, int bytes) {
    const int g = q.col_groups;
    if (g <= 0 || q.threads % g != 0 || q.n_tiles % g != 0)
        return slice_makespan(grid_of(q), M, std::max(1, q.threads), bytes);
    const int unit = std::max(1, q.row_unit);
    const int64_t row_units = (static_cast<int64_t>(M) + unit - 1) / unit;
    return row_group_work(q, g, row_units, M, bytes);
}

/// col_groups of the traffic rule, or 0 for the slice: when the source and
/// the B slabs of a domain's threads exceed the domain's share of L3, the
/// schedule whose L3 domains read fewer source and weight bytes from memory.
/// In the slice the threads of a domain read the source once per sweep and
/// per distinct row block their slices start at, and their share of the
/// weight.  With g column groups a domain reads only its rows of the source
/// (once, or once per column tile of a thread when they do not fit either)
/// and the whole weight once per row group.  Source elements are elem
/// bytes, weight elements wei_elem bytes.
int traffic_col_groups(const native_plan_t &q, int M, int N, int K, int elem,
        double wei_elem, int dt, int64_t l3_share) {
    const int workers = q.threads;
    const int g = std::gcd(std::gcd(q.n_tiles, workers), dt);
    if (q.m_blocks <= 1 || workers <= 1 || g < 1) return 0;
    const int de = std::min(dt, workers);
    const double esz = elem;
    const double a = static_cast<double>(M) * K * esz;
    const double b = static_cast<double>(N) * K * wei_elem;
    const double slabs = static_cast<double>(de) * q.nr * K * wei_elem;
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
            && epilogue == o.epilogue && int8_scheme == o.int8_scheme
            && wei_group_size == o.wei_group_size
            && wei_scale_bytes == o.wei_scale_bytes;
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
    h = hash_combine(h, flags);
    // Zero but for an s4 weight: the other problems hash as before.
    if (p.wei_group_size == 0 && p.wei_scale_bytes == 0) return h;
    h = hash_combine(h, p.wei_group_size);
    return hash_combine(h, p.wei_scale_bytes);
}

native_problem_t make_native_problem(const GemmDescriptor &desc,
        const matmul_params &params, const void *bias, int pack_nr,
        const native_int8_decision_t &decision) {
    native_problem_t p;
    p.M = desc.M;
    // The tiles of a weight of N not a multiple of its blocks run on its
    // padded width.
    p.N = desc.N % kNativeLayoutPackNr == 0 ? desc.N
                                            : native_padded_n(desc.N, pack_nr);
    p.K = desc.K;
    p.pack_nr = pack_nr;
    p.threads = omp_in_parallel() ? 1 : std::max(1, desc.num_threads);
    p.src_dt = desc.src_dt;
    p.wei_dt = desc.wei_dt;
    p.dst_dt = desc.dst_dt;
    p.trans_a = desc.transA;
    const bool int8
            = desc.wei_dt == data_type_t::s8 || desc.wei_dt == data_type_t::s4;
    p.dynamic_quant = int8 && params.dynamic_quant
            && (desc.src_dt == data_type_t::bf16
                    || desc.src_dt == data_type_t::f32);
    p.has_bias = bias != nullptr;
    const native_int8_decision_t int8_decision
            = int8 ? decision : native_int8_decision_t {};
    p.int8_scheme = int8_decision.scheme;
    if (desc.wei_dt == data_type_t::s4 && int8_decision.groups > 0) {
        p.wei_group_size = desc.K / int8_decision.groups;
        p.wei_scale_bytes = static_cast<int>(
                zendnnl::common::size_of(params.quant_params.wei_scale.dt));
    }

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

namespace {

native_plan_t rules_plan(const native_problem_t &p, const UarchParams &uarch,
        const heuristic_params_t &hp) {
    const int M = p.M, N = p.N, K = p.K;
    // FP32 takes the rules of BF16 with its own element size, but not the
    // BF16 tiles of several slabs.
    const bool bf16 = p.bf16();
    const bool int8 = p.int8();
    const int esz = elem_bytes(p);
    const double wei_esz = wei_elem_bytes(p);
    const int threads = hp.threads_cap > 0 ? std::min(p.threads, hp.threads_cap)
                                           : p.threads;
    const rule_switches_t on = rule_switches(hp);
    const int bytes = M <= hp.decode_m ? row_byte_units(p) : 0;
    // Every division below has a positive divisor for any input, so a pin
    // that cannot serve the problem reaches validate_native_plan.
    native_plan_t q;
    q.kind = p.int4() ? native_kernel_kind_t::int4 : kernel_kind(p);
    const bool f16_ph = q.kind == native_kernel_kind_t::f16_ph;
    q.nr = hp.nr > 0 ? hp.nr : widest_tile(p, q.kind, hp);
    q.mb = std::max(1, std::min(M, hp.mb > 0 ? hp.mb : hp.block_rows));
    auto count = [&q, M, N]() {
        q.m_blocks = ceil_div(M, q.mb);
        q.n_tiles = q.nr > 0 ? N / q.nr : 0;
        return q.tiles();
    };
    // Every thread needs a tile: narrow the tiles first (one row block streams
    // each B slab once), then split the rows.
    // A pinned BF16 or FP32 mr stops at the narrowest width with its kernels.
    const auto narrower_has_mr = [&]() {
        return hp.mr == 0 || int8
                || native_kernel_registry().covers(
                        kernel_kind(p), hp.mr, q.nr / 2);
    };
    if (hp.nr > 0)
        count();
    else
        while (count() < threads && q.m_blocks == 1 && q.nr > hp.min_tile_cols
                && narrower_has_mr())
            q.nr /= 2;
    q.mr = call_rows(hp, int8, q.nr);
    if (hp.mb == 0 && hp.row_split && count() < threads && q.n_tiles > 0
            && q.mr > 0 && q.mb > q.mr) {
        const int blocks = ceil_div(threads, q.n_tiles);
        q.mb = std::max(q.mr, ceil_div(ceil_div(M, blocks), q.mr) * q.mr);
        count();
    }
    if (on.makespan) {
        const grid_t g = makespan_grid(p, hp, threads, grid_of(q));
        q.nr = g.nr;
        q.mr = g.mr;
        q.mb = g.mb;
        count();
    }
    // Source rows K elements apart, a multiple of 4 KiB: the rows of one
    // call map to one L1 set, so a block of more than bf16_alias_mr rows
    // takes at most bf16_alias_mr rows per call (a block of 9 to 12 rows
    // then takes two calls).  A row-major source counts as rows K apart (the
    // problem carries no row stride).  The widened rows of an f16 source
    // the F16 kernels read are never a multiple of 4 KiB apart.
    constexpr int kAliasBytes = 4096;
    if (!int8 && q.kind != native_kernel_kind_t::f16 && on.bf16_alias
            && hp.mr == 0 && K % (kAliasBytes / esz) == 0 && M > hp.decode_m
            && q.nr >= 32 && q.mb > hp.bf16_alias_mr)
        q.mr = std::min(q.mr, hp.bf16_alias_mr);
    q.threads = std::max(1,
            std::min(
                    threads, q.tiles() / std::max(1, hp.min_tiles_per_thread)));

    // Row tiles of a block stream the same B slab; a slab over the
    // bf16_l2_pct share of L2 is split along K.
    q.kc = K;
    if (hp.kc > 0) {
        q.kc = std::min(hp.kc, K);
    } else if (!int8 && q.nr > 0 && q.mb > q.mr) {
        const int kc = static_cast<int>(static_cast<int64_t>(uarch.l2_bytes)
                * hp.bf16_l2_pct / 100 / (q.nr * esz));
        if (kc < K) q.kc = std::max(std::max(2, hp.min_pass_k), kc & ~1);
        if (hp.kc_split == native_kc_split_t::equal && q.kc < K)
            q.kc = (ceil_div(K, ceil_div(K, q.kc)) + 1) & ~1;
        // A block of a few row calls reads each pass's B slab from L1 after
        // its first call when the slab and one call's A rows fit the l1_pct
        // share of L1; the passes are equal.
        if (on.l1_pass && q.mr > 0 && 2 * q.mr <= q.mb
                && q.mb <= hp.l1_pass_max_mb) {
            const int64_t l1 = static_cast<int64_t>(uarch.l1d_bytes) * hp.l1_pct
                    / 100 / ((q.nr + q.mr) * static_cast<int64_t>(esz));
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
    // The F16_PH kernels accumulate blocks of kNativeF16AccumBlockK steps
    // from the start of each call's K: K passes of whole blocks keep every
    // block at the same K whatever the plan.
    if (f16_ph && q.kc < K)
        q.kc = std::max(kNativeF16AccumBlockK,
                q.kc / kNativeF16AccumBlockK * kNativeF16AccumBlockK);

    // Unless every thread takes whole columns, the slices of the tile order
    // start at different row blocks.  When the source and the B slabs of one
    // L3 domain's threads (domain_threads adjacent threads) exceed the l3_pct
    // share of its L3, the source is then read from memory once per starting
    // row block.  There the threads of each domain take the same rows
    // instead: col_groups equal column ranges times threads / col_groups row
    // ranges.
    const int dt = native_plan_domain_threads(uarch, hp);
    q.row_unit = hp.row_unit;
    const auto take_blocks = [&]() {
        q.col_groups = 0;
        q.unit_tiles = std::min(std::max(1, q.n_tiles),
                hp.unit_tiles > 0 ? hp.unit_tiles : kBlocksTiles);
        // BF16 and FP32 K passes keep the accumulators of every tile of a
        // unit, in the worker's scratch, which is kept for the thread's
        // lifetime: the unit takes at most the tiles whose accumulators fit
        // the bf16_l2_pct share of L2 that sizes the K pass, and at least
        // one.
        if (!int8 && q.kc < K && q.unit_tiles > 1) {
            const int64_t tile_bytes = static_cast<int64_t>(q.mb) * q.nr
                    * static_cast<int64_t>(sizeof(float));
            const int64_t share = static_cast<int64_t>(uarch.l2_bytes)
                    * hp.bf16_l2_pct / 100;
            q.unit_tiles = static_cast<int>(std::max<int64_t>(1,
                    std::min<int64_t>(q.unit_tiles,
                            share / std::max<int64_t>(1, tile_bytes))));
        }
        q.domain_threads = std::max(1, dt);
    };
    bool narrow_blocks = false;
    if (hp.col_groups >= 0) {
        q.col_groups = hp.col_groups;
    } else if (hp.schedule == native_schedule_t::row_groups) {
        q.col_groups = std::gcd(std::gcd(q.n_tiles, q.threads), dt);
    } else if (hp.schedule == native_schedule_t::blocks) {
        take_blocks();
    } else if (hp.schedule == native_schedule_t::automatic) {
        const int64_t l3_share
                = static_cast<int64_t>(uarch.l3_domain_bytes) * hp.l3_pct / 100;
        if (on.makespan && q.n_tiles > 0) {
            const int64_t slice
                    = slice_makespan(grid_of(q), M, q.threads, bytes);
            const double team
                    = static_cast<double>(threads) * static_cast<double>(slice);
            const double used = static_cast<double>(grid_work(q, M, bytes));
            if (slice > 0 && 100.0 * (team - used) > hp.idle_pct * team)
                q.col_groups = idle_col_groups(q, M, std::max(1, dt), bytes,
                        gain_bound(slice, hp.min_gain_pct));
        }
        // The L3 rules below need an L3 (l3_share above 0).
        if (q.col_groups == 0 && on.traffic) {
            if (dt > 1 && l3_share > 0)
                q.col_groups = traffic_col_groups(
                        q, M, N, K, esz, wei_esz, dt, l3_share);
        } else if (q.col_groups == 0) {
            const int64_t domain_bytes = static_cast<int64_t>(M) * K * esz
                    + static_cast<int64_t>(
                            static_cast<double>(dt) * q.nr * K * wei_esz);
            if (q.m_blocks > 1 && q.n_tiles % q.threads != 0 && dt > 1
                    && l3_share > 0 && domain_bytes > l3_share)
                q.col_groups = std::gcd(std::gcd(q.n_tiles, q.threads), dt);
        }
        // Under the blocks schedule the workers of each L3 domain take the
        // units of the domain's share of the row blocks from one counter, so
        // the domain reads only those source rows.  When they fit the l3_pct
        // share of its L3, it replaces the schedule of a BF16 or FP32 plan
        // whose B slab over all of K exceeds the bf16_l2_pct share of L2, and
        // the slice of an INT8 plan.  The deal gives each domain row blocks
        // in proportion to its workers, so a full domain's share is
        // M * de / threads rows.
        const int domains = ceil_div(q.threads, std::max(1, dt));
        const int de = std::min(std::max(1, dt), q.threads);
        const int64_t share_bytes
                = (static_cast<int64_t>(M) * de + q.threads - 1) / q.threads * K
                * esz;
        const int64_t slab_bytes
                = static_cast<int64_t>(static_cast<double>(q.nr) * K * wei_esz);
        if (on.blocks && domains > 1
                && M >= (!int8 ? hp.bf16_blocks_min_m : hp.int8_blocks_min_m)
                && (!int8 ? slab_bytes * 100
                                        > static_cast<int64_t>(uarch.l2_bytes)
                                                * hp.bf16_l2_pct
                          : q.col_groups == 0)
                && share_bytes <= l3_share)
            take_blocks();
        // A BF16 or FP32 slice of 32-column tiles with at most two column
        // tiles per domain: the threads of every domain share column tiles
        // over all the rows, so every domain reads the whole source.  Under
        // the same bounds it takes the blocks schedule, unless bf16_narrow_n
        // widens the tiles of a BF16 plan (below).
        narrow_blocks = on.blocks && !int8 && domains > 1
                && M >= hp.bf16_blocks_min_m && q.unit_tiles == 0
                && q.col_groups == 0 && q.nr == kMaxTileCols
                && q.n_tiles <= 2 * domains && share_bytes <= l3_share;
    }
    q.traversal = hp.traversal;
    q.scratch_floats = native_plan_scratch_floats(q, p);
    // A per-group call runs the group kernel, with its own rows per call,
    // of an s4 weight the INT4 group kernel, or the INT4 kernel of one group.
    const bool group_kernel
            = int8 && p.int8_scheme == native_int8_scheme_t::per_group;
    const bool int4 = p.int4();
    const bool one_group = p.int4_one_group();
    const auto int8_max_mr = [group_kernel, int4, one_group](int nv) {
        if (int4)
            return one_group ? native_int4_one_group_ukernel_max_mr(nv)
                             : native_int4_group_ukernel_max_mr(nv);
        return group_kernel ? native_int8_group_ukernel_max_mr(nv)
                            : native_int8_ukernel_max_mr(nv);
    };
    q.int8_rows_nv1 = std::min(hp.int8_mr, int8_max_mr(1));
    q.int8_rows_nv2 = std::min(hp.int8_mr, int8_max_mr(2));
    q.int8_cw_mask = hp.int8_cw_mask;
    q.int8_pass_budget
            = static_cast<int64_t>(uarch.l2_bytes) * hp.int8_l2_pct / 100;
    // The later row calls of an s4 tile read each batch of its groups again:
    // in a plan of at most decode_m rows, batches of the int4_l1_pct share of
    // L1.
    if (int4 && hp.int4_l1_pct > 0 && M <= hp.decode_m)
        q.int8_pass_budget = std::min(q.int8_pass_budget,
                static_cast<int64_t>(uarch.l1d_bytes) * hp.int4_l1_pct / 100);
    q.int8_pass_quads
            = hp.int8_kc > 0 ? std::min(hp.int8_kc / 4, ceil_div(K, 4)) : 0;
    // The INT8 form of the L1 pass, at the tile's call width and rows per
    // call; equal passes in whole quads.  Only the per-channel tile reads it.
    if (int8 && !group_kernel && hp.int8_kc == 0 && on.l1_pass && q.nr > 0) {
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
    q.copy_par_min = !int8 ? hp.bf16_copy_par_min : hp.int8_copy_par_min;
    // Few column tiles against the team: with fewer than two 32-column tiles
    // per L3 domain in the slice, each domain reads from memory the source
    // rows of its column tiles, and each row is read once per column tile,
    // by up to all the domains.  BF16 tiles of several 32-column slabs that
    // are fewer than the domains cut the domains that read each row.  The
    // widest such tile whose plan, by these rules at its width, gives the
    // busiest worker at most min_gain_pct percent more work replaces the
    // 32-column plan.  The wide plan may take the blocks schedule; it is
    // weighed as the slice of its grid.  The domains are those of the
    // planned team.
    const int team_domains = ceil_div(q.threads, std::max(1, dt));
    if (bf16 && on.bf16_narrow_n && hp.nr == 0 && M >= hp.bf16_narrow_n_min_m
            && q.nr == kMaxTileCols && q.col_groups == 0 && q.unit_tiles == 0
            && q.n_tiles < 2 * team_domains) {
        const int64_t work = busiest_work(q, M, bytes);
        const int64_t most
                = work + work * std::clamp(hp.min_gain_pct, 0, 100) / 100;
        heuristic_params_t wide = hp;
        for (int nr = kNativeBf16MaxTileCols; nr > q.nr; nr /= 2) {
            if (N % nr != 0 || N / nr >= team_domains
                    || !bf16_wide_width(nr, p.pack_nr))
                continue;
            wide.nr = nr;
            const native_plan_t w = rules_plan(p, uarch, wide);
            if (busiest_work(w, M, bytes) <= most
                    && validate_native_plan(w, p) == nullptr)
                return w;
        }
    }
    if (narrow_blocks) {
        take_blocks();
        q.scratch_floats = native_plan_scratch_floats(q, p);
    }
    return q;
}

/// Whether two row groups of whole units of unit rows, the first of
/// ceil(M / unit) / 2 units, each take more than cm_short_rows rows and the
/// larger at most cm_group_rows_pct percent of half of M.
bool row_groups_split(int M, int unit, const heuristic_params_t &hp) {
    unit = std::max(1, unit);
    const int first = std::min(M, ceil_div(M, unit) / 2 * unit);
    return std::min(first, M - first) > hp.cm_short_rows
            && static_cast<int64_t>(std::max(first, M - first)) * 200
            <= static_cast<int64_t>(M) * hp.cm_group_rows_pct;
}

/// The cost model's plan for the rules' BF16 or F16 plan q of tiles of 32
/// columns or more and at least as many column tiles as threads: q, unless a
/// variant of the rules or a pair of variants is modeled more than
/// cm_gain_pct percent faster; then the fastest, the earlier on a tie.  The
/// variants vary only what hp leaves to the rules: cm_short_rows rows per
/// call, and two row groups of half the team each for a single row block
/// whose slice leaves its busiest worker cm_group_skew_pct percent of the
/// mean column tiles or more, whose column tile's slab over all of K fits
/// cm_group_slab_pct percent of L2 and whose weights fit cm_group_l3_pct
/// percent of the L3 of the domains the team spans; row groups of
/// cm_short_rows rows per call split the rows by those calls unless
/// row_unit pins another unit.  A single row block of a long weight stream
/// then takes M blocks of tile_rows rows, or, where one core streams
/// slower than its share of the domain's link, the long K pass (below), at
/// the rules' rows per call and row unit; there the M blocks of a BF16 plan
/// whose team fills each L3 domain it spans need a K above cm_block_k.
/// The chosen row groups take either only where that unit splits the rows
/// within the bounds above.  Fewer
/// column tiles than threads are left to the narrowing, the narrow-N and
/// the blocks rules.
native_plan_t model_plan(const native_problem_t &p, const UarchParams &uarch,
        const heuristic_params_t &hp, const native_plan_t &q) {
    if ((!p.bf16() && !p.f16()) || q.nr < kMaxTileCols || q.mr < 1
            || q.n_tiles < q.threads)
        return q;
    // Shorter calls change only blocks taller than them, and past one row
    // block they gain only through the price of taller calls.  Two row
    // groups take whole row units each (units of cm_short_rows rows for
    // groups of such calls unless row_unit pins another), and each must
    // have more than cm_short_rows rows: a shorter group streams the weights
    // both groups read at the rate of one core.
    const bool short_rows = hp.mr == 0 && hp.cm_short_rows < hp.tall_rows
            && q.mr > hp.cm_short_rows && q.mb > hp.cm_short_rows
            && (q.m_blocks == 1 || native_plan_tall_pct(uarch, hp) > 0);
    const int g = q.threads / 2;
    const bool split_rows = row_groups_split(p.M, q.row_unit, hp);
    // row_unit has no auto value: a unit other than the built-in one is the
    // pin.
    const int short_unit = hp.row_unit != heuristic_params_t {}.row_unit
            ? hp.row_unit
            : hp.cm_short_rows;
    const bool split_short
            = short_rows && row_groups_split(p.M, short_unit, hp);
    const bool row_groups = hp.col_groups < 0
            && hp.schedule == native_schedule_t::automatic && q.col_groups == 0
            && q.unit_tiles == 0 && q.m_blocks == 1 && g >= 1
            && q.threads % 2 == 0 && q.n_tiles % g == 0
            && (split_rows || split_short) && native_plan_slice_skewed(q, hp)
            && static_cast<int64_t>(q.nr) * p.K * 2 * 100
                    <= static_cast<int64_t>(uarch.l2_bytes)
                            * hp.cm_group_slab_pct
            && native_plan_groups_fit_l3(p, q, uarch, hp);
    // A single row block of cm_pass_calls calls or more at the rules' rows
    // per call may take a K pass of cm_pass_kib KiB of slab.
    const int64_t pass_k = (static_cast<int64_t>(hp.cm_pass_kib) * 1024
                                   / (static_cast<int64_t>(q.nr) * 2))
            & ~int64_t {1};
    const bool long_pass = hp.kc == 0 && q.m_blocks == 1
            && ceil_div(q.mb, q.mr) >= hp.cm_pass_calls && pass_k >= 2
            && pass_k > q.kc && pass_k < p.K;
    // A single row block of more than two calls whose source rows and slab
    // over all of K fit the bf16_l2_pct share of L2 may take M blocks of
    // tile_rows rows; where one core streams slower than its share of the
    // domain's link, a BF16 plan whose team fills each L3 domain it spans
    // only over a K above cm_block_k.
    const bool block_k_bound = p.bf16() && native_plan_core_streams(uarch, hp)
            && q.threads % native_plan_domain_threads(uarch, hp) == 0;
    const bool short_blocks = hp.mb == 0 && hp.kc == 0 && q.m_blocks == 1
            && p.M > 2 * q.mr && hp.tile_rows < q.mr
            && (static_cast<int64_t>(p.M) + q.nr) * p.K * 2 * 100
                    <= static_cast<int64_t>(uarch.l2_bytes) * hp.bf16_l2_pct
            && (p.K > hp.cm_block_k || !block_k_bound);
    if (!short_rows && !row_groups && !long_pass && !short_blocks) return q;
    const double base = native_plan_cost(q, p, uarch, hp).cycles;
    if (!(base > 0)) return q;
    native_plan_t best = q;
    double bound = base * (100 - std::clamp(hp.cm_gain_pct, 0, 100)) / 100.0;
    const auto consider = [&](const native_plan_t &c) {
        if (validate_native_plan(c, p) != nullptr) return false;
        const double cycles = native_plan_cost(c, p, uarch, hp).cycles;
        if (!(cycles < bound)) return false;
        best = c;
        bound = cycles;
        return true;
    };
    heuristic_params_t rows = hp;
    rows.tall_rows = hp.cm_short_rows;
    if (short_rows) consider(rules_plan(p, uarch, rows));
    bool took_groups = false;
    if (row_groups) {
        heuristic_params_t groups = hp;
        groups.col_groups = g;
        if (split_rows) took_groups = consider(rules_plan(p, uarch, groups));
        if (split_short) {
            groups.tall_rows = hp.cm_short_rows;
            groups.row_unit = short_unit;
            if (consider(rules_plan(p, uarch, groups))) took_groups = true;
        }
    }
    // The busiest worker of a long weight stream decides how the block
    // takes K.  Where one core streams slower than its share of the
    // domain's link, one that computes longer than it streams, over more
    // than cm_drift_kib KiB of weights, runs out of step with the others
    // (its stream runs ahead of its calls) and takes the long pass, which
    // saves calls.  One of the slice that streams longer than it computes,
    // over more than cm_block_kib KiB, takes M blocks of tile_rows rows:
    // each tile streams its slab once over all of K, and the later blocks
    // re-read it from L2.  The block keeps best's schedule (its column
    // groups, or the blocks schedule and its unit tiles) at the rules' rows
    // per call and row unit; row groups whose rows that unit does not split
    // within the bounds above keep best.
    const bool drift = long_pass && native_plan_core_streams(uarch, hp);
    if ((drift || short_blocks) && (!took_groups || split_rows)) {
        const native_plan_cost_t c = native_plan_cost(best, p, uarch, hp);
        heuristic_params_t r = hp;
        if (best.unit_tiles > 0) {
            r.schedule = native_schedule_t::blocks;
            r.unit_tiles = best.unit_tiles;
        } else {
            r.col_groups = best.col_groups;
        }
        if (drift && c.core > c.stream && c.stream_kib > hp.cm_drift_kib)
            r.kc = static_cast<int>(pass_k);
        else if (short_blocks && best.col_groups == 0 && best.unit_tiles == 0
                && c.core <= c.stream && c.stream_kib > hp.cm_block_kib)
            r.mb = hp.tile_rows;
        else
            return best;
        const native_plan_t s = rules_plan(p, uarch, r);
        if (validate_native_plan(s, p) == nullptr) return s;
    }
    return best;
}

/// The cost model's plan for the rules' INT8 plan q that the model prices
/// (native_plan_cost_int8), of a single row block of tiles of 32 columns or
/// more and at least as many column tiles as threads: q, unless two row
/// groups of half the team, under the BF16 bounds on the split, the slice's
/// balance, the slab and the weights, are modeled more than cm_gain_pct
/// percent faster.  Where one core streams slower than its share of the
/// domain's link and cm_int8_l1_pct is set, the slice of a block of three
/// calls then takes tile_rows rows per call and the INT8 L1 pass at
/// cm_int8_l1_pct percent of L1 while the slab and those rows over all of K
/// fit the int8_l2_pct share of L2.  An int8_mr below 12, int8_kc, mb and
/// the schedule pin what they set.
native_plan_t int8_model_plan(const native_problem_t &p,
        const UarchParams &uarch, const heuristic_params_t &hp,
        const native_plan_t &q) {
    if (!native_plan_cost_int8(p) || q.nr < kMaxTileCols
            || q.n_tiles < q.threads)
        return q;
    const int cw = native_int8_col_width(q, q.nr);
    const int rows = native_int8_call_rows(q, cw / 16);
    if (rows < 1 || hp.int8_mr < kMaxCallRows || q.m_blocks != 1) return q;
    const int g = q.threads / 2;
    const bool row_groups = hp.col_groups < 0
            && hp.schedule == native_schedule_t::automatic && q.col_groups == 0
            && q.unit_tiles == 0 && g >= 1 && q.threads % 2 == 0
            && q.n_tiles % g == 0 && row_groups_split(p.M, q.row_unit, hp)
            && native_plan_slice_skewed(q, hp)
            && static_cast<int64_t>(q.nr) * p.K * 100
                    <= static_cast<int64_t>(uarch.l2_bytes)
                            * hp.cm_group_slab_pct
            && native_plan_groups_fit_l3(p, q, uarch, hp);
    if (row_groups) {
        const double base = native_plan_cost(q, p, uarch, hp).cycles;
        heuristic_params_t v = hp;
        v.col_groups = g;
        const native_plan_t c = rules_plan(p, uarch, v);
        if (base > 0 && validate_native_plan(c, p) == nullptr
                && native_plan_cost(c, p, uarch, hp).cycles < base
                                * (100 - std::clamp(hp.cm_gain_pct, 0, 100))
                                / 100.0)
            return c;
    }
    const int call_rows = std::min(hp.tile_rows, kMaxCallRows);
    if (hp.cm_int8_l1_pct > 0 && q.col_groups == 0 && q.unit_tiles == 0
            && hp.int8_kc == 0 && hp.mb == 0 && ceil_div(q.mb, rows) == 3
            && call_rows < rows && native_plan_core_streams(uarch, hp)
            && (static_cast<int64_t>(q.nr) + call_rows) * p.K * 100
                    <= static_cast<int64_t>(uarch.l2_bytes) * hp.int8_l2_pct) {
        heuristic_params_t v = hp;
        v.int8_mr = call_rows;
        v.l1_pct = hp.cm_int8_l1_pct;
        v.col_groups = 0;
        const native_plan_t r = rules_plan(p, uarch, v);
        if (r.int8_pass_quads > 0 && validate_native_plan(r, p) == nullptr)
            return r;
    }
    return q;
}

/// Row blocks the cost model weighs for a single s4 row block: 2 to this.
constexpr int kInt4RowBlocks = 4;

/// The cost model's plan for the rules' s4 plan q of a single row block of
/// cm_int4_min_m rows or more, of tiles of 32 columns or more, whose slice
/// does not divide its column tiles over the threads: q, unless M blocks of
/// ceil(M / b) rows, b 2 to kInt4RowBlocks, each of at least tile_rows
/// rows, are modeled more than cm_gain_pct percent faster; then the
/// fastest, the fewer blocks on a tie.  The row blocks deal the tiles of
/// the slice more evenly.  mb, col_groups and the schedule pin what they
/// set.  The model's s4 tile describes the INT4 group kernel: the plans of
/// a weight of one group are weighed at its rows per call, and keep their
/// own.
native_plan_t int4_model_plan(const native_problem_t &p,
        const UarchParams &uarch, const heuristic_params_t &hp,
        const native_plan_t &q) {
    if (hp.cm_int4_min_m <= 0 || p.M < hp.cm_int4_min_m || hp.mb != 0
            || hp.col_groups >= 0 || hp.schedule != native_schedule_t::automatic
            || q.nr < kMaxTileCols || q.m_blocks != 1 || q.col_groups != 0
            || q.unit_tiles != 0 || q.n_tiles < q.threads
            || q.n_tiles % q.threads == 0)
        return q;
    const auto cost = [&](native_plan_t c) {
        c.int8_rows_nv1 = std::min(
                c.int8_rows_nv1, native_int4_group_ukernel_max_mr(1));
        c.int8_rows_nv2 = std::min(
                c.int8_rows_nv2, native_int4_group_ukernel_max_mr(2));
        return native_plan_cost(c, p, uarch, hp).cycles;
    };
    const double base = cost(q);
    if (!(base > 0)) return q;
    native_plan_t best = q;
    double bound = base * (100 - std::clamp(hp.cm_gain_pct, 0, 100)) / 100.0;
    for (int b = 2; b <= kInt4RowBlocks; ++b) {
        heuristic_params_t v = hp;
        v.mb = ceil_div(p.M, b);
        if (v.mb < hp.tile_rows) break;
        const native_plan_t c = rules_plan(p, uarch, v);
        if (c.m_blocks < 2 || validate_native_plan(c, p) != nullptr) continue;
        const double cycles = cost(c);
        if (cycles < bound) {
            best = c;
            bound = cycles;
        }
    }
    return best;
}

/// The cost model's pass budget for the s4 plan q of one group over K whose
/// row blocks take K passes at its budget (the int4_l1_pct share of L1,
/// native_int4_pass_quads at the widest call): the int8_l2_pct share of L2,
/// one pass where the group's slab fits it.  Passes of the L1 share add row
/// calls that cost more than they save once the weight is in cache, which
/// the model cannot tell from a weight streamed from memory.
native_plan_t int4_group_passes(const native_problem_t &p,
        const UarchParams &uarch, const heuristic_params_t &hp,
        const native_plan_t &q) {
    if (!p.int4_one_group()) return q;
    const int cw = native_int8_col_width(q, q.nr);
    const int rows = native_int8_call_rows(q, cw / 16);
    const int quads = ceil_div(p.K, 4);
    if (rows < 1 || native_int4_pass_quads(q, quads, cw, q.mb, rows) >= quads)
        return q;
    native_plan_t one = q;
    one.int8_pass_budget = std::max(q.int8_pass_budget,
            static_cast<int64_t>(uarch.l2_bytes) * hp.int8_l2_pct / 100);
    return one;
}

} // namespace

native_plan_t heuristic_plan(const native_problem_t &p,
        const UarchParams &uarch, const heuristic_params_t &hp) {
    const native_plan_t q = rules_plan(p, uarch, hp);
    if (hp.cost_model == 0) return q;
    if (p.int4())
        return int4_group_passes(
                p, uarch, hp, int4_model_plan(p, uarch, hp, q));
    return p.int8() ? int8_model_plan(p, uarch, hp, q)
                    : model_plan(p, uarch, hp, q);
}

const char *validate_native_plan(
        const native_plan_t &plan, const native_problem_t &p) {
    const native_plan_t &q = plan;
    const bool bf16 = p.bf16();
    const bool narrow = (q.nr == 16 || q.nr == 32) && q.nr <= p.pack_nr
            && p.pack_nr % q.nr == 0;
    const bool f16_ph_wide = q.kind == native_kernel_kind_t::f16_ph
            && p.pack_nr == kMaxTileCols && q.nr == 2 * kMaxTileCols;
    if (!narrow && !(bf16 && bf16_wide_width(q.nr, p.pack_nr)) && !f16_ph_wide)
        return "nr is not 16 or 32 dividing pack_nr, or for BF16 64 or 128, "
               "or for F16_PH two blocks";
    if (q.mb < 1 || q.mb > p.M || q.m_blocks != ceil_div(p.M, q.mb)
            || q.n_tiles < 1 || p.N % q.nr != 0 || q.n_tiles != p.N / q.nr)
        return "the tile grid does not match M, N, mb and nr";
    const native_kernel_registry_t &kernels = native_kernel_registry();
    if (q.mr < 1 || q.mr > kMaxCallRows) return "mr is outside 1 .. 12";
    if (bf16 && !kernels.covers(native_kernel_kind_t::bf16, q.mr, q.nr))
        return "no BF16 kernel for every row count 1 .. mr at nr";
    if (bf16 && q.kc < p.K && (q.kc < 2 || q.kc % 2 != 0))
        return "kc is below K and not an even number of at least 2";
    if (p.f32() && !kernels.covers(native_kernel_kind_t::f32, q.mr, q.nr))
        return "no FP32 kernel for every row count 1 .. mr at nr";
    const bool f16_ph = q.kind == native_kernel_kind_t::f16_ph;
    if (p.f16() && !f16_ph
            && !kernels.covers(native_kernel_kind_t::f16, q.mr, q.nr))
        return "no F16 kernel for every row count 1 .. mr at nr";
    if (f16_ph
            && (!p.f16()
                    || !kernels.covers(
                            native_kernel_kind_t::f16_ph, q.mr, q.nr)))
        return "no F16_PH kernel for every row count 1 .. mr at nr";
    if ((p.f32() || p.f16()) && q.kc < 1) return "kc is below 1";
    if (f16_ph && q.kc < p.K && q.kc % kNativeF16AccumBlockK != 0)
        return "kc is below K and not a whole number of F16_PH blocks";
    if (q.threads < 1 || q.threads > p.threads)
        return "threads is outside 1 .. the requested team";
    if (q.col_groups < 0
            || (q.col_groups > 0
                    && (q.threads % q.col_groups != 0
                            || q.n_tiles % q.col_groups != 0)))
        return "col_groups does not divide threads and n_tiles";
    if (q.unit_tiles < 0 || q.unit_tiles > q.n_tiles || q.domain_threads < 1
            || (q.unit_tiles > 0 && q.col_groups != 0))
        return "unit_tiles is outside 0 .. n_tiles, domain_threads is below "
               "1, or the blocks schedule has col_groups";
    if (q.row_unit < 1 || q.copy_block < 1 || q.copy_par_min < 0)
        return "row_unit, copy_block or copy_par_min is out of range";
    if (p.int8()) {
        if ((q.int8_cw_mask & kNativeInt8Cw16) == 0)
            return "int8_cw does not allow 16";
        for (const int nv : {1, 2}) {
            const int width = 16 * nv;
            const uint8_t bit = nv == 2 ? kNativeInt8Cw32 : kNativeInt8Cw16;
            if (width > q.nr || (q.int8_cw_mask & bit) == 0) continue;
            const int rows = native_int8_call_rows(q, nv);
            const bool served = p.int4()
                    ? p.int8_scheme == native_int8_scheme_t::per_group
                            && kernels.covers(native_kernel_kind_t::int4, rows,
                                    width, native_int4_scheme(p))
                    : kernels.covers_int8(rows, width, p.int8_scheme);
            if (rows < 1 || rows > kMaxCallRows || !served)
                return p.int4() ? "no INT4 kernel for the rows per call at an "
                                  "allowed width (the kernels of one group "
                                  "over K, or the group kernels)"
                                : "no INT8 kernel of the call's scheme for the "
                                  "rows per call at an allowed width";
        }
        if (q.int8_pass_quads < 0 || q.int8_pass_budget < 0)
            return "int8_pass_quads or int8_pass_budget is negative";
    }
    if (q.scratch_floats != native_plan_scratch_floats(q, p))
        return "scratch_floats does not match the tile";
    return nullptr;
}

size_t native_plan_scratch_floats(
        const native_plan_t &plan, const native_problem_t &p) {
    const bool passes = !p.int8();
    if (passes && plan.kc >= p.K) return static_cast<size_t>(plan.mr) * plan.nr;
    // BF16 and FP32 K passes keep the accumulators of every tile of a unit.
    const int tiles = passes ? std::max(1, plan.unit_tiles) : 1;
    return static_cast<size_t>(plan.mb) * plan.nr * tiles;
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

int native_int4_pass_quads(
        const native_plan_t &p, int group_quads, int cw, int rows, int max_mr) {
    // A K quad is 2 bytes of each s4 weight column and 4 of each source row.
    const int64_t quad_bytes = int64_t {2} * cw + int64_t {4} * max_mr;
    const int64_t budget = p.int8_pass_budget;
    if (rows <= max_mr || group_quads * quad_bytes <= budget)
        return group_quads;
    const int64_t per = std::max<int64_t>(2, budget / quad_bytes);
    const int64_t passes = (group_quads + per - 1) / per;
    const int64_t quads = (group_quads + passes - 1) / passes;
    return static_cast<int>(
            std::min<int64_t>(group_quads, (quads + 1) & ~int64_t {1}));
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
    if (!has_sources()) return heuristic_plan(p, uarch_, kBuiltInParams);
    // The checks and the warnings of the layers allocate; a call that finds
    // no memory for them gets the built-in plan.
    try {
        return resolve_layers(p);
    } catch (...) { return heuristic_plan(p, uarch_, kBuiltInParams); }
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
            = p.int8() && p.int8_scheme != native_int8_scheme_t::per_channel;
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
        return heuristic_plan(p, uarch_, kBuiltInParams);

    // The plan of a merged config, or nullptr with why set.
    std::string bad;
    const auto legal = [&](const native_config_t &cfg, native_plan_t &out) {
        if (!check_native_config(cfg, bad)) return bad.c_str();
        out = heuristic_plan(
                p, uarch_, apply_native_config(kBuiltInParams, cfg));
        return validate_native_plan(out, p);
    };
    native_plan_t plan;
    if (legal(merged, plan) != nullptr) {
        // Rebuild from the highest layer down: each layer stays only if the
        // plan with it under the layers kept above it is legal, so a lower
        // layer never removes a higher one.  No layer kept: the heuristic
        // plan, which is always legal.
        plan = heuristic_plan(p, uarch_, kBuiltInParams);
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
                            " threads=", p.threads, " ", dtype_name(p), ": ",
                            reason, "; dropped");
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
