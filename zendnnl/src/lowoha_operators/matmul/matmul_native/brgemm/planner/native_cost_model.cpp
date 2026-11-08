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

#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_cost_model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/native_kernel_registry.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

namespace {

/// Bytes of a BF16 element.
constexpr double kBf16Bytes = 2.0;

/// Columns of one vector of f32 accumulators.
constexpr int kVectorCols = 16;

/// What one tile of rows rows costs: its core cycles and, when the worker
/// streams its weights, the cycles its stream idles.
struct tile_cost_t {
    double core = 0;
    double idle = 0;
};

/// The rows of row group rg of row_groups, as native_partition_t takes them:
/// whole row units of unit rows.
int row_group_rows(int M, int unit, int rg, int row_groups) {
    const int64_t units = (static_cast<int64_t>(M) + unit - 1) / unit;
    const int64_t r0 = std::min<int64_t>(M, units * rg / row_groups * unit);
    const int64_t r1
            = std::min<int64_t>(M, units * (rg + 1) / row_groups * unit);
    return static_cast<int>(r1 - r0);
}

/// L3 domains the model counts the memory time of; the workers past the
/// last share its stream.
constexpr int kMaxDomains = 64;

/// Bytes of an INT8 weight or source element, and K of one step of the INT8
/// kernel (a quad).
constexpr double kInt8Bytes = 1.0;
constexpr int kInt8StepK = 4;

/// The kernel calls of a tile, priced per K step of the kernel: a BF16 pair
/// of K over 32-column slabs, or an INT8 quad over the plan's call width.
class tile_model_t {
public:
    tile_model_t(const native_plan_t &q, const native_problem_t &p,
            const UarchParams &uarch, const heuristic_params_t &hp)
        : q_(q), p_(p), hp_(hp), int8_(p.int8()) {
        pipes_ = hp.cm_fma_pipes > 0 ? hp.cm_fma_pipes
                                     : std::max(1, uarch.fma_ports);
        if (int8_) {
            slab_cols_ = native_int8_col_width(q, q.nr);
            mr_ = native_int8_call_rows(q, slab_cols_ / kVectorCols);
            k_steps_ = static_cast<double>(ceil_div(p.K, kInt8StepK));
            step_bytes_ = kInt8StepK * kInt8Bytes;
            src_bytes_ = kInt8Bytes;
            l2_share_ = static_cast<double>(uarch.l2_bytes) * hp.int8_l2_pct
                    / 100.0;
        } else {
            slab_cols_ = native_bf16_slab_cols(q.nr);
            mr_ = q.mr;
            kc_ = std::min(std::max(1, q.kc), p.K);
            passes_ = ceil_div(p.K, kc_);
            k_steps_ = (p.K + 1) / 2;
            step_bytes_ = 2 * kBf16Bytes;
            src_bytes_ = kBf16Bytes;
            l1_fit_ = static_cast<int64_t>(q.nr + q.mr) * kc_
                            * static_cast<int64_t>(kBf16Bytes) * 100
                    <= static_cast<int64_t>(uarch.l1d_bytes) * hp.l1_pct;
            l2_share_ = static_cast<double>(uarch.l2_bytes) * hp.bf16_l2_pct
                    / 100.0;
        }
        slabs_ = std::max(1, q.nr / std::max(1, slab_cols_));
        vectors_ = static_cast<double>(slab_cols_) / kVectorCols;
        l1_bytes_ = static_cast<int64_t>(uarch.l1d_bytes) * hp.l1_pct;
        mem_cycles_ = native_plan_mem_cycles(uarch, hp);
        tall_ = 1.0 + native_plan_tall_pct(uarch, hp) / 100.0;
        link_bound_ = mem_cycles_ > hp.cm_stream_cycles;
    }

    double mem_cycles() const { return mem_cycles_; }

    /// Tiles of a worker's tiles whose stream idle counts: all of them while
    /// a core streams no faster than its share of the domain's link, else
    /// the first cm_sync_tiles (the workers start in step; later, the other
    /// streams of the domain fill the link while one idles).
    double idle_tiles(int64_t tiles) const {
        return link_bound_ ? static_cast<double>(std::min<int64_t>(
                                     tiles, hp_.cm_sync_tiles))
                           : static_cast<double>(tiles);
    }

    /// Cycles of one call of rows rows over all of K, per slab; from_l2:
    /// the call re-reads the pass's slab from L2, at most cm_l2_bytes bytes
    /// a cycle.
    double call(int rows, bool from_l2) const {
        double fma = rows * vectors_ / pipes_;
        if (rows > hp_.cm_short_rows) fma *= tall_;
        double per = std::max(fma, static_cast<double>(hp_.cm_fma_lat));
        if (from_l2)
            per = std::max(per,
                    slab_cols_ * step_bytes_ / std::max(1, hp_.cm_l2_bytes));
        return per * k_steps_;
    }

    tile_cost_t tile(int rows) const {
        tile_cost_t t;
        if (rows <= 0) return t;
        // An INT8 tile's K pass follows its rows (native_int8_pass_quads).
        int passes = passes_;
        bool l1_fit = l1_fit_;
        if (int8_) {
            const int k_quads = ceil_div(p_.K, kInt8StepK);
            const int pass = std::max(1,
                    native_int8_pass_quads(q_, k_quads, slab_cols_, rows, mr_));
            passes = ceil_div(k_quads, pass);
            l1_fit = static_cast<int64_t>(slab_cols_ + mr_) * pass * kInt8StepK
                            * 100
                    <= l1_bytes_;
        }
        const int full = rows / mr_, rest = rows % mr_;
        const int calls = full + (rest > 0 ? 1 : 0);
        // The first call of a pass streams the slab; the later ones re-read
        // it from L1 when the pass fits, else from L2.
        const double first = call(full > 0 ? mr_ : rest, false);
        const double later = full > 0 ? (full - 1) * call(mr_, !l1_fit)
                        + (rest > 0 ? call(rest, !l1_fit) : 0.0)
                                      : 0.0;
        t.core = slabs_
                * (first + later
                        + static_cast<double>(passes) * calls
                                * hp_.cm_call_cycles);
        // The tile's source rows come back from L2 only when they fit with
        // the tile's slab over all of K.
        if ((rows + q_.nr) * static_cast<double>(p_.K) * src_bytes_ > l2_share_)
            t.core += rows * static_cast<double>(p_.K) * src_bytes_ / 1024.0
                    * hp_.cm_l3_cycles;
        const double per_pass = slabs_ * later / passes;
        t.idle = passes
                * std::max(0.0,
                        per_pass
                                - static_cast<double>(hp_.cm_prefetch_kib)
                                        * mem_cycles_);
        return t;
    }

    double combine(double core, double memory) const {
        return std::max(core, memory)
                + hp_.cm_overlap_pct / 100.0 * std::min(core, memory);
    }

private:
    const native_plan_t &q_;
    const native_problem_t &p_;
    const heuristic_params_t &hp_;
    bool int8_ = false;
    int slab_cols_ = 32;
    int slabs_ = 1;
    int mr_ = 1;
    double vectors_ = 2;
    int pipes_ = 2;
    int kc_ = 1;
    int passes_ = 1;
    double k_steps_ = 1;
    double step_bytes_ = 4;
    double src_bytes_ = 2;
    bool l1_fit_ = false;
    int64_t l1_bytes_ = 0;
    double l2_share_ = 0;
    double mem_cycles_ = 1;
    double tall_ = 1;
    bool link_bound_ = false;
};

} // namespace

int native_plan_domain_threads(
        const UarchParams &uarch, const heuristic_params_t &hp) {
    const int dt = hp.domain_threads > 0 ? hp.domain_threads
            : uarch.l3_domain_cores > 0  ? uarch.l3_domain_cores
                                         : uarch.family_l3_cores;
    return std::max(1, dt);
}

double native_plan_mem_cycles(
        const UarchParams &uarch, const heuristic_params_t &hp) {
    if (hp.cm_mem_cycles > 0) return hp.cm_mem_cycles;
    return 2400.0 / std::sqrt(native_plan_domain_threads(uarch, hp));
}

double native_plan_tall_pct(
        const UarchParams &uarch, const heuristic_params_t &hp) {
    if (hp.cm_tall_pct >= 0) return hp.cm_tall_pct;
    const double doublings
            = std::log2(native_plan_domain_threads(uarch, hp) / 8.0);
    return std::clamp(12.5 * doublings, 0.0, 50.0);
}

bool native_plan_core_streams(
        const UarchParams &uarch, const heuristic_params_t &hp) {
    return native_plan_mem_cycles(uarch, hp) < hp.cm_stream_cycles;
}

bool native_plan_cost_int8(const native_problem_t &p) {
    return p.int8() && p.int8_scheme == native_int8_scheme_t::per_channel;
}

bool native_plan_groups_fit_l3(const native_problem_t &p,
        const native_plan_t &q, const UarchParams &uarch,
        const heuristic_params_t &hp) {
    const int domains = ceil_div(
            std::max(1, q.threads), native_plan_domain_threads(uarch, hp));
    const double bytes = static_cast<double>(p.N) * p.K
            * (p.int8() ? kInt8Bytes : kBf16Bytes);
    return bytes * 100.0 <= static_cast<double>(uarch.l3_domain_bytes) * domains
            * hp.cm_group_l3_pct;
}

bool native_plan_slice_skewed(
        const native_plan_t &q, const heuristic_params_t &hp) {
    const int64_t threads = std::max(1, q.threads);
    const int64_t tiles = std::max(1, q.n_tiles);
    return ceil_div(tiles, threads) * threads * 100
            >= tiles * hp.cm_group_skew_pct;
}

native_plan_cost_t native_plan_cost(const native_plan_t &q,
        const native_problem_t &p, const UarchParams &uarch,
        const heuristic_params_t &hp) {
    native_plan_cost_t out;
    const bool int8 = native_plan_cost_int8(p);
    if ((!p.bf16() && !int8) || q.nr < 1 || q.mr < 1 || q.mb < 1
            || q.threads < 1 || q.n_tiles < 1 || q.m_blocks < 1)
        return out;
    if (int8
            && native_int8_call_rows(
                       q, native_int8_col_width(q, q.nr) / kVectorCols)
                    < 1)
        return out;
    const tile_model_t model(q, p, uarch, hp);
    const int M = p.M;
    const double tile_kib = static_cast<double>(q.nr) * p.K
            * (int8 ? kInt8Bytes : kBf16Bytes) / 1024.0;
    const int workers = q.threads;
    const int g = q.col_groups;
    const int dt = native_plan_domain_threads(uarch, hp);
    // Domain d holds threads d * dt .. (d + 1) * dt; the last counted domain
    // holds the rest.
    const int domains = std::min(ceil_div(workers, dt), kMaxDomains);
    const auto first_of = [&](int d) { return d * dt; };
    const auto end_of = [&](int d) {
        return d == domains - 1 ? workers : std::min(workers, (d + 1) * dt);
    };
    // The memory time of each domain: the weight KiB it streams, shared by
    // its workers.
    double domain_kib[kMaxDomains] = {};
    const auto domain_cycles = [&](int d) {
        return domain_kib[d] * model.mem_cycles()
                / std::max(1, end_of(d) - first_of(d));
    };
    const auto share_kib = [&](int d) {
        return domain_kib[d] / std::max(1, end_of(d) - first_of(d));
    };
    const auto keep = [&](double core, double memory, double stream, int d) {
        const double cycles = model.combine(core, memory);
        if (cycles > out.cycles) {
            out.core = core;
            out.memory = memory;
            out.cycles = cycles;
            out.stream = stream;
            out.stream_kib = share_kib(d);
        }
    };
    if (q.unit_tiles == 0 && g > 0 && workers % g == 0 && q.n_tiles % g == 0) {
        const int row_groups = workers / g;
        const int cols = q.n_tiles / g;
        const int unit = std::max(1, q.row_unit);
        const auto rows_of = [&](int rg) {
            return row_group_rows(M, unit, rg, row_groups);
        };
        // A domain streams each tile of the column groups its threads take
        // once while its share of the weights times cm_reuse_pct percent
        // fits its share of L3 (its other reads of the tile hit L3), else
        // once per thread that reads it.  The threads that read are those
        // of the row groups with rows: a range of whole row groups, whose
        // column groups (thread w takes w % g) are min(g, their count); a
        // domain of more than g of them reads some tile twice.
        const double weight_kib = static_cast<double>(q.n_tiles) * tile_kib;
        const double l3_share_kib = static_cast<double>(uarch.l3_domain_bytes)
                * hp.l3_pct / 100.0 / 1024.0;
        const bool reuse
                = weight_kib / ceil_div(workers, dt) * hp.cm_reuse_pct / 100.0
                <= l3_share_kib;
        bool rereads[kMaxDomains] = {};
        for (int d = 0; d < domains; ++d) {
            const int a = first_of(d), b = end_of(d);
            int reads = 0;
            for (int rg = a / g; rg * g < b; ++rg)
                if (rows_of(rg) > 0)
                    reads += std::min(b, (rg + 1) * g) - std::max(a, rg * g);
            domain_kib[d]
                    = (reuse ? std::min(g, reads) : reads) * cols * tile_kib;
            rereads[d] = reuse && reads > g;
        }
        for (int rg = 0; rg < row_groups; ++rg) {
            const int rows = rows_of(rg);
            if (rows <= 0) continue;
            double core = 0, idle = 0;
            for (int m0 = 0; m0 < rows; m0 += q.mb) {
                const tile_cost_t t = model.tile(std::min(q.mb, rows - m0));
                core += t.core;
                idle += t.idle;
            }
            // Every worker of the row group has the same work; in a domain
            // that reads a tile twice, the second reads come from L3.
            const double l3 = cols * tile_kib * hp.cm_l3_cycles;
            const int last_domain
                    = std::min(domains - 1, ((rg + 1) * g - 1) / dt);
            for (int d = std::min(domains - 1, rg * g / dt); d <= last_domain;
                    ++d) {
                const double reread = rereads[d] ? l3 : 0.0;
                keep(cols * core,
                        domain_cycles(d) + model.idle_tiles(cols) * idle
                                + reread,
                        domain_cycles(d) + reread, d);
            }
        }
        return out;
    }
    // The slice of the grid: worker w takes tiles tiles * w / workers ..
    // tiles * (w + 1) / workers of the m_inner order (row blocks inner);
    // each column it visits streams once.  The workers' ranges step by
    // per or per + 1 tiles, and a range's row blocks by whole columns, so
    // the loops divide nothing.
    const int last = M - (q.m_blocks - 1) * q.mb;
    const tile_cost_t full_tile = model.tile(q.mb);
    const tile_cost_t last_tile = model.tile(last);
    const int64_t mbs = q.m_blocks;
    const int64_t tiles = mbs * q.n_tiles;
    const bool n_inner = q.traversal == native_traversal_t::n_inner;
    const int64_t per = tiles / workers, extra = tiles % workers;
    const int64_t first_last = (mbs - 1) * q.n_tiles;
    struct share_t {
        int64_t count = 0, lasts = 0, cols = 0;
    };
    // The ranges of the workers in order; step(s) gives worker w's share.
    struct walk_t {
        int64_t t0 = 0, col = 0, row = 0, err = 0;
    };
    const int64_t col_step[2] = {per / mbs, (per + 1) / mbs};
    const int64_t row_step[2] = {per % mbs, (per + 1) % mbs};
    const auto step = [&](walk_t &c) {
        share_t s;
        c.err += extra;
        const int longer = c.err >= workers ? 1 : 0;
        if (longer) c.err -= workers;
        s.count = per + longer;
        int64_t col = c.col + col_step[longer];
        int64_t row = c.row + row_step[longer];
        if (row >= mbs) {
            row -= mbs;
            ++col;
        }
        if (s.count > 0) {
            if (!n_inner) {
                s.lasts = col - c.col;
                s.cols = (row == 0 ? col - 1 : col) - c.col + 1;
            } else {
                const int64_t t1 = c.t0 + s.count;
                s.lasts = std::max<int64_t>(0, t1 - std::max(c.t0, first_last));
                s.cols = std::min<int64_t>(q.n_tiles, s.count);
            }
        }
        c.t0 += s.count;
        c.col = col;
        c.row = row;
        return s;
    };
    walk_t c;
    for (int d = 0; d < domains; ++d)
        for (int w = first_of(d); w < end_of(d); ++w)
            domain_kib[d] += step(c).cols * tile_kib;
    c = walk_t {};
    for (int d = 0; d < domains; ++d) {
        const double memory = domain_cycles(d);
        for (int w = first_of(d); w < end_of(d); ++w) {
            const share_t s = step(c);
            if (s.count == 0) continue;
            const int64_t fulls = s.count - s.lasts;
            const double core
                    = fulls * full_tile.core + s.lasts * last_tile.core;
            // The first row block of each column streams the column's
            // weights.
            const double idle = model.idle_tiles(s.cols)
                    * (mbs > 1 || s.lasts == 0 ? full_tile.idle
                                               : last_tile.idle);
            keep(core, memory + idle, memory, d);
        }
    }
    return out;
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
