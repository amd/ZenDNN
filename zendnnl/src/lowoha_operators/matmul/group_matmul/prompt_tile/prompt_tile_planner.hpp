/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

/// @file prompt_tile_planner.hpp
/// @brief Pure tile planner for the ALGO 3 prompt tile executor.
///
/// Splits one grouped-matmul call into (expert, M-block, N-slice) tiles
/// and partitions the tile list into contiguous, equal-cost ranges, one
/// per thread team (a team is the set of threads sharing one L3).  No
/// OpenMP, no globals and no env reads, so it is unit-testable in
/// isolation; the executor supplies the knobs, cache sizes, kernel facts
/// and team sizes through `PlannerConfig`.
///
/// Tile order inside the list:
///   experts heaviest first, then per expert
///     L3 superblock (group of M-blocks whose A fits the team's L3 share)
///       N-slice (B slice sized to stay in L2)
///         M-block
/// so consecutive tiles share a B slice (it stays hot in L2 / L3 while the
/// team walks the M-blocks) and a heavy expert's A is re-read from L3, not
/// DRAM, once per slice.
///
/// An expert whose A outweighs its B (M > N), or that is heavy enough to
/// span teams (e.g. a shared expert over a whole chunk), is walked M-block
/// major instead: each team then reads only its share of A and re-reads
/// the L3-resident B, rather than every team streaming the whole A, and
/// the rows a team writes in the gate/up pass are the rows it reads back
/// in the down pass.

#ifndef ZENDNNL_GROUP_MATMUL_PROMPT_TILE_PLANNER_HPP
#define ZENDNNL_GROUP_MATMUL_PROMPT_TILE_PLANNER_HPP

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

#include "prompt_tile_tunables.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace prompt_tile {

/// Rows [m0, m0 + rows) x wide columns [n0, n0 + ncols) of expert `e`.
/// `n0` and `ncols` are multiples of the CK `pack_nr`.
struct Tile {
    int32_t e;
    int32_t m0;
    int32_t rows;
    int32_t n0;
    int32_t ncols;
};

/// Order of N-slices vs M-blocks inside one expert superblock (the values
/// of the `order` knob).
enum class TileOrder : int {
    kSliceMajor = 0, ///< N-slice outer, M-block inner (B-slice reuse).
    kBlockMajor = 1, ///< M-block outer, N-slice inner (A-block reuse).
    kAuto = 2, ///< Per expert: block major when its A outweighs its B.
};

/// One call's knobs plus the cache and kernel facts the executor derives
/// for it.  The defaults are a BF16 call at pack NR 32.
struct PlannerConfig {
    Tunables tun = defaults();
    int pack_nr = 32;
    /// CK max MR of the call (BF16: 8 at pack NR 32, 6 at 64).
    int max_mr = 8;
    int64_t l2_bytes = 1 << 20;
    int64_t l3_bytes_per_team = int64_t(32) << 20;
    /// `FamilyFacts` of the call, with `row_align` resolved.
    int a_bytes = 2;
    int wei_bits = 16;
    int min_mr = 4;
    int row_align = 8;
    /// Bytes the CK reads per weight column beyond its K weights.
    int64_t wei_col_extra_bytes = 0;
    int num_threads = 1;
};

/// Planner temporaries, kept with the plan so a reused `Plan` plans
/// without allocating.
struct PlanScratch {
    std::vector<std::pair<int64_t, int>> order;
    std::vector<int> rows, cols;
    std::vector<int64_t> cost_prefix;
};

/// Moves a scratch vector into a local for one call and back on exit.  The
/// hot loops then index a local that stores through `Plan &` cannot alias.
template <typename T>
class Borrowed {
public:
    explicit Borrowed(std::vector<T> &home) : home_(home), v_(std::move(home)) {
        v_.clear();
    }
    ~Borrowed() { home_ = std::move(v_); }
    Borrowed(const Borrowed &) = delete;
    Borrowed &operator=(const Borrowed &) = delete;
    std::vector<T> &get() { return v_; }

private:
    std::vector<T> &home_;
    std::vector<T> v_;
};

struct Plan {
    std::vector<Tile> tiles;
    /// Team `t` owns tiles [team_begin[t], team_begin[t + 1]).
    std::vector<int> team_begin;
    /// Representative sizes of the heaviest expert (for logging).
    int mb = 0;
    int ns = 0;
    int64_t total_cost = 0;
    /// max over teams of (team cost / team size) divided by the ideal
    /// per-thread cost; 1.0 means a perfect static split.
    double est_imbalance = 1.0;
    PlanScratch scratch;
};

// ── Blocking ─────────────────────────────────────────────────────────

/// Packed bytes of one weight column of depth `K`.
inline int64_t wei_col_bytes(int K, const PlannerConfig &cfg) {
    return static_cast<int64_t>(K) * cfg.wei_bits / 8 + cfg.wei_col_extra_bytes;
}

/// Resolved order for one expert.  `kAuto` picks M-block major when the
/// expert's A outweighs its B or it `spans_teams`, and only while its B
/// fits `bm_l3_pct` of the team's L3, so walking the M-blocks re-reads B
/// from L3 rather than DRAM.
inline TileOrder expert_order(
        int M, int N, int K, bool spans_teams, const PlannerConfig &cfg) {
    if (cfg.tun[kOrder] == 0) return TileOrder::kSliceMajor;
    if (cfg.tun[kOrder] == 1) return TileOrder::kBlockMajor;
    const int64_t a_bytes = static_cast<int64_t>(M) * K * cfg.a_bytes;
    const int64_t b_bytes = static_cast<int64_t>(N) * wei_col_bytes(K, cfg);
    return ((a_bytes > b_bytes || spans_teams)
                   && b_bytes
                           <= cfg.l3_bytes_per_team * cfg.tun[kBmL3Pct] / 100)
            ? TileOrder::kBlockMajor
            : TileOrder::kSliceMajor;
}

/// L2-derived N-slice width: the slice's B fills `ns_l2_pct` of L2.
inline int auto_ns(int K, int N, const PlannerConfig &cfg) {
    const int64_t col_bytes
            = std::max<int64_t>(wei_col_bytes(std::max(K, 1), cfg), 1);
    int64_t ns = (cfg.l2_bytes * cfg.tun[kNsL2Pct] / 100) / col_bytes;
    ns = (ns / cfg.pack_nr) * cfg.pack_nr;
    ns = std::max<int64_t>(ns, cfg.pack_nr);
    return static_cast<int>(std::min<int64_t>(ns, N));
}

/// L2-derived M-block rows: the block's A fills `mb_l2_pct` of L2, within
/// [`mb_min`, `mb_max`].
inline int auto_mb(int K, const PlannerConfig &cfg) {
    const int64_t row_bytes = std::max<int64_t>(
            static_cast<int64_t>(std::max(K, 1)) * cfg.a_bytes, 1);
    const int64_t align = std::max(cfg.row_align, 1);
    int64_t mb = (cfg.l2_bytes * cfg.tun[kMbL2Pct] / 100) / row_bytes;
    mb = (mb / align) * align;
    mb = std::min<int64_t>(
            std::max<int64_t>(mb, cfg.tun[kMbMin]), cfg.tun[kMbMax]);
    return static_cast<int>(mb);
}

/// Rows in the smallest of the MR calls the CK splits `rows` into.
inline int min_call_rows(int rows, int max_mr) {
    const int mr = std::max(max_mr, 1);
    return rows / std::max(1, (rows + mr - 1) / mr);
}

/// Row ranges for one expert: blocks of a common `row_align`-aligned size
/// close to `mb`; the last block takes the remainder and merges into its
/// neighbour while one of its MR calls would fall under `min_mr`.
inline void split_rows(
        int M, int mb, const PlannerConfig &cfg, std::vector<int> &starts) {
    starts.clear();
    if (M <= 0) return;
    if (M <= mb) {
        starts.push_back(0);
        starts.push_back(M);
        return;
    }
    const int align = std::max(cfg.row_align, 1);
    const int max_mr = cfg.max_mr;
    const int min_mr = cfg.min_mr;
    const int n0 = (M + mb - 1) / mb;
    int bs = (M + n0 - 1) / n0;
    bs = ((bs + align - 1) / align) * align;
    for (int m = 0; m < M; m += bs)
        starts.push_back(m);
    while (starts.size() > 1
            && min_call_rows(M - starts.back(), max_mr) < min_mr)
        starts.pop_back();
    starts.push_back(M);
}

/// Column ranges for one expert: `ceil(N / ns)` slices whose widths are
/// balanced whole multiples of `pack_nr`.
inline void split_cols(int N, int ns, int pack_nr, std::vector<int> &starts) {
    starts.clear();
    if (N <= 0) return;
    const int nblk = std::max(1, N / pack_nr);
    const int nslices
            = std::clamp((N + std::max(ns, 1) - 1) / std::max(ns, 1), 1, nblk);
    const int base = nblk / nslices;
    const int rem = nblk % nslices;
    int col = 0;
    for (int s = 0; s < nslices; ++s) {
        starts.push_back(col);
        col += (base + (s < rem ? 1 : 0)) * pack_nr;
    }
    starts.push_back(N);
}

/// Count of tiles a given (mb, ns) choice produces for the active experts.
inline int64_t count_tiles(
        const int *M, const int *N, int num_ops, int mb, int ns, int pack_nr) {
    int64_t total = 0;
    for (int e = 0; e < num_ops; ++e) {
        if (M[e] <= 0 || N[e] <= 0) continue;
        const int64_t mblk = (M[e] <= mb) ? 1 : (M[e] + mb - 1) / mb;
        const int64_t nblk = N[e] / pack_nr;
        const int64_t nsl = std::min<int64_t>(
                std::max<int64_t>(1, (N[e] + ns - 1) / ns), nblk);
        total += mblk * nsl;
    }
    return total;
}

// ── Load balance ─────────────────────────────────────────────────────

/// A tile's MACs plus `rb` rows' worth of B streaming, which keeps tiny-M
/// tiles from looking free to the team partition.
inline int64_t tile_cost(const Tile &t, int K, int rb) {
    return static_cast<int64_t>(t.ncols) * K * (t.rows + rb);
}

/// Cuts the tile list into contiguous team ranges, each team's cost in
/// proportion to its size: team t ends at the first tile whose prefix cost
/// (`cost_prefix[i]` = cost of tiles [0, i)) reaches the team's cumulative
/// share.  Boundaries are monotone and the last is the tile count.
inline void partition_teams(const std::vector<int64_t> &cost_prefix,
        const int *team_size, int n_teams, Plan &out) {
    const int ntiles = static_cast<int>(cost_prefix.size()) - 1;
    out.total_cost = cost_prefix.back();
    int64_t threads = 0;
    for (int t = 0; t < n_teams; ++t)
        threads += std::max(team_size[t], 0);
    if (threads <= 0) {
        for (int t = 1; t <= n_teams; ++t)
            out.team_begin[t] = ntiles;
        return;
    }
    int64_t seats = 0;
    double worst = 0.0;
    const double ideal_per_thread = static_cast<double>(out.total_cost)
            / static_cast<double>(threads);
    for (int t = 0; t < n_teams; ++t) {
        seats += std::max(team_size[t], 0);
        int end = ntiles;
        if (t + 1 < n_teams) {
            const int64_t target = static_cast<int64_t>(
                    static_cast<double>(out.total_cost) * seats / threads);
            end = static_cast<int>(std::lower_bound(cost_prefix.begin(),
                                           cost_prefix.end(), target)
                    - cost_prefix.begin());
            end = std::clamp(end, out.team_begin[t], ntiles);
        }
        out.team_begin[t + 1] = end;
        if (team_size[t] > 0 && ideal_per_thread > 0.0) {
            const double per_thread = static_cast<double>(cost_prefix[end]
                                              - cost_prefix[out.team_begin[t]])
                    / team_size[t];
            worst = std::max(worst, per_thread / ideal_per_thread);
        }
    }
    out.est_imbalance = worst > 0.0 ? worst : 1.0;
}

// ── Planner ──────────────────────────────────────────────────────────

/// Build the tile list and the per-team ranges.
///
/// `team_size[t]` is the thread count of team `t`; teams receive cost in
/// proportion to their size.  Every expert must have `N % pack_nr == 0`
/// (the CK contract `prepare_for_call` already enforced).  Experts with
/// `M <= 0` produce no tiles.  `out` is reused across calls without
/// shrinking its buffers.
inline void plan_tiles(const int *M, const int *N, const int *K, int num_ops,
        const PlannerConfig &cfg, const int *team_size, int n_teams,
        Plan &out) {
    out.tiles.clear();
    out.team_begin.assign(static_cast<size_t>(std::max(n_teams, 1)) + 1, 0);
    out.total_cost = 0;
    out.est_imbalance = 1.0;
    out.mb = 0;
    out.ns = 0;

    // Heaviest expert first, ties by index: (-flops, e) pairs in ascending
    // order.
    Borrowed<std::pair<int64_t, int>> order_buf(out.scratch.order);
    std::vector<std::pair<int64_t, int>> &order = order_buf.get();
    for (int e = 0; e < num_ops; ++e) {
        if (M[e] > 0 && N[e] > 0)
            order.emplace_back(-static_cast<int64_t>(M[e]) * N[e] * K[e], e);
    }
    if (order.empty() || n_teams <= 0) return;
    std::sort(order.begin(), order.end());

    // Sizes come from the heaviest expert's K; MoE calls share one (N, K)
    // per pass, and a heterogeneous call still gets valid (if less tuned)
    // tiles because every expert is split with the same mb / ns.
    const int lead = order.front().second;
    const int mb_knob = cfg.tun[kMb];
    const int ns_knob = cfg.tun[kNs];
    const int mb_min = cfg.tun[kMbMin];
    const int align = std::max(cfg.row_align, 1);
    int mb = mb_knob > 0 ? mb_knob : auto_mb(K[lead], cfg);
    int ns = ns_knob > 0 ? ns_knob : auto_ns(K[lead], N[lead], cfg);
    const int64_t want = static_cast<int64_t>(cfg.tun[kMinTiles])
            * std::max(cfg.num_threads, 1);
    while (count_tiles(M, N, num_ops, mb, ns, cfg.pack_nr) < want) {
        if (mb_knob <= 0 && mb > mb_min) {
            mb = std::max(mb_min, ((mb / 2) / align) * align);
        } else if (ns_knob <= 0 && ns > cfg.pack_nr) {
            ns = std::max(cfg.pack_nr, ((ns / 2) / cfg.pack_nr) * cfg.pack_nr);
        } else {
            break;
        }
    }
    out.mb = mb;
    out.ns = ns;

    double total_flops = 0.0;
    for (const auto &o : order)
        total_flops
                += static_cast<double>(M[o.second]) * N[o.second] * K[o.second];

    // An upper bound: tail merges only remove tiles.
    const auto max_tiles = static_cast<size_t>(
            count_tiles(M, N, num_ops, mb, ns, cfg.pack_nr));
    Borrowed<Tile> tiles_buf(out.tiles);
    Borrowed<int> rows_buf(out.scratch.rows), cols_buf(out.scratch.cols);
    Borrowed<int64_t> prefix_buf(out.scratch.cost_prefix);
    std::vector<Tile> &tiles = tiles_buf.get();
    std::vector<int> &rows = rows_buf.get();
    std::vector<int> &cols = cols_buf.get();
    std::vector<int64_t> &cost_prefix = prefix_buf.get();
    tiles.reserve(max_tiles);
    cost_prefix.reserve(max_tiles + 1);
    cost_prefix.push_back(0);

    const int rb = cfg.tun[kRb];
    const bool l3sb = cfg.tun[kL3sb] != 0;
    const int64_t sb_budget = cfg.l3_bytes_per_team * cfg.tun[kSbL3Pct] / 100;
    // An expert spans teams when it carries `span_pct` of one team's share.
    const double span_scale = 100.0 / cfg.tun[kSpanPct];
    for (const auto &o : order) {
        const int e = o.second;
        split_rows(M[e], mb, cfg, rows);
        split_cols(N[e], ns, cfg.pack_nr, cols);
        const int nblocks = static_cast<int>(rows.size()) - 1;
        const int nslices = static_cast<int>(cols.size()) - 1;

        int blocks_per_sb = nblocks;
        if (l3sb) {
            const int64_t a_bytes
                    = static_cast<int64_t>(M[e]) * K[e] * cfg.a_bytes;
            if (a_bytes > sb_budget && nblocks > 1) {
                const int64_t blk_bytes
                        = static_cast<int64_t>(rows[1]) * K[e] * cfg.a_bytes;
                blocks_per_sb = static_cast<int>(std::clamp<int64_t>(
                        sb_budget / std::max<int64_t>(blk_bytes, 1), 1,
                        nblocks));
            }
        }

        auto emit = [&](int b, int s) {
            const Tile t {e, rows[b], rows[b + 1] - rows[b], cols[s],
                    cols[s + 1] - cols[s]};
            tiles.push_back(t);
            cost_prefix.push_back(cost_prefix.back() + tile_cost(t, K[e], rb));
        };
        const bool spans_teams = n_teams > 1
                && span_scale * n_teams * M[e] * N[e] * K[e] >= total_flops;
        const TileOrder eo = expert_order(M[e], N[e], K[e], spans_teams, cfg);
        for (int sb = 0; sb < nblocks; sb += blocks_per_sb) {
            const int sb_end = std::min(nblocks, sb + blocks_per_sb);
            if (eo == TileOrder::kSliceMajor) {
                for (int s = 0; s < nslices; ++s)
                    for (int b = sb; b < sb_end; ++b)
                        emit(b, s);
            } else {
                for (int b = sb; b < sb_end; ++b)
                    for (int s = 0; s < nslices; ++s)
                        emit(b, s);
            }
        }
    }
    partition_teams(cost_prefix, team_size, n_teams, out);
}

} // namespace prompt_tile
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_PROMPT_TILE_PLANNER_HPP
