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
/// and partitions the tile list into contiguous ranges, one per thread team
/// (a team is the set of threads sharing one L3), each in proportion to its
/// team's size; a single dense layer may instead get one range per thread
/// inside its team's range (`plan_balanced`).  No
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
    /// `kernel_traits_t` of the call, with `row_align` resolved.
    int a_bytes = 2;
    int wei_bits = 16;
    int min_mr = 4;
    int row_align = 8;
    /// Bytes the CK reads per weight column beyond its K weights.
    int64_t wei_col_extra_bytes = 0;
    int num_threads = 1;
    /// The call is one dense layer (one BF16 expert): auto sizes give
    /// every team equal M-blocks and every thread equal N-slices.
    bool single_dense = false;
    /// The call is one dense layer split per thread (`plan_balanced`); takes
    /// precedence over `single_dense` when the layer fits it.
    bool balanced = false;
    /// K chunk the tiles accumulate in (`CallContext::prompt_kc`; 0: the
    /// full K).  The L2-derived M-block and N-slice sizes then hold one
    /// chunk of A and B rather than the whole depth.
    int kc = 0;
};

/// One team's column split in a per-thread plan (`plan_balanced`): `s`
/// slices of N's pack blocks, `c = ceil(nblk / s)` on the busiest, and
/// `q = nblk / s` whole blocks a thread with `rem = nblk % s` left over.
struct TeamCols {
    int s, c, q, rem;
};

/// Planner temporaries, kept with the plan so a reused `Plan` plans
/// without allocating.
struct PlanScratch {
    std::vector<std::pair<int64_t, int>> order;
    std::vector<int> rows, cols;
    std::vector<int64_t> cost_prefix;
    std::vector<TeamCols> team_cols;
    std::vector<int> team_rows;
    std::vector<int64_t> busy;
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
    /// Empty, or one slot per thread in team-major, rank-minor order: rank
    /// `r` of team `t` owns tiles [owner_begin[s], owner_begin[s + 1]) with
    /// `s` = (threads of teams before t) + r.  Slots of a team tile its range.
    std::vector<int> owner_begin;
    /// Representative sizes of the heaviest expert (for logging).
    int mb = 0;
    int ns = 0;
    int64_t total_cost = 0;
    /// Busiest thread's planned cost over the ideal per-thread cost: the
    /// costliest owner slot of a per-thread plan, else the max over teams of
    /// team cost / team size; 1.0 means a perfect static split.
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
    if (cfg.kc > 0) K = std::min(K, cfg.kc);
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
    if (cfg.kc > 0) K = std::min(K, cfg.kc);
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

/// Whether `dense_sizes` suits a single dense layer: every team gets at
/// least `mb_min` rows, every team thread at least one pack-width column
/// block, and the weights fit the L3 share a team walks M-block major
/// (`bm_l3_pct`).  A wide-N head (e.g. a 262144-column LM head), a narrow N
/// or a short M keeps the shrink loop's many small tiles.
inline bool dense_fits(int M, int N, int K, const PlannerConfig &cfg,
        int n_teams, int team_max) {
    const int64_t b_bytes = static_cast<int64_t>(N) * wei_col_bytes(K, cfg);
    return M >= static_cast<int64_t>(n_teams) * cfg.tun[kMbMin]
            && N / std::max(cfg.pack_nr, 1) >= team_max
            && b_bytes <= cfg.l3_bytes_per_team * cfg.tun[kBmL3Pct] / 100;
}

/// Auto sizes of a single dense layer.  A dense layer's tiles all cost the
/// same, so instead of many small tiles drained dynamically, M splits into
/// one run of equal blocks per team (each block's A within the L2 share
/// `auto_mb` allows) and N into one equal slice per team thread: every
/// thread then gets the same work and no team waits on a last tile.
inline void dense_sizes(int M, int N, int K, const PlannerConfig &cfg,
        const int *team_size, int n_teams, int &mb, int &ns) {
    const int align = std::max(cfg.row_align, 1);
    const int per_team = (M + n_teams - 1) / n_teams;
    const int cap = std::max(auto_mb(K, cfg), align);
    const int runs = (per_team + cap - 1) / cap;
    mb = (per_team + runs - 1) / runs;
    mb = std::max(((mb + align - 1) / align) * align, align);
    int team_max = 1;
    for (int t = 0; t < n_teams; ++t)
        team_max = std::max(team_max, team_size[t]);
    const int nblk = std::max(1, N / cfg.pack_nr);
    const int slices = std::min(nblk, team_max);
    // `split_cols` cuts ceil(N / ns) balanced slices.
    ns = (N + slices - 1) / slices;
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

/// Busiest thread's cost of `plan`, with tiles indivisible: owner slots are
/// summed as planned; a team without slots is list-scheduled in tile order
/// over its threads (`busy` is scratch).
inline int64_t plan_busiest(const Plan &plan, const int *K, int rb,
        const int *team_size, int n_teams, std::vector<int64_t> &busy) {
    const auto cost = [&](int i) {
        const Tile &tl = plan.tiles[static_cast<size_t>(i)];
        return tile_cost(tl, K[tl.e], rb);
    };
    int64_t worst = 0;
    if (!plan.owner_begin.empty()) {
        for (size_t s = 0; s + 1 < plan.owner_begin.size(); ++s) {
            int64_t c = 0;
            for (int i = plan.owner_begin[s]; i < plan.owner_begin[s + 1]; ++i)
                c += cost(i);
            worst = std::max(worst, c);
        }
        return worst;
    }
    for (int t = 0; t < n_teams; ++t) {
        busy.assign(static_cast<size_t>(std::max(team_size[t], 1)), 0);
        for (int i = plan.team_begin[t]; i < plan.team_begin[t + 1]; ++i)
            *std::min_element(busy.begin(), busy.end()) += cost(i);
        worst = std::max(worst, *std::max_element(busy.begin(), busy.end()));
    }
    return worst;
}

/// Busiest thread's cost of `plan` over the ideal per-thread cost
/// (`plan_busiest`).
inline double plan_makespan(const Plan &plan, const int *K, int rb,
        const int *team_size, int n_teams) {
    if (plan.tiles.empty() || n_teams <= 0) return 1.0;
    int64_t threads = 0;
    for (int t = 0; t < n_teams; ++t)
        threads += std::max(team_size[t], 0);
    if (threads <= 0 || plan.total_cost <= 0) return 1.0;
    std::vector<int64_t> busy;
    return static_cast<double>(
                   plan_busiest(plan, K, rb, team_size, n_teams, busy))
            * static_cast<double>(threads)
            / static_cast<double>(plan.total_cost);
}

/// Per-thread plan of one dense layer.  Each of the `s_t` threads of team
/// `t` owns one balanced N-slice of its team's rows, or, where that leaves
/// the busiest thread more work, `nblk / s_t` whole column blocks of the
/// rows and one `s_t`-th of the rows of the `nblk % s_t` blocks left over.
/// Each team takes the most row units its busiest thread can carry under
/// the smallest cost bound that holds every unit, and the last few units go
/// where the busiest thread grows least, so every thread of every team
/// carries close to `M * N / T` work at any thread count (the dense equal
/// split gives a 4-thread team of a 31-thread call a second whole tile).
/// A team's rows split into runs of at most `auto_mb(K)` plus a quarter.
/// Returns false, planning nothing, when the layer does not fit
/// (`dense_fits`) or no team can take the rows left past whole units
/// without an MR call under `min_mr`.
inline bool plan_balanced(int e, int M, int N, int K, const PlannerConfig &cfg,
        const int *team_size, int n_teams, Plan &out) {
    const int pack = std::max(cfg.pack_nr, 1);
    const int nblk = N / pack;
    if (M <= 0 || nblk <= 0 || n_teams <= 0) return false;
    int team_max = 1;
    int threads = 0;
    for (int t = 0; t < n_teams; ++t) {
        team_max = std::max(team_max, team_size[t]);
        threads += std::max(team_size[t], 0);
    }
    if (threads <= 0 || !dense_fits(M, N, K, cfg, n_teams, team_max))
        return false;

    // Row unit: the family's alignment, halved for a short M while it stays
    // at or above `min_mr` (the split's rounding error is one unit per team).
    int unit = std::max(cfg.row_align, 1);
    if (unit % 2 == 0 && unit / 2 >= std::max(cfg.min_mr, 1)
            && M / unit < 16 * n_teams)
        unit /= 2;
    // Runs may exceed `auto_mb(K)` by a quarter: with the cap exact, an even
    // share a unit past a multiple of it pays one more B pass on some teams.
    const int cap0 = std::max(unit, (auto_mb(K, cfg) / unit) * unit);
    const int cap = cap0 + cap0 / 4;
    const int rb = cfg.tun[kRb];

    Borrowed<TeamCols> cols_buf(out.scratch.team_cols);
    Borrowed<int> rows_buf(out.scratch.team_rows), runs_buf(out.scratch.rows),
            pieces_buf(out.scratch.cols);
    std::vector<TeamCols> &tc = cols_buf.get();
    std::vector<int> &rows = rows_buf.get();
    std::vector<int> &runs = runs_buf.get();
    std::vector<int> &pieces = pieces_buf.get();
    tc.resize(n_teams);
    rows.assign(n_teams, 0);
    for (int t = 0; t < n_teams; ++t) {
        const int s = std::clamp(team_size[t], 0, nblk);
        tc[t] = s > 0 ? TeamCols {s, (nblk + s - 1) / s, nblk / s, nblk % s}
                      : TeamCols {0, 0, 0, 0};
    }
    // A team whose thread count does not divide `nblk` lays its blocks out
    // in one of two ways, whichever leaves its busiest thread less work:
    // `c`-block slices, or `q` whole blocks per thread plus a `ceil(r / s)`-
    // row piece of the `rem` blocks left over (a short piece pays its `rb`
    // for few rows).  Costs are in column blocks x rows, `rb` per tile
    // row-run.
    PlannerConfig piece_cfg = cfg;
    piece_cfg.row_align = unit;
    const auto piece_rows = [&](int r, int s) {
        const int n0 = (r + (r + s - 1) / s - 1) / ((r + s - 1) / s);
        return ((r + n0 - 1) / n0 + unit - 1) / unit * unit;
    };
    // Runs `split_rows(r, cap)` cuts r rows into, its tail merge aside: at
    // most `ceil(r / cap)`, fewer where its aligned block size rounds up.
    const int align = std::max(cfg.row_align, 1);
    const auto run_count = [&](int r) {
        if (r <= cap) return 1;
        const int n0 = (r + cap - 1) / cap;
        const int bs = ((r + n0 - 1) / n0 + align - 1) / align * align;
        return (r + bs - 1) / bs;
    };
    const auto slices_cost = [&](int t, int r) -> int64_t {
        return static_cast<int64_t>(tc[t].c) * (r + rb * run_count(r));
    };
    const auto split_cost = [&](int t, int r, int piece) -> int64_t {
        return static_cast<int64_t>(tc[t].q) * (r + rb * run_count(r))
                + static_cast<int64_t>(tc[t].rem) * (piece + rb);
    };
    // Busiest-thread cost of team t at r rows.
    const auto load = [&](int t, int r) -> int64_t {
        if (r <= 0) return 0;
        const int64_t whole = slices_cost(t, r);
        return tc[t].rem == 0
                ? whole
                : std::min(whole, split_cost(t, r, piece_rows(r, tc[t].s)));
    };
    const int units = M / unit;
    // Most whole units team t takes with its busiest thread at cost <= L,
    // counting `ceil(r / cap)` runs (at least `run_count(r)`): with k runs,
    // r <= min(k * cap, L / c - rb * k) as slices of c blocks, or, as whole
    // blocks and a piece of at most `r / s + unit` rows,
    // r <= min(k * cap, (L - rb * (q * k + rem) - rem * unit) * s / nblk).
    const auto units_within = [&](int t, int64_t L) -> int {
        const int64_t s = tc[t].s;
        if (s == 0) return 0;
        const int64_t c = tc[t].c, q = tc[t].q, rem = tc[t].rem;
        int64_t best = 0;
        for (int64_t k = 1;; ++k) {
            const int64_t as_slices = L / c - rb * k;
            const int64_t as_split = rem == 0
                    ? as_slices
                    : (L - rb * (q * k + rem) - rem * unit) * s / nblk;
            const int64_t r
                    = std::min<int64_t>(k * cap, std::max(as_slices, as_split));
            if (r <= (k - 1) * cap) break;
            best = std::max(best, r / unit);
            if (r < k * cap || best >= units) break;
        }
        return static_cast<int>(std::min<int64_t>(best, units));
    };
    const auto units_at = [&](int64_t L) {
        int64_t s = 0;
        for (int t = 0; t < n_teams; ++t)
            s += units_within(t, L);
        return s;
    };
    // Planned on every call, so the rows come from a search on the cost, not
    // a unit-by-unit fill: the smallest busiest-thread cost that holds every
    // unit, then each team at the cost just below it; the few units left go
    // one at a time where the busiest thread grows least.  A run's fixed
    // `rb` cost would make a proportional split overshoot a small team (a
    // 1-thread team of a 33-thread call holds all column blocks).
    // A team carries at most `L * s / nblk` rows at cost L, so a cost below
    // `units * unit * nblk / threads` cannot hold every unit; the search
    // starts there and widens until a cost holds them.
    int64_t cap_cost = 0;
    const int64_t all_rows = static_cast<int64_t>(units) * unit;
    for (int t = 0; t < n_teams; ++t)
        if (tc[t].s > 0)
            cap_cost = std::max(cap_cost,
                    static_cast<int64_t>(tc[t].c)
                            * (all_rows + rb * ((all_rows + cap - 1) / cap)));
    int64_t lo = std::max<int64_t>(
            0, static_cast<int64_t>(units) * unit * nblk / threads - 1);
    int64_t step = std::max<int64_t>(lo / 16, 1);
    int64_t hi = std::min(cap_cost, lo + step);
    while (hi < cap_cost && units_at(hi) < units) {
        lo = hi;
        step *= 2;
        hi = std::min(cap_cost, lo + step);
    }
    while (lo + 1 < hi) {
        const int64_t mid = lo + (hi - lo) / 2;
        (units_at(mid) >= units ? hi : lo) = mid;
    }
    int given = 0;
    for (int t = 0; t < n_teams; ++t) {
        rows[t] = units_within(t, lo) * unit;
        given += rows[t] / unit;
    }
    if (given < units) {
        // Each team's cost with one more unit; only the taker's changes.
        Borrowed<int64_t> next_buf(out.scratch.cost_prefix);
        std::vector<int64_t> &next = next_buf.get();
        next.resize(n_teams);
        for (int t = 0; t < n_teams; ++t)
            next[t] = tc[t].s > 0 ? load(t, rows[t] + unit) : INT64_MAX;
        for (; given < units; ++given) {
            const int best = static_cast<int>(
                    std::min_element(next.begin(), next.end()) - next.begin());
            rows[best] += unit;
            next[best] = load(best, rows[best] + unit);
        }
    }
    // Every run of a team's rows keeps its MR calls at `min_mr` or spans the
    // expert's M (`split_rows` cannot split a run its aligned block covers).
    const auto rows_ok = [&](int r) {
        if (r == M) return true;
        if (r <= cap) return min_call_rows(r, cfg.max_mr) >= cfg.min_mr;
        split_rows(r, cap, cfg, runs);
        for (size_t b = 0; b + 1 < runs.size(); ++b) {
            const int n = runs[b + 1] - runs[b];
            if (n != M && min_call_rows(n, cfg.max_mr) < cfg.min_mr)
                return false;
        }
        return true;
    };
    if (const int left = M - units * unit; left > 0) {
        int best = -1;
        int64_t best_load = 0;
        for (int t = 0; t < n_teams; ++t) {
            if (tc[t].s == 0 || (units > 0 && rows[t] <= 0)
                    || !rows_ok(rows[t] + left))
                continue;
            const int64_t l = load(t, rows[t] + left);
            if (best < 0 || l < best_load) {
                best = t;
                best_load = l;
            }
        }
        if (best < 0) return false;
        rows[best] += left;
    }
    for (int t = 0; t < n_teams; ++t)
        if (rows[t] > 0 && !rows_ok(rows[t])) return false;

    out.tiles.clear();
    out.owner_begin.assign(static_cast<size_t>(threads) + 1, 0);
    out.mb = 0;
    out.ns = 0;
    int m0 = 0;
    int slot = 0;
    int64_t total = 0;
    int64_t worst = 0;
    for (int t = 0; t < n_teams; ++t) {
        out.team_begin[t] = static_cast<int>(out.tiles.size());
        const auto [s, c, q, rem] = tc[t];
        const int R = rows[t];
        split_rows(R, cap, cfg, runs);
        pieces.clear();
        if (rem > 0 && R > 0) {
            split_rows(R, (R + s - 1) / s, piece_cfg, pieces);
            int piece = 0;
            for (size_t i = 0; i + 1 < pieces.size(); ++i)
                piece = std::max(piece, pieces[i + 1] - pieces[i]);
            // Busiest thread: (q + 1) blocks of every run as slices, against
            // q blocks plus `rem` blocks of its piece.
            const int64_t nruns = static_cast<int64_t>(runs.size()) - 1;
            if (static_cast<int64_t>(rem) * (piece + rb) >= R + rb * nruns)
                pieces.clear();
        }
        // With pieces, thread r owns blocks [r * q, (r + 1) * q) and piece r
        // of blocks [s * q, nblk); otherwise balanced slices of q or q + 1.
        int n0 = 0;
        for (int r = 0; r < team_size[t]; ++r, ++slot) {
            out.owner_begin[slot] = static_cast<int>(out.tiles.size());
            if (r >= s || R <= 0) continue;
            int64_t mine = 0;
            const auto emit = [&](int m, int nrows, int c0, int ncols) {
                const Tile tl {e, m0 + m, nrows, c0, ncols};
                out.tiles.push_back(tl);
                mine += tile_cost(tl, K, rb);
                out.mb = std::max(out.mb, nrows);
                out.ns = std::max(out.ns, ncols);
            };
            const int ncols
                    = (pieces.empty() ? q + (r < rem ? 1 : 0) : q) * pack;
            for (size_t b = 0; b + 1 < runs.size(); ++b)
                emit(runs[b], runs[b + 1] - runs[b], n0, ncols);
            n0 += ncols;
            if (static_cast<size_t>(r) + 1 < pieces.size())
                emit(pieces[r], pieces[r + 1] - pieces[r], s * q * pack,
                        rem * pack);
            total += mine;
            worst = std::max(worst, mine);
        }
        m0 += R;
    }
    out.owner_begin[slot] = static_cast<int>(out.tiles.size());
    out.team_begin[n_teams] = static_cast<int>(out.tiles.size());
    out.total_cost = total;
    out.est_imbalance = total > 0
            ? static_cast<double>(worst) * threads / static_cast<double>(total)
            : 1.0;
    return true;
}

// ── Planner ──────────────────────────────────────────────────────────

/// The tile list and per-team ranges a team's threads drain together: the
/// dense split of one dense layer (`single_dense`), else the shrink loop's
/// tiles.  See `plan_tiles`.
inline void plan_team_tiles(const int *M, const int *N, const int *K,
        int num_ops, const PlannerConfig &cfg, const int *team_size,
        int n_teams, Plan &out) {
    out.tiles.clear();
    out.team_begin.assign(static_cast<size_t>(std::max(n_teams, 1)) + 1, 0);
    out.owner_begin.clear();
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
    int team_max = 1;
    for (int t = 0; t < n_teams; ++t)
        team_max = std::max(team_max, team_size[t]);
    const bool dense = cfg.single_dense && order.size() == 1 && mb_knob <= 0
            && ns_knob <= 0
            && dense_fits(M[lead], N[lead], K[lead], cfg, n_teams, team_max);
    if (dense)
        dense_sizes(M[lead], N[lead], K[lead], cfg, team_size, n_teams, mb, ns);
    const int64_t want = static_cast<int64_t>(cfg.tun[kMinTiles])
            * std::max(cfg.num_threads, 1);
    while (!dense && count_tiles(M, N, num_ops, mb, ns, cfg.pack_nr) < want) {
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

/// Build the tile list and the per-team ranges.
///
/// `team_size[t]` is the thread count of team `t`; teams receive cost in
/// proportion to their size.  Every expert must have `N % pack_nr == 0`
/// (the CK contract `prepare_for_call` already enforced).  Experts with
/// `M <= 0` produce no tiles.  `out` is reused across calls without
/// shrinking its buffers.  Under `cfg.balanced` one active expert without a
/// forced block size gets the per-thread plan (`plan_balanced`) where its
/// busiest thread carries no more than the team plan's: a team draining its
/// tiles can interleave uneven slice widths that fixed per-thread slices
/// cannot.
inline void plan_tiles(const int *M, const int *N, const int *K, int num_ops,
        const PlannerConfig &cfg, const int *team_size, int n_teams,
        Plan &out) {
    plan_team_tiles(M, N, K, num_ops, cfg, team_size, n_teams, out);
    if (!cfg.balanced || cfg.tun[kMb] > 0 || cfg.tun[kNs] > 0) return;
    int lead = -1;
    for (int e = 0; e < num_ops; ++e) {
        if (M[e] <= 0 || N[e] <= 0) continue;
        if (lead >= 0) return;
        lead = e;
    }
    if (lead < 0 || out.tiles.empty()) return;
    const int rb = cfg.tun[kRb];
    Borrowed<int64_t> busy_buf(out.scratch.busy);
    std::vector<int64_t> &busy = busy_buf.get();
    const int64_t shared = plan_busiest(out, K, rb, team_size, n_teams, busy);
    if (!plan_balanced(
                lead, M[lead], N[lead], K[lead], cfg, team_size, n_teams, out))
        return;
    if (plan_busiest(out, K, rb, team_size, n_teams, busy) > shared)
        plan_team_tiles(M, N, K, num_ops, cfg, team_size, n_teams, out);
}

} // namespace prompt_tile
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_PROMPT_TILE_PLANNER_HPP
