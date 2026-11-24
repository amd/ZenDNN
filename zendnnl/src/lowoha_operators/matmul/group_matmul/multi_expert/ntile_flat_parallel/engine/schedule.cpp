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

#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/engine/schedule.hpp"

#include <algorithm>
#include <cmath>

#include "lowoha_operators/matmul/group_matmul/common/grp_matmul_machine.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/ntile_flat_parallel.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {
namespace detail {

namespace {

/// About `target` pieces of `groups` over the call, `g`'s share in proportion
/// to its cost (weight bytes streamed plus rows times weight elements), at
/// most `room`.
int group_pieces(const block_group_t &g, const double total,
        const float wei_bytes, const int target, const int room) {
    const double cost = 16.0 * wei_bytes + g.rows;
    return std::clamp(
            static_cast<int>(std::lround(target * cost / total)), 1, room);
}

double groups_cost(
        const std::vector<block_group_t> &groups, const float wei_bytes) {
    double total = 0.0;
    for (const block_group_t &g : groups)
        total += 16.0 * wei_bytes + g.rows;
    return total;
}

/// Cuts each group into runs of whole blocks: with `split_rows`, as many as
/// keep the source rows and the weights its units read about equal (each unit
/// reads its rows' source and its columns' weights, so sqrt(pieces * rows / N)
/// runs balance the two), else one.
void plan_runs(std::vector<row_run_t> &runs, std::vector<block_group_t> &groups,
        const int N, const int pack_nr, const float wei_bytes, const int target,
        const bool split_rows) {
    const int col_blocks = N / pack_nr;
    const double total = groups_cost(groups, wei_bytes);
    runs.clear();
    for (block_group_t &g : groups) {
        int cuts = 1;
        if (split_rows) {
            const int pieces = group_pieces(
                    g, total, wei_bytes, target, col_blocks * g.blocks);
            cuts = std::clamp(
                    static_cast<int>(std::lround(std::sqrt(
                            static_cast<double>(pieces) * g.rows / N))),
                    1, g.blocks);
        }
        g.run = static_cast<int>(runs.size());
        g.runs = cuts;
        for (int q = 0; q < cuts; ++q) {
            const int r0 = g.block + g.blocks * q / cuts;
            const int r1 = g.block + g.blocks * (q + 1) / cuts;
            runs.push_back({r0, r1 - r0});
        }
    }
}

/// Appends, group by group in the order of `groups`, each group's `N` columns
/// cut into whole `pack_nr` blocks and crossed with runs of its row runs,
/// about `target` units over the call in proportion to each group's cost.
/// With `merge_runs`, a unit covers as many of the group's row runs as keep
/// its source rows and weights balanced for this projection's `N` (the row
/// runs were cut for the W13 projection), else one.
void plan_units(std::vector<unit_t> &out,
        const std::vector<block_group_t> &groups, const int N,
        const int pack_nr, const float wei_bytes, const int target,
        const bool merge_runs) {
    const int col_blocks = N / pack_nr;
    const double total = groups_cost(groups, wei_bytes);
    for (const block_group_t &g : groups) {
        int cuts = g.runs;
        if (merge_runs) {
            const int pieces = group_pieces(
                    g, total, wei_bytes, target, col_blocks * g.runs);
            cuts = std::clamp(
                    static_cast<int>(std::lround(std::sqrt(
                            static_cast<double>(pieces) * g.rows / N))),
                    1, g.runs);
        }
        const int pieces
                = group_pieces(g, total, wei_bytes, target, col_blocks * cuts);
        const int col_cuts
                = std::clamp((pieces + cuts - 1) / cuts, 1, col_blocks);
        for (int p = 0; p < col_cuts; ++p) {
            const int b0 = col_blocks * p / col_cuts;
            const int b1 = col_blocks * (p + 1) / col_cuts;
            for (int q = 0; q < cuts; ++q) {
                const int r0 = g.runs * q / cuts;
                const int r1 = g.runs * (q + 1) / cuts;
                out.push_back({g.run + r0, r1 - r0, b0 * pack_nr,
                        (b1 - b0) * pack_nr});
            }
        }
    }
}

/// Reorders `n` units planned heaviest group first to take turns from the two
/// ends of the plan, so the heaviest groups' units, which keep the cores busy,
/// run beside the lightest groups', which mostly stream weights.
void take_turns(unit_t *units, const int n, std::vector<unit_t> &planned) {
    planned.assign(units, units + n);
    for (int k = 0; k < n; ++k)
        units[k] = planned[k % 2 == 0 ? k / 2 : n - 1 - k / 2];
}

/// The widest run of whole `step` columns whose weights fit `l2_budget`
/// bytes beside one block's rows of `K` elements of `es` bytes.
int l2_chunk_cols(const int K, const size_t es, const float wei_bytes,
        const int step, const int64_t l2_budget) {
    const double room = static_cast<double>(l2_budget)
            - static_cast<double>(kDecodeMaxM) * K * es;
    const int cols = static_cast<int>(room / (K * wei_bytes + 4.0));
    return std::max(cols / step * step, step);
}

} // namespace

queue_t plan_queue(scratch_t &S, const int *rows, const int n,
        const nr32::tile_plan &p13, const int K13, const nr32::tile_plan *p2,
        const int K2, const size_t es, const float wei_bytes,
        const int num_threads, const grp_matmul_phase phase,
        const call_knobs &k, const int64_t l2_budget) {
    const bool prompt = phase == grp_matmul_phase::prompt;
    const bool sweep = *std::max_element(rows, rows + n) > k.sweep_rows;
    S.blocks.clear();
    S.groups.clear();
    for (int e = 0; e < n; ++e) {
        const int first = static_cast<int>(S.blocks.size());
        for (int r = 0; r < rows[e]; r += kDecodeMaxM) {
            S.blocks.push_back(
                    {e, S.offset[e] + r, std::min(kDecodeMaxM, rows[e] - r)});
            if (!sweep) {
                S.groups.push_back({static_cast<int>(S.blocks.size()) - 1, 1,
                        S.blocks.back().rows});
            }
        }
        if (sweep && rows[e] > 0) {
            S.groups.push_back({first,
                    static_cast<int>(S.blocks.size()) - first, rows[e]});
        }
    }
    std::stable_sort(S.groups.begin(), S.groups.end(),
            [](const block_group_t &a, const block_group_t &b) {
        return a.rows > b.rows;
    });
    const int target = (prompt ? k.prompt_units : k.units) * num_threads;
    queue_t q;
    q.prompt = prompt;
    q.sweep = sweep;
    q.order = k.order;
    q.chunk13 = sweep ? l2_chunk_cols(K13, es, wei_bytes, p13.step, l2_budget)
                      : p13.cols;
    plan_runs(S.runs, S.groups, p13.cols, p13.step, wei_bytes, target, sweep);
    q.num_runs = static_cast<int>(S.runs.size());
    S.queue.clear();
    plan_units(S.queue, S.groups, p13.cols, p13.step, wei_bytes, target,
            /*merge_runs=*/false);
    q.n13 = static_cast<int>(S.queue.size());
    if (p2 != nullptr) {
        q.chunk2 = sweep ? l2_chunk_cols(K2, es, wei_bytes, p2->step, l2_budget)
                         : p2->cols;
        plan_units(S.queue, S.groups, p2->cols, p2->step, wei_bytes, target,
                sweep);
    }
    q.n_units = static_cast<int>(S.queue.size());
    if (sweep && k.heavy_light) {
        take_turns(S.queue.data(), q.n13, S.planned);
        take_turns(S.queue.data() + q.n13, q.n_units - q.n13, S.planned);
    }
    if (S.left.size() < static_cast<size_t>(q.num_runs)) {
        const size_t runs = static_cast<size_t>(q.num_runs);
        S.left = std::vector<std::atomic<int>>(runs);
        S.ready = std::vector<std::atomic<int>>(runs);
        S.gather_next = std::vector<std::atomic<int>>(runs);
        S.gather_done = std::vector<std::atomic<int>>(runs);
    }
    for (int r = 0; r < q.num_runs; ++r) {
        S.left[r].store(0, std::memory_order_relaxed);
        S.ready[r].store(0, std::memory_order_relaxed);
        S.gather_next[r].store(0, std::memory_order_relaxed);
        S.gather_done[r].store(0, std::memory_order_relaxed);
    }
    for (int u = 0; u < q.n13; ++u)
        S.left[S.queue[u].run].fetch_add(1, std::memory_order_relaxed);
    if (S.claimed.size() < static_cast<size_t>(q.n_units)) {
        S.claimed = std::vector<std::atomic<uint8_t>>(
                static_cast<size_t>(q.n_units));
    }
    for (int u = 0; u < q.n_units; ++u)
        S.claimed[u].store(0, std::memory_order_relaxed);
    if (test_api::s_capture_ntile_flat_parallel.load(
                std::memory_order_relaxed)) {
        test_api::s_ntile_flat_parallel_last_units.store(
                q.n_units, std::memory_order_relaxed);
        test_api::s_ntile_flat_parallel_last_sweep.store(
                q.sweep, std::memory_order_relaxed);
    }
    return q;
}

bool own_units_first(
        const int team, const bool prompt, const unit_order order) {
    if (order != unit_order::automatic) return order == unit_order::own;
    if (prompt) return false;
    const int pin = test_api::s_ntile_flat_parallel_own_units_override.load(
            std::memory_order_relaxed);
    if (pin >= 0) return pin != 0;
    const int l3_cores = grp_machine::machine().l3_domain_cores;
    return l3_cores <= 0 || team > l3_cores;
}

} // namespace detail
} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
