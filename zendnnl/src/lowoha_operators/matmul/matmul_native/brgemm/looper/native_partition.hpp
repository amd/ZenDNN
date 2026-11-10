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

#ifndef MATMUL_NATIVE_BRGEMM_LOOPER_NATIVE_PARTITION_HPP
#define MATMUL_NATIVE_BRGEMM_LOOPER_NATIVE_PARTITION_HPP

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>

#include <omp.h>

#include "lowoha_operators/matmul/lowoha_common.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_planner.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

/// Rows m0 .. m0 + rows of the output by tiles column tiles of the plan's nr
/// pack columns from n0, over all of K.
struct native_work_unit_t {
    int m0 = 0;
    int rows = 0;
    int n0 = 0;
    int tiles = 1;
};

enum class native_partition_kind_t : uint8_t { slice, row_groups, blocks };

/// The counter of one domain of a blocks run: the next of its units, the
/// row blocks r0 .. r0 + rows they cover, and their count.
struct alignas(64) native_unit_cursor_t {
    std::atomic<int64_t> next;
    int64_t end;
    int r0;
    int rows;
};

/// Domains of a blocks run, hence counters, at most.
inline constexpr int kNativeMaxUnitCursors = 64;

/// The counters of one blocks run: each worker w takes units from counter
/// w / domain_threads, then from the others in turn.
struct native_unit_deal_t {
    native_unit_cursor_t cursor[kNativeMaxUnitCursors];
    int count = 0;
    int domain_threads = 1;
};

/// The tiles of column tiles c0 .. c1 of a plan, assigned to the workers of
/// a team: statically, or under the blocks schedule from counters.  Units
/// are generated, never stored.
class native_partition_t {
public:
    native_partition_t(const native_plan_t &plan, int M, int c0, int c1);
    /// blocks when plan.unit_tiles > 0; else row_groups when
    /// plan.col_groups > 0, workers == plan.threads and col_groups divides
    /// workers and c1 - c0; slice otherwise.
    native_partition_kind_t kind(int workers) const noexcept;
    /// Units of the range: m_blocks times c1 - c0 tiles, or under the blocks
    /// schedule m_blocks times the groups of unit_tiles column tiles (the
    /// last group may be narrower).
    int64_t units() const noexcept;
    /// The counters of a blocks run of workers workers: each domain of
    /// domain_threads adjacent workers counts the units of its share of the
    /// row blocks, column group by column group.
    void deal(int workers, native_unit_deal_t &deal) const noexcept;
    /// Calls f(unit) for every unit of worker w of workers, in execution
    /// order; a blocks run takes its units from deal, filled by deal() of
    /// this partition for this run (a run drains it).  Every unit of the
    /// range goes to exactly one worker.
    template <typename F>
    void for_each(int w, int workers, F &&f,
            native_unit_deal_t *deal = nullptr) const;

private:
    /// Rows r0 .. r1 of row group rg of row_groups (whole row units).
    void row_range(int rg, int row_groups, int &r0, int &r1) const noexcept;

    int M_;
    int mb_;
    int nr_;
    int m_blocks_;
    int c0_;
    int c1_;
    int col_groups_;
    int threads_;
    int row_unit_;
    int unit_tiles_;
    int domain_threads_;
    native_traversal_t traversal_;
};

/// The calling thread's accumulator scratch of at least floats floats,
/// 64-byte aligned; kept for the thread's lifetime and grown, never shrunk.
/// nullptr when it cannot be allocated.
float *native_thread_scratch(size_t floats);

/// A step every thread of a team region runs once all its units are done:
/// run(ctx, i, n) on thread i of the region's n.
struct native_team_step_t {
    void (*run)(const void *ctx, int i, int n) = nullptr;
    const void *ctx = nullptr;
};

/// The call's OpenMP team: a region of team threads, the caller's team, of
/// which the first workers (at most the threads the runtime gives) run the
/// units of part, worker tid of workers, with their thread-local scratch of
/// scratch_floats floats; the other threads only meet them at the end of the
/// region.  Opening every region with the caller's team keeps one team size
/// across calls, so the runtime never resizes its team between them.  One
/// worker or a team of 1 runs inline.  team_used, when set, receives the
/// threads of the region (1 inline).  after, when set, runs on every thread
/// of the region (inline: once, 0 of 1) after a barrier past every unit.
/// Fails when a worker gets no scratch, after the other workers ran their
/// units.
template <typename Tile>
status_t native_run_cpu_team(const native_partition_t &part, int workers,
        int team, size_t scratch_floats, Tile &&tile, int *team_used = nullptr,
        const native_team_step_t *after = nullptr);

template <typename F>
void native_partition_t::for_each(
        int w, int workers, F &&f, native_unit_deal_t *deal) const {
    auto emit = [&](int m0, int rows, int tile) {
        const native_work_unit_t u {m0, rows, tile * nr_, 1};
        f(u);
    };
    if (kind(workers) == native_partition_kind_t::blocks) {
        assert(deal != nullptr && deal->count > 0);
        const int count = deal->count;
        const int first = std::min(w / deal->domain_threads, count - 1);
        const int span = c1_ - c0_;
        for (int k = 0; k < count; ++k) {
            native_unit_cursor_t &c = deal->cursor[(first + k) % count];
            for (;;) {
                const int64_t i
                        = c.next.fetch_add(1, std::memory_order_relaxed);
                if (i >= c.end) break;
                const int group = static_cast<int>(i / c.rows);
                const int m0 = (c.r0 + static_cast<int>(i % c.rows)) * mb_;
                const int t0 = group * unit_tiles_;
                const native_work_unit_t u {m0, std::min(mb_, M_ - m0),
                        (c0_ + t0) * nr_, std::min(unit_tiles_, span - t0)};
                f(u);
            }
        }
        return;
    }
    if (kind(workers) == native_partition_kind_t::slice) {
        const int64_t tiles = units();
        const int64_t t1 = tiles * (w + 1) / workers;
        if (traversal_ == native_traversal_t::m_inner) {
            for (int64_t t = tiles * w / workers; t < t1; ++t) {
                const int m0 = static_cast<int>(t % m_blocks_) * mb_;
                emit(m0, std::min(mb_, M_ - m0),
                        c0_ + static_cast<int>(t / m_blocks_));
            }
        } else {
            const int cols = c1_ - c0_;
            for (int64_t t = tiles * w / workers; t < t1; ++t) {
                const int m0 = static_cast<int>(t / cols) * mb_;
                emit(m0, std::min(mb_, M_ - m0),
                        c0_ + static_cast<int>(t % cols));
            }
        }
        return;
    }
    // Row ranges in whole row units; the workers of one row range are
    // adjacent.
    int r0 = 0, r1 = 0;
    row_range(w / col_groups_, workers / col_groups_, r0, r1);
    const int cols = (c1_ - c0_) / col_groups_;
    const int first = c0_ + (w % col_groups_) * cols;
    for (int c = first; c < first + cols; ++c)
        for (int m0 = r0; m0 < r1; m0 += mb_)
            emit(m0, std::min(mb_, r1 - m0), c);
}

template <typename Tile>
status_t native_run_cpu_team(const native_partition_t &part, int workers,
        int team, size_t scratch_floats, Tile &&tile, int *team_used,
        const native_team_step_t *after) {
    std::atomic<bool> no_scratch {false};
    // The counters of a blocks run; the static schedules leave them unset.
    native_unit_deal_t deal;
    if (part.kind(workers) == native_partition_kind_t::blocks)
        part.deal(std::max(1, workers), deal);
    auto run = [&](int w, int n) {
        float *acc = native_thread_scratch(scratch_floats);
        if (acc == nullptr) {
            no_scratch.store(true, std::memory_order_relaxed);
            return;
        }
        part.for_each(w, n, [&](const native_work_unit_t &u) { tile(u, acc); },
                &deal);
    };
    if (workers <= 1 || team <= 1) {
        run(0, 1);
        if (team_used != nullptr) *team_used = 1;
        if (after != nullptr && !no_scratch.load(std::memory_order_relaxed))
            after->run(after->ctx, 0, 1);
    } else {
        const int threads = std::max(workers, team);
#pragma omp parallel num_threads(threads)
        {
            const int given = omp_get_num_threads();
            const int n = std::min(workers, given);
            const int tid = omp_get_thread_num();
            if (tid < n) run(tid, n);
            if (tid == 0 && team_used != nullptr) *team_used = given;
            if (after != nullptr) {
#pragma omp barrier
                if (!no_scratch.load(std::memory_order_relaxed))
                    after->run(after->ctx, tid, given);
            }
        }
    }
    return no_scratch.load(std::memory_order_relaxed) ? status_t::failure
                                                      : status_t::success;
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_BRGEMM_LOOPER_NATIVE_PARTITION_HPP
