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

/// The unit queue: a call's rows in blocks and runs, its projections in units
/// of row runs and column ranges, the order of the queue, and how the
/// region's threads take its units.

#ifndef ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_ENGINE_SCHEDULE_HPP
#define ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_ENGINE_SCHEDULE_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include <omp.h>

#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/engine/scratch.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/ntile_flat_parallel_config.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/tile/nr32_tile.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {
namespace detail {

inline void spin_pause() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#endif
}

/// A call's unit queue and how its units sweep their columns.
struct queue_t {
    /// The call's phase, which sets its units per thread and their order.
    bool prompt = false;
    /// Its units sweep runs of blocks against L2-sized column chunks (by
    /// default a prompt call's).
    bool sweep = false;
    int n13 = 0; // W13 units, ahead of the W2 units
    int n_units = 0;
    int num_runs = 0;
    int chunk13 = 0;
    int chunk2 = 0;
    unit_order order = unit_order::automatic;
};

/// Plans a call of `n` experts whose `rows[e]` rows start at `S.offset[e]`:
/// each expert's rows in blocks of at most kDecodeMaxM rows, so an expert of a
/// decode call is one block and one run.  A sweeping call's units (by default
/// a prompt call's) each cover a run of an expert's blocks and sweep them
/// against one L2-sized chunk of columns at a time, so a unit streams each of
/// its weights once.  A W2 unit waits only for its runs' W13 units.  The unit
/// queue: every W13 unit, then (with `p2`) every W2 unit, heaviest group
/// first, with a sweeping call's taking turns from both ends of each
/// projection's units.  `K13` and `K2` are the projections' K, `es` their
/// source bytes per element; `phase` the call's, `k` its settings and
/// `l2_budget` its weights' L2 bytes.
queue_t plan_queue(scratch_t &S, const int *rows, const int n,
        const nr32::tile_plan &p13, const int K13, const nr32::tile_plan *p2,
        const int K2, const size_t es, const float wei_bytes,
        const int num_threads, const grp_matmul_phase phase,
        const call_knobs &k, const int64_t l2_budget);

/// Waits, pausing and then yielding, until `flag` reaches `value`.
inline void wait_for(const std::atomic<int> &flag, const int value) {
    const int spins = limits().wait_spins;
    for (int spin = 0; flag.load(std::memory_order_acquire) < value; ++spin) {
        if (spin < spins) {
            spin_pause();
        } else {
            std::this_thread::yield();
        }
    }
}

/// Whether a decode call's team of `team` threads runs each thread's own
/// units first: when it has more threads than one L3 has cores, so that it
/// spans several L3s as far as the team size shows, where a call repeated
/// with the same plan then finds each unit's weights in the L3 that read
/// them last.  A smaller team takes units in queue order, which stays
/// balanced when one of its cores runs slow; the rule reads only the team
/// size, so a small team spread over several L3s takes queue order, and a
/// team of SMT threads on one L3 with more threads than its cores, its own
/// units; `test_api::s_ntile_flat_parallel_own_units_override` pins the rule.
/// A prompt call takes queue order.  The `unit_order` setting `order`
/// replaces both.
bool own_units_first(const int team, const bool prompt, const unit_order order);

/// Runs, with every thread of the calling parallel region, each unit of the
/// queue `q` once through `run(u)`, in queue order, or (`own_units_first`)
/// each thread t of T first running its own units, those u with u % T == t,
/// its W13 units before its W2 units, so a call repeated with the same plan
/// gives each unit the thread, and the L3, that ran it last; then every
/// thread runs the units still unclaimed.  A W2 unit's wait is on W13 units
/// that are running or that their own thread reaches before any W2 unit, and
/// W13 units never wait.
template <typename Run>
void run_units(scratch_t &S, const queue_t &q, std::atomic<int> &next_unit,
        const Run &run) {
    const int T = omp_get_num_threads(), t = omp_get_thread_num();
    if (!own_units_first(T, q.prompt, q.order)) {
        for (;;) {
            const int u = next_unit.fetch_add(1, std::memory_order_relaxed);
            if (u >= q.n_units) break;
            run(u);
        }
        return;
    }
    const auto take = [&S](const int u) {
        return S.claimed[u].load(std::memory_order_relaxed) == 0
                && S.claimed[u].exchange(1, std::memory_order_relaxed) == 0;
    };
    for (int u = t; u < q.n13; u += T) {
        if (take(u)) run(u);
    }
    for (int u = q.n13 + t; u < q.n_units; u += T) {
        if (take(u)) run(u);
    }
    for (int u = 0; u < q.n_units; ++u) {
        if (take(u)) run(u);
    }
}

} // namespace detail
} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_ENGINE_SCHEDULE_HPP
