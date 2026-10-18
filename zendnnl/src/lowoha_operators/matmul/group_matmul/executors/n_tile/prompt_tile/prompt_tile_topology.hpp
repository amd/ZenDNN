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

/// @file prompt_tile_topology.hpp
/// @brief OMP thread -> L3 team map for the ALGO 3 prompt tile executor.
///
/// A domain is the set of OMP threads whose CPUs share one L3; a team is a
/// domain, or a slice of one when sub-teams are on.
/// Splitting a domain into sub-teams gives each its own tile range while
/// an idle sub-team first steals from its siblings, whose B slices are
/// already in the shared L3.  The map is detected once per (thread count,
/// team knob, sub-team knob) by probing `sched_getcpu()` from every
/// thread of a parallel region and reading each CPU's L3 id from sysfs,
/// so it assumes the OMP threads are pinned (vLLM / ZenTorch bind one
/// thread per core).  Unpinned, the map records where that one probe found
/// the threads; a nested or non-Linux call, or one whose CPUs sysfs does
/// not describe, gets contiguous domains of `ccx_cores` threads.  Either
/// map is valid; only locality suffers.

#ifndef ZENDNNL_GROUP_MATMUL_PROMPT_TILE_TOPOLOGY_HPP
#define ZENDNNL_GROUP_MATMUL_PROMPT_TILE_TOPOLOGY_HPP

#include <cstdint>
#include <vector>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace prompt_tile {

struct TeamMap {
    int num_threads = 0;
    int n_teams = 0;
    /// Team of OMP thread `tid`, size `num_threads`.
    std::vector<int> team_of;
    /// Thread count of each team, size `n_teams`; every entry is >= 1.
    std::vector<int> team_size;
    /// L3 domain of each team, size `n_teams`; sub-teams of one domain
    /// share an id.
    std::vector<int> domain_of_team;
    /// Victims of idle team `t` are `steal_order[t * (n_teams - 1) + i]`:
    /// same-domain teams first, then the rest, each in ring order from t.
    std::vector<int> steal_order;
    /// L2 per core and L3 per domain used to size tiles.
    int64_t l2_bytes = 0;
    int64_t l3_bytes = 0;
    /// True when domains came from sysfs L3 ids (false = contiguous).
    bool from_l3 = false;
};

/// Map for an OMP team of `num_threads` threads.  `team_override > 0`
/// forces contiguous teams of that many threads (one domain each);
/// otherwise every L3 domain is split into `ceil(size / subteam)`
/// near-equal sub-teams (`subteam <= 0` keeps whole domains).  Cached per
/// argument triple and per whether the caller is inside a parallel
/// region; safe to call concurrently.  Inside a parallel region a probe
/// region would run with a single thread, so the map there is built from
/// contiguous domains of `ccx_cores` threads.
const TeamMap &get_team_map(int num_threads, int team_override, int subteam);

/// Contiguous teams of `team_size` threads, one domain each; the last
/// team takes the rest.
TeamMap make_contiguous_team_map(
        int num_threads, int team_size, int64_t l2_bytes, int64_t l3_bytes);

/// Teams from each thread's domain id (any non-negative labels): domains
/// are numbered by first appearance and each is split into
/// `ceil(size / subteam)` sub-teams of consecutive member threads.
TeamMap make_domain_team_map(const std::vector<int> &domain_of_thread,
        int subteam, int64_t l2_bytes, int64_t l3_bytes);

} // namespace prompt_tile
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_PROMPT_TILE_TOPOLOGY_HPP
