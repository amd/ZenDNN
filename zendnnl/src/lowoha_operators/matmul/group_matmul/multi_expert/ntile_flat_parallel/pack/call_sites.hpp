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

/// Grouped call sites: for each set of weights grouped calls name, pointers to
/// their packs and the calls' tile plans, so a call finds them with one
/// lookup.  A site keeps alive only packs the store holds, so the store
/// replacing a scaled pack drops every site.

#ifndef ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_PACK_CALL_SITES_HPP
#define ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_PACK_CALL_SITES_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/pack/nr32_pack.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/pack/store.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/tile/nr32_tile.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {
namespace detail {

/// The packs of a grouped call site: the weights its calls name (their
/// operations' and their prepack extras'), each with the scales of its scaled
/// pack.  A call that names the same weights, in any order, finds its packs
/// here with one lookup instead of one per expert in the store, and the site
/// keeps them alive for it without counting references to each.
struct call_site_t {
    struct slot_t {
        const void *weight = nullptr;
        scale_id scales;
        const nr32::packed_expert *pack = nullptr;
    };
    /// Open-addressed by weight, a power of two of slots each.
    std::vector<slot_t> w13, w2;
    std::vector<expert_pack> keep;
    /// The projections' layouts and tile plans.
    nr32::pack_layout l13, l2;
    nr32::tile_plan p13, p2;
};

/// `weight`'s slot in `table`, or null.
inline const call_site_t::slot_t *find_slot(
        const std::vector<call_site_t::slot_t> &table, const void *weight) {
    if (table.empty() || weight == nullptr) return nullptr;
    const size_t mask = table.size() - 1;
    for (size_t i = pointer_hash(weight) & mask;; i = (i + 1) & mask) {
        if (table[i].weight == weight) return &table[i];
        if (table[i].weight == nullptr) return nullptr;
    }
}

// The sites keep alive only packs the store holds: the store replacing a
// scaled pack drops every site.  Their state is private to call_sites.cpp.

/// The site of key `key`, or null; `epoch` gets how many times the sites
/// had been dropped, which `publish_site` checks.
std::shared_ptr<const call_site_t> find_site(size_t key, uint64_t &epoch);

/// Publishes `site` as key `key`'s, unless the sites were dropped since
/// `find_site` gave `epoch` or a flush ran since `since`.  Past
/// `limits().call_sites` sites, drops them all first.
void publish_site(size_t key, std::shared_ptr<const call_site_t> site,
        uint64_t epoch, uint64_t since);

/// Drops every site.
void drop_call_sites();

/// How many call sites there are.
size_t call_site_count();

} // namespace detail
} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_PACK_CALL_SITES_HPP
