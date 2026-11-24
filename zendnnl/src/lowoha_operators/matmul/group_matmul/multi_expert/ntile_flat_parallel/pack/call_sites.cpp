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

#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/pack/call_sites.hpp"

#include <utility>
#include <shared_mutex>
#include <unordered_map>

#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/ntile_flat_parallel_config.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {
namespace detail {

namespace {

// Never destroyed, as the store.
std::shared_mutex &sites_mutex() {
    static auto *mutex = new std::shared_mutex;
    return *mutex;
}

std::unordered_map<size_t, std::shared_ptr<const call_site_t>> &sites() {
    static auto *sites = new std::unordered_map<size_t,
            std::shared_ptr<const call_site_t>>;
    return *sites;
}

/// How many times the sites were dropped, under `sites_mutex()`.  A call
/// inserts its site only when they weren't dropped since it looked it up.
uint64_t &site_epoch() {
    static uint64_t epoch = 0;
    return epoch;
}

} // namespace

std::shared_ptr<const call_site_t> find_site(
        const size_t key, uint64_t &epoch) {
    std::shared_lock<std::shared_mutex> lock(sites_mutex());
    epoch = site_epoch();
    const auto it = sites().find(key);
    return it != sites().end() ? it->second : nullptr;
}

void publish_site(const size_t key, std::shared_ptr<const call_site_t> site,
        const uint64_t epoch, const uint64_t since) {
    std::unique_lock<std::shared_mutex> lock(sites_mutex());
    if (site_epoch() != epoch || !may_publish(since)) return;
    if (sites().size() >= limits().call_sites) sites().clear();
    sites()[key] = std::move(site);
}

void drop_call_sites() {
    std::unique_lock<std::shared_mutex> lock(sites_mutex());
    sites().clear();
    ++site_epoch();
}

size_t call_site_count() {
    std::shared_lock<std::shared_mutex> lock(sites_mutex());
    return sites().size();
}

} // namespace detail
} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
