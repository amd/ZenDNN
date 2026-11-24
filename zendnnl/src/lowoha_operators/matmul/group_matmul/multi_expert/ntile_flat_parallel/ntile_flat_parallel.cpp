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

#include "ntile_flat_parallel.hpp"

#include <mutex>

#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/api/routed.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/engine/scratch.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/pack/call_sites.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/pack/store.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {

namespace {

/// Serializes flushes, so the flush count is odd exactly while one runs.
/// Never destroyed, so a flush from an exit handler finds it.
std::mutex &flush_mutex() {
    static auto *mutex = new std::mutex;
    return *mutex;
}

} // namespace

uint64_t flush_count() {
    return detail::flushes().load(std::memory_order_acquire);
}

void flush() {
    // The count is odd from before anything is dropped until everything is:
    // a call that read it before or during the flush publishes no pack and
    // keeps no layer or site, whatever the order of the steps.
    std::lock_guard<std::mutex> serial(flush_mutex());
    detail::flushes().fetch_add(1, std::memory_order_acq_rel);
    detail::clear_store();
    if (void (*const during)()
            = test_api::s_ntile_flat_parallel_during_flush.exchange(
                    nullptr, std::memory_order_relaxed)) {
        during();
    }
    detail::clear_layers();
    detail::drop_call_sites();
    detail::flushes().fetch_add(1, std::memory_order_acq_rel);
}

void reset_thread_local_scratch() {
    detail::scratch() = detail::scratch_t {};
}

} // namespace ntile_flat_parallel

size_t test_api::ntile_flat_parallel_layers() {
    return ntile_flat_parallel::detail::layer_count();
}

size_t test_api::ntile_flat_parallel_packs() {
    return ntile_flat_parallel::detail::pack_count();
}

size_t test_api::ntile_flat_parallel_call_sites() {
    return ntile_flat_parallel::detail::call_site_count();
}

size_t test_api::ntile_flat_parallel_warmed_pools() {
    return ntile_flat_parallel::detail::warmed_pool_count();
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
