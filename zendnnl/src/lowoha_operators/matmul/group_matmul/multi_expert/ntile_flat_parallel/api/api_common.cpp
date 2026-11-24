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

#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/api/api_common.hpp"

#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/formats/formats.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/pack/store.hpp"
#include "lowoha_operators/matmul/lowoha_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {
namespace detail {

bool u8_src_pays(const nr32::format_desc &f, const api_kind api) {
    if (!f.u8_src) return false;
    const int pin = test_api::s_ntile_flat_parallel_u8_src_override.load(
            std::memory_order_relaxed);
    if (pin >= 0) return pin != 0;
    return u8_src(f, api);
}

nr32::plan_budget budget_for(const nr32::format_desc &f, const api_kind api) {
    return {l2_budget_bytes(f, api), plan_knobs_for(f, api).subtile_rows};
}

status_t decline(const char *reason) {
    apilog_error(
            "[GRP_MATMUL.NTILE_FLAT_PARALLEL] cannot serve the call: reason=",
            reason);
    return status_t::unimplemented;
}

status_t refuse(const char *reason) {
    if (reason == kOutOfMemory) {
        apilog_error(
                "[GRP_MATMUL.NTILE_FLAT_PARALLEL] cannot serve the call: "
                "reason=",
                reason);
        return status_t::memory_bad_storage;
    }
    return decline(reason);
}

} // namespace detail
} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
