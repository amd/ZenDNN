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

/// What both API fronts share: how a call is declined, and the settings,
/// epilogue and bias kind a call's types give.

#ifndef ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_API_API_COMMON_HPP
#define ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_API_API_COMMON_HPP

#include <cstdint>

#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/ntile_flat_parallel.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/ntile_flat_parallel_config.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/tile/nr32_tile.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {
namespace detail {

/// Whether layers of format `f` for `api` run the kernels that read
/// source bytes XOR-ed with 0x80, which skip that XOR on every broadcast:
/// for a format that has them (DA8W8's s8), on the machines where that
/// outweighs flipping each row once (the machine view's rule), unless the
/// config says otherwise.  W4A8 layers keep the s8-source kernels, whose
/// decode is the faster.
bool u8_src_pays(const nr32::format_desc &f, const api_kind api);

/// The budget plans of format `f` for `api` size their subtiles by.
nr32::plan_budget budget_for(const nr32::format_desc &f, const api_kind api);

inline nr32::epilogue epilogue_of(const grp_matmul_gated_act_t act) {
    switch (act) {
        case grp_matmul_gated_act_t::swiglu_oai_mul:
            return nr32::epilogue::swiglu_pairs;
        case grp_matmul_gated_act_t::silu_and_mul:
            return nr32::epilogue::silu_split;
        case grp_matmul_gated_act_t::gelu_and_mul:
            return nr32::epilogue::gelu_split;
        default: return nr32::epilogue::plain;
    }
}

inline nr32::bias_kind bias_kind_of(const data_type_t dt) {
    switch (dt) {
        case data_type_t::bf16: return nr32::bias_kind::bf16;
        case data_type_t::f16: return nr32::bias_kind::f16;
        case data_type_t::f32: return nr32::bias_kind::f32;
        default: return nr32::bias_kind::none;
    }
}

status_t decline(const char *reason);

/// The status of a call N-tile flat parallel fails for `reason`:
/// `memory_bad_storage` when an allocation failed, else `unimplemented`.
status_t refuse(const char *reason);

} // namespace detail
} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_API_API_COMMON_HPP
