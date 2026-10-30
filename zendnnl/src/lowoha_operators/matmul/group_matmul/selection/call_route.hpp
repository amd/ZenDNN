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

/// @file call_route.hpp
/// @brief What the dispatcher decides about one grouped call before it picks
///        an ALGO.
///
/// `group_matmul_execute` fills one `grp_matmul_call_route` per
/// call and hands it to the selector (`group_matmul_algo_select`), the N-tile
/// executor (`n_tile_execute`) and prepack (`PrepackRequest::route`), so each
/// reads the same verdict instead of recomputing it.  A new per-call verdict
/// is a field here plus the code that reads it, not another parameter on
/// every layer in between.  Each consumer still classifies the call itself
/// when it is given no route (direct callers and tests).

#ifndef ZENDNNL_GROUP_MATMUL_CALL_ROUTE_HPP
#define ZENDNNL_GROUP_MATMUL_CALL_ROUTE_HPP

#include <cstdint>

#include "lowoha_operators/matmul/group_matmul/multi_expert/multi_expert_policy.hpp"
#include "lowoha_operators/matmul/group_matmul/single_expert/single_expert_policy.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

struct grp_matmul_call_route {
    /// The single-expert custom-kernel policy (`grp_matmul_single_expert_call_ck`);
    /// empty for a call of several experts.
    single_expert_ck_t single_ck {};
    /// `grp_matmul_auto_algo3_family` of the call.
    grp_matmul_algo3_family moe_family = grp_matmul_algo3_family::none;
    /// The custom-kernel pack may be written into the caller's weight buffer
    /// while AUTO mixed in-place mode is active.
    bool ck_inplace_licence = false;
    /// The weight-cache mode the custom kernel packs with: the least of the
    /// process mode and every expert's `weight_cache_type`, taken before any
    /// cap on the AOCL reorders.  -1 derives it from the process and `params`.
    int32_t ck_weight_cache_mode = -1;
};

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_CALL_ROUTE_HPP
