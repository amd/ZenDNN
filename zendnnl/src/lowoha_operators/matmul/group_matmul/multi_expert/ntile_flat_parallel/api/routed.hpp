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

/// Routed calls (`routed_fused_moe_direct`), which `api/routed.cpp` serves as
/// `ntile_flat_parallel::execute`: each layer's state (its experts' packs and
/// tile plans), the routed front, which routes a call's tokens to
/// expert-major rows, and the call.  This header has what `flush()` and the
/// tests need of the layers.

#ifndef ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_API_ROUTED_HPP
#define ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_API_ROUTED_HPP

#include <cstdint>

#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/ntile_flat_parallel.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {
namespace detail {

/// Drops every routed layer's state.  Only `flush()` may, between its two
/// counts, so that no call keeps a layer over packs the store has dropped.
void clear_layers();

/// How many routed layers have a state.
size_t layer_count();

} // namespace detail
} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_API_ROUTED_HPP
