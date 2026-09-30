/*******************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *******************************************************************************/

#ifndef ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_MATMUL_ROUTE_HPP
#define ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_MATMUL_ROUTE_HPP

#include <omp.h>

#include "lowoha_operators/matmul/group_matmul/custom_kernel/dispatch.hpp"
#include "lowoha_operators/matmul/group_matmul/group_matmul_parallel_common.hpp"
#include "lowoha_operators/matmul/group_matmul/n_tile/group_matmul_n_tile.hpp"
#include "lowoha_operators/matmul/lowoha_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

inline constexpr int kMinCustomKernelRouteN = 2048;

inline bool matmul_custom_kernel_bias_supported(
        const void *bias, data_type_t bias_dtype) noexcept {
    const bool has_bias = bias != nullptr;
    const bool has_bias_dtype = bias_dtype != data_type_t::none;
    if (has_bias != has_bias_dtype) return false;
    if (!has_bias) return true;
    return bias_dtype == data_type_t::bf16 || bias_dtype == data_type_t::f32
            || bias_dtype == data_type_t::f16;
}

/// Return true when a single MatMul can be reproduced by one-op grouped AUTO.
///
/// The grouped dispatcher owns prompt/decode selection and cache-mode-2
/// cross-warming once this boundary accepts the call.
inline bool custom_kernel_routable(char layout, int M, int N, int K,
        int batch_count, bool transA, bool transB, float alpha, float beta,
        const void *bias, int ldb, bool is_weights_const,
        const matmul_params &params) {
    auto &config = zendnnl::common::matmul_config_t::instance();
    if (!config.get_custom_kernel_route()) return false;

    const int32_t global_cache = config.get_weight_cache();
    if (global_cache < 0 || global_cache > 2) return false;
    if (!custom_kernel::dispatch_supported() || !get_grp_matmul_custom_kernel())
        return false;
    if (get_grp_n_tile_strategy() == 0) return false;

    if (N < kMinCustomKernelRouteN) return false;
    if (M <= 0 || batch_count != 1 || transA || !is_weights_const) return false;
    if (layout != 'r' && layout != 'R') return false;
    if (params.mem_format_a != 'n') return false;
    if (omp_in_parallel() || omp_get_dynamic()) return false;

    // Preserve a per-operation algorithm chosen through the API.  The
    // process-wide algorithm, however, is the fallback backend when CK is
    // disabled, ineligible, or declined and must not veto an enabled CK
    // attempt.  Layout-specific prepacked inputs remain excluded below by the
    // mem_format_b == 'n' contract.
    if (params.lowoha_algo != matmul_algo_t::none) return false;

    const int32_t effective_cache
            = effective_weight_cache_type(params.weight_cache_type);
    if (effective_cache != global_cache) return false;

    if (alpha != 1.0f || beta != 0.0f) return false;
    if (!matmul_custom_kernel_bias_supported(bias, params.dtypes.bias))
        return false;
    if (!params.postop_.empty()) return false;
    if (params.dtypes.src != data_type_t::bf16
            || params.dtypes.wei != data_type_t::bf16
            || params.dtypes.dst != data_type_t::bf16)
        return false;
    if (params.dynamic_quant || params.packing.pack_format_b == 1) return false;
    if (params.mem_format_b != 'n') return false;

    const int pack_nr = custom_kernel::plan_pack_nr(K, N);
    if (pack_nr != custom_kernel::kNRMin && pack_nr != custom_kernel::kNRMax)
        return false;
    const int min_ldb = transB ? K : N;
    return ldb >= min_ldb;
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_MATMUL_ROUTE_HPP
