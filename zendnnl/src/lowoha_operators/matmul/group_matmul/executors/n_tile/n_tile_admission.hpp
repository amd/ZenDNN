/*******************************************************************************
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

/// ALGO 3 admission (`n_tile_supported`): whether the N-tile executor has
/// a safe column-sliced path for every active expert of a call.

#ifndef ZENDNNL_GROUP_MATMUL_N_TILE_ADMISSION_HPP
#define ZENDNNL_GROUP_MATMUL_N_TILE_ADMISSION_HPP

#include <vector>

#include "lowoha_operators/matmul/group_matmul/common/grp_matmul_constants.hpp"
#include "lowoha_operators/matmul/lowoha_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace detail {

bool n_tile_supported(const std::vector<int> &M, const std::vector<int> &N,
        const std::vector<matmul_params> &params, int num_ops,
        grp_matmul_gated_act_t fused_act);

} // namespace detail

/// Q8_0 custom-kernel shape: s8 weight, symmetric, K-group 32, N packable
/// by the NR=32 kernel, plain layout. Source is either still f32/bf16 with
/// dynamic_quant, or already s8 with a `{M, K/32}` scale.
inline bool q8_per_group_ck_expert(const matmul_params &p, int K, int N) {
    if (p.dtypes.wei != data_type_t::s8 || p.mem_format_b != 'n') return false;
    if (p.dtypes.compute != data_type_t::s8) return false;
    if (p.quant_params.src_zp.buff != nullptr
            || p.quant_params.wei_zp.buff != nullptr)
        return false;
    if (K <= 0 || N <= 0 || (K % 32) != 0 || (N % 32) != 0) return false;
    const auto &ws = p.quant_params.wei_scale;
    if (ws.buff == nullptr || ws.dims.size() != 2 || ws.dims[0] <= 1
            || ws.dims[1] != N)
        return false;
    if ((K % static_cast<int>(ws.dims[0])) != 0) return false;
    if (K / static_cast<int>(ws.dims[0]) != 32) return false;
    if (p.dynamic_quant) {
        return p.dtypes.src == data_type_t::f32
                || p.dtypes.src == data_type_t::bf16;
    }
    if (p.dtypes.src != data_type_t::s8) return false;
    const auto &ss = p.quant_params.src_scale;
    return ss.buff != nullptr && ss.dims.size() == 2
            && ss.dims[1] == ws.dims[0];
}

inline bool q8_per_group_ck_call(const std::vector<int> &M,
        const std::vector<int> &N, const std::vector<int> &K,
        const std::vector<matmul_params> &params) {
    bool any = false;
    const int n = static_cast<int>(std::min(
            M.size(), std::min(N.size(), std::min(K.size(), params.size()))));
    for (int i = 0; i < n; ++i) {
        if (M[i] <= 0) continue;
        if (!q8_per_group_ck_expert(params[static_cast<size_t>(i)], K[i], N[i]))
            return false;
        any = true;
    }
    return any;
}
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_N_TILE_ADMISSION_HPP
