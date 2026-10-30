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

/// ALGO 6 (`hierarchical_execute`): CCD-aware multilevel scheduling of a
/// grouped MoE call over nested OpenMP teams.

#ifndef ZENDNNL_GROUP_MATMUL_PARALLEL_MULTILEVEL_HPP
#define ZENDNNL_GROUP_MATMUL_PARALLEL_MULTILEVEL_HPP

#include <vector>

#include "lowoha_operators/matmul/group_matmul/common/grp_matmul_constants.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace detail {

void hierarchical_execute(const std::vector<char> &layout,
        const std::vector<bool> &transA, const std::vector<bool> &transB,
        const std::vector<int> &M, const std::vector<int> &N,
        const std::vector<int> &K, const std::vector<float> &alpha,
        const std::vector<const void *> &src, const std::vector<int> &lda,
        const std::vector<const void *> &weight, const std::vector<int> &ldb,
        const std::vector<const void *> &bias, const std::vector<float> &beta,
        const std::vector<void *> &dst, const std::vector<int> &ldc,
        const std::vector<bool> &is_weights_const,
        std::vector<matmul_params> &params, int num_threads,
        grp_matmul_gated_act_t fused_act, data_type_t act_dtype,
        const char **execution_path_out);

} // namespace detail
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_PARALLEL_MULTILEVEL_HPP
