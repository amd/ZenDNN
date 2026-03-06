/*******************************************************************************
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#ifndef MATMUL_AI_PLANNER_BF16_GEMM_PLAN_HPP
#define MATMUL_AI_PLANNER_BF16_GEMM_PLAN_HPP

#include "lowoha_operators/matmul/matmul_ai/gemm/planner/gemm_planner.hpp"
#include "lowoha_operators/matmul/matmul_ai/common/gemm_descriptor.hpp"
#include "lowoha_operators/matmul/matmul_ai/common/cost_model.hpp"
#include "lowoha_operators/matmul/matmul_ai/common/kernel_cache.hpp"
#include "lowoha_operators/matmul/lowoha_common.hpp"

#include <vector>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ai {

struct BF16GemmPlan {
    BlockPlan plan;
    const char *path_name;
    bool is_decode;
    bool has_activation;
    bool has_complex_activation;
};

/// Build a complete BF16 GEMM plan: base blocking + BF16-specific
/// MR/NR overrides, KB adjustments, decode path, and load balancing.
///
/// This is the single entry point for all BF16 GEMM planning.
/// Uses thread-local plan caching to avoid recomputation for repeated shapes.
BF16GemmPlan plan_bf16_gemm(
    const GemmDescriptor &desc,
    const UarchParams &uarch,
    const matmul_params &params);

} // namespace ai
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_AI_PLANNER_BF16_GEMM_PLAN_HPP
