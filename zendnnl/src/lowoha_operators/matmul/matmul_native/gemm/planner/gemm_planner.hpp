/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef MATMUL_NATIVE_GEMM_PLANNER_HPP
#define MATMUL_NATIVE_GEMM_PLANNER_HPP

#include "lowoha_operators/matmul/matmul_native/common/gemm_descriptor.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "lowoha_operators/matmul/lowoha_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

/// Cache-blocking and register-tile plan for the GEMM macro-loop.
struct BlockPlan {
  int MB, NB, KB;
  int MR, NR;
  int num_threads;

  BlockPlan()
    : MB(0), NB(0), KB(0), MR(6), NR(16),
      num_threads(1) {}
};

/// BF16 GEMM plan: base blocking + BF16-specific MR/NR, KB, decode path.
struct BF16GemmPlan {
    BlockPlan plan;
    const char *path_name;
    bool is_decode;
    bool has_activation;
    bool has_complex_activation;
};

/// FP32 GEMM plan: base blocking + FP32-specific NR selection.
struct FP32GemmPlan {
    BlockPlan plan;
};

/// Compute base blocking parameters (shared by BF16 and FP32).
BlockPlan plan_blocks(const GemmDescriptor &desc, const UarchParams &uarch);

/// Build BF16 GEMM plan with thread-local plan caching.
BF16GemmPlan plan_bf16_gemm(
    const GemmDescriptor &desc,
    const UarchParams &uarch,
    const matmul_params &params);

/// Build FP32 GEMM plan with thread-local plan caching.
FP32GemmPlan plan_fp32_gemm(
    const GemmDescriptor &desc,
    const UarchParams &uarch,
    const matmul_params &params);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_GEMM_PLANNER_HPP
