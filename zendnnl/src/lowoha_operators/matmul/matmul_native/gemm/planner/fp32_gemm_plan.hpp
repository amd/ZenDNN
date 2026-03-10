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

#ifndef MATMUL_NATIVE_PLANNER_FP32_GEMM_PLAN_HPP
#define MATMUL_NATIVE_PLANNER_FP32_GEMM_PLAN_HPP

#include "lowoha_operators/matmul/matmul_native/gemm/planner/gemm_planner.hpp"
#include "lowoha_operators/matmul/matmul_native/common/gemm_descriptor.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "lowoha_operators/matmul/lowoha_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

struct FP32GemmPlan {
    BlockPlan plan;
};

/// Build FP32 GEMM plan with plan caching.
FP32GemmPlan plan_fp32_gemm(
    const GemmDescriptor &desc,
    const UarchParams &uarch,
    const matmul_params &params);


} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_PLANNER_FP32_GEMM_PLAN_HPP
