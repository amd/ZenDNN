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

#include "lowoha_operators/matmul/matmul_ai/brgemm/planner/fp32_brgemm_plan.hpp"
#include <algorithm>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ai {

FP32BrgemmFullPlan plan_fp32_brgemm(
    const GemmDescriptor &desc,
    const UarchParams &uarch,
    const matmul_params &params) {

    BrgemmPlan plan = plan_brgemm(desc, uarch);
    return FP32BrgemmFullPlan{plan};
}

} // namespace ai
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
