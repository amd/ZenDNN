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

#ifndef MATMUL_AI_GEMM_PLANNER_HPP
#define MATMUL_AI_GEMM_PLANNER_HPP

#include "lowoha_operators/matmul/matmul_ai/common/gemm_descriptor.hpp"
#include "lowoha_operators/matmul/matmul_ai/common/cost_model.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ai {

/// Cache-blocking and register-tile plan for the GEMM macro-loop.
struct BlockPlan {
  int MB, NB, KB;
  int MR, NR;
  int num_threads;
  bool pack_a, pack_b;

  BlockPlan()
    : MB(0), NB(0), KB(0), MR(6), NR(16),
      num_threads(1), pack_a(true), pack_b(true) {}
};

/// Compute a BlockPlan from the problem descriptor and hardware parameters.
BlockPlan plan_blocks(const GemmDescriptor &desc, const UarchParams &uarch);

} // namespace ai
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_AI_BLOCK_PLANNER_HPP
