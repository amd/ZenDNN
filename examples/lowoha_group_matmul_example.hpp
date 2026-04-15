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
 ******************************************************************************/

#ifndef _LOWOHA_GROUP_MATMUL_EXAMPLE_HPP_
#define _LOWOHA_GROUP_MATMUL_EXAMPLE_HPP_

#include "zendnnl.hpp"
#include <cstring>
#include <iostream>
#include <vector>

#ifndef OK
#define OK     (0)
#endif
#ifndef NOT_OK
#define NOT_OK (1)
#endif

namespace zendnnl {
namespace examples {

using namespace zendnnl::lowoha::matmul;

/** @brief FP32 parallel group GEMM: 3 independent matmuls. */
int group_matmul_fp32_example();

/** @brief BF16 parallel group GEMM: 3 independent matmuls. */
int group_matmul_bf16_example();

/**
 * @brief BF16 group GEMM with MoE weighted-reduce post-op.
 *
 * Demonstrates the full MoE workflow:
 *   1. 4 experts, each computing BF16 GEMM on routed tokens.
 *   2. Fused weighted-reduce post-op combines top-2 expert outputs
 *      per token using routing weights.
 *
 * The caller builds row_ptrs during token-to-expert scatter:
 *   row_ptrs[t * topk + k] = &expert_dst[expert_id][row_j * ldc]
 *
 * After group_matmul_direct returns, moe_output contains the
 * reduced [num_tokens, N] tensor ready for the next layer.
 */
int group_matmul_moe_postop_example();

} // namespace examples
} // namespace zendnnl

#endif
