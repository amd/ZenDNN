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

/**
 * @brief FP32 group GEMM with gated activation (silu_and_mul).
 *
 * Demonstrates the gated_act parameter for fused gate+up projections:
 *   - 4 experts, each computing GEMM with N = 2*dim (fused gate+up weights).
 *   - After GEMM, in-place activation: dst[:, 0:dim] = silu(gate) * up.
 *   - Supported activations: silu_and_mul, gelu_and_mul, swiglu_oai_mul.
 *   - Constraints: N must be even, dst dtype must be FP32 or BF16.
 *   - Applied after GEMM and before moe_postop (if both are set).
 */
int group_matmul_gated_act_example();

/**
 * @brief FP32 fused MoE: gate+up → silu → down_proj in one API call.
 *
 * Demonstrates the fused_moe parameter for a single-call MoE workflow:
 *   - 4 experts, each computing Op1 (gate+up GEMM, N=2*dim), activation
 *     (silu_and_mul), and Op2 (down_proj GEMM, N_down=hidden_size).
 *   - Exposed as a single API call regardless of GRP_ALGO.  The current
 *     V1 implementation runs the flow as two internal dispatch passes
 *     (Op1 + activation, then Op2); per-expert / per-M-tile deep fusion
 *     is a future optimization.
 *
 * Legacy / caller-allocated mode: caller supplies both the Op1 dst
 * buffers (`dst[]` / `ldc[]`) and the Op2 dst buffers
 * (`fused.dst_down[]` / `fused.ldc_down[]`).  See
 * `group_matmul_fused_moe_internal_alloc_example` for the alternative
 * mode where the library allocates the Op1 scratch internally and
 * writes Op2 output back into the caller's src buffers in place.
 */
int group_matmul_fused_moe_example();

/**
 * @brief FP32 fused MoE in internal-alloc + src-reuse mode.
 *
 * Demonstrates the alternative fused MoE invocation where the library
 * owns the Op1 (gate+up + activation) intermediate buffer:
 *   - Caller passes `dst[]` as a vector of nullptrs (or empty) and
 *     leaves `fused.dst_down` empty.
 *   - The library obtains Op1 scratch sized for [M[i], N[i]] of the
 *     dst dtype (wide) or [M[i], N[i]/2] (tight / swiglu_oai-compact)
 *     per expert, runs Op1 + activation into it, then runs Op2
 *     reading from the scratch and writing the per-expert output
 *     BACK INTO the caller's `src[]` buffer (in-place reuse).
 *   - Scratch storage is NOT freed at end-of-call — it is backed by
 *     a `static thread_local` arena that grows to the largest
 *     fused-MoE call this thread has ever serviced and is released
 *     only when the thread exits.  Steady-state per-call allocator
 *     traffic is O(num_ops) field writes; per-thread resident-set
 *     reflects the high-water scratch size, and across N worker
 *     threads the total retained footprint is bounded above by
 *     N × max_seen(M_total × N_max × sizeof(dst_elem)).  Frameworks
 *     that need to bound RSS after an outsized MoE shape should
 *     either use the legacy caller-allocated mode or execute such
 *     shapes on a dedicated thread that can be torn down.
 *   - Caller reads the final per-expert Op2 output from the same
 *     `src[]` buffer it provided.
 *
 * Constraints: requires `lda[i] >= N_down[i]` so each Op2 row stride
 * fits within the original src row stride.  Naturally satisfied when
 * the MoE layer uses `hidden_dim = K_input = N_down`.
 */
int group_matmul_fused_moe_internal_alloc_example();

} // namespace examples
} // namespace zendnnl

#endif
