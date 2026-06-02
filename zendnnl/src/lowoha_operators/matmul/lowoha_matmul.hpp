/********************************************************************************
# * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef _LOWOHA_MATMUL_HPP
#define _LOWOHA_MATMUL_HPP

#include <cmath>
#include <cstring>
#include <vector>

#include "lowoha_operators/matmul/lowoha_common.hpp"
#include "lowoha_operators/matmul/group_matmul/group_matmul_direct.hpp"
#include "operators/matmul/matmul_context.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

/**
 * @brief Entry function for different backends supported by ZenDNNL
 */
void matmul_kernel_wrapper(char layout, char transA, char transB,
                           int M, int N, int K,
                           float alpha,
                           const void *A, int lda,
                           const void *B, int ldb,
                           float beta,
                           void *C, int ldc,
                           matmul_data_types &dtypes,
                           zendnnl::ops::matmul_algo_t &kernel,
                           char mem_format_a, char mem_format_b,
                           matmul_params &lowoha_param, matmul_batch_params_t &batch_params,
                           const void *bias, bool is_weights_const);


/**
 * @brief Execute single Matrix Multiplication (Matmul) for batch_count == 1
 *
 * This function handles all single matrix multiplication scenarios including:
 * - Auto-tuner based kernel selection
 * - LIBXSMM blocked execution with tiling
 * - BRGEMM kernel execution
 * - Standard matmul kernel execution
 */
void matmul_execute(const char layout, const bool transA, const bool transB,
                    const int M, const int N, const int K, const float alpha,
                    const void *src, const int lda, const void *weight, const int ldb,
                    const void *bias, const float beta, void *dst, const int ldc,
                    const bool is_weights_const, const size_t src_type_size,
                    const size_t out_type_size, const int num_threads, matmul_algo_t &kernel,
                    matmul_params &params, matmul_batch_params_t &batch_params,
                    unsigned int auto_version);

/**
 * @brief Execute matrix multiplication with automatic kernel selection and optimization
 *
 * This function performs C = alpha * op(A) * op(B) + beta * C + fused post-ops.
 *
 * @param layout           Memory layout ('r' for row-major, 'c' for column-major)
 * @param transA           Whether to transpose matrix A
 * @param transB           Whether to transpose matrix B
 * @param M                Number of rows in A and C
 * @param N                Number of columns in B and C
 * @param K                Number of columns in A and rows in B
 * @param alpha            Scaling factor for A*B
 * @param src              Pointer to matrix A data
 * @param lda              Leading dimension of A
 * @param weight           Pointer to matrix B data
 * @param ldb              Leading dimension of B
 * @param bias             Optional bias vector (can be nullptr)
 * @param beta             Scaling factor for existing C values
 * @param dst              Pointer to matrix C data
 * @param ldc              Leading dimension of C
 * @param is_weights_const Whether the weights are constant (enables caching)
 * @param[in,out] batch_params  On input supplies Batch_A, Batch_B, and
 *     optional batch strides. The function may compute default batch
 *     strides when the caller supplies sentinel values (e.g. -1).
 * @param[in,out] params   Matmul configuration whose following fields
 *     may be mutated:
 *     - @c postop_[].leading_dim : defaulted to N for binary post-ops
 *       when the caller leaves it at -1.
 *     - @c dynamic_quant / @c dtypes : reorder-quantization dispatch may
 *       set the dynamic_quant flag and rewrite the source data type.
 *     - @c quant_params.wei_scale : GGML weight unpacking populates the
 *       weight-scale buffer and associated metadata.
 *
 * @note Thread safety: callers must not share the same @c matmul_params or
 *       @c matmul_batch_params_t instance across concurrent calls, since
 *       the function mutates both in place.
 *
 * @return status_t::success on successful execution, status_t::failure otherwise
 */

status_t matmul_direct(const char layout, const bool transA, const bool transB,
                       const int M, const int N, const int K, const float alpha, const void *src,
                       const int lda, const void *weight, const int ldb, const void *bias,
                       const float beta, void *dst, const int ldc, const bool is_weights_const,
                       matmul_batch_params_t &batch_params, matmul_params &params);

/**
 * @brief Execute group matmul operations (e.g. MoE experts)
 *
 * This function performs multiple independent matrix multiplications in sequence.
 * Each operation computes: C[i] = alpha[i] * op(A[i]) * op(B[i]) + beta[i] * C[i] + fused post-ops
 *
 * @param layout           Vector of memory layouts ('r' for row-major, 'c' for column-major)
 * @param transA           Vector of transpose flags for matrix A
 * @param transB           Vector of transpose flags for matrix B
 * @param M                Vector of row counts for A and C
 * @param N                Vector of column counts for B and C
 * @param K                Vector of column counts for A and row counts for B
 * @param alpha            Vector of scaling factors for A*B
 * @param src              Vector of pointers to matrix A data
 * @param lda              Vector of leading dimensions for A
 * @param weight           Vector of pointers to matrix B data
 * @param ldb              Vector of leading dimensions for B
 * @param bias             Vector of optional bias pointers (can contain nullptr)
 * @param beta             Vector of scaling factors for existing C values
 * @param dst              Vector of pointers to matrix C data
 * @param ldc              Vector of leading dimensions for C
 * @param is_weights_const Vector of flags indicating if weights are constant (enables caching)
 * @param params           Vector of additional parameters including post-ops and data types.
 *                         For MoE workloads the caller may also set the optional
 *                         prepack-extras hint on `params[0]`:
 *                           - `params[0].active_matmul` = number of firing experts
 *                             (must satisfy `active_matmul <= M.size()`).
 *                           - `params[0].total_matmul`  = total expert weight slots
 *                             carried in the call (`>= active_matmul`; `0` means
 *                             "no prepack-extras tail").
 *                         Two input-side sizing patterns are both accepted:
 *                           (a) Compact:  `M.size() == active_matmul`  — input
 *                               vectors carry only the firing experts.
 *                           (b) Padded:   `M.size() == total_matmul`  with
 *                               `M[active_matmul..total_matmul) == 0` placeholders
 *                               — the dispatcher skips the zero-M slots.
 *                         Weight-side sizing depends on whether the caller
 *                         supplies a prepack-extras tail:
 *                           (i)  No tail (`total_matmul == 0` OR
 *                                `total_matmul == active_matmul`): all per-
 *                                expert vectors — weight-side
 *                                (`weight`, `K`, `N`, `ldb`, `transB`,
 *                                `is_weights_const`) and the rest (alpha,
 *                                bias, beta, ldc, params, ...) — need only
 *                                be `>= active_matmul`.
 *                           (ii) With tail (`total_matmul > active_matmul`,
 *                                the rotating-experts MoE case): the six
 *                                weight-side prepack vectors above must be
 *                                `>= total_matmul` so the prepack module
 *                                can warm every advertised expert without
 *                                silent truncation.  All other vectors still
 *                                need only `>= active_matmul`.  Sizes
 *                                shorter than `total_matmul` on the six
 *                                weight-side vectors are rejected by the
 *                                dispatcher up front (no silent
 *                                under-warming).
 *                         The dispatcher always computes only the first
 *                         `active_matmul` GEMMs.  When
 *                         `ZENDNNL_GRP_MATMUL_PREPACK=1` (the default), it
 *                         eagerly pre-warms the weight cache for ALL
 *                         `total_matmul` experts so any future firing hits
 *                         a warm cache.  Leave both fields at `0` for the
 *                         legacy contract: every per-expert vector must be
 *                         exactly `num_ops = M.size()` long and every
 *                         supplied weight fires.  See
 *                         `docs/operator/low_overhead_operator/lowoha_group_matmul_operator.md`
 *                         (Framework prepack-extras contract) for a worked
 *                         example.
 * @param moe_postop       Optional MoE weighted-reduce over pre-gathered expert rows;
 *                         nullptr disables (default). Parallel mode only; see
 *                         group_matmul_moe_postop_params.
 * @param gated_act        Optional gated activation applied in-place after GEMM
 *                         and before moe_postop: dst[:, 0:dim] = act(gate) * up
 *                         where dim = N/2. Requires N even and dst dtype f32 or
 *                         bf16. nullptr disables (default). Parallel mode only;
 *                         see grp_matmul_gated_act_params.
 * @param fused_moe        Optional fused MoE parameters describing the full
 *                         Op1 (gate+up) → activation → Op2 (down_proj) block
 *                         in a single call. V1 executes this flow as two
 *                         passes for every GRP_ALGO: Pass 1 runs Op1 plus
 *                         the gated activation via the parallel dispatcher
 *                         (honoring ZENDNNL_GRP_MATMUL_ALGO), Pass 2 runs
 *                         Op2 via the same dispatcher. When moe_postop is
 *                         also provided, the weighted reduce runs afterward
 *                         in its own pass. Deep single-pass Op1→Act→Op2
 *                         chaining is a future optimization.
 *                         nullptr disables (default).
 *
 * @return status_t::success if all operations succeed, status_t::failure if any operation fails
 */
status_t group_matmul_direct(const std::vector<char> &layout,
                             const std::vector<bool> &transA,
                             const std::vector<bool> &transB,
                             const std::vector<int> &M,
                             const std::vector<int> &N,
                             const std::vector<int> &K,
                             const std::vector<float> &alpha,
                             const std::vector<const void *> &src,
                             const std::vector<int> &lda,
                             const std::vector<const void *> &weight,
                             const std::vector<int> &ldb,
                             const std::vector<const void *> &bias,
                             const std::vector<float> &beta,
                             const std::vector<void *> &dst,
                             const std::vector<int> &ldc,
                             const std::vector<bool> &is_weights_const,
                             std::vector<matmul_params> &params,
                             const group_matmul_moe_postop_params *moe_postop = nullptr,
                             const grp_matmul_gated_act_params *gated_act = nullptr,
                             const grp_matmul_fused_moe_params *fused_moe = nullptr);

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif

