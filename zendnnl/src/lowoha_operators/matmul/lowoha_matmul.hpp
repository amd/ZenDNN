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

#define ENABLE_LIBXSMM_BRGEMM_KERNEL 0

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
 * @param batch_params     Batch parameters including batch sizes and strides
 * @param params           Additional parameters including post-ops and data types
 *
 * @return status_t::success on successful execution, status_t::failure otherwise
 */

status_t matmul_direct(const char layout, const bool transA, const bool transB,
                       const int M, const int N, const int K, const float alpha, const void *src,
                       const int lda, const void *weight, const int ldb, const void *bias,
                       const float beta, void *dst, const int ldc, const bool is_weights_const,
                       matmul_batch_params_t batch_params, matmul_params params);

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
 * @param params           Vector of additional parameters including post-ops and data types
 * @param moe_postop       Optional MoE weighted-reduce over pre-gathered expert rows;
 *                         nullptr disables (default). Parallel mode only; see
 *                         group_matmul_moe_postop_params.
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
                             const group_matmul_moe_postop_params *moe_postop = nullptr);

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif

