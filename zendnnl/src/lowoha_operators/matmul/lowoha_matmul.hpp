/********************************************************************************
# * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <omp.h>
#include <cmath>
#include <cstring>

#include "lowoha_operators/matmul/lowoha_common.hpp"
#include "operators/matmul/matmul_context.hpp"

#define M_FLOPS 6.0
#define ENABLE_BRGEMM_KERNEL 0

namespace zendnnl {
namespace lowoha {
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
                           data_types &dtypes,
                           zendnnl::ops::matmul_algo_t kernel,
                           char mem_format_a, char mem_format_b,
                           lowoha_params &lowoha_param, const void *bias,
                           bool is_weights_const, bool can_reorder = false);

/**
 * @brief Execute Batch Matrix Multiplication (BMM) for batch_count > 1
 *
 * This function handles all batched matrix multiplication scenarios including:
 * - Batch GEMM using batched_sgemm kernel
 * - OneDNN batched execution
 * - Parallel partitioning across batches and M dimension
 */
void bmm_execute(const char layout, const char trans_input,
                 const char trans_weight,
                 const bool transA, const bool transB,
                 const int M, const int N, const int K, const float alpha,
                 const void *src, const int lda,
                 const void *weight, const int ldb,
                 const void *bias, const float beta,
                 void *dst, const int ldc,
                 const bool is_weights_const,
                 const int batch_count, const int Batch_A, const int Batch_B,
                 const size_t src_batch_stride, const size_t weight_batch_stride,
                 const size_t dst_batch_stride,
                 const size_t src_type_size, const size_t out_type_size,
                 const int num_threads,
                 matmul_algo_t kernel, lowoha_params &params);

/**
 * @brief Execute single Matrix Multiplication (Matmul) for batch_count == 1
 *
 * This function handles all single matrix multiplication scenarios including:
 * - Auto-tuner based kernel selection
 * - LIBXSMM blocked execution with tiling
 * - BRGEMM kernel execution
 * - Standard matmul kernel execution
 */
void matmul_execute(const char layout, const char trans_input,
                    const char trans_weight,
                    const bool transA, const bool transB,
                    const int M, const int N, const int K, const float alpha,
                    const void *src, const int lda,
                    const void *weight, const int ldb,
                    const void *bias, const float beta,
                    void *dst, const int ldc,
                    const bool is_weights_const,
                    const size_t src_type_size, const size_t out_type_size,
                    const int num_threads,
                    matmul_algo_t kernel, lowoha_params &params,
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
                       batch_params_t batch_params, lowoha_params params);

} // namespace lowoha
} // namespace zendnnl

#endif

