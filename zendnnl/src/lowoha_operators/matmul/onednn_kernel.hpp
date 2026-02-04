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

#ifndef _ONEDNN_KERNEL_HPP
#define _ONEDNN_KERNEL_HPP

#include "lowoha_operators/matmul/lowoha_common.hpp"
#if ZENDNNL_DEPENDS_ONEDNN
  #include "operators/matmul/onednn/matmul_onednn_kernel.hpp"
  using namespace dnnl;
#endif

namespace zendnnl {
namespace lowoha {
namespace matmul {

#if ZENDNNL_DEPENDS_ONEDNN
/**
 * @brief Wrapper function for OneDNN-based matrix multiplication
 *
 * Executes C = alpha * op(A) * op(B) + beta * C using OneDNN primitives.
 * Handles data type conversion, post-operations, and batch processing.
 *
 * @param transA/transB Transpose flags for matrices A and B
 * @param M,N,K Matrix dimensions
 * @param alpha,beta Scaling factors for GEMM operation
 * @param A,B,C Pointers to matrix data
 * @param lda,ldb,ldc Leading dimensions of matrices
 * @param lowoha_params Parameters including data types and post-ops
 * @param batch_params Batch parameters including stride information
 * @param bias Optional bias vector pointer
 * @param kernel Algorithm selection for the matmul operation
 * @param is_weights_const Whether weights are constant (for caching)
 */
void matmul_onednn_wrapper(char transA, char transB, int M, int N,
                           int K, float alpha, const void *A, int lda, const void *B, int ldb, float beta,
                           void *C, int ldc, matmul_params &lowoha_params,
                           matmul_batch_params_t &batch_params,
                           const void *bias, zendnnl::ops::matmul_algo_t kernel, bool is_weights_const,
                           size_t src_batch_stride=static_cast<size_t>(-1),
                           size_t weight_batch_stride=static_cast<size_t>(-1),
                           size_t dst_batch_stride=static_cast<size_t>(-1));
/**
 * @brief Gets or creates blocked weights with thread-safe caching
 *
 * This function handles the weight blocking and caching logic for oneDNN matmul.
 * It uses a two-level caching strategy:
 * 1. hash_values: Maps full key to blocking format hash
 * 2. matmul_weight_cache: LRU cache for actual blocked weight memory
 *
 * Thread safety is ensured by a mutex protecting all cache operations.
 *
 * @param transA Whether input A is transposed
 * @param transB Whether input B (weights) is transposed
 * @param M Number of rows in output
 * @param K Inner dimension
 * @param N Number of columns in output
 * @param lda Leading dimension of A
 * @param ldb Leading dimension of B
 * @param dnnl_params OneDNN parameters (weights.mem will be set)
 * @param eng OneDNN engine
 * @param matmul_attr Primitive attributes
 * @param weight_cache_type 0 = disabled, otherwise enabled
 */
void getOrCreateBlockedWeights(bool transA, bool transB, int M, int K, int N,
                               int lda, int ldb, onednn_utils_t::onednn_matmul_params &dnnl_params,
                               const dnnl::engine &eng, const dnnl::primitive_attr &matmul_attr,
                               int32_t weight_cache_type);
#endif

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif