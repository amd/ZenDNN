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

#ifndef _ONEDNN_KERNEL_HPP
#define _ONEDNN_KERNEL_HPP

#include "lowoha_operators/matmul/lowoha_common.hpp"
#if ZENDNNL_DEPENDS_ONEDNN
  #include "operators/matmul/onednn/matmul_onednn_kernel.hpp"
  using namespace dnnl;
#endif

namespace zendnnl {
namespace lowoha {

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
 * @param batchA,batchB Batch sizes for matrices A and B
 * @param bias Optional bias vector pointer
 * @param kernel Algorithm selection for the matmul operation
 * @param weight_cache_type Caching strategy for weight reordering
 */
void matmul_onednn_wrapper(char transA, char transB, int M, int N,
                           int K, float alpha, const void *A, int lda, const void *B, int ldb, float beta,
                           void *C, int ldc, lowoha_params &lowoha_params, int batchA, int batchB,
                           const void *bias, zendnnl::ops::matmul_algo_t kernel, bool is_weights_const);
/**
 * @brief Reorder weights to OneDNN's optimal format with caching
 *
 * Converts weight matrix to OneDNN's internal optimized memory layout
 * and caches the result to avoid redundant reordering operations.
 *
 * @param key Unique cache key for the weight tensor
 * @param dnnl_params OneDNN parameters containing weight memory info
 * @param weight_cache_type Type of caching mechanism to use
 * @param matmul_attr OneDNN primitive attributes for the matmul operation
 * @param eng OneDNN engine for memory operations
 * @return true if reordering was performed
 */
bool reorderAndCacheWeights(Key_matmul key,
                            onednn_utils_t::onednn_matmul_params &dnnl_params, int weight_cache_type,
                            dnnl::primitive_attr matmul_attr, dnnl::engine &eng);
/**
 * @brief Reorder weights to OneDNN's optimal format without caching
 *
 * Performs one-time weight matrix reordering from user format to
 * OneDNN's internal optimized memory layout for better performance.
 *
 * @param dnnl_params OneDNN parameters containing source and destination memory
 * @param matmul_attr OneDNN primitive attributes for the matmul operation
 * @param eng OneDNN engine for executing the reorder operation
 */
void reorderWeights(onednn_utils_t::onednn_matmul_params &dnnl_params,
                    dnnl::primitive_attr matmul_attr, dnnl::engine &eng);
#endif

} // lowoha namespace
} // zendnnl namespace

#endif