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

#ifndef _AOCL_KERNEL_HPP
#define _AOCL_KERNEL_HPP

#include "lowoha_operators/matmul/lowoha_common.hpp"

#if ZENDNNL_DEPENDS_AOCLDLP
  #include "aocl_dlp.h"
#else
  #include "blis.h"
  using md_t = dim_t;
#endif
namespace zendnnl {
namespace lowoha {
using get_reorder_buff_size_func_ptr = long unsigned int (*)(const char,
                                       const char, const char, const md_t, const md_t
#if ZENDNNL_DEPENDS_AOCLDLP
  ,dlp_metadata_t *
#endif
                                                            );

template <typename T>
using reorder_func_ptr = void (*)(const char, const char, const char, const T *,
                                  T *, const md_t, const md_t, const md_t
#if ZENDNNL_DEPENDS_AOCLDLP
  ,dlp_metadata_t *
#endif
                                 );

/**
 * @brief Reorders and caches weight matrices for optimized memory access patterns
 *
 * This template function performs weight reordering to optimize memory access patterns
 * for specific GEMM kernels and implements a caching mechanism to avoid redundant
 * reordering operations for the same weight tensors. The function checks if the weights
 * have been previously reordered and cached; if so, it returns the cached version.
 * Otherwise, it performs the reordering operation and stores the result in the cache.
 *
 * @tparam T Data type of the weight elements (e.g., float, bfloat16, int8_t)
 * @param key Unique key identifying the weight tensor and reordering parameters
 * @param weights Pointer to the original (non-reordered) weight data
 * @param reorder_weights Reference to pointer that will hold the reordered weights
 * @param k Matrix dimension K (inner dimension for GEMM operation)
 * @param n Matrix dimension N (number of columns in the output matrix)
 * @param ldb Leading dimension of matrix B (must be >= k or n depending on layout)
 * @param order Memory layout order ('r' for row-major, 'c' for column-major)
 * @param trans Transpose flag ('t' for transposed, 'n' for not transposed)
 * @param mem_format_b Memory format specifier for matrix B
 * @param get_reorder_buf_size Function pointer to calculate required buffer size for reordering
 * @param reorder_func Function pointer to perform the actual reordering operation
 * @param weight_cache_type Type of caching strategy to use (0: no cache, 1: persistent cache, etc.)
 * @return true if reordering was performed (cache miss), false if cached version was used (cache hit)
 */
template <typename T>
bool reorderAndCacheWeights(Key_matmul key, const void *weights,
                            void *&reorder_weights, const int k, const int n, const int ldb,
                            const char order, const char trans, char mem_format_b,
                            get_reorder_buff_size_func_ptr get_reorder_buf_size,
                            reorder_func_ptr<T> reorder_func, int weight_cache_type);

#if ZENDNNL_DEPENDS_AOCLDLP
/**
 * @brief Creates DLP metadata for post-operations
 *
 * This function initializes and returns a pointer to dlp_metadata_t that
 * encapsulates the post-operation metadata for matrix multiplication
 *
 * @param lowoha_param The parameters containing the post-operations chain specification
 * @param bias Pointer to the bias data to be added (can be nullptr if no bias)
 * @param dtypes Data types structure specifying source, weight, and destination tensor types
 * @param N The number of columns in the output matrix
 * @return Pointer to the created dlp_metadata_t object
 */
dlp_metadata_t *create_dlp_post_op(const lowoha_params &lowoha_param,
                                   const void *bias, const data_types &dtypes, int N);

/**
* @brief Cleans up DLP (Deep Learning Post-op) metadata.
*
* This function releases the resources allocated for the `dlp_metadata_t`
* object used in post-operations.
*
* @param aocl_po Pointer to the `dlp_metadata_t` object to be cleaned up.
* @param post_op The parameters for the post-operation.
*/
void cleanup_dlp_post_op(dlp_metadata_t *aocl_po, const lowoha_params &post_op);

#else
/**
* @brief Creates BLIS (Basic Linear Algebra Subprograms) post-op metadata.
*
* This function initializes and returns a pointer to `aocl_post_op` that
* encapsulates the post-operation metadata for matrix multiplication.
*
* @param lowoha_param The parameters for the low-overhead matrix multiplication.
* @param bias Pointer to the bias data.
* @param dtypes Data types for the source, weight, and destination tensors.
* @param N The number of columns in the output matrix.
* @return Pointer to the created `aocl_post_op` object.
*/
aocl_post_op *create_blis_post_op(const lowoha_params &lowoha_param,
                                  const void *bias, const data_types &dtypes, int N);

/**
* @brief Cleans up BLIS (Basic Linear Algebra Subprograms) post-op metadata.
*
* This function releases the resources allocated for the `aocl_post_op`
* object used in post-operations.
*
* @param aocl_po Pointer to the `aocl_post_op` object to be cleaned up.
* @param post_op The parameters for the post-operation.
*/
void cleanup_blis_post_op(aocl_post_op *aocl_po, const lowoha_params &post_op);
#endif

/**
 * @brief Execute single matrix multiplication using AOCL BLIS backend
 *
 * Performs C = alpha * op(A) * op(B) + beta * C using AMD's optimized
 * AOCL library with support for post-operations and bias addition.
 * Automatically dispatches to appropriate data type specializations.
 *
 * @param layout Memory layout ('r' for row-major, 'c' for column-major)
 * @param transA Transpose flag for matrix A ('t' for transpose, 'n' for no transpose)
 * @param transB Transpose flag for matrix B ('t' for transpose, 'n' for no transpose)
 * @param M Number of rows in matrix A and output matrix C
 * @param N Number of columns in matrix B and output matrix C
 * @param K Inner dimension (columns of A, rows of B after potential transpose)
 * @param alpha Scaling factor for the product of A and B
 * @param beta Scaling factor for the existing values in C
 * @param lda Leading dimension of matrix A (stride between rows/columns)
 * @param ldb Leading dimension of matrix B (stride between rows/columns)
 * @param ldc Leading dimension of matrix C (stride between rows/columns)
 * @param mem_format_a Memory format specifier for matrix A
 * @param mem_format_b Memory format specifier for matrix B
 * @param A Pointer to matrix A data buffer
 * @param B Pointer to matrix B (weights) data buffer
 * @param C Pointer to matrix C (output) data buffer
 * @param dtypes Data types structure specifying src, weight, and dst tensor types
 * @param lowoha_param Parameters containing the post-operations chain
 * @param bias Optional bias vector pointer (can be nullptr if no bias)
 * @param kernel Algorithm selection for GEMM execution
 * @param is_weights_const Flag indicating if weights are constant (enables caching)
 * @param can_reorder Flag indicating if weight reordering is allowed (default: false)
 */
void run_blis(char layout, char transA, char transB, int M, int N,
              int K,
              float alpha, float beta, int lda, int ldb, int ldc,
              char mem_format_a, char mem_format_b, const void *A,
              const void *B, void *C, const data_types &dtypes,
              const lowoha_params &lowoha_param, const void *bias,
              zendnnl::ops::matmul_algo_t kernel,
              bool is_weights_const, bool can_reorder = false);

/**
 * @brief Execute batched matrix multiplication using AOCL backend
 *
 * @param layout Memory layout ('r' for row-major, 'c' for column-major)
 * @param transA Transpose flag for matrices A ('t' for transpose, 'n' for no transpose)
 * @param transB Transpose flag for matrices B ('t' for transpose, 'n' for no transpose)
 * @param M Number of rows in each A matrix and output C matrix
 * @param N Number of columns in each B matrix and output C matrix
 * @param K Inner dimension for each GEMM operation
 * @param alpha Scaling factor applied to all A*B products across all batches
 * @param A Base pointer to the first batch of matrix A data
 * @param lda Leading dimension of each matrix A
 * @param B Base pointer to the first batch of matrix B (weights) data
 * @param ldb Leading dimension of each matrix B
 * @param beta Scaling factor applied to all existing C values across all batches
 * @param C Base pointer to the first batch of matrix C (output) data
 * @param ldc Leading dimension of each matrix C
 * @param dtypes Data types structure specifying src, weight, and dst tensor types
 * @param batch_count Number of independent matrix multiplications to perform
 * @param mem_format_a Memory format specifier for matrices A
 * @param mem_format_b Memory format specifier for matrices B
 * @param src_stride Byte offset between consecutive A matrices in the batch
 * @param weight_stride Byte offset between consecutive B matrices in the batch
 * @param dst_stride Byte offset between consecutive C matrices in the batch
 * @param lowoha_param Parameters containing post-operations chain applied to all batches
 * @param bias Optional bias vector pointer applied to all batches (can be nullptr)
 */
void matmul_batch_gemm_wrapper(char layout, char transA, char transB, int M,
                               int N, int K, float alpha, const void *A, int lda, const void *B, int ldb,
                               float beta,
                               void *C, int ldc, data_types &dtypes, int batch_count, char mem_format_a,
                               char mem_format_b, size_t src_stride, size_t weight_stride,
                               size_t dst_stride, const lowoha_params &lowoha_param, const void *bias);
}
}

#endif //_AOCL_KERNEL_HPP