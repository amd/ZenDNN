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
#endif
namespace zendnnl {
namespace lowoha {

#if ZENDNNL_DEPENDS_AOCLDLP
/**
* @brief Creates DLP (Deep Learning Post-op) metadata for post-operations.
*
* This function initializes and returns a pointer to `dlp_metadata_t` that
* encapsulates the post-operation metadata for matrix multiplication.
*
* @param lowoha_param The parameters for the low-overhead matrix multiplication.
* @param bias Pointer to the bias data.
* @param dtypes Data types for the source, weight, and destination tensors.
* @param N The number of columns in the output matrix.
* @return Pointer to the created `dlp_metadata_t` object.
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
 * @param transA/transB Transpose flags for matrices A and B ('t'/'n')
 * @param M,N,K Matrix dimensions (rows of A/C, cols of B/C, inner dimension)
 * @param alpha,beta Scaling factors for GEMM operation
 * @param lda,ldb,ldc Leading dimensions of matrices A, B, C
 * @param mem_format_a/b Memory format specifiers for matrices A and B
 * @param A,B,C Pointers to matrix data buffers
 * @param dtypes Data types for source, weight, and destination tensors
 * @param lowoha_param Parameters containing post-operations chain
 * @param bias Optional bias vector pointer (can be nullptr)
 */
void run_blis(char layout, char transA, char transB, int M, int N,
              int K,
              float alpha, float beta, int lda, int ldb, int ldc,
              char mem_format_a, char mem_format_b, const void *A,
              const void *B, void *C, const data_types &dtypes,
              const lowoha_params &lowoha_param, const void *bias);

/**
 * @brief Execute batched matrix multiplication using AOCL backend
 *
 * @param layout Memory layout ('r' for row-major, 'c' for column-major)
 * @param transA/transB Transpose flags for matrices A and B ('t'/'n')
 * @param M,N,K Matrix dimensions for each batch operation
 * @param alpha,beta Scaling factors applied to all batch operations
 * @param A,B,C Base pointers to batched matrix data
 * @param lda,ldb,ldc Leading dimensions of matrices A, B, C
 * @param dtypes Data types for source, weight, and destination tensors
 * @param batch_count Number of matrix multiplications in the batch
 * @param mem_format_a/b Memory format specifiers for matrices A and B
 * @param src_stride,weight_stride,dst_stride Byte offsets between batches
 * @param lowoha_param Parameters containing post-operations chain
 * @param bias Optional bias vector pointer (applied to all batches)
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