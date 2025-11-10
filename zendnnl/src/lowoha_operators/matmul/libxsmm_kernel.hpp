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

#ifndef _LIBXSMM_KERNEL_HPP
#define _LIBXSMM_KERNEL_HPP

#include "lowoha_operators/matmul/lowoha_common.hpp"
#if ZENDNNL_DEPENDS_LIBXSMM
  #include "libxsmm.h"
#endif

namespace zendnnl {
namespace lowoha {

/**
 * @brief Check if LibXSMM can be used for the given matrix multiplication parameters
 *
 * @param transA Transpose flag for matrix A ('t' or 'n')
 * @param transB Transpose flag for matrix B ('t' or 'n')
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param alpha Scaling factor for A*B (must be 1.0 for LibXSMM)
 * @param beta Scaling factor for C (must be 0.0 for LibXSMM)
 * @param dtypes Data types for the operands
 * @return true if LibXSMM can handle this operation, false otherwise
 */
static inline bool can_use_libxsmm(char transA, char transB, int M,
                                   int N, int K, float alpha, float beta,
                                   const data_types &dtypes) {

  // Check if the matrix dimensions are within acceptable limits for LIBXSMM kernel selection
  // This heuristic prevents LIBXSMM from being used for matrices that are either:
  // 1. Too tall (M > 512) - LIBXSMM throws Segfault on very tall matrices
  // 2. Too large in terms of element count - when weight matrix B[K×N] > 1.0 Millions elements
#if ZENDNNL_DEPENDS_LIBXSMM
  float Max_Matrix_B_Elements = static_cast<float>(K * N) / 1000000.0f;
  if ((Max_Matrix_B_Elements > 1.0f) || (M > 512 &&
                                         Max_Matrix_B_Elements > 1.0f)) {
    return false;  // Fallback to BLIS
  }

  const bool scalars_ok = (alpha == 1.0f && beta == 0.0f);
  if (!scalars_ok) {
    return false;
  }

  //LIBXSMM throws segfault for transA='t' cases
  if (transA == 't') {
    return false;
  }

  if (transA == 't' && transB == 'n' &&
      dtypes.src == data_type_t::bf16 && (K & 1)) {
    return false;
  }

  const bool dtype_ok =
    (dtypes.src == data_type_t::f32  && dtypes.dst == data_type_t::f32) ||
    (dtypes.src == data_type_t::bf16 && dtypes.dst == data_type_t::f32) ||
    (dtypes.src == data_type_t::bf16 && dtypes.dst == data_type_t::bf16);

  return dtype_ok;
#endif
  return false;
}

#if ZENDNNL_DEPENDS_LIBXSMM
/**
 * @brief Template function for LibXSMM GEMM dispatch and execution
 */
template<typename TA, typename TB, typename TC>
int libxsmm_gemm(const TA *A, const TB *B, TC *C, int M, int N, int K, int lda,
                 int ldb, int ldc,
                 char transA, char transB, libxsmm_datatype a_type, libxsmm_datatype b_type,
                 libxsmm_datatype c_type, libxsmm_datatype comp_type) {
  libxsmm_bitfield l_flags = 0;
  if (transA == 'T' || transA == 't') {
    l_flags |= LIBXSMM_GEMM_FLAG_TRANS_B;
  }
  if (transB == 'T' || transB == 't') {
    l_flags |= LIBXSMM_GEMM_FLAG_TRANS_A;
  }
  l_flags |= LIBXSMM_GEMM_FLAG_BETA_0;

  libxsmm_gemm_shape shape{};
  shape.m   = N;
  shape.n   = M;
  shape.k   = K;
  shape.lda = ldb;
  shape.ldb = lda;
  shape.ldc = ldc;
  shape.a_in_type = a_type;
  shape.b_in_type = b_type;
  shape.out_type  = c_type;
  shape.comp_type = comp_type;

  libxsmm_gemm_batch_reduce_config brcfg{};
  brcfg.br_type = LIBXSMM_GEMM_BATCH_REDUCE_NONE;

  libxsmm_gemmfunction ker =
    libxsmm_dispatch_brgemm(shape, l_flags, 0, brcfg);

  if (!ker) {
    return 0;
  }

  libxsmm_gemm_param p{};
  p.a.primary = const_cast<TB *>(B);
  p.b.primary = const_cast<TA *>(A);
  p.c.primary = C;

  ker(&p);
  return 1;
}

/**
 * @brief Run LibXSMM GEMM with automatic type dispatch
 */
static inline int run_libxsmm(char transA, char transB, int M, int N, int K,
                              int lda, int ldb, int ldc,
                              const void *A, const void *B, void *C,
                              const data_types &dtypes) {
  int kernel_status = 0;
  if (dtypes.src == data_type_t::f32 && dtypes.dst == data_type_t::f32) {
    kernel_status = libxsmm_gemm<float,float,float>(
                      static_cast<const float *>(A),
                      static_cast<const float *>(B),
                      static_cast<float *>(C),
                      M,N,K, lda,ldb,ldc, transA,transB,
                      LIBXSMM_DATATYPE_F32,LIBXSMM_DATATYPE_F32,
                      LIBXSMM_DATATYPE_F32,LIBXSMM_DATATYPE_F32);
  }
  else if (dtypes.src == data_type_t::bf16 && dtypes.dst == data_type_t::f32) {
    kernel_status = libxsmm_gemm<libxsmm_bfloat16,libxsmm_bfloat16,float>(
                      reinterpret_cast<const libxsmm_bfloat16 *>(A),
                      reinterpret_cast<const libxsmm_bfloat16 *>(B),
                      static_cast<float *>(C),
                      M,N,K, lda,ldb,ldc, transA,transB,
                      LIBXSMM_DATATYPE_BF16,LIBXSMM_DATATYPE_BF16,
                      LIBXSMM_DATATYPE_F32,LIBXSMM_DATATYPE_F32);
  }
  else if (dtypes.src == data_type_t::bf16 && dtypes.dst == data_type_t::bf16) {
    kernel_status =
      libxsmm_gemm<libxsmm_bfloat16,libxsmm_bfloat16,libxsmm_bfloat16>(
        reinterpret_cast<const libxsmm_bfloat16 *>(A),
        reinterpret_cast<const libxsmm_bfloat16 *>(B),
        reinterpret_cast<libxsmm_bfloat16 *>(C),
        M,N,K, lda,ldb,ldc, transA,transB,
        LIBXSMM_DATATYPE_BF16,LIBXSMM_DATATYPE_BF16,
        LIBXSMM_DATATYPE_BF16,LIBXSMM_DATATYPE_F32);
  }
  return kernel_status;
}
#endif
}
}

#endif