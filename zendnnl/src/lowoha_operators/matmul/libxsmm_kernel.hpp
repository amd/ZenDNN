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

#ifndef _LIBXSMM_KERNEL_HPP
#define _LIBXSMM_KERNEL_HPP

#include "lowoha_operators/matmul/lowoha_common.hpp"
#include "lowoha_operators/matmul/libxsmm_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

#if ZENDNNL_DEPENDS_LIBXSMM
/**
 * @brief Template function for LibXSMM GEMM dispatch and execution
 */
template<typename TA, typename TB, typename TC>
int libxsmm_gemm(const TA *A, const TB *B, TC *C, int M, int N, int K,
                 float beta, int lda, int ldb, int ldc,
                 char transA, char transB, libxsmm_datatype a_type, libxsmm_datatype b_type,
                 libxsmm_datatype c_type, libxsmm_datatype comp_type,
                 const matmul_params &lowoha_param, const void *bias,
                 const data_type_t &bias_type) {
  libxsmm_bitfield l_flags = 0;
  if (transA == 'T' || transA == 't') {
    l_flags |= LIBXSMM_GEMM_FLAG_TRANS_B;
  }
  if (transB == 'T' || transB == 't') {
    l_flags |= LIBXSMM_GEMM_FLAG_TRANS_A;
  }
  if (beta == 0.0f) {
    l_flags |= LIBXSMM_GEMM_FLAG_BETA_0;
  }

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

  if (bias != nullptr) {
    if (bias_type == data_type_t::f32) {
      libxsmm_bias<TC, float>(M, N, ldc, C, bias);
    }
    else if (bias_type == data_type_t::bf16) {
      libxsmm_bias<TC, libxsmm_bfloat16>(M, N, ldc, C, bias);
    }
  }

  if (lowoha_param.postop_.size() > 0) {
    for (const auto &postop : lowoha_param.postop_) {
      libxsmm_postop<TC>(M, N, ldc, C, postop);
    }
  }
  return 1;
}
#if ENABLE_LIBXSMM_BRGEMM_KERNEL
/**
 * @brief Template function for LibXSMM BRGEMM dispatch and execution
 */
template<typename TA, typename TB, typename TC>
int libxsmm_brgemm(const TA **A_ptrs, const TB **B_ptrs, TC *C,
                   int M, int N, int K, int batch,
                   float beta, int lda, int ldb, int ldc,
                   char transA, char transB,
                   libxsmm_datatype a_type, libxsmm_datatype b_type,
                   libxsmm_datatype c_type, libxsmm_datatype comp_type,
                   const matmul_params &lowoha_param, const void *bias,
                   const data_type_t &bias_type, bool apply_postops = true) {

  libxsmm_bitfield l_flags = 0;

  if (transA == 'T' || transA == 't') {
    l_flags |= LIBXSMM_GEMM_FLAG_TRANS_B;
  }
  if (transB == 'T' || transB == 't') {
    l_flags |= LIBXSMM_GEMM_FLAG_TRANS_A;
  }
  if (beta == 0.0f) {
    l_flags |= LIBXSMM_GEMM_FLAG_BETA_0;
  }
  libxsmm_gemm_shape shape{};
  shape.m   = N;
  shape.n   = M;
  shape.k   = K;
  shape.lda = ldb;
  shape.ldb = lda;
  shape.ldc = ldc;
  shape.a_in_type = b_type;
  shape.b_in_type = a_type;
  shape.out_type  = c_type;
  shape.comp_type = comp_type;

  libxsmm_gemm_batch_reduce_config brcfg{};
  brcfg.br_type = LIBXSMM_GEMM_BATCH_REDUCE_ADDRESS;
  brcfg.br_unroll_hint = batch;

  libxsmm_gemmfunction ker = libxsmm_dispatch_brgemm(shape, l_flags, 0, brcfg);

  if (!ker) {
    log_error("Failed to dispatch LibXSMM BRGEMM kernel");
    return 0;
  }

  libxsmm_gemm_param p{};
  p.a.primary = const_cast<void *>(static_cast<const void *>(B_ptrs));
  p.b.primary = const_cast<void *>(static_cast<const void *>(A_ptrs));
  p.c.primary = C;
  p.op.tertiary = &batch;

  ker(&p);

  if (bias != nullptr) {
    if (bias_type == data_type_t::f32) {
      libxsmm_bias<TC, float>(M, N, ldc, C, bias);
    }
    else if (bias_type == data_type_t::bf16) {
      libxsmm_bias<TC, libxsmm_bfloat16>(M, N, ldc, C, bias);
    }
  }

  if (lowoha_param.postop_.size() > 0 && apply_postops) {
    for (const auto &postop : lowoha_param.postop_) {
      libxsmm_postop<TC>(M, N, ldc, C, postop);
    }
  }

  return 1;
}



static inline int run_libxsmm_brgemm(char transA, char transB,
                                     int M, int N, int K, int batch,
                                     float beta, int lda, int ldb, int ldc,
                                     const void **A_ptrs, const void **B_ptrs, void *C,
                                     const matmul_data_types &dtypes,
                                     const matmul_params &lowoha_para,
                                     const void *bias, bool apply_postops) {
  int kernel_status = 0;

  if (dtypes.src == data_type_t::f32 && dtypes.dst == data_type_t::f32) {
    log_info("Using libxsmm BRGEMM kernel");
    kernel_status = libxsmm_brgemm<float, float, float>(
                      reinterpret_cast<const float **>(A_ptrs),
                      reinterpret_cast<const float **>(B_ptrs),
                      static_cast<float *>(C),
                      M, N, K, batch, beta, lda, ldb, ldc, transA, transB,
                      LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32,
                      LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32,
                      lowoha_para, bias, dtypes.bias, apply_postops);
  }
  return kernel_status;
}
#endif

/**
 * @brief Run LibXSMM GEMM with automatic type dispatch
 */
static inline int run_libxsmm(char transA, char transB, int M, int N, int K,
                              float beta, int lda, int ldb, int ldc,
                              const void *A, const void *B, void *C,
                              const matmul_data_types &dtypes, const matmul_params &lowoha_para,
                              const void *bias) {
  int kernel_status = 0;
  if (dtypes.src == data_type_t::f32 && dtypes.dst == data_type_t::f32) {
    kernel_status = libxsmm_gemm<float,float,float>(
                      static_cast<const float *>(A),
                      static_cast<const float *>(B),
                      static_cast<float *>(C),
                      M,N,K, beta, lda,ldb,ldc, transA,transB,
                      LIBXSMM_DATATYPE_F32,LIBXSMM_DATATYPE_F32,
                      LIBXSMM_DATATYPE_F32,LIBXSMM_DATATYPE_F32, lowoha_para, bias, dtypes.bias);
  }
  else if (dtypes.src == data_type_t::bf16 && dtypes.dst == data_type_t::f32) {
    kernel_status = libxsmm_gemm<libxsmm_bfloat16,libxsmm_bfloat16,float>(
                      reinterpret_cast<const libxsmm_bfloat16 *>(A),
                      reinterpret_cast<const libxsmm_bfloat16 *>(B),
                      static_cast<float *>(C),
                      M,N,K, beta, lda,ldb,ldc, transA,transB,
                      LIBXSMM_DATATYPE_BF16,LIBXSMM_DATATYPE_BF16,
                      LIBXSMM_DATATYPE_F32,LIBXSMM_DATATYPE_F32, lowoha_para, bias, dtypes.bias);
  }
  else if (dtypes.src == data_type_t::bf16 && dtypes.dst == data_type_t::bf16) {
    kernel_status =
      libxsmm_gemm<libxsmm_bfloat16,libxsmm_bfloat16,libxsmm_bfloat16>(
        reinterpret_cast<const libxsmm_bfloat16 *>(A),
        reinterpret_cast<const libxsmm_bfloat16 *>(B),
        reinterpret_cast<libxsmm_bfloat16 *>(C),
        M,N,K, beta, lda,ldb,ldc, transA,transB,
        LIBXSMM_DATATYPE_BF16,LIBXSMM_DATATYPE_BF16,
        LIBXSMM_DATATYPE_BF16,LIBXSMM_DATATYPE_F32, lowoha_para, bias, dtypes.bias);
  }
  return kernel_status;
}
#endif
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif
