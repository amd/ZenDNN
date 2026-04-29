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
#include "lowoha_operators/matmul/backends/libxsmm/libxsmm_utils.hpp"
#include <cstring>

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

  libxsmm_gemmfunction kernel =
    libxsmm_dispatch_brgemm(shape, l_flags, 0, brcfg);

  if (!kernel) {
    return 0;
  }

  libxsmm_gemm_param p{};
  p.a.primary = const_cast<TB *>(B);
  p.b.primary = const_cast<TA *>(A);
  p.c.primary = C;

  kernel(&p);

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
                   const data_type_t &bias_type, bool apply_postops = true,
                   libxsmm_gemmfunction precompiled = nullptr) {
  libxsmm_gemmfunction kernel = precompiled;

  if (!kernel) {
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

    kernel = libxsmm_dispatch_brgemm(shape, l_flags, 0, brcfg);
  }

  if (!kernel) {
    log_error("Failed to dispatch LibXSMM BRGEMM kernel");
    return 0;
  }

  libxsmm_gemm_param p{};
  p.a.primary = const_cast<void *>(static_cast<const void *>(B_ptrs));
  p.b.primary = const_cast<void *>(static_cast<const void *>(A_ptrs));
  p.c.primary = C;
  p.op.tertiary = &batch;

  kernel(&p);

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

/**
 * @brief Pre-dispatched stride-based BRGEMM kernel set.
 *
 * Holds JIT function pointers for every unique (tile-shape, K-variant)
 * combination so the hot loop needs zero dispatch / hash-map lookups.
 * Indexed as [is_m_tail][is_n_tail].
 *
 * main_ker: kernels with K = k_block_size  (full K-blocks), count = k_blocks_per_reduce
 * tail_ker: kernels with K = k_block_rem   (K-tail),        count = 1
 */
struct PreDispatchedBrgemm {
  libxsmm_gemmfunction main_ker[2][2] = {};
  libxsmm_gemmfunction tail_ker[2][2] = {};
  int m_sizes[2] = {};
  int n_sizes[2] = {};
};

/**
 * @brief Resolve libxsmm data types from matmul_data_types.
 */
static inline void resolve_xsmm_types(
  const matmul_data_types &dtypes,
  libxsmm_datatype &a_dt, libxsmm_datatype &b_dt, libxsmm_datatype &c_dt) {
  if (dtypes.src == data_type_t::f32 && dtypes.dst == data_type_t::f32) {
    a_dt = LIBXSMM_DATATYPE_F32;
    b_dt = LIBXSMM_DATATYPE_F32;
    c_dt = LIBXSMM_DATATYPE_F32;
  }
  else if (dtypes.src == data_type_t::bf16 && dtypes.dst == data_type_t::f32) {
    a_dt = LIBXSMM_DATATYPE_BF16;
    b_dt = LIBXSMM_DATATYPE_BF16;
    c_dt = LIBXSMM_DATATYPE_F32;
  }
  else {
    a_dt = LIBXSMM_DATATYPE_BF16;
    b_dt = LIBXSMM_DATATYPE_BF16;
    c_dt = LIBXSMM_DATATYPE_BF16;
  }
}

/**
 * @brief Dispatch a BRGEMM kernel directly via libxsmm JIT.
 *
 * Called only from predispatch_brgemm_kernels(); the result is stored
 * in a plain array so no hash cache or mutex is needed.
 *
 * M/N are in the caller's (row-major) convention; the function swaps
 * them for libxsmm's column-major expectation.
 *
 * @param vnni  If true, sets LIBXSMM_GEMM_FLAG_VNNI_A (blocked BF16 weight)
 */
static inline libxsmm_gemmfunction dispatch_brgemm(
  char transA, char transB,
  int M, int N, int K, int count,
  int lda, int ldb, int ldc,
  unsigned long long stride_a, unsigned long long stride_b,
  const matmul_data_types &dtypes, bool vnni = false) {

  libxsmm_bitfield l_flags = 0;
  if (transA == 'T' || transA == 't') {
    l_flags |= LIBXSMM_GEMM_FLAG_TRANS_B;
  }
  if (transB == 'T' || transB == 't') {
    l_flags |= LIBXSMM_GEMM_FLAG_TRANS_A;
  }
  if (vnni) {
    l_flags |= LIBXSMM_GEMM_FLAG_VNNI_A;
  }

  libxsmm_datatype a_dt, b_dt, c_dt;
  resolve_xsmm_types(dtypes, a_dt, b_dt, c_dt);

  libxsmm_gemm_shape shape{};
  shape.m   = N;
  shape.n   = M;
  shape.k   = K;
  shape.lda = ldb;
  shape.ldb = lda;
  shape.ldc = ldc;
  shape.a_in_type  = b_dt;
  shape.b_in_type  = a_dt;
  shape.out_type   = c_dt;
  shape.comp_type  = LIBXSMM_DATATYPE_F32;

  libxsmm_gemm_batch_reduce_config brcfg{};
  brcfg.br_type          = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
  brcfg.br_unroll_hint   = count;
  brcfg.br_stride_a_hint = stride_b;   // libxsmm's A = our B
  brcfg.br_stride_b_hint = stride_a;   // libxsmm's B = our A

  return libxsmm_dispatch_brgemm(shape, l_flags, /*prefetch_flags=*/0, brcfg);
}

/**
 * @brief Execute a pre-dispatched stride-based BRGEMM kernel.
 *
 * No dispatch or hash lookup — just fills gemm_param and calls the kernel.
 * Strides are baked into the JIT kernel at dispatch time.
 *
 * @param kernel  Pre-dispatched function pointer
 * @param A_base  Base pointer to the first A panel for this tile
 * @param B_base  Base pointer to the first B panel for this tile
 * @param C       Pointer to the output tile (already initialized)
 * @param count   Number of K-panels to batch-reduce
 */
static inline void run_brgemm(
  libxsmm_gemmfunction kernel,
  const void *A_base, const void *B_base, void *C,
  unsigned long long count) {

  if (!kernel) {
    log_error("Failed to dispatch LibXSMM BRGEMM kernel");
    return;
  }

  libxsmm_gemm_param p{};
  p.a.primary  = const_cast<void *>(B_base);   // libxsmm's A = our B
  p.b.primary  = const_cast<void *>(A_base);   // libxsmm's B = our A
  p.c.primary  = C;
  p.op.tertiary = &count;

  kernel(&p);
}

/**
 * @brief Pre-dispatch BRGEMM kernels for all tile shapes.
 *
 * Called once before the parallel region. Dispatches kernel variants for
 * every (M, N, K) tile combination:
 *   main_ker[is_m_rem][is_n_rem] — full K-blocks, count = k_blocks_per_reduce
 *   tail_ker[is_m_rem][is_n_rem] — K-tail (k_block_rem), count = 1
 *
 * For blocked weights (no K/N remainders), only main_ker[0][0] and
 * main_ker[1][0] are populated; tail_ker stays null.
 *
 * All kernels use beta=1.0 (output tile is pre-initialized with bias or zero).
 *
 * @param vnni  If true, sets LIBXSMM_GEMM_FLAG_VNNI_A (blocked BF16 weight)
 */
static inline PreDispatchedBrgemm predispatch_brgemm_kernels(
  char transA, char transB,
  int m_block_size, int m_block_rem,
  int n_block_size, int n_block_rem,
  int k_block_size, int k_block_rem,
  int num_k_blocks, int k_blocks_per_reduce,
  int lda, int ldb, int ldc,
  unsigned long long stride_a, unsigned long long stride_b,
  const matmul_data_types &dtypes, bool vnni = false) {

  PreDispatchedBrgemm pd;
  pd.m_sizes[0] = m_block_size;
  pd.m_sizes[1] = m_block_rem;
  pd.n_sizes[0] = n_block_size;
  pd.n_sizes[1] = n_block_rem;

  for (int mi = 0; mi < 2; ++mi) {
    if (pd.m_sizes[mi] <= 0) {
      continue;
    }
    for (int ni = 0; ni < 2; ++ni) {
      if (pd.n_sizes[ni] <= 0) {
        continue;
      }
      int m = pd.m_sizes[mi];
      int n = pd.n_sizes[ni];

      if (num_k_blocks > 0 && k_block_size > 0) {
        pd.main_ker[mi][ni] = dispatch_brgemm(
                                transA, transB, m, n, k_block_size, k_blocks_per_reduce,
                                lda, ldb, ldc, stride_a, stride_b, dtypes, vnni);
      }
      if (k_block_rem > 0) {
        pd.tail_ker[mi][ni] = dispatch_brgemm(
                                transA, transB, m, n, k_block_rem, 1,
                                lda, ldb, ldc, stride_a, stride_b, dtypes, vnni);
      }
    }
  }
  return pd;
}

/**
 * @brief Initialize an output tile before BRGEMM accumulation.
 *
 * When beta == 0: zeros the tile, then adds bias (if present).
 * When beta == 1: preserves existing C values, then adds bias (if present).
 *
 * After this call the tile is ready for beta=1.0 BRGEMM accumulation (C += A*B).
 */
static inline void init_output_tile(
  void *C_tile, const void *bias,
  int m_len, int n_len, int ldc,
  const matmul_data_types &dtypes, float beta = 0.0f) {

  if (beta == 0.0f) {
    size_t elem_size = (dtypes.dst == data_type_t::f32) ? sizeof(float)
                       : sizeof(libxsmm_bfloat16);
    uint8_t *C = static_cast<uint8_t *>(C_tile);
    size_t row_bytes = static_cast<size_t>(n_len) * elem_size;

    if (n_len == ldc) {
      std::memset(C, 0, static_cast<size_t>(m_len) * row_bytes);
    }
    else {
      size_t stride = static_cast<size_t>(ldc) * elem_size;
      for (int i = 0; i < m_len; ++i) {
        std::memset(C + i * stride, 0, row_bytes);
      }
    }
  }

  if (bias) {
    if (dtypes.dst == data_type_t::f32) {
      if (dtypes.bias == data_type_t::f32) {
        libxsmm_bias<float, float>(m_len, n_len, ldc, C_tile, bias);
      }
      else if (dtypes.bias == data_type_t::bf16) {
        libxsmm_bias<float, libxsmm_bfloat16>(m_len, n_len, ldc, C_tile, bias);
      }
    }
    else {
      if (dtypes.bias == data_type_t::bf16)
        libxsmm_bias<libxsmm_bfloat16, libxsmm_bfloat16>(m_len, n_len, ldc,
            C_tile, bias);
      else if (dtypes.bias == data_type_t::f32) {
        libxsmm_bias<libxsmm_bfloat16, float>(m_len, n_len, ldc, C_tile, bias);
      }
    }
  }
}

/**
 * @brief Run LibXSMM BRGEMM with automatic type dispatch
 */
static inline int run_libxsmm_brgemm(char transA, char transB,
                                     int M, int N, int K, int batch,
                                     float beta, int lda, int ldb, int ldc,
                                     const void **A_ptrs, const void **B_ptrs, void *C,
                                     const matmul_data_types &dtypes,
                                     const matmul_params &lowoha_para,
                                     const void *bias, bool apply_postops,
                                     libxsmm_gemmfunction precompiled = nullptr) {
  int kernel_status = 0;

  if (dtypes.src == data_type_t::f32 && dtypes.dst == data_type_t::f32) {
    log_info("Using libxsmm BRGEMM f32->f32 kernel");
    kernel_status = libxsmm_brgemm<float, float, float>(
                      reinterpret_cast<const float **>(A_ptrs),
                      reinterpret_cast<const float **>(B_ptrs),
                      static_cast<float *>(C),
                      M, N, K, batch, beta, lda, ldb, ldc, transA, transB,
                      LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32,
                      LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32,
                      lowoha_para, bias, dtypes.bias, apply_postops,
                      precompiled);
  }
  else if (dtypes.src == data_type_t::bf16 && dtypes.dst == data_type_t::f32) {
    log_info("Using libxsmm BRGEMM bf16->f32 kernel");
    kernel_status = libxsmm_brgemm<libxsmm_bfloat16, libxsmm_bfloat16, float>(
                      reinterpret_cast<const libxsmm_bfloat16 **>(A_ptrs),
                      reinterpret_cast<const libxsmm_bfloat16 **>(B_ptrs),
                      static_cast<float *>(C),
                      M, N, K, batch, beta, lda, ldb, ldc, transA, transB,
                      LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16,
                      LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32,
                      lowoha_para, bias, dtypes.bias, apply_postops,
                      precompiled);
  }
  else if (dtypes.src == data_type_t::bf16 && dtypes.dst == data_type_t::bf16) {
    log_info("Using libxsmm BRGEMM bf16->bf16 kernel");
    kernel_status =
      libxsmm_brgemm<libxsmm_bfloat16, libxsmm_bfloat16, libxsmm_bfloat16>(
        reinterpret_cast<const libxsmm_bfloat16 **>(A_ptrs),
        reinterpret_cast<const libxsmm_bfloat16 **>(B_ptrs),
        reinterpret_cast<libxsmm_bfloat16 *>(C),
        M, N, K, batch, beta, lda, ldb, ldc, transA, transB,
        LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16,
        LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32,
        lowoha_para, bias, dtypes.bias, apply_postops,
        precompiled);
  }
  return kernel_status;
}

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
    log_info("Using libxsmm GEMM f32->f32 kernel");
    kernel_status = libxsmm_gemm<float,float,float>(
                      static_cast<const float *>(A),
                      static_cast<const float *>(B),
                      static_cast<float *>(C),
                      M,N,K, beta, lda,ldb,ldc, transA,transB,
                      LIBXSMM_DATATYPE_F32,LIBXSMM_DATATYPE_F32,
                      LIBXSMM_DATATYPE_F32,LIBXSMM_DATATYPE_F32, lowoha_para, bias, dtypes.bias);
  }
  else if (dtypes.src == data_type_t::bf16 && dtypes.dst == data_type_t::f32) {
    log_info("Using libxsmm GEMM bf16->f32 kernel");
    kernel_status = libxsmm_gemm<libxsmm_bfloat16,libxsmm_bfloat16,float>(
                      reinterpret_cast<const libxsmm_bfloat16 *>(A),
                      reinterpret_cast<const libxsmm_bfloat16 *>(B),
                      static_cast<float *>(C),
                      M,N,K, beta, lda,ldb,ldc, transA,transB,
                      LIBXSMM_DATATYPE_BF16,LIBXSMM_DATATYPE_BF16,
                      LIBXSMM_DATATYPE_F32,LIBXSMM_DATATYPE_F32, lowoha_para, bias, dtypes.bias);
  }
  else if (dtypes.src == data_type_t::bf16 && dtypes.dst == data_type_t::bf16) {
    log_info("Using libxsmm GEMM bf16->bf16 kernel");
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
