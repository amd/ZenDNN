/*******************************************************************************
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

#include "lowoha_matmul.hpp"
#include "matmul_kernel_bf16_avx512.hpp"
#include <cmath>
#include <cstring>
#if ZENDNNL_DEPENDS_LIBXSMM
  #include "libxsmm.h"
#endif

namespace zendnnl {
namespace lowoha {

using namespace zendnnl::ops;

inline int64_t divup(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

template <class F>
inline void zendnnl_parallel_for(
  const int64_t begin,
  const int64_t end,
  const int64_t grain_size,
  const F &f) {

  if (begin >= end) {
    return;
  }
  std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
  std::exception_ptr eptr;
  // choose number of tasks based on grain size and number of threads
  int64_t num_threads = omp_in_parallel() ? 1 : omp_get_max_threads();
  if (grain_size > 0) {
    num_threads = std::min(num_threads, divup((end - begin), grain_size));
  }

  #pragma omp parallel num_threads(num_threads)
  {
    int64_t num_threads = omp_get_num_threads();
    int64_t tid = omp_get_thread_num();
    int64_t chunk_size = divup((end - begin), num_threads);
    int64_t begin_tid = begin + tid * chunk_size;
    if (begin_tid < end) {
      try {
        f(begin_tid, std::min(end, chunk_size + begin_tid));
      }
      catch (...) {
        if (!err_flag.test_and_set()) {
          eptr = std::current_exception();
        }
      }
    }
  }
  if (eptr) {
    std::rethrow_exception(eptr);
  }
}

void matmul_direct_native(char layout, char transA, char transB, int M, int N,
                          int K, float alpha, const void *A, int lda,
                          const void *B, int ldb, float beta, void *C, int ldc, data_types dtypes) {

  const bool is_f32_src  = (dtypes.src == data_type_t::f32);
  const bool is_bf16_src = (dtypes.src == data_type_t::bf16);
  const bool is_f32_out  = (dtypes.dst == data_type_t::f32);
  const bool is_bf16_out = (dtypes.dst == data_type_t::bf16);

  const float *A_f32    = is_f32_src ? static_cast<const float *>(A) : nullptr;
  const float *B_f32    = is_f32_src ? static_cast<const float *>(B) : nullptr;
  const int16_t *A_bf16 = is_bf16_src ? static_cast<const int16_t *>(A) : nullptr;
  const int16_t *B_bf16 = is_bf16_src ? static_cast<const int16_t *>(B) : nullptr;

  float *C_f32    = is_f32_out ? static_cast<float *>(C) : nullptr;
  int16_t *C_bf16 = is_bf16_out ? static_cast<int16_t *>(C) : nullptr;

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        float a_val = 0.0f;
        float b_val = 0.0f;

        if (is_f32_src) {
          a_val = (transA == 'n') ? A_f32[m * lda + k] : A_f32[k * lda + m];
          b_val = (transB == 'n') ? B_f32[k * ldb + n] : B_f32[n * ldb + k];
        }
        else if (is_bf16_src) {
          a_val = bfloat16_t::bf16_to_f32_val((transA == 'n') ? A_bf16[m * lda + k] :
                                              A_bf16[k * lda + m]);
          b_val = bfloat16_t::bf16_to_f32_val((transB == 'n') ? B_bf16[k * ldb + n] :
                                              B_bf16[n * ldb + k]);
        }

        acc += a_val * b_val;
      }

      if (is_f32_out) {
        float c_val        = C_f32[m * ldc + n];
        C_f32[m * ldc + n] = alpha * acc + beta * c_val;
      }
      else if (is_bf16_out) {
        float c_val         = bfloat16_t::bf16_to_f32_val(C_bf16[m * ldc + n]);
        float result        = alpha * acc + beta * c_val;
        C_bf16[m * ldc + n] = bfloat16_t::f32_to_bf16_val(result);
      }
    }
  }
}


#if ZENDNNL_DEPENDS_LIBXSMM
template<typename TA, typename TB, typename TC>
void libxsmm_gemm(const TA *A, const TB *B, TC *C,
                  int M, int N, int K,
                  int lda, int ldb, int ldc,
                  char transA, char transB,
                  libxsmm_datatype a_type,
                  libxsmm_datatype b_type,
                  libxsmm_datatype c_type,
                  libxsmm_datatype comp_type) {
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
    return;
  }

  libxsmm_gemm_param p{};
  p.a.primary = const_cast<TB *>(B);
  p.b.primary = const_cast<TA *>(A);
  p.c.primary = C;

  ker(&p);
}
#endif

static inline void run_blis(char        layout,
                            char        transA,
                            char        transB,
                            int         M, int N, int K,
                            float       alpha, float beta,
                            int         lda, int ldb, int ldc,
                            char        mem_format_a,
                            char        mem_format_b,
                            const void *A, const void *B, void *C,
                            const data_types &dtypes) {
  if (dtypes.src == data_type_t::f32 && dtypes.dst == data_type_t::f32) {
    aocl_gemm_f32f32f32of32(layout,transA,transB,M,N,K,alpha,
                            static_cast<const float *>(A),lda,mem_format_a,
                            static_cast<const float *>(B),ldb,mem_format_b,
                            beta,static_cast<float *>(C),ldc,nullptr);
  }
  else if (dtypes.src == data_type_t::bf16 && dtypes.dst == data_type_t::bf16) {
    aocl_gemm_bf16bf16f32obf16(layout,transA,transB,M,N,K,alpha,
                               static_cast<const int16_t *>(A),lda,mem_format_a,
                               static_cast<const int16_t *>(B),ldb,mem_format_b,
                               beta,static_cast<int16_t *>(C),ldc,nullptr);
  }
  else if (dtypes.src == data_type_t::bf16 && dtypes.dst == data_type_t::f32) {
    aocl_gemm_bf16bf16f32of32(layout,transA,transB,M,N,K,alpha,
                              static_cast<const int16_t *>(A),lda,mem_format_a,
                              static_cast<const int16_t *>(B),ldb,mem_format_b,
                              beta,static_cast<float *>(C),ldc,nullptr);
  }
  else {
    apilog_info("Data type not supported, falling back native kernel");
    matmul_direct_native(layout,transA,transB,M,N,K,alpha,
                         A,lda,B,ldb,beta,C,ldc,dtypes);
  }
}

#if ZENDNNL_DEPENDS_LIBXSMM
static inline void run_libxsmm(char       transA,
                               char       transB,
                               int        M, int N, int K,
                               int        lda, int ldb, int ldc,
                               const void *A, const void *B, void *C,
                               const data_types &dtypes) {
  if (dtypes.src == data_type_t::f32 && dtypes.dst == data_type_t::f32) {
    libxsmm_gemm<float,float,float>(
      static_cast<const float *>(A),
      static_cast<const float *>(B),
      static_cast<float *>(C),
      M,N,K, lda,ldb,ldc, transA,transB,
      LIBXSMM_DATATYPE_F32,LIBXSMM_DATATYPE_F32,
      LIBXSMM_DATATYPE_F32,LIBXSMM_DATATYPE_F32);
  }
  else if (dtypes.src == data_type_t::bf16 && dtypes.dst == data_type_t::f32) {
    libxsmm_gemm<libxsmm_bfloat16,libxsmm_bfloat16,float>(
      reinterpret_cast<const libxsmm_bfloat16 *>(A),
      reinterpret_cast<const libxsmm_bfloat16 *>(B),
      static_cast<float *>(C),
      M,N,K, lda,ldb,ldc, transA,transB,
      LIBXSMM_DATATYPE_BF16,LIBXSMM_DATATYPE_BF16,
      LIBXSMM_DATATYPE_F32,LIBXSMM_DATATYPE_F32);
  }
  else if (dtypes.src == data_type_t::bf16 && dtypes.dst == data_type_t::bf16) {
    libxsmm_gemm<libxsmm_bfloat16,libxsmm_bfloat16,libxsmm_bfloat16>(
      reinterpret_cast<const libxsmm_bfloat16 *>(A),
      reinterpret_cast<const libxsmm_bfloat16 *>(B),
      reinterpret_cast<libxsmm_bfloat16 *>(C),
      M,N,K, lda,ldb,ldc, transA,transB,
      LIBXSMM_DATATYPE_BF16,LIBXSMM_DATATYPE_BF16,
      LIBXSMM_DATATYPE_BF16,LIBXSMM_DATATYPE_F32);
  }
}
#endif

#if ZENDNNL_DEPENDS_LIBXSMM
static inline bool can_use_libxsmm(char        transA,
                                   char        transB,
                                   int         K,
                                   float       alpha,
                                   float       beta,
                                   const data_types &dtypes) {

  const bool scalars_ok = (alpha == 1.0f && beta == 0.0f);
  if (!scalars_ok) {
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
}
#endif

void matmul_kernel_wrapper(char layout, char transA, char transB,
                           int M, int N, int K,
                           float alpha,
                           const void *A, int lda,
                           const void *B, int ldb,
                           float beta,
                           void *C, int ldc,
                           data_types &dtypes,
                           zendnnl::ops::matmul_algo_t kernel,
                           char mem_format_a, char mem_format_b) {
#if ZENDNNL_DEPENDS_LIBXSMM
  if (kernel == zendnnl::ops::matmul_algo_t::libxsmm) {
    if (can_use_libxsmm(transA,transB,K,alpha,beta,dtypes)) {
      apilog_info("Using libxsmm kernel");
      run_libxsmm(transA,transB,M,N,K,lda,ldb,ldc,A,B,C,dtypes);
      return;
    }
    else {
      kernel  = zendnnl::ops::matmul_algo_t::aocl_blis;
    }
  }
#endif

  if (kernel == zendnnl::ops::matmul_algo_t::aocl_blis) {
    apilog_info("Using BLIS/AOCL kernel");
    run_blis(layout,transA,transB,M,N,K,alpha,beta,
             lda,ldb,ldc,mem_format_a,mem_format_b,
             A,B,C,dtypes);
    return;
  }
//   TODO: To implement native AVX512 BF16 kernel
//   else if (0) {
//     if (dtypes.src == data_type_t::bf16 &&
//         dtypes.dst == data_type_t::bf16) {
//       matmul_bf16_dispatch(static_cast<const int16_t *>(A),
//                            static_cast<const int16_t *>(B),
//                            static_cast<int16_t *>(C), nullptr /*bias.data()*/,
//                            alpha, beta, M, N, K, lda, ldb, ldc, false);
//     }
//   }

  apilog_info("Using native kernel");
  matmul_direct_native(layout,transA,transB,M,N,K,alpha,
                       A,lda,B,ldb,beta,C,ldc,dtypes);
}

void matmul_batch_gemm_wrapper(char layout, char transA, char transB, int M,
                               int N,
                               int K, float alpha, const void *A, int lda, const void *B, int ldb, float beta,
                               void *C, int ldc, data_types &dtypes, int batch_count,
                               char mem_format_a, char mem_format_b, size_t src_stride, size_t weight_stride,
                               size_t dst_stride) {


#if ZENDNNL_DEPENDS_AOCLDLP
  md_t m_ = M;
  md_t n_ = N;
  md_t k_ = K;
  md_t lda_ = lda;
  md_t ldb_ = ldb;
  md_t ldc_ = ldc;
  dlp_metadata_t *metadata_array = nullptr;
  md_t group_size = batch_count;
#else
  dim_t m_ = M;
  dim_t n_ = N;
  dim_t k_ = K;
  dim_t lda_ = lda;
  dim_t ldb_ = ldb;
  dim_t ldc_ = ldc;
  aocl_post_op *metadata_array = nullptr;
  dim_t group_size = batch_count;
#endif

  // Prepare pointer arrays for matrices
  std::vector<const void *> a_ptrs(batch_count);
  std::vector<const void *> b_ptrs(batch_count);
  std::vector<void *> c_ptrs(batch_count);

  // Set up pointers for each batch
  #pragma omp parallel for
  for (int b = 0; b < batch_count; ++b) {
    a_ptrs[b] = static_cast<const uint8_t *>(A) + b * src_stride;
    b_ptrs[b] = static_cast<const uint8_t *>(B) + b * weight_stride;
    c_ptrs[b] = static_cast<uint8_t *>(C) + b * dst_stride;
  }

  // Call appropriate batch GEMM based on data types
  if (dtypes.src == data_type_t::f32 && dtypes.dst == data_type_t::f32) {
    apilog_info("executing aocl_batch_gemm_f32f32f32of32");
    aocl_batch_gemm_f32f32f32of32(
      &layout, &transA, &transB,
      &m_, &n_, &k_,
      &alpha,
      reinterpret_cast<const float **>(a_ptrs.data()), &lda_,
      reinterpret_cast<const float **>(b_ptrs.data()), &ldb_,
      &beta,
      reinterpret_cast<float **>(c_ptrs.data()), &ldc_,
      1, // single group
      &group_size,
      &mem_format_a, &mem_format_b,
      &metadata_array);
  }
  else if (dtypes.src == data_type_t::bf16 && dtypes.dst == data_type_t::f32) {
    apilog_info("executing aocl_batch_gemm_bf16bf16f32of32");
    aocl_batch_gemm_bf16bf16f32of32(
      &layout, &transA, &transB,
      &m_, &n_, &k_,
      &alpha,
      reinterpret_cast<const bfloat16 **>(a_ptrs.data()), &lda_,
      reinterpret_cast<const bfloat16 **>(b_ptrs.data()), &ldb_,
      &beta,
      reinterpret_cast<float **>(c_ptrs.data()), &ldc_,
      1, // single group
      &group_size,
      &mem_format_a, &mem_format_b,
      &metadata_array);
  }
  else if (dtypes.src == data_type_t::bf16 && dtypes.dst == data_type_t::bf16) {
    apilog_info("executing aocl_batch_gemm_bf16bf16f32obf16");
    aocl_batch_gemm_bf16bf16f32obf16(
      &layout, &transA, &transB,
      &m_, &n_, &k_,
      &alpha,
      reinterpret_cast<const bfloat16 **>(a_ptrs.data()), &lda_,
      reinterpret_cast<const bfloat16 **>(b_ptrs.data()), &ldb_,
      &beta,
      reinterpret_cast<bfloat16 **>(c_ptrs.data()), &ldc_,
      1, // single group
      &group_size,
      &mem_format_a, &mem_format_b,
      &metadata_array);
  }
  else {
    apilog_info("Unsupported data type combination for batch GEMM, falling back to native");
    // Fall back to native implementation for each batch
    for (int b = 0; b < batch_count; ++b) {
      matmul_kernel_wrapper(layout, transA, transB, M, N, K, alpha,
                            a_ptrs[b], lda, b_ptrs[b], ldb, beta, c_ptrs[b], ldc,
                            dtypes, matmul_algo_t::aocl_blis, mem_format_a, mem_format_b);
    }
  }
}

inline const void *get_matrix_block(const void *base, int row_start,
                                    int col_start,
                                    int lda, bool trans, size_t type_size) {
  if (trans) {
    // Accessing column-major layout when transposed
    return static_cast<const uint8_t *>(base) + (col_start * lda + row_start) *
           type_size;
  }
  else {
    return static_cast<const uint8_t *>(base) + (row_start * lda + col_start) *
           type_size;
  }
}

inline void *get_output_block(void *base, int row_start, int col_start,
                              int ldc, size_t type_size) {
  return static_cast<uint8_t *>(base) + (row_start * ldc + col_start) * type_size;
}

inline int get_batch_index(int b, int batch_size) {
  return (batch_size == 1) ? 0 : (b % batch_size);
}

inline bool may_i_use_blis_partition(int batch_count, int M, int N,
                                     int num_threads, data_type_t dtype) {

  // Set thresholds based on thread count and data type (powers of 2 only)
  int M_threshold = 0, N_threshold = 0, work_threshold = 0;

  /*BLIS performs better when M and N are large and thread count is moderate to high.
   It uses internal tiling and cache-aware scheduling,
   where each 8-core cluster shares a 32MB L3 cache. Manual OpenMP partitioning
   can disrupt BLIS's optimized workload distribution, leading to contention.
   Delegating to BLIS ensures better throughput and efficient hardware utilization.*/
  // TODO: Tune it more based on heuristics (threshold relies on problem size and data type)
  if (num_threads <= 16) {
    M_threshold    = 512;
    N_threshold    = 256;
    work_threshold = 128;
  }
  else if (num_threads <= 32) {
    M_threshold    = 1024;
    N_threshold    = 512;
    work_threshold = 256;
  }
  else {
    M_threshold    = 2048;
    N_threshold    = 1024;
    work_threshold = 512;
  }
  // Estimate effective workload per thread
  int work_per_thread = (batch_count * M) / num_threads;

  // Allow BLIS if batch size is small and M is reasonably large
  bool small_batch_override = (batch_count <= 8 && M >= 1024);

  return ((M >= M_threshold &&
           N >= N_threshold &&
           work_per_thread >= work_threshold)
          || small_batch_override);
}

// TODO: Further tune the heuristics based on num_threads and other params
inline matmul_algo_t select_algo_by_heuristics_bf16(int BS, int M, int N, int K,
    int num_threads) {
  if (BS <= 512) {
    if (N <= 512) {
      if (N <= 48) {
        if (M <= 196) {
          return matmul_algo_t::libxsmm;
        }
        else {
          return matmul_algo_t::aocl_blis;
        }
      }
      else {
        return matmul_algo_t::libxsmm;
      }
    }
    else {
      if (K <= 512) {
        return matmul_algo_t::libxsmm;
      }
      else {
        return matmul_algo_t::aocl_blis;
      }
    }
  }
  else {
    if (K <= 48) {
      return matmul_algo_t::libxsmm;
    }
    else {
      if (K < 50) {
        return matmul_algo_t::aocl_blis;
      }
      else {
        if (K <= 196) {
          return matmul_algo_t::libxsmm;
        }
        else {
          return matmul_algo_t::aocl_blis;
        }
      }
    }
  }
}


status_t matmul_direct(const char layout,const bool transA,const bool transB,
                       const int M, const int N, const int K,const float alpha, const void *src,
                       const int lda, const void *weight,const int ldb,const void *bias,
                       const float beta, void *dst, const int ldc,
                       lowoha_params params,
                       int Batch_A, int Batch_B) {
  log_info("Executing matmul LOWOHA kernel");

  if (!src || !weight || !dst) {
    log_error("Null pointer input to matmul_direct");
    return status_t::failure;
  }

  if (M <= 0 || N <= 0 || K <= 0 || Batch_A <= 0 || Batch_B <= 0) {
    log_error("Invalid matrix dimensions/Batch size");
    return status_t::failure;
  }

  if (params.quant_params.src_scale.buff || params.quant_params.wei_scale.buff ||
      params.quant_params.dst_scale.buff ||
      params.quant_params.src_zp.buff || params.quant_params.wei_zp.buff ||
      params.quant_params.dst_zp.buff) {
    log_error("Quantization params are not supported in LOWOHA matmul_direct yet");
    return status_t::failure;
  }

  const bool is_f32_src  = (params.dtypes.src == data_type_t::f32);
  const bool is_bf16_src = (params.dtypes.src == data_type_t::bf16);
  const bool is_f32_out  = (params.dtypes.dst == data_type_t::f32);
  const bool is_bf16_out = (params.dtypes.dst == data_type_t::bf16);

  if ((!is_f32_src && !is_bf16_src) || (!is_f32_out && !is_bf16_out)) {
    log_error("Unsupported data type combination");
    return status_t::failure;
  }

  if (bias) {
    log_error("Bias is not supported in LOWOHA matmul_direct");
    return status_t::failure;
  }

  if (params.postop_.size()) {
    log_error("Post-op is not supported in LOWOHA matmul_direct");
    return status_t::failure;
  }

  if (std::max(Batch_A, Batch_B) % std::min(Batch_A, Batch_B) != 0) {
    log_error("Broadcasting is not compatible with given Batch_A and Batch_B");
    return status_t::failure;
  }

  const char trans_input  = transA ? 't' : 'n';
  const char trans_weight = transB ? 't' : 'n';

  size_t src_type_size = is_f32_src ? sizeof(float) : sizeof(int16_t);
  size_t out_type_size = is_f32_out ? sizeof(float) : sizeof(int16_t);

  size_t src_stride = (transA ? K *lda : M * lda) * src_type_size;
  size_t weight_stride = (transB ? N *ldb : K * ldb) * src_type_size;
  size_t dst_stride = M * ldc * out_type_size;

  const int batch_count = std::max(Batch_A, Batch_B);
  const int num_threads = omp_get_max_threads();
  const bool use_blis_partitioning = may_i_use_blis_partition(batch_count, M, N,
                                     num_threads, params.dtypes.src);
  matmul_algo_t kernel = static_cast<matmul_algo_t>
                         (matmul_config_t::instance().get_algo());
  if (kernel==matmul_algo_t::dynamic_dispatch) {
    kernel = select_algo_by_heuristics_bf16(batch_count, M, N, K, num_threads);
  }
  const bool use_blis_bmm = false;

  if (use_blis_bmm) {
    // Use batch GEMM for multiple batches
    matmul_batch_gemm_wrapper(layout, trans_input, trans_weight,
                              M, N, K, alpha,
                              src, lda, weight, ldb, beta, dst, ldc,
                              params.dtypes, batch_count,
                              params.mem_format_a, params.mem_format_b,
                              src_stride, weight_stride, dst_stride);
  }
  else if ((num_threads > 1 && !use_blis_partitioning) ||
           kernel==zendnnl::ops::matmul_algo_t::libxsmm) {
    /*
    * Parallel partitioning strategy:
    * The total number of available threads is divided across batches to compute `threads_per_batch`,
    * ensuring that each batch gets a fair share of compute resources. Within each batch, the M dimension
    * (rows of the output matrix) is further partitioned into blocks of size `M_block`, calculated to
    * evenly distribute the workload among the threads assigned to that batch. The OpenMP `collapse(2)`
    * directive enables parallelization over both batch and row-block loops, while `schedule(dynamic)`
    * ensures better load balancing, especially when M is not divisible evenly or when thread workloads vary.
    */
    int threads_per_batch = std::max(1, num_threads / batch_count);
    int M_block = std::max(1, (M + threads_per_batch - 1) / threads_per_batch);

    // Optimize M_block based on batch count and M size
    // TODO: Further refine the tuning based on heuristics
    // involving batch_count, M, and num_threads.
    if ((batch_count >= 1024 && M <= 2048) ||
        (batch_count >= 512 && M <= 64)) {
      M_block = std::min(36, M);
    }
    else if (batch_count == 64 && M >= 512) {
      M_block = std::min(192, M);
    }
    else {
      M_block = std::min(M_block, M);  // Ensure M_block <= M
    }

#if ENABLE_ZENDNNL_PARALLEL_FOR

    // Calculate total number of work items (batch_count * number of M blocks)
    int total_m_blocks = (M + M_block - 1) / M_block;
    int total_work_items = batch_count * total_m_blocks;

    zendnnl_parallel_for(0, total_work_items, 1, [&](int start_idx, int end_idx) {
      for (int work_idx = start_idx; work_idx < end_idx; ++work_idx) {
        // Convert linear work index back to (batch, m_block) coordinates
        int b = work_idx / total_m_blocks;
        int m_block_idx = work_idx % total_m_blocks;
        int m_start = m_block_idx * M_block;
        int m_len = std::min(M_block, M - m_start);

        const uint8_t *src_ptr    = static_cast<const uint8_t *>(src) +
                                    get_batch_index(b, Batch_A) * src_stride;
        const uint8_t *weight_ptr = static_cast<const uint8_t *>(weight) +
                                    get_batch_index(b, Batch_B) * weight_stride;
        uint8_t *dst_ptr          = static_cast<uint8_t *>(dst) + b * dst_stride;

        const void *A = get_matrix_block(src_ptr, m_start, 0, lda, transA,
                                         src_type_size);
        void *C       = get_output_block(dst_ptr, m_start, 0, ldc, out_type_size);

        matmul_kernel_wrapper(layout, trans_input, trans_weight,
                              m_len, N, K, alpha,
                              A, lda, weight_ptr, ldb,
                              beta, C, ldc,
                              params.dtypes, kernel,
                              params.mem_format_a, params.mem_format_b);
      }
    });
#else

    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_count; ++b) {
      for (int m_start = 0; m_start < M; m_start += M_block) {
        int m_len = std::min(M_block, M - m_start);

        const uint8_t *src_ptr    = static_cast<const uint8_t *>(src) +
                                    get_batch_index(b, Batch_A) * src_stride;
        const uint8_t *weight_ptr = static_cast<const uint8_t *>(weight) +
                                    get_batch_index(b, Batch_B) * weight_stride;
        uint8_t *dst_ptr          = static_cast<uint8_t *>(dst) + b * dst_stride;

        const void *A = get_matrix_block(src_ptr, m_start, 0, lda, transA,
                                         src_type_size);
        void *C       = get_output_block(dst_ptr, m_start, 0, ldc, out_type_size);

        matmul_kernel_wrapper(layout, trans_input, trans_weight,
                              m_len, N, K, alpha,
                              A, lda, weight_ptr, ldb,
                              beta, C, ldc,
                              params.dtypes, kernel,
                              params.mem_format_a, params.mem_format_b);
      }
    }
#endif
  }
  else {
    for (int b = 0; b < batch_count; ++b) {
      const uint8_t *src_ptr    = static_cast<const uint8_t *>(src) +
                                  get_batch_index(b, Batch_A) * src_stride;
      const uint8_t *weight_ptr = static_cast<const uint8_t *>(weight) +
                                  get_batch_index(b, Batch_B) * weight_stride;
      uint8_t *dst_ptr          = static_cast<uint8_t *>(dst) + b * dst_stride;

      matmul_kernel_wrapper(layout, trans_input, trans_weight,
                            M, N, K, alpha,
                            src_ptr, lda, weight_ptr, ldb,
                            beta, dst_ptr, ldc,
                            params.dtypes, kernel,
                            params.mem_format_a, params.mem_format_b);
    }
  }
  return status_t::success;
}

} // namespace lowoha
} // namespace zendnnl

