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

namespace zendnnl {
namespace lowoha {

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

void matmul_kernel_wrapper(char layout, char transA, char transB, int M, int N,
                           int K, float alpha, const void *A, int lda, const void *B, int ldb, float beta,
                           void *C, int ldc, data_types &dtypes, bool use_blis) {

  if (use_blis) {
    if (dtypes.src == data_type_t::f32 &&
        dtypes.dst == data_type_t::f32) {
      aocl_gemm_f32f32f32of32(layout, transA, transB, M, N, K, alpha,
                              static_cast<const float *>(A), lda, 'n',
                              static_cast<const float *>(B), ldb, 'n',
                              beta, static_cast<float *>(C), ldc, nullptr);
    }
    else if (dtypes.src == data_type_t::bf16 &&
             dtypes.dst == data_type_t::bf16) {
      aocl_gemm_bf16bf16f32obf16(layout, transA, transB, M, N, K, alpha,
                                 static_cast<const int16_t *>(A), lda, 'n',
                                 static_cast<const int16_t *>(B), ldb, 'n',
                                 beta, static_cast<int16_t *>(C), ldc, nullptr);
    }
    else if (dtypes.src == data_type_t::bf16 &&
             dtypes.dst == data_type_t::f32) {
      aocl_gemm_bf16bf16f32of32(layout, transA, transB, M, N, K, alpha,
                                static_cast<const int16_t *>(A), lda, 'n',
                                static_cast<const int16_t *>(B), ldb, 'n',
                                beta, static_cast<float *>(C), ldc, nullptr);
    }
    else {
      log_info("Unsupported data type combination for BLIS, Redirecting it to native");
      matmul_direct_native(layout, transA, transB, M, N, K, alpha,
                           A, lda, B, ldb, beta, C, ldc, dtypes);
    }
  }
  //TODO: To implement native AVX512 BF16 kernel
  else if (0) {
    if (dtypes.src == data_type_t::bf16 &&
        dtypes.dst == data_type_t::bf16) {
      matmul_bf16_dispatch(static_cast<const int16_t *>(A),
                           static_cast<const int16_t *>(B),
                           static_cast<int16_t *>(C), nullptr /*bias.data()*/,
                           alpha, beta, M, N, K, lda, ldb, ldc, false);
    }
  }
  else {
    // Native kernel call
    matmul_direct_native(layout, transA, transB, M, N, K, alpha,
                         A, lda, B, ldb, beta, C, ldc, dtypes);
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


void matmul_direct(const void *src, const void *weight, void *dst, void *bias,
                   float alpha, float beta, int M, int N, int K,
                   bool transA, bool transB, int lda, int ldb, int ldc,
                   data_types &dtypes, lowoha_post_op post_op,
                   const lowoha_quantization_params_t &quant_params,
                   int Batch_A, int Batch_B) {
  log_info("Executing matmul LOWOHA kernel");

  if (!src || !weight || !dst) {
    log_error("Null pointer input to matmul_direct");
    return;
  }

  if (M <= 0 || N <= 0 || K <= 0 || Batch_A <= 0 || Batch_B <= 0) {
    log_error("Invalid matrix dimensions/Batch size");
    return;
  }

  if (quant_params.src_scale.buff || quant_params.wei_scale.buff ||
      quant_params.dst_scale.buff ||
      quant_params.src_zp.buff || quant_params.wei_zp.buff ||
      quant_params.dst_zp.buff) {
    log_error("Quantization parameters are not supported in LOWOHA matmul_direct yet");
    return;
  }

  const bool is_f32_src  = (dtypes.src == data_type_t::f32);
  const bool is_bf16_src = (dtypes.src == data_type_t::bf16);
  const bool is_f32_out  = (dtypes.dst == data_type_t::f32);
  const bool is_bf16_out = (dtypes.dst == data_type_t::bf16);

  if ((!is_f32_src && !is_bf16_src) || (!is_f32_out && !is_bf16_out)) {
    log_error("Unsupported data type combination");
    return;
  }

  if (bias) {
    log_error("Bias is not supported in LOWOHA matmul_direct");
    return;
  }

  if (post_op.postop_.size()) {
    log_error("Post-op is not supported in LOWOHA matmul_direct");
    return;
  }

  if (std::max(Batch_A, Batch_B) % std::min(Batch_A, Batch_B) != 0) {
    log_error("Broadcasting is not compatible with given Batch_A and Batch_B");
    return;
  }

  const char trans_input  = transA ? 't' : 'n';
  const char trans_weight = transB ? 't' : 'n';
  const char layout       = 'r';

  size_t src_type_size = is_f32_src ? sizeof(float) : sizeof(int16_t);
  size_t out_type_size = is_f32_out ? sizeof(float) : sizeof(int16_t);

  size_t src_stride = (transA ? K *lda : M * lda) * src_type_size;
  size_t weight_stride = (transB ? N *ldb : K * ldb) * src_type_size;
  size_t dst_stride = M * ldc * out_type_size;

  const int batch_count = std::max(Batch_A, Batch_B);
  const int num_threads = omp_get_max_threads();
  const bool use_blis   = true;

  if (num_threads > 1 &&
      !may_i_use_blis_partition(batch_count, M, N, num_threads, dtypes.src)) {
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
                              dtypes, use_blis);
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
                              dtypes, use_blis);
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
                            dtypes, use_blis);
    }
  }
}

} // namespace lowoha
} // namespace zendnnl

