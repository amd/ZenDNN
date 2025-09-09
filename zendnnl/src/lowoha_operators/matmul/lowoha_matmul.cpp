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
#include <cmath>
#include <cstring>

namespace zendnnl {
namespace lowoha {


inline float bf16_to_float(int16_t val) {
  uint32_t temp = static_cast<uint16_t>(val) << 16;
  float result;
  std::memcpy(&result, &temp, sizeof(result));
  return result;
}

inline int16_t float_to_bf16(float val) {
  uint32_t temp;
  std::memcpy(&temp, &val, sizeof(temp));
  return static_cast<int16_t>(temp >> 16);
}

void matmul_direct_native(char layout, char transA, char transB, int M, int N,
                          int K, float alpha, const void *A, int lda,
                          const void *B, int ldb, float beta, void *C, int ldc, data_type_t src_data_type,
                          data_type_t out_data_type) {

  const bool is_f32_src  = (src_data_type == data_type_t::f32);
  const bool is_bf16_src = (src_data_type == data_type_t::bf16);
  const bool is_f32_out  = (out_data_type == data_type_t::f32);
  const bool is_bf16_out = (out_data_type == data_type_t::bf16);

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
          a_val = bf16_to_float((transA == 'n') ? A_bf16[m * lda + k] : A_bf16[k * lda +
                                m]);
          b_val = bf16_to_float((transB == 'n') ? B_bf16[k * ldb + n] : B_bf16[n * ldb +
                                k]);
        }

        acc += a_val * b_val;
      }

      if (is_f32_out) {
        float c_val        = C_f32[m * ldc + n];
        C_f32[m * ldc + n] = alpha * acc + beta * c_val;
      }
      else if (is_bf16_out) {
        float c_val         = bf16_to_float(C_bf16[m * ldc + n]);
        float result        = alpha * acc + beta * c_val;
        C_bf16[m * ldc + n] = float_to_bf16(result);
      }
    }
  }
}

void matmul_kernel_wrapper(char layout, char transA, char transB, int M, int N,
                           int K, float alpha,
                           const void *A, int lda, const void *B, int ldb, float beta, void *C, int ldc,
                           data_type_t src_data_type, data_type_t out_data_type, bool use_blis) {

  if (use_blis) {
    if (src_data_type == data_type_t::f32 && out_data_type == data_type_t::f32) {
      aocl_gemm_f32f32f32of32(layout, transA, transB, M, N, K, alpha,
                              static_cast<const float *>(A), lda, 'n',
                              static_cast<const float *>(B), ldb, 'n',
                              beta, static_cast<float *>(C), ldc, nullptr);
    }
    else if (src_data_type == data_type_t::bf16 &&
             out_data_type == data_type_t::bf16) {
      aocl_gemm_bf16bf16f32obf16(layout, transA, transB, M, N, K, alpha,
                                 static_cast<const int16_t *>(A), lda, 'n',
                                 static_cast<const int16_t *>(B), ldb, 'n',
                                 beta, static_cast<int16_t *>(C), ldc, nullptr);
    }
    else if (src_data_type == data_type_t::bf16 &&
             out_data_type == data_type_t::f32) {
      aocl_gemm_bf16bf16f32of32(layout, transA, transB, M, N, K, alpha,
                                static_cast<const int16_t *>(A), lda, 'n',
                                static_cast<const int16_t *>(B), ldb, 'n',
                                beta, static_cast<float *>(C), ldc, nullptr);
    }
    else {
      log_info("Unsupported data type combination for BLIS, Redirecting it to native");
      matmul_direct_native(layout, transA, transB, M, N, K, alpha,
                           A, lda, B, ldb, beta, C, ldc, src_data_type, out_data_type);
    }
  }
  else {
    // Native kernel call
    matmul_direct_native(layout, transA, transB, M, N, K, alpha,
                         A, lda, B, ldb, beta, C, ldc, src_data_type, out_data_type);
  }
}

const void *get_matrix_block(const void *base, int row_start, int col_start,
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

void *get_output_block(void *base, int row_start, int col_start,
                       int ldc, size_t type_size) {
  return static_cast<uint8_t *>(base) + (row_start * ldc + col_start) * type_size;
}

int get_batch_index(int b, int batch_size) {
  return (batch_size == 1) ? 0 : (b % batch_size);
}

void matmul_direct(const void *src, const void *weight, void *dst, void *bias,
                   float alpha, float beta, int M, int N, int K,
                   bool transA, bool transB, int lda, int ldb, int ldc,
                   data_type_t src_data_type, data_type_t out_data_type,
                   post_op_type_t post_op, void *post_op_buff,
                   int Batch_A, int Batch_B) {
  log_info("Executing matmul LOWOHA kernel");

  if (!src || !weight || !dst) {
    log_error("Error: Null pointer input to matmul_direct");
    return;
  }

  const bool is_f32_src  = (src_data_type == data_type_t::f32);
  const bool is_bf16_src = (src_data_type == data_type_t::bf16);
  const bool is_f32_out  = (out_data_type == data_type_t::f32);
  const bool is_bf16_out = (out_data_type == data_type_t::bf16);

  if ((!is_f32_src && !is_bf16_src) || (!is_f32_out && !is_bf16_out)) {
    log_error("Error: Unsupported data type combination");
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

  if (num_threads > 1) {
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

    #pragma omp parallel for collapse(2) schedule(dynamic)
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
                              src_data_type, out_data_type, use_blis);
      }
    }
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
                            src_data_type, out_data_type, use_blis);
    }
  }
}


} // namespace lowoha
} // namespace zendnnl

