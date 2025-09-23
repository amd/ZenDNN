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

#ifndef _MATMUL_KERNEL_BF16_AVX512_HPP
#define _MATMUL_KERNEL_BF16_AVX512_HPP


#include <immintrin.h>
#include <algorithm>
#include <cstdint>

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512bf16")))
void matmul_bf16_6x16(const uint16_t *A, const uint16_t *B, float *dst_f32,
                      uint16_t *dst_bf16,
                      const float *bias, float alpha, float beta, int K,
                      int lda, int ldb, int ldc, bool output_fp32,
                      int row_offset, int col_offset) {
  const int full_cols = 16;
  const int col_blocks = 1;

  __m512 alpha_vec = _mm512_set1_ps(alpha);
  __m512 beta_vec  = _mm512_set1_ps(beta);
  __m512 bias_vec  = _mm512_loadu_ps(&bias[col_offset]);

  for (int cb = 0; cb < col_blocks; ++cb) {
    int col_base = col_offset + cb * full_cols;

    // Initialize accumulators
    __m512 acc[6];
    for (int i = 0; i < 6; ++i) {
      acc[i] = _mm512_setzero_ps();
    }

    int k = 0;
    for (; k + 5 < K; k += 6) {
      // Load 6 B vectors
      __m512bh b_vec[6];
      for (int j = 0; j < 6; ++j) {
        const uint16_t *b_ptr = &B[(k + j) * ldb + col_base];
        __m256i b_vals = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b_ptr));
        b_vec[j] = (__m512bh)_mm512_cvtepu16_epi32(b_vals);
      }

      // Prefetch next B and A
      //_mm_prefetch((const char*)&B[(k + 6) * ldb + col_base], _MM_HINT_T0);
      //_mm_prefetch((const char*)&A[(row_offset + 0) * lda + k + 6], _MM_HINT_T0);

      // Accumulate for each row
      for (int i = 0; i < 6; ++i) {
        const uint16_t *a_row = &A[(row_offset + i) * lda + k];
        for (int j = 0; j < 6; ++j) {
          __m512bh a_kj = (__m512bh)_mm512_set1_epi16(a_row[j]);
          acc[i] = _mm512_dpbf16_ps(acc[i], a_kj, b_vec[j]);
        }
      }
    }

    // Handle tail K
    for (; k < K; ++k) {
      const uint16_t *b_ptr = &B[k * ldb + col_base];
      __m256i b_vals = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b_ptr));
      __m512bh b_vec = (__m512bh)_mm512_cvtepu16_epi32(b_vals);

      for (int i = 0; i < 6; ++i) {
        __m512bh a_k = (__m512bh)_mm512_set1_epi16(A[(row_offset + i) * lda + k]);
        acc[i] = _mm512_dpbf16_ps(acc[i], a_k, b_vec);
      }
    }

    // Apply alpha and beta scaling
    for (int i = 0; i < 6; ++i) {
      acc[i] = _mm512_fmadd_ps(alpha_vec, acc[i], _mm512_mul_ps(beta_vec, bias_vec));
    }

    // Store result
    for (int i = 0; i < 6; ++i) {
      if (output_fp32) {
        _mm512_storeu_ps(&dst_f32[(row_offset + i) * ldc + col_base], acc[i]);
      }
      else {
        __m256i bf16_out = (__m256i)_mm512_cvtneps_pbh(acc[i]);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(&dst_bf16[(row_offset + i) * ldc
                            + col_base]), bf16_out);
      }
    }
  }
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512bf16")))
void matmul_bf16_6x32(const uint16_t *A, const uint16_t *B, float *dst_f32,
                      uint16_t *dst_bf16,
                      const float *bias, float alpha, float beta, int K,
                      int lda, int ldb, int ldc, bool output_fp32,
                      int row_offset, int col_offset) {
  const int full_cols = 32;
  const int col_blocks = 1;

  __m512 alpha_vec = _mm512_set1_ps(alpha);
  __m512 beta_vec  = _mm512_set1_ps(beta);
  __m512 bias_vec0 = _mm512_loadu_ps(&bias[col_offset + 0]);
  __m512 bias_vec1 = _mm512_loadu_ps(&bias[col_offset + 16]);

  for (int cb = 0; cb < col_blocks; ++cb) {
    int col_base = col_offset + cb * full_cols;

    // Initialize accumulators
    __m512 acc[6][2];  // 6 rows × 2 accumulators (32 cols = 2 × 16)
    for (int i = 0; i < 6; ++i) {
      acc[i][0] = _mm512_setzero_ps();
      acc[i][1] = _mm512_setzero_ps();
    }

    int k = 0;
    for (; k + 5 < K; k += 6) {
      __m512bh b_vec[6][2];  // 6 K slices × 2 halves of 32 cols

      for (int kk = 0; kk < 6; ++kk) {
        const uint16_t *b_ptr0 = &B[(k + kk) * ldb + col_base + 0];
        const uint16_t *b_ptr1 = &B[(k + kk) * ldb + col_base + 16];

        __m256i b_vals0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b_ptr0));
        __m256i b_vals1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b_ptr1));

        b_vec[kk][0] = (__m512bh)_mm512_cvtepu16_epi32(b_vals0);
        b_vec[kk][1] = (__m512bh)_mm512_cvtepu16_epi32(b_vals1);
      }

      //_mm_prefetch((const char*)&B[(k + 6) * ldb + col_base], _MM_HINT_T0);
      //_mm_prefetch((const char*)&A[(row_offset + 0) * lda + k + 6], _MM_HINT_T0);

      for (int i = 0; i < 6; ++i) {
        const uint16_t *a_row = &A[(row_offset + i) * lda + k];
        for (int kk = 0; kk < 6; ++kk) {
          __m512bh a_k = (__m512bh)_mm512_set1_epi16(a_row[kk]);
          acc[i][0] = _mm512_dpbf16_ps(acc[i][0], a_k, b_vec[kk][0]);
          acc[i][1] = _mm512_dpbf16_ps(acc[i][1], a_k, b_vec[kk][1]);
        }
      }
    }

    // Handle tail K
    for (; k < K; ++k) {
      const uint16_t *b_ptr0 = &B[k * ldb + col_base + 0];
      const uint16_t *b_ptr1 = &B[k * ldb + col_base + 16];

      __m256i b_vals0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b_ptr0));
      __m256i b_vals1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b_ptr1));

      __m512bh b_vec0 = (__m512bh)_mm512_cvtepu16_epi32(b_vals0);
      __m512bh b_vec1 = (__m512bh)_mm512_cvtepu16_epi32(b_vals1);

      for (int i = 0; i < 6; ++i) {
        __m512bh a_k = (__m512bh)_mm512_set1_epi16(A[(row_offset + i) * lda + k]);
        acc[i][0] = _mm512_dpbf16_ps(acc[i][0], a_k, b_vec0);
        acc[i][1] = _mm512_dpbf16_ps(acc[i][1], a_k, b_vec1);
      }
    }

    // Apply alpha and beta scaling
    for (int i = 0; i < 6; ++i) {
      acc[i][0] = _mm512_fmadd_ps(alpha_vec, acc[i][0], _mm512_mul_ps(beta_vec,
                                  bias_vec0));
      acc[i][1] = _mm512_fmadd_ps(alpha_vec, acc[i][1], _mm512_mul_ps(beta_vec,
                                  bias_vec1));
    }

    // Store result
    for (int i = 0; i < 6; ++i) {
      if (output_fp32) {
        _mm512_storeu_ps(&dst_f32[(row_offset + i) * ldc + col_base + 0], acc[i][0]);
        _mm512_storeu_ps(&dst_f32[(row_offset + i) * ldc + col_base + 16], acc[i][1]);
      }
      else {
        __m256i bf16_out0 = (__m256i)_mm512_cvtneps_pbh(acc[i][0]);
        __m256i bf16_out1 = (__m256i)_mm512_cvtneps_pbh(acc[i][1]);

        _mm256_storeu_si256(reinterpret_cast<__m256i *>(&dst_bf16[(row_offset + i) * ldc
                            + col_base + 0]), bf16_out0);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(&dst_bf16[(row_offset + i) * ldc
                            + col_base + 16]), bf16_out1);
      }
    }
  }
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512bf16")))
void matmul_bf16_6x64(const uint16_t *A, const uint16_t *B, float *dst_f32,
                      uint16_t *dst_bf16,
                      const float *bias, float alpha, float beta, int K,
                      int lda, int ldb, int ldc, bool output_fp32,
                      int row_offset, int col_offset) {
  const int full_cols = 64;
  const int col_blocks = 1;

  __m512 alpha_vec = _mm512_set1_ps(alpha);
  __m512 beta_vec  = _mm512_set1_ps(beta);
  __m512 bias_vec0 = _mm512_loadu_ps(&bias[col_offset + 0]);
  __m512 bias_vec1 = _mm512_loadu_ps(&bias[col_offset + 16]);
  __m512 bias_vec2 = _mm512_loadu_ps(&bias[col_offset + 32]);
  __m512 bias_vec3 = _mm512_loadu_ps(&bias[col_offset + 48]);

  for (int cb = 0; cb < col_blocks; ++cb) {
    int col_base = col_offset + cb * full_cols;

    // Initialize accumulators: 6 rows × 4 accumulators (64 cols = 4 × 16)
    __m512 acc[6][4];
    for (int i = 0; i < 6; ++i) {
      acc[i][0] = _mm512_setzero_ps();
      acc[i][1] = _mm512_setzero_ps();
      acc[i][2] = _mm512_setzero_ps();
      acc[i][3] = _mm512_setzero_ps();
    }

    int k = 0;
    for (; k + 3 < K; k += 4) {
      __m512bh b_vec[4][4];  // 4 K slices × 4 halves of 64 cols

      for (int kk = 0; kk < 4; ++kk) {
        const uint16_t *b_ptr0 = &B[(k + kk) * ldb + col_base + 0];
        const uint16_t *b_ptr1 = &B[(k + kk) * ldb + col_base + 16];
        const uint16_t *b_ptr2 = &B[(k + kk) * ldb + col_base + 32];
        const uint16_t *b_ptr3 = &B[(k + kk) * ldb + col_base + 48];

        __m256i b_vals0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b_ptr0));
        __m256i b_vals1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b_ptr1));
        __m256i b_vals2 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b_ptr2));
        __m256i b_vals3 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b_ptr3));

        b_vec[kk][0] = (__m512bh)_mm512_cvtepu16_epi32(b_vals0);
        b_vec[kk][1] = (__m512bh)_mm512_cvtepu16_epi32(b_vals1);
        b_vec[kk][2] = (__m512bh)_mm512_cvtepu16_epi32(b_vals2);
        b_vec[kk][3] = (__m512bh)_mm512_cvtepu16_epi32(b_vals3);
      }

      for (int i = 0; i < 6; ++i) {
        const uint16_t *a_row = &A[(row_offset + i) * lda + k];
        for (int kk = 0; kk < 4; ++kk) {
          __m512bh a_k = (__m512bh)_mm512_set1_epi16(a_row[kk]);
          acc[i][0] = _mm512_dpbf16_ps(acc[i][0], a_k, b_vec[kk][0]);
          acc[i][1] = _mm512_dpbf16_ps(acc[i][1], a_k, b_vec[kk][1]);
          acc[i][2] = _mm512_dpbf16_ps(acc[i][2], a_k, b_vec[kk][2]);
          acc[i][3] = _mm512_dpbf16_ps(acc[i][3], a_k, b_vec[kk][3]);
        }
      }
    }

    // Handle tail K
    for (; k < K; ++k) {
      const uint16_t *b_ptr0 = &B[k * ldb + col_base + 0];
      const uint16_t *b_ptr1 = &B[k * ldb + col_base + 16];
      const uint16_t *b_ptr2 = &B[k * ldb + col_base + 32];
      const uint16_t *b_ptr3 = &B[k * ldb + col_base + 48];

      __m256i b_vals0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b_ptr0));
      __m256i b_vals1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b_ptr1));
      __m256i b_vals2 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b_ptr2));
      __m256i b_vals3 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b_ptr3));

      __m512bh b_vec0 = (__m512bh)_mm512_cvtepu16_epi32(b_vals0);
      __m512bh b_vec1 = (__m512bh)_mm512_cvtepu16_epi32(b_vals1);
      __m512bh b_vec2 = (__m512bh)_mm512_cvtepu16_epi32(b_vals2);
      __m512bh b_vec3 = (__m512bh)_mm512_cvtepu16_epi32(b_vals3);

      for (int i = 0; i < 6; ++i) {
        __m512bh a_k = (__m512bh)_mm512_set1_epi16(A[(row_offset + i) * lda + k]);
        acc[i][0] = _mm512_dpbf16_ps(acc[i][0], a_k, b_vec0);
        acc[i][1] = _mm512_dpbf16_ps(acc[i][1], a_k, b_vec1);
        acc[i][2] = _mm512_dpbf16_ps(acc[i][2], a_k, b_vec2);
        acc[i][3] = _mm512_dpbf16_ps(acc[i][3], a_k, b_vec3);
      }
    }

    // Apply alpha and beta scaling
    for (int i = 0; i < 6; ++i) {
      acc[i][0] = _mm512_fmadd_ps(alpha_vec, acc[i][0], _mm512_mul_ps(beta_vec,
                                  bias_vec0));
      acc[i][1] = _mm512_fmadd_ps(alpha_vec, acc[i][1], _mm512_mul_ps(beta_vec,
                                  bias_vec1));
      acc[i][2] = _mm512_fmadd_ps(alpha_vec, acc[i][2], _mm512_mul_ps(beta_vec,
                                  bias_vec2));
      acc[i][3] = _mm512_fmadd_ps(alpha_vec, acc[i][3], _mm512_mul_ps(beta_vec,
                                  bias_vec3));
    }

    // Store result
    for (int i = 0; i < 6; ++i) {
      if (output_fp32) {
        _mm512_storeu_ps(&dst_f32[(row_offset + i) * ldc + col_base + 0], acc[i][0]);
        _mm512_storeu_ps(&dst_f32[(row_offset + i) * ldc + col_base + 16], acc[i][1]);
        _mm512_storeu_ps(&dst_f32[(row_offset + i) * ldc + col_base + 32], acc[i][2]);
        _mm512_storeu_ps(&dst_f32[(row_offset + i) * ldc + col_base + 48], acc[i][3]);
      }
      else {
        __m256i bf16_out0 = (__m256i)_mm512_cvtneps_pbh(acc[i][0]);
        __m256i bf16_out1 = (__m256i)_mm512_cvtneps_pbh(acc[i][1]);
        __m256i bf16_out2 = (__m256i)_mm512_cvtneps_pbh(acc[i][2]);
        __m256i bf16_out3 = (__m256i)_mm512_cvtneps_pbh(acc[i][3]);

        _mm256_storeu_si256(reinterpret_cast<__m256i *>(&dst_bf16[(row_offset + i) * ldc
                            + col_base + 0]), bf16_out0);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(&dst_bf16[(row_offset + i) * ldc
                            + col_base + 16]), bf16_out1);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(&dst_bf16[(row_offset + i) * ldc
                            + col_base + 32]), bf16_out2);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(&dst_bf16[(row_offset + i) * ldc
                            + col_base + 48]), bf16_out3);
      }
    }
  }
}


__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512bf16")))
void matmul_bf16_Tail(const uint16_t *A, const uint16_t *B, float *dst_f32,
                      uint16_t *dst_bf16,
                      const float *bias, float alpha, float beta, int K,
                      int lda, int ldb, int ldc, bool output_fp32,
                      int row_offset, int col_offset, int row_block, int col_block) {
  __m512 alpha_vec = _mm512_set1_ps(alpha);
  __m512 beta_vec  = _mm512_set1_ps(beta);

  for (int i = 0; i < row_block; i += 4) {
    int row[4] = { row_offset + i, row_offset + i + 1, row_offset + i + 2, row_offset + i + 3 };
    bool valid[4] = { true, i + 1 < row_block, i + 2 < row_block, i + 3 < row_block };

    for (int j = 0; j < col_block; j += 16) {
      int col = col_offset + j;
      int remaining_cols = std::min(16, col_block - j);
      __mmask16 mask = (remaining_cols == 16) ? 0xFFFF : ((1 << remaining_cols) - 1);

      __m512 bias_vec = bias ? _mm512_maskz_loadu_ps(mask,
                        &bias[col]) : _mm512_setzero_ps();

      // 2 accumulators per row
      __m512 acc[4][2] = { { _mm512_setzero_ps(), _mm512_setzero_ps() },
        { _mm512_setzero_ps(), _mm512_setzero_ps() },
        { _mm512_setzero_ps(), _mm512_setzero_ps() },
        { _mm512_setzero_ps(), _mm512_setzero_ps() }
      };

      int k = 0;
      for (; k + 3 < K; k += 4) {
        __m512bh b_vec0 = (__m512bh)_mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(mask,
                                                &B[(k + 0) * ldb + col]));
        __m512bh b_vec1 = (__m512bh)_mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(mask,
                                                &B[(k + 1) * ldb + col]));
        __m512bh b_vec2 = (__m512bh)_mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(mask,
                                                &B[(k + 2) * ldb + col]));
        __m512bh b_vec3 = (__m512bh)_mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(mask,
                                                &B[(k + 3) * ldb + col]));

        for (int r = 0; r < 4; ++r) {
          if (!valid[r]) {
            continue;
          }
          __m512bh a0 = (__m512bh)_mm512_set1_epi16(A[row[r] * lda + k + 0]);
          __m512bh a1 = (__m512bh)_mm512_set1_epi16(A[row[r] * lda + k + 1]);
          __m512bh a2 = (__m512bh)_mm512_set1_epi16(A[row[r] * lda + k + 2]);
          __m512bh a3 = (__m512bh)_mm512_set1_epi16(A[row[r] * lda + k + 3]);

          acc[r][0] = _mm512_dpbf16_ps(acc[r][0], a0, b_vec0);
          acc[r][0] = _mm512_dpbf16_ps(acc[r][0], a1, b_vec1);
          acc[r][1] = _mm512_dpbf16_ps(acc[r][1], a2, b_vec2);
          acc[r][1] = _mm512_dpbf16_ps(acc[r][1], a3, b_vec3);
        }
      }

      // Tail K
      for (; k < K; ++k) {
        __m512bh b_vec = (__m512bh)_mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(mask,
                                               &B[k * ldb + col]));
        for (int r = 0; r < 4; ++r) {
          if (!valid[r]) {
            continue;
          }
          __m512bh a = (__m512bh)_mm512_set1_epi16(A[row[r] * lda + k]);
          acc[r][0] = _mm512_dpbf16_ps(acc[r][0], a, b_vec);
        }
      }

      // Finalize and store
      for (int r = 0; r < 4; ++r) {
        if (!valid[r]) {
          continue;
        }
        __m512 acc_final = _mm512_add_ps(acc[r][0], acc[r][1]);
        acc_final = _mm512_mul_ps(acc_final, alpha_vec);
        acc_final = _mm512_fmadd_ps(beta_vec, bias_vec, acc_final);

        if (output_fp32) {
          _mm512_mask_storeu_ps(&dst_f32[row[r] * ldc + col], mask, acc_final);
        }
        else {
          __m256i bf16_out = (__m256i)_mm512_cvtneps_pbh(acc_final);
          _mm256_mask_storeu_epi16(&dst_bf16[row[r] * ldc + col], mask, bf16_out);
        }
      }
    }
  }
}


void matmul_bf16_dispatch(const void *src, const void *weight, void *dst,
                          const void *bias,
                          float alpha, float beta, int M, int N, int K,
                          int lda, int ldb, int ldc, bool output_fp32) {
  const uint16_t *A = static_cast<const uint16_t *>(src);
  const uint16_t *B = static_cast<const uint16_t *>(weight);
  const float *bias_f32 = static_cast<const float *>(bias);
  float *dst_f32 = static_cast<float *>(dst);
  uint16_t *dst_bf16 = static_cast<uint16_t *>(dst);

  for (int i = 0; i < M; i += 6) {
    int row_block = std::min(6, M - i);

    for (int j = 0; j < N;) {
      int col_block = 0;
      // Try 6x64 block
      if (row_block == 6 && (N - j) >= 64) {
        col_block = 64;
        matmul_bf16_6x64(A, B, dst_f32, dst_bf16, bias_f32,
                         alpha, beta, K, lda, ldb, ldc,
                         output_fp32, i, j);
      }
      // Try 6x32 block
      else if (row_block == 6 && (N - j) >= 32) {
        col_block = 32;
        matmul_bf16_6x32(A, B, dst_f32, dst_bf16, bias_f32,
                         alpha, beta, K, lda, ldb, ldc,
                         output_fp32, i, j);
      }
      // Try 6x16 block
      else if (row_block == 6 && (N - j) >= 16) {
        col_block = 16;
        matmul_bf16_6x16(A, B, dst_f32, dst_bf16, bias_f32,
                         alpha, beta, K, lda, ldb, ldc,
                         output_fp32, i, j);
      }
      // Fallback to tail kernel
      else {
        col_block = std::min(16, N - j);  // Tail granularity
        matmul_bf16_Tail(A, B, dst_f32, dst_bf16, bias_f32,
                         alpha, beta, K, lda, ldb, ldc,
                         output_fp32, i, j, row_block, col_block);
      }

      j += col_block;
    }
  }
}

#endif