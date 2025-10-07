/*******************************************************************************
* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
*
*******************************************************************************/
#ifndef EMBAG_AVX2_FP32_BF16_UTILS_HPP
#define EMBAG_AVX2_FP32_BF16_UTILS_HPP

#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <limits>
#include <omp.h>
#include <type_traits>

#include "embag_avx2_kernels.hpp"
#define ENABLE_PREFETCH
#ifdef ENABLE_PREFETCH
  #include <xmmintrin.h>
#endif

namespace zendnnl {
namespace ops {

// Prefetch helper for input rows (same as AVX512 version)
template <typename T>
inline void maybe_prefetch_input(const T *input, int32_t idx, int64_t width,
                                 int32_t padidx) {
#ifdef ENABLE_PREFETCH
  if (idx != padidx) {
    _mm_prefetch(reinterpret_cast<const char *>(&input[idx * width]), _MM_HINT_T0);
  }
#endif
}

// Prefetch helper for weights (same as AVX512 version)
inline void maybe_prefetch_weight(const float *weights, int32_t i,
                                  int32_t end) {
#ifdef ENABLE_PREFETCH
  const float *pf_wt_ptr = (i < end) ? &weights[i] : &weights[end - 1];
  _mm_prefetch(reinterpret_cast<const char *>(pf_wt_ptr), _MM_HINT_T0);
#endif
}

/*-----------------------------------------------------------------------------
  embag_avx2_kernel:
  Templated AVX2 embedding bag kernel for broad CPU compatibility.
  This kernel implements an embedding bag operator using AVX2 vectorization,
  supporting both FP32 and BF16 data types via templates. It is optimized for
  CPUs without AVX-512 support and uses OpenMP for parallelism across bags.

  Template Parameters:
  - InType   : Input data type (float or uint16_t for BF16)
  - OutType  : Output data type (float or uint16_t for BF16)

  Features:
  - SIMD vectorization using AVX2 (256-bit)
  - Tail-safe handling for non-multiple-of-8 embedding widths
  - Compile-time branching via C++ templates
  - OpenMP-based parallelism over bags

  Differences from AVX-512:
  - Uses 256-bit registers (__m256) instead of 512-bit (__m512)
  - SIMD width is 8 floats instead of 16
  - No AVX-512 mask registers; uses manual masking
  - BF16 support requires software-based conversion

  Register Usage:
  AVX2 provides 16 vector registers (ymm0–ymm15), each 256 bits wide.
  BF16 requires manual conversion between BF16 and FP32.

  Accumulator Array:
  - acc[full_blocks + 1], where each element is a __m256 (256-bit float vector)
  - Example: For width = 64 → full_blocks = 8 → acc uses 9 ymm registers

  Temporary Registers:
  - in_vec, wt_vec, div_vec
  - Manual BF16 conversion intermediates

  Tail Handling:
  - Uses manual masking and scalar loops for tail elements

  Register Usage Summary:

  | Component                 | FP32 Version              | BF16 Version                      |
  |---------------------------|---------------------------|-----------------------------------|
  | Accumulators (acc[])      | full_blocks + 1           | Same                              |
  | Input vector (in_vec)     | 1 × __m256                | 1 × __m256 + conversion overhead  |
  | Weight vector (wt_vec)    | 1 × __m256                | Same                              |
  | Division vector (div_vec) | 1 × __m256 (if mean)      | Same                              |
  | Tail buffer               | Manual loops              | Manual loops + conversion         |
  | Total ymm registers (est.)| ~6–10                     | ~8–12 (due to conversion overhead)|

*/

// Helper function for BF16 to FP32 conversion
__attribute__((target("avx2,avx512vl")))
inline __m256 bf16_to_fp32_avx2(const uint16_t *bf16_data) {
  // Load 8 BF16 values (16-bit each)
  __m128i bf16_vec = _mm_loadu_si128(reinterpret_cast<const __m128i *>
                                     (bf16_data));

  // Unpack to 32-bit by shifting left by 16 bits
  __m256i fp32_int = _mm256_slli_epi32(_mm256_cvtepu16_epi32(bf16_vec), 16);

  // Reinterpret as float
  return _mm256_castsi256_ps(fp32_int);
}

// Helper function for FP32 to BF16 conversion
__attribute__((target("avx2,avx512vl")))
inline void fp32_to_bf16_avx2(__m256 fp32_vec, uint16_t *bf16_data) {
  // Convert to int32, then extract upper 16 bits
  __m256i fp32_int = _mm256_castps_si256(fp32_vec);
  __m256i bf16_32 = _mm256_srli_epi32(fp32_int, 16);

  // Pack to 16-bit and store
  __m128i bf16_vec = _mm256_cvtepi32_epi16(bf16_32);
  _mm_storeu_si128(reinterpret_cast<__m128i *>(bf16_data), bf16_vec);
}

template <
  typename InType,
  typename IndexType,
  typename OffsetType,
  typename OutType>
__attribute__((target("avx2,fma")))
void embag_avx2_kernel(
  const InType *input,           // [num_embeddings, width] - embedding table
  const float *weights,          // [indsz] or nullptr if is_weights == false
  const IndexType *indices,     // [indsz] - indices into embedding table
  const OffsetType *offsets,        // [offsz] - start positions of each bag
  OutType *dst,                  // [offsz, width] with stride dst_stride - output buffer
  int64_t width,                 // embedding dimension
  int64_t indsz,                 // number of indices
  int64_t offsz,                 // number of bags
  int64_t padidx,                // padding index to skip
  bool is_weights,               // whether weights are used
  embag_algo_t algo,             // REDUCE_SUM, REDUCE_MEAN, REDUCE_MAX
  int64_t dst_stride,            // stride between output rows
  bool include_last_offset       // whether to include the last offset
) {

  constexpr int simd_width = 8;  // AVX2 processes 8 floats per vector
  const int full_blocks = width / simd_width;
  const int tail = width % simd_width;
  constexpr int prefetch_distance = 8;
  bool is_embedding = (offsets == nullptr) ? true : false;
  int outer_loop = is_embedding ? indsz : offsz;

  #pragma omp parallel for
  for (int oi = 0; oi < outer_loop; ++oi) {
    int64_t start = is_embedding ? oi : offsets[oi];
    int64_t end = is_embedding ? oi + 1 : (include_last_offset ? offsets[oi + 1] :
                                           (oi < offsz - 1 ? offsets[oi + 1] : indsz));
    int64_t dst_offset = oi * dst_stride;
    bool first_valid_index = true;
    float wt_sum = 0.0f;

    // Accumulator registers for SIMD blocks
    int acc_size = full_blocks + (tail > 0 ? 1 : 0);
    __m256 *acc = (__m256 *)_mm_malloc(sizeof(__m256) * acc_size, 32);

    for (int i = 0; i < acc_size; ++i) {
      acc[i] = _mm256_setzero_ps();
    }

    for (int i = start; i < end; ++i) {
      int32_t pf_i = i + prefetch_distance;
      int32_t pf_idx = (pf_i < end) ? indices[pf_i] : padidx;
      maybe_prefetch_input(input, pf_idx, width, padidx);
      if (is_weights) {
        maybe_prefetch_weight(weights, pf_i, end);
      }

      int32_t idx = indices[i];
      if (idx == padidx) {
        continue;
      }

      float wt = is_weights ? weights[i] : 1.0f;
      if (algo != embag_algo_t::max) {
        wt_sum += wt;
      }
      int64_t input_offset = idx * width;
      __m256 wt_vec = _mm256_set1_ps(wt);

      // Process full SIMD blocks
      for (int b = 0; b < full_blocks; ++b) {
        __m256 in_vec;
        if constexpr(std::is_same_v<InType, float>) {
          in_vec = _mm256_loadu_ps(&input[input_offset + b * simd_width]);
        }
        else {
          // BF16 to FP32 conversion for AVX2
          in_vec = bf16_to_fp32_avx2(reinterpret_cast<const uint16_t *>(
                                       &input[input_offset + b * simd_width]));
        }

        if (is_embedding) {
          acc[b] = in_vec;
        }
        else {
          if (algo == embag_algo_t::max) {
            if (first_valid_index) {
              acc[b] = in_vec;
            }
            else {
              acc[b] = _mm256_max_ps(acc[b], in_vec);
            }
          }
          else {
            acc[b] = _mm256_fmadd_ps(in_vec, wt_vec, acc[b]);
          }
        }
      }

      // Process tail elements
      if (tail > 0) {
        float tail_acc[simd_width] = {0};
        float tail_input[simd_width] = {0};

        // Load tail elements
        for (int t = 0; t < tail && t < simd_width; ++t) {
          if constexpr(std::is_same_v<InType, float>) {
            tail_input[t] = input[input_offset + full_blocks * simd_width + t];
          }
          else {
            // Manual BF16 to FP32 conversion
            uint16_t bf16_val = reinterpret_cast<const uint16_t *>(
                                  &input[input_offset + full_blocks * simd_width + t])[0];
            uint32_t fp32_val = static_cast<uint32_t>(bf16_val) << 16;
            std::memcpy(&tail_input[t], &fp32_val, sizeof(float));

          }
        }

        if (is_embedding) {
          // Load back to accumulator
          acc[full_blocks] = _mm256_loadu_ps(tail_input);
        }
        else {
          // Extract current accumulator values
          _mm256_storeu_ps(tail_acc, acc[full_blocks]);

          for (int t = 0; t < tail; ++t) {
            if (algo == embag_algo_t::max) {
              if (first_valid_index) {
                tail_acc[t] = tail_input[t];
              }
              else {
                tail_acc[t] = std::max(tail_acc[t], tail_input[t]);
              }
            }
            else {
              tail_acc[t] += wt * tail_input[t];
            }
          }

          // Load back to accumulator
          acc[full_blocks] = _mm256_loadu_ps(tail_acc);
        }
      }
      first_valid_index = false;
    }

    if (!is_embedding) {
      // Normalize for mean reduction
      if (algo == embag_algo_t::mean && wt_sum > 0.0f) {
        __m256 div_vec = _mm256_set1_ps(1.0f / wt_sum);
        for (int b = 0; b < full_blocks; ++b) {
          acc[b] = _mm256_mul_ps(acc[b], div_vec);
        }
        if (tail > 0) {
          acc[full_blocks] = _mm256_mul_ps(acc[full_blocks], div_vec);
        }
      }
    }

    // Store full block elements
    for (int b = 0; b < full_blocks; ++b) {
      if constexpr(std::is_same_v<OutType, float>) {
        _mm256_storeu_ps(&dst[dst_offset + b * simd_width], acc[b]);
      }
      else {
        // FP32 to BF16 conversion for AVX2
        fp32_to_bf16_avx2(acc[b], reinterpret_cast<uint16_t *>(
                            &dst[dst_offset + b * simd_width]));
      }
    }

    // Store tail elements
    if (tail > 0) {
      if constexpr(std::is_same_v<OutType, float>) {
        float tail_result[simd_width];
        _mm256_storeu_ps(tail_result, acc[full_blocks]);
        std::memcpy(&dst[dst_offset + full_blocks * simd_width], tail_result,
                    tail * sizeof(float));
      }
      else {
        uint16_t tail_result_bf16[simd_width];
        fp32_to_bf16_avx2(acc[full_blocks], tail_result_bf16);
        std::memcpy(&dst[dst_offset + full_blocks * simd_width], tail_result_bf16,
                    tail * sizeof(uint16_t));
      }
    }

    // Free allocated memory
    _mm_free(acc);
  }
}

} //namespace ops
} //namespace zendnnl

#endif
