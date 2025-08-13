
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

#ifndef EMBAG_AVX512_FP32_BF16_UTILS_HPP
#define EMBAG_AVX512_FP32_BF16_UTILS_HPP

#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <limits>
#include <omp.h>
#include <type_traits>

#include "embag_avx512_kernels.hpp"

#define ENABLE_PREFETCH

#ifdef ENABLE_PREFETCH
  #include <xmmintrin.h>
#endif

namespace zendnnl {
namespace ops {

// Prefetch helper for input rows
template <typename T>
inline void maybe_prefetch_input(const T *input, int32_t idx, int64_t width,
                                 int32_t padidx) {
#ifdef ENABLE_PREFETCH
  if (idx != padidx) {
    _mm_prefetch(reinterpret_cast<const char *>(&input[idx * width]), _MM_HINT_T0);
  }
#endif
}

// Prefetch helper for weights
inline void maybe_prefetch_weight(const float *weights, int32_t i,
                                  int32_t end) {
#ifdef ENABLE_PREFETCH
  const float *pf_wt_ptr = (i < end) ? &weights[i] : &weights[end - 1];
  _mm_prefetch(reinterpret_cast<const char *>(pf_wt_ptr), _MM_HINT_T0);
#endif
}

/*-----------------------------------------------------------------------------
  embag_avx512_kernel:
  Templated AVX-512 embedding bag kernel optimized for AMD Zen 5 architecture.
  This kernel implements a high-performance embedding bag operator using AVX-512 vectorization,
  tailored for the Zen 5 architecture. It supports both FP32 and BF16 data types via templating.
  The kernel is designed to handle tail cases safely and leverages OpenMP for parallelism across bags.

  Template Parameters:
  - InType   : Input data type (float or uint16_t for BF16)
  - OutType  : Output data type (float or uint16_t for BF16)

  Features:
  - SIMD vectorization using AVX-512
  - Tail-safe handling for non-multiple-of-16 embedding widths
  - Compile-time branching via C++ templates
  - OpenMP-based parallelism over bags

  Register Usage:
  AVX-512 provides 32 vector registers (zmm0–zmm31), each 512 bits wide. BF16 conversions use:
  - _mm512_cvtpbh_ps for BF16 → FP32
  - _mm512_cvtneps_pbh for FP32 → BF16
  These conversions may introduce additional temporary registers.

  Accumulator Array:
  - acc[full_blocks + 1], where each element is a __m512 (512-bit float vector)
  - Example: For width = 64 → full_blocks = 4 → acc uses 5 zmm registers

  Temporary Registers:
  - in_vec, wt_vec, div_vec
  - Mask registers and conversion intermediates
  - BF16 path uses additional __m256i and __m256bh registers

  Tail Handling:
  - Uses masked loads/stores
  - BF16 path uses temporary arrays for tail processing

  Register Usage Summary:

  | Component                 | FP32 Version              | BF16 Version                             |
  |---------------------------|---------------------------|------------------------------------------|
  | Accumulators (acc[])      | full_blocks + 1           | Same                                     |
  | Input vector (in_vec)     | 1 × __m512                | 1 × __m512 + 1 × __m256i + 1 × __m256bh  |
  | Weight vector (wt_vec)    | 1 × __m512                | Same                                     |
  | Division vector (div_vec) | 1 × __m512 (if mean)      | Same                                     |
  | Tail buffer (BF16 only)   | —                         | 1 × uint16_t[16] + 1 × __m256i           |
  | Mask registers            | 1–2 × __mmask16           | Same                                     |
  | Total zmm registers (est.)| ~8–12                     | ~10–14 (due to conversion overhead)      |

*/

template <typename InType, typename OutType>
__attribute__((target("avx512f,avx512vl,avx512bf16")))
void embag_avx512_kernel(
  const InType *input,           // [num_embeddings, width] - embedding table
  const float *weights,          // [indsz] or nullptr if is_weights == false
  const int32_t *indices,        // [indsz] - indices into embedding table
  const int32_t *offsets,        // [offsz] - start positions of each bag
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

  constexpr int simd_width = 16;
  const int full_blocks = width / simd_width;
  const int tail = width % simd_width;
  constexpr int prefetch_distance = 8;

  #pragma omp parallel for
  for (int oi = 0; oi < offsz; ++oi) {
    int32_t start = offsets[oi];
    int32_t end = 0;
    if (include_last_offset==0) {
      end = oi < (offsz -1) ? offsets[oi+1] : indsz;
    }
    else {
      end = offsets[oi+1];
    }

    int64_t dst_offset = oi * dst_stride;
    float wt_sum = 0.0f;

    // Initialize output buffer
    if (algo == embag_algo_t::max) {
      for (int j = 0; j < width; ++j) {
        if constexpr(std::is_same_v<OutType, float>) {
          dst[dst_offset + j] = -std::numeric_limits<float>::infinity();
        }
        else {
          dst[dst_offset + j] = 0xFF7F;  // BF16 representation of -inf
        }
      }
    }
    else {
      std::memset(&dst[dst_offset], 0, sizeof(OutType) * width);
    }

    // Accumulator registers for SIMD blocks
    __m512 acc[full_blocks + 1];
    for (int b = 0; b < full_blocks; ++b) {
      acc[b] = _mm512_setzero_ps();
    }
    if (tail > 0) {
      acc[full_blocks] = _mm512_setzero_ps();
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
      wt_sum += wt;
      int64_t input_offset = idx * width;
      __m512 wt_vec = _mm512_set1_ps(wt);

      for (int b = 0; b < full_blocks; ++b) {
        __m512 in_vec = _mm512_loadu_ps(&input[input_offset + b * simd_width]);

        //TODO: Implement BF16 support for gcc < 12
        // if constexpr(std::is_same_v<InType, float>) {
        //   in_vec = _mm512_loadu_ps(&input[input_offset + b * simd_width]);
        // }
        // else {
        //   __m256i bf16_data = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(
        //       &input[input_offset + b * simd_width]));
        //   in_vec = _mm512_cvtpbh_ps((__m256bh)bf16_data);
        // }

        if (algo == embag_algo_t::max) {
          acc[b] = _mm512_max_ps(acc[b], in_vec);
        }
        else {
          acc[b] = _mm512_fmadd_ps(in_vec, wt_vec, acc[b]);
        }
      }

      if (tail > 0) {
        __mmask16 tail_mask = (1 << tail) - 1;
        __m512 in_vec = _mm512_maskz_loadu_ps(tail_mask,
                                              &input[input_offset + full_blocks * simd_width]);

        //TODO: Implement BF16 support for gcc < 12
        // if constexpr(std::is_same_v<InType, float>) {
        //   in_vec = _mm512_maskz_loadu_ps(tail_mask,
        //                                  &input[input_offset + full_blocks * simd_width]);
        // }
        // else {
        //   uint16_t tmp[simd_width] = {0};
        //   std::memcpy(tmp, &input[input_offset + full_blocks * simd_width],
        //               tail * sizeof(uint16_t));
        //   __m256i bf16_data = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(tmp));
        //   in_vec = _mm512_cvtpbh_ps((__m256bh)bf16_data);
        // }

        if (algo == embag_algo_t::max) {
          acc[full_blocks] = _mm512_max_ps(acc[full_blocks], in_vec);
        }
        else {
          acc[full_blocks] = _mm512_fmadd_ps(in_vec, wt_vec, acc[full_blocks]);
        }
      }
    }

    // Normalize for mean reduction
    if (algo == embag_algo_t::mean && wt_sum > 0.0f) {
      __m512 div_vec = _mm512_set1_ps(wt_sum);
      for (int b = 0; b < full_blocks; ++b) {
        acc[b] = _mm512_div_ps(acc[b], div_vec);
      }
      if (tail > 0) {
        acc[full_blocks] = _mm512_div_ps(acc[full_blocks], div_vec);
      }
    }

    // Store results
    for (int b = 0; b < full_blocks; ++b) {
      if constexpr(std::is_same_v<OutType, float>) {
        _mm512_storeu_ps(&dst[dst_offset + b * simd_width], acc[b]);
      }
      //TODO: Implement BF16 support for gcc < 12
      // else {
      //   __m256bh bf16_vec = _mm512_cvtneps_pbh(acc[b]);
      //   _mm256_storeu_si256(reinterpret_cast<__m256i *>(&dst[dst_offset + b *
      //                       simd_width]), (__m256i)bf16_vec);
      // }
    }

    if (tail > 0) {
      if constexpr(std::is_same_v<OutType, float>) {
        __mmask16 tail_mask = (1 << tail) - 1;
        _mm512_mask_storeu_ps(&dst[dst_offset + full_blocks * simd_width], tail_mask,
                              acc[full_blocks]);
      }
      //TODO: Implement BF16 support for gcc < 12
      // else {
      //   __m256bh bf16_vec = _mm512_cvtneps_pbh(acc[full_blocks]);
      //   for (int t = 0; t < tail; ++t) {
      //     dst[dst_offset + full_blocks * simd_width + t] = ((uint16_t *)&bf16_vec)[t];
      //   }
      // }
    }
  }

}

} //namespace ops
} //namespace zendnnl

#endif