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

#ifndef EMBAG_AVX512_FP32_BF16_UTILS_HPP
#define EMBAG_AVX512_FP32_BF16_UTILS_HPP

#include <immintrin.h>
#include <cstdint>
#include <limits>
#include <omp.h>
#include <type_traits>

#include "embag_avx512_kernels.hpp"
#include "common/float16.hpp"
#define ENABLE_PREFETCH
#ifdef ENABLE_PREFETCH
  #include <xmmintrin.h>
#endif

namespace zendnnl {
namespace ops {

using common::float16_t;

// Prefetch helper for input rows
template <typename T>
inline void maybe_prefetch_input(const T *input, int64_t idx, int64_t width,
                                 int64_t padidx) {
#ifdef ENABLE_PREFETCH
  if (idx != padidx) {
    _mm_prefetch(reinterpret_cast<const char *>(&input[idx * width]), _MM_HINT_T0);
  }
#endif
}

// Prefetch helper for weights
inline void maybe_prefetch_weight(const float *weights, int64_t i,
                                  int64_t end) {
#ifdef ENABLE_PREFETCH
  const float *pf_wt_ptr = (i < end) ? &weights[i] : &weights[end - 1];
  _mm_prefetch(reinterpret_cast<const char *>(pf_wt_ptr), _MM_HINT_T0);
#endif
}

// BF16 (16 x uint16_t packed in a __m256i) -> 16 x FP32.
__attribute__((target("avx512f,avx512bw,avx512bf16")))
static inline __m512 embag_bf16x16_to_fp32(__m256i bf16) {
#if __GNUC__ >= 12
  return _mm512_cvtpbh_ps((__m256bh)bf16);
#else
  return _mm512_castsi512_ps(
           _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16), 16));
#endif
}

/*-----------------------------------------------------------------------------
  embag_avx512_kernel:
  Templated AVX-512 embedding bag kernel that accumulates in FP32. Used as
  the primary kernel for FP32 and BF16, and as the F16 fallback path.

  F16 dispatch:
  - F16 is gated to AVX-512-FP16 hardware at the operator/lowoha entry
    points (see embag_operator_impl.cpp, lowoha_embedding_bag.cpp). On
    CPUs without AVX-512-FP16 the operator returns
    status_t::isa_unsupported and never reaches this kernel for F16.
  - On AVX-512-FP16 hardware, default GCC >= 12 builds route F16 through
    embag_avx512_f16_fma_kernel (native FP16 FMA, FP16 accumulation).
    This kernel is NOT used for F16 in that configuration.
  - This kernel handles F16 only when (the gate above has passed and)
    one of these holds:
      * GCC < 12 (native FP16 intrinsics unavailable), or
      * the build was configured with -DZENDNNL_NATIVE_F32_ACCUM=ON
        (force F32 accumulation for reproducibility).
    In that path, F16 inputs are widened to FP32 via _mm512_cvtph_ps at
    load, accumulation runs in FP32, and results are narrowed back via
    _mm512_cvtps_ph at store.

  Template Parameters:
  - InType   : Input data type (float, uint16_t for BF16, or float16_t)
  - OutType  : Output data type (float, uint16_t for BF16, or float16_t)

  Features:
  - SIMD vectorization using AVX-512
  - Tail-safe handling for non-multiple-of-16 embedding widths
  - Compile-time type branching via if-constexpr on InType / OutType
  - OpenMP-based parallelism over bags

  Conversion intrinsics used at load/store boundaries:
  - BF16: _mm512_cvtpbh_ps (load), _mm512_cvtneps_pbh (store)
  - F16 : _mm512_cvtph_ps  (load), _mm512_cvtps_ph    (store)

  Accumulator array:
  - acc[full_blocks + 1], where each element is a __m512 (16 × FP32 lanes).
  - Example: width = 64 -> full_blocks = 4 -> acc uses 5 zmm registers.

  Tail handling:
  - FP32: masked loads/stores via __mmask16 (_mm512_maskz_loadu_ps /
    _mm512_mask_storeu_ps).
  - BF16/F16: masked 16-bit loads/stores via __mmask16
    (_mm256_maskz_loadu_epi16 / _mm256_mask_storeu_epi16) followed by the
    BF16/F16 <-> FP32 conversion intrinsics. Requires AVX-512BW (already
    universal on AVX-512BF16-capable CPUs).

  Mean reduction:
  - Computed via reciprocal-multiply (1/wt_sum then vmulps) instead of
    vdivps — vdivps has very low throughput on a single divider port,
    while vmulps runs at 1/cycle. Adds ~1 ULP rounding vs true division;
    standard tradeoff used by PyTorch / FBGEMM.

  Approximate register footprint:

  | Component                       | FP32           | BF16 / F16                     |
  |---------------------------------|----------------|--------------------------------|
  | Accumulators (acc[])            | full_blocks+1  | Same                           |
  | Input vector (in_vec)           | 1 × __m512     | 1 × __m512 + 1 × __m256i (+bh) |
  | Weight vector (wt_vec)          | 1 × __m512     | Same                           |
  | Reciprocal vector (mean only)   | 1 × __m512     | Same                           |
  | Mask registers                  | 1–2 × __mmask16| Same                           |
  | Total zmm registers (estimate)  | ~8–12          | ~10–14 (conversion overhead)   |

*/

template <
  typename InType,
  typename IndexType,
  typename OffsetType,
  typename OutType>
__attribute__((target("avx512f,avx512vl,avx512bw,avx512bf16")))
void embag_avx512_kernel(
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

  constexpr int simd_width = 16;
  const int full_blocks = width / simd_width;
  const int tail = width % simd_width;
  constexpr int prefetch_distance = 8;
  bool is_embedding = (offsets == nullptr) ? true : false;
  int outer_loop = is_embedding ? indsz : offsz;

  const int acc_size = full_blocks + (tail > 0 ? 1 : 0);

  #pragma omp parallel
  {
    __m512 *acc = (__m512 *)_mm_malloc(sizeof(__m512) * acc_size, 64);

    #pragma omp for
    for (int oi = 0; oi < outer_loop; ++oi) {
      int64_t start = is_embedding ? oi : offsets[oi];
      int64_t end = is_embedding ? oi + 1 : (include_last_offset ? offsets[oi + 1] :
                                             (oi < offsz - 1 ? offsets[oi + 1] : indsz));
      int64_t dst_offset = oi * dst_stride;
      bool first_valid_index = true;
      float wt_sum = 0.0f;

      for (int i = 0; i < acc_size; ++i) {
        acc[i] = _mm512_setzero_ps();
      }

      for (int i = start; i < end; ++i) {
        int64_t pf_i = i + prefetch_distance;
        int64_t pf_idx = (pf_i < end) ? indices[pf_i] : padidx;
        maybe_prefetch_input(input, pf_idx, width, padidx);
        if (is_weights) {
          maybe_prefetch_weight(weights, pf_i, end);
        }

        int64_t idx = indices[i];
        if (idx == padidx) {
          continue;
        }

        float wt = is_weights ? weights[i] : 1.0f;
        wt_sum += wt;
        int64_t input_offset = idx * width;
        __m512 wt_vec = _mm512_set1_ps(wt);

        for (int b = 0; b < full_blocks; ++b) {
          __m512 in_vec;
          if constexpr(std::is_same_v<InType, float>) {
            in_vec = _mm512_loadu_ps(&input[input_offset + b * simd_width]);
          }
          else if constexpr(std::is_same_v<InType, float16_t>) {
            __m256i f16_data = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(
                                                    &input[input_offset + b * simd_width]));
            in_vec = _mm512_cvtph_ps(f16_data);
          }
          else {
            __m256i bf16_data = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(
                &input[input_offset + b * simd_width]));
            in_vec = embag_bf16x16_to_fp32(bf16_data);
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
                acc[b] = _mm512_max_ps(acc[b], in_vec);
              }
            }
            else {
              acc[b] = _mm512_fmadd_ps(in_vec, wt_vec, acc[b]);
            }
          }
        }

        if (tail > 0) {
          __mmask16 tail_mask = (1 << tail) - 1;
          __m512 in_vec;
          if constexpr(std::is_same_v<InType, float>) {
            in_vec = _mm512_maskz_loadu_ps(tail_mask,
                                           &input[input_offset + full_blocks * simd_width]);
          }
          else if constexpr(std::is_same_v<InType, float16_t>) {
            __m256i f16_data = _mm256_maskz_loadu_epi16(
                                 tail_mask,
                                 reinterpret_cast<const __m256i *>(&input[input_offset +
                                                  full_blocks * simd_width]));
            in_vec = _mm512_cvtph_ps(f16_data);
          }
          else {
            __m256i bf16_data = _mm256_maskz_loadu_epi16(
                                  tail_mask,
                                  reinterpret_cast<const __m256i *>(&input[input_offset +
                                                   full_blocks * simd_width]));
            in_vec = embag_bf16x16_to_fp32(bf16_data);
          }
          if (is_embedding) {
            acc[full_blocks] = in_vec;
          }
          else {
            if (algo == embag_algo_t::max) {
              if (first_valid_index) {
                acc[full_blocks] = in_vec;
              }
              else {
                acc[full_blocks] = _mm512_max_ps(acc[full_blocks], in_vec);
              }
            }
            else {
              acc[full_blocks] = _mm512_fmadd_ps(in_vec, wt_vec, acc[full_blocks]);
            }
          }
        }
        first_valid_index = false;
      }

      if (!is_embedding) {
        if (algo == embag_algo_t::mean && wt_sum > 0.0f) {
          __m512 recip_vec = _mm512_set1_ps(1.0f / wt_sum);
          for (int b = 0; b < full_blocks; ++b) {
            acc[b] = _mm512_mul_ps(acc[b], recip_vec);
          }
          if (tail > 0) {
            acc[full_blocks] = _mm512_mul_ps(acc[full_blocks], recip_vec);
          }
        }
      }

      for (int b = 0; b < full_blocks; ++b) {
        if constexpr(std::is_same_v<OutType, float>) {
          _mm512_storeu_ps(&dst[dst_offset + b * simd_width], acc[b]);
        }
        else if constexpr(std::is_same_v<OutType, float16_t>) {
          __m256i f16_vec = _mm512_cvtps_ph(acc[b],
                                            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
          _mm256_storeu_si256(reinterpret_cast<__m256i *>(&dst[dst_offset + b *
                                         simd_width]), f16_vec);
        }
        else {
          __m256bh bf16_vec = _mm512_cvtneps_pbh(acc[b]);
          _mm256_storeu_si256(reinterpret_cast<__m256i *>(&dst[dst_offset + b *
                                         simd_width]), (__m256i)bf16_vec);
        }
      }

      if (tail > 0) {
        __mmask16 tail_mask = (1 << tail) - 1;
        if constexpr(std::is_same_v<OutType, float>) {
          _mm512_mask_storeu_ps(&dst[dst_offset + full_blocks * simd_width], tail_mask,
                                acc[full_blocks]);
        }
        else if constexpr(std::is_same_v<OutType, float16_t>) {
          __m256i f16_vec = _mm512_cvtps_ph(acc[full_blocks],
                                            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
          _mm256_mask_storeu_epi16(
            reinterpret_cast<__m256i *>(&dst[dst_offset + full_blocks * simd_width]),
            tail_mask, f16_vec);
        }
        else {
          __m256bh bf16_vec = _mm512_cvtneps_pbh(acc[full_blocks]);
          _mm256_mask_storeu_epi16(
            reinterpret_cast<__m256i *>(&dst[dst_offset + full_blocks * simd_width]),
            tail_mask, (__m256i)bf16_vec);
        }
      }
    } // end for oi

    _mm_free(acc);
  } // end omp parallel
}

} //namespace ops
} //namespace zendnnl

#endif