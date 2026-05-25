
/********************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef EMBAG_AVX512_F16_UTILS_HPP
#define EMBAG_AVX512_F16_UTILS_HPP

#include <immintrin.h>
#include <cstdint>
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

#if __GNUC__ >= 12

using common::float16_t;
// Boundary load/store helpers shared with the normalization F16-FMA
// kernels (defined in common/float16.hpp). The typed variants pick the
// right intrinsic via `if constexpr` on the storage type, so the FMA
// inner loop below stays in __m512h regardless of whether table/output
// are stored as f16 or f32 in memory.
using common::f16x32_load_typed;
using common::f16x32_load_tail_typed;
using common::f16x32_store_typed;
using common::f16x32_store_tail_typed;

/*-----------------------------------------------------------------------------
  embag_avx512_f16_fma_kernel:
  Unified AVX-512-FP16 embedding bag kernel with native F16 FMA accumulation.

  All compute (FMA, max, div) is performed in FP16 using __m512h registers
  (32 elements per ZMM, 2x throughput over the FP32 path). Type conversions
  happen only at the load/store boundaries, selected at compile time via
  if constexpr on InType / OutType in the shared common::f16x32_*_typed helpers.

  Supported type combinations:
    InType=float16_t, OutType=float16_t  — pure F16: no conversion
    InType=float16_t, OutType=float      — F16 load, widen to F32 on store
    InType=float,     OutType=float16_t  — narrow F32 to F16 on load, F16 store

  Requires AVX512-FP16 ISA (Zen 5 / Sapphire Rapids or later).
  The caller must have verified ISA support before reaching this kernel.

  Template Parameters:
  - InType     : Input table data type (float16_t or float)
  - IndexType  : Index data type (int32_t or int64_t)
  - OffsetType : Offset data type (int32_t or int64_t)
  - OutType    : Output data type (float16_t or float)
*/

template <typename InType, typename IndexType, typename OffsetType,
          typename OutType>
__attribute__((target("avx512f,avx512vl,avx512bw,avx512fp16")))
void embag_avx512_f16_fma_kernel(
  const InType *input,
  const float *weights,
  const IndexType *indices,
  const OffsetType *offsets,
  OutType *dst,
  int64_t width,
  int64_t indsz,
  int64_t offsz,
  int64_t padidx,
  bool is_weights,
  embag_algo_t algo,
  int64_t dst_stride,
  bool include_last_offset
) {
  constexpr int simd_width = 32;
  const int full_blocks = width / simd_width;
  const int tail = width % simd_width;
  // 16 indices of lookahead to better hide L3 / DRAM latency on
  // memory-bound EmbBag workloads (random gathers defeat the HW
  // prefetcher).
  constexpr int prefetch_distance = 16;
  bool is_embedding = (offsets == nullptr);
  int outer_loop = is_embedding ? indsz : offsz;

  const int acc_size = full_blocks + (tail > 0 ? 1 : 0);

  #pragma omp parallel
  {
    __m512h *acc = (__m512h *)_mm_malloc(sizeof(__m512h) * acc_size, 64);

    #pragma omp for
    for (int oi = 0; oi < outer_loop; ++oi) {
      int64_t start = is_embedding ? oi : offsets[oi];
      int64_t end = is_embedding ? oi + 1 : (include_last_offset ? offsets[oi + 1] :
                                             (oi < offsz - 1 ? offsets[oi + 1] : indsz));
      int64_t dst_offset = oi * dst_stride;
      bool first_valid_index = true;
      float wt_sum = 0.0f;

      for (int i = 0; i < acc_size; ++i) {
        acc[i] = _mm512_setzero_ph();
      }

      for (int i = start; i < end; ++i) {
        int64_t pf_i = i + prefetch_distance;
        // Cross-bag prefetch: gate on indsz, not end, so short bags
        // (avg 1-6 indices) still issue meaningful prefetches.
        int64_t pf_idx = (pf_i < indsz) ? indices[pf_i] : padidx;
#ifdef ENABLE_PREFETCH
        if (pf_idx != padidx) {
          // Multi-line prefetch: a single _mm_prefetch fetches only one
          // 64-byte line. For width=128 + F16 the row spans 4 lines; for
          // F32 input it spans 8. HW prefetcher can't predict random
          // gathers, so we issue every line ourselves.
          const char *pf_row = reinterpret_cast<const char *>(
                                 &input[pf_idx * width]);
          const int64_t row_bytes = width * sizeof(InType);
          for (int64_t off = 0; off < row_bytes; off += 64) {
            _mm_prefetch(pf_row + off, _MM_HINT_T0);
          }
        }
        if (is_weights && pf_i < indsz) {
          _mm_prefetch(reinterpret_cast<const char *>(&weights[pf_i]),
                       _MM_HINT_T0);
        }
#endif

        int64_t idx = indices[i];
        if (idx == padidx) {
          continue;
        }

        float wt_f32 = is_weights ? weights[i] : 1.0f;
        wt_sum += wt_f32;
        int64_t input_offset = idx * width;
        // Single f32 -> _Float16 narrowing + broadcast.
        // Avoids a redundant float16_t -> float -> _Float16 round-trip per index.
        __m512h wt_vec = _mm512_set1_ph((_Float16)wt_f32);

        for (int b = 0; b < full_blocks; ++b) {
          __m512h in_vec = f16x32_load_typed<InType>(
                             &input[input_offset + b * simd_width]);

          if (is_embedding) {
            acc[b] = in_vec;
          }
          else {
            if (algo == embag_algo_t::max) {
              acc[b] = first_valid_index ? in_vec
                       : _mm512_max_ph(acc[b], in_vec);
            }
            else {
              acc[b] = _mm512_fmadd_ph(in_vec, wt_vec, acc[b]);
            }
          }
        }

        if (tail > 0) {
          __mmask32 tail_mask = (1u << tail) - 1;
          __m512h in_vec = f16x32_load_tail_typed<InType>(
                             &input[input_offset + full_blocks * simd_width],
                             tail_mask, tail);

          if (is_embedding) {
            acc[full_blocks] = in_vec;
          }
          else {
            if (algo == embag_algo_t::max) {
              acc[full_blocks] = first_valid_index
                                 ? in_vec
                                 : _mm512_mask_max_ph(
                                   acc[full_blocks], tail_mask, acc[full_blocks], in_vec);
            }
            else {
              acc[full_blocks] = _mm512_mask3_fmadd_ph(
                                   in_vec, wt_vec, acc[full_blocks], tail_mask);
            }
          }
        }
        first_valid_index = false;
      }

      if (!is_embedding) {
        if (algo == embag_algo_t::mean && wt_sum > 0.0f) {
          __m512h recip_vec = _mm512_set1_ph((_Float16)(1.0f / wt_sum));
          for (int b = 0; b < full_blocks; ++b) {
            acc[b] = _mm512_mul_ph(acc[b], recip_vec);
          }
          if (tail > 0) {
            acc[full_blocks] = _mm512_mul_ph(acc[full_blocks], recip_vec);
          }
        }
      }

      for (int b = 0; b < full_blocks; ++b) {
        f16x32_store_typed<OutType>(&dst[dst_offset + b * simd_width], acc[b]);
      }

      if (tail > 0) {
        __mmask32 tail_mask = (1u << tail) - 1;
        f16x32_store_tail_typed<OutType>(
          &dst[dst_offset + full_blocks * simd_width], acc[full_blocks],
          tail_mask, tail);
      }
    } // end for oi

    _mm_free(acc);
  } // end omp parallel
}

#endif // __GNUC__ >= 12

} // namespace ops
} // namespace zendnnl

#endif // EMBAG_AVX512_F16_UTILS_HPP
