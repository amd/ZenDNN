/*******************************************************************************
* Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef EMBAG_AVX2_INT8_INT4_UTILS_HPP
#define EMBAG_AVX2_INT8_INT4_UTILS_HPP

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

// Convert float16 (stored as uint16_t) to float32
inline float half_to_float(uint16_t h) {
  uint32_t sign = (h >> 15) & 0x1;
  uint32_t exponent = (h >> 10) & 0x1F;
  uint32_t mantissa = h & 0x3FF;
  uint32_t f_bits;

  if (exponent == 0) {
    if (mantissa == 0) {
      f_bits = sign << 31; // Zero
    }
    else {
      // Use int32_t for subnormal exponent calculation
      int32_t exp_signed = 1;

      while ((mantissa & 0x400) == 0) {
        mantissa <<= 1;
        exp_signed--;
      }
      mantissa &= 0x3FF;
      exp_signed += 127 - 15;  // Add bias

      // Convert back to uint32_t for bit packing
      exponent = static_cast<uint32_t>(exp_signed);
      f_bits = (sign << 31) | (exponent << 23) | (mantissa << 13);
    }
  }
  else if (exponent == 0x1F) {
    f_bits = (sign << 31) | (0xFF << 23) | (mantissa << 13);
  }
  else {
    exponent += 127 - 15;
    f_bits = (sign << 31) | (exponent << 23) | (mantissa << 13);
  }

  float result;
  std::memcpy(&result, &f_bits, sizeof(result));
  return result;
}

/*-----------------------------------------------------------------------------
| INT8 row processing function.
| Processes a single row of INT8 quantized data and accumulates results.
|
| Parameters:
|   row             - Pointer to the quantized INT8 row data
|   acc             - Accumulator array for SIMD blocks
|   full_blocks     - Number of full 16-element blocks
|   tail            - Number of remaining elements after full blocks
|   scale           - Dequantization scale factor
|   bias            - Dequantization bias
|   wt_vec          - Weight vector (broadcast)
|   is_embedding    - True if embedding lookup (direct store)
|   algo            - Aggregation algorithm (sum, mean, max)
|   first_valid_index - True if this is the first valid row being processed
*/
template <typename InType>
__attribute__((target("avx512f,avx512vl,avx512bf16")))
inline void process_int8_row(
  const InType *row,
  __m512 *acc,
  int full_blocks,
  int tail,
  float scale,
  float bias,
  __m512 wt_vec,
  bool is_embedding,
  embag_algo_t algo,
  bool first_valid_index
) {
  constexpr int simd_width = 16;
  __m512 scale_vec = _mm512_set1_ps(scale);
  __m512 bias_vec = _mm512_set1_ps(bias);

  // Process full blocks (16 elements each)
  for (int b = 0; b < full_blocks; ++b) {
    __m128i qvals_i8 = _mm_loadu_si128(reinterpret_cast<const __m128i *>
                                       (row + b * simd_width));
    __m512i qvals_i32 = _mm512_cvtepi8_epi32(qvals_i8);
    __m512 val_f32 = _mm512_add_ps(
                       _mm512_mul_ps(_mm512_cvtepi32_ps(qvals_i32), scale_vec),
                       bias_vec);

    val_f32 = _mm512_mul_ps(val_f32, wt_vec);

    // Accumulate values
    if (is_embedding) {
      acc[b] = val_f32;
    }
    else {
      if (algo == embag_algo_t::max) {
        if (first_valid_index) {
          acc[b] = val_f32;
        }
        else {
          acc[b] = _mm512_max_ps(acc[b], val_f32);
        }
      }
      else {
        acc[b] = _mm512_add_ps(acc[b], val_f32);
      }
    }
  }

  // Process tail elements
  if (tail > 0) {
    alignas(64) int8_t tmp[simd_width] = {0};
    std::memcpy(tmp, row + full_blocks * simd_width, tail);
    __m128i qvals_i8 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(tmp));
    __m512i qvals_i32 = _mm512_cvtepi8_epi32(qvals_i8);
    __m512 val_f32 = _mm512_add_ps(
                       _mm512_mul_ps(_mm512_cvtepi32_ps(qvals_i32), scale_vec),
                       bias_vec);

    val_f32 = _mm512_mul_ps(val_f32, wt_vec);

    // Accumulate values
    if (is_embedding) {
      acc[full_blocks] = val_f32;
    }
    else {
      if (algo == embag_algo_t::max) {
        if (first_valid_index) {
          acc[full_blocks] = val_f32;
        }
        else {
          acc[full_blocks] = _mm512_max_ps(acc[full_blocks], val_f32);
        }
      }
      else {
        acc[full_blocks] = _mm512_add_ps(acc[full_blocks], val_f32);
      }
    }
  }
}

/*-----------------------------------------------------------------------------
| INT4 row processing function.
| Processes a single row of INT4 quantized data and accumulates results.
| Uses vectorized unpacking of packed INT4 nibbles.
|
| Parameters:
|   row             - Pointer to the quantized INT4 row data (packed nibbles)
|   acc             - Accumulator array for SIMD blocks
|   full_blocks     - Number of full 16-element blocks
|   tail            - Number of remaining elements after full blocks
|   scale           - Dequantization scale factor
|   bias            - Dequantization bias
|   wt_vec          - Weight vector (broadcast)
|   is_embedding    - True if embedding lookup (direct store)
|   algo            - Aggregation algorithm (sum, mean, max)
|   first_valid_index - True if this is the first valid row being processed
|   table_dtype     - Data type (s4 for signed, u4 for unsigned)
*/
template <typename InType>
__attribute__((target("avx512f,avx512vl,avx512bf16")))
inline void process_int4_row(
  const InType *row,
  __m512 *acc,
  int full_blocks,
  int tail,
  float scale,
  float bias,
  __m512 wt_vec,
  bool is_embedding,
  embag_algo_t algo,
  bool first_valid_index,
  data_type_t table_dtype
) {
  __m512 scale_vec = _mm512_set1_ps(scale);
  __m512 bias_vec = _mm512_set1_ps(bias);

  // Process full blocks (16 elements = 8 packed bytes each)
  for (int b = 0; b < full_blocks; ++b) {
    // Load 8 bytes containing 16 packed INT4 values
    __m128i packed_data = _mm_loadl_epi64(reinterpret_cast<const __m128i *>
                                          (row + b * 8));

    // Extract low nibbles (even elements: 0, 2, 4, ..., 14)
    __m128i low_nibbles = _mm_and_si128(packed_data, _mm_set1_epi8(0x0F));

    // Extract high nibbles (odd elements: 1, 3, 5, ..., 15)
    __m128i high_nibbles = _mm_and_si128(_mm_srli_epi16(packed_data, 4),
                                         _mm_set1_epi8(0x0F));

    // Interleave to get correct element order: [e0, e1, e2, e3, ..., e15]
    __m128i interleaved = _mm_unpacklo_epi8(low_nibbles, high_nibbles);

    // Sign extend for signed INT4 (s4): if bit 3 is set, extend to signed byte
    if (table_dtype == data_type_t::s4) {
      __m128i sign_bit = _mm_and_si128(interleaved, _mm_set1_epi8(0x08));
      __m128i needs_extend = _mm_cmpeq_epi8(sign_bit, _mm_set1_epi8(0x08));
      interleaved = _mm_or_si128(interleaved,
                                 _mm_and_si128(needs_extend, _mm_set1_epi8(static_cast<char>(0xF0))));
    }

    // Convert to 32-bit integers and then to float
    __m512i qvals_i32 = _mm512_cvtepi8_epi32(interleaved);
    __m512 val_f32 = _mm512_add_ps(
                       _mm512_mul_ps(_mm512_cvtepi32_ps(qvals_i32), scale_vec),
                       bias_vec);

    val_f32 = _mm512_mul_ps(val_f32, wt_vec);

    // Accumulate values
    if (is_embedding) {
      acc[b] = val_f32;
    }
    else {
      if (algo == embag_algo_t::max) {
        if (first_valid_index) {
          acc[b] = val_f32;
        }
        else {
          acc[b] = _mm512_max_ps(acc[b], val_f32);
        }
      }
      else {
        acc[b] = _mm512_add_ps(acc[b], val_f32);
      }
    }
  }

  // Process tail elements
  if (tail > 0) {
    alignas(16) int8_t tmp[8] = {0};
    std::memcpy(tmp, row + full_blocks * 8, (tail + 1) / 2);

    __m128i packed_data = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(tmp));

    __m128i low_nibbles = _mm_and_si128(packed_data, _mm_set1_epi8(0x0F));
    __m128i high_nibbles = _mm_and_si128(_mm_srli_epi16(packed_data, 4),
                                         _mm_set1_epi8(0x0F));
    __m128i interleaved = _mm_unpacklo_epi8(low_nibbles, high_nibbles);

    if (table_dtype == data_type_t::s4) {
      __m128i sign_bit = _mm_and_si128(interleaved, _mm_set1_epi8(0x08));
      __m128i needs_extend = _mm_cmpeq_epi8(sign_bit, _mm_set1_epi8(0x08));
      interleaved = _mm_or_si128(interleaved,
                                 _mm_and_si128(needs_extend, _mm_set1_epi8(static_cast<char>(0xF0))));
    }

    __m512i qvals_i32 = _mm512_cvtepi8_epi32(interleaved);
    __m512 val_f32 = _mm512_add_ps(
                       _mm512_mul_ps(_mm512_cvtepi32_ps(qvals_i32), scale_vec),
                       bias_vec);

    val_f32 = _mm512_mul_ps(val_f32, wt_vec);

    // Accumulate values
    if (is_embedding) {
      acc[full_blocks] = val_f32;
    }
    else {
      if (algo == embag_algo_t::max) {
        if (first_valid_index) {
          acc[full_blocks] = val_f32;
        }
        else {
          acc[full_blocks] = _mm512_max_ps(acc[full_blocks], val_f32);
        }
      }
      else {
        acc[full_blocks] = _mm512_add_ps(acc[full_blocks], val_f32);
      }
    }
  }
}

/*-----------------------------------------------------------------------------
| AVX-512 optimized kernel for quantized embedding aggregation.
| Supports INT8 and INT4 input formats and FP32 or BF16 output formats.
|
| INT8 Path Register Usage
| | Register Type    | Count | Examples                           |
| |------------------|-------|------------------------------------|
| | Scalar Float     |   3   | scale, wt, wt_sum                  |
| | Scalar Integer   |   6   | idx, zp, start, end, pf_i, pf_idx  |
| | Mask Registers   |   1   | tail_mask                          |
| | Loop Counters    |   4   | oi, i, b, t                        |
| | Vector Registers |   7   | __m128i, __m512i, __m512, __m256bh |
|   Total Registers Used (INT8): 21
|
| INT4 Path Register Usage
| | Register Type    | Count | Examples                                           |
| |------------------|-------|----------------------------------------------------|
| | Scalar Float     |   4   | scale, wt, wt_sum, val                             |
| | Scalar Integer   |   9   | idx, zp, start, end, pf_i, pf_idx, j, packed, qval |
| | Mask Registers   |   1   | tail_mask                                          |
| | Loop Counters    |   4   | oi, i, j, t                                        |
| | Vector Registers |   7   | __m128i, __m512i, __m512, __m256bh                 |
|   Total Registers Used (INT4): 21
|
| Both INT8 and INT4 paths use vectorized processing with AVX-512 intrinsics.
*/

template <
  bool IsInt4,
  typename InType,
  typename IndexType,
  typename OffsetType,
  typename OutType>
__attribute__((target("avx512f,avx512vl,avx512bf16")))
void embag_avx512_int8_int4_kernel(
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
  bool include_last_offset,
  data_type_t table_dtype,
  bool fp16_scale_bias
) {
  constexpr int simd_width = 16;
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
    __m512 *acc = (__m512 *)_mm_malloc(sizeof(__m512) * acc_size, 64);

    for (int i = 0; i < acc_size; ++i) {
      acc[i] = _mm512_setzero_ps();
    }

    for (int i = start; i < end; ++i) {
      int64_t pf_i = i + prefetch_distance;
      int64_t pf_idx = (pf_i < end) ? indices[pf_i] : padidx;
      if (fp16_scale_bias) {
        maybe_prefetch_input(input, pf_idx, width + 4, padidx);
      }
      else {
        maybe_prefetch_input(input, pf_idx, width + 8, padidx);
      }
      if (is_weights) {
        maybe_prefetch_weight(weights, pf_i, end);
      }

      int64_t idx = indices[i];
      if (idx == padidx) {
        continue;
      }

      float wt = is_weights ? weights[i] : 1.0f;
      wt_sum += wt;

      int quantized_size = IsInt4 ? (width + 1) / 2 : width;
      const auto *row = input + idx * (quantized_size + (fp16_scale_bias ? 4 : 8));
      float scale, bias;
      if (fp16_scale_bias) {
        uint16_t scale_fp16, bias_fp16;
        std::memcpy(&scale_fp16, row + quantized_size, sizeof(uint16_t));
        std::memcpy(&bias_fp16, row + quantized_size + 2, sizeof(uint16_t));
        scale = half_to_float(scale_fp16);
        bias = half_to_float(bias_fp16);
      }
      else {
        std::memcpy(&scale, row + quantized_size, sizeof(float));
        std::memcpy(&bias, row + quantized_size + 4, sizeof(float));
      }
      __m512 wt_vec = _mm512_set1_ps(wt);

      // Dispatch to appropriate processing function based on data type
      if constexpr(!IsInt4) {
        process_int8_row(row, acc, full_blocks, tail, scale, bias,
                         wt_vec, is_embedding, algo, first_valid_index);
      }
      else {
        process_int4_row(row, acc, full_blocks, tail, scale, bias,
                         wt_vec, is_embedding, algo, first_valid_index, table_dtype);
      }
      first_valid_index = false;
    }

    if (!is_embedding) {
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
    }

    // Store results
    for (int b = 0; b < full_blocks; ++b) {
      if constexpr(std::is_same_v<OutType, float>) {
        _mm512_storeu_ps(&dst[dst_offset + b * simd_width], acc[b]);
      }
      else {
        __m256bh bf16_vec = _mm512_cvtneps_pbh(acc[b]);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(&dst[dst_offset + b *
                            simd_width]), (__m256i)bf16_vec);
      }
    }
    if (tail > 0) {
      if constexpr(std::is_same_v<OutType, float>) {
        __mmask16 tail_mask = (1 << tail) - 1;
        _mm512_mask_storeu_ps(&dst[dst_offset + full_blocks * simd_width], tail_mask,
                              acc[full_blocks]);
      }
      else {
        __m256bh bf16_vec = _mm512_cvtneps_pbh(acc[full_blocks]);
        uint16_t tmp_store[simd_width];
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(tmp_store), (__m256i)bf16_vec);
        std::memcpy(&dst[dst_offset + full_blocks * simd_width], tmp_store,
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
