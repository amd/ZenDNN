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
#include <cassert>
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

/*
 * INT8 row processing function.
 * Processes a single row of INT8 quantized data and accumulates results.
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
    __m512 val_f32 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(qvals_i32), scale_vec,
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
    __m512 val_f32 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(qvals_i32), scale_vec,
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

/*
 * INT4 row processing function.
 * Processes a single row of INT4 quantized data and accumulates results.
 * Uses vectorized unpacking of packed INT4 nibbles.
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

    __m512i qvals_i32 = _mm512_cvtepi8_epi32(interleaved);
    __m512 val_f32 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(qvals_i32), scale_vec,
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
    __m512 val_f32 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(qvals_i32), scale_vec,
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

// Maximum supported width for stack-allocated accumulators
// Supports up to width=4096 (256 blocks * 16 elements per block)
constexpr int MAX_ACC_BLOCKS = 256;

/*-----------------------------------------------------------------------------
 * FULLY SPECIALIZED KERNEL for the most common case:
 *   - Width = 128 (embedding dimension)
 *   - INT4 data type (U4 unsigned)
 *   - SUM algorithm
 *   - FP32 or BF16 output
 *   - FP16 scale/bias
 *   - No weights
 */
template <typename IndexType, typename OffsetType, typename OutType>
__attribute__((target("avx512f,avx512vl,avx512bw,avx512bf16,f16c")))
__attribute__((hot))
__attribute__((flatten))
void embag_int4_w128_sum_specialized(
  const int8_t *__restrict__ input,
  const IndexType *__restrict__ indices,
  const OffsetType *__restrict__ offsets,
  OutType *__restrict__ dst,
  int64_t indsz,
  int64_t offsz,
  int64_t padidx,
  int64_t dst_stride,
  bool include_last_offset
) {
  // Constants for width=128, INT4, FP16 scale/bias
  // Width=128 -> 64 packed bytes (2 INT4 per byte) -> 8 SIMD blocks of 16
  constexpr int PACKED_SIZE = 64;  // 128 INT4 values = 64 bytes
  constexpr int ROW_STRIDE = PACKED_SIZE + 4;  // +4 for FP16 scale/bias

  const __m128i lo_mask = _mm_set1_epi8(0x0F);
  const bool is_embedding = (offsets == nullptr);
  const int outer_loop = is_embedding ? indsz : offsz;

  // Macro to unpack INT4 pairs and accumulate with FMA
#define PROCESS_PAIR_U4(packed, acc_lo, acc_hi, scale_v, bias_v) do { \
    __m128i lo = _mm_and_si128(packed, lo_mask); \
    __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), lo_mask); \
    __m128i u_lo = _mm_unpacklo_epi8(lo, hi); \
    __m128i u_hi = _mm_unpackhi_epi8(lo, hi); \
    __m512i q_lo = _mm512_cvtepu8_epi32(u_lo); \
    __m512i q_hi = _mm512_cvtepu8_epi32(u_hi); \
    __m512 v_lo = _mm512_fmadd_ps(_mm512_cvtepi32_ps(q_lo), scale_v, bias_v); \
    __m512 v_hi = _mm512_fmadd_ps(_mm512_cvtepi32_ps(q_hi), scale_v, bias_v); \
    acc_lo = _mm512_add_ps(acc_lo, v_lo); \
    acc_hi = _mm512_add_ps(acc_hi, v_hi); \
  } while(0)

  #pragma omp parallel for schedule(static)
  for (int oi = 0; oi < outer_loop; ++oi) {
    const int64_t start = is_embedding ? oi : offsets[oi];
    const int64_t end = is_embedding ? oi + 1 : (include_last_offset ? offsets[oi +
                        1] :
                        (oi < offsz - 1 ? offsets[oi + 1] : indsz));

    // Stack-allocated accumulators - fully unrolled for 8 blocks
    alignas(64) __m512 acc0 = _mm512_setzero_ps();
    alignas(64) __m512 acc1 = _mm512_setzero_ps();
    alignas(64) __m512 acc2 = _mm512_setzero_ps();
    alignas(64) __m512 acc3 = _mm512_setzero_ps();
    alignas(64) __m512 acc4 = _mm512_setzero_ps();
    alignas(64) __m512 acc5 = _mm512_setzero_ps();
    alignas(64) __m512 acc6 = _mm512_setzero_ps();
    alignas(64) __m512 acc7 = _mm512_setzero_ps();

    // Main processing loop with aggressive prefetching
    int64_t i = start;

    // Process pairs of rows for better ILP when possible
    for (; i + 1 < end; i += 2) {
      const int64_t idx0 = indices[i];
      const int64_t idx1 = indices[i + 1];

      // Skip padding indices
      const bool valid0 = (idx0 != padidx);
      const bool valid1 = (idx1 != padidx);

      if (valid0) {
        const int8_t *row0 = input + idx0 * ROW_STRIDE;

        // Load scale/bias using hardware F16C
        float scale0, bias0;
        __m128i fp16_pair0 = _mm_cvtsi32_si128(*reinterpret_cast<const int32_t *>
                                               (row0 + PACKED_SIZE));
        __m128 fp32_pair0 = _mm_cvtph_ps(fp16_pair0);
        scale0 = _mm_cvtss_f32(fp32_pair0);
        bias0 = _mm_cvtss_f32(_mm_shuffle_ps(fp32_pair0, fp32_pair0, 1));

        const __m512 scale_vec0 = _mm512_set1_ps(scale0);
        const __m512 bias_vec0 = _mm512_set1_ps(bias0);

        // Load all 64 bytes at once
        const __m128i *ptr0 = reinterpret_cast<const __m128i *>(row0);
        __m128i p0_0 = _mm_loadu_si128(ptr0);
        __m128i p0_1 = _mm_loadu_si128(ptr0 + 1);
        __m128i p0_2 = _mm_loadu_si128(ptr0 + 2);
        __m128i p0_3 = _mm_loadu_si128(ptr0 + 3);

        // Unpack and process all 8 blocks for row 0
        PROCESS_PAIR_U4(p0_0, acc0, acc1, scale_vec0, bias_vec0);
        PROCESS_PAIR_U4(p0_1, acc2, acc3, scale_vec0, bias_vec0);
        PROCESS_PAIR_U4(p0_2, acc4, acc5, scale_vec0, bias_vec0);
        PROCESS_PAIR_U4(p0_3, acc6, acc7, scale_vec0, bias_vec0);
      }

      if (valid1) {
        const int8_t *row1 = input + idx1 * ROW_STRIDE;

        // Load scale/bias using hardware F16C
        float scale1, bias1;
        __m128i fp16_pair1 = _mm_cvtsi32_si128(*reinterpret_cast<const int32_t *>
                                               (row1 + PACKED_SIZE));
        __m128 fp32_pair1 = _mm_cvtph_ps(fp16_pair1);
        scale1 = _mm_cvtss_f32(fp32_pair1);
        bias1 = _mm_cvtss_f32(_mm_shuffle_ps(fp32_pair1, fp32_pair1, 1));

        const __m512 scale_vec1 = _mm512_set1_ps(scale1);
        const __m512 bias_vec1 = _mm512_set1_ps(bias1);

        // Load all 64 bytes at once
        const __m128i *ptr1 = reinterpret_cast<const __m128i *>(row1);
        __m128i p1_0 = _mm_loadu_si128(ptr1);
        __m128i p1_1 = _mm_loadu_si128(ptr1 + 1);
        __m128i p1_2 = _mm_loadu_si128(ptr1 + 2);
        __m128i p1_3 = _mm_loadu_si128(ptr1 + 3);

        // Unpack and process all 8 blocks for row 1
        PROCESS_PAIR_U4(p1_0, acc0, acc1, scale_vec1, bias_vec1);
        PROCESS_PAIR_U4(p1_1, acc2, acc3, scale_vec1, bias_vec1);
        PROCESS_PAIR_U4(p1_2, acc4, acc5, scale_vec1, bias_vec1);
        PROCESS_PAIR_U4(p1_3, acc6, acc7, scale_vec1, bias_vec1);
      }
    }

    // Handle odd remaining row
    for (; i < end; ++i) {
      const int64_t idx = indices[i];
      [[unlikely]] if (idx == padidx) {
        continue;
      }

      const int8_t *row = input + idx * ROW_STRIDE;

      // Load scale/bias
      float scale, bias;
      __m128i fp16_pair = _mm_cvtsi32_si128(*reinterpret_cast<const int32_t *>
                                            (row + PACKED_SIZE));
      __m128 fp32_pair = _mm_cvtph_ps(fp16_pair);
      scale = _mm_cvtss_f32(fp32_pair);
      bias = _mm_cvtss_f32(_mm_shuffle_ps(fp32_pair, fp32_pair, 1));

      const __m512 scale_vec = _mm512_set1_ps(scale);
      const __m512 bias_vec = _mm512_set1_ps(bias);

      const __m128i *ptr = reinterpret_cast<const __m128i *>(row);
      __m128i p0 = _mm_loadu_si128(ptr);
      __m128i p1 = _mm_loadu_si128(ptr + 1);
      __m128i p2 = _mm_loadu_si128(ptr + 2);
      __m128i p3 = _mm_loadu_si128(ptr + 3);

      PROCESS_PAIR_U4(p0, acc0, acc1, scale_vec, bias_vec);
      PROCESS_PAIR_U4(p1, acc2, acc3, scale_vec, bias_vec);
      PROCESS_PAIR_U4(p2, acc4, acc5, scale_vec, bias_vec);
      PROCESS_PAIR_U4(p3, acc6, acc7, scale_vec, bias_vec);
    }

    // Store results
    OutType *dst_ptr = dst + oi * dst_stride;
    if constexpr(std::is_same_v<OutType, float>) {
      // FP32 output path
      _mm512_storeu_ps(dst_ptr, acc0);
      _mm512_storeu_ps(dst_ptr + 16, acc1);
      _mm512_storeu_ps(dst_ptr + 32, acc2);
      _mm512_storeu_ps(dst_ptr + 48, acc3);
      _mm512_storeu_ps(dst_ptr + 64, acc4);
      _mm512_storeu_ps(dst_ptr + 80, acc5);
      _mm512_storeu_ps(dst_ptr + 96, acc6);
      _mm512_storeu_ps(dst_ptr + 112, acc7);
    }
    else {
      // BF16 output path
      __m256bh bf0 = _mm512_cvtneps_pbh(acc0);
      __m256bh bf1 = _mm512_cvtneps_pbh(acc1);
      __m256bh bf2 = _mm512_cvtneps_pbh(acc2);
      __m256bh bf3 = _mm512_cvtneps_pbh(acc3);
      __m256bh bf4 = _mm512_cvtneps_pbh(acc4);
      __m256bh bf5 = _mm512_cvtneps_pbh(acc5);
      __m256bh bf6 = _mm512_cvtneps_pbh(acc6);
      __m256bh bf7 = _mm512_cvtneps_pbh(acc7);
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst_ptr), (__m256i)bf0);
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst_ptr + 16), (__m256i)bf1);
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst_ptr + 32), (__m256i)bf2);
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst_ptr + 48), (__m256i)bf3);
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst_ptr + 64), (__m256i)bf4);
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst_ptr + 80), (__m256i)bf5);
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst_ptr + 96), (__m256i)bf6);
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst_ptr + 112), (__m256i)bf7);
    }
  }
#undef PROCESS_PAIR_U4
}

/*-----------------------------------------------------------------------------
 * AVX-512 optimized kernel for quantized embedding aggregation.
| Supports INT8 and INT4 input formats and FP32 or BF16 output formats.
 * INT4 Path Register Usage
 * | Register Type    | Count | Examples                                           |
 * |------------------|-------|----------------------------------------------------|
 * | Scalar Float     |   4   | scale, wt, wt_sum, val                             |
 * | Scalar Integer   |   9   | idx, zp, start, end, pf_i, pf_idx, j, packed, qval |
 * | Mask Registers   |   1   | tail_mask                                          |
 * | Loop Counters    |   4   | oi, i, j, t                                        |
 * | Vector Registers |   7   | __m128i, __m512i, __m512, __m256bh                 |
 *   Total Registers Used (INT4): 21
 * Both INT8 and INT4 paths use vectorized processing with AVX-512 intrinsics.
 */
template <
  bool IsInt4,
  typename InType,
  typename IndexType,
  typename OffsetType,
  typename OutType>
__attribute__((target("avx512f,avx512vl,avx512bf16,f16c")))
__attribute__((hot))
void embag_avx512_int8_int4_kernel(
  const InType *__restrict__ input,
  const float *__restrict__ weights,
  const IndexType *__restrict__ indices,
  const OffsetType *__restrict__ offsets,
  OutType *__restrict__ dst,
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

  // =========================================================================
  // FAST PATH: Dispatch to specialized kernels for common cases
  // Eliminates all runtime branches for the hot path
  // =========================================================================
  if constexpr(IsInt4) {
    // Check for width=128, SUM algorithm, no weights, U4 data type, FP16 scale/bias
    if (width == 128 && algo == embag_algo_t::sum && !is_weights &&
        table_dtype == data_type_t::u4 && fp16_scale_bias) {
      embag_int4_w128_sum_specialized<IndexType, OffsetType, OutType>(
        reinterpret_cast<const int8_t *>(input), indices, offsets, dst,
        indsz, offsz, padidx, dst_stride, include_last_offset);
      return;
    }
  }

  // =========================================================================
  // GENERIC PATH: Handles all other cases
  // =========================================================================
  constexpr int simd_width = 16;
  const int full_blocks = width / simd_width;
  const int tail = width % simd_width;
  const int acc_size = full_blocks + (tail > 0 ? 1 : 0);
  constexpr int prefetch_distance = 4;  // Reduced for lower latency
  const bool is_embedding = (offsets == nullptr);
  const int outer_loop = is_embedding ? indsz : offsz;

  // Hoist loop-invariant computations
  const int quantized_size = IsInt4 ? (width + 1) / 2 : width;
  const int scale_bias_offset = fp16_scale_bias ? 4 : 8;
  const int64_t row_stride = quantized_size + scale_bias_offset;

  // Ensure width doesn't exceed maximum supported size
  assert(acc_size <= MAX_ACC_BLOCKS && "Width exceeds maximum supported size");

  #pragma omp parallel for schedule(static)
  for (int oi = 0; oi < outer_loop; ++oi) {
    const int64_t start = is_embedding ? oi : offsets[oi];
    const int64_t end = is_embedding ? oi + 1 : (include_last_offset ? offsets[oi +
                        1] :
                        (oi < offsz - 1 ? offsets[oi + 1] : indsz));
    const int64_t dst_offset = oi * dst_stride;
    bool first_valid_index = true;
    float wt_sum = 0.0f;

    alignas(64) __m512 acc[MAX_ACC_BLOCKS];
    for (int i = 0; i < acc_size; ++i) {
      acc[i] = _mm512_setzero_ps();
    }

    for (int64_t i = start; i < end; ++i) {
      // Prefetch future rows
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

      const int64_t idx = indices[i];
      [[unlikely]] if (idx == padidx) {
        continue;
      }

      const float wt = is_weights ? weights[i] : 1.0f;
      wt_sum += wt;

      const auto *row = input + idx * row_stride;
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
      const __m512 wt_vec = _mm512_set1_ps(wt);

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
        const __m512 inv_wt = _mm512_set1_ps(1.0f / wt_sum);
        for (int b = 0; b < full_blocks; ++b) {
          acc[b] = _mm512_mul_ps(acc[b], inv_wt);
        }
        if (tail > 0) {
          acc[full_blocks] = _mm512_mul_ps(acc[full_blocks], inv_wt);
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
  }
}

} //namespace ops
} //namespace zendnnl

#endif
