/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
 ******************************************************************************/

#include "lowoha_operators/reorder/reorder_data_type/static_quant_dequant_impl/static_kernels.hpp"
#include "lowoha_operators/reorder/lowoha_reorder_utils.hpp"

#include <immintrin.h>
#include <cstring>
#include <cmath>

namespace zendnnl {
namespace lowoha {
namespace reorder {

//==============================================================================
// SIMD helpers for f16 / bf16 conversion using AVX-512.
//==============================================================================

/**
 * @brief Convert 16 IEEE 754 half-precision (f16) values to 16 float32 values.
 *
 * Uses VCVTPH2PS to widen 16 packed f16 values from a 256-bit register
 * into 16 float32 values in a 512-bit register.
 */
__attribute__((target("avx512f")))
static inline __m512 f16_to_float_vec(__m256i f16) {
  return _mm512_cvtph_ps(f16);
}

/**
 * @brief Convert 16 float32 values to 16 IEEE 754 half-precision (f16) values.
 *
 * Uses VCVTPS2PH with round-to-nearest-even rounding mode to narrow 16
 * float32 values from a 512-bit register into 16 packed f16 values in a
 * 256-bit register.
 */
__attribute__((target("avx512f")))
static inline __m256i float_to_f16_vec(__m512 val) {
  return _mm512_cvtps_ph(val, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

/**
 * @brief Convert 16 BF16 values to 16 float32 values using AVX512.
 *
 */
__attribute__((target("avx512f,avx512bw")))
static inline __m512 bf16_to_float_vec(__m256i bf16) {
  __m512i extended = _mm512_cvtepu16_epi32(bf16);
  __m512i shifted = _mm512_slli_epi32(extended, 16);
  return _mm512_castsi512_ps(shifted);
}

/**
 * @brief Convert 16 float32 values to 16 BF16 values using round-to-nearest-even.
 */
__attribute__((target("avx512f")))
static inline __m256i float_to_bf16_vec(__m512 val) {
  __m512i int_val = _mm512_castps_si512(val);
  __m512i lsb = _mm512_and_si512(_mm512_srli_epi32(int_val, 16),
                                 _mm512_set1_epi32(1));
  __m512i rounding_bias = _mm512_add_epi32(_mm512_set1_epi32(0x7FFF), lsb);
  __m512i rounded = _mm512_add_epi32(int_val, rounding_bias);
  __m512i bf16 = _mm512_srli_epi32(rounded, 16);
  return _mm512_cvtepi32_epi16(bf16);
}

//==============================================================================
// FP32 <-> F16 conversion kernels with optional scale/zero-point
//==============================================================================

/**
 * @brief Convert float32 array to F16 array with optional scale/zero-point.
 *
 * Computation steps (vectorized for 16 elements):
 *   1. Load 16 float32 values (512-bit)
 *   2. If scaling is enabled: scaled_val = val / scale + zero_point
 *   3. Convert float32 -> f16 with round-to-nearest-even (VCVTPS2PH)
 *   4. Store 16 f16 values (256-bit)
 *   5. Scalar fallback handles remaining elements with the same rounding
 *
 * Formula: f16_val = f16(f32_val / scale + zero_point)
 *
 * Register usage (per vectorized iteration):
 *   - Scaling path:   4 ZMM (inv_scale_vec, zp_vec, f32_vals, scaled_vals)
 *                   + 1 YMM (f16_packed) = 5 vector registers total
 *   - No-scaling path: 1 ZMM (f32_vals) + 1 YMM (f16_packed) = 2 vector registers
 */
__attribute__((target("avx512f")))
void convert_f32_to_f16_avx512(const float *input, uint16_t *output,
                               size_t nelems, float scale, int zero_point) {
  const bool apply_scaling = (scale != 1.0f || zero_point != 0);

  size_t i = 0;
  if (apply_scaling) {
    __m512 inv_scale_vec = _mm512_set1_ps(1.0f / scale);
    __m512 zp_vec        = _mm512_set1_ps(static_cast<float>(zero_point));

    for (; i + 15 < nelems; i += 16) {
      __m512 f32_vals = _mm512_loadu_ps(input + i);
      __m512 scaled_vals =
        _mm512_add_ps(_mm512_mul_ps(f32_vals, inv_scale_vec), zp_vec);
      __m256i f16_packed = float_to_f16_vec(scaled_vals);
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(output + i), f16_packed);
    }
  }
  else {
    for (; i + 15 < nelems; i += 16) {
      __m512 f32_vals = _mm512_loadu_ps(input + i);
      __m256i f16_packed = float_to_f16_vec(f32_vals);
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(output + i), f16_packed);
    }
  }

  for (; i < nelems; ++i) {
    float val = input[i];
    if (apply_scaling) {
      val = val / scale + static_cast<float>(zero_point);
    }
    output[i] = float_to_f16(val);
  }
}

/**
 * @brief Convert F16 array to float32 array with optional scale/zero-point.
 *
 * Computation steps (vectorized for 16 elements):
 *   1. Load 16 f16 values (256-bit)
 *   2. Convert f16 -> float32 (VCVTPH2PS)
 *   3. If scaling is enabled: result = (val - zero_point) * scale
 *   4. Store 16 float32 values (512-bit)
 *   5. Scalar fallback handles remaining elements
 *
 * Formula: f32_val = (f16_as_f32 - zero_point) * scale
 *
 * Register usage (per vectorized iteration):
 *   - Scaling path:   4 ZMM (scale_vec, zp_vec, f32_vals, scaled_vals)
 *                   + 1 YMM (f16_vals) = 5 vector registers total
 *   - No-scaling path: 1 ZMM (f32_vals) + 1 YMM (f16_vals) = 2 vector registers
 */
__attribute__((target("avx512f")))
void convert_f16_to_f32_avx512(const uint16_t *input, float *output,
                               size_t nelems, float scale, int zero_point) {
  const bool apply_scaling = (scale != 1.0f || zero_point != 0);

  size_t i = 0;
  if (apply_scaling) {
    __m512 scale_vec = _mm512_set1_ps(scale);
    __m512 zp_vec    = _mm512_set1_ps(static_cast<float>(zero_point));

    for (; i + 15 < nelems; i += 16) {
      __m256i f16_vals =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input + i));
      __m512 f32_vals = f16_to_float_vec(f16_vals);
      __m512 scaled_vals =
        _mm512_mul_ps(_mm512_sub_ps(f32_vals, zp_vec), scale_vec);
      _mm512_storeu_ps(output + i, scaled_vals);
    }
  }
  else {
    for (; i + 15 < nelems; i += 16) {
      __m256i f16_vals =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input + i));
      __m512 f32_vals = f16_to_float_vec(f16_vals);
      _mm512_storeu_ps(output + i, f32_vals);
    }
  }

  for (; i < nelems; ++i) {
    float val = f16_to_float(input[i]);
    if (apply_scaling) {
      val = (val - static_cast<float>(zero_point)) * scale;
    }
    output[i] = val;
  }
}

//==============================================================================
// BF16 <-> F16 conversion kernels with optional scale/zero-point
//
// These conversions go through float32 in registers:
//   - bf16 -> f32 (bit shift)         -> [optional scale/zp] -> f32 -> f16
//   - f16  -> f32 (vcvtph2ps)         -> [optional scale/zp] -> f32 -> bf16
//
// Both bf16 and f16 are stored as uint16_t.
//==============================================================================

/**
 * @brief Convert BF16 array to F16 array with optional scale/zero-point.
 *
 * Steps (vectorized for 16 elements):
 *   1. Load 16 bf16 values (256-bit)
 *   2. Widen bf16 -> float32 (zero-extend + shift left by 16)
 *   3. If scaling is enabled: scaled_val = val / scale + zero_point
 *   4. Convert float32 -> f16 (VCVTPS2PH, round-to-nearest-even)
 *   5. Store 16 f16 values (256-bit)
 *   6. Scalar fallback handles remaining elements
 *
 * Formula: f16_val = f16(bf16_as_f32 / scale + zero_point)
 *
 * Register usage (per vectorized iteration, including bf16_to_float_vec temps):
 *   - Scaling path:   6 ZMM (inv_scale_vec, zp_vec, f32_vals, scaled_vals,
 *                            extended, shifted)
 *                   + 2 YMM (bf16_vals, f16_packed) = 8 vector registers total
 *   - No-scaling path: 3 ZMM (f32_vals, extended, shifted)
 *                   + 2 YMM (bf16_vals, f16_packed) = 5 vector registers
 */
__attribute__((target("avx512f,avx512bw")))
void convert_bf16_to_f16_avx512(const uint16_t *input, uint16_t *output,
                                size_t nelems, float scale, int zero_point) {
  const bool apply_scaling = (scale != 1.0f || zero_point != 0);

  size_t i = 0;
  if (apply_scaling) {
    __m512 inv_scale_vec = _mm512_set1_ps(1.0f / scale);
    __m512 zp_vec        = _mm512_set1_ps(static_cast<float>(zero_point));

    for (; i + 15 < nelems; i += 16) {
      __m256i bf16_vals =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input + i));
      __m512 f32_vals = bf16_to_float_vec(bf16_vals);
      __m512 scaled_vals =
        _mm512_add_ps(_mm512_mul_ps(f32_vals, inv_scale_vec), zp_vec);
      __m256i f16_packed = float_to_f16_vec(scaled_vals);
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(output + i), f16_packed);
    }
  }
  else {
    for (; i + 15 < nelems; i += 16) {
      __m256i bf16_vals =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input + i));
      __m512 f32_vals = bf16_to_float_vec(bf16_vals);
      __m256i f16_packed = float_to_f16_vec(f32_vals);
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(output + i), f16_packed);
    }
  }

  for (; i < nelems; ++i) {
    float val = bf16_to_float(input[i]);
    if (apply_scaling) {
      val = val / scale + static_cast<float>(zero_point);
    }
    output[i] = float_to_f16(val);
  }
}

/**
 * @brief Convert F16 array to BF16 array with optional scale/zero-point.
 *
 * Steps (vectorized for 16 elements):
 *   1. Load 16 f16 values (256-bit)
 *   2. Widen f16 -> float32 (VCVTPH2PS)
 *   3. If scaling is enabled: result = (val - zero_point) * scale
 *   4. Convert float32 -> bf16 with round-to-nearest-even
 *   5. Store 16 bf16 values (256-bit)
 *   6. Scalar fallback handles remaining elements
 *
 * Formula: bf16_val = bf16((f16_as_f32 - zero_point) * scale)
 *
 * Register usage (per vectorized iteration, including float_to_bf16_vec temps):
 *   - Scaling path:   ~9 ZMM (scale_vec, zp_vec, f32_vals, scaled_vals,
 *                             int_val, lsb, rounding_bias, rounded, two
 *                             set1_epi32 constants)
 *                   + 2 YMM (f16_vals, bf16_packed) = ~11 vector registers
 *   - No-scaling path: ~7 ZMM (f32_vals + 6 from float_to_bf16_vec)
 *                   + 2 YMM (f16_vals, bf16_packed) = ~9 vector registers
 */
__attribute__((target("avx512f")))
void convert_f16_to_bf16_avx512(const uint16_t *input, uint16_t *output,
                                size_t nelems, float scale, int zero_point) {
  const bool apply_scaling = (scale != 1.0f || zero_point != 0);

  size_t i = 0;
  if (apply_scaling) {
    __m512 scale_vec = _mm512_set1_ps(scale);
    __m512 zp_vec    = _mm512_set1_ps(static_cast<float>(zero_point));

    for (; i + 15 < nelems; i += 16) {
      __m256i f16_vals =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input + i));
      __m512 f32_vals = f16_to_float_vec(f16_vals);
      __m512 scaled_vals =
        _mm512_mul_ps(_mm512_sub_ps(f32_vals, zp_vec), scale_vec);
      __m256i bf16_packed = float_to_bf16_vec(scaled_vals);
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(output + i), bf16_packed);
    }
  }
  else {
    for (; i + 15 < nelems; i += 16) {
      __m256i f16_vals =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input + i));
      __m512 f32_vals = f16_to_float_vec(f16_vals);
      __m256i bf16_packed = float_to_bf16_vec(f32_vals);
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(output + i), bf16_packed);
    }
  }

  for (; i < nelems; ++i) {
    float val = f16_to_float(input[i]);
    if (apply_scaling) {
      val = (val - static_cast<float>(zero_point)) * scale;
    }
    output[i] = float_to_bf16(val);
  }
}

} // namespace reorder
} // namespace lowoha
} // namespace zendnnl
