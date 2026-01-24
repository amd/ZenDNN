/*******************************************************************************
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

#include "lowoha_operators/reorder/reorder_kernels.hpp"

#include <immintrin.h>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace zendnnl {
namespace lowoha {
namespace reorder {

/**
 * @brief Convert 16 BF16 values to 16 float32 values using AVX512.
 */
__attribute__((target("avx512f")))
static inline __m512 bf16_to_float_vec(__m256i bf16) {
  // Convert 16 uint16_t to 32-bit integers
  __m512i extended = _mm512_cvtepu16_epi32(bf16);
  // Shift left by 16 bits to place BF16 bits in the upper half of float32
  __m512i shifted = _mm512_slli_epi32(extended, 16);
  // Reinterpret the integer bits as float32
  return _mm512_castsi512_ps(shifted);
}

/**
 * @brief Convert 16 float32 values to 16 BF16 values using round-to-nearest-even.
 */
__attribute__((target("avx512f")))
static inline __m256i float_to_bf16_vec(__m512 val) {
  // Reinterpret float32 as int32 for bit manipulation
  __m512i int_val = _mm512_castps_si512(val);
  // Extract LSB of the BF16 part to determine rounding direction
  __m512i lsb = _mm512_and_si512(_mm512_srli_epi32(int_val, 16),
                                 _mm512_set1_epi32(1));
  // Add rounding bias (0x7FFF + lsb) for round-to-nearest-even
  __m512i rounding_bias = _mm512_add_epi32(_mm512_set1_epi32(0x7FFF), lsb);
  // Add bias to original bits
  __m512i rounded = _mm512_add_epi32(int_val, rounding_bias);
  // Shift right to extract upper 16 bits (BF16)
  __m512i bf16 = _mm512_srli_epi32(rounded, 16);
  // Narrow 32-bit integers to 16-bit integers
  return _mm512_cvtepi32_epi16(bf16);
}

/**
 * @brief Batch convert BF16 array to float32 array using AVX512.
 *
 * Computation steps (vectorized for 16 elements):
 *   1. Load 16 BF16 values (256-bit)
 *   2. Zero-extend each 16-bit value to 32-bit integer
 *   3. Shift left by 16 bits to place BF16 mantissa/exponent in float32 format
 *   4. Reinterpret integer bits as float32
 *   5. Scalar fallback handles remaining elements
 */
__attribute__((target("avx512f")))
void bf16_to_float32_avx512(const uint16_t *input, float *output, size_t nelems) {
  size_t i = 0;

  // Process 16 elements at a time using AVX512
  for (; i + 15 < nelems; i += 16) {
    __m256i bf16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input + i));
    __m512 val = bf16_to_float_vec(bf16);
    _mm512_storeu_ps(output + i, val);
  }

  // Handle remaining elements with scalar code
  for (; i < nelems; ++i) {
    uint32_t bits = static_cast<uint32_t>(input[i]) << 16;
    std::memcpy(&output[i], &bits, sizeof(float));
  }
}

/**
 * @brief Batch convert float32 array to BF16 array using AVX512.
 *
 * Computation steps (vectorized for 16 elements):
 *   1. Load 16 float32 values (512-bit)
 *   2. Apply round-to-nearest-even: add rounding bias (0x7FFF + LSB)
 *   3. Shift right by 16 bits to extract upper 16 bits as BF16
 *   4. Narrow 32-bit integers to 16-bit
 *   5. Scalar fallback handles remaining elements with same rounding
 */
__attribute__((target("avx512f")))
void float32_to_bf16_avx512(const float *input, uint16_t *output, size_t nelems) {
  size_t i = 0;

  // Process 16 elements at a time using AVX512
  for (; i + 15 < nelems; i += 16) {
    __m512 val = _mm512_loadu_ps(input + i);
    __m256i bf16 = float_to_bf16_vec(val);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(output + i), bf16);
  }

  // Handle remaining elements with scalar code (round-to-nearest-even)
  for (; i < nelems; ++i) {
    uint32_t bits;
    std::memcpy(&bits, &input[i], sizeof(float));
    uint32_t lsb = (bits >> 16) & 1;
    uint32_t rounding_bias = 0x7FFF + lsb;
    bits += rounding_bias;
    output[i] = static_cast<uint16_t>(bits >> 16);
  }
}

/**
 * @brief Quantize BF16 array to int8 array using AVX512.
 *
 * Computation steps (vectorized for 16 elements):
 *   1. Load 16 BF16 values and convert to float32
 *   2. Divide by scale: scaled_val = val / scale
 *   3. Round to nearest integer (banker's rounding via _mm512_cvtps_epi32)
 *   4. Add zero_point: q = round(val/scale) + zp
 *   5. Clamp to int8 range [-128, 127]
 *   6. Narrow to int8 with saturation
 *   7. Scalar fallback uses nearbyint() for consistent rounding
 */
__attribute__((target("avx512f")))
void quantize_bf16_to_int8_avx512(const uint16_t *input, int8_t *output,
                                   size_t nelems, float scale, int zero_point) {
  size_t i = 0;

  // Prepare broadcast vectors for scale and zero_point
  __m512 scale_vec = _mm512_set1_ps(scale);
  __m512i zp_vec_i32 = _mm512_set1_epi32(zero_point);
  __m512i min_val_i32 = _mm512_set1_epi32(-128);
  __m512i max_val_i32 = _mm512_set1_epi32(127);

  // Process 16 elements at a time using AVX512
  for (; i + 15 < nelems; i += 16) {
    // Load 16 BF16 values
    __m256i bf16_vals = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input + i));

    // Convert BF16 to float32
    __m512 float_vals = bf16_to_float_vec(bf16_vals);

    // Step 1: Divide by scale (in float)
    __m512 scaled_vals = _mm512_div_ps(float_vals, scale_vec);

    // Step 2: Round to nearest integer FIRST (matches reference nearbyint behavior)
    __m512i rounded_vals = _mm512_cvtps_epi32(scaled_vals);

    // Step 3: Add zero_point (in int32)
    __m512i with_zp = _mm512_add_epi32(rounded_vals, zp_vec_i32);

    // Step 4: Clamp to int8 range [-128, 127] AFTER rounding (matches reference)
    __m512i clamped_vals = _mm512_max_epi32(min_val_i32, _mm512_min_epi32(max_val_i32, with_zp));

    // Narrow to int8 with saturation
    __m128i int8_vals = _mm512_cvtsepi32_epi8(clamped_vals);

    // Store 16 int8 values
    _mm_storeu_si128(reinterpret_cast<__m128i *>(output + i), int8_vals);
  }

  // Handle remaining elements with scalar code
  for (; i < nelems; ++i) {
    // Convert BF16 to float32
    uint32_t bits = static_cast<uint32_t>(input[i]) << 16;
    float val;
    std::memcpy(&val, &bits, sizeof(float));

    // Apply quantization (use nearbyint for consistent rounding with reference)
    int32_t q = static_cast<int32_t>(std::nearbyint(val / scale)) + zero_point;
    q = std::max(-128, std::min(127, q));
    output[i] = static_cast<int8_t>(q);
  }
}

/**
 * @brief Dequantize int8 array to BF16 array using AVX512.
 *
 * Computation steps (vectorized for 16 elements):
 *   1. Load 16 int8 values (128-bit)
 *   2. Sign-extend to 32-bit integers
 *   3. Convert to float32
 *   4. Dequantize: val = (x - zero_point) * scale
 *   5. Convert float32 to BF16 with round-to-nearest-even
 *   6. Scalar fallback handles remaining elements
 */
__attribute__((target("avx512f")))
void dequantize_int8_to_bf16_avx512(const int8_t *input, uint16_t *output,
                                     size_t nelems, float scale, int zero_point) {
  size_t i = 0;

  // Prepare broadcast vectors
  __m512 scale_vec = _mm512_set1_ps(scale);
  __m512 zp_vec = _mm512_set1_ps(static_cast<float>(zero_point));

  // Process 16 elements at a time using AVX512
  for (; i + 15 < nelems; i += 16) {
    // Load 16 int8 values
    __m128i int8_vals = _mm_loadu_si128(reinterpret_cast<const __m128i *>(input + i));

    // Extend to 32-bit integers (sign extension)
    __m512i int32_vals = _mm512_cvtepi8_epi32(int8_vals);

    // Dequantize: (x - zp) * scale
    __m512 float_vals = _mm512_mul_ps(
        _mm512_sub_ps(_mm512_cvtepi32_ps(int32_vals), zp_vec),
        scale_vec
    );

    // Convert float32 to BF16
    __m256i bf16_vals = float_to_bf16_vec(float_vals);

    // Store 16 BF16 values
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(output + i), bf16_vals);
  }

  // Handle remaining elements with scalar code
  for (; i < nelems; ++i) {
    float val = (static_cast<float>(input[i]) - zero_point) * scale;
    uint32_t bits;
    std::memcpy(&bits, &val, sizeof(float));
    // Simple truncation for scalar fallback (matches bf16 conversion behavior)
    uint32_t lsb = (bits >> 16) & 1;
    uint32_t rounding_bias = 0x7FFF + lsb;
    bits += rounding_bias;
    output[i] = static_cast<uint16_t>(bits >> 16);
  }
}

void quantize_bf16_to_int8_ref(const uint16_t *input, int8_t *output,
                                size_t nelems, float scale, int zero_point) {
  for (size_t i = 0; i < nelems; ++i) {
    // Convert BF16 to float32
    uint32_t bits = static_cast<uint32_t>(input[i]) << 16;
    float val;
    std::memcpy(&val, &bits, sizeof(float));

    // Apply quantization: (val / scale) + zero_point (use nearbyint for consistent rounding)
    int32_t q = static_cast<int32_t>(std::nearbyint(val / scale)) + zero_point;
    q = std::max(-128, std::min(127, q));
    output[i] = static_cast<int8_t>(q);
  }
}

void dequantize_int8_to_bf16_ref(const int8_t *input, uint16_t *output,
                                  size_t nelems, float scale, int zero_point) {
  for (size_t i = 0; i < nelems; ++i) {
    // Dequantize: (x - zp) * scale
    float val = (static_cast<float>(input[i]) - zero_point) * scale;

    // Convert float32 to BF16 with rounding
    uint32_t bits;
    std::memcpy(&bits, &val, sizeof(float));
    uint32_t lsb = (bits >> 16) & 1;
    uint32_t rounding_bias = 0x7FFF + lsb;
    bits += rounding_bias;
    output[i] = static_cast<uint16_t>(bits >> 16);
  }
}

/**
 * @brief Quantize BF16 array to uint8 array using AVX512.
 *
 * Computation steps (vectorized for 16 elements):
 *   1. Load 16 BF16 values and convert to float32
 *   2. Divide by scale: scaled_val = val / scale
 *   3. Round to nearest integer (banker's rounding via _mm512_cvtps_epi32)
 *   4. Add zero_point: q = round(val/scale) + zp
 *   5. Clamp to uint8 range [0, 255]
 *   6. Narrow to uint8 with unsigned saturation
 *   7. Scalar fallback uses nearbyint() for consistent rounding
 */
__attribute__((target("avx512f")))
void quantize_bf16_to_uint8_avx512(const uint16_t *input, uint8_t *output,
                                    size_t nelems, float scale, int zero_point) {
  size_t i = 0;

  // Prepare broadcast vectors for scale and zero_point
  __m512 scale_vec = _mm512_set1_ps(scale);
  __m512i zp_vec_i32 = _mm512_set1_epi32(zero_point);
  __m512i min_val_i32 = _mm512_set1_epi32(0);
  __m512i max_val_i32 = _mm512_set1_epi32(255);

  // Process 16 elements at a time using AVX512
  for (; i + 15 < nelems; i += 16) {
    // Load 16 BF16 values
    __m256i bf16_vals = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input + i));

    // Convert BF16 to float32
    __m512 float_vals = bf16_to_float_vec(bf16_vals);

    // Step 1: Divide by scale (in float)
    __m512 scaled_vals = _mm512_div_ps(float_vals, scale_vec);

    // Step 2: Round to nearest integer FIRST (matches reference nearbyint behavior)
    __m512i rounded_vals = _mm512_cvtps_epi32(scaled_vals);

    // Step 3: Add zero_point (in int32)
    __m512i with_zp = _mm512_add_epi32(rounded_vals, zp_vec_i32);

    // Step 4: Clamp to uint8 range [0, 255] AFTER rounding (matches reference)
    __m512i int32_vals = _mm512_max_epi32(min_val_i32, _mm512_min_epi32(max_val_i32, with_zp));

    // Narrow to uint8 with saturation using unsigned saturation
    __m128i uint8_vals = _mm512_cvtusepi32_epi8(int32_vals);

    // Store 16 uint8 values
    _mm_storeu_si128(reinterpret_cast<__m128i *>(output + i), uint8_vals);
  }

  // Handle remaining elements with scalar code
  for (; i < nelems; ++i) {
    // Convert BF16 to float32
    uint32_t bits = static_cast<uint32_t>(input[i]) << 16;
    float val;
    std::memcpy(&val, &bits, sizeof(float));

    // Apply quantization (use nearbyint for consistent rounding with reference)
    int32_t q = static_cast<int32_t>(std::nearbyint(val / scale)) + zero_point;
    q = std::max(0, std::min(255, q));
    output[i] = static_cast<uint8_t>(q);
  }
}

/**
 * @brief Dequantize uint8 array to BF16 array using AVX512.
 *
 * Computation steps (vectorized for 16 elements):
 *   1. Load 16 uint8 values (128-bit)
 *   2. Zero-extend to 32-bit integers (unsigned extension)
 *   3. Convert to float32
 *   4. Dequantize: val = (x - zero_point) * scale
 *   5. Convert float32 to BF16 with round-to-nearest-even
 *   6. Scalar fallback handles remaining elements
 */
__attribute__((target("avx512f")))
void dequantize_uint8_to_bf16_avx512(const uint8_t *input, uint16_t *output,
                                      size_t nelems, float scale, int zero_point) {
  size_t i = 0;

  // Prepare broadcast vectors
  __m512 scale_vec = _mm512_set1_ps(scale);
  __m512 zp_vec = _mm512_set1_ps(static_cast<float>(zero_point));

  // Process 16 elements at a time using AVX512
  for (; i + 15 < nelems; i += 16) {
    // Load 16 uint8 values
    __m128i uint8_vals = _mm_loadu_si128(reinterpret_cast<const __m128i *>(input + i));

    // Extend to 32-bit integers (zero extension for unsigned)
    __m512i int32_vals = _mm512_cvtepu8_epi32(uint8_vals);

    // Dequantize: (x - zp) * scale
    __m512 float_vals = _mm512_mul_ps(
        _mm512_sub_ps(_mm512_cvtepi32_ps(int32_vals), zp_vec),
        scale_vec
    );

    // Convert float32 to BF16
    __m256i bf16_vals = float_to_bf16_vec(float_vals);

    // Store 16 BF16 values
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(output + i), bf16_vals);
  }

  // Handle remaining elements with scalar code
  for (; i < nelems; ++i) {
    float val = (static_cast<float>(input[i]) - zero_point) * scale;
    uint32_t bits;
    std::memcpy(&bits, &val, sizeof(float));
    // Round-to-nearest-even for scalar fallback
    uint32_t lsb = (bits >> 16) & 1;
    uint32_t rounding_bias = 0x7FFF + lsb;
    bits += rounding_bias;
    output[i] = static_cast<uint16_t>(bits >> 16);
  }
}

void quantize_bf16_to_uint8_ref(const uint16_t *input, uint8_t *output,
                                 size_t nelems, float scale, int zero_point) {
  for (size_t i = 0; i < nelems; ++i) {
    // Convert BF16 to float32
    uint32_t bits = static_cast<uint32_t>(input[i]) << 16;
    float val;
    std::memcpy(&val, &bits, sizeof(float));

    // Apply quantization: (val / scale) + zero_point
    int32_t q = static_cast<int32_t>(std::round(val / scale) + zero_point);
    q = std::max(0, std::min(255, q));
    output[i] = static_cast<uint8_t>(q);
  }
}

void dequantize_uint8_to_bf16_ref(const uint8_t *input, uint16_t *output,
                                   size_t nelems, float scale, int zero_point) {
  for (size_t i = 0; i < nelems; ++i) {
    // Dequantize: (x - zp) * scale
    float val = (static_cast<float>(input[i]) - zero_point) * scale;

    // Convert float32 to BF16 with rounding
    uint32_t bits;
    std::memcpy(&bits, &val, sizeof(float));
    uint32_t lsb = (bits >> 16) & 1;
    uint32_t rounding_bias = 0x7FFF + lsb;
    bits += rounding_bias;
    output[i] = static_cast<uint16_t>(bits >> 16);
  }
}

//==============================================================================
// FP32 <-> INT8 Conversion Functions
//==============================================================================

/**
 * @brief Quantize float32 array to int8 array using AVX512.
 *
 * Computation steps (vectorized for 16 elements):
 *   1. Load 16 float32 values (512-bit)
 *   2. Divide by scale: scaled_val = val / scale
 *   3. Round to nearest integer (banker's rounding via _mm512_cvtps_epi32)
 *   4. Add zero_point: q = round(val/scale) + zp
 *   5. Clamp to int8 range [-128, 127]
 *   6. Narrow to int8 with signed saturation
 *   7. Scalar fallback uses nearbyint() for consistent rounding
 */
__attribute__((target("avx512f")))
void quantize_f32_to_int8_avx512(const float *input, int8_t *output,
                                  size_t nelems, float scale, int zero_point) {
  size_t i = 0;

  // Prepare broadcast vectors for scale and zero_point
  __m512 scale_vec = _mm512_set1_ps(scale);
  __m512i zp_vec_i32 = _mm512_set1_epi32(zero_point);
  __m512i min_val_i32 = _mm512_set1_epi32(-128);
  __m512i max_val_i32 = _mm512_set1_epi32(127);

  // Process 16 elements at a time using AVX512
  for (; i + 15 < nelems; i += 16) {
    // Load 16 float32 values
    __m512 float_vals = _mm512_loadu_ps(input + i);

    // Step 1: Divide by scale (in float)
    __m512 scaled_vals = _mm512_div_ps(float_vals, scale_vec);

    // Step 2: Round to nearest integer FIRST (matches reference nearbyint behavior)
    __m512i rounded_vals = _mm512_cvtps_epi32(scaled_vals);

    // Step 3: Add zero_point (in int32)
    __m512i with_zp = _mm512_add_epi32(rounded_vals, zp_vec_i32);

    // Step 4: Clamp to int8 range [-128, 127] AFTER rounding (matches reference)
    __m512i clamped_vals = _mm512_max_epi32(min_val_i32, _mm512_min_epi32(max_val_i32, with_zp));

    // Narrow to int8 with saturation
    __m128i int8_vals = _mm512_cvtsepi32_epi8(clamped_vals);

    // Store 16 int8 values
    _mm_storeu_si128(reinterpret_cast<__m128i *>(output + i), int8_vals);
  }

  // Handle remaining elements with scalar code
  for (; i < nelems; ++i) {
    // Apply quantization (use nearbyint for consistent rounding with reference)
    int32_t q = static_cast<int32_t>(std::nearbyint(input[i] / scale)) + zero_point;
    q = std::max(-128, std::min(127, q));
    output[i] = static_cast<int8_t>(q);
  }
}

/**
 * @brief Dequantize int8 array to float32 array using AVX512.
 *
 * Computation steps (vectorized for 16 elements):
 *   1. Load 16 int8 values (128-bit)
 *   2. Sign-extend to 32-bit integers
 *   3. Convert to float32
 *   4. Dequantize: val = (x - zero_point) * scale
 *   5. Store 16 float32 values directly
 *   6. Scalar fallback handles remaining elements
 */
__attribute__((target("avx512f")))
void dequantize_int8_to_f32_avx512(const int8_t *input, float *output,
                                    size_t nelems, float scale, int zero_point) {
  size_t i = 0;

  // Prepare broadcast vectors
  __m512 scale_vec = _mm512_set1_ps(scale);
  __m512 zp_vec = _mm512_set1_ps(static_cast<float>(zero_point));

  // Process 16 elements at a time using AVX512
  for (; i + 15 < nelems; i += 16) {
    // Load 16 int8 values
    __m128i int8_vals = _mm_loadu_si128(reinterpret_cast<const __m128i *>(input + i));

    // Extend to 32-bit integers (sign extension)
    __m512i int32_vals = _mm512_cvtepi8_epi32(int8_vals);

    // Dequantize: (x - zp) * scale
    __m512 float_vals = _mm512_mul_ps(
        _mm512_sub_ps(_mm512_cvtepi32_ps(int32_vals), zp_vec),
        scale_vec
    );

    // Store 16 float32 values
    _mm512_storeu_ps(output + i, float_vals);
  }

  // Handle remaining elements with scalar code
  for (; i < nelems; ++i) {
    output[i] = (static_cast<float>(input[i]) - zero_point) * scale;
  }
}

void quantize_f32_to_int8_ref(const float *input, int8_t *output,
                               size_t nelems, float scale, int zero_point) {
  for (size_t i = 0; i < nelems; ++i) {
    // Apply quantization: (val / scale) + zero_point (use nearbyint for consistent rounding)
    int32_t q = static_cast<int32_t>(std::nearbyint(input[i] / scale)) + zero_point;
    q = std::max(-128, std::min(127, q));
    output[i] = static_cast<int8_t>(q);
  }
}

void dequantize_int8_to_f32_ref(const int8_t *input, float *output,
                                 size_t nelems, float scale, int zero_point) {
  for (size_t i = 0; i < nelems; ++i) {
    // Dequantize: (x - zp) * scale
    output[i] = (static_cast<float>(input[i]) - zero_point) * scale;
  }
}

//==============================================================================
// FP32 <-> UINT8 Conversion Functions
//==============================================================================

/**
 * @brief Quantize float32 array to uint8 array using AVX512.
 *
 * Computation steps (vectorized for 16 elements):
 *   1. Load 16 float32 values (512-bit)
 *   2. Divide by scale: scaled_val = val / scale
 *   3. Round to nearest integer (banker's rounding via _mm512_cvtps_epi32)
 *   4. Add zero_point: q = round(val/scale) + zp
 *   5. Clamp to uint8 range [0, 255]
 *   6. Narrow to uint8 with unsigned saturation
 *   7. Scalar fallback uses nearbyint() for consistent rounding
 */
__attribute__((target("avx512f")))
void quantize_f32_to_uint8_avx512(const float *input, uint8_t *output,
                                   size_t nelems, float scale, int zero_point) {
  size_t i = 0;

  // Prepare broadcast vectors for scale and zero_point
  __m512 scale_vec = _mm512_set1_ps(scale);
  __m512i zp_vec_i32 = _mm512_set1_epi32(zero_point);
  __m512i min_val_i32 = _mm512_set1_epi32(0);
  __m512i max_val_i32 = _mm512_set1_epi32(255);

  // Process 16 elements at a time using AVX512
  for (; i + 15 < nelems; i += 16) {
    // Load 16 float32 values
    __m512 float_vals = _mm512_loadu_ps(input + i);

    // Step 1: Divide by scale (in float)
    __m512 scaled_vals = _mm512_div_ps(float_vals, scale_vec);

    // Step 2: Round to nearest integer FIRST (matches reference nearbyint behavior)
    __m512i rounded_vals = _mm512_cvtps_epi32(scaled_vals);

    // Step 3: Add zero_point (in int32)
    __m512i with_zp = _mm512_add_epi32(rounded_vals, zp_vec_i32);

    // Step 4: Clamp to uint8 range [0, 255] AFTER rounding (matches reference)
    __m512i int32_vals = _mm512_max_epi32(min_val_i32, _mm512_min_epi32(max_val_i32, with_zp));

    // Narrow to uint8 with saturation using unsigned saturation
    __m128i uint8_vals = _mm512_cvtusepi32_epi8(int32_vals);

    // Store 16 uint8 values
    _mm_storeu_si128(reinterpret_cast<__m128i *>(output + i), uint8_vals);
  }

  // Handle remaining elements with scalar code
  for (; i < nelems; ++i) {
    // Apply quantization (use nearbyint for consistent rounding with reference)
    int32_t q = static_cast<int32_t>(std::nearbyint(input[i] / scale)) + zero_point;
    q = std::max(0, std::min(255, q));
    output[i] = static_cast<uint8_t>(q);
  }
}

/**
 * @brief Dequantize uint8 array to float32 array using AVX512.
 *
 * Computation steps (vectorized for 16 elements):
 *   1. Load 16 uint8 values (128-bit)
 *   2. Zero-extend to 32-bit integers (unsigned extension)
 *   3. Convert to float32
 *   4. Dequantize: val = (x - zero_point) * scale
 *   5. Store 16 float32 values directly
 *   6. Scalar fallback handles remaining elements
 */
__attribute__((target("avx512f")))
void dequantize_uint8_to_f32_avx512(const uint8_t *input, float *output,
                                     size_t nelems, float scale, int zero_point) {
  size_t i = 0;

  // Prepare broadcast vectors
  __m512 scale_vec = _mm512_set1_ps(scale);
  __m512 zp_vec = _mm512_set1_ps(static_cast<float>(zero_point));

  // Process 16 elements at a time using AVX512
  for (; i + 15 < nelems; i += 16) {
    // Load 16 uint8 values
    __m128i uint8_vals = _mm_loadu_si128(reinterpret_cast<const __m128i *>(input + i));

    // Extend to 32-bit integers (zero extension for unsigned)
    __m512i int32_vals = _mm512_cvtepu8_epi32(uint8_vals);

    // Dequantize: (x - zp) * scale
    __m512 float_vals = _mm512_mul_ps(
        _mm512_sub_ps(_mm512_cvtepi32_ps(int32_vals), zp_vec),
        scale_vec
    );

    // Store 16 float32 values
    _mm512_storeu_ps(output + i, float_vals);
  }

  // Handle remaining elements with scalar code
  for (; i < nelems; ++i) {
    output[i] = (static_cast<float>(input[i]) - zero_point) * scale;
  }
}

void quantize_f32_to_uint8_ref(const float *input, uint8_t *output,
                                size_t nelems, float scale, int zero_point) {
  for (size_t i = 0; i < nelems; ++i) {
    // Apply quantization: (val / scale) + zero_point
    int32_t q = static_cast<int32_t>(std::round(input[i] / scale) + zero_point);
    q = std::max(0, std::min(255, q));
    output[i] = static_cast<uint8_t>(q);
  }
}

void dequantize_uint8_to_f32_ref(const uint8_t *input, float *output,
                                  size_t nelems, float scale, int zero_point) {
  for (size_t i = 0; i < nelems; ++i) {
    // Dequantize: (x - zp) * scale
    output[i] = (static_cast<float>(input[i]) - zero_point) * scale;
  }
}

//==============================================================================
// FP32 <-> BF16 conversion kernels with optional scale/zero-point
//==============================================================================

/**
 * @brief Convert float32 array to BF16 array with optional scale/zero-point using AVX512.
 *
 * Computation steps (vectorized for 16 elements):
 *   1. If scale=1.0 and zp=0, delegate to simple float32_to_bf16_avx512
 *   2. Load 16 float32 values (512-bit)
 *   3. Apply inverse scaling: scaled_val = val / scale + zero_point
 *   4. Convert to BF16 using round-to-nearest-even
 *   5. Scalar fallback handles remaining elements with same rounding
 *
 * Formula: bf16_val = bf16(f32_val / scale + zero_point)
 */
__attribute__((target("avx512f")))
void convert_f32_to_bf16_avx512(const float *input, uint16_t *output,
                                 size_t nelems, float scale, int zero_point) {
  // If no scaling needed, delegate to the existing simple conversion kernel
  if (scale == 1.0f && zero_point == 0) {
    float32_to_bf16_avx512(input, output, nelems);
    return;
  }
  
  // With scaling: bf16_val = bf16(f32_val / scale + zero_point)
  size_t i = 0;
  __m512 inv_scale_vec = _mm512_set1_ps(1.0f / scale);
  __m512 zp_vec = _mm512_set1_ps(static_cast<float>(zero_point));
  
  // Process 16 elements at a time
  for (; i + 15 < nelems; i += 16) {
    // Load 16 float32 values
    __m512 f32_vals = _mm512_loadu_ps(input + i);
    
    // Apply scaling: val / scale + zero_point
    __m512 scaled_vals = _mm512_add_ps(_mm512_mul_ps(f32_vals, inv_scale_vec), zp_vec);
    
    // Convert to BF16 using float_to_bf16_vec helper
    __m256i bf16_packed = float_to_bf16_vec(scaled_vals);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(output + i), bf16_packed);
  }
  
  // Handle remaining elements with scalar code
  for (; i < nelems; ++i) {
    float val = input[i] / scale + static_cast<float>(zero_point);
    // Convert to BF16 using bit manipulation with rounding
    uint32_t bits;
    std::memcpy(&bits, &val, sizeof(bits));
    uint32_t rounding = 0x7FFF + ((bits >> 16) & 1);
    bits += rounding;
    output[i] = static_cast<uint16_t>(bits >> 16);
  }
}

/**
 * @brief Convert BF16 array to float32 array with optional scale/zero-point using AVX512.
 *
 * Computation steps (vectorized for 16 elements):
 *   1. If scale=1.0 and zp=0, delegate to simple bf16_to_float32_avx512
 *   2. Load 16 BF16 values (256-bit)
 *   3. Convert BF16 to float32 (zero-extend and shift left by 16)
 *   4. Apply dequantization: result = (val - zero_point) * scale
 *   5. Store 16 float32 values (512-bit)
 *   6. Scalar fallback handles remaining elements
 *
 * Formula: f32_val = (bf16_as_f32 - zero_point) * scale
 */
__attribute__((target("avx512f")))
void convert_bf16_to_f32_avx512(const uint16_t *input, float *output,
                                 size_t nelems, float scale, int zero_point) {
  // If no scaling needed, delegate to the existing simple conversion kernel
  if (scale == 1.0f && zero_point == 0) {
    bf16_to_float32_avx512(input, output, nelems);
    return;
  }
  
  // With scaling: f32_val = (bf16_as_f32 - zero_point) * scale
  size_t i = 0;
  __m512 scale_vec = _mm512_set1_ps(scale);
  __m512 zp_vec = _mm512_set1_ps(static_cast<float>(zero_point));
  
  // Process 16 elements at a time
  for (; i + 15 < nelems; i += 16) {
    // Load 16 BF16 values and convert to float32 using helper
    __m256i bf16_vals = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input + i));
    __m512 f32_vals = bf16_to_float_vec(bf16_vals);
    
    // Apply scaling: (val - zero_point) * scale
    __m512 scaled_vals = _mm512_mul_ps(_mm512_sub_ps(f32_vals, zp_vec), scale_vec);
    
    // Store 16 float32 values
    _mm512_storeu_ps(output + i, scaled_vals);
  }
  
  // Handle remaining elements with scalar code
  for (; i < nelems; ++i) {
    // Convert BF16 to float32
    uint32_t bits = static_cast<uint32_t>(input[i]) << 16;
    float val;
    std::memcpy(&val, &bits, sizeof(val));
    
    val = (val - static_cast<float>(zero_point)) * scale;
    output[i] = val;
  }
}

void convert_f32_to_bf16_ref(const float *input, uint16_t *output,
                              size_t nelems, float scale, int zero_point) {
  bool apply_scaling = (scale != 1.0f || zero_point != 0);
  
  for (size_t i = 0; i < nelems; ++i) {
    float val = input[i];
    if (apply_scaling) {
      val = val / scale + static_cast<float>(zero_point);
    }
    // Convert to BF16 using bit manipulation with rounding
    uint32_t bits;
    std::memcpy(&bits, &val, sizeof(bits));
    uint32_t rounding = 0x7FFF + ((bits >> 16) & 1);
    bits += rounding;
    output[i] = static_cast<uint16_t>(bits >> 16);
  }
}

void convert_bf16_to_f32_ref(const uint16_t *input, float *output,
                              size_t nelems, float scale, int zero_point) {
  bool apply_scaling = (scale != 1.0f || zero_point != 0);
  
  for (size_t i = 0; i < nelems; ++i) {
    // Convert BF16 to float32
    uint32_t bits = static_cast<uint32_t>(input[i]) << 16;
    float val;
    std::memcpy(&val, &bits, sizeof(val));
    
    if (apply_scaling) {
      val = (val - static_cast<float>(zero_point)) * scale;
    }
    output[i] = val;
  }
}

} // namespace reorder
} // namespace lowoha
} // namespace zendnnl

