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

__attribute__((target("avx512f")))
void quantize_bf16_to_int8_avx512(const uint16_t *input, int8_t *output,
                                   size_t nelems, float scale, int zero_point) {
  size_t i = 0;

  // Prepare broadcast vectors for scale and zero_point
  __m512 scale_vec = _mm512_set1_ps(scale);
  __m512 zp_vec = _mm512_set1_ps(static_cast<float>(zero_point));
  __m512 min_val = _mm512_set1_ps(-128.0f);
  __m512 max_val = _mm512_set1_ps(127.0f);

  // Process 16 elements at a time using AVX512
  for (; i + 15 < nelems; i += 16) {
    // Load 16 BF16 values
    __m256i bf16_vals = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input + i));

    // Convert BF16 to float32
    __m512 float_vals = bf16_to_float_vec(bf16_vals);

    // Apply quantization: (val / scale) + zero_point
    __m512 scaled_vals = _mm512_add_ps(_mm512_div_ps(float_vals, scale_vec), zp_vec);

    // Clamp to int8 range [-128, 127]
    __m512 clamped_vals = _mm512_max_ps(min_val, _mm512_min_ps(max_val, scaled_vals));

    // Convert to int32 with rounding
    __m512i int32_vals = _mm512_cvtps_epi32(clamped_vals);

    // Narrow to int8 with saturation
    __m128i int8_vals = _mm512_cvtsepi32_epi8(int32_vals);

    // Store 16 int8 values
    _mm_storeu_si128(reinterpret_cast<__m128i *>(output + i), int8_vals);
  }

  // Handle remaining elements with scalar code
  for (; i < nelems; ++i) {
    // Convert BF16 to float32
    uint32_t bits = static_cast<uint32_t>(input[i]) << 16;
    float val;
    std::memcpy(&val, &bits, sizeof(float));

    // Apply quantization
    int32_t q = static_cast<int32_t>(std::round(val / scale) + zero_point);
    q = std::max(-128, std::min(127, q));
    output[i] = static_cast<int8_t>(q);
  }
}

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

    // Apply quantization: (val / scale) + zero_point
    int32_t q = static_cast<int32_t>(std::round(val / scale) + zero_point);
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

__attribute__((target("avx512f")))
void quantize_bf16_to_uint8_avx512(const uint16_t *input, uint8_t *output,
                                    size_t nelems, float scale, int zero_point) {
  size_t i = 0;

  // Prepare broadcast vectors for scale and zero_point
  __m512 scale_vec = _mm512_set1_ps(scale);
  __m512 zp_vec = _mm512_set1_ps(static_cast<float>(zero_point));
  __m512 min_val = _mm512_set1_ps(0.0f);
  __m512 max_val = _mm512_set1_ps(255.0f);

  // Process 16 elements at a time using AVX512
  for (; i + 15 < nelems; i += 16) {
    // Load 16 BF16 values
    __m256i bf16_vals = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input + i));

    // Convert BF16 to float32
    __m512 float_vals = bf16_to_float_vec(bf16_vals);

    // Apply quantization: (val / scale) + zero_point
    __m512 scaled_vals = _mm512_add_ps(_mm512_div_ps(float_vals, scale_vec), zp_vec);

    // Clamp to uint8 range [0, 255]
    __m512 clamped_vals = _mm512_max_ps(min_val, _mm512_min_ps(max_val, scaled_vals));

    // Convert to int32 with rounding
    __m512i int32_vals = _mm512_cvtps_epi32(clamped_vals);

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

    // Apply quantization
    int32_t q = static_cast<int32_t>(std::round(val / scale) + zero_point);
    q = std::max(0, std::min(255, q));
    output[i] = static_cast<uint8_t>(q);
  }
}

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

} // namespace reorder
} // namespace lowoha
} // namespace zendnnl

