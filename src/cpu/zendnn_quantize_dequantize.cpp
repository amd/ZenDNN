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
*******************************************************************************/

#include <immintrin.h>
#include <cstring>
#include <vector>
#include <random>
#include "zendnn.hpp"
#include "zendnn_quantize_dequantize.hpp"

namespace zendnn {

#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <iostream>
#include <limits>

/**
 * @brief Convert 16 BF16 values (stored as uint16_t) to 16 float32 values.
 *
 * Each BF16 value is zero-extended to 32 bits and shifted left by 16 bits to form a valid float32.
 */
__attribute__((target("avx512f")))
inline __m512 bf16_to_float_avx512(__m256i bf16) {
    // Convert 16 uint16_t to 32-bit integers
    __m512i extended = _mm512_cvtepu16_epi32(bf16);
    // Shift left by 16 bits to place BF16 bits in the upper half of float32
    __m512i shifted = _mm512_slli_epi32(extended, 16);
    // Reinterpret the integer bits as float32
    return _mm512_castsi512_ps(shifted);
}

/**
 * @brief Convert 16 float32 values to 16 BF16 values using rounding to nearest-even.
 */
__attribute__((target("avx512f")))
inline __m256i float_to_bf16_avx512(__m512 val) {
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
 * @brief Convert float32 array to BF16 array with rounding.
 */
__attribute__((target("avx512f")))
void float32_to_bf16(const float *input, uint16_t *output, size_t count) {
    size_t i = 0;
    for (; i + 15 < count; i += 16) {
        // Load 16 float32 values
        __m512 val = _mm512_loadu_ps(input + i);
        // Convert to BF16 with rounding
        __m256i bf16 = float_to_bf16_avx512(val);
        // Store 16 BF16 values
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(output + i), bf16);
    }
    // Handle remaining elements
    for (; i < count; ++i) {
        uint32_t bits;
        std::memcpy(&bits, &input[i], sizeof(float));
        uint32_t lsb = (bits >> 16) & 1;
        uint32_t rounding_bias = 0x7FFF + lsb;
        bits += rounding_bias;
        output[i] = static_cast<uint16_t>(bits >> 16);
    }
}

/**
 * @brief Convert BF16 array to float32 array.
 */
__attribute__((target("avx512f")))
void bf16_to_float32(const uint16_t *input, float *output, size_t count) {
    size_t i = 0;
    for (; i + 15 < count; i += 16) {
        // Load 16 BF16 values
        __m256i bf16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input + i));
        // Convert to float32
        __m512 val = bf16_to_float_avx512(bf16);
        // Store 16 float32 values
        _mm512_storeu_ps(output + i, val);
    }
    // Handle remaining elements
    for (; i < count; ++i) {
        uint32_t bits = static_cast<uint32_t>(input[i]) << 16;
        std::memcpy(&output[i], &bits, sizeof(float));
    }
}

/**
 * @brief Quantize BF16 input to int8 output using scale and zero-point.
 */
__attribute__((target("avx512f")))
void zendnn_custom_op::quantize_bf16_to_int8(const void *input_bf16,
        void *output_int8, size_t count, float scale, int zero_point) {
    if (scale <= 0.0f || !std::isfinite(scale)) {
        std::cerr << "Error: scale must be positive and finite." << std::endl;
        return;
    }

    const uint16_t *in = static_cast<const uint16_t *>(input_bf16);
    int8_t *out = static_cast<int8_t *>(output_int8);

    size_t i = 0;
    for (; i + 15 < count; i += 16) {
        // Load 16 BF16 values
        __m256i bf16_vals = _mm256_loadu_si256(reinterpret_cast<const __m256i *>
                                               (in + i));
        // Convert to float32
        __m512 float_vals = bf16_to_float_avx512(bf16_vals);
        // Apply quantization: (val / scale) + zero_point
        __m512 scaled_vals = _mm512_add_ps(_mm512_div_ps(float_vals,
                                           _mm512_set1_ps(scale)), _mm512_set1_ps(zero_point));
        // Clamp to int8 range [-128, 127]
        __m512 clamped_vals = _mm512_max_ps(_mm512_set1_ps(-128.0f),
                                            _mm512_min_ps(_mm512_set1_ps(127.0f), scaled_vals));
        // Convert to int32
        __m512i int32_vals = _mm512_cvtps_epi32(clamped_vals);
        // Narrow to int8 with saturation
        __m128i int8_vals = _mm512_cvtsepi32_epi8(int32_vals);
        // Store 16 int8 values
        _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), int8_vals);
    }

    // Handle remaining elements
    for (; i < count; ++i) {
        float val = std::ldexp(static_cast<float>(in[i]), -16); // BF16 to float32
        int32_t q = static_cast<int32_t>(std::round(val / scale) + zero_point);
        q = std::max(-128, std::min(127, q));
        out[i] = static_cast<int8_t>(q);
    }
}

/**
 * @brief Dequantize int8 input to BF16 output using FMA for performance.
 */
__attribute__((target("avx512f")))
void zendnn_custom_op::dequantize_int8_to_bf16(const void *input_int8,
        void *output_bf16, size_t count, float scale, int zero_point) {
    if (scale <= 0.0f || !std::isfinite(scale)) {
        std::cerr << "Error: scale must be positive and finite." << std::endl;
        return;
    }

    const int8_t *in = static_cast<const int8_t *>(input_int8);
    uint16_t *out = static_cast<uint16_t *>(output_bf16);

    size_t i = 0;
    for (; i + 15 < count; i += 16) {
        // Load 16 int8 values
        __m128i int8_vals = _mm_loadu_si128(reinterpret_cast<const __m128i *>(in + i));
        // Extend to 32-bit integers
        __m512i int32_vals = _mm512_cvtepi8_epi32(int8_vals);
        // Dequantize using FMA: (x - zp) * scale
        __m512 float_vals = _mm512_fmadd_ps(
                                _mm512_sub_ps(_mm512_cvtepi32_ps(int32_vals), _mm512_set1_ps(zero_point)),
                                _mm512_set1_ps(scale),
                                _mm512_setzero_ps()
                            );
        // Convert float32 to BF16
        __m256i bf16_vals = float_to_bf16_avx512(float_vals);
        // Store 16 BF16 values
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i), bf16_vals);
    }

    // Handle remaining elements
    for (; i < count; ++i) {
        float val = (static_cast<float>(in[i]) - zero_point) * scale;
        uint32_t bits;
        std::memcpy(&bits, &val, sizeof(float));
        out[i] = static_cast<uint16_t>(bits >> 16);
    }
}

}