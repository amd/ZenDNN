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

#include "bfloat16.hpp"
#include <cstring>
#include <immintrin.h>
#include "common/zendnnl_compat.hpp"

namespace zendnnl {
namespace common {

void bfloat16_t::bf16_to_f32_buf(
        const uint16_t *bf16_buf, float *f32_buf, int64_t size_) {
    for (int64_t j = 0; j < size_; ++j) {
        f32_buf[j] = bfloat16_t::bf16_to_f32_val(
                static_cast<int16_t>(bf16_buf[j]));
    }
}

void bfloat16_t::f32_to_bf16(
        const float *input, bfloat16_t *output, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        const int16_t bits = f32_to_bf16_val(input[i]);
        output[i] = bfloat16_t::from_bits(static_cast<uint16_t>(bits));
    }
}

ZENDNNL_TARGET("avx512f")
__m256i bfloat16_t::f32_to_bf16_avx512(__m512 val) {
    // Reinterpret float32 as int32 for bit manipulation
    __m512i int_val = _mm512_castps_si512(val);
    // Extract LSB of the BF16 part to determine rounding direction
    __m512i lsb = _mm512_and_si512(
            _mm512_srli_epi32(int_val, 16), _mm512_set1_epi32(1));
    // Add rounding bias (0x7FFF + lsb) for round-to-nearest-even
    __m512i rounding_bias = _mm512_add_epi32(_mm512_set1_epi32(0x7FFF), lsb);
    // Add bias to original bits
    __m512i rounded = _mm512_add_epi32(int_val, rounding_bias);
    // Shift right to extract upper 16 bits (BF16)
    __m512i bf16 = _mm512_srli_epi32(rounded, 16);
    // Narrow 32-bit integers to 16-bit integers
    return _mm512_cvtepi32_epi16(bf16);
}

ZENDNNL_TARGET("avx512f")
void bfloat16_t::f32_to_bf16_vec(
        const float *input, int16_t *output, size_t count) {
    size_t i = 0;
    for (; i + 15 < count; i += 16) {
        // Load 16 float32 values
        __m512 val = _mm512_loadu_ps(input + i);
        // Convert to BF16 with rounding
        __m256i bf16 = bfloat16_t::f32_to_bf16_avx512(val);
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

} //namespace common
} //namespace zendnnl
