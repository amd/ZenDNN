/********************************************************************************
# * Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

namespace zendnnl {
namespace common {

//implementation
bfloat16_t::bfloat16_t():raw_bits_{0} {
}

bfloat16_t::bfloat16_t(float f) : raw_bits_{0} {
  switch (std::fpclassify(f)) {
  case FP_SUBNORMAL:
  case FP_ZERO:
    // sign preserving zero (denormal go to zero)
    raw_bits_  = fp32bf16_t{f}.bf16[1];
    raw_bits_ &= 0x8000;
    break;
  case FP_INFINITE:
    raw_bits_ = fp32bf16_t{f}.bf16[1];
    break;
  case FP_NAN:
    // truncate and set MSB of the mantissa force QNAN
    raw_bits_  = fp32bf16_t{f}.bf16[1];;
    raw_bits_ |= (0x01 << 6);
    break;
  case FP_NORMAL:
    // round to nearest even and truncate
    fp32bf16_t  rbits{f};
    const uint32_t rounding_bias = 0x00007FFF + (rbits.bf16[1] & 0x1);
    rbits.u32                   += rounding_bias;
    raw_bits_                    = rbits.bf16[1];
    break;
  }
}

bfloat16_t::operator float() const {
  return fp32bf16_t{raw_bits_}.fp32;
}

bfloat16_t::operator int() const {
  return int(fp32bf16_t{raw_bits_}.fp32);
}

bfloat16_t &bfloat16_t::operator=(float f) {
  return (*this) = bfloat16_t(f);
}

float bfloat16_t::bf16_to_f32_val(int16_t bf16_val) {
  int32_t inter_temp = *((int16_t *) &bf16_val);
  inter_temp = inter_temp << 16;
  float float_value = 0.0;
  memcpy(&float_value, &inter_temp, sizeof(int32_t));
  return float_value;
}

int16_t bfloat16_t::f32_to_bf16_val(float val) {
  // Use round-to-nearest-even to match hardware behavior
  uint32_t bits;
  std::memcpy(&bits, &val, sizeof(float));
  uint32_t lsb = (bits >> 16) & 1;
  uint32_t rounding_bias = 0x7FFF + lsb;
  bits += rounding_bias;
  return static_cast<int16_t>(bits >> 16);
}

__attribute__((target("avx512f")))
__m256i bfloat16_t::f32_to_bf16_avx512(__m512 val) {
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
void bfloat16_t::f32_to_bf16(const float *input, int16_t *output,
                                  size_t count) {
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

void bfloat16_t::bf16_to_f32_buf(const uint16_t *bf16_buf, float *f32_buf,
                                 int64_t size_) {
  for (int64_t j = 0; j < size_; ++j) {
    f32_buf[j] = bfloat16_t::bf16_to_f32_val(static_cast<int16_t>(bf16_buf[j]));
  }
}

}//namespace common
}//namespace zendnnl

