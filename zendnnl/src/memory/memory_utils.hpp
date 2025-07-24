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
 *******************************************************************************/
#ifndef  _MEMORY_UTILS_HPP_
#define  _MEMORY_UTILS_HPP_

#include <immintrin.h>
#include <cstring>
#include <cstdint>
#include <numeric>
#include <vector>
#include "common/data_types.hpp"
#include "common/zendnnl_global.hpp"

/** @namespace zendnnl
 *  @brief ZenDNNL top level namespace.
 */
namespace zendnnl {
/** @namespace zendnnl::memory
 *  @brief A namespace to contain all memory management related classes, enumerations,
 *  variables and functions.
 */
namespace memory {

using namespace zendnnl::common;

// Utility function to compute the product of all elements in a vector
template <typename T>
int compute_product(const std::vector<T> &vec) {
  return std::accumulate(vec.begin(), vec.end(), 1, std::multiplies<T>());
}

/**
 * @brief Convert BF16 value to float32 value using rounding to nearest-even.
 * @param bf16_val The BF16 value to be converted.
 * @return The converted float32 value.
 */
inline float bf16_to_float_val(int16_t bf16_val) {
  int32_t inter_temp = *((int16_t *) &bf16_val);
  inter_temp = inter_temp << 16;
  float float_value = 0.0;
  memcpy(&float_value, &inter_temp, sizeof(int32_t));
  return float_value;
}

/**
 * @brief Convert float32 value to bf16 value using rounding to nearest-even.
 * @param val The float32 value to be converted.
 * @return The converted bf16 value.
 */
inline int16_t float_to_bf16_val(float val) {
  uint32_t temp;
  std::memcpy(&temp, &val, sizeof(temp));
  return static_cast<int16_t>(temp >> 16);
}

/**
 * @brief Convert 16 float32 values to 16 BF16 values using AVX512 instructions.
 * @param val The 16 float32 values packed in an AVX512 register.
 * @return The converted 16 BF16 values packed in an AVX512 register.
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
 * @brief Convert an array of float32 values to BF16 values with rounding.
 * @param input Pointer to the input array of float32 values.
 * @param output Pointer to the output array of BF16 values.
 * @param count Number of elements to convert.
 */
__attribute__((target("avx512f")))
inline void float32_to_bf16_(const float *input, int16_t *output,
                             size_t count) {
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
 * @brief Convert a BF16 buffer to float32.
 * @param bf16_buf Pointer to the BF16 buffer.
 * @param f32_buf Pointer to the output float32 buffer.
 * @param size size of the buffer.
 */
inline void bf16_to_f32_buf(const uint16_t *bf16_buf, float *f32_buf,
  int64_t size_) {
  for (int64_t j = 0; j < size_; ++j) {
    f32_buf[j] = bf16_to_float_val(static_cast<int16_t>(bf16_buf[j]));
  }
}

/**
 * @brief Reads a value of any data_type_t at a given index and returns it as the specified type.
 * @tparam T The return type (e.g., int32_t or float).
 * @param value Pointer to the array of values.
 * @param data_type The data type of the values (data_type_t).
 * @param index The index of the value to be read and cast.
 * @return The value at the specified index casted to the specified type.
 */
template <typename T>
T read_and_cast(const void *value, data_type_t data_type, size_t index = 0) {
  switch (data_type) {
  case data_type_t::s8:
    return static_cast<T>(reinterpret_cast<const int8_t *>(value)[index]);
  case data_type_t::u8:
    return static_cast<T>(reinterpret_cast<const uint8_t *>(value)[index]);
  case data_type_t::s32:
    return static_cast<T>(reinterpret_cast<const int32_t *>(value)[index]);
  case data_type_t::f32:
    return static_cast<T>(reinterpret_cast<const float *>(value)[index]);
  case data_type_t::bf16:
    return static_cast<T>(bf16_to_float_val(reinterpret_cast<const int16_t *>
                                         (value)[index]));
  default:
    log_error("Unsupported data type for casting");
    return static_cast<T>(0); // Return 0 as a fallback for unsupported types
  }
}

}
}

#endif // _MEMORY_UTILS_HPP_