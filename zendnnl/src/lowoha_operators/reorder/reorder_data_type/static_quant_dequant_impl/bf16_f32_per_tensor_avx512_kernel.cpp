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
 ******************************************************************************/

#include "lowoha_operators/reorder/reorder_data_type/static_quant_dequant_impl/static_kernels.hpp"

#include <immintrin.h>
#include <cstring>
#include <cmath>

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

} // namespace reorder
} // namespace lowoha
} // namespace zendnnl
