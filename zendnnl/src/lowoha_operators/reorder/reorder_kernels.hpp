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

#ifndef _REORDER_KERNELS_HPP
#define _REORDER_KERNELS_HPP

#include <cstdint>
#include <cstddef>

namespace zendnnl {
namespace lowoha {
namespace reorder {

/**
 * @brief Convert BF16 array to float32 array using AVX512.
 *
 * Each BF16 value is zero-extended to 32 bits and shifted left by 16 bits
 * to form a valid float32.
 *
 * @param input Pointer to source BF16 data (stored as uint16_t)
 * @param output Pointer to destination float32 data
 * @param nelems Number of elements to convert
 */
void bf16_to_float32_avx512(const uint16_t *input, float *output, size_t nelems);

/**
 * @brief Convert float32 array to BF16 array using AVX512 with rounding.
 *
 * Uses round-to-nearest-even rounding mode for accuracy.
 *
 * @param input Pointer to source float32 data
 * @param output Pointer to destination BF16 data (stored as uint16_t)
 * @param nelems Number of elements to convert
 */
void float32_to_bf16_avx512(const float *input, uint16_t *output, size_t nelems);

/**
 * @brief Quantize BF16 input to int8 output using AVX512.
 *
 * Formula: int8_val = clamp(round(bf16_val / scale) + zero_point, -128, 127)
 *
 * @param input Pointer to source BF16 data (stored as uint16_t)
 * @param output Pointer to destination int8 data
 * @param nelems Number of elements to convert
 * @param scale Scale factor for quantization (must be positive and finite)
 * @param zero_point Zero point offset
 */
void quantize_bf16_to_int8_avx512(const uint16_t *input, int8_t *output,
                                   size_t nelems, float scale, int zero_point);

/**
 * @brief Dequantize int8 input to BF16 output using AVX512.
 *
 * Formula: bf16_val = (int8_val - zero_point) * scale
 *
 * @param input Pointer to source int8 data
 * @param output Pointer to destination BF16 data (stored as uint16_t)
 * @param nelems Number of elements to convert
 * @param scale Scale factor for dequantization (must be positive and finite)
 * @param zero_point Zero point offset
 */
void dequantize_int8_to_bf16_avx512(const int8_t *input, uint16_t *output,
                                     size_t nelems, float scale, int zero_point);

/**
 * @brief Quantize BF16 input to int8 output using reference scalar implementation.
 *
 * @param input Pointer to source BF16 data (stored as uint16_t)
 * @param output Pointer to destination int8 data
 * @param nelems Number of elements to convert
 * @param scale Scale factor for quantization
 * @param zero_point Zero point offset
 */
void quantize_bf16_to_int8_ref(const uint16_t *input, int8_t *output,
                                size_t nelems, float scale, int zero_point);

/**
 * @brief Dequantize int8 input to BF16 output using reference scalar implementation.
 *
 * @param input Pointer to source int8 data
 * @param output Pointer to destination BF16 data (stored as uint16_t)
 * @param nelems Number of elements to convert
 * @param scale Scale factor for dequantization
 * @param zero_point Zero point offset
 */
void dequantize_int8_to_bf16_ref(const int8_t *input, uint16_t *output,
                                  size_t nelems, float scale, int zero_point);

/**
 * @brief Quantize BF16 input to uint8 output using AVX512.
 *
 * Formula: uint8_val = clamp(round(bf16_val / scale) + zero_point, 0, 255)
 *
 * @param input Pointer to source BF16 data (stored as uint16_t)
 * @param output Pointer to destination uint8 data
 * @param nelems Number of elements to convert
 * @param scale Scale factor for quantization (must be positive and finite)
 * @param zero_point Zero point offset
 */
void quantize_bf16_to_uint8_avx512(const uint16_t *input, uint8_t *output,
                                    size_t nelems, float scale, int zero_point);

/**
 * @brief Dequantize uint8 input to BF16 output using AVX512.
 *
 * Formula: bf16_val = (uint8_val - zero_point) * scale
 *
 * @param input Pointer to source uint8 data
 * @param output Pointer to destination BF16 data (stored as uint16_t)
 * @param nelems Number of elements to convert
 * @param scale Scale factor for dequantization (must be positive and finite)
 * @param zero_point Zero point offset
 */
void dequantize_uint8_to_bf16_avx512(const uint8_t *input, uint16_t *output,
                                      size_t nelems, float scale, int zero_point);

/**
 * @brief Quantize BF16 input to uint8 output using reference scalar implementation.
 *
 * @param input Pointer to source BF16 data (stored as uint16_t)
 * @param output Pointer to destination uint8 data
 * @param nelems Number of elements to convert
 * @param scale Scale factor for quantization
 * @param zero_point Zero point offset
 */
void quantize_bf16_to_uint8_ref(const uint16_t *input, uint8_t *output,
                                 size_t nelems, float scale, int zero_point);

/**
 * @brief Dequantize uint8 input to BF16 output using reference scalar implementation.
 *
 * @param input Pointer to source uint8 data
 * @param output Pointer to destination BF16 data (stored as uint16_t)
 * @param nelems Number of elements to convert
 * @param scale Scale factor for dequantization
 * @param zero_point Zero point offset
 */
void dequantize_uint8_to_bf16_ref(const uint8_t *input, uint16_t *output,
                                   size_t nelems, float scale, int zero_point);

} // namespace reorder
} // namespace lowoha
} // namespace zendnnl

#endif // _REORDER_KERNELS_HPP

