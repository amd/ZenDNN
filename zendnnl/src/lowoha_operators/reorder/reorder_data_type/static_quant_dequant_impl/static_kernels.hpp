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

#ifndef _STATIC_KERNELS_HPP
#define _STATIC_KERNELS_HPP

#include <cstdint>
#include <cstddef>

namespace zendnnl {
namespace lowoha {
namespace reorder {

void bf16_to_float32_avx512(const uint16_t *input, float *output, size_t nelems);
void float32_to_bf16_avx512(const float *input, uint16_t *output, size_t nelems);
void quantize_bf16_to_int8_avx512(const uint16_t *input, int8_t *output,
                                   size_t nelems, float scale, int zero_point);
void dequantize_int8_to_bf16_avx512(const int8_t *input, uint16_t *output,
                                     size_t nelems, float scale, int zero_point);
void quantize_bf16_to_uint8_avx512(const uint16_t *input, uint8_t *output,
                                    size_t nelems, float scale, int zero_point);
void dequantize_uint8_to_bf16_avx512(const uint8_t *input, uint16_t *output,
                                      size_t nelems, float scale, int zero_point);
void quantize_f32_to_int8_avx512(const float *input, int8_t *output,
                                  size_t nelems, float scale, int zero_point);
void dequantize_int8_to_f32_avx512(const int8_t *input, float *output,
                                    size_t nelems, float scale, int zero_point);
void quantize_f32_to_uint8_avx512(const float *input, uint8_t *output,
                                   size_t nelems, float scale, int zero_point);
void dequantize_uint8_to_f32_avx512(const uint8_t *input, float *output,
                                     size_t nelems, float scale, int zero_point);
void convert_f32_to_bf16_avx512(const float *input, uint16_t *output,
                                 size_t nelems, float scale, int zero_point);
void convert_bf16_to_f32_avx512(const uint16_t *input, float *output,
                                 size_t nelems, float scale, int zero_point);

} // namespace reorder
} // namespace lowoha
} // namespace zendnnl

#endif // _STATIC_KERNELS_HPP
