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

void bf16_to_float32_avx512(const uint16_t *input, float *output,
                            size_t nelems);
void float32_to_bf16_avx512(const float *input, uint16_t *output,
                            size_t nelems);
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
void convert_f32_to_f16_avx512(const float *input, uint16_t *output,
                               size_t nelems, float scale, int zero_point);
void convert_f16_to_f32_avx512(const uint16_t *input, float *output,
                               size_t nelems, float scale, int zero_point);
void convert_bf16_to_f16_avx512(const uint16_t *input, uint16_t *output,
                                size_t nelems, float scale, int zero_point);
void convert_f16_to_bf16_avx512(const uint16_t *input, uint16_t *output,
                                size_t nelems, float scale, int zero_point);

// FP16 AVX-512 kernels — F32-FMA backend (F16C load/store + __m512 math).
// Requires AVX-512F and F16C. These are architecturally independent CPUID
// bits, but every shipping CPU with AVX-512F (Skylake-X 2017 and later) also
// has F16C (Ivy Bridge 2012 and later), so the implementations request both
// via the GCC target attribute and the dispatcher does not add a separate
// runtime ISA probe.
void quantize_f16_to_int8_avx512(const uint16_t *input, int8_t *output,
                                  size_t nelems, float scale, int zero_point);
void dequantize_int8_to_f16_avx512(const int8_t *input, uint16_t *output,
                                    size_t nelems, float scale, int zero_point);
void quantize_f16_to_uint8_avx512(const uint16_t *input, uint8_t *output,
                                   size_t nelems, float scale, int zero_point);
void dequantize_uint8_to_f16_avx512(const uint8_t *input, uint16_t *output,
                                     size_t nelems, float scale, int zero_point);

// FP16 AVX-512 kernels — FP16-FMA backend (__m512h native, AVX512-FP16 ISA).
// On toolchains older than GCC 12, the implementations compile to no-op
// stubs and the dispatcher must select the F32-FMA backend instead.
void quantize_f16_to_int8_avx512fp16(const uint16_t *input, int8_t *output,
                                      size_t nelems, float scale, int zero_point);
void dequantize_int8_to_f16_avx512fp16(const int8_t *input, uint16_t *output,
                                        size_t nelems, float scale, int zero_point);
void quantize_f16_to_uint8_avx512fp16(const uint16_t *input, uint8_t *output,
                                       size_t nelems, float scale, int zero_point);
void dequantize_uint8_to_f16_avx512fp16(const uint8_t *input, uint16_t *output,
                                         size_t nelems, float scale, int zero_point);

} // namespace reorder
} // namespace lowoha
} // namespace zendnnl

#endif // _STATIC_KERNELS_HPP
