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

#ifndef _SCALAR_KERNELS_HPP
#define _SCALAR_KERNELS_HPP

#include "lowoha_operators/reorder/lowoha_reorder_common.hpp"
#include <cstdint>
#include <cstddef>

namespace zendnnl {
namespace lowoha {
namespace reorder {

// Ref bulk per-tensor quant/dequant kernels
void quantize_bf16_to_int8_ref(const uint16_t *input, int8_t *output,
                                size_t nelems, float scale, int zero_point);
void dequantize_int8_to_bf16_ref(const int8_t *input, uint16_t *output,
                                  size_t nelems, float scale, int zero_point);
void quantize_bf16_to_uint8_ref(const uint16_t *input, uint8_t *output,
                                 size_t nelems, float scale, int zero_point);
void dequantize_uint8_to_bf16_ref(const uint8_t *input, uint16_t *output,
                                   size_t nelems, float scale, int zero_point);
void quantize_f32_to_int8_ref(const float *input, int8_t *output,
                               size_t nelems, float scale, int zero_point);
void dequantize_int8_to_f32_ref(const int8_t *input, float *output,
                                 size_t nelems, float scale, int zero_point);
void quantize_f32_to_uint8_ref(const float *input, uint8_t *output,
                                size_t nelems, float scale, int zero_point);
void dequantize_uint8_to_f32_ref(const uint8_t *input, float *output,
                                  size_t nelems, float scale, int zero_point);
void convert_f32_to_bf16_ref(const float *input, uint16_t *output,
                              size_t nelems, float scale, int zero_point);
void convert_bf16_to_f32_ref(const uint16_t *input, float *output,
                              size_t nelems, float scale, int zero_point);

// Ref fused per-token dynamic quantization kernels
void dynamic_per_token_quant_bf16_s8_ref(const uint16_t *src, int8_t *dst,
                                          float *scales, int64_t M, int64_t N);
void dynamic_per_token_quant_f32_s8_ref(const float *src, int8_t *dst,
                                         float *scales, int64_t M, int64_t N);
void dynamic_per_token_quant_bf16_u8_ref(const uint16_t *src, uint8_t *dst,
                                          float *scales, int32_t *zps,
                                          int64_t M, int64_t N);
void dynamic_per_token_quant_f32_u8_ref(const float *src, uint8_t *dst,
                                         float *scales, int32_t *zps,
                                         int64_t M, int64_t N);

// Scalar granular implementations
void reorder_granular_scaler_impl_2d(const void *src, void *dst,
                                      const reorder_params_t &params);
void reorder_granular_scaler_impl_3d(const void *src, void *dst,
                                      const reorder_params_t &params);

// Scalar fused dynamic per-token dispatch
bool dispatch_fused_per_token_ref(const void *src, void *dst,
                                   const reorder_params_t &params,
                                   int64_t M, int64_t N);

} // namespace reorder
} // namespace lowoha
} // namespace zendnnl

#endif // _SCALAR_KERNELS_HPP
