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

#ifndef _PER_TOKEN_AVX512_KERNEL_HPP
#define _PER_TOKEN_AVX512_KERNEL_HPP

#include <cstdint>
#include <cstddef>

namespace zendnnl {
namespace lowoha {
namespace reorder {

//==============================================================================
// Fused Per-Token Dynamic Quantization Kernels  (AVX-512F)
//
// These kernels compute per-row scale (and zero-point for asymmetric) and
// quantize in a single fused operation per row, keeping row data in cache
// between the min/max reduction and quantization passes.
//
// Symmetric (S8):  scale = max(|min|,|max|) / 127,  zp = 0
//   Q[i] = clamp(round(src[i] / scale), -128, 127)
//
// Asymmetric (U8): scale = (max - min) / 255,  zp = round(-min / scale)
//   Q[i] = clamp(round(src[i] / scale) + zp, 0, 255)
//==============================================================================

// --- BF16 -> S8 Symmetric ---

void dynamic_per_token_quant_bf16_s8_native(const uint16_t *src, int8_t *dst,
                                             float *scales, int64_t M, int64_t N);

// --- F32 -> S8 Symmetric ---

void dynamic_per_token_quant_f32_s8_native(const float *src, int8_t *dst,
                                            float *scales, int64_t M, int64_t N);

// --- BF16 -> U8 Asymmetric ---

void dynamic_per_token_quant_bf16_u8_native(const uint16_t *src, uint8_t *dst,
                                             float *scales, int32_t *zps,
                                             int64_t M, int64_t N);

// --- F32 -> U8 Asymmetric ---

void dynamic_per_token_quant_f32_u8_native(const float *src, uint8_t *dst,
                                            float *scales, int32_t *zps,
                                            int64_t M, int64_t N);

//==============================================================================
// Unfused 2-pass per-token dynamic quantization kernels (AVX-512F)
//
// Pass 1: compute per-row scale/zp (parallel over M rows, AVX-512).
// Pass 2: quantize (parallel over M*N contiguous elements, AVX-512).
// Better thread utilization than fused kernels when M < num_threads.
//==============================================================================

// --- BF16 -> S8 Symmetric ---

void dynamic_per_token_quant_bf16_s8_unfused_native(const uint16_t *src,
                                                     int8_t *dst, float *scales,
                                                     int64_t M, int64_t N);

// --- F32 -> S8 Symmetric ---

void dynamic_per_token_quant_f32_s8_unfused_native(const float *src,
                                                    int8_t *dst, float *scales,
                                                    int64_t M, int64_t N);

// --- BF16 -> U8 Asymmetric ---

void dynamic_per_token_quant_bf16_u8_unfused_native(const uint16_t *src,
                                                     uint8_t *dst, float *scales,
                                                     int32_t *zps,
                                                     int64_t M, int64_t N);

// --- F32 -> U8 Asymmetric ---

void dynamic_per_token_quant_f32_u8_unfused_native(const float *src,
                                                    uint8_t *dst, float *scales,
                                                    int32_t *zps,
                                                    int64_t M, int64_t N);

//==============================================================================
// Scalar (reference) fused per-token dynamic quantization kernels
//
// Identical fused logic as the native AVX-512 kernels above (compute
// per-row scale/zp and quantize in a single cache-friendly pass per row),
// but using scalar C++ code only.  Used when algo == reference.
//==============================================================================

// --- BF16 -> S8 Symmetric ---

void dynamic_per_token_quant_bf16_s8_ref(const uint16_t *src, int8_t *dst,
                                          float *scales, int64_t M, int64_t N);

// --- F32 -> S8 Symmetric ---

void dynamic_per_token_quant_f32_s8_ref(const float *src, int8_t *dst,
                                         float *scales, int64_t M, int64_t N);

// --- BF16 -> U8 Asymmetric ---

void dynamic_per_token_quant_bf16_u8_ref(const uint16_t *src, uint8_t *dst,
                                          float *scales, int32_t *zps,
                                          int64_t M, int64_t N);

// --- F32 -> U8 Asymmetric ---

void dynamic_per_token_quant_f32_u8_ref(const float *src, uint8_t *dst,
                                         float *scales, int32_t *zps,
                                         int64_t M, int64_t N);

} // namespace reorder
} // namespace lowoha
} // namespace zendnnl

#endif // _PER_TOKEN_AVX512_KERNEL_HPP
