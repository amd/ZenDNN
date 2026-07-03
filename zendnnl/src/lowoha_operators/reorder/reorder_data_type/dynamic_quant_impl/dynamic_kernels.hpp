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

#ifndef _DYNAMIC_KERNELS_HPP
#define _DYNAMIC_KERNELS_HPP

#include "lowoha_operators/reorder/lowoha_reorder_common.hpp"
#include <cstdint>
#include <cstddef>
#include <vector>

namespace zendnnl {
namespace lowoha {
namespace reorder {

// Fused per-token AVX512 native kernels (F32-FMA backend, Strategy B)
void dynamic_per_token_quant_bf16_s8_native(const uint16_t *src, int8_t *dst,
                                             float *scales, int64_t M, int64_t N);
void dynamic_per_token_quant_f32_s8_native(const float *src, int8_t *dst,
                                            float *scales, int64_t M, int64_t N);
void dynamic_per_token_quant_f16_s8_native(const uint16_t *src, int8_t *dst,
                                            float *scales, int64_t M, int64_t N);
void dynamic_per_token_quant_bf16_u8_native(const uint16_t *src, uint8_t *dst,
                                             float *scales, int32_t *zps,
                                             int64_t M, int64_t N);
void dynamic_per_token_quant_f32_u8_native(const float *src, uint8_t *dst,
                                            float *scales, int32_t *zps,
                                            int64_t M, int64_t N);
void dynamic_per_token_quant_f16_u8_native(const uint16_t *src, uint8_t *dst,
                                            float *scales, int32_t *zps,
                                            int64_t M, int64_t N);

// Fused per-group AVX512 native kernels. Scale/zp layout is {M, G};
// group size is K / G and must divide K exactly.
void dynamic_per_group_quant_bf16_s8_native(const uint16_t *src, int8_t *dst,
                                             float *scales,
                                             int64_t M, int64_t K, int64_t G);
void dynamic_per_group_quant_f32_s8_native(const float *src, int8_t *dst,
                                            float *scales,
                                            int64_t M, int64_t K, int64_t G);
void dynamic_per_group_quant_f16_s8_native(const uint16_t *src, int8_t *dst,
                                            float *scales,
                                            int64_t M, int64_t K, int64_t G);
void dynamic_per_group_quant_bf16_u8_native(const uint16_t *src, uint8_t *dst,
                                             float *scales, int32_t *zps,
                                             int64_t M, int64_t K, int64_t G);
void dynamic_per_group_quant_f32_u8_native(const float *src, uint8_t *dst,
                                            float *scales, int32_t *zps,
                                            int64_t M, int64_t K, int64_t G);
void dynamic_per_group_quant_f16_u8_native(const uint16_t *src, uint8_t *dst,
                                            float *scales, int32_t *zps,
                                            int64_t M, int64_t K, int64_t G);

// Per-row scale / zp only (AVX-512F pass 1), contiguous [M,N], batch = 1.
// Used by compute_dynamic_quant_params when dst is nullptr (compute scales
// only) and by the unfused 2-pass kernels below. num_threads <= 0 uses
// omp_get_max_threads() for the OpenMP team size.
void dynamic_per_token_compute_scales_bf16_s8_symmetric(const uint16_t *src,
                                                         float *scales,
                                                         int64_t M, int64_t N,
                                                         int num_threads);
void dynamic_per_token_compute_scales_f32_s8_symmetric(const float *src,
                                                        float *scales,
                                                        int64_t M, int64_t N,
                                                        int num_threads);
void dynamic_per_token_compute_scales_bf16_u8_asymmetric(const uint16_t *src,
                                                          float *scales,
                                                          int32_t *zps,
                                                          int64_t M, int64_t N,
                                                          int num_threads);
void dynamic_per_token_compute_scales_f32_u8_asymmetric(const float *src,
                                                         float *scales,
                                                         int32_t *zps,
                                                         int64_t M, int64_t N,
                                                         int num_threads);
void dynamic_per_token_compute_scales_f16_s8_symmetric(const uint16_t *src,
                                                        float *scales,
                                                        int64_t M, int64_t N,
                                                        int num_threads);
void dynamic_per_token_compute_scales_f16_u8_asymmetric(const uint16_t *src,
                                                         float *scales,
                                                         int32_t *zps,
                                                         int64_t M, int64_t N,
                                                         int num_threads);

// Unfused 2-pass per-token AVX512 native kernels (F32-FMA backend)
void dynamic_per_token_quant_bf16_s8_unfused_native(const uint16_t *src,
                                                     int8_t *dst, float *scales,
                                                     int64_t M, int64_t N);
void dynamic_per_token_quant_f32_s8_unfused_native(const float *src,
                                                    int8_t *dst, float *scales,
                                                    int64_t M, int64_t N);
void dynamic_per_token_quant_f16_s8_unfused_native(const uint16_t *src,
                                                    int8_t *dst, float *scales,
                                                    int64_t M, int64_t N);
void dynamic_per_token_quant_bf16_u8_unfused_native(const uint16_t *src,
                                                     uint8_t *dst, float *scales,
                                                     int32_t *zps,
                                                     int64_t M, int64_t N);
void dynamic_per_token_quant_f32_u8_unfused_native(const float *src,
                                                    uint8_t *dst, float *scales,
                                                    int32_t *zps,
                                                    int64_t M, int64_t N);
void dynamic_per_token_quant_f16_u8_unfused_native(const uint16_t *src,
                                                    uint8_t *dst, float *scales,
                                                    int32_t *zps,
                                                    int64_t M, int64_t N);

// FP16-FMA backend (Strategy A) — native __m512h kernels requiring
// AVX512-FP16 ISA. On toolchains older than GCC 12, these compile to
// no-op stubs and the dispatcher must select the F32-FMA backend.
void dynamic_per_token_quant_f16_s8_avx512fp16(const uint16_t *src, int8_t *dst,
                                             float *scales,
                                             int64_t M, int64_t N);
void dynamic_per_token_quant_f16_u8_avx512fp16(const uint16_t *src, uint8_t *dst,
                                             float *scales, int32_t *zps,
                                             int64_t M, int64_t N);
// Unfused 2-pass FP16-FMA per-token kernels: Pass 1 parallel over M, Pass 2
// parallel over M*N elements via zendnnl_parallel_for. Used when
// ZENDNNL_DYNAMIC_QUANT_ALGO=2 selects the unfused override and the FP16-FMA
// backend is active.
void dynamic_per_token_quant_f16_s8_unfused_avx512fp16(const uint16_t *src,
                                                     int8_t *dst,
                                                     float *scales,
                                                     int64_t M, int64_t N);
void dynamic_per_token_quant_f16_u8_unfused_avx512fp16(const uint16_t *src,
                                                     uint8_t *dst,
                                                     float *scales,
                                                     int32_t *zps,
                                                     int64_t M, int64_t N);
void dynamic_per_group_quant_f16_s8_avx512fp16(const uint16_t *src, int8_t *dst,
                                             float *scales,
                                             int64_t M, int64_t K, int64_t G);
void dynamic_per_group_quant_f16_u8_avx512fp16(const uint16_t *src, uint8_t *dst,
                                             float *scales, int32_t *zps,
                                             int64_t M, int64_t K, int64_t G);

// Grouped per-token BF16/F32/F16 -> S8 symmetric dynamic quantization.
// Sources are independent [M_i, K_i] matrices; destinations are packed
// [M_i, K_i]. One global row loop schedules across sum(M_i).
void dynamic_per_token_group_quant_bf16_s8_native(
    const std::vector<const void *> &src,
    const std::vector<int> &M,
    const std::vector<int> &K,
    const std::vector<int> &lda,
    const std::vector<void *> &dst,
    const std::vector<int> &dst_lda,
    const std::vector<float *> &scales,
    int num_threads);
void dynamic_per_token_group_quant_f32_s8_native(
    const std::vector<const void *> &src,
    const std::vector<int> &M,
    const std::vector<int> &K,
    const std::vector<int> &lda,
    const std::vector<void *> &dst,
    const std::vector<int> &dst_lda,
    const std::vector<float *> &scales,
    int num_threads);
void dynamic_per_token_group_quant_f16_s8_native(
    const std::vector<const void *> &src,
    const std::vector<int> &M,
    const std::vector<int> &K,
    const std::vector<int> &lda,
    const std::vector<void *> &dst,
    const std::vector<int> &dst_lda,
    const std::vector<float *> &scales,
    int num_threads);

// Grouped per-group (per-K-block) BF16/F32 -> S8 symmetric dynamic
// quantization.  Sources are independent [M_i, K_i] matrices; each expert's
// scale buffer is {M_i, G} (linear index m*G + g).  G is uniform across
// experts; group_size = K_i / G and must divide K_i exactly.  One global
// row loop schedules across sum(M_i); each row iterates its G groups.
void dynamic_per_group_group_quant_bf16_s8_native(
    const std::vector<const void *> &src,
    const std::vector<int> &M,
    const std::vector<int> &K,
    const std::vector<int> &lda,
    const std::vector<void *> &dst,
    const std::vector<int> &dst_lda,
    const std::vector<float *> &scales,
    int64_t G,
    int num_threads);
void dynamic_per_group_group_quant_f32_s8_native(
    const std::vector<const void *> &src,
    const std::vector<int> &M,
    const std::vector<int> &K,
    const std::vector<int> &lda,
    const std::vector<void *> &dst,
    const std::vector<int> &dst_lda,
    const std::vector<float *> &scales,
    int64_t G,
    int num_threads);

// Dynamic dispatch functions (moved from lowoha_reorder.cpp)
bool dispatch_fused_per_token(const void *src, void *dst,
                               const reorder_params_t &params,
                               int64_t M, int64_t N);
bool dispatch_unfused_per_token(const void *src, void *dst,
                                 const reorder_params_t &params,
                                 int64_t M, int64_t N);
bool dispatch_fused_per_group(const void *src, void *dst,
                               const reorder_params_t &params,
                               int64_t M, int64_t K);
bool dispatch_group_dynamic_per_token(
    const std::vector<const void *> &src,
    const std::vector<int> &M,
    const std::vector<int> &K,
    const std::vector<int> &lda,
    const std::vector<void *> &dst,
    const std::vector<int> &dst_lda,
    const std::vector<void *> &scale,
    const group_dynamic_quant_params_t &params);
bool dispatch_group_dynamic_per_group(
    const std::vector<const void *> &src,
    const std::vector<int> &M,
    const std::vector<int> &K,
    const std::vector<int> &lda,
    const std::vector<void *> &dst,
    const std::vector<int> &dst_lda,
    const std::vector<void *> &scale,
    const group_dynamic_quant_params_t &params);

} // namespace reorder
} // namespace lowoha
} // namespace zendnnl

#endif // _DYNAMIC_KERNELS_HPP
