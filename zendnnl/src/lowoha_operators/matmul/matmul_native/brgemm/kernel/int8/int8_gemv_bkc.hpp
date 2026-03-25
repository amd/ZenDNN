/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#pragma once
#include <cstdint>
#include <cstddef>
#include "lowoha_operators/matmul/matmul_native/common/avx512_math.hpp"
#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/bf16/bf16_gemv_bkc.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

// Use INT8_VNNI_GRP from kernel_cache.hpp (canonical constant).

/// INT8 K-contiguous GEMV kernel.
///
/// Computes for M=1:
///   acc[n]    = vpdpbusd(A_u8, B_s8_packed)   (u8 × s8 → i32)
///   result[n] = (float)acc[n] * combined_scale[n] + effective_bias[n]
///   result[n] = postop(result[n])
///   store as BF16 or FP32
///
/// A must be u8. If source is s8, caller adds 128 to each element and
/// adjusts src_zp accordingly (folded into effective_bias).
void int8_gemv_bkc(
    const uint8_t *__restrict__ A,
    const int8_t  *__restrict__ B_kc,
    const float   *__restrict__ combined_scale,
    const float   *__restrict__ effective_bias,
    uint16_t *__restrict__ C_bf16,
    float    *__restrict__ C_fp32,
    fused_postop_t fused_op,
    bool dst_is_bf16,
    int K, int N);

/// Pack s8 weight matrix B into blocked K-contiguous (BKC) INT8 VNNI layout.
/// N is partitioned into blocks via choose_blk_n(), padded to NR_PACK (64).
/// col_sum[n] = sum_k(B[k][n]) for n in [0, N_padded); caller must allocate
/// col_sum with length >= N_padded. Padding entries are zero-initialized.
void pack_b_int8_bkc(
    const int8_t *B, int ldb, int K, int N, bool transB,
    int8_t *packed, int32_t *col_sum);

/// Wide-block dispatch for NP=5,6 (separate CU to avoid i-cache pollution).
void int8_gemv_bkc_wide_dispatch(
    const uint8_t *__restrict__ A,
    const int8_t  *__restrict__ B_bkc,
    const float   *__restrict__ combined_scale,
    const float   *__restrict__ effective_bias,
    uint16_t *__restrict__ C_bf16,
    float    *__restrict__ C_fp32,
    fused_postop_t fused_op,
    bool dst_is_bf16,
    int k_quads, int n_stride, int K, int N,
    int jc, int nb);

/// Precompute combined_scale and effective_bias for static quantization.
///   combined_scale[n] = src_scale * wei_scale[n]
///   effective_bias[n] = bias[n] - src_zp * col_sum[n] * combined_scale[n]
/// wei_scale is per-tensor (wei_scale_count == 1) or per-channel (== N).
void precompute_int8_dequant(
    const int32_t *col_sum,
    const float *bias,         ///< may be nullptr
    float src_scale,
    int32_t src_zp,
    const float *wei_scale,
    int wei_scale_count,       ///< 1 = per-tensor, N = per-channel
    int N, int N_padded,
    float *combined_scale,
    float *effective_bias);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
