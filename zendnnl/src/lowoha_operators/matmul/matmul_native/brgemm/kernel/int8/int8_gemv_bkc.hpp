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
#include <cstddef>
#include <cstdint>
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/bf16/bf16_gemv_bkc.hpp"
#include "lowoha_operators/matmul/matmul_native/common/avx512_math.hpp"
#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

// Use INT8_VNNI_GRP from kernel_cache.hpp (canonical constant).

/// The dequantization of an INT8 GEMV call, applied per column n:
///   combined_scale[n] = src_scale * wei_scale[n] (wei_scale[0] when not
///                       per_channel)
///   effective_bias[n] = bias[n] - src_zp * col_sum[n] * combined_scale[n]
/// col_sum holds the N_padded column sums of pack_b_int8_bkc; bias (nullptr
/// for none) and a per-channel wei_scale hold N values.
struct int8_gemv_dequant_t {
    const int32_t *col_sum = nullptr;
    const float *bias = nullptr;
    const float *wei_scale = nullptr;
    bool per_channel = false;
    float src_scale = 1.0f;
    int32_t src_zp = 0;
};

/// The values of dq a kernel broadcasts once per call.
struct int8_gemv_dequant_bcast_t {
    __m512 src_scale, src_zp, wei_scale;
};

ZENDNNL_INLINE_TARGET("avx512f")
static inline int8_gemv_dequant_bcast_t int8_gemv_dequant_bcast(
        const int8_gemv_dequant_t &dq) {
    return {_mm512_set1_ps(dq.src_scale),
            _mm512_set1_ps(static_cast<float>(dq.src_zp)),
            dq.per_channel ? _mm512_setzero_ps()
                           : _mm512_set1_ps(dq.wei_scale[0])};
}

/// combined_scale and effective_bias of dq (b its broadcasts) for the 16
/// columns from n; the lanes outside mask (columns at or past N) read no
/// bias or weight scale.  effective_bias takes the product with
/// combined_scale fused into the subtraction (one rounding), an explicit
/// instruction so the bits do not depend on the compiler's contraction.
ZENDNNL_INLINE_TARGET("avx512f")
static inline void int8_gemv_dequant_cols(const int8_gemv_dequant_t &dq,
        const int8_gemv_dequant_bcast_t &b, int n, __mmask16 mask,
        __m512 &combined_scale, __m512 &effective_bias) {
    const __m512 ws = dq.per_channel
            ? _mm512_maskz_loadu_ps(mask, dq.wei_scale + n)
            : b.wei_scale;
    combined_scale = _mm512_mul_ps(b.src_scale, ws);
    const __m512 col_sum
            = _mm512_cvtepi32_ps(_mm512_loadu_si512(dq.col_sum + n));
    const __m512 bias = dq.bias != nullptr
            ? _mm512_maskz_loadu_ps(mask, dq.bias + n)
            : _mm512_setzero_ps();
    effective_bias = _mm512_fnmadd_ps(
            _mm512_mul_ps(b.src_zp, col_sum), combined_scale, bias);
}

/// INT8 K-contiguous GEMV kernel.
///
/// Computes for M=1:
///   acc[n]    = vpdpbusd(A_u8, B_s8_packed)   (u8 × s8 → i32)
///   result[n] = (float)acc[n] * combined_scale[n] + effective_bias[n]
///               (int8_gemv_dequant_cols of dq, per 16 columns)
///   result[n] = postop(result[n])
///   store as BF16 or FP32
///
/// A must be u8. If source is s8, caller adds 128 to each element and
/// adjusts src_zp accordingly (folded into effective_bias).
void int8_gemv_bkc(const uint8_t *__restrict A, const int8_t *__restrict B_kc,
        const int8_gemv_dequant_t &dq, uint16_t *__restrict C_bf16,
        float *__restrict C_fp32, fused_postop_t fused_op, float alpha,
        float beta, bool dst_is_bf16, int K, int N);

/// Pack s8 weight matrix B into blocked K-contiguous (BKC) INT8 VNNI layout.
/// N is partitioned into blocks via choose_blk_n(), padded to BKC_NR_PAD (16).
/// col_sum[n] = sum_k(B[k][n]) for n in [0, N_padded); caller must allocate
/// col_sum with length >= N_padded. Padding entries are zero-initialized.
void pack_b_int8_bkc(const int8_t *B, int ldb, int K, int N, bool transB,
        int8_t *packed, int32_t *col_sum);

/// Wide-block dispatch for NP=5,6 (separate CU to avoid i-cache pollution).
void int8_gemv_bkc_wide_dispatch(const uint8_t *__restrict A,
        const int8_t *__restrict B_bkc, const int8_gemv_dequant_t &dq,
        uint16_t *__restrict C_bf16, float *__restrict C_fp32,
        fused_postop_t fused_op, float alpha, float beta, bool dst_is_bf16,
        int k_quads, int n_stride, int K, int N, int jc, int nb);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
