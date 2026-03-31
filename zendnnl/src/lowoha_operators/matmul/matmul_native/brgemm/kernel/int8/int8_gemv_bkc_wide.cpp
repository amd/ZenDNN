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

// Separate compilation unit for NP=5,6 INT8 BKC GEMV templates.
// Isolated to prevent i-cache pollution of the NP=1-4 hot path.

#include <cstdint>
#include <cstring>
#include <algorithm>
#include <immintrin.h>
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/int8/int8_gemv_bkc.hpp"
#include "lowoha_operators/matmul/matmul_native/common/avx512_math.hpp"
#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

template<int NP>
__attribute__((noinline, target("avx512f,avx512bf16,avx512bw,avx512vl,avx512vnni,fma")))
static void int8_gemv_bkc_wide_core(
    const uint8_t *__restrict__ A,
    const int8_t  *__restrict__ B_bkc,
    const float   *__restrict__ combined_scale,
    const float   *__restrict__ effective_bias,
    uint16_t *__restrict__ C_bf16,
    float    *__restrict__ C_fp32,
    fused_postop_t fused_op,
    float alpha, float beta,
    bool dst_is_bf16,
    int k_quads, int n_stride, int K, int N, int jc) {

    constexpr int NV = 4;
    constexpr int NR = 64;
    __m512i acc[NP * NV];

    for (int i = 0; i < NP * NV; ++i)
        acc[i] = _mm512_setzero_si512();

    const int k_quads_full = K / 4;
    const int8_t *bp = B_bkc;

    for (int kq = 0; kq < k_quads_full; ++kq) {
        int32_t a_quad;
        std::memcpy(&a_quad, &A[4 * kq], sizeof(a_quad));
        __m512i av = _mm512_set1_epi32(a_quad);

        for (int p = 0; p < NP; ++p) {
            const int8_t *bpp = bp + p * NR * INT8_VNNI_GRP;
            for (int v = 0; v < NV; ++v) {
                __m512i bv = _mm512_loadu_si512(bpp + v * 16 * INT8_VNNI_GRP);
                acc[p * NV + v] = _mm512_dpbusd_epi32(acc[p * NV + v], av, bv);
            }
        }
        bp += n_stride;
    }
    if (K & 3) {
        int32_t a_quad = 0;
        std::memcpy(&a_quad, &A[4 * k_quads_full], K - 4 * k_quads_full);
        __m512i av = _mm512_set1_epi32(a_quad);
        for (int p = 0; p < NP; ++p) {
            const int8_t *bpp = bp + p * NR * INT8_VNNI_GRP;
            for (int v = 0; v < NV; ++v) {
                __m512i bv = _mm512_loadu_si512(bpp + v * 16 * INT8_VNNI_GRP);
                acc[p * NV + v] = _mm512_dpbusd_epi32(acc[p * NV + v], av, bv);
            }
        }
    }

    for (int i = 0; i < NP * NV; ++i) {
        const int n_off = jc + i * 16;
        if (n_off >= N) break;
        const int elems = std::min(16, N - n_off);
        const __mmask16 mask = (elems == 16) ? __mmask16(0xFFFF)
            : static_cast<__mmask16>((1u << elems) - 1);

        __m512 val = _mm512_cvtepi32_ps(acc[i]);
        __m512 cs = (elems == 16)
            ? _mm512_loadu_ps(combined_scale + n_off)
            : _mm512_maskz_loadu_ps(mask, combined_scale + n_off);
        __m512 eb = (elems == 16)
            ? _mm512_loadu_ps(effective_bias + n_off)
            : _mm512_maskz_loadu_ps(mask, effective_bias + n_off);
        val = _mm512_fmadd_ps(val, cs, eb);

        if (alpha != 1.0f)
            val = _mm512_mul_ps(val, _mm512_set1_ps(alpha));

        if (beta != 0.0f) {
            __m512 c_old;
            if (dst_is_bf16 && C_bf16) {
                __m256i raw = (elems == 16)
                    ? _mm256_loadu_si256(reinterpret_cast<const __m256i *>(C_bf16 + n_off))
                    : _mm256_maskz_loadu_epi16(mask, C_bf16 + n_off);
                c_old = _mm512_castsi512_ps(_mm512_slli_epi32(
                    _mm512_cvtepu16_epi32(raw), 16));
            } else if (C_fp32) {
                c_old = (elems == 16)
                    ? _mm512_loadu_ps(C_fp32 + n_off)
                    : _mm512_maskz_loadu_ps(mask, C_fp32 + n_off);
            } else {
                c_old = _mm512_setzero_ps();
            }
            val = _mm512_fmadd_ps(_mm512_set1_ps(beta), c_old, val);
        }

        if (fused_op != fused_postop_t::none)
            val = apply_fused_postop(val, fused_op);

        if (dst_is_bf16) {
            __m256bh bf = _mm512_cvtneps_pbh(val);
            if (elems == 16)
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i *>(C_bf16 + n_off), (__m256i)bf);
            else
                _mm256_mask_storeu_epi16(C_bf16 + n_off, mask, (__m256i)bf);
        } else {
            if (elems == 16)
                _mm512_storeu_ps(C_fp32 + n_off, val);
            else
                _mm512_mask_storeu_ps(C_fp32 + n_off, mask, val);
        }
    }
}

void int8_gemv_bkc_wide_dispatch(
    const uint8_t *__restrict__ A,
    const int8_t  *__restrict__ B_bkc,
    const float   *__restrict__ combined_scale,
    const float   *__restrict__ effective_bias,
    uint16_t *__restrict__ C_bf16,
    float    *__restrict__ C_fp32,
    fused_postop_t fused_op,
    float alpha, float beta,
    bool dst_is_bf16,
    int k_quads, int n_stride, int K, int N,
    int jc, int nb) {

    constexpr int NR = 64;
    const int np = nb / NR;

    switch (np) {
    case 6:
        int8_gemv_bkc_wide_core<6>(
            A, B_bkc, combined_scale, effective_bias,
            C_bf16, C_fp32, fused_op, alpha, beta, dst_is_bf16,
            k_quads, n_stride, K, N, jc);
        break;
    case 5:
        int8_gemv_bkc_wide_core<5>(
            A, B_bkc, combined_scale, effective_bias,
            C_bf16, C_fp32, fused_op, alpha, beta, dst_is_bf16,
            k_quads, n_stride, K, N, jc);
        break;
    default:
        break;
    }
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
