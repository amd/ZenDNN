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

// ── Separate compilation unit for NP=5,6 BKC GEMV templates ────────────
//
// NP=5,6 templates are isolated here to prevent their machine code from
// polluting the L1 i-cache when only NP=1-4 are executing (the common case
// for N ≤ 256). Each NP template generates ~600 bytes of hot-loop code;
// keeping NP=5,6 in the main CU would add ~1.2KB to the instruction
// footprint, evicting frequently-used NP=4 code.

#include <cstdint>
#include <cstring>
#include <algorithm>
#include <immintrin.h>
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/bf16/bf16_gemv_bkc.hpp"
#include "lowoha_operators/matmul/matmul_native/common/avx512_math.hpp"
#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

template<int NP>
__attribute__((noinline, target("avx512f,avx512bf16,avx512bw,avx512vl,fma")))
static void bf16_gemv_bkc_wide_core(
    const uint16_t *__restrict__ A,
    const uint16_t *__restrict__ B_bkc,
    uint16_t *__restrict__ C_bf16,
    float *__restrict__ C_fp32,
    const float *__restrict__ bias_f,
    fused_postop_t fused_op,
    float beta,
    bool dst_is_bf16,
    int k_pairs, int n_stride, int K, int N, int jc) {

    constexpr int NV = 4;
    constexpr int NR = 64;
    __m512 acc[NP * NV];

    if (beta != 0.0f && dst_is_bf16 && C_bf16) {
        __m512 bv = _mm512_set1_ps(beta);
        for (int i = 0; i < NP * NV; ++i) {
            const int n_off = jc + i * 16;
            if (n_off >= N) { acc[i] = _mm512_setzero_ps(); continue; }
            const int elems = std::min(16, N - n_off);
            __m256i raw = (elems == 16)
                ? _mm256_loadu_si256(reinterpret_cast<const __m256i *>(C_bf16 + n_off))
                : _mm256_maskz_loadu_epi16(
                    static_cast<__mmask16>((1u << elems) - 1), C_bf16 + n_off);
            __m512 fp = _mm512_castsi512_ps(_mm512_slli_epi32(
                _mm512_cvtepu16_epi32(raw), 16));
            acc[i] = _mm512_mul_ps(bv, fp);
        }
    } else if (beta != 0.0f && C_fp32) {
        __m512 bv = _mm512_set1_ps(beta);
        for (int i = 0; i < NP * NV; ++i) {
            const int n_off = jc + i * 16;
            if (n_off >= N) { acc[i] = _mm512_setzero_ps(); continue; }
            const int elems = std::min(16, N - n_off);
            acc[i] = (elems == 16)
                ? _mm512_mul_ps(bv, _mm512_loadu_ps(C_fp32 + n_off))
                : _mm512_mul_ps(bv, _mm512_maskz_loadu_ps(
                    static_cast<__mmask16>((1u << elems) - 1), C_fp32 + n_off));
        }
    } else {
        for (int i = 0; i < NP * NV; ++i)
            acc[i] = _mm512_setzero_ps();
    }

    const int k_pairs_even = K / 2;

    for (int kp = 0; kp < k_pairs_even; ++kp) {
        int32_t a_pair;
        std::memcpy(&a_pair, &A[2 * kp], sizeof(a_pair));
        __m512bh av = (__m512bh)_mm512_set1_epi32(a_pair);
        const uint16_t *bp = B_bkc + kp * n_stride;
        for (int p = 0; p < NP; ++p) {
            const uint16_t *bpp = bp + p * NR * VNNI_PAIR;
            for (int v = 0; v < NV; ++v) {
                __m512bh bv = (__m512bh)_mm512_loadu_si512(
                    bpp + v * 16 * VNNI_PAIR);
                acc[p * NV + v] = _mm512_dpbf16_ps(
                    acc[p * NV + v], av, bv);
            }
        }
    }
    if (K & 1) {
        __m512bh av = (__m512bh)_mm512_set1_epi32(
            static_cast<int32_t>(static_cast<uint32_t>(A[K - 1])));
        const uint16_t *bp = B_bkc + k_pairs_even * n_stride;
        for (int p = 0; p < NP; ++p) {
            const uint16_t *bpp = bp + p * NR * VNNI_PAIR;
            for (int v = 0; v < NV; ++v) {
                __m512bh bv = (__m512bh)_mm512_loadu_si512(
                    bpp + v * 16 * VNNI_PAIR);
                acc[p * NV + v] = _mm512_dpbf16_ps(
                    acc[p * NV + v], av, bv);
            }
        }
    }

    for (int i = 0; i < NP * NV; ++i) {
        const int n_off = jc + i * 16;
        if (n_off >= N) break;
        const int elems = std::min(16, N - n_off);
        __m512 val = acc[i];

        if (bias_f) {
            if (elems == 16)
                val = _mm512_add_ps(val, _mm512_loadu_ps(bias_f + n_off));
            else
                val = _mm512_add_ps(val, _mm512_maskz_loadu_ps(
                    static_cast<__mmask16>((1u << elems) - 1), bias_f + n_off));
        }
        if (fused_op != fused_postop_t::none)
            val = apply_fused_postop(val, fused_op);

        if (dst_is_bf16) {
            __m256bh bf = _mm512_cvtneps_pbh(val);
            if (elems == 16)
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i *>(C_bf16 + n_off), (__m256i)bf);
            else
                _mm256_mask_storeu_epi16(C_bf16 + n_off,
                    static_cast<__mmask16>((1u << elems) - 1), (__m256i)bf);
        } else {
            if (elems == 16)
                _mm512_storeu_ps(C_fp32 + n_off, val);
            else
                _mm512_mask_storeu_ps(C_fp32 + n_off,
                    static_cast<__mmask16>((1u << elems) - 1), val);
        }
    }
}

void bf16_gemv_bkc_wide_dispatch(
    const uint16_t *__restrict__ A,
    const uint16_t *__restrict__ B_bkc,
    uint16_t *__restrict__ C_bf16,
    float *__restrict__ C_fp32,
    const float *__restrict__ bias_f,
    fused_postop_t fused_op,
    float beta,
    bool dst_is_bf16,
    int k_pairs, int n_stride, int K, int N,
    int jc, int nb) {

    constexpr int NR = 64;
    const int np = nb / NR;

    switch (np) {
    case 6:
        bf16_gemv_bkc_wide_core<6>(
            A, B_bkc, C_bf16, C_fp32, bias_f, fused_op,
            beta, dst_is_bf16, k_pairs, n_stride, K, N, jc);
        break;
    case 5:
        bf16_gemv_bkc_wide_core<5>(
            A, B_bkc, C_bf16, C_fp32, bias_f, fused_op,
            beta, dst_is_bf16, k_pairs, n_stride, K, N, jc);
        break;
    default:
        break;
    }
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
