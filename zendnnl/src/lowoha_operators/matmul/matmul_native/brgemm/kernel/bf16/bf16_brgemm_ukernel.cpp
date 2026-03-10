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

#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/bf16/bf16_brgemm_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"
#include "lowoha_operators/matmul/matmul_native/common/avx512_math.hpp"

#include <cstdint>
#include <cstring>
#include <algorithm>
#include <immintrin.h>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

template<int MR, int NV>
__attribute__((target("avx512f,avx512bf16,fma"), noinline))
void bf16_brgemm_ukernel(
    const uint16_t *__restrict__ A, int lda,
    const uint16_t *__restrict__ B_vnni, int b_stride,
    float *__restrict__ C, int ldc,
    int K, int BK, float beta,
    const float *__restrict__ bias,
    fused_postop_t fused_op,
    uint16_t *__restrict__ C_bf16, int ldc_bf16) {

    __m512 acc[MR][NV];

    if (beta != 0.0f) {
        __m512 bv = _mm512_set1_ps(beta);
        for (int m = 0; m < MR; ++m)
            for (int v = 0; v < NV; ++v)
                acc[m][v] = _mm512_mul_ps(
                    bv, _mm512_loadu_ps(C + m * ldc + v * 16));
    } else {
        for (int m = 0; m < MR; ++m)
            for (int v = 0; v < NV; ++v)
                acc[m][v] = _mm512_setzero_ps();
    }

    for (int pc = 0; pc < K; pc += BK) {
        const int kb_orig = std::min(BK, K - pc);
        const uint16_t *a_off = A + pc;
        const uint16_t *b_off = B_vnni + (pc / 2) * b_stride;

        const int k_full_pairs = kb_orig / 2;
        const bool has_odd_tail = (kb_orig & 1) != 0;

        int kk = 0;
        for (; kk + 1 < k_full_pairs; kk += 2) {
            for (int u = 0; u < 2; ++u) {
                const uint16_t *b_kp = b_off + (kk + u) * b_stride;
                __m512bh bv[NV];
                for (int v = 0; v < NV; ++v)
                    bv[v] = (__m512bh)_mm512_loadu_si512(
                        b_kp + v * 16 * VNNI_PAIR);
                for (int m = 0; m < MR; ++m) {
                    uint32_t a_pair;
                    std::memcpy(&a_pair, &a_off[m * lda + 2 * (kk + u)],
                                sizeof(a_pair));
                    __m512bh av = (__m512bh)_mm512_set1_epi32(
                        static_cast<int>(a_pair));
                    for (int v = 0; v < NV; ++v)
                        acc[m][v] = _mm512_dpbf16_ps(acc[m][v], av, bv[v]);
                }
            }
        }
        for (; kk < k_full_pairs; ++kk) {
            const uint16_t *b_kp = b_off + kk * b_stride;
            __m512bh bv[NV];
            for (int v = 0; v < NV; ++v)
                bv[v] = (__m512bh)_mm512_loadu_si512(
                    b_kp + v * 16 * VNNI_PAIR);
            for (int m = 0; m < MR; ++m) {
                uint32_t a_pair;
                std::memcpy(&a_pair, &a_off[m * lda + 2 * kk],
                            sizeof(a_pair));
                __m512bh av = (__m512bh)_mm512_set1_epi32(
                    static_cast<int>(a_pair));
                for (int v = 0; v < NV; ++v)
                    acc[m][v] = _mm512_dpbf16_ps(acc[m][v], av, bv[v]);
            }
        }
        if (has_odd_tail) {
            const uint16_t *b_kp = b_off + k_full_pairs * b_stride;
            __m512bh bv[NV];
            for (int v = 0; v < NV; ++v)
                bv[v] = (__m512bh)_mm512_loadu_si512(
                    b_kp + v * 16 * VNNI_PAIR);
            for (int m = 0; m < MR; ++m) {
                uint32_t a_pair = static_cast<uint32_t>(
                    a_off[m * lda + 2 * k_full_pairs]);
                __m512bh av = (__m512bh)_mm512_set1_epi32(
                    static_cast<int>(a_pair));
                for (int v = 0; v < NV; ++v)
                    acc[m][v] = _mm512_dpbf16_ps(acc[m][v], av, bv[v]);
            }
        }
    }

    for (int m = 0; m < MR; ++m) {
        for (int v = 0; v < NV; ++v) {
            __m512 val = acc[m][v];
            if (bias)
                val = _mm512_add_ps(val, _mm512_loadu_ps(bias + v * 16));
            if (fused_op != fused_postop_t::none)
                val = apply_fused_postop(val, fused_op);
            if (C_bf16) {
                __m256bh bf = _mm512_cvtneps_pbh(val);
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i *>(C_bf16 + m * ldc_bf16 + v * 16),
                    (__m256i)bf);
            } else {
                _mm512_storeu_ps(C + m * ldc + v * 16, val);
            }
        }
    }
}

// ── Explicit instantiations ──────────────────────────────────────────────
// Same register pressure analysis as GEMM ukernel (see bf16_gemm_ukernel.cpp).
// BRGEMM has one extra parameter (BK) but identical ZMM usage in K-loop.
#define INST(MR, NV) \
    template void bf16_brgemm_ukernel<MR,NV>( \
        const uint16_t*, int, const uint16_t*, int, float*, int, \
        int, int, float, const float*, fused_postop_t, uint16_t*, int);

// NR=64 (NV=4): MR=1-6
INST(1,4) INST(2,4) INST(3,4) INST(4,4) INST(6,4)
// NR=32 (NV=2): MR=1-6
INST(1,2) INST(2,2) INST(3,2) INST(4,2) INST(6,2)
// NR=16 (NV=1): MR=1-6
INST(1,1) INST(2,1) INST(3,1) INST(4,1) INST(6,1)
#undef INST

using bf16_brgemm_fn_t = void (*)(const uint16_t*, int, const uint16_t*, int,
                                   float*, int, int, int, float, const float*,
                                   fused_postop_t, uint16_t*, int);

__attribute__((target("avx512f,avx512bf16,fma")))
bf16_brgemm_fn_t select_bf16_brgemm_kernel(int MR, int NR) {
    switch (NR) {
    case 64:
        switch (MR) {
        case 1: return bf16_brgemm_ukernel<1, 4>;
        case 2: return bf16_brgemm_ukernel<2, 4>;
        case 3: return bf16_brgemm_ukernel<3, 4>;
        case 4: return bf16_brgemm_ukernel<4, 4>;
        case 6: return bf16_brgemm_ukernel<6, 4>;
        }
        break;
    case 32:
        switch (MR) {
        case 1: return bf16_brgemm_ukernel<1, 2>;
        case 2: return bf16_brgemm_ukernel<2, 2>;
        case 3: return bf16_brgemm_ukernel<3, 2>;
        case 4: return bf16_brgemm_ukernel<4, 2>;
        case 6: return bf16_brgemm_ukernel<6, 2>;
        }
        break;
    case 16:
        switch (MR) {
        case 1: return bf16_brgemm_ukernel<1, 1>;
        case 2: return bf16_brgemm_ukernel<2, 1>;
        case 3: return bf16_brgemm_ukernel<3, 1>;
        case 4: return bf16_brgemm_ukernel<4, 1>;
        case 6: return bf16_brgemm_ukernel<6, 1>;
        }
        break;
    }
    return nullptr;
}

// ============================================================================
// BF16 BRGEMM tail kernel (dynamic MR/NR for edge tiles)
// Same pattern as FP32 BRGEMM tail but with dpbf16ps
// ============================================================================
__attribute__((target("avx512f,avx512bf16,avx512bw,avx512vl,fma")))
void bf16_brgemm_tail_kernel(
    const uint16_t *__restrict__ A, int lda,
    const uint16_t *__restrict__ B_vnni, int b_stride,
    float *__restrict__ C, int ldc,
    int K, int BK, int mr_act, int nr_act, float beta,
    const float *__restrict__ bias, fused_postop_t fused_op,
    uint16_t *__restrict__ C_bf16, int ldc_bf16) {

    const int nv_full = nr_act / 16;
    const int nr_tail = nr_act % 16;
    const int nv = (nr_act + 15) / 16;
    const __mmask16 tail_mask = nr_tail
        ? static_cast<__mmask16>((1u << nr_tail) - 1) : 0;

    __m512 acc[12][4];
    if (beta != 0.0f) {
        __m512 bv = _mm512_set1_ps(beta);
        for (int m = 0; m < mr_act; ++m) {
            for (int v = 0; v < nv_full; ++v)
                acc[m][v] = _mm512_mul_ps(
                    bv, _mm512_loadu_ps(C + m * ldc + v * 16));
            if (nr_tail)
                acc[m][nv_full] = _mm512_mul_ps(
                    bv, _mm512_maskz_loadu_ps(tail_mask,
                        C + m * ldc + nv_full * 16));
        }
    } else {
        for (int m = 0; m < mr_act; ++m)
            for (int v = 0; v < nv; ++v)
                acc[m][v] = _mm512_setzero_ps();
    }

    for (int pc = 0; pc < K; pc += BK) {
        const int kb_orig = std::min(BK, K - pc);
        const uint16_t *a_off = A + pc;
        const uint16_t *b_off = B_vnni + (pc / 2) * b_stride;
        const int k_full_pairs = kb_orig / 2;
        const bool has_odd_tail = (kb_orig & 1) != 0;

        for (int kk = 0; kk < k_full_pairs; ++kk) {
            for (int m = 0; m < mr_act; ++m) {
                uint32_t a_pair;
                std::memcpy(&a_pair, &a_off[m * lda + 2 * kk],
                            sizeof(a_pair));
                __m512bh a_bf16 = (__m512bh)_mm512_set1_epi32(
                    static_cast<int>(a_pair));
                for (int v = 0; v < nv; ++v) {
                    __m512bh b_bf16 = (__m512bh)_mm512_loadu_si512(
                        b_off + kk * b_stride + v * 16 * VNNI_PAIR);
                    acc[m][v] = _mm512_dpbf16_ps(acc[m][v], a_bf16, b_bf16);
                }
            }
        }
        if (has_odd_tail) {
            for (int m = 0; m < mr_act; ++m) {
                uint32_t a_pair = static_cast<uint32_t>(
                    a_off[m * lda + 2 * k_full_pairs]);
                __m512bh a_bf16 = (__m512bh)_mm512_set1_epi32(
                    static_cast<int>(a_pair));
                for (int v = 0; v < nv; ++v) {
                    __m512bh b_bf16 = (__m512bh)_mm512_loadu_si512(
                        b_off + k_full_pairs * b_stride + v * 16 * VNNI_PAIR);
                    acc[m][v] = _mm512_dpbf16_ps(acc[m][v], a_bf16, b_bf16);
                }
            }
        }
    }

    // Epilogue
    for (int m = 0; m < mr_act; ++m) {
        for (int v = 0; v < nv_full; ++v) {
            __m512 val = acc[m][v];
            if (bias) val = _mm512_add_ps(val, _mm512_loadu_ps(bias + v * 16));
            if (fused_op != fused_postop_t::none)
                val = apply_fused_postop(val, fused_op);
            if (C_bf16) {
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i *>(C_bf16 + m * ldc_bf16 + v * 16),
                    (__m256i)_mm512_cvtneps_pbh(val));
            } else {
                _mm512_storeu_ps(C + m * ldc + v * 16, val);
            }
        }
        if (nr_tail) {
            __m512 val = acc[m][nv_full];
            if (bias) val = _mm512_add_ps(val,
                _mm512_maskz_loadu_ps(tail_mask, bias + nv_full * 16));
            if (fused_op != fused_postop_t::none)
                val = apply_fused_postop(val, fused_op);
            if (C_bf16) {
                _mm256_mask_storeu_epi16(
                    C_bf16 + m * ldc_bf16 + nv_full * 16, tail_mask,
                    (__m256i)_mm512_cvtneps_pbh(val));
            } else {
                _mm512_mask_storeu_ps(C + m * ldc + nv_full * 16, tail_mask, val);
            }
        }
    }
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
