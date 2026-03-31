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

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <algorithm>

#include <immintrin.h>
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/bf16/bf16_gemv_bkc.hpp"
#include "lowoha_operators/matmul/matmul_native/common/avx512_math.hpp"
#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

// ── Blocked K-contiguous (BKC) VNNI packing ────────────────────────────
// Packs B into independent column blocks of adaptive width (256 or 384).
// Layout: packed[block_offset + kp * blk_stride + n_local * VNNI_PAIR]
__attribute__((target("avx512f,avx512bw,avx512vl")))
static void pack_b_bkc(
    const uint16_t *B, int ldb, int K, int n_cols, bool transB,
    int col0,
    uint16_t *packed) {

    const int blk_n = choose_blk_n(n_cols);
    const int K_padded = (K + 1) & ~1;
    const int k_pairs = K_padded / 2;

    const __m512i idx_lo = _mm512_setr_epi32(
        0,1,2,3, 16,17,18,19, 4,5,6,7, 20,21,22,23);
    const __m512i idx_hi = _mm512_setr_epi32(
        8,9,10,11, 24,25,26,27, 12,13,14,15, 28,29,30,31);

    size_t dst_offset = 0;

    for (int jblk = 0; jblk < n_cols; jblk += blk_n) {
        const int nb = std::min(blk_n, n_cols - jblk);
        const int nb_padded = ((nb + BKC_NR_PAD - 1) / BKC_NR_PAD) * BKC_NR_PAD;
        const int blk_stride = nb_padded * VNNI_PAIR;

        for (int kp = 0; kp < k_pairs; ++kp) {
            const int k0 = kp * 2;
            const int k1 = k0 + 1;
            uint16_t *dst = packed + dst_offset + kp * blk_stride;

            if (!transB) {
                const int g0 = col0 + jblk;
                const uint16_t *row0 = (k0 < K) ? B + k0 * ldb + g0 : nullptr;
                const uint16_t *row1 = (k1 < K) ? B + k1 * ldb + g0 : nullptr;

                int n = 0;
                for (; n + 31 < nb; n += 32) {
                    __m512i r0 = row0 ? _mm512_loadu_si512(row0 + n)
                                      : _mm512_setzero_si512();
                    __m512i r1 = row1 ? _mm512_loadu_si512(row1 + n)
                                      : _mm512_setzero_si512();
                    __m512i lo = _mm512_unpacklo_epi16(r0, r1);
                    __m512i hi = _mm512_unpackhi_epi16(r0, r1);
                    _mm512_storeu_si512(dst + n * VNNI_PAIR,
                        _mm512_permutex2var_epi32(lo, idx_lo, hi));
                    _mm512_storeu_si512(dst + (n + 16) * VNNI_PAIR,
                        _mm512_permutex2var_epi32(lo, idx_hi, hi));
                }
                for (; n < nb; ++n) {
                    dst[n * VNNI_PAIR + 0] = (row0 && k0 < K) ? row0[n] : 0;
                    dst[n * VNNI_PAIR + 1] = (row1 && k1 < K) ? row1[n] : 0;
                }
            } else {
                for (int n = 0; n < nb; ++n) {
                    const int gc = col0 + jblk + n;
                    dst[n * VNNI_PAIR + 0] = (k0 < K) ? B[gc * ldb + k0] : 0;
                    dst[n * VNNI_PAIR + 1] = (k1 < K) ? B[gc * ldb + k1] : 0;
                }
            }
            for (int n = nb; n < nb_padded; ++n) {
                dst[n * VNNI_PAIR + 0] = 0;
                dst[n * VNNI_PAIR + 1] = 0;
            }
        }
        dst_offset += static_cast<size_t>(k_pairs) * blk_stride;
    }
}

// ── Core NR=64 GEMV kernel ─────────────────────────────────────────────
// Templated on NP (panel count): compile-time unrolling keeps all
// accumulators in ZMM registers.  b_col_off is 0 for block-aware layout.
template<int NP>
__attribute__((noinline, target("avx512f,avx512bf16,avx512bw,avx512vl,fma")))
static void bf16_gemv_bkc_nr64_core(
    const uint16_t *__restrict__ A,
    const uint16_t *__restrict__ B_bkc,
    uint16_t *__restrict__ C_bf16,
    float *__restrict__ C_fp32,
    const float *__restrict__ bias_f,
    fused_postop_t fused_op,
    float alpha, float beta,
    bool dst_is_bf16,
    int k_pairs, int n_stride, int K, int N, int jc, int b_col_off) {

    constexpr int NV = 4;
    constexpr int NR = 64;
    __m512 acc[NP * NV];

    for (int i = 0; i < NP * NV; ++i)
        acc[i] = _mm512_setzero_ps();

    const int k_pairs_even = K / 2;

    for (int kp = 0; kp < k_pairs_even; ++kp) {
        int32_t a_pair;
        std::memcpy(&a_pair, &A[2 * kp], sizeof(a_pair));
        __m512bh av = (__m512bh)_mm512_set1_epi32(a_pair);
        const uint16_t *bp = B_bkc + kp * n_stride + b_col_off;
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
        const uint16_t *bp = B_bkc + k_pairs_even * n_stride + b_col_off;
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
        const __mmask16 mask = (elems == 16) ? __mmask16(0xFFFF)
            : static_cast<__mmask16>((1u << elems) - 1);

        // val = α · dot_product
        __m512 val = (alpha != 1.0f)
            ? _mm512_mul_ps(acc[i], _mm512_set1_ps(alpha)) : acc[i];

        // val += β · C_old
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

        // val += bias
        if (bias_f)
            val = _mm512_add_ps(val, (elems == 16)
                ? _mm512_loadu_ps(bias_f + n_off)
                : _mm512_maskz_loadu_ps(mask, bias_f + n_off));

        if (fused_op != fused_postop_t::none)
            val = apply_fused_postop(val, fused_op);

        if (dst_is_bf16 && C_bf16) {
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

// ── Narrow-tail kernel ─────────────────────────────────────────────────
// Processes exactly NVT vectors (NVT×16 columns) for the last partial panel.
template<int NVT>
__attribute__((noinline, target("avx512f,avx512bf16,avx512bw,avx512vl,fma")))
static void bf16_gemv_bkc_tail(
    const uint16_t *__restrict__ A,
    const uint16_t *__restrict__ B_bkc,
    uint16_t *__restrict__ C_bf16,
    float *__restrict__ C_fp32,
    const float *__restrict__ bias_f,
    fused_postop_t fused_op,
    float alpha, float beta,
    bool dst_is_bf16,
    int k_pairs, int n_stride, int K, int N, int jc, int b_col_off) {

    __m512 acc[NVT];

    for (int i = 0; i < NVT; ++i)
        acc[i] = _mm512_setzero_ps();

    const int k_pairs_even = K / 2;

    for (int kp = 0; kp < k_pairs_even; ++kp) {
        int32_t a_pair;
        std::memcpy(&a_pair, &A[2 * kp], sizeof(a_pair));
        __m512bh av = (__m512bh)_mm512_set1_epi32(a_pair);
        const uint16_t *bpp = B_bkc + kp * n_stride + b_col_off;
        for (int v = 0; v < NVT; ++v) {
            __m512bh bv = (__m512bh)_mm512_loadu_si512(
                bpp + v * 16 * VNNI_PAIR);
            acc[v] = _mm512_dpbf16_ps(acc[v], av, bv);
        }
    }
    if (K & 1) {
        __m512bh av = (__m512bh)_mm512_set1_epi32(
            static_cast<int32_t>(static_cast<uint32_t>(A[K - 1])));
        const uint16_t *bpp = B_bkc + k_pairs_even * n_stride + b_col_off;
        for (int v = 0; v < NVT; ++v) {
            __m512bh bv = (__m512bh)_mm512_loadu_si512(
                bpp + v * 16 * VNNI_PAIR);
            acc[v] = _mm512_dpbf16_ps(acc[v], av, bv);
        }
    }

    for (int i = 0; i < NVT; ++i) {
        const int n_off = jc + i * 16;
        if (n_off >= N) break;
        const int elems = std::min(16, N - n_off);
        const __mmask16 mask = (elems == 16) ? __mmask16(0xFFFF)
            : static_cast<__mmask16>((1u << elems) - 1);

        __m512 val = (alpha != 1.0f)
            ? _mm512_mul_ps(acc[i], _mm512_set1_ps(alpha)) : acc[i];

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

        if (bias_f)
            val = _mm512_add_ps(val, (elems == 16)
                ? _mm512_loadu_ps(bias_f + n_off)
                : _mm512_maskz_loadu_ps(mask, bias_f + n_off));

        if (fused_op != fused_postop_t::none)
            val = apply_fused_postop(val, fused_op);
        if (dst_is_bf16 && C_bf16) {
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

// ── Block dispatch: full panels + optional narrow tail ─────────────────
static inline void dispatch_block(
    const uint16_t *A, const uint16_t *B_bkc,
    uint16_t *C_bf16, float *C_fp32,
    const float *bias_f, fused_postop_t fused_op,
    float alpha, float beta, bool dst_is_bf16,
    int k_pairs, int n_stride, int K, int N,
    int jc, int nb, int b_col_off) {

    constexpr int NR = 64;
    const int np = nb / NR;

    #define DISPATCH_BKC(NP) bf16_gemv_bkc_nr64_core<NP>( \
        A, B_bkc, C_bf16, C_fp32, bias_f, fused_op, alpha, beta, dst_is_bf16, \
        k_pairs, n_stride, K, N, jc, b_col_off)

    switch (np) {
    case 4: DISPATCH_BKC(4); break;
    case 3: DISPATCH_BKC(3); break;
    case 2: DISPATCH_BKC(2); break;
    case 1: DISPATCH_BKC(1); break;
    default: break;
    }
    #undef DISPATCH_BKC

    const int tail_local = np * NR;
    const int tail_global = jc + tail_local;
    if (tail_global < N && tail_local < nb) {
        const int nvt = (N - tail_global + 15) / 16;
        const int tail_b_off = b_col_off + tail_local * VNNI_PAIR;

        #define DISPATCH_TAIL(NVT) bf16_gemv_bkc_tail<NVT>( \
            A, B_bkc, C_bf16, C_fp32, bias_f, fused_op, alpha, beta, dst_is_bf16, \
            k_pairs, n_stride, K, N, tail_global, tail_b_off)

        switch (nvt) {
        case 1: DISPATCH_TAIL(1); break;
        case 2: DISPATCH_TAIL(2); break;
        case 3: DISPATCH_TAIL(3); break;
        default: DISPATCH_TAIL(4); break;
        }
        #undef DISPATCH_TAIL
    }
}

// ── Public API ─────────────────────────────────────────────────────────

__attribute__((noinline))
static void bf16_gemv_bkc_jc_range(
    const uint16_t *__restrict__ A,
    const uint16_t *__restrict__ B_bkc,
    uint16_t *__restrict__ C_bf16,
    float *__restrict__ C_fp32,
    const float *__restrict__ bias_f,
    fused_postop_t fused_op,
    float alpha, float beta,
    bool dst_is_bf16,
    int K, int N,
    int jc_begin, int jc_end) {

    if (jc_begin >= jc_end || jc_begin < 0 || jc_end > N)
        return;

    const int blk_n = choose_blk_n(N);
    const int K_padded = (K + 1) & ~1;
    const int k_pairs = K_padded / 2;

    size_t b_offset = 0;
    for (int jc = 0; jc < jc_begin; jc += blk_n) {
        const int nb = std::min(blk_n, N - jc);
        const int nb_padded = ((nb + BKC_NR_PAD - 1) / BKC_NR_PAD) * BKC_NR_PAD;
        const int blk_n_stride = nb_padded * VNNI_PAIR;
        b_offset += static_cast<size_t>(k_pairs) * blk_n_stride;
    }

    for (int jc = jc_begin; jc < jc_end; jc += blk_n) {
        const int nb = std::min(blk_n, N - jc);
        const int nb_padded = ((nb + BKC_NR_PAD - 1) / BKC_NR_PAD) * BKC_NR_PAD;
        const int blk_n_stride = nb_padded * VNNI_PAIR;

        const uint16_t *B_blk = B_bkc + b_offset;

        if (nb > 256)
            bf16_gemv_bkc_wide_dispatch(
                A, B_blk, C_bf16, C_fp32, bias_f, fused_op,
                alpha, beta, dst_is_bf16, k_pairs, blk_n_stride, K, N,
                jc, nb);
        else
            dispatch_block(A, B_blk, C_bf16, C_fp32, bias_f, fused_op,
                           alpha, beta, dst_is_bf16, k_pairs, blk_n_stride, K, N,
                           jc, nb, /*b_col_off=*/0);

        b_offset += static_cast<size_t>(k_pairs) * blk_n_stride;
    }
}

__attribute__((noinline))
void bf16_gemv_bkc(
    const uint16_t *__restrict__ A,
    const uint16_t *__restrict__ B_bkc,
    uint16_t *__restrict__ C_bf16,
    float *__restrict__ C_fp32,
    const float *__restrict__ bias_f,
    fused_postop_t fused_op,
    float alpha, float beta,
    bool dst_is_bf16,
    int K, int N) {

    bf16_gemv_bkc_jc_range(
        A, B_bkc, C_bf16, C_fp32, bias_f, fused_op,
        alpha, beta, dst_is_bf16, K, N, 0, N);
}

void pack_b_bkc_ext(
    const uint16_t *B, int ldb, int K, int N, bool transB,
    uint16_t *packed,
    int col0) {
    pack_b_bkc(B, ldb, K, N, transB, col0, packed);
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
