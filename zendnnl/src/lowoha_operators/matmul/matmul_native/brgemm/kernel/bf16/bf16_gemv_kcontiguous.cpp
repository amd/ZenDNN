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
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/bf16/bf16_gemv_kcontiguous.hpp"
#include "lowoha_operators/matmul/matmul_native/common/avx512_math.hpp"
#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

// ── K-contiguous VNNI packing ──────────────────────────────────────────
// Packs B into K-contiguous VNNI layout. Supports both row-major (!transB)
// and column-major (transB) source layouts.
//   !transB: B[K×N] row-major, element (k,n) = B[k * ldb + n]
//   transB:  B[N×K] col-major, element (k,n) = B[n * ldb + k]
// Output: for each k-pair, all N columns are contiguous as VNNI pairs,
// padded to NR=64 boundary.
__attribute__((target("avx512f,avx512bw,avx512vl")))
static void pack_b_kcontiguous(
    const uint16_t *B, int ldb, int K, int N, bool transB,
    uint16_t *packed) {

    const int K_padded = (K + 1) & ~1;
    const int k_pairs = K_padded / 2;
    const int N_padded = ((N + NR_PACK - 1) / NR_PACK) * NR_PACK;
    const int n_stride = N_padded * VNNI_PAIR;

    const __m512i idx_lo = _mm512_setr_epi32(
        0,1,2,3, 16,17,18,19, 4,5,6,7, 20,21,22,23);
    const __m512i idx_hi = _mm512_setr_epi32(
        8,9,10,11, 24,25,26,27, 12,13,14,15, 28,29,30,31);

    for (int kp = 0; kp < k_pairs; ++kp) {
        const int k0 = kp * 2;
        const int k1 = k0 + 1;
        uint16_t *dst = packed + kp * n_stride;

        if (!transB) {
            const uint16_t *row0 = (k0 < K) ? B + k0 * ldb : nullptr;
            const uint16_t *row1 = (k1 < K) ? B + k1 * ldb : nullptr;

            int n = 0;
            for (; n + 31 < N; n += 32) {
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
            for (; n < N; ++n) {
                dst[n * VNNI_PAIR + 0] = (row0 && k0 < K) ? row0[n] : 0;
                dst[n * VNNI_PAIR + 1] = (row1 && k1 < K) ? row1[n] : 0;
            }
        } else {
            // transB: B is N×K, element (k,n) = B[n * ldb + k]
            for (int n = 0; n < N; ++n) {
                dst[n * VNNI_PAIR + 0] = (k0 < K) ? B[n * ldb + k0] : 0;
                dst[n * VNNI_PAIR + 1] = (k1 < K) ? B[n * ldb + k1] : 0;
            }
        }
        // Zero-pad to N_padded
        for (int n = N; n < N_padded; ++n) {
            dst[n * VNNI_PAIR + 0] = 0;
            dst[n * VNNI_PAIR + 1] = 0;
        }
    }
}

// Templated K-contiguous NR=64 GEMV: compile-time panel count eliminates
// register spills. The compiler fully unrolls the N-inner loop and keeps
// all accumulators in ZMM registers (no stack spills).
template<int NP>  // NP = number of NR=64 panels (compile-time)
__attribute__((noinline, target("avx512f,avx512bf16,avx512bw,avx512vl,fma")))
static void bf16_gemv_kc_nr64_core(
    const uint16_t *__restrict__ A,
    const uint16_t *__restrict__ B_kc,
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

    for (int kp = 0; kp < k_pairs; ++kp) {
        // Safe A load: for odd K, last k-pair has only one valid element.
        // Use single-element load to avoid out-of-bounds A[K] read.
        uint32_t a_pair;
        if (2 * kp + 1 < K)
            std::memcpy(&a_pair, &A[2 * kp], sizeof(a_pair));
        else
            a_pair = static_cast<uint32_t>(A[2 * kp]);
        int32_t a_bits;
        std::memcpy(&a_bits, &a_pair, sizeof(a_bits));
        __m512bh av = (__m512bh)_mm512_set1_epi32(a_bits);

        const uint16_t *bp = B_kc + kp * n_stride + jc * VNNI_PAIR;
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

    // Epilogue: bias + fused activation + store (with tail masking)
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

// Dispatch to the right template instantiation based on panel count.
// Handles BF16 or FP32 output, beta, bias, and fused activation post-ops.
__attribute__((noinline))
void bf16_gemv_kcontiguous(
    const uint16_t *__restrict__ A,
    const uint16_t *__restrict__ B_kc,
    uint16_t *__restrict__ C_bf16,
    float *__restrict__ C_fp32,
    const float *__restrict__ bias_f,
    fused_postop_t fused_op,
    float beta,
    bool dst_is_bf16,
    int K, int N) {

    const int K_padded = (K + 1) & ~1;
    const int k_pairs = K_padded / 2;
    const int N_padded = ((N + NR_PACK - 1) / NR_PACK) * NR_PACK;
    const int n_stride = N_padded * VNNI_PAIR;
    constexpr int NR = 64;
    constexpr int MAX_NP = 4;
    const int NB = MAX_NP * NR;

    #define DISPATCH_KC(NP) bf16_gemv_kc_nr64_core<NP>( \
        A, B_kc, C_bf16, C_fp32, bias_f, fused_op, beta, dst_is_bf16, \
        k_pairs, n_stride, K, N, jc)

    for (int jc = 0; jc < N; jc += NB) {
        const int nb = std::min(NB, N - jc);
        const int np = nb / NR;

        switch (np) {
        case 4: DISPATCH_KC(4); break;
        case 3: DISPATCH_KC(3); break;
        case 2: DISPATCH_KC(2); break;
        case 1: DISPATCH_KC(1); break;
        default: break;
        }

        // Vectorized tail: N%64 remaining columns. B is zero-padded to N_padded,
        // so dispatch NP=1 at tail_start offset. The epilogue's masked stores
        // and `if (n_off >= N) break` handle the N boundary correctly.
        const int tail_start = jc + np * NR;
        if (tail_start < N && tail_start < jc + nb) {
            bf16_gemv_kc_nr64_core<1>(
                A, B_kc, C_bf16, C_fp32, bias_f, fused_op, beta, dst_is_bf16,
                k_pairs, n_stride, K, N, tail_start);
        }
    }
    #undef DISPATCH_KC
}

void pack_b_kcontiguous_ext(
    const uint16_t *B, int ldb, int K, int N, bool transB,
    uint16_t *packed) {
    pack_b_kcontiguous(B, ldb, K, N, transB, packed);
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
