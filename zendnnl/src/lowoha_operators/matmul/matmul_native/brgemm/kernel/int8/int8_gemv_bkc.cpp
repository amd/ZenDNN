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

#include <algorithm>
#include <cstdint>
#include <cstring>
#include "common/zendnnl_compat.hpp"

#include <immintrin.h>
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/int8/int8_gemv_bkc.hpp"
#include "lowoha_operators/matmul/matmul_native/common/avx512_math.hpp"
#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

// Block width is adaptive via choose_blk_n(N) from bf16_gemv_bkc.hpp:
// 256 for N ≤ 256 or N > 512, 384 for N ∈ (256, 512] with N%64==0.

// ── INT8 Blocked K-contiguous (BKC) VNNI packing ──────────────────────
//
// For vpdpbusd: groups of 4 consecutive K elements per column.
// Layout: packed[block_offset + kq * blk_stride + n_local * 4 + i]
// Simultaneously computes col_sum[n] = sum_k(B[k][n]) for zero-point
// compensation.
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl")
void pack_b_int8_bkc(const int8_t *B, int ldb, int K, int N, bool transB,
        int8_t *packed, int32_t *col_sum) {

    const int blk_n = choose_blk_n(N);
    const int K_padded = (K + 3) & ~3;
    const int k_quads = K_padded / 4;

    std::memset(col_sum, 0,
            static_cast<size_t>(
                    ((N + BKC_NR_PAD - 1) / BKC_NR_PAD) * BKC_NR_PAD)
                    * sizeof(int32_t));

    size_t dst_offset = 0;

    for (int jc = 0; jc < N; jc += blk_n) {
        const int nb = std::min(blk_n, N - jc);
        const int nb_padded = ((nb + BKC_NR_PAD - 1) / BKC_NR_PAD) * BKC_NR_PAD;
        const int blk_stride = nb_padded * INT8_VNNI_GRP;

        for (int kq = 0; kq < k_quads; ++kq) {
            int8_t *dst = packed + dst_offset + kq * blk_stride;
            const int k_base = kq * 4;

            if (!transB) {
                for (int n = 0; n < nb; ++n) {
                    int32_t sum = 0;
                    for (int i = 0; i < 4; ++i) {
                        const int k = k_base + i;
                        int8_t val = (k < K) ? B[k * ldb + (jc + n)] : 0;
                        dst[n * INT8_VNNI_GRP + i] = val;
                        sum += val;
                    }
                    col_sum[jc + n] += sum;
                }
            } else {
                for (int n = 0; n < nb; ++n) {
                    int32_t sum = 0;
                    for (int i = 0; i < 4; ++i) {
                        const int k = k_base + i;
                        int8_t val = (k < K) ? B[(jc + n) * ldb + k] : 0;
                        dst[n * INT8_VNNI_GRP + i] = val;
                        sum += val;
                    }
                    col_sum[jc + n] += sum;
                }
            }
            for (int n = nb; n < nb_padded; ++n) {
                dst[n * INT8_VNNI_GRP + 0] = 0;
                dst[n * INT8_VNNI_GRP + 1] = 0;
                dst[n * INT8_VNNI_GRP + 2] = 0;
                dst[n * INT8_VNNI_GRP + 3] = 0;
            }
        }
        dst_offset += static_cast<size_t>(k_quads) * blk_stride;
    }
}

// ── INT8 BKC GEMV core kernel (templated by panel count) ──────────────
template <int NP>
ZENDNNL_TARGET_NOINLINE("avx512f,avx512bf16,avx512bw,avx512vl,avx512vnni,fma")
static void int8_gemv_bkc_nr64_core(const uint8_t *__restrict A,
        const int8_t *__restrict B_bkc, const int8_gemv_dequant_t &dq,
        uint16_t *__restrict C_bf16, float *__restrict C_fp32,
        fused_postop_t fused_op, float alpha, float beta, bool dst_is_bf16,
        int k_quads, int n_stride, int K, int N, int jc, int b_col_off) {

    constexpr int NV = 4;
    constexpr int NR = 64;
    __m512i acc[NP * NV];

    for (int i = 0; i < NP * NV; ++i)
        acc[i] = _mm512_setzero_si512();

    const int k_quads_full = K / 4;
    const int8_t *bp = B_bkc + b_col_off;

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

    // A copy that the stores to the destination cannot alias, and its
    // broadcasts, made after the K loop, whose accumulators take the
    // registers.
    const int8_gemv_dequant_t d = dq;
    const int8_gemv_dequant_bcast_t db = int8_gemv_dequant_bcast(d);
    for (int i = 0; i < NP * NV; ++i) {
        const int n_off = jc + i * 16;
        if (n_off >= N) break;
        const int elems = std::min(16, N - n_off);
        const __mmask16 mask = (elems == 16)
                ? __mmask16(0xFFFF)
                : static_cast<__mmask16>((1u << elems) - 1);

        // val = acc * combined_scale + effective_bias (dequantize)
        __m512 val = _mm512_cvtepi32_ps(acc[i]);
        __m512 cs, eb;
        int8_gemv_dequant_cols(d, db, n_off, mask, cs, eb);
        val = _mm512_fmadd_ps(val, cs, eb);

        // val = α · val
        if (alpha != 1.0f) val = _mm512_mul_ps(val, _mm512_set1_ps(alpha));

        // val += β · C_old
        if (beta != 0.0f) {
            __m512 c_old;
            if (dst_is_bf16 && C_bf16) {
                __m256i raw = (elems == 16)
                        ? _mm256_loadu_si256(reinterpret_cast<const __m256i *>(
                                  C_bf16 + n_off))
                        : _mm256_maskz_loadu_epi16(mask, C_bf16 + n_off);
                c_old = _mm512_castsi512_ps(
                        _mm512_slli_epi32(_mm512_cvtepu16_epi32(raw), 16));
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
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(C_bf16 + n_off),
                        (__m256i)bf);
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

// ── Flat INT8 GEMV epilogue: dequant + alpha/beta/postop/store ─────────
ZENDNNL_INLINE_TARGET("avx512f,avx512bf16,avx512bw,avx512vl,fma")
static inline void int8_gemv_flat_epilogue(const int32_t *__restrict acc_i32,
        int nvt, const int8_gemv_dequant_t &dq, uint16_t *__restrict C_bf16,
        float *__restrict C_fp32, fused_postop_t fused_op, float alpha,
        float beta, bool dst_is_bf16, int N) {
    // A copy that the stores to the destination cannot alias.
    const int8_gemv_dequant_t d = dq;
    const int8_gemv_dequant_bcast_t db = int8_gemv_dequant_bcast(d);

    for (int v = 0; v < nvt; ++v) {
        const int n_off = v * 16;
        if (n_off >= N) break;
        const int elems = std::min(16, N - n_off);
        const __mmask16 mask = (elems == 16)
                ? __mmask16(0xFFFF)
                : static_cast<__mmask16>((1u << elems) - 1);

        __m512 val = _mm512_cvtepi32_ps(_mm512_load_si512(acc_i32 + v * 16));
        __m512 cs, eb;
        int8_gemv_dequant_cols(d, db, n_off, mask, cs, eb);
        val = _mm512_fmadd_ps(val, cs, eb);

        if (alpha != 1.0f) val = _mm512_mul_ps(val, _mm512_set1_ps(alpha));

        if (beta != 0.0f) {
            __m512 c_old;
            if (dst_is_bf16 && C_bf16) {
                __m256i raw = (elems == 16)
                        ? _mm256_loadu_si256(reinterpret_cast<const __m256i *>(
                                  C_bf16 + n_off))
                        : _mm256_maskz_loadu_epi16(mask, C_bf16 + n_off);
                c_old = _mm512_castsi512_ps(
                        _mm512_slli_epi32(_mm512_cvtepu16_epi32(raw), 16));
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

        if (dst_is_bf16 && C_bf16) {
            __m256bh bf = _mm512_cvtneps_pbh(val);
            if (elems == 16)
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(C_bf16 + n_off),
                        (__m256i)bf);
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

// ── INT8 intrinsics flat K-loop: compile-time unrolled, single K-loop ─
// NVT i32 accumulators live in ZMM registers for the entire K dimension.
template <int NVT>
ZENDNNL_TARGET_NOINLINE("avx512f,avx512bw,avx512vl,avx512vnni")
static void int8_gemv_flat_kloop_intrinsic(const uint8_t *__restrict A,
        const int8_t *__restrict B_bkc, int32_t *__restrict acc, int K,
        int n_stride) {

    __m512i a[NVT];
    for (int v = 0; v < NVT; ++v)
        a[v] = _mm512_setzero_si512();

    const int k_quads_full = K / 4;
    const int8_t *bp = B_bkc;

    for (int kq = 0; kq < k_quads_full; ++kq) {
        int32_t aq;
        std::memcpy(&aq, &A[4 * kq], sizeof(aq));
        __m512i av = _mm512_set1_epi32(aq);
        for (int v = 0; v < NVT; ++v)
            a[v] = _mm512_dpbusd_epi32(
                    a[v], av, _mm512_loadu_si512(bp + v * 16 * INT8_VNNI_GRP));
        bp += n_stride;
    }
    if (K & 3) {
        int32_t aq = 0;
        std::memcpy(&aq, &A[4 * k_quads_full], K - 4 * k_quads_full);
        __m512i av = _mm512_set1_epi32(aq);
        for (int v = 0; v < NVT; ++v)
            a[v] = _mm512_dpbusd_epi32(
                    a[v], av, _mm512_loadu_si512(bp + v * 16 * INT8_VNNI_GRP));
    }

    for (int v = 0; v < NVT; ++v)
        _mm512_store_si512(acc + v * 16, a[v]);
}

ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,avx512vnni")
static bool int8_gemv_flat_intrinsic_dispatch(const uint8_t *A,
        const int8_t *B_bkc, int32_t *acc, int K, int n_stride, int nvt) {

#define CASE_INT8_NVT(N) \
    case N: \
        int8_gemv_flat_kloop_intrinsic<N>(A, B_bkc, acc, K, n_stride); \
        return true

    switch (nvt) {
        CASE_INT8_NVT(1);
        CASE_INT8_NVT(2);
        CASE_INT8_NVT(3);
        CASE_INT8_NVT(4);
        CASE_INT8_NVT(5);
        CASE_INT8_NVT(6);
        CASE_INT8_NVT(7);
        CASE_INT8_NVT(8);
        CASE_INT8_NVT(9);
        CASE_INT8_NVT(10);
        CASE_INT8_NVT(11);
        CASE_INT8_NVT(12);
        CASE_INT8_NVT(13);
        CASE_INT8_NVT(14);
        CASE_INT8_NVT(15);
        CASE_INT8_NVT(16);
        default: return false;
    }
#undef CASE_INT8_NVT
}

// ── Flat INT8 GEMV entry: single K-loop + dequant epilogue ────────────
ZENDNNL_TARGET_NOINLINE("avx512f,avx512bf16,avx512bw,avx512vl,avx512vnni,fma")
static bool int8_gemv_flat(const uint8_t *__restrict A,
        const int8_t *__restrict B_bkc, const int8_gemv_dequant_t &dq,
        uint16_t *__restrict C_bf16, float *__restrict C_fp32,
        fused_postop_t fused_op, float alpha, float beta, bool dst_is_bf16,
        int K, int N) {

    const int N_padded = ((N + BKC_NR_PAD - 1) / BKC_NR_PAD) * BKC_NR_PAD;
    const int nvt = N_padded / 16;
    if (nvt < 1 || nvt > 16) return false;

    const int n_stride = N_padded * INT8_VNNI_GRP;
    alignas(64) int32_t acc[256];

    if (!int8_gemv_flat_intrinsic_dispatch(A, B_bkc, acc, K, n_stride, nvt))
        return false;

    int8_gemv_flat_epilogue(acc, nvt, dq, C_bf16, C_fp32, fused_op, alpha, beta,
            dst_is_bf16, N);
    return true;
}

// ── Block dispatch: full panels + tail ─────────────────────────────────
static inline void int8_dispatch_block(const uint8_t *A, const int8_t *B_bkc,
        const int8_gemv_dequant_t &dq, uint16_t *C_bf16, float *C_fp32,
        fused_postop_t fused_op, float alpha, float beta, bool dst_is_bf16,
        int k_quads, int n_stride, int K, int N, int jc, int nb,
        int b_col_off) {

    constexpr int NR = 64;
    const int np = nb / NR;

#define DISPATCH_INT8_BKC(NP) \
    int8_gemv_bkc_nr64_core<NP>(A, B_bkc, dq, C_bf16, C_fp32, fused_op, alpha, \
            beta, dst_is_bf16, k_quads, n_stride, K, N, jc, b_col_off)

    switch (np) {
        case 4: DISPATCH_INT8_BKC(4); break;
        case 3: DISPATCH_INT8_BKC(3); break;
        case 2: DISPATCH_INT8_BKC(2); break;
        case 1: DISPATCH_INT8_BKC(1); break;
        default: break;
    }
#undef DISPATCH_INT8_BKC

    const int tail_local = np * NR;
    const int tail_global = jc + tail_local;
    if (tail_global < N && tail_local < nb) {
        int8_gemv_bkc_nr64_core<1>(A, B_bkc, dq, C_bf16, C_fp32, fused_op,
                alpha, beta, dst_is_bf16, k_quads, n_stride, K, N, tail_global,
                b_col_off + tail_local * INT8_VNNI_GRP);
    }
}

// ── Public API ─────────────────────────────────────────────────────────

ZENDNNL_NOINLINE
void int8_gemv_bkc(const uint8_t *__restrict A, const int8_t *__restrict B_bkc,
        const int8_gemv_dequant_t &dq, uint16_t *__restrict C_bf16,
        float *__restrict C_fp32, fused_postop_t fused_op, float alpha,
        float beta, bool dst_is_bf16, int K, int N) {

    // Flat path: single K-loop for the entire N when the block dispatch would
    // produce both a main panel AND a tail (N > 64, N not 64-aligned).
    if (N > 64 && (N & 63) != 0 && N <= choose_blk_n(N)) {
        if (int8_gemv_flat(A, B_bkc, dq, C_bf16, C_fp32, fused_op, alpha, beta,
                    dst_is_bf16, K, N))
            return;
    }

    const int blk_n = choose_blk_n(N);
    const int K_padded = (K + 3) & ~3;
    const int k_quads = K_padded / 4;

    size_t b_offset = 0;

    for (int jc = 0; jc < N; jc += blk_n) {
        const int nb = std::min(blk_n, N - jc);
        const int nb_padded = ((nb + BKC_NR_PAD - 1) / BKC_NR_PAD) * BKC_NR_PAD;
        const int blk_n_stride = nb_padded * INT8_VNNI_GRP;

        const int8_t *B_blk = B_bkc + b_offset;

        if (nb > 256)
            int8_gemv_bkc_wide_dispatch(A, B_blk, dq, C_bf16, C_fp32, fused_op,
                    alpha, beta, dst_is_bf16, k_quads, blk_n_stride, K, N, jc,
                    nb);
        else
            int8_dispatch_block(A, B_blk, dq, C_bf16, C_fp32, fused_op, alpha,
                    beta, dst_is_bf16, k_quads, blk_n_stride, K, N, jc, nb,
                    /*b_col_off=*/0);

        b_offset += static_cast<size_t>(k_quads) * blk_n_stride;
    }
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
