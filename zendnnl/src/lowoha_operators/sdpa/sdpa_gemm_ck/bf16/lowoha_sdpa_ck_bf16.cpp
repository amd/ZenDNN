/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

/// BF16 custom-kernel GEMM for SDPA (GEMM-only mode): VNNI packers and one
/// `vdpbf16ps` tile kernel.
///
/// Layout: B is packed per 32-column block as [pairs][32][2] BF16 (a K pair
/// of each column side by side). A tile is MR x 32 FP32 accumulators fed by
/// broadcast A pairs.

#include "lowoha_operators/sdpa/sdpa_gemm_ck/lowoha_sdpa_ck.hpp"
#include "lowoha_operators/sdpa/sdpa_gemm_ck/lowoha_sdpa_ck_common.hpp"

#include "common/zendnnl_compat.hpp"

#include <immintrin.h>

#include <algorithm>
#include <cstring>
#include <vector>

namespace zendnnl {
namespace lowoha {
namespace sdpa {
namespace ck {
namespace bf16 {

namespace {

/// Fewest rows of A a GEMM call runs on CK; fewer go to AOCL-DLP.
constexpr int64_t kMinRows = 3;

int64_t pairs(int64_t k) {
    return (k + 1) / 2;
}

/// How a tile combines with C.
enum class acc_t {
    store, ///< C = A * B
    add, ///< C += A * B
};

using tile_fn = void (*)(const bfloat16_t *, int64_t, const bfloat16_t *,
        int64_t, float *, int64_t, int);

#define CK_BF16_INLINE \
    ZENDNNL_INLINE_TARGET("avx512f,avx512bf16,avx512bw,avx512vl,fma")
#define CK_BF16_NOINLINE \
    ZENDNNL_TARGET_NOINLINE("avx512f,avx512bf16,avx512bw,avx512vl,fma")

// ---------------------------------------------------------------------------
// Tile kernel.
// ---------------------------------------------------------------------------

/// acc[MR][2] += A[:, 2p : 2p + 2] * packed pair p.
template <int MR>
CK_BF16_INLINE inline void pair_step(__m512 (&acc)[MR][2],
        const bfloat16_t *__restrict a, int64_t lda,
        const bfloat16_t *__restrict b, int64_t p) {
    const __m512bh b0 = (__m512bh)_mm512_loadu_si512(b + p * 2 * kNR);
    const __m512bh b1 = (__m512bh)_mm512_loadu_si512(b + p * 2 * kNR + kNR);
#pragma GCC unroll 8
    for (int m = 0; m < MR; ++m) {
        uint32_t pair;
        std::memcpy(&pair, a + m * lda + 2 * p, sizeof(pair));
        const __m512bh av = (__m512bh)_mm512_set1_epi32(int(pair));
        acc[m][0] = _mm512_dpbf16_ps(acc[m][0], av, b0);
        acc[m][1] = _mm512_dpbf16_ps(acc[m][1], av, b1);
    }
}

/// C[MR][nv] (acc) A[MR][K] * one packed block, nv <= 32.
template <int MR, acc_t Acc>
CK_BF16_NOINLINE void tile(const bfloat16_t *__restrict a, int64_t lda,
        const bfloat16_t *__restrict b, int64_t k, float *__restrict c,
        int64_t ldc, int nv) {
    const __mmask16 m0
            = nv >= 16 ? __mmask16(0xFFFF) : __mmask16((1u << nv) - 1);
    const __mmask16 m1 = nv >= 32 ? __mmask16(0xFFFF)
            : nv > 16             ? __mmask16((1u << (nv - 16)) - 1)
                                  : __mmask16(0);
    __m512 acc[MR][2];
    if constexpr (Acc != acc_t::store) {
#pragma GCC unroll 8
        for (int m = 0; m < MR; ++m) {
            acc[m][0] = _mm512_maskz_loadu_ps(m0, c + m * ldc);
            acc[m][1] = _mm512_maskz_loadu_ps(m1, c + m * ldc + 16);
        }
    } else {
#pragma GCC unroll 8
        for (int m = 0; m < MR; ++m)
            acc[m][0] = acc[m][1] = _mm512_setzero_ps();
    }

    const int64_t full = k / 2;
    int64_t p = 0;
    for (; p + 1 < full; p += 2) {
        pair_step<MR>(acc, a, lda, b, p);
        pair_step<MR>(acc, a, lda, b, p + 1);
    }
    if (p < full) pair_step<MR>(acc, a, lda, b, p);
    if (k & 1) {
        // The pack zeroed the second half of the last pair; A's second
        // element may lie past the row, so only the first is read.
        const __m512bh b0 = (__m512bh)_mm512_loadu_si512(b + full * 2 * kNR);
        const __m512bh b1
                = (__m512bh)_mm512_loadu_si512(b + full * 2 * kNR + kNR);
#pragma GCC unroll 8
        for (int m = 0; m < MR; ++m) {
            uint16_t lo;
            std::memcpy(&lo, a + m * lda + 2 * full, sizeof(lo));
            const __m512bh av = (__m512bh)_mm512_set1_epi32(int(uint32_t(lo)));
            acc[m][0] = _mm512_dpbf16_ps(acc[m][0], av, b0);
            acc[m][1] = _mm512_dpbf16_ps(acc[m][1], av, b1);
        }
    }

    if (nv == kNR) {
#pragma GCC unroll 8
        for (int m = 0; m < MR; ++m) {
            _mm512_storeu_ps(c + m * ldc, acc[m][0]);
            _mm512_storeu_ps(c + m * ldc + 16, acc[m][1]);
        }
        return;
    }
#pragma GCC unroll 8
    for (int m = 0; m < MR; ++m) {
        _mm512_mask_storeu_ps(c + m * ldc, m0, acc[m][0]);
        _mm512_mask_storeu_ps(c + m * ldc + 16, m1, acc[m][1]);
    }
}

/// Tile kernels by row count, for one accumulation kind.
template <acc_t Acc>
constexpr tile_fn kTiles[kMR + 1]
        = {nullptr, tile<1, Acc>, tile<2, Acc>, tile<3, Acc>, tile<4, Acc>,
                tile<5, Acc>, tile<6, Acc>, tile<7, Acc>, tile<8, Acc>};

// ---------------------------------------------------------------------------
// Packers.
// ---------------------------------------------------------------------------

/// K x N row-major B: interleaves rows 2p and 2p+1, writing `kp_out` >=
/// ceil(K/2) pairs per block (pairs past K are zero).
ZENDNNL_TARGET_NOINLINE("avx512f,avx512bw,avx512vl")
void pack_rows(int64_t k, int64_t n, const bfloat16_t *b, int64_t ldb,
        int64_t kp_out, bfloat16_t *out) {
    alignas(64) static const uint16_t kLo[32]
            = {0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39, 8, 40, 9,
                    41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47};
    alignas(64) static const uint16_t kHi[32] = {16, 48, 17, 49, 18, 50, 19, 51,
            20, 52, 21, 53, 22, 54, 23, 55, 24, 56, 25, 57, 26, 58, 27, 59, 28,
            60, 29, 61, 30, 62, 31, 63};
    const __m512i lo = _mm512_load_si512(kLo), hi = _mm512_load_si512(kHi);
    // Row pair outer, so B rows a large stride apart (BSHD V) are read in
    // order across all blocks of one call.
    for (int64_t p = 0; p < kp_out; ++p) {
        const bfloat16_t *r0p = 2 * p < k ? b + 2 * p * ldb : nullptr;
        const bfloat16_t *r1p = 2 * p + 1 < k ? b + (2 * p + 1) * ldb : nullptr;
        for (int64_t o = 0; o * kNR < n; ++o) {
            const int64_t n0 = o * kNR;
            const int nv = int(std::min<int64_t>(kNR, n - n0));
            const __mmask32 mask
                    = nv == 32 ? __mmask32(~0u) : __mmask32((1u << nv) - 1);
            const __m512i r0 = r0p ? _mm512_maskz_loadu_epi16(mask, r0p + n0)
                                   : _mm512_setzero_si512();
            const __m512i r1 = r1p ? _mm512_maskz_loadu_epi16(mask, r1p + n0)
                                   : _mm512_setzero_si512();
            bfloat16_t *dst = out + (o * kp_out + p) * 2 * kNR;
            _mm512_storeu_si512(dst, _mm512_permutex2var_epi16(r0, lo, r1));
            _mm512_storeu_si512(
                    dst + kNR, _mm512_permutex2var_epi16(r0, hi, r1));
        }
    }
}

/// N x K row-major B (op(B) = B^T): each BF16 pair is one dword, so a block
/// is 16 x 16 dword transposes of 16 rows by 16 pairs. Rows past N are zero.
ZENDNNL_TARGET_NOINLINE("avx512f,avx512bw,avx512vl")
void pack_cols(int64_t k, int64_t n, const bfloat16_t *b, int64_t ldb,
        bfloat16_t *out) {
    const int64_t kp = pairs(k);
    for (int64_t o = 0; o * kNR < n; ++o) {
        bfloat16_t *dst = out + o * kp * 2 * kNR;
        for (int half = 0; half < 2; ++half) {
            const int64_t j0 = o * kNR + half * 16;
            const int rows = int(std::clamp<int64_t>(n - j0, 0, 16));
            for (int64_t p0 = 0; p0 < kp; p0 += 16) {
                const int64_t elems = std::min<int64_t>(32, k - 2 * p0);
                const __mmask32 mask = elems >= 32
                        ? __mmask32(~0u)
                        : __mmask32((1u << elems) - 1);
                __m512i r[16], t[16];
                for (int j = 0; j < 16; ++j)
                    r[j] = j < rows ? _mm512_maskz_loadu_epi16(
                                              mask, b + (j0 + j) * ldb + 2 * p0)
                                    : _mm512_setzero_si512();
                for (int i = 0; i < 8; ++i) {
                    t[2 * i] = _mm512_unpacklo_epi32(r[2 * i], r[2 * i + 1]);
                    t[2 * i + 1]
                            = _mm512_unpackhi_epi32(r[2 * i], r[2 * i + 1]);
                }
                for (int i = 0; i < 4; ++i) {
                    r[4 * i] = _mm512_unpacklo_epi64(t[4 * i], t[4 * i + 2]);
                    r[4 * i + 1]
                            = _mm512_unpackhi_epi64(t[4 * i], t[4 * i + 2]);
                    r[4 * i + 2]
                            = _mm512_unpacklo_epi64(t[4 * i + 1], t[4 * i + 3]);
                    r[4 * i + 3]
                            = _mm512_unpackhi_epi64(t[4 * i + 1], t[4 * i + 3]);
                }
                // r[4i + c] holds rows 4i..4i+3 of pair 4L + c in lane L.
                const int64_t np = std::min<int64_t>(16, kp - p0);
                for (int c = 0; c < 4; ++c) {
                    const __m512i x0
                            = _mm512_shuffle_i32x4(r[c], r[4 + c], 0x44);
                    const __m512i x1
                            = _mm512_shuffle_i32x4(r[c], r[4 + c], 0xEE);
                    const __m512i y0
                            = _mm512_shuffle_i32x4(r[8 + c], r[12 + c], 0x44);
                    const __m512i y1
                            = _mm512_shuffle_i32x4(r[8 + c], r[12 + c], 0xEE);
                    const __m512i col[4] = {_mm512_shuffle_i32x4(x0, y0, 0x88),
                            _mm512_shuffle_i32x4(x0, y0, 0xDD),
                            _mm512_shuffle_i32x4(x1, y1, 0x88),
                            _mm512_shuffle_i32x4(x1, y1, 0xDD)};
                    for (int l = 0; l < 4; ++l) {
                        const int64_t pp = 4 * l + c;
                        if (pp < np)
                            _mm512_storeu_si512(
                                    dst + ((p0 + pp) * kNR + half * 16) * 2,
                                    col[l]);
                    }
                }
            }
        }
    }
}

#undef CK_BF16_INLINE
#undef CK_BF16_NOINLINE

} // namespace

// ---------------------------------------------------------------------------
// GEMM.
// ---------------------------------------------------------------------------

size_t packed_size(int64_t k, int64_t n) {
    return static_cast<size_t>((n + kNR - 1) / kNR) * pairs(k) * 2 * kNR;
}

void pack_b(int64_t k, int64_t n, const bfloat16_t *b, int64_t ldb,
        bool trans_b, bfloat16_t *packed) {
    if (trans_b)
        pack_cols(k, n, b, ldb, packed);
    else
        pack_rows(k, n, b, ldb, pairs(k), packed);
}

void gemm_packed(int64_t m, int64_t n, int64_t k, const bfloat16_t *a,
        int64_t lda, const bfloat16_t *packed, bool accumulate, float *c,
        int64_t ldc) {
    const int64_t block = pairs(k) * 2 * kNR;
    for (int64_t o = 0; o * kNR < n; ++o) {
        const int nv = int(std::min<int64_t>(kNR, n - o * kNR));
        const bfloat16_t *bp = packed + o * block;
        float *co = c + o * kNR;
        const tile_fn *fns
                = accumulate ? kTiles<acc_t::add> : kTiles<acc_t::store>;
        int64_t r = 0;
        for (; r + kMR <= m; r += kMR)
            fns[kMR](a + r * lda, lda, bp, k, co + r * ldc, ldc, nv);
        if (r < m) fns[m - r](a + r * lda, lda, bp, k, co + r * ldc, ldc, nv);
    }
}

bool gemm(int64_t m, int64_t n, int64_t k, float alpha, const bfloat16_t *a,
        int64_t lda, const bfloat16_t *b, int64_t ldb, float beta, float *c,
        int64_t ldc, bool trans_a, bool trans_b) {
    if (trans_a || alpha != 1.f || (beta != 0.f && beta != 1.f)) return false;
    // Packing a B block costs more than AOCL-DLP's GEMV for one or two rows.
    if (m > 0 && m < kMinRows) return false;
    if (m <= 0 || n <= 0) return true;
    // One 32-column block at a time: the packed block stays in L1 while every
    // row of A runs over it, and the A, B and C tiles of a flash step stay
    // within L2.
    thread_local std::vector<bfloat16_t> t_block;
    if (!grow(t_block, packed_size(k, kNR))) return false;
    count_bf16_call();
    for (int64_t n0 = 0; n0 < n; n0 += kNR) {
        const int64_t nv = std::min<int64_t>(kNR, n - n0);
        pack_b(k, nv, trans_b ? b + n0 * ldb : b + n0, ldb, trans_b,
                t_block.data());
        gemm_packed(m, nv, k, a, lda, t_block.data(), beta == 1.f, c + n0, ldc);
    }
    return true;
}

} // namespace bf16
} // namespace ck
} // namespace sdpa
} // namespace lowoha
} // namespace zendnnl
