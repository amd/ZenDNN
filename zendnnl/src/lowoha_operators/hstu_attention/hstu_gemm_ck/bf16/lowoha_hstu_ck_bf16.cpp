/********************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lowoha_operators/hstu_attention/hstu_gemm_ck/lowoha_hstu_ck.hpp"

#include <algorithm>

namespace zendnnl {
namespace lowoha {
namespace hstu_attention {
namespace ck {
namespace bf16 {

namespace {

/// Rows of a P*V tile: 8 x 2 accumulators.
constexpr int kMR = 8;

int64_t pairs(int64_t k) {
    return (k + 1) / 2;
}

using tile_fn = void (*)(const bfloat16_t *, int64_t, const bfloat16_t *,
        int64_t, float *, int64_t, int);

/// C[MR][nv] += A[MR][K] * one packed block, nv <= 32.
template <int MR>
HSTU_AVX512_NOINLINE void tile(const bfloat16_t *__restrict a, int64_t lda,
        const bfloat16_t *__restrict b, int64_t k, float *__restrict c,
        int64_t ldc, int nv) {
    const __mmask16 m0
            = nv >= 16 ? __mmask16(0xFFFF) : __mmask16((1u << nv) - 1);
    const __mmask16 m1 = nv >= 32 ? __mmask16(0xFFFF)
            : nv > 16             ? __mmask16((1u << (nv - 16)) - 1)
                                  : __mmask16(0);
    __m512 acc[MR][2];
#pragma GCC unroll 8
    for (int m = 0; m < MR; ++m) {
        acc[m][0] = _mm512_maskz_loadu_ps(m0, c + m * ldc);
        acc[m][1] = _mm512_maskz_loadu_ps(m1, c + m * ldc + 16);
    }
    const int64_t full = k / 2;
    int64_t q = 0;
    for (; q + 1 < full; q += 2) {
        pair_step<MR, 1>(acc, a, lda, b, 0, q, false);
        pair_step<MR, 1>(acc, a, lda, b, 0, q + 1, false);
    }
    if (q < full) pair_step<MR, 1>(acc, a, lda, b, 0, q, false);
    if (k & 1) pair_step<MR, 1>(acc, a, lda, b, 0, full, true);
#pragma GCC unroll 8
    for (int m = 0; m < MR; ++m) {
        _mm512_mask_storeu_ps(c + m * ldc, m0, acc[m][0]);
        _mm512_mask_storeu_ps(c + m * ldc + 16, m1, acc[m][1]);
    }
}

constexpr tile_fn kTiles[kMR + 1] = {nullptr, tile<1>, tile<2>, tile<3>,
        tile<4>, tile<5>, tile<6>, tile<7>, tile<8>};

/// K x N row-major B: interleaves rows 2p and 2p+1 (pairs past K are zero).
ZENDNNL_TARGET_NOINLINE("avx512f,avx512bw,avx512vl")
void pack_rows(int64_t k, int64_t n, const bfloat16_t *b, int64_t ldb,
        bfloat16_t *out) {
    alignas(64) static const uint16_t kLo[32]
            = {0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39, 8, 40, 9,
                    41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47};
    alignas(64) static const uint16_t kHi[32] = {16, 48, 17, 49, 18, 50, 19, 51,
            20, 52, 21, 53, 22, 54, 23, 55, 24, 56, 25, 57, 26, 58, 27, 59, 28,
            60, 29, 61, 30, 62, 31, 63};
    const __m512i lo = _mm512_load_si512(kLo), hi = _mm512_load_si512(kHi);
    const int64_t kp = pairs(k);
    // Row pair outer, so B rows a large stride apart are read in order across
    // all blocks.
    for (int64_t p = 0; p < kp; ++p) {
        const bfloat16_t *r0p = b + 2 * p * ldb;
        const bfloat16_t *r1p = 2 * p + 1 < k ? b + (2 * p + 1) * ldb : nullptr;
        for (int64_t o = 0; o * kNR < n; ++o) {
            const int64_t n0 = o * kNR;
            const int nv = int(std::min<int64_t>(kNR, n - n0));
            const __mmask32 mask
                    = nv == 32 ? __mmask32(~0u) : __mmask32((1u << nv) - 1);
            const __m512i r0 = _mm512_maskz_loadu_epi16(mask, r0p + n0);
            const __m512i r1 = r1p ? _mm512_maskz_loadu_epi16(mask, r1p + n0)
                                   : _mm512_setzero_si512();
            bfloat16_t *dst = out + (o * kp + p) * 2 * kNR;
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

} // namespace

size_t packed_size(int64_t k, int64_t n) {
    return static_cast<size_t>((n + kNR - 1) / kNR) * pairs(k) * 2 * kNR;
}

void pack_b(int64_t k, int64_t n, const bfloat16_t *b, int64_t ldb,
        bool trans_b, bfloat16_t *packed) {
    if (trans_b)
        pack_cols(k, n, b, ldb, packed);
    else
        pack_rows(k, n, b, ldb, packed);
}

void gemm_block(int64_t m, int nv, int64_t k, const bfloat16_t *a, int64_t lda,
        const bfloat16_t *b, float *c, int64_t ldc) {
    int64_t r = 0;
    for (; r + kMR <= m; r += kMR)
        kTiles[kMR](a + r * lda, lda, b, k, c + r * ldc, ldc, nv);
    if (r < m) kTiles[m - r](a + r * lda, lda, b, k, c + r * ldc, ldc, nv);
}

} // namespace bf16
} // namespace ck
} // namespace hstu_attention
} // namespace lowoha
} // namespace zendnnl
