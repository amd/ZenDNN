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
namespace int8 {

namespace {

int64_t quads(int64_t k) {
    return (k + 3) / 4;
}

/// Bytes of one packed 32-column block over K.
int64_t block_bytes(int64_t k) {
    return quads(k) * 4 * kNR;
}

/// Rows of a P*V tile: 8 x 2 accumulators.
constexpr int kMR = 8;

/// How a tile turns its int32 sums into C: C += (acc - comp[n]) *
/// row_scale[m] * col_scale[n].
struct epilogue_t {
    const int32_t *comp;
    const float *row_scale;
    const float *col_scale;
};

using tile_fn = void (*)(const uint8_t *, int64_t, const int8_t *, int64_t,
        float *, int64_t, int, const epilogue_t &);

/// C[MR][nv] += epilogue(A[MR][K] * one packed block), nv <= 32.
template <int MR>
HSTU_I8_NOINLINE void tile(const uint8_t *__restrict a, int64_t lda,
        const int8_t *__restrict b, int64_t k, float *__restrict c, int64_t ldc,
        int nv, const epilogue_t &ep) {
    const __mmask16 m0
            = nv >= 16 ? __mmask16(0xFFFF) : __mmask16((1u << nv) - 1);
    const __mmask16 m1 = nv >= 32 ? __mmask16(0xFFFF)
            : nv > 16             ? __mmask16((1u << (nv - 16)) - 1)
                                  : __mmask16(0);
    __m512i acc[MR][2];
#pragma GCC unroll 8
    for (int m = 0; m < MR; ++m)
        acc[m][0] = acc[m][1] = _mm512_setzero_si512();
    const int64_t full = k / 4;
    int64_t q = 0;
    for (; q + 1 < full; q += 2) {
        quad_step<MR, 1>(acc, a, lda, b, 0, q);
        quad_step<MR, 1>(acc, a, lda, b, 0, q + 1);
    }
    if (q < full) quad_step<MR, 1>(acc, a, lda, b, 0, q);
    if (const int rem = int(k & 3)) {
        // K ends at the block's last key (see gemm_block), so the pack
        // zeroed B past K and the A bytes there only need to be in bounds:
        // copy the `rem` that are.
        alignas(64) uint8_t tail[MR][4] = {};
        for (int m = 0; m < MR; ++m)
            std::memcpy(tail[m], a + m * lda + 4 * full, size_t(rem));
        quad_step<MR, 1>(acc, &tail[0][0], 4, b + full * 4 * kNR, 0, 0);
    }
    const __m512i comp0 = _mm512_maskz_loadu_epi32(m0, ep.comp);
    const __m512i comp1 = _mm512_maskz_loadu_epi32(m1, ep.comp + 16);
    const __m512 cs0 = _mm512_maskz_loadu_ps(m0, ep.col_scale);
    const __m512 cs1 = _mm512_maskz_loadu_ps(m1, ep.col_scale + 16);
#pragma GCC unroll 8
    for (int m = 0; m < MR; ++m) {
        const __m512 rs = _mm512_set1_ps(ep.row_scale[m]);
        __m512 f0 = _mm512_mul_ps(
                _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[m][0], comp0)),
                _mm512_mul_ps(rs, cs0));
        __m512 f1 = _mm512_mul_ps(
                _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[m][1], comp1)),
                _mm512_mul_ps(rs, cs1));
        f0 = _mm512_add_ps(f0, _mm512_maskz_loadu_ps(m0, c + m * ldc));
        f1 = _mm512_add_ps(f1, _mm512_maskz_loadu_ps(m1, c + m * ldc + 16));
        _mm512_mask_storeu_ps(c + m * ldc, m0, f0);
        _mm512_mask_storeu_ps(c + m * ldc + 16, m1, f1);
    }
}

constexpr tile_fn kTiles[kMR + 1] = {nullptr, tile<1>, tile<2>, tile<3>,
        tile<4>, tile<5>, tile<6>, tile<7>, tile<8>};

/// Column sums over the `kq` quads of the packed blocks of `n` columns,
/// `bytes` apart.
HSTU_I8_NOINLINE void sum_blocks(int64_t kq, int64_t n, const int8_t *packed,
        int64_t bytes, int32_t *colsum) {
    const __m512i ones = _mm512_set1_epi8(1);
    for (int64_t o = 0; o * kNR < n; ++o) {
        const int8_t *b = packed + o * bytes;
        __m512i s0 = _mm512_setzero_si512(), s1 = _mm512_setzero_si512();
        for (int64_t q = 0; q < kq; ++q) {
            s0 = dpbusd(s0, ones, _mm512_loadu_si512(b + q * 4 * kNR));
            s1 = dpbusd(s1, ones, _mm512_loadu_si512(b + q * 4 * kNR + 64));
        }
        _mm512_storeu_si512(colsum + o * kNR, s0);
        _mm512_storeu_si512(colsum + o * kNR + 16, s1);
    }
}

} // namespace

size_t packed_size(int64_t k, int64_t n) {
    return static_cast<size_t>((n + kNR - 1) / kNR * block_bytes(k));
}

/// Each quad is one dword, so a block is 16 x 16 dword transposes of 16 rows
/// by 16 quads. Rows past N are zero.
HSTU_I8_NOINLINE void pack_kt(int64_t k, int64_t n, const int8_t *b,
        int64_t ldk, int8_t *packed, int32_t *colsum) {
    const int64_t kq = quads(k);
    const __m512i ones = _mm512_set1_epi8(1);
    for (int64_t o = 0; o * kNR < n; ++o) {
        int8_t *dst = packed + o * kq * 4 * kNR;
        for (int half = 0; half < 2; ++half) {
            const int64_t j0 = o * kNR + half * 16;
            const int rows = int(std::clamp<int64_t>(n - j0, 0, 16));
            __m512i sum = _mm512_setzero_si512();
            for (int64_t q0 = 0; q0 < kq; q0 += 16) {
                const int64_t bytes = std::min<int64_t>(64, k - 4 * q0);
                const __mmask64 mask = bytes >= 64
                        ? __mmask64(~0ull)
                        : __mmask64((1ull << bytes) - 1);
                __m512i r[16], t[16];
                for (int j = 0; j < 16; ++j)
                    r[j] = j < rows ? _mm512_maskz_loadu_epi8(
                                              mask, b + (j0 + j) * ldk + 4 * q0)
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
                // r[4i + c] holds rows 4i..4i+3 of quad 4L + c in lane L.
                const int64_t nq = std::min<int64_t>(16, kq - q0);
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
                        const int64_t qq = 4 * l + c;
                        if (qq >= nq) continue;
                        _mm512_storeu_si512(
                                dst + ((q0 + qq) * kNR + half * 16) * 4,
                                col[l]);
                        sum = dpbusd(sum, ones, col[l]);
                    }
                }
            }
            if (colsum) _mm512_storeu_si512(colsum + o * kNR + half * 16, sum);
        }
    }
}

/// Interleaves rows 4q..4q+3 of each column.
HSTU_I8_NOINLINE void pack_kn(int64_t k, int64_t n, const int8_t *b,
        int64_t ldv, int8_t *packed, int32_t *colsum) {
    const int64_t kq = quads(k);
    for (int64_t q = 0; q < kq; ++q) {
        const int8_t *row[4];
        for (int r = 0; r < 4; ++r)
            row[r] = 4 * q + r < k ? b + (4 * q + r) * ldv : nullptr;
        for (int64_t o = 0; o * kNR < n; ++o) {
            const int64_t n0 = o * kNR;
            const int nv = int(std::min<int64_t>(kNR, n - n0));
            const __mmask32 mask
                    = nv == 32 ? __mmask32(~0u) : __mmask32((1u << nv) - 1);
            __m512i w[4];
            for (int r = 0; r < 4; ++r)
                w[r] = row[r] ? _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(
                                        mask, row[r] + n0))
                              : _mm512_setzero_si512();
            // Words of rows (0, 1) and (2, 3) per column, then dwords.
            const __m512i p01
                    = _mm512_or_si512(w[0], _mm512_slli_epi16(w[1], 8));
            const __m512i p23
                    = _mm512_or_si512(w[2], _mm512_slli_epi16(w[3], 8));
            const __m512i lo = _mm512_or_si512(
                    _mm512_cvtepu16_epi32(_mm512_castsi512_si256(p01)),
                    _mm512_slli_epi32(
                            _mm512_cvtepu16_epi32(_mm512_castsi512_si256(p23)),
                            16));
            const __m512i hi = _mm512_or_si512(
                    _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(p01, 1)),
                    _mm512_slli_epi32(
                            _mm512_cvtepu16_epi32(
                                    _mm512_extracti64x4_epi64(p23, 1)),
                            16));
            int8_t *dst = packed + (o * kq + q) * 4 * kNR;
            _mm512_storeu_si512(dst, lo);
            _mm512_storeu_si512(dst + 4 * 16, hi);
        }
    }
    if (colsum) sum_blocks(kq, n, packed, block_bytes(k), colsum);
}

void packed_colsum(int64_t k, int64_t n, const int8_t *b, int32_t *colsum) {
    sum_blocks(quads(k), n, b, block_bytes(k), colsum);
}

void gemm_block(int64_t m, int nv, int64_t k, const uint8_t *a, int64_t lda,
        const int8_t *b, const int32_t *colsum, const float *row_scale,
        const float *col_scale, float *c, int64_t ldc) {
    alignas(64) int32_t comp[kNR];
    for (int j = 0; j < kNR; ++j)
        comp[j] = j < nv ? 128 * colsum[j] : 0;
    epilogue_t ep {comp, row_scale, col_scale};
    int64_t r = 0;
    for (; r + kMR <= m; r += kMR) {
        ep.row_scale = row_scale + r;
        kTiles[kMR](a + r * lda, lda, b, k, c + r * ldc, ldc, nv, ep);
    }
    if (r < m) {
        ep.row_scale = row_scale + r;
        kTiles[m - r](a + r * lda, lda, b, k, c + r * ldc, ldc, nv, ep);
    }
}

} // namespace int8
} // namespace ck
} // namespace hstu_attention
} // namespace lowoha
} // namespace zendnnl
