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

/// INT8 custom-kernel GEMMs of the dynamic-INT8 flash path: VNNI packers and
/// one `vpdpbusd` tile kernel.
///
/// Layout: B is packed per 32-column block as [quads][32][4] int8 (four K
/// values of each column side by side). A tile is MR x 32 int32 accumulators
/// fed by broadcast A quads, which `vpdpbusd` reads as unsigned: an S8 A
/// (Q) is offset by 128 once per call and the offset removed with the column
/// sums of B.

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
namespace int8 {

namespace {

/// Fewest rows of A a P*V call runs on CK (one-row P*V is at parity with
/// AOCL-DLP); Q*K^T takes any M.
constexpr int64_t kPvMinRows = 2;

int64_t quads(int64_t k) {
    return (k + 3) / 4;
}

/// Bytes of one packed 32-column block over K.
int64_t block_bytes(int64_t k) {
    return quads(k) * 4 * kNR;
}

/// How a tile turns its int32 sums into C:
/// C = (acc - comp[n]) * row_scale[m] * col_scale[n].
struct epilogue_t {
    const int32_t *comp; ///< per column, nullptr for none
    const float *row_scale; ///< per row, nullptr for `scale`
    float scale;
    const float *col_scale; ///< per column
};

using tile_fn = void (*)(const uint8_t *, int64_t, const int8_t *, int64_t,
        float *, int64_t, int, const epilogue_t &);

#define CK_I8_INLINE \
    ZENDNNL_INLINE_TARGET("avx512f,avx512bw,avx512vl,avx512vnni")
#define CK_I8_NOINLINE \
    ZENDNNL_TARGET_NOINLINE("avx512f,avx512bw,avx512vl,avx512vnni")

// ---------------------------------------------------------------------------
// Tile kernel.
// ---------------------------------------------------------------------------

/// acc += U8 a * S8 b (vpdpbusd).
CK_I8_INLINE inline __m512i dpbusd(__m512i acc, __m512i a, __m512i b) {
#if defined(__GNUC__) && !defined(__clang__)
    // GCC copies the accumulator around the intrinsic in unrolled loops,
    // about one register move per vpdpbusd; tying it keeps it in place.
    __asm__("vpdpbusd %2, %1, %0" : "+v"(acc) : "v"(a), "v"(b));
    return acc;
#else
    return _mm512_dpbusd_epi32(acc, a, b);
#endif
}

/// acc[MR][2] += A[:, 0 : 4] * one packed quad, with A rows `lda` apart.
template <int MR>
CK_I8_INLINE inline void quad_step(__m512i (&acc)[MR][2],
        const uint8_t *__restrict a, int64_t lda, const int8_t *__restrict b) {
    const __m512i b0 = _mm512_loadu_si512(b);
    const __m512i b1 = _mm512_loadu_si512(b + 4 * 16);
#pragma GCC unroll 8
    for (int m = 0; m < MR; ++m) {
        uint32_t quad;
        std::memcpy(&quad, a + m * lda, sizeof(quad));
        const __m512i av = _mm512_set1_epi32(int(quad));
        acc[m][0] = dpbusd(acc[m][0], av, b0);
        acc[m][1] = dpbusd(acc[m][1], av, b1);
    }
}

/// C[MR][nv] = epilogue(A[MR][K] * one packed block), U8 A, nv <= 32.
template <int MR>
CK_I8_NOINLINE void tile(const uint8_t *__restrict a, int64_t lda,
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
        quad_step<MR>(acc, a + 4 * q, lda, b + q * 4 * kNR);
        quad_step<MR>(acc, a + 4 * q + 4, lda, b + (q + 1) * 4 * kNR);
    }
    if (q < full) quad_step<MR>(acc, a + 4 * q, lda, b + q * 4 * kNR);
    if (const int rem = int(k & 3)) {
        // The pack zeroed B past K, so the A bytes there only need to be
        // in bounds: copy the `rem` that are.
        alignas(64) uint8_t tail[MR][4] = {};
        for (int m = 0; m < MR; ++m)
            std::memcpy(tail[m], a + m * lda + 4 * full, size_t(rem));
        quad_step<MR>(acc, &tail[0][0], 4, b + full * 4 * kNR);
    }

    const __m512i comp0 = ep.comp ? _mm512_maskz_loadu_epi32(m0, ep.comp)
                                  : _mm512_setzero_si512();
    const __m512i comp1 = ep.comp ? _mm512_maskz_loadu_epi32(m1, ep.comp + 16)
                                  : _mm512_setzero_si512();
    const __m512 cs0 = _mm512_maskz_loadu_ps(m0, ep.col_scale);
    const __m512 cs1 = _mm512_maskz_loadu_ps(m1, ep.col_scale + 16);
#pragma GCC unroll 8
    for (int m = 0; m < MR; ++m) {
        const __m512 rs
                = _mm512_set1_ps(ep.row_scale ? ep.row_scale[m] : ep.scale);
        const __m512 f0 = _mm512_mul_ps(
                _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[m][0], comp0)),
                _mm512_mul_ps(rs, cs0));
        const __m512 f1 = _mm512_mul_ps(
                _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[m][1], comp1)),
                _mm512_mul_ps(rs, cs1));
        _mm512_mask_storeu_ps(c + m * ldc, m0, f0);
        _mm512_mask_storeu_ps(c + m * ldc + 16, m1, f1);
    }
}

/// Tile kernels by row count.
constexpr tile_fn kTiles[kMR + 1] = {nullptr, tile<1>, tile<2>, tile<3>,
        tile<4>, tile<5>, tile<6>, tile<7>, tile<8>};

/// dst = src + 128 bytewise (S8 to the U8 `vpdpbusd` reads), `n` bytes.
CK_I8_NOINLINE void offset_s8(const int8_t *src, int64_t n, uint8_t *dst) {
    const __m512i flip = _mm512_set1_epi8(char(0x80));
    int64_t i = 0;
    for (; i + 64 <= n; i += 64)
        _mm512_storeu_si512(
                dst + i, _mm512_xor_si512(_mm512_loadu_si512(src + i), flip));
    if (i < n) {
        const __mmask64 mask = (1ull << (n - i)) - 1;
        _mm512_mask_storeu_epi8(dst + i, mask,
                _mm512_xor_si512(_mm512_maskz_loadu_epi8(mask, src + i), flip));
    }
}

// ---------------------------------------------------------------------------
// Packers.
// ---------------------------------------------------------------------------

/// K x N row-major B (V): interleaves rows 4q..4q+3 of each column.
CK_I8_NOINLINE void pack_rows(
        int64_t k, int64_t n, const int8_t *b, int64_t ldb, int8_t *out) {
    const int64_t kq = quads(k);
    for (int64_t q = 0; q < kq; ++q) {
        const int8_t *row[4];
        for (int r = 0; r < 4; ++r)
            row[r] = 4 * q + r < k ? b + (4 * q + r) * ldb : nullptr;
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
            int8_t *dst = out + (o * kq + q) * 4 * kNR;
            _mm512_storeu_si512(dst, lo);
            _mm512_storeu_si512(dst + 4 * 16, hi);
        }
    }
}

/// N x K row-major B (K, op(B) = B^T): each quad is one dword, so a block
/// is 16 x 16 dword transposes of 16 rows by 16 quads. Rows past N are
/// zero. With `colsum`, writes the sum of each column over K.
CK_I8_NOINLINE void pack_cols(int64_t k, int64_t n, const int8_t *b,
        int64_t ldb, int8_t *out, int32_t *colsum) {
    const int64_t kq = quads(k);
    const __m512i ones = _mm512_set1_epi8(1);
    for (int64_t o = 0; o * kNR < n; ++o) {
        int8_t *dst = out + o * kq * 4 * kNR;
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
                                              mask, b + (j0 + j) * ldb + 4 * q0)
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
                        sum = _mm512_dpbusd_epi32(sum, ones, col[l]);
                    }
                }
            }
            if (colsum) _mm512_storeu_si512(colsum + o * kNR + half * 16, sum);
        }
    }
}

#undef CK_I8_INLINE
#undef CK_I8_NOINLINE

/// The per-thread packed B block over `k`, or null when it cannot grow.
int8_t *packed_block(int64_t k) {
    thread_local std::vector<int8_t> t_block;
    return grow(t_block, size_t(block_bytes(k))) ? t_block.data() : nullptr;
}

/// C[M, N] = epilogue(A * B) for U8 A and B packed one 32-column block at a
/// time into `block` by `pack(n0, nv, block, colsum)`. `Offset`: A is an S8
/// matrix plus 128, removed with the column sums.
template <bool Offset, typename PackFn>
void gemm_blocks(int64_t m, int64_t n, int64_t k, const uint8_t *a, int64_t lda,
        float *c, int64_t ldc, const float *row_scale, float scale,
        const float *col_scale, int8_t *block, PackFn pack) {
    alignas(64) int32_t comp[kNR];
    const tile_fn *fns = kTiles;
    for (int64_t n0 = 0; n0 < n; n0 += kNR) {
        const int nv = int(std::min<int64_t>(kNR, n - n0));
        pack(n0, nv, block, comp);
        if constexpr (Offset)
            for (int j = 0; j < kNR; ++j)
                comp[j] *= 128;
        epilogue_t ep {
                Offset ? comp : nullptr, row_scale, scale, col_scale + n0};
        int64_t r = 0;
        for (; r + kMR <= m; r += kMR) {
            if (row_scale) ep.row_scale = row_scale + r;
            fns[kMR](a + r * lda, lda, block, k, c + r * ldc + n0, ldc, nv, ep);
        }
        if (r < m) {
            if (row_scale) ep.row_scale = row_scale + r;
            fns[m - r](
                    a + r * lda, lda, block, k, c + r * ldc + n0, ldc, nv, ep);
        }
    }
}

} // namespace

bool qk(const int8_t *q, const int8_t *k, float *scores, int64_t m, int64_t n,
        int64_t d, const float *q_scales, const float *k_scales) {
    if (m <= 0 || n <= 0) return true;
    // Q as U8 once, rather than per packed block in the tile.
    thread_local std::vector<uint8_t> t_q;
    int8_t *block = packed_block(d);
    if (!block || !grow(t_q, size_t(m * d))) return false;
    count_int8_call();
    offset_s8(q, m * d, t_q.data());
    gemm_blocks<true>(m, n, d, t_q.data(), d, scores, n, q_scales, 1.f,
            k_scales, block,
            [&](int64_t n0, int nv, int8_t *block, int32_t *colsum) {
        pack_cols(d, nv, k + n0 * d, d, block, colsum);
    });
    return true;
}

bool pv(const uint8_t *p, const int8_t *v, float *out, int64_t m, int64_t n,
        int64_t kv, float p_scale, const float *v_scales) {
    if (m > 0 && m < kPvMinRows) return false;
    if (m <= 0 || n <= 0) return true;
    int8_t *block = packed_block(kv);
    if (!block) return false;
    count_int8_call();
    gemm_blocks<false>(m, n, kv, p, kv, out, n, nullptr, p_scale, v_scales,
            block, [&](int64_t n0, int nv, int8_t *block, int32_t *) {
        pack_rows(kv, nv, v + n0, n, block);
    });
    return true;
}

} // namespace int8
} // namespace ck
} // namespace sdpa
} // namespace lowoha
} // namespace zendnnl
