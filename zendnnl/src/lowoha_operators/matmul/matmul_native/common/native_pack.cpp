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

#include "lowoha_operators/matmul/matmul_native/common/native_pack.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>

#include <immintrin.h>
#include <omp.h>

#include "common/zendnnl_compat.hpp"
#include "common/zendnnl_global.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::error_handling::log_error;

namespace {

/// Weight column held at pack column p.
template <bool Interleave>
inline int source_col(int p, int half) {
    if constexpr (Interleave) {
        return (p & 1) ? half + (p >> 1) : (p >> 1);
    } else {
        (void)half;
        return p;
    }
}

/// A K group row of a block: 32 columns of 4 bytes (a bf16 pair, an s8 quad
/// or an f32), two vectors of 16 columns.
constexpr int kBlockCols = 32;
constexpr size_t kGroupRowBytes = kBlockCols * 4;
static_assert(kNativeLayoutPackNr == kBlockCols,
        "the pack writes blocks of 32 columns");

/// Lane i of out[j] is lane j of in[i]: a transpose of 16 x 16 groups of 4
/// bytes.
ZENDNNL_TARGET("avx512f")
inline void transpose_groups(const __m512i (&in)[16], __m512i (&out)[16]) {
    __m512i t[16], u[16];
    for (int i = 0; i < 16; i += 2) {
        t[i] = _mm512_unpacklo_epi32(in[i], in[i + 1]);
        t[i + 1] = _mm512_unpackhi_epi32(in[i], in[i + 1]);
    }
    // u[i + c], lane L: group 4L + c of rows i .. i + 3.
    for (int i = 0; i < 16; i += 4) {
        u[i] = _mm512_unpacklo_epi64(t[i], t[i + 2]);
        u[i + 1] = _mm512_unpackhi_epi64(t[i], t[i + 2]);
        u[i + 2] = _mm512_unpacklo_epi64(t[i + 1], t[i + 3]);
        u[i + 3] = _mm512_unpackhi_epi64(t[i + 1], t[i + 3]);
    }
    for (int c = 0; c < 4; ++c) {
        const __m512i v0 = _mm512_shuffle_i32x4(u[c], u[4 + c], 0x88);
        const __m512i v1 = _mm512_shuffle_i32x4(u[c], u[4 + c], 0xDD);
        const __m512i v2 = _mm512_shuffle_i32x4(u[8 + c], u[12 + c], 0x88);
        const __m512i v3 = _mm512_shuffle_i32x4(u[8 + c], u[12 + c], 0xDD);
        out[c] = _mm512_shuffle_i32x4(v0, v2, 0x88);
        out[4 + c] = _mm512_shuffle_i32x4(v1, v3, 0x88);
        out[8 + c] = _mm512_shuffle_i32x4(v0, v2, 0xDD);
        out[12 + c] = _mm512_shuffle_i32x4(v1, v3, 0xDD);
    }
}

/// Per group of 4 bytes of v, the sum of its signed bytes.
ZENDNNL_TARGET("avx512f,avx512bw")
inline __m512i quad_sums(__m512i v) {
    return _mm512_madd_epi16(
            _mm512_maddubs_epi16(_mm512_set1_epi8(1), v), _mm512_set1_epi16(1));
}

ZENDNNL_TARGET("avx512f,avx512bw,avx512vl")
inline __m256i quad_sums(__m256i v) {
    return _mm256_madd_epi16(
            _mm256_maddubs_epi16(_mm256_set1_epi8(1), v), _mm256_set1_epi16(1));
}

/// The groups of 4 bytes from byte at on of the 16 rows rows[i], 16 or as
/// many as the left bytes of a row hold (the rest read as zero), transposed:
/// lane i of out[q] is group q of row i.  Tail: only the first nrows rows
/// are read, the others read as zero.
template <bool Tail>
ZENDNNL_TARGET("avx512f,avx512bw")
inline void load_group_columns(const uint8_t *const (&rows)[16], size_t at,
        size_t left, __m512i (&out)[16], int nrows) {
    __m512i in[16];
    const int n = Tail ? nrows : 16;
    if (left >= 64) {
        for (int i = 0; i < n; ++i)
            in[i] = _mm512_loadu_si512(rows[i] + at);
    } else {
        const __mmask64 m = (__mmask64 {1} << left) - 1;
        for (int i = 0; i < n; ++i)
            in[i] = _mm512_maskz_loadu_epi8(m, rows[i] + at);
    }
    for (int i = n; i < 16; ++i)
        in[i] = _mm512_setzero_si512();
    transpose_groups(in, out);
}

/// The 8 two's-complement nibbles n0 .. n7 (n_i at bits 4i) of each 32-bit
/// lane of v, one s4 octet of one column as the caller stores it, in the
/// layout: byte q holds n_q + 8 in its low nibble and n_(q + 4) + 8 in its
/// high nibble.  Zero bytes give the layout's zero, nibble 8.
ZENDNNL_TARGET("avx512f,avx512bw")
inline __m512i s4_octets(__m512i v) {
    // Bytes 1 and 2 of each lane swap places: n0 n1 n4 n5 n2 n3 n6 n7.
    const __m512i bytes
            = _mm512_set4_epi32(0x0F0D0E0C, 0x0B090A08, 0x07050604, 0x03010200);
    v = _mm512_shuffle_epi8(v, bytes);
    // Then nibbles 1 and 2 of each 16 bits: n0 n4 n1 n5 n2 n6 n3 n7.
    const __m512i t
            = _mm512_and_si512(_mm512_xor_si512(_mm512_srli_epi32(v, 4), v),
                    _mm512_set1_epi32(0x00F000F0));
    v = _mm512_xor_si512(v, _mm512_xor_si512(t, _mm512_slli_epi32(t, 4)));
    return _mm512_xor_si512(v, _mm512_set1_epi8(static_cast<char>(0x88)));
}

/// The block of pack columns p0 .. p0 + 31 of a weight stored [N, K]
/// (transB): each pack column is a row of the buffer, row_bytes apart, of
/// which k_bytes are the weight's; groups K groups of 4 bytes.  Sums: the s8
/// column sums follow the groups.  S4: each group of 4 bytes is an s4 octet,
/// stored as s4_octets makes it.  Tail: only the first cols pack columns
/// hold weight columns (cols even when interleaved); the others are zero and
/// their rows are not read.
template <bool Interleave, bool Sums, bool Tail, bool S4 = false>
ZENDNNL_TARGET("avx512f,avx512bw")
void pack_block_rows_of_n(const uint8_t *w, size_t row_bytes, size_t k_bytes,
        int groups, int half, int p0, uint8_t *blk, int cols) {
    // Pack columns 0 .. 15 and 16 .. 31, or, interleaved, the gate and the up
    // columns of the block.
    const uint8_t *first[16], *second[16];
    const int n_first = Tail ? std::min(16, Interleave ? cols / 2 : cols) : 16;
    const int n_second
            = Tail ? (Interleave ? n_first : std::max(0, cols - 16)) : 16;
    for (int i = 0; i < 16; ++i) {
        const int c0 = Interleave ? p0 / 2 + i : p0 + i;
        const int c1 = Interleave ? half + p0 / 2 + i : p0 + 16 + i;
        first[i] = i < n_first ? w + static_cast<size_t>(c0) * row_bytes : w;
        second[i] = i < n_second ? w + static_cast<size_t>(c1) * row_bytes : w;
    }
    // Interleaved, pack column 2i is gate i and 2i + 1 up i.
    const __m512i lo_idx = _mm512_setr_epi32(
            0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
    const __m512i hi_idx = _mm512_setr_epi32(
            8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
    __m512i sum_lo = _mm512_setzero_si512();
    __m512i sum_hi = _mm512_setzero_si512();
    for (int g0 = 0; g0 < groups; g0 += 16) {
        const int count = std::min(16, groups - g0);
        const size_t at = static_cast<size_t>(g0) * 4;
        __m512i a[16], b[16];
        load_group_columns<Tail>(first, at, k_bytes - at, a, n_first);
        load_group_columns<Tail>(second, at, k_bytes - at, b, n_second);
        for (int q = 0; q < count; ++q) {
            __m512i lo = a[q], hi = b[q];
            if constexpr (Interleave) {
                lo = _mm512_permutex2var_epi32(a[q], lo_idx, b[q]);
                hi = _mm512_permutex2var_epi32(a[q], hi_idx, b[q]);
            }
            if constexpr (S4) {
                lo = s4_octets(lo);
                hi = s4_octets(hi);
            }
            uint8_t *d = blk + static_cast<size_t>(g0 + q) * kGroupRowBytes;
            _mm512_storeu_si512(d, lo);
            _mm512_storeu_si512(d + 64, hi);
            if constexpr (Sums) {
                sum_lo = _mm512_add_epi32(sum_lo, quad_sums(lo));
                sum_hi = _mm512_add_epi32(sum_hi, quad_sums(hi));
            }
        }
    }
    if constexpr (Sums) {
        uint8_t *d = blk + static_cast<size_t>(groups) * kGroupRowBytes;
        _mm512_storeu_si512(d, sum_lo);
        _mm512_storeu_si512(d + 64, sum_hi);
    }
}

/// Pack columns p0 .. p0 + 31 of a row of a bf16 weight stored [K, N];
/// interleaved, the 16 gate columns, then the 16 up columns.  Tail: only the
/// first cols pack columns (cols below 32, even when interleaved) hold
/// weight columns; the others read as zero.
template <bool Interleave, bool Tail>
ZENDNNL_TARGET("avx512f,avx512bw")
inline __m512i
        load_bf16_columns(const uint16_t *row, int half, int p0, int cols) {
    if constexpr (Tail) {
        if constexpr (Interleave) {
            const __mmask32 m = (__mmask32 {1} << (cols / 2)) - 1;
            const __m256i gate = _mm512_castsi512_si256(
                    _mm512_maskz_loadu_epi16(m, row + p0 / 2));
            const __m256i up = _mm512_castsi512_si256(
                    _mm512_maskz_loadu_epi16(m, row + half + p0 / 2));
            return _mm512_inserti64x4(_mm512_castsi256_si512(gate), up, 1);
        } else {
            (void)half;
            return _mm512_maskz_loadu_epi16(
                    (__mmask32 {1} << cols) - 1, row + p0);
        }
    } else if constexpr (Interleave) {
        (void)cols;
        const __m256i gate = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(row + p0 / 2));
        const __m256i up = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(row + half + p0 / 2));
        return _mm512_inserti64x4(_mm512_castsi256_si512(gate), up, 1);
    } else {
        (void)half;
        (void)cols;
        return _mm512_loadu_si512(row + p0);
    }
}

/// The word permutes of a K pair row of 32 pack columns of a bf16 weight
/// stored [K, N], built at compile time: v[0] makes its first 16 columns,
/// v[1] its last 16.
template <bool Interleave>
struct bf16_pair_idx_t {
    alignas(64) uint16_t v[2][32] {};
    constexpr bf16_pair_idx_t() {
        // The pair of a column takes row k (index below 32) and row k + 1
        // (index 32 and up) of its column.
        for (int t = 0; t < 32; ++t) {
            const int from_next = (t & 1) ? 32 : 0;
            if constexpr (Interleave) {
                const int up = (t & 2) ? 16 : 0;
                v[0][t] = static_cast<uint16_t>(from_next + up + t / 4);
                v[1][t] = static_cast<uint16_t>(from_next + up + 8 + t / 4);
            } else {
                v[0][t] = static_cast<uint16_t>(from_next + t / 2);
                v[1][t] = static_cast<uint16_t>(from_next + 16 + t / 2);
            }
        }
    }
};

template <bool Interleave>
inline constexpr bf16_pair_idx_t<Interleave> kBf16PairIdx {};

template <bool Interleave>
ZENDNNL_TARGET("avx512f,avx512bw")
inline void bf16_pair_index(__m512i &lo, __m512i &hi) {
    lo = _mm512_load_si512(kBf16PairIdx<Interleave>.v[0]);
    hi = _mm512_load_si512(kBf16PairIdx<Interleave>.v[1]);
}

/// K pair row kp of pack columns p0 .. p0 + 31 of a bf16 weight stored
/// [K, N], at d; Tail as load_bf16_columns.
template <bool Interleave, bool Tail>
ZENDNNL_TARGET("avx512f,avx512bw")
inline void pack_bf16_pair_row(const uint16_t *w, int K, size_t ldb, int half,
        int p0, int kp, __m512i lo_idx, __m512i hi_idx, uint16_t *d, int cols) {
    const int k = kp * VNNI_PAIR;
    const uint16_t *row = w + static_cast<size_t>(k) * ldb;
    const __m512i a = load_bf16_columns<Interleave, Tail>(row, half, p0, cols);
    const __m512i b = k + 1 < K
            ? load_bf16_columns<Interleave, Tail>(row + ldb, half, p0, cols)
            : _mm512_setzero_si512();
    _mm512_storeu_si512(d, _mm512_permutex2var_epi16(a, lo_idx, b));
    _mm512_storeu_si512(d + 32, _mm512_permutex2var_epi16(a, hi_idx, b));
}

/// The block of pack columns p0 .. p0 + 31 of a bf16 weight stored [K, N];
/// Tail as load_bf16_columns.
template <bool Interleave, bool Tail>
ZENDNNL_TARGET("avx512f,avx512bw")
void pack_bf16_block_rows_of_k(const uint16_t *w, int K, size_t ldb, int half,
        int p0, uint16_t *blk, int cols) {
    __m512i lo_idx, hi_idx;
    bf16_pair_index<Interleave>(lo_idx, hi_idx);
    const int k_pairs = native_k_groups(data_type_t::bf16, K);
    for (int kp = 0; kp < k_pairs; ++kp)
        pack_bf16_pair_row<Interleave, Tail>(w, K, ldb, half, p0, kp, lo_idx,
                hi_idx, blk + static_cast<size_t>(kp) * kBlockCols * VNNI_PAIR,
                cols);
}

/// A bf16 or f32 weight stored [K, N] of at least kTileMinK rows packs by
/// tiles of kTileGroups K groups by kTileBlocks blocks.  A block reads one
/// or two lines of each of its rows, a row stride apart: a walk down that
/// many rows loses the lines it fetches ahead (all in one cache set when the
/// stride is a multiple of the page size).  A tile reads the segments of its
/// rows in order and writes the part of each block whole.
constexpr int kTileGroups = 8;
constexpr int kTileBlocks = 16;
constexpr int kTileMinK = 1024;

/// K pairs kp0 .. kp1 (at most kTileGroups) of blocks nb0 .. nb1 (at most
/// kTileBlocks) of a bf16 weight stored [K, N] into the layout at out, whose
/// blocks are block_bytes long.
template <bool Interleave>
ZENDNNL_TARGET("avx512f,avx512bw")
void pack_bf16_tile_rows_of_k(const uint16_t *w, int K, size_t ldb, int half,
        int kp0, int kp1, int nb0, int nb1, size_t block_bytes, uint8_t *out) {
    alignas(64)
            uint16_t tile[kTileBlocks][kTileGroups * kBlockCols * VNNI_PAIR];
    __m512i lo_idx, hi_idx;
    bf16_pair_index<Interleave>(lo_idx, hi_idx);
    for (int kp = kp0; kp < kp1; ++kp)
        for (int nb = nb0; nb < nb1; ++nb)
            pack_bf16_pair_row<Interleave, false>(w, K, ldb, half,
                    nb * kBlockCols, kp, lo_idx, hi_idx,
                    tile[nb - nb0] + (kp - kp0) * kBlockCols * VNNI_PAIR,
                    kBlockCols);
    const size_t part = static_cast<size_t>(kp1 - kp0) * kGroupRowBytes;
    for (int nb = nb0; nb < nb1; ++nb) {
        const uint8_t *s = reinterpret_cast<const uint8_t *>(tile[nb - nb0]);
        uint8_t *d = out + static_cast<size_t>(nb) * block_bytes
                + static_cast<size_t>(kp0) * kGroupRowBytes;
        for (size_t i = 0; i < part; i += 64)
            _mm512_storeu_si512(d + i, _mm512_load_si512(s + i));
    }
}

/// The lanes below lanes (0 .. 16) of a 16-lane mask.
inline __mmask16 lanes_below(int lanes) {
    return lanes >= 16
            ? static_cast<__mmask16>(0xFFFF)
            : static_cast<__mmask16>((1u << std::max(0, lanes)) - 1u);
}

/// Pack columns p0 .. p0 + 31 of a row of an f32 weight stored [K, N], at
/// d; interleaved, gate column i at pack column 2i and up column i at
/// 2i + 1.  Tail: only the first cols pack columns (even when interleaved)
/// hold weight columns; the others are zero.
template <bool Interleave, bool Tail>
ZENDNNL_TARGET("avx512f")
inline void pack_f32_row(
        const float *row, int half, int p0, float *d, int cols) {
    if constexpr (Interleave) {
        const __m512i lo_idx = _mm512_setr_epi32(
                0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
        const __m512i hi_idx = _mm512_setr_epi32(
                8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
        __m512 gate, up;
        if constexpr (Tail) {
            const __mmask16 m = lanes_below(cols / 2);
            gate = _mm512_maskz_loadu_ps(m, row + p0 / 2);
            up = _mm512_maskz_loadu_ps(m, row + half + p0 / 2);
        } else {
            (void)cols;
            gate = _mm512_loadu_ps(row + p0 / 2);
            up = _mm512_loadu_ps(row + half + p0 / 2);
        }
        _mm512_storeu_ps(d, _mm512_permutex2var_ps(gate, lo_idx, up));
        _mm512_storeu_ps(d + 16, _mm512_permutex2var_ps(gate, hi_idx, up));
    } else if constexpr (Tail) {
        (void)half;
        _mm512_storeu_ps(d, _mm512_maskz_loadu_ps(lanes_below(cols), row + p0));
        _mm512_storeu_ps(d + 16,
                _mm512_maskz_loadu_ps(lanes_below(cols - 16), row + p0 + 16));
    } else {
        (void)half;
        (void)cols;
        _mm512_storeu_ps(d, _mm512_loadu_ps(row + p0));
        _mm512_storeu_ps(d + 16, _mm512_loadu_ps(row + p0 + 16));
    }
}

/// The block of pack columns p0 .. p0 + 31 of an f32 weight stored [K, N];
/// Tail as pack_f32_row.
template <bool Interleave, bool Tail>
ZENDNNL_TARGET("avx512f")
void pack_f32_block_rows_of_k(const float *w, int K, size_t ldb, int half,
        int p0, float *blk, int cols) {
    for (int k = 0; k < K; ++k)
        pack_f32_row<Interleave, Tail>(w + static_cast<size_t>(k) * ldb, half,
                p0, blk + static_cast<size_t>(k) * kBlockCols, cols);
}

/// Rows k0 .. k1 (at most kTileGroups) of blocks nb0 .. nb1 (at most
/// kTileBlocks) of an f32 weight stored [K, N] into the layout at out, whose
/// blocks are block_bytes long.
template <bool Interleave>
ZENDNNL_TARGET("avx512f")
void pack_f32_tile_rows_of_k(const float *w, size_t ldb, int half, int k0,
        int k1, int nb0, int nb1, size_t block_bytes, uint8_t *out) {
    alignas(64) float tile[kTileBlocks][kTileGroups * kBlockCols];
    for (int k = k0; k < k1; ++k)
        for (int nb = nb0; nb < nb1; ++nb)
            pack_f32_row<Interleave, false>(w + static_cast<size_t>(k) * ldb,
                    half, nb * kBlockCols,
                    tile[nb - nb0] + (k - k0) * kBlockCols, kBlockCols);
    const size_t part = static_cast<size_t>(k1 - k0) * kGroupRowBytes;
    for (int nb = nb0; nb < nb1; ++nb) {
        const uint8_t *s = reinterpret_cast<const uint8_t *>(tile[nb - nb0]);
        uint8_t *d = out + static_cast<size_t>(nb) * block_bytes
                + static_cast<size_t>(k0) * kGroupRowBytes;
        for (size_t i = 0; i < part; i += 64)
            _mm512_storeu_si512(d + i, _mm512_load_si512(s + i));
    }
}

/// Pack columns p0 .. p0 + 31 of a row of an s8 weight stored [K, N];
/// interleaved, gate and up columns alternate.  Tail: only the first cols
/// pack columns (cols below 32, even when interleaved) hold weight columns;
/// the others read as zero.
template <bool Interleave, bool Tail>
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl")
inline __m256i load_s8_columns(const int8_t *row, int half, int p0, int cols) {
    if constexpr (Interleave) {
        __m128i gate, up;
        if constexpr (Tail) {
            const __mmask16 m = static_cast<__mmask16>((1u << (cols / 2)) - 1u);
            gate = _mm_maskz_loadu_epi8(m, row + p0 / 2);
            up = _mm_maskz_loadu_epi8(m, row + half + p0 / 2);
        } else {
            (void)cols;
            gate = _mm_loadu_si128(
                    reinterpret_cast<const __m128i *>(row + p0 / 2));
            up = _mm_loadu_si128(
                    reinterpret_cast<const __m128i *>(row + half + p0 / 2));
        }
        return _mm256_set_m128i(
                _mm_unpackhi_epi8(gate, up), _mm_unpacklo_epi8(gate, up));
    } else if constexpr (Tail) {
        (void)half;
        return _mm256_maskz_loadu_epi8((__mmask32 {1} << cols) - 1, row + p0);
    } else {
        (void)half;
        (void)cols;
        return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(row + p0));
    }
}

/// The block of pack columns p0 .. p0 + 31 of an s8 weight stored [K, N],
/// its column sums after the quads; Tail as load_s8_columns (the zero
/// columns sum to zero).
template <bool Interleave, bool Tail>
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl")
void pack_s8_block_rows_of_k(const int8_t *w, int K, size_t ldb, int half,
        int p0, int8_t *blk, int cols) {
    const int k_quads = native_k_groups(data_type_t::s8, K);
    __m256i sums[4];
    for (__m256i &s : sums)
        s = _mm256_setzero_si256();
    for (int kq = 0; kq < k_quads; ++kq) {
        const int k = kq * INT8_VNNI_GRP;
        __m256i r[4];
        for (int i = 0; i < 4; ++i)
            r[i] = k + i < K ? load_s8_columns<Interleave, Tail>(
                                       w + static_cast<size_t>(k + i) * ldb,
                                       half, p0, cols)
                             : _mm256_setzero_si256();
        // Byte pairs of rows 0 | 1 and 2 | 3, then the quads of columns
        // 0 .. 3 | 16 .. 19 (e), 4 .. 7 | 20 .. 23 (f), 8 .. 11 | 24 .. 27 (g)
        // and 12 .. 15 | 28 .. 31 (h) per 128-bit lane.
        const __m256i a = _mm256_unpacklo_epi8(r[0], r[1]);
        const __m256i b = _mm256_unpackhi_epi8(r[0], r[1]);
        const __m256i c = _mm256_unpacklo_epi8(r[2], r[3]);
        const __m256i d = _mm256_unpackhi_epi8(r[2], r[3]);
        const __m256i e = _mm256_unpacklo_epi16(a, c);
        const __m256i f = _mm256_unpackhi_epi16(a, c);
        const __m256i g = _mm256_unpacklo_epi16(b, d);
        const __m256i h = _mm256_unpackhi_epi16(b, d);
        const __m256i quads[4] = {_mm256_permute2x128_si256(e, f, 0x20),
                _mm256_permute2x128_si256(g, h, 0x20),
                _mm256_permute2x128_si256(e, f, 0x31),
                _mm256_permute2x128_si256(g, h, 0x31)};
        int8_t *dst = blk + static_cast<size_t>(kq) * kGroupRowBytes;
        for (int i = 0; i < 4; ++i) {
            _mm256_storeu_si256(
                    reinterpret_cast<__m256i *>(dst + 32 * i), quads[i]);
            sums[i] = _mm256_add_epi32(sums[i], quad_sums(quads[i]));
        }
    }
    int8_t *dst = blk + static_cast<size_t>(k_quads) * kGroupRowBytes;
    for (int i = 0; i < 4; ++i)
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst + 32 * i), sums[i]);
}

/// The two's-complement nibble of element (k, n) of an s4 weight stored
/// [K, N] ldb elements apart, or [N, K] when transB.
inline uint8_t s4_nibble(
        const uint8_t *w, size_t ldb, bool transB, int k, int n) {
    const size_t i = transB
            ? static_cast<size_t>(n) * ldb + static_cast<size_t>(k)
            : static_cast<size_t>(k) * ldb + static_cast<size_t>(n);
    return static_cast<uint8_t>((w[i >> 1] >> ((i & 1) * 4)) & 0x0F);
}

/// The block of pack columns p0 .. p0 + 31 of an s4 weight read element by
/// element (any ldb, either storage); pack columns from N on and K past the
/// weight's hold the layout's zero.
template <bool Interleave>
void pack_s4_block_of_elements(const uint8_t *w, int K, int N, size_t ldb,
        bool transB, int p0, uint8_t *blk) {
    const int half = N / 2;
    const int octets = native_k_groups(data_type_t::s4, K);
    for (int o = 0; o < octets; ++o) {
        uint8_t *row = blk + static_cast<size_t>(o) * kGroupRowBytes;
        for (int q = 0; q < 4; ++q) {
            const int k0 = kNativeS4Octet * o + q;
            const int k1 = k0 + kNativeS4Octet / 2;
            for (int j = 0; j < kBlockCols; ++j) {
                const int p = p0 + j;
                uint8_t lo = 0, hi = 0;
                if (p < N) {
                    const int n = source_col<Interleave>(p, half);
                    if (k0 < K) lo = s4_nibble(w, ldb, transB, k0, n);
                    if (k1 < K) hi = s4_nibble(w, ldb, transB, k1, n);
                }
                row[j * 4 + q] = static_cast<uint8_t>(
                        (lo ^ 0x08u) | ((hi ^ 0x08u) << 4));
            }
        }
    }
}

/// The block nb of an s4 weight: by rows of whole bytes for a weight stored
/// [N, K] with even ldb and K, else element by element; Tail as
/// pack_bf16_block.
template <bool TransB, bool Interleave, bool Tail>
void pack_s4_block(const uint8_t *w, int K, int N, size_t ldb, int pack_nr,
        int nb, uint8_t *blk) {
    const int p0 = nb * pack_nr;
    if (TransB && ldb % 2 == 0 && K % 2 == 0)
        pack_block_rows_of_n<Interleave, false, Tail, true>(w, ldb / 2,
                static_cast<size_t>(K) / 2, native_k_groups(data_type_t::s4, K),
                N / 2, p0, blk, N - p0);
    else
        pack_s4_block_of_elements<Interleave>(w, K, N, ldb, TransB, p0, blk);
}

/// The block nb of a bf16 weight; Tail: the last block, of fewer than
/// pack_nr columns of the weight (pack columns from N on are zero).
template <bool TransB, bool Interleave, bool Tail>
void pack_bf16_block(const uint16_t *w, int K, int N, size_t ldb, int pack_nr,
        int nb, uint16_t *blk) {
    const int p0 = nb * pack_nr;
    const int cols = N - p0;
    if constexpr (TransB)
        pack_block_rows_of_n<Interleave, false, Tail>(
                reinterpret_cast<const uint8_t *>(w), ldb * sizeof(uint16_t),
                static_cast<size_t>(K) * sizeof(uint16_t),
                native_k_groups(data_type_t::bf16, K), N / 2, p0,
                reinterpret_cast<uint8_t *>(blk), cols);
    else
        pack_bf16_block_rows_of_k<Interleave, Tail>(
                w, K, ldb, N / 2, p0, blk, cols);
}

/// The block nb of an s8 weight; Tail as pack_bf16_block.
template <bool TransB, bool Interleave, bool Tail>
void pack_s8_block(const int8_t *w, int K, int N, size_t ldb, int pack_nr,
        int nb, int8_t *blk) {
    const int p0 = nb * pack_nr;
    const int cols = N - p0;
    if constexpr (TransB)
        pack_block_rows_of_n<Interleave, true, Tail>(
                reinterpret_cast<const uint8_t *>(w), ldb,
                static_cast<size_t>(K), native_k_groups(data_type_t::s8, K),
                N / 2, p0, reinterpret_cast<uint8_t *>(blk), cols);
    else
        pack_s8_block_rows_of_k<Interleave, Tail>(
                w, K, ldb, N / 2, p0, blk, cols);
}

/// The block nb of an f32 weight; Tail as pack_bf16_block.
template <bool TransB, bool Interleave, bool Tail>
void pack_f32_block(const float *w, int K, int N, size_t ldb, int pack_nr,
        int nb, float *blk) {
    const int p0 = nb * pack_nr;
    const int cols = N - p0;
    if constexpr (TransB)
        pack_block_rows_of_n<Interleave, false, Tail>(
                reinterpret_cast<const uint8_t *>(w), ldb * sizeof(float),
                static_cast<size_t>(K) * sizeof(float), K, N / 2, p0,
                reinterpret_cast<uint8_t *>(blk), cols);
    else
        pack_f32_block_rows_of_k<Interleave, Tail>(
                w, K, ldb, N / 2, p0, blk, cols);
}

template <bool TransB, bool Interleave>
void unpack_bf16_block(const uint16_t *blk, int K, int N, size_t ldb,
        int pack_nr, int nb, uint16_t *w) {
    const int half = N / 2;
    const int p0 = nb * pack_nr;
    const int k_pairs = native_k_groups(data_type_t::bf16, K);
    for (int kp = 0; kp < k_pairs; ++kp) {
        const int k = kp * VNNI_PAIR;
        const bool pair = k + 1 < K;
        const uint16_t *d = blk + static_cast<size_t>(kp) * pack_nr * VNNI_PAIR;
        for (int j = 0; j < pack_nr; ++j) {
            const size_t c
                    = static_cast<size_t>(source_col<Interleave>(p0 + j, half));
            if constexpr (TransB) {
                uint16_t *s = w + c * ldb + k;
                s[0] = d[2 * j];
                if (pair) s[1] = d[2 * j + 1];
            } else {
                w[static_cast<size_t>(k) * ldb + c] = d[2 * j];
                if (pair)
                    w[static_cast<size_t>(k + 1) * ldb + c] = d[2 * j + 1];
            }
        }
    }
}

/// fn(0) .. fn(parts - 1), each part on one thread.
template <typename Fn>
void for_each_part(int parts, int num_threads, Fn fn) {
    // The caller's team, which the other regions of its call use: a region
    // of another size makes the runtime resize the team, for it and again
    // for the call's next region.
    const int nt
            = omp_in_parallel() || parts < 2 ? 1 : std::max(1, num_threads);
#pragma omp parallel for num_threads(nt) schedule(static) if (nt > 1)
    for (int i = 0; i < parts; ++i)
        fn(i);
}

template <bool TransB, bool Interleave>
void pack_typed(data_type_t wei_dt, const void *weight, int K, int N,
        size_t ldb, int pack_nr, void *dst, int num_threads) {
    const size_t block_bytes = native_block_bytes(wei_dt, K, pack_nr);
    uint8_t *out = static_cast<uint8_t *>(dst);
    // The whole blocks; then, on the calling thread, the block of the last
    // N % pack_nr columns, padded with zero columns.
    const int n_blocks = N / pack_nr;
    const bool tail = n_blocks * pack_nr < N;
    uint8_t *tail_blk = out + static_cast<size_t>(n_blocks) * block_bytes;
    if (wei_dt == data_type_t::s8) {
        const int8_t *w = static_cast<const int8_t *>(weight);
        for_each_part(n_blocks, num_threads, [=](int nb) {
            pack_s8_block<TransB, Interleave, false>(w, K, N, ldb, pack_nr, nb,
                    reinterpret_cast<int8_t *>(out + nb * block_bytes));
        });
        if (tail)
            pack_s8_block<TransB, Interleave, true>(w, K, N, ldb, pack_nr,
                    n_blocks, reinterpret_cast<int8_t *>(tail_blk));
        return;
    }
    if (wei_dt == data_type_t::s4) {
        const uint8_t *w = static_cast<const uint8_t *>(weight);
        for_each_part(n_blocks, num_threads, [=](int nb) {
            pack_s4_block<TransB, Interleave, false>(
                    w, K, N, ldb, pack_nr, nb, out + nb * block_bytes);
        });
        if (tail)
            pack_s4_block<TransB, Interleave, true>(
                    w, K, N, ldb, pack_nr, n_blocks, tail_blk);
        return;
    }
    if (wei_dt == data_type_t::f32) {
        const float *w = static_cast<const float *>(weight);
        if (!TransB && K >= kTileMinK) {
            const int groups = ceil_div(n_blocks, kTileBlocks);
            const int chunks = ceil_div(K, kTileGroups);
            // Row chunks outer: the tiles of a thread take its rows in order.
            for_each_part(groups * chunks, num_threads, [=](int t) {
                const int g = t % groups;
                const int c = t / groups;
                pack_f32_tile_rows_of_k<Interleave>(w, ldb, N / 2,
                        c * kTileGroups, std::min(K, (c + 1) * kTileGroups),
                        g * kTileBlocks,
                        std::min(n_blocks, (g + 1) * kTileBlocks), block_bytes,
                        out);
            });
        } else {
            for_each_part(n_blocks, num_threads, [=](int nb) {
                pack_f32_block<TransB, Interleave, false>(w, K, N, ldb, pack_nr,
                        nb, reinterpret_cast<float *>(out + nb * block_bytes));
            });
        }
        if (tail)
            pack_f32_block<TransB, Interleave, true>(w, K, N, ldb, pack_nr,
                    n_blocks, reinterpret_cast<float *>(tail_blk));
        return;
    }
    const uint16_t *w = static_cast<const uint16_t *>(weight);
    if (!TransB && K >= kTileMinK) {
        const int k_pairs = native_k_groups(data_type_t::bf16, K);
        const int groups = ceil_div(n_blocks, kTileBlocks);
        const int chunks = ceil_div(k_pairs, kTileGroups);
        // K pair chunks outer: the tiles of a thread take its rows in order.
        for_each_part(groups * chunks, num_threads, [=](int t) {
            const int g = t % groups;
            const int c = t / groups;
            pack_bf16_tile_rows_of_k<Interleave>(w, K, ldb, N / 2,
                    c * kTileGroups, std::min(k_pairs, (c + 1) * kTileGroups),
                    g * kTileBlocks, std::min(n_blocks, (g + 1) * kTileBlocks),
                    block_bytes, out);
        });
    } else {
        for_each_part(n_blocks, num_threads, [=](int nb) {
            pack_bf16_block<TransB, Interleave, false>(w, K, N, ldb, pack_nr,
                    nb, reinterpret_cast<uint16_t *>(out + nb * block_bytes));
        });
    }
    if (tail)
        pack_bf16_block<TransB, Interleave, true>(w, K, N, ldb, pack_nr,
                n_blocks, reinterpret_cast<uint16_t *>(tail_blk));
}

template <bool TransB, bool Interleave>
void unpack_typed(const void *packed, int K, int N, size_t ldb, int pack_nr,
        void *weight, int num_threads) {
    const size_t block_bytes
            = native_block_bytes(data_type_t::bf16, K, pack_nr);
    const uint8_t *in = static_cast<const uint8_t *>(packed);
    uint16_t *w = static_cast<uint16_t *>(weight);
    for_each_part(N / pack_nr, num_threads, [=](int nb) {
        unpack_bf16_block<TransB, Interleave>(
                reinterpret_cast<const uint16_t *>(in + nb * block_bytes), K, N,
                ldb, pack_nr, nb, w);
    });
}

} // namespace

size_t native_packed_bytes(data_type_t wei_dt, int K, int N, int pack_nr) {
    assert(native_layout_fits(wei_dt, K, N, pack_nr));
    const size_t bytes = native_block_bytes(wei_dt, K, pack_nr)
            * static_cast<size_t>(native_padded_n(N, pack_nr) / pack_nr);
    return (bytes + 63) & ~size_t {63};
}

status_t native_pack_weight(data_type_t wei_dt, const void *weight, int K,
        int N, int ldb, int pack_nr, bool transB, bool interleave_split_halves,
        void *dst, int num_threads) {
    assert(native_layout_fits(wei_dt, K, N, pack_nr));
    if (weight == nullptr || dst == nullptr || ldb < (transB ? K : N)) {
        log_error("native pack: no native layout for wei=",
                zendnnl::common::dtype_info(wei_dt), " K=", K, " N=", N,
                " ldb=", ldb, " pack_nr=", pack_nr, " transB=", transB,
                " weight=", weight, " dst=", dst);
        return status_t::failure;
    }
    const size_t ld = static_cast<size_t>(ldb);
    if (transB) {
        if (interleave_split_halves)
            pack_typed<true, true>(
                    wei_dt, weight, K, N, ld, pack_nr, dst, num_threads);
        else
            pack_typed<true, false>(
                    wei_dt, weight, K, N, ld, pack_nr, dst, num_threads);
    } else {
        if (interleave_split_halves)
            pack_typed<false, true>(
                    wei_dt, weight, K, N, ld, pack_nr, dst, num_threads);
        else
            pack_typed<false, false>(
                    wei_dt, weight, K, N, ld, pack_nr, dst, num_threads);
    }
    return status_t::success;
}

void native_unpack_bf16(const void *packed, int K, int N, int ldb, int pack_nr,
        bool transB, bool interleave_split_halves, void *weight,
        int num_threads) {
    const size_t ld = static_cast<size_t>(ldb);
    if (transB) {
        if (interleave_split_halves)
            unpack_typed<true, true>(
                    packed, K, N, ld, pack_nr, weight, num_threads);
        else
            unpack_typed<true, false>(
                    packed, K, N, ld, pack_nr, weight, num_threads);
    } else {
        if (interleave_split_halves)
            unpack_typed<false, true>(
                    packed, K, N, ld, pack_nr, weight, num_threads);
        else
            unpack_typed<false, false>(
                    packed, K, N, ld, pack_nr, weight, num_threads);
    }
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
