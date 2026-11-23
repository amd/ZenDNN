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
#include <limits>

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

/// A block's columns: a K group row of 4-byte words is two vectors of 16
/// columns.
constexpr int kBlockCols = 32;
constexpr size_t kGroupRowBytes = kBlockCols * 4;
static_assert(kNativeLayoutPackNr == kBlockCols,
        "the pack writes blocks of 32 columns");

/// The geometries pack_block writes: words of one or two 2-byte elements,
/// of four 1-byte elements with column sums or of one 4-byte element, and
/// words of nibbles in two planes of a K quad each, two's complement plus 8.
/// 1-byte elements always take column sums: the paths of a weight stored
/// [K, N] write the sums after the quads, and the tile path, which takes
/// the words without sums, reads 2-byte elements.
constexpr bool packs(const native_geometry_t &g) {
    if (g.planes == 2)
        return g.elem_bits == 4 && g.word_bytes == 4
                && g.k_per_word == 2 * INT8_VNNI_GRP
                && g.encode == native_encode_t::nibble_offset && !g.col_sums;
    if (g.planes != 1 || g.encode != native_encode_t::copy
            || g.k_per_word * g.elem_bits != g.word_bytes * 8
            || g.col_sums != (g.elem_bits == 8))
        return false;
    return (g.elem_bits == 16 && g.k_per_word <= 2)
            || (g.elem_bits == 8 && g.k_per_word == INT8_VNNI_GRP)
            || (g.elem_bits == 32 && g.k_per_word == 1);
}
static_assert(!packs({8, 4, INT8_VNNI_GRP, 1, native_encode_t::copy, false}),
        "1-byte elements without column sums are not a geometry the pack "
        "writes");

/// Every format packs, and those that unpack hold 2-byte elements.
constexpr bool packs_every_format() {
    for (const native_format_desc_t &f : kNativeFormats)
        if (!packs(f.geometry)
                || (f.cache.unpack && f.geometry.elem_bits != 16))
            return false;
    return true;
}
static_assert(packs_every_format(), "a format the pack does not write");

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
/// lane of v, a word of one column as the caller stores it, in the layout's
/// two nibble planes: byte q holds n_q + 8 in its low nibble and n_(q + 4) +
/// 8 in its high nibble.  Zero bytes give the layout's zero, nibble 8.
ZENDNNL_TARGET("avx512f,avx512bw")
inline __m512i nibble_planes(__m512i v) {
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

/// The word permutes of 32 pack columns of groups of two 2-byte words (lo
/// columns 0 .. 15, hi 16 .. 31) into the two K rows the words hold, built
/// at compile time: v[0] makes the row of the first words, v[1] of the
/// second.
struct half_words_idx_t {
    alignas(64) uint16_t v[2][32] {};
    constexpr half_words_idx_t() {
        for (int j = 0; j < 32; ++j) {
            v[0][j] = static_cast<uint16_t>(2 * j);
            v[1][j] = static_cast<uint16_t>(2 * j + 1);
        }
    }
};

inline constexpr half_words_idx_t kHalfWordsIdx {};

/// The block of pack columns p0 .. p0 + 31 of a weight stored [N, K]
/// (transB): each pack column is a row of the buffer, row_bytes apart, of
/// which k_bytes are the weight's, read in groups of 4 bytes of consecutive
/// K values, groups of them.  WordBytes 4: each group is a word of the
/// layout, in nibble planes (nibble_planes) with Nibbles; WordBytes 2: each
/// group is two words, stored as the two K rows of 32 columns they make
/// (none past the k_bytes / 2 rows of the weight).  Sums: the column sums
/// of the words' signed bytes follow the K groups.  Tail: only the first
/// cols pack columns hold weight columns; the others are zero and their
/// rows are not read.
template <bool Tail, bool Sums, bool Nibbles, int WordBytes>
ZENDNNL_TARGET("avx512f,avx512bw")
void pack_block_rows_of_n(const uint8_t *w, size_t row_bytes, size_t k_bytes,
        int groups, int p0, uint8_t *blk, int cols) {
    constexpr size_t kRowBytes = kBlockCols * WordBytes;
    // Pack columns 0 .. 15 and 16 .. 31 of the block.
    const uint8_t *first[16], *second[16];
    const int n_first = Tail ? std::min(16, cols) : 16;
    const int n_second = Tail ? std::max(0, cols - 16) : 16;
    for (int i = 0; i < 16; ++i) {
        const int c0 = p0 + i;
        const int c1 = p0 + 16 + i;
        first[i] = i < n_first ? w + static_cast<size_t>(c0) * row_bytes : w;
        second[i] = i < n_second ? w + static_cast<size_t>(c1) * row_bytes : w;
    }
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
            if constexpr (Nibbles) {
                lo = nibble_planes(lo);
                hi = nibble_planes(hi);
            }
            if constexpr (WordBytes == 2) {
                const size_t k = 2 * static_cast<size_t>(g0 + q);
                uint8_t *d = blk + k * kRowBytes;
                _mm512_storeu_si512(d,
                        _mm512_permutex2var_epi16(
                                lo, _mm512_load_si512(kHalfWordsIdx.v[0]), hi));
                if (k + 1 < k_bytes / WordBytes)
                    _mm512_storeu_si512(d + kRowBytes,
                            _mm512_permutex2var_epi16(lo,
                                    _mm512_load_si512(kHalfWordsIdx.v[1]), hi));
                continue;
            }
            uint8_t *d = blk + static_cast<size_t>(g0 + q) * kRowBytes;
            _mm512_storeu_si512(d, lo);
            _mm512_storeu_si512(d + 64, hi);
            if constexpr (Sums) {
                sum_lo = _mm512_add_epi32(sum_lo, quad_sums(lo));
                sum_hi = _mm512_add_epi32(sum_hi, quad_sums(hi));
            }
        }
    }
    if constexpr (Sums) {
        uint8_t *d = blk + static_cast<size_t>(groups) * kRowBytes;
        _mm512_storeu_si512(d, sum_lo);
        _mm512_storeu_si512(d + 64, sum_hi);
    }
}

/// The lanes below lanes (0 .. 16) of a 16-lane mask.
inline __mmask16 lanes_below(int lanes) {
    return lanes >= 16
            ? static_cast<__mmask16>(0xFFFF)
            : static_cast<__mmask16>((1u << std::max(0, lanes)) - 1u);
}

/// Pack columns p0 .. p0 + 31 of a row of 2-byte elements stored [K, N].
/// Tail: only the first cols pack columns (cols below 32) hold weight
/// columns; the others read as zero.
template <bool Tail>
ZENDNNL_TARGET("avx512f,avx512bw")
inline __m512i load_words16(const uint8_t *row, int p0, int cols) {
    const uint16_t *r = reinterpret_cast<const uint16_t *>(row) + p0;
    if constexpr (Tail) {
        return _mm512_maskz_loadu_epi16((__mmask32 {1} << cols) - 1, r);
    } else {
        (void)cols;
        return _mm512_loadu_si512(r);
    }
}

/// The word permutes of a K pair row of 32 pack columns of 2-byte elements
/// stored [K, N], built at compile time: v[0] makes its first 16 columns,
/// v[1] its last 16.
struct pair_idx_t {
    alignas(64) uint16_t v[2][32] {};
    constexpr pair_idx_t() {
        // The pair of a column takes row k (index below 32) and row k + 1
        // (index 32 and up) of its column.
        for (int t = 0; t < 32; ++t) {
            const int from_next = (t & 1) ? 32 : 0;
            v[0][t] = static_cast<uint16_t>(from_next + t / 2);
            v[1][t] = static_cast<uint16_t>(from_next + 16 + t / 2);
        }
    }
};

inline constexpr pair_idx_t kPairIdx {};

/// K group row kg of pack columns p0 .. p0 + 31 of a weight of ElemBytes
/// elements stored [K, N], rows ld bytes apart, at d: with KPerWord 1 row kg
/// itself, with KPerWord 2 rows 2kg and 2kg + 1 in pairs (the second zero
/// past K).  Tail: only the first cols pack columns hold weight columns;
/// the others are zero.
template <int KPerWord, int ElemBytes, bool Tail>
ZENDNNL_TARGET("avx512f,avx512bw")
inline void pack_group_row(const uint8_t *w, int K, size_t ld, int p0, int kg,
        uint8_t *d, int cols) {
    const uint8_t *row = w + static_cast<size_t>(kg) * KPerWord * ld;
    if constexpr (KPerWord == 2) {
        static_assert(ElemBytes == 2, "pairs of 2-byte elements");
        const __m512i a = load_words16<Tail>(row, p0, cols);
        const __m512i b = KPerWord * kg + 1 < K
                ? load_words16<Tail>(row + ld, p0, cols)
                : _mm512_setzero_si512();
        _mm512_storeu_si512(d,
                _mm512_permutex2var_epi16(
                        a, _mm512_load_si512(kPairIdx.v[0]), b));
        _mm512_storeu_si512(d + 64,
                _mm512_permutex2var_epi16(
                        a, _mm512_load_si512(kPairIdx.v[1]), b));
    } else if constexpr (ElemBytes == 4) {
        static_assert(KPerWord == 1, "a word of one 4-byte element");
        const float *r = reinterpret_cast<const float *>(row) + p0;
        float *o = reinterpret_cast<float *>(d);
        if constexpr (Tail) {
            _mm512_storeu_ps(o, _mm512_maskz_loadu_ps(lanes_below(cols), r));
            _mm512_storeu_ps(o + 16,
                    _mm512_maskz_loadu_ps(lanes_below(cols - 16), r + 16));
        } else {
            _mm512_storeu_ps(o, _mm512_loadu_ps(r));
            _mm512_storeu_ps(o + 16, _mm512_loadu_ps(r + 16));
        }
    } else {
        static_assert(KPerWord == 1 && ElemBytes == 2,
                "a word of one 2-byte element");
        _mm512_storeu_si512(d, load_words16<Tail>(row, p0, cols));
    }
}

/// The block of pack columns p0 .. p0 + 31 of a weight stored [K, N], rows
/// ld bytes apart: its K group rows (pack_group_row).
template <int KPerWord, int ElemBytes, bool Tail>
ZENDNNL_TARGET("avx512f,avx512bw")
void pack_block_rows_of_k(
        const uint8_t *w, int K, size_t ld, int p0, uint8_t *blk, int cols) {
    constexpr size_t kRowBytes = kBlockCols * KPerWord * ElemBytes;
    const int groups = K / KPerWord + (K % KPerWord != 0);
    for (int kg = 0; kg < groups; ++kg)
        pack_group_row<KPerWord, ElemBytes, Tail>(w, K, ld, p0, kg,
                blk + static_cast<size_t>(kg) * kRowBytes, cols);
}

/// Pack columns p0 .. p0 + 31 of a row of one-byte elements stored [K, N].
/// Tail: only the first cols pack columns (cols below 32) hold weight
/// columns; the others read as zero.
template <bool Tail>
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl")
inline __m256i load_bytes(const uint8_t *row, int p0, int cols) {
    if constexpr (Tail) {
        return _mm256_maskz_loadu_epi8((__mmask32 {1} << cols) - 1, row + p0);
    } else {
        (void)cols;
        return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(row + p0));
    }
}

/// The block of pack columns p0 .. p0 + 31 of a weight of one-byte elements
/// stored [K, N], ld bytes apart, in words of K quads, their column sums
/// after them; Tail as load_bytes (the zero columns sum to zero).
template <bool Tail>
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl")
void pack_block_quads_of_k(
        const uint8_t *w, int K, size_t ld, int p0, uint8_t *blk, int cols) {
    const int k_quads = K / INT8_VNNI_GRP + (K % INT8_VNNI_GRP != 0);
    __m256i sums[4];
    for (__m256i &s : sums)
        s = _mm256_setzero_si256();
    for (int kq = 0; kq < k_quads; ++kq) {
        const int k = kq * INT8_VNNI_GRP;
        __m256i r[4];
        for (int i = 0; i < 4; ++i)
            r[i] = k + i < K
                    ? load_bytes<Tail>(
                              w + static_cast<size_t>(k + i) * ld, p0, cols)
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
        uint8_t *dst = blk + static_cast<size_t>(kq) * kGroupRowBytes;
        for (int i = 0; i < 4; ++i) {
            _mm256_storeu_si256(
                    reinterpret_cast<__m256i *>(dst + 32 * i), quads[i]);
            sums[i] = _mm256_add_epi32(sums[i], quad_sums(quads[i]));
        }
    }
    uint8_t *dst = blk + static_cast<size_t>(k_quads) * kGroupRowBytes;
    for (int i = 0; i < 4; ++i)
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst + 32 * i), sums[i]);
}

/// K values of a word of nibbles: two planes of a K quad.
constexpr int kNibbleWordK = 2 * INT8_VNNI_GRP;

/// The two's-complement nibble of element (k, n) of a weight of nibbles
/// stored [K, N] ldb elements apart, or [N, K] when transB.
inline uint8_t nibble_at(
        const uint8_t *w, size_t ldb, bool transB, int k, int n) {
    const size_t i = transB
            ? static_cast<size_t>(n) * ldb + static_cast<size_t>(k)
            : static_cast<size_t>(k) * ldb + static_cast<size_t>(n);
    return static_cast<uint8_t>((w[i >> 1] >> ((i & 1) * 4)) & 0x0F);
}

/// The block of pack columns p0 .. p0 + 31 of a weight of nibbles read
/// element by element (any ldb, either storage), as nibble_planes writes a
/// word; pack columns from N on and K past the weight's hold the layout's
/// zero.
void pack_block_of_nibbles(const uint8_t *w, int K, int N, size_t ldb,
        bool transB, int p0, uint8_t *blk) {
    const int words = K / kNibbleWordK + (K % kNibbleWordK != 0);
    for (int o = 0; o < words; ++o) {
        uint8_t *row = blk + static_cast<size_t>(o) * kGroupRowBytes;
        for (int q = 0; q < 4; ++q) {
            const int k0 = kNibbleWordK * o + q;
            const int k1 = k0 + kNibbleWordK / 2;
            for (int j = 0; j < kBlockCols; ++j) {
                const int p = p0 + j;
                uint8_t lo = 0, hi = 0;
                if (p < N) {
                    if (k0 < K) lo = nibble_at(w, ldb, transB, k0, p);
                    if (k1 < K) hi = nibble_at(w, ldb, transB, k1, p);
                }
                row[j * 4 + q] = static_cast<uint8_t>(
                        (lo ^ 0x08u) | ((hi ^ 0x08u) << 4));
            }
        }
    }
}

/// The block nb of a weight of geometry g, stored [K, N] ldb elements apart
/// or [N, K] when TransB; Tail: the last block, of fewer than 32 columns of
/// the weight (pack columns from N on are zero).
template <bool TransB, bool Tail>
void pack_block(const native_geometry_t &g, const uint8_t *w, int K, int N,
        size_t ldb, int nb, uint8_t *blk) {
    const int p0 = nb * kBlockCols;
    const int cols = N - p0;
    if (g.planes == 2) {
        // Rows of whole bytes as groups of 4 bytes, else element by element.
        if (TransB && ldb % 2 == 0 && K % 2 == 0)
            pack_block_rows_of_n<Tail, false, true, 4>(w, ldb / 2,
                    static_cast<size_t>(K) / 2, native_k_groups(g, K), p0, blk,
                    cols);
        else
            pack_block_of_nibbles(w, K, N, ldb, TransB, p0, blk);
        return;
    }
    const size_t esz = static_cast<size_t>(g.elem_bits / 8);
    if constexpr (TransB) {
        const size_t k_bytes = static_cast<size_t>(K) * esz;
        const int groups = static_cast<int>((k_bytes + 3) / 4);
        if (g.word_bytes == 2)
            pack_block_rows_of_n<Tail, false, false, 2>(
                    w, ldb * esz, k_bytes, groups, p0, blk, cols);
        else if (g.col_sums)
            pack_block_rows_of_n<Tail, true, false, 4>(
                    w, ldb * esz, k_bytes, groups, p0, blk, cols);
        else
            pack_block_rows_of_n<Tail, false, false, 4>(
                    w, ldb * esz, k_bytes, groups, p0, blk, cols);
    } else if (g.k_per_word == INT8_VNNI_GRP) {
        pack_block_quads_of_k<Tail>(w, K, ldb, p0, blk, cols);
    } else if (g.k_per_word == 2) {
        pack_block_rows_of_k<2, 2, Tail>(w, K, ldb * 2, p0, blk, cols);
    } else if (esz == 4) {
        pack_block_rows_of_k<1, 4, Tail>(w, K, ldb * 4, p0, blk, cols);
    } else {
        pack_block_rows_of_k<1, 2, Tail>(w, K, ldb * 2, p0, blk, cols);
    }
}

/// A weight stored [K, N] of at least kTileMinK rows, of words of one
/// element or of K pairs, packs by tiles of kTileGroups K groups by
/// kTileBlocks blocks.  A block reads one or two lines of each of its rows,
/// a row stride apart: a walk down that many rows loses the lines it fetches
/// ahead (all in one cache set when the stride is a multiple of the page
/// size).  A tile reads the segments of its rows in order and writes the
/// part of each block whole.
constexpr int kTileGroups = 8;
constexpr int kTileBlocks = 16;
constexpr int kTileMinK = 1024;

/// K groups kg0 .. kg1 (at most kTileGroups) of blocks nb0 .. nb1 (at most
/// kTileBlocks) of a weight stored [K, N], rows ld bytes apart, into the
/// layout at out, whose blocks are block_bytes long.
template <int KPerWord, int ElemBytes>
ZENDNNL_TARGET("avx512f,avx512bw")
void pack_tile_rows_of_k(const uint8_t *w, int K, size_t ld, int kg0, int kg1,
        int nb0, int nb1, size_t block_bytes, uint8_t *out) {
    constexpr size_t kRowBytes = kBlockCols * KPerWord * ElemBytes;
    alignas(64) uint8_t tile[kTileBlocks][kTileGroups * kRowBytes];
    for (int kg = kg0; kg < kg1; ++kg)
        for (int nb = nb0; nb < nb1; ++nb)
            pack_group_row<KPerWord, ElemBytes, false>(w, K, ld,
                    nb * kBlockCols, kg,
                    tile[nb - nb0] + (kg - kg0) * kRowBytes, kBlockCols);
    const size_t part = static_cast<size_t>(kg1 - kg0) * kRowBytes;
    for (int nb = nb0; nb < nb1; ++nb) {
        const uint8_t *s = tile[nb - nb0];
        uint8_t *d = out + static_cast<size_t>(nb) * block_bytes
                + static_cast<size_t>(kg0) * kRowBytes;
        for (size_t i = 0; i < part; i += 64)
            _mm512_storeu_si512(d + i, _mm512_load_si512(s + i));
    }
}

/// Tile t of a weight of geometry g stored [K, N] ldb elements apart, of
/// n_blocks whole blocks: K group chunks outer, so that the tiles of a
/// thread take its rows in order.
void pack_tile(const native_geometry_t &g, const uint8_t *w, int K, size_t ldb,
        int t, int n_blocks, size_t block_bytes, uint8_t *out) {
    const int groups = ceil_div(n_blocks, kTileBlocks);
    const int k_groups = native_k_groups(g, K);
    const int kg0 = t / groups * kTileGroups;
    const int kg1 = std::min(k_groups, kg0 + kTileGroups);
    const int nb0 = t % groups * kTileBlocks;
    const int nb1 = std::min(n_blocks, nb0 + kTileBlocks);
    if (g.k_per_word == 2)
        pack_tile_rows_of_k<2, 2>(
                w, K, ldb * 2, kg0, kg1, nb0, nb1, block_bytes, out);
    else if (g.elem_bits == 32)
        pack_tile_rows_of_k<1, 4>(
                w, K, ldb * 4, kg0, kg1, nb0, nb1, block_bytes, out);
    else
        pack_tile_rows_of_k<1, 2>(
                w, K, ldb * 2, kg0, kg1, nb0, nb1, block_bytes, out);
}

/// Inverse of the pack of block nb of a weight of 2-byte elements in words
/// of KPerWord K values: each element the block holds of the weight's K,
/// back to the weight.  The word width is a template argument: with a width
/// known only at run time the copies of a row do not vectorize.
template <bool TransB, int KPerWord>
void unpack_block(const uint16_t *blk, int K, size_t ldb, int pack_nr, int nb,
        uint16_t *w) {
    static_assert(KPerWord == 1 || KPerWord == 2,
            "words of one or two 2-byte elements");
    const int p0 = nb * pack_nr;
    const int groups = K / KPerWord + (K % KPerWord != 0);
    for (int kg = 0; kg < groups; ++kg) {
        const int k = kg * KPerWord;
        // The second element of the last word of an odd K is past the
        // weight.
        const bool second = KPerWord == 2 && k + 1 < K;
        const uint16_t *d = blk + static_cast<size_t>(kg) * pack_nr * KPerWord;
        for (int j = 0; j < pack_nr; ++j) {
            const size_t c = static_cast<size_t>(p0 + j);
            if constexpr (TransB) {
                uint16_t *s = w + c * ldb + k;
                s[0] = d[KPerWord * j];
                if (second) s[1] = d[KPerWord * j + 1];
            } else {
                w[static_cast<size_t>(k) * ldb + c] = d[KPerWord * j];
                if (second)
                    w[static_cast<size_t>(k + 1) * ldb + c]
                            = d[KPerWord * j + 1];
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
    native_for(parts, num_threads, parts > 1 && !omp_in_parallel(), fn);
}

template <bool TransB>
void pack_layout(const native_geometry_t &g, const void *weight, int K, int N,
        size_t ldb, int pack_nr, void *dst, int num_threads) {
    const size_t block_bytes = native_block_bytes(g, K, pack_nr);
    const uint8_t *w = static_cast<const uint8_t *>(weight);
    uint8_t *out = static_cast<uint8_t *>(dst);
    // The whole blocks; then, on the calling thread, the block of the last
    // N % pack_nr columns, padded with zero columns.
    const int n_blocks = N / pack_nr;
    const bool tail = n_blocks * pack_nr < N;
    const int tiles = !TransB && g.planes == 1 && !g.col_sums && K >= kTileMinK
            ? native_pack_tiles(n_blocks, native_k_groups(g, K))
            : 0;
    if (tiles > 0) {
        for_each_part(tiles, num_threads, [=](int t) {
            pack_tile(g, w, K, ldb, t, n_blocks, block_bytes, out);
        });
    } else {
        for_each_part(n_blocks, num_threads, [=](int nb) {
            pack_block<TransB, false>(
                    g, w, K, N, ldb, nb, out + nb * block_bytes);
        });
    }
    if (tail)
        pack_block<TransB, true>(g, w, K, N, ldb, n_blocks,
                out + static_cast<size_t>(n_blocks) * block_bytes);
}

template <bool TransB, int KPerWord>
void unpack_layout(const native_geometry_t &g, const void *packed, int K, int N,
        size_t ldb, int pack_nr, void *weight, int num_threads) {
    const size_t block_bytes = native_block_bytes(g, K, pack_nr);
    const uint8_t *in = static_cast<const uint8_t *>(packed);
    uint16_t *w = static_cast<uint16_t *>(weight);
    for_each_part(N / pack_nr, num_threads, [=](int nb) {
        unpack_block<TransB, KPerWord>(
                reinterpret_cast<const uint16_t *>(in + nb * block_bytes), K,
                ldb, pack_nr, nb, w);
    });
}

} // namespace

int native_pack_tiles(int n_blocks, int k_groups) {
    const int64_t tiles = int64_t {ceil_div(n_blocks, kTileBlocks)}
            * ceil_div(k_groups, kTileGroups);
    return tiles <= std::numeric_limits<int>::max() ? static_cast<int>(tiles)
                                                    : 0;
}

size_t native_packed_bytes(data_type_t wei_dt, int K, int N, int pack_nr) {
    assert(native_layout_fits(wei_dt, K, N, pack_nr));
    assert(pack_nr == kNativeLayoutPackNr);
    const size_t blocks = (static_cast<size_t>(N) + kNativeLayoutPackNr - 1)
            / kNativeLayoutPackNr;
    const size_t bytes = native_block_bytes(wei_dt, K, pack_nr) * blocks;
    return (bytes + 63) & ~size_t {63};
}

status_t native_pack_weight(data_type_t wei_dt, const void *weight, int K,
        int N, int ldb, int pack_nr, bool transB, void *dst, int num_threads) {
    assert(native_layout_fits(wei_dt, K, N, pack_nr));
    if (weight == nullptr || dst == nullptr || ldb < (transB ? K : N)) {
        log_error("native pack: no native layout for wei=",
                zendnnl::common::dtype_info(wei_dt), " K=", K, " N=", N,
                " ldb=", ldb, " pack_nr=", pack_nr, " transB=", transB,
                " weight=", weight, " dst=", dst);
        return status_t::failure;
    }
    const native_geometry_t g = native_geometry(wei_dt);
    const size_t ld = static_cast<size_t>(ldb);
    if (transB)
        pack_layout<true>(g, weight, K, N, ld, pack_nr, dst, num_threads);
    else
        pack_layout<false>(g, weight, K, N, ld, pack_nr, dst, num_threads);
    return status_t::success;
}

void native_unpack_weight(data_type_t wei_dt, const void *packed, int K, int N,
        int ldb, int pack_nr, bool transB, void *weight, int num_threads) {
    assert(native_format(wei_dt).cache.unpack);
    const native_geometry_t g = native_geometry(wei_dt);
    const auto unpack = g.k_per_word == 2
            ? (transB ? unpack_layout<true, 2> : unpack_layout<false, 2>)
            : (transB ? unpack_layout<true, 1> : unpack_layout<false, 1>);
    unpack(g, packed, K, N, static_cast<size_t>(ldb), pack_nr, weight,
            num_threads);
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
