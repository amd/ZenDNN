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

/// The INT8 policy of `run_tiled`. Every scale is local to what one
/// tile reads, so quantization needs no pass of its own: Q per token in the
/// Q tile copy, K per token and V per (key block, channel) in the pack loop,
/// and P per (row, key block) right after the tile's Q*K^T of the block, with
/// the quantizers of `hstu_quant/`.

#include "lowoha_operators/hstu_attention/tiled/lowoha_hstu_tiled.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <immintrin.h>
#include <utility>
#include "common/bfloat16.hpp"
#include "common/zendnnl_global.hpp"
#include "lowoha_operators/hstu_attention/common/lowoha_hstu_select.hpp"
#include "lowoha_operators/hstu_attention/hstu_gemm_ck/lowoha_hstu_ck.hpp"
#include "lowoha_operators/hstu_attention/hstu_quant/lowoha_hstu_quant.hpp"
#include "lowoha_operators/hstu_attention/tiled/lowoha_hstu_tiled_common.hpp"
#include "lowoha_operators/hstu_attention/tiled/lowoha_hstu_tiled_driver.hpp"
#include "lowoha_operators/matmul/group_matmul/common/group_matmul_act_avx512.hpp"
#include <type_traits>

namespace zendnnl {
namespace lowoha {
namespace hstu_attention {

namespace {

using zendnnl::common::bfloat16_t;
namespace act = zendnnl::lowoha::matmul::group_matmul_act_avx512;

int64_t round4(int64_t n) {
    return (n + 3) / 4 * 4;
}

/// Entries of per-column arrays of packed blocks (column sums, scales).
int64_t round32(int64_t n) {
    return (n + kNR - 1) / kNR * kNR;
}

/// V of a block (nb rows `ldv` apart, dv columns) to S8 per channel, 32
/// channels at a time through `stage` (nb x 32 bytes), each packed with
/// `ck::int8::pack_kn`; writes the column sums and scales (round32(dv) each,
/// the scales 0 past dv).
void pack_v(const bfloat16_t *v, int64_t ldv, int64_t nb, int64_t dv,
        int8_t *stage, int8_t *packed, int32_t *sum, float *scale) {
    const int64_t block = ck::int8::packed_size(nb, kNR);
    for (int64_t c = 0; c < dv; c += kNR) {
        const int64_t n = std::min<int64_t>(kNR, dv - c);
        quant::quantize_cols(v + c, ldv, nb, n, stage, kNR, scale + c);
        ck::int8::pack_kn(nb, n, stage, kNR, packed + c / kNR * block, sum + c);
    }
}

/// What a Q*K^T tile needs besides its operands, from its first row and
/// column on.
struct qk_epilogue_t {
    const int32_t *lo, *hi; ///< per row: the kept columns [lo, hi)
    const float *row_scale; ///< INT8 Q, per row: Q scale * alpha
    float alpha; ///< BF16 Q
    const int32_t *colsum; ///< INT8 Q, per column: sum of the S8 K row
    const float *col_scale; ///< INT8 Q, per column: K scale
    const float *scale; ///< BF16 P, per row: scale(i)
    void *p; ///< P, rows kBlockN apart: BF16 or FP32
    float *amax; ///< FP32 P: 16 lanes of max |P| per row
};

/// x = alpha * A B for INT8 Q: A = Q + 128 (rows of k bytes, k % 4 == 0)
/// against S8 K^T, the offset removed with the K column sums and the sums
/// dequantized by the Q and K scales.
template <int MR, int NB>
HSTU_I8_INLINE inline void scores(__m512 (&x)[MR][2 * NB],
        const uint8_t *__restrict a, int64_t lda, const int8_t *__restrict b,
        int64_t bstride, int64_t k, const qk_epilogue_t &e) {
    __m512i acc[MR][2 * NB];
#pragma GCC unroll 8
    for (int m = 0; m < MR; ++m)
        for (int j = 0; j < 2 * NB; ++j)
            acc[m][j] = _mm512_setzero_si512();
    const int64_t kq = k / 4;
    int64_t q = 0;
    for (; q + 1 < kq; q += 2) {
        quad_step<MR, NB>(acc, a, lda, b, bstride, q);
        quad_step<MR, NB>(acc, a, lda, b, bstride, q + 1);
    }
    if (q < kq) quad_step<MR, NB>(acc, a, lda, b, bstride, q);

    // The column sums and scales cover whole packed blocks.
    __m512i comp[2 * NB];
    __m512 cs[2 * NB];
#pragma GCC unroll 4
    for (int j = 0; j < 2 * NB; ++j) {
        comp[j] = _mm512_slli_epi32(_mm512_loadu_si512(e.colsum + 16 * j), 7);
        cs[j] = _mm512_loadu_ps(e.col_scale + 16 * j);
    }
#pragma GCC unroll 8
    for (int m = 0; m < MR; ++m) {
        const __m512 rs = _mm512_set1_ps(e.row_scale[m]);
#pragma GCC unroll 4
        for (int j = 0; j < 2 * NB; ++j)
            x[m][j] = _mm512_mul_ps(
                    _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[m][j], comp[j])),
                    _mm512_mul_ps(cs[j], rs));
    }
}

/// x = alpha * A B for BF16 Q (rows of k elements) against BF16 K^T.
template <int MR, int NB>
HSTU_I8_INLINE inline void scores(__m512 (&x)[MR][2 * NB],
        const bfloat16_t *__restrict a, int64_t lda,
        const bfloat16_t *__restrict b, int64_t bstride, int64_t k,
        const qk_epilogue_t &e) {
#pragma GCC unroll 8
    for (int m = 0; m < MR; ++m)
        for (int j = 0; j < 2 * NB; ++j)
            x[m][j] = _mm512_setzero_ps();
    const int64_t full = k / 2;
    int64_t q = 0;
    for (; q + 1 < full; q += 2) {
        pair_step<MR, NB>(x, a, lda, b, bstride, q, false);
        pair_step<MR, NB>(x, a, lda, b, bstride, q + 1, false);
    }
    if (q < full) pair_step<MR, NB>(x, a, lda, b, bstride, q, false);
    if (k & 1) pair_step<MR, NB>(x, a, lda, b, bstride, full, true);
    const __m512 va = _mm512_set1_ps(e.alpha);
#pragma GCC unroll 8
    for (int m = 0; m < MR; ++m)
        for (int j = 0; j < 2 * NB; ++j)
            x[m][j] = _mm512_mul_ps(x[m][j], va);
}

/// silu(x): `act::silu_avx512` or, with FastSilu, `fast_silu_scaled`.
template <bool FastSilu>
HSTU_I8_INLINE inline __m512 silu(__m512 x) {
    if constexpr (FastSilu)
        return fast_silu_scaled(
                x, _mm512_set1_ps(-1.44269504f), _mm512_set1_ps(1.f));
    else
        return act::silu_avx512(x);
}

/// silu(x) * s.
template <bool FastSilu>
HSTU_I8_INLINE inline __m512 silu_scaled(__m512 x, __m512 s) {
    if constexpr (FastSilu)
        return fast_silu_scaled(x, _mm512_set1_ps(-1.44269504f), s);
    else
        return _mm512_mul_ps(act::silu_avx512(x), s);
}

/// One tile of the fused Q*K^T over NB packed blocks of K^T: P[MR][nv] =
/// mask * silu(x), stored as FP32 with the running max |P| of each row
/// (`F32P`), or as BF16(mask * silu(x) * scale). Row m keeps columns
/// [lo[m], hi[m]) of the tile; a tile with no kept column skips the GEMM.
template <int MR, int NB, typename a_t, typename b_t, bool F32P, bool FastSilu>
HSTU_I8_NOINLINE void qk_silu_tile(const a_t *__restrict a, int64_t lda,
        const b_t *__restrict b, int64_t bstride, int64_t k, int nv,
        const qk_epilogue_t &e) {
    const uint64_t cols = range_bits(0, nv);
    uint64_t keep[MR];
    uint64_t any = 0;
    for (int m = 0; m < MR; ++m) {
        keep[m] = range_bits(e.lo[m], e.hi[m]) & cols;
        any |= keep[m];
    }
    float *pf = static_cast<float *>(e.p);
    bfloat16_t *pb = static_cast<bfloat16_t *>(e.p);
    if (!any) {
        for (int m = 0; m < MR; ++m)
            for (int j = 0; j < 2 * NB; ++j) {
                const __mmask16 c16 = __mmask16(cols >> (16 * j));
                if constexpr (F32P)
                    _mm512_mask_storeu_ps(pf + m * kBlockN + 16 * j, c16,
                            _mm512_setzero_ps());
                else
                    _mm256_mask_storeu_epi16(pb + m * kBlockN + 16 * j, c16,
                            _mm256_setzero_si256());
            }
        return;
    }

    __m512 x[MR][2 * NB];
    scores<MR, NB>(x, a, lda, b, bstride, k, e);
#pragma GCC unroll 8
    for (int m = 0; m < MR; ++m) {
        if constexpr (F32P) {
            __m512 am = _mm512_loadu_ps(e.amax + 16 * m);
#pragma GCC unroll 4
            for (int j = 0; j < 2 * NB; ++j) {
                const __m512 y
                        = _mm512_maskz_mov_ps(__mmask16(keep[m] >> (16 * j)),
                                silu<FastSilu>(x[m][j]));
                _mm512_mask_storeu_ps(pf + m * kBlockN + 16 * j,
                        __mmask16(cols >> (16 * j)), y);
                am = _mm512_max_ps(am, _mm512_abs_ps(y));
            }
            _mm512_storeu_ps(e.amax + 16 * m, am);
        } else {
            const __m512 vs = _mm512_set1_ps(e.scale[m]);
#pragma GCC unroll 2
            for (int j = 0; j < NB; ++j) {
                const __m512 y0 = silu_scaled<FastSilu>(x[m][2 * j], vs);
                const __m512 y1 = silu_scaled<FastSilu>(x[m][2 * j + 1], vs);
                const __m512i v = (__m512i)_mm512_cvtne2ps_pbh(y1, y0);
                _mm512_mask_storeu_epi16(pb + m * kBlockN + j * kNR,
                        __mmask32(cols >> (32 * j)),
                        _mm512_maskz_mov_epi16(
                                __mmask32(keep[m] >> (32 * j)), v));
            }
        }
    }
}

template <typename a_t, typename b_t>
using qk_tile_fn = void (*)(const a_t *, int64_t, const b_t *, int64_t, int64_t,
        int, const qk_epilogue_t &);

template <typename a_t, typename b_t, bool F32P, bool FastSilu, int NB,
        int... M>
constexpr std::array<qk_tile_fn<a_t, b_t>, sizeof...(M) + 1> qk_tiles(
        std::integer_sequence<int, M...>) {
    return {nullptr, qk_silu_tile<M + 1, NB, a_t, b_t, F32P, FastSilu>...};
}

/// Tile kernels by row count, over one or two packed blocks.
template <typename a_t, typename b_t, bool F32P, bool FastSilu>
struct qk_tile_table_t {
    static constexpr auto one = qk_tiles<a_t, b_t, F32P, FastSilu, 1>(
            std::make_integer_sequence<int, kMR1>());
    static constexpr auto two = qk_tiles<a_t, b_t, F32P, FastSilu, 2>(
            std::make_integer_sequence<int, kMR2>());
};

/**
 * @brief The dynamic-INT8 policy of `run_tiled`.
 *
 * `QkInt8`: Q*K^T on INT8; Q tile = the per-row Q scales (alpha folded in),
 * then rows of Q + 128 padded to whole quads. Otherwise BF16, as the BF16
 * kernel.
 * `PvInt8`: P*V on INT8. The Q*K^T epilogue writes FP32 P = mask * silu
 * and its row max; P is then quantized per (row, block) to S8 + 128 and
 * multiplied by S8 V on `ck::int8::gemm_block`, which removes the offset
 * with the V column sums and dequantizes by P scale * scale(i) and the V
 * scales, adding to O. Otherwise BF16 P (with scale(i)) and the BF16 P*V.
 * `FastSilu`: the Q*K^T epilogue uses `fast_silu_scaled`, as the BF16
 * kernel with `fast_silu`.
 *
 * Slot of a block of nb keys: K^T (S8 packed by `ck::int8::pack_kt`, the
 * column sums and scales of its keys, round32(nb) each; or BF16), then V
 * (S8 packed, the column sums and scales of its channels, round32(dv)
 * each; or BF16).
 */
template <bool QkInt8, bool PvInt8, bool FastSilu>
struct int8_policy_t {
    using qa_t = std::conditional_t<QkInt8, uint8_t, bfloat16_t>;
    using kb_t = std::conditional_t<QkInt8, int8_t, bfloat16_t>;
    using vb_t = std::conditional_t<PvInt8, int8_t, bfloat16_t>;
    using tiles = qk_tile_table_t<qa_t, kb_t, PvInt8, FastSilu>;

    /// Packed V holds key quads (INT8) or pairs (BF16).
    static constexpr int64_t kKeyAlign = PvInt8 ? 4 : 2;
    static constexpr size_t kQScaleBytes
            = QkInt8 ? kMaxBlockM * sizeof(float) : 0;
    /// INT8 P: 16 lanes of max |P| and the P*V scale per row, the S8 P rows,
    /// then the FP32 P rows.
    static constexpr size_t kAmaxBytes = kMaxBlockM * 16 * sizeof(float);
    static constexpr size_t kPvScaleBytes = kMaxBlockM * sizeof(float);
    static constexpr size_t kPqBytes = kMaxBlockM * kBlockN;

    static size_t q_bytes(int64_t bm, int64_t d) {
        return kQScaleBytes + bm * (QkInt8 ? round4(d) : d * 2);
    }
    static constexpr size_t p_bytes(int64_t bm) {
        if (!PvInt8) return bm * kBlockN * sizeof(bfloat16_t);
        return kAmaxBytes + kPvScaleBytes + kPqBytes
                + bm * kBlockN * sizeof(float);
    }

    int8_policy_t(const hstu_attention_params &p, const bfloat16_t *q,
            const bfloat16_t *k, const bfloat16_t *v)
        : p(p)
        , q(q)
        , k(k)
        , v(v)
        , D(p.qk_dim)
        , Dv(p.v_dim)
        , q_ld(QkInt8 ? round4(p.qk_dim) : p.qk_dim)
        , k_block(QkInt8 ? ck::int8::packed_size(p.qk_dim, kNR)
                         : (p.qk_dim + 1) / 2 * 2 * kNR) {}

    /// Pointers into a slot of nb keys.
    struct slot_t {
        kb_t *k;
        int32_t *k_sum;
        float *k_scale;
        vb_t *v;
        int32_t *v_sum;
        float *v_scale;
    };
    size_t k_bytes(int64_t nb) const {
        if (!QkInt8) return round_up(ck::bf16::packed_size(D, nb) * 2);
        return round_up(ck::int8::packed_size(D, nb))
                + 2 * round_up(round32(nb) * 4);
    }
    size_t v_bytes(int64_t nb) const {
        if (!PvInt8) return ck::bf16::packed_size(nb, Dv) * 2;
        return round_up(ck::int8::packed_size(nb, Dv))
                + 2 * round_up(round32(Dv) * 4);
    }
    slot_t layout(const char *slot, int64_t nb) const {
        char *c = const_cast<char *>(slot);
        slot_t s {};
        s.k = reinterpret_cast<kb_t *>(c);
        if constexpr (QkInt8) {
            const size_t pack = round_up(ck::int8::packed_size(D, nb));
            s.k_sum = reinterpret_cast<int32_t *>(c + pack);
            s.k_scale = reinterpret_cast<float *>(
                    c + pack + round_up(round32(nb) * 4));
        }
        c += k_bytes(nb);
        s.v = reinterpret_cast<vb_t *>(c);
        if constexpr (PvInt8) {
            const size_t pack = round_up(ck::int8::packed_size(nb, Dv));
            s.v_sum = reinterpret_cast<int32_t *>(c + pack);
            s.v_scale = reinterpret_cast<float *>(
                    c + pack + round_up(round32(Dv) * 4));
        }
        return s;
    }
    size_t slot_bytes(int64_t nb) const { return k_bytes(nb) + v_bytes(nb); }

    /// INT8 K per token, 32 keys at a time quantized into the scratch tile's
    /// Q area (bm >= 32 rows of at least D bytes); INT8 V through its P area
    /// (at least kBlockN x 32 bytes).
    void pack(const kv_block_t &blk, int64_t h, char *slot,
            const tile_t &scratch) const {
        const int64_t k0 = p.seq_offsets[blk.b] + blk.k0;
        const slot_t s = layout(slot, blk.nb);
        const bfloat16_t *kb = k + k0 * p.k_stride_t + h * p.k_stride_h;
        const bfloat16_t *vb = v + k0 * p.v_stride_t + h * p.v_stride_h;
        if constexpr (QkInt8) {
            auto *stage = static_cast<int8_t *>(scratch.q);
            for (int64_t c = 0; c < blk.nb; c += kNR) {
                const int64_t n = std::min<int64_t>(kNR, blk.nb - c);
                for (int64_t r = 0; r < kNR; ++r)
                    s.k_scale[c + r] = r < n
                            ? quant::quantize_row(kb + (c + r) * p.k_stride_t,
                                      D, 0,
                                      reinterpret_cast<uint8_t *>(
                                              stage + r * D))
                            : 0.f;
                ck::int8::pack_kt(
                        D, n, stage, D, s.k + c / kNR * k_block, s.k_sum + c);
            }
        } else {
            ck::bf16::pack_b(D, blk.nb, kb, p.k_stride_t, true, s.k);
        }
        if constexpr (PvInt8)
            pack_v(vb, p.v_stride_t, blk.nb, Dv,
                    static_cast<int8_t *>(scratch.p), s.v, s.v_sum, s.v_scale);
        else
            ck::bf16::pack_b(blk.nb, Dv, vb, p.v_stride_t, false, s.v);
    }

    /// INT8: Q per token as U8 Q + 128 rows of q_ld bytes (the padding is a
    /// zero), with alpha folded into the scale. BF16: a copy, so that rows
    /// of a strided view do not alias in L1.
    void load_q(const work_item_t &w, const tile_t &t) const {
        qa_t *q_tile = q_rows(t);
        const bfloat16_t *qt
                = q + (w.s.q_row + w.row0) * p.q_stride_t + w.h * p.q_stride_h;
        for (int64_t r = 0; r < w.rows; ++r) {
            if constexpr (QkInt8) {
                q_scales(t)[r] = p.alpha
                        * quant::quantize_row(qt + r * p.q_stride_t, D, 128,
                                q_tile + r * q_ld);
                std::memset(q_tile + r * q_ld + D, 128, q_ld - D);
            } else {
                std::memcpy(q_tile + r * q_ld, qt + r * p.q_stride_t,
                        D * sizeof(bfloat16_t));
            }
        }
    }

    /// P of rows [ra, rb) x keys [plo, phi), 64 (or 32) keys at a time, then
    /// (INT8 P) quantized.
    void qk(const block_window_t &bw, const char *slot, const tile_t &t) const {
        const slot_t s = layout(slot, bw.nb);
        const qa_t *q_tile = q_rows(t);
        qk_epilogue_t e;
        e.alpha = p.alpha;
        if constexpr (PvInt8)
            std::fill(amax(t) + 16 * bw.ra, amax(t) + 16 * bw.rb, 0.f);
        for (int64_t c = bw.plo / (2 * kNR) * (2 * kNR); c < bw.phi;
                c += 2 * kNR) {
            const int nv = int(std::min<int64_t>(2 * kNR, bw.nb - c));
            chunk_bounds(t.bands, bw.ra, bw.rb, bw.k0 + c, 2 * kNR, t.lo, t.hi);
            const bool two = nv > kNR;
            const int mr = two ? kMR2 : kMR1;
            if constexpr (QkInt8) {
                e.colsum = s.k_sum + c;
                e.col_scale = s.k_scale + c;
            }
            for (int64_t r = bw.ra; r < bw.rb; r += mr) {
                const int m = int(std::min<int64_t>(mr, bw.rb - r));
                const qk_tile_fn<qa_t, kb_t> tile
                        = two ? tiles::two[m] : tiles::one[m];
                e.lo = t.lo + r;
                e.hi = t.hi + r;
                e.row_scale = q_scales(t) + r;
                e.scale = t.scales + r;
                e.amax = amax(t) + 16 * r;
                if constexpr (PvInt8)
                    e.p = p_f32(t) + r * kBlockN + c;
                else
                    e.p = static_cast<bfloat16_t *>(t.p) + r * kBlockN + c;
                tile(q_tile + r * q_ld, q_ld, s.k + c / kNR * k_block, k_block,
                        q_ld, nv, e);
            }
        }
        if constexpr (PvInt8)
            quant::quantize_rows_amax(p_f32(t), kBlockN, amax(t), t.scales,
                    bw.ra, bw.rb, bw.plo, bw.phi, 128, p_q(t), pv_scales(t));
    }

    /// O[ra, rb) += P[:, plo:phi] V[plo:phi], one packed 32-column V block
    /// at a time. INT8: the V column sums over [plo, phi) remove the offset
    /// of P (the stored sums cover the whole block).
    void pv(const block_window_t &bw, const char *slot, const tile_t &t) const {
        const slot_t s = layout(slot, bw.nb);
        if constexpr (!PvInt8) {
            bf16_pv(bw, static_cast<const bfloat16_t *>(t.p), s.v, Dv, t.o);
        } else {
            const int64_t v_block = ck::int8::packed_size(bw.nb, kNR);
            const bool whole = bw.plo == 0 && bw.phi == bw.nb;
            alignas(64) int32_t sum[kNR];
            for (int64_t c = 0; c < Dv; c += kNR) {
                const int64_t n = std::min<int64_t>(kNR, Dv - c);
                const int8_t *vb = s.v + c / kNR * v_block + bw.plo * kNR;
                if (!whole)
                    ck::int8::packed_colsum(bw.phi - bw.plo, n, vb, sum);
                ck::int8::gemm_block(bw.rb - bw.ra, int(n), bw.phi - bw.plo,
                        p_q(t) + bw.ra * kBlockN + bw.plo, kBlockN, vb,
                        whole ? s.v_sum + c : sum, pv_scales(t) + bw.ra,
                        s.v_scale + c, t.o + bw.ra * Dv + c, Dv);
            }
        }
    }

    /// `diag_bf16` from the BF16 Q row (the tile holds the quantized one).
    void diag(
            const work_item_t &w, int64_t r, const tile_t &t, float *o) const {
        const int64_t i = w.s.q_pos + w.row0 + r;
        diag_bf16(q + (w.s.q_row + w.row0 + r) * p.q_stride_t
                        + w.h * p.q_stride_h,
                k + (w.s.kv_row + i) * p.k_stride_t + w.h * p.k_stride_h,
                v + (w.s.kv_row + i) * p.v_stride_t + w.h * p.v_stride_h, D, Dv,
                p.alpha, t.scales[r], o);
    }

    static float *q_scales(const tile_t &t) {
        return static_cast<float *>(t.q);
    }
    static qa_t *q_rows(const tile_t &t) {
        return reinterpret_cast<qa_t *>(
                static_cast<char *>(t.q) + kQScaleBytes);
    }
    static float *amax(const tile_t &t) { return static_cast<float *>(t.p); }
    static float *pv_scales(const tile_t &t) {
        return reinterpret_cast<float *>(static_cast<char *>(t.p) + kAmaxBytes);
    }
    static uint8_t *p_q(const tile_t &t) {
        return reinterpret_cast<uint8_t *>(t.p) + kAmaxBytes + kPvScaleBytes;
    }
    static float *p_f32(const tile_t &t) {
        return reinterpret_cast<float *>(static_cast<char *>(t.p) + kAmaxBytes
                + kPvScaleBytes + kPqBytes);
    }

    const hstu_attention_params &p;
    const bfloat16_t *q, *k, *v;
    int64_t D, Dv;
    /// Row stride of the Q tile, in elements.
    int64_t q_ld;
    /// Elements of one packed 32-key block of K^T.
    int64_t k_block;
};

// pack() stages kNR K rows in the Q area and kBlockN x kNR V bytes in the P
// area of the smallest tile.
static_assert(kMinBlockM >= kNR, "K staging needs kNR rows of the Q tile");
static_assert(int8_policy_t<false, true, false>::p_bytes(kMinBlockM)
                >= size_t(kBlockN * kNR),
        "V staging needs kBlockN x kNR bytes of the P tile");

template <bool QkInt8, bool PvInt8, bool FastSilu>
void run(const hstu_attention_params &params, const void *query,
        const void *key, const void *value, void *output) {
    const int8_policy_t<QkInt8, PvInt8, FastSilu> policy(params,
            static_cast<const bfloat16_t *>(query),
            static_cast<const bfloat16_t *>(key),
            static_cast<const bfloat16_t *>(value));
    if (params.out_dt == data_type_t::f32)
        run_tiled(params, policy, static_cast<float *>(output));
    else
        run_tiled(params, policy, static_cast<bfloat16_t *>(output));
}

template <bool FastSilu>
void run_flags(const hstu_attention_params &params, const void *query,
        const void *key, const void *value, void *output) {
    if (params.is_qk_quant && params.is_pv_quant)
        run<true, true, FastSilu>(params, query, key, value, output);
    else if (params.is_qk_quant)
        run<true, false, FastSilu>(params, query, key, value, output);
    else
        run<false, true, FastSilu>(params, query, key, value, output);
}

} // namespace

bool hstu_attention_tiled_int8_isa() {
    static const bool isa = [] {
        const auto &pi = zendnnl::common::zendnnl_platform_info();
        return pi.get_avx512_vnni_status() && pi.get_avx512_bw_vl_status();
    }();
    return isa;
}

status_t hstu_attention_tiled_int8(const void *query, const void *key,
        const void *value, void *output, const hstu_attention_params &params,
        bool fast_silu) {
    if (!hstu_attention_tiled_supported(params)
            || !hstu_attention_tiled_int8_isa()
            || !(params.is_qk_quant || params.is_pv_quant)
            || (params.is_qk_quant && params.qk_dim > kInt8MaxQkDim))
        return status_t::unimplemented;
    if (fast_silu)
        run_flags<true>(params, query, key, value, output);
    else
        run_flags<false>(params, query, key, value, output);
    return status_t::success;
}

} // namespace hstu_attention
} // namespace lowoha
} // namespace zendnnl
