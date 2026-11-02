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

#include "lowoha_operators/hstu_attention/tiled/lowoha_hstu_tiled.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <immintrin.h>
#include <utility>
#include "common/bfloat16.hpp"
#include "common/zendnnl_global.hpp"
#include "lowoha_operators/hstu_attention/tiled/lowoha_hstu_tiled_common.hpp"
#include "lowoha_operators/hstu_attention/tiled/lowoha_hstu_tiled_driver.hpp"
#include "lowoha_operators/matmul/group_matmul/common/group_matmul_act_avx512.hpp"

namespace zendnnl {
namespace lowoha {
namespace hstu_attention {

namespace {

using zendnnl::common::bfloat16_t;
namespace act = zendnnl::lowoha::matmul::group_matmul_act_avx512;

/// One tile of the fused Q*K^T over NB packed blocks of K^T:
/// P[MR][nv] = BF16(mask * silu(alpha * A * B) * scale), silu from
/// `act::silu_avx512` or, with FastSilu, `fast_silu_scaled`. Row m keeps
/// columns [lo[m], hi[m]) of the tile; a tile with no kept column skips the
/// GEMM.
template <int MR, int NB, bool FastSilu>
HSTU_AVX512_NOINLINE void qk_silu_tile(const bfloat16_t *__restrict a,
        int64_t lda, const bfloat16_t *__restrict b, int64_t bstride, int64_t k,
        int nv, const int32_t *lo, const int32_t *hi, const float *scale,
        float alpha, bfloat16_t *__restrict p, int64_t ldp) {
    const uint64_t cols = range_bits(0, nv);
    uint64_t keep[MR];
    uint64_t any = 0;
    for (int m = 0; m < MR; ++m) {
        keep[m] = range_bits(lo[m], hi[m]) & cols;
        any |= keep[m];
    }
    if (!any) {
        for (int m = 0; m < MR; ++m)
            for (int j = 0; j < NB; ++j)
                _mm512_mask_storeu_epi16(p + m * ldp + j * kNR,
                        __mmask32(cols >> (32 * j)), _mm512_setzero_si512());
        return;
    }

    __m512 acc[MR][2 * NB];
#pragma GCC unroll 8
    for (int m = 0; m < MR; ++m)
        for (int j = 0; j < 2 * NB; ++j)
            acc[m][j] = _mm512_setzero_ps();
    const int64_t full = k / 2;
    int64_t q = 0;
    for (; q + 1 < full; q += 2) {
        pair_step<MR, NB>(acc, a, lda, b, bstride, q, false);
        pair_step<MR, NB>(acc, a, lda, b, bstride, q + 1, false);
    }
    if (q < full) pair_step<MR, NB>(acc, a, lda, b, bstride, q, false);
    if (k & 1) pair_step<MR, NB>(acc, a, lda, b, bstride, full, true);

    [[maybe_unused]] const __m512 va = _mm512_set1_ps(alpha);
    [[maybe_unused]] const __m512 vt = _mm512_set1_ps(-alpha * 1.44269504f);
#pragma GCC unroll 8
    for (int m = 0; m < MR; ++m) {
        const __m512 vs
                = _mm512_set1_ps(FastSilu ? alpha * scale[m] : scale[m]);
#pragma GCC unroll 2
        for (int j = 0; j < NB; ++j) {
            __m512 y0, y1;
            if constexpr (FastSilu) {
                y0 = fast_silu_scaled(acc[m][2 * j], vt, vs);
                y1 = fast_silu_scaled(acc[m][2 * j + 1], vt, vs);
            } else {
                y0 = _mm512_mul_ps(
                        act::silu_avx512(_mm512_mul_ps(acc[m][2 * j], va)), vs);
                y1 = _mm512_mul_ps(
                        act::silu_avx512(_mm512_mul_ps(acc[m][2 * j + 1], va)),
                        vs);
            }
            const __m512i v = (__m512i)_mm512_cvtne2ps_pbh(y1, y0);
            _mm512_mask_storeu_epi16(p + m * ldp + j * kNR,
                    __mmask32(cols >> (32 * j)),
                    _mm512_maskz_mov_epi16(__mmask32(keep[m] >> (32 * j)), v));
        }
    }
}

using qk_tile_fn = void (*)(const bfloat16_t *, int64_t, const bfloat16_t *,
        int64_t, int64_t, int, const int32_t *, const int32_t *, const float *,
        float, bfloat16_t *, int64_t);

template <int NB, bool FastSilu, int... M>
constexpr std::array<qk_tile_fn, sizeof...(M) + 1> qk_tiles(
        std::integer_sequence<int, M...>) {
    return {nullptr, qk_silu_tile<M + 1, NB, FastSilu>...};
}

/// Tile kernels by row count, over one or two packed blocks.
template <bool FastSilu>
constexpr auto kQkTiles1
        = qk_tiles<1, FastSilu>(std::make_integer_sequence<int, kMR1>());
template <bool FastSilu>
constexpr auto kQkTiles2
        = qk_tiles<2, FastSilu>(std::make_integer_sequence<int, kMR2>());

/// The BF16 policy of `run_tiled`: K^T and V packed by
/// `ck::bf16::pack_b`, BF16 Q and P tiles, Q*K^T on the fused silu tile kernels
/// above (FastSilu: `fast_silu_scaled`) and P*V on `ck::bf16::gemm_block`.
template <bool FastSilu>
struct bf16_policy_t {
    using p_t = bfloat16_t;
    /// Packed V holds key pairs.
    static constexpr int64_t kKeyAlign = 2;
    static size_t q_bytes(int64_t bm, int64_t d) {
        return bm * d * sizeof(bfloat16_t);
    }
    static size_t p_bytes(int64_t bm) { return bm * kBlockN * sizeof(p_t); }

    bf16_policy_t(const hstu_attention_params &p, const bfloat16_t *q,
            const bfloat16_t *k, const bfloat16_t *v)
        : p(p)
        , q(q)
        , k(k)
        , v(v)
        , D(p.qk_dim)
        , Dv(p.v_dim)
        , k_block((D + 1) / 2 * 2 * kNR) {}

    /// Packed K^T, then packed V.
    size_t slot_bytes(int64_t nb) const {
        return (ck::bf16::packed_size(D, nb) + ck::bf16::packed_size(nb, Dv))
                * sizeof(bfloat16_t);
    }

    void pack(const kv_block_t &blk, int64_t h, char *slot,
            const tile_t &) const {
        const int64_t k0 = p.seq_offsets[blk.b] + blk.k0;
        auto *pk = reinterpret_cast<bfloat16_t *>(slot);
        ck::bf16::pack_b(D, blk.nb, k + k0 * p.k_stride_t + h * p.k_stride_h,
                p.k_stride_t, true, pk);
        ck::bf16::pack_b(blk.nb, Dv, v + k0 * p.v_stride_t + h * p.v_stride_h,
                p.v_stride_t, false, pk + ck::bf16::packed_size(D, blk.nb));
    }

    /// The Q tile is copied so that rows of a strided view (e.g. q of the
    /// fused uvqk projection) do not alias in L1.
    void load_q(const work_item_t &w, const tile_t &t) const {
        auto *q_tile = static_cast<bfloat16_t *>(t.q);
        const bfloat16_t *qt
                = q + (w.s.q_row + w.row0) * p.q_stride_t + w.h * p.q_stride_h;
        for (int64_t r = 0; r < w.rows; ++r)
            std::memcpy(q_tile + r * D, qt + r * p.q_stride_t,
                    D * sizeof(bfloat16_t));
    }

    /// P = mask * silu(alpha * Q K^T) * scale, 64 (or 32) keys at a time.
    void qk(const block_window_t &bw, const char *slot, const tile_t &t) const {
        const auto *pk = reinterpret_cast<const bfloat16_t *>(slot);
        const auto *q_tile = static_cast<const bfloat16_t *>(t.q);
        auto *p_tile = static_cast<p_t *>(t.p);
        for (int64_t c = bw.plo / (2 * kNR) * (2 * kNR); c < bw.phi;
                c += 2 * kNR) {
            const int nv = int(std::min<int64_t>(2 * kNR, bw.nb - c));
            chunk_bounds(t.bands, bw.ra, bw.rb, bw.k0 + c, 2 * kNR, t.lo, t.hi);
            const bool two = nv > kNR;
            const int mr = two ? kMR2 : kMR1;
            for (int64_t r = bw.ra; r < bw.rb; r += mr) {
                const int m = int(std::min<int64_t>(mr, bw.rb - r));
                const qk_tile_fn tile
                        = two ? kQkTiles2<FastSilu>[m] : kQkTiles1<FastSilu>[m];
                tile(q_tile + r * D, D, pk + c / kNR * k_block, k_block, D, nv,
                        t.lo + r, t.hi + r, t.scales + r, p.alpha,
                        p_tile + r * kBlockN + c, kBlockN);
            }
        }
    }

    void pv(const block_window_t &bw, const char *slot, const tile_t &t) const {
        bf16_pv(bw, static_cast<const p_t *>(t.p),
                reinterpret_cast<const bfloat16_t *>(slot)
                        + ck::bf16::packed_size(D, bw.nb),
                Dv, t.o);
    }

    /// `diag_bf16` from the staged Q row.
    void diag(
            const work_item_t &w, int64_t r, const tile_t &t, float *o) const {
        const int64_t i = w.s.q_pos + w.row0 + r;
        diag_bf16(static_cast<const bfloat16_t *>(t.q) + r * D,
                k + (w.s.kv_row + i) * p.k_stride_t + w.h * p.k_stride_h,
                v + (w.s.kv_row + i) * p.v_stride_t + w.h * p.v_stride_h, D, Dv,
                p.alpha, t.scales[r], o);
    }

    const hstu_attention_params &p;
    const bfloat16_t *q, *k, *v;
    int64_t D, Dv;
    /// Elements of one packed 32-key block of K^T.
    int64_t k_block;
};

} // namespace

bool hstu_attention_tiled_supported(const hstu_attention_params &params) {
    static const bool isa = [] {
        const auto &pi = zendnnl::common::zendnnl_platform_info();
        return pi.get_avx512_bf16_status() && pi.get_avx512_bw_vl_status();
    }();
    return isa && params.causal && params.qkv_dt == data_type_t::bf16
            && (params.out_dt == data_type_t::bf16
                    || params.out_dt == data_type_t::f32);
}

namespace {

template <bool FastSilu>
void run_bf16(const void *query, const void *key, const void *value,
        void *output, const hstu_attention_params &params) {
    const bf16_policy_t<FastSilu> policy(params,
            static_cast<const bfloat16_t *>(query),
            static_cast<const bfloat16_t *>(key),
            static_cast<const bfloat16_t *>(value));
    if (params.out_dt == data_type_t::f32)
        run_tiled(params, policy, static_cast<float *>(output));
    else
        run_tiled(params, policy, static_cast<bfloat16_t *>(output));
}

} // namespace

status_t hstu_attention_tiled_bf16(const void *query, const void *key,
        const void *value, void *output, const hstu_attention_params &params,
        bool fast_silu) {
    if (!hstu_attention_tiled_supported(params)) return status_t::unimplemented;
    if (fast_silu)
        run_bf16<true>(query, key, value, output, params);
    else
        run_bf16<false>(query, key, value, output, params);
    return status_t::success;
}

} // namespace hstu_attention
} // namespace lowoha
} // namespace zendnnl
