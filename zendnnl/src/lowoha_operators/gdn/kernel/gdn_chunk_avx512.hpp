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

#ifndef _LOWOHA_GDN_CHUNK_AVX512_HPP
#define _LOWOHA_GDN_CHUNK_AVX512_HPP

// Chunked gated delta rule (the WY form of FLA's chunk_gated_delta_rule) for
// one (sequence, value head, block of value columns) of a prefill. Per chunk
// of kChunk tokens, with gamma the cumulative log decay inside the chunk and
// S the state at its start:
//   A_ij  = beta_i e^(gamma_i - gamma_j) k_i.k_j        (j < i)
//   T     = (I + A)^-1
//   U     = T diag(beta) V,   W = T diag(beta e^gamma) K
//   Delta = U - W S
//   O     = diag(e^gamma) Q S + (tril(Q K^T) . e^(gamma_i - gamma_j)) Delta
//   S    <- e^gamma_last S + K^T diag(e^(gamma_last - gamma)) Delta
// The products run on GDN's CK GEMMs with operands of type ODT (bf16 for bf16
// sources, else f32; see operand_type) and FP32 accumulation; the decays, T
// and the state stay FP32. A, T and W depend only on q, k, g and beta; U,
// Delta, O and the update of S are computed column by column of V, each in
// the same order whatever the width of the block, so a block of a head's
// value columns gets exactly the values the whole head gives them.

#include <algorithm>
#include <cmath>
#include <cstring>
#include "lowoha_operators/gdn/gdn_gemm_ck/lowoha_gdn_ck.hpp"
#include "lowoha_operators/gdn/kernel/gdn_avx512_utils.hpp"
#include "lowoha_operators/gdn/kernel/gdn_native_common.hpp"
#include "lowoha_operators/gdn/lowoha_gdn_utils.hpp"

GDN_AVX512_BEGIN

namespace zendnnl {
namespace lowoha {
namespace gdn {
namespace chunk {

using namespace avx512;

constexpr int64_t kChunk = kChunkLen;

// The operand type of source type SRC. f16 sources take FP32 operands: without
// AVX512-FP16 their products are FP32 FMA either way, and f16 operands would
// overflow on values the FP32 recurrence keeps, such as e^gamma past 65504
// for growing decays or an FP32 state.
template <data_type_t SRC>
constexpr data_type_t operand_type
        = SRC == data_type_t::bf16 ? data_type_t::bf16 : data_type_t::f32;

// Scratch floats one unit needs for head dims K and V: the FP32 buffers, then
// the A and B operand buffers of ODT carved from the same slab.
template <data_type_t ODT>
inline int64_t scratch_floats(int64_t K, int64_t V) {
    const int64_t C = kChunk;
    // tmp also holds T (C x C).
    const int64_t f32
            = 3 * C * K + 4 * C * V + C * std::max(V, C) + 2 * C * C + 3 * C;
    const int64_t a_bytes = static_cast<int64_t>(sizeof(ck::a_elem_t<ODT>))
            * (3 * C * K + C * C);
    const int64_t b_bytes = static_cast<int64_t>(sizeof(ck::b_elem_t<ODT>))
            * (2 * C * K + K * V + 2 * C * V);
    return f32 + (a_bytes + b_bytes + 3) / 4 + 64;
}

// log1p(y) for y in [0, 1]: 2 atanh(s), s = y / (2 + y) <= 1/3.
inline __m512 log1p01(__m512 y) {
    const __m512 s = _mm512_div_ps(y, _mm512_add_ps(_mm512_set1_ps(2.0f), y));
    const __m512 s2 = _mm512_mul_ps(s, s);
    __m512 p = _mm512_set1_ps(1.0f / 11.0f);
    p = _mm512_fmadd_ps(p, s2, _mm512_set1_ps(1.0f / 9.0f));
    p = _mm512_fmadd_ps(p, s2, _mm512_set1_ps(1.0f / 7.0f));
    p = _mm512_fmadd_ps(p, s2, _mm512_set1_ps(1.0f / 5.0f));
    p = _mm512_fmadd_ps(p, s2, _mm512_set1_ps(1.0f / 3.0f));
    p = _mm512_fmadd_ps(p, s2, _mm512_set1_ps(1.0f));
    return _mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(2.0f), s), p);
}

// g = -e^A_log softplus(a + dt_bias) and beta = beta_scale sigmoid(b) for kChunk
// tokens; rows past n get g = beta = 0. softplus(x) = x above threshold, else
// max(x, 0) + log1p(e^-|x|).
inline void gates(const float *a, const float *b, int64_t n, float A_log,
        float dt_bias, float threshold, float beta_scale, float *g,
        float *beta) {
    const __m512 ea = _mm512_set1_ps(-std::exp(A_log));
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 zero = _mm512_setzero_ps();
    for (int64_t j = 0; j < kChunk; j += 16) {
        const __mmask16 m = tail_mask(n - j > 0 ? n - j : 0);
        const __m512 x = _mm512_add_ps(
                _mm512_maskz_loadu_ps(m, a + j), _mm512_set1_ps(dt_bias));
        const __m512 ax = _mm512_abs_ps(x);
        __m512 sp = _mm512_add_ps(_mm512_max_ps(x, zero),
                log1p01(Ops::vec_exp_u20(_mm512_sub_ps(zero, ax))));
        sp = _mm512_mask_blend_ps(
                _mm512_cmp_ps_mask(x, _mm512_set1_ps(threshold), _CMP_GT_OQ),
                sp, x);
        _mm512_storeu_ps(g + j, _mm512_maskz_mul_ps(m, ea, sp));
        const __m512 e = Ops::vec_exp_u20(
                _mm512_sub_ps(zero, _mm512_maskz_loadu_ps(m, b + j)));
        _mm512_storeu_ps(beta + j,
                _mm512_mul_ps(_mm512_set1_ps(beta_scale),
                        _mm512_maskz_div_ps(m, one, _mm512_add_ps(one, e))));
    }
}

// Per-chunk inputs, filled by the caller: q (scaled, normalised), k
// (normalised), v as FP32 rows, g the log decay and beta, for n <= kChunk
// tokens; rows past n are zero and g = beta = 0 there.
struct chunk_io_t {
    float *q, *k, *v, *g, *beta;
};

// A unit's buffers, carved from its slab: FP32 rows, then the A operands
// (qb, kb, wb, pb) and the B operands (ktv, sv, dv, kvn, ddv) of ODT.
template <data_type_t ODT>
struct chunk_scratch_t {
    using a_t = ck::a_elem_t<ODT>;
    using b_t = ck::b_elem_t<ODT>;
    chunk_io_t in;
    float *gam, *G, *P, *U, *W, *tmp, *delta, *o;
    a_t *qb, *kb, *wb, *pb;
    b_t *ktv, *sv, *dv, *kvn, *ddv;

    chunk_scratch_t(float *base, int64_t K, int64_t V) {
        const int64_t C = kChunk;
        float *f = base;
        in.q = f;
        f += C * K;
        in.k = f;
        f += C * K;
        in.v = f;
        f += C * V;
        in.g = f;
        f += C;
        in.beta = f;
        f += C;
        gam = f;
        f += C;
        G = f;
        f += C * C;
        P = f;
        f += C * C;
        U = f;
        f += C * V;
        W = f;
        f += C * K;
        tmp = f;
        f += C * std::max(V, C);
        delta = f;
        f += C * V;
        o = f;
        f += C * V;
        a_t *a = reinterpret_cast<a_t *>(f);
        qb = a;
        a += C * K;
        kb = a;
        a += C * K;
        wb = a;
        a += C * K;
        pb = a;
        a += C * C;
        b_t *b = reinterpret_cast<b_t *>(a);
        ktv = b;
        b += C * K;
        sv = b;
        b += K * V;
        dv = b;
        b += C * V;
        kvn = b;
        b += C * K;
        ddv = b;
    }
};

// The A operand of FP32 rows x: x itself for f32 operands, else converted
// into buf.
template <data_type_t ODT>
inline const ck::a_elem_t<ODT> *a_operand(
        const float *x, int64_t n, ck::a_elem_t<ODT> *buf) {
    if constexpr (ODT == data_type_t::f32) {
        return x;
    } else {
        ck::to_a<ODT>(x, n, buf);
        return buf;
    }
}

// The B operand of FP32 [R][N] (rows N apart): x itself for f32 operands,
// else packed into buf.
template <data_type_t ODT>
inline const ck::b_elem_t<ODT> *b_operand(
        const float *x, int64_t r, int64_t n, ck::b_elem_t<ODT> *buf) {
    if constexpr (ODT == data_type_t::f32) {
        return x;
    } else {
        ck::pack_b<ODT>(x, r, n, n, buf);
        return buf;
    }
}

// Advances the FP32 state S [K][V] (V value columns, a multiple of 32) over
// one chunk of n tokens and writes the n FP32 output rows to sc.o.
template <data_type_t ODT>
inline void chunk_step(
        chunk_scratch_t<ODT> &sc, int64_t n, int64_t K, int64_t V, float *S) {
    using a_t = ck::a_elem_t<ODT>;
    using b_t = ck::b_elem_t<ODT>;
    constexpr int64_t kPack = ck::operand<ODT>::kPack;
    const int64_t C = kChunk;
    float *gam = sc.gam;
    float acc = 0.0f;
    for (int64_t i = 0; i < C; ++i) {
        acc += sc.in.g[i];
        gam[i] = acc;
    }
    const float gam_last = gam[C - 1];

    // K K^T and Q K^T, with K^T packed; only their lower triangles are read.
    const a_t *qa = a_operand<ODT>(sc.in.q, C * K, sc.qb);
    const a_t *ka = a_operand<ODT>(sc.in.k, C * K, sc.kb);
    ck::pack_b_transposed<ODT>(ka, C, K, sc.ktv);
    ck::gemm_rows<ODT>(C, C, K, ka, sc.ktv, sc.G, C, false, ck::tri_t::lower_c);
    ck::gemm_rows<ODT>(C, C, K, qa, sc.ktv, sc.P, C, false, ck::tri_t::lower_c);

    // Decays e^(gamma_i - gamma_j): A strictly lower, P lower with its
    // diagonal. Above the diagonal, which both drop, the exponent is 0, so
    // gates of either sign (g > 0 grows the state) cannot overflow there.
    alignas(64) float d[kChunk];
    const __m512i lanes = _mm512_set_epi32(
            15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    for (int64_t i = 0; i < C; ++i) {
        const __m512i ii = _mm512_set1_epi32(static_cast<int>(i));
        for (int64_t j = 0; j < C; j += 16) {
            const __mmask16 le = _mm512_cmple_epi32_mask(
                    _mm512_add_epi32(
                            _mm512_set1_epi32(static_cast<int>(j)), lanes),
                    ii);
            _mm512_storeu_ps(d + j,
                    Ops::vec_exp_u20(_mm512_maskz_sub_ps(le,
                            _mm512_set1_ps(gam[i]), _mm512_loadu_ps(gam + j))));
        }
        float *Gi = sc.G + i * C;
        float *Pi = sc.P + i * C;
        const __m512 bi = _mm512_set1_ps(sc.in.beta[i]);
        for (int64_t j = 0; j < C; j += 16) {
            const __m512i jj = _mm512_add_epi32(
                    _mm512_set1_epi32(static_cast<int>(j)), lanes);
            const __mmask16 lt = _mm512_cmplt_epi32_mask(jj, ii);
            const __mmask16 le = _mm512_cmple_epi32_mask(jj, ii);
            const __m512 dj = _mm512_loadu_ps(d + j);
            _mm512_storeu_ps(Gi + j,
                    _mm512_maskz_mul_ps(lt, _mm512_mul_ps(bi, dj),
                            _mm512_loadu_ps(Gi + j)));
            _mm512_storeu_ps(Pi + j,
                    _mm512_maskz_mul_ps(le, dj, _mm512_loadu_ps(Pi + j)));
        }
    }

    // T = (I + A)^-1 by rows: T_i = e_i - sum_{j<i} A_ij T_j (C wide).
    for (int64_t i = 0; i < C; ++i) {
        __m512 r[kChunk / 16];
        for (int64_t b = 0; b < C / 16; ++b) {
            r[b] = _mm512_maskz_mov_ps(
                    static_cast<__mmask16>(i / 16 == b ? 1u << (i % 16) : 0u),
                    _mm512_set1_ps(1.0f));
        }
        const float *Ai = sc.G + i * C;
        for (int64_t j = 0; j < i; ++j) {
            const __m512 a = _mm512_set1_ps(-Ai[j]);
            for (int64_t b = 0; b < C / 16; ++b) {
                r[b] = _mm512_fmadd_ps(
                        a, _mm512_loadu_ps(sc.tmp + j * C + 16 * b), r[b]);
            }
        }
        for (int64_t b = 0; b < C / 16; ++b) {
            _mm512_storeu_ps(sc.tmp + i * C + 16 * b, r[b]);
        }
    }

    // U = T diag(beta) V and W = T diag(beta e^gamma) K, T lower triangular.
    // The products overwrite their B sources, so B is always packed here.
    const a_t *ta = a_operand<ODT>(sc.tmp, C * C, sc.pb);
    for (int64_t i = 0; i < C; ++i) {
        const __m512 bv = _mm512_set1_ps(sc.in.beta[i]);
        const __m512 bk = _mm512_set1_ps(sc.in.beta[i] * std::exp(gam[i]));
        for (int64_t c = 0; c < V; c += 16) {
            _mm512_storeu_ps(sc.U + i * V + c,
                    _mm512_mul_ps(bv, _mm512_loadu_ps(sc.in.v + i * V + c)));
        }
        for (int64_t c = 0; c < K; c += 16) {
            _mm512_storeu_ps(sc.W + i * K + c,
                    _mm512_mul_ps(bk, _mm512_loadu_ps(sc.in.k + i * K + c)));
        }
    }
    ck::pack_b<ODT>(sc.U, C, V, V, sc.dv);
    ck::pack_b<ODT>(sc.W, C, K, K, sc.kvn);
    ck::gemm_rows<ODT>(C, V, C, ta, sc.dv, sc.U, V, false, ck::tri_t::lower_a);
    ck::gemm_rows<ODT>(C, K, C, ta, sc.kvn, sc.W, K, false, ck::tri_t::lower_a);

    // Delta = U - W S; O = diag(e^gamma) Q S + P Delta.
    const a_t *wa = a_operand<ODT>(sc.W, C * K, sc.wb);
    const b_t *sb = b_operand<ODT>(S, K, V, sc.sv);
    ck::gemm_rows<ODT>(C, V, K, wa, sb, sc.tmp, V, false);
    for (int64_t i = 0; i < C * V; i += 16) {
        _mm512_storeu_ps(sc.delta + i,
                _mm512_sub_ps(_mm512_loadu_ps(sc.U + i),
                        _mm512_loadu_ps(sc.tmp + i)));
    }
    ck::gemm_rows<ODT>(C, V, K, qa, sb, sc.o, V, false);
    for (int64_t i = 0; i < n; ++i) {
        const __m512 e = _mm512_set1_ps(std::exp(gam[i]));
        for (int64_t c = 0; c < V; c += 16) {
            _mm512_storeu_ps(sc.o + i * V + c,
                    _mm512_mul_ps(e, _mm512_loadu_ps(sc.o + i * V + c)));
        }
    }
    const a_t *pa = a_operand<ODT>(sc.P, C * C, sc.pb);
    const b_t *db = b_operand<ODT>(sc.delta, C, V, sc.dv);
    ck::gemm_rows<ODT>(C, V, C, pa, db, sc.o, V, true, ck::tri_t::lower_a);

    // S <- e^gamma_last S + K^T diag(e^(gamma_last - gamma)) Delta, with K^T
    // read as the transpose of K's B operand ([C / kPack][K] steps).
    for (int64_t j = 0; j < C; ++j) {
        const __m512 e = _mm512_set1_ps(std::exp(gam_last - gam[j]));
        for (int64_t c = 0; c < V; c += 16) {
            _mm512_storeu_ps(sc.tmp + j * V + c,
                    _mm512_mul_ps(e, _mm512_loadu_ps(sc.delta + j * V + c)));
        }
    }
    const b_t *ddb = b_operand<ODT>(sc.tmp, C, V, sc.ddv);
    const b_t *kt = b_operand<ODT>(sc.in.k, C, K, sc.kvn);
    const __m512 el = _mm512_set1_ps(std::exp(gam_last));
    for (int64_t i = 0; i < K * V; i += 16) {
        _mm512_storeu_ps(S + i, _mm512_mul_ps(el, _mm512_loadu_ps(S + i)));
    }
    ck::gemm<ODT>(K, V, C / kPack,
            reinterpret_cast<const ck::a_step_t<ODT> *>(kt), 1, K, ddb, S, V,
            true);
}

} // namespace chunk
} // namespace gdn
} // namespace lowoha
} // namespace zendnnl

GDN_AVX512_END

#endif // _LOWOHA_GDN_CHUNK_AVX512_HPP
