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

#ifndef _LOWOHA_GDN_NATIVE_COMMON_HPP
#define _LOWOHA_GDN_NATIVE_COMMON_HPP

#include <cmath>
#include <cstring>
#include <vector>
#include "lowoha_operators/gdn/kernel/gdn_avx512_utils.hpp"
#include "lowoha_operators/gdn/lowoha_gdn_utils.hpp"
#include <type_traits>

GDN_AVX512_BEGIN

namespace zendnnl {
namespace lowoha {
namespace gdn {

constexpr int64_t kConvBlock = 64;

// Per-calling-thread FP32 scratch that only grows, so steady-state calls do
// not allocate.
inline float *scratch_f32(int id, size_t n) {
    static thread_local std::vector<float> bufs[4];
    std::vector<float> &v = bufs[id];
    if (v.size() < n) { v.resize(n); }
    return v.data();
}

// Number of value tiles per head: the fewest that still give every thread a
// unit. Tiles stay multiples of 16 columns.
inline int64_t select_v_splits(
        int64_t heads_total, int64_t V, int32_t nthr, bool kv_layout) {
    int64_t splits = 1;
    const int64_t min_tile = kv_layout ? 16 : 1;
    while (heads_total * splits < nthr && (V / (splits * 2)) >= min_tile
            && (V % (splits * 2)) == 0 && ((V / (splits * 2)) % 16) == 0) {
        splits *= 2;
    }
    return splits;
}

// qn = scale * l2norm(q), kn = l2norm(k), both FP32. K is a multiple of 16.
inline void prepare_qk(const float *q, const float *k, int64_t K, bool l2,
        float eps, float scale, float *qn, float *kn) {
    using namespace avx512;
    float rq = 1.0f, rk = 1.0f;
    if (l2) {
        __m512 sq = _mm512_setzero_ps(), sk = _mm512_setzero_ps();
        for (int64_t i = 0; i < K; i += 16) {
            const __m512 qv = _mm512_loadu_ps(q + i);
            const __m512 kv = _mm512_loadu_ps(k + i);
            sq = _mm512_fmadd_ps(qv, qv, sq);
            sk = _mm512_fmadd_ps(kv, kv, sk);
        }
        rq = 1.0f / std::sqrt(_mm512_reduce_add_ps(sq) + eps);
        rk = 1.0f / std::sqrt(_mm512_reduce_add_ps(sk) + eps);
    }
    const __m512 qs = _mm512_set1_ps(rq * scale);
    const __m512 ks = _mm512_set1_ps(rk);
    for (int64_t i = 0; i < K; i += 16) {
        _mm512_storeu_ps(qn + i, _mm512_mul_ps(_mm512_loadu_ps(q + i), qs));
        _mm512_storeu_ps(kn + i, _mm512_mul_ps(_mm512_loadu_ps(k + i), ks));
    }
}

// Delta-rule update of one K x (NV * 16) state tile with row stride ld, read
// from S_in and written to S_out (which may be S_in), writing the FP32 output
// of the tile to o. Both sweeps walk the same block, so the second one is
// served from cache. Narrow tiles keep P partial sums over k to hide the FMA
// latency.
template <data_type_t ST, int NV>
inline void kv_tile_step(const avx512::elem_t<ST> *S_in,
        avx512::elem_t<ST> *S_out, int64_t ld, int64_t K, const float *vin,
        const float *qn, const float *kn, float eg, float beta, __m512 *o) {
    using namespace avx512;
    constexpr int P = NV >= 4 ? 1 : (NV == 2 ? 2 : 4);
    __m512 acc[P][NV];
    for (int i = 0; i < P; ++i) {
        for (int n = 0; n < NV; ++n) {
            acc[i][n] = _mm512_setzero_ps();
        }
    }
    int64_t k = 0;
    for (; k + P <= K; k += P) {
        for (int i = 0; i < P; ++i) {
            const __m512 kk = _mm512_set1_ps(kn[k + i]);
            const elem_t<ST> *row = S_in + (k + i) * ld;
            for (int n = 0; n < NV; ++n) {
                acc[i][n] = _mm512_fmadd_ps(
                        load16<ST>(row + 16 * n), kk, acc[i][n]);
            }
        }
    }
    for (; k < K; ++k) {
        const __m512 kk = _mm512_set1_ps(kn[k]);
        const elem_t<ST> *row = S_in + k * ld;
        for (int n = 0; n < NV; ++n) {
            acc[0][n]
                    = _mm512_fmadd_ps(load16<ST>(row + 16 * n), kk, acc[0][n]);
        }
    }
    const __m512 egv = _mm512_set1_ps(eg);
    const __m512 betav = _mm512_set1_ps(beta);
    __m512 dt[NV];
    __m512 op[P][NV];
    for (int n = 0; n < NV; ++n) {
        __m512 s = acc[0][n];
        for (int i = 1; i < P; ++i) {
            s = _mm512_add_ps(s, acc[i][n]);
        }
        dt[n] = _mm512_mul_ps(
                _mm512_fnmadd_ps(s, egv, _mm512_loadu_ps(vin + 16 * n)), betav);
        for (int i = 0; i < P; ++i) {
            op[i][n] = _mm512_setzero_ps();
        }
    }
    k = 0;
    for (; k + P <= K; k += P) {
        for (int i = 0; i < P; ++i) {
            const __m512 kk = _mm512_set1_ps(kn[k + i]);
            const __m512 qq = _mm512_set1_ps(qn[k + i]);
            const elem_t<ST> *rin = S_in + (k + i) * ld;
            elem_t<ST> *rout = S_out + (k + i) * ld;
            for (int n = 0; n < NV; ++n) {
                const __m512 s = _mm512_fmadd_ps(load16<ST>(rin + 16 * n), egv,
                        _mm512_mul_ps(kk, dt[n]));
                store16<ST>(rout + 16 * n, s);
                op[i][n] = _mm512_fmadd_ps(s, qq, op[i][n]);
            }
        }
    }
    for (; k < K; ++k) {
        const __m512 kk = _mm512_set1_ps(kn[k]);
        const __m512 qq = _mm512_set1_ps(qn[k]);
        const elem_t<ST> *rin = S_in + k * ld;
        elem_t<ST> *rout = S_out + k * ld;
        for (int n = 0; n < NV; ++n) {
            const __m512 s = _mm512_fmadd_ps(
                    load16<ST>(rin + 16 * n), egv, _mm512_mul_ps(kk, dt[n]));
            store16<ST>(rout + 16 * n, s);
            op[0][n] = _mm512_fmadd_ps(s, qq, op[0][n]);
        }
    }
    for (int n = 0; n < NV; ++n) {
        __m512 s = op[0][n];
        for (int i = 1; i < P; ++i) {
            s = _mm512_add_ps(s, op[i][n]);
        }
        o[n] = s;
    }
}

inline void gate_scalars(float a, float b, float A_log, float dt_bias,
        float threshold, float beta_scale, float &eg, float &beta) {
    const float g = -std::exp(A_log) * softplus_f32(a + dt_bias, threshold);
    eg = std::exp(g);
    beta = beta_scale * sigmoid_f32(b);
}

// n values of type dt to FP32.
inline void load_f32_row(
        const void *src, data_type_t dt, int64_t n, float *dst) {
    using namespace avx512;
    if (dt == data_type_t::f16) {
        row_f32<data_type_t::f16>(static_cast<const uint16_t *>(src), n, dst);
    } else if (dt == data_type_t::bf16) {
        row_f32<data_type_t::bf16>(static_cast<const uint16_t *>(src), n, dst);
    } else {
        row_f32<data_type_t::f32>(static_cast<const float *>(src), n, dst);
    }
}

// dst[j * ld + c] = w[c * W + j] in FP32 for the nc channels and W taps of
// a depthwise conv weight w ([channel][tap], from the first channel).
template <data_type_t SRC>
inline void conv_taps(const avx512::elem_t<SRC> *w, int64_t nc, int64_t W,
        float *dst, int64_t ld) {
    using namespace avx512;
    alignas(64) float buf[kConvBlock * kMaxConvWidth];
    for (int64_t c0 = 0; c0 < nc; c0 += kConvBlock) {
        const int64_t n = std::min<int64_t>(kConvBlock, nc - c0);
        row_f32<SRC>(w + c0 * W, n * W, buf);
        for (int64_t c = 0; c < n; ++c) {
            for (int64_t j = 0; j < W; ++j) {
                dst[j * ld + c0 + c] = buf[c * W + j];
            }
        }
    }
}

// bias + sum_j wt[j] . h_j + wt[nt] . x over channels n .. n + 15 (mask m),
// where h_j, tap j of the channels, is the HDT row h + j * hld.
template <data_type_t SRC, data_type_t HDT>
inline __m512 conv16(const float (*wt)[kConvBlock], const float *bias,
        int64_t nt, const avx512::elem_t<HDT> *h, int64_t hld,
        const avx512::elem_t<SRC> *x, int64_t n, __mmask16 m,
        const gdn_params &p) {
    using namespace avx512;
    __m512 acc = _mm512_loadu_ps(bias + n);
    for (int64_t j = 0; j < nt; ++j) {
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(wt[j] + n),
                load16_mask<HDT>(h + j * hld + n, m), acc);
    }
    acc = _mm512_fmadd_ps(
            _mm512_loadu_ps(wt[nt] + n), load16_mask<SRC>(x + n, m), acc);
    if (p.silu) { acc = silu16(acc); }
    if (p.round_conv_out) { acc = round16<SRC>(acc); }
    return acc;
}

// Whether conv16 reads a conv state of these strides: unit-stride channels
// (vLLM's SD layout) or each channel's taps contiguous (its DS layout, also
// with draft taps or padding after them). Other layouts take conv1.
inline bool conv16_layout(int64_t tap_stride, int64_t dim_stride, int64_t nt) {
    return dim_stride == 1
            || (tap_stride == 1
                    && (dim_stride == nt || (nt > 0 && dim_stride > nt)));
}

// One channel of the conv in scalar arithmetic: bias + sum_j wt[j][c] *
// tap(j) + wt[nt][c] * x, then the activation and rounding.
template <data_type_t SRC, typename Tap>
inline float conv1(const float (*wt)[kConvBlock], const float *bias, int64_t c,
        int64_t nt, Tap tap, float x, const gdn_params &p) {
    float acc = bias[c];
    for (int64_t j = 0; j < nt; ++j) {
        acc += wt[j][c] * tap(j);
    }
    acc += wt[nt][c] * x;
    if (p.silu) { acc = silu_f32(acc); }
    return p.round_conv_out ? round_to(acc, SRC) : acc;
}

// Conv1d update of channels [c0, c0 + nc) for every decode sequence:
// xbuf[b][c] = act(bias + sum_j w[c][j] * window_j), then the conv state
// shifts by one token. With conv_width == 0 the input is copied as is.
template <data_type_t SRC>
inline void conv_update_block(int64_t c0, int64_t nc, int64_t B,
        const avx512::elem_t<SRC> *in, int64_t in_stride,
        const void *conv_weight, const void *conv_bias, void *conv_state,
        const int32_t *state_indices, float *xbuf, const gdn_params &p) {
    using namespace avx512;
    using T = elem_t<SRC>;
    const int64_t W = p.conv_width;
    const int64_t D = p.conv_dim();

    if (W == 0) {
        for (int64_t bi = 0; bi < B; ++bi) {
            const T *x = in + bi * in_stride + c0;
            float *xo = xbuf + bi * D + c0;
            for (int64_t n = 0; n < nc; n += 16) {
                const __mmask16 m = tail_mask(nc - n);
                _mm512_mask_storeu_ps(xo + n, m, load16_mask<SRC>(x + n, m));
            }
        }
        return;
    }

    alignas(64) float wt[kMaxConvWidth][kConvBlock];
    alignas(64) float bias[kConvBlock];
    conv_taps<SRC>(static_cast<const T *>(conv_weight) + c0 * W, nc, W, wt[0],
            kConvBlock);
    row_f32<SRC>(conv_bias ? static_cast<const T *>(conv_bias) + c0 : nullptr,
            nc, bias);

    T *cs = static_cast<T *>(conv_state);
    const int64_t tap = p.conv_state_tap_stride;
    const int64_t cstr = p.conv_state_dim_stride;
    const int64_t nt = W - 1;
    // vLLM's DS layout keeps each channel's taps contiguous ([nc][cstr], the
    // first nt used).
    const bool ds_block = cstr != 1 && conv16_layout(tap, cstr, nt);
    for (int64_t bi = 0; bi < B; ++bi) {
        const int32_t slot = state_indices[bi];
        if (slot == p.pad_slot_id) { continue; }
        const T *x = in + bi * in_stride + c0;
        T *srow = cs + slot * p.conv_state_slot_stride + c0 * cstr;
        float *xo = xbuf + bi * D + c0;
        if (cstr == 1) {
            for (int64_t n = 0; n < nc; n += 16) {
                const __mmask16 m = tail_mask(nc - n);
                _mm512_mask_storeu_ps(xo + n, m,
                        conv16<SRC, SRC>(wt, bias, nt, srow, tap, x, n, m, p));
            }
            for (int64_t j = 0; j + 1 < W - 1; ++j) {
                std::memcpy(
                        srow + j * tap, srow + (j + 1) * tap, nc * sizeof(T));
            }
            if (W > 1) { std::memcpy(srow + (W - 2) * tap, x, nc * sizeof(T)); }
        } else if (ds_block) {
            // The taps by channel in FP32, then by tap for the vector loop.
            alignas(64) float hb[kConvBlock * kMaxConvWidth];
            alignas(64) float ht[kMaxConvWidth][kConvBlock];
            if (cstr == nt) {
                row_f32<SRC>(srow, nc * nt, hb);
            } else {
                for (int64_t c = 0; c < nc; ++c) {
                    row_f32<SRC>(srow + c * cstr, nt, hb + c * nt);
                }
            }
            for (int64_t c = 0; c < nc; ++c) {
                for (int64_t j = 0; j < nt; ++j) {
                    ht[j][c] = hb[c * nt + j];
                }
            }
            for (int64_t n = 0; n < nc; n += 16) {
                const __mmask16 m = tail_mask(nc - n);
                _mm512_mask_storeu_ps(xo + n, m,
                        conv16<SRC, data_type_t::f32>(
                                wt, bias, nt, ht[0], kConvBlock, x, n, m, p));
            }
            // Shift each channel's taps by one and append its input.
            if (nt > 1 && cstr == nt) {
                std::memmove(srow, srow + 1, (nc * nt - 1) * sizeof(T));
            } else if (nt > 1) {
                for (int64_t c = 0; c < nc; ++c) {
                    std::memmove(srow + c * cstr, srow + c * cstr + 1,
                            (nt - 1) * sizeof(T));
                }
            }
            for (int64_t c = 0; nt > 0 && c < nc; ++c) {
                srow[c * cstr + nt - 1] = x[c];
            }
        } else {
            for (int64_t c = 0; c < nc; ++c) {
                T *cell = srow + c * cstr;
                xo[c] = conv1<SRC>(wt, bias, c, nt, [cell, tap](int64_t j) {
                    return load1<SRC>(cell + j * tap);
                }, load1<SRC>(x + c), p);
                for (int64_t j = 0; j + 1 < W - 1; ++j) {
                    cell[j * tap] = cell[(j + 1) * tap];
                }
                if (W > 1) { cell[(W - 2) * tap] = x[c]; }
            }
        }
    }
}

// out = rmsnorm(o) * w * act(z) for one head of V values. With round_in, o is
// first rounded to SRC; o may alias out when it already holds SRC values.
template <data_type_t SRC, typename O>
inline void gated_rmsnorm_row(const O *o, const avx512::elem_t<SRC> *z,
        const float *w, int64_t V, float eps, bool round_in,
        gdn_norm_gate_t gate, avx512::elem_t<SRC> *out) {
    using namespace avx512;
    static constexpr data_type_t ODT
            = std::is_same<O, float>::value ? data_type_t::f32 : SRC;
    auto load_o = [&](int64_t j, __mmask16 m) {
        const __m512 v = load16_mask<ODT>(o + j, m);
        return round_in ? round16<SRC>(v) : v;
    };
    __m512 ss = _mm512_setzero_ps();
    for (int64_t j = 0; j < V; j += 16) {
        const __m512 v = load_o(j, tail_mask(V - j));
        ss = _mm512_fmadd_ps(v, v, ss);
    }
    const float rs = 1.0f
            / std::sqrt(_mm512_reduce_add_ps(ss) / static_cast<float>(V) + eps);
    const __m512 rsv = _mm512_set1_ps(rs);
    for (int64_t j = 0; j < V; j += 16) {
        const __mmask16 m = tail_mask(V - j);
        const __m512 v = _mm512_mul_ps(load_o(j, m), rsv);
        const __m512 zv = load16_mask<SRC>(z + j, m);
        const __m512 g
                = gate == gdn_norm_gate_t::sigmoid ? sigmoid16(zv) : silu16(zv);
        store16_mask<SRC>(out + j,
                _mm512_mul_ps(
                        _mm512_mul_ps(v, _mm512_maskz_loadu_ps(m, w + j)), g),
                m);
    }
}

} // namespace gdn
} // namespace lowoha
} // namespace zendnnl

GDN_AVX512_END

#endif // _LOWOHA_GDN_NATIVE_COMMON_HPP
