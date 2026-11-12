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

#include <algorithm>
#include <cstring>
#include <omp.h>
#include <vector>
#include "lowoha_operators/common/omp_thread_control.hpp"
#include "lowoha_operators/gdn/kernel/gdn_avx512_utils.hpp"
#include "lowoha_operators/gdn/kernel/gdn_chunk_avx512.hpp"
#include "lowoha_operators/gdn/kernel/gdn_native_common.hpp"
#include "lowoha_operators/gdn/lowoha_gdn_utils.hpp"

GDN_AVX512_BEGIN

namespace zendnnl {
namespace lowoha {
namespace gdn {

namespace {

using namespace avx512;

// Value-tile width of the token-recurrent prefill: a K x 32 FP32 tile
// (16 KB for K = 128) stays in L1 across the whole sequence.
constexpr int64_t kPrefillTile = 32;

// Channels of one unit: q and k of key head h, and the unit's value tile.
// Section s starts at element start[s] of each token row of base[s]: one
// packed mixed_qkv, or separate q, k and v tensors.
struct unit_channels_t {
    const void *base[3];
    int64_t row_stride[3];
    int64_t start[3];
    int64_t count[3];
    int64_t total() const { return count[0] + count[1] + count[2]; }
};

// Tokens ahead whose input slices a unit prefetches.
constexpr int64_t kPrefetchTokens = 8;

inline void prefetch_unit_token(
        const unit_channels_t &ch, int64_t t, size_t elem) {
    for (int s = 0; s < 3; ++s) {
        const char *p = static_cast<const char *>(ch.base[s])
                + (t * ch.row_stride[s] + ch.start[s]) * elem;
        for (int64_t off = 0; off < ch.count[s] * static_cast<int64_t>(elem);
                off += 64) {
            _mm_prefetch(p + off, _MM_HINT_T0);
        }
    }
}

// The FP32 tile [K][VT] of value rows [0, VT) of a [V][K] state matrix S,
// transposed 16 x 16 at a time.
template <data_type_t ST>
inline void vk_tile_load(
        const elem_t<ST> *S, int64_t K, int64_t VT, float *tile) {
    __m512 r[16];
    for (int64_t j0 = 0; j0 < VT; j0 += 16) {
        for (int64_t k0 = 0; k0 < K; k0 += 16) {
            for (int i = 0; i < 16; ++i) {
                r[i] = load16<ST>(S + (j0 + i) * K + k0);
            }
            transpose16x16(r);
            for (int i = 0; i < 16; ++i) {
                _mm512_storeu_ps(tile + (k0 + i) * VT + j0, r[i]);
            }
        }
    }
}

template <data_type_t ST>
inline void vk_tile_store(
        const float *tile, int64_t K, int64_t VT, elem_t<ST> *S) {
    __m512 r[16];
    for (int64_t j0 = 0; j0 < VT; j0 += 16) {
        for (int64_t k0 = 0; k0 < K; k0 += 16) {
            for (int i = 0; i < 16; ++i) {
                r[i] = _mm512_loadu_ps(tile + (k0 + i) * VT + j0);
            }
            transpose16x16(r);
            for (int i = 0; i < 16; ++i) {
                store16<ST>(S + (j0 + i) * K + k0, r[i]);
            }
        }
    }
}

// Depthwise causal conv for the unit's channels at token t. Inputs before the
// sequence start come from hist ([W - 1][nch], oldest first).
template <data_type_t SRC>
inline void conv_unit_token(int64_t t, int64_t t0, const unit_channels_t &ch,
        const float *wt, const float *bias, const float *hist, int64_t W,
        int64_t nch, const gdn_params &p, float *out) {
    int64_t base = 0;
    for (int s = 0; s < 3; ++s) {
        const elem_t<SRC> *in
                = static_cast<const elem_t<SRC> *>(ch.base[s]) + ch.start[s];
        for (int64_t i = 0; i < ch.count[s]; i += 16) {
            const int64_t li = base + i;
            __m512 acc = _mm512_loadu_ps(bias + li);
            for (int64_t j = 0; j < W; ++j) {
                const int64_t tt = t - (W - 1) + j;
                const __m512 x = tt >= t0
                        ? load16<SRC>(in + tt * ch.row_stride[s] + i)
                        : _mm512_loadu_ps(
                                  hist + ((W - 1) + (tt - t0)) * nch + li);
                acc = _mm512_fmadd_ps(
                        _mm512_loadu_ps(wt + j * nch + li), x, acc);
            }
            if (p.silu) { acc = silu16(acc); }
            if (p.round_conv_out) { acc = round16<SRC>(acc); }
            _mm512_storeu_ps(out + li, acc);
        }
        base += ch.count[s];
    }
}

// Sequences [s0, s1) of the batch; sequence si owns token rows
// [offsets[si], offsets[si + 1]). A chunked prefill splits value heads as
// split says.
template <data_type_t SRC, data_type_t ST>
void prefill_typed(const gdn_tensors_t &io, int64_t s0, int64_t s1,
        const int32_t *offsets, bool chunked, const gdn_chunk_split_t &split,
        const gdn_params &p) {
    using T = elem_t<SRC>;
    using S_t = elem_t<ST>;
    constexpr data_type_t ODT = chunk::operand_type<SRC>;

    const int64_t NS = s1 - s0;
    // Token rows of these sequences.
    const int64_t R0 = offsets[s0], R1 = offsets[s1];
    const int32_t *state_indices = io.state_indices;
    const bool *has_initial_state = p.has_initial_state;
    const float *A_log = io.A_log;
    const float *dt_bias = io.dt_bias;
    const void *conv_weight = io.conv_weight;
    const void *conv_bias = io.conv_bias;
    const int64_t H = p.num_k_heads, HV = p.num_v_heads;
    const int64_t K = p.head_k_dim, V = p.head_v_dim;
    const int64_t D = p.conv_dim();
    const int64_t W = p.conv_width;
    const int64_t Wc = W > 0 ? W : 1;
    const int64_t r = HV / H;
    const float scale = p.effective_scale();
    const bool fuse_norm = p.norm_weight != nullptr;
    const int32_t nthr
            = resolve_num_threads(p.num_threads, omp_get_max_threads());

    // A norm on the rounded core output runs as a pass over the stored rows,
    // so value tiles stay narrow; on the FP32 output it needs the whole head
    // row of each token in one unit.
    const bool norm_after = fuse_norm && p.round_core_out;
    // Units u < n_wide take VT value columns: whole heads when chunked. The
    // chunked (sequence, value head) pairs from first_narrow on run in
    // `blocks` units each, which recompute the parts of every chunk that do
    // not depend on the value columns. The chunked form runs only with the
    // norm on the rounded output, i.e. in the pass after all units.
    const int64_t VT
            = (chunked || (fuse_norm && !norm_after) || V % kPrefillTile != 0)
            ? V
            : kPrefillTile;
    const int64_t splits = V / VT;
    const int64_t pairs = NS * HV;
    const int64_t first_narrow
            = chunked ? std::min(split.first_split, pairs) : pairs;
    const int64_t blocks = chunked ? split.blocks : 1;
    const int64_t n_wide = first_narrow * splits;
    const int64_t n_units = n_wide + (pairs - first_narrow) * blocks;

    float *normw = fuse_norm ? scratch_f32(2, static_cast<size_t>(V)) : nullptr;
    if (fuse_norm) { load_f32_row(p.norm_weight, p.norm_weight_dt, V, normw); }

    // Per-thread slab, allocated before the parallel region: wt, bias, hist,
    // xc, tile, qn, kn, orow. Rounded up to 64 bytes so slabs do not share
    // cache lines.
    const int64_t nch_max = 2 * K + VT;
    const int64_t n_hist = std::max<int64_t>(Wc - 1, 1);
    const int64_t slab
            = ((Wc + n_hist + 2) * nch_max + K * VT + 2 * K + V
                      + (chunked ? chunk::scratch_floats<ODT>(K, VT) : 0) + 15)
            / 16 * 16;
    float *slabs = scratch_f32(3, static_cast<size_t>(nthr * slab));

    // Rows R0 + i of the sequences that run (not a pad slot).
    static thread_local std::vector<uint8_t> tok_ok;
    if (norm_after) {
        tok_ok.assign(static_cast<size_t>(R1 - R0), 0);
        for (int64_t si = s0; si < s1; ++si) {
            if (state_indices[si] == p.pad_slot_id) { continue; }
            for (int64_t t = offsets[si]; t < offsets[si + 1]; ++t) {
                tok_ok[t - R0] = 1;
            }
        }
    }
    // The calling thread's buffer, shared with the team below.
    const uint8_t *ok = tok_ok.data();

    const T *in = static_cast<const T *>(io.mixed_qkv);
    const T *a_p = static_cast<const T *>(io.a);
    const T *b_p = static_cast<const T *>(io.b);
    // Precomputed gates: a and b hold g and beta, of their own types.
    const bool pre = p.precomputed_gates();
    const data_type_t a_dt = p.a_type(), b_dt = p.b_type();
    const bool sep = p.separate_qkv();
    const T *z_p = static_cast<const T *>(p.z);
    const T *cw = static_cast<const T *>(conv_weight);
    const T *cb = static_cast<const T *>(conv_bias);
    T *out = static_cast<T *>(io.output);
    S_t *state = static_cast<S_t *>(io.ssm_state);
    T *cs = static_cast<T *>(io.conv_state);

    // Clang outlines a parallel region without this function's target, so
    // there the body runs in a lambda that carries the AVX-512 target. GCC
    // keeps the region as is (a lambda costs it up to 10% on long prefills).
#if defined(__clang__)
    const auto body = [&]() GDN_AVX512_LAMBDA {
#else
#pragma omp parallel num_threads(nthr)
    {
#endif
        float *wt = slabs + omp_get_thread_num() * slab;
        float *bias = wt + Wc * nch_max;
        float *hist = bias + nch_max;
        float *xc = hist + n_hist * nch_max;
        float *tile = xc + nch_max;
        float *qn = tile + K * VT;
        float *kn = qn + K;
        float *orow = kn + K;
        float *cbase = orow + V;

#pragma omp for schedule(dynamic, 1)
        for (int64_t u = 0; u < n_units; ++u) {
            // Pair (si, hv), value columns [v0, v0 + vt).
            const bool wide = u < n_wide;
            const int64_t pair
                    = wide ? u / splits : first_narrow + (u - n_wide) / blocks;
            const int64_t vt = wide ? VT : V / blocks;
            const int64_t v0 = (wide ? u % splits : (u - n_wide) % blocks) * vt;
            const int64_t si = s0 + pair / HV;
            const int64_t hv = pair % HV;
            const int32_t slot = state_indices[si];
            const int64_t t0 = offsets[si], t1 = offsets[si + 1];
            if (slot == p.pad_slot_id || t1 <= t0) { continue; }
            const bool init
                    = has_initial_state != nullptr && has_initial_state[si];
            const int64_t h = hv / r;

            const unit_channels_t ch = sep
                    ? unit_channels_t {{in, p.k, p.v},
                              {p.qkv_stride, p.k_stride_t, p.v_stride_t},
                              {h * p.q_stride_h, h * p.k_stride_h,
                                      hv * p.v_stride_h + v0},
                              {K, K, vt}}
                    : unit_channels_t {{in, in, in},
                              {p.qkv_stride, p.qkv_stride, p.qkv_stride},
                              {h * K, H * K + h * K, 2 * H * K + hv * V + v0},
                              {K, K, vt}};
            const int64_t nch = ch.total();
            // The unit's conv weights, bias and history by tap, in FP32.
            int64_t li = 0;
            for (int s = 0; s < 3; ++s) {
                const int64_t c0 = ch.start[s], cnt = ch.count[s];
                if (W > 0) {
                    conv_taps<SRC>(cw + c0 * W, cnt, W, wt + li, nch);
                    row_f32<SRC>(cb ? cb + c0 : nullptr, cnt, bias + li);
                } else {
                    std::fill(wt + li, wt + li + cnt, 1.0f);
                    std::fill(bias + li, bias + li + cnt, 0.0f);
                }
                for (int64_t j = 0; j < W - 1; ++j) {
                    float *hd = hist + j * nch + li;
                    if (!init) {
                        row_f32<SRC>(nullptr, cnt, hd);
                        continue;
                    }
                    const T *hs = cs + slot * p.conv_state_slot_stride
                            + j * p.conv_state_tap_stride
                            + c0 * p.conv_state_dim_stride;
                    if (p.conv_state_dim_stride == 1) {
                        row_f32<SRC>(hs, cnt, hd);
                        continue;
                    }
                    for (int64_t i = 0; i < cnt; ++i) {
                        hd[i] = load1<SRC>(hs + i * p.conv_state_dim_stride);
                    }
                }
                li += cnt;
            }

            // The pool may be null when this unit neither reads nor writes it.
            S_t *S = init || p.update_state
                    ? state + slot * p.ssm_state_slot_stride
                            + hv * p.ssm_state_head_stride
                    : nullptr;
            const bool kv = p.state_layout == gdn_state_layout_t::kv;
            // Local FP32 tile, always [K][vt]. vt is a multiple of 16.
            if (!init) {
                std::memset(tile, 0, sizeof(float) * K * vt);
            } else if (kv) {
                for (int64_t k = 0; k < K; ++k) {
                    for (int64_t j = 0; j < vt; j += 16) {
                        _mm512_storeu_ps(tile + k * vt + j,
                                load16<ST>(S + k * V + v0 + j));
                    }
                }
            } else {
                vk_tile_load<ST>(S + v0 * K, K, vt, tile);
            }

            gdn_params pc = p;
            if (W == 0) {
                pc.silu = false;
                pc.round_conv_out = false;
            }
            if (chunked) {
                chunk::chunk_scratch_t<ODT> sc(cbase, K, vt);
                alignas(64) float ga[chunk::kChunk], gb[chunk::kChunk];
                for (int64_t c0 = t0; c0 < t1; c0 += chunk::kChunk) {
                    const int64_t n = std::min(chunk::kChunk, t1 - c0);
                    for (int64_t i = 0; i < chunk::kChunk; ++i) {
                        float *qi = sc.in.q + i * K;
                        float *ki = sc.in.k + i * K;
                        float *vi = sc.in.v + i * vt;
                        if (i >= n) {
                            std::memset(qi, 0, sizeof(float) * K);
                            std::memset(ki, 0, sizeof(float) * K);
                            std::memset(vi, 0, sizeof(float) * vt);
                            continue;
                        }
                        const int64_t t = c0 + i;
                        // Rows are a full qkv stride apart, too far for the
                        // hardware prefetcher; fetch this unit's slices ahead.
                        if (t + kPrefetchTokens < t1) {
                            prefetch_unit_token(
                                    ch, t + kPrefetchTokens, sizeof(T));
                        }
                        conv_unit_token<SRC>(
                                t, t0, ch, wt, bias, hist, Wc, nch, pc, xc);
                        prepare_qk(xc, xc + K, K, p.use_qk_l2norm, p.l2norm_eps,
                                scale, qi, ki);
                        std::memcpy(vi, xc + 2 * K, sizeof(float) * vt);
                        if (pre) {
                            ga[i] = load_f32(io.a, a_dt, t * p.a_stride + hv);
                            gb[i] = load_f32(io.b, b_dt, t * p.b_stride + hv);
                        } else {
                            ga[i] = load1<SRC>(a_p + t * p.a_stride + hv);
                            gb[i] = load1<SRC>(b_p + t * p.b_stride + hv);
                        }
                    }
                    if (pre) {
                        // g and beta as given; zero past the chunk's tokens.
                        for (int64_t i = 0; i < chunk::kChunk; ++i) {
                            sc.in.g[i] = i < n ? ga[i] : 0.0f;
                            sc.in.beta[i] = i < n ? gb[i] : 0.0f;
                        }
                    } else {
                        chunk::gates(ga, gb, n, A_log[hv], dt_bias[hv],
                                p.softplus_threshold, sc.in.g, sc.in.beta);
                    }
                    chunk::chunk_step<ODT>(sc, n, K, vt, tile);
                    for (int64_t i = 0; i < n; ++i) {
                        T *o_dst = out + (c0 + i) * p.out_stride_t
                                + hv * p.out_stride_h + v0;
                        for (int64_t j = 0; j < vt; j += 16) {
                            store16<SRC>(o_dst + j,
                                    _mm512_loadu_ps(sc.o + i * vt + j));
                        }
                    }
                }
            }
            for (int64_t t = t0; !chunked && t < t1; ++t) {
                conv_unit_token<SRC>(
                        t, t0, ch, wt, bias, hist, Wc, nch, pc, xc);
                prepare_qk(xc, xc + K, K, p.use_qk_l2norm, p.l2norm_eps, scale,
                        qn, kn);
                float eg, beta;
                if (pre) {
                    eg = std::exp(load_f32(io.a, a_dt, t * p.a_stride + hv));
                    beta = load_f32(io.b, b_dt, t * p.b_stride + hv);
                } else {
                    gate_scalars(load1<SRC>(a_p + t * p.a_stride + hv),
                            load1<SRC>(b_p + t * p.b_stride + hv), A_log[hv],
                            dt_bias[hv], p.softplus_threshold, eg, beta);
                }
                const float *vin = xc + 2 * K;
                for (int64_t j = 0; j < vt; j += kPrefillTile) {
                    __m512 o[2];
                    if (vt - j >= 32) {
                        kv_tile_step<data_type_t::f32, 2>(
                                tile + j, vt, K, vin + j, qn, kn, eg, beta, o);
                        _mm512_storeu_ps(orow + j, o[0]);
                        _mm512_storeu_ps(orow + j + 16, o[1]);
                    } else {
                        for (int64_t jj = j; jj < vt; jj += 16) {
                            kv_tile_step<data_type_t::f32, 1>(tile + jj, vt, K,
                                    vin + jj, qn, kn, eg, beta, o);
                            _mm512_storeu_ps(orow + jj, o[0]);
                        }
                    }
                }
                T *o_dst = out + t * p.out_stride_t + hv * p.out_stride_h + v0;
                if (fuse_norm && !norm_after) {
                    gated_rmsnorm_row<SRC>(orow,
                            z_p + t * p.z_stride_t + hv * p.z_stride_h, normw,
                            V, p.norm_eps, false, p.norm_gate, o_dst);
                } else {
                    for (int64_t j = 0; j < vt; j += 16) {
                        store16<SRC>(o_dst + j, _mm512_loadu_ps(orow + j));
                    }
                }
            }

            if (!p.update_state) { continue; }
            if (kv) {
                for (int64_t k = 0; k < K; ++k) {
                    for (int64_t j = 0; j < vt; j += 16) {
                        store16<ST>(S + k * V + v0 + j,
                                _mm512_loadu_ps(tile + k * vt + j));
                    }
                }
            } else {
                vk_tile_store<ST>(tile, K, vt, S + v0 * K);
            }
        }

        // The norm reads every value tile of a row, so it runs after the
        // barrier above.
        if (norm_after) {
#pragma omp for schedule(static)
            for (int64_t u = 0; u < (R1 - R0) * HV; ++u) {
                const int64_t t = R0 + u / HV, hv = u % HV;
                if (!ok[t - R0]) { continue; }
                T *row = out + t * p.out_stride_t + hv * p.out_stride_h;
                gated_rmsnorm_row<SRC>(row,
                        z_p + t * p.z_stride_t + hv * p.z_stride_h, normw, V,
                        p.norm_eps, false, p.norm_gate, row);
            }
        }

        // Final conv state: the last W - 1 raw inputs of each sequence. Runs
        // after the barrier above because every unit reads the initial state.
        if (W > 1 && p.update_state) {
            const int64_t n_cblk = (D + kConvBlock - 1) / kConvBlock;
#pragma omp for schedule(static)
            for (int64_t u = 0; u < NS * n_cblk; ++u) {
                const int64_t si = s0 + u / n_cblk;
                const int64_t c0 = (u % n_cblk) * kConvBlock;
                const int64_t nc = std::min<int64_t>(kConvBlock, D - c0);
                const int32_t slot = state_indices[si];
                const int64_t t0 = offsets[si];
                const int64_t t1 = offsets[si + 1];
                if (slot == p.pad_slot_id || t1 <= t0) { continue; }
                const bool init
                        = has_initial_state != nullptr && has_initial_state[si];
                const bool contig = p.conv_state_dim_stride == 1;
                for (int64_t j = 0; j < W - 1; ++j) {
                    const int64_t tt = t1 - (W - 1) + j;
                    T *row = cs + slot * p.conv_state_slot_stride
                            + j * p.conv_state_tap_stride + c0;
                    // Tap j only reads taps > j, which are written later.
                    if (contig && tt >= t0) {
                        std::memcpy(row, in + tt * p.qkv_stride + c0,
                                sizeof(T) * nc);
                        continue;
                    }
                    if (contig && init) {
                        std::memmove(row,
                                row
                                        + ((W - 1) + (tt - t0) - j)
                                                * p.conv_state_tap_stride,
                                sizeof(T) * nc);
                        continue;
                    }
                    if (contig) {
                        std::memset(row, 0, sizeof(T) * nc);
                        continue;
                    }
                    for (int64_t c = c0; c < c0 + nc; ++c) {
                        T *dst = cs + slot * p.conv_state_slot_stride
                                + j * p.conv_state_tap_stride
                                + c * p.conv_state_dim_stride;
                        if (tt >= t0) {
                            *dst = in[tt * p.qkv_stride + c];
                        } else if (init) {
                            *dst = cs[slot * p.conv_state_slot_stride
                                    + ((W - 1) + (tt - t0))
                                            * p.conv_state_tap_stride
                                    + c * p.conv_state_dim_stride];
                        } else {
                            *dst = T();
                        }
                    }
                }
            }
        }
#if defined(__clang__)
    };
#pragma omp parallel num_threads(nthr)
    body();
#else
    }
#endif
}

} // namespace

status_t gdn_prefill_native(const gdn_tensors_t &io, int64_t s0, int64_t s1,
        const int32_t *offsets, gdn_prefill_algo_t algo,
        const gdn_chunk_split_t &split, const gdn_params &p) {
    const bool chunked = algo == gdn_prefill_algo_t::chunked;
#define GDN_PREFILL_CASE(SRC, ST) \
    if (p.src_dt == SRC && p.state_dt == ST) { \
        prefill_typed<SRC, ST>(io, s0, s1, offsets, chunked, split, p); \
        return status_t::success; \
    }
    GDN_PREFILL_CASE(data_type_t::bf16, data_type_t::f32)
    GDN_PREFILL_CASE(data_type_t::bf16, data_type_t::bf16)
    GDN_PREFILL_CASE(data_type_t::bf16, data_type_t::f16)
    GDN_PREFILL_CASE(data_type_t::f16, data_type_t::f32)
    GDN_PREFILL_CASE(data_type_t::f16, data_type_t::bf16)
    GDN_PREFILL_CASE(data_type_t::f16, data_type_t::f16)
    GDN_PREFILL_CASE(data_type_t::f32, data_type_t::f32)
    GDN_PREFILL_CASE(data_type_t::f32, data_type_t::bf16)
    GDN_PREFILL_CASE(data_type_t::f32, data_type_t::f16)
#undef GDN_PREFILL_CASE
    return status_t::unimplemented;
}

} // namespace gdn
} // namespace lowoha
} // namespace zendnnl

GDN_AVX512_END
