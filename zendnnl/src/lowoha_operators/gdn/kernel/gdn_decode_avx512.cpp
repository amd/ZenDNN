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
#include "lowoha_operators/gdn/kernel/gdn_native_common.hpp"
#include "lowoha_operators/gdn/lowoha_gdn_utils.hpp"

GDN_AVX512_BEGIN

namespace zendnnl {
namespace lowoha {
namespace gdn {

namespace {

using namespace avx512;

// Delta-rule update of one (sequence, value head, value tile) with the
// [K][V] state layout, NV * 16 value columns, read from S_in and written to
// S_out (which may be S_in). The state streams from memory once in each
// direction.
template <data_type_t SRC, data_type_t ST, int NV>
void kv_tile(const elem_t<ST> *S_in, elem_t<ST> *S_out, int64_t K, int64_t V,
        const float *vin, const float *qn, const float *kn, float eg,
        float beta, float *o_f32, elem_t<SRC> *o_dst) {
    __m512 o[NV];
    kv_tile_step<ST, NV>(S_in, S_out, V, K, vin, qn, kn, eg, beta, o);
    for (int n = 0; n < NV; ++n) {
        if (o_f32 != nullptr) {
            _mm512_storeu_ps(o_f32 + 16 * n, o[n]);
        } else {
            store16<SRC>(o_dst + 16 * n, o[n]);
        }
    }
}

// [V][K] layout: each value row is contiguous over K and is finished with a
// single read and a single write. Outputs are stored 16 rows at a time (VT
// is a multiple of 16).
template <data_type_t SRC, data_type_t ST>
void vk_tile(const elem_t<ST> *S_in, elem_t<ST> *S_out, int64_t K, int64_t VT,
        const float *vin, const float *qn, const float *kn, float eg,
        float beta, float *o_f32, elem_t<SRC> *o_dst) {
    const __m512 egv = _mm512_set1_ps(eg);
    alignas(64) float ob[16];
    for (int64_t j = 0; j < VT; ++j) {
        const elem_t<ST> *row = S_in + j * K;
        elem_t<ST> *row_out = S_out + j * K;
        __m512 acc = _mm512_setzero_ps();
        for (int64_t k = 0; k < K; k += 16) {
            acc = _mm512_fmadd_ps(
                    load16<ST>(row + k), _mm512_loadu_ps(kn + k), acc);
        }
        const float dtv = (vin[j] - _mm512_reduce_add_ps(acc) * eg) * beta;
        const __m512 dtvv = _mm512_set1_ps(dtv);
        __m512 o = _mm512_setzero_ps();
        for (int64_t k = 0; k < K; k += 16) {
            const __m512 s = _mm512_fmadd_ps(load16<ST>(row + k), egv,
                    _mm512_mul_ps(_mm512_loadu_ps(kn + k), dtvv));
            store16<ST>(row_out + k, s);
            o = _mm512_fmadd_ps(s, _mm512_loadu_ps(qn + k), o);
        }
        ob[j % 16] = _mm512_reduce_add_ps(o);
        if (j % 16 == 15) {
            if (o_f32 != nullptr) {
                _mm512_storeu_ps(o_f32 + j - 15, _mm512_load_ps(ob));
            } else {
                store16<SRC>(o_dst + j - 15, _mm512_load_ps(ob));
            }
        }
    }
}

// Value columns [v0, v0 + VT) of one (slot, value head) state, read from S_in
// and written to S_out (which may be S_in), in either layout.
template <data_type_t SRC, data_type_t ST>
void state_tile(const elem_t<ST> *S_in, elem_t<ST> *S_out, int64_t v0,
        int64_t VT, const float *vin, const float *qn, const float *kn,
        float eg, float beta, float *o_f32, elem_t<SRC> *o_dst,
        const gdn_params &p) {
    const int64_t K = p.head_k_dim, V = p.head_v_dim;
    if (p.state_layout != gdn_state_layout_t::kv) {
        vk_tile<SRC, ST>(S_in + v0 * K, S_out + v0 * K, K, VT, vin, qn, kn, eg,
                beta, o_f32, o_dst);
        return;
    }
    S_in += v0;
    S_out += v0;
    switch (VT / 16) {
        case 8:
            kv_tile<SRC, ST, 8>(
                    S_in, S_out, K, V, vin, qn, kn, eg, beta, o_f32, o_dst);
            break;
        case 4:
            kv_tile<SRC, ST, 4>(
                    S_in, S_out, K, V, vin, qn, kn, eg, beta, o_f32, o_dst);
            break;
        case 2:
            kv_tile<SRC, ST, 2>(
                    S_in, S_out, K, V, vin, qn, kn, eg, beta, o_f32, o_dst);
            break;
        default:
            for (int64_t t = 0; t < VT; t += 16) {
                kv_tile<SRC, ST, 1>(S_in + t, S_out + t, K, V, vin + t, qn, kn,
                        eg, beta, o_f32 ? o_f32 + t : nullptr, o_dst + t);
            }
            break;
    }
}

template <data_type_t SRC, data_type_t ST>
void decode_typed(const gdn_tensors_t &io, int64_t n, const gdn_params &p) {
    using T = elem_t<SRC>;
    using S_t = elem_t<ST>;

    const int64_t B = n;
    const int32_t *state_indices = io.state_indices;
    const float *A_log = io.A_log;
    const float *dt_bias = io.dt_bias;
    const int64_t H = p.num_k_heads, HV = p.num_v_heads;
    const int64_t K = p.head_k_dim, V = p.head_v_dim;
    const int64_t D = p.conv_dim();
    const int64_t r = HV / H;
    const float scale = p.effective_scale();
    const bool fuse_norm = p.norm_weight != nullptr;
    const int32_t nthr
            = resolve_num_threads(p.num_threads, omp_get_max_threads());

    const int64_t splits = select_v_splits(
            B * HV, V, nthr, p.state_layout == gdn_state_layout_t::kv);
    const int64_t VT = V / splits;

    float *xbuf = scratch_f32(0, static_cast<size_t>(B * D));
    float *obuf = fuse_norm ? scratch_f32(1, static_cast<size_t>(B * HV * V))
                            : nullptr;
    float *normw = fuse_norm ? scratch_f32(2, static_cast<size_t>(V)) : nullptr;
    if (fuse_norm) { load_f32_row(p.norm_weight, p.norm_weight_dt, V, normw); }

    const int64_t n_cblk = (D + kConvBlock - 1) / kConvBlock;
    const int64_t n_units = B * HV * splits;

    const T *in = static_cast<const T *>(io.mixed_qkv);
    const T *a_p = static_cast<const T *>(io.a);
    const T *b_p = static_cast<const T *>(io.b);
    T *out = static_cast<T *>(io.output);
    S_t *state = static_cast<S_t *>(io.ssm_state);

    // Phase 1 updates the conv per (channel block, sequence), phase 2 runs
    // the recurrence and phase 3 the fused norm, in one parallel region.
#pragma omp parallel num_threads(nthr)
    {
#pragma omp for schedule(static)
        for (int64_t cb = 0; cb < n_cblk; ++cb) {
            const int64_t c0 = cb * kConvBlock;
            const int64_t nc = std::min<int64_t>(kConvBlock, D - c0);
            conv_update_block<SRC>(c0, nc, B, in, p.qkv_stride, io.conv_weight,
                    io.conv_bias, io.conv_state, state_indices, xbuf, p);
        }

        // Phase 2: recurrence for every (sequence, value head, value tile).
        alignas(64) float qn[kMaxHeadDim];
        alignas(64) float kn[kMaxHeadDim];
        int64_t last_bh = -1;
#pragma omp for schedule(static)
        for (int64_t u = 0; u < n_units; ++u) {
            const int64_t bi = u / (HV * splits);
            const int64_t rem = u % (HV * splits);
            const int64_t hv = rem / splits;
            const int64_t v0 = (rem % splits) * VT;
            const int32_t slot = state_indices[bi];
            float *o_f32 = fuse_norm ? obuf + (bi * HV + hv) * V + v0 : nullptr;
            T *o_dst = out + bi * p.out_stride_t + hv * p.out_stride_h + v0;
            if (slot == p.pad_slot_id) {
                if (!fuse_norm) { std::memset(o_dst, 0, VT * sizeof(T)); }
                continue;
            }
            const float *xrow = xbuf + bi * D;
            const int64_t h = hv / r;
            const int64_t bh = bi * H + h;
            if (bh != last_bh) {
                prepare_qk(xrow + h * K, xrow + H * K + h * K, K,
                        p.use_qk_l2norm, p.l2norm_eps, scale, qn, kn);
                last_bh = bh;
            }
            float eg, beta;
            gate_scalars(load1<SRC>(a_p + bi * p.a_stride + hv),
                    load1<SRC>(b_p + bi * p.b_stride + hv), A_log[hv],
                    dt_bias[hv], p.softplus_threshold, p.beta_scale(), eg,
                    beta);

            S_t *S = state + slot * p.ssm_state_slot_stride
                    + hv * p.ssm_state_head_stride;
            state_tile<SRC, ST>(S, S, v0, VT, xrow + 2 * H * K + hv * V + v0,
                    qn, kn, eg, beta, o_f32, o_dst, p);
        }

        // Phase 3: gated RMSNorm on each (sequence, value head).
        if (fuse_norm) {
            const T *z_p = static_cast<const T *>(p.z);
#pragma omp for schedule(static)
            for (int64_t u = 0; u < B * HV; ++u) {
                const int64_t bi = u / HV, hv = u % HV;
                T *dst = out + bi * p.out_stride_t + hv * p.out_stride_h;
                // A pad slot's z row may hold anything, NaN included.
                if (state_indices[bi] == p.pad_slot_id) {
                    std::memset(dst, 0, V * sizeof(T));
                    continue;
                }
                gated_rmsnorm_row<SRC>(obuf + u * V,
                        z_p + bi * p.z_stride_t + hv * p.z_stride_h, normw, V,
                        p.norm_eps, p.round_core_out, p.norm_gate, dst);
            }
        }
    }
}

// Conv of every verify sequence's tokens for channels [c0, c0 + nc) (see
// gdn_params::spec_state_indices): the conv_width - 1 taps of its conv state
// from tap a - 1 on, a its accepted tokens, then its tokens, in the
// arithmetic decode takes on a conv state of the same strides. Afterwards the
// conv_width - 2 taps from tap a move to the front and its q tokens follow.
// hw holds (conv_width - 1 + spec_state_stride) * kConvBlock floats.
template <data_type_t SRC>
void conv_verify_block(int64_t c0, int64_t nc, int64_t B,
        const int32_t *offsets, const elem_t<SRC> *in, const void *conv_weight,
        const void *conv_bias, void *conv_state, float *xbuf, float *hw,
        const gdn_params &p) {
    using T = elem_t<SRC>;
    const int64_t W = p.conv_width;
    const int64_t D = p.conv_dim();

    if (W == 0) {
        for (int64_t row = offsets[0]; row < offsets[B]; ++row) {
            const T *x = in + row * p.qkv_stride + c0;
            float *xo = xbuf + row * D + c0;
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
    const int64_t ts = p.conv_state_tap_stride;
    const int64_t ds = p.conv_state_dim_stride;
    const int64_t nt = W - 1;
    const bool vector = conv16_layout(ts, ds, nt);
    for (int64_t bi = 0; bi < B; ++bi) {
        const int64_t r0 = offsets[bi], q = offsets[bi + 1] - r0;
        const int32_t *slots = p.spec_state_indices + bi * p.spec_state_stride;
        if (q <= 0 || slots[0] == p.pad_slot_id) { continue; }
        const int64_t a = p.num_accepted_tokens[bi];
        T *base = cs + slots[0] * p.conv_state_slot_stride + c0 * ds;
        for (int64_t j = 0; j < nt; ++j) {
            const T *tap = base + (a - 1 + j) * ts;
            float *dst = hw + j * kConvBlock;
            if (ds == 1) {
                row_f32<SRC>(tap, nc, dst);
                continue;
            }
            for (int64_t c = 0; c < nc; ++c) {
                dst[c] = load1<SRC>(tap + c * ds);
            }
        }
        for (int64_t t = 0; t < q; ++t) {
            row_f32<SRC>(in + (r0 + t) * p.qkv_stride + c0, nc,
                    hw + (nt + t) * kConvBlock);
        }
        for (int64_t t = 0; t < q; ++t) {
            const T *x = in + (r0 + t) * p.qkv_stride + c0;
            float *xo = xbuf + (r0 + t) * D + c0;
            if (!vector) {
                const float *h = hw + t * kConvBlock;
                for (int64_t c = 0; c < nc; ++c) {
                    xo[c] = conv1<SRC>(wt, bias, c, nt, [h, c](int64_t j) {
                        return h[j * kConvBlock + c];
                    }, h[nt * kConvBlock + c], p);
                }
                continue;
            }
            for (int64_t n = 0; n < nc; n += 16) {
                const __mmask16 m = tail_mask(nc - n);
                _mm512_mask_storeu_ps(xo + n, m,
                        conv16<SRC, data_type_t::f32>(wt, bias, nt,
                                hw + t * kConvBlock, kConvBlock, x, n, m, p));
            }
        }
        if (nt == 0) { continue; }
        // Tap a + j lands on tap j, which is no later than it, so the move
        // runs forward in place.
        const int64_t keep = nt - 1;
        for (int64_t j = 0; j < keep; ++j) {
            T *dst = base + j * ts;
            const T *src = base + (a + j) * ts;
            if (ds == 1) {
                std::memmove(dst, src, nc * sizeof(T));
                continue;
            }
            for (int64_t c = 0; c < nc; ++c) {
                dst[c * ds] = src[c * ds];
            }
        }
        for (int64_t t = 0; t < q; ++t) {
            const T *x = in + (r0 + t) * p.qkv_stride + c0;
            T *dst = base + (keep + t) * ts;
            if (ds == 1) {
                std::memcpy(dst, x, nc * sizeof(T));
                continue;
            }
            for (int64_t c = 0; c < nc; ++c) {
                dst[c * ds] = x[c];
            }
        }
    }
}

template <data_type_t SRC, data_type_t ST>
void verify_typed(const gdn_tensors_t &io, int64_t B, const int32_t *offsets,
        const gdn_params &p) {
    using T = elem_t<SRC>;
    using S_t = elem_t<ST>;

    const float *A_log = io.A_log;
    const float *dt_bias = io.dt_bias;
    const int64_t H = p.num_k_heads, HV = p.num_v_heads;
    const int64_t K = p.head_k_dim, V = p.head_v_dim;
    const int64_t D = p.conv_dim();
    const int64_t r = HV / H;
    const int64_t SS = p.spec_state_stride;
    const int64_t NT = offsets[B];
    const float scale = p.effective_scale();
    const bool fuse_norm = p.norm_weight != nullptr;
    const int32_t nthr
            = resolve_num_threads(p.num_threads, omp_get_max_threads());

    // Value tiles as a decode call over the same sequences takes them, so a
    // verify call gives the bits of that many decode calls.
    const int64_t splits = select_v_splits(
            B * HV, V, nthr, p.state_layout == gdn_state_layout_t::kv);
    const int64_t VT = V / splits;

    float *xbuf = scratch_f32(0, static_cast<size_t>(NT * D));
    float *obuf = fuse_norm ? scratch_f32(1, static_cast<size_t>(NT * HV * V))
                            : nullptr;
    float *normw = fuse_norm ? scratch_f32(2, static_cast<size_t>(V)) : nullptr;
    if (fuse_norm) { load_f32_row(p.norm_weight, p.norm_weight_dt, V, normw); }
    const int64_t hw_n
            = (std::max<int64_t>(p.conv_width - 1, 0) + SS) * kConvBlock;
    float *hws = scratch_f32(3, static_cast<size_t>(nthr * hw_n));

    const int64_t n_cblk = (D + kConvBlock - 1) / kConvBlock;
    const int64_t n_units = B * HV * splits;

    const T *in = static_cast<const T *>(io.mixed_qkv);
    const T *a_p = static_cast<const T *>(io.a);
    const T *b_p = static_cast<const T *>(io.b);
    T *out = static_cast<T *>(io.output);
    S_t *state = static_cast<S_t *>(io.ssm_state);

#pragma omp parallel num_threads(nthr)
    {
        float *hw = hws + omp_get_thread_num() * hw_n;
#pragma omp for schedule(static)
        for (int64_t cb = 0; cb < n_cblk; ++cb) {
            const int64_t c0 = cb * kConvBlock;
            conv_verify_block<SRC>(c0, std::min<int64_t>(kConvBlock, D - c0), B,
                    offsets, in, io.conv_weight, io.conv_bias, io.conv_state,
                    xbuf, hw, p);
        }

        // Every (sequence, value head, value tile) runs its tokens in order:
        // token t reads the state token t - 1 wrote (the accepted one for the
        // first) and writes its own slot.
        alignas(64) float qn[kMaxHeadDim];
        alignas(64) float kn[kMaxHeadDim];
#pragma omp for schedule(static)
        for (int64_t u = 0; u < n_units; ++u) {
            const int64_t bi = u / (HV * splits);
            const int64_t rem = u % (HV * splits);
            const int64_t hv = rem / splits;
            const int64_t v0 = (rem % splits) * VT;
            const int64_t r0 = offsets[bi], q = offsets[bi + 1] - r0;
            const int32_t *slots = p.spec_state_indices + bi * SS;
            if (q <= 0) { continue; }
            if (slots[0] == p.pad_slot_id) {
                for (int64_t t = 0; !fuse_norm && t < q; ++t) {
                    std::memset(out + (r0 + t) * p.out_stride_t
                                    + hv * p.out_stride_h + v0,
                            0, VT * sizeof(T));
                }
                continue;
            }
            const int64_t h = hv / r;
            const int64_t head = hv * p.ssm_state_head_stride;
            const S_t *S_in = state
                    + slots[p.num_accepted_tokens[bi] - 1]
                            * p.ssm_state_slot_stride
                    + head;
            for (int64_t t = 0; t < q; ++t) {
                const int64_t row = r0 + t;
                const float *xrow = xbuf + row * D;
                prepare_qk(xrow + h * K, xrow + H * K + h * K, K,
                        p.use_qk_l2norm, p.l2norm_eps, scale, qn, kn);
                float eg, beta;
                gate_scalars(load1<SRC>(a_p + row * p.a_stride + hv),
                        load1<SRC>(b_p + row * p.b_stride + hv), A_log[hv],
                        dt_bias[hv], p.softplus_threshold, p.beta_scale(), eg,
                        beta);
                S_t *S_out = state + slots[t] * p.ssm_state_slot_stride + head;
                state_tile<SRC, ST>(S_in, S_out, v0, VT,
                        xrow + 2 * H * K + hv * V + v0, qn, kn, eg, beta,
                        fuse_norm ? obuf + (row * HV + hv) * V + v0 : nullptr,
                        out + row * p.out_stride_t + hv * p.out_stride_h + v0,
                        p);
                S_in = S_out;
            }
        }

        if (fuse_norm) {
            const T *z_p = static_cast<const T *>(p.z);
#pragma omp for schedule(static)
            for (int64_t u = 0; u < B * HV; ++u) {
                const int64_t bi = u / HV, hv = u % HV;
                const bool pad = p.spec_state_indices[bi * SS] == p.pad_slot_id;
                for (int64_t row = offsets[bi]; row < offsets[bi + 1]; ++row) {
                    T *dst = out + row * p.out_stride_t + hv * p.out_stride_h;
                    if (pad) {
                        std::memset(dst, 0, V * sizeof(T));
                        continue;
                    }
                    gated_rmsnorm_row<SRC>(obuf + (row * HV + hv) * V,
                            z_p + row * p.z_stride_t + hv * p.z_stride_h, normw,
                            V, p.norm_eps, p.round_core_out, p.norm_gate, dst);
                }
            }
        }
    }
}

} // namespace

status_t gdn_decode_native(
        const gdn_tensors_t &io, int64_t n, const gdn_params &p) {
#define GDN_DECODE_CASE(SRC, ST) \
    if (p.src_dt == SRC && p.state_dt == ST) { \
        decode_typed<SRC, ST>(io, n, p); \
        return status_t::success; \
    }
    GDN_DECODE_CASE(data_type_t::bf16, data_type_t::f32)
    GDN_DECODE_CASE(data_type_t::bf16, data_type_t::bf16)
    GDN_DECODE_CASE(data_type_t::bf16, data_type_t::f16)
    GDN_DECODE_CASE(data_type_t::f16, data_type_t::f32)
    GDN_DECODE_CASE(data_type_t::f16, data_type_t::bf16)
    GDN_DECODE_CASE(data_type_t::f16, data_type_t::f16)
    GDN_DECODE_CASE(data_type_t::f32, data_type_t::f32)
    GDN_DECODE_CASE(data_type_t::f32, data_type_t::bf16)
    GDN_DECODE_CASE(data_type_t::f32, data_type_t::f16)
#undef GDN_DECODE_CASE
    return status_t::unimplemented;
}

status_t gdn_verify_native(const gdn_tensors_t &io, int64_t n,
        const int32_t *offsets, const gdn_params &p) {
#define GDN_VERIFY_CASE(SRC, ST) \
    if (p.src_dt == SRC && p.state_dt == ST) { \
        verify_typed<SRC, ST>(io, n, offsets, p); \
        return status_t::success; \
    }
    GDN_VERIFY_CASE(data_type_t::bf16, data_type_t::f32)
    GDN_VERIFY_CASE(data_type_t::bf16, data_type_t::bf16)
    GDN_VERIFY_CASE(data_type_t::bf16, data_type_t::f16)
    GDN_VERIFY_CASE(data_type_t::f16, data_type_t::f32)
    GDN_VERIFY_CASE(data_type_t::f16, data_type_t::bf16)
    GDN_VERIFY_CASE(data_type_t::f16, data_type_t::f16)
    GDN_VERIFY_CASE(data_type_t::f32, data_type_t::f32)
    GDN_VERIFY_CASE(data_type_t::f32, data_type_t::bf16)
    GDN_VERIFY_CASE(data_type_t::f32, data_type_t::f16)
#undef GDN_VERIFY_CASE
    return status_t::unimplemented;
}

} // namespace gdn
} // namespace lowoha
} // namespace zendnnl

GDN_AVX512_END
