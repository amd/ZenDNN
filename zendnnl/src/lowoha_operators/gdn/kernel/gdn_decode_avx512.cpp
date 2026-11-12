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
// [K][V] state layout, NV * 16 value columns. The state streams from memory
// once in each direction.
template <data_type_t SRC, data_type_t ST, int NV>
void kv_tile(elem_t<ST> *S, int64_t K, int64_t V, const float *vin,
        const float *qn, const float *kn, float eg, float beta, float *o_f32,
        elem_t<SRC> *o_dst) {
    __m512 o[NV];
    kv_tile_step<ST, NV>(S, V, K, vin, qn, kn, eg, beta, o);
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
void vk_tile(elem_t<ST> *S, int64_t K, int64_t VT, const float *vin,
        const float *qn, const float *kn, float eg, float beta, float *o_f32,
        elem_t<SRC> *o_dst) {
    const __m512 egv = _mm512_set1_ps(eg);
    alignas(64) float ob[16];
    for (int64_t j = 0; j < VT; ++j) {
        elem_t<ST> *row = S + j * K;
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
            store16<ST>(row + k, s);
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
                    dt_bias[hv], p.softplus_threshold, eg, beta);

            S_t *S = state + slot * p.ssm_state_slot_stride
                    + hv * p.ssm_state_head_stride;
            const float *vin = xrow + 2 * H * K + hv * V + v0;
            if (p.state_layout == gdn_state_layout_t::kv) {
                S += v0;
                switch (VT / 16) {
                    case 8:
                        kv_tile<SRC, ST, 8>(
                                S, K, V, vin, qn, kn, eg, beta, o_f32, o_dst);
                        break;
                    case 4:
                        kv_tile<SRC, ST, 4>(
                                S, K, V, vin, qn, kn, eg, beta, o_f32, o_dst);
                        break;
                    case 2:
                        kv_tile<SRC, ST, 2>(
                                S, K, V, vin, qn, kn, eg, beta, o_f32, o_dst);
                        break;
                    default:
                        for (int64_t t = 0; t < VT; t += 16) {
                            kv_tile<SRC, ST, 1>(S + t, K, V, vin + t, qn, kn,
                                    eg, beta, o_f32 ? o_f32 + t : nullptr,
                                    o_dst + t);
                        }
                        break;
                }
            } else {
                vk_tile<SRC, ST>(
                        S + v0 * K, K, VT, vin, qn, kn, eg, beta, o_f32, o_dst);
            }
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

} // namespace gdn
} // namespace lowoha
} // namespace zendnnl

GDN_AVX512_END
