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

#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_int8_tile_internal.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>

#include <immintrin.h>

#include "common/data_types.hpp"
#include "common/zendnnl_compat.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/int8/native_int4_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/int8/native_int8_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/common/avx512_math.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::common::size_of;

namespace {

/// Groups whose gathered weight scales one buffer holds.
constexpr int kScaleGroups = 32;

/// The weight scales of every group of a call whose weight scales apply
/// after the groups.
struct unit_scales_t {
    alignas(64) float v[kMaxTileNr];
    constexpr unit_scales_t() : v() {
        for (float &x : v)
            x = 1.0f;
    }
};
constexpr unit_scales_t kUnitScales;

/// C holds the sum of the groups of rows m0 .. m0 + mr and pack columns p0
/// .. p0 + nr (a multiple of 16) of w, scale the f32 weight scales of those
/// columns.  Applies alpha, beta and the scales one rounding at a time, in
/// the reference algorithm's order: (alpha * S + beta * C) * scale[j], or
/// with RowScale ((alpha * S + beta * C) * src_scale[m]) * scale[j], the
/// source scale of row m (call.src_scale).  Pairs: w is an interleaved
/// pack, and beta reads C at the columns its pack columns hold.  Tail: only
/// the first cols pack columns lie below the weight's N; beta reads no
/// column past it.
template <bool Pairs, bool Tail, bool RowScale>
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,avx512dq")
void finish_after_groups(const native_int8_call_t &call,
        const native_weight_view_t &w, int m0, int mr, int p0, int nr, int cols,
        const float *scale, float *C, int ldc) {
    const bool scale_alpha = call.alpha != 1.0f;
    const bool add_beta = call.beta != 0.0f;
    const __m512 alpha = _mm512_set1_ps(call.alpha);
    const __m512 beta = _mm512_set1_ps(call.beta);
    for (int r = 0; r < mr; ++r) {
        float *c = C + static_cast<size_t>(r) * ldc;
        for (int j = 0; j < nr; j += 16) {
            __m512 v = _mm512_loadu_ps(c + j);
            // No product may fuse with the add that follows it.
            if (scale_alpha) v = avx512_mul_unfused(v, alpha);
            if (add_beta) {
                if constexpr (Tail && Pairs)
                    v = _mm512_add_ps(v,
                            avx512_mul_unfused(
                                    dst_pair_values_below(
                                            call, w, m0 + r, p0 + j, cols - j),
                                    beta));
                else if constexpr (Tail)
                    v = _mm512_add_ps(v,
                            avx512_mul_unfused(dst_values_below(call, m0 + r,
                                                       p0 + j, cols - j),
                                    beta));
                else if constexpr (Pairs)
                    v = _mm512_add_ps(v,
                            avx512_mul_unfused(
                                    dst_pair_values(call, w, m0 + r, p0 + j),
                                    beta));
                else
                    v = _mm512_add_ps(v,
                            avx512_mul_unfused(
                                    dst_values(call, m0 + r, p0 + j), beta));
            }
            if constexpr (RowScale)
                v = _mm512_mul_ps(v,
                        _mm512_set1_ps(
                                call.src_scale[static_cast<size_t>(m0 + r)
                                        * call.src_scale_stride]));
            _mm512_storeu_ps(
                    c + j, _mm512_mul_ps(v, _mm512_loadu_ps(scale + j)));
        }
    }
}

/// The 16 values of S (f32, bf16 or f16) of v at gate, up, gate + 1, up + 1,
/// .., gate + 7, up + 7, as f32; exact.
template <data_type_t S>
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl")
inline __m512 interleaved_as_f32(const void *v, size_t gate, size_t up) {
    if constexpr (S == data_type_t::f32) {
        const float *f = static_cast<const float *>(v);
        const __m512i order = _mm512_set_epi32(
                23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0);
        return _mm512_permutex2var_ps(
                _mm512_castps256_ps512(_mm256_loadu_ps(f + gate)), order,
                _mm512_castps256_ps512(_mm256_loadu_ps(f + up)));
    } else {
        const uint16_t *h = static_cast<const uint16_t *>(v);
        const __m128i g
                = _mm_loadu_si128(reinterpret_cast<const __m128i *>(h + gate));
        const __m128i u
                = _mm_loadu_si128(reinterpret_cast<const __m128i *>(h + up));
        const __m256i x = _mm256_inserti128_si256(
                _mm256_castsi128_si256(_mm_unpacklo_epi16(g, u)),
                _mm_unpackhi_epi16(g, u), 1);
        if constexpr (S == data_type_t::bf16)
            return avx512_bf16_to_f32(x);
        else
            return _mm512_cvtph_ps(x);
    }
}

/// gather_group_wei_scales of scales of type S, N weight columns per group.
template <data_type_t S>
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl")
void gather_interleaved(
        const void *scales, int N, int g0, int n, int p0, int cw, float *buf) {
    const size_t half = static_cast<size_t>(N / 2);
    for (int i = 0; i < n; ++i) {
        const size_t gate = static_cast<size_t>(g0 + i) * N + p0 / 2;
        float *out = buf + static_cast<size_t>(i) * cw;
        for (int j = 0; j < cw; j += 16)
            _mm512_storeu_ps(out + j,
                    interleaved_as_f32<S>(
                            scales, gate + j / 2, gate + half + j / 2));
    }
}

} // namespace

void gather_group_wei_scales(const native_int8_call_t &call,
        const native_weight_view_t &w, int g0, int n, int p0, int cw,
        float *buf) {
    assert(w.interleaved && p0 % 16 == 0 && cw % 16 == 0 && p0 + cw <= w.N);
    if (call.wei_scale_dt == data_type_t::f32)
        gather_interleaved<data_type_t::f32>(
                call.wei_scale, w.N, g0, n, p0, cw, buf);
    else if (call.wei_scale_dt == data_type_t::bf16)
        gather_interleaved<data_type_t::bf16>(
                call.wei_scale, w.N, g0, n, p0, cw, buf);
    else {
        assert(call.wei_scale_dt == data_type_t::f16);
        gather_interleaved<data_type_t::f16>(
                call.wei_scale, w.N, g0, n, p0, cw, buf);
    }
}

void gather_group_wei_scales_below(const native_int8_call_t &call,
        const native_weight_view_t &w, int g0, int n, int p0, int cw, int cols,
        float *buf) {
    assert(cols >= 0 && cols <= cw);
    for (int i = 0; i < n; ++i) {
        const size_t row = static_cast<size_t>(g0 + i) * w.N;
        float *out = buf + static_cast<size_t>(i) * cw;
        for (int j = 0; j < cw; ++j)
            out[j] = j < cols
                    ? zendnnl::memory::read_and_cast<float>(call.wei_scale,
                              call.wei_scale_dt, row + w.logical_col(p0 + j))
                    : 0.0f;
    }
}

int native_int8_group_span(const native_plan_t &plan, int groups,
        int group_quads, int cw, int rows, int max_mr, int quad_bytes) {
    // All groups, unless the tile takes more than one row call and the B
    // slab plus one row call's A rows exceed twice the budget (the L2 at the
    // built-in share), where the row calls no longer find the slab in L2;
    // then equal batches that fit the budget.  A slab that fits L2 is not
    // split: each batch would add a round trip of C to every row call.
    const int64_t group_bytes
            = int64_t {group_quads} * (int64_t {quad_bytes} * cw + 4 * max_mr);
    const int64_t budget = plan.int8_pass_budget;
    if (rows <= max_mr || groups * group_bytes <= 2 * budget) return groups;
    const int64_t per = std::max<int64_t>(1, budget / group_bytes);
    const int64_t batches = (groups + per - 1) / per;
    return static_cast<int>((groups + batches - 1) / batches);
}

template <bool Tail>
void compute_group_tile(const native_plan_t &plan,
        const native_int8_call_t &call, const native_weight_view_t &w, int m0,
        int mr, int n0, int nr, float *C, int ldc,
        const native_int8_out_t *out) {
    // An s8 source has the shifts of its rows; a u8 source its zero point.
    const bool src_u8 = !call.src_signed;
    if (!src_u8) ready_group_shifts(call, m0, mr);
    const int nb = n0 / w.pack_nr;
    const int j0 = n0 - nb * w.pack_nr;
    const int8_t *block = reinterpret_cast<const int8_t *>(w.block(nb));
    // The scales of a plain pack are read in place, in their own type, in
    // batches of native_int8_group_span groups; an interleaved pack's are
    // gathered as f32 kScaleGroups groups at a time.  C carries the sum
    // between batches.  Weight scales that apply after the groups leave the
    // groups unit scales, in batches as over a plain pack.
    const bool unit_scales = call.scales_after_groups;
    assert(out == nullptr || !unit_scales);
    const bool in_place = !w.interleaved || unit_scales;
    const size_t out_esz = out != nullptr && out->bf16 ? 2 : 4;
    alignas(64) float scale_buf[kScaleGroups * kMaxTileNr];

    native_int8_group_ukernel_args_t args;
    args.lda = call.lda;
    args.a_step = static_cast<size_t>(call.group_quads) * 4;
    args.b_stride = w.group_stride();
    args.b_step = static_cast<size_t>(call.group_quads) * args.b_stride;
    args.k_quads = call.group_quads;
    args.shift_ld = call.groups;
    args.shift_step = 1;
    args.src_zp = src_u8 && call.src_zp != nullptr ? call.src_zp[0] : 0;
    args.src_scale_ld = call.src_scale_stride;
    args.src_scale_step = call.src_scale_group_stride;
    args.ldc = ldc;
    if (out != nullptr) {
        args.dst_ldc = out->ldc;
        args.dst_bf16 = out->bf16;
    }
    for (int c = 0; c < nr;) {
        const int cw = native_int8_col_width(plan, nr - c);
        assert(cw <= kMaxTileNr);
        const int nv = cw / 16;
        const int max_mr = native_int8_call_rows(plan, nv);
        assert(max_mr >= 1 && max_mr <= native_int8_group_ukernel_max_mr(nv));
        // Tail: the columns of this call below the weight's N; the weight
        // scales of a call of fewer are gathered, kScaleGroups groups at a
        // time.
        const int cols = Tail ? std::min(cw, w.N - (n0 + c)) : cw;
        const bool part = Tail && cols < cw;
        const bool gather_part = part && !unit_scales;
        const int span = in_place && !gather_part
                ? native_int8_group_span(
                          plan, call.groups, call.group_quads, cw, mr, max_mr)
                : kScaleGroups;
        for (int g0 = 0; g0 < call.groups; g0 += span) {
            args.batch = std::min(span, call.groups - g0);
            args.accumulate = g0 > 0;
            const bool last = g0 + args.batch == call.groups;
            args.B = block + static_cast<size_t>(j0 + c) * 4 + g0 * args.b_step;
            if (unit_scales) {
                args.wei_scale = kUnitScales.v;
                args.wei_scale_step = 0;
                args.wei_scale_dt = data_type_t::f32;
            } else if (gather_part) {
                gather_group_wei_scales_below(
                        call, w, g0, args.batch, n0 + c, cw, cols, scale_buf);
                args.wei_scale = scale_buf;
                args.wei_scale_step = cw;
                args.wei_scale_dt = data_type_t::f32;
            } else if (in_place) {
                args.wei_scale = static_cast<const char *>(call.wei_scale)
                        + (static_cast<size_t>(g0) * w.N + n0 + c)
                                * size_of(call.wei_scale_dt);
                args.wei_scale_step = w.N;
                args.wei_scale_dt = call.wei_scale_dt;
            } else {
                gather_group_wei_scales(
                        call, w, g0, args.batch, n0 + c, cw, scale_buf);
                args.wei_scale = scale_buf;
                args.wei_scale_step = cw;
                args.wei_scale_dt = data_type_t::f32;
            }
            // As few calls as max_mr allows, their rows within one of each
            // other.
            const int calls = (mr + max_mr - 1) / max_mr;
            for (int k = 0, i = 0; k < calls; ++k) {
                const int rows = mr / calls + (k < mr % calls ? 1 : 0);
                const size_t row = static_cast<size_t>(m0 + i);
                args.A = call.A + row * call.lda + g0 * args.a_step;
                args.shift = src_u8 ? nullptr
                                    : call.group_shift + row * call.groups + g0;
                args.src_scale = call.src_scale + call.src_scale_stride * row
                        + static_cast<size_t>(call.src_scale_group_stride) * g0;
                args.C = C + static_cast<size_t>(i) * ldc + c;
                args.dst = last && out != nullptr
                        ? static_cast<char *>(out->dst)
                                + (row * out->ldc + n0 + c) * out_esz
                        : nullptr;
                select_native_int8_group_ukernel(
                        rows, nv, src_u8, call.wide_zp)(args);
                i += rows;
            }
        }
        if (unit_scales) {
            const float *scale = part
                    ? column_values_below(call.wei_scale, call.wei_scale_dt,
                              call.wei_scale_stride, w, n0 + c, cw, cols,
                              scale_buf)
                    : column_values(call.wei_scale, call.wei_scale_dt,
                              call.wei_scale_stride, w, n0 + c, cw, scale_buf);
            if (w.interleaved)
                finish_after_groups<true, Tail, false>(
                        call, w, m0, mr, n0 + c, cw, cols, scale, C + c, ldc);
            else
                finish_after_groups<false, Tail, false>(
                        call, w, m0, mr, n0 + c, cw, cols, scale, C + c, ldc);
        }
        c += cw;
    }
}

template void compute_group_tile<false>(const native_plan_t &,
        const native_int8_call_t &, const native_weight_view_t &, int, int, int,
        int, float *, int, const native_int8_out_t *);
template void compute_group_tile<true>(const native_plan_t &,
        const native_int8_call_t &, const native_weight_view_t &, int, int, int,
        int, float *, int, const native_int8_out_t *);

template <bool Tail>
void compute_int4_group_tile(const native_plan_t &plan,
        const native_int8_call_t &call, const native_weight_view_t &w, int m0,
        int mr, int n0, int nr, float *C, int ldc,
        const native_int8_out_t *out) {
    assert(call.src_signed);
    ready_group_shifts(call, m0, mr);
    const int nb = n0 / w.pack_nr;
    const int j0 = n0 - nb * w.pack_nr;
    const uint8_t *block = w.block(nb);
    // The scales of a plain pack are read in place, in their own type, in
    // batches of native_int8_group_span groups; an interleaved pack's are
    // gathered as f32 kScaleGroups groups at a time.  C carries the sum
    // between batches; the fused order takes a source scale per row after
    // the last, and source scales per group with each group.  Scales that
    // apply after the groups leave the groups unit scales, in batches as
    // over a plain pack.
    const bool unit_scales = call.scales_after_groups;
    assert(out == nullptr || !unit_scales);
    const bool in_place = !w.interleaved || unit_scales;
    const bool fused = call.order == native_int8_order_t::fused;
    const bool src_groups = call.src_scale_group_stride != 0;
    const size_t out_esz = out != nullptr && out->bf16 ? 2 : 4;
    alignas(64) float scale_buf[kScaleGroups * kMaxTileNr];

    native_int4_group_ukernel_args_t args;
    args.lda = call.lda;
    args.a_step = static_cast<size_t>(call.group_quads) * 4;
    args.b_stride = w.group_stride();
    args.k_quads = call.group_quads;
    args.shift_ld = static_cast<size_t>(call.groups);
    args.shift_step = 1;
    args.src_scale_ld
            = unit_scales ? 0 : static_cast<size_t>(call.src_scale_stride);
    args.src_scale_step = unit_scales ? 0 : call.src_scale_group_stride;
    args.ldc = ldc;
    if (out != nullptr) {
        args.dst_ldc = out->ldc;
        args.dst_bf16 = out->bf16;
    }
    for (int c = 0; c < nr;) {
        const int cw = native_int8_col_width(plan, nr - c);
        assert(cw <= kMaxTileNr);
        const int nv = cw / 16;
        const int max_mr = native_int8_call_rows(plan, nv);
        assert(max_mr >= 1 && max_mr <= native_int4_group_ukernel_max_mr(nv));
        // Tail: the columns of this call below the weight's N; the weight
        // scales of a call of fewer are gathered, kScaleGroups groups at a
        // time.
        const int cols = Tail ? std::min(cw, w.N - (n0 + c)) : cw;
        const bool part = Tail && cols < cw;
        const bool gather_part = part && !unit_scales;
        // A K quad of a column of an s4 weight is 2 bytes.
        const int span = in_place && !gather_part
                ? native_int8_group_span(plan, call.groups, call.group_quads,
                          cw, mr, max_mr, 2)
                : kScaleGroups;
        for (int g0 = 0; g0 < call.groups; g0 += span) {
            args.batch = std::min(span, call.groups - g0);
            args.accumulate = g0 > 0;
            const bool last = g0 + args.batch == call.groups;
            args.finish = fused && last && !src_groups;
            // The batch's first K quad: the low or the high nibbles of its
            // octet.
            const int64_t quad = static_cast<int64_t>(g0) * call.group_quads;
            args.B = block + static_cast<size_t>(j0 + c) * 4
                    + static_cast<size_t>(quad >> 1) * args.b_stride;
            args.quad0 = static_cast<int>(quad & 1);
            if (unit_scales) {
                args.wei_scale = kUnitScales.v;
                args.wei_scale_step = 0;
                args.wei_scale_dt = data_type_t::f32;
            } else if (gather_part) {
                gather_group_wei_scales_below(
                        call, w, g0, args.batch, n0 + c, cw, cols, scale_buf);
                args.wei_scale = scale_buf;
                args.wei_scale_step = cw;
                args.wei_scale_dt = data_type_t::f32;
            } else if (in_place) {
                args.wei_scale = static_cast<const char *>(call.wei_scale)
                        + (static_cast<size_t>(g0) * w.N + n0 + c)
                                * size_of(call.wei_scale_dt);
                args.wei_scale_step = w.N;
                args.wei_scale_dt = call.wei_scale_dt;
            } else {
                gather_group_wei_scales(
                        call, w, g0, args.batch, n0 + c, cw, scale_buf);
                args.wei_scale = scale_buf;
                args.wei_scale_step = cw;
                args.wei_scale_dt = data_type_t::f32;
            }
            // As few calls as max_mr allows, their rows within one of each
            // other.
            const int calls = (mr + max_mr - 1) / max_mr;
            for (int k = 0, i = 0; k < calls; ++k) {
                const int rows = mr / calls + (k < mr % calls ? 1 : 0);
                const size_t row = static_cast<size_t>(m0 + i);
                args.A = call.A + row * call.lda + g0 * args.a_step;
                args.shift = call.group_shift + row * call.groups + g0;
                args.src_scale = unit_scales
                        ? kUnitScales.v
                        : call.src_scale + call.src_scale_stride * row
                                + static_cast<size_t>(
                                          call.src_scale_group_stride)
                                        * g0;
                args.C = C + static_cast<size_t>(i) * ldc + c;
                args.dst = last && out != nullptr
                        ? static_cast<char *>(out->dst)
                                + (row * out->ldc + n0 + c) * out_esz
                        : nullptr;
                select_native_int4_group_ukernel(
                        rows, nv, call.order, src_groups)(args);
                i += rows;
            }
        }
        if (unit_scales) {
            const float *scale = part
                    ? column_values_below(call.wei_scale, call.wei_scale_dt,
                              call.wei_scale_stride, w, n0 + c, cw, cols,
                              scale_buf)
                    : column_values(call.wei_scale, call.wei_scale_dt,
                              call.wei_scale_stride, w, n0 + c, cw, scale_buf);
            if (w.interleaved)
                finish_after_groups<true, Tail, true>(
                        call, w, m0, mr, n0 + c, cw, cols, scale, C + c, ldc);
            else
                finish_after_groups<false, Tail, true>(
                        call, w, m0, mr, n0 + c, cw, cols, scale, C + c, ldc);
        }
        c += cw;
    }
}

template void compute_int4_group_tile<false>(const native_plan_t &,
        const native_int8_call_t &, const native_weight_view_t &, int, int, int,
        int, float *, int, const native_int8_out_t *);
template void compute_int4_group_tile<true>(const native_plan_t &,
        const native_int8_call_t &, const native_weight_view_t &, int, int, int,
        int, float *, int, const native_int8_out_t *);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
