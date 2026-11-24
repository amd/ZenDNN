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
#include <limits>

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

/// C holds the sum of the groups of rows m0 .. m0 + mr and weight columns
/// p0 .. p0 + nr (a multiple of 16), scale the f32 weight scales of those
/// columns.  Applies alpha, the scales and beta one rounding at a time, in
/// the reference algorithm's order: (alpha * S) * scale[j] + beta * C into
/// an f32, bf16 or f16 destination, (alpha * S + beta * C) * scale[j] into
/// an integer one; with RowScale the source scale of row m (call.src_scale) applies
/// before scale[j]; beta reads C at those columns.  Tail: only the first
/// cols columns lie below the weight's N; beta reads no column past it.
template <bool Tail, bool RowScale>
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,avx512dq")
void finish_after_groups(const native_int8_call_t &call, int m0, int mr, int p0,
        int nr, int cols, const float *scale, float *C, int ldc) {
    const bool scale_alpha = call.alpha != 1.0f;
    const bool beta_last
            = call.beta != 0.0f && native_int8_beta_after_scales(call.dst_dt);
    const bool beta_first = call.beta != 0.0f && !beta_last;
    const __m512 alpha = _mm512_set1_ps(call.alpha);
    const __m512 beta = _mm512_set1_ps(call.beta);
    for (int r = 0; r < mr; ++r) {
        float *c = C + static_cast<size_t>(r) * ldc;
        for (int j = 0; j < nr; j += 16) {
            __m512 v = _mm512_loadu_ps(c + j);
            // No product may fuse with the add that follows it.
            if (scale_alpha) v = avx512_mul_unfused(v, alpha);
            if (beta_first)
                v = _mm512_add_ps(v,
                        native_int8_beta_term<Tail>(
                                call, m0 + r, p0 + j, cols - j, beta));
            if constexpr (RowScale)
                v = _mm512_mul_ps(v,
                        _mm512_set1_ps(
                                call.src_scale[static_cast<size_t>(m0 + r)
                                        * call.src_scale_stride]));
            if (beta_last)
                v = _mm512_add_ps(
                        avx512_mul_unfused(v, _mm512_loadu_ps(scale + j)),
                        native_int8_beta_term<Tail>(
                                call, m0 + r, p0 + j, cols - j, beta));
            else
                v = _mm512_mul_ps(v, _mm512_loadu_ps(scale + j));
            _mm512_storeu_ps(c + j, v);
        }
    }
}

/// finish_after_groups of the call of cw columns at weight columns p0 (its
/// first 16) and p1 (its second 16), with the call's weight scales read
/// into scale_buf.
template <bool Tail, bool RowScale>
void finish_call_after_groups(const native_int8_call_t &call,
        const native_weight_view_t &w, int m0, int mr, int p0, int p1, int cw,
        int cols, float *scale_buf, float *C, int ldc) {
    if (second_apart(w, cw / 16, p0, p1)) {
        const float *scale = column_values_apart(call.wei_scale,
                call.wei_scale_type, call.wei_scale_stride, p0, p1, scale_buf);
        finish_after_groups<Tail, RowScale>(
                call, m0, mr, p0, 16, 16, scale, C, ldc);
        finish_after_groups<Tail, RowScale>(
                call, m0, mr, p1, 16, 16, scale + 16, C + 16, ldc);
        return;
    }
    const float *scale = Tail && cols < cw
            ? column_values_below(call.wei_scale, call.wei_scale_type,
                      call.wei_scale_stride, p0, cw, cols, scale_buf)
            : column_values(call.wei_scale, call.wei_scale_type,
                      call.wei_scale_stride, p0, cw, scale_buf);
    finish_after_groups<Tail, RowScale>(
            call, m0, mr, p0, cw, cols, scale, C, ldc);
}

} // namespace

void gather_group_wei_scales_below(const native_int8_call_t &call,
        const native_weight_view_t &w, int g0, int n, int p0, int cw, int cols,
        float *buf) {
    assert(cols >= 0 && cols <= cw);
    for (int i = 0; i < n; ++i) {
        const size_t row = static_cast<size_t>(g0 + i) * w.N;
        float *out = buf + static_cast<size_t>(i) * cw;
        for (int j = 0; j < cw; ++j)
            out[j] = j < cols ? native_scale_value(call.wei_scale,
                                        call.wei_scale_type, row + p0 + j)
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
    // The scales are read in place, in their own type, in batches of
    // native_int8_group_span groups; C carries the sum between batches.
    // Weight scales that apply after the groups leave the groups unit
    // scales.
    const bool unit_scales = call.scales_after_groups;
    assert(out == nullptr || !unit_scales);
    const size_t out_esz = out != nullptr && out->bf16 ? 2 : 4;
    alignas(64) float scale_buf[kScaleGroups * kMaxTileNr];
    // The weight columns of each of the calls that take a row call in turn,
    // their K group 0 and their weight scales of group 0.
    struct call_t {
        int p0, p1;
        size_t b_second;
        const int8_t *b0;
        const char *scale0;
    } turn[kMaxTileCalls];
    const size_t scale_bytes = native_scale_bytes(call.wei_scale_type);
    const native_format_t format = native_route(call.route).format;

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
    const int in_turn = native_int8_calls_in_turn(
            plan, nr, native_int8_col_width(plan, nr), Tail);
    for (int c = 0; c < nr;) {
        const int cw = native_int8_col_width(plan, nr - c);
        assert(cw <= kMaxTileNr);
        const int nv = cw / 16;
        const int max_mr = native_int8_call_rows(plan, nv);
        assert(max_mr >= 1 && max_mr <= native_int8_group_ukernel_max_mr(nv));
        // The weight columns of each call's first 16 columns, and of its
        // second 16 (a paired tile's up run, or the next block).
        for (int j = 0; j < in_turn; ++j) {
            call_t &k = turn[j];
            k.p0 = w.tile_col(n0, nr, c + j * cw);
            k.p1 = second_col(w, n0, nr, c + j * cw, nv, k.p0);
            k.b_second = second_apart(w, nv, k.p0, k.p1)
                    ? w.column_distance(k.p0, k.p1)
                    : 64;
            k.b0 = reinterpret_cast<const int8_t *>(w.column(k.p0));
            k.scale0 = static_cast<const char *>(call.wei_scale)
                    + static_cast<size_t>(k.p0) * scale_bytes;
        }
        const int p0 = turn[0].p0;
        // Tail: the columns of this call below the weight's N; the weight
        // scales of a call of fewer are gathered, kScaleGroups groups at a
        // time.
        const int cols = Tail ? std::min(cw, w.N - (n0 + c)) : cw;
        const bool part = Tail && cols < cw;
        assert(!(part && second_apart(w, nv, p0, turn[0].p1)));
        const bool gather_part = part && !unit_scales;
        const int span = !gather_part
                ? native_int8_group_span(plan, call.groups, call.group_quads,
                          native_int8_turn_width(plan, nr, cw, Tail), mr,
                          max_mr, native_quad_bytes(native_kernel_kind_t::int8))
                : kScaleGroups;
        for (int g0 = 0; g0 < call.groups; g0 += span) {
            args.batch = std::min(span, call.groups - g0);
            args.accumulate = g0 > 0;
            const bool last = g0 + args.batch == call.groups;
            args.scale_second = 16;
            if (unit_scales) {
                args.wei_scale = kUnitScales.v;
                args.wei_scale_step = 0;
                args.wei_scale_type = native_scale_t::f32;
            } else if (gather_part) {
                gather_group_wei_scales_below(
                        call, w, g0, args.batch, p0, cw, cols, scale_buf);
                args.wei_scale = scale_buf;
                args.wei_scale_step = cw;
                args.wei_scale_type = native_scale_t::f32;
            } else {
                args.wei_scale_step = w.N;
                args.wei_scale_type = call.wei_scale_type;
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
                for (int j = 0; j < in_turn; ++j) {
                    const call_t &t = turn[j];
                    const int cj = c + j * cw;
                    args.b_second = t.b_second;
                    args.B = t.b0 + g0 * args.b_step;
                    if (!unit_scales && !gather_part) {
                        args.wei_scale = t.scale0
                                + static_cast<size_t>(g0) * w.N * scale_bytes;
                        args.scale_second = t.p1 - t.p0;
                    }
                    args.C = C + static_cast<size_t>(i) * ldc + cj;
                    args.dst = last && out != nullptr
                            ? static_cast<char *>(out->dst)
                                    + (row * out->ldc + n0 + cj) * out_esz
                            : nullptr;
                    select_native_int8_group_ukernel(
                            format, rows, nv, src_u8, call.wide_zp)(args);
                }
                i += rows;
            }
        }
        if (unit_scales)
            for (int j = 0; j < in_turn; ++j)
                finish_call_after_groups<Tail, false>(call, w, m0, mr,
                        turn[j].p0, turn[j].p1, cw, cols, scale_buf,
                        C + c + j * cw, ldc);
        c += in_turn * cw;
    }
}

void finish_int8_tile_after_groups(const native_plan_t &plan,
        const native_int8_call_t &call, const native_weight_view_t &w, int m0,
        int mr, int n0, int nr, float *C, int ldc) {
    assert(call.scales_after_groups);
    alignas(64) float scale_buf[kMaxTileNr];
    for (int c = 0; c < nr;) {
        const int cw = native_int8_col_width(plan, nr - c);
        const int p0 = w.tile_col(n0, nr, c);
        const int p1 = second_col(w, n0, nr, c, cw / 16, p0);
        if (native_on_family<native_kernel_kind_t::int4>(call.route))
            finish_call_after_groups<false, true>(
                    call, w, m0, mr, p0, p1, cw, cw, scale_buf, C + c, ldc);
        else
            finish_call_after_groups<false, false>(
                    call, w, m0, mr, p0, p1, cw, cw, scale_buf, C + c, ldc);
        c += cw;
    }
}

template void compute_group_tile<false>(const native_plan_t &,
        const native_int8_call_t &, const native_weight_view_t &, int, int, int,
        int, float *, int, const native_int8_out_t *);
template void compute_group_tile<true>(const native_plan_t &,
        const native_int8_call_t &, const native_weight_view_t &, int, int, int,
        int, float *, int, const native_int8_out_t *);

int64_t native_int4_request_octets(
        const native_weight_view_t &w, int p0, int nv, size_t b_second) {
    const size_t room = w.bytes() - static_cast<size_t>(w.column(p0) - w.base)
            - (nv == 2 ? b_second : 0);
    const size_t stride = static_cast<size_t>(w.group_stride());
    return static_cast<int64_t>((room + stride - 1) / stride);
}

template <bool Tail>
void compute_int4_group_tile(const native_plan_t &plan,
        const native_int8_call_t &call, const native_weight_view_t &w, int m0,
        int mr, int n0, int nr, float *C, int ldc,
        const native_int8_out_t *out) {
    assert(call.src_signed);
    ready_group_shifts(call, m0, mr);
    // The scales are read in place, in their own type, in batches of
    // native_int8_group_span groups; C carries the sum between batches; the
    // fused order takes a source scale per row after the last, and source
    // scales per group with each group.  Scales that apply after the groups
    // leave the groups unit scales.  The one group of a weight of one group
    // over K takes K passes of native_int4_pass_quads quads, C carrying
    // their exact int32 sum, on the kernels of one group.
    const bool unit_scales = call.scales_after_groups;
    assert(out == nullptr || !unit_scales);
    const bool fused = call.order == native_int8_order_t::fused;
    const bool src_groups = call.src_scale_group_stride != 0;
    const bool one_group = call.groups == 1;
    assert(!(one_group && src_groups));
    const size_t out_esz = out != nullptr && out->bf16 ? 2 : 4;
    alignas(64) float scale_buf[kScaleGroups * kMaxTileNr];
    // The weight columns of each of the calls that take a row call in turn,
    // their K octet 0, their weight scales of group 0 and the octets of one
    // group their requests may reach.
    struct call_t {
        int p0, p1;
        size_t b_second;
        const uint8_t *b0;
        const char *scale0;
        int64_t reach;
    } turn[kMaxTileCalls];
    const size_t scale_bytes = native_scale_bytes(call.wei_scale_type);
    const native_format_t format = native_route(call.route).format;

    native_int4_group_ukernel_args_t args;
    args.lda = call.lda;
    args.a_step = static_cast<size_t>(call.group_quads) * 4;
    args.b_stride = w.group_stride();
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
    const int in_turn = native_int8_calls_in_turn(
            plan, nr, native_int8_col_width(plan, nr), Tail);
    for (int c = 0; c < nr;) {
        const int cw = native_int8_col_width(plan, nr - c);
        assert(cw <= kMaxTileNr);
        const int nv = cw / 16;
        const int max_mr = native_int8_call_rows(plan, nv);
        assert(max_mr >= 1
                && max_mr <= (one_group
                                   ? native_int4_one_group_ukernel_max_mr(nv)
                                   : native_int4_group_ukernel_max_mr(nv)));
        // The weight columns of each call's first 16 columns, and of its
        // second 16 (a paired tile's up run, or the next block).
        for (int j = 0; j < in_turn; ++j) {
            call_t &k = turn[j];
            k.p0 = w.tile_col(n0, nr, c + j * cw);
            k.p1 = second_col(w, n0, nr, c + j * cw, nv, k.p0);
            k.b_second = second_apart(w, nv, k.p0, k.p1)
                    ? w.column_distance(k.p0, k.p1)
                    : 64;
            k.b0 = w.column(k.p0);
            k.scale0 = static_cast<const char *>(call.wei_scale)
                    + static_cast<size_t>(k.p0) * scale_bytes;
            k.reach = one_group
                    ? native_int4_request_octets(w, k.p0, nv, k.b_second)
                    : 0;
        }
        const int p0 = turn[0].p0;
        // Tail: the columns of this call below the weight's N; the weight
        // scales of a call of fewer are gathered, kScaleGroups groups at a
        // time.
        const int cols = Tail ? std::min(cw, w.N - (n0 + c)) : cw;
        const bool part = Tail && cols < cw;
        assert(!(part && second_apart(w, nv, p0, turn[0].p1)));
        const bool gather_part = part && !unit_scales;
        const int span = !gather_part
                ? native_int8_group_span(plan, call.groups, call.group_quads,
                          native_int8_turn_width(plan, nr, cw, Tail), mr,
                          max_mr, native_quad_bytes(native_kernel_kind_t::int4))
                : kScaleGroups;
        const int pass = one_group
                ? native_int4_pass_quads(plan, call.group_quads,
                          native_int8_turn_width(plan, nr, cw, Tail), mr,
                          max_mr)
                : call.group_quads;
        for (int g0 = 0; g0 < call.groups; g0 += span) {
            args.batch = std::min(span, call.groups - g0);
            args.accumulate = g0 > 0;
            const bool last = g0 + args.batch == call.groups;
            args.scale_second = 16;
            if (unit_scales) {
                args.wei_scale = kUnitScales.v;
                args.wei_scale_step = 0;
                args.wei_scale_type = native_scale_t::f32;
            } else if (gather_part) {
                gather_group_wei_scales_below(
                        call, w, g0, args.batch, p0, cw, cols, scale_buf);
                args.wei_scale = scale_buf;
                args.wei_scale_step = cw;
                args.wei_scale_type = native_scale_t::f32;
            } else {
                args.wei_scale_step = w.N;
                args.wei_scale_type = call.wei_scale_type;
            }
            for (int k0 = 0; k0 < call.group_quads; k0 += pass) {
                const bool whole = k0 + pass >= call.group_quads;
                args.k_quads = whole ? call.group_quads - k0 : pass;
                args.partial_in = k0 > 0;
                args.partial_out = !whole;
                args.finish = fused && last && whole && !src_groups;
                // The pass's first K quad: the low or the high nibbles of its
                // octet.
                const int64_t quad
                        = static_cast<int64_t>(g0) * call.group_quads + k0;
                args.quad0 = static_cast<int>(quad & 1);
                // As few calls as max_mr allows, their rows within one of
                // each other.  The first reads the pass's slab from memory,
                // the later ones from cache.
                const int calls = (mr + max_mr - 1) / max_mr;
                for (int k = 0, i = 0; k < calls; ++k) {
                    const int rows = mr / calls + (k < mr % calls ? 1 : 0);
                    const size_t row = static_cast<size_t>(m0 + i);
                    args.A = call.A + row * call.lda + g0 * args.a_step
                            + static_cast<size_t>(k0) * 4;
                    args.shift = call.group_shift + row * call.groups + g0;
                    args.src_scale = unit_scales
                            ? kUnitScales.v
                            : call.src_scale + call.src_scale_stride * row
                                    + static_cast<size_t>(
                                              call.src_scale_group_stride)
                                            * g0;
                    for (int j = 0; j < in_turn; ++j) {
                        const call_t &t = turn[j];
                        const int cj = c + j * cw;
                        args.b_second = t.b_second;
                        args.B = t.b0
                                + static_cast<size_t>(quad >> 1)
                                        * args.b_stride;
                        if (!unit_scales && !gather_part) {
                            args.wei_scale = t.scale0
                                    + static_cast<size_t>(g0) * w.N
                                            * scale_bytes;
                            args.scale_second = t.p1 - t.p0;
                        }
                        args.request_limit_octets = one_group && k == 0
                                ? static_cast<int>(std::min<int64_t>(
                                          t.reach - (quad >> 1),
                                          std::numeric_limits<int>::max()))
                                : 0;
                        args.C = C + static_cast<size_t>(i) * ldc + cj;
                        args.dst = last && whole && out != nullptr
                                ? static_cast<char *>(out->dst)
                                        + (row * out->ldc + n0 + cj) * out_esz
                                : nullptr;
                        (one_group ? select_native_int4_one_group_ukernel(
                                             format, rows, nv, call.order)
                                   : select_native_int4_group_ukernel(format,
                                             rows, nv, call.order, src_groups))(
                                args);
                    }
                    i += rows;
                }
            }
        }
        if (unit_scales)
            for (int j = 0; j < in_turn; ++j)
                finish_call_after_groups<Tail, true>(call, w, m0, mr,
                        turn[j].p0, turn[j].p1, cw, cols, scale_buf,
                        C + c + j * cw, ldc);
        c += in_turn * cw;
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
