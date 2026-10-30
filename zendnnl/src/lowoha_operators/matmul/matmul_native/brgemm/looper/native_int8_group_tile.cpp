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

#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/int8/native_int8_ukernel.hpp"
#include "memory/memory_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::common::size_of;
using zendnnl::memory::read_and_cast;

namespace {

/// Groups whose gathered weight scales one buffer holds.
constexpr int kScaleGroups = 32;

/// The weight scales of groups g0 .. g0 + n at pack columns p0 .. p0 + cw of
/// an interleaved pack, as f32, cw apart, into buf.
void gather_group_wei_scales(const native_int8_call_t &call,
        const native_weight_view_t &w, int g0, int n, int p0, int cw,
        float *buf) {
    for (int i = 0; i < n; ++i) {
        const size_t row = static_cast<size_t>(g0 + i) * w.N;
        float *out = buf + static_cast<size_t>(i) * cw;
        for (int j = 0; j < cw; ++j)
            out[j] = read_and_cast<float>(call.wei_scale, call.wei_scale_dt,
                    row + w.logical_col(p0 + j));
    }
}

} // namespace

void compute_group_tile(const native_plan_t &plan,
        const native_int8_call_t &call, const native_weight_view_t &w, int m0,
        int mr, int n0, int nr, float *C, int ldc,
        const native_int8_out_t *out) {
    ready_group_shifts(call, m0, mr);
    const int nb = n0 / w.pack_nr;
    const int j0 = n0 - nb * w.pack_nr;
    const int8_t *block = reinterpret_cast<const int8_t *>(w.block(nb));
    // The scales of a plain pack are read in place, in their own type, in
    // one batch of all groups; an interleaved pack's are gathered as f32
    // kScaleGroups groups at a time, and C carries the sum between those
    // batches.
    const bool in_place = !w.interleaved;
    const int span = in_place ? call.groups : kScaleGroups;
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
        for (int g0 = 0; g0 < call.groups; g0 += span) {
            args.batch = std::min(span, call.groups - g0);
            args.accumulate = g0 > 0;
            const bool last = g0 + args.batch == call.groups;
            args.B = block + static_cast<size_t>(j0 + c) * 4 + g0 * args.b_step;
            if (in_place) {
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
                args.src_scale = call.src_scale + call.src_scale_stride * row
                        + static_cast<size_t>(call.src_scale_group_stride) * g0;
                args.C = C + static_cast<size_t>(i) * ldc + c;
                args.dst = last && out != nullptr
                        ? static_cast<char *>(out->dst)
                                + (row * out->ldc + n0 + c) * out_esz
                        : nullptr;
                select_native_int8_group_ukernel(rows, nv)(args);
                i += rows;
            }
        }
        c += cw;
    }
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
