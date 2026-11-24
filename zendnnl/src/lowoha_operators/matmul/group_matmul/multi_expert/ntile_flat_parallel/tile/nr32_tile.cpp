/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/tile/nr32_tile.hpp"

#include <algorithm>
#include <limits>

#include "lowoha_operators/matmul/group_matmul/common/grp_matmul_machine.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {
namespace nr32 {

namespace {

/// Columns, a multiple of `kNr`, whose weights fit `b`'s L2 budget beside
/// its subtile rows of `K` elements.
int l2_subtile_cols(const int K, const int src_bytes, const int wei_bytes,
        const int extra_bytes_per_col, const plan_budget &b) {
    const int64_t input = static_cast<int64_t>(b.subtile_rows) * K * src_bytes;
    const int64_t per_col
            = static_cast<int64_t>(K) * wei_bytes + extra_bytes_per_col;
    if (b.l2_bytes <= input) return kNr;
    const int64_t cols = (b.l2_bytes - input) / per_col;
    return std::max(static_cast<int>(cols / kNr * kNr), kNr);
}

} // namespace

plan_budget default_budget() {
    return {grp_machine::machine().l2_budget_bytes, 16};
}

bool make_plan(const pack_layout &layout, const epilogue ep, const bool src_u8,
        tile_plan &plan, const bool src_groups, const plan_budget &budget) {
    const weight_desc &d = layout.desc;
    const format_desc &f = layout.format();
    plan = tile_plan {};
    if (!isa_supported(f) || (gated(ep) && d.N % 2 != 0)) return false;
    plan.ep = ep;
    plan.src_u8 = src_u8;
    // Source scales per weight scale group are s8 sources'.
    plan.src_groups
            = f.src == data_type_t::s8 && src_groups && layout.groups > 1;
    const scale_groups groups {
            f.scaled() && layout.groups > 1, plan.src_groups};
    // A split step is a gate vector and an up vector.
    const int vec = f.vec_cols();
    plan.step = split(ep) ? 2 * vec : kNr;
    if (split(ep)) {
        plan.up = d.N / 2;
        plan.unaligned = plan.up % vec != 0;
        plan.cols = 2 * ((plan.up + vec - 1) / vec * vec);
        plan.out_cols = plan.up;
    } else {
        plan.cols = layout.n_pad;
        plan.out_cols = gated(ep) ? d.N / 2 : d.N;
    }
    plan.max_mr = f.max_mr;
    for (int mr = 1; mr <= plan.max_mr; ++mr) {
        plan.kernels[mr] = select(f, mr, ep, src_u8, plan.unaligned, groups);
        if (plan.kernels[mr] == nullptr) return false;
    }
    if (split(ep) && plan.up % kNr == 0 && f.max_mr_wide > 0) {
        plan.wide = true;
        plan.step = 2 * kNr;
        plan.wide_max_mr = f.max_mr_wide;
        // Wide kernels of fewer rows than the narrow ones (s4's take 3)
        // leave taller tiles to the narrow ones.
        plan.wide_rows = f.wide_rows > 0 ? f.wide_rows
                                         : std::numeric_limits<int>::max();
        for (int mr = 1; mr <= plan.wide_max_mr; ++mr) {
            for (const bool two : {false, true}) {
                plan.wide_kernels[two][mr]
                        = select_wide(f, mr, ep, src_u8, two, groups);
            }
            if (plan.wide_kernels[0][mr] == nullptr) return false;
            if (mr <= f.two_chain_rows && plan.wide_kernels[1][mr] == nullptr)
                return false;
        }
    }
    // A scaled weight's columns also take their f32 scales.
    const int sub = l2_subtile_cols(
            d.K, f.src_bytes(), f.whole_bytes(), f.scaled() ? 4 : 0, budget);
    plan.subtile = std::max(sub / plan.step * plan.step, plan.step);
    return true;
}

void run_tile(const tile_plan &plan, const pack_layout &layout,
        const packed_expert &px, const int M, const tile_source &src,
        const void *bias, const bias_kind bias_t, void *dst, const int ldc,
        const int col, const int width) {
    const int lda = src.lda;
    const float *const src_scale = src.scales;
    const weight_desc &d = layout.desc;
    const format_desc &f = layout.format();
    const uint8_t *const blocks = px.blocks.get();
    const size_t bb = layout.block_bytes;
    const size_t src_es = static_cast<size_t>(f.src_bytes());
    const size_t bias_es = bias_t == bias_kind::f32 ? 4 : 2;
    const char *const bias_bytes = static_cast<const char *>(bias);
    const int32_t *const comp = px.comp == nullptr || px.comp->empty()
            ? nullptr
            : px.comp->data();
    const float *const scale = px.scale.empty() ? nullptr : px.scale.data();
    char *const out = static_cast<char *>(dst);
    const size_t dst_es = static_cast<size_t>(f.dst_bytes());

    // A split step's vectors are one 64-byte line of a block row each, 16
    // columns (32 for f16): half a block, or a block.  Its up vector starts
    // `shift` columns into vector q and runs into vector q + 1, which a step
    // whose up columns all lie in q may not have; any vector stands in for
    // it then.
    const int vec = f.vec_cols();
    const int per_block = kNr / vec;
    const int last_vec = layout.n_pad / vec - 1;
    const auto vector_at = [&](const int q) {
        return blocks + static_cast<size_t>(q / per_block) * bb
                + (q % per_block) * 64;
    };

    // A wide plan's narrow kernels take half its steps.
    const int narrow = plan.wide ? kNr : plan.step;

    ukernel_args a;
    a.lda = lda;
    a.K = d.K;
    a.group_size = d.group_size;
    a.col_group = layout.n_pad;
    a.src_groups = plan.src_groups ? layout.groups : 0;
    a.max_scale = px.max_scale;
    const size_t src_scales = static_cast<size_t>(plan.src_scales(layout));
    a.bias_t = bias == nullptr ? bias_kind::none : bias_t;
    a.ldc = ldc;
    // Points `a` at the narrow kernels' step at plan column `v` for the rows
    // from `m_off`.
    const auto set_step = [&](const int v, const int m_off) {
        int w0, out0;
        if (split(plan.ep)) {
            w0 = v / 2;
            out0 = w0;
            const int u0 = plan.up + w0;
            a.b = vector_at(w0 / vec);
            a.b_next = vector_at(u0 / vec) - a.b;
            if (plan.unaligned) {
                a.shift = u0 % vec;
                a.b_next_hi = vector_at(std::min(u0 / vec + 1, last_vec));
            }
            a.col_next = plan.up;
            a.cols = std::min(narrow / 2, plan.up - w0);
        } else {
            w0 = v;
            out0 = gated(plan.ep) ? v / 2 : v;
            a.b = blocks + static_cast<size_t>(v / kNr) * bb;
            a.b_next = per_block > 1 ? 64 : 0;
            a.col_next = 16;
            a.cols = gated(plan.ep) ? std::min(kNr / 2, (d.N - v) / 2)
                                    : std::min(kNr, d.N - v);
        }
        a.comp = comp != nullptr ? comp + w0 : nullptr;
        a.scale = scale != nullptr ? scale + w0 : nullptr;
        a.bias = bias_bytes != nullptr ? bias_bytes + w0 * bias_es : nullptr;
        a.c = out + (static_cast<size_t>(m_off) * ldc + out0) * dst_es;
    };
    // The same for a wide kernel: the gate block holding output column
    // `v / 2` and the up block `plan.up` columns on, both whole.
    const auto set_wide_step = [&](const int v, const int m_off) {
        const int w0 = v / 2;
        const size_t gate = static_cast<size_t>(w0 / kNr);
        const size_t up = static_cast<size_t>((plan.up + w0) / kNr);
        a.b = blocks + gate * bb;
        a.b_next = static_cast<ptrdiff_t>((up - gate) * bb);
        a.col_next = plan.up;
        a.cols = std::min(kNr, plan.up - w0);
        a.comp = comp != nullptr ? comp + w0 : nullptr;
        a.scale = scale != nullptr ? scale + w0 : nullptr;
        a.bias = bias_bytes != nullptr ? bias_bytes + w0 * bias_es : nullptr;
        a.c = out + (static_cast<size_t>(m_off) * ldc + w0) * dst_es;
    };

    const bool wide = plan.wide && M <= plan.wide_rows;
    const int call_mr = wide ? plan.wide_max_mr : plan.max_mr;
    const int stride = wide ? plan.step : narrow;
    // The narrow bf16 kernels sum in two chains at three rows or fewer, which
    // a tile of more rows never cuts into.
    const bool two_chains = wide && M <= f.two_chain_rows;
    const int n_calls = (M + call_mr - 1) / call_mr;
    const int mr_base = M / n_calls;
    const int n_big = M - mr_base * n_calls;
    for (int sub = col; sub < col + width; sub += plan.subtile) {
        const int sub_end = std::min(col + width, sub + plan.subtile);
        int m_off = 0;
        for (int c = 0; c < n_calls; ++c) {
            const int mr = c < n_big ? mr_base + 1 : mr_base;
            const ukernel_fn fn = wide ? plan.wide_kernels[two_chains][mr]
                                       : plan.kernels[mr];
            a.a = static_cast<const uint8_t *>(src.rows)
                    + static_cast<size_t>(m_off) * lda * src_es;
            a.src_scale = src_scale != nullptr
                    ? src_scale + static_cast<size_t>(m_off) * src_scales
                    : nullptr;
            for (int v = sub; v < sub_end; v += stride) {
                if (wide) {
                    set_wide_step(v, m_off);
                } else {
                    set_step(v, m_off);
                }
                fn(a);
            }
            m_off += mr;
        }
    }
}

} // namespace nr32
} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
