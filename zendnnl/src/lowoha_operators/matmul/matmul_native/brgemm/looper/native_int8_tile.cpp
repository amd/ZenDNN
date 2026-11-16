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

#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_int8_tile.hpp"

#include <cassert>

#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_int8_tile_internal.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

void compute_int8_tile(const native_plan_t &plan,
        const native_int8_call_t &call, const native_weight_view_t &w, int m0,
        int mr, int n0, int nr, float *C, int ldc,
        const native_int8_out_t *out) {
    assert(nr % 16 == 0
            && (w.wei_dt == data_type_t::s8 || w.wei_dt == data_type_t::s4));
    // A direct store writes tile column t as output column n0 + t: a plain
    // tile.  A paired tile holds runs of 16 columns of each half.
    assert(out == nullptr || !w.paired);
    assert(!w.paired || (n0 % 32 == 0 && nr % 32 == 0));
    if (w.wei_dt == data_type_t::s4) {
        compute_int4_group_tile<false>(
                plan, call, w, m0, mr, n0, nr, C, ldc, out);
        return;
    }
    if (call.scheme == native_int8_scheme_t::per_group) {
        assert(call.bias == nullptr);
        compute_group_tile<false>(plan, call, w, m0, mr, n0, nr, C, ldc, out);
        return;
    }
    compute_channel_tile<false>(plan, call, w, m0, mr, n0, nr, C, ldc, out);
}

void compute_int8_tail_tile(const native_plan_t &plan,
        const native_int8_call_t &call, const native_weight_view_t &w, int m0,
        int mr, int n0, int cols, float *C, int ldc) {
    assert(cols > 0 && n0 + cols == w.N
            && (w.wei_dt == data_type_t::s8 || w.wei_dt == data_type_t::s4));
    const int nr = (cols + 15) / 16 * 16;
    if (w.wei_dt == data_type_t::s4) {
        compute_int4_group_tile<true>(
                plan, call, w, m0, mr, n0, nr, C, ldc, nullptr);
        return;
    }
    if (call.scheme == native_int8_scheme_t::per_group) {
        assert(call.bias == nullptr);
        compute_group_tile<true>(
                plan, call, w, m0, mr, n0, nr, C, ldc, nullptr);
        return;
    }
    compute_channel_tile<true>(plan, call, w, m0, mr, n0, nr, C, ldc, nullptr);
}

void finish_int8_channel_tile(const native_plan_t &plan,
        const native_int8_call_t &call, const native_weight_view_t &w, int m0,
        int mr, int n0, int nr, int cols, float *C, int ldc) {
    assert(w.wei_dt == data_type_t::s8
            && call.scheme == native_int8_scheme_t::per_channel);
    if (cols < nr) {
        assert(n0 + cols == w.N);
        compute_channel_tile<true>(plan, call, w, m0, mr, n0,
                (cols + 15) / 16 * 16, C, ldc, nullptr, true);
        return;
    }
    compute_channel_tile<false>(
            plan, call, w, m0, mr, n0, nr, C, ldc, nullptr, true);
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
