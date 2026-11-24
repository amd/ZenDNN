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

#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/engine/source.hpp"

#include "lowoha_operators/reorder/lowoha_reorder.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {
namespace detail {

source_spec source_of(
        const nr32::pack_layout &layout, const nr32::tile_plan &plan) {
    source_spec s;
    s.s8 = layout.format().src == data_type_t::s8;
    s.groups = plan.src_scales(layout);
    s.u8 = s.s8 && plan.src_u8;
    s.elem = layout.format().src_bytes();
    return s;
}

void quantize_row(const source_spec &spec, const uint16_t *in, const int len,
        source_rows &out, const int r) {
    int8_t *const row = reinterpret_cast<int8_t *>(out.row(spec, r, len));
    float *const scale = out.row_scales(spec, r);
    const int run = len / spec.groups;
    for (int i = 0; i < spec.groups; ++i) {
        reorder::dynamic_per_token_quant_bf16_s8_native(
                in + i * run, row + i * run, scale + i, 1, run);
    }
    if (spec.u8) flip_to_u8(reinterpret_cast<uint8_t *>(row), len);
}

} // namespace detail
} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
