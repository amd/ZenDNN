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

#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/pack/scales.hpp"

#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/formats/formats.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {
namespace detail {

bool scale_form(const std::vector<int64_t> &dims, const data_type_t dt,
        const nr32::format_desc &f, const int K, const int N,
        scale_form_t &out) {
    out = scale_form_t {};
    out.dt = dt;
    out.dims = dims;
    const int64_t count = element_count(dims);
    if (!scale_type(dt) || count <= 0) return false;
    if (count == 1 || count == N) {
        out.per_tensor = count == 1 && N != 1;
        // One group along K, which a format that always groups its scales
        // (s4) takes in whole words.
        return !f.always_grouped || K % f.k_per_word == 0;
    }
    if (count % N != 0) return false;
    out.groups = count / N;
    // A group holds whole words.
    return K % out.groups == 0 && (K / out.groups) % f.k_per_word == 0;
}

int pack_group_size(
        const nr32::format_desc &f, const int K, const scale_form_t &scales) {
    return f.always_grouped || scales.groups > 1
            ? static_cast<int>(K / scales.groups)
            : 0;
}

} // namespace detail
} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
