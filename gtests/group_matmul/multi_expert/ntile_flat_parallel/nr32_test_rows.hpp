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

/// N-tile flat parallel's format rows by weight type, for the tests of
/// today's formats, each of which has a weight type of its own.

#ifndef ZENDNNL_GTESTS_NTILE_FLAT_PARALLEL_NR32_TEST_ROWS_HPP
#define ZENDNNL_GTESTS_NTILE_FLAT_PARALLEL_NR32_TEST_ROWS_HPP

#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/formats/formats.hpp"

namespace nr32_test {

/// The only row of weight type `wei`, or null when it has none or several.
inline const zendnnl::lowoha::matmul::ntile_flat_parallel::nr32::format_desc *
row_of(const zendnnl::common::data_type_t wei) {
    namespace nr32 = zendnnl::lowoha::matmul::ntile_flat_parallel::nr32;
    const nr32::format_desc *found = nullptr;
    for (const nr32::format_desc &f : nr32::kFormats) {
        if (f.wei != wei) continue;
        if (found != nullptr) return nullptr;
        found = &f;
    }
    return found;
}

} // namespace nr32_test

#endif // ZENDNNL_GTESTS_NTILE_FLAT_PARALLEL_NR32_TEST_ROWS_HPP
