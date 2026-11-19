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

/// Per-group sums of s8 source rows, for a four-bit format whose kernel
/// seeds each group from them (`w4_needs_src_group_sums`).

#ifndef ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_UKERNEL_S8_GROUP_SUMS_HPP
#define ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_UKERNEL_S8_GROUP_SUMS_HPP

#include <cstdint>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace custom_kernel {

/// Σ of each source row's s8 bytes per `group_size` block, written to
/// `out[m * (K / group_size) + g]` for `rows` rows of `A` (stride `lda`).
/// The sums depend on the source alone: `n_tile_execute` computes them once
/// for a call with `kSrcGroupSumsHoistMinBytes` of source
/// (`hoist_src_group_sums`), and `dispatch_tile_w4` per tile for a smaller
/// one.  `group_size` is a multiple of 8.
void s8_group_sums(const uint8_t *A, int lda, int rows, int K, int group_size,
        int32_t *out);

} // namespace custom_kernel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_UKERNEL_S8_GROUP_SUMS_HPP
