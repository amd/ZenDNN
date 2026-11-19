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

/// The source group sums a four-bit format's kernel seeds its groups from
/// (`w4_needs_src_group_sums`: MXFP4, -12 times each row's sum per group).
/// The sums depend on the source alone, so a large call computes them once,
/// before the tiles; a small one leaves them to `dispatch_tile_w4`, per tile.

#ifndef ZENDNNL_GROUP_MATMUL_N_TILE_SRC_GROUP_SUMS_HPP
#define ZENDNNL_GROUP_MATMUL_N_TILE_SRC_GROUP_SUMS_HPP

#include <cstdint>
#include <vector>

namespace zendnnl {
namespace lowoha {
namespace matmul {

/// One expert's quantized source rows and the buffer their sums go to,
/// `rows * (K / group_size)` entries once computed.
struct SrcGroupSumsJob {
    const uint8_t *src;
    int lda;
    int rows;
    int K;
    std::vector<int32_t> *sums;
};

/// Computes every job's `s8_group_sums` in one parallel pass when the jobs
/// hold at least `kSrcGroupSumsHoistMinBytes` of source, and returns whether
/// it did.
bool hoist_src_group_sums(const std::vector<SrcGroupSumsJob> &jobs,
        int group_size, int num_threads);

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_N_TILE_SRC_GROUP_SUMS_HPP
