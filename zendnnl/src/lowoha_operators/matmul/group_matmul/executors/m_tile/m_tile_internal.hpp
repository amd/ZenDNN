/*******************************************************************************
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

/// M-tile slice helpers shared by the ALGO 2 executor and the fused-MoE
/// vertical-fusion pipeline.

#ifndef ZENDNNL_GROUP_MATMUL_M_TILE_INTERNAL_HPP
#define ZENDNNL_GROUP_MATMUL_M_TILE_INTERNAL_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "lowoha_operators/matmul/group_matmul/common/grp_matmul_constants.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

// ── A.1  offset_quant_by_row ────────────────────────────────────────────

// Row-offset src quant buffer for M-tile.
// Handles per-token {M,1} and per-group {M,G}: offset = row_start × row_stride.
// Per-tensor (dims empty, total elements == 1, or first dim == 1): no offset.
//
// `slice_M` rewrites the slice's view of dims[0] from the full per-expert M
// down to the per-thread row count.  This is required for dynamic-quant: the
// reorder dispatcher gates the per-group kernel on `is_per_group_col_dims`,
// which compares `dims[0]` against `src_shape[0]` (= slice_M for the M-tile).
// Static-quant kernels that read scales by row index also benefit.
inline void offset_quant_by_row(matmul_quantization_params_t::matmul_quant_t &q,
        int row_start, int slice_M) {
    if (q.dims.empty()) return;
    int64_t nelems = 1;
    for (auto dim : q.dims) {
        if (dim <= 0) return;
        nelems *= dim;
    }
    if (nelems <= 1 || q.dims[0] <= 1) return;
    const size_t rows = static_cast<size_t>(q.dims[0]);
    if (rows == 0 || (static_cast<size_t>(nelems) % rows) != 0) return;
    if (q.buff != nullptr) {
        const size_t row_stride = static_cast<size_t>(nelems) / rows;
        const size_t elem = size_of(q.dt);
        q.buff = static_cast<const uint8_t *>(q.buff)
                + static_cast<size_t>(row_start) * row_stride * elem;
    }
    if (slice_M > 0 && static_cast<size_t>(slice_M) <= rows) {
        q.dims[0] = static_cast<int64_t>(slice_M);
    }
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_M_TILE_INTERNAL_HPP
