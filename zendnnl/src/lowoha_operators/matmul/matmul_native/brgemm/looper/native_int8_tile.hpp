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

#ifndef MATMUL_NATIVE_BRGEMM_LOOPER_NATIVE_INT8_TILE_HPP
#define MATMUL_NATIVE_BRGEMM_LOOPER_NATIVE_INT8_TILE_HPP

#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_int8_call.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_planner.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_weight_view.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

/// Where compute_int8_tile stores a finished tile instead of C: element
/// (m, n) of a bf16 or f32 matrix at dst + m * ldc + n.  Pack column n must
/// be output column n (no interleaved halves).
struct native_int8_out_t {
    void *dst = nullptr;
    bool bf16 = false;
    int ldc = 0;
};

/// C[mr][nr] (f32) = dequantized rows m0 .. m0 + mr of the source times pack
/// columns n0 .. n0 + nr (a multiple of 16) of one block, plus call.bias when
/// the call carries it.  With per-tensor or per-channel weight scales alpha
/// and beta apply here (see native_int8_call_t::order) and the epilogue then
/// takes alpha 1 and beta 0; with per-group weight scales the epilogue
/// applies them.  Post-ops are left to the epilogue.  With out (alpha 1 and
/// beta 0 only), the result is stored there instead and C holds only what
/// the tile carries between K passes or batches of groups.  The column
/// widths, rows per kernel call and K passes are plan's; with per-group
/// weight scales the group kernel takes the groups in order, in batches of
/// whole groups instead of K passes.  Safe to call concurrently on disjoint
/// tiles.
void compute_int8_tile(const native_plan_t &plan,
        const native_int8_call_t &call, const native_weight_view_t &w, int m0,
        int mr, int n0, int nr, float *C, int ldc,
        const native_int8_out_t *out = nullptr);

/// compute_int8_tile of the last column tile of a weight of padded columns,
/// whose pack columns n0 .. n0 + cols lie below w.N (0 < cols, and n0 +
/// cols == w.N): C holds them, and the columns after them up to a multiple
/// of 16 hold values no store reads.  No operand of the call is read past
/// its N columns (weight scales, a bias of the dequantization, the
/// destination beta reads), and the result is never stored to the
/// destination: the caller's epilogue stores the cols columns.
void compute_int8_tail_tile(const native_plan_t &plan,
        const native_int8_call_t &call, const native_weight_view_t &w, int m0,
        int mr, int n0, int cols, float *C, int ldc);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_BRGEMM_LOOPER_NATIVE_INT8_TILE_HPP
