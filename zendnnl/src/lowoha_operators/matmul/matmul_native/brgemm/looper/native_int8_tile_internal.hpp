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

#ifndef MATMUL_NATIVE_BRGEMM_LOOPER_NATIVE_INT8_TILE_INTERNAL_HPP
#define MATMUL_NATIVE_BRGEMM_LOOPER_NATIVE_INT8_TILE_INTERNAL_HPP

#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_int8_tile.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

inline constexpr int kMaxTileMr = 12;
inline constexpr int kMaxTileNr = 64;

/// compute_int8_tile of a call with per-tensor or per-channel weight scales.
void compute_channel_tile(const native_plan_t &plan,
        const native_int8_call_t &call, const native_weight_view_t &w, int m0,
        int mr, int n0, int nr, float *C, int ldc,
        const native_int8_out_t *out);

/// compute_int8_tile of a call with per-group weight scales: C holds the sum
/// of the scaled groups; alpha, beta and the bias are the epilogue's.
void compute_group_tile(const native_plan_t &plan,
        const native_int8_call_t &call, const native_weight_view_t &w, int m0,
        int mr, int n0, int nr, float *C, int ldc);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_BRGEMM_LOOPER_NATIVE_INT8_TILE_INTERNAL_HPP
