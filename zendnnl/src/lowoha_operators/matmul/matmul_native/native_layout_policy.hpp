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

#ifndef MATMUL_NATIVE_NATIVE_LAYOUT_POLICY_HPP
#define MATMUL_NATIVE_NATIVE_LAYOUT_POLICY_HPP

#include <cstdint>
#include <limits>

#include "lowoha_operators/matmul/lowoha_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

/// Columns per block of every weight the layout rule places on the native
/// layout.
inline constexpr int kNativeLayoutPackNr = 32;

/// Largest N of a weight whose columns, rounded up to whole blocks
/// (native_padded_n: the width the planner, the packers and the tiles run
/// on), an int holds.
inline constexpr int kMaxNativeN = std::numeric_limits<int32_t>::max()
        / kNativeLayoutPackNr * kNativeLayoutPackNr;

/// Largest K of a bf16 weight whose K pairs the bf16 kernels offset in an
/// int: each pair of a block lies 64 elements after the one before.
inline constexpr int kMaxNativeBf16K = 1 << 26;

/// Largest K of an f16 weight whose source rows a call on the F16 kernels
/// widens in steps of 16 elements, and whose K pairs the [N, K] pack counts,
/// in an int.
inline constexpr int kMaxNativeF16K
        = std::numeric_limits<int32_t>::max() / 16 * 16;

/// The weight-only facts the layout rule decides on.  Execution-only fields
/// (M, threads, alpha / beta, bias, post-ops, dst type) must not enter the
/// decision: a weight rewritten in place can only be served by the backend
/// that rewrote it, on every later call.
struct native_layout_query_t {
    data_type_t wei_dt = data_type_t::none;
    /// Source type the kernel multiplies with: bf16 for BF16, u8 / s8 for
    /// INT8 (s8 or s4 weights), including a bf16 or f32 source the call
    /// quantizes dynamically, f32 for FP32, f16 for F16.
    data_type_t src_dt = data_type_t::none;
    int K = 0;
    int N = 0;
    int ldb = 0;
    bool transB = false;
    /// Weight scale group along K; 0 or K means per-channel or per-tensor,
    /// -1 groups that do not split K evenly.
    int wei_group_size = 0;
};

struct native_layout_decision_t {
    bool native = false;
    int pack_nr = 0; ///< kNativeLayoutPackNr when native
    const char *reason = ""; ///< why the native layout is declined
    bool isa = false; ///< declined for the CPU's instruction set
};

/// Whether ALGO 10 packs the weight in the native VNNI layout with pack_nr
/// columns per block, read only by the native BRGEMM.  Native when all hold:
///   * a CPU with AVX-512 F, BW, VL and DQ and FMA (native_base_isa);
///   * wei bf16 with src bf16 on a CPU with AVX512-BF16, wei s8 with src
///     u8 / s8 or wei s4 with src s8 on a CPU with AVX512-VNNI, wei f32
///     with src f32, or wei f16 with src f16 (on AVX-512 F; the kernels
///     of a CPU with AVX512-FP16 read the same layout);
///   * N from 1 to kMaxNativeN, whose whole blocks of 32 columns an int
///     holds (a weight of N not a multiple of 32 is held padded with zero
///     columns to whole blocks), K >= 1, ldb >= (transB ? K : N);
///   * bf16: K at most kMaxNativeBf16K, whose pairs the kernels offset in an
///     int;
///   * f16: K at most kMaxNativeF16K, whose steps of 16 an int holds;
///   * s8: K <= INT32_MAX / (255 * 128), the bound that keeps the int32
///     accumulation exact, and per-group weight scales only over groups of
///     whole K quads (a group size that divides K and is a multiple of 4);
///   * s4: groups of whole K quads (one group over K without per-group
///     weight scales) of at most kMaxSafeInt4GroupK values, and K at most
///     kMaxNativeInt4K, whose whole octets an int holds.
native_layout_decision_t decide_native_layout(const native_layout_query_t &q);

/// AVX-512 F, BW, VL and DQ and FMA: what the native kernels and epilogue
/// use besides AVX512-BF16 / AVX512-VNNI (native_hw_t::base_isa).
bool native_base_isa();

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_NATIVE_LAYOUT_POLICY_HPP
