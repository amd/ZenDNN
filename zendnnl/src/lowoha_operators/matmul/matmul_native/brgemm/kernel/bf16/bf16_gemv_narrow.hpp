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

/// BF16 narrow-N GEMV (M=1, N ∈ {1..4}) — pack-free, row-major streamer.
///
/// Companion to `bf16_gemv_bkc` for the extreme-narrow tail of the M=1
/// decode dispatch.  When N ≤ 4 the BKC kernel's pack layout pads N
/// up to `BKC_NR_PAD = 16`, wasting 75–94% of the VDPBF16PS compute
/// and 4–16× the pack footprint — catastrophic for narrow-N decode
/// shapes (e.g. K~6000, N=3).  This kernel instead
/// streams B directly out of the caller's row-major [K, N] layout and
/// accumulates in a single 128-bit xmm register (4 fp32 lanes = one
/// per output column), so there is 0% waste at N=4 and 25/50/75%
/// unused-lane waste at N=3/2/1 — still far cheaper than BKC's N-pad
/// waste because the xmm footprint is 4× smaller than the 512-bit
/// VDPBF16PS data path BKC uses.
///
/// Throughput expectation on Zen 5 (1-thread, const weights,
/// L2-resident B):
///   * N=4: FMA-throughput limited → ~1 FP32/cycle per output lane.
///   * N=3: same as N=4 (one xmm FMA does 4 dot-products regardless).
///   * N=1, 2: same upper bound; lower effective lane utilisation.
///
/// The kernel is two-chain double-buffered over K-pairs (even/odd)
/// to cover the 4-cycle VDPBF16PS latency with two independent
/// accumulators, matching the pattern in the group-matmul custom
/// microkernel.

#ifndef ZENDNNL_BF16_GEMV_NARROW_HPP
#define ZENDNNL_BF16_GEMV_NARROW_HPP

#include <cstdint>

#include "common/bfloat16.hpp"
#include "common/data_types.hpp"
#include "lowoha_operators/matmul/matmul_native/common/avx512_math.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::common::bfloat16_t;
using zendnnl::common::data_type_t;

/// Maximum N handled by the narrow kernel.  Engaging above this
/// falls back to the BKC dispatcher (which is efficient once N
/// fills a non-trivial fraction of its 16-wide tail kernel).  The
/// threshold matches the number of FP32 lanes in the 128-bit
/// VDPBF16PS result register.
inline constexpr int kBf16GemvNarrowMaxN = 4;

/// Compute `C[0..N) = alpha · (A[0..K) · B[0..K, 0..N)) + beta · C[0..N)
/// [ + bias[0..N) ] [ + fused_op(acc) ]` for a single M=1 row.
///
/// Contract:
///   * `A`   — caller's input vector (BF16), length K, contiguous.
///   * `B`   — caller's weight matrix (BF16), shape [K, N] row-major,
///             leading dim `ldb`.  Must NOT be transposed (caller
///             checks transB elsewhere).  Reads are sequential per
///             row, so the hardware prefetcher handles the stream.
///   * `N`   — 1, 2, 3, or 4.  Values outside this range violate the
///             kernel's contract; higher-N callers should use
///             `bf16_gemv_bkc` instead (runtime dispatch is a tight
///             switch in `bf16_gemv_direct`).
///   * `C_bf16` / `C_fp32` — exactly one non-null.  `dst_is_bf16`
///             selects which is written.
///   * `bias_f` — optional FP32 bias row, N elements.  `nullptr` skips.
///   * `fused_op` — optional post-op applied on the FP32 accumulator
///             just before the final BF16 cvt / FP32 store.
///
/// Returns nothing; always succeeds under the contract above.  Safe
/// to call on ultra-short K (including K=1); a single odd-K tail is
/// handled with a zero second-pair BF16 so VDPBF16PS contributes
/// only the live lane.
void bf16_gemv_narrow(
    const uint16_t *A, int K,
    const uint16_t *B, int ldb,
    uint16_t *C_bf16, float *C_fp32,
    const float *bias_f,
    fused_postop_t fused_op,
    float alpha, float beta,
    bool dst_is_bf16,
    int N);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_BF16_GEMV_NARROW_HPP
