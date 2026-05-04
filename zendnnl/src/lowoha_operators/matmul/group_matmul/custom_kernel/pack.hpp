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

/// BF16 custom kernel — adaptive weight pack and per-process pack cache.
/// Shared by both non-fused flat_n_tile (ALGO 3) and fused MoE V2.
///
/// PACK LAYOUT (BF16, VDPBF16PS-friendly, "VNNI" style — same as
/// `bf16_brgemm_ukernel.cpp`'s NR-templated layout):
///
///     packed[O / pack_nr][K / 2][pack_nr][2]
///
///   * `pack_nr` is a runtime parameter — supported values 32 and 64.
///     Larger NR = fewer microkernel calls per N-tile, smaller NR =
///     more N-divisibility flexibility and tighter register budget.
///   * The trailing dim 2 packs two consecutive K rows, byte-laid as
///       byte[0..1] = row k        (BF16, lo)
///       byte[2..3] = row k+1      (BF16, hi)
///     so each `vmovdqu` of 64 bytes loads 16 N-cols of one K-pair
///     as 16 contiguous (k_lo, k_hi) FP32-lane-formatted pairs.
///   * For odd K the implementation pads the trailing K-pair with a
///     zero second element (no contribution to the dot product).
///
/// CACHE: shared `lru_cache_t<Key_matmul, void *>` (the same template
/// the AOCL DLP / oneDNN reorder caches in `aocl_kernel.cpp` use).
/// Keyed on (weight_ptr, K, N, pack_nr) — different NR variants of
/// the same weight occupy distinct cache entries.  Eviction follows
/// `ZENDNNL_LRU_CACHE_CAPACITY`.

#ifndef ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_PACK_HPP
#define ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_PACK_HPP

#include "common/bfloat16.hpp"
#include "common/error_status.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace custom_kernel {

using zendnnl::common::bfloat16_t;
using zendnnl::error_handling::status_t;

/// VNNI K-pair multiplicity (= 2: two consecutive K-rows per VDPBF16PS lane).
inline constexpr int kVNNIPair = 2;

/// Supported pack widths.  Match the `bf16_brgemm_ukernel.cpp` set
/// minus NR=16 (gated activation needs at least 32 cols for one
/// (gate, up) pair to fit in a single (acc_lo, acc_hi) zmm pair).
inline constexpr int kNRMin = 32;
inline constexpr int kNRMax = 64;

/// Look up the pre-packed BF16 weight for one (weight, K, N, pack_nr)
/// tuple, packing on the first miss and caching for subsequent calls.
///
/// Contract:
///   * `weight`  — caller-owned BF16 row-major buffer of shape [K, N].
///                 Pointer identity is part of the cache key (callers
///                 must keep it alive while the entry is cached).
///   * `K`, `N`  — dimensions; `N` MUST be a multiple of `pack_nr`.
///   * `pack_nr` — pack width, MUST be one of {kNRMin, kNRMax}.
///   * `transB`  — false (only row-major weight is supported today).
///
/// On success `*out_packed` points into cache-owned storage that is
/// retained for the lifetime of the process (or until
/// `clear_custom_kernel_pack_cache()` is called during a quiescent
/// window).  The cache never evicts during regular operation so raw
/// pointers held in `CallContext.packed_ptrs[]` across an OMP region
/// cannot be freed underneath the microkernel.  Returns failure on
/// bad arguments or out-of-memory while packing.
status_t get_or_pack_weight_bf16(
    const bfloat16_t *weight,
    int K, int N, int pack_nr,
    bool transB,
    const bfloat16_t **out_packed);

/// Release every cached packed BF16 weight and reset the cache to
/// empty.  Intended for weight-rotating deployments (dynamic
/// LoRA hot-swap, retraining loops, recompile / redeploy cycles)
/// that would otherwise accumulate entries across workload phases
/// and eventually exhaust RSS.
///
/// ⚠ SAFETY: the pack cache has no eviction during regular
/// operation because raw packed pointers are captured into
/// `CallContext.packed_ptrs[]` and read by OMP workers outside the
/// cache mutex (see the detailed comment in `pack.cpp`).  Calling
/// this function while ANY other thread is in an active custom-
/// kernel dispatch would use-after-free whatever entry the
/// dispatcher is currently reading.  Callers MUST ensure a
/// quiescent window — no in-flight calls to
/// `group_matmul_direct(...)` or its fused-MoE variants with
/// `ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL=1` on ANY thread — before
/// invoking.  Typical integration points are:
///   * Between inference batches in a serving framework that joins
///     its worker threads at a batch boundary.
///   * After model unload / before reload in a LoRA hot-swap
///     sequence that serialises model-lifecycle events through a
///     single control thread.
///   * During graceful shutdown.
///
/// After return the cache is empty and future calls to
/// `get_or_pack_weight_bf16()` pay the repack cost on first miss.
void clear_custom_kernel_pack_cache();

} // namespace custom_kernel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_PACK_HPP
