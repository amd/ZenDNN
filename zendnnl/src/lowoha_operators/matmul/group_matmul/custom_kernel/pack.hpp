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
/// Shared by both non-fused flat_n_tile (ALGO 3) and the fused-MoE
/// tight-arena path.
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
/// Keyed on (weight_ptr, K, N, pack_nr, ldb, transB) — different
/// NR / layout / stride variants of the same weight occupy distinct
/// cache entries.
///
/// Eviction policy: the per-process custom-kernel pack cache uses a
/// hard-capped capacity of `UINT32_MAX` entries (effectively
/// unbounded for any realistic workload) and DOES NOT EVICT during
/// regular operation — `ZENDNNL_LRU_CACHE_CAPACITY` does NOT govern
/// this cache.  The reason: raw packed-weight pointers are captured
/// into `CallContext.packed_ptrs[]` and read by OMP workers outside
/// the cache mutex, so any eviction during in-flight dispatch would
/// be a use-after-free.  The cache is reset only by an explicit
/// call to `clear_custom_kernel_pack_cache()` in a quiescent window
/// (see safety contract on that function).

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

/// Look up the pre-packed BF16 weight for one
/// (weight, K, N, pack_nr, transB) tuple, packing on the first miss
/// and caching for subsequent calls.
///
/// Contract:
///   * `weight`  — caller-owned BF16 buffer.  Logical layout depends
///                 on `transB`:
///                   * `transB == false` → row-major [K, N] with row
///                     stride `ldb` (typically `ldb == N`).
///                   * `transB == true`  → row-major [N, K] with row
///                     stride `ldb` (typically `ldb == K`).  This
///                     matches the standard `nn.Linear` weight
///                     convention used by major LLM serving stacks
///                     (`y = x @ w^T`, weight stored as
///                     `[out_features, in_features]`).
///                 Pointer identity is part of the cache key (callers
///                 must keep it alive while the entry is cached).
///   * `K`, `N`  — logical dimensions; `N` MUST be a multiple of
///                 `pack_nr` regardless of `transB`.
///   * `ldb`     — caller's row stride.  Pack reads the weight as
///                 `weight[row * ldb + col]` (row-major) — passing the
///                 actual stride (rather than assuming N or K) lets
///                 the pack handle non-contiguous slices correctly.
///   * `pack_nr` — pack width, MUST be one of {kNRMin, kNRMax}.
///   * `transB`  — false: standard [K, N] caller layout.
///                 true:  PyTorch [N, K] caller layout.  The pack
///                 produces an identical VNNI output for both
///                 layouts; only the input access pattern differs.
///
/// On success `*out_packed` points into cache-owned storage that is
/// retained for the lifetime of the process (or until
/// `clear_custom_kernel_pack_cache()` is called during a quiescent
/// window).  The cache never evicts during regular operation so raw
/// pointers held in `CallContext.packed_ptrs[]` across an OMP region
/// cannot be freed underneath the microkernel.  Returns failure on
/// bad arguments or out-of-memory while packing.
///
/// `transB` is folded into the cache key so transB=true and
/// transB=false on the same weight pointer occupy distinct entries
/// (would otherwise alias and produce silent-wrong output).
///
/// `interleave_split_halves` (default false) — when true, the pack
/// reads source columns in interleaved order so the resulting packed
/// arena physically matches the `swiglu_oai_mul` layout.  Used for
/// the `silu_and_mul` and `gelu_and_mul` fused-CK paths: in both
/// cases the caller's W13 weight is in canonical split-halves
/// `[gate_cols | up_cols]` layout, but the CK kernel's in-register
/// pair-store epilogues (`silu_and_mul_store_pair`,
/// `gelu_and_mul_store_pair`) require the interleaved
/// `[g0, u0, g1, u1, ...]` order.  silu and gelu share the SAME
/// permutation — only the kernel-side activation math differs — so
/// the cache-key bit is shared between them and a packed arena
/// warmed for silu can be reused for gelu on the same weight pointer.
/// When the flag is set, pack output column `2i+0` reads canonical
/// column `i` (gate i) and pack output column `2i+1` reads canonical
/// column `I + i` (up i), where `I = N / 2`.  Requires N to be even.
/// Folded into the cache key so the same weight pointer packed with
/// and without interleaving occupies distinct entries.
///
/// `was_hit_out` (optional) — when non-null, set to `true` if this
/// call served from the cache (no pack work) or `false` if it had
/// to allocate + pack on a miss.  Used by the prepack-all probe
/// helper (see `warm_pack_all_custom_kernel_experts`) to report
/// per-call HIT/MISS counts at apilog level.  Default `nullptr`
/// preserves the pre-existing call sites unchanged.
status_t get_or_pack_weight_bf16(
    const bfloat16_t *weight,
    int K, int N, int ldb, int pack_nr,
    bool transB,
    bool interleave_split_halves,
    const bfloat16_t **out_packed,
    bool *was_hit_out = nullptr);

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
