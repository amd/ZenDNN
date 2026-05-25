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

/// Custom-kernel BF16 weight pack warm-up — `group_matmul/prepack/`
/// backend helper that pre-packs every expert's weight in
/// `[0, total_count)` so the per-process custom-kernel pack cache is
/// warm regardless of which experts get routed at run time.
/// Symmetric with the AOCL DLP warm-pack in
/// `prepack/prepack_aocl_dlp.hpp`.
///
/// Called by the per-ALGO functions in `prepack/prepack.cpp` (which
/// own the env-knob / fingerprint / inner-kernel detection); this
/// header is the backend-specific surface.  Lifted out of
/// `custom_kernel/dispatch.hpp` so the dispatcher no longer carries
/// warm-pack symbols and the prepack subdir owns every warmer.

#ifndef ZENDNNL_GROUP_MATMUL_PREPACK_CUSTOM_KERNEL_HPP
#define ZENDNNL_GROUP_MATMUL_PREPACK_CUSTOM_KERNEL_HPP

#include <vector>

#include "common/error_status.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace group_matmul_prepack {
namespace custom_kernel {

using zendnnl::error_handling::status_t;

// ─────────────────────────────────────────────────────────────────────
// Counts returned by `warm_pack_all_custom_kernel_experts()` so the
// caller can log a single PROBE summary line per fused-MoE call (or
// extend the diagnostic later — e.g. raise a counter when
// `cache_misses > 0` after warm-up to catch weight-pointer churn).
//
// Field semantics:
//   * `total_attempted` — total entries the helper iterated over.
//   * `packed_ok`       — entries that passed the pack-contract gate
//                         AND the pack call returned success
//                         (regardless of hit/miss).
//   * `cache_hits`      — entries served from the LRU cache (zero
//                         pack work).
//   * `cache_misses`    — entries that allocated + packed (the cost
//                         we're trying to amortise to call zero).
//   * `skipped_invalid` — entries refused by the contract (null
//                         weight pointer, K/N/ldb non-positive, N
//                         not a multiple of pack_nr).  Counted
//                         separately from pack-call failures so a
//                         framework that legitimately leaves a slot
//                         empty (no expert at this index in this
//                         model) is distinguishable from a real
//                         pack error.
// ─────────────────────────────────────────────────────────────────────
struct PackProbeStats {
  int total_attempted = 0;
  int packed_ok       = 0;
  int cache_hits      = 0;
  int cache_misses    = 0;
  int skipped_invalid = 0;
};

/// Pre-pack every expert's weight in `[0, total_count)` so the per-
/// process custom-kernel pack cache is warm regardless of which
/// experts get routed at run time.  Idempotent — cached entries are
/// pure pointer fetches.  Best-effort: per-entry pack failures
/// (invalid metadata, OOM during VNNI pack) are counted in
/// `stats.skipped_invalid` and ignored, not propagated to the
/// caller, because the active-set dispatcher will independently
/// re-validate any expert it actually processes.
///
/// Thread-safety: same as `get_or_pack_weight_bf16()` — runs single-
/// threaded on the caller's thread.  Each per-entry cache lookup
/// acquires the SINGLE process-wide pack mutex
/// (`pack_mutex_singleton()` in `pack.cpp`); the lock is held only
/// for the duration of the `find_key` / pack call, so concurrent
/// warmer invocations on different threads serialise rather than
/// deadlock, but are NOT parallelised.  Safe to call before an OMP
/// parallel region; callers must NOT invoke this concurrently with
/// any in-flight `dispatch_tile()` (which reads cached pointers
/// outside the mutex) or `clear_custom_kernel_pack_cache()` (which
/// drops every cached entry) on other threads.
///
/// `total_count` is the number of expert slots to probe; the helper
/// iterates `[0, min(total_count, weight.size(), K.size(), N.size(),
/// ldb.size(), transB.size()))` to defend against asymmetric vector
/// sizes the framework might pass (e.g. `weight[]` sized to
/// `total_matmul` but K/N/ldb sized to `active_matmul`).
/// `is_weights_const.size()` is INTENTIONALLY not part of the loop
/// bound: the empty-vector sentinel ("treat every entry as const",
/// the legacy behaviour for callers that don't supply the field)
/// must not clamp the loop to zero.  Instead, the per-iteration
/// guard inside the loop body checks
/// `i < is_weights_const.size() && !is_weights_const[i]` so an
/// empty vector falls through and every reachable expert is warmed.
///
/// `is_weights_const` mirrors the AOCL DLP warmer's per-expert gate
/// at `prepack_aocl_dlp.cpp::warm_pack_all_aocl_dlp_experts` — when
/// the caller flags expert `i` as variable-weight
/// (`is_weights_const[i] == false`), the warmer skips packing for
/// that expert and counts it in `stats.skipped_invalid`.  The CK
/// runtime path (`custom_kernel/dispatch.cpp::prepare_for_call`)
/// independently refuses CK for any call containing a non-const
/// active expert, so warming such experts here would always be
/// wasted work — and worse, would pollute the pack cache with
/// entries the runtime would refuse to consult coherently.  An
/// empty `is_weights_const` vector means "treat every entry as
/// const" (legacy behaviour for callers that don't pass the field).
status_t warm_pack_all_custom_kernel_experts(
    const std::vector<const void *> &weight,
    const std::vector<int>          &K,
    const std::vector<int>          &N,
    const std::vector<int>          &ldb,
    const std::vector<bool>         &transB,
    const std::vector<bool>         &is_weights_const,
    int                              total_count,
    PackProbeStats                  &stats,
    bool                             interleave_split_halves = false);

} // namespace custom_kernel
} // namespace group_matmul_prepack
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_PREPACK_CUSTOM_KERNEL_HPP
