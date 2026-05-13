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

/// AOCL DLP weight pack warm-up — `group_matmul/prepack/` backend
/// helper that preloads the per-process AOCL DLP reorder cache for
/// every expert in `[0, total_count)` ahead of inference, so
/// subsequent group_matmul / fused-MoE dispatches never pay a reorder
/// cost regardless of which expert gets routed.  Symmetric with the
/// custom-kernel `warm_pack_all_custom_kernel_experts` in
/// `prepack/prepack_custom_kernel.hpp`.
///
/// Implementation note: the helper lives in `group_matmul/prepack/`
/// (not `backends/aocl/`) so the shared AOCL kernel files stay
/// untouched.  It uses only the public API surface of
/// `aocl_kernel.hpp` (specifically `reorderAndCacheWeights<T>`) plus
/// the AOCL DLP libdlp reorder primitives.  Per-call HIT/MISS counts
/// are NOT tracked on the AOCL side because the per-dtype LRU cache
/// is private to `aocl_kernel.cpp`; we only report total
/// pack-attempts and skip-invalid counts at the apilog probe line.
///
/// Called by the per-ALGO functions in `prepack/prepack.cpp` (which
/// own the env-knob / fingerprint / inner-kernel detection); this
/// header is the backend-specific surface, agnostic to scheduling
/// ALGO and to whether the upstream caller is fused-MoE or plain
/// group_matmul.

#ifndef ZENDNNL_GROUP_MATMUL_PREPACK_AOCL_DLP_HPP
#define ZENDNNL_GROUP_MATMUL_PREPACK_AOCL_DLP_HPP

#include <vector>

#include "common/data_types.hpp"
#include "common/error_status.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace group_matmul_prepack {
namespace aocl_dlp {

using zendnnl::common::data_type_t;
using zendnnl::error_handling::status_t;

/// PROBE counters for the AOCL DLP warm-pack.  Reported in the
/// fused-MoE entry's `Level3 PACK_PROBE` line so a model-level run
/// can see at a glance how many entries the helper actually touched
/// and how many were rejected.  HIT/MISS is intentionally not tracked
/// here — that would require modifying `aocl_kernel.cpp` to expose
/// the private LRU cache, and the user wants
/// `backends/aocl/aocl_kernel.{hpp,cpp}` left strictly alone.
struct AoclDlpPackProbeStats {
  int total_attempted = 0;
  int packed_ok       = 0;
  int skipped_invalid = 0;
};

/// Pre-populate the AOCL DLP weight reorder cache for every expert
/// in `[0, total_count)`.  Same purpose as the custom-kernel
/// `warm_pack_all_custom_kernel_experts(...)` — eliminate the
/// per-decode-step pack-cost spike when an expert is routed for the
/// first time mid-inference.  Idempotent; cached entries fast-path
/// inside `reorderAndCacheWeights<int16_t>(...)`.
///
/// Currently supports `wei_dtype = data_type_t::bf16` (matches the
/// current target envelope and the only dtype the bench actually
/// exercises); any other dtype causes the helper to count every
/// entry as `skipped_invalid` and return success without touching
/// the cache.  Adding f32/f16/int8 is a one-`else if` branch
/// extension when a workload demands it.
///
/// Defensive `min()` against every metadata-vector size so a
/// framework that passes `weight[]` at total_matmul but
/// `K[]`/`N[]`/`ldb[]` at active_matmul still works — the helper
/// only packs experts where every metadata vector reaches that
/// index.  Per-entry pack failures (null weight, OOM during
/// reorder, dtype mismatch) are counted in
/// `stats.skipped_invalid` and ignored, so a transient error on a
/// non-fired slot can't break the inference call.
///
/// Production-parity gates:
///
///   * `is_weights_const[i]` — `run_dlp(...)` only enters the AOCL
///     reorder cache when this is `true` for the expert (see
///     aocl_kernel.cpp:1700-1702).  The warmer mirrors that gate so
///     experts the dispatcher will not cache also won't be packed
///     here — counted as `skipped_invalid` to keep telemetry
///     consistent.  Pass an empty vector to disable the check (every
///     entry treated as const, the legacy behaviour).
///
///   * `matmul_config_t::instance().get_weight_cache()` — when the
///     environment variable `ZENDNNL_MATMUL_WEIGHT_CACHE=0` (or the
///     config is otherwise overridden) `reorderAndCacheWeights<T>`
///     takes the in-place / no-cache branch (`weight_cache_type=0`)
///     so persistent caching is off across the entire process.  The
///     warmer detects this once at entry and short-circuits with
///     `total_attempted = 0` so no work is wasted.
///
/// Thread-safety: same as the underlying
/// `reorderAndCacheWeights<T>` — runs single-threaded on the
/// caller's thread, takes the AOCL pack mutex per entry.  Safe to
/// call before an OMP parallel region; callers must NOT invoke
/// this concurrently with any in-flight `run_dlp(...)` or
/// `clear_aocl_matmul_weight_caches()` on other threads.
status_t warm_pack_all_aocl_dlp_experts(
    const std::vector<const void *> &weight,
    const std::vector<int>          &K,
    const std::vector<int>          &N,
    const std::vector<int>          &ldb,
    const std::vector<bool>         &transB,
    const std::vector<bool>         &is_weights_const,
    int                              total_count,
    data_type_t                      wei_dtype,
    AoclDlpPackProbeStats           &stats);

/// Per-tile variant of `warm_pack_all_aocl_dlp_experts`, sized for
/// ALGO 3 flat_n_tile's strict-stable plan
/// (`ZENDNNL_GRP_MATMUL_AOCL_STABLE_NTILE=1`, default).
///
/// The strict-stable plan (group_matmul_n_tile.cpp `plan_group_n_tile`,
/// strict-stable branch) forces every expert team to exactly
/// `stable = aocl_stable_n_thr(num_threads)` participating threads,
/// then `do_tile()` partitions each expert's `[0, N[e])` columns via
/// `aligned_n_split(N[e], stable_e, tid, nr_align)` where
/// `stable_e = min(stable, N[e]/nr_align, num_threads)` accounts for
/// the per-expert `participating_n_thr` clamps.  Each tile builds an
/// AOCL DLP cache key from the SLICED weight pointer, sliced N
/// (= `n_tile`), and unchanged K/ldb/transB — so the legacy
/// full-weight `warm_pack_all_aocl_dlp_experts` populates a key the
/// runtime never queries (silent miss / wasted reorders).
///
/// This warmer mirrors the runtime decomposition so every key the
/// runtime builds at ALGO 3 dispatch time has a populated entry in
/// the AOCL LRU after warm-up.
///
/// Callers (today: `prepack_for_algo_3` only) should fall back to
/// the full-weight `warm_pack_all_aocl_dlp_experts` when:
///   * `ZENDNNL_GRP_MATMUL_AOCL_STABLE_NTILE=0` (legacy dynamic plan
///     — per-call n_thr varies, no single per-tile key set matches
///     all calls; legacy full-weight warm preserves the old
///     partial-coverage behaviour).
///   * Narrow-N escape: `stable * nr_align > max_N` — planner
///     routes to Sequential which uses the full-weight key; per-tile
///     warm would populate keys the runtime never queries.
///   * `num_threads <= 0` (no thread context passed).
///
/// Per-tile total memory ≈ full-weight total memory (one large
/// reorder buffer per expert split into `stable` smaller buffers).
/// Per-tile reorder count = `total_count * effective_stable` vs
/// full-weight `total_count`; the AOCL reorder primitives fast-path
/// on duplicate keys so re-warming an already-cached tile is free.
status_t warm_pack_all_aocl_dlp_experts_n_tile(
    const std::vector<const void *> &weight,
    const std::vector<int>          &K,
    const std::vector<int>          &N,
    const std::vector<int>          &ldb,
    const std::vector<bool>         &transB,
    const std::vector<bool>         &is_weights_const,
    int                              total_count,
    data_type_t                      wei_dtype,
    int                              num_threads,
    int                              stable,
    int                              nr_align,
    AoclDlpPackProbeStats           &stats);

} // namespace aocl_dlp
} // namespace group_matmul_prepack
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_PREPACK_AOCL_DLP_HPP
