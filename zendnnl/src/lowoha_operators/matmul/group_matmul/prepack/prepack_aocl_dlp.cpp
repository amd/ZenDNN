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

#include "prepack_aocl_dlp.hpp"

#include <algorithm>

#include "common/zendnnl_global.hpp"
#include "lowoha_operators/matmul/backends/aocl/aocl_kernel.hpp"
#include "lowoha_operators/matmul/group_matmul/group_matmul_parallel_common.hpp"
#include "lowoha_operators/matmul/lowoha_matmul.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace group_matmul_prepack {
namespace aocl_dlp {

using zendnnl::error_handling::status_t;
using zendnnl::ops::matmul_algo_t;
using zendnnl::ops::matmul_config_t;

// ─────────────────────────────────────────────────────────────────────
// AOCL DLP warm-pack — populate the per-dtype LRU reorder cache for
// every expert in `[0, total_count)`.  See prepack_aocl_dlp.hpp for
// the contract.
//
// Implementation strategy:
//   * Build a `Key_matmul` matching exactly what the production
//     `run_dlp(...)` uses (transB, K, N, ldb, weight ptr, algo
//     marker = `aocl_dlp_blocked`, extra_input_hash = 0).
//   * Call `reorderAndCacheWeights<int16_t>(...)` with the BF16
//     reorder primitives — the same call the fused-MoE Op1 / Op2
//     dispatch path makes when `dtypes.wei == bf16`.
//   * `weight_cache_type = 1` selects out-of-place reorder + LRU
//     cache (the production setting); the function fast-paths on a
//     cache hit so re-warming a previously seen expert is free.
//
// We don't track HIT vs MISS separately on the AOCL side because
// the LRU cache is private to `aocl_kernel.cpp` (anonymous-
// namespace `get_aocl_weight_cache<T>()`) and the user's review
// constraint forbids modifying that file.  Per-pack hit / miss
// detail is still surfaced through `reorderAndCacheWeights`'s own
// apilog lines (`AOCL reorder weights WEIGHT_CACHE_OUT_OF_PLACE` on
// miss, `Read AOCL cached weights WEIGHT_CACHE_OUT_OF_PLACE` on
// hit), which are already gated on apilog level info.
//
// `AoclDlpPackProbeStats` counter semantics:
//   * `total_attempted` — items entering the warm loop (per expert
//     for the full-weight warmer, per (expert × participating tile)
//     for the per-tile warmer).
//   * `skipped_invalid` — items the warmer chose not to pack.  The
//     pre-pack reasons today are:
//        1) wei_dtype != bf16          (no BF16 reorder wired yet)
//        2) weight ptr null / K|N|ldb non-positive
//        3) ldb below minimum row stride (transB ? K : N) — mirrors
//           the runtime dispatch + CK warm-pack rule; prevents the
//           AOCL reorder from aliasing adjacent rows on an
//           undersized tail-slot stride
//        4) is_weights_const flag says "variable weight"
//        5) per-tile aligned_n_split returns n_tile <= 0
//     `reorderAndCacheWeights<int16_t>` itself returns `true`
//     unconditionally today (see aocl_kernel.cpp:1460-1528), so a
//     post-pack "return false" branch is unreachable — we used to
//     have a dead `else ++stats.skipped_invalid;` after the call,
//     dropped to keep the wiring honest.
//   * `packed_ok` — items where `reorderAndCacheWeights<int16_t>`
//     was actually invoked (cache miss → reorder + insert, or
//     cache hit → no-op return).
//
// Skipped entries are ignored — a transient validity-check failure
// on a non-fired slot must not break the inference call.  The
// active-set dispatcher independently re-validates anything it
// processes.
// ─────────────────────────────────────────────────────────────────────
status_t warm_pack_all_aocl_dlp_experts(
    const std::vector<const void *> &weight,
    const std::vector<int>          &K,
    const std::vector<int>          &N,
    const std::vector<int>          &ldb,
    const std::vector<bool>         &transB,
    const std::vector<bool>         &is_weights_const,
    int                              total_count,
    data_type_t                      wei_dtype,
    AoclDlpPackProbeStats           &stats) {

  if (total_count <= 0) return status_t::success;

  // ── Production-cache gate ────────────────────────────────────────
  // `run_dlp(...)` reads `matmul_config_t::instance().get_weight_cache()`
  // (driven by `ZENDNNL_MATMUL_WEIGHT_CACHE`) and routes through the
  // no-cache branch (`weight_cache_type=0`) when it returns 0.
  // Warming the LRU here would put entries in a cache that the
  // dispatcher will not consult — wasted CPU + memory.
  // Short-circuit at entry; `stats` stays zeroed so the PREPACK
  // log line surfaces "no work attempted" rather than "all skipped".
  const int32_t weight_cache_type =
      matmul_config_t::instance().get_weight_cache();
  if (weight_cache_type != 1) {
    return status_t::success;
  }

  // Reachable-entry bound: `[0, bound)` is the largest range every
  // metadata vector covers.  Compute it once up front so the
  // dtype-skip path and the normal path agree on what `total_attempted`
  // and `skipped_invalid` should sum to.  Bounding by only
  // `weight.size()` (or only `total_count`) in the dtype-skip path
  // would overcount skips when one of K / N / ldb / transB happens
  // to be shorter than weight.
  const size_t bound = std::min<size_t>({
      static_cast<size_t>(total_count),
      weight.size(),
      K.size(),
      N.size(),
      ldb.size(),
      transB.size()});

  // Only BF16 wired today (matches the current target envelope and
  // our active bench config).  Other dtypes return success with every reachable
  // entry counted as `skipped_invalid` so the caller's PROBE line
  // still surfaces accurate counts, and we can extend in-place with
  // one `else if` per dtype when a workload needs it.  Mirrors the
  // normal path's `total_attempted` accounting so the two regimes
  // are comparable across runs.
  if (wei_dtype != data_type_t::bf16) {
    stats.total_attempted += static_cast<int>(bound);
    stats.skipped_invalid += static_cast<int>(bound);
    return status_t::success;
  }

  for (size_t i = 0; i < bound; ++i) {
    ++stats.total_attempted;

    if (weight[i] == nullptr || K[i] <= 0 || N[i] <= 0 || ldb[i] <= 0) {
      ++stats.skipped_invalid;
      continue;
    }

    // Minimum-row-stride gate: mirrors the same check the dispatch
    // path applies in `group_matmul_direct.cpp` /
    // `group_matmul_fused_moe.cpp`, the CK dispatcher
    // (`custom_kernel/dispatch.cpp`) and the CK warm-pack
    // (`prepack_custom_kernel.cpp`).  Without this guard, an
    // undersized `ldb[i]` (e.g. a caller setting `ldb` below
    // `transB ? K : N` for a non-firing prepack-extras slot) would
    // make `aocl_reorder_bf16bf16f32of32` read past the row stride
    // and silently alias adjacent rows.  The warm-pack must apply
    // the same minimum-ldb rule the runtime applies, so a tail-slot
    // that the dispatcher would refuse cannot be silently warmed
    // with a corrupted reorder.
    const int min_ldb = transB[i] ? K[i] : N[i];
    if (ldb[i] < min_ldb) {
      ++stats.skipped_invalid;
      continue;
    }

    // Mirror `run_dlp(...)`'s `is_weights_const` gate
    // (aocl_kernel.cpp:1700-1702): when the caller flags an expert
    // as variable-weight, the dispatcher takes the no-cache path
    // for that expert, so warming would be wasted.  An empty
    // `is_weights_const` vector means "treat every entry as const"
    // (legacy behaviour for callers that don't pass the field).
    if (!is_weights_const.empty()
        && i < is_weights_const.size()
        && !is_weights_const[i]) {
      ++stats.skipped_invalid;
      continue;
    }

    // Cache key matches `run_dlp(...)` in aocl_kernel.cpp:1696 so
    // the warmer's MISS populates the same slot a subsequent
    // dispatcher will look up.
    Key_matmul key(transB[i], K[i], N[i], ldb[i], weight[i],
                   static_cast<uint32_t>(matmul_algo_t::aocl_dlp_blocked),
                   /*extra_input_hash=*/0);

    void *reordered_unused = nullptr;
    // `reorderAndCacheWeights<int16_t>` returns `true` unconditionally
    // today (aocl_kernel.cpp:1460-1528); the dead `else ++skipped`
    // branch was dropped — see file-level counter semantics block.
    (void)reorderAndCacheWeights<int16_t>(
        key, weight[i], reordered_unused, K[i], N[i], ldb[i],
        /*order=*/'r',
        /*trans=*/(transB[i] ? 't' : 'n'),
        /*mem_format_b=*/'n',
        aocl_get_reorder_buf_size_bf16bf16f32of32,
        aocl_reorder_bf16bf16f32of32,
        /*weight_cache_type=*/1);
    ++stats.packed_ok;
  }

  return status_t::success;
}

// ─────────────────────────────────────────────────────────────────────
// Per-tile AOCL DLP warm-pack for ALGO 3 strict-stable plan.
//
// Mirrors the runtime decomposition in
// group_matmul_n_tile.cpp::do_tile():
//
//     n_thr_e = participating_n_thr(plan, e, team_size, min_n_tile)
//             = std::max(1, std::min({stable, N[e]/nr_align, team_size}))
//             = std::max(1, std::min(stable, N[e]/nr_align))   [team_size==stable]
//     for tid in [0, n_thr_e):
//         (col_start, col_end) = aligned_n_split(N[e], n_thr_e, tid, nr_align)
//         n_tile = col_end - col_start
//         w_tile = weight[e] + (transB ? col_start*ldb*elem : col_start*elem)
//         key    = (transB, K[e], n_tile, ldb[e], w_tile, aocl_dlp_blocked, 0)
//         reorder + cache
//
// Critically:
//   * `team_size == stable` under the strict-stable plan
//     (group_matmul_n_tile.cpp:1277-1281), so the participating_n_thr
//     `team_size` clamp is a no-op here.
//   * `N[e]/nr_align` clamp protects very narrow experts inside a
//     batch where most experts have larger N — the global narrow-N
//     escape only fires when `stable > max_N/nr_align`, so we still
//     reach this function for a batch with mixed wide/narrow experts
//     and need the per-expert clamp to mirror the runtime.
//
// Per-call invariants (caller responsibility):
//   * `num_threads > 0`, `stable > 0`, `nr_align > 0`.
//   * Caller already verified `stable <= max_N / nr_align`
//     (otherwise the runtime falls back to Sequential and the
//     full-weight warmer above is the right tool).
//
// ─────────────────────────────────────────────────────────────────────
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
    AoclDlpPackProbeStats           &stats) {

  if (total_count <= 0 || num_threads <= 0
      || stable <= 0 || nr_align <= 0) {
    return status_t::success;
  }

  // Production-cache gate — same as the full-weight warmer above.
  // If `ZENDNNL_MATMUL_WEIGHT_CACHE != 1`, `run_dlp(...)` won't
  // consult the LRU cache, so warming would be wasted CPU + memory.
  const int32_t weight_cache_type =
      matmul_config_t::instance().get_weight_cache();
  if (weight_cache_type != 1) {
    return status_t::success;
  }

  // Reachable-entry bound: `[0, bound)` is the largest range every
  // metadata vector covers.  Compute it once up front so the
  // dtype-skip path and the normal path agree on what `total_attempted`
  // and `skipped_invalid` should sum to.  Bounding by only
  // `weight.size()` (or only `total_count`) in the dtype-skip path
  // would overcount skips when K / N / ldb / transB happen to be
  // shorter than weight.  Mirrors the full-weight warmer above.
  const size_t bound = std::min<size_t>({
      static_cast<size_t>(total_count),
      weight.size(),
      K.size(),
      N.size(),
      ldb.size(),
      transB.size()});

  // BF16 only today (the production envelope); other dtypes count
  // every reachable entry as `skipped_invalid` so the PROBE line still
  // surfaces accurate counts.  Note this counts at the EXPERT level
  // (not per tile) since we never enter the per-thread loop on the
  // skip path — `total_attempted` matches the same per-expert
  // granularity to keep the dtype-skip branch comparable to a
  // hypothetical extended (non-BF16) path.  Same rationale as the
  // full-weight warmer.
  if (wei_dtype != data_type_t::bf16) {
    stats.total_attempted += static_cast<int>(bound);
    stats.skipped_invalid += static_cast<int>(bound);
    return status_t::success;
  }

  // Element size for slice pointer arithmetic.  BF16 is the only
  // dtype that reaches the per-tile loop today (early-return above);
  // 2 bytes per element regardless of transB.
  const size_t wei_elem = sizeof(int16_t);

  for (size_t i = 0; i < bound; ++i) {
    if (weight[i] == nullptr || K[i] <= 0 || N[i] <= 0 || ldb[i] <= 0) {
      ++stats.total_attempted;
      ++stats.skipped_invalid;
      continue;
    }

    // Minimum-row-stride gate — same rationale as the full-weight
    // warmer above.  The per-tile path slices the weight pointer by
    // `col_start * ldb * elem` (transB) or `col_start * elem`
    // (non-transposed) and then calls `aocl_reorder_bf16bf16f32of32`
    // with the original `ldb`; an undersized `ldb[i]` would make the
    // reorder read past the row stride and alias neighbouring rows.
    // Counted as `skipped_invalid` at the expert level (no per-tile
    // total_attempted increments here), matching how the basic
    // validity skip above accounts for the same class of failure.
    const int min_ldb = transB[i] ? K[i] : N[i];
    if (ldb[i] < min_ldb) {
      ++stats.total_attempted;
      ++stats.skipped_invalid;
      continue;
    }

    // `is_weights_const` gate: same as full-weight warmer.
    if (!is_weights_const.empty()
        && i < is_weights_const.size()
        && !is_weights_const[i]) {
      ++stats.total_attempted;
      ++stats.skipped_invalid;
      continue;
    }

    // Per-expert participating-thread count, mirroring
    // `do_tile`'s `participating_n_thr` for the strict-stable plan
    // (the planner forces team_size == stable, so the team_size
    // clamp is implicit).  See the function-header comment for the
    // formula.
    const int align_cap =
        std::max(1, N[i] / std::max(1, nr_align));
    const int n_thr_e =
        std::max(1, std::min(stable, align_cap));

    for (int tid = 0; tid < n_thr_e; ++tid) {
      const auto split = aligned_n_split(N[i], n_thr_e, tid, nr_align);
      const int col_start = split.first;
      const int col_end   = split.second;
      const int n_tile    = col_end - col_start;

      ++stats.total_attempted;

      if (n_tile <= 0) {
        ++stats.skipped_invalid;
        continue;
      }

      // Weight pointer offset matches `do_tile()`
      // (group_matmul_n_tile.cpp:444-447):
      //   transB == 't': transposed weight has shape [N, K] in
      //                  row-major, ldb=K → col_start advances
      //                  `col_start * ldb` rows of K elements each.
      //   transB == 'n': weight has shape [K, N], ldb=N → col_start
      //                  advances `col_start` columns within a row.
      const size_t wei_off = transB[i]
          ? static_cast<size_t>(col_start) * ldb[i] * wei_elem
          : static_cast<size_t>(col_start) * wei_elem;
      const void *w_tile =
          static_cast<const char *>(weight[i]) + wei_off;

      // Cache key matches what `run_dlp(...)` builds at runtime
      // (aocl_kernel.cpp:1696) when called from `do_tile()` with
      // sliced (M, n_tile, K, w_tile, ldb).  Symmetric-quant
      // `extra_input_hash` is 0 because the strict-stable plan is
      // BF16/BF16/BF16 (no symmetric-quant code path engaged).
      Key_matmul key(transB[i], K[i], n_tile, ldb[i], w_tile,
                     static_cast<uint32_t>(matmul_algo_t::aocl_dlp_blocked),
                     /*extra_input_hash=*/0);

      void *reordered_unused = nullptr;
      // `reorderAndCacheWeights<int16_t>` returns `true` unconditionally
      // today (aocl_kernel.cpp:1460-1528); the dead `else ++skipped`
      // branch was dropped — see file-level counter semantics block.
      (void)reorderAndCacheWeights<int16_t>(
          key, w_tile, reordered_unused, K[i], n_tile, ldb[i],
          /*order=*/'r',
          /*trans=*/(transB[i] ? 't' : 'n'),
          /*mem_format_b=*/'n',
          aocl_get_reorder_buf_size_bf16bf16f32of32,
          aocl_reorder_bf16bf16f32of32,
          /*weight_cache_type=*/1);
      ++stats.packed_ok;
    }
  }

  return status_t::success;
}

} // namespace aocl_dlp
} // namespace group_matmul_prepack
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
