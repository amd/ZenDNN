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

/// ALGO 2 — M-tile planner.
///
/// =====================================================================
/// PURPOSE
/// =====================================================================
///
/// This header is the **single home for M-tile planning logic**.
///
/// "Planning" = deciding, per-call, **how many threads** each active
/// expert gets, **which physical CCDs** those threads land on, and
/// **which branch** of the M-tile executor runs (round-based,
/// multi-tier hybrid, wide-N fallback, or default Phase-2 single-
/// tier).  It is a pure transformation from `(M, num_threads,
/// active_ops, topology)` to a `tid → (expert, local_tid, team_size)`
/// mapping plus a few diagnostic fields.  Side-effect-free: no env
/// reads, no atomics, no logging — the executor (in
/// `group_matmul_m_tile.cpp`) owns observability and gating I/O.
///
/// **All future M-tile planner optimizations land in this file**:
///
///   * New surplus-thread distribution heuristics (currently "give
///     to the heaviest per-thread slice, capped at M[e] and the
///     expert's CCD capacity").
///   * Alternative wide-N gates (currently `max_M > 1 &&
///     total_need * 2 <= num_threads`).
///   * Smarter CCD rotation (currently `active_pos[e] % num_ccds`)
///     and packing (currently sequential placement w/ spill-to-next).
///   * New planner outputs (e.g. per-thread scratch sizing, NUMA-
///     aware affinity hints, fused-pipeline staging budgets).
///   * Phase-2 cost-model refinements (currently proportional-to-M
///     when over-subscribed).
///
/// The runtime side — `flat_m_tile` (legacy executor) and
/// `flat_m_tile_pipeline_bf16` (vertical-fusion executor) in
/// `group_matmul_m_tile.cpp` — consumes `m_tile_single_tier_plan_t`
/// and walks its `tid_to_*` arrays inside an OMP region.  As long as
/// the planner's output contract is preserved, the runtime side
/// needs no changes when a new heuristic is added here.
///
/// =====================================================================
/// FILE LAYOUT
/// =====================================================================
///
///   Section P.1  M-tile branch path tags (`test_api::m_tile_path_tag`).
///                One named constant per planner-driven executor branch
///                so tests can assert which branch fired without using
///                magic numbers.  The capture machinery
///                (`s_capture_m_tile_path` / `s_last_m_tile_path`) lives
///                in the companion `group_matmul_m_tile.hpp`.
///
///   Section P.2  Planner output type (`m_tile_single_tier_plan_t`).
///                Carries the `tid_to_*` mapping plus the wide-N gate
///                signal and the Phase-1b diagnostic fields.  Sized
///                exactly to `num_threads` after the planner returns
///                (slack threads get `-1` in `tid_to_expert`).
///
///   Section P.3  Planner function (`plan_m_tile_single_tier_assignment`).
///                The Phase 1b + Phase 2 + Phase 3 logic.  Pure
///                computation — no env reads, no capture writes, no
///                logging.  Called AFTER the executor has ruled out the
///                round-based and multi-tier branches (those are decided
///                upstream in `flat_m_tile`).
///
/// =====================================================================
/// DEPENDENCY DIRECTION
/// =====================================================================
///
///   * This header pulls only `<algorithm>`, `<climits>`, `<cstdint>`,
///     `<vector>` and the shared cross-tile common header
///     `../group_matmul_parallel_common.hpp` (the latter only for
///     `static_cast` to host types like `int64_t` — no operator
///     types).  No M-tile executor symbols.
///   * `group_matmul_m_tile.hpp` re-includes this header so any
///     translation unit that includes the public M-tile interface
///     transitively gets the planner output type and path tags.
///   * `group_matmul_m_tile.cpp` includes this header directly
///     (alongside `group_matmul_m_tile.hpp`) and calls the planner
///     from Section A.4 of the executor.
///   * gtests can include this header alone to unit-test the planner
///     in isolation — there is no link-time dependency on the
///     M-tile runtime.

#ifndef ZENDNNL_GROUP_MATMUL_M_TILE_PLANNER_HPP
#define ZENDNNL_GROUP_MATMUL_M_TILE_PLANNER_HPP

#include <algorithm>
#include <climits>
#include <cstdint>
#include <vector>

namespace zendnnl {
namespace lowoha {
namespace matmul {

namespace test_api {

// =====================================================================
// Section P.1 — M-tile branch path tags
// =====================================================================
//
// One named constant per M-tile executor branch.  Set by the executor
// (`flat_m_tile` in `group_matmul_m_tile.cpp` Section B, and
// `flat_m_tile_pipeline_bf16` in Section C) on the very first store
// after it commits to that branch; read by tests via the capture
// machinery in `group_matmul_m_tile.hpp` Section H.2
// (`s_capture_m_tile_path` / `s_last_m_tile_path`).
//
// Constants stay here (not in the main `group_matmul_m_tile.hpp`)
// because every value is conceptually a *planner output*: the
// branches `kRoundBased` and `kMultiTier` are picked upstream in the
// executor's Section B before the single-tier planner even runs;
// `kWideNFallback` is the `wide_n_fallback` signal from the planner
// itself; `kPhase2Single` is the standard single-tier planner output;
// and `kVerticalFusionBF16` / `kVerticalFusionWOQ` / `kVerticalFusionDQINT8`
// are set when the vertical-fusion executor passes its own eligibility
// gate (which uses the same planner).  The three vertical-fusion tags
// differ ONLY in the per-stage dtype work that runs inside the SAME
// per-thread slice plan + scratch sizing math + stage layout
// (W13 → gated-act → W2):
//
//   * `kVerticalFusionBF16`   — bf16 src / bf16 wei / bf16 dst, the
//     phase-1 baseline.  Scratch element size 2 B (bf16).
//   * `kVerticalFusionWOQ`    — bf16 src / bf16 dst with INT4
//     weight-only-quant weights (`s4` or `u4`).  Per-channel
//     wei_scale + `is_weights_const=true` required (the AOCL DLP
//     WOQ fast path uses cached dequant prepack).  Activations
//     stay bf16 across stages (no source-side quant), so the
//     scratch tile element size is the same as BF16 (2 bytes).
//   * `kVerticalFusionDQINT8` — bf16 src / s8 wei / bf16 dst with
//     per-token symmetric dynamic-quant on BOTH Op1 and Op2.  Op1
//     src is bf16→s8 hoisted ONCE per expert pre-OMP (mirrors the
//     N-tile hoist pattern); Op2 src is the post-activation bf16
//     scratch tile re-quantized to s8 per-thread per-slice between
//     Stages 2 and 3 (the NEW Stage 2b).  The bf16 staging scratch
//     element size is still 2 B; Stage 2b adds per-thread
//     `slice_M × K_w2 × 3 + slice_M × 4` bytes of RAII-owned scratch
//     allocated pre-OMP at the dispatcher.
//
// Tests use these constants via `m_tile_path_tag::kPhase2Single`
// etc.; the call sites in the executor use the same constants so
// the test expectations and the production stores stay in sync.
namespace m_tile_path_tag {
inline constexpr int kNone                = -1;  // sentinel — no tag this call
inline constexpr int kRoundBased          = 0;   // active_ops > num_threads
inline constexpr int kMultiTier           = 1;   // multi-tier hybrid engaged
inline constexpr int kWideNFallback       = 2;   // wide-N memory-bound fallback
inline constexpr int kPhase2Single        = 3;   // default M-weighted Phase 2
inline constexpr int kVerticalFusionBF16  = 4;   // vertical fusion engaged
                                                 // (W13→gated-act→W2 per slice,
                                                 // bf16 end-to-end)
inline constexpr int kVerticalFusionWOQ   = 5;   // vertical fusion engaged with
                                                 // WOQ-INT4 (s4/u4) weights;
                                                 // src + dst + scratch are bf16
inline constexpr int kVerticalFusionDQINT8 = 6;  // vertical fusion engaged with
                                                 // per-token symmetric DQ-INT8
                                                 // on both halves (s8 wei +
                                                 // bf16 src/dst; Op1 hoisted
                                                 // pre-OMP, Op2 re-quantized
                                                 // per-slice in Stage 2b)
inline constexpr int kManyExpertsSeqFallback = 7;  // active_ops > num_threads:
                                                   // pure M-tile is infeasible
                                                   // (< 1 thread/expert), so a
                                                   // FORCED ALGO 2 clamps to the
                                                   // sequential full-team path
                                                   // (ALGO 1 equivalent) + WARN.
                                                   // AUTO never reaches this (it
                                                   // routes active>threads to
                                                   // ALGO 5); only an explicit
                                                   // ZENDNNL_GRP_MATMUL_ALGO=2
                                                   // force lands here.
}  // namespace m_tile_path_tag

}  // namespace test_api

// =====================================================================
// Section P.2 — Planner output type
// =====================================================================
//
// Phase 1b/2/3 single-tier planner output.
//
// Returned by `plan_m_tile_single_tier_assignment()`.  `flat_m_tile`
// (now a pure M-tile executor) reaches this helper after ruling out
// the many-experts clamp and multi-tier branches and runs the
// Phase-2 single-tier CCD-stripe it produces.  The helper emits both
// the Phase-1b stats (`total_need` / `max_M`, used by the advisory
// wide-N gate and any caller diagnostics) and the Phase-3 tid mapping
// in one struct.  Vertical-fusion (phase 1) reuses the same helper so
// the pipeline executor's slice math is bit-identical to the legacy
// executor's for every shape that reaches Phase 2.
//
// All `tid_to_*` arrays are sized exactly to `num_threads` after the
// planner returns; threads with `tid_to_expert[tid] == -1` sit idle
// on the join barrier (the planner intentionally leaves slack
// threads when `active_ops < num_threads` and the surplus loop fully
// caps every expert at `M[e]`).
struct m_tile_single_tier_plan_t {
  // Phase 3 CCD-striped tid → (expert, local_tid, team_size) map.
  std::vector<int> tid_to_expert;
  std::vector<int> tid_to_local;
  std::vector<int> tid_to_team;

  // Wide-N memory-bound signal.  True when the Phase-1b sum of
  // per-expert ideal thread counts is at most half of `num_threads`
  // AND `max_M > 1`.
  //
  // By default the planner does NOT short-circuit on this signal: Phase
  // 2/3 still run so the `tid_to_*` arrays are fully populated (a forced
  // ALGO 2 on a wide-N shape needs a complete single-tier mapping —
  // `flat_m_tile` is pure M-tile, ignores this flag, and `auto_select_algo`
  // routes the AUTO wide-N regime to ALGO 1 instead).  A caller that passes
  // `short_circuit_on_wide_n=true` (today only the vertical-fusion pipeline
  // `flat_m_tile_pipeline_bf16`, which treats the signal as an eligibility
  // failure and bails to the legacy two-pass) gets an EARLY return with
  // `wide_n_fallback=true` and EMPTY `tid_to_*` arrays — skipping the
  // throwaway Phase-2/3 work it would never read.
  bool wide_n_fallback = false;

  // Phase 1b raw outputs.  Exposed for the wide-N gate (which uses
  // both) and for any downstream perf logging the caller wants to
  // attach.
  int total_need = 0;
  int max_M      = 0;
};

// =====================================================================
// Section P.3 — Planner function
// =====================================================================
//
// Phase 1b + Phase 2 + Phase 3 of the single-tier M-tile thread plan.
// Pure refactor of the inline planner that lived in `flat_m_tile`
// pre-vertical-fusion; behaviour-identical on every shape.  Called
// AFTER the caller has ruled out the round-based and multi-tier
// branches (those execute inline in `flat_m_tile` and return before
// reaching this helper).
//
// When the wide-N gate matches the helper sets `plan.wide_n_fallback =
// true`.  By default it then CONTINUES through Phase 2/3 and fully
// populates the `tid_to_*` mapping, because the legacy `flat_m_tile`
// executor ignores the flag and runs the single-tier plan even on wide-N
// shapes (ALGO 2 is pure M-tile; `auto_select_algo` routes the AUTO wide-N
// regime to ALGO 1 before it ever reaches here).  A caller that will bail
// on the signal and never read the mapping passes
// `short_circuit_on_wide_n=true` (today only the vertical-fusion phase-1
// dispatcher) to get an early return with empty `tid_to_*` arrays and skip
// the throwaway Phase-2 surplus loop + Phase-3 placement.
//
// The helper is intentionally side-effect-free (no env reads, no
// capture-tag writes, no logging) so the caller controls
// observability.  All env-tunable constants are passed in by the
// caller as a single integer (`kSliceTarget`).
//
// **Future planner heuristics replace or augment the body of this
// function** — keep the input/output contract stable and the
// executor side keeps working unchanged.
inline m_tile_single_tier_plan_t plan_m_tile_single_tier_assignment(
    const std::vector<int> &M,
    const std::vector<int> &active_pos,
    int num_ops, int num_threads,
    int active_ops,
    int kSliceTarget,
    int cores_per_ccd, int num_ccds,
    bool short_circuit_on_wide_n = false) {

  m_tile_single_tier_plan_t plan;

  // ── Phase 1b: initial t_assign based on target slice size ──
  std::vector<int> t_assign(num_ops, 0);
  int total_need = 0;
  int max_M = 0;
  for (int i = 0; i < num_ops; ++i) {
    if (M[i] <= 0) continue;
    // Overflow-safe ceil-div: the F8 env knob
    // `ZENDNNL_GRP_MATMUL_M_TILE_SLICE_TARGET` accepts any positive
    // int, so `M[i] + kSliceTarget - 1` can overflow signed `int`
    // on pathological inputs.  The ceil-div result is bounded
    // by `M[i]` so the int cast after the ≥ 1 clamp is always safe.
    t_assign[i] = std::min(M[i],
        static_cast<int>(std::max<int64_t>(1,
            (static_cast<int64_t>(M[i]) + kSliceTarget - 1)
                / kSliceTarget)));
    total_need += t_assign[i];
    if (M[i] > max_M) max_M = M[i];
  }
  plan.total_need = total_need;
  plan.max_M      = max_M;

  // ── Wide-N memory-bound fallback signal ──
  //
  // Same gate as the original inline branch: `max_M > 1` excludes
  // pure-decode workloads (every expert has M=1, the surplus loop
  // is a structural no-op there and the CCD-stripe owns each expert
  // on its own CCD with team=1).  `total_need * 2 ≤ num_threads`
  // proxies "M is shallow enough that the M-tile slice plan would
  // shrink each thread's slice below kSliceTarget".  See the
  // doc-block on the inline branch in `flat_m_tile` (pre-refactor)
  // for the full rationale and empirical engagement table.
  //
  // `total_need * 2` promoted to int64_t to defuse hypothetical
  // signed-int overflow (practical bound: total_need ≪ INT_MAX).
  // Wide-N signal.  Consumed ONLY by the vertical-fusion pipeline
  // (`flat_m_tile_pipeline_bf16`), which bails to the legacy two-pass when
  // set.  `flat_m_tile` no longer branches on it (ALGO 2 is pure M-tile;
  // `auto_select_algo` routes the AUTO wide-N regime to ALGO 1).
  //
  // Phase 2/3 below run by default so that a FORCED ALGO 2 on a wide-N shape
  // (`flat_m_tile`, which IGNORES the flag) still gets a complete single-tier
  // tid→expert mapping.  A caller that bails on the wide-N signal and never
  // reads the mapping (today only the vertical-fusion pipeline) passes
  // `short_circuit_on_wide_n=true` to skip the throwaway Phase-2 surplus loop
  // + Phase-3 placement; it gets `wide_n_fallback=true` with empty tid_to_*
  // arrays, which is exactly what it needs to return false.
  if (max_M > 1
      && static_cast<int64_t>(total_need) * 2 <= num_threads) {
    plan.wide_n_fallback = true;
    if (short_circuit_on_wide_n) return plan;
  }

  // ── CCD capacity / cap_at_ccd computation ──
  auto ccd_capacity = [&](int c) -> int {
    const int base = c * cores_per_ccd;
    return std::max(0, std::min(cores_per_ccd, num_threads - base));
  };

  // CCD-aware cap: when active_ops <= num_ccds AND every active expert
  // could fit in a single CCD (active_ops * cores_per_ccd >=
  // num_threads), cap each expert at its assigned CCD's capacity.
  // Prevents an expert's team from spanning CCD boundaries (which
  // would cause L3 contention with a neighbor expert's weight).
  // F2 — predicate uses `active_ops` (not raw `num_ops`); see the
  // doc-block on the call site in `flat_m_tile`.
  const bool cap_at_ccd = (active_ops <= num_ccds)
      && (static_cast<int64_t>(active_ops) * cores_per_ccd >= num_threads);

  // ── Phase 2: fit t_assign to num_threads ──
  if (total_need <= num_threads) {
    // Distribute surplus threads to experts with the heaviest per-thread
    // slice.  Capped at M[e] and optionally at the expert's CCD capacity.
    int surplus = num_threads - total_need;
    while (surplus > 0) {
      int best = -1;
      int best_slice = 0;
      for (int i = 0; i < num_ops; ++i) {
        if (t_assign[i] <= 0 || t_assign[i] >= M[i]) continue;
        if (cap_at_ccd) {
          // F2 — compact CCD stripe over active experts only.
          // `active_pos[i] >= 0` is guaranteed here because
          // `t_assign[i] > 0` ⇒ M[i] > 0 ⇒ `active_pos[i] != -1`.
          const int my_ccd = active_pos[i] % num_ccds;
          if (t_assign[i] >= ccd_capacity(my_ccd)) continue;
        }
        int cur_slice = (M[i] + t_assign[i] - 1) / t_assign[i];
        if (cur_slice > best_slice) {
          best_slice = cur_slice;
          best = i;
        }
      }
      if (best < 0) break;
      ++t_assign[best];
      --surplus;
    }
  } else {
    // Scale down proportionally to M[e], floor at 1 per active expert.
    int64_t total_M = 0;
    for (int i = 0; i < num_ops; ++i) total_M += M[i];
    if (total_M <= 0) {
      // Mirrors the legacy `return;` — the caller's OMP region will
      // observe `tid_to_expert[*] == -1` and do nothing.
      plan.tid_to_expert.assign(num_threads, -1);
      plan.tid_to_local.assign(num_threads, -1);
      plan.tid_to_team.assign(num_threads, 0);
      return plan;
    }

    int assigned = 0;
    for (int i = 0; i < num_ops; ++i) {
      if (M[i] <= 0) { t_assign[i] = 0; continue; }
      t_assign[i] = std::max(1, static_cast<int>(
          static_cast<int64_t>(num_threads) * M[i] / total_M));
      assigned += t_assign[i];
    }
    while (assigned < num_threads) {
      int best = -1, best_slice = 0;
      for (int i = 0; i < num_ops; ++i) {
        if (t_assign[i] <= 0 || t_assign[i] >= M[i]) continue;
        if (cap_at_ccd) {
          const int my_ccd = active_pos[i] % num_ccds;
          if (t_assign[i] >= ccd_capacity(my_ccd)) continue;
        }
        int cur_slice = (M[i] + t_assign[i] - 1) / t_assign[i];
        if (cur_slice > best_slice) { best_slice = cur_slice; best = i; }
      }
      if (best < 0) break;
      ++t_assign[best];
      ++assigned;
    }
    while (assigned > num_threads) {
      int best = -1, least_slice = INT_MAX;
      for (int i = 0; i < num_ops; ++i) {
        if (t_assign[i] <= 1) continue;
        int cur_slice = (M[i] + t_assign[i] - 1) / t_assign[i];
        if (cur_slice < least_slice) { least_slice = cur_slice; best = i; }
      }
      if (best < 0) break;
      --t_assign[best];
      --assigned;
    }
  }

  // ── Phase 3: CCD-striped thread→expert mapping ──
  //
  // Universal layout: works for any num_threads and any t_assign.
  // Each expert's threads are placed sequentially on a starting CCD
  // (rotates per expert for load balance), packing up to cores_per_ccd
  // threads per CCD before spilling to the next.  See the doc-block
  // on the legacy inline call site for the full mapping example
  // catalogue.
  plan.tid_to_expert.assign(num_threads, -1);
  plan.tid_to_local.assign(num_threads, -1);
  plan.tid_to_team.assign(num_threads, 0);
  std::vector<int> ccd_used(num_ccds, 0);

  // F2 — Starting CCD is deterministic per *active position*
  // (`active_pos[e] % num_ccds`), NOT raw expert index.
  for (int e = 0; e < num_ops; ++e) {
    if (t_assign[e] <= 0) continue;
    const int t = t_assign[e];
    int placed = 0;
    int c = active_pos[e] % num_ccds;

    while (placed < t) {
      int tries = 0;
      while (ccd_used[c] >= ccd_capacity(c) && tries < num_ccds) {
        c = (c + 1) % num_ccds;
        ++tries;
      }
      if (tries >= num_ccds) break;  // no capacity (shouldn't happen)

      const int cap = ccd_capacity(c) - ccd_used[c];
      const int can_place = std::min(cap, t - placed);
      for (int k = 0; k < can_place; ++k) {
        const int local = ccd_used[c] + k;
        const int tid = c * cores_per_ccd + local;
        if (tid < num_threads) {
          plan.tid_to_expert[tid] = e;
          plan.tid_to_local[tid] = placed + k;
          plan.tid_to_team[tid] = t;
        }
      }
      ccd_used[c] += can_place;
      placed += can_place;
      if (placed < t) c = (c + 1) % num_ccds;
    }
  }

  return plan;
}

}  // namespace matmul
}  // namespace lowoha
}  // namespace zendnnl

#endif  // ZENDNNL_GROUP_MATMUL_M_TILE_PLANNER_HPP
