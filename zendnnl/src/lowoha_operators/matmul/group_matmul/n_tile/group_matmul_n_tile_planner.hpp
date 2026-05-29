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

/// ALGO 3 — N-tile planner.
///
/// =====================================================================
/// PURPOSE
/// =====================================================================
///
/// This header is the **single home for N-tile planning data types**
/// — strategy enum, topology summary, the full per-call `GroupNTilePlan`,
/// the multi-round cost-model candidate struct, and the round-pick
/// enum.  It is the symmetric counterpart of
/// `../m_tile/group_matmul_m_tile_planner.hpp`.
///
/// "Planning" for ALGO 3 = picking, per-call, **which execution
/// strategy** runs (Sequential / DecodeD / FewExperts / ManyExperts),
/// **how many threads per expert** in each round, **the round batch
/// size**, **the expert-ordering permutation**, and a small bag of
/// strategy-specific knobs (tight-fused-epilogue switch, AOCL strict-
/// stable per-expert overrides, Phase B remainder-distribute flags).
/// The output type `GroupNTilePlan` is the planner-to-executor
/// contract — executors in `group_matmul_n_tile.cpp` consume the
/// plan struct without ever re-deriving heuristic state.
///
/// **All future N-tile planner optimizations land in this file**:
///
///   * New fields on `GroupNTilePlan` (e.g. NUMA-aware scheduling,
///     per-round dynamic-tile reshuffles, alternative remainder
///     distributions).
///   * New strategy variants (extend `GroupNTileStrategy` and the
///     round-pick contract).
///   * New cost-model parameters on `RoundCandidates` (the picker in
///     `n_tile.cpp::pick_round_strategy` will follow).
///   * New test-only diagnostic snapshots beyond `PhaseBSnapshot`
///     (use the same one-shot capture pattern).
///
/// The runtime side — the executors `flat_n_tile`,
/// `execute_sequential` / `execute_decode_d` / `execute_rounds` and
/// the lower-level `GroupNTileContext::do_tile` — consumes the plan
/// fields directly.  When a new planner output is added here, only
/// the consumers in `group_matmul_n_tile.cpp` need to learn about it
/// — no caller-side change to `flat_n_tile`'s public signature.
///
/// =====================================================================
/// FILE LAYOUT
/// =====================================================================
///
///   Section P.1  Strategy enum (`GroupNTileStrategy`).  One value per
///                top-level executor branch the planner can select.
///
///   Section P.2  Topology summary (`GroupNTileTopology`).  Pre-digested
///                inputs (shape stats + cluster topology) consumed by
///                the planner's branch heuristics.
///
///   Section P.3  Per-call plan (`GroupNTilePlan`).  The full planner
///                output: strategy + algo + threading parameters +
///                expert-order permutation + per-expert thread-count
///                array + miscellaneous strategy-specific flags.
///                This is the only struct the executors read.
///
///   Section P.4  Round-scheduler types (`RoundCandidates`, `RoundPick`).
///                Used by `pick_round_strategy()` and `apply_round_pick()`
///                in `group_matmul_n_tile.cpp` to pick between Single /
///                Multi / Balanced round shapes for the ManyExperts
///                executor.
///
///   Section P.5  Test-only Phase-B snapshot (`test_api::PhaseBSnapshot`
///                + `s_capture_phase_b` / `s_last_phase_b_snapshot`).
///                One-shot capture of the planner's per-expert thread
///                assignment for direct end-to-end verification by
///                `test_algos.cpp`.
///
/// =====================================================================
/// DEPENDENCY DIRECTION
/// =====================================================================
///
///   * This header pulls only `<array>`, `<atomic>`, `<cstdint>`,
///     `<vector>` and `../group_matmul_parallel_common.hpp` (for
///     `kNTilePlanMaxExperts` and `matmul_algo_t` via the inline
///     `using namespace zendnnl::ops;` in the common header).
///   * `group_matmul_n_tile.hpp` re-includes this header so any
///     translation unit that includes the public N-tile interface
///     transitively gets the planner output types.
///   * `group_matmul_n_tile.cpp` includes this header directly
///     (alongside `group_matmul_n_tile.hpp`) and the planner
///     function `plan_group_n_tile` (currently in the .cpp's
///     anonymous namespace) is the canonical producer of
///     `GroupNTilePlan`.  Future work can incrementally migrate
///     the planner function itself and its helpers (`summarise_topology`,
///     `build_round_candidates`, `pick_round_strategy`,
///     `apply_round_pick`, `apply_adaptive_tiers`, etc.) into this
///     header as inline functions — the data-type contracts they
///     produce are already here.
///   * gtests can include this header alone to assert on planner
///     output structures (currently via `PhaseBSnapshot`).

#ifndef ZENDNNL_GROUP_MATMUL_N_TILE_PLANNER_HPP
#define ZENDNNL_GROUP_MATMUL_N_TILE_PLANNER_HPP

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

#include "../group_matmul_parallel_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

// Bring `matmul_algo_t` into scope so the `algo` field on
// `GroupNTilePlan` (Section P.3) reads as `matmul_algo_t` — matches
// the unqualified usage in the rest of the group_matmul code (the
// using-decl is also brought in from `group_matmul_parallel_common.hpp`
// via its inline `using namespace zendnnl::ops;`).
using zendnnl::ops::matmul_algo_t;

// =====================================================================
// Section P.0 — N-tile shared constants
// =====================================================================
//
// Constants specific to the ALGO 3 (N-tile) planner / executor.
// Previously lived in `group_matmul_parallel_common.hpp`; moved here
// (PR follow-up) so every N-tile-specific knob sits next to the
// N-tile code it parameterises.  These MUST live in the planner
// header — not in `group_matmul_n_tile.hpp` — because
// `GroupNTilePlan` (Section P.3 below) sizes stack arrays from
// `kNTilePlanMaxExperts` at struct-definition time, and the planner
// header is the EARLIEST file in the include DAG.
//
// Consumers (`group_matmul_n_tile.hpp` and everyone who re-includes
// it: executor, dispatcher, gtests, fused-MoE) see these symbols
// transparently because `group_matmul_n_tile.hpp` includes this
// header on line ~120 of its include block.

/// Default minimum per-thread N width for the decode-phase N-tile
/// path (max_M ≤ kDecodeMaxM=32).  Used:
///   * by `effective_decode_n_tile()` as the default outer N-tile when
///     `ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_N_TILE` is unset / invalid,
///   * by the auto-selector's decode-class N-tile sizing,
///   * by the custom-kernel dispatch as the gate for engaging the
///     fused tile path (see `custom_kernel/dispatch.cpp`).
/// 256 is the AVX-512 BF16 sweet spot for the inner GEMM kernel; tune
/// only with a measured perf justification.
inline constexpr int kDecodeNTile = 256;

/// Maximum number of experts the ALGO 3 (N-tile) planner can
/// represent.  Mirrored on `GroupNTilePlan::kMaxExperts` so the
/// planner's stack-allocated fixed-size arrays (`expert_order`,
/// `stable_n_thr_per_expert`) stay heap-free on the hot path.  Also
/// used to size the heap-free temporaries in the expert-ordering
/// helpers (`fill_ntile_expert_order`) and the round-info stack
/// array in `execute_rounds` (group_matmul_n_tile.cpp).
///
/// Auto-select uses this constant in **rule 0** — the top-level
/// capacity carve-out applied BEFORE the three policy rules.  Any
/// shape with `num_ops > kNTilePlanMaxExperts` (regardless of how it
/// would otherwise be routed by rules 1-3) goes to ALGO 5 (per-expert
/// parallel) because the N-tile planner's R3 gate would silently
/// fall back to its Sequential strategy (one expert at a time, full
/// team each).  Sequential is materially slower than ALGO 5 for
/// many-experts decode-class shapes — ALGO 5 fans `num_ops` over the
/// OMP team and lets each thread own a slice of experts serially,
/// with no fixed-size lookup arrays of its own.
///
/// The carve-out catches both rule-1-territory shapes
/// (`num_ops >= num_threads`, e.g., 300 experts on 128 threads) and
/// the rare rule-3-decode-territory shape
/// (`kNTilePlanMaxExperts < num_ops < num_threads`, e.g., 300
/// experts on a 512-thread host).
///
/// Bump only if `GroupNTilePlan` switches to heap-allocated arrays
/// (or callers start shipping > 256-expert deployments where the
/// N-tile planner outperforms ALGO 5 — neither situation exists
/// today).
inline constexpr int kNTilePlanMaxExperts = 256;

// =====================================================================
// Section P.1 — Strategy enum
// =====================================================================
//
// Execution patterns the planner can pick.  Only DecodeD, FewExperts,
// and ManyExperts run the actual N-tile per-tile dispatch (and route
// through the BF16 custom kernel when its gate accepts the call) —
// those are the strategies callers running ALGO 3 are typically here
// for.  Sequential is the fallback for shapes where the N-tile path
// is either inapplicable (structural) or not the right choice (perf):
//   * `!ntile_viable`  — N too narrow to split usefully.
//   * R3 capacity      — num_ops > kNTilePlanMaxExperts; safety guard
//                        on the fixed-size per-expert arrays.
//   * F3 alignment     — AOCL strict-stable narrow-N escape.
//   * AUTO-MIRROR      — `auto_select_algo` (env=0) would have picked
//                        ALGO 1 for this shape.  Forced env=3 mirrors
//                        that decision so the strategy choice matches
//                        what auto-pick does, with the gemm_mode
//                        label distinguishing the path for telemetry
//                        (`flat_n_tile_sequential` vs
//                        `sequential_experts`).  Skipped under
//                        `ZENDNNL_GRP_MATMUL_N_TILE_STRATEGY=1` or
//                        `=2`; `=2` is the production default, so
//                        the auto-mirror gate is normally inactive
//                        and ALGO 3 runs the Rounds path on every
//                        shape that survives the structural gates.
//
// The auto-mirror replays auto-select's three-rule tree faithfully,
// so Rule 1 (`num_ops ≥ num_threads → ALGO 3`) protects Qwen-class
// prompt — those calls run the actual ntile path even though
// `max_M > kDecodeMaxM` would otherwise mark them as prompt.
//
// `ZENDNNL_GRP_MATMUL_N_TILE_STRATEGY` (auto / decode / rounds) drives
// three things:
//   (1) whether the auto-mirror gate fires (only under value 0);
//   (2) whether DecodeD's perf-eligibility heuristic is consulted
//       (only under value 0) or bypassed in favour of a structural-
//       floor-only force path (under value 1);
//   (3) whether DecodeD is attempted at all (under values 0 and 1)
//       or skipped entirely (under value 2 → straight to Rounds).
// Structural gates (`!ntile_viable`, R3, F3) always fire regardless
// of the knob.  See `get_grp_n_tile_strategy()` in
// `group_matmul_n_tile.hpp` for the full env-knob contract.
enum class GroupNTileStrategy {
  // (F)  Sequential fallback.  Runs experts serially with the full
  //      thread team per kernel via `execute_expert_slice` — the
  //      same mechanics ALGO 1 uses — plus a tight-arena fused-
  //      swiglu OOP branch for fused-MoE callers with `ldc < N`.
  //      Bypasses the per-tile dispatch and the BF16 custom kernel
  //      entirely.  Reached when the N-tile path is inapplicable
  //      (`!ntile_viable`, R3 capacity, F3 alignment) or when the
  //      auto-selector would have picked ALGO 1 (AUTO-MIRROR) — see
  //      the enum's doc-block above for the four reasons.
  Sequential,

  // (D)  Decode parallel — small max_M, balanced M, num_ops ≤ num_ccds.
  //      One flat OMP region with `num_ops × thr_per_expert` threads;
  //      each expert gets an equal CCD-sized team and all experts
  //      run concurrently with no internal barriers.  Skipped when
  //      the env knob is set to "rounds".
  DecodeD,

  // (A)  Few experts (num_ops ≤ num_ccds, non-decode):
  //      L3-aware batches, proportional thr_per_expert per round.
  //      Routes through `execute_rounds`.
  FewExperts,

  // (B)  Many experts (num_ops > num_ccds):
  //      L3-aware barrier-synchronized rounds, fixed n_thr/expert.
  //      Routes through `execute_rounds`.
  ManyExperts,
};

// =====================================================================
// Section P.2 — Topology summary
// =====================================================================
//
// Inputs / topology summary used by the planner — only what's needed
// to decide; no expert vectors are inspected at strategy level.
struct GroupNTileTopology {
  int num_ops;
  int num_threads;
  int ccd_size;          // = min(8, num_threads)
  int num_ccds;          // ceil(num_threads / ccd_size)
  int max_M;
  int max_N;
  int max_K;
  int min_M_active;      // smallest positive M, or max_M if all empty
  size_t wei_elem;       // weight bytes per element
  size_t wei_per_expert; // = max_N * max_K * wei_elem, precomputed once
                         // so compute_l3_batch / compute_target_batch
                         // (round batch sizing under the L3 budget)
                         // don't each re-derive it from the three
                         // inputs.
};

// =====================================================================
// Section P.3 — Per-call plan
// =====================================================================
//
// Knobs the strategy executors consume.  Most fields are filled only
// for the strategy they belong to; the rest stay zero / default.
struct GroupNTilePlan {
  // Always-valid fields
  GroupNTileStrategy strategy = GroupNTileStrategy::Sequential;
  matmul_algo_t algo = matmul_algo_t::aocl_dlp_blocked;
  int  num_threads = 1;
  int  nr_align = 1;
  bool fused_epilogue = false;

  // Per-thread N-slice floor (== `min_n_tile` argument of
  // ctx.do_tile).  Set for DecodeD / FewExperts / ManyExperts.
  int min_n_tile = 1;

  // (D) DecodeD parameters
  int decode_thr_per_expert = 0;
  int decode_total_threads = 0;

  // (A) FewExperts / (B) ManyExperts shared "rounds" parameters.
  // Threads per expert in a round is `n_thr_fixed` if non-zero
  // (ManyExperts), else min(num_threads / round_size, max_n_thr)
  // (FewExperts — proportional to round_size, capped by N-tile count).
  int batch_size = 0;
  int n_thr_fixed = 0;
  int max_n_thr = 0;

  // Optional permutation of expert indices used by FewExperts /
  // ManyExperts when assigning experts to rounds.  When
  // `expert_order_size == 0` callers use input order; when > 0 the
  // first `expert_order_size` slots of `expert_order` hold the
  // sorted permutation.  Populated by `fill_sorted_expert_order()`
  // when the planner enables M-descending sort: round time is
  // dominated by max(M) within the round, so sorting puts the
  // heaviest experts in the first round (where they would dominate
  // anyway) and pushes the lightest into the last round, minimising
  // sum_round(max_M).
  //
  // Stack-allocated to keep the hot path heap-free.  `kMaxExperts`
  // sets the upper bound on the in-place expert sort; workloads with
  // more experts skip the sort (perf optimisation, not a correctness
  // requirement) and fall through to input-order assignment.
  //
  // Value lives in `group_matmul_parallel_common.hpp` as
  // `kNTilePlanMaxExperts` so the auto-selector can route huge-
  // experts workloads to ALGO 5 without pulling this header.  Keep
  // the two in sync — the alias here is the planner-facing name.
  static constexpr int kMaxExperts = kNTilePlanMaxExperts;
  std::array<int, kMaxExperts> expert_order{};
  int expert_order_size = 0;

  // ── Per-expert thread count override (dual-use) ────────────────────
  // The array is populated by ONE of two disjoint producers, and
  // consumed by `execute_rounds` (via `per_expert_remainder` below)
  // and `participating_n_thr` (via the strict-stable safety branch).
  //
  // Producer 1 — AOCL strict-stable plan (`!use_custom &&
  //              get_grp_matmul_aocl_stable_ntile()`)
  //   `plan_group_n_tile` writes
  //     stable_n_thr_per_expert[e] = aocl_stable_n_thr(num_threads)
  //   for every active expert.  Implementation in
  //   `group_matmul_parallel_common.hpp::aocl_stable_n_thr` ignores
  //   its `N` parameter and returns a value derived solely from
  //   `num_threads` and `target_slots` — num_ops-, shape-, and
  //   phase-INDEPENDENT.  For a fixed model + OMP team size, the
  //   chosen value is invariant across calls regardless of per-call
  //   gating, strategy, batch size, or phase (prompt vs decode) —
  //   exactly what the AOCL reorder cache key (col_start, n_tile)
  //   needs to stay byte-identical across calls.  All entries are
  //   uniform (== stable); `per_expert_remainder` is left FALSE so
  //   `execute_rounds` takes its O(1) `tid / tpe` mapping.
  //
  // Producer 2 — CK Single-round remainder-distribute (Phase B /
  //              T4-simple), populated only when
  //              `use_custom == true` (gate enforced by
  //              `apply_round_pick`).
  //   `apply_round_pick` writes NON-uniform per-expert values:
  //   among experts whose per-expert N capacity can absorb one more
  //   thread (`N[e] / ab_min_tile >= base + 1`, i.e. the per-expert
  //   eligibility filter), the M-heaviest get `base + 1`; the
  //   remaining experts (eligible-but-not-heaviest, plus all
  //   ineligible experts) stay at `base`.  This lets
  //   `execute_rounds`' prefix-sum mapping saturate the full thread
  //   team instead of leaving `num_threads % num_ops` slots idle on
  //   uniform-N workloads, while preventing the surplus from landing
  //   on threads that cannot use it on non-uniform-N workloads.
  //   `per_expert_remainder` is set TRUE so the executor switches
  //   to the per-round prefix-sum thread→expert mapping.  Safe for
  //   CK because the pack cache is shape-keyed (full-N pack per
  //   expert), so per-expert thread variation does not destabilise
  //   the cache key.
  //
  // Consumer A — `execute_rounds`
  //   Reads `plan.per_expert_remainder` (not the array directly) to
  //   decide between uniform O(1) mapping and per-round prefix-sum
  //   scan.  Producer-2's flag triggers the scan; Producer-1 leaves
  //   the flag false and keeps the uniform fast path.
  //
  // Consumer B — `participating_n_thr`
  //   On the non-custom path (`!use_custom`), the safety-clamp
  //   branch fires when `stable_n_thr_per_expert[e] > 0` and returns
  //   `min({stable_n_thr_per_expert[e], N[e]/nr_align, team_size})`.
  //
  //   Producer 1 uses `plan.nr_align` (backend-determined and
  //   phase-independent) rather than `min_n_tile` (planner-derived
  //   from `max_M`, phase-dependent) for the inner clamp.  That
  //   keeps the AOCL reorder cache inputs free of phase-specific
  //   planner state.  The planner's narrow-N escape uses
  //   `topo.max_N` (not per-expert N), so for non-uniform-N callers
  //   a small-N expert may still hit `N[e] / nr_align < stable` and
  //   get clamped down — the cache key for THAT expert effectively
  //   becomes `(col_start, n_tile)` at the clamped tile count.  For
  //   uniform-N MoE workloads (the typical case) the clamp is a
  //   no-op and the cache key is byte-identical across calls; for
  //   non-uniform-N callers the byte-identical invariant holds only
  //   for the experts at full `stable` capacity.
  //
  //   On the custom path (`use_custom`), this branch is unreachable
  //   regardless of the array state — Producer 2's values flow into
  //   `participating_n_thr` via the dynamic-tile branch with
  //   `team_size = stable_n_thr_per_expert[e]` set by
  //   `execute_rounds`' prefix-sum scan.
  //
  // Sentinel: an all-zero array means "no producer ran" — both
  // consumers take their respective default fast paths.
  std::array<int16_t, kMaxExperts> stable_n_thr_per_expert{};

  // When the env `ZENDNNL_GRP_MATMUL_N_ORDER` is 0 (auto), the picker
  // resolves a concrete sub-mode and stashes it here for APILOG
  // transparency.  Left at -1 when the env was explicitly set
  // (no auto-resolution performed).
  int auto_resolved_order = -1;

  // Tight-dst fused-epilogue switch.  Set to true at `flat_n_tile`
  // entry when `fused_epilogue=swiglu` AND the caller's dst is a
  // tight [M, I]-layout buffer (ldc[0] < N[0]).  Triggers the
  // per-thread-scratch + out-of-place swiglu flow inside `do_tile`;
  // when false, the classic matmul-then-in-place-compact wide flow
  // runs.
  //
  // Contract when true: all experts have `ldc[e] == N[e] / 2`
  // (the fused-MoE caller allocates a uniform-stride arena).
  // `execute_*` skip the barrier + `apply_swiglu_oai()` pass because
  // the activation is already fused into `do_tile`.
  bool tight_fused_epilogue = false;

  // True iff the Phase B / T4-simple remainder-distribute populated
  // `stable_n_thr_per_expert[]` with NON-uniform per-expert values on
  // the CK Single-round path (the M-heaviest *eligible* experts —
  // those whose per-expert `N / ab_min_tile >= base + 1` — get
  // `base + 1`, the rest get `base`).
  //
  // Source of truth for `execute_rounds`'s thread→expert mapping
  // decision: `execute_rounds` checks THIS flag (and only this
  // flag) to switch between the per-round prefix-sum scan
  // (`per_expert_remainder == true`) and the O(1)
  // `tid / thr_per_expert` div/mod (`per_expert_remainder == false`).
  // An earlier sentinel approach inferred "non-uniform allocation"
  // from `stable_n_thr_per_expert[0] > 0`, which is incorrect because
  // the strict-stable AOCL plan also populates that array (uniformly,
  // with `stable`).  Do NOT add new code that infers the mode from
  // the array contents — extend the explicit flag instead.
  //
  // Why not infer from the array: the strict-stable AOCL plan also
  // populates `stable_n_thr_per_expert[]` (uniformly, with `stable`)
  // so `participating_n_thr` can take its safety-clamp branch.  The
  // executor must NOT take the prefix-sum scan on that uniform plan
  // — it would walk `O(round_size)` per thread per round instead of
  // the O(1) div/mod path, all while every per-expert value equals
  // `n_thr_fixed`.  This flag distinguishes the two cases by intent
  // rather than by value inspection.
  //
  // Set true only by `apply_round_pick` when ALL of:
  //   * pick == Single
  //   * use_custom (CK path) — uniform safety re-clamps require this
  //   * remainder > 0 && remainder < num_ops
  //   * (base + 1) <= min(ccd_size, max_tiles)        — outer cap
  //   * eligible_count > 0 after the per-expert
  //     N-capacity filter (`N[e] / ab_min_tile >= base + 1`) — at
  //     least one expert must be able to absorb the extra thread.
  //     If every M-heaviest candidate fails the filter the planner
  //     falls back to the uniform-`base` allocation; the flag stays
  //     false and the executor takes the O(1) `tid / thr_per_expert`
  //     mapping (bit-for-bit identical to pre-Phase-B behaviour).
  bool per_expert_remainder = false;
};

// =====================================================================
// Section P.4 — Round-scheduler cost-model types
// =====================================================================
//
// Cost-model candidate parameters for the (B) ManyExperts round
// scheduler.  Three competing shapes — Single / Multi / Balanced —
// share derivations (max_tiles, capped_batch) and each carry their
// own n_thr / batch / round count + an estimated wall time the
// picker compares.  Filled by `build_round_candidates()` and
// consumed by `pick_round_strategy()` + `apply_round_pick()`.
struct RoundCandidates {
  // Shared derivations
  int max_tiles = 0;
  int capped_batch = 0;

  // Per-candidate parameters
  int    n_thr_single     = 0;
  bool   single_eligible  = false;
  double wall_single      = 0.0;

  int    n_thr_multi      = 0;
  int    batch_multi      = 0;
  int    n_rounds_multi   = 0;
  double wall_multi       = 0.0;

  int    balanced_batch   = 0;
  double wall_balanced    = 0.0;
};

// Round-scheduler choice picked from `RoundCandidates` by either the
// auto cost model or the `ZENDNNL_GRP_MATMUL_N_ROUNDS` env knob.
enum class RoundPick { Single, Multi, Balanced };

// =====================================================================
// Section P.5 — Test-only snapshot of the planner's Phase B output
// =====================================================================
//
// Locks down the heaviest-first / eligibility-filter behaviour of
// `apply_round_pick`'s Phase B body: the end-to-end correctness test
// in `test_algos.cpp` cannot directly observe per-expert thread
// assignments, so the snapshot below captures the relevant subset of
// the plan on the next `flat_n_tile()` call when capture is armed.
//
// Tests arm capture by storing `true` to `s_capture_phase_b`, then
// invoke `group_matmul_direct` with shapes that route to ALGO 3.
// `flat_n_tile` checks the flag right after the planner returns and,
// if armed, atomically swaps it back to false (one-shot capture) and
// copies the relevant plan fields into `s_last_phase_b_snapshot`.
//
// Single-process scope; safe for single-threaded gtests.  The flag is
// written once (test) and consumed once (library) per logical capture
// cycle, so a relaxed atomic exchange suffices.
namespace test_api {

struct PhaseBSnapshot {
  bool valid = false;
  bool per_expert_remainder = false;
  GroupNTileStrategy strategy = GroupNTileStrategy::Sequential;
  int batch_size = 0;
  int n_thr_fixed = 0;
  // Per-expert thread counts at indices [0, num_ops_active);
  // remaining slots stay zero.
  int num_ops_active = 0;
  std::array<int16_t, GroupNTilePlan::kMaxExperts>
      stable_n_thr_per_expert{};
};

inline std::atomic<bool> s_capture_phase_b{false};
inline PhaseBSnapshot     s_last_phase_b_snapshot;

}  // namespace test_api

}  // namespace matmul
}  // namespace lowoha
}  // namespace zendnnl

#endif  // ZENDNNL_GROUP_MATMUL_N_TILE_PLANNER_HPP
