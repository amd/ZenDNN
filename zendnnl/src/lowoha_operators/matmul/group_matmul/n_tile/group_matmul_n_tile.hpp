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

/// ALGO 3 — N-tile parallel GEMM, public interface header.
///
/// This header carries the planner-shaped data types AND the public
/// surface (env override atoms, env getters, `flat_n_tile()` forward
/// declaration) that the `flat_n_tile()` translation unit, planner
/// gtests, and the top-level dispatcher need to share.  It is
/// library-internal: not part of the public ZenDNN API and not meant
/// for inclusion outside `src/lowoha_operators/matmul/group_matmul/`
/// (plus the `gtests/group_matmul/` test files that exercise N-tile
/// path overrides).
///
/// Layout — section banners match the .cpp where applicable:
///
///   N-tile shared constants `kDecodeNTile` and `kNTilePlanMaxExperts`
///   live in the companion planner header
///   `group_matmul_n_tile_planner.hpp` (top of that file, near the
///   `GroupNTilePlan` definition that consumes `kNTilePlanMaxExperts`
///   at compile time).  This header re-includes the planner so every
///   N-tile consumer (executor, dispatcher, gtests) sees them through
///   `n_tile/group_matmul_n_tile.hpp`.
///
///   Section A.0  PerThreadScratch — per-thread aligned heap buffer
///                used by the tight-fused-epilogue path of `do_tile()`.
///                Lives here so a future memory-audit gtest can
///                exercise `grow_scratch()` directly.
///
///   Section A.1  Strategy enum + planner-shaped types (Topology,
///                Plan, RoundCandidates, RoundPick).  All POD-ish:
///                only fundamental types and `std::array`, no
///                references or owning resources.  Safe to include
///                from gtests for planner property tests.
///
///   Section A.2  test_api::PhaseBSnapshot capture (existing).  Locks
///                down the heaviest-first / eligibility-filter
///                behaviour of `apply_round_pick`'s Phase B body for
///                the end-to-end correctness test in `test_algos.cpp`.
///
///   Section A.3  test_api::s_grp_*_n_tile_* override atoms.
///                Sentinel-`-1` ("no override") or sentinel-`INT_MIN`
///                atomics that shadow the env-cache.  Flipped by
///                gtests (via RAII helpers in
///                `gtests/group_matmul/moe_test_utils.hpp`) and
///                long-running services that want to switch
///                configuration without re-launching the process.
///
///   Section A.4  N-tile env getters: `get_grp_n_tile_*()` /
///                `get_grp_matmul_n_*()`.  Cached-static-const +
///                atomic-override read pattern; see
///                `parse_env_int_strict` (in
///                `group_matmul_parallel_common.hpp`) for the strict
///                env parsing convention.
///
///   Section A.5  N-tile shared utilities (`sort_indices_by_m`,
///                `engage_ntile_custom_kernel`,
///                `ntile_effective_nr_align`, `auto_pick_n_order`,
///                `fill_ntile_expert_order`) — free functions at
///                namespace scope (NOT in `test_api`) consumed by
///                `group_matmul_n_tile.cpp` and any future additional
///                N-tile executor.  Previously lived in
///                `group_matmul_parallel_common.hpp`.
///
///   Section A.6  Forward declaration for the N-tile executor
///                `flat_n_tile()`.  Defined in
///                `group_matmul_n_tile.cpp`.
///
/// What stays in `group_matmul_n_tile.cpp`:
///   * `GroupNTileContext`  — captures the caller's `std::vector<…> &`
///                            inputs by reference; private impl detail
///                            of one OMP region.
///   * Planner functions    — pure decision logic on the structs above.
///   * Strategy executors   — OMP-parallel matmul drivers.
///   * `flat_n_tile()`      — body of the public entry declared in
///                            Section A.5 above.
///
/// Dependency direction:
///   * This header depends on `group_matmul_parallel_common.hpp` for
///     shared types (`matmul_params`, `grp_matmul_gated_act_t`,
///     `data_type_t`) and the `parse_env_int_strict` helper, and on
///     `group_matmul_n_tile_planner.hpp` for the planner output types
///     and the test-only Phase-B capture machinery.
///   * `../group_matmul_parallel_common.hpp` does NOT include this
///     header — consumers of N-tile interfaces include this file
///     explicitly.  One-way dependency, mirrors
///     `../m_tile/group_matmul_m_tile.hpp`.

#ifndef ZENDNNL_GROUP_MATMUL_N_TILE_HPP
#define ZENDNNL_GROUP_MATMUL_N_TILE_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <vector>

#include "operators/matmul/matmul_config.hpp"  // matmul_algo_t
// `group_matmul_parallel_common.hpp` provides:
//   * `parse_env_int_strict`            — strict env parsing helper
//   * `matmul_params` / `data_type_t`   — used by `flat_n_tile()` decl
//   * `grp_matmul_gated_act_t`          — fused-act argument type
//   * the inline `using namespace zendnnl::ops;` declaration
//
// NOTE: `kNTilePlanMaxExperts` previously came from
// `group_matmul_parallel_common.hpp`; it now lives in the companion
// planner header `group_matmul_n_tile_planner.hpp` (re-included
// below) with the other N-tile-specific constants so `GroupNTilePlan`
// can size its stack arrays at struct-definition time.  Consumers
// that include this public N-tile header transitively see the
// symbol via that re-include — no separate include is needed.
#include "../group_matmul_parallel_common.hpp"
// `group_matmul_n_tile_planner.hpp` provides the planner output
// types (`GroupNTileStrategy`, `GroupNTileTopology`, `GroupNTilePlan`,
// `RoundCandidates`, `RoundPick`) plus the test-only Phase-B capture
// machinery (`PhaseBSnapshot`, `s_capture_phase_b`,
// `s_last_phase_b_snapshot`).  Re-included here so any TU that
// includes the public N-tile interface transitively gets the planner
// surface — symmetric with the M-tile pattern in
// `../m_tile/group_matmul_m_tile.hpp`.
#include "group_matmul_n_tile_planner.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

// Bring `matmul_algo_t` into scope so the `algo` field on
// `GroupNTilePlan` reads as `matmul_algo_t` (matches the unqualified
// usage in the rest of the group_matmul code, which gets the same
// using-decl via `group_matmul_parallel_common.hpp`).
using zendnnl::ops::matmul_algo_t;

// N-tile shared constants (`kDecodeNTile`, `kNTilePlanMaxExperts`)
// live in the companion planner header
// `group_matmul_n_tile_planner.hpp` (re-included via line above), at
// the top of that file.  They must be defined there — not here — so
// `GroupNTilePlan`'s stack arrays (sized by `kNTilePlanMaxExperts` at
// struct definition time) see the constant before the struct body is
// parsed.  This header re-includes the planner unconditionally, so
// every consumer of the N-tile public surface (executor, dispatcher,
// gtests, fused-MoE) still observes the constants through
// `n_tile/group_matmul_n_tile.hpp`.

// =====================================================================
// Section A.0 — PerThreadScratch
// =====================================================================
//
// Per-thread aligned heap buffer used by the tight-fused-epilogue
// path of `do_tile()`.  When `fused_epilogue=swiglu` AND the caller's
// dst is a tight [M, I]-layout buffer (ldc < N), the classic
// matmul-then-in-place-compact pattern no longer fits (matmul writes
// 2I cols but dst only has I cols per row).  In that case `do_tile()`
// switches to a per-thread scratch + out-of-place activation flow:
//
//   1. matmul the thread's N-tile slice into a thread-local scratch
//      buffer (wide, `n_tile` cols, ldc=n_tile),
//   2. run `apply_swiglu_oai_tile_rows_oop()` from that scratch into
//      the caller's tight dst at halved col offset `col_start / 2`.
//
// This is barrier-free: each thread writes to its own scratch then to
// a disjoint column range of the caller's tight dst — no cross-thread
// reads needed inside the fused-epilogue step.
//
// Lifetime: `static thread_local` inside the OMP region; monotonically
// grows to the high-water mark this thread has seen.  Freed by the
// destructor on thread exit.  Per-thread, so no contention.
//
// The buffer is 64-byte aligned so the custom kernel's `vmovdqu`
// reads start on a cache-line boundary.
//
// Move/copy operations are deleted because the struct owns `buf` via
// `std::free` in the destructor — accidental copying would leave two
// instances pointing at the same allocation and double-free on
// destruction.  Production use is always `static thread_local` inside
// an OMP region (one instance per worker, fixed lifetime); the delete
// catches misuse if a future caller or test instantiates it as a
// stack variable and accidentally returns / passes it by value.
struct PerThreadScratch {
  void *buf = nullptr;
  size_t cap = 0;
  PerThreadScratch() = default;
  ~PerThreadScratch() { std::free(buf); }
  PerThreadScratch(const PerThreadScratch &)            = delete;
  PerThreadScratch &operator=(const PerThreadScratch &) = delete;
  PerThreadScratch(PerThreadScratch &&)                 = delete;
  PerThreadScratch &operator=(PerThreadScratch &&)      = delete;
};

// Grow a per-thread scratch to at least `need` bytes, 64-byte aligned.
// Returns false on alloc failure (caller signals via alloc_fail atomic
// + post-OMP-region check).
inline bool grow_scratch(PerThreadScratch &s, size_t need) {
  if (need <= s.cap) return true;
  std::free(s.buf);
  s.buf = nullptr;
  s.cap = 0;
  void *tmp = nullptr;
  if (posix_memalign(&tmp, 64, need) != 0 || tmp == nullptr) return false;
  s.buf = tmp;
  s.cap = need;
  return true;
}

// =====================================================================
// Sections A.1 / A.2 — PLANNER — moved to
//                       `group_matmul_n_tile_planner.hpp` (Sections
//                       P.1 – P.5)
// =====================================================================
//
// The strategy enum (`GroupNTileStrategy`), topology summary
// (`GroupNTileTopology`), per-call plan (`GroupNTilePlan`),
// multi-round cost-model types (`RoundCandidates` / `RoundPick`),
// and the test-only Phase-B capture machinery (`PhaseBSnapshot`,
// `s_capture_phase_b`, `s_last_phase_b_snapshot`) all live in the
// companion planner header so future N-tile planner optimizations
// land in a single, isolated optimization surface — symmetric with
// the M-tile pattern (see `../m_tile/group_matmul_m_tile_planner.hpp`).
//
// This file re-includes the planner header at the top of the
// `#include` block, so every TU that pulls in the public N-tile
// interface continues to see all of the above symbols unchanged.

// =====================================================================
// Section A.3 — N-tile test-only override atoms (test_api)
// =====================================================================
//
// Cached env getters (Section A.4 below) capture their value at the
// first call (`static const` lambda) so production reads are
// branch-predictor-friendly.  That precludes a unit test that runs
// AFTER another test has already cached a non-default value from
// flipping the cached value back via `setenv` — the getter returns
// the cached snapshot regardless.
//
// The atomics below let a test override the cached value on the
// production read path (one relaxed-load + branch per getter call,
// negligible vs the surrounding planner / OMP work).  Sentinel `-1`
// (or `INT_MIN` for the heavy-threshold knob, which uses `-1` as a
// meaningful "DISABLED" value) means "use the cached env path"
// (production default).  Tests should set the override via the RAII
// helpers in `gtests/group_matmul/moe_test_utils.hpp` to guarantee
// the override is cleared on scope exit, including on test failure
// / fixture teardown.
//
// Mirrors the M-tile equivalents in `group_matmul_m_tile.hpp`
// (Section H.1) — same sentinel convention, same RAII pattern, same
// strict-parse-or-default behaviour on bogus values.
namespace test_api {

// Sentinel `-1` = no override.  Settable values: 0 (no custom N-tile —
// auto-pick `effective_decode_n_tile()`), or any positive multiple of
// 32 (e.g. 128, 256, 512).  Other positive values pass `ovr >= 0` but
// are normalised to 0 by `get_grp_matmul_custom_kernel_n_tile()` (see
// the doc-block on that getter for the full override semantics).
inline std::atomic<int> s_grp_matmul_custom_kernel_n_tile_override{-1};

// Sentinel `-1` = no override.  Settable values: 0 (auto, default —
// try DecodeD if eligible, fall through to Rounds), 1 (decode —
// prefer DecodeD when its eligibility passes; same behaviour as auto
// today, kept distinct for explicit user intent + apilog hint), 2
// (rounds — skip DecodeD attempt entirely, always run Rounds-based
// FewExperts/ManyExperts).  See `get_grp_n_tile_strategy()` for the
// production env path.
inline std::atomic<int> s_grp_n_tile_strategy_override{-1};

// Sentinel `INT_MIN` = no override; falls through to the cached env
// path (which itself applies the documented default 0 = AUTO).
// `-1` is no longer usable as the "no override" marker because it
// now carries a meaningful value (DISABLED) — see the three-mode
// doc-block on `get_grp_matmul_n_tile_heavy_threshold()` below.
//
// Settable values:
//   * INT_MIN   — no override (falls through to env-cache, default 0
//                 = AUTO).  Tests should never set this explicitly;
//                 it is the production state.
//   * -1        — explicit DISABLED (prompt → uniform Phase B base+1).
//   *  0        — explicit AUTO.  Engages
//                 `apply_adaptive_tiers()` in the planner.
//   *  > 0      — explicit MANUAL single-threshold override.  Heavy
//                 iff `M[e] > value`.
//   * Anything more negative than -1 → undefined.  Tests should
//     only pass values from the documented set above.
//
// The RAII helper `NTileHeavyThresholdOverride` in
// `gtests/group_matmul/moe_test_utils.hpp` saves and restores the
// previous value across test scopes; it must be used for any test
// that touches this atomic to guarantee teardown ordering on test
// failure.
inline std::atomic<int> s_grp_matmul_n_tile_heavy_threshold_override{
    std::numeric_limits<int>::min()};

// Sentinel `-1` = no override (falls through to env / default OFF).
// `0` = force OFF (decode uses uniform Phase B base+1), `1` = force ON
// (decode M-proportional split).  RAII helper `DecodeProportionalOverride`
// in `gtests/group_matmul/moe_test_utils.hpp` saves/restores it.
inline std::atomic<int> s_grp_matmul_decode_proportional_override{-1};

}  // namespace test_api

// =====================================================================
// Section A.4 — N-tile env getters
// =====================================================================

// ZENDNNL_GRP_MATMUL_N_TILE_FUSED_ACT = { "0", "1" } — cached, default ON.
//   ALGO 3 folds a supported gated activation into the per-thread
//   epilogue (saves a second OMP pass over dst).  Adds one OMP barrier
//   between matmul-write and activation-read for correctness.  ON by
//   default; the env is retained as an escape hatch.  Mid-process env
//   changes have no effect (static const).
inline bool get_grp_n_tile_fused_act() {
  static const bool v = []() {
    const char *e = std::getenv("ZENDNNL_GRP_MATMUL_N_TILE_FUSED_ACT");
    if (e == nullptr || e[0] == '\0') return true;  // default: ON
    return e[0] != '0';
  }();
  return v;
}

// ZENDNNL_GRP_MATMUL_N_TILE_STRATEGY = { 0, 1, 2, 3 } — cached, default 2 (rounds).
//
// Selects the ALGO 3 (flat_n_tile) per-tile dispatch shape AND
// controls whether the planner's auto-mirror perf gate fires.
// Four values with distinct semantics:
//
//     0 = auto (opt-in heuristic).  Honour the planner's auto-mirror
//         gate (route to Sequential when `auto_select_algo` would
//         have picked ALGO 1 for this shape — few experts (num_ops ≤
//         kFewExpertsAlgo1), or prompt-class with num_ops < num_threads).
//         If the call survives auto-mirror, try DecodeD WITH its
//         eligibility heuristic (decode-class max_M ≤ 32,
//         num_ops ∈ [6, num_ccds], min_M_active ≥ 3, skew_ratio ≤ 4,
//         max_N / decode_n_tile ≤ team_size_est); fall through to
//         Rounds (FewExperts / ManyExperts) when the heuristic
//         refuses.  Not the default because DecodeD's eligibility gate
//         rarely engages, so the auto branch reduces to "Rounds with
//         extra precondition checks" in practice.
//
//     1 = decode (FORCE).  SKIP the auto-mirror gate AND skip the
//         eligibility heuristic — run DecodeD on every shape where it
//         is STRUCTURALLY feasible (`num_threads >= num_ops`; smaller
//         teams would over-subscribe DecodeD's OMP region and collide
//         the `tid → expert` mapping).  Lets the caller run DecodeD on
//         shapes the heuristic would route away (e.g. a decode call
//         with active experts > num_ccds, where eligibility's `num_ops
//         ≤ num_ccds` gate refuses but DecodeD's executor still runs
//         correctly with thin per-expert teams of size
//         `num_threads / num_ops`).  Logs an apilog L3 line
//         describing the resulting allocation, OR a fallback line
//         if num_threads < num_ops forces a Rounds fall-through.
//
//     2 = rounds (DEFAULT, FORCE).  SKIP the auto-mirror gate AND
//         the viability perf heuristic (same as 1) — when the caller
//         (or ALGO 0 auto-pick) routed to ALGO 3 we mean to run
//         N-tile, not silently bounce back to Sequential on a perf
//         preference.  Skip the DecodeD attempt entirely; always run
//         the Rounds path (FewExperts / ManyExperts).  This is the
//         production default: deterministic ALGO 3 = "true N-tile with
//         rounds" behaviour across all decode and prompt shapes that
//         survive the structural gates, and the path exercised by the
//         in-tree MoE gtests.
//
//     3 = decode_dynamic (FORCE).  SKIP the auto-mirror gate AND the
//         viability heuristics (same as 1/2).  Per op, a gate decides
//         between the barrier-free CCD-cohesive DecodeDynamic executor
//         (`DecodeDynamic` -> `execute_decode_dynamic`) and Rounds, from
//         the active expert count and per-expert weight (see the
//         DecodeDynamic knob block below).  DecodeDynamic targets the
//         decode-class regime `num_ops > num_ccds` (max_M <= decode
//         threshold); it maps whole experts onto CCDs and processes them
//         in per-CCD waves, so it also runs when `num_ops > num_threads`.
//         Covers the CK custom-kernel path (swiglu fused in-register, so
//         no matmul->activation barrier) plus standard-backend fused /
//         non-fused calls (non-custom wide-fused runs one team-wide
//         barrier + apply_swiglu_oai post-pass inside the executor).
//         Only a use_custom DQ-INT8 fused call falls back to Rounds
//         (defense-in-depth).  AUTO (value 0) also engages DecodeDynamic
//         under the SAME gate (decode-class many-active-expert shapes,
//         active_ops >= 4*num_ccds); below the gate AUTO uses its normal
//         Rounds / AOCL path.
//
// What survives `n_tile_strategy = {1, 2, 3}` (genuinely STRUCTURAL —
// memory safety / kernel correctness, not perf):
//
//   * R3 — capacity overflow (`num_ops > GroupNTilePlan::kMaxExperts
//     = 256`).  Stack-array bound on the planner; demoting to
//     Sequential is the only safe recourse.  Auto-select rule 0 also
//     captures this upstream by routing to ALGO 5.
//
//   * F3 narrow-N escape — only reachable when the strict-stable
//     AOCL path runs (`CUSTOM_KERNEL=0 && AOCL_STABLE_NTILE=1`).
//     When `stable * nr_align > max_N`, `aligned_n_split` cannot
//     produce stable aligned partitions and the AOCL kernel's
//     nr-alignment contract would be violated.  Sequential bypasses
//     tile-level keys entirely.  Not reachable under the production
//     default `CUSTOM_KERNEL=1`.
//
//   * tight split-halves CK refusal — silu_and_mul / gelu_and_mul +
//     tight caller (`ldc < N`) when the custom kernel refuses
//     (typically silu/gelu + bias).  Handled post-plan in
//     `flat_n_tile`; Sequential allocates the wide [M, N] scratch +
//     `apply_gated_act_inplace` + tight memcpy that the tight
//     swiglu-only fast path cannot.
//
// What is GATED behind `!force_ntile` (auto-mode-only perf
// heuristics — honoured under env=0, ignored under env={1,2}):
//
//   * `auto_mirror` — replays auto-select's ALGO 1 preference.
//     Lets `ALGO=3` behave like `ALGO=0` on shapes the auto-picker
//     would have routed to ALGO 1, with a distinct gemm_mode label
//     for telemetry.
//
//   * `!ntile_viable` — heuristic "N too thin for a useful per-
//     thread split".  Under explicit env=1/2 the user accepts
//     whatever cost a thin N gives them — we run N-tile and emit a
//     `[GRP_MATMUL.PLAN.HINT]` line so the env-honoured-over-
//     heuristic decision is visible in the L3 trail.
//
// See `plan_group_n_tile` in `group_matmul_n_tile.cpp` for the
// authoritative precedence diagram and emission sites.
//
// Mid-process env changes have no effect (cached static const);
// tests should use `s_grp_n_tile_strategy_override` via the RAII
// helper `NTileStrategyOverride` in `gtests/group_matmul/
// moe_test_utils.hpp` to flip it deterministically inside the same
// process.  Existing tests that pin the planner to its heuristic
// path use `NTileStrategyOverride(0)` and continue to work — only
// the unset / invalid default changed.
//
// Validation paths differ slightly between the env and the override:
//   * Env path  — invalid values (< 0 OR > 3) parse to 2 (rounds),
//                 matching the "unset → safe default" convention used
//                 by the other knobs in this header.  Note this
//                 differs from the historical default of 0 (auto)
//                 documented in older notes.
//   * Override path — `-1` is the sentinel for "no override" and
//                 falls through to the cached env path; any other
//                 negative value also falls through (so a bogus
//                 negative typo cannot accidentally pin a strategy).
//                 Non-negative override values > 3 clamp to 2
//                 (rounds), mirroring the env path on the upper end.
inline int get_grp_n_tile_strategy() {
  // Unset / invalid → 2 (rounds): production default; ALGO 3 always
  // runs FewExperts / ManyExperts when the structural gates pass.
  // See the doc-block above for the rationale and the precedence
  // diagram in `plan_group_n_tile`.  Strict env parsing — non-
  // numeric input (e.g. `"abc"`) falls back to the documented
  // default 2, NOT silently to mode 0 via legacy atoi-returns-0
  // behaviour.  See `parse_env_int_strict`.
  constexpr int kDefault = 2;
  constexpr int kMaxValue = 3;  // 0=auto, 1=decode_d, 2=rounds, 3=decode_dynamic
  const int ovr = test_api::s_grp_n_tile_strategy_override.load(
      std::memory_order_relaxed);
  if (ovr >= 0) return (ovr <= kMaxValue) ? ovr : kDefault;
  static const int v = []() {
    const char *e = std::getenv("ZENDNNL_GRP_MATMUL_N_TILE_STRATEGY");
    int parsed = 0;
    if (!parse_env_int_strict(e, parsed)) return kDefault;
    return (parsed >= 0 && parsed <= kMaxValue) ? parsed : kDefault;
  }();
  return v;
}

// ── DecodeDynamic generic decision-tree knobs ────────────────────────
//
// These take effect whenever DecodeDynamic is eligible for selection,
// i.e. under BOTH:
//   * `ZENDNNL_GRP_MATMUL_N_TILE_STRATEGY=3` (forced DecodeDynamic), and
//   * `ZENDNNL_GRP_MATMUL_N_TILE_STRATEGY=0` (smart AUTO), where the
//     planner may route decode-class shapes to DecodeDynamic.
// Under the production default (2 = rounds) — and under `=1` (DecodeD) —
// the flow is untouched and these getters are never consulted.
//
// The route DecodeDynamic-vs-Rounds is decided PER OP from two signals
// available in `topo` — the active expert count, the machine CCD count,
// and this op's per-expert weight bytes vs a CCD's L3.  No layer/model
// identity is used:
//
//   use_decode_dynamic =
//        (active_ops  >= EPC_MULT  * num_ccds)         // enough experts/CCD
//     || (wei_per_expert >= WEI_L3_MULT * kL3PerCcdBytes) // weight >> CCD L3
//
// Structural rationale for the two branches:
//   * experts-per-CCD (epc = active_ops/num_ccds): the CCD-cohesive map
//     assigns whole experts to CCDs, so its load balance is governed by
//     how evenly `active_ops` divides `num_ccds`.  At low epc a single
//     extra expert overloads one CCD (the imbalance is ~ceil(epc)/epc),
//     so DecodeDynamic only pays off once epc is large enough that this
//     rounding imbalance is small — EPC_MULT (default 4).  epc grows with
//     batch size, so this is the decode-throughput-sensitive branch.
//   * per-expert weight vs CCD L3: the Rounds path batches experts to
//     fit their weights in aggregate L3, so when a single expert's
//     weight already exceeds a CCD's L3 it serialises down to ~1 expert
//     per round; DecodeDynamic instead streams experts concurrently, one
//     per CCD.  WEI_L3_MULT (default 2, i.e. >= 2 CCD-L3 worth) selects
//     that regime.
// Both thresholds are env-overridable for tuning to other shapes.

// ZENDNNL_GRP_MATMUL_DECDYN_EPC_MULT — cached, default 4.
//   DecodeDynamic when active_ops >= this * num_ccds (experts-per-CCD).
inline int get_grp_decdyn_epc_mult() {
  constexpr int kDefault = 4;
  static const int v = []() {
    const char *e = std::getenv("ZENDNNL_GRP_MATMUL_DECDYN_EPC_MULT");
    int parsed = 0;
    if (!parse_env_int_strict(e, parsed) || parsed < 1) return kDefault;
    return parsed;
  }();
  return v;
}

// ZENDNNL_GRP_MATMUL_DECDYN_WEI_L3_MULT — cached, default 2.
//   DecodeDynamic when wei_per_expert >= this * kL3PerCcdBytes (32 MB),
//   i.e. the per-expert weight is large enough that Rounds' L3-batching
//   serialises.  0 disables this branch (epc rule only).
inline int get_grp_decdyn_wei_l3_mult() {
  constexpr int kDefault = 2;
  static const int v = []() {
    const char *e = std::getenv("ZENDNNL_GRP_MATMUL_DECDYN_WEI_L3_MULT");
    int parsed = 0;
    if (!parse_env_int_strict(e, parsed) || parsed < 0) return kDefault;
    return parsed;
  }();
  return v;
}

// ZENDNNL_GRP_MATMUL_N_ORDER = { 0..4 } — cached, default 3 (pair-balanced).
//   Permutation of experts walked by ALGO 3 FewExperts/ManyExperts.
//     0 = auto: shape-aware picker (auto_pick_n_order); resolved
//         sub-mode is logged in `[GRP_MATMUL.PLAN]` APILOG.
//     1 = ascending  — by M, lightest first.  Was the previous
//         default for AOCL DLP cache-key stability (n_thr_fixed
//         schedule keeps thread-id → expert mapping shape-invariant
//         when permutation is shape-invariant too).  That rationale
//         no longer applies under `CK=1` (production default): the
//         per-tile cache is shape-keyed in the CK pack arena, not
//         thread-id-keyed, so the ordering is free to optimise purely
//         for load balance.
//     2 = descending — by M, heaviest first; minimises
//                      Σ max_M_per_round under fixed-batch rounds.
//                      Historically considered for multi-round
//                      configurations; the default (3) is preferred in
//                      practice (see below).
//     3 = pair-balanced — desc, then interleave largest with smallest
//         (heavy/light alternation).  CURRENT DEFAULT.  Single-round
//         wall time is bounded by the slowest thread's per-expert
//         duty cycle; pair-balanced flattens this duty cycle across
//         the OMP team by alternating heavy/light experts so all
//         threads finish around the same time.
//     4 = balanced-spread — prefix-sum-balanced: any K-way consecutive
//                           split yields Σ M per chunk ≈ total / K
//                           (heavies evenly distributed throughout).
//   Mid-process env changes have no effect; relaunch to change it.
inline int get_grp_matmul_n_order() {
  // Default: 3 (pair-balanced).  Strict env parsing — non-numeric
  // input (e.g. `"abc"`) falls back to the documented default 3,
  // NOT silently to mode 0 via the legacy `std::atoi`-returns-0
  // behaviour.  See `parse_env_int_strict`.
  constexpr int kDefault = 3;
  static const int v = []() {
    const char *e = std::getenv("ZENDNNL_GRP_MATMUL_N_ORDER");
    int parsed = 0;
    if (!parse_env_int_strict(e, parsed)) return kDefault;
    return (parsed >= 0 && parsed <= 4) ? parsed : kDefault;
  }();
  return v;
}

// ZENDNNL_GRP_MATMUL_N_TILE_HEAVY_THRESHOLD = { -1, 0, positive int } —
//   cached, default 0 (AUTO).  NOTE: this knob is PROMPT-ONLY; decode
//   uses the unconditional M-proportional split regardless of its value
//   (gate `ZENDNNL_GRP_MATMUL_DECODE_PROPORTIONAL`).
//
// Three-mode dispatch for asymmetric per-expert thread distribution
// inside the ALGO 3 ManyExperts Single-round plan.  ALL active modes
// are PROMPT-ONLY (`max_M > kDecodeMaxM`); decode-class calls bypass
// the N_TILE heavy-threshold dispatch entirely.  NOTE: decode no longer
// uses uniform Phase B base+1 — it uses the UNCONDITIONAL M-proportional
// thread split (see `apply_round_pick` Single case), which is
// independent of this knob.  This knob only governs the PROMPT path.
//
//   -1  DISABLED (explicit opt-out).  Phase B
//       base+1 only: top few experts by M get one extra thread, all
//       others get a uniform share via `n_thr_fixed`.  Same as the
//       legacy behaviour when the env was unset.
//
//    0  AUTO  (DEFAULT, prompt-only).  Planner-driven adaptive 3-tier policy.
//       `apply_adaptive_tiers()` (group_matmul_n_tile.cpp) inspects
//       the per-call M distribution, num_threads and num_active and
//       builds a per-expert thread allocation with:
//         - high   tier (M ≥ ~0.40 × M_max): target up to 8 threads
//         - mid    tier (M ≥ ~0.20 × M_max): target up to 4 threads
//         - low    tier (M ≥ ~0.10 × M_max): target up to 2 threads
//         - baseline (everyone else):        1 thread
//       Tier targets are scaled down uniformly when the
//       `num_threads − num_active` extras-budget is tight, then
//       water-filled by M-weight to consume any rounding leftover.
//       Falls back silently to Phase B when the workload doesn't
//       benefit (low skew, thread-starved, etc.).  Adapts to
//       num_threads ∈ {64, 128, 256} automatically — no manual
//       per-CPU tuning required.
//
//       Decode bypass: on `max_M ≤ kDecodeMaxM` the AUTO path
//       returns immediately (defence-in-depth check at the top of
//       `apply_adaptive_tiers()`) and the planner runs Phase B
//       base+1 instead.
//
//   >0  MANUAL single-threshold (legacy) — prompt-only.  Experts
//       with `M[e] > value` are tagged HEAVY; each active light
//       expert reserves exactly 1 thread; the remaining heavy-budget
//       is water-filled across heavies by M, capped at
//       `min(ccd_size, max_tiles, N[e] / ab_min_tile)`.  A value near
//       1024 is a reasonable starting point for large-max_M prompt
//       shapes.
//
// Invalid values (< -1, "abc", etc.) → silently treated as the default
// (0 / AUTO — the prompt adaptive-tier policy described above), matching
// the strict-parse convention of the other ZENDNNL_GRP_MATMUL_* env
// vars.  Mid-process env changes have no effect; relaunch to change it.
//
// All three modes share the same executor consumer
// (`stable_n_thr_per_expert[]` + `per_expert_remainder = true`); the
// only difference is how the planner populates that array.
inline int get_grp_matmul_n_tile_heavy_threshold() {
  constexpr int kDefault = 0;  // AUTO (prompt adaptive tiers)
  // Test override sentinel: INT_MIN = no override.  Cannot use `-1`
  // any more since `-1` is now a meaningful (DISABLED) value.
  // Production keeps the static-const env-cache for branch-
  // predictor-friendly reads.
  const int ovr = test_api::s_grp_matmul_n_tile_heavy_threshold_override
      .load(std::memory_order_relaxed);
  if (ovr != std::numeric_limits<int>::min()) return ovr;
  static const int v = []() {
    const char *e =
        std::getenv("ZENDNNL_GRP_MATMUL_N_TILE_HEAVY_THRESHOLD");
    int parsed = 0;
    if (!parse_env_int_strict(e, parsed)) return kDefault;
    // Accept -1 (DISABLED), 0 (AUTO), positive int (MANUAL).  Reject
    // anything more negative — silently clamp to default.
    return (parsed >= -1) ? parsed : kDefault;
  }();
  return v;
}

// ZENDNNL_GRP_MATMUL_DECODE_PROPORTIONAL = { "0", "1" } — cached,
//   default OFF.  Gates the DECODE-class M-proportional per-expert
//   thread split in the ALGO 3 CK Single-round plan (allocate threads
//   to each active expert in proportion to its M, clamped to its
//   N-tile capacity).
//   OFF (default): uniform Phase B base/base+1 distribution.
//   ON ("1"):      M-proportional split for decode CK (opt-in).
//   Independent of ZENDNNL_GRP_MATMUL_N_TILE_HEAVY_THRESHOLD (which is
//   prompt-only).  Mid-process env changes have no effect (static const).
inline bool get_grp_matmul_decode_proportional() {
  const int ovr = test_api::s_grp_matmul_decode_proportional_override.load(
      std::memory_order_relaxed);
  if (ovr >= 0) return ovr != 0;
  static const bool v = []() {
    const char *e = std::getenv("ZENDNNL_GRP_MATMUL_DECODE_PROPORTIONAL");
    if (e == nullptr || e[0] == '\0') return false;  // default: OFF
    return e[0] != '0';
  }();
  return v;
}

// ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_N_TILE = { unset, multiple of 32 } — cached.
//   Override the ALGO 3 outer N-tile minimum (default kDecodeNTile=256).
//   Use 128 for high threads/num_ops (more thread participation), 512
//   for prompt-class (wider tiles amortise kernel-call overhead).
//   Non-multiples of 32 → ignored (silently safe vs typos).
inline int get_grp_matmul_custom_kernel_n_tile() {
  const int ovr =
      test_api::s_grp_matmul_custom_kernel_n_tile_override.load(
          std::memory_order_relaxed);
  // Override semantics:
  //   * `-1`  — sentinel, no test override; fall through to the
  //             cached env / default path below.
  //   * `0`   — explicit "no custom N-tile" override; the planner
  //             reads 0 here and `effective_decode_n_tile()` falls
  //             back to `kDecodeNTile` — same as an unset env.
  //   * `> 0` AND multiple of 32 — adopted as the override value.
  //   * any other positive value — normalised to 0 ("no custom
  //             N-tile"), mirroring the `parsed > 0 && parsed % 32 == 0`
  //             validation the env-cached path applies below.  This
  //             still counts as a "test has spoken" override (it does
  //             NOT fall through to the env path) — the planner reads
  //             0 and `effective_decode_n_tile()` picks `kDecodeNTile`,
  //             keeping the test API noise-free against typos.
  //
  // The `ovr >= 0` branch covers all three "test has spoken" cases
  // (0 and any positive value, valid or not); only `-1` falls
  // through to the env path.
  if (ovr >= 0) return (ovr > 0 && (ovr % 32) == 0) ? ovr : 0;
  // Strict env parsing — non-numeric input falls back to 0
  // (auto-pick the planner's `effective_decode_n_tile()`).
  static const int v = []() {
    const char *e = std::getenv("ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_N_TILE");
    int parsed = 0;
    if (!parse_env_int_strict(e, parsed)) return 0;
    return (parsed > 0 && (parsed % 32) == 0) ? parsed : 0;
  }();
  return v;
}

// =====================================================================
// Section A.5 — N-tile shared utilities (consumed by
//               group_matmul_n_tile.cpp and a future additional N-tile
//               executor; previously lived in
//               `group_matmul_parallel_common.hpp`).
// =====================================================================
//
// Free functions exposed at namespace scope (NOT inside `test_api`):
// these are real library primitives the production N-tile executor
// calls into, not test hooks.  They moved here from
// `group_matmul_parallel_common.hpp` together with the env knobs they
// read (`get_grp_matmul_n_order` for `fill_ntile_expert_order`,
// `get_grp_matmul_custom_kernel` (which stays in parallel_common.hpp)
// for `engage_ntile_custom_kernel`).

/// Sort `indices[0..n)` by `M[idx]` (asc or desc).  Heap-free; caller
/// owns the buffer and guarantees `indices.size() >= n` and `M.size() >= n`.
inline void sort_indices_by_m(int *indices, int n,
                              const std::vector<int> &M, bool ascending) {
  for (int i = 0; i < n; ++i) {
    indices[i] = i;
  }
  std::sort(indices, indices + n, [&M, ascending](int a, int b) {
    return ascending ? (M[a] < M[b]) : (M[a] > M[b]);
  });
}

/// Env-gated custom-microkernel engagement for an N-tile executor.
/// Covers BOTH compute regimes:
///
///   * BF16 (`dynamic_quant=false`, default): bf16×bf16→bf16 (or
///     bf16×bf16→f32) — the original `engage_ntile_custom_kernel`
///     contract.
///   * DQ-INT8 (`dynamic_quant=true`, `compute_dtype∈{s8,u8}`):
///     hoist-quantised src(s8 or u8) × pre-packed wei(s8) → bf16,
///     per-token src_scale + optional src_zp, per-channel wei_scale.
///
/// Caller stack-allocates a fresh `CallContext` and reads `kctx.enabled`
/// on return.  Controlled by `ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL` (master,
/// cached, default ON) AND `ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_INT8` (int8
/// sub-toggle, cached, default ON — only consulted for DQ-INT8 calls;
/// bf16 calls ignore it).  Both default to ON when unset; set the env
/// to "0" to opt out (master "0" disables CK for all dtypes; int8 "0"
/// disables only the DQ-INT8 fast path, leaving bf16 CK active).
///
/// Even with the envs enabled, the dispatcher's per-call contract
/// check (dtype tuple, no transA, α=1, β=0, N % pack_nr, supported
/// act/bias dtypes, is_weights_const = true / empty for every active
/// expert) decides whether the path can actually run; caller falls
/// back to its standard path when kctx.enabled=false.
///
/// Supported `act`: none, swiglu_oai_mul (fused gate+up → halved out),
/// silu_and_mul, gelu_and_mul (canonical split-halves; CK arena
/// pack-permuted at prepack time).
inline void engage_ntile_custom_kernel(
    grp_matmul_gated_act_t act,
    data_type_t src_dtype,
    data_type_t wei_dtype,
    data_type_t dst_dtype,
    data_type_t act_dtype,
    data_type_t bias_dtype,
    const std::vector<bool>          &transA,
    const std::vector<bool>          &transB,
    const std::vector<int>           &M,
    const std::vector<int>           &N,
    const std::vector<int>           &K,
    const std::vector<int>           &ldb,
    const std::vector<float>         &alpha,
    const std::vector<float>         &beta,
    const std::vector<const void *>  &weight,
    const std::vector<bool>          &is_weights_const,
    custom_kernel::CallContext       &kctx,
    bool                              dynamic_quant = false,
    data_type_t                       compute_dtype = data_type_t::none) {
  if (!get_grp_matmul_custom_kernel()) return;
  // Master CK env is ON; gate the DQ-INT8 sub-toggle separately so
  // operators can toggle int8 without disabling bf16.  The int8 CK path
  // arrives two ways and BOTH must honour the sub-toggle:
  //   * runtime hoist  — `dynamic_quant=true` (bf16 src quantized
  //     per-tile);
  //   * grouped pre-quant — `group_dynamic_quant` already produced an
  //     s8 src and CLEARED `dynamic_quant`, so detect it via
  //     `src=s8 && wei=s8`.  Without this, a grouped-s8 call would
  //     engage CK even with the int8 sub-toggle OFF (inconsistent with
  //     `ck_eligible_int8` / prepack which honour the sub-toggle).
  // Bf16 calls (`src=bf16, wei=bf16`) never satisfy either clause.
  const bool is_dq_int8_call =
      dynamic_quant
      || (src_dtype == data_type_t::s8 && wei_dtype == data_type_t::s8);
  if (is_dq_int8_call && !get_grp_matmul_custom_kernel_int8()) return;
  custom_kernel::prepare_for_call(
      act, src_dtype, wei_dtype, dst_dtype, act_dtype, bias_dtype,
      transA, transB, M, N, K, ldb, alpha, beta, weight,
      is_weights_const, kctx, dynamic_quant, compute_dtype);
}

/// Effective N-column alignment for the per-thread split:
///   max(backend_nr, kctx.pack_nr if enabled, 2 if pair_aligned).
/// `backend_nr` from backend_n_align(algo); `pair_aligned`=true when
/// activation requires even col boundaries (e.g. swiglu_oai_mul must
/// keep gate+up pairs on the same thread).
inline int ntile_effective_nr_align(
  int backend_nr,
  const custom_kernel::CallContext &kctx,
  bool pair_aligned) {
  int a = backend_nr;
  if (kctx.enabled) {
    a = std::max(a, kctx.pack_nr);
  }
  if (pair_aligned) {
    a = std::max(a, 2);
  }
  return a;
}

// `kNTileMaxExperts` previously duplicated `kNTilePlanMaxExperts` as
// a separate `= 256` constant for the expert-ordering helpers below.
// Removed to keep `kNTilePlanMaxExperts` (defined in the companion
// planner header `group_matmul_n_tile_planner.hpp`, re-included by
// this file at the top) as the SINGLE source of truth for the N-tile
// expert capacity.  Callers below now reference
// `kNTilePlanMaxExperts` directly so a future bump cannot leave the
// order helpers out of sync with the auto-selector and the planner's
// stack-array sizing.

/// Auto-pick N_ORDER from num_ops.  Returns 3 (pair_balanced) or 0
/// (walk input order); explicit modes 1/2/4 only reachable via env.
///
/// The ordering choice depends on num_ops:
///   * Few-experts (≤ ~kSmallExpertsCutoff): pair-balanced ordering's
///     bin-packing keeps per-round max-M balanced.
///   * Many-experts (≥ ~kLargeExpertsCutoff): pair-balanced applies
///     again.
///   * Mid-band: default to walk-input (lower overhead, zero
///     permutation cost).
/// The cutoffs are calibrated against the dispatcher's stable-N-tile
/// plan and should be re-evaluated together if that planner changes.
inline int auto_pick_n_order(int num_ops) {
  if (num_ops <= 18) {
    return 3;  // few-experts regime — pair-balanced is the better default
  }
  if (num_ops >= 26) {
    return 3;  // many-experts regime — pair-balanced is the better default
  }
  return 0;    // mid-band — walk input order
}

/// Write `out[0..out_size)` with expert indices ordered per
/// `get_grp_matmul_n_order()`.  `out_size = 0` signals "walk input
/// order, ignore out" (auto-mode resolved to no permutation).
///
/// Heap-free: stack array of kNTilePlanMaxExperts for the desc-sort
/// temp; beyond that the ordering is skipped (correct, just unsorted).
/// Mode 4 is O(num_ops²) ≤ 64K comparisons — well under 10 µs.
///
/// `auto_resolved_out` (optional): when env mode = 0, the resolved
/// concrete sub-mode is written here for APILOG diagnostics.
inline void fill_ntile_expert_order(
  int *out, int &out_size, int max_size,
  const std::vector<int> &M, int num_ops,
  int *auto_resolved_out = nullptr) {

  if (num_ops <= 0 || num_ops > max_size
      || num_ops > kNTilePlanMaxExperts) {
    out_size = 0;
    return;
  }

  int order = get_grp_matmul_n_order();

  // Mode 0 — auto: shape-aware sub-mode selection.
  if (order == 0) {
    order = auto_pick_n_order(num_ops);
    if (auto_resolved_out != nullptr) {
      *auto_resolved_out = order;
    }
    if (order == 0) {
      // Auto chose walk-input; leave out empty.
      out_size = 0;
      return;
    }
  }

  // Modes 1 (ascending) and 2 (descending) are direct sorts.
  if (order == 1 || order == 2) {
    const bool ascending = (order == 1);
    sort_indices_by_m(out, num_ops, M, ascending);
    out_size = num_ops;
    return;
  }

  // Mode 3 — pair-balanced: descending sort, then interleave
  // (largest, smallest, 2nd-largest, 2nd-smallest, …) so each round
  // sees a mix of heavy and light experts.
  if (order == 3) {
    std::array<int, kNTilePlanMaxExperts> sorted_desc{};
    sort_indices_by_m(sorted_desc.data(), num_ops, M,
                      /*ascending=*/false);
    int lo = 0, hi = num_ops - 1, o = 0;
    while (lo <= hi) {
      out[o++] = sorted_desc[lo++];
      if (lo <= hi) {
        out[o++] = sorted_desc[hi--];
      }
    }
    out_size = num_ops;
    return;
  }

  // Mode 4 — balanced-spread: at each output position p, pick the
  // remaining (sorted-desc) item whose M brings the running prefix
  // sum closest to the ideal line `y = (p + 1) * total / num_ops`.
  //
  // Result: for any K, splitting the output into K equal-length
  // consecutive chunks yields Σ M per chunk ≈ total / K.  Heavy
  // experts land at positions ≈ i × num_ops / num_heavies, so the
  // round scheduler sees at most ONE heavy expert per CCX slot for
  // typical (batch_size, ccd_size) choices.
  {
    std::array<int, kNTilePlanMaxExperts> sorted_desc{};
    sort_indices_by_m(sorted_desc.data(), num_ops, M,
                      /*ascending=*/false);

    int64_t total = 0;
    for (int i = 0; i < num_ops; ++i) {
      total += M[i];
    }

    std::array<bool, kNTilePlanMaxExperts> used{};  // zero-init
    int64_t cum = 0;
    for (int p = 0; p < num_ops; ++p) {
      // Error metric: |target_scaled − num_ops × new_cum| where
      // target_scaled = (p + 1) × total.  Integer-only; scales
      // cancel out across candidates.
      const int64_t target_scaled =
        static_cast<int64_t>(p + 1) * total;
      int best_j = -1;
      int64_t best_err = std::numeric_limits<int64_t>::max();
      for (int j = 0; j < num_ops; ++j) {
        if (used[j]) {
          continue;
        }
        const int64_t new_cum = cum + M[sorted_desc[j]];
        const int64_t err = std::llabs(
                              target_scaled - static_cast<int64_t>(num_ops) * new_cum);
        if (err < best_err) {
          best_err = err;
          best_j = j;
        }
      }
      used[best_j] = true;
      out[p] = sorted_desc[best_j];
      cum += M[sorted_desc[best_j]];
    }
    out_size = num_ops;
    return;
  }
}

// =====================================================================
// Section A.6 — flat_n_tile() forward declaration
// =====================================================================

/// ALGO 3 — N-tile parallel GEMM, with optional fused-swiglu-oai
/// epilogue.  Defined in group_matmul_n_tile.cpp.
///
/// `gemm_mode_out` (optional) receives a static string naming the
/// concrete path that ran: `"flat_n_tile"`, `"flat_n_tile_custom"`,
/// `"flat_n_tile_fused_swiglu_oai"`, or `"flat_n_tile_fused_swiglu_oai_custom"`.
/// The caller is expected to thread this through to its own
/// gemm_mode_out so benchdnn / profiler output reveals whether the
/// custom BF16 microkernel engaged.
void flat_n_tile(
  const std::vector<char> &layout,
  const std::vector<bool> &transA, const std::vector<bool> &transB,
  const std::vector<int> &M, const std::vector<int> &N,
  const std::vector<int> &K, const std::vector<float> &alpha,
  const std::vector<const void *> &src, const std::vector<int> &lda,
  const std::vector<const void *> &weight, const std::vector<int> &ldb,
  const std::vector<const void *> &bias, const std::vector<float> &beta,
  const std::vector<void *> &dst, const std::vector<int> &ldc,
  const std::vector<bool> &is_weights_const,
  std::vector<matmul_params> &params,
  int num_threads,
  grp_matmul_gated_act_t fused_act = grp_matmul_gated_act_t::none,
  data_type_t act_dtype = data_type_t::none,
  const char **gemm_mode_out = nullptr);

}  // namespace matmul
}  // namespace lowoha
}  // namespace zendnnl

#endif  // ZENDNNL_GROUP_MATMUL_N_TILE_HPP
