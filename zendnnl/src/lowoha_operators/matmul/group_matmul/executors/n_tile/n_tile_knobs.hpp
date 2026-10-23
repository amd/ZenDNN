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

/// ALGO 3 N-tile env knobs: the test-only override atoms and the cached
/// `get_grp_n_tile_*` / `get_grp_matmul_n_*` getters.

#ifndef ZENDNNL_GROUP_MATMUL_N_TILE_KNOBS_HPP
#define ZENDNNL_GROUP_MATMUL_N_TILE_KNOBS_HPP

#include <atomic>
#include <cstdlib>
#include <limits>

#include "lowoha_operators/matmul/group_matmul/common/grp_matmul_env.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

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
inline std::atomic<int> s_grp_matmul_custom_kernel_n_tile_override {-1};

// Sentinel `-1` = no override.  Settable values: 0 (auto, default —
// try DecodeD if eligible, fall through to Rounds), 1 (decode —
// prefer DecodeD when its eligibility passes; same behaviour as auto
// today, kept distinct for explicit user intent + apilog hint), 2
// (rounds — skip DecodeD attempt entirely, always run Rounds-based
// FewExperts/ManyExperts).  See `get_grp_n_tile_strategy()` for the
// production env path.
inline std::atomic<int> s_grp_n_tile_strategy_override {-1};

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
inline std::atomic<int> s_grp_matmul_n_tile_heavy_threshold_override {
        std::numeric_limits<int>::min()};

// Sentinel `-1` = no override (falls through to the env, then the auto
// default).  `0` = force OFF (decode uses uniform Phase B base+1), `1` =
// force ON (decode M-proportional split).  RAII helper
// `DecodeProportionalOverride` in `gtests/group_matmul/moe_test_utils.hpp`
// saves/restores it.
inline std::atomic<int> s_grp_matmul_decode_proportional_override {-1};

} // namespace test_api

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
    const int ovr = test_api::s_grp_n_tile_fused_act_override.load(
            std::memory_order_relaxed);
    if (ovr >= 0) return ovr != 0;
    static const bool v = []() {
        const char *e = std::getenv("ZENDNNL_GRP_MATMUL_N_TILE_FUSED_ACT");
        if (e == nullptr || e[0] == '\0') return true; // default: ON
        return e[0] != '0';
    }();
    return v;
}

// ZENDNNL_GRP_MATMUL_N_TILE_STRATEGY = { 0, 1, 2, 3 } — cached, default 3
// (decode_dynamic; see `kGrpNTileStrategyDefault`).
//
// Selects the ALGO 3 (flat_n_tile) per-tile dispatch shape AND
// controls whether the planner's viability perf gate fires.
// Four values with distinct semantics:
//
//     0 = auto (opt-in heuristic).  Honour the planner's viability
//         gate (route to Sequential when `ntile_viable` says N is too
//         narrow to split usefully) unless the custom kernel engaged.
//         If the call survives that gate, try DecodeD WITH its
//         eligibility heuristic (decode-class max_M ≤ 32,
//         num_ops ∈ [6, num_ccds], min_M_active ≥ 3, skew_ratio ≤ 4,
//         max_N / decode_n_tile ≤ team_size_est); fall through to
//         Rounds (FewExperts / ManyExperts) when the heuristic
//         refuses.  Not the default because DecodeD's eligibility gate
//         rarely engages, so the auto branch reduces to "Rounds with
//         extra precondition checks" in practice.
//
//     1 = decode (FORCE).  SKIP the viability gate AND skip the
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
//     2 = rounds (FORCE).  SKIP the viability perf heuristic (same
//         as 1) — when the caller
//         (or ALGO 0 auto-pick) routed to ALGO 3 we mean to run
//         N-tile, not silently bounce back to Sequential on a perf
//         preference.  Skip the DecodeDynamic gate entirely; always run
//         the Rounds path (FewExperts / ManyExperts).  The former
//         production default, kept as the one-knob escape back to
//         deterministic ALGO 3 = "true N-tile with rounds" behaviour on
//         every shape that survives the structural gates.
//
//     3 = decode_dynamic (DEFAULT, FORCE).  SKIP the viability
//         heuristic (same as 1/2).  Per op, a gate
//         decides between the barrier-free CCD-cohesive DecodeDynamic
//         executor (`DecodeDynamic` -> `execute_decode_dynamic`) and
//         Rounds, from the active expert count and per-expert weight
//         (see the DecodeDynamic knob block below).  DecodeDynamic targets the
//         decode-class regime `num_ops > num_ccds` (max_M <= decode
//         threshold); it maps whole experts onto CCDs and processes them
//         in per-CCD waves, so it also runs when `num_ops > num_threads`.
//         Covers the CK custom-kernel path (swiglu fused in-register, so
//         no matmul->activation barrier) plus standard-backend fused /
//         non-fused calls (non-custom wide-fused runs one team-wide
//         barrier + apply_swiglu_oai post-pass inside the executor).
//         A use_custom DQ-INT8 fused call is served too by default; set
//         ZENDNNL_GRP_MATMUL_DECDYN_CK_INT8_FUSED=0 to fall it back to
//         Rounds.  AUTO (value 0) engages DecodeDynamic under
//         the SAME gate (active_ops >= 4*num_ccds).
//
//         PRODUCTION DEFAULT, and a STRICT SUPERSET of value 2: the gate is
//         a narrow, decode-class, MULTI-expert test, and every shape that
//         does not clear it falls through to the exact Rounds planner value
//         2 would have run.  `force_ntile` stays true (it is
//         `strategy != 0`), so changing the default does not reopen the
//         auto-mirror or viability heuristics.  Set `=2` to restore the
//         legacy Rounds-everywhere behaviour.
//
// What survives `n_tile_strategy = {1, 2, 3}` (genuinely STRUCTURAL —
// memory safety / kernel correctness, not perf):
//
//   * R3 — capacity overflow (`num_ops > GroupNTilePlan::kMaxExperts
//     = 256`).  Stack-array bound on the planner; demoting to
//     Sequential is the only safe recourse.  Auto-select rule 0 also
//     captures this upstream by routing to ALGO 1.  `flat_n_tile`
//     slices a larger ALGO-3 MoE call, or one whose weights only the
//     custom kernel can read, so neither reaches the planner.
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
// What is GATED behind `!force_ntile` (an auto-mode-only perf
// heuristic — honoured under env=0, ignored under env={1,2,3}):
//
//   * `!ntile_viable` — heuristic "N too thin for a useful per-
//     thread split".  Under explicit env=1/2/3 the user accepts
//     whatever cost a thin N gives them — we run N-tile and emit a
//     `[GRP_MATMUL.PLAN.HINT]` line so the env-honoured-over-
//     heuristic decision is visible in the L3 trail.
//
// See `plan_group_n_tile` in `group_matmul_n_tile.cpp` for the
// authoritative precedence diagram and emission sites.
//
// Mid-process env changes have no effect (cached static const); tests
// should use `s_grp_n_tile_strategy_override` via the RAII helper
// `NTileStrategyOverride` in `gtests/group_matmul/moe_test_utils.hpp` to
// flip it deterministically inside the same process.
//
// Validation paths differ slightly between the env and the override:
//   * Env path  — invalid values (< 0 OR > 3) parse to the default,
//                 matching the "unset → safe default" convention used by
//                 the other knobs in this header.
//   * Override path — `-1` is the sentinel for "no override" and falls
//                 through to the cached env path; any other negative value
//                 also falls through, so a bogus negative typo cannot
//                 accidentally pin a strategy.  Non-negative values > 3
//                 clamp to 3, mirroring the env path.
inline constexpr int kGrpNTileStrategyDefault = 3;
inline constexpr int kGrpNTileStrategyMax
        = 3; // 0=auto, 1=decode_d, 2=rounds, 3=decode_dynamic

// Resolved ONCE from the env, because two questions are asked of the same
// variable — "which strategy?" and "did the user pick it, or is this just
// the default?" — and a second `getenv` + parse could drift from the first.
// Strict parsing: non-numeric input falls back to the documented default,
// NOT silently to mode 0 via legacy atoi-returns-0 behaviour, and leaves
// `is_set` false so junk reads as "not selected".
struct GrpNTileStrategyEnv {
    int value = kGrpNTileStrategyDefault;
    bool is_set = false;
};
inline const GrpNTileStrategyEnv &grp_n_tile_strategy_env() {
    static const GrpNTileStrategyEnv s = []() {
        const char *e = std::getenv("ZENDNNL_GRP_MATMUL_N_TILE_STRATEGY");
        int parsed = 0;
        if (!parse_env_int_strict(e, parsed)) return GrpNTileStrategyEnv {};
        const bool in_range = (parsed >= 0 && parsed <= kGrpNTileStrategyMax);
        return GrpNTileStrategyEnv {
                in_range ? parsed : kGrpNTileStrategyDefault, true};
    }();
    return s;
}

inline int get_grp_n_tile_strategy() {
    // See the doc-block above for the measured rationale and the
    // precedence diagram in `plan_group_n_tile`.
    const int ovr = test_api::s_grp_n_tile_strategy_override.load(
            std::memory_order_relaxed);
    if (ovr >= 0)
        return (ovr <= kGrpNTileStrategyMax) ? ovr : kGrpNTileStrategyDefault;
    return grp_n_tile_strategy_env().value;
}

// True when the caller EXPLICITLY selected a strategy — via the env or
// the test override — as opposed to inheriting the default.  Mirrors
// `grp_matmul_auto_decode_algo_is_set()`.  Needed because the planner
// logs "DecodeDynamic NOT engaged" diagnostics that are useful when a
// user asked for strategy 3 and wants to know why it did not fire, but
// are pure per-call spam now that 3 is the default and the common case
// is a shape that legitimately falls through to Rounds.
inline bool grp_n_tile_strategy_is_set() {
    if (test_api::s_grp_n_tile_strategy_override.load(std::memory_order_relaxed)
            >= 0)
        return true;
    return grp_n_tile_strategy_env().is_set;
}

// NOTE: the DecodeDynamic knobs (`get_grp_decdyn_*`) moved to
// `multi_expert/multi_expert_policy.hpp`.

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
    static constexpr int kDefault = 3;
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
//   uses either the decode-proportional policy or Phase B according to
//   `ZENDNNL_GRP_MATMUL_DECODE_PROPORTIONAL`.
//
// Three-mode dispatch for asymmetric per-expert thread distribution
// inside the ALGO 3 ManyExperts Single-round plan.  ALL active modes
// are PROMPT-ONLY (`max_M > kDecodeMaxM`); decode-class calls bypass
// the N_TILE heavy-threshold dispatch entirely.  Decode balancing is governed
// by `ZENDNNL_GRP_MATMUL_DECODE_PROPORTIONAL`; this knob only governs the
// PROMPT path.
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
    static constexpr int kDefault = 0; // AUTO (prompt adaptive tiers)
    // Test override sentinel: INT_MIN = no override.  Cannot use `-1`
    // any more since `-1` is now a meaningful (DISABLED) value.
    // Production keeps the static-const env-cache for branch-
    // predictor-friendly reads.
    const int ovr = test_api::s_grp_matmul_n_tile_heavy_threshold_override.load(
            std::memory_order_relaxed);
    if (ovr != std::numeric_limits<int>::min()) return ovr;
    static const int v = []() {
        const char *e
                = std::getenv("ZENDNNL_GRP_MATMUL_N_TILE_HEAVY_THRESHOLD");
        int parsed = 0;
        if (!parse_env_int_strict(e, parsed)) return kDefault;
        // Accept -1 (DISABLED), 0 (AUTO), positive int (MANUAL).  Reject
        // anything more negative — silently clamp to default.
        return (parsed >= -1) ? parsed : kDefault;
    }();
    return v;
}

// ZENDNNL_GRP_MATMUL_DECODE_PROPORTIONAL = { "0", "1" } — cached.
//   Gates the DECODE-class M-proportional per-expert thread split in the
//   ALGO 3 CK Single-round plan (allocate threads to each active expert
//   in proportion to its M, clamped to its N-tile capacity).
//   0:       uniform Phase B base/base+1 distribution.
//   1:       force M-proportional split for decode CK.
//   unset or invalid: default auto policy, enabled only for uniform-N/K
//   decode groups where M is the work-proportional signal.
//   Independent of ZENDNNL_GRP_MATMUL_N_TILE_HEAVY_THRESHOLD (which is
//   prompt-only).  Mid-process env changes have no effect (static const).
//
// Return values:
//   -1 = force OFF, 0 = auto default, 1 = force ON.
inline int parse_grp_matmul_decode_proportional(const char *e) {
    if (e == nullptr || e[0] == '\0') return 0; // default: auto
    int parsed = 0;
    if (!parse_env_int_strict(e, parsed) || (parsed != 0 && parsed != 1)) {
        return 0; // invalid: default auto
    }
    return parsed != 0 ? 1 : -1;
}

inline int get_grp_matmul_decode_proportional_mode() {
    const int ovr = test_api::s_grp_matmul_decode_proportional_override.load(
            std::memory_order_relaxed);
    if (ovr >= 0) return ovr != 0 ? 1 : -1;
    static const int v = []() {
        return parse_grp_matmul_decode_proportional(
                std::getenv("ZENDNNL_GRP_MATMUL_DECODE_PROPORTIONAL"));
    }();
    return v;
}

// ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_N_TILE = { unset, multiple of 32 } — cached.
//   Override the ALGO 3 outer N-tile minimum (default kDecodeNTile=256).
//   Use 128 for high threads/num_ops (more thread participation), 512
//   for prompt-class (wider tiles amortise kernel-call overhead).
//   Non-multiples of 32 → ignored (silently safe vs typos).
inline int get_grp_matmul_custom_kernel_n_tile() {
    const int ovr = test_api::s_grp_matmul_custom_kernel_n_tile_override.load(
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

// NOTE: `get_grp_n_tile_oversub()` (ZENDNNL_GRP_MATMUL_N_TILE_OVERSUB)
// moved to `single_expert/single_expert_policy.hpp`.

// ZENDNNL_GRP_MATMUL_N_TILE_FLOOR = { positive multiple of 32 } — cached,
//   default 32 (= pack_nr).  Lower bound the adaptive N-tile sizer will
//   shrink a per-expert tile to.  A 32 floor lets narrow-N shapes (e.g.
//   N=2048 on 64 cores) saturate every thread with one pack-wide slice
//   each; deep-K down-proj tolerates the narrow tile well.  Raise it to
//   trade parallelism for microkernel efficiency on shallow-K shapes.
//   Non-multiples of 32 (or non-numeric) → default 32.
inline int get_grp_n_tile_floor() {
    static const int v = []() {
        const char *e = std::getenv("ZENDNNL_GRP_MATMUL_N_TILE_FLOOR");
        int parsed = 0;
        if (!parse_env_int_strict(e, parsed)) return 32;
        return (parsed > 0 && (parsed % 32) == 0) ? parsed : 32;
    }();
    return v;
}

// NOTE: `get_grp_n_tile_ccd_fill()` (ZENDNNL_GRP_MATMUL_N_TILE_CCD_FILL)
// moved to `multi_expert/multi_expert_policy.hpp`.

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_N_TILE_KNOBS_HPP
