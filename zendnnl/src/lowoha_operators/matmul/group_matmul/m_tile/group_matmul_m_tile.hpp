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

/// ALGO 2 — M-tile parallel GEMM, public interface header.
///
/// This header carries the M-tile-specific public surface that the
/// dispatcher and the fused-MoE entry need to compile against.  It is
/// library-internal: not part of the public ZenDNN API and not meant
/// for inclusion outside `src/lowoha_operators/matmul/group_matmul/`
/// (plus the `gtests/group_matmul/` test files that exercise the path
/// tags and env override atoms).
///
/// File layout — mirrors the .cpp's section banners so the two files
/// can be read side-by-side:
///
///   Section H.1  test_api::s_grp_matmul_m_tile_* override atoms.
///                Sentinel-`INT_MIN` (tri-state knobs) or sentinel-`-1`
///                (positive-int knobs) "no-override" atomics that
///                shadow the env-cache for tests.  RAII helpers live
///                in `gtests/group_matmul/moe_test_utils.hpp`.
///
///                NOTE: these are header-only `inline std::atomic<>`
///                variables and are therefore SAME-BINARY only.  They
///                affect the library reads only when the m_tile TU is
///                statically linked into the same image as the caller
///                (e.g. the gtest binary builds the source in-tree).
///                Cross-image use — a host process that loaded a
///                pre-built `libzendnn.so` and tries to twiddle these
///                atoms via this header — would mutate a separate
///                copy in the host's TU and would NOT change the
///                library's behaviour, so this is NOT a production
///                runtime knob.  Use the documented env variables
///                (`ZENDNNL_GRP_MATMUL_*`) for cross-image control.
///
///   Section H.2  test_api::s_capture_m_tile_path / s_last_m_tile_path
///                capture machinery.  Tests arm the capture and read
///                the tag after a dispatcher call to assert which
///                `flat_m_tile` (or `flat_m_tile_pipeline_bf16`)
///                branch fired.  The `m_tile_path_tag::*` named
///                constants themselves live in the companion planner
///                header `group_matmul_m_tile_planner.hpp` (Section
///                P.1) — re-included transitively from this file.
///
///   Section H.3  M-tile env getters: `get_grp_matmul_m_tile_*()`.
///                Cached-static-const + atomic-override read pattern;
///                see `parse_env_int_strict` (in
///                `group_matmul_parallel_common.hpp`) for the strict
///                env parsing convention.
///
///   Section H.4  Forward declarations for the two M-tile executors:
///                * `flat_m_tile`               (legacy single matmul,
///                                               Section B of the .cpp)
///                * `flat_m_tile_pipeline_bf16` (vertical-fusion
///                                               W13→act→W2, Section C
///                                               of the .cpp)
///
///   Section H.5  M-tile shared eligibility predicate
///                (`check_m_tile_safe`).  Previously lived in
///                `../group_matmul_parallel_common.hpp`; moved here
///                (PR follow-up) so the M-tile-only structural gate
///                sits next to the M-tile executor it protects.  Both
///                the dispatcher and the fused-MoE entry point call
///                it — they include this header anyway to reach
///                `flat_m_tile()` / `flat_m_tile_pipeline_bf16()`.
///
/// What stays in `group_matmul_m_tile.cpp`:
///   * Per-thread slice executors and inline runtime helpers
///     (anonymous namespace).
///   * Definitions for the two executors declared in Section H.4.
///
/// What moved to `group_matmul_m_tile_planner.hpp` (companion header):
///   * `m_tile_path_tag::*` named constants (one per executor branch).
///   * `m_tile_single_tier_plan_t` planner output struct.
///   * `plan_m_tile_single_tier_assignment` (Phase 1b/2/3 planner).
///
/// This header re-includes the planner header so any translation unit
/// that includes the public M-tile interface transitively gets the
/// planner types and path tags — the single-include contract for
/// downstream callers (`group_matmul_dispatch.cpp`,
/// `group_matmul_fused_moe.cpp`, gtests) is preserved.
///
/// Dependency direction:
///   * This header depends on `../group_matmul_parallel_common.hpp`
///     for shared types (`matmul_params`, `grp_matmul_gated_act_t`,
///     `data_type_t`) and the `parse_env_int_strict` helper, and on
///     `group_matmul_m_tile_planner.hpp` for planner types / tags.
///   * `../group_matmul_parallel_common.hpp` does NOT include this
///     header — consumers of M-tile interfaces include this file
///     explicitly.  One-way dependency, same shape as
///     `../n_tile/group_matmul_n_tile.hpp`.

#ifndef ZENDNNL_GROUP_MATMUL_M_TILE_HPP
#define ZENDNNL_GROUP_MATMUL_M_TILE_HPP

#include <atomic>
#include <cstdlib>
#include <limits>
#include <vector>

#include "../group_matmul_parallel_common.hpp"
#include "group_matmul_m_tile_planner.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

// =====================================================================
// Section H.1 — M-tile env override atoms (test_api)
// =====================================================================
//
// Settable atomics that shadow each cached-env knob below.  All live
// in `test_api` because production code never touches them — the env
// cache is the production path; the atoms are flipped by gtests
// (via the RAII helpers in `moe_test_utils.hpp`) and long-running
// services that want to change the knob without re-launching the
// process.
//
// Naming convention (matches the rest of `test_api`):
//   * sentinel `INT_MIN`  — tri-state knobs (-1 / 0 / 1), AND the
//                            scratch-KB knob (which now treats -1 as a
//                            meaningful UNBOUNDED value, so it can no
//                            longer use -1 as its "unset" marker).
//                            Override wins on any value != INT_MIN;
//                            otherwise the cached env is read.
//   * sentinel `-1`       — legacy positive-int knobs.  Override wins
//                            on any value >= 1; otherwise the cached
//                            env is read.
//
// UNBOUNDED scratch-budget sentinel for
// `ZENDNNL_GRP_MATMUL_M_TILE_PIPELINE_SCRATCH_KB`.  Defined in the
// `matmul` namespace (not `test_api`) so both the env getter and the
// executor translation unit name it unqualified.  See the doc-block
// on `s_grp_matmul_m_tile_pipeline_scratch_kb_override` for semantics.
inline constexpr int kMTilePipelineScratchKbUnbounded = -1;

namespace test_api {

// Sentinel `INT_MIN` = no override; falls through to the cached env
// path (which itself applies the documented default 0 = AUTO).
//
// Two-mode dispatch for the M-tile (ALGO 2) planner's light/heavy
// load balancer.  See the doc-block on
// `get_grp_matmul_m_tile_hybrid()` below for the gating heuristic
// and rationale.
//
// Settable values:
//   * INT_MIN — no override (env-cache path).  Production state.
//   * -1      — DISABLED.  Forces the legacy single-tier M-tile
//               planner (floor=1 per active expert + surplus to
//               heaviest).
//   *  0      — AUTO (default).  Enables the multi-tier dispatch
//               when the per-call shape matches the skewed many-
//               expert gating.
//   * any other value → undefined; tests should only use the
//               documented set.
inline std::atomic<int> s_grp_matmul_m_tile_hybrid_override{
    std::numeric_limits<int>::min()};

// ── M-tile (ALGO 2) heuristic-constant overrides ─────────────────────
//
// These four sentinel-`-1` atomics back the env-tunable surface for
// the M-tile planner's hard-coded constants.  They exist so the
// planner's baked-in thresholds can be exercised and tuned in tests
// without editing production defaults.  Each atomic shadows a getter
// declared below the file's main test_api block, using the same
// "non-negative override wins; cached env otherwise" pattern as
// `s_grp_matmul_custom_kernel_n_tile_override`.
//
//   * `s_grp_matmul_m_tile_slice_target_override`         → kSliceTarget=16
//   * `s_grp_matmul_m_tile_hybrid_min_max_m_override`     → kHybridMinMaxM=256
//   * `s_grp_matmul_m_tile_hybrid_min_skew_override`      → kHybridMinSkewX=4
//   * `s_grp_matmul_m_tile_hybrid_lights_per_thread_override` → kLightsPerThread=8
//
// Settable values are positive ints; any value < 1 (including the
// sentinel `-1`) falls through to the cached env path.  RAII helpers
// live in `gtests/group_matmul/moe_test_utils.hpp` so tests can flip
// these mid-process without re-launching the process.
inline std::atomic<int> s_grp_matmul_m_tile_slice_target_override{-1};
inline std::atomic<int> s_grp_matmul_m_tile_hybrid_min_max_m_override{-1};
inline std::atomic<int> s_grp_matmul_m_tile_hybrid_min_skew_override{-1};
inline std::atomic<int> s_grp_matmul_m_tile_hybrid_lights_per_thread_override{-1};

// ── Vertical fusion (MoE FFN W13 → gated act → W2) dispatch knob ─────
//
// Three-mode dispatch (matches the M-tile hybrid style; settable
// values are -1 / 0 / 1).  Default `INT_MIN` = no override (caller
// reads the env-cache; production state).
//
//   -1 DISABLED — force legacy two-pass (W13+act then W2) regardless
//                  of eligibility.  Used by tests to capture the
//                  pre-fusion baseline.
//    0 AUTO     — engage vertical fusion when ALL eligibility gates
//                  in `try_flat_m_tile_pipeline_bf16` pass (one of:
//                  bf16 end-to-end OR WOQ-INT4 s4/u4 on BOTH halves;
//                  supported gated activation; single-tier planner
//                  outcome; scratch budget admits slice_M ≥ 1 for
//                  every active expert).  This is the production
//                  default once the gate lands.
//    1 FORCED   — engage vertical fusion even when the planner would
//                  prefer two-pass (e.g. when scratch is unusually
//                  tight on huge `I`).  Reserved for testing /
//                  benchmarking; production should not set this.
//
// Settable via `ZENDNNL_GRP_MATMUL_M_TILE_VERTICAL_FUSION={-1,0,1}`
// (env), `MoEVerticalFusionOverride` RAII helper (gtests) or this
// atomic directly (long-running services).  All other values fall
// through to the documented default (0 / AUTO).
//
// Naming: the `M_TILE_` infix matches the sibling M-tile knobs
// (`M_TILE_HYBRID`, `M_TILE_SLICE_TARGET`, `M_TILE_HYBRID_*`).  The
// vertical fusion path is an additive M-tile (ALGO 2) executor — it
// is NOT a separate ALGO, it slots into the M-tile branch dispatch
// at `flat_m_tile_dispatch` (see `group_matmul_m_tile.cpp`).  Grouping
// the knob name under `M_TILE_*` keeps the env namespace flat and
// signals scope at a glance.
inline std::atomic<int> s_grp_matmul_m_tile_vertical_fusion_override{
    std::numeric_limits<int>::min()};

// ── Vertical fusion per-thread scratch budget (KB) ──────────────────
//
// Caps `slice_M` per expert so the thread-local `(slice_M × 2·I)`
// bf16 staging buffer fits in L2 with headroom for the W13 and W2
// weight blocks loaded by the inner matmul kernels.  Default 512 KB
// matches the Zen 4 / Zen 5 per-core L2 capacity (1 MB) split
// roughly 50/50 between staging and weight footprint.  Settable
// via `ZENDNNL_GRP_MATMUL_M_TILE_PIPELINE_SCRATCH_KB={-1, 1..N}` (env)
// or this atomic.
//
// Accepted values:
//   *  `>= 1`  — explicit per-thread budget in KB.
//   *  `-1`    — UNBOUNDED (`kMTilePipelineScratchKbUnbounded`): the
//                scratch budget gate is disabled so vertical fusion
//                engages regardless of per-thread slice size.  Use to
//                force the fused path on large-M / prompt frames that
//                otherwise bail on the budget.  WARNING: when the
//                staging tile exceeds L2 the DRAM-avoidance premise of
//                vertical fusion no longer holds (the tile spills to
//                L3/DRAM and the single-thread-per-slice GEMM is less
//                efficient than the legacy multi-threaded two-pass) —
//                this is a testing / experimentation knob, not a
//                production default.
//   *  any other value (0, < -1) — falls through to the cached default.
//
// `M_TILE_` infix matches the sibling M-tile knobs — see the
// rationale on `s_grp_matmul_m_tile_vertical_fusion_override`.
//
// On hosts with smaller L2 (e.g. legacy Zen 2 / 3 at 512 KB / core)
// callers should lower to ~256 KB; on c-class large-L2 parts the
// default leaves headroom and a larger value (1024 KB+) can be
// tried.  The planner emits zero `slice_M` for any expert whose
// `2 · I[e] · sizeof(bf16)` already exceeds the budget — those
// experts cause the eligibility gate to fail and the call falls
// back to legacy two-pass for the whole batch (unless the budget is
// UNBOUNDED, in which case the budget gate never fires).
//
// The "no override" sentinel is `std::numeric_limits<int>::min()`
// (NOT -1, which now carries the UNBOUNDED meaning) — mirrors
// `s_grp_matmul_m_tile_vertical_fusion_override`.  The UNBOUNDED
// sentinel value itself is `matmul::kMTilePipelineScratchKbUnbounded`
// (defined in the enclosing `matmul` namespace so the env getter and
// the executor `.cpp` can both name it unqualified).
inline std::atomic<int> s_grp_matmul_m_tile_pipeline_scratch_kb_override{
    std::numeric_limits<int>::min()};

// =====================================================================
// Section H.2 — M-tile path tag capture machinery (test_api)
// =====================================================================
//
// ── M-tile (ALGO 2) branch-tag capture hook ─────────────────────────────
//
// `flat_m_tile` dispatches between four internal branches based on
// workload shape:
//   round-based            (active_ops > num_threads)
//   multi-tier hybrid      (skewed many-expert Qwen3-class gate)
//   wide-N memory-bound    (total_need * 2 ≤ num_threads, max_M > 1)
//   phase-2 single-tier    (default M-weighted fallthrough)
//
// Three additional tags, `kVerticalFusionBF16`, `kVerticalFusionWOQ`,
// and `kVerticalFusionDQINT8`, are set by the vertical-fusion executor
// `flat_m_tile_pipeline_bf16` when it commits to the fused
// W13→gated-act→W2 path (engaged by the
// `group_matmul_fused_moe_execute` dispatcher when its eligibility
// gate passes and `ZENDNNL_GRP_MATMUL_M_TILE_VERTICAL_FUSION ∈ {0,1}`).
// The three differ only in the per-stage dtype work routed by the
// shared per-thread slice plan:
//   * `kVerticalFusionBF16`   — bf16 weights both halves.
//   * `kVerticalFusionWOQ`    — int4 (s4/u4) weights both halves
//                               (bf16 src + bf16 dst still).
//   * `kVerticalFusionDQINT8` — s8 weights both halves with per-token
//                               symmetric DQ-INT8 on both Op1 and Op2
//                               (bf16 src + bf16 dst; Op1 src
//                               bf16→s8 hoisted ONCE per expert
//                               pre-OMP, Op2 src re-quantized per
//                               slice in the new Stage 2b between
//                               activation and W2).  All extra
//                               buffers are RAII-owned at dispatcher
//                               scope — no per-slice mallocs.
// Tests use these tags to assert vertical fusion actually engaged on
// shapes it was designed for, and that it fell back to legacy
// two-pass (any of the four `flat_m_tile` tags) on shapes where the
// gate fails.
//
// Tests need to assert *which* branch fired on a given shape so the
// gating heuristic can evolve without silently regressing the
// out-of-the-box auto policy.  This is the same capture-gated atomic
// pattern as `s_capture_gemm_mode` / `s_last_group_matmul_direct_*`
// in `group_matmul_parallel_common.hpp`: tests arm the flag via
// `MTilePathCaptureGuard` in `moe_test_utils.hpp` and read the tag
// after the dispatcher call.
//
// CAPTURE GATE — `s_capture_m_tile_path` (atomic bool, default false):
//   Production builds never arm this, so the store path inside each
//   branch short-circuits on a single relaxed load of a cache-line-
//   shared `false` value — no coherence traffic, branch-predictable.
//   Without this gate, four unconditional stores on the hot M-tile
//   path would invalidate the tag's cache line on every prompt /
//   decode call, taxing concurrent dispatcher invocations across
//   multi-rank serving deployments that have no use for the hook.
//
// Tag values are exported as named constants in
// `test_api::m_tile_path_tag::*` in the companion planner header
// `group_matmul_m_tile_planner.hpp` (Section P.1) so the call sites
// in `group_matmul_m_tile.cpp` stay readable and test expectations
// stay type-safe (no magic numbers).  This file already re-includes
// the planner header, so any TU that pulls in
// `group_matmul_m_tile.hpp` sees the tags.
inline std::atomic<bool> s_capture_m_tile_path{false};
inline std::atomic<int>  s_last_m_tile_path{-1};

}  // namespace test_api

// =====================================================================
// Section H.3 — M-tile env getters
// =====================================================================
//
// All getters use the standard cached-static-const + atomic-override
// pattern.  Each call performs one relaxed atomic load + branch on
// the test override, then returns either the override or the cached
// static-const env value.  The env `getenv` / parse happens only
// once on first use; every subsequent call is the atomic-load +
// branch + cached-const read (a few nanoseconds on x86_64,
// comfortably below noise in the planner's hot path).  Invalid env
// values (non-numeric, ≤ 0 where positive is required) fall back to
// the documented default — see `parse_env_int_strict` in
// `group_matmul_parallel_common.hpp` for the strict env parsing
// contract.

// `ZENDNNL_GRP_MATMUL_M_TILE_HYBRID` = { -1, 0 } — cached, default 0.
//
//   * -1 DISABLED — single-tier planner only (M-weighted CCD-stripe).
//   *  0 AUTO    — multi-tier engages when the per-call gate matches:
//         * `max_M >= kHybridMinMaxM (=256)` (skip decode-class)
//         * `max_M >= kHybridMinSkewX (=4) × avg_M` (skewed)
//         * `num_light >= num_threads / 8`  (enough light to
//                                            free meaningful
//                                            heavy-pool budget)
//       where `light_cut = max(8, avg_M / 4)` and `num_light`
//       is the count of active experts with `M[e] <= light_cut`.
//       Light experts share a small dedicated thread team
//       (`light_pool = min(cores_per_ccd, ceil(num_light / 8))`)
//       via round-robin (each thread runs full-M, team=1 on a
//       stride of the light list); the remaining
//       `num_threads − light_pool` threads run the standard
//       M-weighted proportional-scale + decrement Phase-2 logic
//       over heavy experts only.  Falls back silently to
//       single-tier when the call doesn't match the gating
//       (few actives, decode shapes, low-skew, etc.).
//
// Invalid values (< -1, "abc", etc.) → silently treated as default
// (0 / AUTO), matching the strict-parse convention of the other
// ZENDNNL_GRP_MATMUL_* env vars.
//
// The test-override atomic
// `test_api::s_grp_matmul_m_tile_hybrid_override` is the canonical
// way to flip this mid-process from gtests; the static-const
// env-cache is a one-shot read taken at first call to keep the
// production hot path branch-predictor-friendly.
inline int get_grp_matmul_m_tile_hybrid() {
  constexpr int kDefault = 0;  // AUTO
  const int ovr = test_api::s_grp_matmul_m_tile_hybrid_override
      .load(std::memory_order_relaxed);
  if (ovr != std::numeric_limits<int>::min()) return ovr;
  static const int v = []() {
    const char *e = std::getenv("ZENDNNL_GRP_MATMUL_M_TILE_HYBRID");
    int parsed = 0;
    if (!parse_env_int_strict(e, parsed)) return kDefault;
    // Accept only -1 (DISABLED) or 0 (AUTO).  Reject anything else
    // and fall back to default.
    return (parsed == -1 || parsed == 0) ? parsed : kDefault;
  }();
  return v;
}

// ── M-tile heuristic-constant tuning knobs (F8 from the ALGO 2 review) ──
//
// These four getters expose the previously-hard-coded heuristic
// constants in `group_matmul_m_tile.cpp` so production deployments
// can tune them on new MoE workload shapes without source edits.
// All four use the standard cached-static-const + atomic-override
// pattern; every call does one relaxed atomic load + branch to
// check the test override, then returns either the override or
// the cached static-const env value.  The env `getenv` / parse
// happens only once on first use; every subsequent call is the
// atomic-load + branch + cached-const read (a few nanoseconds on
// x86_64, comfortably below noise in the planner's hot path).
// Invalid env values (non-numeric, ≤ 0) fall back to the
// documented default.
//
// Production behaviour is unchanged: every default matches the
// original literal constant.  These knobs are tuning hatches, not
// behavioural changes.
//
//   * `ZENDNNL_GRP_MATMUL_M_TILE_SLICE_TARGET` (default 16)
//     Minimum rows per M-tile thread.  Raising the value gives
//     each thread more work (better arith / memory amortisation,
//     fewer threads engaged); lowering it spreads work thinner
//     (better balance on shallow-M shapes).  The original
//     `kSliceTarget = 16` matches AOCL DLP's row-block-quantum.
//
//   * `ZENDNNL_GRP_MATMUL_M_TILE_HYBRID_MIN_MAX_M` (default 256)
//     Multi-tier engagement gate: `max_M ≥ this` is required for
//     the multi-tier hybrid to fire.  Lowering it engages
//     multi-tier on shallower workloads; raising it restricts
//     multi-tier to heavier shapes only.  Set very high to
//     effectively disable multi-tier without setting
//     `ZENDNNL_GRP_MATMUL_M_TILE_HYBRID=-1` (which also kills the
//     code path entirely; this knob just makes the gate harder
//     to pass).
//
//   * `ZENDNNL_GRP_MATMUL_M_TILE_HYBRID_MIN_SKEW` (default 4)
//     Multi-tier engagement gate: `max_M ≥ skew × avg_M`.  Lower
//     values let multi-tier engage on less-skewed workloads
//     (closer to uniform-M); higher values restrict multi-tier
//     to extremely skewed shapes only.
//
//   * `ZENDNNL_GRP_MATMUL_M_TILE_HYBRID_LIGHTS_PER_THREAD` (default 8)
//     Light-pool packing density: `light_pool = min(cores_per_ccd,
//     ceil(n_light / this))`.  Higher values pack more lights per
//     light-pool thread (smaller light pool, larger heavy pool);
//     lower values give the light pool more threads.
inline int get_grp_matmul_m_tile_slice_target() {
  constexpr int kDefault = 16;
  const int ovr = test_api::s_grp_matmul_m_tile_slice_target_override
      .load(std::memory_order_relaxed);
  if (ovr >= 1) return ovr;
  static const int v = []() {
    const char *e =
        std::getenv("ZENDNNL_GRP_MATMUL_M_TILE_SLICE_TARGET");
    int parsed = 0;
    if (!parse_env_int_strict(e, parsed)) return kDefault;
    return (parsed >= 1) ? parsed : kDefault;
  }();
  return v;
}

// ── ALGO 2 (M-tile) regime classifier ──────────────────────────────────
// ALGO 2 (`flat_m_tile`) is a PURE M-tile executor (multi-tier hybrid +
// Phase-2 single-tier).  The two non-M-tile regimes it historically
// handled via internal fallbacks — the wide-N memory-bound fallback
// (sequential full-team, ALGO-1-equivalent) and the experts-exceed-threads
// regime (per-expert, ALGO-5-equivalent) — are no longer chosen inside
// flat_m_tile for AUTO.  This classifier lets `auto_select_algo` (ALGO 0)
// detect those regimes at SELECTION time and route them to the dedicated
// algos (ALGO 1 / ALGO 5) so AUTO reproduces the executor flat_m_tile used
// to pick internally.  The gates mirror flat_m_tile's old internal gates
// EXACTLY (same kSliceTarget, same total_need / max_M math) so the routing
// is parity-preserving, not coincidental:
//   * kManyExperts — `active_ops > num_threads`.  A pure M-tile plan cannot
//                    give < 1 thread per active expert, so this regime is
//                    M-tile-INFEASIBLE; AUTO routes it to ALGO 5.
//   * kWideN       — `max_M > 1 && total_need*2 <= num_threads`, where
//                    `total_need = Σ_active min(M[i], ceil(M[i]/kSliceTarget))`.
//                    M is too shallow to feed the slicer; AUTO routes to
//                    ALGO 1 (sequential full-team).
//   * kMTile       — everything else (multi-tier or single-tier M-tile).
enum class m_tile_regime { kMTile, kWideN, kManyExperts };

inline m_tile_regime classify_m_tile_regime(const std::vector<int> &M,
                                            int num_threads) {
  const int kSliceTarget = get_grp_matmul_m_tile_slice_target();
  int active_ops = 0;
  int max_M = 0;
  int64_t total_need = 0;
  for (int m : M) {
    if (m <= 0) continue;
    ++active_ops;
    if (m > max_M) max_M = m;
    total_need += std::min<int64_t>(
        m, std::max<int64_t>(
               1, (static_cast<int64_t>(m) + kSliceTarget - 1) / kSliceTarget));
  }
  if (active_ops > num_threads) return m_tile_regime::kManyExperts;
  if (max_M > 1 && total_need * 2 <= static_cast<int64_t>(num_threads))
    return m_tile_regime::kWideN;
  return m_tile_regime::kMTile;
}

inline int get_grp_matmul_m_tile_hybrid_min_max_m() {
  constexpr int kDefault = 256;
  const int ovr = test_api::s_grp_matmul_m_tile_hybrid_min_max_m_override
      .load(std::memory_order_relaxed);
  if (ovr >= 1) return ovr;
  static const int v = []() {
    const char *e =
        std::getenv("ZENDNNL_GRP_MATMUL_M_TILE_HYBRID_MIN_MAX_M");
    int parsed = 0;
    if (!parse_env_int_strict(e, parsed)) return kDefault;
    return (parsed >= 1) ? parsed : kDefault;
  }();
  return v;
}

inline int get_grp_matmul_m_tile_hybrid_min_skew() {
  constexpr int kDefault = 4;
  const int ovr = test_api::s_grp_matmul_m_tile_hybrid_min_skew_override
      .load(std::memory_order_relaxed);
  if (ovr >= 1) return ovr;
  static const int v = []() {
    const char *e =
        std::getenv("ZENDNNL_GRP_MATMUL_M_TILE_HYBRID_MIN_SKEW");
    int parsed = 0;
    if (!parse_env_int_strict(e, parsed)) return kDefault;
    return (parsed >= 1) ? parsed : kDefault;
  }();
  return v;
}

inline int get_grp_matmul_m_tile_hybrid_lights_per_thread() {
  constexpr int kDefault = 8;
  const int ovr =
      test_api::s_grp_matmul_m_tile_hybrid_lights_per_thread_override
          .load(std::memory_order_relaxed);
  if (ovr >= 1) return ovr;
  static const int v = []() {
    const char *e = std::getenv(
        "ZENDNNL_GRP_MATMUL_M_TILE_HYBRID_LIGHTS_PER_THREAD");
    int parsed = 0;
    if (!parse_env_int_strict(e, parsed)) return kDefault;
    return (parsed >= 1) ? parsed : kDefault;
  }();
  return v;
}

// ZENDNNL_GRP_MATMUL_M_TILE_VERTICAL_FUSION = { -1, 0, 1 } — cached,
// default -1 (DISABLED).
//
// Three-mode dispatch for the MoE FFN vertical-fusion (W13 → gated
// act → W2 per M-tile slice) path.  See the doc-block on
// `s_grp_matmul_m_tile_vertical_fusion_override` above for the per-
// mode semantics and the `test_api` override contract.  Strict env
// parsing — anything other than exactly `"-1"`, `"0"`, `"1"` falls
// back to the documented default (NOT silently mode-0 via legacy
// atoi-returns-0-for-junk).
//
// Production callers should leave this unset (DISABLED).  Setting
// `ZENDNNL_GRP_MATMUL_M_TILE_VERTICAL_FUSION=-1` is the kill switch
// when a per-shape regression is discovered in the field; setting
// `=1` is reserved for testing the FORCED engagement against the
// planner's AUTO heuristic.
inline int get_grp_matmul_m_tile_vertical_fusion() {
  constexpr int kDefault = -1;  // DISABLED
  const int ovr = test_api::s_grp_matmul_m_tile_vertical_fusion_override
      .load(std::memory_order_relaxed);
  if (ovr != std::numeric_limits<int>::min()) return ovr;
  static const int v = []() {
    const char *e = std::getenv(
        "ZENDNNL_GRP_MATMUL_M_TILE_VERTICAL_FUSION");
    int parsed = 0;
    if (!parse_env_int_strict(e, parsed)) return kDefault;
    return (parsed >= -1 && parsed <= 1) ? parsed : kDefault;
  }();
  return v;
}

// ZENDNNL_GRP_MATMUL_M_TILE_PIPELINE_SCRATCH_KB = { -1, 1..N } —
// cached, default 512.
//
// Per-thread scratch budget (in KB) for the vertical-fusion pipeline
// executor.  Caps `slice_M` per expert so the staging buffer fits in
// L2 alongside the inner matmul weight tile.  See doc-block on
// `s_grp_matmul_m_tile_pipeline_scratch_kb_override` above for the
// per-microarchitecture sizing guidance AND the `-1` UNBOUNDED
// semantics.
//
// Return contract:
//   *  `>= 1`  — explicit budget in KB.
//   *  `-1`    — `kMTilePipelineScratchKbUnbounded`: budget gate
//                disabled (caller treats it as an infinite budget so
//                the per-thread slice never bails on size).
//   *  default — any other input (0, < -1, non-numeric, unset) falls
//                back to 512.
//
// Strict env parsing — only exactly `"-1"` or a positive integer is
// honoured; everything else uses the default.
inline int get_grp_matmul_m_tile_pipeline_scratch_kb() {
  constexpr int kDefault = 512;
  const int ovr = test_api::s_grp_matmul_m_tile_pipeline_scratch_kb_override
      .load(std::memory_order_relaxed);
  if (ovr != std::numeric_limits<int>::min())
    return (ovr >= 1 || ovr == kMTilePipelineScratchKbUnbounded)
        ? ovr : kDefault;
  static const int v = []() {
    const char *e = std::getenv(
        "ZENDNNL_GRP_MATMUL_M_TILE_PIPELINE_SCRATCH_KB");
    int parsed = 0;
    if (!parse_env_int_strict(e, parsed)) return kDefault;
    return (parsed >= 1 || parsed == kMTilePipelineScratchKbUnbounded)
        ? parsed : kDefault;
  }();
  return v;
}

// =====================================================================
// Section H.4 — M-tile executor forward declarations
// =====================================================================
//
// Both executors are defined in `group_matmul_m_tile.cpp` (Sections B
// and C respectively) and called from:
//   * `group_matmul_dispatch.cpp` — dispatcher invokes `flat_m_tile`
//     when ALGO 2 wins the per-call route.
//   * `group_matmul_fused_moe.cpp` — `try_run_fused_moe_m_tile_
//     pipeline_bf16` invokes `flat_m_tile_pipeline_bf16` when the
//     vertical fusion eligibility gate passes.

/// ALGO 2 — M-tile parallel GEMM (legacy single-matmul executor).
/// Defined in `group_matmul_m_tile.cpp` (Section B).
///
/// `gemm_mode_out` (optional): the executor writes the concrete branch it
/// actually ran — one of `"flat_m_tile_multitier"`, `"flat_m_tile_single_tier"`,
/// `"flat_m_tile_seq_clamp"` (the many-experts sequential-full-team clamp,
/// which is ALGO-1 behaviour), or `"flat_m_tile_skip"` (no-op early return:
/// empty call, `num_threads <= 0`, or no active expert — nothing executed;
/// maps to `exec_algo=0`) — so the post-exec `[GRP_MATMUL.CALL]` line and
/// benchdnn/profiler output reveal the real path (mirrors `flat_n_tile`).
void flat_m_tile(
  const std::vector<char> &layout,
  const std::vector<bool> &transA, const std::vector<bool> &transB,
  const std::vector<int> &M, const std::vector<int> &N,
  const std::vector<int> &K, const std::vector<float> &alpha,
  const std::vector<const void *> &src, const std::vector<int> &lda,
  const std::vector<const void *> &weight, const std::vector<int> &ldb,
  const std::vector<const void *> &bias, const std::vector<float> &beta,
  const std::vector<void *> &dst, const std::vector<int> &ldc,
  grp_matmul_gated_act_t fused_act, data_type_t act_dtype,
  const std::vector<bool> &is_weights_const,
  std::vector<matmul_params> &params,
  int num_threads,
  const char **gemm_mode_out = nullptr);

/// ALGO 2 — vertical-fusion pipeline (W13 → gated act → W2) at
/// M-tile slice granularity.  Tri-regime: BF16 end-to-end, WOQ-INT4
/// (s4 / u4 weights), and DQ-INT8 per-token symmetric (s8 weights
/// with bf16 src/dst).  Defined in `group_matmul_m_tile.cpp`
/// (Section C).  See the doc-block on the function for the
/// engagement contract, return semantics, and per-regime memory
/// management notes.
///
/// Returns `true` when the pipeline engaged AND completed
/// successfully (caller skips legacy two-pass); `false` when it
/// declined to engage (round-based / multi-tier / wide-N regime,
/// scratch budget exhausted, per-thread alloc failure, or — in the
/// DQ-INT8 regime — pre-OMP source-quant hoist failure) — in
/// which case dst_w2 and dst_w13 are guaranteed to be untouched
/// and the caller MUST run the legacy two-pass.
bool flat_m_tile_pipeline_bf16(
  const std::vector<char> &layout,
  const std::vector<bool> &transA,
  const std::vector<bool> &transA_w2,
  const std::vector<bool> &transB,
  const std::vector<int> &M,
  const std::vector<int> &N_w13,
  const std::vector<int> &K_in,
  const std::vector<float> &alpha_w13,
  const std::vector<const void *> &src, const std::vector<int> &lda,
  const std::vector<const void *> &weight_w13,
  const std::vector<int> &ldb_w13,
  const std::vector<const void *> &bias_w13,
  const std::vector<float> &beta_w13,
  const std::vector<void *> &dst_w13,
  const std::vector<int> &ldc_w13,
  bool dst_w13_is_caller_alloc,
  const std::vector<int> &N_w2,
  const std::vector<int> &K_w2,
  const std::vector<float> &alpha_w2,
  const std::vector<const void *> &weight_w2,
  const std::vector<int> &ldb_w2,
  const std::vector<const void *> &bias_w2,
  const std::vector<float> &beta_w2,
  const std::vector<void *> &dst_w2,
  const std::vector<int> &ldc_w2,
  grp_matmul_gated_act_t fused_act, data_type_t act_dtype,
  const std::vector<bool> &is_weights_const,
  std::vector<matmul_params> &params_w13,
  std::vector<matmul_params> &params_w2,
  int num_threads);

/// Eligibility-gated wrapper around `flat_m_tile_pipeline_bf16`.
///
/// Returns `true` iff ALL of the following hold, in which case the
/// pipeline has produced both Op1 (W13) and Op2 (W2) outputs and the
/// caller MUST NOT re-run a legacy two-pass over the same buffers:
///
///   * Env knob `ZENDNNL_GRP_MATMUL_M_TILE_VERTICAL_FUSION != -1`
///     (cheapest gate — short-circuits before any data inspection).
///   * BOTH params_w13[0] and params_w2[0] are in the SAME dtype
///     regime, where the eligible regimes are:
///       (A) BF16 end-to-end — `src/wei/dst == bf16` and
///           `dynamic_quant == false`.
///       (B) WOQ-INT4 — `src/dst == bf16`, `wei ∈ {s4, u4}`,
///           `dynamic_quant == false`, `quant_params.wei_scale.buff
///           != nullptr`, and `is_weights_const[*] == true` for
///           every active expert (the AOCL DLP WOQ fast path caches
///           the dequant prepack on the const-weight side).
///       (C) DQ-INT8 per-token symmetric — `src/dst == bf16`,
///           `wei == s8`, `compute == s8`, `dynamic_quant == true`,
///           `quant_params.src_scale.dims == {M[i], 1}`,
///           `src_scale.dt == f32`, `quant_params.wei_scale.buff
///           != nullptr` (per-channel wei scale required by the
///           AOCL DLP s8s8 → bf16 kernel).  Symmetric only —
///           asymmetric u8 src_zp is rejected at the `compute`
///           dtype check.
///     A mixed-regime call (e.g. BF16 W13 + WOQ-INT4 W2, or BF16
///     W13 + DQ-INT8 W2) is rejected because the executor's
///     scratch uses a single staging element size AND the Stage 2b
///     re-quant runs only in the DQ-INT8 arm; mixed-half scratch
///     sizing / Stage-2b plumbing is not in scope for the unified
///     executor today.
///   * `fused_act` is in the supported set: `none`, `silu_and_mul`,
///     `gelu_and_mul`, or `swiglu_oai_mul` (the four activations
///     `apply_gated_act_inplace` handles on the thread-local scratch).
///   * Both Op1 and synthesized Op2 matmul shapes pass
///     `check_m_tile_safe` (row-major, dtype uniformity, no packed-B,
///     etc. — see the predicate's doc-block in Section H.5 below).
///   * `flat_m_tile_pipeline_bf16` itself committed (single-tier
///     planner agreed, scratch budget fits including any DQ-INT8
///     Stage 2b bytes, per-thread alloc succeeded, and — in the
///     DQ-INT8 regime — the pre-OMP per-expert src hoist succeeded
///     for every active expert).
///
/// When this returns `false`, NO writes have been made to either
/// `dst_w13` or `dst_w2`, so the caller can safely fall through to a
/// legacy two-pass dispatch over the same buffers.
bool try_flat_m_tile_pipeline_bf16(
  const std::vector<char> &layout,
  const std::vector<bool> &transA,
  const std::vector<bool> &transA_w2,
  const std::vector<bool> &transB,
  const std::vector<int> &M,
  const std::vector<int> &N_w13,
  const std::vector<int> &K_in,
  const std::vector<float> &alpha_w13,
  const std::vector<const void *> &src, const std::vector<int> &lda,
  const std::vector<const void *> &weight_w13,
  const std::vector<int> &ldb_w13,
  const std::vector<const void *> &bias_w13,
  const std::vector<float> &beta_w13,
  const std::vector<void *> &dst_w13,
  const std::vector<int> &ldc_w13,
  bool dst_w13_is_caller_alloc,
  const std::vector<int> &N_w2,
  const std::vector<int> &K_w2,
  const std::vector<float> &alpha_w2,
  const std::vector<const void *> &weight_w2,
  const std::vector<int> &ldb_w2,
  const std::vector<const void *> &bias_w2,
  const std::vector<float> &beta_w2,
  const std::vector<void *> &dst_w2,
  const std::vector<int> &ldc_w2,
  grp_matmul_gated_act_t fused_act, data_type_t act_dtype,
  const std::vector<bool> &is_weights_const,
  std::vector<matmul_params> &params_w13,
  std::vector<matmul_params> &params_w2,
  int num_threads);

// =====================================================================
// Section H.5 — M-tile shared eligibility predicate
// =====================================================================
//
// M-tile (ALGO 2) eligibility predicate.  Returns `true` when every
// active expert satisfies the structural invariants the M-tile
// per-thread slicer requires: row-major layout, cross-expert dtype
// uniformity (src/wei/dst/bias), no non-default mem_format flags,
// no packed-B (the M-tile row slicer doesn't unpack GGML in-thread),
// no softmax / pooling post-ops (which require the full M block),
// and — when `dynamic_quant` is set — per-token / per-group-on-K
// source-quant granularity (`src_scale.dims[0] == M[i]`) so the
// per-thread reorder operates on its own rows without racing.
//
// Previously a private static helper in `group_matmul_dispatch.cpp`;
// hoisted to inline in `group_matmul_parallel_common.hpp` so BOTH the
// legacy dispatcher (`group_matmul_run_parallel_dispatch`) AND the
// MoE vertical-fusion dispatcher fork
// (`group_matmul_fused_moe_execute`) could use the same predicate
// for engaging an M-tile-based path on Op1 / Op2.  Now moved here
// (PR follow-up) so the M-tile-only structural gate sits next to the
// M-tile executor it protects — both callers already include this
// header anyway.
//
// The check is O(num_ops) × a small constant of field comparisons —
// trivially cheap in the hot path.
//
// Dynamic-quant row-locality rationale: per-thread reorder of the
// source quant tensor only races if rows assigned to different
// threads share a quant-param row — per-token / per-group-on-K
// granularity guarantees `dims[0] == M[i]` so the slicer's row
// partition aligns with the quant-param partition.  Per-channel
// (`dims[0] == 1`, K-broadcast) and per-group-on-M
// (`dims[1] == ngroups > 1`) both share a single source quant entry
// across multiple rows, so two threads' slices would write to the
// same scale buffer index — hence rejected here.  The `M[i] == 1`
// decode-class case is handled identically by the predicate: the
// per-thread reorder degenerates to a one-row reorder per expert,
// which is always race-free regardless of granularity.
inline bool check_m_tile_safe(
    const std::vector<char> &layout,
    const std::vector<int> &M,
    const std::vector<matmul_params> &params,
    int num_ops) {
  // Dtype-uniformity reference = the FIRST ACTIVE expert, not params[0].
  // The grouped / per-expert fallback DQ pre-pass rewrites ONLY active
  // experts to s8 (inactive M==0 experts keep their pre-quant bf16/f32
  // dtype), so a leading inactive expert at index 0 would otherwise become
  // a bf16 reference that every active s8 expert mismatches — falsely
  // flipping m_tile_safe to false and vetoing ALGO 2/3 on the common MoE
  // decode case.  Fall back to 0 when all experts are inactive (no
  // compute, so the result is irrelevant).
  int ref = 0;
  for (int i = 0; i < num_ops; ++i) { if (M[i] > 0) { ref = i; break; } }

  // Iterate the active range only — the framework may pad params[]
  // (and layout[]) past `num_ops` for prepack-extras tail metadata
  // that the matmul-processing loop never reaches.  See doc-block
  // on `params[i].active_matmul` / `total_matmul` for the contract.
  for (int i = 0; i < num_ops; ++i) {
    // Inactive experts (M==0) do no compute and carry no rewritten quant
    // metadata; skip them so they cannot veto the whole call.
    if (M[i] == 0) continue;
    if (layout[i] != 'r' && layout[i] != 'R') return false;
    if (params[i].dtypes.src  != params[ref].dtypes.src)  return false;
    if (params[i].dtypes.wei  != params[ref].dtypes.wei)  return false;
    if (params[i].dtypes.dst  != params[ref].dtypes.dst)  return false;
    if (params[i].dtypes.bias != params[ref].dtypes.bias) return false;
    if (params[i].mem_format_a != 'n') return false;
    if (params[i].mem_format_b != 'n') return false;
    if (params[i].packing.pack_format_b != 0) return false;
    if (params[i].dynamic_quant) {
      const auto &sd = params[i].quant_params.src_scale.dims;
      if (sd.empty() || sd[0] != static_cast<int64_t>(M[i])) return false;
      const auto &zd = params[i].quant_params.src_zp.dims;
      if (!zd.empty() && zd[0] != static_cast<int64_t>(M[i])) return false;
    }
    for (const auto &po : params[i].postop_) {
      if (po.po_type == post_op_type_t::softmax
          || po.po_type == post_op_type_t::pooling) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace matmul
}  // namespace lowoha
}  // namespace zendnnl

#endif  // ZENDNNL_GROUP_MATMUL_M_TILE_HPP
