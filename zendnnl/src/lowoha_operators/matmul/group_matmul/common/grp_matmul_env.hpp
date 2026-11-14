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

/// Group-matmul environment knobs: strict env parsing, the ALGO selector,
/// the shared test_api override atoms and the generic cached getters.

#ifndef ZENDNNL_GRP_MATMUL_ENV_HPP
#define ZENDNNL_GRP_MATMUL_ENV_HPP

#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <limits>

#include "lowoha_operators/matmul/group_matmul/common/grp_matmul_constants.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

// ──────────────────────────────────────────────────────────────────────
// Env-driven feature flags.
// ──────────────────────────────────────────────────────────────────────

/// Parse `e` as a base-10 integer with strict validation.
///
/// Returns `true` only when the string is non-empty, parses to a full
/// numeric value (no trailing junk like `"1abc"` or `"abc"`), did not
/// overflow `long`, and fits in `int`.  On success, writes the parsed
/// value into `out`.  On any failure mode `out` is left untouched and
/// the function returns `false` — callers should then fall back to
/// the documented default for the env knob.
///
/// Rationale: the legacy `std::atoi(e)` pattern silently returns `0`
/// for non-numeric inputs (e.g. `"abc"` → 0).  For env knobs whose
/// documented default is NOT `0` (e.g. N_ORDER default 3, N_ROUNDS
/// default 1, N_TILE_STRATEGY default 3) `atoi` would coincidentally
/// pick mode 0 — a valid value but NOT the documented default the
/// user intended when they typo'd the env value.  Strict validation
/// makes "invalid env value → fall back to documented default" the
/// observable behaviour.
inline bool parse_env_int_strict(const char *e, int &out) {
    if (e == nullptr || e[0] == '\0') return false;
    char *end = nullptr;
    errno = 0;
    const long v = std::strtol(e, &end, 10);
    if (end == e) return false; // no digits consumed
    if (*end != '\0') return false; // trailing junk (e.g. "1abc")
    if (errno == ERANGE) return false; // overflowed long
    if (v < static_cast<long>(std::numeric_limits<int>::min())
            || v > static_cast<long>(std::numeric_limits<int>::max()))
        return false;
    out = static_cast<int>(v);
    return true;
}

/// ZENDNNL_GRP_MATMUL_ALGO selects AUTO (0), a generic scheduler
/// ({1,2,3,5,6}), or the W8A8 whole-call interceptor (4). Single-digit
/// parsing preserves the historical first-byte behaviour: e.g. `"5xyz"`
/// returns 5 because only the first byte is inspected.
///
/// Cached + override pattern (matches `get_grp_matmul_auto_prompt_algo`).
/// Return the process-requested group-matmul selector, including W8A8 ALGO 4.
///
/// ALGO 4 is an additive W8A8 fused-MoE fast-path request. Generic group
/// matmul sees it as AUTO (0) after an eligibility decline. Main's generic
/// fused-MoE path accepts both BF16 and caller-prequantized S8.
///
/// The cached `static const` snapshot of `std::getenv` is taken on the first
/// call; the override atomic (sentinel `-1` = no override) lets gtests flip
/// the requested value mid-process.
inline std::atomic<int> &test_api_algo_override();
inline int get_grp_matmul_requested_algo() {
    const int ovr = test_api_algo_override().load(std::memory_order_relaxed);
    if (ovr >= 0)
        return is_grp_matmul_requested_algo(ovr) ? ovr : kGrpMatmulAlgoAuto;
    static const int v = []() {
        const char *env = std::getenv("ZENDNNL_GRP_MATMUL_ALGO");
        const int requested = (env != nullptr) ? (env[0] - '0') : -1;
        return is_grp_matmul_requested_algo(requested) ? requested
                                                       : kGrpMatmulAlgoAuto;
    }();
    return v;
}

/// True when `ZENDNNL_GRP_MATMUL_ALGO` is explicitly `0`: AUTO with the
/// single-expert prompt left on AOCL-DLP (ALGO 1) while its decode stays on
/// the custom kernel.  Unset, AUTO runs a dense single-expert layer on the
/// custom kernel in both phases.  An `AlgoEnvGuard` pin is plain AUTO.
inline std::atomic<int> &test_api_algo_legacy_override();
inline bool get_grp_matmul_algo_legacy() {
    const int ovr
            = test_api_algo_legacy_override().load(std::memory_order_relaxed);
    if (ovr >= 0) return ovr != 0;
    if (test_api_algo_override().load(std::memory_order_relaxed) >= 0)
        return false;
    static const bool v = []() {
        const char *env = std::getenv("ZENDNNL_GRP_MATMUL_ALGO");
        return env != nullptr && env[0] == '0';
    }();
    return v;
}

/// Generic scheduler selection. ALGO 4 is deliberately normalized to AUTO:
/// it selects only the private W8A8 MoE hook and is not a sixth generic
/// scheduler.
inline int get_grp_matmul_algo() {
    const int requested = get_grp_matmul_requested_algo();
    return requested == kGrpMatmulAlgoNTileFlatParallel ? kGrpMatmulAlgoAuto
                                                        : requested;
}

/// Legacy global-only predicate retained for tests and callers that only need
/// to inspect `ZENDNNL_GRP_MATMUL_ALGO`. Production interception uses the
/// phase-aware resolver below.
inline bool get_grp_matmul_ntile_flat_parallel() {
    return get_grp_matmul_requested_algo() == kGrpMatmulAlgoNTileFlatParallel;
}

// ZENDNNL_GRP_MATMUL_N_ROUNDS = { 0, 1, 2, 3 } — cached, default 1.
//   ALGO 3 FixedTeamRounds round-mode selection.  Internal tuning knob.
//     0 = auto: planner picks single-round / multi-round / balanced
//               via cost-model on wall time.
//     1 = force single-round (all experts in one round, n_thr =
//         num_threads / num_ops).  When single-round is infeasible
//         (num_threads < num_ops) the fallback is the SAME
//         {multi, balanced} cost-model comparison AUTO would run —
//         not an unconditional balanced pick — so mode 1 degrades to
//         a two-way cost model rather than to one fixed strategy.
//         CURRENT DEFAULT: production
//         sweeps showed single-round dominates on the target MoE
//         envelope at high thread counts; the auto cost-model
//         occasionally picked balanced/multi-round at boundaries
//         where single-round was within noise but had simpler
//         cache-key behaviour.  Pin to single-round so the
//         planner's choice is shape-independent and the AOCL DLP
//         per-tile cache key set stays stable across decode
//         iterations.  Set "0" to restore the original auto
//         behaviour for comparisons or shape exploration.
//     2 = force multi-round legacy fixed-shape (batch experts × ccd_size
//         threads each, possibly wasteful tail round).
//     3 = force balanced (n_rounds = ceil(num_ops / target_batch),
//         experts evenly redistributed across rounds).
//   Mid-process env changes have no effect; relaunch to change it.
// ── Test-only overrides for cached env getters ─────────────────────
//
// The N_ROUNDS / CUSTOM_KERNEL / CUSTOM_KERNEL_N_TILE getters cache
// their value at first call (`static const`) so production reads are
// branch-predictor-friendly.  That precludes a unit test that runs
// AFTER another test has already cached a non-default value from
// flipping the cached value back via `setenv` — the getter returns
// the cached snapshot regardless.
//
// The atomics below let a test override the cached value on the
// production read path (one relaxed-load + branch per getter call,
// negligible vs the surrounding planner / OMP work).  Sentinel `-1`
// means "use the cached env path" (production default).  Tests
// should set the override via the RAII helpers in
// `gtests/group_matmul/moe_test_utils.hpp` to guarantee the override
// is cleared on scope exit, including on test failure / fixture
// teardown.
namespace test_api {
inline std::atomic<int> s_grp_n_rounds_mode_override {-1};
inline std::atomic<int> s_grp_matmul_custom_kernel_override {-1};
// Fused-MoE arena knob.  Needed because the getter latches in a
// `static const`, so a test that flips the env with setenv changes
// nothing and silently compares a configuration against itself.
// Sentinel `-1` = no override (env / default).
inline std::atomic<int> s_grp_matmul_fused_moe_tight_override {-1};
// ALGO 3 fused-epilogue knob.  Same latching problem as the arena knob
// above.  Sentinel `-1` = no override (env / default).
inline std::atomic<int> s_grp_n_tile_fused_act_override {-1};
// DQ-INT8 CK sub-knob — independent from the master CK switch so
// tests / deployments can toggle the int8 fast path without
// disturbing the bf16 path.  Sentinel `-1` = no override (env / default).
inline std::atomic<int> s_grp_matmul_custom_kernel_int8_override {-1};
// FP16 CK sub-knob — independent from the master CK switch and the
// int8 sub-knob so tests / deployments can A/B the native
// AVX-512-FP16 fast path in isolation.  Sentinel `-1` = no override
// (env / default).
inline std::atomic<int> s_grp_matmul_custom_kernel_f16_override {-1};
// NOTE: `s_grp_matmul_custom_kernel_n_tile_override` and
// `s_grp_n_tile_strategy_override` moved to `group_matmul_n_tile.hpp`
// (Section A.3) together with the rest of the N-tile override atoms.
// Include that header to access them.

// Sentinel `-1` = no override.  Settable values: 0 (auto / unset),
// 32 (NR=32), 64 (NR=64).  Override semantics in
// `get_grp_matmul_custom_kernel_nr()`:
//   * any negative value (including the `-1` sentinel and any other
//     negative typo) → fall through to the cached env path, so test
//     code never accidentally pins NR via a bogus negative.
//   * any non-negative value other than 32 / 64 → clamped to 0 by
//     the getter, matching the env-parse "validate or treat as
//     unset" behaviour.
inline std::atomic<int> s_grp_matmul_custom_kernel_nr_override {-1};

// Sentinel `-1` = no override. Settable values: 0 (explicit legacy
// 3-rule cascade), {1,2,3,5,6} (force the matching generic ALGO), and 4
// (request the matching phase's W8A8 whole-call attempt). Override semantics
// live in the unified `grp_matmul_auto_phase_setting` getters:
//   * any negative value (including the `-1` sentinel) → fall
//     through to the cached env path (which itself applies the
//     documented defaults — 2 for prompt, 3 for decode).
//   * 0          — explicit legacy 3-rule cascade selection.
//                  Production deployments that want pre-default-flip
//                  behaviour use this (or the env equivalent
//                  `AUTO_*_ALGO=0`).
//   * 1,2,3,5,6  — adopted as the override value.
//   * 4          — retained as the raw W8A8 request while generic routing
//                  sees the inherited default (2 prompt / 3 decode).
//   * > 6        — invalid: inherits the default and is not a policy pin.
inline std::atomic<int> s_grp_matmul_auto_prompt_algo_override {-1};
inline std::atomic<int> s_grp_matmul_auto_decode_algo_override {-1};

// NOTE: `s_grp_matmul_dense_decode_ntile_override`,
// `s_grp_matmul_single_expert_*_override` and their `test_api_*_override()`
// accessors moved to `single_expert/single_expert_policy.hpp` (same
// `test_api` namespace).

// NOTE: `s_grp_matmul_{decode,prompt}_algo5_gate_override` moved to
// `multi_expert/expert_parallel/group_matmul_expert_parallel_policy.hpp` (same `test_api`
// namespace).  Include that header to access them.

// Sentinel `-1` = no override (use the cached env path, default ON).
// `0` / `1` force `ZENDNNL_ENABLE_GROUP_DQ` off / on.  Needed because
// `get_grp_matmul_enable_group_dq()` latches its env read, so a plain
// `setenv` after any prior library call is invisible; a test that wants
// the per-expert DQ fallback must use this.
inline std::atomic<int> s_grp_matmul_enable_group_dq_override {-1};

// Sentinel `-1` = no override (use the cached env path).  `0` / `1` force
// `ZENDNNL_GRP_MATMUL_KBLOCK` off / on, and `2` restores the automatic
// policy, so a gtest can A/B the K-blocked kernel against the straight-line
// one inside a single process — the getter caches its env read, so the env
// var alone cannot flip the path once latched.
inline std::atomic<int> s_grp_matmul_kblock_override {-1};

// Sentinel `-1` = no override (use cached env path).  Settable values
// 0..6 mirror the requested selector surface (`0` = AUTO,
// `{1,2,3,5,6}` = forced generic ALGO_N, `4` = W8A8 whole-call hook,
// invalid values clamped to AUTO by the getter).  The
// `AlgoEnvGuard` RAII helper in `gtests/group_matmul/moe_test_utils.hpp`
// sets the env-var AND stores into this atomic so that any gtest using
// `AlgoEnvGuard(N)` continues to flip the effective algo mid-process —
// without paying the `std::getenv` cost on every production call site.
inline std::atomic<int> s_grp_matmul_algo_override {-1};
// Override of `get_grp_matmul_algo_legacy` (-1 = none, 0 = off, 1 = on).
inline std::atomic<int> s_grp_matmul_algo_legacy_override {-1};

// NOTE: `s_grp_matmul_n_tile_heavy_threshold_override` moved to
// `group_matmul_n_tile.hpp` (Section A.3) together with the rest of
// the N-tile override atoms.  Include that header to access it.

// NOTE: The M-tile (ALGO 2) override atoms — `s_grp_matmul_m_tile_*`
// for hybrid / slice_target / hybrid_min_max_m / hybrid_min_skew /
// hybrid_lights_per_thread / vertical_fusion / pipeline_scratch_kb
// — used to live here.  They moved out to
// `group_matmul_m_tile.hpp` (Section H.1) together with the M-tile
// path-tag capture machinery and the matching `get_grp_matmul_m_tile_*`
// getters, so all M-tile-specific knobs live next to the executor
// declarations they configure.  Consumers (gtests, dispatcher, fused
// MoE entry) include `group_matmul_m_tile.hpp` directly — mirror of
// the `group_matmul_n_tile.hpp` pattern.

// Sentinel `-1` = no override.  Settable values: 0 (per-expert
// subtile sizing OFF — use one m_max-sized `subtile_cols` for every
// active expert), 1 (ON — populate `subtile_cols_per_expert[e]`
// individually).  Override semantics in
// `get_grp_matmul_custom_kernel_subtile_per_expert()`:
//   * any negative value (including the `-1` sentinel) → fall
//     through to the cached env path.
//   * 0 → ON returns false; 1 (or any other positive value) → ON
//     returns true.  Mirrors the env-parse "0 means off, anything
//     else means on" convention of `get_grp_n_tile_fused_act()`.
inline std::atomic<int> s_grp_matmul_custom_kernel_subtile_per_expert_override {
        -1};

// Gated-activation form knobs (`get_grp_matmul_gelu_tanh`,
// `get_grp_matmul_fast_act`).  Sentinel `-1` = no override (env / default);
// 0 / 1 = off / on.
inline std::atomic<int> s_grp_matmul_gelu_tanh_override {-1};
inline std::atomic<int> s_grp_matmul_fast_act_override {-1};

// Last `execution_path` string set by `group_matmul_direct` on a
// successful return.  Read-only inspection hook for tests that need
// to verify which executor path actually ran (e.g. asserting the
// custom BF16 microkernel engaged vs the call falling back to AOCL
// DLP or the Sequential strategy that bypasses CK entirely).
//
// Strings come from `n_tile_execute`'s `execution_path_label` (defined in
// `group_matmul_n_tile.cpp`) or the per-algo executor labels.  They
// are static literals — never freed — so the atomic stores a stable
// pointer that test code can read after the call returns.
//
// CAPTURE GATE — `s_capture_execution_path` (atomic bool, default false):
//   Production builds never set this flag, so the store path in
//   `group_matmul_direct` short-circuits on a single relaxed load
//   of a cache-line-shared `false` value (no coherence traffic).
//   Tests arm the flag (via `ExecutionPathCaptureGuard` in
//   `moe_test_utils.hpp`) for the test's scope, in which case the
//   gated store DOES fire and writes through to the atomic below.
//   Without this gate the unconditional store would mark its
//   cache line Modified on every successful dispatcher call,
//   forcing a coherence ping-pong across any cores running
//   concurrent `group_matmul_direct` invocations — a measurable
//   tax on multi-rank serving deployments that have no use for
//   the test hook.  Same pattern as `s_capture_phase_b` (see
//   `group_matmul_n_tile.hpp`).
inline std::atomic<bool> s_capture_execution_path {false};
inline std::atomic<const char *> s_last_group_matmul_direct_execution_path {
        nullptr};

// Whether a deep-K K-blocked custom-kernel tile ran (BF16 or symmetric INT8).
// Gated by its own capture flag because the decision sits per TILE, not per
// call: an unconditional store would write a shared line from every worker on
// every tile.  The gating load is of a flag that only a test ever flips, so it
// stays in shared state and costs nothing in production.
inline std::atomic<bool> s_capture_kblock {false};
inline std::atomic<bool> s_last_kblock_used {false};

// NOTE: The M-tile (ALGO 2) branch-tag capture hook —
// `s_capture_m_tile_path`, `s_last_m_tile_path`, and the
// `m_tile_path_tag::*` named constants — moved out to
// `group_matmul_m_tile.hpp` (Section H.2).  See the file header
// there for the same capture-gate rationale that used to live here.
} // namespace test_api

// Out-of-namespace accessors for the AUTO_*_ALGO override atomics.
// Forward-declared above the getters (which are defined inline near
// the top of this header, above the `test_api` namespace block) so
// the getter doesn't depend on header-ordering between its own
// definition and the override atomic.  Same single-relaxed-load
// pattern as the other `test_api::*` consumers.
inline std::atomic<int> &test_api_auto_prompt_algo_override() {
    return test_api::s_grp_matmul_auto_prompt_algo_override;
}
inline std::atomic<int> &test_api_auto_decode_algo_override() {
    return test_api::s_grp_matmul_auto_decode_algo_override;
}
inline std::atomic<int> &test_api_algo_override() {
    return test_api::s_grp_matmul_algo_override;
}
inline std::atomic<int> &test_api_algo_legacy_override() {
    return test_api::s_grp_matmul_algo_legacy_override;
}
// NOTE: `test_api_{decode,prompt}_algo5_gate_override()` moved to
// `multi_expert/expert_parallel/group_matmul_expert_parallel_policy.hpp` together with the
// atoms they return.  Include that header to call them.

inline int get_grp_n_rounds_mode() {
    const int ovr = test_api::s_grp_n_rounds_mode_override.load(
            std::memory_order_relaxed);
    if (ovr >= 0) return ovr;
    // Default: 1 (single-round).  Strict env parsing — anything that
    // is not exactly `"0"`, `"1"`, `"2"`, or `"3"` falls back to the
    // documented default (NOT silently to mode 0 via the legacy
    // atoi-returns-0-for-junk behaviour).  See `parse_env_int_strict`.
    static constexpr int kDefault = 1;
    static const int v = []() {
        const char *e = std::getenv("ZENDNNL_GRP_MATMUL_N_ROUNDS");
        int parsed = 0;
        if (!parse_env_int_strict(e, parsed)) return kDefault;
        return (parsed >= 0 && parsed <= 3) ? parsed : kDefault;
    }();
    return v;
}

// NOTE: `get_grp_n_tile_strategy()` (ZENDNNL_GRP_MATMUL_N_TILE_STRATEGY)
// moved to `group_matmul_n_tile.hpp` (Section A.4) together with the
// rest of the N-tile env getters.  Include that header to call it.
// The full three-mode (auto / decode-force / rounds-force) doc-block
// lives at the new home.

// `kDecodeTileAbOn` documents the production decode-tile-AB
// behaviour as an unconditional constant: when max_M ≤ kDecodeMaxM,
// ProportionalRounds/FixedTeamRounds use kDecodeNTile (256) instead of kMinNTile
// (512) as the per-thread N-tile bound — doubles max_n_thr for
// decode-shape down_proj.  Was previously a getter
// (`get_grp_n_decode_tile_ab`) preserving function shape for a
// future experimental env knob, but it never read any env and has
// stayed unconditionally `true` since introduction.  Replaced with
// a `constexpr` so the compiler folds it at the call site
// (`group_matmul_n_tile.cpp::plan_group_n_tile`); a future env knob
// can be re-introduced cleanly by replacing this constant with the
// getter when needed.
inline constexpr bool kDecodeTileAbOn = true;

// NOTE: `get_grp_matmul_n_order()` (ZENDNNL_GRP_MATMUL_N_ORDER) moved
// to `group_matmul_n_tile.hpp` (Section A.4) together with the rest
// of the N-tile env getters.  Include that header to call it.

// ZENDNNL_GRP_MATMUL_FUSED_MOE_TIGHT = { "0", "1" } — cached, default 1.
//   Fused MoE Op1 → act → Op2 arena layout: tight [M, I] when 1,
//   wide [M, 2I] when "0".  Tight halves Op2's src DRAM traffic.
//   The dispatcher
//   only engages tight when ALL of: act is a CK-fusible gated
//   activation (swiglu_oai_mul, silu_and_mul, gelu_and_mul — see
//   `n_tile_can_fuse_activation` for the live predicate; silu and gelu require
//   `CUSTOM_KERNEL=1` because only the CK in-register epilogue
//   implements them), Op1 internal-alloc on, ALGO 3 selected (auto
//   or env-forced), shape-adaptive picker agrees — otherwise falls
//   back to wide silently regardless of this env.  Note:
//   ZENDNNL_GRP_MATMUL_N_TILE_FUSED_ACT is NOT a tight-engagement
//   gate — when the fused-MoE picker hands the dispatcher a tight
//   destination (ldc < N), the dispatcher auto-enables ALGO 3 fused
//   activation regardless of N_TILE_FUSED_ACT (tight is a
//   correctness constraint on the writer, not a perf toggle).  See
//   `should_use_tight_intermediate` in group_matmul_fused_moe.cpp for
//   the full predicate.  Set "0" here to force wide (debug /
//   layout-regression bisection).
inline int get_grp_matmul_fused_moe_tight() {
    const int ovr = test_api::s_grp_matmul_fused_moe_tight_override.load(
            std::memory_order_relaxed);
    if (ovr >= 0) return ovr;
    static const int v = []() {
        const char *e = std::getenv("ZENDNNL_GRP_MATMUL_FUSED_MOE_TIGHT");
        if (e == nullptr || e[0] == '\0') return 1; // default: force-tight
        return (e[0] == '0') ? 0 : 1;
    }();
    return v;
}

// ZENDNNL_GRP_MATMUL_CROSS_WARM = { "0", "1" } — cached, default ON.
//   When ON, each `prepack_for_algo_X` opportunistically populates the
//   cache regime that auto-select would route the OTHER phase to in the
//   same process, so a deployment that fires only prompt during warmup
//   still arrives at decode with both regimes warm — no first-decode-
//   call prepack spike.  An ALGO-3 MoE layer runs both unset phases on
//   ALGO 3, so for it this only fires when an
//   `AUTO_{PROMPT,DECODE}_ALGO` pin splits them
//   (`grp_matmul_cross_warm_target_reachable`).
//
//   Auto-select-only: cross-warm fires exclusively under
//   `ZENDNNL_GRP_MATMUL_ALGO=0` (AUTO).  When a single ALGO is pinned
//   ({1,2,3,5,6}) the same scheduling path serves every call, so the
//   cross-warm target regime (which belongs to a DIFFERENT ALGO) would
//   never be queried — the helper short-circuits and the pinned path
//   prepacks only what it itself uses.  A pinned-ALGO fallback (e.g. an
//   unsafe-shape safety clamp to ALGO 1) takes a one-time lazy reorder
//   on first miss instead — not performant, but correct and bounded.
//
//   Cross-warm decision is `ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL`-aware:
//
//     * `CK=1` (custom-kernel on, production decode path):
//         - prompt → ALGO 1 (full-weight AOCL)
//                  + cross-warm regime 3 (custom-kernel pack)
//         - decode → ALGO 3 + custom kernel (regimes 2 + 3)
//                  + cross-warm regime 1 (full-weight AOCL)
//       Memory cost on many-experts MoE at high thread counts is
//       sizeable (full-weight + custom-kernel pack).  Regime 2
//       (per-tile AOCL) is populated by `prepack_n_tile` when
//       `STABLE_NTILE=1` AND the custom kernel is not eligible; when
//       CK is eligible the per-tile AOCL warm is skipped (the runtime
//       takes the CK path, so those entries would never be queried).
//
//     * `CK=0` (AOCL DLP for both phases):
//         - prompt → ALGO 1 (full-weight AOCL)
//                  + cross-warm regime 2 (per-tile AOCL with nr_align=1)
//         - decode → ALGO 3 + AOCL DLP (regime 2)
//                  + cross-warm regime 1 (full-weight AOCL)
//       Memory cost is dominated by the full-weight + per-tile sum
//       and can be substantial on many-experts MoE at high thread
//       counts.  The cross-warmed regime 2 uses nr_align=1 (Op2
//       non-tight path).  Op1 tight (nr_align=2 under CK=0) still
//       pays a one-time lazy warm on its first decode call — full
//       coverage of both nr_align variants is a separate option
//       (2b) not yet implemented.
//
//   OFF — reverts to the strict per-ALGO regime populated by
//   `prepack_for_algo_X` itself: each ALGO only warms what it would
//   itself use at runtime.  Use this when memory is constrained or
//   to compare against the pre-cross-warm behaviour.
//
//   Independent of `ZENDNNL_GRP_MATMUL_PREPACK`: the master knob
//   short-circuits everything to a no-op; this knob only controls
//   the cross-regime fan-out when the master is ON.
inline bool get_grp_matmul_cross_warm() {
    static const bool v = []() {
        const char *e = std::getenv("ZENDNNL_GRP_MATMUL_CROSS_WARM");
        if (e == nullptr || e[0] == '\0') return true; // default: On
        return e[0] != '0';
    }();
    return v;
}

// ZENDNNL_GRP_MATMUL_PREPACK = { "0", "1" } — cached, default ON.
//   Master switch for the ahead-of-time weight prepack module
//   (group_matmul/prepack/).  Single uniform semantic:
//
//   ON  — each scheduling-ALGO body invokes its matching
//         `prepack_for_algo_X(...)` as the first action, which
//         eagerly warms the inner-kernel weight cache for
//         `p.num_ops_total` experts BEFORE the matmul kicks off.
//         `p.num_ops_total` is resolved by
//         `group_matmul_prepack::build_prepack_params` from the
//         framework-hint fields, in priority order:
//
//           a) `params[0].total_matmul`  when set, or
//           b) `params[0].active_matmul` when set, or
//           c) `M.size()`                 (legacy fall-back).
//
//         By construction `p.num_ops_total >= M.size()` for every
//         supported call pattern, so the warmed set covers every
//         firing expert plus any prepack-extras tail.  Two regimes
//         share this single code path:
//
//           * Framework-hint regime (`params[0].total_matmul >
//             params[0].active_matmul`):  prepack warms the full
//             `total` set, including the prepack-extras tail of
//             experts that aren't firing this call but may fire on
//             a future call (the production MoE rotating-experts
//             use case the module was designed for).
//
//           * Active-only regime (`active_matmul > 0 &&
//             total_matmul == 0`): no rotating-experts hint, so
//             `build_prepack_params` resolves `num_ops_total` to
//             `active_matmul`; the warmer prefills exactly the
//             firing experts and skips the prepack-extras tail.
//
//           * Legacy / no-hint regime (`active=total=0` →
//             `build_prepack_params` resolves both to `M.size()`):
//             prepack warms exactly the firing experts up front.
//             This is a one-time first-iter serial reorder cost
//             paid in exchange for `execute_tile()` cache hits in
//             subsequent iterations of the same configuration
//             (subsequent calls short-circuit via the per-thread
//             fingerprint cache).  Steady-state throughput is
//             identical to the lazy path; first-iter latency is
//             measurably higher (N × reorder_per_expert vs the
//             ~one-reorder parallel-cache-fill the lazy path
//             achieved).  Callers that care about first-iter
//             latency more than they care about steady-state
//             determinism should set this knob to "0".
//
//   OFF — every per-ALGO function short-circuits at entry.  No
//         warm-pack runs.  AOCL DLP / custom-kernel caches still
//         populate lazily inside `run_dlp(...)` / `prepare_for_call`
//         on first miss.  Behaviour is identical to a build without
//         the prepack module compiled in (the original pre-PR
//         library semantics).
//
//   Why default ON: a single coherent semantic is easier to reason
//   about than a conditional gate.  Production deployments that
//   integrate the framework `total_matmul` contract get the
//   prepack-extras benefit out of the box; deployments that don't
//   integrate (legacy / unit tests / single-shot inference) get
//   eager warm-up of the firing experts (small one-time cost) and
//   warm caches for everything afterwards.  Callers that need the
//   strict "no behaviour change vs pre-PR" guarantee set this knob
//   to "0" — that path is also covered by the env-matrix gtests in
//   `group_matmul/prepack/test_prepack.cpp` ([26]-[28]).
inline bool get_grp_matmul_prepack() {
    static const bool v = []() {
        const char *e = std::getenv("ZENDNNL_GRP_MATMUL_PREPACK");
        if (e == nullptr || e[0] == '\0') return true; // default: On
        return e[0] != '0';
    }();
    return v;
}

// ──────────────────────────────────────────────────────────────────────
// Custom microkernel sub-knobs (only consumed when
// ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL=1).  All cached as static const.
// ──────────────────────────────────────────────────────────────────────

// ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_NR = { unset, "32", "64" } — cached.
//   Pack/microkernel NR override.  Auto (unset) → 32 (cleanest
//   register budget, MR=8 on NV=2).  64 doubles N-lanes per zmm at
//   MR cap 6 — worth trying on prompt shapes.  Other values → auto.
//
// Tests can pin the value via `s_grp_matmul_custom_kernel_nr_override`
// (sentinel -1 = no override).  Without the override the env value is
// captured once on first call; later env mutations are invisible —
// the override is the only deterministic way to flip this knob in
// the same process.
inline int get_grp_matmul_custom_kernel_nr() {
    const int ovr = test_api::s_grp_matmul_custom_kernel_nr_override.load(
            std::memory_order_relaxed);
    if (ovr >= 0) { return (ovr == 32 || ovr == 64) ? ovr : 0; }
    // Strict env parsing — non-numeric input (or anything other than
    // exactly "32" / "64") falls back to 0 (auto-pick).
    static const int v = []() {
        const char *e = std::getenv("ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_NR");
        int parsed = 0;
        if (!parse_env_int_strict(e, parsed)) return 0;
        return (parsed == 32 || parsed == 64) ? parsed : 0;
    }();
    return v;
}

// ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_SUBTILE_PER_EXPERT = { "0", "1" } — cached, default OFF.
//   Per-expert L2-friendly subtile_cols (vs. one m_max-sized value
//   for the whole call).  Noise-floor on typical MoE decode shapes;
//   may help on large-L2 hosts or workloads with extreme M variance.
//
// Tests can pin the value via
// `s_grp_matmul_custom_kernel_subtile_per_expert_override`; sentinel
// `-1` falls through to the cached env path.  Without the override
// the env value is captured once on first call (`static const`
// lambda) and later env mutations are invisible — the override is
// the only deterministic way to flip this knob in the same process.
inline bool get_grp_matmul_custom_kernel_subtile_per_expert() {
    const int ovr
            = test_api::s_grp_matmul_custom_kernel_subtile_per_expert_override
                      .load(std::memory_order_relaxed);
    if (ovr >= 0) return (ovr != 0);
    static const bool v = []() {
        const char *e = std::getenv(
                "ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_SUBTILE_PER_EXPERT");
        return (e != nullptr && e[0] != '\0' && e[0] != '0');
    }();
    return v;
}

// ZENDNNL_GRP_MATMUL_KBLOCK = { unset, "0", "1" } — cached, TRI-STATE.
//   unset : automatic policy (`CallContext::kblock_auto` decides).
//   0     : forced OFF — overrides `kblock_auto`, so this is the escape
//           hatch when the K-blocked path is suspected.
//   1     : forced ON wherever the shape gates allow it.
//   Anything else parses as unset, so a typo cannot silently enable a
//   different numerical path.
//   Deep-K single-thread K-blocking for the BF16 and symmetric dynamic-INT8
//   custom kernels (act=none, bias-free).  When ON, the per-tile dispatcher
//   splits the K reduction into L2-resident chunks so each o-block's B strip
//   is streamed ONCE and reused across all M-blocks instead of being
//   re-streamed `ceil(M / max_mr)` times.  Engages only when the strip exceeds
//   the L2 budget and M > max_mr; every other shape and the fused / biased
//   paths are untouched.  `CallContext::kblock_auto` turns it on for
//   single-expert decode without the env.
//   `ZENDNNL_GRP_MATMUL_INT8_KBLOCK={0,1}` is a narrower dispatch-side
//   benchmark override: when set it controls only symmetric INT8 K-blocking,
//   leaving this shared policy (and therefore BF16) unchanged.  When unset,
//   INT8 defers to the shared policy above.  A set
//   `test_api::s_grp_matmul_kblock_override` wins over both env vars.
enum class kblock_env_t { kAuto, kOff, kOn };

inline kblock_env_t get_grp_matmul_kblock_env() {
    const int ovr = test_api::s_grp_matmul_kblock_override.load(
            std::memory_order_relaxed);
    if (ovr >= 0) {
        return ovr == 0 ? kblock_env_t::kOff
                        : (ovr == 1 ? kblock_env_t::kOn : kblock_env_t::kAuto);
    }
    static const kblock_env_t v = []() {
        const char *e = std::getenv("ZENDNNL_GRP_MATMUL_KBLOCK");
        int parsed = 0;
        if (!parse_env_int_strict(e, parsed)) return kblock_env_t::kAuto;
        if (parsed == 0) return kblock_env_t::kOff;
        if (parsed == 1) return kblock_env_t::kOn;
        return kblock_env_t::kAuto;
    }();
    return v;
}

// Resolves the tri-state against the per-call automatic decision.  An
// explicit setting wins in both directions; unset defers to `kblock_auto`.
// Split from the env read so the precedence is testable without the cache.
constexpr bool resolve_grp_matmul_kblock(kblock_env_t env, bool kblock_auto) {
    return env == kblock_env_t::kOff   ? false
            : env == kblock_env_t::kOn ? true
                                       : kblock_auto;
}

inline bool grp_matmul_kblock_enabled(bool kblock_auto) {
    return resolve_grp_matmul_kblock(get_grp_matmul_kblock_env(), kblock_auto);
}

// `ZENDNNL_GRP_MATMUL_INT8_KBLOCK`: -1 unset, 0 / 1 forced off / on for the
// symmetric INT8 path only.  Cached on first read.
inline int get_grp_matmul_int8_kblock_env() {
    static const int v = []() {
        const char *e = std::getenv("ZENDNNL_GRP_MATMUL_INT8_KBLOCK");
        int parsed = 0;
        if (!parse_env_int_strict(e, parsed)) return -1;
        return (parsed == 0 || parsed == 1) ? parsed : -1;
    }();
    return v;
}

// A set INT8 value wins over the shared policy; unset defers to it.  Split
// from the env read so the precedence is testable without the cache.
constexpr bool resolve_grp_matmul_int8_kblock(int int8_env, bool shared) {
    return int8_env == 0 ? false : (int8_env == 1 ? true : shared);
}

// The shared test override wins over the cached INT8 env, so an exported
// benchmark switch cannot pin both arms of an in-process A/B test.
inline bool grp_matmul_int8_kblock_enabled(bool kblock_auto) {
    if (test_api::s_grp_matmul_kblock_override.load(std::memory_order_relaxed)
            >= 0) {
        return grp_matmul_kblock_enabled(kblock_auto);
    }
    return resolve_grp_matmul_int8_kblock(get_grp_matmul_int8_kblock_env(),
            grp_matmul_kblock_enabled(kblock_auto));
}

// ZENDNNL_GRP_MATMUL_GELU_TANH = { "0", "1" } — cached, default OFF.
//   ON: `gelu_and_mul` evaluates GELU in the tanh form
//     0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
//   (PyTorch `gelu(approximate="tanh")`, the GELU of `gelu_pytorch_tanh`
//   models) instead of the erf form, everywhere group_matmul evaluates it:
//   the custom-kernel epilogues of every dtype, single and multi expert,
//   the separate-pass and AOCL-DLP fallbacks, and routed MoE.  The two
//   forms differ by at most 4.8e-4.  Any value other than 0 / 1 keeps the
//   default.
inline bool get_grp_matmul_gelu_tanh() {
    const int ovr = test_api::s_grp_matmul_gelu_tanh_override.load(
            std::memory_order_relaxed);
    if (ovr >= 0) return ovr != 0;
    static const bool v = []() {
        int parsed = 0;
        return parse_env_int_strict(
                       std::getenv("ZENDNNL_GRP_MATMUL_GELU_TANH"), parsed)
                && parsed == 1;
    }();
    return v;
}

// ZENDNNL_GRP_MATMUL_FAST_ACT = { "0", "1" } — cached, default OFF.
//   ON: the gated activations (silu_and_mul, gelu_and_mul, swiglu_oai_mul)
//   and the DQ-INT8 sigmoid post-op take shorter AVX-512 forms wherever
//   group_matmul evaluates them: a degree-3 exp polynomial (max relative
//   error 8.6e-5, against 5.1e-5), sigmoid without the Newton step after
//   `rcp14` (2^-14), and GELU in the tanh form (within 4.8e-4 of the erf
//   form).  The AOCL-DLP fallback keeps its exact SiLU, and routed MoE
//   its SiLU, which is a short form already.  Any value other than 0 / 1
//   keeps the default.
inline bool get_grp_matmul_fast_act() {
    const int ovr = test_api::s_grp_matmul_fast_act_override.load(
            std::memory_order_relaxed);
    if (ovr >= 0) return ovr != 0;
    static const bool v = []() {
        int parsed = 0;
        return parse_env_int_strict(
                       std::getenv("ZENDNNL_GRP_MATMUL_FAST_ACT"), parsed)
                && parsed == 1;
    }();
    return v;
}

// The GELU form of every group_matmul path: tanh under either knob (the
// fast forms have no erf GELU), erf otherwise.
inline bool get_grp_matmul_gelu_is_tanh() {
    return get_grp_matmul_gelu_tanh() || get_grp_matmul_fast_act();
}

// NOTE: `get_grp_matmul_custom_kernel_n_tile()`
// (ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_N_TILE) moved to
// `group_matmul_n_tile.hpp` (Section A.4) together with the rest of
// the N-tile env getters.  Include that header to call it.

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GRP_MATMUL_ENV_HPP
