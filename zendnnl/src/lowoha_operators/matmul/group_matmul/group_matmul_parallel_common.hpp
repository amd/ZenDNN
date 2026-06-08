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

/// Library-internal helpers shared by the ALGO 1..5 parallel paths.
///
/// Each ALGO implementation (`sequential_experts`, `flat_m_tile`,
/// `flat_n_tile`, `parallel_multilevel`, `parallel_per_expert`) is
/// split into its own translation unit so the files stay small and
/// ownership is clear.  This header hosts the bits they all need:
///   - env-driven feature flags
///   - the `resolve_kernel()` / `execute_expert_slice()` primitives
///   - tile-size constants referenced by ALGO 0 auto-select
///   - forward declarations of the per-strategy entry points
///
/// None of these symbols are part of the public ZenDNN API.

#ifndef ZENDNNL_GROUP_MATMUL_PARALLEL_COMMON_HPP
#define ZENDNNL_GROUP_MATMUL_PARALLEL_COMMON_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "custom_kernel/dispatch.hpp"
#include "group_matmul_direct.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"
#include "lowoha_operators/matmul/quantization/reorder_quantization.hpp"
#include "lowoha_operators/common/omp_thread_control.hpp"
#include "operators/matmul/matmul_config.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

// Match the original group_matmul_dispatch.cpp using-declarations so
// source files including this header can refer to matmul_algo_t,
// matmul_config_t, post_op_type_t, etc. without namespace prefixes.
using namespace zendnnl::ops;
using zendnnl::common::size_of;

// Short string renderer for `grp_matmul_gated_act_t` — used by APILOG
// lines and gemm_mode_out strings; keeps activation names consistent
// across executors.
inline const char *act_name(grp_matmul_gated_act_t a) {
  switch (a) {
  case grp_matmul_gated_act_t::none:
    return "none";
  case grp_matmul_gated_act_t::silu_and_mul:
    return "silu_and_mul";
  case grp_matmul_gated_act_t::gelu_and_mul:
    return "gelu_and_mul";
  case grp_matmul_gated_act_t::swiglu_oai_mul:
    return "swiglu_oai_mul";
  }
  return "?";
}

// Tile-size + weight-class constants shared by ALGO 0 auto-select and
// the tile kernels.  Cutoffs separate per-CCD-L3-resident shapes
// (small) from L3-tight shapes (medium) from DRAM-streaming (large).
inline constexpr int    kDecodeMaxM   = 32;                // per-expert M ≤ this → "decode"
inline constexpr int    kMinNTile     = 512;               // prompt-path per-thread N

// DQ-INT8 sibling of `kMinNTile`.  Today set equal to the bf16
// value so the dtype-aware split is a structural no-op; final
// value baked from the D.1 tuning sweep.  Independent constants
// let D.1 retune the int8 family without perturbing bf16 prompt
// thresholds.
inline constexpr int    kMinNTileInt8 = 512;

// NOTE: `kDecodeNTile` (decode-path per-thread N, default 256) now
// lives in `n_tile/group_matmul_n_tile_planner.hpp` (re-included by
// the public N-tile header `n_tile/group_matmul_n_tile.hpp`) with
// the rest of the N-tile-specific constants.  Consumers (dispatcher,
// N-tile executor, custom-kernel dispatch, gtests) include the
// public N-tile header anyway to reach `flat_n_tile()`, so the
// symbol remains visible to them transparently.
//
// NOTE: `kMediumWeight` (64 MB / expert, "referenced by N-tile
// planner") was unused throughout the tree — no production or test
// site read it on either `origin/main` or this branch — so it was
// dropped instead of moved.  Re-introduce only with a measured
// consumer.

/// Few-experts threshold for ALGO 0 auto-select rule 2.  Workloads
/// with `num_ops ≤ kFewExpertsAlgo1` AND `num_ops < num_threads`
/// (rule 1's strict `>=` would otherwise win on a ≤ 8-thread host)
/// pin to ALGO 1 (sequential experts with full-team AOCL DLP) for
/// prompt AND decode.  Targets Mixtral-8x*-class deployments (8
/// experts), where per-expert weight footprint is large enough that
/// the full-team sequential path beats N-tile's column slices on a
/// thin per-expert thread budget.  Bump only with a measured perf
/// justification — most few-expert MoEs in the wild stay at exactly 8.
///
/// The `num_ops < num_threads` precondition is satisfied for every
/// realistic Mixtral deployment (8 experts on 32–128t hosts always
/// hits rule 2).  An 8-expert workload on a ≤ 8-thread host (rare —
/// only seen on local dev / single-CCD profiling) instead falls to
/// rule 1 and routes to ALGO 3.  Documented in `auto_select_algo`'s
/// rule precedence comment in `group_matmul_dispatch.cpp`.
inline constexpr int    kFewExpertsAlgo1 = 8;

/// Few-experts threshold for the ALGO 0 default-policy ALGO 2 preference.
/// When the GLOBAL ALGO env is auto (0) AND neither active phase env is
/// explicitly pinned, a workload whose TOTAL expert count (framework
/// `total_matmul` when set, else the active op count) is `≤
/// kFewExpertsAlgo2Pref` routes to ALGO 2 (flat_m_tile) for BOTH prompt
/// and decode — overriding the per-phase defaults (prompt 2 / decode 3).
/// Targets Mixtral-8x*-class deployments (8 experts), where ALGO 2 is the
/// measured winner across prompt AND decode.  An explicit
/// `AUTO_{PROMPT,DECODE}_ALGO` pin (including `=0` to request the legacy
/// cascade) still wins — this is a DEFAULT refinement, not a hard clamp.
/// Falls back to ALGO 1 when the shape is not m_tile_safe.
inline constexpr int    kFewExpertsAlgo2Pref = 8;

// ── Executed-ALGO from gemm_mode ────────────────────────────────────────
// Maps the executor-written `gemm_mode` string (the authoritative record of
// what ACTUALLY ran) to the ALGO that actually executed (1..5), so the
// post-exec `[GRP_MATMUL.CALL]` line can surface `exec_algo=` alongside
// `mode=`.  Together with the pre-exec `[GRP_MATMUL.ALGO] chosen=` selection
// line, this makes any selection-vs-execution divergence explicit instead of
// silent (e.g. a forced ALGO 2 that clamps to sequential-full-team reports
// `chosen=ALGO_2 ... exec_algo=1 mode=flat_m_tile_seq_clamp`).
//
// Returns 0 for null / unrecognised modes OR for explicit no-op markers
// (`*_skip` — nothing executed), and 1..5 for a recognised executed path.
// So `exec_algo=0` means "no GEMM ran or mode not understood", NOT "ALGO 0".
// ORDER MATTERS: the `flat_m_tile_seq_clamp` special case (ALGO-1 behaviour
// wearing an ALGO-2 mode prefix) and the `*_skip` markers must be checked
// before the generic `flat_m_tile` / `multilevel` / `fused_moe` prefixes
// they share.  Fused composites
// (`fused_moe_*(op1=..,op2=..)`) derive the algo from the Op1 sub-mode; the
// full per-op detail stays in the `mode=` string itself.
inline int executed_algo_from_gemm_mode(const char *mode) {
  if (mode == nullptr) return 0;
  auto starts = [&](const char *p) {
    return std::strncmp(mode, p, std::strlen(p)) == 0;
  };
  // No-op / nothing-executed markers map to 0 (must precede the generic
  // prefixes they share, e.g. "flat_m_tile_skip" before "flat_m_tile").
  if (starts("skip"))                  return 0;  // whole-call no-op (all M<=0)
  if (starts("flat_m_tile_skip"))      return 0;  // empty / no active expert
  if (starts("multilevel_skip"))       return 0;
  if (starts("fused_moe_skip"))        return 0;
  if (starts("flat_m_tile_seq_clamp")) return 1;  // sequential full-team
  if (starts("sequential"))            return 1;  // "sequential" / "..._experts"
  if (starts("flat_m_tile"))           return 2;
  if (starts("vertical_fusion"))       return 2;  // M-tile fused pipeline
  if (starts("flat_n_tile"))           return 3;
  if (starts("multilevel"))            return 4;
  if (starts("per_expert"))            return 5;
  if (starts("fused_moe")) {
    // Composite: derive from the Op1 executor sub-mode.
    const char *op1 = std::strstr(mode, "op1=");
    return (op1 != nullptr) ? executed_algo_from_gemm_mode(op1 + 4) : 0;
  }
  return 0;
}

// NOTE: `kNTilePlanMaxExperts` (max experts the ALGO 3 N-tile planner
// can represent, = 256) now lives in
// `n_tile/group_matmul_n_tile_planner.hpp` (re-included by the
// public N-tile header `n_tile/group_matmul_n_tile.hpp`) with the
// other N-tile-specific constants.  The dispatcher and the
// auto-selector still read it (the rule-0 capacity carve-out routes
// `num_ops > 256` to ALGO 5); both translation units include the
// public N-tile header directly to call `flat_n_tile()`, so the
// symbol remains visible unchanged.

// Op2's K-dimension as a function of the fused activation.  Gated
// activations (swiglu/silu/gelu_and_mul) collapse the [gate, up] pair
// into half the columns, so Op2 sees K_down = N/2.  Without an
// activation Op1's full output flows into Op2, so K_down = N.  The
// caller's `down_weight[i]` must be shaped accordingly:
//   * act != none → [N/2, N_down] row-major (or [N_down, N/2] transB).
//   * act == none → [N,   N_down] row-major (or [N_down, N  ] transB).
//
// Shared between the dispatcher's Phase-F validator
// (`group_matmul_direct.cpp::validate_group_matmul_direct_inputs`)
// and the fused-MoE execute path (`group_matmul_fused_moe.cpp`) so
// both apply the same `ldb_down` minimum.  Was previously a private
// helper in `group_matmul_fused_moe.cpp`'s anonymous namespace; the
// validator was independently using `N[i] / 2` unconditionally,
// which under-restricted `ldb_down` for `act == none` callers.
inline int op2_k_for_act(int n_op1, grp_matmul_gated_act_t act) {
  return (act == grp_matmul_gated_act_t::none) ? n_op1 : (n_op1 / 2);
}

// NOTE: `check_m_tile_safe` (M-tile ALGO 2 structural eligibility
// predicate) moved to `m_tile/group_matmul_m_tile.hpp` (Section H.5)
// so the M-tile-only gate sits next to the M-tile executor it
// protects.  Both callers — the legacy dispatcher
// (`group_matmul_run_parallel_dispatch`) and the MoE vertical-fusion
// dispatcher fork (`group_matmul_fused_moe_execute`) — already
// include `m_tile/group_matmul_m_tile.hpp` to reach `flat_m_tile()`
// / `flat_m_tile_pipeline_bf16()`, so they see the predicate
// unchanged.

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
/// default 1, N_TILE_STRATEGY default 2) `atoi` would coincidentally
/// pick mode 0 — a valid value but NOT the documented default the
/// user intended when they typo'd the env value.  Strict validation
/// makes "invalid env value → fall back to documented default" the
/// observable behaviour.
inline bool parse_env_int_strict(const char *e, int &out) {
  if (e == nullptr || e[0] == '\0') return false;
  char *end = nullptr;
  errno = 0;
  const long v = std::strtol(e, &end, 10);
  if (end == e) return false;        // no digits consumed
  if (*end != '\0') return false;    // trailing junk (e.g. "1abc")
  if (errno == ERANGE) return false; // overflowed long
  if (v < static_cast<long>(std::numeric_limits<int>::min())
      || v > static_cast<long>(std::numeric_limits<int>::max()))
    return false;
  out = static_cast<int>(v);
  return true;
}

/// ZENDNNL_GRP_MATMUL_ALGO = "1".."5" force a specific ALGO, "0"/unset
/// = auto-select.  Strict single-digit validation: only the literal
/// characters `'1'..'5'` (as the FIRST byte, with no further bytes
/// implied here) are honoured.  This is already strict — `"5xyz"`
/// returns 5 because we only inspect the first byte, but no current
/// ZenDNN deployment passes such values and there is no doc-promised
/// behaviour to disagree with.
///
/// Cached + override pattern (matches `get_grp_matmul_auto_prompt_algo`).
/// The cached `static const` snapshot of `std::getenv` is taken on the
/// first call; the override atomic `s_grp_matmul_algo_override` (sentinel
/// `-1` = no override) lets gtests flip the effective value mid-process
/// without paying the `std::getenv` cost on every production call.  The
/// `AlgoEnvGuard` RAII helper in `moe_test_utils.hpp` sets BOTH the env
/// AND the override atomic, so every existing call site that wraps with
/// `AlgoEnvGuard(N)` continues to observe `N` without source-level changes.
inline std::atomic<int> &test_api_algo_override();
inline int get_grp_matmul_algo() {
  const int ovr = test_api_algo_override().load(
      std::memory_order_relaxed);
  if (ovr >= 0) return (ovr >= 1 && ovr <= 5) ? ovr : 0;
  static const int v = []() {
    const char *env = std::getenv("ZENDNNL_GRP_MATMUL_ALGO");
    return (env && env[0] >= '1' && env[0] <= '5') ? (env[0] - '0') : 0;
  }();
  return v;
}

// ── Auto-select per-phase overrides (consulted only under ALGO=0) ─────
//
// The auto-selector (`auto_select_algo` in `group_matmul_dispatch.cpp`)
// classifies every call as either DECODE (`max_M ≤ kDecodeMaxM=32`) or
// PROMPT (otherwise) and picks an ALGO via these two phase envs.  When
// the active phase env is `0`, the legacy 3-rule cascade fires instead
// (Rule 1: num_ops ≥ num_threads → ALGO 3; Rule 2: num_ops ≤ 8 →
// ALGO 1; Rule 3: prompt → ALGO 1, decode → ALGO 3).
//
// IMPORTANT: these envs are ONLY consulted under `ZENDNNL_GRP_MATMUL_
// ALGO=0` (auto).  When the global ALGO env is set to 1..5 the user
// has explicitly pinned that algo for every call regardless of phase;
// the phase envs are never read in that path (see
// `select_grp_matmul_algo` in `group_matmul_dispatch.cpp`).
//
// Defaults — chosen to give a sensible out-of-the-box auto policy
// that captures the measured-best path per phase across the MoE
// envelope we benchmark:
//
//   AUTO_PROMPT_ALGO default = 2 (flat_m_tile, the M-tile planner).
//     With the M-tile multi-tier hybrid (engaged on Qwen3-class
//     skewed prompts, gated by `ZENDNNL_GRP_MATMUL_M_TILE_HYBRID=0`)
//     and the wide-N memory-bound fallback (always-on; engages on
//     `total_need * 2 <= num_threads`), ALGO 2 covers the prompt
//     envelope across the benchmarked MoE workloads:
//       * Qwen3-30B-A3B prompt: multi-tier hybrid path → 2.33×
//         heavy-cluster mean speedup (108-frame sweep @ 128t).
//       * Mixtral / GPT-OSS prompt-light frames: wide-N fallback
//         routes to sequential-with-full-team — equivalent to the
//         legacy ALGO 1 path on these shapes.
//       * Other prompt frames: ALGO 2 single-tier Phase 2
//         (M-weighted distribution) — the existing M-tile baseline.
//     Safety clamp: when `!m_tile_safe` the auto path falls back
//     to ALGO 1, matching the global `ALGO=2` env path.  Set
//     `AUTO_PROMPT_ALGO=1` to restore the legacy sequential_experts
//     default for A/B regression testing or as an escape hatch on
//     workloads where ALGO 2 single-tier Phase 2 is not yet
//     validated against ALGO 1's full-team weight cache.
//
//   AUTO_DECODE_ALGO default = 3 (flat_n_tile / N-tile rounds path).
//     ALGO 3 + CK rounds is the measured decode winner across every
//     benchmarked MoE workload (Qwen3-30B-A3B, GPT-OSS, Mixtral
//     8x7B).  Safety clamp: when `!n_tile_safe` the auto path falls
//     back to ALGO 1, matching the global `ALGO=3` env path.
//
// Set the env to `0` explicitly to restore the legacy 3-rule cascade
// for the matching phase (escape hatch for A/B regression testing
// against pre-default-flip behaviour).  Set to a specific algo
// (1..5) to pin that phase to a single ALGO for tuning.
//
// Structural gates that ALWAYS fire (independent of phase env):
//   * R0 capacity (`num_ops > kNTilePlanMaxExperts=256`) → ALGO 5.
//     Phase env cannot override the N-tile planner's capacity
//     ceiling.
//
// Telemetry: the `[GRP_MATMUL.ALGO]` apilog line surfaces `phase=`
// (prompt/decode) and `auto_prompt_env=` / `auto_decode_env=` (the
// active value of each phase env, post-override) so operators can
// confirm in one grep which routing decision fired.
//
// Mid-process env changes have no effect (cached static const); tests
// override via the atomics below.

// ZENDNNL_GRP_MATMUL_AUTO_PROMPT_ALGO = { 0, 1..5 } — cached, default 2.
//   Phase env consulted by `auto_select_algo` when `ALGO=0` (auto) AND
//   the call classifies as PROMPT (`max_M > kDecodeMaxM`).  Value 0
//   defers to the legacy 3-rule cascade; 1..5 forces that ALGO for
//   the prompt phase (with the same m_tile_safe / n_tile_safe safety
//   clamps the global `ALGO` env path applies).  Default 2
//   (flat_m_tile + multi-tier hybrid + wide-N fallback) is the
//   out-of-the-box prompt choice — see the doc-block on this group
//   of envs for the rationale.
inline std::atomic<int> &test_api_auto_prompt_algo_override();
inline int get_grp_matmul_auto_prompt_algo() {
  // Strict env parsing — non-numeric input falls back to the documented
  // default 2 (flat_m_tile).  Bogus values (< 0 OR > 5) also clamp
  // to the default so a typo cannot accidentally pin an unintended
  // algo.
  constexpr int kDefault = 2;
  const int ovr = test_api_auto_prompt_algo_override().load(
      std::memory_order_relaxed);
  if (ovr >= 0) return (ovr <= 5) ? ovr : kDefault;
  static const int v = []() {
    const char *e = std::getenv("ZENDNNL_GRP_MATMUL_AUTO_PROMPT_ALGO");
    int parsed = 0;
    if (!parse_env_int_strict(e, parsed)) return kDefault;
    return (parsed >= 0 && parsed <= 5) ? parsed : kDefault;
  }();
  return v;
}

// ZENDNNL_GRP_MATMUL_AUTO_DECODE_ALGO = { 0, 1..5 } — cached, default 3.
//   Phase env consulted by `auto_select_algo` when `ALGO=0` (auto) AND
//   the call classifies as DECODE (`max_M ≤ kDecodeMaxM`).  Value 0
//   defers to the legacy 3-rule cascade; 1..5 forces that ALGO for
//   the decode phase (with the same safety clamps as the prompt
//   path).  Default 3 (N-tile rounds + CK) is the new out-of-the-
//   box decode choice — the measured decode winner across the MoE
//   envelope we benchmark.
inline std::atomic<int> &test_api_auto_decode_algo_override();
inline int get_grp_matmul_auto_decode_algo() {
  constexpr int kDefault = 3;
  const int ovr = test_api_auto_decode_algo_override().load(
      std::memory_order_relaxed);
  if (ovr >= 0) return (ovr <= 5) ? ovr : kDefault;
  static const int v = []() {
    const char *e = std::getenv("ZENDNNL_GRP_MATMUL_AUTO_DECODE_ALGO");
    int parsed = 0;
    if (!parse_env_int_strict(e, parsed)) return kDefault;
    return (parsed >= 0 && parsed <= 5) ? parsed : kDefault;
  }();
  return v;
}

// True when the operator has EXPLICITLY chosen a prompt phase algo —
// either via the test override atomic (`>= 0`) or a parseable
// `ZENDNNL_GRP_MATMUL_AUTO_PROMPT_ALGO` env value (including `0`, the
// explicit "legacy cascade" request).  Lets the default-policy few-
// experts ALGO 2 preference defer to an explicit pin while still firing
// out-of-the-box.  Env presence is cached on first call (same pattern as
// the value getter); test overrides read the live atomic.
inline bool grp_matmul_auto_prompt_algo_is_set() {
  if (test_api_auto_prompt_algo_override().load(std::memory_order_relaxed) >= 0)
    return true;
  static const bool s = []() {
    const char *e = std::getenv("ZENDNNL_GRP_MATMUL_AUTO_PROMPT_ALGO");
    int parsed = 0;
    return parse_env_int_strict(e, parsed);
  }();
  return s;
}

// Decode counterpart of `grp_matmul_auto_prompt_algo_is_set()`.
inline bool grp_matmul_auto_decode_algo_is_set() {
  if (test_api_auto_decode_algo_override().load(std::memory_order_relaxed) >= 0)
    return true;
  static const bool s = []() {
    const char *e = std::getenv("ZENDNNL_GRP_MATMUL_AUTO_DECODE_ALGO");
    int parsed = 0;
    return parse_env_int_strict(e, parsed);
  }();
  return s;
}

// NOTE: `get_grp_n_tile_fused_act()` moved to
// `group_matmul_n_tile.hpp` (Section A.4) together with the rest of
// the N-tile env getters.  Include that header to call it.

/// True when ALGO 3 can fuse `act` into the per-thread epilogue.
/// Extend here when new activation kinds gain epilogue support.
///
/// Currently supports:
///   * `swiglu_oai_mul` — interleaved-input path (caller-side
///     interleaved W13).  Both the CK in-register epilogue
///     (`swiglu_oai_store_pair`) and the standard-backend
///     wide-arena helper (`apply_swiglu_oai_tile_rows`) handle it,
///     so the fused path is supported regardless of `use_custom_kernel`.
///   * `silu_and_mul` — split-halves input (canonical W13).  Only
///     the CK in-register epilogue (`silu_and_mul_store_pair`)
///     handles it today; the standard backend's
///     `apply_swiglu_oai_tile_rows` is swiglu-only and has no silu
///     sibling, so this kind is fusible only when CK is engaged.
///   * `gelu_and_mul` — split-halves input (canonical W13).  Same
///     story as silu: only the CK in-register epilogue
///     (`gelu_and_mul_store_pair`) implements the fused form, with
///     a `gelu_tanh` polynomial approximation that matches the
///     reference's `gelu_erf` to within BF16 tolerance.  The
///     standard backend's wide-arena helper is swiglu-only, so
///     gelu fused requires `use_custom_kernel=true`.
///
/// Standard-backend silu / gelu tile helpers are a planned follow-up;
/// until then, callers with CK off fall back to the separate-pass
/// path (`act_fused = false`) by way of this gate returning `false`.
inline bool a3_can_fuse_act(grp_matmul_gated_act_t act,
                            bool use_custom_kernel) {
  if (act == grp_matmul_gated_act_t::swiglu_oai_mul) return true;
  if (act == grp_matmul_gated_act_t::silu_and_mul) return use_custom_kernel;
  if (act == grp_matmul_gated_act_t::gelu_and_mul) return use_custom_kernel;
  return false;
}

// ZENDNNL_GRP_MATMUL_N_ROUNDS = { 0, 1, 2, 3 } — cached, default 1.
//   ALGO 3 ManyExperts round-mode selection.  Internal tuning knob.
//     0 = auto: planner picks single-round / multi-round / balanced
//               via cost-model on wall time.
//     1 = force single-round (all experts in one round, n_thr =
//         num_threads / num_ops).  Falls back to balanced when
//         num_threads < num_ops.  CURRENT DEFAULT: production
//         sweeps showed single-round dominates on the target MoE
//         envelope at high thread counts; the auto cost-model
//         occasionally picked balanced/multi-round at boundaries
//         where single-round was within noise but had simpler
//         cache-key behaviour.  Pin to single-round so the
//         planner's choice is shape-independent and the AOCL DLP
//         per-tile cache key set stays stable across decode
//         iterations.  Set "0" to restore the original auto
//         behaviour for A/B comparisons or shape exploration.
//     2 = force multi-round legacy fixed-shape (batch experts × ccd_size
//         threads each, possibly wasteful tail round).
//     3 = force balanced (n_rounds = ceil(num_ops / target_batch),
//         experts evenly redistributed across rounds).
//   Mid-process env changes have no effect; relaunch for A/B.
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
inline std::atomic<int> s_grp_n_rounds_mode_override{-1};
inline std::atomic<int> s_grp_matmul_custom_kernel_override{-1};
// DQ-INT8 CK sub-knob — independent from the master CK switch so
// tests / deployments can A/B the int8 fast path without
// disturbing the bf16 path.  Sentinel `-1` = no override (env / default).
inline std::atomic<int> s_grp_matmul_custom_kernel_int8_override{-1};
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
inline std::atomic<int> s_grp_matmul_custom_kernel_nr_override{-1};

// Sentinel `-1` = no override.  Settable values: 0 (explicit legacy
// 3-rule cascade — escape hatch from the new default phase pin),
// 1..5 (force the matching ALGO for the phase matching the override's
// name).  Override semantics in `get_grp_matmul_auto_prompt_algo()` /
// `get_grp_matmul_auto_decode_algo()`:
//   * any negative value (including the `-1` sentinel) → fall
//     through to the cached env path (which itself applies the
//     documented defaults — 1 for prompt, 3 for decode).
//   * 0          — explicit legacy 3-rule cascade selection.
//                  Production deployments that want pre-default-flip
//                  behaviour use this (or the env equivalent
//                  `AUTO_*_ALGO=0`).
//   * 1..5       — adopted as the override value.
//   * > 5        — clamped to the documented default (1 for prompt,
//                  3 for decode), matching the env-parse validation
//                  behaviour.
inline std::atomic<int> s_grp_matmul_auto_prompt_algo_override{-1};
inline std::atomic<int> s_grp_matmul_auto_decode_algo_override{-1};

// Sentinel `-1` = no override (use cached env path).  Settable values
// 0..5 mirror `get_grp_matmul_algo()` parse output (`0` = AUTO,
// `1..5` = forced ALGO_N, `> 5` clamped to AUTO by the getter).  The
// `AlgoEnvGuard` RAII helper in `gtests/group_matmul/moe_test_utils.hpp`
// sets the env-var AND stores into this atomic so that any gtest using
// `AlgoEnvGuard(N)` continues to flip the effective algo mid-process —
// without paying the `std::getenv` cost on every production call site.
inline std::atomic<int> s_grp_matmul_algo_override{-1};

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
inline std::atomic<int> s_grp_matmul_custom_kernel_subtile_per_expert_override{-1};

// Last `gemm_mode` string set by `group_matmul_direct` on a
// successful return.  Read-only inspection hook for tests that need
// to verify which executor path actually ran (e.g. asserting the
// custom BF16 microkernel engaged vs the call falling back to AOCL
// DLP or the Sequential strategy that bypasses CK entirely).
//
// Strings come from `flat_n_tile`'s `gemm_mode_label` (defined in
// `group_matmul_n_tile.cpp`) or the per-algo executor labels.  They
// are static literals — never freed — so the atomic stores a stable
// pointer that test code can read after the call returns.
//
// CAPTURE GATE — `s_capture_gemm_mode` (atomic bool, default false):
//   Production builds never set this flag, so the store path in
//   `group_matmul_direct` short-circuits on a single relaxed load
//   of a cache-line-shared `false` value (no coherence traffic).
//   Tests arm the flag (via `GemmModeCaptureGuard` in
//   `moe_test_utils.hpp`) for the test's scope, in which case the
//   gated store DOES fire and writes through to the atomic below.
//   Without this gate the unconditional store would mark its
//   cache line Modified on every successful dispatcher call,
//   forcing a coherence ping-pong across any cores running
//   concurrent `group_matmul_direct` invocations — a measurable
//   tax on multi-rank serving deployments that have no use for
//   the test hook.  Same pattern as `s_capture_phase_b` (see
//   `group_matmul_n_tile.hpp`).
inline std::atomic<bool>         s_capture_gemm_mode{false};
inline std::atomic<const char *> s_last_group_matmul_direct_gemm_mode{nullptr};

// NOTE: The M-tile (ALGO 2) branch-tag capture hook —
// `s_capture_m_tile_path`, `s_last_m_tile_path`, and the
// `m_tile_path_tag::*` named constants — moved out to
// `group_matmul_m_tile.hpp` (Section H.2).  See the file header
// there for the same capture-gate rationale that used to live here.
}  // namespace test_api

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

inline int get_grp_n_rounds_mode() {
  const int ovr = test_api::s_grp_n_rounds_mode_override.load(
      std::memory_order_relaxed);
  if (ovr >= 0) return ovr;
  // Default: 1 (single-round).  Strict env parsing — anything that
  // is not exactly `"0"`, `"1"`, `"2"`, or `"3"` falls back to the
  // documented default (NOT silently to mode 0 via the legacy
  // atoi-returns-0-for-junk behaviour).  See `parse_env_int_strict`.
  constexpr int kDefault = 1;
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
// FewExperts/ManyExperts use kDecodeNTile (256) instead of kMinNTile
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
//   wide [M, 2I] when "0".  Tight halves Op2's src DRAM traffic and
//   is neutral-or-positive on every measured workload.  The dispatcher
//   only engages tight when ALL of: act is a CK-fusible gated
//   activation (swiglu_oai_mul, silu_and_mul, gelu_and_mul — see
//   `a3_can_fuse_act` for the live predicate; silu and gelu require
//   `CUSTOM_KERNEL=1` because only the CK in-register epilogue
//   implements them), Op1 internal-alloc on, ALGO 3 selected (auto
//   or env-forced), shape-adaptive picker agrees — otherwise falls
//   back to wide silently regardless of this env.  Note:
//   ZENDNNL_GRP_MATMUL_N_TILE_FUSED_ACT is NOT a tight-engagement
//   gate — when the fused-MoE picker hands the dispatcher a tight
//   destination (ldc < N), the dispatcher auto-enables ALGO 3 fused
//   activation regardless of N_TILE_FUSED_ACT (tight is a
//   correctness constraint on the writer, not a perf toggle).  See
//   `pick_fused_moe_want_tight` in group_matmul_fused_moe.cpp for
//   the full predicate.  Set "0" here to force wide (debug /
//   layout-regression bisection).
inline int get_grp_matmul_fused_moe_tight() {
  static const int v = []() {
    const char *e = std::getenv("ZENDNNL_GRP_MATMUL_FUSED_MOE_TIGHT");
    if (e == nullptr || e[0] == '\0') return 1;  // default: force-tight
    return (e[0] == '0') ? 0 : 1;
  }();
  return v;
}

// ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL = { "0", "1" } — cached, default ON.
//   Master switch for the hand-rolled AVX-512-BF16 microkernel
//   (custom_kernel/).  ON: ALGO 3 flat_n_tile dispatches per-tile
//   GEMM through VDPBF16PS, with swiglu_oai_mul applied in-register
//   (writes activated I cols at caller's ldc — covers both wide and
//   tight fused-MoE layouts in one path).  OFF: per-tile GEMM goes
//   through the standard AOCL DLP / BRGEMM dispatch, fused activation
//   runs as a separate per-tile pass.
//
//   Default ON history: an earlier revision flipped this to OFF
//   because the CK pack cache (see `custom_kernel/pack.cpp`) is
//   keyed by the raw weight pointer, and frameworks that recycle
//   freed allocator addresses (e.g. PyTorch CPU allocator) could
//   silently serve stale packed bytes for a new tensor at the
//   same address.  That hazard is now addressed at the library
//   level: the CK path honours `ZENDNNL_MATMUL_WEIGHT_CACHE` with
//   the same semantics as the AOCL DLP path.
//     - `ZENDNNL_MATMUL_WEIGHT_CACHE=1` (default): pointer-keyed
//       LRU cache stays warm across calls (production MoE serving
//       with a stable model — the common case).
//     - `ZENDNNL_MATMUL_WEIGHT_CACHE=0`: per-call caller-owned
//       packed buffers are allocated fresh and freed after the
//       call, so a recycled pointer can never hit a stale entry.
//       CK remains engaged, just without the cache.
//   See `custom_kernel/dispatch.cpp::prepare_for_call` (owned-ptr
//   path) and `prepack/prepack_custom_kernel.cpp` (warm-pack skip
//   when WEIGHT_CACHE=0).  Default-ON is therefore safe for both
//   stable-pointer and recycled-pointer regimes; callers that
//   want to bypass CK entirely (e.g. parity bisection against
//   AOCL DLP) can still set `ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL=0`.
//
//   The dispatcher refuses cleanly and falls back to the standard
//   AOCL DLP path for any expert that violates the CK contract
//   (non-bf16, transA, alpha≠1, β≠0, N % pack_nr ≠ 0, non-const
//   weights, etc. — see `custom_kernel/dispatch.cpp::prepare_for_call`
//   for the full gate cascade), so callers outside the supported
//   envelope see no behaviour change regardless of this knob.
inline bool get_grp_matmul_custom_kernel() {
  const int ovr = test_api::s_grp_matmul_custom_kernel_override.load(
      std::memory_order_relaxed);
  if (ovr >= 0) return ovr != 0;
  static const bool v = []() {
    const char *e = std::getenv("ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL");
    if (e == nullptr || e[0] == '\0') return true;  // default: ON
    return e[0] != '0';
  }();
  return v;
}

// ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_INT8 = { "0", "1" } — cached, default ON.
//   Independent sub-switch for the hand-rolled AVX-512 VNNI int8
//   microkernel (`custom_kernel/ukernel/int8_microkernel.{hpp,cpp}`).
//   Lets deployments A/B the DQ-INT8 fast path against the AOCL DLP
//   `aocl_gemm_s8s8s32obf16_sym_quant` reference without affecting
//   the bf16 CK path.
//
//   Cascade with the master `_CUSTOM_KERNEL` switch:
//     * `_CUSTOM_KERNEL=0`                       → both regimes off
//                                                  (every CK route falls
//                                                  back to standard DLP /
//                                                  BRGEMM dispatch).
//     * `_CUSTOM_KERNEL=1 && _CUSTOM_KERNEL_INT8=0` → bf16 CK on,
//                                                  int8 CK off (DQ-INT8
//                                                  N-tile calls fall back
//                                                  to AOCL DLP sym_quant).
//     * `_CUSTOM_KERNEL=1 && _CUSTOM_KERNEL_INT8=1` → both on (default).
//
//   Implemented as a STATIC sub-knob: even if the master `_CUSTOM_KERNEL`
//   knob is on, eligibility code paths that route a DQ-INT8 call to
//   CK first consult this getter — when it returns false the call
//   routes to the AOCL DLP sym-quant fallback exactly as it does
//   under `_CUSTOM_KERNEL=0`.
//
//   Tests can pin the value via `s_grp_matmul_custom_kernel_int8_override`
//   (sentinel `-1` = no override).
inline bool get_grp_matmul_custom_kernel_int8() {
  const int ovr = test_api::s_grp_matmul_custom_kernel_int8_override.load(
      std::memory_order_relaxed);
  if (ovr >= 0) return ovr != 0;
  static const bool v = []() {
    const char *e = std::getenv("ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_INT8");
    if (e == nullptr || e[0] == '\0') return true;  // default: ON
    return e[0] != '0';
  }();
  return v;
}

// ZENDNNL_GRP_MATMUL_CROSS_WARM = { "0", "1" } — cached, default ON.
//   When ON, each `prepack_for_algo_X` opportunistically populates the
//   cache regime that auto-select would route the OTHER phase to in the
//   same process, so a deployment that fires only prompt during warmup
//   still arrives at decode with both regimes warm — no first-decode-
//   call prepack spike.
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
//       (per-tile AOCL) is populated by `prepack_for_algo_3` when
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
//   for A/B comparison with the pre-cross-warm behaviour.
//
//   Independent of `ZENDNNL_GRP_MATMUL_PREPACK`: the master knob
//   short-circuits everything to a no-op; this knob only controls
//   the cross-regime fan-out when the master is ON.
inline bool get_grp_matmul_cross_warm() {
  static const bool v = []() {
    const char *e = std::getenv("ZENDNNL_GRP_MATMUL_CROSS_WARM");
    if (e == nullptr || e[0] == '\0') return true;   // default: On
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
//             paid in exchange for `do_tile()` cache hits in
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
//   `group_matmul/test_prepack.cpp` ([26]-[28]).
inline bool get_grp_matmul_prepack() {
  static const bool v = []() {
    const char *e = std::getenv("ZENDNNL_GRP_MATMUL_PREPACK");
    if (e == nullptr || e[0] == '\0') return true;   // default: On
    return e[0] != '0';
  }();
  return v;
}

// ──────────────────────────────────────────────────────────────────────
// AOCL-path stable N-tile (strict, num_threads-only)
//
// AOCL DLP / BRGEMM / oneDNN reorder caches key on the per-thread
// slice — `(transB, K, n_tile, ldb, B + col_start·elem, algo)` —
// NOT on the full weight.  When `n_thr_per_expert` varies between
// calls (active-expert filtering, batch-size shifts, …), col_start
// and n_tile rotate, the cache thrashes, and under churn the LRU
// can free entries another thread is still reading via raw pointer
// → use-after-free → garbage rows.  The custom kernel sidesteps
// this (its pack cache is shape-keyed), so this whole subsystem is
// non-custom only.
//
// Mitigation: for non-custom dispatch under ALGO 3 flat_n_tile,
// pin the per-expert thread count to a `num_threads`-only formula:
//
//     stable = max(1, num_threads / kAoclTargetConcurrentSlots)
//
// `stable` depends ONLY on `num_threads` and the env-static
// kAoclTargetConcurrentSlots — invariant across MoE routing,
// expert filtering, N shifts, and num_ops shifts.  The planner
// (`plan_group_n_tile`) then forces `n_thr_fixed = stable` and
// `batch_size = num_threads / stable` so every expert team has
// exactly `stable` threads in every round.  `col_start` and
// `n_tile` therefore stay byte-identical across calls → AOCL
// cache reaches a steady hit-rate post-warmup, regardless of any
// caller-side variation.
//
// Why the formula does not include an N-dependent density floor:
// an earlier `by_density = N / kAoclBlisNc` term protected thin-N
// shapes above AOCL's NC=128 amortisation point, but for variable-
// N MoE callers it re-introduced an N-dependence into `stable`,
// rotating the cache key per expert and undermining the stability
// contract.  Narrow-N protection is now handled by a planner-side
// escape: when `stable * nr_align > max_N` (the regime where
// `aligned_n_split` cannot produce stable aligned slices),
// `plan_group_n_tile` routes the call to Sequential which uses
// the full thread team per expert and bypasses tile-level cache
// keys entirely.
//
// Trade-off: at low num_ops (num_ops × stable < num_threads),
// some threads idle in the strict-stable plan.  Accepted as the
// cost of the cache-stability guarantee — for typical MoE decode
// workloads, the per-call cache-hit savings (tens of milliseconds
// of avoided reorders) dominate the per-call thread-utilisation
// loss (sub-ms).
//
// `participating_n_thr` (group_matmul_n_tile.cpp) retains secondary
// clamps by `align_cap = N / nr_align` and `team_size` as defence-
// in-depth: the strict-stable planner already guarantees
// `team_size == stable` and `align_cap >= stable` (else the
// narrow-N escape fires), so the clamps are no-ops in the strict-
// stable plan — they only fire if a future planner regression
// breaks an invariant, in which case they degrade gracefully to
// dynamic-tile behaviour rather than silently corrupting output.
// ──────────────────────────────────────────────────────────────────────

// ZENDNNL_GRP_MATMUL_AOCL_STABLE_NTILE = { "0", "1" } — cached, default ON.
//   "0" restores the legacy dynamic plan topology (cache thrash)
//   for A/B benchmarking.
inline bool get_grp_matmul_aocl_stable_ntile() {
  static const bool v = []() {
    const char *e = std::getenv("ZENDNNL_GRP_MATMUL_AOCL_STABLE_NTILE");
    if (e == nullptr || e[0] == '\0') return true;
    return e[0] != '0';
  }();
  return v;
}

inline constexpr int kAoclTargetConcurrentSlots = 16;   // team-budget divisor
inline constexpr int kAoclBlisNc                = 128;  // BLIS-bf16 inner-N block

// ZENDNNL_GRP_MATMUL_AOCL_TARGET_SLOTS = positive int — cached, default 16.
//   Team-budget divisor.  Lower (e.g. 8) for few-expert deployments;
//   raise for many-expert deployments where reducing per-expert fan-
//   out helps.  Non-positive → default.
inline int get_grp_matmul_aocl_target_slots() {
  // Strict env parsing — non-numeric input falls back to default.
  static const int v = []() {
    const char *e = std::getenv("ZENDNNL_GRP_MATMUL_AOCL_TARGET_SLOTS");
    int parsed = 0;
    if (!parse_env_int_strict(e, parsed)) return kAoclTargetConcurrentSlots;
    return (parsed > 0) ? parsed : kAoclTargetConcurrentSlots;
  }();
  return v;
}

// ZENDNNL_GRP_MATMUL_AOCL_BLIS_NC = positive int — cached, default 128.
//   Informational / telemetry only as of the strict-stable cache-key
//   simplification.  The original `aocl_stable_n_thr` formula included
//   a `by_density = N / kAoclBlisNc` term that protected thin-N
//   shapes above AOCL's NC=128 amortisation point; that term was
//   removed because it re-introduced an N-dependence into the per-
//   expert thread count and rotated the AOCL DLP cache key per call.
//   Narrow-N protection is now handled by the planner's narrow-N
//   escape (see `aocl_stable_n_thr` and the F3 escape comment in
//   `plan_group_n_tile`).
//
//   The env value is parsed (strict) and emitted in the
//   `[GRP_MATMUL.PLAN] flat_n_tile ...` apilog line so external
//   telemetry can correlate user-set tuning with the planner's
//   actual choices; setting it does NOT change planning behaviour.
//   Non-positive → default.  Kept as a getter (rather than removed
//   outright) so reintroducing a density floor in future is a one-
//   line change in `aocl_stable_n_thr` and we don't churn the env
//   surface area.
inline int get_grp_matmul_aocl_blis_nc() {
  // Strict env parsing — non-numeric input falls back to default.
  static const int v = []() {
    const char *e = std::getenv("ZENDNNL_GRP_MATMUL_AOCL_BLIS_NC");
    int parsed = 0;
    if (!parse_env_int_strict(e, parsed)) return kAoclBlisNc;
    return (parsed > 0) ? parsed : kAoclBlisNc;
  }();
  return v;
}

// NOTE: `get_grp_matmul_n_tile_heavy_threshold()`
// (ZENDNNL_GRP_MATMUL_N_TILE_HEAVY_THRESHOLD) moved to
// `group_matmul_n_tile.hpp` (Section A.4) together with the rest of
// the N-tile env getters.  Include that header to call it.  The
// three-mode (DISABLED / AUTO / MANUAL) doc-block lives at the new
// home.

// NOTE: All `get_grp_matmul_m_tile_*` getters (hybrid /
// slice_target / hybrid_min_max_m / hybrid_min_skew /
// hybrid_lights_per_thread / vertical_fusion / pipeline_scratch_kb)
// moved to `group_matmul_m_tile.hpp` (Section H.3) together with
// the override atoms they read.  Include that header to pull them
// in.

// Per-expert thread count for the AOCL DLP / BRGEMM / oneDNN execute
// path inside ALGO 3 flat_n_tile.  See the strict-stable doc-block
// above for the cache-stability contract and rationale.
//
// Depends ONLY on `num_threads`.  An earlier formula included a
// `by_density = N / kAoclBlisNc` term which re-introduced an
// N-dependence into the cache key.  Narrow-N protection (where
// `by_density` was needed in the first place) is now handled by
// `plan_group_n_tile`'s narrow-N escape — calls that
// can't produce stable-aligned tiles are routed to Sequential
// instead.
//
// The `N` parameter is retained for source-level compatibility with
// existing callers; it is intentionally unused.
inline int aocl_stable_n_thr(int num_threads, int /*N*/) {
  if (num_threads <= 0) return 1;
  return std::max(1, num_threads / get_grp_matmul_aocl_target_slots());
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
  if (ovr >= 0) {
    return (ovr == 32 || ovr == 64) ? ovr : 0;
  }
  // Strict env parsing — non-numeric input (or anything other than
  // exactly "32" / "64") falls back to 0 (auto-pick).
  static const int v = []() {
    const char *e = std::getenv("ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_NR");
    int parsed = 0;
    if (!parse_env_int_strict(e, parsed)) return 0;
    return (parsed == 32 || parsed == 64) ? parsed : 0;
  }
  ();
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
  const int ovr =
      test_api::s_grp_matmul_custom_kernel_subtile_per_expert_override
          .load(std::memory_order_relaxed);
  if (ovr >= 0) return (ovr != 0);
  static const bool v = []() {
    const char *e = std::getenv(
                      "ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_SUBTILE_PER_EXPERT");
    return (e != nullptr && e[0] != '\0' && e[0] != '0');
  }
  ();
  return v;
}

// NOTE: `get_grp_matmul_custom_kernel_n_tile()`
// (ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_N_TILE) moved to
// `group_matmul_n_tile.hpp` (Section A.4) together with the rest of
// the N-tile env getters.  Include that header to call it.

/// N alignment the inner kernel prefers for each per-thread slice
/// (1 = no alignment).  ALGO 3's column partitioner (aligned_n_split)
/// honours it when the slowest thread stays within 2× of the fastest
/// after rounding; otherwise it falls back to the unaligned even split.
inline int backend_n_align(matmul_algo_t algo) {
  switch (algo) {
  case matmul_algo_t::native_brgemm:
  case matmul_algo_t::native_gemm:
    return 64;
  default:
    return 1;
  }
}

/// Aligned column-slice partitioner for ALGO 3.  Returns {col_start,
/// col_end} for `tid` of `n_thr` over [0, N).
///
/// Aligned branch: per-thread = aligned_per_thr (a multiple of `align`),
/// last thread takes the remainder.  Engaged only when last ≥
/// aligned_per_thr/2 (BLIS-style 2× imbalance bound).  Search walks
/// DOWN in `align` quanta from ceil(N/n_thr) rounded up to align,
/// so feasible alignments aren't rejected just because the first
/// candidate over-sized the last slice (e.g. N=2880, n_thr=11,
/// align=64 → 256 cols/thread fits, 320 doesn't).
///
/// Falls back to even split (N*tid/n_thr) when n_thr<=1 or align<=1
/// or no aligned slice meets the 2× bound.
inline std::pair<int, int> aligned_n_split(int N, int n_thr, int tid,
    int align) {
  // Hardened against pathological inputs: n_thr<=0 used to hit
  // even_split's divide-by-zero (`N * tid / n_thr`).
  if (n_thr <= 0 || N <= 0) {
    return std::make_pair(0, 0);
  }

  auto even_split = [&]() {
    const int s = static_cast<int>(static_cast<int64_t>(N) * tid / n_thr);
    const int e = static_cast<int>(
                    static_cast<int64_t>(N) * (tid + 1) / n_thr);
    return std::make_pair(s, e);
  };

  if (align <= 1 || n_thr <= 1) {
    return even_split();
  }

  // Walk slice size down in `align` quanta from ceil(N/n_thr) until
  // the imbalance bound holds.  Cost is at most a handful of
  // iterations in practice (slice sizes converge in 1-2 steps for
  // realistic N / n_thr / align triples).
  //
  // Intermediates promoted to int64_t to keep the products
  // aligned_per_thr * (n_thr - 1) and aligned_per_thr * tid free of
  // signed overflow (UB) for any (N, n_thr) combination representable
  // as int.
  const int64_t even_per_thr =
    (static_cast<int64_t>(N) + n_thr - 1) / n_thr;
  for (int64_t aligned_per_thr =
         ((even_per_thr + align - 1) / align) * align;
       aligned_per_thr >= align;
       aligned_per_thr -= align) {
    const int64_t n_full = aligned_per_thr * (n_thr - 1);
    const int64_t last = static_cast<int64_t>(N) - n_full;
    if (last > 0 && last * 2 >= aligned_per_thr) {
      const int64_t s = aligned_per_thr * tid;
      const int64_t e =
        (tid < n_thr - 1) ? aligned_per_thr * (tid + 1) : N;
      return std::make_pair(static_cast<int>(s), static_cast<int>(e));
    }
  }
  return even_split();
}

/// 32 MB of L3 per CCD on Zen 3 / 4 / 5 classic-CCD topologies.
inline constexpr size_t kL3PerCcdBytes = 32UL * 1024UL * 1024UL;

/// Aggregate L3 the planner uses to bound the experts-per-round
/// budget (ALGO 3 N-tile, ALGO 4 multilevel).  `num_ccds` comes from
/// summarise_topology() so this stays consistent with how the rest
/// of the planner partitions the team.
inline size_t get_grp_l3_total_bytes(int num_ccds) {
  return static_cast<size_t>(std::max(1, num_ccds)) * kL3PerCcdBytes;
}

// ──────────────────────────────────────────────────────────────────────
// Shared per-expert primitives.
// ──────────────────────────────────────────────────────────────────────

/// Resolves the effective matmul algo ID from the runtime config,
/// falling back to AOCL DLP blocked when the config is unset/invalid.
inline matmul_algo_t resolve_kernel() {
  static const matmul_algo_t algo = []() {
    int32_t a = matmul_config_t::instance().get_algo();
    if (a <= 0 || a >= static_cast<int32_t>(matmul_algo_t::algo_count)) {
      return matmul_algo_t::aocl_dlp_blocked;
    }
    return static_cast<matmul_algo_t>(a);
  }
  ();
  return algo;
}

/// Thin wrapper around matmul_execute that packages per-expert slice
/// arguments into the batch/params objects the kernel expects.
inline void execute_expert_slice(
  char layout, bool transA, bool transB,
  int M, int N, int K, float alpha,
  const void *src, int lda,
  const void *weight, int ldb,
  const void *bias, float beta,
  void *dst, int ldc,
  bool is_weights_const, int num_thr,
  matmul_params &params,
  matmul_algo_t algo) {

  matmul_batch_params_t bp;
  bp.Batch_A = 1;
  bp.Batch_B = 1;
  matmul_algo_t kernel = algo;

  // Per-expert dynamic-quant fallback path.  This fires only when the
  // source is still bf16/f32 and `params.dynamic_quant == true` (the
  // wrapper's own eligibility gate).  When the caller already ran the
  // grouped `group_dynamic_quant` pre-pass (ZENDNNL_ENABLE_GROUP_DQ on),
  // it rewrote `params.dtypes.src` to s8 and cleared `dynamic_quant`, so
  // this wrapper short-circuits to a no-op and no double-quant occurs.
  // When grouped DQ is disabled (env off) this is the active per-expert
  // quantization path, preserving the legacy behaviour.
  int         reordered_lda = lda;
  size_t      src_type_size = size_of(params.dtypes.src);
  reorder_quant_buffers_t quant_buffers;
  if (reorder_quantization_wrapper(src, lda, reordered_lda, src_type_size,
                                   params, bp, transA, M, K,
                                   num_thr, quant_buffers) != status_t::success) {
    log_error("execute_expert_slice: reorder_quantization_wrapper failed");
    return;
  }

  matmul_execute(layout, transA, transB,
                 M, N, K, alpha, src,
                 params.dynamic_quant ? reordered_lda : lda,
                 weight, ldb, bias, beta, dst, ldc,
                 is_weights_const, src_type_size,
                 size_of(params.dtypes.dst),
                 num_thr, kernel, params, bp, 0);
}

/// ZENDNNL_ENABLE_GROUP_DQ — opt-in/out for the grouped dynamic-quant
/// pre-pass (`group_dynamic_quant`).  Default ON (unset or any value
/// other than a literal "0").  When OFF, group matmul skips the grouped
/// source-quant pre-pass and dynamic quantization falls back to the
/// per-expert `reorder_quantization_wrapper` inside `execute_expert_slice`
/// (legacy path).  Intentionally NOT cached so gtests can flip it
/// per-scope via setenv; the getenv cost is negligible next to a GEMM.
inline bool get_grp_matmul_enable_group_dq() {
  const char *env = std::getenv("ZENDNNL_ENABLE_GROUP_DQ");
  if (env == nullptr || env[0] == '\0') return true;     // default ON
  return !(env[0] == '0' && env[1] == '\0');             // "0" => OFF
}

// NOTE: The N-tile shared utilities — `sort_indices_by_m`,
// `engage_ntile_custom_kernel`, `ntile_effective_nr_align`,
// `auto_pick_n_order`, `fill_ntile_expert_order` — moved to
// `group_matmul_n_tile.hpp` (Section A.5) so they live alongside
// the N-tile env knobs they read.  Include that header to call
// them — currently only `group_matmul_n_tile.cpp` does.

// ──────────────────────────────────────────────────────────────────────
// Tile-strategy entry points — library-internal linkage; the
// dispatcher in group_matmul_dispatch.cpp forwards to these.
// ──────────────────────────────────────────────────────────────────────

// NOTE: The M-tile executor forward declarations (`flat_m_tile`
// and `flat_m_tile_pipeline_bf16`) moved to
// `group_matmul_m_tile.hpp` (Section H.4).  Include that header to
// call them — the dispatcher in `group_matmul_dispatch.cpp` and
// the vertical-fusion entry in `group_matmul_fused_moe.cpp` both
// do this.

// NOTE: The N-tile executor forward declaration (`flat_n_tile`)
// moved to `group_matmul_n_tile.hpp` (Section A.5) together with the
// rest of the N-tile public surface.  Include that header to call
// it — the dispatcher in `group_matmul_dispatch.cpp` and the
// fused-MoE legacy path in `group_matmul_fused_moe.cpp` both do this.

/// Peek at the ALGO the dispatcher would pick for this call (1=seq,
/// 2=m_tile, 3=n_tile, 4=multilevel, 5=per_expert).  Mirrors the
/// dispatcher's full gating: ZENDNNL_GRP_MATMUL_ALGO override, m/n
/// tile-safety checks, auto_select_algo on env=0.  Pure observer
/// (no side-effects); used by the fused-MoE entry to choose tight
/// vs wide arena before committing the buffer layout.
int select_grp_matmul_algo(
  const std::vector<char> &layout,
  const std::vector<int> &M,
  const std::vector<int> &N,
  const std::vector<int> &K,
  const std::vector<matmul_params> &params,
  int num_threads);

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif  // ZENDNNL_GROUP_MATMUL_PARALLEL_COMMON_HPP
