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

/// Library-internal helpers shared by the generic scheduling-ALGO paths.
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
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "common/op_config.hpp"
#include "custom_kernel/dispatch.hpp"
#include "group_matmul_direct.hpp"
#include "lowoha_operators/common/omp_thread_control.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"
#include "lowoha_operators/matmul/quantization/reorder_quantization.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

// Match the original group_matmul_dispatch.cpp using-declarations so
// source files including this header can refer to matmul_algo_t,
// matmul_config_t, post_op_type_t, etc. without namespace prefixes.
using namespace zendnnl::common;
using zendnnl::common::size_of;

// Group-matmul selector identities.  These are deliberately independent of
// `matmul_algo_t`: that enum names inner GEMM kernels, while these values name
// whole-call grouped scheduling/interception modes.
//
// Selector 4 is the W8A8 whole-call interceptor; selector 6 owns the generic
// multilevel scheduler.
inline constexpr int kGrpMatmulAlgoAuto = 0;
inline constexpr int kGrpMatmulAlgoMultilevel = 6;
inline constexpr int kGrpMatmulAlgoNTileFlatParallel = 4;
// One whole expert per thread, no intra-expert split.  Named for the same
// reason as its siblings above: the selector compares against it in several
// files, and a bare `5` there says nothing about the strategy it selects.
inline constexpr int kGrpMatmulAlgoExpertParallel = 5;

inline constexpr bool is_grp_matmul_generic_algo(int algo) {
    return algo == 1 || algo == 2 || algo == 3
            || algo == kGrpMatmulAlgoExpertParallel
            || algo == kGrpMatmulAlgoMultilevel;
}

inline constexpr bool is_grp_matmul_phase_algo(int algo) {
    return algo == kGrpMatmulAlgoAuto || is_grp_matmul_generic_algo(algo);
}

inline constexpr bool is_grp_matmul_requested_algo(int algo) {
    return is_grp_matmul_phase_algo(algo)
            || algo == kGrpMatmulAlgoNTileFlatParallel;
}

// Short string renderer for `grp_matmul_gated_act_t` — used by APILOG
// lines and gemm_mode_out strings; keeps activation names consistent
// across executors.
inline const char *act_name(grp_matmul_gated_act_t a) {
    switch (a) {
        case grp_matmul_gated_act_t::none: return "none";
        case grp_matmul_gated_act_t::silu_and_mul: return "silu_and_mul";
        case grp_matmul_gated_act_t::gelu_and_mul: return "gelu_and_mul";
        case grp_matmul_gated_act_t::swiglu_oai_mul: return "swiglu_oai_mul";
    }
    return "?";
}

// Tile-size + weight-class constants shared by ALGO 0 auto-select and
// the tile kernels.  Cutoffs separate per-CCD-L3-resident shapes
// (small) from L3-tight shapes (medium) from DRAM-streaming (large).
inline constexpr int kDecodeMaxM = 32; // per-expert M ≤ this → "decode"
inline constexpr int kMinNTile = 512; // prompt-path per-thread N

/// Runtime phase used by AUTO group-matmul policy and W8A8 interception.
///
/// A call is decode when the maximum M in its active operation prefix is at
/// most `kDecodeMaxM`; otherwise it is prompt. Empty prefixes are decode (their
/// effective max M is zero), although normal dispatch short-circuits them
/// before phase-specific execution.
enum class grp_matmul_phase { decode, prompt };

inline constexpr grp_matmul_phase classify_grp_matmul_phase(int max_active_m) {
    return max_active_m <= kDecodeMaxM ? grp_matmul_phase::decode
                                       : grp_matmul_phase::prompt;
}

inline int max_active_grp_matmul_m(
        const std::vector<int> &M, size_t active_prefix) {
    const size_t count = std::min(active_prefix, M.size());
    if (count == 0) return 0;
    return *std::max_element(
            M.begin(), M.begin() + static_cast<std::ptrdiff_t>(count));
}

/// Classify only the first `active_prefix` entries. This overload is used by
/// `group_matmul_direct`, where trailing M entries may describe prepack-only
/// experts and must not turn a decode call into prompt.
inline grp_matmul_phase classify_grp_matmul_phase(
        const std::vector<int> &M, size_t active_prefix) {
    return classify_grp_matmul_phase(max_active_grp_matmul_m(M, active_prefix));
}

inline grp_matmul_phase classify_grp_matmul_phase(const std::vector<int> &M) {
    return classify_grp_matmul_phase(M, M.size());
}

inline constexpr const char *grp_matmul_phase_name(grp_matmul_phase phase) {
    return phase == grp_matmul_phase::decode ? "decode" : "prompt";
}

// ── Single-op (num_ops == 1) specialisation scope ───────────────────────
// SINGLE SOURCE OF TRUTH for the optimisations added for a lone expert.
//
// WHAT IS ACTUALLY TESTED: `num_ops == 1`, plus `max_M <= kDecodeMaxM` for
// the decode refinement.  Nothing here inspects operand shapes, so "dense
// FFN" / "W13/W2" in these doc-blocks names the workload the paths were
// TUNED for, not a property the predicates verify: "dense" means "a
// single-op group" (i.e. not MoE), and any other single-op grouped matmul
// in the decode M band is in scope too.  That is deliberate — all three
// consumers are shape-generic perf heuristics with their own guards (Rule
// 0.45 also requires `n_tile_safe`; the kblock gate keys on per-expert M
// and dtype) — so a non-FFN single-op caller gets a differently-tuned but
// equally valid plan.  Add operand-shape inputs here if a consumer ever
// needs FFN specifically.
//
// Two predicates, so each consumer takes exactly the scope it needs:
//   `is_single_dense_expert` (phase-agnostic) guards adaptive N-tile sizing
//       in `adaptive_ab_min_tile`, which serves BOTH phases and so must not
//       be decode-gated.
//   `is_dense_ffn_decode` (adds the decode gate) guards Rule 0.45 ALGO-3
//       routing and the auto deep-K K-blocking in `prepare_for_call`; both
//       must fire only in decode, since single-expert prompt has its own
//       path (ALGO 1).
inline bool is_single_dense_expert(int num_ops) {
    return num_ops == 1;
}
inline bool is_dense_ffn_decode(int num_ops, int max_M) {
    return is_single_dense_expert(num_ops) && max_M <= kDecodeMaxM;
}

// DQ-INT8 sibling of `kMinNTile`.  Today set equal to the bf16
// value so the dtype-aware split is a structural no-op.  Kept as an
// independent constant so the int8 family can be retuned without
// perturbing the bf16 prompt thresholds.
inline constexpr int kMinNTileInt8 = 512;

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
// dropped instead of moved.  Re-introduce only when a consumer needs
// it.

/// Few-experts threshold for ALGO 0 auto-select rule 2.  Workloads
/// with `num_ops ≤ kFewExpertsAlgo1` AND `num_ops < num_threads`
/// (rule 1's strict `>=` would otherwise win on a ≤ 8-thread host)
/// pin to ALGO 1 (sequential experts with full-team AOCL DLP) for
/// prompt AND decode.  Targets few-expert layers, where the per-expert
/// weight footprint is large enough that the full-team sequential path
/// is a better fit than N-tile's column slices on a thin per-expert
/// thread budget.
///
/// The `num_ops < num_threads` precondition holds for a few-expert
/// layer on typical multi-CCD hosts.  An 8-expert workload on a
/// ≤ 8-thread host instead falls to rule 1 and routes to ALGO 3.
/// Documented in `auto_select_algo`'s rule precedence comment in
/// `group_matmul_dispatch.cpp`.
inline constexpr int kFewExpertsAlgo1 = 8;

/// Total-expert ceiling for Rule 0.5's decode occupancy policy. When the
/// global ALGO is AUTO and the decode phase is not explicitly pinned, calls
/// at or below this threshold choose between ALGO 3 and ALGO 1 using active
/// expert occupancy. Larger expert pools inherit the normal decode default.
inline constexpr int kFewExpertsDecodeThreshold = 8;

/// Threads-per-active-expert factor for the few-expert DECODE arrow
/// (Rule 0.5 in `auto_select_algo`).  ALGO 3 splits each expert's N across
/// the team, so it needs enough ACTIVE experts to be worth its round-based
/// schedule: the rule takes ALGO 3 when
/// `active_ops * kDecodeNTileThreadFactor >= num_threads`, i.e. when there is
/// at most this many threads per active expert.  Below that the team is too
/// wide for the expert count and full-team sequential (ALGO 1) wins — the
/// Mixtral-class shape (8 experts, topk=2 → 2 active on a 32+ thread team)
/// lands there.  Dtype-agnostic by design; only the `n_tile_safe` legality
/// clamp differs per dtype.
inline constexpr int kDecodeNTileThreadFactor = 4;

// ── Abort-class gemm_mode sentinels ─────────────────────────────────────
// The group_matmul executors are void and the surrounding API is noexcept, so
// a condition that leaves the caller's dst wrong is reported by writing one of
// these strings through `gemm_mode_out`.  `group_matmul_direct` and the
// fused-MoE dispatcher turn any mode matching `gemm_mode_is_error` into
// `status_t::failure`, so the caller never consumes a dst the library knows is
// bad.
//
// New sentinels MUST keep the `error_` prefix: the check is by prefix so a
// sentinel added in an executor fails the call closed even if a translation
// site is not updated.
inline constexpr const char *kGrpMatmulErrPrefix = "error_";
inline constexpr const char *kGrpMatmulErrPrepackedNoCk
        = "error_prepacked_no_ck";
/// ALGO 3 could not leave dst defined: a per-thread scratch allocation failed,
/// or a fused-epilogue layout invariant was violated.  `flat_n_tile` logs the
/// specific reason; the sentinel only has to fail the call.
inline constexpr const char *kGrpNTileErrUndefinedDst
        = "error_ntile_undefined_dst";
/// The caller passed a tight destination (`ldc < N`) with a gated activation,
/// but the resolved route has no tight-aware writer, so applying the
/// activation would walk past the end of the caller's buffer.
inline constexpr const char *kGrpMatmulErrTightNoFusedWriter
        = "error_tight_dst_no_fused_writer";
/// A weight-cache lookup missed on a buffer an earlier in-place (WC=2) reorder
/// had already rewritten, so the AOCL backend refused to reorder it a second
/// time (the raw weights are gone and the result would be silently wrong).
/// The backend can only count the refusal -- `run_dlp` returns void and runs
/// inside OMP regions -- so the dispatcher compares the count across the call
/// and raises this, which fails the call rather than returning the dst that
/// the refused reorder left undefined.
inline constexpr const char *kGrpMatmulErrMutatedWeightMiss
        = "error_mutated_weight_miss";

inline bool gemm_mode_is_error(const char *mode) {
    return mode != nullptr
            && std::strncmp(mode, kGrpMatmulErrPrefix,
                       std::strlen(kGrpMatmulErrPrefix))
            == 0;
}

// ── Executed-ALGO from gemm_mode ────────────────────────────────────────
// Maps the executor-written `gemm_mode` string (the authoritative record of
// what ACTUALLY ran) to the ALGO that actually executed ({1,2,3,5,6}), so the
// post-exec `[GRP_MATMUL.CALL]` line can surface `exec_algo=` alongside
// `mode=`.  Together with the pre-exec `[GRP_MATMUL.ALGO] chosen=` selection
// line, this makes any selection-vs-execution divergence explicit instead of
// silent (e.g. a forced ALGO 2 that clamps to sequential-full-team reports
// `chosen=ALGO_2 ... exec_algo=1 mode=flat_m_tile_seq_clamp`).
//
// Returns 0 for null / unrecognised modes OR for explicit no-op markers
// (`*_skip` — nothing executed), and a generic scheduler ID for a recognised
// executed path.
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
    // Abort-class sentinels: the call is being failed, so no ALGO produced a
    // usable result.  Checked first — an `error_` mode must never be reported
    // as a successful execution of the path that raised it.
    if (gemm_mode_is_error(mode)) return 0;
    // No-op / nothing-executed markers map to 0 (must precede the generic
    // prefixes they share, e.g. "flat_m_tile_skip" before "flat_m_tile").
    if (starts("skip")) return 0; // whole-call no-op (all M<=0)
    if (starts("flat_m_tile_skip")) return 0; // empty / no active expert
    if (starts("multilevel_skip")) return 0;
    if (starts("fused_moe_skip")) return 0;
    if (starts("flat_m_tile_seq_clamp")) return 1; // sequential full-team
    if (starts("sequential")) return 1; // "sequential" / "..._experts"
    if (starts("flat_m_tile")) return 2;
    // Must precede the generic `vertical_fusion` arm, which would otherwise
    // claim it by prefix and report an ALGO 5 execution as ALGO 2.
    if (starts("vertical_fusion_expert_parallel")) {
        return kGrpMatmulAlgoExpertParallel;
    }
    if (starts("vertical_fusion")) return 2; // M-tile fused pipeline
    if (starts("flat_n_tile")) return 3;
    if (starts("ntile_flat_parallel")) return kGrpMatmulAlgoNTileFlatParallel;
    if (starts("multilevel")) return kGrpMatmulAlgoMultilevel;
    if (starts("per_expert")) return kGrpMatmulAlgoExpertParallel;
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

// ── Auto-select per-phase settings (consulted only under ALGO=0) ───────
//
// AUTO classifies each call with `classify_grp_matmul_phase()` and reads one
// cached setting for the matching phase:
//
//   AUTO_PROMPT_ALGO default = 2 (flat_m_tile and its default refinements).
//   AUTO_DECODE_ALGO default = 3 (flat_n_tile and its default refinements).
//
// Accepted explicit values are {0,1,2,3,4,5,6}:
// scheduler identities beyond that set:
//   * 0 restores the legacy 3-rule cascade.
//   * {1,2,3,5,6} pin that generic scheduler for the matching phase.
//   * 4 requests the W8A8 whole-call interceptor for the matching phase. If
//     eligibility declines, generic routing inherits the phase's normal
//     default policy (including refinements and safety clamps); numeric 4 is
//     never dispatched as a generic scheduler.
//
// PRECEDENCE: an explicitly-set phase knob for the ACTIVE phase outranks the
// global `ZENDNNL_GRP_MATMUL_ALGO`, whatever the global is set to.  A phase
// `=4` therefore requests W8A8 even under a global generic pin, and an
// explicit non-4 phase knob suppresses a global `=4` for that phase.  The
// global only applies when the matching phase knob is unset.
// Any `unimplemented` attempt may fall through to generic fused-MoE; actual
// ALGO4 allocation/execution errors remain terminal in `group_matmul_direct`.
//
// Strict integer parsing is preserved. Unset, malformed, and out-of-range
// values inherit the documented default and do not count as an explicit
// policy pin. Mid-process env changes have no effect; tests use the atomics.
//
// Value and explicit-presence state intentionally live in ONE cached object.
// Keeping separate static snapshots lets a first-call race or later parser
// edit make `value` and `is_set` disagree.

// Named decode default is also consumed by the qualified ALGO-5 pin fallback
// in auto_select_algo; keeping one constexpr prevents those policies drifting.
inline constexpr int kGrpMatmulAutoDecodeAlgoDefault = 3;

// NOTE: `kGrpMatmulAlgo5PromptDeclineAlgo` moved to
// `expert_parallel/group_matmul_expert_parallel_policy.hpp` together with the
// rest of the ALGO 5 pin qualifier.  Include that header to read it.

inline constexpr int grp_matmul_default_algo_for_phase(grp_matmul_phase phase) {
    return phase == grp_matmul_phase::decode ? kGrpMatmulAutoDecodeAlgoDefault
                                             : 2;
}

struct grp_matmul_auto_phase_setting {
    static constexpr int kNoRequest = -1;

    /// Accepted explicit env/override value, or `kNoRequest` when the setting
    /// is unset, malformed, or outside the accepted selector set.
    int requested_algo = kNoRequest;

    /// Value visible to generic routing. Phase request 4 maps to the inherited
    /// phase default so it can never reach the generic ALGO switch.
    int generic_effective_algo = kGrpMatmulAlgoAuto;

    constexpr bool has_explicit_request() const {
        return requested_algo != kNoRequest;
    }
    constexpr bool requests_ntile_flat_parallel() const {
        return requested_algo == kGrpMatmulAlgoNTileFlatParallel;
    }
    /// Explicit {0,1,2,3,5,6} overrides the inherited policy. Explicit 0
    /// counts because it deliberately selects the legacy cascade. Phase 4
    /// does not: after an ALGO4 decline, all default refinements run.
    constexpr bool pins_generic_policy() const {
        return has_explicit_request() && !requests_ntile_flat_parallel();
    }
};

inline constexpr grp_matmul_auto_phase_setting
make_grp_matmul_auto_phase_setting(grp_matmul_phase phase, int requested) {
    const int default_algo = grp_matmul_default_algo_for_phase(phase);
    if (!is_grp_matmul_requested_algo(requested)) {
        return {grp_matmul_auto_phase_setting::kNoRequest, default_algo};
    }
    return {requested,
            requested == kGrpMatmulAlgoNTileFlatParallel ? default_algo
                                                         : requested};
}

inline grp_matmul_auto_phase_setting parse_grp_matmul_auto_phase_setting(
        const char *env, grp_matmul_phase phase) {
    int parsed = 0;
    if (!parse_env_int_strict(env, parsed)) {
        return make_grp_matmul_auto_phase_setting(
                phase, grp_matmul_auto_phase_setting::kNoRequest);
    }
    return make_grp_matmul_auto_phase_setting(phase, parsed);
}

inline std::atomic<int> &test_api_auto_prompt_algo_override();
inline std::atomic<int> &test_api_auto_decode_algo_override();

inline grp_matmul_auto_phase_setting get_grp_matmul_auto_prompt_setting() {
    const int ovr = test_api_auto_prompt_algo_override().load(
            std::memory_order_relaxed);
    if (ovr >= 0) {
        return make_grp_matmul_auto_phase_setting(
                grp_matmul_phase::prompt, ovr);
    }
    static const grp_matmul_auto_phase_setting setting = []() {
        return parse_grp_matmul_auto_phase_setting(
                std::getenv("ZENDNNL_GRP_MATMUL_AUTO_PROMPT_ALGO"),
                grp_matmul_phase::prompt);
    }();
    return setting;
}

inline grp_matmul_auto_phase_setting get_grp_matmul_auto_decode_setting() {
    const int ovr = test_api_auto_decode_algo_override().load(
            std::memory_order_relaxed);
    if (ovr >= 0) {
        return make_grp_matmul_auto_phase_setting(
                grp_matmul_phase::decode, ovr);
    }
    static const grp_matmul_auto_phase_setting setting = []() {
        return parse_grp_matmul_auto_phase_setting(
                std::getenv("ZENDNNL_GRP_MATMUL_AUTO_DECODE_ALGO"),
                grp_matmul_phase::decode);
    }();
    return setting;
}

inline grp_matmul_auto_phase_setting get_grp_matmul_auto_phase_setting(
        grp_matmul_phase phase) {
    return phase == grp_matmul_phase::decode
            ? get_grp_matmul_auto_decode_setting()
            : get_grp_matmul_auto_prompt_setting();
}

// Compatibility observers used by existing telemetry/tests. Both fields are
// derived from the same setting snapshot; there is no independent cached
// presence bit.
inline int get_grp_matmul_auto_prompt_algo() {
    return get_grp_matmul_auto_prompt_setting().generic_effective_algo;
}
inline int get_grp_matmul_auto_decode_algo() {
    return get_grp_matmul_auto_decode_setting().generic_effective_algo;
}
inline bool grp_matmul_auto_prompt_algo_is_set() {
    return get_grp_matmul_auto_prompt_setting().pins_generic_policy();
}
inline bool grp_matmul_auto_decode_algo_is_set() {
    return get_grp_matmul_auto_decode_setting().pins_generic_policy();
}

/// Can this phase be ruled OUT of ALGO 3 for the whole process?
///
/// PROVABLE, not a heuristic — callers rely on it to skip a warm even
/// while the prompt reorder mutates weights in place.
/// Tracing every route a pinned phase can take through
/// `select_grp_matmul_algo`:
///
///   * pinned 1        -- honoured;
///   * pinned 2        -- honoured, or clamped to 1 when !m_tile_safe;
///   * pinned 6        -- honoured, no tiling precondition to clamp on;
///   * pinned 5        -- honoured, OR DECLINED by the Rule 0.6a occupancy
///                        qualifier, after which decode substitutes the
///                        decode default (3) and prompt substitutes 3
///                        outright.  The ONLY pin that can still land on
///                        ALGO 3;
///   * unset           -- Rules 0.45/0.5/0.6/2a/2c pick 3 on shape.
///
/// So {1, 2, 6} prove ALGO 3 unreachable for that phase and {5, unset} do
/// not.  Treating 5 as reachable costs one redundant warm on a rare pin
/// and buys the guarantee outright, which is a far better trade than
/// forbidding a documented fallback.
inline bool grp_matmul_phase_cannot_reach_algo3(grp_matmul_phase phase) {
    const auto setting = get_grp_matmul_auto_phase_setting(phase);
    if (!setting.pins_generic_policy()) {
        return false;
    } // unset: shape decides
    const int pinned = setting.generic_effective_algo;
    return pinned == 1 || pinned == 2 || pinned == kGrpMatmulAlgoMultilevel;
}

/// The mirror: this phase will run ALGO 3 and nothing else.
inline bool grp_matmul_phase_pinned_to_algo3(grp_matmul_phase phase) {
    const auto setting = get_grp_matmul_auto_phase_setting(phase);
    return setting.pins_generic_policy() && setting.generic_effective_algo == 3;
}

/// Will anything read the layout this cross-warm is about to pack?
///
/// Cross-warm prefills the layout the OTHER inference phase will read, so
/// it only pays off when the two phases land on different ALGOs.
/// `cross_warm` already skips a globally pinned ALGO for that reason, but
/// the per-phase knobs open the same hole under AUTO: with
/// `ZENDNNL_GRP_MATMUL_ALGO=0` and `ZENDNNL_GRP_MATMUL_AUTO_DECODE_ALGO`
/// pinned away from 3, decode never reaches ALGO 3, yet an ALGO 1/2/5/6
/// prepack still eagerly packs the ALGO 3 CK pack or AOCL per-tile arena —
/// on an MoE, a full resident copy of every expert weight warmed for an
/// executor that never runs.
///
/// The two directions are NOT symmetric, because they target different
/// layouts:
///
///   * from ALGO {1,2,5,6} the target is the ALGO 3 DECODE arena (CK pack
///     or AOCL per-tile).  Useless exactly when decode provably cannot
///     reach ALGO 3.
///   * from ALGO 3 the target is the PROMPT full-weight reorder, which
///     every non-3 generic ALGO shares.  Useless only when prompt is
///     pinned to ALGO 3 as well, i.e. when both phases run the same
///     layout.  An UNSET prompt must stay reachable: Rule 0.7 sends it to
///     ALGO 1, which is precisely the case the warm exists for.
inline bool grp_matmul_cross_warm_target_reachable(int current_algo) {
    return current_algo == 3
            ? !grp_matmul_phase_pinned_to_algo3(grp_matmul_phase::prompt)
            : !grp_matmul_phase_cannot_reach_algo3(grp_matmul_phase::decode);
}

enum class grp_matmul_ntile_flat_parallel_request_source {
    none,
    global,
    auto_decode,
    auto_prompt,
};

/// Resolve whether this call should attempt the W8A8 whole-call interceptor.
///
/// PRECEDENCE — the per-phase knob for THIS phase outranks the global
/// selector, matching `select_grp_matmul_algo`.  So:
///   * phase knob `=4`            → attempt W8A8 for this phase, even when a
///                                  global generic pin is set.
///   * phase knob explicitly set
///     to anything else           → no W8A8; the explicit phase choice wins
///                                  over a global `=4`.
///   * phase knob unset           → fall back to the global selector.
///
/// This MUST inspect `get_grp_matmul_requested_algo()` for the global leg: the
/// normalized generic getter maps a global 4 to AUTO and would lose it.
inline grp_matmul_ntile_flat_parallel_request_source
resolve_grp_matmul_ntile_flat_parallel_request(grp_matmul_phase phase) {
    const auto phase_setting = get_grp_matmul_auto_phase_setting(phase);
    if (phase_setting.requests_ntile_flat_parallel()) {
        return phase == grp_matmul_phase::decode
                ? grp_matmul_ntile_flat_parallel_request_source::auto_decode
                : grp_matmul_ntile_flat_parallel_request_source::auto_prompt;
    }
    // An explicitly-set non-4 phase knob is a deliberate choice of a generic
    // scheduler (or of the legacy cascade); it suppresses a global W8A8
    // attempt for this phase.
    if (phase_setting.has_explicit_request()) {
        return grp_matmul_ntile_flat_parallel_request_source::none;
    }

    const int global_requested = get_grp_matmul_requested_algo();
    if (global_requested == kGrpMatmulAlgoNTileFlatParallel) {
        return grp_matmul_ntile_flat_parallel_request_source::global;
    }
    return grp_matmul_ntile_flat_parallel_request_source::none;
}

inline constexpr const char *grp_matmul_ntile_flat_parallel_request_source_name(
        grp_matmul_ntile_flat_parallel_request_source source) {
    switch (source) {
        case grp_matmul_ntile_flat_parallel_request_source::none: return "none";
        case grp_matmul_ntile_flat_parallel_request_source::global:
            return "global";
        case grp_matmul_ntile_flat_parallel_request_source::auto_decode:
            return "auto_decode";
        case grp_matmul_ntile_flat_parallel_request_source::auto_prompt:
            return "auto_prompt";
    }
    return "none";
}

// ZENDNNL_GRP_MATMUL_DENSE_DECODE_NTILE = { 0, 1 } — cached, default 1 (ON).
//   AUTO-only (`ALGO=0`) routing knob for Rule 0.45: a lone expert
//   (`num_ops == 1`) in decode goes to ALGO 3 rather than being diverted by
//   Rule 0.5's low-occupancy arm to ALGO 1, so it reaches the N-tile planner.
//   Set `=0` to restore the generic occupancy decision for A/B measurement.
//   Scope:
//   num_ops==1 + decode + n_tile_safe + phase env NOT explicitly pinned.
//   Non-numeric input → default 1.  The `test_api` override atom below lets
//   a gtest A/B the rule mid-process, which the cached env read cannot.
inline std::atomic<int> &test_api_dense_decode_ntile_override();
inline bool get_grp_matmul_dense_decode_ntile() {
    const int ovr = test_api_dense_decode_ntile_override().load(
            std::memory_order_relaxed);
    if (ovr >= 0) return ovr != 0;
    static const bool v = []() {
        const char *e = std::getenv("ZENDNNL_GRP_MATMUL_DENSE_DECODE_NTILE");
        int parsed = 0;
        if (!parse_env_int_strict(e, parsed)) return true; // default / junk: On
        return parsed != 0;
    }();
    return v;
}

// NOTE: `get_grp_matmul_{decode,prompt}_algo5_gate()` moved to
// `expert_parallel/group_matmul_expert_parallel_policy.hpp` together with the
// ALGO 5 pin qualifier they gate.  Include that header to call them.

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
inline bool a3_can_fuse_act(
        grp_matmul_gated_act_t act, bool use_custom_kernel) {
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

// Sentinel `-1` = no override (use the cached env path, default ON).
// `0` / `1` force Rule 0.45's env gate
// (`ZENDNNL_GRP_MATMUL_DENSE_DECODE_NTILE`) off / on for the lifetime of
// a test.  Needed because the getter caches its env read in a
// function-local `static const`, so a gtest cannot A/B the rule by
// setting the env var after the first call has already latched it.
inline std::atomic<int> s_grp_matmul_dense_decode_ntile_override {-1};

// NOTE: `s_grp_matmul_{decode,prompt}_algo5_gate_override` moved to
// `expert_parallel/group_matmul_expert_parallel_policy.hpp` (same `test_api`
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
inline std::atomic<bool> s_capture_gemm_mode {false};
inline std::atomic<const char *> s_last_group_matmul_direct_gemm_mode {nullptr};

// Whether the deep-K K-blocked BF16 custom-kernel tile ran.  Gated by its
// own capture flag because the decision sits per TILE, not per call: an
// unconditional store would write a shared line from every worker on every
// tile.  The gating load is of a flag that only a test ever flips, so it
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
inline std::atomic<int> &test_api_dense_decode_ntile_override() {
    return test_api::s_grp_matmul_dense_decode_ntile_override;
}
// NOTE: `test_api_{decode,prompt}_algo5_gate_override()` moved to
// `expert_parallel/group_matmul_expert_parallel_policy.hpp` together with the
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
//   wide [M, 2I] when "0".  Tight halves Op2's src DRAM traffic.
//   The dispatcher
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
//     - `ZENDNNL_MATMUL_WEIGHT_CACHE=2` (default): in-place reuse
//       where the path supports it; otherwise pointer-keyed LRU cache
//       stays warm across calls (production MoE serving with a stable
//       model — the common case).  Mode `1` forces out-of-place LRU.
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
//   W4A8 forces the effective per-call value OFF regardless of this env;
//   CK has no s4 microkernel.
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
        int parsed = 0;
        // Strict parse AND domain check.  `e[0] != '0'` used to read
        // "off" / "false" / "no" as ON, i.e. the opposite of what the
        // operator wrote; `parse_env_int_strict` alone still accepted
        // any integer, so `2` or `-1` would enable the kernel.  The
        // documented domain is {0, 1} -- anything else is junk and
        // resolves to the default.
        if (!parse_env_int_strict(e, parsed)) return true;
        if (parsed != 0 && parsed != 1) return true;
        return parsed != 0;
    }();
    return v;
}

/// W4A8 is always AOCL-DLP-only; CK has no s4 microkernel.  Keep this
/// call-scoped so CK remains available for its BF16, INT8, and FP16 families.
///
/// NOTE: this reads the MASTER knob only.  It is the right predicate for a
/// caller that just needs "is the custom kernel family available at all",
/// but NOT for one that must agree with whether the kernel will actually
/// run -- that is `grp_matmul_custom_kernel_effective` below.
inline bool grp_matmul_custom_kernel_enabled(data_type_t wei_dtype,
        data_type_t dst_dtype, data_type_t compute_dtype) {
    return get_grp_matmul_custom_kernel()
            && !(wei_dtype == data_type_t::s4 && dst_dtype == data_type_t::bf16
                    && compute_dtype == data_type_t::s8);
}

// ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_INT8 = { "0", "1" } — cached, default ON.
//   Independent sub-switch for the hand-rolled AVX-512 VNNI int8
//   microkernel (`custom_kernel/ukernel/int8_microkernel.{hpp,cpp}`).
//   Lets deployments compare the DQ-INT8 fast path against the AOCL DLP
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
//     * `_CUSTOM_KERNEL=1 && _CUSTOM_KERNEL_INT8=1` → both on.  DEFAULT.
//
//   Defaults ON so the master knob means one thing for every dtype: a
//   deployment that leaves `_CUSTOM_KERNEL` at its default gets the custom
//   kernel, not the custom kernel for bf16 and the DLP fallback for int8.
//   The split default was a standing trap -- an operator A/B-ing
//   `_CUSTOM_KERNEL` on a W8A8 model changed nothing about which int8
//   kernel ran and drew conclusions from two identical configurations.
//
//   This knob previously defaulted OFF on throughput grounds: the int8 CK
//   can be slower than the DLP sym-quant path on some shapes.  Do not flip
//   it back on that basis alone -- the two paths are not interchangeable on
//   accuracy for every W8A8 configuration, and the CK path is the one this
//   default is validated against.
//
//   Set `0` to route DQ-INT8 to the DLP sym-quant path, for a parity
//   bisection or to measure the throughput difference.
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
        int parsed = 0;
        // Strict parse AND domain check, mirroring the master knob: junk
        // input resolves to the default (now ON) rather than being read as
        // a request to disable.
        if (!parse_env_int_strict(e, parsed)) return true;
        if (parsed != 0 && parsed != 1) return true;
        return parsed != 0;
    }();
    return v;
}

// ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_F16 = { "0", "1" } — cached, default ON.
//   Independent sub-switch for the hand-rolled native AVX-512-FP16
//   microkernel (`custom_kernel/ukernel/f16_microkernel.{hpp,cpp}`).
//   Lets deployments A/B the f16×f16→{f16,f32} fast path against the
//   AOCL DLP F16 reference without affecting the bf16 / int8 CK paths.
//
//   Cascade with the master `_CUSTOM_KERNEL` switch (same shape as the
//   `_INT8` sub-knob):
//     * `_CUSTOM_KERNEL=0`                      → every CK route off.
//     * `_CUSTOM_KERNEL=1 && _CUSTOM_KERNEL_F16=0` → bf16 CK on, f16 CK
//                                                 off (f16 N-tile calls
//                                                 fall back to AOCL DLP
//                                                 F16).  int8 is unaffected
//                                                 and stays ON unless
//                                                 `_INT8=0` turns it off.
//     * `_CUSTOM_KERNEL=1 && _CUSTOM_KERNEL_F16=1` → f16 CK on (default).
//
//   Even when ON, the f16 CK also needs `avx512f16_available()` true —
//   which folds TWO independent conditions with DIFFERENT routing when
//   they fail:
//     * TOOLCHAIN missing the FP16 intrinsics (built with GCC < 12, so
//       the compile-time gate `ZENDNNL_GRP_F16_CK_AVAILABLE == 0`) while
//       the CPU DOES have the ISA: `prepare_for_call` refuses the f16 CK
//       and the call falls back to AOCL DLP F16 (which runs natively on
//       the FP16-capable CPU) — same routing as `_CUSTOM_KERNEL_F16=0`.
//     * CPU missing the AVX-512-FP16 ISA (`get_avx512_f16_status()`
//       false): this is NOT a CK-vs-DLP fallback.  `group_matmul_direct`
//       hard-rejects the WHOLE call for any f16 operand up front with
//       `status_t::isa_unsupported` (the AOCL DLP F16 path cannot serve
//       it either), before the CK dispatch is even reached.  The caller
//       must handle the isa_unsupported status; there is no f16 GEMM on
//       such a host.
//
//   Tests can pin the value via `s_grp_matmul_custom_kernel_f16_override`
//   (sentinel `-1` = no override).
inline bool get_grp_matmul_custom_kernel_f16() {
    const int ovr = test_api::s_grp_matmul_custom_kernel_f16_override.load(
            std::memory_order_relaxed);
    if (ovr >= 0) return ovr != 0;
    static const bool v = []() {
        const char *e = std::getenv("ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_F16");
        int parsed = 0;
        // Strict parse AND domain check.  `e[0] != '0'` used to read
        // "off" / "false" / "no" as ON, i.e. the opposite of what the
        // operator wrote; `parse_env_int_strict` alone still accepted
        // any integer, so `2` or `-1` would enable the kernel.  The
        // documented domain is {0, 1} -- anything else is junk and
        // resolves to the default.
        if (!parse_env_int_strict(e, parsed)) return true;
        if (parsed != 0 && parsed != 1) return true;
        return parsed != 0;
    }();
    return v;
}

/// The EFFECTIVE per-call custom-kernel verdict: the dtype carve-out folded
/// with the family sub-toggle and the per-group disqualifier.
///
/// This exists because `grp_matmul_custom_kernel_enabled` reads the master
/// knob alone, while `engage_ntile_custom_kernel` additionally honours
/// `..._INT8` / `..._F16`, and `flat_n_tile` additionally refuses a per-group
/// `{G, N}` weight scale.  Any decision that has to AGREE with the kernel's
/// real engagement must use this, not the master knob.
///
/// The fused-MoE arena choice is the case that matters.  Granting a tight
/// arena on the master knob and then refusing the kernel on a sub-knob leaves
/// a tight destination with no tight-aware writer, and the split-halves
/// fallback collapses the whole call to the serial Sequential strategy -- so
/// `..._INT8=0` was an order-of-magnitude decode regression rather than the
/// "int8 on DLP, bf16 on CK" split it reads like.
///
/// A representative ACTIVE expert is used: slot 0 may be an inactive padding
/// placeholder whose dtypes were never filled in.
template <typename ParamsVec>
inline bool grp_matmul_custom_kernel_effective(
        const ParamsVec &params, const std::vector<int> &M) {
    size_t rep = params.size();
    for (size_t i = 0; i < params.size(); ++i) {
        if (i < M.size() && M[i] <= 0) { continue; }
        rep = i;
        break;
    }
    if (rep >= params.size()) { return false; }

    const auto &d = params[rep].dtypes;
    if (!grp_matmul_custom_kernel_enabled(d.wei, d.dst, d.compute)) {
        return false;
    }
    // Discriminate the DQ-INT8 family the same way the rest of the tree does
    // (`int8_aocl_warm_candidate` / `ck_eligible_int8`): weight s8 with an
    // s8/u8 compute.  Keying on `dynamic_quant` or on an s8 SOURCE misses the
    // default production shape, because the `group_dynamic_quant` pre-pass
    // produces the s8 source and CLEARS `dynamic_quant` before this runs --
    // so the sub-toggle went unread, the arena came back tight, the kernel
    // then refused, and the tight split-halves path demoted the layer to
    // serial Sequential.  That is the exact cliff this helper exists to stop.
    const bool is_int8_call = d.wei == data_type_t::s8
            && (d.compute == data_type_t::s8 || d.compute == data_type_t::u8);
    if (is_int8_call && !get_grp_matmul_custom_kernel_int8()) { return false; }
    const bool is_f16_call
            = (d.src == data_type_t::f16 && d.wei == data_type_t::f16);
    if (is_f16_call && !get_grp_matmul_custom_kernel_f16()) { return false; }

    // A per-group `{G, N}` weight scale on ANY active expert disqualifies the
    // custom kernel for the whole call -- the CK tiler slices the source
    // scale one scalar per row, i.e. per-token only.  Mirrors `ck_per_group`
    // in `flat_n_tile`.
    for (size_t i = 0; i < params.size(); ++i) {
        if (i >= M.size() || M[i] <= 0) { continue; }
        const auto &ws = params[i].quant_params.wei_scale;
        if (ws.dims.size() == 2 && ws.dims[0] > 1) { return false; }
    }
    return true;
}

// ZENDNNL_GRP_MATMUL_CROSS_WARM = { "0", "1" } — cached, default ON.
//   When ON, each `prepack_for_algo_X` opportunistically populates the
//   cache regime that auto-select would route the OTHER phase to in the
//   same process, so a deployment that fires only prompt during warmup
//   still arrives at decode with both regimes warm — no first-decode-
//   call prepack spike.
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
        if (e == nullptr || e[0] == '\0') return true; // default: On
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
// workloads, the per-call cache-hit savings from avoided reorders
// dominate the per-call thread-utilisation loss from the idle
// threads in a strict-stable plan.
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
//   "0" restores the legacy dynamic plan topology (cache thrash).
inline bool get_grp_matmul_aocl_stable_ntile() {
    static const bool v = []() {
        const char *e = std::getenv("ZENDNNL_GRP_MATMUL_AOCL_STABLE_NTILE");
        if (e == nullptr || e[0] == '\0') return true;
        return e[0] != '0';
    }();
    return v;
}

inline constexpr int kAoclTargetConcurrentSlots = 16; // team-budget divisor
inline constexpr int kAoclBlisNc = 128; // BLIS-bf16 inner-N block

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
//   Deep-K single-thread K-blocking for the BF16 custom kernel (act=none,
//   bias-free).  When ON, the per-tile dispatcher splits the K reduction
//   into L2-resident chunks so each o-block's B strip is streamed ONCE and
//   reused across all M-blocks instead of being re-streamed
//   `ceil(M / max_mr)` times.  Engages only when the strip exceeds the L2
//   budget and M > max_mr; every other shape and the fused / biased paths
//   are untouched.  `CallContext::kblock_auto` turns it on for single-expert
//   decode without the env.
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
        case matmul_algo_t::native_gemm: return 64;
        default: return 1;
    }
}

/// The per-thread N-slice alignment an upcoming ALGO 3 decode will split
/// on, resolved from a prompt-shaped call so `cross_warm` can warm
/// per-tile AOCL DLP keys the decode will actually query.
///
/// The per-tile cache key embeds `n_tile = aligned_n_split(N, n_thr, ...,
/// nr_align)`, so warming at a different alignment prefills keys the
/// runtime never looks up and every tile misses on the first decode call.
///
/// This mirrors `flat_n_tile`'s own derivation and must keep mirroring it:
///
///   * the dispatcher hands ALGO 3 `a3_can_fuse_act(act, CK) ? act : none`,
///     so silu / gelu with the custom kernel off arrive as `none` and the
///     arena is wide;
///   * a fused epilogue with `ldc[0] < N[0]` is the tight arena, whose OOP
///     writer packs at `col_start / 2` and so needs an even `col_start` on
///     every thread;
///   * with the custom kernel engaged `pack_nr` (32 or 64) is already even,
///     and that regime is warmed by the custom-kernel warmer instead, so
///     only the CK-off case has to be right here.
///
/// Returns the `aocl_dlp_blocked` backend alignment (1), widened to 2 for
/// the tight pair-aligned case.  This only selects which keys get warmed;
/// a wrong answer costs a first-call reorder, never correctness.
/// `ck_enabled` must be the EFFECTIVE per-call verdict, not the master env
/// knob: the dtype-aware `grp_matmul_custom_kernel_enabled` folded with the
/// family sub-toggle.  Passing the raw env mispredicts every family the
/// custom kernel structurally cannot serve -- W4A8 always, and int8 or f16
/// whenever their sub-knob is off -- because those reach decode with the
/// custom kernel disabled and therefore split on the tight alignment.
///
/// `decode_arena_tight` is the arena the DECODE call will use, which is not
/// observable from the prompt call's `ldc`.  Under AUTO the prompt resolves
/// to ALGO 1, and `pick_fused_moe_want_tight` requires `resolved_algo == 3`,
/// so a library-owned Op1 arena is ALWAYS wide on the prompt and always
/// tight on the decode.  Reading `ldc[0] < N[0]` here would therefore invert
/// the prediction on exactly the fused-MoE layers this exists to serve.
/// Callers that know the decode arena (the fused-MoE entry) pass it; callers
/// that own their own destination and keep one stride across both phases
/// pass their observed tightness.
inline int algo3_decode_nr_align(
        grp_matmul_gated_act_t act, bool ck_enabled, bool decode_arena_tight) {
    const int backend_nr = backend_n_align(matmul_algo_t::aocl_dlp_blocked);
    // Mirrors `ntile_effective_nr_align`: the custom kernel widens to its
    // own pack_nr and owns the pairing, so only the CK-off tight case needs
    // the pair alignment.  When the custom kernel will run, the per-tile
    // AOCL keys are not what decode queries at all -- the custom-kernel pack
    // arena is -- so the value here is immaterial and `backend_nr` is right.
    const bool tight_pair_align
            = !ck_enabled && decode_arena_tight && a3_can_fuse_act(act, false);
    return tight_pair_align ? std::max(backend_nr, 2) : backend_nr;
}

/// Decode-arena hint, published by the fused-MoE entry for the duration of
/// one Op1 dispatch on the SAME thread.
///
/// The fused-MoE entry is the only place that can answer "will the decode
/// call get a tight arena?", because it owns the arena and can evaluate
/// `pick_fused_moe_want_tight(..., resolved_algo = 3)` regardless of what
/// this particular call resolved to.  The prompt-side prepack lives several
/// frames down inside the ALGO executors, and threading a parameter there
/// would touch the dispatcher signature and every executor.  A scoped
/// thread-local is the narrower change; the RAII guard means it cannot
/// outlive the dispatch that set it, and nothing reads it off-thread.
inline thread_local bool tls_decode_arena_tight = false;
inline thread_local bool tls_decode_arena_known = false;

class scoped_decode_arena_hint {
public:
    explicit scoped_decode_arena_hint(bool tight)
        : prev_tight_(tls_decode_arena_tight)
        , prev_known_(tls_decode_arena_known) {
        tls_decode_arena_tight = tight;
        tls_decode_arena_known = true;
    }
    ~scoped_decode_arena_hint() {
        tls_decode_arena_tight = prev_tight_;
        tls_decode_arena_known = prev_known_;
    }
    scoped_decode_arena_hint(const scoped_decode_arena_hint &) = delete;
    scoped_decode_arena_hint &operator=(const scoped_decode_arena_hint &)
            = delete;

private:
    bool prev_tight_;
    bool prev_known_;
};

/// Convenience overload for the ALGO executors.  Uses the fused-MoE hint
/// when one is published, and otherwise falls back to the caller's own
/// observed stride -- correct for a caller that owns its destination and
/// keeps one layout across prompt and decode.
template <typename ParamsVec>
inline int algo3_decode_nr_align(const std::vector<int> &M,
        const std::vector<int> &N, const std::vector<int> &ldc,
        grp_matmul_gated_act_t act, const ParamsVec &params) {
    // Classify from the first ACTIVE expert, not slot 0.  An inactive slot
    // carries arbitrary placeholder strides, and both the dispatcher and
    // `flat_n_tile` pick their representative the same way -- reading slot 0
    // here instead could call a wide arena tight, warm alignment-2 keys, and
    // leave decode querying alignment-1 keys.  That reinstates the very
    // first-call reorder this alignment agreement exists to remove.
    size_t rep = M.size();
    for (size_t i = 0; i < M.size(); ++i) {
        if (M[i] > 0) {
            rep = i;
            break;
        }
    }
    // All-inactive (or no metadata): nothing to classify from, so keep the
    // backend default rather than inventing a verdict.
    const bool have_rep = rep < M.size() && rep < N.size() && rep < ldc.size();
    const bool observed_tight = have_rep && ldc[rep] < N[rep];
    const bool tight
            = tls_decode_arena_known ? tls_decode_arena_tight : observed_tight;
    return algo3_decode_nr_align(
            act, grp_matmul_custom_kernel_effective(params, M), tight);
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
inline std::pair<int, int> aligned_n_split(
        int N, int n_thr, int tid, int align) {
    // Hardened against pathological inputs: n_thr<=0 used to hit
    // even_split's divide-by-zero (`N * tid / n_thr`).
    if (n_thr <= 0 || N <= 0) { return std::make_pair(0, 0); }

    auto even_split = [&]() {
        const int s = static_cast<int>(static_cast<int64_t>(N) * tid / n_thr);
        const int e
                = static_cast<int>(static_cast<int64_t>(N) * (tid + 1) / n_thr);
        return std::make_pair(s, e);
    };

    if (align <= 1 || n_thr <= 1) { return even_split(); }

    // Walk slice size down in `align` quanta from ceil(N/n_thr) until
    // the imbalance bound holds.  Cost is at most a handful of
    // iterations in practice (slice sizes converge in 1-2 steps for
    // realistic N / n_thr / align triples).
    //
    // Intermediates promoted to int64_t to keep the products
    // aligned_per_thr * (n_thr - 1) and aligned_per_thr * tid free of
    // signed overflow (UB) for any (N, n_thr) combination representable
    // as int.
    const int64_t even_per_thr = (static_cast<int64_t>(N) + n_thr - 1) / n_thr;
    for (int64_t aligned_per_thr = ((even_per_thr + align - 1) / align) * align;
            aligned_per_thr >= align; aligned_per_thr -= align) {
        const int64_t n_full = aligned_per_thr * (n_thr - 1);
        const int64_t last = static_cast<int64_t>(N) - n_full;
        if (last > 0 && last * 2 >= aligned_per_thr) {
            const int64_t s = aligned_per_thr * tid;
            const int64_t e
                    = (tid < n_thr - 1) ? aligned_per_thr * (tid + 1) : N;
            return std::make_pair(static_cast<int>(s), static_cast<int>(e));
        }
    }
    return even_split();
}

/// Wrapper around aligned_n_split with optional even boundary snapping for
/// packed-s4 N-tile slices.
inline std::pair<int, int> n_split_for_tile(
        int N, int n_thr, int tid, int align, bool even_boundaries) {
    auto s = aligned_n_split(N, n_thr, tid, align);
    if (!even_boundaries) { return s; }
    s.first &= ~1;
    s.second = (tid == n_thr - 1) ? N : (s.second & ~1);
    return s;
}

/// Byte offset of column `col_start` in a nibble-packed s4 B matrix.
inline size_t packed_s4_col_byte_off(int col_start, int ldb, bool transB) {
    const size_t nib = transB
            ? static_cast<size_t>(col_start) * static_cast<size_t>(ldb)
            : static_cast<size_t>(col_start);
    return nib >> 1;
}

/// 32 MB of L3 per CCD on current-generation classic-CCD CPU topologies.
inline constexpr size_t kL3PerCcdBytes = 32UL * 1024UL * 1024UL;

/// Aggregate L3 the planner uses to bound the experts-per-round
/// budget (ALGO 3 N-tile, ALGO 6 multilevel).  `num_ccds` comes from
/// summarise_topology() so this stays consistent with how the rest
/// of the planner partitions the team.
inline size_t get_grp_l3_total_bytes(int num_ccds) {
    return static_cast<size_t>(std::max(1, num_ccds)) * kL3PerCcdBytes;
}

// ──────────────────────────────────────────────────────────────────────
// Shared per-expert primitives.
// ──────────────────────────────────────────────────────────────────────

/// Resolves the effective matmul algo ID from the runtime config,
/// falling back to AOCL DLP blocked (or reference when AOCL is absent)
/// when the config is unset/invalid.
inline matmul_algo_t resolve_kernel() {
    static const matmul_algo_t algo = []() {
        int32_t a = matmul_config_t::instance().get_algo();
        matmul_algo_t kernel;
        if (a <= 0 || a >= static_cast<int32_t>(matmul_algo_t::algo_count)) {
#if ZENDNNL_DEPENDS_AOCLDLP
            kernel = matmul_algo_t::aocl_dlp_blocked;
#else
            kernel = matmul_algo_t::reference;
#endif
        } else {
            kernel = static_cast<matmul_algo_t>(a);
        }
#if !ZENDNNL_DEPENDS_AOCLDLP
        if (kernel == matmul_algo_t::aocl_dlp
                || kernel == matmul_algo_t::aocl_dlp_blocked
                || kernel == matmul_algo_t::batched_sgemm) {
            kernel = matmul_algo_t::reference;
        }
#endif
        return kernel;
    }();
    return algo;
}

/// W4A8 matmul policy:
///   * group AUTO: ALGO 3 is native s4; every other scheduler is simulated;
///   * pinned group ALGO: preserve the inner-kernel choice.
inline matmul_algo_t w4a8_runtime_algo(
        int scheduling_algo, matmul_algo_t inner_kernel) {
    if (get_grp_matmul_algo() == kGrpMatmulAlgoAuto) {
        return scheduling_algo == 3 ? matmul_algo_t::aocl_dlp_blocked
                                    : matmul_algo_t::aocl_dlp;
    }
    return inner_kernel == matmul_algo_t::aocl_dlp_blocked
            ? matmul_algo_t::aocl_dlp_blocked
            : matmul_algo_t::aocl_dlp;
}

/// Resolve the effective inner kernel for one expert under the selected
/// group-matmul scheduler.  Most dtype families preserve `inner_kernel`;
/// scheduler-specific policies (currently W4A8 native-vs-simulated routing)
/// are centralized here so individual executors do not encode dtype policy.
inline matmul_algo_t resolve_expert_kernel(int scheduling_algo,
        matmul_algo_t inner_kernel, const matmul_params &params) {
    return is_w4a8_config(params)
            ? w4a8_runtime_algo(scheduling_algo, inner_kernel)
            : inner_kernel;
}

/// The per-thread dynamic-quant buffer pool `execute_expert_slice` quantizes
/// through.  Pooled because the per-expert executors (ALGO 1 / 5 / 6) reach it
/// once per expert inside the parallel region, where a stack instance would
/// cost a malloc/free pair each time.  Named rather than function-local so
/// `reset_thread_local_fused_moe_state()` can `release()` it, as with
/// `ntile_flat_parallel::reset_thread_local_scratch()`.
inline reorder_quant_buffers_t &get_thread_local_quant_buffers() {
    static thread_local reorder_quant_buffers_t buffers;
    return buffers;
}

/// Status-bearing form of `execute_expert_slice`, for callers that can
/// propagate a failure instead of only logging it.  The per-expert source
/// quantization can fail at RUNTIME (a pooled allocation, not just bad
/// metadata a gate could screen), and on that path the destination is left
/// unwritten — so a caller that reports success regardless would publish a
/// stale intermediate.  `matmul_execute` returns void, so the quantization
/// is the only stage that can report anything.
///
/// `execute_expert_slice` remains the void form for the callers that have no
/// way to carry a status out (ALGO 1/6 and the N-tile executors run this
/// inside parallel regions whose contracts are unchanged).
inline status_t execute_expert_slice_checked(char layout, bool transA,
        bool transB, int M, int N, int K, float alpha, const void *src, int lda,
        const void *weight, int ldb, const void *bias, float beta, void *dst,
        int ldc, bool is_weights_const, int num_thr, matmul_params &params,
        matmul_algo_t algo) {

    // Inactive expert (no routed tokens): nothing to compute.  Returning
    // early keeps the per-expert ALGOs (1/5/6) from driving the backend
    // GEMM with M == 0, where tile-count math divides by the row count and
    // traps (SIGFPE).  The M-tile / N-tile ALGOs flatten over rows and skip
    // empty experts implicitly, so this is the only path that needs the
    // guard.  Sparse MoE routing (e.g. 6 of 15 experts firing) relies on it.
    if (M <= 0) { return status_t::success; }

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
    int reordered_lda = lda;
    size_t src_type_size = size_of(params.dtypes.src);
    // Pooled per thread, not constructed per expert: each worker owns its
    // experts and consumes the buffer within one call, so the grow-only
    // reuse is safe.
    reorder_quant_buffers_t &quant_buffers = get_thread_local_quant_buffers();
    if (reorder_quantization_wrapper(src, lda, reordered_lda, src_type_size,
                params, bp, transA, M, K, num_thr, quant_buffers)
            != status_t::success) {
        log_error("execute_expert_slice: reorder_quantization_wrapper failed");
        return status_t::failure;
    }

    matmul_execute(layout, transA, transB, M, N, K, alpha, src,
            params.dynamic_quant ? reordered_lda : lda, weight, ldb, bias, beta,
            dst, ldc, is_weights_const, src_type_size,
            size_of(params.dtypes.dst), num_thr, kernel, params, bp, 0);
    return status_t::success;
}

/// Thin wrapper around matmul_execute that packages per-expert slice
/// arguments into the batch/params objects the kernel expects.
inline void execute_expert_slice(char layout, bool transA, bool transB, int M,
        int N, int K, float alpha, const void *src, int lda, const void *weight,
        int ldb, const void *bias, float beta, void *dst, int ldc,
        bool is_weights_const, int num_thr, matmul_params &params,
        matmul_algo_t algo) {
    // Failure is already logged by the checked form; discarding it here keeps
    // the historical contract of this entry point.
    (void)execute_expert_slice_checked(layout, transA, transB, M, N, K, alpha,
            src, lda, weight, ldb, bias, beta, dst, ldc, is_weights_const,
            num_thr, params, algo);
}

/// ZENDNNL_ENABLE_GROUP_DQ — opt-in/out for the grouped dynamic-quant
/// pre-pass (`group_dynamic_quant`).  Default ON (unset or any value
/// other than a literal "0").  When OFF, group matmul skips the grouped
/// source-quant pre-pass and dynamic quantization falls back to the
/// per-expert `reorder_quantization_wrapper` inside `execute_expert_slice`
/// (legacy path).
///
/// The env read is latched: this sits on the serial critical path once per
/// fused-MoE call, and an int8 MoE decode call is only a few microseconds
/// against a `getenv` that scans the whole environment block.  Latching makes
/// a later `setenv` invisible, so tests use
/// `s_grp_matmul_enable_group_dq_override` (sentinel -1 = no override).
inline bool get_grp_matmul_enable_group_dq() {
    const int ovr = test_api::s_grp_matmul_enable_group_dq_override.load(
            std::memory_order_relaxed);
    if (ovr >= 0) return ovr != 0;
    static const bool v = []() {
        const char *env = std::getenv("ZENDNNL_ENABLE_GROUP_DQ");
        if (env == nullptr || env[0] == '\0') return true; // default ON
        return !(env[0] == '0' && env[1] == '\0'); // "0" => OFF
    }();
    return v;
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

/// Peek at the generic ALGO the dispatcher would pick for this call (1=seq,
/// 2=m_tile, 3=n_tile, 5=per_expert, 6=multilevel).  Mirrors the
/// dispatcher's full gating: ZENDNNL_GRP_MATMUL_ALGO override, m/n
/// tile-safety checks, auto_select_algo on env=0.  Pure observer
/// (no side-effects); used by the fused-MoE entry to choose tight
/// vs wide arena before committing the buffer layout.
// What the AUTO heuristic decided and why.  Filled by the selector itself so
// the `[GRP_MATMUL.ALGO]` line reports the rule that actually matched instead
// of re-deriving a parallel copy of the rule table that drifts out of step
// with it.  `reason` stays null when AUTO did not run (a global ALGO pin), and
// `unclamped` is what the matched rule answered before the m_tile_safe /
// n_tile_safe correctness clamps, so the log can tell a deliberate pick from
// a safety downgrade.
struct auto_algo_trace {
    const char *reason = nullptr;
    int unclamped = 0;
    // Set true when Rule 0.6a's qualifier HONOURED a decode `AUTO_DECODE_ALGO=5`
    // pin for an s8 saturated-team frame.  The pin then resolves to ALGO 5
    // through Rule 1 (`reason=auto_phase_env`), so this dedicated boolean is
    // what keeps the `[GRP_MATMUL.ALGO]` line greppable for the qualified
    // pin→ALGO-5 route the generic reason string does not name.
    bool decode5_pin_honoured = false;
    // Set true when a decode `AUTO_DECODE_ALGO=5` pin was DECLINED by the
    // qualifier (not all-s8, or active_ops < num_threads) and the call fell back
    // to the decode default exactly as if the pin were unset.
    bool decode5_pin_declined = false;
    // Prompt twins of the two booleans above, kept separate so one
    // `[GRP_MATMUL.ALGO]` line still names which phase's pin was involved.
    bool prompt5_pin_honoured = false;
    // Declined on occupancy alone (the prompt gate has no dtype term); the
    // call falls back to ALGO 3, not to the prompt default.
    bool prompt5_pin_declined = false;
};

int select_grp_matmul_algo(const std::vector<char> &layout,
        const std::vector<int> &M, const std::vector<int> &N,
        const std::vector<int> &K, const std::vector<matmul_params> &params,
        int num_threads, auto_algo_trace *trace = nullptr);

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_PARALLEL_COMMON_HPP
