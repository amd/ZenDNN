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

/// AUTO algorithm selection: per-phase settings, WC=2 and cross-warm
/// phase checks, the N-tile flat parallel request resolver and
/// `group_matmul_algo_select`.

#ifndef ZENDNNL_GROUP_MATMUL_ALGO_SELECT_HPP
#define ZENDNNL_GROUP_MATMUL_ALGO_SELECT_HPP

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <vector>

#include "common/op_config.hpp"
#include "lowoha_operators/matmul/group_matmul/common/grp_matmul_constants.hpp"
#include "lowoha_operators/matmul/group_matmul/common/grp_matmul_env.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/multi_expert_policy.hpp"
#include "lowoha_operators/matmul/group_matmul/selection/call_route.hpp"
#include "lowoha_operators/matmul/group_matmul/single_expert/single_expert_policy.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

// ── Auto-select per-phase settings (consulted only under ALGO=0) ───────
//
// AUTO classifies each call with `classify_grp_matmul_phase()` and reads one
// cached setting for the matching phase:
//
//   AUTO_PROMPT_ALGO inherited value = 2 for compatibility; Rule 0.7 sends
//   every unpinned prompt to ALGO 1. ALGO 2 therefore remains explicit-only.
//   AUTO_DECODE_ALGO default = 3 (n_tile_execute and its default refinements).
//
// An ALGO-3 MoE layer (`grp_matmul_auto_algo3_layer`: at least two experts
// in a custom-kernel family) runs ALGO 3 in both phases instead while its
// phase is unset.
//
// Accepted explicit values are {0,1,2,3,4,5,6}:
// scheduler identities beyond that set:
//   * 0 restores the legacy 3-rule cascade.
//   * {1,2,3,5,6} pin that generic scheduler for the matching phase.
//   * 4 requests the N-tile flat-parallel MoE executor for that
//     phase's `routed_fused_moe_direct` calls and its `group_matmul_direct`
//     calls of two or more operations, which `group_matmul_direct` sends to
//     it before this selector. A call that reaches this selector under it (a
//     grouped call of one operation) inherits the phase's normal default
//     policy (including refinements and safety clamps); numeric 4 is never
//     dispatched as a generic scheduler.
//
// PRECEDENCE: an explicitly-set phase knob for the ACTIVE phase outranks the
// global `ZENDNNL_GRP_MATMUL_ALGO`, whatever the global is set to.  A phase
// `=4` therefore requests N-tile flat parallel even under a global generic pin,
// and an explicit non-4 phase knob suppresses a global `=4` for that phase.
// The global only applies when the matching phase knob is unset.
// A call N-tile flat parallel owns and cannot serve fails with `unimplemented`
// (or a memory status) before writing anything; it does not run elsewhere.
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
// `multi_expert/expert_parallel/group_matmul_expert_parallel_policy.hpp` together with the
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
    /// does not: under N-tile flat parallel, generic calls run all default
    /// refinements.
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

/// Statically-known scheduler after phase-over-global precedence, or 0 when
/// the phase request is shape-dependent (`=0`, `=4`, or qualified `=5`).
inline int grp_matmul_static_algo_for_phase(grp_matmul_phase phase) {
    const auto setting = get_grp_matmul_auto_phase_setting(phase);
    if (setting.has_explicit_request()) {
        if (!setting.pins_generic_policy()) return 0;
        const int algo = setting.generic_effective_algo;
        return (algo == 0 || algo == kGrpMatmulAlgoExpertParallel) ? 0 : algo;
    }
    const int global = get_grp_matmul_algo();
    return is_grp_matmul_generic_algo(global) ? global : 0;
}

/// A non-AUTO global pin is unsupported under WC=2 when higher-precedence
/// phase policy can select different/unknown layout owners.
inline bool grp_matmul_wc2_global_phase_split_unsupported() {
    if (get_grp_matmul_algo() == kGrpMatmulAlgoAuto) return false;
    const int prompt
            = grp_matmul_static_algo_for_phase(grp_matmul_phase::prompt);
    const int decode
            = grp_matmul_static_algo_for_phase(grp_matmul_phase::decode);
    return prompt == 0 || decode == 0 || prompt != decode;
}

/// Call-level form of the policy above. Each weight pool is considered
/// independently so fused MoE covers both W13 and W2: a pool is hazardous only
/// when one of its active weights can be rewritten in place at WC=2 — a
/// BF16 / FP16 multi-expert layer or a single dense layer under a
/// single-expert CK policy (their custom-kernel pack), or an S8 weight with a
/// declared writable capacity (its AOCL reorder).  INT8 / W4A8 custom-kernel
/// packs are always out of place.  The AUTO-only MoE knobs do not enter: under
/// a pin they do not decide who writes the buffer.
inline bool grp_matmul_wc2_global_phase_split_unsupported_for_call(
        const std::vector<int> &M, const std::vector<int> &N,
        const std::vector<int> &K, const std::vector<matmul_params> &params,
        int num_threads,
        const std::vector<matmul_params> *secondary_params = nullptr,
        const std::vector<int> *secondary_N = nullptr,
        const std::vector<int> *secondary_K = nullptr) {
    if (matmul_config_t::instance().get_weight_cache() != 2
            || !grp_matmul_wc2_global_phase_split_unsupported()) {
        return false;
    }
    // While N-tile flat parallel is requested, `group_matmul_direct` caps the
    // MoE layer calls it does not run (the other phase's, and calls of one
    // operation with a fused down projection, a weighted reduce or prepack
    // extras) at out of place, so they reach this check with no weight to
    // rewrite.
    const int algo3_layer_prompt
            = grp_matmul_static_algo_for_phase(grp_matmul_phase::prompt);
    const int algo3_layer_decode
            = grp_matmul_static_algo_for_phase(grp_matmul_phase::decode);
    const bool algo3_layer_split = algo3_layer_prompt == 0
            || algo3_layer_decode == 0
            || algo3_layer_prompt != algo3_layer_decode;
    const auto pool_is_hazardous = [&](const std::vector<matmul_params> &pool,
                                           const std::vector<int> &pool_N,
                                           const std::vector<int> &pool_K) {
        const bool float_layer = grp_matmul_float_moe_layer(M, pool)
                && (algo3_layer_split
                        || grp_matmul_auto_algo3_family(M, pool_N, pool_K, pool)
                                != grp_matmul_algo3_family::float_family);
        const bool ck_owned = float_layer
                || grp_matmul_single_dense_call(
                        M, pool_N, pool_K, pool, num_threads);
        for (size_t i = 0; i < M.size() && i < pool.size(); ++i) {
            if (M[i] > 0 && pool[i].weight_cache_type >= 2
                    && (ck_owned
                            || (pool[i].dtypes.wei == data_type_t::s8
                                    && pool[i].wei_buffer_capacity_bytes > 0)))
                return true;
        }
        return false;
    };
    return pool_is_hazardous(params, N, K)
            || (secondary_params != nullptr && secondary_N != nullptr
                    && secondary_K != nullptr
                    && pool_is_hazardous(
                            *secondary_params, *secondary_N, *secondary_K));
}

/// Can this phase be ruled OUT of ALGO 3 for the whole process?
///
/// PROVABLE, not a heuristic — callers rely on it to skip a warm even
/// while the prompt reorder mutates weights in place.
/// Tracing every route a pinned phase can take through
/// `group_matmul_algo_select`:
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

/// Can this phase of a `grp_matmul_auto_algo3_layer` run an
/// n_tile_safe call on an ALGO other than 3?  Unset, it runs ALGO 3.  A pin
/// other than 3 can; the legacy cascade (`=0`) and an `=5` pin may pick
/// another ALGO, so they count.
inline bool grp_matmul_phase_can_leave_algo3(grp_matmul_phase phase) {
    const auto setting = get_grp_matmul_auto_phase_setting(phase);
    return setting.pins_generic_policy() && setting.generic_effective_algo != 3;
}

/// Does AUTO keep both phases of a `grp_matmul_auto_algo3_layer` layer on
/// ALGO 3?  Its custom-kernel pack is then shared by both phases, and no
/// AOCL DLP layout of the layer is written in place.
inline bool grp_matmul_auto_algo3_only(bool auto_algo3_layer) {
    return auto_algo3_layer
            && !grp_matmul_phase_can_leave_algo3(grp_matmul_phase::prompt)
            && !grp_matmul_phase_can_leave_algo3(grp_matmul_phase::decode);
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
///
/// A `grp_matmul_auto_algo3_layer` runs both unset phases on
/// ALGO 3, so either target has a reader only when a pin sends some phase
/// off ALGO 3, and the ALGO 3 arena only while some phase can still reach it.
inline bool grp_matmul_cross_warm_target_reachable(
        int current_algo, bool auto_algo3_layer) {
    if (!auto_algo3_layer) {
        return current_algo == 3
                ? !grp_matmul_phase_pinned_to_algo3(grp_matmul_phase::prompt)
                : !grp_matmul_phase_cannot_reach_algo3(
                          grp_matmul_phase::decode);
    }
    const bool a_phase_leaves_algo3
            = grp_matmul_phase_can_leave_algo3(grp_matmul_phase::prompt)
            || grp_matmul_phase_can_leave_algo3(grp_matmul_phase::decode);
    if (current_algo == 3) { return a_phase_leaves_algo3; }
    return a_phase_leaves_algo3
            && (!grp_matmul_phase_cannot_reach_algo3(grp_matmul_phase::prompt)
                    || !grp_matmul_phase_cannot_reach_algo3(
                            grp_matmul_phase::decode));
}

enum class grp_matmul_ntile_flat_parallel_request_source {
    none,
    global,
    auto_decode,
    auto_prompt,
};

/// Resolve whether this phase requests the N-tile flat-parallel MoE executor.
///
/// PRECEDENCE — the per-phase knob for THIS phase outranks the global
/// selector, matching `group_matmul_algo_select`.  So:
///   * phase knob `=4`            → N-tile flat parallel for this phase,
///                                  even when a global generic pin is set.
///   * phase knob explicitly set
///     to anything else           → no N-tile flat parallel; the explicit
///                                  phase choice wins over a global `=4`.
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
    // scheduler (or of the legacy cascade); it suppresses a global N-tile flat
    // parallel request for this phase.
    if (phase_setting.has_explicit_request()) {
        return grp_matmul_ntile_flat_parallel_request_source::none;
    }

    const int global_requested = get_grp_matmul_requested_algo();
    if (global_requested == kGrpMatmulAlgoNTileFlatParallel) {
        return grp_matmul_ntile_flat_parallel_request_source::global;
    }
    return grp_matmul_ntile_flat_parallel_request_source::none;
}

// NOTE: the single-expert policy — the Rule 0.45 knob, the single-expert
// custom-kernel knobs, `grp_matmul_single_expert_ck` and
// `single_expert_ck_epilogue_supported` — moved to
// `single_expert/single_expert_policy.hpp`.

// NOTE: `get_grp_matmul_{decode,prompt}_algo5_gate()` moved to
// `multi_expert/expert_parallel/group_matmul_expert_parallel_policy.hpp` together with the
// ALGO 5 pin qualifier they gate.  Include that header to call them.

// NOTE: `get_grp_n_tile_fused_act()` moved to
// `group_matmul_n_tile.hpp` (Section A.4) together with the rest of
// the N-tile env getters.  Include that header to call it.

// ──────────────────────────────────────────────────────────────────────
// Tile-strategy entry points — library-internal linkage; the
// dispatcher in group_matmul_dispatch.cpp forwards to these.
// ──────────────────────────────────────────────────────────────────────

// NOTE: The M-tile executor forward declarations (`m_tile_execute`
// and `try_fused_m_tile_core`) moved to
// `group_matmul_m_tile.hpp` (Section H.4).  Include that header to
// call them — the dispatcher in `group_matmul_dispatch.cpp` and
// the vertical-fusion entry in `group_matmul_fused_moe.cpp` both
// do this.

// NOTE: The N-tile executor forward declaration (`n_tile_execute`)
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
struct GroupMatmulAlgoTrace {
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

/// `single_expert_epilogue_ok` is false when the custom kernel would refuse a
/// single expert's epilogue (see `single_expert_ck_epilogue_supported`); the
/// selector sees none of those terms itself.  `route`, when given, carries
/// the dispatcher's single-expert and MoE verdicts for these inputs, so the
/// call is classified once.
int group_matmul_algo_select(const std::vector<char> &layout,
        const std::vector<int> &M, const std::vector<int> &N,
        const std::vector<int> &K, const std::vector<matmul_params> &params,
        int num_threads, GroupMatmulAlgoTrace *trace = nullptr,
        grp_matmul_gated_act_t fused_act = grp_matmul_gated_act_t::none,
        bool single_expert_epilogue_ok = true,
        const grp_matmul_call_route *route = nullptr);

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_ALGO_SELECT_HPP
