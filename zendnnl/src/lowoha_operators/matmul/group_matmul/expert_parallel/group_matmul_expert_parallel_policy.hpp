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

/// ALGO 5 — expert-parallel SELECTION POLICY.
///
/// Companion to `group_matmul_expert_parallel.hpp`: that header declares the
/// executor, this one owns the policy deciding whether a call may reach it.
/// ALGO 5 has no planner, so this is its analogue of the M-tile / N-tile
/// planner headers — the per-ALGO policy surface living with its ALGO rather
/// than in the shared dispatcher.  It holds the two pin-qualifier env getters
/// and their `test_api` atoms, `kGrpMatmulAlgo5PromptDeclineAlgo`, and
/// `decide_algo5_pin()` itself.
///
/// Library-internal.  Included by `group_matmul_dispatch.cpp` (the selector)
/// and `gtests/group_matmul/moe_test_utils.hpp` (the RAII override guards).

#ifndef ZENDNNL_GROUP_MATMUL_EXPERT_PARALLEL_POLICY_HPP
#define ZENDNNL_GROUP_MATMUL_EXPERT_PARALLEL_POLICY_HPP

#include <atomic>
#include <vector>

#include "../group_matmul_parallel_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

// Where a DECLINED `AUTO_PROMPT_ALGO=5` pin lands.  Deliberately NOT the
// prompt default (2): the pin is declined because the active experts cannot
// saturate the team, and ALGO 3's intra-expert N-split is what fills it.
// Decode declines to `kGrpMatmulAutoDecodeAlgoDefault`, the same ALGO but
// meaning "as if the pin were unset", so the two stay separate constants and
// a change to the decode default cannot silently redirect the prompt one.
inline constexpr int kGrpMatmulAlgo5PromptDeclineAlgo = 3;

namespace test_api {

// Sentinel `-1` = no override (cached env path, default ON).  `0` / `1` force
// `ZENDNNL_GRP_MATMUL_DECODE_ALGO5_GATE` off / on for a test's lifetime.  The
// getters below latch their env read, so a mid-process `setenv` cannot flip
// the path; every atom here exists for that reason.
inline std::atomic<int> s_grp_matmul_decode_algo5_gate_override {-1};

// Prompt twin of the atom above
// (`ZENDNNL_GRP_MATMUL_PROMPT_ALGO5_GATE`).
inline std::atomic<int> s_grp_matmul_prompt_algo5_gate_override {-1};

// Forces the fused W13 -> act -> W2 pipeline
// (`ZENDNNL_GRP_MATMUL_ALGO5_VERTICAL_FUSION`) off / on, so a gtest can A/B it
// against the two-pass over the same buffers.
inline std::atomic<int> s_grp_matmul_algo5_vertical_fusion_override {-1};

} // namespace test_api

inline std::atomic<int> &test_api_decode_algo5_gate_override() {
    return test_api::s_grp_matmul_decode_algo5_gate_override;
}

inline std::atomic<int> &test_api_prompt_algo5_gate_override() {
    return test_api::s_grp_matmul_prompt_algo5_gate_override;
}

inline std::atomic<int> &test_api_algo5_vertical_fusion_override() {
    return test_api::s_grp_matmul_algo5_vertical_fusion_override;
}

// ZENDNNL_GRP_MATMUL_ALGO5_VERTICAL_FUSION = { 0, 1 } — cached, default 1 (ON).
//   Enables the fused per-expert MoE FFN pipeline: one OMP region running
//   W13 -> gated act -> W2 per expert, instead of two `parallel_per_expert`
//   passes separated by a fork/join and a second dispatcher prologue.  See
//   `try_expert_parallel_pipeline()` in `group_matmul_expert_parallel.hpp`.
//
//   Boolean, unlike the tri-state ALGO 2 sibling
//   (`ZENDNNL_GRP_MATMUL_M_TILE_VERTICAL_FUSION`): that one needs a FORCED
//   mode to override a planner that may prefer two-pass on a tight scratch
//   budget.  This executor has no planner and allocates no scratch, so AUTO
//   and FORCED would coincide.
//
//   Defaults ON where the sibling defaults OFF because it means "fuse where
//   the regime is implemented", not "fuse this call": the executor serves
//   float end-to-end and DA8W8 and declines everything else back to two-pass.
//   Set `=0` to force two-pass for every call (comparison baseline and
//   escape hatch).  Non-numeric input → default 1.
inline bool get_grp_matmul_algo5_vertical_fusion() {
    const int ovr = test_api_algo5_vertical_fusion_override().load(
            std::memory_order_relaxed);
    if (ovr >= 0) return ovr != 0;
    static const bool v = []() {
        const char *e = std::getenv("ZENDNNL_GRP_MATMUL_ALGO5_VERTICAL_FUSION");
        int parsed = 0;
        if (!parse_env_int_strict(e, parsed)) return true; // default / junk: On
        return parsed != 0;
    }();
    return v;
}

/// True when a resolved-ALGO-5 fused-MoE call should attempt the fused
/// pipeline.  Kept here so the shared fused-MoE entry only has to ask, rather
/// than knowing which knob gates which executor.
inline bool expert_parallel_fusion_eligible(int resolved_algo) {
    return resolved_algo == kGrpMatmulAlgoExpertParallel
            && get_grp_matmul_algo5_vertical_fusion();
}

/// Instrumentation tag for a fused ALGO 5 pass.  Deliberately distinct from
/// the ALGO 2 tags: both fuse the same three stages but parallelise
/// differently (per-expert vs per-M-slice), so a shared tag would merge two
/// routes into one timing bucket.
/// `wei_probe` must come from an ACTIVE expert, not from slot 0: an inactive
/// slot is a padded placeholder whose dtypes were never classified, so slot 0
/// would mislabel a DA8W8 layer as float on the usual MoE decode shape.
inline const char *expert_parallel_fusion_mode_tag(data_type_t wei_probe) {
    return (wei_probe == data_type_t::s8)
            ? "vertical_fusion_expert_parallel_da8w8"
            : "vertical_fusion_expert_parallel_float";
}

// ZENDNNL_GRP_MATMUL_DECODE_ALGO5_GATE = { 0, 1 } — cached, default 1 (ON).
//   QUALIFIER kill-switch for the `AUTO_DECODE_ALGO=5` pin (see
//   `decide_algo5_pin()` below).  AUTO never emits ALGO 5 on its own; an
//   operator opts in with the pin, and this gate honours it only where the
//   measured win holds — every ACTIVE expert on s8 weights AND
//   `active_ops >= num_threads`.  Model-agnostic: it fences on dtype and
//   occupancy, not on a shape fingerprint.  Any other decode frame declines
//   and falls back exactly as if the pin were unset.  Set `=0` to honour the
//   pin verbatim (pre-gate semantics).  Non-numeric input → default 1.
inline bool get_grp_matmul_decode_algo5_gate() {
    const int ovr = test_api_decode_algo5_gate_override().load(
            std::memory_order_relaxed);
    if (ovr >= 0) return ovr != 0;
    static const bool v = []() {
        const char *e = std::getenv("ZENDNNL_GRP_MATMUL_DECODE_ALGO5_GATE");
        int parsed = 0;
        if (!parse_env_int_strict(e, parsed)) return true; // default / junk: On
        return parsed != 0;
    }();
    return v;
}

// ZENDNNL_GRP_MATMUL_PROMPT_ALGO5_GATE = { 0, 1 } — cached, default 1 (ON).
//   Prompt twin of the decode gate, for `AUTO_PROMPT_ALGO=5`.  Carries the
//   SAME occupancy term, because ALGO 5's inability to fill an under-occupied
//   team is structural rather than phase-specific.  A declined prompt pin
//   falls back to `kGrpMatmulAlgo5PromptDeclineAlgo`, whose N-split does fill
//   the team.
//
//   Deliberately does NOT carry decode's all-active-s8 term: that encodes a
//   measured bf16 *decode* regression, and prompt's large per-expert M is a
//   different regime, so inheriting it would fence bf16 prompt on evidence
//   never collected for this phase.  Add it if such a regression is measured.
inline bool get_grp_matmul_prompt_algo5_gate() {
    const int ovr = test_api_prompt_algo5_gate_override().load(
            std::memory_order_relaxed);
    if (ovr >= 0) return ovr != 0;
    static const bool v = []() {
        const char *e = std::getenv("ZENDNNL_GRP_MATMUL_PROMPT_ALGO5_GATE");
        int parsed = 0;
        if (!parse_env_int_strict(e, parsed)) return true; // default / junk: On
        return parsed != 0;
    }();
    return v;
}

/// Outcome of the ALGO 5 pin qualifier for one selection.
///
/// The two `*_declined` flags are what the selector acts on; the two term
/// flags below them say WHICH test failed and exist so the decline warning
/// can name the cause instead of just reporting the fact.
struct algo5_pin_verdict {
    /// An `AUTO_DECODE_ALGO=5` / `AUTO_PROMPT_ALGO=5` pin was present for
    /// this call's phase.  Set regardless of whether it survived the gate.
    bool decode_pin_is_5 = false;
    bool prompt_pin_is_5 = false;

    /// The pin failed at least one qualifying term and must be treated as
    /// unset (decode) or redirected to `kGrpMatmulAlgo5PromptDeclineAlgo`
    /// (prompt).
    bool decode_declined = false;
    bool prompt_declined = false;

    /// Failing terms.  `wei_not_s8` is decode-only; `low_occupancy` is
    /// shared, so it may describe either decline.
    bool wei_not_s8 = false;
    bool low_occupancy = false;
};

/// Apply the ALGO 5 pin qualifier — Rule 0.6a and its prompt twin.
///
/// ALGO 5 beats the ALGO 3 decode default on INT8 MoE decode at full-team
/// occupancy, but regresses on bf16 MoE and on under-occupied frames, where
/// one-expert-per-thread runs at the speed of the heaviest expert and cannot
/// fill the team.  So AUTO never emits ALGO 5 on
/// its own; an operator opts in with an explicit pin and this qualifier
/// honours it only where the win holds.
///
/// DECODE terms — ALL of: decode phase; every ACTIVE expert on s8 weights;
/// and `active_ops >= num_threads`.
///
/// PROMPT terms — the occupancy term only; see the prompt gate above for why
/// it does not inherit the s8 one.
///
/// Not fenced to a model or shape: the pin is a per-deployment opt-in, so any
/// INT8 saturated decode is honoured on the same terms.  ALGO 5 has no tiling
/// precondition, so an honoured pin needs no m_tile_safe / n_tile_safe clamp.
///
/// A decline is SURFACED (trace + a once-per-process WARN naming the failing
/// term).  `active_ops` must be the ACTIVE count `|{ i : M[i] > 0 }|`, not
/// `M.size()`.
///
/// Defined in the .cpp rather than inline: the decline warnings call
/// `apilog_warning`, which no header here pulls into scope.
algo5_pin_verdict decide_algo5_pin(
        const grp_matmul_auto_phase_setting &phase_setting, bool is_decode,
        const std::vector<int> &M, const std::vector<matmul_params> &params,
        int active_ops, int num_threads, auto_algo_trace *trace);

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_EXPERT_PARALLEL_POLICY_HPP
