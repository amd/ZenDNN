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

/// AUTO algorithm selection: `auto_select_algo` and
/// `select_grp_matmul_algo`.

#include "lowoha_operators/matmul/group_matmul/selection/algo_select.hpp"

#include <algorithm>
#include <vector>

#include "lowoha_operators/matmul/group_matmul/common/group_matmul_parallel_common.hpp"
#include "lowoha_operators/matmul/group_matmul/executors/m_tile/group_matmul_m_tile.hpp"
#include "lowoha_operators/matmul/group_matmul/executors/n_tile/group_matmul_n_tile_planner.hpp"
#include "lowoha_operators/matmul/group_matmul/executors/n_tile/n_tile_admission.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/expert_parallel/group_matmul_expert_parallel_policy.hpp"
#include "lowoha_operators/matmul/group_matmul/single_expert/single_expert_policy.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

using detail::check_n_tile_extra;

namespace {

// Auto-select (ALGO 0) heuristic — used when the caller leaves
// ZENDNNL_GRP_MATMUL_ALGO unset.  The out-of-the-box routing is:
//   * BF16 or FP16 layer of at least two experts
//     (`grp_matmul_auto_algo3_layer`) that ALGO 3 can serve (`n_tile_safe`) —
//     ALGO 3 in both phases while the phase is not pinned: the custom kernel
//     serves prompt and decode from one pack.
//   * Single expert (Rule 0.46) — ALGO 3 for each phase whose
//     `ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_*` knob lists its dtype.  Every other
//     call takes the rules below.
//   * Prompt — Rule 0.7 selects ALGO 1 (sequential full-team) unless
//     AUTO_PROMPT_ALGO is explicitly set. Setting AUTO_PROMPT_ALGO=0 falls
//     back to the legacy 3-rule cascade.
//   * Decode — Rule 0.5 / 0.45 / 0.6 apply shape and phase heuristics,
//     defaulting to ALGO 3 (N-tile + CK).  Setting AUTO_DECODE_ALGO=0
//     restores the legacy 3-rule cascade.
//
// AUTO's built-in heuristics never emit ALGO 5 or ALGO 6. Both generic
// schedulers remain reachable through an explicit global or phase pin.
// The one refinement is Rule 0.6a, which does NOT invent an ALGO-5 pick — it
// QUALIFIES an operator's decode `AUTO_DECODE_ALGO=5` pin, honouring it only for
// INT8 (s8) saturated-team decode and declining it (→ decode default) otherwise;
// with no pin set the invariant is exact.
// value 4 requests the earlier W8A8 whole-call interceptor; after an
// eligibility decline it is deliberately NOT a generic pin and this selector
// runs the inherited phase-default policy.
//
// Decision precedence (tightest first):
//
//   BF16/FP16 MoE — an `n_tile_safe` call of a
//      `grp_matmul_auto_algo3_layer` whose phase is not pinned → 3 (label:
//      auto_algo3).  Such a layer skips the single-thread exit and Rule 0,
//      because ALGO 3 slices a call past `kNTilePlanMaxExperts`; a pinned
//      phase follows Rules 1 and 2, and the legacy cascade still applies
//      both exits.
//
//   0. STRUCTURAL — num_ops > kNTilePlanMaxExperts (=256) → ALGO 1
//      Capacity carve-out: beyond `GroupNTilePlan::kMaxExperts` the
//      N-tile planner's R3 gate falls back to its Sequential strategy,
//      so ALGO 3 would be no better than ALGO 1.  This is the one site
//      where the no-5-no-6 invariant costs throughput — ALGO 5's
//      per-expert wave schedule was the only PARALLEL option past that
//      ceiling; `ZENDNNL_GRP_MATMUL_ALGO=5` recovers it.  Phase env
//      cannot override this — the R3 gate is structural.
//
//   0.45. DECODE only, single dense-FFN shape → ALGO 3 (N-tile).
//      Fires when num_ops==1 and the single expert's M qualifies as
//      "dense decode" (is_dense_ffn_decode).  ALGO 3's adaptive tiling
//      and K-blocking are tuned for this shape; ALGO 1 would treat it
//      as a plain sequential GEMM.  Yields to an explicit decode pin.
//      (label: auto_rule045_single_dense_decode)
//
//   0.5. DECODE only, few experts (total_experts ≤ 8) → occupancy arrow.
//      A pure thread-occupancy test, dtype-agnostic, two outcomes:
//        active_ops * 4 >= num_threads → ALGO 3 (N-tile; clamps to ALGO 1
//                                        when !n_tile_safe)
//        otherwise                     → ALGO 1 (full-team sequential)
//      Never ALGO 2 — that is opt-in only (see the no-auto-2 invariant).
//      Mixtral-class shapes (8 experts, topk=2 → 2 active) take the ALGO 1
//      arm on any real host.  Models with >8 experts never reach this rule
//      and inherit the ALGO 3 decode default from Rule 1.
//      Prompt is handled by Rule 0.7; this rule never fires for prompt.
//      Yields to an explicit decode pin.
//      (labels: auto_rule05_ntile_occupancy, auto_rule05_few_active)
//
//   0.6. DECODE, active_ops > num_threads → ALGO 3 (N-tile).
//      `active_ops` counts the experts that actually fire (`M[i] > 0`),
//      not the padded slot count.  ALGO 3's single round is infeasible
//      here (it needs one thread per active expert) but DecodeDynamic
//      has no such ceiling, so the regime stays on ALGO 3 and the
//      planner picks the strategy.  Yields to an explicit
//      `AUTO_DECODE_ALGO` pin; prompt is never routed by this rule.
//
//   0.7. PROMPT, AUTO_PROMPT_ALGO not explicitly set → ALGO 1.
//      All dtype families, including W4A8, use the sequential full-team
//      default. ALGO 2 is reachable only through an explicit global or
//      prompt-phase pin.
//      (label: auto_rule07_prompt_seq)
//
//   1. PHASE ENV — `max_M ≤ kDecodeMaxM` (decode) →
//                  `ZENDNNL_GRP_MATMUL_AUTO_DECODE_ALGO` (default 3)
//                  `max_M >  kDecodeMaxM` (prompt) →
//                  `ZENDNNL_GRP_MATMUL_AUTO_PROMPT_ALGO` (explicitly set; see
//                   Rule 0.7 for the unset case)
//      A generic non-zero value is returned directly with the same
//      m_tile_safe / n_tile_safe clamps the global ALGO env path applies in
//      `select_grp_matmul_algo`. Value 4 has already requested W8A8 and, on
//      decline, behaves as an unset setting here so the inherited default
//      refinements still run. When the prompt setting does not pin a generic
//      policy (unset or a declined `4`), Rule 0.7 sends the prompt to ALGO 1.
//      This block handles explicit generic/legacy prompt settings.
//      For decode, the default is ALGO 3 (N-tile rounds + CK).  Set
//      `AUTO_PROMPT_ALGO=0` for the legacy 3-rule cascade.
//
//   2. LEGACY RULES (phase env == 0):
//
//      a. num_ops ≥ num_threads               → ALGO 3
//         (many experts: at this expert/thread ratio every expert
//          sees a thin per-expert team and N-tile's round-based
//          scheduling fits better than ALGO 1's serial-experts-with-
//          full-team approach.  Honors n_tile_safe — quantised paths
//          fall back to ALGO 1.)
//
//      b. num_ops ≤ kFewExpertsAlgo1 (=8)     → ALGO 1
//         (few experts: the per-expert weight footprint is large
//          enough that the full-weight AOCL DLP cache key + serial
//          expert iteration amortises DRAM traffic better than
//          N-tile's per-thread column slices on a thin per-expert
//          team.)
//
//      c. otherwise (9 ≤ num_ops < num_threads) — M-driven:
//           prompt (max_M >  kDecodeMaxM)     → ALGO 1
//           decode (max_M ≤  kDecodeMaxM)     → ALGO 3
//         (moderate experts: prompt uses ALGO 1's thread-count-stable
//          full-weight cache key; decode uses ALGO 3's custom-kernel
//          + per-tile path.  N-tile's internal Sequential-strategy
//          fallback handles narrow-N shapes where the planner can't
//          satisfy `tiles_per_expert ≥ min`.)
//
// The historical large-weight wide-N prompt carve-out and weight-class
// branching are intentionally dropped — the simpler M-driven default
// keeps the same routing with explicit expert-count arrows, and the
// auto-selector now reads as a 3-rule table.
// Callers that need a non-default decision on a specific deployment
// can still pin via `ZENDNNL_GRP_MATMUL_ALGO` (global pin) or via
// `ZENDNNL_GRP_MATMUL_AUTO_{PROMPT,DECODE}_ALGO` (per-phase pin while
// keeping the global env unset / 0).

static int auto_select_algo(const std::vector<int> &M,
        const std::vector<int> &N, const std::vector<int> &K,
        const std::vector<matmul_params> &params, int num_threads,
        bool m_tile_safe, bool n_tile_safe, auto_algo_trace *trace,
        grp_matmul_gated_act_t fused_act, bool single_expert_epilogue_ok,
        const single_expert_ck_t *precomputed_single_ck) {
    // Every return below goes through this, so the reported rule cannot
    // disagree with the rule that ran.  `want` is the rule's own answer;
    // `algo` is that answer after the safety clamps, and the two differing is
    // exactly what the log's `_clamp` suffix means.
    const auto pick = [&](int algo, const char *reason, int want) {
        if (trace != nullptr) {
            trace->reason = reason;
            trace->unclamped = want;
        }
        return algo;
    };

    const int num_ops = static_cast<int>(M.size());
    const bool algo3_layer
            = n_tile_safe && grp_matmul_auto_algo3_call(M, params);
    // Rule 0.46's layer keeps ALGO 3 at any team size, like an MoE layer: its
    // custom-kernel pack may already own the weight buffer.
    const single_expert_ck_t single_ck
            = !(n_tile_safe && single_expert_epilogue_ok && num_ops == 1)
            ? single_expert_ck_t {}
            : precomputed_single_ck != nullptr
            ? *precomputed_single_ck
            : grp_matmul_single_expert_call_ck(
                      M, N, K, params, fused_act, num_threads);
    // A single expert whose decode runs on the custom kernel keeps ALGO 3 at
    // one thread too.
    const bool dense_algo3_layer = single_ck.decode;
    if (num_ops == 0
            || (num_threads <= 1 && !algo3_layer && !dense_algo3_layer)) {
        return pick(1, "auto_single_thread", 1);
    }

    // INVARIANT — AUTO never returns ALGO 2, 5 or 6 of its own accord.
    //
    //   * no-auto-2: ALGO 2 requires an explicit
    //     `ZENDNNL_GRP_MATMUL_ALGO=2` global pin or
    //     `AUTO_{DECODE,PROMPT}_ALGO=2` phase pin.
    //   * no-5-no-6: every rule that once answered 5 answers
    //     `n_tile_safe ? 3 : 1`.
    //
    // Rule 0.6a below does NOT break this: it never SELECTS 5, it only
    // QUALIFIES an operator's
    // explicit decode `AUTO_DECODE_ALGO=5` pin (honour for INT8 saturated-team
    // decode, decline → decode default otherwise). Keep new rules inside the
    // no-auto-2 / no-5-no-6 set. The invariant constrains the HEURISTICS, not the operator:
    // an explicit phase pin or global force of 5/6 is honoured.
    //
    // Rule 0 — STRUCTURAL capacity carve-out, placed before the phase env so
    // it catches every shape that would otherwise reach the N-tile planner's
    // R3 Sequential fallback.  See the rule table above for why it is ALGO 1.
    if (num_ops > kNTilePlanMaxExperts && !algo3_layer) {
        return pick(1, "auto_rule0_capacity", 1);
    }

    const int max_M = max_active_grp_matmul_m(M, M.size());
    const grp_matmul_phase phase = classify_grp_matmul_phase(max_M);
    const bool is_decode = phase == grp_matmul_phase::decode;
    // One snapshot carries raw request, generic effective value, and pin
    // status. In particular, a request 4 fallback maps to the default
    // effective value without suppressing any default-policy refinement.
    const grp_matmul_auto_phase_setting phase_setting
            = get_grp_matmul_auto_phase_setting(phase);

    // ACTIVE-COMPUTE expert count = |{ i : M[i] > 0 }|, NOT M.size() and NOT the
    // framework `total_matmul` pool: a legacy caller may pass a padded vector
    // with `M[i]==0` placeholders, and only the M[i]>0 experts consume a thread.
    // Read by the decode `=5` pin's occupancy term (Rule 0.6a) and by Rule 0.6.
    const int active_ops = static_cast<int>(
            std::count_if(M.begin(), M.end(), [](int m) { return m > 0; }));

    // TOTAL expert count scopes Rule 0.5 to few-expert layers. `total_matmul`
    // is only meaningful under the framework opt-in (`active_matmul > 0`); a
    // legacy caller may leave it stale, so read it only then and only when it
    // exceeds the active count (padded layout).
    int total_experts = num_ops;
    if (!params.empty() && params[0].active_matmul > 0
            && params[0].total_matmul > static_cast<uint32_t>(total_experts)) {
        total_experts = static_cast<int>(params[0].total_matmul);
    }

    // Rule 0.6a — QUALIFY an `AUTO_{DECODE,PROMPT}_ALGO=5` pin.  The qualifier
    // itself lives with the ALGO it governs, in
    // `multi_expert/expert_parallel/group_matmul_expert_parallel_policy.hpp`; what stays
    // here is only how its verdict feeds the rule table below.  A declined
    // DECODE pin behaves exactly as if `AUTO_DECODE_ALGO` were unset; a
    // declined PROMPT pin redirects to `kGrpMatmulAlgo5PromptDeclineAlgo`.
    const algo5_pin_verdict algo5_pin = decide_algo5_pin(phase_setting,
            is_decode, M, params, active_ops, num_threads, trace);
    const bool decode_algo5_pin_declined = algo5_pin.decode_declined;
    const bool prompt_algo5_pin_declined = algo5_pin.prompt_declined;

    // A gate-declined DECODE pin is treated as UNSET, re-enabling the decode
    // rules (0.45 / 0.5 / 0.6) below.  A declined PROMPT pin is deliberately
    // NOT folded in: the redirect to ALGO 3 is decided once in `phase_algo`
    // below, so a future prompt rule gating on `!phase_env_pinned` cannot
    // silently preempt it.
    const bool phase_env_pinned
            = !decode_algo5_pin_declined && phase_setting.pins_generic_policy();

    if (algo3_layer && !phase_env_pinned) { return pick(3, "auto_algo3", 3); }

    // Rule 0.46 — SINGLE DENSE EXPERT → ALGO 3 (custom kernel), per phase,
    // then Rule 0.45 — SINGLE DENSE EXPERT DECODE → ALGO 3 (N-tile).  Both
    // live with the single-expert policy, in
    // `grp_matmul_single_expert_auto_rule`.
    if (const auto se = num_ops == 1
                    ? grp_matmul_single_expert_auto_rule(single_ck, is_decode,
                              phase_env_pinned, n_tile_safe, num_ops, max_M)
                    : single_expert_rule_pick {};
            se.hit) {
        return pick(se.algo, se.reason, se.want);
    }

    // Rule 0.5 — DECODE-ONLY few-expert occupancy arrow. Prompt
    // few-expert calls are handled by Rule 0.7 below. There are only two
    // outcomes: ALGO 3 or ALGO 1; the rule is deliberately independent of
    // dtype and the old M-tile regime classification.
    //
    // ALGO 3 splits each expert's N across the team, so it earns its
    // round-based schedule only when there are enough ACTIVE experts to keep
    // that team busy — at most `kDecodeNTileThreadFactor` threads per active
    // expert.  Below that the team is far wider than the expert count and the
    // full-team sequential ALGO 1 wins, because each expert then gets the
    // whole team for its own N-split with no round overhead.  The
    // Mixtral-class shape lands here: 8 experts with topk=2 means 2 active,
    // so `2*4 = 8 < 32` on any real host → ALGO 1.  Models with more than
    // `kFewExpertsDecodeThreshold` experts never reach this rule and take
    // the ALGO 3 decode default from Rule 1.
    //
    // Deliberately DTYPE-AGNOSTIC: the same arrow applies to bf16, f16, f32
    // and every INT8 / WOQ variant.  The only dtype-dependent part is the
    // `n_tile_safe` clamp, which is a legality gate (a shape ALGO 3 cannot
    // column-slice must not be handed to it), not a performance heuristic.
    if (is_decode && total_experts <= kFewExpertsDecodeThreshold
            && !phase_env_pinned) {
        if (active_ops * kDecodeNTileThreadFactor >= num_threads) {
            return pick(n_tile_safe ? 3 : 1, "auto_rule05_ntile_occupancy", 3);
        }
        return pick(1, "auto_rule05_few_active", 1);
    }

    // Rule 0.6 — DECODE with MORE ACTIVE EXPERTS THAN THREADS.
    // `active_ops` (hoisted above) is the ACTIVE-COMPUTE expert count that
    // drives ALGO 3's per-expert thread budget — NOT `M.size()` and NOT the
    // framework `total_matmul` pool.
    //
    // DECODE ONLY — prompt is compute-bound on large M and follows its own
    // policy.  YIELDS TO AN EXPLICIT PIN: this is a POLICY rule (which algo
    // benchmarks best), not a legality rule, so `AUTO_DECODE_ALGO` outranks
    // it, matching sibling rules 0.45 and 0.5.  Only the correctness gates
    // (R0 capacity, tile-safety clamps) still override a pin.
    if (is_decode && !phase_env_pinned) {
        if (active_ops > num_threads) {
            // ALGO 3, so the CCD-cohesive DecodeDynamic pool can engage in
            // `plan_group_n_tile` (its `active_ops >= 4*num_ccds` gate is
            // trivially met here).  Unlike the Rounds schedule, DecodeDynamic
            // has no `active_ops <= num_threads` ceiling — it maps whole
            // experts onto CCDs rather than thread-id onto expert.
            // `N_TILE_STRATEGY` does not participate in the ALGO decision; it
            // only selects WHICH ALGO 3 strategy runs.  The legacy per-expert
            // schedule is still reachable via `ZENDNNL_GRP_MATMUL_ALGO=5`.
            return pick(n_tile_safe ? 3 : 1, "auto_decode_ops_gt_threads", 3);
        }
    }

    // Rule 0.7 — PROMPT. Unpinned prompt calls use the same sequential
    // full-team default for every dtype. ALGO 2 remains explicitly selectable
    // through the global or prompt-phase knobs.
    if (!is_decode && !grp_matmul_auto_prompt_algo_is_set()) {
        return pick(1, "auto_rule07_prompt_seq", 1);
    }

    // Rule 1 — PHASE SETTING. Shared phase classification drives which
    // setting is consulted. A generic non-zero request pins that algo for
    // the phase; request 4 reaches here only after a BF16 W8A8 hook decline
    // and exposes the inherited default as its generic effective value, with
    // `phase_env_pinned=false` so Rules 0.45/0.5/0.6/0.7 above retain their
    // normal authority. Apply the same m_tile_safe / n_tile_safe correctness
    // clamps as the global ALGO env path. Non-tile-safe + ALGO 3 falls to
    // ALGO 1; non-m-tile-safe + ALGO 2 falls to ALGO 1. These clamps emit no
    // WARN: the
    // `[GRP_MATMUL.ALGO WARN]` line belongs to `select_grp_matmul_algo`'s
    // global-env branch, which this path does not reach — reaching here
    // means the global ALGO was AUTO.  Operators see the clamp on the
    // `[GRP_MATMUL.ALGO]` line as `chosen=ALGO_X reason=auto_phase_env_clamp`.
    //
    // A gate-declined decode `=5` pin (Rule 0.6a) resolves to the DECODE DEFAULT
    // here, so the call lands exactly where an UNSET `AUTO_DECODE_ALGO` would.
    // Clearing `phase_env_pinned` above re-enabled the decode policy rules
    // (0.45 / 0.5 / 0.6); this is the last step that would otherwise still
    // read the pinned `5` and hand it back.  Substituting the default — rather
    // than falling through to the Rule 2 legacy cascade — is what makes
    // "declined" mean "as if unset" instead of a third, otherwise-unreachable
    // policy.
    // A declined PROMPT pin substitutes ALGO 3, not the prompt default.  The
    // `phase_algo == 3 && !n_tile_safe` clamp below still applies, so a shape
    // N-tile cannot serve falls to ALGO 1 rather than running an unsafe plan.
    const int phase_algo = decode_algo5_pin_declined
            ? kGrpMatmulAutoDecodeAlgoDefault
            : (prompt_algo5_pin_declined
                              ? kGrpMatmulAlgo5PromptDeclineAlgo
                              : phase_setting.generic_effective_algo);
    if (is_grp_matmul_generic_algo(phase_algo)) {
        if (phase_algo == 2 && !m_tile_safe) {
            return pick(1, "auto_phase_env", phase_algo);
        }
        if (phase_algo == 3 && !n_tile_safe) {
            return pick(1, "auto_phase_env", phase_algo);
        }
        // ALGO 5 and 6 are honoured here.  The no-5-no-6 invariant governs
        // what auto-select picks ON ITS OWN, not what an operator may ask
        // for: a phase pin is an explicit request, and silently rewriting it
        // to 3 would make `AUTO_{PROMPT,DECODE}_ALGO={5,6}`
        // look supported while doing something else, with a warning on every
        // single call.  Neither algo has a tiling precondition, so no clamp
        // applies.
        return pick(phase_algo, "auto_phase_env", phase_algo);
    }

    // Rule 2 — LEGACY RULES (phase env == 0).
    //
    // A BF16/FP16 MoE layer skipped the single-thread and capacity exits above;
    // the cascade applies them, as the callers that opt into it expect.
    if (algo3_layer && num_threads <= 1) {
        return pick(1, "auto_single_thread", 1);
    }
    if (algo3_layer && num_ops > kNTilePlanMaxExperts) {
        return pick(1, "auto_rule0_capacity", 1);
    }

    // 2a. num_ops ≥ num_threads (many experts).  Highest of the three
    //     legacy rules so an 8-expert deployment on a ≤ 8-thread host
    //     (rare but possible for local dev / single-CCD profiling)
    //     routes here, not to rule 2b.
    //
    // SCOPE NOTE — N-tile viability NOT consulted by design.
    //   The previous heuristic gated rule-1-like cases on
    //   `tiles_per_expert ≥ min_ntiles`.  The new rule deliberately
    //   skips that check: the N-tile planner's `ntile_viable` runs
    //   anyway as part of `plan_group_n_tile`.  Since the
    //   `N_TILE_STRATEGY=2` (rounds) fix to the planner,
    //   `!viable` no longer demotes to Sequential under force_ntile —
    //   it stays on rounds with a `[GRP_MATMUL.PLAN.HINT]` line.
    //   Under `n_tile_strategy=0` (auto) the planner still uses
    //   viability as a perf hint.
    if (num_ops >= num_threads) {
        return pick(n_tile_safe ? 3 : 1, "auto_rule2a_ops_ge_threads", 3);
    }

    // 2b. num_ops ≤ kFewExpertsAlgo1 (few experts).
    if (num_ops <= kFewExpertsAlgo1) {
        return pick(1, "auto_rule2b_few_experts", 1);
    }

    // 2c. M-driven default (prompt → ALGO 1, decode → ALGO 3).
    // The decode arrow does NOT consult N-tile viability for the same
    // reason rule 2a doesn't — see the SCOPE NOTE on rule 2a above.
    if (!is_decode) { return pick(1, "auto_rule2c_prompt", 1); }
    return pick(n_tile_safe ? 3 : 1, "auto_rule2c_decode", 3);
}

} // namespace

// ── ALGO selection ──────────────────────────────────────────────────────
//
// Returns a canonical generic ALGO number ({1,2,3,5,6}).  Driven by:
//   * `check_m_tile_safe` / `check_n_tile_extra` — helper checks that
//     determine whether the M-tile slicer is safe to use and whether
//     the extra constraints required by the N-tile path are satisfied
//     without corrupting packed-B / post-op buffers.
//   * `auto_select_algo` — cost-model-free heuristic used when the
//     caller leaves ZENDNNL_GRP_MATMUL_ALGO unset (== 0).

int select_grp_matmul_algo(const std::vector<char> &layout,
        const std::vector<int> &M, const std::vector<int> &N,
        const std::vector<int> &K, const std::vector<matmul_params> &params,
        int num_threads, auto_algo_trace *trace,
        grp_matmul_gated_act_t fused_act, bool single_expert_epilogue_ok,
        const single_expert_ck_t *single_ck) {

    // `M.size()` is the active matmul count after `group_matmul_direct`
    // sliced the M vector to honour `params[0].active_matmul`.  Pass it
    // explicitly so the safety helpers iterate only the active slots
    // rather than `params.size()` (which still carries the framework's
    // prepack-extras tail).
    const int num_ops_eff = static_cast<int>(M.size());
    const bool m_tile_safe = check_m_tile_safe(layout, M, params, num_ops_eff);
    // ALGO 3 allows CK-VNNI prepacked B; W4A8 native prepack stays on full-N ALGOs.
    const bool n_tile_safe = check_m_tile_safe(layout, M, params, num_ops_eff,
                                     /*allow_prepacked_b=*/true)
            && check_n_tile_extra(M, N, params, num_ops_eff, fused_act);

    // Manual generic override: ZENDNNL_GRP_MATMUL_ALGO={1,2,3,5,6}.
    //   ALGO 2 (M-tile): needs m_tile_safe (row-major, uniform dtypes).
    //   ALGO 3 (N-tile): needs n_tile_safe (+ unpacked B and slice-safe
    //                      metadata, or an explicit full-N fallback).
    //   ALGO 1/5/6:      no tiling → no safety guard needed (BLAS handles all).
    // Unsafe env overrides fall back to ALGO 1 rather than failing, so
    // callers that force-deploy a given ALGO never hit a hard error on
    // shape edge cases.
    // PRECEDENCE — the per-phase knobs outrank the global selector.
    //
    // `ZENDNNL_GRP_MATMUL_AUTO_{DECODE,PROMPT}_ALGO`, when explicitly set for
    // THIS call's phase, wins over `ZENDNNL_GRP_MATMUL_ALGO`.  An operator
    // tuning one phase should not have to clear the global knob first, and
    // the phase knob is the more specific statement of intent.
    //
    // It is applied by SUPPRESSING the global branch below rather than by
    // returning here, so the phase value still flows through
    // `auto_select_algo`'s Rule 1.  That keeps three things in one place:
    // the m_tile / n_tile legality clamps, the ALGO 5 pin qualifier
    // (Rule 0.6a — which must still be able to decline a `=5` phase pin on
    // bf16 or under-occupied decode), and the `=0` legacy-cascade meaning.
    //
    // `has_explicit_request()` (not `pins_generic_policy()`) is the test, so
    // a phase `=4` W8A8 request also outranks the global; if that attempt
    // declines it inherits the phase default policy rather than silently
    // falling back to the global pin.
    const grp_matmul_phase call_phase = classify_grp_matmul_phase(M, M.size());
    const bool phase_knob_set = get_grp_matmul_auto_phase_setting(call_phase)
                                        .has_explicit_request();

    // Inspect the raw requested selector so numeric 4 is unmistakably outside
    // this generic branch. Global 4 has already attempted W8A8 and falls
    // through to AUTO below.
    const int requested_algo = get_grp_matmul_requested_algo();
    if (!phase_knob_set && is_grp_matmul_generic_algo(requested_algo)) {
        int algo = requested_algo;
        // Silent-override → apilog_warning so a user debugging
        // `ZENDNNL_GRP_MATMUL_ALGO=3 but actually ran ALGO 1` sees the
        // reason in the library log.  Gated by apilog_warning_enabled()
        // (cached) so the warning fires whenever the API log level is
        // ≥ warning — the framework already filters by level, but the
        // cached bool lets us skip the message-construction overhead
        // when warnings are suppressed without a per-call level query.
        if (algo == 2 && !m_tile_safe) {
            static const bool s_log = apilog_warning_enabled();
            if (s_log) {
                apilog_warning(
                        "[GRP_MATMUL.ALGO WARN] env_algo=2 (flat_m_tile) "
                        "REJECTED: m_tile unsafe (non-row-major, per-expert "
                        "dtype "
                        "mismatch, packed B, softmax/pooling post-op, or "
                        "dynamic-quant with non-row-local src granularity — "
                        "src_scale.dims[0] must equal the per-expert M[i], "
                        "including the M[i]=1 decode case `{1, 1}`). "
                        "FALLBACK algo=1 (sequential_experts).");
            }
            algo = 1;
        }
        if (algo == 3 && !n_tile_safe) {
            static const bool s_log = apilog_warning_enabled();
            if (s_log) {
                apilog_warning(
                        "[GRP_MATMUL.ALGO WARN] env_algo=3 "
                        "(flat_n_tile) "
                        "REJECTED: n_tile unsafe.  Common rejection reasons: "
                        "non-row-major layout, per-expert dtype mismatch, "
                        "buffer post-op, or a quant configuration outside the "
                        "supported N-tile scope.  Accepted quant shapes "
                        "include: "
                        "(a) DQ-INT8: `dynamic_quant=true` with `{M[i], 1}` "
                        "src "
                        "(incl. `{1, 1}` when M[i]=1) + per-channel `{1, N}` "
                        "wei; "
                        "(b) W4A8: `wei=s4`, `dynamic_quant=true`, "
                        "`compute=s8`, "
                        "symmetric, per-group `{G, N}` wei + per-token `{M[i], "
                        "1}` "
                        "or per-group `{M[i], G}` src.  Static src, per-tensor "
                        "src/wei, pure WOQ, and unsigned u4 stay on ALGO 1.  "
                        "See "
                        "`check_n_tile_extra` SCOPE NOTE for the full table.  "
                        "FALLBACK algo=1 (sequential_experts).");
            }
            algo = 1;
        }
        return algo;
    }

    return auto_select_algo(M, N, K, params, num_threads, m_tile_safe,
            n_tile_safe, trace, fused_act, single_expert_epilogue_ok,
            single_ck);
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
