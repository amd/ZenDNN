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

/// Top-level dispatch entry-point for `group_matmul_direct`.
///
/// This translation unit owns the routing: the grouped-AUTO in-place
/// weight-cache policy and `group_matmul_run_parallel_dispatch` itself.
///
/// ALGO 2 (M-tile, `flat_m_tile` + `flat_m_tile_pipeline_bf16`) and
/// ALGO 3 (N-tile, `flat_n_tile`) live in their own folder-scoped
/// translation units (`executors/m_tile/group_matmul_m_tile.cpp`,
/// `executors/n_tile/group_matmul_n_tile.cpp`) and are called through forward
/// declarations re-included by `executors/m_tile/group_matmul_m_tile.hpp` and
/// `executors/n_tile/group_matmul_n_tile.hpp`.
///
/// Historical note: this file was named `group_matmul_parallel.cpp`
/// until the PR follow-up that renamed it to `group_matmul_dispatch.cpp`
/// (the prior name advertised "parallel" but the file's actual job is
/// dispatch + the small serial ALGOs).

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <omp.h>

#include "../common/group_matmul_parallel_common.hpp"
#include "../custom_kernel/dispatch.hpp"
#include "../executors/m_tile/group_matmul_m_tile.hpp" // flat_m_tile + M-tile env knobs
#include "../executors/n_tile/group_matmul_n_tile.hpp" // flat_n_tile + N-tile env knobs
#include "../executors/n_tile/prompt_tile/prompt_tile_tunables.hpp"
#include "../multi_expert/expert_parallel/group_matmul_expert_parallel.hpp" // parallel_per_expert
#include "../multi_expert/expert_parallel/group_matmul_expert_parallel_policy.hpp" // ALGO 5 pin gate
#include "../prepack/prepack.hpp"
#include "../single_expert/single_expert_policy.hpp"
#include "lowoha_operators/matmul/backends/aocl/aocl_kernel.hpp"
#include "lowoha_operators/matmul/group_matmul/executors/sequential/sequential_experts.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/multilevel/parallel_multilevel.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

using detail::parallel_multilevel;
using detail::sequential_experts;

// NOTE: ALGO 1 `sequential_experts` moved to
// `executors/sequential/sequential_experts.cpp`, ALGO 6
// `parallel_multilevel` to `multi_expert/multilevel/parallel_multilevel.cpp`,
// the ALGO 3 admission check `check_n_tile_extra` to
// `executors/n_tile/n_tile_admission.cpp`, and `auto_select_algo` /
// `select_grp_matmul_algo` to `selection/algo_select.cpp`.

// ALGO 5 (per-expert parallel) moved to
// `multi_expert/expert_parallel/group_matmul_expert_parallel.cpp`.

// M-tile (ALGO 2) safety predicate hoisted to
// `group_matmul_parallel_common.hpp` as `check_m_tile_safe` (inline).
// Both the legacy dispatcher in this TU and the MoE vertical-fusion
// dispatcher fork in `group_matmul_fused_moe.cpp` use the same
// predicate, so it lives next to `op2_k_for_act` in the common
// header.  See that header's doc-block for the row-locality
// rationale (dynamic-quant, postop softmax / pooling, etc.) and the
// `M[i] == 1` decode-class special case.

// ── Grouped-AUTO in-place cache policy ─────────────────────────────────

bool prepare_grouped_auto_weight_cache(std::vector<matmul_params> &params,
        bool custom_kernel_effective,
        std::vector<matmul_params> *secondary_params, bool auto_algo3_layer,
        bool *ck_inplace_licence) {
    auto &config = matmul_config_t::instance();
    if (ck_inplace_licence != nullptr) { *ck_inplace_licence = false; }
    if (config.get_weight_cache() != 2) { return false; }

    // A pinned ALGO has one weight layout. Mixed mode is AUTO-only; clear a
    // stale AUTO verdict so it cannot force a pinned CK pack out-of-place.
    if (get_grp_matmul_algo() != 0) {
        config.set_grp_auto_mixed_inplace(false);
        return false;
    }

    // A call whose matmul_params cap every weight below 2 never writes one in
    // place, so the AUTO policy below has nothing to protect for it: leave
    // the process mode and the mixed flag as the other callers set them.
    const auto at_wc2 = [](const std::vector<matmul_params> *pool) {
        return pool != nullptr
                && std::any_of(
                        pool->begin(), pool->end(), [](const matmul_params &p) {
            return p.weight_cache_type >= 2;
        });
    };
    if (!at_wc2(&params) && !at_wc2(secondary_params)) { return false; }

    bool has_caller_packed_weight = false;
    bool has_raw_weight = false;
    auto scan_formats = [&](const std::vector<matmul_params> &pool) {
        for (const auto &p : pool) {
            if (p.mem_format_b == 'r')
                has_caller_packed_weight = true;
            else
                has_raw_weight = true;
        }
    };
    scan_formats(params);
    if (secondary_params != nullptr) { scan_formats(*secondary_params); }

    // A per-call mixed raw/caller-packed pool cannot share the process-wide
    // in-place licence. Cap both fused-MoE halves together when present.
    if (has_caller_packed_weight && has_raw_weight) {
        auto cap_out_of_place = [](std::vector<matmul_params> &pool) {
            for (auto &p : pool) {
                p.weight_cache_type = std::min<int32_t>(p.weight_cache_type, 1);
            }
        };
        cap_out_of_place(params);
        if (secondary_params != nullptr) {
            cap_out_of_place(*secondary_params);
        }
        // Sticky by design: a concurrent caller may already hold a WC=2
        // snapshot, which must not become permission for an in-place CK pack.
        config.set_grp_auto_mixed_inplace(true);
        apilog_info(
                "[GRP_MATMUL.WEIGHT_CACHE] mixed raw/caller-packed pool: "
                "preserving process WC/mixed policy and capping this call at "
                "out-of-place weight caching");
        return false;
    }

#if ZENDNNL_DEPENDS_AOCLDLP
    const bool aocl_dlp_compiled = true;
#else
    const bool aocl_dlp_compiled = false;
#endif
    const bool process_mixed_eligible = aocl_dlp_compiled
            && get_grp_matmul_prepack() && get_grp_matmul_cross_warm()
            && custom_kernel_effective
            && config.get_lru_cache_capacity()
                    == std::numeric_limits<uint32_t>::max();
    if (process_mixed_eligible) {
        config.set_grp_auto_mixed_inplace(true);
        static std::atomic<bool> s_wc2_mixed_announced {false};
        if (!s_wc2_mixed_announced.exchange(true, std::memory_order_relaxed)) {
            apilog_info(
                    "[GRP_MATMUL.WEIGHT_CACHE] weight_cache_type=2 AUTO "
                    "mixed-in-place ENABLED "
                    "(prepack+cross_warm+custom_kernel on, unlimited LRU "
                    "capacity): AOCL full-weight prompt reorder mutates the "
                    "weight buffer in place; CK / AOCL per-tile decode stay "
                    "out-of-place, pre-warmed from raw weights before the "
                    "mutation.");
        }
        // Per call, a BF16/FP16 MoE layer AUTO keeps on ALGO 3
        // (`grp_matmul_auto_algo3_only`) has the custom-kernel pack as the
        // shared prompt/decode layout.  The dispatcher caps AOCL DLP reorders
        // out-of-place.  BF16 (even K) and FP16 may write their pack into the
        // caller's buffer.  A later non-CK reader of an in-place image fails closed
        // (`kGrpMatmulErrPackedInPlaceNoCk`).
        if (ck_inplace_licence != nullptr
                && grp_matmul_auto_algo3_only(auto_algo3_layer)
                && (at_wc2(&params) || at_wc2(secondary_params))) {
            *ck_inplace_licence = true;
            static std::atomic<bool> s_wc2_ck_announced {false};
            if (!s_wc2_ck_announced.exchange(true, std::memory_order_relaxed)) {
                apilog_info(
                        "[GRP_MATMUL.WEIGHT_CACHE] weight_cache_type=2 AUTO "
                        "custom-kernel layout ownership ENABLED for bf16/f16 "
                        "MoE layers and single experts whose both phases run "
                        "ALGO 3: "
                        "AOCL DLP layouts stay out-of-place; supported bf16 "
                        "and f16 packs may use the caller's buffer.");
            }
        }
        return true;
    }

    // Without a split-independent CK decode pack, AUTO cannot safely retain
    // raw bytes after the prompt's irreversible in-place AOCL reorder.
    config.downgrade_weight_cache(1);
    static std::atomic<bool> s_wc2_downgrade_warned {false};
    if (!s_wc2_downgrade_warned.exchange(true, std::memory_order_relaxed)) {
#if !ZENDNNL_DEPENDS_AOCLDLP
        apilog_warning(
                "[GRP_MATMUL.WEIGHT_CACHE] weight_cache_type=2 (in-place) "
                "requires AOCL-DLP (ZENDNNL_DEPENDS_AOCLDLP=OFF): "
                "mixed-in-place is unavailable. Downgrading process-wide to "
                "out-of-place (weight_cache_type=1) for the rest of the run; "
                "kernel selection unchanged.");
#else
        apilog_warning(
                "[GRP_MATMUL.WEIGHT_CACHE] weight_cache_type=2 (in-place) is "
                "unsafe under AUTO scheduling without prepack+cross_warm, an "
                "effective custom-kernel decode pack and unlimited LRU "
                "capacity. Downgrading process-wide to out-of-place "
                "(weight_cache_type=1) for the rest of the run; kernel "
                "selection unchanged.");
#endif
    }
    return false;
}

// ── Dispatch ────────────────────────────────────────────────────────────

bool group_matmul_run_parallel_dispatch(const std::vector<char> &layout,
        const std::vector<bool> &transA, const std::vector<bool> &transB,
        const std::vector<int> &M, const std::vector<int> &N,
        const std::vector<int> &K, const std::vector<float> &alpha,
        const std::vector<const void *> &src, const std::vector<int> &lda,
        const std::vector<const void *> &weight, const std::vector<int> &ldb,
        const std::vector<const void *> &bias, const std::vector<float> &beta,
        const std::vector<void *> &dst, const std::vector<int> &ldc,
        const std::vector<bool> &is_weights_const,
        std::vector<matmul_params> &params, const int num_threads,
        const char **gemm_mode_out, grp_matmul_gated_act_t fused_act,
        data_type_t act_dtype) {

    scoped_grp_auto_mutation_tracking mutation_tracking;
    bool track_grouped_inplace_mutation = false;

    // Must be the EFFECTIVE verdict, not the master knob: this feeds
    // `a3_fuses`, which in turn decides whether the tight-dst guard runs.
    // Reading the master knob (or `params[0]`, which in MoE decode is
    // routinely an inactive padding slot whose dtypes are unset) says "CK
    // will fuse" for calls the kernel then refuses -- and the tight
    // split-halves path answers that refusal by demoting the layer to serial
    // Sequential.  `grp_matmul_custom_kernel_effective` folds the family
    // sub-toggles and the per-group disqualifier, and picks its
    // representative from the first ACTIVE expert.
    // Must be the EFFECTIVE verdict, not the master knob: this feeds
    // `a3_fuses`, which decides whether the tight-dst guard runs.  The master
    // knob (and `params[0]`, routinely an inactive padding slot in MoE
    // decode) says "CK will fuse" for calls the kernel then refuses, and the
    // tight split-halves path answers that refusal by demoting the layer to
    // serial Sequential.
    const bool custom_kernel_en
            = grp_matmul_custom_kernel_effective(params, M, N, fused_act);

    // ── WEIGHT_CACHE=2 (in-place) safety downgrade for grouped matmul ─────
    // In-place reorder/pack mutates the caller's weight buffer into a
    // backend-/layout-specific blocked form, so it is only safe when a
    // SINGLE layout ever touches a given weight buffer.  A phase pin can
    // violate that under AUTO on the SAME W13/W2 buffers: the prompt routes
    // to an AOCL full-weight reorder while decode routes to AOCL per-tile or
    // the custom-kernel pack — two layouts over one buffer (corruption).
    // So under AUTO we take one of three deterministic verdicts (detailed at
    // the (A)/(B) branch below):
    //   (A) keep WC=2 in a MIXED in-place mode when it is provably safe
    //       (PREPACK + CROSS_WARM + unlimited LRU pre-warm every out-of-
    //       place layout from raw W before the single in-place prompt
    //       mutation), or
    //   (B) downgrade the process-wide cache mode to out-of-place (1)
    //       otherwise — a mirror of the single-matmul guard in
    //       `lowoha_matmul_utils.cpp`, or
    //   (C) for a per-call mixed raw/caller-packed pool under an otherwise
    //       eligible process policy, keep mixed mode active but cap only that
    //       call's weights at out-of-place caching.
    // Process-wide verdicts (A)/(B) are derived from process-constant env, so
    // their decision (and one-shot log) is stable for the rest of the run.
    // Under (A), a BF16/FP16 MoE call (`grp_matmul_auto_algo3_call`) that
    // runs ALGO 3 while neither phase can leave it also gets a per-call
    // custom-kernel layout licence: AOCL DLP reorders are capped
    // out-of-place, and families with in-place CK support may use it.
    //
    // A generic global pin keeps WC=2 only when phase overrides do not split
    // BF16/FP16 layout ownership. Split global/phase policy is unsupported
    // under WC=2 and fails before any prepack or destination write.
    //
    // Single-layout pinned algos keep one layout per weight:
    //   * ALGO 1/2/5/6 — one AOCL reorder layout per buffer, via AOCL's
    //     own in-place path (aocl_kernel.cpp, size-gated with an out-of-
    //     place fall-back when the blocked layout exceeds the plain size).
    //   * ALGO 3 + CUSTOM_KERNEL off — pure AOCL DLP, single layout.
    //   * ALGO 3 + CUSTOM_KERNEL on — made safe in the pack layer:
    //     bf16/even-K packs IN PLACE into the (single-consumer) weight
    //     buffer; int8 (extra compensation row) and odd-K bf16 (VNNI
    //     K-pair pad) fall back to an out-of-place cache that never
    //     mutates the weight (see prepare_for_call +
    //     get_or_pack_weight_bf16).
    const bool single_expert_epilogue_ok = M.size() != 1
            || ck_serves_single_expert_epilogue(alpha[0], beta[0], bias[0],
                    /*tight=*/ldc[0] == N[0] / 2, fused_act, act_dtype);
    // One single-expert classification serves the licence and the selector.
    const single_expert_ck_t single_ck
            = single_expert_epilogue_ok && M.size() == 1
            ? grp_matmul_single_expert_call_ck(
                      M, N, K, params, fused_act, num_threads)
            : single_expert_ck_t {};
    const bool auto_algo3_layer
            = grp_matmul_auto_algo3_call(M, params) || single_ck.both();
    // Capture the effective per-call cache mode before any ownership policy
    // caps the AOCL side; CK preparation consumes this original value.
    int32_t ck_weight_cache_mode
            = matmul_config_t::instance().get_weight_cache();
    for (size_t i = 0; i < params.size() && i < M.size(); ++i) {
        if (M[i] <= 0) { continue; }
        ck_weight_cache_mode
                = std::min(ck_weight_cache_mode, params[i].weight_cache_type);
    }
    if (grp_matmul_wc2_global_phase_split_unsupported_for_call(
                M, N, K, params, num_threads)) {
        log_error(
                "group_matmul: WEIGHT_CACHE=2 with a non-AUTO global ALGO "
                "and phase overrides that split BF16/FP16 prompt/decode "
                "layout ownership is unsupported. Use WEIGHT_CACHE=1, remove "
                "the phase override, or set ZENDNNL_GRP_MATMUL_ALGO=0.");
        if (gemm_mode_out != nullptr) {
            *gemm_mode_out = kGrpMatmulErrWc2PhaseSplitUnsupported;
        }
        return false;
    }

    bool ck_inplace_licence = false;
    track_grouped_inplace_mutation = prepare_grouped_auto_weight_cache(params,
            custom_kernel_en, nullptr, auto_algo3_layer, &ck_inplace_licence);
    // Without the licence, a process downgraded to WEIGHT_CACHE=1 just now
    // must not see this first call pack in place.
    if (!ck_inplace_licence) {
        ck_weight_cache_mode = std::min(ck_weight_cache_mode,
                matmul_config_t::instance().get_weight_cache());
    }

    mutation_tracking.set(track_grouped_inplace_mutation);

    auto_algo_trace algo_trace;
    const int use_algo = select_grp_matmul_algo(layout, M, N, K, params,
            num_threads, &algo_trace, fused_act, single_expert_epilogue_ok,
            &single_ck);

    // A call ALGO 3 cannot serve follows the mixed policy.  `params` is a
    // per-call copy, so the cap does not reach the caller.
    ck_inplace_licence = ck_inplace_licence && use_algo == 3;
    if (ck_inplace_licence) {
        for (auto &p : params) {
            p.weight_cache_type = std::min<int32_t>(p.weight_cache_type, 1);
        }
    }

    // Decide whether the chosen ALGO fuses the gated activation inline.
    //   - ALGOs 1/2/5/6 always fuse (per-expert or per-M-tile).
    //   - ALGO 3 fuses whenever the activation layout fits the N-tile
    //     split, either because:
    //       (i) the caller passed a tight [M, I]-layout destination
    //           (ldc[0] < N[0]) — a separate-pass swiglu would overrun
    //           that buffer, so fused activation is a correctness
    //           requirement, not a perf toggle.  This is the fused-MoE
    //           internal-alloc tight path auto-engage.
    //      (ii) `ZENDNNL_GRP_MATMUL_N_TILE_FUSED_ACT=1` is set
    //           (explicit opt-in from non-tight callers who want to
    //           avoid the separate-pass round-trip).
    //   - For any fused_act we cannot fuse, the caller runs a separate
    //     activation pass after this function returns.
    // Read the stride from the first ACTIVE expert, not slot 0: an inactive
    // slot is a padding placeholder whose stride is arbitrary, and the
    // caller-boundary uniformity check deliberately skips those, so slot 0
    // could classify a uniformly wide active set as tight or the reverse.
    // `flat_n_tile` infers the same property the same way; the two must agree
    // or the executor plans a layout the dispatcher did not authorise.
    size_t layout_rep = 0;
    for (size_t i = 0; i < M.size(); ++i) {
        if (M[i] > 0) {
            layout_rep = i;
            break;
        }
    }
    const bool caller_layout_tight = (use_algo == 3) && layout_rep < ldc.size()
            && layout_rep < N.size() && ldc[layout_rep] < N[layout_rep];
    // Wide-fused (caller's ldc ≥ N) routes through the standard
    // backend's `apply_swiglu_oai_tile_rows`; that helper handles
    // swiglu_oai_mul only.  silu_and_mul and gelu_and_mul have no
    // wide-helper siblings yet, so they can only fuse on the tight
    // layout (CK path).  `a3_can_fuse_act` already gates silu/gelu
    // on `use_custom_kernel=true` — combined with this tight-only
    // gate, the silu/gelu fused path engages exclusively when
    // (CK-on AND tight caller).  Wide non-CK silu/gelu falls through
    // to the dispatcher's separate-pass post-pass.
    const bool wide_fuse_supported
            = (fused_act == grp_matmul_gated_act_t::swiglu_oai_mul)
            && get_grp_n_tile_fused_act();
    const bool a3_fuses = (use_algo == 3)
            && a3_can_fuse_act(fused_act, custom_kernel_en)
            && (caller_layout_tight || wide_fuse_supported);
    const bool act_fused = a3_fuses
            || ((use_algo != 3) && (fused_act != grp_matmul_gated_act_t::none));

    // ── ALGO-decision APILOG ──────────────────────────────────────────
    // Emits the chosen ALGO and the discriminators that drove the
    // decision (POST env-override, POST auto-select).  Single line at
    // info level; users debugging "why did my shape land on ALGO X"
    // get a complete story — shape + all gates + the chosen algo +
    // CK-eligibility hint.  Sister line to `[GRP_MATMUL.CALL]` (emitted
    // at the top of group_matmul_direct.cpp) which carries the framework
    // input metadata, and `[GRP_MATMUL.EXEC]` / `[GRP_MATMUL.PLAN]` /
    // `[GRP_MATMUL.PREPACK]` which cover the rest of the per-call trail.
    // Gated by apilog_info_enabled() (cached); free when logging is off.
    static const bool s_dispatch_log = apilog_info_enabled();
    if (s_dispatch_log && !M.empty()) {
        const int requested_algo = get_grp_matmul_requested_algo();
        const int env_algo = requested_algo == kGrpMatmulAlgoNTileFlatParallel
                ? kGrpMatmulAlgoAuto
                : requested_algo;
        const int max_M_v = max_active_grp_matmul_m(M, M.size());
        const int max_N_v = *std::max_element(N.begin(), N.end());
        const int max_K_v = *std::max_element(K.begin(), K.end());
        // Representative expert for the quant-mode hint fields below.  In MoE
        // decode a LEADING expert is often inactive (M==0), and the grouped /
        // fallback DQ pre-pass rewrites ONLY active experts to s8 + cleared
        // dynamic_quant + per-token src_scale.  Reading params[0] blindly
        // would mislabel the log (ck_family=none / dynamic_quant=no) even when
        // CK int8 runs on every active tile.  Pick the first ACTIVE expert
        // (mirrors flat_n_tile's routing classifier); fall back to 0 when all
        // inactive.  wei dtype is uniform across active/inactive, so the
        // wei/expert(MB) telemetry below is unaffected by the choice.
        size_t rep = 0;
        for (size_t i = 0; i < M.size() && i < params.size(); ++i) {
            if (M[i] > 0) {
                rep = i;
                break;
            }
        }
        const size_t wei_elem_b = size_of(params[rep].dtypes.wei);
        const size_t wei_per_expert_mb
                = (static_cast<size_t>(max_K_v) * max_N_v * wei_elem_b) >> 20;
        // Phase + unified per-phase settings for telemetry. Surface both the
        // raw accepted request and generic effective value so phase request 4
        // is visible without implying that numeric 4 reached this dispatcher.
        const grp_matmul_phase phase = classify_grp_matmul_phase(max_M_v);
        const auto prompt_setting = get_grp_matmul_auto_prompt_setting();
        const auto decode_setting = get_grp_matmul_auto_decode_setting();
        // Reason — which gate drove the chosen ALGO.
        //
        //   * env_ok / env_fallback  — global `ZENDNNL_GRP_MATMUL_ALGO` hit
        //                              or safety-clamped (the clamp emits a
        //                              [WARN] line as well).
        //   * auto_*                 — whatever rule `auto_select_algo`
        //                              matched, named by that rule itself.
        //                              A `_clamp` suffix means m_tile_safe /
        //                              n_tile_safe downgraded the rule's own
        //                              answer, so it never appears on a rule
        //                              that deliberately picked ALGO 1.
        //
        // The AUTO arm reads what the selector recorded rather than walking
        // the rule table a second time.  Two independent copies of that table
        // cannot be kept in step: the previous one had already lost Rules 0.45
        // and 0.5, so a single-expert decode routed by 0.45 was reported as
        // `auto_phase_env`, and a few-expert decode default could be labelled
        // `auto_phase_env_clamp` with no clamp anywhere in the call.
        std::string reason_buf;
        const char *reason = nullptr;
        if (is_grp_matmul_generic_algo(env_algo)) {
            reason = (env_algo == use_algo) ? "env_ok" : "env_fallback";
        } else if (algo_trace.reason != nullptr) {
            reason = algo_trace.reason;
            if (algo_trace.unclamped != use_algo) {
                reason_buf = std::string(reason) + "_clamp";
                reason = reason_buf.c_str();
            }
        } else {
            reason = "auto_rule_legacy";
        }
        // CK eligibility hint: a single boolean that combines the
        // structurally-knowable conditions a level-3 reader can see
        // without consulting the deeper dispatcher.  The runtime CK
        // gate (`custom_kernel::prepare_for_call`) adds per-expert
        // checks not visible here (`transA`, `alpha`, `beta`,
        // `is_weights_const`, `ldb` min-row-stride, fused-act/bias dtype
        // matrix).  Surface as a hint, not a guarantee.
        const int log_custom_kernel = custom_kernel_en;
        const int log_custom_kernel_int8 = get_grp_matmul_custom_kernel_int8();
        // BF16 family hint — same gate as before.
        const bool ck_hint_bf16 = (use_algo == 3) && log_custom_kernel
                && (params[rep].dtypes.src == data_type_t::bf16)
                && (params[rep].dtypes.wei == data_type_t::bf16)
                && !params[rep].dynamic_quant;
        // FP16 family hint — native AVX-512-FP16 f16×f16→{f16,f32}.
        // Gated on the master CK env AND the F16 sub-knob; the runtime
        // `prepare_for_call` adds the AVX-512-FP16 ISA / toolchain check
        // not visible here, so this is a structural hint, not a guarantee.
        const int log_custom_kernel_f16 = get_grp_matmul_custom_kernel_f16();
        const bool ck_hint_f16 = (use_algo == 3) && log_custom_kernel
                && log_custom_kernel_f16
                && (params[rep].dtypes.src == data_type_t::f16)
                && (params[rep].dtypes.wei == data_type_t::f16)
                && !params[rep].dynamic_quant;
        // B.6 hardening — DQ-INT8 family hint.  Mirrors the upstream
        // `ck_eligible_int8` predicate (in prepack/prepack.cpp) so the
        // PLAN apilog surfaces both regimes.  Evaluating either family
        // independently lets a level-3 reader see at a glance whether
        // a DQ-INT8 call is structurally CK-eligible (separate from
        // master env knob + INT8 sub-knob cascade).  The runtime
        // `prepare_for_call` adds shape / pack_nr / per-expert checks
        // not visible here; the hint is informational, not a guarantee.
        // Two structural forms reach the DQ-INT8 CK microkernel, both with
        // wei=s8, dst=bf16, compute=s8/u8 (mirrors `resolve_variant`):
        //   1. runtime hoist  — dynamic_quant=true with a bf16 src that the
        //      N-tile executor quantizes to s8 before dispatch_tile.
        //   2. grouped pre-quant — the group_dynamic_quant pre-pass already
        //      produced an s8 src + per-token src_scale and CLEARED
        //      dynamic_quant.  Without this branch the hint mislabelled the
        //      grouped decode path as ck_family=none / dynamic_quant=no even
        //      though CK runs on 100% of tiles.
        const bool ck_int8_common = (use_algo == 3) && log_custom_kernel
                && log_custom_kernel_int8
                && (params[rep].dtypes.wei == data_type_t::s8);
        const bool ck_dq_int8_shapes = ck_int8_common
                && (params[rep].dtypes.dst == data_type_t::bf16
                        || params[rep].dtypes.dst == data_type_t::f32)
                && (params[rep].dtypes.compute == data_type_t::s8
                        || params[rep].dtypes.compute == data_type_t::u8);
        const bool ck_static_u8s8_shape = ck_int8_common
                && is_static_u8s8_quant_config(params[rep])
                && (params[rep].dtypes.dst == data_type_t::bf16
                        || params[rep].dtypes.dst == data_type_t::f32
                        || params[rep].dtypes.dst == data_type_t::u8);
        const bool ck_hint_int8
                = (ck_dq_int8_shapes
                          && ((params[rep].dynamic_quant
                                      && params[rep].dtypes.src
                                              == data_type_t::bf16)
                                  || (params[rep].dtypes.src == data_type_t::s8
                                          && params[rep].quant_params.src_scale
                                                          .buff
                                                  != nullptr)))
                || ck_static_u8s8_shape;
        const bool ck_hint_s4 = (use_algo == 3) && log_custom_kernel
                && (params[rep].dtypes.wei == data_type_t::s4)
                && (params[rep].dtypes.dst == data_type_t::bf16)
                && (params[rep].dtypes.compute == data_type_t::s8)
                && ((params[rep].dynamic_quant
                            && params[rep].dtypes.src == data_type_t::bf16)
                        || (params[rep].dtypes.src == data_type_t::s8
                                && params[rep].quant_params.src_scale.buff
                                        != nullptr));
        const bool ck_hint
                = ck_hint_bf16 || ck_hint_int8 || ck_hint_f16 || ck_hint_s4;
        const char *ck_family = ck_hint_bf16 ? "bf16"
                : ck_hint_int8
                ? ((params[rep].dtypes.src == data_type_t::u8
                           || params[rep].dtypes.compute == data_type_t::u8)
                                  ? "int8_asym"
                                  : "int8_sym")
                : ck_hint_f16 ? "f16"
                : ck_hint_s4  ? "w4a8_s4"
                              : "none";
        // SELECTION record (emitted BEFORE the executor runs): `chosen=ALGO_X`
        // is the algo the selector picked, with `reason` explaining the gate.
        // The ALGO that ACTUALLY executed — including any in-executor clamp or
        // fork (e.g. ALGO 2 -> sequential-full-team when active_ops>num_threads,
        // or ALGO 2 -> vertical fusion in the fused path) — is reported by the
        // post-exec `[GRP_MATMUL.CALL]` line via `mode=` (precise branch) and
        // `exec_algo=` (the real generic scheduler ID).  Compare those two
        // lines to see any selection-vs-execution divergence.
        apilog_info("[GRP_MATMUL.ALGO] chosen=ALGO_", use_algo,
                " requested_algo=", requested_algo, " env_algo=", env_algo,
                " reason=", reason, " phase=", grp_matmul_phase_name(phase),
                " auto_prompt_requested=", prompt_setting.requested_algo,
                " auto_prompt_env=", prompt_setting.generic_effective_algo,
                " auto_decode_requested=", decode_setting.requested_algo,
                " auto_decode_env=", decode_setting.generic_effective_algo,
                " act=", act_name(fused_act),
                " act_fused=", (act_fused ? "yes" : "no"),
                " ck_eligible_hint=", (ck_hint ? "yes" : "no"),
                " ck_family=", ck_family,
                " dynamic_quant=", (params[rep].dynamic_quant ? "yes" : "no"),
                " num_ops=", static_cast<int>(M.size()),
                " num_threads=", num_threads, " max_M=", max_M_v,
                " max_N=", max_N_v, " max_K=", max_K_v,
                " wei/expert(MB)=", wei_per_expert_mb,
                " wide_N=", (max_N_v > max_K_v ? "yes" : "no"),
                " many_experts=",
                (static_cast<int>(M.size()) >= 16 ? "yes" : "no"),
                " decode5_pin_honoured=",
                (algo_trace.decode5_pin_honoured ? "yes" : "no"),
                " decode5_pin_declined=",
                (algo_trace.decode5_pin_declined ? "yes" : "no"),
                " prompt5_pin_honoured=",
                (algo_trace.prompt5_pin_honoured ? "yes" : "no"),
                " prompt5_pin_declined=",
                (algo_trace.prompt5_pin_declined ? "yes" : "no"),
                " caller_tight=", (caller_layout_tight ? "yes" : "no"));
    }

    auto set_mode = [&](const char *s) {
        if (gemm_mode_out != nullptr) { *gemm_mode_out = s; }
    };

    // Whole-call no-op short-circuit.  When no expert fires (every M[i] <= 0,
    // including the empty-vector case), there is no GEMM to run on ANY algo.
    // Mark the executed path "skip" (exec_algo=0 on the post-exec
    // [GRP_MATMUL.CALL] line) and return before the switch so we neither
    // prepack nor mislabel exec_algo with the *selected* ALGO for a call that
    // executed nothing.  The per-slot M<=0 guards inside the executors still
    // handle the mixed case (some experts active, some padded/inactive); this
    // covers the all-inactive call once, in one place.
    //
    // Return `true` (activation already handled) — NOT `act_fused`.  There is
    // nothing to activate, and the caller's separate-pass post-op fires on
    // `!return_value`; with no active rows the dst slots may be null by
    // contract, so a separate pass would dereference null.  Reporting the
    // no-op as "fused" makes every caller skip that post-pass.
    if (std::none_of(M.begin(), M.end(), [](int m) { return m > 0; })) {
        set_mode("skip");
        return true;
    }

    // CK-only-or-fail: a caller-prepacked CUSTOM-KERNEL VNNI weight is
    // consumable ONLY by the custom kernel, which runs exclusively on
    // ALGO 3 (flat_n_tile).  If the call routed to any other ALGO, the
    // executor would read the packed bytes as a raw weight → silent
    // corruption.  Signal failure via the gemm_mode sentinel (group_matmul
    // _direct returns status_t::failure on this value).  ALGO 3 itself
    // re-checks CK engagement and raises the same sentinel from flat_n_tile.
    //
    // Detection MUST match the CK-VNNI classifier in flat_n_tile:
    // `mem_format_b == 'r'` is ambiguous (it also marks AOCL-DLP-blocked
    // and GGML unpack+reorder outputs), so gate on
    // `lowoha_algo == moe_custom_kernel`.  An AOCL-blocked / GGML-reordered
    // 'r' weight is legitimately consumable by the non-CK executors, so it
    // must NOT trip this guard.
    //
    // A buffer an earlier in-place CK pack (WEIGHT_CACHE=2) rewrote is in the
    // same position while still marked 'n': it holds the custom-kernel image,
    // not the raw weight.
    {
        bool has_prepacked_b = false;
        bool has_unsupported_prepacked_b = false;
#if !ZENDNNL_DEPENDS_AOCLDLP
        bool has_aocl_prepacked_b = false;
#endif
        for (size_t i = 0; i < params.size() && i < M.size(); ++i) {
            if (M[i] <= 0 || params[i].mem_format_b != 'r') { continue; }
            if (params[i].lowoha_algo == matmul_algo_t::moe_custom_kernel) {
                has_prepacked_b = true;
            } else if (is_aocl_prepacked_weight(params[i])) {
#if !ZENDNNL_DEPENDS_AOCLDLP
                has_aocl_prepacked_b = true;
#endif
            } else {
                has_unsupported_prepacked_b = true;
            }
        }
        if (has_unsupported_prepacked_b) {
            log_error(
                    "group_matmul: pre-reordered weight has unsupported "
                    "backend provenance; only AOCL-DLP and custom-kernel "
                    "packed layouts are accepted");
            set_mode(kGrpMatmulErrPrepackedNoCk);
            return false;
        }
#if !ZENDNNL_DEPENDS_AOCLDLP
        if (has_aocl_prepacked_b) {
            log_error(
                    "group_matmul: AOCL-blocked weight cannot execute because "
                    "ZenDNNL was built without AOCL-DLP; refusing to "
                    "reinterpret packed bytes as raw");
            set_mode(kGrpMatmulErrPrepackedNoCk);
            return false;
        }
#endif
        if (has_prepacked_b && use_algo != 3) {
            set_mode(kGrpMatmulErrPrepackedNoCk);
            return false;
        }
        if (use_algo != 3
                && custom_kernel::any_weight_packed_in_place(weight, M)) {
            log_error(
                    "group_matmul: an active expert's weight was packed in "
                    "place for the custom kernel by an earlier ALGO 3 call "
                    "(ZENDNNL_MATMUL_WEIGHT_CACHE=2), but this call resolved "
                    "to ALGO ",
                    use_algo,
                    ", which reads the raw weight.  Keep every call on these "
                    "weights on ALGO 3, or use "
                    "ZENDNNL_MATMUL_WEIGHT_CACHE=1.");
            set_mode(kGrpMatmulErrPackedInPlaceNoCk);
            return false;
        }
    }

    // Tight-destination-without-a-tight-writer guard.
    //
    // A tight dst (`ldc < N`) has no room for the wide 2I result, so it is
    // only safe when a writer that knows to compact 2I into I actually runs.
    // That writer exists in exactly one place: ALGO 3's fused epilogue, i.e.
    // `a3_fuses`.  Every other route -- ALGO 1/2/5/6, and ALGO 3 when it does
    // not fuse -- applies the activation in place at the caller's stride via
    // `apply_gated_act_inplace`, which walks `2 * (N/2)` columns across a row
    // that physically holds only `ldc` of them.  That reads the following row
    // as the up half and, on the last row, writes past the end of the buffer.
    //
    // The caller-boundary check in `group_matmul_direct` rejects only a MIXED
    // tight/wide set, so a uniformly tight caller reaches here unscreened.
    // Two ways in under AUTO:
    //
    //   * A prompt phase pin routes prompt-shaped calls to ALGO 1, and so does
    //     the n_tile_safe clamp in either phase, so a tight caller that is
    //     correct on ALGO 3 (fused) silently corrupts there -- for every
    //     activation family, custom kernel on or off.
    //   * With the custom kernel off, `a3_can_fuse_act` declines silu/gelu, so
    //     those lose the tight writer at decode as well.
    //
    // `use_algo` and `a3_fuses` are both resolved by this point, so this is
    // the first place that can tell the safe case from the unsafe one.
    if (fused_act != grp_matmul_gated_act_t::none && !a3_fuses) {
        for (size_t i = 0; i < M.size() && i < ldc.size() && i < N.size();
                ++i) {
            if (M[i] > 0 && ldc[i] < N[i]) {
                log_error("group_matmul: expert ", i,
                        " has a tight destination (ldc=", ldc[i], " < N=", N[i],
                        ") with a gated activation, but the resolved route "
                        "(ALGO ",
                        use_algo,
                        ") applies the activation in place and would write "
                        "past the end of that buffer.  Pass a destination with "
                        "ldc >= N, or use the fused-MoE entry point, which "
                        "owns the tight arena and guarantees ALGO 3.");
                set_mode(kGrpMatmulErrTightNoFusedWriter);
                return false;
            }
        }
    }
    // ALGO 3/AUTO: materialize plain s8 only for ALGO 3's effective
    // simulated W4A8 path.
    const int dispatch_num_ops = static_cast<int>(M.size());
    static thread_local std::vector<void *> w4a8_s8_ptrs;
    bool any_w4a8 = false;
    const bool w4a8_algo3_simulated
            = w4a8_runtime_algo(/*scheduling_algo=*/3, resolve_kernel())
            == matmul_algo_t::aocl_dlp;
    if ((use_algo == 3 || use_algo == 0) && w4a8_algo3_simulated) {
        w4a8_populate_plain_s8_cache(weight, K, N, ldb, transB, params,
                dispatch_num_ops, w4a8_s8_ptrs, any_w4a8);
    }

    // Fail-closed baseline for the in-place mutation registry.  A reorder that
    // missed on an already-mutated weight buffer is refused inside the AOCL
    // backend, which can only COUNT it: `run_dlp` returns void and the refusal
    // can happen on any thread inside an executor's OMP region.  Snapshot the
    // count here and compare after the executor, so a refusal anywhere in the
    // call turns into a failed call rather than a dst nobody reordered for.
    const uint64_t mutated_conflicts_before
            = aocl_mutated_source_conflict_count(
                    weight, /*secondary_weights=*/nullptr, &M);

    switch (use_algo) {
        case 1:
            set_mode("sequential_experts");
            sequential_experts(layout, transA, transB, M, N, K, alpha, src, lda,
                    weight, ldb, bias, beta, dst, ldc, is_weights_const, params,
                    num_threads, fused_act, act_dtype, gemm_mode_out);
            break;
        case 2:
            // flat_m_tile owns its gemm_mode — it writes the concrete branch it ran
            // (flat_m_tile_multitier / _single_tier / _seq_clamp) into gemm_mode_out,
            // like flat_n_tile does, so the post-exec [GRP_MATMUL.CALL] line reflects
            // the real M-tile path rather than a generic "flat_m_tile".
            flat_m_tile(layout, transA, transB, M, N, K, alpha, src, lda,
                    weight, ldb, bias, beta, dst, ldc, fused_act, act_dtype,
                    is_weights_const, params, num_threads, gemm_mode_out);
            break;
        case 3:
            // flat_n_tile handles both the legacy non-fused path and the fused
            // epilogue.  Pass fused_act when a3_fuses; pass `none` otherwise so
            // the legacy path runs (and the caller does the separate activation).
            //
            // The executor writes the concrete path name to `gemm_mode_out`
            // itself — one of `"flat_n_tile"`, `"flat_n_tile_custom"`,
            // `"flat_n_tile_fused_swiglu_oai"`, or
            // `"flat_n_tile_fused_swiglu_oai_custom"` — so benchdnn /
            // profiler output reveals whether the custom BF16 microkernel
            // engaged for this call without needing APILOG enabled.
            flat_n_tile(layout, transA, transB, M, N, K, alpha, src, lda,
                    weight, ldb, bias, beta, dst, ldc, is_weights_const, params,
                    num_threads,
                    a3_fuses ? fused_act : grp_matmul_gated_act_t::none,
                    act_dtype, gemm_mode_out,
                    any_w4a8 ? &w4a8_s8_ptrs : nullptr, ck_weight_cache_mode,
                    ck_inplace_licence);
            break;
        case kGrpMatmulAlgoMultilevel:
            // parallel_multilevel owns its gemm_mode (multilevel_concurrent /
            // multilevel_rounds), written into gemm_mode_out.
            parallel_multilevel(layout, transA, transB, M, N, K, alpha, src,
                    lda, weight, ldb, bias, beta, dst, ldc, is_weights_const,
                    params, num_threads, fused_act, act_dtype, gemm_mode_out);
            break;
        case 5:
            set_mode("per_expert");
            parallel_per_expert(layout, transA, transB, M, N, K, alpha, src,
                    lda, weight, ldb, bias, beta, dst, ldc, is_weights_const,
                    params, num_threads, fused_act, act_dtype);
            break;
        default:
            set_mode("sequential_experts");
            sequential_experts(layout, transA, transB, M, N, K, alpha, src, lda,
                    weight, ldb, bias, beta, dst, ldc, is_weights_const, params,
                    num_threads, fused_act, act_dtype, gemm_mode_out);
            break;
    }

    // Overwrites whatever path name the executor wrote: an error sentinel
    // outranks the record of which strategy ran, and the `error_` prefix is
    // what `group_matmul_direct` / `group_matmul_fused_moe` translate into
    // `status_t::failure`.  Deliberately compares against the snapshot rather
    // than testing for non-zero, so an unrelated earlier refusal in the
    // process cannot fail an otherwise good call.
    if (aocl_mutated_source_conflict_count(
                weight, /*secondary_weights=*/nullptr, &M)
            != mutated_conflicts_before) {
        set_mode(kGrpMatmulErrMutatedWeightMiss);
    }
    return act_fused;
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
