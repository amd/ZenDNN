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

/// ALGO 5 — expert-parallel grouped GEMM executor, plus the selection policy
/// that decides whether a call may reach it.  See the two headers alongside
/// for the strategy summary and the pin qualifier's rationale.

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <vector>

#include <omp.h>

#include "group_matmul_expert_parallel.hpp"
#include "group_matmul_expert_parallel_policy.hpp"

#include "../group_matmul_direct.hpp" // apply_gated_act_inplace
#include "../group_matmul_parallel_common.hpp"
#include "../prepack/prepack.hpp"
#include "lowoha_operators/common/omp_thread_control.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

namespace {

/// Compacted list of FIRING (`M[i] > 0`) expert indices.
///
/// Compacting up front means the dynamic schedule hands out one ticket per
/// real unit of work.  Inactive slots are padded placeholders that may carry
/// null pointers, and MoE routing leaves most of them inactive, so skipping
/// them inside the loop would still pay an atomic fetch-add on the shared
/// iteration counter — the only cross-thread state on this path — for each.
/// Filtering first also lets the thread count be clamped to the real work.
///
/// Inline storage for every shipping expert count; heap only beyond it.
struct active_experts_t {
    static constexpr size_t kStackSlots = 256;

    explicit active_experts_t(size_t num_ops) {
        if (num_ops > kStackSlots) {
            heap.resize(num_ops);
            idx = heap.data();
        } else {
            idx = stack;
        }
    }

    void add(size_t i) { idx[count++] = static_cast<int>(i); }

    int stack[kStackSlots];
    std::vector<int> heap;
    int *idx = nullptr;
    int count = 0;
};

/// Emit a WARNING-level message at most once per process, latched on `once`.
/// Decode re-evaluates a declined pin on every token, so a per-call warning
/// would flood the default log level.  The caller owns the latch so each
/// distinct message gets its own.
template <typename... Args>
void warn_once(std::atomic<bool> &once, Args &&...args) {
    static const bool enabled = apilog_warning_enabled();
    if (!enabled) { return; }
    if (once.exchange(true, std::memory_order_relaxed)) { return; }
    apilog_warning(std::forward<Args>(args)...);
}

} // namespace

// `schedule(dynamic, 1)`: MoE routing is M-skewed, so a static partition
// leaves threads that drew light experts idle — worst when `num_ops >
// num_threads` and each thread runs several experts in sequence.  Each
// iteration writes only its own `dst[i]` slice, so dynamic ordering violates
// no dependency.

void parallel_per_expert(const std::vector<char> &layout,
        const std::vector<bool> &transA, const std::vector<bool> &transB,
        const std::vector<int> &M, const std::vector<int> &N,
        const std::vector<int> &K, const std::vector<float> &alpha,
        const std::vector<const void *> &src, const std::vector<int> &lda,
        const std::vector<const void *> &weight, const std::vector<int> &ldb,
        const std::vector<const void *> &bias, const std::vector<float> &beta,
        const std::vector<void *> &dst, const std::vector<int> &ldc,
        const std::vector<bool> &is_weights_const,
        std::vector<matmul_params> &params, int num_threads,
        grp_matmul_gated_act_t fused_act, data_type_t act_dtype) {

    const size_t num_ops = M.size();
    if (num_ops == 0 || num_threads <= 0) { return; }

    // `num_threads` is forwarded so cross_warm can prefill regime 2 for the
    // ALGO 3 decode path when CUSTOM_KERNEL=0.
    group_matmul_prepack::prepack_for_algo_5(
            group_matmul_prepack::build_prepack_params(weight, K, N, ldb,
                    transB, is_weights_const, params, M,
                    get_grp_matmul_custom_kernel(), num_threads,
                    algo3_decode_nr_align(M, N, ldc, fused_act, params),
                    fused_act, act_dtype,
                    /*transA=*/&transA, /*alpha=*/&alpha, /*beta=*/&beta));

    // See `active_experts_t` for why the firing experts are compacted first.
    active_experts_t act(num_ops);
    for (size_t i = 0; i < num_ops; ++i) {
        if (M[i] > 0) { act.add(i); }
    }
    const int num_active = act.count;
    const int *active = act.idx;
    if (num_active == 0) { return; }

    matmul_algo_t algo = resolve_kernel();
    scoped_active_levels guard(1);

    const bool want_act = (fused_act != grp_matmul_gated_act_t::none);
    // No intra-expert split, so threads beyond the firing-expert count can
    // never win a ticket and only widen the fork/join.
    const int nthr = std::min(num_threads, num_active);

    // ALGO 5 has no planner, so this is the only record of what it decided.
    // `active < num_threads` means idle cores no tuning can reach.
    static const bool s_log_plan = apilog_info_enabled();
    if (s_log_plan) {
        const matmul_algo_t first_kernel
                = resolve_expert_kernel(5, algo, params[active[0]]);
        bool mixed_kernels = false;
        for (int a = 1; a < num_active; ++a) {
            if (resolve_expert_kernel(5, algo, params[active[a]])
                    != first_kernel) {
                mixed_kernels = true;
                break;
            }
        }
        apilog_info("[GRP_MATMUL.PLAN] algo=5 expert_parallel num_ops=",
                num_ops, " active=", num_active, " team_req=", nthr, "/",
                num_threads, " act=", (want_act ? "fused_pass" : "none"),
                " kernel=", static_cast<int>(first_kernel),
                " kernel_mixed=", (mixed_kernels ? 1 : 0));
    }

    // `if`: one firing expert has nothing to distribute, and that is the M=1
    // decode shape where the fixed cost is the whole cost.
#pragma omp parallel for num_threads(nthr) \
        schedule(dynamic, 1) if (num_active > 1)
    for (int a = 0; a < num_active; ++a) {
        const int i = active[a];
        execute_expert_slice(layout[i], transA[i], transB[i], M[i], N[i], K[i],
                alpha[i], src[i], lda[i], weight[i], ldb[i], bias[i], beta[i],
                dst[i], ldc[i], is_weights_const[i], 1, params[i],
                resolve_expert_kernel(5, algo, params[i]));
        if (want_act) {
            apply_gated_act_inplace(
                    fused_act, dst[i], 0, M[i], N[i], ldc[i], act_dtype);
        }
    }
}

// ── Fused pipeline (W13 -> gated act -> W2) ─────────────────────────────
//
// See `try_expert_parallel_pipeline` in the header for the decline contract.

namespace {

// Dtypes `apply_gated_act_inplace` implements.  It returns SILENTLY for
// anything else, so an unchecked dtype would publish an unactivated
// intermediate rather than report a problem.
bool is_pipeline_act_dtype(data_type_t dt) {
    return dt == data_type_t::f32 || dt == data_type_t::bf16
            || dt == data_type_t::f16;
}

// Dtypes the DA8W8 source quantization implements — deliberately NOT the same
// set as above.  `is_dynamic_quant_config` admits only bf16/f32, so an f16
// source with `dynamic_quant` set would reach the s8s8 kernel unquantized.
bool is_pipeline_dq_src_dtype(data_type_t dt) {
    return dt == data_type_t::f32 || dt == data_type_t::bf16;
}

// Weight-dtype family of one matmul half of ONE expert, or `none` when the
// pipeline does not serve it.  Called per active expert — see the compaction
// loop for why one representative slot is not enough.
enum class half_regime { none, bf16, da8w8 };

half_regime classify_pipeline_half(const matmul_params &p) {
    // Both regimes produce a FLOAT destination, and both need it: the
    // activation runs on the W13 destination, and under DA8W8 that same
    // buffer is what W2 re-quantizes its source from.
    if (!is_pipeline_act_dtype(p.dtypes.dst)) { return half_regime::none; }

    // Unquantized: one float dtype throughout the half.
    if (p.dtypes.wei == p.dtypes.dst) {
        if (p.dynamic_quant) { return half_regime::none; }
        if (p.dtypes.src != p.dtypes.dst) { return half_regime::none; }
        return half_regime::bf16;
    }

    if (p.dtypes.wei == data_type_t::s8) {
        // DA8W8 — s8 weights with s8 compute.  Symmetric only, matching the
        // ALGO 2 DQ-INT8 scope: `compute == s8` is the symmetric marker, and
        // an asymmetric u8 source would need its zero point carried through
        // the re-quant, which is not plumbed.
        if (p.dtypes.compute != data_type_t::s8) { return half_regime::none; }
        // The AOCL DLP s8s8 -> float kernel reads the weight scale from here;
        // without it the dispatch falls through to a path that produces zeros.
        if (p.quant_params.wei_scale.buff == nullptr) {
            return half_regime::none;
        }
        // Two accepted source forms, which is what lets one regime cover both
        // halves: s8 already (caller pre-quantized), or float +
        // `dynamic_quant`, which `execute_expert_slice` quantizes per expert.
        // W2 always takes the latter, so no separate re-quant stage is needed.
        if (p.dtypes.src == data_type_t::s8) {
            // s8 source with the flag still set is a half-applied pre-pass
            // state; decline rather than risk a double quantization.
            return p.dynamic_quant ? half_regime::none : half_regime::da8w8;
        }
        if (is_pipeline_dq_src_dtype(p.dtypes.src)) {
            if (!p.dynamic_quant) { return half_regime::none; }
            // Per-expert source quantization needs its scale metadata:
            // `reorder_quantization_wrapper` rejects an empty `dims` or a
            // `none` dtype, and `execute_expert_slice` cannot report that from
            // inside the OMP region — it logs, skips the GEMM and returns
            // void, leaving the STALE intermediate to be activated and fed to
            // W2 while the call still reports success.  The two-pass fails
            // closed on this input, so the gate has to screen it here, before
            // the region opens.  Matches the contract ALGO 2's pipeline gate
            // checks for the same reason.
            if (p.quant_params.src_scale.dims.empty()
                    || p.quant_params.src_scale.dt == data_type_t::none) {
                return half_regime::none;
            }
            // PER-TOKEN source scale only (`{M}` or `{M, 1}`).  A per-group
            // scale `{M, G>1}` keeps its contiguity guard in `reorder_direct`,
            // so W2's strided source would fall through to the generic reorder
            // — a third quantization route, neither the per-token one this
            // pipeline documents nor the grouped kernel the two-pass uses.
            // `setup_op2_dispatch_scratch` already routes per-group to the
            // two-pass; declining here makes that actually happen.
            const auto &sdims = p.quant_params.src_scale.dims;
            if (sdims.size() > 1 && sdims[sdims.size() - 1] > 1) { // per-group
                return half_regime::none;
            }
            return half_regime::da8w8;
        }
        return half_regime::none;
    }

    return half_regime::none;
}

} // namespace

expert_parallel_result try_expert_parallel_pipeline(
        const std::vector<char> &layout, const std::vector<bool> &transB,
        const std::vector<int> &M, const std::vector<const void *> &src,
        const std::vector<int> &lda, const std::vector<bool> &is_weights_const,
        grp_matmul_gated_act_t fused_act, data_type_t act_dtype,
        int num_threads, const pipeline_half_t &w13,
        const pipeline_half_t &w2) {

    // Cheapest gate first: the knob short-circuits before any data is read.
    if (!get_grp_matmul_algo5_vertical_fusion()) {
        return expert_parallel_result::declined;
    }

    // Length precondition.  This function is documented as self-gating, and
    // both the gate and the loop index ~28 per-expert vectors, so it cannot
    // rely on the current single caller happening to pre-validate them.
    const size_t n_ops = M.size();
    auto too_short = [n_ops](const pipeline_half_t &h) {
        return h.transA.size() < n_ops || h.N.size() < n_ops
                || h.K.size() < n_ops || h.alpha.size() < n_ops
                || h.weight.size() < n_ops || h.ldb.size() < n_ops
                || h.bias.size() < n_ops || h.beta.size() < n_ops
                || h.dst.size() < n_ops || h.ldc.size() < n_ops
                || h.params.size() < n_ops;
    };
    if (layout.size() < n_ops || transB.size() < n_ops || src.size() < n_ops
            || lda.size() < n_ops || is_weights_const.size() < n_ops
            || too_short(w13) || too_short(w2)) {
        return expert_parallel_result::declined;
    }

    // Whitelist the activation kinds rather than testing `!= none`, mirroring
    // ALGO 2's gate.  `apply_gated_act_inplace` ends its per-dtype switches in
    // `default: break`, so a kind added to the enum later would be silently
    // skipped and W2 would consume an unactivated intermediate at half width.
    switch (fused_act) {
        case grp_matmul_gated_act_t::none:
        case grp_matmul_gated_act_t::silu_and_mul:
        case grp_matmul_gated_act_t::gelu_and_mul:
        case grp_matmul_gated_act_t::swiglu_oai_mul: break;
        default: return expert_parallel_result::declined;
    }

    const size_t num_ops = M.size();
    if (num_ops == 0 || num_threads <= 0) {
        return expert_parallel_result::declined;
    }
    if (w13.params.empty() || w2.params.empty()) {
        return expert_parallel_result::declined;
    }

    // Compact the firing experts and check each one's fusability in the same
    // pass, so a decline costs one traversal.  See `active_experts_t`.
    active_experts_t act(num_ops);

    // Established by the first ACTIVE expert, then required of every other.
    // Not `params[0]`: ALGO 5 has no uniform-dtype clamp, so mixed dtypes are
    // reachable, and decode routinely leaves a leading expert inactive —
    // slot 0 may be a padded placeholder.
    half_regime regime = half_regime::none;

    for (size_t i = 0; i < num_ops; ++i) {
        if (M[i] <= 0) { continue; }

        const half_regime w13_regime = classify_pipeline_half(w13.params[i]);
        if (w13_regime == half_regime::none) {
            return expert_parallel_result::declined;
        }
        // Same regime for both halves.  Source dtypes may still differ within
        // DA8W8; the regime is about the weights and compute path.
        if (classify_pipeline_half(w2.params[i]) != w13_regime) {
            return expert_parallel_result::declined;
        }
        if (regime == half_regime::none) {
            regime = w13_regime;
        } else if (regime != w13_regime) {
            return expert_parallel_result::declined;
        }

        // Both halves must be PLAIN row-major, UNPACKED weights.
        //
        // This pipeline executes through `execute_expert_slice` ->
        // `matmul_execute` with `resolve_kernel()`, i.e. the regular AOCL /
        // BRGEMM path, and it has no caller-prepacked-weight consumption path
        // — exactly like ALGO 2, whose `check_m_tile_safe` therefore passes
        // `allow_prepacked_b = false`.  Only ALGO 3 + the custom kernel can
        // read a caller-prepacked CK VNNI weight.
        //
        // `mem_format_b == 'r'` is ambiguous: it marks caller-prepacked CK
        // VNNI (with `lowoha_algo == moe_custom_kernel`), AOCL-DLP-blocked,
        // and GGML unpack+reorder outputs alike.  None of their physical
        // layouts are what the regular GEMM here would read, so screen the
        // whole class rather than just the CK case, and screen it BEFORE
        // `prepack_for_algo_5` below, which would otherwise warm the regular
        // path over the caller's already-packed bytes.
        //
        // Declining sends the call to the two-pass, which reaches the
        // CK-only-or-fail guard in `group_matmul_run_parallel_dispatch`
        // (`error_prepacked_no_ck`) and fails closed there.  The non-fused
        // ALGO 5 executor already goes through that dispatcher, so this
        // restores the contract the fused branch was bypassing rather than
        // inventing a second one.
        //
        // ALGO 2's `reordered_pergroup_s8` carve-out has no analogue here:
        // it requires a per-group `{M, G>1}` source scale, which
        // `classify_pipeline_half` has already declined above (per-token
        // only).
        for (const auto *h : {&w13, &w2}) {
            if (h->params[i].mem_format_b != 'n') {
                return expert_parallel_result::declined;
            }
            if (h->params[i].packing.pack_format_b != 0) {
                return expert_parallel_result::declined;
            }
        }

        // W2 reads the buffer W13 wrote, so the dtypes must agree or it would
        // silently reinterpret the intermediate.
        if (w2.params[i].dtypes.src != w13.params[i].dtypes.dst) {
            return expert_parallel_result::declined;
        }

        // The activation is a silent no-op on a dtype it does not implement,
        // so a mismatch would hand W2 an unactivated intermediate.
        if (fused_act != grp_matmul_gated_act_t::none
                && act_dtype != w13.params[i].dtypes.dst) {
            return expert_parallel_result::declined;
        }

        // The intermediate is row-major at `w13.ldc[i]`; a transposed W2
        // source would reinterpret it.
        if (w2.transA[i]) { return expert_parallel_result::declined; }

        // The activation halves the width in place at the same base and
        // stride, so W2's K must be that post-activation width and the
        // pre-activation output must fit the row.  A mismatch means this is
        // not the FFN shape the pipeline assumes.
        const int inter_n = (fused_act == grp_matmul_gated_act_t::none)
                ? w13.N[i]
                : w13.N[i] / 2;
        if (fused_act != grp_matmul_gated_act_t::none && (w13.N[i] % 2) != 0) {
            return expert_parallel_result::declined;
        }
        if (w2.K[i] != inter_n) { return expert_parallel_result::declined; }
        if (w13.ldc[i] < w13.N[i]) { return expert_parallel_result::declined; }
        if (w13.dst[i] == nullptr || w2.dst[i] == nullptr) {
            return expert_parallel_result::declined;
        }

        act.add(i);
    }
    const int num_active = act.count;
    const int *active = act.idx;
    if (num_active == 0) {
        return expert_parallel_result::completed;
    } // nothing to do: both halves complete

    // Warm BOTH halves before the region: the two-pass got this implicitly by
    // running the dispatcher twice, so fusing must do it explicitly or the
    // first expert to reach each GEMM pays the reorder inside the region.
    const bool ck_on = get_grp_matmul_custom_kernel();
    group_matmul_prepack::prepack_for_algo_5(
            group_matmul_prepack::build_prepack_params(w13.weight, w13.K, w13.N,
                    w13.ldb, transB, is_weights_const, w13.params, M, ck_on,
                    num_threads,
                    algo3_decode_nr_align(
                            M, w13.N, w13.ldc, fused_act, w13.params),
                    fused_act, act_dtype,
                    /*transA=*/&w13.transA, /*alpha=*/&w13.alpha,
                    /*beta=*/&w13.beta));
    group_matmul_prepack::prepack_for_algo_5(
            group_matmul_prepack::build_prepack_params(w2.weight, w2.K, w2.N,
                    w2.ldb, transB, is_weights_const, w2.params, M, ck_on,
                    num_threads, /*nr_align=*/0,
                    /*fused_act=*/grp_matmul_gated_act_t::none,
                    /*act_dtype=*/data_type_t::none, /*transA=*/&w2.transA,
                    /*alpha=*/&w2.alpha, /*beta=*/&w2.beta));

    const matmul_algo_t algo = resolve_kernel();
    scoped_active_levels guard(1);

    const bool want_act = (fused_act != grp_matmul_gated_act_t::none);
    // Clamped for the same reason `parallel_per_expert` clamps.
    const int nthr = std::min(num_threads, num_active);

    static const bool s_log_plan = apilog_info_enabled();
    if (s_log_plan) {
        apilog_info("[GRP_MATMUL.PLAN] algo=5 expert_parallel_fused num_ops=",
                num_ops, " active=", num_active, " team_req=", nthr, "/",
                num_threads, " act=", (want_act ? "fused_pass" : "none"),
                " regime=", (regime == half_regime::da8w8 ? "da8w8" : "float"),
                " kernel=", static_cast<int>(algo));
    }

    // The per-expert source quantization can fail at RUNTIME — a pooled
    // allocation, which no gate can screen — and it leaves its destination
    // unwritten.  Activating that buffer and feeding it to W2 would publish a
    // stale intermediate, so the failure has to escape the region rather than
    // be logged and dropped.  An OMP `for` cannot carry a status out or break
    // early, so latch it and convert after the join.  Relaxed ordering is
    // enough: nothing is published through this flag, and the post-join read
    // is ordered by the implicit barrier.
    std::atomic<bool> slice_failed {false};

    // See the note on `parallel_per_expert`'s loop for the `if` clause.
#pragma omp parallel for num_threads(nthr) \
        schedule(dynamic, 1) if (num_active > 1)
    for (int a = 0; a < num_active; ++a) {
        // Drain the remaining iterations cheaply once any expert has failed:
        // the result is discarded, so the work would be wasted.
        if (slice_failed.load(std::memory_order_relaxed)) { continue; }

        const int i = active[a];

        if (execute_expert_slice_checked(layout[i], w13.transA[i], transB[i],
                    M[i], w13.N[i], w13.K[i], w13.alpha[i], src[i], lda[i],
                    w13.weight[i], w13.ldb[i], w13.bias[i], w13.beta[i],
                    w13.dst[i], w13.ldc[i], is_weights_const[i], 1,
                    w13.params[i],
                    resolve_expert_kernel(5, algo, w13.params[i]))
                != status_t::success) {
            slice_failed.store(true, std::memory_order_relaxed);
            continue;
        }

        if (want_act) {
            apply_gated_act_inplace(fused_act, w13.dst[i], 0, M[i], w13.N[i],
                    w13.ldc[i], act_dtype);
        }

        // W2 reads what this thread just wrote, so the intermediate never
        // leaves its cache.  `w13.ldc[i]` is the source stride: the activation
        // narrowed the width in place but left the row pitch alone.
        if (execute_expert_slice_checked(layout[i], w2.transA[i], transB[i],
                    M[i], w2.N[i], w2.K[i], w2.alpha[i], w13.dst[i], w13.ldc[i],
                    w2.weight[i], w2.ldb[i], w2.bias[i], w2.beta[i], w2.dst[i],
                    w2.ldc[i], is_weights_const[i], 1, w2.params[i],
                    resolve_expert_kernel(5, algo, w2.params[i]))
                != status_t::success) {
            slice_failed.store(true, std::memory_order_relaxed);
            continue;
        }
    }

    // TERMINAL, not a decline.  A two-pass rerun over this state is unsafe,
    // not merely wasteful: `reorder_quantization_wrapper` rebinds `params` as
    // a side effect (`dtypes.src` to the quantized dtype, the scale/zp buffs
    // into the worker's thread_local pool), and `pipeline_half_t::params` is a
    // reference into the CALLER's vector, so a rerun would reinterpret the
    // original float source as s8 and read scale pointers owned by another
    // thread.  Independently, already-activated `w13.dst` slices would be
    // activated a second time and a non-zero Op1 `beta` would accumulate onto
    // a destination that already holds a result.  See the
    // `expert_parallel_result` doc-block.
    if (slice_failed.load(std::memory_order_relaxed)) {
        log_error(
                "try_expert_parallel_pipeline: expert slice failed inside "
                "the parallel region; failing the call (a two-pass retry "
                "would see mutated params and potentially double-activated "
                "output)");
        return expert_parallel_result::failed;
    }
    return expert_parallel_result::completed;
}

// ── Selection policy ────────────────────────────────────────────────────
//
// The qualifier for an `AUTO_{DECODE,PROMPT}_ALGO=5` pin.  See the policy
// header for the measured rationale behind each term.

algo5_pin_verdict decide_algo5_pin(
        const grp_matmul_auto_phase_setting &phase_setting, bool is_decode,
        const std::vector<int> &M, const std::vector<matmul_params> &params,
        int active_ops, int num_threads, auto_algo_trace *trace) {
    algo5_pin_verdict v;

    const bool pin_is_5 = phase_setting.pins_generic_policy()
            && phase_setting.requested_algo == kGrpMatmulAlgoExpertParallel;
    v.decode_pin_is_5 = pin_is_5 && is_decode;
    v.prompt_pin_is_5 = pin_is_5 && !is_decode;

    if (v.prompt_pin_is_5 && get_grp_matmul_prompt_algo5_gate()) {
        v.low_occupancy = active_ops < num_threads;
        v.prompt_declined = v.low_occupancy;
    }
    if (v.decode_pin_is_5 && get_grp_matmul_decode_algo5_gate()) {
        // EVERY ACTIVE expert must carry s8 weights: ALGO 5 has no
        // uniform-dtype clamp, so probing one slot would hand the very bf16
        // experts this gate excludes to ALGO 5.  Inactive placeholders are
        // skipped; a missing `params` entry declines conservatively.
        bool all_active_wei_s8 = true;
        int active_seen = 0;
        for (size_t i = 0; i < M.size(); ++i) {
            if (M[i] <= 0) { continue; }
            ++active_seen;
            if (i >= params.size() || params[i].dtypes.wei != data_type_t::s8) {
                all_active_wei_s8 = false;
                break;
            }
        }
        v.wei_not_s8 = !(all_active_wei_s8 && active_seen > 0);
        v.low_occupancy = active_ops < num_threads;
        v.decode_declined = v.wei_not_s8 || v.low_occupancy;
    }

    if (trace != nullptr) {
        if (v.decode_pin_is_5 && !v.decode_declined) {
            trace->decode5_pin_honoured = true;
        }
        if (v.prompt_pin_is_5 && !v.prompt_declined) {
            trace->prompt5_pin_honoured = true;
        }
        if (v.decode_declined) { trace->decode5_pin_declined = true; }
        if (v.prompt_declined) { trace->prompt5_pin_declined = true; }
    }

    // Emitted where the decline is decided so every caller inherits it,
    // including the fused path, which bypasses the parallel dispatcher.
    if (v.decode_declined) {
        static std::atomic<bool> once {false};
        warn_once(once,
                "[GRP_MATMUL.ALGO WARN] "
                "ZENDNNL_GRP_MATMUL_AUTO_DECODE_ALGO=5 DECLINED and NOT in "
                "effect: ",
                (v.wei_not_s8 ? "weights are not s8 on every ACTIVE expert"
                              : ""),
                ((v.wei_not_s8 && v.low_occupancy) ? "; " : ""),
                (v.low_occupancy ? "active_ops < num_threads" : ""),
                " (active_ops=", active_ops, " num_threads=", num_threads,
                ").  ALGO 5 (parallel_per_expert) only beats ALGO 3 for INT8 "
                "decode at or above full-team occupancy; this call falls back "
                "to the normal decode policy, exactly as if the pin were "
                "unset.  Set ZENDNNL_GRP_MATMUL_DECODE_ALGO5_GATE=0 to honour "
                "the pin verbatim, or ZENDNNL_GRP_MATMUL_ALGO=5 to force ALGO "
                "5 for every call.  Logged once per process.");
    }
    if (v.prompt_declined) {
        static std::atomic<bool> once {false};
        warn_once(once,
                "[GRP_MATMUL.ALGO WARN] "
                "ZENDNNL_GRP_MATMUL_AUTO_PROMPT_ALGO=5 DECLINED and NOT in "
                "effect: active_ops < num_threads (active_ops=",
                active_ops, " num_threads=", num_threads,
                ").  ALGO 5 (parallel_per_expert) runs one expert per thread "
                "with no intra-expert split, so it cannot fill the team below "
                "full occupancy; this call falls back to ALGO 3 (N-tile), "
                "whose intra-expert N-split can — or to ALGO 1 when the shape "
                "is not N-tile-safe.  Set "
                "ZENDNNL_GRP_MATMUL_PROMPT_ALGO5_GATE=0 to honour the pin "
                "verbatim, or ZENDNNL_GRP_MATMUL_ALGO=5 to force ALGO 5 for "
                "every call.  Logged once per process.");
    }

    return v;
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
