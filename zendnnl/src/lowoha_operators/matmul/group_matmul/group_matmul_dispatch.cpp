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
/// This translation unit owns the ALGO selection and routing logic and
/// the ALGO implementations that are NOT large enough to warrant their
/// own translation unit:
///
///   * ALGO 1  (`sequential_experts`)     — serial over experts.
///                                          Kept here despite the file
///                                          name "dispatch" because it
///                                          is the universal safety
///                                          fallback every other ALGO
///                                          may route to under safety
///                                          clamps.
///   * ALGO 4  (`parallel_multilevel`)    — CCD-aware adaptive
///                                          scheduling.
///   * ALGO 5  (`parallel_per_expert`)    — per-expert parallel.
///   * ALGO 0  auto-select (`auto_select_algo`) + safety clamps.
///   * `group_matmul_run_parallel_dispatch`  — the dispatcher entry
///                                             point itself.
///
/// ALGO 2 (M-tile, `flat_m_tile` + `flat_m_tile_pipeline_bf16`) and
/// ALGO 3 (N-tile, `flat_n_tile`) live in their own folder-scoped
/// translation units (`m_tile/group_matmul_m_tile.cpp`,
/// `n_tile/group_matmul_n_tile.cpp`) and are called through forward
/// declarations re-included by `m_tile/group_matmul_m_tile.hpp` and
/// `n_tile/group_matmul_n_tile.hpp`.
///
/// Historical note: this file was named `group_matmul_parallel.cpp`
/// until the PR follow-up that renamed it to `group_matmul_dispatch.cpp`
/// (the prior name advertised "parallel" but the file's actual job is
/// dispatch + the small serial ALGOs).

#include <algorithm>
#include <atomic>
#include <climits>
#include <limits>
#include <string>
#include <vector>

#include <omp.h>

#include "group_matmul_parallel_common.hpp"
#include "lowoha_operators/matmul/backends/aocl/aocl_kernel.hpp"
#include "m_tile/group_matmul_m_tile.hpp" // flat_m_tile + M-tile env knobs
#include "n_tile/group_matmul_n_tile.hpp" // flat_n_tile + N-tile env knobs
#include "prepack/prepack.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace {

// ── ALGO=1: sequential — serial over experts ────────────────────────────

void sequential_experts(const std::vector<char> &layout,
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

    // Generic ahead-of-time weight pre-pack for ALGO 1.  Idempotent:
    // short-circuits when `ZENDNNL_GRP_MATMUL_PREPACK=0` or when this
    // thread already warmed the same fingerprint (per-thread cache
    // covers process-lifetime calls of the same model/layer).  Under
    // the uniform-eager semantic, PREPACK=ON warms the firing experts
    // (legacy callers, `total = active = M.size()` after
    // `build_prepack_params`) AND the full prepack-extras pool when
    // the framework opted into `total > active`.  The module owns its
    // own AOCL DLP backend gating via `resolve_kernel()`.
    //
    // `num_threads` is forwarded so `cross_warm` inside prepack.cpp can
    // compute `stable = aocl_stable_n_thr(num_threads, max_N)` and
    // prefill regime 2 (per-tile AOCL with nr_align=1) for the
    // upcoming ALGO 3 decode path when CUSTOM_KERNEL=0.  Without it,
    // that branch silently drops to a no-op and decode pays a one-time
    // first-call reorder cost.  `nr_align` is left at 0 because the
    // primary warm here is the full-weight key (which is nr_align-
    // independent); cross_warm uses its own internal nr_align for the
    // regime-2 path.
    group_matmul_prepack::prepack_for_algo_1(
            group_matmul_prepack::build_prepack_params(weight, K, N, ldb,
                    transB, is_weights_const, params, M,
                    get_grp_matmul_custom_kernel(), num_threads, /*nr_align=*/0,
                    fused_act, act_dtype,
                    /*transA=*/&transA, /*alpha=*/&alpha, /*beta=*/&beta));

    matmul_algo_t algo = resolve_kernel();

    for (size_t i = 0; i < num_ops; ++i) {
        // Inactive experts (M<=0) are padded placeholder slots that may carry
        // null src/dst/weight pointers (validate_group_matmul_direct_inputs
        // allows null for M==0 because dispatch is expected to short-circuit
        // empty rows); skip them so the slice/activation calls never
        // dereference null.
        if (M[i] <= 0) { continue; }
        execute_expert_slice(layout[i], transA[i], transB[i], M[i], N[i], K[i],
                alpha[i], src[i], lda[i], weight[i], ldb[i], bias[i], beta[i],
                dst[i], ldc[i], is_weights_const[i], num_threads, params[i],
                algo);
        // Fused activation: dst[i] is hot in L3 from the GEMM that just finished.
        if (fused_act != grp_matmul_gated_act_t::none) {
            apply_gated_act_inplace(fused_act, dst[i], 0, M[i], N[i], ldc[i],
                    act_dtype, num_threads);
        }
    }
    return;
}

// ── ALGO=4: multilevel — CCD-aware adaptive scheduling ──────────────────
//
// (A) Few experts, large M: multi-CCD per expert, all concurrent.
// (B) Few experts + small M, or many experts: round-based, 1 CCD each.
// Uses nested OMP (scoped_active_levels(2)).

void parallel_multilevel(const std::vector<char> &layout,
        const std::vector<bool> &transA, const std::vector<bool> &transB,
        const std::vector<int> &M, const std::vector<int> &N,
        const std::vector<int> &K, const std::vector<float> &alpha,
        const std::vector<const void *> &src, const std::vector<int> &lda,
        const std::vector<const void *> &weight, const std::vector<int> &ldb,
        const std::vector<const void *> &bias, const std::vector<float> &beta,
        const std::vector<void *> &dst, const std::vector<int> &ldc,
        const std::vector<bool> &is_weights_const,
        std::vector<matmul_params> &params, int num_threads,
        grp_matmul_gated_act_t fused_act, data_type_t act_dtype,
        const char **gemm_mode_out) {

    // The executor owns its gemm_mode and writes the concrete regime it ran
    // ("multilevel_concurrent" vs "multilevel_rounds") so the post-exec
    // [GRP_MATMUL.CALL] line reflects the real path.  No-op when nullptr.
    auto set_ml_mode = [&](const char *s) {
        if (gemm_mode_out != nullptr) { *gemm_mode_out = s; }
    };
    // Default to SKIP so a no-op early return (empty call / num_threads<=0)
    // reports exec_algo=0 rather than a real ALGO-4 run; the two regime
    // branches below overwrite it with the executed path.
    set_ml_mode("multilevel_skip");

    const int num_ops = static_cast<int>(M.size());
    if (num_ops == 0 || num_threads <= 0) { return; }

    // NOTE: the all-inactive (every M<=0) call is short-circuited to a "skip"
    // mode by the dispatcher (group_matmul_run_parallel_dispatch) before this
    // executor is ever entered, so no per-regime all-inactive guard is needed
    // here.  The per-slot M<=0 guards below still cover the mixed case.

    // Generic ahead-of-time weight pre-pack for ALGO 4.
    // See sequential_experts above for the contract; identical short-
    // circuits, only the scheduling-algo tag differs.  `num_threads`
    // is forwarded so cross_warm can prefill regime 2 for the upcoming
    // ALGO 3 decode path when CUSTOM_KERNEL=0 (see the comment on the
    // ALGO 1 call site for the full rationale).
    group_matmul_prepack::prepack_for_algo_4(
            group_matmul_prepack::build_prepack_params(weight, K, N, ldb,
                    transB, is_weights_const, params, M,
                    get_grp_matmul_custom_kernel(), num_threads, /*nr_align=*/0,
                    fused_act, act_dtype,
                    /*transA=*/&transA, /*alpha=*/&alpha, /*beta=*/&beta));

    matmul_algo_t algo = resolve_kernel();

    const int ccd_size = std::min(8, num_threads);
    // Ceiling to match flat_m_tile / flat_n_tile: partial last CCD counts as one.
    const int num_ccds = std::max(1, (num_threads + ccd_size - 1) / ccd_size);
    const int max_M = *std::max_element(M.begin(), M.end());

    if (num_ops <= num_ccds && max_M >= ccd_size) {
        // (A) Few experts, large M: multi-CCD per expert, all concurrent.
        set_ml_mode("multilevel_concurrent");
        int64_t total_M = 0;
        for (int i = 0; i < num_ops; ++i) {
            total_M += M[i];
        }
        if (total_M <= 0) { total_M = num_ops; }

        std::vector<int> ccds_per_op(num_ops, 1);
        int remaining = num_ccds - num_ops;
        if (remaining > 0) {
            for (int i = 0; i < num_ops; ++i) {
                int extra = static_cast<int>(
                        static_cast<int64_t>(remaining) * M[i] / total_M);
                ccds_per_op[i] += extra;
            }
            int used = 0;
            for (int i = 0; i < num_ops; ++i) {
                used += ccds_per_op[i];
            }
            for (int i = 0; used < num_ccds; ++i, ++used) {
                ccds_per_op[i % num_ops]++;
            }
        }
        std::vector<int> thr_per_op(num_ops);
        for (int i = 0; i < num_ops; ++i) {
            thr_per_op[i] = ccds_per_op[i] * ccd_size;
        }

        scoped_active_levels guard(2);
#pragma omp parallel num_threads(num_ops)
        {
            const int i = omp_get_thread_num();
            // Inactive experts (M<=0) are padded placeholder slots that may carry
            // null src/dst/weight pointers; skip them so the slice/activation
            // calls never dereference null (matches the M==0 guards in the
            // sequential / m-tile / n-tile executors).
            if (i < num_ops && M[i] > 0) {
                execute_expert_slice(layout[i], transA[i], transB[i], M[i],
                        N[i], K[i], alpha[i], src[i], lda[i], weight[i], ldb[i],
                        bias[i], beta[i], dst[i], ldc[i], is_weights_const[i],
                        thr_per_op[i], params[i], algo);
                if (fused_act != grp_matmul_gated_act_t::none) {
                    apply_gated_act_inplace(fused_act, dst[i], 0, M[i], N[i],
                            ldc[i], act_dtype);
                }
            }
        }
    } else {
        // (B) Round-based, 1 CCD per expert.
        set_ml_mode("multilevel_rounds");
        const int batch = std::min(num_ops, num_ccds);

        scoped_active_levels guard(2);
        for (int round_start = 0; round_start < num_ops; round_start += batch) {
            const int round_end = std::min(num_ops, round_start + batch);
            const int round_size = round_end - round_start;

#pragma omp parallel num_threads(round_size)
            {
                const int slot = omp_get_thread_num();
                // Inactive experts (M<=0) are padded placeholder slots that may carry
                // null src/dst/weight pointers; skip them so the slice/activation
                // calls never dereference null (matches the M==0 guards in the
                // sequential / m-tile / n-tile executors).
                if (slot < round_size && M[round_start + slot] > 0) {
                    const int e = round_start + slot;
                    execute_expert_slice(layout[e], transA[e], transB[e], M[e],
                            N[e], K[e], alpha[e], src[e], lda[e], weight[e],
                            ldb[e], bias[e], beta[e], dst[e], ldc[e],
                            is_weights_const[e], ccd_size, params[e], algo);
                    if (fused_act != grp_matmul_gated_act_t::none) {
                        apply_gated_act_inplace(fused_act, dst[e], 0, M[e],
                                N[e], ldc[e], act_dtype);
                    }
                }
            }
        }
    }
    return;
}

// ── ALGO=5: per_expert ─────────────────────────────────────────────────
// Parallel-for over experts; each expert is executed by a single thread (no
// intra-expert N-split).  This is NOT a 1:1 expert↔thread mapping: with
// `schedule(dynamic, 1)`, and whenever `num_ops > num_threads`, a thread
// processes several experts sequentially (~num_ops/num_threads each).
//
// `schedule(dynamic, 1)`: MoE routing is M-skewed (per-expert token counts
// range from 1 to many), so a static partition leaves threads that drew
// light experts idle while heavy-expert threads run long — worst when
// `num_ops > num_threads` and each thread must process several experts in
// sequence.  Dynamic scheduling with chunk 1 lets a thread grab the next
// unprocessed expert as soon as it finishes, balancing the skew.  Each
// iteration writes only its own expert's `dst[i]` slice, so there is no
// cross-iteration dependency that dynamic ordering could violate.

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

    // Generic ahead-of-time weight pre-pack for ALGO 5.  `num_threads`
    // is forwarded so cross_warm can prefill regime 2 for the upcoming
    // ALGO 3 decode path when CUSTOM_KERNEL=0 (see the comment on the
    // ALGO 1 call site for the full rationale).
    group_matmul_prepack::prepack_for_algo_5(
            group_matmul_prepack::build_prepack_params(weight, K, N, ldb,
                    transB, is_weights_const, params, M,
                    get_grp_matmul_custom_kernel(), num_threads, /*nr_align=*/0,
                    fused_act, act_dtype,
                    /*transA=*/&transA, /*alpha=*/&alpha, /*beta=*/&beta));

    matmul_algo_t algo = resolve_kernel();
    scoped_active_levels guard(1);

#pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1)
    for (size_t i = 0; i < num_ops; ++i) {
        // Inactive experts (M<=0) are padded placeholder slots that may carry
        // null src/dst/weight pointers; skip them so the slice/activation calls
        // never dereference null (matches the M==0 guards in the sequential /
        // m-tile / n-tile executors).
        if (M[i] <= 0) { continue; }
        execute_expert_slice(layout[i], transA[i], transB[i], M[i], N[i], K[i],
                alpha[i], src[i], lda[i], weight[i], ldb[i], bias[i], beta[i],
                dst[i], ldc[i], is_weights_const[i], 1, params[i], algo);
        if (fused_act != grp_matmul_gated_act_t::none) {
            apply_gated_act_inplace(
                    fused_act, dst[i], 0, M[i], N[i], ldc[i], act_dtype);
        }
    }
    return;
}

// M-tile (ALGO 2) safety predicate hoisted to
// `group_matmul_parallel_common.hpp` as `check_m_tile_safe` (inline).
// Both the legacy dispatcher in this TU and the MoE vertical-fusion
// dispatcher fork in `group_matmul_fused_moe.cpp` use the same
// predicate, so it lives next to `op2_k_for_act` in the common
// header.  See that header's doc-block for the row-locality
// rationale (dynamic-quant, postop softmax / pooling, etc.) and the
// `M[i] == 1` decode-class special case.

// N-tile (ALGO 3) slices columns of B, so the executor must be able
// to re-anchor any N-indexed metadata (weight scales / zero-points,
// binary post-op tensors, packed-B tables) onto each thread's
// `[col_start, col_start + n_tile)` window before the kernel call.
// This helper enumerates which configurations the column-slicer in
// `group_matmul_n_tile.cpp::do_tile` actually supports today and
// rejects the rest.  Buffer-free element-wise post-ops (gelu, relu,
// swish, …) always pass.
//
// SCOPE NOTE — what `n_tile_safe = false` actually does to a
// quantised workload's routing.
//
//   This helper only computes `n_tile_safe`; it does NOT make the
//   final ALGO decision.  `select_grp_matmul_algo` consults
//   `n_tile_safe` ONLY at the ALGO 3 decision points:
//
//     * Forced `env_algo == 3`: rejected → falls back to ALGO 1.
//     * Auto-select (`env_algo == 0`) on a shape that would
//       otherwise pick ALGO 3: redirected to ALGO 1.  The current
//       auto-select rule (see `auto_select_algo` below) picks
//       ALGO 3 in two cases — `num_ops >= num_threads` (many experts)
//       and the M-driven decode arrow (`max_M <= kDecodeMaxM`).
//       Both honour `n_tile_safe`; on quantised inputs that fall
//       outside the supported sub-set below, both arrows collapse
//       to ALGO 1.  (Rule 0's capacity carve-out routes
//       `num_ops > kNTilePlanMaxExperts` to ALGO 1 before either
//       ALGO 3 arrow can fire, so it is unaffected by n_tile_safe.)
//
//   Other ALGOs are unaffected by this helper:
//     * Forced `env_algo ∈ {1, 2, 4, 5}` is respected as-is
//       (m_tile_safe is checked separately for ALGO 2; ALGO 1/4/5
//       have no tile-safety gate).
//
//   What N-tile accepts today is intentionally one single shape:
//   the per-token dynamic-INT8 deployment.  ALL of the following
//   must hold simultaneously for ALGO 3 to be selected on a
//   quantised call:
//
//     * `params[i].dynamic_quant == true` — runtime BF16/F32→S8
//       source reorder is required.  Static src quant (where the
//       caller pre-quantised src and passed `src_scale.buff`)
//       falls back to ALGO 1.  This is a deliberate scope
//       restriction — static src + per-channel wei works
//       structurally, but the current deployment target is
//       dynamic-INT8 and the static path stays on ALGO 1 until
//       there's a reason to widen it.
//
//     * src_scale dims `{M[i], 1}` — per-token granularity (one
//       scale per row, scalar across K).  `buff` is null on entry
//       (the pre-OMP hoist loop in `flat_n_tile` allocates the
//       internal scale buffer and the wrapper writes the computed
//       per-row scales into it); the caller need only populate
//       `src_scale.dims` and `src_scale.dt`.  The single-row
//       decode case `{1, 1}` is accepted when `M[i] == 1` — the
//       hoist runs the source reorder once for that expert's one
//       row, no parallelism splitting is needed and N-tile
//       threads share the resulting scalar scale read-only.
//       Other row-local granularities such as `{M[i], G}`
//       per-group on K — which `check_m_tile_safe` would accept
//       — are rejected here.
//
//     * wei_scale dims `{N}` or `{1, N}` with `buff != nullptr`
//       — per-channel granularity, statically quantised by the
//       caller.  Per-tensor (`{}`, `{1}`) and per-group `{G, N}`
//       wei are both rejected.  Sliced per-thread by
//       `offset_quant_by_col` (advances `buff` by
//       `col_start × elem_size`; rewrites the trailing dim to
//       `n_tile`); the AOCL DLP / native int8 kernels detect
//       per-channel via `qsize == N` and index the sliced buffer
//       with `scale[col]` for `col ∈ [0, n_tile)`.
//
//     * Optional SOURCE asymmetry only: `src_zp` (if `buff`
//       non-null) must be `{M[i], 1}` per-token.  WEIGHT zero-points
//       are NOT supported — a non-null `wei_zp` rejects ALGO 3
//       outright (the CK microkernel and AOCL sym-quant fallback are
//       both symmetric-weight), so the call falls back to ALGO 1.
//
//   The `params[i].dynamic_quant` flag controls the SOURCE side
//   ONLY.  The WEIGHT side is ALWAYS statically quantised — the
//   caller pre-quantises the weights offline and hands the
//   library a non-null `wei_scale.buff`.  There is no "dynamic
//   weight quant" concept in the API, and the AOCL DLP / native
//   int8 kernels don't support one either (runtime weight
//   reorder would have to fire on every call and re-key the
//   weight-reorder cache against per-call scale data).
//
//   End-to-end this is the dynamic INT8 per-token + per-channel
//   wei case.  `flat_n_tile`'s pre-OMP hoist loop runs the
//   SOURCE-side reorder ONCE per expert and stashes the resulting
//   S8 src + scale buffer in a `HoistedSrcQuant` slot; per-tile
//   threads then read the shared S8 src + the column-sliced wei
//   scale.  Without the hoist, the source-side reorder inside
//   `execute_expert_slice` would race on the caller's scale
//   buffer and duplicate the (M, K) work `num_threads` times per
//   call.
//
//   What N-tile rejects (everything outside the single accepted
//   shape):
//
//     A. Static source quantisation with a non-S8 source, or without
//        per-token src_scale.  The grouped dynamic-quant path produces
//        S8 src + `{M[i], 1}` src_scale and is accepted.
//
//     B. Per-tensor weight or source scale (`{}`, `{1}`, or any
//        product-1 shape).
//
//     C. Per-group weight scale `{G, N}` with `G > 1`.  The
//        column slice is `G` non-contiguous strips of length
//        `n_tile` in the original buffer; supporting it would
//        require a per-thread `G × n_tile` repack scratch that
//        is intentionally absent in this scope.
//
//     D. Per-group source scale `{M[i], G}` with `G > 1`
//        (per-group on K).  Mechanically safe under N-tile column
//        slicing (K is N-independent) but excluded from the
//        current scope.
//
//     E. Pure WOQ S4 / U4 / S8 (caller provides wei_scale but no
//        src_scale, and `dynamic_quant == false`).  Stays on
//        ALGO 1.
//
//     F. Binary post-op tensors with non-null `buff`
//        (`binary_add` / `binary_mul`) — these can have N-indexed
//        layouts (`{N}`, `{1, N}`, `{M, N}`) that need the same
//        column-slice treatment.  The slicer is not yet wired
//        for post-ops; reject for now.
//
//   The `check_m_tile_safe` precondition still applies — it
//   gates dynamic_quant to row-local granularities
//   (`src_scale.dims[0] == M[i]`).  With the per-token-only src
//   gate below, the only granularity that passes BOTH checks is
//   `{M[i], 1}` (including the single-row `M[i] == 1` decode
//   case, where `{1, 1}` is the per-token shape for a one-token
//   expert).
//
//   The custom microkernel family covers two compute regimes:
//
//     * BF16 — bf16×bf16→bf16, no quant.  Refuses every quantised
//       combo at `prepare_for_call`.
//     * DQ-INT8 — s8×s8→bf16 (symmetric) or u8×s8→bf16 (asymmetric);
//       per-token src scale + optional src_zp, per-channel wei scale
//       (weight zero-points are NOT supported on the N-tile DQ-INT8
//       path — a non-null wei_zp rejects ALGO 3, see below), all four
//       gated activations.
//
//   When `dynamic_quant=true` with `src=bf16, wei=s8, dst=bf16` and
//   the shape passes `plan_pack_nr_int8(rep_K, rep_N) ∈ {32, 64}`,
//   the call routes through the DQ-INT8 custom microkernel.  Calls
//   that fall outside both regimes (e.g., static src quant, S4/U4
//   WOQ, per-group on K) fall back to AOCL DLP int8 via the
//   `s8s8s32obf16_sym_quant` reorder cache as before.
//
// PRECONDITION: the caller has already run `check_m_tile_safe` and
// confirmed it returned true.  This helper intentionally does NOT
// re-run those checks — the orchestrator `select_grp_matmul_algo`
// always calls M-tile first and only invokes this when m_tile_safe is
// true, so a second pass would just be duplicated work.
static bool check_n_tile_extra(const std::vector<int> &M,
        const std::vector<matmul_params> &params, int num_ops) {
    // Per-channel weight side: dims must be exactly `{N}` (rank-1) or
    // `{1, N}` (rank-2 with broadcast outer dim).  `buff` must be
    // non-null (wei is always statically quantised by the caller).
    // The column slice is a contiguous `n_tile`-long sub-array —
    // handled by `offset_quant_by_col` in `do_tile`.
    auto is_per_channel_wei =
            [](const matmul_quantization_params_t::matmul_quant_t &q) -> bool {
        if (q.buff == nullptr) { return false; }
        if (q.dims.size() == 1 && q.dims[0] > 1) { return true; }
        if (q.dims.size() == 2 && q.dims[0] == 1 && q.dims[1] > 1) {
            return true;
        }
        return false;
    };

    // Per-token source side: dims must be exactly `{M[i], 1}`
    // (rank-2, first dim equals this expert's row count, scalar
    // across K).  The `M[i]` match is the key row-locality signal —
    // it excludes per-tensor / per-column / per-channel-on-src
    // layouts where the first dim is 1 (or empty) while accepting
    // the single-row decode case `M[i] == 1` with dims `{1, 1}`.
    //
    // Dynamic-quant input reaches this gate with `buff == nullptr` and
    // is hoisted by flat_n_tile. Grouped dynamic-quant reaches this gate
    // after pre-quantizing to S8, so `buff != nullptr` and the same
    // per-token dims describe the ready-to-use scale buffer.
    auto is_per_token_dyn_src
            = [](const matmul_quantization_params_t::matmul_quant_t &q,
                      int M_expert) -> bool {
        return q.dims.size() == 2 && q.dims[0] == static_cast<int64_t>(M_expert)
                && q.dims[1] == 1;
    };

    // Per-group source side: dims `{M[i], G}` with G > 1 (one scale per
    // K-group, row-local).  Pairs with a `{G, N}` per-group weight scale.
    // The source scale is N-independent, so N-tile column slicing leaves it
    // whole; `do_tile` slices the WEIGHT scale to `{G, n_tile}`.  Reaches
    // this gate either as dynamic (`buff == nullptr`, hoisted by flat_n_tile)
    // or grouped-pre-quantized S8 (`buff != nullptr`).
    auto is_per_group_src
            = [](const matmul_quantization_params_t::matmul_quant_t &q,
                      int M_expert) -> bool {
        return q.dims.size() == 2 && q.dims[0] == static_cast<int64_t>(M_expert)
                && q.dims[1] > 1;
    };

    // Per-group weight side: dims `{G, N}` with G > 1 and N > 1, non-null
    // buff (statically quantised by the caller).  do_tile column-slices the
    // {G, N} scale into a contiguous {G, n_tile} per-thread scratch for the
    // AOCL sym-quant GEMM.
    auto is_per_group_wei =
            [](const matmul_quantization_params_t::matmul_quant_t &q) -> bool {
        return q.buff != nullptr && q.dims.size() == 2 && q.dims[0] > 1
                && q.dims[1] > 1;
    };

    // Same active-range constraint as `check_m_tile_safe` above —
    // tail slots carry framework prepack metadata, not real per-call
    // state, and would falsely flip n-tile-safe to false.
    for (int i = 0; i < num_ops; ++i) {
        // Inactive experts (M==0) do no compute.  The grouped / fallback DQ
        // pre-pass clears their `dynamic_quant` and leaves the original bf16
        // src (no per-token src_scale), so evaluating them here would hit the
        // `!dynamic_quant && !grouped_s8_src` source-side reject below and
        // veto ALGO 3 for the WHOLE call — even though every ACTIVE expert is
        // a valid grouped-s8 / dynamic-INT8 shape.  Skip them (matches the
        // first-active reference in `check_m_tile_safe`).
        if (M[i] == 0) { continue; }
        const auto &qp = params[i].quant_params;

        // Detect any quant intent on this expert.  If every quant field
        // is empty AND `dynamic_quant` is false, this is a pure
        // non-quantised call and there's nothing to gate — ALGO 3 is
        // free to run.
        const bool any_quant = qp.wei_scale.buff != nullptr
                || qp.wei_zp.buff != nullptr || qp.src_scale.buff != nullptr
                || qp.src_zp.buff != nullptr || params[i].dynamic_quant;

        if (any_quant) {
            // W4A8 (s4 weight): accepted on the N-tile path when is_w4a8_config
            // passes AND the shape is a valid per-group layout that flat_n_tile
            // can column-slice.  flat_n_tile expands s4→s8 into the W4A8 LRU
            // (no caller-param mutation) and rewrites tile_params to flow
            // through the sym-quant GEMM path.
            // u4 remains rejected (no symmetric W4A8 support).
            if (params[i].dtypes.wei == data_type_t::u4) { return false; }
            if (is_w4a8_config(params[i])) {
#if !ZENDNNL_DEPENDS_AOCLDLP
                // W4A8 N-tile needs AOCL-DLP (s4→s8 plain cache, per-tile
                // sym-quant reorder, broadcast_w4a8_src_scale).  Without AOCL
                // the ALGO-1 path falls through to the reference W4A8 kernel.
                return false;
#else
                // N-tile-specific extra gates beyond is_w4a8_config:
                // plain row-major weight required for column slicing.
                if (params[i].mem_format_b != 'n') { return false; }
#endif
                const bool w4a8_src_ok
                        = is_per_token_dyn_src(qp.src_scale, M[i])
                        || is_per_group_src(qp.src_scale, M[i]);
                const bool w4a8_wei_ok = is_per_group_wei(qp.wei_scale);
                if (!w4a8_src_ok || !w4a8_wei_ok) { return false; }
                continue;
            }

            // Source side: accept either (a) dynamic BF16/F32 input that
            // flat_n_tile will hoist, or (b) already grouped-quantized S8
            // input with a ready per-token source scale buffer.
            const bool grouped_s8_src = !params[i].dynamic_quant
                    && params[i].dtypes.src == data_type_t::s8
                    && qp.src_scale.buff != nullptr;
            if (!params[i].dynamic_quant && !grouped_s8_src) { return false; }

            // Source dims: `{M[i], 1}` per-token (incl. the `M[i] == 1`
            // decode case `{1, 1}`) OR `{M[i], G}` per-group (G > 1).
            // For grouped_s8_src the scale buffer must be non-null (checked
            // above). For dynamic input, nullness is a hoist-allocation
            // contract, not a per-token-scope contract.
            const bool src_per_token = is_per_token_dyn_src(qp.src_scale, M[i]);
            const bool src_per_group = is_per_group_src(qp.src_scale, M[i]);
            if (!src_per_token && !src_per_group) { return false; }
            // Per-group is symmetric-only on the N-tile path (the AOCL sym-quant
            // kernel + the per-group source reorder are both symmetric); a
            // non-null src_zp on a per-group call is out of scope → ALGO 1.
            if (src_per_group && qp.src_zp.buff != nullptr) { return false; }
            if (grouped_s8_src && qp.src_zp.buff != nullptr) { return false; }
            if (src_per_token && qp.src_zp.buff != nullptr
                    && !is_per_token_dyn_src(qp.src_zp, M[i])) {
                return false;
            }

            // Weight side: per-channel `{N}` / `{1, N}` (per-token src) OR
            // per-group `{G, N}` (per-group src).  Per-tensor wei is rejected.
            const bool wei_per_channel = is_per_channel_wei(qp.wei_scale);
            const bool wei_per_group = is_per_group_wei(qp.wei_scale);
            if (!wei_per_channel && !wei_per_group) { return false; }
            // Granularity pairing rules for the remaining non-W4A8 quant paths:
            //   per-group src + per-group wei   -> accepted
            //   per-token src + per-channel wei -> accepted
            //   per-token src + per-group wei   -> accepted; do_tile repacks
            //      wei_scale to {G, n_tile}, while source scale is N-independent
            //   per-group src + per-channel wei -> rejected (incoherent)
            if (src_per_group && !wei_per_group) { return false; }
            // Per-group weight is column-sliced by a `{G, n_tile}` repack in
            // do_tile, which needs a plain row-major s8 weight.  A pre-reordered
            // weight (`mem_format_b == 'r'`, e.g. an unpacked GGML weight) is not
            // column-sliceable, so it must take the full-N ALGO 1 path instead.
            if (wei_per_group && params[i].mem_format_b != 'n') {
                return false;
            }
            // Weight zero-point is NOT supported anywhere on the N-tile
            // DQ-INT8 path: the CK microkernel assumes symmetric weights
            // (its compensation row only folds the src +128 / src_zp bias),
            // and the AOCL sym-quant fallback is likewise symmetric.  A
            // non-null wei_zp must therefore reject ALGO 3 entirely so the
            // call falls back to ALGO 1 (general AOCL DLP) instead of
            // silently dropping the weight zero-point.
            if (qp.wei_zp.buff != nullptr) { return false; }
        }

        // Buffer-bearing post-ops (binary_add / binary_mul) may carry
        // N-indexed layouts (`{N}`, `{1, N}`, `{M, N}`) that need the
        // same column-slice treatment as wei_scale.  The slicer is not
        // yet wired for post-ops; keep them rejected.  Buffer-free
        // elementwise post-ops (gelu, relu, swish, …) have null `buff`
        // and pass through.
        for (const auto &po : params[i].postop_) {
            if (po.buff != nullptr) { return false; }
        }
    }
    return true;
}

// Auto-select (ALGO 0) heuristic — used when the caller leaves
// ZENDNNL_GRP_MATMUL_ALGO unset.  Picks between {ALGO 1, ALGO 2,
// ALGO 3} via the legacy 3-rule cascade, OR pins to a specific
// ALGO per phase via the AUTO_PROMPT_ALGO / AUTO_DECODE_ALGO envs
// (defaults: PROMPT=2 (flat_m_tile + multi-tier hybrid + wide-N
// fallback) / DECODE=3 (N-tile + CK) — the out-of-the-box auto
// policy; set the env to `0` for the legacy cascade, or `1` to
// pin the legacy sequential_experts prompt path).
//
// AUTO never emits ALGO 4 or 5 of its own accord.  ALGO 4/5 are reached only by
// an explicit request the selector honours: a global `ZENDNNL_GRP_MATMUL_ALGO=
// {4,5}` force, or an `AUTO_{PROMPT,DECODE}_ALGO={4,5}` phase-env pin (Rule 1).
// The one refinement is Rule 0.6a, which does NOT invent an ALGO-5 pick — it
// QUALIFIES an operator's decode `AUTO_DECODE_ALGO=5` pin, honouring it only for
// INT8 (s8) saturated-team decode and declining it (→ decode default) otherwise;
// with no pin set the invariant is exact.
//
// Decision precedence (tightest first):
//
//   0. STRUCTURAL — num_ops > kNTilePlanMaxExperts (=256) → ALGO 1
//      Capacity carve-out: beyond `GroupNTilePlan::kMaxExperts` the
//      N-tile planner's R3 gate falls back to its Sequential strategy,
//      so ALGO 3 would be no better than ALGO 1.  This is the one site
//      where the no-4-no-5 invariant costs throughput — ALGO 5's
//      per-expert wave schedule was the only PARALLEL option past that
//      ceiling; `ZENDNNL_GRP_MATMUL_ALGO=5` recovers it.  Phase env
//      cannot override this — the R3 gate is structural.
//
//   0.6. DECODE, active_ops > num_threads → ALGO 3 (N-tile).
//      `active_ops` counts the experts that actually fire (`M[i] > 0`),
//      not the padded slot count.  ALGO 3's single round is infeasible
//      here (it needs one thread per active expert) but DecodeDynamic
//      has no such ceiling, so the regime stays on ALGO 3 and the
//      planner picks the strategy.  Yields to an explicit
//      `AUTO_DECODE_ALGO` pin; prompt is never routed by this rule.
//
//   1. PHASE ENV — `max_M ≤ kDecodeMaxM` (decode) →
//                  `ZENDNNL_GRP_MATMUL_AUTO_DECODE_ALGO` (default 3)
//                  `max_M >  kDecodeMaxM` (prompt) →
//                  `ZENDNNL_GRP_MATMUL_AUTO_PROMPT_ALGO` (default 2)
//      When the active phase env is non-zero (the default cases),
//      that ALGO is returned directly with the same m_tile_safe /
//      n_tile_safe clamps the global ALGO env path applies in
//      `select_grp_matmul_algo`.  The defaults give an out-of-the-
//      box auto policy: ALGO 2 (flat_m_tile + multi-tier hybrid +
//      wide-N fallback) for prompt — the M-tile multi-tier targets
//      skewed-M prompt shapes, while the wide-N fallback keeps
//      few-expert light frames on the legacy ALGO 1 path; safety
//      clamp falls back to ALGO 1 when `!m_tile_safe` — and ALGO 3
//      (N-tile rounds + CK) for decode.  Set `AUTO_PROMPT_ALGO=1` to
//      restore the legacy sequential_experts prompt path, or the env
//      to `0` for the legacy 3-rule cascade.
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
        bool m_tile_safe, bool n_tile_safe, auto_algo_trace *trace) {
    (void)N; // Kept in the signature for symmetry with the M-tile / N-tile
    (void)K; // safety helpers and to ease future shape/dtype heuristic work.

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
    if (num_threads <= 1 || num_ops == 0) {
        return pick(1, "auto_single_thread", 1);
    }

    // INVARIANT — AUTO never returns ALGO 4 or 5 of its own accord; every rule
    // that once answered 5 answers `n_tile_safe ? 3 : 1`.  Rule 0.6a below does
    // NOT break this: it never SELECTS 5, it only QUALIFIES an operator's
    // explicit decode `AUTO_DECODE_ALGO=5` pin (honour for INT8 saturated-team
    // decode, decline → decode default otherwise).  Keep new rules inside the
    // no-4-no-5 set.  The invariant constrains the HEURISTICS, not the operator:
    // an explicit phase pin or global force of 4/5 is honoured.
    //
    // Rule 0 — STRUCTURAL capacity carve-out, placed before the phase env so
    // it catches every shape that would otherwise reach the N-tile planner's
    // R3 Sequential fallback.  See the rule table above for why it is ALGO 1.
    if (num_ops > kNTilePlanMaxExperts) {
        return pick(1, "auto_rule0_capacity", 1);
    }

    const int max_M = *std::max_element(M.begin(), M.end());
    const bool is_decode = (max_M <= kDecodeMaxM);

    // ACTIVE-COMPUTE expert count = |{ i : M[i] > 0 }|, NOT M.size() and NOT the
    // framework `total_matmul` pool: a legacy caller may pass a padded vector
    // with `M[i]==0` placeholders, and only the M[i]>0 experts consume a thread.
    // Read by the decode `=5` pin's occupancy term (Rule 0.6a) and by Rule 0.6.
    const int active_ops = static_cast<int>(
            std::count_if(M.begin(), M.end(), [](int m) { return m > 0; }));

    // TOTAL expert count for Rule 0.5's ALGO 2 preference.  `total_matmul` is
    // only meaningful under the framework opt-in (`active_matmul > 0`); a legacy
    // caller may leave it stale, so read it only then and only when it exceeds
    // the active count (padded layout).
    int total_experts = num_ops;
    if (!params.empty() && params[0].active_matmul > 0
            && params[0].total_matmul > static_cast<uint32_t>(total_experts)) {
        total_experts = static_cast<int>(params[0].total_matmul);
    }

    // Rule 0.6a — QUALIFY the decode `AUTO_DECODE_ALGO=5` pin.
    // ALGO 5 (parallel_per_expert: one active expert per thread, no intra-expert
    // N-split) was measured to BEAT the ALGO 3 decode default on INT8 MoE decode
    // (e.g. Qwen3 / Qwen3.6 +3-14% at 32 threads) but to REGRESS on bf16 MoE and
    // on under-occupied frames — one-expert-per-thread then runs at the speed of
    // the heaviest expert and cannot fill the team.  So AUTO never emits ALGO 5
    // on its own; an operator opts in with the explicit
    // `ZENDNNL_GRP_MATMUL_AUTO_DECODE_ALGO=5` pin, and this gate HONOURS that pin
    // only where the win holds — ALL of:
    //   * decode phase (`is_decode`);
    //   * every ACTIVE expert carries s8 weights (bf16 loses across the board);
    //   * `active_ops >= num_threads` (team saturated one-expert-per-thread;
    //     below that ALGO 3's intra-expert N-tile keeps the machine busier).
    // The pin is deliberately NOT fenced to a model or shape: it is a per-
    // deployment opt-in, so an operator sets it only where ALGO 5 helps (the
    // measured INT8 Qwen3/3.6 case) and any other INT8 saturated decode the
    // deployment runs is honoured on the same terms.  When the pin is 5 but a
    // term fails it is DECLINED: `phase_env_pinned` is cleared just below so the
    // normal decode rules (0.45 / 0.5 / 0.6) run, and Rule 1 substitutes the
    // decode DEFAULT for the pinned 5 — the call behaves EXACTLY as if
    // `AUTO_DECODE_ALGO` were unset.  A decline is surfaced (trace + a
    // once-per-process WARN naming the failing term) rather than silently doing
    // nothing.  `ZENDNNL_GRP_MATMUL_DECODE_ALGO5_GATE=0` honours the pin verbatim
    // (pre-gate semantics).  ALGO 5 has no tiling precondition, so an honoured
    // pin needs no m_tile_safe / n_tile_safe clamp.
    const bool decode_pin_is_5 = is_decode
            && grp_matmul_auto_decode_algo_is_set()
            && get_grp_matmul_auto_decode_algo() == 5;
    bool decode_algo5_pin_declined = false;
    bool decode5_wei_not_s8 = false;
    bool decode5_low_occupancy = false;
    if (decode_pin_is_5 && get_grp_matmul_decode_algo5_gate()) {
        // EVERY ACTIVE expert must carry s8 weights — probing one representative
        // slot is not enough.  ALGO 5 imposes no uniform-dtype clamp (only the
        // tiled ALGO 2 / ALGO 3 paths reject a per-expert mismatch), so a group
        // whose first active expert is s8 and whose later active experts are
        // bf16 is reachable through the public API, and probing only the first
        // slot would hand the very bf16 experts this gate excludes to ALGO 5.
        // Inactive `M[i] <= 0` placeholders are skipped (they never reach a
        // kernel); an active slot with no `params` entry declines conservatively.
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
        decode5_wei_not_s8 = !(all_active_wei_s8 && active_seen > 0);
        decode5_low_occupancy = active_ops < num_threads;
        decode_algo5_pin_declined = decode5_wei_not_s8 || decode5_low_occupancy;
    }
    if (decode_pin_is_5 && !decode_algo5_pin_declined && trace != nullptr) {
        trace->decode5_pin_honoured = true;
    }
    if (decode_algo5_pin_declined) {
        if (trace != nullptr) { trace->decode5_pin_declined = true; }
        // WARNING level (the default api-log level), ONCE per process: a
        // declined pin is a persistent misconfiguration and decode re-runs this
        // selector on every token, so a per-call warning would flood the log.
        // Named CAUSE + escape hatch, not just the fact; the per-call detail
        // stays on the trace and the info-level `[GRP_MATMUL.ALGO]` line.
        // Emitted HERE, where the decline is decided, so every caller inherits
        // it — including the fused path, which bypasses the parallel dispatcher.
        static const bool s_warn = apilog_warning_enabled();
        if (s_warn) {
            static std::atomic<bool> s_warned {false};
            if (!s_warned.exchange(true, std::memory_order_relaxed)) {
                apilog_warning(
                        "[GRP_MATMUL.ALGO WARN] "
                        "ZENDNNL_GRP_MATMUL_AUTO_DECODE_ALGO=5 DECLINED and "
                        "NOT "
                        "in effect: ",
                        (decode5_wei_not_s8 ? "weights are not s8 on every "
                                              "ACTIVE expert"
                                            : ""),
                        ((decode5_wei_not_s8 && decode5_low_occupancy) ? "; "
                                                                       : ""),
                        (decode5_low_occupancy ? "active_ops < num_threads"
                                               : ""),
                        " (active_ops=", active_ops,
                        " num_threads=", num_threads,
                        ").  ALGO 5 (parallel_per_expert) only beats ALGO 3 "
                        "for "
                        "INT8 decode at or above full-team occupancy; this "
                        "call "
                        "falls back to the normal decode policy, exactly as if "
                        "the pin were unset.  Set "
                        "ZENDNNL_GRP_MATMUL_DECODE_ALGO5_GATE=0 to honour the "
                        "pin "
                        "verbatim, or ZENDNNL_GRP_MATMUL_ALGO=5 to force ALGO "
                        "5 "
                        "for every call.  Logged once per process.");
            }
        }
    }
    // A gate-declined decode `=5` pin is treated as UNSET here, which re-enables
    // the decode policy rules (0.45 / 0.5 / 0.6) below and, with Rule 1's
    // default substitution, lands the call exactly where an unset pin would.
    const bool phase_env_pinned = !decode_algo5_pin_declined
            && (is_decode ? grp_matmul_auto_decode_algo_is_set()
                          : grp_matmul_auto_prompt_algo_is_set());

    // Rule 0.45 — SINGLE DENSE EXPERT DECODE → ALGO 3 (N-tile).
    // A lone expert (`num_ops == 1`) in decode would otherwise be diverted by
    // Rule 0.5 to `kWideN → ALGO 1`, since a single expert can never fill the
    // team via M-tiling.  That path never reaches the N-tile planner where the
    // single-expert optimisations live (adaptive tiling, ragged-N fallback,
    // K-blocking), so route it to ALGO 3, mirroring the multi-expert decode
    // default (Rule 2c).  `is_dense_ffn_decode` is the scope predicate shared
    // with the adaptive N-tile sizer and the auto-K-blocking gate, keeping MoE
    // and prompt untouched.  Honours an explicit decode pin and the
    // n_tile_safe clamp; env-gated (default ON) for A/B.  Default and gates
    // are pinned by the `SingleDenseExpertDecode*` tests in test_algos.cpp.
    if (is_dense_ffn_decode(num_ops, max_M) && !phase_env_pinned && n_tile_safe
            && get_grp_matmul_dense_decode_ntile()) {
        return pick(3, "auto_rule045_single_dense_decode", 3);
    }

    // DECODE-ONLY: prompt few-experts is handled by Rule 0.7 below, which
    // preserves the peel to ALGO 1 that flat_m_tile's internal wide-N
    // fallback used to do.  ALGO 2 is now a PURE M-tile executor without
    // that fallback, so shallow-M decode (`M[i] < team_size`) would
    // under-fill the team; classify the regime here — as Rule 0.7 does for
    // prompt — to reproduce the old internal routing:
    //   * kWideN (shallow M) → ALGO 1 (full-team sequential = old wide-N).
    //   * kMTile             → ALGO 2 (tier chosen inside flat_m_tile).
    // kManyExperts cannot fire here (total_experts ≤ kFewExpertsAlgo2Pref(8)
    // ≤ num_threads on any real host); the arm is defensive only.
    if (is_decode && total_experts <= kFewExpertsAlgo2Pref
            && !phase_env_pinned) {
        switch (classify_m_tile_regime(M, num_threads)) {
            case m_tile_regime::kManyExperts:
                return pick(n_tile_safe ? 3 : 1, "auto_rule05_many_experts", 3);
            case m_tile_regime::kWideN: return pick(1, "auto_rule05_wide_n", 1);
            case m_tile_regime::kMTile:
                return pick(m_tile_safe ? 2 : 1, "auto_rule05_m_tile", 2);
        }
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

    // Rule 0.7 — PROMPT → ALGO 1
    if (!is_decode && !grp_matmul_auto_prompt_algo_is_set()) {
        return pick(1, "auto_rule07_prompt_seq", 1);
    }

    // Rule 1 — PHASE ENV.  Single-line phase classification (decode iff
    // `max_M ≤ kDecodeMaxM`) drives which env is consulted.  When the
    // active phase env is non-zero the operator has explicitly pinned
    // that algo for the phase — return it directly, with the same
    // m_tile_safe / n_tile_safe correctness clamps the global ALGO env
    // path applies.  Non-tile-safe + ALGO 3 falls to ALGO 1; non-m-tile-
    // safe + ALGO 2 falls to ALGO 1.  These clamps emit no WARN: the
    // `[GRP_MATMUL.ALGO WARN]` line belongs to `select_grp_matmul_algo`'s
    // global-env branch, which this path does not reach — reaching here
    // means the global ALGO was AUTO.  Operators see the clamp on the
    // `[GRP_MATMUL.ALGO]` line as `chosen=ALGO_X reason=auto_phase_env_clamp`.
    //
    // A gate-declined decode `=5` pin (Rule 0.6a) resolves to the DECODE DEFAULT
    // here, so the call lands exactly where an UNSET `AUTO_DECODE_ALGO` would.
    // Clearing `phase_env_pinned` above re-enabled the decode policy rules
    // (0.45 / 0.5 / 0.6); this is the last step that would otherwise still read
    // the pinned `5` and hand it back.  Substituting the default — rather than
    // falling through to the Rule 2 legacy cascade — is what makes "declined"
    // mean "as if unset" instead of a third, otherwise-unreachable policy.
    const int phase_algo = decode_algo5_pin_declined
            ? kGrpMatmulAutoDecodeAlgoDefault
            : (is_decode ? get_grp_matmul_auto_decode_algo()
                         : get_grp_matmul_auto_prompt_algo());
    if (phase_algo >= 1 && phase_algo <= 5) {
        if (phase_algo == 2 && !m_tile_safe) {
            return pick(1, "auto_phase_env", phase_algo);
        }
        if (phase_algo == 3 && !n_tile_safe) {
            return pick(1, "auto_phase_env", phase_algo);
        }
        // ALGO 4 and 5 are honoured here.  The no-4-no-5 invariant governs
        // what auto-select picks ON ITS OWN, not what an operator may ask
        // for: a phase pin is an explicit request, and silently rewriting it
        // to 3 (as this did until now) made `AUTO_{PROMPT,DECODE}_ALGO={4,5}`
        // look supported while doing something else, with a warning on every
        // single call.  Neither algo has a tiling precondition, so no clamp
        // applies.
        return pick(phase_algo, "auto_phase_env", phase_algo);
    }

    // Rule 2 — LEGACY RULES (phase env == 0).
    //
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
// Returns ALGO number (1-5).  Driven by:
//   * `check_m_tile_safe` / `check_n_tile_extra` — helper checks that
//     determine whether the M-tile slicer is safe to use and whether
//     the extra constraints required by the N-tile path are satisfied
//     without corrupting packed-B / post-op buffers.
//   * `auto_select_algo` — cost-model-free heuristic used when the
//     caller leaves ZENDNNL_GRP_MATMUL_ALGO unset (== 0).

int select_grp_matmul_algo(const std::vector<char> &layout,
        const std::vector<int> &M, const std::vector<int> &N,
        const std::vector<int> &K, const std::vector<matmul_params> &params,
        int num_threads, auto_algo_trace *trace) {

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
            && check_n_tile_extra(M, params, num_ops_eff);

    // Manual override: ZENDNNL_GRP_MATMUL_ALGO=1..5.
    //   ALGO 2 (M-tile): needs m_tile_safe (row-major, uniform dtypes).
    //   ALGO 3 (N-tile): needs n_tile_safe (+ unpacked B, no buffer post-ops).
    //   ALGO 1/4/5:      no tiling → no safety guard needed (BLAS handles all).
    // Unsafe env overrides fall back to ALGO 1 rather than failing, so
    // callers that force-deploy a given ALGO never hit a hard error on
    // shape edge cases.
    const int env_algo = get_grp_matmul_algo();
    if (env_algo >= 1 && env_algo <= 5) {
        int algo = env_algo;
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
                        "[GRP_MATMUL Level2 dispatch WARN] env_algo=3 "
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

    return auto_select_algo(
            M, N, K, params, num_threads, m_tile_safe, n_tile_safe, trace);
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

    // ── WEIGHT_CACHE=2 (in-place) safety downgrade for grouped matmul ─────
    // In-place reorder/pack mutates the caller's weight buffer into a
    // backend-/layout-specific blocked form, so it is only safe when a
    // SINGLE layout ever touches a given weight buffer.  AUTO scheduling
    // violates that on the SAME W13/W2 buffers: the prompt phase routes to
    // an AOCL full-weight reorder while decode routes to AOCL per-tile or
    // the custom-kernel pack — two layouts over one buffer (corruption).
    // So under AUTO we take ONE of two deterministic verdicts (detailed at
    // the (A)/(B) branch below):
    //   (A) keep WC=2 in a MIXED in-place mode when it is provably safe
    //       (PREPACK + CROSS_WARM + unlimited LRU pre-warm every out-of-
    //       place layout from raw W before the single in-place prompt
    //       mutation), or
    //   (B) downgrade the process-wide cache mode to out-of-place (1)
    //       otherwise — a mirror of the single-matmul guard in
    //       `lowoha_matmul_utils.cpp`.
    // Both verdicts are derived from process-constant env, so the decision
    // (and its one-shot log) is stable for the rest of the run.
    //
    // PINNED algos keep a single layout per weight and so KEEP WC=2:
    //   * ALGO 1/2/4/5 — one AOCL reorder layout per buffer, via AOCL's
    //     own in-place path (aocl_kernel.cpp, size-gated with an out-of-
    //     place fall-back when the blocked layout exceeds the plain size).
    //   * ALGO 3 + CUSTOM_KERNEL off — pure AOCL DLP, single layout.
    //   * ALGO 3 + CUSTOM_KERNEL on — made safe in the pack layer:
    //     bf16/even-K packs IN PLACE into the (single-consumer) weight
    //     buffer; int8 (extra compensation row) and odd-K bf16 (VNNI
    //     K-pair pad) fall back to an out-of-place cache that never
    //     mutates the weight (see prepare_for_call +
    //     get_or_pack_weight_bf16).
    if (matmul_config_t::instance().get_weight_cache() == 2) {
        // Only AUTO (env_algo==0) mixes layouts on one buffer; pinned algos
        // are left at WC=2 (see the route table above).
        const int wc_env_algo = get_grp_matmul_algo();
        if (wc_env_algo == 0) {
            // AUTO has two outcomes for WC=2:
            //
            //  (A) MIXED in-place (preferred when safe) — keep WC=2 and route
            //      asymmetrically: the AOCL full-weight (prompt) reorder mutates
            //      the weight buffer IN PLACE, while CK decode and AOCL per-tile
            //      decode stay OUT-OF-PLACE.  This is only correct because
            //      cross-warm prefetches every out-of-place layout from the RAW
            //      weights BEFORE the single in-place mutation, and nothing
            //      re-reads raw W afterwards.  Preconditions:
            //        * PREPACK on   — cross-warm runs upfront (before compute);
            //        * CROSS_WARM on — both phases' layouts are warmed on the
            //          first call;
            //        * AOCL-DLP compiled in — mixed mode's in-place prompt
            //          reorder and cross-warm both route through the AOCL DLP
            //          inner kernel (`aocl_dlp_blocked`); without it the warmers
            //          stub out and cross-warm is a no-op, so mixed-in-place is
            //          disabled at dispatch rather than enabled-then-failed in
            //          prepack;
            //        * CUSTOM_KERNEL on — decode runs through the CK pack, which
            //          cross-warm fully pre-warms (regime 3) from raw W.  With CK
            //          OFF, decode falls to the AOCL per-tile path whose tight
            //          (nr_align=2) variant cross-warm leaves lazy, so a decode
            //          after the prompt mutation could reorder from the already-
            //          mutated buffer and corrupt.  Requiring CK keeps the one
            //          in-place-mutated layout class (bf16 even-K) on the warmed
            //          CK decode path;
            //        * tight fused-MoE layout (FUSED_MOE_TIGHT != 0) — a WIDE fused
            //          layout makes `flat_n_tile` DISABLE the custom kernel even
            //          with CK enabled, routing decode through the AOCL per-tile
            //          path that would re-read the already-mutated buffer.  This is
            //          required UNCONDITIONALLY (NOT gated on the per-call
            //          fused_act): the WC=2 verdict is process-wide and sticky, so
            //          it must derive only from process-constant env.  Keying it on
            //          a per-call fused_act would let a non-fused call ENABLE
            //          in-place mutation and a later wide-fused call flip the
            //          verdict + downgrade 2->1 AFTER weights are mutated, so any
            //          out-of-place reorder that then reads them as raw corrupts;
            //        * unlimited LRU capacity — the in-place sentinel can never
            //          be evicted and lazily re-derived from the mutated buffer.
            //      The grouped dispatcher sets `grp_auto_mixed_inplace`; the CK
            //      runtime and the prepack warmers consult it (see
            //      custom_kernel/dispatch.cpp + prepack/prepack.cpp).
            //
            //  (B) DOWNGRADE to out-of-place (1) — the historical safe fallback
            //      whenever (A)'s preconditions do not hold.
            //
            // Both writes are IDEMPOTENT (every AUTO call re-derives the same
            // deterministic verdict from process-constant env, so concurrent
            // cold-start calls converge); only the log lines are one-shot.
#if ZENDNNL_DEPENDS_AOCLDLP
            const bool aocl_dlp_compiled = true;
#else
            const bool aocl_dlp_compiled = false;
#endif
            const bool mixed_eligible = aocl_dlp_compiled
                    && get_grp_matmul_prepack() && get_grp_matmul_cross_warm()
                    && get_grp_matmul_custom_kernel()
                    && get_grp_matmul_fused_moe_tight() != 0
                    && matmul_config_t::instance().get_lru_cache_capacity()
                            == std::numeric_limits<uint32_t>::max();
            if (mixed_eligible) {
                matmul_config_t::instance().set_grp_auto_mixed_inplace(true);
                static std::atomic<bool> s_wc2_mixed_announced {false};
                if (!s_wc2_mixed_announced.exchange(
                            true, std::memory_order_relaxed)) {
                    apilog_info(
                            "[GRP_MATMUL.WEIGHT_CACHE] weight_cache_type=2 "
                            "AUTO "
                            "mixed-in-place ENABLED "
                            "(prepack+cross_warm+custom_kernel on, "
                            "unlimited LRU capacity): AOCL full-weight prompt "
                            "reorder "
                            "mutates the weight "
                            "buffer in place; CK / AOCL per-tile decode stay "
                            "out-of-place, "
                            "pre-warmed from raw weights before the mutation.");
                }
            } else {
                // Store weight_cache=1 before clearing the mixed-mode flag so THIS
                // thread's own later reads never see (weight_cache==2, mixed false).
                // NOTE: both are RELAXED atomics on independent locations, so this
                // program order is NOT a cross-thread visibility guarantee — another
                // thread may observe the two updates in either order.  Correctness
                // does not rely on it: the AUTO verdict is deterministic from
                // process-constant env, so for an ineligible config the mixed flag is
                // never set true (the eligible branch above is never taken); this
                // branch only ever clears an already-false flag and pins WC=1.  The
                // first-call cold-start window (before any thread runs this downgrade)
                // is covered by the framework's single warm-up call — see the
                // mixed-mode contract in the header block above.
                matmul_config_t::instance().set_weight_cache(1);
                matmul_config_t::instance().set_grp_auto_mixed_inplace(false);
                static std::atomic<bool> s_wc2_downgrade_warned {false};
                if (!s_wc2_downgrade_warned.exchange(
                            true, std::memory_order_relaxed)) {
#if !ZENDNNL_DEPENDS_AOCLDLP
                    apilog_warning(
                            "[GRP_MATMUL.WEIGHT_CACHE] weight_cache_type=2 "
                            "(in-place) "
                            "requires AOCL-DLP (ZENDNNL_DEPENDS_AOCLDLP=OFF): "
                            "mixed-in-place "
                            "is unavailable.  Downgrading process-wide to "
                            "out-of-place "
                            "(weight_cache_type=1) for the rest of the run; "
                            "kernel selection "
                            "unchanged.");
#else
                    apilog_warning(
                            "[GRP_MATMUL.WEIGHT_CACHE] weight_cache_type=2 "
                            "(in-place) is "
                            "unsafe under AUTO scheduling (env_algo=0) without "
                            "prepack+cross_warm+custom_kernel and unlimited "
                            "LRU capacity.  "
                            "Downgrading "
                            "process-wide to out-of-place "
                            "(weight_cache_type=1) for the "
                            "rest of the run; kernel selection unchanged.");
#endif
                }
            }
        } else {
            // Pinned ALGO (1..5) under WC=2: a single reorder layout owns each
            // weight buffer, so the backends' normal in-place path is already
            // safe.  Mixed mode is AUTO-only — clear the flag deterministically
            // so a value left set by a prior AUTO run cannot leak into the
            // pinned path (which would wrongly force the CK pack out-of-place).
            // env_algo is process-constant, so all threads agree on this branch.
            matmul_config_t::instance().set_grp_auto_mixed_inplace(false);
        }
    }

    auto_algo_trace algo_trace;
    const int use_algo = select_grp_matmul_algo(
            layout, M, N, K, params, num_threads, &algo_trace);

    // Decide whether the chosen ALGO fuses the gated activation inline.
    //   - ALGOs 1/2/4/5 always fuse (per-expert or per-M-tile).
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
    const bool caller_layout_tight
            = (use_algo == 3) && !ldc.empty() && !N.empty() && ldc[0] < N[0];
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
            && a3_can_fuse_act(fused_act, get_grp_matmul_custom_kernel())
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
        const int env_algo = get_grp_matmul_algo();
        const int max_M_v = *std::max_element(M.begin(), M.end());
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
        // Phase + per-phase env values for telemetry.  The phase
        // classification mirrors `auto_select_algo`'s phase gate so the
        // log reflects the routing decision the planner actually made.
        const bool is_decode = (max_M_v <= kDecodeMaxM);
        const int phase_env_prompt = get_grp_matmul_auto_prompt_algo();
        const int phase_env_decode = get_grp_matmul_auto_decode_algo();
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
        if (env_algo >= 1 && env_algo <= 5) {
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
        // Hoist `get_grp_matmul_custom_kernel()` to a named local — the
        // value is consumed only here, but parking it up front matches
        // the same "single read per log-line" pattern we applied to the
        // PLAN apilog in `group_matmul_n_tile.cpp` and makes the log's
        // read-set explicit at a glance.  The underlying getter caches
        // its env value, so the saving is microscopic; clarity is the
        // deliverable.
        const int log_custom_kernel = get_grp_matmul_custom_kernel();
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
        const bool ck_int8_shapes = (use_algo == 3) && log_custom_kernel
                && log_custom_kernel_int8
                && (params[rep].dtypes.wei == data_type_t::s8)
                && (params[rep].dtypes.dst == data_type_t::bf16)
                && (params[rep].dtypes.compute == data_type_t::s8
                        || params[rep].dtypes.compute == data_type_t::u8);
        const bool ck_hint_int8 = ck_int8_shapes
                && ((params[rep].dynamic_quant
                            && params[rep].dtypes.src == data_type_t::bf16)
                        || (params[rep].dtypes.src == data_type_t::s8
                                && params[rep].quant_params.src_scale.buff
                                        != nullptr));
        const bool ck_hint = ck_hint_bf16 || ck_hint_int8 || ck_hint_f16;
        const char *ck_family = ck_hint_bf16 ? "bf16"
                : ck_hint_int8
                ? (params[rep].dtypes.compute == data_type_t::u8 ? "int8_asym"
                                                                 : "int8_sym")
                : ck_hint_f16 ? "f16"
                              : "none";
        // SELECTION record (emitted BEFORE the executor runs): `chosen=ALGO_X`
        // is the algo the selector picked, with `reason` explaining the gate.
        // The ALGO that ACTUALLY executed — including any in-executor clamp or
        // fork (e.g. ALGO 2 -> sequential-full-team when active_ops>num_threads,
        // or ALGO 2 -> vertical fusion in the fused path) — is reported by the
        // post-exec `[GRP_MATMUL.CALL]` line via `mode=` (precise branch) and
        // `exec_algo=` (the real 1..5).  Compare those two lines to see any
        // selection-vs-execution divergence.
        apilog_info("[GRP_MATMUL.ALGO] chosen=ALGO_", use_algo,
                " env_algo=", env_algo, " reason=", reason,
                " phase=", (is_decode ? "decode" : "prompt"),
                " auto_prompt_env=", phase_env_prompt,
                " auto_decode_env=", phase_env_decode,
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
    {
        bool has_prepacked_b = false;
        for (size_t i = 0; i < params.size() && i < M.size(); ++i) {
            if (M[i] > 0 && params[i].mem_format_b == 'r'
                    && params[i].lowoha_algo
                            == matmul_algo_t::moe_custom_kernel) {
                has_prepacked_b = true;
                break;
            }
        }
        if (has_prepacked_b && use_algo != 3) {
            set_mode("error_prepacked_no_ck");
            return false;
        }
    }
    // ALGO 3/AUTO: always simulated W4A8 (s4→s8). Inner 1 or 4 both
    // run blocked s8s8_sym_quant after this widen; there is no native s4.
    const int dispatch_num_ops = static_cast<int>(M.size());
    static thread_local std::vector<void *> w4a8_s8_ptrs;
    bool any_w4a8 = false;
    if (use_algo == 3 || use_algo == 0) {
        w4a8_populate_plain_s8_cache(weight, K, N, ldb, transB, params,
                dispatch_num_ops, w4a8_s8_ptrs, any_w4a8);
    }

    switch (use_algo) {
        case 1:
            set_mode("sequential_experts");
            sequential_experts(layout, transA, transB, M, N, K, alpha, src, lda,
                    weight, ldb, bias, beta, dst, ldc, is_weights_const, params,
                    num_threads, fused_act, act_dtype);
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
                    any_w4a8 ? &w4a8_s8_ptrs : nullptr);
            break;
        case 4:
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
                    num_threads, fused_act, act_dtype);
            break;
    }
    return act_fused;
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
