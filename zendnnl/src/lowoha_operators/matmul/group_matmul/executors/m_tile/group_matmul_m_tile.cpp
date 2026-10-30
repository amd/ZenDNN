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

/// ALGO 2 — M-tile parallel GEMM for grouped expert matmul.
///
/// The vertical-fusion pipeline (formerly Section C) lives in
/// `fused_moe/vertical_fusion_pipeline.cpp`; the slice helper both use is
/// in `m_tile_internal.hpp`.
///
/// This file owns the single-projection M-tile executor
/// `m_tile_execute`.  The vertical-fusion executor
/// `try_fused_m_tile_core` (W13→act→W2) and its wrapper
/// `try_fused_m_tile` share the planner in
/// `group_matmul_m_tile_planner.hpp` but live in
/// `fused_moe/vertical_fusion_pipeline.cpp`.
///
/// Both are exposed to higher layers via the public M-tile interface
/// header `m_tile/group_matmul_m_tile.hpp`; the inner-slice helpers
/// stay private to this translation unit (anonymous namespace).
///
/// File layout (search for the banners below to jump):
///
///   ┌────────────────────────────────────────────────────────────┐
///   │  Section A — Shared M-tile runtime helpers                  │
///   │    A.1  offset_quant_by_row                                 │
///   │    A.2  execute_m_tile / execute_m_tile_act                 │
///   │    A.3 / A.4  MTileSingleTierPlan + planner —              │
///   │               MOVED to `group_matmul_m_tile_planner.hpp`    │
///   │               (Sections P.2 / P.3).  Re-included via the    │
///   │               public header for                             │
///   │               `make_m_tile_single_tier_plan`.               │
///   │    A.5  execute_light_expert (multi-tier light pool)       │
///   │    A.6  DQ-INT8 hoist primitives (HoistedSrcQuant_mtile +  │
///   │           dqint8_compact_and_requant_slice)                │
///   ├────────────────────────────────────────────────────────────┤
///   │  Section B — Legacy single-matmul M-tile executor          │
///   │    B.1  m_tile_execute (round-based / multi-tier / shallow-M /│
///   │           Phase-2 single-tier)                             │
///   └────────────────────────────────────────────────────────────┘
///
/// The fused-MoE dispatch lives one level up, inside
/// `group_matmul_fused_moe_execute` in `group_matmul_fused_moe.cpp`:
/// it tries the M-tile pipeline (`try_fused_m_tile`, ALGO 2) or the
/// expert-parallel pipeline (`try_fused_expert_parallel`, ALGO 5), and
/// otherwise runs the two-pass (`fused_moe_two_pass_execute`), whose
/// Op1 and Op2 each go through `group_matmul_execute`.

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstring>
#include <vector>

#include <omp.h>

#include "group_matmul_m_tile.hpp" // own decls + env knobs
// (re-includes planner header)
#include "../../common/group_matmul_parallel_common.hpp"
#include "../../prepack/prepack.hpp"
#include "lowoha_operators/matmul/group_matmul/executors/m_tile/m_tile_internal.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

// ═══════════════════════════════════════════════════════════════════════
// SECTION A — Shared M-tile runtime helpers
// ═══════════════════════════════════════════════════════════════════════
//
// Private (anonymous namespace) helpers used by BOTH the legacy
// executor (Section B, `m_tile_execute`) and the vertical-fusion
// pipeline executor (Section C, `try_fused_m_tile_core`).
//
// What stays here:  per-thread slice executors and post-op / quant
//                   row-offset helpers (`offset_quant_by_row`,
//                   `execute_m_tile`, `execute_m_tile_act`,
//                   `execute_light_expert`).  These are tightly
//                   coupled to BLAS / oneDNN dispatch — they are
//                   runtime work, not planning.
//
// What moved out:   the planner itself (Sections A.3 / A.4 in the
//                   pre-refactor layout).  `MTileSingleTierPlan`
//                   and `make_m_tile_single_tier_plan` live in
//                   `group_matmul_m_tile_planner.hpp` (Sections P.2 /
//                   P.3) so future planner optimizations get a single
//                   home and can be unit-tested in isolation from the
//                   OMP runtime.  `group_matmul_m_tile.hpp`
//                   re-includes the planner header, so the names
//                   stay available to both executors below
//                   unchanged.
//
// Keep the helpers here behaviour-pure (no env reads, no capture-tag
// writes, no logging) so the call site controls observability and
// either executor can audit / unit-test the planner output in
// isolation.
//
// ═══════════════════════════════════════════════════════════════════════

namespace {

// NOTE: A.1 `offset_quant_by_row` moved to `m_tile_internal.hpp`, shared
// with the vertical-fusion pipeline.

// ── A.2  execute_m_tile / execute_m_tile_act ────────────────────────────
//
// Per-thread slice executor: given the planner's (expert, local_tid,
// team_size) triple, compute the row range owned by this thread and
// dispatch a single matmul (BLAS / oneDNN / custom-kernel as routed by
// `algo`) on that slice.  Both legacy and vertical-fusion executors
// call this for their inner stages (legacy: once per pass; vertical
// fusion: twice per slice — once for W13, once for W2).
inline void execute_m_tile(int e, int local_tid, int team_size,
        const std::vector<char> &layout, const std::vector<bool> &transA,
        const std::vector<bool> &transB, const std::vector<int> &M,
        const std::vector<int> &N, const std::vector<int> &K,
        const std::vector<float> &alpha, const std::vector<const void *> &src,
        const std::vector<int> &lda, const std::vector<const void *> &weight,
        const std::vector<int> &ldb, const std::vector<const void *> &bias,
        const std::vector<float> &beta, const std::vector<void *> &dst,
        const std::vector<int> &ldc, const std::vector<bool> &is_weights_const,
        const std::vector<matmul_params> &params, size_t src_elem,
        size_t dst_elem, matmul_algo_t algo) {

    const int row_start = static_cast<int>(
            static_cast<int64_t>(M[e]) * local_tid / team_size);
    const int row_end = static_cast<int>(
            static_cast<int64_t>(M[e]) * (local_tid + 1) / team_size);
    const int slice_M = row_end - row_start;
    if (slice_M <= 0) return;

    // src: row offset.
    const size_t src_off = transA[e]
            ? static_cast<size_t>(row_start) * src_elem
            : static_cast<size_t>(row_start) * lda[e] * src_elem;
    const auto *s = static_cast<const char *>(src[e]) + src_off;
    auto *d = static_cast<char *>(dst[e])
            + static_cast<size_t>(row_start) * ldc[e] * dst_elem;
    static thread_local matmul_params slice_params;
    slice_params = params[e];
    // Each M-slice GEMM is serial even though the enclosing ALGO 2 call owns
    // the outer team.  Match the runtime DLP hint to the primary prepack.
    slice_params.num_threads = 1;

    // Row-offset binary post-op buffers.  Determine broadcast-vs-2D from
    // po.dims (not leading_dim, which can be -1 for "unset" 2D tensors).
    // 1D broadcast includes: rank-0/1 tensors AND rank-2 {1,N} tensors.
    // Row-varying 2D/3D tensors: offset by row_start × effective_ld.
    for (auto &po : slice_params.postop_) {
        if ((po.po_type == post_op_type_t::binary_add
                    || po.po_type == post_op_type_t::binary_mul)
                && po.buff != nullptr) {
            const bool is_broadcast_1d = (po.dims.size() <= 1)
                    || (po.dims.size() == 2 && po.dims[0] == 1);
            if (!is_broadcast_1d) {
                const int eff_ld = (po.leading_dim > 0) ? po.leading_dim : N[e];
                const size_t po_elem = size_of(po.dtype);
                po.buff = static_cast<uint8_t *>(po.buff)
                        + static_cast<size_t>(row_start) * eff_ld * po_elem;
            }
        }
    }

    // Row-offset per-token / per-group src quantization (dims={M,1} or {M,G}).
    // Also rewrites dims[0] from the full M to slice_M so the dynamic-quant
    // reorder dispatcher's per-group gate (is_per_group_col_dims) matches the
    // sliced src_shape.  Per-tensor (dims empty or {1}) needs no offset.
    // Wei quant is N-dependent but M-tile keeps full N → unchanged.
    offset_quant_by_row(
            slice_params.quant_params.src_scale, row_start, slice_M);
    offset_quant_by_row(slice_params.quant_params.src_zp, row_start, slice_M);

    execute_expert_slice(layout[e], transA[e], transB[e], slice_M, N[e], K[e],
            alpha[e], s, lda[e], weight[e], ldb[e], bias[e], beta[e], d, ldc[e],
            is_weights_const[e], 1, slice_params,
            resolve_expert_backend(2, algo, slice_params));
}

// M-tile with fused activation: apply immediately after GEMM while
// output rows are hot in L1/L2.

inline void execute_m_tile_act(int e, int local_tid, int team_size,
        const std::vector<char> &layout, const std::vector<bool> &transA,
        const std::vector<bool> &transB, const std::vector<int> &M,
        const std::vector<int> &N, const std::vector<int> &K,
        const std::vector<float> &alpha, const std::vector<const void *> &src,
        const std::vector<int> &lda, const std::vector<const void *> &weight,
        const std::vector<int> &ldb, const std::vector<const void *> &bias,
        const std::vector<float> &beta, const std::vector<void *> &dst,
        const std::vector<int> &ldc, const std::vector<bool> &is_weights_const,
        const std::vector<matmul_params> &params, size_t src_elem,
        size_t dst_elem, matmul_algo_t algo, grp_matmul_gated_act_t fused_act,
        data_type_t act_dtype) {

    execute_m_tile(e, local_tid, team_size, layout, transA, transB, M, N, K,
            alpha, src, lda, weight, ldb, bias, beta, dst, ldc,
            is_weights_const, params, src_elem, dst_elem, algo);

    if (fused_act != grp_matmul_gated_act_t::none) {
        const int row_start = static_cast<int>(
                static_cast<int64_t>(M[e]) * local_tid / team_size);
        const int row_end = static_cast<int>(
                static_cast<int64_t>(M[e]) * (local_tid + 1) / team_size);
        if (row_start < row_end)
            apply_gated_act_inplace(fused_act, dst[e], row_start, row_end, N[e],
                    ldc[e], act_dtype);
    }
}

// ── A.3 / A.4  PLANNER — moved to `group_matmul_m_tile_planner.hpp` ────
//
// `MTileSingleTierPlan` (Section P.2) and the planner function
// `make_m_tile_single_tier_plan` (Section P.3) live in the
// companion header so the planner has a single, isolated optimization
// surface — future surplus heuristics, alternative shallow-M gates, NUMA
// / CCD affinity refinements all land there, not here.
//
// `group_matmul_m_tile.hpp` re-includes the planner header, so both
// symbols stay available to the executors below (Section B's
// `m_tile_execute`, Section C's `try_fused_m_tile_core`) and to
// gtests with no caller-side change.

// ── A.5  execute_light_expert (multi-tier light pool) ──────────────────
//
// Multi-tier "light pool" helper.  Each light-pool thread processes a
// strided slice of light experts sequentially with `team=1` (full M,
// no slicing).  Memory-traffic pattern is essentially identical to
// ALGO 1's sequential_execute on these tiny experts — light expert
// work is dominated by the per-call activation kernel-launch
// overhead, not per-thread arithmetic, so extra threads on the same
// light expert would idle on the OMP barrier rather than speed it up.
// Pulling them out of the standard t_assign pool is the whole point.
//
// Used from the multi-tier branch in `m_tile_execute` (Section B) when
// the AUTO gating engages.  See `get_grp_matmul_m_tile_hybrid()` for
// the gating heuristic.  NOT used by the vertical-fusion executor
// (Section C) — phase 1 explicitly bails out of multi-tier shapes;
// see `try_fused_m_tile_core`'s multi-tier-prediction guard.
inline void execute_light_expert(int e, const std::vector<char> &layout,
        const std::vector<bool> &transA, const std::vector<bool> &transB,
        const std::vector<int> &M, const std::vector<int> &N,
        const std::vector<int> &K, const std::vector<float> &alpha,
        const std::vector<const void *> &src, const std::vector<int> &lda,
        const std::vector<const void *> &weight, const std::vector<int> &ldb,
        const std::vector<const void *> &bias, const std::vector<float> &beta,
        const std::vector<void *> &dst, const std::vector<int> &ldc,
        const std::vector<bool> &is_weights_const,
        const std::vector<matmul_params> &params, matmul_algo_t algo,
        grp_matmul_gated_act_t fused_act, data_type_t act_dtype) {
    static thread_local matmul_params local_params;
    local_params = params[e];
    local_params.num_threads = 1;
    execute_expert_slice(layout[e], transA[e], transB[e], M[e], N[e], K[e],
            alpha[e], src[e], lda[e], weight[e], ldb[e], bias[e], beta[e],
            dst[e], ldc[e], is_weights_const[e], 1, local_params,
            resolve_expert_backend(2, algo, local_params));
    if (fused_act != grp_matmul_gated_act_t::none) {
        apply_gated_act_inplace(
                fused_act, dst[e], 0, M[e], N[e], ldc[e], act_dtype);
    }
}

// NOTE: A.6, the DQ-INT8 vertical-fusion hoist primitives, moved to
// `fused_moe/vertical_fusion_pipeline.cpp` with their only user.

} // namespace

// ═══════════════════════════════════════════════════════════════════════
// SECTION B — Legacy single-matmul M-tile executor
// ═══════════════════════════════════════════════════════════════════════
//
// `m_tile_execute` is the legacy (pre-vertical-fusion) M-tile executor.
// Runs a SINGLE matmul over a grouped expert batch with optional
// fused gated activation (silu / gelu / swiglu_oai).  The MoE FFN
// uses two of these calls back-to-back today (Op1 = W13 + activation,
// Op2 = W2) when vertical fusion's eligibility gate fails — see the
// fork in `group_matmul_fused_moe_execute`.
//
// ALGO 2 is a PURE M-tile executor.  Two genuine M-tile branches plus one
// correctness clamp (capture-tag values, see `test_api::m_tile_path_tag::*`
// in `group_matmul_m_tile_planner.hpp`, Section P.1):
//
//   1. Multi-tier hybrid (`kMultiTier`      = 1) — skewed expert M
//                        distribution; splits threads into light + heavy
//                        pools (see `execute_light_expert`).
//   2. Phase-2 single-tier(`kPhase2Single` = 3) — default M-weighted
//                        CCD-striped slice plan (see Section A.4).
//   *  FixedTeamRounds clamp (`kManyExpertsSeqFallback` = 7) — `active_ops >
//                        num_threads` makes a pure M-tile plan infeasible
//                        (< 1 thread/expert).  AUTO never reaches it
//                        (`auto_select_algo` routes that regime to ALGO 5);
//                        a FORCED ALGO 2 clamps to sequential full-team
//                        (ALGO 1 equivalent) + a one-time WARN.
//
// ALGO 2 is a pure M-tile executor, with no per-expert or sequential
// full-team fallback.  AUTO never selects ALGO 2; calls reaching this
// executor come from an explicit global or phase pin
// (`ZENDNNL_GRP_MATMUL_ALGO=2` or `AUTO_{DECODE,PROMPT}_ALGO=2`).
//
// ═══════════════════════════════════════════════════════════════════════
//
// ── B.1  m_tile_execute (legacy single-matmul M-tile executor) ────────────
//
// ALGO=2: planned M-tile — work-balanced, CCD-spread row-parallel.
//
// Pre-plans thread assignment based on actual M distribution, then
// maps thread teams onto physical CCDs using a universal CCD-striped
// layout that works for ANY num_threads (128, 126, 127, 192, 256,
// etc.).
//
// Key insights:
//   1. Each M-tile thread reads the full weight matrix B.  Adding
//      threads to an expert only helps while compute > weight_read.
//   2. OMP thread IDs map to physical cores via KMP_AFFINITY=compact:
//      tid 0..(cores_per_ccd-1) on CCD 0, tid cores..2*cores-1 on
//      CCD 1, etc.  num_threads that aren't a multiple of 8 produce
//      a partial last CCD (e.g., 126t → 15 full CCDs + 1 with 6
//      cores).
//   3. For cache bandwidth, each expert's weight should touch
//      DISTINCT CCD L3 slices — not cluster on CCDs 0-1.  This is
//      critical when t_assign is small relative to num_threads
//      (decode with few rows).
//
// Planning algorithm (three phases):
//   Phase 1 — Compute ideal threads per expert:
//     t_need[e] = clamp(ceil(M[e] / slice_target), 0, M[e])
//
//   Phase 2 — Fit t_assign to num_threads:
//     (a) total_need ≤ num_threads: distribute surplus to heaviest-
//         load experts (cap at M[e]).  When experts are fully capped
//         (decode with M=1), the remaining threads stay idle.
//     (b) total_need > num_threads: scale down proportionally by M[e].
//
//   Phase 3 — CCD-striped thread→expert mapping (universal layout):
//     Place each expert's t_assign[e] threads sequentially on CCDs
//     starting from a rotating base CCD.  Pack up to cores_per_ccd
//     threads per CCD before spilling to the next.  This spreads
//     experts across CCDs (each expert's weight read hits its own L3
//     slice) while keeping large expert teams localized.
//
//     The OMP region always uses num_threads(num_threads) so physical
//     thread placement is consistent regardless of t_assign sum.
//     Threads without a slot assignment simply exit.

// Minimum rows per M-tile thread for compute to dominate weight read.
// Tunable via `ZENDNNL_GRP_MATMUL_M_TILE_SLICE_TARGET`
// (see `get_grp_matmul_m_tile_slice_target` in
// `group_matmul_parallel_common.hpp`).  Defaulted to 16 to match
// AOCL-DLP's BRGEMM row-block quantum; cached at m_tile_execute entry.

void m_tile_execute(const std::vector<char> &layout,
        const std::vector<bool> &transA, const std::vector<bool> &transB,
        const std::vector<int> &M, const std::vector<int> &N,
        const std::vector<int> &K, const std::vector<float> &alpha,
        const std::vector<const void *> &src, const std::vector<int> &lda,
        const std::vector<const void *> &weight, const std::vector<int> &ldb,
        const std::vector<const void *> &bias, const std::vector<float> &beta,
        const std::vector<void *> &dst, const std::vector<int> &ldc,
        grp_matmul_gated_act_t fused_act, data_type_t act_dtype,
        const std::vector<bool> &is_weights_const,
        std::vector<matmul_params> &params, int num_threads,
        const char **execution_path_out) {

    // The executor OWNS its execution_path: each branch below writes the concrete
    // path it ran into `*execution_path_out` (mirrors `n_tile_execute`), so the
    // post-exec [GRP_MATMUL.CALL] line reports the real M-tile branch rather
    // than a generic "m_tile_execute".  No-op when the caller passes nullptr.
    auto set_mtile_mode = [&](const char *s) {
        if (execution_path_out != nullptr) *execution_path_out = s;
    };
    // Default to a SKIP mode so a no-op early return (empty call,
    // num_threads<=0, or no active expert) reports that nothing executed
    // (exec_algo=0) rather than mislabelling it as a real ALGO-2 run.  The
    // concrete branches below overwrite this with the executed M-tile path.
    set_mtile_mode("flat_m_tile_skip");

    const int num_ops = static_cast<int>(M.size());
    if (num_ops == 0 || num_threads <= 0) return;

    // Generic ahead-of-time weight pre-pack for ALGO 2.  See
    // sequential_execute in group_matmul_dispatch.cpp for the full
    // contract; identical short-circuits here.  `num_threads` is
    // forwarded so cross_warm can prefill regime 2 for the upcoming
    // ALGO 3 decode path when CUSTOM_KERNEL=0 (see the comment on the
    // ALGO 1 call site for the full rationale).
    group_matmul_prepack::prepack_m_tile(
            group_matmul_prepack::build_prepack_params(weight, K, N, ldb,
                    transB, is_weights_const, params, M,
                    group_matmul_custom_kernel_candidate(
                            params, M, N, fused_act),
                    num_threads,
                    n_tile_decode_nr_align(M, N, ldc, fused_act, params),
                    fused_act, act_dtype,
                    /*transA=*/&transA, /*alpha=*/&alpha, /*beta=*/&beta));

    matmul_algo_t algo = resolve_matmul_backend();

    // Element sizes must come from an expert that actually runs.  Grouped
    // quantisation leaves an inactive slot at the caller's default dtype —
    // `params[0].dtypes.src` stays bf16 while the active experts are s8 — and
    // these sizes scale every slice's row offset, so reading slot 0 on a call
    // with a leading inactive expert strides the source at twice the rate.
    const int elem_ref = [&]() {
        for (int e = 0; e < num_ops; ++e) {
            if (M[e] > 0) return e;
        }
        return 0;
    }();
    const size_t src_elem = size_of(params[elem_ref].dtypes.src);
    const size_t dst_elem = size_of(params[elem_ref].dtypes.dst);

    scoped_active_levels guard(1);

    // F8 — Heuristic-constant tuning hatch.  All four reads are cached
    // (static const in the getter) so the first call hits the env once
    // per process and every later call is a single relaxed atomic load
    // + branch.  Production defaults match the previous literal
    // constants exactly (16 / 256 / 4 / 8).  See the doc-block above
    // each getter in `group_matmul_parallel_common.hpp` for the
    // semantics of each knob.
    const int kSliceTarget = get_grp_matmul_m_tile_slice_target();

    // ── Phase 1: count active experts ──
    int active_ops = 0;
    for (int i = 0; i < num_ops; ++i)
        if (M[i] > 0) ++active_ops;
    if (active_ops == 0) return;

    // F2 — Active-position map.  When an expert is inactive (M[i]==0)
    // its raw index still claimed a CCD slot in the prior
    // `i % num_ccds` stripe mapping, leaving the modulus-collision
    // CCD idle (e.g. an 8-expert layer on 64t with one unrouted
    // expert wastes exactly one CCD).
    // Use the active-position index instead: inactive experts get
    // `active_pos[i] = -1` and never participate in the stripe, while
    // active experts get a compact 0..active_ops-1 numbering that
    // covers every CCD when `active_ops >= num_ccds`.  Cheap O(num_ops)
    // pass, used by Phase 2 (cap_at_ccd) and Phase 3 (starting CCD).
    std::vector<int> active_pos(num_ops, -1);
    {
        int next_pos = 0;
        for (int i = 0; i < num_ops; ++i) {
            if (M[i] > 0) active_pos[i] = next_pos++;
        }
    }

    // F1 — One-time HINT when m_tile_execute is pinned on a decode-class
    // call (max_M == 1) AND the active count is so small that most of
    // `num_threads` will sit on the Phase-2 join barrier (e.g. 8 active
    // experts on 128t leaves most threads idle; the single-thread-per-
    // active CCD-parallel mapping is correct for that shape, but the
    // user almost certainly meant
    // ZENDNNL_GRP_MATMUL_AUTO_DECODE_ALGO=3 instead).
    //
    // Mechanics — gate ordering is "cheapest → most expensive" so that
    // a prompt workload running with APILOG=info pays at most a few
    // branches per call (NEVER the O(num_ops) max-scan):
    //   * Cached `apilog_info_enabled()` (`s_hint_log`) — INFO off in
    //     production ⇒ first conjunct is `false` ⇒ the whole block
    //     is one branch.
    //   * `!s_hint_fired.load(...)` — relaxed atomic load; one `mov`
    //     + branch (≈ 1 ns).  After the hint emits once per process
    //     this flips to `false` and the block becomes one branch
    //     forever.
    //   * `active_ops * 4 < num_threads` — single integer compare
    //     against an already-computed local.  This is the CHEAP
    //     decode-shape proxy: only when ≤ 25 % of `num_threads` are
    //     covered by active experts is it even plausible that the
    //     call is decode-class.  We check this BEFORE the max_M
    //     scan so prompt-class workloads (large active count) never
    //     touch `M[i]` here.
    //   * `max_M_for_hint == 1` — O(num_ops) scan only reached when
    //     all three earlier gates pass.  Bounded by num_ops ≤ 256
    //     and only runs on candidate-decode calls (which are rare in
    //     a prompt-only run and stop entirely once the hint has
    //     fired).  Each prompt call avoids this scan completely.
    //
    // No effect on the planner's behaviour; this is diagnostic only.
    // Production paths that intentionally pin ALGO 2 for a decode test
    // will see the hint once and the user can ignore it.
    {
        static std::atomic<bool> s_hint_fired {false};
        static const bool s_hint_log = apilog_info_enabled();
        if (s_hint_log && !s_hint_fired.load(std::memory_order_relaxed)
                && active_ops * 4 < num_threads) {
            int max_M_for_hint = 0;
            for (int i = 0; i < num_ops; ++i) {
                if (M[i] > max_M_for_hint) max_M_for_hint = M[i];
            }
            if (max_M_for_hint == 1) {
                bool expected = false;
                if (s_hint_fired.compare_exchange_strong(
                            expected, true, std::memory_order_relaxed)) {
                    const int idle_pct
                            = (num_threads - active_ops) * 100 / num_threads;
                    apilog_info(
                            "[GRP_MATMUL.HINT] ALGO_2 (m_tile_execute) invoked "
                            "on "
                            "a "
                            "decode-class call (max_M=1, active_ops=",
                            active_ops, ", num_threads=", num_threads,
                            ").  Approximately ", idle_pct,
                            "% of threads will idle on the Phase-2 join "
                            "barrier "
                            "because the M-tile slice plan cannot subdivide "
                            "M=1 "
                            "across threads.  If this routing was unintended, "
                            "set "
                            "ZENDNNL_GRP_MATMUL_AUTO_DECODE_ALGO=3 to keep "
                            "decode "
                            "calls on the wide-N parallel path, or raise the "
                            "per-expert M (larger batch / sequence) so the "
                            "M-tile planner has rows to subdivide.  Emitted "
                            "once "
                            "per process.");
                }
            }
        }
    }

    // ── CCD topology (universal: handles any num_threads) ──
    // F7 — classic-CCD topology assumption: 8 cores per CCD with shared
    // L3 per CCD.  Dense (c-class) CPU variants deviate — they use
    // 16-core CCDs with one L3 per CCD; the planner's striping math
    // still schedules correctly there but treats each large CCD as two
    // 8-core groups (i.e. CCD locality is per 8-core stripe rather
    // than per L3 slice).  Make this a runtime detect when ZenDNN
    // builds against a dense part; until then the constant matches the
    // classic-CCD server parts this planner targets.
    const int cores_per_ccd = std::min(8, num_threads);
    const int num_ccds
            = std::max(1, (num_threads + cores_per_ccd - 1) / cores_per_ccd);
    // CCD capacity is needed by both `make_m_tile_single_tier_plan`
    // (Phase-2 cap_at_ccd checks + Phase-3 striped placement) and by
    // future pipeline executors.  Both compute it from `num_threads` /
    // `cores_per_ccd` internally so no shared lambda is needed at this
    // scope; the multi-tier branch below uses linear (not CCD-striped)
    // mapping over the heavy pool.

    // ── `active_ops > num_threads` is infeasible for SINGLE-TIER only ──
    //
    // A pure single-tier M-tile plan splits an expert's M rows across a thread
    // team; it cannot hand out fewer than one thread per active expert, so
    // when more experts fire than there are threads there is NO valid
    // single-tier slice plan (the planner would floor every active expert at
    // 1 thread, overflow `num_threads`, and silently DROP the surplus experts
    // in its Phase-3 tid mapping — a correctness bug).  That clamp therefore
    // now lives immediately before the single-tier planner below, NOT here.
    //
    // The MULTI-TIER path has no such limit and is the right executor for this
    // regime: heavy experts still get ≥ 1 thread each (enforced by the
    // `candidate_heavy_pool >= n_heavy` gate), while the light tail is drained
    // by an atomic counter whose loop is bounded by `n_light`, not by the pool
    // size — so an arbitrary number of light experts is already handled.
    // Clamping earlier sent these frames to sequential full-team, running
    // every expert one at a time on the whole team.

    // ── Multi-tier hybrid (skewed many-expert / Qwen3-class prompt) ──
    //
    // Engaged by `ZENDNNL_GRP_MATMUL_M_TILE_HYBRID=0` (AUTO, default;
    // set `-1` to force the legacy single-tier path).  See the
    // doc-block on `get_grp_matmul_m_tile_hybrid()` for the gating
    // heuristic and rationale.
    //
    // Problem the tier fixes: with `num_active ≈ num_threads` and
    // `max_M / avg_M ≈ 14×`, the legacy Phase-2 `floor=1 per active`
    // step consumes ~num_active threads as floors, leaving only
    // (num_threads - num_active) ≈ 1-15 surplus for the M-weighted
    // distribution.  The giant expert ends up with ~2 threads while
    // dozens of tiny experts each get 1 — the OMP barrier waits on
    // the under-resourced giant.
    //
    // Tier mechanism: classify experts by M against
    // `light_cut = max(8, avg_M / 4)`; light experts share a small
    // dedicated thread team (`light_pool`) via round-robin (each
    // light thread runs a stride of the light list with `team=1`,
    // full M), while the remaining `heavy_pool = num_threads −
    // light_pool` threads run the standard M-tile distribution over
    // heavy experts only.  Freeing the tiny experts from the floor
    // budget lets the heavies absorb the surplus, drastically
    // improving slice balance.
    //
    // Gating conservatism: the heuristic only engages when ALL of
    // {actives ≥ num_threads/2, max_M ≥ 256, max_M ≥ 4×avg_M,
    //  num_light ≥ num_threads/8} hold simultaneously.
    //
    // This is a workload-shape gate, NOT a model-name gate:
    //   * Architectures with small total-expert count
    //     (E ≤ ~num_threads/4) can never reach
    //     `actives ≥ num_threads/2` at any batch size — `actives`
    //     is bounded by E.  Small-expert-count layers fall in this
    //     category on high-thread-count systems.
    //   * High-BS workloads tend to smooth routing imbalance:
    //     the per-expert M stddev/mean ratio shrinks as
    //     ~1/√(BS·seq), so the `max_M ≥ 4×avg_M` skew gate gets
    //     harder to meet, not easier, as BS grows.
    //   * The high-skew regime this branch fixes is small-BS,
    //     many-experts, sparse-top-K routing (e.g. Qwen3-30B-A3B
    //     with E=128, K=8 at BS≈32).
    // Phase 2 (legacy single-tier, below) is itself M-weighted and
    // already handles architectures / regimes outside this gate
    // correctly; this branch is a targeted fix for the specific
    // `num_active ≈ num_threads` floor-saturated pathology
    // described above.
    if (get_grp_matmul_m_tile_hybrid() == 0) {
        int max_M = 0;
        int64_t sum_M_total = 0;
        for (int i = 0; i < num_ops; ++i) {
            if (M[i] > 0) {
                sum_M_total += M[i];
                if (M[i] > max_M) max_M = M[i];
            }
        }
        const int avg_M = static_cast<int>(sum_M_total / active_ops);

        // F8 — Same env-tunable hatch as `kSliceTarget`.  Defaults match
        // the original literal constants exactly (256 / 4 / 8); each
        // getter caches its env on first call so the four lookups cost
        // four relaxed loads + branches in steady state.
        const int kHybridMinMaxM = get_grp_matmul_m_tile_hybrid_min_max_m();
        const int kHybridMinSkewX = get_grp_matmul_m_tile_hybrid_min_skew();
        const int kLightsPerThread
                = get_grp_matmul_m_tile_hybrid_lights_per_thread();
        const int min_actives = num_threads / 2;
        const int min_lights = num_threads / 8;

        // Overflow-safe skew test: with the F8 env knob
        // `ZENDNNL_GRP_MATMUL_M_TILE_HYBRID_MIN_SKEW` allowing any
        // positive int and `avg_M` bounded only by `int`, the product
        // `kHybridMinSkewX * avg_M` can overflow signed `int` (UB) on
        // pathological inputs.  Promote one operand to `int64_t`
        // so the comparison stays well-defined for all valid int
        // inputs.  At stock defaults (skew=4, avg_M ≤ a few thousand)
        // the cast compiles to a no-op on x86_64.
        const bool gate_skew = (max_M >= kHybridMinMaxM) && (avg_M > 0)
                && (static_cast<int64_t>(max_M)
                        >= static_cast<int64_t>(kHybridMinSkewX) * avg_M);

        if (active_ops >= min_actives && gate_skew) {
            // Light/heavy cut.  The stock rule (a quarter of the mean)
            // assumes every active expert can hold a thread, which stops
            // being true once experts outnumber threads: it then marks more
            // experts heavy than the heavy pool can cover, the
            // `heavy_pool >= n_heavy` gate below declines, and the call falls
            // back to sequential full-team.  So in that regime cut by RANK
            // instead — keep only the largest `num_threads/3` experts heavy
            // and let the tail drain through the light pool.  The cut never
            // moves DOWN, so this only ever reclassifies heavy → light.
            const int light_cut = std::max(8, avg_M / 4);
            // Rank cap, held as a SET of indices rather than a value cut.  A
            // value cut cannot express "the largest `max_heavy` experts" when
            // M ties across the boundary: with `M = {400 x 11, 1 x 53}` and 32
            // threads the boundary value is 400, every 400-expert then tests
            // `M <= light_cut` and lands light, no expert is heavy, and the
            // hybrid declines the whole call.  Ranking by (M desc, index asc)
            // is total, so the cap is exact and reproducible.  With no tie at
            // the boundary this selects the same experts the value cut did.
            std::vector<int> heavy_rank;
            // Experts tied with the boundary M but ranked just outside the
            // cap.  Demoting them is the worst of both worlds: a light runs
            // single-threaded off the atomic queue, and roles are fixed, so
            // an expert the size of the heavy tier becomes the long pole
            // while its identical twins hold whole teams.  Kept aside so the
            // cap can absorb the whole tie group when the heavy pool has
            // room for it.
            std::vector<int> heavy_ties;
            if (active_ops > num_threads) {
                heavy_rank.reserve(active_ops);
                for (int i = 0; i < num_ops; ++i) {
                    if (M[i] > 0) { heavy_rank.push_back(i); }
                }
                const int max_heavy = std::max(1, num_threads / 3);
                if (static_cast<int>(heavy_rank.size()) > max_heavy) {
                    std::nth_element(heavy_rank.begin(),
                            heavy_rank.begin() + max_heavy, heavy_rank.end(),
                            [&](int a, int b) {
                        return M[a] != M[b] ? M[a] > M[b] : a < b;
                    });
                    const int boundary_M = M[heavy_rank[max_heavy - 1]];
                    for (auto it = heavy_rank.begin() + max_heavy;
                            it != heavy_rank.end(); ++it) {
                        if (M[*it] == boundary_M) { heavy_ties.push_back(*it); }
                    }
                    heavy_rank.resize(max_heavy);
                }
                std::sort(heavy_rank.begin(), heavy_rank.end());
                std::sort(heavy_ties.begin(), heavy_ties.end());
            }

            // Pool sizing for a given classification.  Shared by the trial
            // below and the committed plan so the two can never disagree.
            struct pool_plan {
                int light_pool;
                int heavy_pool;
                bool ok;
            };
            const auto plan_pools = [&](int n_light, int n_heavy,
                                            int64_t heavy_M_sum) -> pool_plan {
                // Overflow-safe light-pool ceil-div.  Same defense-in-depth
                // rationale as the skew gate above: with the F8 env knob
                // `ZENDNNL_GRP_MATMUL_M_TILE_HYBRID_LIGHTS_PER_THREAD`
                // accepting any positive int, the intermediate
                // `n_light + kLightsPerThread - 1` can overflow signed `int`.
                // The final ceil-div result is bounded by `n_light` (≤
                // kNTilePlanMaxExperts = 256), so the int cast after the
                // ≥ 1 clamp is safe regardless of the env value.
                int light_pool = std::min(cores_per_ccd,
                        static_cast<int>(std::max<int64_t>(1,
                                (static_cast<int64_t>(n_light)
                                        + kLightsPerThread - 1)
                                        / kLightsPerThread)));
                // Many-expert regime: size the two pools by WORK, not by
                // expert count.  The `cores_per_ccd` cap was calibrated for a
                // negligible tail; once experts outnumber threads the tail
                // carries a real share of `sum_M_total` and a small pool
                // draining it becomes the critical path.  Give the light pool
                // its proportional share, bounded so the heavy pool still
                // covers every heavy expert.
                if (active_ops > num_threads) {
                    const int64_t light_M_sum = sum_M_total - heavy_M_sum;
                    int prop = static_cast<int>(
                            static_cast<int64_t>(num_threads) * light_M_sum
                            / std::max<int64_t>(1, sum_M_total));
                    prop = std::max(1, std::min(prop, num_threads - n_heavy));
                    light_pool = std::max(light_pool, prop);
                }
                const int heavy_pool = num_threads - light_pool;
                const bool ok = n_light > 0 && n_light >= min_lights
                        && n_heavy > 0 && heavy_M_sum > 0
                        && light_pool < num_threads && heavy_pool >= n_heavy;
                return pool_plan {light_pool, heavy_pool, ok};
            };

            const auto classify = [&](const std::vector<int> &heavy_set,
                                          std::vector<int> &light_exp,
                                          std::vector<int> &heavy_exp,
                                          int64_t &heavy_M_sum) {
                light_exp.clear();
                heavy_exp.clear();
                heavy_M_sum = 0;
                light_exp.reserve(num_ops);
                heavy_exp.reserve(num_ops);
                for (int i = 0; i < num_ops; ++i) {
                    if (M[i] <= 0) continue;
                    const bool capped = heavy_set.empty()
                            || std::binary_search(
                                    heavy_set.begin(), heavy_set.end(), i);
                    if (M[i] <= light_cut || !capped) {
                        light_exp.push_back(i);
                    } else {
                        heavy_exp.push_back(i);
                        heavy_M_sum += M[i];
                    }
                }
            };

            std::vector<int> light_exp;
            std::vector<int> heavy_exp;
            int64_t heavy_M_sum = 0;
            classify(heavy_rank, light_exp, heavy_exp, heavy_M_sum);

            // Absorb the tie group when the wider heavy tier still passes
            // every pool gate.  Trying it first and keeping it only on
            // success means this can promote a demoted expert but can never
            // turn an engaging call into a declining one.
            if (!heavy_ties.empty()) {
                std::vector<int> wide_set;
                wide_set.reserve(heavy_rank.size() + heavy_ties.size());
                std::merge(heavy_rank.begin(), heavy_rank.end(),
                        heavy_ties.begin(), heavy_ties.end(),
                        std::back_inserter(wide_set));
                std::vector<int> wide_light;
                std::vector<int> wide_heavy;
                int64_t wide_heavy_M_sum = 0;
                classify(wide_set, wide_light, wide_heavy, wide_heavy_M_sum);
                if (plan_pools(static_cast<int>(wide_light.size()),
                            static_cast<int>(wide_heavy.size()),
                            wide_heavy_M_sum)
                                .ok) {
                    light_exp.swap(wide_light);
                    heavy_exp.swap(wide_heavy);
                    heavy_M_sum = wide_heavy_M_sum;
                }
            }
            const int n_light = static_cast<int>(light_exp.size());
            const int n_heavy = static_cast<int>(heavy_exp.size());

            // Hybrid only profitable when there are enough lights to free
            // significant heavy-pool budget AND ≥ 1 heavy expert.  If the
            // gating saw skew but everything ended up "heavy" (e.g.,
            // `light_cut` too small for the actual M distribution),
            // fall through to the single-tier path.
            //
            // Defensive safety guards (each independently sufficient to
            // prevent incorrect output, kept together for defense-in-depth):
            //
            //   * `n_light > 0`                          — there must be at
            //     least one light expert to actually peel into the light
            //     pool.  Without this guard the `min_lights = num_threads/8`
            //     rule reduces to 0 on `num_threads < 8`, letting
            //     `n_light == 0` pass the count check; combined with the
            //     `candidate_light_pool = std::max(1, ...)` clamp below it
            //     would consume one heavy slot for an empty light pool.
            //     Multi-tier's whole premise is "peel lights off the
            //     team", so engaging without lights also has no payoff.
            //
            //   * `candidate_light_pool < num_threads`   — multi-tier needs
            //     at least 1 thread reserved for the heavy pool, otherwise
            //     heavy experts would never execute.  Triggers on
            //     degenerate num_threads = 1 with n_light ≥ 1 (light pool
            //     would consume the only thread).
            //
            //   * `heavy_pool >= n_heavy`                — every heavy
            //     expert needs its own thread because the scale-down's
            //     `floor = 1 per heavy active` cannot decrement below 1;
            //     if `Σ ht_assign = n_heavy > heavy_pool` the final
            //     mapping loop (line ~654) hits `next_tid < num_threads`
            //     before placing the last heavy and that expert never
            //     executes (stale `dst`).  Combined with the env knob
            //     `ZENDNNL_GRP_MATMUL_M_TILE_HYBRID_MIN_MAX_M` (F8) this
            //     is a real reachable corner — lowering the threshold
            //     pushes the skew gate into the `active_ops == num_threads`
            //     regime where the n_heavy = active_ops shape becomes
            //     legal.  At stock defaults (kHybridMinMaxM = 256) the
            //     bound is structurally unreachable (the skew gate
            //     max_M ≥ 4·avg_M cannot coincide with all-heavy
            //     classification for active_ops ≤ 8), but the guard is
            //     cheap insurance against env-driven misconfiguration.
            //
            // Production 128t / 64t / 32t hosts at defaults: cores_per_ccd
            // = 8 caps light_pool ≤ 8 ≪ num_threads, so heavy_pool ≥
            // num_threads − 8 ≥ 24, and n_heavy ≤ active_ops ≤ num_threads
            // = 32+ is comfortably below the heavy_pool budget.  When any
            // guard fires the call falls through to Phase 2 single-tier —
            // the M-weighted distribution there handles every active
            // expert correctly, and multi-tier's load-balancing payoff is
            // negligible at the offending scale anyway.
            const pool_plan committed
                    = plan_pools(n_light, n_heavy, heavy_M_sum);
            const int candidate_light_pool = committed.light_pool;
            if (committed.ok) {
                // Capture-gated branch tag — commit point for multi-tier
                // hybrid.  Tagged here (not at the outer `if
                // (get_grp_matmul_m_tile_hybrid() == 0)`) so the tag fires
                // ONLY when all four shape gates pass AND there are enough
                // light/heavy experts to make the split profitable (AND the
                // safety guard above keeps heavy_pool ≥ 1); outer-gate hits
                // that fall through to single-tier do not poison the tag.
                // See doc-block on `test_api::s_capture_m_tile_path` in
                // `group_matmul_parallel_common.hpp`.
                set_mtile_mode("flat_m_tile_multitier");
                if (test_api::s_capture_m_tile_path.load(
                            std::memory_order_relaxed)) {
                    test_api::s_last_m_tile_path.store(
                            test_api::m_tile_path_tag::kMultiTier,
                            std::memory_order_relaxed);
                }
                const int light_pool = candidate_light_pool;
                const int heavy_pool = num_threads - light_pool;

                // ── Build heavy-expert t_assign (Phase 1b/2 over heavies only) ──
                // Mirrors the single-tier logic below, but scoped to heavy_exp[]
                // with `heavy_pool` as the thread budget.  `kSliceTarget` is
                // unchanged — the same slice-size heuristic applies inside the
                // heavy pool.
                std::vector<int> ht_assign(num_ops, 0);
                int ht_total_need = 0;
                for (int idx : heavy_exp) {
                    // Overflow-safe ceil-div: with the F8 env knob
                    // `ZENDNNL_GRP_MATMUL_M_TILE_SLICE_TARGET` accepting any
                    // positive int, `M[idx] + kSliceTarget - 1` can overflow
                    // signed `int` for pathological inputs.  The
                    // ceil-div result is bounded by `M[idx]` so the int cast
                    // after the ≥ 1 clamp is always safe.  See the sister
                    // hardening on the single-tier `t_assign[i]` init below.
                    ht_assign[idx] = std::min(M[idx],
                            static_cast<int>(std::max<int64_t>(1,
                                    (static_cast<int64_t>(M[idx]) + kSliceTarget
                                            - 1)
                                            / kSliceTarget)));
                    ht_total_need += ht_assign[idx];
                }

                if (ht_total_need <= heavy_pool) {
                    // Surplus → heaviest per-thread slice.
                    int surplus = heavy_pool - ht_total_need;
                    while (surplus > 0) {
                        int best = -1;
                        int best_slice = 0;
                        for (int idx : heavy_exp) {
                            if (ht_assign[idx] <= 0 || ht_assign[idx] >= M[idx])
                                continue;
                            int cur_slice = (M[idx] + ht_assign[idx] - 1)
                                    / ht_assign[idx];
                            if (cur_slice > best_slice) {
                                best_slice = cur_slice;
                                best = idx;
                            }
                        }
                        if (best < 0) break;
                        ++ht_assign[best];
                        --surplus;
                    }
                } else {
                    // Scale-down proportional to M, floor=1 per heavy active.
                    int assigned = 0;
                    for (int idx : heavy_exp) {
                        ht_assign[idx] = std::max(1,
                                static_cast<int>(
                                        static_cast<int64_t>(heavy_pool)
                                        * M[idx] / heavy_M_sum));
                        assigned += ht_assign[idx];
                    }
                    while (assigned < heavy_pool) {
                        int best = -1;
                        int best_slice = 0;
                        for (int idx : heavy_exp) {
                            if (ht_assign[idx] <= 0 || ht_assign[idx] >= M[idx])
                                continue;
                            int cur_slice = (M[idx] + ht_assign[idx] - 1)
                                    / ht_assign[idx];
                            if (cur_slice > best_slice) {
                                best_slice = cur_slice;
                                best = idx;
                            }
                        }
                        if (best < 0) break;
                        ++ht_assign[best];
                        ++assigned;
                    }
                    while (assigned > heavy_pool) {
                        int best = -1;
                        int least_slice = INT_MAX;
                        for (int idx : heavy_exp) {
                            if (ht_assign[idx] <= 1) continue;
                            int cur_slice = (M[idx] + ht_assign[idx] - 1)
                                    / ht_assign[idx];
                            if (cur_slice < least_slice) {
                                least_slice = cur_slice;
                                best = idx;
                            }
                        }
                        if (best < 0) break;
                        --ht_assign[best];
                        --assigned;
                    }
                }

                // F3 — Multi-tier heavy-underfill perf guard.
                //
                // Compute the actual heavy-pool occupancy after surplus /
                // scale-down completes.  When the heavies absorb < 75 % of
                // `heavy_pool` the multi-tier branch leaves > 25 % of the
                // heavy-thread budget idle on the OMP join barrier — the
                // team-split overhead (per-team setup + barrier latency)
                // exceeds the load-balancing payoff in that regime, and
                // Phase 2's M-weighted single-tier distribution is strictly
                // better (every thread carries a meaningful slice, weighted
                // by the actual M distribution rather than an
                // arbitrary-skew gate).
                //
                // Inert on ≤ 128t (current ship targets): the engagement
                // gate above requires `max_M ≥ 256`, hence
                // `Σ M_heavy ≥ 256 ≥ heavy_pool ≤ 128`, so the surplus
                // loop fully consumes `heavy_pool` and the guard always
                // passes.  The guard's purpose is correctness on ≥ 256t
                // hosts (dense c-class parts) where a single heavy with
                // `M = 256` cannot absorb a `heavy_pool > 256` budget.
                int ht_assigned_final = 0;
                for (int idx : heavy_exp)
                    ht_assigned_final += ht_assign[idx];
                if (ht_assigned_final * 4 >= heavy_pool * 3) {
                    // ── Thread mapping ───────────────────────────────────────────
                    // tids [0 .. light_pool):                light pool (round-robin).
                    // tids [light_pool .. num_threads):      heavy pool (M-tile).
                    //
                    // The heavy mapping is intentionally linear (not CCD-striped)
                    // here — heavy experts already get multi-thread teams via
                    // `ht_assign`, so the team itself spans the natural CCD
                    // boundary in the tid range it owns.  The CCD-stripe used in
                    // the single-tier path below targets a different regime
                    // (few-team-mid-thread) where per-CCD locality dominates;
                    // for the multi-tier heavy pool (40-100 threads on ~20-50
                    // heavies) the tid-range layout already matches CCD ordering
                    // because tid → physical core via KMP_AFFINITY compact.
                    //
                    // F6 (open): the linear walk packs heavy teams
                    // across CCDs by tid order.  When `ht_assign[idx]` does not
                    // divide cores_per_ccd cleanly (e.g. 9 threads/heavy with
                    // cores_per_ccd=8), adjacent heavy teams share the boundary
                    // CCD.  Heavy weights ≤ 16 MB fit two-per-CCD-L3 (e.g. K=2048,
                    // M_heavy ≤ 4096); larger heavies (K=4096, M_heavy > 4096)
                    // could thrash the boundary CCD — recheck if future
                    // deployments push past `M_heavy = 4096` on K = 4096
                    // backbones, in which case
                    // switch to a CCD-aware heavy mapping (each heavy gets
                    // ceil(ht_assign[idx] / cores_per_ccd) contiguous CCDs,
                    // padding the last with zero threads).
                    constexpr int kRoleInactive = 0;
                    constexpr int kRoleLight = 1;
                    constexpr int kRoleHeavy = 2;
                    std::vector<int> mt_tid_to_expert(num_threads, -1);
                    std::vector<int> mt_tid_to_local(num_threads, -1);
                    std::vector<int> mt_tid_to_team(num_threads, 0);
                    std::vector<int> mt_tid_to_role(num_threads, kRoleInactive);

                    for (int t = 0; t < light_pool; ++t) {
                        mt_tid_to_role[t] = kRoleLight;
                    }
                    int next_tid = light_pool;
                    for (int idx : heavy_exp) {
                        const int t = ht_assign[idx];
                        for (int k = 0; k < t && next_tid < num_threads;
                                ++k, ++next_tid) {
                            mt_tid_to_role[next_tid] = kRoleHeavy;
                            mt_tid_to_expert[next_tid] = idx;
                            mt_tid_to_local[next_tid] = k;
                            mt_tid_to_team[next_tid] = t;
                        }
                    }

                    // F5 — Atomic-counter light-pool dispatch.  Replaces the prior
                    // `for (int j = tid; j < n_light; j += light_pool)` static
                    // stride, which on workloads with non-uniform light-M
                    // distributions stride-locked one thread onto a sequence of
                    // heavy lights while peers walked sequences of tiny lights
                    // (worst observed: light_pool=8 lights packed so that
                    // tid=0 carries 2× the wall time of tid=7, costing
                    // ~150 µs / prompt at 32-batch on Qwen3-30B).
                    //
                    // The shared counter is one cache-line wide, accessed only
                    // by the light_pool threads (all on CCD 0 via KMP_AFFINITY
                    // compact), so the relaxed fetch-add costs ~50-100 ns per
                    // light expert and is well-amortised against per-light
                    // execute_light_expert times in the 10-100 µs range.
                    // Dispatch order is `light_exp[]` order (the order in which
                    // experts were appended during the active-experts pass);
                    // no pre-sort is needed because dynamic dispatch rebalances
                    // tail load automatically — the last thread idle picks the
                    // last remaining expert and total wall time is bounded by
                    // (Σ M_light / light_pool) + max_light_M.
                    std::atomic<int> light_next {0};

#pragma omp parallel num_threads(num_threads)
                    {
                        const int tid = omp_get_thread_num();
                        const int role = mt_tid_to_role[tid];
                        if (role == kRoleLight) {
                            for (int j = light_next.fetch_add(
                                         1, std::memory_order_relaxed);
                                    j < n_light;
                                    j = light_next.fetch_add(
                                            1, std::memory_order_relaxed)) {
                                execute_light_expert(light_exp[j], layout,
                                        transA, transB, M, N, K, alpha, src,
                                        lda, weight, ldb, bias, beta, dst, ldc,
                                        is_weights_const, params, algo,
                                        fused_act, act_dtype);
                            }
                        } else if (role == kRoleHeavy) {
                            const int e = mt_tid_to_expert[tid];
                            execute_m_tile_act(e, mt_tid_to_local[tid],
                                    mt_tid_to_team[tid], layout, transA, transB,
                                    M, N, K, alpha, src, lda, weight, ldb, bias,
                                    beta, dst, ldc, is_weights_const, params,
                                    src_elem, dst_elem, algo, fused_act,
                                    act_dtype);
                        }
                    }
                    return;
                } // end F3 ht_assigned_final guard — if heavies underfill
                // heavy_pool we fall through to single-tier instead.
            }
        }
    }
    // (multi-tier gating did not engage — fall through to single-tier)

    // ── Single-tier cannot represent `active_ops > num_threads` ──
    // Placed after the multi-tier hybrid so that gets first refusal on this
    // regime.  Reaching here means multi-tier declined, so the only correct
    // answer left is the sequential full-team path — the single-tier planner
    // would silently drop the surplus experts.
    if (active_ops > num_threads) {
        static const bool s_warn = apilog_warning_enabled();
        static std::atomic<bool> s_warned {false};
        if (s_warn && !s_warned.exchange(true, std::memory_order_relaxed)) {
            apilog_warning(
                    "[GRP_MATMUL.ALGO WARN] env_algo=2 (m_tile_execute) on a "
                    "shape with active_ops > num_threads that the multi-tier "
                    "hybrid declined: single-tier M-tile is infeasible "
                    "(cannot give < 1 thread per active expert).  CLAMP to "
                    "sequential full-team (ALGO 1 equivalent).");
        }
        set_mtile_mode("flat_m_tile_seq_clamp");
        if (test_api::s_capture_m_tile_path.load(std::memory_order_relaxed)) {
            test_api::s_last_m_tile_path.store(
                    test_api::m_tile_path_tag::kManyExpertsSeqFallback,
                    std::memory_order_relaxed);
        }
        for (int e = 0; e < num_ops; ++e) {
            if (M[e] <= 0) continue;
            static thread_local matmul_params local_params;
            local_params = params[e];
            execute_expert_slice(layout[e], transA[e], transB[e], M[e], N[e],
                    K[e], alpha[e], src[e], lda[e], weight[e], ldb[e], bias[e],
                    beta[e], dst[e], ldc[e], is_weights_const[e], num_threads,
                    local_params,
                    resolve_expert_backend(2, algo, local_params));
            if (fused_act != grp_matmul_gated_act_t::none) {
                apply_gated_act_inplace(
                        fused_act, dst[e], 0, M[e], N[e], ldc[e], act_dtype);
            }
        }
        return;
    }

    // ── Single-tier Phase-1b/2/3 plan (refactored helper) ──
    //
    // The shared helper computes the Phase-1b `t_assign` totals + the
    // Phase-3 CCD-striped tid mapping in one call, AND emits the
    // shallow-M fallback signal used by vertical-fusion eligibility.  Pure
    // refactor (no behaviour change) — every shape produces an
    // identical mapping to the previous inline planner.  The pipeline
    // executor (`try_fused_m_tile_core`) reuses the same helper so
    // its slice plan is bit-identical to this one for every workload
    // that reaches Phase 2 single-tier.
    const auto plan = make_m_tile_single_tier_plan(M, active_pos, num_ops,
            num_threads, active_ops, kSliceTarget, cores_per_ccd, num_ccds);

    // ── Shallow-M regime: no sequential fallback (ALGO 2 is pure M-tile) ──
    //
    // The shallow-M regime — few actives × shallow M,
    // i.e. `max_M > 1 && total_need*2 ≤ num_threads` — runs the M-tile plan
    // like any other shape.  AUTO never selects ALGO 2; calls reaching this
    // executor come from an explicit global or phase pin.
    //
    // A FORCED `ZENDNNL_GRP_MATMUL_ALGO=2` on a shallow-M shape therefore runs
    // the single-tier M-tile plan below: this is CORRECT (the regime is
    // M-tile-feasible — `total_need ≤ num_threads/2 < num_threads`, so the
    // planner's Phase-2/3 mapping covers every active expert), just less
    // memory-optimal than ALGO 1 would be.  The planner still computes
    // `plan.shallow_m_fallback` because the vertical-fusion pipeline
    // (`try_fused_m_tile_core`) consults it to bail to the legacy
    // two-pass; `m_tile_execute` itself no longer branches on it.

    // Capture-gated branch tag — Phase 2 single-tier is the default
    // fallthrough when none of the earlier branches commit.  See
    // doc-block on `test_api::s_capture_m_tile_path` in
    // `group_matmul_parallel_common.hpp`.
    set_mtile_mode("flat_m_tile_single_tier");
    if (test_api::s_capture_m_tile_path.load(std::memory_order_relaxed)) {
        test_api::s_last_m_tile_path.store(
                test_api::m_tile_path_tag::kPhase2Single,
                std::memory_order_relaxed);
    }

    // ── Execute: always use full num_threads OMP team ──
    // Threads without a slot assignment simply exit.  Using the full team
    // ensures consistent physical thread placement across CCDs regardless
    // of how many threads are actively doing work.  The tid → (expert,
    // local_tid, team_size) mapping was filled in by
    // `make_m_tile_single_tier_plan` above (Phase 2 fit + Phase 3
    // CCD-striped placement).
    bool mt_short_team = false;
#pragma omp parallel num_threads(num_threads)
    {
        // `plan.tid_to_expert` was sized for the REQUESTED team. A runtime
        // that grants fewer threads (OMP_DYNAMIC / OMP_THREAD_LIMIT / an
        // already-active outer level) leaves every expert mapped above the
        // real team uncomputed, with the call still reporting success.
        if (omp_get_num_threads() < num_threads) {
#pragma omp single
            { mt_short_team = true; }
        } else {
            const int tid = omp_get_thread_num();
            const int e = plan.tid_to_expert[tid];
            if (e >= 0) {
                execute_m_tile_act(e, plan.tid_to_local[tid],
                        plan.tid_to_team[tid], layout, transA, transB, M, N, K,
                        alpha, src, lda, weight, ldb, bias, beta, dst, ldc,
                        is_weights_const, params, src_elem, dst_elem, algo,
                        fused_act, act_dtype);
            }
        }
    }
    if (mt_short_team) { set_mtile_mode("error_m_tile_short_team"); }
}

// NOTE: SECTION C, the vertical-fusion pipeline executor and its
// eligibility-gated wrapper, moved to
// `fused_moe/vertical_fusion_pipeline.cpp`.

// ═══════════════════════════════════════════════════════════════════════
// End of file
// ═══════════════════════════════════════════════════════════════════════

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
