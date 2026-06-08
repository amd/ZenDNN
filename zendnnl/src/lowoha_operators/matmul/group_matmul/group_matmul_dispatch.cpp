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
#include <climits>
#include <vector>

#include <omp.h>

#include "m_tile/group_matmul_m_tile.hpp"  // flat_m_tile + M-tile env knobs
#include "n_tile/group_matmul_n_tile.hpp"  // flat_n_tile + N-tile env knobs
#include "group_matmul_parallel_common.hpp"
#include "prepack/prepack.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace {

// ── ALGO=1: sequential — serial over experts ────────────────────────────

void sequential_experts(
  const std::vector<char> &layout,
  const std::vector<bool> &transA, const std::vector<bool> &transB,
  const std::vector<int> &M, const std::vector<int> &N,
  const std::vector<int> &K, const std::vector<float> &alpha,
  const std::vector<const void *> &src, const std::vector<int> &lda,
  const std::vector<const void *> &weight, const std::vector<int> &ldb,
  const std::vector<const void *> &bias, const std::vector<float> &beta,
  const std::vector<void *> &dst, const std::vector<int> &ldc,
  const std::vector<bool> &is_weights_const,
  std::vector<matmul_params> &params,
  int num_threads,
  grp_matmul_gated_act_t fused_act, data_type_t act_dtype) {

  const size_t num_ops = M.size();
  if (num_ops == 0 || num_threads <= 0) {
    return;
  }

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
      group_matmul_prepack::build_prepack_params(
          weight, K, N, ldb, transB, is_weights_const, params, M,
          get_grp_matmul_custom_kernel(),
          num_threads, /*nr_align=*/0,
          fused_act, act_dtype,
          /*transA=*/&transA, /*alpha=*/&alpha, /*beta=*/&beta));

  matmul_algo_t algo = resolve_kernel();

  for (size_t i = 0; i < num_ops; ++i) {
    // Inactive experts (M<=0) are padded placeholder slots that may carry
    // null src/dst/weight pointers (validate_group_matmul_direct_inputs
    // allows null for M==0 because dispatch is expected to short-circuit
    // empty rows); skip them so the slice/activation calls never
    // dereference null.
    if (M[i] <= 0) continue;
    execute_expert_slice(layout[i], transA[i], transB[i],
                         M[i], N[i], K[i], alpha[i],
                         src[i], lda[i], weight[i], ldb[i],
                         bias[i], beta[i], dst[i], ldc[i],
                         is_weights_const[i], num_threads, params[i], algo);
    // Fused activation: dst[i] is hot in L3 from the GEMM that just finished.
    if (fused_act != grp_matmul_gated_act_t::none) {
        apply_gated_act_inplace(fused_act, dst[i], 0, M[i],
                                N[i], ldc[i], act_dtype);
    }
  }
}

// ── ALGO=4: multilevel — CCD-aware adaptive scheduling ──────────────────
//
// (A) Few experts, large M: multi-CCD per expert, all concurrent.
// (B) Few experts + small M, or many experts: round-based, 1 CCD each.
// Uses nested OMP (scoped_active_levels(2)).

void parallel_multilevel(
  const std::vector<char> &layout,
  const std::vector<bool> &transA, const std::vector<bool> &transB,
  const std::vector<int> &M, const std::vector<int> &N,
  const std::vector<int> &K, const std::vector<float> &alpha,
  const std::vector<const void *> &src, const std::vector<int> &lda,
  const std::vector<const void *> &weight, const std::vector<int> &ldb,
  const std::vector<const void *> &bias, const std::vector<float> &beta,
  const std::vector<void *> &dst, const std::vector<int> &ldc,
  const std::vector<bool> &is_weights_const,
  std::vector<matmul_params> &params,
  int num_threads,
  grp_matmul_gated_act_t fused_act, data_type_t act_dtype,
  const char **gemm_mode_out) {

  // The executor owns its gemm_mode and writes the concrete regime it ran
  // ("multilevel_concurrent" vs "multilevel_rounds") so the post-exec
  // [GRP_MATMUL.CALL] line reflects the real path.  No-op when nullptr.
  auto set_ml_mode = [&](const char *s) {
    if (gemm_mode_out != nullptr) *gemm_mode_out = s;
  };
  // Default to SKIP so a no-op early return (empty call / num_threads<=0)
  // reports exec_algo=0 rather than a real ALGO-4 run; the two regime
  // branches below overwrite it with the executed path.
  set_ml_mode("multilevel_skip");

  const int num_ops = static_cast<int>(M.size());
  if (num_ops == 0 || num_threads <= 0) {
    return;
  }

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
      group_matmul_prepack::build_prepack_params(
          weight, K, N, ldb, transB, is_weights_const, params, M,
          get_grp_matmul_custom_kernel(),
          num_threads, /*nr_align=*/0,
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
    if (total_M <= 0) {
      total_M = num_ops;
    }

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
        execute_expert_slice(layout[i], transA[i], transB[i],
                             M[i], N[i], K[i], alpha[i],
                             src[i], lda[i], weight[i], ldb[i],
                             bias[i], beta[i], dst[i], ldc[i],
                             is_weights_const[i], thr_per_op[i],
                             params[i], algo);
        if (fused_act != grp_matmul_gated_act_t::none) {
            apply_gated_act_inplace(fused_act, dst[i], 0, M[i],
                                    N[i], ldc[i], act_dtype);
        }
      }
    }
  }
  else {
    // (B) Round-based, 1 CCD per expert.
    set_ml_mode("multilevel_rounds");
    const int batch = std::min(num_ops, num_ccds);

    scoped_active_levels guard(2);
    for (int round_start = 0; round_start < num_ops;
         round_start += batch) {
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
          execute_expert_slice(layout[e], transA[e], transB[e],
                               M[e], N[e], K[e], alpha[e],
                               src[e], lda[e], weight[e], ldb[e],
                               bias[e], beta[e], dst[e], ldc[e],
                               is_weights_const[e], ccd_size, params[e], algo);
          if (fused_act != grp_matmul_gated_act_t::none) {
              apply_gated_act_inplace(fused_act, dst[e], 0, M[e],
                                      N[e], ldc[e], act_dtype);
          }
        }
      }
    }
  }
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

void parallel_per_expert(
  const std::vector<char> &layout,
  const std::vector<bool> &transA, const std::vector<bool> &transB,
  const std::vector<int> &M, const std::vector<int> &N,
  const std::vector<int> &K, const std::vector<float> &alpha,
  const std::vector<const void *> &src, const std::vector<int> &lda,
  const std::vector<const void *> &weight, const std::vector<int> &ldb,
  const std::vector<const void *> &bias, const std::vector<float> &beta,
  const std::vector<void *> &dst, const std::vector<int> &ldc,
  const std::vector<bool> &is_weights_const,
  std::vector<matmul_params> &params,
  int num_threads,
  grp_matmul_gated_act_t fused_act, data_type_t act_dtype) {

  const size_t num_ops = M.size();
  if (num_ops == 0 || num_threads <= 0) {
    return;
  }

  // Generic ahead-of-time weight pre-pack for ALGO 5.  `num_threads`
  // is forwarded so cross_warm can prefill regime 2 for the upcoming
  // ALGO 3 decode path when CUSTOM_KERNEL=0 (see the comment on the
  // ALGO 1 call site for the full rationale).
  group_matmul_prepack::prepack_for_algo_5(
      group_matmul_prepack::build_prepack_params(
          weight, K, N, ldb, transB, is_weights_const, params, M,
          get_grp_matmul_custom_kernel(),
          num_threads, /*nr_align=*/0,
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
    if (M[i] <= 0) continue;
    execute_expert_slice(layout[i], transA[i], transB[i],
                         M[i], N[i], K[i], alpha[i],
                         src[i], lda[i], weight[i], ldb[i],
                         bias[i], beta[i], dst[i], ldc[i],
                         is_weights_const[i], 1, params[i], algo);
    if (fused_act != grp_matmul_gated_act_t::none) {
        apply_gated_act_inplace(fused_act, dst[i], 0, M[i],
                                N[i], ldc[i], act_dtype);
    }
  }
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
//       ALGO 3 in two cases — `num_ops >= num_threads` (Qwen-style)
//       and the M-driven decode arrow (`max_M <= kDecodeMaxM`).
//       Both honour `n_tile_safe`; on quantised inputs that fall
//       outside the supported sub-set below, both arrows collapse
//       to ALGO 1.  (Rule 0's capacity carve-out routes
//       `num_ops > kNTilePlanMaxExperts` to ALGO 5 before either
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
static bool check_n_tile_extra(
  const std::vector<int> &M,
  const std::vector<matmul_params> &params,
  int num_ops) {
  // Per-channel weight side: dims must be exactly `{N}` (rank-1) or
  // `{1, N}` (rank-2 with broadcast outer dim).  `buff` must be
  // non-null (wei is always statically quantised by the caller).
  // The column slice is a contiguous `n_tile`-long sub-array —
  // handled by `offset_quant_by_col` in `do_tile`.
  auto is_per_channel_wei =
      [](const matmul_quantization_params_t::matmul_quant_t &q) -> bool {
    if (q.buff == nullptr) return false;
    if (q.dims.size() == 1 && q.dims[0] > 1) return true;
    if (q.dims.size() == 2 && q.dims[0] == 1 && q.dims[1] > 1) return true;
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
  auto is_per_token_dyn_src =
      [](const matmul_quantization_params_t::matmul_quant_t &q,
         int M_expert) -> bool {
    return q.dims.size() == 2
        && q.dims[0] == static_cast<int64_t>(M_expert)
        && q.dims[1] == 1;
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
    if (M[i] == 0) continue;
    const auto &qp = params[i].quant_params;

    // Detect any quant intent on this expert.  If every quant field
    // is empty AND `dynamic_quant` is false, this is a pure
    // non-quantised call and there's nothing to gate — ALGO 3 is
    // free to run.
    const bool any_quant =
        qp.wei_scale.buff != nullptr ||
        qp.wei_zp.buff   != nullptr ||
        qp.src_scale.buff != nullptr ||
        qp.src_zp.buff   != nullptr ||
        params[i].dynamic_quant;

    if (any_quant) {
      // Source side: accept either (a) dynamic BF16/F32 input that
      // flat_n_tile will hoist, or (b) already grouped-quantized S8
      // input with a ready per-token source scale buffer.
      const bool grouped_s8_src =
          !params[i].dynamic_quant &&
          params[i].dtypes.src == data_type_t::s8 &&
          qp.src_scale.buff != nullptr;
      if (!params[i].dynamic_quant && !grouped_s8_src) {
        return false;
      }

      // Source dims: `{M[i], 1}` per-token (including `M[i] == 1`
      // → `{1, 1}`, the single-token-per-expert decode case).
      // For grouped_s8_src the scale buffer must be non-null (checked
      // above). For dynamic input, nullness is a hoist-allocation
      // contract, not a per-token-scope contract.
      if (!is_per_token_dyn_src(qp.src_scale, M[i])) {
        return false;
      }
      if (grouped_s8_src && qp.src_zp.buff != nullptr) {
        return false;
      }
      if (qp.src_zp.buff != nullptr &&
          !is_per_token_dyn_src(qp.src_zp, M[i])) {
        return false;
      }

      // Weight side: exactly per-channel `{N}` / `{1, N}` with a
      // non-null buff.  Per-tensor and per-group `{G, N}` wei are
      // both rejected.
      if (!is_per_channel_wei(qp.wei_scale)) {
        return false;
      }
      // Weight zero-point is NOT supported anywhere on the N-tile
      // DQ-INT8 path: the CK microkernel assumes symmetric weights
      // (its compensation row only folds the src +128 / src_zp bias),
      // and the AOCL sym-quant fallback is likewise symmetric.  A
      // non-null wei_zp must therefore reject ALGO 3 entirely so the
      // call falls back to ALGO 1 (general AOCL DLP) instead of
      // silently dropping the weight zero-point.
      if (qp.wei_zp.buff != nullptr) {
        return false;
      }
    }

    // Buffer-bearing post-ops (binary_add / binary_mul) may carry
    // N-indexed layouts (`{N}`, `{1, N}`, `{M, N}`) that need the
    // same column-slice treatment as wei_scale.  The slicer is not
    // yet wired for post-ops; keep them rejected.  Buffer-free
    // elementwise post-ops (gelu, relu, swish, …) have null `buff`
    // and pass through.
    for (const auto &po : params[i].postop_) {
      if (po.buff != nullptr) {
        return false;
      }
    }
  }
  return true;
}

// Auto-select (ALGO 0) heuristic — used when the caller leaves
// ZENDNNL_GRP_MATMUL_ALGO unset.  Picks between {ALGO 1, ALGO 3,
// ALGO 5} via the legacy 3-rule cascade, OR pins to a specific
// ALGO per phase via the AUTO_PROMPT_ALGO / AUTO_DECODE_ALGO envs
// (defaults: PROMPT=2 (flat_m_tile + multi-tier hybrid + wide-N
// fallback) / DECODE=3 (N-tile + CK) — the out-of-the-box auto
// policy; set the env to `0` for the legacy cascade, or `1` to
// pin the legacy sequential_experts prompt path).
//
// Decision precedence (tightest first):
//
//   0. STRUCTURAL — num_ops > kNTilePlanMaxExperts (=256) → ALGO 5
//      Capacity carve-out: the N-tile planner's R3 gate rejects
//      calls beyond `GroupNTilePlan::kMaxExperts` and silently
//      falls back to its Sequential strategy (one expert at a
//      time, full thread team each).  Sequential is materially
//      slower than ALGO 5 (per-expert parallel, dynamic OMP
//      schedule) on every num_ops > 256 shape we have measured —
//      e.g., a hypothetical 300-expert MoE on 128 threads would
//      run ~300 serial full-team matmuls instead of ~3 waves of
//      128 parallel per-expert tasks.  Phase env cannot override
//      this — the planner's R3 gate is structural.  ALGO 5 has no
//      m_tile/n_tile safety dependency so it covers unsafe paths too.
//
//   0.6. DECODE, active_ops > num_threads → ALGO 5 (parallel_per_expert).
//      Decode-class only (`max_M ≤ kDecodeMaxM`).  `active_ops` counts the
//      experts that actually fire (`M[i] > 0`), not the padded slot count.
//      When more experts fire
//      than there are threads, ALGO 3's decode-optimal single round is
//      infeasible (needs one thread per active expert) and degrades to a
//      multi-round schedule; the flat per-expert path matches or beats it
//      on the measured Qwen-class shapes (e.g. ~88 active experts on 64
//      threads).  Fires REGARDLESS of an `AUTO_DECODE_ALGO` pin (it
//      supersedes Rule 1 for this regime); only the global
//      `ZENDNNL_GRP_MATMUL_ALGO` force overrides it.  Prompt is never
//      routed by THIS rule — it follows its own policy (Rule 1/2 +
//      safety clamps), typically ALGO 2 when m_tile_safe.
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
//      wide-N fallback) for prompt — the M-tile multi-tier captures
//      the Qwen3-class skewed-M speedup, while the wide-N fallback
//      keeps Mixtral / GPT-OSS light frames at parity with the
//      legacy ALGO 1 path; safety clamp falls back to ALGO 1 when
//      `!m_tile_safe` — and ALGO 3 (N-tile rounds + CK) for decode,
//      the measured decode winner across every benchmarked MoE
//      workload.  Set `AUTO_PROMPT_ALGO=1` to restore the legacy
//      sequential_experts prompt path, or the env to `0` for the
//      legacy 3-rule cascade.
//
//   2. LEGACY RULES (phase env == 0):
//
//      a. num_ops ≥ num_threads               → ALGO 3
//         (Qwen3-30B-A3B-class: 128 experts on 64-128t hosts.  At
//          this expert/thread ratio every expert sees a thin per-
//          expert team and N-tile's round-based scheduling
//          consistently outperforms ALGO 1's serial-experts-with-
//          full-team approach.  Honors n_tile_safe — quantised paths
//          fall back to ALGO 1.)
//
//      b. num_ops ≤ kFewExpertsAlgo1 (=8)     → ALGO 1
//         (Mixtral-8x*-class: 8 experts.  Per-expert weight footprint
//          is large enough that the full-weight AOCL DLP cache key
//          + serial expert iteration amortises DRAM traffic better
//          than N-tile's per-thread column slices on a thin per-
//          expert team.)
//
//      c. otherwise (9 ≤ num_ops < num_threads) — M-driven:
//           prompt (max_M >  kDecodeMaxM)     → ALGO 1
//           decode (max_M ≤  kDecodeMaxM)     → ALGO 3
//         (gpt-oss-20B-class: ops typically 9..32 on 64-128t hosts.
//          Prompt uses ALGO 1's thread-count-stable full-weight cache
//          key; decode uses ALGO 3's custom-kernel + per-tile path
//          which is the measured win on the MoE decode hot path.
//          N-tile's internal Sequential-strategy fallback handles
//          narrow-N shapes where the planner can't satisfy
//          `tiles_per_expert ≥ min`.)
//
// The historical large-weight wide-N prompt carve-out and weight-class
// branching are intentionally dropped — the simpler M-driven default
// preserves gpt-oss prompt routing, gives Mixtral and Qwen explicit
// per-arch arrows, and the auto-selector now reads as a 3-rule table.
// Callers that need a non-default decision on a specific deployment
// can still pin via `ZENDNNL_GRP_MATMUL_ALGO` (global pin) or via
// `ZENDNNL_GRP_MATMUL_AUTO_{PROMPT,DECODE}_ALGO` (per-phase pin while
// keeping the global env unset / 0).
static int auto_select_algo(
  const std::vector<int> &M,
  const std::vector<int> &N,
  const std::vector<int> &K,
  const std::vector<matmul_params> &params,
  int num_threads,
  bool m_tile_safe,
  bool n_tile_safe) {
  (void)N;        // Kept in the signature for symmetry with the M-tile
  (void)K;        // / N-tile safety helpers and to ease future heuristic
                  // refinements that re-introduce shape/dtype tests.

  const int num_ops = static_cast<int>(M.size());
  if (num_threads <= 1 || num_ops == 0) {
    return 1;
  }

  // Rule 0 — STRUCTURAL capacity carve-out (ignores phase env).
  // Placed before the phase env so it catches every shape that would
  // otherwise reach the N-tile planner's R3 Sequential fallback.
  if (num_ops > kNTilePlanMaxExperts) {
    return 5;
  }

  const int max_M = *std::max_element(M.begin(), M.end());
  const bool is_decode = (max_M <= kDecodeMaxM);

  // Rule 0.5 — FEW-EXPERTS ALGO 2 PREFERENCE (Mixtral-class default).
  // TOTAL expert count (framework `total_matmul` when set, else the
  // active op count) ≤ kFewExpertsAlgo2Pref → ALGO 2 for BOTH phases.
  // ALGO 2 (flat_m_tile) is the measured winner on Mixtral-8x* prompt
  // AND decode, so fold that benefit into the out-of-the-box AUTO
  // policy.  This is a DEFAULT refinement: it fires only when the
  // phase env relevant to THIS call is NOT explicitly pinned — an
  // explicit `AUTO_{PROMPT,DECODE}_ALGO` (including `=0` for the legacy
  // cascade) still wins via Rule 1 below.  Honours the m_tile_safe
  // clamp (non-row-major / dtype-mismatch / unsafe quant → ALGO 1).
  // `total_matmul` is only meaningful under the framework opt-in
  // (`active_matmul > 0`); a legacy caller may leave it stale, so read it
  // only then and only when it exceeds the active count (padded/extras
  // layout).  Otherwise the active op count IS the total.
  int total_experts = num_ops;
  if (!params.empty() && params[0].active_matmul > 0 &&
      params[0].total_matmul > static_cast<uint32_t>(total_experts)) {
    total_experts = static_cast<int>(params[0].total_matmul);
  }
  const bool phase_env_pinned = is_decode
      ? grp_matmul_auto_decode_algo_is_set()
      : grp_matmul_auto_prompt_algo_is_set();
  // DECODE-ONLY now: ALGO 2 (flat_m_tile) is the measured Mixtral-class
  // DECODE winner.  For PROMPT, the few-experts case is handled by the
  // prompt M-tile regime routing (Rule 0.7) below — which may peel a
  // light Mixtral prompt frame off to ALGO 1 (wide-N) exactly as
  // flat_m_tile's internal wide-N fallback used to.  Restricting Rule 0.5
  // to decode preserves that prompt behaviour now that ALGO 2 is pure.
  //
  // CONSISTENCY: ALGO 2 is now a PURE M-tile executor — it no longer has
  // the internal wide-N fallback that the old "ALGO 2 wins Mixtral decode"
  // measurement relied on.  For shallow-M decode (`M[i] < team_size`)
  // single-tier M-tile under-fills the team (it splits M rows, capped at
  // M[i]), whereas the old wide-N branch gave the full team an N-split.
  // So apply the SAME regime classification Rule 0.7 uses for prompt, so
  // ALGO 0 reproduces the old flat_m_tile internal routing for decode too:
  //   * kWideN (shallow M) → ALGO 1 (full-team sequential = old wide-N).
  //   * kMTile             → ALGO 2 (single-tier / multi-tier — old deep-M
  //                          and max_M==1 behaviour, unchanged).
  // (kManyExperts can't fire here: total_experts ≤ kFewExpertsAlgo2Pref(8)
  //  ≤ num_threads on any real host, so active_ops ≤ num_threads.)
  if (is_decode && total_experts <= kFewExpertsAlgo2Pref && !phase_env_pinned) {
    switch (classify_m_tile_regime(M, num_threads)) {
      case m_tile_regime::kManyExperts: return 5;
      case m_tile_regime::kWideN:       return 1;
      case m_tile_regime::kMTile:       return m_tile_safe ? 2 : 1;
    }
  }

  // Rule 0.6 — DECODE with MORE ACTIVE EXPERTS THAN THREADS → ALGO 5.
  // The count compared here is the ACTIVE-COMPUTE expert count
  // `active_ops = |{ i : M[i] > 0 }|` — the experts that actually fire this
  // call — NOT `M.size()` and NOT the framework `total_matmul` expert pool.
  // The distinction matters: framework opt-in callers pass M already sliced
  // to the contiguous active set (so `active_ops == M.size()` there), but a
  // legacy caller may pass a padded vector with `M[i]==0` placeholders
  // in-range (dispatch short-circuits those empty rows), and only the
  // M[i]>0 experts consume a thread.  It is this active count that drives
  // ALGO 3's per-expert thread budget.  When active experts exceed the
  // thread count (`active_ops > num_threads`), ALGO 3's decode-optimal
  // ManyExperts single round is infeasible (single round needs one thread
  // per active expert, i.e. `num_threads >= active_ops`) and it degrades to
  // a multi-round schedule that benchmarks no better than the flat per-
  // expert path on the measured Qwen-class decode shapes (e.g. Qwen3-128
  // decode with ~88 active experts on a 64-core host).  ALGO 5
  // (parallel_per_expert) is a single flat, non-nested `omp parallel for
  // schedule(dynamic)` over experts — no N-split, no round barriers — which
  // fully occupies the team (~active_ops/num_threads experts each) and lets
  // the dynamic schedule balance the M-skew.  When `num_threads >=
  // active_ops` (e.g. the same call on 128 cores) the rule does not fire and
  // ALGO 3's single round stays selected via Rule 1.  ALGO 5 has no tiling-
  // safety dependency (BLAS handles every layout/dtype), so no clamp.
  //
  // DECODE ONLY: the `is_decode` guard keeps prompt out of this rule (a prior
  // version without it could route an unpinned, many-expert prompt frame to
  // ALGO 5).  Prompt is compute-bound on large M, so it follows its own
  // policy (Rule 1/2 + safety clamps) — typically ALGO 2 (M-tile) when
  // m_tile_safe, but it can still clamp to ALGO 1 or honour an explicit pin.  This rule fires REGARDLESS of an explicit
  // `AUTO_DECODE_ALGO` pin — in the experts-exceed-threads regime the
  // per-expert path is the policy and an explicit decode pin no longer wins
  // here (use the global `ZENDNNL_GRP_MATMUL_ALGO` force, applied before
  // `auto_select_algo`, to override a specific call).
  if (is_decode) {
    const int active_ops =
        static_cast<int>(std::count_if(M.begin(), M.end(),
                                        [](int m) { return m > 0; }));
    if (active_ops > num_threads) {
      return 5;
    }
  }

  // Rule 0.7 — PROMPT M-tile REGIME routing.  ALGO 2 (flat_m_tile) is now a
  // PURE M-tile executor (multi-tier hybrid + Phase-2 single-tier); the two
  // non-M-tile regimes it used to handle via internal fallbacks are routed
  // to the dedicated algos HERE, at selection time, so AUTO reproduces the
  // exact executor flat_m_tile picked internally before the cleanup:
  //   * kManyExperts (active_ops > num_threads) → ALGO 5 (per-expert): a
  //     pure M-tile plan cannot give < 1 thread/active-expert, so this
  //     regime is M-tile-infeasible.  (Equivalent to the old round-based
  //     branch, which was itself a flat schedule(dynamic) per-expert pool.)
  //   * kWideN (shallow M: max_M>1 && total_need*2 ≤ num_threads) → ALGO 1
  //     (sequential full-team): M is too shallow to feed the M-tile slicer;
  //     the whole team streams each expert's weight once (the old wide-N
  //     fallback's behaviour).
  //   * kMTile → ALGO 2: the genuine M-tile regime (multi-tier or single-
  //     tier is then chosen INSIDE flat_m_tile).
  // Scope: applies only when the resolved prompt algo is the M-tile-family
  // default (phase algo == 2 — i.e. AUTO_PROMPT_ALGO unset → default 2, or
  // an explicit `=2` pin).  An explicit non-2 prompt pin (1/3/4/5) or the
  // `=0` legacy escape hatch is honoured via Rule 1 / Rule 2 below.  Decode
  // is unaffected (ALGO 3 default + Rule 0.6).  The classifier's gates
  // mirror flat_m_tile's old internal gates exactly (same kSliceTarget),
  // so the routing is parity-preserving.
  if (!is_decode && get_grp_matmul_auto_prompt_algo() == 2) {
    switch (classify_m_tile_regime(M, num_threads)) {
      case m_tile_regime::kManyExperts: return 5;
      case m_tile_regime::kWideN:       return 1;
      case m_tile_regime::kMTile:       return m_tile_safe ? 2 : 1;
    }
  }

  // Rule 1 — PHASE ENV.  Single-line phase classification (decode iff
  // `max_M ≤ kDecodeMaxM`) drives which env is consulted.  When the
  // active phase env is non-zero the operator has explicitly pinned
  // that algo for the phase — return it directly, with the same
  // m_tile_safe / n_tile_safe correctness clamps the global ALGO env
  // path applies.  Non-tile-safe + ALGO 3 falls to ALGO 1; non-m-tile-
  // safe + ALGO 2 falls to ALGO 1; the clamps are silent here because
  // the matching `[GRP_MATMUL.ALGO WARN]` apilog already fires from
  // `select_grp_matmul_algo`'s safety branch when env_algo asks for
  // the same algo on the same unsafe shape — emitting the WARN twice
  // would be confusing.  Operators see the clamp via the
  // `[GRP_MATMUL.ALGO]` line's `chosen=ALGO_X reason=auto_phase_env_clamp`.
  const int phase_algo = is_decode
      ? get_grp_matmul_auto_decode_algo()
      : get_grp_matmul_auto_prompt_algo();
  if (phase_algo >= 1 && phase_algo <= 5) {
    if (phase_algo == 2 && !m_tile_safe) return 1;
    if (phase_algo == 3 && !n_tile_safe) return 1;
    return phase_algo;
  }

  // Rule 2 — LEGACY RULES (phase env == 0).
  //
  // 2a. num_ops ≥ num_threads (Qwen-style).  Highest of the three
  //     legacy rules so an 8-expert deployment on a ≤ 8-thread host
  //     (rare but possible for local dev / single-CCD profiling)
  //     routes here, not to rule 2b.
  //
  // SCOPE NOTE — N-tile viability NOT consulted by design.
  //   The previous heuristic gated rule-1-like cases on
  //   `tiles_per_expert ≥ min_ntiles`.  The new rule deliberately
  //   skips that check: the N-tile planner's `ntile_viable` runs
  //   anyway as part of `plan_group_n_tile`.  Since the
  //   `N_TILE_STRATEGY=2` (rounds, default) fix to the planner,
  //   `!viable` no longer demotes to Sequential under force_ntile —
  //   it stays on rounds with a `[GRP_MATMUL.PLAN.HINT]` line.
  //   Under `n_tile_strategy=0` (auto) the planner still uses
  //   viability as a perf hint.
  if (num_ops >= num_threads) {
    return n_tile_safe ? 3 : 1;
  }

  // 2b. num_ops ≤ kFewExpertsAlgo1 (Mixtral-style).
  if (num_ops <= kFewExpertsAlgo1) {
    return 1;
  }

  // 2c. M-driven default (prompt → ALGO 1, decode → ALGO 3).
  // The decode arrow does NOT consult N-tile viability for the same
  // reason rule 2a doesn't — see the SCOPE NOTE on rule 2a above.
  if (!is_decode) {
    return 1;
  }
  return n_tile_safe ? 3 : 1;
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

int select_grp_matmul_algo(
  const std::vector<char> &layout,
  const std::vector<int> &M,
  const std::vector<int> &N,
  const std::vector<int> &K,
  const std::vector<matmul_params> &params,
  int num_threads) {

  // `M.size()` is the active matmul count after `group_matmul_direct`
  // sliced the M vector to honour `params[0].active_matmul`.  Pass it
  // explicitly so the safety helpers iterate only the active slots
  // rather than `params.size()` (which still carries the framework's
  // prepack-extras tail).
  const int num_ops_eff = static_cast<int>(M.size());
  const bool m_tile_safe = check_m_tile_safe(layout, M, params, num_ops_eff);
  const bool n_tile_safe = m_tile_safe
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
            "REJECTED: m_tile unsafe (non-row-major, per-expert dtype "
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
            "[GRP_MATMUL Level2 dispatch WARN] env_algo=3 (flat_n_tile) "
            "REJECTED: n_tile unsafe.  Common rejection reasons: "
            "non-row-major layout, per-expert dtype mismatch, "
            "buffer post-op, or a quant configuration outside the "
            "per-token dynamic-INT8 scope.  ALGO 3 currently "
            "accepts ONE quant shape: `dynamic_quant=true` with "
            "`{M[i], 1}` src (the single-row decode case "
            "`{1, 1}` when M[i]=1 is included) + per-channel "
            "`{1, N}` wei (statically quantised wei buff supplied "
            "by caller).  Static src, per-tensor src/wei, "
            "per-group `{M[i], G}` src, per-group `{G, N}` wei, "
            "and pure WOQ workloads stay on ALGO 1.  See "
            "`check_n_tile_extra` SCOPE NOTE for the full table.  "
            "FALLBACK algo=1 (sequential_experts).");
      }
      algo = 1;
    }
    return algo;
  }

  return auto_select_algo(M, N, K, params, num_threads,
                          m_tile_safe, n_tile_safe);
}

// ── Dispatch ────────────────────────────────────────────────────────────

bool group_matmul_run_parallel_dispatch(
  const std::vector<char> &layout,
  const std::vector<bool> &transA,
  const std::vector<bool> &transB,
  const std::vector<int> &M,
  const std::vector<int> &N,
  const std::vector<int> &K,
  const std::vector<float> &alpha,
  const std::vector<const void *> &src,
  const std::vector<int> &lda,
  const std::vector<const void *> &weight,
  const std::vector<int> &ldb,
  const std::vector<const void *> &bias,
  const std::vector<float> &beta,
  const std::vector<void *> &dst,
  const std::vector<int> &ldc,
  const std::vector<bool> &is_weights_const,
  std::vector<matmul_params> &params,
  const int num_threads,
  const char **gemm_mode_out,
  grp_matmul_gated_act_t fused_act,
  data_type_t act_dtype) {

  const int use_algo = select_grp_matmul_algo(layout, M, N, K, params,
                       num_threads);

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
  const bool caller_layout_tight = (use_algo == 3)
                                   && !ldc.empty() && !N.empty() && ldc[0] < N[0];
  // Wide-fused (caller's ldc ≥ N) routes through the standard
  // backend's `apply_swiglu_oai_tile_rows`; that helper handles
  // swiglu_oai_mul only.  silu_and_mul and gelu_and_mul have no
  // wide-helper siblings yet, so they can only fuse on the tight
  // layout (CK path).  `a3_can_fuse_act` already gates silu/gelu
  // on `use_custom_kernel=true` — combined with this tight-only
  // gate, the silu/gelu fused path engages exclusively when
  // (CK-on AND tight caller).  Wide non-CK silu/gelu falls through
  // to the dispatcher's separate-pass post-pass.
  const bool wide_fuse_supported =
      (fused_act == grp_matmul_gated_act_t::swiglu_oai_mul)
      && get_grp_n_tile_fused_act();
  const bool a3_fuses = (use_algo == 3)
                        && a3_can_fuse_act(fused_act,
                                           get_grp_matmul_custom_kernel())
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
      if (M[i] > 0) { rep = i; break; }
    }
    const size_t wei_elem_b = size_of(params[rep].dtypes.wei);
    const size_t wei_per_expert_mb =
        (static_cast<size_t>(max_K_v) * max_N_v * wei_elem_b) >> 20;
    // Phase + per-phase env values for telemetry.  The phase
    // classification mirrors `auto_select_algo`'s phase gate so the
    // log reflects the routing decision the planner actually made.
    const bool is_decode = (max_M_v <= kDecodeMaxM);
    const int phase_env_prompt = get_grp_matmul_auto_prompt_algo();
    const int phase_env_decode = get_grp_matmul_auto_decode_algo();
    const int phase_env_active = is_decode ? phase_env_decode
                                           : phase_env_prompt;
    // Reason hierarchy — surfaces which gate drove the chosen ALGO.
    // ORDER MUST MIRROR `auto_select_algo`'s precedence so the log
    // line reflects the actual decision path:
    //
    //   1. env_ok / env_fallback   — global `ZENDNNL_GRP_MATMUL_ALGO`
    //                                hit OR safety-clamped (clamp
    //                                emits a [WARN] line too).
    //   2. auto_single_thread      — `auto_select_algo`'s
    //                                `num_threads <= 1 || num_ops == 0`
    //                                early-exit branch (returns 1
    //                                before any other rule).
    //   3. auto_rule0_capacity     — `num_ops > kNTilePlanMaxExperts`
    //                                → ALGO 5 (structural).
    //   3b. auto_decode_ops_gt_threads — decode-class call with
    //                                `active_ops > num_threads` → ALGO 5
    //                                (Rule 0.6; `active_ops = count(M[i]>0)`).
    //                                Fires before the phase
    //                                env so it is labelled distinctly even
    //                                when an `AUTO_DECODE_ALGO` pin is set
    //                                (which it overrides).
    //   4. auto_phase_env*         — `ZENDNNL_GRP_MATMUL_AUTO_*_ALGO`
    //                                non-zero AND honoured (`_clamp`
    //                                suffix when the m_tile_safe /
    //                                n_tile_safe clamp downgraded to
    //                                ALGO 1).
    //   5. auto_rule_legacy        — fell through to the legacy 3-rule
    //                                cascade (phase env explicitly =0).
    const char *reason = nullptr;
    if (env_algo >= 1 && env_algo <= 5) {
      reason = (env_algo == use_algo) ? "env_ok" : "env_fallback";
    } else if (num_threads <= 1 || M.empty()) {
      reason = "auto_single_thread";
    } else if (static_cast<int>(M.size()) > kNTilePlanMaxExperts) {
      reason = "auto_rule0_capacity";
    } else if (is_decode &&
               static_cast<int>(std::count_if(
                   M.begin(), M.end(), [](int m) { return m > 0; }))
                   > num_threads) {
      reason = "auto_decode_ops_gt_threads";
    } else if (phase_env_active >= 1 && phase_env_active <= 5) {
      reason = (phase_env_active == use_algo) ? "auto_phase_env"
                                              : "auto_phase_env_clamp";
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
    const bool ck_hint_bf16 =
        (use_algo == 3)
        && log_custom_kernel
        && (params[rep].dtypes.src == data_type_t::bf16)
        && (params[rep].dtypes.wei == data_type_t::bf16)
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
    const bool ck_int8_shapes =
        (use_algo == 3)
        && log_custom_kernel
        && log_custom_kernel_int8
        && (params[rep].dtypes.wei == data_type_t::s8)
        && (params[rep].dtypes.dst == data_type_t::bf16)
        && (params[rep].dtypes.compute == data_type_t::s8
            || params[rep].dtypes.compute == data_type_t::u8);
    const bool ck_hint_int8 =
        ck_int8_shapes
        && ((params[rep].dynamic_quant
             && params[rep].dtypes.src == data_type_t::bf16)
            || (params[rep].dtypes.src == data_type_t::s8
                && params[rep].quant_params.src_scale.buff != nullptr));
    const bool ck_hint = ck_hint_bf16 || ck_hint_int8;
    const char *ck_family =
        ck_hint_bf16  ? "bf16"
      : ck_hint_int8  ? (params[rep].dtypes.compute == data_type_t::u8
                            ? "int8_asym" : "int8_sym")
      :                 "none";
    // SELECTION record (emitted BEFORE the executor runs): `chosen=ALGO_X`
    // is the algo the selector picked, with `reason` explaining the gate.
    // The ALGO that ACTUALLY executed — including any in-executor clamp or
    // fork (e.g. ALGO 2 -> sequential-full-team when active_ops>num_threads,
    // or ALGO 2 -> vertical fusion in the fused path) — is reported by the
    // post-exec `[GRP_MATMUL.CALL]` line via `mode=` (precise branch) and
    // `exec_algo=` (the real 1..5).  Compare those two lines to see any
    // selection-vs-execution divergence.
    apilog_info(
        "[GRP_MATMUL.ALGO] chosen=ALGO_", use_algo,
        " env_algo=", env_algo,
        " reason=", reason,
        " phase=", (is_decode ? "decode" : "prompt"),
        " auto_prompt_env=", phase_env_prompt,
        " auto_decode_env=", phase_env_decode,
        " act=", act_name(fused_act),
        " act_fused=", (act_fused ? "yes" : "no"),
        " ck_eligible_hint=", (ck_hint ? "yes" : "no"),
        " ck_family=", ck_family,
        " dynamic_quant=", (params[rep].dynamic_quant ? "yes" : "no"),
        " num_ops=", static_cast<int>(M.size()),
        " num_threads=", num_threads,
        " max_M=", max_M_v,
        " max_N=", max_N_v,
        " max_K=", max_K_v,
        " wei/expert(MB)=", wei_per_expert_mb,
        " wide_N=", (max_N_v > max_K_v ? "yes" : "no"),
        " many_experts=",
        (static_cast<int>(M.size()) >= 16 ? "yes" : "no"),
        " caller_tight=", (caller_layout_tight ? "yes" : "no"));
  }

  auto set_mode = [&](const char *s) {
    if (gemm_mode_out != nullptr) {
      *gemm_mode_out = s;
    }
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

  switch (use_algo) {
  case 1:
    set_mode("sequential_experts");
    sequential_experts(layout, transA, transB, M, N, K, alpha,
                       src, lda, weight, ldb, bias, beta, dst, ldc,
                       is_weights_const, params, num_threads, fused_act, act_dtype);
    break;
  case 2:
    // flat_m_tile owns its gemm_mode — it writes the concrete branch it ran
    // (flat_m_tile_multitier / _single_tier / _seq_clamp) into gemm_mode_out,
    // like flat_n_tile does, so the post-exec [GRP_MATMUL.CALL] line reflects
    // the real M-tile path rather than a generic "flat_m_tile".
    flat_m_tile(layout, transA, transB, M, N, K, alpha,
                src, lda, weight, ldb, bias, beta, dst, ldc,
                fused_act, act_dtype, is_weights_const, params, num_threads,
                gemm_mode_out);
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
    flat_n_tile(layout, transA, transB, M, N, K, alpha,
                src, lda, weight, ldb, bias, beta, dst, ldc,
                is_weights_const, params, num_threads,
                a3_fuses ? fused_act : grp_matmul_gated_act_t::none,
                act_dtype, gemm_mode_out);
    break;
  case 4:
    // parallel_multilevel owns its gemm_mode (multilevel_concurrent /
    // multilevel_rounds), written into gemm_mode_out.
    parallel_multilevel(layout, transA, transB, M, N, K, alpha,
                        src, lda, weight, ldb, bias, beta, dst, ldc,
                        is_weights_const, params, num_threads, fused_act, act_dtype,
                        gemm_mode_out);
    break;
  case 5:
    set_mode("per_expert");
    parallel_per_expert(layout, transA, transB, M, N, K, alpha,
                        src, lda, weight, ldb, bias, beta, dst, ldc,
                        is_weights_const, params, num_threads, fused_act, act_dtype);
    break;
  default:
    set_mode("sequential_experts");
    sequential_experts(layout, transA, transB, M, N, K, alpha,
                       src, lda, weight, ldb, bias, beta, dst, ldc,
                       is_weights_const, params, num_threads, fused_act, act_dtype);
    break;
  }
  return act_fused;
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
