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

/// Parallel dispatch for group_matmul.
///
/// Contains:
///   - ALGO 1  (sequential_experts)
///   - ALGO 4  (parallel_multilevel)
///   - ALGO 5  (parallel_per_expert)
///   - ALGO 0  auto-select (select_grp_matmul_algo)
///   - group_matmul_run_parallel_dispatch  (the entry point)
///
/// ALGO 2 (M-tile) and ALGO 3 (N-tile) live in their own translation
/// units (group_matmul_m_tile.cpp, group_matmul_n_tile.cpp) and are
/// called through forward declarations in group_matmul_parallel_common.hpp.

#include <algorithm>
#include <climits>
#include <vector>

#include <omp.h>

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
  grp_matmul_gated_act_t fused_act, data_type_t act_dtype) {

  const int num_ops = static_cast<int>(M.size());
  if (num_ops == 0 || num_threads <= 0) {
    return;
  }

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
      if (i < num_ops) {
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
    const int batch = std::min(num_ops, num_ccds);

    scoped_active_levels guard(2);
    for (int round_start = 0; round_start < num_ops;
         round_start += batch) {
      const int round_end = std::min(num_ops, round_start + batch);
      const int round_size = round_end - round_start;

      #pragma omp parallel num_threads(round_size)
      {
        const int slot = omp_get_thread_num();
        if (slot < round_size) {
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
// Parallel-for over experts; each expert gets 1 thread.

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

  #pragma omp parallel for num_threads(num_threads)
  for (size_t i = 0; i < num_ops; ++i) {
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

// M-tile (ALGO 2) slices rows; N-indexed metadata passes through.
// M-indexed metadata (src_scale/zp {M,1} or {M,G}; 2D binary post-ops)
// is per-slice mutated in execute_m_tile (buff advance + dims[0]
// rewrite to slice_M).  Dynamic-quant additionally requires the
// source quant granularity to be row-local (M-indexed): the first
// dim of src_scale/zp must equal the per-expert `M[i]`, i.e.
// `{M[i], 1}` per-token or `{M[i], G}` per-group on K.  Per-tensor
// / per-column / per-channel src layouts (`{}`, `{1}`, `{1, K}`,
// `{1, N}`, etc.) are rejected because the per-thread reorder
// would race on the shared scale/zp buffer and produce slice-local
// statistics instead of the full-matrix statistics those
// granularities semantically require.  The single-row decode case
// (`M[i] == 1` with dims `{1, 1}`) is accepted — only one thread
// reorders that expert's one row, so there is no race and the
// single scale equals the full-matrix statistic trivially.  Still
// blocked: packed B (skipped GGML unpack), non-row-major layout,
// and per-expert dtype mismatches.
static bool check_m_tile_safe(
  const std::vector<char> &layout,
  const std::vector<int> &M,
  const std::vector<matmul_params> &params,
  int num_ops) {
  // Iterate the active range only: when the framework signals
  // `params[0].active_matmul > 0` the caller already trimmed the
  // matmul-processing count to the active prefix, but kept
  // `params[]` (and `layout[]`) at the framework's original size to
  // preserve weight-side prepack metadata at the tail.  Iterating
  // up to `params.size()` here would scan that tail and falsely
  // reject m-tile on a uniformity mismatch among non-fired slots.
  for (int i = 0; i < num_ops; ++i) {
    if (layout[i] != 'r' && layout[i] != 'R') {
      return false;
    }
    if (params[i].dtypes.src  != params[0].dtypes.src) {
      return false;
    }
    if (params[i].dtypes.wei  != params[0].dtypes.wei) {
      return false;
    }
    if (params[i].dtypes.dst  != params[0].dtypes.dst) {
      return false;
    }
    if (params[i].dtypes.bias != params[0].dtypes.bias) {
      return false;
    }
    if (params[i].mem_format_a != 'n') {
      return false;
    }
    if (params[i].mem_format_b != 'n') {
      return false;
    }
    if (params[i].packing.pack_format_b != 0) {
      return false;
    }
    // Dynamic-quant row-locality check.  The first dim of
    // src_scale/zp must equal this expert's `M[i]`:
    //
    //   * `{M[i], 1}`  — per-token (one scale per row), the
    //                    canonical dynamic-INT8 shape.
    //   * `{M[i], G}`  — per-group on K (one scale per row per
    //                    K-group), also row-local.
    //
    // For both shapes `offset_quant_by_row` cleanly splits the
    // scale buffer onto disjoint per-thread slices, and each
    // thread's reorder operates on its own rows in isolation.
    //
    // Rejected (first dim != M[i]):
    //
    //   * Per-tensor (`{}`, `{1}`, `{1, 1}` on M > 1) — every
    //     thread would write the same global slot.
    //   * Per-column / per-channel-on-src (`{1, K}`, `{1, N}`) —
    //     same shared-write race, and each slice would compute
    //     statistics over the wrong subset.
    //
    // Special case `M[i] == 1`: dims `{1, 1}` is BOTH per-tensor
    // and per-token (one scale = one row).  Accepted because the
    // M-tile executor only assigns one productive thread to a
    // one-row expert (`slice_M = 1`); no race, and the slice's
    // single-row max-abs IS the full-matrix max-abs.  This is the
    // common sparse-MoE decode pattern where some experts receive
    // exactly one token.
    if (params[i].dynamic_quant) {
      const auto &sd = params[i].quant_params.src_scale.dims;
      if (sd.empty() || sd[0] != static_cast<int64_t>(M[i])) {
        return false;
      }
      const auto &zd = params[i].quant_params.src_zp.dims;
      if (!zd.empty() && zd[0] != static_cast<int64_t>(M[i])) {
        return false;
      }
    }
    for (const auto &po : params[i].postop_) {
      if (po.po_type == post_op_type_t::softmax
          || po.po_type == post_op_type_t::pooling) {
        return false;
      }
    }
  }
  return true;
}

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
//     * Optional asymmetry: `src_zp` (if `buff` non-null) must
//       be `{M[i], 1}` per-token; `wei_zp` (if `buff` non-null)
//       must be per-channel `{N}` / `{1, N}` matching wei_scale.
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
//     A. Static source quantisation (`dynamic_quant == false`)
//        — regardless of src / wei granularity.
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
//   The custom BF16 microkernel is bf16×bf16→bf16 only and refuses
//   every non-bf16 combo at `prepare_for_call`, so int8 callers
//   always route through AOCL DLP int8 regardless of the
//   custom-kernel env flag.
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
  // We only accept the dynamic-quant form, so `buff` is expected
  // to be `nullptr` on entry — the pre-OMP hoist loop in
  // `flat_n_tile` allocates the internal scale buffer from the
  // dims.  A non-null `buff` would imply the caller pre-quantised
  // src, which is the static path the scope intentionally
  // excludes.
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
      // Dynamic source quant is the ONLY src-side shape we accept.
      // Static src (caller pre-quantised, `dynamic_quant == false`)
      // is rejected even when src/wei dims match the per-token /
      // per-channel layouts — keep the gate single-shape.
      if (!params[i].dynamic_quant) {
        return false;
      }

      // Source dims: `{M[i], 1}` per-token (including `M[i] == 1`
      // → `{1, 1}`, the single-token-per-expert decode case).
      // Buff is expected to be null (hoist allocates).  A non-null
      // buff here would be a caller pre-fill on a dynamic-quant
      // call — rare but not technically wrong; the wrapper will
      // overwrite it anyway.  We don't gate on buff here because
      // that nullness is a hoist-allocation contract, not a
      // per-token-scope contract.
      if (!is_per_token_dyn_src(qp.src_scale, M[i])) {
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
      // wei_zp: either symmetric (`buff == nullptr`) or per-channel
      // matching wei_scale's granularity.
      if (qp.wei_zp.buff != nullptr && !is_per_channel_wei(qp.wei_zp)) {
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
// (defaults: PROMPT=1 (sequential_experts) / DECODE=3 (N-tile + CK)
// — the measured-best out-of-the-box auto policy; set the env to
// `0` for the legacy cascade).
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
//   1. PHASE ENV — `max_M ≤ kDecodeMaxM` (decode) →
//                  `ZENDNNL_GRP_MATMUL_AUTO_DECODE_ALGO` (default 3)
//                  `max_M >  kDecodeMaxM` (prompt) →
//                  `ZENDNNL_GRP_MATMUL_AUTO_PROMPT_ALGO` (default 1)
//      When the active phase env is non-zero (the default cases),
//      that ALGO is returned directly with the same m_tile_safe /
//      n_tile_safe clamps the global ALGO env path applies in
//      `select_grp_matmul_algo`.  The defaults give a sensible
//      out-of-the-box auto policy: ALGO 1 (sequential_experts) for
//      prompt — the legacy auto-Rule-3 choice; same wall time as
//      the post-fix ALGO 3 path on skewed-M prompt until the
//      planner gains M-weighted multi-thread-per-expert, and lower
//      code surface — and ALGO 3 (N-tile rounds + CK) for decode,
//      the measured decode winner across every benchmarked MoE
//      workload.  Set the env to `0` for the legacy 3-rule cascade.
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
  (void)params;   // refinements that re-introduce shape/dtype tests.

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
  const int max_M = *std::max_element(M.begin(), M.end());
  const bool is_decode = (max_M <= kDecodeMaxM);
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
    const size_t wei_elem_b = size_of(params[0].dtypes.wei);
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
    const bool ck_hint =
        (use_algo == 3)
        && log_custom_kernel
        && (params[0].dtypes.src == data_type_t::bf16)
        && (params[0].dtypes.wei == data_type_t::bf16);
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

  switch (use_algo) {
  case 1:
    set_mode("sequential_experts");
    sequential_experts(layout, transA, transB, M, N, K, alpha,
                       src, lda, weight, ldb, bias, beta, dst, ldc,
                       is_weights_const, params, num_threads, fused_act, act_dtype);
    break;
  case 2:
    set_mode("flat_m_tile");
    flat_m_tile(layout, transA, transB, M, N, K, alpha,
                src, lda, weight, ldb, bias, beta, dst, ldc,
                fused_act, act_dtype, is_weights_const, params, num_threads);
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
    set_mode("multilevel");
    parallel_multilevel(layout, transA, transB, M, N, K, alpha,
                        src, lda, weight, ldb, bias, beta, dst, ldc,
                        is_weights_const, params, num_threads, fused_act, act_dtype);
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
