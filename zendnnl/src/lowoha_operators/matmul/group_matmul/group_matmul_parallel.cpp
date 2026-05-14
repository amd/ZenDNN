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
          fused_act, act_dtype));

  matmul_algo_t algo = resolve_kernel();

  for (size_t i = 0; i < num_ops; ++i) {
    execute_expert_slice(layout[i], transA[i], transB[i],
                         M[i], N[i], K[i], alpha[i],
                         src[i], lda[i], weight[i], ldb[i],
                         bias[i], beta[i], dst[i], ldc[i],
                         is_weights_const[i], num_threads, params[i], algo);
    // Fused activation: dst[i] is hot in L3 from the GEMM that just finished.
    if (fused_act != grp_matmul_gated_act_t::none)
      apply_gated_act_inplace(fused_act, dst[i], 0, M[i], N[i], ldc[i],
                              act_dtype);
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
          fused_act, act_dtype));

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
        if (fused_act != grp_matmul_gated_act_t::none)
          apply_gated_act_inplace(fused_act, dst[i], 0, M[i],
                                  N[i], ldc[i], act_dtype);
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
          if (fused_act != grp_matmul_gated_act_t::none)
            apply_gated_act_inplace(fused_act, dst[e], 0, M[e],
                                    N[e], ldc[e], act_dtype);
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
          fused_act, act_dtype));

  matmul_algo_t algo = resolve_kernel();
  scoped_active_levels guard(1);

  #pragma omp parallel for num_threads(num_threads)
  for (size_t i = 0; i < num_ops; ++i) {
    execute_expert_slice(layout[i], transA[i], transB[i],
                         M[i], N[i], K[i], alpha[i],
                         src[i], lda[i], weight[i], ldb[i],
                         bias[i], beta[i], dst[i], ldc[i],
                         is_weights_const[i], 1, params[i], algo);
    if (fused_act != grp_matmul_gated_act_t::none)
      apply_gated_act_inplace(fused_act, dst[i], 0, M[i],
                              N[i], ldc[i], act_dtype);
  }
}

// M-tile (ALGO 2) slices rows; N-indexed metadata passes through.
// M-indexed metadata (src_scale/zp {M,1} or {M,G}; 2D binary post-ops)
// is per-slice mutated in execute_m_tile (buff advance + dims[0]
// rewrite to slice_M).  Dynamic-quant additionally requires the
// source quant granularity to be row-local (dims[0] > 1, i.e.
// {M,1} or {M,G}); per-tensor / per-column / per-channel src layouts
// are rejected because the per-thread reorder would race on the
// shared scale/zp buffer and use slice-local statistics.  Still
// blocked: packed B (skipped GGML unpack), non-row-major layout,
// and per-expert dtype mismatches.
static bool check_m_tile_safe(
  const std::vector<char> &layout,
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
    // Dynamic-quant under M-tile only works when the source quant
    // granularity is row-local (M-indexed): each per-thread reorder
    // operates on its slice's rows independently, writes to a
    // disjoint slice of the scale/zp buffer, and produces stats
    // that match the granularity contract.  Layouts where the first
    // dim is not M — per-tensor (`{}` / `{1}` / `{1, 1}`), per-column
    // (`{1, K}`), per-channel-on-src (`{1, N}`), etc. — would race
    // on the shared scale/zp buffer AND each thread would compute
    // slice-local statistics instead of the full-matrix statistics
    // those granularities semantically require.  `offset_quant_by_row`
    // silently no-ops on those layouts (no row dim to slice), so we
    // must reject upstream.
    if (params[i].dynamic_quant) {
      const auto &sd = params[i].quant_params.src_scale.dims;
      if (sd.empty() || sd[0] <= 1) {
        return false;
      }
      const auto &zd = params[i].quant_params.src_zp.dims;
      if (!zd.empty() && zd[0] <= 1) {
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

// N-tile (ALGO 3) adds additional restrictions on top of the M-tile
// invariants: it slices columns of B, so any quantization metadata /
// packed-B / binary post-op buffer is disqualifying because none of
// them can be column-sliced without a dims update we don't do here.
// Only buffer-free element-wise post-ops (gelu, relu, swish, …) pass.
//
// PRECONDITION: the caller has already run `check_m_tile_safe` and
// confirmed it returned true.  This helper intentionally does NOT
// re-run those checks — the orchestrator `select_grp_matmul_algo`
// always calls M-tile first and only invokes this when m_tile_safe is
// true, so a second pass would just be duplicated work.
static bool check_n_tile_extra(
  const std::vector<matmul_params> &params,
  int num_ops) {
  // Same active-range constraint as `check_m_tile_safe` above —
  // tail slots carry framework prepack metadata, not real per-call
  // state, and would falsely flip n-tile-safe to false.
  for (int i = 0; i < num_ops; ++i) {
    if (params[i].quant_params.wei_scale.buff != nullptr) {
      return false;
    }
    if (params[i].quant_params.wei_zp   .buff != nullptr) {
      return false;
    }
    if (params[i].quant_params.src_scale.buff != nullptr) {
      return false;
    }
    if (params[i].quant_params.src_zp   .buff != nullptr) {
      return false;
    }
    for (const auto &po : params[i].postop_) {
      if (po.buff != nullptr) {
        return false;
      }
    }
  }
  return true;
}

// Auto-select (ALGO 0) heuristic — used when the caller leaves
// ZENDNNL_GRP_MATMUL_ALGO unset.  Picks among {ALGO 1, 3, 5} only;
// ALGO 2 (M-tile) and ALGO 4 (multilevel nested OMP) are reachable
// only via the env override because:
//   * ALGO 3 (N-tile) covers the same prompt-class shapes as ALGO 2.
//   * ALGO 4's nested OMP regions interact poorly with framework-
//     side OMP teams, so we prefer the flat strategies by default.
//
// Decision rule:
//   * num_ops > num_threads                                 → ALGO 5
//   * Large weight + prompt + N-tile viable + ≥5 experts +
//     (wide-N or many-experts)                              → ALGO 3
//     (large-weight wide-N prompt carve-out; large-weight
//      tall-N few-experts still routes to ALGO 1)
//   * Decode (max_M ≤ kDecodeMaxM) + ≥4 experts + N-tile OK → ALGO 3
//   * Small / medium weight (≤ kMediumWeight) +
//     prompt (max_M > kDecodeMaxM)                          → ALGO 1
//     (full-weight AOCL DLP cache key, no per-tile fan-out,
//      stable across thread counts.  Callers that have
//      validated ALGO 3 as a win on a specific shape can
//      still force it via ZENDNNL_GRP_MATMUL_ALGO=3.)
//   * Otherwise                                             → ALGO 1
//
// The four discriminators the heuristic uses:
//   1. num_ops vs num_threads — does each expert deserve a thread team?
//   2. weight_per_expert      — DRAM-streaming vs L3-resident?
//   3. max_M vs kDecodeMaxM   — decode-class vs prompt-class?
//   4. n_tile_safe + tiles_per_expert ≥ min_ntiles — N-tile viable?
static int auto_select_algo(
  const std::vector<int> &M,
  const std::vector<int> &N,
  const std::vector<int> &K,
  const std::vector<matmul_params> &params,
  int num_threads,
  bool n_tile_safe) {

  const int num_ops = static_cast<int>(M.size());
  if (num_threads <= 1 || num_ops == 0) {
    return 1;
  }

  // num_ops > num_threads → per-expert is the only way to cover every
  // expert in one wave; ALGO 1 would run them serially.
  if (num_ops > num_threads) {
    return 5;
  }

  const int max_M = *std::max_element(M.begin(), M.end());
  const int max_N = *std::max_element(N.begin(), N.end());
  const int max_K = *std::max_element(K.begin(), K.end());

  const size_t wei_elem = size_of(params[0].dtypes.wei);
  const size_t weight_per_expert =
    static_cast<size_t>(max_K) * max_N * wei_elem;

  // Small / medium weights (≤ kMediumWeight per expert)
  // + prompt-class shape (max_M > kDecodeMaxM): unconditionally
  // **ALGO 1**.  Short-circuit before computing N-tile arithmetic.
  //
  // Routing this regime to ALGO 3 would inflate the AOCL DLP LRU
  // footprint by `stable_n_thr` × `per-tile size` (a substantial
  // additional cache residency at high thread counts in many-expert
  // × layered MoE block), and the per-tile cache keys rotate with
  // thread count which forces a re-warm whenever the deployment
  // scales OMP teams.  ALGO 1 uses the FULL-weight AOCL key which
  // is thread-count-stable, deduplicates across prompt + decode in
  // the same process, and gives identical or better steady-state
  // perf on small/medium-weight prompt workloads across the typical
  // BS × seq × topk grid we target.
  // Callers that have validated ALGO 3 as a win on a specific
  // small/medium-weight prompt shape can still force it explicitly
  // via ZENDNNL_GRP_MATMUL_ALGO=3.
  //
  // This rule does NOT affect the large-weight (> kMediumWeight)
  // wide-N prompt path below — that carve-out keeps its measured
  // ALGO 3 advantage.
  if (weight_per_expert <= kMediumWeight && max_M > kDecodeMaxM) {
    return 1;
  }

  // N-tile viability — only the two branches below read this, so
  // the prompt small/medium path above never pays the arithmetic.
  // Ceiling division models partial last CCD (e.g., 126 threads →
  // 16 CCDs, last has 6 cores) consistently with flat_m_tile's
  // num_ccds.
  const int ccd_size_est = std::min(8, num_threads);
  const int num_ccds_est = std::max(1,
                                    (num_threads + ccd_size_est - 1) / ccd_size_est);
  const int eff_tile = (max_M <= kDecodeMaxM) ? kDecodeNTile : kMinNTile;
  const int tiles_per_expert = max_N / eff_tile;
  const int team_est = num_threads / std::max(1, num_ops);
  const int min_ntiles = (num_ops > num_ccds_est)
                         ? std::max(2, ccd_size_est / 2)
                         : std::max(2, team_est / 2);
  const bool ntile_ok =
    n_tile_safe && (tiles_per_expert >= min_ntiles);

  // Large weights (DRAM-streaming): ALGO 1 by default.  Escape to
  // ALGO 3 only when column-parallel has enough per-thread work to
  // amortise its per-thread BKC pack + round-scheduling overhead.
  //
  // Internal prompt sweeps on the large-weight regime show:
  //   * Few-experts tall shapes (K > N) — ALGO 1 is the clear win.
  //     With a low thread-per-expert count, ALGO 3 splits N into
  //     thin column slices and each thread packs a large K×slice
  //     working set, paying per-thread memory + pack traffic that
  //     AOCL DLP's sequential-experts-with-full-team avoids.
  //   * Few-experts wide shapes (N > K) — ALGO 3 wins; more N
  //     to split gives the per-expert team better parallelism.
  //   * Many-experts (either orientation) — ALGO 3 wins; the
  //     round scheduling and pack cost amortise across experts.
  //
  // Heuristic: route to ALGO 3 iff (wide-N || many-experts).
  // Otherwise fall back to ALGO 1 for tall-N few-experts shapes.
  if (weight_per_expert > kMediumWeight) {
    const bool wide_N        = (max_N > max_K);
    const bool many_experts  = (num_ops >= 16);
    if (num_ops >= 5 && ntile_ok && max_M > kDecodeMaxM
        && (wide_N || many_experts))
      return 3;
    return 1;
  }

  // Small / medium weights + decode (max_M ≤ kDecodeMaxM): ≥4
  // experts + N-tile viable → ALGO 3.  This is the MoE decode
  // hot path where the custom-kernel-enabled configuration shows
  // a measurable advantage over ALGO 1 in our internal sweeps.
  // Smaller expert counts fall to ALGO 1.
  // (The prompt sibling regime is short-circuited above.)
  return (num_ops >= 4 && ntile_ok) ? 3 : 1;
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
  const bool m_tile_safe = check_m_tile_safe(layout, params, num_ops_eff);
  const bool n_tile_safe = m_tile_safe
      && check_n_tile_extra(params, num_ops_eff);

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
            "[GRP_MATMUL Level2 dispatch WARN] env_algo=2 (flat_m_tile) "
            "REJECTED: m_tile unsafe (non-row-major, per-expert dtype "
            "mismatch, packed B, softmax/pooling post-op, or "
            "dynamic-quant with non-row-local src granularity). "
            "FALLBACK algo=1 (sequential_experts).");
      }
      algo = 1;
    }
    if (algo == 3 && !n_tile_safe) {
      static const bool s_log = apilog_warning_enabled();
      if (s_log) {
        apilog_warning(
            "[GRP_MATMUL Level2 dispatch WARN] env_algo=3 (flat_n_tile) "
            "REJECTED: n_tile unsafe (non-row-major, dtype mismatch, "
            "quantised weights, or buffer post-op). FALLBACK algo=1 "
            "(sequential_experts).");
      }
      algo = 1;
    }
    return algo;
  }

  return auto_select_algo(M, N, K, params, num_threads, n_tile_safe);
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
  const bool a3_fuses = (use_algo == 3)
                        && a3_can_fuse_act(fused_act)
                        && (caller_layout_tight || get_grp_n_tile_fused_act());
  const bool act_fused = a3_fuses
                         || ((use_algo != 3) && (fused_act != grp_matmul_gated_act_t::none));

  // ── Top-level dispatch APILOG ─────────────────────────────────────
  // Emits the final routing decision (POST env-override, POST auto-
  // select) with the discriminator values that drove it.  Users
  // debugging "why did my shape land on ALGO X" get a single line
  // that tells the complete story — shape + all gates + the
  // chosen algo.  Gated by apilog_info_enabled() (cached); free
  // when logging is off.
  static const bool s_dispatch_log = apilog_info_enabled();
  if (s_dispatch_log && !M.empty()) {
    const int env_algo = get_grp_matmul_algo();
    const int max_M_v = *std::max_element(M.begin(), M.end());
    const int max_N_v = *std::max_element(N.begin(), N.end());
    const int max_K_v = *std::max_element(K.begin(), K.end());
    const size_t wei_elem_b = size_of(params[0].dtypes.wei);
    const size_t wei_per_expert_mb =
        (static_cast<size_t>(max_K_v) * max_N_v * wei_elem_b) >> 20;
    const char *reason =
        (env_algo >= 1 && env_algo <= 5)
            ? (env_algo == use_algo ? "env_ok" : "env_fallback")
            : "auto";
    apilog_info(
        "[GRP_MATMUL Level2 dispatch] algo=", use_algo,
        " env=", env_algo,
        " reason=", reason,
        " act=", act_name(fused_act),
        " act_fused=", act_fused,
        " num_ops=", static_cast<int>(M.size()),
        " num_threads=", num_threads,
        " max_M=", max_M_v,
        " max_N=", max_N_v,
        " max_K=", max_K_v,
        " wei/expert(MB)=", wei_per_expert_mb,
        " wide_N=", (max_N_v > max_K_v),
        " many_experts=", (static_cast<int>(M.size()) >= 16),
        " caller_tight=", caller_layout_tight);
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
