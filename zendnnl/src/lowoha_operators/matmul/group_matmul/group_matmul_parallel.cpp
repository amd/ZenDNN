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

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace {

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
  if (num_ops == 0 || num_threads <= 0) return;
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
  if (num_ops == 0 || num_threads <= 0) return;
  matmul_algo_t algo = resolve_kernel();

  const int ccd_size = std::min(8, num_threads);
  // Ceiling to match flat_m_tile / flat_n_tile: partial last CCD counts as one.
  const int num_ccds = std::max(1, (num_threads + ccd_size - 1) / ccd_size);
  const int max_M = *std::max_element(M.begin(), M.end());

  if (num_ops <= num_ccds && max_M >= ccd_size) {
    // (A) Few experts, large M: multi-CCD per expert, all concurrent.
    int64_t total_M = 0;
    for (int i = 0; i < num_ops; ++i) total_M += M[i];
    if (total_M <= 0) total_M = num_ops;

    std::vector<int> ccds_per_op(num_ops, 1);
    int remaining = num_ccds - num_ops;
    if (remaining > 0) {
      for (int i = 0; i < num_ops; ++i) {
        int extra = static_cast<int>(
            static_cast<int64_t>(remaining) * M[i] / total_M);
        ccds_per_op[i] += extra;
      }
      int used = 0;
      for (int i = 0; i < num_ops; ++i) used += ccds_per_op[i];
      for (int i = 0; used < num_ccds; ++i, ++used)
        ccds_per_op[i % num_ops]++;
    }
    std::vector<int> thr_per_op(num_ops);
    for (int i = 0; i < num_ops; ++i)
      thr_per_op[i] = ccds_per_op[i] * ccd_size;

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
  } else {
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
  if (num_ops == 0 || num_threads <= 0) return;
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

// ── ALGO selection ──────────────────────────────────────────────────────
//
// Returns ALGO number (1-5).  Handles manual override, tiling_safe
// validation, and auto-select heuristics.
//
// After refactor:
//   ALGO 2 = pure M-tile (row-parallel).
//   ALGO 3 = pure N-tile (column-parallel), including decode paths.

int select_grp_matmul_algo(
    const std::vector<char> &layout,
    const std::vector<int> &M,
    const std::vector<int> &N,
    const std::vector<int> &K,
    const std::vector<matmul_params> &params,
    int num_threads) {

  const int env_algo = get_grp_matmul_algo();
  const int num_ops = static_cast<int>(M.size());

  // ── Tiling safety checks ──
  //
  // M-tile (ALGO 2) slices rows only — columns (N) are unchanged.
  // B, bias, quantization scales, and binary post-op buffers work
  // as-is (binary uses row offsetting for 2D broadcasts).
  // However, dynamic_quant and packed B are blocked because
  // execute_expert_slice calls matmul_execute directly, skipping
  // the preprocessing that matmul_direct performs (dynamic source
  // quantization, GGML unpack).
  //
  // N-tile (ALGO 3) slices columns — needs column-sliceable B, bias,
  // and post-op buffers.  Packed B, quantization, and binary post-ops
  // with broadcast dims are NOT column-sliceable.
  //
  // Both require: row-major layout, uniform dtypes across experts.
  bool m_tile_safe = true;
  for (int i = 0; i < num_ops && m_tile_safe; ++i) {
    if (layout[i] != 'r' && layout[i] != 'R') m_tile_safe = false;
    if (params[i].dtypes.src != params[0].dtypes.src) m_tile_safe = false;
    if (params[i].dtypes.wei != params[0].dtypes.wei) m_tile_safe = false;
    if (params[i].dtypes.dst != params[0].dtypes.dst) m_tile_safe = false;
    if (params[i].dtypes.bias != params[0].dtypes.bias) m_tile_safe = false;
    if (params[i].mem_format_a != 'n') m_tile_safe = false;
    if (params[i].mem_format_b != 'n') m_tile_safe = false;
    if (params[i].dynamic_quant) m_tile_safe = false;
    if (params[i].packing.pack_format_b != 0) m_tile_safe = false;
    for (const auto &po : params[i].postop_) {
      if (po.po_type == post_op_type_t::softmax
          || po.po_type == post_op_type_t::pooling)
        m_tile_safe = false;
    }
  }

  // n_tile_safe: stricter than m_tile_safe.
  //
  // N-tile slices columns of B.  Allows only:
  //   - Buffer-free element-wise post-ops (gelu, relu, swish, etc.).
  //   - Standard unpacked A and B layouts.
  //
  // Blocks:
  //   - Quantization: wei_scale/wei_zp dims metadata cannot be safely
  //     column-sliced without updating dims to match n_tile.
  //   - dynamic_quant: src quantization per tile not validated.
  //   - Packed B (pack_format_b != 0): not column-sliceable.
  //   - Binary post-ops with buffers (broadcast dims not column-sliceable).
  bool n_tile_safe = m_tile_safe;
  for (int i = 0; i < num_ops && n_tile_safe; ++i) {
    if (params[i].dynamic_quant) n_tile_safe = false;
    if (params[i].quant_params.wei_scale.buff != nullptr) n_tile_safe = false;
    if (params[i].quant_params.wei_zp.buff != nullptr) n_tile_safe = false;
    if (params[i].quant_params.src_scale.buff != nullptr) n_tile_safe = false;
    if (params[i].quant_params.src_zp.buff != nullptr) n_tile_safe = false;
    if (params[i].mem_format_a != 'n') n_tile_safe = false;
    if (params[i].mem_format_b != 'n') n_tile_safe = false;
    if (params[i].packing.pack_format_b != 0) n_tile_safe = false;
    for (const auto &po : params[i].postop_) {
      if (po.buff != nullptr) n_tile_safe = false;
    }
  }

  // Manual override: ZENDNNL_GRP_MATMUL_ALGO=1..5.
  // ALGO 2 (M-tile): needs m_tile_safe (row-major, uniform dtypes).
  // ALGO 3 (N-tile): needs n_tile_safe (+ unpacked B, no binary post-ops).
  // ALGO 1/4/5: no tiling → no safety guard needed (BLAS handles all).
  if (env_algo >= 1 && env_algo <= 5) {
    int algo = env_algo;
    if (algo == 2 && !m_tile_safe) algo = 1;
    if (algo == 3 && !n_tile_safe) algo = 1;
    return algo;
  }

  // Auto-select (ALGO=0).
  //
  // Data-driven decision tree based on benchmarks across three models
  // with 128 threads and BF16 dtypes (360+ config comparisons):
  //
  //   Mixtral-8x7B  (224MB/expert, 8 experts)
  //     Decode: A1 wins 79%. Prompt gate+up: A3 wins. Down_proj: A1.
  //   GPT-OSS       (32MB/expert, 2-32 experts)
  //     Decode: A3 wins 86% (4+ experts). Prompt: A2 for totM>1500, else A3.
  //   Qwen3-30B     (6MB/expert, 8-128 experts)
  //     A1 never wins. Decode: A3 57%, A2 43%. Prompt: A2 100%.
  //
  // Primary discriminators:
  //   1. weight_per_expert — L3 fit determines tiling benefit
  //   2. total_M (sum of all M[i]) — aggregate batch size
  //   3. num_ops — expert count drives inter-expert parallelism
  //   4. max_N — N-tiling viability (large N = more tiles)
  //   5. avg_M (total_M / num_ops) — per-expert workload

  if (num_threads <= 1 || num_ops == 0)
    return 1;

  // num_ops > 0 guaranteed from here on.
  const int max_M = *std::max_element(M.begin(), M.end());
  const int max_N = *std::max_element(N.begin(), N.end());
  const int max_K = *std::max_element(K.begin(), K.end());

  int64_t total_M = 0;
  for (int i = 0; i < num_ops; ++i) total_M += M[i];
  const int avg_M = static_cast<int>(total_M / num_ops);

  const size_t wei_elem = size_of(params[0].dtypes.wei);
  const size_t weight_per_expert =
      static_cast<size_t>(max_K) * max_N * wei_elem;

  static constexpr size_t kSmallWeight  = 16UL * 1024 * 1024;  // 16MB
  static constexpr size_t kMediumWeight = 64UL * 1024 * 1024;  // 64MB

  // N-tile viability check (mirrors flat_n_tile's ntile_viable).
  // Ceiling division models partial last CCD (e.g., 126t → 16 CCDs, last has
  // 6 cores) consistently with flat_m_tile's num_ccds.  Without ceiling, 126t
  // would report 15 CCDs here while the M-tile planner uses 16, which could
  // flip the num_ops > num_ccds_est boundary and skew A2 vs A3 routing for
  // configs like 16-expert GPT-OSS / Qwen3 at 126 threads.
  const int ccd_size_est = std::min(8, num_threads);
  const int num_ccds_est = std::max(1,
      (num_threads + ccd_size_est - 1) / ccd_size_est);
  const int eff_tile = (max_M <= kDecodeMaxM) ? kDecodeNTile : kMinNTile;
  const int tiles_per_expert = max_N / eff_tile;
  const int team_est = num_threads / std::max(1, num_ops);
  const int min_ntiles = (num_ops > num_ccds_est)
      ? std::max(2, ccd_size_est / 2)
      : std::max(2, team_est / 2);
  const bool ntile_useful = (tiles_per_expert >= min_ntiles);

  // ════════════════════════════════════════════════════════════════════
  // Large weights (> 64MB): Mixtral-class (224MB gate+up, 112MB down)
  // ════════════════════════════════════════════════════════════════════
  // DLP's internal panel blocking with all threads is near-optimal for
  // DRAM-streaming workloads.
  //
  // Exception 1: prompt gate+up with large N → A3 N-tiling wins 1.1-1.7x.
  // Exception 2: down_proj (N=4096, K=14336) with 5-8 experts and
  //   moderate per-expert M (4-14) → A3 wins 1.5-2.1x because the
  //   huge K makes compute-bound, and N-tiling distributes it well.
  //   Trade-off: enables A3 for Mixtral down_proj (improves accuracy)
  //   at the cost of a rare decode gate+up miss (~5% of configs).
  if (weight_per_expert > kMediumWeight) {
    if (num_ops >= 5 && n_tile_safe && ntile_useful && max_M > kDecodeMaxM)
      return 3;
    return 1;
  }

  // ════════════════════════════════════════════════════════════════════
  // Small weights (< 16MB): Qwen3/DeepSeek/Switch-class (3-6MB)
  // ════════════════════════════════════════════════════════════════════
  // Weights fit in per-CCD L3.  A1 never wins on Qwen3 benchmarks.
  // A2 (M-tile) and A3 (N-tile) both deliver 4-27x over A1.
  //
  // Key insight from Qwen3 data: the A2 vs A3 choice depends heavily on
  // tiles_per_expert (N / tile_width) and avg_M.  When tiles are few
  // (N=1536 → 6 tiles) or per-expert M is moderate (avg_M >= 4),
  // A2's M-tile row-sharing wins.  A3 only dominates when there are
  // enough N-tiles AND per-expert M is tiny (1-2 rows).
  if (weight_per_expert < kSmallWeight) {
    if (num_ops > num_threads) return 5;

    // Prompt (large per-expert M): A2 wins 100% on Qwen3 prompt.
    if (max_M > kDecodeMaxM) {
      if (m_tile_safe) return 2;
      if (n_tile_safe && ntile_useful) return 3;
      return 5;
    }

    // ── Decode heuristic for small weights ──
    //
    // N-tile quality: tiles_per_expert < 8 means too few tiles for
    // effective N-parallel dispatch → prefer A2 M-tile.
    // Qwen3 N=1536 → 6 tiles (borderline), N=2048 → 8 tiles (OK).
    static constexpr int kMinTilesForA3 = 8;
    const bool good_ntiles = (tiles_per_expert >= kMinTilesForA3)
        && n_tile_safe && ntile_useful;

    // When avg_M >= 4 AND N-tiles are few, A2's row sharing across
    // threads outperforms A3's column splitting.  But when N is wide
    // enough for many tiles (e.g., GPT-OSS N=2880 → 11 tiles), A3
    // still wins even with avg_M=4-12 because N-tiling parallelism
    // is highly effective.  Only prefer A2 when tiles are scarce.
    // Benchmarked: Qwen3 N=1536 (6 tiles) → A2 wins avg_M >= 4.
    //              GPT-OSS N=2880 (11 tiles) → A3 wins avg_M 4-12.
    if (num_ops >= 8 && avg_M >= 4 && !good_ntiles) {
      if (m_tile_safe) return 2;
    }

    // Many experts (20+) with tiny per-expert M (1-2): A3 wins
    // when N-tiles are sufficient (N=2048+ → 8+ tiles).
    // For narrow N (1536 → 6 tiles), A3 still wins up to ~64 experts
    // but A2 wins at 80+ experts (round overhead dominates).
    if (num_ops >= 20 && avg_M <= 2) {
      if (num_ops >= 80 && !good_ntiles) {
        if (m_tile_safe) return 2;
      }
      if (n_tile_safe && ntile_useful) return 3;
      if (m_tile_safe) return 2;
      return 5;
    }

    // 8-19 experts with small avg_M (1-3): A3 if good N-tiles.
    if (num_ops >= 8) {
      if (good_ntiles) return 3;
      if (m_tile_safe) return 2;
      return 5;
    }

    // 5-7 experts: A2 for moderate M, A3 for tiny M with good tiles.
    if (num_ops >= 5) {
      if (avg_M >= 4 && m_tile_safe) return 2;
      if (good_ntiles) return 3;
      if (m_tile_safe) return 2;
    }

    if (!m_tile_safe && !n_tile_safe && num_ops >= 5)
      return 5;
    if (m_tile_safe && num_ops >= 4) return 2;
    return 1;
  }

  // ════════════════════════════════════════════════════════════════════
  // Medium weights (16-64MB): GPT-OSS-class (16-32MB per expert)
  // ════════════════════════════════════════════════════════════════════
  // Weights partially fit in L3.  A3 dominates decode (86% wins).
  // A2 wins large prompt (totM > ~1500).  A1 only wins with 2-3 experts.
  if (num_ops <= 3)
    return 1;

  // Prompt: large aggregate M → A2 M-tile distributes rows efficiently.
  // Crossover depends on total_M and distribution skew:
  //   - Skewed (max_M > 4*avg_M): A2 handles imbalance better,
  //     wins at lower total_M (~1200).  GPT-OSS data: 1472 skewed → A2.
  //   - Uniform/moderate skew: A3 holds longer, crossover ~2500.
  //     GPT-OSS data: 1984 moderate → A3 wins.
  if (max_M > kDecodeMaxM) {
    const int skew = (avg_M > 0) ? (max_M / avg_M) : 1;
    const int64_t a2_threshold = (skew > 4) ? 1200 : 2500;
    if (total_M > a2_threshold && m_tile_safe) return 2;
    if (n_tile_safe && ntile_useful) return 3;
    if (m_tile_safe) return 2;
    return 1;
  }

  // Decode: A3 wins 86% of GPT-OSS decode configs with 4+ experts.
  // Large N (gate+up, N=5760) has excellent N-tiling opportunity.
  // Small N (down_proj, N=2880) also benefits from A3 at 8+ experts
  // but A2 can be competitive when per-expert M is moderate (4-8).
  if (num_ops >= 4 && n_tile_safe && ntile_useful)
    return 3;

  // N-tile not viable but M-tile possible: use A2 for 6+ experts.
  if (num_ops >= 6 && m_tile_safe)
    return 2;

  return 1;
}

} // namespace

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
  //   - ALGO 3 fuses only when the activation's layout fits the N-tile
  //     split AND the env flag allows it (default: allow).
  //   - For any fused_act we cannot fuse, the caller runs a separate
  //     activation pass after this function returns.
  const bool a3_fuses = (use_algo == 3)
      && a3_can_fuse_act(fused_act)
      && get_grp_n_tile_fused_act();
  const bool act_fused = a3_fuses
      || ((use_algo != 3) && (fused_act != grp_matmul_gated_act_t::none));

  auto set_mode = [&](const char *s) {
    if (gemm_mode_out != nullptr) *gemm_mode_out = s;
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
    set_mode(a3_fuses ? "flat_n_tile_fused_swiglu_oai" : "flat_n_tile");
    flat_n_tile(layout, transA, transB, M, N, K, alpha,
        src, lda, weight, ldb, bias, beta, dst, ldc,
        is_weights_const, params, num_threads,
        a3_fuses ? fused_act : grp_matmul_gated_act_t::none,
        act_dtype);
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
