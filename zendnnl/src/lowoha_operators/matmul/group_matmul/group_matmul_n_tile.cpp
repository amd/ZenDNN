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

/// ALGO 3 — N-tile parallel GEMM for grouped expert matmul, with
/// optional fused swiglu_oai epilogue (see
/// group_matmul_parallel_common.hpp).
///
/// Self-contained translation unit.  Exposes `flat_n_tile` to the
/// dispatcher via the common header.  All other helpers are private.

#include <algorithm>
#include <vector>

#include <omp.h>

#include "group_matmul_parallel_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace {

inline void execute_n_tile(
    int e, int local_tid, int team_size, int min_n_tile,
    const std::vector<char> &layout,
    const std::vector<bool> &transA, const std::vector<bool> &transB,
    const std::vector<int> &M, const std::vector<int> &N,
    const std::vector<int> &K, const std::vector<float> &alpha,
    const std::vector<const void *> &src, const std::vector<int> &lda,
    const std::vector<const void *> &weight, const std::vector<int> &ldb,
    const std::vector<const void *> &bias, const std::vector<float> &beta,
    const std::vector<void *> &dst, const std::vector<int> &ldc,
    const std::vector<bool> &is_weights_const,
    const std::vector<matmul_params> &params,
    size_t wei_elem, size_t dst_elem, size_t bias_elem,
    matmul_algo_t algo) {

  if (M[e] <= 0) return;
  const int n_thr = std::max(1, std::min(team_size, N[e] / min_n_tile));
  if (local_tid >= n_thr) return;

  const int col_start = static_cast<int>(
      static_cast<int64_t>(N[e]) * local_tid / n_thr);
  const int col_end = static_cast<int>(
      static_cast<int64_t>(N[e]) * (local_tid + 1) / n_thr);
  const int n_tile = col_end - col_start;
  if (n_tile <= 0) return;

  // Weight: slice columns of op(B).
  const size_t wei_off = transB[e]
      ? static_cast<size_t>(col_start) * ldb[e] * wei_elem
      : static_cast<size_t>(col_start) * wei_elem;
  const auto *w = static_cast<const char *>(weight[e]) + wei_off;

  // dst: column offset within each row (ldc unchanged).
  auto *d = static_cast<char *>(dst[e])
      + static_cast<size_t>(col_start) * dst_elem;

  // bias: offset by col_start if present.
  const void *b = nullptr;
  if (bias[e] != nullptr)
    b = static_cast<const char *>(bias[e])
        + static_cast<size_t>(col_start) * bias_elem;

  static thread_local matmul_params tile_params;
  tile_params = params[e];

  // Quantization is blocked by n_tile_safe (quant dims metadata cannot
  // be safely column-sliced without updating dims to match n_tile).
  // Post-ops with buffers (binary_add/mul) also blocked by n_tile_safe.
  // Only buffer-free element-wise activations reach this path.

  execute_expert_slice(layout[e], transA[e], transB[e],
      M[e], n_tile, K[e], alpha[e],
      src[e], lda[e], w, ldb[e],
      b, beta[e], d, ldc[e],
      is_weights_const[e], 1, tile_params, algo);
}

// Pair-aligned N-tile matmul for the fused-swiglu-oai path.
//
// Differs from execute_n_tile in the column split: both col_start and
// col_end are forced to be even so each thread owns complete interleaved
// (g, u) pairs.  For the common case (N[e] divisible by 2 and by n_thr)
// the split is identical to execute_n_tile; only ragged divisions differ
// by ≤1 column per boundary.
//
// Historical note: the epilogue used to be column-parallel with the same
// split as this matmul, so pair alignment on the matmul side was
// required for correctness.  The current epilogue
// (apply_n_tile_paired_swiglu_oai) splits by M (rows) instead, so pair
// alignment is no longer a correctness constraint — it's kept only so
// the two halves of each pair are produced by the same thread, which
// keeps each thread's matmul write stride simple and matches the
// numerical behavior tests were written against.

inline void execute_n_tile_paired(
    int e, int local_tid, int team_size, int min_n_tile,
    const std::vector<char> &layout,
    const std::vector<bool> &transA, const std::vector<bool> &transB,
    const std::vector<int> &M, const std::vector<int> &N,
    const std::vector<int> &K, const std::vector<float> &alpha,
    const std::vector<const void *> &src, const std::vector<int> &lda,
    const std::vector<const void *> &weight, const std::vector<int> &ldb,
    const std::vector<const void *> &bias, const std::vector<float> &beta,
    const std::vector<void *> &dst, const std::vector<int> &ldc,
    const std::vector<bool> &is_weights_const,
    const std::vector<matmul_params> &params,
    size_t wei_elem, size_t dst_elem, size_t bias_elem,
    matmul_algo_t algo) {

  if (M[e] <= 0) return;
  const int n_thr = std::max(1, std::min(team_size, N[e] / min_n_tile));
  if (local_tid >= n_thr) return;

  // Pair-aligned split: work in pair-units, then convert to cols (×2).
  const int pairs_total = N[e] / 2;
  const int p_start = static_cast<int>(
      static_cast<int64_t>(pairs_total) * local_tid / n_thr);
  const int p_end = static_cast<int>(
      static_cast<int64_t>(pairs_total) * (local_tid + 1) / n_thr);
  const int col_start = 2 * p_start;
  const int col_end = 2 * p_end;
  const int n_tile = col_end - col_start;
  if (n_tile <= 0) return;

  const size_t wei_off = transB[e]
      ? static_cast<size_t>(col_start) * ldb[e] * wei_elem
      : static_cast<size_t>(col_start) * wei_elem;
  const auto *w = static_cast<const char *>(weight[e]) + wei_off;

  auto *d = static_cast<char *>(dst[e])
      + static_cast<size_t>(col_start) * dst_elem;

  const void *b = nullptr;
  if (bias[e] != nullptr)
    b = static_cast<const char *>(bias[e])
        + static_cast<size_t>(col_start) * bias_elem;

  static thread_local matmul_params tile_params;
  tile_params = params[e];

  execute_expert_slice(layout[e], transA[e], transB[e],
      M[e], n_tile, K[e], alpha[e],
      src[e], lda[e], w, ldb[e],
      b, beta[e], d, ldc[e],
      is_weights_const[e], 1, tile_params, algo);
}

// Swiglu_oai epilogue for the ALGO 3 paired-N-tile path.
//
// The caller MUST place a `#pragma omp barrier` between the matmul and
// this function so every thread's matmul writes are globally visible
// before any thread starts reading for activation.
//
// Correctness note — why we split M, not N:
//
//   The matmul split was column-wise (thread t owns cols [2·p_start_t,
//   2·p_end_t)).  Splitting the epilogue the same way would create a
//   cross-thread write-after-read race on the in-place compaction:
//   thread t writes compact output cols [p_start_t, p_end_t) while
//   thread t' < t still needs to read its own pair cols
//   [2·p_start_{t'}, 2·p_start_t) — and the t-write range starts at
//   p_start_t < 2·p_start_t, so the two ranges overlap.
//
//   Splitting by M instead makes every thread own a disjoint row slice
//   of the (M × N) output.  Reads and writes stay on that thread's own
//   rows, so no cross-thread aliasing is possible on any column.  The
//   in-place compaction within one row is still safe because writing
//   col n happens after reads at cols 2n, 2n+1 (both ≥ n) — see the
//   in-place safety note in swiglu_oai_tile_*.
//
//   Cache-locality cost: threads no longer re-use the exact column
//   range their matmul populated.  The activation is O(M × N) light
//   arithmetic (~15 flops/elt) with ~1 GB/s read·write bandwidth per
//   layer — negligible vs the matmul and vs a second OMP pass.  Any
//   remaining hit comes through L3, which has ample bandwidth.
//
//   When M[e] < n_thr some threads get m_slice == 0 and no-op — the
//   outer omp parallel region barrier still lets them exit cleanly.

inline void apply_n_tile_paired_swiglu_oai(
    int e, int local_tid, int team_size, int min_n_tile,
    const std::vector<int> &M, const std::vector<int> &N,
    const std::vector<void *> &dst, const std::vector<int> &ldc,
    data_type_t act_dtype) {

  if (M[e] <= 0) return;
  const int n_thr = std::max(1, std::min(team_size, N[e] / min_n_tile));
  if (local_tid >= n_thr) return;

  // Row split: this thread owns rows [m_start, m_end) of expert e's
  // (M × N) output and applies the full-width compaction in place.
  const int m_start = static_cast<int>(
      static_cast<int64_t>(M[e]) * local_tid / n_thr);
  const int m_end = static_cast<int>(
      static_cast<int64_t>(M[e]) * (local_tid + 1) / n_thr);
  const int m_slice = m_end - m_start;
  if (m_slice <= 0) return;

  const size_t dst_elem = size_of(act_dtype);
  char *row_base = static_cast<char *>(dst[e])
      + static_cast<size_t>(m_start) * ldc[e] * dst_elem;
  const int pairs = N[e] / 2;
  apply_swiglu_oai_tile_rows(row_base, m_slice, /*col_start=*/0, pairs,
                             ldc[e], act_dtype);
}

} // namespace

// ── ALGO=3: flat N-tile — pure column-parallel, single-level OMP ────────
//
// Each thread handles full M but a unique N-slice of weight B.
// Per-thread weight read = K × (N/n_thr) instead of K × N.
// Framework-safe: single flat OMP region, num_thr=1 per tile.
//
// N-tile viability check: N-tiling only helps when N is large enough
// to create useful tiles.  When max_N / kMinNTile < team_size / 2,
// most threads would be idle.
//
// Regimes:
//   (D) Decode parallel — 6+ experts, enough N-tiles per team, min_M≥3.
//   (A) Prompt/large-M parallel — proportional CCDs, concurrent experts.
//   (B) Many experts (> num_ccds) — barrier-synchronized rounds.
//   (F) Fallback — per-expert parallel dispatch (1 thread per expert,
//       all experts concurrent) when N is too small for N-tiling.

// (kDecodeNTile is defined in group_matmul_parallel_common.hpp)

// flat_n_tile
//
// ALGO 3 — pure N-tile parallel GEMM for grouped expert matmul.
//
// The function optionally fuses a gated activation into a per-thread
// epilogue when the activation's layout allows it.  This saves a
// separate OMP pass over the matmul output and keeps per-thread tiles
// hot in L1/L2 for decode shapes.
//
// Parameters
//   fused_act:
//     none           → legacy N-tile only; the caller runs any gated
//                      activation as a separate pass afterward.
//     swiglu_oai_mul → N-tile + per-thread interleaved-pair epilogue.
//     (other values are treated as none; see a3_can_fuse_act.)
//   act_dtype:
//     Element type of the output buffer when fusing; unused when
//     fused_act == none.
void flat_n_tile(
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
    grp_matmul_gated_act_t fused_act,
    data_type_t act_dtype) {

  const int num_ops = static_cast<int>(M.size());
  if (num_ops == 0 || num_threads <= 0) return;

  // Engage the per-thread fused epilogue only for activations whose
  // interleaved layout puts complete (g, u) pairs on every thread's
  // tile.  Everything else falls through the legacy path.
  const bool fused_epilogue =
      (fused_act == grp_matmul_gated_act_t::swiglu_oai_mul);

  matmul_algo_t algo = resolve_kernel();
  const int ccd_size = std::min(8, num_threads);
  // Ceiling so a partial last CCD (e.g., 126t → 16 CCDs, last = 6 cores)
  // is counted — keeps num_ccds consistent with flat_m_tile's planner.
  const int num_ccds = std::max(1, (num_threads + ccd_size - 1) / ccd_size);

  const size_t wei_elem = size_of(params[0].dtypes.wei);
  const size_t dst_elem = size_of(params[0].dtypes.dst);
  const size_t bias_elem = (params[0].dtypes.bias != data_type_t::none)
      ? size_of(params[0].dtypes.bias) : sizeof(float);

  const int max_M = *std::max_element(M.begin(), M.end());
  const int max_N = *std::max_element(N.begin(), N.end());

  scoped_active_levels guard(1);

  // Per-thread matmul tile.  For the fused epilogue we need
  // pair-aligned column splits so each thread owns complete (g, u)
  // pairs; otherwise the legacy split is used unchanged.
  auto do_tile = [&](int e, int local_tid, int team_size, int min_n_tile) {
    if (fused_epilogue) {
      execute_n_tile_paired(e, local_tid, team_size, min_n_tile,
          layout, transA, transB, M, N, K, alpha,
          src, lda, weight, ldb, bias, beta, dst, ldc,
          is_weights_const, params, wei_elem, dst_elem, bias_elem, algo);
    } else {
      execute_n_tile(e, local_tid, team_size, min_n_tile,
          layout, transA, transB, M, N, K, alpha,
          src, lda, weight, ldb, bias, beta, dst, ldc,
          is_weights_const, params, wei_elem, dst_elem, bias_elem, algo);
    }
  };

  // N-tile viability: N must be large enough for useful tiling.
  // For few-expert path (A): need tiles >= team_size / 2.
  // For round path (B): need tiles >= ccd_size / 2 (each round-expert
  //   gets ccd_size threads; if tiles < ccd_size/2, most are idle).
  // Use the stricter of the two checks.
  const int team_size_est = num_threads / std::max(1, num_ops);
  const int min_tile = (max_M <= kDecodeMaxM) ? kDecodeNTile : kMinNTile;
  const int tiles_available = max_N / min_tile;
  const int min_useful = (num_ops > num_ccds)
      ? std::max(1, ccd_size / 2)
      : std::max(1, team_size_est / 2);
  const bool ntile_viable = (tiles_available >= min_useful);

  // When N is too small for useful N-tiling (ntile_viable=false), take
  // one of two fallback paths:
  //
  //   ZENDNNL_GRP_N_FALLBACK_V1=1 (default):
  //     Run experts sequentially with `num_threads` threads per GEMM —
  //     identical to ALGO 1's per-kernel threading.  Dominates when
  //     num_ops is small (e.g. num_ops=4 warmup: the per-expert-parallel
  //     path below uses only 4 threads out of 128).
  //
  //   ZENDNNL_GRP_N_FALLBACK_V1=0:
  //     Legacy path — 1 OMP thread per expert, num_ops experts in
  //     parallel, each GEMM runs with num_thr=1.  Only beats V1 when
  //     num_ops is comparable to num_threads and each expert is small
  //     enough to fit on one thread.
  //
  // (No inner scoped_active_levels here — the outer `guard(1)` already
  // set max_active_levels to 1.)
  if (!ntile_viable) {
    if (get_grp_n_fallback_v1()) {
      for (int e = 0; e < num_ops; ++e) {
        if (M[e] <= 0) continue;
        static thread_local matmul_params local_params;
        local_params = params[e];
        execute_expert_slice(layout[e], transA[e], transB[e],
            M[e], N[e], K[e], alpha[e],
            src[e], lda[e], weight[e], ldb[e],
            bias[e], beta[e], dst[e], ldc[e],
            is_weights_const[e], num_threads, local_params, algo);
        if (fused_epilogue) {
          apply_gated_act_inplace(fused_act, dst[e], 0, M[e],
                                  N[e], ldc[e], act_dtype);
        }
      }
      return;
    }

    #pragma omp parallel for num_threads(std::min(num_ops, num_threads)) \
        schedule(dynamic)
    for (int e = 0; e < num_ops; ++e) {
      if (M[e] <= 0) continue;
      static thread_local matmul_params local_params;
      local_params = params[e];
      execute_expert_slice(layout[e], transA[e], transB[e],
          M[e], N[e], K[e], alpha[e],
          src[e], lda[e], weight[e], ldb[e],
          bias[e], beta[e], dst[e], ldc[e],
          is_weights_const[e], 1, local_params, algo);
      if (fused_epilogue) {
        apply_gated_act_inplace(fused_act, dst[e], 0, M[e],
                                N[e], ldc[e], act_dtype);
      }
    }
    return;
  }

  // For paths (A) and (B): when max_M is small (decode shape), use the
  // smaller kDecodeNTile as min-tile so max_n_thr is high enough to
  // actually saturate all threads.  Gated by ZENDNNL_GRP_N_DECODE_TILE_AB
  // (default on) — see header for rationale.
  const int ab_min_tile = (max_M <= kDecodeMaxM && get_grp_n_decode_tile_ab())
      ? kDecodeNTile : kMinNTile;

  // ── Decode path (D): parallel N-tile, proportional CCD allocation ──
  if (max_M <= kDecodeMaxM) {
    int min_M = max_M;
    for (int i = 0; i < num_ops; ++i)
      if (M[i] > 0) min_M = std::min(min_M, M[i]);

    // Decode parallel requires: enough experts, fits in CCDs, and
    // balanced M distribution.  High M-skew (max_M/min_M > 4) causes
    // the proportional CCD allocation to give the large-M expert too
    // many threads relative to its N-tiles, leaving threads idle while
    // the small-M expert (the bottleneck) gets too few.
    // Benchmarked: skew ratio ≤ 4 gives 1.05-1.83x wins on down_proj;
    //              skew ratio > 4 gives 0.68x losses.
    const int skew_ratio = (min_M > 0) ? (max_M / min_M) : max_M;
    const bool decode_parallel = (num_ops >= 6)
        && (num_ops <= num_ccds)
        && (min_M >= 3)
        && (skew_ratio <= 4)
        && (max_N / kDecodeNTile <= team_size_est);

    if (decode_parallel) {
      // Equal thread allocation: each expert gets the same number of
      // threads, capped at N-tiles available.  This avoids the problem
      // with proportional allocation where large-M experts get extra
      // threads they can't use (more threads than N-tiles = idle threads)
      // while small-M experts become the bottleneck.
      const int max_tiles = max_N / kDecodeNTile;
      const int thr_per_expert = std::max(1,
          std::min(num_threads / num_ops, max_tiles));
      const int total_threads = num_ops * thr_per_expert;

      #pragma omp parallel num_threads(total_threads)
      {
        const int tid = omp_get_thread_num();
        const int e = tid / thr_per_expert;
        const int local_tid = tid % thr_per_expert;

        if (e < num_ops) {
          do_tile(e, local_tid, thr_per_expert, kDecodeNTile);
        }

        // Fused activation: require a barrier so every thread's matmul
        // write is globally visible before any thread reads it back for
        // its swiglu_oai epilogue.  Non-fused mode has no barrier here
        // (matches legacy behavior exactly).
        if (fused_epilogue) {
          #pragma omp barrier
          if (e < num_ops) {
            apply_n_tile_paired_swiglu_oai(e, local_tid, thr_per_expert,
                kDecodeNTile, M, N, dst, ldc, act_dtype);
          }
        }
      }
      return;
    }

    // Decode shapes that don't qualify for (D): fall through to paths
    // (A) or (B) below for N-tile with appropriate thread allocation.
  }

  // ── Non-decode paths (max_M > kDecodeMaxM) ──

  // (A) Few experts: adaptive batch N-tiling.
  //
  // Key optimization: instead of running all experts concurrently with
  // few threads each (L3 thrashing for large weights), process in
  // batches sized to maximize threads per expert AND fit weight data
  // in L3.  Batch size = min(num_ops, L3_capacity / weight_per_expert).
  //
  // Example for Mixtral 8 experts, 234MB weight each:
  //   All concurrent: 8×234MB = 1.87GB >> L3 (512MB) → thrashing
  //   Batches of 2:   2×234MB = 468MB ≈ L3 → fits, 63 threads/expert
  if (num_ops <= num_ccds) {
    const size_t wei_per_expert = static_cast<size_t>(max_N)
        * (*std::max_element(K.begin(), K.end())) * wei_elem;
    // Default 512 MB = Zen 4 EPYC aggregate L3.  Other SKUs can override
    // via ZENDNNL_GRP_L3_TOTAL_MB (see group_matmul_parallel_common.hpp).
    const size_t kL3Total = get_grp_l3_total_bytes();
    const int l3_batch = (wei_per_expert > 0)
        ? std::max(1, static_cast<int>(kL3Total / wei_per_expert))
        : num_ops;
    const int batch_size = std::min(num_ops, std::max(1, l3_batch));
    const int max_n_thr = std::max(1, max_N / ab_min_tile);

    #pragma omp parallel num_threads(num_threads)
    {
      const int tid = omp_get_thread_num();

      for (int round_start = 0; round_start < num_ops;
           round_start += batch_size) {
        const int round_end = std::min(num_ops, round_start + batch_size);
        const int round_size = round_end - round_start;
        const int thr_per_expert = std::min(
            num_threads / round_size, max_n_thr);
        const int round_threads = round_size * thr_per_expert;

        int e = -1;
        int local_tid = -1;
        if (tid < round_threads) {
          const int local_expert = tid / thr_per_expert;
          local_tid = tid % thr_per_expert;
          e = round_start + local_expert;
          do_tile(e, local_tid, thr_per_expert, ab_min_tile);
        }
        // Barrier serves two purposes:
        //   1. Next round's matmul must wait for this round's matmul.
        //   2. (Fused) All matmul writes visible before activation.
        #pragma omp barrier

        if (fused_epilogue && e >= 0) {
          apply_n_tile_paired_swiglu_oai(e, local_tid, thr_per_expert,
              ab_min_tile, M, N, dst, ldc, act_dtype);
        }
        // No post-activation barrier: next round's matmul targets a
        // disjoint set of experts (round_start += batch_size) so its
        // reads/writes cannot race with any still-running activation
        // from this round.
      }
    }
    return;
  }

  // (B) Many experts: L3-aware barrier-synchronized rounds.
  //
  // Two competing goals:
  //   1. Minimize rounds (fewer barriers) → more experts per round
  //   2. Keep concurrent weight in L3 → fewer experts per round
  //
  // Example for 512MB shared L3:
  //   down_proj at 16MB/expert: up to 32 experts fit → 1 round when ≤32.
  //   gate+up at 32MB/expert: up to 16 experts fit → rounds ≈ ceil(E/16).
  const int max_tiles = std::max(1, max_N / ab_min_tile);
  const int max_K_val = *std::max_element(K.begin(), K.end());
  const size_t wei_per_expert = static_cast<size_t>(max_N)
      * static_cast<size_t>(max_K_val) * wei_elem;
  const size_t kL3Total = get_grp_l3_total_bytes();

  // L3-aware batch: limit concurrent experts so weights fit in L3.
  const int l3_batch = (wei_per_expert > 0)
      ? std::max(1, static_cast<int>(kL3Total / wei_per_expert))
      : num_ops;

  // Derive L3-aware target batch, then cap so OMP team covers every expert
  // in a round (batch must not exceed num_threads).
  const int target_batch = (l3_batch >= num_ops) ? num_ops
      : std::min(num_ops, l3_batch);
  const int capped_batch = std::max(1, std::min(target_batch, num_threads));
  int n_thr = std::max(1, std::min({ccd_size, max_tiles,
      num_threads / capped_batch}));
  int batch = std::max(1, std::min(capped_batch, num_threads / n_thr));

  #pragma omp parallel num_threads(num_threads)
  {
    const int tid = omp_get_thread_num();

    for (int round_start = 0; round_start < num_ops;
         round_start += batch) {
      const int round_end = std::min(num_ops, round_start + batch);
      const int round_size = round_end - round_start;
      const int round_threads = round_size * n_thr;

      int e = -1;
      int local_tid = -1;
      if (tid < round_threads) {
        const int local_expert = tid / n_thr;
        local_tid = tid % n_thr;
        e = round_start + local_expert;
        do_tile(e, local_tid, n_thr, ab_min_tile);
      }
      // Barrier doubles as "matmul→activation" sync when fused.
      #pragma omp barrier

      if (fused_epilogue && e >= 0) {
        apply_n_tile_paired_swiglu_oai(e, local_tid, n_thr, ab_min_tile,
            M, N, dst, ldc, act_dtype);
      }
    }
  }
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
