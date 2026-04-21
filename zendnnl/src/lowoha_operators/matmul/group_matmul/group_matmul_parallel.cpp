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


#include <algorithm>
#include <climits>
#include <cstdlib>
#include <vector>

#include <omp.h>

#include "group_matmul_direct.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"
#include "lowoha_operators/common/omp_thread_control.hpp"
#include "operators/matmul/matmul_config.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

using namespace zendnnl::ops;
using zendnnl::common::size_of;

namespace {

// Read on every call (not cached) so that tests can switch ALGOs
// via setenv within the same process.
inline int get_grp_matmul_algo() {
  const char *env = std::getenv("ZENDNNL_GRP_MATMUL_ALGO");
  return (env && env[0] >= '1' && env[0] <= '5') ? (env[0] - '0') : 0;
}

inline matmul_algo_t resolve_kernel() {
  static const matmul_algo_t algo = []() {
    int32_t a = matmul_config_t::instance().get_algo();
    if (a <= 0 || a >= static_cast<int32_t>(matmul_algo_t::algo_count))
      return matmul_algo_t::aocl_dlp_blocked;
    return static_cast<matmul_algo_t>(a);
  }();
  return algo;
}

inline void execute_expert_slice(
    char layout, bool transA, bool transB,
    int M, int N, int K, float alpha,
    const void *src, int lda,
    const void *weight, int ldb,
    const void *bias, float beta,
    void *dst, int ldc,
    bool is_weights_const, int num_thr,
    matmul_params &params,
    matmul_algo_t algo) {

  matmul_batch_params_t bp;
  bp.Batch_A = 1;
  bp.Batch_B = 1;
  matmul_algo_t kernel = algo;
  matmul_execute(layout, transA, transB,
      M, N, K, alpha, src, lda, weight, ldb, bias, beta, dst, ldc,
      is_weights_const, size_of(params.dtypes.src), size_of(params.dtypes.dst),
      num_thr, kernel, params, bp, 0);
}

// ── ALGO=1: sequential — experts run one at a time, all threads per GEMM
// Each expert's GEMM uses all num_threads for maximum per-op parallelism.
// No inter-expert parallelism — experts are serialized.

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

// ── Shared tile helpers ─────────────────────────────────────────────────
//
// Tiling pointer arithmetic assumes row-major layout (layout == 'r'/'R').
// Column-major is not supported by tiled algorithms (ALGO 2/3).
// Dtype uniformity is enforced by m_tile_safe / n_tile_safe in
// select_grp_matmul_algo.
//
// Per-thread params copy (thread_local) avoids data race: concurrent
// tiles of the same expert must not share mutable matmul_params.

// Minimum N-tile width.  Below this, kernel startup overhead and poor
// register utilization dominate.  Threads beyond N/min_n_tile are idle.
static constexpr int kMinNTile = 512;

// Decode threshold: shapes with max_M at or below this value are
// bandwidth-bound (arithmetic intensity < machine balance point).
static constexpr int kDecodeMaxM = 32;

// Row-offset src quant buffer for M-tile.
// Handles per-token {M,1} and per-group {M,G}: offset = row_start × row_stride.
// Per-tensor (dims empty, total elements == 1, or first dim == 1): no offset.
inline void offset_quant_by_row(
    matmul_quantization_params_t::matmul_quant_t &q,
    int row_start) {
  if (q.buff == nullptr) return;
  if (q.dims.empty()) return;
  int64_t nelems = 1;
  for (auto dim : q.dims) {
    if (dim <= 0) return;
    nelems *= dim;
  }
  if (nelems <= 1 || q.dims[0] <= 1) return;
  const size_t rows = static_cast<size_t>(q.dims[0]);
  if (rows == 0 || (static_cast<size_t>(nelems) % rows) != 0) return;
  const size_t row_stride = static_cast<size_t>(nelems) / rows;
  const size_t elem = size_of(q.dt);
  q.buff = static_cast<const uint8_t *>(q.buff)
      + static_cast<size_t>(row_start) * row_stride * elem;
}

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

inline void execute_m_tile(
    int e, int local_tid, int team_size,
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
    size_t src_elem, size_t dst_elem, matmul_algo_t algo) {

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

  // Row-offset per-token src quantization (dims={M,1} or similar).
  // Per-tensor (dims empty or {1}) needs no offset.
  // Wei quant is N-dependent but M-tile keeps full N → unchanged.
  offset_quant_by_row(slice_params.quant_params.src_scale, row_start);
  offset_quant_by_row(slice_params.quant_params.src_zp, row_start);

  execute_expert_slice(layout[e], transA[e], transB[e],
      slice_M, N[e], K[e], alpha[e],
      s, lda[e], weight[e], ldb[e],
      bias[e], beta[e], d, ldc[e],
      is_weights_const[e], 1, slice_params, algo);
}

// M-tile with fused activation: apply immediately after GEMM while
// output rows are hot in L1/L2.
inline void execute_m_tile_act(
    int e, int local_tid, int team_size,
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
    size_t src_elem, size_t dst_elem, matmul_algo_t algo,
    grp_matmul_gated_act_t fused_act, data_type_t act_dtype) {

  execute_m_tile(e, local_tid, team_size, layout, transA, transB,
      M, N, K, alpha, src, lda, weight, ldb, bias, beta, dst, ldc,
      is_weights_const, params, src_elem, dst_elem, algo);

  if (fused_act != grp_matmul_gated_act_t::none) {
    const int row_start = static_cast<int>(
        static_cast<int64_t>(M[e]) * local_tid / team_size);
    const int row_end = static_cast<int>(
        static_cast<int64_t>(M[e]) * (local_tid + 1) / team_size);
    if (row_start < row_end)
      apply_gated_act_inplace(fused_act, dst[e], row_start, row_end,
                              N[e], ldc[e], act_dtype);
  }
}

// ── ALGO=2: planned M-tile — work-balanced row-parallel ─────────────────
//
// Pre-plans thread assignment based on actual M distribution across all
// experts BEFORE execution.  Each thread handles a contiguous slice of
// M rows for its assigned expert.  Framework-safe: flat OMP, num_thr=1.
//
// Key insight: each M-tile thread reads the full weight matrix B.
// Adding threads to an expert only helps while compute > weight_read.
// Beyond that, extra threads just add redundant DRAM traffic.
//
// Planning algorithm (three phases):
//   Phase 1 — Compute threads needed per expert:
//     t_need[e] = clamp(ceil(M[e] / slice_target), 0, M[e])
//     M[e]=0 → 0 threads (skip expert entirely).
//     M[e]=1..slice_target → 1 thread (no slicing benefit).
//     M[e]>slice_target → enough threads for ~slice_target rows each.
//
//   Phase 2 — Fit into num_threads:
//     (a) total_need ≤ num_threads: assign t_need, then distribute
//         surplus one-at-a-time to the expert with the largest
//         per-thread slice (heaviest load).  Cap at M[e] threads
//         per expert (can't have more threads than rows).
//     (b) total_need > num_threads: scale down proportionally by M[e],
//         floor at 1 (every non-zero-M expert gets at least 1 thread).
//         Fix rounding to hit exactly num_threads.
//
//   Phase 3 — Build contiguous thread→expert map and execute in flat OMP.
//
// No decode fallback: ALGO 2 always uses the M-tiling planner regardless
// of M size.  For small M (decode), the planner gives each expert 1 thread
// (kSliceTarget=16, M<16 → 1 thread).  ALGO 0 routes decode to ALGO 3.

// Minimum rows per M-tile thread for compute to dominate weight read.
static constexpr int kSliceTarget = 16;

void flat_m_tile(
    const std::vector<char> &layout,
    const std::vector<bool> &transA, const std::vector<bool> &transB,
    const std::vector<int> &M, const std::vector<int> &N,
    const std::vector<int> &K, const std::vector<float> &alpha,
    const std::vector<const void *> &src, const std::vector<int> &lda,
    const std::vector<const void *> &weight, const std::vector<int> &ldb,
    const std::vector<const void *> &bias, const std::vector<float> &beta,
    const std::vector<void *> &dst, const std::vector<int> &ldc,
    grp_matmul_gated_act_t fused_act, data_type_t act_dtype,
    const std::vector<bool> &is_weights_const,
    std::vector<matmul_params> &params,
    int num_threads) {

  const int num_ops = static_cast<int>(M.size());
  if (num_ops == 0) return;

  matmul_algo_t algo = resolve_kernel();

  const size_t src_elem = size_of(params[0].dtypes.src);
  const size_t dst_elem = size_of(params[0].dtypes.dst);

  scoped_active_levels guard(1);

  // NOTE: no decode fallback here.  When ALGO 2 is selected (manually
  // or by auto-select), it always M-tiles.  For small M (decode), the
  // planner gives each expert 1 thread (kSliceTarget=16, M<16 → 1 thread),
  // which is functionally equivalent to sequential BLAS but runs all
  // experts in parallel via the flat OMP region.

  // ── Phase 1: compute ideal threads per expert ──

  int active_ops = 0;
  for (int i = 0; i < num_ops; ++i)
    if (M[i] > 0) ++active_ops;
  if (active_ops == 0) return;

  // When more active experts than threads, process in sequential rounds
  // of num_threads experts each.  Each expert gets 1 thread with the
  // full M (no M-tiling) — equivalent to per-expert parallel dispatch.
  // Build a compact list of active expert indices to skip inactive
  // experts and avoid unnecessary barrier rounds.
  if (active_ops > num_threads) {
    std::vector<int> active_idx;
    active_idx.reserve(active_ops);
    for (int i = 0; i < num_ops; ++i)
      if (M[i] > 0) active_idx.push_back(i);

    #pragma omp parallel num_threads(num_threads)
    {
      const int tid = omp_get_thread_num();
      for (int round = 0; round < active_ops; round += num_threads) {
        const int slot = round + tid;
        if (slot < active_ops) {
          const int e = active_idx[slot];
          static thread_local matmul_params local_params;
          local_params = params[e];
          execute_expert_slice(layout[e], transA[e], transB[e],
              M[e], N[e], K[e], alpha[e],
              src[e], lda[e], weight[e], ldb[e],
              bias[e], beta[e], dst[e], ldc[e],
              is_weights_const[e], 1, local_params, algo);
          if (fused_act != grp_matmul_gated_act_t::none)
            apply_gated_act_inplace(fused_act, dst[e], 0, M[e],
                                    N[e], ldc[e], act_dtype);
        }
        #pragma omp barrier
      }
    }
    return;
  }

  // active_ops <= num_threads: every active expert gets at least 1 thread.
  std::vector<int> t_assign(num_ops, 0);
  int total_need = 0;
  for (int i = 0; i < num_ops; ++i) {
    if (M[i] <= 0) continue;
    t_assign[i] = std::min(M[i],
        std::max(1, (M[i] + kSliceTarget - 1) / kSliceTarget));
    total_need += t_assign[i];
  }

  // ── Phase 2: fit into num_threads ──

  if (total_need <= num_threads) {
    // (a) Surplus: distribute extra threads to the most loaded experts.
    int surplus = num_threads - total_need;
    while (surplus > 0) {
      int best = -1;
      int best_slice = 0;
      for (int i = 0; i < num_ops; ++i) {
        if (t_assign[i] <= 0 || t_assign[i] >= M[i]) continue;
        int cur_slice = (M[i] + t_assign[i] - 1) / t_assign[i];
        if (cur_slice > best_slice) {
          best_slice = cur_slice;
          best = i;
        }
      }
      if (best < 0) break;
      ++t_assign[best];
      --surplus;
    }
  } else {
    // (b) Scale down: proportional to M.  Since active_ops <= num_threads,
    //     every active expert can get at least 1 thread.
    int64_t total_M = 0;
    for (int i = 0; i < num_ops; ++i) total_M += M[i];
    if (total_M <= 0) return;

    int assigned = 0;
    for (int i = 0; i < num_ops; ++i) {
      if (M[i] <= 0) { t_assign[i] = 0; continue; }
      t_assign[i] = std::max(1, static_cast<int>(
          static_cast<int64_t>(num_threads) * M[i] / total_M));
      assigned += t_assign[i];
    }

    // Fix rounding to hit exactly num_threads.
    while (assigned < num_threads) {
      int best = -1;
      int best_slice = 0;
      for (int i = 0; i < num_ops; ++i) {
        if (t_assign[i] <= 0 || t_assign[i] >= M[i]) continue;
        int cur_slice = (M[i] + t_assign[i] - 1) / t_assign[i];
        if (cur_slice > best_slice) {
          best_slice = cur_slice;
          best = i;
        }
      }
      if (best < 0) break;
      ++t_assign[best];
      ++assigned;
    }
    while (assigned > num_threads) {
      int best = -1;
      int least_slice = INT_MAX;
      for (int i = 0; i < num_ops; ++i) {
        if (t_assign[i] <= 1) continue;
        int cur_slice = (M[i] + t_assign[i] - 1) / t_assign[i];
        if (cur_slice < least_slice) {
          least_slice = cur_slice;
          best = i;
        }
      }
      if (best < 0) break;
      --t_assign[best];
      --assigned;
    }
  }

  // ── Phase 3: build thread→expert map and execute ──

  std::vector<int> thr_start(num_ops + 1, 0);
  for (int i = 0; i < num_ops; ++i)
    thr_start[i + 1] = thr_start[i] + t_assign[i];
  const int total_threads = thr_start[num_ops];
  if (total_threads <= 0) return;

  // ── CCD-aware row spreading for small M ──
  //
  // When total_threads < num_threads (under-utilizing cores), the default
  // OMP scheduling packs all active threads on CCD 0, saturating one L3
  // slice while other CCDs are idle (see diagram: "Default OMP schedule").
  //
  // CCD-aware dispatch spreads experts across CCDs: expert i maps to
  // CCD (i % num_ccds), using local threads within that CCD.  Each
  // expert's weight read then hits a different CCD's L3, multiplying
  // effective memory bandwidth by up to num_ccds.
  //
  // Triggered when: total assigned threads fit within one CCD per expert
  // and we have idle CCDs to exploit.
  const int cores_per_ccd = std::min(8, num_threads);
  const int num_ccds_m = std::max(1, num_threads / cores_per_ccd);
  const int max_t_assign = *std::max_element(t_assign.begin(), t_assign.end());

  const bool ccd_spread = (total_threads < num_threads)
      && (active_ops <= num_ccds_m)
      && (max_t_assign <= cores_per_ccd);

  if (ccd_spread) {
    // Map each active expert to a distinct CCD.
    // Expert at active index j → CCD j → global threads [j*cores_per_ccd, ...).
    // Only the first t_assign[e] threads on that CCD are active.
    std::vector<int> expert_to_ccd(num_ops, -1);
    {
      int ccd = 0;
      for (int i = 0; i < num_ops; ++i) {
        if (t_assign[i] > 0) {
          expert_to_ccd[i] = ccd++;
        }
      }
    }

    #pragma omp parallel num_threads(num_threads)
    {
      const int tid = omp_get_thread_num();
      const int my_ccd = tid / cores_per_ccd;
      const int local_id = tid % cores_per_ccd;

      // Find which expert maps to my CCD
      for (int e = 0; e < num_ops; ++e) {
        if (expert_to_ccd[e] == my_ccd && local_id < t_assign[e]) {
          execute_m_tile_act(e, local_id, t_assign[e],
              layout, transA, transB, M, N, K, alpha,
              src, lda, weight, ldb, bias, beta, dst, ldc,
              is_weights_const, params, src_elem, dst_elem, algo,
              fused_act, act_dtype);
          break;
        }
      }
    }
  } else {
    // Standard packed scheduling (total_threads covers most/all CCDs).
    #pragma omp parallel num_threads(total_threads)
    {
      const int tid = omp_get_thread_num();
      int e = 0;
      while (e < num_ops - 1 && tid >= thr_start[e + 1]) ++e;

      const int team_size = thr_start[e + 1] - thr_start[e];
      const int local_tid = tid - thr_start[e];

      if (team_size > 0) {
        execute_m_tile_act(e, local_tid, team_size,
            layout, transA, transB, M, N, K, alpha,
            src, lda, weight, ldb, bias, beta, dst, ldc,
            is_weights_const, params, src_elem, dst_elem, algo,
            fused_act, act_dtype);
      }
    }
  }
}

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

static constexpr int kDecodeNTile = 256;

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
    int num_threads) {

  const int num_ops = static_cast<int>(M.size());
  if (num_ops == 0) return;

  matmul_algo_t algo = resolve_kernel();
  const int ccd_size = std::min(8, num_threads);
  const int num_ccds = std::max(1, num_threads / ccd_size);

  const size_t wei_elem = size_of(params[0].dtypes.wei);
  const size_t dst_elem = size_of(params[0].dtypes.dst);
  const size_t bias_elem = (params[0].dtypes.bias != data_type_t::none)
      ? size_of(params[0].dtypes.bias) : sizeof(float);

  const int max_M = *std::max_element(M.begin(), M.end());
  const int max_N = *std::max_element(N.begin(), N.end());

  scoped_active_levels guard(1);

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

  // When N is too small for useful N-tiling (ntile_viable=false),
  // use per-expert parallel dispatch (1 thread per expert, all experts
  // concurrent) instead of sequential BLAS.  This is better than ALGO 1
  // when num_ops > 1 because experts run in parallel.
  if (!ntile_viable) {
    scoped_active_levels guard_inner(1);
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
    }
    return;
  }

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
          execute_n_tile(e, local_tid, thr_per_expert, kDecodeNTile,
              layout, transA, transB, M, N, K, alpha,
              src, lda, weight, ldb, bias, beta, dst, ldc,
              is_weights_const, params, wei_elem, dst_elem, bias_elem, algo);
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
    // Zen4/5 typical total L3: 512MB (8 CCDs × 32MB + shared victim).
    // TODO: derive from runtime uarch detection when platform_info
    // exposes per-CCD L3 capacity.
    static constexpr size_t kL3Total = 512UL * 1024 * 1024;
    const int l3_batch = (wei_per_expert > 0)
        ? std::max(1, static_cast<int>(kL3Total / wei_per_expert))
        : num_ops;
    const int batch_size = std::min(num_ops, std::max(1, l3_batch));
    const int max_n_thr = std::max(1, max_N / kMinNTile);

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

        if (tid < round_threads) {
          const int local_expert = tid / thr_per_expert;
          const int local_tid = tid % thr_per_expert;
          const int e = round_start + local_expert;
          execute_n_tile(e, local_tid, thr_per_expert, kMinNTile,
              layout, transA, transB, M, N, K, alpha,
              src, lda, weight, ldb, bias, beta, dst, ldc,
              is_weights_const, params, wei_elem, dst_elem, bias_elem, algo);
        }
        #pragma omp barrier
      }
    }
    return;
  }

  // (B) Many experts: barrier-synchronized rounds, adaptive n_thr.
  const int n_thr = std::max(1, std::min(ccd_size, max_N / kMinNTile));
  const int batch = num_threads / n_thr;

  #pragma omp parallel num_threads(num_threads)
  {
    const int tid = omp_get_thread_num();

    for (int round_start = 0; round_start < num_ops;
         round_start += batch) {
      const int round_end = std::min(num_ops, round_start + batch);
      const int round_size = round_end - round_start;
      const int round_threads = round_size * n_thr;

      if (tid < round_threads) {
        const int local_expert = tid / n_thr;
        const int local_tid = tid % n_thr;
        execute_n_tile(round_start + local_expert, local_tid, n_thr,
            kMinNTile,
            layout, transA, transB, M, N, K, alpha,
            src, lda, weight, ldb, bias, beta, dst, ldc,
            is_weights_const, params, wei_elem, dst_elem, bias_elem,
            algo);
      }
      #pragma omp barrier
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
  matmul_algo_t algo = resolve_kernel();

  const int ccd_size = std::min(8, num_threads);
  const int num_ccds = std::max(1, num_threads / ccd_size);
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
  // Data-driven decision tree based on benchmarks across three models:
  //   Mixtral-8x7B  (224MB/expert, 8 experts)    → ALGO 1 always
  //   GPT-OSS       (32MB/expert, 2-32 experts)   → ALGO 3 for 12+ experts
  //   Qwen3-30B     (6MB/expert, 8-128 experts)   → ALGO 3 for 8+, ALGO 2 for 100+
  //
  // Primary discriminator: weight_per_expert (determines L3 fit).
  // Secondary: num_ops (expert count) and max_M (decode vs prompt).
  //
  // Weight thresholds (BF16, per-expert):
  //   Small  (≤ 16MB): fits in per-CCD L3 → concurrent experts win
  //   Medium (16-64MB): boundary zone → ALGO 3 for many experts
  //   Large  (> 64MB): streams from DRAM → sequential ALGO 1

  if (num_threads <= 1 || num_ops == 0)
    return 1;

  const int max_M = *std::max_element(M.begin(), M.end());
  const int max_N = *std::max_element(N.begin(), N.end());
  const int max_K = *std::max_element(K.begin(), K.end());

  // Weight size per expert in bytes (max across K/N combinations).
  const size_t wei_elem = size_of(params[0].dtypes.wei);
  const size_t weight_per_expert =
      static_cast<size_t>(max_K) * max_N * wei_elem;

  static constexpr size_t kSmallWeight  = 16UL * 1024 * 1024;  // 16MB
  static constexpr size_t kMediumWeight = 64UL * 1024 * 1024;  // 64MB

  // ── Few experts: ALGO 1 always (any weight size) ──
  if (num_ops <= 4)
    return 1;

  // ── Large weights (> 64MB): Mixtral-class ──
  // AOCL DLP's internal panel blocking with all threads is unbeatable.
  // Benchmarked: ALGO 1 wins 144/144 on Mixtral (224MB/112MB weights).
  if (weight_per_expert > kMediumWeight)
    return 1;

  // ── Small weights (≤ 16MB): Qwen3/DeepSeek/Switch-class ──
  // Weights fit in per-CCD L3 → concurrent expert dispatch gives 2-5x.
  if (weight_per_expert <= kSmallWeight) {
    // More experts than threads: per-expert parallel.
    if (num_ops > num_threads) return 5;
    // 100+ experts with prompt-size M: CCD-aware M-tile wins (2.5-3.3x).
    if (num_ops >= 100 && max_M > kDecodeMaxM) {
      if (m_tile_safe) return 2;
    }
    // 8+ experts: N-tile wins for both decode and prompt (up to 5x).
    if (num_ops >= 8) {
      if (n_tile_safe) return 3;
      if (m_tile_safe) return 2;
      return 5;
    }
    // 5-7 experts: ALGO 3 if safe, else ALGO 1.
    if (num_ops >= 5 && n_tile_safe) return 3;
    return 1;
  }

  // ── Medium weights (16-64MB): GPT-OSS-class ──
  // Weights partially fit in L3 → ALGO 3 N-tiling helps for many experts.
  // Benchmarked: A3 wins 22/38 on GPT-OSS (32MB), A2 wins on large prompt.
  if (num_ops >= 12) {
    if (n_tile_safe) return 3;
    if (m_tile_safe) return 2;
  }
  // 6-11 experts, prompt-size M: ALGO 3 still helps (1.2-1.4x).
  if (num_ops >= 6 && max_M > kDecodeMaxM) {
    if (n_tile_safe) return 3;
  }
  // M-tile for quant/packed when n_tile fails.
  if (m_tile_safe && max_M > kDecodeMaxM && num_ops >= 6)
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

  // ALGO 3 (N-tile) cannot fuse split-layout activation (gate/up in
  // different column ranges).  All other ALGOs fuse activation inline.
  const bool act_fused = (use_algo != 3)
      && (fused_act != grp_matmul_gated_act_t::none);

  switch (use_algo) {
  case 1:
    if (gemm_mode_out != nullptr)
      *gemm_mode_out = "sequential_experts";
    sequential_experts(layout, transA, transB, M, N, K, alpha,
        src, lda, weight, ldb, bias, beta, dst, ldc,
        is_weights_const, params, num_threads, fused_act, act_dtype);
    break;
  case 2:
    if (gemm_mode_out != nullptr)
      *gemm_mode_out = "flat_m_tile";
    flat_m_tile(layout, transA, transB, M, N, K, alpha,
        src, lda, weight, ldb, bias, beta, dst, ldc,
        fused_act, act_dtype, is_weights_const, params, num_threads);
    break;
  case 3:
    if (gemm_mode_out != nullptr)
      *gemm_mode_out = "flat_n_tile";
    flat_n_tile(layout, transA, transB, M, N, K, alpha,
        src, lda, weight, ldb, bias, beta, dst, ldc,
        is_weights_const, params, num_threads);
    break;
  case 4:
    if (gemm_mode_out != nullptr)
      *gemm_mode_out = "multilevel";
    parallel_multilevel(layout, transA, transB, M, N, K, alpha,
        src, lda, weight, ldb, bias, beta, dst, ldc,
        is_weights_const, params, num_threads, fused_act, act_dtype);
    break;
  case 5:
    if (gemm_mode_out != nullptr)
      *gemm_mode_out = "per_expert";
    parallel_per_expert(layout, transA, transB, M, N, K, alpha,
        src, lda, weight, ldb, bias, beta, dst, ldc,
        is_weights_const, params, num_threads, fused_act, act_dtype);
    break;
  default:
    if (gemm_mode_out != nullptr)
      *gemm_mode_out = "sequential_experts";
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
