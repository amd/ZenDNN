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
/// Self-contained translation unit.  Exposes `flat_m_tile` to the
/// dispatcher via group_matmul_parallel_common.hpp.  All other helpers
/// in this file are private (anonymous-namespace).

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

} // namespace

// ── ALGO=2: planned M-tile — work-balanced, CCD-spread row-parallel ─────
//
// Pre-plans thread assignment based on actual M distribution, then maps
// thread teams onto physical CCDs using a universal CCD-striped layout
// that works for ANY num_threads (128, 126, 127, 192, 256, etc.).
//
// Key insights:
//   1. Each M-tile thread reads the full weight matrix B.  Adding threads
//      to an expert only helps while compute > weight_read.
//   2. OMP thread IDs map to physical cores via KMP_AFFINITY=compact:
//      tid 0..(cores_per_ccd-1) on CCD 0, tid cores..2*cores-1 on CCD 1,
//      etc.  num_threads that aren't a multiple of 8 produce a partial
//      last CCD (e.g., 126t → 15 full CCDs + 1 with 6 cores).
//   3. For cache bandwidth, each expert's weight should touch DISTINCT
//      CCD L3 slices — not cluster on CCDs 0-1.  This is critical when
//      t_assign is small relative to num_threads (decode with few rows).
//
// Planning algorithm (three phases):
//   Phase 1 — Compute ideal threads per expert:
//     t_need[e] = clamp(ceil(M[e] / slice_target), 0, M[e])
//
//   Phase 2 — Fit t_assign to num_threads:
//     (a) total_need ≤ num_threads: distribute surplus to heaviest-load
//         experts (cap at M[e]).  When experts are fully capped (decode
//         with M=1), the remaining threads stay idle.
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
  if (num_ops == 0 || num_threads <= 0) return;

  // Generic ahead-of-time weight pre-pack for ALGO 2.  See
  // sequential_experts in group_matmul_parallel.cpp for the full
  // contract; identical short-circuits here.  `num_threads` is
  // forwarded so cross_warm can prefill regime 2 for the upcoming
  // ALGO 3 decode path when CUSTOM_KERNEL=0 (see the comment on the
  // ALGO 1 call site for the full rationale).
  group_matmul_prepack::prepack_for_algo_2(
      group_matmul_prepack::build_prepack_params(
          weight, K, N, ldb, transB, is_weights_const, params, M,
          get_grp_matmul_custom_kernel(),
          num_threads, /*nr_align=*/0,
          fused_act, act_dtype));

  matmul_algo_t algo = resolve_kernel();

  const size_t src_elem = size_of(params[0].dtypes.src);
  const size_t dst_elem = size_of(params[0].dtypes.dst);

  scoped_active_levels guard(1);

  // ── Phase 1: count active experts ──
  int active_ops = 0;
  for (int i = 0; i < num_ops; ++i)
    if (M[i] > 0) ++active_ops;
  if (active_ops == 0) return;

  // ── CCD topology (universal: handles any num_threads) ──
  // Zen 3/4/5: 8 cores per CCD.  Last CCD may be partial when
  // num_threads % 8 != 0 (e.g., 126t → CCDs 0..14 full, CCD 15 has 6).
  const int cores_per_ccd = std::min(8, num_threads);
  const int num_ccds = std::max(1,
      (num_threads + cores_per_ccd - 1) / cores_per_ccd);
  auto ccd_capacity = [&](int c) -> int {
    const int base = c * cores_per_ccd;
    return std::max(0, std::min(cores_per_ccd, num_threads - base));
  };

  // ── Many active experts > num_threads: round-based, CCD-spread rounds ──
  // Each round runs up to num_threads experts concurrently (1 thread each).
  // Thread tid handles expert active_idx[round + tid].  Thread IDs map
  // naturally onto CCDs via KMP_AFFINITY compact — experts are implicitly
  // spread across CCDs.
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

  // ── Phase 1b: initial t_assign based on target slice size ──
  std::vector<int> t_assign(num_ops, 0);
  int total_need = 0;
  for (int i = 0; i < num_ops; ++i) {
    if (M[i] <= 0) continue;
    t_assign[i] = std::min(M[i],
        std::max(1, (M[i] + kSliceTarget - 1) / kSliceTarget));
    total_need += t_assign[i];
  }

  // ── Phase 2: fit t_assign to num_threads ──
  //
  // CCD-aware cap: when num_ops <= num_ccds AND every expert could fit
  // in a single CCD (num_ops * cores_per_ccd >= num_threads), cap each
  // expert at its assigned CCD's capacity.  This prevents an expert's
  // team from spanning CCD boundaries, which would cause L3 contention
  // with a neighbor expert's weight.
  //
  // Critical for num_threads not a clean multiple of cores_per_ccd.
  // Example 16 experts × M=8 at 126t (cores_per_ccd=8, num_ccds=16):
  //   Without cap: t_assign = [8]*14 + [7]*2 — last 2 experts cram into
  //     CCDs 14 and 15, expert 15 spills 1 thread onto CCD 14 (L3 contention).
  //   With cap:    t_assign = [8]*15 + [6]   — expert 15 fits on partial
  //     CCD 15 (6 slots); other 15 experts each own their CCD.
  const bool cap_at_ccd = (num_ops <= num_ccds)
      && (static_cast<int64_t>(num_ops) * cores_per_ccd >= num_threads);

  if (total_need <= num_threads) {
    // Distribute surplus threads to experts with the heaviest per-thread
    // slice.  Capped at M[e] and optionally at the expert's CCD capacity.
    int surplus = num_threads - total_need;
    while (surplus > 0) {
      int best = -1;
      int best_slice = 0;
      for (int i = 0; i < num_ops; ++i) {
        if (t_assign[i] <= 0 || t_assign[i] >= M[i]) continue;
        if (cap_at_ccd) {
          const int my_ccd = i % num_ccds;
          if (t_assign[i] >= ccd_capacity(my_ccd)) continue;
        }
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
    // Scale down proportionally to M[e], floor at 1 per active expert.
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
    while (assigned < num_threads) {
      int best = -1, best_slice = 0;
      for (int i = 0; i < num_ops; ++i) {
        if (t_assign[i] <= 0 || t_assign[i] >= M[i]) continue;
        if (cap_at_ccd) {
          const int my_ccd = i % num_ccds;
          if (t_assign[i] >= ccd_capacity(my_ccd)) continue;
        }
        int cur_slice = (M[i] + t_assign[i] - 1) / t_assign[i];
        if (cur_slice > best_slice) { best_slice = cur_slice; best = i; }
      }
      if (best < 0) break;
      ++t_assign[best];
      ++assigned;
    }
    while (assigned > num_threads) {
      int best = -1, least_slice = INT_MAX;
      for (int i = 0; i < num_ops; ++i) {
        if (t_assign[i] <= 1) continue;
        int cur_slice = (M[i] + t_assign[i] - 1) / t_assign[i];
        if (cur_slice < least_slice) { least_slice = cur_slice; best = i; }
      }
      if (best < 0) break;
      --t_assign[best];
      --assigned;
    }
  }

  // ── Phase 3: CCD-striped thread→expert mapping ──
  //
  // Universal layout: works for any num_threads and any t_assign.
  // Each expert's threads are placed sequentially on a starting CCD
  // (rotates per expert for load balance), packing up to cores_per_ccd
  // threads per CCD before spilling to the next.
  //
  // Examples:
  //   128t, 16 experts × 1 thr: each expert → distinct CCD (local 0).
  //     CCD 0..15 each host 1 expert using 1 of 8 cores.
  //   126t, 16 experts × 1 thr: each expert → distinct CCD (local 0).
  //     CCDs 0..15 host 1 expert each; CCD 15 has only 6 cores but
  //     only needs 1.
  //   128t, 8 experts × 16 thr: each expert spans 2 adjacent CCDs.
  //     CCD 0-1 host expert 0, CCD 2-3 host expert 1, etc.
  //   126t, 8 experts × ~16 thr: last expert truncated to fit 126.

  std::vector<int> tid_to_expert(num_threads, -1);
  std::vector<int> tid_to_local(num_threads, -1);
  std::vector<int> tid_to_team(num_threads, 0);
  std::vector<int> ccd_used(num_ccds, 0);

  // Starting CCD is deterministic per expert index (e % num_ccds).  This
  // matches the cap_at_ccd assumption above (my_ccd = i % num_ccds) exactly,
  // regardless of which experts are inactive.  If an earlier expert has
  // already filled this CCD, the while-loop below skips forward to the next
  // CCD with remaining capacity.
  for (int e = 0; e < num_ops; ++e) {
    if (t_assign[e] <= 0) continue;
    const int t = t_assign[e];
    int placed = 0;
    int c = e % num_ccds;

    while (placed < t) {
      // Skip CCDs that are full.
      int tries = 0;
      while (ccd_used[c] >= ccd_capacity(c) && tries < num_ccds) {
        c = (c + 1) % num_ccds;
        ++tries;
      }
      if (tries >= num_ccds) break;  // no capacity (shouldn't happen)

      // Place as many threads on CCD c as fit (up to remaining capacity).
      const int cap = ccd_capacity(c) - ccd_used[c];
      const int can_place = std::min(cap, t - placed);
      for (int k = 0; k < can_place; ++k) {
        const int local = ccd_used[c] + k;
        const int tid = c * cores_per_ccd + local;
        if (tid < num_threads) {
          tid_to_expert[tid] = e;
          tid_to_local[tid] = placed + k;
          tid_to_team[tid] = t;
        }
      }
      ccd_used[c] += can_place;
      placed += can_place;
      if (placed < t) c = (c + 1) % num_ccds;
    }
  }

  // ── Execute: always use full num_threads OMP team ──
  // Threads without a slot assignment simply exit.  Using the full team
  // ensures consistent physical thread placement across CCDs regardless
  // of how many threads are actively doing work.
  #pragma omp parallel num_threads(num_threads)
  {
    const int tid = omp_get_thread_num();
    const int e = tid_to_expert[tid];
    if (e >= 0) {
      execute_m_tile_act(e, tid_to_local[tid], tid_to_team[tid],
          layout, transA, transB, M, N, K, alpha,
          src, lda, weight, ldb, bias, beta, dst, ldc,
          is_weights_const, params, src_elem, dst_elem, algo,
          fused_act, act_dtype);
    }
  }
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
