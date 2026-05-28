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
#include <atomic>
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
//
// `slice_M` rewrites the slice's view of dims[0] from the full per-expert M
// down to the per-thread row count.  This is required for dynamic-quant: the
// reorder dispatcher gates the per-group kernel on `is_per_group_col_dims`,
// which compares `dims[0]` against `src_shape[0]` (= slice_M for the M-tile).
// Static-quant kernels that read scales by row index also benefit.
inline void offset_quant_by_row(
    matmul_quantization_params_t::matmul_quant_t &q,
    int row_start, int slice_M) {
  if (q.dims.empty()) return;
  int64_t nelems = 1;
  for (auto dim : q.dims) {
    if (dim <= 0) return;
    nelems *= dim;
  }
  if (nelems <= 1 || q.dims[0] <= 1) return;
  const size_t rows = static_cast<size_t>(q.dims[0]);
  if (rows == 0 || (static_cast<size_t>(nelems) % rows) != 0) return;
  if (q.buff != nullptr) {
    const size_t row_stride = static_cast<size_t>(nelems) / rows;
    const size_t elem = size_of(q.dt);
    q.buff = static_cast<const uint8_t *>(q.buff)
        + static_cast<size_t>(row_start) * row_stride * elem;
  }
  if (slice_M > 0 && static_cast<size_t>(slice_M) <= rows) {
    q.dims[0] = static_cast<int64_t>(slice_M);
  }
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

  // Row-offset per-token / per-group src quantization (dims={M,1} or {M,G}).
  // Also rewrites dims[0] from the full M to slice_M so the dynamic-quant
  // reorder dispatcher's per-group gate (is_per_group_col_dims) matches the
  // sliced src_shape.  Per-tensor (dims empty or {1}) needs no offset.
  // Wei quant is N-dependent but M-tile keeps full N → unchanged.
  offset_quant_by_row(slice_params.quant_params.src_scale, row_start, slice_M);
  offset_quant_by_row(slice_params.quant_params.src_zp, row_start, slice_M);

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

// ── Multi-tier "light pool" helper ─────────────────────────────────────
//
// Each light-pool thread processes a strided slice of light experts
// sequentially with `team=1` (full M, no slicing).  Memory-traffic
// pattern is essentially identical to ALGO 1's sequential_experts on
// these tiny experts — light expert work is dominated by the per-call
// activation kernel-launch overhead, not per-thread arithmetic, so
// extra threads on the same light expert would idle on the OMP barrier
// rather than speed it up.  Pulling them out of the standard t_assign
// pool is the whole point.
//
// Used from the multi-tier branch in `flat_m_tile` when the AUTO
// gating engages.  See `get_grp_matmul_m_tile_hybrid()` for the
// gating heuristic.
inline void execute_light_expert(
    int e,
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
    matmul_algo_t algo,
    grp_matmul_gated_act_t fused_act, data_type_t act_dtype) {
  static thread_local matmul_params local_params;
  local_params = params[e];
  execute_expert_slice(layout[e], transA[e], transB[e],
      M[e], N[e], K[e], alpha[e],
      src[e], lda[e], weight[e], ldb[e],
      bias[e], beta[e], dst[e], ldc[e],
      is_weights_const[e], 1, local_params, algo);
  if (fused_act != grp_matmul_gated_act_t::none) {
    apply_gated_act_inplace(fused_act, dst[e], 0, M[e],
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
// Tunable via `ZENDNNL_GRP_MATMUL_M_TILE_SLICE_TARGET`
// (see `get_grp_matmul_m_tile_slice_target` in
// `group_matmul_parallel_common.hpp`).  Defaulted to 16 to match
// AOCL-DLP's BRGEMM row-block quantum; cached at flat_m_tile entry.

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
          fused_act, act_dtype,
          /*transA=*/&transA, /*alpha=*/&alpha, /*beta=*/&beta));

  matmul_algo_t algo = resolve_kernel();

  const size_t src_elem = size_of(params[0].dtypes.src);
  const size_t dst_elem = size_of(params[0].dtypes.dst);

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
  // CCD idle (e.g. Mixtral 8-expert MoE on 64t with one unrouted
  // expert wasted exactly one CCD = 8 / 64 = 12.5 % of the budget).
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

  // F1 — One-time HINT when flat_m_tile is pinned on a decode-class
  // call (max_M == 1) AND the active count is so small that more
  // than 75 % of `num_threads` will sit on the Phase-2 join barrier
  // (Mixtral E=8 / 8 active on 128t: 120/128 = 93.75 % idle; the
  // single-thread-per-active CCD-parallel mapping is correct for
  // that shape, but the user almost certainly meant
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
    static std::atomic<bool> s_hint_fired{false};
    static const bool s_hint_log = apilog_info_enabled();
    if (s_hint_log
        && !s_hint_fired.load(std::memory_order_relaxed)
        && active_ops * 4 < num_threads) {
      int max_M_for_hint = 0;
      for (int i = 0; i < num_ops; ++i) {
        if (M[i] > max_M_for_hint) max_M_for_hint = M[i];
      }
      if (max_M_for_hint == 1) {
        bool expected = false;
        if (s_hint_fired.compare_exchange_strong(
                expected, true, std::memory_order_relaxed)) {
          const int idle_pct = (num_threads - active_ops) * 100
                             / num_threads;
          apilog_info(
              "[GRP_MATMUL.HINT] ALGO_2 (flat_m_tile) invoked on a "
              "decode-class call (max_M=1, active_ops=", active_ops,
              ", num_threads=", num_threads,
              ").  Approximately ", idle_pct,
              "% of threads will idle on the Phase-2 join barrier "
              "because the M-tile slice plan cannot subdivide M=1 "
              "across threads.  If this routing was unintended, set "
              "ZENDNNL_GRP_MATMUL_AUTO_DECODE_ALGO=3 to keep decode "
              "calls on the wide-N parallel path, or raise the "
              "per-expert M (larger batch / sequence) so the "
              "M-tile planner has rows to subdivide.  Emitted once "
              "per process.");
        }
      }
    }
  }

  // ── CCD topology (universal: handles any num_threads) ──
  // F7 — Zen 3 / 4 / 5 classic-CCD assumption: 8 cores per CCD with
  // shared L3 per CCD.  c-class parts (Zen 4c "Bergamo", Zen 5c
  // "Turin-Dense") deviate from this — Zen 5c uses 16-core CCDs
  // with one L3 per CCD; the planner's striping math still
  // schedules correctly there but treats each large CCD as two
  // 8-core groups (i.e. CCD locality is per 8-core stripe rather
  // than per L3 slice).  Make this a runtime detect when ZenDNN
  // builds against a c-class part; until then the constant matches
  // every shipped MI300 / Genoa / Turin head node.
  const int cores_per_ccd = std::min(8, num_threads);
  const int num_ccds = std::max(1,
      (num_threads + cores_per_ccd - 1) / cores_per_ccd);
  auto ccd_capacity = [&](int c) -> int {
    const int base = c * cores_per_ccd;
    return std::max(0, std::min(cores_per_ccd, num_threads - base));
  };

  // ── Many active experts > num_threads: dynamic-scheduled work pool ──
  //
  // F4 — replaces the prior round-based `(round + tid)` static layout
  // (which idled (num_threads − tail_round) threads on the final
  // round whenever `active_ops % num_threads != 0`) with an
  // `omp for schedule(dynamic, 1)` work-stealing pool.  Each thread
  // pulls the next active expert from a shared work-queue (an
  // atomic fetch-add inside the OMP runtime, ~50–100 ns per task)
  // when it finishes its current one; the per-expert GEMM is
  // 10–100 µs at minimum, so the dispatch overhead is in the noise.
  //
  // Behaviour change:
  //   * No `#pragma omp barrier` per round → threads that pick a
  //     fast expert immediately grab the next instead of waiting.
  //   * The implicit barrier at end of the `parallel` region still
  //     joins all threads before the function returns.
  //   * Thread → expert mapping is no longer deterministic; CCD
  //     spreading is still preserved structurally because tids
  //     0..cores_per_ccd-1 sit on CCD 0 etc. via KMP_AFFINITY
  //     compact, and dynamic dispatch fills idle slots uniformly.
  //   * `static thread_local matmul_params local_params` is reused
  //     across iterations on the same thread (declared per-thread
  //     scope, re-assigned at the start of every task) — safe.
  if (active_ops > num_threads) {
    if (test_api::s_capture_m_tile_path.load(std::memory_order_relaxed)) {
      test_api::s_last_m_tile_path.store(
          test_api::m_tile_path_tag::kRoundBased,
          std::memory_order_relaxed);
    }
    std::vector<int> active_idx;
    active_idx.reserve(active_ops);
    for (int i = 0; i < num_ops; ++i)
      if (M[i] > 0) active_idx.push_back(i);

    #pragma omp parallel num_threads(num_threads)
    {
      #pragma omp for schedule(dynamic, 1) nowait
      for (int j = 0; j < active_ops; ++j) {
        const int e = active_idx[j];
        static thread_local matmul_params local_params;
        local_params = params[e];
        execute_expert_slice(layout[e], transA[e], transB[e],
            M[e], N[e], K[e], alpha[e],
            src[e], lda[e], weight[e], ldb[e],
            bias[e], beta[e], dst[e], ldc[e],
            is_weights_const[e], 1, local_params, algo);
        if (fused_act != grp_matmul_gated_act_t::none) {
          apply_gated_act_inplace(fused_act, dst[e], 0, M[e],
                                  N[e], ldc[e], act_dtype);
        }
      }
    }
    return;
  }

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
  //     is bounded by E.  Mixtral (E=8) and GPT-OSS (E=32) fall
  //     in this category on 128-thread systems.
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
    const int avg_M = (active_ops > 0)
        ? static_cast<int>(sum_M_total / active_ops) : 0;

    // F8 — Same env-tunable hatch as `kSliceTarget`.  Defaults match
    // the original literal constants exactly (256 / 4 / 8); each
    // getter caches its env on first call so the four lookups cost
    // four relaxed loads + branches in steady state.
    const int kHybridMinMaxM   = get_grp_matmul_m_tile_hybrid_min_max_m();
    const int kHybridMinSkewX  = get_grp_matmul_m_tile_hybrid_min_skew();
    const int kLightsPerThread =
        get_grp_matmul_m_tile_hybrid_lights_per_thread();
    const int min_actives = num_threads / 2;
    const int min_lights  = num_threads / 8;

    // Overflow-safe skew test: with the F8 env knob
    // `ZENDNNL_GRP_MATMUL_M_TILE_HYBRID_MIN_SKEW` allowing any
    // positive int and `avg_M` bounded only by `int`, the product
    // `kHybridMinSkewX * avg_M` can overflow signed `int` (UB) on
    // pathological tuning sweeps.  Promote one operand to `int64_t`
    // so the comparison stays well-defined for all valid int
    // inputs.  At stock defaults (skew=4, avg_M ≤ a few thousand)
    // the cast compiles to a no-op on x86_64.
    const bool gate_skew = (max_M >= kHybridMinMaxM)
                        && (avg_M > 0)
                        && (static_cast<int64_t>(max_M)
                            >= static_cast<int64_t>(kHybridMinSkewX)
                               * avg_M);

    if (active_ops >= min_actives && gate_skew) {
      const int light_cut = std::max(8, avg_M / 4);

      std::vector<int> light_exp;
      std::vector<int> heavy_exp;
      int64_t heavy_M_sum = 0;
      light_exp.reserve(num_ops);
      heavy_exp.reserve(num_ops);
      for (int i = 0; i < num_ops; ++i) {
        if (M[i] <= 0) continue;
        if (M[i] <= light_cut) {
          light_exp.push_back(i);
        } else {
          heavy_exp.push_back(i);
          heavy_M_sum += M[i];
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
      // Overflow-safe light-pool ceil-div.  Same defense-in-depth
      // rationale as the skew gate above: with the F8 env knob
      // `ZENDNNL_GRP_MATMUL_M_TILE_HYBRID_LIGHTS_PER_THREAD`
      // accepting any positive int, the intermediate
      // `n_light + kLightsPerThread - 1` can overflow signed `int`.
      // The final ceil-div result is bounded by `n_light` (≤
      // kNTilePlanMaxExperts = 256), so the int cast after the
      // ≥ 1 clamp is safe regardless of the env value.
      const int candidate_light_pool = std::min(cores_per_ccd,
          static_cast<int>(std::max<int64_t>(1,
              (static_cast<int64_t>(n_light) + kLightsPerThread - 1)
                  / kLightsPerThread)));
      const int candidate_heavy_pool = num_threads - candidate_light_pool;
      if (n_light > 0 && n_light >= min_lights
          && n_heavy > 0 && heavy_M_sum > 0
          && candidate_light_pool < num_threads
          && candidate_heavy_pool >= n_heavy) {
        // Capture-gated branch tag — commit point for multi-tier
        // hybrid.  Tagged here (not at the outer `if
        // (get_grp_matmul_m_tile_hybrid() == 0)`) so the tag fires
        // ONLY when all four shape gates pass AND there are enough
        // light/heavy experts to make the split profitable (AND the
        // safety guard above keeps heavy_pool ≥ 1); outer-gate hits
        // that fall through to single-tier do not poison the tag.
        // See doc-block on `test_api::s_capture_m_tile_path` in
        // `group_matmul_parallel_common.hpp`.
        if (test_api::s_capture_m_tile_path.load(std::memory_order_relaxed)) {
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
          // signed `int` for pathological tuning sweeps.  The
          // ceil-div result is bounded by `M[idx]` so the int cast
          // after the ≥ 1 clamp is always safe.  See the sister
          // hardening on the single-tier `t_assign[i]` init below.
          ht_assign[idx] = std::min(M[idx],
              static_cast<int>(std::max<int64_t>(1,
                  (static_cast<int64_t>(M[idx]) + kSliceTarget - 1)
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
              if (ht_assign[idx] <= 0 || ht_assign[idx] >= M[idx]) continue;
              int cur_slice = (M[idx] + ht_assign[idx] - 1) / ht_assign[idx];
              if (cur_slice > best_slice) { best_slice = cur_slice; best = idx; }
            }
            if (best < 0) break;
            ++ht_assign[best];
            --surplus;
          }
        } else {
          // Scale-down proportional to M, floor=1 per heavy active.
          int assigned = 0;
          for (int idx : heavy_exp) {
            ht_assign[idx] = std::max(1, static_cast<int>(
                static_cast<int64_t>(heavy_pool) * M[idx] / heavy_M_sum));
            assigned += ht_assign[idx];
          }
          while (assigned < heavy_pool) {
            int best = -1;
            int best_slice = 0;
            for (int idx : heavy_exp) {
              if (ht_assign[idx] <= 0 || ht_assign[idx] >= M[idx]) continue;
              int cur_slice = (M[idx] + ht_assign[idx] - 1) / ht_assign[idx];
              if (cur_slice > best_slice) { best_slice = cur_slice; best = idx; }
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
              int cur_slice = (M[idx] + ht_assign[idx] - 1) / ht_assign[idx];
              if (cur_slice < least_slice) { least_slice = cur_slice; best = idx; }
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
        // hosts (c-class parts such as Zen 4c "Bergamo" or Zen 5c
        // "Turin-Dense") where a single heavy with `M = 256` cannot
        // absorb a `heavy_pool > 256` budget.
        int ht_assigned_final = 0;
        for (int idx : heavy_exp) ht_assigned_final += ht_assign[idx];
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
          // F6 (open / unmeasured): the linear walk packs heavy teams
          // across CCDs by tid order.  When `ht_assign[idx]` does not
          // divide cores_per_ccd cleanly (e.g. 9 threads/heavy with
          // cores_per_ccd=8), adjacent heavy teams share the boundary
          // CCD.  Heavy weights ≤ 16 MB fit two-per-CCD-L3 (Qwen3
          // K=2048, M_heavy ≤ 4096); larger heavies (K=4096,
          // M_heavy > 4096) could thrash the boundary CCD.  No
          // production workload measured today exhibits the failure
          // mode — recheck if future MoE deployments push past
          // `M_heavy = 4096` on K = 4096 backbones, in which case
          // switch to a CCD-aware heavy mapping (each heavy gets
          // ceil(ht_assign[idx] / cores_per_ccd) contiguous CCDs,
          // padding the last with zero threads).
          constexpr int kRoleInactive = 0;
          constexpr int kRoleLight    = 1;
          constexpr int kRoleHeavy    = 2;
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
            for (int k = 0; k < t && next_tid < num_threads; ++k, ++next_tid) {
              mt_tid_to_role[next_tid]   = kRoleHeavy;
              mt_tid_to_expert[next_tid] = idx;
              mt_tid_to_local[next_tid]  = k;
              mt_tid_to_team[next_tid]   = t;
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
          std::atomic<int> light_next{0};

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
                execute_light_expert(light_exp[j], layout, transA, transB,
                    M, N, K, alpha, src, lda, weight, ldb, bias, beta,
                    dst, ldc, is_weights_const, params, algo,
                    fused_act, act_dtype);
              }
            } else if (role == kRoleHeavy) {
              const int e = mt_tid_to_expert[tid];
              execute_m_tile_act(e, mt_tid_to_local[tid], mt_tid_to_team[tid],
                  layout, transA, transB, M, N, K, alpha,
                  src, lda, weight, ldb, bias, beta, dst, ldc,
                  is_weights_const, params, src_elem, dst_elem, algo,
                  fused_act, act_dtype);
            }
          }
          return;
        }  // end F3 ht_assigned_final guard — if heavies underfill
           // heavy_pool we fall through to single-tier instead.
      }
    }
  }
  // (multi-tier gating did not engage — fall through to single-tier)

  // ── Phase 1b: initial t_assign based on target slice size ──
  std::vector<int> t_assign(num_ops, 0);
  int total_need = 0;
  int max_M_single_tier = 0;
  for (int i = 0; i < num_ops; ++i) {
    if (M[i] <= 0) continue;
    // Overflow-safe ceil-div: see the matching note on the
    // multi-tier `ht_assign[idx]` init above.  Result bounded by
    // `M[i]` so the final int cast after the ≥ 1 clamp is safe
    // for every env-tunable `kSliceTarget` value.
    t_assign[i] = std::min(M[i],
        static_cast<int>(std::max<int64_t>(1,
            (static_cast<int64_t>(M[i]) + kSliceTarget - 1)
                / kSliceTarget)));
    total_need += t_assign[i];
    if (M[i] > max_M_single_tier) max_M_single_tier = M[i];
  }

  // ── Wide-N memory-bound fallback (few actives × small M × large N) ──
  //
  // When `total_need * 2 ≤ num_threads`, the M dimension across all
  // active experts is too shallow to absorb the available parallelism
  // at the kSliceTarget=16 floor.  Phase 2's surplus distribution
  // below would then pile leftover threads onto small-M experts and
  // shrink each thread's slice well below kSliceTarget; brgemm hits
  // its narrow-M path and per-thread efficiency collapses.
  //
  // In this regime (e.g., Mixtral prompt light frames with 7-8
  // actives × M ≤ 60 × N=28672, or GPT-OSS prompt tiny frames with
  // 12-18 actives × M ≤ 64) sequential-with-full-team is strictly
  // better: each expert's GEMM is parallelized across all
  // `num_threads` by the inner matmul algo (M-and-N-aware), the
  // weight matrix is loaded once per expert by the whole team via
  // shared L3, and the inactive M-tile slots that would otherwise
  // sit on the OMP barrier are avoided altogether.
  //
  // Decode safety: `max_M_single_tier > 1` excludes pure-decode
  // workloads (M=1 per expert).  In that regime Phase 2's surplus
  // step is a structural no-op — `t_assign[i] < M[i]` is false for
  // every expert, so the surplus loop breaks immediately and the
  // existing M-tile CCD-stripe runs each expert on its own CCD with
  // 1 thread (Mixtral decode: 8 actives → 8 parallel single-thread
  // calls on 8 distinct CCDs, the latency-optimal mapping).  Sending
  // that to the sequential-with-full-team path would serialize 8
  // GEMMs back-to-back and trade away the CCD-parallel decode win.
  //
  // Mutually exclusive with the round-based (`active_ops >
  // num_threads`) branch above (different active-count regime) and
  // with the multi-tier path (which requires `active_ops ≥
  // num_threads/2 ≥ 64` ⇒ total_need ≫ num_threads/2 in practice).
  //
  // The 2× margin is conservative on purpose so we do not chase the
  // sequential path for healthy mid-range workloads.  Empirical
  // regimes at 128 threads:
  //   Mixtral prompt light  7 × ceil(36/16) = 21   →   42 ≤ 128  → fallback
  //   GPT-OSS prompt tiny  12 × ceil(42/16) = 36   →   72 ≤ 128  → fallback
  //   Mixtral prompt heavy  8 × ceil(992/16)= 496  →  992 > 128  → M-tile
  //   GPT-OSS prompt heavy 32 × ceil(496/16)= 992  → 1984 > 128  → M-tile
  //   Qwen3 prompt light  100 × 1          ≈ 100  →  200 > 128  → M-tile
  //   Mixtral decode        max_M=1                          → M-tile (excluded)
  //   GPT-OSS decode        max_M=1                          → M-tile (excluded)
  //
  // `total_need * 2` is promoted to int64_t to defuse any
  // hypothetical signed-int overflow.  Practical bounds keep
  // total_need ≪ INT_MAX (num_ops ≤ 256, t_assign[i] ≤
  // ceil(M[i]/16) ≤ ~62K even at M=1M), so the cast is purely
  // defensive (silences static analysers and keeps the guard
  // robust if kSliceTarget or kNTilePlanMaxExperts ever grow).
  if (max_M_single_tier > 1
      && static_cast<int64_t>(total_need) * 2 <= num_threads) {
    // Capture-gated branch tag — see doc-block on
    // `test_api::s_capture_m_tile_path` in
    // `group_matmul_parallel_common.hpp`.
    if (test_api::s_capture_m_tile_path.load(std::memory_order_relaxed)) {
      test_api::s_last_m_tile_path.store(
          test_api::m_tile_path_tag::kWideNFallback,
          std::memory_order_relaxed);
    }
    for (int e = 0; e < num_ops; ++e) {
      if (M[e] <= 0) continue;
      static thread_local matmul_params local_params;
      local_params = params[e];
      execute_expert_slice(layout[e], transA[e], transB[e],
          M[e], N[e], K[e], alpha[e],
          src[e], lda[e], weight[e], ldb[e],
          bias[e], beta[e], dst[e], ldc[e],
          is_weights_const[e], num_threads, local_params, algo);
      if (fused_act != grp_matmul_gated_act_t::none) {
        apply_gated_act_inplace(fused_act, dst[e], 0, M[e],
                                N[e], ldc[e], act_dtype);
      }
    }
    return;
  }

  // Capture-gated branch tag — Phase 2 single-tier is the default
  // fallthrough when none of the earlier branches commit.  See
  // doc-block on `test_api::s_capture_m_tile_path` in
  // `group_matmul_parallel_common.hpp`.
  if (test_api::s_capture_m_tile_path.load(std::memory_order_relaxed)) {
    test_api::s_last_m_tile_path.store(
        test_api::m_tile_path_tag::kPhase2Single,
        std::memory_order_relaxed);
  }

  // ── Phase 2: fit t_assign to num_threads ──
  //
  // CCD-aware cap: when active_ops <= num_ccds AND every active expert
  // could fit in a single CCD (active_ops * cores_per_ccd >=
  // num_threads), cap each expert at its assigned CCD's capacity.
  // This prevents an expert's team from spanning CCD boundaries, which
  // would cause L3 contention with a neighbor expert's weight.
  //
  // F2 — Predicate uses `active_ops` (not raw `num_ops`).  The prior
  // form `(num_ops <= num_ccds)` would refuse to engage cap_at_ccd on
  // the typical MoE shape (e.g. Mixtral E=8 with one unrouted expert
  // on 64t: num_ops=8 but active_ops=7), and even when it did engage,
  // the `i % num_ccds` stripe placed inactive experts on CCDs they
  // never used — leaving exactly one CCD idle for every inactive
  // expert.  Using `active_ops` here and `active_pos[i]` in the
  // modulus below makes the CCD layout compact over the actually-
  // executing experts.
  //
  // Critical for num_threads not a clean multiple of cores_per_ccd.
  // Example 16 experts × M=8 at 126t (cores_per_ccd=8, num_ccds=16):
  //   Without cap: t_assign = [8]*14 + [7]*2 — last 2 experts cram into
  //     CCDs 14 and 15, expert 15 spills 1 thread onto CCD 14 (L3 contention).
  //   With cap:    t_assign = [8]*15 + [6]   — expert 15 fits on partial
  //     CCD 15 (6 slots); other 15 experts each own their CCD.
  const bool cap_at_ccd = (active_ops <= num_ccds)
      && (static_cast<int64_t>(active_ops) * cores_per_ccd >= num_threads);

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
          // F2 — Compact CCD stripe over active experts only.
          // `active_pos[i] >= 0` is guaranteed here because
          // `t_assign[i] > 0` ⇒ M[i] > 0 ⇒ `active_pos[i] != -1`.
          const int my_ccd = active_pos[i] % num_ccds;
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
          // F2 — see Phase 2 surplus-branch comment above.
          const int my_ccd = active_pos[i] % num_ccds;
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

  // F2 — Starting CCD is deterministic per *active position*
  // (`active_pos[e] % num_ccds`), NOT raw expert index.  This matches
  // the cap_at_ccd assumption above (`my_ccd = active_pos[i] % num_ccds`)
  // exactly, and crucially compacts the stripe across only the
  // experts that actually execute.  The legacy `e % num_ccds` form
  // wasted one CCD per inactive expert whenever `active_ops <= num_ccds`
  // (canonical case: Mixtral E=8 MoE on 64t with 7 active routes →
  // CCD 0 starved = 8/64 = 12.5 % of the budget on the OMP join
  // barrier).  If an earlier expert has already filled this CCD, the
  // while-loop below skips forward to the next CCD with remaining
  // capacity.
  for (int e = 0; e < num_ops; ++e) {
    if (t_assign[e] <= 0) continue;
    const int t = t_assign[e];
    int placed = 0;
    int c = active_pos[e] % num_ccds;

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
