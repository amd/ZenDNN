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
// Best for few experts with very large M (e.g. prompt with 2-4 experts),
// where maximizing threads within each GEMM matters more than concurrency.

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
    int num_threads) {

  const size_t num_ops = M.size();
  matmul_algo_t algo = resolve_kernel();

  for (size_t i = 0; i < num_ops; ++i) {
    execute_expert_slice(layout[i], transA[i], transB[i],
        M[i], N[i], K[i], alpha[i],
        src[i], lda[i], weight[i], ldb[i],
        bias[i], beta[i], dst[i], ldc[i],
        is_weights_const[i], num_threads, params[i], algo);
  }
}

// ── ALGO=2: flat CCD tiling — single-level OMP, CCD-aware ──────────────
//
// Single flat #pragma omp parallel (no nesting) — framework-safe.
// Guarantees: ZERO nested OMP (num_thr=1 per tile), CCD-aligned teams.
//
// Per-expert adaptive tiling (decided inside the OMP region):
//   M[e] >= team_size → M-tiling: each thread gets unique M-tile, full N.
//   M[e] <  team_size → N-tiling: each thread gets full M, unique N-slice.
//
//   (A) num_ops <= num_ccds: proportional CCD allocation, single region.
//   (B) num_ops > num_ccds: barrier-synchronized rounds.
//
// N-tiling pointer arithmetic:
//   transB=false → B is K×ldb, col offset = col_start * wei_elem
//   transB=true  → B is N×ldb, row offset = col_start * ldb * wei_elem
//   dst col offset = col_start * dst_elem (ldc unchanged)
//   bias col offset = col_start * bias_elem (if non-null)
//   src is unchanged (full A for every thread)
//
// M-tiling pointer arithmetic:
//   transA=false → src row offset = row_start * lda * src_elem
//   transA=true  → src col offset = row_start * src_elem
//   dst row offset = row_start * ldc * dst_elem

// Tiling pointer arithmetic assumes row-major layout (layout == 'r' or 'R').
// Column-major ('c') would require different offset formulas.
// All MoE/group-matmul callers use row-major; column-major is not
// supported by the tiled algorithms (ALGO 2/3).
//
// Element sizes are derived from params[0].dtypes.  Dtype uniformity
// across all experts is enforced by the dispatch function (tiling_safe
// check) before calling any tiled algorithm.  Mixed-dtype experts fall
// back to ALGO 1 (sequential).
//
// Per-thread params copy avoids data race: concurrent tiles of the
// same expert must not share mutable matmul_params (packing/scratch).
// thread_local amortises allocation — only copies the struct fields,
// reusing the existing vector/string capacity across calls.

// Minimum N-tile width.  Below this, kernel startup overhead and poor
// register utilization dominate.  Threads beyond N/min_n_tile are idle.
static constexpr int kMinNTile = 512;

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

  // Cap active threads so each N-tile >= min_n_tile.
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

  // Weight caching: each N-slice gets its own cache entry keyed by
  // (w, K, n_tile, ldb, transB).  Total entries = experts × tiles_per_expert.
  // For MoE (8 experts × 16 tiles = 128 entries) this is manageable.
  // Caching is kept enabled so repeated inference calls hit the cache.
  static thread_local matmul_params tile_params;
  tile_params = params[e];
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

  const size_t src_off = transA[e]
      ? static_cast<size_t>(row_start) * src_elem
      : static_cast<size_t>(row_start) * lda[e] * src_elem;
  const auto *s = static_cast<const char *>(src[e]) + src_off;
  auto *d = static_cast<char *>(dst[e])
      + static_cast<size_t>(row_start) * ldc[e] * dst_elem;
  static thread_local matmul_params slice_params;
  slice_params = params[e];
  execute_expert_slice(layout[e], transA[e], transB[e],
      slice_M, N[e], K[e], alpha[e],
      s, lda[e], weight[e], ldb[e],
      bias[e], beta[e], d, ldc[e],
      is_weights_const[e], 1, slice_params, algo);
}

void flat_ccd_m_tile(
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
  matmul_algo_t algo = resolve_kernel();
  const int ccd_size = std::min(8, num_threads);
  const int num_ccds = std::max(1, num_threads / ccd_size);

  const size_t src_elem = size_of(params[0].dtypes.src);
  const size_t wei_elem = size_of(params[0].dtypes.wei);
  const size_t dst_elem = size_of(params[0].dtypes.dst);
  const size_t bias_elem = (params[0].dtypes.bias != data_type_t::none)
      ? size_of(params[0].dtypes.bias) : sizeof(float);

  // Per-expert tiling: N-tile when M[e] < team_size, M-tile otherwise.
  // N-tiling reduces per-thread B reads from K×N to K×(N/team).

  scoped_active_levels guard(1);

  if (num_ops <= num_ccds) {
    // (A) Few experts: proportional CCD allocation, single flat OMP region.
    int64_t total_M = 0;
    for (int i = 0; i < num_ops; ++i) total_M += M[i];
    if (total_M <= 0) return;

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

    std::vector<int> thr_start(num_ops + 1, 0);
    for (int i = 0; i < num_ops; ++i)
      thr_start[i + 1] = thr_start[i] + ccds_per_op[i] * ccd_size;
    const int total_threads = thr_start[num_ops];
    #pragma omp parallel num_threads(total_threads)
    {
      const int tid = omp_get_thread_num();
      int e = 0;
      while (e < num_ops - 1 && tid >= thr_start[e + 1]) ++e;

      const int team_size = thr_start[e + 1] - thr_start[e];
      const int local_tid = tid - thr_start[e];

      // Per-expert decision: N-tile when this expert's M < its team size.
      if (M[e] < team_size) {
        execute_n_tile(e, local_tid, team_size, kMinNTile,
            layout, transA, transB, M, N, K, alpha,
            src, lda, weight, ldb, bias, beta, dst, ldc,
            is_weights_const, params, wei_elem, dst_elem, bias_elem, algo);
      } else {
        execute_m_tile(e, local_tid, team_size,
            layout, transA, transB, M, N, K, alpha,
            src, lda, weight, ldb, bias, beta, dst, ldc,
            is_weights_const, params, src_elem, dst_elem, algo);
      }
    }
  } else {
    // (B) Many experts: single OMP region, barrier-synchronized rounds.
    const int batch = num_ccds;

    #pragma omp parallel num_threads(num_threads)
    {
      const int tid = omp_get_thread_num();

      for (int round_start = 0; round_start < num_ops;
           round_start += batch) {
        const int round_end = std::min(num_ops, round_start + batch);
        const int round_size = round_end - round_start;
        const int round_threads = round_size * ccd_size;

        if (tid < round_threads) {
          const int local_expert = tid / ccd_size;
          const int local_tid = tid % ccd_size;
          const int e = round_start + local_expert;
          if (M[e] < ccd_size) {
            execute_n_tile(e, local_tid, ccd_size, kMinNTile,
                layout, transA, transB, M, N, K, alpha,
                src, lda, weight, ldb, bias, beta, dst, ldc,
                is_weights_const, params, wei_elem, dst_elem, bias_elem,
                algo);
          } else {
            execute_m_tile(e, local_tid, ccd_size,
                layout, transA, transB, M, N, K, alpha,
                src, lda, weight, ldb, bias, beta, dst, ldc,
                is_weights_const, params, src_elem, dst_elem, algo);
          }
        }
        #pragma omp barrier
      }
    }
  }
}

// ── ALGO=3: flat_ccd_n_tile — single-level OMP, N-tiling ────────────────
//
// Each thread gets full M but a unique N-slice of weight B.
// Per-thread B read = K × (N/team) instead of K × N.
//
// Three regimes:
//   (S) Small M (decode): process experts sequentially, ALL threads
//       N-tile each expert.  Avoids weight-cache fragmentation across
//       experts.  Lower min_n_tile (64) to engage all threads.
//   (A) Few experts, large M: proportional threads per expert, concurrent.
//       N-tile-aware cap (kMinNTile=512) to keep tiles efficient.
//   (B) Many experts: barrier-synchronized rounds, adaptive n_thr.

void flat_ccd_n_tile(
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

  if (max_M <= ccd_size && num_ops <= num_ccds) {
    // (S) Decode: experts sequentially, ALL threads N-tile each expert.
    // All threads focus on one expert at a time → one set of weight
    // cache entries per expert, no fragmentation across concurrent experts.
    // Lower min_n_tile (64) so all threads participate even for large N.
    static constexpr int kDecodeMinNTile = 64;

    #pragma omp parallel num_threads(num_threads)
    {
      const int tid = omp_get_thread_num();
      for (int e = 0; e < num_ops; ++e) {
        execute_n_tile(e, tid, num_threads, kDecodeMinNTile,
            layout, transA, transB, M, N, K, alpha,
            src, lda, weight, ldb, bias, beta, dst, ldc,
            is_weights_const, params, wei_elem, dst_elem, bias_elem, algo);
        #pragma omp barrier
      }
    }
  } else if (num_ops <= num_ccds) {
    // (A) Few experts, large M: proportional concurrent N-tiling.
    const int max_n_thr = std::max(1, max_N / kMinNTile);
    const int thr_per_expert = std::min(
        num_ccds / num_ops * ccd_size, max_n_thr);
    const int total_threads = num_ops * thr_per_expert;

    #pragma omp parallel num_threads(total_threads)
    {
      const int tid = omp_get_thread_num();
      const int e = tid / thr_per_expert;
      const int local_tid = tid % thr_per_expert;

      if (e < num_ops) {
        execute_n_tile(e, local_tid, thr_per_expert, kMinNTile,
            layout, transA, transB, M, N, K, alpha,
            src, lda, weight, ldb, bias, beta, dst, ldc,
            is_weights_const, params, wei_elem, dst_elem, bias_elem, algo);
      }
    }
  } else {
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
}

// ── ALGO=4: multilevel — CCD-aware adaptive scheduling ────────────────
//
// Single strategy that adapts to any combination of expert count, M size,
// and thread count.  Three regimes:
//
// (A) Few experts, large M (num_ops <= num_ccds AND max_M >= ccd_size):
//     All experts run concurrently.  Each expert gets multiple CCDs
//     (proportional to M[e]), full M, and a multi-threaded GEMM kernel
//     that parallelizes internally across M/K/N.
//     Best for prompt/prefill (4-9× over V2).
//
// (B) Few experts, small M (num_ops <= num_ccds AND max_M < ccd_size):
//     All experts run concurrently but each gets only 1 CCD (8 threads).
//     Avoids over-threading tiny GEMMs where the kernel can't utilize
//     many threads.  Wins for decode with small K×N (e.g. Qwen2).
//
// (C) Many experts (num_ops > num_ccds):
//     Process in rounds of num_ccds experts at a time.  Each CCD
//     handles exactly one expert per round with ccd_size (8) threads.
//     No two experts share a CCD → no L3 weight-matrix thrashing.
//     ~13% faster than V1 at 32 experts; neutral at 64-128.

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
    int num_threads) {

  const int num_ops = static_cast<int>(M.size());
  matmul_algo_t algo = resolve_kernel();

  // CCD size: 8 cores sharing L3 on AMD Zen 3/4/5.  Capped to num_threads
  // so we never oversubscribe when the caller requests fewer than 8 threads.
  const int ccd_size = std::min(8, num_threads);
  const int num_ccds = std::max(1, num_threads / ccd_size);
  const int max_M = *std::max_element(M.begin(), M.end());

  if (num_ops <= num_ccds && max_M >= ccd_size) {
    // (A) Few experts, large M: multi-CCD per expert, all concurrent.
    int64_t total_M = 0;
    for (int i = 0; i < num_ops; ++i) total_M += M[i];
    if (total_M <= 0) total_M = num_ops;

    // Allocate CCDs: start with 1 per expert, distribute remainder by M.
    // Invariant: sum(ccds_per_op) == num_ccds (no oversubscription).
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
      }
    }
  } else {
    // (B) Few experts + small M, or (C) many experts:
    // Round-based, 1 CCD per expert.  Each expert gets ccd_size threads.
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
        }
      }
    }
  }
}

// ── ALGO=5: per_expert ─────────────────────────────────────────────────
// Parallel-for over experts; each iteration runs one expert's GEMM with
// 1 thread.  OMP implicit barrier at loop completion.
// Best when num_ops >= num_threads (every thread busy, minimal overhead).
// Falls behind when num_ops << num_threads (most threads idle).

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
    int num_threads) {

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
  }
}

// ── ALGO selection ──────────────────────────────────────────────────────
//
// Separated from dispatch for maintainability.  Returns the ALGO number
// (1-5) to use.  Handles manual override, tiling_safe validation, and
// auto-select heuristics.

int select_grp_matmul_algo(
    const std::vector<char> &layout,
    const std::vector<int> &M,
    const std::vector<int> &N,
    const std::vector<matmul_params> &params,
    int num_threads) {

  const int env_algo = get_grp_matmul_algo();
  const int num_ops = static_cast<int>(M.size());

  // Tiled algos (2/3) require:
  //   (a) Row-major layout (pointer arithmetic assumes row-major).
  //   (b) Uniform dtypes across all experts (elem sizes from params[0]).
  //   (c) No per-channel/per-group quantization.
  //   (d) No post-ops (tiled path rejects any non-empty postop_).
  //   (e) Standard unpacked A and B
  //       (mem_format_a=='n', mem_format_b=='n', pack_format_b==0).
  bool tiling_safe = true;
  for (int i = 0; i < num_ops && tiling_safe; ++i) {
    if (layout[i] != 'r' && layout[i] != 'R') tiling_safe = false;
    if (params[i].dtypes.src != params[0].dtypes.src) tiling_safe = false;
    if (params[i].dtypes.wei != params[0].dtypes.wei) tiling_safe = false;
    if (params[i].dtypes.dst != params[0].dtypes.dst) tiling_safe = false;
    if (params[i].dtypes.bias != params[0].dtypes.bias) tiling_safe = false;
    if (params[i].dynamic_quant) tiling_safe = false;
    if (!params[i].postop_.empty()) tiling_safe = false;
    if (params[i].quant_params.wei_scale.buff != nullptr) tiling_safe = false;
    if (params[i].quant_params.wei_zp.buff != nullptr) tiling_safe = false;
    if (params[i].quant_params.src_scale.buff != nullptr) tiling_safe = false;
    if (params[i].quant_params.src_zp.buff != nullptr) tiling_safe = false;
    if (params[i].mem_format_a != 'n') tiling_safe = false;
    if (params[i].mem_format_b != 'n') tiling_safe = false;
    if (params[i].packing.pack_format_b != 0) tiling_safe = false;
  }

  // Manual override: ZENDNNL_GRP_MATMUL_ALGO=1..5.
  if (env_algo >= 1 && env_algo <= 5) {
    int algo = env_algo;
    if ((algo == 2 || algo == 3) && !tiling_safe)
      algo = 1;
    return algo;
  }

  // Auto-select (ALGO=0).
  // Decision tree (benchmarked on 70 shapes, Zen4 128-core):
  //
  //   1. Tiling not safe OR num_ops <= 4 → ALGO 1.
  //      Too few experts or unsupported config for tiling.
  //
  //   2. num_ops >= 32, max_M >= 8 → ALGO 3 (N-tile).
  //   3. num_ops >= 32, max_M < 8, max_N <= 2048 → ALGO 3 (small B in L3).
  //   4. num_ops >= 32, max_M < 8, max_N > 2048 → ALGO 2 (adaptive).
  //   5. num_ops >= 8, max_M >= 8 → ALGO 3 (N-tile, 1.1-3.9× speedup).
  //   6. num_ops >= 16, max_M < 8 → ALGO 2 (CCD rounds for decode).
  //   7. num_ops >= 8, max_M <= 1 → ALGO 2 (BS ≤ 4 decode).
  //   8. Fallback → ALGO 1 (sequential).
  if (!tiling_safe || num_ops <= 4 || num_threads <= 1)
    return 1;

  const int max_M = *std::max_element(M.begin(), M.end());

  const int max_N = *std::max_element(N.begin(), N.end());

  // Many experts (32+) with meaningful M: N-tile wins.
  if (num_ops >= 32 && max_M >= 8)
    return 3;

  // Many experts (32+) with small M (decode):
  //   - Small N (≤ 2048, OLMoE class): N-tile wins (B fits in L3).
  //   - Large N (> 2048): adaptive tile's CCD rounds are better.
  if (num_ops >= 32 && max_M < 8)
    return (max_N <= 2048) ? 3 : 2;

  // 8-16 experts with meaningful M: N-tile wins for both
  // gate_proj (N=14336) and down_proj (N=4096).
  if (num_ops >= 8 && max_M >= 8)
    return 3;

  // 16+ experts, decode (small M): adaptive tile.
  if (num_ops >= 16 && max_M < 8)
    return 2;

  // 8 experts, M <= 1 (BS ≤ 4 decode): adaptive tile.
  if (num_ops >= 8 && max_M <= 1)
    return 2;

  return 1;
}

} // namespace

// ── Dispatch ────────────────────────────────────────────────────────────

void group_matmul_run_parallel_dispatch(
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
    const char **gemm_mode_out) {

  const int use_algo = select_grp_matmul_algo(layout, M, N, params,
                                               num_threads);

  switch (use_algo) {
  case 1:
    if (gemm_mode_out != nullptr)
      *gemm_mode_out = "sequential_experts";
    sequential_experts(layout, transA, transB, M, N, K, alpha,
        src, lda, weight, ldb, bias, beta, dst, ldc,
        is_weights_const, params, num_threads);
    break;
  case 2:
    if (gemm_mode_out != nullptr)
      *gemm_mode_out = "flat_ccd_m_tile";
    flat_ccd_m_tile(layout, transA, transB, M, N, K, alpha,
        src, lda, weight, ldb, bias, beta, dst, ldc,
        is_weights_const, params, num_threads);
    break;
  case 3:
    if (gemm_mode_out != nullptr)
      *gemm_mode_out = "flat_ccd_n_tile";
    flat_ccd_n_tile(layout, transA, transB, M, N, K, alpha,
        src, lda, weight, ldb, bias, beta, dst, ldc,
        is_weights_const, params, num_threads);
    break;
  case 4:
    if (gemm_mode_out != nullptr)
      *gemm_mode_out = "multilevel";
    parallel_multilevel(layout, transA, transB, M, N, K, alpha,
        src, lda, weight, ldb, bias, beta, dst, ldc,
        is_weights_const, params, num_threads);
    break;
  case 5:
    if (gemm_mode_out != nullptr)
      *gemm_mode_out = "per_expert";
    parallel_per_expert(layout, transA, transB, M, N, K, alpha,
        src, lda, weight, ldb, bias, beta, dst, ldc,
        is_weights_const, params, num_threads);
    break;
  default:
    if (gemm_mode_out != nullptr)
      *gemm_mode_out = "sequential_experts";
    sequential_experts(layout, transA, transB, M, N, K, alpha,
        src, lda, weight, ldb, bias, beta, dst, ldc,
        is_weights_const, params, num_threads);
    break;
  }
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
