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

inline int get_grp_matmul_algo() {
  static const int ver = []() {
    const char *env = std::getenv("ZENDNNL_GRP_MATMUL_ALGO");
    return (env && env[0] >= '1' && env[0] <= '4') ? (env[0] - '0') : 0;
  }();
  return ver;
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

// ── ALGO=2: per_expert ─────────────────────────────────────────────────
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

// ── ALGO=3: multilevel — CCD-aware adaptive scheduling ────────────────
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
    int total_M = 0;
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

// ── ALGO=4: flat_ccd_m_slice — single-level OMP, CCD-aware M-slicing ──
//
// Single flat #pragma omp parallel (no nesting) — framework-safe.
// Guarantees: ZERO nested OMP (num_thr=1 per slice), CCD-aligned teams.
//
//   (A) num_ops <= num_ccds (few experts): proportional CCD allocation.
//       All threads participate in a single flat OMP region.
//       Expert with larger M gets more threads (M-slicing, num_thr=1).
//
//   (B) num_ops > num_ccds (many experts): single OMP region with
//       barrier-synchronized rounds.  Each round: num_ccds experts
//       concurrently, 1 CCD per expert, flat M-slicing.
//
// M-slicing pointer arithmetic (regimes A and B):
//   transA=false → src is M×lda, row offset = row_start * lda * elem
//   transA=true  → src is K×lda, col offset = row_start * elem
//   dst is always M×ldc, row offset = row_start * ldc * elem

// Per-thread params copy avoids data race: concurrent M-slices of the
// same expert must not share mutable matmul_params (packing/scratch).
// thread_local amortises allocation — only copies the struct fields,
// reusing the existing vector/string capacity across calls.
inline void execute_m_slice(
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

void flat_ccd_m_slice(
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
  const size_t dst_elem = size_of(params[0].dtypes.dst);

  scoped_active_levels guard(1);

  if (num_ops <= num_ccds) {
    // (A) Few experts: proportional CCD allocation, single flat OMP region.
    // Uses num_ccds * ccd_size threads (== num_threads when evenly divisible;
    // up to ccd_size-1 threads may be idle when num_threads % ccd_size != 0).
    int total_M = 0;
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

    // Cumulative thread offsets for expert→thread mapping.
    std::vector<int> thr_start(num_ops + 1, 0);
    for (int i = 0; i < num_ops; ++i)
      thr_start[i + 1] = thr_start[i] + ccds_per_op[i] * ccd_size;
    const int total_threads = thr_start[num_ops];

    #pragma omp parallel num_threads(total_threads)
    {
      const int tid = omp_get_thread_num();

      // Linear search for expert (num_ops is small, <= num_ccds).
      int e = 0;
      while (e < num_ops - 1 && tid >= thr_start[e + 1]) ++e;

      const int team_size = thr_start[e + 1] - thr_start[e];
      const int local_tid = tid - thr_start[e];

      execute_m_slice(e, local_tid, team_size,
          layout, transA, transB, M, N, K, alpha,
          src, lda, weight, ldb, bias, beta, dst, ldc,
          is_weights_const, params, src_elem, dst_elem, algo);
    }
  } else {
    // (B) Many experts: single OMP region, barrier-synchronized rounds.
    // One fork/join for the entire dispatch instead of one per round.
    // Each round: num_ccds experts run concurrently, 1 CCD per expert.
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
          execute_m_slice(round_start + local_expert, local_tid, ccd_size,
              layout, transA, transB, M, N, K, alpha,
              src, lda, weight, ldb, bias, beta, dst, ldc,
              is_weights_const, params, src_elem, dst_elem, algo);
        }
        #pragma omp barrier
      }
    }
  }
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

  const int env_algo = get_grp_matmul_algo();

  // Auto-select (ALGO=0 or unset): pick the best strategy based on shape.
  //   - experts >= threads → V2 (per-expert parallel-for, 1 thread each)
  //   - experts < threads  → V3 (multilevel, multi-CCD per expert)
  // Manual override: 1=sequential, 2=per_expert, 3=multilevel, 4=flat_ccd_m_slice.
  int use_algo;
  if (env_algo >= 1 && env_algo <= 4) {
    use_algo = env_algo;
  } else {
    use_algo = (static_cast<int>(M.size()) >= num_threads) ? 2 : 3;
  }

  switch (use_algo) {
  case 1:
    if (gemm_mode_out != nullptr)
      *gemm_mode_out = "sequential_experts";
    sequential_experts(layout, transA, transB, M, N, K, alpha,
        src, lda, weight, ldb, bias, beta, dst, ldc,
        is_weights_const, params, num_threads);
    break;
  case 3:
    if (gemm_mode_out != nullptr)
      *gemm_mode_out = "multilevel";
    parallel_multilevel(layout, transA, transB, M, N, K, alpha,
        src, lda, weight, ldb, bias, beta, dst, ldc,
        is_weights_const, params, num_threads);
    break;
  case 4:
    if (gemm_mode_out != nullptr)
      *gemm_mode_out = "flat_ccd_m_slice";
    flat_ccd_m_slice(layout, transA, transB, M, N, K, alpha,
        src, lda, weight, ldb, bias, beta, dst, ldc,
        is_weights_const, params, num_threads);
    break;
  default:
    if (gemm_mode_out != nullptr)
      *gemm_mode_out = "per_expert";
    parallel_per_expert(layout, transA, transB, M, N, K, alpha,
        src, lda, weight, ldb, bias, beta, dst, ldc,
        is_weights_const, params, num_threads);
    break;
  }
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
