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

/// Group MatMul direct API implementation.
///
/// Sequential mode (`src.size() == 1`): chained matmuls where output[i-1]
/// feeds as input[i]. Uses full thread team via matmul_execute.
///
/// Parallel mode (`src.size() > 1`): independent matmuls (e.g. MoE experts).
/// Two strategies selectable via env ZENDNNL_GRP_MATMUL_ALGO:
///
///   1 (per-expert): omp parallel for over num_ops. Each expert runs
///       single-threaded matmul_execute. Simple baseline.
///
///   2 (multilevel): All experts run concurrently in a single nested
///       parallel region. Each expert gets M-proportional inner threads
///       so heavy experts (large M) get more compute. Thread budget is
///       num_ops outer + sum(inner[i]) = num_threads exactly.
///
/// Default: 1.
/// Kernel: ZENDNNL_MATMUL_ALGO from env; defaults to aocl_dlp_blocked.

#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <vector>
#include <omp.h>

#include "lowoha_matmul_utils.hpp"
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
        return (env && env[0] >= '1' && env[0] <= '2') ? (env[0] - '0') : 0;
    }();
    return ver;
}

/// Resolve kernel algo from ZENDNNL_MATMUL_ALGO env. Default: aocl_dlp_blocked.
inline matmul_algo_t resolve_kernel() {
    static const matmul_algo_t algo = []() {
        int32_t a = matmul_config_t::instance().get_algo();
        if (a <= 0 || a >= static_cast<int32_t>(matmul_algo_t::algo_count))
            return matmul_algo_t::aocl_dlp_blocked;
        return static_cast<matmul_algo_t>(a);
    }();
    return algo;
}

/// Execute a single expert's matmul via matmul_execute.
inline void execute_expert(
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

// ── ALGO 1: Per-expert ──────────────────────────────────────────────────
// One OMP iteration per expert. Each expert runs single-threaded.
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
        execute_expert(layout[i], transA[i], transB[i],
            M[i], N[i], K[i], alpha[i],
            src[i], lda[i], weight[i], ldb[i],
            bias[i], beta[i], dst[i], ldc[i],
            is_weights_const[i], 1, params[i], algo);
    }
}

// ── ALGO 2: Multilevel (outer experts x inner threads) ──────────────────
// Each expert gets M-proportional inner threads so heavy experts (large M)
// get more compute resources than light ones.
//
// Adapts to the inner-thread-per-expert ratio:
//   max(thr_per_op) == 1: all experts are single-threaded → flat parallel
//     for (same as ALGO 1 but respects the M-proportional intent). No
//     nested OMP overhead.
//   max(thr_per_op) > 1: nested OMP with active_levels=2. Outer region
//     has num_ops threads, each spawns thr_per_op[i] inner threads.
//     Thread budget: num_ops outer + sum(thr_per_op[i]) = num_threads.
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

    // Inner pool = total threads minus one outer thread per expert.
    const int inner_pool = std::max(1, num_threads - num_ops);
    int total_M = 0;
    for (int i = 0; i < num_ops; ++i) total_M += M[i];
    if (total_M <= 0) total_M = num_ops;

    // M-proportional allocation with round-robin leftover.
    std::vector<int> thr_per_op(num_ops);
    int assigned = 0;
    for (int i = 0; i < num_ops; ++i) {
        thr_per_op[i] = std::max(1, inner_pool * M[i] / total_M);
        assigned += thr_per_op[i];
    }
    for (int i = 0, left = inner_pool - assigned; left > 0; ++i, --left)
        thr_per_op[i % num_ops]++;

    if (inner_pool < 2 * num_ops) {
        // Not enough inner threads per expert on average to justify
        // nesting overhead. Use flat parallel for (same as ALGO 1).
        scoped_active_levels guard(1);
        #pragma omp parallel for num_threads(num_threads)
        for (size_t i = 0; i < static_cast<size_t>(num_ops); ++i) {
            execute_expert(layout[i], transA[i], transB[i],
                M[i], N[i], K[i], alpha[i],
                src[i], lda[i], weight[i], ldb[i],
                bias[i], beta[i], dst[i], ldc[i],
                is_weights_const[i], 1, params[i], algo);
        }
    } else {
        // Multi-threaded experts: nested OMP.
        scoped_active_levels guard(2);
        #pragma omp parallel num_threads(num_ops)
        {
            const int i = omp_get_thread_num();
            if (i < num_ops) {
                execute_expert(layout[i], transA[i], transB[i],
                    M[i], N[i], K[i], alpha[i],
                    src[i], lda[i], weight[i], ldb[i],
                    bias[i], beta[i], dst[i], ldc[i],
                    is_weights_const[i], thr_per_op[i],
                    params[i], algo);
            }
        }
    }
}

} // namespace

// ── Public API ──────────────────────────────────────────────────────────

status_t group_matmul_direct(const std::vector<char> &layout,
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
                             std::vector<matmul_params> &params) {

  if (M.empty() || N.empty() || K.empty() || params.empty() || src.empty() ||
      weight.empty() || dst.empty() || bias.empty() || is_weights_const.empty() ||
      lda.empty() || ldb.empty() || ldc.empty()) {
    log_error("group_matmul_direct: empty input vectors");
    return status_t::failure;
  }

  profiler_t profiler;
  bool is_profile = is_profile_enabled();
  if (is_profile)
    profiler.tbp_start();

  const size_t num_ops = M.size();
  const char *gemm_mode = nullptr;
  static unsigned int auto_version = get_auto_tuner_ver();

  const int32_t omp_mt = thread_guard::max_threads();
  const int32_t num_threads = resolve_num_threads(params[0].num_threads, omp_mt);
  thread_guard tg(num_threads, omp_mt);

  if (src.size() == 1) {
    // Sequential: chained matmuls (output[i-1] feeds input[i]).
    gemm_mode = "sequential";

    if (validate_sequential_gemm_inputs(layout, transA, transB, M, N, K,
            alpha, src, lda, weight, ldb, bias, beta, dst, ldc,
            is_weights_const, params) != status_t::success)
      return status_t::failure;

    for (size_t i = 0; i < num_ops; ++i) {
      matmul_batch_params_t bp;
      bp.Batch_A = 1;
      bp.Batch_B = 1;
      const void *cur_src = (i == 0) ? src[i] : dst[i - 1];
      int cur_lda = (i == 0) ? lda[i] : ldc[i - 1];

      matmul_algo_t kernel = kernel_select(
          params[i], bp.Batch_A, bp.Batch_B,
          1, M[i], N[i], K[i], num_threads, bias[i], is_weights_const[i]);

      params[i].num_threads = num_threads;
      matmul_execute(
          layout[i], transA[i], transB[i],
          M[i], N[i], K[i], alpha[i],
          cur_src, cur_lda, weight[i], ldb[i],
          bias[i], beta[i], dst[i], ldc[i],
          is_weights_const[i],
          size_of(params[i].dtypes.src), size_of(params[i].dtypes.dst),
          num_threads, kernel, params[i], bp, auto_version);
    }
  } else {
    // Parallel: independent matmuls. Select strategy.
    if (validate_parallel_gemm_inputs(layout, transA, transB, M, N, K,
            alpha, src, lda, weight, ldb, bias, beta, dst, ldc,
            is_weights_const, params) != status_t::success)
      return status_t::failure;

    const int algo = get_grp_matmul_algo();
    int use_algo = (algo >= 1 && algo <= 2) ? algo : 1;

    switch (use_algo) {
    case 2:
        gemm_mode = "multilevel";
        parallel_multilevel(layout, transA, transB, M, N, K, alpha,
            src, lda, weight, ldb, bias, beta, dst, ldc,
            is_weights_const, params, num_threads);
        break;
    default:
        gemm_mode = "per_expert";
        parallel_per_expert(layout, transA, transB, M, N, K, alpha,
            src, lda, weight, ldb, bias, beta, dst, ldc,
            is_weights_const, params, num_threads);
        break;
    }
  }

  if (is_profile)
    profiler.tbp_stop();

  if (apilog_info_enabled() || is_profile) {
    [[maybe_unused]] std::ostringstream ss;
    ss << "LOWOHA group_matmul_direct: "
       << "num_ops=" << num_ops
       << ", mode=" << gemm_mode
       << ", num_threads=" << num_threads;
    apilog_info(ss.str());
    if (is_profile)
      profilelog_verbose(ss.str(), ", time=", profiler.tbp_elapsedtime(),
                         profiler.get_res_str());
  }

  return status_t::success;
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
