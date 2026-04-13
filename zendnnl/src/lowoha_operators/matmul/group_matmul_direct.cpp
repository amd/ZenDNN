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

/// Group MatMul direct — public entry point and input validation.
///
/// Implementation details:
///   - group_matmul/group_matmul_parallel.cpp — parallel expert dispatch (OMP).
///   - group_matmul/group_matmul_moe_postop.cpp — optional MoE weighted-reduce post-op.

#include <sstream>
#include <vector>

#include "group_matmul/group_matmul_direct.hpp"
#include "lowoha_matmul_utils.hpp"
#include "lowoha_operators/common/omp_thread_control.hpp"
#include "lowoha_operators/common/operator_instrumentation.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

using namespace zendnnl::ops;
using zendnnl::common::size_of;
using zendnnl::common::op_instrumentation;

namespace {

status_t validate_group_matmul_nonempty_vectors(
    const std::vector<int> &M,
    const std::vector<int> &N,
    const std::vector<int> &K,
    const std::vector<matmul_params> &params,
    const std::vector<const void *> &src,
    const std::vector<const void *> &weight,
    const std::vector<void *> &dst,
    const std::vector<const void *> &bias,
    const std::vector<bool> &is_weights_const,
    const std::vector<int> &lda,
    const std::vector<int> &ldb,
    const std::vector<int> &ldc) {

  if (M.empty() || N.empty() || K.empty() || params.empty() || src.empty() ||
      weight.empty() || dst.empty() || bias.empty() || is_weights_const.empty() ||
      lda.empty() || ldb.empty() || ldc.empty()) {
    log_error("group_matmul_direct: empty input vectors");
    return status_t::failure;
  }
  return status_t::success;
}

status_t validate_parallel_gemm_inputs(
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
    const std::vector<matmul_params> &params) {

  const size_t num_ops = M.size();

  if (num_ops == 0) {
    log_error("group_matmul parallel: num_ops is 0");
    return status_t::failure;
  }

  if (layout.size() != num_ops || transA.size() != num_ops ||
      transB.size() != num_ops || N.size() != num_ops ||
      K.size() != num_ops || alpha.size() != num_ops ||
      src.size() != num_ops || lda.size() != num_ops ||
      weight.size() != num_ops || ldb.size() != num_ops ||
      bias.size() != num_ops || beta.size() != num_ops ||
      dst.size() != num_ops || ldc.size() != num_ops ||
      is_weights_const.size() != num_ops || params.size() != num_ops) {
    log_error("group_matmul parallel: vector size mismatch, num_ops=", num_ops);
    return status_t::failure;
  }

  for (size_t i = 0; i < num_ops; ++i) {
    if (!src[i] || !weight[i] || !dst[i]) {
      log_error("group_matmul parallel: null pointer at operation ", i);
      return status_t::failure;
    }
    if (M[i] <= 0 || N[i] <= 0 || K[i] <= 0) {
      log_error("group_matmul parallel: invalid dimensions at operation ", i,
                ": M=", M[i], ", N=", N[i], ", K=", K[i]);
      return status_t::failure;
    }
  }

  return status_t::success;
}

status_t validate_sequential_gemm_inputs(
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
    const std::vector<matmul_params> &params) {

  const size_t num_ops = M.size();

  if (num_ops == 0) {
    log_error("group_matmul sequential: num_ops is 0");
    return status_t::failure;
  }

  if (src.size() != 1) {
    log_error("group_matmul sequential: src.size() must be 1, got ", src.size());
    return status_t::failure;
  }

  if (layout.size() != num_ops || transA.size() != num_ops ||
      transB.size() != num_ops || N.size() != num_ops ||
      K.size() != num_ops || alpha.size() != num_ops ||
      lda.size() != num_ops || weight.size() != num_ops ||
      ldb.size() != num_ops || bias.size() != num_ops ||
      beta.size() != num_ops || dst.size() != num_ops ||
      ldc.size() != num_ops || is_weights_const.size() != num_ops ||
      params.size() != num_ops) {
    log_error("group_matmul sequential: vector size mismatch, num_ops=", num_ops);
    return status_t::failure;
  }

  if (!src[0]) {
    log_error("group_matmul sequential: null src pointer");
    return status_t::failure;
  }

  for (size_t i = 0; i < num_ops; ++i) {
    if (!weight[i] || !dst[i]) {
      log_error("group_matmul sequential: null pointer at operation ", i);
      return status_t::failure;
    }
    if (M[i] <= 0 || N[i] <= 0 || K[i] <= 0) {
      log_error("group_matmul sequential: invalid dimensions at operation ", i,
                ": M=", M[i], ", N=", N[i], ", K=", K[i]);
      return status_t::failure;
    }
  }

  for (size_t i = 1; i < num_ops; ++i) {
    if (M[i] != M[0]) {
      log_error("group_matmul sequential: M must be constant across layers, "
                "M[0]=", M[0], ", M[", i, "]=", M[i]);
      return status_t::failure;
    }
  }

  for (size_t i = 1; i < num_ops; ++i) {
    if (K[i] != N[i - 1]) {
      log_error("group_matmul sequential: dimension mismatch at layer ", i,
                ": K[", i, "]=", K[i], " != N[", i - 1, "]=", N[i - 1]);
      return status_t::failure;
    }
  }

  return status_t::success;
}

} // namespace

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
                             std::vector<matmul_params> &params,
                             const group_matmul_moe_postop_params *moe_postop) {

  // Always-on guard: check all vectors before any indexing.
  // Enforces size consistency without per-element loops — single branch.
  if (M.empty() || params.empty() || src.empty())
    return status_t::failure;

  const size_t num_ops = M.size();

  if (N.size() != num_ops || K.size() != num_ops || weight.size() != num_ops ||
      dst.size() != num_ops || lda.size() != num_ops || ldb.size() != num_ops ||
      ldc.size() != num_ops || layout.size() != num_ops ||
      transA.size() != num_ops || transB.size() != num_ops ||
      alpha.size() != num_ops || beta.size() != num_ops ||
      bias.size() != num_ops || is_weights_const.size() != num_ops ||
      params.size() != num_ops) {
    log_error("group_matmul_direct: vector size mismatch");
    return status_t::failure;
  }
  if (src.size() != 1 && src.size() != num_ops) {
    log_error("group_matmul_direct: src.size() must be 1 or num_ops");
    return status_t::failure;
  }

  // MoE post-op is parallel mode only.
  if (moe_postop != nullptr && src.size() == 1) {
    log_error("group_matmul_direct: moe_postop is only supported in parallel mode");
    return status_t::failure;
  }

  // Validate remaining inputs only when ZENDNNL_DIAGNOSTICS_ENABLE=1.
  status_t val = op_instrumentation::validate([&]() {
    if (validate_group_matmul_nonempty_vectors(M, N, K, params, src, weight, dst,
            bias, is_weights_const, lda, ldb, ldc) != status_t::success)
      return status_t::failure;
    // When MoE is enabled, all experts must have identical N (hidden dim)
    // and dst dtype — the weighted-reduce reads all expert rows uniformly.
    if (moe_postop != nullptr) {
      for (size_t i = 1; i < num_ops; ++i) {
        if (N[i] != N[0]) {
          log_error("group_matmul_direct: moe_postop requires identical N across experts");
          return status_t::failure;
        }
        if (params[i].dtypes.dst != params[0].dtypes.dst) {
          log_error("group_matmul_direct: moe_postop requires identical dst dtype across experts");
          return status_t::failure;
        }
      }
    }
    if (validate_group_matmul_moe_postop(moe_postop, N[0],
                                          params[0].dtypes.dst) != status_t::success)
      return status_t::failure;
    return status_t::success;
  });
  if (val != status_t::success)
    return val;

  profiler_t profiler;
  bool is_profile = is_profile_enabled();
  if (is_profile)
    profiler.tbp_start();

  const char *gemm_mode = nullptr;

  const int32_t omp_mt = thread_guard::max_threads();
  const int32_t num_threads = resolve_num_threads(params[0].num_threads, omp_mt);
  thread_guard tg(num_threads, omp_mt);

  if (src.size() == 1) {
    val = op_instrumentation::validate([&]() {
      return validate_sequential_gemm_inputs(layout, transA, transB, M, N, K,
                 alpha, src, lda, weight, ldb, bias, beta, dst, ldc,
                 is_weights_const, params);
    });
    if (val != status_t::success)
      return val;

    static unsigned int auto_version = get_auto_tuner_ver();
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
    gemm_mode = "sequential";
  } else {
    val = op_instrumentation::validate([&]() {
      return validate_parallel_gemm_inputs(layout, transA, transB, M, N, K,
                 alpha, src, lda, weight, ldb, bias, beta, dst, ldc,
                 is_weights_const, params);
    });
    if (val != status_t::success)
      return val;

    group_matmul_run_parallel_dispatch(layout, transA, transB, M, N, K, alpha,
        src, lda, weight, ldb, bias, beta, dst, ldc,
        is_weights_const, params, num_threads, &gemm_mode);

    if (moe_postop != nullptr) {
      status_t moe_val = validate_group_matmul_moe_postop(
          moe_postop, N[0], params[0].dtypes.dst);
      if (moe_val != status_t::success)
        return moe_val;
      status_t moe_st = group_matmul_moe_postop_execute(moe_postop, N[0],
          num_threads, params[0].dtypes.dst);
      if (moe_st != status_t::success)
        return moe_st;
    }
  }

  if (is_profile)
    profiler.tbp_stop();

  if (apilog_info_enabled() || is_profile) {
    std::ostringstream ss;
    ss << "LOWOHA group_matmul_direct: "
       << "num_ops=" << num_ops
       << ", mode=" << (gemm_mode != nullptr ? gemm_mode : "null")
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
