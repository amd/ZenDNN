/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
 ******************************************************************************/

#include "lowoha_operators/matmul/partitioning/bmm/looper/bmm_looper.hpp"
#include "lowoha_operators/matmul/partitioning/bmm/planner/bmm_planner.hpp"
#include "lowoha_operators/matmul/partitioning/bmm/kernel/bmm_kernel.hpp"
#include "lowoha_operators/matmul/lowoha_matmul.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"
#include "lowoha_operators/matmul/backends/libxsmm/libxsmm_utils.hpp"
#include "lowoha_operators/matmul/backends/aocl/aocl_kernel.hpp"
#include "lowoha_operators/matmul/backends/onednn/onednn_kernel.hpp"
#include "lowoha_operators/common/omp_thread_control.hpp"
#include <algorithm>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace bmm {

// ─── Parallel loop implementations ─────────────────────────────────────────

// Linearize the (batch, m_block) iteration space into a 1-D range and
// distribute it with zendnnl_parallel_for.  Each chunk of work items is
// mapped back to (batch_idx, m_start, m_len) coordinates before invoking
// the tile callback.  Preferred for large workloads where the chunked
// task-based scheduler gives better load balance than static partitioning.
static void run_parallel_zendnnl(
  const void *src, const void *weight, void *dst,
  const BmmConfig &config,
  matmul_batch_params_t &batch_params,
  int M_block,
  const bmm_tile_callback_t &callback) {

  apilog_info("Using zendnnl_parallel_for");

  int64_t total_m_blocks = (config.M + M_block - 1) / M_block;
  int64_t total_work_items = static_cast<int64_t>(config.batch_count) *
                             total_m_blocks;

  zendnnl_parallel_for(0, total_work_items, 1, [&](int64_t start_idx,
  int64_t end_idx) {
    for (int64_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
      int b = static_cast<int>(work_idx / total_m_blocks);
      int m_block_idx = static_cast<int>(work_idx % total_m_blocks);
      int m_start = m_block_idx * M_block;
      int m_len = std::min(M_block, config.M - m_start);

      const uint8_t *src_ptr = static_cast<const uint8_t *>(src) +
                               get_batch_index(b, batch_params.Batch_A) *
                               config.src_batch_stride_bytes;
      const uint8_t *weight_ptr = static_cast<const uint8_t *>(weight) +
                                  get_batch_index(b, batch_params.Batch_B) *
                                  config.weight_batch_stride_bytes;
      uint8_t *dst_ptr = static_cast<uint8_t *>(dst) +
                         b * config.dst_batch_stride_bytes;

      callback(b, m_start, m_len, src_ptr, weight_ptr, dst_ptr);
    }
  });
}

// Use OpenMP collapse(2) over (batch, m_start) to statically partition
// tiles across threads.  Lower scheduling overhead than zendnnl_parallel_for,
// so this is preferred for small workloads (MFLOPs <= threshold) where the
// per-tile cost is small and dynamic scheduling would dominate runtime.
static void run_parallel_omp(
  const void *src, const void *weight, void *dst,
  const BmmConfig &config,
  matmul_batch_params_t &batch_params,
  int M_block,
  const bmm_tile_callback_t &callback) {

  apilog_info("Using OpenMP parallel for");

  #pragma omp parallel for collapse(2)
  for (int b = 0; b < config.batch_count; ++b) {
    for (int m_start = 0; m_start < config.M; m_start += M_block) {
      int m_len = std::min(M_block, config.M - m_start);

      const uint8_t *src_ptr = static_cast<const uint8_t *>(src) +
                               get_batch_index(b, batch_params.Batch_A) *
                               config.src_batch_stride_bytes;
      const uint8_t *weight_ptr = static_cast<const uint8_t *>(weight) +
                                  get_batch_index(b, batch_params.Batch_B) *
                                  config.weight_batch_stride_bytes;
      uint8_t *dst_ptr = static_cast<uint8_t *>(dst) +
                         b * config.dst_batch_stride_bytes;

      callback(b, m_start, m_len, src_ptr, weight_ptr, dst_ptr);
    }
  }
}

// ─── Multi-thread partitioned execution ─────────────────────────────────────

static void execute_partitioned(
  const void *src, const void *weight, void *dst,
  const void *bias,
  BmmConfig &config,
  matmul_batch_params_t &batch_params,
  matmul_params &params,
  char layout,
  char trans_input, char trans_weight,
  bool transA,
  float alpha, float beta,
  int lda, int ldb, int ldc,
  size_t src_type_size, size_t out_type_size,
  bool is_weights_const) {

  BmmPlan plan = plan_bmm(config);

  if (config.kernel == matmul_algo_t::libxsmm &&
      !(can_use_libxsmm(trans_input, trans_weight, plan.M_block, config.N,
                        config.K, alpha, beta, params))) {
    apilog_info("Using AOCL DLP kernel as fallback for libxsmm, algo: ",
                static_cast<int>(config.kernel));
    config.kernel = matmul_algo_t::aocl_dlp;
  }

  BmmKernelContext ctx;
  ctx.layout = layout;
  ctx.trans_input = trans_input;
  ctx.trans_weight = trans_weight;
  ctx.transA = transA;
  ctx.N = config.N;
  ctx.K = config.K;
  ctx.alpha = alpha;
  ctx.beta = beta;
  ctx.lda = lda;
  ctx.ldb = ldb;
  ctx.ldc = ldc;
  ctx.src_type_size = src_type_size;
  ctx.out_type_size = out_type_size;
  ctx.kernel = config.kernel;
  ctx.bias = bias;
  ctx.is_weights_const = is_weights_const;

  auto process_tile = [&](int batch_idx, int m_start, int m_len,
                          const uint8_t *src_ptr, const uint8_t *weight_ptr,
                          uint8_t *dst_ptr) {
    bmm_tile_execute(batch_idx, m_start, m_len,
                     src_ptr, weight_ptr, dst_ptr,
                     ctx, params, batch_params);
  };

  apilog_info("Executing BMM LOWOHA kernel with parallel partitioning, algo: ",
              static_cast<int>(config.kernel));

  scoped_active_levels active_levels_guard(1);

  if (plan.use_zendnnl_parallel) {
    run_parallel_zendnnl(src, weight, dst, config, batch_params,
                         plan.M_block, process_tile);
  }
  else {
    run_parallel_omp(src, weight, dst, config, batch_params,
                     plan.M_block, process_tile);
  }
}

// ─── BMM entry point ────────────────────────────────────────────────────────

void bmm_execute(const char layout, const bool transA, const bool transB,
                 const int M, const int N, const int K, const float alpha,
                 const void *src, const int lda,
                 const void *weight, const int ldb,
                 const void *bias, const float beta,
                 void *dst, const int ldc,
                 const bool is_weights_const,
                 matmul_batch_params_t &batch_params,
                 const size_t src_type_size, const size_t weight_type_size,
                 const size_t out_type_size, const int num_threads,
                 matmul_algo_t &kernel, matmul_params &params) {

  const int batch_count = std::max(batch_params.Batch_A, batch_params.Batch_B);
  const char trans_input  = transA ? 't' : 'n';
  const char trans_weight = transB ? 't' : 'n';

  size_t src_batch_stride_elems = (batch_params.batch_stride_src !=
                                   static_cast<size_t>(-1))
                                  ? batch_params.batch_stride_src
                                  : (transA ? K *lda : M * lda);
  size_t weight_batch_stride_elems = (batch_params.batch_stride_wei !=
                                      static_cast<size_t>(-1))
                                     ? batch_params.batch_stride_wei
                                     : (transB ? N *ldb : K * ldb);
  size_t dst_batch_stride_elems = (batch_params.batch_stride_dst !=
                                   static_cast<size_t>(-1))
                                  ? batch_params.batch_stride_dst
                                  : M * ldc;

  size_t src_batch_stride_bytes = src_batch_stride_elems * src_type_size;
  size_t weight_batch_stride_bytes = weight_batch_stride_elems * weight_type_size;
  size_t dst_batch_stride_bytes = dst_batch_stride_elems * out_type_size;

  // ── Path 1: Batched SGEMM ──
  if (kernel == matmul_algo_t::batched_sgemm) {
    apilog_info("Executing BMM LOWOHA kernel with batch SGEMM, algo: ",
                static_cast<int>(kernel));

    matmul_batch_gemm_wrapper(layout, trans_input, trans_weight,
                              M, N, K, alpha,
                              src, lda, weight, ldb, beta, dst, ldc,
                              params.dtypes, batch_count,
                              batch_params.Batch_A, batch_params.Batch_B,
                              params.mem_format_a, params.mem_format_b,
                              src_batch_stride_bytes, weight_batch_stride_bytes,
                              dst_batch_stride_bytes,
                              params, bias, num_threads);
    return;
  }

  // ── Path 2: oneDNN ──
#if ZENDNNL_DEPENDS_ONEDNN
  if (kernel == matmul_algo_t::onednn ||
      kernel == matmul_algo_t::onednn_blocked) {
    apilog_info("Executing BMM LOWOHA kernel with oneDNN, algo: ",
                static_cast<int>(kernel));

    matmul_onednn_wrapper(trans_input, trans_weight, M, N, K, alpha, src, lda,
                          weight, ldb, beta, dst, ldc, params, batch_params,
                          bias, kernel, is_weights_const,
                          src_batch_stride_elems, weight_batch_stride_elems,
                          dst_batch_stride_elems);
    return;
  }
#endif

  // ── Path 3: Multi-thread partitioned ──
  if (num_threads > 1) {
    BmmConfig config;
    config.M = M;
    config.N = N;
    config.K = K;
    config.batch_count = batch_count;
    config.num_threads = num_threads;
    config.kernel = kernel;
    config.src_batch_stride_bytes = src_batch_stride_bytes;
    config.weight_batch_stride_bytes = weight_batch_stride_bytes;
    config.dst_batch_stride_bytes = dst_batch_stride_bytes;

    execute_partitioned(src, weight, dst, bias,
                        config, batch_params, params,
                        layout, trans_input, trans_weight, transA,
                        alpha, beta, lda, ldb, ldc,
                        src_type_size, out_type_size, is_weights_const);
  }
  // ── Path 4: Single-thread fallback ──
  else {
    if (kernel == matmul_algo_t::libxsmm &&
        !(can_use_libxsmm(trans_input, trans_weight, M, N, K, alpha, beta,
                          params))) {
      kernel = matmul_algo_t::aocl_dlp;
    }

    apilog_info("Executing BMM LOWOHA kernel without zendnnl-partitioner, algo: ",
                static_cast<int>(kernel));

    BmmKernelContext ctx;
    ctx.layout = layout;
    ctx.trans_input = trans_input;
    ctx.trans_weight = trans_weight;
    ctx.transA = transA;
    ctx.N = N;
    ctx.K = K;
    ctx.alpha = alpha;
    ctx.beta = beta;
    ctx.lda = lda;
    ctx.ldb = ldb;
    ctx.ldc = ldc;
    ctx.src_type_size = src_type_size;
    ctx.out_type_size = out_type_size;
    ctx.kernel = kernel;
    ctx.bias = bias;
    ctx.is_weights_const = is_weights_const;

    for (int b = 0; b < batch_count; ++b) {
      const uint8_t *src_ptr = static_cast<const uint8_t *>(src) +
                               get_batch_index(b, batch_params.Batch_A) *
                               src_batch_stride_bytes;
      const uint8_t *weight_ptr = static_cast<const uint8_t *>(weight) +
                                  get_batch_index(b, batch_params.Batch_B) *
                                  weight_batch_stride_bytes;
      uint8_t *dst_ptr = static_cast<uint8_t *>(dst) +
                         b * dst_batch_stride_bytes;

      bmm_tile_execute(b, 0, M,
                       src_ptr, weight_ptr, dst_ptr,
                       ctx, params, batch_params);
    }
  }
}

} // namespace bmm
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
