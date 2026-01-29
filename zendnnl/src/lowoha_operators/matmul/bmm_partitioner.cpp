/*******************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *******************************************************************************/

#include "bmm_partitioner.hpp"
#include "lowoha_matmul_utils.hpp"
#include "lowoha_matmul.hpp"
#include "libxsmm_utils.hpp"
#include <algorithm>
#include <omp.h>

namespace zendnnl {
namespace lowoha {

int calculate_optimal_m_block(const bmm_partition_config_t &config) {
  /*
   * Parallel partitioning strategy:
   * The total number of available threads is divided across batches to compute
   * `threads_per_batch`, ensuring that each batch gets a fair share of compute
   * resources. Within each batch, the M dimension (rows of the output matrix)
   * is further partitioned into blocks of size `M_block`, calculated to evenly
   * distribute the workload among the threads assigned to that batch.
   */
  int threads_per_batch = std::max(1, config.num_threads / config.batch_count);
  int M_block = std::max(1,
                         (config.M + threads_per_batch - 1) / threads_per_batch);

  // Optimize M_block based on batch count and M size
  // TODO: Further refine the tuning based on heuristics
  // involving batch_count, M, and num_threads.
  if ((config.batch_count >= 1024 && config.M <= 2048) ||
      (config.batch_count >= 512 && config.M <= 256) ||
      (config.batch_count > 128 && config.batch_count < 192 && config.M <= 512)) {
    if (config.kernel == matmul_algo_t::libxsmm) {
      M_block = std::min(128, config.M);
    }
    else {
      M_block = std::min(36, config.M);
    }
  }
  else if ((config.batch_count == 64 && config.M >= 512) ||
           (config.batch_count == 128 && config.M >= 512)) {
    M_block = std::min(192, config.M);
  }
  else {
    M_block = std::min(M_block, config.M); // Ensure M_block <= M
  }

  return M_block;
}

bool should_use_zendnnl_parallel(int M, int N, int K) {
  // Decide parallelization strategy based on MFLOPs:
  // Use zendnn_parallel_for when M_FLOPs > 6.0 for better performance on larger workloads.
  // Use omp_parallel_for when M_FLOPs <= 6.0 to avoid overhead on smaller tasks.
  float flops = static_cast<float>(2LL * M * K * N) / 1000000.0f;
  return flops > M_FLOPS;
}

void execute_parallel_zendnnl(
  const void *src,
  const void *weight,
  void *dst,
  const bmm_partition_config_t &config,
  matmul::matmul_batch_params_t &batch_params,
  int M_block,
  const tile_callback_t &callback) {

  interface::apilog_info("Using zendnnl_parallel_for");

  // Calculate total number of work items (batch_count * number of M blocks)
  int total_m_blocks = (config.M + M_block - 1) / M_block;
  int total_work_items = config.batch_count * total_m_blocks;

  matmul::zendnnl_parallel_for(0, total_work_items, 1, [&](int start_idx,
  int end_idx) {
    for (int work_idx = start_idx; work_idx < end_idx; ++work_idx) {
      // Convert linear work index back to (batch, m_block) coordinates
      int b = work_idx / total_m_blocks;
      int m_block_idx = work_idx % total_m_blocks;
      int m_start = m_block_idx * M_block;
      int m_len = std::min(M_block, config.M - m_start);

      const uint8_t *src_ptr = static_cast<const uint8_t *>(src) +
                               matmul::get_batch_index(b, batch_params.Batch_A) *
                               config.src_batch_stride_bytes;
      const uint8_t *weight_ptr = static_cast<const uint8_t *>(weight) +
                                  matmul::get_batch_index(b, batch_params.Batch_B) *
                                  config.weight_batch_stride_bytes;
      uint8_t *dst_ptr = static_cast<uint8_t *>(dst) +
                         b * config.dst_batch_stride_bytes;

      callback(b, m_start, m_len, src_ptr, weight_ptr, dst_ptr);
    }
  });
}

void execute_parallel_omp(
  const void *src,
  const void *weight,
  void *dst,
  const bmm_partition_config_t &config,
  matmul::matmul_batch_params_t &batch_params,
  int M_block,
  const tile_callback_t &callback) {

  interface::apilog_info("Using OpenMP parallel for");

  #pragma omp parallel for collapse(2)
  for (int b = 0; b < config.batch_count; ++b) {
    for (int m_start = 0; m_start < config.M; m_start += M_block) {
      int m_len = std::min(M_block, config.M - m_start);

      const uint8_t *src_ptr = static_cast<const uint8_t *>(src) +
                               matmul::get_batch_index(b, batch_params.Batch_A) *
                               config.src_batch_stride_bytes;
      const uint8_t *weight_ptr = static_cast<const uint8_t *>(weight) +
                                  matmul::get_batch_index(b, batch_params.Batch_B) *
                                  config.weight_batch_stride_bytes;
      uint8_t *dst_ptr = static_cast<uint8_t *>(dst) +
                         b * config.dst_batch_stride_bytes;

      callback(b, m_start, m_len, src_ptr, weight_ptr, dst_ptr);
    }
  }
}

void execute_bmm_partition(
  const void *src,
  const void *weight,
  void *dst,
  const void *bias,
  bmm_partition_config_t &config,
  matmul::matmul_batch_params_t &batch_params,
  matmul::matmul_params &params,
  char layout,
  char trans_input,
  char trans_weight,
  bool transA,
  float alpha,
  float beta,
  int lda,
  int ldb,
  int ldc,
  size_t src_type_size,
  size_t out_type_size,
  bool is_weights_const) {

  // Calculate optimal M_block size
  int M_block = calculate_optimal_m_block(config);

  // Check libxsmm compatibility and fallback if needed
  if (config.kernel == matmul_algo_t::libxsmm &&
      !(matmul::can_use_libxsmm(trans_input, trans_weight, M_block, config.N,
                                config.K, alpha, beta, params,
                                config.kernel))) {
    // Fallback to AOCL DLP kernel when libxsmm is not supported
    interface::apilog_info("Using AOCL DLP kernel as fallback for libxsmm, algo: ",
                             static_cast<int>(config.kernel));
    config.kernel = matmul_algo_t::aocl_dlp;
  }

  // Define the tile processing callback
  auto process_tile = [&](int batch_idx, int m_start, int m_len,
                          const uint8_t *src_ptr, const uint8_t *weight_ptr,
  uint8_t *dst_ptr) {
    const void *A = matmul::get_matrix_block(src_ptr, m_start, 0, lda, transA,
                    src_type_size);
    void *C = matmul::get_output_block(dst_ptr, m_start, 0, ldc, out_type_size);

    // Create a modified post_op with offset binary tensor buffers
    // Supports both 2D (M x N) and 3D (Batch x M x N) post-op tensors
    matmul::matmul_params thread_lowoha_params = params;
    apply_bmm_postop_offsets(thread_lowoha_params, batch_idx, m_start, config.N);

    matmul::matmul_kernel_wrapper(layout, trans_input, trans_weight,
                                  m_len, config.N, config.K, alpha,
                                  A, lda, weight_ptr, ldb,
                                  beta, C, ldc,
                                  params.dtypes, config.kernel,
                                  params.mem_format_a, params.mem_format_b,
                                  thread_lowoha_params, batch_params,
                                  bias, is_weights_const);
  };

  interface::apilog_info("Executing BMM LOWOHA kernel with parallel partitioning, algo: ",
                           static_cast<int>(config.kernel));

  // Execute with automatic strategy selection
  matmul::matmul_active_levels active_levels_guard(1);

  if (should_use_zendnnl_parallel(config.M, config.N, config.K)) {
    execute_parallel_zendnnl(src, weight, dst, config, batch_params,
                             M_block, process_tile);
  }
  else {
    execute_parallel_omp(src, weight, dst, config, batch_params,
                         M_block, process_tile);
  }
}

} // namespace lowoha
} // namespace zendnnl

