/*******************************************************************************
# * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lowoha_matmul_utils.hpp"
#include "bmm_partitioner.hpp"
#include "matmul_partitioner.hpp"
#include "lowoha_operators/matmul/libxsmm_kernel.hpp"
#include "lowoha_operators/matmul/aocl_kernel.hpp"
#include "lowoha_operators/matmul/onednn_kernel.hpp"
#include "lowoha_operators/matmul/auto_tuner.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

using namespace zendnnl::ops;
using zendnnl::common::size_of;

void matmul_kernel_wrapper(char layout, char transA, char transB,
                           int M, int N, int K,
                           float alpha,
                           const void *A, int lda,
                           const void *B, int ldb,
                           float beta,
                           void *C, int ldc,
                           matmul_data_types &dtypes,
                           zendnnl::ops::matmul_algo_t kernel,
                           char mem_format_a, char mem_format_b,
                           matmul_params &lowoha_param, matmul_batch_params_t &batch_params,
                           const void *bias, bool is_weights_const) {
#if ZENDNNL_DEPENDS_LIBXSMM
  if (kernel == matmul_algo_t::libxsmm) {
    log_info("Using libxsmm kernel");
    if (run_libxsmm(transA, transB, M, N, K, beta, lda, ldb, ldc, A, B, C, dtypes,
                    lowoha_param, bias)) {
      return;
    }
  }
#endif
#if ZENDNNL_DEPENDS_ONEDNN
  if (kernel == matmul_algo_t::onednn ||
      kernel == matmul_algo_t::onednn_blocked) {
    log_info("Using onednn kernel");
    matmul_onednn_wrapper(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C,
                          ldc, lowoha_param, batch_params, bias, kernel, is_weights_const);
    return;
  }
#endif
  log_info("Using AOCL DLP kernel");
  run_dlp(layout, transA, transB, M, N, K, alpha, beta,
          lda, ldb, ldc, mem_format_a, mem_format_b,
          A, B, C, dtypes, lowoha_param, bias, kernel, is_weights_const);
  return;

  //   TODO: To implement native AVX512 BF16 kernel
  //   else if (0) {
  //     if (dtypes.src == data_type_t::bf16 &&
  //         dtypes.dst == data_type_t::bf16) {
  //       matmul_bf16_dispatch(static_cast<const int16_t *>(A),
  //                            static_cast<const int16_t *>(B),
  //                            static_cast<int16_t *>(C), nullptr /*bias.data()*/,
  //                            alpha, beta, M, N, K, lda, ldb, ldc, false);
  //     }
  //   }
}

void bmm_execute(const char layout, const bool transA, const bool transB,
                 const int M, const int N, const int K, const float alpha,
                 const void *src, const int lda,
                 const void *weight, const int ldb,
                 const void *bias, const float beta,
                 void *dst, const int ldc,
                 const bool is_weights_const,
                 matmul_batch_params_t &batch_params,
                 const size_t src_type_size, const size_t out_type_size,
                 const int num_threads,
                 matmul_algo_t kernel, matmul_params &params) {

  const int batch_count = std::max(batch_params.Batch_A, batch_params.Batch_B);
  const char trans_input  = transA ? 't' : 'n';
  const char trans_weight = transB ? 't' : 'n';

  // Calculate batch strides in elements (not bytes)
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
  size_t weight_batch_stride_bytes = weight_batch_stride_elems * src_type_size;
  size_t dst_batch_stride_bytes = dst_batch_stride_elems * out_type_size;

  if (kernel == matmul_algo_t::batched_sgemm) {
    apilog_info("Executing BMM LOWOHA kernel with batch SGEMM, algo: ",
                static_cast<int>(kernel));

    matmul_batch_gemm_wrapper(layout, trans_input, trans_weight,
                              M, N, K, alpha,
                              src, lda, weight, ldb, beta, dst, ldc,
                              params.dtypes, batch_count,
                              batch_params.Batch_A, batch_params.Batch_B,
                              params.mem_format_a, params.mem_format_b,
                              src_batch_stride_bytes, weight_batch_stride_bytes, dst_batch_stride_bytes,
                              params, bias, num_threads);
    return;
  }

#if ZENDNNL_DEPENDS_ONEDNN
  if (kernel == matmul_algo_t::onednn ||
      kernel == matmul_algo_t::onednn_blocked) {
    apilog_info("Executing BMM LOWOHA kernel with oneDNN, algo: ",
                static_cast<int>(kernel));

    matmul_onednn_wrapper(trans_input, trans_weight, M, N, K, alpha, src, lda,
                          weight, ldb, beta, dst, ldc, params, batch_params, bias, kernel,
                          is_weights_const, src_batch_stride_elems, weight_batch_stride_elems,
                          dst_batch_stride_elems);
    return;
  }
#endif

  if (num_threads > 1) {
    // Setup partition configuration
    bmm_partition_config_t part_config;
    part_config.M = M;
    part_config.N = N;
    part_config.K = K;
    part_config.batch_count = batch_count;
    part_config.num_threads = num_threads;
    part_config.kernel = kernel;
    part_config.src_batch_stride_bytes = src_batch_stride_bytes;
    part_config.weight_batch_stride_bytes = weight_batch_stride_bytes;
    part_config.dst_batch_stride_bytes = dst_batch_stride_bytes;

    int M_block = calculate_optimal_m_block(part_config);

    if (kernel == matmul_algo_t::libxsmm &&
        !(can_use_libxsmm(trans_input, trans_weight, M_block, N, K, alpha, beta,
                          params.dtypes, params, kernel))) {
      // Fallback to AOCL DLP kernel when libxsmm is not supported
      apilog_info("Using AOCL DLP kernel as fallback for libxsmm, algo: ",
                  static_cast<int>(kernel));
      kernel = matmul_algo_t::aocl_dlp;
    }
    // Define the tile processing callback
    auto process_tile = [&](int batch_idx, int m_start, int m_len,
                            const uint8_t *src_ptr, const uint8_t *weight_ptr,
    uint8_t *dst_ptr) {
      const void *A = get_matrix_block(src_ptr, m_start, 0, lda, transA,
                                       src_type_size);
      void *C = get_output_block(dst_ptr, m_start, 0, ldc, out_type_size);

      // Create a modified post_op with offset binary tensor buffers
      // Supports both 2D (M x N) and 3D (Batch x M x N) post-op tensors
      matmul_params thread_lowoha_params = params;
      apply_bmm_postop_offsets(thread_lowoha_params, batch_idx, m_start, N);

      matmul_kernel_wrapper(layout, trans_input, trans_weight,
                            m_len, N, K, alpha,
                            A, lda, weight_ptr, ldb,
                            beta, C, ldc,
                            params.dtypes, kernel,
                            params.mem_format_a, params.mem_format_b,
                            thread_lowoha_params, batch_params,
                            bias, is_weights_const);
    };

    // Execute partitioned BMM with automatic strategy selection
    matmul_active_levels active_levels_guard(1);
    execute_partitioned_bmm(src, weight, dst, part_config, batch_params,
                            process_tile);
  }
  else {
    // Single thread execution for batches
    if (kernel == matmul_algo_t::libxsmm &&
        !(can_use_libxsmm(trans_input, trans_weight, M, N, K, alpha, beta,
                          params.dtypes, params, kernel))) {
      kernel = matmul_algo_t::aocl_dlp;
    }

    apilog_info("Executing BMM LOWOHA kernel without zendnnl-partitioner, algo: ",
                static_cast<int>(kernel));

    for (int b = 0; b < batch_count; ++b) {
      const uint8_t *src_ptr = static_cast<const uint8_t *>(src) +
                               get_batch_index(b, batch_params.Batch_A) * src_batch_stride_bytes;
      const uint8_t *weight_ptr = static_cast<const uint8_t *>(weight) +
                                  get_batch_index(b, batch_params.Batch_B) * weight_batch_stride_bytes;
      uint8_t *dst_ptr = static_cast<uint8_t *>(dst) + b * dst_batch_stride_bytes;

      // Create batch-specific params with offset post-op buffers for 3D post-ops
      matmul_params batch_lowoha_params = params;
      apply_bmm_postop_offsets(batch_lowoha_params, b, 0, N);

      matmul_kernel_wrapper(layout, trans_input, trans_weight,
                            M, N, K, alpha,
                            src_ptr, lda, weight_ptr,
                            ldb, beta, dst_ptr, ldc,
                            params.dtypes, kernel,
                            params.mem_format_a, params.mem_format_b, batch_lowoha_params,
                            batch_params, bias, is_weights_const);
    }
  }
}

void matmul_execute(const char layout,
                    const bool transA, const bool transB,
                    const int M, const int N, const int K, const float alpha,
                    const void *src, const int lda,
                    const void *weight, const int ldb,
                    const void *bias, const float beta,
                    void *dst, const int ldc,
                    const bool is_weights_const,
                    const size_t src_type_size, const size_t out_type_size,
                    const int num_threads,
                    matmul_algo_t kernel, matmul_params &params,
                    matmul_batch_params_t &batch_params,
                    unsigned int auto_version) {

  const char trans_input  = transA ? 't' : 'n';
  const char trans_weight = transB ? 't' : 'n';

  // Auto-tuner kernel selection for single batch
  if (kernel == matmul_algo_t::auto_tuner) {
    if (auto_version == 1) {
      kernel = auto_compute_matmul_v1(layout, trans_input, trans_weight, M,
                                      N, K, alpha, src, lda, weight, ldb,
                                      beta, dst, ldc, params.dtypes,
                                      kernel, params.mem_format_a, params.mem_format_b,
                                      params, batch_params, bias, is_weights_const);
    }
    else {
      kernel = auto_compute_matmul_v2(layout, trans_input, trans_weight, M,
                                      N, K, alpha, src, lda, weight, ldb,
                                      beta, dst, ldc, params.dtypes,
                                      kernel, params.mem_format_a, params.mem_format_b,
                                      params, batch_params, bias, is_weights_const);
    }
    return;
  }

// Currently supported only for LIBXSMM BACKEND
  if (should_use_mm_partitioner()) {
    // Setup partition configuration
    matmul_partition_config_t part_config;
    part_config.M = M;
    part_config.N = N;
    part_config.K = K;
    part_config.num_threads = num_threads;
    part_config.kernel = kernel;
    part_config.src_type_size = src_type_size;
    part_config.out_type_size = out_type_size;
    part_config.lda = lda;
    part_config.ldb = ldb;
    part_config.ldc = ldc;
    part_config.transA = transA;
    part_config.transB = transB;
    part_config.dtypes = params.dtypes;

    execute_partitioned_matmul(
      layout, trans_input, trans_weight,
      src, weight, dst, bias,
      part_config, params, batch_params,
      is_weights_const, alpha, beta
    );
    return;
  }

#if ZENDNNL_DEPENDS_ONEDNN
  if (kernel == matmul_algo_t::onednn ||
      kernel == matmul_algo_t::onednn_blocked) {
    apilog_info("Executing matmul LOWOHA kernel with oneDNN, algo: ",
                static_cast<int>(kernel));

    matmul_onednn_wrapper(trans_input, trans_weight, M, N, K, alpha, src, lda,
                          weight, ldb, beta, dst, ldc, params, batch_params, bias, kernel,
                          is_weights_const);
    return;
  }

#endif

  if (kernel == matmul_algo_t::libxsmm &&
      !(can_use_libxsmm(trans_input, trans_weight, M, N, K, alpha, beta,
                        params.dtypes, params, kernel))) {
    kernel = matmul_algo_t::aocl_dlp;
  }

  apilog_info("Executing matmul LOWOHA kernel without zendnnl-partitioner, algo: ",
              static_cast<int>(kernel));
  matmul_kernel_wrapper(layout, trans_input, trans_weight,
                        M, N, K, alpha,
                        src, lda, weight,
                        ldb, beta, dst, ldc,
                        params.dtypes, kernel,
                        params.mem_format_a, params.mem_format_b, params,
                        batch_params, bias, is_weights_const);
}

status_t matmul_direct(const char layout, const bool transA, const bool transB,
                       const int M, const int N, const int K, const float alpha, const void *src,
                       const int lda, const void *weight, const int ldb, const void *bias,
                       const float beta, void *dst, const int ldc, const bool is_weights_const,
                       matmul_batch_params_t batch_params, matmul_params params) {
  // Create profiler instance for timing
  profiler_t profiler;
  bool is_profile = is_profile_enabled();
  if (is_profile) {
    profiler.tbp_start();
  }
  if (validate_matmul_direct_inputs(src, weight, dst, M, N, K,
                                    batch_params.Batch_A, batch_params.Batch_B,
                                    params, is_weights_const) != status_t::success) {
    return status_t::failure;
  }

  size_t src_type_size = size_of(params.dtypes.src);
  size_t out_type_size = size_of(params.dtypes.dst);

  const int batch_count = std::max(batch_params.Batch_A, batch_params.Batch_B);
  const int num_threads = params.num_threads > 0 ? params.num_threads :
                          omp_get_max_threads();

  matmul_algo_t kernel = kernel_select(params, batch_params.Batch_A,
                                       batch_params.Batch_B, batch_count, M,
                                       N, K, num_threads, bias, is_weights_const);
  static unsigned int auto_version = get_auto_tuner_ver();

  [[maybe_unused]] std::ostringstream ss;
  if (apilog_info_enabled() || is_profile) {
    ss << "LOWOHA matmul_direct: M=" << M << ", N=" << N << ", K=" << K
       << ", alpha=" << alpha << ", beta=" << beta
       << ", lda=" << lda << ", ldb=" << ldb << ", ldc=" << ldc
       << ", transA=" << (transA ? "true" : "false")
       << ", transB=" << (transB ? "true" : "false")
       << ", input_dtype=" << data_type_to_string(params.dtypes.src)
       << ", output_dtype=" << data_type_to_string(params.dtypes.dst)
       << ", bias=" << (bias != nullptr ? "true" : "false")
       << ", is_weights_const=" << (is_weights_const ? "true" : "false")
       << ", post_op=[" << post_op_names_to_string(params) << "]"
    << ", post_op_dtype=[" << ([&]() {
      std::string dtypes = post_op_data_types_to_string(params);
      return dtypes.empty() ? "none" : dtypes;
    })()
        << "]"
        << ", Batch_A=" << batch_params.Batch_A << ", Batch_B=" << batch_params.Batch_B
        << ", plugin_op=" << params.plugin_op;

    if (kernel == matmul_algo_t::auto_tuner) {
      apilog_info(ss.str(), ", kernel=", kernel_to_string(kernel),
                  ", auto_tuner version=", auto_version);
    }
    else {
      apilog_info(ss.str(), ", kernel=", kernel_to_string(kernel));
    }
  }

  // TODO: Add memory unreordering logic
  // Unreorder if onednn/ libxsmm is used
  // Implement the necessary logic for memory reordering here
  // if (params.mem_format_b) {}

  // Dispatch to BMM or Matmul based on batch_count
  matmul_threadlimit thread_guard(num_threads);
  if (batch_count > 1) {
    // Batch Matrix Multiplication (BMM)
    bmm_execute(layout, transA, transB,
                M, N, K, alpha, src, lda, weight, ldb, bias, beta, dst, ldc,
                is_weights_const, batch_params,
                src_type_size, out_type_size, num_threads, kernel, params);
  }
  else {
    // Single Matrix Multiplication (Matmul)
    matmul_execute(layout, transA, transB,
                   M, N, K, alpha, src, lda, weight, ldb, bias, beta, dst, ldc,
                   is_weights_const, src_type_size, out_type_size, num_threads,
                   kernel, params, batch_params, auto_version);
  }

  if (is_profile) {
    profiler.tbp_stop();
    profilelog_verbose(ss.str(), ", kernel=", kernel_to_string(kernel),
                       ", weight_address=", static_cast<const void *>(weight),
                       ", time=", profiler.tbp_elapsedtime(), profiler.get_res_str());
  }

  return status_t::success;
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
