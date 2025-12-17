/*******************************************************************************
# * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "lowoha_operators/matmul/libxsmm_kernel.hpp"
#include "lowoha_operators/matmul/aocl_kernel.hpp"
#include "lowoha_operators/matmul/onednn_kernel.hpp"
#include "lowoha_operators/matmul/auto_tuner.hpp"
#include "lowoha_operators/matmul/brgemm_matmul.hpp"

namespace zendnnl {
namespace lowoha {

using namespace zendnnl::ops;
using zendnnl::common::size_of;

void matmul_kernel_wrapper(char layout, char transA, char transB,
                           int M, int N, int K,
                           float alpha,
                           const void *A, int lda,
                           const void *B, int ldb,
                           float beta,
                           void *C, int ldc,
                           data_types &dtypes,
                           zendnnl::ops::matmul_algo_t kernel,
                           char mem_format_a, char mem_format_b,
                           lowoha_params &lowoha_param, batch_params_t &batch_params,
                           const void *bias, bool is_weights_const, bool can_reorder) {
#if ZENDNNL_DEPENDS_LIBXSMM
  if (kernel == matmul_algo_t::libxsmm) {
    if (run_libxsmm(transA, transB, M, N, K, beta, lda, ldb, ldc, A, B, C, dtypes,
                    lowoha_param, bias)) {
      log_info("Using libxsmm kernel");
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
  log_info("Using BLIS/AOCL kernel");
  run_blis(layout, transA, transB, M, N, K, alpha, beta,
           lda, ldb, ldc, mem_format_a, mem_format_b,
           A, B, C, dtypes, lowoha_param,bias, kernel, is_weights_const, can_reorder);
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
                 batch_params_t &batch_params,
                 const size_t src_type_size, const size_t out_type_size,
                 const int num_threads,
                 matmul_algo_t kernel, lowoha_params &params) {

  const int batch_count = std::max(batch_params.Batch_A, batch_params.Batch_B);
  const char trans_input  = transA ? 't' : 'n';
  const char trans_weight = transB ? 't' : 'n';

  // Calculate batch strides in elements (not bytes)
  size_t src_batch_stride_elems = (batch_params.batch_stride_src !=
                                   static_cast<size_t>(-1))
                                  ? batch_params.batch_stride_src
                                  : (transA ? K * lda : M * lda);
  size_t weight_batch_stride_elems = (batch_params.batch_stride_wei !=
                                      static_cast<size_t>(-1))
                                     ? batch_params.batch_stride_wei
                                     : (transB ? N * ldb : K * ldb);
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
                              params.mem_format_a, params.mem_format_b,
                              src_batch_stride_bytes, weight_batch_stride_bytes, dst_batch_stride_bytes,
                              params, bias);
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
    /*
     * Parallel partitioning strategy:
     * The total number of available threads is divided across batches to compute `threads_per_batch`,
     * ensuring that each batch gets a fair share of compute resources. Within each batch, the M dimension
     * (rows of the output matrix) is further partitioned into blocks of size `M_block`, calculated to
     * evenly distribute the workload among the threads assigned to that batch. The OpenMP `collapse(2)`
     * directive enables parallelization over both batch and row-block loops, while `schedule(dynamic)`
     * ensures better load balancing, especially when M is not divisible evenly or when thread workloads vary.
     */
    int threads_per_batch = std::max(1, num_threads / batch_count);
    int M_block = std::max(1, (M + threads_per_batch - 1) / threads_per_batch);

    // Optimize M_block based on batch count and M size
    // TODO: Further refine the tuning based on heuristics
    // involving batch_count, M, and num_threads.
    if ((batch_count >= 1024 && M <= 2048) ||
        (batch_count >= 512 && M <= 256) ||
        (batch_count > 128 && batch_count < 192 && M <= 512)) {
      if (kernel == matmul_algo_t::libxsmm) {
        M_block = std::min(128, M);
      }
      else {
        M_block = std::min(36, M);
      }
    }
    else if ((batch_count == 64 && M >= 512) ||
             (batch_count == 128 && M >= 512)) {
      M_block = std::min(192, M);
    }
    else {
      M_block = std::min(M_block, M); // Ensure M_block <= M
    }

    if (kernel == matmul_algo_t::libxsmm &&
        !(can_use_libxsmm(trans_input, trans_weight, M_block, N, K, alpha, beta,
                          params.dtypes, params, kernel))) {
      kernel = matmul_algo_t::aocl_blis;
    }

    apilog_info("Executing BMM LOWOHA kernel with parallel partitioning, algo: ",
                static_cast<int>(kernel));
    // Decide parallelization strategy based on MFLOPs:
    // Use zendnn_parallel_for when M_FLOPs > 6.0 for better performance on larger workloads.
    // Use omp_parallel_for when M_FLOPs <= 6.0 to avoid overhead on smaller tasks.
    float flops = static_cast<float>(2LL * M * K * N) / 1000000.0f;

    if (flops > M_FLOPS) {
      apilog_info("Using zendnnl_parallel_for");
      // Calculate total number of work items (batch_count * number of M blocks)
      int total_m_blocks = (M + M_block - 1) / M_block;
      int total_work_items = batch_count * total_m_blocks;

      zendnnl_parallel_for(0, total_work_items, 1, [&](int start_idx, int end_idx) {
        for (int work_idx = start_idx; work_idx < end_idx; ++work_idx) {
          // Convert linear work index back to (batch, m_block) coordinates
          int b = work_idx / total_m_blocks;
          int m_block_idx = work_idx % total_m_blocks;
          int m_start = m_block_idx * M_block;
          int m_len = std::min(M_block, M - m_start);

          const uint8_t *src_ptr    = static_cast<const uint8_t *>(src) +
                                      get_batch_index(b, batch_params.Batch_A) * src_batch_stride_bytes;
          const uint8_t *weight_ptr = static_cast<const uint8_t *>(weight) +
                                      get_batch_index(b, batch_params.Batch_B) * weight_batch_stride_bytes;
          uint8_t *dst_ptr          = static_cast<uint8_t *>(dst) + b *
                                      dst_batch_stride_bytes;

          const void *A = get_matrix_block(src_ptr, m_start, 0, lda, transA,
                                           src_type_size);
          void *C       = get_output_block(dst_ptr, m_start, 0, ldc, out_type_size);

          // Create a modified post_op with offset binary tensor buffers
          lowoha_params thread_lowoha_params = params;
          for (auto &po : thread_lowoha_params.postop_) {
            if (po.po_type == post_op_type_t::binary_add ||
                po.po_type == post_op_type_t::binary_mul) {
              if (po.buff != nullptr) {
                // Calculate offset based on m_start and data type
                size_t element_size = size_of(po.dtype);
                size_t row_offset = m_start * N * element_size;
                po.buff = static_cast<uint8_t *>(const_cast<void *>(po.buff)) + row_offset;
              }
            }
          }

          matmul_kernel_wrapper(layout, trans_input, trans_weight,
                                m_len, N, K, alpha,
                                A, lda, weight_ptr, ldb,
                                beta, C, ldc,
                                params.dtypes, kernel,
                                params.mem_format_a, params.mem_format_b, thread_lowoha_params, batch_params,
                                bias, is_weights_const);
        }
      });
    }
    else {
      apilog_info("Using OpenMP parallel for");
      #pragma omp parallel for collapse(2)
      for (int b = 0; b < batch_count; ++b) {
        for (int m_start = 0; m_start < M; m_start += M_block) {
          int m_len = std::min(M_block, M - m_start);

          const uint8_t *src_ptr = static_cast<const uint8_t *>(src) +
                                   get_batch_index(b, batch_params.Batch_A) * src_batch_stride_bytes;
          const uint8_t *weight_ptr = static_cast<const uint8_t *>(weight) +
                                      get_batch_index(b, batch_params.Batch_B) * weight_batch_stride_bytes;
          uint8_t *dst_ptr = static_cast<uint8_t *>(dst) + b * dst_batch_stride_bytes;

          const void *A = get_matrix_block(src_ptr, m_start, 0, lda, transA,
                                           src_type_size);
          void *C = get_output_block(dst_ptr, m_start, 0, ldc, out_type_size);

          // Create a modified post_op with offset binary tensor buffers
          lowoha_params thread_lowoha_params = params;
          for (auto &po : thread_lowoha_params.postop_) {
            if (po.po_type == post_op_type_t::binary_add ||
                po.po_type == post_op_type_t::binary_mul) {
              if (po.buff != nullptr) {
                // Calculate offset based on m_start and data type
                size_t element_size = size_of(po.dtype);
                size_t row_offset = m_start * N * element_size;
                po.buff = static_cast<uint8_t *>(const_cast<void *>(po.buff)) + row_offset;
              }
            }
          }

          matmul_kernel_wrapper(layout, trans_input, trans_weight,
                                m_len, N, K, alpha,
                                A, lda, weight_ptr, ldb,
                                beta, C, ldc,
                                params.dtypes, kernel,
                                params.mem_format_a, params.mem_format_b, thread_lowoha_params, batch_params,
                                bias, is_weights_const);
        }
      }
    }
  }
  else {
    // Single thread execution for batches
    if (kernel == matmul_algo_t::libxsmm &&
        !(can_use_libxsmm(trans_input, trans_weight, M, N, K, alpha, beta,
                          params.dtypes, params, kernel))) {
      kernel = matmul_algo_t::aocl_blis;
    }

    apilog_info("Executing BMM LOWOHA kernel without zendnnl-partitioner, algo: ",
                static_cast<int>(kernel));
    for (int b = 0; b < batch_count; ++b) {
      const uint8_t *src_ptr = static_cast<const uint8_t *>(src) +
                               get_batch_index(b, batch_params.Batch_A) * src_batch_stride_bytes;
      const uint8_t *weight_ptr = static_cast<const uint8_t *>(weight) +
                                  get_batch_index(b, batch_params.Batch_B) * weight_batch_stride_bytes;
      uint8_t *dst_ptr = static_cast<uint8_t *>(dst) + b * dst_batch_stride_bytes;

      matmul_kernel_wrapper(layout, trans_input, trans_weight,
                            M, N, K, alpha,
                            src_ptr, lda, weight_ptr,
                            ldb, beta, dst_ptr, ldc,
                            params.dtypes, kernel,
                            params.mem_format_a, params.mem_format_b, params, batch_params,
                            bias, is_weights_const, false);
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
                    matmul_algo_t kernel, lowoha_params &params,
                    batch_params_t &batch_params,
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

  // LIBXSMM blocked execution with tiling
  if (kernel == matmul_algo_t::libxsmm_blocked) {
    apilog_info("Executing matmul LOWOHA kernel with libxsmm tiling, algo: ",
                static_cast<int>(kernel));
#if ENABLE_BRGEMM_KERNEL
    auto [tileM, tileN] = selectTileBF16(M, N, K, num_threads);
    int M_BLOCK = get_tile_size_from_env("ZENDNN_MATMUL_M_TILE", tileM);
    int N_BLOCK = get_tile_size_from_env("ZENDNN_MATMUL_N_TILE", tileN);
    int KC_BLOCK = 64;

    const uint8_t *src_ptr = static_cast<const uint8_t *>(src);
    const uint8_t *weight_ptr = static_cast<const uint8_t *>(weight);
    uint8_t *dst_ptr = static_cast<uint8_t *>(dst);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i += M_BLOCK) {
      for (int j = 0; j < N; j += N_BLOCK) {
        int m_tile = std::min(M_BLOCK, M - i);
        int n_tile = std::min(N_BLOCK, N - j);

        // Prepare batch arrays
        std::vector<const void *> A_batch;
        std::vector<const void *> B_batch;
        std::vector<int> K_sizes;

        for (int k = 0; k < K; k += KC_BLOCK) {
          int k_tile = std::min(KC_BLOCK, K - k);
          const void *A_tile = get_matrix_block(src_ptr, i, k, lda, transA,
                                                src_type_size);
          const void *B_tile = get_matrix_block(weight_ptr, k, j, ldb, transB,
                                                src_type_size);
          A_batch.push_back(A_tile);
          B_batch.push_back(B_tile);
          K_sizes.push_back(k_tile);
        }

        void *C_tile = get_output_block(dst_ptr, i, j, ldc, out_type_size);

        // Offset bias pointer for the current N tile
        const void *tile_bias = nullptr;
        if (bias != nullptr) {
          size_t bias_element_size = size_of(params.dtypes.bias);
          size_t bias_offset_bytes = static_cast<size_t>(j) * bias_element_size;
          tile_bias = static_cast<const uint8_t *>(bias) + bias_offset_bytes;
        }

        brgemm_ref_kernel(transA, transB,
                          m_tile, n_tile, K_sizes.data(),
                          A_batch.data(), B_batch.data(),
                          A_batch.size(),
                          alpha, beta, C_tile, ldc,
                          lda, ldb, params.dtypes, tile_bias);
      }
    }
#else
    if ((can_use_libxsmm(trans_input, trans_weight, M, N, K, alpha,
                         beta, params.dtypes, params, kernel))) {
      auto [tileM, tileN] = selectTileBF16(M, N, K, num_threads);
      int M_BLOCK = get_tile_size_from_env("ZENDNN_MATMUL_M_TILE", tileM);
      int N_BLOCK = get_tile_size_from_env("ZENDNN_MATMUL_N_TILE", tileN);
      const uint8_t *src_ptr = static_cast<const uint8_t *>(src);
      const uint8_t *weight_ptr = static_cast<const uint8_t *>(weight);
      uint8_t *dst_ptr = static_cast<uint8_t *>(dst);
      matmul_algo_t tile_kernel = matmul_algo_t::libxsmm;

      #pragma omp parallel for collapse(2)
      for (int i = 0; i < M; i += M_BLOCK) {
        for (int j = 0; j < N; j += N_BLOCK) {
          int m_tile = std::min(M_BLOCK, M - i);
          int n_tile = std::min(N_BLOCK, N - j);

          const void *A_tile = get_matrix_block(src_ptr, i, 0, lda, transA,
                                                src_type_size);
          const void *B_tile = get_matrix_block(weight_ptr, 0, j, ldb, transB,
                                                src_type_size);
          void *C_tile = get_output_block(dst_ptr, i, j, ldc, out_type_size);

          float tile_alpha = alpha;
          float tile_beta = beta;

          lowoha_params tile_params = params;
          for (auto &po : tile_params.postop_) {
            if ((po.po_type == post_op_type_t::binary_add ||
                 po.po_type == post_op_type_t::binary_mul) &&
                po.buff != nullptr) {

              size_t element_size = size_of(po.dtype);
              size_t offset_elems = static_cast<size_t>(i) * static_cast<size_t>
                                    (po.leading_dim) + static_cast<size_t>(j);

              size_t offset_bytes = offset_elems * element_size;

              po.buff = static_cast<uint8_t *>(const_cast<void *>(po.buff)) + offset_bytes;
            }
          }
          const void *tile_bias = nullptr;
          if (bias != nullptr) {
            size_t bias_element_size = size_of(params.dtypes.bias);
            size_t bias_offset_bytes = static_cast<size_t>(j) * bias_element_size;
            tile_bias = static_cast<const uint8_t *>(bias) + bias_offset_bytes;
          }

          matmul_kernel_wrapper(layout, trans_input, trans_weight,
                                m_tile, n_tile, K, tile_alpha,
                                A_tile, lda,
                                B_tile,
                                ldb, tile_beta, C_tile, ldc,
                                tile_params.dtypes, tile_kernel,
                                tile_params.mem_format_a, tile_params.mem_format_b,
                                tile_params, batch_params, tile_bias, is_weights_const);
        }
      }
    }
    else {
      matmul_kernel_wrapper(layout, trans_input, trans_weight,
                            M, N, K, alpha,
                            src, lda,
                            weight,
                            ldb, beta, dst, ldc,
                            params.dtypes,  matmul_algo_t::aocl_blis,
                            params.mem_format_a, params.mem_format_b,
                            params, batch_params, bias, is_weights_const);
    }
#endif
    return;
  }

  // Standard single matmul execution
  if (kernel == matmul_algo_t::libxsmm &&
      !(can_use_libxsmm(trans_input, trans_weight, M, N, K, alpha, beta,
                        params.dtypes, params, kernel))) {
    kernel = matmul_algo_t::aocl_blis;
  }

  apilog_info("Executing matmul LOWOHA kernel without zendnnl-partitioner, algo: ",
              static_cast<int>(kernel));
  matmul_kernel_wrapper(layout, trans_input, trans_weight,
                        M, N, K, alpha,
                        src, lda, weight,
                        ldb, beta, dst, ldc,
                        params.dtypes, kernel,
                        params.mem_format_a, params.mem_format_b, params,
                        batch_params, bias, is_weights_const, true);
}

status_t matmul_direct(const char layout, const bool transA, const bool transB,
                       const int M, const int N, const int K, const float alpha, const void *src,
                       const int lda, const void *weight, const int ldb, const void *bias,
                       const float beta, void *dst, const int ldc, const bool is_weights_const,
                       batch_params_t batch_params, lowoha_params params) {
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
        << ", Batch_A=" << batch_params.Batch_A << ", Batch_B=" << batch_params.Batch_B;

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

} // namespace lowoha
} // namespace zendnnl
