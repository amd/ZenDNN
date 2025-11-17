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

namespace zendnnl {
namespace lowoha {

using namespace zendnnl::ops;

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
                           const lowoha_params &lowoha_param, const void *bias) {
#if ZENDNNL_DEPENDS_LIBXSMM
  if (kernel == matmul_algo_t::libxsmm) {
    if (run_libxsmm(transA,transB,M,N,K,beta,lda,ldb,ldc,A,B,C,dtypes)) {
      log_info("Using libxsmm kernel");
      return;
    }
  }
#endif

  log_info("Using BLIS/AOCL kernel");
  run_blis(layout,transA,transB,M,N,K,alpha,beta,
           lda,ldb,ldc,mem_format_a,mem_format_b,
           A,B,C,dtypes,lowoha_param,bias);
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

status_t matmul_direct(const char layout,const bool transA,const bool transB,
                       const int M, const int N, const int K,const float alpha, const void *src,
                       const int lda, const void *weight,const int ldb,const void *bias,
                       const float beta, void *dst, const int ldc, const bool is_weights_const,
                       batch_params_t batch_params, lowoha_params params) {
  // Create profiler instance for timing
  profiler_t profiler;
  bool is_profile = is_profile_enabled();
  if (is_profile) {
    profiler.tbp_start();
  }

  int Batch_A = batch_params.Batch_A;
  int Batch_B = batch_params.Batch_B;


  if (validate_matmul_direct_inputs(src, weight, dst, M, N, K, Batch_A, Batch_B,
                                    params) != status_t::success) {
    return status_t::failure;
  }

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
    })() << "]"
         << ", Batch_A=" << Batch_A << ", Batch_B=" << Batch_B;

    apilog_info(ss.str());
  }

  const bool is_f32_src  = (params.dtypes.src == data_type_t::f32);
  const bool is_f32_out  = (params.dtypes.dst == data_type_t::f32);

  const char trans_input  = transA ? 't' : 'n';
  const char trans_weight = transB ? 't' : 'n';
  char mem_format_b = params.mem_format_b;

  size_t src_type_size = is_f32_src ? sizeof(float) : sizeof(int16_t);
  size_t out_type_size = is_f32_out ? sizeof(float) : sizeof(int16_t);

  size_t src_batch_stride = batch_params.batch_stride_src ==
                            static_cast<size_t>(-1) ? (transA ? K *lda : M * lda) * src_type_size :
                            batch_params.batch_stride_src * src_type_size;
  size_t weight_batch_stride = batch_params.batch_stride_wei ==
                               static_cast<size_t>(-1) ? (transB ? N *ldb : K * ldb) * src_type_size :
                               batch_params.batch_stride_wei * src_type_size;
  size_t dst_batch_stride = batch_params.batch_stride_dst ==
                            static_cast<size_t>(-1) ? M * ldc * out_type_size :
                            batch_params.batch_stride_dst * out_type_size;

  const int batch_count = std::max(Batch_A, Batch_B);
  const int num_threads = params.num_threads > 0 ? params.num_threads : omp_get_max_threads();
  // const bool use_blis_partitioning = may_i_use_blis_partition(batch_count, M, N,
  //                                    num_threads, params.dtypes.src);
  matmul_algo_t kernel = kernel_select(params, Batch_A, Batch_B, batch_count, M,
                                       N, K, num_threads, bias);
  // TODO: Add memory unreordering logic
  // Unreorder if onednn/ libxsmm is used
  // Implement the necessary logic for memory reordering here
  //if (params.mem_format_b) {}

  bool is_weight_blocked = false;
  void *reordered_mem = nullptr;
  matmul_config_t &matmul_config = matmul_config_t::instance();
  [[maybe_unused]] int32_t weight_cache_type = matmul_config.get_weight_cache();
  // AOCL blocked kernel reordering for 2D MatMul
  if (kernel==zendnnl::ops::matmul_algo_t::aocl_blis_blocked &&
      batch_count == 1 && is_weights_const) {
    Key_matmul key_(transB, K, N, ldb, weight,
                    static_cast<uint32_t>(matmul_algo_t::aocl_blis_blocked));
    //call reorder and cache function
    char trans = transB ? 't' : 'n';
    bool blocked_flag = false;
    if (params.dtypes.wei == data_type_t::f32) {
      blocked_flag = reorderAndCacheWeights<float>(key_, weight, reordered_mem, K, N,
                     ldb,
                     'r', trans, mem_format_b,
                     aocl_get_reorder_buf_size_f32f32f32of32, aocl_reorder_f32f32f32of32,
                     weight_cache_type);
    }
    else if (params.dtypes.wei == data_type_t::bf16) {
      blocked_flag = reorderAndCacheWeights<int16_t>(key_, weight, reordered_mem, K,
                     N, ldb,
                     'r', trans, mem_format_b,
                     aocl_get_reorder_buf_size_bf16bf16f32of32, aocl_reorder_bf16bf16f32of32,
                     weight_cache_type);
    }
    if (blocked_flag) {
      is_weight_blocked = true;
      mem_format_b = 'r';
    }
  }

  if (kernel == matmul_algo_t::batched_sgemm && batch_count > 1) {
    // Use batch GEMM for multiple batches
    apilog_info("Executing matmul LOWOHA kernel with batch GEMM, algo: ",
                static_cast<int>(kernel));
    matmul_batch_gemm_wrapper(layout, trans_input, trans_weight,
                              M, N, K, alpha,
                              src, lda, weight, ldb, beta, dst, ldc,
                              params.dtypes, batch_count,
                              params.mem_format_a, params.mem_format_b,
                              src_batch_stride, weight_batch_stride, dst_batch_stride,
                              params, bias);
  }
#if ZENDNNL_DEPENDS_ONEDNN
  else if (kernel == matmul_algo_t::onednn ||
           kernel == matmul_algo_t::onednn_blocked) {
    apilog_info("Executing matmul LOWOHA kernel with oneDNN, algo: ",
                static_cast<int>(kernel));
    matmul_onednn_wrapper(trans_input, trans_weight, M, N, K, alpha, src, lda,
                          weight, ldb, beta, dst, ldc, params, Batch_A, Batch_B, bias, weight_cache_type, is_weights_const);
  }
#endif
  else if (num_threads > 1 && batch_count > 1) {
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
      if (kernel==matmul_algo_t::libxsmm) {
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
      M_block = std::min(M_block, M);  // Ensure M_block <= M
    }

    if (kernel == matmul_algo_t::libxsmm &&
        !(can_use_libxsmm(trans_input,trans_weight,M_block,N,K,alpha,beta,
                          params.dtypes))) {
      kernel = matmul_algo_t::aocl_blis;
    }

    apilog_info("Executing matmul LOWOHA kernel with parallel partitioning, algo: ",
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
                                      get_batch_index(b, Batch_A) * src_batch_stride;
          const uint8_t *weight_ptr = static_cast<const uint8_t *>(weight) +
                                      get_batch_index(b, Batch_B) * weight_batch_stride;
          uint8_t *dst_ptr          = static_cast<uint8_t *>(dst) + b * dst_batch_stride;

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
                size_t element_size = (po.dtype == data_type_t::f32) ? sizeof(float) : sizeof(
                                        uint16_t);
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
                                params.mem_format_a, params.mem_format_b, thread_lowoha_params, bias);
        }
      });
    }
    else {
      apilog_info("Using OpenMP parallel for");
      #pragma omp parallel for collapse(2)
      for (int b = 0; b < batch_count; ++b) {
        for (int m_start = 0; m_start < M; m_start += M_block) {
          int m_len = std::min(M_block, M - m_start);

          const uint8_t *src_ptr    = static_cast<const uint8_t *>(src) +
                                      get_batch_index(b, Batch_A) * src_batch_stride;
          const uint8_t *weight_ptr = static_cast<const uint8_t *>(weight) +
                                      get_batch_index(b, Batch_B) * weight_batch_stride;
          uint8_t *dst_ptr          = static_cast<uint8_t *>(dst) + b * dst_batch_stride;

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
                size_t element_size = (po.dtype == data_type_t::f32) ? sizeof(float) : sizeof(
                                        uint16_t);
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
                                params.mem_format_a, params.mem_format_b, thread_lowoha_params, bias);
        }
      }
    }
  }
  else if (kernel==matmul_algo_t::libxsmm_blocked && batch_count==1) {
#if ENABLE_K_TILE_OPTIMIZATION
    constexpr int M_BLOCK = 64;
    constexpr int N_BLOCK = 64;
    constexpr int K_BLOCK = 64;

    const uint8_t *src_ptr = static_cast<const uint8_t *>(src);
    const uint8_t *weight_ptr = static_cast<const uint8_t *>(weight);
    uint8_t *dst_ptr = static_cast<uint8_t *>(dst);
    matmul_algo_t tile_kernel = matmul_algo_t::aocl_blis; // default fallback

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i += M_BLOCK) {
      for (int j = 0; j < N; j += N_BLOCK) {
        int m_tile = std::min(M_BLOCK, M - i);
        int n_tile = std::min(N_BLOCK, N - j);

        void *C_tile = get_output_block(dst_ptr, i, j, ldc, out_type_size);

        for (int k = 0; k < K; k += K_BLOCK) {
          int k_tile = std::min(K_BLOCK, K - k);
          bool is_first_k = (k == 0);

          const void *A_tile = get_matrix_block(src_ptr, i, k, lda, transA,
                                                src_type_size);
          const void *B_tile = get_matrix_block(weight_ptr, k, j, ldb, transB,
                                                src_type_size);

          float tile_alpha = alpha;
          float tile_beta = is_first_k ? beta : 1.0f;

          // Use libxsmm only for perfect tiles
          if (m_tile == M_BLOCK && n_tile == N_BLOCK && k_tile == K_BLOCK &&
              (can_use_libxsmm(trans_input,trans_weight,m_tile,n_tile,k_tile,tile_alpha,
                               tile_beta,params.dtypes))) {
            tile_kernel = matmul_algo_t::libxsmm;
          }
          else {
            // For irregular tiles (including K tail), always use BLIS
            tile_kernel = matmul_algo_t::aocl_blis;
          }

          matmul_kernel_wrapper(layout, trans_input, trans_weight,
                                m_tile, n_tile, k_tile, tile_alpha,
                                A_tile, lda,
                                is_weight_blocked ? reordered_mem : B_tile,
                                ldb, tile_beta, C_tile, ldc,
                                params.dtypes, tile_kernel,
                                params.mem_format_a, mem_format_b,
                                params, (k == 0) ? bias : nullptr);
        }
      }
    }
#else
    constexpr int M_BLOCK = 64;
    constexpr int N_BLOCK = 64;

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
        if (!(can_use_libxsmm(trans_input,trans_weight,m_tile,n_tile,K,tile_alpha,
                              tile_beta,params.dtypes))) {
          tile_kernel = matmul_algo_t::aocl_blis;
        }
        matmul_kernel_wrapper(layout, trans_input, trans_weight,
                              m_tile, n_tile, K, tile_alpha,
                              A_tile, lda,
                              is_weight_blocked ? reordered_mem : B_tile,
                              ldb, tile_beta, C_tile, ldc,
                              params.dtypes, tile_kernel,
                              params.mem_format_a, mem_format_b,
                              params, bias);
      }
    }
#endif
    apilog_info("Executing matmul LOWOHA kernel with libxsmm tiling, algo: ",
                static_cast<int>(kernel));
  }
  else {
    if (kernel == matmul_algo_t::libxsmm &&
        !(can_use_libxsmm(trans_input,trans_weight,M,N,K,alpha,beta,params.dtypes))) {
      kernel = matmul_algo_t::aocl_blis;
    }

    apilog_info("Executing matmul LOWOHA kernel without zendnnl-partitioner, algo: ",
                static_cast<int>(kernel));
    for (int b = 0; b < batch_count; ++b) {
      const uint8_t *src_ptr    = static_cast<const uint8_t *>(src) +
                                  get_batch_index(b, Batch_A) * src_batch_stride;
      const uint8_t *weight_ptr = static_cast<const uint8_t *>(weight) +
                                  get_batch_index(b, Batch_B) * weight_batch_stride;
      uint8_t *dst_ptr          = static_cast<uint8_t *>(dst) + b * dst_batch_stride;

      matmul_kernel_wrapper(layout, trans_input, trans_weight,
                            M, N, K, alpha,
                            src_ptr, lda, is_weight_blocked ? reordered_mem : weight_ptr,
                            ldb, beta, dst_ptr, ldc,
                            params.dtypes, kernel,
                            params.mem_format_a, mem_format_b, params, bias);
    }
    // Free reordered buffer for AOCL blocked non-cached
    bool free_buff = (weight_cache_type == 0 && reordered_mem != nullptr &&
                      params.mem_format_b != 'r'
                      && kernel==zendnnl::ops::matmul_algo_t::aocl_blis_blocked && batch_count == 1);
    if (free_buff)  {
      free(reordered_mem);
      reordered_mem = nullptr;
    }
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

