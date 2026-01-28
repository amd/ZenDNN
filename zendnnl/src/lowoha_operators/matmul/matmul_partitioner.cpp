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

#include "matmul_partitioner.hpp"

#include <algorithm>
#include <cstdlib>
#include <omp.h>

#include "lowoha_matmul.hpp"
#include "lowoha_matmul_utils.hpp"
#include "libxsmm_utils.hpp"
#include "lowoha_operators/matmul/onednn_kernel.hpp"
#include "libxsmm_kernel.hpp"
namespace zendnnl {
namespace lowoha {
namespace matmul {


std::pair<int, int> get_tile_sizes_from_config(int default_m, int default_n) {
  matmul_config_t &matmul_config = matmul_config_t::instance();

  int tile_m = matmul_config.get_tile_m();
  int tile_n = matmul_config.get_tile_n();

  if (tile_m <= 0) {
    tile_m = default_m;
  }
  if (tile_n <= 0) {
    tile_n = default_n;
  }

  return {tile_m, tile_n};
}

// This function selects tile sizes for BF16 matmul based on heuristics
// TODO: Further tune the heuristics based on num_threads
std::tuple<int, int> select_tile(int M, int N, int K, int num_threads) {
  if (M <= 2048 && N <= 128) {
    return {32, 32};
  }

  if (M <= 4096 && N > 768 && N <= 1024) {
    return {64, 64};
  }

  return {128, 64};
}

std::pair<int, int> calculate_optimal_tile_sizes(
  const matmul_partition_config_t &config
) {
  auto [M_block, N_block] = select_tile(config.M, config.N, config.K,
                                        config.num_threads);

  // Get tile sizes from config (which reads from env vars)
  auto [tile_m, tile_n] = get_tile_sizes_from_config(M_block, N_block);

  M_block = std::min(tile_m, config.M);
  N_block = std::min(tile_n, config.N);

  return {M_block, N_block};
}

bool should_use_kc_blocking(
  const matmul_partition_config_t &config,
  const matmul_params &params
) {
#if ENABLE_LIBXSMM_BRGEMM_KERNEL
  // TODO: Need to support bf16 brgemm kernels later
  if (config.dtypes.src != data_type_t::f32 ||
      config.kernel != matmul_algo_t::libxsmm_blocked) {
    apilog_info("Only F32 BRGEMM Supported, Executing matmul LOWOHA kernel with fallback Algo");
    return false;
  }
  return true;
#else
  return false;
#endif
}

size_t compute_postop_offset(
  int row_start,
  int col_start,
  int leading_dim,
  data_type_t dtype
) {
  size_t element_size = size_of(dtype);
  return (static_cast<size_t>(row_start) * leading_dim + col_start) *
         element_size;
}

void *apply_offset(void *buffer, size_t offset) {
  if (buffer == nullptr) {
    return nullptr;
  }
  return static_cast<uint8_t *>(buffer) + offset;
}

const void *compute_bias_offset(
  const void *bias,
  int col_start,
  data_type_t bias_dtype
) {
  if (bias == nullptr) {
    return nullptr;
  }
  size_t bias_element_size = size_of(bias_dtype);
  return static_cast<const uint8_t *>(bias) +
         (static_cast<size_t>(col_start) * bias_element_size);
}


static matmul_algo_t select_partition_kernel(
  const char trans_input,
  const char trans_weight,
  float alpha, float beta,
  const matmul_partition_config_t &config,
  const matmul_params &params
) {
  if (config.kernel == matmul_algo_t::libxsmm_blocked) {
    matmul_algo_t selected_kernel = config.kernel;
    if (can_use_libxsmm(trans_input, trans_weight, config.M, config.N, config.K,
                        alpha, beta,
                        params.dtypes, params, selected_kernel)) {
      return matmul_algo_t::libxsmm;
    }
    else {
      apilog_info("LibXSMM kernel cannot be used for current configuration, falling back to DLP");
      return matmul_algo_t::aocl_dlp;

    }

  }
  else {
    return config.kernel;
  }

}

static tile_kernel_invoker_t create_tile_callback(
  const char layout,
  const char trans_input,
  const char trans_weight,
  const matmul_partition_config_t &config,
  matmul_params &params,
  matmul_batch_params_t &batch_params,
  bool is_weights_const,
  float alpha,
  float beta
) {
  return [ &, layout, trans_input, trans_weight, alpha, beta, is_weights_const](
           int m_start, int m_len,
           int n_start, int n_len,
           const void *A_tile,
           const void *B_tile,
           void *C_tile,
           const void *tile_bias
  ) {
    // Adjust post-op buffers for this tile
    matmul_params tile_params = params;
    for (auto &po : tile_params.postop_) {
      if ((po.po_type == post_op_type_t::binary_add ||
           po.po_type == post_op_type_t::binary_mul) &&
          po.buff != nullptr) {

        size_t offset = compute_postop_offset(
                          m_start, n_start, po.leading_dim, po.dtype
                        );
        po.buff = apply_offset(const_cast<void *>(po.buff), offset);
      }
    }

    // Call standard kernel wrapper
    matmul_kernel_wrapper(
      layout, trans_input, trans_weight,
      m_len, n_len, config.K, alpha,
      A_tile, config.lda,
      B_tile, config.ldb,
      beta, C_tile, config.ldc,
      tile_params.dtypes, config.kernel,
      tile_params.mem_format_a, tile_params.mem_format_b,
      tile_params, batch_params, tile_bias, is_weights_const
    );
  };
}

static brgemm_kernel_invoker_t create_brgemm_callback(
  const char layout,
  const char trans_input,
  const char trans_weight,
  const matmul_partition_config_t &config,
  matmul_params &params,
  matmul_batch_params_t &batch_params,
  float beta
) {
  return [ &, layout, trans_input, trans_weight, beta](
           int m_start, int m_len,
           int n_start, int n_len,
           const void **A_batch_main,
           const void **B_batch_main,
           const void *A_batch_tail,
           const void *B_batch_tail,
           void *C_tile,
           const void *tile_bias,
           int num_main_blocks,
           int KC_BLOCK,
           int K_tail
  ) {
#if ENABLE_LIBXSMM_BRGEMM_KERNEL
    // Adjust post-op buffers for this tile
    matmul_params tile_params = params;
    for (auto &po : tile_params.postop_) {
      if ((po.po_type == post_op_type_t::binary_add ||
           po.po_type == post_op_type_t::binary_mul) &&
          po.buff != nullptr) {

        size_t offset = compute_postop_offset(
                          m_start, n_start, po.leading_dim, po.dtype
                        );
        po.buff = apply_offset(const_cast<void *>(po.buff), offset);
      }
    }

    // Process main K blocks
    if (num_main_blocks > 0) {
      run_libxsmm_brgemm(
        trans_input, trans_weight,
        m_len, n_len, KC_BLOCK,
        num_main_blocks,
        beta,
        config.lda, config.ldb, config.ldc,
        A_batch_main,
        B_batch_main,
        C_tile,
        config.dtypes,
        tile_params,
        tile_bias,
        (K_tail == 0)  // apply_postops if no tail
      );
    }

    // Process tail K block
    if (K_tail > 0) {
      const void *A_tail_array[1] = {A_batch_tail};
      const void *B_tail_array[1] = {B_batch_tail};

      float tail_beta = (num_main_blocks > 0) ? 1.0f : beta;

      run_libxsmm_brgemm(
        trans_input, trans_weight,
        m_len, n_len, K_tail,
        1,  // single tail block
        tail_beta,
        config.lda, config.ldb, config.ldc,
        A_tail_array,
        B_tail_array,
        C_tile,
        config.dtypes,
        tile_params,
        (num_main_blocks == 0 ? tile_bias : nullptr),
        true  // apply_postops
      );
    }
#endif
  };
}

void execute_partitioned_matmul_libxsmm_brgemm(
  const void *src,
  const void *weight,
  void *dst,
  const void *bias,
  const matmul_partition_config_t &config,
  const brgemm_kernel_invoker_t &brgemm_callback
) {
  auto [M_block, N_block] = calculate_optimal_tile_sizes(config);

  constexpr int DEFAULT_KC_BLOCK = 64;
  const int K_main = (config.K / DEFAULT_KC_BLOCK) * DEFAULT_KC_BLOCK;
  const int K_tail = config.K - K_main;
  // Handle case where K < KC_BLOCK (K_main would be 0)
  const int num_main_blocks = (K_main > 0) ? (K_main / DEFAULT_KC_BLOCK) : 0;

  const uint8_t *src_ptr = static_cast<const uint8_t *>(src);
  const uint8_t *weight_ptr = static_cast<const uint8_t *>(weight);
  uint8_t *dst_ptr = static_cast<uint8_t *>(dst);

  #pragma omp parallel num_threads(config.num_threads)
  {
    //TODO: Per thread malloc need to be optimized
    std::vector<const void *> A_batch_main;
    std::vector<const void *> B_batch_main;

    if (num_main_blocks > 0) {
      A_batch_main.resize(num_main_blocks);
      B_batch_main.resize(num_main_blocks);
    }

    #pragma omp for collapse(2) schedule(static)
    for (int i = 0; i < config.M; i += M_block) {
      for (int j = 0; j < config.N; j += N_block) {

        int m_len = std::min(M_block, config.M - i);
        int n_len = std::min(N_block, config.N - j);

        void *C_tile = get_output_block(
                         dst_ptr, i, j, config.ldc, config.out_type_size
                       );

        const void *tile_bias = compute_bias_offset(
                                  bias, j, config.dtypes.bias
                                );

        for (int kb = 0; kb < num_main_blocks; ++kb) {
          int k_offset = kb * DEFAULT_KC_BLOCK;

          A_batch_main[kb] = get_matrix_block(
                               src_ptr, i, k_offset, config.lda,
                               config.transA, config.src_type_size
                             );

          B_batch_main[kb] = get_matrix_block(
                               weight_ptr, k_offset, j, config.ldb,
                               config.transB, config.src_type_size
                             );
        }

        const void *A_tail = (K_tail > 0) ?
                             get_matrix_block(src_ptr, i, K_main, config.lda,
                                              config.transA, config.src_type_size) : nullptr;

        const void *B_tail = (K_tail > 0) ?
                             get_matrix_block(weight_ptr, K_main, j, config.ldb,
                                              config.transB, config.src_type_size) : nullptr;
        //TODO: cache the kernel and avoid re-dispatch
        brgemm_callback(
          i, m_len, j, n_len,
          A_batch_main.data(), B_batch_main.data(),
          A_tail, B_tail,
          C_tile, tile_bias,
          num_main_blocks, DEFAULT_KC_BLOCK, K_tail
        );
      }
    }
  }
}

void execute_partitioned_matmul_standard(
  const void *src,
  const void *weight,
  void *dst,
  const void *bias,
  const matmul_partition_config_t &config,
  const tile_kernel_invoker_t &tile_callback
) {
  auto [M_block, N_block] = calculate_optimal_tile_sizes(config);

  const uint8_t *src_ptr = static_cast<const uint8_t *>(src);
  const uint8_t *weight_ptr = static_cast<const uint8_t *>(weight);
  uint8_t *dst_ptr = static_cast<uint8_t *>(dst);

  #pragma omp parallel for collapse(2) schedule(static) num_threads(config.num_threads)
  for (int i = 0; i < config.M; i += M_block) {
    for (int j = 0; j < config.N; j += N_block) {

      int m_len = std::min(M_block, config.M - i);
      int n_len = std::min(N_block, config.N - j);

      const void *A_tile = get_matrix_block(
                             src_ptr, i, 0, config.lda,
                             config.transA, config.src_type_size
                           );

      const void *B_tile = get_matrix_block(
                             weight_ptr, 0, j, config.ldb,
                             config.transB, config.src_type_size
                           );

      void *C_tile = get_output_block(
                       dst_ptr, i, j, config.ldc, config.out_type_size
                     );

      const void *tile_bias = compute_bias_offset(
                                bias, j, config.dtypes.bias
                              );

      tile_callback(
        i, m_len, j, n_len,
        A_tile, B_tile, C_tile, tile_bias
      );
    }
  }
}

void execute_partitioned_matmul(
  const char layout,
  const char trans_input,
  const char trans_weight,
  const void *src,
  const void *weight,
  void *dst,
  const void *bias,
  matmul_partition_config_t config,
  matmul_params &params,
  matmul_batch_params_t &batch_params,
  bool is_weights_const,
  float alpha,
  float beta
) {


  bool use_libxsmm_brgemm = should_use_kc_blocking(config,
                            params);

  config.kernel = select_partition_kernel(
                    trans_input, trans_weight,
                    alpha, beta, config, params
                  );
  if (config.kernel == matmul_algo_t::libxsmm) {

    if (use_libxsmm_brgemm) {

      auto brgemm_callback = create_brgemm_callback(
                               layout, trans_input, trans_weight,
                               config, params, batch_params, beta
                             );

      execute_partitioned_matmul_libxsmm_brgemm(
        src, weight, dst, bias, config, brgemm_callback
      );
    }
    else {
      auto tile_callback = create_tile_callback(
                             layout, trans_input, trans_weight,
                             config, params, batch_params,
                             is_weights_const, alpha, beta
                           );

      execute_partitioned_matmul_standard(
        src, weight, dst, bias, config, tile_callback
      );
    }
  }
  else {
    // Fallback to OneDNN or DLP if libxsmm can't be used
#if ZENDNNL_DEPENDS_ONEDNN
    if (config.kernel == matmul_algo_t::onednn ||
        config.kernel == matmul_algo_t::onednn_blocked) {
      apilog_info("Given combination is not supported for matmul parallel primitive, executing matmul LOWOHA kernel with OneDNN fallback, algo: ",
                  static_cast<int>(config.kernel));

      matmul_onednn_wrapper(trans_input, trans_weight,
                            config.M, config.N, config.K, alpha,
                            src, config.lda,
                            weight, config.ldb,
                            beta, dst, config.ldc,
                            params, batch_params, bias, config.kernel,
                            is_weights_const);
      return;
    }
#endif

    config.kernel = matmul_algo_t::aocl_dlp;
    apilog_info("Given combination is not supported for matmul parallel primitive, executing matmul LOWOHA kernel with DLP fallback, algo: ",
                static_cast<int>(config.kernel));

    matmul_kernel_wrapper(layout, trans_input, trans_weight,
                          config.M, config.N, config.K, alpha,
                          src, config.lda, weight, config.ldb,
                          beta, dst, config.ldc,
                          params.dtypes, config.kernel,
                          params.mem_format_a, params.mem_format_b,
                          params, batch_params,
                          bias, is_weights_const);
  }
}
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl