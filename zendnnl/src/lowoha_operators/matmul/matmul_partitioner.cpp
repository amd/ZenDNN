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
namespace zendnnl {
namespace lowoha {
namespace matmul {


int get_tile_size_from_env(const char *env_var, int default_value) {
  const char *env_value = std::getenv(env_var);
  if (env_value != nullptr) {
    int value = std::stoi(env_value);
    if (value > 0) {
      return value;
    }
    else {
      return default_value;
    }
  }
  return default_value;
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

  M_block = get_tile_size_from_env("ZENDNN_MATMUL_M_TILE", M_block);
  N_block = get_tile_size_from_env("ZENDNN_MATMUL_N_TILE", N_block);

  M_block = std::min(M_block, config.M);
  N_block = std::min(N_block, config.N);

  return {M_block, N_block};
}

bool should_use_kc_blocking(
  const matmul_partition_config_t &config,
  const matmul_params &params
) {
#if ENABLE_LIBXSMM_BRGEMM_KERNEL
  // TODO: Need to support bf16 brgemm kernels later
  if (config.dtypes.src != data_type_t::f32) {
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

void execute_partitioned_matmul_brgemm(
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
  const void *src,
  const void *weight,
  void *dst,
  const void *bias,
  const matmul_partition_config_t &config,
  const matmul_params &params,
  const brgemm_kernel_invoker_t &brgemm_callback,
  const tile_kernel_invoker_t &tile_callback
) {
  bool use_brgemm = should_use_kc_blocking(config, params);

  if (use_brgemm) {
    execute_partitioned_matmul_brgemm(
      src, weight, dst, bias, config, brgemm_callback
    );
  }
  else {
    execute_partitioned_matmul_standard(
      src, weight, dst, bias, config, tile_callback
    );
  }
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl