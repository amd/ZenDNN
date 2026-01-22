/********************************************************************************
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

#ifndef _LOWOHA_CACHE_HPP
#define _LOWOHA_CACHE_HPP

#include "lowoha_common.hpp"
#include "lru_cache.hpp"
#include "zendnnl_key.hpp"
#include "lowoha_matmul_utils.hpp"
#include "operators/matmul/matmul_config.hpp"
#include <cstdlib>
#include <cstring>
#include <vector>

namespace zendnnl {
namespace lowoha {
namespace matmul {

/**
 * @brief Get cached or compute zero-point compensation
 * 
 * @param key_obj Cache key for lookup
 * @param M Number of rows in source matrix
 * @param N Number of columns in output matrix
 * @param K Inner dimension (reduction dimension)
 * @param src Source matrix pointer
 * @param wei Weight matrix pointer
 * @param src_zp Source zero-point value
 * @param wei_zp Weight zero-point value
 * @param transA Whether source is transposed
 * @param transB Whether weights are transposed
 * @param lda Leading dimension of source
 * @param ldb Leading dimension of weights
 * @param src_dtype Source data type (u8 or s8)
 * @param is_weights_const Whether weights are constant across inferences
 * @param zp_comp_ndim [out] Dimensionality of compensation (0=none, 1=1D, 2=2D)
 * @return Pointer to compensation buffer (owned by cache or caller based on config)
 */
inline int32_t* cache_or_compute_zp_compensation(
    const Key_matmul& key_obj,
    int M, int N, int K,
    const void* src, const void* wei,
    int32_t src_zp, int32_t wei_zp,
    bool transA, bool transB,
    int lda, int ldb,
    data_type_t src_dtype,
    bool is_weights_const,
    int& zp_comp_ndim) {
  
  // No compensation needed if both zero-points are zero
  if (src_zp == 0 && wei_zp == 0) {
    zp_comp_ndim = 0;
    return nullptr;
  }
  
  // Only cache 1D compensation (src_zp only case) since it depends only on weights
  // 2D compensation depends on source data which changes per inference
  // Caching is enabled by default and requires weights to be constant
  const bool can_cache = (wei_zp == 0 && src_zp != 0) && 
                         is_weights_const &&
                         ops::matmul_config_t::instance().get_zp_comp_cache();
  
  // Static LRU cache for 1D zero-point compensation
  static lru_cache_t<Key_matmul, int32_t*> zp_comp_cache;
  
  // Compute strides based on transpose flags
  int src_s0 = transA ? 1 : lda;
  int src_s1 = transA ? lda : 1;
  int wei_s0 = transB ? 1 : ldb;
  int wei_s1 = transB ? ldb : 1;
  
  const int8_t* wei_buff = static_cast<const int8_t*>(wei);
  int32_t* zp_comp_acc = nullptr;
  
  if (wei_zp == 0 && src_zp != 0) {
    // Only src has zero-point: 1D compensation (cacheable)
    // zp_comp[n] = -src_zp * sum(weights[:, n])
    zp_comp_ndim = 1;
    
    // Check cache first
    if (can_cache && zp_comp_cache.find_key(key_obj)) {
      log_info("Cache hit: reading cached zero-point compensation");
      return zp_comp_cache.get(key_obj);
    }
    
    // Compute compensation
    size_t alignment = 64;
    size_t comp_size = (N * sizeof(int32_t) + alignment - 1) & ~(alignment - 1);
    zp_comp_acc = static_cast<int32_t*>(aligned_alloc(alignment, comp_size));
    if (!zp_comp_acc) return nullptr;
    
    // Compute column sums of weights
    std::vector<int32_t> wei_col_sum(N, 0);
    #pragma omp parallel for
    for (int n = 0; n < N; ++n) {
      for (int k = 0; k < K; ++k) {
        wei_col_sum[n] += wei_buff[wei_s0 * k + wei_s1 * n];
      }
    }
    
    // Compute compensation: zp_comp[n] = -src_zp * wei_col_sum[n]
    #pragma omp parallel for
    for (int n = 0; n < N; ++n) {
      zp_comp_acc[n] = -src_zp * wei_col_sum[n];
    }
    
    // Add to cache if enabled
    if (can_cache) {
      std::lock_guard<std::mutex> lock(get_lowoha_mutex());
      zp_comp_cache.add(key_obj, zp_comp_acc);
      log_info("Cache add: storing zero-point compensation");
    }
  }
  else if (src_zp == 0 && wei_zp != 0) {
    // Only weights have zero-point: 2D compensation (not cacheable - depends on src)
    // zp_comp[m,n] = -wei_zp * sum(src[m, :])
    zp_comp_ndim = 2;
    
    size_t alignment = 64;
    size_t comp_size = (M * N * sizeof(int32_t) + alignment - 1) & ~(alignment - 1);
    zp_comp_acc = static_cast<int32_t*>(aligned_alloc(alignment, comp_size));
    if (!zp_comp_acc) return nullptr;
    
    // Compute row sums of source
    std::vector<int32_t> src_row_sum(M, 0);
    if (src_dtype == data_type_t::u8) {
      const uint8_t* src_buff = static_cast<const uint8_t*>(src);
      for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
          src_row_sum[m] += src_buff[src_s0 * m + src_s1 * k];
        }
      }
    } else {
      const int8_t* src_buff = static_cast<const int8_t*>(src);
      for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
          src_row_sum[m] += src_buff[src_s0 * m + src_s1 * k];
        }
      }
    }
    
    // Compute 2D compensation
    for (int m = 0; m < M; ++m) {
      int32_t comp = -wei_zp * src_row_sum[m];
      for (int n = 0; n < N; ++n) {
        zp_comp_acc[m * N + n] = comp;
      }
    }
  }
  else {
    // Both have zero-points: 2D compensation (not cacheable - depends on src)
    // zp_comp[m,n] = -src_zp * wei_col_sum[n] - wei_zp * src_row_sum[m] + src_zp * wei_zp * K
    zp_comp_ndim = 2;
    
    size_t alignment = 64;
    size_t comp_size = (M * N * sizeof(int32_t) + alignment - 1) & ~(alignment - 1);
    zp_comp_acc = static_cast<int32_t*>(aligned_alloc(alignment, comp_size));
    if (!zp_comp_acc) return nullptr;
    
    // Compute row sums of source
    std::vector<int32_t> src_row_sum(M, 0);
    if (src_dtype == data_type_t::u8) {
      const uint8_t* src_buff = static_cast<const uint8_t*>(src);
      for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
          src_row_sum[m] += src_buff[src_s0 * m + src_s1 * k];
        }
      }
    } else {
      const int8_t* src_buff = static_cast<const int8_t*>(src);
      for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
          src_row_sum[m] += src_buff[src_s0 * m + src_s1 * k];
        }
      }
    }
    
    // Compute column sums of weights
    std::vector<int32_t> wei_col_sum(N, 0);
    for (int k = 0; k < K; ++k) {
      for (int n = 0; n < N; ++n) {
        wei_col_sum[n] += wei_buff[wei_s0 * k + wei_s1 * n];
      }
    }
    
    // Compute 2D compensation with full formula
    int32_t base_comp = src_zp * wei_zp * K;
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        zp_comp_acc[m * N + n] = -src_zp * wei_col_sum[n]
                                 - wei_zp * src_row_sum[m]
                                 + base_comp;
      }
    }
  }
  
  return zp_comp_acc;
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_CACHE_HPP

