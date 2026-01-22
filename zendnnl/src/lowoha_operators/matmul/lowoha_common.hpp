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

#ifndef _LOWOHA_COMMON_HPP
#define _LOWOHA_COMMON_HPP

#include "memory/memory_utils.hpp"
#include "operators/common/post_op.hpp"
#include "lowoha_operators/matmul/zendnnl_key.hpp"
#include "lowoha_operators/matmul/lru_cache.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

using namespace zendnnl::memory;
using namespace zendnnl::ops;

/**
 * @brief Structure to hold data types for matrix multiplication operands
 */
struct matmul_data_types {
  data_type_t src = data_type_t::none;     ///< Source matrix data type
  data_type_t wei = data_type_t::none;     ///< Weight matrix data type
  data_type_t dst = data_type_t::none;     ///< Destination matrix data type
  data_type_t bias = data_type_t::none;    ///< Bias vector data type
  data_type_t compute = data_type_t::none; ///< Computation data type
};

/**
 * @brief Structure for post-operation parameters
 */
struct matmul_post_op {
  zendnnl::ops::post_op_type_t po_type;    ///< Type of post-operation
  void *buff;                              ///< Buffer for binary operations
  data_type_t dtype;                       ///< Data type of the buffer
  std::vector<int64_t> dims;               ///< Dimensions of the buffer
  float alpha;                             ///< Alpha parameter for operations
  float beta;                              ///< Beta parameter for operations
  int leading_dim;                         ///< Leading dimension for the buffer

  /**
   * @brief Default constructor for matmul_post_op
   */
  matmul_post_op() : po_type(zendnnl::ops::post_op_type_t::none), buff(nullptr),
    dtype(data_type_t::none), dims(), alpha(0.0f), beta(0.0f), leading_dim(-1) {}
};

/**
 * @brief Structure for quantization parameters (scales and zero-points)
 */
struct matmul_quantization_params_t {
  /**
   * @brief Individual quantization parameter (scale or zero-point)
   * 
   * Dimensions determine quantization granularity for weight matrix [K, N]:
   *   - Per-tensor:  dims = {} or {1}     → single scale for all weights
   *   - Per-channel: dims = {1, N}        → one scale per output channel
   *   - Per-group:   dims = {G, N}        → G groups along K, where G = K/group_size
   */
  struct matmul_quant_t {
    const void *buff;              ///< Pointer to quantization data buffer
    data_type_t dt;                ///< Data type of the buffer
    std::vector<int64_t> dims;     ///< Dimensions of the quantization tensor

    /**
     * @brief Default constructor for matmul_quant_t
     */
    matmul_quant_t() : buff(nullptr), dt(data_type_t::none), dims() {}
  };

  matmul_quant_t src_scale;  ///< Source tensor scale
  matmul_quant_t wei_scale;  ///< Weight tensor scale
  matmul_quant_t dst_scale;  ///< Destination tensor scale
  matmul_quant_t src_zp;     ///< Source tensor zero-point
  matmul_quant_t wei_zp;     ///< Weight tensor zero-point
  matmul_quant_t dst_zp;     ///< Destination tensor zero-point

  /**
   * @brief Default constructor for quantization parameters
   */
  matmul_quantization_params_t() : src_scale(), wei_scale(), dst_scale(),
    src_zp(), wei_zp(), dst_zp() {}
};

/**
 * @struct batch_params_t
 * @brief A structure to encapsulate batch dimensions and batch strides.
 *
 * This structure contains batch sizes (Batch_A and Batch_B) and batch strides
 * for source, weight, and destination tensors. The batch strides specify the
 * byte offset between consecutive batches in memory.
 */
 struct matmul_batch_params_t {
  int Batch_A = 1;              /**< Batch size for source tensor. */
  int Batch_B = 1;              /**< Batch size for weight tensor. */
  size_t batch_stride_src =
    -1;  /**< Byte stride between batches for source tensor (-1 means calculate from dimensions). */
  size_t batch_stride_wei =
    -1;  /**< Byte stride between batches for weight tensor (-1 means calculate from dimensions). */
  size_t batch_stride_dst =
    -1;  /**< Byte stride between batches for destination tensor (-1 means calculate from dimensions). */

  /**
   * @brief Default constructor for `matmul_batch_params_t`.
   *
   * Initializes Batch_A and Batch_B to 1, and all strides to -1.
   */
  matmul_batch_params_t() : Batch_A(1), Batch_B(1), batch_stride_src(-1),
    batch_stride_wei(-1), batch_stride_dst(-1) {}
};

/**
 * @brief Main parameter structure for LOWOHA matrix multiplication
 */
struct matmul_params {
  matmul_data_types dtypes;                              ///< Data types for operands
  std::vector<matmul_post_op> postop_;                    ///< Post-operation chain
  matmul_quantization_params_t quant_params;     ///< Quantization parameters
  char mem_format_a;                              ///< Memory format for matrix A
  char mem_format_b;                              ///< Memory format for matrix B
  matmul_algo_t lowoha_algo;                      ///< Selected algorithm
  uint64_t num_threads;                            ///< Number of threads
  std::string plugin_op;                       ///< Plugin op name
  /**
   * @brief Default constructor for matmul_params
   */
  matmul_params() : dtypes(), postop_(), quant_params(), mem_format_a('n'),
    mem_format_b('n'), lowoha_algo(matmul_algo_t::none), num_threads(0), plugin_op("") {}
};

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_COMMON_HPP