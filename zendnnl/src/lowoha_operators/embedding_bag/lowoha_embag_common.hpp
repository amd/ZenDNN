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

#ifndef _LOWOHA_EMBAG_COMMON_HPP
#define _LOWOHA_EMBAG_COMMON_HPP

#include <cstdint>
#include <cstdlib>
#include <vector>
#include "memory/memory_utils.hpp"
#include "operators/embag/embag_context.hpp"
#include "operators/embag/embag_config.hpp"

namespace zendnnl {
namespace lowoha {
namespace embag {

using namespace zendnnl::memory;

// Use the same embag_algo_t and embag_kernel_t from ops namespace
using embag_algo_t = zendnnl::ops::embag_algo_t;
using embag_kernel_t = zendnnl::ops::embag_kernel_t;

/**
 * @brief Structure to hold data types for embedding bag operands
 */
struct embag_data_types_t {
  data_type_t table   = data_type_t::none;   // Embedding table data type
  data_type_t output  = data_type_t::none;   // Output tensor data type
  data_type_t indices = data_type_t::none;   // Data type of indices (s32 or s64)
  data_type_t offsets = data_type_t::none;   // Data type of offsets (s32 or s64)
  data_type_t scale = data_type_t::none;     // Data type of scale(f32 or f16)
  data_type_t bias = data_type_t::none;      // Data type of bias(f32 or f16)

  /**
   * @brief Default constructor for embag_data_types_t
   */
  embag_data_types_t() : table(data_type_t::none), output(data_type_t::none),
    indices(data_type_t::s64), offsets(data_type_t::s64), scale(data_type_t::f32),
    bias(data_type_t::f32) {}
};

/**
 * @struct embag_params_t
 * @brief Main parameter structure for LOWOHA embedding bag operation
 *
 * This structure aggregates all configuration parameters needed for
 * embedding bag computation, including data types, algorithm selection,
 * and threading configuration.
 */
struct embag_params_t {
  embag_data_types_t dtypes;     // Data types for operands
  embag_algo_t algo;             // Reduction algorithm (sum/mean/max/none)
  uint64_t num_embeddings;       // Number of rows in the embedding table
  uint64_t embedding_dim;        // Dimension of each embedding vector
  uint64_t num_threads;          // Number of threads
  uint64_t num_indices;          // Total number of indices
  uint64_t num_bags;             // Number of bags (output rows)
  bool is_weights;               // Whether weights are present
  bool include_last_offset;      // Whether offsets includes the last offset (num_indices)
  int64_t padding_idx;           // Index to ignore during lookup (-1 means no padding)
  bool fp16_scale_bias;          // Whether data type of scale and bias is fp16
  uint64_t dst_stride;           // Destination tensor stride value
  embag_kernel_t kernel;         // Embag kernel
  /**
   * @brief Default constructor for embag_params_t
   */
  embag_params_t() : dtypes(), algo(embag_algo_t::sum),
    num_embeddings(0), embedding_dim(0), num_threads(0),
    num_indices(0), num_bags(0), is_weights(false), include_last_offset(false),
    padding_idx(-1), fp16_scale_bias(false), dst_stride(0),
    kernel(embag_kernel_t::fbgemm) {}
};

/**
 * @brief Select embedding bag kernel based on parameters and environment variable
 *
 * This function selects the appropriate kernel for embedding bag operation based on:
 * 1. The kernel specified in params (if not none)
 * 2. The ZENDNNL_EMBAG_ALGO environment variable
 *
 * @param params The embedding bag parameters
 * @return embag_kernel_t The selected kernel
 */
inline static embag_kernel_t kernel_select(embag_params_t &params) {
  using namespace zendnnl::ops;

  // Get config instance and initialize from environment
  embag_config_t &embag_config = embag_config_t::instance();
  embag_config.set_env_config();

  // Check if kernel is already specified in params
  int32_t algo = params.kernel == embag_kernel_t::none ?
                 embag_config.get_kernel() : static_cast<int32_t>(params.kernel);

  // Default to native kernel if none specified
  embag_kernel_t kernel = (algo == static_cast<int32_t>(embag_kernel_t::none)) ?
                          embag_kernel_t::native : static_cast<embag_kernel_t>(algo);

  // TODO: Add auto_tuner kernel selection
  if (kernel == embag_kernel_t::auto_tuner) {
    kernel = embag_kernel_t::native;
  }

  // Update params with selected kernel
  params.kernel = kernel;
  return kernel;
}

/**
 * @brief Validate embedding bag input parameters
 */
inline static status_t validate_embag_inputs(
  const void *table,
  const void *indices,
  void *dst,
  const embag_params_t &params) {

  if (table == nullptr) {
    log_error("embedding_bag_direct: table pointer is null");
    return status_t::failure;
  }
  if (indices == nullptr) {
    log_error("embedding_bag_direct: indices pointer is null");
    return status_t::failure;
  }
  if (dst == nullptr) {
    log_error("embedding_bag_direct: destination pointer is null");
    return status_t::failure;
  }

  if (params.num_embeddings <= 0) {
    log_error("embedding_bag_direct: num_embeddings must be positive");
    return status_t::failure;
  }
  if (params.embedding_dim <= 0) {
    log_error("embedding_bag_direct: embedding_dim must be positive");
    return status_t::failure;
  }

  if (params.num_indices < 0) {
    log_error("embedding_bag_direct: num_indices cannot be negative");
    return status_t::failure;
  }
  if (params.num_bags <= 0 && params.algo != embag_algo_t::none) {
    log_error("embedding_bag_direct: num_bags must be positive for reduction ops");
    return status_t::failure;
  }

  if (params.dtypes.table == data_type_t::none) {
    log_error("embedding_bag_direct: table data type not specified");
    return status_t::failure;
  }
  if (params.dtypes.output == data_type_t::none) {
    log_error("embedding_bag_direct: output data type not specified");
    return status_t::failure;
  }

  return status_t::success;
}

/**
 * @brief Convert embag_algo_t to string for logging
 */
inline static const char *algo_to_string(embag_algo_t algo) {
  switch (algo) {
  case embag_algo_t::none:
    return "none";
  case embag_algo_t::sum:
    return "sum";
  case embag_algo_t::mean:
    return "mean";
  case embag_algo_t::max:
    return "max";
  default:
    return "unknown";
  }
}

/**
 * @brief Convert data_type_t to string for logging
 */
inline static const char *dtype_to_string(data_type_t dtype) {
  switch (dtype) {
  case data_type_t::none:
    return "none";
  case data_type_t::f32:
    return "f32";
  case data_type_t::bf16:
    return "bf16";
  case data_type_t::s8:
    return "s8";
  case data_type_t::s4:
    return "s4";
  case data_type_t::u4:
    return "u4";
  default:
    return "unknown";
  }
}

/**
 * @brief Convert embag_kernel_t to string for logging
 *
 * @param kernel The kernel enum value
 * @return const char* string representation
 */
inline static const char *kernel_to_string(embag_kernel_t kernel) {
  switch (kernel) {
  case embag_kernel_t::none:
    return "none";
  case embag_kernel_t::auto_tuner:
    return "auto_tuner";
  case embag_kernel_t::native:
    return "native";
  case embag_kernel_t::fbgemm:
    return "fbgemm";
  case embag_kernel_t::reference:
    return "reference";
  default:
    return "unknown";
  }
}

} // namespace embag
} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_EMBAG_COMMON_HPP

