/********************************************************************************
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
#ifndef _MATMUL_PARSER_HPP_
#define _MATMUL_PARSER_HPP_

#include "zendnnl.hpp"
#include "example_utils.hpp"
#include "utils/benchdnn_utils.hpp"
#include <chrono>

namespace zendnnl {
namespace benchdnn {
namespace matmul {

/**
 * @struct MatmulConfig
 * @brief Holds user-defined parameters for configuring and benchmarking matrix multiplication.
 *
 * This structure encapsulates all the parameters required to configure a matrix multiplication
 * benchmark, including matrix dimensions, number of iterations, data types, bias, post-operations,
 * kernel selection, and warmup iterations.
 *
 * @var m Number of rows in matrix A (output rows).
 * @var k Number of columns in matrix A / rows in matrix B (inner dimension).
 * @var n_values Vector of output columns for each layer (multi-layer support).
 * @var iters Number of iterations to run the benchmark.
 * @var dt Data types for input, weights, and output (e.g., f32:f32:f32).
 * @var isBiasEnabled Flag indicating if bias is enabled in the matmul operation.
 * @var bias_dt Data type for the bias tensor (e.g., f32, bf16). Defaults to f32 if not specified.
 * @var post_ops List of post-operations to apply after matmul (e.g., relu, gelu). Parsed from a colon-separated string (e.g., "relu:gelu").
 * @var binary_post_ops_pos List of positions for binary post-operations (indices in the post_ops list where binary ops are applied).
 * @var kernel_name Name of the kernel backend to invoke (e.g., aocl_blis, aocl_blis_blocked).
 * @var warmup_iters Number of warmup iterations to run before actual benchmarking.
 */
struct MatmulConfig {
  // TODO: Extend support to BMM for 3D tensors
  size_t m; /**< Number of rows in matrix A (output rows). */
  size_t k; /**< Number of columns in matrix A / rows in matrix B (inner dimension). */
  std::vector<size_t> n_values; /**< Vector of output columns
                                for each layer (multi-layer support). */
  int iters; /**< Number of iterations to run the benchmark. */
  std::vector<zendnnl::common::data_type_t> dt; /**< Data types for
                                                input, weights, and output (e.g., f32:f32:f32). */
  bool isBiasEnabled; /**< Flag indicating if bias is enabled in the matmul operation. */
  zendnnl::common::data_type_t bias_dt; /**< Data type for the bias tensor
                                        (e.g., f32, bf16). Defaults to f32 if not specified. */
  std::vector<zendnnl::ops::post_op_type_t> post_ops; /**< List of post operations
                                                      to apply (e.g., relu, gelu). */
  std::vector<int> binary_post_ops_pos; /**< List of positions for
                                        binary post-operations. */
  std::string kernel_name; /**< Name of the kernel backend
                            to invoke (e.g., aocl_blis, aocl_blis_blocked). */
  int warmup_iters; /**< Number of warmup iterations to run before actual benchmarking. */
};

/**
 * @brief Parses the input file and populates a vector of MatmulConfig structures.
 *
 * Reads each line from the provided input file stream, parses the matrix multiplication
 * configuration parameters, and appends a MatmulConfig object to the configs vector.
 * Supports multi-layer and post-op parsing, as well as bias and kernel selection.
 *
 * The function also sets the isPipeline flag to true if the input describes a pipeline (multi-layer) configuration.
 *
 * @param infile Reference to an open std::ifstream containing the input configurations.
 * @param configs Reference to a vector of MatmulConfig to be populated.
 * @param isPipeline Reference to a boolean flag that will be set to true if the input describes a pipeline configuration.
 */
void inputParser(std::ifstream &infile, std::vector<MatmulConfig> &configs,
                 bool &isPipeline);

} // namespace matmul
} // namespace benchdnn
} // namespace zendnnl

#endif