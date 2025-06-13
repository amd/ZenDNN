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
#ifndef _REORDER_PARSER_HPP_
#define _REORDER_PARSER_HPP_

#include "zendnnl.hpp"
#include "example_utils.hpp"
#include "utils/benchdnn_utils.hpp"
#include <chrono>

namespace zendnnl {
namespace benchdnn {
namespace reorder {

/**
 * @struct ReorderConfig
 * @brief Holds user-defined parameters for configuring and benchmarking reorder operations.
 *
 * This structure encapsulates all the parameters required to configure a reorder
 * benchmark, including tensor dimensions, number of iterations, data type, kernel selection,
 * in-place flag, and warmup iterations.
 *
 * @var rows Number of rows in the tensor to reorder.
 * @var cols Number of columns in the tensor to reorder.
 * @var iters Number of iterations to run the benchmark.
 * @var dt Data type for the reorder operation (e.g., f32, bf16).
 * @var kernel_name Name of the kernel backend to invoke (e.g., aocl).
 * @var isInplace Flag indicating if the operation is in-place (true for in-place, false for out-of-place).
 * @var warmup_iters Number of warmup iterations to run before actual benchmarking.
 */
struct ReorderConfig {
  size_t rows; /**< Number of rows in the tensor to reorder. */
  size_t cols; /**< Number of columns in the tensor to reorder. */
  int iters;   /**< Number of iterations to run the benchmark. */
  zendnnl::common::data_type_t dt; /**< Data type for the reorder operation. */
  std::string kernel_name; /**< Name of the kernel backend to invoke. */
  bool isInplace; /**< Flag indicating if the operation is in-place. */
  int warmup_iters; /**< Number of warmup iterations to run before actual benchmarking. */
};

/**
 * @brief Parses the input file and populates a vector of ReorderConfig structures.
 *
 * Reads each line from the provided input file stream, parses the reorder
 * configuration parameters, and appends a ReorderConfig object to the configs vector.
 *
 * @param infile Reference to an open std::ifstream containing the input configurations.
 * @param configs Reference to a vector of ReorderConfig to be populated.
 */
void inputParser(std::ifstream &infile, std::vector<ReorderConfig> &configs);

} // namespace reorder
} // namespace benchdnn
} // namespace zendnnl

#endif