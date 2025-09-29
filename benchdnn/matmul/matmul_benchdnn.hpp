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
#ifndef _MATMUL_BENCHDNN_HPP_
#define _MATMUL_BENCHDNN_HPP_

#include "example_utils.hpp"
#include "benchdnn.hpp"
#include "matmul_utils.hpp"
#include "matmul_tensor_factory.hpp"
#include "matmul_lowoha.hpp"

namespace zendnnl {
namespace benchdnn {
namespace matmul {

using namespace zendnnl::interface;
using namespace zendnnl::examples;

/**
 * @brief Runs a single matmul operation with optional bias and post-ops, and measures timing.
 *
 * Sets up the matmul context and operator, executes the operation, and records timing statistics if enabled.
 * Supports both warmup and measured runs. Bias and post-ops are optional and controlled by parameters.
 *
 * @param output_tensor Output tensor for the matmul operation.
 * @param input_tensor Input tensor for the matmul operation.
 * @param weights Weights tensor for the matmul operation.
 * @param bias Bias tensor for the matmul operation (used only if cfg.isBiasEnabled is true).
 * @param cfg MatmulConfig structure containing all configuration parameters (dimensions, data types, bias, post-ops, kernel, etc.).
 * @param binary_post_ops_tensors Vector of tensors for binary post-operations (used if any binary post-ops are present in cfg.post_ops).
 * @param stats Reference to TimingStats struct to accumulate timing results.
 * @param isNotWarmup If true, measures and accumulates timing; if false, only runs the operation.
 * @return int OK (0) on success, NOT_OK (1) on failure.
 */
int run_matmul(tensor_t output_tensor, tensor_t input_tensor, tensor_t weights,
               tensor_t bias, MatmulConfig cfg, std::vector<tensor_t> binary_post_ops_tensors,
               TimingStats &stats, bool isNotWarmup = false);

/**
 * @brief Benchmarks matmul (optionally fused with post-ops) using user-specified parameters.
 *
 * For each configuration in the provided vector, performs warmup and measured iterations,
 * collects timing statistics, and stores the results in the provided vector.
 *
 * @param configs Vector of MatmulConfig objects specifying benchmark parameters for each run.
 * @param matmul_results Vector to store pairs of configuration and timing statistics (std::pair<MatmulConfig, std::vector<TimingStats>>).
 *        Each entry corresponds to a unique configuration and its associated vector of timing results for all runs.
 * @return int Returns OK (0) on success, NOT_OK (1) on failure.
 */
int matmul_benchdnn(std::vector<MatmulConfig> configs,
                    std::vector<std::pair<MatmulConfig, std::vector<TimingStats>>> &matmul_results);

/**
 * @brief Runs the full matmul benchmark suite from an input file and writes results to a CSV file.
 *
 * Reads benchmark configurations from the specified input file, executes all matmul benchmarks,
 * collects timing and performance statistics, and writes the results to the specified output CSV file.
 *
 * @param in_filename Path to the input file containing benchmark configurations.
 * @param out_filename Path to the output CSV file for writing results.
 * @param inputMode Mode of input (FILE, MODEL, COMMAND_LINE).
 * @param options Global options for command-line configuration.
 * @param isLOWOHA If true, runs the LOWOHA (Low Overhead API) benchmark variant.
 * @return int Returns OK (0) on success, NOT_OK (1) on failure.
 */
int bench(const std::string &in_filename, const std::string &out_filename,
          const InputMode inputMode, const global_options &options, const bool isLOWOHA);

} // namespace matmul
} // namespace benchdnn
} // namespace zendnnl

#endif