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
#ifndef _EMBAG_BENCHDNN_HPP_
#define _EMBAG_BENCHDNN_HPP_

#include "example_utils.hpp"
#include "benchdnn.hpp"
#include "embag_utils.hpp"
#include "embag_tensor_factory.hpp"

namespace zendnnl {
namespace benchdnn {
namespace embag {

using namespace zendnnl::interface;
using namespace zendnnl::examples;

/**
 * @brief Runs a single embag operation and measures timing.
 *
 * Sets up the embag context and operator, executes the operation, and records timing statistics if enabled.
 * Supports both warmup and measured runs. Weights are optional and controlled by parameters.
 *
 * @param output_tensor Output tensor for the embag operation.
 * @param table_tensor Table tensor for the embag operation.
 * @param indices_tensor Indices tensor for the embag operation.
 * @param offsets_tensor Offsets tensor for the embag operation.
 * @param weights Weights tensor for the embag operation (used only if cfg.is_weights is true).
 * @param cfg EmbagConfig structure containing all configuration parameters (dimensions, data types, algo, padding_index, etc.).
 * @param stats Reference to TimingStats struct to accumulate timing results.
 * @param isNotWarmup If true, measures and accumulates timing; if false, only runs the operation.
 * @return int OK (0) on success, NOT_OK (1) on failure.
 */
int run_embag(tensor_t output_tensor, tensor_t table_tensor,
              tensor_t indices_tensor,
              tensor_t offsets_tensor, tensor_t weights_tensor, EmbagConfig cfg,
              TimingStats &stats, bool isNotWarmup = false);

/**
 * @brief Benchmarks embag using user-specified parameters.
 *
 * For each configuration in the provided vector, performs warmup and measured iterations,
 * collects timing statistics, and stores the results in the provided vector.
 *
 * @param configs Vector of EmbagConfig objects specifying benchmark parameters for each run.
 * @param embag_results Vector to store pairs of configuration and timing statistics (std::pair<EmbagConfig, std::vector<TimingStats>>).
 *        Each entry corresponds to a unique configuration and its associated vector of timing results for all runs.
 * @return int Returns OK (0) on success, NOT_OK (1) on failure.
 */
int embag_benchdnn(std::vector<EmbagConfig> configs,
                   std::vector<std::pair<EmbagConfig, TimingStats>> &embag_results);

/**
 * @brief Runs the full embag benchmark suite from an input file and writes results to a CSV file.
 *
 * Reads benchmark configurations from the specified input file, executes all embag benchmarks,
 * collects timing and performance statistics, and writes the results to the specified output CSV file.
 *
 * @param in_filename Path to the input file containing benchmark configurations.
 * @param out_filename Path to the output CSV file for writing results.
 * @return int Returns OK (0) on success, NOT_OK (1) on failure.
 */
int bench(const std::string &in_filename, const std::string &out_filename,
          size_t cache_size);

} // namespace embag
} // namespace benchdnn
} // namespace zendnnl

#endif