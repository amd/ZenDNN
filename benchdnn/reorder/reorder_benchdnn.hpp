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
#ifndef _REORDER_BENCHDNN_HPP_
#define _REORDER_BENCHDNN_HPP_

#include "example_utils.hpp"
#include "benchdnn.hpp"
#include "reorder_utils.hpp"
#include "reorder_tensor_factory.hpp"
#include "reorder_lowoha.hpp"

namespace zendnnl {
namespace benchdnn {
namespace reorder {

using namespace zendnnl::interface;
using namespace zendnnl::examples;

/**
 * @brief Runs a single reorder operation and records timing statistics.
 *
 * @param input_tensor Input tensor to be reordered.
 * @param cfg ReorderConfig with dimensions, data type, kernel, in-place flag.
 * @param stats Reference to TimingStats struct to store timing results.
 * @param isNotWarmup If true, measures and accumulates detailed timings.
 * @return int OK (0) on success, NOT_OK (1) on failure.
 */
int run_reorder(tensor_t input_tensor, const ReorderConfig &cfg,
                TimingStats &stats, bool isNotWarmup = false);

/**
 * @brief Runs the regular reorder benchmark for a set of configurations.
 *
 * Iterates through each provided configuration, performs warmup and measured iterations,
 * collects timing statistics, and stores the results in the provided vector.
 *
 * @param configs Vector of ReorderConfig objects specifying each benchmark run.
 * @param reorder_results Vector to store pairs of configuration and timing statistics (std::pair<ReorderConfig, TimingStats>).
 *        Each entry corresponds to a unique configuration and its associated timing results.
 * @param cache_size Cache size for cold cache flushing (if enabled).
 * @return int Returns OK (0) on success, NOT_OK (1) on failure.
 */
int reorder_benchdnn(const std::vector<ReorderConfig> &configs,
                     std::vector<std::pair<ReorderConfig, TimingStats>> &reorder_results,
                     size_t cache_size);

/**
 * @brief Entry point for the reorder benchmark.
 *
 * Reads input configurations, dispatches to regular or LOWOHA benchmark,
 * and writes results to console and CSV file.
 *
 * @param in_filename Path to the input configuration file.
 * @param out_filename Path to the output CSV file for results.
 * @param isLOWOHA If true, runs LOWOHA benchmark; otherwise regular reorder.
 * @param cache_size Cache size for cold cache flushing (if enabled).
 * @return int Returns OK (0) on success, NOT_OK (1) on failure.
 */
int bench(const std::string &in_filename, const std::string &out_filename,
          const bool isLOWOHA, size_t cache_size);

} // namespace reorder
} // namespace benchdnn
} // namespace zendnnl

#endif