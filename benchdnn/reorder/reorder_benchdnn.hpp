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
#ifndef _REORDER_BENCHDNN_HPP_
#define _REORDER_BENCHDNN_HPP_

#include "zendnnl.hpp"
#include "example_utils.hpp"
#include "benchdnn.hpp"
#include "reorder_parser.hpp"
#include <chrono>

namespace zendnnl {
namespace benchdnn {
namespace reorder {

using namespace zendnnl::interface; ///< ZenDNN C++ interface
using namespace zendnnl::examples;  ///< Example utilities

/**
 * @brief Runs a single reorder operation and records timing statistics.
 *
 * @param input_tensor Input tensor to be reordered.
 * @param kernel_name Name of the kernel backend to use for reorder.
 * @param stats Reference to TimingStats struct to store timing results.
 * @param isNotWarmup If true, this is a measured iteration; if false, a warmup iteration.
 * @return Status code (0 for success, non-zero for error).
 */
int run_reorder(tensor_t input_tensor, std::string kernel_name,
                TimingStats &stats,
                bool isNotWarmup = false);

/**
 * @brief Runs the reorder benchmark for a set of configurations.
 *
 * Iterates through each provided configuration, performs warmup and measured iterations,
 * collects timing statistics, and stores the results in the provided vector.
 *
 * @param configs Vector of ReorderConfig objects specifying each benchmark run.
 * @param reorder_results Vector to store pairs of configuration and timing statistics (std::pair<ReorderConfig, TimingStats>).
 *        Each entry corresponds to a unique configuration and its associated timing results.
 * @return Status code (0 for success, non-zero for error).
 */
int reorder_benchdnn(std::vector<ReorderConfig> configs,
                     std::vector<std::pair<ReorderConfig, TimingStats>> &reorder_results);

/**
 * @brief Logs a detailed error message for a failed reorder benchmark configuration.
 *
 * Prints configuration parameters to the error log for debugging failed runs.
 *
 * @param cfg ReorderConfig structure for which the benchmark failed.
 */
void log_benchmark_failure(const ReorderConfig &cfg);


/**
 * @brief Prints the reorder benchmark results in a formatted table to the given output stream.
 *
 * Dynamically calculates column widths for neat alignment and prints a table of results for each configuration.
 * Includes detailed timing breakdowns if enabled at compile time.
 *
 * @param reorder_results Vector of pairs (ReorderConfig, TimingStats) containing benchmark results.
 * @param outfile Output stream to print the table (e.g., std::cout or file stream).
 */
void print_results(std::vector<std::pair<ReorderConfig, TimingStats>>
                   &reorder_results, std::ostream &outfile);

/**
 * @brief Logs the reorder benchmark results in CSV format to the given output stream.
 *
 * Writes a CSV header and a row for each configuration, including timing breakdowns if enabled.
 *
 * @param reorder_results Vector of pairs (ReorderConfig, TimingStats) containing benchmark results.
 * @param outfile Output stream to write the CSV data (e.g., file stream).
 */
void log_results(std::vector<std::pair<ReorderConfig, TimingStats>>
                 &reorder_results, std::ostream &outfile);

/**
 * @brief Entry point for the reorder benchmark utility.
 *
 * Reads input configurations from a file, runs the benchmark, and writes results to output file.
 *
 * @param in_filename Path to the input configuration file.
 * @param out_filename Path to the output CSV file for results.
 * @return Status code (0 for success, non-zero for error).
 */
int bench(const std::string &in_filename, const std::string &out_filename,
          size_t cache_size);

} // namespace reorder
} // namespace benchdnn
} // namespace zendnnl

#endif