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

#include "zendnnl.hpp"
#include "example_utils.hpp"
#include "benchdnn.hpp"
#include "matmul_parser.hpp"
#include <chrono>

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
 * @brief Logs a detailed error message for a failed benchmark configuration.
 *
 * Prints configuration parameters and post-ops to the error log for debugging failed runs.
 *
 * @param cfg MatmulConfig structure for which the benchmark failed.
 */
void log_benchmark_failure(const MatmulConfig &cfg);

/**
 * @brief Creates weight tensors for each layer in the matmul benchmark.
 *
 * Handles blocked/reordered layouts if required by the kernel. Populates the weights vector.
 *
 * @param tensor_factory Factory object for tensor creation.
 * @param cfg MatmulConfig structure specifying tensor dimensions and data types.
 * @param weights Vector to store created weight tensors.
 * @return int OK (0) on success, NOT_OK (1) on failure.
 */
int create_weights_tensor(tensor_factory_t &tensor_factory, MatmulConfig cfg,
                          std::vector<tensor_t> &weights);

/**
 * @brief Creates bias tensors for each layer if bias is enabled.
 *
 * Populates the bias vector with tensors of appropriate shape and data type.
 *
 * @param tensor_factory Factory object for tensor creation.
 * @param cfg MatmulConfig structure specifying tensor dimensions and data types.
 * @param bias Vector to store created bias tensors.
 * @return int OK (0) on success, NOT_OK (1) on failure.
 */
int create_bias_tensor(tensor_factory_t tensor_factory, const MatmulConfig &cfg,
                       std::vector<tensor_t> &bias);

/**
 * @brief Creates the input tensor for the matmul benchmark.
 *
 * Populates the input tensor with random or uniform values as specified.
 *
 * @param tensor_factory Factory object for tensor creation.
 * @param cfg MatmulConfig structure specifying tensor dimensions and data types.
 * @param input Reference to input tensor to be created.
 * @return int OK (0) on success, NOT_OK (1) on failure.
 */
int create_input_tensor(tensor_factory_t &tensor_factory,
                        const MatmulConfig &cfg, tensor_t &input);

/**
 * @brief Creates output tensors for each layer in the matmul benchmark.
 *
 * Populates the output vector with zero-initialized tensors of appropriate shape and data type.
 *
 * @param tensor_factory Factory object for tensor creation.
 * @param cfg MatmulConfig structure specifying tensor dimensions and data types.
 * @param output Vector to store created output tensors.
 * @return int OK (0) on success, NOT_OK (1) on failure.
 */
int create_output_tensor(tensor_factory_t &tensor_factory,
                         const MatmulConfig &cfg, std::vector<tensor_t> &output);

/**
 * @brief Creates tensors for binary post-operations for each layer.
 *
 * Populates a vector of vectors, where each inner vector contains tensors for binary post-ops for a layer.
 *
 * @param tensor_factory Factory object for tensor creation.
 * @param cfg MatmulConfig structure specifying tensor dimensions and post-ops.
 * @param binary_post_ops_tensors Vector of vectors to store created binary post-op tensors.
 * @return int OK (0) on success, NOT_OK (1) on failure.
 */
int create_binary_post_ops_tensors(tensor_factory_t &tensor_factory,
                                   const MatmulConfig &cfg,
                                   std::vector<std::vector<tensor_t>> &binary_post_ops_tensors);


/**
 * @brief Writes a single configuration's result as a CSV row to the output stream.
 *
 * Formats and writes the timing and performance statistics for a single configuration/layer
 * to the provided output stream, suitable for CSV output. Used for both single-layer and pipeline results.
 *
 * @param config      The matmul configuration for the current run/layer.
 * @param stat        The vector of timing statistics for all layers (per configuration).
 * @param outfile     The output stream to write the CSV row to.
 * @param layer_num   The index of the layer to extract results for (default: 0).
 * @param percentage  The percentage of total time this layer took (used in pipeline mode, default: 0.0).
 * @param isPipeline  Whether this is a pipeline (multi-layer) result (default: false).
 */
void write_each_config_result(const MatmulConfig &config,
                              const std::vector<TimingStats> &stat, std::ostream &outfile, int layer_num = 0,
                              double percentage = 0.0, bool isPipeline = false);

/**
 * @brief Calculates and updates the maximum column widths for table formatting.
 *
 * Examines the formatted data for a single configuration/layer and updates the col_widths vector
 * to ensure proper alignment of table columns for console output.
 *
 * @param config      The matmul configuration for the current run/layer.
 * @param stat        The vector of timing statistics for all layers (per configuration).
 * @param col_widths  The vector of column widths to update.
 * @param st_index    The starting index in col_widths for this row.
 * @param layer_num   The index of the layer to extract results for (default: 0).
 * @param percentage  The percentage of total time this layer took (used in pipeline mode, default: 0.0).
 * @param isPipeline  Whether this is a pipeline (multi-layer) result (default: false).
 */
void cal_column_width(const MatmulConfig &config,
                      const std::vector<TimingStats> &stat, std::vector<size_t> &col_widths,
                      int st_index, int layer_num = 0, double percentage = 0.0,
                      bool isPipeline = false);

/**
 * @brief Fills a vector of strings with formatted benchmarking results for a single matmul configuration/layer.
 *
 * Extracts relevant statistics and configuration details for a given layer of a matmul benchmark run,
 * formats them as strings, and appends them to the provided row vector. The resulting row can be used for table or CSV output.
 *
 * @param config      The matmul configuration for the current run/layer.
 * @param stat        The vector of timing statistics for all layers (per configuration).
 * @param row         The vector to be filled with formatted result strings for this layer.
 * @param layer_num   The index of the layer to extract results for (default: 0).
 * @param percentage  The percentage of total time this layer took (used in pipeline mode, default: 0.0).
 * @param isPipeline  Whether this is a pipeline (multi-layer) result (default: false).
 */
void fill_row(const MatmulConfig &config,
              const std::vector<TimingStats> &stat, std::vector<std::string> &row,
              int layer_num = 0, double percentage = 0.0,
              bool isPipeline = false);

/**
 * @brief Logs pipeline (multi-layer) matmul benchmark results to a CSV file.
 *
 * Writes a summary and per-layer timing and performance statistics for each configuration
 * in CSV format to the provided output stream. Includes optional timing breakdowns if enabled.
 *
 * @param matmul_results Vector of pairs of MatmulConfig and per-layer TimingStats.
 * @param outfile Output stream to write CSV results to (e.g., std::ofstream).
 */
void log_pipeline_results(
  std::vector<std::pair<MatmulConfig, std::vector<TimingStats>>> &matmul_results,
  std::ostream &outfile);

/**
 * @brief Prints pipeline (multi-layer) matmul benchmark results as a formatted table.
 *
 * Outputs a summary and per-layer timing and performance statistics for each configuration
 * in a human-readable table format to the provided output stream. Includes optional timing breakdowns if enabled.
 *
 * @param matmul_results Vector of pairs of MatmulConfig and per-layer TimingStats.
 * @param outfile Output stream to print table results to (e.g., std::cout).
 */
void print_pipeline_results(
  std::vector<std::pair<MatmulConfig, std::vector<TimingStats>>> &matmul_results,
  std::ostream &outfile);

/**
 * @brief Logs single-layer matmul benchmark results to a CSV file.
 *
 * Writes timing and performance statistics for each configuration in CSV format
 * to the provided output stream. Includes optional timing breakdowns if enabled.
 *
 * @param matmul_results Vector of pairs of MatmulConfig and TimingStats.
 * @param outfile Output stream to write CSV results to (e.g., std::ofstream).
 */
void log_results(
  std::vector<std::pair<MatmulConfig, std::vector<TimingStats>>> &matmul_results,
  std::ostream &outfile);

/**
 * @brief Prints single-layer matmul benchmark results as a formatted table.
 *
 * Outputs timing and performance statistics for each configuration in a human-readable
 * table format to the provided output stream. Includes optional timing breakdowns if enabled.
 *
 * @param matmul_results Vector of pairs of MatmulConfig and TimingStats.
 * @param outfile Output stream to print table results to (e.g., std::cout).
 */
void print_results(
  std::vector<std::pair<MatmulConfig, std::vector<TimingStats>>> &matmul_results,
  std::ostream &outfile);

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
 * @return int Returns OK (0) on success, NOT_OK (1) on failure.
 */
int bench(const std::string &in_filename, const std::string &out_filename);

} // namespace matmul
} // namespace benchdnn
} // namespace zendnnl

#endif