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
#ifndef _MATMUL_UTILS_HPP_
#define _MATMUL_UTILS_HPP_

#include "benchdnn.hpp"

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

/**
* @brief Logs a detailed error message for a failed benchmark configuration.
*
* Prints configuration parameters and post-ops to the error log for debugging failed runs.
*
* @param cfg MatmulConfig structure for which the benchmark failed.
*/
void log_benchmark_failure(const MatmulConfig &cfg);

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

} // namespace matmul
} // namespace benchdnn
} // namespace zendnnl

#endif