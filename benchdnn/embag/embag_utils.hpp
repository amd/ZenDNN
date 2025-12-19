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
#ifndef _EMBAG_UTILS_HPP_
#define _EMBAG_UTILS_HPP_

#include "benchdnn.hpp"

namespace zendnnl {
namespace benchdnn {
namespace embag {

/**
 * @struct EmbagConfig
 * @brief Holds user-defined parameters for configuring and benchmarking Embag operations.
 *
 * This structure encapsulates all the parameters required to configure a Embag
 * benchmark, including table dimensions, number of indices, bags, type of algorithm,
 * number of iterations, data type, padding index, include last offset, is weights flag,
 * scatter stride, and warmup iterations.
 *
 * @var num_embeddings Size of the dictionary of embeddings.
 * @var embedding_dim Size of each embedding vector.
 * @var num_bags Number of bags (groups of indices) used in embedding bag operation.
 * @var num_indices Total number of indices across all bags.
 * @var algo Algorithm used for embag computation (e.g., "sum", "mean", "max").
 * @var iters Number of iterations to run the benchmark.
 * @var dt Data types for table and output (e.g., f32:f32).
 * @var fp16_scale_bias Flag indicating the data type of scale and bias.
 * @var padding_index Index used for padding; ignored during computation.
 * @var include_last_offset Flag indicating whether to include the last offset in the offsets array.
 * @var is_weights Flag indicating if weights are used for each index in the embag.
 * @var scatter_stride Stride used when scattering embeddings in memory.
 * @var warmup_iters Number of warmup iterations to run before actual benchmarking.
 */
struct EmbagConfig {
  size_t num_embeddings; /**< Size of the dictionary of embeddings. */
  size_t embedding_dims; /**< Size of each embedding vector. */
  size_t num_bags; /**< Number of bags (groups of indices) used in embedding bag operation. */
  size_t num_indices; /**< Total number of indices across all bags. */
  zendnnl::ops::embag_algo_t algo; /**< Algorithm used for embag
                                   computation (e.g., "sum", "mean", "max"). */
  int iters; /**< Number of iterations to run the benchmark. */
  std::vector<zendnnl::common::data_type_t> dt; /**< Data type for
                                                input and output (e.g., f32:f32). */
  bool fp16_scale_bias; /**< Flag indicating the data type of scale and bias. */
  int64_t padding_index; /**< Index used for padding; ignored during computation. */
  bool include_last_offset; /**< Flag indicating whether to include the last offset in the offsets array. */
  bool is_weights; /**< Flag indicating if weights are used for each index in the embag. */
  int64_t scatter_stride; /**< Stride used when scattering embeddings in memory. */
  int warmup_iters; /**< Number of warmup iterations to run before actual benchmarking. */
};

embag_algo_t strToEmbagalgo(const std::string &algo_str);

std::string embagalgoToStr(embag_algo_t algo);

/**
 * @brief Parses the input file and populates a vector of EmbagConfig structures.
 *
 * Reads each line from the provided input file stream, parses the embag
 * configuration parameters, and appends a EmbagConfig object to the configs vector.
 *
 * @param infile Reference to an open std::ifstream containing the input configurations.
 * @param configs Reference to a vector of EmbagConfig to be populated.
 */
void inputParser(std::ifstream &infile, std::vector<EmbagConfig> &configs);

/**
* @brief Logs a detailed error message for a failed benchmark configuration.
*
* Prints configuration parameters to the error log for debugging failed runs.
*
* @param cfg EmbagConfig structure for which the benchmark failed.
*/
void log_benchmark_failure(const EmbagConfig &cfg);

/**
 * @brief Prints the embag benchmark results in a formatted table to the given output stream.
 *
 * Dynamically calculates column widths for neat alignment and prints a table of results for each configuration.
 * Includes detailed timing breakdowns if enabled at compile time.
 *
 * @param embag_results Vector of pairs (EmbagConfig, TimingStats) containing benchmark results.
 * @param outfile Output stream to print the table (e.g., std::cout or file stream).
 */
void print_results(std::vector<std::pair<EmbagConfig, TimingStats>>
                   &embag_results, std::ostream &outfile);

/**
 * @brief Logs the embag benchmark results in CSV format to the given output stream.
 *
 * Writes a CSV header and a row for each configuration, including timing breakdowns if enabled.
 *
 * @param embag_results Vector of pairs (EmbagConfig, TimingStats) containing benchmark results.
 * @param outfile Output stream to write the CSV data (e.g., file stream).
 */
void log_results(std::vector<std::pair<EmbagConfig, TimingStats>>
                 &embag_results, std::ostream &outfile);

} // namespace embag
} // namespace benchdnn
} // namespace zendnnl

#endif