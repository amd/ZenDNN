/********************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef _BENCHDNN_REORDER_UTILS_HPP_
#define _BENCHDNN_REORDER_UTILS_HPP_

#include "benchdnn.hpp"

namespace zendnnl {
namespace benchdnn {
namespace reorder {

using zendnnl::lowoha::reorder::reorder_algo_t;

/**
 * @struct ReorderConfig
 * @brief Holds user-defined parameters for configuring and benchmarking reorder operations.
 *
 * Contains both regular (AOCL layout) reorder fields and LOWOHA-specific
 * quantization/type-conversion fields. LOWOHA fields have default values
 * to maintain backward compatibility.
 */
struct ReorderConfig {
  size_t rows;
  size_t cols;
  int iters;
  zendnnl::common::data_type_t dt;
  std::string kernel_name;
  bool isInplace;
  int warmup_iters;

  size_t batch_size = 1;
  zendnnl::common::data_type_t src_dtype = zendnnl::common::data_type_t::f32;
  zendnnl::common::data_type_t dst_dtype = zendnnl::common::data_type_t::f32;
  std::string algo = "DT";
  std::string scale_granularity = "per_tensor";
  uint64_t group_size = 0;
  bool dynamic_quant = false;
  uint64_t num_threads = 0;
};

/**
 * @brief Converts a string to a reorder_algo_t enum value.
 */
reorder_algo_t strToReorderAlgo(const std::string &str);

/**
 * @brief Converts a reorder_algo_t enum value to its string representation.
 */
std::string reorderAlgoToStr(reorder_algo_t algo);

/**
 * @brief Parses the input file and populates a vector of ReorderConfig structures.
 *
 * Branches on is_lowoha to parse either the regular format or the LOWOHA format.
 *
 * @param infile Reference to an open std::ifstream containing the input configurations.
 * @param configs Reference to a vector of ReorderConfig to be populated.
 * @param is_lowoha If true, parse LOWOHA format; otherwise parse regular format.
 */
void inputParser(std::ifstream &infile, std::vector<ReorderConfig> &configs,
                 bool is_lowoha);

/**
 * @brief Logs a detailed error message for a failed benchmark configuration.
 *
 * @param cfg ReorderConfig for which the benchmark failed.
 * @param is_lowoha If true, logs LOWOHA-specific fields; otherwise regular fields.
 */
void log_benchmark_failure(const ReorderConfig &cfg, bool is_lowoha);

/**
 * @brief Prints reorder benchmark results in a formatted table.
 *
 * Column headers and data fields are determined by isLOWOHA.
 *
 * @param reorder_results Vector of (ReorderConfig, TimingStats) pairs.
 * @param outfile Output stream to print the table.
 * @param isLOWOHA If true, prints LOWOHA-specific columns.
 */
void print_results(std::vector<std::pair<ReorderConfig, TimingStats>>
                   &reorder_results, std::ostream &outfile, const bool isLOWOHA);

/**
 * @brief Logs reorder benchmark results in CSV format.
 *
 * CSV headers and data fields are determined by isLOWOHA.
 *
 * @param reorder_results Vector of (ReorderConfig, TimingStats) pairs.
 * @param outfile Output stream to write CSV data.
 * @param isLOWOHA If true, writes LOWOHA-specific columns.
 */
void log_results(std::vector<std::pair<ReorderConfig, TimingStats>>
                 &reorder_results, std::ostream &outfile, const bool isLOWOHA);

} // namespace reorder
} // namespace benchdnn
} // namespace zendnnl

#endif
