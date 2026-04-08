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
#ifndef _NORMALIZATION_UTILS_HPP_
#define _NORMALIZATION_UTILS_HPP_

#include "benchdnn.hpp"

#define NORM_REQUIRED_FIELD_COUNT 8

using zendnnl::lowoha::normalization::norm_type_t;
using zendnnl::lowoha::normalization::norm_algo_t;

namespace zendnnl {
namespace benchdnn {
namespace normalization {

/**
 * @struct NormalizationConfig
 * @brief Holds user-defined parameters for configuring and benchmarking normalization operations.
 *
 * This structure encapsulates all the parameters required to configure a normalization
 * benchmark, including tensor shape, normalization type, data types, epsilon, scale/shift
 * flags, and iteration counts.
 *
 * @var norm_type Normalization variant: "layer_norm", "batch_norm", "rms_norm", or "fused_add_rms_norm".
 * @var shape Tensor dimensions (e.g., {batch, hidden_dim} or {N, C, H, W}).
 * @var norm_ndims Number of trailing dimensions to normalize (LayerNorm/RMSNorm/FusedAddRMSNorm).
 * @var src_dt Source (input) data type (f32 or bf16).
 * @var dst_dt Destination (output) data type (f32 or bf16).
 * @var gamma_dt Gamma (scale) parameter data type (default: f32).
 * @var beta_dt Beta (shift) parameter data type (default: f32).
 * @var epsilon Numerical stability constant (e.g., 1e-5).
 * @var use_scale Whether to apply learned scale (gamma).
 * @var use_shift Whether to apply learned shift (beta); ignored by RMSNorm/FusedAddRMSNorm.
 * @var algorithm Backend selection: "none" (auto), "dynamic_dispatch", or "reference".
 * @var iters Number of benchmark iterations.
 * @var warmup_iters Number of warmup iterations before benchmarking.
 * @var num_threads Number of threads for parallel execution (0 = auto/all available).
 * @var isInplace Whether to perform in-place normalization (output overwrites input buffer).
 *                Requires src_dt == dst_dt.
 */
struct NormalizationConfig {
  std::string norm_type;
  std::vector<uint64_t> shape;
  int norm_ndims;
  zendnnl::common::data_type_t src_dt;
  zendnnl::common::data_type_t dst_dt;
  zendnnl::common::data_type_t gamma_dt;
  zendnnl::common::data_type_t beta_dt;
  float epsilon;
  bool use_scale;
  bool use_shift;
  std::string algorithm;
  int iters;
  int warmup_iters;
  int num_threads;
  bool isInplace;
};

/**
 * @brief Converts a normalization type string to its canonical form.
 *
 * Accepts strings like "layer_norm", "batch_norm", "rms_norm", "fused_add_rms_norm"
 * (case-insensitive) and returns the canonical lowercase form.
 *
 * @param str Input string representing the normalization type.
 * @return std::string Canonical normalization type string, or empty string if unknown.
 */
std::string strToNormType(const std::string &str);

/**
 * @brief Parses a shape string (e.g., "2x4096" or "32x64x56x56") into a vector of dimensions.
 *
 * @param shape_str Shape string with dimensions separated by 'x'.
 * @return std::vector<uint64_t> Parsed dimensions.
 */
std::vector<uint64_t> parseShape(const std::string &shape_str);

/**
 * @brief Computes the product of the last norm_ndims dimensions of the shape.
 *
 * @param cfg NormalizationConfig containing shape and norm_ndims.
 * @return uint64_t The normalization size.
 */
uint64_t compute_norm_size(const NormalizationConfig &cfg);

/**
 * @brief Extracts the number of channels (second dimension) from the shape.
 *
 * @param cfg NormalizationConfig containing the tensor shape.
 * @return uint64_t The number of channels, or 0 if shape has fewer than 2 dimensions.
 */
uint64_t compute_num_channels(const NormalizationConfig &cfg);

/**
 * @brief Converts a normalization type string to the LOWOHA norm_type_t enum.
 *
 * @param norm_type Canonical normalization type string (e.g., "layer_norm").
 * @return norm_type_t The corresponding LOWOHA enum value, or norm_type_t::NONE if unknown.
 */
norm_type_t strToLowohaType(const std::string &norm_type);

/**
 * @brief Converts an algorithm string to the LOWOHA norm_algo_t enum.
 *
 * @param algo Algorithm string (e.g., "dynamic_dispatch", "reference", "none").
 * @return norm_algo_t The corresponding LOWOHA enum value, or norm_algo_t::none if unknown.
 */
norm_algo_t strToLowohaAlgo(const std::string &algo);

/**
 * @brief Parses the input file and populates a vector of NormalizationConfig structures.
 *
 * Each line in the input file specifies a benchmark configuration in CSV format:
 *   norm_type, shape, norm_ndims, src_dt:dst_dt, epsilon, use_scale, use_shift, iters
 *   [, warmup_iters, gamma_dt, beta_dt, algorithm, num_threads, isInplace]
 *
 * @param infile Reference to an open std::ifstream containing the input configurations.
 * @param configs Reference to a vector of NormalizationConfig to be populated.
 */
void inputParser(std::ifstream &infile,
                 std::vector<NormalizationConfig> &configs);

/**
 * @brief Logs a detailed error message for a failed benchmark configuration.
 *
 * @param cfg NormalizationConfig structure for which the benchmark failed.
 */
void log_benchmark_failure(const NormalizationConfig &cfg);

/**
 * @brief Prints the normalization benchmark results in a formatted table to the given output stream.
 *
 * @param normalization_results Vector of pairs (NormalizationConfig, TimingStats) containing benchmark results.
 * @param outfile Output stream to print the table (e.g., std::cout or file stream).
 */
void print_results(std::vector<std::pair<NormalizationConfig, TimingStats>>
                   &normalization_results, std::ostream &outfile);

/**
 * @brief Logs the normalization benchmark results in CSV format to the given output stream.
 *
 * @param normalization_results Vector of pairs (NormalizationConfig, TimingStats) containing benchmark results.
 * @param outfile Output stream to write the CSV data (e.g., file stream).
 */
void log_results(std::vector<std::pair<NormalizationConfig, TimingStats>>
                 &normalization_results, std::ostream &outfile);

} // namespace normalization
} // namespace benchdnn
} // namespace zendnnl

#endif
