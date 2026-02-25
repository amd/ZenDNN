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

#ifndef _LOWOHA_NORMALIZATION_COMMON_HPP
#define _LOWOHA_NORMALIZATION_COMMON_HPP

#include <cstdint>
#include <vector>
#include "memory/memory_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace normalization {

using namespace zendnnl::common;

/**
 * @brief Normalization type
 *
 * Specifies which normalization variant to apply.
 *
 * - LAYER_NORM:  Normalizes across the last `norm_ndims` dimensions for each sample.
 *                Mean and variance are computed on-the-fly from the input.
 *                Formula: y = gamma * (x - mean) / sqrt(var + eps) + beta
 *
 * - RMS_NORM:    Root Mean Square normalization across the last `norm_ndims` dimensions.
 *                RMS is computed on-the-fly from the input. No mean subtraction, no beta.
 *                Formula: y = gamma * x / sqrt(mean(x^2) + eps)
 *
 * - BATCH_NORM:  Normalizes per-channel using pre-computed running mean/variance.
 *                Running statistics must be provided (from training).
 *                Formula: y = gamma * (x - running_mean) / sqrt(running_var + eps) + beta
 */
enum class norm_type_t : int {
  NONE        = -1,   ///< No normalization type selected
  LAYER_NORM  = 0,    ///< Layer Normalization
  BATCH_NORM  = 1,    ///< Batch Normalization
  RMS_NORM    = 2,    ///< Root Mean Square Normalization
  FUSED_ADD_RMS_NORM = 3  ///< Fused Add and RMS Normalization
};

/**
 * @brief Normalization algorithm / backend selection
 */
enum class norm_algo_t : int {
  none             = -1,  ///< No algorithm selected
  dynamic_dispatch = 0,   ///< Dynamic dispatch - selects best available backend
  reference        = 1    ///< Reference (scalar) implementation
};

/**
 * @brief Maximum supported tensor dimensions for normalization
 */
constexpr int NORM_MAX_NDIMS = 5;

/**
 * @brief Parameter structure for LOWOHA normalization operations
 *
 * This structure contains all parameters needed to configure and execute
 * LayerNorm, RMSNorm, or BatchNorm operations.
 *
 * Tensor layout assumptions:
 *   - LayerNorm / RMSNorm: Input shape [batch_dims..., normalized_dims...]
 *     Mean/variance (or RMS) computed on-the-fly from the input.
 *   - BatchNorm: Input shape [N, C, spatial_dims...]
 *     Uses pre-computed running_mean and running_var (from training).
 */
struct norm_params {
  // --- Normalization variant ---
  norm_type_t norm_type;          ///< Which normalization to apply

  // --- Tensor shape ---
  std::vector<uint64_t>
  shape;    ///< Tensor dimensions (e.g. {batch, hidden_dim})

  // --- Derived / flattened dimensions (populated by setup) ---
  uint64_t batch;                 ///< Outer (batch) size: product of non-normalized dims
  uint64_t norm_size;             ///< Inner (normalized) size: product of normalized dims
  uint64_t num_channels;          ///< Number of channels (used by BatchNorm, dim[1])

  // --- Normalization parameters ---
  float epsilon;                  ///< Small constant for numerical stability (default 1e-5)
  int norm_ndims;                 ///< Number of trailing dims to normalize (LayerNorm/RMSNorm)
  bool use_scale;                 ///< Whether to apply learned scale (gamma)
  bool use_shift;                 ///< Whether to apply learned shift (beta); ignored by RMSNorm

  // --- Data types ---
  data_type_t src_dt;             ///< Source / input data type
  data_type_t dst_dt;             ///< Destination / output data type
  data_type_t gamma_dt;           ///< Gamma (scale) parameter data type
  data_type_t beta_dt;            ///< Beta (shift) parameter data type

  // --- Backend selection ---
  norm_algo_t algorithm;          ///< Selected algorithm / backend

  // --- Threading ---
  uint64_t num_threads;           ///< Number of threads (0 = auto)

  /**
   * @brief Default constructor
   */
  norm_params()
    : norm_type(norm_type_t::NONE),
      batch(1),
      norm_size(0),
      num_channels(0),
      epsilon(1e-5f),
      norm_ndims(1),
      use_scale(false),
      use_shift(false),
      src_dt(data_type_t::none),
      dst_dt(data_type_t::none),
      gamma_dt(data_type_t::f32),
      beta_dt(data_type_t::f32),
      algorithm(norm_algo_t::none),
      num_threads(0) {}
};

} // namespace normalization
} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_NORMALIZATION_COMMON_HPP

