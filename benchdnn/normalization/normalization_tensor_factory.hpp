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
#ifndef _NORMALIZATION_TENSOR_FACTORY_HPP_
#define _NORMALIZATION_TENSOR_FACTORY_HPP_

#include "benchdnn.hpp"

namespace zendnnl {
namespace benchdnn {
namespace normalization {

using namespace zendnnl::examples;
struct NormalizationConfig;

/**
 * @brief Creates the input tensor for normalization benchmark.
 *
 * Populates the tensor with uniform random values matching the configured shape and src_dt.
 *
 * @param tensor_factory Factory object for tensor creation.
 * @param cfg NormalizationConfig structure specifying tensor dimensions and data types.
 * @param input Tensor reference to store the created input tensor.
 * @return int OK (0) on success, NOT_OK (1) on failure.
 */
int create_input_tensor(tensor_factory_t &tensor_factory,
                        const NormalizationConfig &cfg, tensor_t &input);

/**
 * @brief Creates the output tensor for normalization benchmark.
 *
 * Populates the tensor with zeros matching the configured shape and dst_dt.
 *
 * @param tensor_factory Factory object for tensor creation.
 * @param cfg NormalizationConfig structure specifying tensor dimensions and data types.
 * @param output Tensor reference to store the created output tensor.
 * @return int OK (0) on success, NOT_OK (1) on failure.
 */
int create_output_tensor(tensor_factory_t &tensor_factory,
                         const NormalizationConfig &cfg, tensor_t &output);

/**
 * @brief Creates the gamma (scale) parameter tensor.
 *
 * Shape depends on norm type:
 *   - LayerNorm / RMSNorm / FusedAddRMSNorm: [norm_size]
 *   - BatchNorm: [num_channels]
 * Always FP32. Populated with uniform random values.
 *
 * @param tensor_factory Factory object for tensor creation.
 * @param cfg NormalizationConfig structure specifying tensor dimensions.
 * @param gamma Tensor reference to store the created gamma tensor.
 * @return int OK (0) on success, NOT_OK (1) on failure.
 */
int create_gamma_tensor(tensor_factory_t &tensor_factory,
                        const NormalizationConfig &cfg, tensor_t &gamma);

/**
 * @brief Creates the beta (shift) parameter tensor.
 *
 * Shape depends on norm type:
 *   - LayerNorm: [norm_size]
 *   - BatchNorm: [num_channels]
 *   - RMSNorm / FusedAddRMSNorm: unused (empty tensor returned)
 * Always FP32. Populated with uniform random values.
 *
 * @param tensor_factory Factory object for tensor creation.
 * @param cfg NormalizationConfig structure specifying tensor dimensions.
 * @param beta Tensor reference to store the created beta tensor.
 * @return int OK (0) on success, NOT_OK (1) on failure.
 */
int create_beta_tensor(tensor_factory_t &tensor_factory,
                       const NormalizationConfig &cfg, tensor_t &beta);

/**
 * @brief Creates the running mean tensor (BatchNorm only).
 *
 * Shape: [num_channels]. Always FP32. Populated with uniform random values.
 * Returns an empty tensor for non-BatchNorm types.
 *
 * @param tensor_factory Factory object for tensor creation.
 * @param cfg NormalizationConfig structure specifying tensor dimensions.
 * @param running_mean Tensor reference to store the created tensor.
 * @return int OK (0) on success, NOT_OK (1) on failure.
 */
int create_running_mean_tensor(tensor_factory_t &tensor_factory,
                               const NormalizationConfig &cfg,
                               tensor_t &running_mean);

/**
 * @brief Creates the running variance tensor (BatchNorm only).
 *
 * Shape: [num_channels]. Always FP32. Populated with positive uniform random values.
 * Returns an empty tensor for non-BatchNorm types.
 *
 * @param tensor_factory Factory object for tensor creation.
 * @param cfg NormalizationConfig structure specifying tensor dimensions.
 * @param running_var Tensor reference to store the created tensor.
 * @return int OK (0) on success, NOT_OK (1) on failure.
 */
int create_running_var_tensor(tensor_factory_t &tensor_factory,
                              const NormalizationConfig &cfg,
                              tensor_t &running_var);

/**
 * @brief Creates the residual tensor (FusedAddRMSNorm only).
 *
 * Same shape and data type as input (src_dt). Populated with uniform random values.
 * Returns an empty tensor for non-FusedAddRMSNorm types.
 *
 * @param tensor_factory Factory object for tensor creation.
 * @param cfg NormalizationConfig structure specifying tensor dimensions.
 * @param residual Tensor reference to store the created tensor.
 * @return int OK (0) on success, NOT_OK (1) on failure.
 */
int create_residual_tensor(tensor_factory_t &tensor_factory,
                           const NormalizationConfig &cfg,
                           tensor_t &residual);

} // namespace normalization
} // namespace benchdnn
} // namespace zendnnl

#endif
