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
#ifndef _MATMUL_TENSOR_FACTORY_HPP_
#define _MATMUL_TENSOR_FACTORY_HPP_

#include "benchdnn.hpp"

namespace zendnnl {
namespace benchdnn {
namespace matmul {

using namespace zendnnl::examples;
struct MatmulConfig;

/**
 * @brief Creates weight tensors for each layer in the matmul benchmark.
 *
 * Handles blocked/reordered layouts if required by the kernel. Populates the weights vector.
 *
 * @param tensor_factory Factory object for tensor creation.
 * @param cfg MatmulConfig structure specifying tensor dimensions and data types.
 * @param weights Vector to store created weight tensors.
 * @param options Global options for command-line configuration.
 * @param isLOWOHA Flag to indicate LOWOHA mode (true) or regular API mode (false).
 *                 When false, reorder is applied for aocl_dlp_blocked kernels.
 * @return int OK (0) on success, NOT_OK (1) on failure.
 */
int create_weights_tensor(tensor_factory_t &tensor_factory, MatmulConfig cfg,
                          std::vector<tensor_t> &weights, const global_options &options,
                          bool isLOWOHA = false);

/**
 * @brief Creates bias tensors for each layer if bias is enabled.
 *
 * Populates the bias vector with tensors of appropriate shape and data type.
 *
 * @param tensor_factory Factory object for tensor creation.
 * @param cfg MatmulConfig structure specifying tensor dimensions and data types.
 * @param bias Vector to store created bias tensors.
 * @param options Global options for command-line configuration.
 * @return int OK (0) on success, NOT_OK (1) on failure.
 */
int create_bias_tensor(tensor_factory_t tensor_factory, const MatmulConfig &cfg,
                       std::vector<tensor_t> &bias, const global_options &options);

/**
 * @brief Creates the input tensor for the matmul benchmark.
 *
 * Populates the input tensor with random or uniform values as specified.
 *
 * @param tensor_factory Factory object for tensor creation.
 * @param cfg MatmulConfig structure specifying tensor dimensions and data types.
 * @param input Reference to input tensor to be created.
 * @param options Global options for command-line configuration.
 * @return int OK (0) on success, NOT_OK (1) on failure.
 */
int create_input_tensor(tensor_factory_t &tensor_factory,
                        const MatmulConfig &cfg, tensor_t &input, const global_options &options);

/**
 * @brief Creates output tensors for each layer in the matmul benchmark.
 *
 * Populates the output vector with zero-initialized tensors of appropriate shape and data type.
 *
 * @param tensor_factory Factory object for tensor creation.
 * @param cfg MatmulConfig structure specifying tensor dimensions and data types.
 * @param output Vector to store created output tensors.
 * @param options Global options for command-line configuration.
 * @return int OK (0) on success, NOT_OK (1) on failure.
 */
int create_output_tensor(tensor_factory_t &tensor_factory,
                         const MatmulConfig &cfg, std::vector<tensor_t> &output,
                         const global_options &options);

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

}
}
}

#endif