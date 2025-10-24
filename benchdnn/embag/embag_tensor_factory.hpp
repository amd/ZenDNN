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
#ifndef _EMBAG_TENSOR_FACTORY_HPP_
#define _EMBAG_TENSOR_FACTORY_HPP_

#include "benchdnn.hpp"

namespace zendnnl {
namespace benchdnn {
namespace embag {

using namespace zendnnl::examples;
struct EmbagConfig;

/**
 * @brief Creates table tensor for the embag benchmark.
 *
 * Populates the table vector with uniform values.
 *
 * @param tensor_factory Factory object for tensor creation.
 * @param cfg EmbagConfig structure specifying tensor dimensions and data types.
 * @param table Vector to store created table tensors.
 * @return int OK (0) on success, NOT_OK (1) on failure.
 */
int create_table_tensor(tensor_factory_t &tensor_factory, EmbagConfig cfg,
                        tensor_t &table);

/**
 * @brief Creates indices tensor for the embag benchmark.
 *
 * Populates the indices vector with random values.
 *
 * @param tensor_factory Factory object for tensor creation.
 * @param cfg EmbagConfig structure specifying tensor dimensions and data types.
 * @param indices Vector to store created indices values.
 * @return int OK (0) on success, NOT_OK (1) on failure.
 */
int create_indices_tensor(tensor_factory_t &tensor_factory, EmbagConfig cfg,
                          tensor_t &indices);

/**
 * @brief Creates offsets tensor for the embag benchmark.
 *
 * Populates the offsets vector with random values.
 *
 * @param tensor_factory Factory object for tensor creation.
 * @param cfg EmbagConfig structure specifying tensor dimensions and data types.
 * @param offsets Vector to store created offsets values.
 * @return int OK (0) on success, NOT_OK (1) on failure.
 */
int create_offsets_tensor(tensor_factory_t &tensor_factory, EmbagConfig cfg,
                          tensor_t &offsets);

/**
 * @brief Creates weight tensor for the embag benchmark.
 *
 * Populates the weights vector with random values.
 *
 * @param tensor_factory Factory object for tensor creation.
 * @param cfg EmbagConfig structure specifying tensor dimensions and data types.
 * @param weights Vector to store created weights tensor.
 * @return int OK (0) on success, NOT_OK (1) on failure.
 */
int create_weights_tensor(tensor_factory_t &tensor_factory, EmbagConfig cfg,
                          tensor_t &weights);

/**
 * @brief Creates output tensors for the embag benchmark.
 *
 * Populates the output vector with zero-initialized tensors of appropriate shape and data type.
 *
 * @param tensor_factory Factory object for tensor creation.
 * @param cfg EmbagConfig structure specifying tensor dimensions and data types.
 * @param output Vector to store created output tensors.
 * @return int OK (0) on success, NOT_OK (1) on failure.
 */
int create_output_tensor(tensor_factory_t &tensor_factory, EmbagConfig cfg,
                         tensor_t &output);

} // namespace embag
} // namespace benchdnn
} // namespace zendnnl

#endif