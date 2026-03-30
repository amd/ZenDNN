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
#ifndef _REORDER_TENSOR_FACTORY_HPP_
#define _REORDER_TENSOR_FACTORY_HPP_

#include "benchdnn.hpp"

namespace zendnnl {
namespace benchdnn {
namespace reorder {

using namespace zendnnl::examples;
struct ReorderConfig;

/**
 * @brief Computes the quantization dimension vector from the ReorderConfig granularity.
 *
 * Maps scale_granularity strings (per_tensor, per_channel_row, per_channel_col,
 * per_group_row, per_group_col) to an int64_t dimension vector following the
 * convention in lowoha_reorder_common.hpp.
 *
 * @param cfg ReorderConfig with rows, cols, batch_size, scale_granularity, group_size.
 * @return std::vector<int64_t> Quantization dimensions for scale/zero-point tensors.
 */
std::vector<int64_t> compute_quant_dims(const ReorderConfig &cfg);

/**
 * @brief Creates the source tensor for the reorder benchmark.
 *
 * For regular reorder: creates a uniform tensor with cfg.dt.
 * For LOWOHA: creates a uniform_dist tensor with cfg.src_dtype,
 * optionally 3D when batch_size > 1.
 *
 * @param tensor_factory Factory object for tensor creation.
 * @param cfg ReorderConfig specifying tensor dimensions and data types.
 * @param src Reference to the source tensor to be created.
 * @param is_lowoha If true, uses LOWOHA-specific data types and dimensions.
 * @return int OK (0) on success, NOT_OK (1) on failure.
 */
int create_src_tensor(tensor_factory_t &tensor_factory,
                      const ReorderConfig &cfg,
                      tensor_t &src, bool is_lowoha);

/**
 * @brief Creates the destination tensor for the LOWOHA reorder benchmark.
 *
 * Zero-initialized tensor with cfg.dst_dtype, optionally 3D when batch_size > 1.
 *
 * @param tensor_factory Factory object for tensor creation.
 * @param cfg ReorderConfig specifying tensor dimensions and data types.
 * @param dst Reference to the destination tensor to be created.
 * @return int OK (0) on success, NOT_OK (1) on failure.
 */
int create_dst_tensor(tensor_factory_t &tensor_factory,
                      const ReorderConfig &cfg,
                      tensor_t &dst);

/**
 * @brief Creates the scale tensor for the LOWOHA reorder benchmark.
 *
 * For static quantization: creates a uniform_dist tensor with f32 values.
 * For dynamic quantization: creates a zero-initialized tensor (to be filled at runtime).
 *
 * @param tensor_factory Factory object for tensor creation.
 * @param cfg ReorderConfig specifying quantization granularity and dynamic_quant flag.
 * @param scale Reference to the scale tensor to be created.
 * @return int OK (0) on success, NOT_OK (1) on failure.
 */
int create_scale_tensor(tensor_factory_t &tensor_factory,
                        const ReorderConfig &cfg,
                        tensor_t &scale);

/**
 * @brief Creates the zero-point tensor for the LOWOHA reorder benchmark.
 *
 * Zero-initialized s32 tensor with dimensions matching the quantization granularity.
 *
 * @param tensor_factory Factory object for tensor creation.
 * @param cfg ReorderConfig specifying quantization granularity.
 * @param zp Reference to the zero-point tensor to be created.
 * @return int OK (0) on success, NOT_OK (1) on failure.
 */
int create_zp_tensor(tensor_factory_t &tensor_factory, const ReorderConfig &cfg,
                     tensor_t &zp);

} // namespace reorder
} // namespace benchdnn
} // namespace zendnnl

#endif