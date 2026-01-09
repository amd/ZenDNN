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

#ifndef _LOWOHA_POOLING_HPP
#define _LOWOHA_POOLING_HPP

#include <cmath>
#include <cstring>
#include "lowoha_operators/pooling/lowoha_pooling_common.hpp"
#include "lowoha_operators/pooling/lowoha_pooling_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace pooling {

/**
 * @brief Execute pooling with automatic kernel selection
 *
 * Performs pooling operation (max or average) on input tensor
 *
 * Max Pooling: output[i] = max(input[window_i])
 * Average Pooling: output[i] = mean(input[window_i])
 *
 * @param input   Input tensor [N, H, W, C] (NHWC format)
 * @param output  Output tensor [N, H_out, W_out, C]
 * @param params  Unified pooling parameters (dimensions, strides, padding, types, etc.)
 *
 * @return status_t::success or status_t::failure
 */
status_t pooling_direct(
    const void *input,
    void *output,
    pool_params &params
);

/**
 * @brief Execute pooling with automatic kernel selection (legacy interface)
 *
 * Performs pooling operation (max or average) on input tensor.
 * This is the legacy interface maintained for backward compatibility.
 *
 * Max Pooling: output[i] = max(input[window_i])
 * Average Pooling: output[i] = mean(input[window_i])
 *
 * @param input            Input tensor [N, H, W, C] (NHWC format)
 * @param output           Output tensor [N, H_out, W_out, C]
 * @param params           Pooling parameters (kernel, stride, padding, type)
 *
 * @return status_t::success or status_t::failure
 */
void pooling_kernel_wrapper(
    const void *input,
    void *output,
    pool_params &params
);

} // namespace pooling
} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_POOLING_HPP
