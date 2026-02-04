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

#ifndef _LOWOHA_CONV_REFERENCE_KERNEL_HPP
#define _LOWOHA_CONV_REFERENCE_KERNEL_HPP

#include "lowoha_conv_common.hpp"
#include "lowoha_conv_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace conv {

/**
 * @brief Reference implementation for Conv2D
 *
 * Provides a simple, correct implementation of 2D convolution for validation
 * and debugging purposes. This kernel implements:
 *   - Standard 2D convolution
 *   - Depthwise convolution
 *   - Grouped convolution
 *   - Strided convolution
 *   - Dilated (atrous) convolution
 *   - Bias addition
 *   - Post-operations (ReLU, ReLU6, etc.)
 *
 * @param input            Input tensor [N, H, W, C] (NHWC format)
 * @param filter           Filter tensor [KH, KW, C_in, C_out]
 * @param bias             Bias tensor [C_out] (optional, can be nullptr)
 * @param output           Output tensor [N, H_out, W_out, C_out]
 * @param is_weights_const Flag indicating if weights are constant
 * @param params           Convolution parameters (dimensions, strides, padding, data types)
 *
 * @return status_t::success on successful execution, status_t::failure otherwise
 */
status_t conv_reference_wrapper(
    const void *input,
    const void *filter,
    const void *bias,
    void *output,
    const bool is_weights_const,
    conv_params &params
);

} // namespace conv
} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_CONV_REFERENCE_KERNEL_HPP
