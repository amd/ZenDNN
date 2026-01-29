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

#ifndef _LOWOHA_CONV_HPP
#define _LOWOHA_CONV_HPP

#include <cmath>
#include <cstring>
#include "lowoha_conv_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace conv {

/**
 * @brief Execute convolution with automatic kernel selection
 *
 * Performs: output = conv(input, filter) + bias + post-ops
 *
 * @param input            Input tensor [N, H, W, C] (NHWC format)
 * @param filter           Filter tensor [KH, KW, C_in, C_out]
 * @param bias             Optional bias [C_out]
 * @param output           Output tensor [N, H_out, W_out, C_out]
 * @param is_weights_const Flag indicating if weights are constant (enables caching)
 * @param params           Convolution parameters
 *
 * @return status_t::success or status_t::failure
 */
status_t conv_direct(
    const void *input,           // [N, H, W, C]
    const void *filter,          // [KH, KW, C_in, C_out]
    const void *bias,            // [C_out] or nullptr
    void *output,                // [N, H_out, W_out, C_out]
    const bool is_weights_const, // Enable weight caching for constant filters
    conv_params &params
);

/**
 * @brief Kernel dispatcher - selects appropriate backend
 *
 * @param input            Input tensor data
 * @param filter           Filter tensor data
 * @param bias             Bias tensor data (optional)
 * @param output           Output tensor data
 * @param is_weights_const Flag indicating if weights are constant
 * @param params           Convolution parameters
 * @return status_t::success or status_t::failure
 */
status_t conv_kernel_wrapper(
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

#endif // _LOWOHA_CONV_HPP
