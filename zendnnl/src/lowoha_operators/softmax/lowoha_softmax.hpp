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

#ifndef _LOWOHA_SOFTMAX_HPP
#define _LOWOHA_SOFTMAX_HPP

#include <cmath>
#include <cstring>
#include "lowoha_operators/softmax/lowoha_softmax_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace softmax {
/**
 * @brief Execute softmax with unified parameters structure
 *
 * Performs: output = softmax(input) along specified axis
 *
 * Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
 *
 * @param input            Input tensor
 * @param output           Output tensor (same shape as input)
 * @param params           Softmax parameters (dims, softmax params, data types)
 *
 * @return status_t::success or status_t::failure
 */
status_t softmax_direct(
    const void *input,
    void *output,
    softmax_params &params
);

/**
 * @brief Kernel dispatcher - selects appropriate backend
 *
 * @param input                   Input tensor data
 * @param output                  Output tensor data
 * @param softmax_params          Softmax parameters (dims, softmax params, data types)
 */
void softmax_kernel_wrapper(
    const void *input,
    void *output,
    softmax_params &params
);

} // namespace softmax
} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_SOFTMAX_HPP
