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

#ifndef _LOWOHA_SOFTMAX_ONEDNN_KERNEL_HPP
#define _LOWOHA_SOFTMAX_ONEDNN_KERNEL_HPP

#include "lowoha_softmax_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace softmax {

/**
 * @brief Wrapper function for OneDNN-based Softmax
 *
 * This function implements softmax using OneDNN backend.
 * It handles:
 * - Memory descriptor creation
 * - Softmax primitive setup
 * - Execution and synchronization
 *
 * @param input                     Input tensor
 * @param output                    Output tensor
 * @param softmax_params            Softmax parameters (dims, softmax params, data types)
 *
 * @return status_t::success on successful execution, status_t::failure otherwise
 */
status_t softmax_onednn_wrapper(
    const void *input,
    void *output,
    const softmax_params &params
);

} // namespace softmax
} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_SOFTMAX_ONEDNN_KERNEL_HPP
