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

#ifndef _LOWOHA_CONV_ONEDNN_KERNEL_HPP
#define _LOWOHA_CONV_ONEDNN_KERNEL_HPP

#include "lowoha_conv_utils.hpp"
#include "lowoha_conv_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace conv {

/**
 * @brief Wrapper function for OneDNN-based Conv
 *
 * This function implements convolution using OneDNN backend.
 * It handles:
 * - Data layout conversion (NHWC to NCHW)
 * - Memory descriptor creation
 * - Convolution primitive setup
 * - Post-operation fusion (Relu, etc.)
 * - Execution and synchronization
 *
 * @param input            Input tensor [N, H, W, C] in NHWC format
 * @param filter           Filter tensor [KH, KW, C_in, C_out]
 * @param bias             Optional bias [C_out]
 * @param output           Output tensor [N, H_out, W_out, C_out]
 * @param params           Convolution parameters
 *
 * @return status_t::success on successful execution, status_t::failure otherwise
 */
status_t conv_onednn_wrapper(
    const void *input,
    const void *filter,
    const void *bias,
    void *output,
    conv_params &params
);

} // namespace conv
} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_CONV_ONEDNN_KERNEL_HPP
