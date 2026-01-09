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

#ifndef _LOWOHA_POOLING_ONEDNN_KERNEL_HPP
#define _LOWOHA_POOLING_ONEDNN_KERNEL_HPP

#include "lowoha_pooling_common.hpp"
#include "lowoha_pooling_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace pooling {

/**
 * @brief Wrapper function for OneDNN-based Pooling
 *
 * This function implements pooling using OneDNN backend.
 * It handles:
 * - Data layout conversion (NHWC to NCHW if needed)
 * - Memory descriptor creation
 * - Pooling primitive setup (max or average)
 * - Execution and synchronization
 *
 * @param input   Input tensor [N, H, W, C] in NHWC format
 * @param output  Output tensor [N, H_out, W_out, C]
 * @param params  Pooling parameters (dimensions, strides, padding, data types)
 *
 * @return status_t::success on successful execution, status_t::failure otherwise
 */
status_t pooling_onednn_wrapper(
    const void *input,
    void *output,
    pool_params &params
);

} // namespace pooling
} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_POOLING_ONEDNN_KERNEL_HPP

