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

#ifndef _LOWOHA_CONV_UTILS_HPP
#define _LOWOHA_CONV_UTILS_HPP

#include <cstdint>
#include <string>
#include "lowoha_conv_common.hpp"
#include "common/zendnnl_global.hpp"
#include "common/logging.hpp"

namespace zendnnl {
namespace lowoha {
namespace conv {

using namespace zendnnl::common;

/**
 * @brief Validate Conv inputs
 *
 * @param input        Input tensor pointer
 * @param filter       Filter tensor pointer
 * @param output       Output tensor pointer
 * @param params       Convolution parameters
 * @return status_t::success if valid, status_t::failure otherwise
 */
status_t validate_conv_inputs(
    const void *input,
    const void *filter,
    const void *output,
    conv_params &params
);

} // namespace conv
} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_CONV_UTILS_HPP
