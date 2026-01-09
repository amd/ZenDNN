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

#ifndef _LOWOHA_POOLING_UTILS_HPP
#define _LOWOHA_POOLING_UTILS_HPP

#include <cstdint>
#include <string>
#include "common/zendnnl_global.hpp"
#include "common/logging.hpp"
#include "lowoha_pooling_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace pooling {

using namespace zendnnl::common;

/**
 * @brief Validate Pooling inputs
 *
 * @param input        Input tensor pointer
 * @param output       Output tensor pointer
 * @param params       Pooling parameters
 * @return status_t::success if valid, status_t::failure otherwise
 */
status_t validate_pooling_inputs(
    const void *input,
    const void *output,
    pool_params &params
);

} // namespace pooling
} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_POOLING_UTILS_HPP
