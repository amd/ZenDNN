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

#ifndef _LOWOHA_SDPA_UTILS_HPP
#define _LOWOHA_SDPA_UTILS_HPP

#include <cstdint>
#include <string>
#include "common/zendnnl_global.hpp"
#include "common/logging.hpp"
#include "lowoha_operators/sdpa/lowoha_sdpa_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace sdpa {

using namespace zendnnl::common;

/**
 * @brief Validate SDPA inputs
 *
 * @param query        Query tensor pointer
 * @param key          Key tensor pointer
 * @param value        Value tensor pointer
 * @param output       Output tensor pointer
 * @param params       SDPA parameters
 * @return status_t::success if valid, status_t::failure otherwise
 */
status_t validate_sdpa_inputs(
  const void *query,
  const void *key,
  const void *value,
  const void *output,
  sdpa_params &params
);

/**
 * @brief Calculate default scale factor (1/sqrt(head_dim))
 *
 * @param head_dim     Head dimension
 * @return Default scale factor
 */
float calculate_default_scale(uint64_t head_dim);

} // namespace sdpa
} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_SDPA_UTILS_HPP
