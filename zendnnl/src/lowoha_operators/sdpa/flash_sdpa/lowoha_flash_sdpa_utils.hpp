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

#ifndef LOWOHA_FLASH_SDPA_UTILS_HPP
#define LOWOHA_FLASH_SDPA_UTILS_HPP

#include <cstdint>
#include "common/zendnnl_global.hpp"
#include "common/logging.hpp"
#include "lowoha_operators/sdpa/lowoha_sdpa_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace sdpa {

using namespace zendnnl::common;

/**
 * @brief Validate flash SDPA inputs
 *
 * @param query        Query tensor pointer
 * @param key          Key tensor pointer
 * @param value        Value tensor pointer
 * @param output       Output tensor pointer
 * @param attn_mask    Attention mask pointer
 * @param params       Flash SDPA parameters
 * @return status_t::success if valid, status_t::failure otherwise
 */
status_t validate_flash_sdpa_inputs(
  const void *query,
  const void *key,
  const void *value,
  void *output,
  const void *attn_mask,
  const sdpa_params &params
);

} // namespace sdpa
} // namespace lowoha
} // namespace zendnnl

#endif // LOWOHA_FLASH_SDPA_UTILS_HPP
