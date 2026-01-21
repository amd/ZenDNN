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

#include "lowoha_sdpa_utils.hpp"
#include <cmath>
#include <sstream>

namespace zendnnl {
namespace lowoha {
namespace sdpa {

float calculate_default_scale(uint64_t head_dim) {
  return 1.0f / std::sqrt(static_cast<float>(head_dim));
}

status_t validate_sdpa_inputs(
  const void *query,
  const void *key,
  const void *value,
  const void *output,
  sdpa_params &params
) {
  // Check for null pointers
  if (query == nullptr) {
    log_error("SDPA: Query pointer is null");
    return status_t::failure;
  }

  if (key == nullptr) {
    log_error("SDPA: Key pointer is null");
    return status_t::failure;
  }

  if (value == nullptr) {
    log_error("SDPA: Value pointer is null");
    return status_t::failure;
  }

  if (output == nullptr) {
    log_error("SDPA: Output pointer is null");
    return status_t::failure;
  }

  // Validate dimensions
  if (params.batch == 0) {
    log_error("SDPA: Batch size cannot be zero");
    return status_t::failure;
  }

  if (params.num_heads == 0) {
    log_error("SDPA: Number of heads cannot be zero");
    return status_t::failure;
  }

  if (params.seq_len == 0) {
    log_error("SDPA: Sequence length cannot be zero");
    return status_t::failure;
  }

  if (params.head_dim == 0) {
    log_error("SDPA: Head dimension cannot be zero");
    return status_t::failure;
  }

  // Validate dropout probability
  if (params.dropout_p < 0.0f || params.dropout_p > 1.0f) {
    log_error("SDPA: Dropout probability must be between 0 and 1");
    return status_t::failure;
  }

  return status_t::success;
}

} // namespace sdpa
} // namespace lowoha
} // namespace zendnnl
