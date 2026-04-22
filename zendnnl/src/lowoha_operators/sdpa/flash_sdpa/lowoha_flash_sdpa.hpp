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

#ifndef LOWOHA_FLASH_SDPA_HPP
#define LOWOHA_FLASH_SDPA_HPP

#include <cmath>
#include <cstring>
#include "lowoha_operators/sdpa/flash_sdpa/lowoha_flash_sdpa_utils.hpp"
#include "lowoha_operators/sdpa/flash_sdpa/lowoha_sdpa_flash_cpu.hpp"

namespace zendnnl {
namespace lowoha {
namespace sdpa {

/**
 * @brief Flash-style SDPA on CPU (LowOHA matmul + fused softmax).
 *
 * Inference-only: no logsumexp output (not needed without backward pass).
 * Tensor layout and strides are described by @c sdpa_params.
 *
 * @param query     Query data  [batch, num_heads, seq_len, head_dim]
 * @param key       Key data    [batch, num_heads, seq_len, head_dim]
 * @param value     Value data  [batch, num_heads, seq_len, head_dim]
 * @param attn_mask Optional attention mask (nullptr if none)
 * @param output    Output data [batch, num_heads, seq_len, head_dim]
 * @param params    Shapes, strides, dtypes, scale, flags
 *
 * @return status_t::success or status_t::failure
 */
status_t sdpa_flash_cpu_standalone(
  const void *query,
  const void *key,
  const void *value,
  const void *attn_mask,
  void *output,
  const sdpa_params &params);

} // namespace sdpa
} // namespace lowoha
} // namespace zendnnl

#endif // LOWOHA_FLASH_SDPA_HPP
