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

#ifndef _LOWOHA_SDPA_HPP
#define _LOWOHA_SDPA_HPP

#include <cmath>
#include <cstring>
#include "lowoha_operators/sdpa/lowoha_sdpa_utils.hpp"
#include "lowoha_operators/matmul/lowoha_matmul.hpp"
#include "lowoha_operators/softmax/lowoha_softmax.hpp"

namespace zendnnl {
namespace lowoha {
namespace sdpa {

/**
 * @brief Execute Scaled Dot-Product Attention
 *
 * Performs: output = softmax(Q * K^T / scale) * V
 *
 * With optional attention mask:
 *   output = softmax(Q * K^T / scale + mask) * V
 *
 * With causal mask:
 *   output = softmax(Q * K^T / scale + causal_mask) * V
 *   where causal_mask[i,j] = -inf if j > i, else 0
 *
 * @param query        Query tensor [batch, num_heads, seq_len, head_dim]
 * @param key          Key tensor [batch, num_heads, seq_len, head_dim]
 * @param value        Value tensor [batch, num_heads, seq_len, head_dim]
 * @param attn_mask    Optional attention mask (can be nullptr)
 * @param output       Output tensor [batch, num_heads, seq_len, head_dim]
 * @param params       SDPA parameters
 *
 * @return status_t::success or status_t::failure
 */
status_t sdpa_direct(
  const void *query,
  const void *key,
  const void *value,
  const void *attn_mask,
  void *output,
  sdpa_params &params
);

} // namespace sdpa
} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_SDPA_HPP
