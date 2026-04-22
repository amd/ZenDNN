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

#ifndef LOWOHA_SDPA_BMM_HPP
#define LOWOHA_SDPA_BMM_HPP

#include "lowoha_operators/sdpa/bmm_sdpa/lowoha_sdpa_utils.hpp"
#include "lowoha_operators/matmul/lowoha_matmul.hpp"
#include "lowoha_operators/softmax/lowoha_softmax.hpp"

namespace zendnnl {
namespace lowoha {
namespace sdpa {

/**
 * @brief BMM-based Scaled Dot-Product Attention
 *
 * Three-step implementation using batched matrix multiplications:
 *   1. scores = Q @ K^T * scale  (+ optional mask)
 *   2. attn_weights = softmax(scores)
 *   3. output = attn_weights @ V
 *
 * @param query        Query tensor [batch*num_heads, seq_len, head_dim]
 * @param key          Key tensor   [batch*num_heads, seq_len, head_dim]
 * @param value        Value tensor [batch*num_heads, seq_len, head_dim]
 * @param attn_mask    Optional attention mask (can be nullptr)
 * @param output       Output tensor [batch*num_heads, seq_len, head_dim]
 * @param params       SDPA parameters
 *
 * @return status_t::success or status_t::failure
 */
status_t bmm_based_sdpa(
  const void *query,
  const void *key,
  const void *value,
  const void *attn_mask,
  void *output,
  sdpa_params &params
);

/// Free the thread-local BMM scratch buffer. Must be called from the same thread.
void sdpa_free_scratch();

} // namespace sdpa
} // namespace lowoha
} // namespace zendnnl

#endif // LOWOHA_SDPA_BMM_HPP
