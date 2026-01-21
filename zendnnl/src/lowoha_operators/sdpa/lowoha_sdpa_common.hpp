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

#ifndef _LOWOHA_SDPA_COMMON_HPP
#define _LOWOHA_SDPA_COMMON_HPP

#include <cstdint>
#include "memory/memory_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace sdpa {

using namespace zendnnl::common;

/**
 * @brief Attention mask type
 */
enum class mask_type_t {
  none = 0,              /*!< No mask */
  causal = 1,            /*!< Causal mask (upper triangular) */
  custom = 2             /*!< Custom attention mask provided */
};

/**
 * @brief Parameter structure for LOWOHA SDPA operations
 *
 * This structure contains all parameters specific to Scaled Dot-Product
 * Attention computation.
 *
 * SDPA computes: Attention(Q, K, V) = softmax(Q * K^T / scale) * V
 *
 * Tensor shapes (4D):
 *   Q: [batch, num_heads, seq_len, head_dim]
 *   K: [batch, num_heads, seq_len, head_dim]
 *   V: [batch, num_heads, seq_len, head_dim]
 *   Output: [batch, num_heads, seq_len, head_dim]
 *   Attention mask (optional): [batch, 1, seq_len, seq_len] or broadcastable
 */
struct sdpa_params {
  // Tensor dimensions
  uint64_t batch;             ///< Batch size
  uint64_t num_heads;         ///< Number of attention heads
  uint64_t seq_len;           ///< Sequence length (same for Q, K, V)
  uint64_t head_dim;          ///< Head dimension (d_k)

  // Computation parameters
  float scale;                ///< Scale factor (default: 1/sqrt(head_dim))
  bool is_causal;             ///< If true, apply causal mask
  float dropout_p;            ///< Dropout probability (0 = no dropout)

  // Mask parameters
  mask_type_t mask_type;      ///< Type of attention mask
  bool has_attn_mask;         ///< Whether attention mask is provided

  // Data types
  data_type_t q_dt;           ///< Query data type
  data_type_t k_dt;           ///< Key data type
  data_type_t v_dt;           ///< Value data type
  data_type_t out_dt;         ///< Output data type
  data_type_t mask_dt;        ///< Attention mask data type

  // Threading
  uint64_t num_threads;       ///< Number of threads

  /**
   * @brief Default constructor
   */
  sdpa_params() : batch(1), num_heads(1), seq_len(0),
    head_dim(0), scale(0.0f), is_causal(false),
    dropout_p(0.0f), mask_type(mask_type_t::none),
    has_attn_mask(false),
    q_dt(data_type_t::none), k_dt(data_type_t::none),
    v_dt(data_type_t::none), out_dt(data_type_t::none),
    mask_dt(data_type_t::none),
    num_threads(0) {
  }
};

} // namespace sdpa
} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_SDPA_COMMON_HPP
