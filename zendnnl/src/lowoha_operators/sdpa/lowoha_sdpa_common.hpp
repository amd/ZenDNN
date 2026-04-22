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
 * @brief Unified parameter structure for all LOWOHA SDPA backends
 *        (BMM-based and flash-style).
 *
 * SDPA computes: Attention(Q, K, V) = softmax(Q * K^T / scale) * V
 *
 * Tensor shapes (4D BHSD):
 *   Q/Output     : [batch, num_heads, seq_len, head_dim]
 *   K/V          : [batch, num_heads, kv_seq_len, head_dim]
 *   Attention mask: broadcastable 2-D or 4-D (optional)
 *
 * For self-attention seq_len == kv_seq_len.  For cross-attention
 * (e.g. encoder-decoder models) they may differ.
 *
 * The flash backend uses the per-tensor BHSD strides to support
 * non-contiguous layouts.  The BMM backend ignores strides and
 * expects pre-packed contiguous [batch*heads, seq, dim] tensors.
 */
struct sdpa_params {
  // Tensor dimensions [Batch, Heads, Seq, Dim]
  int64_t batch;
  int64_t num_heads;
  int64_t seq_len;
  int64_t kv_seq_len;
  int64_t head_dim;

  // Per-tensor BHSD strides (flash backend)
  int64_t q_stride_b, q_stride_h, q_stride_s, q_stride_d;
  int64_t k_stride_b, k_stride_h, k_stride_s, k_stride_d;
  int64_t v_stride_b, v_stride_h, v_stride_s, v_stride_d;
  int64_t o_stride_b, o_stride_h, o_stride_s, o_stride_d;

  // Mask parameters — flash backend (raw 4-D sizes + strides)
  int mask_ndims;
  int64_t mask_sizes[4];
  int64_t mask_strides[4];

  // Mask parameters — BMM backend (reshaped 3-D)
  mask_type_t mask_type;
  int64_t mask_dims[3];

  // Data types
  data_type_t qkv_dt;
  data_type_t out_dt;
  data_type_t mask_dt;

  // Computation parameters
  double scale;
  bool is_causal;
  double dropout_p;

  // num_threads is int32_t to match the type used by OpenMP APIs
  int32_t num_threads;

  sdpa_params()
    : batch(1), num_heads(1), seq_len(0), kv_seq_len(0), head_dim(0),
      q_stride_b(0), q_stride_h(0), q_stride_s(0), q_stride_d(1),
      k_stride_b(0), k_stride_h(0), k_stride_s(0), k_stride_d(1),
      v_stride_b(0), v_stride_h(0), v_stride_s(0), v_stride_d(1),
      o_stride_b(0), o_stride_h(0), o_stride_s(0), o_stride_d(1),
      mask_ndims(0), mask_sizes{0, 0, 0, 0}, mask_strides{0, 0, 0, 0},
      mask_type(mask_type_t::none), mask_dims{0, 0, 0},
      qkv_dt(data_type_t::none), out_dt(data_type_t::none),
      mask_dt(data_type_t::none),
      scale(0.0), is_causal(false), dropout_p(0.0),
      num_threads(0) {}
};

} // namespace sdpa
} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_SDPA_COMMON_HPP
