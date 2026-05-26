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
#ifndef _SDPA_TENSOR_FACTORY_HPP_
#define _SDPA_TENSOR_FACTORY_HPP_

// This header intentionally declares only the QKV/output/mask tensor-creation
// helpers used by the SDPA benchmark driver. It does NOT pull in
// `sdpa_utils.hpp` (and its `SdpaConfig`) because nothing here references it;
// callers that need `SdpaConfig` already include `sdpa_utils.hpp` directly.
#include "benchdnn.hpp"

namespace zendnnl {
namespace benchdnn {
namespace sdpa {

using namespace zendnnl::examples;

/**
 * @enum qkv_layout_t
 * @brief Physical memory layout used for the Q/K/V (and matching output) tensors.
 *
 * The semantic axes are always [B, H, S, D] -- only the in-memory ordering
 * changes. The flash backend honours per-axis strides, so the runner can
 * support either layout by allocating the tensor with a permuted shape and
 * setting strides accordingly.
 *
 *   - bhsd : memory order [B, H, S, D]
 *            strides {H*S*D, S*D, D, 1}     (current default; "head-major")
 *   - bshd : memory order [B, S, H, D]
 *            strides {S*H*D, D,   H*D, 1}   ("token-major", common in many
 *                                             LLM frameworks)
 *
 * The kernel requires `stride_d == 1` (head_dim must be the innermost
 * contiguous dimension), which both layouts satisfy.
 *
 * The explicit `: int` underlying type lets `sdpa_utils.hpp` forward-declare
 * this enum (`enum class qkv_layout_t : int;`) and embed it as an SdpaConfig
 * member without pulling this whole header in -- which is necessary because
 * including this header from `sdpa_utils.hpp` would close an
 * `sdpa_utils.hpp <-> benchdnn.hpp <-> sdpa_benchdnn.hpp <-> sdpa_tensor_factory.hpp`
 * cycle.
 */
enum class qkv_layout_t : int {
  bhsd = 0,
  bshd = 1
};

/**
 * @brief Creates a Q, K or V tensor with random uniform values in the
 *        requested QKV layout.
 *
 * @param tensor_factory Factory object for tensor creation.
 * @param B Batch dimension.
 * @param H Number of heads.
 * @param S Sequence length (S_q for Q/output, S_kv for K/V).
 * @param D Per-head dimension.
 * @param dt Tensor data type (f32, bf16 or f16).
 * @param layout Physical memory layout: `bhsd` (shape {B,H,S,D}) or `bshd`
 *               (shape {B,S,H,D}). Only the underlying allocation order
 *               changes; the runner derives matching strides.
 * @param name Tensor name (used in logs).
 * @param out  Output tensor reference.
 * @return int OK (0) on success, NOT_OK (1) on failure.
 */
int create_qkv_tensor(tensor_factory_t &tensor_factory,
                      int64_t B, int64_t H, int64_t S, int64_t D,
                      zendnnl::common::data_type_t dt,
                      qkv_layout_t layout,
                      const std::string &name,
                      tensor_t &out);

/**
 * @brief Creates the output tensor (zero-initialised) in the same QKV layout
 *        as Q so the operator can write back contiguously alongside the
 *        existing Q strides.
 *
 *   - bhsd : shape {B, H, S_q, D}
 *   - bshd : shape {B, S_q, H, D}
 */
int create_output_tensor(tensor_factory_t &tensor_factory,
                         int64_t B, int64_t H, int64_t S_q, int64_t D,
                         zendnnl::common::data_type_t dt,
                         qkv_layout_t layout,
                         tensor_t &out);

/**
 * @brief Creates the attention mask tensor.
 *
 * - mask_ndims == 0  : returns an empty tensor (caller should pass nullptr).
 * - mask_ndims == 2  : shape {S_q, S_kv}.
 * - mask_ndims == 4  : shape {B, H, S_q, S_kv}.
 *
 * Tensor is populated with small uniform random values (representative of
 * additive attention biases such as ALiBi or padding masks); causal masking is
 * applied separately by the operator via `is_causal`.
 */
int create_mask_tensor(tensor_factory_t &tensor_factory,
                       int64_t B, int64_t H, int64_t S_q, int64_t S_kv,
                       int mask_ndims,
                       zendnnl::common::data_type_t mask_dt,
                       tensor_t &out);

} // namespace sdpa
} // namespace benchdnn
} // namespace zendnnl

#endif
