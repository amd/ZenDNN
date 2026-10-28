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
#ifndef LOWOHA_SDPA_REF_HPP
#define LOWOHA_SDPA_REF_HPP

#include "lowoha_operators/sdpa/bmm_sdpa/lowoha_sdpa_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace sdpa {

/**
 * @brief Reference SDPA encoder for LOWOHA (ground-truth path).
 *
 * Computes Attention(Q, K, V) = softmax(Q * K^T * scale + mask) * V,
 * where @c scale is the fused multiplier from @c sdpa_params::scale
 * (typically @c 1/sqrt(d_k)). Q/K/V loads and output stores use the
 * runtime dtype; score accumulation and softmax run in FP32 for numerical
 * stability. Tensor layout and strides are described by @c sdpa_params.
 */
status_t reference_sdpa(const void *query, const void *key, const void *value,
        const void *attn_mask, void *output, const sdpa_params &params);

/**
 * @brief Reference SDPA over a packed variable-length batch
 *        (@c params.cu_seqlens_q set); ground truth for flash_sdpa_varlen.
 */
status_t reference_sdpa_varlen(const void *query, const void *key,
        const void *value, const void *attn_mask, void *output,
        const sdpa_params &params);

} // namespace sdpa
} // namespace lowoha
} // namespace zendnnl

#endif // LOWOHA_SDPA_REF_HPP
