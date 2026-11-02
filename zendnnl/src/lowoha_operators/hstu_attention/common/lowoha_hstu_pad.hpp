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

/// Attention without variable-length support (is_varlen = false): every
/// sequence padded to max_seq_len.

#ifndef LOWOHA_HSTU_PAD_HPP
#define LOWOHA_HSTU_PAD_HPP

#include <functional>
#include "lowoha_operators/hstu_attention/lowoha_hstu_attention.hpp"

namespace zendnnl {
namespace lowoha {
namespace hstu_attention {

/// A kernel call on (Q, K, V, out, params).
using hstu_attention_run_t = std::function<status_t(const void *, const void *,
        const void *, void *, const hstu_attention_params &)>;

/// Full mode only: in delta mode the queries are the last rows of every
/// sequence, which padding would move.
inline bool hstu_attention_pad_supported(const hstu_attention_params &p) {
    return p.delta_size == 0;
}

/**
 * @brief Runs `run` on the batch padded to max_seq_len and copies the real
 * rows back to `output`.
 *
 * Sequence b becomes rows [b * max_seq_len, (b + 1) * max_seq_len) of
 * contiguous Q, K, V (and per-token scale) buffers, its real rows first and
 * zero rows after them. The zero rows count as extra targets, so every real
 * row keeps its mask (ids, and with them the causal, target, window and
 * contextual rules, do not change) and its value (a zero key adds
 * silu(0) * v = 0): the output equals the variable-length one. The cost is
 * the attention of the padded rows and the copies. The buffers are kept per
 * thread for reuse.
 */
status_t hstu_attention_padded(const void *query, const void *key,
        const void *value, void *output, const hstu_attention_params &params,
        const hstu_attention_run_t &run);

} // namespace hstu_attention
} // namespace lowoha
} // namespace zendnnl

#endif // LOWOHA_HSTU_PAD_HPP
