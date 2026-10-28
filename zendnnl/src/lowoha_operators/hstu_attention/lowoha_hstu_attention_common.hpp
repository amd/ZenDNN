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

#ifndef LOWOHA_HSTU_ATTENTION_COMMON_HPP
#define LOWOHA_HSTU_ATTENTION_COMMON_HPP

#include <cstdint>
#include "common/data_types.hpp"

namespace zendnnl {
namespace lowoha {
namespace hstu_attention {

using zendnnl::common::data_type_t;

enum class hstu_attention_kernel_t : int32_t {
    none = -1, ///< Pick the fastest kernel that supports the call.
    tiled = 0, ///< Tiled GEMM kernel (causal, BF16 Q/K/V, AVX512-BF16).
    reference = 1 ///< Scalar multithreaded kernel; supports every call.
};

inline const char *kernel_to_string(hstu_attention_kernel_t kernel) {
    switch (kernel) {
        case hstu_attention_kernel_t::none: return "none";
        case hstu_attention_kernel_t::tiled: return "tiled";
        case hstu_attention_kernel_t::reference: return "reference";
        default: return "unknown";
    }
}

/**
 * @brief Parameters of HSTU attention (DLRM v3 / generative recommenders).
 *
 * Per sequence b of length n = seq_offsets[b + 1] - seq_offsets[b] and head h:
 *
 *   out[i] = sum_j mask(i, j) * silu(alpha * q[i] . k[j]) * scale(i) * v[j]
 *
 * There is no softmax: masked scores contribute zero. `mask` is the HSTU valid
 * attention mask (causal, target aware, optional window and contextual
 * prefix) of `pytorch_hstu_mha` in generative-recommenders.
 *
 * Layout (jagged, last dimension contiguous; strides in elements):
 *   K   : [L, num_heads, qk_dim]   with L = seq_offsets[num_seqs]
 *   V   : [L, num_heads, v_dim]
 *   Q   : [L, num_heads, qk_dim]   full mode (delta_size == 0)
 *         [num_seqs * delta_size, num_heads, qk_dim]   delta mode: the
 *         queries of the last delta_size positions of every sequence
 *         (`pytorch_cached_hstu_mha`)
 *   out : same rows as Q, [rows, num_heads, v_dim]; must not overlap itself
 *         or alias Q, K or V
 *
 * Delta mode also honours causal = false, min_full_attn_seq_len and a scalar
 * attn_scale, which `pytorch_cached_hstu_mha` does not take (it is always
 * causal and scales by 1 / max_seq_len).
 */
struct hstu_attention_params {
    int64_t num_seqs = 0;
    int64_t num_heads = 0;
    int64_t qk_dim = 0;
    int64_t v_dim = 0;
    /// Upper bound of the sequence lengths; the default scale is 1/max_seq_len.
    int64_t max_seq_len = 0;
    /// [num_seqs + 1] row offsets of every sequence in K and V.
    const int64_t *seq_offsets = nullptr;
    /// 0: full mode. > 0: Q and out hold delta_size rows per sequence.
    int64_t delta_size = 0;

    // Mask.
    bool causal = true;
    /// [num_seqs] trailing target (candidate) count per sequence, or nullptr.
    const int64_t *num_targets = nullptr;
    /// Attention window; 0 disables it.
    int64_t max_attn_len = 0;
    /// Leading contextual tokens: their rows see the whole history, and later
    /// rows see them subject to the window; 0 disables them.
    int64_t contextual_seq_len = 0;
    /// The last min_full_attn_seq_len history positions ignore the window.
    int64_t min_full_attn_seq_len = 0;

    // Scaling.
    float alpha = 1.f;
    /// Scale applied after silu, used as given when has_attn_scale; otherwise
    /// the scale is 1 / max_seq_len.
    bool has_attn_scale = false;
    float attn_scale = 0.f;
    /// Optional per-token scale [L] (f32), replacing attn_scale; full mode only.
    const float *attn_scale_per_token = nullptr;

    // Strides of the token and head dimensions.
    int64_t q_stride_t = 0, q_stride_h = 0;
    int64_t k_stride_t = 0, k_stride_h = 0;
    int64_t v_stride_t = 0, v_stride_h = 0;
    int64_t o_stride_t = 0, o_stride_h = 0;

    data_type_t qkv_dt = data_type_t::bf16; ///< f32 or bf16.
    data_type_t out_dt = data_type_t::bf16; ///< f32 or bf16.

    /// true: variable-length attention, each sequence computed over its own
    /// length. false: every sequence padded to max_seq_len and the kernel run
    /// on the uniform layout, as without variable-length support (same
    /// output; full mode only, ignored in delta mode). For comparisons;
    /// ZENDNNL_HSTU_ATTENTION_VARLEN=0|1 overrides it.
    bool is_varlen = true;

    /// Requested kernel; ZENDNNL_HSTU_ATTENTION_KERNEL=tiled|reference
    /// overrides it.
    hstu_attention_kernel_t kernel = hstu_attention_kernel_t::none;
    /// 0 uses the current omp_get_max_threads(); the count in effect before
    /// the call is restored afterwards.
    int32_t num_threads = 0;
};

} // namespace hstu_attention
} // namespace lowoha
} // namespace zendnnl

#endif // LOWOHA_HSTU_ATTENTION_COMMON_HPP
