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
#include "common/logging.hpp"
#include "memory/memory_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace sdpa {

using namespace zendnnl::common;

/**
 * @brief Attention mask type
 */
enum class mask_type_t {
    none = 0, /*!< No mask */
    causal = 1, /*!< Causal mask (upper triangular) */
    custom = 2 /*!< Custom attention mask provided */
};

enum class sdpa_kernel_t : int32_t {
    none = -1, /*!< No kernel selected */
    flash = 0, /*!< Flash kernel */
    bmm = 1, /*!< BMM kernel */
    reference = 2 /*!< Reference kernel */
};

/**
 * @brief Convert sdpa_kernel_t to string for logging
 */
inline const char *kernel_to_string(sdpa_kernel_t kernel) {
    switch (kernel) {
        case sdpa_kernel_t::none: return "none";
        case sdpa_kernel_t::flash: return "flash";
        case sdpa_kernel_t::bmm: return "bmm";
        case sdpa_kernel_t::reference: return "reference";
        default: return "unknown";
    }
}

/**
 * @brief Unified parameter structure for all LOWOHA SDPA backends
 *        (BMM-based and flash-style).
 *
 * SDPA computes: Attention(Q, K, V) = softmax(Q * K^T / scale) * V
 *
 * Tensor shapes (4D BHSD):
 *   Q/Output     : [batch, num_heads, seq_len, head_dim]
 *   K/V          : [batch, kv_num_heads, kv_seq_len, head_dim]
 *   Attention mask: broadcastable 2-D or 4-D (optional)
 *
 * For self-attention seq_len == kv_seq_len.  For cross-attention
 * (e.g. encoder-decoder models) they may differ.
 *
 * The flash backend uses the per-tensor BHSD strides to support
 * non-contiguous layouts.  The BMM backend ignores strides and
 * expects pre-packed contiguous [batch*heads, seq, dim] tensors.
 *
 * Variable-length batches use the packed layout described at
 * cu_seqlens_q instead (flash and reference backends only).
 */
struct sdpa_params {
    // Tensor dimensions [Batch, Heads, Seq, Dim]
    int64_t batch;
    int64_t num_heads;
    // Number of K/V heads for GQA/MQA. 0 means "same as num_heads" (MHA).
    int64_t kv_num_heads;
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
    // Off by default. When true and sliding_window_size > 0, query i attends
    // only to keys j with |i - j| <= sliding_window_size - 1 (L = R = W-1).
    // Combined with is_causal the right radius is 0 (j <= i).
    bool sliding_window;
    int64_t sliding_window_size;
    double dropout_p;
    // Dynamically quantize BF16 Q/K per token for an INT8 QK matmul.
    bool is_qk_quant;
    // Dynamically quantize BF16 V per channel and the softmax tile to U8 for
    // an INT8 PV matmul.  Independent of is_qk_quant -- neither, either, or
    // both may be set.  ZENDNNL_SDPA_INT8_QK / ZENDNNL_SDPA_INT8_PV override
    // the corresponding field when the variable is present.
    bool is_pv_quant;

    // num_threads is int32_t to match the type used by OpenMP APIs
    int32_t num_threads;

    // Backend kernel selection (none = default to flash)
    sdpa_kernel_t kernel;

    // Packed variable-length mode, enabled when cu_seqlens_q is non-null.
    // Sequences are stored back to back: Q / output are
    // [total_q, num_heads, head_dim] and K / V are
    // [total_kv, kv_num_heads, head_dim], and sequence b owns query rows
    // [cu_seqlens_q[b], cu_seqlens_q[b + 1]) and key / value rows
    // [cu_seqlens_k[b], cu_seqlens_k[b + 1]). Each array holds batch + 1
    // non-decreasing offsets starting at 0 (vLLM's query_start_loc / ViT
    // cu_seqlens). cu_seqlens_k == nullptr means the same offsets as
    // cu_seqlens_q (self-attention). In this mode batch is the number of
    // sequences, seq_len / kv_seq_len are upper bounds on the per-sequence
    // query / key lengths (max_seqlen), *_stride_s / *_stride_h are the token
    // and head strides, and *_stride_b are ignored. For the causal and window
    // bands, queries are aligned to the end of their keys: query row r of a
    // sequence sits at key position r + (kv_len - q_len), as in FlashAttention
    // varlen and vLLM prefill with a cached prefix. attn_mask is not
    // supported.
    const int32_t *cu_seqlens_q;
    const int32_t *cu_seqlens_k;

    sdpa_params()
        : batch(1)
        , num_heads(1)
        , kv_num_heads(0)
        , seq_len(0)
        , kv_seq_len(0)
        , head_dim(0)
        , q_stride_b(0)
        , q_stride_h(0)
        , q_stride_s(0)
        , q_stride_d(1)
        , k_stride_b(0)
        , k_stride_h(0)
        , k_stride_s(0)
        , k_stride_d(1)
        , v_stride_b(0)
        , v_stride_h(0)
        , v_stride_s(0)
        , v_stride_d(1)
        , o_stride_b(0)
        , o_stride_h(0)
        , o_stride_s(0)
        , o_stride_d(1)
        , mask_ndims(0)
        , mask_sizes {0, 0, 0, 0}
        , mask_strides {0, 0, 0, 0}
        , mask_type(mask_type_t::none)
        , mask_dims {0, 0, 0}
        , qkv_dt(data_type_t::none)
        , out_dt(data_type_t::none)
        , mask_dt(data_type_t::none)
        , scale(0.0)
        , is_causal(false)
        , sliding_window(false)
        , sliding_window_size(0)
        , dropout_p(0.0)
        , is_qk_quant(false)
        , is_pv_quant(false)
        , num_threads(0)
        , kernel(sdpa_kernel_t::none)
        , cu_seqlens_q(nullptr)
        , cu_seqlens_k(nullptr) {}
};

/**
 * @brief Select SDPA kernel from params.
 *
 * When @p params.kernel is sdpa_kernel_t::none, defaults to flash.
 * Unknown values are logged as errors and returned unchanged so the
 * caller can fail the dispatch instead of silently running flash.
 */
inline sdpa_kernel_t kernel_select(sdpa_params &params) {
    sdpa_kernel_t kernel = params.kernel == sdpa_kernel_t::none
            ? sdpa_kernel_t::flash
            : params.kernel;

    if (kernel != sdpa_kernel_t::flash && kernel != sdpa_kernel_t::bmm
            && kernel != sdpa_kernel_t::reference) {
        log_error("kernel_select: invalid kernel value ",
                static_cast<int32_t>(params.kernel), " (",
                kernel_to_string(params.kernel), ")");
        return params.kernel;
    }

    params.kernel = kernel;
    return kernel;
}

} // namespace sdpa
} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_SDPA_COMMON_HPP
