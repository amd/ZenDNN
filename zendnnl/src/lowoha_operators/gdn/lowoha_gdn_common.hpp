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

#ifndef _LOWOHA_GDN_COMMON_HPP
#define _LOWOHA_GDN_COMMON_HPP

#include <cmath>
#include <cstdint>
#include "common/logging.hpp"
#include "memory/memory_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace gdn {

using namespace zendnnl::common;

enum class gdn_kernel_t : int32_t {
    none = -1, /*!< Default: native kernel when the CPU supports it */
    native = 0, /*!< Fused AVX-512 kernels */
    reference = 1 /*!< Scalar FP32 reference */
};

inline const char *kernel_to_string(gdn_kernel_t kernel) {
    switch (kernel) {
        case gdn_kernel_t::none: return "none";
        case gdn_kernel_t::native: return "native";
        case gdn_kernel_t::reference: return "reference";
        default: return "unknown";
    }
}

/**
 * @brief Memory layout of one (slot, value-head) recurrent state matrix.
 *
 * kv: [head_k_dim][head_v_dim], value dim contiguous. This is how the vLLM
 *     CPU kernels read the SSM state pool, so states written by vLLM's
 *     native kernels can be continued by gdn_direct and vice versa.
 * vk: [head_v_dim][head_k_dim], key dim contiguous (FLA / Triton layout).
 */
enum class gdn_state_layout_t : int32_t { kv = 0, vk = 1 };

/// Gate activation of the fused gated RMSNorm.
enum class gdn_norm_gate_t : int32_t { silu = 0, sigmoid = 1 };

/**
 * @brief What the a and b inputs hold.
 *
 * raw: the gate projections; g = -exp(A_log) * softplus(a + dt_bias) and
 *     beta = sigmoid(b) (2 * sigmoid with allow_neg_eigval) are computed in
 *     the kernel.
 * precomputed: a holds g (the log decay) and b holds beta, as FLA's and
 *     transformers' chunk_gated_delta_rule take them; A_log and dt_bias are
 *     not read.
 */
enum class gdn_gate_input_t : int32_t { raw = 0, precomputed = 1 };

/// Prefill algorithm of the native kernel.
enum class gdn_prefill_algo_t : int32_t {
    automatic = -1, /*!< Chunked for long sequences, else recurrent */
    recurrent = 0, /*!< Token by token, FP32 throughout */
    chunked = 1 /*!< Chunked delta rule: src_dt GEMM operands, FP32 state */
};

inline const char *prefill_algo_to_string(gdn_prefill_algo_t algo) {
    switch (algo) {
        case gdn_prefill_algo_t::automatic: return "automatic";
        case gdn_prefill_algo_t::recurrent: return "recurrent";
        case gdn_prefill_algo_t::chunked: return "chunked";
        default: return "unknown";
    }
}

/**
 * @brief Parameters for the Gated DeltaNet (GDN) linear-attention operator.
 *
 * Per value head hv (key head h = hv / (num_v_heads / num_k_heads)) and token
 * t, with q_t, k_t L2-normalised when use_qk_l2norm is set:
 *
 *   g_t    = -exp(A_log[hv]) * softplus(a_t[hv] + dt_bias[hv])
 *   beta_t = sigmoid(b_t[hv])   (2 * sigmoid with allow_neg_eigval)
 *   S      = exp(g_t) * S
 *   S      = S + k_t (beta_t * (v_t - S^T k_t))^T
 *   o_t    = scale * S^T q_t
 *
 * mixed_qkv packs [q (num_k_heads * head_k_dim), k (same), v (num_v_heads *
 * head_v_dim)] per token. When conv_width > 0 the operator first applies the
 * depthwise causal conv1d (+ SiLU when silu is set) to mixed_qkv, using and
 * advancing conv_state. When conv_width == 0 mixed_qkv is taken as already
 * convolved and conv_state / conv_weight are ignored.
 *
 * Prefill-only calls (num_decodes == 0) may also take the inputs of FLA's and
 * transformers' chunk_gated_delta_rule: q, k and v as separate strided
 * tensors (k and v set, conv_width == 0) and g / beta instead of the gate
 * projections (gate_input precomputed).
 *
 * Optional fused gated RMSNorm (when norm_weight != nullptr), applied per
 * value head on the output:  out = rmsnorm(o) * norm_weight * act(z), with
 * act = silu or sigmoid (norm_gate). With round_core_out, o is first rounded
 * to src_dt, as an unfused core -> norm pipeline does.
 *
 * All strides are in elements.
 */
struct gdn_params {
    // Sequences of the batch, decodes first: batch = num_decodes +
    // num_prefills, as in vLLM's non-speculative GDN metadata.
    int64_t batch;
    // The first num_decodes sequences are single-token continuations and run
    // the decode kernel; the other batch - num_decodes sequences are prefills
    // and run the prefill kernel. 0 <= num_decodes <= batch.
    int64_t num_decodes;
    // Tokens of the batch: cu_seqlens[batch], or batch when cu_seqlens is
    // null. mixed_qkv, a, b, z and output hold one row per token.
    int64_t num_tokens;

    // Packed variable-length offsets: batch + 1 non-decreasing values starting
    // at 0 (SDPA's cu_seqlens convention, vLLM's query_start_loc). Sequence i
    // owns token rows [cu_seqlens[i], cu_seqlens[i + 1]). Decode tokens come
    // first, one per sequence, so cu_seqlens[num_decodes] == num_decodes.
    // nullptr means every sequence has one token: sequence i owns row i.
    const int32_t *cu_seqlens;

    // batch flags indexed like state_indices, i.e. over all sequences, as
    // vLLM's GDNAttentionMetadata::has_initial_state. For a prefill sequence
    // (i >= num_decodes), true continues from the conv and SSM state in its
    // pool slot and false starts from zero states. The flags of the decode
    // sequences are not read: a decode always continues from its slot.
    // nullptr: every prefill starts from zero states.
    const bool *has_initial_state;

    int64_t num_k_heads;
    int64_t num_v_heads;
    int64_t head_k_dim;
    int64_t head_v_dim;

    // Depthwise causal conv1d width; 0 means mixed_qkv is already convolved.
    int64_t conv_width;
    bool silu;

    // Per-token row strides of the packed inputs.
    int64_t qkv_stride;
    int64_t a_stride;
    int64_t b_stride;

    // Separate q, k and v, already convolved (conv_width == 0): when k and v
    // are set, mixed_qkv points at q and each of q, k and v is
    // [token][head][dim] with the token stride (qkv_stride for q) and head
    // stride below. null (default): packed mixed_qkv rows.
    const void *k;
    const void *v;
    int64_t q_stride_h;
    int64_t k_stride_t;
    int64_t k_stride_h;
    int64_t v_stride_t;
    int64_t v_stride_h;

    // Gate inputs (raw by default). With precomputed gates a and b have
    // their own types a_dt and b_dt (f32, bf16 or f16); raw gates read them
    // as src_dt.
    gdn_gate_input_t gate_input;
    data_type_t a_dt;
    data_type_t b_dt;
    // Raw gates: beta = 2 * sigmoid(b), as FLA's allow_neg_eigval (OLMo's
    // hybrid layers), so the transition I - beta k k^T has eigenvalues down
    // to -1 (with L2-normalised k).
    bool allow_neg_eigval;

    // Output [token][v_head][head_v_dim].
    int64_t out_stride_t;
    int64_t out_stride_h;

    // Conv state pool [slot][tap][channel] (tap stride and channel stride
    // describe both the SD and DS layouts used by vLLM).
    int64_t conv_state_slot_stride;
    int64_t conv_state_tap_stride;
    int64_t conv_state_dim_stride;

    // SSM state pool [slot][v_head][...]; the inner matrix is contiguous and
    // laid out as state_layout says.
    int64_t ssm_state_slot_stride;
    int64_t ssm_state_head_stride;
    gdn_state_layout_t state_layout;

    // src_dt covers mixed_qkv, a, b, conv weight / bias, conv state, z and
    // output. state_dt is the SSM state pool. A_log, dt_bias and norm_weight
    // carry their own types (f32 or the src type).
    data_type_t src_dt;
    data_type_t state_dt;
    data_type_t a_log_dt;
    data_type_t dt_bias_dt;
    data_type_t norm_weight_dt;

    // 0 means 1 / sqrt(head_k_dim).
    float scale;
    bool use_qk_l2norm;
    float l2norm_eps;
    float softplus_threshold;

    // Sequences whose slot equals pad_slot_id leave both pools untouched: a
    // padded decode writes a zero output row, a padded prefill leaves its
    // output rows as they are.
    int32_t pad_slot_id;

    // Number of slots in the conv and SSM state pools. When > 0 every
    // non-pad entry of state_indices must lie in [0, num_state_slots); 0
    // skips the range check. Duplicate non-pad slots in one call are not
    // supported, unless update_state is false.
    int64_t num_state_slots;

    // false: the prefills read their initial states as has_initial_state
    // says but leave both pools unchanged, so several sequences may read one
    // slot, and pools nothing reads may be null. Prefill-only calls
    // (num_decodes == 0); decodes always advance their slot.
    bool update_state;

    // Speculative verify (vLLM's speculative decoding), when
    // spec_state_indices is set: every sequence checks up to
    // spec_state_stride draft tokens. Sequence i continues from SSM slot
    // spec_state_indices[i * spec_state_stride + num_accepted_tokens[i] - 1]
    // and writes the state after its token t to slot
    // spec_state_indices[i * spec_state_stride + t]; 1 <=
    // num_accepted_tokens[i] <= spec_state_stride, whatever its token count.
    // Its conv state is slot spec_state_indices[i * spec_state_stride] of
    // conv_taps() taps, laid out as vLLM's speculative conv update keeps it:
    // the conv reads the conv_width - 1 taps from num_accepted_tokens[i] - 1
    // on, then the conv_width - 2 taps from num_accepted_tokens[i] on move to
    // the front and the sequence's q tokens follow, so the next step finds its
    // history at tap num_accepted_tokens - 1 whatever q was; the taps after
    // them keep their values. state_indices and has_initial_state are not
    // read. No decodes, update_state, packed rows and raw gates.
    const int32_t *spec_state_indices;
    int64_t spec_state_stride;
    const int32_t *num_accepted_tokens;

    // Round the conv output to src_dt before the recurrence, which is what an
    // unfused conv -> recurrence pipeline does. Off keeps it in FP32.
    bool round_conv_out;

    // Fused gated RMSNorm.
    const void *norm_weight;
    const void *z;
    int64_t z_stride_t;
    int64_t z_stride_h;
    float norm_eps;
    gdn_norm_gate_t norm_gate;
    bool round_core_out;

    // Prefill: token-recurrent or chunked delta rule (native kernel only).
    // automatic decides per call from the prefills' average length; split
    // passes give one call's bits only with the algorithm pinned.
    // ZENDNNL_GDN_PREFILL overrides it.
    gdn_prefill_algo_t prefill_algo;

    // num_threads is int32_t to match the type used by OpenMP APIs.
    int32_t num_threads;
    // ZENDNNL_GDN_KERNEL overrides it.
    gdn_kernel_t kernel;

    gdn_params()
        : batch(0)
        , num_decodes(0)
        , num_tokens(0)
        , cu_seqlens(nullptr)
        , has_initial_state(nullptr)
        , num_k_heads(0)
        , num_v_heads(0)
        , head_k_dim(0)
        , head_v_dim(0)
        , conv_width(0)
        , silu(true)
        , qkv_stride(0)
        , a_stride(0)
        , b_stride(0)
        , k(nullptr)
        , v(nullptr)
        , q_stride_h(0)
        , k_stride_t(0)
        , k_stride_h(0)
        , v_stride_t(0)
        , v_stride_h(0)
        , gate_input(gdn_gate_input_t::raw)
        , a_dt(data_type_t::none)
        , b_dt(data_type_t::none)
        , allow_neg_eigval(false)
        , out_stride_t(0)
        , out_stride_h(0)
        , conv_state_slot_stride(0)
        , conv_state_tap_stride(0)
        , conv_state_dim_stride(1)
        , ssm_state_slot_stride(0)
        , ssm_state_head_stride(0)
        , state_layout(gdn_state_layout_t::kv)
        , src_dt(data_type_t::none)
        , state_dt(data_type_t::f32)
        , a_log_dt(data_type_t::f32)
        , dt_bias_dt(data_type_t::f32)
        , norm_weight_dt(data_type_t::none)
        , scale(0.0f)
        , use_qk_l2norm(true)
        , l2norm_eps(1e-6f)
        , softplus_threshold(20.0f)
        , pad_slot_id(-1)
        , num_state_slots(0)
        , update_state(true)
        , spec_state_indices(nullptr)
        , spec_state_stride(0)
        , num_accepted_tokens(nullptr)
        , round_conv_out(true)
        , norm_weight(nullptr)
        , z(nullptr)
        , z_stride_t(0)
        , z_stride_h(0)
        , norm_eps(1e-6f)
        , norm_gate(gdn_norm_gate_t::silu)
        , round_core_out(true)
        , prefill_algo(gdn_prefill_algo_t::automatic)
        , num_threads(0)
        , kernel(gdn_kernel_t::none) {}

    int64_t qk_dim() const { return num_k_heads * head_k_dim; }
    int64_t v_dim() const { return num_v_heads * head_v_dim; }
    int64_t conv_dim() const { return 2 * qk_dim() + v_dim(); }
    bool separate_qkv() const { return k != nullptr || v != nullptr; }
    bool spec_verify() const { return spec_state_indices != nullptr; }
    // Taps of one conv state slot: the conv history, plus one per draft token
    // past the first in speculative verify.
    int64_t conv_taps() const {
        if (conv_width <= 1) { return 0; }
        return conv_width - 1 + (spec_verify() ? spec_state_stride - 1 : 0);
    }
    bool precomputed_gates() const {
        return gate_input == gdn_gate_input_t::precomputed;
    }
    // beta = beta_scale() * sigmoid(b) for raw gates.
    float beta_scale() const { return allow_neg_eigval ? 2.0f : 1.0f; }
    // Types a and b are read as.
    data_type_t a_type() const {
        return precomputed_gates() && a_dt != data_type_t::none ? a_dt : src_dt;
    }
    data_type_t b_type() const {
        return precomputed_gates() && b_dt != data_type_t::none ? b_dt : src_dt;
    }
    float effective_scale() const {
        return scale != 0.0f ? scale
                             : 1.0f / std::sqrt(static_cast<float>(head_k_dim));
    }
};

} // namespace gdn
} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_GDN_COMMON_HPP
