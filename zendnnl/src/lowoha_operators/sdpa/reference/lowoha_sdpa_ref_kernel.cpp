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

#include <cmath>
#include <limits>
#include <vector>
#include "lowoha_operators/common/omp_thread_control.hpp"
#include "lowoha_operators/common/operator_instrumentation.hpp"
#include "lowoha_operators/sdpa/flash_sdpa/lowoha_flash_sdpa_utils.hpp"
#include "lowoha_operators/sdpa/reference/lowoha_sdpa_ref_kernel.hpp"
#include <type_traits>

namespace zendnnl {
namespace lowoha {
namespace sdpa {
using namespace zendnnl::error_handling;
using zendnnl::common::bfloat16_t;
using zendnnl::common::float16_t;

/**
 * @brief Per-(batch, head) strides for an attention mask tensor.
 *
 * The reference kernel assumes the inner [seq_len_q, seq_len_kv] slab is
 * contiguous (row-major). Only the leading batch / head dimensions can
 * broadcast — @c stride_b and @c stride_h are 0 when the corresponding
 * dimension broadcasts, and the contiguous element count when it does not.
 */
struct mask_layout {
    int64_t stride_b;
    int64_t stride_h;
};

/**
 * @brief Derive (batch, head) broadcast strides from a mask tensor shape.
 *
 * Supported shapes (via @p mask_sizes and @p mask_ndims):
 *   - 2D [S_q, S_kv]                    -> stride_b = stride_h = 0 (broadcast)
 *   - 4D [B,   H,   S_q, S_kv]          -> stride_b = H*S_q*S_kv, stride_h = S_q*S_kv
 *   - 4D [1,   H,   S_q, S_kv]          -> stride_b = 0,           stride_h = S_q*S_kv
 *   - 4D [B,   1,   S_q, S_kv]          -> stride_b = S_q*S_kv,    stride_h = 0
 *   - 4D [1,   1,   S_q, S_kv]          -> stride_b = 0,           stride_h = 0
 *
 * The inner [S_q, S_kv] slab is assumed row-major contiguous; only leading
 * batch / head dimensions may broadcast. @c sdpa_params::mask_strides are not
 * consulted — callers must supply a mask whose inner slab is contiguous.
 *
 * @param mask_sizes  Per-dim sizes from @c sdpa_params::mask_sizes (2 or 4 dims).
 * @param mask_ndims  Number of mask dimensions (2 or 4).
 * @param seq_len_q   Q sequence length (rows of the inner slab).
 * @param seq_len_kv  K/V sequence length (cols of the inner slab).
 * @return mask_layout with stride_b / stride_h in elements (not bytes).
 *
 * @note Caller is responsible for validating that leading mask dims are either
 *       1 or match the corresponding Q dim (caller's input validation should
 *       enforce this up front).
 */
inline mask_layout compute_mask_strides(const int64_t *mask_sizes,
        int mask_ndims, int64_t seq_len_q, int64_t seq_len_kv) {
    mask_layout layout {0, 0};
    const int64_t inner = seq_len_q * seq_len_kv;
    if (mask_ndims == 2) {
        return layout; // 2D mask broadcasts across both batch and heads.
    }
    // 4D: leading dim of size 1 means broadcast; otherwise the matching
    // physical stride is mask_sizes[1] * inner (batch) or inner (head).
    layout.stride_h = (mask_sizes[1] != 1) ? inner : 0;
    layout.stride_b = (mask_sizes[0] != 1)
            ? static_cast<int64_t>(mask_sizes[1]) * inner
            : 0;
    return layout;
}

/** @brief Convert any element type to float (used at load time). */
template <typename T>
inline float to_float(T v) {
    return static_cast<float>(v);
}

/** @brief Convert float to any element type (used at store time). */
template <typename T>
inline T from_float(float f) {
    return static_cast<T>(f);
}

/**
 * @brief Compute Q @ K^T with fused scale.
 *
 * scores[i, j] = (sum_k q[i, k] * k[j, k]) * scale
 *
 * Q/K are typed (FP32 / BF16 / F16); the score buffer is always FP32 to keep
 * softmax numerically stable for low-precision inputs.
 *
 * Layout (per (batch, head) slice; the head_dim axis must be physically
 * contiguous when @p head_dim > 1, i.e. the underlying tensor's innermost
 * stride == 1. When @p head_dim == 1 the inner head_dim loop runs once
 * and the innermost stride is dead, matching the head_dim == 1 size-1
 * relaxation):
 *   q_data: [seq_len_q,  head_dim] with row stride @p q_seq_stride
 *   k_data: [seq_len_kv, head_dim] with row stride @p k_seq_stride
 *   attention_scores: [seq_len_q, seq_len_kv] row-major (scratch)
 *
 * A row stride of @p head_dim recovers the BHSD case (slab is contiguous);
 * a row stride of @c num_heads*head_dim handles the BSHD case (heads are
 * interleaved between sequence positions).
 */
template <typename qkv_t>
inline void matmul_qk(const qkv_t *q_data, const qkv_t *k_data,
        float *attention_scores, int64_t seq_len_q, int64_t seq_len_kv,
        int64_t head_dim, int64_t q_seq_stride, int64_t k_seq_stride,
        float scale) {
    for (int64_t i = 0; i < seq_len_q; i++) {
        for (int64_t j = 0; j < seq_len_kv; j++) {
            float sum = 0.0f;
            for (int64_t k = 0; k < head_dim; k++) {
                sum += to_float(q_data[i * q_seq_stride + k])
                        * to_float(k_data[j * k_seq_stride + k]);
            }
            attention_scores[i * seq_len_kv + j] = sum * scale;
        }
    }
}

/**
 * @brief Compute scores @ V.
 *
 * output[i, j] = sum_k scores[i, k] * v[k, j]
 *
 * scores stays FP32 (after softmax); V/output are typed.
 *
 * Layout (per (batch, head) slice; the head_dim axis must be physically
 * contiguous when @p head_dim > 1, i.e. the underlying tensor's innermost
 * stride == 1. When @p head_dim == 1 the inner head_dim loop runs once
 * and the innermost stride is dead, matching the head_dim == 1 size-1
 * relaxation):
 *   scores: [seq_len_q,  seq_len_kv] row-major (scratch)
 *   v_data: [seq_len_kv, head_dim]   with row stride @p v_seq_stride
 *   output: [seq_len_q,  head_dim]   with row stride @p o_seq_stride
 *
 * See @c matmul_qk for the BHSD vs BSHD row-stride convention.
 */
template <typename qkv_t>
inline void matmul_sv(const float *scores, const qkv_t *v_data, qkv_t *output,
        int64_t seq_len_q, int64_t seq_len_kv, int64_t head_dim,
        int64_t v_seq_stride, int64_t o_seq_stride) {
    for (int64_t i = 0; i < seq_len_q; i++) {
        for (int64_t j = 0; j < head_dim; j++) {
            float sum = 0.0f;
            for (int64_t k = 0; k < seq_len_kv; k++) {
                sum += scores[i * seq_len_kv + k]
                        * to_float(v_data[k * v_seq_stride + j]);
            }
            output[i * o_seq_stride + j] = from_float<qkv_t>(sum);
        }
    }
}

/**
 * @brief Numerically stable per-row softmax (in-place).
 *
 * Operates on an FP32 [seq_len_q, seq_len_kv] score buffer. Subtracts the row
 * max before the exp, then normalizes by the row sum. Each of the seq_len_q
 * rows is normalised independently.
 */
inline void softmax(float *scores, int64_t seq_len_q, int64_t seq_len_kv) {
    for (int64_t i = 0; i < seq_len_q; i++) {
        float max_val = scores[i * seq_len_kv];
        for (int64_t j = 1; j < seq_len_kv; j++) {
            if (scores[i * seq_len_kv + j] > max_val) {
                max_val = scores[i * seq_len_kv + j];
            }
        }
        // A row with no reachable key (empty sliding window on a short K/V)
        // is all -inf; exp(-inf - -inf) would be NaN. Zero probabilities make
        // the row contribute nothing, matching the flash backend.
        if (max_val == -std::numeric_limits<float>::infinity()) {
            for (int64_t j = 0; j < seq_len_kv; j++) {
                scores[i * seq_len_kv + j] = 0.0f;
            }
            continue;
        }
        float sum = 0.0f;
        for (int64_t j = 0; j < seq_len_kv; j++) {
            scores[i * seq_len_kv + j]
                    = std::exp(scores[i * seq_len_kv + j] - max_val);
            sum += scores[i * seq_len_kv + j];
        }
        for (int64_t j = 0; j < seq_len_kv; j++) {
            scores[i * seq_len_kv + j] /= sum;
        }
    }
}

/**
 * @brief Apply a causal (upper-triangular) mask in place.
 *
 * scores[i, j] = -inf for j > i over a [seq_len_q, seq_len_kv] grid. The lower
 * triangle (j <= i) is left untouched. Matches the flash-SDPA convention:
 * query position i attends to key positions [0..i].
 */
inline void apply_causal_mask(
        float *attention_scores, int64_t seq_len_q, int64_t seq_len_kv) {
    constexpr float neg_inf = -std::numeric_limits<float>::infinity();
    for (int64_t i = 0; i < seq_len_q; i++) {
        for (int64_t j = i + 1; j < seq_len_kv; j++) {
            attention_scores[i * seq_len_kv + j] = neg_inf;
        }
    }
}

/**
 * @brief Apply a bidirectional sliding-window band in place.
 *
 * Query i attends to keys j in [i-(W-1), i+(W-1)] (vLLM/Gemma: L = R = W-1).
 * When @p is_causal is also true the right radius is 0.
 */
inline void apply_sliding_window_mask(float *attention_scores,
        int64_t seq_len_q, int64_t seq_len_kv, int64_t window_size,
        bool is_causal) {
    constexpr float neg_inf = -std::numeric_limits<float>::infinity();
    const int64_t left = window_size - 1;
    const int64_t right = is_causal ? 0 : (window_size - 1);
    for (int64_t i = 0; i < seq_len_q; i++) {
        // j_lo is clamped to seq_len_kv: on cross-attention the band can start
        // past the last key, in which case the whole row is masked.
        const int64_t j_lo = std::min(seq_len_kv, (i > left) ? (i - left) : 0);
        const int64_t j_hi = std::min(seq_len_kv - 1, i + right);
        for (int64_t j = 0; j < j_lo; j++) {
            attention_scores[i * seq_len_kv + j] = neg_inf;
        }
        for (int64_t j = j_hi + 1; j < seq_len_kv; j++) {
            attention_scores[i * seq_len_kv + j] = neg_inf;
        }
    }
}

/**
 * @brief Apply an additive attention mask in place.
 *
 * scores += mask. Mask values are typically 0 (attend) or -inf (ignore).
 *
 * The mask buffer's element type @p mask_t may differ from FP32 (e.g. BF16
 * or F16); each loaded element is converted to float via @c to_float<mask_t>
 * before being added so the score buffer stays in FP32 for numerical
 * stability.
 *
 * @tparam mask_t           Mask element type (float, bfloat16_t, or
 *                          float16_t).
 * @param attention_scores  FP32 [seq_len_q, seq_len_kv] score buffer (in/out).
 * @param mask_ptr          Pointer to the contiguous [seq_len_q, seq_len_kv]
 *                          mask slab for the current (batch, head) — caller
 *                          has already advanced it by `b*stride_b + h*stride_h`
 *                          using @c compute_mask_strides for broadcast
 *                          dimensions.
 * @param seq_len_q         Q sequence length.
 * @param seq_len_kv        K/V sequence length.
 */
template <typename mask_t>
inline void apply_attention_mask(float *attention_scores,
        const mask_t *mask_ptr, int64_t seq_len_q, int64_t seq_len_kv) {
    for (int64_t i = 0; i < seq_len_q; i++) {
        for (int64_t j = 0; j < seq_len_kv; j++) {
            attention_scores[i * seq_len_kv + j]
                    += to_float(mask_ptr[i * seq_len_kv + j]);
        }
    }
}

/**
 * @brief Compute SDPA for a single (batch, head) slice.
 *
 * Performs: output = softmax(Q * K^T * scale + masks) * V
 *
 * Supports cross-attention via independent @p seq_len_q and @p seq_len_kv,
 * and BHSD / BSHD layouts via the per-tensor
 * sequence-axis strides. All four buffers must be physically contiguous
 * along the head_dim axis (innermost stride == 1) whenever
 * @p head_dim > 1. When @p head_dim == 1 the inner head_dim loop runs once
 * and the innermost stride is dead, matching the head_dim == 1 size-1
 * relaxation.
 *
 * @tparam qkv_t        Q/K/V/output element type (float, bfloat16_t, or
 *                      float16_t).
 * @tparam mask_t       Mask element type (float, bfloat16_t, or float16_t).
 *                      Independent of @p qkv_t — reduced-precision QKV
 *                      (bf16 / f16) may be combined with either an f32 or a
 *                      same-precision mask; f32 QKV always pairs with f32
 *                      mask (enforced by @c validate_flash_sdpa_inputs()).
 * @param q_new         Q slice [seq_len_q,  head_dim]
 * @param k_new         K slice [seq_len_kv, head_dim]
 * @param v_new         V slice [seq_len_kv, head_dim]
 * @param out_new       Output slice [seq_len_q, head_dim]
 * @param mask_ptr      Optional additive mask of element type @p mask_t
 *                      (length >= seq_len_q * seq_len_kv) or nullptr.
 * @param seq_len_q     Q sequence length.
 * @param seq_len_kv    K/V sequence length (== seq_len_q for self-attention).
 * @param head_dim      Per-head feature dimension (must match for Q, K, V).
 * @param q_seq_stride  Q row stride (head_dim for BHSD, H*head_dim for BSHD).
 * @param k_seq_stride  K row stride (see q_seq_stride).
 * @param v_seq_stride  V row stride (see q_seq_stride).
 * @param o_seq_stride  Output row stride (see q_seq_stride).
 * @param scale         Scaling factor applied to QK^T.
 * @param is_causal     If true, apply causal mask before softmax.
 * @param has_mask      If true and mask_ptr != nullptr, add mask before softmax.
 * @param sliding_window      If true, restrict each query to a band of width
 *                            @p sliding_window_size instead of the full row.
 * @param sliding_window_size Window width W (L = R = W-1); the band is ignored
 *                            when @p sliding_window is false or W <= 0.
 */
template <typename qkv_t, typename mask_t>
inline void compute_sdpa_per_head(const qkv_t *q_new, const qkv_t *k_new,
        const qkv_t *v_new, qkv_t *out_new, const mask_t *mask_ptr,
        int64_t seq_len_q, int64_t seq_len_kv, int64_t head_dim,
        int64_t q_seq_stride, int64_t k_seq_stride, int64_t v_seq_stride,
        int64_t o_seq_stride, float scale, bool is_causal, bool has_mask,
        bool sliding_window, int64_t sliding_window_size) {
    // FP32 score buffer keeps softmax numerically stable for low-precision QKV.
    std::vector<float> attention_scores(
            static_cast<size_t>(seq_len_q * seq_len_kv), 0.0f);

    matmul_qk<qkv_t>(q_new, k_new, attention_scores.data(), seq_len_q,
            seq_len_kv, head_dim, q_seq_stride, k_seq_stride, scale);

    if (sliding_window && sliding_window_size > 0) {
        apply_sliding_window_mask(attention_scores.data(), seq_len_q,
                seq_len_kv, sliding_window_size, is_causal);
    } else if (is_causal) {
        apply_causal_mask(attention_scores.data(), seq_len_q, seq_len_kv);
    }
    if (has_mask && mask_ptr != nullptr) {
        apply_attention_mask<mask_t>(
                attention_scores.data(), mask_ptr, seq_len_q, seq_len_kv);
    }

    softmax(attention_scores.data(), seq_len_q, seq_len_kv);

    matmul_sv<qkv_t>(attention_scores.data(), v_new, out_new, seq_len_q,
            seq_len_kv, head_dim, v_seq_stride, o_seq_stride);
}

// Per-(qkv_t) implementation of the reference SDPA encoder.
//
// All shared scaffolding (tensor / stride extraction, mask layout resolution,
// per-(b, h) parallel loop) lives here; the public execute() dispatches on
// the runtime QKV dtype to instantiate the correct typed entry.
//
// The lambda parameterises the inner template instantiation on the mask
// element type so the same body services both FP32 and BF16 masks. The
// dispatch cost is paid once per execute() call rather than per (b, h)
// iteration -- the parallel region is fully monomorphic.
template <typename qkv_t>
status_t execute_typed(const void *query, const void *key, const void *value,
        const void *attn_mask, void *output, const sdpa_params &params) {

    const bool has_mask = params.mask_ndims > 0;
    // Mask buffer dtype is decided per-call (@c validate_flash_sdpa_inputs()
    // restricts the supported combinations):
    //   - FP32 QKV  -> mask must be FP32
    //   - BF16 QKV  -> mask may be FP32 or BF16
    //   - F16  QKV  -> mask may be FP32 or F16
    // Softmax always runs in FP32, so each mask element is converted to float
    // at add time inside apply_attention_mask<mask_t>.
    void *mask_base_void = nullptr;
    data_type_t mask_dtype = params.mask_dt;
    mask_layout mask_layout {0, 0};

    // Q is [B, H, S_q, D], K/V are [B, kv_num_heads, S_kv, D]. For self-attention
    // S_q == S_kv; @c validate_flash_sdpa_inputs() permits S_q != S_kv to support
    // cross-attention but enforces K.S == V.S.
    const int64_t batch = params.batch;
    const int64_t num_heads = params.num_heads;
    const int64_t eff_kv_num_heads = (params.kv_num_heads > 0)
            ? params.kv_num_heads
            : params.num_heads;
    const int64_t repeat_factor = num_heads / eff_kv_num_heads;
    const int64_t seq_len_q = params.seq_len;
    const int64_t seq_len_kv
            = (params.kv_seq_len > 0) ? params.kv_seq_len : params.seq_len;
    const int64_t head_dim = params.head_dim;
    const bool is_causal = params.is_causal;
    const bool sliding_window = params.sliding_window;
    const int64_t sliding_window_size = params.sliding_window_size;
    // Match flash_sdpa: scale == 0 means use 1/sqrt(head_dim).
    const float scale = (params.scale != 0.0)
            ? static_cast<float>(params.scale)
            : 1.0f / std::sqrt(static_cast<float>(head_dim));

    if (batch <= 0 || num_heads <= 0 || seq_len_q <= 0 || seq_len_kv <= 0
            || head_dim <= 0 || eff_kv_num_heads <= 0
            || num_heads % eff_kv_num_heads != 0) {
        return status_t::failure;
    }

    // Resolve mask layout: 2D [S_q, S_kv] or 4D [B|1, H|1, S_q, S_kv] are all
    // supported; stride_b / stride_h are 0 for broadcast dimensions.
    if (has_mask) {
        mask_base_void = const_cast<void *>(attn_mask);
        mask_layout = compute_mask_strides(
                params.mask_sizes, params.mask_ndims, seq_len_q, seq_len_kv);
    }

    const qkv_t *q_data = static_cast<const qkv_t *>(query);
    const qkv_t *k_data = static_cast<const qkv_t *>(key);
    const qkv_t *v_data = static_cast<const qkv_t *>(value);
    qkv_t *out_data = static_cast<qkv_t *>(output);

    // Per-tensor BHSD strides. Tensor is logically [B, H, S, D]; the actual
    // physical layout follows the BHSD or BSHD stride conventions above.
    const int64_t q_sb = params.q_stride_b;
    const int64_t q_sh = params.q_stride_h;
    const int64_t q_ss = params.q_stride_s;
    const int64_t k_sb = params.k_stride_b;
    const int64_t k_sh = params.k_stride_h;
    const int64_t k_ss = params.k_stride_s;
    const int64_t v_sb = params.v_stride_b;
    const int64_t v_sh = params.v_stride_h;
    const int64_t v_ss = params.v_stride_s;
    const int64_t o_sb = params.o_stride_b;
    const int64_t o_sh = params.o_stride_h;
    const int64_t o_ss = params.o_stride_s;

    const int32_t num_threads = resolve_num_threads(
            params.num_threads, thread_guard::max_threads());

    auto run_per_head_loop = [&](auto mask_tag) -> status_t {
        using mask_t = decltype(mask_tag);
        const mask_t *mask_base = static_cast<const mask_t *>(mask_base_void);
// Per (b, h) base pointers come from each tensor's own (batch, head)
// strides; the inner [seq, head_dim] slab is then walked using each
// tensor's seq stride (head_dim for BHSD, num_heads*head_dim for BSHD).
// Per (b, h): scores [S_q, S_kv] (FP32) -> output [S_q, D] (qkv_t).
#pragma omp parallel for collapse(2) num_threads(num_threads)
        for (int64_t b = 0; b < batch; b++) {
            for (int64_t h = 0; h < num_heads; h++) {
                const int64_t kv_h = h / repeat_factor;
                const qkv_t *q_bh = q_data + b * q_sb + h * q_sh;
                const qkv_t *k_bh = k_data + b * k_sb + kv_h * k_sh;
                const qkv_t *v_bh = v_data + b * v_sb + kv_h * v_sh;
                qkv_t *o_bh = out_data + b * o_sb + h * o_sh;
                const mask_t *mask_for_bh = has_mask
                        ? mask_base + b * mask_layout.stride_b
                                + h * mask_layout.stride_h
                        : nullptr;
                compute_sdpa_per_head<qkv_t, mask_t>(q_bh, k_bh, v_bh, o_bh,
                        mask_for_bh, seq_len_q, seq_len_kv, head_dim, q_ss,
                        k_ss, v_ss, o_ss, scale, is_causal, has_mask,
                        sliding_window, sliding_window_size);
            }
        }
        return status_t::success;
    };

    // No-mask and FP32-mask paths share the mask_t = float instantiation
    // (mask_for_bh is nullptr in the no-mask case, so mask_t is never read).
    if (!has_mask || mask_dtype == data_type_t::f32) {
        return run_per_head_loop(float {});
    }

    // Reduced-precision mask is only valid when the mask dtype matches the
    // QKV dtype (enforced by @c validate_flash_sdpa_inputs()): bf16 mask with
    // bf16 QKV, f16 mask
    // with f16 QKV. The `if constexpr` blocks keep each reduced-precision
    // mask instantiation out of the QKV specialisations that don't accept it,
    // so the compiler can prove the FP32 specialisation never reaches the
    // unsupported-mask error path with a reduced-precision mask.
    if constexpr (std::is_same_v<qkv_t, bfloat16_t>) {
        if (mask_dtype == data_type_t::bf16) {
            return run_per_head_loop(bfloat16_t {});
        }
    }
    if constexpr (std::is_same_v<qkv_t, float16_t>) {
        if (mask_dtype == data_type_t::f16) {
            return run_per_head_loop(float16_t {});
        }
    }

    // validate_flash_sdpa_inputs() should have rejected any other mask dtype
    // before reaching execute(); fail closed if we somehow get here with an
    // unsupported one.
    apilog_error("SDPA ref kernel: unsupported mask data_type = ",
            static_cast<int>(mask_dtype),
            " for QKV data_type = ", static_cast<int>(params.qkv_dt));
    return status_t::failure;
}

status_t reference_sdpa(const void *query, const void *key, const void *value,
        const void *attn_mask, void *output, const sdpa_params &params) {
    status_t status = zendnnl::common::op_instrumentation::validate([&]() {
        return validate_flash_sdpa_inputs(
                query, key, value, output, attn_mask, params);
    });
    if (status != status_t::success) { return status; }

    // matmul_sv stores with output[i * o_seq_stride + j], i.e. implicit
    // o_stride_d == 1. validate_flash_sdpa_inputs() enforces this for Q/K/V
    // but only requires o_stride_d > 0 (the flash backend supports strided D).
    if (params.o_stride_d != 1) {
        apilog_error(
                "SDPA ref kernel: o_stride_d must be 1 (contiguous head_dim)");
        return status_t::failure;
    }

    const auto query_dtype = params.qkv_dt;

    if (query_dtype == data_type_t::f32) {
        return execute_typed<float>(
                query, key, value, attn_mask, output, params);
    }
    if (query_dtype == data_type_t::bf16) {
        return execute_typed<bfloat16_t>(
                query, key, value, attn_mask, output, params);
    }
    if (query_dtype == data_type_t::f16) {
        return execute_typed<float16_t>(
                query, key, value, attn_mask, output, params);
    }

    apilog_error("SDPA ref kernel: unsupported QKV data_type = ",
            static_cast<int>(query_dtype), " (expected f32, bf16, or f16)");
    return status_t::failure;
}

} //namespace sdpa
} //namespace lowoha
} //namespace zendnnl
