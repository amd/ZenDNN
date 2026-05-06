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
#ifndef _SDPA_ENCODER_KERNEL_HELPERS_HPP_
#define _SDPA_ENCODER_KERNEL_HELPERS_HPP_

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "common/bfloat16.hpp"
#include "common/zendnnl_global.hpp"

namespace zendnnl {
namespace ops {
namespace sdpa_encoder_ref {

/**
 * @brief Per-(batch, head) strides for an attention mask tensor.
 *
 * The reference kernel assumes the inner [seq_len_q, seq_len_kv] slab is
 * contiguous (row-major, kv stride = 1, q stride = seq_len_kv). Only the
 * leading batch / head dimensions can be broadcast — the @c stride_b and
 * @c stride_h fields are 0 when the corresponding dimension broadcasts,
 * and the contiguous element count when it does not.
 */
struct mask_layout {
  int64_t stride_b;
  int64_t stride_h;
};

/**
 * @brief Derive (batch, head) broadcast strides from a mask tensor shape.
 *
 * Supported shapes:
 *   - 2D [S_q, S_kv]                    -> stride_b = stride_h = 0 (broadcast)
 *   - 4D [B,   H,   S_q, S_kv]          -> stride_b = H*S_q*S_kv, stride_h = S_q*S_kv
 *   - 4D [1,   H,   S_q, S_kv]          -> stride_b = 0,           stride_h = S_q*S_kv
 *   - 4D [B,   1,   S_q, S_kv]          -> stride_b = S_q*S_kv,    stride_h = 0
 *   - 4D [1,   1,   S_q, S_kv]          -> stride_b = 0,           stride_h = 0
 *
 * @param mask_shape Mask tensor shape vector (size must be 2 or 4).
 * @param seq_len_q  Q sequence length (rows of the inner slab).
 * @param seq_len_kv K/V sequence length (cols of the inner slab).
 * @return mask_layout with stride_b / stride_h in elements (not bytes).
 *
 * @note Caller is responsible for validating that mask_shape's leading
 *       dims are either 1 or match the corresponding Q dim (the SDPA
 *       operator's @c validate() does this up front).
 */
inline mask_layout compute_mask_strides(
  const std::vector<uint64_t> &mask_shape,
  int64_t seq_len_q, int64_t seq_len_kv) {
  mask_layout layout{0, 0};
  const int64_t inner = seq_len_q * seq_len_kv;
  if (mask_shape.size() == 2) {
    return layout;  // 2D mask broadcasts across both batch and heads.
  }
  // 4D: leading dim of size 1 means broadcast; otherwise the matching
  // physical stride is mask_shape[1] * inner (batch) or inner (head).
  layout.stride_h = (mask_shape[1] != 1) ? inner : 0;
  layout.stride_b = (mask_shape[0] != 1)
                    ? static_cast<int64_t>(mask_shape[1]) * inner
                    : 0;
  return layout;
}

/**
 * @file
 * @brief Templated reference helpers shared by the FP32/BF16 SDPA encoder
 *        kernels.
 *
 * The reference kernels keep the same numerical recipe regardless of the
 * input dtype: typed loads/stores at the I/O boundary, all arithmetic in
 * float (so softmax is numerically stable for every input dtype). These
 * helpers express that recipe once, parametrised on the Q/K/V/output type.
 */

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
 * Q/K are typed (FP32 or BF16); the score buffer is always FP32 to keep
 * softmax numerically stable for low-precision inputs.
 *
 * Layout (per (batch, head) slice; the head_dim axis must be physically
 * contiguous when @p head_dim > 1, i.e. the underlying tensor's innermost
 * stride == 1. When @p head_dim == 1 the inner head_dim loop runs once
 * and the innermost stride is dead, matching the operator validator's
 * size-1 relaxation):
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
                      float *attention_scores,
                      int64_t seq_len_q, int64_t seq_len_kv,
                      int64_t head_dim,
                      int64_t q_seq_stride, int64_t k_seq_stride,
                      float scale) {
  for (int64_t i = 0; i < seq_len_q; i++) {
    for (int64_t j = 0; j < seq_len_kv; j++) {
      float sum = 0.0f;
      for (int64_t k = 0; k < head_dim; k++) {
        sum += to_float(q_data[i * q_seq_stride + k]) *
               to_float(k_data[j * k_seq_stride + k]);
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
 * and the innermost stride is dead, matching the operator validator's
 * size-1 relaxation):
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
        sum += scores[i * seq_len_kv + k] *
               to_float(v_data[k * v_seq_stride + j]);
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
    float sum = 0.0f;
    for (int64_t j = 0; j < seq_len_kv; j++) {
      scores[i * seq_len_kv + j] = std::exp(scores[i * seq_len_kv + j] - max_val);
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
inline void apply_causal_mask(float *attention_scores,
                              int64_t seq_len_q, int64_t seq_len_kv) {
  constexpr float neg_inf = -std::numeric_limits<float>::infinity();
  for (int64_t i = 0; i < seq_len_q; i++) {
    for (int64_t j = i + 1; j < seq_len_kv; j++) {
      attention_scores[i * seq_len_kv + j] = neg_inf;
    }
  }
}

/**
 * @brief Apply an additive attention mask in place.
 *
 * scores += mask. Mask values are typically 0 (attend) or -inf (ignore).
 *
 * The mask buffer's element type @p mask_t may differ from FP32 (e.g. BF16);
 * each loaded element is converted to float via @c to_float<mask_t> before
 * being added so the score buffer stays in FP32 for numerical stability.
 *
 * @tparam mask_t           Mask element type (float or bfloat16_t).
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
                                 const mask_t *mask_ptr,
                                 int64_t seq_len_q, int64_t seq_len_kv) {
  for (int64_t i = 0; i < seq_len_q; i++) {
    for (int64_t j = 0; j < seq_len_kv; j++) {
      attention_scores[i * seq_len_kv + j] +=
        to_float(mask_ptr[i * seq_len_kv + j]);
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
 * and the innermost stride is dead, matching the operator validator's
 * size-1 relaxation.
 *
 * @tparam qkv_t        Q/K/V/output element type (float or bfloat16_t).
 * @tparam mask_t       Mask element type (float or bfloat16_t). Independent of
 *                      @p qkv_t — bf16 QKV may be combined with either an
 *                      f32 or a bf16 mask; f32 QKV always pairs with f32 mask
 *                      (enforced by the operator's validate()).
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
 */
template <typename qkv_t, typename mask_t>
inline void compute_sdpa_per_head(const qkv_t *q_new, const qkv_t *k_new,
                                  const qkv_t *v_new, qkv_t *out_new,
                                  const mask_t *mask_ptr,
                                  int64_t seq_len_q, int64_t seq_len_kv,
                                  int64_t head_dim,
                                  int64_t q_seq_stride, int64_t k_seq_stride,
                                  int64_t v_seq_stride, int64_t o_seq_stride,
                                  float scale,
                                  bool is_causal, bool has_mask) {
  // FP32 score buffer keeps softmax numerically stable for low-precision QKV.
  std::vector<float> attention_scores(
    static_cast<size_t>(seq_len_q * seq_len_kv), 0.0f);

  matmul_qk<qkv_t>(q_new, k_new, attention_scores.data(),
                   seq_len_q, seq_len_kv, head_dim,
                   q_seq_stride, k_seq_stride, scale);

  if (is_causal) {
    apply_causal_mask(attention_scores.data(), seq_len_q, seq_len_kv);
  }
  if (has_mask && mask_ptr != nullptr) {
    apply_attention_mask<mask_t>(attention_scores.data(), mask_ptr,
                                 seq_len_q, seq_len_kv);
  }

  softmax(attention_scores.data(), seq_len_q, seq_len_kv);

  matmul_sv<qkv_t>(attention_scores.data(), v_new, out_new,
                   seq_len_q, seq_len_kv, head_dim,
                   v_seq_stride, o_seq_stride);
}

} // namespace sdpa_encoder_ref
} // namespace ops
} // namespace zendnnl

#endif // _SDPA_ENCODER_KERNEL_HELPERS_HPP_
