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

#include "lowoha_sdpa_bmm.hpp"
#include "lowoha_operators/common/omp_thread_control.hpp"
#include <vector>
#include <limits>
#include <cstdlib>

namespace zendnnl {
namespace lowoha {
namespace sdpa {

static thread_local void  *scores_ptr  = nullptr;
static thread_local size_t scores_size = 0;

void sdpa_free_scratch() {
  free(scores_ptr);
  scores_ptr  = nullptr;
  scores_size = 0;
}

status_t bmm_based_sdpa(
  const void *query,
  const void *key,
  const void *value,
  const void *attn_mask,
  void *output,
  sdpa_params &params
) {
  // Validate inputs
  if (validate_sdpa_inputs(query, key, value, output, params)
      != status_t::success) {
    return status_t::failure;
  }

  float scale = params.scale;
  if (scale == 0.0f) {
    scale = calculate_default_scale(params.head_dim);
  }

  // Calculate dimensions
  const int64_t batch = params.batch;
  const int64_t num_heads = params.num_heads;
  const int64_t seq_len = params.seq_len;
  const int64_t head_dim = params.head_dim;
  const int64_t batch_heads = batch * num_heads;
  const int32_t num_threads = resolve_num_threads(params.num_threads,
                              thread_guard::max_threads());

  // Determine element size based on data type
  size_t elem_size = (params.qkv_dt == data_type_t::bf16) ? sizeof(
                       uint16_t) : sizeof(float);


  // =========================================================================
  // Step 1 prep: Build masks to add to scores (Q @ K^T)
  // =========================================================================
  std::vector<matmul::matmul_post_op> mask_postops;
  if (attn_mask != nullptr) {
    matmul::matmul_post_op mask_po;
    mask_po.po_type = zendnnl::ops::post_op_type_t::binary_add;
    mask_po.buff = const_cast<void *>(attn_mask);
    mask_po.dtype = params.mask_dt == data_type_t::none ? data_type_t::f32
                    : params.mask_dt;

    // Use actual mask dims provided by caller (already reshaped to 3D)
    if (params.mask_ndims > 0) {
      for (int i = 0; i < params.mask_ndims; ++i) {
        mask_po.dims.push_back(params.mask_dims[i]);
      }
    }
    else {
      // Fallback: assume full [batch_heads, seq_len, seq_len]
      if (batch_heads == 1) {
        mask_po.dims = {static_cast<int64_t>(seq_len),
                        static_cast<int64_t>(seq_len)
                       };
      }
      else {
        mask_po.dims = {static_cast<int64_t>(batch_heads),
                        static_cast<int64_t>(seq_len),
                        static_cast<int64_t>(seq_len)
                       };
      }
    }
    mask_po.leading_dim = static_cast<int>(
                            mask_po.dims[mask_po.dims.size() - 1]);
    mask_postops.push_back(mask_po);
  }

  // Scratch buffer for scores: [batch * num_heads, seq_len, seq_len]
  // Reuses buffer across calls; only reallocates when a larger size is needed.
  // Call sdpa_free_scratch() to release.
  const size_t scores_bytes = static_cast<size_t>(batch_heads) * seq_len
                              * seq_len * elem_size;
  if (scores_size < scores_bytes) {
    free(scores_ptr);
    scores_ptr = malloc(scores_bytes);
    if (!scores_ptr) {
      scores_size = 0;
      log_error("SDPA: Failed to allocate scores buffer");
      return status_t::failure;
    }
    scores_size = scores_bytes;
  }
  void *scores = scores_ptr;

  // =========================================================================
  // Step 1: BMM - scores = Q @ K^T (+ attn_mask post-op if provided)
  // Q: [batch * num_heads, seq_len, head_dim]
  // K: [batch * num_heads, seq_len, head_dim] -> K^T: [batch * num_heads, head_dim, seq_len]
  // scores: [batch * num_heads, seq_len, seq_len]
  // =========================================================================
  {
    matmul::matmul_params mm_params;
    mm_params.dtypes.src = params.qkv_dt;
    mm_params.dtypes.wei = params.qkv_dt;
    mm_params.dtypes.dst = params.qkv_dt;
    mm_params.num_threads = num_threads;
    if (!mask_postops.empty()) {
      mm_params.postop_ = mask_postops;
    }

    matmul::matmul_batch_params_t batch_params;
    batch_params.Batch_A = batch_heads;
    batch_params.Batch_B = batch_heads;

    // Q @ K^T: M=seq_len, N=seq_len, K=head_dim, transB=true
    status_t status = matmul::matmul_direct(
                        'r',                    // layout: row-major
                        false,                  // transA: false
                        true,                   // transB: true (K^T)
                        seq_len,                // M
                        seq_len,                // N
                        head_dim,               // K
                        scale,                  // alpha = scale factor
                        query,                  // src (Q)
                        head_dim,               // lda
                        key,                    // weight (K)
                        head_dim,               // ldb
                        nullptr,                // bias
                        0.0f,                   // beta
                        scores,                 // dst (scores)
                        seq_len,                // ldc
                        false,                  // is_weights_const
                        batch_params,
                        mm_params
                      );

    if (status != status_t::success) {
      log_error("SDPA: BMM Q @ K^T failed");
      return status_t::failure;
    }
  }

  // =========================================================================
  // Step 1.6: Apply causal mask separately (after Q @ K^T)
  // =========================================================================
  if (params.is_causal) {
    if (params.qkv_dt != data_type_t::f32) {
      log_error("SDPA: Causal mask currently supports only f32 scores");
      return status_t::failure;
    }
    float *scores_f32 = static_cast<float *>(scores);
    const float neg_inf = -std::numeric_limits<float>::infinity();
    #pragma omp parallel for collapse(2) num_threads(num_threads)
    for (int64_t bh = 0; bh < batch_heads; ++bh) {
      for (int64_t i = 0; i < seq_len; ++i) {
        for (int64_t j = 0; j < seq_len; ++j) {
          int64_t idx = bh * seq_len * seq_len + i * seq_len + j;
          if (j > i) {
            scores_f32[idx] = neg_inf;
          }
        }
      }
    }
  }

  // =========================================================================
  // Step 2: Softmax on scores
  // attn_weights = softmax(scores, dim=-1)
  // =========================================================================
  {
    softmax::softmax_params sm_params;
    sm_params.batch = batch_heads * seq_len;
    sm_params.axis_dim = seq_len;
    sm_params.axis = -1;
    sm_params.log_softmax = false;
    sm_params.src_dt = params.qkv_dt;
    sm_params.dst_dt = params.qkv_dt;
    sm_params.num_threads = num_threads;
    sm_params.algorithm = softmax::softmax_algo_t::onednn;

    // 3D shape [batch_heads, seq_len, seq_len] — single onednn call for all heads
    sm_params.ndims = 3;
    sm_params.shape[0] = batch_heads;
    sm_params.shape[1] = seq_len;
    sm_params.shape[2] = seq_len;

    status_t status = softmax::softmax_direct(
                        scores,     // input (in-place)
                        scores,     // output (in-place)
                        sm_params
                      );

    if (status != status_t::success) {
      log_error("SDPA: Softmax failed");
      return status_t::failure;
    }
  }

  // =========================================================================
  // Step 4: BMM - output = attn_weights @ V
  // attn_weights: [batch * num_heads, seq_len, seq_len]
  // V: [batch * num_heads, seq_len, head_dim]
  // output: [batch * num_heads, seq_len, head_dim]
  // =========================================================================
  {
    matmul::matmul_params mm_params;
    mm_params.dtypes.src = params.qkv_dt;
    mm_params.dtypes.wei = params.qkv_dt;
    mm_params.dtypes.dst = params.out_dt;
    mm_params.num_threads = num_threads;

    matmul::matmul_batch_params_t batch_params;
    batch_params.Batch_A = batch_heads;
    batch_params.Batch_B = batch_heads;
    // Let matmul calculate strides automatically
    batch_params.batch_stride_src = -1;
    batch_params.batch_stride_wei = -1;
    batch_params.batch_stride_dst = -1;

    // attn_weights @ V: M=seq_len, N=head_dim, K=seq_len
    status_t status = matmul::matmul_direct(
                        'r',                    // layout: row-major
                        false,                  // transA: false
                        false,                  // transB: false
                        seq_len,                // M
                        head_dim,               // N
                        seq_len,                // K
                        1.0f,                   // alpha
                        scores,                 // src (attn_weights)
                        seq_len,                // lda
                        value,                  // weight (V)
                        head_dim,               // ldb
                        nullptr,                // bias
                        0.0f,                   // beta
                        output,                 // dst
                        head_dim,               // ldc
                        false,                  // is_weights_const
                        batch_params,
                        mm_params
                      );

    if (status != status_t::success) {
      log_error("SDPA: BMM attn_weights @ V failed");
      return status_t::failure;
    }
  }
  return status_t::success;
}

} // namespace sdpa
} // namespace lowoha
} // namespace zendnnl
