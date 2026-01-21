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

#include "lowoha_sdpa.hpp"
#include "lowoha_operators/matmul/lowoha_matmul.hpp"
#include "lowoha_operators/softmax/lowoha_softmax.hpp"
#include <vector>
#include <limits>

namespace zendnnl {
namespace lowoha {
namespace sdpa {

status_t sdpa_direct(
  const void *query,
  const void *key,
  const void *value,
  const void *attn_mask,
  void *output,
  sdpa_params &params
) {
  // Create profiler instance for timing
  zendnnl::profile::profiler_t profiler;
  bool is_profile = is_profile_enabled();
  if (is_profile) {
    profiler.tbp_start();
  }

  // Validate inputs
  if (validate_sdpa_inputs(query, key, value, output, params)
      != status_t::success) {
    return status_t::failure;
  }

  // Log API call
  [[maybe_unused]] std::ostringstream ss;
  if (apilog_info_enabled() || is_profile) {
    ss << "LOWOHA sdpa_direct: batch=" << params.batch
       << ", num_heads=" << params.num_heads
       << ", seq_len=" << params.seq_len
       << ", head_dim=" << params.head_dim
       << ", scale=" << params.scale
       << ", is_causal=" << (params.is_causal ? "true" : "false")
       << ", has_mask=" << (params.has_attn_mask ? "true" : "false")
       << ", q_dt=" << static_cast<int>(params.q_dt)
       << ", out_dt=" << static_cast<int>(params.out_dt);
  }
  apilog_info(ss.str());

  // Calculate dimensions
  const uint64_t batch = params.batch;
  const uint64_t num_heads = params.num_heads;
  const uint64_t seq_len = params.seq_len;
  const uint64_t head_dim = params.head_dim;
  const uint64_t batch_heads = batch * num_heads;

  // Determine element size based on data type
  size_t elem_size = (params.q_dt == data_type_t::bf16) ? sizeof(
                       uint16_t) : sizeof(float);

  // Allocate intermediate buffer for scores: [batch * num_heads, seq_len, seq_len]
  const uint64_t scores_size = batch_heads * seq_len * seq_len;
  std::vector<char> scores_buffer(scores_size * elem_size);
  void *scores = scores_buffer.data();

  // =========================================================================
  // Step 1: BMM - scores = Q @ K^T
  // Q: [batch * num_heads, seq_len, head_dim]
  // K: [batch * num_heads, seq_len, head_dim] -> K^T: [batch * num_heads, head_dim, seq_len]
  // scores: [batch * num_heads, seq_len, seq_len]
  // =========================================================================
  {
    matmul::matmul_params mm_params;
    mm_params.dtypes.src = params.q_dt;
    mm_params.dtypes.wei = params.k_dt;
    mm_params.dtypes.dst = params.q_dt;  // scores same type as Q
    mm_params.num_threads = params.num_threads;

    matmul::matmul_batch_params_t batch_params;
    batch_params.Batch_A = batch_heads;
    batch_params.Batch_B = batch_heads;
    // Let matmul calculate strides automatically
    batch_params.batch_stride_src = -1;
    batch_params.batch_stride_wei = -1;
    batch_params.batch_stride_dst = -1;

    // Q @ K^T: M=seq_len, N=seq_len, K=head_dim, transB=true
    status_t status = matmul::matmul_direct(
                        'r',                    // layout: row-major
                        false,                  // transA: false
                        true,                   // transB: true (K^T)
                        seq_len,                // M
                        seq_len,                // N
                        head_dim,               // K
                        params.scale,           // alpha = scale factor
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
  // Step 2: Apply attention mask (if provided) or causal mask
  // scores = scores + mask (where mask is -inf for masked positions)
  // =========================================================================
  if (params.is_causal || params.has_attn_mask) {
    float neg_inf = -std::numeric_limits<float>::infinity();

    if (params.q_dt == data_type_t::f32) {
      float *scores_f32 = static_cast<float *>(scores);
      const float *mask_f32 = static_cast<const float *>(attn_mask);

      #pragma omp parallel for collapse(2)
      for (uint64_t bh = 0; bh < batch_heads; ++bh) {
        for (uint64_t i = 0; i < seq_len; ++i) {
          for (uint64_t j = 0; j < seq_len; ++j) {
            uint64_t idx = bh * seq_len * seq_len + i * seq_len + j;

            // Apply causal mask: mask out future positions (j > i)
            if (params.is_causal && j > i) {
              scores_f32[idx] = neg_inf;
            }
            // Apply custom attention mask
            else if (params.has_attn_mask && mask_f32 != nullptr) {
              scores_f32[idx] += mask_f32[idx];
            }
          }
        }
      }
    }
    // TODO: Add BF16 support for masking
  }

  // =========================================================================
  // Step 3: Softmax on scores
  // attn_weights = softmax(scores, dim=-1)
  // =========================================================================
  {
    softmax::softmax_params sm_params;
    sm_params.batch = batch_heads * seq_len;  // Flatten batch and seq_len_q
    sm_params.axis_dim = seq_len;             // Softmax over last dimension
    sm_params.axis = -1;
    sm_params.log_softmax = false;
    sm_params.src_dt = params.q_dt;
    sm_params.dst_dt = params.q_dt;
    sm_params.num_threads = params.num_threads;
    sm_params.algorithm = softmax::softmax_algo_t::onednn;

    // Setup shape for OneDNN backend (required)
    sm_params.ndims = 2;
    sm_params.shape[0] = batch_heads * seq_len;
    sm_params.shape[1] = seq_len;

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
    mm_params.dtypes.src = params.q_dt;  // attn_weights same type as Q
    mm_params.dtypes.wei = params.v_dt;
    mm_params.dtypes.dst = params.out_dt;
    mm_params.num_threads = params.num_threads;

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

  if (is_profile) {
    profiler.tbp_stop();
    profilelog_verbose(ss.str(), ", time=", profiler.tbp_elapsedtime(),
                       profiler.get_res_str());
  }

  return status_t::success;
}

} // namespace sdpa
} // namespace lowoha
} // namespace zendnnl
