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

#include "sdpa_encoder_fp32_kernel.hpp"
#include <limits>

namespace zendnnl {
namespace ops {
using namespace zendnnl::error_handling;

// Computes Q @ K^T: [S, D] x [S, D]^T -> [S, S]
void matmul_qk(float *q_data, float *k_data, float *attention_scores,
               int seq_len, int head_dim, float scale) {
  for (int i = 0; i < seq_len; i++) {
    for (int j = 0; j < seq_len; j++) {
      float sum = 0.0f;
      for (int k = 0; k < head_dim; k++) {
        sum += q_data[i*head_dim+k] * k_data[j*head_dim+k];
      }
      attention_scores[i*seq_len+j] = sum * scale;
    }
  }
}

// Computes scores @ V: [S, S] x [S, D] -> [S, D]
void matmul_sv(float *scores, float *v_data, float *output, int seq_len,
               int head_dim) {
  for (int i = 0; i < seq_len; i++) {
    for (int j = 0; j < head_dim; j++) {
      float sum = 0.0f;
      for (int k = 0; k < seq_len; k++) {
        sum += scores[i*seq_len+k] * v_data[k*head_dim+j];
      }
      output[i*head_dim+j] = sum;
    }
  }
}

void softmax(float *scores, int seq_len) {
  for (int i = 0; i < seq_len; i++) {
    float max_val = scores[i * seq_len];
    for (int j = 1; j < seq_len; j++) {
      if (scores[i * seq_len + j] > max_val) {
        max_val = scores[i * seq_len + j];
      }
    }
    float sum = 0.0f;
    for (int j = 0; j < seq_len; j++) {
      scores[i * seq_len + j] = std::exp(scores[i * seq_len + j] - max_val);
      sum += scores[i * seq_len + j];
    }
    for (int j = 0; j < seq_len; j++) {
      scores[i * seq_len + j] /= sum;
    }
  }
}


// Apply causal mask and/or attention mask
// Causal: set scores[i][j] = -inf where j > i (can't attend to future)
// Mask: add mask values (typically 0 for attend, -inf for don't attend)
void apply_causal_mask(float *attention_scores, int seq_len) {
  constexpr float neg_inf = -std::numeric_limits<float>::infinity();
  for (int i = 0; i < seq_len; i++) {
    for (int j = i + 1; j < seq_len; j++) {
      attention_scores[i * seq_len + j] = neg_inf;
    }
  }
}

void apply_attention_mask(float *attention_scores, float *mask_ptr,
                          int seq_len) {
  for (int i = 0; i < seq_len; i++) {
    for (int j = 0; j < seq_len; j++) {
      // Additive mask: mask contains 0 for attend, -inf for don't attend
      attention_scores[i * seq_len + j] += mask_ptr[i * seq_len + j];
    }
  }
}
status_t sdpa_encoder_fp32_kernel_t::execute(const context_type &context_,
    tensor_map_type &inputs_,
    tensor_map_type &outputs_) {

  // Read input tensors from context
  auto query_opt = context_.get_param("query");
  auto key_opt = context_.get_param("key");
  auto value_opt = context_.get_param("value");
  auto mask_opt = context_.get_param("mask");

  if (!query_opt || !key_opt || !value_opt) {
    return status_t::failure;
  }

  auto query = query_opt.value();
  auto key = key_opt.value();
  auto value = value_opt.value();
  bool has_mask = context_.get_has_mask();
  float *mask_ptr = nullptr;
  if (has_mask) {
    auto mask = mask_opt.value();
    mask_ptr = static_cast<float *>(mask.get_raw_handle_unsafe());
  }

  // Read output tensor
  auto output = outputs_["sdpa_output"];

  float scale = context_.get_scale();

  // Get tensor dimensions [B, H, S, D]
  int batch = query.get_size(0);
  int num_heads = query.get_size(1);
  int seq_len = query.get_size(2);
  int head_dim = query.get_size(3);
  bool is_causal = context_.get_is_causal();

  if (batch <= 0 || num_heads <= 0 || seq_len <= 0 || head_dim <= 0) {
    return status_t::failure;
  }

  float *q_data = static_cast<float *>(query.get_raw_handle_unsafe());
  float *k_data = static_cast<float *>(key.get_raw_handle_unsafe());
  float *v_data = static_cast<float *>(value.get_raw_handle_unsafe());
  float *out_data = static_cast<float *>(output.get_raw_handle_unsafe());

  // Compute Q @ K^T with scaling
  // Q: [B, H, S, D], K: [B, H, S, D] -> scores: [B, H, S, S]
  #pragma omp parallel for collapse(2)
  for (int b = 0; b < batch; b++) {
    for (int h = 0; h < num_heads; h++) {
      // Allocate temporary buffer for attention scores: [S, S] per thread
      int scores_size = seq_len * seq_len;
      std::vector<float> attention_scores(scores_size, 0.0f);

      int offset = b * num_heads + h;
      float *q_new = q_data + offset * seq_len * head_dim;
      float *k_new = k_data + offset * seq_len * head_dim;
      matmul_qk(q_new, k_new, attention_scores.data(), seq_len, head_dim, scale);

      // Apply causal mask if needed (set future positions to -inf)
      if (is_causal) {
        apply_causal_mask(attention_scores.data(), seq_len);
      }
      // Apply attention mask if provided
      if (has_mask) {
        apply_attention_mask(attention_scores.data(), mask_ptr, seq_len);
      }

      softmax(attention_scores.data(), seq_len);
      // Compute attention output: scores @ V
      float *v_new = v_data + offset * seq_len * head_dim;
      float *out_new = out_data + offset * seq_len * head_dim;
      matmul_sv(attention_scores.data(), v_new, out_new, seq_len, head_dim);
    }
  }

  return status_t::success;
}

} //namespace ops
} //namespace zendnnl

extern "C" {
  zendnnl::ops::sdpa_encoder_fp32_kernel_t *get_sdpa_encoder_fp32_kernel() {
    return new zendnnl::ops::sdpa_encoder_fp32_kernel_t();
  }
}