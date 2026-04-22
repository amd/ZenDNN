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
#include "sdpa_example.hpp"
#include <cmath>
#include <cstdint>
#include <vector>

namespace zendnnl {
namespace examples {

using namespace zendnnl::interface;
using zendnnl::lowoha::sdpa::sdpa_direct;
using zendnnl::lowoha::sdpa::sdpa_params;

int sdpa_direct_example() {
  const int64_t b = static_cast<int64_t>(BS);
  const int64_t h = static_cast<int64_t>(NUM_HEADS);
  const int64_t s = static_cast<int64_t>(SEQ_LEN);
  const int64_t d = static_cast<int64_t>(HEAD_DIM);
  const size_t elems = static_cast<size_t>(b * h * s * d);

  std::vector<float> q(elems);
  std::vector<float> k(elems);
  std::vector<float> v(elems);
  std::vector<float> out(elems, 0.0f);

  for (size_t i = 0; i < elems; ++i) {
    const float x = 0.01f * static_cast<float>(static_cast<int>(i % 17) - 8);
    q[i] = x;
    k[i] = x * 1.03f;
    v[i] = x * 0.97f;
  }

  sdpa_params params{};
  params.batch     = b;
  params.num_heads = h;
  params.seq_len   = s;
  params.head_dim  = d;

  params.q_stride_d = 1;
  params.q_stride_s = d;
  params.q_stride_h = s * d;
  params.q_stride_b = h * s * d;
  params.k_stride_d = 1;
  params.k_stride_s = d;
  params.k_stride_h = s * d;
  params.k_stride_b = h * s * d;
  params.v_stride_d = 1;
  params.v_stride_s = d;
  params.v_stride_h = s * d;
  params.v_stride_b = h * s * d;
  params.o_stride_d = 1;
  params.o_stride_s = d;
  params.o_stride_h = s * d;
  params.o_stride_b = h * s * d;

  params.qkv_dt  = data_type_t::f32;
  params.out_dt  = data_type_t::f32;
  params.mask_dt = data_type_t::none;
  params.scale   = 1.0 / std::sqrt(static_cast<double>(d));
  params.is_causal = false;
  params.dropout_p = 0.0;
  params.num_threads = 0;

  if (sdpa_direct(q.data(), k.data(), v.data(), nullptr,
                  out.data(), params) != status_t::success) {
    testlog_error("sdpa_direct_example: sdpa_direct failed");
    return NOT_OK;
  }

  return OK;
}

int sdpa_example() {
  try {

    tensor_factory_t tensor_factory;

    // Create Q, K, V tensors with dimensions [B, H, S, D]
    auto query_tensor = tensor_factory.uniform_tensor({BS, NUM_HEADS, SEQ_LEN, HEAD_DIM},
                        data_type_t::f32, 0.1f, "query");
    auto key_tensor = tensor_factory.uniform_tensor({BS, NUM_HEADS, SEQ_LEN, HEAD_DIM},
                      data_type_t::f32, 0.1f, "key");
    auto value_tensor = tensor_factory.uniform_tensor({BS, NUM_HEADS, SEQ_LEN, HEAD_DIM},
                        data_type_t::f32, 0.1f, "value");

    // Create output tensor
    auto output_tensor = tensor_factory.zero_tensor({BS, NUM_HEADS, SEQ_LEN, HEAD_DIM},
                         data_type_t::f32, "output");

    // Create SDPA encoder context with default parameters
    float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
    auto sdpa_context = sdpa_encoder_context_t()
                        .set_param("query", query_tensor)
                        .set_param("key", key_tensor)
                        .set_param("value", value_tensor)
                        .set_scale(scale)
                        .set_is_dropout(false)
                        .set_is_causal(false)
                        .set_has_mask(false)
                        .create();

    if (!sdpa_context.check()) {
      testlog_error("SDPA encoder context creation failed");
      return NOT_OK;
    }

    // Create SDPA encoder operator
    auto sdpa_operator = sdpa_encoder_operator_t()
                         .set_name("SDPA Encoder FP32")
                         .set_context(sdpa_context)
                         .create();

    if (sdpa_operator.is_bad_object()) {
      log_error("SDPA encoder operator creation failed");
      return NOT_OK;
    }

    sdpa_operator
    .set_output("sdpa_output", output_tensor)
    .execute();

  }
  catch (const exception_t &ex) {
    std::cout << "Exception: " << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

} //examples
} //zendnnl