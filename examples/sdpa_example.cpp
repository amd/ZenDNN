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
#include <chrono>
#include <cmath>
#include <iomanip>

namespace zendnnl {
namespace examples {

using namespace zendnnl::interface;

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