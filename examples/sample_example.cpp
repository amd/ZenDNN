/********************************************************************************
# * Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "sample_example.hpp"

namespace zendnnl {
namespace examples {

//using namespace zendnnl::interface;
using namespace zendnnl::error_handling;
using namespace zendnnl::common;
using namespace zendnnl::memory;
using namespace zendnnl::ops;

int sample_f32_kernel_example() {
  log_info("**sample operator f32 kernel example.");
  try {
    tensor_factory_t tensor_factory;
    auto param_tensor  = tensor_factory.uniform_tensor({ROWS, COLS},
                                                       data_type_t::f32,
                                                       0.5);
    auto input_tensor  = tensor_factory.uniform_tensor({ROWS, COLS},
                                                       data_type_t::f32,
                                                       0.5);
    auto output_tensor = tensor_factory.zero_tensor({ROWS, COLS},
                                                    data_type_t::f32);
    auto relu_post_op  = post_op_t{post_op_type_t::relu};

    auto sample_context = sample_context_t()
      .set_param("sample_param", param_tensor)
      .set_post_op(relu_post_op)
      .create();

    auto sample_operator = sample_operator_t()
      .set_name("sample operator")
      .set_context(sample_context)
      .create();

    if (! sample_operator.check()) {
      log_error("operator ", sample_operator.get_name(), " creation failed");
      return NOT_OK;
    }

    sample_operator
      .set_input("sample_input", input_tensor)
      .set_output("sample_output", output_tensor)
      .execute();
  }  catch(const exception_t& ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

int sample_bf16_kernel_example() {
  log_info("**sample operator bf16 kernel example.");
  try {
    tensor_factory_t tensor_factory;
    auto param_tensor  = tensor_factory.uniform_tensor({ROWS, COLS},
                                                       data_type_t::bf16,
                                                       0.5);
    auto input_tensor  = tensor_factory.uniform_tensor({ROWS, COLS},
                                                       data_type_t::bf16,
                                                       0.5);
    auto output_tensor = tensor_factory.zero_tensor({ROWS, COLS},
                                                    data_type_t::bf16);
    auto relu_post_op  = post_op_t{post_op_type_t::relu};

    auto sample_context = sample_context_t()
      .set_param("sample_param", param_tensor)
      .set_post_op(relu_post_op)
      .create();

    auto sample_operator = sample_operator_t()
      .set_name("sample operator")
      .set_context(sample_context)
      .create();

    if (! sample_operator.check()) {
      log_error("operator ", sample_operator.get_name(), " creation failed");
      return NOT_OK;
    }

    sample_operator
      .set_input("sample_input", input_tensor)
      .set_output("sample_output", output_tensor)
      .execute();
  }  catch(const exception_t& ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

} //examples
} //zendnnl
