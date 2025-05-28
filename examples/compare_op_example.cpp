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

#include "compare_op_example.hpp"

namespace zendnnl {
namespace examples {

using namespace zendnnl::interface;

int compare_operator_execute(tensor_t &input1, tensor_t &input2) {
  try {
    status_t status;
    tensor_factory_t tensor_factory;

    //define compare context
    auto compare_context = compare_context_t()
                           .set_tolerance(1e-07f)
                           .create();

    //define compare operator
    auto compare_operator = compare_operator_t()
                            .set_name("compare_operator")
                            .set_context(compare_context)
                            .create();

    if (! compare_operator.check()) {
      testlog_error("operator ", compare_operator.get_name(), " creation failed");
      return NOT_OK;
    }
    auto diff_tensor = tensor_factory.zero_tensor({MATMUL_M, MATMUL_N},
                       data_type_t::f32);
    diff_tensor.set_name("diff_tensor");

    status = compare_operator
             .set_input("expected_tensor", input1)
             .set_input("test_tensor", input2)
             .set_output("diff_tensor", diff_tensor)
             .execute();

    auto stats = compare_operator.get_compare_stats();
    if (status == status_t::success) {
      testlog_info("operator ", compare_operator.get_name(),
                   " execution successful.");
      testlog_verbose("Match Percent:",stats.match_percent, "%, ",
                      "Mean Deviation:",stats.mean_deviation,", ",
                      "Max Deviation:",stats.max_deviation,", ",
                      "Min Deviation:",stats.min_deviation,", ",
                      "output[", MATMUL_M/2, ",", MATMUL_N/2,"] = ",diff_tensor.at({MATMUL_M/2, MATMUL_N/2}));
    }
    else {
      testlog_info("operator ", compare_operator.get_name(), " execution failed");
      return NOT_OK;
    }

  }
  catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

int compare_op_example() {
  testlog_info("Compare operator example");
  try {
    tensor_factory_t tensor_factory;

    auto input_tensor_1 = tensor_factory.uniform_tensor({MATMUL_M, MATMUL_N},
                          data_type_t::f32,
                          2.0, "compare_input1");

    auto input_tensor_2 = tensor_factory.uniform_tensor({MATMUL_M, MATMUL_N},
                          data_type_t::f32,
                          2.0, "compare_input2");

    //Call to compare operator
    compare_operator_execute(input_tensor_1, input_tensor_2);
  }
  catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }
  return OK;
}

int compare_ref_and_aocl_matmul_kernel_example() {
  testlog_info("Compare ref and aocl matmul kernel example");

  try {
    tensor_factory_t tensor_factory;
    status_t status;
    auto weights = tensor_factory.uniform_tensor({MATMUL_K, MATMUL_N},
                   data_type_t::f32,
                   1.0, "weights");

    auto bias    = tensor_factory.uniform_tensor({MATMUL_N},
                   data_type_t::f32,
                   -10.0, "bias");

    auto relu_post_op = post_op_t{post_op_type_t::relu};

    //define matmul context
    auto matmul_context = matmul_context_t()
                          .set_param("weights", weights)
                          .set_param("bias", bias)
                          .set_post_op(relu_post_op)
                          .create();

    auto input_tensor = tensor_factory.uniform_tensor({MATMUL_M, MATMUL_K},
                        data_type_t::f32,
                        1.0, "matmul_input");

    auto output_tensor_ref = tensor_factory.zero_tensor({MATMUL_M, MATMUL_N},
                             data_type_t::f32,
                             "matmul_output");

    auto output_tensor_aocl = tensor_factory.zero_tensor({MATMUL_M, MATMUL_N},
                              data_type_t::f32,
                              "matmul_output");


    //Call to reference matmul kernel
    auto matmul_operator_ref = matmul_operator_t()
                               .set_name("matmul_forced_ref_operator")
                               .set_context(matmul_context)
                               .create();

    if (! matmul_operator_ref.check()) {
      testlog_error(" operator ", matmul_operator_ref.get_name(),
                    " creation failed.");
      return NOT_OK;
    }

    status = matmul_operator_ref
             .set_input("matmul_input", input_tensor)
             .set_output("matmul_output", output_tensor_ref)
             .set_forced_kernel("reference")
             .execute();
    if (status == status_t::success) {
      testlog_info("operator ", matmul_operator_ref.get_name(),
                   " execution successful.");
    }
    else {
      testlog_info("operator ", matmul_operator_ref.get_name(), " execution failed");
      return NOT_OK;
    }

    //Call to aocl matmul kernel
    auto matmul_operator = matmul_operator_t()
                           .set_name("matmul_f32_operator")
                           .set_context(matmul_context)
                           .create();
    if (! matmul_operator.check()) {
      testlog_error(" operator ", matmul_operator.get_name(), " creation failed");
      return NOT_OK;
    }
    status = matmul_operator
             .set_input("matmul_input", input_tensor)
             .set_output("matmul_output", output_tensor_aocl)
             .execute();
    if (status == status_t::success) {
      testlog_info("operator ", matmul_operator.get_name(), " execution successful");
    }
    else {
      testlog_info("operator ", matmul_operator.get_name(), " execution failed");
      return NOT_OK;
    }

    //Call to compare operator
    compare_operator_execute(output_tensor_ref, output_tensor_aocl);
  }
  catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

} //examples
} //zendnnl
