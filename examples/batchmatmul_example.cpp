/********************************************************************************
# * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "batchmatmul_example.hpp"

namespace zendnnl {
namespace examples {

using namespace zendnnl::interface;

int batch_matmul_relu_f32_kernel_example() {

  try {
    status_t status;
    tensor_factory_t tensor_factory;
    auto weights = tensor_factory.uniform_tensor({BATCH_SIZE, BATCH_MATMUL_K, BATCH_MATMUL_N},
                   data_type_t::f32,
                   1.0, "weights");

    auto bias    = tensor_factory.uniform_tensor({BATCH_MATMUL_N},
                   data_type_t::f32,
                   -10.0, "bias");

    auto relu_post_op = post_op_t{post_op_type_t::relu};

    //define batch_matmul context
    auto batch_matmul_context = matmul_context_t()
                                .set_param("weights", weights)
                                .set_param("bias", bias)
                                .set_post_op(relu_post_op)
                                .create();

    if (! batch_matmul_context.check()) {
      testlog_error("batch_matmul context creation failed");
      return NOT_OK;
    }

    //define batch_matmul operator
    auto batch_matmul_operator = matmul_operator_t()
                                 .set_name("batch_matmul_f32")
                                 .set_context(batch_matmul_context)
                                 .create();

    if (! batch_matmul_operator.check()) {
      testlog_error(" operator ", batch_matmul_operator.get_name(),
                    " creation failed.");
      return NOT_OK;
    }

    auto input_tensor = tensor_factory.uniform_tensor({BATCH_SIZE, BATCH_MATMUL_M, BATCH_MATMUL_K},
                        data_type_t::f32,
                        1.0, "matmul_input");

    auto output_tensor = tensor_factory.zero_tensor({BATCH_SIZE, BATCH_MATMUL_M, BATCH_MATMUL_N},
                         data_type_t::f32, "matmul_output");

    status = batch_matmul_operator
             .set_input("matmul_input", input_tensor)
             .set_output("matmul_output", output_tensor)
             .set_forced_kernel("aocl_blis")
             .execute();

    if (status == status_t::success) {
      testlog_info("<",batch_matmul_operator.get_name(),">",
                   " operator execution successful.");
    }
    else {
      testlog_error("<",batch_matmul_operator.get_name(),">",
                    " operator execution failed.");
      return NOT_OK;
    }

  }
  catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

int batch_matmul_wei2d_relu_f32_kernel_example() {

  try {
    status_t status;
    tensor_factory_t tensor_factory;
    auto weights = tensor_factory.uniform_tensor({BATCH_MATMUL_K, BATCH_MATMUL_N},
                   data_type_t::f32,
                   1.0, "weights");

    auto bias    = tensor_factory.uniform_tensor({BATCH_MATMUL_N},
                   data_type_t::f32,
                   -10.0, "bias");

    auto relu_post_op = post_op_t{post_op_type_t::relu};

    //define batch_matmul context
    auto batch_matmul_context = matmul_context_t()
                                .set_param("weights", weights)
                                .set_param("bias", bias)
                                .set_post_op(relu_post_op)
                                .create();

    if (! batch_matmul_context.check()) {
      testlog_error("batch_matmul context creation failed");
      return NOT_OK;
    }

    //define batch_matmul operator
    auto batch_matmul_operator = matmul_operator_t()
                                 .set_name("batch_matmul_f32")
                                 .set_context(batch_matmul_context)
                                 .create();

    if (! batch_matmul_operator.check()) {
      testlog_error(" operator ", batch_matmul_operator.get_name(),
                    " creation failed.");
      return NOT_OK;
    }

    auto input_tensor = tensor_factory.uniform_tensor({BATCH_SIZE, BATCH_MATMUL_M, BATCH_MATMUL_K},
                        data_type_t::f32,
                        1.0, "matmul_input");

    auto output_tensor = tensor_factory.zero_tensor({BATCH_SIZE, BATCH_MATMUL_M, BATCH_MATMUL_N},
                         data_type_t::f32, "matmul_output");

    status = batch_matmul_operator
             .set_input("matmul_input", input_tensor)
             .set_output("matmul_output", output_tensor)
             .set_forced_kernel("aocl_blis")
             .execute();

    if (status == status_t::success) {
      testlog_info("<",batch_matmul_operator.get_name(),">",
                   " operator execution successful.");
    }
    else {
      testlog_error("<",batch_matmul_operator.get_name(),">",
                    " operator execution failed.");
      return NOT_OK;
    }

  }
  catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

int batch_matmul_inp2d_relu_f32_kernel_example() {

  try {
    status_t status;
    tensor_factory_t tensor_factory;
    auto weights = tensor_factory.uniform_tensor({BATCH_SIZE, BATCH_MATMUL_K, BATCH_MATMUL_N},
                   data_type_t::f32,
                   1.0, "weights");

    auto bias    = tensor_factory.uniform_tensor({BATCH_MATMUL_N},
                   data_type_t::f32,
                   -10.0, "bias");

    auto relu_post_op = post_op_t{post_op_type_t::relu};

    //define batch_matmul context
    auto batch_matmul_context = matmul_context_t()
                                .set_param("weights", weights)
                                .set_param("bias", bias)
                                .set_post_op(relu_post_op)
                                .create();

    if (! batch_matmul_context.check()) {
      testlog_error("batch_matmul context creation failed");
      return NOT_OK;
    }

    //define batch_matmul operator
    auto batch_matmul_operator = matmul_operator_t()
                                 .set_name("batch_matmul_f32")
                                 .set_context(batch_matmul_context)
                                 .create();

    if (! batch_matmul_operator.check()) {
      testlog_error(" operator ", batch_matmul_operator.get_name(),
                    " creation failed.");
      return NOT_OK;
    }

    auto input_tensor = tensor_factory.uniform_tensor({BATCH_MATMUL_M, BATCH_MATMUL_K},
                        data_type_t::f32,
                        1.0, "matmul_input");

    auto output_tensor = tensor_factory.zero_tensor({BATCH_SIZE, BATCH_MATMUL_M, BATCH_MATMUL_N},
                         data_type_t::f32, "matmul_output");

    status = batch_matmul_operator
             .set_input("matmul_input", input_tensor)
             .set_output("matmul_output", output_tensor)
             .set_forced_kernel("aocl_blis")
             .execute();

    if (status == status_t::success) {
      testlog_info("<",batch_matmul_operator.get_name(),">",
                   " operator execution successful.");
    }
    else {
      testlog_error("<",batch_matmul_operator.get_name(),">",
                    " operator execution failed.");
      return NOT_OK;
    }

  }
  catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

int batch_matmul_relu_bf16_kernel_example() {

  try {
    status_t status;
    tensor_factory_t tensor_factory;
    auto weights = tensor_factory.uniform_tensor({BATCH_SIZE, BATCH_MATMUL_K, BATCH_MATMUL_N},
                   data_type_t::bf16,
                   1.0, "weights");

    auto bias    = tensor_factory.uniform_tensor({BATCH_MATMUL_N},
                   data_type_t::f32,
                   3.0, "bias");

    auto relu_post_op = post_op_t{post_op_type_t::relu};

    //define batch_matmul context
    auto batch_matmul_context = matmul_context_t()
                                .set_param("weights", weights)
                                .set_param("bias", bias)
                                .set_post_op(relu_post_op)
                                .create();

    if (! batch_matmul_context.check()) {
      testlog_error("batch_matmul context creation failed");
      return NOT_OK;
    }

    //define batch_matmul operator
    auto batch_matmul_operator = matmul_operator_t()
                                 .set_name("batch_matmul_bf16_operator")
                                 .set_context(batch_matmul_context)
                                 .create();

    if (! batch_matmul_operator.check()) {
      testlog_error(" operator ", batch_matmul_operator.get_name(),
                    " creation failed.");
      return NOT_OK;
    }

    auto input_tensor = tensor_factory.uniform_tensor({BATCH_SIZE, BATCH_MATMUL_M, BATCH_MATMUL_K},
                        data_type_t::bf16,
                        1.0, "matmul_input");

    auto output_tensor = tensor_factory.zero_tensor({BATCH_SIZE, BATCH_MATMUL_M, BATCH_MATMUL_N},
                         data_type_t::f32,
                         "matmul_output");

    status = batch_matmul_operator
             .set_input("matmul_input", input_tensor)
             .set_output("matmul_output", output_tensor)
             .set_forced_kernel("aocl_blis")
             .execute();

    if (status == status_t::success) {
      testlog_info("operator ", batch_matmul_operator.get_name(),
                   " execution successful.");
    }
    else {
      testlog_info("operator ", batch_matmul_operator.get_name(),
                   " execution failed.");
      return NOT_OK;
    }

  }
  catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return NOT_OK;
}

int batch_matmul_mul_silu_mul_f32_kernel_example() {
  testlog_info("**batch_matmul binary_mul + silu + binary_mul operator f32 kernel example.");

  try {
    status_t status;
    tensor_factory_t tensor_factory;
    auto weights = tensor_factory.uniform_tensor({BATCH_SIZE, BATCH_MATMUL_K, BATCH_MATMUL_N},
                   data_type_t::f32,
                   1.0, "weights");

    auto bias    = tensor_factory.uniform_tensor({BATCH_MATMUL_N},
                   data_type_t::f32,
                   3.0, "bias");

    binary_mul_params_t binary_mul;
    binary_mul.scale = 1.0;

    auto binary_mul_po   = post_op_t{binary_mul};
    auto silu_post_op = post_op_t{post_op_type_t::swish};
    auto binary_mul_po_2 = post_op_t{post_op_type_t::binary_mul};

    //define batch_matmul context
    auto batch_matmul_context = matmul_context_t()
                                .set_param("weights", weights)
                                .set_param("bias", bias)
                                .set_post_op(binary_mul_po)
                                .set_post_op(silu_post_op)
                                .set_post_op(binary_mul_po_2)
                                .create();

    if (! batch_matmul_context.check()) {
      testlog_error("batch_matmul context creation failed");
      return NOT_OK;
    }

    //define batch_matmul operator
    auto batch_matmul_operator = matmul_operator_t()
                                 .set_name("batch_matmul_f32_operator")
                                 .set_context(batch_matmul_context)
                                 .create();

    if (! batch_matmul_operator.check()) {
      testlog_error(" operator ", batch_matmul_operator.get_name(),
                    " creation failed.");
      return NOT_OK;
    }

    auto input_tensor = tensor_factory.uniform_tensor({BATCH_SIZE, BATCH_MATMUL_M, BATCH_MATMUL_K},
                        data_type_t::f32,
                        1.0, "matmul_input");

    auto mul_tensor   = tensor_factory.uniform_tensor({BATCH_MATMUL_N},
                        data_type_t::f32,
                        2.0, "binary_mul_0");

    auto mul_tensor_2 = tensor_factory.uniform_tensor({BATCH_MATMUL_M, BATCH_MATMUL_N},
                        data_type_t::bf16,
                        3.0, "binary_mul_1");

    auto output_tensor = tensor_factory.zero_tensor({BATCH_SIZE, BATCH_MATMUL_M, BATCH_MATMUL_N},
                         data_type_t::f32,
                         "matmul_output");

    status = batch_matmul_operator
             .set_input("matmul_input", input_tensor)
             .set_input(batch_matmul_context.get_post_op(0).binary_mul_params.tensor_name,
                        mul_tensor)
             .set_input(batch_matmul_context.get_post_op(2).binary_mul_params.tensor_name,
                        mul_tensor_2)
             .set_output("matmul_output", output_tensor)
             .execute();

    if (status == status_t::success) {
      testlog_info("operator ", batch_matmul_operator.get_name(),
                   " execution successful.");
    }
    else {
      testlog_info("operator ", batch_matmul_operator.get_name(),
                   " execution failed.");
      return NOT_OK;
    }

  }
  catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

int batch_matmul_silu_mul_bf16_kernel_example() {
  testlog_info("**batch_matmul + silu + binary_mul operator bf16 kernel example.");

  try {
    status_t status;
    tensor_factory_t tensor_factory;
    auto weights = tensor_factory.uniform_tensor({BATCH_SIZE, BATCH_MATMUL_K, BATCH_MATMUL_N},
                   data_type_t::bf16,
                   1.0, "weights");

    auto bias    = tensor_factory.uniform_tensor({BATCH_MATMUL_N},
                   data_type_t::f32,
                   3.0, "bias");

    auto silu_post_op = post_op_t{post_op_type_t::swish};
    auto binary_mul_po   = post_op_t{post_op_type_t::binary_mul};

    //define batch_matmul context
    auto batch_matmul_context = matmul_context_t()
                                .set_param("weights", weights)
                                .set_param("bias", bias)
                                .set_post_op(silu_post_op)
                                .set_post_op(binary_mul_po)
                                .create();

    if (! batch_matmul_context.check()) {
      testlog_error("batch_matmul context creation failed");
      return NOT_OK;
    }

    //define batch_matmul operator
    auto batch_matmul_operator = matmul_operator_t()
                                 .set_name("batch_matmul_bf16_operator")
                                 .set_context(batch_matmul_context)
                                 .create();

    if (! batch_matmul_operator.check()) {
      testlog_error(" operator ", batch_matmul_operator.get_name(),
                    " creation failed.");
      return NOT_OK;
    }

    auto input_tensor = tensor_factory.uniform_tensor({BATCH_SIZE, BATCH_MATMUL_M, BATCH_MATMUL_K},
                        data_type_t::bf16,
                        1.0, "matmul_input");

    auto mul_tensor = tensor_factory.uniform_tensor({BATCH_MATMUL_M, BATCH_MATMUL_N},
                      data_type_t::bf16,
                      2.0, "binary_mul_0");

    auto output_tensor = tensor_factory.zero_tensor({BATCH_SIZE, BATCH_MATMUL_M, BATCH_MATMUL_N},
                         data_type_t::f32);
    output_tensor.set_name("matmul_output");

    status = batch_matmul_operator
             .set_input("matmul_input", input_tensor)
             .set_input(batch_matmul_context.get_post_op(1).binary_mul_params.tensor_name,
                        mul_tensor)
             .set_output("matmul_output", output_tensor)
             .execute();

    if (status == status_t::success) {
      testlog_info("operator ", batch_matmul_operator.get_name(),
                   " execution successful.");
    }
    else {
      testlog_info("operator ", batch_matmul_operator.get_name(),
                   " execution failed.");
      return NOT_OK;
    }

  }
  catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return NOT_OK;
}

int batch_matmul_relu_forced_ref_kernel_example() {
  testlog_info("Reference batch_matmul kernel example");

  try {
    status_t status;
    tensor_factory_t tensor_factory;
    auto weights = tensor_factory.uniform_tensor({BATCH_SIZE, BATCH_MATMUL_K, BATCH_MATMUL_N},
                   data_type_t::f32,
                   1.0, "weights");

    auto bias    = tensor_factory.uniform_tensor({BATCH_MATMUL_N},
                   data_type_t::f32,
                   3.0, "bias");

    auto relu_post_op = post_op_t{post_op_type_t::relu};

    //define batch_matmul context
    auto batch_matmul_context = matmul_context_t()
                                .set_param("weights", weights)
                                .set_param("bias", bias)
                                .set_post_op(relu_post_op)
                                .create();

    if (! batch_matmul_context.check()) {
      testlog_error("batch_matmul context creation failed");
      return NOT_OK;
    }

    //define batch_matmul operator
    auto batch_matmul_operator = matmul_operator_t()
                                 .set_name("batch_matmul_forced_ref_operator")
                                 .set_context(batch_matmul_context)
                                 .create();

    if (! batch_matmul_operator.check()) {
      testlog_error("operator ", batch_matmul_operator.get_name(),
                    " creation failed.");
      return NOT_OK;
    }

    auto input_tensor = tensor_factory.uniform_tensor({BATCH_SIZE, BATCH_MATMUL_M, BATCH_MATMUL_K},
                        data_type_t::f32,
                        1.0, "matmul_input");

    auto output_tensor = tensor_factory.zero_tensor({BATCH_SIZE, BATCH_MATMUL_M, BATCH_MATMUL_N},
                         data_type_t::f32,
                         "matmul_output");

    status = batch_matmul_operator
             .set_input("matmul_input", input_tensor)
             .set_output("matmul_output", output_tensor)
             .set_forced_kernel("reference")
             .execute();

    if (status == status_t::success) {
      testlog_info("operator ", batch_matmul_operator.get_name(),
                   " execution successful.");
    }
    else {
      testlog_info("operator ", batch_matmul_operator.get_name(),
                   " execution failed.");
      return NOT_OK;
    }

  }
  catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return NOT_OK;
}

int batchmatmul_broadcast_example() {
  testlog_info("batchmatmul kernel broadcast example");
  try {
    status_t status;
    tensor_factory_t tensor_factory;
    auto weights = tensor_factory.broadcast_uniform_tensor({BATCH_SIZE, BATCH_MATMUL_K, BATCH_MATMUL_N}, {0,BATCH_MATMUL_N, 1},
                   data_type_t::f32, 1.0, "broadcasted_weight");

    auto bias    = tensor_factory.uniform_tensor({BATCH_MATMUL_N},
                   data_type_t::f32,
                   3.0, "bias");

    auto relu_post_op = post_op_t{post_op_type_t::relu};

    //define batch_matmul context
    auto batch_matmul_context = matmul_context_t()
                                .set_param("weights", weights)
                                .set_param("bias", bias)
                                .set_post_op(relu_post_op)
                                .create();

    if (! batch_matmul_context.check()) {
      testlog_error("batch matmul context creation failed");
      return NOT_OK;
    }

    //define matmul operator
    auto batch_matmul_operator = matmul_operator_t()
                                 .set_name("batch_matmul_broadcast_operator")
                                 .set_context(batch_matmul_context)
                                 .create();

    if (! batch_matmul_operator.check()) {
      testlog_error("operator ", batch_matmul_operator.get_name(),
                    " creation failed.");
      return NOT_OK;
    }

    auto input_tensor = tensor_factory.uniform_tensor({BATCH_SIZE, BATCH_MATMUL_M, BATCH_MATMUL_K},
                        data_type_t::f32,
                        1.0, "matmul_input");

    auto output_tensor = tensor_factory.zero_tensor({BATCH_SIZE, BATCH_MATMUL_M, BATCH_MATMUL_N},
                         data_type_t::f32,
                         "matmul_output");

    status = batch_matmul_operator
             .set_input("matmul_input", input_tensor)
             .set_output("matmul_output", output_tensor)
             .execute();
    if (status == status_t::success) {
      testlog_info("operator ", batch_matmul_operator.get_name(),
                   " execution successful.");
    }
    else {
      testlog_info("operator ", batch_matmul_operator.get_name(),
                   " execution failed.");
      return NOT_OK;
    }

  }
  catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return NOT_OK;
}


} //examples
} //zendnnl
