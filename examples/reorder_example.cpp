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
#include "reorder_example.hpp"

namespace zendnnl {
namespace examples {

using namespace zendnnl::interface;

int reorder_f32_kernel_example() {
  testlog_info("Reorder operator f32 kernel example");
  try {
    tensor_factory_t tensor_factory;
    status_t status;

    // Create input tensor with contigious layout.
    auto input_tensor = tensor_factory.uniform_tensor({ROWS, COLS},
                        data_type_t::f32,
                        1.0, "reorder_input");

    // Reorder context creation with backend aocl.
    auto reorder_context = reorder_context_t()
                           .set_algo_format("aocl")
                           .create();

    // Reorder operator creation with name, context and input.
    auto reorder_operator = reorder_operator_t()
                            .set_name("reorder_f32_operator")
                            .set_context(reorder_context)
                            .create()
                            .set_input("reorder_input", input_tensor);

    // Compute the reorder size and create a buffer with reorderd size
    size_t reorder_size = reorder_operator.get_reorder_size();
    void *reorder_weights = aligned_alloc(64, reorder_size);

    // Create output tensor with blocked layout.
    auto output_tensor = tensor_factory.blocked_tensor({ROWS, COLS},
                         data_type_t::f32,
                         reorder_size,
                         reorder_weights,
                         "reorder_output");

    // Check if reorder operation creation is successful.
    if (! reorder_operator.check()) {
      testlog_error("operator ", reorder_operator.get_name(), " creation failed");
      return NOT_OK;
    }

    // Reorder operator execution.
    status = reorder_operator
             .set_output("reorder_output", output_tensor)
             .execute();

    if (status == status_t::success) {
      testlog_info("operator ", reorder_operator.get_name(),
                   " execution successful.");
    }
    else {
      testlog_error("operator ", reorder_operator.get_name(), " execution failed.");
    }

    // Free reordered size buffer.
    free(reorder_weights);
  }
  catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

int reorder_s8_kernel_example() {
  testlog_info("Reorder operator s8 kernel example");
  try {
    tensor_factory_t tensor_factory;
    status_t status;

    // Create input tensor with contigious layout.
    auto input_tensor = tensor_factory.uniform_tensor({ROWS, COLS},
                        data_type_t::f32,
                        1.0, "reorder_input");

    // Reorder context creation with backend aocl.
    auto reorder_context = reorder_context_t()
                           .set_algo_format("aocl")
                           .set_source_dtype(data_type_t::u8)
                           .create();

    // Reorder operator creation with name, context and input.
    auto reorder_operator = reorder_operator_t()
                            .set_name("reorder_s8_operator")
                            .set_context(reorder_context)
                            .create()
                            .set_input("reorder_input", input_tensor);

    // Compute the reorder size and create a buffer with reorderd size
    size_t reorder_size = reorder_operator.get_reorder_size();
    void *reorder_weights = aligned_alloc(64, reorder_size);

    // Create output tensor with blocked layout.
    auto output_tensor = tensor_factory.blocked_tensor({ROWS, COLS},
                         data_type_t::s8,
                         reorder_size,
                         reorder_weights,
                         "reorder_output");

    // Check if reorder operation creation is successful.
    if (! reorder_operator.check()) {
      testlog_error("operator ", reorder_operator.get_name(), " creation failed");
      return NOT_OK;
    }

    // Reorder operator execution.
    status = reorder_operator
             .set_output("reorder_output", output_tensor)
             .execute();

    if (status == status_t::success) {
      testlog_info("operator ", reorder_operator.get_name()," execution successful.");
    }
    else {
      testlog_error("operator ", reorder_operator.get_name()," execution failed.");
    }

    // Free reordered size buffer.
    free(reorder_weights);
  }
  catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

int reorder_matmul_relu_f32_kernel_example() {
  testlog_info("Matmul with reorder weights+relu operator f32 kernel example");

  try {
    tensor_factory_t tensor_factory;
    status_t status;

    // Create weight tensor with contigious layout.
    auto weight_tensor = tensor_factory.uniform_dist_tensor({MATMUL_K, MATMUL_N},
                         data_type_t::f32,
                         5.0, "reorder_input");

    // Reorder context creation with backend aocl.
    auto reorder_context = reorder_context_t()
                           .set_algo_format("aocl")
                           .create();

    // Reorder operator creation with name, context and input.
    auto reorder_operator = reorder_operator_t()
                            .set_name("reorder_f32_operator")
                            .set_context(reorder_context)
                            .create()
                            .set_input("reorder_input", weight_tensor);

    // Compute the reorder size and create a buffer with reorderd size
    size_t reorder_size = reorder_operator.get_reorder_size();
    void *reorder_weights = aligned_alloc(64, reorder_size);

    // Create output tensor with blocked layout to store reordered buffer.
    auto reorder_weights_tensor = tensor_factory.blocked_tensor({MATMUL_K, MATMUL_N},
                                  data_type_t::f32,
                                  reorder_size,
                                  reorder_weights,
                                  "reorder_output");

    // Check if reorder operation creation is successful.
    if (! reorder_operator.check()) {
      testlog_error("operator ", reorder_operator.get_name(), " creation failed");
      return NOT_OK;
    }

    // Reorder operator execution.
    status = reorder_operator
             .set_output("reorder_output", reorder_weights_tensor)
             .execute();

    if (status == status_t::success) {
      testlog_info("operator ", reorder_operator.get_name(),
                   " execution successful.");
    }
    else {
      testlog_error("operator ", reorder_operator.get_name(), " execution failed.");
    }

    reorder_weights_tensor.set_name("weights");

    // Create Bias tensor with contigious layout.
    auto bias    = tensor_factory.uniform_tensor({MATMUL_N},
                   data_type_t::f32,
                   10.0, "bias");

    auto relu_post_op = post_op_t{post_op_type_t::relu};

    // Matmul context creation with weights, Bias and Postop: relu
    auto matmul_context = matmul_context_t()
                          .set_param("weights", reorder_weights_tensor)
                          .set_param("bias", bias)
                          .set_post_op(relu_post_op)
                          .create();

    // Matmul operator creation with name and context
    auto matmul_operator = matmul_operator_t()
                           .set_name("matmul_f32_operator")
                           .set_context(matmul_context)
                           .create();

    // Check if matmul operation creation is successful.
    if (! matmul_operator.check()) {
      testlog_error(" operator ", matmul_operator.get_name(), " creation failed.");
      return NOT_OK;
    }

    // Create input tensor with contigious layout.
    auto input_tensor = tensor_factory.uniform_tensor({MATMUL_M, MATMUL_K},
                        data_type_t::f32,
                        1.0, "matmul_input");

    // Create output tensor with contigious layout.
    auto output_tensor = tensor_factory.zero_tensor({MATMUL_M, MATMUL_N},
                         data_type_t::f32,
                         "matmul_output");

    // Matmul operator execution
    status = matmul_operator
             .set_input("matmul_input", input_tensor)
             .set_output("matmul_output", output_tensor)
             .execute();

    if (status == status_t::success) {
      testlog_info("operator ", matmul_operator.get_name(), " execution successful.");
      testlog_verbose("output[", MATMUL_M/2, ",", MATMUL_N/2,"] = ",
                      output_tensor.at({MATMUL_M/2, MATMUL_N/2}));
    }
    else {
      testlog_error("operator ", matmul_operator.get_name(), " execution failed.");
      return NOT_OK;
    }

    // Free reorderd size buffer
    free(reorder_weights);
  }
  catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

int reorder_inplace_bf16_example() {
  testlog_info("Inplace reorder operator bf16 kernel example");
  try {
    tensor_factory_t tensor_factory;
    status_t status;

    // Create input tensor with contigious layout.
    auto input_tensor = tensor_factory.uniform_tensor({ROWS, COLS},
                        data_type_t::bf16,
                        1.0);
    input_tensor.set_name("reorder_input");

    // Reorder context creation with backend aocl.
    auto reorder_context = reorder_context_t()
                           .set_algo_format("aocl")
                           .create();

    // Reorder operator creation with name, context and input.
    auto reorder_operator = reorder_operator_t()
                            .set_name("inplace_reorder_bf16_operator")
                            .set_context(reorder_context)
                            .create()
                            .set_input("reorder_input", input_tensor);

    [[maybe_unused]] size_t reorder_size = reorder_operator.get_reorder_size();

    // Check if inplace reorder operation creation is successful.
    if (! reorder_operator.check()) {
      testlog_error("operator ", reorder_operator.get_name(), " creation failed");
      return NOT_OK;
    }

    // Inplace Reorder operator execution.
    // input_tensor is passed as output for reorder operation.
    status = reorder_operator
             .set_output("reorder_output", input_tensor)
             .execute();

    if (status == status_t::success) {
      testlog_info("operator ", reorder_operator.get_name(),
                   " execution successful.");
      input_tensor.set_layout(tensor_layout_t::blocked);
    }
    else {
      testlog_error("operator ", reorder_operator.get_name(), " execution failed.");
    }

  }
  catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

int reorder_inplace_matmul_relu_bf16_kernel_example() {
  testlog_info("Matmul reorder weights+relu operator bf16 kernel example");

  try {
    tensor_factory_t tensor_factory;
    status_t status;

    // Create weight tensor with contigious layout.
    auto weight_tensor = tensor_factory.uniform_dist_tensor({MATMUL_K, MATMUL_N},
                         data_type_t::bf16,
                         5.0, "reorder_input");

    // Reorder context creation with backend aocl.
    auto reorder_context = reorder_context_t()
                           .set_algo_format("aocl")
                           .create();

    // Reorder operator creation with name, context and input.
    auto reorder_operator = reorder_operator_t()
                            .set_name("reorder_inplace_bf16_operator")
                            .set_context(reorder_context)
                            .create()
                            .set_input("reorder_input", weight_tensor);

    // Compute and returns the reorder size
    size_t reorder_size = reorder_operator.get_reorder_size();

    // Check if inplace reorder operation creation is successful.
    if (! reorder_operator.check()) {
      testlog_error("operator ", reorder_operator.get_name(), " creation failed");
      return NOT_OK;
    }

    // Returns the memory that weight tensor is pointing.
    auto reorder_buff = weight_tensor.get_raw_handle_unsafe();

    // New Tensor is created with Blocked layout and is used for Reorder
    auto blocked_tensor = tensor_factory.blocked_tensor({MATMUL_K, MATMUL_N},
                          data_type_t::bf16,
                          reorder_size,
                          reorder_buff);

    // Inplace Reorder operator execution.
    // Blocked_tensor that points to same memory is passed
    // as output for reorder operation.
    status = reorder_operator
             .set_output("reorder_output", blocked_tensor)
             .execute();

    bool reorder_status;
    if (status == status_t::success) {
      testlog_info("operator ", reorder_operator.get_name(),
                   " execution successful.");
      reorder_status = true;
    }
    else {
      testlog_error("operator ", reorder_operator.get_name(), " execution failed.");
      reorder_status = false;
    }

    // Create Bias tensor with contigious layout.
    auto bias    = tensor_factory.uniform_tensor({MATMUL_N},
                   data_type_t::bf16,
                   10.0, "bias");

    auto relu_post_op = post_op_t{post_op_type_t::relu};

    // Matmul context creation with weights, Bias and Postop: relu
    auto matmul_context = matmul_context_t();
    if (reorder_status) {
      matmul_context.set_param("weights", blocked_tensor);
    }
    else {
      matmul_context.set_param("weights", weight_tensor);
    }

    matmul_context.set_param("bias", bias)
                  .set_post_op(relu_post_op)
                  .create();

    // Matmul operator creation with name and context
    auto matmul_operator = matmul_operator_t()
                           .set_name("matmul_bf16_operator")
                           .set_context(matmul_context)
                           .create();

    // Check if matmul operation creation is successful.
    if (! matmul_operator.check()) {
      testlog_error(" operator ", matmul_operator.get_name(), " creation failed.");
      return NOT_OK;
    }

    // Create input tensor with contigious layout.
    auto input_tensor = tensor_factory.uniform_tensor({MATMUL_M, MATMUL_K},
                        data_type_t::bf16,
                        1.0, "matmul_input");

    // Create output tensor with contigious layout.
    auto output_tensor = tensor_factory.zero_tensor({MATMUL_M, MATMUL_N},
                         data_type_t::bf16,
                         "matmul_output");

    // Matmul operator execution
    status = matmul_operator
             .set_input("matmul_input", input_tensor)
             .set_output("matmul_output", output_tensor)
             .execute();

    if (status == status_t::success) {
      testlog_info("operator ", matmul_operator.get_name(), " execution successful.");
      testlog_verbose("output[", MATMUL_M/2, ",", MATMUL_N/2,"] = ",
                      output_tensor.at({MATMUL_M/2, MATMUL_N/2}));
    }
    else {
      testlog_error("operator ", matmul_operator.get_name(), " execution failed.");
      return NOT_OK;
    }

  }
  catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

} //examples
} //zendnnl
