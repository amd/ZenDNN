/********************************************************************************
# * Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "gtest_utils.hpp"

MatmulType::MatmulType() {
  matmul_m = MATMUL_SIZE_START + rand() % MATMUL_SIZE_END;
  matmul_k = MATMUL_SIZE_START + rand() % MATMUL_SIZE_END;
  matmul_n = MATMUL_SIZE_START + rand() % MATMUL_SIZE_END;
  po_index = rand() % (po_size + 1);
}

tensor_t tensor_factory_t::zero_tensor(const std::vector<index_type> size_,
                                       data_type dtype_) {

  auto ztensor = tensor_t()
                 .set_name("zero tensor")
                 .set_size(size_)
                 .set_data_type(dtype_)
                 .set_storage()
                 .create();

  if (! ztensor.check()) {
    log_warning("tensor creation of ", ztensor.get_name(), " failed.");
  }
  else {
    auto  buf_size = ztensor.get_buffer_sz_bytes();
    void *buf_ptr  = ztensor.get_raw_handle_unsafe();
    std::memset(buf_ptr, 0, buf_size);
  }
  return ztensor;
}

tensor_t tensor_factory_t::uniform_dist_tensor(const std::vector<index_type>
    size_,
    data_type dtype_,
    float range_) {
  auto udtensor = tensor_t()
                  .set_name("uniform distributed tensor")
                  .set_size(size_)
                  .set_data_type(dtype_)
                  .set_storage()
                  .create();

  if (! udtensor.check()) {
    log_warning("tensor creation of ", udtensor.get_name(), " failed.");
  }
  else {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist2(-1.0*range_, 1.0*range_);

    auto  buf_nelem  = udtensor.get_nelem();
    void *buf_vptr   = udtensor.get_raw_handle_unsafe();

    if (dtype_ == data_type::f32) {
      float *buf_ptr = static_cast<float *>(buf_vptr);
      std::generate(buf_ptr, buf_ptr+buf_nelem, [&] {return dist2(gen);});
    }
    else if (dtype_ == data_type::bf16) {
      bfloat16_t *buf_ptr = static_cast<bfloat16_t *>(buf_vptr);
      std::generate(buf_ptr, buf_ptr+buf_nelem, [&] {return bfloat16_t(dist2(gen));});
    }
    else {
      log_warning("tensor ", udtensor.get_name(), " unsupported data type.");
    }
  }
  return udtensor;
}

void matmul_kernel_test(tensor_t &input_tensor, tensor_t &weights,
                        tensor_t &bias, tensor_t &output_tensor,uint32_t index) {
  try {
    // default postop relu
    post_op_t post_op = post_op_t{po_arr[0]};
    // postop update according to the index
    if (index != po_size && index != 0) post_op = post_op_t{po_arr[index]};
    weights.set_name("weights");
    bias.set_name("bias");

    //define matmul context
    auto matmul_context = matmul_context_t()
                          .set_param("weights", weights)
                          .set_param("bias", bias);
    if (index != po_size) {
      matmul_context = matmul_context.set_post_op(post_op).create();
    }
    else {
      matmul_context = matmul_context.create();//No Postop case
    }

    //define matmul operator
    auto matmul_operator = matmul_operator_t()
                           .set_name("matmul_operator")
                           .set_context(matmul_context)
                           .create();

    if (! matmul_operator.check()) {
      log_error(" operator ", matmul_operator.get_name(), " creation failed.");
      exit(0);
    }

    input_tensor.set_name("matmul_input");
    output_tensor.set_name("matmul_output");
    status_t status = matmul_operator
                      .set_input("matmul_input", input_tensor)
                      .set_output("matmul_output", output_tensor)
                      .execute();

    if (status != status_t::success) {
      log_info("operator ", matmul_operator.get_name(), " execution failed.");
      exit(0);
    }
  }
  catch (const exception_t &ex) {
    log_verbose(ex.what());
  }
}

void matmul_forced_ref_kernel_test(tensor_t &input_tensor, tensor_t &weights,
                                   tensor_t &bias, tensor_t &output_tensor, uint32_t index) {
  try {
    // Default postop relu
    post_op_t post_op = post_op_t{po_arr[0]};
    // postop update according to the index
    if (index != po_size && index != 0) post_op = post_op_t{po_arr[index]};
    weights.set_name("weights");
    bias.set_name("bias");

    //define matmul context
    auto matmul_context = matmul_context_t()
                          .set_param("weights", weights)
                          .set_param("bias", bias);
    if (index != po_size) {
      matmul_context = matmul_context.set_post_op(post_op).create();
    }
    else {
      matmul_context = matmul_context.create(); //No postop case
    }

    //define matmul operator
    auto matmul_operator = matmul_operator_t()
                           .set_name("matmul_forced_ref_operator")
                           .set_context(matmul_context)
                           .create();

    if (! matmul_operator.check()) {
      log_error(" operator ", matmul_operator.get_name(), " creation failed.");
      exit(0);
    }
    input_tensor.set_name("matmul_input");
    output_tensor.set_name("matmul_output");

    status_t status = matmul_operator.set_input("matmul_input", input_tensor)
                      .set_output("matmul_output", output_tensor)
                      .set_forced_kernel("reference")
                      .execute();

    if (status != status_t::success) {
      log_info("operator ", matmul_operator.get_name(), " execution failed.");
      exit(0);
    }
  }
  catch (const exception_t &ex) {
    log_verbose(ex.what());
  }
  return;
}

void compare_tensor_2D(tensor_t &output_tensor, tensor_t &output_tensor_ref,
                       uint64_t m,
                       uint64_t n, const float tol, bool &flag) {
  #pragma omp parallel for collapse(2)
  for (uint64_t i=0; i<m; ++i) {
    for (uint64_t j=0; j<n; ++j) {
      if (!flag) {
        float acutal_val = output_tensor.at({i,j});
        float ref_val = output_tensor_ref.at({i,j});
        if (abs(ref_val - acutal_val) >= tol) {
          log_verbose("actual(",i,",",j,"): ",acutal_val," , ref(",i,",",j,"): ",ref_val);
          flag = true;
        }
      }
    }
  }
  return;
}
