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

#include "gtest_utils.hpp"

MatmulType::MatmulType() {
  matmul_m = MATMUL_SIZE_START + rand() % MATMUL_SIZE_END;
  matmul_k = MATMUL_SIZE_START + rand() % MATMUL_SIZE_END;
  matmul_n = MATMUL_SIZE_START + rand() % MATMUL_SIZE_END;
  transA   = rand() % 2;
  transB   = rand() % 2;
  po_index = rand() % (po_size + 1);
}

bool is_binary_postop(const std::string post_op) {
  return post_op == "binary_add" || post_op == "binary_mul";
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
    float range_,
    bool trans) {
  auto udtensor = tensor_t()
                  .set_name("uniform distributed tensor")
                  .set_size(size_)
                  .set_data_type(dtype_)
                  .set_storage();
  if (trans) {
    udtensor.set_order("ba");
  }
  udtensor.create();

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

tensor_t tensor_factory_t::uniform_dist_strided_tensor(const
    std::vector<index_type> size_, const std::vector<index_type> stride_,
    data_type dtype_, float range_, bool trans) {
  auto udstensor = tensor_t()
                   .set_name("uniform distributed strided tensor")
                   .set_size(size_)
                   .set_data_type(dtype_)
                   .set_stride_size(stride_)
                   .set_storage()
                   .create();

  if (! udstensor.check()) {
    log_warning("tensor creation of ", udstensor.get_name(), " failed.");
  }
  else {
    std::mt19937 gen(100);
    std::uniform_real_distribution<float> dist(-1.0 * range_, 1.0 * range_);

    auto  buf_nelem   = stride_[0];
    for (size_t i = 1; i < stride_.size(); i++) {
      buf_nelem *= stride_[i];
    }
    void *buf_vptr = udstensor.get_raw_handle_unsafe();

    if (dtype_ == data_type::f32) {
      float *buf_ptr = static_cast<float *>(buf_vptr);
      std::generate(buf_ptr, buf_ptr + buf_nelem, [&] {return dist(gen);});
    }
    else if (dtype_ == data_type::bf16) {
      bfloat16_t *buf_ptr = static_cast<bfloat16_t *>(buf_vptr);
      std::generate(buf_ptr, buf_ptr + buf_nelem, [&] {return bfloat16_t(dist(gen));});
    }
    else if (dtype_ == data_type::s8) {
      int8_t *buf_ptr = static_cast<int8_t *>(buf_vptr);
      std::generate(buf_ptr, buf_ptr + buf_nelem, [&] {return int8_t(dist(gen));});
    }
    else {
      log_warning("tensor ", udstensor.get_name(), " unsupported data type.");
    }
  }
  return udstensor;
}

tensor_t tensor_factory_t::blocked_tensor(const std::vector<index_type> size_,
    data_type dtype_,
    StorageParam param) {

  auto btensor = tensor_t()
                 .set_name("blocked tensor")
                 .set_size(size_)
                 .set_data_type(dtype_)
                 .set_layout(tensor_layout_t::blocked);

  if (std::holds_alternative<std::pair<size_t, void *>>(param)) {
    auto [reorder_size, reorder_buff] = std::get<std::pair<size_t, void *>>(param);
    btensor.set_storage(reorder_buff, reorder_size);
  }
  else if (std::holds_alternative<tensor_t>(param)) {
    tensor_t input_tensor = std::get<tensor_t>(param);
    btensor.set_storage(input_tensor);
  }

  btensor.create();

  if (! btensor.check()) {
    log_warning("tensor creation of ", btensor.get_name(), " failed.");
  }

  return btensor;
}

status_t matmul_kernel_test(tensor_t &input_tensor, tensor_t &weights,
                            tensor_t &bias, tensor_t &output_tensor,
                            uint32_t index, tensor_t &binary_tensor) {
  try {
    // default postop relu
    post_op_t post_op = post_op_t{po_arr[0].second};
    // postop update according to the index
    if (index != po_size && index != 0) post_op = post_op_t{po_arr[index].second};
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
      log_error("operator ", matmul_operator.get_name(), " creation failed.");
      return status_t::failure;
    }

    input_tensor.set_name("matmul_input");
    output_tensor.set_name("matmul_output");
    // Set binary tensor for binary postops
    if (index < po_size) {
      if (po_arr[index].second == post_op_type_t::binary_add) {
        matmul_operator.set_input(post_op.binary_add_params.tensor_name, binary_tensor);
      }
      else if (po_arr[index].second == post_op_type_t::binary_mul) {
        matmul_operator.set_input(post_op.binary_mul_params.tensor_name, binary_tensor);
      }
    }
    status_t status = matmul_operator
                      .set_input("matmul_input", input_tensor)
                      .set_output("matmul_output", output_tensor)
                      .execute();

    if (status != status_t::success) {
      log_info("operator ", matmul_operator.get_name(), " execution failed.");
      return status_t::failure;
    }
  }
  catch (const exception_t &ex) {
    log_verbose(ex.what());
    return status_t::failure;
  }
  return status_t::success;
}

status_t matmul_forced_ref_kernel_test(tensor_t &input_tensor,
                                       tensor_t &weights,
                                       tensor_t &bias, tensor_t &output_tensor,
                                       uint32_t index, tensor_t &binary_tensor) {
  try {
    // Default postop relu
    post_op_t post_op = post_op_t{po_arr[0].second};
    // postop update according to the index
    if (index != po_size && index != 0) post_op = post_op_t{po_arr[index].second};
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
      log_error("operator ", matmul_operator.get_name(), " creation failed.");
      return status_t::failure;
    }
    input_tensor.set_name("matmul_input");
    output_tensor.set_name("matmul_output");

    if (index < po_size) {
      if (po_arr[index].second == post_op_type_t::binary_add) {
        matmul_operator.set_input(post_op.binary_add_params.tensor_name, binary_tensor);
      }
      else if (po_arr[index].second == post_op_type_t::binary_mul) {
        // Set binary tensor for binary postops
        matmul_operator.set_input(post_op.binary_mul_params.tensor_name, binary_tensor);
      }
    }
    status_t status = matmul_operator.set_input("matmul_input", input_tensor)
                      .set_output("matmul_output", output_tensor)
                      .set_forced_kernel("reference")
                      .execute();

    if (status != status_t::success) {
      log_info("operator ", matmul_operator.get_name(), " execution failed.");
      return status_t::failure;
    }
  }
  catch (const exception_t &ex) {
    log_verbose(ex.what());
    return status_t::failure;
  }
  return status_t::success;
}

void compare_tensor_2D(tensor_t &output_tensor, tensor_t &output_tensor_ref,
                       uint64_t m,
                       uint64_t n, const float tol, bool &is_comparison_successful) {
  #pragma omp parallel for collapse(2)
  for (uint64_t i=0; i<m; ++i) {
    for (uint64_t j=0; j<n; ++j) {
      if (is_comparison_successful) {
        float acutal_val = output_tensor.at({i,j});
        float ref_val = output_tensor_ref.at({i,j});
        if (abs(ref_val - acutal_val) >= tol) {
          log_verbose("actual(",i,",",j,"): ",acutal_val," , ref(",i,",",j,"): ",ref_val);
          is_comparison_successful = false;
        }
      }
    }
  }
  return;
}

std::pair<tensor_t, status_t> reorder_kernel_test(tensor_t &input_tensor,
    bool inplace_reorder) {
  try {
    tensor_factory_t tensor_factory;
    status_t status;

    input_tensor.set_name("reorder_input");

    // Reorder context creation with backend aocl.
    auto reorder_context = reorder_context_t()
                           .set_algo_format("aocl")
                           .create();

    // Reorder operator creation with name, context and input.
    auto reorder_operator = reorder_operator_t()
                            .set_name("reorder_operator")
                            .set_context(reorder_context)
                            .create()
                            .set_input("reorder_input", input_tensor);

    if (! reorder_operator.check()) {
      log_error("operator ", reorder_operator.get_name(), " creation failed.");
      return std::make_pair(tensor_t(), status_t::failure);
    }

    // Compute the reorder size
    size_t reorder_size         = reorder_operator.get_reorder_size();
    // Extract the input buffer size
    size_t input_buffer_size    = input_tensor.get_buffer_sz_bytes();
    data_type_t dtype           = input_tensor.get_data_type();

    uint64_t rows               = input_tensor.get_size(0);
    uint64_t cols               = input_tensor.get_size(1);
    tensor_t output_tensor;

    // InPlace reorder
    if (inplace_reorder) {
      // InPlace reorder works when reorder size is equal to input buffer size.
      if (reorder_size != input_buffer_size) {
        log_info("Inplace reorder is not possible for given input");
        return std::make_pair(input_tensor, status_t::failure);
      }
      else {
        // Assign input_tensor to buffer_params as a tensor_t variant
        StorageParam buffer_params = input_tensor;

        // Output Tensor creation with seperate view for input tensor
        output_tensor = tensor_factory.blocked_tensor({rows, cols},
                        dtype,
                        buffer_params);
        output_tensor.set_name("reorder_output");
      }
    }
    else {
      // create a buffer with reorderd size
      float *reorder_weights = (float *) aligned_alloc(64, reorder_size);

      // Create a Pair of storage params [reorder size and reorder weights] and
      // use it in tensor creation
      StorageParam buffer_params = std::make_pair(reorder_size, reorder_weights);

      // Create output tensor with blocked layout.
      output_tensor = tensor_factory.blocked_tensor({rows, cols},
                      dtype,
                      buffer_params);
      output_tensor.set_name("reorder_output");
    }

    // Reorder operator execution.
    status = reorder_operator
             .set_output("reorder_output", output_tensor)
             .execute();

    if (status != status_t::success) {
      log_info("operator ", reorder_operator.get_name(), " execution failed.");
    }
    else {
      log_info("operator ", reorder_operator.get_name(), " execution successful.");
    }

    return std::make_pair(output_tensor, status_t::success);
  }
  catch (const exception_t &ex) {
    log_verbose(ex.what());
    return std::make_pair(tensor_t(), status_t::failure);
  }
}
