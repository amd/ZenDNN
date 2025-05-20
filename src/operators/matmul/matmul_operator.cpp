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
#include "matmul_operator.hpp"
#include "matmul_kernel_list.hpp"

namespace zendnnl {
namespace ops {

status_t validate_buffer_post_op(std::vector<uint64_t> &output_size, std::vector<post_op_t> &po,
    std::map<std::string,tensor_t> &inputs) {
  /**
  * @todo Add validation support for per tensor and per channel
  * for binary post-ops
  */
  if (!po.empty()) {
    for (const auto& op : po) {
      if (op.type == post_op_type_t::binary_add) {
        auto add_tensor_obj = inputs.find(op.binary_add_params.tensor_name);
        if (add_tensor_obj == inputs.end()) {
          log_error("Invalid post-op: ",
                    op.binary_add_params.tensor_name, " buffer not passed.");
          return status_t::failure;
        }
        if (output_size.at(0) != add_tensor_obj->second.get_size(0) ||
            output_size.at(1) != add_tensor_obj->second.get_size(1)) {
          log_error(add_tensor_obj->second.get_name(),
                    " Invalid post-op: output size mismatch for binary_add post op.");
          return status_t::failure;
        }
      }
      else if (op.type == post_op_type_t::binary_mul) {
        auto mul_tensor_obj = inputs.find(op.binary_mul_params.tensor_name);
        if (mul_tensor_obj == inputs.end()) {
          log_error("Invalid post-op: ",
                    op.binary_mul_params.tensor_name, " buffer not passed.");
          return status_t::failure;
        }
        if (output_size.at(0) != mul_tensor_obj->second.get_size(0) ||
            output_size.at(1) != mul_tensor_obj->second.get_size(1)) {
          log_error(mul_tensor_obj->second.get_name(),
                    " Invalid post-op: output size mismatch for binary_mul post op.");
          return status_t::failure;
        }
      }
    }
  }
  return status_t::success;
}

status_t matmul_operator_t::validate() {
  LOG_DEBUG_INFO("<", get_name(), "> Validating matmul op parameters matmul_operator_t");
  if (parent_type::validate() != status_t::success)
    return status_t::failure;

  auto input        = get_input("matmul_input");
  auto output       = get_output("matmul_output");

  auto weights      = context.get_param("weights");
  auto weights_size = weights->get_size();

  if (!input || !output)
    return status_t::failure;

  auto input_size  = input->get_size();
  auto output_size = output->get_size();

  if ((input_size.size() != 2) || (output_size.size() != 2))
    return status_t::failure;

  if (input_size.at(0) != output_size.at(0))
    return status_t::failure;

  if (input_size.at(1) != weights_size.at(0))
    return status_t::failure;

  if (weights_size.at(1) != output_size.at(1))
    return status_t::failure;

  // validate post-ops
  auto post_ops = context.get_post_op();
  return validate_buffer_post_op(output_size, post_ops, inputs);
}

status_t matmul_operator_t::validate_forced_kernel() {

  if (forced_kernel.empty())
    return status_t::success;

  LOG_DEBUG_INFO("<", get_name(), "> Validating forced kernel matmul_operator_t");
  if (forced_kernel == "reference") {
    auto input        = get_input("matmul_input");
    auto output       = get_output("matmul_output");
    auto weights      = context.get_param("weights");

    auto in_dtype     = input->get_data_type();
    auto out_dtype    = output->get_data_type();
    auto wt_dtype     = weights->get_data_type();
    auto in_layout    = input->get_layout();
    auto out_layout   = output->get_layout();
    auto wt_layout    = weights->get_layout();


    if ((in_dtype  != data_type_t::f32) ||
        (out_dtype != data_type_t::f32) ||
        (wt_dtype  != data_type_t::f32) ||
        (in_layout != tensor_layout_t::contiguous) ||
        (out_layout != tensor_layout_t::contiguous) ||
        (wt_layout  != tensor_layout_t::contiguous)) {
      log_error("<", get_name(), "> forced reference kernel needs f32 contiguous tensors.");
      return status_t::failure;
    }
  }
  else {
    log_error("<", get_name(), "> ", forced_kernel, " kernel can not be forced.");
    return status_t::failure;
  }
  return status_t::success;
}

status_t matmul_operator_t::preprocess() {
  LOG_DEBUG_INFO("<", get_name(), "> Preprocessing matmul_operator_t");
  //get bias tensor
  auto optional_bias_tensor = context.get_param("bias");
  //get weight tensor for reorder
  auto weight_tensor = context.get_param("weights");

  if (context.aocl_utils_ptr->reorder_weights(weight_tensor)
      == status_t::failure){
    return status_t::failure;
  }
  //initialize aocl po
  if (context.aocl_utils_ptr->alloc_post_op(context.get_post_op(), optional_bias_tensor)
        == status_t::failure){
    return status_t::failure;
  }
  //set runtime post ops from inputs
  return context.aocl_utils_ptr->set_runtime_post_op_buffer(inputs);
}

status_t matmul_operator_t::kernel_factory() {
  LOG_DEBUG_INFO("<", get_name(), "> Executing kernel factory matmul_operator_t");
  auto weight_dtype   = context.get_param("weights")->get_data_type();
  auto input_dtype    = get_input("matmul_input")->get_data_type();
  auto output_dtype   = get_output("matmul_output")->get_data_type();

  // bool force_onednn = true;
  // if (force_onednn)
  //   kernel = get_matmul_onednn_kernel();
  // else if (weight_dtype == data_type_t::f32)
  //   kernel = get_matmul_f32_avx512_kernel();
  // else if (weight_dtype == data_type_t::bf16)
  //   kernel = get_matmul_bf16_avx512_kernel();
  // else
  //   return status_t::unimplemented;

  //get forced kernel if any
  if (forced_kernel.empty() || forced_kernel == "aocl") {
    /**TODO: move the preprocess to specific kernel */
    if (preprocess() != status_t::success) {
      return status_t::failure;
    }
    if ((weight_dtype == data_type_t::f32) &&
        (input_dtype  == data_type_t::f32) &&
        (output_dtype == data_type_t::f32))
      kernel = get_matmul_f32_avx512_kernel();
    else if ((weight_dtype == data_type_t::bf16) &&
             (input_dtype  == data_type_t::bf16) &&
             (output_dtype == data_type_t::f32 ||
             output_dtype == data_type_t::bf16))
      kernel = get_matmul_bf16_avx512_kernel();

    else {
      log_error("<", name, "> kernel unimplemented.");
      return status_t::unimplemented;
    }
  }
  else {
    if (forced_kernel == "reference")
      kernel = get_matmul_f32_ref_kernel();
    else if (forced_kernel == "onednn") {
      //kernel = get_linear_onednn_kernel();
      log_error("<", name, "> kernel unimplemented using onednn");
      return status_t::unimplemented;
    }
    else {
      log_error("<", name, "> kernel unimplemented using forced kernel ", forced_kernel);
      return status_t::unimplemented;
    }
  }

  kernel->create();
  if (! kernel->check())
    return kernel->get_last_status();

  return status_t::success;
}

} //namespace ops
} //namespace zendnnl
