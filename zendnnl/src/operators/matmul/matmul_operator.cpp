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
#include "matmul_operator.hpp"
#include "matmul_kernel_list.hpp"

namespace zendnnl {
namespace ops {

status_t validate_buffer_post_op(std::vector<uint64_t> &output_size,
                                 std::vector<post_op_t> &po,
                                 std::map<std::string,tensor_t> &inputs) {
  /**
  * @todo Add validation support for per tensor and per channel
  * for binary post-ops
  */
  if (!po.empty()) {
    for (const auto &op : po) {
      if (op.type == post_op_type_t::binary_add) {
        auto add_tensor_obj = inputs.find(op.binary_add_params.tensor_name);
        auto tensor_size = add_tensor_obj->second.get_size();
        if (add_tensor_obj == inputs.end()) {
          apilog_error("Invalid post-op: ",
                       op.binary_add_params.tensor_name, " buffer not passed.");
          return status_t::failure;
        }
        if (add_tensor_obj->second.get_order() == "ba") {
          apilog_error("Invalid post-op: ",
                       op.binary_add_params.tensor_name, " transposed buffer not supported.");
          return status_t::failure;
        }
        /** todo: support 1d add scale*/
        if (tensor_size.size() == 1 && tensor_size[0] == output_size.at(1) &&
            op.binary_add_params.scale == 1.0) {
          continue;
        }
        else if (tensor_size.size() == 2 && (tensor_size[0] == output_size.at(0) &&
                                             tensor_size[1] == output_size.at(1))) {
          continue;
        }
        else {
          apilog_error(add_tensor_obj->second.get_name(),
                       " Invalid post-op: size mismatch for binary_add post op.");
          return status_t::failure;
        }
      }
      else if (op.type == post_op_type_t::binary_mul) {
        auto mul_tensor_obj = inputs.find(op.binary_mul_params.tensor_name);
        auto tensor_size = mul_tensor_obj->second.get_size();
        if (mul_tensor_obj == inputs.end()) {
          apilog_error("Invalid post-op: ",
                       op.binary_mul_params.tensor_name, " buffer not passed.");
          return status_t::failure;
        }
        if (mul_tensor_obj->second.get_order() == "ba") {
          apilog_error("Invalid post-op: ",
                       op.binary_mul_params.tensor_name, " transposed buffer not supported.");
          return status_t::failure;
        }
        if (tensor_size.size() == 1 && tensor_size[0] == output_size.at(1) &&
            op.binary_mul_params.scale == 1.0) {
          continue;
        }
        else if (tensor_size.size() == 2 && (tensor_size[0] == output_size.at(0) &&
                                             tensor_size[1] == output_size.at(1))) {
          continue;
        }
        else {
          apilog_error(mul_tensor_obj->second.get_name(),
                       " Invalid post-op: size mismatch for binary_mul post op.");
          return status_t::failure;
        }
      }
    }
  }
  return status_t::success;
}

status_t matmul_operator_t::validate() {
  LOG_DEBUG_INFO("<", get_name(),
                 "> Validating matmul op parameters matmul_operator_t");
  if (parent_type::validate() != status_t::success) {
    return status_t::failure;
  }

  auto input        = get_input("matmul_input");
  auto output       = get_output("matmul_output");

  auto weights      = context.get_param("weights");
  auto weights_size = weights->get_size();

  if (!input || !output) {
    apilog_error("Invalid input or output tensor.");
    return status_t::failure;
  }

  auto input_size  = input->get_size();
  auto output_size = output->get_size();
  auto out_order   = output->get_order();

  if (out_order == "ba") {
    apilog_error("<", get_name(), "> kernel needs non-transposed output tensors.");
    return status_t::failure;
  }

  if ((input_size.size() != 2) || (output_size.size() != 2)) {
    apilog_error("Input or output size is not valid");
    return status_t::failure;
  }

  if (input_size.at(0) != output_size.at(0)) {
    apilog_error("Input and output size mismatch at dim - 0");
    return status_t::failure;
  }

  if (input_size.at(1) != weights_size.at(0)) {
    apilog_error("Dimension mismatch with input and weights");
    return status_t::failure;
  }

  if (weights_size.at(1) != output_size.at(1)) {
    apilog_error("Dimension mismatch with weights and output");
    return status_t::failure;
  }

  // validate post-ops
  auto post_ops = context.get_post_op();
  return validate_buffer_post_op(output_size, post_ops, inputs);
}

status_t matmul_operator_t::validate_forced_kernel() {

  if (forced_kernel.empty() || forced_kernel == "aocl_blis" ||
      forced_kernel == "aocl_blis_blocked") {
    return status_t::success;
  }

  LOG_DEBUG_INFO("<", get_name(), "> Validating forced kernel matmul_operator_t");
  if (forced_kernel == "reference") {
    auto input        = get_input("matmul_input");
    auto output       = get_output("matmul_output");
    auto weights      = context.get_param("weights");

    auto in_dtype     = input->get_data_type();
    auto out_dtype    = output->get_data_type();
    auto wt_dtype     = weights->get_data_type();
    auto out_order   = output->get_order();

    if ((in_dtype  != data_type_t::f32) ||
        (out_dtype != data_type_t::f32) ||
        (wt_dtype  != data_type_t::f32) ||
        (out_order == "ba")) {
      apilog_error("<", get_name(),
                   "> forced reference kernel needs f32 tensors and non-transposed dst.");
      return status_t::failure;
    }
  }
  else {
    apilog_error("<", get_name(), "> ", forced_kernel,
                 " kernel can not be forced.");
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

  if (forced_kernel == "aocl_blis_blocked") {
    if (context.aocl_blis_utils_ptr->reorder_weights(weight_tensor)
        == status_t::failure) {
      return status_t::failure;
    }
  }
  //initialize aocl po
  if (context.aocl_blis_utils_ptr->alloc_post_op(context.get_post_op(),
      optional_bias_tensor, inputs)
      == status_t::failure) {
    return status_t::failure;
  }
  //set runtime post ops from inputs
  return context.aocl_blis_utils_ptr->set_runtime_post_op_buffer(inputs,
         optional_bias_tensor ? true : false);
}

std::string matmul_operator_t::operator_info() {
  std::stringstream ss;
  auto input   = get_input("matmul_input").value();
  auto output  = get_output("matmul_output").value();

  auto weights       = context.get_param("weights").value();
  auto bias          = context.get_param("bias");
  auto post_op_count = context.get_post_op_count();

  ss << "matmul," << get_name() << "," << input.tensor_info()
     << "," << weights.tensor_info() << ",";

  if (bias) {
    ss <<bias.value().tensor_info()<<",";
  }

  ss <<output.tensor_info();

  if (post_op_count) {
    ss << ",post-op";

    for (uint32_t i = 0; i < post_op_count; ++i) {
      post_op_t zen_po = context.get_post_op(i);
      ss << ":" <<zen_po.post_op_info(zen_po);
    }
  }

  return ss.str();
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
  if (forced_kernel.empty() || forced_kernel == "aocl_blis_blocked" ||
      forced_kernel == "aocl_blis") {
    /**TODO: move the preprocess to specific kernel */
    if (preprocess() != status_t::success) {
      return status_t::failure;
    }
    if ((weight_dtype == data_type_t::f32) &&
        (input_dtype  == data_type_t::f32) &&
        (output_dtype == data_type_t::f32)) {
      kernel = get_matmul_f32_avx512_kernel();
    }
    else if ((weight_dtype == data_type_t::bf16) &&
             (input_dtype  == data_type_t::bf16) &&
             (output_dtype == data_type_t::f32 ||
              output_dtype == data_type_t::bf16)) {
      kernel = get_matmul_bf16_avx512_kernel();
    }
    else {
      apilog_error("<", name, "> kernel unimplemented.");
      return status_t::unimplemented;
    }
  }
  else {
    if (forced_kernel == "reference") {
      kernel = get_matmul_f32_ref_kernel();
    }
    else if (forced_kernel == "onednn") {
      //kernel = get_linear_onednn_kernel();
      apilog_error("<", name, "> kernel unimplemented using onednn");
      return status_t::unimplemented;
    }
    else {
      apilog_error("<", name, "> kernel unimplemented using forced kernel ",
                   forced_kernel);
      return status_t::unimplemented;
    }
  }

  kernel->create();
  if (! kernel->check()) {
    return kernel->get_last_status();
  }

  return status_t::success;
}

} //namespace ops
} //namespace zendnnl
