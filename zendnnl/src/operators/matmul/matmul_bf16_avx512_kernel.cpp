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

#include "matmul_bf16_avx512_kernel.hpp"

namespace zendnnl {
namespace ops {
using namespace zendnnl::error_handling;

status_t matmul_bf16_avx512_kernel_t::execute(const context_type& context_,
                                              tensor_map_type& inputs_,
                                              tensor_map_type& outputs_) {

  LOG_DEBUG_INFO("Executing matmul_bf16_avx512 kernel");
  log_info("Executing matmul_bf16_avx512 kernel");

  auto     aocl_po_ptr        = context_.get_aocl_post_op_ptr_unsafe();
  auto     reorder_weights    = (int16_t*)context_.get_aocl_reordered_weights_ptr_unsafe();
  auto     input_tensor       = inputs_.find("matmul_input")->second;
  auto     output_tensor      = outputs_.find("matmul_output")->second;
  auto     weight_tensor      = context_.get_param("weights").value();

  int16_t* input_raw_handle   = (int16_t *)input_tensor.get_raw_handle_unsafe();
  void*    output_raw_handle  = output_tensor.get_raw_handle_unsafe();
  int16_t* weights_raw_handle = (int16_t *)weight_tensor.get_raw_handle_unsafe();

  bool is_transpose_weights = weight_tensor.get_order() == "ba";
  bool is_transpose_src = input_tensor.get_order() == "ba";
  bool is_blocked = weight_tensor.get_layout() == tensor_layout_t::blocked ? true : false;

  bool is_reordered_weights = false;
  if (reorder_weights != nullptr) {
    log_info("Using reordered weights");
    is_blocked = true;
    is_reordered_weights = true;
  }
  const int m           = input_tensor.get_size(0);
  const int k           = input_tensor.get_size(1);
  const int n           = output_tensor.get_size(1);

  const char  order         = 'r';
  const char  trans_input   = is_transpose_src ? 't' : 'n';
  const char  trans_weight  = is_transpose_weights ? 't' : 'n';
  const char  input_format  = 'n';
  const char  weight_format = is_blocked ? 'r': 'n';
  const float alpha         = 1.0;
  const float beta          = 0.0;
  const int   lda           = is_transpose_src ? m : k;
  const int   ldb           = is_transpose_weights ? k : n;
  const int   ldc           = n;

  if (output_tensor.get_data_type() == data_type_t::f32) {
    aocl_gemm_bf16bf16f32of32(order, trans_input, trans_weight,
                              m,n,k,
                              alpha,
                              input_raw_handle, lda, input_format,
                              is_reordered_weights ? reorder_weights : weights_raw_handle,
                              ldb, weight_format,
                              beta,
                              (float*)output_raw_handle, ldc,
                              aocl_po_ptr);
  }
  else if (output_tensor.get_data_type() == data_type_t::bf16) {
    aocl_gemm_bf16bf16f32obf16(order, trans_input, trans_weight,
                                m,n,k,
                                alpha,
                                input_raw_handle, lda, input_format,
                                is_reordered_weights ? reorder_weights : weights_raw_handle,
                                ldb, weight_format,
                                beta,
                                (int16_t*)output_raw_handle, ldc,
                                aocl_po_ptr);
  }
  else {
    log_error("Unsupported output data type");
    return status_t::failure;
  }

  return status_t::success;
}

} //namespace ops
} //namespace zendnnl

extern "C" {
  std::shared_ptr<zendnnl::ops::matmul_bf16_avx512_kernel_t> get_matmul_bf16_avx512_kernel() {
    return std::make_shared<zendnnl::ops::matmul_bf16_avx512_kernel_t>();
  }
}
