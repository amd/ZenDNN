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

#include "matmul_bf16_avx512_kernel.hpp"

namespace zendnnl {
namespace ops {
using namespace zendnnl::error_handling;

status_t matmul_bf16_avx512_kernel_t::execute(const context_type &context_,
    tensor_map_type &inputs_,
    tensor_map_type &outputs_) {

  LOG_DEBUG_INFO("Executing matmul_bf16_avx512 kernel");
  log_info("Executing matmul_bf16_avx512 kernel");

  auto     aocl_dlp_po_ptr   = context_.get_aocl_dlp_post_op_ptr_unsafe();

  auto input_iter = inputs_.find("matmul_input");
  auto output_iter = outputs_.find("matmul_output");

  if (input_iter == inputs_.end()) {
    log_error("matmul_input tensor not found");
    return status_t::failure;
  }
  if (output_iter == outputs_.end()) {
    log_error("matmul_output tensor not found");
    return status_t::failure;
  }

  const auto &input_tensor = input_iter->second;
  const auto &output_tensor = output_iter->second;

  const auto weight_param = context_.get_param("weights");
  const auto &weight_tensor = weight_param.value();

  int16_t *input_raw_handle   = (int16_t *)input_tensor.get_raw_handle_unsafe();
  void    *output_raw_handle  = output_tensor.get_raw_handle_unsafe();
  int16_t *weights_raw_handle = (int16_t *)weight_tensor.get_raw_handle_unsafe();

  auto input_dim              = input_tensor.get_dim();
  auto weight_dim             = weight_tensor.get_dim();
  auto output_dim             = output_tensor.get_dim();

  bool is_transpose_src       = (input_dim == 2)  ? (input_tensor.get_order() ==
                                "ba") : (input_tensor.get_order() == "acb");
  bool is_transpose_weights   = (weight_dim == 2) ? (weight_tensor.get_order() ==
                                "ba") : (weight_tensor.get_order() == "acb");

  bool is_blocked = weight_tensor.get_layout() & uint8_t(
                      tensor_layout_t::blocked);

  auto reorder_weights        = (int16_t *)
                                context_.get_aocl_dlp_reordered_weights_ptr_unsafe();
  bool is_reordered_weights   = false;
  if (reorder_weights != nullptr) {
    log_info("Using reordered weights");
    is_blocked                = true;
    is_reordered_weights      = true;
  }

  const int batch_size        = (output_dim==3) ? output_tensor.get_size(
                                  output_dim-3) : 1;
  const int m                 = output_tensor.get_size(output_dim-2);
  const int k                 = input_tensor.get_size(input_dim-1);
  const int n                 = output_tensor.get_size(output_dim-1);
  const char order            = 'r';
  const char trans_input      = is_transpose_src ? 't' : 'n';
  const char trans_weight     = is_transpose_weights ? 't' : 'n';
  const char input_format     = 'n';
  const char weight_format    = is_blocked ? 'r': 'n';
  const float alpha           = context_.get_alpha();
  const float beta            = context_.get_beta();

  const int   lda             = is_transpose_src ?
                                input_tensor.get_stride(input_dim-1) :
                                input_tensor.get_stride(input_dim-2);
  const int   ldb             = is_transpose_weights ?
                                weight_tensor.get_stride(weight_dim-1):
                                weight_tensor.get_stride(weight_dim-2);
  const int   ldc             = output_tensor.get_stride(output_dim-2);

  unsigned int offset_src     = (input_dim == 3) ? input_tensor.get_stride(
                                  input_dim-3) : 0;
  unsigned int offset_wei     = (weight_dim == 3) ? weight_tensor.get_stride(
                                  weight_dim-3) : 0;
  unsigned int offset_out     = (output_dim == 3) ? output_tensor.get_stride(
                                  output_dim-3) : 0;

  if (output_tensor.get_data_type() == data_type_t::f32) {
    for (auto bs=0; bs<batch_size; ++bs) {
      aocl_gemm_bf16bf16f32of32(order, trans_input, trans_weight,
                                m,n,k,
                                alpha,
                                input_raw_handle + bs * offset_src, lda, input_format,
                                (is_reordered_weights && weight_dim==2) ?
                                reorder_weights : weights_raw_handle + bs * offset_wei,
                                ldb, weight_format, beta,
                                (float *)output_raw_handle + bs * offset_out, ldc,
                                aocl_dlp_po_ptr);
    }
  }
  else if (output_tensor.get_data_type() == data_type_t::bf16) {
    for (auto bs=0; bs<batch_size; ++bs) {
      aocl_gemm_bf16bf16f32obf16(order, trans_input, trans_weight,
                                 m,n,k,
                                 alpha,
                                 input_raw_handle + bs * offset_src, lda, input_format,
                                 (is_reordered_weights && weight_dim==2) ?
                                 reorder_weights : weights_raw_handle + bs * offset_wei,
                                 ldb, weight_format, beta,
                                 (int16_t *)output_raw_handle + bs * offset_out,
                                 ldc, aocl_dlp_po_ptr);
    }
  }
  return status_t::success;
}

} //namespace ops
} //namespace zendnnl

extern "C" {
  zendnnl::ops::matmul_bf16_avx512_kernel_t *
  get_matmul_bf16_avx512_kernel() {
    return new zendnnl::ops::matmul_bf16_avx512_kernel_t();
  }
}
