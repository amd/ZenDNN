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

#include "reorder_kernel.hpp"

namespace zendnnl {
namespace ops {
using namespace zendnnl::error_handling;
status_t reorder_kernel_t::execute(const context_type &context_,
                                   tensor_map_type &inputs_,
                                   tensor_map_type &outputs_) {
  log_info("reorder_kernel");
  auto     reorder_algo    = context_.get_algo_format();
  if (reorder_algo == "aocl") {
    auto     input_tensor  = inputs_.find("reorder_input")->second;
    auto     input_dtype   = input_tensor.get_data_type();
    auto     output_tensor = outputs_.find("reorder_output")->second;
    auto     source_dtype  = context_.get_source_dtype();

    void     *input        = input_tensor.get_raw_handle_unsafe();
    void     *output       = output_tensor.get_raw_handle_unsafe();

    const int K            = input_tensor.get_size(0);
    const int N            = input_tensor.get_size(1);

    const char reorder_param0   = 'B';
    const char order            = 'r';
    bool is_transpose           = input_tensor.get_order() == "ba";
    const char trans            = is_transpose ? 't' : 'n';
    int ldb                     = is_transpose ? K : N;

    size_t reorder_size         = output_tensor.get_buffer_sz_bytes();
    void *reorder_weights       = aligned_alloc(64, reorder_size);

    if (input_dtype == data_type_t::f32) {
      aocl_reorder_f32f32f32of32(order, trans, reorder_param0, (float *)input,
                                 (float *)reorder_weights, K, N, ldb);
      data_copy<float>(output, reorder_weights, reorder_size);
    }
    else if (input_dtype == data_type_t::bf16) {
      aocl_reorder_bf16bf16f32of32(order, trans, reorder_param0,(int16_t *)input,
                                   (int16_t *)reorder_weights, K, N, ldb);
      data_copy<int16_t>(output, reorder_weights, reorder_size);
    }
    else if (input_dtype == data_type_t::s8) {
      if (source_dtype == data_type_t::s8) {
        aocl_reorder_s8s8s32os32(order, trans, reorder_param0, (int8_t *)input,
                                 (int8_t *)reorder_weights, K, N, ldb);
      }
      else if (source_dtype == data_type_t::u8) {
        aocl_reorder_u8s8s32os32(order, trans, reorder_param0, (int8_t *)input,
                                 (int8_t *)reorder_weights, K, N, ldb);
      }
      data_copy<int8_t>(output, reorder_weights, reorder_size);
    }
    else if (input_dtype == data_type_t::s4) {
      // WOQ_BF16 api to reorder.
      aocl_reorder_bf16s4f32of32(order, trans, reorder_param0, (int8_t *)input,
                                 (int8_t *)reorder_weights, K, N, ldb);
      data_copy<int8_t>(output, reorder_weights, reorder_size);
    }
    free(reorder_weights);
  }
  else if (reorder_algo == "onednn") {
    return status_t::failure;
  }
  return status_t::success;
}

} //namespace ops
} //namespace zendnnl

extern "C" {
  std::shared_ptr<zendnnl::ops::reorder_kernel_t> get_reorder_aocl_kernel() {
    return std::make_shared<zendnnl::ops::reorder_kernel_t>();
  }
}
