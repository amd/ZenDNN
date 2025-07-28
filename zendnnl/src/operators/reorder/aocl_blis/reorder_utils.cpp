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

#include "reorder_utils.hpp"

namespace zendnnl {
namespace ops {
using namespace zendnnl::error_handling;

size_t aocl_blis_reorder_utils_t::get_aocl_reorder_size(
  const reorder_context_t &context, const tensor_t &input_tensor) {
  size_t reorder_size        = 0;

  auto source_dtype          = context.get_source_dtype();
  auto reorder_dtype         = input_tensor.get_data_type();

  const char reorder_param0  = 'B';
  const dim_t reorder_param1 = input_tensor.get_size(0);
  const dim_t reorder_param2 = input_tensor.get_size(1);
  const char order           = 'r';
  bool is_transpose          = input_tensor.get_order() == "ba";
  const char trans           = is_transpose ? 't' : 'n';

  if (reorder_dtype == data_type_t::f32) {
    reorder_size = aocl_get_reorder_buf_size_f32f32f32of32(order, trans,
                   reorder_param0,
                   reorder_param1, reorder_param2);
  }
  else if (reorder_dtype == data_type_t::bf16) {
    reorder_size = aocl_get_reorder_buf_size_bf16bf16f32of32(order, trans,
                   reorder_param0,
                   reorder_param1, reorder_param2);
  }
  else if (reorder_dtype == data_type_t::s8) {
    if (source_dtype == data_type_t::s8) {
      reorder_size = aocl_get_reorder_buf_size_s8s8s32os32(order, trans,
                     reorder_param0,
                     reorder_param1, reorder_param2);
    }
    else if (source_dtype == data_type_t::u8) {
      reorder_size = aocl_get_reorder_buf_size_u8s8s32os32(order, trans,
                     reorder_param0,
                     reorder_param1, reorder_param2);
    }
  }
  else if (reorder_dtype == data_type_t::s4) {
    // WOQ_BF16 api to extract the size.
    reorder_size = aocl_get_reorder_buf_size_bf16s4f32of32(order, trans,
                   reorder_param0,
                   reorder_param1, reorder_param2);
  }
  return reorder_size;
}

} // namespace ops
} // namespace zendnnl