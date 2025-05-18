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
#include "sample_operator.hpp"
#include "sample_kernel_list.hpp"

namespace zendnnl {
namespace ops {

status_t sample_operator_t::validate() {

  if (parent_type::validate() != status_t::success)
    return status_t::failure;

  if (!get_input("sample_input") || !get_output("sample_output"))
    return status_t::failure;

  return status_t::success;
}

status_t sample_operator_t::kernel_factory() {

  auto input_dtype = get_input("sample_input")->get_data_type();

  if (input_dtype == data_type_t::f32)
    kernel = get_sample_f32_avx512_kernel();
  else if (input_dtype == data_type_t::bf16)
    kernel = get_sample_bf16_avx512_kernel();
  else
    return status_t::unimplemented;

  kernel->create();
  if (! kernel->check())
    return status_t::failure;

  return status_t::success;
}

} //namespace ops
} //namespace zendnnl

