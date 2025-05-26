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

#include "tensor_options.hpp"

namespace zendnnl {
namespace memory {

using namespace zendnnl::common;
using namespace zendnnl::error_handling;

tensor_option_t::tensor_option_t():
  size{}, stride_size{}, stride{}, base{},
  nelem{0}, strided_nelem{0}, base_offset{0},
  data_type{data_type_t::f32}, layout{tensor_layout_t::contiguous},
  is_const{false}, order{} {
}

void tensor_option_t::reset() {
  LOG_DEBUG_INFO("Resetting the tensor");
  parent_type::reset();

  size.clear();
  stride_size.clear();
  stride.clear();
  base.clear();
  nelem          = 0;
  strided_nelem  = 0;
  base_offset    = 0;
  data_type      = data_type_t::f32;
  layout         = tensor_layout_t::contiguous;
  is_const       = false;
  order          = std::string();
}

std::size_t tensor_option_t::hash() {
  LOG_DEBUG_INFO("Generating tensor option hash");

  if(hash_key)
    return hash_key;

  hash_key = hash_combine(hash_key, size);
  hash_key = hash_combine(hash_key, stride_size);
  hash_key = hash_combine(hash_key, base);
  hash_key = hash_combine(hash_key, uint32_t(data_type));
  hash_key = hash_combine(hash_key, uint32_t(layout));
  hash_key = hash_combine(hash_key, uint32_t(is_const));
  hash_key = hash_combine(hash_key, order);

  return hash_key;
}

tensor_quant_t::tensor_quant_t():
  zero_point{0},scale{0} {
}

void tensor_quant_t::reset() {
  LOG_DEBUG_INFO("Resetting qtensor");
  parent_type::reset();

  zero_point = 0;
  scale      = 0;
}

std::size_t tensor_quant_t::hash() {
  LOG_DEBUG_INFO("Generating qtensor hash");
  return hash_key;
}

} //memory
} //zendnnl
