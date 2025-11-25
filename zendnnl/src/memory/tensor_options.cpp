/********************************************************************************
# * Copyright (c) 2025-2028 Advanced Micro Devices, Inc. All rights reserved.
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
  size{}, aligned_size{}, stride{}, base{},
  nelem{0}, aligned_nelem{0}, base_offset{0},
  data_type{data_type_t::f32}, layout{0},
  is_const{false}, order{} {
}

void tensor_option_t::reset() {
  LOG_DEBUG_INFO("Resetting the tensor");
  parent_type::reset();

  size.clear();
  aligned_size.clear();
  stride.clear();
  base.clear();
  nelem          = 0;
  aligned_nelem  = 0;
  base_offset    = 0;
  data_type      = data_type_t::f32;
  layout         = 0;
  is_const       = false;
  order          = std::string();
}

bool tensor_option_t::is_contiguous() const {
  return (layout == uint16_t(tensor_layout_t::contiguous));
}

bool tensor_option_t::is_aligned() const {
  return (layout & uint16_t(tensor_layout_t::aligned));
}

bool tensor_option_t::is_broadcast() const {
  return (layout & uint16_t(tensor_layout_t::broadcast));
}

bool tensor_option_t::is_transpose() const {
  return (layout & uint16_t(tensor_layout_t::transpose));
}

bool tensor_option_t::is_quantized() const {
  return (layout & uint16_t(tensor_layout_t::quantized));
}

bool tensor_option_t::is_blocked() const {
  return ((layout & uint16_t(tensor_layout_t::blocked)) |
          (layout & uint16_t(tensor_layout_t::blocked_aocl)) |
          (layout & uint16_t(tensor_layout_t::blocked_onednn)) |
          (layout & uint16_t(tensor_layout_t::blocked_libxsmm)));
}

bool tensor_option_t::is_oblique() const {
  return (layout & uint16_t(tensor_layout_t::oblique));
}

std::size_t tensor_option_t::hash() {
  LOG_DEBUG_INFO("Generating tensor option hash");

  if (hash_key) {
    return hash_key;
  }

  hash_key = hash_combine(hash_key, size);
  hash_key = hash_combine(hash_key, aligned_size);
  hash_key = hash_combine(hash_key, base);
  hash_key = hash_combine(hash_key, uint32_t(data_type));
  hash_key = hash_combine(hash_key, uint32_t(layout));
  hash_key = hash_combine(hash_key, uint32_t(is_const));
  hash_key = hash_combine(hash_key, order);

  return hash_key;
}

} //memory
} //zendnnl
