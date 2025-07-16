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

#include "tensor_quant.hpp"

namespace zendnnl {
namespace memory {

using namespace zendnnl::common;
using namespace zendnnl::error_handling;

using quant_storage_t = tensor_storage_t;

tensor_quant_t::tensor_quant_t()
  :type{quant_type_t::none}, subtype{quant_subtype_t::none},
   scale_size{}, scale_stride{}, scale_block_size{},
   scale_data_type{data_type_t::none}, scales{nullptr},
   zero_size{}, zero_stride{}, zero_block_size{},
   zero_data_type{data_type_t::none}, zeros{nullptr} {
}

void tensor_quant_t::reset() {
  type              = quant_type_t::none;
  subtype           = quant_subtype_t::none;

  scale_size.clear();
  scale_stride.clear();
  scale_block_size.clear();
  scale_data_type   = data_type_t::none;
  scales.reset();

  zero_size.clear();
  zero_stride.clear();
  zero_block_size.clear();
  zero_data_type    = data_type_t::none;
  zeros.reset();
}

std::size_t tensor_quant_t::hash() {

  if (hash_key) {
    return hash_key;
  }

  hash_key = hash_combine(hash_key, uint32_t(type));
  hash_key = hash_combine(hash_key, uint32_t(subtype));
  hash_key = hash_combine(hash_key, uint32_t(scale_data_type));
  hash_key = hash_combine(hash_key, uint32_t(zero_data_type));
  hash_key = hash_combine(hash_key, scale_size);
  hash_key = hash_combine(hash_key, zero_size);

  return hash_key;
}

}//memory
}//zendnnl
