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

#include "matmul_onednn_utils.hpp"

namespace zendnnl {
namespace ops {

using namespace zendnnl::common;
using namespace zendnnl::memory;
using namespace zendnnl::error_handling;

#if ZENDNNL_DEPENDS_ONEDNN
dnnl::memory::dims onednn_utils_t::to_dnnl_dims(tensor_t &zendnnl_tensor) {
  auto zendnnl_size =  zendnnl_tensor.get_size();

  dnnl::memory::dims dnnl_dims;
  for (auto &i : zendnnl_size) {
    dnnl_dims.push_back(int64_t(i));
  }

  return dnnl_dims;
}

dnnl::memory::format_tag onednn_utils_t::to_dnnl_format(
  tensor_t &zendnnl_tensor) {

  auto  zendnnl_dim = zendnnl_tensor.get_dim();

  switch (zendnnl_dim) {
  case 1:
    return dnnl::memory::format_tag::a;
  case 2:
    return dnnl::memory::format_tag::ab;
  case 3:
    return dnnl::memory::format_tag::abc;
  default:
    return dnnl::memory::format_tag::ab;
  }

  return dnnl::memory::format_tag::ab;
}

dnnl::memory::data_type onednn_utils_t::to_dnnl_datatype(
  tensor_t &zendnnl_tensor) {

  auto zendnnl_data_type = zendnnl_tensor.get_data_type();

  switch (zendnnl_data_type) {
  case data_type_t::f32:
    return dnnl::memory::data_type::f32;
  case data_type_t::bf16:
    return dnnl::memory::data_type::bf16;
  default:
    return dnnl::memory::data_type::f32;
  }

  return dnnl::memory::data_type::f32;
}

dnnl::memory onednn_utils_t::to_dnnl_tensor(tensor_t &zendnnl_tensor,
    dnnl::engine eng) {

  dnnl::memory::dims tensor_dims       = onednn_utils_t::to_dnnl_dims(
      zendnnl_tensor);
  dnnl::memory::data_type tensor_dtype = onednn_utils_t::to_dnnl_datatype(
      zendnnl_tensor);
  dnnl::memory::format_tag tensor_tag  = onednn_utils_t::to_dnnl_format(
      zendnnl_tensor);

  auto tensor_md  = dnnl::memory::desc(tensor_dims, tensor_dtype, tensor_tag);
  auto tensor_mem = dnnl::memory(tensor_md, eng);

  void *data_handle = zendnnl_tensor.get_raw_handle_unsafe();
  tensor_mem.set_data_handle(data_handle);

  return tensor_mem;
}
#endif


} // namespace ops
} // namespace zendnnl
