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
dnnl::memory::format_tag onednn_utils_t::to_dnnl_format(std::string tag) {
  if (tag == "a") {
    return dnnl::memory::format_tag::a;
  }
  else if (tag == "ab") {
    return dnnl::memory::format_tag::ab;
  }
  else if (tag == "ba") {
    return dnnl::memory::format_tag::ba;
  }
  else if (tag == "abc") {
    return dnnl::memory::format_tag::abc;
  }
  else if (tag == "acb") {
    return dnnl::memory::format_tag::acb;
  }
  else if (tag == "BA16a64b2a") {
    return dnnl::memory::format_tag::BA16a64b2a;
  }
  else if (tag == "BA16a64b") {
    return dnnl::memory::format_tag::BA16a64b;
  }
  else if (tag == "AB8b64a2b") {
    return dnnl::memory::format_tag::AB8b64a2b;
  }
  else {
    return dnnl::memory::format_tag::any;
  }
}

dnnl::memory::data_type onednn_utils_t::to_dnnl_datatype(
  data_type_t dtype) {
  switch (dtype) {
  case data_type_t::f32:
    return dnnl::memory::data_type::f32;
  case data_type_t::bf16:
    return dnnl::memory::data_type::bf16;
  default:
    return dnnl::memory::data_type::f32;
  }

  return dnnl::memory::data_type::f32;
}

dnnl::memory::desc onednn_utils_t::to_dnnl_tensor(const onednn_tensor_params
    &params,
    dnnl::engine eng) {
  dnnl::memory::dims tensor_dims = params.dims;
  [[maybe_unused]] dnnl::memory::dims stride_dims = params.strides;
  dnnl::memory::data_type tensor_dtype = onednn_utils_t::to_dnnl_datatype(
      params.dtype);
  dnnl::memory::format_tag tensor_tag  = onednn_utils_t::to_dnnl_format(
      params.format_tag);

  dnnl::memory::desc tensor_md;
  if (params.format_tag == "any" || params.format_tag == "BA16a64b" ||
      params.format_tag == "BA16a64b2a" || params.format_tag == "AB8b64a2b" || stride_dims.empty()) {
    tensor_md = dnnl::memory::desc(tensor_dims, tensor_dtype, tensor_tag);
  }
  else {
    tensor_md = dnnl::memory::desc(tensor_dims, tensor_dtype, stride_dims);
  }

  return tensor_md;
}

#endif


} // namespace ops
} // namespace zendnnl
