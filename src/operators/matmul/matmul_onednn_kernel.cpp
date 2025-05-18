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

#include "error_handling.hpp"
#include "matmul_onednn_kernel.hpp"

namespace zendnnl {
namespace ops {

using namespace zendnnl::common;
using namespace zendnnl::memory;
using namespace zendnnl::error_handling;
//using namespace dnnl;

matmul_onednn_kernel_t::~matmul_onednn_kernel_t() {
}

dnnl::memory::dims matmul_onednn_kernel_t::to_dnnl_dims(tensor_t& zendnnl_tensor) {
  auto zendnnl_size =  zendnnl_tensor.get_size();

  dnnl::memory::dims dnnl_dims;
  for (auto& i : zendnnl_size) {
    dnnl_dims.push_back(int64_t(i));
  }

  return dnnl_dims;
}

dnnl::memory::format_tag matmul_onednn_kernel_t::to_dnnl_format(tensor_t& zendnnl_tensor) {

  auto  zendnnl_dim = zendnnl_tensor.get_dim();

  switch (zendnnl_dim){
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

dnnl::memory::data_type matmul_onednn_kernel_t::to_dnnl_datatype(tensor_t& zendnnl_tensor) {

  auto zendnnl_data_type = zendnnl_tensor.get_data_type();

  switch(zendnnl_data_type) {
  case data_type_t::f32:
    return dnnl::memory::data_type::f32;
  case data_type_t::bf16:
    return dnnl::memory::data_type::bf16;
  default:
    return dnnl::memory::data_type::f32;
  }

  return dnnl::memory::data_type::f32;
}

dnnl::memory matmul_onednn_kernel_t::to_dnnl_tensor(tensor_t& zendnnl_tensor,
                                                    dnnl::engine eng) {

  dnnl::memory::dims tensor_dims       = to_dnnl_dims(zendnnl_tensor);
  dnnl::memory::data_type tensor_dtype = to_dnnl_datatype(zendnnl_tensor);
  dnnl::memory::format_tag tensor_tag  = to_dnnl_format(zendnnl_tensor);

  auto tensor_md  = dnnl::memory::desc(tensor_dims, tensor_dtype, tensor_tag);
  auto tensor_mem = dnnl::memory(tensor_md, eng);

  void* data_handle = zendnnl_tensor.get_raw_handle_unsafe();
  tensor_mem.set_data_handle(data_handle);

  return tensor_mem;
}

status_t matmul_onednn_kernel_t::execute(const context_type& context_,
                                         tensor_map_type& inputs_,
                                         tensor_map_type& outputs_) {

  log_info("matmul onednn kernel");

  auto     input_tensor  = inputs_.find("matmul_input")->second;
  auto     output_tensor = outputs_.find("matmul_output")->second;
  auto     weight_tensor = context_.get_param("weights").value();

  dnnl::engine eng(dnnl::engine::kind::cpu, 0);
  dnnl::stream eng_stream(eng);

  dnnl::memory  dnnl_input_tensor   = to_dnnl_tensor(input_tensor, eng);
  dnnl::memory  dnnl_output_tensor  = to_dnnl_tensor(output_tensor, eng);
  dnnl::memory  dnnl_weight_tensor  = to_dnnl_tensor(weight_tensor, eng);

  auto dnnl_input_desc  = dnnl_input_tensor.get_desc();
  auto dnnl_output_desc = dnnl_output_tensor.get_desc();
  auto dnnl_weight_desc = dnnl_weight_tensor.get_desc();

  auto matmul_pd = dnnl::matmul::primitive_desc(eng, dnnl_input_desc,
                                                dnnl_weight_desc,
                                                dnnl_output_desc);

  auto matmul_prim = dnnl::matmul(matmul_pd);

  std::unordered_map<int, dnnl::memory> matmul_args;
  matmul_args.insert({DNNL_ARG_SRC, dnnl_input_tensor});
  matmul_args.insert({DNNL_ARG_WEIGHTS, dnnl_weight_tensor});
  matmul_args.insert({DNNL_ARG_DST, dnnl_output_tensor});

  matmul_prim.execute(eng_stream, matmul_args);

  eng_stream.wait();

  return status_t::success;
}

} //namespace ops
} //namespace zendnnl

extern "C" {
  std::shared_ptr<zendnnl::ops::matmul_onednn_kernel_t> get_matmul_onednn_kernel() {
    return std::make_shared<zendnnl::ops::matmul_onednn_kernel_t>();
  }
}
