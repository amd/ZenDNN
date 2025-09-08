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

#include "matmul_onednn_kernel.hpp"

namespace zendnnl {
namespace ops {

using namespace zendnnl::common;
using namespace zendnnl::memory;
using namespace zendnnl::error_handling;

matmul_onednn_kernel_t::~matmul_onednn_kernel_t() {
}

status_t matmul_onednn_kernel_t::execute(const context_type &context_,
    tensor_map_type &inputs_,
    tensor_map_type &outputs_) {
#if ZENDNNL_DEPENDS_ONEDNN
  log_info("matmul onednn kernel");

  auto     input_tensor  = inputs_.find("matmul_input")->second;
  auto     output_tensor = outputs_.find("matmul_output")->second;
  auto     weight_tensor = context_.get_param("weights").value();

  dnnl::engine eng(dnnl::engine::kind::cpu, 0);
  dnnl::stream eng_stream(eng);

  dnnl::memory  dnnl_input_tensor   = onednn_utils_t::to_dnnl_tensor(input_tensor,
                                      eng);
  dnnl::memory  dnnl_output_tensor  = onednn_utils_t::to_dnnl_tensor(
                                        output_tensor, eng);
  dnnl::memory  dnnl_weight_tensor  = onednn_utils_t::to_dnnl_tensor(
                                        weight_tensor, eng);

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
#else
  log_error("onednn dependency is disabled");
  return status_t::failure;
#endif
}

} //namespace ops
} //namespace zendnnl

extern "C" {
  zendnnl::ops::matmul_onednn_kernel_t *get_matmul_onednn_kernel() {
    return new zendnnl::ops::matmul_onednn_kernel_t();
  }
}
