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
#include "dlsample_avx2_kernels.hpp"

namespace zendnnl {
namespace ops {
using namespace zendnnl::error_handling;
status_t dlsample_f32_avx2_kernel_t::execute(const context_type& context_,
                                             tensor_map_type& inputs_,
                                             tensor_map_type& outputs_) {

  log_info("dlsample_fp32_avx2_kernel");

  return status_t::success;
}

status_t dlsample_bf16_avx2_kernel_t::execute(const context_type& context_,
                                              tensor_map_type& inputs_,
                                              tensor_map_type& outputs_) {

  log_info("dlsample_bf16_avx2_kernel");

  return status_t::success;
}

extern "C" {
  std::shared_ptr<dlsample_f32_avx2_kernel_t> get_dlsample_f32_avx2_kernel() {
    return std::make_shared<dlsample_f32_avx2_kernel_t>();
  }

  std::shared_ptr<dlsample_bf16_avx2_kernel_t> get_dlsample_bf16_avx2_kernel() {
    return std::make_shared<dlsample_bf16_avx2_kernel_t>();
  }
}

} //namespace ops
} //namespace zendnnl

