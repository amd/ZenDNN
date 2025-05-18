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
#ifndef _EMBAG_AVX2_KERNELS_HPP_
#define _EMBAG_AVX2_KERNELS_HPP_

#include <iostream>
#include <memory>
#include "operator_kernel.hpp"
#include "error_handling.hpp"
#include "embag_context.hpp"

namespace zendnnl {
namespace ops {

using namespace zendnnl::error_handling;

class embag_f32_avx2_kernel_t final : public op_kernel_t<embag_context_t> {
public:
  ~embag_f32_avx2_kernel_t() = default;

  status_t execute(const context_type& context_,
                   tensor_map_type& inputs_,
                   tensor_map_type& outputs_) override;
};

class embag_bf16_avx2_kernel_t final : public op_kernel_t<embag_context_t> {
public:
  ~embag_bf16_avx2_kernel_t() = default;

  status_t execute(const context_type& context_,
                   tensor_map_type& inputs_,
                   tensor_map_type& outputs_) override;
};

extern "C" {
  std::shared_ptr<embag_f32_avx2_kernel_t> get_embag_f32_avx2_kernel();
  std::shared_ptr<embag_bf16_avx2_kernel_t> get_embag_bf16_avx2_kernel();
}


} //namespace ops
} //namespace zendnnl


#endif
