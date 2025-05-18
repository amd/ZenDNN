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
#ifndef _COMPARE_EXECUTE_HPP_
#define _COMPARE_EXECUTE_HPP_

#include <vector>
#include <iostream>
#include <memory>
#include <cstring>
#include <cstdlib>
#include "compare_context.hpp"
#include "operators/common/operator_kernel.hpp"

namespace zendnnl {
namespace ops {

using namespace zendnnl::error_handling;

class compare_ref_kernel_t final : public op_kernel_t<compare_context_t> {
public:
  status_t execute(const context_type& context_,
                   tensor_map_type& inputs_,
                   tensor_map_type& outputs_) override;
};

} //namespace ops
} //namespace zendnnl

extern "C" {
  std::shared_ptr<zendnnl::ops::compare_ref_kernel_t> get_compare_kernel();
}

#endif
