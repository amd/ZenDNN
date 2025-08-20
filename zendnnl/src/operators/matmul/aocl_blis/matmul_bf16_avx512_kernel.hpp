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
#ifndef _MATMUL_BF16_AVX512_KERNEL_HPP_
#define _MATMUL_BF16_AVX512_KERNEL_HPP_

#include <vector>
#include <iostream>
#include <memory>
#include <cstring>
#include <cstdlib>

#include "common/zendnnl_global.hpp"
#include "operators/common/operator_kernel.hpp"
#include "operators/matmul/matmul_context.hpp"

#if defined(ZENDNNL_DEPENDS_AOCLDLP)
#include "aocl_dlp.h"
#else
#include "blis.h"
#endif

namespace zendnnl {
namespace ops {

class matmul_bf16_avx512_kernel_t final : public op_kernel_t<matmul_context_t> {
public:
  status_t execute(const context_type& context_,
                   tensor_map_type& inputs_,
                   tensor_map_type& outputs_) override;
};

} //namespace ops
} //namespace zendnnl

extern "C" {
  std::shared_ptr<zendnnl::ops::matmul_bf16_avx512_kernel_t> get_matmul_bf16_avx512_kernel();
}

#endif
