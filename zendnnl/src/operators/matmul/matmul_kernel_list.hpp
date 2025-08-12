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
#ifndef _MATMUL_KERNEL_LIST_HPP_
#define _MATMUL_KERNEL_LIST_HPP_

#include "matmul_ref_kernel.hpp"
#include "aocl_blis/matmul_fp32_avx512_kernel.hpp"
#include "aocl_blis/matmul_bf16_avx512_kernel.hpp"
#if ZENDNNL_DEPENDS_ONEDNN
  #include "matmul_onednn_kernel.hpp"
#endif
#endif
