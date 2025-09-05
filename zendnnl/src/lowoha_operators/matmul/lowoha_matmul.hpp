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

#ifndef _LOWOHA_MATMUL_HPP
#define _LOWOHA_MATMUL_HPP

#include "operators/matmul/matmul_context.hpp"

namespace zendnnl {
namespace lowoha {

using namespace zendnnl::common;
using namespace zendnnl::ops;

void matmul_direct(const void *src, const void *weight, void *dst, void *bias,
                   float alpha,
                   float beta, int M, int N, int K, bool transA, bool transB, int lda, int ldb,
                   int ldc, data_type_t src_data_type, data_type_t out_data_type,
                   post_op_type_t post_op, void *post_op_buff, int Batch_A = 1, int Batch_B = 1);

} // namespace lowoha
} // namespace zendnnl

#endif

