/********************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ZENDNNL_LOWOHA_MATMUL_BACKENDS_REFERENCE_REFERENCE_KERNEL_HPP
#define ZENDNNL_LOWOHA_MATMUL_BACKENDS_REFERENCE_REFERENCE_KERNEL_HPP

#include "common/bfloat16.hpp"
#include "lowoha_operators/matmul/lowoha_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

status_t reference_matmul_execute(const char layout, const bool transA,
        const bool transB, const int M, const int N, const int K,
        const float alpha, const void *src, const int lda, const void *weight,
        const int ldb, const void *bias, const float beta, void *dst,
        const int ldc, const bool is_weights_const,
        matmul_batch_params_t &batch_params, matmul_params &params);

// W4A8 / GGML: widen packed s4 [k, n/2] -> s8 [k, n] (sign-extended nibbles).
void cvt_s4_to_s8(const int8_t *weights, int8_t *wei_s8, int k, int n, int ldb,
        bool is_transposed);

} //namespace matmul
} //namespace lowoha
} //namespace zendnnl

#endif
