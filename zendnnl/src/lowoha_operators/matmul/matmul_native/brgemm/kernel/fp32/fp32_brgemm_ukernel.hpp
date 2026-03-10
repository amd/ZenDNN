/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#ifndef MATMUL_NATIVE_FP32_BRGEMM_UKERNEL_HPP
#define MATMUL_NATIVE_FP32_BRGEMM_UKERNEL_HPP

#include "lowoha_operators/matmul/matmul_native/common/avx512_math.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

template<int MR, int NV>
__attribute__((target("avx512f,fma")))
void brgemm_ukernel(
    const float *__restrict__ pa, int a_stride,
    const float *__restrict__ pb, int b_stride,
    float *__restrict__ C, int ldc,
    int K, int BK, float beta,
    const float *__restrict__ bias,
    fused_postop_t fused_op);

using brgemm_fn_t = void (*)(
    const float *__restrict__ pa, int a_stride,
    const float *__restrict__ pb, int b_stride,
    float *__restrict__ C, int ldc,
    int K, int BK, float beta,
    const float *__restrict__ bias,
    fused_postop_t fused_op);

__attribute__((target("avx512f,avx512bw,fma")))
void brgemm_tail_kernel(
    const float *__restrict__ pa, int a_stride,
    const float *__restrict__ pb, int b_stride,
    float *__restrict__ C, int ldc,
    int K, int BK, int mr_act, int nr_act, float beta,
    const float *__restrict__ bias,
    fused_postop_t fused_op);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_FP32_BRGEMM_UKERNEL_HPP
