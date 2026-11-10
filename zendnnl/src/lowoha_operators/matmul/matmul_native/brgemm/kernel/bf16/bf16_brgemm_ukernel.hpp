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

#ifndef MATMUL_NATIVE_BF16_BRGEMM_UKERNEL_HPP
#define MATMUL_NATIVE_BF16_BRGEMM_UKERNEL_HPP

#include <cstddef>
#include <cstdint>
#include "common/zendnnl_compat.hpp"
#include "lowoha_operators/matmul/matmul_native/common/avx512_math.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using bf16_brgemm_fn_t = void (*)(const uint16_t *__restrict A, int lda,
        const uint16_t *__restrict B_vnni, int b_stride, float *__restrict C,
        int ldc, int K, int BK, float beta, const float *__restrict bias,
        fused_postop_t fused_op, uint16_t *__restrict C_bf16, int ldc_bf16);

/// The BF16 BRGEMM kernel of MR rows by 32 columns of a paired tile, as
/// bf16_brgemm_fn_t's but with the second 16 columns of B b_second elements
/// after B_vnni (a gated call's up run, beyond the int range on a large
/// weight).
using bf16_brgemm_paired_fn_t = void (*)(const uint16_t *__restrict A, int lda,
        const uint16_t *__restrict B_vnni, int b_stride, size_t b_second,
        float *__restrict C, int ldc, int K, int BK, float beta,
        const float *__restrict bias, fused_postop_t fused_op,
        uint16_t *__restrict C_bf16, int ldc_bf16);

/// The paired kernels by MR, 1 .. 12 (entry 0 null).
const bf16_brgemm_paired_fn_t *bf16_brgemm_paired_table();

/// The BF16 BRGEMM kernel of MR rows by NR columns: MR 1 .. 6 at NR 16 or
/// 32; nullptr for any other.
ZENDNNL_TARGET("avx512f,avx512bf16,fma")
bf16_brgemm_fn_t select_bf16_brgemm_kernel(int MR, int NR);

/// select_bf16_brgemm_kernel plus MR 7 .. 12 at NR 32: the BF16 kernels of
/// the native kernel registry.
ZENDNNL_TARGET("avx512f,avx512bf16,fma")
bf16_brgemm_fn_t select_bf16_brgemm_tall_kernel(int MR, int NR);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_BF16_BRGEMM_UKERNEL_HPP
