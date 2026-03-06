/*******************************************************************************
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef MATMUL_AI_BF16_GEMM_UKERNEL_HPP
#define MATMUL_AI_BF16_GEMM_UKERNEL_HPP

#include "lowoha_operators/matmul/matmul_ai/common/avx512_math.hpp"
#include <cstdint>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ai {

// Function pointer type for BF16 GEMM microkernel dispatch.
// All microkernels share this signature regardless of MR/NR.
using bf16_ukernel_fn_t = void (*)(
    const uint16_t *__restrict__ A, int lda,
    const uint16_t *__restrict__ B_vnni, int b_stride,
    float *__restrict__ C, int ldc,
    int k, float beta,
    const float *__restrict__ bias,
    fused_postop_t fused_op,
    uint16_t *__restrict__ C_bf16, int ldc_bf16);

// Hand-scheduled 6x64 asm microkernel (peak throughput path).
__attribute__((target("avx512f,avx512bf16,fma")))
void bf16_ukernel_6x64_asm(
    const uint16_t *__restrict__ A, int lda,
    const uint16_t *__restrict__ B_vnni, int b_stride,
    float *__restrict__ C, int ldc,
    int k, float beta,
    const float *__restrict__ bias,
    fused_postop_t fused_op,
    uint16_t *__restrict__ C_bf16, int ldc_bf16);

// Select the best microkernel for given MR and NR.
__attribute__((target("avx512f,avx512bf16,fma")))
bf16_ukernel_fn_t select_bf16_ukernel(int MR, int NR);

// Tail microkernel for edge tiles with dynamic MR/NR (masked operations).
__attribute__((target("avx512f,avx512bf16,avx512bw,avx512vl,fma")))
void bf16_tail_kernel(
    const uint16_t *__restrict__ A, int lda,
    const uint16_t *__restrict__ B_vnni, int b_stride,
    float *__restrict__ C, int ldc,
    int k, int mr_act, int nr_act, float beta,
    const float *__restrict__ bias,
    fused_postop_t fused_op,
    uint16_t *__restrict__ C_bf16, int ldc_bf16);

} // namespace ai
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_AI_BF16_GEMM_UKERNEL_HPP
