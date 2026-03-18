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

#pragma once
#include <cstdint>
#include "lowoha_operators/matmul/matmul_native/common/avx512_math.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

/// K-contiguous GEMV kernel: computes C = A * B_kc + bias, with optional
/// fused activation and beta scaling. Writes BF16 or FP32 output based on
/// dst_is_bf16 flag (pass the corresponding pointer, nullptr for the other).
void bf16_gemv_kcontiguous(
    const uint16_t *__restrict__ A,
    const uint16_t *__restrict__ B_kc,
    uint16_t *__restrict__ C_bf16,
    float *__restrict__ C_fp32,
    const float *__restrict__ bias_f,
    fused_postop_t fused_op,
    float beta,
    bool dst_is_bf16,
    int K, int N);

void pack_b_kcontiguous_ext(
    const uint16_t *B, int ldb, int K, int N, bool transB,
    uint16_t *packed);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
