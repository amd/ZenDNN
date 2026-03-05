/*******************************************************************************
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#ifndef MATMUL_AI_BRGEMM_INTRINSIC_BF16_AVX512_BF16_BRGEMM_HPP
#define MATMUL_AI_BRGEMM_INTRINSIC_BF16_AVX512_BF16_BRGEMM_HPP

#include "lowoha_operators/matmul/matmul_ai/common/gemm_descriptor.hpp"
#include "lowoha_operators/matmul/matmul_ai/common/cost_model.hpp"
#include "lowoha_operators/matmul/lowoha_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ai {

/// BF16 BRGEMM entry point.
///
/// Processes BF16 src × BF16 weight → BF16 or FP32 dst using AVX-512 BF16
/// VNNI dpbf16ps instructions. The microkernel accumulates ALL K in FP32
/// registers (no C store/reload between K-blocks), then stores once.
///
/// B must be in VNNI-packed format (k-pairs interleaved).
/// Uses BF16PrepackedWeightCache for weight caching.
///
/// Called from ai_matmul_execute() when kernel == ai_brgemm && is_bf16.
void bf16_brgemm_execute(
    const GemmDescriptor &desc,
    const UarchParams &uarch,
    const void *src, const void *weight, void *dst,
    const void *bias, matmul_params &params);

} // namespace ai
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_AI_BRGEMM_INTRINSIC_BF16_AVX512_BF16_BRGEMM_HPP
