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

#ifndef MATMUL_AI_INTRINSIC_AVX512_BF16_GEMM_HPP
#define MATMUL_AI_INTRINSIC_AVX512_BF16_GEMM_HPP

#include "lowoha_operators/matmul/matmul_ai/common/gemm_descriptor.hpp"
#include "lowoha_operators/matmul/matmul_ai/common/cost_model.hpp"
#include "lowoha_operators/matmul/lowoha_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ai {

/// High-level BF16 GEMM entry point.
///
/// Input: BF16 src (A) x BF16 weight (B).
/// Accumulation: FP32 (in registers, for accuracy).
/// Output: FP32 or BF16 (based on desc.dst_dt).
///
/// Uses AMD Zen4/Zen5 AVX-512 BF16 extension (_mm512_dpbf16_ps)
/// which processes 2 BF16 pairs per instruction, effectively
/// doubling K-dimension throughput compared to FP32 FMA.
///
/// B matrix is prepacked in VNNI format: consecutive K-pairs are
/// interleaved at the N dimension for optimal dpbf16 feeding.
///
/// Owns the full pipeline: plan caching, B prepacking (VNNI),
/// and the 5-loop BLIS-style thread loop.
void bf16_gemm_execute(
  const GemmDescriptor &desc,
  const UarchParams &uarch,
  const void *src, const void *weight, void *dst,
  const void *bias, matmul_params &params);

} // namespace ai
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_AI_INTRINSIC_AVX512_BF16_GEMM_HPP
