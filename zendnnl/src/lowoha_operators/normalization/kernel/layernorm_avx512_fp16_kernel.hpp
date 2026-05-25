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

#ifndef _LAYERNORM_AVX512_FP16_KERNEL_HPP
#define _LAYERNORM_AVX512_FP16_KERNEL_HPP

#include "lowoha_operators/normalization/lowoha_normalization_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace normalization {

// ---------------------------------------------------------------------------
// AVX512-FP16 (native __m512h) Layer Normalization.
//
//   mean    = (1/N) * Σ input[b,i]
//   var     = (1/N) * Σ input[b,i]² - mean²       (clamped to ≥ 0)
//   inv_std = 1 / sqrt(var + eps)
//   y[b,i]  = gamma[i] * (input[b,i] - mean) * inv_std + beta[i]
//
// Native FP16 path: accumulators, mul, sub, and FMA all run in __m512h
// (32 lanes/iter); the row-wide horizontal sum is widened to FP32 once to
// avoid precision loss across long rows.
//
// Supported dtype combinations (gated upstream by the eligibility predicate
// in lowoha_normalization.cpp; this entry point returns
// status_t::unimplemented for any other combo so the caller can fall back
// to the FP32-accumulating layer_norm_avx512):
//
//   src_dt ∈ {f16, f32}, dst_dt ∈ {f16, f32}, at least one is f16
//   gamma_dt ∈ {f16, f32}
//   beta_dt  ∈ {f16, f32} (only checked when params.use_shift = true)
//
// 12 templated specializations (3 (src,dst) combos × 2 gamma_dt × 2 beta_dt).
// Conversions at the load boundary stay in __m512h; the FMA chain never
// widens.
//
// Dispatch MUST gate this entry point on
// zendnnl_platform_info().get_avx512_f16_status() — calling it on a CPU
// without the AVX512-FP16 ISA will SIGILL. On toolchains older than GCC 12,
// this returns status_t::isa_unsupported.
// ---------------------------------------------------------------------------
status_t layer_norm_avx512_fp16(
  const void *input,
  void       *output,
  const void *gamma,
  const void *beta,
  norm_params &params
);

} // namespace normalization
} // namespace lowoha
} // namespace zendnnl

#endif // _LAYERNORM_AVX512_FP16_KERNEL_HPP
