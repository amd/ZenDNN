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

#ifndef _RMSNORM_AVX512_FP16_KERNEL_HPP
#define _RMSNORM_AVX512_FP16_KERNEL_HPP

#include "lowoha_operators/normalization/lowoha_normalization_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace normalization {

// ---------------------------------------------------------------------------
// AVX512-FP16 (native __m512h) RMS Normalization (plain and fused-add).
//
//   Plain RMS:
//     rms    = sqrt( (1/N) * Σ input[b,i]² + eps )
//     y[b,i] = gamma[i] * input[b,i] / rms
//
//   Fused Add + RMS:
//     residual[b,i] += input[b,i]                  (in-place, FP16)
//     rms    = sqrt( (1/N) * Σ residual[b,i]² + eps )
//     y[b,i] = gamma[i] * residual[b,i] / rms
//
// Supported dtype combinations (gated upstream by the eligibility predicate
// in lowoha_normalization.cpp; this entry point returns
// status_t::unimplemented for any other combo so the caller can fall back
// to the FP32-accumulating rms_norm_avx512):
//
//   Plain RMS  (params.norm_type == RMS_NORM):
//     src_dt ∈ {f16, f32}, dst_dt ∈ {f16, f32}, at least one is f16
//     gamma_dt ∈ {f16, f32}
//     6 templated specializations (3 (src,dst) combos × 2 gamma_dt).
//     Conversions at the load boundary stay in __m512h; the FMA chain
//     never widens.
//
//   Fused Add + RMS (params.norm_type == FUSED_ADD_RMS_NORM):
//     src_dt == dst_dt == gamma_dt == f16 (strict).
//     Residual aliases src in-place storage and is read-modify-written;
//     mixed-dtype fused-add is rejected upstream by the dispatch.
//
// Dispatch MUST gate this entry point on
// zendnnl_platform_info().get_avx512_f16_status() — calling it on a CPU
// without the AVX512-FP16 ISA will SIGILL. On toolchains older than GCC 12,
// this returns status_t::isa_unsupported.
// ---------------------------------------------------------------------------
status_t rms_norm_avx512_fp16(
  const void *input,
  void       *output,
  void       *residual,
  const void *gamma,
  norm_params &params
);

} // namespace normalization
} // namespace lowoha
} // namespace zendnnl

#endif // _RMSNORM_AVX512_FP16_KERNEL_HPP
