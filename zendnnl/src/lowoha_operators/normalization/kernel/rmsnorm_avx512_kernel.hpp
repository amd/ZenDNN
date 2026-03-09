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

#ifndef _RMSNORM_AVX512_KERNEL_HPP
#define _RMSNORM_AVX512_KERNEL_HPP

#include "lowoha_operators/normalization/lowoha_normalization_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace normalization {

// ---------------------------------------------------------------------------
// AVX-512 optimized RMS Normalization (plain and fused-add variants).
//
// Supports both RMS_NORM and FUSED_ADD_RMS_NORM:
//
//   Plain RMS:
//     rms    = sqrt( (1/N) * Σ input[b,i]² + eps )
//     y[b,i] = gamma[i] * input[b,i] / rms
//
//   Fused Add + RMS:
//     residual[b,i] += input[b,i]
//     rms    = sqrt( (1/N) * Σ residual[b,i]² + eps )
//     y[b,i] = gamma[i] * residual[b,i] / rms
//
// @param input      Input tensor (contiguous, row-major)
// @param output     Output tensor
// @param residual   Residual stream (read-write for FUSED_ADD_RMS_NORM,
//                   nullptr for RMS_NORM)
// @param gamma      Scale parameter (FP32, nullptr if !use_scale)
// @param params     Normalization parameters
//
// @return status_t::success on successful execution
// ---------------------------------------------------------------------------
status_t rms_norm_avx512(
  const void *input,
  void       *output,
  void       *residual,
  const void *gamma,
  norm_params &params
);

} // namespace normalization
} // namespace lowoha
} // namespace zendnnl

#endif // _RMSNORM_AVX512_KERNEL_HPP
