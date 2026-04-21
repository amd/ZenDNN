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

#ifndef _LAYERNORM_AVX512_KERNEL_HPP
#define _LAYERNORM_AVX512_KERNEL_HPP

#include "lowoha_operators/normalization/lowoha_normalization_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace normalization {

// ---------------------------------------------------------------------------
// AVX-512 optimized Layer Normalization.
//
//   mean     = (1/N) * Σ input[b,i]
//   var      = (1/N) * Σ input[b,i]² - mean²
//   inv_std  = 1 / sqrt(var + eps)
//   y[b,i]  = gamma[i] * (input[b,i] - mean) * inv_std + beta[i]
//
// Both mean and sum-of-squares are computed in a single pass over the input,
// and the variance is derived via  var = E[x²] - E[x]²  (clamped to ≥ 0).
//
// @param input      Input tensor (contiguous, row-major)
// @param output     Output tensor
// @param gamma      Scale parameter (FP32 or BF16, nullptr if !use_scale)
// @param beta       Shift parameter (FP32 or BF16, nullptr if !use_shift)
// @param params     Normalization parameters
//
// @return status_t::success on successful execution
// ---------------------------------------------------------------------------
status_t layer_norm_avx512(
  const void *input,
  void       *output,
  const void *gamma,
  const void *beta,
  norm_params &params
);

} // namespace normalization
} // namespace lowoha
} // namespace zendnnl

#endif // _LAYERNORM_AVX512_KERNEL_HPP
