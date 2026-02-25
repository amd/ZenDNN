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

#ifndef _LOWOHA_NORMALIZATION_HPP
#define _LOWOHA_NORMALIZATION_HPP

#include <cmath>
#include <cstring>
#include "lowoha_operators/normalization/lowoha_normalization_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace normalization {

/**
 * @brief Execute normalization (LayerNorm / RMSNorm / BatchNorm / FusedAddRMSNorm)
 *        via the unified LOWOHA low-overhead API.
 *
 * This is the user API for all four normalization
 * variants.  The caller populates @c params with the tensor shape, norm type,
 * data types, and other configuration; this function internally derives the
 * flattened dimensions (batch, norm_size, num_channels), validates inputs,
 * and dispatches to the appropriate kernel.
 *
 * Required fields in @c params before calling:
 *   - norm_type   : LAYER_NORM, RMS_NORM, BATCH_NORM, or FUSED_ADD_RMS_NORM
 *   - shape       : tensor dimensions (e.g. params.shape = {batch, hidden_dim})
 *   - norm_ndims  : number of trailing dims to normalize (LayerNorm/RMSNorm/FusedAddRMSNorm)
 *   - src_dt      : source data type (f32 or bf16)
 *   - dst_dt      : destination data type (f32 or bf16)
 *   - epsilon     : numerical-stability constant (e.g. 1e-5f)
 *   - use_scale   : whether gamma is applied
 *   - use_shift   : whether beta is applied (ignored by RMSNorm/FusedAddRMSNorm)
 *   - algorithm   : norm_algo_t::none for auto-select, or a specific backend
 *
 * Inference formulas:
 *
 *   LayerNorm:        y = gamma * (x - mean) / sqrt(var + eps) + beta
 *                     (mean, var computed on-the-fly from the input per sample)
 *
 *   RMSNorm:          y = gamma * x / sqrt(mean(x^2) + eps)
 *                     (RMS computed on-the-fly from the input per sample)
 *
 *   BatchNorm:        y = gamma * (x - running_mean) / sqrt(running_var + eps) + beta
 *                     (uses pre-computed running statistics from training)
 *
 *   FusedAddRMSNorm:  residual[i] += input[i]          (in-place)
 *                     y = gamma * residual / sqrt(mean(residual^2) + eps)
 *                     (fuses residual addition with RMSNorm in a single call)
 *
 * @param input             Pointer to input tensor data (read-only).
 *                          Element type must match params.src_dt.
 * @param output            Pointer to output tensor data (same shape as input).
 *                          Element type must match params.dst_dt.
 * @param gamma             Pointer to scale (gamma) parameters (read-only)
 *                          - LayerNorm / RMSNorm / FusedAddRMSNorm: shape = [norm_size]
 *                          - BatchNorm:           shape = [num_channels]
 *                          May be nullptr if params.use_scale == false.
 * @param beta              Pointer to shift (beta) parameters (read-only)
 *                          - LayerNorm:  shape = [norm_size]
 *                          - BatchNorm:  shape = [num_channels]
 *                          - RMSNorm / FusedAddRMSNorm: unused (may be nullptr)
 *                          May be nullptr if params.use_shift == false.
 * @param running_mean      (BatchNorm only) Pre-computed per-channel mean from
 *                          training, shape = [num_channels]. Required for BatchNorm.
 *                          nullptr for all other norm types.
 * @param running_var       (BatchNorm only) Pre-computed per-channel variance from
 *                          training, shape = [num_channels]. Required for BatchNorm.
 *                          nullptr for all other norm types.
 * @param residual          (FusedAddRMSNorm only) Residual buffer that is updated
 *                          in-place: on return, residual[i] = old_residual[i] + input[i].
 *                          The normalized output is computed from this updated residual.
 *                          Must have the same shape and element type as the input
 *                          (i.e. params.src_dt). Required for FUSED_ADD_RMS_NORM.
 *                          nullptr for all other norm types.
 * @param params            Normalization parameters (type, dims, data types, etc.)
 *                          Must have shape and norm_type populated.
 *
 * @return status_t::success on successful execution, status_t::failure otherwise
 */
status_t normalization_direct(
  const void *input,
  void *output,
  const void *gamma,
  const void *beta,
  const void *running_mean,
  const void *running_var,
  void       *residual,
  norm_params &params
);

/**
 * @brief Kernel dispatcher – selects and invokes the appropriate backend
 *        for the requested normalization type.
 *
 * @param input             Input tensor data (read-only)
 * @param output            Output tensor data
 * @param gamma             Scale parameter data (read-only, may be nullptr)
 * @param beta              Shift parameter data (read-only, may be nullptr)
 * @param running_mean      Pre-computed running mean (BatchNorm only, read-only)
 * @param running_var       Pre-computed running variance (BatchNorm only, read-only)
 * @param residual          Residual buffer (FusedAddRMSNorm only), modified in-place.
 *                          Same shape and element type as input. nullptr otherwise.
 * @param params            Normalization parameters
 */
status_t normalization_kernel_wrapper(
  const void *input,
  void *output,
  const void *gamma,
  const void *beta,
  const void *running_mean,
  const void *running_var,
  void       *residual,
  norm_params &params
);

} // namespace normalization
} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_NORMALIZATION_HPP

