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

#ifndef _LOWOHA_NORMALIZATION_REFERENCE_KERNEL_HPP
#define _LOWOHA_NORMALIZATION_REFERENCE_KERNEL_HPP

#include "lowoha_operators/normalization/lowoha_normalization_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace normalization {

/**
 * @brief Reference implementation wrapper for normalization
 *
 * Dispatches to the appropriate implementation based on params.src_dt / params.dst_dt,
 * and handles LayerNorm / RMSNorm / BatchNorm / FusedAddRMSNorm via params.norm_type.
 *
 * @param input             Input tensor (read-only). Same element type as params.src_dt.
 * @param output            Output tensor. Same shape as input, element type params.dst_dt.
 * @param gamma             Scale parameter (read-only, may be nullptr if !use_scale).
 *                          Shape depends on norm_type:
 *                          - LayerNorm / RMSNorm / FusedAddRMSNorm: [norm_size]
 *                          - BatchNorm: [num_channels]
 * @param beta              Shift parameter (read-only, may be nullptr if !use_shift).
 *                          Unused by RMSNorm and FusedAddRMSNorm.
 * @param running_mean      Pre-computed running mean (read-only, required for BatchNorm,
 *                          nullptr otherwise).
 * @param running_var       Pre-computed running variance (read-only, required for BatchNorm,
 *                          nullptr otherwise).
 * @param residual          Residual buffer, required only for FUSED_ADD_RMS_NORM,
 *                          nullptr for all other norm types.
 *                          - Must have the same shape and element type as the input
 *                            (i.e. params.src_dt).
 *                          - Modified in-place: on return, residual[i] = old_residual[i] + input[i].
 *                          - The normalized output is computed from this updated residual.
 * @param params            Normalization parameters (type, shape, data types, etc.)
 *
 * @return status_t::success on successful execution, status_t::failure otherwise
 */
status_t normalization_reference_wrapper(
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

#endif // _LOWOHA_NORMALIZATION_REFERENCE_KERNEL_HPP

