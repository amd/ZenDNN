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

#ifndef _LOWOHA_NORMALIZATION_EXAMPLE_HPP_
#define _LOWOHA_NORMALIZATION_EXAMPLE_HPP_

#include "zendnnl.hpp"
#include <iostream>
#include <vector>

#define  OK          (0)
#define  NOT_OK      (1)

namespace zendnnl {
namespace examples {

using namespace zendnnl::lowoha::normalization;

/** @fn run_lowoha_layer_norm_fp32_example
 *  @brief Demonstrates LayerNorm on FP32 inputs.
 *
 *  LayerNorm normalizes each sample across the last `norm_ndims` dimensions.
 *  Typical in transformer models for normalizing hidden states.
 *
 *  Configuration:
 *    - Input:  FP32 [batch, hidden_dim]
 *    - Gamma:  FP32 [hidden_dim]
 *    - Beta:   FP32 [hidden_dim]
 *    - Output: FP32 [batch, hidden_dim]
 *    - Normalizes along last axis (norm_ndims=1)
 */
int run_lowoha_layer_norm_fp32_example();

/** @fn run_lowoha_layer_norm_3d_fp32_example
 *  @brief Demonstrates LayerNorm on a 3D tensor (e.g., [batch, seq_len, hidden_dim]).
 *
 *  A more realistic transformer scenario where the input is a sequence of
 *  token embeddings and normalization is applied per-token across hidden_dim.
 *
 *  Configuration:
 *    - Input:  FP32 [batch, seq_len, hidden_dim]
 *    - Gamma:  FP32 [hidden_dim]
 *    - Beta:   FP32 [hidden_dim]
 *    - Output: FP32 [batch, seq_len, hidden_dim]
 *    - Normalizes along last axis (norm_ndims=1)
 */
int run_lowoha_layer_norm_3d_fp32_example();

/** @fn run_lowoha_batch_norm_fp32_example
 *  @brief Demonstrates BatchNorm in inference mode with running statistics.
 *
 *  In inference mode, pre-computed running mean/variance are used instead of
 *  computing batch statistics.
 *
 *  Configuration:
 *    - Input:        FP32 [N, C, H, W]
 *    - Gamma:        FP32 [C]
 *    - Beta:         FP32 [C]
 *    - Running mean: FP32 [C]  (pre-computed)
 *    - Running var:  FP32 [C]  (pre-computed)
 *    - Output:       FP32 [N, C, H, W]
 */
int run_lowoha_batch_norm_fp32_example();

/** @fn run_lowoha_rms_norm_fp32_example
 *  @brief Demonstrates RMSNorm on FP32 inputs.
 *
 *  RMSNorm is a simplified variant of LayerNorm that omits mean subtraction
 *  and the beta (shift) parameter.
 *
 *  Configuration:
 *    - Input:  FP32 [batch, hidden_dim]
 *    - Gamma:  FP32 [hidden_dim]
 *    - Output: FP32 [batch, hidden_dim]
 *    - Normalizes along last axis (norm_ndims=1)
 */
int run_lowoha_rms_norm_fp32_example();

/** @fn run_lowoha_fused_add_rms_norm_fp32_example
 *  @brief Demonstrates FusedAddRMSNorm on FP32 inputs.
 *
 *  Fuses a residual addition with RMSNorm in a single pass, as used in
 *  transformer decoder blocks (e.g. LLaMA).  The residual buffer is updated
 *  in-place (residual += input), then RMSNorm is applied over the updated
 *  residual to produce the output.
 *
 *  Formula:
 *    residual[i] += input[i]
 *    rms = sqrt( mean(residual^2) + eps )
 *    output[i] = gamma[i] * residual[i] / rms
 *
 *  Configuration:
 *    - Input:    FP32 [batch, hidden_dim]
 *    - Residual: FP32 [batch, hidden_dim]  (modified in-place)
 *    - Gamma:    FP32 [hidden_dim]
 *    - Output:   FP32 [batch, hidden_dim]
 *    - Normalizes along last axis (norm_ndims=1)
 */
int run_lowoha_fused_add_rms_norm_fp32_example();

} // examples
} // zendnnl

#endif

