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

#include "lowoha_normalization_example.hpp"
#include <cmath>
#include <numeric>

namespace zendnnl {
namespace examples {

// ============================================================================
// LayerNorm – 2D tensor  [batch, hidden_dim]
// ============================================================================
int run_lowoha_layer_norm_fp32_example() {
  try {
    log_info("** LOWOHA LayerNorm FP32 2D example");

    // Dimensions
    const uint64_t batch      = 4;
    const uint64_t hidden_dim = 8;
    const uint64_t total_size = batch * hidden_dim;

    std::vector<float> input(total_size);
    for (uint64_t i = 0; i < total_size; ++i) {
      input[i] = static_cast<float>(i % hidden_dim) * 0.1f + 0.5f;
    }

    // Gamma (scale) and Beta (shift) – initialized to 1 and 0 respectively
    std::vector<float> gamma(hidden_dim, 1.0f);
    std::vector<float> beta(hidden_dim, 0.0f);

    // Output buffer
    std::vector<float> output(total_size, 0.0f);

    // Setup normalization parameters
    norm_params params;
    params.batch      = batch;
    params.norm_size  = hidden_dim;
    params.norm_type  = norm_type_t::LAYER_NORM;
    params.src_dt     = data_type_t::f32;
    params.dst_dt     = data_type_t::f32;
    params.epsilon    = 1e-5f;
    params.use_scale  = true;
    params.use_shift  = true;

    // Execute
    status_t status = normalization_direct(
                        input.data(), output.data(),
                        gamma.data(), beta.data(),
                        /*running_mean=*/nullptr, /*running_var=*/nullptr,
                        /*residual=*/nullptr, params);

    if (status != status_t::success) {
      log_error("LayerNorm FP32 2D: Execution failed");
      return NOT_OK;
    }
  }
  catch (const exception_t &ex) {
    log_error("Exception: ", ex.what());
    return NOT_OK;
  }
  return OK;
}

// ============================================================================
// LayerNorm – 3D tensor  [batch, seq_len, hidden_dim]
// ============================================================================
int run_lowoha_layer_norm_3d_fp32_example() {
  try {
    log_info("** LOWOHA LayerNorm FP32 3D example");

    // Transformer-like dimensions
    const uint64_t batch      = 2;
    const uint64_t seq_len    = 4;
    const uint64_t hidden_dim = 16;
    const uint64_t total_size = batch * seq_len * hidden_dim;

    // Input – random-ish values
    std::vector<float> input(total_size);
    for (uint64_t i = 0; i < total_size; ++i) {
      input[i] = std::sin(static_cast<float>(i) * 0.3f);
    }

    // Gamma = 1, Beta = 0  (identity affine)
    std::vector<float> gamma(hidden_dim, 1.0f);
    std::vector<float> beta(hidden_dim, 0.0f);

    std::vector<float> output(total_size, 0.0f);

    // Setup: normalize over last 1 dim (hidden_dim)
    norm_params params;
    params.batch      = batch * seq_len;
    params.norm_size  = hidden_dim;
    params.norm_type  = norm_type_t::LAYER_NORM;
    params.src_dt     = data_type_t::f32;
    params.dst_dt     = data_type_t::f32;
    params.epsilon    = 1e-5f;
    params.use_scale  = true;
    params.use_shift  = true;

    status_t status = normalization_direct(
                        input.data(), output.data(),
                        gamma.data(), beta.data(),
                        nullptr, nullptr,
                        /*residual=*/nullptr, params);

    if (status != status_t::success) {
      log_error("LayerNorm FP32 3D: Execution failed");
      return NOT_OK;
    }
  }
  catch (const exception_t &ex) {
    log_error("Exception: ", ex.what());
    return NOT_OK;
  }
  return OK;
}

// ============================================================================
// BatchNorm – Inference mode with pre-computed running mean/var
// ============================================================================
int run_lowoha_batch_norm_fp32_example() {
  try {
    log_info("** LOWOHA BatchNorm FP32 (inference) example");

    const uint64_t N = 2;
    const uint64_t C = 3;
    const uint64_t H = 4;
    const uint64_t W = 4;
    const uint64_t total_size = N * C * H * W;

    // Input
    std::vector<float> input(total_size);
    for (uint64_t i = 0; i < total_size; ++i) {
      input[i] = static_cast<float>(i) * 0.01f;
    }

    // Gamma and Beta
    std::vector<float> gamma = {1.0f, 0.5f, 2.0f};  // per-channel scale
    std::vector<float> beta  = {0.0f, 0.1f, -0.5f};  // per-channel shift

    // Pre-computed running statistics (e.g., from training)
    std::vector<float> running_mean = {0.5f, 1.0f, 1.5f};
    std::vector<float> running_var  = {0.25f, 0.5f, 1.0f};

    std::vector<float> output(total_size, 0.0f);

    // Setup
    norm_params params;
    params.batch        = N;
    params.num_channels = C;
    params.norm_size    = H * W;
    params.norm_type    = norm_type_t::BATCH_NORM;
    params.src_dt       = data_type_t::f32;
    params.dst_dt       = data_type_t::f32;
    params.epsilon      = 1e-5f;
    params.use_scale    = true;
    params.use_shift    = true;

    status_t status = normalization_direct(
                        input.data(), output.data(),
                        gamma.data(), beta.data(),
                        running_mean.data(), running_var.data(),
                        /*residual=*/nullptr, params);

    if (status != status_t::success) {
      log_error("BatchNorm FP32 (inference): Execution failed");
      return NOT_OK;
    }
  }
  catch (const exception_t &ex) {
    log_error("Exception: ", ex.what());
    return NOT_OK;
  }
  return OK;
}

// ============================================================================
// RMSNorm – 2D tensor  [batch, hidden_dim]
// ============================================================================
int run_lowoha_rms_norm_fp32_example() {
  try {
    log_info("** LOWOHA RMSNorm FP32 example");

    const uint64_t batch      = 4;
    const uint64_t hidden_dim = 8;
    const uint64_t total_size = batch * hidden_dim;

    // Input data
    std::vector<float> input(total_size);
    for (uint64_t i = 0; i < total_size; ++i) {
      input[i] = static_cast<float>(i % hidden_dim) * 0.2f + 1.0f;
    }

    // Gamma only (RMSNorm has no beta)
    std::vector<float> gamma(hidden_dim, 1.0f);

    std::vector<float> output(total_size, 0.0f);

    // Setup
    norm_params params;
    params.batch      = batch;
    params.norm_size  = hidden_dim;
    params.norm_type  = norm_type_t::RMS_NORM;
    params.src_dt     = data_type_t::f32;
    params.dst_dt     = data_type_t::f32;
    params.epsilon    = 1e-6f;
    params.use_scale  = true;

    status_t status = normalization_direct(
                        input.data(), output.data(),
                        gamma.data(), /*beta=*/nullptr,
                        nullptr, nullptr,
                        /*residual=*/nullptr, params);

    if (status != status_t::success) {
      log_error("RMSNorm FP32: Execution failed");
      return NOT_OK;
    }
  }
  catch (const exception_t &ex) {
    log_error("Exception: ", ex.what());
    return NOT_OK;
  }
  return OK;
}

// ============================================================================
// FusedAddRMSNorm – 2D tensor  [batch, hidden_dim]
// ============================================================================
int run_lowoha_fused_add_rms_norm_fp32_example() {
  try {
    log_info("** LOWOHA FusedAddRMSNorm FP32 example");

    const uint64_t batch      = 4;
    const uint64_t hidden_dim = 8;
    const uint64_t total_size = batch * hidden_dim;

    // Input (current layer output, to be added to the residual stream)
    std::vector<float> input(total_size);
    for (uint64_t i = 0; i < total_size; ++i) {
      input[i] = static_cast<float>(i % hidden_dim) * 0.2f + 1.0f;
    }

    // Residual buffer – represents the running residual stream; modified in-place
    std::vector<float> residual(total_size);
    for (uint64_t i = 0; i < total_size; ++i) {
      residual[i] = static_cast<float>(i % hidden_dim) * 0.1f + 0.5f;
    }

    std::vector<float> gamma(hidden_dim, 1.0f);
    std::vector<float> output(total_size, 0.0f);

    norm_params params;
    params.batch      = batch;
    params.norm_size  = hidden_dim;
    params.norm_type  = norm_type_t::FUSED_ADD_RMS_NORM;
    params.src_dt     = data_type_t::f32;
    params.dst_dt     = data_type_t::f32;
    params.epsilon    = 1e-6f;
    params.use_scale  = true;

    status_t status = normalization_direct(
                        input.data(), output.data(),
                        gamma.data(), /*beta=*/nullptr,
                        /*running_mean=*/nullptr, /*running_var=*/nullptr,
                        residual.data(), params);

    if (status != status_t::success) {
      log_error("FusedAddRMSNorm FP32: Execution failed");
      return NOT_OK;
    }
  }
  catch (const exception_t &ex) {
    log_error("Exception: ", ex.what());
    return NOT_OK;
  }
  return OK;
}

} // namespace examples
} // namespace zendnnl

