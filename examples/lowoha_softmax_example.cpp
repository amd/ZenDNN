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

#include "lowoha_softmax_example.hpp"

namespace zendnnl {
namespace examples {

int run_lowoha_softmax_fp32_test() {
  try {
    // Input tensor dimensions [batch, axis_dim]
    uint64_t batch = 2;
    uint64_t axis_dim = 5;
    uint64_t total_size = batch * axis_dim;
    // Input data (row-major): 2 batches of 5 elements each
    std::vector<float> input = {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f,     // batch 0
      -1.0f, 0.0f, 1.0f, 2.0f, 3.0f     // batch 1
    };

    std::vector<float> output(total_size, 0.0f);

    // Setup softmax parameters
    softmax_params params;

    // Initialize shape: 2D tensor [batch, axis_dim]
    uint64_t shape[] = {batch, axis_dim};
    status_t setup_status = setup_softmax_shape(params, shape, 2, -1);
    if (setup_status != status_t::success) {
      log_error("Failed to setup softmax shape");
      return NOT_OK;
    }

    // Set data types
    params.src_dt = data_type_t::f32;
    params.dst_dt = data_type_t::f32;

    // Select softmax variant: plain softmax (log_softmax=false, softmin=false)
    params.log_softmax = false;
    params.softmin     = false;

    // Set algorithm (auto-select)
    params.algorithm = softmax_algo_t::none;
    // Call the low-overhead softmax API
    status_t status = softmax_direct(
                        input.data(),
                        output.data(),
                        params);

    if (status != status_t::success) {
      log_error("LOWOHA Softmax: Execution failed");
      return NOT_OK;
    }
  }
  catch (const exception_t &ex) {
    log_error("Exception: ", ex.what());
    return NOT_OK;
  }

  return OK;
}

int run_lowoha_softmax_bf16_test() {
  try {

    // Input tensor dimensions [batch, axis_dim]
    uint64_t batch = 4;
    uint64_t axis_dim = 128;  // Typical attention dimension
    uint64_t total_size = batch * axis_dim;

    // Allocate BF16 tensors (stored as uint16_t)
    std::vector<uint16_t> input(total_size);
    std::vector<uint16_t> output(total_size, 0);

    // Initialize input with BF16 representation of small values
    // Using BF16 representation of 1.0f = 0x3F80
    for (uint64_t i = 0; i < total_size; ++i) {
      // Vary the values slightly
      input[i] = 0x3F80 + static_cast<uint16_t>(i % 16);
    }

    // Setup softmax parameters
    softmax_params params;

    // Initialize shape: 2D tensor [batch, axis_dim]
    uint64_t shape[] = {batch, axis_dim};
    status_t setup_status = setup_softmax_shape(params, shape, 2, -1);
    if (setup_status != status_t::success) {
      log_error("Failed to setup softmax shape");
      return NOT_OK;
    }

    // Set data types
    params.src_dt = data_type_t::bf16;
    params.dst_dt = data_type_t::bf16;

    // Select softmax variant: plain softmax (log_softmax=false, softmin=false)
    params.log_softmax = false;
    params.softmin     = false;

    // Set algorithm (auto-select)
    params.algorithm = softmax_algo_t::none;
    // Call the low-overhead softmax API
    status_t status = softmax_direct(
                        input.data(),
                        output.data(),
                        params);

    if (status != status_t::success) {
      log_error("LOWOHA Softmax: Execution failed");
      return NOT_OK;
    }

  }
  catch (const exception_t &ex) {
    log_error("Exception: ", ex.what());
    return NOT_OK;
  }

  return OK;
}

int run_lowoha_softmin_fp32_test() {
  try {
    // Input tensor dimensions [batch, axis_dim]
    uint64_t batch = 2;
    uint64_t axis_dim = 5;
    uint64_t total_size = batch * axis_dim;
    // Input data (row-major): 2 batches of 5 elements each
    // For softmin, the smallest value receives the largest probability.
    std::vector<float> input = {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f,     // batch 0: 1.0 gets highest probability
      -1.0f, 0.0f, 1.0f, 2.0f, 3.0f     // batch 1: -1.0 gets highest probability
    };

    std::vector<float> output(total_size, 0.0f);

    // Setup softmax parameters
    softmax_params params;

    // Initialize shape: 2D tensor [batch, axis_dim]
    uint64_t shape[] = {batch, axis_dim};
    status_t setup_status = setup_softmax_shape(params, shape, 2, -1);
    if (setup_status != status_t::success) {
      log_error("Failed to setup softmax shape");
      return NOT_OK;
    }

    // Set data types
    params.src_dt = data_type_t::f32;
    params.dst_dt = data_type_t::f32;

    // Select softmax variant: softmin (softmin=true, log_softmax=false)
    // Set both log_softmax=true and softmin=true to compute log-softmin.
    params.log_softmax = false;
    params.softmin     = true;

    // Set algorithm (auto-select)
    params.algorithm = softmax_algo_t::none;
    // Call the low-overhead softmax API (internally computes softmax(-input))
    status_t status = softmax_direct(
                        input.data(),
                        output.data(),
                        params);

    if (status != status_t::success) {
      log_error("LOWOHA Softmin: Execution failed");
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
