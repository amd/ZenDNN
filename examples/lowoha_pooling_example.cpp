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

#include "lowoha_pooling_example.hpp"

namespace zendnnl {
namespace examples {

int run_lowoha_maxpool_fp32_test() {
  try {

    // Input dimensions [N, H, W, C] - NHWC format
    uint64_t N = 1;   // batch size
    uint64_t H = 4;   // input height
    uint64_t W = 4;   // input width
    uint64_t C = 2;   // channels

    // Pooling parameters
    uint64_t kernel_h = 2;
    uint64_t kernel_w = 2;
    uint32_t stride_h = 2;
    uint32_t stride_w = 2;
    uint32_t pad_top = 0, pad_left = 0;
    uint32_t pad_bottom = 0, pad_right = 0;

    // Calculate output dimensions
    uint64_t H_out = (H + pad_top + pad_bottom - kernel_h) / stride_h + 1;
    uint64_t W_out = (W + pad_left + pad_right - kernel_w) / stride_w + 1;

    // Create input tensor (NHWC format)
    uint64_t input_size = N * H * W * C;
    std::vector<float> input(input_size);

    // Initialize with simple pattern for easy verification
    for (uint64_t n = 0; n < N; ++n) {
      for (uint64_t h = 0; h < H; ++h) {
        for (uint64_t w = 0; w < W; ++w) {
          for (uint64_t c = 0; c < C; ++c) {
            uint64_t idx = n * (H * W * C) + h * (W * C) + w * C + c;
            input[idx] = static_cast<float>(h * W + w + c * 0.1f);
          }
        }
      }
    }

    // Create output tensor
    uint64_t output_size = N * H_out * W_out * C;
    std::vector<float> output(output_size, 0.0f);

    // Setup pooling parameters
    pool_params params;

    // Set dimensions
    params.dims.batch = N;
    params.dims.in_height = H;
    params.dims.in_width = W;
    params.dims.channels = C;
    params.dims.kernel_height = kernel_h;
    params.dims.kernel_width = kernel_w;
    params.dims.out_height = H_out;
    params.dims.out_width = W_out;

    // Set strides and padding
    params.stride_h = stride_h;
    params.stride_w = stride_w;
    params.pad_top = pad_top;
    params.pad_left = pad_left;
    params.pad_bottom = pad_bottom;
    params.pad_right = pad_right;

    // Set pooling type (max pooling)
    params.is_max_pooling = true;

    // Set data types
    params.dtypes.src = data_type_t::f32;
    params.dtypes.dst = data_type_t::f32;

    // Set algorithm (auto-select)
    params.algo = pooling_algo_t::none;

    // Call the low-overhead pooling API
    status_t status = pooling_direct(
                        input.data(),
                        output.data(),
                        params);

    if (status != status_t::success) {
      log_error("LOWOHA Max Pooling: Execution failed");
      return NOT_OK;
    }

  }
  catch (const exception_t &ex) {
    log_error("Exception: ", ex.what());
    return NOT_OK;
  }

  return OK;
}

int run_lowoha_avgpool_fp32_test() {
  try {

    // Input dimensions [N, H, W, C] - NHWC format
    uint64_t N = 2;   // batch size
    uint64_t H = 8;   // input height
    uint64_t W = 8;   // input width
    uint64_t C = 3;   // channels

    // Pooling parameters
    uint64_t kernel_h = 3;
    uint64_t kernel_w = 3;
    uint32_t stride_h = 2;
    uint32_t stride_w = 2;
    uint32_t pad_top = 1, pad_left = 1;
    uint32_t pad_bottom = 1, pad_right = 1;

    // Calculate output dimensions
    uint64_t H_out = (H + pad_top + pad_bottom - kernel_h) / stride_h + 1;
    uint64_t W_out = (W + pad_left + pad_right - kernel_w) / stride_w + 1;

    // Create input tensor (NHWC format)
    uint64_t input_size = N * H * W * C;
    std::vector<float> input(input_size);

    // Initialize with simple pattern
    for (uint64_t i = 0; i < input_size; ++i) {
      input[i] = static_cast<float>((i % 100)) / 10.0f;
    }

    // Create output tensor
    uint64_t output_size = N * H_out * W_out * C;
    std::vector<float> output(output_size, 0.0f);

    // Setup pooling parameters
    pool_params params;

    // Set dimensions
    params.dims.batch = N;
    params.dims.in_height = H;
    params.dims.in_width = W;
    params.dims.channels = C;
    params.dims.kernel_height = kernel_h;
    params.dims.kernel_width = kernel_w;
    params.dims.out_height = H_out;
    params.dims.out_width = W_out;

    // Set strides and padding
    params.stride_h = stride_h;
    params.stride_w = stride_w;
    params.pad_top = pad_top;
    params.pad_left = pad_left;
    params.pad_bottom = pad_bottom;
    params.pad_right = pad_right;

    // Set pooling type (average pooling)
    params.is_max_pooling = false;

    // Set average pooling mode (exclude padding)
    params.avg_mode = avg_pooling_mode_t::exclude_padding;

    // Set data types
    params.dtypes.src = data_type_t::f32;
    params.dtypes.dst = data_type_t::f32;

    // Set algorithm (auto-select)
    params.algo = pooling_algo_t::none;

    // Call the low-overhead pooling API
    status_t status = pooling_direct(
                        input.data(),
                        output.data(),
                        params);

    if (status != status_t::success) {
      log_error("LOWOHA Average Pooling: Execution failed");
      return NOT_OK;
    }

  }
  catch (const exception_t &ex) {
    log_error("Exception: ", ex.what());
    return NOT_OK;
  }

  return OK;
}

int run_lowoha_maxpool_bf16_test() {
  try {

    // Input dimensions [N, H, W, C] - NHWC format
    uint64_t N = 1;    // batch size
    uint64_t H = 16;   // input height
    uint64_t W = 16;   // input width
    uint64_t C = 64;   // channels

    // Pooling parameters
    uint64_t kernel_h = 2;
    uint64_t kernel_w = 2;
    uint32_t stride_h = 2;
    uint32_t stride_w = 2;
    uint32_t pad_top = 0, pad_left = 0;
    uint32_t pad_bottom = 0, pad_right = 0;

    // Calculate output dimensions
    uint64_t H_out = (H + pad_top + pad_bottom - kernel_h) / stride_h + 1;
    uint64_t W_out = (W + pad_left + pad_right - kernel_w) / stride_w + 1;

    // Create input tensor (NHWC format) - BF16 stored as uint16_t
    uint64_t input_size = N * H * W * C;
    std::vector<uint16_t> input(input_size);

    // Initialize with BF16 values
    for (uint64_t i = 0; i < input_size; ++i) {
      // Using BF16 representation (0x3F80 = 1.0f)
      input[i] = 0x3F80 + static_cast<uint16_t>(i % 256);
    }

    // Create output tensor
    uint64_t output_size = N * H_out * W_out * C;
    std::vector<uint16_t> output(output_size, 0);

    // Setup pooling parameters
    pool_params params;

    // Set dimensions
    params.dims.batch = N;
    params.dims.in_height = H;
    params.dims.in_width = W;
    params.dims.channels = C;
    params.dims.kernel_height = kernel_h;
    params.dims.kernel_width = kernel_w;
    params.dims.out_height = H_out;
    params.dims.out_width = W_out;

    // Set strides and padding
    params.stride_h = stride_h;
    params.stride_w = stride_w;
    params.pad_top = pad_top;
    params.pad_left = pad_left;
    params.pad_bottom = pad_bottom;
    params.pad_right = pad_right;

    // Set pooling type (max pooling)
    params.is_max_pooling = true;

    // Set data types
    params.dtypes.src = data_type_t::bf16;
    params.dtypes.dst = data_type_t::bf16;

    // Set algorithm (auto-select)
    params.algo = pooling_algo_t::none;

    // Call the low-overhead pooling API
    status_t status = pooling_direct(
                        input.data(),
                        output.data(),
                        params);

    if (status != status_t::success) {
      log_error("LOWOHA Max Pooling BF16: Execution failed");
      return NOT_OK;
    }

  }
  catch (const exception_t &ex) {
    log_error("Exception: ", ex.what());
    return NOT_OK;
  }

  return OK;
}

int run_lowoha_avgpool_padding_modes_test() {
  try {

    // Input dimensions [N, H, W, C] - NHWC format
    uint64_t N = 1;   // batch size
    uint64_t H = 4;   // input height
    uint64_t W = 4;   // input width
    uint64_t C = 1;   // single channel for simplicity

    // Pooling parameters
    uint64_t kernel_h = 2;
    uint64_t kernel_w = 2;
    uint32_t stride_h = 2;
    uint32_t stride_w = 2;
    uint32_t pad_top = 1, pad_left = 1;
    uint32_t pad_bottom = 1, pad_right = 1;

    // Calculate output dimensions
    uint64_t H_out = (H + pad_top + pad_bottom - kernel_h) / stride_h + 1;
    uint64_t W_out = (W + pad_left + pad_right - kernel_w) / stride_w + 1;

    // Create input tensor (NHWC format)
    uint64_t input_size = N * H * W * C;
    std::vector<float> input(input_size);

    // Initialize with constant value for easy understanding
    for (uint64_t i = 0; i < input_size; ++i) {
      input[i] = 4.0f;
    }

    // Create output tensors for both modes
    uint64_t output_size = N * H_out * W_out * C;
    std::vector<float> output_exclude(output_size, 0.0f);
    std::vector<float> output_include(output_size, 0.0f);

    // Setup pooling parameters - exclude padding mode
    pool_params params_exclude;
    params_exclude.dims.batch = N;
    params_exclude.dims.in_height = H;
    params_exclude.dims.in_width = W;
    params_exclude.dims.channels = C;
    params_exclude.dims.kernel_height = kernel_h;
    params_exclude.dims.kernel_width = kernel_w;
    params_exclude.dims.out_height = H_out;
    params_exclude.dims.out_width = W_out;
    params_exclude.stride_h = stride_h;
    params_exclude.stride_w = stride_w;
    params_exclude.pad_top = pad_top;
    params_exclude.pad_left = pad_left;
    params_exclude.pad_bottom = pad_bottom;
    params_exclude.pad_right = pad_right;
    params_exclude.is_max_pooling = false;
    params_exclude.avg_mode = avg_pooling_mode_t::exclude_padding;
    params_exclude.dtypes.src = data_type_t::f32;
    params_exclude.dtypes.dst = data_type_t::f32;
    params_exclude.algo = pooling_algo_t::none;

    status_t status = pooling_direct(
                        input.data(),
                        output_exclude.data(),
                        params_exclude);

    if (status != status_t::success) {
      log_error("LOWOHA Average Pooling (exclude_padding): Execution failed");
      return NOT_OK;
    }

    // Setup pooling parameters - include padding mode
    pool_params params_include = params_exclude;
    params_include.avg_mode = avg_pooling_mode_t::include_padding;

    status = pooling_direct(
               input.data(),
               output_include.data(),
               params_include);

    if (status != status_t::success) {
      log_error("LOWOHA Average Pooling (include_padding): Execution failed");
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
