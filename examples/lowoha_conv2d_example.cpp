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

#include "lowoha_conv2d_example.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

namespace zendnnl {
namespace examples {

int run_lowoha_conv2d_fp32_test() {
  try {
    // Convolution dimensions (typical ResNet configuration)
    int batch = 4;
    int in_h = 56, in_w = 56, in_c = 64;
    int filter_h = 3, filter_w = 3, out_c = 128;
    
    // Output dimensions with SAME padding (stride=1, pad=1)
    int out_h = 56;
    int out_w = 56;
    
    log_info("LOWOHA Conv2D FP32 example");
    log_info("Input: [", batch, ", ", in_h, ", ", in_w, ", ", in_c, "]");
    log_info("Filter: [", filter_h, ", ", filter_w, ", ", in_c, ", ", out_c, "]");
    log_info("Output: [", batch, ", ", out_h, ", ", out_w, ", ", out_c, "]");

    // Allocate tensors
    std::vector<float> input(batch * in_h * in_w * in_c, 1.0f);
    std::vector<float> filter(filter_h * filter_w * in_c * out_c, 0.5f);
    std::vector<float> bias(out_c, 0.1f);
    std::vector<float> output(batch * out_h * out_w * out_c, 0.0f);

    // Configure conv_params
    conv_params params;
    
    // Set dimensions
    params.dims.batch = batch;
    params.dims.in_height = in_h;
    params.dims.in_width = in_w;
    params.dims.in_channels = in_c;
    params.dims.filter_height = filter_h;
    params.dims.filter_width = filter_w;
    params.dims.out_channels = out_c;
    params.dims.out_height = out_h;
    params.dims.out_width = out_w;
    
    // Set stride and padding (SAME padding)
    params.stride_h = 1;
    params.stride_w = 1;
    params.pad_top = 1;
    params.pad_bottom = 1;
    params.pad_left = 1;
    params.pad_right = 1;
    
    // Set data types
    params.dtypes.input = data_type_t::f32;
    params.dtypes.filter = data_type_t::f32;
    params.dtypes.bias = data_type_t::f32;
    params.dtypes.output = data_type_t::f32;
    
    // Add ReLU post-op
    conv_postop relu_op;
    relu_op.po_type = post_op_type_t::relu;
    relu_op.buff = nullptr;
    relu_op.dtype = data_type_t::none;
    params.postop_.push_back(relu_op);

    // Execute Conv2D
    log_info("Executing LOWOHA Conv2D...");
    status_t status = conv_direct(
      input.data(),
      filter.data(),
      bias.data(),
      output.data(),
      true,  // is_weights_const (enables caching)
      params
    );

    if (status != status_t::success) {
      log_error("LOWOHA Conv2D execution failed");
      return NOT_OK;
    }

    log_info("LOWOHA Conv2D FP32 executed successfully!");
    log_info("Output[0] = ", output[0]);

  }
  catch (const exception_t &ex) {
    log_error("Exception: ", ex.what());
    return NOT_OK;
  }
  return OK;
}

int run_lowoha_conv2d_bf16_test() {
  try {
    // Convolution dimensions (ResNet-style residual block)
    int batch = 8;
    int in_h = 28, in_w = 28, in_c = 128;
    int filter_h = 3, filter_w = 3, out_c = 256;
    int out_h = 28, out_w = 28;
    
    log_info("LOWOHA Conv2D BF16 example with fused operations");
    log_info("Input: [", batch, ", ", in_h, ", ", in_w, ", ", in_c, "] (BF16)");
    log_info("Filter: [", filter_h, ", ", filter_w, ", ", in_c, ", ", out_c, "] (BF16)");
    log_info("Output: [", batch, ", ", out_h, ", ", out_w, ", ", out_c, "] (BF16)");

    // Allocate BF16 tensors (stored as uint16_t)
    std::vector<uint16_t> input_bf16(batch * in_h * in_w * in_c);
    std::vector<uint16_t> filter_bf16(filter_h * filter_w * in_c * out_c);
    std::vector<float> bias(out_c, 0.0f);
    std::vector<uint16_t> output_bf16(batch * out_h * out_w * out_c);
    
    // Binary add tensor (for residual connection)
    std::vector<uint16_t> add_tensor(batch * out_h * out_w * out_c);
    
    // Initialize with BF16 representation of 1.0f (0x3F80)
    std::fill(input_bf16.begin(), input_bf16.end(), 0x3F80);
    std::fill(filter_bf16.begin(), filter_bf16.end(), 0x3F00);  // 0.5f in BF16
    std::fill(add_tensor.begin(), add_tensor.end(), 0x3E80);    // 0.25f in BF16

    // Configure conv_params
    conv_params params;
    
    // Set dimensions
    params.dims.batch = batch;
    params.dims.in_height = in_h;
    params.dims.in_width = in_w;
    params.dims.in_channels = in_c;
    params.dims.filter_height = filter_h;
    params.dims.filter_width = filter_w;
    params.dims.out_channels = out_c;
    params.dims.out_height = out_h;
    params.dims.out_width = out_w;
    
    // Set stride and padding
    params.stride_h = 1;
    params.stride_w = 1;
    params.pad_top = 1;
    params.pad_bottom = 1;
    params.pad_left = 1;
    params.pad_right = 1;
    
    // Set data types
    params.dtypes.input = data_type_t::bf16;
    params.dtypes.filter = data_type_t::bf16;
    params.dtypes.bias = data_type_t::f32;
    params.dtypes.output = data_type_t::bf16;
    
    // Post-op 1: Binary Add (residual connection)
    conv_postop add_op;
    add_op.po_type = post_op_type_t::binary_add;
    add_op.buff = add_tensor.data();
    add_op.dtype = data_type_t::bf16;
    add_op.dims = {static_cast<int64_t>(batch), 
                   static_cast<int64_t>(out_h), 
                   static_cast<int64_t>(out_w), 
                   static_cast<int64_t>(out_c)};
    params.postop_.push_back(add_op);
    
    // Post-op 2: ReLU
    conv_postop relu_op;
    relu_op.po_type = post_op_type_t::relu;
    relu_op.buff = nullptr;
    relu_op.dtype = data_type_t::none;
    params.postop_.push_back(relu_op);

    // Execute Conv2D
    log_info("Executing LOWOHA BF16 Conv2D with Binary Add + ReLU...");
    status_t status = conv_direct(
      input_bf16.data(),
      filter_bf16.data(),
      bias.data(),
      output_bf16.data(),
      true,  // is_weights_const (enables caching)
      params
    );

    if (status != status_t::success) {
      log_error("LOWOHA BF16 Conv2D execution failed");
      return NOT_OK;
    }

    log_info("LOWOHA Conv2D BF16 executed successfully!");
    log_info("Expected 1.5-2.5x speedup over FP32 on Zen3/Zen4");

  }
  catch (const exception_t &ex) {
    log_error("Exception: ", ex.what());
    return NOT_OK;
  }
  return OK;
}

int run_lowoha_depthwise_conv2d_test() {
  try {
    // Depthwise convolution dimensions (MobileNet pattern)
    int batch = 4;
    int in_h = 112, in_w = 112, in_c = 32;
    int filter_h = 3, filter_w = 3;
    int depth_multiplier = 1;
    int out_c = in_c * depth_multiplier;  // 32
    int out_h = 112, out_w = 112;
    
    log_info("LOWOHA Depthwise Conv2D example (MobileNet pattern)");
    log_info("Input: [", batch, ", ", in_h, ", ", in_w, ", ", in_c, "]");
    log_info("Filter: [", filter_h, ", ", filter_w, ", ", in_c, ", ", depth_multiplier, "]");
    log_info("Groups: ", in_c, " (depthwise)");
    log_info("Output: [", batch, ", ", out_h, ", ", out_w, ", ", out_c, "]");
    log_info("Using Clip(0, 6) for ReLU6 activation");

    // Allocate tensors
    std::vector<float> input(batch * in_h * in_w * in_c, 1.0f);
    std::vector<float> filter(filter_h * filter_w * in_c * depth_multiplier, 0.5f);
    std::vector<float> bias(out_c, 0.0f);
    std::vector<float> output(batch * out_h * out_w * out_c, 0.0f);

    // Configure conv_params
    conv_params params;
    
    // Set dimensions
    params.dims.batch = batch;
    params.dims.in_height = in_h;
    params.dims.in_width = in_w;
    params.dims.in_channels = in_c;
    params.dims.filter_height = filter_h;
    params.dims.filter_width = filter_w;
    params.dims.out_channels = out_c;
    params.dims.out_height = out_h;
    params.dims.out_width = out_w;
    
    // Set stride and padding
    params.stride_h = 1;
    params.stride_w = 1;
    params.pad_top = 1;
    params.pad_bottom = 1;
    params.pad_left = 1;
    params.pad_right = 1;
    
    // Configure depthwise convolution
    params.depthwise.groups = in_c;
    params.depthwise.is_depthwise = true;
    params.depthwise.depth_multiplier = depth_multiplier;
    
    // Set data types
    params.dtypes.input = data_type_t::f32;
    params.dtypes.filter = data_type_t::f32;
    params.dtypes.bias = data_type_t::f32;
    params.dtypes.output = data_type_t::f32;
    
    // Add Clip post-op for ReLU6 (common in MobileNet)
    // ReLU6 = clip(x, 0, 6)
    conv_postop clip_op;
    clip_op.po_type = post_op_type_t::clip;
    clip_op.buff = nullptr;
    clip_op.dtype = data_type_t::none;
    clip_op.alpha = 0.0f;  // lower bound
    clip_op.beta = 6.0f;   // upper bound
    params.postop_.push_back(clip_op);

    // Execute Depthwise Conv2D
    log_info("Executing LOWOHA Depthwise Conv2D...");
    status_t status = conv_direct(
      input.data(),
      filter.data(),
      bias.data(),
      output.data(),
      true,  // is_weights_const (enables caching)
      params
    );

    if (status != status_t::success) {
      log_error("LOWOHA Depthwise Conv2D execution failed");
      return NOT_OK;
    }

    log_info("LOWOHA Depthwise Conv2D executed successfully!");
    log_info("Output[0] = ", output[0]);
    log_info("Depthwise reduces parameters by ~", (out_c / depth_multiplier), "x");

  }
  catch (const exception_t &ex) {
    log_error("Exception: ", ex.what());
    return NOT_OK;
  }
  return OK;
}

int run_lowoha_strided_conv2d_test() {
  try {
    // Strided convolution for 2x downsampling
    int batch = 4;
    int in_h = 56, in_w = 56, in_c = 64;
    int filter_h = 3, filter_w = 3, out_c = 128;
    
    // Output dimensions with stride=2
    int out_h = (in_h + 2 - filter_h) / 2 + 1;  // 28
    int out_w = (in_w + 2 - filter_w) / 2 + 1;  // 28
    
    log_info("LOWOHA Strided Conv2D example (2x downsampling)");
    log_info("Input: [", batch, ", ", in_h, ", ", in_w, ", ", in_c, "]");
    log_info("Filter: [", filter_h, ", ", filter_w, ", ", in_c, ", ", out_c, "]");
    log_info("Stride: 2x2");
    log_info("Output: [", batch, ", ", out_h, ", ", out_w, ", ", out_c, "]");

    // Allocate tensors
    std::vector<float> input(batch * in_h * in_w * in_c, 1.0f);
    std::vector<float> filter(filter_h * filter_w * in_c * out_c, 0.5f);
    std::vector<float> bias(out_c, 0.0f);
    std::vector<float> output(batch * out_h * out_w * out_c, 0.0f);

    // Configure conv_params
    conv_params params;
    
    // Set dimensions
    params.dims.batch = batch;
    params.dims.in_height = in_h;
    params.dims.in_width = in_w;
    params.dims.in_channels = in_c;
    params.dims.filter_height = filter_h;
    params.dims.filter_width = filter_w;
    params.dims.out_channels = out_c;
    params.dims.out_height = out_h;
    params.dims.out_width = out_w;
    
    // Set stride=2 for downsampling
    params.stride_h = 2;
    params.stride_w = 2;
    params.pad_top = 1;
    params.pad_bottom = 1;
    params.pad_left = 1;
    params.pad_right = 1;
    
    // Set data types
    params.dtypes.input = data_type_t::f32;
    params.dtypes.filter = data_type_t::f32;
    params.dtypes.bias = data_type_t::f32;
    params.dtypes.output = data_type_t::f32;
    
    // Add ReLU post-op
    conv_postop relu_op;
    relu_op.po_type = post_op_type_t::relu;
    relu_op.buff = nullptr;
    relu_op.dtype = data_type_t::none;
    params.postop_.push_back(relu_op);

    // Execute Strided Conv2D
    log_info("Executing LOWOHA Strided Conv2D...");
    status_t status = conv_direct(
      input.data(),
      filter.data(),
      bias.data(),
      output.data(),
      true,  // is_weights_const (enables caching)
      params
    );

    if (status != status_t::success) {
      log_error("LOWOHA Strided Conv2D execution failed");
      return NOT_OK;
    }

    log_info("LOWOHA Strided Conv2D executed successfully!");
    log_info("Spatial dimensions downsampled from ", in_h, "x", in_w, " to ", out_h, "x", out_w);

  }
  catch (const exception_t &ex) {
    log_error("Exception: ", ex.what());
    return NOT_OK;
  }
  return OK;
}

int run_lowoha_dilated_conv2d_test() {
  try {
    // Dilated convolution (used in DeepLab, WaveNet)
    int batch = 2;
    int in_h = 64, in_w = 64, in_c = 128;
    int filter_h = 3, filter_w = 3, out_c = 128;
    
    // With dilation=2, effective kernel size is 5x5
    int dilation = 2;
    int effective_filter_h = (filter_h - 1) * dilation + 1;  // 5
    int effective_filter_w = (filter_w - 1) * dilation + 1;  // 5
    
    // Calculate output dimensions
    int out_h = (in_h + 4 - effective_filter_h) + 1;  // 64 (with pad=2)
    int out_w = (in_w + 4 - effective_filter_w) + 1;  // 64
    
    log_info("LOWOHA Dilated Conv2D example (atrous convolution)");
    log_info("Input: [", batch, ", ", in_h, ", ", in_w, ", ", in_c, "]");
    log_info("Filter: [", filter_h, ", ", filter_w, ", ", in_c, ", ", out_c, "]");
    log_info("Dilation: ", dilation, " (effective kernel: ", effective_filter_h, "x", effective_filter_w, ")");
    log_info("Output: [", batch, ", ", out_h, ", ", out_w, ", ", out_c, "]");

    // Allocate tensors
    std::vector<float> input(batch * in_h * in_w * in_c, 1.0f);
    std::vector<float> filter(filter_h * filter_w * in_c * out_c, 0.5f);
    std::vector<float> bias(out_c, 0.0f);
    std::vector<float> output(batch * out_h * out_w * out_c, 0.0f);

    // Configure conv_params
    conv_params params;
    
    // Set dimensions
    params.dims.batch = batch;
    params.dims.in_height = in_h;
    params.dims.in_width = in_w;
    params.dims.in_channels = in_c;
    params.dims.filter_height = filter_h;
    params.dims.filter_width = filter_w;
    params.dims.out_channels = out_c;
    params.dims.out_height = out_h;
    params.dims.out_width = out_w;
    
    // Set stride and padding
    params.stride_h = 1;
    params.stride_w = 1;
    params.pad_top = 2;
    params.pad_bottom = 2;
    params.pad_left = 2;
    params.pad_right = 2;
    
    // Set dilation
    params.dilation_h = dilation;
    params.dilation_w = dilation;
    
    // Set data types
    params.dtypes.input = data_type_t::f32;
    params.dtypes.filter = data_type_t::f32;
    params.dtypes.bias = data_type_t::f32;
    params.dtypes.output = data_type_t::f32;
    
    // Add ReLU post-op
    conv_postop relu_op;
    relu_op.po_type = post_op_type_t::relu;
    relu_op.buff = nullptr;
    relu_op.dtype = data_type_t::none;
    params.postop_.push_back(relu_op);

    // Execute Dilated Conv2D
    log_info("Executing LOWOHA Dilated Conv2D...");
    status_t status = conv_direct(
      input.data(),
      filter.data(),
      bias.data(),
      output.data(),
      true,  // is_weights_const (enables caching)
      params
    );

    if (status != status_t::success) {
      log_error("LOWOHA Dilated Conv2D execution failed");
      return NOT_OK;
    }

    log_info("LOWOHA Dilated Conv2D executed successfully!");
    log_info("Expanded receptive field without increasing parameters");

  }
  catch (const exception_t &ex) {
    log_error("Exception: ", ex.what());
    return NOT_OK;
  }
  return OK;
}

} // namespace examples
} // namespace zendnnl
