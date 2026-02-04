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

#ifndef _LOWOHA_CONV2D_EXAMPLE_HPP_
#define _LOWOHA_CONV2D_EXAMPLE_HPP_

#include "zendnnl.hpp"
#include <iostream>
#include <vector>

#define  OK          (0)
#define  NOT_OK      (1)

namespace zendnnl {
namespace examples {

using namespace zendnnl::lowoha::conv;
using namespace zendnnl::common;
using namespace zendnnl::ops;

/** @fn run_lowoha_conv2d_fp32_test
 *  @brief Demonstrates Conv2D operator on FP32 inputs.
 *
 *  Conv2D operator implements 2D convolution on 4D tensors (NHWC format).
 *
 *  This example demonstrates Conv2D with low overhead API using:
 *    - Input: [N, H, W, C] in NHWC format
 *    - Filter: [KH, KW, C_in, C_out]
 *    - ReLU post-operation
 */
int run_lowoha_conv2d_fp32_test();

/** @fn run_lowoha_conv2d_bf16_test
 *  @brief Demonstrates Conv2D operator on BF16 inputs.
 *
 *  This example demonstrates BF16 Conv2D with fused post-operations:
 *    - Input: BF16 [N, H, W, C]
 *    - Filter: BF16 [KH, KW, C_in, C_out]
 *    - Bias: FP32 [C_out]
 *    - Binary Add (residual connection)
 *    - ReLU activation
 *    - Output: BF16
 *
 *  Configuration demonstrates a typical ResNet-style residual block.
 */
int run_lowoha_conv2d_bf16_test();

/** @fn run_lowoha_depthwise_conv2d_test
 *  @brief Demonstrates Depthwise Conv2D (MobileNet pattern).
 *
 *  This example shows depthwise convolution commonly used in MobileNet:
 *    - Input: FP32 [N, H, W, C]
 *    - Filter: FP32 [KH, KW, C_in, multiplier]
 *    - groups = in_channels (depthwise)
 *    - Clip post-operation (ReLU6: clip(0, 6))
 *
 *  Depthwise convolution dramatically reduces parameters and computation
 *  compared to standard convolution.
 */
int run_lowoha_depthwise_conv2d_test();

/** @fn run_lowoha_strided_conv2d_test
 *  @brief Demonstrates Strided Conv2D for downsampling.
 *
 *  This example shows strided convolution (stride=2) for 2x downsampling:
 *    - Input: FP32 [N, H, W, C]
 *    - Output: FP32 [N, H/2, W/2, C_out]
 *    - Stride: 2x2
 *    - ReLU post-operation
 *
 *  Strided convolutions are commonly used instead of pooling for
 *  learnable downsampling in modern architectures.
 */
int run_lowoha_strided_conv2d_test();

/** @fn run_lowoha_dilated_conv2d_test
 *  @brief Demonstrates Dilated Conv2D (atrous convolution).
 *
 *  This example shows dilated convolution used in DeepLab and WaveNet:
 *    - Input: FP32 [N, H, W, C]
 *    - Filter: 3x3 with dilation=2 (effective 5x5)
 *    - Expands receptive field without increasing parameters
 *    - ReLU post-operation
 *
 *  Dilated convolutions are useful for semantic segmentation and
 *  capturing larger context.
 */
int run_lowoha_dilated_conv2d_test();

} // examples
} // zendnnl

#endif
