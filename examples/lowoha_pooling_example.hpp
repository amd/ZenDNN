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

#ifndef _LOWOHA_POOLING_EXAMPLE_HPP_
#define _LOWOHA_POOLING_EXAMPLE_HPP_

#include "zendnnl.hpp"
#include <iostream>
#include <vector>

#define  OK          (0)
#define  NOT_OK      (1)

namespace zendnnl {
namespace examples {

using namespace zendnnl::lowoha::pooling;

/** @fn run_lowoha_maxpool_fp32_test
 *  @brief Demonstrates max pooling operator on fp32 inputs.
 *
 *  Max Pooling operator performs spatial downsampling by taking the maximum
 *  value in each pooling window.
 *
 *  This example demonstrates max pooling operator with low overhead API using
 *  FP32 data type on a simple NHWC tensor.
 *
 *  Configuration:
 *    - Input: F32 [N, H, W, C] in NHWC format
 *    - Kernel: 2x2
 *    - Stride: 2x2
 *    - Padding: 0 (VALID padding)
 */
int run_lowoha_maxpool_fp32_test();

/** @fn run_lowoha_avgpool_fp32_test
 *  @brief Demonstrates average pooling operator on fp32 inputs.
 *
 *  Average Pooling operator performs spatial downsampling by computing the
 *  average value in each pooling window.
 *
 *  This example demonstrates average pooling operator with low overhead API
 *  using FP32 data type.
 *
 *  Configuration:
 *    - Input: F32 [N, H, W, C] in NHWC format
 *    - Kernel: 3x3
 *    - Stride: 2x2
 *    - Padding: 1 (SAME padding)
 *    - Mode: Exclude padding from average calculation
 */
int run_lowoha_avgpool_fp32_test();

/** @fn run_lowoha_maxpool_bf16_test
 *  @brief Demonstrates max pooling operator on bf16 inputs.
 *
 *  This example demonstrates max pooling operator with low overhead API using
 *  BF16 data type, which is commonly used in inference workloads.
 *
 *  Configuration:
 *    - Input: BF16 [N, H, W, C] in NHWC format
 *    - Kernel: 2x2
 *    - Stride: 2x2
 *    - Output: BF16 [N, H_out, W_out, C]
 */
int run_lowoha_maxpool_bf16_test();

/** @fn run_lowoha_avgpool_padding_modes_test
 *  @brief Demonstrates average pooling with different padding modes.
 *
 *  This example shows the difference between include_padding and
 *  exclude_padding modes in average pooling.
 *
 *  Configuration:
 *    - Input: F32 [N, H, W, C] in NHWC format
 *    - Kernel: 2x2
 *    - Stride: 2x2
 *    - Padding: 1
 *    - Tests both include_padding and exclude_padding modes
 */
int run_lowoha_avgpool_padding_modes_test();

} // examples
} // zendnnl

#endif
