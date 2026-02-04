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

#ifndef _LOWOHA_SOFTMAX_EXAMPLE_HPP_
#define _LOWOHA_SOFTMAX_EXAMPLE_HPP_

#include "zendnnl.hpp"
#include <iostream>
#include <vector>

#define  OK          (0)
#define  NOT_OK      (1)

namespace zendnnl {
namespace examples {

using namespace zendnnl::lowoha::softmax;

/** @fn run_lowoha_softmax_fp32_test
 *  @brief Demonstrates softmax operator on fp32 inputs.
 *
 *  Softmax operator implements softmax activation on multi-dimensional tensors
 *  along a specified axis.
 *
 *  This example demonstrates softmax operator with low overhead API using
 *  FP32 data type on a simple 2D tensor.
 */
int run_lowoha_softmax_fp32_test();

/** @fn run_lowoha_softmax_bf16_test
 *  @brief Demonstrates softmax operator on bf16 inputs.
 *
 *  This example demonstrates softmax operator with low overhead API using
 *  BF16 data type, which is commonly used in inference workloads.
 *
 *  Configuration:
 *    - Input: BF16 [batch, axis_dim]
 *    - Output: BF16 [batch, axis_dim]
 *    - Performs softmax along last axis
 */
int run_lowoha_softmax_bf16_test();

} // examples
} // zendnnl

#endif
