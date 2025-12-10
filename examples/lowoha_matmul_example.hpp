/********************************************************************************
# * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef _LOWOHA_MATMUL_EXAMPLE_HPP_
#define _LOWOHA_MATMUL_EXAMPLE_HPP_

#include "zendnnl.hpp"
#include <iostream>
#include <vector>

#define  OK          (0)
#define  NOT_OK      (1)

namespace zendnnl {
namespace examples {

using namespace zendnnl::lowoha;

/** @fn run_lowoha_matmul_fp32_test
 *  @brief Demonstrates matmul operator on fp32 inputs.
 *
 *  matmul operator implements matrix multiplication on 2D tensors.
 *
 *  This example demonstrates matmul operator with low overhead API.
 */
int run_lowoha_matmul_fp32_test();

/** @fn run_lowoha_matmul_woq_bf16s4_test
 *  @brief Demonstrates LOWOHA WOQ matmul with BF16 input and S4 weights.
 *
 *  This example demonstrates weight-only quantization (WOQ) with per-group
 *  quantization using the low-overhead LOWOHA API.
 *
 *  Configuration:
 *    - Input: BF16 [M, K]
 *    - Weights: S4 packed [K, N]
 *    - Output: F32 [M, N]
 *    - Per-group quantization with scale and zero point
 */
int run_lowoha_matmul_woq_bf16s4_test();

} // examples
} // zendnnl

#endif