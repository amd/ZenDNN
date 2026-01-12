/********************************************************************************
# * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

using namespace zendnnl::lowoha::matmul;

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

/** @fn run_lowoha_matmul_int8_caching_test
 *  @brief Demonstrates INT8 matmul with zero-point compensation caching.
 *
 *  This example shows how 1D zero-point compensation is cached when:
 *    - Source has a zero-point (src_zp != 0)
 *    - Weights have no zero-point (wei_zp == 0, symmetric quantization)
 *
 *  Configuration:
 *    - Input: U8 [M, K]
 *    - Weights: S8 [K, N]
 *    - Output: F32 [M, N]
 *    - Per-tensor source scale and zero-point
 *    - Per-channel weight scale
 *
 *  Environment variables:
 *    - ZENDNNL_ZP_COMP_CACHE=1/0 (enable/disable ZP compensation caching)
 */
int run_lowoha_matmul_int8_caching_test();

} // examples
} // zendnnl

#endif