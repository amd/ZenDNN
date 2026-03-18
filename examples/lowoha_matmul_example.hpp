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

/** @fn run_lowoha_matmul_f16_test
 *  @brief Demonstrates F16 (half precision) matmul using OneDNN backend.
 *
 *  This example demonstrates matmul with all F16 tensors.
 *  F16 matmul is only supported via OneDNN backend.
 *
 *  Configuration:
 *    - Input: F16 [M, K]
 *    - Weights: F16 [K, N]
 *    - Output: F16 [M, N]
 */
int run_lowoha_matmul_f16_test();

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

/** @fn run_lowoha_matmul_woq_bf16u4_test
 *  @brief Demonstrates LOWOHA WOQ matmul with BF16 input and U4 weights.
 *
 *  This example demonstrates weight-only quantization (WOQ) with per-group
 *  quantization using unsigned 4-bit (U4) weights and BF16 scales/zero-points.
 *
 *  Configuration:
 *    - Input: BF16 [M, K]
 *    - Weights: U4 packed [K, N]
 *    - Output: F32 [M, N]
 *    - Per-group quantization with BF16 scale and BF16 zero point (simulated WOQ)
 */
int run_lowoha_matmul_woq_bf16u4_test();

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

/** @fn group_gemm_f32_kernel_example
 *  @brief Demonstrates group GEMM API for multiple independent matmul operations.
 *
 *  group_gemm executes multiple independent matrix multiplications in sequence.
 *  Each operation computes: C[i] = alpha[i] * op(A[i]) * op(B[i]) + beta[i] * C[i]
 *
 *  This example demonstrates how to use the group_gemm API with multiple
 *  f32 matmul operations of varying dimensions.
 */
int group_gemm_f32_kernel_example();

/** @fn sequential_gemm_f32_kernel_example
 *  @brief Demonstrates sequential GEMM using group_gemm_direct API.
 *
 *  Sequential GEMM chains multiple matmul operations where the output of each
 *  operation feeds as input to the next, simulating a multi-layer perceptron.
 *  Triggered by passing src with size 1 (single input).
 *
 *  Chain: Input -> Linear1 -> Linear2 -> Linear3 -> Output
 *
 *  This example demonstrates the sequential execution mode where each
 *  operation uses all available threads for maximum throughput.
 */
int sequential_gemm_f32_kernel_example();

} // examples
} // zendnnl

#endif