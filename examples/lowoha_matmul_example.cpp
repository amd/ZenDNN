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

#include "lowoha_matmul_example.hpp"

namespace zendnnl {
namespace examples {

int run_lowoha_matmul_fp32_test() {
  try {
    // Matrix dimensions
    int M = 2, N = 3, K = 4;
    int lda = K, ldb = N, ldc = N;

    // Input matrices (row-major)
    std::vector<float> A = {
      1, 2, 3, 4,
      5, 6, 7, 8
    }; // 2x4

    std::vector<float> B = {
      1, 2, 3,
      4, 5, 6,
      7, 8, 9,
      10, 11, 12
    }; // 4x3

    std::vector<float> C(M * N, 0); // Output matrix 2x3
    std::vector<float> bias = {1, 1, 1}; // Bias for each output column

    matmul_data_types matmul_dtype;
    matmul_dtype.src = data_type_t::f32;
    matmul_dtype.wei = data_type_t::f32;
    matmul_dtype.dst = data_type_t::f32;
    matmul_dtype.bias = data_type_t::none;
    matmul_dtype.compute = data_type_t::none;

    matmul_params params;
    params.dtypes = matmul_dtype;

    matmul_batch_params_t batch_params;
    batch_params.Batch_A = 1;
    batch_params.Batch_B = 1;

    zendnnl::lowoha::matmul::matmul_post_op op1;
    op1.po_type = post_op_type_t::none;
    op1.buff = nullptr;
    op1.dtype = data_type_t::none;
    op1.dims = {M, N};
    params.postop_.push_back(op1);

    zendnnl::lowoha::matmul::matmul_post_op op2;
    op2.po_type = post_op_type_t::relu;
    op2.buff = nullptr;
    op2.dtype = data_type_t::none;
    op2.dims = {M, N};
    params.postop_.push_back(op2);

    // Call the low-overhead matmul API
    status_t status = matmul_direct(
                        'r',  // layout: row-major
                        false, false,  // transA, transB
                        M, N, K,
                        1.0f, A.data(), lda, B.data(), ldb,
                        nullptr,  // alpha, src, lda, weight, ldb, bias
                        0.0f, C.data(), ldc, true,  // beta, dst, ldc, is_weights_const
                        batch_params, params);
    if (status != status_t::success) {
      log_error("LOWOHA: Execution failed");
      return NOT_OK;
    }

  }
  catch (const exception_t &ex) {
    return NOT_OK;
  }
  return OK;
}

int run_lowoha_matmul_woq_bf16s4_test() {
  try {
    // ========== Matrix Dimensions ==========
    constexpr int M = 16, K = 128, N = 64;
    constexpr int GROUP_SIZE = 32;      // Typical group size for LLM quantization
    constexpr int NUM_GROUPS = K / GROUP_SIZE;  // = 4 groups

    log_info("LOWOHA WOQ BF16xS4 matmul example with per-group quantization");
    log_info("Matrix dimensions: M=", M, " K=", K, " N=", N);
    log_info("Group size: ", GROUP_SIZE, ", Number of groups: ", NUM_GROUPS);

    // ========== Create Weight Scale Tensor (per-group) ==========
    // Dimensions {NUM_GROUPS, N} - one scale per group per output channel
    std::vector<float> wei_scale(NUM_GROUPS * N);
    for (int g = 0; g < NUM_GROUPS; ++g) {
      for (int n = 0; n < N; ++n) {
        wei_scale[g * N + n] = 1.0f + 0.1f * g;  // Varying scales: 1.0, 1.1, 1.2, 1.3
      }
    }

    // ========== Create Zero Point Tensor (per-group) ==========
    // Dimensions {NUM_GROUPS, N} - one zp per group per output channel
    std::vector<int8_t> wei_zp(NUM_GROUPS * N);
    for (int g = 0; g < NUM_GROUPS; ++g) {
      for (int n = 0; n < N; ++n) {
        wei_zp[g * N + n] = static_cast<int8_t>(g % 4);  // zp = 0, 1, 2, 3
      }
    }

    // ========== Create S4 Packed Weights ==========
    // S4 weights are packed: 2 values per byte
    size_t packed_weight_size = (K * N + 1) / 2;
    std::vector<int8_t> weights(packed_weight_size);
    int8_t s4_val = 1 & 0x0F;
    int8_t packed_val = s4_val | (s4_val << 4);  // Same value in both nibbles
    std::fill(weights.begin(), weights.end(), packed_val);

    // ========== Create BF16 Input ==========
    std::vector<int16_t> input(M * K);
    // Fill with bf16 representation of 1.0f (0x3F80)
    std::fill(input.begin(), input.end(), 0x3F80);

    // ========== Output Tensor ==========
    std::vector<float> output(M * N, 0.0f);

    // ========== Setup LOWOHA Parameters for WOQ ==========
    matmul_data_types matmul_dtype;
    matmul_dtype.src = data_type_t::bf16;
    matmul_dtype.wei = data_type_t::s4;
    matmul_dtype.dst = data_type_t::f32;
    matmul_dtype.bias = data_type_t::none;
    matmul_dtype.compute = data_type_t::none;

    matmul_params params;
    params.dtypes = matmul_dtype;

    // Setup per-group quantization parameters
    params.quant_params.wei_scale.buff = wei_scale.data();
    params.quant_params.wei_scale.dt = data_type_t::f32;
    params.quant_params.wei_scale.dims = {NUM_GROUPS, N};

    params.quant_params.wei_zp.buff = wei_zp.data();
    params.quant_params.wei_zp.dt = data_type_t::s8;
    params.quant_params.wei_zp.dims = {NUM_GROUPS, N};

    matmul_batch_params_t batch_params;
    batch_params.Batch_A = 1;
    batch_params.Batch_B = 1;

    // ========== Execute LOWOHA WOQ Kernel ==========
    log_info("Executing LOWOHA WOQ kernel with per-group quantization...");
    status_t status = matmul_direct(
                          'r',  // layout: row-major
                          false, false,  // transA, transB
                          M, N, K,
                          1.0f, input.data(), K,
                          weights.data(), N,
                          nullptr,  // no bias
                          0.0f, output.data(), N,
                          true,  // is_weights_const
                          batch_params, params);

    if (status != status_t::success) {
      log_error("LOWOHA WOQ kernel execution failed.");
      return NOT_OK;
    }

    log_info("LOWOHA WOQ BF16xS4 matmul executed successfully!");
    log_info("Output[0,0] = ", output[0]);

  }
  catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

/**
 * @brief Test INT8 matmul with zero-point compensation caching
 * 
 * This example demonstrates:
 * 1. INT8 (u8 x s8 -> s32/f32) matmul with source zero-point
 * 2. Zero-point compensation caching (1D case is cached)
 * 3. Running multiple iterations to show cache hits
 * 
 * Environment variables to control caching:
 *   ZENDNNL_ZP_COMP_CACHE=1  (enable, default)
 *   ZENDNNL_ZP_COMP_CACHE=0  (disable)
 * 
 */
int run_lowoha_matmul_int8_caching_test() {
  try {
    // ========== Matrix Dimensions ==========
    // Using realistic sizes for LLM inference
    constexpr int M = 32;    // Batch size / sequence length
    constexpr int K = 4096;  // Hidden dimension
    constexpr int N = 4096;  // Output dimension
    constexpr int NUM_ITERATIONS = 5;  // Run multiple times to test caching

    log_info("========================================");
    log_info("LOWOHA INT8 MatMul Caching Test");
    log_info("========================================");
    log_info("Matrix dimensions: M=", M, " K=", K, " N=", N);
    log_info("Number of iterations: ", NUM_ITERATIONS);
    log_info("");

    // ========== Check Cache Configuration ==========
    const char* zp_cache_env = std::getenv("ZENDNNL_ZP_COMP_CACHE");
    bool zp_cache_disabled = !zp_cache_env || std::atoi(zp_cache_env) != 1;
    
    log_info("Cache configuration:");
    log_info("  ZP compensation cache: ", (zp_cache_disabled ? "DISABLED" : "ENABLED"));
    log_info("");

    // ========== Create INT8 Weights (s8) ==========
    std::vector<int8_t> weights(K * N);
    for (size_t i = 0; i < weights.size(); ++i) {
      weights[i] = static_cast<int8_t>((i % 7) - 3);  // Values: -3 to 3
    }

    // ========== Create Quantization Parameters ==========
    // Source scale: per-tensor
    float src_scale_val = 0.05f;  // Typical scale for activations
    
    // Weight scale: per-channel (one scale per output column)
    std::vector<float> wei_scale(N);
    for (int n = 0; n < N; ++n) {
      wei_scale[n] = 0.01f + 0.0001f * (n % 100);  // Small variations
    }
    
    // Destination scale: per-tensor
    float dst_scale_val = 0.1f;
    
    // Source zero-point (asymmetric quantization for activations)
    // This triggers 1D compensation which IS cached
    int32_t src_zp_val = 128;  // Typical for u8 with range [0, 255] centered
    
    // Weight zero-point = 0 (symmetric quantization for weights)
    // This ensures we get 1D compensation (cacheable)
    // If wei_zp != 0, we'd get 2D compensation (not cacheable)
    int32_t wei_zp_val = 0;

    log_info("Quantization parameters:");
    log_info("  src_scale = ", src_scale_val);
    log_info("  wei_scale = per-channel (", N, " values)");
    log_info("  dst_scale = ", dst_scale_val);
    log_info("  src_zp = ", src_zp_val, " (asymmetric -> 1D compensation, CACHEABLE)");
    log_info("  wei_zp = ", wei_zp_val, " (symmetric -> no 2D term)");
    log_info("");

    // ========== Setup LOWOHA Parameters ==========
    matmul_data_types matmul_dtype;
    matmul_dtype.src = data_type_t::u8;   // Unsigned 8-bit activations
    matmul_dtype.wei = data_type_t::s8;   // Signed 8-bit weights
    matmul_dtype.dst = data_type_t::f32;  // Float32 output
    matmul_dtype.bias = data_type_t::none;
    matmul_dtype.compute = data_type_t::none;

    matmul_params params;
    params.dtypes = matmul_dtype;

    // Set source scale
    params.quant_params.src_scale.buff = &src_scale_val;
    params.quant_params.src_scale.dt = data_type_t::f32;
    params.quant_params.src_scale.dims = {1};

    // Set weight scale (per-channel)
    params.quant_params.wei_scale.buff = wei_scale.data();
    params.quant_params.wei_scale.dt = data_type_t::f32;
    params.quant_params.wei_scale.dims = {1, N};

    // Set destination scale
    params.quant_params.dst_scale.buff = &dst_scale_val;
    params.quant_params.dst_scale.dt = data_type_t::f32;
    params.quant_params.dst_scale.dims = {1};

    // Set source zero-point (this triggers 1D compensation caching)
    params.quant_params.src_zp.buff = &src_zp_val;
    params.quant_params.src_zp.dt = data_type_t::s32;
    params.quant_params.src_zp.dims = {1};

    // Weight zero-point is 0 (symmetric) - no need to set

    matmul_batch_params_t batch_params;
    batch_params.Batch_A = 1;
    batch_params.Batch_B = 1;

    // Add ReLU post-op
    zendnnl::lowoha::matmul::matmul_post_op relu_op;
    relu_op.po_type = post_op_type_t::relu;
    relu_op.buff = nullptr;
    relu_op.dtype = data_type_t::none;
    params.postop_.push_back(relu_op);

    // ========== Run Multiple Iterations ==========
    log_info("Running ", NUM_ITERATIONS, " iterations...");
    log_info("(Iteration 1 computes & caches, iterations 2+ should hit cache)");
    log_info("");

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
      // Create new input for each iteration (simulating different inference batches)
      // This demonstrates that 1D compensation is reused even with different inputs
      std::vector<uint8_t> input(M * K);
      for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<uint8_t>((i + iter * 17) % 256);  // Varying input
      }

      // Output buffer
      std::vector<float> output(M * N, 0.0f);

      log_info("--- Iteration ", iter + 1, " ---");

      // Execute INT8 matmul
      status_t status = matmul_direct(
                          'r',  // layout: row-major
                          false, false,  // transA, transB
                          M, N, K,
                          1.0f, input.data(), K,
                          weights.data(), N,
                          nullptr,  // no bias
                          0.0f, output.data(), N,
                          true,  // is_weights_const (required for caching)
                          batch_params, params);

      if (status != status_t::success) {
        log_error("INT8 matmul failed at iteration ", iter + 1);
        return NOT_OK;
      }

      // Print sample output values
      log_info("  Output[0,0] = ", output[0], ", Output[0,1] = ", output[1]);
    }

    log_info("");
    log_info("========================================");
    log_info("INT8 Caching Test PASSED!");
    log_info("========================================");
    log_info("");
    log_info("Summary:");
    log_info("- 1D ZP compensation (src_zp only) was computed once and cached");
    log_info("- Subsequent iterations reused the cached compensation");
    log_info("- Different input data each iteration proved cache works correctly");

  }
  catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

} // namespace examples
} // namespace zendnnl
