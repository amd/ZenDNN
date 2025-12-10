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

    data_types matmul_dtype;
    matmul_dtype.src = data_type_t::f32;
    matmul_dtype.wei = data_type_t::f32;
    matmul_dtype.dst = data_type_t::f32;
    matmul_dtype.bias = data_type_t::none;
    matmul_dtype.compute = data_type_t::none;

    lowoha_params params;
    params.dtypes = matmul_dtype;

    batch_params_t batch_params;
    batch_params.Batch_A = 1;
    batch_params.Batch_B = 1;

    zendnnl::lowoha::postop op1;
    op1.po_type = post_op_type_t::none;
    op1.buff = nullptr;
    op1.dtype = data_type_t::none;
    op1.dims = {M, N};
    params.postop_.push_back(op1);

    zendnnl::lowoha::postop op2;
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
    data_types matmul_dtype;
    matmul_dtype.src = data_type_t::bf16;
    matmul_dtype.wei = data_type_t::s4;
    matmul_dtype.dst = data_type_t::f32;
    matmul_dtype.bias = data_type_t::none;
    matmul_dtype.compute = data_type_t::none;

    lowoha_params params;
    params.dtypes = matmul_dtype;

    // Setup per-group quantization parameters
    params.quant_params.wei_scale.buff = wei_scale.data();
    params.quant_params.wei_scale.dt = data_type_t::f32;
    params.quant_params.wei_scale.dims = {NUM_GROUPS, N};

    params.quant_params.wei_zp.buff = wei_zp.data();
    params.quant_params.wei_zp.dt = data_type_t::s8;
    params.quant_params.wei_zp.dims = {NUM_GROUPS, N};

    batch_params_t batch_params;
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

} // namespace examples
} // namespace zendnnl
