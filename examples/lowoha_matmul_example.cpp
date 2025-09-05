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

    // Call the low-overhead matmul API
    matmul_direct(
      A.data(), B.data(), C.data(), bias.data(),
      1.0f, 0.0f,  // alpha, beta
      M, N, K,
      false, false,  // transA, transB
      lda, ldb, ldc,
      data_type_t::f32,
      data_type_t::f32,
      post_op_type_t::none,
      NULL,
      1, 1  // Batch_A, Batch_B
    );

  }
  catch (const exception_t &ex) {
    return NOT_OK;
  }
  return OK;
}

} // namespace examples
} // namespace zendnnl
