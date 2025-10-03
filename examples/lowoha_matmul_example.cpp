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
                        0.0f, C.data(), ldc,  // beta, dst, ldc
                        params,
                        1, 1  // Batch_A, Batch_B
                      );
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

} // namespace examples
} // namespace zendnnl
