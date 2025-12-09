/*******************************************************************************
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

#ifndef _BRGEMM_MATMUL_HPP
#define _BRGEMM_MATMUL_HPP

#include "lowoha_matmul_utils.hpp"

namespace zendnnl {
namespace lowoha {

using namespace zendnnl::ops;

// Reference BRGEMM kernel
void brgemm_ref_kernel(bool transA, bool transB,
                       int M, int N, const int *K_blocks,
                       const void **A_batch, const void **B_batch,
                       size_t batch_size,
                       float alpha, float beta,
                       void *C_tile, int ldc,
                       int lda, int ldb,
                       data_types &dtypes,
                       const void *bias) {

  if (dtypes.src == data_type_t::f32 && dtypes.dst == data_type_t::f32) {
    float *C = static_cast<float *>(C_tile);

    // Scale C by beta
    if (beta != 1.0f) {
      for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
          C[i * ldc + j] *= beta;
        }
    }

    // Accumulate across all K blocks
    for (size_t b = 0; b < batch_size; ++b) {
      const float *A = static_cast<const float *>(A_batch[b]);
      const float *B = static_cast<const float *>(B_batch[b]);
      int K_block = K_blocks[b];

      for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
          double sum = 0.0;
          for (int k = 0; k < K_block; ++k) {
            int a_idx = transA ? (k * lda + i) : (i * lda + k);
            int b_idx = transB ? (j * ldb + k) : (k * ldb + j);
            sum += A[a_idx] * B[b_idx];
          }
          C[i * ldc + j] += alpha * static_cast<float>(sum);
        }
      }
    }

    // Add bias
    if (bias != nullptr) {
      const float *bias_ptr = static_cast<const float *>(bias);
      for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
          C[i * ldc + j] += bias_ptr[j];
        }
    }
  }
  else {
    apilog_error("Unsupported data type for brgemm ref kernel");
  }
  apilog_info("Executing matmul LOWOHA with brgemm ref kernel");
}

} // namespace lowoha
} // namespace zendnnl

#endif