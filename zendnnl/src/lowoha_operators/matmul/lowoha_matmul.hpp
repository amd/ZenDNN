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

#ifndef _LOWOHA_MATMUL_HPP
#define _LOWOHA_MATMUL_HPP

#include <omp.h>
#include <cmath>
#include <cstring>

#include "lowoha_operators/matmul/lowoha_common.hpp"
#include "operators/matmul/matmul_context.hpp"

#define M_FLOPS 6.0
#define ENABLE_K_TILE_OPTIMIZATION 0

namespace zendnnl {
namespace lowoha {

/**
 * @brief Execute matrix multiplication with automatic kernel selection and optimization
 *
 * This function performs C = alpha * op(A) * op(B) + beta * C + fused post-ops.
 *
 * @param layout           Memory layout ('r' for row-major, 'c' for column-major)
 * @param transA           Whether to transpose matrix A
 * @param transB           Whether to transpose matrix B
 * @param M                Number of rows in A and C
 * @param N                Number of columns in B and C
 * @param K                Number of columns in A and rows in B
 * @param alpha            Scaling factor for A*B
 * @param src              Pointer to matrix A data
 * @param lda              Leading dimension of A
 * @param weight           Pointer to matrix B data
 * @param ldb              Leading dimension of B
 * @param bias             Optional bias vector (can be nullptr)
 * @param beta             Scaling factor for existing C values
 * @param dst              Pointer to matrix C data
 * @param ldc              Leading dimension of C
 * @param is_weights_const Whether the weights are constant (enables caching)
 * @param batch_params     Batch parameters including batch sizes and strides
 * @param params           Additional parameters including post-ops and data types
 *
 * @return status_t::success on successful execution, status_t::failure otherwise
 */
status_t matmul_direct(const char layout, const bool transA, const bool transB,
                       const int M, const int N, const int K, const float alpha, const void *src,
                       const int lda, const void *weight, const int ldb, const void *bias,
                       const float beta, void *dst, const int ldc, const bool is_weights_const,
                       batch_params_t batch_params, lowoha_params params);

} // namespace lowoha
} // namespace zendnnl

#endif

