/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#ifndef MATMUL_NATIVE_BRGEMM_KERNEL_F16_NATIVE_F16_UKERNEL_HPP
#define MATMUL_NATIVE_BRGEMM_KERNEL_F16_NATIVE_F16_UKERNEL_HPP

#include <cstdint>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

/// One MR x NR tile (NR = 16 * nv) over a native-layout f16 block, from an
/// f32 source: C[m][n] = beta * C[m][n] + the sum over k of A[m][k] *
/// B[k][n], each B element widened to f32 exactly.  A holds MR rows lda
/// apart, B the K rows of the tile's first pack column b_stride elements
/// apart, C MR rows ldc apart.  K runs as a batch of blocks of BK (the last
/// one shorter); the accumulators stay in registers across the batch, from
/// beta * C, rounded once before the first block (C is not read when beta is
/// 0, where they start at +0.0), and every element then takes one fused
/// multiply-add per k in K order: equal inputs give equal bits, whatever BK,
/// and the bits of the FP32 kernel over the widened weight.
using native_f16_ukernel_fn_t = void (*)(const float *__restrict A, int lda,
        const uint16_t *__restrict B, int b_stride, float *__restrict C,
        int ldc, int K, int BK, float beta);

/// Largest MR served for NR = 16 * nv: 12 at nv 1 and 2, 0 otherwise.  The
/// accumulators, the nv B vectors and the broadcast source fit the 32 vector
/// registers.
int native_f16_ukernel_max_mr(int nv);

/// nullptr outside MR 1 .. native_f16_ukernel_max_mr(nv).
native_f16_ukernel_fn_t select_native_f16_ukernel(int mr, int nv);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_BRGEMM_KERNEL_F16_NATIVE_F16_UKERNEL_HPP
