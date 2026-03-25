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

#ifndef MATMUL_NATIVE_INT8_BRGEMM_UKERNEL_HPP
#define MATMUL_NATIVE_INT8_BRGEMM_UKERNEL_HPP

#include "lowoha_operators/matmul/matmul_native/common/avx512_math.hpp"
#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"
#include <cstdint>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

// Use INT8_VNNI_GRP from kernel_cache.hpp (canonical constant).

/// INT8 BRGEMM microkernel function pointer type.
///
/// Computes an MR×NR tile: C_i32[MR][NR] += A_u8[MR][K] * B_s8_vnni[K][NR]
/// then dequantizes: C_fp32 = (C_i32 - zp*col_sum) * scale + bias
///
/// B is in VNNI INT8 layout: groups of 4 consecutive K elements per column,
/// packed into NR_PACK=64 wide panels. Same panel format as BF16 VNNI but
/// with 4-byte groups instead of 2.
///
/// Parameters:
///   A          - u8 source [MR × lda], row-major
///   lda        - leading dimension of A
///   B_vnni     - packed s8 weights in INT8 VNNI layout
///   b_stride   - stride between k-quads in B (= NR_PACK * 4 bytes)
///   C_fp32     - fp32 output [MR × ldc]
///   ldc        - leading dimension of C
///   K          - full K dimension
///   BK         - K-block size for register tiling
///   col_sum    - precomputed column sums of B [NR]
///   src_zp     - source zero point (0 for s8 after +128 adjustment)
///   src_scale  - source quantization scale (per-tensor)
///   wei_scale  - weight quantization scale (per-tensor or per-channel)
///   wei_scale_count - 1 for per-tensor, NR for per-channel
///   bias       - fp32 bias [NR] (may be nullptr)
///   fused_op   - fused activation post-op
///   C_bf16     - bf16 output [MR × ldc_bf16] (nullptr if fp32 output)
///   ldc_bf16   - leading dimension of bf16 output
using int8_brgemm_fn_t = void (*)(
    const uint8_t *__restrict__ A, int lda,
    const int8_t  *__restrict__ B_vnni, int b_stride,
    float *__restrict__ C_fp32, int ldc,
    int K, int BK,
    const int32_t *__restrict__ col_sum,
    int32_t src_zp, float src_scale,
    const float *__restrict__ wei_scale, int wei_scale_count,
    const float *__restrict__ bias,
    fused_postop_t fused_op,
    uint16_t *__restrict__ C_bf16, int ldc_bf16);

__attribute__((target("avx512f,avx512vnni,fma")))
int8_brgemm_fn_t select_int8_brgemm_kernel(int MR, int NR);

__attribute__((target("avx512f,avx512bf16,avx512bw,avx512vl,avx512vnni,fma")))
void int8_brgemm_tail_kernel(
    const uint8_t *__restrict__ A, int lda,
    const int8_t  *__restrict__ B_vnni, int b_stride,
    float *__restrict__ C_fp32, int ldc,
    int K, int BK, int mr_act, int nr_act,
    const int32_t *__restrict__ col_sum,
    int32_t src_zp, float src_scale,
    const float *__restrict__ wei_scale, int wei_scale_count,
    const float *__restrict__ bias,
    fused_postop_t fused_op,
    uint16_t *__restrict__ C_bf16, int ldc_bf16);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_INT8_BRGEMM_UKERNEL_HPP
