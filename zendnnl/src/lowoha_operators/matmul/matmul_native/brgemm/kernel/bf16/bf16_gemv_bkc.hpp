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

#pragma once
#include <cstdint>
#include "lowoha_operators/matmul/matmul_native/common/avx512_math.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

/// Choose block width for BKC packing based on N.
/// Returns 384 (NP=6) when N is 64-aligned and in (256, 512], else 256 (NP=4).
/// N > 512 stays on 256: the 384+256 split causes cross-CU i-cache thrashing
/// (NP=6 from wide CU + NP=4 from standard CU).
/// Both packing and kernel must use the same block width for a given N.
inline int choose_blk_n(int N) {
    constexpr int BLK_N_STD  = 4 * 64;   // 256
    constexpr int BLK_N_WIDE = 6 * 64;   // 384
    return (N > BLK_N_STD && N <= 512 && (N % 64) == 0)
        ? BLK_N_WIDE : BLK_N_STD;
}

/// Blocked K-contiguous (BKC) GEMV kernel with block-aware packing.
/// B_bkc must be packed with pack_b_bkc_ext().
/// Blocked K-contiguous (BKC) packing: B is partitioned into blocks
/// (256 or 384 columns depending on N alignment), each packed with
/// K-contiguous VNNI layout. Within each block, all k-pairs are
/// contiguous with stride = blk_N_padded × VNNI_PAIR.
void bf16_gemv_bkc(
    const uint16_t *__restrict__ A,
    const uint16_t *__restrict__ B_bkc,
    uint16_t *__restrict__ C_bf16,
    float *__restrict__ C_fp32,
    const float *__restrict__ bias_f,
    fused_postop_t fused_op,
    float alpha, float beta,
    bool dst_is_bf16,
    int K, int N);

/// Pack B into Blocked K-contiguous (BKC) VNNI block layout.
/// Total size: K_padded × N_padded × sizeof(uint16_t) bytes (same as before).
/// \p col0  Starting column in B (0 = pack columns [0, N); else [col0, col0+N)).
void pack_b_bkc_ext(
    const uint16_t *B, int ldb, int K, int N, bool transB,
    uint16_t *packed,
    int col0 = 0);

/// Wide-block dispatch for NP=5,6 (separate CU to avoid i-cache pollution).
/// Called from bf16_gemv_bkc when block width is 384.
void bf16_gemv_bkc_wide_dispatch(
    const uint16_t *__restrict__ A,
    const uint16_t *__restrict__ B_bkc,
    uint16_t *__restrict__ C_bf16,
    float *__restrict__ C_fp32,
    const float *__restrict__ bias_f,
    fused_postop_t fused_op,
    float alpha, float beta,
    bool dst_is_bf16,
    int k_pairs, int n_stride, int K, int N,
    int jc, int nb);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
