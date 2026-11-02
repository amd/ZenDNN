/********************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

/// HSTU attention's own custom-kernel (CK) GEMMs, one namespace and one
/// source file per data type (bf16/, int8/), laid out like
/// sdpa/sdpa_gemm_ck/ but separate from it: HSTU attention shares no
/// kernel code with SDPA. B (K^T or V) is packed per 32-column block
/// (`kNR`), in the layout the data type's dot-product instruction reads.

#ifndef LOWOHA_HSTU_CK_HPP
#define LOWOHA_HSTU_CK_HPP

#include <cstddef>
#include <cstdint>
#include "common/bfloat16.hpp"
#include "lowoha_operators/hstu_attention/hstu_gemm_ck/lowoha_hstu_ck_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace hstu_attention {
namespace ck {

/// BF16: blocks of [ceil(K/2)][32][2] (a K pair of each column side by
/// side, as `vdpbf16ps` reads them, zero padded past K and N); tiles of MR
/// rows x 32 FP32 accumulators fed by broadcast A pairs.
namespace bf16 {

using zendnnl::common::bfloat16_t;

/// Elements of a packed op(B) of K x N.
size_t packed_size(int64_t k, int64_t n);

/// Packs op(B) (K x N). `trans_b`: B is stored N x K (K^T from row-major K),
/// otherwise K x N (V); rows `ldb` elements apart.
void pack_b(int64_t k, int64_t n, const bfloat16_t *b, int64_t ldb,
        bool trans_b, bfloat16_t *packed);

/// C[M, nv] += A[M, K] * one packed block `b` (nv <= 32), FP32 C. `b` may
/// start at any pair of a block, so K can be a key window of it.
void gemm_block(int64_t m, int nv, int64_t k, const bfloat16_t *a, int64_t lda,
        const bfloat16_t *b, float *c, int64_t ldc);

} // namespace bf16

/// INT8: blocks of [ceil(K/4)][32][4] int8 (four K values of each column
/// side by side, as `vpdpbusd` reads them, zero padded past K and N);
/// tiles of MR rows x 32 int32 accumulators fed by broadcast U8 A quads.
namespace int8 {

/// Bytes of a packed op(B) of K x N.
size_t packed_size(int64_t k, int64_t n);

/// Packs S8 K^T from N x K row-major K (rows `ldk` bytes apart). With
/// `colsum` (round_up(N, 32) entries), writes the sum of each column over K
/// (0 past N).
void pack_kt(int64_t k, int64_t n, const int8_t *b, int64_t ldk, int8_t *packed,
        int32_t *colsum);

/// Packs S8 V from K x N row-major V (rows `ldv` bytes apart), with the
/// column sums as in `pack_kt`.
void pack_kn(int64_t k, int64_t n, const int8_t *b, int64_t ldv, int8_t *packed,
        int32_t *colsum);

/// The sums over K of the N <= 32 columns of one packed block (32 entries).
/// `b` may start at any quad of the block, so K can be a key window of it
/// that ends on a quad (K % 4 == 0) or at the block's last key.
void packed_colsum(int64_t k, int64_t n, const int8_t *b, int32_t *colsum);

/// C[M, nv] += (A * B - 128 * colsum[n]) * row_scale[m] * col_scale[n] for
/// U8 A (an S8 matrix plus 128; M x K, rows `lda` apart) and one packed
/// block `b` (nv <= 32, starting at any quad of the block). `colsum` covers
/// the same keys as A, and K ends on a quad or at the block's last key: A
/// reads as 0 past K in the last quad, which only the pack's zero padding
/// past the block's keys cancels.
void gemm_block(int64_t m, int nv, int64_t k, const uint8_t *a, int64_t lda,
        const int8_t *b, const int32_t *colsum, const float *row_scale,
        const float *col_scale, float *c, int64_t ldc);

} // namespace int8

} // namespace ck
} // namespace hstu_attention
} // namespace lowoha
} // namespace zendnnl

#endif // LOWOHA_HSTU_CK_HPP
