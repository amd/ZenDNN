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

} // namespace ck
} // namespace hstu_attention
} // namespace lowoha
} // namespace zendnnl

#endif // LOWOHA_HSTU_CK_HPP
