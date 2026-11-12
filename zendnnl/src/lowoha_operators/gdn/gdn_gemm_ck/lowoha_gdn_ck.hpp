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

/// GDN's own custom-kernel (CK) GEMMs for the chunked prefill, generic over
/// the operand type: one tiling and packing structure
/// (lowoha_gdn_ck_kernel.hpp) and one translation unit per operand type
/// (bf16/, f32/) with its micro-kernel and conversions, laid out like
/// hstu_attention/hstu_gemm_ck/ but separate from it: GDN shares no kernel
/// code with SDPA or HSTU attention. All shapes are small (a chunk of 32
/// tokens against head dims of 32 to 256) and each call runs on the calling
/// thread.
///
/// Every GEMM computes C[M][N] (+)= A B with FP32 C and FP32 accumulation, in
/// tiles of up to 4 rows x 64 columns. B is stored as [K / kPack][N][kPack]
/// with N a multiple of 32; A is read one K step (kPack values) at a time
/// through a row and a step stride, so a matrix and its transpose can share
/// one copy.

#ifndef LOWOHA_GDN_CK_HPP
#define LOWOHA_GDN_CK_HPP

#include <cstdint>
#include "common/data_types.hpp"

namespace zendnnl {
namespace lowoha {
namespace gdn {
namespace ck {

using zendnnl::common::data_type_t;

/// How operand type DT is stored and multiplied.
template <data_type_t DT>
struct operand;

/// bf16 on `vdpbf16ps`: an A step is a pair of K values (one 32-bit word) and
/// B is in the pair layout [K / 2][N][2] (the two K values of a column side
/// by side), rounded to nearest even.
template <>
struct operand<data_type_t::bf16> {
    using a_elem = uint16_t; ///< element of a stored A
    using a_step = uint32_t; ///< one K step of A
    using b_elem = uint16_t;
    static constexpr int64_t kPack = 2;
};

/// FP32 FMA; B row-major [K][N].
template <>
struct operand<data_type_t::f32> {
    using a_elem = float;
    using a_step = float;
    using b_elem = float;
    static constexpr int64_t kPack = 1;
};

template <data_type_t DT>
using a_elem_t = typename operand<DT>::a_elem;
template <data_type_t DT>
using a_step_t = typename operand<DT>::a_step;
template <data_type_t DT>
using b_elem_t = typename operand<DT>::b_elem;

/// A operand from n FP32 values (n a multiple of 16), row-major like the
/// source: bf16 rounds, f32 copies.
template <data_type_t DT>
void to_a(const float *src, int64_t n, a_elem_t<DT> *dst);

/// B operand [R / kPack][N][kPack] of FP32 [R][N] (rows `ld` apart, R a
/// multiple of kPack, N a multiple of 16).
template <data_type_t DT>
void pack_b(
        const float *src, int64_t r, int64_t n, int64_t ld, b_elem_t<DT> *dst);

/// B operand of the transpose of a row-major A operand [R][K] (R a multiple
/// of 16, K a multiple of kPack): dst step (k, r) = src step (r, k), i.e.
/// [K / kPack][R][kPack].
template <data_type_t DT>
void pack_b_transposed(
        const a_elem_t<DT> *src, int64_t r, int64_t k, b_elem_t<DT> *dst);

/// Triangles a product may skip: with lower_a, A(m, k) is zero for k > m, so
/// a row tile stops at its last row's K step; with lower_c, only C(m, n) for
/// n <= m is wanted, so a column block skips the row tiles above it (those
/// entries of C are left as they were). Skipping exact zeros leaves every
/// computed entry as the full product gives it.
enum class tri_t : int32_t { full, lower_a, lower_c };

/// C[M][N] (+)= sum over ks K steps of A(m, p) . B[p][n], FP32 C with rows
/// `ldc` apart. The A step (m, p) is a[m * a_row + p * a_step]: a row-major
/// A operand [M][K] has a_row = K / kPack and a_step = 1, and a B operand
/// [K / kPack][M][kPack] read as its transpose has a_row = 1 and a_step = M.
template <data_type_t DT>
void gemm(int64_t m, int64_t n, int64_t ks, const a_step_t<DT> *a,
        int64_t a_row, int64_t a_step, const b_elem_t<DT> *b, float *c,
        int64_t ldc, bool accumulate, tri_t tri = tri_t::full);

/// gemm() on a row-major A operand [M][K].
template <data_type_t DT>
inline void gemm_rows(int64_t m, int64_t n, int64_t k, const a_elem_t<DT> *a,
        const b_elem_t<DT> *b, float *c, int64_t ldc, bool accumulate,
        tri_t tri = tri_t::full) {
    constexpr int64_t P = operand<DT>::kPack;
    gemm<DT>(m, n, k / P, reinterpret_cast<const a_step_t<DT> *>(a), k / P, 1,
            b, c, ldc, accumulate, tri);
}

// Specialised in bf16/ and f32/.
#define GDN_CK_DECLARE(DT) \
    template <> \
    void to_a<DT>(const float *src, int64_t n, a_elem_t<DT> *dst); \
    template <> \
    void pack_b<DT>(const float *src, int64_t r, int64_t n, int64_t ld, \
            b_elem_t<DT> *dst); \
    template <> \
    void pack_b_transposed<DT>( \
            const a_elem_t<DT> *src, int64_t r, int64_t k, b_elem_t<DT> *dst); \
    template <> \
    void gemm<DT>(int64_t m, int64_t n, int64_t ks, const a_step_t<DT> *a, \
            int64_t a_row, int64_t a_step, const b_elem_t<DT> *b, float *c, \
            int64_t ldc, bool accumulate, tri_t tri);
GDN_CK_DECLARE(data_type_t::bf16)
GDN_CK_DECLARE(data_type_t::f32)
#undef GDN_CK_DECLARE

} // namespace ck
} // namespace gdn
} // namespace lowoha
} // namespace zendnnl

#endif // LOWOHA_GDN_CK_HPP
