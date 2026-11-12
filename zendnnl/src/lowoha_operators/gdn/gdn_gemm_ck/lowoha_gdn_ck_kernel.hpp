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

/// The tiling and packing structure of GDN's CK GEMMs, shared by every
/// operand type. Internal to gdn_gemm_ck: each translation unit (bf16/,
/// f32/) defines the micro-kernel and conversions of its type and
/// instantiates the entry points of lowoha_gdn_ck.hpp from these templates.
///
/// A micro-kernel M provides
///   a_t, b_t, kPack       the A step, the B element and the B pack factor;
///   load_b(p)             16 columns x kPack values of B;
///   bcast_a(p)            one A step broadcast to every lane;
///   madd(acc, a, b)       acc + a . b in FP32;
///   to_a16(src, dst)      16 FP32 values to A elements;
///   pack16(src, ld, dst)  16 columns of kPack FP32 rows (`ld` apart) to B.

#ifndef LOWOHA_GDN_CK_KERNEL_HPP
#define LOWOHA_GDN_CK_KERNEL_HPP

#include <algorithm>
#include <immintrin.h>
#include "lowoha_operators/gdn/gdn_gemm_ck/lowoha_gdn_ck.hpp"
#include "lowoha_operators/gdn/kernel/gdn_avx512_utils.hpp"

GDN_AVX512_BEGIN

namespace zendnnl {
namespace lowoha {
namespace gdn {
namespace ck {
namespace kernel {

/// The full tile: kMR rows x kNV vectors of 16 FP32 accumulators (64
/// columns). A 64-column block needs four B vectors per K step for four A
/// broadcasts, which keeps Zen5's two FMA pipes busier than 6 x 32 did.
constexpr int kMR = 4;
constexpr int kNV = 4;

/// FP32 FMA on FP32 operands.
struct fma_micro {
    using a_t = float;
    using b_t = float;
    static constexpr int64_t kPack = 1;
    static inline __m512 load_b(const float *p) { return _mm512_loadu_ps(p); }
    static inline __m512 bcast_a(const float *p) { return _mm512_set1_ps(*p); }
    static inline __m512 madd(__m512 acc, __m512 a, __m512 b) {
        return _mm512_fmadd_ps(a, b, acc);
    }
};

template <typename M, int MR, int NV>
inline void gemm_tile(const typename M::a_t *a, int64_t a_row, int64_t a_step,
        const typename M::b_t *b, int64_t n, int64_t ks, int64_t n0, float *c,
        int64_t ldc, bool accumulate) {
    __m512 acc[MR][NV];
    for (int r = 0; r < MR; ++r) {
        for (int v = 0; v < NV; ++v) {
            acc[r][v] = accumulate ? _mm512_loadu_ps(c + r * ldc + n0 + 16 * v)
                                   : _mm512_setzero_ps();
        }
    }
    for (int64_t p = 0; p < ks; ++p) {
        const typename M::b_t *bp = b + (p * n + n0) * M::kPack;
        decltype(M::load_b(bp)) bv[NV];
        for (int v = 0; v < NV; ++v) {
            bv[v] = M::load_b(bp + 16 * v * M::kPack);
        }
        for (int r = 0; r < MR; ++r) {
            const auto av = M::bcast_a(a + r * a_row + p * a_step);
            for (int v = 0; v < NV; ++v) {
                acc[r][v] = M::madd(acc[r][v], av, bv[v]);
            }
        }
    }
    for (int r = 0; r < MR; ++r) {
        for (int v = 0; v < NV; ++v) {
            _mm512_storeu_ps(c + r * ldc + n0 + 16 * v, acc[r][v]);
        }
    }
}

/// K steps a tile of `rows` rows from row m0 needs: all of them, or with A
/// lower triangular those up to its last row.
template <typename M>
inline int64_t tile_steps(int64_t ks, int64_t m0, int64_t rows, tri_t tri) {
    return tri == tri_t::lower_a ? std::min(ks, (m0 + rows - 1) / M::kPack + 1)
                                 : ks;
}

/// The column block [n0, n0 + 16 NV): rows in tiles of kMR, then 2 and 1.
template <typename M, int NV>
inline void gemm_block(int64_t m, int64_t n, int64_t ks,
        const typename M::a_t *a, int64_t a_row, int64_t a_step,
        const typename M::b_t *b, float *c, int64_t ldc, bool accumulate,
        tri_t tri, int64_t n0) {
    // With C lower triangular, the row tiles above the block need none of
    // it; the tiles keep their kMR grid.
    int64_t m0 = tri == tri_t::lower_c ? n0 / kMR * kMR : 0;
    for (; m0 + kMR <= m; m0 += kMR) {
        gemm_tile<M, kMR, NV>(a + m0 * a_row, a_row, a_step, b, n,
                tile_steps<M>(ks, m0, kMR, tri), n0, c + m0 * ldc, ldc,
                accumulate);
    }
    if (m0 + 2 <= m) {
        gemm_tile<M, 2, NV>(a + m0 * a_row, a_row, a_step, b, n,
                tile_steps<M>(ks, m0, 2, tri), n0, c + m0 * ldc, ldc,
                accumulate);
        m0 += 2;
    }
    if (m0 < m) {
        gemm_tile<M, 1, NV>(a + m0 * a_row, a_row, a_step, b, n,
                tile_steps<M>(ks, m0, 1, tri), n0, c + m0 * ldc, ldc,
                accumulate);
    }
}

/// Column blocks of 64, then one of 32 when N is an odd multiple of 32.
template <typename M>
void gemm(int64_t m, int64_t n, int64_t ks, const typename M::a_t *a,
        int64_t a_row, int64_t a_step, const typename M::b_t *b, float *c,
        int64_t ldc, bool accumulate, tri_t tri) {
    int64_t n0 = 0;
    for (; n0 + 16 * kNV <= n; n0 += 16 * kNV) {
        gemm_block<M, kNV>(
                m, n, ks, a, a_row, a_step, b, c, ldc, accumulate, tri, n0);
    }
    if (n0 < n) {
        gemm_block<M, 2>(
                m, n, ks, a, a_row, a_step, b, c, ldc, accumulate, tri, n0);
    }
}

template <typename M, typename AE>
void to_a(const float *src, int64_t n, AE *dst) {
    for (int64_t i = 0; i < n; i += 16) {
        M::to_a16(src + i, dst + i);
    }
}

template <typename M>
void pack_b(const float *src, int64_t r, int64_t n, int64_t ld,
        typename M::b_t *dst) {
    for (int64_t g = 0; g < r / M::kPack; ++g) {
        const float *x = src + g * M::kPack * ld;
        typename M::b_t *d = dst + g * n * M::kPack;
        for (int64_t j = 0; j < n; j += 16) {
            M::pack16(x + j, ld, d + j * M::kPack);
        }
    }
}

/// Transposes the K steps of a row-major A operand [R][K]: every step is one
/// 32-bit word (a bf16 pair or an FP32 value), so one gather serves all types.
template <typename M, typename AE>
void pack_b_transposed(
        const AE *src, int64_t r, int64_t k, typename M::b_t *dst) {
    static_assert(sizeof(AE) * M::kPack == 4, "A steps must be 32-bit words");
    const int64_t kw = k / M::kPack;
    const int32_t *s = reinterpret_cast<const int32_t *>(src);
    int32_t *d = reinterpret_cast<int32_t *>(dst);
    const __m512i rows
            = _mm512_mullo_epi32(_mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8,
                                         7, 6, 5, 4, 3, 2, 1, 0),
                    _mm512_set1_epi32(static_cast<int>(kw)));
    for (int64_t c = 0; c < kw; ++c) {
        for (int64_t i = 0; i < r; i += 16) {
            _mm512_storeu_si512(d + c * r + i,
                    _mm512_i32gather_epi32(rows, s + i * kw + c, 4));
        }
    }
}

} // namespace kernel
} // namespace ck
} // namespace gdn
} // namespace lowoha
} // namespace zendnnl

GDN_AVX512_END

#endif // LOWOHA_GDN_CK_KERNEL_HPP
