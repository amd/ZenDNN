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

/// Building blocks of HSTU attention's custom kernels (CK): the packed
/// block width, the ISA targets and the inner GEMM steps, shared by the CK
/// GEMMs (`lowoha_hstu_ck.hpp`) and the fused Q*K^T tile kernels of the
/// tiled kernel policies. Internal to hstu_attention.

#ifndef LOWOHA_HSTU_CK_COMMON_HPP
#define LOWOHA_HSTU_CK_COMMON_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include "common/bfloat16.hpp"
#include "common/zendnnl_global.hpp"

#define HSTU_AVX512_INLINE \
    ZENDNNL_INLINE_TARGET("avx512f,avx512bw,avx512vl,avx512bf16,fma")
#define HSTU_AVX512_NOINLINE \
    ZENDNNL_TARGET_NOINLINE("avx512f,avx512bw,avx512vl,avx512bf16,fma")

#define HSTU_I8_ISA "avx512f,avx512bw,avx512vl,avx512bf16,avx512vnni,fma"
#define HSTU_I8_INLINE ZENDNNL_INLINE_TARGET(HSTU_I8_ISA)
#define HSTU_I8_NOINLINE ZENDNNL_TARGET_NOINLINE(HSTU_I8_ISA)

namespace zendnnl {
namespace lowoha {
namespace hstu_attention {

/// Columns of a packed block.
constexpr int kNR = 32;

/// acc[MR][2 NB] += A[:, 2q : 2q + 2] * pair q of NB packed blocks `bstride`
/// elements apart; `single` reads only A[:, 2q] (odd K: the pack zeroed the
/// second half of the last pair, and A may end at 2q).
template <int MR, int NB>
HSTU_AVX512_INLINE inline void pair_step(__m512 (&acc)[MR][2 * NB],
        const zendnnl::common::bfloat16_t *__restrict a, int64_t lda,
        const zendnnl::common::bfloat16_t *__restrict b, int64_t bstride,
        int64_t q, bool single) {
    __m512bh bv[2 * NB];
#pragma GCC unroll 2
    for (int j = 0; j < NB; ++j) {
        bv[2 * j] = (__m512bh)_mm512_loadu_si512(b + j * bstride + q * 2 * kNR);
        bv[2 * j + 1] = (__m512bh)_mm512_loadu_si512(
                b + j * bstride + q * 2 * kNR + kNR);
    }
#pragma GCC unroll 8
    for (int m = 0; m < MR; ++m) {
        uint32_t pair = 0;
        std::memcpy(&pair, a + m * lda + 2 * q, single ? 2 : 4);
        const __m512bh av = (__m512bh)_mm512_set1_epi32(int(pair));
#pragma GCC unroll 4
        for (int j = 0; j < 2 * NB; ++j)
            acc[m][j] = _mm512_dpbf16_ps(acc[m][j], av, bv[j]);
    }
}

/// acc += U8 a * S8 b (vpdpbusd). GCC copies the accumulator around the
/// intrinsic in unrolled loops; tying it keeps it in place.
HSTU_I8_INLINE inline __m512i dpbusd(__m512i acc, __m512i a, __m512i b) {
#if defined(__GNUC__) && !defined(__clang__)
    __asm__("vpdpbusd %2, %1, %0" : "+v"(acc) : "v"(a), "v"(b));
    return acc;
#else
    return _mm512_dpbusd_epi32(acc, a, b);
#endif
}

/// acc[MR][2 NB] += A[:, 4q : 4q + 4] * quad q of NB packed blocks `bstride`
/// bytes apart.
template <int MR, int NB>
HSTU_I8_INLINE inline void quad_step(__m512i (&acc)[MR][2 * NB],
        const uint8_t *__restrict a, int64_t lda, const int8_t *__restrict b,
        int64_t bstride, int64_t q) {
    __m512i bv[2 * NB];
#pragma GCC unroll 2
    for (int j = 0; j < NB; ++j) {
        bv[2 * j] = _mm512_loadu_si512(b + j * bstride + q * 4 * kNR);
        bv[2 * j + 1] = _mm512_loadu_si512(b + j * bstride + q * 4 * kNR + 64);
    }
#pragma GCC unroll 8
    for (int m = 0; m < MR; ++m) {
        uint32_t quad;
        std::memcpy(&quad, a + m * lda + 4 * q, sizeof(quad));
        const __m512i av = _mm512_set1_epi32(int(quad));
#pragma GCC unroll 4
        for (int j = 0; j < 2 * NB; ++j)
            acc[m][j] = dpbusd(acc[m][j], av, bv[j]);
    }
}

} // namespace hstu_attention
} // namespace lowoha
} // namespace zendnnl

#endif // LOWOHA_HSTU_CK_COMMON_HPP
