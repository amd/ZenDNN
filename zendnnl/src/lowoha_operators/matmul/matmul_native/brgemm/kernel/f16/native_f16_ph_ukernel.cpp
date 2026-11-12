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

#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/f16/native_f16_ph_ukernel.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <utility>

#include <immintrin.h>

#include "common/zendnnl_compat.hpp"

// The compilers whose headers have the AVX512-FP16 intrinsics, as
// common/float16.hpp takes them.
#if defined(ZENDNNL_HAS_AVX512FP16_MASK_LOAD_STORE_INTRINSICS) \
        || (defined(__GNUC__) && (__GNUC__ >= 12))
#define NATIVE_F16_PH_KERNELS 1
#else
#define NATIVE_F16_PH_KERNELS 0
#endif

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

#if NATIVE_F16_PH_KERNELS
namespace {

constexpr int kMaxMr = 12;

/// The f16 at a as the scalar of a broadcast: a copy of its bits.
ZENDNNL_INLINE_TARGET("avx512f,avx512bw,avx512vl,avx512fp16,fma")
inline _Float16 f16_at(const uint16_t *a) {
    _Float16 h;
    std::memcpy(&h, a, sizeof(h));
    return h;
}

/// The f16 accumulators of a row: 32 lanes at nv 2, 16 at nv 1.
template <int NV>
struct f16_acc;
template <>
struct f16_acc<1> {
    using type = __m256h;
};
template <>
struct f16_acc<2> {
    using type = __m512h;
};
template <int NV>
using f16_acc_t = typename f16_acc<NV>::type;

/// Row k of B into the accumulators: one f16 fused multiply-add per
/// element.
template <int MR, int NV>
ZENDNNL_INLINE_TARGET("avx512f,avx512bw,avx512vl,avx512fp16,fma")
inline void fma_row(f16_acc_t<NV> (&acc)[MR], const uint16_t *__restrict a,
        int lda, const uint16_t *__restrict b) {
    if constexpr (NV == 2) {
        const __m512h bv = _mm512_loadu_ph(b);
        for (int m = 0; m < MR; ++m)
            acc[m] = _mm512_fmadd_ph(
                    _mm512_set1_ph(f16_at(a + static_cast<size_t>(m) * lda)),
                    bv, acc[m]);
    } else {
        const __m256h bv = _mm256_loadu_ph(b);
        for (int m = 0; m < MR; ++m)
            acc[m] = _mm256_fmadd_ph(
                    _mm256_set1_ph(f16_at(a + static_cast<size_t>(m) * lda)),
                    bv, acc[m]);
    }
}

/// c[0 .. 15] = s0 + lanes 0 .. 15 of acc widened to f32, c[16 .. 31] =
/// s1 + lanes 16 .. 31.
ZENDNNL_INLINE_TARGET("avx512f,avx512bw,avx512vl,avx512fp16,fma")
inline void add_block32(__m512h acc, __m512 s0, __m512 s1, float *c) {
    const __m512i bits = _mm512_castph_si512(acc);
    _mm512_storeu_ps(c,
            _mm512_add_ps(s0, _mm512_cvtph_ps(_mm512_castsi512_si256(bits))));
    _mm512_storeu_ps(c + 16,
            _mm512_add_ps(
                    s1, _mm512_cvtph_ps(_mm512_extracti64x4_epi64(bits, 1))));
}

/// c[v * 16 ..] = s[v] + the 16 lanes v of acc widened to f32.
template <int NV>
ZENDNNL_INLINE_TARGET("avx512f,avx512bw,avx512vl,avx512fp16,fma")
inline void add_block(f16_acc_t<NV> acc, const __m512 (&s)[NV], float *c) {
    if constexpr (NV == 2) {
        add_block32(acc, s[0], s[1], c);
    } else {
        _mm512_storeu_ps(c,
                _mm512_add_ps(s[0], _mm512_cvtph_ps(_mm256_castph_si256(acc))));
    }
}

template <int MR, int NV>
ZENDNNL_TARGET_NOINLINE("avx512f,avx512bw,avx512vl,avx512fp16,fma")
void native_f16_ph_ukernel(const uint16_t *__restrict A, int lda,
        const uint16_t *__restrict B, int b_stride, size_t, float *__restrict C,
        int ldc, int K, float beta) {
    // A block ends at its start plus what is left: the start plus a block
    // may pass the int bound.
    for (int k0 = 0, kb = 0; k0 < K; k0 += kb) {
        kb = std::min(kNativeF16AccumBlockK, K - k0);
        f16_acc_t<NV> acc[MR];
        for (int m = 0; m < MR; ++m) {
            if constexpr (NV == 2)
                acc[m] = _mm512_setzero_ph();
            else
                acc[m] = _mm256_setzero_ph();
        }
        const uint16_t *a = A + k0;
        const uint16_t *b = B + static_cast<size_t>(k0) * b_stride;
        int kk = 0;
        for (; kk + 3 < kb; kk += 4)
            for (int u = 0; u < 4; ++u)
                fma_row<MR, NV>(acc, a + kk + u, lda,
                        b + static_cast<size_t>(kk + u) * b_stride);
        for (; kk < kb; ++kk)
            fma_row<MR, NV>(
                    acc, a + kk, lda, b + static_cast<size_t>(kk) * b_stride);
        for (int m = 0; m < MR; ++m) {
            float *c = C + static_cast<size_t>(m) * ldc;
            __m512 s[NV];
            for (int v = 0; v < NV; ++v) {
                if (k0 > 0)
                    s[v] = _mm512_loadu_ps(c + v * 16);
                else if (beta != 0.0f)
                    s[v] = _mm512_mul_ps(
                            _mm512_set1_ps(beta), _mm512_loadu_ps(c + v * 16));
                else
                    s[v] = _mm512_setzero_ps();
            }
            add_block<NV>(acc[m], s, c);
        }
    }
}

/// native_f16_ph_ukernel at nv 4: per row an f16 accumulator for each of
/// the tile's two 32-column blocks, both fed by one broadcast of the row's
/// source element.  The loops are unrolled before the accumulators go to
/// registers: else the compiler keeps the B vectors in memory.
template <int MR>
ZENDNNL_TARGET_NOINLINE("avx512f,avx512bw,avx512vl,avx512fp16,fma")
void native_f16_ph_ukernel_nv4(const uint16_t *__restrict A, int lda,
        const uint16_t *__restrict B, int b_stride, size_t b_next,
        float *__restrict C, int ldc, int K, float beta) {
    // A block ends at its start plus what is left: the start plus a block
    // may pass the int bound.
    for (int k0 = 0, kb = 0; k0 < K; k0 += kb) {
        kb = std::min(kNativeF16AccumBlockK, K - k0);
        __m512h lo[MR], hi[MR];
#pragma GCC unroll 12
        for (int m = 0; m < MR; ++m) {
            lo[m] = _mm512_setzero_ph();
            hi[m] = _mm512_setzero_ph();
        }
        const uint16_t *a = A + k0;
        const uint16_t *b = B + static_cast<size_t>(k0) * b_stride;
        for (int k = 0; k < kb; ++k) {
            const uint16_t *bk = b + static_cast<size_t>(k) * b_stride;
            const __m512h b0 = _mm512_loadu_ph(bk);
            const __m512h b1 = _mm512_loadu_ph(bk + b_next);
#pragma GCC unroll 12
            for (int m = 0; m < MR; ++m) {
                const __m512h av = _mm512_set1_ph(
                        f16_at(a + static_cast<size_t>(m) * lda + k));
                lo[m] = _mm512_fmadd_ph(av, b0, lo[m]);
                hi[m] = _mm512_fmadd_ph(av, b1, hi[m]);
            }
        }
#pragma GCC unroll 12
        for (int m = 0; m < MR; ++m) {
            float *c = C + static_cast<size_t>(m) * ldc;
            __m512 s[4];
#pragma GCC unroll 4
            for (int v = 0; v < 4; ++v) {
                if (k0 > 0)
                    s[v] = _mm512_loadu_ps(c + v * 16);
                else if (beta != 0.0f)
                    s[v] = _mm512_mul_ps(
                            _mm512_set1_ps(beta), _mm512_loadu_ps(c + v * 16));
                else
                    s[v] = _mm512_setzero_ps();
            }
            add_block32(lo[m], s[0], s[1], c);
            add_block32(hi[m], s[2], s[3], c + 32);
        }
    }
}

template <int NV, int... I>
constexpr std::array<native_f16_ph_ukernel_fn_t, sizeof...(I)> make_table(
        std::integer_sequence<int, I...>) {
    return {{&native_f16_ph_ukernel<I + 1, NV>...}};
}

template <int... I>
constexpr std::array<native_f16_ph_ukernel_fn_t, sizeof...(I)> make_nv4_table(
        std::integer_sequence<int, I...>) {
    return {{&native_f16_ph_ukernel_nv4<I + 1>...}};
}

constexpr auto kNv1 = make_table<1>(std::make_integer_sequence<int, kMaxMr> {});
constexpr auto kNv2 = make_table<2>(std::make_integer_sequence<int, kMaxMr> {});
constexpr auto kNv4
        = make_nv4_table(std::make_integer_sequence<int, kMaxMr> {});

} // namespace
#endif

int native_f16_ph_ukernel_max_mr(int nv) {
#if NATIVE_F16_PH_KERNELS
    return nv == 1 || nv == 2 || nv == 4 ? kMaxMr : 0;
#else
    (void)nv;
    return 0;
#endif
}

native_f16_ph_ukernel_fn_t select_native_f16_ph_ukernel(int mr, int nv) {
    if (mr < 1 || mr > native_f16_ph_ukernel_max_mr(nv)) return nullptr;
#if NATIVE_F16_PH_KERNELS
    return (nv == 1 ? kNv1 : (nv == 2 ? kNv2 : kNv4))[mr - 1];
#else
    return nullptr;
#endif
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
