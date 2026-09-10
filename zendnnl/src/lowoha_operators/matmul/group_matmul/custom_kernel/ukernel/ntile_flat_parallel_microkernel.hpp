/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

/**
 * @file ntile_flat_parallel_microkernel.hpp
 * @brief AVX-512 primitives for the W8A8 grouped-MoE fast path.
 *
 * Library-internal. Header-only on purpose: the executor and the unit tests
 * both need these, and keeping them inline preserves the whole schedule
 * (accumulators in registers, epilogue fused into the tile loop) that a call
 * across a translation-unit boundary would break.
 *
 * The block-VNNI weight layout is defined by the sibling pack module and
 * asserted independently by the unit tests.
 *
 * Nothing here includes or references a framework: bf16 is carried as
 * @c uint16_t storage and every entry point takes raw pointers.
 */

#ifndef LOWOHA_CUSTOM_KERNEL_UKERNEL_NTILE_FLAT_PARALLEL_MICROKERNEL_HPP
#define LOWOHA_CUSTOM_KERNEL_UKERNEL_NTILE_FLAT_PARALLEL_MICROKERNEL_HPP

#include "../ntile_flat_parallel_pack.hpp"

#if ZENDNNL_NTILE_FLAT_PARALLEL_KERNELS_COMPILED

#include <immintrin.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "common/zendnnl_compat.hpp"
#include "lowoha_operators/matmul/group_matmul/group_matmul_act_avx512.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {

// The gate/up epilogue's SiLU is the shared group_matmul activation
// primitive, the same one the generic/ALGO-3 paths use. Note the ISA lists
// in the target annotations below must remain a superset of the ones those
// helpers carry
// (`avx512f,avx512bw,avx512vl,fma`), or GCC declines to inline across the
// target-attribute boundary and emits a call in the middle of the tile loop.
using group_matmul_act_avx512::silu_avx512;

#define ZENDNNL_NTILE_FLAT_PARALLEL_TARGET_ISA \
    "avx512f,avx512bw,avx512dq,avx512vl,avx512vnni,avx512bf16,fma"

inline int64_t div_up(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

// Compile-time unrolled `for (i = 0; i < n; ++i) f(i, args...)`, so `i` stays
// a constant expression and the register indices in the micro-kernels resolve
// at compile time.
template <int n>
struct unroll_t {
    template <typename Func, typename... Args>
    ZENDNNL_INLINE_TARGET(ZENDNNL_NTILE_FLAT_PARALLEL_TARGET_ISA)
    inline void operator()(const Func &f, Args... args) const {
        unroll_t<n - 1> {}(f, args...);
        f(std::integral_constant<int, n - 1> {}, args...);
    }
};
template <>
struct unroll_t<1> {
    template <typename Func, typename... Args>
    ZENDNNL_INLINE_TARGET(ZENDNNL_NTILE_FLAT_PARALLEL_TARGET_ISA)
    inline void operator()(const Func &f, Args... args) const {
        f(std::integral_constant<int, 0> {}, args...);
    }
};

// ---------------------------------------------------------------------------
// Per-row symmetric activation quantization, bf16 -> uint8.
//
// The micro-kernels use vpdpbusd, whose A operand is unsigned, so the signed
// int8 value is biased by +128 here and the bias is removed in the epilogue by
// subtracting the weight compensation row. `K` is a multiple of 32.
// ---------------------------------------------------------------------------
ZENDNNL_TARGET(ZENDNNL_NTILE_FLAT_PARALLEL_TARGET_ISA)
inline void quantize_row_u8(uint8_t *ZENDNNL_NTILE_FLAT_PARALLEL_RESTRICT Aq,
        float &As, const uint16_t *ZENDNNL_NTILE_FLAT_PARALLEL_RESTRICT A,
        int64_t K) {
    const __m512 sign_bit = _mm512_set1_ps(-0.0f);
    const __m512i off = _mm512_set1_epi32(128);

    __m512 vamax0 = _mm512_set1_ps(0.f);
    __m512 vamax1 = _mm512_set1_ps(0.f);
    for (int64_t k = 0; k < K; k += 32) {
        const __m512i va
                = _mm512_loadu_si512(reinterpret_cast<const void *>(A + k));
        const __m512 va0 = _mm512_castsi512_ps(_mm512_slli_epi32(
                _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(va, 0)), 16));
        const __m512 va1 = _mm512_castsi512_ps(_mm512_slli_epi32(
                _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(va, 1)), 16));
        vamax0 = _mm512_max_ps(vamax0, _mm512_andnot_ps(sign_bit, va0));
        vamax1 = _mm512_max_ps(vamax1, _mm512_andnot_ps(sign_bit, va1));
    }
    float amax = _mm512_reduce_max_ps(_mm512_max_ps(vamax0, vamax1));
    amax = std::max(amax, 1e-7f);
    const float scale = amax / 127;
    const float inv_scale = 127 / amax;
    const __m512 vd = _mm512_set1_ps(inv_scale);

    for (int64_t k = 0; k < K; k += 32) {
        const __m512i va
                = _mm512_loadu_si512(reinterpret_cast<const void *>(A + k));
        __m512 va0 = _mm512_castsi512_ps(_mm512_slli_epi32(
                _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(va, 0)), 16));
        __m512 va1 = _mm512_castsi512_ps(_mm512_slli_epi32(
                _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(va, 1)), 16));
        va0 = _mm512_mul_ps(va0, vd);
        va1 = _mm512_mul_ps(va1, vd);
        va0 = _mm512_roundscale_ps(
                va0, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        va1 = _mm512_roundscale_ps(
                va1, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        const __m128i i0 = _mm512_cvtepi32_epi8(
                _mm512_add_epi32(_mm512_cvtps_epi32(va0), off));
        const __m128i i1 = _mm512_cvtepi32_epi8(
                _mm512_add_epi32(_mm512_cvtps_epi32(va1), off));
        _mm256_storeu_si256(
                reinterpret_cast<__m256i *>(Aq + k), _mm256_set_m128i(i1, i0));
    }
    As = scale;
}

// ---------------------------------------------------------------------------
// Gate/up micro-kernel: two independent int32 accumulator sets over a shared A
// operand, with silu(gate) * up folded into the epilogue so the intermediate
// never reaches memory as int32.
//
//   A     : [BLOCK_M, K]  biased uint8 or signed int8, row stride lda
//   B0/B1 : [K/4, 32, 4]  int8  (gate half / up half), k stride ldb * 4
//   C     : [BLOCK_M, N]  bf16, row stride ldc
//
// `SignedA=false` is the original stage-0 output encoding: each signed
// activation was already biased by +128 into uint8. `SignedA=true` consumes a
// caller-prequantized signed-S8 row and applies the identical bias to each
// four-byte broadcast with one XOR before VPDPBUSD. Both encodings therefore
// use the same packed-weight compensation row (128 * sum(weight)).
// ---------------------------------------------------------------------------
template <int BLOCK_M, int BLOCK_N, bool SignedA = false>
struct tiny_gemm_gate_up {
    ZENDNNL_INLINE_TARGET(ZENDNNL_NTILE_FLAT_PARALLEL_TARGET_ISA)
    static inline void apply(
            const uint8_t *ZENDNNL_NTILE_FLAT_PARALLEL_RESTRICT A,
            const int8_t *ZENDNNL_NTILE_FLAT_PARALLEL_RESTRICT B0,
            const int8_t *ZENDNNL_NTILE_FLAT_PARALLEL_RESTRICT B1,
            uint16_t *ZENDNNL_NTILE_FLAT_PARALLEL_RESTRICT C,
            const float *ZENDNNL_NTILE_FLAT_PARALLEL_RESTRICT As,
            const float *ZENDNNL_NTILE_FLAT_PARALLEL_RESTRICT Bs0,
            const float *ZENDNNL_NTILE_FLAT_PARALLEL_RESTRICT Bs1,
            const int32_t *ZENDNNL_NTILE_FLAT_PARALLEL_RESTRICT Bcomp0,
            const int32_t *ZENDNNL_NTILE_FLAT_PARALLEL_RESTRICT Bcomp1,
            int64_t K, int64_t lda, int64_t ldb, int64_t ldc) {
        constexpr int ROWS = BLOCK_M;
        constexpr int COLS = BLOCK_N / 16;
        static_assert(COLS % 2 == 0, "BLOCK_N must be a multiple of 32");

        __m512i va;
        alignas(64) __m512i vb0[COLS];
        alignas(64) __m512i vb1[COLS];
        alignas(64) __m512i vc0[ROWS * COLS];
        alignas(64) __m512i vc1[ROWS * COLS];
        alignas(64) __m512i vcomp0[COLS];
        alignas(64) __m512i vcomp1[COLS];
        __m512 vas;
        alignas(64) __m512 vbs0[COLS];
        alignas(64) __m512 vbs1[COLS];

        auto loadc
                = [&](auto i)
                          ZENDNNL_TARGET(ZENDNNL_NTILE_FLAT_PARALLEL_TARGET_ISA)
                                  ZENDNNL_LAMBDA_ALWAYS_INLINE {
            vc0[i] = _mm512_set1_epi32(0);
            vc1[i] = _mm512_set1_epi32(0);
        };
        unroll_t<ROWS * COLS> {}(loadc);

        const int64_t K4 = K >> 2;
        const int64_t ldb4 = ldb;
        const int32_t *b0_ptr = reinterpret_cast<const int32_t *>(B0);
        const int32_t *b1_ptr = reinterpret_cast<const int32_t *>(B1);

        // One broadcast of 4 packed activation bytes per row, one 64-byte B load
        // per stream per column, then ROWS * COLS * 2 vpdpbusd.
        auto compute
                = [&](auto i, int64_t k)
                          ZENDNNL_TARGET(ZENDNNL_NTILE_FLAT_PARALLEL_TARGET_ISA)
                                  ZENDNNL_LAMBDA_ALWAYS_INLINE {
            constexpr int row = i / COLS;
            constexpr int col = i % COLS;

            if constexpr (col == 0) {
                int32_t a_word;
                std::memcpy(&a_word,
                        A + static_cast<int64_t>(row) * lda + k * vnni_step,
                        sizeof(a_word));
                va = _mm512_set1_epi32(a_word);
                if constexpr (SignedA) {
                    constexpr int32_t s8_to_u8_bias
                            = static_cast<int32_t>(0x80808080U);
                    va = _mm512_xor_si512(va, _mm512_set1_epi32(s8_to_u8_bias));
                }
            }
            if constexpr (row == 0) {
                vb0[col] = _mm512_loadu_si512(reinterpret_cast<const void *>(
                        b0_ptr + k * ldb4 + col * 16));
                vb1[col] = _mm512_loadu_si512(reinterpret_cast<const void *>(
                        b1_ptr + k * ldb4 + col * 16));
            }
            vc0[i] = _mm512_dpbusd_epi32(vc0[i], va, vb0[col]);
            vc1[i] = _mm512_dpbusd_epi32(vc1[i], va, vb1[col]);
        };
        for (int64_t k = 0; k < K4; ++k) {
            unroll_t<ROWS * COLS> {}(compute, k);
        }

        // x = As * (acc0 - comp0) * Bs0 ; y = As * (acc1 - comp1) * Bs1
        auto scalec
                = [&](auto i)
                          ZENDNNL_TARGET(ZENDNNL_NTILE_FLAT_PARALLEL_TARGET_ISA)
                                  ZENDNNL_LAMBDA_ALWAYS_INLINE {
            constexpr int row = i / COLS;
            constexpr int col = i % COLS;

            if constexpr (col == 0) { vas = _mm512_set1_ps(As[row]); }
            if constexpr (row == 0) {
                vbs0[col] = _mm512_loadu_ps(Bs0 + col * 16);
                vbs1[col] = _mm512_loadu_ps(Bs1 + col * 16);
                vcomp0[col] = _mm512_loadu_si512(
                        reinterpret_cast<const void *>(Bcomp0 + col * 16));
                vcomp1[col] = _mm512_loadu_si512(
                        reinterpret_cast<const void *>(Bcomp1 + col * 16));
            }
            const __m512 c0
                    = _mm512_cvtepi32_ps(_mm512_sub_epi32(vc0[i], vcomp0[col]));
            const __m512 c1
                    = _mm512_cvtepi32_ps(_mm512_sub_epi32(vc1[i], vcomp1[col]));
            vc0[i] = _mm512_castps_si512(
                    _mm512_mul_ps(_mm512_mul_ps(c0, vas), vbs0[col]));
            vc1[i] = _mm512_castps_si512(
                    _mm512_mul_ps(_mm512_mul_ps(c1, vas), vbs1[col]));
        };
        unroll_t<ROWS * COLS> {}(scalec);

        // silu(x) * y for two 16-lane groups, packed into one 32-lane bf16
        // store.
        auto storec
                = [&](auto i)
                          ZENDNNL_TARGET(ZENDNNL_NTILE_FLAT_PARALLEL_TARGET_ISA)
                                  ZENDNNL_LAMBDA_ALWAYS_INLINE {
            constexpr int row = i / COLS;
            constexpr int col = i % COLS;
            if constexpr (col % 2 == 0) {
                __m512 x0 = _mm512_castsi512_ps(vc0[row * COLS + col + 0]);
                __m512 x1 = _mm512_castsi512_ps(vc0[row * COLS + col + 1]);
                const __m512 y0
                        = _mm512_castsi512_ps(vc1[row * COLS + col + 0]);
                const __m512 y1
                        = _mm512_castsi512_ps(vc1[row * COLS + col + 1]);
                x0 = _mm512_mul_ps(silu_avx512(x0), y0);
                x1 = _mm512_mul_ps(silu_avx512(x1), y1);
                _mm512_storeu_si512(
                        reinterpret_cast<void *>(C + row * ldc + col * 16),
                        (__m512i)(_mm512_cvtne2ps_pbh(x1, x0)));
            }
        };
        unroll_t<ROWS * COLS> {}(storec);
    }
};

// ---------------------------------------------------------------------------
// Down-projection micro-kernel: single accumulator set, bf16 output written
// straight to its final destination.
//
// The router weight is deliberately NOT applied here. This path hands its
// rows to the caller's existing weighted-reduce post-op, which reads them
// via `moe_postop->row_ptrs` and applies `topk_weights` itself -- the same
// division of labour the generic fused-MoE path uses. Writing bf16 directly
// therefore removes the f32 tile round-trip the routed variant needed to
// keep a pre-weighted value in flight.
// ---------------------------------------------------------------------------
template <int BLOCK_M, int BLOCK_N>
struct tiny_gemm_down {
    ZENDNNL_INLINE_TARGET(ZENDNNL_NTILE_FLAT_PARALLEL_TARGET_ISA)
    static inline void apply(
            const uint8_t *ZENDNNL_NTILE_FLAT_PARALLEL_RESTRICT A,
            const int8_t *ZENDNNL_NTILE_FLAT_PARALLEL_RESTRICT B,
            uint16_t *ZENDNNL_NTILE_FLAT_PARALLEL_RESTRICT C,
            const float *ZENDNNL_NTILE_FLAT_PARALLEL_RESTRICT As,
            const float *ZENDNNL_NTILE_FLAT_PARALLEL_RESTRICT Bs,
            const int32_t *ZENDNNL_NTILE_FLAT_PARALLEL_RESTRICT Bcomp,
            int64_t K, int64_t lda, int64_t ldb, int64_t ldc) {
        constexpr int ROWS = BLOCK_M;
        constexpr int COLS = BLOCK_N / 16;
        static_assert(COLS % 2 == 0, "BLOCK_N must be a multiple of 32");

        __m512i va;
        alignas(64) __m512i vb[COLS];
        alignas(64) __m512i vc[ROWS * COLS];
        alignas(64) __m512i vcomp[COLS];
        __m512 vas;
        alignas(64) __m512 vbs[COLS];

        auto loadc
                = [&](auto i)
                          ZENDNNL_TARGET(ZENDNNL_NTILE_FLAT_PARALLEL_TARGET_ISA)
                                  ZENDNNL_LAMBDA_ALWAYS_INLINE {
            vc[i] = _mm512_set1_epi32(0);
        };
        unroll_t<ROWS * COLS> {}(loadc);

        const int64_t K4 = K >> 2;
        const int64_t ldb4 = ldb;
        const int32_t *b_ptr = reinterpret_cast<const int32_t *>(B);

        auto compute
                = [&](auto i, int64_t k)
                          ZENDNNL_TARGET(ZENDNNL_NTILE_FLAT_PARALLEL_TARGET_ISA)
                                  ZENDNNL_LAMBDA_ALWAYS_INLINE {
            constexpr int row = i / COLS;
            constexpr int col = i % COLS;

            if constexpr (col == 0) {
                int32_t a_word;
                std::memcpy(&a_word,
                        A + static_cast<int64_t>(row) * lda + k * vnni_step,
                        sizeof(a_word));
                va = _mm512_set1_epi32(a_word);
            }
            if constexpr (row == 0) {
                vb[col] = _mm512_loadu_si512(reinterpret_cast<const void *>(
                        b_ptr + k * ldb4 + col * 16));
            }
            vc[i] = _mm512_dpbusd_epi32(vc[i], va, vb[col]);
        };
        for (int64_t k = 0; k < K4; ++k) {
            unroll_t<ROWS * COLS> {}(compute, k);
        }

        auto scalec
                = [&](auto i)
                          ZENDNNL_TARGET(ZENDNNL_NTILE_FLAT_PARALLEL_TARGET_ISA)
                                  ZENDNNL_LAMBDA_ALWAYS_INLINE {
            constexpr int row = i / COLS;
            constexpr int col = i % COLS;

            if constexpr (col == 0) { vas = _mm512_set1_ps(As[row]); }
            if constexpr (row == 0) {
                vbs[col] = _mm512_loadu_ps(Bs + col * 16);
                vcomp[col] = _mm512_loadu_si512(
                        reinterpret_cast<const void *>(Bcomp + col * 16));
            }
            __m512 x = _mm512_cvtepi32_ps(_mm512_sub_epi32(vc[i], vcomp[col]));
            x = _mm512_mul_ps(_mm512_mul_ps(x, vas), vbs[col]);
            vc[i] = _mm512_castps_si512(x);
        };
        unroll_t<ROWS * COLS> {}(scalec);

        auto storec
                = [&](auto i)
                          ZENDNNL_TARGET(ZENDNNL_NTILE_FLAT_PARALLEL_TARGET_ISA)
                                  ZENDNNL_LAMBDA_ALWAYS_INLINE {
            constexpr int row = i / COLS;
            constexpr int col = i % COLS;
            if constexpr (col % 2 == 0) {
                const __m512 x0 = _mm512_castsi512_ps(vc[row * COLS + col + 0]);
                const __m512 x1 = _mm512_castsi512_ps(vc[row * COLS + col + 1]);
                _mm512_storeu_si512(
                        reinterpret_cast<void *>(C + row * ldc + col * 16),
                        (__m512i)(_mm512_cvtne2ps_pbh(x1, x0)));
            }
        };
        unroll_t<ROWS * COLS> {}(storec);
    }
};

// Row-count dispatch. BLOCK_N is fixed at 32 by the caller, so only the row
// count varies and each instantiation keeps its accumulators in registers.
constexpr int64_t max_kernel_rows = 4;

template <bool SignedA = false>
ZENDNNL_TARGET(ZENDNNL_NTILE_FLAT_PARALLEL_TARGET_ISA)
inline void tinygemm_gate_up(
        const uint8_t *ZENDNNL_NTILE_FLAT_PARALLEL_RESTRICT A,
        const int8_t *ZENDNNL_NTILE_FLAT_PARALLEL_RESTRICT B0,
        const int8_t *ZENDNNL_NTILE_FLAT_PARALLEL_RESTRICT B1,
        uint16_t *ZENDNNL_NTILE_FLAT_PARALLEL_RESTRICT C,
        const float *ZENDNNL_NTILE_FLAT_PARALLEL_RESTRICT As,
        const float *ZENDNNL_NTILE_FLAT_PARALLEL_RESTRICT Bs0,
        const float *ZENDNNL_NTILE_FLAT_PARALLEL_RESTRICT Bs1,
        const int32_t *ZENDNNL_NTILE_FLAT_PARALLEL_RESTRICT Bcomp0,
        const int32_t *ZENDNNL_NTILE_FLAT_PARALLEL_RESTRICT Bcomp1, int64_t M,
        int64_t K, int64_t lda, int64_t ldb, int64_t ldc) {
    const int64_t MB = div_up(M, max_kernel_rows);
    for (int64_t mb = 0; mb < MB; ++mb) {
        const int64_t mb_start = mb * max_kernel_rows;
        const int64_t mb_size = std::min(max_kernel_rows, M - mb_start);
#define ZENDNNL_NTILE_FLAT_PARALLEL_GATE_UP(MS) \
    tiny_gemm_gate_up<MS, 32, SignedA>::apply(A + mb_start * lda, B0, B1, \
            C + mb_start * ldc, As + mb_start, Bs0, Bs1, Bcomp0, Bcomp1, K, \
            lda, ldb, ldc)
        switch (mb_size) {
            case 1: ZENDNNL_NTILE_FLAT_PARALLEL_GATE_UP(1); break;
            case 2: ZENDNNL_NTILE_FLAT_PARALLEL_GATE_UP(2); break;
            case 3: ZENDNNL_NTILE_FLAT_PARALLEL_GATE_UP(3); break;
            default: ZENDNNL_NTILE_FLAT_PARALLEL_GATE_UP(4); break;
        }
#undef ZENDNNL_NTILE_FLAT_PARALLEL_GATE_UP
    }
}

ZENDNNL_TARGET(ZENDNNL_NTILE_FLAT_PARALLEL_TARGET_ISA)
inline void tinygemm_down(const uint8_t *ZENDNNL_NTILE_FLAT_PARALLEL_RESTRICT A,
        const int8_t *ZENDNNL_NTILE_FLAT_PARALLEL_RESTRICT B,
        uint16_t *ZENDNNL_NTILE_FLAT_PARALLEL_RESTRICT C,
        const float *ZENDNNL_NTILE_FLAT_PARALLEL_RESTRICT As,
        const float *ZENDNNL_NTILE_FLAT_PARALLEL_RESTRICT Bs,
        const int32_t *ZENDNNL_NTILE_FLAT_PARALLEL_RESTRICT Bcomp, int64_t M,
        int64_t K, int64_t lda, int64_t ldb, int64_t ldc) {
    const int64_t MB = div_up(M, max_kernel_rows);
    for (int64_t mb = 0; mb < MB; ++mb) {
        const int64_t mb_start = mb * max_kernel_rows;
        const int64_t mb_size = std::min(max_kernel_rows, M - mb_start);
#define ZENDNNL_NTILE_FLAT_PARALLEL_DOWN(MS) \
    tiny_gemm_down<MS, 32>::apply(A + mb_start * lda, B, C + mb_start * ldc, \
            As + mb_start, Bs, Bcomp, K, lda, ldb, ldc)
        switch (mb_size) {
            case 1: ZENDNNL_NTILE_FLAT_PARALLEL_DOWN(1); break;
            case 2: ZENDNNL_NTILE_FLAT_PARALLEL_DOWN(2); break;
            case 3: ZENDNNL_NTILE_FLAT_PARALLEL_DOWN(3); break;
            default: ZENDNNL_NTILE_FLAT_PARALLEL_DOWN(4); break;
        }
#undef ZENDNNL_NTILE_FLAT_PARALLEL_DOWN
    }
}

#undef ZENDNNL_NTILE_FLAT_PARALLEL_TARGET_ISA

} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_NTILE_FLAT_PARALLEL_KERNELS_COMPILED

#endif // LOWOHA_CUSTOM_KERNEL_UKERNEL_NTILE_FLAT_PARALLEL_MICROKERNEL_HPP
