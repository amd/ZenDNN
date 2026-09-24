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
 * @file routed_moe_kernels.hpp
 * @brief AVX-512 primitives for the routed-MoE executor.
 *
 * Library-internal.  Header-only on purpose: the executor and the unit
 * tests both need these, and keeping them inline preserves the whole
 * schedule (accumulators in registers, activation gather fused into the
 * tile loop) that a call across a translation-unit boundary would break.
 *
 * The int8 VNNI weight layout, the micro-kernel accumulator/load
 * schedule, the exp/SiLU polynomial and the sorted/padded routing
 * ordering are reimplemented from the Apache-2.0 CPU fused-MoE kernels
 * in sgl-kernel and vLLM:
 *
 *   https://github.com/sgl-project/sglang/tree/main/sgl-kernel/csrc/cpu
 *   vllm/csrc/cpu/sgl-kernels/{gemm.cpp, moe.cpp, moe_int8.cpp, vec.h}
 *
 * Byte-for-byte layout compatibility with those kernels is a deliberate
 * property (and is asserted by the unit tests), so a host framework can
 * hand the same checkpoint to either implementation.
 *
 * Nothing here includes or references a framework: bf16 is carried as
 * @c uint16_t storage and every entry point takes raw pointers.
 */

#ifndef LOWOHA_ROUTED_MOE_KERNELS_HPP
#define LOWOHA_ROUTED_MOE_KERNELS_HPP

#include "lowoha_operators/matmul/routed_moe/routed_moe_internal.hpp"

#if ZENDNNL_ROUTED_MOE_KERNELS_COMPILED

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
namespace routed_moe {

// ---------------------------------------------------------------------------
// Blocking constants.
//
// block_n = 32 is the width the epilogue is written for: two 16-lane f32
// vectors combine into one 32-lane bf16 store, and the compensation / scale
// loads are two full zmm each.  block_m = 32 is the routing padding quantum;
// the micro-kernel itself is instantiated for 1..4 rows and iterates.
// ---------------------------------------------------------------------------
inline int64_t div_up(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

/// Amortized packed bytes per logical output channel.  Physical storage is
/// block-major; this value is for total-size and block-stride arithmetic.
inline int64_t packed_row_bytes(int64_t in_channels) {
    return packed_row_bytes_unchecked(in_channels);
}

// Compile-time unrolled `for (i = 0; i < n; ++i) f(i, args...)`, so `i` stays a
// constant expression and the register indices in the micro-kernels resolve at
// compile time.
template <int n>
struct unroll_t {
    template <typename Func, typename... Args>
    ZENDNNL_ALWAYS_INLINE inline void operator()(
            const Func &f, Args... args) const {
        unroll_t<n - 1> {}(f, args...);
        f(std::integral_constant<int, n - 1> {}, args...);
    }
};
template <>
struct unroll_t<1> {
    template <typename Func, typename... Args>
    ZENDNNL_ALWAYS_INLINE inline void operator()(
            const Func &f, Args... args) const {
        f(std::integral_constant<int, 0> {}, args...);
    }
};

// `fma` is listed so the shared group_matmul_act_avx512 helpers, which are
// declared for "avx512f,avx512bw,avx512vl[,avx512dq],fma", are a target subset
// of these kernels and therefore inline into the tile epilogue.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC target( \
        "avx512f,avx512bw,avx512dq,avx512vl,avx512vnni,avx512bf16,fma")
#elif defined(__clang__)
#pragma clang attribute push( \
        __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl," \
                              "avx512vnni,avx512bf16,fma"))), \
        apply_to = function)
#endif

// ---------------------------------------------------------------------------
// exp / sigmoid / SiLU
//
// Degree-5 polynomial on the ln(2) remainder plus exponent assembly, then
// sigmoid via rcp14 instead of a divide.  Reproduced in this exact form so
// SiLU matches the reference epilogue bit-for-bit.
// ---------------------------------------------------------------------------
ZENDNNL_ALWAYS_INLINE inline __m512 exp_u20_ps(const __m512 values) {
    const __m512 vec_factorial_1 = _mm512_set1_ps(0.999999701f);
    const __m512 vec_factorial_2 = _mm512_set1_ps(0.499991506f);
    const __m512 vec_factorial_3 = _mm512_set1_ps(0.166676521f);
    const __m512 vec_factorial_4 = _mm512_set1_ps(0.0418978221f);
    const __m512 vec_factorial_5 = _mm512_set1_ps(0.00828929059f);
    const __m512 vec_exp_log2ef
            = _mm512_castsi512_ps(_mm512_set1_epi32(0x3fb8aa3b)); // log2(e)
    const __m512 vec_half = _mm512_set1_ps(0.5f);
    const __m512 vec_one = _mm512_set1_ps(1.f);
    const __m512 vec_zero = _mm512_set1_ps(0.f);
    const __m512 vec_two = _mm512_set1_ps(2.f);
    const __m512 vec_ln2f = _mm512_castsi512_ps(_mm512_set1_epi32(0x3f317218));
    const __m512 vec_ln_flt_min
            = _mm512_castsi512_ps(_mm512_set1_epi32(0xc2aeac50));
    const __m512 vec_ln_flt_max
            = _mm512_castsi512_ps(_mm512_set1_epi32(0x42b17218));
    const __m512i vec_127 = _mm512_set1_epi32(0x0000007f);
    constexpr int n_mantissa_bits = 23;

    const __mmask16 less_ln_flt_min_mask
            = _mm512_cmp_ps_mask(values, vec_ln_flt_min, 1 /* _CMP_LT_OS */);
    __m512 vec_src = _mm512_min_ps(values, vec_ln_flt_max);
    vec_src = _mm512_max_ps(vec_src, vec_ln_flt_min);

    // fx = floorf(x * log2(e) + 0.5)
    __m512 vec_fx = _mm512_fmadd_ps(vec_src, vec_exp_log2ef, vec_half);
    const __m512i vec_fx_i = _mm512_cvt_roundps_epi32(
            vec_fx, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    vec_fx = _mm512_cvtepi32_ps(vec_fx_i);

    const __m512 vec_exp_poly = _mm512_fnmadd_ps(vec_fx, vec_ln2f, vec_src);

    __m512 vec_res
            = _mm512_fmadd_ps(vec_exp_poly, vec_factorial_5, vec_factorial_4);
    vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_3);
    vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_2);
    vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_1);
    vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_one);

    // 2^(n-1), then two multiplies so the exponent add cannot overflow to inf
    const __m512 vec_exp_number = _mm512_sub_ps(vec_fx, vec_one);
    const __m512i vec_exp_number_i = _mm512_cvtps_epi32(vec_exp_number);
    __m512i vec_two_pow_n_i = _mm512_add_epi32(vec_exp_number_i, vec_127);
    vec_two_pow_n_i = _mm512_slli_epi32(vec_two_pow_n_i, n_mantissa_bits);
    __m512 vec_two_pow_n = _mm512_castsi512_ps(vec_two_pow_n_i);
    vec_two_pow_n = _mm512_mask_blend_ps(
            less_ln_flt_min_mask, vec_two_pow_n, vec_zero);

    vec_res = _mm512_mul_ps(vec_res, vec_two_pow_n);
    vec_res = _mm512_mul_ps(vec_res, vec_two);
    return vec_res;
}

ZENDNNL_ALWAYS_INLINE inline __m512 silu_ps(__m512 x) {
    const __m512 minus_x = _mm512_xor_ps(_mm512_set1_ps(-0.f), x);
    const __m512 denom
            = _mm512_add_ps(exp_u20_ps(minus_x), _mm512_set1_ps(1.f));
    return _mm512_mul_ps(x, _mm512_rcp14_ps(denom));
}

// ---------------------------------------------------------------------------
// GELU
//
// grp_matmul_gated_act_t::gelu_and_mul is the erf form everywhere in the
// grouped-Matmul stack: the separate-pass row helpers and the custom-kernel
// in-register epilogue both call group_matmul_act_avx512::gelu_avx512.  The
// routed epilogue calls the same function so a call that falls back to the
// grouped path computes the identical activation.
// ---------------------------------------------------------------------------
ZENDNNL_ALWAYS_INLINE inline __m512 gelu_ps(__m512 x) {
    return group_matmul_act_avx512::gelu_avx512(x);
}

template <routed_moe_activation_t ACT>
ZENDNNL_ALWAYS_INLINE inline __m512 gated_act_ps(__m512 x) {
    static_assert(ACT == routed_moe_activation_t::silu_and_mul
                    || ACT == routed_moe_activation_t::gelu_and_mul,
            "activation has no routed epilogue");
    if constexpr (ACT == routed_moe_activation_t::gelu_and_mul) {
        return gelu_ps(x);
    } else {
        return silu_ps(x);
    }
}

// ---------------------------------------------------------------------------
// Per-row symmetric activation quantization, bf16 -> uint8.
//
// The micro-kernels use vpdpbusd, whose A operand is unsigned, so the signed
// int8 value is biased by +128 here and the bias is removed in the epilogue by
// subtracting the weight compensation row.  `K` is a multiple of 32.
// ---------------------------------------------------------------------------
inline void quantize_row_u8(uint8_t *ZENDNNL_ROUTED_RESTRICT Aq, float &As,
        const uint16_t *ZENDNNL_ROUTED_RESTRICT A, int64_t K) {
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
// Weight packing.
//
// For each block of block_n = 32 output channels the layout is
//
//   [K/4][32][4] int8 quants        (32 * K bytes)
//   [32]         int32 compensation (128 bytes)
//
// so one output-channel block occupies 32 * (K + 4) bytes and the whole tensor
// is [E, OC, K + 4].  The compensation entry for channel n is
// 128 * sum_k w[n][k]: exactly the bias vpdpbusd introduces by treating the
// +128-shifted activation as unsigned, so the epilogue removes it with a
// single vpsubd.
// ---------------------------------------------------------------------------
inline void pack_weight_block(int8_t *ZENDNNL_ROUTED_RESTRICT dst,
        const int8_t *ZENDNNL_ROUTED_RESTRICT src, int64_t K) {
    const int64_t K4 = K / vnni_step;

    // k-major so the stores walk the destination linearly; the 32 source rows
    // stay resident as 32 cache lines.
    for (int64_t k4 = 0; k4 < K4; ++k4) {
        uint32_t *ZENDNNL_ROUTED_RESTRICT d
                = reinterpret_cast<uint32_t *>(dst + k4 * block_n * vnni_step);
        for (int64_t n = 0; n < block_n; ++n) {
            uint32_t v;
            std::memcpy(&v, src + n * K + k4 * vnni_step, sizeof(uint32_t));
            d[n] = v;
        }
    }

    // compensation: accumulate 128 * w over k with vpdpbusd against a constant
    // 0x80 unsigned operand, mirroring the runtime bias exactly.
    constexpr int cols = block_n / 16;
    alignas(64) __m512i vcomp[cols];
    for (int col = 0; col < cols; ++col) {
        vcomp[col] = _mm512_setzero_si512();
    }
    const __m512i off = _mm512_set1_epi8(static_cast<char>(0x80));
    for (int64_t k4 = 0; k4 < K4; ++k4) {
        for (int col = 0; col < cols; ++col) {
            const __m512i vb
                    = _mm512_loadu_si512(reinterpret_cast<const void *>(
                            dst + k4 * block_n * vnni_step + col * 64));
            vcomp[col] = _mm512_dpbusd_epi32(vcomp[col], off, vb);
        }
    }
    for (int col = 0; col < cols; ++col) {
        _mm512_storeu_si512(
                reinterpret_cast<void *>(dst + block_n * K + col * 64),
                vcomp[col]);
    }
}

// ---------------------------------------------------------------------------
// Gate/up micro-kernel: two independent int32 accumulator sets over a shared A
// operand, with act(gate) * up folded into the epilogue so the intermediate
// never reaches memory as int32.
//
//   A     : [BLOCK_M, K]  uint8, row stride lda
//   B0/B1 : [K/4, 32, 4]  int8  (gate half / up half), k stride ldb * 4
//   C     : [BLOCK_M, N]  bf16, row stride ldc
// ---------------------------------------------------------------------------
template <int BLOCK_M, int BLOCK_N, routed_moe_activation_t ACT>
struct tiny_gemm_gate_up {
    static inline void apply(const uint8_t *ZENDNNL_ROUTED_RESTRICT A,
            const int8_t *ZENDNNL_ROUTED_RESTRICT B0,
            const int8_t *ZENDNNL_ROUTED_RESTRICT B1,
            uint16_t *ZENDNNL_ROUTED_RESTRICT C,
            const float *ZENDNNL_ROUTED_RESTRICT As,
            const float *ZENDNNL_ROUTED_RESTRICT Bs0,
            const float *ZENDNNL_ROUTED_RESTRICT Bs1,
            const int32_t *ZENDNNL_ROUTED_RESTRICT Bcomp0,
            const int32_t *ZENDNNL_ROUTED_RESTRICT Bcomp1, int64_t K,
            int64_t lda, int64_t ldb, int64_t ldc) {
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

        auto loadc = [&](auto i) {
            vc0[i] = _mm512_set1_epi32(0);
            vc1[i] = _mm512_set1_epi32(0);
        };
        unroll_t<ROWS * COLS> {}(loadc);

        const int64_t K4 = K >> 2;
        const int64_t lda4 = lda >> 2;
        const int64_t ldb4 = ldb;
        const int32_t *a_ptr = reinterpret_cast<const int32_t *>(A);
        const int32_t *b0_ptr = reinterpret_cast<const int32_t *>(B0);
        const int32_t *b1_ptr = reinterpret_cast<const int32_t *>(B1);

        // One broadcast of 4 packed activation bytes per row, one 64-byte B load
        // per stream per column, then ROWS * COLS * 2 vpdpbusd.
        auto compute = [&](auto i, int64_t k) {
            constexpr int row = i / COLS;
            constexpr int col = i % COLS;

            if constexpr (col == 0) {
                va = _mm512_set1_epi32(a_ptr[row * lda4 + k]);
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
        auto scalec = [&](auto i) {
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

        // act(x) * y for two 16-lane groups, packed into one 32-lane bf16 store.
        auto storec = [&](auto i) {
            constexpr int row = i / COLS;
            constexpr int col = i % COLS;
            if constexpr (col % 2 == 0) {
                __m512 x0 = _mm512_castsi512_ps(vc0[row * COLS + col + 0]);
                __m512 x1 = _mm512_castsi512_ps(vc0[row * COLS + col + 1]);
                const __m512 y0
                        = _mm512_castsi512_ps(vc1[row * COLS + col + 0]);
                const __m512 y1
                        = _mm512_castsi512_ps(vc1[row * COLS + col + 1]);
                x0 = _mm512_mul_ps(gated_act_ps<ACT>(x0), y0);
                x1 = _mm512_mul_ps(gated_act_ps<ACT>(x1), y1);
                _mm512_storeu_si512(
                        reinterpret_cast<void *>(C + row * ldc + col * 16),
                        (__m512i)(_mm512_cvtne2ps_pbh(x1, x0)));
            }
        };
        unroll_t<ROWS * COLS> {}(storec);
    }
};

// ---------------------------------------------------------------------------
// Down-projection micro-kernel: single accumulator set, f32 output kept in the
// tile buffer so the router weight can be applied during the scatter.
// ---------------------------------------------------------------------------
template <int BLOCK_M, int BLOCK_N>
struct tiny_gemm_down {
    static inline void apply(const uint8_t *ZENDNNL_ROUTED_RESTRICT A,
            const int8_t *ZENDNNL_ROUTED_RESTRICT B,
            float *ZENDNNL_ROUTED_RESTRICT C,
            const float *ZENDNNL_ROUTED_RESTRICT As,
            const float *ZENDNNL_ROUTED_RESTRICT Bs,
            const int32_t *ZENDNNL_ROUTED_RESTRICT Bcomp, int64_t K,
            int64_t lda, int64_t ldb, int64_t ldc) {
        constexpr int ROWS = BLOCK_M;
        constexpr int COLS = BLOCK_N / 16;
        static_assert(COLS % 2 == 0, "BLOCK_N must be a multiple of 32");

        __m512i va;
        alignas(64) __m512i vb[COLS];
        alignas(64) __m512i vc[ROWS * COLS];
        alignas(64) __m512i vcomp[COLS];
        __m512 vas;
        alignas(64) __m512 vbs[COLS];

        auto loadc = [&](auto i) { vc[i] = _mm512_set1_epi32(0); };
        unroll_t<ROWS * COLS> {}(loadc);

        const int64_t K4 = K >> 2;
        const int64_t lda4 = lda >> 2;
        const int64_t ldb4 = ldb;
        const int32_t *a_ptr = reinterpret_cast<const int32_t *>(A);
        const int32_t *b_ptr = reinterpret_cast<const int32_t *>(B);

        auto compute = [&](auto i, int64_t k) {
            constexpr int row = i / COLS;
            constexpr int col = i % COLS;

            if constexpr (col == 0) {
                va = _mm512_set1_epi32(a_ptr[row * lda4 + k]);
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

        auto storec = [&](auto i) {
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
            _mm512_storeu_ps(C + row * ldc + col * 16, x);
        };
        unroll_t<ROWS * COLS> {}(storec);
    }
};

// Row-count dispatch.  BLOCK_N is fixed at 32 by the caller, so only the row
// count varies and each instantiation keeps its accumulators in registers.
template <routed_moe_activation_t ACT>
inline void tinygemm_gate_up(const uint8_t *ZENDNNL_ROUTED_RESTRICT A,
        const int8_t *ZENDNNL_ROUTED_RESTRICT B0,
        const int8_t *ZENDNNL_ROUTED_RESTRICT B1,
        uint16_t *ZENDNNL_ROUTED_RESTRICT C,
        const float *ZENDNNL_ROUTED_RESTRICT As,
        const float *ZENDNNL_ROUTED_RESTRICT Bs0,
        const float *ZENDNNL_ROUTED_RESTRICT Bs1,
        const int32_t *ZENDNNL_ROUTED_RESTRICT Bcomp0,
        const int32_t *ZENDNNL_ROUTED_RESTRICT Bcomp1, int64_t M, int64_t K,
        int64_t lda, int64_t ldb, int64_t ldc) {
    const int64_t MB = div_up(M, max_kernel_rows);
    for (int64_t mb = 0; mb < MB; ++mb) {
        const int64_t mb_start = mb * max_kernel_rows;
        const int64_t mb_size = std::min(max_kernel_rows, M - mb_start);
#define ZENDNNL_RMOE_GATE_UP(MS) \
    tiny_gemm_gate_up<MS, 32, ACT>::apply(A + mb_start * lda, B0, B1, \
            C + mb_start * ldc, As + mb_start, Bs0, Bs1, Bcomp0, Bcomp1, K, \
            lda, ldb, ldc)
        switch (mb_size) {
            case 1: ZENDNNL_RMOE_GATE_UP(1); break;
            case 2: ZENDNNL_RMOE_GATE_UP(2); break;
            case 3: ZENDNNL_RMOE_GATE_UP(3); break;
            default: ZENDNNL_RMOE_GATE_UP(4); break;
        }
#undef ZENDNNL_RMOE_GATE_UP
    }
}

inline void tinygemm_down(const uint8_t *ZENDNNL_ROUTED_RESTRICT A,
        const int8_t *ZENDNNL_ROUTED_RESTRICT B,
        float *ZENDNNL_ROUTED_RESTRICT C,
        const float *ZENDNNL_ROUTED_RESTRICT As,
        const float *ZENDNNL_ROUTED_RESTRICT Bs,
        const int32_t *ZENDNNL_ROUTED_RESTRICT Bcomp, int64_t M, int64_t K,
        int64_t lda, int64_t ldb, int64_t ldc) {
    const int64_t MB = div_up(M, max_kernel_rows);
    for (int64_t mb = 0; mb < MB; ++mb) {
        const int64_t mb_start = mb * max_kernel_rows;
        const int64_t mb_size = std::min(max_kernel_rows, M - mb_start);
#define ZENDNNL_RMOE_DOWN(MS) \
    tiny_gemm_down<MS, 32>::apply(A + mb_start * lda, B, C + mb_start * ldc, \
            As + mb_start, Bs, Bcomp, K, lda, ldb, ldc)
        switch (mb_size) {
            case 1: ZENDNNL_RMOE_DOWN(1); break;
            case 2: ZENDNNL_RMOE_DOWN(2); break;
            case 3: ZENDNNL_RMOE_DOWN(3); break;
            default: ZENDNNL_RMOE_DOWN(4); break;
        }
#undef ZENDNNL_RMOE_DOWN
    }
}

/// out[0:size] = bf16(in[0:size] * weight), size a multiple of 32.
inline void copy_mul_bf16(uint16_t *ZENDNNL_ROUTED_RESTRICT out,
        const float *ZENDNNL_ROUTED_RESTRICT in, float weight, int64_t size) {
    const __m512 vw = _mm512_set1_ps(weight);
    for (int64_t d = 0; d < size; d += 32) {
        const __m512 x0 = _mm512_mul_ps(_mm512_loadu_ps(in + d), vw);
        const __m512 x1 = _mm512_mul_ps(_mm512_loadu_ps(in + d + 16), vw);
        _mm512_storeu_si512(reinterpret_cast<void *>(out + d),
                (__m512i)(_mm512_cvtne2ps_pbh(x1, x0)));
    }
}

/// out[0:K] = bf16(sum_t in[t * K + 0:K]), accumulating in f32.
inline void sum_rows_bf16(uint16_t *ZENDNNL_ROUTED_RESTRICT out,
        const uint16_t *ZENDNNL_ROUTED_RESTRICT in, int64_t topk, int64_t K) {
    for (int64_t d = 0; d < K; d += 32) {
        __m512 s0 = _mm512_setzero_ps();
        __m512 s1 = _mm512_setzero_ps();
        for (int64_t t = 0; t < topk; ++t) {
            const __m512i v = _mm512_loadu_si512(
                    reinterpret_cast<const void *>(in + t * K + d));
            s0 = _mm512_add_ps(s0,
                    _mm512_castsi512_ps(_mm512_slli_epi32(
                            _mm512_cvtepu16_epi32(
                                    _mm512_extracti32x8_epi32(v, 0)),
                            16)));
            s1 = _mm512_add_ps(s1,
                    _mm512_castsi512_ps(_mm512_slli_epi32(
                            _mm512_cvtepu16_epi32(
                                    _mm512_extracti32x8_epi32(v, 1)),
                            16)));
        }
        _mm512_storeu_si512(reinterpret_cast<void *>(out + d),
                (__m512i)(_mm512_cvtne2ps_pbh(s1, s0)));
    }
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#elif defined(__clang__)
#pragma clang attribute pop
#endif

} // namespace routed_moe
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_ROUTED_MOE_KERNELS_COMPILED

#endif // LOWOHA_ROUTED_MOE_KERNELS_HPP
