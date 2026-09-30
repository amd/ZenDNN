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
// the micro-kernels are instantiated for up to gate_up_kernel_rows /
// down_kernel_rows rows and iterate over row groups.
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
// K loops of the micro-kernels, for 32 output columns (two zmm per row).
//
// The loop-carried accumulators are named scalars, each pinned by an empty asm
// after its update: GCC otherwise keeps every accumulator in a second register
// and copies it around each vpdpbusd (or, for a pinned array element, stores
// it back to the stack), which leaves the compute-bound (prefill) loop front-
// end bound.  One broadcast of 4 packed activation bytes per row and one
// 64-byte B load per stream per column feed the vpdpbusd; the int32 sums over
// k are exact, so the results match any other loop schedule.
// ---------------------------------------------------------------------------
template <int ROWS>
ZENDNNL_ALWAYS_INLINE inline void gate_up_k_loop(
        const int32_t *ZENDNNL_ROUTED_RESTRICT a_ptr,
        const int32_t *ZENDNNL_ROUTED_RESTRICT b0_ptr,
        const int32_t *ZENDNNL_ROUTED_RESTRICT b1_ptr, int64_t K4, int64_t lda4,
        int64_t ldb4, __m512i *vc0, __m512i *vc1) {
    static_assert(ROWS >= 1 && ROWS <= 6, "gate/up K loop covers 1..6 rows");
    const __m512i zero = _mm512_setzero_si512();
    __m512i g00 = zero, g01 = zero, g10 = zero, g11 = zero, g20 = zero,
            g21 = zero, g30 = zero, g31 = zero, g40 = zero, g41 = zero,
            g50 = zero, g51 = zero;
    __m512i u00 = zero, u01 = zero, u10 = zero, u11 = zero, u20 = zero,
            u21 = zero, u30 = zero, u31 = zero, u40 = zero, u41 = zero,
            u50 = zero, u51 = zero;
    for (int64_t k = 0; k < K4; ++k) {
        const __m512i x0 = _mm512_loadu_si512(
                reinterpret_cast<const void *>(b0_ptr + k * ldb4));
        const __m512i x1 = _mm512_loadu_si512(
                reinterpret_cast<const void *>(b0_ptr + k * ldb4 + 16));
        const __m512i y0 = _mm512_loadu_si512(
                reinterpret_cast<const void *>(b1_ptr + k * ldb4));
        const __m512i y1 = _mm512_loadu_si512(
                reinterpret_cast<const void *>(b1_ptr + k * ldb4 + 16));
#define ZENDNNL_RMOE_ROW(r) \
    if constexpr (ROWS > r) { \
        const __m512i va = _mm512_set1_epi32(a_ptr[r * lda4 + k]); \
        g##r##0 = _mm512_dpbusd_epi32(g##r##0, va, x0); \
        g##r##1 = _mm512_dpbusd_epi32(g##r##1, va, x1); \
        u##r##0 = _mm512_dpbusd_epi32(u##r##0, va, y0); \
        u##r##1 = _mm512_dpbusd_epi32(u##r##1, va, y1); \
        asm("" : "+v"(g##r##0), "+v"(g##r##1), "+v"(u##r##0), "+v"(u##r##1)); \
    }
        ZENDNNL_RMOE_ROW(0)
        ZENDNNL_RMOE_ROW(1)
        ZENDNNL_RMOE_ROW(2)
        ZENDNNL_RMOE_ROW(3)
        ZENDNNL_RMOE_ROW(4)
        ZENDNNL_RMOE_ROW(5)
#undef ZENDNNL_RMOE_ROW
    }
#define ZENDNNL_RMOE_OUT(r) \
    if constexpr (ROWS > r) { \
        vc0[2 * r] = g##r##0; \
        vc0[2 * r + 1] = g##r##1; \
        vc1[2 * r] = u##r##0; \
        vc1[2 * r + 1] = u##r##1; \
    }
    ZENDNNL_RMOE_OUT(0)
    ZENDNNL_RMOE_OUT(1)
    ZENDNNL_RMOE_OUT(2)
    ZENDNNL_RMOE_OUT(3)
    ZENDNNL_RMOE_OUT(4)
    ZENDNNL_RMOE_OUT(5)
#undef ZENDNNL_RMOE_OUT
}

template <int ROWS>
ZENDNNL_ALWAYS_INLINE inline void down_k_loop(
        const int32_t *ZENDNNL_ROUTED_RESTRICT a_ptr,
        const int32_t *ZENDNNL_ROUTED_RESTRICT b_ptr, int64_t K4, int64_t lda4,
        int64_t ldb4, __m512i *vc) {
    static_assert(ROWS >= 1 && ROWS <= 8, "down K loop covers 1..8 rows");
    const __m512i zero = _mm512_setzero_si512();
    __m512i c00 = zero, c01 = zero, c10 = zero, c11 = zero, c20 = zero,
            c21 = zero, c30 = zero, c31 = zero, c40 = zero, c41 = zero,
            c50 = zero, c51 = zero, c60 = zero, c61 = zero, c70 = zero,
            c71 = zero;
    for (int64_t k = 0; k < K4; ++k) {
        const __m512i b0 = _mm512_loadu_si512(
                reinterpret_cast<const void *>(b_ptr + k * ldb4));
        const __m512i b1 = _mm512_loadu_si512(
                reinterpret_cast<const void *>(b_ptr + k * ldb4 + 16));
#define ZENDNNL_RMOE_ROW(r) \
    if constexpr (ROWS > r) { \
        const __m512i va = _mm512_set1_epi32(a_ptr[r * lda4 + k]); \
        c##r##0 = _mm512_dpbusd_epi32(c##r##0, va, b0); \
        c##r##1 = _mm512_dpbusd_epi32(c##r##1, va, b1); \
        asm("" : "+v"(c##r##0), "+v"(c##r##1)); \
    }
        ZENDNNL_RMOE_ROW(0)
        ZENDNNL_RMOE_ROW(1)
        ZENDNNL_RMOE_ROW(2)
        ZENDNNL_RMOE_ROW(3)
        ZENDNNL_RMOE_ROW(4)
        ZENDNNL_RMOE_ROW(5)
        ZENDNNL_RMOE_ROW(6)
        ZENDNNL_RMOE_ROW(7)
#undef ZENDNNL_RMOE_ROW
    }
#define ZENDNNL_RMOE_OUT(r) \
    if constexpr (ROWS > r) { \
        vc[2 * r] = c##r##0; \
        vc[2 * r + 1] = c##r##1; \
    }
    ZENDNNL_RMOE_OUT(0)
    ZENDNNL_RMOE_OUT(1)
    ZENDNNL_RMOE_OUT(2)
    ZENDNNL_RMOE_OUT(3)
    ZENDNNL_RMOE_OUT(4)
    ZENDNNL_RMOE_OUT(5)
    ZENDNNL_RMOE_OUT(6)
    ZENDNNL_RMOE_OUT(7)
#undef ZENDNNL_RMOE_OUT
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
        static_assert(COLS == 2, "gate_up_k_loop is written for BLOCK_N == 32");

        alignas(64) __m512i vc0[ROWS * COLS];
        alignas(64) __m512i vc1[ROWS * COLS];
        alignas(64) __m512i vcomp0[COLS];
        alignas(64) __m512i vcomp1[COLS];
        __m512 vas;
        alignas(64) __m512 vbs0[COLS];
        alignas(64) __m512 vbs1[COLS];

        gate_up_k_loop<ROWS>(reinterpret_cast<const int32_t *>(A),
                reinterpret_cast<const int32_t *>(B0),
                reinterpret_cast<const int32_t *>(B1), K >> 2, lda >> 2, ldb,
                vc0, vc1);

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
        static_assert(COLS == 2, "down_k_loop is written for BLOCK_N == 32");

        alignas(64) __m512i vc[ROWS * COLS];
        alignas(64) __m512i vcomp[COLS];
        __m512 vas;
        alignas(64) __m512 vbs[COLS];

        down_k_loop<ROWS>(reinterpret_cast<const int32_t *>(A),
                reinterpret_cast<const int32_t *>(B), K >> 2, lda >> 2, ldb,
                vc);

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

// Splits M rows into the fewest passes of at most MAX_ROWS rows, sized as
// evenly as possible, so no pass is left with a row or two and too few
// accumulator chains.  Each row accumulates independently, so the split never
// changes a result.
template <int64_t MAX_ROWS, typename Launch>
ZENDNNL_ALWAYS_INLINE inline void for_row_groups(
        int64_t M, const Launch &launch) {
    const int64_t groups = div_up(M, MAX_ROWS);
    int64_t start = 0;
    for (int64_t g = 0; g < groups; ++g) {
        const int64_t rows = div_up(M - start, groups - g);
        launch(start, rows);
        start += rows;
    }
}

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
    static_assert(gate_up_kernel_rows == 6, "update the dispatch below");
    for_row_groups<gate_up_kernel_rows>(M, [&](int64_t start, int64_t rows) {
#define ZENDNNL_RMOE_GATE_UP(MS) \
    tiny_gemm_gate_up<MS, 32, ACT>::apply(A + start * lda, B0, B1, \
            C + start * ldc, As + start, Bs0, Bs1, Bcomp0, Bcomp1, K, lda, \
            ldb, ldc)
        switch (rows) {
            case 1: ZENDNNL_RMOE_GATE_UP(1); break;
            case 2: ZENDNNL_RMOE_GATE_UP(2); break;
            case 3: ZENDNNL_RMOE_GATE_UP(3); break;
            case 4: ZENDNNL_RMOE_GATE_UP(4); break;
            case 5: ZENDNNL_RMOE_GATE_UP(5); break;
            default: ZENDNNL_RMOE_GATE_UP(6); break;
        }
#undef ZENDNNL_RMOE_GATE_UP
    });
}

inline void tinygemm_down(const uint8_t *ZENDNNL_ROUTED_RESTRICT A,
        const int8_t *ZENDNNL_ROUTED_RESTRICT B,
        float *ZENDNNL_ROUTED_RESTRICT C,
        const float *ZENDNNL_ROUTED_RESTRICT As,
        const float *ZENDNNL_ROUTED_RESTRICT Bs,
        const int32_t *ZENDNNL_ROUTED_RESTRICT Bcomp, int64_t M, int64_t K,
        int64_t lda, int64_t ldb, int64_t ldc) {
    static_assert(down_kernel_rows == 8, "update the dispatch below");
    for_row_groups<down_kernel_rows>(M, [&](int64_t start, int64_t rows) {
#define ZENDNNL_RMOE_DOWN(MS) \
    tiny_gemm_down<MS, 32>::apply(A + start * lda, B, C + start * ldc, \
            As + start, Bs, Bcomp, K, lda, ldb, ldc)
        switch (rows) {
            case 1: ZENDNNL_RMOE_DOWN(1); break;
            case 2: ZENDNNL_RMOE_DOWN(2); break;
            case 3: ZENDNNL_RMOE_DOWN(3); break;
            case 4: ZENDNNL_RMOE_DOWN(4); break;
            case 5: ZENDNNL_RMOE_DOWN(5); break;
            case 6: ZENDNNL_RMOE_DOWN(6); break;
            case 7: ZENDNNL_RMOE_DOWN(7); break;
            default: ZENDNNL_RMOE_DOWN(8); break;
        }
#undef ZENDNNL_RMOE_DOWN
    });
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

/// acc[0:size] += in[0:size] * weight in f32, size a multiple of 16.
inline void add_mul_f32(float *ZENDNNL_ROUTED_RESTRICT acc,
        const float *ZENDNNL_ROUTED_RESTRICT in, float weight, int64_t size) {
    const __m512 vw = _mm512_set1_ps(weight);
    for (int64_t d = 0; d < size; d += 16) {
        _mm512_storeu_ps(acc + d,
                _mm512_fmadd_ps(
                        _mm512_loadu_ps(in + d), vw, _mm512_loadu_ps(acc + d)));
    }
}

/// out[0:size] = bf16(in[0:size]), size a multiple of 32.
inline void f32_to_bf16_row(uint16_t *ZENDNNL_ROUTED_RESTRICT out,
        const float *ZENDNNL_ROUTED_RESTRICT in, int64_t size) {
    for (int64_t d = 0; d < size; d += 32) {
        _mm512_storeu_si512(reinterpret_cast<void *>(out + d),
                (__m512i)(_mm512_cvtne2ps_pbh(_mm512_loadu_ps(in + d + 16),
                        _mm512_loadu_ps(in + d))));
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
