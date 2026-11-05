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

#ifndef MATMUL_NATIVE_INTRINSIC_AVX512_MATH_HPP
#define MATMUL_NATIVE_INTRINSIC_AVX512_MATH_HPP

#include <algorithm>
#include <cmath>
#include <immintrin.h>
#include "common/zendnnl_compat.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

/// Fused post-op types that can be applied inside the microkernel epilogue.
enum class fused_postop_t : int {
    none = 0,
    relu,
    gelu_tanh,
    gelu_erf,
    sigmoid,
    swish,
    tanh_op
};

// ============================================================================
// AVX-512 vectorized transcendental approximations for FP32
// Used by both the microkernel epilogue and the standalone postop pass.
// ============================================================================

/// The arguments avx512_exp clamps to: below the first it returns the e^x of
/// the first, above the second the e^x of the second.
constexpr float kAvx512ExpArgLo = -87.3f;
constexpr float kAvx512ExpArgHi = 88.3f;

ZENDNNL_TARGET("avx512f,fma")
static inline __m512 avx512_exp(__m512 x) {
    const __m512 log2e = _mm512_set1_ps(1.44269504089f);
    const __m512 ln2_hi = _mm512_set1_ps(0.693359375f);
    const __m512 ln2_lo = _mm512_set1_ps(-2.12194440e-4f);
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 c1 = _mm512_set1_ps(1.9875691500e-4f);
    const __m512 c2 = _mm512_set1_ps(1.3981999507e-3f);
    const __m512 c3 = _mm512_set1_ps(8.3334519073e-3f);
    const __m512 c4 = _mm512_set1_ps(4.1665795894e-2f);
    const __m512 c5 = _mm512_set1_ps(1.6666665459e-1f);
    const __m512 c6 = _mm512_set1_ps(5.0000001201e-1f);

    x = _mm512_max_ps(x, _mm512_set1_ps(kAvx512ExpArgLo));
    x = _mm512_min_ps(x, _mm512_set1_ps(kAvx512ExpArgHi));

    __m512 fx = _mm512_mul_ps(x, log2e);
    __m512 n = _mm512_roundscale_ps(fx, _MM_FROUND_TO_NEAREST_INT);
    x = _mm512_fnmadd_ps(n, ln2_hi, x);
    x = _mm512_fnmadd_ps(n, ln2_lo, x);

    __m512 y = _mm512_fmadd_ps(c1, x, c2);
    y = _mm512_fmadd_ps(y, x, c3);
    y = _mm512_fmadd_ps(y, x, c4);
    y = _mm512_fmadd_ps(y, x, c5);
    y = _mm512_fmadd_ps(y, x, c6);
    y = _mm512_fmadd_ps(y, _mm512_mul_ps(x, x), _mm512_add_ps(x, one));

    __m512i ni = _mm512_cvtps_epi32(n);
    ni = _mm512_slli_epi32(ni, 23);
    __m512 pow2n = _mm512_castsi512_ps(
            _mm512_add_epi32(ni, _mm512_castps_si512(one)));
    return _mm512_mul_ps(y, pow2n);
}

ZENDNNL_TARGET("avx512f,fma")
static inline __m512 avx512_tanh(__m512 x) {
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 two = _mm512_set1_ps(2.0f);
    const __m512 neg_one = _mm512_set1_ps(-1.0f);
    __m512 exp2x = avx512_exp(_mm512_mul_ps(two, x));
    __m512 result
            = _mm512_sub_ps(one, _mm512_div_ps(two, _mm512_add_ps(one, exp2x)));
    result = _mm512_max_ps(result, neg_one);
    result = _mm512_min_ps(result, one);
    return result;
}

ZENDNNL_TARGET("avx512f,fma")
static inline __m512 avx512_sigmoid(__m512 x) {
    const __m512 one = _mm512_set1_ps(1.0f);
    __m512 neg_x = _mm512_sub_ps(_mm512_setzero_ps(), x);
    return _mm512_div_ps(one, _mm512_add_ps(one, avx512_exp(neg_x)));
}

ZENDNNL_TARGET("avx512f,fma")
static inline __m512 avx512_erf(__m512 x) {
    const __m512 a1 = _mm512_set1_ps(0.254829592f);
    const __m512 a2 = _mm512_set1_ps(-0.284496736f);
    const __m512 a3 = _mm512_set1_ps(1.421413741f);
    const __m512 a4 = _mm512_set1_ps(-1.453152027f);
    const __m512 a5 = _mm512_set1_ps(1.061405429f);
    const __m512 p = _mm512_set1_ps(0.3275911f);
    const __m512 one = _mm512_set1_ps(1.0f);

    __m512 abs_x = _mm512_abs_ps(x);
    __mmask16 neg_mask = _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_LT_OQ);
    __m512 sign = _mm512_mask_sub_ps(one, neg_mask, _mm512_setzero_ps(), one);

    __m512 t = _mm512_div_ps(one, _mm512_fmadd_ps(p, abs_x, one));
    __m512 poly = _mm512_fmadd_ps(a5, t, a4);
    poly = _mm512_fmadd_ps(poly, t, a3);
    poly = _mm512_fmadd_ps(poly, t, a2);
    poly = _mm512_fmadd_ps(poly, t, a1);

    __m512 neg_x2
            = _mm512_sub_ps(_mm512_setzero_ps(), _mm512_mul_ps(abs_x, abs_x));
    __m512 exp_neg_x2 = avx512_exp(neg_x2);
    __m512 result = _mm512_sub_ps(
            one, _mm512_mul_ps(_mm512_mul_ps(t, poly), exp_neg_x2));
    return _mm512_mul_ps(sign, result);
}

/// v with each lane of lanes replaced by f of the same lane of x.
template <typename F>
ZENDNNL_TARGET("avx512f")
static inline __m512
        avx512_scalar_lanes(__m512 v, __m512 x, __mmask16 lanes, F f) {
    alignas(64) float out[16];
    alignas(64) float in[16];
    _mm512_store_ps(out, v);
    _mm512_store_ps(in, x);
    for (int l = 0; l < 16; ++l)
        if ((lanes >> l) & 1) out[l] = f(in[l]);
    return _mm512_load_ps(out);
}

/// v with each lane of lanes replaced by f of the same lanes of a and b.
template <typename F>
ZENDNNL_TARGET("avx512f")
static inline __m512 avx512_scalar_lanes(
        __m512 v, __m512 a, __m512 b, __mmask16 lanes, F f) {
    alignas(64) float out[16];
    alignas(64) float in_a[16];
    alignas(64) float in_b[16];
    _mm512_store_ps(out, v);
    _mm512_store_ps(in_a, a);
    _mm512_store_ps(in_b, b);
    for (int l = 0; l < 16; ++l)
        if ((lanes >> l) & 1) out[l] = f(in_a[l], in_b[l]);
    return _mm512_load_ps(out);
}

/// mish(x) = x * tanh(log(1 + e^x)) = x * n / (n + 2) with n = e^x (e^x + 2).
/// From x = 20 on, n / (n + 2) rounds to 1, so e^x is evaluated at min(x, 20).
/// Below kAvx512ExpArgLo, x times the clamped e^x is not mish(x): those
/// lanes take the scalar formula of the post-op reference.
ZENDNNL_TARGET("avx512f,fma")
static inline __m512 avx512_mish(__m512 x) {
    const __m512 two = _mm512_set1_ps(2.0f);
    __m512 e = avx512_exp(_mm512_min_ps(x, _mm512_set1_ps(20.0f)));
    __m512 n = _mm512_mul_ps(e, _mm512_add_ps(e, two));
    const __m512 y = _mm512_mul_ps(x, _mm512_div_ps(n, _mm512_add_ps(n, two)));
    const __mmask16 tail = _mm512_cmp_ps_mask(
            x, _mm512_set1_ps(kAvx512ExpArgLo), _CMP_LT_OQ);
    if (tail == 0) return y;
    return avx512_scalar_lanes(y, x, tail, [](float v) {
        return v
                * std::tanh(std::fmax(v, 0.0f)
                        + std::log1p(std::exp(-std::fabs(v))));
    });
}

/// silu(x) = x * sigmoid(x).  Below -kAvx512ExpArgHi, x times the sigmoid of
/// the clamped e^-x is not silu(x): those lanes take the scalar formula of
/// the gated activation reference.
ZENDNNL_TARGET("avx512f,fma")
static inline __m512 avx512_silu(__m512 x) {
    const __m512 y = _mm512_mul_ps(x, avx512_sigmoid(x));
    const __mmask16 tail = _mm512_cmp_ps_mask(
            x, _mm512_set1_ps(-kAvx512ExpArgHi), _CMP_LT_OQ);
    if (tail == 0) return y;
    return avx512_scalar_lanes(y, x, tail,
            [](float v) { return v * (1.0f / (1.0f + std::exp(-v))); });
}

/// swish(x) = x * sigmoid(alpha * x).  Where alpha * x is below
/// -kAvx512ExpArgHi, x times the sigmoid of the clamped exponential is not
/// swish(x), and where it is NaN avx512_sigmoid returns a number: those
/// lanes take the scalar formula of the post-op reference, which divides in
/// double.
ZENDNNL_TARGET("avx512f,fma")
static inline __m512 avx512_swish(__m512 x, float alpha) {
    const __m512 ax = _mm512_mul_ps(x, _mm512_set1_ps(alpha));
    const __m512 y = _mm512_mul_ps(x, avx512_sigmoid(ax));
    const __mmask16 tail = _mm512_cmp_ps_mask(
            ax, _mm512_set1_ps(-kAvx512ExpArgHi), _CMP_NGE_UQ);
    if (tail == 0) return y;
    return avx512_scalar_lanes(y, x, tail, [alpha](float v) {
        return static_cast<float>(v * (1.0 / (1.0 + std::exp(-(v * alpha)))));
    });
}

/// gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2))).
ZENDNNL_TARGET("avx512f,fma")
static inline __m512 avx512_gelu_erf(__m512 x) {
    const __m512 e
            = avx512_erf(_mm512_mul_ps(x, _mm512_set1_ps(0.7071067811865476f)));
    return _mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(0.5f), x),
            _mm512_add_ps(_mm512_set1_ps(1.0f), e));
}

/// swiglu_oai(g, u) = (1 + u') * g' * sigmoid(1.702 * g'), with g' and u'
/// clamped to [-7, 7].  The gated activation reference clamps with std::min
/// then std::max, which turns a NaN into the lower bound where the vector min
/// and max give the upper one: lanes with a NaN gate or up take its scalar
/// formula.
ZENDNNL_TARGET("avx512f,fma")
static inline __m512 avx512_swiglu_oai(__m512 gate, __m512 up) {
    const __m512 lo = _mm512_set1_ps(-7.0f);
    const __m512 hi = _mm512_set1_ps(7.0f);
    const __m512 g = _mm512_max_ps(_mm512_min_ps(gate, hi), lo);
    const __m512 u = _mm512_max_ps(_mm512_min_ps(up, hi), lo);
    const __m512 sig = avx512_sigmoid(_mm512_mul_ps(g, _mm512_set1_ps(1.702f)));
    const __m512 y = _mm512_mul_ps(
            _mm512_add_ps(_mm512_set1_ps(1.0f), u), _mm512_mul_ps(g, sig));
    const __mmask16 nans = _mm512_cmp_ps_mask(gate, up, _CMP_UNORD_Q);
    if (nans == 0) return y;
    return avx512_scalar_lanes(y, gate, up, nans, [](float a, float b) {
        const float gc = std::max(-7.0f, std::min(a, 7.0f));
        const float uc = std::max(-7.0f, std::min(b, 7.0f));
        return (1.0f + uc) * gc * (1.0f / (1.0f + std::exp(-(gc * 1.702f))));
    });
}

/// a * b rounded on its own: a compiler that contracts a multiply and the
/// add that follows it into one fused multiply-add must not fuse this one.
#if defined(_MSC_VER) && !defined(__clang__)
ZENDNNL_TARGET_NOINLINE("avx512f")
static inline __m512 avx512_mul_unfused(__m512 a, __m512 b) {
    return _mm512_mul_ps(a, b);
}
#else
ZENDNNL_TARGET("avx512f")
static inline __m512 avx512_mul_unfused(__m512 a, __m512 b) {
    __m512 p = _mm512_mul_ps(a, b);
    asm volatile("" : "+v"(p));
    return p;
}
#endif

/// 16 bf16 values widened to f32.
ZENDNNL_TARGET("avx512f")
static inline __m512 avx512_bf16_to_f32(__m256i v) {
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(v), 16));
}

/// 16 f32 values rounded to bf16, to nearest even, by integer arithmetic:
/// subnormals are kept whatever the MXCSR mode, and a NaN stays a quiet NaN
/// of its sign (the rounding would carry a low payload into Inf or 0).
ZENDNNL_TARGET("avx512f")
static inline __m256i avx512_f32_to_bf16_rne(__m512 v) {
    const __m512i bits = _mm512_castps_si512(v);
    const __m512i lsb = _mm512_and_si512(
            _mm512_srli_epi32(bits, 16), _mm512_set1_epi32(1));
    const __m512i bias = _mm512_add_epi32(_mm512_set1_epi32(0x7FFF), lsb);
    const __m512i rounded = _mm512_srli_epi32(_mm512_add_epi32(bits, bias), 16);
    const __mmask16 nan = _mm512_cmp_ps_mask(v, v, _CMP_UNORD_Q);
    return _mm512_cvtepi32_epi16(_mm512_mask_or_epi32(rounded, nan,
            _mm512_srli_epi32(bits, 16), _mm512_set1_epi32(0x40)));
}

/// Apply a fused post-op to a single ZMM register.
ZENDNNL_TARGET("avx512f,fma")
static inline __m512 apply_fused_postop(__m512 v, fused_postop_t op) {
    switch (op) {
        case fused_postop_t::relu: return _mm512_max_ps(v, _mm512_setzero_ps());
        case fused_postop_t::gelu_tanh: {
            const __m512 half = _mm512_set1_ps(0.5f);
            const __m512 one = _mm512_set1_ps(1.0f);
            const __m512 c = _mm512_set1_ps(0.7978845608028654f);
            const __m512 c2 = _mm512_set1_ps(0.044715f);
            __m512 x3 = _mm512_mul_ps(_mm512_mul_ps(v, v), v);
            __m512 inner = _mm512_mul_ps(c, _mm512_fmadd_ps(c2, x3, v));
            return _mm512_mul_ps(half,
                    _mm512_mul_ps(v, _mm512_add_ps(one, avx512_tanh(inner))));
        }
        case fused_postop_t::gelu_erf: {
            const __m512 half = _mm512_set1_ps(0.5f);
            const __m512 one = _mm512_set1_ps(1.0f);
            const __m512 inv_sqrt2 = _mm512_set1_ps(0.7071067811865476f);
            return _mm512_mul_ps(half,
                    _mm512_mul_ps(v,
                            _mm512_add_ps(one,
                                    avx512_erf(_mm512_mul_ps(v, inv_sqrt2)))));
        }
        case fused_postop_t::sigmoid: return avx512_sigmoid(v);
        case fused_postop_t::tanh_op: return avx512_tanh(v);
        case fused_postop_t::swish:
            // Note: fused swish always uses alpha=1. Non-unit alpha must use
            // the post-op chain of the epilogue (native_epilogue.cpp).
            return _mm512_mul_ps(v, avx512_sigmoid(v));
        default: return v;
    }
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_INTRINSIC_AVX512_MATH_HPP
