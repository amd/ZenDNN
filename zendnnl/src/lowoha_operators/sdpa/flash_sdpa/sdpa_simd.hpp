/******************************************************************************
 * Modifications Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Phase 1 SIMD abstraction for SDPA standalone kernel (see
 * zen_Sdpa_standalone_performance_plan.md). No ATen dependency.
 *
 * Two specializations selected at runtime via SimdOps<Tag>:
 *   - avx512_tag: 16 floats per vector, AVX-512 intrinsics via
 *                 __attribute__((target(...)))  (no -mavx512f flag needed)
 *   - scalar_tag: 1 float per vector, portable scalar fallback
 ******************************************************************************/

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#include <immintrin.h>

namespace zendnnl {
namespace lowoha {
namespace sdpa {
namespace simd {

// ---------------------------------------------------------------------------
// Tag types for compile-time SIMD dispatch.
// ---------------------------------------------------------------------------
struct avx512_tag {};
struct scalar_tag {};

template <typename Tag>
struct SimdOps;

// ===========================================================================
// Scalar specialization — one float lane, portable fallback.
// ===========================================================================
template <>
struct SimdOps<scalar_tag> {

  struct VecF32 {
    float v;
  };
  static constexpr int kFloatLanes = 1;

  static inline VecF32 vec_loadu(const float *p) {
    return VecF32{*p};
  }

  static inline void vec_storeu(float *p, VecF32 x) {
    *p = x.v;
  }

  static inline VecF32 vec_set1(float f) {
    return VecF32{f};
  }

  static inline VecF32 vec_add(VecF32 a, VecF32 b) {
    return VecF32{a.v + b.v};
  }

  static inline VecF32 vec_sub(VecF32 a, VecF32 b) {
    return VecF32{a.v - b.v};
  }

  static inline VecF32 vec_mul(VecF32 a, VecF32 b) {
    return VecF32{a.v * b.v};
  }

  static inline VecF32 vec_fmadd(VecF32 a, VecF32 b, VecF32 c) {
    return VecF32{a.v *b.v + c.v};
  }

  static inline VecF32 vec_max(VecF32 a, VecF32 b) {
    return VecF32{std::fmax(a.v, b.v)};
  }

  static inline VecF32 vec_min(VecF32 a, VecF32 b) {
    return VecF32{std::fmin(a.v, b.v)};
  }

  static inline float vec_hsum(VecF32 v) {
    return v.v;
  }

  static inline float vec_hmax(VecF32 v) {
    return v.v;
  }

  static inline VecF32 vec_exp_u20(VecF32 v) {
    return VecF32{std::exp(v.v)};
  }

  static inline VecF32 vec_fexp_u20(VecF32 v) {
    return VecF32{std::exp(v.v)};
  }

  static inline float vec_reduce_sum(VecF32 acc) {
    return acc.v;
  }

  static inline float vec_reduce_max(VecF32 acc) {
    return acc.v;
  }

  static inline VecF32 vec_mask_bf16_loadu(const uint16_t *p) {
    uint32_t u = static_cast<uint32_t>(p[0]) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return VecF32{f};
  }

  static inline void vec_bf16_storeu(uint16_t *dst, VecF32 v) {
    uint32_t u;
    std::memcpy(&u, &v.v, sizeof(u));
    uint32_t rounding_bias = ((u >> 16) & 1) + 0x7FFF;
    dst[0] = static_cast<uint16_t>((u + rounding_bias) >> 16);
  }
};

// ===========================================================================
// AVX-512 specialization — 16 float lanes, enabled via target attribute.
// ===========================================================================

#define SDPA_AVX512_ATTR __attribute__((target("avx512f,avx512bw,avx512vl,fma")))

template <>
struct SimdOps<avx512_tag> {

  using VecF32 = __m512;
  static constexpr int kFloatLanes = 16;

  SDPA_AVX512_ATTR
  static inline VecF32 vec_loadu(const float *p) {
    return _mm512_loadu_ps(p);
  }

  SDPA_AVX512_ATTR
  static inline void vec_storeu(float *p, VecF32 x) {
    _mm512_storeu_ps(p, x);
  }

  SDPA_AVX512_ATTR
  static inline VecF32 vec_set1(float f) {
    return _mm512_set1_ps(f);
  }

  SDPA_AVX512_ATTR
  static inline VecF32 vec_add(VecF32 a, VecF32 b) {
    return _mm512_add_ps(a, b);
  }

  SDPA_AVX512_ATTR
  static inline VecF32 vec_sub(VecF32 a, VecF32 b) {
    return _mm512_sub_ps(a, b);
  }

  SDPA_AVX512_ATTR
  static inline VecF32 vec_mul(VecF32 a, VecF32 b) {
    return _mm512_mul_ps(a, b);
  }

  SDPA_AVX512_ATTR
  static inline VecF32 vec_fmadd(VecF32 a, VecF32 b, VecF32 c) {
    return _mm512_fmadd_ps(a, b, c);
  }

  SDPA_AVX512_ATTR
  static inline VecF32 vec_max(VecF32 a, VecF32 b) {
    return _mm512_max_ps(a, b);
  }

  SDPA_AVX512_ATTR
  static inline VecF32 vec_min(VecF32 a, VecF32 b) {
    return _mm512_min_ps(a, b);
  }

  SDPA_AVX512_ATTR
  static inline float vec_hsum(VecF32 v) {
    return _mm512_reduce_add_ps(v);
  }

  SDPA_AVX512_ATTR
  static inline float vec_hmax(VecF32 v) {
    return _mm512_reduce_max_ps(v);
  }

  // ── Fast exp (~20 ULP, Malossi et al.) ──────────────────────────────

  SDPA_AVX512_ATTR
  static inline __m512 fexp_u20_ps_512(__m512 values) {
    static const __m512 vec_c0 = _mm512_set1_ps(0.00010703434948458272f);
    static const __m512 vec_c1 = _mm512_set1_ps(0.30354260500649682f);
    static const __m512 vec_c2 = _mm512_set1_ps(-0.22433836478672356f);
    static const __m512 vec_c3 = _mm512_set1_ps(-0.079204240219773236f);
    static const __m512 vec_exp_log2ef =
      _mm512_castsi512_ps(_mm512_set1_epi32(0x3fb8aa3b));
    static const __m512 vec_a =
      _mm512_set1_ps(
        static_cast<float>(std::pow(2.0, 23) / std::log2(2.0)));
    static const __m512 vec_b =
      _mm512_set1_ps(static_cast<float>(std::pow(2.0, 23) * 127.0));
    static const __m512 vec_ln_flt_min =
      _mm512_castsi512_ps(_mm512_set1_epi32(0xc2aeac50));
    static const __m512 vec_ln_flt_max =
      _mm512_castsi512_ps(_mm512_set1_epi32(0x42b17218));
    static const __m512i vec_infinity = _mm512_set1_epi32(0x7F800000);
    static const __m512i vec_zero = _mm512_setzero_epi32();

    const __mmask16 min_mask =
      _mm512_cmp_ps_mask(values, vec_ln_flt_min, _CMP_LT_OS);
    const __mmask16 max_mask =
      _mm512_cmp_ps_mask(values, vec_ln_flt_max, _CMP_GT_OS);

    __m512 vec_src = _mm512_mul_ps(values, vec_exp_log2ef);
    __m512 vec_fractional =
      _mm512_sub_ps(vec_src, _mm512_floor_ps(vec_src));

    __m512 vec_res = _mm512_fmadd_ps(vec_fractional, vec_c3, vec_c2);
    vec_res = _mm512_fmadd_ps(vec_fractional, vec_res, vec_c1);
    vec_res = _mm512_fmadd_ps(vec_fractional, vec_res, vec_c0);

    vec_src = _mm512_sub_ps(vec_src, vec_res);
    __m512 tmp = _mm512_fmadd_ps(vec_a, vec_src, vec_b);
    __m512i casted_integer = _mm512_cvttps_epi32(tmp);
    casted_integer =
      _mm512_mask_mov_epi32(casted_integer, min_mask, vec_zero);
    casted_integer =
      _mm512_mask_mov_epi32(casted_integer, max_mask, vec_infinity);
    return _mm512_castsi512_ps(casted_integer);
  }

  SDPA_AVX512_ATTR
  static inline __m512 exp_u20_ps_512(__m512 values) {
    static const __m512 vec_factorial_1 = _mm512_set1_ps(0.999999701f);
    static const __m512 vec_factorial_2 = _mm512_set1_ps(0.499991506f);
    static const __m512 vec_factorial_3 = _mm512_set1_ps(0.166676521f);
    static const __m512 vec_factorial_4 = _mm512_set1_ps(0.0418978221f);
    static const __m512 vec_factorial_5 = _mm512_set1_ps(0.00828929059f);
    static const __m512 vec_exp_log2ef =
      _mm512_castsi512_ps(_mm512_set1_epi32(0x3fb8aa3b));
    static const __m512 vec_half = _mm512_set1_ps(0.5f);
    static const __m512 vec_one = _mm512_set1_ps(1.f);
    static const __m512 vec_zero = _mm512_set1_ps(0.f);
    static const __m512 vec_ln2f =
      _mm512_castsi512_ps(_mm512_set1_epi32(0x3f317218));
    static const __m512 vec_ln_flt_min =
      _mm512_castsi512_ps(_mm512_set1_epi32(0xc2aeac50));
    static const __m512 vec_ln_flt_max =
      _mm512_castsi512_ps(_mm512_set1_epi32(0x42b17218));
    static const __m512i vec_126 = _mm512_set1_epi32(126);
    constexpr int n_mantissa_bits = 23;

    const __mmask16 less_ln_flt_min_mask =
      _mm512_cmp_ps_mask(values, vec_ln_flt_min, _CMP_LT_OS);
    __m512 vec_src = _mm512_min_ps(values, vec_ln_flt_max);
    vec_src = _mm512_max_ps(vec_src, vec_ln_flt_min);

    __m512 vec_fx = _mm512_fmadd_ps(vec_src, vec_exp_log2ef, vec_half);
    __m512i vec_fx_i = _mm512_cvt_roundps_epi32(
                         vec_fx, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    vec_fx = _mm512_cvtepi32_ps(vec_fx_i);

    __m512 vec_exp_poly = _mm512_fnmadd_ps(vec_fx, vec_ln2f, vec_src);

    __m512 vec_res =
      _mm512_fmadd_ps(vec_exp_poly, vec_factorial_5, vec_factorial_4);
    vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_3);
    vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_2);
    vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_1);
    vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_one);

    // Construct 2^fx_i as two multiplications: poly * 2^(fx_i-1) * 2.
    // Direct construction of 2^fx_i would overflow the IEEE exponent field
    // when fx_i == 128 (exponent 255 = infinity). Splitting via (fx_i-1)+127
    // keeps the intermediate in the normal range (max exponent 254 = 2^127),
    // and the final *2 reaches 2^128 through normal FP multiplication.
    // Merge the (fx_i - 1 + 127) into a single (fx_i + 126).
    __m512i vec_two_pow_n_i = _mm512_add_epi32(vec_fx_i, vec_126);
    vec_two_pow_n_i = _mm512_slli_epi32(vec_two_pow_n_i, n_mantissa_bits);
    __m512 vec_two_pow_n = _mm512_castsi512_ps(vec_two_pow_n_i);
    vec_two_pow_n = _mm512_mask_blend_ps(
                      less_ln_flt_min_mask, vec_two_pow_n, vec_zero);

    vec_res = _mm512_mul_ps(vec_res, vec_two_pow_n);
    vec_res = _mm512_mul_ps(vec_res, _mm512_set1_ps(2.f));
    return vec_res;
  }

  // ── Public exp wrappers ─────────────────────────────────────────────

  SDPA_AVX512_ATTR
  static inline VecF32 vec_exp_u20(VecF32 v) {
    return exp_u20_ps_512(v);
  }

  SDPA_AVX512_ATTR
  static inline VecF32 vec_fexp_u20(VecF32 v) {
    return fexp_u20_ps_512(v);
  }

  // ── Reductions ──────────────────────────────────────────────────────

  SDPA_AVX512_ATTR
  static inline float vec_reduce_sum(VecF32 acc) {
    return vec_hsum(acc);
  }

  SDPA_AVX512_ATTR
  static inline float vec_reduce_max(VecF32 acc) {
    return vec_hmax(acc);
  }

  // ── BF16 ────────────────────────────────────────────────────────────

  SDPA_AVX512_ATTR
  static inline VecF32 vec_mask_bf16_loadu(const uint16_t *p) {
    __m256i u = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p));
    __m512i wide = _mm512_slli_epi32(_mm512_cvtepu16_epi32(u), 16);
    return _mm512_castsi512_ps(wide);
  }

  // FP32 → BF16 store with round-to-nearest-even (no avx512bf16 ISA needed).
  // Inverse of vec_mask_bf16_loadu: packs 16 × FP32 into 16 × BF16.
  SDPA_AVX512_ATTR
  static inline void vec_bf16_storeu(uint16_t *dst, VecF32 v) {
    __m512i u = _mm512_castps_si512(v);
    __m512i rounding_bias = _mm512_add_epi32(
      _mm512_and_si512(_mm512_srli_epi32(u, 16), _mm512_set1_epi32(1)),
      _mm512_set1_epi32(0x7FFF));
    __m512i rounded = _mm512_srli_epi32(_mm512_add_epi32(u, rounding_bias), 16);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst),
                        _mm512_cvtepi32_epi16(rounded));
  }
};

#undef SDPA_AVX512_ATTR

} // namespace simd
} // namespace sdpa
} // namespace lowoha
} // namespace zendnnl
