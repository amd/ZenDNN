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

#include <immintrin.h>

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

__attribute__((target("avx512f,fma")))
static inline __m512 avx512_exp(__m512 x) {
  const __m512 log2e  = _mm512_set1_ps(1.44269504089f);
  const __m512 ln2_hi = _mm512_set1_ps(0.693359375f);
  const __m512 ln2_lo = _mm512_set1_ps(-2.12194440e-4f);
  const __m512 one  = _mm512_set1_ps(1.0f);
  const __m512 c1 = _mm512_set1_ps(1.9875691500e-4f);
  const __m512 c2 = _mm512_set1_ps(1.3981999507e-3f);
  const __m512 c3 = _mm512_set1_ps(8.3334519073e-3f);
  const __m512 c4 = _mm512_set1_ps(4.1665795894e-2f);
  const __m512 c5 = _mm512_set1_ps(1.6666665459e-1f);
  const __m512 c6 = _mm512_set1_ps(5.0000001201e-1f);

  x = _mm512_max_ps(x, _mm512_set1_ps(-87.3f));
  x = _mm512_min_ps(x, _mm512_set1_ps(88.3f));

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

__attribute__((target("avx512f,fma")))
static inline __m512 avx512_tanh(__m512 x) {
  const __m512 one = _mm512_set1_ps(1.0f);
  const __m512 two = _mm512_set1_ps(2.0f);
  const __m512 neg_one = _mm512_set1_ps(-1.0f);
  __m512 exp2x = avx512_exp(_mm512_mul_ps(two, x));
  __m512 result = _mm512_sub_ps(one,
    _mm512_div_ps(two, _mm512_add_ps(one, exp2x)));
  result = _mm512_max_ps(result, neg_one);
  result = _mm512_min_ps(result, one);
  return result;
}

__attribute__((target("avx512f,fma")))
static inline __m512 avx512_sigmoid(__m512 x) {
  const __m512 one = _mm512_set1_ps(1.0f);
  __m512 neg_x = _mm512_sub_ps(_mm512_setzero_ps(), x);
  return _mm512_div_ps(one, _mm512_add_ps(one, avx512_exp(neg_x)));
}

__attribute__((target("avx512f,fma")))
static inline __m512 avx512_erf(__m512 x) {
  const __m512 a1 = _mm512_set1_ps(0.254829592f);
  const __m512 a2 = _mm512_set1_ps(-0.284496736f);
  const __m512 a3 = _mm512_set1_ps(1.421413741f);
  const __m512 a4 = _mm512_set1_ps(-1.453152027f);
  const __m512 a5 = _mm512_set1_ps(1.061405429f);
  const __m512 p  = _mm512_set1_ps(0.3275911f);
  const __m512 one = _mm512_set1_ps(1.0f);

  __m512 abs_x = _mm512_abs_ps(x);
  __mmask16 neg_mask = _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_LT_OQ);
  __m512 sign = _mm512_mask_sub_ps(one, neg_mask, _mm512_setzero_ps(), one);

  __m512 t = _mm512_div_ps(one, _mm512_fmadd_ps(p, abs_x, one));
  __m512 poly = _mm512_fmadd_ps(a5, t, a4);
  poly = _mm512_fmadd_ps(poly, t, a3);
  poly = _mm512_fmadd_ps(poly, t, a2);
  poly = _mm512_fmadd_ps(poly, t, a1);

  __m512 neg_x2 = _mm512_sub_ps(_mm512_setzero_ps(),
                   _mm512_mul_ps(abs_x, abs_x));
  __m512 exp_neg_x2 = avx512_exp(neg_x2);
  __m512 result = _mm512_sub_ps(one,
    _mm512_mul_ps(_mm512_mul_ps(t, poly), exp_neg_x2));
  return _mm512_mul_ps(sign, result);
}

/// Apply a fused post-op to a single ZMM register.
__attribute__((target("avx512f,fma")))
static inline __m512 apply_fused_postop(__m512 v, fused_postop_t op) {
  switch (op) {
  case fused_postop_t::relu:
    return _mm512_max_ps(v, _mm512_setzero_ps());
  case fused_postop_t::gelu_tanh: {
    const __m512 half = _mm512_set1_ps(0.5f);
    const __m512 one  = _mm512_set1_ps(1.0f);
    const __m512 c  = _mm512_set1_ps(0.7978845608028654f);
    const __m512 c2   = _mm512_set1_ps(0.044715f);
    __m512 x3 = _mm512_mul_ps(_mm512_mul_ps(v, v), v);
    __m512 inner = _mm512_mul_ps(c, _mm512_fmadd_ps(c2, x3, v));
    return _mm512_mul_ps(half, _mm512_mul_ps(v,
      _mm512_add_ps(one, avx512_tanh(inner))));
  }
  case fused_postop_t::gelu_erf: {
    const __m512 half = _mm512_set1_ps(0.5f);
    const __m512 one  = _mm512_set1_ps(1.0f);
    const __m512 inv_sqrt2 = _mm512_set1_ps(0.7071067811865476f);
    return _mm512_mul_ps(half, _mm512_mul_ps(v,
      _mm512_add_ps(one, avx512_erf(_mm512_mul_ps(v, inv_sqrt2)))));
  }
  case fused_postop_t::sigmoid:
    return avx512_sigmoid(v);
  case fused_postop_t::tanh_op:
    return avx512_tanh(v);
  case fused_postop_t::swish:
    // Note: fused swish always uses alpha=1. Non-unit alpha must use
    // the non-fused postop path (apply_postops_tile in postop.cpp).
    return _mm512_mul_ps(v, avx512_sigmoid(v));
  default:
    return v;
  }
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_INTRINSIC_AVX512_MATH_HPP
