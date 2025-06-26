/*******************************************************************************
* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
*
*******************************************************************************/

#include <algorithm>
#include <cmath>
#include "zendnn.hpp"

#include <immintrin.h>

namespace zendnn {


// Constants for polynomial approximation of erf
#define lpgemm_erf_c0 1.1283793786592402
#define lpgemm_erf_c1 -2.5468861568875563e-05
#define lpgemm_erf_c2 -0.3756169877289898
#define lpgemm_erf_c3 -0.004025179163741976
#define lpgemm_erf_c4 0.12947984300439994
#define lpgemm_erf_c5 -0.0412525204794885
#define lpgemm_erf_c6 0.03918550001070417
#define lpgemm_erf_c7 -0.07104542913277255
#define lpgemm_erf_c8 0.05717052146749476
#define lpgemm_erf_c9 -0.025310822854733135
#define lpgemm_erf_c10 0.0067305713376882076
#define lpgemm_erf_c11 -0.0010410692067591445
#define lpgemm_erf_c12 6.921588102382636e-05
#define lpgemm_erf_c13 4.092409485758739e-06
#define lpgemm_erf_c14 -1.033131746125426e-06
#define lpgemm_erf_c15 5.2927177513236435e-08

// Polynomial coefficients for exp approximation
#define lpgemm_exp_c0 1.0000000754895704
#define lpgemm_exp_c1 0.6931472254087585
#define lpgemm_exp_c2 0.2402210737432219
#define lpgemm_exp_c3 0.05550297297702539
#define lpgemm_exp_c4 0.009676036358193323
#define lpgemm_exp_c5 0.001341000536524434

// Constants for exp function
#define TBL_LN2 1.4426950408889634
#define EXPF_HUGE 12582912.0
#define EXPF_MIN -136.0
#define EXPF_MAX 136.0
#define sign -142929835592.0

void transpose_matrix(float *input, float *output, int N, int K) {

    //#pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            output[j * N + i] = input[i * K + j];
        }
    }
}

float fast_exp_scalar(float x) {
    // Simple and fast approximation of exp(x)
    return expf(x); // Replace with a faster approximation if needed
}

float fast_tanh_scalar(float x) {
    // Simple and fast approximation of tanh(x)
    return tanhf(x); // Replace with a faster approximation if needed
}

float apply_post_op_scalar(float val, ActivationPostOp post_op) {
    switch (post_op) {
    case ActivationPostOp::RELU:
        return val > 0.0f ? val : 0.0f;
    case ActivationPostOp::SIGMOID:
        return 1.0f / (1.0f + fast_exp_scalar(-val));
    case ActivationPostOp::TANH:
        return fast_tanh_scalar(val);
    case ActivationPostOp::GELU_TANH: {
        float x3 = val * val * val;
        float inner = val + 0.044715f * x3;
        float tanh_val = fast_tanh_scalar(0.79788456f * inner);
        return 0.5f * val * (1.0f + tanh_val);
    }
    case ActivationPostOp::SILU:
        return val / (1.0f + fast_exp_scalar(-val));
    default:
        return val;
    }
}

// Polynomial evaluation for exp(x)
__attribute__((target("avx512f")))
__m512 poly_eval_exp_avx512(__m512 r) {
    __m512 r2 = _mm512_mul_ps(r, r);
    __m512 z = _mm512_fmadd_ps(r2,
        _mm512_fmadd_ps(r, _mm512_set1_ps(lpgemm_exp_c3), _mm512_set1_ps(lpgemm_exp_c2)),
        _mm512_fmadd_ps(r, _mm512_set1_ps(lpgemm_exp_c1), _mm512_set1_ps(lpgemm_exp_c0)));
    r2 = _mm512_mul_ps(r2, r2);
    return _mm512_fmadd_ps(r2,
        _mm512_fmadd_ps(r, _mm512_set1_ps(lpgemm_exp_c5), _mm512_set1_ps(lpgemm_exp_c4)), z);
}

// Vectorized exp approximation
__attribute__((target("avx512f")))
__m512i expf_avx512(__m512 x) {
    __m512 z = _mm512_mul_ps(x, _mm512_set1_ps(TBL_LN2));
    __m512 dn = _mm512_add_ps(z, _mm512_set1_ps(EXPF_HUGE));
    __m512 r = _mm512_sub_ps(z, _mm512_sub_ps(dn, _mm512_set1_ps(EXPF_HUGE)));

    __m512 poly = poly_eval_exp_avx512(r);

    __m512i q = _mm512_add_epi32((__m512i)poly,
        _mm512_sllv_epi32((__m512i)dn, _mm512_set1_epi32(23)));

    q = _mm512_mask_and_epi32(q,
        _mm512_cmpnle_ps_mask(_mm512_set1_ps(EXPF_MIN), x), q, _mm512_set1_epi32(0));

    q = _mm512_mask_xor_epi32(
        (__m512i)_mm512_set1_ps(std::numeric_limits<float>::infinity()),
        _mm512_cmpnle_ps_mask(_mm512_set1_ps(EXPF_MAX), x),
        q, _mm512_set1_epi32(0));

    return q;
}

// Final tanh function
__attribute__((target("avx512f")))
__m512 fast_tanh_avx512(__m512 x_tanh) {
    __m512 x = _mm512_mul_ps(_mm512_abs_ps(x_tanh), _mm512_set1_ps(-2.0f));
    __m512i q = expf_avx512(x);

    __m512 z = _mm512_add_ps((__m512)q, _mm512_set1_ps(-1.0f));
    z = _mm512_div_ps(z, _mm512_add_ps(z, _mm512_set1_ps(2.0f)));
    z = _mm512_mul_ps(z, _mm512_set1_ps(-1.0f));

    return (__m512)(_mm512_xor_epi32(
        _mm512_and_epi32((__m512i)x_tanh, _mm512_set1_epi32(sign)),
        (__m512i)z));
}

// GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
__attribute__((target("avx512f")))
__m512 fast_gelu_tanh_avx512(__m512 x) {
    const __m512 c0 = _mm512_set1_ps(0.044715f);
    const __m512 c1 = _mm512_set1_ps(0.79788456f); // sqrt(2/pi)
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 half = _mm512_set1_ps(0.5f);

    // Compute x^3 = x * x * x
    __m512 x2 = _mm512_mul_ps(x, x);
    __m512 x3 = _mm512_mul_ps(x2, x);

    // inner = sqrt(2/pi) * (x + 0.044715 * x^3)
    __m512 inner = _mm512_mul_ps(c1, _mm512_fmadd_ps(c0, x3, x));

    // tanh(inner)
    __m512 tanh_val = fast_tanh_avx512(inner);

    // GELU = 0.5 * x * (1 + tanh(inner))
    return _mm512_mul_ps(half, _mm512_mul_ps(x, _mm512_add_ps(one, tanh_val)));
}


// Polynomial evaluation for erf(x)
__attribute__((target("avx512f")))
__m512 poly_eval_erf_avx512(__m512 x) {
    __m512 x2 = _mm512_mul_ps(x, x);
    __m512 p = _mm512_fmadd_ps(x, _mm512_set1_ps(lpgemm_erf_c15), _mm512_set1_ps(lpgemm_erf_c14));
    p = _mm512_fmadd_ps(x, p, _mm512_set1_ps(lpgemm_erf_c13));
    p = _mm512_fmadd_ps(x, p, _mm512_set1_ps(lpgemm_erf_c12));
    p = _mm512_fmadd_ps(x, p, _mm512_set1_ps(lpgemm_erf_c11));
    p = _mm512_fmadd_ps(x, p, _mm512_set1_ps(lpgemm_erf_c10));
    p = _mm512_fmadd_ps(x, p, _mm512_set1_ps(lpgemm_erf_c9));
    p = _mm512_fmadd_ps(x, p, _mm512_set1_ps(lpgemm_erf_c8));
    p = _mm512_fmadd_ps(x, p, _mm512_set1_ps(lpgemm_erf_c7));
    p = _mm512_fmadd_ps(x, p, _mm512_set1_ps(lpgemm_erf_c6));
    p = _mm512_fmadd_ps(x, p, _mm512_set1_ps(lpgemm_erf_c5));
    p = _mm512_fmadd_ps(x, p, _mm512_set1_ps(lpgemm_erf_c4));
    p = _mm512_fmadd_ps(x, p, _mm512_set1_ps(lpgemm_erf_c3));
    p = _mm512_fmadd_ps(x, p, _mm512_set1_ps(lpgemm_erf_c2));
    p = _mm512_fmadd_ps(x, p, _mm512_set1_ps(lpgemm_erf_c1));
    return _mm512_fmadd_ps(x, p, _mm512_set1_ps(lpgemm_erf_c0));
}

// Vectorized GELU using erf approximation
__attribute__((target("avx512f")))
__m512 fast_gelu_erf_avx512(__m512 x) {
    const __m512 inv_sqrt2 = _mm512_set1_ps(0.70710678f); // 1 / sqrt(2)
    const __m512 half = _mm512_set1_ps(0.5f);
    const __m512 one = _mm512_set1_ps(1.0f);

    __m512 scaled = _mm512_mul_ps(x, inv_sqrt2);

    __m512 erf_val = poly_eval_erf_avx512(scaled);

    __m512 one_plus_erf = _mm512_add_ps(one, erf_val);
    __m512 prod = _mm512_mul_ps(x, one_plus_erf);
    return _mm512_mul_ps(half, prod);
}


// Vectorized sigmoid using exp
__attribute__((target("avx512f")))
__m512 fast_sigmoid_avx512(__m512 x) {
    __m512 neg_x = _mm512_mul_ps(x, _mm512_set1_ps(-1.0f));
    __m512i exp_neg_x = expf_avx512(neg_x);

    __m512 one_vec = _mm512_set1_ps(1.0f);
    __m512 denom = _mm512_add_ps(one_vec, (__m512)exp_neg_x);
    __m512 result = _mm512_div_ps(one_vec, denom);

    return result;
   
}

// Vectorized SiLU (Sigmoid Linear Unit) using sigmoid approximation
__attribute__((target("avx512f")))
__m512 fast_silu_avx512(__m512 x) {
    __m512 sigmoid_val = fast_sigmoid_avx512(x);
    return _mm512_mul_ps(x, sigmoid_val);
}


__attribute__((target("avx512f")))
__m512 apply_post_op(__m512 vec, ActivationPostOp post_op) {
    switch (post_op) {
    case ActivationPostOp::RELU:
        return _mm512_max_ps(vec, _mm512_setzero_ps());
    case ActivationPostOp::SIGMOID:
        return fast_sigmoid_avx512(vec);
    case ActivationPostOp::TANH:
        return fast_tanh_avx512(vec);
    case ActivationPostOp::GELU_TANH:
        return fast_gelu_tanh_avx512(vec);
    case ActivationPostOp::GELU_ERF:
        return fast_gelu_erf_avx512(vec);
    case ActivationPostOp::SILU:
        return fast_silu_avx512(vec);
    default:
        return vec;
    }
}

}

