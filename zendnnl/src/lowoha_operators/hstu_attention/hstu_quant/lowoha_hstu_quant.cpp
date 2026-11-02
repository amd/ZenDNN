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

#include "lowoha_operators/hstu_attention/hstu_quant/lowoha_hstu_quant.hpp"

#include <immintrin.h>
#include <limits>
#include "common/zendnnl_global.hpp"
#include "lowoha_operators/hstu_attention/common/lowoha_hstu_utils.hpp"

#define HSTU_Q_INLINE ZENDNNL_INLINE_TARGET("avx512f,avx512bw,avx512vl,fma")
#define HSTU_Q_NOINLINE ZENDNNL_TARGET_NOINLINE("avx512f,avx512bw,avx512vl,fma")

namespace zendnnl {
namespace lowoha {
namespace hstu_attention {
namespace quant {

namespace {

HSTU_Q_INLINE inline __m512 load_bf16(const bfloat16_t *x, __mmask16 m) {
    return _mm512_castsi512_ps(_mm512_slli_epi32(
            _mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(m, x)), 16));
}

/// `magic(offset)` for quantize16: 1.5 * 2^23 + offset.
HSTU_Q_INLINE inline __m512 magic(int offset) {
    return _mm512_set1_ps(12582912.f + float(offset));
}

/// RNE(x * inv) + offset, as 16 bytes, with one FMA: fma(x, inv, magic) has
/// the rounded value in its low mantissa bits while |x * inv| < 2^22, so its
/// low byte is the code. The exact product is rounded once, to nearest even
/// whatever the caller's MXCSR rounding mode.
HSTU_Q_INLINE inline __m128i quantize16(__m512 x, __m512 inv, __m512 magic) {
    return _mm512_cvtepi32_epi8(_mm512_castps_si512(_mm512_fmadd_round_ps(
            x, inv, magic, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)));
}

/// Lanes of x that are NaN. An infinity needs no check: it makes the
/// group's max, and so its scale, infinite.
HSTU_Q_INLINE inline __mmask16 is_nan(__m512 x) {
    return _mm512_cmp_ps_mask(x, x, _CMP_UNORD_Q);
}

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

} // namespace

HSTU_Q_NOINLINE float quantize_row(
        const bfloat16_t *x, int64_t n, int offset, uint8_t *y) {
    __m512 am = _mm512_setzero_ps();
    __mmask16 bad = 0;
    for (int64_t c = 0; c < n; c += 16) {
        const __mmask16 m
                = __mmask16(range_bits(0, std::min<int64_t>(16, n - c)));
        const __m512 a = _mm512_abs_ps(load_bf16(x + c, m));
        bad |= is_nan(a);
        am = _mm512_max_ps(am, a);
    }
    const float amax = _mm512_reduce_max_ps(am);
    const __m512 vi = _mm512_set1_ps(quant_inv(amax));
    const __m512 vo = magic(offset);
    for (int64_t c = 0; c < n; c += 16) {
        const __mmask16 m
                = __mmask16(range_bits(0, std::min<int64_t>(16, n - c)));
        _mm_mask_storeu_epi8(y + c, m, quantize16(load_bf16(x + c, m), vi, vo));
    }
    return bad ? kNaN : quant_scale(amax);
}

HSTU_Q_NOINLINE void quantize_cols(const bfloat16_t *x, int64_t ldx,
        int64_t rows, int64_t n, int8_t *y, int64_t ldy, float *scale) {
    const __m512 eps = _mm512_set1_ps(1e-10f), v127 = _mm512_set1_ps(127.f);
    const __m512 vo = magic(0);
    const __mmask16 m0 = __mmask16(range_bits(0, std::min<int64_t>(16, n)));
    const __mmask16 m1 = __mmask16(range_bits(16, n) >> 16);
    __m512 am0 = _mm512_setzero_ps(), am1 = _mm512_setzero_ps();
    __mmask16 bad0 = 0, bad1 = 0;
    for (int64_t r = 0; r < rows; ++r) {
        const bfloat16_t *row = x + r * ldx;
        const __m512 a0 = _mm512_abs_ps(load_bf16(row, m0));
        const __m512 a1 = _mm512_abs_ps(load_bf16(row + 16, m1));
        bad0 |= is_nan(a0);
        bad1 |= is_nan(a1);
        am0 = _mm512_max_ps(am0, a0);
        am1 = _mm512_max_ps(am1, a1);
    }
    am0 = _mm512_max_ps(am0, eps);
    am1 = _mm512_max_ps(am1, eps);
    const __m512 nan = _mm512_set1_ps(kNaN);
    _mm512_storeu_ps(scale,
            _mm512_mask_mov_ps(_mm512_maskz_div_ps(m0, am0, v127), bad0, nan));
    _mm512_storeu_ps(scale + 16,
            _mm512_mask_mov_ps(_mm512_maskz_div_ps(m1, am1, v127), bad1, nan));
    const __m512 inv0 = _mm512_div_ps(v127, am0);
    const __m512 inv1 = _mm512_div_ps(v127, am1);
    for (int64_t r = 0; r < rows; ++r) {
        const bfloat16_t *row = x + r * ldx;
        _mm_storeu_si128(reinterpret_cast<__m128i *>(y + r * ldy),
                quantize16(load_bf16(row, m0), inv0, vo));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(y + r * ldy + 16),
                quantize16(load_bf16(row + 16, m1), inv1, vo));
    }
}

HSTU_Q_NOINLINE void quantize_rows_amax(const float *x, int64_t ld,
        const float *amax, const float *row_mul, int64_t ra, int64_t rb,
        int64_t lo, int64_t hi, int offset, uint8_t *y, float *scale) {
    const __m512 vo = magic(offset);
    for (int64_t r = ra; r < rb; ++r) {
        const float am = _mm512_reduce_max_ps(_mm512_loadu_ps(amax + 16 * r));
        const __m512 vi = _mm512_set1_ps(quant_inv(am));
        const float *xr = x + r * ld;
        uint8_t *yr = y + r * ld;
        __mmask16 bad = 0;
        for (int64_t c = lo; c < hi; c += 16) {
            const __mmask16 m
                    = __mmask16(range_bits(0, std::min<int64_t>(16, hi - c)));
            const __m512 xv = _mm512_maskz_loadu_ps(m, xr + c);
            bad |= is_nan(xv);
            _mm_mask_storeu_epi8(yr + c, m, quantize16(xv, vi, vo));
        }
        scale[r] = bad ? kNaN : quant_scale(am) * row_mul[r];
    }
}

} // namespace quant
} // namespace hstu_attention
} // namespace lowoha
} // namespace zendnnl
