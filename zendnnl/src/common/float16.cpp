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

#include "float16.hpp"
#include <immintrin.h>
#include "common/zendnnl_compat.hpp"

namespace zendnnl {
namespace common {

void float16_t::f16_to_f32_buf(
        const uint16_t *f16_buf, float *f32_buf, int64_t size_) {
    for (int64_t j = 0; j < size_; ++j) {
        f32_buf[j] = float16_t::f16_to_f32_val(f16_buf[j]);
    }
}

void float16_t::f32_to_f16(
        const float *f32_buf, uint16_t *f16_buf, int64_t size_) {
    for (int64_t j = 0; j < size_; ++j) {
        f16_buf[j] = f32_to_f16_val(f32_buf[j]);
    }
}

void float16_t::f32_to_f16(
        const float *f32_buf, float16_t *f16_buf, int64_t size_) {
    for (int64_t j = 0; j < size_; ++j) {
        f16_buf[j] = float16_t::from_bits(f32_to_f16_val(f32_buf[j]));
    }
}

//===----------------------------------------------------------------------===//
// SIMD vector conversions
//===----------------------------------------------------------------------===//

#if defined(ZENDNNL_HAS_AVX512FP16_MASK_LOAD_STORE_INTRINSICS) \
        || (defined(__GNUC__) && (__GNUC__ >= 12))

ZENDNNL_TARGET("avx512f,avx512vl,avx512bw,avx512fp16")
__m512h float16_t::cvt_f32_to_f16_vec(__m512 lo, __m512 hi) {
    __m256i h_lo = _mm512_cvtps_ph(
            lo, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256i h_hi = _mm512_cvtps_ph(
            hi, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    return (__m512h)_mm512_inserti64x4(_mm512_castsi256_si512(h_lo), h_hi, 1);
}

ZENDNNL_TARGET("avx512f,avx512vl,avx512bw,avx512fp16")
void float16_t::cvt_f16_to_f32_vec(__m512h val, __m512 &lo, __m512 &hi) {
    __m256i lo_half = _mm512_castsi512_si256((__m512i)val);
    __m256i hi_half = _mm512_extracti64x4_epi64((__m512i)val, 1);
    lo = _mm512_cvtph_ps(lo_half);
    hi = _mm512_cvtph_ps(hi_half);
}

#endif // ZENDNNL_HAS_AVX512FP16_MASK_LOAD_STORE_INTRINSICS || __GNUC__ >= 12

} //namespace common
} //namespace zendnnl
