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

#ifndef _LOWOHA_GDN_AVX512_UTILS_HPP
#define _LOWOHA_GDN_AVX512_UTILS_HPP

#include <cstdint>
#include <immintrin.h>
#include "common/data_types.hpp"
#include "lowoha_operators/common/simd_ops.hpp"

// Code between GDN_AVX512_BEGIN and GDN_AVX512_END is compiled for the ISA
// that lowoha_gdn.cpp checks before dispatching to the native kernels. Open
// it only after every #include so that shared inline helpers (load_f32,
// logging, ...) keep the baseline ISA and stay safe on the reference path.
// _Pragma takes one string literal, so keep each pragma on one line.
// clang-format off
#if defined(__clang__)
#define GDN_AVX512_BEGIN _Pragma("clang attribute push(__attribute__((target(\"avx512f,avx512bw,avx512vl,avx512bf16,fma\"))), apply_to = function)")
#define GDN_AVX512_END _Pragma("clang attribute pop")
#define GDN_AVX512_LAMBDA __attribute__((target("avx512f,avx512bw,avx512vl,avx512bf16,fma")))
#elif defined(__GNUC__)
#define GDN_AVX512_BEGIN _Pragma("GCC push_options") _Pragma("GCC target(\"avx512f,avx512bw,avx512vl,avx512bf16,fma\")")
#define GDN_AVX512_END _Pragma("GCC pop_options")
#else
#define GDN_AVX512_BEGIN
#define GDN_AVX512_END
#endif
// clang-format on

GDN_AVX512_BEGIN

namespace zendnnl {
namespace lowoha {
namespace gdn {
namespace avx512 {

using zendnnl::common::data_type_t;
using Ops = zendnnl::lowoha::simd::SimdOps<zendnnl::lowoha::simd::avx512_tag>;

template <data_type_t DT>
struct elem;
template <>
struct elem<data_type_t::f32> {
    using type = float;
};
template <>
struct elem<data_type_t::bf16> {
    using type = uint16_t;
};
template <>
struct elem<data_type_t::f16> {
    using type = uint16_t;
};

template <data_type_t DT>
using elem_t = typename elem<DT>::type;

template <data_type_t DT>
inline __m512 load16(const elem_t<DT> *p);

template <>
inline __m512 load16<data_type_t::f32>(const float *p) {
    return _mm512_loadu_ps(p);
}
template <>
inline __m512 load16<data_type_t::bf16>(const uint16_t *p) {
    const __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p));
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16));
}
template <>
inline __m512 load16<data_type_t::f16>(const uint16_t *p) {
    return _mm512_cvtph_ps(
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p)));
}

template <data_type_t DT>
inline __m512 load16_mask(const elem_t<DT> *p, __mmask16 m);

template <>
inline __m512 load16_mask<data_type_t::f32>(const float *p, __mmask16 m) {
    return _mm512_maskz_loadu_ps(m, p);
}
template <>
inline __m512 load16_mask<data_type_t::bf16>(const uint16_t *p, __mmask16 m) {
    const __m256i h = _mm256_maskz_loadu_epi16(m, p);
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16));
}
template <>
inline __m512 load16_mask<data_type_t::f16>(const uint16_t *p, __mmask16 m) {
    return _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(m, p));
}

template <data_type_t DT>
inline void store16(elem_t<DT> *p, __m512 v);

template <>
inline void store16<data_type_t::f32>(float *p, __m512 v) {
    _mm512_storeu_ps(p, v);
}
template <>
inline void store16<data_type_t::bf16>(uint16_t *p, __m512 v) {
    _mm256_storeu_si256(
            reinterpret_cast<__m256i *>(p), (__m256i)_mm512_cvtneps_pbh(v));
}
template <>
inline void store16<data_type_t::f16>(uint16_t *p, __m512 v) {
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(p),
            _mm512_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

template <data_type_t DT>
inline void store16_mask(elem_t<DT> *p, __m512 v, __mmask16 m);

template <>
inline void store16_mask<data_type_t::f32>(float *p, __m512 v, __mmask16 m) {
    _mm512_mask_storeu_ps(p, m, v);
}
template <>
inline void store16_mask<data_type_t::bf16>(
        uint16_t *p, __m512 v, __mmask16 m) {
    _mm256_mask_storeu_epi16(p, m, (__m256i)_mm512_cvtneps_pbh(v));
}
template <>
inline void store16_mask<data_type_t::f16>(uint16_t *p, __m512 v, __mmask16 m) {
    _mm256_mask_storeu_epi16(p, m,
            _mm512_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

// One element as FP32.
template <data_type_t DT>
inline float load1(const elem_t<DT> *p);

template <>
inline float load1<data_type_t::f32>(const float *p) {
    return *p;
}
template <>
inline float load1<data_type_t::bf16>(const uint16_t *p) {
    return _mm_cvtss_f32(_mm_castsi128_ps(_mm_cvtsi32_si128(
            static_cast<int>(static_cast<uint32_t>(*p) << 16))));
}
template <>
inline float load1<data_type_t::f16>(const uint16_t *p) {
    return _mm512_cvtss_f32(_mm512_cvtph_ps(
            _mm256_zextsi128_si256(_mm_cvtsi32_si128(static_cast<int>(*p)))));
}

// Value after a round trip through DT.
template <data_type_t DT>
inline __m512 round16(__m512 v);
template <>
inline __m512 round16<data_type_t::f32>(__m512 v) {
    return v;
}
template <>
inline __m512 round16<data_type_t::bf16>(__m512 v) {
    const __m256i h = (__m256i)_mm512_cvtneps_pbh(v);
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16));
}
template <>
inline __m512 round16<data_type_t::f16>(__m512 v) {
    return _mm512_cvtph_ps(
            _mm512_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

// x * sigmoid(x)
inline __m512 silu16(__m512 x) {
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 e = Ops::vec_exp_u20(_mm512_sub_ps(_mm512_setzero_ps(), x));
    return _mm512_div_ps(x, _mm512_add_ps(one, e));
}

inline __m512 sigmoid16(__m512 x) {
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 e = Ops::vec_exp_u20(_mm512_sub_ps(_mm512_setzero_ps(), x));
    return _mm512_div_ps(one, _mm512_add_ps(one, e));
}

inline __mmask16 tail_mask(int64_t n) {
    return n >= 16 ? static_cast<__mmask16>(0xffff)
                   : static_cast<__mmask16>((1u << n) - 1u);
}

// n values of DT to FP32 (masked at the tail, so dst may be packed);
// src == nullptr writes zeros.
template <data_type_t DT>
inline void row_f32(const elem_t<DT> *src, int64_t n, float *dst) {
    for (int64_t i = 0; i < n; i += 16) {
        const __mmask16 m = tail_mask(n - i);
        _mm512_mask_storeu_ps(dst + i, m,
                src ? load16_mask<DT>(src + i, m) : _mm512_setzero_ps());
    }
}

// r[i][j] <- r[j][i] for 16 rows of 16 floats.
inline void transpose16x16(__m512 *r) {
    __m512 t[16];
    for (int i = 0; i < 16; i += 2) {
        t[i] = _mm512_unpacklo_ps(r[i], r[i + 1]);
        t[i + 1] = _mm512_unpackhi_ps(r[i], r[i + 1]);
    }
    // r[4g + c], lane L: rows 4g .. 4g + 3 at column 4L + c.
    for (int g = 0; g < 16; g += 4) {
        r[g] = _mm512_shuffle_ps(t[g], t[g + 2], 0x44);
        r[g + 1] = _mm512_shuffle_ps(t[g], t[g + 2], 0xee);
        r[g + 2] = _mm512_shuffle_ps(t[g + 1], t[g + 3], 0x44);
        r[g + 3] = _mm512_shuffle_ps(t[g + 1], t[g + 3], 0xee);
    }
    // Row 4L + c gathers lane L of r[c], r[4 + c], r[8 + c] and r[12 + c].
    for (int c = 0; c < 4; ++c) {
        t[c] = _mm512_shuffle_f32x4(r[c], r[4 + c], 0x44);
        t[4 + c] = _mm512_shuffle_f32x4(r[c], r[4 + c], 0xee);
        t[8 + c] = _mm512_shuffle_f32x4(r[8 + c], r[12 + c], 0x44);
        t[12 + c] = _mm512_shuffle_f32x4(r[8 + c], r[12 + c], 0xee);
    }
    for (int c = 0; c < 4; ++c) {
        r[c] = _mm512_shuffle_f32x4(t[c], t[8 + c], 0x88);
        r[4 + c] = _mm512_shuffle_f32x4(t[c], t[8 + c], 0xdd);
        r[8 + c] = _mm512_shuffle_f32x4(t[4 + c], t[12 + c], 0x88);
        r[12 + c] = _mm512_shuffle_f32x4(t[4 + c], t[12 + c], 0xdd);
    }
}

} // namespace avx512
} // namespace gdn
} // namespace lowoha
} // namespace zendnnl

GDN_AVX512_END

#endif // _LOWOHA_GDN_AVX512_UTILS_HPP
