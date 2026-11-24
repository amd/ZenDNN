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

#ifndef MATMUL_NATIVE_BRGEMM_KERNEL_NATIVE_DECODE_HPP
#define MATMUL_NATIVE_BRGEMM_KERNEL_NATIVE_DECODE_HPP

#include <cassert>
#include <cstdint>

#include <immintrin.h>

#include "common/zendnnl_compat.hpp"
#include "lowoha_operators/matmul/matmul_native/common/avx512_math.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_format.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

/// 16 weight scales of type S (f32, bf16 or f16) at p as f32; exact.
template <native_scale_t S>
ZENDNNL_TARGET("avx512f")
inline __m512 native_scales_as_f32(const char *p) {
    static_assert(S == native_scale_t::f32 || S == native_scale_t::bf16
                    || S == native_scale_t::f16,
            "a scale type the loader decodes");
    if constexpr (S == native_scale_t::f32) {
        return _mm512_loadu_ps(reinterpret_cast<const float *>(p));
    } else {
        const __m256i h
                = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p));
        if constexpr (S == native_scale_t::bf16)
            return avx512_bf16_to_f32(h);
        else
            return _mm512_cvtph_ps(h);
    }
}

/// 16 weight scales of type s (bf16 or f16) in h as f32; exact.
ZENDNNL_INLINE_TARGET("avx512f")
inline __m512 native_halves_as_f32(__m256i h, native_scale_t s) {
    switch (s) {
        case native_scale_t::bf16: return avx512_bf16_to_f32(h);
        case native_scale_t::f32:
        case native_scale_t::f16:
        case native_scale_t::none: break;
    }
    assert(s == native_scale_t::f16);
    return _mm512_cvtph_ps(h);
}

/// native_scales_as_f32 of scale type s, which the call picks.
ZENDNNL_TARGET("avx512f")
inline __m512 native_scales_as_f32(const char *p, native_scale_t s) {
    switch (s) {
        case native_scale_t::f32:
            return native_scales_as_f32<native_scale_t::f32>(p);
        case native_scale_t::f16:
        case native_scale_t::bf16:
        case native_scale_t::none: break;
    }
    return native_halves_as_f32(
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p)), s);
}

/// The kernels' reading of the layout's words of a format of decode D
/// (native_format_desc_t::decode): the decode of 64 loaded bytes with a
/// constant the kernel broadcasts once and holds in a register over its K
/// loop.
template <native_decode_t D>
struct native_decode;

/// Each byte the weight plus 128: the unsigned operand of a dot product with
/// an s8 source.
template <>
struct native_decode<native_decode_t::byte_offset> {
    /// The constant of bytes, in 32-bit lanes.
    static constexpr int32_t kFlip = static_cast<int32_t>(0x80808080u);
    ZENDNNL_INLINE_TARGET("avx512f")
    static __m512i bytes(__m512i bits, __m512i flip) {
        return _mm512_xor_si512(bits, flip);
    }
};

/// The low (High false) or the high nibble plane, each nibble a weight plus
/// 8 as the pack stored it.
template <>
struct native_decode<native_decode_t::nibble_offset> {
    /// The constant of plane, in bytes.
    static constexpr char kLow = 0x0F;
    template <bool High>
    ZENDNNL_INLINE_TARGET("avx512f,avx512bw")
    static __m512i plane(__m512i bits, __m512i low) {
        // No byte shift exists: the bits a 16-bit shift brings in from the
        // next byte are masked off.
        return High ? _mm512_and_si512(_mm512_srli_epi16(bits, 4), low)
                    : _mm512_and_si512(bits, low);
    }
};

/// The decode of format F's kernels.
template <native_format_t F>
using native_decode_of = native_decode<native_format(F).decode>;

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_BRGEMM_KERNEL_NATIVE_DECODE_HPP
