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

#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/int8/native_int4_ukernel.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <utility>

#include <immintrin.h>

#include "common/zendnnl_compat.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/native_decode.hpp"
#include "lowoha_operators/matmul/matmul_native/common/avx512_math.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_format.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

namespace {

constexpr int kMaxMrNv1 = 12;
constexpr int kMaxMrNv2 = 6;
constexpr int kMaxMrOneGroup = 12;

// Pins an accumulator in its register after each update: without it the
// scheduler hoists the next rows' broadcasts and the large tiles spill
// their accumulators.
#if defined(_MSC_VER) && !defined(__clang__)
#define ZENDNNL_NATIVE_INT4_ACC_BARRIER(acc)
#else
#define ZENDNNL_NATIVE_INT4_ACC_BARRIER(acc) asm("" : "+v"(acc))
#endif

// Unrolls the loop that follows by n; MSVC has no such pragma.
#if defined(_MSC_VER) && !defined(__clang__)
#define ZENDNNL_NATIVE_INT4_UNROLL(n)
#else
#define ZENDNNL_NATIVE_INT4_PRAGMA(text) _Pragma(#text)
#define ZENDNNL_NATIVE_INT4_UNROLL(n) ZENDNNL_NATIVE_INT4_PRAGMA(GCC unroll n)
#endif

/// c plus one group's exact int32 sum s, in the order the result bits
/// depend on: conversion, then one FMA with the weight scale ws (fused), or
/// with the product of the source scale sa and ws, zero where s is zero
/// (fused over source scales per group), or sa, ws and one add (reference).
template <bool Fused, bool SrcGroups>
ZENDNNL_INLINE_TARGET("avx512f,fma")
inline __m512 int4_group_step(__m512i s, __m512 ws, __m512 sa, __m512 c) {
    if constexpr (SrcGroups)
        return _mm512_fmadd_ps(_mm512_cvtepi32_ps(s),
                _mm512_maskz_mul_ps(_mm512_test_epi32_mask(s, s), sa, ws), c);
    else if constexpr (Fused)
        return _mm512_fmadd_ps(_mm512_cvtepi32_ps(s), ws, c);
    else
        return _mm512_add_ps(c,
                avx512_mul_unfused(
                        _mm512_mul_ps(_mm512_cvtepi32_ps(s), sa), ws));
}

/// The fused order's finish: row m of c times its source scale.
template <int MR, int NV>
ZENDNNL_INLINE_TARGET("avx512f")
inline void int4_finish(
        const native_int4_group_ukernel_args_t &p, __m512 (&c)[MR][NV]) {
    ZENDNNL_NATIVE_INT4_UNROLL(12)
    for (int m = 0; m < MR; ++m) {
        const __m512 sa = _mm512_set1_ps(p.src_scale[m * p.src_scale_ld]);
        ZENDNNL_NATIVE_INT4_UNROLL(4)
        for (int v = 0; v < NV; ++v)
            c[m][v] = _mm512_mul_ps(c[m][v], sa);
    }
}

/// Stores the tile c to C (f32, ldc apart), or with dst to dst (bf16
/// rounded to nearest even, or f32; dst_ldc apart).
template <int MR, int NV>
ZENDNNL_INLINE_TARGET("avx512f")
inline void int4_store(
        const native_int4_group_ukernel_args_t &p, const __m512 (&c)[MR][NV]) {
    const size_t ldc = static_cast<size_t>(p.ldc);
    const size_t dst_ldc = static_cast<size_t>(p.dst_ldc);
    if (p.dst == nullptr) {
        float *const out = p.C;
        ZENDNNL_NATIVE_INT4_UNROLL(12)
        for (int m = 0; m < MR; ++m) {
            ZENDNNL_NATIVE_INT4_UNROLL(4)
            for (int v = 0; v < NV; ++v)
                _mm512_storeu_ps(out + m * ldc + v * 16, c[m][v]);
        }
    } else if (p.dst_bf16) {
        uint16_t *const out = static_cast<uint16_t *>(p.dst);
        ZENDNNL_NATIVE_INT4_UNROLL(12)
        for (int m = 0; m < MR; ++m) {
            ZENDNNL_NATIVE_INT4_UNROLL(4)
            for (int v = 0; v < NV; ++v)
                _mm256_storeu_si256(
                        reinterpret_cast<__m256i *>(out + m * dst_ldc + v * 16),
                        avx512_f32_to_bf16_rne(c[m][v]));
        }
    } else {
        float *const out = static_cast<float *>(p.dst);
        ZENDNNL_NATIVE_INT4_UNROLL(12)
        for (int m = 0; m < MR; ++m) {
            ZENDNNL_NATIVE_INT4_UNROLL(4)
            for (int v = 0; v < NV; ++v)
                _mm512_storeu_ps(out + m * dst_ldc + v * 16, c[m][v]);
        }
    }
}

/// acc[m][v] += the dot products of the K quads of one octet at b (vector v
/// at b + v * b_second; the low nibble plane, then the high one, as Decode
/// reads them with low) with source bytes a .. a + 7 of row m.  The loads
/// outer and the rows inner keep two nibble planes live.
template <int MR, int NV, typename Decode>
ZENDNNL_TARGET("avx512f,avx512bw,avx512vnni")
inline void octet_sums(const uint8_t *b, size_t b_second, const uint8_t *a,
        size_t lda, __m512i (&acc)[MR][NV], __m512i low) {
    ZENDNNL_NATIVE_INT4_UNROLL(4)
    for (int v = 0; v < NV; ++v) {
        const __m512i bits = _mm512_loadu_si512(b + v * b_second);
        const __m512i lo = Decode::template plane<false>(bits, low);
        const __m512i hi = Decode::template plane<true>(bits, low);
        ZENDNNL_NATIVE_INT4_UNROLL(12)
        for (int m = 0; m < MR; ++m) {
            uint32_t q0, q1;
            std::memcpy(&q0, a + m * lda, sizeof(q0));
            std::memcpy(&q1, a + m * lda + 4, sizeof(q1));
            acc[m][v] = _mm512_dpbusd_epi32(
                    acc[m][v], lo, _mm512_set1_epi32(static_cast<int32_t>(q0)));
            acc[m][v] = _mm512_dpbusd_epi32(
                    acc[m][v], hi, _mm512_set1_epi32(static_cast<int32_t>(q1)));
            ZENDNNL_NATIVE_INT4_ACC_BARRIER(acc[m][v]);
        }
    }
}

/// quad_sums of the octet's NV weight vectors bits, with the rows outer:
/// each broadcast source quad feeds every vector, with the NV nibble planes
/// of the quad live.
template <int MR, int NV, bool High, typename Decode>
ZENDNNL_TARGET("avx512f,avx512bw,avx512vnni")
inline void quad_sums_by_row(const __m512i (&bits)[NV], const uint8_t *a,
        size_t lda, __m512i (&acc)[MR][NV], __m512i low) {
    __m512i w[NV];
    ZENDNNL_NATIVE_INT4_UNROLL(4)
    for (int v = 0; v < NV; ++v)
        w[v] = Decode::template plane<High>(bits[v], low);
    ZENDNNL_NATIVE_INT4_UNROLL(12)
    for (int m = 0; m < MR; ++m) {
        uint32_t q;
        std::memcpy(&q, a + m * lda, sizeof(q));
        const __m512i av = _mm512_set1_epi32(static_cast<int32_t>(q));
        ZENDNNL_NATIVE_INT4_UNROLL(4)
        for (int v = 0; v < NV; ++v) {
            acc[m][v] = _mm512_dpbusd_epi32(acc[m][v], w[v], av);
            ZENDNNL_NATIVE_INT4_ACC_BARRIER(acc[m][v]);
        }
    }
}

/// acc[m][v] += the dot products of the octet at b (vector v at b + v *
/// b_second), its low nibbles and then its high ones, with source bytes a
/// .. a + 7 of row m, the rows outer; each weight vector is loaded once.
template <int MR, int NV, typename Decode>
ZENDNNL_TARGET("avx512f,avx512bw,avx512vnni")
inline void octet_sums_by_row(const uint8_t *b, size_t b_second,
        const uint8_t *a, size_t lda, __m512i (&acc)[MR][NV], __m512i low) {
    __m512i bits[NV];
    ZENDNNL_NATIVE_INT4_UNROLL(4)
    for (int v = 0; v < NV; ++v)
        bits[v] = _mm512_loadu_si512(b + v * b_second);
    quad_sums_by_row<MR, NV, false, Decode>(bits, a, lda, acc, low);
    quad_sums_by_row<MR, NV, true, Decode>(bits, a + 4, lda, acc, low);
}

/// The kernels of one group over K: octets o0 .. o0 + octets - 1 of the
/// weight at B (octet o at B + o * b_stride) with the source bytes from a
/// on; with Prefetch, each octet whose octet kNativeInt4PrefetchOctets on
/// lies below limit_octets requests that octet's weight lines after its own
/// sums.  A pointer moves to the next octet only while there is one, so
/// that none is formed past the last octet read or requested.
template <int MR, int NV, bool Prefetch, typename Decode>
ZENDNNL_TARGET("avx512f,avx512bw,avx512vnni")
inline void octets_by_row(const uint8_t *B, int o0, int octets,
        int limit_octets, size_t b_stride, size_t b_second, const uint8_t *a,
        size_t lda, __m512i (&acc)[MR][NV], __m512i low) {
    if (octets <= 0) return;
    const uint8_t *b = B + static_cast<size_t>(o0) * b_stride;
    int left = octets;
    if constexpr (Prefetch) {
        const int ahead = std::min(
                octets, limit_octets - kNativeInt4PrefetchOctets - o0);
        if (ahead > 0) {
            const uint8_t *pf = b + kNativeInt4PrefetchOctets * b_stride;
            for (int n = ahead;;) {
                octet_sums_by_row<MR, NV, Decode>(
                        b, b_second, a, lda, acc, low);
                const char *f = reinterpret_cast<const char *>(pf);
                _mm_prefetch(f, _MM_HINT_T0);
                if constexpr (NV == 2) _mm_prefetch(f + b_second, _MM_HINT_T0);
                if (--n == 0) break;
                b += b_stride;
                a += 8;
                pf += b_stride;
            }
            left -= ahead;
            if (left == 0) return;
            b += b_stride;
            a += 8;
        }
    }
    for (;;) {
        octet_sums_by_row<MR, NV, Decode>(b, b_second, a, lda, acc, low);
        if (--left == 0) break;
        b += b_stride;
        a += 8;
    }
}

/// acc[m][v] += the dot products of one K quad, the high nibbles (High) or
/// the low ones of the octet at b (vector v at b + v * b_second), with
/// source bytes a .. a + 3 of row m.
template <int MR, int NV, bool High, typename Decode>
ZENDNNL_TARGET("avx512f,avx512bw,avx512vnni")
inline void quad_sums(const uint8_t *b, size_t b_second, const uint8_t *a,
        size_t lda, __m512i (&acc)[MR][NV], __m512i low) {
    ZENDNNL_NATIVE_INT4_UNROLL(4)
    for (int v = 0; v < NV; ++v) {
        const __m512i bits = _mm512_loadu_si512(b + v * b_second);
        const __m512i w = Decode::template plane<High>(bits, low);
        ZENDNNL_NATIVE_INT4_UNROLL(12)
        for (int m = 0; m < MR; ++m) {
            uint32_t q;
            std::memcpy(&q, a + m * lda, sizeof(q));
            acc[m][v] = _mm512_dpbusd_epi32(
                    acc[m][v], w, _mm512_set1_epi32(static_cast<int32_t>(q)));
            ZENDNNL_NATIVE_INT4_ACC_BARRIER(acc[m][v]);
        }
    }
}

/// The group kernel of a weight of format F, its nibbles read by F's decode,
/// for weight scales of type S, in the fused order (Fused; over source
/// scales per group with SrcGroups) or the reference order; Passes: the
/// call may take a K pass (partial_in, partial_out).
template <native_format_t F, int MR, int NV, native_scale_t S, bool Fused,
        bool SrcGroups, bool Passes>
ZENDNNL_TARGET_NOINLINE("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni,fma")
void native_int4_group_ukernel(const native_int4_group_ukernel_args_t &p) {
    static_assert(Fused || !SrcGroups, "source groups are a fused order");
    // Small tiles split each group's octets over independent accumulator
    // sets; the int32 sum is exact in any order.
    constexpr int kSets = MR * NV <= 2 ? 4 : (MR * NV <= 4 ? 2 : 1);
    const size_t ldc = static_cast<size_t>(p.ldc);
    __m512 c[MR][NV];
    ZENDNNL_NATIVE_INT4_UNROLL(12)
    for (int m = 0; m < MR; ++m) {
        ZENDNNL_NATIVE_INT4_UNROLL(4)
        for (int v = 0; v < NV; ++v)
            c[m][v] = p.accumulate ? _mm512_loadu_ps(p.C + m * ldc + v * 16)
                                   : _mm512_setzero_ps();
    }

    using decode = native_decode_of<F>;
    const __m512i low = _mm512_set1_epi8(decode::kLow);
    const size_t lda = static_cast<size_t>(p.lda);
    const size_t b_stride = static_cast<size_t>(p.b_stride);
    const size_t b_second = p.b_second;
    const int k_quads = p.k_quads;
    const uint8_t *a = p.A;
    const int32_t *shift = p.shift;
    const float *src_scale = p.src_scale;
    constexpr size_t wei_scale_esz = native_scale_bytes(S);
    const char *wei_scale = static_cast<const char *>(p.wei_scale);
    const size_t wei_scale_step
            = static_cast<size_t>(p.wei_scale_step) * wei_scale_esz;
    const size_t scale_second
            = static_cast<size_t>(p.scale_second) * wei_scale_esz;
    const bool partial_in = Passes && p.partial_in;
    const bool partial_out = Passes && p.partial_out;
    // The group's first K quad, counted from the octet at B.
    int first = p.quad0;
    for (int g = 0; g < p.batch; ++g) {
        // The sum of each row starts at its shift, or at the int32 sum of
        // the earlier K passes in C; the other sets at zero.
        __m512i acc[kSets][MR][NV];
        ZENDNNL_NATIVE_INT4_UNROLL(12)
        for (int m = 0; m < MR; ++m) {
            if (partial_in) {
                ZENDNNL_NATIVE_INT4_UNROLL(4)
                for (int v = 0; v < NV; ++v)
                    acc[0][m][v] = _mm512_loadu_si512(p.C + m * ldc + v * 16);
                continue;
            }
            const __m512i start = _mm512_set1_epi32(shift[m * p.shift_ld]);
            ZENDNNL_NATIVE_INT4_UNROLL(4)
            for (int v = 0; v < NV; ++v)
                acc[0][m][v] = start;
        }
        ZENDNNL_NATIVE_INT4_UNROLL(4)
        for (int s = 1; s < kSets; ++s) {
            ZENDNNL_NATIVE_INT4_UNROLL(12)
            for (int m = 0; m < MR; ++m) {
                ZENDNNL_NATIVE_INT4_UNROLL(4)
                for (int v = 0; v < NV; ++v)
                    acc[s][m][v] = _mm512_setzero_si512();
            }
        }
        // A group of an odd number of quads starts or ends half way
        // through an octet.
        int taken = 0;
        if ((first & 1) != 0) {
            quad_sums<MR, NV, true, decode>(p.B + (first >> 1) * b_stride,
                    b_second, a, lda, acc[0], low);
            taken = 1;
        }
        const int octets = (k_quads - taken) / 2;
        const uint8_t *bo = p.B + ((first + taken) >> 1) * b_stride;
        const uint8_t *ao = a + taken * 4;
        int o = 0;
        for (; o + kSets <= octets; o += kSets) {
            ZENDNNL_NATIVE_INT4_UNROLL(4)
            for (int s = 0; s < kSets; ++s)
                octet_sums<MR, NV, decode>(bo + (o + s) * b_stride, b_second,
                        ao + (o + s) * 8, lda, acc[s], low);
        }
        for (; o < octets; ++o)
            octet_sums<MR, NV, decode>(
                    bo + o * b_stride, b_second, ao + o * 8, lda, acc[0], low);
        if (taken + 2 * octets < k_quads)
            quad_sums<MR, NV, false, decode>(bo + octets * b_stride, b_second,
                    ao + octets * 8, lda, acc[0], low);
        ZENDNNL_NATIVE_INT4_UNROLL(4)
        for (int s = 1; s < kSets; ++s) {
            ZENDNNL_NATIVE_INT4_UNROLL(12)
            for (int m = 0; m < MR; ++m) {
                ZENDNNL_NATIVE_INT4_UNROLL(4)
                for (int v = 0; v < NV; ++v)
                    acc[0][m][v] = _mm512_add_epi32(acc[0][m][v], acc[s][m][v]);
            }
        }
        if (partial_out) {
            ZENDNNL_NATIVE_INT4_UNROLL(12)
            for (int m = 0; m < MR; ++m) {
                ZENDNNL_NATIVE_INT4_UNROLL(4)
                for (int v = 0; v < NV; ++v)
                    _mm512_storeu_si512(p.C + m * ldc + v * 16, acc[0][m][v]);
            }
            return;
        }

        __m512 ws[NV];
        ZENDNNL_NATIVE_INT4_UNROLL(4)
        for (int v = 0; v < NV; ++v)
            ws[v] = native_scales_as_f32<S>(wei_scale + v * scale_second);
        ZENDNNL_NATIVE_INT4_UNROLL(12)
        for (int m = 0; m < MR; ++m) {
            __m512 sa = _mm512_setzero_ps();
            if constexpr (SrcGroups || !Fused)
                sa = _mm512_set1_ps(src_scale[m * p.src_scale_ld]);
            ZENDNNL_NATIVE_INT4_UNROLL(4)
            for (int v = 0; v < NV; ++v)
                c[m][v] = int4_group_step<Fused, SrcGroups>(
                        acc[0][m][v], ws[v], sa, c[m][v]);
        }
        first += k_quads;
        a += p.a_step;
        shift += p.shift_step;
        src_scale += p.src_scale_step;
        wei_scale += wei_scale_step;
    }
    if (Fused && !SrcGroups && p.finish) int4_finish(p, c);
    int4_store(p, c);
}

/// The instance of the call's weight scale type and of whether it takes a K
/// pass, picked once per call: the batch loop tests neither.
template <native_format_t F, int MR, int NV, bool Fused, bool SrcGroups,
        bool Passes>
void native_int4_group_ukernel_of_scale(
        const native_int4_group_ukernel_args_t &p) {
    switch (p.wei_scale_type) {
        case native_scale_t::f32:
            native_int4_group_ukernel<F, MR, NV, native_scale_t::f32, Fused,
                    SrcGroups, Passes>(p);
            return;
        case native_scale_t::bf16:
            native_int4_group_ukernel<F, MR, NV, native_scale_t::bf16, Fused,
                    SrcGroups, Passes>(p);
            return;
        case native_scale_t::f16:
        case native_scale_t::none: break;
    }
    assert(p.wei_scale_type == native_scale_t::f16);
    native_int4_group_ukernel<F, MR, NV, native_scale_t::f16, Fused, SrcGroups,
            Passes>(p);
}

/// Only a weight of one group over K takes K passes, and source scales per
/// group need groups: a call over them has no pass instance.
template <native_format_t F, int MR, int NV, bool Fused, bool SrcGroups>
void native_int4_group_ukernel_of_call(
        const native_int4_group_ukernel_args_t &p) {
    if constexpr (!SrcGroups) {
        if (p.partial_in || p.partial_out) {
            native_int4_group_ukernel_of_scale<F, MR, NV, Fused, false, true>(
                    p);
            return;
        }
    }
    assert(!p.partial_in && !p.partial_out);
    native_int4_group_ukernel_of_scale<F, MR, NV, Fused, SrcGroups, false>(p);
}

/// The group kernel of a weight of format F at a batch of one group, in the
/// fused order (Fused) or the reference order, its int32 sums the only
/// accumulators across the K loop: the f32 sum starts after it, with the
/// group kernel's int4_group_step, int4_finish and int4_store, so its bits
/// are the group kernel's.
template <native_format_t F, int MR, int NV, bool Fused>
ZENDNNL_TARGET_NOINLINE("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni,fma")
void native_int4_one_group_ukernel(const native_int4_group_ukernel_args_t &p) {
    assert(p.batch == 1);
    const size_t ldc = static_cast<size_t>(p.ldc);
    using decode = native_decode_of<F>;
    const __m512i low = _mm512_set1_epi8(decode::kLow);
    const size_t lda = static_cast<size_t>(p.lda);
    const size_t b_stride = static_cast<size_t>(p.b_stride);
    const size_t b_second = p.b_second;
    const int k_quads = p.k_quads;
    // The sum of each row starts at its shift, or at the int32 sum of the
    // earlier K passes in C.
    __m512i acc[MR][NV];
    ZENDNNL_NATIVE_INT4_UNROLL(12)
    for (int m = 0; m < MR; ++m) {
        if (p.partial_in) {
            ZENDNNL_NATIVE_INT4_UNROLL(4)
            for (int v = 0; v < NV; ++v)
                acc[m][v] = _mm512_loadu_si512(p.C + m * ldc + v * 16);
            continue;
        }
        const __m512i start = _mm512_set1_epi32(p.shift[m * p.shift_ld]);
        ZENDNNL_NATIVE_INT4_UNROLL(4)
        for (int v = 0; v < NV; ++v)
            acc[m][v] = start;
    }
    // A pass of an odd number of quads starts or ends half way through an
    // octet.
    const int first = p.quad0;
    int taken = 0;
    if ((first & 1) != 0) {
        quad_sums<MR, NV, true, decode>(
                p.B + (first >> 1) * b_stride, b_second, p.A, lda, acc, low);
        taken = 1;
    }
    const int octets = (k_quads - taken) / 2;
    const int o0 = (first + taken) >> 1;
    const uint8_t *ao = p.A + taken * 4;
    assert(p.request_limit_octets == 0
            || p.request_limit_octets >= (first + k_quads + 1) / 2);
    if (p.request_limit_octets > 0)
        octets_by_row<MR, NV, true, decode>(p.B, o0, octets,
                p.request_limit_octets, b_stride, b_second, ao, lda, acc, low);
    else
        octets_by_row<MR, NV, false, decode>(
                p.B, o0, octets, 0, b_stride, b_second, ao, lda, acc, low);
    if (taken + 2 * octets < k_quads)
        quad_sums<MR, NV, false, decode>(
                p.B + static_cast<size_t>(o0 + octets) * b_stride, b_second,
                ao + octets * 8, lda, acc, low);
    if (p.partial_out) {
        ZENDNNL_NATIVE_INT4_UNROLL(12)
        for (int m = 0; m < MR; ++m) {
            ZENDNNL_NATIVE_INT4_UNROLL(4)
            for (int v = 0; v < NV; ++v)
                _mm512_storeu_si512(p.C + m * ldc + v * 16, acc[m][v]);
        }
        return;
    }

    // The group kernel's step, finish and store for its one group; the
    // weight scales are read in their type once per call.
    const size_t wei_scale_esz = native_scale_bytes(p.wei_scale_type);
    const char *wei_scale = static_cast<const char *>(p.wei_scale);
    __m512 ws[NV];
    ZENDNNL_NATIVE_INT4_UNROLL(4)
    for (int v = 0; v < NV; ++v)
        ws[v] = native_scales_as_f32(wei_scale
                        + static_cast<size_t>(v)
                                * static_cast<size_t>(p.scale_second)
                                * wei_scale_esz,
                p.wei_scale_type);
    __m512 c[MR][NV];
    ZENDNNL_NATIVE_INT4_UNROLL(12)
    for (int m = 0; m < MR; ++m) {
        __m512 sa = _mm512_setzero_ps();
        if constexpr (!Fused)
            sa = _mm512_set1_ps(p.src_scale[m * p.src_scale_ld]);
        ZENDNNL_NATIVE_INT4_UNROLL(4)
        for (int v = 0; v < NV; ++v)
            c[m][v] = int4_group_step<Fused, false>(acc[m][v], ws[v], sa,
                    p.accumulate ? _mm512_loadu_ps(p.C + m * ldc + v * 16)
                                 : _mm512_setzero_ps());
    }
    if (Fused && p.finish) int4_finish(p, c);
    int4_store(p, c);
}

template <native_format_t F, int NV, bool Fused, bool SrcGroups, int... I>
constexpr std::array<native_int4_group_ukernel_fn_t, sizeof...(I)> make_table(
        std::integer_sequence<int, I...>) {
    return {{&native_int4_group_ukernel_of_call<F, I + 1, NV, Fused,
            SrcGroups>...}};
}

/// The kernels of one group of the rows above the group kernel's: MR
/// kMaxMrNv2 + 1 .. kMaxMrOneGroup at two vectors.
template <native_format_t F, bool Fused, int... I>
constexpr std::array<native_int4_group_ukernel_fn_t, sizeof...(I)>
make_one_group_table(std::integer_sequence<int, I...>) {
    return {{&native_int4_one_group_ukernel<F, kMaxMrNv2 + I + 1, 2,
            Fused>...}};
}

/// The kernels of a weight of format F by row count: the group kernel at
/// one and two vectors in the reference order, the fused order and the
/// fused order over source scales per group; a weight of one group takes
/// the group kernel's instances where it has the rows (12 at one vector, 6
/// at two), and the kernels of one group above them, in both orders.
template <native_format_t F>
struct int4_kernels_t {
    static constexpr auto kNv1 = make_table<F, 1, false, false>(
            std::make_integer_sequence<int, kMaxMrNv1> {});
    static constexpr auto kNv2 = make_table<F, 2, false, false>(
            std::make_integer_sequence<int, kMaxMrNv2> {});
    static constexpr auto kNv1Fused = make_table<F, 1, true, false>(
            std::make_integer_sequence<int, kMaxMrNv1> {});
    static constexpr auto kNv2Fused = make_table<F, 2, true, false>(
            std::make_integer_sequence<int, kMaxMrNv2> {});
    static constexpr auto kNv1FusedGroups = make_table<F, 1, true, true>(
            std::make_integer_sequence<int, kMaxMrNv1> {});
    static constexpr auto kNv2FusedGroups = make_table<F, 2, true, true>(
            std::make_integer_sequence<int, kMaxMrNv2> {});
    static constexpr auto kNv2OneGroup = make_one_group_table<F, false>(
            std::make_integer_sequence<int, kMaxMrOneGroup - kMaxMrNv2> {});
    static constexpr auto kNv2OneGroupFused = make_one_group_table<F, true>(
            std::make_integer_sequence<int, kMaxMrOneGroup - kMaxMrNv2> {});
};

/// One format's tables of int4_kernels_t.
struct int4_tables_t {
    const native_int4_group_ukernel_fn_t *nv1, *nv2, *nv1_fused, *nv2_fused,
            *nv1_fused_groups, *nv2_fused_groups, *nv2_one_group,
            *nv2_one_group_fused;
};

template <native_format_t F>
constexpr int4_tables_t int4_tables_of() {
    using k = int4_kernels_t<F>;
    return {k::kNv1.data(), k::kNv2.data(), k::kNv1Fused.data(),
            k::kNv2Fused.data(), k::kNv1FusedGroups.data(),
            k::kNv2FusedGroups.data(), k::kNv2OneGroup.data(),
            k::kNv2OneGroupFused.data()};
}

/// The formats of the routes on these kernels, which they are instantiated
/// for, and their tables in the same order.
constexpr auto kFormats = native_family_formats<native_kernel_kind_t::int4>();

template <size_t... I>
constexpr std::array<int4_tables_t, sizeof...(I)> make_tables(
        std::index_sequence<I...>) {
    return {{int4_tables_of<kFormats[I]>()...}};
}

constexpr auto kTables
        = make_tables(std::make_index_sequence<kFormats.size()> {});

/// The tables of format; nullptr for a format of no route on these kernels.
const int4_tables_t *tables_of(native_format_t format) {
    for (size_t i = 0; i < kFormats.size(); ++i)
        if (kFormats[i] == format) return &kTables[i];
    return nullptr;
}

} // namespace

int native_int4_group_ukernel_max_mr(int nv) {
    return nv == 1 ? kMaxMrNv1 : (nv == 2 ? kMaxMrNv2 : 0);
}

native_int4_group_ukernel_fn_t select_native_int4_group_ukernel(
        native_format_t format, int mr, int nv, native_int8_order_t order,
        bool src_groups) {
    const int4_tables_t *t = tables_of(format);
    if (t == nullptr || mr < 1 || mr > native_int4_group_ukernel_max_mr(nv))
        return nullptr;
    const int i = mr - 1;
    if (order != native_int8_order_t::fused)
        return (nv == 1 ? t->nv1 : t->nv2)[i];
    if (src_groups)
        return (nv == 1 ? t->nv1_fused_groups : t->nv2_fused_groups)[i];
    return (nv == 1 ? t->nv1_fused : t->nv2_fused)[i];
}

int native_int4_one_group_ukernel_max_mr(int nv) {
    return nv == 1 ? kMaxMrNv1 : (nv == 2 ? kMaxMrOneGroup : 0);
}

native_int4_group_ukernel_fn_t select_native_int4_one_group_ukernel(
        native_format_t format, int mr, int nv, native_int8_order_t order) {
    const int4_tables_t *t = tables_of(format);
    if (t == nullptr || mr < 1 || mr > native_int4_one_group_ukernel_max_mr(nv))
        return nullptr;
    if (mr <= native_int4_group_ukernel_max_mr(nv))
        return select_native_int4_group_ukernel(format, mr, nv, order);
    const int i = mr - kMaxMrNv2 - 1;
    return (order == native_int8_order_t::fused ? t->nv2_one_group_fused
                                                : t->nv2_one_group)[i];
}

bool native_int4_kernels_serve(
        native_format_t format, native_int8_scheme_t scheme, int mr, int nv) {
    using order_t = native_int8_order_t;
    if (scheme == native_int8_scheme_t::per_group)
        return select_native_int4_group_ukernel(format, mr, nv, order_t::fused)
                != nullptr
                && select_native_int4_group_ukernel(
                           format, mr, nv, order_t::fused, true)
                != nullptr
                && select_native_int4_group_ukernel(
                           format, mr, nv, order_t::reference)
                != nullptr;
    return select_native_int4_one_group_ukernel(format, mr, nv, order_t::fused)
            != nullptr
            && select_native_int4_one_group_ukernel(
                       format, mr, nv, order_t::reference)
            != nullptr;
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
