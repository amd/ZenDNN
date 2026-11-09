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

#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <utility>

#include <immintrin.h>

#include "common/zendnnl_compat.hpp"
#include "lowoha_operators/matmul/matmul_native/common/avx512_math.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

namespace {

constexpr int kMaxMrNv1 = 12;
constexpr int kMaxMrNv2 = 6;

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

/// 16 values of S (f32, bf16 or f16) at p as f32; exact.
template <data_type_t S>
ZENDNNL_TARGET("avx512f")
inline __m512 scales_as_f32(const char *p) {
    if constexpr (S == data_type_t::f32) {
        return _mm512_loadu_ps(reinterpret_cast<const float *>(p));
    } else {
        const __m256i h
                = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p));
        if constexpr (S == data_type_t::bf16)
            return avx512_bf16_to_f32(h);
        else
            return _mm512_cvtph_ps(h);
    }
}

/// acc[m][v] += the dot products of the K quads of one octet at b (the low
/// nibbles, then the high ones) with source bytes a .. a + 7 of row m.  The
/// loads outer and the rows inner keep two nibble planes live.
template <int MR, int NV>
ZENDNNL_TARGET("avx512f,avx512bw,avx512vnni")
inline void octet_sums(const uint8_t *b, const uint8_t *a, size_t lda,
        __m512i (&acc)[MR][NV], __m512i low) {
    ZENDNNL_NATIVE_INT4_UNROLL(4)
    for (int v = 0; v < NV; ++v) {
        const __m512i bits = _mm512_loadu_si512(b + v * 64);
        const __m512i lo = _mm512_and_si512(bits, low);
        // No byte shift exists: the bits a 16-bit shift brings in from the
        // next byte are masked off.
        const __m512i hi = _mm512_and_si512(_mm512_srli_epi16(bits, 4), low);
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

/// acc[m][v] += the dot products of one K quad, the high nibbles (High) or
/// the low ones of the octet at b, with source bytes a .. a + 3 of row m.
template <int MR, int NV, bool High>
ZENDNNL_TARGET("avx512f,avx512bw,avx512vnni")
inline void quad_sums(const uint8_t *b, const uint8_t *a, size_t lda,
        __m512i (&acc)[MR][NV], __m512i low) {
    ZENDNNL_NATIVE_INT4_UNROLL(4)
    for (int v = 0; v < NV; ++v) {
        const __m512i bits = _mm512_loadu_si512(b + v * 64);
        const __m512i w = High
                ? _mm512_and_si512(_mm512_srli_epi16(bits, 4), low)
                : _mm512_and_si512(bits, low);
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

/// The group kernel for weight scales of type S, in the fused order
/// (Fused; over source scales per group with SrcGroups) or the reference
/// order.
template <int MR, int NV, data_type_t S, bool Fused, bool SrcGroups>
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

    const __m512i low = _mm512_set1_epi8(0x0F);
    const size_t lda = static_cast<size_t>(p.lda);
    const size_t b_stride = static_cast<size_t>(p.b_stride);
    const int k_quads = p.k_quads;
    const uint8_t *a = p.A;
    const int32_t *shift = p.shift;
    const float *src_scale = p.src_scale;
    constexpr size_t wei_scale_esz = S == data_type_t::f32 ? 4 : 2;
    const char *wei_scale = static_cast<const char *>(p.wei_scale);
    const size_t wei_scale_step
            = static_cast<size_t>(p.wei_scale_step) * wei_scale_esz;
    // The group's first K quad, counted from the octet at B.
    int first = p.quad0;
    for (int g = 0; g < p.batch; ++g) {
        // The sum of each row starts at its shift; the other sets at zero.
        __m512i acc[kSets][MR][NV];
        ZENDNNL_NATIVE_INT4_UNROLL(12)
        for (int m = 0; m < MR; ++m) {
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
            quad_sums<MR, NV, true>(
                    p.B + (first >> 1) * b_stride, a, lda, acc[0], low);
            taken = 1;
        }
        const int octets = (k_quads - taken) / 2;
        const uint8_t *bo = p.B + ((first + taken) >> 1) * b_stride;
        const uint8_t *ao = a + taken * 4;
        int o = 0;
        for (; o + kSets <= octets; o += kSets) {
            ZENDNNL_NATIVE_INT4_UNROLL(4)
            for (int s = 0; s < kSets; ++s)
                octet_sums<MR, NV>(bo + (o + s) * b_stride, ao + (o + s) * 8,
                        lda, acc[s], low);
        }
        for (; o < octets; ++o)
            octet_sums<MR, NV>(bo + o * b_stride, ao + o * 8, lda, acc[0], low);
        if (taken + 2 * octets < k_quads)
            quad_sums<MR, NV, false>(
                    bo + octets * b_stride, ao + octets * 8, lda, acc[0], low);
        ZENDNNL_NATIVE_INT4_UNROLL(4)
        for (int s = 1; s < kSets; ++s) {
            ZENDNNL_NATIVE_INT4_UNROLL(12)
            for (int m = 0; m < MR; ++m) {
                ZENDNNL_NATIVE_INT4_UNROLL(4)
                for (int v = 0; v < NV; ++v)
                    acc[0][m][v] = _mm512_add_epi32(acc[0][m][v], acc[s][m][v]);
            }
        }

        // The result bits depend on this operation order: the exact int32
        // sum, conversion, then one FMA with the weight scale (fused) or with
        // the product of the source and weight scales, zero where the sum is
        // zero (fused over source groups), or the source scale, the weight
        // scale and one add (reference).
        __m512 ws[NV];
        ZENDNNL_NATIVE_INT4_UNROLL(4)
        for (int v = 0; v < NV; ++v)
            ws[v] = scales_as_f32<S>(wei_scale + v * 16 * wei_scale_esz);
        ZENDNNL_NATIVE_INT4_UNROLL(12)
        for (int m = 0; m < MR; ++m) {
            if constexpr (SrcGroups) {
                const __m512 sa = _mm512_set1_ps(src_scale[m * p.src_scale_ld]);
                ZENDNNL_NATIVE_INT4_UNROLL(4)
                for (int v = 0; v < NV; ++v) {
                    const __m512i s = acc[0][m][v];
                    c[m][v] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s),
                            _mm512_maskz_mul_ps(
                                    _mm512_test_epi32_mask(s, s), sa, ws[v]),
                            c[m][v]);
                }
            } else if constexpr (Fused) {
                ZENDNNL_NATIVE_INT4_UNROLL(4)
                for (int v = 0; v < NV; ++v)
                    c[m][v] = _mm512_fmadd_ps(
                            _mm512_cvtepi32_ps(acc[0][m][v]), ws[v], c[m][v]);
            } else {
                const __m512 sa = _mm512_set1_ps(src_scale[m * p.src_scale_ld]);
                ZENDNNL_NATIVE_INT4_UNROLL(4)
                for (int v = 0; v < NV; ++v)
                    c[m][v] = _mm512_add_ps(c[m][v],
                            avx512_mul_unfused(
                                    _mm512_mul_ps(
                                            _mm512_cvtepi32_ps(acc[0][m][v]),
                                            sa),
                                    ws[v]));
            }
        }
        first += k_quads;
        a += p.a_step;
        shift += p.shift_step;
        src_scale += p.src_scale_step;
        wei_scale += wei_scale_step;
    }
    if (Fused && !SrcGroups && p.finish) {
        ZENDNNL_NATIVE_INT4_UNROLL(12)
        for (int m = 0; m < MR; ++m) {
            const __m512 sa = _mm512_set1_ps(p.src_scale[m * p.src_scale_ld]);
            ZENDNNL_NATIVE_INT4_UNROLL(4)
            for (int v = 0; v < NV; ++v)
                c[m][v] = _mm512_mul_ps(c[m][v], sa);
        }
    }

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

/// The instance of the call's weight scale type, picked once per call: the
/// batch loop tests no type.
template <int MR, int NV, bool Fused, bool SrcGroups>
void native_int4_group_ukernel_of_dt(
        const native_int4_group_ukernel_args_t &p) {
    if (p.wei_scale_dt == data_type_t::f32)
        native_int4_group_ukernel<MR, NV, data_type_t::f32, Fused, SrcGroups>(
                p);
    else if (p.wei_scale_dt == data_type_t::bf16)
        native_int4_group_ukernel<MR, NV, data_type_t::bf16, Fused, SrcGroups>(
                p);
    else {
        assert(p.wei_scale_dt == data_type_t::f16);
        native_int4_group_ukernel<MR, NV, data_type_t::f16, Fused, SrcGroups>(
                p);
    }
}

template <int NV, bool Fused, bool SrcGroups, int... I>
constexpr std::array<native_int4_group_ukernel_fn_t, sizeof...(I)> make_table(
        std::integer_sequence<int, I...>) {
    return {{&native_int4_group_ukernel_of_dt<I + 1, NV, Fused, SrcGroups>...}};
}

constexpr auto kNv1 = make_table<1, false, false>(
        std::make_integer_sequence<int, kMaxMrNv1> {});
constexpr auto kNv2 = make_table<2, false, false>(
        std::make_integer_sequence<int, kMaxMrNv2> {});
constexpr auto kNv1Fused = make_table<1, true, false>(
        std::make_integer_sequence<int, kMaxMrNv1> {});
constexpr auto kNv2Fused = make_table<2, true, false>(
        std::make_integer_sequence<int, kMaxMrNv2> {});
constexpr auto kNv1FusedGroups = make_table<1, true, true>(
        std::make_integer_sequence<int, kMaxMrNv1> {});
constexpr auto kNv2FusedGroups = make_table<2, true, true>(
        std::make_integer_sequence<int, kMaxMrNv2> {});

} // namespace

int native_int4_group_ukernel_max_mr(int nv) {
    return nv == 1 ? kMaxMrNv1 : (nv == 2 ? kMaxMrNv2 : 0);
}

native_int4_group_ukernel_fn_t select_native_int4_group_ukernel(
        int mr, int nv, native_int8_order_t order, bool src_groups) {
    if (mr < 1 || mr > native_int4_group_ukernel_max_mr(nv)) return nullptr;
    const int i = mr - 1;
    if (order != native_int8_order_t::fused) return nv == 1 ? kNv1[i] : kNv2[i];
    if (src_groups) return nv == 1 ? kNv1FusedGroups[i] : kNv2FusedGroups[i];
    return nv == 1 ? kNv1Fused[i] : kNv2Fused[i];
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
