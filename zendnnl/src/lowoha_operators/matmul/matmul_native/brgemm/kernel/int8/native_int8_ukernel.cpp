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

#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/int8/native_int8_ukernel.hpp"

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

constexpr int kMaxMr = 12;

// Pins a row's accumulators in registers at the end of each row: without
// it the scheduler hoists the next rows' broadcasts and the large tiles
// spill their accumulators.
#if defined(_MSC_VER) && !defined(__clang__)
#define ZENDNNL_NATIVE_INT8_ACC_BARRIER(acc)
#else
#define ZENDNNL_NATIVE_INT8_ACC_BARRIER(acc) asm("" : "+v"(acc))
#endif

// Unrolls the loop that follows by n; MSVC has no such pragma.
#if defined(_MSC_VER) && !defined(__clang__)
#define ZENDNNL_NATIVE_INT8_UNROLL(n)
#else
#define ZENDNNL_NATIVE_INT8_PRAGMA(text) _Pragma(#text)
#define ZENDNNL_NATIVE_INT8_UNROLL(n) ZENDNNL_NATIVE_INT8_PRAGMA(GCC unroll n)
#endif

/// f32(acc - zp * cs) for 16 lanes with the zero-point term in int64: exact
/// before the one rounding to f32, so equal to the int32 form wherever that
/// one does not wrap.
ZENDNNL_TARGET("avx512f,avx512dq")
inline __m512 sum_minus_zp_wide(__m512i acc, int64_t zp, __m512i cs) {
    const __m512i z = _mm512_set1_epi64(zp);
    const __m512i lo = _mm512_sub_epi64(
            _mm512_cvtepi32_epi64(_mm512_castsi512_si256(acc)),
            _mm512_mullo_epi64(
                    z, _mm512_cvtepi32_epi64(_mm512_castsi512_si256(cs))));
    const __m512i hi = _mm512_sub_epi64(
            _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(acc, 1)),
            _mm512_mullo_epi64(z,
                    _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(cs, 1))));
    return _mm512_insertf32x8(_mm512_castps256_ps512(_mm512_cvtepi64_ps(lo)),
            _mm512_cvtepi64_ps(hi), 1);
}

template <int MR, int NV, bool Signed>
ZENDNNL_TARGET_NOINLINE("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni,fma")
void native_int8_ukernel(const native_int8_ukernel_args_t &p) {
    // Small tiles split K over independent accumulator sets so enough
    // VPDPBUSD chains are in flight; the int32 sum is exact in any order.
    constexpr int kSets = MR * NV <= 2 ? 4 : (MR * NV <= 4 ? 2 : 1);
    __m512i acc[kSets][MR][NV];
    ZENDNNL_NATIVE_INT8_UNROLL(4)
    for (int s = 0; s < kSets; ++s) {
        ZENDNNL_NATIVE_INT8_UNROLL(12)
        for (int m = 0; m < MR; ++m) {
            ZENDNNL_NATIVE_INT8_UNROLL(4)
            for (int v = 0; v < NV; ++v)
                acc[s][m][v] = _mm512_setzero_si512();
        }
    }

    const __m512i flip = _mm512_set1_epi32(static_cast<int32_t>(0x80808080u));
    if constexpr (!Signed) (void)flip;
    const uint8_t *a = p.A;
    const int8_t *b = p.B;
    const size_t lda = static_cast<size_t>(p.lda);
    const size_t b_stride = static_cast<size_t>(p.b_stride);
    const size_t b_second = p.b_second;
    const int k_quads = p.k_quads;
    const int k_main = k_quads - k_quads % kSets;

    int kq = 0;
    for (; kq < k_main; kq += kSets) {
        ZENDNNL_NATIVE_INT8_UNROLL(4)
        for (int s = 0; s < kSets; ++s) {
            const int8_t *bq = b + static_cast<size_t>(kq + s) * b_stride;
            const uint8_t *aq = a + static_cast<size_t>(kq + s) * 4;
            __m512i bv[NV];
            ZENDNNL_NATIVE_INT8_UNROLL(4)
            for (int v = 0; v < NV; ++v)
                bv[v] = _mm512_loadu_si512(bq + v * b_second);
            ZENDNNL_NATIVE_INT8_UNROLL(12)
            for (int m = 0; m < MR; ++m) {
                uint32_t quad;
                std::memcpy(&quad, aq + m * lda, sizeof(quad));
                __m512i av = _mm512_set1_epi32(static_cast<int32_t>(quad));
                if constexpr (Signed) av = _mm512_xor_si512(av, flip);
                ZENDNNL_NATIVE_INT8_UNROLL(4)
                for (int v = 0; v < NV; ++v) {
                    acc[s][m][v] = _mm512_dpbusd_epi32(acc[s][m][v], av, bv[v]);
                    ZENDNNL_NATIVE_INT8_ACC_BARRIER(acc[s][m][v]);
                }
            }
        }
    }
    for (; kq < k_quads; ++kq) {
        const int8_t *bq = b + static_cast<size_t>(kq) * b_stride;
        const uint8_t *aq = a + static_cast<size_t>(kq) * 4;
        __m512i bv[NV];
        ZENDNNL_NATIVE_INT8_UNROLL(4)
        for (int v = 0; v < NV; ++v)
            bv[v] = _mm512_loadu_si512(bq + v * b_second);
        ZENDNNL_NATIVE_INT8_UNROLL(12)
        for (int m = 0; m < MR; ++m) {
            uint32_t quad;
            std::memcpy(&quad, aq + m * lda, sizeof(quad));
            __m512i av = _mm512_set1_epi32(static_cast<int32_t>(quad));
            if constexpr (Signed) av = _mm512_xor_si512(av, flip);
            ZENDNNL_NATIVE_INT8_UNROLL(4)
            for (int v = 0; v < NV; ++v) {
                acc[0][m][v] = _mm512_dpbusd_epi32(acc[0][m][v], av, bv[v]);
                ZENDNNL_NATIVE_INT8_ACC_BARRIER(acc[0][m][v]);
            }
        }
    }
    ZENDNNL_NATIVE_INT8_UNROLL(4)
    for (int s = 1; s < kSets; ++s) {
        ZENDNNL_NATIVE_INT8_UNROLL(12)
        for (int m = 0; m < MR; ++m) {
            ZENDNNL_NATIVE_INT8_UNROLL(4)
            for (int v = 0; v < NV; ++v)
                acc[0][m][v] = _mm512_add_epi32(acc[0][m][v], acc[s][m][v]);
        }
    }

    const size_t ldc = static_cast<size_t>(p.ldc);
    if (p.partial != nullptr) {
        ZENDNNL_NATIVE_INT8_UNROLL(12)
        for (int m = 0; m < MR; ++m) {
            int32_t *q = p.partial + m * ldc;
            ZENDNNL_NATIVE_INT8_UNROLL(4)
            for (int v = 0; v < NV; ++v) {
                if (!p.first_pass)
                    acc[0][m][v] = _mm512_add_epi32(
                            acc[0][m][v], _mm512_loadu_si512(q + v * 16));
                if (!p.last_pass) _mm512_storeu_si512(q + v * 16, acc[0][m][v]);
            }
        }
        if (!p.last_pass) return;
    }

    // The result bits depend on this operation order: int32 compensation,
    // conversion, scale product, one FMA with the bias.
    const bool has_bias = p.bias != nullptr;
    const size_t dst_ldc = static_cast<size_t>(p.dst_ldc);
    ZENDNNL_NATIVE_INT8_UNROLL(12)
    for (int m = 0; m < MR; ++m) {
        const __m512i zp = _mm512_set1_epi32(p.zp[m]);
        const __m512 src_scale = _mm512_set1_ps(p.src_scale[m]);
        float *c = p.C + m * ldc;
        ZENDNNL_NATIVE_INT8_UNROLL(4)
        for (int v = 0; v < NV; ++v) {
            __m512 f;
            if (p.wide_zp) {
                f = sum_minus_zp_wide(acc[0][m][v],
                        static_cast<int64_t>(p.zp[m]) + (Signed ? 128 : 0),
                        _mm512_loadu_si512(p.col_sum + v * 16));
            } else {
                const __m512i corr = _mm512_mullo_epi32(
                        zp, _mm512_loadu_si512(p.col_sum + v * 16));
                f = _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[0][m][v], corr));
            }
            const __m512 scale = _mm512_mul_ps(
                    src_scale, _mm512_loadu_ps(p.wei_scale + v * 16));
            const __m512 out = has_bias
                    ? _mm512_fmadd_ps(
                              f, scale, _mm512_loadu_ps(p.bias + v * 16))
                    : _mm512_mul_ps(f, scale);
            if (p.dst == nullptr)
                _mm512_storeu_ps(c + v * 16, out);
            else if (p.dst_bf16)
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(
                                            static_cast<uint16_t *>(p.dst)
                                            + m * dst_ldc + v * 16),
                        avx512_f32_to_bf16_rne(out));
            else
                _mm512_storeu_ps(
                        static_cast<float *>(p.dst) + m * dst_ldc + v * 16,
                        out);
        }
    }
}

template <int NV, bool Signed, int... I>
constexpr std::array<native_int8_ukernel_fn_t, sizeof...(I)> make_table(
        std::integer_sequence<int, I...>) {
    return {{&native_int8_ukernel<I + 1, NV, Signed>...}};
}

constexpr auto kNv1
        = make_table<1, false>(std::make_integer_sequence<int, kMaxMr> {});
constexpr auto kNv1Signed
        = make_table<1, true>(std::make_integer_sequence<int, kMaxMr> {});
constexpr auto kNv2
        = make_table<2, false>(std::make_integer_sequence<int, kMaxMr> {});
constexpr auto kNv2Signed
        = make_table<2, true>(std::make_integer_sequence<int, kMaxMr> {});

/// Rows of the group kernel per width.  At two vectors the 4 * MR int32 and
/// f32 accumulators, two B or weight-scale vectors, the broadcast and the
/// flip constant take 4 * MR + 4 of the 32 registers; 6 rows leave four for
/// the conversion and the scale widening (two for a u8 source, whose ones
/// and two column sums replace the flip constant).  At one vector the rows
/// are kMaxMr's.
constexpr int kMaxGroupMrNv1 = kMaxMr;
constexpr int kMaxGroupMrNv2 = 6;

/// The group kernel of a weight of format F, for weight scales of type S,
/// over an s8 source (the weight read by F's decode), or a u8 source with U8,
/// whose zero-point term is in int64 with Wide.
template <native_format_t F, int MR, int NV, native_scale_t S, bool U8,
        bool Wide>
ZENDNNL_TARGET_NOINLINE("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni")
void native_int8_group_ukernel(const native_int8_group_ukernel_args_t &p) {
    static_assert(U8 || !Wide, "only a u8 source has a zero point");
    // Small tiles split each element's K over independent accumulator sets;
    // the int32 sum is exact in any order.
    constexpr int kSets = MR * NV <= 2 ? 4 : (MR * NV <= 4 ? 2 : 1);
    const size_t ldc = static_cast<size_t>(p.ldc);
    __m512 c[MR][NV];
    ZENDNNL_NATIVE_INT8_UNROLL(12)
    for (int m = 0; m < MR; ++m) {
        ZENDNNL_NATIVE_INT8_UNROLL(4)
        for (int v = 0; v < NV; ++v)
            c[m][v] = p.accumulate ? _mm512_loadu_ps(p.C + m * ldc + v * 16)
                                   : _mm512_setzero_ps();
    }

    // An s8 source reads the weight by its format's decode, as w + 128; a
    // u8 source reads it as it is and sums its columns against a u8 one in
    // every byte.
    using decode = native_decode_of<F>;
    const __m512i flip = _mm512_set1_epi32(decode::kFlip);
    const __m512i ones = _mm512_set1_epi8(1);
    if constexpr (U8)
        (void)flip;
    else
        (void)ones;
    const size_t lda = static_cast<size_t>(p.lda);
    const size_t b_stride = static_cast<size_t>(p.b_stride);
    const size_t b_second = p.b_second;
    const int k_quads = p.k_quads;
    const int k_main = k_quads - k_quads % kSets;
    const uint8_t *a = p.A;
    const int8_t *b = p.B;
    const int32_t *shift = p.shift;
    const float *src_scale = p.src_scale;
    constexpr size_t wei_scale_esz = native_scale_bytes(S);
    const char *wei_scale = static_cast<const char *>(p.wei_scale);
    const size_t wei_scale_step
            = static_cast<size_t>(p.wei_scale_step) * wei_scale_esz;
    const size_t scale_second
            = static_cast<size_t>(p.scale_second) * wei_scale_esz;
    for (int g = 0; g < p.batch; ++g) {
        // The sum of each row starts at its shift (zero for a u8 source);
        // the other sets at zero.
        __m512i acc[kSets][MR][NV];
        __m512i col[kSets][NV];
        ZENDNNL_NATIVE_INT8_UNROLL(12)
        for (int m = 0; m < MR; ++m) {
            const __m512i start = U8 ? _mm512_setzero_si512()
                                     : _mm512_set1_epi32(shift[m * p.shift_ld]);
            ZENDNNL_NATIVE_INT8_UNROLL(4)
            for (int v = 0; v < NV; ++v)
                acc[0][m][v] = start;
        }
        ZENDNNL_NATIVE_INT8_UNROLL(4)
        for (int s = 1; s < kSets; ++s) {
            ZENDNNL_NATIVE_INT8_UNROLL(12)
            for (int m = 0; m < MR; ++m) {
                ZENDNNL_NATIVE_INT8_UNROLL(4)
                for (int v = 0; v < NV; ++v)
                    acc[s][m][v] = _mm512_setzero_si512();
            }
        }
        if constexpr (U8) {
            ZENDNNL_NATIVE_INT8_UNROLL(4)
            for (int s = 0; s < kSets; ++s) {
                ZENDNNL_NATIVE_INT8_UNROLL(4)
                for (int v = 0; v < NV; ++v)
                    col[s][v] = _mm512_setzero_si512();
            }
        }
        int kq = 0;
        for (; kq < k_main; kq += kSets) {
            ZENDNNL_NATIVE_INT8_UNROLL(4)
            for (int s = 0; s < kSets; ++s) {
                const int8_t *bq = b + static_cast<size_t>(kq + s) * b_stride;
                const uint8_t *aq = a + static_cast<size_t>(kq + s) * 4;
                __m512i bv[NV];
                ZENDNNL_NATIVE_INT8_UNROLL(4)
                for (int v = 0; v < NV; ++v) {
                    if constexpr (U8) {
                        bv[v] = _mm512_loadu_si512(bq + v * b_second);
                        col[s][v] = _mm512_dpbusd_epi32(col[s][v], ones, bv[v]);
                        ZENDNNL_NATIVE_INT8_ACC_BARRIER(col[s][v]);
                    } else {
                        bv[v] = decode::bytes(
                                _mm512_loadu_si512(bq + v * b_second), flip);
                    }
                }
                ZENDNNL_NATIVE_INT8_UNROLL(12)
                for (int m = 0; m < MR; ++m) {
                    uint32_t quad;
                    std::memcpy(&quad, aq + m * lda, sizeof(quad));
                    const __m512i av
                            = _mm512_set1_epi32(static_cast<int32_t>(quad));
                    ZENDNNL_NATIVE_INT8_UNROLL(4)
                    for (int v = 0; v < NV; ++v) {
                        acc[s][m][v] = U8
                                ? _mm512_dpbusd_epi32(acc[s][m][v], av, bv[v])
                                : _mm512_dpbusd_epi32(acc[s][m][v], bv[v], av);
                        ZENDNNL_NATIVE_INT8_ACC_BARRIER(acc[s][m][v]);
                    }
                }
            }
        }
        for (; kq < k_quads; ++kq) {
            const int8_t *bq = b + static_cast<size_t>(kq) * b_stride;
            const uint8_t *aq = a + static_cast<size_t>(kq) * 4;
            __m512i bv[NV];
            ZENDNNL_NATIVE_INT8_UNROLL(4)
            for (int v = 0; v < NV; ++v) {
                if constexpr (U8) {
                    bv[v] = _mm512_loadu_si512(bq + v * b_second);
                    col[0][v] = _mm512_dpbusd_epi32(col[0][v], ones, bv[v]);
                    ZENDNNL_NATIVE_INT8_ACC_BARRIER(col[0][v]);
                } else {
                    bv[v] = decode::bytes(
                            _mm512_loadu_si512(bq + v * b_second), flip);
                }
            }
            ZENDNNL_NATIVE_INT8_UNROLL(12)
            for (int m = 0; m < MR; ++m) {
                uint32_t quad;
                std::memcpy(&quad, aq + m * lda, sizeof(quad));
                const __m512i av
                        = _mm512_set1_epi32(static_cast<int32_t>(quad));
                ZENDNNL_NATIVE_INT8_UNROLL(4)
                for (int v = 0; v < NV; ++v) {
                    acc[0][m][v] = U8
                            ? _mm512_dpbusd_epi32(acc[0][m][v], av, bv[v])
                            : _mm512_dpbusd_epi32(acc[0][m][v], bv[v], av);
                    ZENDNNL_NATIVE_INT8_ACC_BARRIER(acc[0][m][v]);
                }
            }
        }
        ZENDNNL_NATIVE_INT8_UNROLL(4)
        for (int s = 1; s < kSets; ++s) {
            ZENDNNL_NATIVE_INT8_UNROLL(12)
            for (int m = 0; m < MR; ++m) {
                ZENDNNL_NATIVE_INT8_UNROLL(4)
                for (int v = 0; v < NV; ++v)
                    acc[0][m][v] = _mm512_add_epi32(acc[0][m][v], acc[s][m][v]);
            }
        }
        if constexpr (U8 && !Wide) {
            // The int32 arithmetic wraps: the difference is exact whenever
            // the element's sum fits int32.
            const __m512i zp = _mm512_set1_epi32(p.src_zp);
            ZENDNNL_NATIVE_INT8_UNROLL(4)
            for (int v = 0; v < NV; ++v) {
                __m512i sum = col[0][v];
                ZENDNNL_NATIVE_INT8_UNROLL(4)
                for (int s = 1; s < kSets; ++s)
                    sum = _mm512_add_epi32(sum, col[s][v]);
                const __m512i term = _mm512_mullo_epi32(sum, zp);
                ZENDNNL_NATIVE_INT8_UNROLL(12)
                for (int m = 0; m < MR; ++m)
                    acc[0][m][v] = _mm512_sub_epi32(acc[0][m][v], term);
            }
        }
        // Wide: the column sums, for the zero-point term in int64 at the
        // conversion.
        [[maybe_unused]] __m512i cols[NV];
        if constexpr (Wide) {
            ZENDNNL_NATIVE_INT8_UNROLL(4)
            for (int v = 0; v < NV; ++v) {
                cols[v] = col[0][v];
                ZENDNNL_NATIVE_INT8_UNROLL(4)
                for (int s = 1; s < kSets; ++s)
                    cols[v] = _mm512_add_epi32(cols[v], col[s][v]);
            }
        }

        // The result bits depend on this operation order: the exact sum
        // (int32, or int64 with Wide), conversion, source scale, weight
        // scale, one add.
        __m512 ws[NV];
        ZENDNNL_NATIVE_INT8_UNROLL(4)
        for (int v = 0; v < NV; ++v)
            ws[v] = native_scales_as_f32<S>(wei_scale + v * scale_second);
        ZENDNNL_NATIVE_INT8_UNROLL(12)
        for (int m = 0; m < MR; ++m) {
            const __m512 sa = _mm512_set1_ps(src_scale[m * p.src_scale_ld]);
            ZENDNNL_NATIVE_INT8_UNROLL(4)
            for (int v = 0; v < NV; ++v) {
                __m512 d;
                if constexpr (Wide)
                    d = sum_minus_zp_wide(acc[0][m][v], p.src_zp, cols[v]);
                else
                    d = _mm512_cvtepi32_ps(acc[0][m][v]);
                c[m][v] = _mm512_add_ps(c[m][v],
                        avx512_mul_unfused(_mm512_mul_ps(d, sa), ws[v]));
            }
        }
        a += p.a_step;
        b += p.b_step;
        // A u8 source has no shifts: its shift pointer may be null.
        if constexpr (!U8) shift += p.shift_step;
        src_scale += p.src_scale_step;
        wei_scale += wei_scale_step;
    }

    const size_t dst_ldc = static_cast<size_t>(p.dst_ldc);
    if (p.dst == nullptr) {
        float *const out = p.C;
        ZENDNNL_NATIVE_INT8_UNROLL(12)
        for (int m = 0; m < MR; ++m) {
            ZENDNNL_NATIVE_INT8_UNROLL(4)
            for (int v = 0; v < NV; ++v)
                _mm512_storeu_ps(out + m * ldc + v * 16, c[m][v]);
        }
    } else if (p.dst_bf16) {
        uint16_t *const out = static_cast<uint16_t *>(p.dst);
        ZENDNNL_NATIVE_INT8_UNROLL(12)
        for (int m = 0; m < MR; ++m) {
            ZENDNNL_NATIVE_INT8_UNROLL(4)
            for (int v = 0; v < NV; ++v)
                _mm256_storeu_si256(
                        reinterpret_cast<__m256i *>(out + m * dst_ldc + v * 16),
                        avx512_f32_to_bf16_rne(c[m][v]));
        }
    } else {
        float *const out = static_cast<float *>(p.dst);
        ZENDNNL_NATIVE_INT8_UNROLL(12)
        for (int m = 0; m < MR; ++m) {
            ZENDNNL_NATIVE_INT8_UNROLL(4)
            for (int v = 0; v < NV; ++v)
                _mm512_storeu_ps(out + m * dst_ldc + v * 16, c[m][v]);
        }
    }
}

/// The instance of the call's weight scale type, picked once per call: the
/// batch loop tests no type.
template <native_format_t F, int MR, int NV, bool U8, bool Wide>
void native_int8_group_ukernel_of_scale(
        const native_int8_group_ukernel_args_t &p) {
    switch (p.wei_scale_type) {
        case native_scale_t::f32:
            native_int8_group_ukernel<F, MR, NV, native_scale_t::f32, U8, Wide>(
                    p);
            return;
        case native_scale_t::bf16:
            native_int8_group_ukernel<F, MR, NV, native_scale_t::bf16, U8,
                    Wide>(p);
            return;
        case native_scale_t::f16:
        case native_scale_t::none: break;
    }
    assert(p.wei_scale_type == native_scale_t::f16);
    native_int8_group_ukernel<F, MR, NV, native_scale_t::f16, U8, Wide>(p);
}

template <native_format_t F, int NV, bool U8, bool Wide, int... I>
constexpr std::array<native_int8_group_ukernel_fn_t, sizeof...(I)>
make_group_table(std::integer_sequence<int, I...>) {
    return {{&native_int8_group_ukernel_of_scale<F, I + 1, NV, U8, Wide>...}};
}

/// The group kernels of a weight of format F by row count, at one and two
/// vectors, over an s8 source, a u8 one and a u8 one in the wide form.
template <native_format_t F>
struct int8_group_kernels_t {
    static constexpr auto kNv1 = make_group_table<F, 1, false, false>(
            std::make_integer_sequence<int, kMaxGroupMrNv1> {});
    static constexpr auto kNv2 = make_group_table<F, 2, false, false>(
            std::make_integer_sequence<int, kMaxGroupMrNv2> {});
    static constexpr auto kNv1U8 = make_group_table<F, 1, true, false>(
            std::make_integer_sequence<int, kMaxGroupMrNv1> {});
    static constexpr auto kNv2U8 = make_group_table<F, 2, true, false>(
            std::make_integer_sequence<int, kMaxGroupMrNv2> {});
    static constexpr auto kNv1U8Wide = make_group_table<F, 1, true, true>(
            std::make_integer_sequence<int, kMaxGroupMrNv1> {});
    static constexpr auto kNv2U8Wide = make_group_table<F, 2, true, true>(
            std::make_integer_sequence<int, kMaxGroupMrNv2> {});
};

/// One format's tables of int8_group_kernels_t.
struct int8_group_tables_t {
    const native_int8_group_ukernel_fn_t *nv1, *nv2, *nv1_u8, *nv2_u8,
            *nv1_u8_wide, *nv2_u8_wide;
};

template <native_format_t F>
constexpr int8_group_tables_t int8_group_tables_of() {
    using k = int8_group_kernels_t<F>;
    return {k::kNv1.data(), k::kNv2.data(), k::kNv1U8.data(), k::kNv2U8.data(),
            k::kNv1U8Wide.data(), k::kNv2U8Wide.data()};
}

/// The formats of the routes on these kernels, which the group kernels are
/// instantiated for, and their tables in the same order.
constexpr auto kFormats = native_family_formats<native_kernel_kind_t::int8>();

template <size_t... I>
constexpr std::array<int8_group_tables_t, sizeof...(I)> make_group_tables(
        std::index_sequence<I...>) {
    return {{int8_group_tables_of<kFormats[I]>()...}};
}

constexpr auto kGroupTables
        = make_group_tables(std::make_index_sequence<kFormats.size()> {});

/// The group tables of format; nullptr for a format of no route on these
/// kernels.
const int8_group_tables_t *group_tables_of(native_format_t format) {
    for (size_t i = 0; i < kFormats.size(); ++i)
        if (kFormats[i] == format) return &kGroupTables[i];
    return nullptr;
}

/// Whether the per-channel kernels serve format: a format of a route on
/// these kernels whose decode reads each byte as the weight plus 128, which
/// they take as the same sums of the weight's bytes as they are and the
/// source plus 128.
bool channel_kernels_take(native_format_t format) {
    return group_tables_of(format) != nullptr
            && native_format(format).decode == native_decode_t::byte_offset;
}

} // namespace

int native_int8_ukernel_max_mr(int nv) {
    return nv == 1 || nv == 2 ? kMaxMr : 0;
}

native_int8_ukernel_fn_t select_native_int8_ukernel(
        native_format_t format, int mr, int nv, bool src_signed) {
    if (!channel_kernels_take(format) || mr < 1
            || mr > native_int8_ukernel_max_mr(nv))
        return nullptr;
    if (nv == 1) return (src_signed ? kNv1Signed : kNv1)[mr - 1];
    return (src_signed ? kNv2Signed : kNv2)[mr - 1];
}

int native_int8_group_ukernel_max_mr(int nv) {
    return nv == 1 ? kMaxGroupMrNv1 : (nv == 2 ? kMaxGroupMrNv2 : 0);
}

native_int8_group_ukernel_fn_t select_native_int8_group_ukernel(
        native_format_t format, int mr, int nv, bool src_u8, bool wide_zp) {
    const int8_group_tables_t *t = group_tables_of(format);
    if (t == nullptr || mr < 1 || mr > native_int8_group_ukernel_max_mr(nv))
        return nullptr;
    if (nv == 1)
        return (src_u8 ? (wide_zp ? t->nv1_u8_wide : t->nv1_u8)
                       : t->nv1)[mr - 1];
    return (src_u8 ? (wide_zp ? t->nv2_u8_wide : t->nv2_u8) : t->nv2)[mr - 1];
}

bool native_int8_kernels_serve(
        native_format_t format, native_int8_scheme_t scheme, int mr, int nv) {
    if (scheme == native_int8_scheme_t::per_group)
        return select_native_int8_group_ukernel(format, mr, nv, false)
                != nullptr
                && select_native_int8_group_ukernel(format, mr, nv, true)
                != nullptr;
    return select_native_int8_ukernel(format, mr, nv, false) != nullptr
            && select_native_int8_ukernel(format, mr, nv, true) != nullptr;
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
