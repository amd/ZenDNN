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
#include "lowoha_operators/matmul/matmul_native/common/avx512_math.hpp"

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

/// f32(acc - zp * col_sum) for 16 lanes with the zero-point term in int64:
/// exact before the one rounding to f32, so equal to the int32 form wherever
/// that one does not wrap.
ZENDNNL_TARGET("avx512f,avx512dq")
inline __m512 sum_minus_zp_wide(
        __m512i acc, int64_t zp, const int32_t *col_sum) {
    const __m512i z = _mm512_set1_epi64(zp);
    const __m512i cs = _mm512_loadu_si512(col_sum);
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
                bv[v] = _mm512_loadu_si512(bq + v * 64);
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
            bv[v] = _mm512_loadu_si512(bq + v * 64);
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
                        p.col_sum + v * 16);
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
/// the conversion and the scale widening.  At one vector the rows are
/// kMaxMr's.
constexpr int kMaxGroupMrNv1 = kMaxMr;
constexpr int kMaxGroupMrNv2 = 6;

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

/// The group kernel for weight scales of type S.
template <int MR, int NV, data_type_t S>
ZENDNNL_TARGET_NOINLINE("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni")
void native_int8_group_ukernel(const native_int8_group_ukernel_args_t &p) {
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

    const __m512i flip = _mm512_set1_epi32(static_cast<int32_t>(0x80808080u));
    const size_t lda = static_cast<size_t>(p.lda);
    const size_t b_stride = static_cast<size_t>(p.b_stride);
    const int k_quads = p.k_quads;
    const int k_main = k_quads - k_quads % kSets;
    const uint8_t *a = p.A;
    const int8_t *b = p.B;
    const int32_t *shift = p.shift;
    const float *src_scale = p.src_scale;
    constexpr size_t wei_scale_esz = S == data_type_t::f32 ? 4 : 2;
    const char *wei_scale = static_cast<const char *>(p.wei_scale);
    const size_t wei_scale_step
            = static_cast<size_t>(p.wei_scale_step) * wei_scale_esz;
    for (int g = 0; g < p.batch; ++g) {
        // The sum of each row starts at its shift; the other sets at zero.
        __m512i acc[kSets][MR][NV];
        ZENDNNL_NATIVE_INT8_UNROLL(12)
        for (int m = 0; m < MR; ++m) {
            const __m512i start = _mm512_set1_epi32(shift[m * p.shift_ld]);
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
        int kq = 0;
        for (; kq < k_main; kq += kSets) {
            ZENDNNL_NATIVE_INT8_UNROLL(4)
            for (int s = 0; s < kSets; ++s) {
                const int8_t *bq = b + static_cast<size_t>(kq + s) * b_stride;
                const uint8_t *aq = a + static_cast<size_t>(kq + s) * 4;
                __m512i bv[NV];
                ZENDNNL_NATIVE_INT8_UNROLL(4)
                for (int v = 0; v < NV; ++v)
                    bv[v] = _mm512_xor_si512(
                            _mm512_loadu_si512(bq + v * 64), flip);
                ZENDNNL_NATIVE_INT8_UNROLL(12)
                for (int m = 0; m < MR; ++m) {
                    uint32_t quad;
                    std::memcpy(&quad, aq + m * lda, sizeof(quad));
                    const __m512i av
                            = _mm512_set1_epi32(static_cast<int32_t>(quad));
                    ZENDNNL_NATIVE_INT8_UNROLL(4)
                    for (int v = 0; v < NV; ++v) {
                        acc[s][m][v]
                                = _mm512_dpbusd_epi32(acc[s][m][v], bv[v], av);
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
                bv[v] = _mm512_xor_si512(_mm512_loadu_si512(bq + v * 64), flip);
            ZENDNNL_NATIVE_INT8_UNROLL(12)
            for (int m = 0; m < MR; ++m) {
                uint32_t quad;
                std::memcpy(&quad, aq + m * lda, sizeof(quad));
                const __m512i av
                        = _mm512_set1_epi32(static_cast<int32_t>(quad));
                ZENDNNL_NATIVE_INT8_UNROLL(4)
                for (int v = 0; v < NV; ++v) {
                    acc[0][m][v] = _mm512_dpbusd_epi32(acc[0][m][v], bv[v], av);
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

        // The result bits depend on this operation order: the exact int32
        // sum, conversion, source scale, weight scale, one add.
        __m512 ws[NV];
        ZENDNNL_NATIVE_INT8_UNROLL(4)
        for (int v = 0; v < NV; ++v)
            ws[v] = scales_as_f32<S>(wei_scale + v * 16 * wei_scale_esz);
        ZENDNNL_NATIVE_INT8_UNROLL(12)
        for (int m = 0; m < MR; ++m) {
            const __m512 sa = _mm512_set1_ps(src_scale[m * p.src_scale_ld]);
            ZENDNNL_NATIVE_INT8_UNROLL(4)
            for (int v = 0; v < NV; ++v) {
                const __m512 d = _mm512_cvtepi32_ps(acc[0][m][v]);
                c[m][v] = _mm512_add_ps(c[m][v],
                        avx512_mul_unfused(_mm512_mul_ps(d, sa), ws[v]));
            }
        }
        a += p.a_step;
        b += p.b_step;
        shift += p.shift_step;
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
template <int MR, int NV>
void native_int8_group_ukernel_of_dt(
        const native_int8_group_ukernel_args_t &p) {
    if (p.wei_scale_dt == data_type_t::f32)
        native_int8_group_ukernel<MR, NV, data_type_t::f32>(p);
    else if (p.wei_scale_dt == data_type_t::bf16)
        native_int8_group_ukernel<MR, NV, data_type_t::bf16>(p);
    else {
        assert(p.wei_scale_dt == data_type_t::f16);
        native_int8_group_ukernel<MR, NV, data_type_t::f16>(p);
    }
}

template <int NV, int... I>
constexpr std::array<native_int8_group_ukernel_fn_t, sizeof...(I)>
make_group_table(std::integer_sequence<int, I...>) {
    return {{&native_int8_group_ukernel_of_dt<I + 1, NV>...}};
}

constexpr auto kGroupNv1 = make_group_table<1>(
        std::make_integer_sequence<int, kMaxGroupMrNv1> {});
constexpr auto kGroupNv2 = make_group_table<2>(
        std::make_integer_sequence<int, kMaxGroupMrNv2> {});

} // namespace

int native_int8_ukernel_max_mr(int nv) {
    return nv == 1 || nv == 2 ? kMaxMr : 0;
}

native_int8_ukernel_fn_t select_native_int8_ukernel(
        int mr, int nv, bool src_signed) {
    if (mr < 1 || mr > native_int8_ukernel_max_mr(nv)) return nullptr;
    if (nv == 1) return (src_signed ? kNv1Signed : kNv1)[mr - 1];
    return (src_signed ? kNv2Signed : kNv2)[mr - 1];
}

int native_int8_group_ukernel_max_mr(int nv) {
    return nv == 1 ? kMaxGroupMrNv1 : (nv == 2 ? kMaxGroupMrNv2 : 0);
}

native_int8_group_ukernel_fn_t select_native_int8_group_ukernel(
        int mr, int nv) {
    if (mr < 1 || mr > native_int8_group_ukernel_max_mr(nv)) return nullptr;
    return nv == 1 ? kGroupNv1[mr - 1] : kGroupNv2[mr - 1];
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
