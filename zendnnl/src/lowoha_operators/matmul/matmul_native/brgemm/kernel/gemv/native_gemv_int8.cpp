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

#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/gemv/native_gemv_ukernel.hpp"

#include <array>
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

/// Bytes of one K group of a block: 32 columns of a quad (s8) or an octet
/// (s4), 4 bytes each.
constexpr size_t kGroupBytes = 128;

/// The source's quad at a, broadcast.
ZENDNNL_INLINE_TARGET("avx512f")
inline __m512i quad_at(const uint8_t *a) {
    uint32_t q;
    std::memcpy(&q, a, sizeof(q));
    return _mm512_set1_epi32(static_cast<int32_t>(q));
}

/// The two 16-column vectors of block i at off: the second 64 bytes on in
/// the block, or with Runs at hi[i].
template <int NB, bool Runs>
ZENDNNL_INLINE_TARGET("avx512f")
inline const uint8_t *second_of(
        const uint8_t *const (&lo)[NB], const uint8_t *const (&hi)[NB], int i) {
    return Runs ? hi[i] : lo[i] + 64;
}

/// The per-channel sums of NB blocks into acc.  Few blocks split K over
/// independent sets of accumulators so enough VPDPBUSD chains are in flight;
/// the int32 sum is exact in any order.
template <int NB, bool Runs, bool Signed>
ZENDNNL_INLINE_TARGET("avx512f,avx512bw,avx512vl,avx512vnni")
inline void s8_sums(const uint8_t *__restrict a, int k_quads,
        const native_gemv_cols_t &cols, __m512i (&sum)[NB][2]) {
    constexpr int kSets = NB == 1 ? 4 : 2;
    const uint8_t *lo[NB];
    const uint8_t *hi[NB];
    for (int i = 0; i < NB; ++i) {
        lo[i] = cols.lo[i];
        hi[i] = Runs ? cols.hi[i] : nullptr;
    }
    __m512i acc[kSets][NB][2];
    for (int t = 0; t < kSets; ++t)
        for (int i = 0; i < NB; ++i)
            acc[t][i][0] = acc[t][i][1] = _mm512_setzero_si512();
    const __m512i flip = _mm512_set1_epi32(static_cast<int32_t>(0x80808080u));
    const int k_main = k_quads - k_quads % kSets;
    int kq = 0;
    size_t off = 0;
    for (; kq < k_main; kq += kSets) {
        for (int t = 0; t < kSets; ++t, off += kGroupBytes) {
            __m512i av = quad_at(a + static_cast<size_t>(kq + t) * 4);
            if constexpr (Signed) av = _mm512_xor_si512(av, flip);
            for (int i = 0; i < NB; ++i) {
                acc[t][i][0] = _mm512_dpbusd_epi32(
                        acc[t][i][0], av, _mm512_loadu_si512(lo[i] + off));
                acc[t][i][1] = _mm512_dpbusd_epi32(acc[t][i][1], av,
                        _mm512_loadu_si512(
                                second_of<NB, Runs>(lo, hi, i) + off));
            }
        }
    }
    for (; kq < k_quads; ++kq, off += kGroupBytes) {
        __m512i av = quad_at(a + static_cast<size_t>(kq) * 4);
        if constexpr (Signed) av = _mm512_xor_si512(av, flip);
        for (int i = 0; i < NB; ++i) {
            acc[0][i][0] = _mm512_dpbusd_epi32(
                    acc[0][i][0], av, _mm512_loadu_si512(lo[i] + off));
            acc[0][i][1] = _mm512_dpbusd_epi32(acc[0][i][1], av,
                    _mm512_loadu_si512(second_of<NB, Runs>(lo, hi, i) + off));
        }
    }
    for (int i = 0; i < NB; ++i)
        for (int v = 0; v < 2; ++v) {
            sum[i][v] = acc[0][i][v];
            for (int t = 1; t < kSets; ++t)
                sum[i][v] = _mm512_add_epi32(sum[i][v], acc[t][i][v]);
        }
}

/// The per-channel sums.
template <int NB, bool Runs, bool Signed>
ZENDNNL_TARGET_NOINLINE("avx512f,avx512bw,avx512vl,avx512vnni")
void s8_gemv(const uint8_t *__restrict a, int k_quads,
        const native_gemv_cols_t &cols, int32_t *__restrict s) {
    __m512i sum[NB][2];
    s8_sums<NB, Runs, Signed>(a, k_quads, cols, sum);
    for (int i = 0; i < NB; ++i)
        for (int v = 0; v < 2; ++v)
            _mm512_storeu_si512(s + 32 * i + 16 * v, sum[i][v]);
}

/// 16 values of a per-column vector of f32, bf16 or f16 at element at, or
/// with per_column false its one value, as f32; exact: a bias of type dt.
ZENDNNL_INLINE_TARGET("avx512f,avx512bw,avx512vl")
inline __m512 column_f32(
        const void *v, data_type_t dt, bool per_column, size_t at) {
    if (!per_column) {
        const float x = dt == data_type_t::f32
                ? *static_cast<const float *>(v)
                : _mm512_cvtss_f32(dt == data_type_t::bf16
                                  ? avx512_bf16_to_f32(_mm256_set1_epi16(
                                            *static_cast<const short *>(v)))
                                  : _mm512_cvtph_ps(_mm256_set1_epi16(
                                            *static_cast<const short *>(v))));
        return _mm512_set1_ps(x);
    }
    if (dt == data_type_t::f32)
        return _mm512_loadu_ps(static_cast<const float *>(v) + at);
    const __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(
            static_cast<const uint16_t *>(v) + at));
    return dt == data_type_t::bf16 ? avx512_bf16_to_f32(h) : _mm512_cvtph_ps(h);
}

/// column_f32 of weight scales of type s.
ZENDNNL_INLINE_TARGET("avx512f,avx512bw,avx512vl")
inline __m512 column_f32(
        const void *v, native_scale_t s, bool per_column, size_t at) {
    if (!per_column) {
        switch (s) {
            case native_scale_t::f32:
                return _mm512_set1_ps(*static_cast<const float *>(v));
            case native_scale_t::f16:
            case native_scale_t::bf16:
            case native_scale_t::none: break;
        }
        return _mm512_set1_ps(_mm512_cvtss_f32(native_halves_as_f32(
                _mm256_set1_epi16(*static_cast<const short *>(v)), s)));
    }
    switch (s) {
        case native_scale_t::f32:
            return _mm512_loadu_ps(static_cast<const float *>(v) + at);
        case native_scale_t::f16:
        case native_scale_t::bf16:
        case native_scale_t::none: break;
    }
    return native_halves_as_f32(
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(
                    static_cast<const uint16_t *>(v) + at)),
            s);
}

/// f32(acc - zp * cs) for 16 lanes with the zero-point term in int64: exact
/// before the one rounding to f32.
ZENDNNL_INLINE_TARGET("avx512f,avx512dq")
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

/// The per-channel sums finished in the fused order, as the per-channel
/// kernel finishes them: the int32 compensation (int64 with wide_zp),
/// conversion, the scale product, one FMA with the bias.
template <int NB, bool Runs, bool Signed>
ZENDNNL_TARGET_NOINLINE("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni,fma")
void s8_fused_gemv(const native_s8_gemv_args_t &p) {
    __m512i sum[NB][2];
    s8_sums<NB, Runs, Signed>(p.a, p.k_quads, p.cols, sum);
    const __m512i zp = _mm512_set1_epi32(static_cast<int32_t>(
            static_cast<uint32_t>(p.zp) + (Signed ? 128u : 0u)));
    const int64_t wide = static_cast<int64_t>(p.zp) + (Signed ? 128 : 0);
    const __m512 src_scale = _mm512_set1_ps(p.src_scale);
    for (int i = 0; i < NB; ++i)
        for (int v = 0; v < 2; ++v) {
            const int col = v == 0 ? p.col_lo[i] : p.col_hi[i];
            const __m512i cs
                    = _mm512_loadu_si512(v == 0 ? p.cs_lo[i] : p.cs_hi[i]);
            const __m512 f = p.wide_zp
                    ? sum_minus_zp_wide(sum[i][v], wide, cs)
                    : _mm512_cvtepi32_ps(_mm512_sub_epi32(
                              sum[i][v], _mm512_mullo_epi32(zp, cs)));
            const __m512 scale = _mm512_mul_ps(src_scale,
                    column_f32(p.wei_scale, p.wei_scale_type,
                            p.wei_scale_per_column, col));
            const __m512 out = p.bias != nullptr
                    ? _mm512_fmadd_ps(f, scale,
                              column_f32(p.bias, p.bias_dt, true, col))
                    : _mm512_mul_ps(f, scale);
            _mm512_storeu_ps(p.c + 32 * i + 16 * v, out);
        }
}

/// 16 weight scales of group g at column col, as f32; exact.
ZENDNNL_INLINE_TARGET("avx512f,avx512bw,avx512vl")
inline __m512 group_scales(const native_gemv_group_args_t &p, int g, int col) {
    if (p.unit_scales) return _mm512_set1_ps(1.0f);
    const size_t at = static_cast<size_t>(g) * p.wei_scale_ld + col;
    switch (p.wei_scale_type) {
        case native_scale_t::f32:
            return _mm512_loadu_ps(
                    static_cast<const float *>(p.wei_scale) + at);
        case native_scale_t::f16:
        case native_scale_t::bf16:
        case native_scale_t::none: break;
    }
    return native_halves_as_f32(
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(
                    static_cast<const uint16_t *>(p.wei_scale) + at)),
            p.wei_scale_type);
}

/// The column of vector v of block i.
inline int col_of(const native_gemv_group_args_t &p, int i, int v) {
    return v == 0 ? p.col_lo[i] : p.col_hi[i];
}

/// The sums of the 32 * NB columns to p.c.
template <int NB>
ZENDNNL_INLINE_TARGET("avx512f")
inline void store_sums(const native_gemv_group_args_t &p, __m512 (&c)[NB][2]) {
    for (int i = 0; i < NB; ++i)
        for (int v = 0; v < 2; ++v)
            _mm512_storeu_ps(p.c + 32 * i + 16 * v, c[i][v]);
}

/// One K quad of a weight of format F with per-group weight scales at off
/// into acc, and over a u8 source the column sums of its weight quads into
/// col; over an s8 source the weight read by F's decode, as w + 128.
template <native_format_t F, int NB, bool Runs, bool U8>
ZENDNNL_INLINE_TARGET("avx512f,avx512bw,avx512vnni")
inline void group_quad_sums(const uint8_t *const (&lo)[NB],
        const uint8_t *const (&hi)[NB], size_t off, const uint8_t *a,
        __m512i (&acc)[NB][2], __m512i (&col)[NB][2]) {
    using decode = native_decode_of<F>;
    const __m512i flip = _mm512_set1_epi32(decode::kFlip);
    const __m512i ones = _mm512_set1_epi8(1);
    const __m512i av = quad_at(a);
    for (int i = 0; i < NB; ++i) {
        const __m512i b0 = _mm512_loadu_si512(lo[i] + off);
        const __m512i b1
                = _mm512_loadu_si512(second_of<NB, Runs>(lo, hi, i) + off);
        if constexpr (U8) {
            col[i][0] = _mm512_dpbusd_epi32(col[i][0], ones, b0);
            col[i][1] = _mm512_dpbusd_epi32(col[i][1], ones, b1);
            acc[i][0] = _mm512_dpbusd_epi32(acc[i][0], av, b0);
            acc[i][1] = _mm512_dpbusd_epi32(acc[i][1], av, b1);
        } else {
            acc[i][0] = _mm512_dpbusd_epi32(
                    acc[i][0], decode::bytes(b0, flip), av);
            acc[i][1] = _mm512_dpbusd_epi32(
                    acc[i][1], decode::bytes(b1, flip), av);
        }
    }
}

/// The group kernel of a weight of format F on the INT8 kernels with
/// per-group weight scales: per group the exact sum (s8 source: from its
/// shift, the weight read by F's decode as w + 128; u8 source: less the zero
/// point times the group's column sums), converted, times the source scale,
/// times the weight scale, one rounding each, added to the column's sum.
template <native_format_t F, int NB, bool Runs, bool U8>
ZENDNNL_TARGET_NOINLINE("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni")
void int8_group_gemv(const native_gemv_group_args_t &p) {
    const uint8_t *lo[NB];
    const uint8_t *hi[NB];
    for (int i = 0; i < NB; ++i) {
        lo[i] = p.cols.lo[i];
        hi[i] = Runs ? p.cols.hi[i] : nullptr;
    }
    __m512 c[NB][2];
    for (int i = 0; i < NB; ++i)
        c[i][0] = c[i][1] = _mm512_setzero_ps();
    const uint8_t *a = p.a;
    size_t base = 0;
    for (int g = 0; g < p.groups; ++g, a += p.a_step, base += p.b_step) {
        // The quads alternate between two sets of accumulators; the int32
        // sum is exact in any order.
        __m512i acc[2][NB][2];
        __m512i col[2][NB][2];
        const __m512i start
                = U8 ? _mm512_setzero_si512() : _mm512_set1_epi32(p.shift[g]);
        for (int i = 0; i < NB; ++i)
            for (int v = 0; v < 2; ++v) {
                acc[0][i][v] = start;
                acc[1][i][v] = col[0][i][v] = col[1][i][v]
                        = _mm512_setzero_si512();
            }
        size_t off = base;
        int kq = 0;
        for (; kq + 1 < p.k_quads; kq += 2, off += 2 * kGroupBytes) {
            group_quad_sums<F, NB, Runs, U8>(lo, hi, off,
                    a + static_cast<size_t>(kq) * 4, acc[0], col[0]);
            group_quad_sums<F, NB, Runs, U8>(lo, hi, off + kGroupBytes,
                    a + static_cast<size_t>(kq + 1) * 4, acc[1], col[1]);
        }
        if (kq < p.k_quads)
            group_quad_sums<F, NB, Runs, U8>(lo, hi, off,
                    a + static_cast<size_t>(kq) * 4, acc[0], col[0]);
        for (int i = 0; i < NB; ++i)
            for (int v = 0; v < 2; ++v) {
                acc[0][i][v] = _mm512_add_epi32(acc[0][i][v], acc[1][i][v]);
                col[0][i][v] = _mm512_add_epi32(col[0][i][v], col[1][i][v]);
            }
        const __m512 sa = _mm512_set1_ps(p.src_scale[g * p.src_scale_step]);
        for (int i = 0; i < NB; ++i)
            for (int v = 0; v < 2; ++v) {
                __m512 d;
                if (U8 && p.wide_zp) {
                    d = sum_minus_zp_wide(acc[0][i][v], p.src_zp, col[0][i][v]);
                } else {
                    __m512i sum = acc[0][i][v];
                    if (U8)
                        sum = _mm512_sub_epi32(sum,
                                _mm512_mullo_epi32(col[0][i][v],
                                        _mm512_set1_epi32(p.src_zp)));
                    d = _mm512_cvtepi32_ps(sum);
                }
                c[i][v] = _mm512_add_ps(c[i][v],
                        avx512_mul_unfused(_mm512_mul_ps(d, sa),
                                group_scales(p, g, col_of(p, i, v))));
            }
    }
    store_sums<NB>(p, c);
}

/// acc[i][v] += the dot products of one K quad, the high nibble plane
/// (High) or the low one of the octets at off as Decode reads them with low,
/// with the source's quad at a.
template <int NB, bool Runs, bool High, typename Decode>
ZENDNNL_INLINE_TARGET("avx512f,avx512bw,avx512vnni")
inline void quad_sums(const uint8_t *const (&lo)[NB],
        const uint8_t *const (&hi)[NB], size_t off, const uint8_t *a,
        __m512i (&acc)[NB][2], __m512i low) {
    const __m512i av = quad_at(a);
    for (int i = 0; i < NB; ++i)
        for (int v = 0; v < 2; ++v) {
            const __m512i bits = _mm512_loadu_si512(
                    (v == 0 ? lo[i] : second_of<NB, Runs>(lo, hi, i)) + off);
            const __m512i w = Decode::template plane<High>(bits, low);
            acc[i][v] = _mm512_dpbusd_epi32(acc[i][v], w, av);
        }
}

/// set[i][v] += the dot products of the two K quads of the octets at off,
/// the low nibble plane then the high one as Decode reads them with low,
/// with the source's quads at a and a + 4.
template <int NB, bool Runs, typename Decode>
ZENDNNL_INLINE_TARGET("avx512f,avx512bw,avx512vnni")
inline void octet_sums(const uint8_t *const (&lo)[NB],
        const uint8_t *const (&hi)[NB], size_t off, const uint8_t *a,
        __m512i (&set)[NB][2], __m512i low) {
    const __m512i q0 = quad_at(a);
    const __m512i q1 = quad_at(a + 4);
    for (int i = 0; i < NB; ++i)
        for (int v = 0; v < 2; ++v) {
            const __m512i bits = _mm512_loadu_si512(
                    (v == 0 ? lo[i] : second_of<NB, Runs>(lo, hi, i)) + off);
            set[i][v] = _mm512_dpbusd_epi32(
                    set[i][v], Decode::template plane<false>(bits, low), q0);
            set[i][v] = _mm512_dpbusd_epi32(
                    set[i][v], Decode::template plane<true>(bits, low), q1);
        }
}

/// The group kernel of a weight of format F on the INT4 kernels
/// (native_int4_group_ukernel_fn_t's steps): per group the exact sum from
/// its shift over the weight read by F's decode as w + 8, then the step of
/// the call's order.
template <native_format_t F, int NB, bool Runs>
ZENDNNL_TARGET_NOINLINE("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni,fma")
void int4_group_gemv(const native_gemv_group_args_t &p) {
    const uint8_t *lo[NB];
    const uint8_t *hi[NB];
    for (int i = 0; i < NB; ++i) {
        lo[i] = p.cols.lo[i];
        hi[i] = Runs ? p.cols.hi[i] : nullptr;
    }
    using decode = native_decode_of<F>;
    const __m512i low = _mm512_set1_epi8(decode::kLow);
    const bool fused = p.order == native_int8_order_t::fused;
    __m512 c[NB][2];
    for (int i = 0; i < NB; ++i)
        c[i][0] = c[i][1] = _mm512_setzero_ps();
    const uint8_t *a = p.a;
    // The group's first K quad, counted from the octets at lo / hi.
    int first = p.quad0;
    for (int g = 0; g < p.groups; ++g, a += p.a_step) {
        // The octets alternate between two sets of accumulators; the int32
        // sum is exact in any order.
        __m512i acc[NB][2];
        __m512i odd[NB][2];
        const __m512i start = _mm512_set1_epi32(p.shift[g]);
        for (int i = 0; i < NB; ++i)
            for (int v = 0; v < 2; ++v) {
                acc[i][v] = start;
                odd[i][v] = _mm512_setzero_si512();
            }
        // A group of an odd number of quads starts or ends half way
        // through an octet.
        int taken = 0;
        if ((first & 1) != 0) {
            quad_sums<NB, Runs, true, decode>(lo, hi,
                    static_cast<size_t>(first >> 1) * kGroupBytes, a, acc, low);
            taken = 1;
        }
        const int octets = (p.k_quads - taken) / 2;
        size_t off = static_cast<size_t>((first + taken) >> 1) * kGroupBytes;
        const uint8_t *ao = a + taken * 4;
        int o = 0;
        for (; o + 1 < octets; o += 2, off += 2 * kGroupBytes, ao += 16) {
            octet_sums<NB, Runs, decode>(lo, hi, off, ao, acc, low);
            octet_sums<NB, Runs, decode>(
                    lo, hi, off + kGroupBytes, ao + 8, odd, low);
        }
        if (o < octets) {
            octet_sums<NB, Runs, decode>(lo, hi, off, ao, acc, low);
            off += kGroupBytes;
            ao += 8;
        }
        if (taken + 2 * octets < p.k_quads)
            quad_sums<NB, Runs, false, decode>(lo, hi, off, ao, acc, low);
        for (int i = 0; i < NB; ++i)
            for (int v = 0; v < 2; ++v)
                acc[i][v] = _mm512_add_epi32(acc[i][v], odd[i][v]);
        const float sa_g
                = p.unit_scales ? 1.0f : p.src_scale[g * p.src_scale_step];
        const __m512 sa = _mm512_set1_ps(sa_g);
        for (int i = 0; i < NB; ++i)
            for (int v = 0; v < 2; ++v) {
                const __m512 ws = group_scales(p, g, col_of(p, i, v));
                const __m512i s = acc[i][v];
                if (fused && p.src_groups)
                    c[i][v] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s),
                            _mm512_maskz_mul_ps(
                                    _mm512_test_epi32_mask(s, s), sa, ws),
                            c[i][v]);
                else if (fused)
                    c[i][v] = _mm512_fmadd_ps(
                            _mm512_cvtepi32_ps(s), ws, c[i][v]);
                else
                    c[i][v] = _mm512_add_ps(c[i][v],
                            avx512_mul_unfused(
                                    _mm512_mul_ps(_mm512_cvtepi32_ps(s), sa),
                                    ws));
            }
        first += p.k_quads;
    }
    if (fused && !p.src_groups && p.finish) {
        const __m512 sa = _mm512_set1_ps(p.src_scale[0]);
        for (int i = 0; i < NB; ++i)
            for (int v = 0; v < 2; ++v)
                c[i][v] = _mm512_mul_ps(c[i][v], sa);
    }
    store_sums<NB>(p, c);
}

template <bool Runs, bool Signed, int... I>
constexpr std::array<native_s8_gemv_fn_t, sizeof...(I)> make_s8_table(
        std::integer_sequence<int, I...>) {
    return {{&s8_gemv<I + 1, Runs, Signed>...}};
}

template <native_format_t F, bool Runs, bool U8, int... I>
constexpr std::array<native_gemv_group_fn_t, sizeof...(I)>
make_int8_group_table(std::integer_sequence<int, I...>) {
    return {{&int8_group_gemv<F, I + 1, Runs, U8>...}};
}

template <native_format_t F, bool Runs, int... I>
constexpr std::array<native_gemv_group_fn_t, sizeof...(I)>
make_int4_group_table(std::integer_sequence<int, I...>) {
    return {{&int4_group_gemv<F, I + 1, Runs>...}};
}

using blocks_t = std::make_integer_sequence<int, kNativeGemvMaxIntBlocks>;

template <bool Runs, bool Signed, int... I>
constexpr std::array<native_s8_fused_gemv_fn_t, sizeof...(I)>
make_s8_fused_table(std::integer_sequence<int, I...>) {
    return {{&s8_fused_gemv<I + 1, Runs, Signed>...}};
}

constexpr auto kS8 = make_s8_table<false, false>(blocks_t {});
constexpr auto kS8Signed = make_s8_table<false, true>(blocks_t {});
constexpr auto kS8Runs = make_s8_table<true, false>(blocks_t {});
constexpr auto kS8RunsSigned = make_s8_table<true, true>(blocks_t {});
constexpr auto kS8Fused = make_s8_fused_table<false, false>(blocks_t {});
constexpr auto kS8FusedSigned = make_s8_fused_table<false, true>(blocks_t {});
constexpr auto kS8FusedRuns = make_s8_fused_table<true, false>(blocks_t {});
constexpr auto kS8FusedRunsSigned
        = make_s8_fused_table<true, true>(blocks_t {});

/// The group kernels of a weight of format F, by blocks: on the INT8 kernels
/// over an s8 source and a u8 one, on the INT4 kernels over an s8 source
/// (its u8 tables none), each with runs and without.
struct group_tables_t {
    const native_gemv_group_fn_t *s8, *u8, *s8_runs, *u8_runs;
};

template <native_format_t F>
struct int8_group_gemvs_t {
    static constexpr auto kS8
            = make_int8_group_table<F, false, false>(blocks_t {});
    static constexpr auto kU8
            = make_int8_group_table<F, false, true>(blocks_t {});
    static constexpr auto kS8Runs
            = make_int8_group_table<F, true, false>(blocks_t {});
    static constexpr auto kU8Runs
            = make_int8_group_table<F, true, true>(blocks_t {});
};

template <native_format_t F>
struct int4_group_gemvs_t {
    static constexpr auto kS8 = make_int4_group_table<F, false>(blocks_t {});
    static constexpr auto kS8Runs = make_int4_group_table<F, true>(blocks_t {});
};

/// The formats of the routes on the INT8 and on the INT4 kernels.
constexpr auto kInt8Formats
        = native_family_formats<native_kernel_kind_t::int8>();
constexpr auto kInt4Formats
        = native_family_formats<native_kernel_kind_t::int4>();

template <size_t... I>
constexpr std::array<group_tables_t, sizeof...(I)> make_int8_tables(
        std::index_sequence<I...>) {
    return {{{int8_group_gemvs_t<kInt8Formats[I]>::kS8.data(),
            int8_group_gemvs_t<kInt8Formats[I]>::kU8.data(),
            int8_group_gemvs_t<kInt8Formats[I]>::kS8Runs.data(),
            int8_group_gemvs_t<kInt8Formats[I]>::kU8Runs.data()}...}};
}

template <size_t... I>
constexpr std::array<group_tables_t, sizeof...(I)> make_int4_tables(
        std::index_sequence<I...>) {
    return {{{int4_group_gemvs_t<kInt4Formats[I]>::kS8.data(), nullptr,
            int4_group_gemvs_t<kInt4Formats[I]>::kS8Runs.data(), nullptr}...}};
}

constexpr auto kInt8Tables
        = make_int8_tables(std::make_index_sequence<kInt8Formats.size()> {});
constexpr auto kInt4Tables
        = make_int4_tables(std::make_index_sequence<kInt4Formats.size()> {});

/// The group kernels of format; nullptr for a format of no route on the
/// INT8 or INT4 kernels.
const group_tables_t *group_tables_of(native_format_t format) {
    for (size_t i = 0; i < kInt8Formats.size(); ++i)
        if (kInt8Formats[i] == format) return &kInt8Tables[i];
    for (size_t i = 0; i < kInt4Formats.size(); ++i)
        if (kInt4Formats[i] == format) return &kInt4Tables[i];
    return nullptr;
}

/// Whether the per-channel kernels serve format: a format of a route on the
/// INT8 kernels whose decode reads each byte as the weight plus 128, which
/// they take as the same sums of the weight's bytes as they are and the
/// source plus 128.
bool channel_gemvs_take(native_format_t format) {
    for (const native_format_t f : kInt8Formats)
        if (f == format)
            return native_format(format).decode == native_decode_t::byte_offset;
    return false;
}

} // namespace

native_s8_gemv_fn_t select_native_s8_gemv(
        int nb, bool runs, native_format_t format, bool src_signed) {
    if (!channel_gemvs_take(format) || nb < 1 || nb > kNativeGemvMaxIntBlocks)
        return nullptr;
    if (runs) return (src_signed ? kS8RunsSigned : kS8Runs)[nb - 1];
    return (src_signed ? kS8Signed : kS8)[nb - 1];
}

native_s8_fused_gemv_fn_t select_native_s8_fused_gemv(
        int nb, bool runs, native_format_t format, bool src_signed) {
    if (!channel_gemvs_take(format) || nb < 1 || nb > kNativeGemvMaxIntBlocks)
        return nullptr;
    if (runs) return (src_signed ? kS8FusedRunsSigned : kS8FusedRuns)[nb - 1];
    return (src_signed ? kS8FusedSigned : kS8Fused)[nb - 1];
}

native_gemv_group_fn_t select_native_group_gemv(
        int nb, bool runs, native_format_t format, bool src_u8) {
    const group_tables_t *t = group_tables_of(format);
    if (t == nullptr || nb < 1 || nb > kNativeGemvMaxIntBlocks) return nullptr;
    const native_gemv_group_fn_t *k = runs ? (src_u8 ? t->u8_runs : t->s8_runs)
                                           : (src_u8 ? t->u8 : t->s8);
    return k == nullptr ? nullptr : k[nb - 1];
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
