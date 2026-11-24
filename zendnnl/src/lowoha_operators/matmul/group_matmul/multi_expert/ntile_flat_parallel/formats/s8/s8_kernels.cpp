/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
 *******************************************************************************/

/// s8 kernels: symmetric s8 weights against s8 rows.
///
/// Per K quad, the step's weights load once and each row broadcasts four
/// source bytes into VPDPBUSD, XOR-ed with 0x80 to make them the unsigned
/// operand unless the rows already are.  With one weight scale per column and
/// one source scale per row, the sums run over all of K, and the epilogue
/// removes the resulting `128 * sum(w)` from each column, then scales by the
/// row's and the column's scales and adds the bias in one FMA.  With per-group
/// weight scales or per-group source scales, the group kernels start each
/// scale group's sums from the group's `-128 * sum(w)` and scale them into
/// FP32 totals by the group's weight scales, times the row's scale of the
/// group under per-group source scales; after the last group the totals take
/// the row's scale (per-row source scales), then the bias.

#include <cstdint>
#include <cstring>

#include <immintrin.h>

#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/formats/s8/s8.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/kernel/nr32_epilogue.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {
namespace nr32 {

namespace {

using namespace kernel;

/// What the kernels store: their row's output type.
constexpr data_type_t kDst = s8::kFormat.dst;

/// Bytes from one block row to the next.
constexpr int kRow = 128;

// MSVC (cl) has no x64 inline asm, so the accumulator register barrier is
// GCC/Clang only.
#if defined(_MSC_VER) && !defined(__clang__)
#define NR32_S8_BARRIER(lo, hi)
#else
#define NR32_S8_BARRIER(lo, hi) asm("" : "+v"(lo), "+v"(hi))
#endif

ZENDNNL_TARGET("avx512f,avx512dq,fma")
static inline __m512 dequant(__m512i acc, __m512i correction, __m512 scale,
        __m512 bias, bool has_bias) {
    const __m512 f = _mm512_cvtepi32_ps(_mm512_sub_epi32(acc, correction));
    return has_bias ? _mm512_fmadd_ps(f, scale, bias) : _mm512_mul_ps(f, scale);
}

template <int MR, epilogue Ep, bool kSrcU8, bool kUnaligned>
ZENDNNL_TARGET_NOINLINE("avx512f,avx512vnni,avx512bw,avx512vl,avx512dq,fma")
static void s8_kernel(const ukernel_args &p) {
    static_assert(MR >= 1 && MR <= 12, "s8 kernels take 1..12 rows");
    static_assert(!kUnaligned || split(Ep),
            "only split epilogues read misaligned columns");
    const auto *__restrict A = static_cast<const uint8_t *>(p.a);
    const int lda = p.lda;
    const auto *__restrict B = reinterpret_cast<const int8_t *>(p.b);
    const ptrdiff_t next = p.b_next;
    const int K_quad = (p.K + 3) / 4;

    const __m512i zero = _mm512_setzero_si512();
    __m512i c00 = zero, c01 = zero, c10 = zero, c11 = zero;
    __m512i c20 = zero, c21 = zero, c30 = zero, c31 = zero;
    __m512i c40 = zero, c41 = zero, c50 = zero, c51 = zero;
    __m512i c60 = zero, c61 = zero, c70 = zero, c71 = zero;
    __m512i c80 = zero, c81 = zero, c90 = zero, c91 = zero;
    __m512i c100 = zero, c101 = zero, c110 = zero, c111 = zero;

    const __m512i s8_to_u8
            = _mm512_set1_epi32(static_cast<int32_t>(0x80808080U));
    if constexpr (kSrcU8) (void)s8_to_u8;
    [[maybe_unused]] __m512i shift_idx = zero;
    [[maybe_unused]] const int8_t *__restrict B_hi = nullptr;
    if constexpr (kUnaligned) {
        shift_idx = shift_index32(p.shift);
        B_hi = reinterpret_cast<const int8_t *>(p.b_next_hi);
    }
    for (int kq = 0; kq < K_quad; ++kq) {
        const int8_t *bp = B + static_cast<size_t>(kq) * kRow;
        const __m512i b0
                = _mm512_load_si512(reinterpret_cast<const __m512i *>(bp));
        __m512i b1;
        if constexpr (kUnaligned) {
            b1 = _mm512_permutex2var_epi32(
                    _mm512_load_si512(
                            reinterpret_cast<const __m512i *>(bp + next)),
                    shift_idx,
                    _mm512_load_si512(reinterpret_cast<const __m512i *>(
                            B_hi + static_cast<size_t>(kq) * kRow)));
        } else {
            b1 = _mm512_load_si512(
                    reinterpret_cast<const __m512i *>(bp + next));
        }
#define NR32_S8_ROW(r) \
    if constexpr (MR > r) { \
        uint32_t a_quad; \
        std::memcpy(&a_quad, A + static_cast<size_t>(r) * lda + kq * 4, \
                sizeof(a_quad)); \
        __m512i av = _mm512_set1_epi32(static_cast<int32_t>(a_quad)); \
        if constexpr (!kSrcU8) { av = _mm512_xor_si512(av, s8_to_u8); } \
        c##r##0 = _mm512_dpbusd_epi32(c##r##0, av, b0); \
        c##r##1 = _mm512_dpbusd_epi32(c##r##1, av, b1); \
        NR32_S8_BARRIER(c##r##0, c##r##1); \
    }
        NR32_S8_ROW(0)
        NR32_S8_ROW(1)
        NR32_S8_ROW(2)
        NR32_S8_ROW(3)
        NR32_S8_ROW(4)
        NR32_S8_ROW(5)
        NR32_S8_ROW(6)
        NR32_S8_ROW(7)
        NR32_S8_ROW(8)
        NR32_S8_ROW(9)
        NR32_S8_ROW(10)
        NR32_S8_ROW(11)
#undef NR32_S8_ROW
    }

    const int cols = p.cols;
    const __m512i k_sym = _mm512_set1_epi32(128);
    const __m512i comp0 = _mm512_mullo_epi32(k_sym, _mm512_loadu_si512(p.comp));
    const __m512i comp1 = _mm512_mullo_epi32(
            k_sym, _mm512_loadu_si512(p.comp + p.col_next));
    const __m512 ws0 = _mm512_loadu_ps(p.scale);
    const __m512 ws1 = _mm512_loadu_ps(p.scale + p.col_next);
    const bool has_bias = p.bias != nullptr && p.bias_t != bias_kind::none;
    const __m512 bias0 = has_bias
            ? load_bias16(p.bias, p.bias_t, 0, lanes16<Ep>(cols, 0))
            : _mm512_setzero_ps();
    const __m512 bias1 = has_bias
            ? load_bias16(p.bias, p.bias_t, p.col_next, lanes16<Ep>(cols, 1))
            : _mm512_setzero_ps();
    auto *C = static_cast<out_elem_t<kDst> *>(p.c);
    const int ldc = p.ldc;
    [[maybe_unused]] const act_mode_t mode
            = gated(Ep) ? act_mode() : act_mode_t {};

#define NR32_S8_STORE(r) \
    if constexpr (MR > r) { \
        const __m512 ss = _mm512_set1_ps(p.src_scale[r]); \
        const __m512 lo = dequant( \
                c##r##0, comp0, _mm512_mul_ps(ss, ws0), bias0, has_bias); \
        const __m512 hi = dequant( \
                c##r##1, comp1, _mm512_mul_ps(ss, ws1), bias1, has_bias); \
        store_step<Ep, kDst>( \
                C + static_cast<size_t>(r) * ldc, lo, hi, cols, mode); \
    }
    NR32_S8_STORE(0)
    NR32_S8_STORE(1)
    NR32_S8_STORE(2)
    NR32_S8_STORE(3)
    NR32_S8_STORE(4)
    NR32_S8_STORE(5)
    NR32_S8_STORE(6)
    NR32_S8_STORE(7)
    NR32_S8_STORE(8)
    NR32_S8_STORE(9)
    NR32_S8_STORE(10)
    NR32_S8_STORE(11)
#undef NR32_S8_STORE
}
// Wide split: the gate block's two vectors at `b`, the up block's at
// `b + b_next`; output column j of the step is act(gate j) * up j.
template <int MR, epilogue Ep, bool kSrcU8>
ZENDNNL_TARGET_NOINLINE("avx512f,avx512vnni,avx512bw,avx512vl,avx512dq,fma")
static void s8_wide_kernel(const ukernel_args &p) {
    static_assert(MR >= 1 && MR <= 6, "s8 wide kernels take 1..6 rows");
    static_assert(split(Ep), "wide kernels serve split epilogues");
    constexpr int NV = 4;
    const auto *__restrict A = static_cast<const uint8_t *>(p.a);
    const int lda = p.lda;
    const auto *__restrict B = reinterpret_cast<const int8_t *>(p.b);
    const ptrdiff_t next = p.b_next;
    const int K_quad = (p.K + 3) / 4;

    __m512i c[MR][NV];
#pragma GCC unroll 6
    for (int m = 0; m < MR; ++m)
#pragma GCC unroll 4
        for (int v = 0; v < NV; ++v)
            c[m][v] = _mm512_setzero_si512();
    const __m512i s8_to_u8
            = _mm512_set1_epi32(static_cast<int32_t>(0x80808080U));
    if constexpr (kSrcU8) (void)s8_to_u8;
    for (int kq = 0; kq < K_quad; ++kq) {
        const int8_t *bp = B + static_cast<size_t>(kq) * kRow;
        const __m512i b0
                = _mm512_load_si512(reinterpret_cast<const __m512i *>(bp));
        const __m512i b1
                = _mm512_load_si512(reinterpret_cast<const __m512i *>(bp + 64));
        const __m512i b2 = _mm512_load_si512(
                reinterpret_cast<const __m512i *>(bp + next));
        const __m512i b3 = _mm512_load_si512(
                reinterpret_cast<const __m512i *>(bp + next + 64));
#pragma GCC unroll 6
        for (int m = 0; m < MR; ++m) {
            uint32_t a_quad;
            std::memcpy(&a_quad, A + static_cast<size_t>(m) * lda + kq * 4,
                    sizeof(a_quad));
            __m512i av = _mm512_set1_epi32(static_cast<int32_t>(a_quad));
            if constexpr (!kSrcU8) av = _mm512_xor_si512(av, s8_to_u8);
            c[m][0] = _mm512_dpbusd_epi32(c[m][0], av, b0);
            c[m][1] = _mm512_dpbusd_epi32(c[m][1], av, b1);
            c[m][2] = _mm512_dpbusd_epi32(c[m][2], av, b2);
            c[m][3] = _mm512_dpbusd_epi32(c[m][3], av, b3);
            NR32_S8_BARRIER(c[m][0], c[m][1]);
            NR32_S8_BARRIER(c[m][2], c[m][3]);
        }
    }

    const int cols = p.cols;
    const int col_of[NV] = {0, 16, p.col_next, p.col_next + 16};
    const int lanes_of[NV] = {cols, cols - 16, cols, cols - 16};
    const __m512i k_sym = _mm512_set1_epi32(128);
    const bool has_bias = p.bias != nullptr && p.bias_t != bias_kind::none;
    __m512i comp[NV];
    __m512 ws[NV], bias[NV];
#pragma GCC unroll 4
    for (int v = 0; v < NV; ++v) {
        comp[v] = _mm512_mullo_epi32(
                k_sym, _mm512_loadu_si512(p.comp + col_of[v]));
        ws[v] = _mm512_loadu_ps(p.scale + col_of[v]);
        bias[v] = has_bias
                ? load_bias16(p.bias, p.bias_t, col_of[v], lanes_of[v])
                : _mm512_setzero_ps();
    }
    auto *C = static_cast<out_elem_t<kDst> *>(p.c);
    const act_mode_t mode = act_mode();
#pragma GCC unroll 6
    for (int m = 0; m < MR; ++m) {
        const __m512 ss = _mm512_set1_ps(p.src_scale[m]);
        __m512 f[NV];
#pragma GCC unroll 4
        for (int v = 0; v < NV; ++v) {
            f[v] = dequant(c[m][v], comp[v], _mm512_mul_ps(ss, ws[v]), bias[v],
                    has_bias);
        }
        store_wide_step<Ep, kDst>(C + static_cast<size_t>(m) * p.ldc, f[0],
                f[1], f[2], f[3], cols, mode);
    }
}

/// Adds group `g`'s int32 sums `sacc` of `MR` rows and `NV` vectors, the
/// vectors' weight scales at `scale + col_of[v]`, into the FP32 totals.  The
/// totals stay in L1 so the sums keep the registers the per-channel kernels
/// give them, and as many rows.
template <int MR, int NV, bool kSrcGroups>
ZENDNNL_TARGET("avx512f,avx512dq,fma")
static inline void scale_group(float (&facc)[MR][NV][16],
        const __m512i (&sacc)[MR][NV], const float *scale,
        const int (&col_of)[NV], const ukernel_args &p, const int g) {
#pragma GCC unroll 4
    for (int v = 0; v < NV; ++v) {
        const __m512 ws = _mm512_loadu_ps(scale + col_of[v]);
#pragma GCC unroll 12
        for (int m = 0; m < MR; ++m) {
            const __m512 f = _mm512_cvtepi32_ps(sacc[m][v]);
            __m512 scale_mv = ws;
            if constexpr (kSrcGroups) {
                // A zero sum adds zero, even where the scale product
                // overflows.
                scale_mv = _mm512_maskz_mul_ps(
                        _mm512_cmpneq_epi32_mask(
                                sacc[m][v], _mm512_setzero_si512()),
                        ws,
                        _mm512_set1_ps(p.src_scale[static_cast<size_t>(m)
                                        * p.src_groups
                                + g]));
            }
            _mm512_store_ps(facc[m][v],
                    _mm512_fmadd_ps(f, scale_mv, _mm512_load_ps(facc[m][v])));
        }
    }
}

/// Zeroes the FP32 totals of a group kernel.
template <int MR, int NV>
ZENDNNL_TARGET("avx512f")
static inline void zero_totals(float (&facc)[MR][NV][16]) {
#pragma GCC unroll 12
    for (int m = 0; m < MR; ++m)
#pragma GCC unroll 4
        for (int v = 0; v < NV; ++v)
            _mm512_store_ps(facc[m][v], _mm512_setzero_ps());
}

/// Row `m`'s output of a group kernel from its FP32 totals: times the row's
/// scale under per-row source scales, then the bias.
template <bool kSrcGroups>
ZENDNNL_TARGET("avx512f,avx512dq,fma")
static inline __m512 group_out(const __m512 total, const ukernel_args &p,
        const int m, const __m512 bias, const bool has_bias) {
    __m512 f = total;
    if constexpr (!kSrcGroups) {
        f = _mm512_mul_ps(f, _mm512_set1_ps(p.src_scale[m]));
    }
    return has_bias ? _mm512_add_ps(f, bias) : f;
}

template <int MR, epilogue Ep, bool kSrcU8, bool kUnaligned, bool kSrcGroups>
ZENDNNL_TARGET_NOINLINE("avx512f,avx512vnni,avx512bw,avx512vl,avx512dq,fma")
static void s8_group_kernel(const ukernel_args &p) {
    static_assert(MR >= 1 && MR <= 12, "s8 group kernels take 1..12 rows");
    static_assert(!kUnaligned || split(Ep),
            "only split epilogues read misaligned columns");
    constexpr int NV = 2;
    const auto *__restrict A = static_cast<const uint8_t *>(p.a);
    const int lda = p.lda;
    const auto *__restrict B = reinterpret_cast<const int8_t *>(p.b);
    const ptrdiff_t next = p.b_next;
    [[maybe_unused]] const auto *__restrict B_hi
            = reinterpret_cast<const int8_t *>(p.b_next_hi);
    [[maybe_unused]] __m512i shift_idx = _mm512_setzero_si512();
    if constexpr (kUnaligned) shift_idx = shift_index32(p.shift);
    const int quads_per_group = p.group_size / 4;
    const int G = p.K / p.group_size;
    const size_t col_group = static_cast<size_t>(p.col_group);
    const int col_of[NV] = {0, p.col_next};
    const __m512i s8_to_u8
            = _mm512_set1_epi32(static_cast<int32_t>(0x80808080U));
    if constexpr (kSrcU8) (void)s8_to_u8;
    const __m512i neg_sym = _mm512_set1_epi32(-128);

    alignas(64) float facc[MR][NV][16];
    zero_totals(facc);

    for (int g = 0; g < G; ++g) {
        const int32_t *comp = p.comp + static_cast<size_t>(g) * col_group;
        __m512i sacc[MR][NV];
#pragma GCC unroll 2
        for (int v = 0; v < NV; ++v) {
            const __m512i seed = _mm512_mullo_epi32(
                    neg_sym, _mm512_loadu_si512(comp + col_of[v]));
#pragma GCC unroll 12
            for (int m = 0; m < MR; ++m)
                sacc[m][v] = seed;
        }
        for (int ql = 0; ql < quads_per_group; ++ql) {
            const int kq = g * quads_per_group + ql;
            const int8_t *bp = B + static_cast<size_t>(kq) * kRow;
            const __m512i b0
                    = _mm512_load_si512(reinterpret_cast<const __m512i *>(bp));
            __m512i b1;
            if constexpr (kUnaligned) {
                b1 = _mm512_permutex2var_epi32(
                        _mm512_load_si512(
                                reinterpret_cast<const __m512i *>(bp + next)),
                        shift_idx,
                        _mm512_load_si512(reinterpret_cast<const __m512i *>(
                                B_hi + static_cast<size_t>(kq) * kRow)));
            } else {
                b1 = _mm512_load_si512(
                        reinterpret_cast<const __m512i *>(bp + next));
            }
#pragma GCC unroll 12
            for (int m = 0; m < MR; ++m) {
                uint32_t a_quad;
                std::memcpy(&a_quad, A + static_cast<size_t>(m) * lda + kq * 4,
                        sizeof(a_quad));
                __m512i av = _mm512_set1_epi32(static_cast<int32_t>(a_quad));
                if constexpr (!kSrcU8) av = _mm512_xor_si512(av, s8_to_u8);
                sacc[m][0] = _mm512_dpbusd_epi32(sacc[m][0], av, b0);
                sacc[m][1] = _mm512_dpbusd_epi32(sacc[m][1], av, b1);
                NR32_S8_BARRIER(sacc[m][0], sacc[m][1]);
            }
        }
        scale_group<MR, NV, kSrcGroups>(facc, sacc,
                p.scale + static_cast<size_t>(g) * col_group, col_of, p, g);
    }

    const int cols = p.cols;
    const bool has_bias = p.bias != nullptr && p.bias_t != bias_kind::none;
    const __m512 bias0 = has_bias
            ? load_bias16(p.bias, p.bias_t, 0, lanes16<Ep>(cols, 0))
            : _mm512_setzero_ps();
    const __m512 bias1 = has_bias
            ? load_bias16(p.bias, p.bias_t, p.col_next, lanes16<Ep>(cols, 1))
            : _mm512_setzero_ps();
    auto *C = static_cast<out_elem_t<kDst> *>(p.c);
    [[maybe_unused]] const act_mode_t mode
            = gated(Ep) ? act_mode() : act_mode_t {};
#pragma GCC unroll 12
    for (int m = 0; m < MR; ++m) {
        const __m512 lo = group_out<kSrcGroups>(
                _mm512_load_ps(facc[m][0]), p, m, bias0, has_bias);
        const __m512 hi = group_out<kSrcGroups>(
                _mm512_load_ps(facc[m][1]), p, m, bias1, has_bias);
        store_step<Ep, kDst>(
                C + static_cast<size_t>(m) * p.ldc, lo, hi, cols, mode);
    }
}

// Wide split by scale group: the gate block's two vectors at `b`, the up
// block's at `b + b_next`.
template <int MR, epilogue Ep, bool kSrcU8, bool kSrcGroups>
ZENDNNL_TARGET_NOINLINE("avx512f,avx512vnni,avx512bw,avx512vl,avx512dq,fma")
static void s8_group_wide_kernel(const ukernel_args &p) {
    static_assert(MR >= 1 && MR <= 6, "s8 wide group kernels take 1..6 rows");
    static_assert(split(Ep), "wide kernels serve split epilogues");
    constexpr int NV = 4;
    const auto *__restrict A = static_cast<const uint8_t *>(p.a);
    const int lda = p.lda;
    const auto *__restrict B = reinterpret_cast<const int8_t *>(p.b);
    const ptrdiff_t next = p.b_next;
    const ptrdiff_t v_off[NV] = {0, 64, next, next + 64};
    const int quads_per_group = p.group_size / 4;
    const int G = p.K / p.group_size;
    const size_t col_group = static_cast<size_t>(p.col_group);
    const int col_of[NV] = {0, 16, p.col_next, p.col_next + 16};
    const __m512i s8_to_u8
            = _mm512_set1_epi32(static_cast<int32_t>(0x80808080U));
    if constexpr (kSrcU8) (void)s8_to_u8;
    const __m512i neg_sym = _mm512_set1_epi32(-128);

    alignas(64) float facc[MR][NV][16];
    zero_totals(facc);

    for (int g = 0; g < G; ++g) {
        const int32_t *comp = p.comp + static_cast<size_t>(g) * col_group;
        __m512i sacc[MR][NV];
#pragma GCC unroll 4
        for (int v = 0; v < NV; ++v) {
            const __m512i seed = _mm512_mullo_epi32(
                    neg_sym, _mm512_loadu_si512(comp + col_of[v]));
#pragma GCC unroll 6
            for (int m = 0; m < MR; ++m)
                sacc[m][v] = seed;
        }
        for (int ql = 0; ql < quads_per_group; ++ql) {
            const int kq = g * quads_per_group + ql;
            const int8_t *bp = B + static_cast<size_t>(kq) * kRow;
            __m512i bv[NV];
#pragma GCC unroll 4
            for (int v = 0; v < NV; ++v) {
                bv[v] = _mm512_load_si512(
                        reinterpret_cast<const __m512i *>(bp + v_off[v]));
            }
#pragma GCC unroll 6
            for (int m = 0; m < MR; ++m) {
                uint32_t a_quad;
                std::memcpy(&a_quad, A + static_cast<size_t>(m) * lda + kq * 4,
                        sizeof(a_quad));
                __m512i av = _mm512_set1_epi32(static_cast<int32_t>(a_quad));
                if constexpr (!kSrcU8) av = _mm512_xor_si512(av, s8_to_u8);
#pragma GCC unroll 4
                for (int v = 0; v < NV; ++v)
                    sacc[m][v] = _mm512_dpbusd_epi32(sacc[m][v], av, bv[v]);
                NR32_S8_BARRIER(sacc[m][0], sacc[m][1]);
                NR32_S8_BARRIER(sacc[m][2], sacc[m][3]);
            }
        }
        scale_group<MR, NV, kSrcGroups>(facc, sacc,
                p.scale + static_cast<size_t>(g) * col_group, col_of, p, g);
    }

    const int cols = p.cols;
    const int lanes_of[NV] = {cols, cols - 16, cols, cols - 16};
    const bool has_bias = p.bias != nullptr && p.bias_t != bias_kind::none;
    __m512 bias[NV];
#pragma GCC unroll 4
    for (int v = 0; v < NV; ++v) {
        bias[v] = has_bias
                ? load_bias16(p.bias, p.bias_t, col_of[v], lanes_of[v])
                : _mm512_setzero_ps();
    }
    auto *C = static_cast<out_elem_t<kDst> *>(p.c);
    const act_mode_t mode = act_mode();
#pragma GCC unroll 6
    for (int m = 0; m < MR; ++m) {
        __m512 f[NV];
#pragma GCC unroll 4
        for (int v = 0; v < NV; ++v) {
            f[v] = group_out<kSrcGroups>(
                    _mm512_load_ps(facc[m][v]), p, m, bias[v], has_bias);
        }
        store_wide_step<Ep, kDst>(C + static_cast<size_t>(m) * p.ldc, f[0],
                f[1], f[2], f[3], cols, mode);
    }
}

#undef NR32_S8_BARRIER

template <epilogue Ep, bool kSrcU8, bool kSrcGroups>
ukernel_fn pick_groups_wide(const int mr) {
    switch (mr) {
        case 1: return s8_group_wide_kernel<1, Ep, kSrcU8, kSrcGroups>;
        case 2: return s8_group_wide_kernel<2, Ep, kSrcU8, kSrcGroups>;
        case 3: return s8_group_wide_kernel<3, Ep, kSrcU8, kSrcGroups>;
        case 4: return s8_group_wide_kernel<4, Ep, kSrcU8, kSrcGroups>;
        case 5: return s8_group_wide_kernel<5, Ep, kSrcU8, kSrcGroups>;
        case 6: return s8_group_wide_kernel<6, Ep, kSrcU8, kSrcGroups>;
        default: return nullptr;
    }
}

template <epilogue Ep, bool kSrcU8, bool kUnaligned, bool kSrcGroups>
ukernel_fn pick_groups(const int mr) {
    switch (mr) {
        case 1: return s8_group_kernel<1, Ep, kSrcU8, kUnaligned, kSrcGroups>;
        case 2: return s8_group_kernel<2, Ep, kSrcU8, kUnaligned, kSrcGroups>;
        case 3: return s8_group_kernel<3, Ep, kSrcU8, kUnaligned, kSrcGroups>;
        case 4: return s8_group_kernel<4, Ep, kSrcU8, kUnaligned, kSrcGroups>;
        case 5: return s8_group_kernel<5, Ep, kSrcU8, kUnaligned, kSrcGroups>;
        case 6: return s8_group_kernel<6, Ep, kSrcU8, kUnaligned, kSrcGroups>;
        case 7: return s8_group_kernel<7, Ep, kSrcU8, kUnaligned, kSrcGroups>;
        case 8: return s8_group_kernel<8, Ep, kSrcU8, kUnaligned, kSrcGroups>;
        case 9: return s8_group_kernel<9, Ep, kSrcU8, kUnaligned, kSrcGroups>;
        case 10: return s8_group_kernel<10, Ep, kSrcU8, kUnaligned, kSrcGroups>;
        case 11: return s8_group_kernel<11, Ep, kSrcU8, kUnaligned, kSrcGroups>;
        case 12: return s8_group_kernel<12, Ep, kSrcU8, kUnaligned, kSrcGroups>;
        default: return nullptr;
    }
}

template <bool kSrcU8, bool kSrcGroups>
ukernel_fn pick_groups_ep(
        const int mr, const epilogue ep, const bool unaligned) {
    switch (ep) {
        case epilogue::plain:
            return unaligned
                    ? nullptr
                    : pick_groups<epilogue::plain, kSrcU8, false, kSrcGroups>(
                              mr);
        case epilogue::swiglu_pairs:
            return unaligned ? nullptr
                             : pick_groups<epilogue::swiglu_pairs, kSrcU8,
                                       false, kSrcGroups>(mr);
        case epilogue::silu_split:
            return unaligned ? pick_groups<epilogue::silu_split, kSrcU8, true,
                                       kSrcGroups>(mr)
                             : pick_groups<epilogue::silu_split, kSrcU8, false,
                                       kSrcGroups>(mr);
        case epilogue::gelu_split:
            return unaligned ? pick_groups<epilogue::gelu_split, kSrcU8, true,
                                       kSrcGroups>(mr)
                             : pick_groups<epilogue::gelu_split, kSrcU8, false,
                                       kSrcGroups>(mr);
    }
    return nullptr;
}

template <bool kSrcU8, bool kSrcGroups>
ukernel_fn pick_groups_wide_ep(const int mr, const epilogue ep) {
    switch (ep) {
        case epilogue::silu_split:
            return pick_groups_wide<epilogue::silu_split, kSrcU8, kSrcGroups>(
                    mr);
        case epilogue::gelu_split:
            return pick_groups_wide<epilogue::gelu_split, kSrcU8, kSrcGroups>(
                    mr);
        default: return nullptr;
    }
}

template <epilogue Ep, bool kSrcU8>
ukernel_fn pick_wide(const int mr) {
    switch (mr) {
        case 1: return s8_wide_kernel<1, Ep, kSrcU8>;
        case 2: return s8_wide_kernel<2, Ep, kSrcU8>;
        case 3: return s8_wide_kernel<3, Ep, kSrcU8>;
        case 4: return s8_wide_kernel<4, Ep, kSrcU8>;
        case 5: return s8_wide_kernel<5, Ep, kSrcU8>;
        case 6: return s8_wide_kernel<6, Ep, kSrcU8>;
        default: return nullptr;
    }
}

template <epilogue Ep, bool kSrcU8, bool kUnaligned>
ukernel_fn pick(const int mr) {
    switch (mr) {
        case 1: return s8_kernel<1, Ep, kSrcU8, kUnaligned>;
        case 2: return s8_kernel<2, Ep, kSrcU8, kUnaligned>;
        case 3: return s8_kernel<3, Ep, kSrcU8, kUnaligned>;
        case 4: return s8_kernel<4, Ep, kSrcU8, kUnaligned>;
        case 5: return s8_kernel<5, Ep, kSrcU8, kUnaligned>;
        case 6: return s8_kernel<6, Ep, kSrcU8, kUnaligned>;
        case 7: return s8_kernel<7, Ep, kSrcU8, kUnaligned>;
        case 8: return s8_kernel<8, Ep, kSrcU8, kUnaligned>;
        case 9: return s8_kernel<9, Ep, kSrcU8, kUnaligned>;
        case 10: return s8_kernel<10, Ep, kSrcU8, kUnaligned>;
        case 11: return s8_kernel<11, Ep, kSrcU8, kUnaligned>;
        case 12: return s8_kernel<12, Ep, kSrcU8, kUnaligned>;
        default: return nullptr;
    }
}

template <bool kSrcU8>
ukernel_fn pick_ep(const int mr, const epilogue ep, const bool unaligned) {
    switch (ep) {
        case epilogue::plain:
            return unaligned ? nullptr
                             : pick<epilogue::plain, kSrcU8, false>(mr);
        case epilogue::swiglu_pairs:
            return unaligned ? nullptr
                             : pick<epilogue::swiglu_pairs, kSrcU8, false>(mr);
        case epilogue::silu_split:
            return unaligned ? pick<epilogue::silu_split, kSrcU8, true>(mr)
                             : pick<epilogue::silu_split, kSrcU8, false>(mr);
        case epilogue::gelu_split:
            return unaligned ? pick<epilogue::gelu_split, kSrcU8, true>(mr)
                             : pick<epilogue::gelu_split, kSrcU8, false>(mr);
    }
    return nullptr;
}

} // namespace

namespace s8 {

ukernel_fn narrow(const int mr, const epilogue ep, const bool src_u8,
        const bool unaligned, const scale_groups groups) {
    if (!groups.any()) {
        return src_u8 ? pick_ep<true>(mr, ep, unaligned)
                      : pick_ep<false>(mr, ep, unaligned);
    }
    if (src_u8) {
        return groups.source ? pick_groups_ep<true, true>(mr, ep, unaligned)
                             : pick_groups_ep<true, false>(mr, ep, unaligned);
    }
    return groups.source ? pick_groups_ep<false, true>(mr, ep, unaligned)
                         : pick_groups_ep<false, false>(mr, ep, unaligned);
}

ukernel_fn wide(const int mr, const epilogue ep, const bool src_u8, bool,
        const scale_groups groups) {
    if (groups.any()) {
        if (src_u8) {
            return groups.source ? pick_groups_wide_ep<true, true>(mr, ep)
                                 : pick_groups_wide_ep<true, false>(mr, ep);
        }
        return groups.source ? pick_groups_wide_ep<false, true>(mr, ep)
                             : pick_groups_wide_ep<false, false>(mr, ep);
    }
    switch (ep) {
        case epilogue::silu_split:
            return src_u8 ? pick_wide<epilogue::silu_split, true>(mr)
                          : pick_wide<epilogue::silu_split, false>(mr);
        case epilogue::gelu_split:
            return src_u8 ? pick_wide<epilogue::gelu_split, true>(mr)
                          : pick_wide<epilogue::gelu_split, false>(mr);
        default: return nullptr;
    }
}

} // namespace s8

} // namespace nr32
} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
