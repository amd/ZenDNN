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

#ifndef MATMUL_NATIVE_BRGEMM_LOOPER_NATIVE_INT8_TILE_INTERNAL_HPP
#define MATMUL_NATIVE_BRGEMM_LOOPER_NATIVE_INT8_TILE_INTERNAL_HPP

#include <algorithm>
#include <cstddef>

#include <immintrin.h>

#include "common/zendnnl_compat.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_int8_tile.hpp"
#include "lowoha_operators/matmul/matmul_native/common/avx512_math.hpp"
#include "memory/memory_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

/// Rows and columns of one kernel call, which the per-call buffers of the
/// tiles hold: up to 12 rows, a width of 16 or the widest call width.
inline constexpr int kMaxTileMr = 12;
inline constexpr int kMaxTileNr = kNativeInt8MaxCallCols;

/// The f32 values at weight columns p0 .. p0 + cw (cw a multiple of 16) of
/// a per-column (stride 1) or per-tensor (stride 0) vector of f32, bf16 or
/// f16: in place for f32 per column, else converted, exactly, into buf.
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl")
static inline const float *column_values(
        const void *v, data_type_t dt, int stride, int p0, int cw, float *buf) {
    if (stride == 0) {
        std::fill(
                buf, buf + cw, zendnnl::memory::read_and_cast<float>(v, dt, 0));
        return buf;
    }
    if (dt == data_type_t::f32) return static_cast<const float *>(v) + p0;
    const uint16_t *h = static_cast<const uint16_t *>(v) + p0;
    for (int j = 0; j < cw; j += 16) {
        const __m256i x
                = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(h + j));
        _mm512_storeu_ps(buf + j,
                dt == data_type_t::bf16 ? avx512_bf16_to_f32(x)
                                        : _mm512_cvtph_ps(x));
    }
    return buf;
}

/// column_values of a call of 32 columns whose first 16 are the weight
/// columns p0 .. p0 + 16 and whose second 16 are p1 .. p1 + 16 (a paired
/// tile's up run, or the next block), into buf.
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl")
static inline const float *column_values_apart(
        const void *v, data_type_t dt, int stride, int p0, int p1, float *buf) {
    for (int h = 0; h < 2; ++h) {
        float *out = buf + 16 * h;
        const float *in
                = column_values(v, dt, stride, h == 0 ? p0 : p1, 16, out);
        if (in != out) _mm512_storeu_ps(out, _mm512_loadu_ps(in));
    }
    return buf;
}

/// The column sums of that call, into buf.
inline const int32_t *col_sums_apart(
        const native_weight_view_t &w, int p0, int p1, int32_t *buf) {
    std::copy(w.col_sum(p0), w.col_sum(p0) + 16, buf);
    std::copy(w.col_sum(p1), w.col_sum(p1) + 16, buf + 16);
    return buf;
}

/// The weight columns of the second 16 columns of the call of nv vectors at
/// column t of the tile of pack columns n0 .. n0 + nr: the next 16 columns
/// of the block, or on a paired view the up run (native_weight_view_t).
inline int second_col(
        const native_weight_view_t &w, int n0, int nr, int t, int nv, int p0) {
    return w.paired && nv == 2 ? w.tile_col(n0, nr, t + 16) : p0 + 16;
}

/// Whether that call's second 16 columns lie elsewhere than right after its
/// first 16 inside one block.
inline bool second_apart(
        const native_weight_view_t &w, int nv, int p0, int p1) {
    return nv == 2 && (p1 != p0 + 16 || p0 % w.pack_nr != 0);
}

/// column_values of the columns p0 .. p0 + cols only (cols <= cw, the
/// columns below the weight's N), the other cw - cols values zero, into
/// buf: no element of v is read past the weight's N columns.
static inline const float *column_values_below(const void *v, data_type_t dt,
        int stride, int p0, int cw, int cols, float *buf) {
    for (int j = 0; j < cw; ++j)
        buf[j] = j < cols ? zendnnl::memory::read_and_cast<float>(
                                    v, dt, stride == 0 ? 0 : p0 + j)
                          : 0.0f;
    return buf;
}

/// The destination at row and columns p .. p + 16, read as float the way
/// the reference reads it: exactly, an s32 value rounded to nearest.
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl")
static inline __m512 dst_values(
        const native_int8_call_t &call, int row, int p) {
    const size_t at = static_cast<size_t>(row) * call.ldc + p;
    switch (call.dst_dt) {
        case data_type_t::f32:
            return _mm512_loadu_ps(static_cast<const float *>(call.dst) + at);
        case data_type_t::bf16:
        case data_type_t::f16: {
            const __m256i h
                    = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(
                            static_cast<const uint16_t *>(call.dst) + at));
            return call.dst_dt == data_type_t::bf16 ? avx512_bf16_to_f32(h)
                                                    : _mm512_cvtph_ps(h);
        }
        case data_type_t::s8:
            return _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
                    _mm_loadu_si128(reinterpret_cast<const __m128i *>(
                            static_cast<const int8_t *>(call.dst) + at))));
        case data_type_t::u8:
            return _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(
                    _mm_loadu_si128(reinterpret_cast<const __m128i *>(
                            static_cast<const uint8_t *>(call.dst) + at))));
        case data_type_t::s32:
            return _mm512_cvtepi32_ps(_mm512_loadu_si512(
                    static_cast<const int32_t *>(call.dst) + at));
        default: break;
    }
    alignas(64) float v[16];
    for (int l = 0; l < 16; ++l)
        v[l] = zendnnl::memory::read_and_cast<float>(
                call.dst, call.dst_dt, at + l);
    return _mm512_load_ps(v);
}

/// The k lanes of the destination at element at, as dst_values reads them.
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl")
static inline __m512 dst_lanes(
        const native_int8_call_t &call, size_t at, __mmask16 k) {
    switch (call.dst_dt) {
        case data_type_t::f32:
            return _mm512_maskz_loadu_ps(
                    k, static_cast<const float *>(call.dst) + at);
        case data_type_t::bf16:
        case data_type_t::f16: {
            const __m256i h = _mm256_maskz_loadu_epi16(
                    k, static_cast<const uint16_t *>(call.dst) + at);
            return call.dst_dt == data_type_t::bf16 ? avx512_bf16_to_f32(h)
                                                    : _mm512_cvtph_ps(h);
        }
        case data_type_t::s8:
            return _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_maskz_loadu_epi8(
                    k, static_cast<const int8_t *>(call.dst) + at)));
        case data_type_t::u8:
            return _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(
                    k, static_cast<const uint8_t *>(call.dst) + at)));
        case data_type_t::s32:
            return _mm512_cvtepi32_ps(_mm512_maskz_loadu_epi32(
                    k, static_cast<const int32_t *>(call.dst) + at));
        default: break;
    }
    alignas(64) float v[16] = {};
    for (int l = 0; l < 16; ++l)
        if ((k >> l) & 1)
            v[l] = zendnnl::memory::read_and_cast<float>(
                    call.dst, call.dst_dt, at + l);
    return _mm512_load_ps(v);
}

/// dst_values of the lanes below lanes (the destination's columns below the
/// weight's N), the others zero; no column from N on is read.
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl")
static inline __m512 dst_values_below(
        const native_int8_call_t &call, int row, int p, int lanes) {
    const __mmask16 k = lanes >= 16
            ? static_cast<__mmask16>(0xFFFF)
            : static_cast<__mmask16>((1u << std::max(0, lanes)) - 1u);
    return dst_lanes(call, static_cast<size_t>(row) * call.ldc + p, k);
}

/// Whether beta * C of a call whose integer sum the tile scales follows the
/// scales: into an f32, bf16 or f16 destination, as matmul_direct defines
/// it; into an integer one it joins the sum before them.
inline bool native_int8_beta_after_scales(data_type_t dst_dt) {
    return dst_dt == data_type_t::f32 || dst_dt == data_type_t::bf16
            || dst_dt == data_type_t::f16;
}

/// beta * C at row and columns p .. p + 16 of call's destination, the
/// product rounded on its own.  Tail: only the first lanes columns lie below
/// the weight's N, and no column past it is read.
template <bool Tail>
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl")
static inline __m512 native_int8_beta_term(const native_int8_call_t &call,
        int row, int p, int lanes, __m512 beta) {
    if constexpr (Tail)
        return avx512_mul_unfused(dst_values_below(call, row, p, lanes), beta);
    else
        return avx512_mul_unfused(dst_values(call, row, p), beta);
}

/// compute_int8_tile of a call with per-tensor or per-channel weight scales;
/// Tail: compute_int8_tail_tile's (out is nullptr).  summed: C already holds
/// the int32 sums of all of K at its float positions, as the per-channel
/// kernel sums them, and only the finish runs.
template <bool Tail>
void compute_channel_tile(const native_plan_t &plan,
        const native_int8_call_t &call, const native_weight_view_t &w, int m0,
        int mr, int n0, int nr, float *C, int ldc, const native_int8_out_t *out,
        bool summed = false);

/// compute_int8_tile of a call with per-group weight scales: C (or out) holds
/// the sum of the scaled groups; alpha, beta and the bias are the
/// epilogue's.  Tail: compute_int8_tail_tile's (out is nullptr).
template <bool Tail>
void compute_group_tile(const native_plan_t &plan,
        const native_int8_call_t &call, const native_weight_view_t &w, int m0,
        int mr, int n0, int nr, float *C, int ldc,
        const native_int8_out_t *out);

/// compute_group_tile of an s4 weight, on the INT4 group kernels in the
/// call's order (native_int8_call_t::order); an s8 source with its group
/// shifts, and weight scales per group.
template <bool Tail>
void compute_int4_group_tile(const native_plan_t &plan,
        const native_int8_call_t &call, const native_weight_view_t &w, int m0,
        int mr, int n0, int nr, float *C, int ldc,
        const native_int8_out_t *out);

/// Groups per batch of a per-group tile over a plain pack, cw columns wide
/// and rows tall, max_mr rows per call, groups of group_quads K quads of
/// quad_bytes bytes per column (4 for an s8 weight, 2 for s4): all groups
/// while their slab is within twice the pass budget of the plan, else equal
/// batches within the budget.
int native_int8_group_span(const native_plan_t &plan, int groups,
        int group_quads, int cw, int rows, int max_mr, int quad_bytes = 4);

/// The per-group weight scales of call at the weight columns p0 .. p0 +
/// cols of w (cols <= cw, the columns below the weight's N), for groups g0
/// .. g0 + n, as f32, exactly, cw apart, into buf; the other cw - cols
/// values of a group are zero and no scale past N is read.
void gather_group_wei_scales_below(const native_int8_call_t &call,
        const native_weight_view_t &w, int g0, int n, int p0, int cw, int cols,
        float *buf);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_BRGEMM_LOOPER_NATIVE_INT8_TILE_INTERNAL_HPP
