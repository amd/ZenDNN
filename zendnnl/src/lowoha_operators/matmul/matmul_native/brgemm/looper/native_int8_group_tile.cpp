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

#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_int8_tile_internal.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>

#include <immintrin.h>

#include "common/zendnnl_compat.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/int8/native_int8_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/common/avx512_math.hpp"
#include "memory/memory_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::memory::read_and_cast;

namespace {

/// shift[j] = 128 times the sum over k_quads quads of pack column j, 16 * NV
/// columns from b: what reading an s8 source as s8 + 128 adds to column j.
/// Independent sums keep several VPDPBUSD chains in flight.
template <int NV>
ZENDNNL_TARGET("avx512f,avx512bw,avx512vnni")
inline void group_shifts(
        const int8_t *b, int b_stride, int k_quads, int32_t *shift) {
    constexpr int kSets = NV == 4 ? 2 : 4;
    const __m512i ones = _mm512_set1_epi8(1);
    __m512i acc[kSets][NV];
    for (int s = 0; s < kSets; ++s)
        for (int v = 0; v < NV; ++v)
            acc[s][v] = _mm512_setzero_si512();
    int q = 0;
    for (; q + kSets <= k_quads; q += kSets)
        for (int s = 0; s < kSets; ++s)
            for (int v = 0; v < NV; ++v)
                acc[s][v] = _mm512_dpbusd_epi32(acc[s][v], ones,
                        _mm512_loadu_si512(b
                                + static_cast<size_t>(q + s) * b_stride
                                + v * 64));
    for (; q < k_quads; ++q)
        for (int v = 0; v < NV; ++v)
            acc[0][v] = _mm512_dpbusd_epi32(acc[0][v], ones,
                    _mm512_loadu_si512(
                            b + static_cast<size_t>(q) * b_stride + v * 64));
    for (int v = 0; v < NV; ++v) {
        for (int s = 1; s < kSets; ++s)
            acc[0][v] = _mm512_add_epi32(acc[0][v], acc[s][v]);
        _mm512_storeu_si512(shift + v * 16, _mm512_slli_epi32(acc[0][v], 7));
    }
}

/// The f32 weight scales of group g at pack columns p0 .. p0 + cw (cw a
/// multiple of 16): in place for an f32 buffer and a plain pack, else
/// written to buf.
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl")
inline const float *group_wei_scales(const native_int8_call_t &call,
        const native_weight_view_t &w, int g, int p0, int cw, float *buf) {
    const size_t row = static_cast<size_t>(g) * w.N;
    if (w.interleaved) {
        for (int j = 0; j < cw; ++j)
            buf[j] = read_and_cast<float>(call.wei_scale, call.wei_scale_dt,
                    row + w.logical_col(p0 + j));
        return buf;
    }
    if (call.wei_scale_dt == data_type_t::f32)
        return static_cast<const float *>(call.wei_scale) + row + p0;
    const uint16_t *h
            = static_cast<const uint16_t *>(call.wei_scale) + row + p0;
    for (int j = 0; j < cw; j += 16) {
        const __m256i v
                = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(h + j));
        _mm512_storeu_ps(buf + j,
                call.wei_scale_dt == data_type_t::bf16 ? avx512_bf16_to_f32(v)
                                                       : _mm512_cvtph_ps(v));
    }
    return buf;
}

/// c[r][j] = (first ? 0 : c[r][j]) + (f32(acc[r][j] - shift[j])
/// * sa[r * sa_stride]) * sb[j] over rows x cols (cols a multiple of 16), one
/// rounding per operation.  acc - shift is the exact group sum.
ZENDNNL_TARGET("avx512f")
inline void add_group(const int32_t *acc, int lda, const int32_t *shift,
        const float *sa, int sa_stride, const float *sb, int rows, int cols,
        bool first, float *c, int ldc) {
    for (int r = 0; r < rows; ++r) {
        const __m512 a = _mm512_set1_ps(sa[static_cast<size_t>(r) * sa_stride]);
        const int32_t *s = acc + static_cast<size_t>(r) * lda;
        float *cr = c + static_cast<size_t>(r) * ldc;
        for (int j = 0; j < cols; j += 16) {
            const __m512 d = _mm512_cvtepi32_ps(_mm512_sub_epi32(
                    _mm512_loadu_si512(s + j), _mm512_loadu_si512(shift + j)));
            const __m512 v = avx512_mul_unfused(
                    _mm512_mul_ps(d, a), _mm512_loadu_ps(sb + j));
            _mm512_storeu_ps(cr + j,
                    _mm512_add_ps(first ? _mm512_setzero_ps()
                                        : _mm512_loadu_ps(cr + j),
                            v));
        }
    }
}

/// C[mr][nr] = the scaled group sums of the tile added in group order (see
/// native_int8_call_t::groups).  Each kernel call is
/// the first, not the last, pass over one group: it leaves the group's int32
/// sums of the source read as s8 + 128.
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,avx512vnni")
void sum_groups(const native_plan_t &plan, const native_int8_call_t &call,
        const native_weight_view_t &w, int m0, int mr, int n0, int nr, float *C,
        int ldc) {
    const int nb = n0 / w.pack_nr;
    const int j0 = n0 - nb * w.pack_nr;
    const int8_t *block = reinterpret_cast<const int8_t *>(w.block(nb));
    alignas(64) int32_t shift[kMaxTileNr];
    alignas(64) float wei_scale[kMaxTileNr];
    alignas(64) int32_t sums[kMaxTileMr * kMaxTileNr];

    native_int8_ukernel_args_t args;
    args.b_stride = w.group_stride();
    args.k_quads = call.group_quads;
    args.lda = call.lda;
    args.partial = sums;
    args.first_pass = true;
    args.last_pass = false;
    const size_t group_a = static_cast<size_t>(call.group_quads) * 4;
    const size_t group_b
            = static_cast<size_t>(call.group_quads) * args.b_stride;
    for (int c = 0; c < nr;) {
        const int cw = native_int8_col_width(plan, nr - c);
        const int nv = cw / 16;
        const int max_mr = native_int8_call_rows(plan, nv);
        assert(max_mr >= 1 && max_mr <= kMaxTileMr);
        args.ldc = cw;
        for (int g = 0; g < call.groups; ++g) {
            args.B = block + static_cast<size_t>(j0 + c) * 4 + g * group_b;
            if (nv == 1)
                group_shifts<1>(args.B, args.b_stride, args.k_quads, shift);
            else if (nv == 2)
                group_shifts<2>(args.B, args.b_stride, args.k_quads, shift);
            else
                group_shifts<4>(args.B, args.b_stride, args.k_quads, shift);
            const float *sb
                    = group_wei_scales(call, w, g, n0 + c, cw, wei_scale);
            for (int i = 0; i < mr;) {
                const int rows = std::min(max_mr, mr - i);
                const size_t row = static_cast<size_t>(m0 + i);
                args.A = call.A + row * call.lda + g * group_a;
                select_native_int8_ukernel(rows, nv, true)(args);
                add_group(sums, cw, shift,
                        call.src_scale + call.src_scale_stride * row
                                + static_cast<size_t>(
                                          call.src_scale_group_stride)
                                        * g,
                        call.src_scale_stride, sb, rows, cw, g == 0,
                        C + static_cast<size_t>(i) * ldc + c, ldc);
                i += rows;
            }
        }
        c += cw;
    }
}

} // namespace

void compute_group_tile(const native_plan_t &plan,
        const native_int8_call_t &call, const native_weight_view_t &w, int m0,
        int mr, int n0, int nr, float *C, int ldc) {
    sum_groups(plan, call, w, m0, mr, n0, nr, C, ldc);
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
