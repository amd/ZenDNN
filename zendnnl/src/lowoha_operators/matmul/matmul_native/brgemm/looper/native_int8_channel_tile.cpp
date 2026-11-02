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

namespace {

/// C holds f32(S) for rows m0 .. m0 + mr and pack columns p0 .. p0 + nr (a
/// multiple of 16), wei_scale the f32 weight scales of those columns.
/// Applies alpha, beta, the zero-point term and the scales one rounding at a
/// time, in the reference algorithm's order: ((alpha * S + beta * C + Z) *
/// src_scale) * wei_scale[j].  Pairs: w is an interleaved pack, and beta
/// reads C at the columns its pack columns hold.  Tail: only the first cols
/// pack columns lie below the weight's N; beta reads no column past it.
template <bool Pairs, bool Tail>
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,avx512dq")
void finish_in_reference_order(const native_int8_call_t &call,
        const native_weight_view_t &w, int m0, int mr, int p0, int nr, int cols,
        const int32_t *col_sum, const float *wei_scale, float *C, int ldc) {
    const bool scale_alpha = call.alpha != 1.0f;
    const bool add_beta = call.beta != 0.0f;
    assert(Pairs == w.interleaved);
    const __m512 alpha = _mm512_set1_ps(call.alpha);
    const __m512 beta = _mm512_set1_ps(call.beta);
    for (int r = 0; r < mr; ++r) {
        const int row = m0 + r;
        const int64_t zp = call.src_zp != nullptr
                ? call.src_zp[call.src_zp_stride * row]
                : 0;
        const __m512 a
                = _mm512_set1_ps(call.src_scale[call.src_scale_stride * row]);
        const __m512i neg_zp = _mm512_set1_epi64(-zp);
        float *c = C + static_cast<size_t>(r) * ldc;
        for (int j = 0; j < nr; j += 16) {
            __m512 v = _mm512_loadu_ps(c + j);
            // No product may fuse with the add that follows it.
            if (scale_alpha) v = avx512_mul_unfused(v, alpha);
            if (add_beta) {
                if constexpr (Tail && Pairs)
                    v = _mm512_add_ps(v,
                            avx512_mul_unfused(dst_pair_values_below(call, w,
                                                       row, p0 + j, cols - j),
                                    beta));
                else if constexpr (Tail)
                    v = _mm512_add_ps(v,
                            avx512_mul_unfused(dst_values_below(call, row,
                                                       p0 + j, cols - j),
                                    beta));
                else if constexpr (Pairs)
                    v = _mm512_add_ps(v,
                            avx512_mul_unfused(
                                    dst_pair_values(call, w, row, p0 + j),
                                    beta));
                else
                    v = _mm512_add_ps(v,
                            avx512_mul_unfused(
                                    dst_values(call, row, p0 + j), beta));
            }
            if (zp != 0) {
                // In int64, each product rounded once: exact for any zero
                // point, and equal to the wrapping int32 product wherever
                // that one does not wrap.
                const __m512i cs = _mm512_loadu_si512(col_sum + j);
                const __m256 lo = _mm512_cvtepi64_ps(_mm512_mullo_epi64(
                        _mm512_cvtepi32_epi64(_mm512_castsi512_si256(cs)),
                        neg_zp));
                const __m256 hi = _mm512_cvtepi64_ps(_mm512_mullo_epi64(
                        _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(cs, 1)),
                        neg_zp));
                v = _mm512_add_ps(v,
                        _mm512_insertf32x8(_mm512_castps256_ps512(lo), hi, 1));
            }
            v = _mm512_mul_ps(
                    _mm512_mul_ps(v, a), _mm512_loadu_ps(wei_scale + j));
            _mm512_storeu_ps(c + j, v);
        }
    }
}

} // namespace

template <bool Tail>
void compute_channel_tile(const native_plan_t &plan,
        const native_int8_call_t &call, const native_weight_view_t &w, int m0,
        int mr, int n0, int nr, float *C, int ldc,
        const native_int8_out_t *out) {
    const int nb = n0 / w.pack_nr;
    const int j0 = n0 - nb * w.pack_nr;
    const int8_t *block = reinterpret_cast<const int8_t *>(w.block(nb));
    const int32_t *col_sums = w.col_sums(nb);
    const uint32_t zp_shift = call.src_signed ? 128u : 0u;
    // The kernel then yields f32(S) (unit scales, only the +128 term), and
    // the reference-order finish does the rest.
    const bool reference_order = call.order == native_int8_order_t::reference;
    assert(out == nullptr || !reference_order);
    const size_t out_esz = out != nullptr && out->bf16 ? 2 : 4;

    alignas(64) float scale_buf[kMaxTileNr];
    alignas(64) float bias_buf[kMaxTileNr];
    alignas(64) float ones[kMaxTileNr];
    alignas(64) int32_t zp[kMaxTileMr];
    alignas(64) float src_scale[kMaxTileMr];

    native_int8_ukernel_args_t args;
    args.b_stride = w.group_stride();
    args.k_quads = w.k_groups;
    args.lda = call.lda;
    args.zp = zp;
    args.src_scale = src_scale;
    if (reference_order) {
        std::fill(ones, ones + kMaxTileNr, 1.0f);
        args.wei_scale = ones;
    }
    args.wide_zp = call.wide_zp && !reference_order;
    args.ldc = ldc;
    if (out != nullptr) {
        args.dst_ldc = out->ldc;
        args.dst_bf16 = out->bf16;
    }

    for (int c = 0; c < nr;) {
        const int cw = native_int8_col_width(plan, nr - c);
        assert(cw <= kMaxTileNr);
        const int nv = cw / 16;
        // Tail: the columns of this call below the weight's N.
        const int cols = Tail ? std::min(cw, w.N - (n0 + c)) : cw;
        const bool part = Tail && cols < cw;
        const float *col_scale = part
                ? column_values_below(call.wei_scale, call.wei_scale_dt,
                          call.wei_scale_stride, w, n0 + c, cw, cols, scale_buf)
                : column_values(call.wei_scale, call.wei_scale_dt,
                          call.wei_scale_stride, w, n0 + c, cw, scale_buf);
        if (!reference_order) args.wei_scale = col_scale;
        if (call.bias == nullptr)
            args.bias = nullptr;
        else if (part)
            args.bias = column_values_below(
                    call.bias, call.bias_dt, 1, w, n0 + c, cw, cols, bias_buf);
        else
            args.bias = column_values(
                    call.bias, call.bias_dt, 1, w, n0 + c, cw, bias_buf);
        args.col_sum = col_sums + j0 + c;
        const int max_mr = native_int8_call_rows(plan, nv);
        assert(max_mr >= 1 && max_mr <= kMaxTileMr);
        const int pass
                = native_int8_pass_quads(plan, w.k_groups, cw, mr, max_mr);
        for (int kq = 0; kq < w.k_groups; kq += pass) {
            args.k_quads = std::min(pass, w.k_groups - kq);
            args.B = block + static_cast<size_t>(j0 + c) * 4
                    + static_cast<size_t>(kq) * args.b_stride;
            args.first_pass = kq == 0;
            args.last_pass = kq + args.k_quads == w.k_groups;
            for (int i = 0; i < mr;) {
                const int rows = std::min(max_mr, mr - i);
                for (int r = 0; r < rows; ++r) {
                    const int row = m0 + i + r;
                    const int32_t z = call.src_zp != nullptr && !reference_order
                            ? call.src_zp[call.src_zp_stride * row]
                            : 0;
                    zp[r] = args.wide_zp
                            ? z
                            : static_cast<int32_t>(
                                      static_cast<uint32_t>(z) + zp_shift);
                    src_scale[r] = reference_order
                            ? 1.0f
                            : call.src_scale[call.src_scale_stride * row];
                }
                args.A = call.A + static_cast<size_t>(m0 + i) * call.lda
                        + static_cast<size_t>(kq) * 4;
                args.C = C + static_cast<size_t>(i) * ldc + c;
                args.partial = pass < w.k_groups
                        ? reinterpret_cast<int32_t *>(args.C)
                        : nullptr;
                if (out != nullptr)
                    args.dst = static_cast<char *>(out->dst)
                            + (static_cast<size_t>(m0 + i) * out->ldc + n0 + c)
                                    * out_esz;
                select_native_int8_ukernel(rows, nv, call.src_signed)(args);
                if (reference_order && args.last_pass) {
                    if (w.interleaved)
                        finish_in_reference_order<true, Tail>(call, w, m0 + i,
                                rows, n0 + c, cw, cols, args.col_sum, col_scale,
                                args.C, ldc);
                    else
                        finish_in_reference_order<false, Tail>(call, w, m0 + i,
                                rows, n0 + c, cw, cols, args.col_sum, col_scale,
                                args.C, ldc);
                }
                i += rows;
            }
        }
        c += cw;
    }
}

template void compute_channel_tile<false>(const native_plan_t &,
        const native_int8_call_t &, const native_weight_view_t &, int, int, int,
        int, float *, int, const native_int8_out_t *);
template void compute_channel_tile<true>(const native_plan_t &,
        const native_int8_call_t &, const native_weight_view_t &, int, int, int,
        int, float *, int, const native_int8_out_t *);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
