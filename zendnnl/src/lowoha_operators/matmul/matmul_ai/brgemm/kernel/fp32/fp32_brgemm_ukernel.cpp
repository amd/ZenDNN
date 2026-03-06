/*******************************************************************************
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lowoha_operators/matmul/matmul_ai/brgemm/kernel/fp32/fp32_brgemm_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_ai/common/avx512_math.hpp"
#include <cstring>
#include <algorithm>
#include <cmath>
#include <immintrin.h>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ai {

template<int MR, int NV>
__attribute__((target("avx512f,fma"), noinline))
void brgemm_ukernel(
    const float * __restrict__ A, int lda,
    const float * __restrict__ pb, int pb_stride,
    float * __restrict__ C, int ldc,
    int K, int BK, float beta,
    const float * __restrict__ bias, fused_postop_t fused_op) {

    __m512 acc[MR][NV];
    if (beta != 0.0f) {
        __m512 bv = _mm512_set1_ps(beta);
        for (int m = 0; m < MR; ++m)
            for (int v = 0; v < NV; ++v)
                acc[m][v] = _mm512_mul_ps(
                    bv, _mm512_loadu_ps(C + m * ldc + v * 16));
    } else {
        for (int m = 0; m < MR; ++m)
            for (int v = 0; v < NV; ++v)
                acc[m][v] = _mm512_setzero_ps();
    }

    // Batch-reduce: iterate ALL K, keeping accumulators live
    for (int pc = 0; pc < K; pc += BK) {
        const int kb = std::min(BK, K - pc);
        const float *a_off = A + pc;        // A offset for this K-block
        const float *b_off = pb + pc * pb_stride; // B offset for this K-block

        // K-loop: 4x unrolled
        int kk = 0;
        for (; kk + 3 < kb; kk += 4) {
            for (int u = 0; u < 4; ++u) {
                const float *bp = b_off + (kk + u) * pb_stride;
                __m512 bv[NV];
                for (int v = 0; v < NV; ++v)
                    bv[v] = _mm512_loadu_ps(bp + v * 16);
                for (int m = 0; m < MR; ++m) {
                    __m512 a = _mm512_set1_ps(a_off[m * lda + kk + u]);
                    for (int v = 0; v < NV; ++v)
                        acc[m][v] = _mm512_fmadd_ps(a, bv[v], acc[m][v]);
                }
            }
        }
        for (; kk < kb; ++kk) {
            __m512 bv[NV];
            for (int v = 0; v < NV; ++v)
                bv[v] = _mm512_loadu_ps(b_off + kk * pb_stride + v * 16);
            for (int m = 0; m < MR; ++m) {
                __m512 a = _mm512_set1_ps(a_off[m * lda + kk]);
                for (int v = 0; v < NV; ++v)
                    acc[m][v] = _mm512_fmadd_ps(a, bv[v], acc[m][v]);
            }
        }
    }

    // Epilogue (ONCE after all K): bias → activation → store
    if (bias) {
        __m512 bias_v[NV];
        for (int v = 0; v < NV; ++v)
            bias_v[v] = _mm512_loadu_ps(bias + v * 16);
        for (int m = 0; m < MR; ++m)
            for (int v = 0; v < NV; ++v)
                acc[m][v] = _mm512_add_ps(acc[m][v], bias_v[v]);
    }

    if (fused_op != fused_postop_t::none) {
        for (int m = 0; m < MR; ++m)
            for (int v = 0; v < NV; ++v)
                acc[m][v] = apply_fused_postop(acc[m][v], fused_op);
    }

    for (int m = 0; m < MR; ++m)
        for (int v = 0; v < NV; ++v)
            _mm512_storeu_ps(C + m * ldc + v * 16, acc[m][v]);
}

// Explicit instantiation for MR=6, NR=16
template void brgemm_ukernel<6,1>(const float*, int, const float*, int,
    float*, int, int, int, float, const float*, fused_postop_t);

// ============================================================================
// BRGEMM tail kernel (dynamic MR/NR for edge tiles)
// ============================================================================
__attribute__((target("avx512f,avx512bw,fma")))
void brgemm_tail_kernel(
    const float * __restrict__ A, int lda,
    const float * __restrict__ pb, int pb_stride,
    float * __restrict__ C, int ldc,
    int K, int BK, int mr_count, int nr_count, float beta,
    const float * __restrict__ bias, fused_postop_t fused_op) {

    const int full_vecs = nr_count / 16;
    const int rem = nr_count & 15;
    const __mmask16 rem_mask = rem ? static_cast<__mmask16>((1u << rem) - 1)
                                   : static_cast<__mmask16>(0);
    const int nv = (nr_count + 15) / 16;

    __m512 acc[12][4];

    if (beta != 0.0f) {
        __m512 bv = _mm512_set1_ps(beta);
        for (int m = 0; m < mr_count; ++m) {
            for (int v = 0; v < full_vecs; ++v)
                acc[m][v] = _mm512_mul_ps(
                    bv, _mm512_loadu_ps(C + m * ldc + v * 16));
            if (rem)
                acc[m][full_vecs] = _mm512_mul_ps(
                    bv, _mm512_maskz_loadu_ps(rem_mask, C + m * ldc + full_vecs * 16));
        }
    } else {
        for (int m = 0; m < mr_count; ++m)
            for (int v = 0; v < nv; ++v)
                acc[m][v] = _mm512_setzero_ps();
    }

    // Batch-reduce over all K
    for (int pc = 0; pc < K; pc += BK) {
        const int kb = std::min(BK, K - pc);
        const float *a_off = A + pc;
        const float *b_off = pb + pc * pb_stride;

        for (int kk = 0; kk < kb; ++kk) {
            __m512 bv[4];
            for (int v = 0; v < full_vecs; ++v)
                bv[v] = _mm512_loadu_ps(b_off + kk * pb_stride + v * 16);
            if (rem)
                bv[full_vecs] = _mm512_maskz_loadu_ps(
                    rem_mask, b_off + kk * pb_stride + full_vecs * 16);
            for (int m = 0; m < mr_count; ++m) {
                __m512 a = _mm512_set1_ps(a_off[m * lda + kk]);
                for (int v = 0; v < full_vecs; ++v)
                    acc[m][v] = _mm512_fmadd_ps(a, bv[v], acc[m][v]);
                if (rem)
                    acc[m][full_vecs] = _mm512_fmadd_ps(a, bv[full_vecs],
                                                         acc[m][full_vecs]);
            }
        }
    }

    // Epilogue
    if (bias) {
        for (int v = 0; v < full_vecs; ++v) {
            __m512 bv = _mm512_loadu_ps(bias + v * 16);
            for (int m = 0; m < mr_count; ++m)
                acc[m][v] = _mm512_add_ps(acc[m][v], bv);
        }
        if (rem) {
            __m512 bv = _mm512_maskz_loadu_ps(rem_mask, bias + full_vecs * 16);
            for (int m = 0; m < mr_count; ++m)
                acc[m][full_vecs] = _mm512_add_ps(acc[m][full_vecs], bv);
        }
    }

    if (fused_op != fused_postop_t::none) {
        for (int m = 0; m < mr_count; ++m)
            for (int v = 0; v < nv; ++v)
                acc[m][v] = apply_fused_postop(acc[m][v], fused_op);
    }

    for (int m = 0; m < mr_count; ++m) {
        for (int v = 0; v < full_vecs; ++v)
            _mm512_storeu_ps(C + m * ldc + v * 16, acc[m][v]);
        if (rem)
            _mm512_mask_storeu_ps(C + m * ldc + full_vecs * 16,
                                  rem_mask, acc[m][full_vecs]);
    }
}

} // namespace ai
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
