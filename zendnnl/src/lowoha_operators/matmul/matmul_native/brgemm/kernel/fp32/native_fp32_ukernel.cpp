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

#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/fp32/native_fp32_ukernel.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>

#include <immintrin.h>

#include "common/zendnnl_compat.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

namespace {

constexpr int kMaxMr = 12;

/// Row k of B into the accumulators: one fused multiply-add per element.
template <int MR, int NV>
ZENDNNL_INLINE_TARGET("avx512f,fma")
inline void fma_row(__m512 (&acc)[MR][NV], const float *__restrict a, int lda,
        const float *__restrict b) {
    __m512 bv[NV];
    for (int v = 0; v < NV; ++v)
        bv[v] = _mm512_loadu_ps(b + v * 16);
    for (int m = 0; m < MR; ++m) {
        const __m512 av = _mm512_set1_ps(a[static_cast<size_t>(m) * lda]);
        for (int v = 0; v < NV; ++v)
            acc[m][v] = _mm512_fmadd_ps(av, bv[v], acc[m][v]);
    }
}

template <int MR, int NV>
ZENDNNL_TARGET_NOINLINE("avx512f,fma")
void native_fp32_ukernel(const float *__restrict A, int lda,
        const float *__restrict B, int b_stride, float *__restrict C, int ldc,
        int K, int BK, float beta) {
    __m512 acc[MR][NV];
    if (beta != 0.0f) {
        const __m512 bv = _mm512_set1_ps(beta);
        for (int m = 0; m < MR; ++m)
            for (int v = 0; v < NV; ++v)
                acc[m][v] = _mm512_mul_ps(bv,
                        _mm512_loadu_ps(
                                C + static_cast<size_t>(m) * ldc + v * 16));
    } else {
        for (int m = 0; m < MR; ++m)
            for (int v = 0; v < NV; ++v)
                acc[m][v] = _mm512_setzero_ps();
    }

    // A block ends at its start plus what is left: the start plus a block
    // may pass the int bound.
    for (int pc = 0, kb = 0; pc < K; pc += kb) {
        kb = std::min(BK, K - pc);
        const float *a_off = A + pc;
        const float *b_off = B + static_cast<size_t>(pc) * b_stride;
        int kk = 0;
        for (; kk + 3 < kb; kk += 4)
            for (int u = 0; u < 4; ++u)
                fma_row<MR, NV>(acc, a_off + kk + u, lda,
                        b_off + static_cast<size_t>(kk + u) * b_stride);
        for (; kk < kb; ++kk)
            fma_row<MR, NV>(acc, a_off + kk, lda,
                    b_off + static_cast<size_t>(kk) * b_stride);
    }

    for (int m = 0; m < MR; ++m)
        for (int v = 0; v < NV; ++v)
            _mm512_storeu_ps(
                    C + static_cast<size_t>(m) * ldc + v * 16, acc[m][v]);
}

template <int NV, int... I>
constexpr std::array<native_fp32_ukernel_fn_t, sizeof...(I)> make_table(
        std::integer_sequence<int, I...>) {
    return {{&native_fp32_ukernel<I + 1, NV>...}};
}

constexpr auto kNv1 = make_table<1>(std::make_integer_sequence<int, kMaxMr> {});
constexpr auto kNv2 = make_table<2>(std::make_integer_sequence<int, kMaxMr> {});

} // namespace

int native_fp32_ukernel_max_mr(int nv) {
    return nv == 1 || nv == 2 ? kMaxMr : 0;
}

native_fp32_ukernel_fn_t select_native_fp32_ukernel(int mr, int nv) {
    if (mr < 1 || mr > native_fp32_ukernel_max_mr(nv)) return nullptr;
    return (nv == 1 ? kNv1 : kNv2)[mr - 1];
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
