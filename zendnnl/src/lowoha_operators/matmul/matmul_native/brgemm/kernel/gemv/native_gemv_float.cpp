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
#include <utility>

#include <immintrin.h>

#include "common/zendnnl_compat.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

namespace {

/// The weight's 16 values at p as f32: f32 as they are, f16 widened exactly.
template <bool F16>
ZENDNNL_INLINE_TARGET("avx512f,fma")
inline __m512 weight_values(const uint8_t *p) {
    if constexpr (F16)
        return _mm512_cvtph_ps(
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p)));
    else
        return _mm512_loadu_ps(reinterpret_cast<const float *>(p));
}

/// Row k (at off) of every block into its two accumulators: one fused
/// multiply-add per element.
template <int NB, bool F16>
ZENDNNL_INLINE_TARGET("avx512f,fma")
inline void row_step(__m512 (&acc)[NB][2], __m512 av,
        const uint8_t *const (&b)[NB], size_t off) {
    constexpr size_t half = F16 ? 32 : 64;
    for (int i = 0; i < NB; ++i) {
        acc[i][0] = _mm512_fmadd_ps(
                av, weight_values<F16>(b[i] + off), acc[i][0]);
        acc[i][1] = _mm512_fmadd_ps(
                av, weight_values<F16>(b[i] + off + half), acc[i][1]);
    }
}

/// The FP32 kernel, or with F16 the F16 kernel over the widened source.
template <int NB, bool F16>
ZENDNNL_TARGET_NOINLINE("avx512f,fma")
void float_gemv(const float *__restrict a, int K,
        const native_gemv_cols_t &cols, float *__restrict c) {
    constexpr size_t row = F16 ? 64 : 128;
    const uint8_t *b[NB];
    for (int i = 0; i < NB; ++i)
        b[i] = cols.lo[i];
    __m512 acc[NB][2];
    for (int i = 0; i < NB; ++i)
        acc[i][0] = acc[i][1] = _mm512_setzero_ps();
    size_t off = 0;
    int k = 0;
    for (; k + 1 < K; k += 2, off += 2 * row) {
        row_step<NB, F16>(acc, _mm512_set1_ps(a[k]), b, off);
        row_step<NB, F16>(acc, _mm512_set1_ps(a[k + 1]), b, off + row);
    }
    if (k < K) row_step<NB, F16>(acc, _mm512_set1_ps(a[k]), b, off);
    for (int i = 0; i < NB; ++i) {
        _mm512_storeu_ps(c + 32 * i, acc[i][0]);
        _mm512_storeu_ps(c + 32 * i + 16, acc[i][1]);
    }
}

template <bool F16, int... I>
constexpr std::array<native_f32_gemv_fn_t, sizeof...(I)> make_table(
        std::integer_sequence<int, I...>) {
    return {{&float_gemv<I + 1, F16>...}};
}

constexpr auto kF32 = make_table<false>(
        std::make_integer_sequence<int, kNativeGemvMaxBlocks> {});
constexpr auto kF16 = make_table<true>(
        std::make_integer_sequence<int, kNativeGemvMaxBlocks> {});

} // namespace

native_f32_gemv_fn_t select_native_f32_gemv(int nb) {
    if (nb < 1 || nb > kNativeGemvMaxBlocks) return nullptr;
    return kF32[nb - 1];
}

native_f16_gemv_fn_t select_native_f16_gemv(int nb) {
    if (nb < 1 || nb > kNativeGemvMaxBlocks) return nullptr;
    return kF16[nb - 1];
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
