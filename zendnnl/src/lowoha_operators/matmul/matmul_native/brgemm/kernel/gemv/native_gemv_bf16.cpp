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

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

namespace {

/// Bytes of one K pair of a block: 32 columns of two bf16.
constexpr size_t kPairBytes = 128;

/// The source's K pair kp in every lane, loaded by the broadcast itself.
ZENDNNL_INLINE_TARGET("avx512f,avx512bw,avx512bf16")
inline __m512bh pair_at(const uint16_t *a, int kp) {
    uint32_t ap;
    std::memcpy(&ap, a + 2 * static_cast<size_t>(kp), sizeof(ap));
    return (__m512bh)_mm512_set1_epi32(static_cast<int>(ap));
}

/// The K pair at off of every block into its two accumulators.
template <int NB, bool Runs>
ZENDNNL_INLINE_TARGET("avx512f,avx512bw,avx512bf16,fma")
inline void pair_step(__m512 (&acc)[NB][2], __m512bh av,
        const uint8_t *const (&lo)[NB], const uint8_t *const (&hi)[NB],
        size_t off) {
    for (int i = 0; i < NB; ++i) {
        const uint8_t *second = Runs ? hi[i] : lo[i] + 64;
        acc[i][0] = _mm512_dpbf16_ps(
                acc[i][0], av, (__m512bh)_mm512_loadu_si512(lo[i] + off));
        acc[i][1] = _mm512_dpbf16_ps(
                acc[i][1], av, (__m512bh)_mm512_loadu_si512(second + off));
    }
}

template <int NB, bool Runs>
ZENDNNL_TARGET_NOINLINE("avx512f,avx512bw,avx512bf16,fma")
void bf16_gemv(const uint16_t *__restrict a, int K,
        const native_gemv_cols_t &cols, float *__restrict c) {
    const uint8_t *lo[NB];
    const uint8_t *hi[NB];
    for (int i = 0; i < NB; ++i) {
        lo[i] = cols.lo[i];
        hi[i] = Runs ? cols.hi[i] : nullptr;
    }
    __m512 acc[NB][2];
    for (int i = 0; i < NB; ++i)
        acc[i][0] = acc[i][1] = _mm512_setzero_ps();
    const int pairs = K / 2;
    size_t off = 0;
    int kp = 0;
    for (; kp + 1 < pairs; kp += 2, off += 2 * kPairBytes) {
        pair_step<NB, Runs>(acc, pair_at(a, kp), lo, hi, off);
        pair_step<NB, Runs>(acc, pair_at(a, kp + 1), lo, hi, off + kPairBytes);
    }
    for (; kp < pairs; ++kp, off += kPairBytes)
        pair_step<NB, Runs>(acc, pair_at(a, kp), lo, hi, off);
    if ((K & 1) != 0) {
        // The last element of an odd K pairs with zero, as the pack's.
        const uint32_t ap = a[K - 1];
        pair_step<NB, Runs>(acc,
                (__m512bh)_mm512_set1_epi32(static_cast<int>(ap)), lo, hi, off);
    }
    for (int i = 0; i < NB; ++i) {
        _mm512_storeu_ps(c + 32 * i, acc[i][0]);
        _mm512_storeu_ps(c + 32 * i + 16, acc[i][1]);
    }
}

template <bool Runs, int... I>
constexpr std::array<native_bf16_gemv_fn_t, sizeof...(I)> make_table(
        std::integer_sequence<int, I...>) {
    return {{&bf16_gemv<I + 1, Runs>...}};
}

constexpr auto kBlocks = make_table<false>(
        std::make_integer_sequence<int, kNativeGemvMaxBlocks> {});
constexpr auto kRuns = make_table<true>(
        std::make_integer_sequence<int, kNativeGemvMaxBlocks> {});

} // namespace

native_bf16_gemv_fn_t select_native_bf16_gemv(int nb, bool runs) {
    if (nb < 1 || nb > kNativeGemvMaxBlocks) return nullptr;
    return (runs ? kRuns : kBlocks)[nb - 1];
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
