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

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <utility>

#include <immintrin.h>

#include "common/zendnnl_compat.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/f16/native_f16_ph_ukernel.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

#if ZENDNNL_NATIVE_F16_PH_KERNELS
namespace {

/// Bytes of one K row of a block: 32 f16.
constexpr size_t kRowBytes = 64;

/// The f16 at a as the scalar of a broadcast: a copy of its bits.
ZENDNNL_INLINE_TARGET("avx512f,avx512bw,avx512vl,avx512fp16,fma")
inline _Float16 f16_at(const uint16_t *a) {
    _Float16 h;
    std::memcpy(&h, a, sizeof(h));
    return h;
}

template <int NB>
ZENDNNL_TARGET_NOINLINE("avx512f,avx512bw,avx512vl,avx512fp16,fma")
void f16_ph_gemv(const uint16_t *__restrict a, int K,
        const native_gemv_cols_t &cols, float *__restrict c) {
    const uint8_t *b[NB];
    for (int i = 0; i < NB; ++i)
        b[i] = cols.lo[i];
    __m512 s[NB][2];
    for (int i = 0; i < NB; ++i)
        s[i][0] = s[i][1] = _mm512_setzero_ps();
    // A block ends at its start plus what is left: the start plus a block
    // may pass the int bound.
    for (int k0 = 0, kb = 0; k0 < K; k0 += kb) {
        kb = std::min(kNativeF16AccumBlockK, K - k0);
        __m512h acc[NB];
        for (int i = 0; i < NB; ++i)
            acc[i] = _mm512_setzero_ph();
        size_t off = static_cast<size_t>(k0) * kRowBytes;
        for (int k = k0; k < k0 + kb; ++k, off += kRowBytes) {
            const __m512h av = _mm512_set1_ph(f16_at(a + k));
            for (int i = 0; i < NB; ++i)
                acc[i] = _mm512_fmadd_ph(
                        av, _mm512_loadu_ph(b[i] + off), acc[i]);
        }
        for (int i = 0; i < NB; ++i) {
            const __m512i bits = _mm512_castph_si512(acc[i]);
            s[i][0] = _mm512_add_ps(
                    s[i][0], _mm512_cvtph_ps(_mm512_castsi512_si256(bits)));
            s[i][1] = _mm512_add_ps(s[i][1],
                    _mm512_cvtph_ps(_mm512_extracti64x4_epi64(bits, 1)));
        }
    }
    for (int i = 0; i < NB; ++i) {
        _mm512_storeu_ps(c + 32 * i, s[i][0]);
        _mm512_storeu_ps(c + 32 * i + 16, s[i][1]);
    }
}

template <int... I>
constexpr std::array<native_f16_ph_gemv_fn_t, sizeof...(I)> make_table(
        std::integer_sequence<int, I...>) {
    return {{&f16_ph_gemv<I + 1>...}};
}

constexpr auto kTable
        = make_table(std::make_integer_sequence<int, kNativeGemvMaxBlocks> {});

} // namespace
#endif

native_f16_ph_gemv_fn_t select_native_f16_ph_gemv(int nb) {
    if (nb < 1 || nb > kNativeGemvMaxBlocks) return nullptr;
#if ZENDNNL_NATIVE_F16_PH_KERNELS
    return kTable[nb - 1];
#else
    return nullptr;
#endif
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
