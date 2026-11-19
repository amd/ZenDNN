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

#include "s8_group_sums.hpp"
#include "common/zendnnl_compat.hpp"

#include <cstddef>
#include <cstdint>

#include <immintrin.h>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace custom_kernel {

namespace {

// Σ of `n` signed source bytes (n a multiple of 8).  SAD sums unsigned
// bytes, so each byte is flipped to x + 128 and the 128s removed after.
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,avx2")
static inline int32_t sum_s8_group(const uint8_t *p, int n) {
    int64_t total = 0;
    int i = 0;
    if (n >= 32) {
        const __m256i flip = _mm256_set1_epi8(static_cast<char>(0x80));
        __m256i acc = _mm256_setzero_si256();
        for (; i + 32 <= n; i += 32) {
            const __m256i v = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(p + i));
            acc = _mm256_add_epi64(acc,
                    _mm256_sad_epu8(
                            _mm256_xor_si256(v, flip), _mm256_setzero_si256()));
        }
        const __m128i s = _mm_add_epi64(
                _mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
        total = _mm_cvtsi128_si64(_mm_add_epi64(s, _mm_unpackhi_epi64(s, s)));
    }
    const __m128i flip8 = _mm_set1_epi8(static_cast<char>(0x80));
    for (; i < n; i += 8) {
        const __m128i v
                = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(p + i));
        total += _mm_cvtsi128_si64(
                _mm_sad_epu8(_mm_xor_si128(v, flip8), _mm_setzero_si128()));
    }
    return static_cast<int32_t>(total - 128 * n);
}

ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,avx2")
static void s8_group_sums_impl(const uint8_t *A, int lda, int rows, int K,
        int group_size, int32_t *out) {
    const int G = K / group_size;
    for (int m = 0; m < rows; ++m) {
        const uint8_t *row = A + static_cast<size_t>(m) * lda;
        int32_t *dst = out + static_cast<size_t>(m) * G;
        for (int g = 0; g < G; ++g) {
            dst[g] = sum_s8_group(
                    row + static_cast<size_t>(g) * group_size, group_size);
        }
    }
}

} // namespace

void s8_group_sums(const uint8_t *A, int lda, int rows, int K, int group_size,
        int32_t *out) {
    s8_group_sums_impl(A, lda, rows, K, group_size, out);
}

} // namespace custom_kernel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
