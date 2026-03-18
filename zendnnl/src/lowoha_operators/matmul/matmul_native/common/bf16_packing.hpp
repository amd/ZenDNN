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

#ifndef MATMUL_NATIVE_COMMON_BF16_PACKING_HPP
#define MATMUL_NATIVE_COMMON_BF16_PACKING_HPP

#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"

#include <cstdint>
#include <cstring>
#include <algorithm>
#include <immintrin.h>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

static_assert(NR_PACK % 32 == 0, "NR_PACK must be a multiple of 32 for AVX-512 VNNI packing");

// BF16 A micro-panel packing: row-major A -> contiguous MR x KB BF16 blocks
inline void pack_a_bf16_block(
    const uint16_t *A_src, uint16_t *pack_buf,
    int ic, int pc, int M, int K, int lda,
    int mb, int kb, int MR) {

    const int m_actual = std::min(mb, M - ic);
    const int k_actual = std::min(kb, K - pc);
    const int m_panels = (m_actual + MR - 1) / MR;
    const uint16_t *A_block = A_src + static_cast<size_t>(ic) * lda + pc;

    for (int ip = 0; ip < m_panels; ++ip) {
        const int i0 = ip * MR;
        const int mr = std::min(MR, m_actual - i0);
        uint16_t *dst = pack_buf + ip * MR * k_actual;
        for (int m = 0; m < mr; ++m)
            std::memcpy(dst + m * k_actual,
                        A_block + (i0 + m) * lda,
                        k_actual * sizeof(uint16_t));
        for (int m = mr; m < MR; ++m)
            std::memset(dst + m * k_actual, 0, k_actual * sizeof(uint16_t));
    }
}

// On-the-fly VNNI strip pack: pack NR_PACK columns for a K-block
__attribute__((target("avx512f,avx512bw")))
inline void pack_b_vnni_strip(
    const uint16_t *B, int ldb, bool transB,
    int col_start, int nr_act, int K, [[maybe_unused]] int K_padded,
    int pc, int kb,
    uint16_t *packed) {

    const int kb_padded = (kb + 1) & ~1;
    const int k_pairs = kb_padded / 2;
    const int out_stride = NR_PACK * VNNI_PAIR;

    for (int kp = 0; kp < k_pairs; ++kp) {
        uint16_t *d = packed + kp * out_stride;
        const int k0 = pc + kp * 2;
        const int k1 = k0 + 1;

        if (!transB && nr_act == NR_PACK) {
            const uint16_t *row0 = (k0 < K) ? B + k0 * ldb + col_start
                                             : nullptr;
            const uint16_t *row1 = (k1 < K) ? B + k1 * ldb + col_start
                                             : nullptr;
            for (int n = 0; n < NR_PACK; n += 32) {
                __m512i r0 = row0 ? _mm512_loadu_si512(row0 + n)
                                  : _mm512_setzero_si512();
                __m512i r1 = row1 ? _mm512_loadu_si512(row1 + n)
                                  : _mm512_setzero_si512();
                __m512i lo = _mm512_unpacklo_epi16(r0, r1);
                __m512i hi = _mm512_unpackhi_epi16(r0, r1);
                const __m512i idx_lo = _mm512_setr_epi32(
                    0,1,2,3, 16,17,18,19, 4,5,6,7, 20,21,22,23);
                const __m512i idx_hi = _mm512_setr_epi32(
                    8,9,10,11, 24,25,26,27, 12,13,14,15, 28,29,30,31);
                _mm512_storeu_si512(d + n * VNNI_PAIR,
                    _mm512_permutex2var_epi32(lo, idx_lo, hi));
                _mm512_storeu_si512(d + (n + 16) * VNNI_PAIR,
                    _mm512_permutex2var_epi32(lo, idx_hi, hi));
            }
        } else {
            for (int n = 0; n < nr_act; ++n) {
                uint16_t v0, v1;
                if (!transB) {
                    v0 = (k0 < K) ? B[k0 * ldb + (col_start + n)] : 0;
                    v1 = (k1 < K) ? B[k1 * ldb + (col_start + n)] : 0;
                } else {
                    v0 = (k0 < K) ? B[(col_start + n) * ldb + k0] : 0;
                    v1 = (k1 < K) ? B[(col_start + n) * ldb + k1] : 0;
                }
                d[n * VNNI_PAIR + 0] = v0;
                d[n * VNNI_PAIR + 1] = v1;
            }
            for (int n = nr_act; n < NR_PACK; ++n) {
                d[n * VNNI_PAIR + 0] = 0;
                d[n * VNNI_PAIR + 1] = 0;
            }
        }
    }
}


// BRGEMM overload: packs full K (no pc/kb subset)
__attribute__((target("avx512f,avx512bw")))
inline void pack_b_vnni_strip_full(
    const uint16_t *B, int ldb, bool transB,
    int col_start, int nr_act, int K, int K_padded,
    uint16_t *packed) {

    const int k_pairs = K_padded / 2;
    const int out_stride = NR_PACK * VNNI_PAIR;

    for (int kp = 0; kp < k_pairs; ++kp) {
        uint16_t *d = packed + kp * out_stride;
        const int k0 = kp * 2;
        const int k1 = k0 + 1;

        if (!transB && nr_act == NR_PACK) {
            const uint16_t *row0 = (k0 < K) ? B + k0 * ldb + col_start : nullptr;
            const uint16_t *row1 = (k1 < K) ? B + k1 * ldb + col_start : nullptr;
            for (int n = 0; n < NR_PACK; n += 32) {
                __m512i r0 = row0 ? _mm512_loadu_si512(row0 + n) : _mm512_setzero_si512();
                __m512i r1 = row1 ? _mm512_loadu_si512(row1 + n) : _mm512_setzero_si512();
                __m512i lo = _mm512_unpacklo_epi16(r0, r1);
                __m512i hi = _mm512_unpackhi_epi16(r0, r1);
                const __m512i idx_lo = _mm512_setr_epi32(0,1,2,3, 16,17,18,19, 4,5,6,7, 20,21,22,23);
                const __m512i idx_hi = _mm512_setr_epi32(8,9,10,11, 24,25,26,27, 12,13,14,15, 28,29,30,31);
                _mm512_storeu_si512(d + n * VNNI_PAIR, _mm512_permutex2var_epi32(lo, idx_lo, hi));
                _mm512_storeu_si512(d + (n + 16) * VNNI_PAIR, _mm512_permutex2var_epi32(lo, idx_hi, hi));
            }
        } else {
            for (int n = 0; n < nr_act; ++n) {
                uint16_t v0, v1;
                if (!transB) {
                    v0 = (k0 < K) ? B[k0 * ldb + (col_start + n)] : 0;
                    v1 = (k1 < K) ? B[k1 * ldb + (col_start + n)] : 0;
                } else {
                    v0 = (k0 < K) ? B[(col_start + n) * ldb + k0] : 0;
                    v1 = (k1 < K) ? B[(col_start + n) * ldb + k1] : 0;
                }
                d[n * VNNI_PAIR + 0] = v0;
                d[n * VNNI_PAIR + 1] = v1;
            }
            for (int n = nr_act; n < NR_PACK; ++n) {
                d[n * VNNI_PAIR + 0] = 0;
                d[n * VNNI_PAIR + 1] = 0;
            }
        }
    }
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_COMMON_BF16_PACKING_HPP
