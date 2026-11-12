/********************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *******************************************************************************/

#include "lowoha_operators/gdn/gdn_gemm_ck/lowoha_gdn_ck.hpp"

#include <immintrin.h>
#include "lowoha_operators/gdn/gdn_gemm_ck/lowoha_gdn_ck_kernel.hpp"
#include "lowoha_operators/gdn/kernel/gdn_avx512_utils.hpp"

GDN_AVX512_BEGIN

namespace zendnnl {
namespace lowoha {
namespace gdn {
namespace ck {

namespace {

constexpr data_type_t kBf16 = data_type_t::bf16;

// vdpbf16ps on pairs: B in the pair layout, A broadcast one pair at a time.
struct bf16_micro {
    using a_t = uint32_t;
    using b_t = uint16_t;
    static constexpr int64_t kPack = 2;
    static inline __m512bh load_b(const uint16_t *p) {
        return (__m512bh)_mm512_loadu_si512(p);
    }
    static inline __m512bh bcast_a(const uint32_t *p) {
        return (__m512bh)_mm512_set1_epi32(static_cast<int>(*p));
    }
    static inline __m512 madd(__m512 acc, __m512bh a, __m512bh b) {
        return _mm512_dpbf16_ps(acc, a, b);
    }
    static inline void to_a16(const float *src, uint16_t *dst) {
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst),
                (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(src)));
    }
    // Interleaves the 16 bf16 values of two rows: out[2j] = row0[j],
    // out[2j + 1] = row1[j].
    static inline void pack16(const float *src, int64_t ld, uint16_t *dst) {
        const __m512i idx = _mm512_set_epi16(31, 15, 30, 14, 29, 13, 28, 12, 27,
                11, 26, 10, 25, 9, 24, 8, 23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18,
                2, 17, 1, 16, 0);
        const __m256i lo = (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(src));
        const __m256i hi
                = (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(src + ld));
        const __m512i v = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
        _mm512_storeu_si512(dst, _mm512_permutexvar_epi16(idx, v));
    }
};

} // namespace

template <>
void to_a<kBf16>(const float *src, int64_t n, uint16_t *dst) {
    kernel::to_a<bf16_micro>(src, n, dst);
}

template <>
void pack_b<kBf16>(
        const float *src, int64_t r, int64_t n, int64_t ld, uint16_t *dst) {
    kernel::pack_b<bf16_micro>(src, r, n, ld, dst);
}

template <>
void pack_b_transposed<kBf16>(
        const uint16_t *src, int64_t r, int64_t k, uint16_t *dst) {
    kernel::pack_b_transposed<bf16_micro>(src, r, k, dst);
}

template <>
void gemm<kBf16>(int64_t m, int64_t n, int64_t ks, const uint32_t *a,
        int64_t a_row, int64_t a_step, const uint16_t *b, float *c, int64_t ldc,
        bool accumulate, tri_t tri) {
    kernel::gemm<bf16_micro>(
            m, n, ks, a, a_row, a_step, b, c, ldc, accumulate, tri);
}

} // namespace ck
} // namespace gdn
} // namespace lowoha
} // namespace zendnnl

GDN_AVX512_END
