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

constexpr data_type_t kF32 = data_type_t::f32;

struct f32_micro : kernel::fma_micro {
    static inline void to_a16(const float *src, float *dst) {
        _mm512_storeu_ps(dst, _mm512_loadu_ps(src));
    }
    static inline void pack16(const float *src, int64_t, float *dst) {
        to_a16(src, dst);
    }
};

} // namespace

template <>
void to_a<kF32>(const float *src, int64_t n, float *dst) {
    kernel::to_a<f32_micro>(src, n, dst);
}

template <>
void pack_b<kF32>(
        const float *src, int64_t r, int64_t n, int64_t ld, float *dst) {
    kernel::pack_b<f32_micro>(src, r, n, ld, dst);
}

template <>
void pack_b_transposed<kF32>(
        const float *src, int64_t r, int64_t k, float *dst) {
    kernel::pack_b_transposed<f32_micro>(src, r, k, dst);
}

template <>
void gemm<kF32>(int64_t m, int64_t n, int64_t ks, const float *a, int64_t a_row,
        int64_t a_step, const float *b, float *c, int64_t ldc, bool accumulate,
        tri_t tri) {
    kernel::gemm<f32_micro>(
            m, n, ks, a, a_row, a_step, b, c, ldc, accumulate, tri);
}

} // namespace ck
} // namespace gdn
} // namespace lowoha
} // namespace zendnnl

GDN_AVX512_END
