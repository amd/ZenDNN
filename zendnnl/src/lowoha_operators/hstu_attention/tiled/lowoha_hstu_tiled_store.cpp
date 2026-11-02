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

#include "lowoha_operators/hstu_attention/tiled/lowoha_hstu_tiled_store.hpp"

#include <algorithm>
#include <cstring>
#include <immintrin.h>
#include "common/zendnnl_global.hpp"
#include "lowoha_operators/hstu_attention/common/lowoha_hstu_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace hstu_attention {

using zendnnl::common::bfloat16_t;

ZENDNNL_TARGET_NOINLINE("avx512f,avx512bw,avx512vl,avx512bf16,fma")
void store_row(const float *o, int64_t n, bfloat16_t *dst) {
    for (int64_t c = 0; c < n; c += 16) {
        const __mmask16 m
                = __mmask16(range_bits(0, std::min<int64_t>(16, n - c)));
        _mm256_mask_storeu_epi16(dst + c, m,
                (__m256i)_mm512_cvtneps_pbh(_mm512_maskz_loadu_ps(m, o + c)));
    }
}

void store_row(const float *o, int64_t n, float *dst) {
    std::memcpy(dst, o, n * sizeof(float));
}

} // namespace hstu_attention
} // namespace lowoha
} // namespace zendnnl
