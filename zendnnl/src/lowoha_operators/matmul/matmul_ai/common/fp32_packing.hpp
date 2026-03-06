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

#ifndef MATMUL_AI_COMMON_FP32_PACKING_HPP
#define MATMUL_AI_COMMON_FP32_PACKING_HPP

#include <cstdint>
#include <cstring>
#include <algorithm>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ai {

inline void pack_a_block(
    const float *A_src, float *pack_buf,
    int ic, int pc, int M, int K, int lda, bool transA,
    int mb, int kb, int MR) {

    const int m_actual = std::min(mb, M - ic);
    const int k_actual = std::min(kb, K - pc);
    const int m_panels = (m_actual + MR - 1) / MR;

    if (!transA) {
        const float *A_block = A_src + ic * lda + pc;
        for (int ip = 0; ip < m_panels; ++ip) {
            const int i0 = ip * MR;
            const int mr = std::min(MR, m_actual - i0);
            float *dst = pack_buf + ip * MR * k_actual;
            for (int m = 0; m < mr; ++m)
                std::memcpy(dst + m * k_actual,
                            A_block + (i0 + m) * lda,
                            k_actual * sizeof(float));
            for (int m = mr; m < MR; ++m)
                std::memset(dst + m * k_actual, 0, k_actual * sizeof(float));
        }
    } else {
        const float *A_block = A_src + pc * lda + ic;
        for (int ip = 0; ip < m_panels; ++ip) {
            const int i0 = ip * MR;
            const int mr = std::min(MR, m_actual - i0);
            float *dst = pack_buf + ip * MR * k_actual;
            for (int m = 0; m < mr; ++m)
                for (int kk = 0; kk < k_actual; ++kk)
                    dst[m * k_actual + kk] = A_block[kk * lda + (i0 + m)];
            for (int m = mr; m < MR; ++m)
                std::memset(dst + m * k_actual, 0, k_actual * sizeof(float));
        }
    }
}

} // namespace ai
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_AI_COMMON_FP32_PACKING_HPP
