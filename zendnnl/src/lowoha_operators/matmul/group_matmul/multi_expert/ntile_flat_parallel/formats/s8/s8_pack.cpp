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

#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/formats/s8/s8.hpp"

#include <cstring>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {
namespace nr32 {
namespace s8 {

void pack(const void *weight, const weight_desc &d, const pack_out &out) {
    const int n_pad = out.n_pad;
    uint8_t *const blocks = out.blocks;
    const size_t bb = out.block_bytes;
    int32_t *const comp = out.sums;
    const auto *w = static_cast<const int8_t *>(weight);
    const int K_quad = (d.K + 3) / 4;
    const int full_quads = d.K / 4;
    // A quad never straddles a scale group.
    const int group_quads = d.group_size > 0 ? d.group_size / 4 : K_quad;
    for (int b = 0; b < n_pad / kNr; ++b) {
        auto *dst = reinterpret_cast<int8_t *>(blocks + b * bb);
        for (int n = 0; n < kNr; ++n) {
            const int col = b * kNr + n;
            const auto add = [&](const int kq, const int32_t sum) {
                comp[static_cast<size_t>(kq / group_quads) * n_pad + col]
                        += sum;
            };
            int kq = 0;
            if (d.trans && col < d.N) {
                // A column's weights are one contiguous row.
                const int8_t *row = w + static_cast<size_t>(col) * d.ldb;
                for (; kq < full_quads; ++kq) {
                    int8_t *quad
                            = dst + (static_cast<size_t>(kq) * kNr + n) * 4;
                    std::memcpy(quad, row + kq * 4, 4);
                    add(kq, quad[0] + quad[1] + quad[2] + quad[3]);
                }
            }
            for (; kq < K_quad; ++kq) {
                int8_t *quad = dst + (static_cast<size_t>(kq) * kNr + n) * 4;
                int32_t sum = 0;
                for (int q = 0; q < 4; ++q) {
                    const int k = kq * 4 + q;
                    quad[q] = col < d.N && k < d.K ? weight_at(w, d, k, col)
                                                   : 0;
                    sum += quad[q];
                }
                add(kq, sum);
            }
        }
    }
}

} // namespace s8
} // namespace nr32
} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
