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

#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/formats/bf16/bf16.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {
namespace nr32 {
namespace bf16 {

// One column at a time: reading a column's row along K, rather than the
// block's rows at one K, keeps a row stride of a multiple of 4 KiB from
// mapping every read of a step to the same cache sets.
void pack(const void *weight, const weight_desc &d, const pack_out &out) {
    const int n_pad = out.n_pad;
    uint8_t *const blocks = out.blocks;
    const size_t bb = out.block_bytes;
    const auto *w = static_cast<const uint16_t *>(weight);
    const int K_pair = (d.K + 1) / 2;
    for (int b = 0; b < n_pad / kNr; ++b) {
        auto *dst = reinterpret_cast<uint16_t *>(blocks + b * bb);
        for (int n = 0; n < kNr; ++n) {
            const int col = b * kNr + n;
            for (int kp = 0; kp < K_pair; ++kp) {
                uint16_t *pair = dst + (static_cast<size_t>(kp) * kNr + n) * 2;
                const int k = kp * 2;
                pair[0] = col < d.N ? weight_at(w, d, k, col) : 0;
                pair[1] = col < d.N && k + 1 < d.K ? weight_at(w, d, k + 1, col)
                                                   : 0;
            }
        }
    }
}

} // namespace bf16
} // namespace nr32
} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
