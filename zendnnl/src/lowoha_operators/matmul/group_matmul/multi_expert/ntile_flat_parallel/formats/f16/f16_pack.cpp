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

#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/formats/f16/f16.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {
namespace nr32 {
namespace f16 {

// One column at a time, as bf16's packer and for its reason.
void pack(const void *weight, const weight_desc &d, const pack_out &out) {
    const int n_pad = out.n_pad;
    uint8_t *const blocks = out.blocks;
    const size_t bb = out.block_bytes;
    const auto *w = static_cast<const uint16_t *>(weight);
    for (int b = 0; b < n_pad / kNr; ++b) {
        auto *dst = reinterpret_cast<uint16_t *>(blocks + b * bb);
        for (int n = 0; n < kNr; ++n) {
            const int col = b * kNr + n;
            for (int k = 0; k < d.K; ++k) {
                dst[static_cast<size_t>(k) * kNr + n]
                        = col < d.N ? weight_at(w, d, k, col) : 0;
            }
        }
    }
}

} // namespace f16
} // namespace nr32
} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
