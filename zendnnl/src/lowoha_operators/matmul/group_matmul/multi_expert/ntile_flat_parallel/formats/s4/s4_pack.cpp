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

#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/formats/s4/s4.hpp"

#include <cstring>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {
namespace nr32 {
namespace s4 {

namespace {

/// The raw nibble of s4 element (k, col).
uint8_t nibble(const int8_t *w, const weight_desc &d, int k, int col) {
    const size_t ld = static_cast<size_t>(d.ldb);
    const size_t idx = d.trans ? static_cast<size_t>(col) * ld + k
                               : static_cast<size_t>(k) * ld + col;
    const auto byte = static_cast<uint8_t>(w[idx >> 1]);
    return (idx & 1u) == 0u ? (byte & 0x0Fu) : ((byte >> 4) & 0x0Fu);
}

int32_t s4_value(uint8_t raw) {
    return (raw & 0x08u) ? static_cast<int32_t>(raw) - 16
                         : static_cast<int32_t>(raw);
}

/// The packed octet of a column whose eight raw nibbles, k = 8o..8o+7, are
/// the four bytes of `raw` in memory order (low nibble first), with the sum
/// of their values.
inline uint32_t s4_octet(uint32_t raw, int32_t &sum) {
    const uint32_t lo = raw & 0x0F0F0F0Fu; // k = 0, 2, 4, 6
    const uint32_t hi = (raw >> 4) & 0x0F0F0F0Fu; // k = 1, 3, 5, 7
    const uint32_t q0 = (lo & 0xFFu) | (((lo >> 16) & 0xFFu) << 4);
    const uint32_t q1 = (hi & 0xFFu) | (((hi >> 16) & 0xFFu) << 4);
    const uint32_t q2 = ((lo >> 8) & 0xFFu) | ((lo >> 24) << 4);
    const uint32_t q3 = ((hi >> 8) & 0xFFu) | ((hi >> 24) << 4);
    const uint32_t out
            = (q0 | (q1 << 8) | (q2 << 16) | (q3 << 24)) ^ 0x88888888u;
    // Each stored nibble is its value plus 8.
    const uint32_t pairs = (out & 0x0F0F0F0Fu) + ((out >> 4) & 0x0F0F0F0Fu);
    sum = static_cast<int32_t>((pairs * 0x01010101u) >> 24) - 64;
    return out;
}

} // namespace

void pack(const void *weight, const weight_desc &d, const pack_out &out) {
    const int n_pad = out.n_pad;
    uint8_t *const blocks = out.blocks;
    const size_t bb = out.block_bytes;
    int32_t *const comp = out.sums;
    const auto *w = static_cast<const int8_t *>(weight);
    const int K_oct = d.K / 8;
    const int oct_per_group = d.group_size / 8;
    // [N, K] rows of an even length start on a byte: a column's octets are
    // four bytes each.
    const bool rows_of_bytes = d.trans && d.ldb % 2 == 0;
    for (int b = 0; b < n_pad / kNr; ++b) {
        uint8_t *dst = blocks + b * bb;
        for (int n = 0; n < kNr; ++n) {
            const int col = b * kNr + n;
            if (rows_of_bytes || col >= d.N) {
                const uint8_t *row = col < d.N
                        ? reinterpret_cast<const uint8_t *>(w)
                                + static_cast<size_t>(col) * (d.ldb / 2)
                        : nullptr;
                for (int o = 0; o < K_oct; ++o) {
                    uint32_t raw = 0;
                    if (row != nullptr) std::memcpy(&raw, row + o * 4, 4);
                    int32_t sum = 0;
                    const uint32_t out = s4_octet(raw, sum);
                    std::memcpy(dst + (static_cast<size_t>(o) * kNr + n) * 4,
                            &out, 4);
                    comp[static_cast<size_t>(o / oct_per_group) * n_pad + col]
                            -= sum * 128;
                }
                continue;
            }
            for (int o = 0; o < K_oct; ++o) {
                uint8_t *oct = dst + (static_cast<size_t>(o) * kNr + n) * 4;
                int32_t sum = 0;
                for (int q = 0; q < 4; ++q) {
                    const uint8_t lo
                            = col < d.N ? nibble(w, d, o * 8 + q, col) : 0;
                    const uint8_t hi
                            = col < d.N ? nibble(w, d, o * 8 + 4 + q, col) : 0;
                    oct[q] = static_cast<uint8_t>(
                            (lo ^ 0x08u) | ((hi ^ 0x08u) << 4));
                    sum += s4_value(lo) + s4_value(hi);
                }
                // An octet never straddles a group.
                comp[static_cast<size_t>(o * 8 / d.group_size) * n_pad + col]
                        -= sum * 128;
            }
        }
    }
}

} // namespace s4
} // namespace nr32
} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
