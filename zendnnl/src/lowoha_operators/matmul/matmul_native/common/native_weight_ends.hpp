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

#ifndef MATMUL_NATIVE_COMMON_NATIVE_WEIGHT_ENDS_HPP
#define MATMUL_NATIVE_COMMON_NATIVE_WEIGHT_ENDS_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

/// Bytes each end digest of a buffer covers.
inline constexpr size_t kWeightEndBytes = 64;

/// A digest of one end of [p, p + bytes) and of bytes: the first
/// kWeightEndBytes, or the last ones when tail (all bytes below that).  It
/// reads only those bytes, so an end can be checked while the rest of the
/// range may no longer be readable.  A buffer rewritten in place keeps the
/// digests of its two ends until those bytes change: they tell a buffer that
/// was overwritten, or freed and reused, from the image it held, unless the
/// new bytes equal the image at both ends.  No allocation.
inline uint64_t weight_end_digest(
        const void *p, size_t bytes, bool tail) noexcept {
    uint64_t h = (0x9e3779b97f4a7c15ull ^ static_cast<uint64_t>(bytes))
            + (tail ? 0x632be59bd9b4e019ull : 0);
    if (p == nullptr || bytes < sizeof(uint64_t)) return h;
    const unsigned char *b = static_cast<const unsigned char *>(p);
    const size_t span = bytes < kWeightEndBytes ? bytes : kWeightEndBytes;
    const size_t from = tail ? bytes - span : 0;
    for (size_t at = 0; at < span; at += sizeof(uint64_t)) {
        const size_t word
                = at + sizeof(uint64_t) <= span ? at : span - sizeof(uint64_t);
        uint64_t w = 0;
        std::memcpy(&w, b + from + word, sizeof(w));
        h = (h ^ w) * 0xff51afd7ed558ccdull;
        h ^= h >> 29;
    }
    return h;
}

/// The digests of the two ends of a range of bytes (weight_end_digest).
struct weight_ends_t {
    uint64_t head = 0;
    uint64_t tail = 0;

    bool operator==(const weight_ends_t &o) const {
        return head == o.head && tail == o.tail;
    }
};

inline weight_ends_t weight_ends_of(const void *p, size_t bytes) noexcept {
    return {weight_end_digest(p, bytes, false),
            weight_end_digest(p, bytes, true)};
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_COMMON_NATIVE_WEIGHT_ENDS_HPP
