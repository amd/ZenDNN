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

/// Source rows as the kernels read them.  `source_spec` says what one
/// projection's kernels read: the caller's rows, or Op1's, as they are (a
/// 16-bit format), or s8 rows the call makes, from bf16 rows or the caller's
/// s8 rows, with their scales, XOR-ed with 0x80 for the u8-source kernels.
/// `source_rows` holds the rows a call makes in the scratch, and the
/// functions below make and move them, so the API fronts and the engine never
/// handle a row's parts themselves.  A format whose kernels read more about
/// their rows (per-block sums for a code-bias correction) adds it to
/// `source_spec`, `source_rows` and `nr32::tile_source`, and to these
/// functions.

#ifndef ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_ENGINE_SOURCE_HPP
#define ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_ENGINE_SOURCE_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/tile/nr32_tile.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {
namespace detail {

/// What one projection's kernels read as source rows.
struct source_spec {
    /// s8 rows the call makes in the scratch, else the rows as they are.
    bool s8 = false;
    /// Scales per s8 row: one, or one per weight scale group.
    int groups = 1;
    /// The s8 rows are XOR-ed with 0x80, as the u8-source kernels read them.
    bool u8 = false;
    /// Bytes of a source element.
    int elem = 2;

    size_t elem_bytes() const { return static_cast<size_t>(elem); }
};

/// The source of a projection over `layout` planned as `plan`.
source_spec source_of(
        const nr32::pack_layout &layout, const nr32::tile_plan &plan);

/// Bytes past the last s8 row the kernels may read (`nr32::run_tile`).
inline constexpr size_t kRowSlack = nr32::kRowReadSlack;

/// One projection's source rows a call makes in the scratch: row r's `len`
/// elements and, for s8 rows, its `groups` scales.
struct source_rows {
    std::vector<uint8_t> rows;
    std::vector<float> scales;

    /// Room for `n` rows of `len` elements of source `spec`, and the bytes
    /// the kernels may read past the last; grow-only, as the scratch is.
    void grow(const source_spec &spec, const size_t n, const int len) {
        const size_t bytes
                = n * static_cast<size_t>(len) * spec.elem_bytes() + kRowSlack;
        if (rows.size() < bytes) rows.resize(bytes);
        const size_t count = n * static_cast<size_t>(spec.groups);
        if (spec.s8 && scales.size() < count) scales.resize(count);
    }

    uint8_t *row(const source_spec &spec, const int r, const int len) {
        return rows.data()
                + static_cast<size_t>(r) * static_cast<size_t>(len)
                * spec.elem_bytes();
    }

    float *row_scales(const source_spec &spec, const int r) {
        return scales.data() + static_cast<size_t>(r) * spec.groups;
    }

    /// The rows from row `r` on, as a tile reads them.
    nr32::tile_source view(
            const source_spec &spec, const int r, const int len) const {
        nr32::tile_source t;
        t.rows = rows.data()
                + static_cast<size_t>(r) * static_cast<size_t>(len)
                        * spec.elem_bytes();
        t.lda = len;
        t.scales = spec.s8
                ? scales.data() + static_cast<size_t>(r) * spec.groups
                : nullptr;
        return t;
    }

    /// Bytes the rows and scales keep.
    size_t bytes() const {
        return rows.capacity() + scales.capacity() * sizeof(float);
    }
};

/// XORs each of `n` bytes with 0x80.
inline void flip_to_u8(uint8_t *bytes, const int n) {
    constexpr uint64_t kFlip = 0x8080808080808080ULL;
    int c = 0;
    for (; c + 8 <= n; c += 8) {
        uint64_t w;
        std::memcpy(&w, bytes + c, sizeof(w));
        w ^= kFlip;
        std::memcpy(bytes + c, &w, sizeof(w));
    }
    for (; c < n; ++c)
        bytes[c] = static_cast<uint8_t>(bytes[c] ^ 0x80u);
}

/// s8 row `r` of `out`, of source `spec`, from the bf16 row `in` of `len`
/// values: quantized in `spec.groups` equal runs, each with its scale, and
/// XOR-ed with 0x80 when `spec.u8`.  One row per call: the quantizer runs a
/// single row on the calling thread, but may fork a team for several.
void quantize_row(const source_spec &spec, const uint16_t *in, int len,
        source_rows &out, int r);

/// s8 row `r` of `out` from the caller's s8 row `in` of `len` values, whose
/// scales the caller gives, XOR-ed with 0x80 when `spec.u8`.
inline void copy_s8_row(const source_spec &spec, const int8_t *in,
        const int len, source_rows &out, const int r) {
    uint8_t *const row = out.row(spec, r, len);
    std::memcpy(row, in, static_cast<size_t>(len));
    if (spec.u8) flip_to_u8(row, len);
}

/// Row `r` of `to` from row `from_r` of `from`, its elements and scales: a
/// row the call made once and reads in several places (a routed token's
/// slots).
inline void copy_row(const source_spec &spec, source_rows &from,
        const int from_r, source_rows &to, const int r, const int len) {
    std::memcpy(to.row(spec, r, len), from.row(spec, from_r, len),
            static_cast<size_t>(len) * spec.elem_bytes());
    if (spec.s8) {
        std::memcpy(to.row_scales(spec, r), from.row_scales(spec, from_r),
                sizeof(float) * static_cast<size_t>(spec.groups));
    }
}

/// Row `r` of `out` from `in`: `len` source elements as they are (16-bit
/// rows a call gathers).
inline void copy_in(const source_spec &spec, const void *in, const int len,
        source_rows &out, const int r) {
    std::memcpy(out.row(spec, r, len), in,
            static_cast<size_t>(len) * spec.elem_bytes());
}

} // namespace detail
} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_ENGINE_SOURCE_HPP
