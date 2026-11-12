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

#ifndef MATMUL_NATIVE_COMMON_NATIVE_WEIGHT_VIEW_HPP
#define MATMUL_NATIVE_COMMON_NATIVE_WEIGHT_VIEW_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "common/error_status.hpp"
#include "lowoha_operators/matmul/lowoha_common.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_weight_cache.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::error_handling::status_t;

/// A weight in the native VNNI layout, read in place.  Per block of
/// pack_nr columns:
///   bf16: [ceil(K / 2)][pack_nr][2] bf16, odd K padded with zero.
///   s8:   [ceil(K / 4)][pack_nr][4] s8, zero padded, then [pack_nr] int32
///         column sums over the real K.
///   s4:   [ceil(K / 8)][pack_nr][4] bytes: byte (o, n, q) holds w + 8 of
///         element (8o + q, n) in its low nibble and of (8o + 4 + q, n) in
///         its high nibble, padded with w = 0 (nibble 8): the low nibbles of
///         an octet are K quad 2o, the high nibbles K quad 2o + 1.
///   f32:  [K][pack_nr] f32.
///   f16:  [K][pack_nr] f16.
/// Pack column n holds weight column n, whatever the call: the layout
/// depends on the weight alone.  The blocks hold native_padded_n(N, pack_nr)
/// pack columns: the N of the weight, then zero columns (column sums zero)
/// up to whole blocks.  Every type but f16 takes 4 bytes per column in a K
/// group, f16 2.  A view with paired serves a gated call over [gate | up]
/// halves of N a multiple of 32: the tile of pack columns n0 .. n0 + nr (nr
/// a multiple of 32) computes the gate columns n0 / 2 .. n0 / 2 + nr / 2 in
/// its first nr / 2 columns and the up columns N / 2 + n0 / 2 on in the
/// others, so that its column t < nr / 2 and column nr / 2 + t pair up.
/// Every run of 16 tile columns from a multiple of 16 lies inside one block;
/// a kernel call of 32 columns takes its second 16 from wherever they lie
/// (the up run of a tile of 32).
struct native_weight_view_t {
    const uint8_t *base = nullptr;
    data_type_t wei_dt = data_type_t::none;
    int K = 0;
    int N = 0;
    int pack_nr = 0;
    /// K pairs (bf16), K quads (s8), K octets (s4) or K rows (f32, f16)
    int k_groups = 0;
    size_t block_bytes = 0; ///< one block, column sums included
    bool paired = false;

    const uint8_t *block(int nb) const {
        return base + static_cast<size_t>(nb) * block_bytes;
    }

    /// s8 only: the column sums of block nb.
    const int32_t *col_sums(int nb) const {
        return reinterpret_cast<const int32_t *>(
                block(nb) + static_cast<size_t>(k_groups) * pack_nr * 4);
    }

    /// Elements of wei_dt (bytes of an s4 weight) between consecutive K
    /// groups of one column.
    int group_stride() const {
        if (wei_dt == data_type_t::f32 || wei_dt == data_type_t::f16)
            return pack_nr;
        return pack_nr * (wei_dt == data_type_t::bf16 ? 2 : 4);
    }

    /// K group 0 of weight column c.
    const uint8_t *column(int c) const {
        return block(c / pack_nr) + static_cast<size_t>(c % pack_nr) * 4;
    }

    /// Bytes from K group 0 of weight column c0 to that of a later column c1
    /// (the up run a paired tile's call reads): beyond the int range on a
    /// large weight.
    size_t column_distance(int c0, int c1) const {
        return static_cast<size_t>(column(c1) - column(c0));
    }

    /// s8 only: the column sum of weight column c, the next ones of its
    /// block after it.
    const int32_t *col_sum(int c) const {
        return col_sums(c / pack_nr) + c % pack_nr;
    }

    /// Weight column of column t of the tile of pack columns n0 .. n0 + nr.
    int tile_col(int n0, int nr, int t) const {
        if (!paired) return n0 + t;
        const int half = nr / 2;
        return t < half ? n0 / 2 + t : N / 2 + n0 / 2 + (t - half);
    }
};

/// The tile width at which a gated call over [gate | up] halves of N
/// columns pairs its halves (native_weight_view_t), from a plan of tiles nr
/// columns wide over m_blocks row blocks for a team of threads; 0 when it
/// stages its rows instead (N not a multiple of 32).  Tiles of 64 columns or
/// more pair whole blocks of each half.  A plan of narrower tiles pairs
/// whole blocks at 64 columns where those tiles deal out evenly over the
/// team or number 8 per thread, else runs of 16 columns of each half in
/// tiles of 32, whose kernel calls read the up run at a second address:
/// the tile count its plan was balanced for (half of it for tiles of 16).
inline int native_paired_width(int N, int nr, int m_blocks, int threads) {
    if (nr <= 0 || nr % 16 != 0 || N % 32 != 0) return 0;
    if (nr >= 64) return nr;
    if (N % 64 == 0) {
        const int64_t team = std::max(1, threads);
        const int64_t tiles = static_cast<int64_t>(N / 64) * m_blocks;
        if (tiles % team == 0 || tiles >= 8 * team) return 64;
    }
    return 32;
}

/// How the call's weight reaches the native BRGEMM.
struct native_weight_ref_t {
    /// The raw weight, an AOCL prepack (aocl_blocked), or an image an earlier
    /// native call rewrote in place.
    const void *weight = nullptr;
    int pack_nr = 0;
    /// Effective weight cache mode (see get_native_layout): 0 packs for this
    /// call only, 1 uses the native layout cache, 2 rewrites the buffer in
    /// place when the image fits it (a bf16, f16 or s4 raw weight, or an AOCL
    /// prepack), else as 1 (an s8 or f32 raw weight, or a weight of padded
    /// columns).
    int32_t weight_cache_mode = 1;
    /// Writable bytes at weight (0: the logical extent, or for an AOCL
    /// prepack its size).
    size_t capacity_bytes = 0;
    /// The buffer holds the weight as an aocl_dlp_blocked prepack (or an
    /// image an earlier native call rewrote in place): the raw weight is its
    /// AOCL inverse, rows transB ? K : N elements apart.
    bool aocl_blocked = false;
    /// The call's check of the buffer for an in-place image, which the
    /// layout lookup does not repeat (get_native_layout).
    native_image_check_t image;
};

/// Owns a pack made for one call and frees it on scope exit.
class native_weight_lease_t {
public:
    native_weight_lease_t() = default;
    ~native_weight_lease_t();
    native_weight_lease_t(const native_weight_lease_t &) = delete;
    native_weight_lease_t &operator=(const native_weight_lease_t &) = delete;

    void own(void *packed);

private:
    void *packed_ = nullptr;
};

/// Resolves ref to a view through get_native_layout under its weight cache
/// mode, for a weight native_layout_fits takes; a missing pack uses up to
/// num_threads threads.  Fails rather than guessing when the buffer cannot be
/// read as the layout.
status_t acquire_native_weight(const native_weight_ref_t &ref,
        data_type_t wei_dt, int K, int N, int ldb, bool transB,
        native_weight_view_t &view, native_weight_lease_t &lease,
        int num_threads = 1);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_COMMON_NATIVE_WEIGHT_VIEW_HPP
