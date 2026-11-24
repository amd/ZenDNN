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

/// N-tile flat parallel's weight packs.
///
/// A pack is built from a weight and its metadata alone: its format, shape,
/// layout and weight scales.  Nothing else about the calls that will read it
/// (the activation, a bias or the batch) enters it, so the same pack serves
/// every call of its format, and a pack can be made before any call does.
///
/// Each expert's weight is stored as blocks of `kNr` output columns, in the
/// weight's own column order.  A block holds its columns' weights for every
/// K group of the type's dot-product width, one word of its format
/// (formats/<format>/) per column and group:
///
///   bf16: [ceil(K/2)][32][2]    the VDPBF16PS pair order
///   f16:  [K][32]               one K element per lane
///   s8:   [ceil(K/4)][32][4]    the VPDPBUSD quad order
///   s4:   [K/8][32][4]          K octets; byte q of a column holds
///                               k = 8o + q in its low nibble and k = 8o + 4 + q
///                               in its high one, each XOR-ed with 8
///
/// K is zero-padded to the group width.  Every block row is 128 bytes (64 for
/// f16), so its two 16-column halves are one 64-byte line each.  A scaled
/// pack also keeps, per expert and in the weight's column order, the f32
/// weight scales of its G scale groups along K and, for a format with `sums`,
/// the sums its kernels' source recentering subtracts:
///
///   s8: comp[G][N_pad] = sum over the group of w,          scale[G][N_pad]
///   s4: comp[G][N_pad] = -128 * sum over the group of w,   scale[G][N_pad]
///
/// G is 1 for a per-channel s8 weight.  `N_pad` is `N` rounded up to whole
/// blocks.  Padded columns hold zero weights, zero sums and zero scales.

#ifndef ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_PACK_NR32_PACK_HPP
#define ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_PACK_NR32_PACK_HPP

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "lowoha_operators/matmul/group_matmul/common/group_matmul_parallel_common.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/formats/formats.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {
namespace nr32 {

/// Columns a pack of `N` columns holds.
int padded_cols(int N);

/// Bytes of one block of a pack of `desc`.
size_t block_bytes(const weight_desc &desc);

/// Whether `desc` is a weight N-tile flat parallel packs: a format's type, a
/// positive shape, a leading dimension that holds a row, and scale groups that
/// hold whole words and divide K, which a format that always groups its scales
/// needs and another scaled one may have: a group size that is a multiple
/// of 8 for s4 and, when an s8 weight has one, of 4.
bool packable(const weight_desc &desc);

/// One expert's pack.
struct packed_expert {
    /// `padded_cols(N) / kNr` blocks, 64-byte aligned.
    std::shared_ptr<uint8_t> blocks;
    /// Formats with `sums` only, in the weight's column order (see the file
    /// comment).  A weight's packs share its sums.
    std::shared_ptr<const std::vector<int32_t>> comp;
    std::vector<float> scale;
    /// The largest weight scale's magnitude, NaN when a scale is NaN.
    float max_scale = 0.0f;
};

/// What the packs of every expert of a weight share.
struct pack_layout {
    weight_desc desc;
    int n_pad = 0;
    /// Scale groups along K: K / group_size with a group size, else 1.
    int groups = 1;
    size_t block_bytes = 0;

    /// Its format, `desc.fmt`: set, as `desc` is `packable`.
    const format_desc &format() const {
        assert(desc.fmt != nullptr);
        return *desc.fmt;
    }
};

/// The layout of every pack of a `packable` weight `desc`.
pack_layout layout_of(const weight_desc &desc);

/// Every expert's pack of one weight.
struct packed_weights : pack_layout {
    std::vector<packed_expert> experts;

    const uint8_t *blocks(int e) const { return experts[e].blocks.get(); }
    /// Bytes the packs hold.
    size_t bytes() const;
};

/// Packs every expert of a weight on `num_threads` threads.  `weights[e]` is
/// expert e's weight in `desc`'s layout.  For s8 and s4, `scales[e]` is its
/// `{G, N}` row-major weight scales, G = K / group_size with a group size and
/// 1 without, in `scale_dt` (f32 or bf16); with `scales` empty, each pack
/// holds its blocks and sums only, for `with_scales` to complete.  bf16 and
/// f16 take no scales.  Returns `unimplemented` for a weight `packable`
/// rejects or a missing scale, and `memory_bad_size` when an allocation
/// fails; `out` then holds nothing.
status_t prepack(const weight_desc &desc,
        const std::vector<const void *> &weights,
        const std::vector<const void *> &scales, data_type_t scale_dt,
        int num_threads, packed_weights &out);

/// `out`: the pack `bare` of an s8 or s4 weight of `layout`, made without
/// scales, with the weight scales `scales` (as `prepack` takes them).  It
/// shares `bare`'s blocks and sums.  Returns `unimplemented` for another
/// scale type.
status_t with_scales(const pack_layout &layout, const packed_expert &bare,
        const void *scales, data_type_t scale_dt, packed_expert &out);

namespace test_api {

/// Test-only: while set, every expert's allocation in `prepack` fails inside
/// its parallel loop, as when memory runs out.
inline std::atomic<bool> s_pack_allocation_fails {false};

} // namespace test_api

/// Expert e's weight read back from its pack as `[N, K]` rows of `desc.wei`
/// (s4 nibble pairs, low nibble first, at the given row stride in nibbles).
/// For tests: the inverse of `prepack` on the unpadded columns.
void unpack(const packed_weights &pw, int e, void *dst, int ld);

} // namespace nr32
} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_PACK_NR32_PACK_HPP
