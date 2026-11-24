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

/// What a weight format of N-tile flat parallel is: one `format_desc` row,
/// which formats/<format>/<format>.hpp defines and formats/formats.hpp lists.
/// A row says how the format's packs store it and its scales, what a call
/// over it passes and stores, the shape of its kernels, and which functions
/// pack it and pick its kernels; a packer reads the weight `weight_desc`
/// describes.  A call finds its row by its weight type, the type of the
/// source rows its kernels read and its scales' encoding (`format_for`), so
/// formats may share a weight type.  From there the packs, the plans, the
/// tiles, a unit's cost and both APIs' checks read the row, not the types.

#ifndef ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_FORMATS_FORMAT_HPP
#define ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_FORMATS_FORMAT_HPP

#include <cstddef>
#include <cstdint>

#include "common/data_types.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/kernel/nr32_ukernel.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {
namespace nr32 {

using zendnnl::common::data_type_t;

/// Output columns per pack block.
inline constexpr int kNr = 32;

/// How a pack stores a column's K values in one word of a block row.
enum class encode : uint8_t {
    /// As the weight stores them, in K order.
    copy,
    /// K octets: byte q holds k = 8o + q in its low nibble and k = 8o + 4 + q
    /// in its high one, each XOR-ed with 8.
    nibble_octet,
};

/// How a format's weight scales are encoded.
enum class scale_enc : uint8_t {
    /// No scales: the weights are the values (bf16, f16).
    none,
    /// One real scale, f32, bf16 or f16, per column or per (scale group,
    /// column), which the packs keep as f32.
    real,
    // Block-scaled formats add their encodings here: e8m0 per 32 elements
    // for the MX formats, e4m3 per 16 with an f32 scale per tensor for
    // NVFP4.
};

/// The encoding of the weight scales a caller gives at `buff` in `dt`: none
/// without a buffer, else real numbers, the only kind a caller gives today
/// (f32, bf16 or f16).
constexpr scale_enc scale_enc_of(const void *buff, const data_type_t dt) {
    (void)dt;
    return buff == nullptr ? scale_enc::none : scale_enc::real;
}

/// What a format's kernels need besides AVX-512 F, BW and VL: a set.
enum class isa_need : uint8_t {
    none = 0,
    vnni = 1 << 0,
    bf16 = 1 << 1,
    fp16 = 1 << 2,
};

constexpr isa_need operator|(const isa_need a, const isa_need b) {
    return static_cast<isa_need>(
            static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
}

/// Whether `set` holds `one`.
constexpr bool needs(const isa_need set, const isa_need one) {
    return (static_cast<uint8_t>(set) & static_cast<uint8_t>(one)) != 0;
}

struct format_desc;

/// A weight as a pack reads it.
struct weight_desc {
    /// Its format: a row of `kFormats`.
    const format_desc *fmt = nullptr;
    int K = 0;
    int N = 0;
    /// `[N, K]` rows (`nn.Linear`) when true, else `[K, N]`.
    bool trans = true;
    /// Elements between consecutive rows of the weight.
    int ldb = 0;
    /// Weights per scale group along K: s4 always, s8 with per-group scales.
    /// Zero for an s8 weight with one scale per column, and for bf16 and f16.
    int group_size = 0;
};

/// Element (k, col) of a weight of `d` whole-byte elements, as its raw bit
/// pattern (for the packers).
template <typename T>
T weight_at(const T *w, const weight_desc &d, int k, int col) {
    const size_t ld = static_cast<size_t>(d.ldb);
    return d.trans ? w[static_cast<size_t>(col) * ld + k]
                   : w[static_cast<size_t>(k) * ld + col];
}

/// Where a packer writes one expert's pack.  Formats whose packers write
/// more, such as scales they derive from the weight or a scale per tensor,
/// add it here; the other packers leave it alone.
struct pack_out {
    /// `n_pad / kNr` blocks of `block_bytes` each.
    uint8_t *blocks = nullptr;
    size_t block_bytes = 0;
    int n_pad = 0;
    /// For a format with `sums`: `[groups][n_pad]`, zeroed, to which the
    /// packer adds each column's sums; else null.
    int32_t *sums = nullptr;
};

/// Writes expert weight `weight`'s pack for `d` into `out`.
using pack_fn = void (*)(
        const void *weight, const weight_desc &d, const pack_out &out);
/// The narrow kernel for `mr` rows with epilogue `ep`, or nullptr; `src_u8`,
/// `unaligned` and `groups` as `select` takes them, each read only by a
/// format that has such variants.
using narrow_fn = ukernel_fn (*)(
        int mr, epilogue ep, bool src_u8, bool unaligned, scale_groups groups);
/// The wide split kernel for `mr` rows with split epilogue `ep`, or nullptr.
using wide_fn = ukernel_fn (*)(
        int mr, epilogue ep, bool src_u8, bool two_chains, scale_groups groups);

struct format_desc {
    /// The weight's element type.
    data_type_t wei;
    /// The format's name, as the config's dtype filter takes it: unique.
    const char *name;

    // How its packs store it (pack/nr32_pack.hpp).
    /// Bits of one weight element.
    int elem_bits;
    /// Bytes a column takes in one block row, and the K values they hold.
    int word_bytes;
    int k_per_word;
    /// How a word holds its K values.
    encode enc;
    /// How its weight scales are encoded; the packs keep real scales as f32,
    /// per column or per scale group of whole words along K.
    scale_enc scales;
    /// Its packs always take scale groups, one over K at least; a scaled
    /// weight without may have one scale per column.
    bool always_grouped;
    /// Its packs keep each column's weight sums per scale group, which its
    /// kernels subtract when they recenter s8 sources.
    bool sums;

    // What a call over it passes and stores.
    /// The source rows its kernels read: the weight's own 16-bit type, or s8
    /// rows the call makes, quantized from bf16 or the caller's (`s8_rows`).
    data_type_t src;
    /// What its kernels store.
    data_type_t dst;
    /// A call may give s8 rows and their scales in place of bf16 ones (the
    /// grouped API's s8 sources), which the grouped front copies into the
    /// rows its kernels read.
    bool s8_rows;

    // Its kernels (kernel/nr32_ukernel.hpp).
    /// The instructions they need.
    isa_need isa;
    /// Rows one narrow kernel call takes, and one wide call (0: no wide
    /// kernels).
    int max_mr;
    int max_mr_wide;
    /// Tiles of more rows run a wide plan's narrow kernels; 0 for no limit.
    int wide_rows;
    /// Rows at or below which the wide kernels sum in two chains, as the
    /// narrow ones do; 0 for none.
    int two_chain_rows;
    /// Kernels that read source bytes already XOR-ed with 0x80.
    bool u8_src;

    // Its functions, in formats/<format>/.
    /// Writes one expert's pack.
    pack_fn pack;
    /// Picks its narrow kernels.
    narrow_fn narrow;
    /// Picks its wide split kernels; null for a format without them.
    wide_fn wide;
    /// Whether the build has its kernels; null when it always has (a
    /// compiler without AVX-512 FP16 builds none of f16's).
    bool (*built)();

    /// Whether its weights have scales.
    constexpr bool scaled() const { return scales != scale_enc::none; }
    /// Columns of one kernel vector: one 64-byte line of a block row.
    constexpr int vec_cols() const { return 64 / word_bytes; }
    /// Bytes of a source element its kernels read.
    constexpr int src_bytes() const {
        return src == data_type_t::s8 ? 1 : src == data_type_t::f32 ? 4 : 2;
    }
    /// Bytes of an output element its kernels store.
    constexpr int dst_bytes() const { return dst == data_type_t::f32 ? 4 : 2; }
    /// Bytes a weight element takes, rounded up to whole bytes.
    constexpr int whole_bytes() const { return (elem_bits + 7) / 8; }
};

} // namespace nr32
} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_FORMATS_FORMAT_HPP
