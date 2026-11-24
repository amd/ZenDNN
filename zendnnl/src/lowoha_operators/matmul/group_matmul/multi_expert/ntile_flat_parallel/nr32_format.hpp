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

/// N-tile flat parallel's weight formats, one row each: how a pack stores the
/// weight, what a call over it passes and stores, and the shape of its kernels.
/// The packs, the plans, the tiles, a unit's cost and both APIs' checks read a
/// weight's row.  A new format is a row here, its packer (`kPackers`,
/// nr32_pack.cpp) and its kernel family (`kFamilies`, nr32_ukernel.cpp); the
/// build fails until the three agree.

#ifndef ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_NR32_FORMAT_HPP
#define ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_NR32_FORMAT_HPP

#include <algorithm>
#include <cstdint>
#include <iterator>

#include "common/data_types.hpp"

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

/// What a format's kernels need besides AVX-512 F, BW and VL; fp16 also a
/// build that has them (`f16_kernels_built`).
enum class isa_need : uint8_t { vnni, bf16, fp16 };

struct format_desc {
    data_type_t wei;
    /// The weight type's name, as the config's dtype filter takes it.
    const char *name;

    // How its packs store it (nr32_pack.hpp).
    int elem_bits;
    /// Bytes a column takes in one block row, and the K values they hold.
    int word_bytes;
    int k_per_word;
    encode enc;
    /// Weight scales in groups of whole words along K, which the packs keep
    /// as f32 beside the sums the kernels' source recentering subtracts.
    bool quantized;
    /// Its packs always take scale groups, one over K at least; a quantized
    /// weight without may have one scale per column.
    bool always_grouped;

    // What a call over it passes and stores.
    /// The source rows its kernels read: the weight's own 16-bit type, or s8
    /// rows the call quantizes from bf16.
    data_type_t src;
    /// What its kernels store.
    data_type_t dst;
    /// Its kernels also take the caller's s8 rows and their scales, which
    /// the grouped API passes as they are.
    bool s8_rows;

    // Its kernels (nr32_ukernel.hpp).
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

    /// Columns of one kernel vector: one 64-byte line of a block row.
    constexpr int vec_cols() const { return 64 / word_bytes; }
    /// Bytes of a source element its kernels read.
    constexpr int src_bytes() const { return src == data_type_t::s8 ? 1 : 2; }
    /// Bytes a weight element takes, rounded up to whole bytes.
    constexpr int whole_bytes() const { return (elem_bits + 7) / 8; }
};

inline constexpr format_desc kFormats[] = {
        {data_type_t::bf16, "bf16", 16, 4, 2, encode::copy, false, false,
                data_type_t::bf16, data_type_t::bf16, false, isa_need::bf16, 8,
                6, 0, 3, false},
        {data_type_t::f16, "f16", 16, 2, 1, encode::copy, false, false,
                data_type_t::f16, data_type_t::f16, false, isa_need::fp16, 8, 0,
                0, 0, false},
        {data_type_t::s8, "s8", 8, 4, 4, encode::copy, true, false,
                data_type_t::s8, data_type_t::bf16, true, isa_need::vnni, 12, 6,
                0, 0, true},
        {data_type_t::s4, "s4", 4, 4, 8, encode::nibble_octet, true, true,
                data_type_t::s8, data_type_t::bf16, false, isa_need::vnni, 6, 3,
                3, 0, false},
};

inline constexpr int kFormatCount = static_cast<int>(std::size(kFormats));

/// `wei`'s row index, or kFormatCount for a type N-tile flat parallel does not
/// pack.
constexpr int format_index(const data_type_t wei) {
    for (int i = 0; i < kFormatCount; ++i)
        if (kFormats[i].wei == wei) return i;
    return kFormatCount;
}

/// `wei`'s row, or nullptr for a type N-tile flat parallel does not pack.
constexpr const format_desc *format_of(const data_type_t wei) {
    const int i = format_index(wei);
    return i < kFormatCount ? &kFormats[i] : nullptr;
}

/// Rows a kernel call of any format takes, at most.
inline constexpr int kMaxMr = [] {
    int rows = 0;
    for (const format_desc &f : kFormats)
        rows = std::max({rows, f.max_mr, f.max_mr_wide});
    return rows;
}();

/// Whether `f` holds together: its words hold whole elements, copied ones of
/// whole bytes (a nibble format brings its own encoding, as s4's octets are
/// four bytes of nibbles), a block row is whole vectors of whole words, wide
/// kernels read half-block vectors, the row limits lie within the kernels',
/// and the quantized formats, and only they, read s8 sources, group their
/// scales, read u8 sources or the caller's s8 rows, and store another type.
constexpr bool format_holds(const format_desc &f) {
    const bool words = f.enc == encode::copy
            ? f.elem_bits % 8 == 0
                    && f.word_bytes * 8 == f.elem_bits * f.k_per_word
            : f.elem_bits == 4 && f.word_bytes == 4 && f.k_per_word == 8;
    return words && f.word_bytes > 0 && 64 % f.word_bytes == 0
            && f.vec_cols() <= kNr && kNr % f.vec_cols() == 0 && f.max_mr > 0
            && (f.max_mr_wide == 0 || f.vec_cols() < kNr)
            && f.wide_rows <= f.max_mr_wide && f.two_chain_rows <= f.max_mr_wide
            && f.quantized == (f.src == data_type_t::s8)
            && (!f.always_grouped || f.quantized) && (!f.u8_src || f.quantized)
            && (!f.s8_rows || f.quantized)
            && (f.quantized || (f.src == f.wei && f.dst == f.wei));
}

constexpr bool formats_hold() {
    for (int i = 0; i < kFormatCount; ++i) {
        if (!format_holds(kFormats[i]) || format_index(kFormats[i].wei) != i)
            return false;
    }
    return true;
}
static_assert(formats_hold(),
        "an N-tile flat parallel format row does not hold together");

} // namespace nr32
} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_NR32_FORMAT_HPP
