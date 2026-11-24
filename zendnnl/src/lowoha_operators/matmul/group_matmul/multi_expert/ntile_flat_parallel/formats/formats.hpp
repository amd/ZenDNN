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

/// The weight formats N-tile flat parallel packs, one row each
/// (formats/format.hpp), how a call finds its row, and the checks that each
/// row holds together.  A new format is a directory beside the others with
/// its row, its packer and its kernels, its header included here, one entry
/// in `kFormats`, and its two sources in group_matmul's CMakeLists.txt; the
/// README lists what else a format may need.

#ifndef ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_FORMATS_FORMATS_HPP
#define ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_FORMATS_FORMATS_HPP

#include <algorithm>
#include <iterator>

#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/formats/bf16/bf16.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/formats/f16/f16.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/formats/format.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/formats/s4/s4.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/formats/s8/s8.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {
namespace nr32 {

inline constexpr format_desc kFormats[] = {
        bf16::kFormat,
        f16::kFormat,
        s8::kFormat,
        s4::kFormat,
};

inline constexpr int kFormatCount = static_cast<int>(std::size(kFormats));

/// The row a call takes: of weight type `wei`, whose kernels read `src` rows,
/// and, where several such rows differ only in how their scales are encoded,
/// whose scales are `scales`.  Null when there is none.  A single such row is
/// the call's whatever its scales, and its API's checks then decide whether
/// it takes them.
constexpr const format_desc *format_for(
        const data_type_t wei, const data_type_t src, const scale_enc scales) {
    const format_desc *found = nullptr;
    int candidates = 0;
    for (const format_desc &f : kFormats) {
        if (f.wei != wei || f.src != src) continue;
        if (f.scales == scales) return &f;
        found = &f;
        ++candidates;
    }
    return candidates == 1 ? found : nullptr;
}

/// `f`'s index in `kFormats`, found by what the row says, not where it is
/// (another module's copy of the table has other addresses).
constexpr int format_index(const format_desc &f) {
    for (int i = 0; i < kFormatCount; ++i) {
        const format_desc &r = kFormats[i];
        if (r.wei == f.wei && r.src == f.src && r.scales == f.scales) return i;
    }
    return kFormatCount;
}

/// Whether a row of weight type `wei` exists.
constexpr bool has_format(const data_type_t wei) {
    for (const format_desc &f : kFormats)
        if (f.wei == wei) return true;
    return false;
}

/// Rows a kernel call of any format takes, at most.
inline constexpr int kMaxMr = [] {
    int rows = 0;
    for (const format_desc &f : kFormats)
        rows = std::max({rows, f.max_mr, f.max_mr_wide});
    return rows;
}();

/// Source rows the fronts and the engine pass or make (engine/source.hpp).
constexpr bool runs_source(const data_type_t t) {
    return t == data_type_t::s8 || t == data_type_t::bf16
            || t == data_type_t::f16;
}

/// Outputs the step stores, the scratch and the reduce take.
constexpr bool runs_output(const data_type_t t) {
    return t == data_type_t::bf16 || t == data_type_t::f16;
}

/// Whether `f` holds together:
///   * its words hold whole elements: copied ones of whole bytes (a nibble
///     format brings its own encoding, as s4's octets are four bytes of
///     nibbles), and a block row is whole vectors of whole words;
///   * a format with wide kernels has half-block vectors, and its wide and
///     two-chain row limits lie within the wide kernels';
///   * its kernels read rows and store outputs the shared code runs (an FP32
///     source or output adds its type there first: the README lists where);
///   * s8 sources come with weight scales and a bf16 output (W2's sources
///     are quantized from bf16 rows); weight sums, u8 sources and the
///     caller's s8 rows are s8 sources' only; a format that groups its scales
///     has scales; a format without scales reads and stores its own type, of
///     its elements' size;
///   * it has a packer and narrow kernels, wide kernels exactly when
///     `max_mr_wide > 0`, and, when its kernels need AVX-512 FP16, which a
///     compiler may not build, a check of whether the build has them.
constexpr bool format_holds(const format_desc &f) {
    const bool s8_src = f.src == data_type_t::s8;
    const bool words = f.enc == encode::copy
            ? f.elem_bits % 8 == 0
                    && f.word_bytes * 8 == f.elem_bits * f.k_per_word
            : f.elem_bits == 4 && f.word_bytes == 4 && f.k_per_word == 8;
    return words && f.word_bytes > 0 && 64 % f.word_bytes == 0
            && f.vec_cols() <= kNr && kNr % f.vec_cols() == 0 && f.max_mr > 0
            && (f.max_mr_wide == 0 || f.vec_cols() * 2 == kNr)
            && f.wide_rows <= f.max_mr_wide && f.two_chain_rows <= f.max_mr_wide
            && runs_source(f.src) && runs_output(f.dst)
            && (!s8_src || (f.scaled() && f.dst == data_type_t::bf16))
            && (!f.sums || s8_src) && (!f.u8_src || s8_src)
            && (!f.s8_rows || s8_src) && (!f.always_grouped || f.scaled())
            && (f.scaled()
                    || (f.src == f.wei && f.dst == f.wei
                            && f.elem_bits == 8 * f.src_bytes()))
            && f.pack != nullptr && f.narrow != nullptr
            && (f.wide != nullptr) == (f.max_mr_wide > 0)
            && (!needs(f.isa, isa_need::fp16) || f.built != nullptr);
}

constexpr bool same_name(const char *a, const char *b) {
    while (*a != '\0' && *a == *b) {
        ++a;
        ++b;
    }
    return *a == *b;
}

/// Whether every row holds together and differs from the others in its
/// types or its scales' encoding, and in its name.
constexpr bool formats_hold() {
    for (int i = 0; i < kFormatCount; ++i) {
        if (!format_holds(kFormats[i]) || format_index(kFormats[i]) != i)
            return false;
        for (int j = 0; j < i; ++j)
            if (same_name(kFormats[i].name, kFormats[j].name)) return false;
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

#endif // ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_FORMATS_FORMATS_HPP
