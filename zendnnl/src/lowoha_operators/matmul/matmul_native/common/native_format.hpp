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

#ifndef MATMUL_NATIVE_COMMON_NATIVE_FORMAT_HPP
#define MATMUL_NATIVE_COMMON_NATIVE_FORMAT_HPP

#include <cassert>
#include <cstdint>
#include <iterator>
#include <limits>

#include "lowoha_operators/matmul/lowoha_common.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/int8/native_int4_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/int8/native_int8_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/native_kernel_registry.hpp"
#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_int8_scheme.hpp"
#include "lowoha_operators/matmul/matmul_native/native_layout_policy.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

// The weights and kernels of the native BRGEMM, in three tables.  A format
// is how a weight is stored in the layout and which calls over it the
// layout rule takes; a family is how a kernel multiplies; a route is a
// format on a family for one INT8 scheme, as the kernel registry serves it.

/// How a weight is stored in the layout.
enum class native_format_t : uint8_t { bf16, s8, s4, f32, f16 };

/// The instruction set a format's or a family's kernels need besides
/// native_base_isa.
enum class native_isa_t : uint8_t {
    base,
    avx512_bf16,
    avx512_vnni,
    avx512_fp16
};

/// The granularities of weight scales; a set of them is a mask of
/// native_grain_bit.
enum class native_grain_t : uint8_t { tensor, channel, group };

constexpr uint8_t native_grain_bit(native_grain_t g) {
    return static_cast<uint8_t>(1u << static_cast<unsigned>(g));
}

/// A set of data types is a mask of these bits.
constexpr uint32_t native_type_bit(data_type_t dt) {
    return 1u << static_cast<unsigned>(dt);
}
static_assert(static_cast<unsigned>(data_type_t::f8_e8m0) < 32,
        "every data type has a bit of a native_type_bit mask");

/// The foreign prepack a weight may come in, which the weight cache
/// unreorders: the aocl_dlp_blocked one of a bf16 weight, or of an s8 weight
/// for a u8 source.
enum class native_prepack_t : uint8_t { none, bf16, u8s8 };

/// How the pack writes a weight's elements into the layout's words.
enum class native_encode_t : uint8_t {
    copy, ///< as the weight stores them
    nibble_offset, ///< two's-complement nibbles plus 8
};

/// The layout's blocks of kNativeLayoutPackNr columns: K groups of one word
/// per column, zero past K, then with col_sums one int32 per column, the
/// sum of its weights.  A word of 2 planes holds its first k_per_word / 2
/// K values in the low nibbles of its bytes and the others in the high
/// nibbles, byte q K values q and k_per_word / 2 + q.
struct native_geometry_t {
    int elem_bits;
    int word_bytes;
    int k_per_word;
    int planes;
    native_encode_t encode;
    bool col_sums;
};

/// Sources the layout rule refuses, and why.
struct native_refusal_t {
    uint32_t types; ///< a native_type_bit mask
    const char *why;
};

/// The largest K or weight scale group taken, and why a larger one is
/// refused.
struct native_limit_t {
    int max;
    const char *why;
};

inline constexpr native_limit_t kNativeNoLimit {
        std::numeric_limits<int32_t>::max(), nullptr};

/// The calls over a format's layout that the layout rule takes, checked in
/// this order: the source (each refusal in turn), the instruction set, K
/// against k, the weight scale groups, then K against the largest whose
/// whole K groups an int holds (native_max_layout_k).
struct native_contract_t {
    native_refusal_t src_refusals[2];
    native_isa_t isa;
    /// A bf16 or f32 source the call quantizes to u8 or s8 multiplies as
    /// the quantized source.
    bool dynamic_quant;
    /// Bits of a source element the kernels read (of the quantized source
    /// when the call quantizes it).
    int src_bits;
    /// The granularities of its weight scales, 0 for a weight without
    /// scales.  With group, a weight in groups along K takes groups of whole
    /// K quads, within group; with channel too, one group over K is a scale
    /// per column, which takes any K.
    uint8_t scale_grains;
    native_limit_t k;
    const char *group_refusal;
    native_limit_t group;
    /// Why a K past native_max_layout_k is refused; nullptr where k already
    /// keeps K below it.
    const char *layout_k_refusal;
};

/// What the weight cache does with a format's weights.
struct native_cache_rules_t {
    /// Rewritten in place at weight cache mode 2 when the layout fits the
    /// buffer.
    bool in_place;
    /// An image written in place unpacks to the raw weight.
    bool unpack;
    native_prepack_t foreign_prepack;
};

struct native_format_desc_t {
    native_format_t id;
    /// The weight type's name in the planner's lines.
    const char *name;
    data_type_t wei_dt;
    native_geometry_t geometry;
    native_contract_t contract;
    native_cache_rules_t cache;

    constexpr bool quantized() const { return contract.scale_grains != 0; }
};

/// Indexed by native_format_t.
inline constexpr native_format_desc_t kNativeFormats[] = {
        {native_format_t::bf16, "bf16", data_type_t::bf16,
                {16, VNNI_PAIR * 2, VNNI_PAIR, 1, native_encode_t::copy, false},
                {{{~native_type_bit(data_type_t::bf16),
                         "bf16 weight with a non-bf16 source"}},
                        native_isa_t::avx512_bf16, false, 16, 0,
                        {kMaxNativeBf16K,
                                "a bf16 weight of K past the bf16 kernels' "
                                "int offsets"},
                        nullptr, kNativeNoLimit, nullptr},
                {true, true, native_prepack_t::bf16}},
        {native_format_t::s8, "s8", data_type_t::s8,
                {8, INT8_VNNI_GRP, INT8_VNNI_GRP, 1, native_encode_t::copy,
                        true},
                {{{~(native_type_bit(data_type_t::u8)
                           | native_type_bit(data_type_t::s8)),
                         "s8 weight with a source other than u8 / s8"}},
                        native_isa_t::avx512_vnni, true, 8,
                        native_grain_bit(native_grain_t::tensor)
                                | native_grain_bit(native_grain_t::channel)
                                | native_grain_bit(native_grain_t::group),
                        {kMaxSafeInt8AccumK,
                                "K exceeds the int32 accumulation bound"},
                        "per-group weight scales over groups of other than "
                        "whole K quads",
                        kNativeNoLimit, nullptr},
                {false, false, native_prepack_t::u8s8}},
        {native_format_t::s4, "s4", data_type_t::s4,
                {4, INT8_VNNI_GRP, 8, 2, native_encode_t::nibble_offset, false},
                {{{native_type_bit(data_type_t::u8),
                          "a u8 source with an s4 weight"},
                         {~native_type_bit(data_type_t::s8),
                                 "s4 weight with a source other than s8"}},
                        native_isa_t::avx512_vnni, true, 8,
                        native_grain_bit(native_grain_t::group), kNativeNoLimit,
                        "an s4 weight in groups of other than whole K quads",
                        {kMaxSafeInt4GroupK,
                                "an s4 weight group above the int32 "
                                "accumulation bound"},
                        "an s4 weight of K whose whole octets exceed int"},
                {true, false, native_prepack_t::none}},
        {native_format_t::f32, "f32", data_type_t::f32,
                {32, 4, 1, 1, native_encode_t::copy, false},
                {{{~native_type_bit(data_type_t::f32),
                         "f32 weight with a non-f32 source"}},
                        native_isa_t::base, false, 32, 0, kNativeNoLimit,
                        nullptr, kNativeNoLimit, nullptr},
                {false, false, native_prepack_t::none}},
        {native_format_t::f16, "f16", data_type_t::f16,
                {16, 2, 1, 1, native_encode_t::copy, false},
                {{{~native_type_bit(data_type_t::f16),
                         "f16 weight with a non-f16 source"}},
                        native_isa_t::base, false, 16, 0,
                        {kMaxNativeF16K,
                                "an f16 weight of K whose steps of 16 exceed "
                                "int"},
                        nullptr, kNativeNoLimit, nullptr},
                {true, true, native_prepack_t::none}},
};

constexpr const native_format_desc_t &native_format(native_format_t id) {
    return kNativeFormats[static_cast<int>(id)];
}

/// The format of weights of wei_dt; nullptr for a type the layout does not
/// hold.  Out of line, so that one translation unit holds the table.
const native_format_desc_t *native_format_of(data_type_t wei_dt);

/// The format of a weight the layout holds (native_layout_fits).
inline const native_format_desc_t &native_format(data_type_t wei_dt) {
    const native_format_desc_t *f = native_format_of(wei_dt);
    assert(f != nullptr);
    return *f;
}

/// The geometry of a weight the layout holds, by value: for the reads of
/// the layout's paths, compares against constants without a copy of the
/// table.
template <size_t I = 0>
constexpr native_geometry_t native_geometry(data_type_t wei_dt) {
    constexpr data_type_t dt = kNativeFormats[I].wei_dt;
    constexpr native_geometry_t g = kNativeFormats[I].geometry;
    if constexpr (I + 1 == std::size(kNativeFormats)) {
        assert(wei_dt == dt);
        return g;
    } else {
        return wei_dt == dt ? g : native_geometry<I + 1>(wei_dt);
    }
}

constexpr uint32_t native_quantized_types() {
    uint32_t types = 0;
    for (const native_format_desc_t &f : kNativeFormats)
        if (f.quantized()) types |= native_type_bit(f.wei_dt);
    return types;
}

/// The weight types of the quantized formats, a native_type_bit mask.
inline constexpr uint32_t kNativeQuantizedTypes = native_quantized_types();

/// The largest K whose K groups, padded to whole words, an int holds.
constexpr int native_max_layout_k(const native_format_desc_t &f) {
    return std::numeric_limits<int32_t>::max() / f.geometry.k_per_word
            * f.geometry.k_per_word;
}

struct native_family_desc_t {
    /// Its kernel kind, whose name the plan line and the schema print.
    native_kernel_kind_t id;
    const char *name;
    native_isa_t isa;
    /// K values its kernels take from one word of the layout, their K unit
    /// (the schema's k_unit): a pair of bf16, a quad of s8, an octet of s4
    /// nibbles (two VNNI steps of a quad), one f32 or f16 value.
    int k_step;
};

/// Indexed by native_kernel_kind_t.
inline constexpr native_family_desc_t kNativeFamilies[] = {
        {native_kernel_kind_t::bf16, "bf16", native_isa_t::avx512_bf16,
                VNNI_PAIR},
        {native_kernel_kind_t::int8, "int8", native_isa_t::avx512_vnni,
                INT8_VNNI_GRP},
        {native_kernel_kind_t::f32, "f32", native_isa_t::base, 1},
        {native_kernel_kind_t::int4, "int4", native_isa_t::avx512_vnni, 8},
        {native_kernel_kind_t::f16, "f16", native_isa_t::base, 1},
        {native_kernel_kind_t::f16_ph, "f16_ph", native_isa_t::avx512_fp16, 1},
};

constexpr const native_family_desc_t &native_family(native_kernel_kind_t id) {
    return kNativeFamilies[static_cast<int>(id)];
}

struct native_route_desc_t {
    native_format_t format;
    native_kernel_kind_t family;
    /// The INT8 scheme it serves, a native_int8_scheme_bit; 0 for a float
    /// route.
    uint8_t schemes;
};

inline constexpr native_route_desc_t kNativeRoutes[] = {
        {native_format_t::bf16, native_kernel_kind_t::bf16, 0},
        {native_format_t::s8, native_kernel_kind_t::int8,
                native_int8_scheme_bit(native_int8_scheme_t::per_channel)},
        {native_format_t::s8, native_kernel_kind_t::int8,
                native_int8_scheme_bit(native_int8_scheme_t::per_group)},
        {native_format_t::s4, native_kernel_kind_t::int4,
                native_int8_scheme_bit(native_int8_scheme_t::per_group)},
        // One group over K: weight scales {1, N}.
        {native_format_t::s4, native_kernel_kind_t::int4,
                native_int8_scheme_bit(native_int8_scheme_t::per_channel)},
        {native_format_t::f32, native_kernel_kind_t::f32, 0},
        {native_format_t::f16, native_kernel_kind_t::f16, 0},
        {native_format_t::f16, native_kernel_kind_t::f16_ph, 0},
};

/// Every refusal of f's contract has its reason, which the layout rule
/// returns (a refusal without one would take the call): each refused
/// source, a K past k, with group scales groups of other than whole K quads
/// and groups past group, and a K past native_max_layout_k unless k keeps K
/// below it.
constexpr bool native_refusals_have_reasons(const native_format_desc_t &f) {
    const native_contract_t &c = f.contract;
    for (const native_refusal_t &r : c.src_refusals)
        if (r.types != 0 && r.why == nullptr) return false;
    const bool groups
            = (c.scale_grains & native_grain_bit(native_grain_t::group)) != 0;
    return (c.k.max == kNativeNoLimit.max || c.k.why != nullptr)
            && (!groups
                    || (c.group_refusal != nullptr
                            && (c.group.max == kNativeNoLimit.max
                                    || c.group.why != nullptr)))
            && (c.layout_k_refusal != nullptr
                    || c.k.max <= native_max_layout_k(f));
}

/// The tables agree: each entry at its index; every refusal with its
/// reason; every format and family on a route; a route's family stepping K
/// by its format's words, on its format's instruction set or one that adds
/// to base, with an INT8 scheme exactly for a quantized format.
constexpr bool native_tables_agree() {
    for (size_t i = 0; i < std::size(kNativeFormats); ++i) {
        const native_format_desc_t &f = kNativeFormats[i];
        bool routed = false;
        for (const native_route_desc_t &r : kNativeRoutes)
            routed = routed || r.format == f.id;
        if (static_cast<size_t>(f.id) != i || !routed
                || !native_refusals_have_reasons(f))
            return false;
    }
    for (size_t i = 0; i < std::size(kNativeFamilies); ++i) {
        bool routed = false;
        for (const native_route_desc_t &r : kNativeRoutes)
            routed = routed || r.family == kNativeFamilies[i].id;
        if (static_cast<size_t>(kNativeFamilies[i].id) != i || !routed)
            return false;
    }
    for (const native_route_desc_t &r : kNativeRoutes) {
        const native_format_desc_t &f = native_format(r.format);
        const native_family_desc_t &fam = native_family(r.family);
        if (fam.k_step != f.geometry.k_per_word
                || (fam.isa != f.contract.isa
                        && f.contract.isa != native_isa_t::base)
                || (r.schemes != 0) != f.quantized())
            return false;
    }
    return true;
}
static_assert(native_tables_agree(), "the native format tables disagree");
static_assert(native_max_layout_k(native_format(native_format_t::s4))
                == kMaxNativeInt4K,
        "the s4 layout bound is the octets an int holds");

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_COMMON_NATIVE_FORMAT_HPP
