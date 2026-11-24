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

#include <array>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <limits>

#include "common/data_types.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/f16/native_f16_ph_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/int8/native_int4_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/int8/native_int8_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_int8_scheme.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_types.hpp"
#include "lowoha_operators/matmul/matmul_native/native_layout_policy.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

// The weights and kernels of the native BRGEMM, in three tables.  A format
// is how a weight is stored in the layout and which calls over it the
// layout rule takes; a family is how a kernel multiplies; a route is a
// format on a family for one INT8 scheme, as the kernel registry serves it.

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

/// How the kernels read the layout's words: none, as stored; byte_offset,
/// each byte as the unsigned operand of the u8 x s8 dot product, the weight
/// plus 128 (over an s8 source); nibble_offset, the two nibble planes of each
/// byte, the weights plus 8 as the pack stored them.
enum class native_decode_t : uint8_t { none, byte_offset, nibble_offset };

/// Where a format's weight scales lie: caller, the call's {G, N} buffer,
/// element (g, n) at g * N + n, which the tiles read in place.
enum class native_place_t : uint8_t { caller };

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
    /// Where its weight scales lie.
    native_place_t scale_at;
    /// The INT8 call checks, after the layout rule: why weight zero points
    /// are refused (a quantized format), why a source zero point is refused
    /// (nullptr: the call's quantization decides), and why weight scales
    /// without groups are refused (a format of group scales only).
    const char *wei_zp_refusal;
    const char *src_zp_refusal;
    const char *ungrouped_refusal;
    /// The types its weight scales take, a native_scale_bit mask; 0 for a
    /// weight without scales.
    uint8_t scale_types;
    /// A transposed source the call quantizes is quantized as given, its
    /// quantized rows still transposed, as matmul's other route of the
    /// format quantizes it; otherwise only one quantized to s8 with scales
    /// per row or per group narrower than f32 is (quantize_source).
    bool transposed_as_given;
};

/// The scale types of the weight scales of today's quantized formats.
inline constexpr uint8_t kNativeFloatScales
        = native_scale_bit(native_scale_t::f32)
        | native_scale_bit(native_scale_t::f16)
        | native_scale_bit(native_scale_t::bf16);

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
    native_decode_t decode;
    /// The offset decode adds to each weight is 1 << offset_bits, which the
    /// group shifts cancel; 0 with none.
    int offset_bits;
    native_contract_t contract;
    native_cache_rules_t cache;

    constexpr bool quantized() const { return contract.scale_grains != 0; }
};

/// Indexed by native_format_t.
inline constexpr native_format_desc_t kNativeFormats[] = {
        {native_format_t::bf16, "bf16", data_type_t::bf16,
                {16, VNNI_PAIR * 2, VNNI_PAIR, 1, native_encode_t::copy, false},
                native_decode_t::none, 0,
                {{{~native_type_bit(data_type_t::bf16),
                         "bf16 weight with a non-bf16 source"}},
                        native_isa_t::avx512_bf16, false, 16, 0,
                        {kMaxNativeBf16K,
                                "a bf16 weight of K past the bf16 kernels' "
                                "int offsets"},
                        nullptr, kNativeNoLimit, nullptr,
                        native_place_t::caller, nullptr, nullptr, nullptr, 0,
                        false},
                {true, true, native_prepack_t::bf16}},
        {native_format_t::s8, "s8", data_type_t::s8,
                {8, INT8_VNNI_GRP, INT8_VNNI_GRP, 1, native_encode_t::copy,
                        true},
                native_decode_t::byte_offset, 7,
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
                        kNativeNoLimit, nullptr, native_place_t::caller,
                        "weight zero points", nullptr, nullptr,
                        kNativeFloatScales, false},
                {false, false, native_prepack_t::u8s8}},
        {native_format_t::s4, "s4", data_type_t::s4,
                {4, INT8_VNNI_GRP, 8, 2, native_encode_t::nibble_offset, false},
                native_decode_t::nibble_offset, 3,
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
                        "an s4 weight of K whose whole octets exceed int",
                        native_place_t::caller, "an s4 weight with zero points",
                        "an s4 weight with a source zero point",
                        "an s4 weight with weight scales other than {G, N}",
                        kNativeFloatScales, true},
                {true, false, native_prepack_t::none}},
        {native_format_t::f32, "f32", data_type_t::f32,
                {32, 4, 1, 1, native_encode_t::copy, false},
                native_decode_t::none, 0,
                {{{~native_type_bit(data_type_t::f32),
                         "f32 weight with a non-f32 source"}},
                        native_isa_t::base, false, 32, 0, kNativeNoLimit,
                        nullptr, kNativeNoLimit, nullptr,
                        native_place_t::caller, nullptr, nullptr, nullptr, 0,
                        false},
                {false, false, native_prepack_t::none}},
        {native_format_t::f16, "f16", data_type_t::f16,
                {16, 2, 1, 1, native_encode_t::copy, false},
                native_decode_t::none, 0,
                {{{~native_type_bit(data_type_t::f16),
                         "f16 weight with a non-f16 source"}},
                        native_isa_t::base, false, 16, 0,
                        {kMaxNativeF16K,
                                "an f16 weight of K whose steps of 16 exceed "
                                "int"},
                        nullptr, kNativeNoLimit, nullptr,
                        native_place_t::caller, nullptr, nullptr, nullptr, 0,
                        false},
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

/// Whether a weight of format f takes weight scales the call gives as dt:
/// a scale type of its contract's (native_contract_t::scale_types).
constexpr bool native_takes_scales(
        const native_format_desc_t &f, data_type_t dt) {
    return (f.contract.scale_types & native_scale_bit(native_scale_of(dt)))
            != 0;
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

constexpr uint32_t native_grouped_types() {
    uint32_t types = 0;
    for (const native_format_desc_t &f : kNativeFormats)
        if (f.contract.scale_grains == native_grain_bit(native_grain_t::group))
            types |= native_type_bit(f.wei_dt);
    return types;
}

/// The weight types of the formats whose weight scales are per group only
/// ({G, N}, G >= 1, one group over K included), a native_type_bit mask.
inline constexpr uint32_t kNativeGroupedTypes = native_grouped_types();

/// The largest K whose K groups, padded to whole words, an int holds.
constexpr int native_max_layout_k(const native_format_desc_t &f) {
    return std::numeric_limits<int32_t>::max() / f.geometry.k_per_word
            * f.geometry.k_per_word;
}

/// The families: int4: the group kernels of an s8 source and an s4 weight;
/// f16: the kernels of an f16 weight over a source widened to f32; f16_ph:
/// the kernels of an f16 weight and an f16 source with AVX512-FP16
/// multiply-adds, which accumulate in f16 over blocks of
/// kNativeF16AccumBlockK steps and add the blocks in f32.
enum class native_kernel_kind_t : uint8_t {
    bf16,
    int8,
    f32,
    int4,
    f16,
    f16_ph
};

struct native_family_desc_t {
    /// Its kernel kind, whose name the plan line and the schema print.
    native_kernel_kind_t id;
    const char *name;
    native_isa_t isa;
    /// K values its kernels take from one word of the layout, their K unit
    /// (the schema's k_unit): a pair of bf16, a quad of s8, an octet of s4
    /// nibbles (two VNNI steps of a quad), one f32 or f16 value.
    int k_step;
    /// Bits of a source element its kernels multiply: the widened f32 of
    /// the F16 kernels, the quantized source of the INT8 and INT4 ones.
    int operand_bits;
    /// Columns of one multiply-add vector.
    int vector_cols;
    /// Columns of its widest kernel call, a slab, which a wider tile runs
    /// slab by slab: two vectors.
    int slab_cols;
    /// K steps of one block its kernels accumulate on its own, from the
    /// start of each call's K, so that a K pass of whole blocks keeps every
    /// block at the same K whatever the plan; 0 for none.
    int k_block;
};

/// Indexed by native_kernel_kind_t.
inline constexpr native_family_desc_t kNativeFamilies[] = {
        {native_kernel_kind_t::bf16, "bf16", native_isa_t::avx512_bf16,
                VNNI_PAIR, 16, 16, 32, 0},
        {native_kernel_kind_t::int8, "int8", native_isa_t::avx512_vnni,
                INT8_VNNI_GRP, 8, 16, 32, 0},
        {native_kernel_kind_t::f32, "f32", native_isa_t::base, 1, 32, 16, 32,
                0},
        {native_kernel_kind_t::int4, "int4", native_isa_t::avx512_vnni, 8, 8,
                16, 32, 0},
        {native_kernel_kind_t::f16, "f16", native_isa_t::base, 1, 32, 16, 32,
                0},
        {native_kernel_kind_t::f16_ph, "f16_ph", native_isa_t::avx512_fp16, 1,
                16, 32, 64, kNativeF16AccumBlockK},
};

constexpr const native_family_desc_t &native_family(native_kernel_kind_t id) {
    return kNativeFamilies[static_cast<int>(id)];
}

/// The routes: s4_one_group is an s4 weight of one group over K, whose
/// weight scales are {1, N}; f16 the widened source.
enum class native_route_t : uint8_t {
    bf16,
    s8_channel,
    s8_group,
    s4_group,
    s4_one_group,
    f32,
    f16,
    f16_ph
};

/// How the planner plans a route: rows, one kernel call per row call and K
/// pass (the float routes); channel, the INT8 tile of per-channel scales;
/// group, the INT8 tile of per-group scales, a batch of groups per call.
enum class native_route_class_t : uint8_t { rows, channel, group };

struct native_route_desc_t {
    native_route_t id;
    native_format_t format;
    native_kernel_kind_t family;
    /// The INT8 scheme it serves, a native_int8_scheme_bit; 0 for a float
    /// route.
    uint8_t schemes;
    native_route_class_t cls;
    /// The cost model prices its plans.
    bool priced;
    /// Rows of its kernel calls by width, 16, 32 and 64 columns (the
    /// kernels' registers: accumulators, weight vectors, broadcasts); 0 past
    /// its family's slab.  The kernel registry probes every row count up to
    /// them.
    uint8_t rows[3];
    /// The name of its GEMV kernels, which run its M = 1 calls, in the plan
    /// line (gemv=).
    const char *gemv;
};

/// Indexed by native_route_t, in the order the registry probes and lists
/// its kernels.
inline constexpr native_route_desc_t kNativeRoutes[] = {
        {native_route_t::bf16, native_format_t::bf16,
                native_kernel_kind_t::bf16, 0, native_route_class_t::rows, true,
                {6, 12, 0}, "bf16"},
        {native_route_t::s8_channel, native_format_t::s8,
                native_kernel_kind_t::int8,
                native_int8_scheme_bit(native_int8_scheme_t::per_channel),
                native_route_class_t::channel, true, {12, 12, 0}, "s8"},
        {native_route_t::s8_group, native_format_t::s8,
                native_kernel_kind_t::int8,
                native_int8_scheme_bit(native_int8_scheme_t::per_group),
                native_route_class_t::group, false, {12, 6, 0}, "s8_group"},
        {native_route_t::s4_group, native_format_t::s4,
                native_kernel_kind_t::int4,
                native_int8_scheme_bit(native_int8_scheme_t::per_group),
                native_route_class_t::group, true, {12, 6, 0}, "s4"},
        // One group over K: the kernels of {1, N} weight scales, in the
        // group tile of the format's per-group calls.
        {native_route_t::s4_one_group, native_format_t::s4,
                native_kernel_kind_t::int4,
                native_int8_scheme_bit(native_int8_scheme_t::per_channel),
                native_route_class_t::group, true, {12, 12, 0}, "s4"},
        {native_route_t::f32, native_format_t::f32, native_kernel_kind_t::f32,
                0, native_route_class_t::rows, false, {12, 12, 0}, "f32"},
        {native_route_t::f16, native_format_t::f16, native_kernel_kind_t::f16,
                0, native_route_class_t::rows, true, {12, 12, 0}, "f16"},
        {native_route_t::f16_ph, native_format_t::f16,
                native_kernel_kind_t::f16_ph, 0, native_route_class_t::rows,
                true, {12, 12, 12}, "f16_ph"},
};

constexpr const native_route_desc_t &native_route(native_route_t id) {
    return kNativeRoutes[static_cast<int>(id)];
}

/// Rows of route r's kernel calls of nv vectors (1, 2 or 4); 0 for any
/// other nv.
constexpr int native_route_rows(const native_route_desc_t &r, int nv) {
    return nv == 1 ? r.rows[0]
                   : (nv == 2 ? r.rows[1] : (nv == 4 ? r.rows[2] : 0));
}

/// Whether a route of format f runs on the kernels of family kind.
constexpr bool native_family_takes(
        native_kernel_kind_t kind, native_format_t f) {
    for (const native_route_desc_t &r : kNativeRoutes)
        if (r.format == f && r.family == kind) return true;
    return false;
}

template <native_kernel_kind_t Kind>
constexpr bool native_family_takes(native_format_t f) {
    return native_family_takes(Kind, f);
}

template <native_kernel_kind_t Kind>
constexpr size_t native_family_format_count() {
    size_t n = 0;
    for (const native_format_desc_t &f : kNativeFormats)
        n += native_family_takes<Kind>(f.id) ? 1 : 0;
    return n;
}

/// The formats of the routes on family Kind's kernels, in format order: the
/// formats its kernels are instantiated for.
template <native_kernel_kind_t Kind>
constexpr std::array<native_format_t, native_family_format_count<Kind>()>
native_family_formats() {
    std::array<native_format_t, native_family_format_count<Kind>()> out {};
    size_t n = 0;
    for (const native_format_desc_t &f : kNativeFormats)
        if (native_family_takes<Kind>(f.id)) out[n++] = f.id;
    return out;
}

/// Whether route runs on family Kind's kernels: a bit test against a mask
/// the table gives at compile time.
template <native_kernel_kind_t Kind>
constexpr bool native_on_family(native_route_t route) {
    constexpr uint32_t routes = [] {
        uint32_t mask = 0;
        for (const native_route_desc_t &r : kNativeRoutes)
            if (r.family == Kind) mask |= 1u << static_cast<unsigned>(r.id);
        return mask;
    }();
    return ((routes >> static_cast<unsigned>(route)) & 1u) != 0;
}

/// Whether the routes of family Kind are of the rows class: its tiles take
/// one kernel per row count (native_tile_kernel), which the kernel registry
/// holds.
template <native_kernel_kind_t Kind>
constexpr bool native_rows_family() {
    for (const native_route_desc_t &r : kNativeRoutes)
        if (r.family == Kind) return r.cls == native_route_class_t::rows;
    return false;
}

/// Whether route is of class Cls: a bit test against a mask the table gives
/// at compile time.
template <native_route_class_t Cls>
constexpr bool native_of_class(native_route_t route) {
    constexpr uint32_t routes = [] {
        uint32_t mask = 0;
        for (const native_route_desc_t &r : kNativeRoutes)
            if (r.cls == Cls) mask |= 1u << static_cast<unsigned>(r.id);
        return mask;
    }();
    return ((routes >> static_cast<unsigned>(route)) & 1u) != 0;
}

/// The INT8 scheme route r's kernels serve; per_channel for a float route.
constexpr native_int8_scheme_t native_route_scheme(
        const native_route_desc_t &r) {
    return r.schemes == native_int8_scheme_bit(native_int8_scheme_t::per_group)
            ? native_int8_scheme_t::per_group
            : native_int8_scheme_t::per_channel;
}

/// The route of quantized format f whose kernels serve scheme.
constexpr native_route_t native_int8_route(
        native_format_t f, native_int8_scheme_t scheme) {
    for (const native_route_desc_t &r : kNativeRoutes)
        if (r.format == f && (r.schemes & native_int8_scheme_bit(scheme)) != 0)
            return r.id;
    assert(!"a quantized format has a route of each INT8 scheme");
    return kNativeRoutes[0].id;
}

/// Whether the weight scales of every quantized format lie where at says.
constexpr bool native_scales_at(native_place_t at) {
    for (const native_format_desc_t &f : kNativeFormats)
        if (f.quantized() && f.contract.scale_at != at) return false;
    return true;
}

/// Every refusal of f's contract has its reason, which the layout rule and
/// the call checks return (a refusal without one would take the call): each
/// refused source, a K past k, with group scales groups of other than whole
/// K quads and groups past group, and a K past native_max_layout_k unless k
/// keeps K below it; weight zero points of a quantized format, and weight
/// scales without groups of a format of group scales only.
constexpr bool native_refusals_have_reasons(const native_format_desc_t &f) {
    const native_contract_t &c = f.contract;
    for (const native_refusal_t &r : c.src_refusals)
        if (r.types != 0 && r.why == nullptr) return false;
    const bool groups
            = (c.scale_grains & native_grain_bit(native_grain_t::group)) != 0;
    const bool group_only
            = c.scale_grains == native_grain_bit(native_grain_t::group);
    return (c.k.max == kNativeNoLimit.max || c.k.why != nullptr)
            && (!groups
                    || (c.group_refusal != nullptr
                            && (c.group.max == kNativeNoLimit.max
                                    || c.group.why != nullptr)))
            && (c.layout_k_refusal != nullptr
                    || c.k.max <= native_max_layout_k(f))
            && (!f.quantized() || c.wei_zp_refusal != nullptr)
            && (!group_only || c.ungrouped_refusal != nullptr);
}

/// The tables agree: each entry at its index; every refusal with its
/// reason; a format's kernels decoding what its pack wrote (the nibble
/// planes of nibbles, an offset exactly for a quantized format), with
/// scale types exactly for a quantized format, none of them none; every
/// format and family on a
/// route; a family's slab two vectors; a route's family stepping K by its
/// format's words, on its format's instruction set or one that adds to
/// base, with an INT8 scheme exactly for a quantized format and of the rows
/// class exactly for a float one, and rows at the widths up to its family's
/// slab only.
constexpr bool native_tables_agree() {
    for (size_t i = 0; i < std::size(kNativeFormats); ++i) {
        const native_format_desc_t &f = kNativeFormats[i];
        bool routed = false;
        for (const native_route_desc_t &r : kNativeRoutes)
            routed = routed || r.format == f.id;
        const bool decodes = f.decode != native_decode_t::none;
        uint8_t schemes = 0;
        for (const native_route_desc_t &r : kNativeRoutes)
            if (r.format == f.id) schemes |= r.schemes;
        if (static_cast<size_t>(f.id) != i || !routed
                || (f.quantized() && schemes != kNativeInt8AllSchemes)
                || !native_refusals_have_reasons(f)
                || (f.decode == native_decode_t::nibble_offset)
                        != (f.geometry.planes == 2)
                || decodes != f.quantized() || decodes != (f.offset_bits > 0)
                || (f.contract.scale_types != 0) != f.quantized()
                || (f.contract.scale_types
                           & native_scale_bit(native_scale_t::none))
                        != 0)
            return false;
    }
    for (size_t i = 0; i < std::size(kNativeFamilies); ++i) {
        bool routed = false;
        for (const native_route_desc_t &r : kNativeRoutes)
            routed = routed || r.family == kNativeFamilies[i].id;
        if (static_cast<size_t>(kNativeFamilies[i].id) != i || !routed
                || kNativeFamilies[i].slab_cols
                        != 2 * kNativeFamilies[i].vector_cols)
            return false;
    }
    for (size_t i = 0; i < std::size(kNativeRoutes); ++i) {
        const native_route_desc_t &r = kNativeRoutes[i];
        const native_format_desc_t &f = native_format(r.format);
        const native_family_desc_t &fam = native_family(r.family);
        if (static_cast<size_t>(r.id) != i
                || fam.k_step != f.geometry.k_per_word
                || (fam.isa != f.contract.isa
                        && f.contract.isa != native_isa_t::base)
                || (r.schemes != 0) != f.quantized()
                || (r.cls == native_route_class_t::rows) != (r.schemes == 0))
            return false;
        for (int w = 0; w < 3; ++w)
            if ((r.rows[w] > 0) != ((16 << w) <= fam.slab_cols)) return false;
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
