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

/// The four-bit weight formats of the custom kernel.  A format is what the
/// packed nibbles mean (`W4CodeKind`) and where the per-group weight scales
/// live (`W4ScaleStore`).  Everything the shared code asks about a format is
/// answered here: the weight type that selects it, its log names, whether a
/// refusal falls back, whether its pack holds the scales, whether its
/// kernel needs the source rows' group sums, and its pack tail per group.
/// A new four-bit format adds its entry here, its pack tail in `pack.cpp`,
/// its microkernel file and its route.

#ifndef ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_W4_FORMAT_HPP
#define ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_W4_FORMAT_HPP

#include <cstddef>
#include <cstdint>

#include "common/data_types.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace custom_kernel {

/// What the packed nibbles mean.  Both kinds share the `[K/8][pack_nr][4]`
/// nibble slab.
enum class W4CodeKind : uint8_t {
    /// s4: uniform signed grid, stored biased by XOR 8; value is `nib - 8`.
    kS4 = 0,
    /// f4_e2m1, stored verbatim; value is `kE2M1ToInt8x2[nib]`, twice the
    /// code's value, so the scale the kernel applies is halved.
    kE2M1 = 1
};

/// Where a W4 pack's per-group weight scales live.
enum class W4ScaleStore : uint8_t {
    /// The caller's f32 / bf16 `{G, N}` buffer, read through `wei_scale` on
    /// every call; the o-block tail holds s4's int32 compensation rows.
    kExternal = 0,
    /// f8_e8m0 bytes copied into the o-block tail, one per (group, column),
    /// already halved for the doubled E2M1 decode, so the kernel rebuilds the
    /// f32 scale as `byte << 23` and streams one byte per group, not four.
    kE8M0InPack = 1
};

/// A W4 pack and microkernel format.  Each W4 kernel variant has exactly
/// one (`w4_format_for_variant`); a new four-bit format is a new pair here,
/// not a flag on an existing one.
struct W4Format {
    W4CodeKind code;
    W4ScaleStore scales;
};
inline constexpr bool operator==(W4Format a, W4Format b) {
    return a.code == b.code && a.scales == b.scales;
}
inline constexpr bool operator!=(W4Format a, W4Format b) {
    return !(a == b);
}

/// W4A8: s4 codes, external f32 / bf16 scales, compensation in the pack.
inline constexpr W4Format kW4FormatS4 {
        W4CodeKind::kS4, W4ScaleStore::kExternal};
/// MXFP4: f4_e2m1 codes, f8_e8m0 scales in the pack, no compensation.
inline constexpr W4Format kW4FormatMxfp4 {
        W4CodeKind::kE2M1, W4ScaleStore::kE8M0InPack};

/// One entry per format: the weight type that selects it, the names its
/// log lines use (`fmt=<name>`, then `<label>`), and whether a call the
/// custom kernel refuses fails rather than falling back.  The routes
/// enforce that last rule (`selection/mxfp4_route` for MXFP4); log lines
/// read it here.
struct W4FormatEntry {
    W4Format fmt;
    zendnnl::common::data_type_t wei;
    const char *name;
    const char *label;
    bool ck_only;
};

inline constexpr W4FormatEntry kW4Formats[] = {
        {kW4FormatS4, zendnnl::common::data_type_t::s4, "s4", "INT4", false},
        {kW4FormatMxfp4, zendnnl::common::data_type_t::f4_e2m1, "mxfp4",
                "MXFP4", true},
};

/// The format a weight type selects, or nullptr for a type the custom
/// kernel does not run as four-bit weights (u4 included).
inline const W4FormatEntry *w4_format_entry(zendnnl::common::data_type_t wei) {
    for (const W4FormatEntry &e : kW4Formats) {
        if (e.wei == wei) return &e;
    }
    return nullptr;
}

/// A format's entry, or nullptr for a pair no kernel implements.
inline const W4FormatEntry *w4_format_entry(W4Format fmt) {
    for (const W4FormatEntry &e : kW4Formats) {
        if (e.fmt == fmt) return &e;
    }
    return nullptr;
}

/// Whether the custom kernel runs `wei` as four-bit weights.
inline bool is_w4_weight(zendnnl::common::data_type_t wei) {
    return w4_format_entry(wei) != nullptr;
}

/// A format's `fmt=` log name and its label, "unknown" for a pair no
/// kernel implements.
inline const char *w4_format_name(W4Format fmt) {
    const W4FormatEntry *e = w4_format_entry(fmt);
    return e != nullptr ? e->name : "unknown";
}
inline const char *w4_format_label(W4Format fmt) {
    const W4FormatEntry *e = w4_format_entry(fmt);
    return e != nullptr ? e->label : "unknown";
}

/// Whether the pack holds the weight scales, so the kernel never reads the
/// caller's and the pack needs them at pack time.
inline constexpr bool w4_scales_in_pack(W4Format fmt) {
    return fmt.scales != W4ScaleStore::kExternal;
}

/// Whether the kernel seeds each group from its source rows' sums
/// (`s8_group_sums`): E2M1 enters VPDPBUSD as the unsigned operand, offset
/// by `kE2M1UnsignedBias`, and the seed removes the offset.
inline constexpr bool w4_needs_src_group_sums(W4Format fmt) {
    return fmt.code == W4CodeKind::kE2M1;
}

/// Bytes the pack stores per (group, column) after the nibble slab: s4's
/// int32 compensation, or one in-pack scale byte.
inline constexpr size_t w4_tail_bytes_per_group_col(W4Format fmt) {
    return w4_scales_in_pack(fmt)         ? 1
            : fmt.code == W4CodeKind::kS4 ? sizeof(int32_t)
                                          : 0;
}

} // namespace custom_kernel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_W4_FORMAT_HPP
