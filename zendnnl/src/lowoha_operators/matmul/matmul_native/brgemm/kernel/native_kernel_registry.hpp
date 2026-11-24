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

#ifndef MATMUL_NATIVE_BRGEMM_KERNEL_NATIVE_KERNEL_REGISTRY_HPP
#define MATMUL_NATIVE_BRGEMM_KERNEL_NATIVE_KERNEL_REGISTRY_HPP

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/bf16/bf16_brgemm_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/f16/native_f16_ph_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/f16/native_f16_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/fp32/native_fp32_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_format.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_int8_scheme.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

/// Columns of a slab of kind's kernels: their widest call, two vectors, which
/// a wider tile runs slab by slab (native_family_desc_t::slab_cols).
constexpr int native_slab_cols(native_kernel_kind_t kind) {
    return native_family(kind).slab_cols;
}

/// Widest tile of every kind: whole slabs, at most this many columns.
inline constexpr int kNativeMaxTileCols = 128;
static_assert(
        kNativeMaxTileCols % native_slab_cols(native_kernel_kind_t::bf16) == 0
                && kNativeMaxTileCols
                                % native_slab_cols(native_kernel_kind_t::f16_ph)
                        == 0,
        "the widest tile is whole slabs");

/// Columns of one kernel call of kind in a tile nr wide: the tile, or a slab
/// for a tile of several slabs.
constexpr int native_tile_slab_cols(native_kernel_kind_t kind, int nr) {
    return nr < native_slab_cols(kind) ? nr : native_slab_cols(kind);
}

/// The kernels the registry holds of a family whose tiles take one kernel
/// per row count: their type, and select(mr, nv), the kernel of mr rows
/// and nv vectors of 16 columns, or nullptr.
template <native_kernel_kind_t Kind>
struct native_tile_kernel;

template <>
struct native_tile_kernel<native_kernel_kind_t::bf16> {
    using fn_t = bf16_brgemm_fn_t;
    static fn_t select(int mr, int nv) {
        return select_bf16_brgemm_tall_kernel(mr, 16 * nv);
    }
};

template <>
struct native_tile_kernel<native_kernel_kind_t::f32> {
    using fn_t = native_fp32_ukernel_fn_t;
    static fn_t select(int mr, int nv) {
        return select_native_fp32_ukernel(mr, nv);
    }
};

template <>
struct native_tile_kernel<native_kernel_kind_t::f16> {
    using fn_t = native_f16_ukernel_fn_t;
    static fn_t select(int mr, int nv) {
        return select_native_f16_ukernel(mr, nv);
    }
};

template <>
struct native_tile_kernel<native_kernel_kind_t::f16_ph> {
    using fn_t = native_f16_ph_ukernel_fn_t;
    static fn_t select(int mr, int nv) {
        return select_native_f16_ph_ukernel(mr, nv);
    }
};

/// What the micro-kernels of one route offer at one tile width.
struct native_kernel_caps_t {
    native_route_t route = native_route_t::bf16;
    /// The route's family.
    native_kernel_kind_t kind = native_kernel_kind_t::bf16;
    /// 16 * nv, or a tile of several slabs (native_slab_cols) up to
    /// kNativeMaxTileCols, which has the rows and schemes of its slab
    int nr = 0;
    int max_mr = 0; ///< every row count 1 .. max_mr has a kernel
    /// The route's INT8 scheme, a mask of native_int8_scheme_bit: the
    /// per-channel kernels serve per_channel, the group kernels per_group (as
    /// do the INT4 group kernels; the INT4 kernels of one group over K, whose
    /// weight scales are {1, N}, serve per_channel).  validate_native_plan
    /// checks the call's scheme against the mask, so a kernel is never
    /// planned for the other.
    uint8_t int8_schemes = 0;

    bool serves(native_int8_scheme_t scheme) const noexcept {
        return (int8_schemes & native_int8_scheme_bit(scheme)) != 0;
    }
};

class native_kernel_registry_t {
public:
    /// nullptr when kind has no kernel at nr on this CPU; for INT8, the
    /// per-channel kernels; for INT4, the group kernels.
    const native_kernel_caps_t *caps(
            native_kernel_kind_t kind, int nr) const noexcept;
    /// The kernels of kind at nr that serve scheme (INT8 and INT4:
    /// per_channel, or per_group for the group kernels; the INT4 kernels of
    /// one group over K serve per_channel); nullptr when none.
    const native_kernel_caps_t *caps(native_kernel_kind_t kind, int nr,
            native_int8_scheme_t scheme) const noexcept;
    /// A kernel for every row count 1 .. mr at nr.
    bool covers(native_kernel_kind_t kind, int mr, int nr) const noexcept;
    /// A kernel of kind that serves scheme, for every row count 1 .. mr at
    /// nr.
    bool covers(native_kernel_kind_t kind, int mr, int nr,
            native_int8_scheme_t scheme) const noexcept;
    /// The kernels of Kind (native_tile_kernel) by row count (index 0
    /// unused, size 13) of the calls of a tile nr wide,
    /// native_tile_slab_cols(Kind, nr) columns each; nullptr when
    /// caps(Kind, nr) is nullptr.  F16_PH calls are 64 columns from nr 64.
    template <native_kernel_kind_t Kind>
    const typename native_tile_kernel<Kind>::fn_t *table(int nr) const noexcept;
    /// Whether the registry has kernels of family kind: the CPU has its
    /// instruction set (for F16_PH also f16 accumulation; see
    /// native_kernel_registry).
    bool has(native_kernel_kind_t kind) const noexcept;
    /// The kernels of an f16 call: f16_ph when the registry has them, else
    /// f16, as native_problem_t::route() takes them.
    native_kernel_kind_t f16_kind() const noexcept;
    /// Every entry, by route (kNativeRoutes), then by width.
    const std::vector<native_kernel_caps_t> &all() const noexcept {
        return all_;
    }

private:
    friend const native_kernel_registry_t &native_kernel_registry();
    /// families: the families to probe, bit 1 << native_kernel_kind_t.
    explicit native_kernel_registry_t(uint32_t families);
    /// Whether route has every kernel its calls take at mr rows and 16 << w
    /// columns: the probe of its family (kNativeFamilies), probe_family.
    bool probe(const native_route_desc_t &route, int w, int mr);
    template <size_t... I>
    bool probe_of(const native_route_desc_t &route, int w, int mr,
            std::index_sequence<I...>);
    /// A family of one kernel per row count (native_rows_family) writes its
    /// kernel into its table; the families of the INT8 tile answer for every
    /// source and order a call over the route's format and scheme takes.
    template <native_kernel_kind_t Kind>
    bool probe_family(const native_route_desc_t &route, int w, int mr);

    static constexpr int kMaxRows = 12;
    /// Kernel variants of a family; variant 0, the default, is the only one.
    static constexpr int kVariants = 1;
    static constexpr int kWidths = 3; ///< nr 16, 32, 64
    template <native_kernel_kind_t Kind>
    using table_t = typename native_tile_kernel<Kind>::fn_t[kVariants][kWidths]
                                                           [kMaxRows + 1];
    /// The table of Kind in registry.
    template <native_kernel_kind_t Kind, typename Registry>
    static auto &tables(Registry &registry) noexcept;

    std::vector<native_kernel_caps_t> all_;
    table_t<native_kernel_kind_t::bf16> bf16_ = {};
    table_t<native_kernel_kind_t::f32> f32_ = {};
    table_t<native_kernel_kind_t::f16> f16_ = {};
    table_t<native_kernel_kind_t::f16_ph> f16_ph_ = {};
};

/// Built once by probing each route (kNativeRoutes) at the widths up to its
/// family's slab (native_slab_cols), every row count up to the route's rows
/// at that width (native_route_desc_t::rows), by its family: the
/// native_tile_kernel selectors, native_int8_kernels_serve (both sources)
/// and native_int4_kernels_serve (every order), over the route's format.  A
/// route lacking a kernel of any of those rows has no entry at that width.
/// Tiles of several slabs, up to kNativeMaxTileCols columns, take the row
/// counts and schemes of the slab.  A family is probed only when the CPU has its ISA
/// (native_family_desc_t: AVX512-BF16, AVX512-VNNI for INT8 and INT4,
/// AVX-512 F for FP32 and F16, AVX512-FP16 for F16_PH); otherwise it has no
/// entries.  F16_PH is probed only while ZENDNNL_MATMUL_NATIVE_F16_ACCUM,
/// read here, is unset or f16; at f32 an f16 call takes the F16 kernels,
/// and another value keeps the default with one warning.
const native_kernel_registry_t &native_kernel_registry();

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_BRGEMM_KERNEL_NATIVE_KERNEL_REGISTRY_HPP
