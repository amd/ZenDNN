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

#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/native_kernel_registry.hpp"

#include <cassert>
#include <cctype>
#include <cstdlib>
#include <iterator>
#include <string>

#include "common/zendnnl_global.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/int8/native_int4_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/int8/native_int8_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::error_handling::apilog_warning;

namespace {

/// Whether ZENDNNL_MATMUL_NATIVE_F16_ACCUM lets f16 calls accumulate in f16
/// blocks: unset, empty or f16 (any case) yes, f32 no; any other value keeps
/// the default with one warning.
bool f16_accum_in_f16() {
    const char *v = std::getenv("ZENDNNL_MATMUL_NATIVE_F16_ACCUM");
    if (v == nullptr || *v == '\0') return true;
    std::string s(v);
    for (char &ch : s)
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    if (s == "f16") return true;
    if (s == "f32") return false;
    apilog_warning("ZENDNNL_MATMUL_NATIVE_F16_ACCUM=", v,
            " is neither f16 nor f32; the default, f16, applies");
    return true;
}

int width_index(int nr) {
    switch (nr) {
        case 16: return 0;
        case 32: return 1;
        case 64: return 2;
        default: return -1;
    }
}

/// The CPU runs kernels of isa; of base, it has AVX-512 F.
bool runs(const UarchParams &u, native_isa_t isa) {
    switch (isa) {
        case native_isa_t::base: return u.avx512f;
        case native_isa_t::avx512_bf16: return u.avx512bf16;
        case native_isa_t::avx512_vnni: return u.avx512vnni;
        case native_isa_t::avx512_fp16: return u.avx512fp16;
    }
    return false;
}

/// The kernels of a family of the INT8 tile as the registry probes them:
/// whether every kernel a call of scheme over a weight of format takes at mr
/// rows and nv vectors exists.
template <native_kernel_kind_t Kind>
struct native_call_kernels;

template <>
struct native_call_kernels<native_kernel_kind_t::int8> {
    static bool serve(native_format_t format, native_int8_scheme_t scheme,
            int mr, int nv) {
        return native_int8_kernels_serve(format, scheme, mr, nv);
    }
};

template <>
struct native_call_kernels<native_kernel_kind_t::int4> {
    static bool serve(native_format_t format, native_int8_scheme_t scheme,
            int mr, int nv) {
        return native_int4_kernels_serve(format, scheme, mr, nv);
    }
};

} // namespace

template <native_kernel_kind_t Kind, typename Registry>
auto &native_kernel_registry_t::tables(Registry &registry) noexcept {
    if constexpr (Kind == native_kernel_kind_t::bf16) {
        return registry.bf16_;
    } else if constexpr (Kind == native_kernel_kind_t::f32) {
        return registry.f32_;
    } else if constexpr (Kind == native_kernel_kind_t::f16) {
        return registry.f16_;
    } else {
        static_assert(Kind == native_kernel_kind_t::f16_ph,
                "a family of one kernel per row count");
        return registry.f16_ph_;
    }
}

template <native_kernel_kind_t Kind>
bool native_kernel_registry_t::probe_family(
        const native_route_desc_t &route, int w, int mr) {
    if constexpr (native_rows_family<Kind>()) {
        auto &table = tables<Kind>(*this);
        table[0][w][mr] = native_tile_kernel<Kind>::select(mr, 1 << w);
        return table[0][w][mr] != nullptr;
    } else {
        return native_call_kernels<Kind>::serve(
                route.format, native_route_scheme(route), mr, 1 << w);
    }
}

template <size_t... I>
bool native_kernel_registry_t::probe_of(const native_route_desc_t &route, int w,
        int mr, std::index_sequence<I...>) {
    bool has = false;
    (void)((route.family == kNativeFamilies[I].id
                   && (has = probe_family<kNativeFamilies[I].id>(route, w, mr),
                           true))
            || ...);
    return has;
}

bool native_kernel_registry_t::probe(
        const native_route_desc_t &route, int w, int mr) {
    return probe_of(route, w, mr,
            std::make_index_sequence<std::size(kNativeFamilies)> {});
}

native_kernel_registry_t::native_kernel_registry_t(uint32_t families) {
    for (const native_route_desc_t &route : kNativeRoutes) {
        if (((families >> static_cast<int>(route.family)) & 1u) == 0) continue;
        const int slab = native_slab_cols(route.family);
        for (int w = 0; 16 << w <= slab; ++w) {
            const int rows = native_route_rows(route, 1 << w);
            assert(rows <= kMaxRows);
            bool served = rows > 0;
            for (int mr = 1; served && mr <= rows; ++mr)
                served = probe(route, w, mr);
            if (!served) continue;
            native_kernel_caps_t c;
            c.route = route.id;
            c.kind = route.family;
            c.nr = 16 << w;
            c.max_mr = rows;
            c.int8_schemes = route.schemes;
            all_.push_back(c);
            // Wider tiles run the kernels of the slab width slab by slab.
            if (c.nr != slab) continue;
            for (c.nr = 2 * slab; c.nr <= kNativeMaxTileCols; c.nr *= 2)
                all_.push_back(c);
        }
    }
}

const native_kernel_caps_t *native_kernel_registry_t::caps(
        native_kernel_kind_t kind, int nr) const noexcept {
    for (const native_kernel_caps_t &c : all_) {
        if (c.kind == kind && c.nr == nr) return &c;
    }
    return nullptr;
}

bool native_kernel_registry_t::covers(
        native_kernel_kind_t kind, int mr, int nr) const noexcept {
    const native_kernel_caps_t *c = caps(kind, nr);
    return c != nullptr && mr >= 1 && mr <= c->max_mr;
}

const native_kernel_caps_t *native_kernel_registry_t::caps(
        native_kernel_kind_t kind, int nr,
        native_int8_scheme_t scheme) const noexcept {
    for (const native_kernel_caps_t &c : all_) {
        if (c.kind == kind && c.nr == nr && c.serves(scheme)) return &c;
    }
    return nullptr;
}

bool native_kernel_registry_t::covers(native_kernel_kind_t kind, int mr, int nr,
        native_int8_scheme_t scheme) const noexcept {
    const native_kernel_caps_t *c = caps(kind, nr, scheme);
    return c != nullptr && mr >= 1 && mr <= c->max_mr;
}

template <native_kernel_kind_t Kind>
const typename native_tile_kernel<Kind>::fn_t *native_kernel_registry_t::table(
        int nr) const noexcept {
    if (caps(Kind, nr) == nullptr) return nullptr;
    return tables<Kind>(*this)[0][width_index(native_tile_slab_cols(Kind, nr))];
}

template const native_tile_kernel<native_kernel_kind_t::bf16>::fn_t *
native_kernel_registry_t::table<native_kernel_kind_t::bf16>(int) const noexcept;
template const native_tile_kernel<native_kernel_kind_t::f32>::fn_t *
native_kernel_registry_t::table<native_kernel_kind_t::f32>(int) const noexcept;
template const native_tile_kernel<native_kernel_kind_t::f16>::fn_t *
native_kernel_registry_t::table<native_kernel_kind_t::f16>(int) const noexcept;
template const native_tile_kernel<native_kernel_kind_t::f16_ph>::fn_t *
native_kernel_registry_t::table<native_kernel_kind_t::f16_ph>(
        int) const noexcept;

bool native_kernel_registry_t::has(native_kernel_kind_t kind) const noexcept {
    for (const native_kernel_caps_t &c : all_) {
        if (c.kind == kind) return true;
    }
    return false;
}

native_kernel_kind_t native_kernel_registry_t::f16_kind() const noexcept {
    return has(native_kernel_kind_t::f16_ph) ? native_kernel_kind_t::f16_ph
                                             : native_kernel_kind_t::f16;
}

const native_kernel_registry_t &native_kernel_registry() {
    // Never destroyed: calls may still plan while static objects are
    // destroyed.
    static const native_kernel_registry_t *const registry = [] {
        const UarchParams &u = detect_uarch();
        const bool f16_accum = f16_accum_in_f16();
        uint32_t families = 0;
        for (const native_family_desc_t &f : kNativeFamilies)
            if (runs(u, f.isa)
                    && (f.id != native_kernel_kind_t::f16_ph || f16_accum))
                families |= 1u << static_cast<int>(f.id);
        return new native_kernel_registry_t(families);
    }();
    return *registry;
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
