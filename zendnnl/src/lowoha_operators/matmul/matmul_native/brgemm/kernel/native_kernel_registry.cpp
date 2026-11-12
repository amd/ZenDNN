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

#include <cctype>
#include <cstdlib>
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

/// Largest m such that every row count 1 .. m is in mask.
int full_rows(uint32_t mask) {
    int m = 0;
    while (m < 32 && ((mask >> m) & 1u) != 0)
        ++m;
    return m;
}

} // namespace

native_kernel_registry_t::native_kernel_registry_t(
        bool bf16_isa, bool int8_isa, bool f32_isa, bool f16_ph_isa) {
    for (int w = 0; bf16_isa && w < kWidths; ++w) {
        native_kernel_caps_t c;
        c.kind = native_kernel_kind_t::bf16;
        c.nr = 16 << w;
        uint32_t mask = 0;
        for (int mr = 1; mr <= kMaxRows; ++mr) {
            bf16_[w][mr] = select_bf16_brgemm_tall_kernel(mr, c.nr);
            if (bf16_[w][mr] != nullptr) mask |= 1u << (mr - 1);
        }
        c.max_mr = full_rows(mask);
        if (c.max_mr > 0) all_.push_back(c);
    }
    // Wider tiles run the slab kernels slab by slab.
    const native_kernel_caps_t *slab
            = caps(native_kernel_kind_t::bf16, kNativeBf16SlabCols);
    if (slab != nullptr) {
        const native_kernel_caps_t c32 = *slab;
        for (int nr = 2 * kNativeBf16SlabCols; nr <= kNativeBf16MaxTileCols;
                nr *= 2) {
            native_kernel_caps_t c = c32;
            c.nr = nr;
            all_.push_back(c);
        }
    }
    for (int w = 0; int8_isa && w < kWidths; ++w) {
        native_kernel_caps_t c;
        c.kind = native_kernel_kind_t::int8;
        c.nr = 16 << w;
        const int nv = c.nr / 16;
        // A call takes the signed or the unsigned kernel by its source.
        uint32_t mask = 0;
        for (int mr = 1; mr <= kMaxRows; ++mr) {
            if (select_native_int8_ukernel(mr, nv, false) != nullptr
                    && select_native_int8_ukernel(mr, nv, true) != nullptr)
                mask |= 1u << (mr - 1);
        }
        c.max_mr = full_rows(mask);
        c.int8_schemes
                = native_int8_scheme_bit(native_int8_scheme_t::per_channel);
        if (c.max_mr > 0) all_.push_back(c);
    }
    for (int w = 0; int8_isa && w < kWidths; ++w) {
        native_kernel_caps_t c;
        c.kind = native_kernel_kind_t::int8;
        c.nr = 16 << w;
        const int nv = c.nr / 16;
        // A call takes the s8 or the u8 kernel by its source.
        uint32_t mask = 0;
        for (int mr = 1; mr <= kMaxRows; ++mr) {
            if (select_native_int8_group_ukernel(mr, nv, false) != nullptr
                    && select_native_int8_group_ukernel(mr, nv, true)
                            != nullptr)
                mask |= 1u << (mr - 1);
        }
        c.max_mr = full_rows(mask);
        c.int8_schemes
                = native_int8_scheme_bit(native_int8_scheme_t::per_group);
        if (c.max_mr > 0) all_.push_back(c);
    }
    for (int w = 0; int8_isa && w < kWidths; ++w) {
        native_kernel_caps_t c;
        c.kind = native_kernel_kind_t::int4;
        c.nr = 16 << w;
        const int nv = c.nr / 16;
        // A call takes the fused order (over a source scale per row or per
        // group) or the reference order by its scales.
        uint32_t mask = 0;
        for (int mr = 1; mr <= kMaxRows; ++mr) {
            if (select_native_int4_group_ukernel(
                        mr, nv, native_int8_order_t::fused)
                            != nullptr
                    && select_native_int4_group_ukernel(
                               mr, nv, native_int8_order_t::fused, true)
                            != nullptr
                    && select_native_int4_group_ukernel(
                               mr, nv, native_int8_order_t::reference)
                            != nullptr)
                mask |= 1u << (mr - 1);
        }
        c.max_mr = full_rows(mask);
        c.int8_schemes
                = native_int8_scheme_bit(native_int8_scheme_t::per_group);
        if (c.max_mr > 0) all_.push_back(c);
    }
    for (int w = 0; f32_isa && w < kWidths; ++w) {
        native_kernel_caps_t c;
        c.kind = native_kernel_kind_t::f32;
        c.nr = 16 << w;
        uint32_t mask = 0;
        for (int mr = 1; mr <= kMaxRows; ++mr) {
            f32_[w][mr] = select_native_fp32_ukernel(mr, c.nr / 16);
            if (f32_[w][mr] != nullptr) mask |= 1u << (mr - 1);
        }
        c.max_mr = full_rows(mask);
        if (c.max_mr > 0) all_.push_back(c);
    }
    for (int w = 0; f32_isa && w < kWidths; ++w) {
        native_kernel_caps_t c;
        c.kind = native_kernel_kind_t::f16;
        c.nr = 16 << w;
        uint32_t mask = 0;
        for (int mr = 1; mr <= kMaxRows; ++mr) {
            f16_[w][mr] = select_native_f16_ukernel(mr, c.nr / 16);
            if (f16_[w][mr] != nullptr) mask |= 1u << (mr - 1);
        }
        c.max_mr = full_rows(mask);
        if (c.max_mr > 0) all_.push_back(c);
    }
    for (int w = 0; f16_ph_isa && w < kF16PhWidths; ++w) {
        native_kernel_caps_t c;
        c.kind = native_kernel_kind_t::f16_ph;
        c.nr = 16 << w;
        uint32_t mask = 0;
        for (int mr = 1; mr <= kMaxRows; ++mr) {
            f16_ph_[w][mr] = select_native_f16_ph_ukernel(mr, c.nr / 16);
            if (f16_ph_[w][mr] != nullptr) mask |= 1u << (mr - 1);
        }
        c.max_mr = full_rows(mask);
        if (c.max_mr > 0) all_.push_back(c);
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

const native_kernel_caps_t *native_kernel_registry_t::int8_caps(
        int nr, native_int8_scheme_t scheme) const noexcept {
    for (const native_kernel_caps_t &c : all_) {
        if (c.kind == native_kernel_kind_t::int8 && c.nr == nr
                && c.serves(scheme))
            return &c;
    }
    return nullptr;
}

bool native_kernel_registry_t::covers_int8(
        int mr, int nr, native_int8_scheme_t scheme) const noexcept {
    const native_kernel_caps_t *c = int8_caps(nr, scheme);
    return c != nullptr && mr >= 1 && mr <= c->max_mr;
}

const bf16_brgemm_fn_t *native_kernel_registry_t::bf16_table(
        int nr) const noexcept {
    if (caps(native_kernel_kind_t::bf16, nr) == nullptr) return nullptr;
    return bf16_[width_index(native_bf16_slab_cols(nr))];
}

const native_fp32_ukernel_fn_t *native_kernel_registry_t::f32_table(
        int nr) const noexcept {
    if (caps(native_kernel_kind_t::f32, nr) == nullptr) return nullptr;
    return f32_[width_index(nr)];
}

const native_f16_ukernel_fn_t *native_kernel_registry_t::f16_table(
        int nr) const noexcept {
    if (caps(native_kernel_kind_t::f16, nr) == nullptr) return nullptr;
    return f16_[width_index(nr)];
}

const native_f16_ph_ukernel_fn_t *native_kernel_registry_t::f16_ph_table(
        int nr) const noexcept {
    if (caps(native_kernel_kind_t::f16_ph, nr) == nullptr) return nullptr;
    return f16_ph_[width_index(nr)];
}

native_kernel_kind_t native_kernel_registry_t::f16_kind() const noexcept {
    for (const native_kernel_caps_t &c : all_) {
        if (c.kind == native_kernel_kind_t::f16_ph)
            return native_kernel_kind_t::f16_ph;
    }
    return native_kernel_kind_t::f16;
}

const native_kernel_registry_t &native_kernel_registry() {
    // Never destroyed: calls may still plan while static objects are
    // destroyed.
    static const native_kernel_registry_t *const registry = [] {
        const UarchParams &u = detect_uarch();
        const bool f16_accum = f16_accum_in_f16();
        return new native_kernel_registry_t(u.avx512bf16, u.avx512vnni,
                u.avx512f, u.avx512fp16 && f16_accum);
    }();
    return *registry;
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
