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

#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/int8/native_int8_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

namespace {

int width_index(int nr) {
    switch (nr) {
        case 16: return 0;
        case 32: return 1;
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
        bool bf16_isa, bool int8_isa) {
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
        // The per-channel tile runs the kernel over its K passes, the
        // per-group tile as the first pass over each group.
        c.int8_schemes = kNativeInt8AllSchemes;
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

bool native_kernel_registry_t::covers_int8(
        int mr, int nr, native_int8_scheme_t scheme) const noexcept {
    for (const native_kernel_caps_t &c : all_) {
        if (c.kind == native_kernel_kind_t::int8 && c.nr == nr
                && c.serves(scheme) && mr >= 1 && mr <= c.max_mr)
            return true;
    }
    return false;
}

const bf16_brgemm_fn_t *native_kernel_registry_t::bf16_table(
        int nr) const noexcept {
    if (caps(native_kernel_kind_t::bf16, nr) == nullptr) return nullptr;
    return bf16_[width_index(nr)];
}

const native_kernel_registry_t &native_kernel_registry() {
    // Never destroyed: calls may still plan while static objects are
    // destroyed.
    static const native_kernel_registry_t *const registry
            = new native_kernel_registry_t(
                    detect_uarch().avx512bf16, detect_uarch().avx512vnni);
    return *registry;
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
