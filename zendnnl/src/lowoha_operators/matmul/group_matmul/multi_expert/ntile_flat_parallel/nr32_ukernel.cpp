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

#include "nr32_ukernel.hpp"

#include "common/zendnnl_global.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {
namespace nr32 {

namespace {

// Each family's selectors take every option; those a family has no variants
// for are left unread (s4's kernels sum by group whatever the weight's
// groups).

using narrow_fn = ukernel_fn (*)(
        int mr, epilogue ep, bool src_u8, bool unaligned, scale_groups groups);
using wide_fn = ukernel_fn (*)(
        int mr, epilogue ep, bool src_u8, bool two_chains, scale_groups groups);

ukernel_fn bf16_narrow(const int mr, const epilogue ep, bool,
        const bool unaligned, scale_groups) {
    return select_bf16(mr, ep, unaligned);
}

ukernel_fn bf16_wide(const int mr, const epilogue ep, bool,
        const bool two_chains, scale_groups) {
    return select_bf16_wide(mr, ep, two_chains);
}

ukernel_fn f16_narrow(const int mr, const epilogue ep, bool,
        const bool unaligned, scale_groups) {
    return select_f16(mr, ep, unaligned);
}

ukernel_fn s8_narrow(const int mr, const epilogue ep, const bool src_u8,
        const bool unaligned, const scale_groups groups) {
    return groups.any()
            ? select_s8_groups(mr, ep, src_u8, unaligned, groups.source)
            : select_s8(mr, ep, src_u8, unaligned);
}

ukernel_fn s8_wide(const int mr, const epilogue ep, const bool src_u8, bool,
        const scale_groups groups) {
    return groups.any() ? select_s8_groups_wide(mr, ep, src_u8, groups.source)
                        : select_s8_wide(mr, ep, src_u8);
}

ukernel_fn s4_narrow(const int mr, const epilogue ep, bool,
        const bool unaligned, const scale_groups groups) {
    return select_s4(mr, ep, unaligned, groups.source);
}

ukernel_fn s4_wide(const int mr, const epilogue ep, bool, bool,
        const scale_groups groups) {
    return select_s4_wide(mr, ep, groups.source);
}

/// A format's kernels: narrow, and wide for a format with `max_mr_wide`.
struct family_t {
    data_type_t wei;
    narrow_fn narrow;
    wide_fn wide;
};

constexpr family_t kFamilies[] = {
        {data_type_t::bf16, bf16_narrow, bf16_wide},
        {data_type_t::f16, f16_narrow, nullptr},
        {data_type_t::s8, s8_narrow, s8_wide},
        {data_type_t::s4, s4_narrow, s4_wide},
};

constexpr const family_t *family_of(const data_type_t wei) {
    for (const family_t &k : kFamilies)
        if (k.wei == wei) return &k;
    return nullptr;
}

constexpr bool families_match_formats() {
    for (const format_desc &f : kFormats) {
        const family_t *k = family_of(f.wei);
        if (k == nullptr || k->narrow == nullptr
                || (k->wide != nullptr) != (f.max_mr_wide > 0))
            return false;
    }
    return std::size(kFamilies) == static_cast<size_t>(kFormatCount);
}
static_assert(families_match_formats(),
        "every N-tile flat parallel format has one kernel family, wide kernels "
        "where it "
        "takes wide rows, and every family a format");

} // namespace

int max_mr(const data_type_t wei) {
    const format_desc *f = format_of(wei);
    return f != nullptr ? f->max_mr : 0;
}

int max_mr_wide(const data_type_t wei) {
    const format_desc *f = format_of(wei);
    return f != nullptr ? f->max_mr_wide : 0;
}

ukernel_fn select_wide(const data_type_t wei, const int mr, const epilogue ep,
        const bool src_u8, const bool two_chains, const scale_groups groups) {
    const format_desc *f = format_of(wei);
    if (f == nullptr || mr < 1 || mr > f->max_mr_wide || !split(ep))
        return nullptr;
    if ((src_u8 && !f->u8_src) || (two_chains && mr > f->two_chain_rows))
        return nullptr;
    return family_of(wei)->wide(mr, ep, src_u8, two_chains, groups);
}

bool isa_supported(const data_type_t wei) {
    const auto &cpu = zendnnl::common::zendnnl_platform_info();
    const format_desc *f = format_of(wei);
    if (f == nullptr || !cpu.get_avx512f_status()
            || !cpu.get_avx512_bw_vl_status())
        return false;
    switch (f->isa) {
        case isa_need::vnni: return cpu.get_avx512_vnni_status();
        case isa_need::bf16: return cpu.get_avx512_bf16_status();
        case isa_need::fp16:
            return f16_kernels_built() && cpu.get_avx512_f16_status();
    }
    return false;
}

ukernel_fn select(const data_type_t wei, const int mr, const epilogue ep,
        const bool src_u8, const bool unaligned, const scale_groups groups) {
    const format_desc *f = format_of(wei);
    if (f == nullptr || mr < 1 || mr > f->max_mr) return nullptr;
    if (src_u8 && !f->u8_src) return nullptr;
    return family_of(wei)->narrow(mr, ep, src_u8, unaligned, groups);
}

} // namespace nr32
} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
