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

int max_mr(const data_type_t wei) {
    switch (wei) {
        case data_type_t::s8: return 12;
        case data_type_t::s4: return 6;
        case data_type_t::bf16:
        case data_type_t::f16: return 8;
        default: return 0;
    }
}

int max_mr_wide(const data_type_t wei) {
    switch (wei) {
        case data_type_t::s8:
        case data_type_t::bf16: return 6;
        case data_type_t::s4: return 3;
        default: return 0;
    }
}

ukernel_fn select_wide(const data_type_t wei, const int mr, const epilogue ep,
        const bool src_u8, const bool two_chains, const scale_groups groups) {
    if (mr < 1 || mr > max_mr_wide(wei) || !split(ep)) return nullptr;
    if (src_u8 && wei != data_type_t::s8) return nullptr;
    if (two_chains && (wei != data_type_t::bf16 || mr > 3)) return nullptr;
    switch (wei) {
        case data_type_t::s8:
            return groups.any()
                    ? select_s8_groups_wide(mr, ep, src_u8, groups.source)
                    : select_s8_wide(mr, ep, src_u8);
        case data_type_t::bf16: return select_bf16_wide(mr, ep, two_chains);
        case data_type_t::s4: return select_s4_wide(mr, ep, groups.source);
        default: return nullptr;
    }
}

int step_cols(const data_type_t wei, const epilogue ep) {
    if (ep == epilogue::plain) return 32;
    if (wei == data_type_t::f16 && split(ep)) return 32;
    return 16;
}

bool isa_supported(const data_type_t wei) {
    const auto &cpu = zendnnl::common::zendnnl_platform_info();
    if (!cpu.get_avx512f_status() || !cpu.get_avx512_bw_vl_status())
        return false;
    switch (wei) {
        case data_type_t::s8:
        case data_type_t::s4: return cpu.get_avx512_vnni_status();
        case data_type_t::bf16: return cpu.get_avx512_bf16_status();
        case data_type_t::f16:
            return f16_kernels_built() && cpu.get_avx512_f16_status();
        default: return false;
    }
}

ukernel_fn select(const data_type_t wei, const int mr, const epilogue ep,
        const bool src_u8, const bool unaligned, const scale_groups groups) {
    if (mr < 1 || mr > max_mr(wei)) return nullptr;
    if (src_u8 && wei != data_type_t::s8) return nullptr;
    switch (wei) {
        case data_type_t::s8:
            return groups.any()
                    ? select_s8_groups(mr, ep, src_u8, unaligned, groups.source)
                    : select_s8(mr, ep, src_u8, unaligned);
        case data_type_t::s4:
            return select_s4(mr, ep, unaligned, groups.source);
        case data_type_t::bf16: return select_bf16(mr, ep, unaligned);
        case data_type_t::f16: return select_f16(mr, ep, unaligned);
        default: return nullptr;
    }
}

} // namespace nr32
} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
