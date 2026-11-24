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

#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/kernel/nr32_ukernel.hpp"

#include "common/zendnnl_global.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/formats/formats.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {
namespace nr32 {

ukernel_fn select_wide(const format_desc &f, const int mr, const epilogue ep,
        const bool src_u8, const bool two_chains, const scale_groups groups) {
    if (mr < 1 || mr > f.max_mr_wide || !split(ep)) return nullptr;
    if ((src_u8 && !f.u8_src) || (two_chains && mr > f.two_chain_rows))
        return nullptr;
    return f.wide(mr, ep, src_u8, two_chains, groups);
}

bool isa_supported(const format_desc &f) {
    const auto &cpu = zendnnl::common::zendnnl_platform_info();
    if ((f.built != nullptr && !f.built()) || !cpu.get_avx512f_status()
            || !cpu.get_avx512_bw_vl_status())
        return false;
    return (!needs(f.isa, isa_need::vnni) || cpu.get_avx512_vnni_status())
            && (!needs(f.isa, isa_need::bf16) || cpu.get_avx512_bf16_status())
            && (!needs(f.isa, isa_need::fp16) || cpu.get_avx512_f16_status());
}

ukernel_fn select(const format_desc &f, const int mr, const epilogue ep,
        const bool src_u8, const bool unaligned, const scale_groups groups) {
    if (mr < 1 || mr > f.max_mr) return nullptr;
    if (src_u8 && !f.u8_src) return nullptr;
    return f.narrow(mr, ep, src_u8, unaligned, groups);
}

} // namespace nr32
} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
