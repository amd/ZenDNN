/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#include "cpu/reorder/cpu_reorder.hpp"

namespace zendnn {
namespace impl {
namespace cpu {

// clang-format off

const impl_list_map_t &regular_u8_impl_list_map() {
    static const impl_list_map_t the_map = REG_REORDER_P({
        // u8 ->
        {{u8, data_type::undef, 0}, {
            REG_FAST_DIRECT_COPY(u8, f32)
            REG_FAST_DIRECT_COPY(u8, s32)
            REG_FAST_DIRECT_COPY(u8, bf16)
            REG_FAST_DIRECT_COPY(u8, s8)
            REG_FAST_DIRECT_COPY(u8, u8)

            ZENDNN_X64_ONLY(CPU_REORDER_INSTANCE(x64::jit_blk_reorder_t))
            ZENDNN_X64_ONLY(CPU_REORDER_INSTANCE(x64::jit_uni_reorder_t))

            ZENDNN_AARCH64_ONLY(CPU_REORDER_INSTANCE(aarch64::jit_uni_reorder_t))

            ZENDNN_NON_X64_ONLY(REG_SR_BIDIR(u8, any, f32, nChw16c))
            ZENDNN_NON_X64_ONLY(REG_SR_BIDIR(u8, any, s32, nChw16c))
            ZENDNN_NON_X64_ONLY(REG_SR_BIDIR(u8, any, bf16, nChw16c))
            ZENDNN_NON_X64_ONLY(REG_SR_BIDIR(u8, any, s8, nChw16c))
            ZENDNN_NON_X64_ONLY(REG_SR_BIDIR(u8, any, u8, nChw16c))

            REG_SR(u8, any, f32, any, fmt_order::any, spec::reference)
            REG_SR(u8, any, s32, any, fmt_order::any, spec::reference)
            REG_SR(u8, any, bf16, any, fmt_order::any, spec::reference)
            REG_SR(u8, any, u8, any, fmt_order::any, spec::reference)
            REG_SR(u8, any, s8, any, fmt_order::any, spec::reference)

            nullptr,
        }},
    });
    return the_map;
}

// clang-format on

} // namespace cpu
} // namespace impl
} // namespace zendnn
