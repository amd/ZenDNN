/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
* Copyright 2022 FUJITSU LIMITED
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

#include "cpu/cpu_engine.hpp"

#include "common/bfloat16.hpp"
#include "cpu/ref_shuffle.hpp"

#if ZENDNN_X64
#include "cpu/x64/shuffle/jit_uni_shuffle.hpp"
using namespace zendnn::impl::cpu::x64;
#elif ZENDNN_AARCH64
#include "cpu/aarch64/shuffle/jit_uni_shuffle.hpp"
using namespace zendnn::impl::cpu::aarch64;
#endif

namespace zendnn {
namespace impl {
namespace cpu {

namespace {
using namespace zendnn::impl::data_type;

// clang-format off
constexpr impl_list_item_t impl_list[] = REG_SHUFFLE_P({
        CPU_INSTANCE_X64(jit_uni_shuffle_t<avx512_core>)
        CPU_INSTANCE_X64(jit_uni_shuffle_t<avx>)
        CPU_INSTANCE_X64(jit_uni_shuffle_t<sse41>)
        CPU_INSTANCE_AARCH64(jit_uni_shuffle_t<sve_512>)
        CPU_INSTANCE(ref_shuffle_t)
        /* eol */
        nullptr,
});
// clang-format on
} // namespace

const impl_list_item_t *get_shuffle_impl_list(const shuffle_desc_t *desc) {
    UNUSED(desc);
    return impl_list;
}

} // namespace cpu
} // namespace impl
} // namespace zendnn
