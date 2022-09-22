/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
* Copyright 2022 Arm Ltd. and affiliates
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

#include "cpu/ref_binary.hpp"

#if ZENDNN_X64
#include "cpu/x64/jit_uni_binary.hpp"
using namespace zendnn::impl::cpu::x64;
#elif ZENDNN_AARCH64 && ZENDNN_AARCH64_USE_ACL
#include "cpu/aarch64/acl_binary.hpp"
using namespace zendnn::impl::cpu::aarch64;
#endif

namespace zendnn {
namespace impl {
namespace cpu {

namespace {
using namespace zendnn::impl::data_type;

// clang-format off
constexpr impl_list_item_t impl_list[] = REG_BINARY_P({
        CPU_INSTANCE_X64(jit_uni_binary_t)
        CPU_INSTANCE_AARCH64_ACL(acl_binary_t)
        CPU_INSTANCE(ref_binary_t)
        /* eol */
        nullptr,
});
// clang-format on
} // namespace

const impl_list_item_t *get_binary_impl_list(const binary_desc_t *desc) {
    UNUSED(desc);
    return impl_list;
}

} // namespace cpu
} // namespace impl
} // namespace zendnn
