/*******************************************************************************
* Modifications Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include "cpu/ref_lrn.hpp"

#if ZENDNN_X64
#include "cpu/x64/lrn/jit_avx512_common_lrn.hpp"
#include "cpu/x64/lrn/jit_uni_lrn.hpp"
using namespace zendnn::impl::cpu::x64;
#endif

namespace zendnn {
namespace impl {
namespace cpu {

using pd_create_f = engine_t::primitive_desc_create_f;

namespace {
using namespace zendnn::impl::data_type;

// clang-format off
const pd_create_f impl_list[] = {
        CPU_INSTANCE_X64(jit_avx512_common_lrn_fwd_t<f32>)
        CPU_INSTANCE_X64(jit_avx512_common_lrn_bwd_t<f32>)
        CPU_INSTANCE_X64(jit_avx512_common_lrn_fwd_t<bf16>)
        CPU_INSTANCE_X64(jit_avx512_common_lrn_bwd_t<bf16>)
        CPU_INSTANCE_X64(jit_uni_lrn_fwd_t<avx512_common, f32>)
        CPU_INSTANCE_X64(jit_uni_lrn_bwd_t<avx512_common, f32>)
        CPU_INSTANCE_X64(jit_uni_lrn_fwd_t<avx512_common, bf16>)
        CPU_INSTANCE_X64(jit_uni_lrn_bwd_t<avx512_common, bf16>)
        CPU_INSTANCE_X64(jit_uni_lrn_fwd_t<avx2, f32>)
        CPU_INSTANCE_X64(jit_uni_lrn_bwd_t<avx2, f32>)
        CPU_INSTANCE_X64(jit_uni_lrn_fwd_t<sse41, f32>)
        CPU_INSTANCE(ref_lrn_fwd_t<f32>)
        CPU_INSTANCE(ref_lrn_bwd_t<f32>)
        CPU_INSTANCE(ref_lrn_fwd_t<bf16>)
        CPU_INSTANCE(ref_lrn_bwd_t<bf16>)
        /* eol */
        nullptr,
};
// clang-format on
} // namespace

const pd_create_f *get_lrn_impl_list(const lrn_desc_t *desc) {
    UNUSED(desc);
    return impl_list;
}

} // namespace cpu
} // namespace impl
} // namespace zendnn
