/*******************************************************************************
* Modifications Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#include "cpu/ref_deconvolution.hpp"

#if ZENDNN_X64
#include "cpu/x64/jit_avx512_core_amx_int8_deconvolution.hpp"
#include "cpu/x64/jit_avx512_core_x8s8s32x_1x1_deconvolution.hpp"
#include "cpu/x64/jit_avx512_core_x8s8s32x_deconvolution.hpp"
#include "cpu/x64/jit_uni_x8s8s32x_1x1_deconvolution.hpp"
#include "cpu/x64/jit_uni_x8s8s32x_deconvolution.hpp"
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
        CPU_INSTANCE_X64(jit_avx512_core_amx_int8_deconvolution_fwd_t)
        CPU_INSTANCE_X64(jit_avx512_core_x8s8s32x_1x1_deconvolution_fwd_t<u8, f32>)
        CPU_INSTANCE_X64(jit_avx512_core_x8s8s32x_1x1_deconvolution_fwd_t<u8, s32>)
        CPU_INSTANCE_X64(jit_avx512_core_x8s8s32x_1x1_deconvolution_fwd_t<u8, u8>)
        CPU_INSTANCE_X64(jit_avx512_core_x8s8s32x_1x1_deconvolution_fwd_t<u8, s8>)
        CPU_INSTANCE_X64(jit_avx512_core_x8s8s32x_1x1_deconvolution_fwd_t<s8, f32>)
        CPU_INSTANCE_X64(jit_avx512_core_x8s8s32x_1x1_deconvolution_fwd_t<s8, s32>)
        CPU_INSTANCE_X64(jit_avx512_core_x8s8s32x_1x1_deconvolution_fwd_t<s8, u8>)
        CPU_INSTANCE_X64(jit_avx512_core_x8s8s32x_1x1_deconvolution_fwd_t<s8, s8>)
        CPU_INSTANCE_X64(_jit_avx512_core_x8s8s32x_deconvolution_fwd_t<u8, s32>)
        CPU_INSTANCE_X64(_jit_avx512_core_x8s8s32x_deconvolution_fwd_t<u8, u8>)
        CPU_INSTANCE_X64(_jit_avx512_core_x8s8s32x_deconvolution_fwd_t<u8, s8>)
        CPU_INSTANCE_X64(_jit_avx512_core_x8s8s32x_deconvolution_fwd_t<u8, f32>)
        CPU_INSTANCE_X64(_jit_avx512_core_x8s8s32x_deconvolution_fwd_t<s8, s32>)
        CPU_INSTANCE_X64(_jit_avx512_core_x8s8s32x_deconvolution_fwd_t<s8, u8>)
        CPU_INSTANCE_X64(_jit_avx512_core_x8s8s32x_deconvolution_fwd_t<s8, s8>)
        CPU_INSTANCE_X64(_jit_avx512_core_x8s8s32x_deconvolution_fwd_t<s8, f32>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_1x1_deconvolution_fwd_t<avx2, u8, s32>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_1x1_deconvolution_fwd_t<avx2, u8, u8>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_1x1_deconvolution_fwd_t<avx2, u8, s8>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_1x1_deconvolution_fwd_t<avx2, u8, f32>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_1x1_deconvolution_fwd_t<avx2, s8, s32>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_1x1_deconvolution_fwd_t<avx2, s8, u8>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_1x1_deconvolution_fwd_t<avx2, s8, s8>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_1x1_deconvolution_fwd_t<avx2, s8, f32>)
        CPU_INSTANCE_X64(_jit_uni_x8s8s32x_deconvolution_fwd_t<avx2, u8, s32>)
        CPU_INSTANCE_X64(_jit_uni_x8s8s32x_deconvolution_fwd_t<avx2, u8, u8>)
        CPU_INSTANCE_X64(_jit_uni_x8s8s32x_deconvolution_fwd_t<avx2, u8, s8>)
        CPU_INSTANCE_X64(_jit_uni_x8s8s32x_deconvolution_fwd_t<avx2, u8, f32>)
        CPU_INSTANCE_X64(_jit_uni_x8s8s32x_deconvolution_fwd_t<avx2, s8, s32>)
        CPU_INSTANCE_X64(_jit_uni_x8s8s32x_deconvolution_fwd_t<avx2, s8, u8>)
        CPU_INSTANCE_X64(_jit_uni_x8s8s32x_deconvolution_fwd_t<avx2, s8, s8>)
        CPU_INSTANCE_X64(_jit_uni_x8s8s32x_deconvolution_fwd_t<avx2, s8, f32>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_1x1_deconvolution_fwd_t<sse41, u8, s32>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_1x1_deconvolution_fwd_t<sse41, u8, u8>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_1x1_deconvolution_fwd_t<sse41, u8, s8>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_1x1_deconvolution_fwd_t<sse41, u8, f32>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_1x1_deconvolution_fwd_t<sse41, s8, s32>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_1x1_deconvolution_fwd_t<sse41, s8, u8>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_1x1_deconvolution_fwd_t<sse41, s8, s8>)
        CPU_INSTANCE_X64(jit_uni_x8s8s32x_1x1_deconvolution_fwd_t<sse41, s8, f32>)
        CPU_INSTANCE_X64(_jit_uni_x8s8s32x_deconvolution_fwd_t<sse41, u8, s32>)
        CPU_INSTANCE_X64(_jit_uni_x8s8s32x_deconvolution_fwd_t<sse41, u8, u8>)
        CPU_INSTANCE_X64(_jit_uni_x8s8s32x_deconvolution_fwd_t<sse41, u8, s8>)
        CPU_INSTANCE_X64(_jit_uni_x8s8s32x_deconvolution_fwd_t<sse41, u8, f32>)
        CPU_INSTANCE_X64(_jit_uni_x8s8s32x_deconvolution_fwd_t<sse41, s8, s32>)
        CPU_INSTANCE_X64(_jit_uni_x8s8s32x_deconvolution_fwd_t<sse41, s8, u8>)
        CPU_INSTANCE_X64(_jit_uni_x8s8s32x_deconvolution_fwd_t<sse41, s8, s8>)
        CPU_INSTANCE_X64(_jit_uni_x8s8s32x_deconvolution_fwd_t<sse41, s8, f32>)
        CPU_INSTANCE(ref_deconvolution_bwd_weights_t)
        CPU_INSTANCE(ref_deconvolution_bwd_data_t)
        CPU_INSTANCE(ref_deconvolution_fwd_t)
        /* eol */
        nullptr,
};
// clang-format on
} // namespace

const pd_create_f *get_deconvolution_impl_list(
        const deconvolution_desc_t *desc) {
    UNUSED(desc);
    return impl_list;
}

} // namespace cpu
} // namespace impl
} // namespace zendnn
