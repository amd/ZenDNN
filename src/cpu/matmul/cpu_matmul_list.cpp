/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
* Copyright 2021 Arm Ltd. and affiliates
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

#include "cpu/matmul/gemm_bf16_matmul.hpp"
#include "cpu/matmul/gemm_f32_matmul.hpp"
#include "cpu/matmul/gemm_x8s8s32x_matmul.hpp"
#include "cpu/matmul/ref_matmul.hpp"
#include "cpu/matmul/ref_matmul_int8.hpp"

#if ZENDNN_X64
#include "cpu/x64/matmul/brgemm_matmul.hpp"
#include "cpu/matmul/zendnn_f32_matmul.hpp"
using namespace zendnn::impl::cpu::x64::matmul;
using namespace zendnn::impl::cpu::x64;
#elif ZENDNN_AARCH64 && ZENDNN_AARCH64_USE_ACL
#include "cpu/aarch64/matmul/acl_matmul.hpp"
using namespace zendnn::impl::cpu::aarch64::matmul;
using namespace zendnn::impl::cpu::aarch64;

#endif

namespace zendnn {
namespace impl {
namespace cpu {

namespace {
using namespace zendnn::impl::data_type;
using namespace zendnn::impl::cpu::matmul;

// clang-format off
constexpr impl_list_item_t impl_list[] = REG_MATMUL_P({
        CPU_INSTANCE_AARCH64_ACL(acl_matmul_t)
        CPU_INSTANCE_AVX512(brgemm_matmul_t<avx512_core>)
        CPU_INSTANCE(zendnn_f32_matmul_t)
        CPU_INSTANCE(gemm_f32_matmul_t)
        CPU_INSTANCE_AMX(brgemm_matmul_t<avx512_core_bf16_amx_bf16>)
        CPU_INSTANCE_AVX512(brgemm_matmul_t<avx512_core_bf16>)
        CPU_INSTANCE(gemm_bf16_matmul_t<f32>)
        CPU_INSTANCE(gemm_bf16_matmul_t<bf16>)
        CPU_INSTANCE_AMX(brgemm_matmul_t<avx512_core_bf16_amx_int8>)
        CPU_INSTANCE_AVX512(brgemm_matmul_t<avx512_core_vnni>)
        CPU_INSTANCE(gemm_x8s8s32x_matmul_t)
        CPU_INSTANCE(ref_matmul_t)
        CPU_INSTANCE(ref_matmul_int8_t)
        /* eol */
        nullptr,
});
// clang-format on
} // namespace

const impl_list_item_t *get_matmul_impl_list(const matmul_desc_t *desc) {
    UNUSED(desc);
    return impl_list;
}

} // namespace cpu
} // namespace impl
} // namespace zendnn
