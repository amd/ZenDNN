/*******************************************************************************
* Modifications Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Copyright 2017-2019 Intel Corporation
* Copyright 2018 YANDEX LLC
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

#include "cpu/x64/zendnn_pool_kernel.hpp"
#include "cpu/cpu_pooling_pd.hpp"
#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"

namespace zendnn {
namespace impl {
namespace cpu {
namespace x64 {
    
using namespace Xbyak;
using namespace alg_kind;

#define GET_OFF(field) offsetof(jit_pool_call_s, field)

template <cpu_isa_t isa>
zendnn_pool_kernel<isa>::~zendnn_pool_kernel() = default;

template <cpu_isa_t isa>
zendnn_pool_kernel<isa>::zendnn_pool_kernel(
        const jit_pool_conf_t &ajpp, const memory_desc_t *dst_md)
    : jit_generator(nullptr, MAX_CODE_SIZE, true, isa)
    , jpp(ajpp)
    , bf16_emu_(nullptr) {
    }

template <cpu_isa_t isa>
status_t zendnn_pool_kernel<isa>::init_conf(jit_pool_conf_t &jpp,
        memory_tracking::registrar_t &scratchpad, const pooling_pd_t *ppd,
        int nthreads) {

    const auto &pd = *ppd->desc();
    const memory_desc_wrapper src_d(
        ppd->is_fwd() ? ppd->src_md() : ppd->diff_src_md());
    const memory_desc_wrapper dst_d(
        ppd->is_fwd() ? ppd->dst_md() : ppd->diff_dst_md());

    jpp.is_bf16 = (src_d.data_type() == data_type::bf16
                   && dst_d.data_type() == data_type::bf16);

    jpp.isa = (jpp.is_bf16 && mayiuse(avx512_core_bf16)) ? avx512_core_bf16
              : isa;

    bool args_ok = true && mayiuse(isa)
                   && IMPLICATION(jpp.is_bf16, mayiuse(avx512_core))
                   && utils::one_of(pd.alg_kind, pooling_max,
                                    pooling_avg_include_padding, pooling_avg_exclude_padding);
    if (!args_ok) {
        return status::unimplemented;
    }

    bool is_avx512 = utils::one_of(isa, avx512_common, avx512_core);
    const int simd_w = is_avx512 ? 16 : 8;
    const int ndims = src_d.ndims();

    jpp.ndims = ndims;
    jpp.mb = src_d.dims()[0];

    jpp.c = utils::rnd_up(src_d.dims()[1], simd_w);
    if (jpp.c > src_d.padded_dims()[1]) {
        return status::unimplemented;
    }

    jpp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jpp.ih = src_d.dims()[ndims - 2];
    jpp.iw = src_d.dims()[ndims - 1];
    jpp.od = (ndims == 5) ? dst_d.dims()[2] : 1;
    jpp.oh = dst_d.dims()[ndims - 2];
    jpp.ow = dst_d.dims()[ndims - 1];

    jpp.stride_d = (ndims == 5) ? pd.strides[0] : 1;
    jpp.stride_h = pd.strides[ndims - 4];
    jpp.stride_w = pd.strides[ndims - 3];
    jpp.kd = (ndims == 5) ? pd.kernel[0] : 1;
    jpp.kh = pd.kernel[ndims - 4];
    jpp.kw = pd.kernel[ndims - 3];

    jpp.f_pad = (ndims == 5) ? pd.padding[0][0] : 0;
    jpp.t_pad = pd.padding[0][ndims - 4];
    jpp.l_pad = pd.padding[0][ndims - 3];

    return status::success;
}
template <cpu_isa_t isa>
void zendnn_pool_kernel<isa>::generate() {}

template struct zendnn_pool_kernel<avx2>; // implements both <avx> and <avx2>

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace zendnn

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
