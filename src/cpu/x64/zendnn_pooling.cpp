/*******************************************************************************
* Modifications Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Copyright 2017-2019 Intel Corporation
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

#include <functional>
#include <new>

#include "zendnn_types.h"

#include "common/c_types_map.hpp"
#include "common/zendnn_thread.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"

#include "zendnn_logging.hpp"
#include "cpu/x64/zendnn_pooling.hpp"
#include "common/zendnn_private.hpp"

namespace zendnn {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa, impl::data_type_t d_type>
zendnn_pooling_fwd_t<isa, d_type>::zendnn_pooling_fwd_t(const pd_t *apd)
    : primitive_t(apd), kernel_(nullptr) {}

template <cpu_isa_t isa, impl::data_type_t d_type>
status_t zendnn_pooling_fwd_t<isa, d_type>::init(engine_t *engine) {

    CHECK(safe_ptr_assign(kernel_,
            new zendnn_pool_kernel<isa>(
                    pd()->jpp_, pd()->invariant_dst_md())));

    return kernel_->create_kernel();
}

template <cpu_isa_t isa, impl::data_type_t d_type>
zendnn_pooling_fwd_t<isa, d_type>::~zendnn_pooling_fwd_t() = default;

template <cpu_isa_t isa, data_type_t d_type>
void zendnn_pooling_fwd_t<isa, d_type>::execute_forward(
    const data_t *src, data_t *dst, char *indices,
    const exec_ctx_t &ctx) const {

    const memory_desc_wrapper src_d = pd()->src_md();
    const memory_desc_wrapper dst_d = pd()->dst_md();
    const memory_desc_wrapper indices_d = pd()->workspace_md();
    const size_t ind_dt_size
        = indices ? types::data_type_size(indices_d.data_type()) : 0;
    const auto &jpp = pd()->jpp_;
    zendnnInfo(ZENDNN_CORELOG, "ZENDNN implementation path in zendnn_pooling_fwd_t::execute_forward [cpu/pooling]");

    max_pooling(
        (float *)src,
        jpp.mb,
        jpp.c,
        jpp.ih,
        jpp.iw,
        jpp.kh,
        jpp.kw,
        jpp.stride_w,
        jpp.stride_h,
        jpp.l_pad,
        jpp.l_pad,
        jpp.l_pad,
        jpp.l_pad,
        (float *)dst,
        0 // 1 for NCHW and 0 for NHWC
    );
}

template struct zendnn_pooling_fwd_t<avx2, data_type::f32>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace zendnn

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
