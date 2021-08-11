/*******************************************************************************
* Modifications Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Copyright 2016-2019 Intel Corporation
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

#include "common/c_types_map.hpp"
#include "common/memory.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/zendnn_conv_kernel_f32.hpp"

#define GET_OFF(field) offsetof(jit_conv_call_s, field)

namespace zendnn {
namespace impl {
namespace cpu {
namespace x64 {

using namespace zendnn::impl::prop_kind;
using namespace zendnn::impl::format_tag;
using namespace zendnn::impl::memory_tracking::names;
using namespace zendnn::impl::utils;

zendnn_conv_fwd_kernel_f32::zendnn_conv_fwd_kernel_f32(
        const jit_conv_conf_t &ajcp, const primitive_attr_t &attr,
        const memory_desc_t &dst_md)
    : jcp(ajcp)
    , attr_(attr) {}

status_t zendnn_conv_fwd_kernel_f32::init_conf(jit_conv_conf_t &jcp,
        const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, const memory_desc_wrapper &dst_d,
        const primitive_attr_t &attr) {

    jcp.prop_kind = cd.prop_kind;
    jcp.alg_kind = cd.alg_kind;

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    int ndims = src_d.ndims();
    jcp.ndims = ndims;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];

    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;

    jcp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jcp.ih = (ndims == 3) ? 1 : src_d.dims()[ndims - 2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.od = (ndims == 5) ? dst_d.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 : dst_d.dims()[ndims - 2];
    jcp.ow = dst_d.dims()[ndims - 1];
    jcp.kd = (ndims == 5) ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = (ndims == 3) ? 1 : weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];

    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims - 4];
    jcp.l_pad = cd.padding[0][ndims - 3];
    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[ndims - 4];
    jcp.stride_w = cd.strides[ndims - 3];

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = (ndims == 3) ? 0 : cd.dilates[ndims - 4];
    jcp.dilate_w = cd.dilates[ndims - 3];

    //filling pad parameters as passed from user level api
    jcp.t_pad = cd.padding[0][0];
    jcp.l_pad = cd.padding[0][1];
    jcp.b_pad = cd.padding[1][0];
    jcp.r_pad = cd.padding[1][1];

    //We achieve fusion with Post op
    const auto &post_ops = attr.post_ops_;
    jcp.with_sum = post_ops.find(primitive_kind::sum) != -1;
    const int eltwise_ind = post_ops.find(primitive_kind::eltwise);
    jcp.with_eltwise = eltwise_ind != -1;
    const int binary_ind = post_ops.find(primitive_kind::binary);
    jcp.with_binary = binary_ind != -1;

    jcp.post_ops = post_ops;

    //ReLU and BatchNorm fusion flags
    jcp.reluFused      = cd.reluFused;
    jcp.batchNormFused = cd.batchNormFused;

    return status::success;
}

void zendnn_conv_fwd_kernel_f32::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp) {
    if (jcp.with_bias && jcp.oc != jcp.oc_without_padding)
        scratchpad.book<float>(key_conv_padded_bias, jcp.oc);
}

void zendnn_conv_fwd_kernel_f32::generate() {}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace zendnn

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
