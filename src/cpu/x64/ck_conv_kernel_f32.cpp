/*******************************************************************************
* Modifications Copyright (c) 2021-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <unordered_map>

#include "common/c_types_map.hpp"
#include "common/memory.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/ck_conv_kernel_f32.hpp"

#include "zendnn_logging.hpp"

#define GET_OFF(field) offsetof(jit_conv_call_s, field)

namespace zendnn {
namespace impl {
namespace cpu {
namespace x64 {

// ck_conv_prb_desc_t will be extended to include all relevant parameters of convolution problem
// and relevant environment variables, e.g., number of OpenMP threads
struct ck_conv_prb_desc_t {
    size_t mb;
    size_t oc;
    size_t ic;
    size_t kh;
    size_t kw;
    size_t ih;
    size_t iw;
    size_t sh;
    size_t sw;
    size_t dh;
    size_t dw;
    size_t ph;
    size_t pw;
};

// we need operator== in order to use ck_conv_prb_desc_t as a key type in std::unordered_map
bool operator==(const ck_conv_prb_desc_t &lhs, const ck_conv_prb_desc_t &rhs) {
    return lhs.mb == rhs.mb &&
           lhs.oc == rhs.oc &&
           lhs.ic == rhs.ic &&
           lhs.kh == rhs.kh &&
           lhs.kw == rhs.kw &&
           lhs.ih == rhs.ih &&
           lhs.iw == rhs.iw &&
           lhs.sh == rhs.sh &&
           lhs.sw == rhs.sw &&
           lhs.dh == rhs.dh &&
           lhs.dw == rhs.dw &&
           lhs.ph == rhs.ph &&
           lhs.pw == rhs.pw;
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace zendnn

// The following code is derived from Boost C++ library
// Copyright 2005-2014 Daniel James.
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
template <typename T>
static size_t hash_combine(size_t seed, const T &v) {
    return seed ^= std::hash<T> {}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// code taken and modified from https://en.cppreference.com/w/cpp/utility/hash
// custom specialization of std::hash can be injected in namespace std
// this hash will be used by std::unordered_map<ck_conv_prb_desc_t, int>
template<>
struct std::hash<zendnn::impl::cpu::x64::ck_conv_prb_desc_t> {
    size_t operator()(zendnn::impl::cpu::x64::ck_conv_prb_desc_t const &s) const noexcept {
        size_t seed = 0;
        seed = hash_combine(seed, s.mb);
        seed = hash_combine(seed, s.oc);
        seed = hash_combine(seed, s.ic);
        seed = hash_combine(seed, s.kh);
        seed = hash_combine(seed, s.kw);
        seed = hash_combine(seed, s.ih);
        seed = hash_combine(seed, s.iw);
        seed = hash_combine(seed, s.sh);
        seed = hash_combine(seed, s.sw);
        seed = hash_combine(seed, s.dh);
        seed = hash_combine(seed, s.dw);
        seed = hash_combine(seed, s.ph);
        seed = hash_combine(seed, s.pw);
        return seed;
    }
};


namespace zendnn {
namespace impl {
namespace cpu {
namespace x64 {

using namespace zendnn::impl::prop_kind;
using namespace zendnn::impl::format_tag;
using namespace zendnn::impl::memory_tracking::names;
using namespace zendnn::impl::utils;

ck_conv_fwd_kernel_f32::ck_conv_fwd_kernel_f32(
    const jit_conv_conf_t &ajcp, const primitive_attr_t &attr,
    const memory_desc_t &dst_md)
    : jcp(ajcp)
    , attr_(attr) {}

status_t ck_conv_fwd_kernel_f32::init_conf(jit_conv_conf_t &jcp,
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

    // Check whether number of output channels is a multiple of 8
    if (jcp.oc % 8 != 0) {
        return status::unimplemented;
    }

    // Check whether layout tag in memory descriptor for src
    // is consistent with memory format that CK expects
    auto src_tag_nhwc = nhwc;
    auto src_tag_any = zendnn_format_tag_any;
    jcp.src_tag = src_d.matches_one_of_tag(src_tag_nhwc, src_tag_any);
    bool src_tag_ok = ( jcp.src_tag != zendnn_format_tag_undef );
    if (!src_tag_ok) return status::unimplemented;

    // Check whether layout tag in memory descriptor for dst
    // is consistent with memory format that CK expects
    auto dst_tag_nhwc = nhwc;
    auto dst_tag_any = zendnn_format_tag_any;
    jcp.dst_tag = dst_d.matches_one_of_tag(dst_tag_nhwc, dst_tag_any);
    bool dst_tag_ok = ( jcp.dst_tag != zendnn_format_tag_undef );
    if (!dst_tag_ok) return status::unimplemented;

    // Check whether layout tag in memory descriptor for weights
    // is consistent with memory format that CK expects
    auto wei_tag_Ohwi8o = Ohwi8o;
    auto wei_tag_any = zendnn_format_tag_any;
    jcp.wei_tag = weights_d.matches_one_of_tag(wei_tag_Ohwi8o, wei_tag_any);
    bool wei_tag_ok = ( jcp.wei_tag != zendnn_format_tag_undef );
    if (!wei_tag_ok) return status::unimplemented;

    return status::success;
}

// global_ck_idx_cache defines mapping between convolution problem description
// and fastest CK kernel index
static std::unordered_map<ck_conv_prb_desc_t, int> global_ck_idx_cache = {
//  {{mb, oc, ic, kh, kw, ih, iw, ...}, fastest_ck_idx}
    {{4, 512, 256, 3, 3, 28, 28, 1, 1, 1, 1, 1, 1}, 55}, // vgg_19:conv4_1
    {{4, 512, 512, 3, 3, 14, 14, 1, 1, 1, 1, 1, 1}, 75}, // vgg_19:conv5_1*4
    {{4, 64, 64, 3, 3, 224, 224, 1, 1, 1, 1, 1, 1}, 80}, // vgg_19:conv1_2
    {{4, 128, 64, 3, 3, 112, 112, 1, 1, 1, 1, 1, 1}, 81}, // vgg_19:conv2_1
    {{4, 128, 128, 3, 3, 112, 112, 1, 1, 1, 1, 1, 1}, 80}, // vgg_19:conv2_2
    {{4, 64, 3, 3, 3, 224, 224, 1, 1, 1, 1, 1, 1}, 37}, // vgg_19:conv1_1
    {{4, 512, 512, 3, 3, 28, 28, 1, 1, 1, 1, 1, 1}, 65}, // vgg_19:conv4_2*3
    {{4, 256, 256, 3, 3, 56, 56, 1, 1, 1, 1, 1, 1}, 77}, // vgg_19:conv3_2*3
    {{4, 256, 128, 3, 3, 56, 56, 1, 1, 1, 1, 1, 1}, 57}, // vgg_19:conv3_1
    {{4, 48, 256, 1, 1, 35, 35, 1, 1, 1, 1, 0, 0}, 33}, // googlenet_v3-mixed_1_tower_conv_conv2d
    {{4, 128, 128, 1, 7, 17, 17, 1, 1, 1, 1, 0, 3}, 80}, // googlenet_v3-mixed_4_tower_conv_1_conv2dx2
    {{4, 64, 288, 1, 1, 35, 35, 1, 1, 1, 1, 0, 0}, 80}, // googlenet_v3-mixed_2_conv_conv2dx4
    {{4, 160, 160, 1, 7, 17, 17, 1, 1, 1, 1, 0, 3}, 81}, // googlenet_v3-mixed_5_tower_conv_1_conv2dx4
    {{4, 192, 1280, 1, 1, 8, 8, 1, 1, 1, 1, 0, 0}, 33}, // googlenet_v3-mixed_9_tower_2_conv_conv2d
    {{4, 64, 192, 1, 1, 35, 35, 1, 1, 1, 1, 0, 0}, 68}, // googlenet_v3-mixed_conv_conv2dx2
    {{4, 192, 128, 7, 1, 17, 17, 1, 1, 1, 1, 3, 0}, 80}, // googlenet_v3-mixed_4_tower_conv_2_conv2d
    {{4, 384, 2048, 1, 1, 8, 8, 1, 1, 1, 1, 0, 0}, 33}, // googlenet_v3-mixed_10_tower_conv_conv2d
    {{4, 192, 192, 7, 1, 17, 17, 1, 1, 1, 1, 3, 0}, 81}, // googlenet_v3-mixed_7_tower_conv_2_conv2dx4
    {{4, 64, 256, 1, 1, 35, 35, 1, 1, 1, 1, 0, 0}, 80}, // googlenet_v3-mixed_1_conv_conv2dx3
    {{4, 384, 384, 3, 1, 8, 8, 1, 1, 1, 1, 1, 0}, 83}, // googlenet_v3-mixed_9_tower_mixed_conv_1_conv2dx4
    {{4, 64, 48, 5, 5, 35, 35, 1, 1, 1, 1, 2, 2}, 80}, // googlenet_v3-mixed_tower_conv_1_conv2dx3
    {{4, 32, 3, 3, 3, 299, 299, 2, 2, 1, 1, 0, 0}, 80}, // googlenet_v3-conv_conv2d
    {{4, 384, 288, 3, 3, 35, 35, 2, 2, 1, 1, 0, 0}, 67}, // googlenet_v3-mixed_3_conv_conv2d
    {{4, 48, 192, 1, 1, 35, 35, 1, 1, 1, 1, 0, 0}, 38}, // googlenet_v3-mixed_tower_conv_conv2d
    {{4, 96, 64, 3, 3, 35, 35, 1, 1, 1, 1, 1, 1}, 80}, // googlenet_v3-mixed_tower_1_conv_1_conv2dx4
    {{4, 192, 2048, 1, 1, 8, 8, 1, 1, 1, 1, 0, 0}, 82}, // googlenet_v3-mixed_10_tower_2_conv_conv2d
    {{4, 192, 160, 7, 1, 17, 17, 1, 1, 1, 1, 3, 0}, 80}, // googlenet_v3-mixed_5_tower_conv_2_conv2dx2
    {{4, 128, 128, 7, 1, 17, 17, 1, 1, 1, 1, 3, 0}, 80}, // googlenet_v3-mixed_4_tower_1_conv_1_conv2dx2
    {{4, 32, 32, 3, 3, 149, 149, 1, 1, 1, 1, 0, 0}, 80}, // googlenet_v3-conv_1_1_conv2d
    {{4, 192, 80, 3, 3, 73, 73, 1, 1, 1, 1, 0, 0}, 37}, // googlenet_v3-conv_4_4_conv2d
    {{4, 384, 384, 1, 3, 8, 8, 1, 1, 1, 1, 0, 1}, 83}, // googlenet_v3-mixed_9_tower_mixed_conv_conv2dx4
    {{4, 448, 1280, 1, 1, 8, 8, 1, 1, 1, 1, 0, 0}, 13}, // googlenet_v3-mixed_9_tower_1_conv_conv2d
    {{4, 320, 2048, 1, 1, 8, 8, 1, 1, 1, 1, 0, 0}, 13}, // googlenet_v3-mixed_10_conv_conv2d
    {{4, 448, 2048, 1, 1, 8, 8, 1, 1, 1, 1, 0, 0}, 63}, // googlenet_v3-mixed_10_tower_1_conv_conv2d
    {{4, 192, 160, 1, 7, 17, 17, 1, 1, 1, 1, 0, 3}, 81}, // googlenet_v3-mixed_5_tower_1_conv_4_conv2dx2
    {{4, 192, 128, 1, 7, 17, 17, 1, 1, 1, 1, 0, 3}, 80}, // googlenet_v3-mixed_4_tower_1_conv_4_conv2d
    {{4, 192, 192, 1, 7, 17, 17, 1, 1, 1, 1, 0, 3}, 80}, // googlenet_v3-mixed_7_tower_conv_1_conv2dx4
    {{4, 160, 768, 1, 1, 17, 17, 1, 1, 1, 1, 0, 0}, 80}, // googlenet_v3-mixed_5_tower_conv_conv2dx4
    {{4, 192, 192, 3, 3, 17, 17, 2, 2, 1, 1, 0, 0}, 79}, // googlenet_v3-mixed_8_tower_1_conv_3_conv2d
    {{4, 384, 1280, 1, 1, 8, 8, 1, 1, 1, 1, 0, 0}, 33}, // googlenet_v3-mixed_9_tower_conv_conv2d
    {{4, 384, 448, 3, 3, 8, 8, 1, 1, 1, 1, 1, 1}, 83}, // googlenet_v3-mixed_9_tower_1_conv_1_conv2dx2
    {{4, 64, 32, 3, 3, 147, 147, 1, 1, 1, 1, 1, 1}, 80}, // googlenet_v3-conv_2_2_conv2d
    {{4, 160, 160, 7, 1, 17, 17, 1, 1, 1, 1, 3, 0}, 80}, // googlenet_v3-mixed_5_tower_1_conv_1_conv2dx4
    {{4, 96, 96, 3, 3, 35, 35, 2, 2, 1, 1, 0, 0}, 81}, // googlenet_v3-mixed_3_tower_conv_2_conv2d
    {{4, 32, 192, 1, 1, 35, 35, 1, 1, 1, 1, 0, 0}, 13}, // googlenet_v3-mixed_tower_2_conv_conv2d
    {{4, 192, 768, 1, 1, 17, 17, 1, 1, 1, 1, 0, 0}, 13}, // googlenet_v3-mixed_4_conv_conv2dx12
    {{4, 320, 1280, 1, 1, 8, 8, 1, 1, 1, 1, 0, 0}, 18}, // googlenet_v3-mixed_9_conv_conv2d
    {{4, 96, 96, 3, 3, 35, 35, 1, 1, 1, 1, 1, 1}, 80}, // googlenet_v3-mixed_tower_1_conv_2_conv2dx3
    {{4, 48, 288, 1, 1, 35, 35, 1, 1, 1, 1, 0, 0}, 33}, // googlenet_v3-mixed_2_tower_conv_conv2d
    {{4, 80, 64, 1, 1, 73, 73, 1, 1, 1, 1, 0, 0}, 58}, // googlenet_v3-conv_3_3_conv2d
    {{4, 320, 192, 3, 3, 17, 17, 2, 2, 1, 1, 0, 0}, 19}, // googlenet_v3-mixed_8_tower_conv_1_conv2d
    {{4, 128, 768, 1, 1, 17, 17, 1, 1, 1, 1, 0, 0}, 80}, // googlenet_v3-mixed_4_tower_conv_conv2dx2
};

status_t ck_conv_fwd_kernel_f32::init_ck_idx(jit_conv_conf_t &jcp,
        const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, const memory_desc_wrapper &dst_d,
        const primitive_attr_t &attr) {

    ck_conv_prb_desc_t prb;
    prb.mb = jcp.mb;
    prb.oc = jcp.oc;
    prb.ic = jcp.ic;
    prb.kh = jcp.kh;
    prb.kw = jcp.kw;
    prb.ih = jcp.ih;
    prb.iw = jcp.iw;
    prb.sh = jcp.stride_h;
    prb.sw = jcp.stride_w;
    prb.dh = jcp.dilate_h+1;
    prb.dw = jcp.dilate_w+1;
    prb.ph = jcp.t_pad;
    prb.pw = jcp.l_pad;

    zendnnInfo(ZENDNN_CORELOG,
        "ZENDNN implementation path in ck_conv_fwd_kernel_f32::init_ck_idx [cpu/convolution]");
    zendnnInfo(ZENDNN_CORELOG,
        " jcp.mb=", jcp.mb,
        " jcp.oc=", jcp.oc,
        " jcp.ic=", jcp.ic,
        " jcp.kh=", jcp.kh,
        " jcp.kw=", jcp.kw,
        " jcp.ih=", jcp.ih,
        " jcp.iw=", jcp.iw,
        " jcp.stride_h=", jcp.stride_h,
        " jcp.stride_w=", jcp.stride_w,
        " jcp.dilate_h=", jcp.dilate_h,
        " jcp.dilate_w=", jcp.dilate_w,
        " jcp.t_pad=", jcp.t_pad,
        " jcp.l_pad=", jcp.l_pad,
        " [cpu/convolution]");

    auto found_obj = global_ck_idx_cache.find(prb);

    if (found_obj != global_ck_idx_cache.end()) {
        // prb is in global_ck_idx_cache
        jcp.ck_fastest_kernel_idx = global_ck_idx_cache[prb];
        return status::success;
    } else {
        // prb is not in global_ck_idx_cache
        return status::unimplemented;
    }

}

void ck_conv_fwd_kernel_f32::init_scratchpad(
    memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp) {
    if (jcp.with_bias && jcp.oc != jcp.oc_without_padding) {
        scratchpad.book<float>(key_conv_padded_bias, jcp.oc);
    }
}

void ck_conv_fwd_kernel_f32::generate() {}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace zendnn

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
