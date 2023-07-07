/*******************************************************************************
* Modifications Copyright (c) 2021-2023 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Copyright 2016-2019 Intel Corporation
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
#include "common/zendnn_thread.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "common/zendnn_private.hpp"

#include "cpu/x64/zendnn_lpgemm_convolution.hpp"
#ifdef ZENDNN_ENABLE_LPGEMM_CONV
    #include "cpu/x64/zendnn_lpgemm_utils.hpp"
#endif
#include "zendnn_logging.hpp"
#include <type_traits>

using namespace zendnn;

#ifdef ZENDNN_ENABLE_LPGEMM_CONV
// Direct Convolution
// for Auto-tuner
template <typename T, typename K> void convolution_ref_direct(engine eng, K src,
        int batch, int channel, int height, int width,
        int8_t *weights, int no_of_filter, int kernel_h,
        int kernel_w, int pad_h,
        int pad_w, int stride_h, int stride_w, int32_t *bias,
        T dst, int out_height, int out_width, bool reluFused, float *output_scales,
        int scale_size) {

    using tag = memory::format_tag;
    using dt = memory::data_type;
    auto dtype = dt::s8;
    if (std::is_same<T, int32_t>::value) {
        dtype = dt::s32;
    }
    auto stype = dt::u8;
    if (std::is_same<K, int8_t>::value) {
        dtype = dt::s8;
    }

    stream s(eng);

    memory::dims conv1_src_tz = {batch, channel, height, width};
    memory::dims conv1_weights_tz = {no_of_filter, channel, kernel_h, kernel_w};
    memory::dims conv1_bias_tz = {no_of_filter};
    memory::dims conv1_dst_tz = {batch, no_of_filter, out_height, out_width};
    memory::dims conv1_strides = {stride_h, stride_w};
    memory::dims conv1_padding = {pad_h, pad_w};

    memory user_src_memory, user_weights_memory, conv1_user_bias_memory,
           conv1_dst_memory;

    user_src_memory = memory({{conv1_src_tz}, stype, tag::nhwc}, eng, src);
    user_weights_memory = memory({{conv1_weights_tz}, dt::s8, tag::hwcn},
    eng, weights);
    conv1_user_bias_memory = memory({{conv1_bias_tz}, dt::s32, tag::x}, eng, bias);
    conv1_dst_memory = memory({{conv1_dst_tz}, dtype, tag::nhwc }, eng, dst);

    auto conv1_src_md = memory::desc({conv1_src_tz}, stype, tag::acdb);
    auto conv1_bias_md = memory::desc({conv1_bias_tz}, dt::s32, tag::x);
    auto conv1_weights_md = memory::desc({conv1_weights_tz}, dt::s8, tag::hwcn);
    auto conv1_dst_md = memory::desc({conv1_dst_tz}, dtype, tag::acdb);
    auto conv1_desc = convolution_forward::desc(prop_kind::forward_inference,
                      algorithm::convolution_direct, conv1_src_md, conv1_weights_md,
                      conv1_bias_md, conv1_dst_md, conv1_strides, conv1_padding,
                      conv1_padding);

    zendnn::primitive_attr conv_attr;
    zendnn::post_ops post_ops;
    float relu_scale = 1.0f;
    bool relu_alpha = true;
    std::vector<float> output_scales_vector {output_scales, output_scales + scale_size};


    if (reluFused) {
        //post_ops.append_binary(zendnn::algorithm::binary_add, conv1_dst_memory.get_desc());
        post_ops.append_eltwise(1.0f, zendnn::algorithm::eltwise_relu, 0.0, 0.0f);
        //conv_attr.set_post_ops(post_ops);
        //attr.set_post_ops(conv_post_ops);
    }

    /*
    if (reluFused) {
        auto relu_algo = zendnn::algorithm::eltwise_relu;
        if (relu_alpha) {
            relu_algo = zendnn::algorithm::eltwise_bounded_relu;
            relu_alpha = 6.0f * std::pow(2, output_scales[0]);
        }
        post_ops.append_eltwise(relu_scale, relu_algo, relu_alpha, 0.0f);
    }
    else if (relu_scale != 1.0f) {
        post_ops.append_eltwise(relu_scale, zendnn::algorithm::eltwise_linear, 1.0f,
                                0.0f);
    }*/

    conv_attr.set_output_scales(0, output_scales_vector);
    conv_attr.set_post_ops(post_ops);

    auto conv1_prim_desc = convolution_forward::primitive_desc(conv1_desc,
                           conv_attr, eng);
    auto conv1_src_memory = user_src_memory;
    auto conv1_weights_memory = user_weights_memory;


    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;

    net.push_back(convolution_forward(conv1_prim_desc));
    net_args.push_back({{ZENDNN_ARG_SRC, conv1_src_memory},
        {ZENDNN_ARG_WEIGHTS, conv1_weights_memory},
        {ZENDNN_ARG_BIAS, conv1_user_bias_memory},
        {ZENDNN_ARG_DST, conv1_dst_memory}
    });

    assert(net.size() == net_args.size() && "something is missing");
    for (size_t i = 0; i < net.size(); ++i) {
        net.at(i).execute(s, net_args.at(i));
    }
    s.wait();

}
#endif

namespace zendnn {
namespace impl {
namespace cpu {
namespace x64 {

using namespace zendnn::impl::status;
using namespace zendnn::impl::memory_tracking::names;
using namespace zendnn::impl::utils;
using namespace nstl;

int isSupportedAutoPath(int lpgemm_auto_type, alg_kind_t alg_kind) {
    // 0 = direct path with s8 output
    if (lpgemm_auto_type == 2 && (alg_kind == zendnn_convolution_gemm_u8s8s16os8 ||
                                  alg_kind == zendnn_convolution_gemm_u8s8s32os8)) {
        return 0;
    }
    // 1 = direct path with s32 output
    else if (lpgemm_auto_type == 2 &&
             alg_kind == zendnn_convolution_gemm_u8s8s32os32) {
        return 1;
    }
    else if (lpgemm_auto_type == 2 &&
             alg_kind == zendnn_convolution_gemm_s8s8s32os8) {
        return 2;
    }
    else if (lpgemm_auto_type == 2 &&
             alg_kind == zendnn_convolution_gemm_s8s8s32os32) {
        return 3;
    }
    // All cases with lpgemm_auto_type = 1 are valid
    else if (lpgemm_auto_type == 1) {
        return 4;
    }
    // -1 means the conditions are not supported
    // This include cases when lpgemm_auto_type = 2 (direct path) and
    // alg_kind is set to zendnn_convolution_gemm_u8s8s16os16 (dst as s16) or bf16 APIs
    return -1;
}

void zendnn_lpgemm_convolution_fwd_t::execute_forward(const exec_ctx_t &ctx)
const {
#ifdef ZENDNN_ENABLE_LPGEMM_CONV
    const auto &jcp = kernel_->jcp;
    auto src = CTX_IN_MEM(const data_t *, ZENDNN_ARG_SRC);
    auto weights = CTX_IN_MEM(const data_t *, ZENDNN_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const data_t *, ZENDNN_ARG_BIAS);
    auto dst = CTX_OUT_MEM(data_t *, ZENDNN_ARG_DST);
    auto batchNormScale = CTX_IN_MEM(const data_t *, ZENDNN_ARG_BN_SCALE);
    auto batchNormMean = CTX_IN_MEM(const data_t *, ZENDNN_ARG_BN_MEAN);
    auto batchNormOffset = CTX_IN_MEM(const data_t *, ZENDNN_ARG_BN_OFFSET);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper bias_d(pd()->weights_md(1));
    const memory_desc_wrapper batchNormScale_d(pd()->weights_md(2));
    const memory_desc_wrapper batchNormMean_d(pd()->weights_md(3));
    const memory_desc_wrapper batchNormOffset_d(pd()->weights_md(4));

    zendnnInfo(ZENDNN_CORELOG,
               "LPGEMM ZENDNN implementation path in zendnn_lpgemm_convolution_fwd_t::execute_forward [cpu/convolution]");
    zendnnInfo(ZENDNN_CORELOG, "algo=", jcp.alg_kind, " mb=",jcp.mb, " ih=",jcp.ih,
               " iw=",jcp.iw,
               " id=",jcp.id, " oh=",jcp.oh, " ow=",jcp.ow, " od=",jcp.od, " kh=",jcp.kh,
               " kw=",jcp.kw, " kd=",jcp.kd, " stride_h=",jcp.stride_h,
               " stride_w=",jcp.stride_w, " l_pad=",jcp.l_pad, " t_pad=",jcp.t_pad,
               " f_pad=",jcp.f_pad, " ngroups=",jcp.ngroups, " ic=",jcp.ic, " oc=",jcp.oc,
               " [cpu/convolution]");

    int filter_offset = pd()->dst_md()->offset0;
    int total_filters = pd()->dst_md()->format_desc.blocking.strides[3];
    bool concat = true;

    float *output_scales {nullptr};
    output_scales = pd()->attr()->output_scales_.scales_;
    int scale_size = pd()->attr()->output_scales_.count_;
    const int *zero_point_dst {nullptr};
    zero_point_dst = pd()->attr()->zero_points_.get(ZENDNN_ARG_DST);

    int elementwise_index =  pd()->attr()->post_ops_.find(primitive_kind::eltwise);
    bool has_eltwise_relu = elementwise_index>=0 ?
                            pd()->attr()->post_ops_.entry_[elementwise_index].eltwise.alg ==
                            alg_kind::eltwise_relu : 0;

    bool has_eltwise_gelu = elementwise_index>=0 ?
                            pd()->attr()->post_ops_.entry_[elementwise_index].eltwise.alg ==
                            alg_kind::eltwise_gelu : 0;

    int elementwiseType = 0;
    // 1 -> ReLU
    // 2 -> GeLU tanh
    // 3 -> GeLU erf
    if (has_eltwise_relu) {
        elementwiseType = 1;
    }
    else if (has_eltwise_gelu) {
        elementwiseType = 2;
    }

    if (total_filters == jcp.oc) {
        concat = false;
    }

    // ZENDNN_LPGEMM_AUTO_TYPE (for auto-tuner)
    // 1 = LPGEMM path is taken (default)
    // 2 = DIRECT path is taken
    int lpgemm_auto_type = zendnn_getenv_int("ZENDNN_LPGEMM_AUTO_TYPE", 1);

    zendnnInfo(ZENDNN_CORELOG,
               "zendnn_lpgemm_convolution_fwd_t::execute_forward zenConvolution2D [cpu/convolution]");

    int supportedPath = isSupportedAutoPath(lpgemm_auto_type, jcp.alg_kind);

    if (supportedPath == -1) {
        zendnnInfo(ZENDNN_CORELOG,
                   "ZENDNN_LPGEMM_AUTO_TYPE is set to 2 (direct path) but no valid path found for given algotype [cpu/convolution]");
        exit(0);
    }
    else if (supportedPath < 4) {
        char **cpu = NULL;
        engine::kind engine_kind = parse_engine_kind(1, cpu);
        engine eng(engine_kind, 0);
        stream s(eng);


        convolution_ref_direct(eng,
                               (supportedPath<2)?(uint8_t *)src:(true? (int8_t *)src: (void *)src), jcp.mb,
                               jcp.ic, jcp.ih, jcp.iw,
                               (int8_t *)weights,
                               jcp.oc, jcp.kh, jcp.kw, jcp.t_pad, jcp.l_pad, jcp.stride_h, jcp.stride_w,
                               (int32_t *)bias, (supportedPath==0 ||
                                       supportedPath==2)?(int8_t *)dst:(true? (int32_t *)dst:
                                               (void *)dst), jcp.oh, jcp.ow, jcp.reluFused, output_scales, scale_size);
    }
    else if (jcp.alg_kind == zendnn_convolution_gemm_u8s8s16os16) {
        zenConvolution2D_u8s8s16os16(
            (uint8_t *)src,
            jcp.mb,
            jcp.ic,
            jcp.ih,
            jcp.iw,
            (int8_t *)weights,
            jcp.oc,
            jcp.kh,
            jcp.kw,
            jcp.t_pad,
            jcp.l_pad,
            jcp.b_pad,
            jcp.r_pad,
            jcp.stride_h,
            jcp.stride_w,
            (int16_t *)bias,
            (int16_t *)dst,
            jcp.oh,
            jcp.ow,
            concat,
            filter_offset,
            total_filters,
            jcp.reluFused,
            output_scales
        );
    }
    else if (jcp.alg_kind == zendnn_convolution_gemm_u8s8s16os8) {
        zenConvolution2D_u8s8s16os8(
            (uint8_t *)src,
            jcp.mb,
            jcp.ic,
            jcp.ih,
            jcp.iw,
            (int8_t *)weights,
            jcp.oc,
            jcp.kh,
            jcp.kw,
            jcp.t_pad,
            jcp.l_pad,
            jcp.b_pad,
            jcp.r_pad,
            jcp.stride_h,
            jcp.stride_w,
            (int16_t *)bias,
            (int8_t *)dst,
            jcp.oh,
            jcp.ow,
            concat,
            filter_offset,
            total_filters,
            jcp.reluFused,
            output_scales,
            zero_point_dst,
            scale_size
        );
    }
    else if (jcp.alg_kind == zendnn_convolution_gemm_bf16bf16f32of32) {
        zenConvolution2D_bf16bf16f32of32(
            (int16_t *)src,
            jcp.mb,
            jcp.ic,
            jcp.ih,
            jcp.iw,
            (int16_t *)weights,
            jcp.oc,
            jcp.kh,
            jcp.kw,
            jcp.t_pad,
            jcp.l_pad,
            jcp.b_pad,
            jcp.r_pad,
            jcp.stride_h,
            jcp.stride_w,
            (float *)bias,
            (float *)dst,
            jcp.oh,
            jcp.ow,
            concat,
            filter_offset,
            total_filters,
            jcp.reluFused,
            output_scales
        );
    }
    else if (jcp.alg_kind == zendnn_convolution_gemm_bf16bf16f32obf16) {
        zenConvolution2D_bf16bf16f32obf16(
            (int16_t *)src,
            jcp.mb,
            jcp.ic,
            jcp.ih,
            jcp.iw,
            (int16_t *)weights,
            jcp.oc,
            jcp.kh,
            jcp.kw,
            jcp.t_pad,
            jcp.l_pad,
            jcp.b_pad,
            jcp.r_pad,
            jcp.stride_h,
            jcp.stride_w,
            (float *)bias,
            (int16_t *)dst,
            jcp.oh,
            jcp.ow,
            concat,
            filter_offset,
            total_filters,
            jcp.reluFused,
            output_scales,
            zero_point_dst,
            scale_size
        );
    }
    else if (jcp.alg_kind == zendnn_convolution_gemm_u8s8s32os32) {

        zenConvolution2D_u8s8s32os32(
            (uint8_t *)src,
            jcp.mb,
            jcp.ic,
            jcp.ih,
            jcp.iw,
            (int8_t *)weights,
            jcp.oc,
            jcp.kh,
            jcp.kw,
            jcp.t_pad,
            jcp.l_pad,
            jcp.b_pad,
            jcp.r_pad,
            jcp.stride_h,
            jcp.stride_w,
            (int32_t *)bias,
            (int32_t *)dst,
            jcp.oh,
            jcp.ow,
            concat,
            filter_offset,
            total_filters,
            jcp.reluFused,
            output_scales
        );
    }
    else if (jcp.alg_kind == zendnn_convolution_gemm_u8s8s32os8) {

        zenConvolution2D_u8s8s32os8(
            (uint8_t *)src,
            jcp.mb,
            jcp.ic,
            jcp.ih,
            jcp.iw,
            (int8_t *)weights,
            jcp.oc,
            jcp.kh,
            jcp.kw,
            jcp.t_pad,
            jcp.l_pad,
            jcp.b_pad,
            jcp.r_pad,
            jcp.stride_h,
            jcp.stride_w,
            (int32_t *)bias,
            (int8_t *)dst,
            jcp.oh,
            jcp.ow,
            concat,
            filter_offset,
            total_filters,
            jcp.reluFused,
            output_scales,
            zero_point_dst,
            scale_size
        );
    }
    else if (jcp.alg_kind == zendnn_convolution_gemm_s8s8s32os32) {

        zenConvolution2D_s8s8s32os32(
            (int8_t *)src,
            jcp.mb,
            jcp.ic,
            jcp.ih,
            jcp.iw,
            (int8_t *)weights,
            jcp.oc,
            jcp.kh,
            jcp.kw,
            jcp.t_pad,
            jcp.l_pad,
            jcp.b_pad,
            jcp.r_pad,
            jcp.stride_h,
            jcp.stride_w,
            (int32_t *)bias,
            (int32_t *)dst,
            jcp.oh,
            jcp.ow,
            concat,
            filter_offset,
            total_filters,
            jcp.reluFused,
            elementwiseType,
            output_scales
        );
    }
    else if (jcp.alg_kind == zendnn_convolution_gemm_s8s8s32os8) {

        zenConvolution2D_s8s8s32os8(
            (int8_t *)src,
            jcp.mb,
            jcp.ic,
            jcp.ih,
            jcp.iw,
            (int8_t *)weights,
            jcp.oc,
            jcp.kh,
            jcp.kw,
            jcp.t_pad,
            jcp.l_pad,
            jcp.b_pad,
            jcp.r_pad,
            jcp.stride_h,
            jcp.stride_w,
            (int32_t *)bias,
            (int8_t *)dst,
            jcp.oh,
            jcp.ow,
            concat,
            filter_offset,
            total_filters,
            jcp.reluFused,
            elementwiseType,
            output_scales,
            zero_point_dst,
            scale_size
        );
    }
    else if (jcp.alg_kind == zendnn_convolution_gemm_s8s8s16os16) {

        zenConvolution2D_s8s8s16os16(
            (int8_t *)src,
            jcp.mb,
            jcp.ic,
            jcp.ih,
            jcp.iw,
            (int8_t *)weights,
            jcp.oc,
            jcp.kh,
            jcp.kw,
            jcp.t_pad,
            jcp.l_pad,
            jcp.b_pad,
            jcp.r_pad,
            jcp.stride_h,
            jcp.stride_w,
            (int16_t *)bias,
            (int16_t *)dst,
            jcp.oh,
            jcp.ow,
            concat,
            filter_offset,
            total_filters,
            jcp.reluFused,
            elementwiseType,
            output_scales
        );
    }
    else if (jcp.alg_kind == zendnn_convolution_gemm_s8s8s16os8) {

        zenConvolution2D_s8s8s16os8(
            (int8_t *)src,
            jcp.mb,
            jcp.ic,
            jcp.ih,
            jcp.iw,
            (int8_t *)weights,
            jcp.oc,
            jcp.kh,
            jcp.kw,
            jcp.t_pad,
            jcp.l_pad,
            jcp.b_pad,
            jcp.r_pad,
            jcp.stride_h,
            jcp.stride_w,
            (int16_t *)bias,
            (int8_t *)dst,
            jcp.oh,
            jcp.ow,
            concat,
            filter_offset,
            total_filters,
            jcp.reluFused,
            elementwiseType,
            output_scales,
            zero_point_dst,
            scale_size
        );
    }
    if (pd()->wants_zero_pad_dst()) {
        ctx.zero_pad_output(ZENDNN_ARG_DST);
    }
#endif
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace zendnn
// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
