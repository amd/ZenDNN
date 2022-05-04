/*******************************************************************************
* Modifications Copyright (c) 2021-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "cpu/x64/zendnn_convolution.hpp"
#include "zendnn_logging.hpp"

namespace zendnn {
namespace impl {
namespace cpu {
namespace x64 {

using namespace zendnn::impl::status;
using namespace zendnn::impl::memory_tracking::names;
using namespace zendnn::impl::utils;
using namespace nstl;

void zendnn_convolution_fwd_t::execute_forward(const exec_ctx_t &ctx) const {
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
    //FIXME: what about BatchNorm fusion
    const memory_desc_wrapper batchNormScale_d(pd()->weights_md(2));
    const memory_desc_wrapper batchNormMean_d(pd()->weights_md(3));
    const memory_desc_wrapper batchNormOffset_d(pd()->weights_md(4));

    zendnnInfo(ZENDNN_CORELOG,
               "ZENDNN implementation path in zendnn_convolution_fwd_t::execute_forward [cpu/convolution]");
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

    if (total_filters == jcp.oc) {
        concat = false;
    }

    //TBD: To add support for gemm, ref, direct, winograd, fft
    //we need to move else part to [ZENDNN ALGO] code
    if (jcp.alg_kind == zendnn_convolution_ref) {
        if ((jcp.reluFused == false) &&
                (jcp.batchNormFused == true)) {
            //Only BatchNorm fused with conv
            zendnnInfo(ZENDNN_CORELOG,
                       "zendnn_convolution_fwd_t::execute_forward zenConvolution2DwithBatchNormRef [cpu/convolution]");
            zenConvolution2DwithBatchNormRef(
                (float *)src,
                jcp.mb,
                jcp.ic,
                jcp.ih,
                jcp.iw,
                weights,
                jcp.oc,
                jcp.kh,
                jcp.kw,
                jcp.t_pad,
                jcp.l_pad,
                jcp.b_pad,
                jcp.r_pad,
                jcp.stride_h,
                jcp.stride_w,
                (float *)batchNormScale,
                (float *)batchNormMean,
                (float *)batchNormOffset,
                (float *)dst,
                jcp.oh,
                jcp.ow
            );
        }
        else if ((jcp.reluFused == true) &&
                 (jcp.batchNormFused == true)) {
            //ReLU and  BatchNorm fused with conv
            zendnnInfo(ZENDNN_CORELOG,
                       "zendnn_convolution_fwd_t::execute_forward zenConvolution2DwithBatchNormReluRef [cpu/convolution]");
            zenConvolution2DwithBatchNormReluRef(
                (float *)src,
                jcp.mb,
                jcp.ic,
                jcp.ih,
                jcp.iw,
                weights,
                jcp.oc,
                jcp.kh,
                jcp.kw,
                jcp.t_pad,
                jcp.l_pad,
                jcp.b_pad,
                jcp.r_pad,
                jcp.stride_h,
                jcp.stride_w,
                (float *)batchNormScale,
                (float *)batchNormMean,
                (float *)batchNormOffset,
                (float *)dst,
                jcp.oh,
                jcp.ow
            );
        }
        else if (bias != NULL && !jcp.with_eltwise) {
            zendnnInfo(ZENDNN_CORELOG,
                       "zendnn_convolution_fwd_t::execute_forward zenConvolution2DwithBiasRef [cpu/convolution]");
            zenConvolution2DwithBiasRef(
                (float *)src,
                jcp.mb,
                jcp.ic,
                jcp.ih,
                jcp.iw,
                weights,
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
                jcp.ow
            );
        }
        else if (bias != NULL && jcp.with_eltwise) {
            zendnnInfo(ZENDNN_CORELOG,
                       "zendnn_convolution_fwd_t::execute_forward zenConvolution2DwithBiasReluRef [cpu/convolution]");
            zenConvolution2DwithBiasReluRef(
                (float *)src,
                jcp.mb,
                jcp.ic,
                jcp.ih,
                jcp.iw,
                weights,
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
                jcp.ow
            );
        }
        else {
            zendnnInfo(ZENDNN_CORELOG,
                       "zendnn_convolution_fwd_t::execute_forward zenConvolution2DRef [cpu/convolution]");
            zenConvolution2DRef(
                (float *)src,
                jcp.mb,
                jcp.ic,
                jcp.ih,
                jcp.iw,
                weights,
                jcp.oc,
                jcp.kh,
                jcp.kw,
                jcp.t_pad,
                jcp.l_pad,
                jcp.b_pad,
                jcp.r_pad,
                jcp.stride_h,
                jcp.stride_w,
                (float *)dst,
                jcp.oh,
                jcp.ow
            );
        }
    }
    else {
        if ((jcp.reluFused == false) &&
                (jcp.batchNormFused == true)) {
            //Only BatchNorm fused with conv
            zendnnInfo(ZENDNN_CORELOG,
                       "zendnn_convolution_fwd_t::execute_forward zenConvolution2DwithBatchNorm [cpu/convolution]");
            zenConvolution2DwithBatchNorm(
                (float *)src,
                jcp.mb,
                jcp.ic,
                jcp.ih,
                jcp.iw,
                weights,
                jcp.oc,
                jcp.kh,
                jcp.kw,
                jcp.t_pad,
                jcp.l_pad,
                jcp.b_pad,
                jcp.r_pad,
                jcp.stride_h,
                jcp.stride_w,
                (float *)batchNormScale,
                (float *)batchNormMean,
                (float *)batchNormOffset,
                (float *)dst,
                jcp.oh,
                jcp.ow,
                concat,
                filter_offset,
                total_filters
            );
        }
        else if ((jcp.reluFused == true) &&
                 (jcp.batchNormFused == true)) {
            //ReLU and  BatchNorm fused with conv
            zendnnInfo(ZENDNN_CORELOG,
                       "zendnn_convolution_fwd_t::execute_forward zenConvolution2DwithBatchNormRelu [cpu/convolution]");
            zenConvolution2DwithBatchNormRelu(
                (float *)src,
                jcp.mb,
                jcp.ic,
                jcp.ih,
                jcp.iw,
                weights,
                jcp.oc,
                jcp.kh,
                jcp.kw,
                jcp.t_pad,
                jcp.l_pad,
                jcp.b_pad,
                jcp.r_pad,
                jcp.stride_h,
                jcp.stride_w,
                (float *)batchNormScale,
                (float *)batchNormMean,
                (float *)batchNormOffset,
                (float *)dst,
                jcp.oh,
                jcp.ow,
                concat,
                filter_offset,
                total_filters
            );
        }
        else if ((jcp.reluFused == true)) {
            //ReLU fused with conv
            zendnnInfo(ZENDNN_CORELOG,
                       "zendnn_convolution_fwd_t::execute_forward zenConvolution2DwithRelu [cpu/convolution]");
            zenConvolution2DwithBiasRelu(
                (float *)src,
                jcp.mb,
                jcp.ic,
                jcp.ih,
                jcp.iw,
                weights,
                jcp.oc,
                jcp.kh,
                jcp.kw,
                jcp.t_pad,
                jcp.l_pad,
                jcp.b_pad,
                jcp.r_pad,
                jcp.stride_h,
                jcp.stride_w,
                NULL,
                (float *)dst,
                jcp.oh,
                jcp.ow,
                concat,
                filter_offset,
                total_filters
            );

        }
        else if (bias != NULL && !jcp.with_eltwise) {
            if (!jcp.with_sum) {
                zendnnInfo(ZENDNN_CORELOG,
                           "zendnn_convolution_fwd_t::execute_forward zenConvolution2DwithBias [cpu/convolution]");
                zenConvolution2DwithBias(
                    (float *)src,
                    jcp.mb,
                    jcp.ic,
                    jcp.ih,
                    jcp.iw,
                    weights,
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
                    total_filters
                );
            }
            else {
                zendnnInfo(ZENDNN_CORELOG,
                           "zendnn_convolution_fwd_t::execute_forward zenConvolution2DwithBiasSum [cpu/convolution]");
                zenConvolution2DwithBiasSum(
                    (float *)src,
                    jcp.mb,
                    jcp.ic,
                    jcp.ih,
                    jcp.iw,
                    weights,
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
                    total_filters
                );
            }
        }
        else if (bias != NULL && jcp.with_eltwise) {
            if (!jcp.with_sum) {
                //Only ReLU fused with conv
                zendnnInfo(ZENDNN_CORELOG,
                           "zendnn_convolution_fwd_t::execute_forward zenConvolution2DwithBiasRelu [cpu/convolution]");
                zenConvolution2DwithBiasRelu(
                    (float *)src,
                    jcp.mb,
                    jcp.ic,
                    jcp.ih,
                    jcp.iw,
                    weights,
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
                    total_filters
                );
            }
            else {
                zendnnInfo(ZENDNN_CORELOG,
                           "zendnn_convolution_fwd_t::execute_forward zenConvolution2DwithBiasSumRelu [cpu/convolution]");
                zenConvolution2DwithBiasSumRelu(
                    (float *)src,
                    jcp.mb,
                    jcp.ic,
                    jcp.ih,
                    jcp.iw,
                    weights,
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
                    total_filters
                );
            }
        }
        else {
            //Convolution
            zendnnInfo(ZENDNN_CORELOG,
                       "zendnn_convolution_fwd_t::execute_forward zenConvolution2D [cpu/convolution]");
            zenConvolution2D(
                (float *)src,
                jcp.mb,
                jcp.ic,
                jcp.ih,
                jcp.iw,
                weights,
                jcp.oc,
                jcp.kh,
                jcp.kw,
                jcp.t_pad,
                jcp.l_pad,
                jcp.b_pad,
                jcp.r_pad,
                jcp.stride_h,
                jcp.stride_w,
                (float *)dst,
                jcp.oh,
                jcp.ow,
                concat,
                filter_offset,
                total_filters
            );
        }
    }

    if (pd()->wants_zero_pad_dst()) {
        ctx.zero_pad_output(ZENDNN_ARG_DST);
    }
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace zendnn

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
