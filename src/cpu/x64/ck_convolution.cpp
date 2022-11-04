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
#ifdef ENABLE_CK

#include "common/c_types_map.hpp"
#include "common/zendnn_thread.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "common/zendnn_private.hpp"

#include "cpu/x64/ck_convolution.hpp"
#include "zendnn_logging.hpp"

#define AVX2_DATA_ALIGNMENT 32

#define TEST_FUSION_PASSTHROUGH 0
#define TEST_FUSION_RELU 1
#define TEST_FUSION TEST_FUSION_PASSTHROUGH

#define TEST_LAYOUT_NHWC_KYXC_NHWK 0
#define TEST_LAYOUT_NHWC_KYXCK8_NHWK 1
#define TEST_LAYOUT_NHWC_YXCK_NHWK 2
#define TEST_LAYOUT TEST_LAYOUT_NHWC_KYXCK8_NHWK

using PassThrough = ck::tensor_operation::cpu::element_wise::PassThrough;
using DeviceConvFwdNoOpPtr = ck::tensor_operation::cpu::device::
                             DeviceConvFwdPtr<PassThrough, PassThrough, PassThrough>;

namespace ck {
namespace tensor_operation {
namespace cpu {
namespace device {
namespace device_conv2d_fwd_avx2_instance {

using PassThrough = ck::tensor_operation::cpu::element_wise::PassThrough;
using Relu        = ck::tensor_operation::cpu::element_wise::Relu;

// ------------------ nhwc-kyxc-nhwk
void add_device_conv2d_fwd_avx2_nhwc_kyxc_nhwk(
    std::vector<DeviceConvFwdPtr<PassThrough, PassThrough, PassThrough>> &instances);

void add_device_conv2d_fwd_avx2_nhwc_kyxc_nhwk_local_c(
    std::vector<DeviceConvFwdPtr<PassThrough, PassThrough, PassThrough>> &instances);

void add_device_conv2d_fwd_avx2_nhwc_kyxc_nhwk_mt(
    std::vector<DeviceConvFwdPtr<PassThrough, PassThrough, PassThrough>> &instances);

void add_device_conv2d_fwd_avx2_nhwc_kyxc_nhwk_relu(
    std::vector<DeviceConvFwdPtr<PassThrough, PassThrough, Relu>> &instances);

void add_device_conv2d_fwd_avx2_nhwc_kyxc_nhwk_local_c_relu(
    std::vector<DeviceConvFwdPtr<PassThrough, PassThrough, Relu>> &instances);

void add_device_conv2d_fwd_avx2_nhwc_kyxc_nhwk_mt_relu(
    std::vector<DeviceConvFwdPtr<PassThrough, PassThrough, Relu>> &instances);

// ------------------ nhwc-kcyxk8-nhwk
void add_device_conv2d_fwd_avx2_nhwc_kyxck8_nhwk(
    std::vector<DeviceConvFwdPtr<PassThrough, PassThrough, PassThrough>> &instances);

void add_device_conv2d_fwd_avx2_nhwc_kyxck8_nhwk_local_c(
    std::vector<DeviceConvFwdPtr<PassThrough, PassThrough, PassThrough>> &instances);

void add_device_conv2d_fwd_avx2_nhwc_kyxck8_nhwk_mt(
    std::vector<DeviceConvFwdPtr<PassThrough, PassThrough, PassThrough>> &instances);

void add_device_conv2d_fwd_avx2_nhwc_kyxck8_nhwk_relu(
    std::vector<DeviceConvFwdPtr<PassThrough, PassThrough, Relu>> &instances);

void add_device_conv2d_fwd_avx2_nhwc_kyxck8_nhwk_local_c_relu(
    std::vector<DeviceConvFwdPtr<PassThrough, PassThrough, Relu>> &instances);

void add_device_conv2d_fwd_avx2_nhwc_kyxck8_nhwk_mt_relu(
    std::vector<DeviceConvFwdPtr<PassThrough, PassThrough, Relu>> &instances);

void add_device_conv2d_fwd_avx2_nhwc_yxck_nhwk(
    std::vector<DeviceConvFwdPtr<PassThrough, PassThrough, PassThrough>> &instances);

// ------------------ nhwc-yxck-nhwk
void add_device_conv2d_fwd_avx2_nhwc_yxck_nhwk_local_c(
    std::vector<DeviceConvFwdPtr<PassThrough, PassThrough, PassThrough>> &instances);

void add_device_conv2d_fwd_avx2_nhwc_yxck_nhwk_mt(
    std::vector<DeviceConvFwdPtr<PassThrough, PassThrough, PassThrough>> &instances);

void add_device_conv2d_fwd_avx2_nhwc_yxck_nhwk_relu(
    std::vector<DeviceConvFwdPtr<PassThrough, PassThrough, Relu>> &instances);

void add_device_conv2d_fwd_avx2_nhwc_yxck_nhwk_local_c_relu(
    std::vector<DeviceConvFwdPtr<PassThrough, PassThrough, Relu>> &instances);

void add_device_conv2d_fwd_avx2_nhwc_yxck_nhwk_mt_relu(
    std::vector<DeviceConvFwdPtr<PassThrough, PassThrough, Relu>> &instances);

// ------------------ direct-conv nhwc-kcyxk8-nhwk
void add_device_conv2d_direct_fwd_avx2_nhwc_kyxck8_nhwk(
    std::vector<DeviceConvFwdPtr<PassThrough, PassThrough, PassThrough>> &instances);

} // namespace device_conv2d_fwd_avx2_instance
} // namespace device
} // namespace cpu
} // namespace tensor_operation
} // namespace ck

namespace zendnn {
namespace impl {
namespace cpu {
namespace x64 {

using PassThrough = ck::tensor_operation::cpu::element_wise::PassThrough;
using DeviceConvFwdNoOpPtr = ck::tensor_operation::cpu::device::
                             DeviceConvFwdPtr<PassThrough, PassThrough, PassThrough>;

status_t ck_convolution_fwd_t::add_device_conv_ptrs(
    std::vector<DeviceConvFwdNoOpPtr> &conv_ptrs,
    float input_type, float wei_type, float out_type) {
    zendnnInfo(ZENDNN_CORELOG,
               "ZENDNN implementation path in ck_convolution_fwd_t::add_device_conv_ptrs (1) [cpu/convolution]");
    using InDataType  = decltype(input_type);
    using WeiDataType = decltype(wei_type);
    using OutDataType = decltype(out_type);

    if constexpr(ck::is_same_v<ck::remove_cv_t<InDataType>, float> &&
                 ck::is_same_v<ck::remove_cv_t<WeiDataType>, float> &&
                 ck::is_same_v<ck::remove_cv_t<OutDataType>, float>) {
#if TEST_LAYOUT == TEST_LAYOUT_NHWC_KYXC_NHWK
#if TEST_FUSION == TEST_FUSION_PASSTHROUGH
        if (omp_get_max_threads() > 1) {
            ck::tensor_operation::cpu::device::device_conv2d_fwd_avx2_instance::
            add_device_conv2d_fwd_avx2_nhwc_kyxc_nhwk_mt(conv_ptrs);
            ck::tensor_operation::cpu::device::device_conv2d_fwd_avx2_instance::
            add_device_conv2d_fwd_avx2_nhwc_kyxc_nhwk(conv_ptrs);
        }
        else {
            ck::tensor_operation::cpu::device::device_conv2d_fwd_avx2_instance::
            add_device_conv2d_fwd_avx2_nhwc_kyxc_nhwk(conv_ptrs);
        }
#endif
#if TEST_FUSION == TEST_FUSION_RELU
        if (omp_get_max_threads() > 1) {
            ck::tensor_operation::cpu::device::device_conv2d_fwd_avx2_instance::
            add_device_conv2d_fwd_avx2_nhwc_kyxc_nhwk_mt_relu(conv_ptrs);
            ck::tensor_operation::cpu::device::device_conv2d_fwd_avx2_instance::
            add_device_conv2d_fwd_avx2_nhwc_kyxc_nhwk_relu(conv_ptrs);
        }
        else {
            ck::tensor_operation::cpu::device::device_conv2d_fwd_avx2_instance::
            add_device_conv2d_fwd_avx2_nhwc_kyxc_nhwk_relu(conv_ptrs);
        }
#endif
#endif
#if TEST_LAYOUT == TEST_LAYOUT_NHWC_KYXCK8_NHWK
#if TEST_FUSION == TEST_FUSION_PASSTHROUGH
        if (omp_get_max_threads() > 1) {
            ck::tensor_operation::cpu::device::device_conv2d_fwd_avx2_instance::
            add_device_conv2d_fwd_avx2_nhwc_kyxck8_nhwk_mt(conv_ptrs);
            ck::tensor_operation::cpu::device::device_conv2d_fwd_avx2_instance::
            add_device_conv2d_fwd_avx2_nhwc_kyxck8_nhwk(conv_ptrs);
        }
        else {
            ck::tensor_operation::cpu::device::device_conv2d_fwd_avx2_instance::
            add_device_conv2d_fwd_avx2_nhwc_kyxck8_nhwk(conv_ptrs);
        }
        ck::tensor_operation::cpu::device::device_conv2d_fwd_avx2_instance::
        add_device_conv2d_direct_fwd_avx2_nhwc_kyxck8_nhwk(conv_ptrs);
#endif
#if TEST_FUSION == TEST_FUSION_RELU
        if (omp_get_max_threads() > 1) {
            ck::tensor_operation::cpu::device::device_conv2d_fwd_avx2_instance::
            add_device_conv2d_fwd_avx2_nhwc_kyxck8_nhwk_mt_relu(conv_ptrs);
            ck::tensor_operation::cpu::device::device_conv2d_fwd_avx2_instance::
            add_device_conv2d_fwd_avx2_nhwc_kyxck8_nhwk_relu(conv_ptrs);
        }
        else {
            ck::tensor_operation::cpu::device::device_conv2d_fwd_avx2_instance::
            add_device_conv2d_fwd_avx2_nhwc_kyxck8_nhwk_relu(conv_ptrs);
        }
#endif
#endif
#if TEST_LAYOUT == TEST_LAYOUT_NHWC_YXCK_NHWK
#if TEST_FUSION == TEST_FUSION_PASSTHROUGH
        if (omp_get_max_threads() > 1) {
            ck::tensor_operation::cpu::device::device_conv2d_fwd_avx2_instance::
            add_device_conv2d_fwd_avx2_nhwc_yxck_nhwk_mt(conv_ptrs);
            ck::tensor_operation::cpu::device::device_conv2d_fwd_avx2_instance::
            add_device_conv2d_fwd_avx2_nhwc_yxck_nhwk(conv_ptrs);
        }
        else {
            ck::tensor_operation::cpu::device::device_conv2d_fwd_avx2_instance::
            add_device_conv2d_fwd_avx2_nhwc_yxck_nhwk(conv_ptrs);
        }
#endif
#if TEST_FUSION == TEST_FUSION_RELU
        if (omp_get_max_threads() > 1) {
            ck::tensor_operation::cpu::device::device_conv2d_fwd_avx2_instance::
            add_device_conv2d_fwd_avx2_nhwc_yxck_nhwk_mt_relu(conv_ptrs);
            ck::tensor_operation::cpu::device::device_conv2d_fwd_avx2_instance::
            add_device_conv2d_fwd_avx2_nhwc_yxck_nhwk_relu(conv_ptrs);
        }
        else {
            ck::tensor_operation::cpu::device::device_conv2d_fwd_avx2_instance::
            add_device_conv2d_fwd_avx2_nhwc_yxck_nhwk_relu(conv_ptrs);
        }
#endif
#endif
    }

    if (conv_ptrs.size() <= 0) {
        throw std::runtime_error("wrong! no device Conv instance found");
    }
    zendnnInfo(ZENDNN_CORELOG,
               "ZENDNN implementation path in ck_convolution_fwd_t::add_device_conv_ptrs (2) [cpu/convolution]");
    return status::success;
}

using namespace zendnn::impl::status;
using namespace zendnn::impl::memory_tracking::names;
using namespace zendnn::impl::utils;
using namespace nstl;

status_t ck_convolution_fwd_t::execute_forward(const exec_ctx_t &ctx) const {
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

    // TODO - throw error if K % 8 != 0

    zendnnInfo(ZENDNN_CORELOG,
               "ZENDNN implementation path in ck_convolution_fwd_t::execute_forward (1) [cpu/convolution]");
    zendnnInfo(ZENDNN_CORELOG, "algo=", jcp.alg_kind, " mb=",jcp.mb, " ih=",jcp.ih,
               " iw=",jcp.iw,
               " id=",jcp.id, " oh=",jcp.oh, " ow=",jcp.ow, " od=",jcp.od, " kh=",jcp.kh,
               " kw=",jcp.kw, " kd=",jcp.kd, " stride_h=",jcp.stride_h,
               " stride_w=",jcp.stride_w, " l_pad=",jcp.l_pad, " t_pad=",jcp.t_pad,
               " f_pad=",jcp.f_pad, " ngroups=",jcp.ngroups, " ic=",jcp.ic, " oc=",jcp.oc,
               " idx=", jcp.ck_fastest_kernel_idx,
               " [cpu/convolution]");

    if (jcp.oc % 8 != 0) {
        return status::unimplemented;
    }

    using InDataType  = float;
    using WeiDataType = float;
    using OutDataType = float;

    // const std::vector<ck::index_t> input_spatial_lengths{{Hi, Wi}};
    const std::vector<ck::index_t> input_spatial_lengths{{jcp.ih, jcp.iw}};
    // const std::vector<ck::index_t> filter_spatial_lengths{{Y, X}};
    const std::vector<ck::index_t> filter_spatial_lengths{{jcp.kh, jcp.kw}};

    // const ck::index_t Ho = (Hi + in_left_pad_h + in_right_pad_h - YEff) / conv_stride_h + 1;
    // const ck::index_t Wo = (Wi + in_left_pad_w + in_right_pad_w - XEff) / conv_stride_w + 1;
    // const std::vector<ck::index_t> output_spatial_lengths{{Ho, Wo}};
    const std::vector<ck::index_t> output_spatial_lengths{{jcp.oh, jcp.ow}};

    // const std::vector<ck::index_t> conv_filter_strides{{conv_stride_h, conv_stride_w}};
    const std::vector<ck::index_t> conv_filter_strides{{jcp.stride_h, jcp.stride_w}};

    // const std::vector<ck::index_t> conv_filter_dilations{{conv_dilation_h, conv_dilation_w}};
    const std::vector<ck::index_t> conv_filter_dilations{{jcp.dilate_h+1, jcp.dilate_w+1}};

    // const std::vector<ck::index_t> input_left_pads{{in_left_pad_h, in_left_pad_w}};
    const std::vector<ck::index_t> input_left_pads{{jcp.t_pad, jcp.l_pad}};

    // const std::vector<ck::index_t> input_right_pads{{in_right_pad_h, in_right_pad_w}};
    const std::vector<ck::index_t> input_right_pads{{jcp.b_pad, jcp.r_pad}};

    auto &conv_ptr = conv_ptrs[jcp.ck_fastest_kernel_idx];

    using InElementOp  = ck::tensor_operation::cpu::element_wise::PassThrough;
    using WeiElementOp = ck::tensor_operation::cpu::element_wise::PassThrough;
#if TEST_FUSION == TEST_FUSION_PASSTHROUGH
    using OutElementOp = ck::tensor_operation::cpu::element_wise::PassThrough;
#endif
#if TEST_FUSION == TEST_FUSION_RELU
    using OutElementOp = ck::tensor_operation::cpu::element_wise::Relu;
#endif

    auto argument_ptr = conv_ptr->MakeArgumentPointer(
                            // static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
                            src,
                            // static_cast<WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
                            weights,
                            // static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
                            dst,
                            // N,
                            jcp.mb,
                            // K,
                            jcp.oc,
                            // C,
                            jcp.ic,
                            input_spatial_lengths,
                            filter_spatial_lengths,
                            output_spatial_lengths,
                            conv_filter_strides,
                            conv_filter_dilations,
                            input_left_pads,
                            input_right_pads,
                            InElementOp{},
                            WeiElementOp{},
                            OutElementOp{});

    if (conv_ptr->IsSupportedArgument(argument_ptr.get())) {
        auto invoker_ptr = conv_ptr->MakeInvokerPointer();
        double time      = invoker_ptr->Run(argument_ptr.get(), StreamConfig{}, 1);
    }
    else {
        std::cout << "Not support Info: " << conv_ptr->GetTypeString() << std::endl;
    }

#undef USE_THIS_CODE
#ifdef USE_THIS_CODE
    std::cout << "src[0] = " << src[0] ;
    std::cout << ", wei[0] = " << weights[0] ;
    std::cout << ", dst[0] = " << dst[0] ;
    std::cout << std::endl;
    std:: cout << "N = " << jcp.mb ;
    std:: cout << ", K = " << jcp.oc ;
    std:: cout << ", C = " << jcp.ic ;
    std::cout << std::endl;
    std:: cout << "input_spatial_lengths[0] = " << input_spatial_lengths[0] ;
    std:: cout << ", input_spatial_lengths[1] = " << input_spatial_lengths[1] ;
    std::cout << std::endl;
    std:: cout << "filter_spatial_lengths[0] = " << filter_spatial_lengths[0] ;
    std:: cout << ", filter_spatial_lengths[1] = " << filter_spatial_lengths[1] ;
    std::cout << std::endl;
    std:: cout << "output_spatial_lengths[0] = " << output_spatial_lengths[0] ;
    std:: cout << ", output_spatial_lengths[1] = " << output_spatial_lengths[1] ;
    std::cout << std::endl;
    std:: cout << "conv_filter_strides[0] = " << conv_filter_strides[0] ;
    std:: cout << ", conv_filter_strides[1] = " << conv_filter_strides[1] ;
    std::cout << std::endl;
    std:: cout << "conv_filter_dilations[0] = " << conv_filter_dilations[0] ;
    std:: cout << ", conv_filter_dilations[1] = " << conv_filter_dilations[1] ;
    std::cout << std::endl;
    std:: cout << "input_left_pads[0] = " << input_left_pads[0] ;
    std:: cout << ", input_left_pads[1] = " << input_left_pads[1] ;
    std::cout << std::endl;
    std:: cout << "input_right_pads[0] = " << input_right_pads[0] ;
    std:: cout << ", input_right_pads[1] = " << input_right_pads[1] ;
    std::cout << std::endl;
#endif

    zendnnInfo(ZENDNN_CORELOG,
               "ZENDNN implementation path in ck_convolution_fwd_t::execute_forward (2) [cpu/convolution]");

    return status::success;
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace zendnn

#endif // #ifdef ENABLE_CK
// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
