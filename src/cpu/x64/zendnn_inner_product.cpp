/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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

#include "cpu/binary_injector_utils.hpp"

#include "zendnn_logging.hpp"
#include "zendnn_inner_product.hpp"
#include "common/zendnn_private.hpp"

namespace zendnn {
namespace impl {
namespace cpu {
namespace x64 {

using namespace zendnn::impl::status;
using namespace zendnn::impl::prop_kind;
using namespace zendnn::impl::data_type;
using namespace zendnn::impl::format_tag;
using namespace zendnn::impl::primitive_kind;

template <impl::data_type_t data_type>
status_t zendnn_inner_product_fwd_t<data_type>::execute_forward(
    const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const data_t *, ZENDNN_ARG_SRC);
    auto weights = CTX_IN_MEM(const data_t *, ZENDNN_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const data_t *, ZENDNN_ARG_BIAS);
    auto dst = CTX_OUT_MEM(data_t *, ZENDNN_ARG_DST);
    const auto post_ops_binary_rhs_arg_vec
        = binary_injector_utils::prepare_binary_args(
              this->pd()->attr()->post_ops_, ctx);

    const dim_t MB = pd()->MB();
    const dim_t OC = pd()->OC();
    const dim_t IC = pd()->IC_total_padded();

    const auto &wmd = *pd()->weights_md();
    // check if OC is NOT the leading dimension
    bool wei_tr = wmd.format_desc.blocking.strides[0] != 1;

    bool has_eltwise = pd()->attr()->post_ops_.find(primitive_kind::eltwise) >= 0;

    int elementwise_index =  pd()->attr()->post_ops_.find(primitive_kind::eltwise);
    bool has_eltwise_relu = elementwise_index>=0 ?
                            pd()->attr()->post_ops_.entry_[elementwise_index].eltwise.alg ==
                            alg_kind::eltwise_relu : 0;

    //alg_kind::eltwise_gelu is same as alg_kind::eltwise_gelu_tanh
    bool has_eltwise_gelu = elementwise_index>=0 ?
                            pd()->attr()->post_ops_.entry_[elementwise_index].eltwise.alg ==
                            alg_kind::eltwise_gelu : 0;

    bool has_eltwise_gelu_erf = elementwise_index>=0 ?
                                pd()->attr()->post_ops_.entry_[elementwise_index].eltwise.alg ==
                                alg_kind::eltwise_gelu_erf : 0;

    unsigned int geluType = has_eltwise_gelu?1:(has_eltwise_gelu_erf?2:0);

    const float *scales = pd()->attr()->output_scales_.scales_;

    // The mask value of 0 implies a common output scaling factor for the
    // whole output tensor.
    float alpha = pd()->attr()->output_scales_.mask_ == 0 ? scales[0] : 1.0;
    //TODO(aakar): Handle case where (mask == 1 << 1) or (mask != 0)
    // Modify inner_product API to support Layout
    bool Layout = true; //CblasRowMajor

    zendnnInfo(ZENDNN_CORELOG,
               "ZENDNN implementation path in zendnn_inner_product_fwd_t::execute_forward [cpu/inner_product]");

    int input_offsets[] = {0};
    int weight_offsets[] = {0};
    int dst_offsets[] = {0};

    if (bias == NULL) {
        zendnnInfo(ZENDNN_CORELOG,
                   "zendnn_inner_product_fwd_t::execute_forward zenMatMul [cpu/inner_product]");
        zenMatMul(
            Layout, false, wei_tr, 1, input_offsets, weight_offsets, dst_offsets, MB, IC,
            OC, alpha, (float *)src, IC,
            (float *)weights, wei_tr ? IC : OC, NULL, has_eltwise_relu, geluType, beta_,
            (float *)dst, OC);
    }
    else if (!has_eltwise) {
        zendnnInfo(ZENDNN_CORELOG,
                   "zendnn_inner_product_fwd_t::execute_forward zenMatMulWithBias [cpu/inner_product]");
        zenMatMulWithBias(
            Layout, false, wei_tr, 1, input_offsets, weight_offsets, dst_offsets, MB, IC,
            OC, alpha, (float *)src, IC,
            (float *)weights, wei_tr ? IC : OC, (float *)bias, beta_,
            (float *)dst, OC);
    }
    else {
        if (has_eltwise_relu) {
            //MatMul with BiasRelu
            zendnnInfo(ZENDNN_CORELOG,
                       "zendnn_inner_product_fwd_t::execute_forward zenMatMulWithBiasReLU [cpu/inner_product]");
            zenMatMulWithBiasReLU(
                Layout, false, wei_tr, 1, input_offsets, weight_offsets, dst_offsets, MB, IC,
                OC, alpha, (float *)src, IC,
                (float *)weights, wei_tr ? IC : OC, (float *)bias, beta_,
                (float *)dst, OC);
        }
        else if (has_eltwise_gelu) {
            //MatMul with BiasGelu
            //gelu_type is passed as last argument, 1 refers to tanh based gelu
            zendnnInfo(ZENDNN_CORELOG,
                       "zendnn_inner_product_fwd_t::execute_forward zenMatMulWithBiasGeLU [cpu/inner_product]");
            zenMatMulWithBiasGeLU(
                Layout, false, wei_tr, 1, input_offsets, weight_offsets, dst_offsets, MB, IC,
                OC, alpha, (float *)src, IC,
                (float *)weights, wei_tr ? IC : OC, (float *)bias, beta_,
                (float *)dst, OC, 1);

        }
        else if (has_eltwise_gelu_erf) {
            //MatMul with BiasGelu
            //gelu_type is passed as last argument, 2 refers to erf based gelu
            zendnnInfo(ZENDNN_CORELOG,
                       "zendnn_inner_product_fwd_t::execute_forward zenMatMulWithBiasGeLU [cpu/inner_product]");
            zenMatMulWithBiasGeLU(
                Layout, false, wei_tr, 1, input_offsets, weight_offsets, dst_offsets, MB, IC,
                OC, alpha, (float *)src, IC,
                (float *)weights, wei_tr ? IC : OC, (float *)bias, beta_,
                (float *)dst, OC, 2);

        }
        else {
            return status::unimplemented;
        }
    }

    return status::success;
}

template <impl::data_type_t data_type>
status_t zendnn_inner_product_bwd_data_t<data_type>::execute_backward_data(
    const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const data_t *, ZENDNN_ARG_DIFF_DST);
    auto weights = CTX_IN_MEM(const data_t *, ZENDNN_ARG_WEIGHTS);
    auto diff_src = CTX_OUT_MEM(data_t *, ZENDNN_ARG_DIFF_SRC);

    const dim_t MB = pd()->MB();
    const dim_t OC = pd()->OC();
    const dim_t IC = pd()->IC_total_padded();

    const auto &wmd = *pd()->weights_md();
    bool wei_tr = wmd.format_desc.blocking.strides[0] == 1;

    float alpha = 1.0, beta = 0.0;
    status_t st = extended_sgemm(wei_tr ? "T" : "N", "N", &IC, &MB, &OC, &alpha,
                                 weights, wei_tr ? &OC : &IC, diff_dst, &OC, &beta, diff_src, &IC);

    return st;
}

template <impl::data_type_t data_type>
status_t zendnn_inner_product_bwd_weights_t<data_type>::execute_backward_weights(
    const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const data_t *, ZENDNN_ARG_DIFF_DST);
    auto src = CTX_IN_MEM(const data_t *, ZENDNN_ARG_SRC);
    auto diff_weights = CTX_OUT_MEM(data_t *, ZENDNN_ARG_DIFF_WEIGHTS);
    auto diff_bias = CTX_OUT_MEM(data_t *, ZENDNN_ARG_DIFF_BIAS);

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_bias_d(pd()->diff_weights_md(1));

    diff_dst += diff_dst_d.offset0();

    const dim_t MB = pd()->MB();
    const dim_t OC = pd()->OC();
    const dim_t IC = pd()->IC_total_padded();

    const auto &wmd = *pd()->diff_weights_md();
    bool wei_tr = wmd.format_desc.blocking.strides[0] == 1;

    float alpha = 1.0, beta = 0.0;
    status_t st;
    if (wei_tr)
        st = extended_sgemm("N", "T", &OC, &IC, &MB, &alpha, diff_dst, &OC, src,
                            &IC, &beta, diff_weights, &OC);
    else
        st = extended_sgemm("N", "T", &IC, &OC, &MB, &alpha, src, &IC, diff_dst,
                            &OC, &beta, diff_weights, &IC);

    if (st != status::success) {
        return st;
    }

    if (diff_bias) {
        diff_bias += diff_bias_d.offset0();
        constexpr dim_t blksize = 8;
        const dim_t OC_blocks = utils::div_up(OC, blksize);
        parallel(0, [&](const int ithr, const int nthr) {
            dim_t oc_s {0}, oc_e {0};
            balance211(OC_blocks, nthr, ithr, oc_s, oc_e);
            oc_s = std::min(oc_s * blksize, OC);
            oc_e = std::min(oc_e * blksize, OC);

            ZENDNN_PRAGMA_OMP_SIMD()
            for (dim_t oc = oc_s; oc < oc_e; ++oc) {
                diff_bias[oc] = diff_dst[oc];
            }

            for (dim_t mb = 1; mb < MB; ++mb) {
                ZENDNN_PRAGMA_OMP_SIMD()
                for (dim_t oc = oc_s; oc < oc_e; ++oc) {
                    diff_bias[oc] += diff_dst[mb * OC + oc];
                }
            }
        });
    }

    return status::success;
}

template struct zendnn_inner_product_fwd_t<data_type::f32>;
template struct zendnn_inner_product_bwd_data_t<data_type::f32>;
template struct zendnn_inner_product_bwd_weights_t<data_type::f32>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace zendnn

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
