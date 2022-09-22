/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include <atomic>

#include <assert.h>
#include <float.h>
#include <math.h>

#include "common/c_types_map.hpp"
#include "common/zendnn_thread.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_primitive.hpp"

#include "cpu/gemm/gemm.hpp"

#include "cpu/binary_injector_utils.hpp"
#include "cpu/matmul/gemm_f32_matmul.hpp"
#include "cpu/matmul/matmul_utils.hpp"
#include "cpu/matmul/zendnn_f32_matmul.hpp"

#include "zendnn_logging.hpp"
#include "common/zendnn_private.hpp"

namespace zendnn {
namespace impl {
namespace cpu {
namespace matmul {

using namespace data_type;

status_t zendnn_f32_matmul_t::pd_t::init(engine_t *engine) {
    zendnnInfo(ZENDNN_CORELOG, "zendnn_f32_matmul_t::pd_t::init()");
    auto check_bias = [&]() -> bool {
        return !with_bias()
        || (weights_md(1)->data_type == f32 && is_bias_1xN());
    };

    bool ok = src_md()->data_type == src_type
              && weights_md()->data_type == weights_type
              && desc()->accum_data_type == acc_type
              && dst_md()->data_type == dst_type && check_bias()
              && attr()->has_default_values(
                  primitive_attr_t::skip_mask_t::oscale_runtime
                  | primitive_attr_t::skip_mask_t::post_ops)
              && set_default_formats()
              && gemm_based::check_gemm_compatible_formats(*this);

    if (!ok) {
        return status::unimplemented;
    }

    // set state
    params_.dst_is_acc_ = true;
    if (!has_runtime_dims_or_strides())
        params_.can_fuse_src_batch_dims_
            = matmul_helper_t(src_md(), weights_md(), dst_md())
              .can_fuse_src_batch_dims();

    return check_and_configure_attributes();
}

status_t zendnn_f32_matmul_t::pd_t::check_and_configure_attributes() {
    zendnnInfo(ZENDNN_CORELOG,
               "zendnn_gemm_f32_matmul_t::pd_t::check_and_configure_attributes");
    auto check_attr_oscale = [&]() -> bool {
        const auto &oscale = attr()->output_scales_;
        return oscale.mask_ == 0
        || (oscale.mask_ == (1 << 1) && batched() == false);
    };

    auto check_attr_post_ops = [&]() -> bool {
        using namespace primitive_kind;
        const auto &p = attr()->post_ops_;
        auto check_sum = [&](int idx) -> bool {
            return p.contain(sum, idx) && params_.gemm_applies_output_scales_;
        };
        switch (p.len()) {
        case 0:
            return true;
        case 1:
            return check_sum(0) || p.contain(eltwise, 0);
        case 2:
            return check_sum(0) && p.contain(eltwise, 1);
        default:
            return false;
        }
    };

    // check basic attributes
    if (!check_attr_oscale()) {
        return status::unimplemented;
    }

    // set state
    CHECK(params_.pp_attr_.copy_from(*attr()));
    params_.gemm_applies_output_scales_
#if ZENDNN_ENABLE
// ZenDNN MatMul can support scalar alpha values with bias
        = attr()->output_scales_.mask_ == 0;
#else
        = attr()->output_scales_.mask_ == 0 && !with_bias();
#endif
    if (params_.gemm_applies_output_scales_) {
        params_.pp_attr_.output_scales_.set(1.f);
    }

    // check post-ops
    if (check_attr_post_ops()) {
        auto &po = params_.pp_attr_.post_ops_;
        const int sum_idx = 0;
        if (po.len() > 0 && po.contain(primitive_kind::sum, sum_idx)) {
            // set state
            params_.gemm_beta_ = po.entry_[sum_idx].sum.scale;
            // drop sum from pp_attributes, as it will be applied by gemm
            po.entry_.erase(po.entry_.begin());
        }
    }
    else {
        return status::unimplemented;
    }

    // set state
    params_.has_pp_kernel_
        = with_bias() || !params_.pp_attr_.has_default_values();

    return status::success;
}

status_t zendnn_f32_matmul_t::execute_ref(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const src_data_t *, ZENDNN_ARG_SRC);
    auto weights = CTX_IN_MEM(const weights_data_t *, ZENDNN_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, ZENDNN_ARG_BIAS);
    auto dst = CTX_OUT_MEM(dst_data_t *, ZENDNN_ARG_DST);

    DEFINE_SCALES_BUFFER(scales);

    const auto src_d = ctx.memory_mdw(ZENDNN_ARG_SRC, pd()->src_md());
    const auto weights_d = ctx.memory_mdw(ZENDNN_ARG_WEIGHTS, pd()->weights_md());
    const auto dst_d = ctx.memory_mdw(ZENDNN_ARG_DST, pd()->dst_md());

    const gemm_based::params_t &params = pd()->params();

    const auto &dst_bd = dst_d.blocking_desc();

    const bool batched = pd()->batched();

    const dim_t M = dst_d.dims()[dst_d.ndims() - 2];
    const dim_t N = dst_d.dims()[dst_d.ndims() - 1];
    const dim_t K = src_d.dims()[src_d.ndims() - 1];

    const auto &src_strides = &src_d.blocking_desc().strides[dst_d.ndims() - 2];
    const auto &weights_strides = &weights_d.blocking_desc().strides[dst_d.ndims() -
                                                2];

    const char *transA
        = src_strides[1] == 1 &&
          src_d.dims()[dst_d.ndims() - 2] > 1 ? "N" : "T";
    const char *transB
        = weights_strides[1] == 1 &&
          weights_d.dims()[dst_d.ndims() - 2] > 1 ? "N" : "T";

    const dim_t M_s32 = (dim_t)M;
    const dim_t N_s32 = (dim_t)N;
    const dim_t K_s32 = (dim_t)K;

    const dim_t lda = (dim_t)src_strides[*transA == 'N' ? 0 : 1];
    const dim_t ldb = (dim_t)weights_strides[*transB == 'N' ? 0 : 1];
    const dim_t ldc = (dim_t)dst_bd.strides[dst_d.ndims() - 2];

    float alpha = params.get_gemm_alpha(scales);
    const float beta = params.gemm_beta_;
    //TODO(aakar): Modify f32_matmul API to include Layout
    const bool Layout = true; // CblasRowMajor

    const dim_t batch = batched ? src_d.dims()[dst_d.ndims() - 3] : 1;
    const auto src_batch_stride = src_d.blocking_desc().strides[0];
    const auto weights_batch_stride = weights_d.blocking_desc().strides[0];
    const auto dst_batch_stride = dst_d.blocking_desc().strides[0];

    zendnnInfo(ZENDNN_CORELOG, "zendnn_f32_matmul_t::execute_ref");
    zendnnInfo(ZENDNN_CORELOG, "M: ",M, " N: ",N, " K: ", K,
               " transA: ", transA, " transB: ", transB,
               " lda: ", lda, " ldb: ", ldb, " ldc: ", ldc,
               " alpha: ", alpha, " beta: ", beta, " batch: ", batch,
               " Layout: ", Layout ? "CblasRowMajor(1)" : "CblasColMajor(0)");
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

#if ZENDNN_ENABLE
    alpha = pd()->attr()->output_scales_.mask_ == 0 ? scales[0] : 1.0;
    if ((float *)bias == NULL) {
        //MatMul without Bias
        zenMatMul(Layout, strcmp(transA, "N"),strcmp(transB, "N"), batch, M, K,
                  N, alpha, (float *)src, lda, (float *)weights, ldb, beta,
                  (float *)dst, ldc);
    }
    else if ((float *)bias != NULL && !has_eltwise) {
        //MatMul with Bias
        zenMatMulWithBias(Layout, strcmp(transA, "N"), strcmp(transB, "N"),
                          batch, M, K, N, alpha, (float *)src, lda, (float *)weights, ldb,
                          (float *)bias, beta, (float *)dst, ldc);
    }
    else {
        if (has_eltwise_relu) {
            //MatMul with BiasRelu
            zenMatMulWithBiasReLU(Layout, strcmp(transA, "N"), strcmp(transB, "N"),
                                  batch, M, K, N, alpha, (float *)src, lda, (float *)weights, ldb,
                                  (float *)bias, beta, (float *)dst, ldc);
        }
        else if (has_eltwise_gelu) {
            //MatMul with BiasGelu
            //gelu_type is passed as last argument, 1 refers to tanh based gelu
            zendnnInfo(ZENDNN_CORELOG,
                       "zendnn_f32_matmul_t::execute_forward zenMatMulWithBiasGeLU [cpu/zendnn_f32_matmul]");
            zenMatMulWithBiasGeLU(Layout, strcmp(transA, "N"), strcmp(transB, "N"),
                                  batch, M, K, N, alpha, (float *)src, lda, (float *)weights, ldb,
                                  (float *)bias, beta, (float *)dst, ldc, 1);

        }
        else if (has_eltwise_gelu_erf) {
            //MatMul with BiasGelu
            //gelu_type is passed as last argument, 2 refers to erf based gelu
            zendnnInfo(ZENDNN_CORELOG,
                       "zendnn_f32_matmul_t::execute_forward zenMatMulWithBiasGeLU [cpu/zendnn_f32_matmul]");
            zenMatMulWithBiasGeLU(Layout, strcmp(transA, "N"), strcmp(transB, "N"),
                                  batch, M, K, N, alpha, (float *)src, lda, (float *)weights, ldb,
                                  (float *)bias, beta, (float *)dst, ldc, 2);
        }
        else {
            return status::unimplemented;
        }
    }
#endif //ZENDNN_ENABLE

    return status::success;
}

} // namespace matmul
} // namespace cpu
} // namespace impl
} // namespace zendnn
