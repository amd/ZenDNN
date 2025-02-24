/*******************************************************************************
* Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
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
#include "common/memory_tracking.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_primitive.hpp"

#include "cpu/gemm/gemm.hpp"

#include "cpu/binary_injector_utils.hpp"
#include "cpu/matmul/zendnn_x8s8s32x_matmul.hpp"
#include "cpu/matmul/matmul_utils.hpp"

#include "zendnn_logging.hpp"
#include "common/zendnn_private.hpp"
#include "zendnn.hpp"


namespace zendnn {
namespace impl {
namespace cpu {
namespace matmul {

using namespace data_type;

namespace {
template <typename pd_t>
bool need_post_processing(const pd_t *pd, float runtime_dst_zero_point = 0.f) {
    return pd->with_bias() || pd->dst_md()->data_type != s32
           || !pd->params().dst_is_acc_
           || !pd->params().pp_attr_.has_default_values()
           || !pd->params().pp_attr_.zero_points_.has_default_values(
               ZENDNN_ARG_DST)
           || runtime_dst_zero_point != 0.f;
}
} // namespace

status_t zendnn_x8s8s32x_matmul_t::pd_t::init(engine_t *engine) {
    using namespace utils;
    using namespace data_type;

    auto check_attr_scales = [&]() -> bool {
        bool ok = attr_scales_ok();
        if (!attr()->scales_.get(ZENDNN_ARG_SRC).has_default_values()
                && !attr()->scales_.get(ZENDNN_ARG_WEIGHTS).has_default_values()
                && attr()->scales_.get(ZENDNN_ARG_WEIGHTS).mask_ != 0) {
            // This case requires scratchpad with unknown size
            if (N() == ZENDNN_RUNTIME_DIM_VAL) {
                ok = false;
            }
        }
        return ok;
    };

    auto check_attr_zero_points
        = [&]() -> bool { return attr()->zero_points_.common(); };

    auto check_attr_post_ops = [&]() -> bool {
        using namespace primitive_kind;
        const auto &post_ops = attr()->post_ops_;
        static const bcast_set_t enabled_bcast_strategy {
            broadcasting_strategy_t::scalar,
            broadcasting_strategy_t::per_oc,
            broadcasting_strategy_t::per_oc_spatial,
            broadcasting_strategy_t::per_mb_spatial,
            broadcasting_strategy_t::per_mb_w,
            broadcasting_strategy_t::per_w,
            broadcasting_strategy_t::no_broadcast};
        const bool is_binary_po_per_oc
            = binary_injector_utils::bcast_strategy_present(
                  binary_injector_utils::extract_bcast_strategies(
                      post_ops.entry_, dst_md()),
                  broadcasting_strategy_t::per_oc);
        return cpu::inner_product_utils::post_ops_ok(
                   post_ops, dst_md(), enabled_bcast_strategy)
               && IMPLICATION(is_binary_po_per_oc,
                              gemm_based::check_gemm_binary_per_oc_compatible_formats(
                                  *this));
    };

    zendnnEnv zenEnvObj = readEnv();
    zendnnOpInfo &obj = zendnnOpInfo::ZenDNNOpInfo();
    if (obj.is_brgemm) {
        return status::unimplemented;
    }

    bool ok = one_of(src_md()->data_type, s8, u8)
              && weights_md()->data_type == s8 && desc()->accum_data_type == s32
              && one_of(dst_md()->data_type, f32, s32, s8, u8, bf16)
              && IMPLICATION(with_bias(),
                             one_of(weights_md(1)->data_type, f32, s32, s8, bf16)
                             && is_bias_1xN())
              && (ndims() - 2) < 1 /*Condition to check batched matmul*/
              && attr()->has_default_values(
                  primitive_attr_t::skip_mask_t::oscale_runtime
                  | primitive_attr_t::skip_mask_t::zero_points_runtime
                  | primitive_attr_t::skip_mask_t::post_ops
                  | primitive_attr_t::skip_mask_t::sum_dt,
                  dst_md()->data_type)
              && attr_.post_ops_.check_sum_consistent_dt(dst_md()->data_type)
              // need to set up default formats first, so that latter checks can
              // be perfomed properly
              && set_default_formats() && check_attr_scales()
              && check_attr_zero_points() && check_attr_post_ops()
              && gemm_based::check_gemm_compatible_formats(*this)
              && attr_.set_default_formats(dst_md(0)) == status::success;
    if (!ok) {
        return status::unimplemented;
    }

    // set states
    // copy attributes and drop src and weights zero points
    CHECK(params_.pp_attr_.copy_from(*attr()));
    params_.pp_attr_.zero_points_.set(ZENDNN_ARG_SRC, 0);
    params_.pp_attr_.zero_points_.set(ZENDNN_ARG_WEIGHTS, 0);

    params_.gemm_applies_output_scales_ = false;
    params_.gemm_beta_ = 0.f;
    auto &po = params_.pp_attr_.post_ops_;
    bool do_sum = po.find(primitive_kind::sum) >= 0;
    params_.dst_is_acc_
        = utils::one_of(dst_md()->data_type, s32, f32) && !do_sum;

    params_.has_pp_kernel_ = need_post_processing(this);

    nthr_ = zendnn_get_max_threads();
    gemm_based::book_acc_scratchpad(*this, params_, sizeof(int32_t), nthr_);

    return check_post_ops_(po);
}

// TODO:Processing the constant values for the quantization
/*
void zendnn_x8s8s32x_matmul_t::post_process_src_and_weights_zero_points(
    std::vector<int32_t> &src_comp, std::vector<int32_t> &wei_comp, dim_t M,
    dim_t N, dim_t K, const char *src, dim_t src_s0, dim_t src_s1,
    const int8_t *wei, dim_t wei_s0, dim_t wei_s1, int32_t *acc, int ldc,
    int32_t src_zero_point, int32_t wei_zero_point) const {
    if (wei_zero_point) {
        for_(dim_t m = 0; m < M; ++m)
        for (dim_t k = 0; k < K; ++k) {
            if (k == 0) {
                src_comp[m] = int32_t(0);
            }
            src_comp[m] += src[src_s0 * m + src_s1 * k];
        }
    }

    if (src_zero_point) {
        for_(dim_t k = 0; k < K; ++k)
        for (dim_t n = 0; n < N; ++n) {
            if (k == 0) {
                wei_comp[n] = int32_t(0);
            }
            wei_comp[n] += wei[wei_s0 * k + wei_s1 * n];
        }
    }

    for_(dim_t m = 0; m < M; ++m)
    for (dim_t n = 0; n < N; ++n)
        acc[m * ldc + n] += 0 - src_zero_point * wei_comp[n]
                            - wei_zero_point * src_comp[m]
                            + src_zero_point * wei_zero_point * (int)K;
}
*/

status_t zendnn_x8s8s32x_matmul_t::execute_ref(const exec_ctx_t &ctx) const {
    using namespace binary_injector_utils;

    auto src = CTX_IN_MEM(const char *, ZENDNN_ARG_SRC);
    auto weights = CTX_IN_MEM(const int8_t *, ZENDNN_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, ZENDNN_ARG_BIAS);
    auto dst = CTX_OUT_MEM(char *, ZENDNN_ARG_DST);
    const auto &po = this->pd()->attr()->post_ops_;
    const auto post_ops_binary_rhs_arg_vec = prepare_binary_args(po, ctx);

    // Is set using set_output_scales()
    DEFINE_SCALES_BUFFER(scales);

    // Passed as runtime memory
    DEFINE_STATIC_SCALES_BUFFER(src_scales, src_scale_size, src_scale_type,
                                ZENDNN_ARG_SRC);
    DEFINE_STATIC_SCALES_BUFFER(wei_scales, wei_scale_size, wei_scale_type,
                                ZENDNN_ARG_WEIGHTS);
    DEFINE_STATIC_SCALES_BUFFER(dst_scales, dst_scale_size, dst_scale_type,
                                ZENDNN_ARG_DST);

    DEFINE_ZERO_POINT_VALUE(src_zero_point, ZENDNN_ARG_SRC);
    DEFINE_ZERO_POINT_VALUE(weights_zero_point, ZENDNN_ARG_WEIGHTS);
    DEFINE_ZERO_POINT_VALUE(dst_zero_point, ZENDNN_ARG_DST);

    const auto src_d = ctx.memory_mdw(ZENDNN_ARG_SRC, pd()->src_md());
    const auto weights_d = ctx.memory_mdw(ZENDNN_ARG_WEIGHTS, pd()->weights_md());
    const auto dst_d = ctx.memory_mdw(ZENDNN_ARG_DST, pd()->dst_md());

    int src_type = src_d.data_type();
    int dst_type = dst_d.data_type();
    int bias_type = pd()->weights_md(1)->data_type;
    bool default_src_scales = pd()->attr()->static_scales_.get(
                                  ZENDNN_ARG_SRC).has_default_values();
    bool default_wei_scales = pd()->attr()->static_scales_.get(
                                  ZENDNN_ARG_WEIGHTS).has_default_values();
    bool default_dst_scales = pd()->attr()->static_scales_.get(
                                  ZENDNN_ARG_DST).has_default_values();

    // Use output_scales only when no other scale is set
    // Modifying the wei_scales to support output_scales
    if (default_src_scales && default_wei_scales && default_dst_scales) {
        wei_scales = scales;
        wei_scale_size = oscale_size;
    }

    zendnnInfo(ZENDNN_PROFLOG, "zendnn_int8_matmul_t::execute");

    matmul_helper_t helper(src_d, weights_d, dst_d);
    const int ndims = pd()->ndims();
    dim_t M = helper.M();
    const dim_t N = helper.N();
    const dim_t K = helper.K();

    const gemm_based::params_t &params = pd()->params();
    const bool Layout = true;
    // In case of normal matrices, the stride of the last dimension will always be 1,
    // as the elements are contiguous. However in case of transposed matrix, the
    // stride of the last dimension will be greater than 1.
    const char transA = helper.transA();
    const char transB = helper.transB();

    const dim_t lda = helper.lda();
    const dim_t ldb = helper.ldb();
    const dim_t ldc = helper.ldc();

    zendnnEnv zenEnvObj = readEnv();
    bool is_weights_const = zenEnvObj.zenWeightCache &&
                            pd()->weights_md()->is_memory_const;

    const float alpha = params.get_gemm_alpha(dst_scales);
    const float beta = params.gemm_beta_;

    if (dst_scale_type == bf16) {
        float *dst_float_scales = const_cast<float *>(dst_scales);
        for (int idx = 0; idx < dst_scale_size; idx++) {
            dst_float_scales[idx] = zendnn::impl::cpu::io::load_float_value((zendnn_bf16),
                                    (void *)dst_scales, idx);
        }
        dst_scales = dst_float_scales;
    }

    int sum_idx = po.find(primitive_kind::sum);
    float do_sum = sum_idx >= 0 ? po.entry_[sum_idx].sum.scale: 0.0f;
    int algo = zenEnvObj.zenINT8GEMMalgo, auto_tuner = 0 ;

    //TODO: Seperate Implementation of Decision Tree for INT8 (MATMUL_DT_INT8)
    if (algo == zenINT8MatMulAlgoType::MATMUL_AUTO_INT8) {
        auto_tuner = 1;
        algo = auto_compute_matmul_int8(ctx, zenEnvObj, src_type, dst_type, bias_type,
                                        Layout, transA == 'N'? 0 : 1, transB == 'N' ? 0 : 1,
                                        M, K, N, alpha, src, lda, weights, ldb, bias, pd()->attr()->post_ops_,
                                        beta, (char *)dst, ldc, src_zero_point, weights_zero_point,
                                        dst_zero_point, do_sum, is_weights_const, const_cast<float *>(src_scales),
                                        src_scale_size, default_src_scales, const_cast<float *>(wei_scales),
                                        wei_scale_size, default_wei_scales,
                                        const_cast<float *>(dst_scales), dst_scale_size, default_dst_scales,
                                        src_scale_type);
    }
    else {
        algo = matmul_int8_wrapper(ctx, zenEnvObj, src_type, dst_type, bias_type,
                                   Layout,
                                   transA == 'N'? 0 : 1, transB == 'N' ? 0 : 1,
                                   M, K, N, alpha, src, lda, weights, ldb, bias, pd()->attr()->post_ops_,
                                   beta, (char *)dst, ldc, src_zero_point, weights_zero_point,
                                   dst_zero_point, do_sum, is_weights_const, const_cast<float *>(src_scales),
                                   src_scale_size, default_src_scales, const_cast<float *>(wei_scales),
                                   wei_scale_size, default_wei_scales,
                                   const_cast<float *>(dst_scales), dst_scale_size, default_dst_scales,
                                   src_scale_type);
    }
    zendnnVerbose(ZENDNN_PROFLOG, "zenMatMul_gemm auto_tuner=",
                  auto_tuner ? "True": "False",
                  " Layout=", 1 ? "CblasRowMajor," : "CblasColMajor,",
                  " m=", M, " k=", K, " n=", N, " lda=", lda, " ldb=", ldb,
                  " ldc=", ldc, " alpha=", alpha, " beta=", beta,
                  " algo_type=", algo, " scale_sum=", do_sum, " src dt=", src_type, " dst_dt=",
                  dst_type, " weight_caching=", is_weights_const ? "True": "False");


    return status::success;
}

} // namespace matmul
} // namespace cpu
} // namespace impl
} // namespace zendnn
