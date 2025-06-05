/*******************************************************************************
* Modifications Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <vector>
#include <unordered_map>
#include <tuple>

#ifndef _WIN32
    #include <sys/time.h>
#else
    #include <windows.h>
#endif

#ifndef ZENDNN_USE_AOCL_BLIS_API
    #include <cblas.h>
#else // ZENDNN_USE_AOCL_BLIS_API
    #include "cblas_with_blis_api.hpp"
#endif // ZENDNN_USE_AOCL_BLIS_API

#include "common/c_types_map.hpp"
#include "common/zendnn_thread.hpp"
#include "common/type_helpers.hpp"
#include "zendnn_helper.hpp"
#include "common/utils.hpp"
#include "cpu/cpu_primitive.hpp"
#include "cpu/platform.hpp"
#include "cpu/ref_io_helper.hpp"

#include "cpu/gemm/gemm.hpp"

#include "cpu/binary_injector_utils.hpp"
#include "cpu/matmul/zendnn_bf16_matmul.hpp"
#include "cpu/matmul/matmul_utils.hpp"

#include "zendnn_logging.hpp"
#include "common/zendnn_private.hpp"
#include "zendnn.hpp"
#include "zendnn_reorder_cache.hpp"

using namespace zendnn;
using namespace zendnn::impl::cpu;
using tag = memory::format_tag;
using dt = memory::data_type;
extern int graph_exe_count;
extern std::mutex map_mutex;

namespace zendnn {
namespace impl {
namespace cpu {
namespace matmul {

using namespace data_type;

template <impl::data_type_t dst_type>
status_t zendnn_bf16_matmul_t<dst_type>::pd_t::init(engine_t *engine) {
    zendnnVerbose(ZENDNN_CORELOG, "zendnn_bf16_matmul_t::pd_t::init()");
    auto check_bias = [&]() -> bool {
        return !with_bias()
        || (utils::one_of(weights_md(1)->data_type, f32, bf16)
            && is_bias_1xN());
    };

    auto check_group = [&]() -> bool {
        if (attr()->woqScales_.ndims_ == 0) {
            return true;
        }
        return !(weights_md()->dims[0] % attr()->woqScales_.group_dims_[0]);
    };

    bool ok = src_md()->data_type == src_type
              && utils::one_of(weights_md()->data_type, bf16, s8, s4)
              && desc()->accum_data_type == acc_type
              && dst_md()->data_type == dst_type
              && platform::has_data_type_support(data_type::bf16) && check_bias()
              && (ndims() - 2) < 1 /*Condition to check batched matmul*/
              && check_group()
              && attr()->has_default_values(
                  primitive_attr_t::skip_mask_t::oscale_runtime
                  | primitive_attr_t::skip_mask_t::post_ops)
              && set_default_formats()
              && gemm_based::check_gemm_compatible_formats(*this);
    //Return unimplemented if BF16 algo set to 4(MATMUL_JIT_BF16)
    zendnnEnv zenEnvObj = readEnv();
    if ((zenEnvObj.zenBF16GEMMalgo == zenBF16MatMulAlgoType::MATMUL_JIT_BF16 ||
        zenEnvObj.zenBF16GEMMalgo == zenBF16MatMulAlgoType::MATMUL_GEMM_JIT_BF16 ||
            (zenEnvObj.zenBF16GEMMalgo != zenBF16MatMulAlgoType::MATMUL_AOCL_BF16 &&
             (weights_md()->is_memory_const == false ||
              (weights_md()->is_inplace == false &&
               zenEnvObj.zenWeightCache > zendnnWeightCacheType::WEIGHT_CACHE_INPLACE)))) &&
            weights_md()->data_type == bf16) {
        return status::unimplemented;
    }

    zendnnOpInfo &obj = zendnnOpInfo::ZenDNNOpInfo();
    bool check_algo_desc = obj.is_brgemm || ((obj.is_ref_gemm_bf16 ||
                           !weights_md()->is_memory_const) &&
                           weights_md()->data_type == bf16);
    if (check_algo_desc) {
        return status::unimplemented;
    }

    if (!ok) {
        return status::unimplemented;
    }

    nthr_ = zendnn_get_max_threads();

    return check_and_configure_attributes();
}

template <impl::data_type_t dst_type>
status_t zendnn_bf16_matmul_t<dst_type>::pd_t::check_and_configure_attributes() {
    zendnnVerbose(ZENDNN_CORELOG,
                  "zendnn_bf16_matmul_t::pd_t::check_and_configure_attributes");
    auto check_attr_oscale = [&]() -> bool {
        const auto &oscale = attr()->output_scales_;
        return oscale.mask_ == 0
        || (oscale.mask_ == (1 << 1) && batched() == false);
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
    auto &po = params_.pp_attr_.post_ops_;
    const int sum_idx = 0;
    if (po.len() > 0 && po.contain(primitive_kind::sum, sum_idx)) {
        // set state
        params_.gemm_beta_ = po.entry_[sum_idx].sum.scale;
        // drop sum from pp_attributes, as it will be applied by gemm
        po.entry_.erase(po.entry_.begin());
    }
    params_.dst_is_acc_ = false;
    // set state
    params_.has_pp_kernel_ = !params_.dst_is_acc_ || with_bias()
                             || !params_.pp_attr_.has_default_values();

    //Checks supported post-ops
    return check_post_ops_(po);
}

template <impl::data_type_t dst_type>
status_t zendnn_bf16_matmul_t<dst_type>::execute_ref(
    const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const src_data_t *, ZENDNN_ARG_SRC);
    auto weights = CTX_IN_MEM(const weights_data_t *, ZENDNN_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, ZENDNN_ARG_BIAS);
    auto dst = CTX_OUT_MEM(dst_data_t *, ZENDNN_ARG_DST);
    DEFINE_SCALES_BUFFER(scales);

    const auto src_d = ctx.memory_mdw(ZENDNN_ARG_SRC, pd()->src_md());
    const auto weights_d = ctx.memory_mdw(ZENDNN_ARG_WEIGHTS, pd()->weights_md());
    const auto dst_d = ctx.memory_mdw(ZENDNN_ARG_DST, pd()->dst_md());

    matmul_helper_t helper(src_d, weights_d, dst_d);
    const int ndims = pd()->ndims();
    const int batch_ndims = ndims - 2;
    const dim_t M = helper.M();
    const dim_t N = helper.N();
    const dim_t K = helper.K();
    const dim_t batch = helper.batch();

    const bool batched = (batch_ndims > 0) ? true : false;

    zendnnEnv zenEnvObj = readEnv();
    const gemm_based::params_t &params = pd()->params();
    bool is_weights_const = pd()->weights_md()->is_memory_const;
    bool is_inplace = pd()->weights_md()->is_inplace;

    const auto &dst_bd = dst_d.blocking_desc();
    const auto &src_strides = &src_d.blocking_desc().strides[dst_d.ndims() - 2];
    const auto &weights_strides = &weights_d.blocking_desc().strides[dst_d.ndims() -
                                                2];
    // In case of normal matrices, the stride of the last dimension will always be 1,
    // as the elements are contiguous. However in case of transposed matrix, the
    // stride of the last dimension will be greater than 1.
    const char transA = helper.transA();
    const char transB = helper.transB();

    const dim_t M_s32 = (dim_t)M;
    const dim_t N_s32 = (dim_t)N;
    const dim_t K_s32 = (dim_t)K;

    const dim_t lda = helper.lda();
    const dim_t ldb = helper.ldb();
    const dim_t ldc = helper.ldc();

    float alpha = params.get_gemm_alpha(scales);
    const float beta = params.gemm_beta_;
    const bool Layout = true; // CblasRowMajor

    float *output_scales {nullptr};
    output_scales = pd()->attr()->output_scales_.scales_;
    int scale_size = pd()->attr()->output_scales_.count_;

    const int *zero_point_dst {nullptr};
    zero_point_dst = pd()->attr()->zero_points_.get(ZENDNN_ARG_DST);
    zendnnInfo(ZENDNN_CORELOG, "zendnn_bf16_matmul_t::execute_ref new");

    int elementwise_index =  pd()->attr()->post_ops_.find(primitive_kind::eltwise);
    int has_binary_index = pd()->attr()->post_ops_.find(primitive_kind::binary);
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
    bool auto_tuner = false;
#if ZENDNN_ENABLE
    alpha = pd()->attr()->output_scales_.mask_ == 0 ? scales[0] : 1.0;
    int bias_dt = pd()->weights_md(1)->data_type;
    int src_type = pd()->src_md()->data_type;
    int weights_type = pd()->weights_md()->data_type;
    int algo_type = zenEnvObj.zenBF16GEMMalgo;

    // Fallback to ALGO 2(blocked BRGEMM) if BF16 weights with weight cache
    // type 2/3/4 for AUTO and DT
    if (zenEnvObj.zenWeightCache > zendnnWeightCacheType::WEIGHT_CACHE_OUT_OF_PLACE
            && weights_type == zendnn_bf16 &&
            (zenEnvObj.zenBF16GEMMalgo == zenBF16MatMulAlgoType::MATMUL_AUTO_BF16 ||
             zenEnvObj.zenBF16GEMMalgo == zenBF16MatMulAlgoType::MATMUL_DT_BF16)) {
        zenEnvObj.zenBF16GEMMalgo = zenBF16MatMulAlgoType::MATMUL_BLOCKED_JIT_BF16;
    }
    if (weights_type == s4 || weights_type == s8) {
        DEFINE_WOQ_SCALES_BUFFER(woqscales);
        float *woq_scales = const_cast<float *>(woqscales);
        int woq_scale_mask = pd()->attr()->woqScales_.mask_;
        const auto scales_d = ctx.memory_mdw(ZENDNN_ARG_ATTR_WOQ_SCALES);
        int woq_scale_size = woq_scale_mask ? scales_d.dims()[0] : 1;
        const auto group_dims = pd()->attr()->woqScales_.group_dims_;
        int group_size = woq_scale_mask & (1 << (ndims - 2)) ? group_dims[0] : K;
        data_type_t woq_scales_type = pd()->attr()->woqScales_.data_type_;
        if (zenEnvObj.zenBF16GEMMalgo == zenBF16MatMulAlgoType::MATMUL_AUTO_BF16) {
            auto_tuner = true;
            auto_compute_matmul_woq(ctx, zenEnvObj, src_type, weights_type, dst_type,
                                    bias_dt,
                                    Layout, transA == 'N'? 0 : 1, transB == 'N' ? 0 : 1,
                                    M, K, N, alpha, (char *)src, lda, (char *)weights, ldb, (char *)bias,
                                    pd()->attr()->post_ops_, has_eltwise_relu, geluType,
                                    beta, (char *)dst, ldc, is_weights_const, woq_scales, woq_scale_size,
                                    group_size, woq_scales_type);
        }
        else {
            matmul_woq_wrapper(ctx, zenEnvObj, src_type, weights_type, dst_type, bias_dt,
                               Layout,
                               transA == 'N'? 0 : 1, transB == 'N' ? 0 : 1,
                               M, K, N, alpha, (char *)src, lda, (char *)weights, ldb, (char *)bias,
                               pd()->attr()->post_ops_, has_eltwise_relu,
                               geluType, beta, (char *)dst, ldc, woq_scales, 0, woq_scale_size,
                               is_weights_const, group_size, woq_scales_type);
        }
        return status::success;
    }
    else if (zenEnvObj.zenBF16GEMMalgo == zenBF16MatMulAlgoType::MATMUL_AUTO_BF16) {
        auto_tuner = true;
        algo_type = auto_compute_matmul_bf16(ctx, zenEnvObj, dst_type, bias_dt,Layout,
                                             transA == 'N'? 0 : 1, transB == 'N' ? 0 : 1,
                                             M, K, N, alpha, src, lda, weights, ldb, bias, has_eltwise_relu,
                                             pd()->attr()->post_ops_, has_binary_index, geluType, beta,
                                             dst, ldc, output_scales, scale_size, is_weights_const, is_inplace);
    }
    else if (zenEnvObj.zenBF16GEMMalgo == zenBF16MatMulAlgoType::MATMUL_DT_BF16) {
        // If M >=64, N and K >=2048 AOCL BLIS kernels gives optimal performance.
        // This is based on heuristic with different models and difference BS
        if (M >= 64) {
            // Blocked JIT Kernels gives optimal performance where either of N and K
            // size is smaller.
            if (N < 2048 || K < 2048) {
                zenEnvObj.zenBF16GEMMalgo = zenBF16MatMulAlgoType::MATMUL_BLOCKED_JIT_BF16;
            }
            else {
                zenEnvObj.zenBF16GEMMalgo = zenBF16MatMulAlgoType::MATMUL_BLOCKED_AOCL_BF16;
            }
        }
        else {
            // For M < 64, where K size is smaller than N AOCL BLIS kernel gives
            // optimal performance.
            if (N <= K) {
                zenEnvObj.zenBF16GEMMalgo = zenBF16MatMulAlgoType::MATMUL_BLOCKED_JIT_BF16;
            }
            else {
                zenEnvObj.zenBF16GEMMalgo = zenBF16MatMulAlgoType::MATMUL_BLOCKED_AOCL_BF16;
            }
        }
        algo_type = matmul_bf16_wrapper(ctx, zenEnvObj, dst_type, bias_dt, Layout,
                                        transA == 'N'? 0 : 1, transB == 'N' ? 0 : 1,
                                        M, K, N, alpha, src, lda, weights, ldb, bias,
                                        has_eltwise_relu, pd()->attr()->post_ops_, has_binary_index,
                                        geluType, beta, dst, ldc, output_scales, scale_size,
                                        is_weights_const, is_inplace);

    }
    else {
        algo_type = matmul_bf16_wrapper(ctx, zenEnvObj, dst_type, bias_dt, Layout,
                                        transA == 'N'? 0 : 1, transB == 'N' ? 0 : 1,
                                        M, K, N, alpha, src, lda, weights, ldb, bias,
                                        has_eltwise_relu, pd()->attr()->post_ops_, has_binary_index,
                                        geluType, beta, dst, ldc, output_scales, scale_size,
                                        is_weights_const, is_inplace);
    }

    zendnnVerbose(ZENDNN_PROFLOG,"zendnn_bf16_matmul auto_tuner=",
                  auto_tuner ? "True": "False",
                  " Layout=", Layout ? "CblasRowMajor(1)" : "CblasColMajor(0)", " M=", M, " N=",N,
                  " K=", K, " transA=", transA, " transB=", transB, " lda=", lda, " ldb=", ldb,
                  " ldc=", ldc, " alpha=", alpha, " beta=", beta, " batch=", batch, " relu=",
                  has_eltwise_relu, " gelu=", geluType, " algo_type=", algo_type,
                  " weight_caching=", is_weights_const ? "True": "False", " weight_address=",
                  (void *)weights);
#endif //ZENDNN_ENABLE
    return status::success;
}

using namespace data_type;
template struct zendnn_bf16_matmul_t<data_type::f32>;
template struct zendnn_bf16_matmul_t<data_type::bf16>;

} // namespace matmul
} // namespace cpu
} // namespace impl
} // namespace zendnn
