/*******************************************************************************
* Modifications Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <vector>

#include "common/c_types_map.hpp"
#include "common/zendnn_thread.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "cpu/ref_io_helper.hpp"

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
    zendnnVerbose(ZENDNN_CORELOG, "zendnn_f32_matmul_t::pd_t::init()");
    auto check_bias = [&]() -> bool {
        return !with_bias()
        || (weights_md(1)->data_type == f32 && is_bias_1xN());
    };

    auto check_group = [&]() -> bool {
        if (attr()->woqScales_.ndims_ == 0) {
            return true;
        }
        return !(weights_md()->dims[0] % attr()->woqScales_.group_dims_[0]);
    };

    bool ok = src_md()->data_type == src_type
              && utils::one_of(weights_md()->data_type, f32, s8, s4)
              && check_group()
              && desc()->accum_data_type == acc_type
              && dst_md()->data_type == dst_type && check_bias()
              && attr()->has_default_values(
                  primitive_attr_t::skip_mask_t::oscale_runtime
                  | primitive_attr_t::skip_mask_t::post_ops)
              && set_default_formats()
              && gemm_based::check_gemm_compatible_formats(*this);

    zendnnEnv zenEnvObj = readEnv();
    unsigned int thread_qty = zenEnvObj.omp_num_threads;
    // Based on heuristics for Smaller Dimensions M(<=128), N and K (i.e <= 512)
    // control is redirected to BRGEMM.
    // TODO: Generate more heuristics for M,N,K for smaller dimensions to make
    // the check generalized

    // Batch MatMul with BLOCKED_JIT is JIT.
    if (ndims() > 2 &&
            (zenEnvObj.zenGEMMalgo == zenMatMulAlgoType::MATMUL_BLOCKED_JIT_FP32 ||
             zenEnvObj.zenGEMMalgo == zenMatMulAlgoType::MATMUL_JIT_FP32)) {
        return status::unimplemented;
    }

    if ((zenEnvObj.zenGEMMalgo == zenMatMulAlgoType::MATMUL_JIT_FP32 ||
            zenEnvObj.zenGEMMalgo == zenMatMulAlgoType::MATMUL_GEMM_JIT_FP32 ||
            (zenEnvObj.zenGEMMalgo == zenMatMulAlgoType::MATMUL_BLOCKED_JIT_FP32 &&
             (weights_md()->is_memory_const == false ||
              weights_md()->is_inplace == false &&
              zenEnvObj.zenWeightCache >= zendnnWeightCacheType::WEIGHT_CACHE_INPLACE))) &&
            weights_md()->data_type == f32) {
        return status::unimplemented;
    }

    zendnnOpInfo &obj = zendnnOpInfo::ZenDNNOpInfo();
    if (obj.is_brgemm || obj.is_ref_gemm_bf16) {
        return status::unimplemented;
    }

    if (!ok) {
        return status::unimplemented;
    }

    // TODO: Remove this check once gelu_erf accuracy is resolved.
    auto elt_idx = attr()->post_ops_.find(primitive_kind::eltwise);
    if (elt_idx >= 0 &&
            attr()->post_ops_.entry_[elt_idx].eltwise.alg == alg_kind::eltwise_gelu_erf) {
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

// temporary solution to deal with format `any`
bool zendnn_f32_matmul_t::pd_t::set_default_formats() {
    for (auto md : {
                &src_md_, &weights_md_, &dst_md_
            }) {
        memory_desc_wrapper mdw(md);
        if (mdw.format_any()) {
            if (mdw.has_runtime_dims_or_strides()) {
                return false;
            }
            status_t status = memory_desc_init_by_strides(*md, nullptr);
            if (status != status::success) {
                return false;
            }
        }
        if (!mdw.matches_tag(zendnn::impl::format_tag::ab) &&
                !mdw.matches_tag(zendnn::impl::format_tag::ba) &&
                !mdw.matches_tag(zendnn::impl::format_tag::abc) &&
                !mdw.matches_tag(zendnn::impl::format_tag::acb) &&
                !mdw.matches_tag(zendnn::impl::format_tag::abcd) &&
                !mdw.matches_tag(zendnn::impl::format_tag::abcde) &&
                !mdw.matches_tag(zendnn::impl::format_tag::abcdef) &&
                !mdw.matches_tag(zendnn::impl::format_tag::abcdefg) &&
                !mdw.matches_tag(zendnn::impl::format_tag::abcdefgh) &&
                !mdw.matches_tag(zendnn::impl::format_tag::abcdefghi) &&
                !mdw.matches_tag(zendnn::impl::format_tag::abcdefghij) &&
                !mdw.matches_tag(zendnn::impl::format_tag::abcdefghijk) &&
                !mdw.matches_tag(zendnn::impl::format_tag::abcdefghijkl)) {
            return false;
        }
    }
    return true;
}


status_t zendnn_f32_matmul_t::pd_t::check_and_configure_attributes() {
    zendnnVerbose(ZENDNN_CORELOG,
                  "zendnn_gemm_f32_matmul_t::pd_t::check_and_configure_attributes");
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
    int sum_index = params_.pp_attr_.post_ops_.find(primitive_kind::sum);
    // check post-ops
    if (check_post_ops_(params_.pp_attr_.post_ops_) == status::success &&
            sum_index <= 0) {
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

static void fill_offset(std::vector<unsigned long> &offsets,
                        unsigned int offset_index,
                        unsigned int curr_offset,
                        int64_t const dims1[],
                        int64_t const dims2[],
                        unsigned int dims_len,
                        unsigned int dims_index,
                        unsigned int mat_size) {

    if (dims_len == 0) {
        return;
    }
    if (dims_index == dims_len - 1) {
        offsets[offset_index] = curr_offset + mat_size;
        offset_index++;
        if (dims1[dims_index] == dims2[dims_index]) {
            for (int i = 1; i < dims1[dims_index]; i++) {
                offsets[offset_index] = offsets[offset_index - 1] + mat_size;
                offset_index++;
            }
        }
        else {
            if (dims1[dims_index] == 1) {
                for (int i = 1; i < dims2[dims_index]; i++) {
                    offsets[offset_index] = offsets[offset_index - 1];
                    offset_index++;
                }
            }
        }
        return;
    }
    unsigned int count = 1;
    for (int j = dims_index + 1; j < dims_len; j++) {
        count = count * dims2[j];
    }
    if (dims1[dims_index] == dims2[dims_index]) {
        int current_offset = curr_offset;
        for (int i = 0; i < dims1[dims_index]; i++) {
            fill_offset(offsets, offset_index, current_offset, dims1, dims2, dims_len,
                        dims_index + 1, mat_size);
            offset_index += count;
            current_offset = offsets[offset_index - 1];
        }
    }
    else {
        if (dims1[dims_index] == 1) {
            for (int i = 0; i < dims2[dims_index]; i++) {
                fill_offset(offsets, offset_index, curr_offset, dims1, dims2, dims_len,
                            dims_index + 1, mat_size);
                offset_index += count;
            }
        }
    }
    return;
}

static void calculate_offsets(std::vector<unsigned long> &offsets,
                              int64_t const dims1[],
                              int64_t const dims2[],
                              unsigned int dims_len,
                              unsigned long mat_dim1,
                              unsigned long mat_dim2) {
    fill_offset(offsets, 0, -1*mat_dim1*mat_dim2, dims1, dims2, dims_len, 0,
                mat_dim1*mat_dim2);
    return;
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

    matmul_helper_t helper(src_d, weights_d, dst_d);
    const int ndims = pd()->ndims();
    const int batch_ndims = ndims - 2;
    const dim_t M = helper.M();
    const dim_t N = helper.N();
    const dim_t K = helper.K();
    const dim_t batch = helper.batch();

    const bool batched = (batch_ndims > 0) ? true : false;

    const gemm_based::params_t &params = pd()->params();
    zendnnEnv zenEnvObj = readEnv();
    bool is_weights_const = pd()->weights_md()->is_memory_const;
    bool is_inplace = pd()->weights_md()->is_inplace;

    // In case of normal matrices, the stride of the last dimension will always be 1,
    // as the elements are contiguous. However in case of transposed matrix, the
    // stride of the last dimension will be greater than 1.

    const char transA = helper.transA();
    const char transB = helper.transB();
    const dim_t lda = helper.lda();
    const dim_t ldb = helper.ldb();
    const dim_t ldc = helper.ldc();

    const dim_t M_s32 = (dim_t)M;
    const dim_t N_s32 = (dim_t)N;
    const dim_t K_s32 = (dim_t)K;

    float alpha = params.get_gemm_alpha(scales);
    const float beta = params.gemm_beta_;
    //TODO(aakar): Modify f32_matmul API to include Layout
    const bool Layout = true; // CblasRowMajor

    zendnnInfo(ZENDNN_CORELOG, "zendnn_f32_matmul_t::execute_ref");
    zendnnVerbose(ZENDNN_CORELOG, "M: ",M, " N: ",N, " K: ", K,
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

    unsigned int geluType = has_eltwise_gelu?1:(has_eltwise_gelu_erf?2:0);

#if ZENDNN_ENABLE
    alpha = pd()->attr()->output_scales_.mask_ == 0 ? scales[0] : 1.0;
    float *output_scales = pd()->attr()->output_scales_.scales_;
    int scale_size = pd()->attr()->output_scales_.count_;
    std::vector<unsigned long> dst_off;
    std::vector<unsigned long> ip_off;
    std::vector<unsigned long> wei_off;

    dst_off.resize(batch);
    ip_off.resize(batch);
    wei_off.resize(batch);

    calculate_offsets(dst_off,(int64_t *)dst_d.dims(),(int64_t *)dst_d.dims(),
                      dst_d.ndims() - 2, M, N);
    calculate_offsets(ip_off,(int64_t *)src_d.dims(),(int64_t *)dst_d.dims(),
                      dst_d.ndims() - 2, M, K);
    calculate_offsets(wei_off,(int64_t *)weights_d.dims(),(int64_t *)dst_d.dims(),
                      dst_d.ndims() - 2, K, N);


    unsigned long *input_offsets = ip_off.data();
    unsigned long *dst_offsets = dst_off.data();
    unsigned long *weight_offsets = wei_off.data();

    int weights_type = pd()->weights_md()->data_type;

    if (weights_type == s4 || weights_type == s8) {
        DEFINE_WOQ_SCALES_BUFFER(woqscales);
        float *woq_scales = const_cast<float *>(woqscales);
        int woq_scale_mask = pd()->attr()->woqScales_.mask_;
        const auto scales_d = ctx.memory_mdw(ZENDNN_ARG_ATTR_WOQ_SCALES);
        int woq_scale_size = woq_scale_mask ? scales_d.dims()[0] : 1;
        const auto group_dims = pd()->attr()->woqScales_.group_dims_;
        int group_size = woq_scale_mask & (1 << (ndims - 2)) ? group_dims[0] : K;
        data_type_t woq_scales_type = pd()->attr()->woqScales_.data_type_;
        matmul_woq_wrapper(ctx, zenEnvObj, zendnn_f32, weights_type, zendnn_f32,
                           zendnn_f32, Layout,
                           transA == 'T', transB == 'T',
                           M, K, N, alpha, (char *)src, lda, (char *)weights, ldb,
                           bias == NULL ? NULL :(char *)bias,
                           pd()->attr()->post_ops_, has_eltwise_relu,
                           geluType, beta, (char *)dst, ldc, woq_scales, 0/*zp*/,
                           woq_scale_size, is_weights_const, group_size,
                           woq_scales_type);
    }
    else if (ndims == 3) {
        //3D MatMul
        zenMatMul(ctx, zenEnvObj, Layout, transA == 'T', transB == 'T',
                  batch, input_offsets,
                  weight_offsets, dst_offsets,M, K, N, alpha, (float *)src, lda,
                  (float *)weights, ldb, (float *)bias, pd()->attr()->post_ops_, has_eltwise_relu,
                  geluType, beta, (float *)dst,
                  ldc, is_weights_const, is_inplace);
    }
    else {
        //2D MatMul
        zenMatMulWithBias(ctx, zenEnvObj, Layout, transA == 'T', transB == 'T',
                          batch, input_offsets, weight_offsets, dst_offsets, M, K, N, alpha, (float *)src,
                          lda, (float *)weights, ldb,
                          (float *)bias, pd()->attr()->post_ops_, beta, (float *)dst, ldc,
                          is_weights_const, is_inplace);
    }
#endif //ZENDNN_ENABLE

    return status::success;
}

} // namespace matmul
} // namespace cpu
} // namespace impl
} // namespace zendnn
