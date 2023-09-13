/*******************************************************************************
* Modifications Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ZENDNN_USE_AOCL_BLIS_API
    #include <cblas.h>
#else // ZENDNN_USE_AOCL_BLIS_API
    #include "cblas_with_blis_api.hpp"
#endif // ZENDNN_USE_AOCL_BLIS_API

#include "common/c_types_map.hpp"
#include "common/zendnn_thread.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_primitive.hpp"
#include "cpu/platform.hpp"

#include "cpu/gemm/gemm.hpp"

#include "cpu/binary_injector_utils.hpp"
#include "cpu/matmul/zendnn_bf16_matmul.hpp"
#include "cpu/matmul/matmul_utils.hpp"

#include "zendnn_logging.hpp"
#include "common/zendnn_private.hpp"
#include "zendnn.hpp"

using namespace zendnn;

extern int graph_exe_count;

std::unordered_map<Key_matmul, const int16_t * >
matmul_weight_caching_map_s16;

void zenMatMul_gemm_bf16bf16f32of32(
    const bool Layout,
    const bool transpose_input,
    const bool transpose_filter,
    const int m,
    const int k,
    const int n,
    const float alpha,
    const int16_t *input,
    const int lda,
    const int16_t *filter,
    const int ldb,
    float *bias,
    const bool relu,
    const int gelu,
    const float beta,
    float *output,
    const int ldc
) {
    zendnnEnv zenEnvObj = readEnv();
    unsigned int thread_qty = zenEnvObj.omp_num_threads;
#ifdef ZENDNN_ENABLE_LPGEMM_V4_2
    Key_matmul key_obj;
    key_obj.transpose_input = transpose_input;
    key_obj.transpose_weights = transpose_filter;
    key_obj.m = m;
    key_obj.k = k;
    key_obj.n = n;
    key_obj.lda = lda;
    key_obj.ldb = ldb;
    key_obj.ldc = ldc;
    key_obj.weights = filter;

    //finds object in map
    auto found_obj = matmul_weight_caching_map_s16.find(key_obj);
    // Blocked BLIS API for matmul
    // Set post_ops to NULL and define reorder_param0 as 'B' for B matrix
    // Define dimentions of B matrix as reorder_param1 and reorder_param2
    // Define memory format as 'n'(non reordered) for A matrix and 'r'(reordered) for B matrix

    const char reorder_param0 = 'B';
    const dim_t reorder_param1 = k;
    const dim_t reorder_param2 = n;
    const char order = 'r';
    char trans = 'n';
    if (transpose_filter) {
        trans = 't';
    }
    char mem_format_a = 'n', mem_format_b = 'r';

    if (found_obj == matmul_weight_caching_map_s16.end()) {
        siz_t b_reorder_buf_siz_req = aocl_get_reorder_buf_size_bf16bf16f32of32(
                                          order, trans, reorder_param0, reorder_param1, reorder_param2);
        bfloat16 *reorder_filter = (bfloat16 *) aligned_alloc(64,
                                   b_reorder_buf_siz_req);
        aocl_reorder_bf16bf16f32of32(order, trans, 'B', filter, reorder_filter, k,
                                     n, ldb);
        //Create new entry
        matmul_weight_caching_map_s16[key_obj] = reorder_filter;
    }

    aocl_post_op *post_ops = NULL;

    int postop_count = 0;
    if (bias != NULL) {
        ++postop_count;
    }
    if (relu || gelu) {
        ++postop_count;
    }

    // Create postop for LPGEMM
    // Order of postops: BIAS -> RELU
    if (postop_count > 0) {
        post_ops = (aocl_post_op *) malloc(sizeof(aocl_post_op));
        dim_t max_post_ops_seq_length = postop_count;
        post_ops->seq_vector = (AOCL_POST_OP_TYPE *) malloc(max_post_ops_seq_length *
                               sizeof(AOCL_POST_OP_TYPE));

        // Iterate through each postop, check and add it if needed.
        int post_op_i = 0;
        if (bias != NULL) {
            // Add bias postop
            post_ops->seq_vector[post_op_i++] = BIAS;
            post_ops->bias.bias = (float *)bias;
        }

        if (relu) {
            // Add ReLU postop
            dim_t eltwise_index = 0;
            post_ops->seq_vector[post_op_i++] = ELTWISE;
            post_ops->eltwise = (aocl_post_op_eltwise *) malloc(sizeof(
                                    aocl_post_op_eltwise));
            (post_ops->eltwise + eltwise_index)->is_power_of_2 = FALSE;
            (post_ops->eltwise + eltwise_index)->scale_factor = NULL;
            (post_ops->eltwise + eltwise_index)->algo.alpha = NULL;
            (post_ops->eltwise + eltwise_index)->algo.beta = NULL;
            (post_ops->eltwise + eltwise_index)->algo.algo_type = RELU;
        }

        else if (gelu == 1) {
            // Gelu tanh.
            dim_t eltwise_index = 0;
            post_ops->seq_vector[post_op_i++] = ELTWISE;
            post_ops->eltwise = (aocl_post_op_eltwise *) malloc(sizeof(
                                    aocl_post_op_eltwise));
            (post_ops->eltwise + eltwise_index)->is_power_of_2 = FALSE;
            (post_ops->eltwise + eltwise_index)->scale_factor = NULL;
            (post_ops->eltwise + eltwise_index)->algo.alpha = NULL;
            (post_ops->eltwise + eltwise_index)->algo.beta = NULL;
            (post_ops->eltwise + eltwise_index)->algo.algo_type = GELU_TANH;
        }
        else if (gelu == 2) {
            // Gelu erf.
            dim_t eltwise_index = 0;
            post_ops->seq_vector[post_op_i++] = ELTWISE;
            post_ops->eltwise = (aocl_post_op_eltwise *) malloc(sizeof(
                                    aocl_post_op_eltwise));
            (post_ops->eltwise + eltwise_index)->is_power_of_2 = FALSE;
            (post_ops->eltwise + eltwise_index)->scale_factor = NULL;
            (post_ops->eltwise + eltwise_index)->algo.alpha = NULL;
            (post_ops->eltwise + eltwise_index)->algo.beta = NULL;
            (post_ops->eltwise + eltwise_index)->algo.algo_type = GELU_ERF;
        }

        post_ops->seq_length = postop_count;
    }
    //Perform MatMul using AMD BLIS
    aocl_gemm_bf16bf16f32of32(Layout? 'r' : 'c',
                              transpose_input ? 't' : 'n',
                              transpose_filter ? 't' : 'n', m, n, k, alpha,
                              input, lda, mem_format_a, matmul_weight_caching_map_s16[key_obj], ldb,
                              mem_format_b,
                              beta,
                              output, ldc,
                              post_ops);

    // Free memory for postops.
    if (bias != NULL) {
        post_ops->bias.bias=NULL;
    }
    if (relu || gelu) {
        free(post_ops->eltwise);
    }

    if (postop_count > 0) {
        free(post_ops->seq_vector);
        free(post_ops);
    }
#endif
}

void zenMatMul_gemm_bf16bf16f32obf16(
    const bool Layout,
    const bool transpose_input,
    const bool transpose_filter,
    const int m,
    const int k,
    const int n,
    const float alpha,
    const int16_t *input,
    const int lda,
    const int16_t *filter,
    const int ldb,
    float *bias,
    const bool relu,
    const int gelu,
    const float beta,
    int16_t *output,
    const int ldc,
    const float *scale,
    const int out_scale_size
) {
    zendnnEnv zenEnvObj = readEnv();
    unsigned int thread_qty = zenEnvObj.omp_num_threads;
#ifdef ZENDNN_ENABLE_LPGEMM_V4_2
    Key_matmul key_obj;
    key_obj.transpose_input = transpose_input;
    key_obj.transpose_weights = transpose_filter;
    key_obj.m = m;
    key_obj.k = k;
    key_obj.n = n;
    key_obj.lda = lda;
    key_obj.ldb = ldb;
    key_obj.ldc = ldc;
    key_obj.weights = filter;

    //finds object in map
    auto found_obj = matmul_weight_caching_map_s16.find(key_obj);
    // Blocked BLIS API for matmul
    // Set post_ops to NULL and define reorder_param0 as 'B' for B matrix
    // Define dimentions of B matrix as reorder_param1 and reorder_param2
    // Define memory format as 'n'(non reordered) for A matrix and 'r'(reordered) for B matrix

    const char reorder_param0 = 'B';
    const dim_t reorder_param1 = k;
    const dim_t reorder_param2 = n;
    const char order = 'r';
    char trans = 'n';
    if (transpose_filter) {
        trans = 't';
    }
    char mem_format_a = 'n', mem_format_b = 'r';

    // Currently filter caching disabled
    /*if (found_obj == matmul_weight_caching_map_s16.end()) {
        siz_t b_reorder_buf_siz_req = aocl_get_reorder_buf_size_bf16bf16f32of32(
                                          order, trans, reorder_param0, reorder_param1, reorder_param2);
        int16_t *reorder_filter = (int16_t *) aligned_alloc(64,
                                  b_reorder_buf_siz_req);
        aocl_reorder_bf16bf16f32of32(order, trans, 'B',filter, reorder_filter, k,
                                     n, ldb);
        //Create new entry
        matmul_weight_caching_map_s16[key_obj] = reorder_filter;
    }*/
    siz_t b_reorder_buf_siz_req = aocl_get_reorder_buf_size_bf16bf16f32of32(
                                      order, trans, reorder_param0, reorder_param1, reorder_param2);
    int16_t *reorder_filter = (int16_t *) aligned_alloc(64, b_reorder_buf_siz_req);
    aocl_reorder_bf16bf16f32of32(order, trans, 'B',filter, reorder_filter, k, n,
                                 ldb);
    //Post ops addition
    aocl_post_op *post_ops = NULL;

    int postop_count = 1;
    if (bias != NULL) {
        ++postop_count;
    }
    if (relu || gelu) {
        ++postop_count;
    }

    // Create postop for LPGEMM
    // Order of postops: BIAS -> RELU -> SCALE
    if (postop_count > 0) {
        post_ops = (aocl_post_op *) malloc(sizeof(aocl_post_op));

        if (post_ops == NULL) {
            zendnnError(ZENDNN_ALGOLOG," ZenDNN BF16 MatMul, Memory Error while allocating post ops");
            return;

        }
        dim_t max_post_ops_seq_length = postop_count;
        post_ops->seq_vector = (AOCL_POST_OP_TYPE *) malloc(max_post_ops_seq_length *
                               sizeof(AOCL_POST_OP_TYPE));

        if (post_ops->seq_vector == NULL) {
            zendnnError(ZENDNN_ALGOLOG," ZenDNN BF16 MatMul, Memory Error while allocating sequence vector");
            return;
        }
        // Iterate through each postop, check and add it if needed.
        int post_op_i = 0;

        //Set all post-ops to NULL
        post_ops->eltwise = NULL;
        post_ops->bias.bias = NULL;
        post_ops->sum.scale_factor = NULL;

        if (bias != NULL) {
            // Add bias postop
            post_ops->seq_vector[post_op_i++] = BIAS;
            post_ops->bias.bias = (float *)bias;
        }

        if (relu != 0) {
            // Add ReLU postop
            dim_t eltwise_index = 0;
            post_ops->seq_vector[post_op_i++] = ELTWISE;
            post_ops->eltwise = (aocl_post_op_eltwise *) malloc(sizeof(
                                    aocl_post_op_eltwise));

            if (post_ops->eltwise == NULL) {
            	zendnnError(ZENDNN_ALGOLOG," ZenDNN BF16 MatMul, Memory Error while allocating eltwise");
                return;
            }
            (post_ops->eltwise + eltwise_index)->is_power_of_2 = FALSE;
            (post_ops->eltwise + eltwise_index)->scale_factor = NULL;
            (post_ops->eltwise + eltwise_index)->algo.alpha = NULL;
            (post_ops->eltwise + eltwise_index)->algo.beta = NULL;
            (post_ops->eltwise + eltwise_index)->algo.algo_type = RELU;
        }
        else if (gelu == 1) {
            // Gelu tanh.
            dim_t eltwise_index = 0;
            post_ops->seq_vector[post_op_i++] = ELTWISE;
            post_ops->eltwise = (aocl_post_op_eltwise *) malloc(sizeof(
                                    aocl_post_op_eltwise));
            if (post_ops->eltwise == NULL) {
            	zendnnError(ZENDNN_ALGOLOG," ZenDNN BF16 MatMul, Memory Error while allocating eltwise");
                return;
            }
            (post_ops->eltwise + eltwise_index)->is_power_of_2 = FALSE;
            (post_ops->eltwise + eltwise_index)->scale_factor = NULL;
            (post_ops->eltwise + eltwise_index)->algo.alpha = NULL;
            (post_ops->eltwise + eltwise_index)->algo.beta = NULL;
            (post_ops->eltwise + eltwise_index)->algo.algo_type = GELU_TANH;
        }
        else if (gelu == 2) {
            // Gelu erf.
            dim_t eltwise_index = 0;
            post_ops->seq_vector[post_op_i++] = ELTWISE;
            post_ops->eltwise = (aocl_post_op_eltwise *) malloc(sizeof(
                                    aocl_post_op_eltwise));
            if (post_ops->eltwise == NULL) {
            	zendnnError(ZENDNN_ALGOLOG," ZenDNN BF16 MatMul, Memory Error while allocating eltwise");
                return;
            }
            (post_ops->eltwise + eltwise_index)->is_power_of_2 = FALSE;
            (post_ops->eltwise + eltwise_index)->scale_factor = NULL;
            (post_ops->eltwise + eltwise_index)->algo.alpha = NULL;
            (post_ops->eltwise + eltwise_index)->algo.beta = NULL;
            (post_ops->eltwise + eltwise_index)->algo.algo_type = GELU_ERF;
        }

        // Add scale postop
        post_ops->seq_vector[post_op_i] = SCALE;
        post_op_i++;

        post_ops->sum.is_power_of_2 = FALSE;
        post_ops->sum.scale_factor = NULL;
        post_ops->sum.buff = NULL;
        post_ops->sum.zero_point = NULL;

        post_ops->sum.scale_factor = malloc(n* sizeof(float));
        post_ops->sum.zero_point = malloc(n * sizeof(int16_t));

        if (post_ops->sum.scale_factor == NULL || post_ops->sum.zero_point == NULL) {
            zendnnError(ZENDNN_ALGOLOG," ZenDNN BF16 MatMul, Memory Error while allocating scale factor or zero point");
            return;
        }

        float *temp_dscale_ptr = (float *)post_ops->sum.scale_factor;
        int16_t *temp_dzero_point_ptr = (int16_t *)post_ops->sum.zero_point;
        if (out_scale_size > 1) {
            for (int i=0; i<n; ++i) {
                temp_dscale_ptr[i] = 1.0f;
                temp_dzero_point_ptr[i] = 0;
            }
        }
        else {
            for (int i=0; i<n; ++i) {
                temp_dscale_ptr[i] = (float)scale[0];
                temp_dzero_point_ptr[i] = 0;
            }
        }

        post_ops->seq_length = postop_count;
    }
    //Perform MatMul using AMD BLIS
    aocl_gemm_bf16bf16f32obf16(Layout? 'r' : 'c',
                               transpose_input ? 't' : 'n',
                               transpose_filter ? 't' : 'n', m, n, k, alpha,
                               input, lda, mem_format_a,
                               reorder_filter/*matmul_weight_caching_map_s16[key_obj]*/, ldb,
                               mem_format_b,
                               beta,
                               output, ldc,
                               post_ops);

    // Free memory for reorder filter
    free(reorder_filter);
    // Free memory for postops.
    if (post_ops != NULL) {
        free(post_ops->sum.scale_factor);
        free(post_ops->sum.zero_point);

        if (post_ops->eltwise != NULL) {
            free(post_ops->eltwise);
        }

        if (post_ops->bias.bias != NULL) {
            post_ops->bias.bias = NULL;
        }

        if (post_ops->seq_vector != NULL) {
            free(post_ops->seq_vector);
        }

        free(post_ops);
    }
#endif
}


namespace zendnn {
namespace impl {
namespace cpu {
namespace matmul {

static inline float bf16_to_float(int16_t bf16_val) {
    int32_t inter_temp = *((int16_t *) &bf16_val);
    inter_temp = inter_temp << 16;
    float float_value = 0.0;
    memcpy(&float_value, &inter_temp, sizeof(int32_t));
    return float_value;
}
using namespace data_type;

template <impl::data_type_t dst_type>
status_t zendnn_bf16_matmul_t<dst_type>::pd_t::init(engine_t *engine) {
    zendnnVerbose(ZENDNN_CORELOG, "zendnn_bf16_matmul_t::pd_t::init()");
    auto check_bias = [&]() -> bool {
        return !with_bias()
        || (utils::one_of(weights_md(1)->data_type, f32, bf16)
            && is_bias_1xN());
    };

    bool ok = src_md()->data_type == src_type
              && weights_md()->data_type == weights_type
              && desc()->accum_data_type == acc_type
              && dst_md()->data_type == dst_type
              && platform::has_data_type_support(data_type::bf16) && check_bias()
              && (ndims() - 2) < 1 /*Condition to check batched matmul*/
              && attr()->has_default_values(
                  primitive_attr_t::skip_mask_t::oscale_runtime
                  | primitive_attr_t::skip_mask_t::post_ops)
              && set_default_formats()
              && gemm_based::check_gemm_compatible_formats(*this);

    unsigned int algoType = zendnn::zendnn_getenv_int("ZENDNN_BLIS_MATMUL_BF16",0);
    if (algoType == 0) {
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

    params_.dst_is_acc_ = false;
    // set state
    params_.has_pp_kernel_ = !params_.dst_is_acc_ || with_bias()
                             || !params_.pp_attr_.has_default_values();
    return status::success;
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

    int bias_dt = data_type::f32;
    float *bias_f32 = NULL;
    if (bias!=NULL) {
        bias_dt = pd()->weights_md(1)->data_type;
        //creating float memory for bf16 bias
        if (bias_dt == data_type::bf16) {
            bias_f32 = (float *)calloc(N, sizeof(float));
            int16_t *bias_bf16 = (int16_t *)bias;
            //conversion fucntion from bf16 to f32
            for (size_t i=0; i<N; i++) {
                bias_f32[i] = bf16_to_float(bias_bf16[i]);
            }
        }
    }
    const bool batched = (batch_ndims > 0) ? true : false;

    const gemm_based::params_t &params = pd()->params();

    const auto &dst_bd = dst_d.blocking_desc();
    const auto &src_strides = &src_d.blocking_desc().strides[dst_d.ndims() - 2];
    const auto &weights_strides = &weights_d.blocking_desc().strides[dst_d.ndims() -
                                                2];
    // In case of normal matrices, the stride of the last dimension will always be 1,
    // as the elements are contiguous. However in case of transposed matrix, the
    // stride of the last dimension will be greater than 1.
    const char *transA
        = src_strides[1] == 1 && (src_strides[0] == K)? "N" : "T";
    const char *transB
        = weights_strides[1] == 1 && (weights_strides[0] == N) ? "N" : "T";

    const dim_t M_s32 = (dim_t)M;
    const dim_t N_s32 = (dim_t)N;
    const dim_t K_s32 = (dim_t)K;

    const dim_t lda = (dim_t)src_strides[*transA == 'N' ? 0 : 1];
    const dim_t ldb = (dim_t)weights_strides[*transB == 'N' ? 0 : 1];
    const dim_t ldc = (dim_t)dst_bd.strides[dst_d.ndims() - 2];

    float alpha = params.get_gemm_alpha(scales);
    const float beta = params.gemm_beta_;
    const bool Layout = true; // CblasRowMajor

    float *output_scales {nullptr};
    output_scales = pd()->attr()->output_scales_.scales_;
    int scale_size = pd()->attr()->output_scales_.count_;

    const int *zero_point_dst {nullptr};
    zero_point_dst = pd()->attr()->zero_points_.get(ZENDNN_ARG_DST);
    zendnnInfo(ZENDNN_CORELOG, "zendnn_bf16_matmul_t::execute_ref new");
    zendnnVerbose(ZENDNN_PROFLOG, "M: ",M, " N: ",N, " K: ", K,
                  " transA: ", transA, " transB: ", transB,
                  " lda: ", lda, " ldb: ", ldb, " ldc: ", ldc,
                  " alpha: ", alpha, " beta: ", beta, " batch: ", batch,
                  " Layout: ", Layout ? "CblasRowMajor(1)" : "CblasColMajor(0)", "Graph count:",
                  graph_exe_count);
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
    if (dst_type == bf16) {
        zenMatMul_gemm_bf16bf16f32obf16(Layout, strcmp(transA, "N"),strcmp(transB, "N"),
                                        M, K, N, alpha, (int16_t *)src, lda, (int16_t *)weights, ldb,
                                        bias_dt == data_type::bf16 ? (float *)bias_f32 : (float *)bias,
                                        has_eltwise_relu, geluType, beta, (int16_t *)dst, ldc, output_scales,
                                        scale_size);
    }
    else if (dst_type == f32) {
        zenMatMul_gemm_bf16bf16f32of32(Layout, strcmp(transA, "N"),strcmp(transB, "N"),
                                       M, K, N, alpha, (int16_t *)src, lda, (int16_t *)weights, ldb,
                                       bias_dt == data_type::bf16 ? (float *)bias_f32 : (float *)bias,
                                       has_eltwise_relu, geluType, beta, (float *)dst, ldc);
    }
    else {
        return status::unimplemented;
    }
#endif //ZENDNN_ENABLE

    // Free memory if bias memory is allocated
    if (bias_dt = data_type::bf16 && bias!=NULL) {
        free(bias_f32);
    }
    return status::success;
}

using namespace data_type;
template struct zendnn_bf16_matmul_t<data_type::f32>;
template struct zendnn_bf16_matmul_t<data_type::bf16>;

} // namespace matmul
} // namespace cpu
} // namespace impl
} // namespace zendnn
