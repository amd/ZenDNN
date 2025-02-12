/*******************************************************************************
* Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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
*
*******************************************************************************/

#include <atomic>
#include <float.h>
#include <math.h>
#include <cmath>

#ifndef ZENDNN_USE_AOCL_BLIS_API
    #include <cblas.h>
#else // ZENDNN_USE_AOCL_BLIS_API
    #include "cblas_with_blis_api.hpp"
#endif // ZENDNN_USE_AOCL_BLIS_API

#include "common/primitive.hpp"
#include "zendnn_logging.hpp"
#include "cpu/matmul/matmul_utils.hpp"
#include "common/zendnn_private.hpp"
#include "zendnn_helper.hpp"
#include "zendnn.hpp"
#include "zendnn_reorder_cache.hpp"

using namespace zendnn;
using tag = memory::format_tag;
using dt = memory::data_type;

extern std::mutex map_mutex;

template<typename T>
aocl_post_op *create_aocl_post_ops_woq(const impl::exec_ctx_t &ctx,
                                       const zendnn_post_ops &po,
                                       int n, const float alpha, const char *bias,
                                       int bias_type, const bool relu, const int gelu,
                                       T *sum_buff, int &postop_count,
                                       const float *scale, float *dummy_scale) {
    aocl_post_op *post_ops = NULL;
    if (bias != NULL) {
        ++postop_count;
    }

    // Create postop for LPGEMM
    // Order of postops: BIAS -> RELU
    if (po.len() + postop_count > 0) {
        post_ops = (aocl_post_op *) malloc(sizeof(aocl_post_op));
        dim_t max_post_ops_seq_length = po.len() + postop_count;
        post_ops->seq_vector = (AOCL_POST_OP_TYPE *) malloc(max_post_ops_seq_length *
                               sizeof(AOCL_POST_OP_TYPE));

        // Iterate through each postop, check and add it if needed.
        int post_op_i = 0;
        //Set all post-ops to NULL
        post_ops->eltwise = NULL;
        post_ops->bias = NULL;
        post_ops->sum = NULL;
        post_ops->matrix_add = NULL;
        post_ops->matrix_mul = NULL;

        dim_t eltwise_index = 0;
        dim_t bias_index = 0;
        dim_t scale_index = 0;
        dim_t add_index = 0;
        dim_t mul_index = 0;

        //Add bias post-op
        if (bias != NULL) {
            // Add bias postop
            post_ops->seq_vector[post_op_i++] = BIAS;
            post_ops->bias = (aocl_post_op_bias *) malloc(sizeof(
                                 aocl_post_op_bias));
            if (bias_type == zendnn_bf16) {
                (post_ops->bias + bias_index)->bias = (int16_t *)bias;
                (post_ops->bias)->stor_type = AOCL_PARAMS_STORAGE_TYPES::AOCL_GEMM_BF16;
            }
            else if (bias_type == zendnn_f32) {
                (post_ops->bias + bias_index)->bias = (float *)bias;
                (post_ops->bias)->stor_type = AOCL_PARAMS_STORAGE_TYPES::AOCL_GEMM_F32;
            }
            else {
                zendnnError(ZENDNN_ALGOLOG,
                            "Check Bias data type, only f32 and bf16 are supported");
            }
            bias_index++;
        }

        //Scale post-op
        if (scale) {
            post_ops->sum = (aocl_post_op_sum *) malloc(sizeof(
                                aocl_post_op_sum));

            post_ops->seq_vector[post_op_i++] = SCALE;
            (post_ops->sum + scale_index)->is_power_of_2 = FALSE;
            (post_ops->sum + scale_index)->scale_factor = NULL;
            (post_ops->sum + scale_index)->buff = NULL;
            (post_ops->sum + scale_index)->zero_point = NULL;

            (post_ops->sum + scale_index)->scale_factor = malloc(sizeof(float));
            (post_ops->sum + scale_index)->zero_point = malloc(sizeof(int16_t));

            //SCALE
            float *temp_dscale_ptr = (float *)(post_ops->sum + scale_index)->scale_factor;
            int16_t *temp_dzero_point_ptr = (int16_t *)(post_ops->sum +
                                            scale_index)->zero_point;
            temp_dscale_ptr[0] = (float)(scale[0]);

            temp_dzero_point_ptr[0] = (int16_t)0;

            (post_ops->sum + scale_index)->scale_factor_len = 1;
            (post_ops->sum + scale_index)->zero_point_len = 1;
            scale_index++;
        }
        //Get count of eltwise and binary post-ops
        int mem_count[3] = {0};
        for (auto idx = 0; idx < po.len(); ++idx) {
            const auto po_type = po.entry_[idx];
            switch (po_type.kind) {
            case impl::primitive_kind::eltwise:
                mem_count[0]++;
                break;
            case impl::primitive_kind::binary:
                if (po_type.binary.alg == impl::alg_kind::binary_add) {
                    mem_count[1]++;
                }
                else if (po_type.binary.alg == impl::alg_kind::binary_mul) {
                    mem_count[2]++;
                }
                break;
            default:
                break;
            }
        }

        if (mem_count[0] > 0) {
            post_ops->eltwise = (aocl_post_op_eltwise *) malloc(sizeof(
                                    aocl_post_op_eltwise)*mem_count[0]);
        }
        if (mem_count[1] > 0) {
            post_ops->matrix_add = (aocl_post_op_matrix_add *) malloc(sizeof(
                                       aocl_post_op_matrix_add)*mem_count[1]);
        }
        if (mem_count[2] > 0) {
            post_ops->matrix_mul = (aocl_post_op_matrix_mul *) malloc(sizeof(
                                       aocl_post_op_matrix_mul)*mem_count[2]);
        }

        //Add eltwise and binary post-ops in given sequence
        for (auto idx = 0; idx < po.len(); ++idx) {

            const auto po_type = po.entry_[idx];
            if (po_type.kind == impl::primitive_kind::eltwise) {

                if (po_type.eltwise.alg == impl::alg_kind::eltwise_relu) {
                    // Add ReLU postop
                    post_ops->seq_vector[post_op_i++] = ELTWISE;
                    (post_ops->eltwise + eltwise_index)->is_power_of_2 = FALSE;
                    (post_ops->eltwise + eltwise_index)->scale_factor = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.alpha = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.beta = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.algo_type = RELU;
                    eltwise_index++;
                }
                else if (po_type.eltwise.alg == impl::alg_kind::eltwise_gelu) {
                    // Gelu tanh.
                    dim_t eltwise_index = 0;
                    post_ops->seq_vector[post_op_i++] = ELTWISE;
                    (post_ops->eltwise + eltwise_index)->is_power_of_2 = FALSE;
                    (post_ops->eltwise + eltwise_index)->scale_factor = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.alpha = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.beta = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.algo_type = GELU_TANH;
                    eltwise_index++;
                }
                else if (po_type.eltwise.alg == impl::alg_kind::eltwise_gelu_erf) {
                    // Gelu erf.
                    dim_t eltwise_index = 0;
                    post_ops->seq_vector[post_op_i++] = ELTWISE;
                    (post_ops->eltwise + eltwise_index)->is_power_of_2 = FALSE;
                    (post_ops->eltwise + eltwise_index)->scale_factor = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.alpha = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.beta = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.algo_type = GELU_ERF;
                    eltwise_index++;
                }
                else if (po_type.eltwise.alg == impl::alg_kind::eltwise_logistic) {
                    // Sigmoid
                    dim_t eltwise_index = 0;
                    post_ops->seq_vector[post_op_i++] = ELTWISE;
                    (post_ops->eltwise + eltwise_index)->is_power_of_2 = FALSE;
                    (post_ops->eltwise + eltwise_index)->scale_factor = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.alpha = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.beta = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.algo_type = SIGMOID;
                    eltwise_index++;
                }
                else if (po_type.eltwise.alg == impl::alg_kind::eltwise_swish) {
                    // Gelu erf.
                    dim_t eltwise_index = 0;
                    post_ops->seq_vector[post_op_i++] = ELTWISE;
                    (post_ops->eltwise + eltwise_index)->is_power_of_2 = FALSE;
                    (post_ops->eltwise + eltwise_index)->scale_factor = NULL;
                    T alpha_val = (T)po_type.eltwise.alpha;
                    (post_ops->eltwise + eltwise_index)->algo.alpha = malloc(sizeof(float));
                    *((float *)(post_ops->eltwise + eltwise_index)->algo.alpha) = (float)1;
                    (post_ops->eltwise + eltwise_index)->algo.beta = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.algo_type = SWISH;
                    eltwise_index++;
                }
            }
            else if (po_type.kind == impl::primitive_kind::binary) {
                if (po_type.binary.alg == impl::alg_kind::binary_add) {
                    post_ops->seq_vector[post_op_i++] = MATRIX_ADD;
                    (post_ops->matrix_add + add_index)->ldm = n;
                    auto binary_po = CTX_IN_MEM(const void *,
                                                (ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(idx) | ZENDNN_ARG_SRC_1));
                    //Currently putting 1.0 as scale
                    (post_ops->matrix_add + add_index)->scale_factor = (float *)dummy_scale;
                    (post_ops->matrix_add + add_index)->scale_factor_len = 1;
                    auto b_dt = po_type.binary.src1_desc.data_type;
                    if (b_dt == zendnn_bf16) {
                        (post_ops->matrix_add + add_index)->stor_type =
                            AOCL_PARAMS_STORAGE_TYPES::AOCL_GEMM_BF16;
                        auto addA = reinterpret_cast<int16_t *>(const_cast<void *>(binary_po));
                        (post_ops->matrix_add + add_index)->matrix = (int16_t *)addA;
                    }
                    else if (b_dt == zendnn_f32) {
                        (post_ops->matrix_add + add_index)->stor_type =
                            AOCL_PARAMS_STORAGE_TYPES::AOCL_GEMM_F32;
                        auto addA = reinterpret_cast<float *>(const_cast<void *>(binary_po));
                        (post_ops->matrix_add + add_index)->matrix = (float *)addA;
                    }
                    add_index++;
                }
                else if (po_type.binary.alg == impl::alg_kind::binary_mul) {
                    post_ops->seq_vector[post_op_i++] = MATRIX_MUL;
                    (post_ops->matrix_mul + mul_index)->ldm = n;
                    auto binary_po = CTX_IN_MEM(const void *,
                                                (ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(idx) | ZENDNN_ARG_SRC_1));
                    //Currently putting 1.0 as scale
                    (post_ops->matrix_mul + mul_index)->scale_factor = (float *)dummy_scale;
                    (post_ops->matrix_mul + mul_index)->scale_factor_len = 1;
                    auto b_dt = po_type.binary.src1_desc.data_type;
                    if (b_dt == zendnn_bf16) {
                        (post_ops->matrix_mul + mul_index)->stor_type =
                            AOCL_PARAMS_STORAGE_TYPES::AOCL_GEMM_BF16;
                        auto mulA = reinterpret_cast<int16_t *>(const_cast<void *>(binary_po));
                        (post_ops->matrix_mul + mul_index)->matrix = (int16_t *)mulA;
                    }
                    else if (b_dt == zendnn_f32) {
                        (post_ops->matrix_mul + mul_index)->stor_type =
                            AOCL_PARAMS_STORAGE_TYPES::AOCL_GEMM_F32;
                        auto mulA = reinterpret_cast<float *>(const_cast<void *>(binary_po));
                        (post_ops->matrix_mul + mul_index)->matrix = (float *)mulA;
                    }
                    mul_index++;
                }
            }
            // Using sum post-op with GEMM_beta
            else if (po_type.kind == impl::primitive_kind::sum) {
                postop_count-=1; //Since sum post-op is applied as GEMM_beta.
            }
        }
        post_ops->seq_length = po.len() + postop_count;
    }
    return (post_ops);
}

int ref_woq_bf16(
    const impl::exec_ctx_t &ctx,
    const impl::post_ops_t &po_ops,
    int src_type,
    int weights_type,
    int dst_type,
    int bias_type,
    const bool Layout,
    const bool transA,
    const bool transB,
    const int M,
    const int K,
    const int N,
    const float alpha,
    const int16_t *src,
    const int lda,
    const int8_t *weights,
    const int ldb,
    const char *bias,
    const bool has_eltwise_relu,
    const int geluType,
    const float beta,
    char *output,
    const int ldc,
    float *wei_scale,
    const int32_t zero_point_weights,
    int scale_size,
    float do_sum,
    bool is_weights_const,
    int group_size,
    zendnn_data_type_t scale_dt
) {
    zendnnEnv zenEnvObj = readEnv();
    zendnnVerbose(ZENDNN_PROFLOG,"aocl bf16 kernel");

    unsigned int thread_qty = zenEnvObj.omp_num_threads;
    //TODO: Create cleaner key for weight caching map
    //Putting hardcoded values for now
    Key_matmul key_obj(transA, transB, M, K, N, lda, ldb, ldc, weights, thread_qty,
                       true);

    // Blocked BLIS API for matmul
    // Set post_ops to NULL and define reorder_param0 as 'B' for B matrix
    // Define dimentions of B matrix as reorder_param1 and reorder_param2
    // Define memory format as 'n'(non reordered) for A matrix and 'r'(reordered) for B matrix
    const char reorder_param0 = 'B';
    const dim_t reorder_param1 = K;
    const dim_t reorder_param2 = N;
    const char order = 'r';
    char trans = 'n';
    if (transB) {
        trans = 't';
    }
    char mem_format_a = 'n', mem_format_b = 'r';

    int16_t *reorder_filter = NULL;

    woqReorderAndCacheWeightsAocl<int16_t>(key_obj, weights, reorder_filter, K, N,
                                           ldb, is_weights_const, order, trans, reorder_param0, reorder_param1,
                                           reorder_param2,
                                           aocl_get_reorder_buf_size_bf16bf16f32of32, aocl_reorder_bf16bf16f32of32,
                                           weights_type, wei_scale, scale_size, group_size, scale_dt
                                          );

    aocl_post_op *post_ops = NULL;
    float dummy_scale = (float)1.0;
    if (dst_type == zendnn_bf16) {
        int postop_count = bias == NULL ? 0:1;
        post_ops = create_aocl_post_ops_woq<int16_t>(ctx, po_ops, N,
                   alpha, (const char *)bias, bias_type,
                   has_eltwise_relu, geluType, NULL/*sum with beta*/,
                   postop_count, postop_count ? &alpha : NULL, &dummy_scale);
        //Perform MatMul using AMD BLIS
        aocl_gemm_bf16bf16f32obf16(Layout? 'r' : 'c',
                                   transA ? 't' : 'n',
                                   transB ? 't' : 'n', M, N, K,
                                   bias == NULL ? alpha : 1.0,
                                   src, lda, mem_format_a,
                                   reorder_filter, ldb,
                                   mem_format_b,
                                   beta,
                                   (int16_t *)output, ldc,
                                   post_ops);
    }
    else {
        int postop_count = bias == NULL ? 0:1;
        post_ops = create_aocl_post_ops_woq<float>(ctx, po_ops, N,
                   alpha, (const char *)bias, bias_type,
                   has_eltwise_relu, geluType, NULL/*sum with beta*/,
                   postop_count, postop_count ? &alpha : NULL,
                   &dummy_scale);
        aocl_gemm_bf16bf16f32of32(Layout? 'r' : 'c',
                                  transA ? 't' : 'n',
                                  transB ? 't' : 'n', M, N, K,
                                  bias == NULL ? alpha : 1.0,
                                  src, lda, mem_format_a,
                                  reorder_filter, ldb,
                                  mem_format_b,
                                  beta,
                                  (float *)output, ldc,
                                  post_ops);
    }
    // Free memory for postops.
    if (post_ops) {
        if (bias != NULL) {
            post_ops->bias->bias=NULL;
            free(post_ops->bias);
        }
        if (post_ops->eltwise != NULL) {
            if (post_ops->eltwise->algo.alpha != NULL) {
                free(post_ops->eltwise->algo.alpha);
            }
            free(post_ops->eltwise);
        }
        if (post_ops->sum) {
            free(post_ops->sum->scale_factor);
            free(post_ops->sum->zero_point);
            free(post_ops->sum);
        }
        if (post_ops->matrix_add != NULL) {
            free(post_ops->matrix_add);
        }
        if (post_ops->matrix_mul != NULL) {
            free(post_ops->matrix_mul);
        }
        free(post_ops->seq_vector);
        free(post_ops);
    }
    if (!is_weights_const) {
        free(reorder_filter);
    }
    return 0;
}

int ref_woq_f32(
    const impl::exec_ctx_t &ctx,
    const impl::post_ops_t &po_ops,
    int src_type,
    int weights_type,
    int dst_type,
    int bias_type,
    const bool Layout,
    const bool transA,
    const bool transB,
    const int M,
    const int K,
    const int N,
    const float alpha,
    const float *src,
    const int lda,
    const int8_t *weights,
    const int ldb,
    const float *bias,
    const bool has_eltwise_relu,
    const int geluType,
    const float beta,
    float *output,
    const int ldc,
    float *wei_scale,
    const int32_t zero_point_weights,
    int scale_size,
    float do_sum,
    bool is_weights_const,
    int group_size,
    zendnn_data_type_t scale_dt
) {
    zendnnEnv zenEnvObj = readEnv();
    unsigned int thread_qty = zenEnvObj.omp_num_threads;
    //TODO: Create cleaner key for weight caching map
    //Putting hardcoded values for now
    Key_matmul key_obj(transA, transB, M, K, N, lda, ldb, ldc, weights, thread_qty,
                       true);

    // Blocked BLIS API for matmul
    // Set post_ops to NULL and define reorder_param0 as 'B' for B matrix
    // Define dimentions of B matrix as reorder_param1 and reorder_param2
    // Define memory format as 'n'(non reordered) for A matrix and 'r'(reordered) for B matrix
    const char reorder_param0 = 'B';
    const dim_t reorder_param1 = K;
    const dim_t reorder_param2 = N;
    const char order = 'r';
    char trans = 'n';
    if (transB) {
        trans = 't';
    }
    char mem_format_a = 'n', mem_format_b = 'r';

    float *reorder_filter = NULL;

    woqReorderAndCacheWeightsAocl<float>(key_obj, weights, reorder_filter, K, N,
                                         ldb, is_weights_const, order, trans, reorder_param0, reorder_param1,
                                         reorder_param2,
                                         aocl_get_reorder_buf_size_f32f32f32of32, aocl_reorder_f32f32f32of32,
                                         weights_type, wei_scale, scale_size, group_size, scale_dt
                                        );

    aocl_post_op *post_ops = NULL;
    int postop_count = 0;
    float dummy_scale = (float)1.0;
    post_ops = create_aocl_post_ops_woq<float>(ctx, po_ops, N,
               alpha, bias == NULL ? NULL :(const char *)bias, bias_type,
               has_eltwise_relu, geluType, (float *)output,
               postop_count, NULL, &dummy_scale);
    aocl_gemm_f32f32f32of32(Layout? 'r' : 'c',
                            transA ? 't' : 'n',
                            transB ? 't' : 'n', M, N, K,
                            alpha,
                            src, lda, mem_format_a,
                            reorder_filter, ldb,
                            mem_format_b,
                            beta,
                            (float *)output, ldc,
                            post_ops);

    // Free memory for postops.
    if (post_ops != NULL) {
        if (bias != NULL) {
            post_ops->bias->bias=NULL;
            free(post_ops->bias);
        }
        if (post_ops->eltwise != NULL) {
            if (post_ops->eltwise->algo.alpha != NULL) {
                free(post_ops->eltwise->algo.alpha);
            }
            free(post_ops->eltwise);
        }
        if (post_ops->sum != NULL) {
            free(post_ops->sum->scale_factor);
            free(post_ops->sum->zero_point);
            free(post_ops->sum);
        }
        if (post_ops->matrix_add != NULL) {
            free(post_ops->matrix_add);
        }
        if (post_ops->matrix_mul != NULL) {
            free(post_ops->matrix_mul);
        }
        free(post_ops->seq_vector);
        free(post_ops);
    }
    if (!is_weights_const) {
        free(reorder_filter);
    }
    return 0;
}
void zenMatMulPrimitiveIntComputeBF16(const impl::exec_ctx_t &ctx,
                                      zendnnEnv zenEnvObj,
                                      int weights_type, int dst_type, int bias_type,
                                      const bool Layout,
                                      const bool TransA, const bool TransB, const int M,
                                      const int N, const int K,
                                      const int16_t *A_Array,
                                      const int8_t *B_Array,
                                      const char *bias, void *C_Array, const float alpha,
                                      const float beta, const int lda, const int ldb,
                                      const int ldc, const impl::post_ops_t &po_ops,
                                      bool blocked_format, float *wei_scale,
                                      const int32_t zero_point_weights, int scale_size,
                                      bool is_weights_const, int group_size,
                                      zendnn_data_type_t scale_dt) {
    zendnn::engine eng(engine::kind::cpu, 0);
    zendnn::stream engine_stream(eng);
    zendnnVerbose(ZENDNN_PROFLOG,"JIT kernel woq");
    //TODO: Create cleaner key for weight caching map
    //Putting hardcoded values for now
    Key_matmul key_obj_reorder;
    key_obj_reorder.transpose_input = TransA;
    key_obj_reorder.transpose_weights = TransB;
    key_obj_reorder.m = M;
    key_obj_reorder.k = K;
    key_obj_reorder.n = N;
    key_obj_reorder.lda = lda;
    key_obj_reorder.ldb = ldb;
    key_obj_reorder.ldc = 0;
    key_obj_reorder.weights = B_Array;
    key_obj_reorder.thread_count = 0;

    std::unordered_map<int, memory> net_args;

    int16_t *in_arr = const_cast<int16_t *>(A_Array);
    int8_t *weights = const_cast<int8_t *>(B_Array);
    char *bias_arr = const_cast<char *>(bias);

    memory::dims src_dims = {M, K};
    memory::dims weight_dims = {K, N};
    memory::dims bias_dims = {1, N};
    memory::dims dst_dims = {M, N};

    memory::dims a_strides = TransA ? memory::dims {1, lda} :
                             memory::dims {lda, 1};
    memory::dims b_strides = TransB ? memory::dims {1, ldb} :
                             memory::dims {ldb, 1};

    memory::desc src_md = memory::desc({src_dims}, dt::bf16, a_strides);
    memory::desc matmul_weights_md = memory::desc({weight_dims}, dt::bf16,
                                     b_strides);
    memory::desc blocked_matmul_weights_md = memory::desc({weight_dims}, dt::bf16,
            tag::any);

    memory::desc bias_md;
    //Bias type bf16 or f32
    if (bias_type == 2)
        bias_md = memory::desc({bias_dims}, dt::bf16, tag::ab);
    else if (bias_type == 3)
        bias_md = memory::desc({bias_dims}, dt::f32, tag::ab);

    memory::desc dst_md = memory::desc({dst_dims}, dst_type == zendnn_f32 ?
                                       dt::f32 :
                                       dt::bf16, {ldc, 1});
    primitive_attr matmul_attr;
    zendnn::post_ops post_ops;
    bool post_attr = false;
    const float scale = 1.0f;
    zendnn::memory po_memory[po_ops.len()];
    for (auto idx = 0; idx < po_ops.len(); ++idx) {
        const auto &e = po_ops.entry_[idx];
        if (e.eltwise.alg == impl::alg_kind::eltwise_relu) {
            post_attr = true;
            post_ops.append_eltwise(scale, algorithm::eltwise_relu, 0.f, 0.f);
            po_memory[idx] = memory({{M,N},dt::f32,tag::ab},eng,nullptr);
        }
        else if (e.eltwise.alg == impl::alg_kind::eltwise_swish) {
            post_attr = true;
            post_ops.append_eltwise(scale, algorithm::eltwise_swish, 1.f, 0.f);
            po_memory[idx] = memory({{M,N},dt::f32,tag::ab},eng,nullptr);
        }
        else if (e.eltwise.alg == impl::alg_kind::eltwise_gelu) {
            post_attr = true;
            post_ops.append_eltwise(scale, algorithm::eltwise_gelu, 1.f, 0.f);
            po_memory[idx] = memory({{M,N},dt::f32,tag::ab},eng,nullptr);
        }
        else if (e.eltwise.alg == impl::alg_kind::eltwise_gelu_erf) {
            post_attr = true;
            post_ops.append_eltwise(scale, algorithm::eltwise_gelu_erf, 1.f, 0.f);
            po_memory[idx] = memory({{M,N},dt::f32,tag::ab},eng,nullptr);
        }
        else if (e.eltwise.alg == impl::alg_kind::eltwise_logistic) {
            //Sigmoid
            post_attr = true;
            post_ops.append_eltwise(scale, algorithm::eltwise_logistic, 0.f, 0.f);
            po_memory[idx] = memory({{M,N},dt::f32,tag::ab},eng,nullptr);
        }
        else if (e.kind == impl::primitive_kind::sum) {
            post_attr = true;
            if (beta != 0.f) {
                post_ops.append_sum(beta);
            }
            po_memory[idx] = memory({{M,N},dt::f32,tag::ab},eng,nullptr);
        }
        else if (e.binary.alg == impl::alg_kind::binary_add) {
            post_attr = true;
            const auto &src1_desc = e.binary.src1_desc;
            post_ops.append_binary(algorithm::binary_add, src1_desc);
            auto add_raw = const_cast<void *>(CTX_IN_MEM(const void *,
                                              (ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(idx) | ZENDNN_ARG_SRC_1)));
            po_memory[idx] = memory(src1_desc,eng,add_raw);
            int t = ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(idx) | ZENDNN_ARG_SRC_1;
            net_args.insert({t,po_memory[idx]});
        }
        else if (e.binary.alg == impl::alg_kind::binary_mul) {
            post_attr = true;
            const auto &src1_desc = e.binary.src1_desc;
            post_ops.append_binary(algorithm::binary_mul, src1_desc);
            auto mul_raw = const_cast<void *>(CTX_IN_MEM(const void *,
                                              (ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(idx) | ZENDNN_ARG_SRC_1)));
            po_memory[idx] = memory(src1_desc,eng,mul_raw);
            int t = ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(idx) | ZENDNN_ARG_SRC_1;
            net_args.insert({t,po_memory[idx]});
        }
    }
    if (alpha != 1.f) {
        matmul_attr.set_output_scales(0, {alpha});
    }
    if (post_attr) {
        matmul_attr.set_post_ops(post_ops);
    }
    matmul_attr.set_autoTunerEnable(true);

    //MatMul desc
    auto matmul_disc = blocked_format ? bias_type? zendnn::matmul::desc(src_md,
                       blocked_matmul_weights_md, bias_md, dst_md): zendnn::matmul::desc(src_md,
                               blocked_matmul_weights_md, dst_md) : bias_type? zendnn::matmul::desc(src_md,
                                       matmul_weights_md, bias_md, dst_md): zendnn::matmul::desc(src_md,
                                               matmul_weights_md, dst_md);

    //MatMul primitive desc
    auto matmul_prim_disc =
        zendnn::matmul::primitive_desc(matmul_disc, matmul_attr, eng);

    //Memory creation
    zendnn::memory user_weights_memory, src_memory, bias_memory, dst_memory;
    src_memory = memory(src_md, eng, in_arr);
    if (bias_type) {
        bias_memory = memory(bias_md, eng, bias_arr);
    }
    dst_memory = memory(dst_md, eng, C_Array);
    //Weight reordering
    zendnn::memory reordered_weights_memory;

    int16_t *wei_bf16 = NULL;
    if (blocked_format) {
        woqReorderAndCacheWeightsBrgemm(
            key_obj_reorder, matmul_prim_disc, user_weights_memory,
            reordered_weights_memory, eng, engine_stream, matmul_weights_md,
            is_weights_const, weights, weights_type, K, N, wei_scale, scale_size,
            group_size, scale_dt);
    }
    else {
        wei_bf16 = (int16_t *)zendnn_aligned_alloc(64, sizeof(int16_t)*K*N);

        if (weights_type == zendnn_s4) { //Convert S4 to BF16
            cvt_int4_to_bf16(weights, wei_bf16, K, N, wei_scale, scale_size, group_size,
                             scale_dt);
        }
        else { //Convert S8 to BF16
            cvt_int8_to_bf16(weights, wei_bf16, K, N, wei_scale, scale_size, group_size,
                             scale_dt);
        }
        user_weights_memory = memory(matmul_weights_md, eng, wei_bf16);
    }

    //net.push_back(zendnn::matmul(matmul_prim_disc));
    zendnn::matmul matmul_prim = zendnn::matmul(matmul_prim_disc);
    net_args.insert({ZENDNN_ARG_SRC, src_memory});
    net_args.insert({ZENDNN_ARG_WEIGHTS, blocked_format?reordered_weights_memory:user_weights_memory});
    if (bias_type) net_args.insert({ZENDNN_ARG_BIAS,bias_memory});
    net_args.insert({ZENDNN_ARG_DST,dst_memory});
    matmul_prim.execute(engine_stream, net_args);

    if (!blocked_format) {
        free(wei_bf16);
    }
}

int aocl_woq_bf16(
    const impl::exec_ctx_t &ctx,
    const impl::post_ops_t &po_ops,
    int src_type,
    int weights_type,
    int dst_type,
    int bias_type,
    const bool Layout,
    const bool transA,
    const bool transB,
    const int M,
    const int K,
    const int N,
    const float alpha,
    const int16_t *src,
    const int lda,
    const int8_t *weights,
    const int ldb,
    const char *bias,
    const bool has_eltwise_relu,
    const int geluType,
    const float beta,
    char *output,
    const int ldc,
    float *wei_scale,
    const int32_t zero_point_weights,
    int scale_size,
    float do_sum,
    bool is_weights_const,
    int group_size,
    zendnn_data_type_t scale_dt
) {
    zendnnEnv zenEnvObj = readEnv();
    unsigned int thread_qty = zenEnvObj.omp_num_threads;
    //TODO: Create cleaner key for weight caching map
    //Putting hardcoded values for now
    Key_matmul key_obj(transA, transB, M, K, N, lda, ldb, ldc, weights, thread_qty,
                       true);

    zendnnVerbose(ZENDNN_PROFLOG,"aocl_bf16s4 kernel");

    // Blocked BLIS API for matmul
    // Set post_ops to NULL and define reorder_param0 as 'B' for B matrix
    // Define dimentions of B matrix as reorder_param1 and reorder_param2
    // Define memory format as 'n'(non reordered) for A matrix and 'r'(reordered) for B matrix
    const char reorder_param0 = 'B';
    const dim_t reorder_param1 = K;
    const dim_t reorder_param2 = N;
    const char order = 'r';
    char trans = 'n';
    if (transB) {
        trans = 't';
    }
    char mem_format_a = 'n', mem_format_b = 'r';

    int8_t *reorder_filter = NULL;

    reorderAndCacheWeights<int8_t>(key_obj, weights, reorder_filter, K, N,
                                   ldb, is_weights_const, 0, order, trans, reorder_param0, reorder_param1,
                                   reorder_param2,
                                   aocl_get_reorder_buf_size_bf16s4f32of32, aocl_reorder_bf16s4f32of32
                                  );

    aocl_post_op *post_ops = NULL;

    float dummy_scale = (float)1.0;
    // postop_count always 1 for WOQ cases
    int postop_count = 1;
    float val = 1.0;
    if (dst_type == zendnn_bf16) {
        post_ops = create_aocl_post_ops_woq<int16_t>(ctx, po_ops, N,
                   alpha, (const char *)bias, bias_type, has_eltwise_relu, geluType,
                   NULL/*sum with beta*/, postop_count, bias !=NULL ? &alpha : &val,
                   &dummy_scale);

        //Add pre-op for S4 API
        post_ops->pre_ops = NULL;
        post_ops->pre_ops = (aocl_pre_op *)malloc(sizeof(aocl_pre_op));
        (post_ops->pre_ops)->b_zp = (aocl_pre_op_zp *)malloc(sizeof(aocl_pre_op_zp));
        (post_ops->pre_ops)->b_scl = (aocl_pre_op_sf *)malloc(sizeof(aocl_pre_op_sf));
        /* Only int8_t zero point supported in pre-ops. */
        ((post_ops->pre_ops)->b_zp)->zero_point = NULL;
        ((post_ops->pre_ops)->b_zp)->zero_point_len = 0;
        /* Only float scale factor supported in pre-ops. */
        ((post_ops->pre_ops)->b_scl)->scale_factor = (float *)wei_scale;
        ((post_ops->pre_ops)->b_scl)->scale_factor_len = scale_size;
        // Pass the scales datatype
        if (scale_dt == zendnn_bf16) {
            ((post_ops->pre_ops)->b_scl)->scale_factor_type =
                AOCL_PARAMS_STORAGE_TYPES::AOCL_GEMM_BF16;
        }
        else if (scale_dt == zendnn_f32) {
            ((post_ops->pre_ops)->b_scl)->scale_factor_type =
                AOCL_PARAMS_STORAGE_TYPES::AOCL_GEMM_F32;
        }
        else {
            zendnnError(ZENDNN_ALGOLOG,
                        "Check scales data type, only f32 and bf16 are supported");
        }
        (post_ops->pre_ops)->seq_length = 1;
        (post_ops->pre_ops)->group_size = group_size;

        //Perform MatMul using AMD BLIS
        aocl_gemm_bf16s4f32obf16(Layout? 'r' : 'c',
                                 transA ? 't' : 'n',
                                 transB ? 't' : 'n', M, N, K,
                                 bias == NULL ? alpha : 1.0,
                                 src, lda, mem_format_a,
                                 reorder_filter, ldb,
                                 mem_format_b,
                                 beta,
                                 (int16_t *)output, ldc,
                                 post_ops);
    }
    else {
        post_ops = create_aocl_post_ops_woq<float>(ctx, po_ops, N, alpha,
                   (const char *)bias, bias_type, has_eltwise_relu, geluType,
                   NULL/*sum with beta*/, postop_count,
                   bias !=NULL ? &alpha : &val, &dummy_scale);
        //Add pre-op for S4 API
        post_ops->pre_ops = NULL;
        post_ops->pre_ops = (aocl_pre_op *)malloc(sizeof(aocl_pre_op));
        (post_ops->pre_ops)->b_zp = (aocl_pre_op_zp *)malloc(sizeof(aocl_pre_op_zp));
        (post_ops->pre_ops)->b_scl = (aocl_pre_op_sf *)malloc(sizeof(aocl_pre_op_sf));
        /* Only int8_t zero point supported in pre-ops. */
        ((post_ops->pre_ops)->b_zp)->zero_point = NULL;
        ((post_ops->pre_ops)->b_zp)->zero_point_len = 0;
        /* Only float scale factor supported in pre-ops. */
        ((post_ops->pre_ops)->b_scl)->scale_factor = (float *)wei_scale;
        ((post_ops->pre_ops)->b_scl)->scale_factor_len = scale_size;
        // Pass the scales datatype
        if (scale_dt == zendnn_bf16) {
            ((post_ops->pre_ops)->b_scl)->scale_factor_type =
                AOCL_PARAMS_STORAGE_TYPES::AOCL_GEMM_BF16;
        }
        else if (scale_dt == zendnn_f32) {
            ((post_ops->pre_ops)->b_scl)->scale_factor_type =
                AOCL_PARAMS_STORAGE_TYPES::AOCL_GEMM_F32;
        }
        else {
            zendnnError(ZENDNN_ALGOLOG,
                        "Check scales data type, only f32 and bf16 are supported");
        }
        (post_ops->pre_ops)->seq_length = 1;
        (post_ops->pre_ops)->group_size = group_size;

        aocl_gemm_bf16s4f32of32(Layout? 'r' : 'c',
                                transA ? 't' : 'n',
                                transB ? 't' : 'n', M, N, K,
                                bias == NULL ? alpha : 1.0,
                                src, lda, mem_format_a,
                                reorder_filter, ldb,
                                mem_format_b,
                                beta,
                                (float *)output, ldc,
                                post_ops);
    }
    // Free memory for postops.
    if (post_ops) {
        if (bias != NULL) {
            post_ops->bias->bias=NULL;
            free(post_ops->bias);
        }
        if (post_ops->eltwise != NULL) {
            if (post_ops->eltwise->algo.alpha != NULL) {
                free(post_ops->eltwise->algo.alpha);
            }
            free(post_ops->eltwise);
        }
        if (post_ops->sum != NULL) {
            free(post_ops->sum->scale_factor);
            free(post_ops->sum->zero_point);
            free(post_ops->sum);
        }
        if (post_ops->matrix_add != NULL) {
            free(post_ops->matrix_add);
        }
        if (post_ops->matrix_mul != NULL) {
            free(post_ops->matrix_mul);
        }
        if (post_ops->seq_vector) {
            free(post_ops->seq_vector);
        }
        ((post_ops->pre_ops)->b_zp)->zero_point = NULL;
        ((post_ops->pre_ops)->b_scl)->scale_factor = NULL;
        free((post_ops->pre_ops)->b_zp);
        free((post_ops->pre_ops)->b_scl);
        free(post_ops->pre_ops);
        free(post_ops);
    }
    if (!is_weights_const) {
        free(reorder_filter);
    }
    return 0;
}

int matmul_woq_wrapper(
    const impl::exec_ctx_t &ctx,
    int src_type,
    int weights_type,
    int dst_type,
    int bias_type,
    const bool Layout,
    const bool transA,
    const bool transB,
    const int M,
    const int K,
    const int N,
    const float alpha,
    const char *src,
    const int lda,
    const char *weights,
    const int ldb,
    const char *bias,
    const impl::post_ops_t &po_ops,
    const bool has_eltwise_relu,
    const int geluType,
    const float beta,
    char *dst,
    const int ldc,
    float *wei_scale,
    const int32_t zero_point_weights,
    int scale_size,
    float do_sum,
    bool is_weights_const,
    int group_size,
    zendnn_data_type_t scale_dt
) {
    //WOQ kernel
    zendnnEnv zenEnvObj = readEnv();
    zendnnOpInfo &obj = zendnnOpInfo::ZenDNNOpInfo();

    //Check data_types of dst, post-ops
    bool can_run_aocl = impl::cpu::matmul::check_dt_(po_ops, dst_type);

    //Due to limitation of current aocl kernels
    //using jit call for cases where BIAS, alpha and beta
    //all are available
    int use_jit = (bias && alpha != 1.0 && beta != 0.0);

    //TODO: Seperate Implementation of Autotuner for WOQ(MATMUL_AUTO_BF16)
    if (src_type == zendnn_bf16) {
        if (zenEnvObj.zenBF16GEMMalgo == zenBF16MatMulAlgoType::MATMUL_AUTO_BF16 ||
                zenEnvObj.zenBF16GEMMalgo == zenBF16MatMulAlgoType::MATMUL_DT_BF16) {
            // For Higher thread count(i.e >128) AOCL S4 Kernels gives optimal performance
            if (zenEnvObj.omp_num_threads > 128) {
                zenEnvObj.zenBF16GEMMalgo = zenBF16MatMulAlgoType::MATMUL_BLOCKED_AOCL_BF16;
            }
            else {
                // If M <= 16 AOCL S4 Kernel gives Optimal Performance.
                // If M >= 128, N and K >=1024 AOCL BLIS kernels with Zen weights conversion
                // gives optimal performance.
                // This is based on heuristic with different models and difference BS
                if (M <= 16) {
                    // AOCL S4 Kernel
                    zenEnvObj.zenBF16GEMMalgo = zenBF16MatMulAlgoType::MATMUL_BLOCKED_AOCL_BF16;
                }
                else if (M >= 128 && N >= 1024 && K >= 1024) {
                    // AOCL BF16 Kernel with Zen Weights Conversion
                    zenEnvObj.zenBF16GEMMalgo = zenBF16MatMulAlgoType::MATMUL_AOCL_BF16;
                }
                else if (M == 32) {
                    if (N <= K) {
                        // Blocked BRGEMM BF16 with Zen Weights Conversion
                        zenEnvObj.zenBF16GEMMalgo = zenBF16MatMulAlgoType::MATMUL_BLOCKED_JIT_BF16;
                    }
                    else {
                        // AOCL S4 Kernel
                        zenEnvObj.zenBF16GEMMalgo = zenBF16MatMulAlgoType::MATMUL_BLOCKED_AOCL_BF16;
                    }
                }
                else {
                    if (N <= K) {
                        // AOCL BF16 Kernel with Zen Weights Conversion
                        zenEnvObj.zenBF16GEMMalgo = zenBF16MatMulAlgoType::MATMUL_AOCL_BF16;
                    }
                    else {
                        // Blocked BRGEMM BF16 with Zen Weights Conversion
                        zenEnvObj.zenBF16GEMMalgo = zenBF16MatMulAlgoType::MATMUL_BLOCKED_JIT_BF16;
                    }
                }
            }
        }

        //If algo is AOCL but can't execute due to limited mixed_data type support
        //then run blocked brgemm
        if ((zenEnvObj.zenBF16GEMMalgo ==
                zenBF16MatMulAlgoType::MATMUL_BLOCKED_AOCL_BF16
                || zenEnvObj.zenBF16GEMMalgo == zenBF16MatMulAlgoType::MATMUL_AOCL_BF16) &&
                (use_jit || !can_run_aocl)) {
            zenEnvObj.zenBF16GEMMalgo = zenBF16MatMulAlgoType::MATMUL_BLOCKED_JIT_BF16;
        }

        if (zenEnvObj.zenBF16GEMMalgo == zenBF16MatMulAlgoType::MATMUL_BLOCKED_AOCL_BF16
                && weights_type == zendnn_s4) {
            aocl_woq_bf16(ctx, po_ops, src_type, weights_type, dst_type, bias_type, Layout,
                          transA, transB,
                          M, K, N, alpha, (int16_t *)src, lda, (int8_t *)weights, ldb, bias,
                          has_eltwise_relu, geluType, beta, (char *)dst, ldc,
                          wei_scale, 0, scale_size, do_sum,
                          is_weights_const, group_size, scale_dt);
        }
        else if (zenEnvObj.zenBF16GEMMalgo == zenBF16MatMulAlgoType::MATMUL_AOCL_BF16) {
            ref_woq_bf16(ctx, po_ops, src_type, weights_type, dst_type, bias_type, Layout,
                         transA, transB,
                         M, K, N, alpha, (int16_t *)src, lda, (int8_t *)weights, ldb, bias,
                         has_eltwise_relu, geluType, beta, (char *)dst, ldc,
                         wei_scale, 0, scale_size, do_sum,
                         is_weights_const, group_size, scale_dt);
        }
        else if (zenEnvObj.zenBF16GEMMalgo =
                     zenBF16MatMulAlgoType::MATMUL_JIT_BF16) {
            obj.is_brgemm = true;
            obj.is_log = false;
            zenMatMulPrimitiveIntComputeBF16(ctx, zenEnvObj, weights_type, dst_type,
                                             bias_type, Layout, transA, transB, M, N, K,
                                             (int16_t *)src, (int8_t *)weights, bias, dst, alpha, beta, lda, ldb, ldc,
                                             po_ops, false, wei_scale, 0, scale_size, is_weights_const, group_size,
                                             scale_dt);
            obj.is_brgemm = false;
            obj.is_log = true;
        }
        else {
            obj.is_brgemm = true;
            obj.is_log = false;
            zenMatMulPrimitiveIntComputeBF16(ctx, zenEnvObj, weights_type, dst_type,
                                             bias_type, Layout, transA, transB, M, N, K,
                                             (int16_t *)src, (int8_t *)weights, bias, dst, alpha, beta, lda, ldb, ldc,
                                             po_ops, true, wei_scale, 0, scale_size, is_weights_const, group_size,
                                             scale_dt);
            obj.is_brgemm = false;
            obj.is_log = true;
        }
    }
    else if (src_type == zendnn_f32) {
        ref_woq_f32(ctx, po_ops, src_type, weights_type, dst_type, bias_type, Layout,
                    transA, transB,
                    M, K, N, alpha, (float *)src, lda, (int8_t *)weights, ldb, (const float *)bias,
                    has_eltwise_relu, geluType, beta, (float *)dst, ldc,
                    wei_scale, 0, scale_size, do_sum,
                    is_weights_const, group_size, scale_dt);
    }
    zendnnVerbose(ZENDNN_PROFLOG,"zendnn_woq_matmul auto_tuner=",
                  0 ? "True": "False", " Weights=", weights_type == zendnn_s4 ? "s4": "s8",
                  " Compute=", src_type == zendnn_f32 ? "FP32": "BF16",
                  " Layout=", Layout ? "CblasRowMajor(1)" : "CblasColMajor(0)", " M=", M, " N=",N,
                  " K=", K, " transA=", transA, " transB=", transB, " lda=", lda, " ldb=", ldb,
                  " ldc=", ldc, " alpha=", alpha, " beta=", beta, " algo_type=",
                  zenEnvObj.zenBF16GEMMalgo);
    return 0;
}
