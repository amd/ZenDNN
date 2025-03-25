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

template<typename T>
aocl_post_op *create_aocl_post_ops_bf16(const impl::exec_ctx_t &ctx,
                                        const impl::post_ops_t &po,
                                        int n, const float alpha, char *bias, int bias_type,
                                        const bool relu, const int gelu, T *sum_buff,
                                        int &postop_count, const float *scale, float *dummy_scale) {
    aocl_post_op *post_ops = NULL;
    if (bias != NULL) {
        ++postop_count;
    }

    // Create postop for LPGEMM
    // Order of postops: BIAS -> RELU
    if (po.len() + postop_count > 0) {
        post_ops = (aocl_post_op *) malloc(sizeof(aocl_post_op));
        if (post_ops == NULL) {
            return NULL;
        }
        dim_t max_post_ops_seq_length = po.len() + postop_count;
        post_ops->seq_vector = (AOCL_POST_OP_TYPE *) malloc(max_post_ops_seq_length *
                               sizeof(AOCL_POST_OP_TYPE));
        if (post_ops->seq_vector == NULL) {
            free(post_ops);
            return NULL;
        }

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
            if (post_ops->bias == NULL) {
                free(post_ops->seq_vector);
                free(post_ops);
                return NULL;
            }
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
        }

        //Scale post-op
        post_ops->sum = (aocl_post_op_sum *) malloc(sizeof(
                            aocl_post_op_sum));
        if (post_ops->sum == NULL) {
            if (post_ops->bias != NULL) {
                free(post_ops->bias);
            }
            free(post_ops->seq_vector);
            free(post_ops);
            return NULL;
        }
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
            case impl::primitive_kind::sum:
                //condition gemm_applies beta for alpha = 1.0
                if (scale[0] != 1.0) {
                    mem_count[1]++;
                }
                break;
            default:
                break;
            }
        }

        if (mem_count[0] > 0) {
            post_ops->eltwise = (aocl_post_op_eltwise *) malloc(sizeof(
                                    aocl_post_op_eltwise)*mem_count[0]);
            if (post_ops->eltwise == NULL) {
                if (post_ops->sum != NULL) {
                    free(post_ops->sum);
                }
                if (post_ops->bias != NULL) {
                    free(post_ops->bias);
                }
                free(post_ops->seq_vector);
                free(post_ops);
                return NULL;
            }
        }
        if (mem_count[1] > 0) {
            post_ops->matrix_add = (aocl_post_op_matrix_add *) malloc(sizeof(
                                       aocl_post_op_matrix_add)*mem_count[1]);
            if (post_ops->matrix_add == NULL) {
                if (post_ops->eltwise != NULL) {
                    free(post_ops->eltwise);
                }
                if (post_ops->sum != NULL) {
                    free(post_ops->sum);
                }
                if (post_ops->bias != NULL) {
                    free(post_ops->bias);
                }
                free(post_ops->seq_vector);
                free(post_ops);
                return NULL;
            }
        }
        if (mem_count[2] > 0) {
            post_ops->matrix_mul = (aocl_post_op_matrix_mul *) malloc(sizeof(
                                       aocl_post_op_matrix_mul)*mem_count[2]);
            if (post_ops->matrix_mul == NULL) {
                if (post_ops->matrix_add != NULL) {
                    free(post_ops->matrix_add);
                }
                if (post_ops->eltwise != NULL) {
                    free(post_ops->eltwise);
                }
                if (post_ops->sum != NULL) {
                    free(post_ops->sum);
                }
                if (post_ops->bias != NULL) {
                    free(post_ops->bias);
                }
                free(post_ops->seq_vector);
                free(post_ops);
                return NULL;
            }

        }
        //Add eltwise and binary post-ops in given sequence
        for (auto idx = 0; idx < po.len(); ++idx) {
            const auto po_type = po.entry_[idx];
            if (po_type.kind == impl::primitive_kind::eltwise &&
                    post_ops->eltwise != NULL) {

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
                else if (po_type.eltwise.alg == impl::alg_kind::eltwise_swish) {
                    // SiLU
                    dim_t eltwise_index = 0;
                    post_ops->seq_vector[post_op_i++] = ELTWISE;
                    (post_ops->eltwise + eltwise_index)->is_power_of_2 = FALSE;
                    (post_ops->eltwise + eltwise_index)->scale_factor = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.alpha = malloc(sizeof(float));
                    *((float *)(post_ops->eltwise + eltwise_index)->algo.alpha) = (float)1;
                    (post_ops->eltwise + eltwise_index)->algo.beta = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.algo_type = SWISH;
                    eltwise_index++;
                }
                else if (po_type.eltwise.alg == impl::alg_kind::eltwise_logistic) {
                    // Sigmoid
                    dim_t eltwise_index = 0;
                    post_ops->seq_vector[post_op_i++] = ELTWISE;
                    (post_ops->eltwise + eltwise_index)->is_power_of_2 = FALSE;
                    (post_ops->eltwise + eltwise_index)->scale_factor = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.alpha = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.algo_type = SIGMOID;
                    eltwise_index++;
                }
            }
            else if (po_type.kind == impl::primitive_kind::binary) {
                if (po_type.binary.alg == impl::alg_kind::binary_add &&
                        post_ops->matrix_add != NULL) {
                    post_ops->seq_vector[post_op_i++] = MATRIX_ADD;
                    (post_ops->matrix_add + add_index)->ldm = n;

                    (post_ops->matrix_add + add_index)->scale_factor = (float *)dummy_scale;
                    (post_ops->matrix_add + add_index)->scale_factor_len = 1;
                    auto b_dt = po_type.binary.src1_desc.data_type;
                    auto binary_po = CTX_IN_MEM(const void *,
                                                (ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(idx) | ZENDNN_ARG_SRC_1));
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
                else if (po_type.binary.alg == impl::alg_kind::binary_mul &&
                         post_ops->matrix_mul != NULL) {
                    post_ops->seq_vector[post_op_i++] = MATRIX_MUL;
                    (post_ops->matrix_mul + mul_index)->ldm = n;
                    (post_ops->matrix_mul + mul_index)->scale_factor = (float *)dummy_scale;
                    (post_ops->matrix_mul + mul_index)->scale_factor_len = 1;
                    auto b_dt = po_type.binary.src1_desc.data_type;
                    auto binary_po = CTX_IN_MEM(const void *,
                                                (ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(idx) | ZENDNN_ARG_SRC_1));
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
            else if (po_type.kind == impl::primitive_kind::sum &&
                     post_ops->matrix_add != NULL) {
                if (scale[0] != 1.0) {
                    post_ops->seq_vector[post_op_i++] = MATRIX_ADD;
                    (post_ops->matrix_add + add_index)->ldm = n;
                    //Currently putting 1.0 as scale
                    (post_ops->matrix_add + add_index)->scale_factor = (float *)dummy_scale;
                    (post_ops->matrix_add + add_index)->scale_factor_len = 1;

                    (post_ops->matrix_add + add_index)->matrix = (T *)sum_buff;
                    add_index++;
                }
                else {
                    postop_count-=1; //Since sum post-op is not applied.
                }
            }
        }
        post_ops->seq_length = po.len() + postop_count;
    }
    return (post_ops);
}

void zenMatMul_gemm_bf16bf16f32of32(
    const impl::exec_ctx_t &ctx,
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
    const impl::post_ops_t &po,
    const int gelu,
    const float beta,
    float *output,
    const int ldc,
    bool is_weights_const,
    int bias_type,
    bool blocked_format
) {
    zendnnEnv zenEnvObj = readEnv();
    unsigned int thread_qty = zenEnvObj.omp_num_threads;
    unsigned int weight_cache_type = zenEnvObj.zenWeightCache;
    //TODO: Create cleaner key for weight caching map
    Key_matmul key_obj(transpose_input, transpose_filter, m, k, n, lda, ldb, ldc,
                       filter, thread_qty, false);

    // Blocked BLIS API for matmul
    // Set post_ops to NULL and define reorder_param0 as 'B' for B matrix
    // Define dimentions of B matrix as reorder_param1 and reorder_param2
    // Define memory format as 'n'(non reordered) for A matrix and 'r'(reordered) for B matrix
    char mem_format_a = 'n', mem_format_b = 'r';
    int16_t *reorder_filter = NULL;
    bool reorder_status = false;

    if (blocked_format) {
        const char reorder_param0 = 'B';
        const dim_t reorder_param1 = k;
        const dim_t reorder_param2 = n;
        const char order = 'r';
        char trans = 'n';
        if (transpose_filter) {
            trans = 't';
        }

        reorder_status = reorderAndCacheWeights<int16_t>(key_obj, filter,
                         reorder_filter, k, n,
                         ldb, is_weights_const, order, trans, reorder_param0, reorder_param1,
                         reorder_param2,
                         aocl_get_reorder_buf_size_bf16bf16f32of32, aocl_reorder_bf16bf16f32of32,
                         weight_cache_type);
        if (!reorder_status) {
            mem_format_b = 'n';
            blocked_format = false;
        }
    }
    else {
        mem_format_b = 'n';
    }

    //Create post-ops
    int postop_count = 1;
    float dummy_scale = (float)1.0;
    aocl_post_op *post_ops = create_aocl_post_ops_bf16<float>(ctx, po, n,
                             alpha, (char *) bias, bias_type, relu, gelu, output,
                             postop_count, &alpha, &dummy_scale);
    //Perform MatMul using AMD BLIS
    aocl_gemm_bf16bf16f32of32(Layout? 'r' : 'c',
                              transpose_input ? 't' : 'n',
                              transpose_filter ? 't' : 'n', m, n, k,
                              1.0,//alpha,
                              input, lda, mem_format_a,
                              blocked_format ? reorder_filter : filter, ldb,
                              mem_format_b,
                              alpha == 1.0 ? beta : 0.0,
                              output, ldc,
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
        free(post_ops->sum->scale_factor);
        free(post_ops->sum->zero_point);
        free(post_ops->sum);
        if (post_ops->matrix_add != NULL) {
            free(post_ops->matrix_add);
        }
        if (post_ops->matrix_mul != NULL) {
            free(post_ops->matrix_mul);
        }
        free(post_ops->seq_vector);
        free(post_ops);
    }
    if (!is_weights_const && blocked_format &&
            weight_cache_type <= zendnnWeightCacheType::WEIGHT_CACHE_INPLACE) {
        free(reorder_filter);
    }
}

void zenMatMul_gemm_parallel_bf16bf16f32of32(
    const impl::exec_ctx_t &ctx,
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
    const impl::post_ops_t &po,
    const int gelu,
    const float beta,
    float *output,
    const int ldc,
    bool is_weights_const,
    int bias_type

) {
    zendnnEnv zenEnvObj = readEnv();
    unsigned int thread_qty = (zenEnvObj.omp_num_threads>m)?m:
                              zenEnvObj.omp_num_threads;
    //TODO: Create cleaner key for weight caching map
    //Putting hardcoded values for now
    Key_matmul key_obj(transpose_input, transpose_filter, m, k, n, lda, ldb, ldc,
                       filter, thread_qty, false);

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

    int16_t *reorder_filter = NULL;

    reorderAndCacheWeights<int16_t>(key_obj, filter, reorder_filter, k, n,
                                    ldb, is_weights_const, order, trans, reorder_param0, reorder_param1,
                                    reorder_param2,
                                    aocl_get_reorder_buf_size_bf16bf16f32of32, aocl_reorder_bf16bf16f32of32
                                   );

    int postop_count = 1;
    float dummy_scale = (float)1.0;
    aocl_post_op *post_ops = create_aocl_post_ops_bf16<float>(ctx, po, n,
                             alpha, (char *) bias, bias_type, relu, gelu, output,
                             postop_count, &alpha, &dummy_scale);
    omp_set_max_active_levels(1);
    int16_t *data_col = (int16_t *)input;
    unsigned int m_rem_dim = m%thread_qty;
    #pragma omp parallel num_threads(thread_qty)
    {
        unsigned int id = omp_get_thread_num();
        unsigned int m_per_threads = m / thread_qty;
        if (m_rem_dim && id < m_rem_dim) {
            m_per_threads++;
        }
        int threadOffset = id * m_per_threads;
        if (m_rem_dim && id>=m_rem_dim) {
            threadOffset += m_rem_dim;
        }

        unsigned int inputOffset = ((unsigned int)lda * threadOffset);
        unsigned long outputOffset = ((unsigned long)ldc * threadOffset);

        //Perform MatMul using AMD BLIS
        //Does not support transpose-input and column-major input
        aocl_gemm_bf16bf16f32of32('r','n',
                                  transpose_filter ? 't' : 'n', m_per_threads, n, k,
                                  1.0,//alpha,
                                  data_col + inputOffset, lda, mem_format_a,
                                  reorder_filter, ldb,
                                  mem_format_b,
                                  alpha == 1.0 ? beta : 0.0,
                                  output + outputOffset, ldc, post_ops);

    }
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
        free(post_ops->sum->scale_factor);
        free(post_ops->sum->zero_point);
        free(post_ops->sum);
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
}

void zenMatMul_gemm_bf16bf16f32obf16(
    const impl::exec_ctx_t &ctx,
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
    int16_t *bias,
    const bool relu,
    const impl::post_ops_t &po,
    const int gelu,
    const float beta,
    int16_t *output,
    const int ldc,
    const float *scale,
    const int out_scale_size,
    bool is_weights_const,
    int bias_type,
    bool blocked_format
) {
    zendnnEnv zenEnvObj = readEnv();
    unsigned int thread_qty = zenEnvObj.omp_num_threads;
    unsigned int weight_cache_type = zenEnvObj.zenWeightCache;
    //TODO: Create cleaner key for weight caching map
    Key_matmul key_obj(transpose_input, transpose_filter, m, k, n, lda, ldb, ldc,
                       filter, thread_qty, false);

    // Blocked BLIS API for matmul
    // Set post_ops to NULL and define reorder_param0 as 'B' for B matrix
    // Define dimentions of B matrix as reorder_param1 and reorder_param2
    // Define memory format as 'n'(non reordered) for A matrix and 'r'(reordered) for B matrix

    bool reorder_status = false;
    char mem_format_a = 'n', mem_format_b = 'r';
    int16_t *reorder_filter = NULL;

    if (blocked_format) {
        const char reorder_param0 = 'B';
        const dim_t reorder_param1 = k;
        const dim_t reorder_param2 = n;
        const char order = 'r';
        char trans = 'n';
        if (transpose_filter) {
            trans = 't';
        }

        reorder_status = reorderAndCacheWeights<int16_t>(key_obj, filter,
                         reorder_filter, k, n,
                         ldb, is_weights_const, order, trans, reorder_param0, reorder_param1,
                         reorder_param2,
                         aocl_get_reorder_buf_size_bf16bf16f32of32, aocl_reorder_bf16bf16f32of32,
                         weight_cache_type);
        if (!reorder_status) {
            mem_format_b = 'n';
            blocked_format = false;
        }

    }
    else {
        mem_format_b = 'n';
    }
    //Post ops addition
    int postop_count= 1;
    float dummy_scale = (float)1.0;
    aocl_post_op *post_ops = create_aocl_post_ops_bf16<int16_t>(ctx, po, n,
                             alpha, (char *) bias, bias_type, relu, gelu, output,
                             postop_count, &alpha, &dummy_scale);
    //Perform MatMul using AMD BLIS
    aocl_gemm_bf16bf16f32obf16(Layout? 'r' : 'c',
                               transpose_input ? 't' : 'n',
                               transpose_filter ? 't' : 'n', m, n, k,
                               1.0,//alpha,
                               input, lda, mem_format_a,
                               blocked_format ? reorder_filter : filter, ldb,
                               mem_format_b,
                               alpha == 1.0 ? beta : 0.0,
                               output, ldc,
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
        free(post_ops->sum->scale_factor);
        free(post_ops->sum->zero_point);
        free(post_ops->sum);
        if (post_ops->matrix_add != NULL) {
            free(post_ops->matrix_add);
        }
        if (post_ops->matrix_mul != NULL) {
            free(post_ops->matrix_mul);
        }
        free(post_ops->seq_vector);
        free(post_ops);
    }
    if (!is_weights_const && blocked_format &&
            weight_cache_type <= zendnnWeightCacheType::WEIGHT_CACHE_INPLACE) {
        free(reorder_filter);
    }
}

void zenMatMul_gemm_parallel_bf16bf16f32obf16(
    const impl::exec_ctx_t &ctx,
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
    int16_t *bias,
    const bool relu,
    const impl::post_ops_t &po,
    const int gelu,
    const float beta,
    int16_t *output,
    const int ldc,
    const float *scale,
    const int out_scale_size,
    bool is_weights_const,
    int bias_type
) {
    zendnnEnv zenEnvObj = readEnv();
    unsigned int thread_qty = (zenEnvObj.omp_num_threads>m)?m:
                              zenEnvObj.omp_num_threads;
    //TODO: Create cleaner key for weight caching map
    Key_matmul key_obj(transpose_input, transpose_filter, m, k, n, lda, ldb, ldc,
                       filter, thread_qty, false);

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

    int16_t *reorder_filter = NULL;

    reorderAndCacheWeights<int16_t>(key_obj, filter, reorder_filter, k, n,
                                    ldb, is_weights_const, order, trans, reorder_param0, reorder_param1,
                                    reorder_param2,
                                    aocl_get_reorder_buf_size_bf16bf16f32of32, aocl_reorder_bf16bf16f32of32
                                   );
    //Post ops addition
    int postop_count=1;
    float dummy_scale = (float)1.0;
    aocl_post_op *post_ops = create_aocl_post_ops_bf16<int16_t>(ctx, po, n,
                             alpha, (char *) bias, bias_type, relu, gelu, output,
                             postop_count, &alpha, &dummy_scale);
    omp_set_max_active_levels(1);
    int16_t *data_col = (int16_t *)input;
    unsigned int m_rem_dim = m%thread_qty;
    #pragma omp parallel num_threads(thread_qty)
    {
        unsigned int id = omp_get_thread_num();
        unsigned int m_per_threads = m / thread_qty;
        if (m_rem_dim && id < m_rem_dim) {
            m_per_threads++;
        }
        int threadOffset = id * m_per_threads;
        if (m_rem_dim && id>=m_rem_dim) {
            threadOffset += m_rem_dim;
        }
        unsigned int inputOffset = ((unsigned int)lda * threadOffset);
        unsigned long outputOffset = ((unsigned long)ldc * threadOffset);

        //Perform MatMul using AMD BLIS
        //Does not support transpose-input and column-major input
        aocl_gemm_bf16bf16f32obf16('r','n',
                                   transpose_filter ? 't' : 'n', m_per_threads, n, k,
                                   1.0,//alpha,
                                   data_col + inputOffset, lda, mem_format_a,
                                   reorder_filter, ldb,
                                   mem_format_b,
                                   alpha == 1.0 ? beta : 0.0,
                                   output + outputOffset, ldc, post_ops);

    }

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
        free(post_ops->sum->scale_factor);
        free(post_ops->sum->zero_point);
        free(post_ops->sum);
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
}
void zenMatMulPrimitiveBF16(const impl::exec_ctx_t &ctx, zendnnEnv zenEnvObj,
                            int dst_type, int bias_type,
                            const bool Layout,
                            const bool TransA, const bool TransB, const int M,
                            const int N, const int K,
                            const zendnn::impl::bfloat16_t *A_Array,
                            const zendnn::impl::bfloat16_t *B_Array,
                            const char *bias, void *C_Array, const float alpha,
                            const float beta, const int lda, const int ldb,
                            const int ldc, const impl::post_ops_t &po_ops,
                            bool blocked_format, bool is_weights_const) {
    zendnn::engine eng(engine::kind::cpu, 0);
    zendnn::stream engine_stream(eng);

    unsigned int thread_qty = zenEnvObj.omp_num_threads;
    unsigned int weight_cache_type = zenEnvObj.zenWeightCache;
    std::unordered_map<int, memory> net_args;

    zendnn::impl::bfloat16_t *in_arr = const_cast<zendnn::impl::bfloat16_t *>
                                       (A_Array);
    zendnn::impl::bfloat16_t *filt_arr = const_cast<zendnn::impl::bfloat16_t *>
                                         (B_Array);
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
            weight_cache_type > zendnnWeightCacheType::WEIGHT_CACHE_OUT_OF_PLACE ?
            tag::BA16a64b2a : tag::any);

    // If size doesn't match with reordered_memory don't do blocking
    // Only for caching type 4 (AOT_INPLACE)
    if (matmul_weights_md.get_size() != blocked_matmul_weights_md.get_size() &&
            weight_cache_type == zendnnWeightCacheType::WEIGHT_CACHE_AOT_INPLACE) {
        blocked_format = false;
    }
    memory::desc bias_md;
    if (bias_type == zendnn_bf16) //Bias type bf16
        bias_md = memory::desc({bias_dims}, dt::bf16, tag::ab);
    else if (bias_type == zendnn_f32) //Bias type f32
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
        // Sigmoid
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
    user_weights_memory = memory(matmul_weights_md, eng, filt_arr);
    if (bias_type) {
        bias_memory = memory(bias_md, eng, bias_arr);
    }
    dst_memory = memory(dst_md, eng, C_Array);
    //Weight reordering
    zendnn::memory reordered_weights_memory;
    auto block_info = matmul_prim_disc.weights_desc().data.format_desc.blocking;
    Key_matmul key_obj(TransA, TransB, M, K, N, lda, ldb, ldc, B_Array,
                       thread_qty, false, block_info);

    if (blocked_format) {
        reorderAndCacheWeightsBrgemm(
            key_obj,
            matmul_prim_disc.weights_desc(), user_weights_memory,
            reordered_weights_memory, eng, engine_stream, is_weights_const,
            weight_cache_type);
    }

    //net.push_back(zendnn::matmul(matmul_prim_disc));
    zendnn::matmul matmul_prim = zendnn::matmul(matmul_prim_disc);
    net_args.insert({ZENDNN_ARG_SRC, src_memory});
    net_args.insert({ZENDNN_ARG_WEIGHTS, blocked_format?reordered_weights_memory:user_weights_memory});
    if (bias_type) net_args.insert({ZENDNN_ARG_BIAS,bias_memory});
    net_args.insert({ZENDNN_ARG_DST,dst_memory});
    matmul_prim.execute(engine_stream, net_args);
}

//ToDo: Add Separate BLIS Postop API for Parallel Implementation
//to Support binary postop.
int matmul_bf16_wrapper(const impl::exec_ctx_t &ctx,
                        zendnn::zendnnEnv zenEnvObj,
                        int dst_type,
                        int bias_type,
                        const bool Layout,
                        const bool transA,
                        const bool transB,
                        const int M,
                        const int K,
                        const int N,
                        const float alpha,
                        const zendnn::impl::bfloat16_t *src,
                        const int lda,
                        const zendnn::impl::bfloat16_t *weights,
                        const int ldb,
                        const char *bias,
                        const bool has_eltwise_relu,
                        const impl::post_ops_t &po_ops,
                        int has_binary_index,
                        const int geluType,
                        const float beta,
                        void *dst,
                        const int ldc,
                        const float *output_scales,
                        const int scale_size,
                        bool is_weights_const) {

    zendnnOpInfo &obj = zendnnOpInfo::ZenDNNOpInfo();

    map_mutex.lock();
    obj.is_log = true;
    map_mutex.unlock();
    if ((zenEnvObj.zenBF16GEMMalgo ==
            zenBF16MatMulAlgoType::MATMUL_BLOCKED_AOCL_BF16 ||
            zenEnvObj.zenBF16GEMMalgo ==
            zenBF16MatMulAlgoType::MATMUL_BLOCKED_AOCL_PAR_BF16)
            && (beta == 0.0 || beta == 1.0 || alpha == 1.0)
       ) {
        if (dst_type == zendnn_bf16) {
            if (has_binary_index<0 &&
                    zenEnvObj.zenBF16GEMMalgo ==
                    zenBF16MatMulAlgoType::MATMUL_BLOCKED_AOCL_PAR_BF16) {
                zendnnInfo(ZENDNN_TESTLOG,"zenEnvObj.zenBF16GEMMalgo : ",
                           zenEnvObj.zenBF16GEMMalgo);
                zenMatMul_gemm_parallel_bf16bf16f32obf16(ctx, Layout, transA, transB, M, K, N,
                        alpha,
                        (int16_t *)src, lda, (int16_t *)weights, ldb,
                        (int16_t *)bias,
                        has_eltwise_relu, po_ops, geluType, beta, (int16_t *)dst, ldc, output_scales,
                        scale_size, is_weights_const, bias_type);
            }
            else {
                zendnnInfo(ZENDNN_TESTLOG,"zenEnvObj.zenBF16GEMMalgo : ",
                           zenEnvObj.zenBF16GEMMalgo);
                zenMatMul_gemm_bf16bf16f32obf16(ctx, Layout, transA, transB, M, K, N, alpha,
                                                (int16_t *)src, lda, (int16_t *)weights, ldb,
                                                (int16_t *)bias,
                                                has_eltwise_relu, po_ops, geluType, beta, (int16_t *)dst,
                                                ldc, output_scales, scale_size, is_weights_const, bias_type, true);
            }
        }
        else if (dst_type == zendnn_f32) {
            if (has_binary_index<0 &&
                    zenEnvObj.zenBF16GEMMalgo ==
                    zenBF16MatMulAlgoType::MATMUL_BLOCKED_AOCL_PAR_BF16) {
                zendnnInfo(ZENDNN_TESTLOG,"zenEnvObj.zenBF16GEMMalgo : ",
                           zenEnvObj.zenBF16GEMMalgo);
                zenMatMul_gemm_parallel_bf16bf16f32of32(ctx, Layout, transA, transB, M, K, N,
                                                        alpha,
                                                        (int16_t *)src, lda, (int16_t *)weights, ldb, (float *)bias,
                                                        has_eltwise_relu, po_ops, geluType, beta, (float *)dst, ldc,
                                                        is_weights_const, bias_type);
            }
            else {
                zendnnInfo(ZENDNN_TESTLOG,"zenEnvObj.zenBF16GEMMalgo : ",
                           zenEnvObj.zenBF16GEMMalgo);
                zenMatMul_gemm_bf16bf16f32of32(ctx, Layout, transA, transB, M, K, N, alpha,
                                               (int16_t *)src, lda, (int16_t *)weights, ldb, (float *)bias,
                                               has_eltwise_relu, po_ops, geluType, beta, (float *)dst, ldc,
                                               is_weights_const, bias_type, true);
            }
        }
    }
    else if ((zenEnvObj.zenBF16GEMMalgo ==
              zenBF16MatMulAlgoType::MATMUL_AOCL_BF16)
             && (beta == 0.0 || beta == 1.0 || alpha == 1.0)
            ) {
        if (dst_type == zendnn_bf16) {
            zendnnInfo(ZENDNN_TESTLOG,"zenEnvObj.zenBF16GEMMalgo : ",
                       zenEnvObj.zenBF16GEMMalgo);
            zenMatMul_gemm_bf16bf16f32obf16(ctx, Layout, transA, transB, M, K, N, alpha,
                                            (int16_t *)src, lda, (int16_t *)weights, ldb,
                                            (int16_t *)bias,
                                            has_eltwise_relu, po_ops, geluType, beta, (int16_t *)dst,
                                            ldc, output_scales, scale_size, is_weights_const, bias_type, false);
        }
        else if (dst_type == zendnn_f32) {
            zendnnInfo(ZENDNN_TESTLOG,"zenEnvObj.zenBF16GEMMalgo : ",
                       zenEnvObj.zenBF16GEMMalgo);
            zenMatMul_gemm_bf16bf16f32of32(ctx, Layout, transA, transB, M, K, N, alpha,
                                           (int16_t *)src, lda, (int16_t *)weights, ldb, (float *)bias,
                                           has_eltwise_relu, po_ops, geluType, beta, (float *)dst, ldc,
                                           is_weights_const, bias_type, true);
        }
    }
    else if (zenEnvObj.zenBF16GEMMalgo ==
             zenBF16MatMulAlgoType::MATMUL_BLOCKED_JIT_BF16
             || zenEnvObj.zenBF16GEMMalgo ==
             zenBF16MatMulAlgoType::MATMUL_BLOCKED_JIT_PAR_BF16) {
        //CALL blocked BRGEMM Primitive
        map_mutex.lock();
        obj.is_brgemm = true;
        obj.is_log = false;
        map_mutex.unlock();
        if (has_binary_index<0 && zenEnvObj.zenBF16GEMMalgo ==
                zenBF16MatMulAlgoType::MATMUL_BLOCKED_JIT_PAR_BF16) {
            zendnnInfo(ZENDNN_TESTLOG,"zenEnvObj.zenBF16GEMMalgo : ",
                       zenEnvObj.zenBF16GEMMalgo);
            unsigned int thread_qty = (zenEnvObj.omp_num_threads>M)?M:
                                      zenEnvObj.omp_num_threads;
            omp_set_max_active_levels(1);
            zendnn::impl::bfloat16_t *data_col = (zendnn::impl::bfloat16_t *)src;
            unsigned int m_rem_dim = M % thread_qty;
            #pragma omp parallel num_threads(thread_qty)
            {
                unsigned int id = omp_get_thread_num();
                unsigned int m_per_threads = M / thread_qty;
                if (m_rem_dim && id < m_rem_dim) {
                    m_per_threads++;
                }
                int threadOffset = id * m_per_threads;
                if (m_rem_dim && id>=m_rem_dim) {
                    threadOffset += m_rem_dim;
                }
                unsigned int inputOffset = ((unsigned int)lda * threadOffset);
                unsigned long outputOffset = ((unsigned long)ldc * threadOffset);
                if (dst_type == zendnn_bf16) {
                    zenMatMulPrimitiveBF16(ctx, zenEnvObj, dst_type, bias_type, Layout, transA,
                                           transB,
                                           m_per_threads, N, K,data_col + inputOffset, weights, bias,
                                           (int16_t *)dst + outputOffset, alpha, beta,
                                           lda, ldb, ldc, po_ops, true, is_weights_const);
                }
                else {
                    zenMatMulPrimitiveBF16(ctx, zenEnvObj, dst_type, bias_type, Layout, transA,
                                           transB,
                                           m_per_threads, N, K,data_col + inputOffset, weights, bias,
                                           (float *)dst + outputOffset, alpha, beta,
                                           lda, ldb, ldc, po_ops, true, is_weights_const);
                }
            }
        }
        else {
            zendnnInfo(ZENDNN_TESTLOG,"zenEnvObj.zenBF16GEMMalgo : ",
                       zenEnvObj.zenBF16GEMMalgo);
            zenMatMulPrimitiveBF16(ctx, zenEnvObj, dst_type, bias_type, Layout, transA,
                                   transB,
                                   M, N, K,
                                   src, weights, bias, dst, alpha, beta, lda, ldb, ldc,
                                   po_ops, true, is_weights_const);
        }
        map_mutex.lock();
        obj.is_log = true;
        obj.is_brgemm = false;
        map_mutex.unlock();
    }
    else {
        //CALL BRGEMM Primitive
        map_mutex.lock();
        obj.is_brgemm = true;
        obj.is_log = false;
        map_mutex.unlock();
        if (has_binary_index<0 &&
                zenEnvObj.zenBF16GEMMalgo == zenBF16MatMulAlgoType::MATMUL_JIT_PAR_BF16) {
            zendnnInfo(ZENDNN_TESTLOG,"zenEnvObj.zenBF16GEMMalgo : ",
                       zenEnvObj.zenBF16GEMMalgo);
            unsigned int thread_qty = (zenEnvObj.omp_num_threads>M)?M:
                                      zenEnvObj.omp_num_threads;
            omp_set_max_active_levels(1);
            zendnn::impl::bfloat16_t *data_col = (zendnn::impl::bfloat16_t *)src;
            unsigned int m_rem_dim = M % thread_qty;
            #pragma omp parallel num_threads(thread_qty)
            {
                unsigned int id = omp_get_thread_num();
                unsigned int m_per_threads = M / thread_qty;
                if (m_rem_dim && id < m_rem_dim) {
                    m_per_threads++;
                }
                int threadOffset = id * m_per_threads;
                if (m_rem_dim && id>=m_rem_dim) {
                    threadOffset += m_rem_dim;
                }
                unsigned int inputOffset = ((unsigned int)lda * threadOffset);
                unsigned long outputOffset = ((unsigned long)ldc * threadOffset);
                if (dst_type == zendnn_bf16) {
                    zenMatMulPrimitiveBF16(ctx, zenEnvObj, dst_type, bias_type, Layout, transA,
                                           transB,
                                           m_per_threads, N, K, data_col + inputOffset, weights, bias,
                                           (int16_t *)dst + outputOffset, alpha, beta, lda, ldb, ldc,
                                           po_ops, false, is_weights_const);
                }
                else {
                    zenMatMulPrimitiveBF16(ctx, zenEnvObj, dst_type, bias_type, Layout, transA,
                                           transB,
                                           m_per_threads, N, K, data_col + inputOffset, weights, bias,
                                           (float *)dst + outputOffset, alpha, beta, lda, ldb, ldc,
                                           po_ops, false, is_weights_const);
                }
            }
        }
        else {
            zendnnInfo(ZENDNN_TESTLOG,"zenEnvObj.zenBF16GEMMalgo : ",
                       zenEnvObj.zenBF16GEMMalgo);
            zenMatMulPrimitiveBF16(ctx, zenEnvObj, dst_type, bias_type, Layout, transA,
                                   transB,
                                   M, N, K,
                                   src, weights, bias, dst, alpha, beta, lda, ldb, ldc,
                                   po_ops, false, is_weights_const);
        }
        map_mutex.lock();
        obj.is_log = true;
        obj.is_brgemm = false;
        map_mutex.unlock();
    }
    return zenEnvObj.zenBF16GEMMalgo;
}