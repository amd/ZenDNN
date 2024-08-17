/*******************************************************************************
* Modifications Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#define NUM_BF16_ALGO 3
#define MATMUL_SKIP_ITER_BF16 10
#define MATMUL_EVALUATE_ITER_BF16 10

using namespace zendnn;
using namespace zendnn::impl::cpu;
using tag = memory::format_tag;
using dt = memory::data_type;
extern int graph_exe_count;
extern std::mutex map_mutex;
//Map for weight caching of jit Primitive(reordered memory)
std::unordered_map<Key_matmul, zendnn::memory >
matmul_weight_caching_map_jit;

//AOCL weight caching map
std::unordered_map<Key_matmul, int16_t * >
matmul_weight_caching_map_aocl;

//AutoTuner Simplified Map having Key as struct and value as Algo.
std::unordered_map<Key_matmul, unsigned int>
matmul_kernel_map_bf16;

//AutoTuner Helper map
std::unordered_map<Key_matmul,std::tuple<unsigned int, float, unsigned int>>
        matmul_kernel_map_bf16_helper;

static inline float bf16_to_float(int16_t bf16_val) {
    int32_t inter_temp = *((int16_t *) &bf16_val);
    inter_temp = inter_temp << 16;
    float float_value = 0.0;
    memcpy(&float_value, &inter_temp, sizeof(int32_t));
    return float_value;
}

static inline void float_to_bf16(float *float_value, bfloat16 *bf16_val) {
    /*Set offset 2 to copy most significant 2 bytes of float
    to convert float values to bf16 values*/
    memcpy((bf16_val), (char *)(float_value) + 2, sizeof(int16_t));
}

template<typename T>
aocl_post_op *create_aocl_post_ops(const impl::exec_ctx_t &ctx,
                                   const impl::post_ops_t &po,
                                   int n, const float alpha, T *bias,
                                   const bool relu, const int gelu, T *sum_buff,
                                   int &postop_count, const float *scale=NULL) {
    aocl_post_op *post_ops = NULL;
    if (bias != NULL) {
        ++postop_count;
    }
#ifdef ZENDNN_ENABLE_LPGEMM_V5_0

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
            (post_ops->bias + bias_index)->bias = (T *)bias;
            bias_index++;
        }

        //Scale post-op
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
                else if (po_type.eltwise.alg == impl::alg_kind::eltwise_swish) {
                    // SiLU
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
                    auto addA = reinterpret_cast<T *>(const_cast<void *>(binary_po));
                    (post_ops->matrix_add + add_index)->matrix = (T *)addA;
                    add_index++;
                }
                else if (po_type.binary.alg == impl::alg_kind::binary_mul) {
                    post_ops->seq_vector[post_op_i++] = MATRIX_MUL;
                    (post_ops->matrix_mul + mul_index)->ldm = n;
                    auto binary_po = CTX_IN_MEM(const void *,
                                                (ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(idx) | ZENDNN_ARG_SRC_1));
                    auto mulA = reinterpret_cast<T *>(const_cast<void *>(binary_po));
                    (post_ops->matrix_mul + mul_index)->matrix = (T *)mulA;
                    mul_index++;
                }
            }
            else if (po_type.kind == impl::primitive_kind::sum) {
                if (scale[0] != 1.0) {
                    post_ops->seq_vector[post_op_i++] = MATRIX_ADD;
                    (post_ops->matrix_add + add_index)->ldm = n;

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
#else
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
        //Set all post-ops to NULL
        post_ops->eltwise = NULL;
        dim_t eltwise_index = 0;
        if (bias != NULL) {
            // Add bias postop
            post_ops->seq_vector[post_op_i++] = BIAS;
            // Multiplying aplha with bias in the Blis path to make the computation same as JIT.
            // TODO (zendnn) : Handle this with the scale from Blis and make alpha as dummy.
            float *bias_fp = NULL;
            int16_t *bias_bf = NULL;
            //creating float memory for bf16 bias
            if (alpha != 1.0) {
                if (typeid(T) == typeid(int16_t)) {
                    bias_bf = new int16_t[n]();
                    //conversion fucntion from bf16 to f32
                    #pragma omp parallel for
                    for (size_t i = 0; i < n; i++) {
                        float inter_val = alpha*bf16_to_float(bias[i]);
                        float_to_bf16(&inter_val, bias_bf + i);
                    }
                }
                else {
                    bias_fp = new float[n]();
                    #pragma omp parallel for
                    for (int i = 0; i < n; i++) {
                        bias_fp[i] = alpha * bias[i];
                    }
                }
            }
            post_ops->bias.bias = (alpha!=1.0f) ? (typeid(T) == typeid(float)) ?
                                  (T *)bias_fp : (T *)bias_bf : (T *)bias;
        }
        if (relu) {
            // Add ReLU postop
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

        // Add scale postop
        if (scale) {
            post_ops->seq_vector[post_op_i++] = SCALE;
            post_ops->sum.is_power_of_2 = FALSE;
            post_ops->sum.scale_factor = NULL;
            post_ops->sum.buff = NULL;
            post_ops->sum.zero_point = NULL;

            post_ops->sum.scale_factor = malloc(sizeof(float));
            post_ops->sum.zero_point = malloc(sizeof(int16_t));

            //SCALE
            float *temp_dscale_ptr = (float *)post_ops->sum.scale_factor;
            int16_t *temp_dzero_point_ptr = (int16_t *)post_ops->sum.zero_point;
            temp_dscale_ptr[0] = (float)scale[0];
            temp_dzero_point_ptr[0] = 0;
        }
        post_ops->seq_length = postop_count;
    }
#endif
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
    bool is_weights_const
) {
    zendnnEnv zenEnvObj = readEnv();
    unsigned int thread_qty = zenEnvObj.omp_num_threads;
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
    key_obj.thread_count = thread_qty;

    //finds object in map
    auto found_obj = matmul_weight_caching_map_aocl.find(key_obj);
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

    //Weight caching
    if (!is_weights_const ||
            found_obj == matmul_weight_caching_map_aocl.end()) {
        siz_t b_reorder_buf_siz_req = aocl_get_reorder_buf_size_bf16bf16f32of32(
                                          order, trans, reorder_param0, reorder_param1, reorder_param2);
        reorder_filter = (int16_t *) aligned_alloc(64,
                         b_reorder_buf_siz_req);
        aocl_reorder_bf16bf16f32of32(order, trans, 'B', filter, reorder_filter, k,
                                     n, ldb);
        if (is_weights_const) {
            //Create new entry
            map_mutex.lock();
            matmul_weight_caching_map_aocl[key_obj] = reorder_filter;
            map_mutex.unlock();
        }
    }
    else {
        reorder_filter = matmul_weight_caching_map_aocl[key_obj];
    }
    //Create post-ops
    int postop_count = 1;
    aocl_post_op *post_ops = create_aocl_post_ops<float>(ctx, po, n,
                             alpha, bias, relu, gelu, output,
                             postop_count, &alpha);

    //Perform MatMul using AMD BLIS
    aocl_gemm_bf16bf16f32of32(Layout? 'r' : 'c',
                              transpose_input ? 't' : 'n',
                              transpose_filter ? 't' : 'n', m, n, k,
#ifdef ZENDNN_ENABLE_LPGEMM_V5_0
                              1.0,//alpha,
#else
                              alpha,
#endif
                              input, lda, mem_format_a,
                              reorder_filter, ldb,
                              mem_format_b,
#ifdef ZENDNN_ENABLE_LPGEMM_V5_0
                              alpha == 1.0 ? beta : 0.0,
#else
                              beta,
#endif
                              output, ldc,
                              post_ops);

    // Free memory for postops.
    if (bias != NULL) {
#ifdef ZENDNN_ENABLE_LPGEMM_V5_0
        post_ops->bias->bias=NULL;
        free(post_ops->bias);
#else
        if (alpha != 1.0f) {
            delete (post_ops->bias.bias);
        }
        post_ops->bias.bias=NULL;
#endif
    }
    if (post_ops->eltwise != NULL) {
        if (post_ops->eltwise->algo.alpha != NULL) {
            free(post_ops->eltwise->algo.alpha);
        }
        free(post_ops->eltwise);
    }
#ifdef ZENDNN_ENABLE_LPGEMM_V5_0
    free(post_ops->sum->scale_factor);
    free(post_ops->sum->zero_point);
    free(post_ops->sum);
    if (post_ops->matrix_add != NULL) {
        post_ops->matrix_add = NULL;
    }
    if (post_ops->matrix_mul != NULL) {
        post_ops->matrix_mul = NULL;
    }
#endif
    free(post_ops->seq_vector);
    free(post_ops);
    if (!is_weights_const) {
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
    bool is_weights_const
) {
    zendnnEnv zenEnvObj = readEnv();
    unsigned int thread_qty = (zenEnvObj.omp_num_threads>m)?m:
                              zenEnvObj.omp_num_threads;
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
    key_obj.thread_count = thread_qty;

    //finds object in map
    auto found_obj = matmul_weight_caching_map_aocl.find(key_obj);
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

    //Weight caching
    if (!is_weights_const ||
            found_obj == matmul_weight_caching_map_aocl.end()) {
        siz_t b_reorder_buf_siz_req = aocl_get_reorder_buf_size_bf16bf16f32of32(
                                          order, trans, reorder_param0, reorder_param1, reorder_param2);
        reorder_filter = (int16_t *) aligned_alloc(64,
                         b_reorder_buf_siz_req);
        aocl_reorder_bf16bf16f32of32(order, trans, 'B', filter, reorder_filter, k,
                                     n, ldb);
        if (is_weights_const) {
            //Create new entry
            map_mutex.lock();
            matmul_weight_caching_map_aocl[key_obj] = reorder_filter;
            map_mutex.unlock();
        }
    }
    else {
        reorder_filter = matmul_weight_caching_map_aocl[key_obj];
    }
    int postop_count = 1;
    aocl_post_op *post_ops = create_aocl_post_ops<float>(ctx, po, n,
                             alpha, bias, relu, gelu, output,
                             postop_count, &alpha);

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
#ifdef ZENDNN_ENABLE_LPGEMM_V5_0
                                  1.0,//alpha,
#else
                                  alpha,
#endif
                                  data_col + inputOffset, lda, mem_format_a,
                                  reorder_filter, ldb,
                                  mem_format_b,
#ifdef ZENDNN_ENABLE_LPGEMM_V5_0
                                  alpha == 1.0 ? beta : 0.0,
#else
                                  beta,
#endif
                                  output + outputOffset, ldc, post_ops);

    }
    // Free memory for postops.
    if (bias != NULL) {
#ifdef ZENDNN_ENABLE_LPGEMM_V5_0
        post_ops->bias->bias=NULL;
        free(post_ops->bias);
#else
        if (alpha != 1.0f) {
            delete (post_ops->bias.bias);
        }
        post_ops->bias.bias=NULL;
#endif
    }
    if (post_ops->eltwise != NULL) {
        if (post_ops->eltwise->algo.alpha != NULL) {
            free(post_ops->eltwise->algo.alpha);
        }
        free(post_ops->eltwise);
    }
#ifdef ZENDNN_ENABLE_LPGEMM_V5_0
    free(post_ops->sum->scale_factor);
    free(post_ops->sum->zero_point);
    free(post_ops->sum);
    if (post_ops->matrix_add != NULL) {
        post_ops->matrix_add = NULL;
    }
    if (post_ops->matrix_mul != NULL) {
        post_ops->matrix_mul = NULL;
    }
#endif
    free(post_ops->seq_vector);
    free(post_ops);
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
    bool is_weights_const
) {
    zendnnEnv zenEnvObj = readEnv();
    unsigned int thread_qty = zenEnvObj.omp_num_threads;
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
    key_obj.thread_count = thread_qty;

    //finds object in map
    auto found_obj = matmul_weight_caching_map_aocl.find(key_obj);
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
    //Weight caching
    if (!is_weights_const ||
            found_obj == matmul_weight_caching_map_aocl.end()) {
        siz_t b_reorder_buf_siz_req = aocl_get_reorder_buf_size_bf16bf16f32of32(
                                          order, trans, reorder_param0, reorder_param1, reorder_param2);
        reorder_filter = (int16_t *) aligned_alloc(64,
                         b_reorder_buf_siz_req);
        aocl_reorder_bf16bf16f32of32(order, trans, 'B',filter, reorder_filter, k,
                                     n, ldb);
        if (is_weights_const) {
            //Create new entry
            map_mutex.lock();
            matmul_weight_caching_map_aocl[key_obj] = reorder_filter;
            map_mutex.unlock();
        }
    }
    else {
        reorder_filter = matmul_weight_caching_map_aocl[key_obj];
    }
    //Post ops addition
    int postop_count= 1;
    aocl_post_op *post_ops = create_aocl_post_ops<int16_t>(ctx, po, n,
                             alpha, bias, relu, gelu, output,
                             postop_count, &alpha);

    //Perform MatMul using AMD BLIS
    aocl_gemm_bf16bf16f32obf16(Layout? 'r' : 'c',
                               transpose_input ? 't' : 'n',
                               transpose_filter ? 't' : 'n', m, n, k,
#ifdef ZENDNN_ENABLE_LPGEMM_V5_0
                               1.0,//alpha,
#else
                               alpha,
#endif
                               input, lda, mem_format_a,
                               reorder_filter, ldb,
                               mem_format_b,
#ifdef ZENDNN_ENABLE_LPGEMM_V5_0
                               alpha == 1.0 ? beta : 0.0,
#else
                               beta,
#endif
                               output, ldc,
                               post_ops);

    // Free memory for postops.
    if (bias != NULL) {
#ifdef ZENDNN_ENABLE_LPGEMM_V5_0
        post_ops->bias->bias=NULL;
        free(post_ops->bias);
#else
        if (alpha != 1.0f) {
            delete (post_ops->bias.bias);
        }
        post_ops->bias.bias=NULL;
#endif
    }
    if (post_ops->eltwise != NULL) {
        if (post_ops->eltwise->algo.alpha != NULL) {
            free(post_ops->eltwise->algo.alpha);
        }
        free(post_ops->eltwise);
    }
#ifdef ZENDNN_ENABLE_LPGEMM_V5_0
    free(post_ops->sum->scale_factor);
    free(post_ops->sum->zero_point);
    free(post_ops->sum);
    if (post_ops->matrix_add != NULL) {
        post_ops->matrix_add = NULL;
    }
    if (post_ops->matrix_mul != NULL) {
        post_ops->matrix_mul = NULL;
    }
#endif
    free(post_ops->seq_vector);
    free(post_ops);
    if (!is_weights_const) {
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
    bool is_weights_const
) {
    zendnnEnv zenEnvObj = readEnv();
    unsigned int thread_qty = (zenEnvObj.omp_num_threads>m)?m:
                              zenEnvObj.omp_num_threads;
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
    key_obj.thread_count = thread_qty;

    //finds object in map
    auto found_obj = matmul_weight_caching_map_aocl.find(key_obj);
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
    //Weight caching
    if (!is_weights_const ||
            found_obj == matmul_weight_caching_map_aocl.end()) {
        siz_t b_reorder_buf_siz_req = aocl_get_reorder_buf_size_bf16bf16f32of32(
                                          order, trans, reorder_param0, reorder_param1, reorder_param2);
        reorder_filter = (int16_t *) aligned_alloc(64,
                         b_reorder_buf_siz_req);
        aocl_reorder_bf16bf16f32of32(order, trans, 'B',filter, reorder_filter, k,
                                     n, ldb);
        if (is_weights_const) {
            //Create new entry
            map_mutex.lock();
            matmul_weight_caching_map_aocl[key_obj] = reorder_filter;
            map_mutex.unlock();
        }
    }
    else {
        reorder_filter = matmul_weight_caching_map_aocl[key_obj];
    }

    //Post ops addition
    int postop_count=1;
    aocl_post_op *post_ops = create_aocl_post_ops<int16_t>(ctx, po, n,
                             alpha, bias, relu, gelu, output,
                             postop_count, &alpha);

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
#ifdef ZENDNN_ENABLE_LPGEMM_V5_0
                                   1.0,//alpha,
#else
                                   alpha,
#endif
                                   data_col + inputOffset, lda, mem_format_a,
                                   reorder_filter, ldb,
                                   mem_format_b,
#ifdef ZENDNN_ENABLE_LPGEMM_V5_0
                                   alpha == 1.0 ? beta : 0.0,
#else
                                   beta,
#endif
                                   output + outputOffset, ldc, post_ops);

    }
    // Free memory for postops.
    if (bias != NULL) {
#ifdef ZENDNN_ENABLE_LPGEMM_V5_0
        post_ops->bias->bias=NULL;
        free(post_ops->bias);
#else
        if (alpha != 1.0f) {
            delete (post_ops->bias.bias);
        }
        post_ops->bias.bias=NULL;
#endif
    }
    if (post_ops->eltwise != NULL) {
        if (post_ops->eltwise->algo.alpha != NULL) {
            free(post_ops->eltwise->algo.alpha);
        }
        free(post_ops->eltwise);
    }
#ifdef ZENDNN_ENABLE_LPGEMM_V5_0
    free(post_ops->sum->scale_factor);
    free(post_ops->sum->zero_point);
    free(post_ops->sum);
    if (post_ops->matrix_add != NULL) {
        post_ops->matrix_add=NULL;
    }
    if (post_ops->matrix_mul != NULL) {
        post_ops->matrix_mul=NULL;
    }
#endif
    free(post_ops->seq_vector);
    free(post_ops);
    if (!is_weights_const) {
        free(reorder_filter);
    }
}

namespace zendnn {
namespace impl {
namespace cpu {
namespace matmul {

using namespace data_type;
void zenMatMulPrimitiveBF16(const exec_ctx_t &ctx, zendnnEnv zenEnvObj,
                            int dst_type, int bias_type,
                            const bool Layout,
                            const bool TransA, const bool TransB, const int M,
                            const int N, const int K,
                            const zendnn::impl::bfloat16_t *A_Array,
                            const zendnn::impl::bfloat16_t *B_Array,
                            const char *bias, void *C_Array, const float alpha,
                            const float beta, const int lda, const int ldb,
                            const int ldc, const post_ops_t &po_ops,
                            bool blocked_format, bool is_weights_const) {
    zendnn::engine eng(engine::kind::cpu, 0);
    zendnn::stream engine_stream(eng);

    Key_matmul key_obj_reorder;
    key_obj_reorder.transpose_input = TransA;
    key_obj_reorder.transpose_weights = TransB;
    key_obj_reorder.m = M;
    key_obj_reorder.k = K;
    key_obj_reorder.n = N;
    key_obj_reorder.lda = lda;
    key_obj_reorder.ldb = ldb;
    key_obj_reorder.ldc = ldc;
    key_obj_reorder.weights = B_Array;
    key_obj_reorder.thread_count = zenEnvObj.omp_num_threads;

    //finds object in map
    auto found_obj_reorder = matmul_weight_caching_map_jit.find(key_obj_reorder);

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
            tag::any);

    memory::desc bias_md;
    //Bias type bf16 or f32
    if (bias_type == 2)
        bias_md = memory::desc({bias_dims}, dt::bf16, tag::ab);
    else if (bias_type == 3)
        bias_md = memory::desc({bias_dims}, dt::f32, tag::ab);

    memory::desc dst_md = memory::desc({dst_dims}, dst_type == f32 ? dt::f32 :
                                       dt::bf16, {ldc, 1});
    primitive_attr matmul_attr;
    zendnn::post_ops post_ops;
    bool post_attr = false;
    const float scale = 1.0f;
    zendnn::memory po_memory[po_ops.len()];
    for (auto idx = 0; idx < po_ops.len(); ++idx) {
        const auto &e = po_ops.entry_[idx];
        if (e.eltwise.alg == alg_kind::eltwise_relu) {
            post_attr = true;
            post_ops.append_eltwise(scale, algorithm::eltwise_relu, 0.f, 0.f);
            po_memory[idx] = memory({{M,N},dt::f32,tag::ab},eng,nullptr);
        }
        else if (e.eltwise.alg == alg_kind::eltwise_swish) {
            post_attr = true;
            post_ops.append_eltwise(scale, algorithm::eltwise_swish, 1.f, 0.f);
            po_memory[idx] = memory({{M,N},dt::f32,tag::ab},eng,nullptr);
        }
        else if (e.eltwise.alg == alg_kind::eltwise_gelu) {
            post_attr = true;
            post_ops.append_eltwise(scale, algorithm::eltwise_gelu, 1.f, 0.f);
            po_memory[idx] = memory({{M,N},dt::f32,tag::ab},eng,nullptr);
        }
        else if (e.eltwise.alg == alg_kind::eltwise_gelu_erf) {
            post_attr = true;
            post_ops.append_eltwise(scale, algorithm::eltwise_gelu_erf, 1.f, 0.f);
            po_memory[idx] = memory({{M,N},dt::f32,tag::ab},eng,nullptr);
        }
        else if (e.kind == primitive_kind::sum) {
            post_attr = true;
            if (beta != 0.f) {
                post_ops.append_sum(beta);
            }
            po_memory[idx] = memory({{M,N},dt::f32,tag::ab},eng,nullptr);
        }
        else if (e.binary.alg == alg_kind::binary_add) {
            post_attr = true;
            const auto &src1_desc = e.binary.src1_desc;
            post_ops.append_binary(algorithm::binary_add, src1_desc);
            auto add_raw = const_cast<void *>(CTX_IN_MEM(const void *,
                                              (ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(idx) | ZENDNN_ARG_SRC_1)));
            po_memory[idx] = memory(src1_desc,eng,add_raw);
            int t = ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(idx) | ZENDNN_ARG_SRC_1;
            net_args.insert({t,po_memory[idx]});
        }
        else if (e.binary.alg == alg_kind::binary_mul) {
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
    if (blocked_format) {
        if (!is_weights_const ||
                found_obj_reorder == matmul_weight_caching_map_jit.end()) {
            reordered_weights_memory = memory(matmul_prim_disc.weights_desc(), eng);
            reorder(user_weights_memory, reordered_weights_memory).execute(engine_stream,
                    user_weights_memory, reordered_weights_memory);
            if (is_weights_const) {
                //Save in map
                map_mutex.lock();
                matmul_weight_caching_map_jit[key_obj_reorder] = reordered_weights_memory;
                map_mutex.unlock();
            }
        }
        else {
            reordered_weights_memory = matmul_weight_caching_map_jit[key_obj_reorder];
        }
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
int matmul_bf16_wrapper(const exec_ctx_t &ctx,
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
                        const post_ops_t &po_ops,
                        int has_binary_index,
                        const int geluType,
                        const float beta,
                        void *dst,
                        const int ldc,
                        const float *output_scales,
                        const int scale_size,
                        bool is_weights_const) {

    zendnnOpInfo &obj = zendnnOpInfo::ZenDNNOpInfo();
    obj.is_log = true;
    float *bias_f32 = NULL;
    if ((zenEnvObj.zenBF16GEMMalgo == zenBF16MatMulAlgoType::MATMUL_AOCL_GEMM ||
            zenEnvObj.zenBF16GEMMalgo == zenBF16MatMulAlgoType::MATMUL_AOCL_GEMM_PAR)
#ifdef ZENDNN_ENABLE_LPGEMM_V5_0
            && (beta == 0.0 || beta == 1.0 || alpha == 1.0)
#else
            && ((beta == 0.0 || (beta != 0.0 && alpha == 1.0)) && has_binary_index<0
                && po_ops.find(primitive_kind::sum)<0)
#endif
       ) {
        if (dst_type == bf16) {
            if (has_binary_index<0 &&
                    zenEnvObj.zenBF16GEMMalgo == zenBF16MatMulAlgoType::MATMUL_AOCL_GEMM_PAR) {
                zendnnInfo(ZENDNN_TESTLOG,"zenEnvObj.zenBF16GEMMalgo : ",
                           zenEnvObj.zenBF16GEMMalgo);
                zenMatMul_gemm_parallel_bf16bf16f32obf16(ctx, Layout, transA, transB, M, K, N,
                        alpha,
                        (int16_t *)src, lda, (int16_t *)weights, ldb,
                        (int16_t *)bias,
                        has_eltwise_relu, po_ops, geluType, beta, (int16_t *)dst, ldc, output_scales,
                        scale_size, is_weights_const);
            }
            else {
                zendnnInfo(ZENDNN_TESTLOG,"zenEnvObj.zenBF16GEMMalgo : ",
                           zenEnvObj.zenBF16GEMMalgo);
                zenMatMul_gemm_bf16bf16f32obf16(ctx, Layout, transA, transB, M, K, N, alpha,
                                                (int16_t *)src, lda, (int16_t *)weights, ldb,
                                                (int16_t *)bias,
                                                has_eltwise_relu, po_ops, geluType, beta, (int16_t *)dst,
                                                ldc, output_scales, scale_size, is_weights_const);
            }
        }
        else if (dst_type == f32) {
            if (has_binary_index<0 &&
                    zenEnvObj.zenBF16GEMMalgo == zenBF16MatMulAlgoType::MATMUL_AOCL_GEMM_PAR) {
                zendnnInfo(ZENDNN_TESTLOG,"zenEnvObj.zenBF16GEMMalgo : ",
                           zenEnvObj.zenBF16GEMMalgo);
                zenMatMul_gemm_parallel_bf16bf16f32of32(ctx, Layout, transA, transB, M, K, N,
                                                        alpha,
                                                        (int16_t *)src, lda, (int16_t *)weights, ldb,
                                                        bias_type == data_type::bf16 ? (float *)bias_f32 : (float *)bias,
                                                        has_eltwise_relu, po_ops, geluType, beta, (float *)dst, ldc,
                                                        is_weights_const);
            }
            else {
                zendnnInfo(ZENDNN_TESTLOG,"zenEnvObj.zenBF16GEMMalgo : ",
                           zenEnvObj.zenBF16GEMMalgo);
                zenMatMul_gemm_bf16bf16f32of32(ctx, Layout, transA, transB, M, K, N, alpha,
                                               (int16_t *)src, lda, (int16_t *)weights, ldb,
                                               bias_type == data_type::bf16 ? (float *)bias_f32 : (float *)bias,
                                               has_eltwise_relu, po_ops, geluType, beta, (float *)dst, ldc,
                                               is_weights_const);
            }
        }
        // Free memory if bias memory is allocated
        if (bias_type == data_type::bf16 && bias!=NULL) {
            free(bias_f32);
        }
    }
    else if (zenEnvObj.zenBF16GEMMalgo == zenBF16MatMulAlgoType::MATMUL_BLOCKED_JIT
             || zenEnvObj.zenBF16GEMMalgo == zenBF16MatMulAlgoType::MATMUL_BLOCKED_JIT_PAR) {
        //CALL blocked BRGEMM Primitive
        obj.is_brgemm = true;
        obj.is_log = false;
        if (has_binary_index<0 && zenEnvObj.zenBF16GEMMalgo ==
                zenBF16MatMulAlgoType::MATMUL_BLOCKED_JIT_PAR) {
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
                if (dst_type == bf16) {
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
        obj.is_log = true;
        obj.is_brgemm = false;
    }
    else {
        //CALL BRGEMM Primitive
        obj.is_log = false;
        obj.is_brgemm = true;
        if (has_binary_index<0 &&
                zenEnvObj.zenBF16GEMMalgo == zenBF16MatMulAlgoType::MATMUL_JIT_PAR) {
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
                if (dst_type == bf16) {
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
        obj.is_log = true;
        obj.is_brgemm = false;
    }
    return 0;
}

/*AutoTuner
  Makes use of eval_count to decide when to fetch value from map.
  Uses iteration count of each unique layer.
  Doesn't need framework to increment the graph_exe_count.

  Map value tuple
  <
    iteration_count,
    time,
    algo
  >
*/
int auto_compute_matmul_bf16(
    const exec_ctx_t &ctx,
    zendnn::zendnnEnv zenEnvObj,
    int dst_type,
    int bias_type,
    const bool Layout,
    const bool transpose_input,
    const bool transpose_filter,
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
    const post_ops_t &po_ops,
    int has_binary_index,
    const int geluType,
    const float beta,
    void *dst,
    const int ldc,
    const float *output_scales,
    const int scale_size,
    bool is_weights_const
) {

    Key_matmul key_obj_auto;

    key_obj_auto.transpose_input = transpose_input;
    key_obj_auto.transpose_weights = transpose_filter;
    key_obj_auto.m = M;
    key_obj_auto.k = K;
    key_obj_auto.n = N;
    key_obj_auto.lda = lda;
    key_obj_auto.ldb = ldb;
    key_obj_auto.ldc = ldc;

    //This condition makes sure that address
    //doesn't gets saved while using persistent map.
    unsigned int map_type =
        zendnn::zendnn_getenv_int("ZENDNN_GEMM_MAP_TYPE",0);
    key_obj_auto.weights =
        map_type == 1 ? weights : NULL;
    key_obj_auto.thread_count = zenEnvObj.omp_num_threads;

    float cur_algo_time; //current algorithm's execution time
    struct timeval start_n, end_n;

    //Number of iterations to run without creating map for each unique layer.
    unsigned int skip_iteration =
        zendnn::zendnn_getenv_int("ZENDNN_MATMUL_SKIP_ITER_BF16",
                                  MATMUL_SKIP_ITER_BF16);

    //Number of iterations to run for creating map for each layer.
    unsigned int evaluate_iteration =
        zendnn::zendnn_getenv_int("ZENDNN_MATMUL_EVALUATE_ITER_BF16",
                                  MATMUL_EVALUATE_ITER_BF16);

    //finds object in map
    auto found_obj = matmul_kernel_map_bf16_helper.find(key_obj_auto);

    //If current iterations less than Skip iteration then run default algo.
    //Checks using the (0) element of map value that denotes count of iterations in map
    if (found_obj == matmul_kernel_map_bf16_helper.end() ||
            std::get<0>(found_obj->second) < skip_iteration) {

        zendnnVerbose(ZENDNN_PROFLOG,"AutoTuner BF16 SKIP Iteration");
        //Set aocl gemm initially
        zenEnvObj.zenBF16GEMMalgo = zenBF16MatMulAlgoType::MATMUL_AOCL_GEMM;

        //If Key not found in map then time the algo and add new element to map
        if (found_obj == matmul_kernel_map_bf16_helper.end()) {

            //Time start
#ifdef _WIN32
            auto start_n = std::chrono::high_resolution_clock::now();
#else
            gettimeofday(&start_n, 0);
#endif

            matmul_bf16_wrapper(ctx, zenEnvObj, dst_type, bias_type, Layout,
                                transpose_input,
                                transpose_filter,
                                M, K, N, alpha, src, lda, weights, ldb, bias,
                                has_eltwise_relu, po_ops, has_binary_index, geluType,
                                beta, dst, ldc, output_scales, scale_size, is_weights_const);

            //Time end
#ifdef _WIN32
            auto end_n = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> difference = end_n - start_n;
            cur_algo_time = difference.count();
#else
            gettimeofday(&end_n, 0);
            cur_algo_time = (end_n.tv_sec - start_n.tv_sec) * 1000.0f
                            + (end_n.tv_usec - start_n.tv_usec)/1000.0f; //time in milliseconds
#endif

            //Create new entry
            matmul_kernel_map_bf16_helper[key_obj_auto] = {1, cur_algo_time, zenBF16MatMulAlgoType::MATMUL_AOCL_GEMM}; // {iter_count, time, algo}
            matmul_kernel_map_bf16[key_obj_auto] = zenBF16MatMulAlgoType::MATMUL_AOCL_GEMM;
        }
        //If key found then increment the iter_count and run aocl algo.
        else {
            std::get<0>(found_obj->second) += 1;
            matmul_bf16_wrapper(ctx, zenEnvObj, dst_type, bias_type, Layout,
                                transpose_input,
                                transpose_filter,
                                M, K, N, alpha, src, lda, weights, ldb, bias,
                                has_eltwise_relu, po_ops, has_binary_index, geluType,
                                beta, dst, ldc, output_scales, scale_size, is_weights_const);

        }
    }
    //Read Value from map.
    //Runs after skip iterations and evaluation iterations are done.
    else if (std::get<0>(found_obj->second) > evaluate_iteration + skip_iteration) {
        //Get best algo for given layer from MAP
        zenEnvObj.zenBF16GEMMalgo = matmul_kernel_map_bf16[key_obj_auto];

        matmul_bf16_wrapper(ctx, zenEnvObj, dst_type, bias_type, Layout,
                            transpose_input,
                            transpose_filter,
                            M, K, N, alpha, src, lda, weights, ldb, bias,
                            has_eltwise_relu, po_ops, has_binary_index, geluType,
                            beta, dst, ldc, output_scales, scale_size, is_weights_const);
    }
    //Updates the map values by running different algorithms
    else {

        //Get the number of iteration already ran and select Algo to run for current iteration
        //get<0>(found_obj->second) = count of iteration
        zenEnvObj.zenBF16GEMMalgo = (std::get<0>(found_obj->second)%NUM_BF16_ALGO) +1;
        std::get<0>(found_obj->second) += 1;
        //timer start
#ifdef _WIN32
        auto start_n = std::chrono::high_resolution_clock::now();
#else
        gettimeofday(&start_n, 0);
#endif

        matmul_bf16_wrapper(ctx, zenEnvObj, dst_type, bias_type, Layout,
                            transpose_input,
                            transpose_filter,
                            M, K, N, alpha, src, lda, weights, ldb, bias,
                            has_eltwise_relu, po_ops, has_binary_index, geluType,
                            beta, dst, ldc, output_scales, scale_size, is_weights_const);

        //timer end
#ifdef _WIN32
        auto end_n = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> difference = end_n - start_n;
        cur_algo_time = difference.count();
#else
        gettimeofday(&end_n, 0);
        cur_algo_time = (end_n.tv_sec - start_n.tv_sec) * 1000.0f +
                        (end_n.tv_usec - start_n.tv_usec)/ 1000.0f; //time in milliseconds
#endif
        zendnnVerbose(ZENDNN_PROFLOG,"AutoTuner BF16 Evaluate Iteration algo:",
                      zenEnvObj.zenBF16GEMMalgo, " time:",cur_algo_time);
        //If current run gives better timing then update
        if (cur_algo_time < std::get<1>(found_obj->second)) {
            std::get<1>(found_obj->second) = cur_algo_time; //Minimum time for chosen algo
            std::get<2>(found_obj->second) =
                zenEnvObj.zenBF16GEMMalgo; //Algo with minimum time (1-NUM_OF_ALGO)
            matmul_kernel_map_bf16[key_obj_auto] = zenEnvObj.zenBF16GEMMalgo;
        }

    }
    return zenEnvObj.zenBF16GEMMalgo;
}

template <impl::data_type_t dst_type>
status_t zendnn_bf16_matmul_t<dst_type>::pd_t::init(engine_t *engine) {
    zendnnVerbose(ZENDNN_CORELOG, "zendnn_bf16_matmul_t::pd_t::init()");
    auto check_bias = [&]() -> bool {
        return !with_bias()
        || (utils::one_of(weights_md(1)->data_type, f32, bf16)
            && is_bias_1xN());
    };

    bool ok = src_md()->data_type == src_type
              && utils::one_of(weights_md()->data_type, bf16, s8, s4)
              && desc()->accum_data_type == acc_type
              && dst_md()->data_type == dst_type
              && platform::has_data_type_support(data_type::bf16) && check_bias()
              && (ndims() - 2) < 1 /*Condition to check batched matmul*/
              && attr()->has_default_values(
                  primitive_attr_t::skip_mask_t::oscale_runtime
                  | primitive_attr_t::skip_mask_t::post_ops)
              && set_default_formats()
              && gemm_based::check_gemm_compatible_formats(*this);

    //Return unimplemented if BF16 algo set to 3(MATMUL_JIT)
    zendnnEnv zenEnvObj = readEnv();
    if (zenEnvObj.zenBF16GEMMalgo == zenBF16MatMulAlgoType::MATMUL_JIT &&
            weights_md()->data_type == bf16) {
        return status::unimplemented;
    }

    zendnnOpInfo &obj = zendnnOpInfo::ZenDNNOpInfo();
    if ((obj.is_brgemm || obj.is_ref_gemm_bf16) &&
            weights_md()->data_type == bf16) {
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

    const bool batched = (batch_ndims > 0) ? true : false;

    zendnnEnv zenEnvObj = readEnv();
    const gemm_based::params_t &params = pd()->params();
    bool is_weights_const = zenEnvObj.zenWeightCache ||
                            pd()->weights_md()->is_memory_const;

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
    if (weights_type == s4 || weights_type == s8) {
        DEFINE_WOQ_SCALES_BUFFER(woqscales);
        float *woq_scales = const_cast<float *>(woqscales);
        int woq_scale_size = pd()->attr()->woqScales_.count_;
        const auto scales_d = ctx.memory_mdw(ZENDNN_ARG_ATTR_WOQ_SCALES);
        woq_scale_size = woq_scale_size > scales_d.dims()[0] ? woq_scale_size :
                         scales_d.dims()[0];
        matmul_woq_wrapper(ctx, src_type, weights_type, dst_type, bias_dt, Layout,
                           strcmp(transA, "N"),
                           strcmp(transB, "N"),
                           M, K, N, alpha, (char *)src, lda, (char *)weights, ldb, (char *)bias,
                           pd()->attr()->post_ops_, has_eltwise_relu,
                           geluType, beta, (char *)dst, ldc, woq_scales, 0, woq_scale_size,
                           beta, is_weights_const);
        return status::success;
    }
    else if (zenEnvObj.zenBF16GEMMalgo == zenBF16MatMulAlgoType::MATMUL_AUTO_BF16) {
        auto_tuner = true;
        algo_type = auto_compute_matmul_bf16(ctx, zenEnvObj, dst_type, bias_dt,Layout,
                                             strcmp(transA, "N"),strcmp(transB, "N"),
                                             M, K, N, alpha, src, lda, weights, ldb, bias, has_eltwise_relu,
                                             pd()->attr()->post_ops_, has_binary_index, geluType, beta,
                                             dst, ldc, output_scales, scale_size, is_weights_const);
    }
    else {
        matmul_bf16_wrapper(ctx, zenEnvObj, dst_type, bias_dt, Layout, strcmp(transA,
                            "N"),
                            strcmp(transB, "N"), M, K, N, alpha, src, lda, weights, ldb, bias,
                            has_eltwise_relu, pd()->attr()->post_ops_, has_binary_index,
                            geluType, beta, dst, ldc, output_scales, scale_size, is_weights_const);
    }

    zendnnVerbose(ZENDNN_PROFLOG,"zendnn_bf16_matmul auto_tuner=",
                  auto_tuner ? "True": "False",
                  " Layout=", Layout ? "CblasRowMajor(1)" : "CblasColMajor(0)", " M=", M, " N=",N,
                  " K=", K, " transA=", transA, " transB=", transB, " lda=", lda, " ldb=", ldb,
                  " ldc=", ldc, " alpha=", alpha, " beta=", beta, " batch=", batch, " relu=",
                  has_eltwise_relu, " gelu=", geluType, " algo_type=", algo_type,
                  " weight_caching=", is_weights_const ? "True": "False");
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
