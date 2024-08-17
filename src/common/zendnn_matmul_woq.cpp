/*******************************************************************************
* Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "common/zendnn_private.hpp"
#include "zendnn.hpp"

using namespace zendnn;
using tag = memory::format_tag;
using dt = memory::data_type;

extern std::mutex map_mutex;

std::unordered_map<Key_matmul, int8_t * >
matmul_weight_caching_map_aocl_woq_bf16;

std::unordered_map<Key_matmul, int16_t * >
matmul_weight_caching_map_ref_woq_bf16;

std::unordered_map<Key_matmul, float * >
matmul_weight_caching_map_ref_woq_f32;

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

int cvt_int4_to_bf16(const int8_t *weights, int16_t *wei_bf16, int k, int n,
                     float *scales, int scale_size) {
    int val_idx = 0;
    for (int i=0; i<k*n; i++) {
        int t1 = impl::int4_t::extract(weights[val_idx],
                                       impl::int4_extract_t::low_half);
        int t2 = impl::int4_t::extract(weights[val_idx],
                                       impl::int4_extract_t::high_half);

        float wei_f32 = 1.001957f*scales[i%scale_size]*((float)(t1));

        float_to_bf16((&wei_f32),(wei_bf16 + i));
        i++;
        if (i < k*n) {
            wei_f32 = 1.001957f*scales[i%scale_size]*((float)(t2));
            float_to_bf16((&wei_f32),(wei_bf16 + i));
        }
        val_idx++;
    }
    return 0;
}

int cvt_int8_to_bf16(const int8_t *weights, int16_t *wei_bf16, int k, int n,
                     float *scales, int scale_size) {
    #pragma omp parallel for
    for (int i=0; i<k*n; i++) {
        float wei_f32 = 1.001957f*scales[i%scale_size]*(weights[i]);
        float_to_bf16((&wei_f32),(wei_bf16 + i));
    }
    return 0;
}

int cvt_int4_to_f32(const int8_t *weights, float *wei_f32, int k, int n,
                    float *scales, int scale_size) {
    int val_idx = 0;
    for (int i=0; i<k*n; i++) {
        int t1 = impl::int4_t::extract(weights[val_idx],
                                       impl::int4_extract_t::low_half);
        int t2 = impl::int4_t::extract(weights[val_idx],
                                       impl::int4_extract_t::high_half);
        wei_f32[i] = scales[i%scale_size]*((float)(t1));
        i++;
        if (i < k*n) {
            wei_f32[i] = scales[i%scale_size]*((float)(t2));
        }
        val_idx++;
    }
    return 0;
}

int cvt_int8_to_f32(const int8_t *weights, float *wei_f32, int k, int n,
                    float *scales, int scale_size) {
    #pragma omp parallel for
    for (int i=0; i<k*n; i++) {
        wei_f32[i] = scales[i%scale_size]*(weights[i]);
    }
    return 0;
}

template<typename T>
aocl_post_op *create_aocl_post_ops(const impl::exec_ctx_t &ctx,
                                   const zendnn_post_ops &po,
                                   int n, const float alpha, const T *bias,
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
    bool is_weights_const
) {
    zendnnEnv zenEnvObj = readEnv();
    unsigned int thread_qty = zenEnvObj.omp_num_threads;
    Key_matmul key_obj;
    key_obj.transpose_input = transA;
    key_obj.transpose_weights = transB;
    key_obj.m = M;
    key_obj.k = K;
    key_obj.n = N;
    key_obj.lda = lda;
    key_obj.ldb = ldb;
    key_obj.ldc = ldc;
    key_obj.weights = weights;
    key_obj.thread_count = thread_qty;

    //finds object in map
    auto found_obj = matmul_weight_caching_map_ref_woq_bf16.find(key_obj);
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
    if (!is_weights_const ||
            found_obj == matmul_weight_caching_map_ref_woq_bf16.end()) {
        int16_t *wei_bf16 = (int16_t *)malloc(sizeof(int16_t)*K*N);

        if (weights_type == zendnn_s4) { //Convert S4 to BF16
            cvt_int4_to_bf16(weights, wei_bf16, K, N, wei_scale, scale_size);
        }
        else { //Convert S8 to BF16
            cvt_int8_to_bf16(weights, wei_bf16, K, N, wei_scale, scale_size);
        }
        siz_t b_reorder_buf_siz_req = aocl_get_reorder_buf_size_bf16bf16f32of32(
                                          'r', trans, reorder_param0, reorder_param1, reorder_param2);
        reorder_filter = (int16_t *) aligned_alloc(64,
                         b_reorder_buf_siz_req);
        aocl_reorder_bf16bf16f32of32('r', trans, 'B', wei_bf16, reorder_filter, K,
                                     N, ldb);
        free(wei_bf16);
    }
    else {
        reorder_filter = matmul_weight_caching_map_ref_woq_bf16[key_obj];
    }
    aocl_post_op *post_ops = NULL;

    if (dst_type == zendnn_bf16) {
        int postop_count= 1;
        post_ops = create_aocl_post_ops<int16_t>(ctx, po_ops, N,
                   alpha, (const int16_t *)bias, has_eltwise_relu, geluType, (int16_t *)output,
                   postop_count, &alpha);
        //Perform MatMul using AMD BLIS
        aocl_gemm_bf16bf16f32obf16(Layout? 'r' : 'c',
                                   transA ? 't' : 'n',
                                   transB ? 't' : 'n', M, N, K,
                                   alpha,
                                   src, lda, mem_format_a,
                                   reorder_filter, ldb,
                                   mem_format_b,
                                   beta,
                                   (int16_t *)output, ldc,
                                   post_ops);
    }
    else {
        int postop_count= 1;
        post_ops = create_aocl_post_ops<float>(ctx, po_ops, N,
                                               alpha, (const float *)bias, has_eltwise_relu, geluType, (float *)output,
                                               postop_count, &alpha);
        aocl_gemm_bf16bf16f32of32(Layout? 'r' : 'c',
                                  transA ? 't' : 'n',
                                  transB ? 't' : 'n', M, N, K,
                                  alpha,
                                  src, lda, mem_format_a,
                                  reorder_filter, ldb,
                                  mem_format_b,
                                  beta,
                                  (float *)output, ldc,
                                  post_ops);
    }
    // Free memory for postops.
    if (bias != NULL) {
#ifdef ZENDNN_ENABLE_LPGEMM_V5_0
        post_ops->bias->bias=NULL;
        free(post_ops->bias);
#else
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
    bool is_weights_const
) {
    zendnnEnv zenEnvObj = readEnv();
    unsigned int thread_qty = zenEnvObj.omp_num_threads;
    Key_matmul key_obj;
    key_obj.transpose_input = transA;
    key_obj.transpose_weights = transB;
    key_obj.m = M;
    key_obj.k = K;
    key_obj.n = N;
    key_obj.lda = lda;
    key_obj.ldb = ldb;
    key_obj.ldc = ldc;
    key_obj.weights = weights;
    key_obj.thread_count = thread_qty;

    //finds object in map
    auto found_obj = matmul_weight_caching_map_ref_woq_f32.find(key_obj);
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
    if (!is_weights_const ||
            found_obj == matmul_weight_caching_map_ref_woq_f32.end()) {
        float *wei_f32 = (float *)malloc(sizeof(float)*K*N);

        if (weights_type == zendnn_s4) { //Convert S4 to FP32
            cvt_int4_to_f32(weights, wei_f32, K, N, wei_scale, scale_size);
        }
        else { //Convert S8 to FP32
            cvt_int8_to_f32(weights, wei_f32, K, N, wei_scale, scale_size);
        }
        siz_t b_reorder_buf_siz_req = aocl_get_reorder_buf_size_f32f32f32of32(
                                          'r', trans, reorder_param0, reorder_param1, reorder_param2);
        reorder_filter = (float *) aligned_alloc(64,
                         b_reorder_buf_siz_req);
        aocl_reorder_f32f32f32of32('r', trans, 'B', wei_f32, reorder_filter, K,
                                   N, ldb);
        free(wei_f32);
    }
    else {
        reorder_filter = matmul_weight_caching_map_ref_woq_f32[key_obj];
    }
    aocl_post_op *post_ops = NULL;
    int postop_count = 0;
    post_ops = create_aocl_post_ops<float>(ctx, po_ops, N,
                                           alpha, bias == NULL ? NULL :(const float *)bias, has_eltwise_relu, geluType,
                                           (float *)output,
                                           postop_count, NULL);
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
#ifdef ZENDNN_ENABLE_LPGEMM_V5_0
            post_ops->bias->bias=NULL;
            free(post_ops->bias);
#else
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
        if (post_ops->sum != NULL) {
            free(post_ops->sum->scale_factor);
            free(post_ops->sum->zero_point);
            free(post_ops->sum);
        }
        if (post_ops->matrix_add != NULL) {
            post_ops->matrix_add = NULL;
        }
        if (post_ops->matrix_mul != NULL) {
            post_ops->matrix_mul = NULL;
        }
#endif
        free(post_ops->seq_vector);
        free(post_ops);
    }
    if (!is_weights_const) {
        free(reorder_filter);
    }
    return 0;
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
    bool is_weights_const
) {
#ifdef ZENDNN_ENABLE_LPGEMM_V5_0
    zendnnEnv zenEnvObj = readEnv();
    unsigned int thread_qty = zenEnvObj.omp_num_threads;
    Key_matmul key_obj;
    key_obj.transpose_input = transA;
    key_obj.transpose_weights = transB;
    key_obj.m = M;
    key_obj.k = K;
    key_obj.n = N;
    key_obj.lda = lda;
    key_obj.ldb = ldb;
    key_obj.ldc = ldc;
    key_obj.weights = weights;
    key_obj.thread_count = thread_qty;

    //finds object in map
    auto found_obj = matmul_weight_caching_map_aocl_woq_bf16.find(key_obj);
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
    if (!is_weights_const ||
            found_obj == matmul_weight_caching_map_aocl_woq_bf16.end()) {
        siz_t b_reorder_buf_siz_req = aocl_get_reorder_buf_size_bf16s4f32of32(
                                          'r', trans, reorder_param0, reorder_param1, reorder_param2);
        reorder_filter = (int8_t *) aligned_alloc(64,
                         b_reorder_buf_siz_req);
        aocl_reorder_bf16s4f32of32('r', trans, 'B', weights, reorder_filter, K,
                                   N, ldb);
        if (is_weights_const) {
            map_mutex.lock();
            matmul_weight_caching_map_aocl_woq_bf16[key_obj] = reorder_filter;
            map_mutex.unlock();
        }
    }
    else {
        reorder_filter = matmul_weight_caching_map_aocl_woq_bf16[key_obj];
    }

    aocl_post_op *post_ops = NULL;

    if (dst_type == zendnn_bf16) {
        int postop_count= 1;
        post_ops = create_aocl_post_ops<int16_t>(ctx, po_ops, N,
                   alpha, (const int16_t *)bias, has_eltwise_relu, geluType, (int16_t *)output,
                   postop_count, &alpha);

        //Add pre-op for S4 API
        post_ops->pre_ops = NULL;
        post_ops->pre_ops = (aocl_pre_op *)malloc(sizeof(aocl_pre_op));
        (post_ops->pre_ops)->b_zp = (aocl_pre_op_zp *)malloc(sizeof(aocl_pre_op_zp));
        (post_ops->pre_ops)->b_scl = (aocl_pre_op_sf *)malloc(sizeof(aocl_pre_op_sf));
        /* Only int8_t zero point supported in pre-ops. */
        int8_t zp = 0;
        ((post_ops->pre_ops)->b_zp)->zero_point = (int8_t *)&zp;
        ((post_ops->pre_ops)->b_zp)->zero_point_len = 1;
        /* Only float scale factor supported in pre-ops. */
        ((post_ops->pre_ops)->b_scl)->scale_factor = (float *)wei_scale;
        ((post_ops->pre_ops)->b_scl)->scale_factor_len = scale_size;
        (post_ops->pre_ops)->seq_length = 1;

        //Perform MatMul using AMD BLIS
        aocl_gemm_bf16s4f32obf16(Layout? 'r' : 'c',
                                 transA ? 't' : 'n',
                                 transB ? 't' : 'n', M, N, K,
                                 alpha,
                                 src, lda, mem_format_a,
                                 reorder_filter, ldb,
                                 mem_format_b,
                                 beta,
                                 (int16_t *)output, ldc,
                                 post_ops);
    }
    else {
        int postop_count= 1;
        post_ops = create_aocl_post_ops<float>(ctx, po_ops, N,
                                               alpha, (const float *)bias, has_eltwise_relu, geluType, (float *)output,
                                               postop_count, &alpha);
        //Add pre-op for S4 API
        post_ops->pre_ops = NULL;
        post_ops->pre_ops = (aocl_pre_op *)malloc(sizeof(aocl_pre_op));
        (post_ops->pre_ops)->b_zp = (aocl_pre_op_zp *)malloc(sizeof(aocl_pre_op_zp));
        (post_ops->pre_ops)->b_scl = (aocl_pre_op_sf *)malloc(sizeof(aocl_pre_op_sf));
        /* Only int8_t zero point supported in pre-ops. */
        int8_t zp = 0;
        ((post_ops->pre_ops)->b_zp)->zero_point = (int8_t *)&zp;
        ((post_ops->pre_ops)->b_zp)->zero_point_len = 1;
        /* Only float scale factor supported in pre-ops. */
        ((post_ops->pre_ops)->b_scl)->scale_factor = (float *)wei_scale;
        ((post_ops->pre_ops)->b_scl)->scale_factor_len = scale_size;
        (post_ops->pre_ops)->seq_length = 1;

        aocl_gemm_bf16s4f32of32(Layout? 'r' : 'c',
                                transA ? 't' : 'n',
                                transB ? 't' : 'n', M, N, K,
                                alpha,
                                src, lda, mem_format_a,
                                reorder_filter, ldb,
                                mem_format_b,
                                beta,
                                (float *)output, ldc,
                                post_ops);
    }
    // Free memory for postops.
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
        post_ops->matrix_add = NULL;
    }
    if (post_ops->matrix_mul != NULL) {
        post_ops->matrix_mul = NULL;
    }
    ((post_ops->pre_ops)->b_zp)->zero_point = NULL;
    ((post_ops->pre_ops)->b_scl)->scale_factor = NULL;
    free((post_ops->pre_ops)->b_zp);
    free((post_ops->pre_ops)->b_scl);
    free(post_ops->pre_ops);
    free(post_ops->seq_vector);
    free(post_ops);
    if (!is_weights_const) {
        free(reorder_filter);
    }
#endif
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
    bool is_weights_const
) {
    //WOQ kernel
    zendnnOpInfo &obj = zendnnOpInfo::ZenDNNOpInfo();
    int is_ref = 0;
    if (src_type == zendnn_bf16) {
#ifdef ZENDNN_ENABLE_LPGEMM_V5_0
        if (!obj.is_ref_gemm_bf16 && weights_type == zendnn_s4)
#else
        if (0)
#endif
        {
            aocl_woq_bf16(ctx, po_ops, src_type, weights_type, dst_type, bias_type, Layout,
                          transA, transB,
                          M, K, N, alpha, (int16_t *)src, lda, (int8_t *)weights, ldb, bias,
                          has_eltwise_relu, geluType, beta, (char *)dst, ldc,
                          wei_scale, 0, scale_size, do_sum,
                          is_weights_const);
        }
        else {
            is_ref = 1;
            ref_woq_bf16(ctx, po_ops, src_type, weights_type, dst_type, bias_type, Layout,
                         transA, transB,
                         M, K, N, alpha, (int16_t *)src, lda, (int8_t *)weights, ldb, bias,
                         has_eltwise_relu, geluType, beta, (char *)dst, ldc,
                         wei_scale, 0, scale_size, do_sum,
                         is_weights_const);
        }
    }
    else if (src_type == zendnn_f32) {
        is_ref = 1;
        ref_woq_f32(ctx, po_ops, src_type, weights_type, dst_type, bias_type, Layout,
                    transA, transB,
                    M, K, N, alpha, (float *)src, lda, (int8_t *)weights, ldb, (const float *)bias,
                    has_eltwise_relu, geluType, beta, (float *)dst, ldc,
                    wei_scale, 0, scale_size, do_sum,
                    is_weights_const);
    }
    zendnnVerbose(ZENDNN_PROFLOG,"zendnn_woq_matmul auto_tuner=",
                  0 ? "True": "False", " Weights=", weights_type == zendnn_s4 ? "s4": "s8",
                  " Compute=", src_type == zendnn_f32 ? "FP32": "BF16",
                  " Layout=", Layout ? "CblasRowMajor(1)" : "CblasColMajor(0)", " M=", M, " N=",N,
                  " K=", K, " transA=", transA, " transB=", transB, " lda=", lda, " ldb=", ldb,
                  " ldc=", ldc, " alpha=", alpha, " beta=", beta, " algo_type=",
                  is_ref ? "REF": "1");
    return 0;
}
