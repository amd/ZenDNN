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
#include <cmath>
#include <vector>
#include <unordered_map>
#include <tuple>

#ifndef ZENDNN_USE_AOCL_BLIS_API
    #include <cblas.h>
#else // ZENDNN_USE_AOCL_BLIS_API
    #include "cblas_with_blis_api.hpp"
#endif // ZENDNN_USE_AOCL_BLIS_API

#include "zendnn_logging.hpp"
#include "zendnn_private.hpp"
#include "zendnn.hpp"

// #define NUM_INT8_ALGO 3
// #define MATMUL_SKIP_ITER_INT8 10
// #define MATMUL_EVALUATE_ITER_INT8 10

using namespace zendnn;
using tag = memory::format_tag;
using dt = memory::data_type;
extern int graph_exe_count;
extern std::mutex map_mutex;
//Map for weight caching of jit Primitive(reordered memory)
std::unordered_map<Key_matmul, zendnn::memory >
matmul_weight_caching_map_jit_int8;

//AOCL weight caching map
std::unordered_map<Key_matmul, int8_t * >
matmul_weight_caching_map_aocl_int8;

// //AutoTuner Simplified Map having Key as struct and value as Algo.
// std::unordered_map<Key_matmul, unsigned int>
// matmul_kernel_map_int8;

// //AutoTuner Helper map
// std::unordered_map<Key_matmul,std::tuple<unsigned int, float, unsigned int>>
//         matmul_kernel_map_int8_helper;

void zenMatMulPrimitiveINT8(int thread_qty, int src_type, int dst_type,
                            int bias_type, const bool Layout, const bool TransA, const bool TransB,
                            const int M,
                            const int N, const int K, const char *A_Array, const int8_t *B_Array,
                            const char *bias, char *C_Array, const float alpha, const float beta,
                            const int lda, const int ldb,
                            const int ldc, bool has_eltwise_relu, int geluType, bool blocked_format,
                            float *scale, const int32_t zero_point_dst, int out_scale_size, float do_sum,
                            bool is_weights_const) {
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
    key_obj_reorder.thread_count = thread_qty;

    //finds object in map
    auto found_obj_reorder = matmul_weight_caching_map_jit_int8.find(
                                 key_obj_reorder);

    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;

    //Prepare Data
    char *in_arr = const_cast<char *>(A_Array);
    int8_t *filt_arr = const_cast<int8_t *>(B_Array);
    char *bias_arr = const_cast<char *>(bias);
    std::vector<float> scale_vector;
    scale_vector.insert(scale_vector.end(), scale, scale + out_scale_size);
    int32_t *zero_point_dst_nc = const_cast<int32_t *>(&zero_point_dst);

    memory::dims src_dims = {M, K};
    memory::dims weight_dims = {K, N};
    memory::dims bias_dims = {1, N};
    memory::dims dst_dims = {M, N};

    memory::dims a_strides = TransA ? memory::dims {1, lda} :
                             memory::dims {lda, 1};
    memory::dims b_strides = TransB ? memory::dims {1, ldb} :
                             memory::dims {ldb, 1};

    memory::desc src_md = memory::desc({src_dims}, src_type == zendnn_s8? dt::s8 :
                                       dt::u8, a_strides);
    memory::desc matmul_weights_md = memory::desc({weight_dims}, dt::s8,
                                     b_strides);
    memory::desc blocked_matmul_weights_md = memory::desc({weight_dims}, dt::s8,
            tag::any);

    memory::desc bias_md;
    //Bias type bf16 or f32
    if (bias_type == zendnn_s32)
        bias_md = memory::desc({bias_dims}, dt::s32, tag::ab);
    else if (bias_type == zendnn_s8)
        bias_md = memory::desc({bias_dims}, dt::s8, tag::ab);
    else if (bias_type == zendnn_f32)
        bias_md = memory::desc({bias_dims}, dt::f32, tag::ab);

    memory::desc dst_md = memory::desc({dst_dims}, dst_type == zendnn_s8 ? dt::s8 :
                                       dst_type == zendnn_s32 ?dt::s32: dt::f32, {ldc, 1});
    primitive_attr matmul_attr;
    zendnn::post_ops post_ops;
    bool post_attr = false;
    const float elt_scale = 1.0f;
    if (do_sum != 0.0) {
        post_attr = true;
        post_ops.append_sum(do_sum);
    }

    //eltwise post-ops
    if (has_eltwise_relu) {
        post_attr = true;
        post_ops.append_eltwise(elt_scale, algorithm::eltwise_relu, 0.f, 0.f);
    }
    else if (geluType == 1) {
        post_attr = true;
        post_ops.append_eltwise(elt_scale, algorithm::eltwise_gelu, 1.f, 0.f);
    }
    else if (geluType == 2) {
        post_attr = true;
        post_ops.append_eltwise(elt_scale, algorithm::eltwise_gelu_erf, 1.f, 0.f);
    }
    if (post_attr) {
        matmul_attr.set_post_ops(post_ops);
    }
    matmul_attr.set_autoTunerEnable(true);
    matmul_attr.set_output_scales(out_scale_size == 1? 0: (1<<1), scale_vector);
    matmul_attr.set_zero_points(ZENDNN_ARG_DST, 0, {ZENDNN_RUNTIME_S32_VAL});
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
    memory zp_C_mem({{1}, memory::data_type::s32, {1}}, eng, zero_point_dst_nc);

    //Weight reordering
    zendnn::memory reordered_weights_memory;
    if (blocked_format) {
        if (!is_weights_const ||
                found_obj_reorder == matmul_weight_caching_map_jit_int8.end()) {
            reordered_weights_memory = memory(matmul_prim_disc.weights_desc(), eng);
            reorder(user_weights_memory, reordered_weights_memory).execute(engine_stream,
                    user_weights_memory, reordered_weights_memory);
            //Save in map
            map_mutex.lock();
            matmul_weight_caching_map_jit_int8[key_obj_reorder] = reordered_weights_memory;
            map_mutex.unlock();
        }
        else {
            reordered_weights_memory = matmul_weight_caching_map_jit_int8[key_obj_reorder];
        }
    }

    net.push_back(zendnn::matmul(matmul_prim_disc));
    if (bias_type) {
        net_args.push_back({{ZENDNN_ARG_SRC, src_memory},
            {ZENDNN_ARG_WEIGHTS, blocked_format?reordered_weights_memory:user_weights_memory},
            {ZENDNN_ARG_BIAS, bias_memory},
            {ZENDNN_ARG_DST, dst_memory},
            {ZENDNN_ARG_ATTR_ZERO_POINTS | ZENDNN_ARG_DST, zp_C_mem}});
    }
    else {
        net_args.push_back({{ZENDNN_ARG_SRC, src_memory},
            {ZENDNN_ARG_WEIGHTS, blocked_format?reordered_weights_memory:user_weights_memory},
            {ZENDNN_ARG_DST, dst_memory},
            {ZENDNN_ARG_ATTR_ZERO_POINTS | ZENDNN_ARG_DST, zp_C_mem}});
    }
    assert(net.size() == net_args.size() && "something is missing");
    for (size_t i = 0; i < net.size(); ++i) {
        net.at(i).execute(engine_stream, net_args.at(i));
    }
}

template<typename T>
aocl_post_op *create_aocl_post_ops(
    int downscale,
    int n,
    int32_t *bias,
    bool relu,
    int gelu,
    const float *scale,
    const int8_t zero_point_dst,
    const int out_scale_size,
    float do_sum,
    T *add_buffer
) {
    aocl_post_op *post_ops = NULL;
    // By default, scale postop is always enabled.
    // Check if Bias and ReLU postops are required.
    int postop_count = downscale ? 1:0;
    if (bias != NULL) {
        ++postop_count;
    }
    if (relu || gelu) {
        ++postop_count;
    }
    if (do_sum) {
        ++postop_count;
    }

    // Create postop for LPGEMM
    // Order of postops: BIAS -> RELU -> SCALE
    if (postop_count > 0) {
        post_ops = (aocl_post_op *) malloc(sizeof(aocl_post_op));
        dim_t max_post_ops_seq_length = postop_count;
        post_ops->seq_vector = (AOCL_POST_OP_TYPE *) malloc(max_post_ops_seq_length *
                               sizeof(AOCL_POST_OP_TYPE));

        // Iterate through each postop, check and add it if needed.
        int post_op_i = 0;
        if (bias != NULL) {
            // Add bias postop
            // Bias is of type int16 (accumulation type)
            post_ops->seq_vector[post_op_i++] = BIAS;
            //int bias_size = n * sizeof(int32_t);
            post_ops->bias.bias = (int32_t *) bias;//malloc(bias_size);
            /*if (post_ops->bias.bias != NULL) {
                memcpy(post_ops->bias.bias, bias, bias_size);
            }*/
        }
        // Add dst scale and dst zero_point postop
        if (downscale) {
            post_ops->seq_vector[post_op_i++] = SCALE;
            post_ops->sum.is_power_of_2 = FALSE;
            //post_ops->sum.buff = NULL;
#ifdef ZENDNN_ENABLE_LPGEMM_V5_0
            post_ops->sum.zero_point = malloc(sizeof(int8_t)); //NULL;
            int8_t *t_zp = (int8_t *)post_ops->sum.zero_point;
            t_zp[0] = zero_point_dst;

            int scale_size = out_scale_size * sizeof(float);
            post_ops->sum.scale_factor = malloc(scale_size);
            float *temp_dscale_ptr = (float *)post_ops->sum.scale_factor;
            if (out_scale_size > 1) {
                for (int i=0; i<out_scale_size; ++i) {
                    temp_dscale_ptr[i] = (float)scale[i];
                }
            }
            else {
                temp_dscale_ptr[0] = (float)scale[0];
            }
            post_ops->sum.scale_factor_len = out_scale_size;
            post_ops->sum.zero_point_len = 1;
#else
            int zp_size = n * sizeof(int8_t);
            post_ops->sum.zero_point = (int8_t *)malloc(zp_size);
            int8_t *temp_zp_ptr = (int8_t *)post_ops->sum.zero_point;
            for (int i=0; i<n; ++i) {
                temp_zp_ptr[i] = (int8_t)zero_point_dst;
            }
            int scale_size = n * sizeof(float);
            post_ops->sum.scale_factor = malloc(scale_size);
            float *temp_dscale_ptr = (float *)post_ops->sum.scale_factor;
            if (out_scale_size > 1) {
                for (int i=0; i<n; ++i) {
                    temp_dscale_ptr[i] = (float)scale[i];
                }
            }
            else {
                for (int i=0; i<n; ++i) {
                    temp_dscale_ptr[i] = (float)scale[0];
                }
            }
#endif
        }
        //Matrix Add
#ifdef ZENDNN_ENABLE_LPGEMM_V5_0
        if (do_sum) {
            post_ops->seq_vector[post_op_i++] = MATRIX_ADD;
            //post_ops->matrix_add.matrix = NULL;
            post_ops->matrix_add.ldm = n;
            post_ops->matrix_add.matrix = (T *)add_buffer;//malloc( m * n * ele_dsize );
        }
#endif
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
            // Add GeLU_TANH postop
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
            // Add GeLU_ERF postop
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
    return post_ops;
}
void zenMatMul_gemm_s8s8s32os8(
    int thread_qty,
    const bool Layout,
    const bool transpose_input,
    const bool transpose_filter,
    const int m,
    const int k,
    const int n,
    const float alpha,
    const int8_t *input,
    const int lda,
    const int8_t *filter,
    const int ldb,
    int32_t *bias,
    const bool relu,
    const int gelu,
    const float beta,
    int8_t *output,
    const int ldc,
    const float *scale,
    const int8_t zero_point_dst,
    const int out_scale_size,
    float do_sum,
    bool is_weights_const
) {
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
    auto found_obj = matmul_weight_caching_map_aocl_int8.find(key_obj);

    const char transa = 'n', transb = 'n';
    char mem_format_a = 'n', mem_format_b = 'r', storage = 'r';
    const char reorder_param0 = 'B';
    const dim_t reorder_param1 = k;
    const dim_t reorder_param2 = n;
    const char order = 'r';
    const char trans = 'n';

    int8_t *b_reorder = NULL;
    if (!is_weights_const ||
            found_obj == matmul_weight_caching_map_aocl_int8.end()) {
        siz_t b_reorder_buf_siz_req = aocl_get_reorder_buf_size_s8s8s32os32(
                                          order, trans,
                                          reorder_param0, reorder_param1, reorder_param2);
        b_reorder = (int8_t *) aligned_alloc(64, b_reorder_buf_siz_req);
        aocl_reorder_s8s8s32os32(order, trans, 'B', filter, b_reorder,k,n,ldb);
        //Create new entry
        map_mutex.lock();
        matmul_weight_caching_map_aocl_int8[key_obj] = b_reorder;
        map_mutex.unlock();
    }
    else {
        b_reorder = matmul_weight_caching_map_aocl_int8[key_obj];
    }


    aocl_post_op *post_ops = NULL;
    //Create post_ops
    post_ops = create_aocl_post_ops<int8_t>(true, n, bias, relu, gelu, scale,
                                            zero_point_dst, out_scale_size, do_sum, output);

    aocl_gemm_s8s8s32os8(storage, transa, transb, m,
                         n,
                         k, alpha, input,
                         lda, mem_format_a, b_reorder,
                         ldb, mem_format_b, beta, output,
                         ldc, post_ops);
    // Free memory for reordered filter and postops.
    if (relu || gelu) {
        free(post_ops->eltwise);
    }
    //if weights are not constant
    if (!is_weights_const) {
        free(b_reorder);
    }
    // Scale postop is always enabled so deleted directly.
    free(post_ops->sum.scale_factor);
    free(post_ops->sum.zero_point);
    free(post_ops->seq_vector);
    free(post_ops);
}

void zenMatMul_gemm_s8s8s32os32(
    int thread_qty,
    const bool Layout,
    const bool transpose_input,
    const bool transpose_filter,
    const int m,
    const int k,
    const int n,
    const float alpha,
    const int8_t *input,
    const int lda,
    const int8_t *filter,
    const int ldb,
    int32_t *bias,
    const bool relu,
    const int gelu,
    const float beta,
    int32_t *output,
    const int ldc,
    const float *scale,
    const int8_t zero_point_dst,
    const int out_scale_size,
    float do_sum,
    bool is_weights_const
) {
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
    auto found_obj = matmul_weight_caching_map_aocl_int8.find(key_obj);

    const char transa = 'n', transb = 'n';
    char mem_format_a = 'n', mem_format_b = 'r', storage = 'r';
    const char reorder_param0 = 'B';
    const dim_t reorder_param1 = k;
    const dim_t reorder_param2 = n;
    const char order = 'r';
    const char trans = 'n';
    int8_t *b_reorder = NULL;
    if (!is_weights_const ||
            found_obj == matmul_weight_caching_map_aocl_int8.end()) {
        siz_t b_reorder_buf_siz_req = aocl_get_reorder_buf_size_s8s8s32os32(
                                          order, trans,
                                          reorder_param0, reorder_param1, reorder_param2);
        b_reorder = (int8_t *) aligned_alloc(64, b_reorder_buf_siz_req);
        aocl_reorder_s8s8s32os32(order, trans, 'B', filter, b_reorder,k,n,ldb);
        //Create new entry
        map_mutex.lock();
        matmul_weight_caching_map_aocl_int8[key_obj] = b_reorder;
        map_mutex.unlock();
    }
    else {
        b_reorder = matmul_weight_caching_map_aocl_int8[key_obj];
    }
    aocl_post_op *post_ops = NULL;
    //Create post_ops
    post_ops = create_aocl_post_ops<int32_t>(true, n, bias, relu, gelu, scale,
               zero_point_dst, out_scale_size, do_sum, output);

    aocl_gemm_s8s8s32os32(storage, transa, transb, m,
                          n,
                          k, alpha, input,
                          lda, mem_format_a, b_reorder,
                          ldb, mem_format_b, beta, output,
                          ldc, post_ops);
    // Free memory for reordered filter and postops.
    if (relu || gelu) {
        free(post_ops->eltwise);
    }
    //if weights are not constant
    if (!is_weights_const) {
        free(b_reorder);
    }
    // Scale postop is always enabled so deleted directly.
    free(post_ops->sum.scale_factor);
    free(post_ops->sum.zero_point);
    free(post_ops->seq_vector);
    free(post_ops);
}

void zenMatMul_gemm_u8s8s32os8(
    int thread_qty,
    const bool Layout,
    const bool transpose_input,
    const bool transpose_filter,
    const int m,
    const int k,
    const int n,
    const float alpha,
    const uint8_t *input,
    const int lda,
    const int8_t *filter,
    const int ldb,
    int32_t *bias,
    const bool relu,
    const int gelu,
    const float beta,
    int8_t *output,
    const int ldc,
    const float *scale,
    const int8_t zero_point_dst,
    const int out_scale_size,
    float do_sum,
    bool is_weights_const
) {
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
    auto found_obj = matmul_weight_caching_map_aocl_int8.find(key_obj);

    const char transa = 'n', transb = 'n';
    char mem_format_a = 'n', mem_format_b = 'r', storage = 'r';
    const char reorder_param0 = 'B';
    const dim_t reorder_param1 = k;
    const dim_t reorder_param2 = n;
    const char order = 'r';
    const char trans = 'n';
    int8_t *b_reorder = NULL;
    if (!is_weights_const ||
            found_obj == matmul_weight_caching_map_aocl_int8.end()) {
        siz_t b_reorder_buf_siz_req = aocl_get_reorder_buf_size_u8s8s32os32(
                                          order, trans,
                                          reorder_param0, reorder_param1, reorder_param2);
        b_reorder = (int8_t *) aligned_alloc(64, b_reorder_buf_siz_req);
        aocl_reorder_u8s8s32os32(order, trans, 'B', filter, b_reorder,k,n,ldb);
        //Create new entry
        map_mutex.lock();
        matmul_weight_caching_map_aocl_int8[key_obj] = b_reorder;
        map_mutex.unlock();
    }
    else {
        b_reorder = matmul_weight_caching_map_aocl_int8[key_obj];
    }
    aocl_post_op *post_ops = NULL;
    //Create post_ops
    post_ops = create_aocl_post_ops<int8_t>(true, n, bias, relu, gelu, scale,
                                            zero_point_dst, out_scale_size, do_sum, output);

    aocl_gemm_u8s8s32os8(storage, transa, transb, m,
                         n,
                         k, alpha, input,
                         lda, mem_format_a, b_reorder,
                         ldb, mem_format_b, beta, output,
                         ldc, post_ops);
    // Free memory for reordered filter and postops.
    if (relu || gelu) {
        free(post_ops->eltwise);
    }
    //if weights are not constant
    if (!is_weights_const) {
        free(b_reorder);
    }
    // Scale postop is always enabled so deleted directly.
    free(post_ops->sum.scale_factor);
    free(post_ops->sum.zero_point);
    free(post_ops->seq_vector);
    free(post_ops);
}

void zenMatMul_gemm_u8s8s32os32(
    int thread_qty,
    const bool Layout,
    const bool transpose_input,
    const bool transpose_filter,
    const int m,
    const int k,
    const int n,
    const float alpha,
    const uint8_t *input,
    const int lda,
    const int8_t *filter,
    const int ldb,
    int32_t *bias,
    const bool relu,
    const int gelu,
    const float beta,
    int32_t *output,
    const int ldc,
    const float *scale,
    const int8_t zero_point_dst,
    const int out_scale_size,
    float do_sum,
    bool is_weights_const
) {
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
    auto found_obj = matmul_weight_caching_map_aocl_int8.find(key_obj);

    const char transa = 'n', transb = 'n';
    char mem_format_a = 'n', mem_format_b = 'r', storage = 'r';
    const char reorder_param0 = 'B';
    const dim_t reorder_param1 = k;
    const dim_t reorder_param2 = n;
    const char order = 'r';
    const char trans = 'n';
    int8_t *b_reorder = NULL;
    if (!is_weights_const ||
            found_obj == matmul_weight_caching_map_aocl_int8.end()) {
        siz_t b_reorder_buf_siz_req = aocl_get_reorder_buf_size_u8s8s32os32(
                                          order, trans,
                                          reorder_param0, reorder_param1, reorder_param2);
        b_reorder = (int8_t *) aligned_alloc(64, b_reorder_buf_siz_req);
        aocl_reorder_u8s8s32os32(order, trans, 'B', filter, b_reorder,k,n,ldb);
        //Create new entry
        map_mutex.lock();
        matmul_weight_caching_map_aocl_int8[key_obj] = b_reorder;
        map_mutex.unlock();
    }
    else {
        b_reorder = matmul_weight_caching_map_aocl_int8[key_obj];
    }
    aocl_post_op *post_ops = NULL;
    //Create post_ops
    post_ops = create_aocl_post_ops<int32_t>(true, n, bias, relu, gelu, scale,
               zero_point_dst, out_scale_size, do_sum, output);

    aocl_gemm_u8s8s32os32(storage, transa, transb, m,
                          n,
                          k, alpha, input,
                          lda, mem_format_a, b_reorder,
                          ldb, mem_format_b, beta, output,
                          ldc, post_ops);
    // Free memory for reordered filter and postops.
    if (relu || gelu) {
        free(post_ops->eltwise);
    }
    //if weights are not constant
    if (!is_weights_const) {
        free(b_reorder);
    }
    // Scale postop is always enabled so deleted directly.
    free(post_ops->sum.scale_factor);
    free(post_ops->sum.zero_point);
    free(post_ops->seq_vector);
    free(post_ops);
}

int matmul_int8_wrapper(
    zendnn::zendnnEnv zenEnvObj,
    int src_type,
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
    const int8_t *weights,
    const int ldb,
    const char *bias,
    const bool has_eltwise_relu,
    const int geluType,
    const float beta,
    char *dst,
    const int ldc,
    float *scale,
    const int32_t zero_point_dst,
    int out_scale_size,
    float do_sum,
    bool is_weights_const
) {
    zendnnOpInfo &obj = zendnnOpInfo::ZenDNNOpInfo();
    int thread_qty = zenEnvObj.omp_num_threads;
    if (zenEnvObj.zenINT8GEMMalgo == zenINT8MatMulAlgoType::MATMUL_AOCL_GEMM_INT8
#ifdef ZENDNN_ENABLE_LPGEMM_V5_0
        && (do_sum == 0.0 || do_sum == 1.0) && bias_type != zendnn_f32
#else
        && do_sum == 0.0 && bias_type != zendnn_f32
#endif
    ) {
        int bias_size = N * sizeof(int32_t);
        int32_t *bias_f32 = NULL;
        //Check if bias is NULL or not s32 type
        if (bias != NULL && bias_type != zendnn_s32) {
            bias_f32 = (int32_t *)malloc(bias_size);
            #pragma omp parallel for num_threads(thread_qty)
            for (int i=0; i<N; i++) {
                bias_f32[i] = (int32_t)bias[i];
            }
            //memcpy(bias_f32, bias, bias_size);
        }
        int8_t zero_point_dst_8 = zero_point_dst;
        if (src_type == zendnn_s8) {
            if (dst_type == zendnn_s8) {
                zenMatMul_gemm_s8s8s32os8(thread_qty, Layout, transA, transB, M, K, N, alpha,
                                          (const int8_t *)src, lda, (const int8_t *)weights, ldb,
                                          bias_type != zendnn_s32 ? bias_f32: (int32_t *)bias,
                                          has_eltwise_relu, geluType, beta, (int8_t *)dst, ldc, scale,
                                          zero_point_dst_8, out_scale_size, do_sum, is_weights_const);
            }
            else if (dst_type == zendnn_s32) {
                zenMatMul_gemm_s8s8s32os32(thread_qty, Layout, transA, transB, M, K, N, alpha,
                                           (const int8_t *)src, lda, (const int8_t *)weights, ldb,
                                           bias_type != zendnn_s32 ? bias_f32: (int32_t *)bias,
                                           has_eltwise_relu, geluType, beta, (int32_t *)dst, ldc, scale,
                                           zero_point_dst_8, out_scale_size, do_sum, is_weights_const);
            }
            else {
                //dst float32
                //CALL BRGEMM Primitive
                obj.is_brgemm = true;
                zenMatMulPrimitiveINT8(thread_qty, src_type, dst_type, bias_type, Layout,
                                       transA, transB, M, N, K, src, weights, bias, dst, alpha, beta, lda, ldb, ldc,
                                       has_eltwise_relu,
                                       geluType, true, scale, zero_point_dst, out_scale_size, do_sum,
                                       is_weights_const);
                obj.is_brgemm = false;
            }
        }
        else { // make function for src:u8
            if (dst_type == zendnn_s8) {
                zenMatMul_gemm_u8s8s32os8(thread_qty, Layout, transA, transB, M, K, N, alpha,
                                          (const uint8_t *)src, lda, (const int8_t *)weights, ldb,
                                          bias_type != zendnn_s32 ? bias_f32: (int32_t *)bias,
                                          has_eltwise_relu, geluType, beta, (int8_t *)dst, ldc, scale,
                                          zero_point_dst_8, out_scale_size, do_sum, is_weights_const);
            }
            else if (dst_type == zendnn_s32) {
                zenMatMul_gemm_u8s8s32os32(thread_qty, Layout, transA, transB, M, K, N, alpha,
                                           (const uint8_t *)src, lda, (const int8_t *)weights, ldb,
                                           bias_type != zendnn_s32 ? bias_f32: (int32_t *)bias,
                                           has_eltwise_relu, geluType, beta, (int32_t *)dst, ldc, scale,
                                           zero_point_dst_8, out_scale_size, do_sum, is_weights_const);
            }
            else {
                //dst float32
                //CALL BRGEMM Primitive
                obj.is_brgemm = true;
                zenMatMulPrimitiveINT8(thread_qty, src_type, dst_type, bias_type, Layout,
                                       transA, transB, M, N, K, src, weights, bias, dst, alpha, beta, lda, ldb, ldc,
                                       has_eltwise_relu,
                                       geluType, true, scale, zero_point_dst, out_scale_size, do_sum,
                                       is_weights_const);
                obj.is_brgemm = false;
            }
        }
    }
    else if (zenEnvObj.zenINT8GEMMalgo ==
             zenINT8MatMulAlgoType::MATMUL_BLOCKED_JIT_INT8) {
        //CALL blocked BRGEMM Primitive
        obj.is_brgemm = true;
        zenMatMulPrimitiveINT8(thread_qty, src_type, dst_type, bias_type, Layout,
                               transA, transB, M, N, K, src, weights, bias, dst, alpha, beta, lda, ldb, ldc,
                               has_eltwise_relu,
                               geluType, true, scale, zero_point_dst, out_scale_size, do_sum,
                               is_weights_const);
        obj.is_brgemm = false;
    }
    else {
        //CALL BRGEMM Primitive
        obj.is_brgemm = true;
        zenMatMulPrimitiveINT8(thread_qty, src_type, dst_type, bias_type, Layout,
                               transA, transB, M, N, K, src, weights, bias, dst, alpha, beta, lda, ldb, ldc,
                               has_eltwise_relu,
                               geluType, false, scale, zero_point_dst, out_scale_size, do_sum,
                               is_weights_const);
        obj.is_brgemm = false;
    }
    obj.is_log = true;
    return 0;
}
