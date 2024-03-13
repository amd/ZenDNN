/*******************************************************************************
* Copyright (c) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "common/zendnn_private.hpp"
#include <omp.h>
#ifndef ZENDNN_USE_AOCL_BLIS_API
    #include <cblas.h>
#else // ZENDNN_USE_AOCL_BLIS_API
    #include "cblas_with_blis_api.hpp"
#endif // ZENDNN_USE_AOCL_BLIS_API
#include <time.h>
#include <vector>
#include <cmath>
#include "zendnn_logging.hpp"
#include "zendnn_private.hpp"
#include "zendnn.hpp"

using namespace zendnn;
using tag = memory::format_tag;
using dt = memory::data_type;

#define BLIS_NORMAL_PATH1        1024
#define BLIS_NORMAL_PATH2        4096

extern float gelu_const;
extern int graph_exe_count;

//Simplified Map having Key as struct and value as Blocked Weight matrix address.
std::unordered_map<Key_matmul, const float * >
matmul_weight_caching_map;

//Map for weight caching(reordered memory) for JIT Primitive
std::unordered_map<Key_matmul, zendnn::memory >
matmul_weight_caching_map_jit_kernel;

void zenMatMul_gemm_blocked(
    zendnnEnv zenEnvObj,
    const bool auto_tuner,
    const bool Layout,
    const bool transpose_input,
    const bool transpose_filter,
    const int m,
    const int k,
    const int n,
    const float alpha,
    const float *input,
    const int lda,
    const float *filter,
    const int ldb,
    const float *bias,
    const bool relu,
    const int gelu,
    const float beta,
    float *output,
    const int ldc
) {
    unsigned int thread_qty = zenEnvObj.omp_num_threads;
    if (!transpose_filter && !transpose_input) {
#ifdef ZENDNN_ENABLE_LPGEMM
        zendnnVerbose(ZENDNN_PROFLOG,"Custom BLIS used");

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
        auto found_obj = matmul_weight_caching_map.find(key_obj);
        // Blocked BLIS API for matmul
        // Set post_ops to NULL and define reorder_param0 as 'B' for B matrix
        // Define dimentions of B matrix as reorder_param1 and reorder_param2
        // Define memory format as 'n'(non reordered) for A matrix and 'r'(reordered) for B matrix
        aocl_post_op *post_ops = NULL;
        const char reorder_param0 = 'B';
        const dim_t reorder_param1 = k;
        const dim_t reorder_param2 = n;
        const char order = 'r';
        const char trans = 'n';
        char mem_format_a = 'n', mem_format_b = 'r';
        if (found_obj == matmul_weight_caching_map.end()) {
#ifdef ZENDNN_ENABLE_LPGEMM_V4_2
            zendnnVerbose(ZENDNN_PROFLOG,"BLIS 4.2 enabled");
            siz_t b_reorder_buf_siz_req = aocl_get_reorder_buf_size_f32f32f32of32(
                                              order, trans, reorder_param0, reorder_param1, reorder_param2);
            float_t *reorder_filter = (float_t *) aligned_alloc(64,
                                      b_reorder_buf_siz_req);
            aocl_reorder_f32f32f32of32(order, trans, 'B', filter, reorder_filter, k,
                                       n, ldb);
#else
            zendnnVerbose(ZENDNN_PROFLOG,"BLIS 4.1 enabled");
            siz_t b_reorder_buf_siz_req = aocl_get_reorder_buf_size_f32f32f32of32(
                                              reorder_param0, reorder_param1, reorder_param2);
            float_t *reorder_filter = (float_t *) aligned_alloc(64,
                                      b_reorder_buf_siz_req);
            aocl_reorder_f32f32f32of32('B', filter, reorder_filter, k,
                                       n, ldb);
#endif
            //Create new entry
            matmul_weight_caching_map[key_obj] = reorder_filter;
        }

#ifdef ZENDNN_ENABLE_LPGEMM_V4_2
        // Currently 4.2 blis post ops are used
        int postop_count = 0;
        if (bias != NULL) {
            ++postop_count;
        }
        if (relu|| gelu) {
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
                float *bias_ = new float[n]();//const_cast<float*>(bias);
                if (alpha != 1.0f) {
                    #pragma omp parallel for num_threads(thread_qty)
                    for (int i=0; i<n; ++i) {
                        bias_[i] = alpha * bias[i];
                    }
                }
                post_ops->seq_vector[post_op_i++] = BIAS;
                post_ops->bias.bias = (alpha!=1.0f) ? bias_ : (float *)bias;
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
                // Add ReLU postop
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
                // Add ReLU postop
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
#endif
        //Perform MatMul using AMD BLIS
        aocl_gemm_f32f32f32of32(Layout? 'r' : 'c',
                                transpose_input ? 't' : 'n',
                                transpose_filter ? 't' : 'n', m, n, k, alpha,
                                input, lda, mem_format_a, matmul_weight_caching_map[key_obj], ldb, mem_format_b,
                                beta,
                                output, ldc,
                                post_ops);

#ifdef ZENDNN_ENABLE_LPGEMM_V4_2
        // Currently 4.2 blis post ops are used
        // Free memory for postops.
        if (bias != NULL) {
            //Bias is directly passed
            post_ops->bias.bias = NULL;
        }
        if (relu || gelu) {
            free(post_ops->eltwise);
        }
        if (postop_count > 0) {
            free(post_ops->seq_vector);
            free(post_ops);
        }
#else
        // ZenDNN post ops used when 4.1 BLIS is used
        if (bias || relu || gelu) {
            zenPostOps(zenEnvObj, output, NULL, m, 1, n,
                       ldc, 0,
                       bias, relu, gelu, NULL,
                       thread_qty, alpha);
        }
#endif

#else
        zendnnVerbose(ZENDNN_PROFLOG,"Custom blis is not used");
        cblas_sgemm(Layout? CblasRowMajor : CblasColMajor,
                    transpose_input ? CblasTrans : CblasNoTrans,
                    transpose_filter ? CblasTrans : CblasNoTrans, m, n, k, alpha,
                    input, lda, filter, ldb, beta, output, ldc);

        if (bias || relu || gelu) {
            zenPostOps(zenEnvObj, output, NULL, m, 1, n,
                       ldc, 0,
                       bias, relu, gelu, NULL,
                       thread_qty, alpha);
        }
#endif
    }
    else {
        zendnnVerbose(ZENDNN_PROFLOG,"cblas is used");
        cblas_sgemm(Layout? CblasRowMajor : CblasColMajor,
                    transpose_input ? CblasTrans : CblasNoTrans,
                    transpose_filter ? CblasTrans : CblasNoTrans, m, n, k, alpha,
                    input, lda, filter, ldb, beta, output, ldc);

        if (bias || relu || gelu) {
            zenPostOps(zenEnvObj, output, NULL, m, 1, n,
                       ldc, 0,
                       bias, relu, gelu, NULL,
                       thread_qty, alpha);
        }
    }
}

void zenMatMulPrimitive(zendnnEnv zenEnvObj, const bool Layout,
                        const bool TransA, const bool TransB, const int M,
                        const int N, const int K,
                        const float *A_Array, const float *B_Array,
                        float *C_Array, const float alpha,
                        const float beta, const int lda, const int ldb,
                        const int ldc, const float *bias, const bool relu,
                        const int gelu, bool blocked_format) {
    zendnn::engine eng(engine::kind::cpu, 0);
    zendnn::stream engine_stream(eng);

    Key_matmul key_obj;
    key_obj.transpose_input = TransA;
    key_obj.transpose_weights = TransB;
    key_obj.m = M;
    key_obj.k = K;
    key_obj.n = N;
    key_obj.lda = lda;
    key_obj.ldb = ldb;
    key_obj.ldc = ldc;
    key_obj.weights = B_Array;
    key_obj.thread_count = zenEnvObj.omp_num_threads;

    //finds object in map
    auto found_obj = matmul_weight_caching_map_jit_kernel.find(key_obj);

    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;

    float *in_arr = const_cast<float *>(A_Array);
    float *filt_arr = const_cast<float *>(B_Array);
    float *bias_arr = const_cast<float *>(bias);

    //memory dims
    memory::dims src_dims = {M, K};
    memory::dims weight_dims = {K, N};
    memory::dims bias_dims = {1, N};
    memory::dims dst_dims = {M, N};

    //strides
    memory::dims a_strides = TransA ? memory::dims {1, lda} :
                             memory::dims {lda, 1};
    memory::dims b_strides = TransB ? memory::dims {1, ldb} :
                             memory::dims {ldb, 1};

    //memory descriptors
    memory::desc src_md = memory::desc({src_dims}, dt::f32, a_strides);
    memory::desc matmul_weights_md = memory::desc({weight_dims}, dt::f32,
                                     b_strides);
    memory::desc blocked_matmul_weights_md = memory::desc({weight_dims}, dt::f32,
            tag::any);
    memory::desc bias_md = memory::desc({bias_dims}, dt::f32, tag::ab);

    memory::desc dst_md = memory::desc({dst_dims}, dt::f32, {ldc, 1});

    primitive_attr matmul_attr;
    zendnn::post_ops post_ops;
    bool post_attr=false;
    if (alpha != 1.f) {
        post_attr=true;
        post_ops.append_eltwise(/* mask */ 1, algorithm::eltwise_linear, alpha, 0);
    }
    if (beta != 0.f) {
        post_attr=true;
        post_ops.append_sum(beta);
    }
    float scale = 1.f;
    //eltwise post-ops
    if (relu) {
        post_attr=true;
        post_ops.append_eltwise(scale, algorithm::eltwise_relu, 0.f, 0.f);
    }
    else if (gelu == 1) {
        post_attr=true;
        post_ops.append_eltwise(scale, algorithm::eltwise_gelu, 1.f, 0.f);
    }
    else if (gelu == 2) {
        post_attr=true;
        post_ops.append_eltwise(scale, algorithm::eltwise_gelu_erf, 1.f, 0.f);
    }
    if (post_attr) {
        matmul_attr.set_post_ops(post_ops);
    }
    matmul_attr.set_autoTunerEnable(true);


    auto matmul_disc = blocked_format ? bias? zendnn::matmul::desc(src_md,
                       blocked_matmul_weights_md, bias_md, dst_md): zendnn::matmul::desc(src_md,
                               blocked_matmul_weights_md, dst_md): bias ? zendnn::matmul::desc(src_md,
                                       matmul_weights_md, bias_md, dst_md): zendnn::matmul::desc(src_md,
                                               matmul_weights_md, dst_md);

    auto matmul_prim_disc =
        zendnn::matmul::primitive_desc(matmul_disc, matmul_attr, eng);

    //Memory creation
    zendnn::memory user_weights_memory, src_memory, bias_memory, dst_memory;
    src_memory = memory({{src_dims}, dt::f32, a_strides}, eng, in_arr);
    user_weights_memory = memory(matmul_weights_md, eng, filt_arr);

    if (bias) {
        bias_memory = memory(bias_md, eng, bias_arr);
    }
    dst_memory = memory({{dst_dims}, dt::f32, memory::dims {ldc, 1}}, eng, C_Array);

    //Weight reordering
    zendnn::memory reordered_weights_memory;
    if (blocked_format && found_obj == matmul_weight_caching_map_jit_kernel.end()) {
        reordered_weights_memory = memory(matmul_prim_disc.weights_desc(), eng);
        reorder(user_weights_memory, reordered_weights_memory).execute(engine_stream,
                user_weights_memory, reordered_weights_memory);

        matmul_weight_caching_map_jit_kernel[key_obj] = reordered_weights_memory;
    }

    net.push_back(zendnn::matmul(matmul_prim_disc));
    if (bias) {
        net_args.push_back({{ZENDNN_ARG_SRC, src_memory},
            {ZENDNN_ARG_WEIGHTS, blocked_format ? matmul_weight_caching_map_jit_kernel[key_obj] : user_weights_memory},
            {ZENDNN_ARG_BIAS, bias_memory},
            {ZENDNN_ARG_DST, dst_memory}});
    }
    else {
        net_args.push_back({{ZENDNN_ARG_SRC, src_memory},
            {ZENDNN_ARG_WEIGHTS, blocked_format ? matmul_weight_caching_map_jit_kernel[key_obj] : user_weights_memory},
            {ZENDNN_ARG_DST, dst_memory}});
    }
    assert(net.size() == net_args.size() && "something is missing");
    for (size_t i = 0; i < net.size(); ++i) {
        net.at(i).execute(engine_stream, net_args.at(i));
    }
}


void zenMatMul_gemm(
    zendnnEnv zenEnvObj,
    const bool auto_tuner,
    const bool Layout,
    const bool transpose_input,
    const bool transpose_filter,
    const int m,
    const int k,
    const int n,
    const float alpha,
    const float *input,
    const int lda,
    const float *filter,
    const int ldb,
    const float *bias,
    const bool relu,
    const int gelu,
    const float beta,
    float *output,
    const int ldc
) {
    //Set Format to GEMM as Matrix multiplication is always GEMM
    zenEnvObj.zenConvAlgo = zenConvAlgoType::GEMM;

    unsigned int thread_qty = zenEnvObj.omp_num_threads;
    //Exploiting BLIS GEMM directly for MatMul is not optimal hence,
    //currently we take a different approach by splitting and parallelizing
    //MatMul with pipelining

    zendnnOpInfo &obj = zendnnOpInfo::ZenDNNOpInfo();
    obj.is_log = true;

    if (zenEnvObj.zenGEMMalgo == zenMatMulAlgoType::MATMUL_BLIS_GEMM1) {
        //Perform MatMul using AMD BLIS
        cblas_sgemm(Layout? CblasRowMajor : CblasColMajor,
                    transpose_input ? CblasTrans : CblasNoTrans,
                    transpose_filter ? CblasTrans : CblasNoTrans, m, n, k, alpha,
                    input, lda, filter, ldb, beta, output, ldc);
        if (bias || relu || gelu) {
            zenPostOps(zenEnvObj, output, NULL, m, 1, n,
                       ldc, 0,
                       bias, relu, gelu, NULL,
                       thread_qty, alpha);
        }
    }
    else if (zenEnvObj.zenGEMMalgo == zenMatMulAlgoType::MATMUL_ZENDNN_GEMM1) {
        //Blocked JIT kernel
        obj.is_brgemm = true;
        zenMatMulPrimitive(zenEnvObj, Layout, transpose_input, transpose_filter, m, n,
                           k,
                           input, filter, output, alpha, beta, lda, ldb, ldc, bias, relu, gelu, true);
        /*
        //Old functionality of ZENDNN_GEMM1
        zendnn_sgemm(transpose_input ? 'T' : 'N', transpose_filter ? 'T' : 'N',
                     m, n, k, alpha, input, lda, filter, ldb, beta, output, ldc);
        if (bias || relu || gelu) {
            zenPostOps(zenEnvObj, output, NULL, m, 1, n,
                       ldc, 0,
                       bias, relu, gelu, NULL,
                       thread_qty, alpha);
        }
        */
    }
    else if (zenEnvObj.zenGEMMalgo ==
             zenMatMulAlgoType::MATMUL_BLIS_BLOCKED_GEMM1) {
        zenMatMul_gemm_blocked(zenEnvObj, auto_tuner, Layout, transpose_input,
                               transpose_filter,
                               m, k, n, alpha, input, lda, filter, ldb, bias, relu, gelu, beta,
                               output, ldc);
    }
    else if (zenEnvObj.zenGEMMalgo == zenMatMulAlgoType::MATMUL_ZENDNN_GEMM2) {
        //JIT kernel call
        obj.is_brgemm = true;
        zenMatMulPrimitive(zenEnvObj, Layout, transpose_input, transpose_filter, m, n,
                           k,
                           input, filter, output, alpha, beta, lda, ldb, ldc, bias, relu, gelu, false);
    }
    else {
        zenMatmulSplit(zenEnvObj, auto_tuner, Layout, transpose_input, transpose_filter,
                       m, k, n, alpha, input, lda, filter, ldb, bias, relu, gelu, beta,
                       output, ldc);
    }
}

// Current parallel implementation for zenMatmulSplit does not consider column
// major layout as input. To overcome that limitation zenMatMul_gemm_wrapper
// re-organizes function parameters to provide expected result.
void zenMatMul_gemm_wrapper(
    const bool Layout,
    const bool transpose_input,
    const bool transpose_filter,
    const int m,
    const int k,
    const int n,
    const float alpha,
    const float *input,
    const int lda,
    const float *filter,
    const int ldb,
    const float *bias,
    const bool relu,
    const int gelu,
    const float beta,
    float *output,
    const int ldc
) {

    zendnnEnv zenEnvObj = readEnv();
    bool auto_tuner=false;
    unsigned int algo_type = zenEnvObj.zenGEMMalgo;
    // prologue code for time profiling of this kernel
#ifdef _WIN32
    auto start = std::chrono::high_resolution_clock::now();
#else
    struct timeval start, end;
    gettimeofday(&start, 0);
#endif

    //Experimental version for auto tuner
    if (zenEnvObj.zenGEMMalgo==zenMatMulAlgoType::MATMUL_AUTO) {
        auto_tuner=true;

        if (false == Layout) { //CblasColMajor
            algo_type = auto_compute_matmul(zenEnvObj, !Layout, transpose_filter,
                                            transpose_input,
                                            n, k, m, alpha, filter, ldb, input, lda, bias, relu, gelu, beta, output, ldc);
        }
        else {
            algo_type = auto_compute_matmul(zenEnvObj, Layout, transpose_input,
                                            transpose_filter,
                                            m, k, n, alpha, input, lda, filter, ldb, bias, relu, gelu, beta, output, ldc);
        }
    }
    else if (false == Layout) { //CblasColMajor
        zenMatMul_gemm(zenEnvObj, auto_tuner, !Layout, transpose_filter,
                       transpose_input, n, k, m,
                       alpha, filter, ldb, input, lda, bias, relu, gelu, beta, output, ldc);
    }
    else {
        zenMatMul_gemm(zenEnvObj, auto_tuner, Layout, transpose_input, transpose_filter,
                       m, k, n,
                       alpha, input, lda, filter, ldb, bias, relu, gelu, beta, output, ldc);
    }
    // Code for time profiling of this kernel
    float elapsed;
#ifdef _WIN32
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> difference = end - start;
    elapsed = difference.count();
#else
    gettimeofday(&end, 0);
    elapsed = timedifference_msec(start, end);
#endif

    zendnnVerbose(ZENDNN_PROFLOG, "zenMatMul_gemm auto_tuner=", auto_tuner,
                  " Layout=",
                  Layout ? "CblasRowMajor," : "CblasColMajor,",
                  " transa=", transpose_input ? "CblasTrans," : "CblasNoTrans,",
                  " transb=", transpose_filter ? "CblasTrans," : "CblasNoTrans,",
                  " m=", m, " k=", k, " n=", n, " lda=", lda, " ldb=", ldb,
                  " ldc=", ldc, " alpha=", alpha, " beta=", beta,
                  " relu=", relu, " gelu=", gelu,
                  " algo_type=", algo_type,
                  " Time=", elapsed, "ms"," graph_exe_count=",graph_exe_count);
    zendnnOpInfo &obj = zendnnOpInfo::ZenDNNOpInfo();
    if (algo_type == 4 || algo_type == 3) {
        obj.is_log = false;
        obj.is_brgemm = false;
    }
}

void zenMatMul(
    const bool Layout,
    const bool transpose_input,
    const bool transpose_filter,
    const int batch_size,
    const int *input_offsets,
    const int *weights_offsets,
    const int *dst_offsets,
    const int no_of_images,
    const int no_of_channels,
    const int no_of_filters,
    const float alpha,
    const float *input,
    const int lda,
    const float *filter,
    const int ldb,
    const float *bias,
    const bool relu,
    const int gelu,
    const float beta,
    float *output,
    const int ldc
) {
    //Check for NULL pointers
    if ((input == NULL)|| (filter == NULL) || (output == NULL)) {
        zendnnError(ZENDNN_ALGOLOG,
                    "zenMatMul Memory is not defined for input or filter or output");
        return;
    }

    if (batch_size == 1) {
        // Perform zen matmul. 'NULL' parameter indicates no biasadd fusion and
        // 'false' parameter disables ReLU activation on the MatMul output.
        zenMatMul_gemm_wrapper(Layout, transpose_input, transpose_filter,
                               no_of_images, no_of_channels, no_of_filters, alpha,
                               input + input_offsets[0], lda,
                               filter + weights_offsets[0], ldb, bias, relu, gelu, beta,
                               output + dst_offsets[0], ldc);
    }
    else {
        int group_count = 1;

        std::vector<int> M_Array;
        std::vector<int> N_Array;
        std::vector<int> K_Array;
        std::vector<float> alpha_Array;
        std::vector<float> beta_Array;
        std::vector<const float *> A_Array;
        std::vector<const float *> B_Array;
        std::vector<float *> C_Array;
        std::vector<const float *> bias_Array;
        std::vector<int> lda_Array;
        std::vector<int> ldb_Array;
        std::vector<int> ldc_Array;
        std::vector<int> group_size;
        std::vector<const float *> Add_Array;


        group_size.resize(group_count);
        M_Array.resize(group_count);
        N_Array.resize(group_count);
        K_Array.resize(group_count);
        alpha_Array.resize(group_count);
        beta_Array.resize(group_count);
        lda_Array.resize(group_count);
        ldb_Array.resize(group_count);
        ldc_Array.resize(group_count);
        A_Array.resize(batch_size);
        B_Array.resize(batch_size);
        C_Array.resize(batch_size);
        bias_Array.resize(batch_size);

        M_Array[0] = no_of_images;
        K_Array[0] = no_of_channels;
        N_Array[0] = no_of_filters;
        alpha_Array[0] = alpha;
        beta_Array[0] = beta;
        lda_Array[0] = lda;
        ldb_Array[0] = ldb;
        ldc_Array[0] = ldc;
        group_size[0] = batch_size;

        for (int i=0; i<batch_size; ++i) {

            A_Array[i] = input + input_offsets[i];
            B_Array[i] = filter+ weights_offsets[i];
            C_Array[i] = output + dst_offsets[i];
            bias_Array[i] = bias + (N_Array[0]*i);
        }

        zenBatchMatMul(Layout, transpose_input, transpose_filter,
                       M_Array.data(), N_Array.data(), K_Array.data(),
                       alpha_Array.data(), A_Array.data(), lda_Array.data(),
                       B_Array.data(), ldb_Array.data(),
                       beta_Array.data(), C_Array.data(), ldc_Array.data(),
                       group_count, group_size.data(), NULL, NULL, 1, 1,
                       bias_Array.data(), relu, gelu);
    }

}

void zenMatMulWithBias(
    const bool Layout,
    const bool transpose_input,
    const bool transpose_filter,
    const int batch_size,
    const int *input_offsets,
    const int *weights_offsets,
    const int *dst_offsets,
    const int no_of_images,
    const int no_of_channels,
    const int no_of_filters,
    const float alpha,
    const float *input,
    const int lda,
    const float *filter,
    const int ldb,
    const float *bias,
    const float beta,
    float *output,
    const int ldc
) {
    //Check for NULL pointers
    if ((input == NULL)|| (filter == NULL) || (output == NULL)) {
        zendnnError(ZENDNN_ALGOLOG,
                    "zenMatMul Memory is not defined for input or filter or output or bias");
        return;
    }
    // Perform zen matmul fusing biasadd. 'false' parameter disables ReLU
    // activation on the MatMul output.
    for (int i=0; i<batch_size; ++i)
        zenMatMul_gemm_wrapper(Layout, transpose_input, transpose_filter,
                               no_of_images, no_of_channels, no_of_filters, alpha,
                               input + input_offsets[i], lda,
                               filter + weights_offsets[i], ldb, bias, false, 0,
                               beta, output + dst_offsets[i], ldc);
}

void zenMatMulWithBiasReLU(
    const bool Layout,
    const bool transpose_input,
    const bool transpose_filter,
    const int batch_size,
    const int *input_offsets,
    const int *weights_offsets,
    const int *dst_offsets,
    const int no_of_images,
    const int no_of_channels,
    const int no_of_filters,
    const float alpha,
    const float *input,
    const int lda,
    const float *filter,
    const int ldb,
    const float *bias,
    const float beta,
    float *output,
    const int ldc
) {
    //Check for NULL pointers
    if ((input == NULL)|| (filter == NULL) || (output == NULL) || (bias == NULL)) {
        zendnnError(ZENDNN_ALGOLOG,
                    "zenMatMul Memory is not defined for input or filter or output or bias");
        return;
    }
    // Perform zen matmul fusing biasadd and ReLU activation. 'true' parameter
    // enables ReLU activation on the MatMul output.
    for (int i=0; i<batch_size; ++i)
        zenMatMul_gemm_wrapper(Layout, transpose_input, transpose_filter,
                               no_of_images, no_of_channels, no_of_filters, alpha,
                               input + input_offsets[i], lda,
                               filter + weights_offsets[i], ldb, bias, true, 0,
                               beta, output + dst_offsets[i], ldc);
}

void zenMatMulWithBiasGeLU(
    const bool Layout,
    const bool transpose_input,
    const bool transpose_filter,
    const int batch_size,
    const int *input_offsets,
    const int *weights_offsets,
    const int *dst_offsets,
    const int no_of_images,
    const int no_of_channels,
    const int no_of_filters,
    const float alpha,
    const float *input,
    const int lda,
    const float *filter,
    const int ldb,
    const float *bias,
    const float beta,
    float *output,
    const int ldc,
    const int geluType
) {
    //Check for NULL pointers
    if ((input == NULL)|| (filter == NULL) || (output == NULL) || (bias == NULL)) {
        zendnnError(ZENDNN_ALGOLOG,
                    "zenMatMul Memory is not defined for input or filter or output or bias");
        return;
    }
    // Perform zen matmul fusing biasadd and ReLU activation. 'true' parameter
    // enables ReLU activation on the MatMul output.
    for (int i=0; i<batch_size; ++i)
        zenMatMul_gemm_wrapper(Layout, transpose_input, transpose_filter,
                               no_of_images, no_of_channels, no_of_filters, alpha,
                               input + input_offsets[i], lda,
                               filter + weights_offsets[i], ldb, bias, false, geluType,
                               beta, output + dst_offsets[i], ldc);
}



//This version internally call zenMatmulSplit for each SGEMM in single batch
//zenBatchMatMul performs better with zenMatmulSplit rather direct call to
//cblas_sgemm_batch
//zenMatmulSplit take care parallelism and dividing data across threads whenever
//required, for some cases it falls back BLIS to parallelize the problem.
//TODO: Fix the weight caching with BatchMatMul for GEMM 3
void zenBatchMatMulSplitV1(zendnnEnv zenEnvObj, bool Layout,
                           CBLAS_TRANSPOSE *TransA_Array,
                           CBLAS_TRANSPOSE *TransB_Array, int *M_Array,
                           int *N_Array, int *K_Array, const float *alpha_Array,
                           const float **A_Array, int *lda_Array,
                           const float **B_Array, int *ldb_Array, const float *beta_Array,
                           float **C_Array, int *ldc_Array, int group_count,
                           int *group_size, const float **Add_Array,  int *add_shape,
                           float mul_node, int batch_size, const float **bias,
                           const bool relu, const int gelu) {

    zendnnVerbose(ZENDNN_ALGOLOG, "zenBatchMatMulSplitV1, Layout=",
                  Layout ? "CblasRowMajor," : "CblasColMajor,",
                  " group_count=", group_count);


    unsigned int thread_qty = zenEnvObj.omp_num_threads;
    unsigned int grp_start = 0;
    for (int i=0; i<group_count; i++) {
        bool transpose_input = (TransA_Array[i] == CblasNoTrans)?0:1;
        bool transpose_filter = (TransB_Array[i] == CblasNoTrans)?0:1;
        unsigned long m = M_Array[i];
        unsigned long n = N_Array[i];
        unsigned long k = K_Array[i];

        for (int j=0; j<group_size[i]; j++) {
            zenMatMul_gemm(zenEnvObj, 0, Layout, transpose_input, transpose_filter,
                           m, k, n, alpha_Array[i],
                           A_Array[grp_start + j], lda_Array[i],
                           B_Array[grp_start + j], ldb_Array[i],
                           NULL, relu, gelu, beta_Array[i],
                           C_Array[grp_start + j], ldc_Array[i]);
            if (relu || gelu)
                zenPostOps(zenEnvObj, C_Array[grp_start + j], NULL, m, 1, n,
                           ldc_Array[i], 0,
                           bias[grp_start + j], relu, gelu, NULL,
                           thread_qty);

            if (*Add_Array != nullptr) {
                // BatchMatMul + Mul + Add
                #pragma omp simd
                for (int k = 0; k < m * n; k++) {
                    // Add Array is traversed, used for computation when we have different shapes
                    // C_Array: [Batchsize x Attentionheads x M x N]
                    // Add_array: [Batchsize x 1 x M x N] or [Batchsize x 1 x 1 x N] or
                    //            [Batchsize x 1 x M x 1]
                    C_Array[grp_start + j][k] =
                        (C_Array[grp_start + j][k] * mul_node) +
                        Add_Array[(grp_start + j) / (group_size[i] / batch_size)][k %
                                (add_shape[1] * add_shape[2])];
                }
            }
            else if (mul_node != 1) {
                // BatchMatMul + Mul
                #pragma omp simd
                for (int k = 0; k < m * n; k++) {
                    C_Array[grp_start + j][k] =
                        (C_Array[grp_start + j][k] * mul_node);
                }
            }
        }
        grp_start +=group_size[i];
    }

}


//This version internally call cblas_sgemm for each SGEMM in single batch
//Parallelism and dividing data across threads happens at batch level.
void zenBatchMatMulSplitV2(zendnnEnv zenEnvObj, bool Layout,
                           CBLAS_TRANSPOSE *TransA_Array,
                           CBLAS_TRANSPOSE *TransB_Array, int *M_Array,
                           int *N_Array, int *K_Array, const float *alpha_Array,
                           const float **A_Array, int *lda_Array,
                           const float **B_Array, int *ldb_Array, const float *beta_Array,
                           float **C_Array, int *ldc_Array, int group_count,
                           int *group_size, const float **Add_Array, int *add_shape,
                           float mul_node, int batch_size, const float **bias,
                           const bool relu, const int gelu) {


    zendnnVerbose(ZENDNN_ALGOLOG, "zenBatchMatMulSplitV2,",
                  " Layout=", Layout ? "CblasRowMajor," : "CblasColMajor,",
                  " group_count=", group_count);

    unsigned int thread_qty = zenEnvObj.omp_num_threads;
    unsigned int grp_start = 0;

    for (int i=0; i<group_count; i++) {
        bool transpose_input = (TransA_Array[i] == CblasNoTrans)?0:1;
        bool transpose_filter = (TransB_Array[i] == CblasNoTrans)?0:1;

        unsigned long m = M_Array[i];
        unsigned long n = N_Array[i];
        unsigned long k = K_Array[i];

        unsigned int loopCount = (group_size[i]%thread_qty)==0 ?
                                 group_size[i]/thread_qty:
                                 (group_size[i]/thread_qty)+1;

        omp_set_max_active_levels(1);
        #pragma omp parallel num_threads(thread_qty)
        {
            for (int j=0; j<loopCount; j++) {

                int threadOffset = omp_get_thread_num()+ (j*thread_qty);
                if (threadOffset >= group_size[i]) {
                    break;
                }

                //if ZENDNN_GEMM_ALGO is set to 3, then zendnn_sgemm
                // jit based kernel will be called.
                // refer src/common/zendnn_utils.cpp
                if (zenEnvObj.zenGEMMalgo == zenMatMulAlgoType::MATMUL_ZENDNN_GEMM1) {
                    zendnn_sgemm(transpose_input ? 'T' : 'N',
                                 transpose_filter ? 'T' : 'N', m, n, k,
                                 alpha_Array[i],
                                 A_Array[grp_start + threadOffset], lda_Array[i],
                                 B_Array[grp_start + threadOffset], ldb_Array[i],
                                 beta_Array[i],
                                 C_Array[grp_start + threadOffset], ldc_Array[i]);
                }
                else {
                    cblas_sgemm(Layout ? CblasRowMajor: CblasColMajor,
                                TransA_Array[i], TransB_Array[i], m, n, k,
                                alpha_Array[i],
                                A_Array[grp_start + threadOffset], lda_Array[i],
                                B_Array[grp_start + threadOffset], ldb_Array[i],
                                beta_Array[i],
                                C_Array[grp_start + threadOffset], ldc_Array[i]);
                }
                if (relu || gelu)
                    zenPostOps(zenEnvObj, C_Array[grp_start + threadOffset], NULL, m, 1, n,
                               ldc_Array[i], 0,
                               bias[grp_start + threadOffset], relu, gelu, NULL,
                               thread_qty);

                if (*Add_Array != nullptr) {
                    // BatchMatMul + Mul + Add
                    #pragma omp simd
                    for (int k = 0; k < m * n; k++) {
                        // Add Array is traversed, used for computation when we have different shapes
                        // C_Array: [Batchsize x Attentionheads x M x N]
                        // Add_array: [Batchsize x 1 x M x N] or [Batchsize x 1 x 1 x N] or
                        //            [Batchsize x 1 x M x 1]
                        C_Array[grp_start + threadOffset][k] =
                            (C_Array[grp_start + threadOffset][k] * mul_node) +
                            Add_Array[(grp_start + threadOffset) / (group_size[i] / batch_size)][k %
                                    (add_shape[1] * add_shape[2])];
                    }
                }
                else if (mul_node != 1) {
                    // BatchMatMul + Mul
                    #pragma omp simd
                    for (int k = 0; k < m * n; k++) {
                        C_Array[grp_start + threadOffset][k] =
                            (C_Array[grp_start + threadOffset][k] * mul_node);
                    }
                }
            }
        }
        grp_start +=group_size[i];
    }
}


//This version internally call zenMatmulSplit for each SGEMM in single batch
//Parallelism and dividing data across threads happens at batch level.
void zenBatchMatMulSplitV3(zendnnEnv zenEnvObj, bool Layout,
                           CBLAS_TRANSPOSE *TransA_Array,
                           CBLAS_TRANSPOSE *TransB_Array, int *M_Array,
                           int *N_Array, int *K_Array, const float *alpha_Array,
                           const float **A_Array, int *lda_Array,
                           const float **B_Array, int *ldb_Array, const float *beta_Array,
                           float **C_Array, int *ldc_Array, int group_count,
                           int *group_size, const float **Add_Array, int *add_shape,
                           float mul_node, int batch_size, const float **bias,
                           const bool relu, const int gelu) {

    zendnnVerbose(ZENDNN_ALGOLOG, "zenBatchMatMulSplitV3, Layout=",
                  Layout ? "CblasRowMajor" : "CblasColMajor",
                  " group_count=", group_count);

    unsigned int grp_start = 0;
    unsigned int thread_qty = zenEnvObj.omp_num_threads;
    for (int i=0; i<group_count; i++) {
        bool transpose_input = (TransA_Array[i] == CblasNoTrans)?0:1;
        bool transpose_filter = (TransB_Array[i] == CblasNoTrans)?0:1;
        unsigned long m = M_Array[i];
        unsigned long n = N_Array[i];
        unsigned long k = K_Array[i];

        int outer_threads = (thread_qty>group_size[i])?(group_size[i]):(thread_qty);
        unsigned int loopCount = (group_size[i]%outer_threads)==0 ?
                                 group_size[i]/outer_threads:
                                 (group_size[i]/outer_threads)+1;

        omp_set_max_active_levels(2);
        #pragma omp parallel num_threads(outer_threads)
        {

            //TODO: Need to test this path with dfferent matrix sizes,
            //give more control over threads with nested parallelism
            int inner_threads = 1;
            //If inner_threads*outer_threads < OMP_NUM_THREADS, inner_threads will be incremented for few parent threads
            //This make sure that all the threads are utilized
            unsigned int temp = thread_qty - (inner_threads*outer_threads);
            int thread_loop = (temp%outer_threads)?(temp/outer_threads):((
                                  temp/outer_threads)+1);
            for (int j=0; j<thread_loop; j++) {
                if (omp_get_thread_num() < temp) {
                    inner_threads++;
                }
                temp = temp - outer_threads;
            }

            for (int j=0; j<loopCount; j++) {

                int threadOffset = omp_get_thread_num()+ (j*outer_threads);
                if (threadOffset >= group_size[i]) {
                    break;
                }
                zenEnvObj.omp_num_threads = inner_threads;
#if 0
                //TODO: Pass one parameter for omp_set_max_active_levels in zenMatmulSplit
                //Need to add this in zendnnEnv class
                zenMatMul_gemm(zenEnvObj, 0, Layout, transpose_input, transpose_filter, m, k, n,
                               alpha_Array[i],
                               A_Array[grp_start + threadOffset], lda_Array[i],
                               B_Array[grp_start + threadOffset], ldb_Array[i], NULL,
                               false, false, beta_Array[i], C_Array[grp_start + threadOffset],
                               ldc_Array[i]);
#else
                omp_set_max_active_levels(1);
                cblas_sgemm(Layout ? CblasRowMajor : CblasColMajor,
                            TransA_Array[i], TransB_Array[i], m, n, k,
                            alpha_Array[i],
                            A_Array[grp_start + threadOffset], lda_Array[i],
                            B_Array[grp_start + threadOffset], ldb_Array[i],
                            beta_Array[i],
                            C_Array[grp_start + threadOffset], ldc_Array[i]);
#endif
                if (relu || gelu)
                    zenPostOps(zenEnvObj, C_Array[grp_start + threadOffset], NULL, m, 1, n,
                               ldc_Array[i], 0,
                               bias[grp_start + threadOffset], relu, gelu, NULL,
                               thread_qty);

                if (*Add_Array != nullptr) {
                    // BatchMatMul + Mul + Add
                    #pragma omp simd
                    for (int k = 0; k < m * n; k++) {
                        // Add Array is traversed, used for computation when we have different shapes
                        // C_Array: [Batchsize x Attentionheads x M x N]
                        // Add_array: [Batchsize x 1 x M x N] or [Batchsize x 1 x 1 x N] or
                        //            [Batchsize x 1 x M x 1]
                        C_Array[grp_start + threadOffset][k] =
                            (C_Array[grp_start + threadOffset][k] * mul_node) +
                            Add_Array[(grp_start + threadOffset) / (group_size[i] / batch_size)][k %
                                    (add_shape[1] * add_shape[2])];
                    }
                }
                else if (mul_node != 1) {
                    // BatchMatMul + Mul
                    #pragma omp simd
                    for (int k = 0; k < m * n; k++) {
                        C_Array[grp_start + threadOffset][k] =
                            (C_Array[grp_start + threadOffset][k] * mul_node);
                    }
                }
            }
        }
        grp_start +=group_size[i];
    }
}

// ZenBatchMatMulPrimitives helps to execute using MatMul primitives.
// TODO: Add support for Primitive caching
void zenBatchMatMulPrimitive(zendnnEnv zenEnvObj, bool Layout,
                             bool TransA, bool TransB, int *M_Array,
                             int *N_Array, int *K_Array,
                             const float **A_Array, const float **B_Array,
                             float **C_Array, int *group_size,
                             const float **Add_Array, int *add_shape,
                             float mul_node, int batch_size) {
    zendnn::engine eng(engine::kind::cpu, 0);
    zendnn::stream engine_stream(eng);

    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;

    float *in_arr = const_cast<float *>(A_Array[0]);
    float *filt_arr = const_cast<float *>(B_Array[0]);
    float *output_array = const_cast<float *>(C_Array[0]);

    long M = M_Array[0];
    long N = N_Array[0];
    long K = K_Array[0];

    memory::dims src_dims = (group_size[0] == 1) ? (memory::dims) {
        M, K
} :
    (memory::dims) {
        batch_size, group_size[0]/batch_size, M, K
    };
    memory::dims weight_dims = (group_size[0] == 1) ? (memory::dims) {
        K, N
} :
    (memory::dims) {
        batch_size, group_size[0]/batch_size, K, N
    };
    memory::dims dst_dims = (group_size[0] == 1) ? (memory::dims) {
        M, N
} :
    (memory::dims) {
        batch_size, group_size[0]/batch_size, M, N
    };
    memory::dims bias_dims = (group_size[0] == 1) ? (memory::dims) {
        1, N
} :
    (memory::dims) {
        1, 1, 1, N
    };

    memory::desc src_md = memory::desc({src_dims}, dt::f32,
                                       (group_size[0] == 1) ? tag::ab : tag::abcd);
    memory::desc dst_md = memory::desc({dst_dims}, dt::f32,
                                       (group_size[0] == 1) ? tag::ab : tag::abcd);
    memory::desc matmul_weights_md =
        memory::desc({weight_dims}, dt::f32,
                     (group_size[0] == 1) ? tag::ab : (TransB ? tag::abdc : tag::abcd));
    memory::desc bias_md = memory::desc();

    zendnn::memory user_weights_memory, src_memory, dst_memory;
    src_memory = memory({{src_dims}, dt::f32,(group_size[0] == 1) ? tag::ab : tag::abcd},
    eng, in_arr);
    dst_memory = memory({{dst_dims}, dt::f32,(group_size[0] == 1) ? tag::ab : tag::abcd},
    eng, output_array);
    user_weights_memory =
    memory({{weight_dims}, dt::f32,(group_size[0] == 1) ? tag::ab : (TransB ? tag::abdc : tag::abcd)},
    eng,
    filt_arr);

    primitive_attr matmul_attr;
    if (*Add_Array != nullptr || mul_node != 1) {

        float *add_arr = const_cast<float *>(Add_Array[0]);

        const float *mul = &mul_node;
        float *mul_arr = const_cast<float *>(mul);

        memory::dims mul_dims = {1, 1, 1, 1};
        memory::dims add_dims;

        zendnn::post_ops post_ops;
        post_ops.append_binary(algorithm::binary_mul,
                               memory::desc({mul_dims}, dt::f32, tag::abcd));
        if (*Add_Array != nullptr) {
            add_dims = {batch_size, 1, add_shape[1], add_shape[2]};
            post_ops.append_binary(algorithm::binary_add,
                                   memory::desc({add_dims}, dt::f32, tag::abcd));
        }


        matmul_attr.set_post_ops(post_ops);
        matmul::desc matmul_pd1 =
            matmul::desc(src_md, matmul_weights_md, bias_md, dst_md);
        matmul::primitive_desc matmul_pd =
            matmul::primitive_desc(matmul_pd1, matmul_attr, eng);

        net.push_back(matmul(matmul_pd));

        zendnn::memory postop_memory1;
        postop_memory1 =
        memory({{mul_dims}, dt::f32, tag::abcd}, eng, mul_arr);
        if (*Add_Array != nullptr) {
            // BatchMatMul + Mul + Add
            zendnn::memory postop_memory2;
            postop_memory2 =
            memory({{add_dims}, dt::f32, tag::abcd}, eng, add_arr);

            net_args.push_back({
                {ZENDNN_ARG_SRC, src_memory},
                {ZENDNN_ARG_WEIGHTS, user_weights_memory},
                {ZENDNN_ARG_DST, dst_memory},
                {
                    ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(0) | ZENDNN_ARG_SRC_1,
                    postop_memory1
                },
                {
                    ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(1) | ZENDNN_ARG_SRC_1,
                    postop_memory2
                }});
        }
        else {
            // BatchMatMul + Mul
            net_args.push_back({
                {ZENDNN_ARG_SRC, src_memory},
                {ZENDNN_ARG_WEIGHTS, user_weights_memory},
                {ZENDNN_ARG_DST, dst_memory},
                {
                    ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(0) | ZENDNN_ARG_SRC_1,
                    postop_memory1
                }});
        }
    }
    else {
        // BatchMatMul
        matmul::desc matmul_pd1 =
            matmul::desc(src_md, matmul_weights_md, bias_md, dst_md);
        matmul::primitive_desc matmul_pd =
            matmul::primitive_desc(matmul_pd1, matmul_attr, eng);

        net.push_back(matmul(matmul_pd));
        net_args.push_back({{ZENDNN_ARG_SRC, src_memory},
            {ZENDNN_ARG_WEIGHTS, user_weights_memory},
            {ZENDNN_ARG_DST, dst_memory}});
    }
    assert(net.size() == net_args.size() && "something is missing");
    for (size_t i = 0; i < net.size(); ++i) {
        net.at(i).execute(engine_stream, net_args.at(i));
    }
}


//Batched MatMul Wrapper, internally calls BLAS cblas_sgemm_batch from BLIS
//or //zenBatchMatMulSplitV1/V2/V3
//TODO: Add support for group_count TransA and TransB
void zenBatchMatMul(bool Layout, bool TransA, bool TransB, int *M_Array,
                    int *N_Array, int *K_Array, const float *alpha_Array,
                    const float **A_Array, int *lda_Array,
                    const float **B_Array, int *ldb_Array, const float *beta_Array,
                    float **C_Array, int *ldc_Array, int group_count, int *group_size,
                    const float **Add_Array, int *add_shape, float mul_node,
                    int batch_size, const float **bias, const bool relu,
                    const int gelu) {
    zendnnOpInfo &obj = zendnnOpInfo::ZenDNNOpInfo();
    obj.is_brgemm = true;

    zendnnEnv zenEnvObj = readEnv();

    //Set Format to GEMM as Matrix multiplication is always GEMM
    zenEnvObj.zenConvAlgo = zenConvAlgoType::GEMM;

    // prologue code for time profiling of this kernel
#ifdef _WIN32
    auto start = std::chrono::high_resolution_clock::now();
#else
    struct timeval start, end;
    gettimeofday(&start, 0);
#endif

    std::vector<CBLAS_TRANSPOSE> TransA_Array(
        group_count, TransA ? CblasTrans : CblasNoTrans);
    std::vector<CBLAS_TRANSPOSE> TransB_Array(
        group_count, TransB ? CblasTrans : CblasNoTrans);

    //if (zenEnvObj.zenGEMMalgo == zenMatMulAlgoType::MATMUL_BLIS_GEMM1) {
    //Direct call to BLIS cblas_sgemm_batch is not performing well
    //TODO: check with BLIS team for optimal cblas_sgemm_batch function
#if 0
    cblas_sgemm_batch(Layout ? CblasRowMajor : CblasColMajor, &TransA_Array[0],
                      &TransB_Array[0], M_Array,
                      N_Array, K_Array, &alpha_Array[0], A_Array, lda_Array,
                      B_Array, ldb_Array, &beta_Array[0], C_Array, ldc_Array,
                      group_count, group_size);
#else
    bool isAlphaOne = std::all_of(alpha_Array,
    alpha_Array+group_count, [](int value) {
        return value == 1.0f;
    });
    if (zenEnvObj.zenGEMMalgo == zenMatMulAlgoType::MATMUL_ZENDNN_GEMM2 &&
            TransA==false && isAlphaOne && group_count==1) {
        //Todo: Fix the BatchedMatMul Primitive
        //Todo: Apply alpha_Array to BatchedMatMul Primitive.
        //Todo: Add Transpose support for input matrix.
        //Todo: Add Group_count support with different sizes.
        zenBatchMatMulPrimitive(zenEnvObj, Layout, TransA, TransB,
                                M_Array, N_Array, K_Array,
                                A_Array, B_Array, C_Array,
                                group_size, Add_Array, add_shape,
                                mul_node, batch_size);
    }
    else {
        //TODO: Test zenBatchMatMulSplitV1/V3 perf with different sizes
        //zenBatchMatMulSplitV1(zenEnvObj, Layout, &TransA_Array[0], &TransB_Array[0],
        //zenBatchMatMulSplitV3(zenEnvObj, Layout, &TransA_Array[0], &TransB_Array[0],
        if (group_size[0] == 1) {
            // For Matrices with 2Dimensions we use zenBatchMatMulSplitV3
            zenBatchMatMulSplitV3(zenEnvObj, Layout, &TransA_Array[0], &TransB_Array[0],
                                  M_Array, N_Array, K_Array, alpha_Array,
                                  A_Array, lda_Array, B_Array, ldb_Array,
                                  beta_Array, C_Array, ldc_Array,
                                  group_count, group_size, Add_Array, add_shape,
                                  mul_node, batch_size, bias, relu, gelu);
        }
        else {
            zenBatchMatMulSplitV2(zenEnvObj, Layout, &TransA_Array[0], &TransB_Array[0],
                                  M_Array, N_Array, K_Array, alpha_Array,
                                  A_Array, lda_Array, B_Array, ldb_Array,
                                  beta_Array, C_Array, ldc_Array,
                                  group_count, group_size, Add_Array, add_shape,
                                  mul_node, batch_size, bias, relu, gelu);
        }
    }
    if (obj.is_brgemm) {
        obj.is_brgemm = false;
    }

#endif
    // Code for time profiling of this kernel
    float elapsed;
#ifdef _WIN32
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> difference = end - start;
    elapsed = difference.count();
#else
    gettimeofday(&end, 0);
    elapsed = timedifference_msec(start, end);
#endif

    zendnnVerbose(ZENDNN_PROFLOG, "zenBatchMatMul, Layout=",
                  Layout ? "CblasRowMajor" : "CblasColMajor",
                  " group_count=", group_count, " group_size[0]=", group_size[0],
                  " M_Array[0]=", M_Array[0], " N_Array[0]=", N_Array[0],
                  " K_Array[0]=", K_Array[0], " alpha_Array[0]=", alpha_Array[0],
                  " beta_Array[0]=", beta_Array[0], " fusion=",
                  (*Add_Array != nullptr) ? "MulAdd" : (mul_node != 1) ? "Mul" : "0",
                  " relu=", relu, " gelu=", gelu, " Time=", elapsed, "ms");

}


//Matmul kernel
void zenMatmulSplit(
    zendnnEnv zenEnvObj,
    const bool auto_tuner,
    const bool Layout,
    const bool transpose_input,
    const bool transpose_filter,
    const int m,
    const int k,
    const int n,
    const float alpha,
    const float *input,
    const int lda,
    const float *filter,
    const int ldb,
    const float *bias,
    const bool relu,
    const int gelu,
    const float beta,
    float *output,
    const int ldc
) {

    unsigned int thread_qty = zenEnvObj.omp_num_threads;

    //TODO: Naming convention need to change like for "images"
    zendnnVerbose(ZENDNN_ALGOLOG, "zenMatmulSplit, Layout=",
                  Layout? "CblasRowMajor" : "CblasColMajor", " transpose_input=",
                  transpose_input, " transpose_filter=", transpose_filter, " M=", m,
                  " K=", k, " N=", n, " lda=", lda, " ldb=", ldb, " ldc=", ldc,
                  " relu=", relu, " gelu=", gelu, " alpha=", alpha, " beta=", beta);

    //l2 is level 2 no. of threads for nested parallelism.
    // thread_qty is level 1 no. of threads
    // Currently nested parallelism is disabled. If l2 is
    // more than 1, then thread_qty is set to 1.
    unsigned int l2_num_threads = 1;

    if (transpose_input) {
        l2_num_threads = thread_qty;
        thread_qty = 1;
        omp_set_max_active_levels(2);
    }
    else {
        l2_num_threads = 1;
        thread_qty = zenEnvObj.omp_num_threads;
        omp_set_max_active_levels(1);
    }

    float *data_col = NULL;
    data_col = (float *)input;

    unsigned int m_merge_count_rem = m%thread_qty;
    omp_set_dynamic(0);

    #pragma omp parallel num_threads(thread_qty)
    {
        if ((thread_qty%l2_num_threads)!=0 && omp_get_num_threads()==(thread_qty-1)) {
            l2_num_threads = thread_qty%l2_num_threads;
        }
#if BLIS_EXPERT
        //creating blis expert interface
        blis_expert blis_obj(l2_num_threads,
                             transpose_input?BLIS_TRANSPOSE:BLIS_NO_TRANSPOSE,
                             transpose_filter?BLIS_TRANSPOSE:BLIS_NO_TRANSPOSE,
                             alpha, beta);
#endif
        unsigned int m_per_thread = m/thread_qty;
        if (m_merge_count_rem && (omp_get_thread_num() < m_merge_count_rem)) {
            m_per_thread++;
        }

        int threadOffset = (omp_get_thread_num() * m_per_thread);
        if (m_merge_count_rem) {
            threadOffset = (omp_get_thread_num() * (m/thread_qty + 1));
            if (omp_get_thread_num() > m_merge_count_rem) {
                threadOffset = (omp_get_thread_num() * (m/thread_qty) +
                                m_merge_count_rem);
            }
        }
        unsigned long inputOffset = ((unsigned long)k * threadOffset);
        unsigned long outputOffset = ((unsigned long)ldc * threadOffset);

        unsigned long gemmRows = m_per_thread;

        if (zenEnvObj.zenGEMMalgo == zenMatMulAlgoType::MATMUL_BLIS_GEMM2) {

#if BLIS_EXPERT
            if (transpose_input)
                bli_obj_create_with_attached_buffer(blis_obj.dt, gemmRows,
                                                    k,
                                                    data_col+inputOffset,
                                                    1, lda, &blis_obj.a);
            else
                bli_obj_create_with_attached_buffer(blis_obj.dt, gemmRows,
                                                    k,
                                                    data_col+inputOffset,
                                                    lda, 1, &blis_obj.a);

            if (transpose_filter)
                bli_obj_create_with_attached_buffer(blis_obj.dt, k,
                                                    n,
                                                    (void *)filter,
                                                    1, ldb, &blis_obj.b);

            else
                bli_obj_create_with_attached_buffer(blis_obj.dt, k,
                                                    n,
                                                    (void *)filter,
                                                    ldb, 1, &blis_obj.b);
            bli_obj_create_with_attached_buffer(blis_obj.dt, gemmRows, n,
                                                output+outputOffset, ldc, 1, &blis_obj.c);
            bli_gemm_ex(&blis_obj.alpha, &blis_obj.a, &blis_obj.b, &blis_obj.beta,
                        &blis_obj.c, NULL, &blis_obj.rntm);

#else
            cblas_sgemm(Layout ? CblasRowMajor : CblasColMajor,
                        transpose_input ? CblasTrans : CblasNoTrans,
                        transpose_filter ? CblasTrans : CblasNoTrans, gemmRows, n, k,
                        alpha, input + inputOffset, lda, filter, ldb, beta,
                        output + outputOffset, ldc);
#endif

            //Below Bias and activation code can be eliminated if not required
            unsigned long biasOffset = outputOffset;
            if (bias || relu || gelu) {
                zenPostOps(zenEnvObj, output, NULL, gemmRows, 1, n,
                           ldc, biasOffset,
                           bias, relu, gelu, NULL,
                           l2_num_threads, alpha);
            }
        }
        else if (zenEnvObj.zenGEMMalgo == zenMatMulAlgoType::MATMUL_ZENDNN_GEMM2) {
            zendnn_sgemm(transpose_input ? 'T' : 'N', transpose_filter ? 'T' : 'N',
                         gemmRows, n, k, alpha, data_col+inputOffset, lda, filter,
                         ldb, beta, output+outputOffset, ldc);

            //Below Bias and activation code can be eliminated if not required
            unsigned long biasOffset = outputOffset;
            if (bias || relu || gelu) {
                zenPostOps(zenEnvObj, output, NULL, gemmRows, 1, n,
                           ldc, biasOffset,
                           bias, relu, gelu, NULL,
                           l2_num_threads, alpha);
            }
        }
        else {
            zenMatMul_gemm_blocked(zenEnvObj, auto_tuner, Layout,
                                   transpose_input, transpose_filter,
                                   gemmRows, k, n, alpha, data_col+inputOffset, lda, filter,
                                   ldb, bias, relu, gelu, beta, output+outputOffset, ldc);
        }

    }
}
