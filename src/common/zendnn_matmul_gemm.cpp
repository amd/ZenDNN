/*******************************************************************************
* Copyright (c) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <cblas.h>
#include <time.h>
#include <vector>
#include <cmath>
#include "zendnn_logging.hpp"
#include "zendnn_private.hpp"
#include "zendnn.hpp"

using namespace zendnn;

#define BLIS_NORMAL_PATH1        1024
#define BLIS_NORMAL_PATH2        4096

extern float gelu_const;
extern unsigned int graph_exe_count;

//Simplified Map having Key as struct and value as Blocked Weight matrix address.
std::unordered_map<Key_matmul, const float * >
matmul_weight_caching_map;

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

#if USE_CUSTOM_BLIS
    zendnnInfo(ZENDNN_PROFLOG,"Custom blis is used");
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
    auto found_obj = matmul_weight_caching_map.find(key_obj);
    // Blocked BLIS API for matmul
    // Set post_ops to NULL and define reorder_param0 as 'B' for B matrix
    // Define dimentions of B matrix as reorder_param1 and reorder_param2
    // Define memory format as 'n'(non reordered) for A matrix and 'r'(reordered) for B matrix
    aocl_post_op *post_ops = NULL;
    const char reorder_param0 = 'B';
    const dim_t reorder_param1 = k;
    const dim_t reorder_param2 = n;
    char mem_format_a = 'n', mem_format_b = 'r';
    if (found_obj == matmul_weight_caching_map.end()) {
        siz_t b_reorder_buf_siz_req = aocl_get_reorder_buf_size_f32f32f32of32(
                                          reorder_param0, reorder_param1, reorder_param2);
        float_t *reorder_filter = (float_t *) aligned_alloc(64,
                                  b_reorder_buf_siz_req);
        aocl_reorder_f32f32f32of32('B', filter, reorder_filter, k,
                                   n, ldb);
        //Create new entry
        matmul_weight_caching_map[key_obj] = reorder_filter;
    }
    //Perform MatMul using AMD BLIS
    aocl_gemm_f32f32f32of32(Layout? CblasRowMajor : CblasColMajor,
                            transpose_input ? CblasTrans : CblasNoTrans,
                            transpose_filter ? CblasTrans : CblasNoTrans, m, n, k, alpha,
                            input, lda, mem_format_a, matmul_weight_caching_map[key_obj], ldb, mem_format_b,
                            beta,
                            output, ldc,
                            post_ops);
#else
    zendnnInfo(ZENDNN_PROFLOG,"Custom blis is not used");
    //Perform MatMul using AMD BLIS
    cblas_sgemm(Layout? CblasRowMajor : CblasColMajor,
                transpose_input ? CblasTrans : CblasNoTrans,
                transpose_filter ? CblasTrans : CblasNoTrans, m, n, k, alpha,
                input, lda, filter, ldb, beta, output, ldc);
#endif
    if (bias || relu || gelu) {
        zenPostOps(zenEnvObj, output, NULL, m, 1, n,
                   ldc, 0,
                   bias, relu, gelu, NULL,
                   thread_qty);
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
                       thread_qty);
        }
    }
    else if (zenEnvObj.zenGEMMalgo == zenMatMulAlgoType::MATMUL_ZENDNN_GEMM1) {
        zendnn_sgemm(transpose_input ? 'T' : 'N', transpose_filter ? 'T' : 'N',
                     m, n, k, alpha, input, lda, filter, ldb, beta, output, ldc);
        if (bias || relu || gelu) {
            zenPostOps(zenEnvObj, output, NULL, m, 1, n,
                       ldc, 0,
                       bias, relu, gelu, NULL,
                       thread_qty);
        }

    }
    else if (zenEnvObj.zenGEMMalgo ==
             zenMatMulAlgoType::MATMUL_BLIS_BLOCKED_GEMM1) {
        zenMatMul_gemm_blocked(zenEnvObj, auto_tuner, Layout, transpose_input,
                               transpose_filter,
                               m, k, n, alpha, input, lda, filter, ldb, bias, relu, gelu, beta,
                               output, ldc);
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

    zendnnInfo(ZENDNN_PROFLOG, "zenMatMul_gemm auto_tuner=", auto_tuner, " Layout=",
               Layout ? "CblasRowMajor," : "CblasColMajor,",
               " transa=", transpose_input ? "CblasTrans," : "CblasNoTrans,",
               " transb=", transpose_filter ? "CblasTrans," : "CblasNoTrans,",
               " m=", m, " k=", k, " n=", n, " lda=", lda, " ldb=", ldb,
               " ldc=", ldc, " alpha=", alpha, " beta=", beta,
               " relu=", relu, " gelu=", gelu,
               " algo_type=", algo_type,
               " Time=", elapsed, "ms"," graph_exe_count=",graph_exe_count);
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
                               filter + weights_offsets[0], ldb, NULL, false, 0, beta,
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

        M_Array[0] = no_of_images;
        K_Array[0] = no_of_channels;
        N_Array[0] = no_of_filters;
        alpha_Array[0] = alpha;
        beta_Array[0] = beta;
        lda_Array[0] = lda;
        ldb_Array[0] = ldb;
        ldc_Array[0] = ldb;
        group_size[0] = batch_size;

        for (int i=0; i<batch_size; ++i) {
            A_Array[i] = input + input_offsets[i];
            B_Array[i] = filter+ weights_offsets[i];
            C_Array[i] = output + dst_offsets[i];
        }


        zenBatchMatMul(Layout, transpose_input, transpose_filter,
                       M_Array.data(), N_Array.data(), K_Array.data(),
                       alpha_Array.data(), A_Array.data(), lda_Array.data(),
                       B_Array.data(), ldb_Array.data(),
                       beta_Array.data(), C_Array.data(), ldc_Array.data(),
                       group_count, group_size.data(), 0, NULL, 1, 1);
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
    if ((input == NULL)|| (filter == NULL) || (output == NULL) || (bias == NULL)) {
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
void zenBatchMatMulSplitV1(zendnnEnv zenEnvObj, bool Layout,
                           CBLAS_TRANSPOSE *TransA_Array,
                           CBLAS_TRANSPOSE *TransB_Array, int *M_Array,
                           int *N_Array, int *K_Array, const float *alpha_Array,
                           const float **A_Array, int *lda_Array,
                           const float **B_Array, int *ldb_Array, const float *beta_Array,
                           float **C_Array, int *ldc_Array,
                           int group_count, int *group_size) {

    zendnnInfo(ZENDNN_ALGOLOG, "zenBatchMatMulSplitV1, Layout=",
               Layout ? "CblasRowMajor," : "CblasColMajor,",
               " group_count=", group_count);


    unsigned int grp_start = 0;
    for (int i=0; i<group_count; i++) {
        bool transpose_input = (TransA_Array[i] == CblasNoTrans)?0:1;
        bool transpose_filter = (TransB_Array[i] == CblasNoTrans)?0:1;
        unsigned long m = M_Array[i];
        unsigned long n = N_Array[i];
        unsigned long k = K_Array[i];

        for (int j=0; j<group_size[i]; j++) {
            zenMatmulSplit(zenEnvObj, 0, Layout, transpose_input, transpose_filter,
                           m, k, n, alpha_Array[i],
                           A_Array[grp_start + j], lda_Array[i],
                           B_Array[grp_start + j], ldb_Array[i],
                           NULL, false, false, beta_Array[i],
                           C_Array[grp_start + j], ldc_Array[i]);
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
                           float **C_Array, int *ldc_Array,
                           int group_count, int *group_size, bool is_mul_add,
                           const float **Add_Array, float mul_node, int batch_size) {


    zendnnInfo(ZENDNN_ALGOLOG, "zenBatchMatMulSplitV2,",
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
                if (is_mul_add) {
                    #pragma omp simd
                    for (int k = 0; k < m * n; k++) {
                        C_Array[grp_start + threadOffset][k] =
                            (C_Array[grp_start + threadOffset][k] * mul_node) +
                            Add_Array[(grp_start + threadOffset) / (group_size[i] / batch_size)][k];
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
                           float **C_Array, int *ldc_Array,
                           int group_count, int *group_size) {


    zendnnInfo(ZENDNN_ALGOLOG, "zenBatchMatMulSplitV3, Layout=",
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
                zenMatmulSplit(zenEnvObj, 0, Layout, transpose_input, transpose_filter, m, k, n,
                               alpha_Array[i]
                               A_Array[i] + (A_offset*threadOffset), lda_Array[i],
                               B_Array[i] + (B_offset*threadOffset), ldb_Array[i], NULL,
                               false, false, beta_Array[i], C_Array[i] + (C_offset*threadOffset),
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

            }
        }
        grp_start +=group_size[i];
    }
}


//Batched MatMul Wrapper, internally calls BLAS cblas_sgemm_batch from BLIS
//or //zenBatchMatMulSplitV1/V2/V3
void zenBatchMatMul(bool Layout, bool TransA, bool TransB, int *M_Array,
                    int *N_Array, int *K_Array, const float *alpha_Array,
                    const float **A_Array, int *lda_Array,
                    const float **B_Array, int *ldb_Array, const float *beta_Array,
                    float **C_Array, int *ldc_Array, int group_count, int *group_size,
                    bool is_mul_add, const float **Add_Array, float mul_node,
                    int batch_size) {

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
    //TODO: Test zenBatchMatMulSplitV1/V3 perf with different sizes
    //zenBatchMatMulSplitV1(zenEnvObj, Layout, &TransA_Array[0], &TransB_Array[0],
    //zenBatchMatMulSplitV3(zenEnvObj, Layout, &TransA_Array[0], &TransB_Array[0],
    zenBatchMatMulSplitV2(zenEnvObj, Layout, &TransA_Array[0], &TransB_Array[0],
                          M_Array, N_Array, K_Array, alpha_Array,
                          A_Array, lda_Array, B_Array, ldb_Array,
                          beta_Array, C_Array, ldc_Array,
                          group_count, group_size, is_mul_add, Add_Array,
                          mul_node, batch_size);
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

    zendnnInfo(ZENDNN_PROFLOG, "zenBatchMatMul, Layout=",
               Layout ? "CblasRowMajor" : "CblasColMajor",
               " group_count=", group_count, " group_size[0]=", group_size[0],
               " M_Array[0]=", M_Array[0], " N_Array[0]=", N_Array[0],
               " K_Array[0]=", K_Array[0], " alpha_Array[0]=", alpha_Array[0],
               " beta_Array[0]=", beta_Array[0], " Time=", elapsed, "ms");

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
    zendnnInfo(ZENDNN_ALGOLOG, "zenMatmulSplit, Layout=",
               Layout? "CblasRowMajor" : "CblasColMajor", " transpose_input=",
               transpose_input, " transpose_filter=", transpose_filter, " M=", m,
               " K=", k, " N=", n, " lda=", lda, " ldb=", ldb, " ldc=", ldc,
               " relu=", relu, " gelu=", gelu, " alpha=", alpha, " beta=", beta);

    //l2 is level 2 no. of threads for nested parallelism.
    // thread_qty is level 1 no. of threads
    // Currently nested parallelism is disabled. If l2 is
    // more than 1, then thread_qty is set to 1.
    unsigned int l2_num_threads = 1;


    //Experimental version for auto tuner
    // Nested Parallelism is disabled
    if (auto_tuner) {
        l2_num_threads = 1;
        thread_qty = zenEnvObj.omp_num_threads;
        omp_set_max_active_levels(1);
    }
    else if (zenEnvObj.zenGEMMalgo == zenMatMulAlgoType::MATMUL_ZENDNN_GEMM1 ||
             transpose_input) {
        //if ZENDNN_GEMM_ALGO is set to 3 and transpose_input is
        // enabled, then zendnn_sgemm jit based kernel will be
        // called.
        // refer src/common/zendnn_utils.cpp
        l2_num_threads = thread_qty;
        thread_qty = 1;
        omp_set_max_active_levels(2);
    }
    else {

#if BLIS_EXPERT
        //If M, N and K >=1024, NORMAL path of BLIS give optimal performance. This is
        //based on heuristic with different model and different BS
        //blis_m_kernel and blis_n_kernel is choosen based on the BLIS mxn kernel(6x16).
        //For optimal performance a thread need to work with atleast m=6 and n=16
        //for mxn kernel. if m/6 < n/16, there is no benifit in splitting m, as it will
        //make more skinny matix sizes.
        //If (m < 6 || n < 16), there is no benefit of splitting,
        //as residual kernel will be called instead of optimal 6x16 kernel
        //Zen2 has 16 registers which can store 128 floats, BLIS uses
        //96(6x16=12 ymm registers) for output and 32 floats (4 ymm  registers)
        //are used to load A and B
        //TODO: check 1024 value with SSP64 or more than 32 threads
        int blis_m_kernel = 6;
        int blis_n_kernel = 16;
        if ((m/blis_m_kernel < n/blis_n_kernel) || (m >=BLIS_NORMAL_PATH1 &&
                n>=BLIS_NORMAL_PATH1 &&  k>=BLIS_NORMAL_PATH1) || (m >BLIS_NORMAL_PATH2 ||
                        n >BLIS_NORMAL_PATH2)) {
            l2_num_threads = thread_qty;
        }
        else {
            //If m is large enough to accomodate total threads*6, then go for split.
            if (m > (thread_qty*blis_m_kernel)) {
                l2_num_threads = 1;
            }
            else { // create nested parallelism
                //TODO: Need to check below code with nested parallelism model
                //temp = ((m%BLIS_KERNEL_DIMS_M)==0?m/BLIS_KERNEL_DIMS_M:
                //              ((m/BLIS_KERNEL_DIMS_M)+1));
                //l2_num_threads = thread_qty/(temp);
                l2_num_threads = thread_qty;
            }
        }
        thread_qty = (thread_qty%l2_num_threads)==0?(thread_qty/l2_num_threads):
                     (thread_qty/l2_num_threads)+1;
        omp_set_max_active_levels(2);
#else
        //UnOptimized path, added to support CBLAS interface
        //TODO: Check if this can be removed completely
        l2_num_threads = thread_qty;
        thread_qty = 1;
        omp_set_max_active_levels(2);
#endif

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
        }
        else {
            zendnn_sgemm(transpose_input ? 'T' : 'N', transpose_filter ? 'T' : 'N',
                         gemmRows, n, k, alpha, data_col+inputOffset, lda, filter,
                         ldb, beta, output+outputOffset, ldc);

        }

        //Below Bias and activation code can be eliminated if not required
        unsigned long biasOffset = outputOffset;
        if (bias || relu || gelu) {
            zenPostOps(zenEnvObj, output, NULL, gemmRows, 1, n,
                       ldc, biasOffset,
                       bias, relu, gelu, NULL,
                       l2_num_threads);
        }
    }
}
