/*******************************************************************************
* Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include "zendnn_logging.hpp"
#include "zendnn.hpp"
#include "zendnn_helper.hpp"


using namespace zendnn;

zendnn_status_t zendnn_sgemm(char transa, char transb, int64_t M, int64_t N,
        int64_t K, float alpha, const float *A, int64_t lda, const float *B,
        const int64_t ldb, float beta, float *C, int64_t ldc);

/*
List of Input Arguments
Layout: A boolean argument to specify row-major or column-major data ordering.
transpose_input: A boolean argument to transpose input or not.
transpose_filter: A boolean argument to transpose filter or not.
m: constant integer
k: constant integer
n: constant integer
alpha: A const float to store the scaling factor for the matrix product.
input: float buffer to store input of size m-by-k.
lda: constant integer representing leading dimension of input.
filter: float buffer to store filter of size k-by-n.
ldb: constant integer representing leading dimension of filter.
bias: float buffer to store bias of size 1-by-n.
relu: A boolean argument to apply relu or not.
beta: A const float to store the scaling factor for output.
output: float buffer to store output of size m-by-n.
ldc: constant integer representing leading dimension of output.
*/
void zenMatMul_ref(
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
    const float beta,
    float *output,
    const int ldc
) {
    // Get the number of threads that could be used for parallelization
    zendnnEnv zenEnvObj = readEnv();

   //Exploiting BLIS GEMM directly, so MatMul is not optimal here,
    unsigned int thread_qty = zenEnvObj.omp_num_threads;

    //Perform MatMul using DNNL SGEMM
    zendnn_sgemm(transpose_input ? 'T' : 'N', transpose_filter ? 'T' : 'N',
            m, n, k, alpha, input, lda, filter, ldb, beta, output, ldc);

    // For biasadd fusion
    if (NULL != bias) {
        // BiasAdd is to be fused, hence populate the output with bias
        if (1 < m) {
            // for batch size more than 1, parallelize bias update for
            // each image
            #pragma omp parallel for num_threads(thread_qty)
            for (int i=0; i<m; i++) {
                for (int j=0; j<n; j++) {
                    output[i*n+j] += bias[j];
                }
            }
        }
        else {
            // for batch size is one
            #pragma omp parallel for num_threads(thread_qty)
            for (int j=0; j<n; j++) {
                output[j] += bias[j];
            }
        }
    }

    // For activation (currently ReLU only) fusion
    if (relu) {
        long int out_values = m*n;
        #pragma omp parallel for num_threads(thread_qty)
        for (long int i=0; i<out_values; i++) {
            if (0 > output[i]) {
                output[i] = 0;
            }
        }
    }
}

void zenMatMul_refWrapper(
    const bool Layout,
    const bool transpose_input,
    const bool transpose_filter,
    const int MB,
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
    const float beta,
    float *output,
    const int ldc
) {
     // prologue code for time profiling of this kernel
#ifdef _WIN32
    auto start = std::chrono::high_resolution_clock::now();
#else
    struct timeval start, end;
    gettimeofday(&start, 0);
#endif

    if (true == Layout) { //CblasRowMajor
        for (int i=0; i<MB; ++i) {
            zenMatMul_ref(Layout, transpose_input, transpose_filter, m, k, n,
                    alpha, input+(i*m*k), lda, filter+(i*k*n), ldb, bias, relu,
                    beta, output+(i*m*n), ldc);
        }
    } else { //CblasColMajor
        for (int i=0; i<MB; ++i) {
            zenMatMul_ref(!Layout, transpose_filter, transpose_input, n, k, m,
                    alpha, filter+(i*k*n), ldb, input+(i*m*k), lda, bias, relu,
                    beta, output+(i*m*n), ldc);
        }
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
    zendnnInfo(ZENDNN_PROFLOG, "zenMatMul_ref, Layout=CblasRowMajor,",
               " transa=", transpose_input ? "CblasTrans" : "CblasNoTrans",
               " transb=", transpose_filter ? "CblasTrans" : "CblasNoTrans",
               " m=", m, " k=", k, " n=", n, " alpha=", alpha, " beta=", beta,
               " lda=", lda, " ldb=", ldb, " ldc=", ldc,
               " Time=", elapsed, "ms");

}
