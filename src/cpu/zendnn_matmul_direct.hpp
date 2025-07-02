/*******************************************************************************
* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "zendnn.hpp"


namespace zendnn {


namespace zendnn_registerBlocking_kernel_fp32 {
void matmul_avx512_fp32_registerBlocking(const float *A, const float *B,
        float *C,
        const float *bias, float alpha, float beta,
        int M, int N, int K, int MR, int NR, bool transB, ActivationPostOp post_op);

void matmul_avx512_fp32_registerBlocking_auto(const float *A, const float *B,
        float *C,
        const float *bias, float alpha, float beta,
        int M, int N, int K, bool transB, ActivationPostOp post_op);
void matmul_avx512_fp32_registerBlocking_batch(const float *A, const float *B,
        float *C,
        const float *bias, float alpha, float beta,
        int M, int N, int K, int MR, int NR, bool transB, ActivationPostOp post_op,
        int BATCH);
void matmul_avx512_fp32_registerBlocking_auto_batch(const float *A,
        const float *B,
        float *C,
        const float *bias, float alpha, float beta,
        int M, int N, int K, bool transB, ActivationPostOp post_op,
        int BATCH);
}


namespace zendnn_registerBlocking_kernel_fp32_ref {
void matmul_avx512_fp32_registerBlocking(const float *A, const float *B,
        float *C,
        const float *bias, float alpha, float beta,
        int M, int N, int K, int MR, int NR, bool transB, ActivationPostOp post_op);

void matmul_avx512_fp32_registerBlocking_auto(const float *A, const float *B,
        float *C,
        const float *bias, float alpha, float beta,
        int M, int N, int K, bool transB, ActivationPostOp post_op);
void matmul_avx512_fp32_registerBlocking_batch(const float *A, const float *B,
        float *C,
        const float *bias, float alpha, float beta,
        int M, int N, int K, int MR, int NR, bool transB, ActivationPostOp post_op,
        int BATCH);
void matmul_avx512_fp32_registerBlocking_auto_batch(const float *A,
        const float *B,
        float *C,
        const float *bias, float alpha, float beta,
        int M, int N, int K, bool transB, ActivationPostOp post_op,
        int BATCH);
}

namespace zendnn_registerBlocking_kernel_fp32_batch {

void matmul_avx512_fp32_registerBlocking_batched(const float *A, const float *B,
        float *C,
        const float *bias, float alpha, float beta,
        int M, int N, int K, int MR, int NR, bool transB, ActivationPostOp post_op,
        int BATCH);
void matmul_avx512_fp32_registerBlocking_batched_auto(const float *A,
        const float *B,
        float *C,
        const float *bias, float alpha, float beta,
        int M, int N, int K, bool transB, ActivationPostOp post_op,
        int BATCH);
}

void matmul_avx512_fp32_registerBlocking(const float *A, const float *B,
        float *C,
        const float *bias, float alpha, float beta,
        int M, int N, int K, int MR, int NR, bool transB, ActivationPostOp post_op);

void transpose_matrix(float *input, float *output, int N, int K);

void compute_tile_1x16(const float *A, const float *B, float *C,
                       const float *bias, float alpha, float beta,
                       int K, int M, int N, int i, int j, bool transB, ActivationPostOp post_op);

void compute_tile_4x16(const float *A, const float *B, float *C,
                       const float *bias, float alpha, float beta,
                       int K, int M, int N, int i, int j, bool transB, ActivationPostOp post_op);
void compute_tile_6x16(const float *A, const float *B, float *C,
                       const float *bias, float alpha, float beta,
                       int K, int M, int N, int i, int j, bool transB, ActivationPostOp post_op);
void compute_tile_8x16(const float *A, const float *B, float *C,
                       const float *bias, float alpha, float beta,
                       int K, int M, int N, int i, int j, bool transB, ActivationPostOp post_op);
void compute_tile_12x16(const float *A, const float *B, float *C,
                        const float *bias, float alpha, float beta,
                        int K, int M, int N, int i, int j, bool transB, ActivationPostOp post_op);
void compute_tile_scalar(const float *A, const float *B, float *C,
                         const float *bias, float alpha, float beta,
                         int K, int M, int N, int MR, int NR, int i, int j, bool transB,
                         ActivationPostOp post_op);
}

