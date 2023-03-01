/*******************************************************************************
* Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef COMMON_BLIS_WRAPPER_HPP
#define COMMON_BLIS_WRAPPER_HPP

#ifndef ZENDNN_USE_AOCL_BLIS_API
#error "This header file should be included only when you disable cblas"
#endif // ZENDNN_USE_AOCL_BLIS_API

#include <blis.h>

typedef enum CBLAS_LAYOUT {
    CblasRowMajor = 101,
    CblasColMajor = 102
} CBLAS_LAYOUT;
typedef enum CBLAS_TRANSPOSE {
    CblasNoTrans = 111,
    CblasTrans = 112,
    CblasConjTrans = 113
} CBLAS_TRANSPOSE;

#define CBLAS_ORDER \
    CBLAS_LAYOUT /* this for backward compatibility with CBLAS_ORDER */

void cblas_sgemm_aocl(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA,
        enum CBLAS_TRANSPOSE TransB, f77_int M, f77_int N, f77_int K,
        float alpha, const float *A, f77_int lda, const float *B, f77_int ldb,
        float beta, float *C, f77_int ldc);

void cblas_sgemv_aocl(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA,
        f77_int M, f77_int N, float alpha, const float *A, f77_int lda,
        const float *X, f77_int incX, float beta, float *Y, f77_int incY);

float cblas_sdot_aocl(
        f77_int N, const float *X, f77_int incX, const float *Y, f77_int incY);

#endif // COMMON_BLIS_WRAPPER_HPP