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

#ifdef ZENDNN_USE_AOCL_BLIS_API

#include "blis_wrapper.hpp"

void cblas_sgemm_aocl(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA,
        enum CBLAS_TRANSPOSE TransB, f77_int M, f77_int N, f77_int K,
        float alpha, const float *A, f77_int lda, const float *B, f77_int ldb,
        float beta, float *C, f77_int ldc) {
    char TA, TB;

#ifdef F77_CHAR
    F77_CHAR F77_TA, F77_TB;
#else
#define F77_TA &TA
#define F77_TB &TB
#endif

#ifdef F77_INT
    F77_INT F77_M = M, F77_N = N, F77_K = K, F77_lda = lda, F77_ldb = ldb;
    F77_INT F77_ldc = ldc;
#else
#define F77_M M
#define F77_N N
#define F77_K K
#define F77_lda lda
#define F77_ldb ldb
#define F77_ldc ldc
#endif

    if (Order == CblasColMajor) {
        if (TransA == CblasTrans)
            TA = 'T';
        else if (TransA == CblasConjTrans)
            TA = 'C';
        else if (TransA == CblasNoTrans)
            TA = 'N';
        else {
            //cblas_xerbla(2, "cblas_sgemm", "Illegal TransA setting, %d\n", TransA);
            return;
        }

        if (TransB == CblasTrans)
            TB = 'T';
        else if (TransB == CblasConjTrans)
            TB = 'C';
        else if (TransB == CblasNoTrans)
            TB = 'N';
        else {
            //cblas_xerbla(3, "cblas_sgemm", "Illegal TransB setting, %d\n", TransB);
            return;
        }

#ifdef F77_CHAR
        F77_TA = C2F_CHAR(&TA);
        F77_TB = C2F_CHAR(&TB);
#endif

        sgemm_blis_impl(F77_TA, F77_TB, &F77_M, &F77_N, &F77_K, &alpha, A,
                &F77_lda, B, &F77_ldb, &beta, C, &F77_ldc);
    } else if (Order == CblasRowMajor) {
        // RowMajorStrg = 1;
        if (TransA == CblasTrans)
            TB = 'T';
        else if (TransA == CblasConjTrans)
            TB = 'C';
        else if (TransA == CblasNoTrans)
            TB = 'N';
        else {
            //cblas_xerbla(2, "cblas_sgemm","Illegal TransA setting, %d\n", TransA);
            return;
        }
        if (TransB == CblasTrans)
            TA = 'T';
        else if (TransB == CblasConjTrans)
            TA = 'C';
        else if (TransB == CblasNoTrans)
            TA = 'N';
        else {
            //cblas_xerbla(2, "cblas_sgemm","Illegal TransB setting, %d\n", TransB);
            return;
        }
#ifdef F77_CHAR
        F77_TA = C2F_CHAR(&TA);
        F77_TB = C2F_CHAR(&TB);
#endif

        sgemm_blis_impl(F77_TA, F77_TB, &F77_N, &F77_M, &F77_K, &alpha, B,
                &F77_ldb, A, &F77_lda, &beta, C, &F77_ldc);
    } else {
        //cblas_xerbla(1, "cblas_sgemm","Illegal Order setting, %d\n", Order);
    }
}

void cblas_sgemv_aocl(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA,
        f77_int M, f77_int N, float alpha, const float *A, f77_int lda,
        const float *X, f77_int incX, float beta, float *Y, f77_int incY) {
    char TA;

#ifdef F77_CHAR
    F77_CHAR F77_TA;
#else
#define F77_TA &TA
#endif

#ifdef F77_INT
    F77_INT F77_M = M, F77_N = N, F77_lda = lda, F77_incX = incX;
    F77_INT F77_incY = incY;
#else
#define F77_M M
#define F77_N N
#define F77_lda lda
#define F77_incX incX
#define F77_incY incY
#endif

    if (Order == CblasColMajor) {
        if (TransA == CblasTrans)
            TA = 'T';
        else if (TransA == CblasConjTrans)
            TA = 'C';
        else if (TransA == CblasNoTrans)
            TA = 'N';
        else {
            //cblas_xerbla(2, "cblas_sgemm", "Illegal TransA setting, %d\n", TransA);
            return;
        }

#ifdef F77_CHAR
        F77_TA = C2F_CHAR(&TA);
#endif

        sgemv_blis_impl(F77_TA, &F77_M, &F77_N, &alpha, A, &F77_lda, X,
                &F77_incX, &beta, Y, &F77_incY);
    } else if (Order == CblasRowMajor) {
        // RowMajorStrg = 1;
        if (TransA == CblasTrans)
            TA = 'N';
        else if (TransA == CblasConjTrans)
            TA = 'N';
        else if (TransA == CblasNoTrans)
            TA = 'T';
        else {
            //cblas_xerbla(2, "cblas_sgemm","Illegal TransA setting, %d\n", TransA);
            return;
        }

#ifdef F77_CHAR
        F77_TA = C2F_CHAR(&TA);
#endif

        sgemv_blis_impl(F77_TA, &F77_N, &F77_M, &alpha, X, &F77_incX, A,
                &F77_lda, &beta, Y, &F77_incY);
    } else {
        //cblas_xerbla(1, "cblas_sgemm","Illegal Order setting, %d\n", Order);
    }
}

float cblas_sdot_aocl(
        f77_int N, const float *X, f77_int incX, const float *Y, f77_int incY) {
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1);
    float dot;

#ifdef F77_INT
    F77_INT F77_N = N, F77_incX = incX, F77_incY = incY;
#else
#define F77_N N
#define F77_incX incX
#define F77_incY incY
#endif

    dot = sdot_blis_impl(&F77_N, X, &F77_incX, Y, &F77_incY);

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);

    return dot;
}
#endif // ZENDNN_USE_AOCL_BLIS_API