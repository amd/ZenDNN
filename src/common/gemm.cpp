/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "zendnn.h"

#if ZENDNN_CPU_RUNTIME == ZENDNN_RUNTIME_THREADPOOL
#include "zendnn_threadpool.hpp"
#include "zendnn_threadpool_iface.hpp"
#endif

#if ZENDNN_CPU_RUNTIME != ZENDNN_RUNTIME_NONE
#include "cpu/gemm/gemm.hpp"
#endif

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"
#include "common/stack_checker.hpp"

using namespace zendnn::impl;

#if ZENDNN_CPU_RUNTIME != ZENDNN_RUNTIME_NONE
namespace {
const char *c2f_offsetC(const char *offC) {
    if (offC) {
        if (offC[0] == 'R' || offC[0] == 'r') return "C";
        if (offC[0] == 'C' || offC[0] == 'c') return "R";
    }
    return offC;
}
} // namespace
#endif

#ifdef ZENDNN_ENABLE_STACK_CHECKER
#define MAYBE_RUN_STACK_CHECKER(api_name, ...) \
    stack_checker::stack_checker_t(#api_name).check(__VA_ARGS__)
#else
#define MAYBE_RUN_STACK_CHECKER(_, func, ...) func(__VA_ARGS__)
#endif

zendnn_status_t zendnn_sgemm(char transa, char transb, dim_t M, dim_t N, dim_t K,
        float alpha, const float *A, dim_t lda, const float *B, const dim_t ldb,
        float beta, float *C, dim_t ldc) {
#if ZENDNN_CPU_RUNTIME != ZENDNN_RUNTIME_NONE
    return MAYBE_RUN_STACK_CHECKER(zendnn_sgemm, cpu::extended_sgemm, &transb,
            &transa, &N, &M, &K, &alpha, B, &ldb, A, &lda, &beta, C, &ldc,
            nullptr, false);
#else
    return zendnn::impl::status::unimplemented;
#endif
}

zendnn_status_t zendnn_gemm_u8s8s32(char transa, char transb, char offsetc, dim_t M,
        dim_t N, dim_t K, float alpha, const uint8_t *A, dim_t lda, uint8_t ao,
        const int8_t *B, dim_t ldb, int8_t bo, float beta, int32_t *C,
        dim_t ldc, const int32_t *co) {
#if ZENDNN_CPU_RUNTIME != ZENDNN_RUNTIME_NONE
    return MAYBE_RUN_STACK_CHECKER(zendnn_gemm_u8s8s32,
            cpu::gemm_s8x8s32<uint8_t>, &transb, &transa, c2f_offsetC(&offsetc),
            &N, &M, &K, &alpha, B, &ldb, &bo, A, &lda, &ao, &beta, C, &ldc, co);
#else
    return zendnn::impl::status::unimplemented;
#endif
}

zendnn_status_t zendnn_gemm_s8s8s32(char transa, char transb, char offsetc, dim_t M,
        dim_t N, dim_t K, float alpha, const int8_t *A, dim_t lda, int8_t ao,
        const int8_t *B, dim_t ldb, int8_t bo, float beta, int32_t *C,
        dim_t ldc, const int32_t *co) {
#if ZENDNN_CPU_RUNTIME != ZENDNN_RUNTIME_NONE
    return MAYBE_RUN_STACK_CHECKER(zendnn_gemm_s8s8s32, cpu::gemm_s8x8s32<int8_t>,
            &transb, &transa, c2f_offsetC(&offsetc), &N, &M, &K, &alpha, B,
            &ldb, &bo, A, &lda, &ao, &beta, C, &ldc, co);
#else
    return zendnn::impl::status::unimplemented;
#endif
}

extern "C" zendnn_status_t ZENDNN_API zendnn_gemm_bf16bf16f32(char transa,
        char transb, dim_t M, dim_t N, dim_t K, float alpha,
        const bfloat16_t *A, dim_t lda, const bfloat16_t *B, dim_t ldb,
        float beta, float *C, dim_t ldc) {
#if ZENDNN_CPU_RUNTIME != ZENDNN_RUNTIME_NONE
    return MAYBE_RUN_STACK_CHECKER(zendnn_gemm_bf16bf16f32, cpu::gemm_bf16bf16f32,
            &transb, &transa, &N, &M, &K, &alpha, B, &ldb, A, &lda, &beta, C,
            &ldc);
#else
    return zendnn::impl::status::unimplemented;
#endif
}

#if ZENDNN_CPU_RUNTIME == ZENDNN_RUNTIME_THREADPOOL
zendnn_status_t zendnn_threadpool_interop_sgemm(char transa, char transb, dim_t M,
        dim_t N, dim_t K, float alpha, const float *A, dim_t lda,
        const float *B, const dim_t ldb, float beta, float *C, dim_t ldc,
        void *th) {
    threadpool_utils::activate_threadpool(
            (zendnn::threadpool_interop::threadpool_iface *)th);
    status_t status = MAYBE_RUN_STACK_CHECKER(zendnn_threadpool_interop_sgemm,
            cpu::extended_sgemm, &transb, &transa, &N, &M, &K, &alpha, B, &ldb,
            A, &lda, &beta, C, &ldc, nullptr, false);
    threadpool_utils::deactivate_threadpool();
    return status;
}

zendnn_status_t zendnn_threadpool_interop_gemm_u8s8s32(char transa, char transb,
        char offsetc, dim_t M, dim_t N, dim_t K, float alpha, const uint8_t *A,
        dim_t lda, uint8_t ao, const int8_t *B, dim_t ldb, int8_t bo,
        float beta, int32_t *C, dim_t ldc, const int32_t *co, void *th) {
    threadpool_utils::activate_threadpool(
            (zendnn::threadpool_interop::threadpool_iface *)th);
    status_t status = MAYBE_RUN_STACK_CHECKER(
            zendnn_threadpool_interop_gemm_u8s8s32, cpu::gemm_s8x8s32<uint8_t>,
            &transb, &transa, c2f_offsetC(&offsetc), &N, &M, &K, &alpha, B,
            &ldb, &bo, A, &lda, &ao, &beta, C, &ldc, co);
    threadpool_utils::deactivate_threadpool();
    return status;
}

zendnn_status_t zendnn_threadpool_interop_gemm_s8s8s32(char transa, char transb,
        char offsetc, dim_t M, dim_t N, dim_t K, float alpha, const int8_t *A,
        dim_t lda, int8_t ao, const int8_t *B, dim_t ldb, int8_t bo, float beta,
        int32_t *C, dim_t ldc, const int32_t *co, void *th) {
    threadpool_utils::activate_threadpool(
            (zendnn::threadpool_interop::threadpool_iface *)th);
    status_t status = MAYBE_RUN_STACK_CHECKER(
            zendnn_threadpool_interop_gemm_s8s8s32, cpu::gemm_s8x8s32<int8_t>,
            &transb, &transa, c2f_offsetC(&offsetc), &N, &M, &K, &alpha, B,
            &ldb, &bo, A, &lda, &ao, &beta, C, &ldc, co);
    threadpool_utils::deactivate_threadpool();
    return status;
}

extern "C" zendnn_status_t ZENDNN_API zendnn_threadpool_interop_gemm_bf16bf16f32(
        char transa, char transb, dim_t M, dim_t N, dim_t K, float alpha,
        const bfloat16_t *A, dim_t lda, const bfloat16_t *B, dim_t ldb,
        float beta, float *C, dim_t ldc, void *th) {
    threadpool_utils::activate_threadpool(
            (zendnn::threadpool_interop::threadpool_iface *)th);
    status_t status
            = MAYBE_RUN_STACK_CHECKER(zendnn_threadpool_interop_gemm_bf16bf16f32,
                    cpu::gemm_bf16bf16f32, &transb, &transa, &N, &M, &K, &alpha,
                    B, &ldb, A, &lda, &beta, C, &ldc);
    threadpool_utils::deactivate_threadpool();
    return status;
}
#endif
