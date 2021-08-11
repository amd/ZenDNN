/*******************************************************************************
* Modifications Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef ZENDNN_THREADPOOL_H
#define ZENDNN_THREADPOOL_H

#include "zendnn_config.h"
#include "zendnn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/// @addtogroup zendnn_api
/// @{

/// @addtogroup zendnn_api_interop
/// @{

/// @addtogroup zendnn_api_threadpool_interop
/// @{

/// Creates an execution stream with specified threadpool.
///
/// @sa @ref dev_guide_threadpool
///
/// @param stream Output execution stream.
/// @param engine Engine to create the execution stream on.
/// @param threadpool Pointer to an instance of a C++ class that implements
///     zendnn::threapdool_iface interface.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_threadpool_interop_stream_create(
        zendnn_stream_t *stream, zendnn_engine_t engine, void *threadpool);

/// Returns a threadpool to be used by the execution stream.
///
/// @sa @ref dev_guide_threadpool
///
/// @param astream Execution stream.
/// @param threadpool Output pointer to an instance of a C++ class that
///     implements zendnn::threapdool_iface interface. Set to NULL if the
///     stream was created without threadpool.
/// @returns #zendnn_success on success and a status describing the error
///     otherwise.
zendnn_status_t ZENDNN_API zendnn_threadpool_interop_stream_get_threadpool(
        zendnn_stream_t astream, void **threadpool);

/// @copydoc zendnn_sgemm()
/// @param threadpool A pointer to a threadpool interface (only when built with
///     the THREADPOOL CPU runtime).
zendnn_status_t ZENDNN_API zendnn_threadpool_interop_sgemm(char transa, char transb,
        zendnn_dim_t M, zendnn_dim_t N, zendnn_dim_t K, float alpha, const float *A,
        zendnn_dim_t lda, const float *B, zendnn_dim_t ldb, float beta, float *C,
        zendnn_dim_t ldc, void *threadpool);

/// @copydoc zendnn_gemm_u8s8s32()
/// @param threadpool A pointer to a threadpool interface (only when built with
///     the THREADPOOL CPU runtime).
zendnn_status_t ZENDNN_API zendnn_threadpool_interop_gemm_u8s8s32(char transa,
        char transb, char offsetc, zendnn_dim_t M, zendnn_dim_t N, zendnn_dim_t K,
        float alpha, const uint8_t *A, zendnn_dim_t lda, uint8_t ao,
        const int8_t *B, zendnn_dim_t ldb, int8_t bo, float beta, int32_t *C,
        zendnn_dim_t ldc, const int32_t *co, void *threadpool);

/// @copydoc zendnn_gemm_s8s8s32()
/// @param threadpool A pointer to a threadpool interface (only when built with
///     the THREADPOOL CPU runtime).
zendnn_status_t ZENDNN_API zendnn_threadpool_interop_gemm_s8s8s32(char transa,
        char transb, char offsetc, zendnn_dim_t M, zendnn_dim_t N, zendnn_dim_t K,
        float alpha, const int8_t *A, zendnn_dim_t lda, int8_t ao,
        const int8_t *B, zendnn_dim_t ldb, int8_t bo, float beta, int32_t *C,
        zendnn_dim_t ldc, const int32_t *co, void *threadpool);

/// @} zendnn_api_threadpool_interop

/// @} zendnn_api_interop

/// @} zendnn_api

#ifdef __cplusplus
}
#endif

#endif
