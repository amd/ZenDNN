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

#ifndef ZENDNN_THREADPOOL_HPP
#define ZENDNN_THREADPOOL_HPP

#include "zendnn.hpp"
#include "zendnn_threadpool.h"

#include "zendnn_threadpool_iface.hpp"

/// @addtogroup zendnn_api
/// @{

namespace zendnn {

/// @addtogroup zendnn_api_interop
/// @{

/// @addtogroup zendnn_api_threadpool_interop Threadpool interoperability API
/// API extensions to interact with the underlying Threadpool run-time.
/// @{

/// Threadpool interoperability namespace
namespace threadpool_interop {

/// Constructs an execution stream for the specified engine and threadpool.
///
/// @sa @ref dev_guide_threadpool
///
/// @param aengine Engine to create the stream on.
/// @param threadpool Pointer to an instance of a C++ class that implements
///     zendnn::threapdool_iface interface.
/// @returns An execution stream.
inline zendnn::stream make_stream(
        const zendnn::engine &aengine, threadpool_iface *threadpool) {
    zendnn_stream_t c_stream;
    zendnn::error::wrap_c_api(zendnn_threadpool_interop_stream_create(
                                    &c_stream, aengine.get(), threadpool),
            "could not create stream");
    return zendnn::stream(c_stream);
}

/// Returns the pointer to a threadpool that is used by an execution stream.
///
/// @sa @ref dev_guide_threadpool
///
/// @param astream An execution stream.
/// @returns Output pointer to an instance of a C++ class that implements
///     zendnn::threapdool_iface interface or NULL if the stream was created
///     without threadpool.
inline threadpool_iface *get_threadpool(const zendnn::stream &astream) {
    void *tp;
    zendnn::error::wrap_c_api(
            zendnn_threadpool_interop_stream_get_threadpool(astream.get(), &tp),
            "could not get stream threadpool");
    return static_cast<threadpool_iface *>(tp);
}

/// @copydoc zendnn_sgemm_tp()
inline status sgemm(char transa, char transb, zendnn_dim_t M, zendnn_dim_t N,
        zendnn_dim_t K, float alpha, const float *A, zendnn_dim_t lda,
        const float *B, zendnn_dim_t ldb, float beta, float *C, zendnn_dim_t ldc,
        threadpool_iface *tp) {
    return static_cast<status>(zendnn_threadpool_interop_sgemm(
            transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, tp));
}
/// @copydoc zendnn_gemm_u8s8s32_tp()
inline status gemm_u8s8s32(char transa, char transb, char offsetc, zendnn_dim_t M,
        zendnn_dim_t N, zendnn_dim_t K, float alpha, const uint8_t *A,
        zendnn_dim_t lda, uint8_t ao, const int8_t *B, zendnn_dim_t ldb, int8_t bo,
        float beta, int32_t *C, zendnn_dim_t ldc, const int32_t *co,
        threadpool_iface *tp) {
    return static_cast<status>(
            zendnn_threadpool_interop_gemm_u8s8s32(transa, transb, offsetc, M, N,
                    K, alpha, A, lda, ao, B, ldb, bo, beta, C, ldc, co, tp));
}

/// @copydoc zendnn_gemm_s8s8s32_tp()
inline status gemm_s8s8s32(char transa, char transb, char offsetc, zendnn_dim_t M,
        zendnn_dim_t N, zendnn_dim_t K, float alpha, const int8_t *A,
        zendnn_dim_t lda, int8_t ao, const int8_t *B, zendnn_dim_t ldb, int8_t bo,
        float beta, int32_t *C, zendnn_dim_t ldc, const int32_t *co,
        threadpool_iface *tp) {
    return static_cast<status>(
            zendnn_threadpool_interop_gemm_s8s8s32(transa, transb, offsetc, M, N,
                    K, alpha, A, lda, ao, B, ldb, bo, beta, C, ldc, co, tp));
}

} // namespace threadpool_interop

/// @} zendnn_api_threadpool_interop

/// @} zendnn_api_interop

} // namespace zendnn

/// @} zendnn_api

#endif
