/*******************************************************************************
* Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Copyright 2016-2022 Intel Corporation
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

#ifndef ZENDNN_REORDER_CACHE_HPP
#define ZENDNN_REORDER_CACHE_HPP

#include "common/zendnn_lru_cache.hpp"

// Define a function pointer type for getting the reorder buffer size.
using GetReorderBufSizeFunc = siz_t (*)(const char, const char, const char,
                                        const dim_t, const dim_t);

// Define a template function pointer type for reordering.
template <typename T>
using ReorderFunc = void (*)(const char, const char, const char, const T *, T *,
                             const dim_t, const dim_t, const dim_t);
void aocl_unreorder(const int8_t *wei, int8_t *plain_buff, int k, int n,
                    int ldb);

template <typename T>
bool reorderAndCacheWeights(
    const Key_matmul &key_obj,
    const T *filter,
    T *&reorder_filter,
    const int k,
    const int n,
    const int ldb,
    const bool is_weights_const,
    const bool is_inplace,
    const char order,
    const char trans,
    const char reorder_param0,
    const dim_t reorder_param1,
    const dim_t reorder_param2,
    GetReorderBufSizeFunc get_reorder_buf_size,
    ReorderFunc<T> reorder_func,
    int weight_cache_type = zendnnWeightCacheType::WEIGHT_CACHE_OUT_OF_PLACE,
    int src_zp = 0
);

bool reorderAndCacheWeightsBrgemm(
    const Key_matmul &key_obj_reorder,
    const zendnn::memory::desc &weight_disc,
    zendnn::memory &user_weights_memory,
    zendnn::memory &reordered_weights_memory,
    zendnn::engine &eng,
    zendnn::stream &engine_stream,
    bool is_weights_const,
    bool is_inplace,
    int weight_cache_type = zendnnWeightCacheType::WEIGHT_CACHE_OUT_OF_PLACE
);

template <typename T>
void woqReorderAndCacheWeightsAocl(
    const Key_matmul &key_obj,
    const int8_t *filter,
    T *&reorder_filter,
    const int k,
    const int n,
    const int ldb,
    const bool is_weights_const,
    const char order,
    const char trans,
    const char reorder_param0,
    const dim_t reorder_param1,
    const dim_t reorder_param2,
    GetReorderBufSizeFunc get_reorder_buf_size,
    ReorderFunc<T> reorder_func,
    int weights_type,
    float *wei_scale,
    int scale_size,
    int group_size,
    zendnn_data_type_t scale_dt
);

void woqReorderAndCacheWeightsBrgemm(
    const Key_matmul &key_obj_reorder,
    const zendnn::matmul::primitive_desc &matmul_prim_disc,
    zendnn::memory &user_weights_memory,
    zendnn::memory &reordered_weights_memory,
    zendnn::engine &eng,
    zendnn::stream &engine_stream,
    zendnn::memory::desc &matmul_weights_md,
    bool is_weights_const,
    int8_t *weights,
    int weights_type,
    const int K,
    const int N,
    float *wei_scale,
    int scale_size,
    int group_size,
    zendnn_data_type_t scale_dt
);

void cacheStaticScales(
    zendnn::zendnnEnv zenEnvObj,
    const Key_matmul &key_obj,
    float *&new_scale,
    float *src_scale,
    float *wei_scale,
    float *dst_scale,
    int src_scale_size,
    int wei_scale_size,
    int dst_scale_size,
    int scale_type
);

void cacheZeroPointCompensation(
    zendnn::zendnnEnv zenEnvObj,
    const Key_matmul &key_obj,
    int M,
    int N,
    int K,
    const char *src,
    int src_s0,
    int src_s1,
    const int8_t *wei,
    int wei_s0,
    int wei_s1,
    int32_t *&acc,
    int ldc,
    int32_t src_zero_point,
    int32_t wei_zero_point,
    bool blocked_format,
    bool is_weights_const,
    const bool is_inplace,
    int algo,
    int weight_cache_type = zendnnWeightCacheType::WEIGHT_CACHE_OUT_OF_PLACE,
    zendnn::engine eng = engine(),
    zendnn::stream engine_stream = stream()
);

void cacheScaledBias(
    zendnn::zendnnEnv zenEnvObj,
    const Key_matmul &key_obj,
    zendnn::stream engine_stream,
    zendnn::engine eng,
    zendnn::memory::desc bias_desc,
    char *&new_bias,
    char *bias,
    int n,
    float *src_scale,
    float *wei_scale,
    int src_scale_size,
    int wei_scale_size
);

#endif // ZENDNN_REORDER_CACHE_HPP
