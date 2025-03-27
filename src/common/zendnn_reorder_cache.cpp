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

#include "zendnn_reorder_cache.hpp"
#include <mutex>

extern std::mutex map_mutex;
using tag = memory::format_tag;
using dt = memory::data_type;

// This function reorders and caches weights for matrix multiplication operations. It uses a cache to store
// reordered weights to optimize performance for repeated operations with the same parameters. The function
// supports different data types through templates and uses function pointers to perform the reordering.

// Template Parameter:
// T - The data type of the weights (e.g., int8_t, float, int16_t).
template <typename T>
bool reorderAndCacheWeights(
    const Key_matmul &key_obj,
    const T *weights,
    T *&reorder_weights,
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
    int weight_cache_type
) {
    // Weight caching
    static zendnn::impl::lru_weight_cache_t<Key_matmul, T *> matmul_weight_cache;
    auto found_obj = matmul_weight_cache.find_key(key_obj);

    T *c_wei = const_cast<T *>(weights);
    if (weight_cache_type == zendnnWeightCacheType::WEIGHT_CACHE_DISABLE) {
        zendnnVerbose(ZENDNN_PROFLOG,"BLIS reorder weights (WEIGHT_CACHE_DISABLE)");
        siz_t b_reorder_buf_siz_req = get_reorder_buf_size(order, trans, reorder_param0,
                                      reorder_param1, reorder_param2);
        reorder_weights = (T *)zendnn_aligned_alloc(64, b_reorder_buf_siz_req);
        reorder_func(order, trans, 'B', weights, reorder_weights, k, n, ldb);
    }
    // Out-of-place reordering
    else if (weight_cache_type ==
             zendnnWeightCacheType::WEIGHT_CACHE_OUT_OF_PLACE) {
        if (!found_obj) {
            zendnnVerbose(ZENDNN_PROFLOG,"BLIS reorder weights WEIGHT_CACHE_OUT_OF_PLACE");
            siz_t b_reorder_buf_siz_req = get_reorder_buf_size(order, trans, reorder_param0,
                                          reorder_param1, reorder_param2);
            reorder_weights = (T *)zendnn_aligned_alloc(64, b_reorder_buf_siz_req);
            reorder_func(order, trans, 'B', weights, reorder_weights, k, n, ldb);
            if (is_weights_const) {
                // Create new entry
                map_mutex.lock();
                matmul_weight_cache.add(key_obj, reorder_weights);
                map_mutex.unlock();
            }
        }
        else {
            zendnnVerbose(ZENDNN_PROFLOG,
                          "Read BLIS cached weights WEIGHT_CACHE_OUT_OF_PLACE");
            reorder_weights = matmul_weight_cache.get(key_obj);
        }
    }
    // In-place reordering
    else if (weight_cache_type == zendnnWeightCacheType::WEIGHT_CACHE_INPLACE) {
        if (!found_obj) {
            siz_t b_reorder_buf_siz_req = get_reorder_buf_size(order, trans, reorder_param0,
                                          reorder_param1, reorder_param2);
            // TODO: Implement scratchpad memory or memory pool
            reorder_weights = (T *)zendnn_aligned_alloc(64, b_reorder_buf_siz_req);
            reorder_func(order, trans, 'B', weights, reorder_weights, k, n, ldb);
            map_mutex.lock();
            if (is_weights_const) {
                if (b_reorder_buf_siz_req == k*n*sizeof(T)) {
                    zendnnVerbose(ZENDNN_PROFLOG,"BLIS reorder weights WEIGHT_CACHE_INPLACE");
                    #pragma omp parallel for
                    for (long long int idx = 0; idx < b_reorder_buf_siz_req/sizeof(T); idx++) {
                        c_wei[idx] = reorder_weights[idx];
                    }
                    //Free the allocated memory
                    T *dummy = NULL;
                    matmul_weight_cache.add(key_obj, dummy);
                    free(reorder_weights);
                    reorder_weights = c_wei;
                }
                else {
                    zendnnVerbose(ZENDNN_PROFLOG,
                                  "BLIS reorder weights WEIGHT_CACHE_INPLACE new memory");
                    matmul_weight_cache.add(key_obj, reorder_weights);
                }
            }
            map_mutex.unlock();
        }
        else {
            reorder_weights = matmul_weight_cache.get(key_obj);
            if (reorder_weights == NULL) {
                reorder_weights = c_wei;
            }
            zendnnVerbose(ZENDNN_PROFLOG,"Read BLIS cached weights WEIGHT_CACHE_INPLACE");
        }
    }
    // Multi-instance read (memory not expanded)
    else if (weight_cache_type == zendnnWeightCacheType::WEIGHT_CACHE_AOT_INPLACE) {
        siz_t b_reorder_buf_siz_req = get_reorder_buf_size(order, trans, reorder_param0,
                                      reorder_param1, reorder_param2);
        if (b_reorder_buf_siz_req != k*n*sizeof(T)) {
            return false;
        }
        zendnnVerbose(ZENDNN_PROFLOG,
                      "Read BLIS cached weights WEIGHT_CACHE_AOT_INPLACE");
        reorder_weights = c_wei;
    }
    // Multi-instance read (memory expanded explicitly)
    else if (weight_cache_type ==
             zendnnWeightCacheType::WEIGHT_CACHE_AOT_RESIZED_INPLACE) {
        // Assuming weights are already reordered using external API.
        zendnnVerbose(ZENDNN_PROFLOG,
                      "Read BLIS cached weights WEIGHT_CACHE_AOT_RESIZED_INPLACE");
        reorder_weights = c_wei;
    }
    else if (weight_cache_type == zendnnWeightCacheType::WEIGHT_CACHE_AOT_REORDER) {
        // Perform reordering (used by Custom Reorder API)
        // This path can't be enabled by ENV
        if (!found_obj) {
            zendnnVerbose(ZENDNN_PROFLOG,"BLIS reorder weights WEIGHT_CACHE_AOT_REORDER");
            siz_t b_reorder_buf_siz_req = get_reorder_buf_size(order, trans, reorder_param0,
                                          reorder_param1, reorder_param2);
            // TODO: Implement scratchpad memory or memory pool
            reorder_weights = (T *)zendnn_aligned_alloc(64, b_reorder_buf_siz_req);
            reorder_func(order, trans, 'B', weights, reorder_weights, k, n, ldb);

            map_mutex.lock();
            #pragma omp parallel for
            for (int idx = 0; idx < b_reorder_buf_siz_req; idx++) {
                c_wei[idx] = reorder_weights[idx];
            }
            //Free the allocated memory
            free(reorder_weights);
            T *dummy = NULL;
            matmul_weight_cache.add(key_obj, dummy);
            map_mutex.unlock();
        }
    }
    return true;
}

bool reorderAndCacheWeightsBrgemm(
    const Key_matmul &key_obj_reorder,
    const zendnn::memory::desc &weight_disc,
    zendnn::memory &user_weights_memory,
    zendnn::memory &reordered_weights_memory,
    zendnn::engine &eng,
    zendnn::stream &engine_stream,
    bool is_weights_const,
    int weight_cache_type
) {
    //weight caching
    static zendnn::impl::lru_weight_cache_t<Key_matmul, zendnn::memory>
    matmul_weight_cache;
    auto found_obj_reorder = matmul_weight_cache.find_key(key_obj_reorder);

    if (weight_cache_type == zendnnWeightCacheType::WEIGHT_CACHE_DISABLE) {
        zendnnVerbose(ZENDNN_PROFLOG,"BRGEMM reorder weights WEIGHT_CACHE_DISABLE");
        reordered_weights_memory = memory(weight_disc, eng);
        reorder(user_weights_memory, reordered_weights_memory).execute(engine_stream,
                user_weights_memory, reordered_weights_memory);
    }
    // Out-of-place reordering
    else if (weight_cache_type ==
             zendnnWeightCacheType::WEIGHT_CACHE_OUT_OF_PLACE) {
        if (!found_obj_reorder) {
            zendnnVerbose(ZENDNN_PROFLOG,
                          "BRGEMM reorder weights WEIGHT_CACHE_OUT_OF_PLACE");
            reordered_weights_memory = memory(weight_disc, eng);
            reorder(user_weights_memory, reordered_weights_memory).execute(engine_stream,
                    user_weights_memory, reordered_weights_memory);
            if (is_weights_const) {
                //Save in map
                map_mutex.lock();
                matmul_weight_cache.add(key_obj_reorder, reordered_weights_memory);
                map_mutex.unlock();
            }
        }
        else {
            zendnnVerbose(ZENDNN_PROFLOG,
                          "Read cached BRGEMM weights WEIGHT_CACHE_OUT_OF_PLACE");
            reordered_weights_memory = matmul_weight_cache.get(key_obj_reorder);
        }
    }
    // In-place reordering
    else if (weight_cache_type == zendnnWeightCacheType::WEIGHT_CACHE_INPLACE) {
        if (!found_obj_reorder) {
            reordered_weights_memory = memory(weight_disc, eng);
            reorder(user_weights_memory, reordered_weights_memory).execute(engine_stream,
                    user_weights_memory, reordered_weights_memory);
            //Save in map
            map_mutex.lock();
            if (is_weights_const) {
                int8_t *reorder_ptr = (int8_t *)reordered_weights_memory.get_data_handle();
                int8_t *user_ptr = (int8_t *)user_weights_memory.get_data_handle();
                if (user_weights_memory.get_desc().get_size() ==
                        weight_disc.get_size()) {
                    zendnnVerbose(ZENDNN_PROFLOG,"BRGEMM reorder weights WEIGHT_CACHE_INPLACE");
                    #pragma omp parallel for
                    for (long long int idx = 0; idx < weight_disc.get_size(); idx++) {
                        user_ptr[idx] = reorder_ptr[idx];
                    }
                    zendnn::memory dummy_mem;
                    matmul_weight_cache.add(key_obj_reorder, dummy_mem);
                }
                else {
                    zendnnVerbose(ZENDNN_PROFLOG,
                                  "BRGEMM reorder weights WEIGHT_CACHE_INPLACE new memory");
                    matmul_weight_cache.add(key_obj_reorder, reordered_weights_memory);
                }
            }
            map_mutex.unlock();
        }
        else {
            zendnnVerbose(ZENDNN_PROFLOG,"Read cached BRGEMM weights WEIGHT_CACHE_INPLACE");
            zendnn::memory dummy_mem;
            reordered_weights_memory = matmul_weight_cache.get(key_obj_reorder);
            if (weight_disc.get_size() == user_weights_memory.get_desc().get_size()) {
                reordered_weights_memory = user_weights_memory;
            }
        }
    }
    // Multi-instance read (memory not expanded)
    else if (weight_cache_type ==zendnnWeightCacheType::WEIGHT_CACHE_AOT_INPLACE) {
        // Check if memory equal to required reorder buffer size
        if (user_weights_memory.get_desc().get_size() != weight_disc.get_size()) {
            return false;
        }
        zendnnVerbose(ZENDNN_PROFLOG,
                      "Read cached BRGEMM weights WEIGHT_CACHE_AOT_INPLACE");
        reordered_weights_memory = user_weights_memory;
    }
    // Multi-instance read (memory expanded explicitly)
    else if (weight_cache_type ==
             zendnnWeightCacheType::WEIGHT_CACHE_AOT_RESIZED_INPLACE) {
        // Assuming weight buffers are reordered using external API
        zendnnVerbose(ZENDNN_PROFLOG,
                      "Read cached BRGEMM weights WEIGHT_CACHE_AOT_RESIZED_INPLACE");
        reordered_weights_memory = user_weights_memory;
    }
    else if (weight_cache_type == zendnnWeightCacheType::WEIGHT_CACHE_AOT_REORDER) {
        // Perform reordering (used by Custom Reorder API)
        // This path can't be enabled by ENV
        if (!found_obj_reorder) {
            zendnnVerbose(ZENDNN_PROFLOG,"BRGEMM reorder weights WEIGHT_CACHE_AOT_REORDER");
            reordered_weights_memory = memory(weight_disc, eng);
            reorder(user_weights_memory, reordered_weights_memory).execute(engine_stream,
                    user_weights_memory, reordered_weights_memory);
            //Save in map
            map_mutex.lock();
            int8_t *reorder_ptr = (int8_t *)reordered_weights_memory.get_data_handle();
            int8_t *user_ptr = (int8_t *)user_weights_memory.get_data_handle();
            #pragma omp parallel for
            for (int idx = 0; idx < weight_disc.get_size(); idx++) {
                user_ptr[idx] = reorder_ptr[idx];
            }
            zendnn::memory dummy_mem;
            matmul_weight_cache.add(key_obj_reorder, dummy_mem);
            map_mutex.unlock();
            reordered_weights_memory = user_weights_memory;
        }
    }

    return true;
}

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
) {
    // Weight caching
    static zendnn::impl::lru_weight_cache_t<Key_matmul, T *> matmul_weight_cache;
    auto found_obj = matmul_weight_cache.find_key(key_obj);

    if (!is_weights_const || !found_obj) {
        zendnnVerbose(ZENDNN_PROFLOG,"BLIS reorder weights");
        T *wei_type = (T *)zendnn_aligned_alloc(64, sizeof(T)*k*n);
        if constexpr(std::is_same<T, int16_t>::value) {
            if (weights_type == zendnn_s4) { //Convert S4 to BF16
                cvt_int4_to_bf16(filter, wei_type, k, n, wei_scale, scale_size, group_size,
                                 scale_dt);
            }
            else { //Convert S8 to BF16
                cvt_int8_to_bf16(filter, wei_type, k, n, wei_scale, scale_size, group_size,
                                 scale_dt);
            }
        }
        else if constexpr(std::is_same<T, float>::value) {
            if (weights_type == zendnn_s4) { //Convert S4 to FP32
                cvt_int4_to_f32(filter, wei_type, k, n, wei_scale, scale_size, group_size,
                                scale_dt);
            }
            else { //Convert S8 to BF16
                cvt_int8_to_f32(filter, wei_type, k, n, wei_scale, scale_size, group_size,
                                scale_dt);
            }
        }
        siz_t b_reorder_buf_siz_req = get_reorder_buf_size(order, trans, reorder_param0,
                                      reorder_param1, reorder_param2);
        reorder_filter = (T *)zendnn_aligned_alloc(64, b_reorder_buf_siz_req);
        reorder_func(order, trans, 'B', wei_type, reorder_filter, k, n, ldb);
        free(wei_type);
        if (is_weights_const) {
            // Create new entry
            map_mutex.lock();
            matmul_weight_cache.add(key_obj, reorder_filter);
            map_mutex.unlock();
        }
    }
    else {
        zendnnVerbose(ZENDNN_PROFLOG,"Read BLIS cached weights");
        reorder_filter = matmul_weight_cache.get(key_obj);
    }
}

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
) {
    //weight caching
    static zendnn::impl::lru_weight_cache_t<Key_matmul, zendnn::memory>
    matmul_weight_cache;
    auto found_obj_reorder = matmul_weight_cache.find_key(key_obj_reorder);

    if (!is_weights_const || !found_obj_reorder) {
        zendnnVerbose(ZENDNN_PROFLOG,"BRGEMM reorder weights");
        int16_t *wei_bf16 = (int16_t *)zendnn_aligned_alloc(64, sizeof(int16_t)*K*N);

        if (weights_type == zendnn_s4) { //Convert S4 to BF16
            cvt_int4_to_bf16(weights, wei_bf16, K, N, wei_scale, scale_size, group_size,
                             scale_dt);
        }
        else { //Convert S8 to BF16
            cvt_int8_to_bf16(weights, wei_bf16, K, N, wei_scale, scale_size, group_size,
                             scale_dt);
        }
        user_weights_memory = memory(matmul_weights_md, eng, wei_bf16);
        reordered_weights_memory = memory(matmul_prim_disc.weights_desc(), eng);
        reorder(user_weights_memory, reordered_weights_memory).execute(engine_stream,
                user_weights_memory, reordered_weights_memory);
        if (is_weights_const) {
            //Save in map
            map_mutex.lock();
            matmul_weight_cache.add(key_obj_reorder, reordered_weights_memory);
            map_mutex.unlock();
        }
        free(wei_bf16);
    }
    else {
        reordered_weights_memory = matmul_weight_cache.get(key_obj_reorder);
    }
}

void aocl_unreorder(const int8_t *wei, int8_t *plain_buff, int k, int n,
                    int ldb) {
    // TODO: Support 1st arg in better way
    aocl_unreorder_s8s8s32os32_reference(ldb == k ? 'c' : 'r', 'B', wei, plain_buff,
                                         k, n, ldb);
}

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
    int algo,
    int weight_cache_type,
    zendnn::engine eng,
    zendnn::stream engine_stream
) {
    // Used to cache compensation when only src_scales exist.
    static zendnn::impl::lru_weight_cache_t<Key_matmul, int32_t *>
    matmul_src_zp_wei_comp_cache;

    bool is_unreorder_wei_req = false;
    if (is_weights_const &&
            weight_cache_type > zendnnWeightCacheType::WEIGHT_CACHE_INPLACE &&
            blocked_format) {
        is_unreorder_wei_req = true;
    }

    zendnn::memory reordered_weight_memory, un_reordered_weight_memory;
    zendnn::memory::desc matmul_weights_md, blocked_matmul_weights_md;
    memory::dims b_strides;

    bool enable_cache = zenEnvObj.zenZpCompCache;
    bool aocl_enable = (algo == 1 || algo == 3);
    bool brgemm_enable = (algo == 2 || algo == 4);

    if (brgemm_enable) {
        b_strides = memory::dims {wei_s0, wei_s1};
        matmul_weights_md = memory::desc({{K, N}}, dt::s8, b_strides);
        blocked_matmul_weights_md = memory::desc({{K, N}}, dt::s8, tag::BA16a64b4a);
    }

    if (!wei_zero_point && !src_zero_point) {
        return;
    }
    else if (!wei_zero_point) {
        auto found_obj = matmul_src_zp_wei_comp_cache.find_key(key_obj);
        if (!found_obj) {
            // acc is freed by lru map
            acc = (int32_t *)zendnn_aligned_alloc(64, sizeof(int32_t)*N);
            std::vector<int32_t> wei_comp(N,0);

            int8_t *filt_array = const_cast<int8_t *>(wei);
            if (is_unreorder_wei_req) {
                if (aocl_enable) {
                    filt_array = (int8_t *)zendnn_aligned_alloc(64, sizeof(int8_t)*K*N);
                    aocl_unreorder((const int8_t *)wei, filt_array, K, N,
                                   wei_s0 > wei_s1 ? wei_s0 : wei_s1);
                }
                else {
                    reordered_weight_memory = memory(blocked_matmul_weights_md, eng, filt_array);
                    un_reordered_weight_memory = memory(matmul_weights_md, eng);
                    reorder(reordered_weight_memory,
                            un_reordered_weight_memory).execute(engine_stream,
                                    reordered_weight_memory, un_reordered_weight_memory);
                    filt_array = (int8_t *)un_reordered_weight_memory.get_data_handle();
                }
            }

            for (dim_t k = 0; k < K; ++k) {
                for (dim_t n = 0; n < N; ++n) {
                    if (k == 0) {
                        wei_comp[n] = int32_t(0);
                    }
                    wei_comp[n] += filt_array[wei_s0 * k + wei_s1 * n];
                }
            }

            if (is_unreorder_wei_req && aocl_enable) {
                free(filt_array);
            }
            for (dim_t n = 0; n < N; ++n) {
                acc[n] = 0 - src_zero_point * wei_comp[n];
            }
            if (enable_cache) {
                zendnnVerbose(ZENDNN_PROFLOG,
                              "Cache Zero-point(src and wei) as 1D compensation");
                map_mutex.lock();
                matmul_src_zp_wei_comp_cache.add(key_obj, acc);
                map_mutex.unlock();
            }
        }
        else {
            zendnnVerbose(ZENDNN_PROFLOG,
                          "Read cached Zero-point(src and wei) as 1D compensation");
            acc = matmul_src_zp_wei_comp_cache.get(key_obj);
        }
    }
    else if (!src_zero_point) {
        std::vector<int32_t> src_comp(M,0);
        // acc is freed in main function for wei_zp != 0
        acc = (int32_t *)zendnn_aligned_alloc(64, sizeof(int32_t)*M*N);

        for (dim_t m = 0; m < M; ++m) {
            for (dim_t k = 0; k < K; ++k) {
                if (k == 0) {
                    src_comp[m] = int32_t(0);
                }
                src_comp[m] += src[src_s0 * m + src_s1 * k];
            }
        }

        for (dim_t m = 0; m < M; ++m) {
            for (dim_t n = 0; n < N; ++n) {
                acc[m * ldc + n] = 0 - wei_zero_point * src_comp[m];
            }
        }
    }
    else {
        std::vector<int32_t> src_comp(M,0);
        std::vector<int32_t> wei_comp(N,0);
        // acc is freed in main function for wei_zp != 0
        acc = (int32_t *)zendnn_aligned_alloc(64, sizeof(int32_t)*M*N);
        //Src comp
        for (dim_t m = 0; m < M; ++m) {
            for (dim_t k = 0; k < K; ++k) {
                if (k == 0) {
                    src_comp[m] = int32_t(0);
                }
                src_comp[m] += src[src_s0 * m + src_s1 * k];
            }
        }

        //Wei comp
        int8_t *filt_array = const_cast<int8_t *>(wei);
        if (is_unreorder_wei_req) {
            if (aocl_enable) {
                filt_array = (int8_t *)zendnn_aligned_alloc(64, sizeof(int8_t)*K*N);
                aocl_unreorder(wei, filt_array, K, N, wei_s0 > wei_s1 ? wei_s0 : wei_s1);
            }
            else {
                reordered_weight_memory = memory(blocked_matmul_weights_md, eng, filt_array);
                un_reordered_weight_memory = memory(matmul_weights_md, eng);
                reorder(reordered_weight_memory,
                        un_reordered_weight_memory).execute(engine_stream,
                                reordered_weight_memory, un_reordered_weight_memory);
                filt_array = (int8_t *)un_reordered_weight_memory.get_data_handle();
            }
        }

        for (dim_t k = 0; k < K; ++k) {
            for (dim_t n = 0; n < N; ++n) {
                if (k == 0) {
                    wei_comp[n] = int32_t(0);
                }
                wei_comp[n] += filt_array[wei_s0 * k + wei_s1 * n];
            }
        }
        if (is_unreorder_wei_req && aocl_enable) {
            free(filt_array);
        }

        for (dim_t m = 0; m < M; ++m) {
            for (dim_t n = 0; n < N; ++n) {
                acc[m * ldc + n] = 0 - src_zero_point * wei_comp[n]
                                   - wei_zero_point * src_comp[m]
                                   + src_zero_point * wei_zero_point * (int)K;
            }
        }
    }
}

// Fucntion to cache BIAS/(src_scale*wei_scale)
// TODO: get proper size of bias
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
) {
    // Bias caching
    static zendnn::impl::lru_weight_cache_t<Key_matmul, char *>
    matmul_bias_cache;
    auto found_obj = matmul_bias_cache.find_key(key_obj);

    // enable cache = 1 (cache the product).
    // enable_cache = 0 (do product always)
    bool enable_cache = zenEnvObj.zenBiasCache;
    // Assuming wei scales size is larger or equal to src scale size.
    if (!found_obj) {
        std::vector<float> reordered_bias_scales(wei_scale_size);
        #pragma omp parallel for
        for (int i = 0; i < wei_scale_size; i++) {
            reordered_bias_scales[i] = 1 / (wei_scale[i]*src_scale[i%src_scale_size]);
        }

        auto bias_memory = memory(bias_desc, eng, bias);
        size_t size_mem = bias_memory.get_desc().get_size();
        new_bias = (char *)zendnn_aligned_alloc(64, size_mem);
        auto reordered_bias_memory = memory(bias_desc, eng, new_bias);
        primitive_attr bias_attr;
        bias_attr.set_output_scales(wei_scale_size == 1 ? 0 : (1<<1),
                                    reordered_bias_scales);
        reorder(bias_memory, reordered_bias_memory, bias_attr).execute(engine_stream,
                bias_memory, reordered_bias_memory);
        if (enable_cache) {
            zendnnVerbose(ZENDNN_PROFLOG,"Cache Scaled Bias");
            map_mutex.lock();
            matmul_bias_cache.add(key_obj, new_bias);
            map_mutex.unlock();
        }
    }
    else {
        zendnnVerbose(ZENDNN_PROFLOG,"Read cached Scaled Bias");
        new_bias = matmul_bias_cache.get(key_obj);
    }

    return;
}

// Current support is for wei_scale*src_scale*dst_scale
// TODO: consider all scale sizes to access data.
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
) {
    // Static scale caching
    static zendnn::impl::lru_weight_cache_t<Key_matmul, float *>
    matmul_SrcWei_scales_cache;
    auto found_obj = matmul_SrcWei_scales_cache.find_key(key_obj);

    bool enable_cache = zenEnvObj.zenStaticScaleCache;
    // Assuming wei scales size is larger or equal to src scale size.
    // enable cache = 1 (don't do product again if already cached).
    // enable_cache = 0 (do product always)
    if (!found_obj || !enable_cache) {
        new_scale = (float *)zendnn_aligned_alloc(64, sizeof(float)*wei_scale_size);
        if (scale_type == zendnn_f32) {
            if (dst_scale != NULL) {
                #pragma omp parallel for
                for (int i = 0; i < wei_scale_size; i++) {
                    new_scale[i] = wei_scale[i] * src_scale[0] * dst_scale[0];
                }
            }
            else {
                #pragma omp parallel for
                for (int i = 0; i < wei_scale_size; i++) {
                    new_scale[i] = wei_scale[i] * src_scale[0];
                }
            }
        }
        else if (scale_type == zendnn_bf16) {
            float src_scale_f = zendnn::impl::cpu::io::load_float_value((zendnn_bf16),
                                (void *)src_scale, 0);
            if (dst_scale != NULL) {
                float dst_scale_f = zendnn::impl::cpu::io::load_float_value((zendnn_bf16),
                                    (void *)dst_scale, 0);
                #pragma omp parallel for
                for (int idx = 0; idx < wei_scale_size; idx++) {
                    new_scale[idx] = zendnn::impl::cpu::io::load_float_value((zendnn_bf16),
                                     (void *)wei_scale, idx) *
                                     src_scale_f * dst_scale_f;
                }
            }
            else {
                #pragma omp parallel for
                for (int idx = 0; idx < wei_scale_size; idx++) {
                    new_scale[idx] = zendnn::impl::cpu::io::load_float_value((zendnn_bf16),
                                     (void *)wei_scale, idx) *
                                     src_scale_f;
                }
            }
        }
        else {
            ZENDNN_THROW_ERROR(zendnn_invalid_arguments,
                               "Data Type not supported for scales.");
        }
        if (enable_cache) {
            zendnnVerbose(ZENDNN_PROFLOG,"Cache static scales");
            map_mutex.lock();
            matmul_SrcWei_scales_cache.add(key_obj, new_scale);
            map_mutex.unlock();
        }
    }
    else {
        zendnnVerbose(ZENDNN_PROFLOG,"Read cached static scales");
        new_scale = matmul_SrcWei_scales_cache.get(key_obj);
    }

    return;
}

// Explicit instantiations for specific data types
template bool reorderAndCacheWeights<int8_t>(
    const Key_matmul &key_obj,
    const int8_t *filter,
    int8_t *&reorder_filter,
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
    ReorderFunc<int8_t> reorder_func,
    int weight_cache_type
);

template bool reorderAndCacheWeights<float>(
    const Key_matmul &key_obj,
    const float *filter,
    float *&reorder_filter,
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
    ReorderFunc<float> reorder_func,
    int weight_cache_type
);

template bool reorderAndCacheWeights<int16_t>(
    const Key_matmul &key_obj,
    const int16_t *filter,
    int16_t *&reorder_filter,
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
    ReorderFunc<int16_t> reorder_func,
    int weight_cache_type
);

// Explicit instantiations for specific data types
template void woqReorderAndCacheWeightsAocl<int16_t>(
    const Key_matmul &key_obj,
    const int8_t *filter,
    int16_t *&reorder_filter,
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
    ReorderFunc<int16_t> reorder_func,
    int weights_type,
    float *wei_scale,
    int scale_size,
    int group_size,
    zendnn_data_type_t scale_dt
);

template void woqReorderAndCacheWeightsAocl<float>(
    const Key_matmul &key_obj,
    const int8_t *filter,
    float *&reorder_filter,
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
    ReorderFunc<float> reorder_func,
    int weights_type,
    float *wei_scale,
    int scale_size,
    int group_size,
    zendnn_data_type_t scale_dt
);
