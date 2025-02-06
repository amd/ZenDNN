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

#include <zendnn.hpp>
#include <unordered_map>
#include <list>
#include <memory>
#include <atomic>
#include "weight_cache.hpp"
#include <mutex>

extern std::mutex map_mutex;

// This function reorders and caches weights for matrix multiplication operations. It uses a cache to store
// reordered weights to optimize performance for repeated operations with the same parameters. The function
// supports different data types through templates and uses function pointers to perform the reordering.

// Template Parameter:
// T - The data type of the weights (e.g., int8_t, float, int16_t).
template <typename T>
void reorderAndCacheWeights(
    const Key_matmul &key_obj,
    const T *weights,
    T *&reorder_weights,
    const int k,
    const int n,
    const int ldb,
    const bool is_weights_const,
    bool inplace_reorder_wei,
    const char order,
    const char trans,
    const char reorder_param0,
    const dim_t reorder_param1,
    const dim_t reorder_param2,
    GetReorderBufSizeFunc get_reorder_buf_size,
    ReorderFunc<T> reorder_func
) {
    // Weight caching
    static zendnn::impl::lru_weight_cache_t<Key_matmul, T *> matmul_weight_cache;
    auto found_obj = matmul_weight_cache.find_key(key_obj);

    T *c_wei = const_cast<T *>(weights);
    if (inplace_reorder_wei) {
        if (!is_weights_const) {
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
            map_mutex.unlock();
        }
        reorder_weights = c_wei;
    }
    else if (!is_weights_const || !found_obj) {
        zendnnVerbose(ZENDNN_PROFLOG,"BLIS reorder weights");
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
        reorder_weights = matmul_weight_cache.get(key_obj);
    }
}

void reorderAndCacheWeightsBrgemm(
    const Key_matmul &key_obj_reorder,
    const zendnn::matmul::primitive_desc &matmul_prim_disc,
    zendnn::memory &user_weights_memory,
    zendnn::memory &reordered_weights_memory,
    zendnn::engine &eng,
    zendnn::stream &engine_stream,
    bool is_weights_const,
    bool inplace_reorder
) {
    //weight caching
    static zendnn::impl::lru_weight_cache_t<Key_matmul, zendnn::memory>
    matmul_weight_cache;
    auto found_obj_reorder = matmul_weight_cache.find_key(key_obj_reorder);

    if (inplace_reorder) {
        if (!is_weights_const) {
            reordered_weights_memory = memory(matmul_prim_disc.weights_desc(), eng);
            reorder(user_weights_memory, reordered_weights_memory).execute(engine_stream,
                    user_weights_memory, reordered_weights_memory);
            //Save in map
            map_mutex.lock();
            int8_t *reorder_ptr = (int8_t *)reordered_weights_memory.get_data_handle();
            int8_t *user_ptr = (int8_t *)user_weights_memory.get_data_handle();
            #pragma omp parallel for
            for (int idx = 0; idx < matmul_prim_disc.weights_desc().get_size(); idx++) {
                user_ptr[idx] = reorder_ptr[idx];
            }
            map_mutex.unlock();
        }
        reordered_weights_memory = user_weights_memory;
    }
    else if (!is_weights_const || !found_obj_reorder) {
        reordered_weights_memory = memory(matmul_prim_disc.weights_desc(), eng);
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
        reordered_weights_memory = matmul_weight_cache.get(key_obj_reorder);
    }
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
    int32_t wei_zero_point
) {
    // Used to cache compensation when only src_scales exist.
    static zendnn::impl::lru_weight_cache_t<Key_matmul, int32_t *>
    matmul_src_zp_wei_comp_cache;

    bool enable_cache = zenEnvObj.zenZpCompCache;
    if (!wei_zero_point && !src_zero_point) {
        return;
    }
    else if (!wei_zero_point) {
        auto found_obj = matmul_src_zp_wei_comp_cache.find_key(key_obj);
        if (!found_obj) {
            // acc is freed by lru map
            acc = (int32_t *)zendnn_aligned_alloc(64, sizeof(int32_t)*N);
            std::vector<int32_t> wei_comp(N,0);
            if (src_zero_point) {
                for_(dim_t k = 0; k < K; ++k)
                for (dim_t n = 0; n < N; ++n) {
                    if (k == 0) {
                        wei_comp[n] = int32_t(0);
                    }
                    wei_comp[n] += wei[wei_s0 * k + wei_s1 * n];
                }
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
            acc = matmul_src_zp_wei_comp_cache.get(key_obj);
        }
    }
    else if (!src_zero_point) {
        std::vector<int32_t> src_comp(M,0);
        // acc is freed in main function for wei_zp != 0
        acc = (int32_t *)zendnn_aligned_alloc(64, sizeof(int32_t)*M*N);
        if (wei_zero_point) {
            for_(dim_t m = 0; m < M; ++m)
            for (dim_t k = 0; k < K; ++k) {
                if (k == 0) {
                    src_comp[m] = int32_t(0);
                }
                src_comp[m] += src[src_s0 * m + src_s1 * k];
            }
        }

        for_(dim_t m = 0; m < M; ++m)
        for (dim_t n = 0; n < N; ++n) {
            acc[m * ldc + n] = 0 - wei_zero_point * src_comp[m];
        }
    }
    else {
        std::vector<int32_t> src_comp(M,0);
        std::vector<int32_t> wei_comp(N,0);
        // acc is freed in main function for wei_zp != 0
        acc = (int32_t *)zendnn_aligned_alloc(64, sizeof(int32_t)*M*N);
        if (wei_zero_point) {
            for_(dim_t m = 0; m < M; ++m)
            for (dim_t k = 0; k < K; ++k) {
                if (k == 0) {
                    src_comp[m] = int32_t(0);
                }
                src_comp[m] += src[src_s0 * m + src_s1 * k];
            }
        }

        if (src_zero_point) {
            for_(dim_t k = 0; k < K; ++k)
            for (dim_t n = 0; n < N; ++n) {
                if (k == 0) {
                    wei_comp[n] = int32_t(0);
                }
                wei_comp[n] += wei[wei_s0 * k + wei_s1 * n];
            }
        }

        for_(dim_t m = 0; m < M; ++m)
        for (dim_t n = 0; n < N; ++n)
            acc[m * ldc + n] = 0 - src_zero_point * wei_comp[n]
                               - wei_zero_point * src_comp[m]
                               + src_zero_point * wei_zero_point * (int)K;
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
        new_scale = matmul_SrcWei_scales_cache.get(key_obj);
    }

    return;
}

// Explicit instantiations for specific data types
template void reorderAndCacheWeights<int8_t>(
    const Key_matmul &key_obj,
    const int8_t *filter,
    int8_t *&reorder_filter,
    const int k,
    const int n,
    const int ldb,
    const bool is_weights_const,
    bool inplace_reorder_wei,
    const char order,
    const char trans,
    const char reorder_param0,
    const dim_t reorder_param1,
    const dim_t reorder_param2,
    GetReorderBufSizeFunc get_reorder_buf_size,
    ReorderFunc<int8_t> reorder_func
);

template void reorderAndCacheWeights<float>(
    const Key_matmul &key_obj,
    const float *filter,
    float *&reorder_filter,
    const int k,
    const int n,
    const int ldb,
    const bool is_weights_const,
    bool inplace_reorder_wei,
    const char order,
    const char trans,
    const char reorder_param0,
    const dim_t reorder_param1,
    const dim_t reorder_param2,
    GetReorderBufSizeFunc get_reorder_buf_size,
    ReorderFunc<float> reorder_func
);

template void reorderAndCacheWeights<int16_t>(
    const Key_matmul &key_obj,
    const int16_t *filter,
    int16_t *&reorder_filter,
    const int k,
    const int n,
    const int ldb,
    const bool is_weights_const,
    bool inplace_reorder_wei,
    const char order,
    const char trans,
    const char reorder_param0,
    const dim_t reorder_param1,
    const dim_t reorder_param2,
    GetReorderBufSizeFunc get_reorder_buf_size,
    ReorderFunc<int16_t> reorder_func
);

namespace zendnn {
namespace impl {

template <typename KEY_T, typename VALUE_T>
lru_weight_cache_t<KEY_T, VALUE_T>::lru_weight_cache_t(int capacity) :
    capacity_(capacity) {
    cache_mapper_ = std::make_unique<
                    std::unordered_map<w_key_t, timed_entry_t>>();
}

template <typename KEY_T, typename VALUE_T>
lru_weight_cache_t<KEY_T, VALUE_T>::~lru_weight_cache_t() {
    evict();
    return;
}


template <typename KEY_T, typename VALUE_T>
void lru_weight_cache_t<KEY_T, VALUE_T>::set_capacity(int capacity) {
    if (capacity < 0) {
        throw std::invalid_argument("Capacity cannot be negative");
    }
    capacity_ = capacity;
    if (capacity_ < cache_mapper_->size()) {
        evict(cache_mapper_->size() - capacity_);
    }
}

template <typename KEY_T, typename VALUE_T>
int lru_weight_cache_t<KEY_T, VALUE_T>::get_capacity() const {
    return static_cast<int>(capacity_);
}

template <typename KEY_T, typename VALUE_T>
typename lru_weight_cache_t<KEY_T, VALUE_T>::value_t
lru_weight_cache_t<KEY_T, VALUE_T>::get_or_add(
    const w_key_t &key,
    const value_t &value) {
    if (cache_mapper_->find(key) == cache_mapper_->end()) {
        add(key, value);
    }
    return get(key);
}

template <typename KEY_T, typename VALUE_T>
bool lru_weight_cache_t<KEY_T, VALUE_T>::find_key(const w_key_t &key) const {
    return (cache_mapper_->find(key) != cache_mapper_->end());
}

template <typename KEY_T, typename VALUE_T>
void lru_weight_cache_t<KEY_T, VALUE_T>::remove_if_invalidated(
    const w_key_t &key) {
    auto it = cache_mapper_->find(key);
    if (it != cache_mapper_->end()) {
        cache_mapper_->erase(it);
    }
}

template <typename KEY_T, typename VALUE_T>
int lru_weight_cache_t<KEY_T, VALUE_T>::get_size() const {
    return static_cast<int>(cache_mapper_->size());
}

template <typename KEY_T, typename VALUE_T>
lru_weight_cache_t<KEY_T, VALUE_T>::timed_entry_t::timed_entry_t(
    const value_t &value,
    size_t timestamp)
    : value_(value), timestamp_(timestamp) {}

template <typename KEY_T, typename VALUE_T>
void lru_weight_cache_t<KEY_T, VALUE_T>::evict(size_t n) {
    while (capacity_ < std::numeric_limits<int>::max() &&
            cache_mapper_->size() > capacity_ - n) {
        auto oldest = std::min_element(
                          cache_mapper_->begin(), cache_mapper_->end(),
        [](const auto &a, const auto &b) {
            return a.second.timestamp_ < b.second.timestamp_;
        });
        if constexpr(std::is_pointer<VALUE_T>::value) {
            if (oldest->second.value_ != NULL) {
                std::free(oldest->second.value_);
            }
        }
        cache_mapper_->erase(oldest);
    }
}

template <typename KEY_T, typename VALUE_T>
void lru_weight_cache_t<KEY_T, VALUE_T>::evict() {
    // Free memory for all entries in the cache
    for (auto &entry : *cache_mapper_) {
        // Assuming VALUE_T is a pointer type
        if constexpr(std::is_pointer<VALUE_T>::value) {
            if (entry.second.value_ != NULL) {
                std::free(entry.second.value_);
            }
        }
    }
    // Clear the cache
    cache_mapper_->clear();
}

template <typename KEY_T, typename VALUE_T>
void lru_weight_cache_t<KEY_T, VALUE_T>::add(const w_key_t &key,
        const value_t &value) {
    evict(1);
    size_t timestamp = current_timestamp_++;
    cache_mapper_->emplace(std::piecewise_construct,
                           std::forward_as_tuple(key),
                           std::forward_as_tuple(value, timestamp));
}

template <typename KEY_T, typename VALUE_T>
typename lru_weight_cache_t<KEY_T, VALUE_T>::value_t
lru_weight_cache_t<KEY_T, VALUE_T>::get(
    const w_key_t &key) {
    auto it = cache_mapper_->find(key);
    if (it != cache_mapper_->end()) {
        it->second.timestamp_ = current_timestamp_++;
        return it->second.value_;
    }
    throw std::runtime_error("Key not found in cache.");
}

template struct lru_weight_cache_t <Key_matmul, memory>;
template struct lru_weight_cache_t <Key_matmul, int16_t *>;
template struct lru_weight_cache_t <Key_matmul, int8_t *>;
template struct lru_weight_cache_t <Key_matmul, float *>;
}
}
