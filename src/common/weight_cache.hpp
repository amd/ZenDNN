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

#ifndef COMMON_WEIGHT_CACHE_HPP
#define COMMON_WEIGHT_CACHE_HPP

#include "common/zendnn_private.hpp"
#include <zendnn.hpp>
#include <unordered_map>
#include <memory>
#include <atomic>
#include <stdexcept>

// Define a function pointer type for getting the reorder buffer size.
using GetReorderBufSizeFunc = siz_t (*)(const char, const char, const char,
                                        const dim_t, const dim_t);

// Define a template function pointer type for reordering.
template <typename T>
using ReorderFunc = void (*)(const char, const char, const char, const T *, T *,
                             const dim_t, const dim_t, const dim_t);

template <typename T>
void reorderAndCacheWeights(
    const Key_matmul &key_obj,
    const T *filter,
    T *&reorder_filter,
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
);

void reorderAndCacheWeightsBrgemm(
    const Key_matmul &key_obj_reorder,
    const zendnn::matmul::primitive_desc &matmul_prim_disc,
    zendnn::memory &user_weights_memory,
    zendnn::memory &reordered_weights_memory,
    zendnn::engine &eng,
    zendnn::stream &engine_stream,
    bool is_weights_const,
    bool inplace_reorder = false
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
    int32_t wei_zero_point
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

namespace zendnn {
namespace impl {
template <typename KEY_T, typename VALUE_T>
struct lru_weight_cache_t {
    using w_key_t = KEY_T;
    using value_t = VALUE_T;

    //Constructor
    lru_weight_cache_t(int capacity=
                           zendnn::impl::getenv_int("ZENDNN_WEIGHT_CACHE_CAPACITY",
                                   std::numeric_limits<int>::max()));
    // Destructor
    ~lru_weight_cache_t();

    // Public methods
    void set_capacity(int capacity);
    int get_capacity() const;
    value_t get_or_add(const w_key_t &key, const value_t &value);
    void remove_if_invalidated(const w_key_t &key);
    int get_size() const;
    void add(const w_key_t &key, const value_t &value);
    value_t get(const w_key_t &key);
    bool find_key(const w_key_t &key) const;

  private:
    // Private struct to hold the cache value and its timestamp
    struct timed_entry_t {
        value_t value_;
        std::atomic<size_t> timestamp_;
        // Constructor
        timed_entry_t(const value_t &value, size_t timestamp);
    };

    // Private methods
    void evict();
    void evict(size_t n);

    // Private members
    size_t capacity_;
    std::atomic<size_t> current_timestamp_;
    std::unique_ptr<std::unordered_map<w_key_t, timed_entry_t>> cache_mapper_;
};
}
}
#endif // COMMON_WEIGHT_CACHE_HPP
