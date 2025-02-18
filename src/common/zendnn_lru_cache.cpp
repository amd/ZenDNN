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
#include "zendnn_lru_cache.hpp"
#include <mutex>

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
template struct lru_weight_cache_t <Key_matmul, char *>;
template struct lru_weight_cache_t <Key_matmul, int32_t *>;
}
}
