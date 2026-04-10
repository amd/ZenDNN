/********************************************************************************
# * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *******************************************************************************/

#ifndef _LRU_CACHE_HPP
#define _LRU_CACHE_HPP

#include <unordered_map>
#include <atomic>
#include <memory>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <algorithm>
#include "operators/matmul/matmul_config.hpp"
namespace zendnnl {
namespace lowoha {
namespace matmul {

template <typename KEY_T, typename VALUE_T>
class lru_cache_t {
  using lru_key_t = KEY_T;
  using value_t = VALUE_T;
 public:
  //Constructor
  lru_cache_t(uint32_t capacity =
                ops::matmul_config_t::instance().get_lru_cache_capacity());
  // Destructor
  ~lru_cache_t();

  // Public methods
  void set_capacity(uint32_t capacity);
  uint32_t get_capacity() const;
  value_t get_or_add(const lru_key_t &key, const value_t &value);
  void remove_if_invalidated(const lru_key_t &key);
  int get_size() const;
  void add(const lru_key_t &key, const value_t &value);
  value_t get(const lru_key_t &key);
  bool find_key(const lru_key_t &key) const;

 private:
  // Private struct to hold the cache value and its timestamp
  struct timed_entry_t {
    value_t value_;
    std::atomic<size_t> timestamp_;
    // Constructor
    timed_entry_t(const value_t &value, size_t timestamp);
  };

  void evict();
  void evict(size_t n);

  // Private members
  uint32_t capacity_;
  std::atomic<size_t> current_timestamp_;
  std::unique_ptr<std::unordered_map<lru_key_t, timed_entry_t>> lru_cache_map_;
  mutable std::mutex mutex_;  // Mutex for thread-safe access
};

template <typename KEY_T, typename VALUE_T>
lru_cache_t<KEY_T, VALUE_T>::lru_cache_t(uint32_t capacity) :
  capacity_(capacity) {
  lru_cache_map_ = std::make_unique<
                   std::unordered_map<lru_key_t, timed_entry_t>>();
}

template <typename KEY_T, typename VALUE_T>
lru_cache_t<KEY_T, VALUE_T>::~lru_cache_t() {
  evict();
}

template <typename KEY_T, typename VALUE_T>
void lru_cache_t<KEY_T, VALUE_T>::set_capacity(uint32_t capacity) {
  std::lock_guard<std::mutex> lock(mutex_);
  capacity_ = capacity;
  if (capacity_ < lru_cache_map_->size()) {
    evict(lru_cache_map_->size() - capacity_);
  }
}

template <typename KEY_T, typename VALUE_T>
uint32_t lru_cache_t<KEY_T, VALUE_T>::get_capacity() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return capacity_;
}

template <typename KEY_T, typename VALUE_T>
typename lru_cache_t<KEY_T, VALUE_T>::value_t
lru_cache_t<KEY_T, VALUE_T>::get_or_add(
  const lru_key_t &key,
  const value_t &value) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (lru_cache_map_->find(key) == lru_cache_map_->end()) {
    // Call internal add without lock (we already hold it)
    evict(1);
    size_t timestamp = current_timestamp_++;
    lru_cache_map_->emplace(std::piecewise_construct,
                            std::forward_as_tuple(key),
                            std::forward_as_tuple(value, timestamp));
  }
  // Return value and update timestamp
  auto it = lru_cache_map_->find(key);
  if (it != lru_cache_map_->end()) {
    it->second.timestamp_ = current_timestamp_++;
    return it->second.value_;
  }
  throw std::runtime_error("Key not found in cache.");
}

template <typename KEY_T, typename VALUE_T>
bool lru_cache_t<KEY_T, VALUE_T>::find_key(const lru_key_t &key) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return (lru_cache_map_->find(key) != lru_cache_map_->end());
}

template <typename KEY_T, typename VALUE_T>
void lru_cache_t<KEY_T, VALUE_T>::remove_if_invalidated(
  const lru_key_t &key) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = lru_cache_map_->find(key);
  if (it != lru_cache_map_->end()) {
    lru_cache_map_->erase(it);
  }
}

template <typename KEY_T, typename VALUE_T>
int lru_cache_t<KEY_T, VALUE_T>::get_size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return static_cast<int>(lru_cache_map_->size());
}

template <typename KEY_T, typename VALUE_T>
lru_cache_t<KEY_T, VALUE_T>::timed_entry_t::timed_entry_t(
  const value_t &value,
  size_t timestamp)
  : value_(value), timestamp_(timestamp) {}

template <typename KEY_T, typename VALUE_T>
void lru_cache_t<KEY_T, VALUE_T>::evict(size_t n) {
  while (capacity_ < std::numeric_limits<uint32_t>::max() &&
         lru_cache_map_->size() > capacity_ - n) {
    auto oldest = std::min_element(
                    lru_cache_map_->begin(), lru_cache_map_->end(),
    [](const auto &a, const auto &b) {
      return a.second.timestamp_ < b.second.timestamp_;
    });
    if constexpr(std::is_pointer<VALUE_T>::value) {
      if (oldest->second.value_ != nullptr) {
        std::free(oldest->second.value_);
      }
    }
    lru_cache_map_->erase(oldest);
  }
}

template <typename KEY_T, typename VALUE_T>
void lru_cache_t<KEY_T, VALUE_T>::evict() {
  // Free memory for all entries in the cache
  for (auto &entry : *lru_cache_map_) {
    // Assuming VALUE_T is a pointer type
    if constexpr(std::is_pointer<VALUE_T>::value) {
      if (entry.second.value_ != nullptr) {
        std::free(entry.second.value_);
      }
    }
  }
  // Clear the cache
  lru_cache_map_->clear();
}

template <typename KEY_T, typename VALUE_T>
void lru_cache_t<KEY_T, VALUE_T>::add(const lru_key_t &key,
                                      const value_t &value) {
  std::lock_guard<std::mutex> lock(mutex_);
  evict(1);
  size_t timestamp = current_timestamp_++;
  lru_cache_map_->emplace(std::piecewise_construct,
                          std::forward_as_tuple(key),
                          std::forward_as_tuple(value, timestamp));
}

template <typename KEY_T, typename VALUE_T>
typename lru_cache_t<KEY_T, VALUE_T>::value_t
lru_cache_t<KEY_T, VALUE_T>::get(
  const lru_key_t &key) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = lru_cache_map_->find(key);
  if (it != lru_cache_map_->end()) {
    it->second.timestamp_ = current_timestamp_++;
    return it->second.value_;
  }
  throw std::runtime_error("Key not found in cache.");
}
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

namespace interface {
template<typename KEY_T, typename VALUE_T>
using lru_cache_t = zendnnl::lowoha::matmul::lru_cache_t<KEY_T, VALUE_T>;
} //export

#endif // ZENDNN_LRU_CACHE_HPP
