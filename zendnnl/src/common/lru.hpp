/*******************************************************************************
 * Copyright (c) 2025-2028 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef _ZENDNNL_LRU_HPP_
#define _ZENDNNL_LRU_HPP_

#include <iostream>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <limits>
#include <optional>

#include "common/error_status.hpp"

namespace zendnnl {
namespace common {

template<typename KEY_T, typename VALUE_T>
class lru_cache_t;

/** @class lru_node_t
 *  @brief A node class in least recenty used (LRU) cache.
 */
template<typename KEY_T, typename VALUE_T>
class lru_node_t {
  friend class lru_cache_t<KEY_T, VALUE_T>;

public:
  /** @brief key type */
  using key_type   = KEY_T;

  /** @brief value type */
  using value_type = VALUE_T;

  /** @brief node type */
  using node_type  = lru_node_t<KEY_T, VALUE_T>;

private:
  /** @brief default constructor */
  lru_node_t();

  /** @brief key-value constructor */
  lru_node_t(const key_type& key_, const value_type& value_);

private:
  key_type           key;   /**< key */
  value_type         value; /**< value */
  node_type*         prev;  /**< privious node pointer */
  node_type*         next;  /**< next node pointer */
};

/** @class lru_cache_t
 *  @brief A Least Recently Used (LRU) cache class
 *
 *  A thread-safe LRU cache class.
 */
template<typename KEY_T, typename VALUE_T>
class lru_cache_t {
public:
  /** @brief all types */
  using key_type             = KEY_T;
  using value_type           = VALUE_T;
  using optional_value_type  = std::optional<value_type>;
  using const_value_type     = const value_type;
  using node_type            = lru_node_t<KEY_T, VALUE_T>;
  using const_node_type      = const node_type;
  using size_type            = uint32_t;
  using container_type       = std::unordered_map<key_type, node_type>;
  using iterator_type        = typename container_type::iterator;
  using const_iterator_type  = typename container_type::const_iterator;

public:
  /** @brief default constructor */
  lru_cache_t();

  /** @brief capacity constructor */
  explicit lru_cache_t(size_type capacity_);

  /** @brief insert a key-value pair */
  status_t                     insert(const key_type&  key_, const value_type& value_);

  /** @brief erase a node */
  status_t                     erase(const key_type& key_);

  /** @brief get a value pointed by a key. the key should exist */
  value_type                   at(const key_type& key_);

  /** @brief get a const reference to value pointed by a key. the key should exist */
  const_value_type&            at(const key_type& key_) const;

  /** @brief get an optional value pointed by a key */
  optional_value_type          get_value(const key_type& key_);

  /** @brief get a const iterator to a node pointed by a key */
  const_iterator_type          find(const key_type& key_) const;

  /** @brief get the size */
  size_type                    get_size() const noexcept;

  /** @brief get the capacity */
  size_type                    get_capacity() const noexcept;

  /** @brief set the capacity */
  status_t                     set_capacity(size_type capacity_);

  /** @brief get a const iterator to begin */
  const_iterator_type          cbegin() const noexcept;

  /** @brief get a const iterator to end */
  const_iterator_type          cend() const noexcept;

private:
  std::mutex                   lru_mutex; /**< mutex for synchronization */
  size_type                    capacity;  /**< capacity */
  size_type                    size;      /**< size */
  container_type               cache;     /**< cache container */
  node_type                    head;      /**< head of doubly linked list */
  node_type                    tail;      /**< tail of doubly linked list */
};

//Implementation
template<typename KEY_T, typename VALUE_T>
lru_node_t<KEY_T, VALUE_T>::lru_node_t():
  key{}, value{},
  prev{nullptr}, next{nullptr} {
}

template<typename KEY_T, typename VALUE_T>
lru_node_t<KEY_T, VALUE_T>::lru_node_t(const key_type& key_, const value_type& value_):
  key{key_}, value{value_},
  prev{nullptr}, next{nullptr} {
}

template<typename KEY_T, typename VALUE_T>
lru_cache_t<KEY_T, VALUE_T>::lru_cache_t():
  capacity{std::numeric_limits<size_type>::max()}, size{0}, cache{},
  head{},tail{} {

  head.next   = &tail;
  tail.prev   = &head;
}

template<typename KEY_T, typename VALUE_T>
lru_cache_t<KEY_T, VALUE_T>::lru_cache_t(size_type capacity_):
  lru_cache_t{} {
  capacity = capacity_;
}

template<typename KEY_T, typename VALUE_T>
status_t lru_cache_t<KEY_T, VALUE_T>::insert(const key_type& key_, const value_type& value_) {
  const std::lock_guard<std::mutex> lru_lock(lru_mutex);

  //if key already exists, return failure
  if(auto search_itr = cache.find(key_); search_itr != cache.end()) {
    return status_t::lru_node_exists;
  }

  //erase least recently used node.
  if(size >= capacity) {
    auto victim_ptr = tail.prev;
    victim_ptr->prev->next = victim_ptr->next;
    victim_ptr->next->prev = victim_ptr->prev;
    cache.erase(victim_ptr->key);
    size--;
  }

  //insert new node
  bool insert_status = cache.insert({key_, lru_node_t{key_, value_}}).second;
  node_type& inserted_node = cache.at(key_);

  inserted_node.next         = head.next;
  inserted_node.prev         = &head;

  (inserted_node.prev)->next = &inserted_node;
  (inserted_node.next)->prev = &inserted_node;

  size++;

  return status_t::success;
}

template<typename KEY_T, typename VALUE_T>
status_t lru_cache_t<KEY_T, VALUE_T>::erase(const key_type& key_) {
  const std::lock_guard<std::mutex> lru_lock(lru_mutex);

  auto search_itr = cache.find(key_);
  if (search_itr == cache.end()) {
    return status_t::lru_node_not_found;
  }

  (search_itr->second).prev->next = (search_itr->second).next;
  (search_itr->second).next->prev = (search_itr->second).prev;

  cache.erase(search_itr->first);
  size--;

  return status_t::success;
}

template<typename KEY_T, typename VALUE_T>
typename lru_cache_t<KEY_T, VALUE_T>::value_type
lru_cache_t<KEY_T, VALUE_T>::at(const key_type& key_) {
  const std::lock_guard<std::mutex> lru_lock(lru_mutex);
  return cache.at(key_).value;
}

template<typename KEY_T, typename VALUE_T>
typename lru_cache_t<KEY_T, VALUE_T>::const_value_type&
lru_cache_t<KEY_T, VALUE_T>::at(const key_type& key_) const {
  const container_type& const_cache = cache;
  return const_cache.at(key_).value;
}

template<typename KEY_T, typename VALUE_T>
typename lru_cache_t<KEY_T, VALUE_T>::optional_value_type
lru_cache_t<KEY_T, VALUE_T>::get_value(const key_type& key_) {
  const std::lock_guard<std::mutex> lru_lock(lru_mutex);

  auto search_itr = cache.find(key_);
  if (search_itr == cache.end()) {
    return optional_value_type{};
  }

  return optional_value_type{(search_itr->second).value};
}

template<typename KEY_T, typename VALUE_T>
typename lru_cache_t<KEY_T, VALUE_T>::const_iterator_type
lru_cache_t<KEY_T, VALUE_T>::find(const key_type& key_) const {
  const container_type& const_cache = cache;
  return const_cache.find(key_);
}

template<typename KEY_T, typename VALUE_T>
typename lru_cache_t<KEY_T, VALUE_T>::size_type
lru_cache_t<KEY_T, VALUE_T>::get_size() const noexcept {
  return size;
}

template<typename KEY_T, typename VALUE_T>
typename lru_cache_t<KEY_T, VALUE_T>::size_type
lru_cache_t<KEY_T, VALUE_T>::get_capacity() const noexcept {
  return capacity;
}

template<typename KEY_T, typename VALUE_T>
status_t lru_cache_t<KEY_T, VALUE_T>::set_capacity(size_type capacity_) {
  const std::lock_guard<std::mutex> lru_lock(lru_mutex);
  if (size >= capacity_) {
    return status_t::failure;
  }
  else {
    capacity = capacity_;
  }

  return status_t::success;
}

template<typename KEY_T, typename VALUE_T>
typename lru_cache_t<KEY_T, VALUE_T>::const_iterator_type
lru_cache_t<KEY_T, VALUE_T>::cbegin() const noexcept {
  const container_type& const_cache = cache;
  return const_cache.cbegin();
}

template<typename KEY_T, typename VALUE_T>
typename lru_cache_t<KEY_T, VALUE_T>::const_iterator_type
lru_cache_t<KEY_T, VALUE_T>::cend() const noexcept {
  const container_type& const_cache = cache;
  return const_cache.cend();
}

//some typedefs
using sptr_lru_t = lru_cache_t<std::size_t, std::shared_ptr<void>>;
using int_lru_t  = lru_cache_t<std::size_t, int>;

}//common

namespace interface {
using int_lru_t      = zendnnl::common::int_lru_t;
} //export

}//zendnnl
#endif
