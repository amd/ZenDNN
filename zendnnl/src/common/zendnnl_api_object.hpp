/********************************************************************************
# * Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef _ZENDNNL_API_OBJECT_TYPE_HPP_
#define _ZENDNNL_API_OBJECT_TYPE_HPP_

#include <type_traits>
#include <utility>
#include <memory>
#include <functional>
#include "common/zendnnl_object.hpp"
#include "common/hashable_object.hpp"
#include "common/zendnnl_global.hpp"

/** @def SET_IMPL_FORWARD_VA(function, ...)
 *  @brief call forward to implementation
 */
#define SET_IMPL_FORWARD_VA(FUNC, ...)                                  \
  do {                                                                  \
    impl->FUNC(std::forward<decltype(__VA_ARGS__)>(__VA_ARGS__));       \
  } while(0);                                                           \

/** @def SET_IMPL_FORWARD(function, ...)
 *  @brief call forward to implementation
 */
#define SET_IMPL_FORWARD(FUNC)                                          \
  do {                                                                  \
    impl->FUNC();                                                       \
  } while(0);                                                           \

/** @def GET_IMPL_FORWARD(function, ...)
 *  @brief call forward to implementation
 */
#define GET_IMPL_FORWARD_VA(FUNC, ...)                                  \
  return impl->FUNC(std::forward<decltype(__VA_ARGS__)>(__VA_ARGS__));  \

/** @def GET_IMPL_FORWARD(function, ...)
 *  @brief call forward to implementation
 */
#define GET_IMPL_FORWARD(FUNC)                                          \
  return impl->FUNC();                                                  \

namespace zendnnl {
namespace common {

using namespace zendnnl::error_handling;
using json         = nlohmann::json;
using ordered_json = nlohmann::ordered_json;

template<typename SELF_T, typename IMPL_T = hashable_object_t>
class api_object_t {
 public:
  using self_type      = SELF_T;
  using impl_type      = IMPL_T;
  using impl_sptr_type = std::shared_ptr<impl_type>;

  // public:
  // template<typename... ARGS_T, typename R_T>
  // R_T forward(R_T (impl_type::*func_ptr)(ARGS_T...), ARGS_T&&... args_) {
  //   return (*impl.*func_ptr)(std::forward<ARGS_T>(args_)...);
  // }

 public:
  self_type     &set_name(std::string name_);
  std::string    get_name() const;

  nlohmann::ordered_json  get_object_info() const;
  nlohmann::ordered_json  get_object_runtime_info() const;
  std::string  get_object_summary() const;
  std::string  get_object_runtime_summary() const;
  status_t  get_last_status() const;
  bool is_bad_object() const;
  bool is_unnamed_object() const;

  std::size_t get_hash() const;

  self_type &cache(bool cache_flag_);
  self_type &load(std::size_t hash_key_);

  virtual self_type &create();

 protected:
  api_object_t();

 protected:
  bool            cache_flag;
  std::size_t     hash_key;
  impl_sptr_type  impl;
};

//implementation
template<typename SELF_T, typename IMPL_T>
api_object_t<SELF_T, IMPL_T>::api_object_t() :
  cache_flag{false}, hash_key{0},
  impl{std::make_shared<IMPL_T>()} {
}

template<typename SELF_T, typename IMPL_T>
SELF_T &api_object_t<SELF_T, IMPL_T>::set_name(std::string name_) {

  SET_IMPL_FORWARD_VA(set_name, name_);

  return static_cast<self_type &>(*this);
}

template<typename SELF_T, typename IMPL_T>
std::string api_object_t<SELF_T, IMPL_T>::get_name() const {

  GET_IMPL_FORWARD(get_name);
}

template<typename SELF_T, typename IMPL_T>
ordered_json api_object_t<SELF_T, IMPL_T>::get_object_info() const {

  GET_IMPL_FORWARD(get_object_info);
}

template<typename SELF_T, typename IMPL_T>
ordered_json api_object_t<SELF_T, IMPL_T>::get_object_runtime_info() const {

  GET_IMPL_FORWARD(get_object_runtime_info);
}

template<typename SELF_T, typename IMPL_T>
std::string api_object_t<SELF_T, IMPL_T>::get_object_summary() const {

  GET_IMPL_FORWARD(get_object_summary);
}

template<typename SELF_T, typename IMPL_T>
std::string api_object_t<SELF_T, IMPL_T>::get_object_runtime_summary() const {

  GET_IMPL_FORWARD(get_object_runtime_summary);
}

template<typename SELF_T, typename IMPL_T>
status_t api_object_t<SELF_T, IMPL_T>::get_last_status() const {

  GET_IMPL_FORWARD(get_object_runtime_summary);
}

template<typename SELF_T, typename IMPL_T>
bool api_object_t<SELF_T, IMPL_T>::is_bad_object() const {

  GET_IMPL_FORWARD(is_bad_object);
}

template<typename SELF_T, typename IMPL_T>
bool api_object_t<SELF_T, IMPL_T>::is_unnamed_object() const {

  GET_IMPL_FORWARD(is_unnamed_object);
}

template<typename SELF_T, typename IMPL_T>
std::size_t api_object_t<SELF_T, IMPL_T>::get_hash() const {

  GET_IMPL_FORWARD(get_hash);
}

template<typename SELF_T, typename IMPL_T>
SELF_T &api_object_t<SELF_T, IMPL_T>::cache(bool cache_flag_) {

  cache_flag = cache_flag_;
  return static_cast<self_type &>(*this);
}

template<typename SELF_T, typename IMPL_T>
SELF_T &api_object_t<SELF_T, IMPL_T>::load(std::size_t hash_key_) {
  auto cached_value = zendnnl_lru_cache().get_value(hash_key_);
  if (cached_value) {
    this->impl = std::static_pointer_cast<impl_type>(cached_value.value());
    hash_key = hash_key_;
  }
  else {
    std::string message = "key <";
    message += std::to_string(hash_key_);
    message += "> not found in cache!";
    EXCEPTION_WITH_LOC(message);
  }

  return static_cast<self_type &>(*this);
}

template<typename SELF_T, typename IMPL_T>
SELF_T &api_object_t<SELF_T, IMPL_T>::create() {

  //hash_key indicates the object is loaded.
  if (hash_key) {
    return static_cast<self_type &>(*this);
  }

  impl->create();
  hash_key = impl->get_hash();

  return static_cast<self_type &>(*this);
}

} //common
} //zendnnl

#endif
