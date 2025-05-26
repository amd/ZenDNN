/********************************************************************************
# * Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef _HASH_UTILS_HPP_
#define _HASH_UTILS_HPP_

#include <iostream>
#include <cstdint>
#include <vector>
#include <bitset>
#include <chrono>
#include <string>

namespace zendnnl {
namespace common {

// std::size_t hash_mix(std::size_t seed, std::size_t hash_key) {
//   return seed ^= hash_key + 0x9e3779b9 + (seed << 6) + (seed >> 2);
// }

// The following code is derived from Boost C++ library
// Copyright 2005-2014 Daniel James.
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
/**
 * Combine hash of an integral type with seed.
 * @param seed : seed to which hash to be combined.
 * @param v    : an integer type variable.
 * @return combined hash.
 */
template <typename T,
          std::enable_if_t<std::is_integral_v<T>, bool> = true>
static inline std::size_t hash_combine(std::size_t seed, const T& v) {
  return seed ^= std::hash<T> {}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

//ToDo : inroduce a check that T has hash() function.
//       c++20 use concepts.
/**
 * Combine hash of a class with seed.
 * @param seed : seed to which hash to be combined.
 * @param v    : object of a class.
 * @return combined hash.
 */
template<typename T,
         std::enable_if_t<std::is_class_v<T>, bool> = true>
static inline std::size_t hash_combine(std::size_t seed, T& v) {
  return seed ^= v.hash() + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

/**
 * Combine hash of a vector with seed.
 * @param seed : seed to which hash to be combined.
 * @param vec  : vector of a type convertible to uint64_t.
 * @return combined hash.
 */
template <typename T>
static inline std::size_t hash_combine(std::size_t seed, std::vector<T>& vec) {
  for (auto v: vec) {
    seed = hash_combine(seed, static_cast<uint64_t>(v));
  }
  return seed;
}

// template <>
// std::size_t hash_combine<std::string>(std::size_t seed, const std::string& v) {
//   return seed ^= std::hash<std::string> {}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
// }

/**
 * Combine hash of a raw pointer with seed.
 * @param seed : seed to which hash to be combined.
 * @param ptr  : a raw pointer, converted to uintptr_t.
 * @return combined hash.
 */
template <typename T>
static inline std::size_t hash_combine(std::size_t seed, T* ptr) {
  seed = hash_combine<std::uintptr_t>(seed, reinterpret_cast<std::uintptr_t>(ptr));
  return seed;
}

/**
 * Combine hash of an array with seed.
 * @param seed : seed to which hash to be combined.
 * @param v  : a raw pointer to array.
 * @param size : array size.
 * @return combined hash.
 */
template <typename T>
static inline std::size_t hash_combine(std::size_t seed, T* v, int size) {
  for (int i = 0; i < size; i++) {
    seed = hash_combine(seed, static_cast<uint64_t>(v[i]));
  }
  return seed;
}

/**
 * Combine hash of a string with seed.
 * @param seed : seed to which hash to be combined.
 * @param v  : a raw pointer to array.
 * @param size : array size.
 * @return combined hash.
 */
static inline std::size_t hash_combine(std::size_t seed, std::string str) {
  std::size_t str_hash = std::hash<std::string>{}(str);
  seed ^= str_hash + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}

} //common
} //zendnnl

#endif
