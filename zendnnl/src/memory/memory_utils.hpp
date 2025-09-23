/*******************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef  _MEMORY_UTILS_HPP_
#define  _MEMORY_UTILS_HPP_

#include <cstdint>
#include <numeric>
#include <vector>
#include "common/data_types.hpp"
#include "common/zendnnl_global.hpp"

/** @namespace zendnnl
 *  @brief ZenDNNL top level namespace.
 */
namespace zendnnl {
/** @namespace zendnnl::memory
 *  @brief A namespace to contain all memory management related classes, enumerations,
 *  variables and functions.
 */
namespace memory {

using namespace zendnnl::common;

// Utility function to compute the product of all elements in a vector
template <typename T>
int compute_product(const std::vector<T> &vec) {
  return std::accumulate(vec.begin(), vec.end(), 1, std::multiplies<T>());
}

/**
 * @brief Reads a value of any data_type_t at a given index and returns it as the specified type.
 * @tparam T The return type (e.g., int32_t or float).
 * @param value Pointer to the array of values.
 * @param data_type The data type of the values (data_type_t).
 * @param index The index of the value to be read and cast.
 * @return The value at the specified index casted to the specified type.
 */
template <typename T>
T read_and_cast(const void *value, data_type_t data_type, size_t index = 0) {
  switch (data_type) {
  case data_type_t::s8:
    return static_cast<T>(reinterpret_cast<const int8_t *>(value)[index]);
  case data_type_t::u8:
    return static_cast<T>(reinterpret_cast<const uint8_t *>(value)[index]);
  case data_type_t::s32:
    return static_cast<T>(reinterpret_cast<const int32_t *>(value)[index]);
  case data_type_t::f32:
    return static_cast<T>(reinterpret_cast<const float *>(value)[index]);
  case data_type_t::bf16:
    return static_cast<T>(bfloat16_t::bf16_to_f32_val(reinterpret_cast<const int16_t *>
                                         (value)[index]));
  default:
    log_error("Unsupported data type for casting");
    return static_cast<T>(0); // Return 0 as a fallback for unsupported types
  }
}

}
}

#endif // _MEMORY_UTILS_HPP_