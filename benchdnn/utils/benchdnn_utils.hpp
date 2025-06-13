/********************************************************************************
# * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef _BENCHDNN_UTILS_HPP_
#define _BENCHDNN_UTILS_HPP_

#include "zendnnl.hpp"

using namespace zendnnl::interface;

#define COLD_CACHE 0

#if COLD_CACHE
  #include <emmintrin.h>
  // TODO: Extract cache sizes dynamically at runtime instead of hardcoding.
  // Size of the buffer (in bytes) used to flush the cache.
  // Sum of cache levels: 512 MB (L3), 96 MB (L2), 3 MB (L1d), 3 MB (L1i).
  // Ensures the buffer covers the entire last-level cache (LLC) and lower caches.
  #define CACHE_SIZE ((512 + 96 + 3 + 3) * 1024 * 1024)
  // Size (in bytes) of a cache line; typically 64 bytes on x86 systems.
  #define CACHE_LINE_SIZE 64
  /**
  * @brief Global accumulator to prevent compiler optimization of cache flush operations.
  *
  * This variable is incremented during cache flushes to ensure the memory operations
  * are not optimized away by the compiler. Only used if COLD_CACHE is enabled.
  */
  extern volatile unsigned long global_sum;
#endif

namespace zendnnl {
namespace benchdnn {

/**
 * @fn trim
 * @brief Removes all whitespace characters from the input string (in-place).
 *
 * @param str Reference to the string to be trimmed.
 */
void trim(std::string &str);

/**
* @fn split
* @brief Splits the input string into a vector of substrings using the given delimiter.
*
* @param s Input string to split.
* @param delimiter Character to split the string on.
* @return std::vector<std::string> Vector of trimmed substrings.
*/
std::vector<std::string> split(const std::string &s, char delimiter);

/**
 * @brief Converts a string representation of a data type to its corresponding enum value.
 *
 * This function maps a string such as "f32", "s8", or "u8" to the corresponding
 * `data_type_t` enum value
 *
 * @param str String representation of the data type (e.g., "f32", "s8").
 * @return data_type_t Corresponding enum value.
 */
data_type_t strToDatatype(const std::string &str);

/**
 * @brief Converts a data_type_t enum value to its string representation.
 *
 * This function maps a data_type_t value (e.g., f32, bf16) to its corresponding
 * string (e.g., "f32", "bf16") for display or output purposes.
 *
 * @param dt The data_type_t enum value to convert.
 * @return std::string The string representation of the data type.
 */
std::string datatypeToStr(data_type_t dt);

/**
* @fn strToPostOps
* @brief Converts a string representation of a post operation to its corresponding enum value
*
* This function translates a string such as "relu", "gelu", or "sum" into the
* corresponding `post_op_type_t`
*
* @param str String representation of the post operation (e.g., "relu", "gelu").
* @return post_op_type_t Corresponding enum value.
*/
post_op_type_t strToPostOps(const std::string &str);

/**
 * @brief Converts a post_op_type_t enum value to its string representation.
 *
 * This function maps a post_op_type_t value (e.g., relu, gelu) to its corresponding
 * string (e.g., "relu", "gelu") for display or output purposes.
 *
 * @param post_op The post_op_type_t enum value to convert.
 * @return std::string The string representation of the post operation.
 */
std::string postOpsToStr(post_op_type_t post_op);

#if COLD_CACHE
  /**
  * @brief Simulates cold cache conditions by flushing the entire cache.
  *
  * Iterates over the provided buffer, incrementing each cache line and using _mm_clflush
  * to evict it from the cache. The global_sum variable is updated to prevent compiler
  * optimization. The buffer should be large enough to cover the LLC and lower cache levels.
  *
  * @param buffer Reference to a buffer (std::vector<char>) covering the full cache size.
  */
  void flush_cache(std::vector<char> &buffer);
#endif

} // namespace benchdnn
} // namespace zendnnl
#endif