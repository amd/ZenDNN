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
#define  OK          (0)
#define  NOT_OK      (1)

#define COLD_CACHE 0

#if COLD_CACHE
  #include <emmintrin.h>
  #include <numa.h>
  // Size (in bytes) of a cache line; typically 64 bytes on x86 systems.
  #define CACHE_LINE_SIZE 64
#endif

namespace zendnnl {
namespace benchdnn {

/**
 * @struct global_options
 * @brief Holds global configuration options for benchmarking.
 *
 * This structure contains options that affect the overall benchmarking behavior,
 * such as the number of dimensions (ndims) for tensor creation and operator setup.
 *
 * @var ndims Number of dimensions for tensors (e.g., 2 for standard matmul, 3 for batched matmul).
 */
struct global_options {
  size_t bs; /**< Batch size (for batched matmul; default 1 for non-batched). */
  size_t m; /**< Number of rows in matrix A (output rows). */
  size_t k; /**< Number of columns in matrix A / rows in matrix B (inner dimension). */
  std::vector<size_t> n_values; /**< Vector of output columns
                               for each layer (multi-layer support). */
  bool isBiasEnabled; /**< Flag indicating if bias is enabled in the matmul operation. */
  std::vector<zendnnl::ops::post_op_type_t> post_ops; /**< List of post operations
                                                      to apply (e.g., relu, gelu). */
  int ndims; /**< Number of dimensions for tensors (e.g., 2 for standard matmul, 3 for batched matmul). */
  int iters; /**< Number of iterations to run the benchmark. */
  data_type_t sdt; /**< Datatype of input. */
  data_type_t wdt; /**< Datatype of weights. */
  data_type_t ddt; /**< Datatype of destination/output. */
  std::string kernel_name; /**< Name of the kernel to use. */
  data_type_t bias_dt; /**< Datatype of bias. */
  bool isTransA; /**< Transpose flag for input matrix */
  bool isTransB; /**< Transpose flag for weight matrix */
  float alpha, beta; /**< Scaling factors for matmul operation. */
  int warmup_iters; /**< Number of warmup iterations to run before actual benchmarking. */

  global_options() : isBiasEnabled(false), ndims(2), iters(100),
    sdt(data_type_t::f32), wdt(data_type_t::f32),
    ddt(data_type_t::f32), kernel_name("aocl_blis"), bias_dt(data_type_t::f32),
    isTransA(false), isTransB(false), warmup_iters(-1), alpha(1.0f), beta(0.0f) {}
};

/**
 * @enum InputMode
 * @brief Specifies the mode of input for the benchmarking utility.
 *
 * This enumeration defines the possible sources of input for the benchmark configuration:
 * - FILE: Input is read from a configuration file.
 * - MODEL: Input is read from a model file.
 * - COMMAND_LINE: Input is provided directly via command-line arguments.
 */
enum class InputMode {
  FILE,
  MODEL,
  COMMAND_LINE
};

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
  * This function allocates a buffer of the specified size and iterates over it,
  * accessing each cache line and using _mm_clflush to evict it from all cache levels
  * (L1, L2, and LLC). This ensures a cold cache state before benchmarking operations,
  * providing more accurate performance measurements under realistic conditions.
  *
  * @param cache_size The total size (in bytes) of the cache to flush, typically covering
  *                   the sum of all cache levels (L1d + L1i + L2 + LLC).
  */
  void flush_cache(size_t cache_size);

  /**
  * @brief Reads and returns the cache size from a specified sysfs path.
  *
  * This function reads the cache size information from the Linux sysfs filesystem,
  * typically from paths like /sys/devices/system/cpu/cpu0/cache/indexN/size.
  * The value is parsed and converted from KB/MB to bytes.
  *
  * @param path The filesystem path to the cache size file (e.g., "/sys/devices/system/cpu/cpu0/cache/index3/size").
  * @return size_t The cache size in bytes, or 0 if the file cannot be read or parsed.
  */
  size_t read_cache_size(const std::string &path);

  /**
  * @brief Retrieves the total cache size by aggregating all cache levels from the system.
  *
  * This function queries the Linux sysfs filesystem to determine the total size of all
  * CPU cache levels (L1 data, L1 instruction, L2, and LLC/L3). It reads cache information
  * from /sys/devices/system/cpu/cpu0/cache/ for each cache index and sums them up.
  * The returned value is used to allocate an appropriate buffer for cache flushing operations.
  *
  * @return size_t The total cache size in bytes across all cache levels, or 0 if unable to determine.
  */
  size_t get_cache_size();
#endif
/**
 * @brief Parses a single command-line argument and updates global benchmarking options.
 *
 * This function examines the provided argument string, determines if it matches any known
 * benchmarking option (such as ndims, iters, data types, kernel name, transpose flags, etc.),
 * and updates the corresponding field in the global_options structure. Returns OK if parsing
 * was successful, NOT_OK otherwise.
 *
 * @param options Reference to the global_options structure to update.
 * @param arg The command-line argument string to parse.
 * @return int OK (0) if parsing was successful, NOT_OK (1) otherwise.
 */
int parseCLArgs(benchdnn::global_options &options, std::string arg);

} // namespace benchdnn
} // namespace zendnnl
#endif