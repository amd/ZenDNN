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
#ifndef _BENCHDNN_UTILS_HPP_
#define _BENCHDNN_UTILS_HPP_

#include "zendnnl.hpp"

using namespace zendnnl::interface;
#define  OK          (0)
#define  NOT_OK      (1)

#include <emmintrin.h>

// Size (in bytes) of a cache line; typically 64 bytes on x86 systems.
#define CACHE_LINE_SIZE 64

namespace zendnnl {
namespace benchdnn {

enum class CacheMode {
  COLD,
  WARM,
  HOT
};

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
  size_t bs; /**< CLI batch size (B). 0 = unset; required for BMM matmul
                 (--ndims>2) and for SDPA. Ignored for non-batched matmul. */
  size_t m;  /**< CLI matmul M dimension. 0 = unset; required for matmul. */
  size_t k;  /**< CLI matmul K dimension. 0 = unset; required for matmul. */
  std::vector<size_t> n_values; /**< Vector of output columns
                               for each layer (multi-layer support). */
  bool isBiasEnabled; /**< Flag indicating if bias is enabled in the matmul operation. */
  std::vector<zendnnl::ops::post_op_type_t> post_ops; /**< List of post operations
                                                      to apply (e.g., relu, gelu). */
  data_type_t post_op_dt; /**< Datatype of post operation. */
  int ndims; /**< Number of dimensions for tensors (e.g., 2 for standard matmul, 3 for batched matmul). */
  int iters; /**< Number of iterations to run the benchmark. */
  data_type_t sdt; /**< Datatype of input. */
  data_type_t wdt; /**< Datatype of weights. */
  data_type_t ddt; /**< Datatype of destination/output. */
  std::string kernel_name; /**< Name of the kernel to use. */
  int is_weights_const; /**< 0: weights are not constant, 1: weights are constant, -1: not set. */
  data_type_t bias_dt; /**< Datatype of bias. */
  bool isTransA; /**< Transpose flag for input matrix */
  bool isTransB; /**< Transpose flag for weight matrix */
  float alpha, beta; /**< Scaling factors for matmul operation. */
  std::string scale_granularity; /**< Scale granularity for weight quantization. */
  uint64_t group_size; /**< Group size for weight quantization. */
  data_type_t scale_dt; /**< Datatype of weight scale. */
  int warmup_iters; /**< Number of warmup iterations to run before actual benchmarking. */
  bool perf_counters; /**< Enable per-shape HW perf counter collection (AMD Zen 4/5 PMU). */
  std::string perf_profile_str; /**< Perf counter profile:
                                "cache" (default), "tlb", "stalls". */
  /**< Cache: cold, warm, or hot. Warm is matmul-only; main rejects warm for other --op. */
  CacheMode cache_mode;
  int num_weight_buffers; /**< Number of weight buffers to use. */
  bool src_dynamic_quant; /**< Enable dynamic source quantization
                          (matmul W8A8 or W4A8 -> s8 compute). */
  std::string src_scale_granularity; /**< Source scale granularity:
                                     per-tensor | per-token | per-group. */
  uint64_t src_group_size; /**< K-direction group size for per-group source scales. */
  data_type_t src_scale_dt; /**< Datatype of source scale (f32 | bf16). */

  // SDPA-specific options (used by --op=sdpa).
  // Two different conventions are used for the default-constructed values below:
  //   * Required dimensions (num_heads, seq_len, head_dim) use 0 as an "unset"
  //     sentinel; the CLI parser rejects values <= 0 so the SDPA driver can
  //     detect when the user forgot to supply them.
  //   * Every other field treats its default (0 / false / data_type_t::none /
  //     "bhsd") as a real, semantic value -- NOT a sentinel. For example:
  //       kv_seq_len = 0          -> self-attention (use seq_len)
  //       mask_ndims = 0          -> no attention mask
  //       mask_dt    = none       -> no mask dtype
  //       is_causal  = false      -> no causal mask (a valid default)
  //       scale      = 0.0        -> auto = 1 / sqrt(head_dim)
  //       num_threads= 0          -> auto (use all available threads)
  //       out_dt     = none       -> use qkv dtype
  //       qkv_layout = "bhsd"     -> head-major BHSD memory layout
  // See each field's docstring for the exact semantics.
  int64_t num_heads;     /**< SDPA: number of attention heads (H). 0 = unset. */
  int64_t seq_len;       /**< SDPA: query sequence length (S_q). 0 = unset. */
  int64_t kv_seq_len;    /**< SDPA: key/value sequence length. 0 = use seq_len. */
  int64_t head_dim;      /**< SDPA: per-head dimension (D). 0 = unset. */
  int mask_ndims;        /**< SDPA: 0 (none), 2 ([S_q,S_kv]) or 4 ([B,H,S_q,S_kv]). */
  data_type_t mask_dt;   /**< SDPA: mask data type when mask_ndims > 0. */
  bool is_causal;        /**< SDPA: apply causal upper-triangular mask. */
  double scale;          /**< SDPA: softmax scale; 0.0 -> auto = 1/sqrt(D). */
  int32_t num_threads;   /**< SDPA: OpenMP thread count; 0 = auto/all available. */
  data_type_t out_dt;    /**< SDPA: output dtype. `none` -> use qkv_dt. */
  std::string qkv_layout; /**< SDPA: physical Q/K/V layout: "bhsd" (default) or "bshd".
                               Stored as a string here to keep this op-agnostic header
                               independent of SDPA-specific enums; converted to
                               `qkv_layout_t` once at config-build time. */

  global_options() :
    bs(0), m(0), k(0),
    isBiasEnabled(false),
    post_op_dt(data_type_t::f32),
    ndims(2), iters(100),
    sdt(data_type_t::f32), wdt(data_type_t::f32), ddt(data_type_t::f32),
    is_weights_const(-1), bias_dt(data_type_t::f32),
    isTransA(false), isTransB(false),
    alpha(1.0f), beta(0.0f),
    scale_granularity("none"), group_size(0), scale_dt(data_type_t::f32),
    warmup_iters(-1),
    perf_counters(false), perf_profile_str("cache"),
    cache_mode(CacheMode::HOT),
    num_weight_buffers(-1),
    src_dynamic_quant(false), src_scale_granularity("per-tensor"),
    src_group_size(0), src_scale_dt(data_type_t::f32),
    num_heads(0), seq_len(0), kv_seq_len(0), head_dim(0),
    mask_ndims(0), mask_dt(data_type_t::none), is_causal(false),
    scale(0.0), num_threads(0), out_dt(data_type_t::none),
    qkv_layout("bhsd") {}
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
 * @throws std::invalid_argument if @p str is not a recognized concrete dtype.
 */
data_type_t strToDatatype(const std::string &str);

/**
 * @brief Like `strToDatatype()`, but additionally maps the literal string "none"
 *        to `data_type_t::none`.
 *
 * Use this only at parse sites where an unset dtype is a meaningful, well-defined
 * value (e.g. SDPA's `--mask_dt` when `mask_ndims == 0`, or `--out_dt` meaning
 * "default to qkv_dt"). Anywhere else, prefer the strict `strToDatatype()` so
 * malformed input is rejected at parse time.
 *
 * @param str String representation of the data type, or the literal "none".
 * @return data_type_t Corresponding enum value, or `data_type_t::none` for "none".
 * @throws std::invalid_argument if @p str is neither "none" nor a recognized dtype.
 */
data_type_t strToOptionalDatatype(const std::string &str);

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

/**
 * @brief Converts a string representation of a matmul algorithm to its corresponding enum value.
 *
 * This function maps a string such as "aocl_dlp", "onednn", "libxsmm", "batched_sgemm",
 * "auto", "dynamic_dispatch", or "reference" to the corresponding matmul_algo_t enum value.
 * Returns matmul_algo_t::none if the string is not recognized.
 *
 * @param str String representation of the matmul algorithm (e.g., "aocl_dlp", "onednn").
 * @return matmul_algo_t Corresponding enum value, or matmul_algo_t::none if unknown.
 */
matmul_algo_t strToAlgo(std::string str);

/**
 * @brief Converts a matmul_algo_t enum value to its string representation.
 *
 * This function maps a matmul_algo_t value to its corresponding string (e.g., "aocl_dlp",
 * "onednn") for display, logging, or configuration output.
 *
 * @param algo The matmul_algo_t enum value to convert.
 * @return std::string The string representation of the algorithm.
 */
std::string algoToStr(matmul_algo_t algo);

/** @brief List of valid kernel names accepted for matmul benchmarking.
 *
 * Supported names: aocl_dlp_blocked, onednn_blocked, libxsmm_blocked, aocl_dlp,
 * onednn, libxsmm, batched_sgemm, auto, dynamic_dispatch, reference.
 * Used when validating user-specified kernel names.
 */
inline const std::vector<std::string> VALID_KERNEL_NAMES = {
  "aocl_dlp_blocked", "onednn_blocked", "libxsmm_blocked", "aocl_dlp", "onednn", "libxsmm",
  "batched_sgemm", "auto", "dynamic_dispatch", "reference"
};

/**
 * @brief Validates that the given kernel name is supported for matmul.
 * Logs an error and returns false if the name is unknown.
 * @param kernel_name Kernel name to validate.
 * @return true if valid, false if unknown (error is logged).
 */
bool validateMatmulKernelName(const std::string &kernel_name);

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