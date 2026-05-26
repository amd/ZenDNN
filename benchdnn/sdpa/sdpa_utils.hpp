/********************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef _SDPA_UTILS_HPP_
#define _SDPA_UTILS_HPP_

#include "benchdnn.hpp"

namespace zendnnl {
namespace benchdnn {
namespace sdpa {

// Required CSV field count for FILE input mode:
// batch, num_heads, seq_len, kv_seq_len, head_dim, qkv_dt,
// is_causal, mask_ndims, mask_dt, iters
constexpr size_t SDPA_REQUIRED_FIELD_COUNT = 10;

// Forward declaration of `qkv_layout_t` (full definition lives in
// `sdpa_tensor_factory.hpp`). We deliberately do NOT `#include` that header
// here because doing so closes an include cycle: this header is reached via
// `sdpa_utils.hpp -> benchdnn.hpp -> sdpa_benchdnn.hpp -> sdpa_utils.hpp`
// while `sdpa_tensor_factory.hpp` itself is still mid-processing, so its
// include guard would skip the re-include and the enum would be unknown.
// The fixed `: int` underlying type makes the forward declaration sufficient
// for embedding the enum as a struct member; .cpp consumers that need the
// enumerator values (`qkv_layout_t::bhsd`, ...) include
// `sdpa_tensor_factory.hpp` directly.
enum class qkv_layout_t : int;

/**
 * @struct SdpaConfig
 * @brief Holds user-defined parameters for benchmarking a single SDPA configuration.
 *
 * Encapsulates the dimensions of the Q/K/V/O tensors, data types, mask metadata
 * (no-mask / 2D / 4D + dtype + causal flag), scale, dropout, thread count, the
 * physical QKV memory layout (BHSD or BSHD), and per-config iteration counters.
 *
 * @var batch        Batch dimension (B).
 * @var num_heads    Number of attention heads (H).
 * @var seq_len      Query sequence length (S_q).
 * @var kv_seq_len   Key/Value sequence length (S_kv).
 *                   0 means "use seq_len" (self-attention).
 * @var head_dim     Per-head dimension (D).
 * @var qkv_dt       Data type of Q/K/V tensors. Must be f32, bf16 or f16.
 *                   `f16` requires AVX512-FP16 at runtime; the kernel checks
 *                   `zendnnl_platform_info().get_avx512_f16_status()` and
 *                   returns a failure status on unsupported CPUs (the runner
 *                   logs and skips the config in that case).
 * @var out_dt       Data type of output tensor. Must equal qkv_dt or be `none`
 *                   (the runner forces it equal to qkv_dt for the call).
 * @var mask_ndims   Mask rank: 0 (no additive mask), 2 ([S_q, S_kv]), or 4
 *                   ([B, H, S_q, S_kv]).
 * @var mask_dt      Mask data type. Required when mask_ndims > 0:
 *                     - qkv_dt == f32  -> mask_dt must be f32.
 *                     - qkv_dt == bf16 -> mask_dt may be f32 or bf16.
 *                     - qkv_dt == f16  -> mask_dt may be f32 or f16.
 * @var is_causal    Apply a causal upper-triangular mask in addition to
 *                   any additive mask.
 * @var scale        Softmax scale factor. 0.0 means auto = 1/sqrt(head_dim).
 * @var dropout_p    Dropout probability. Currently only 0.0 is supported.
 * @var num_threads  OpenMP thread count for the operator (0 = auto).
 * @var qkv_layout   Physical memory layout for Q/K/V (and output). Defaults to
 *                   `bhsd`. The mask layout is independent of this and is not
 *                   permuted.
 * @var iters        Number of timed iterations.
 * @var warmup_iters Number of warmup iterations (untimed).
 * @var modelName    Model name (only populated in MODEL input mode).
 */
struct SdpaConfig {
  int64_t batch;
  int64_t num_heads;
  int64_t seq_len;
  int64_t kv_seq_len;
  int64_t head_dim;
  zendnnl::common::data_type_t qkv_dt;
  zendnnl::common::data_type_t out_dt;
  int mask_ndims;
  zendnnl::common::data_type_t mask_dt;
  bool is_causal;
  double scale;
  double dropout_p;
  int32_t num_threads;
  // No default initializer here: that would name an enumerator and therefore
  // require `qkv_layout_t` to be a complete type at this point, which it
  // isn't (only forward-declared above to break the include cycle). All three
  // parsers explicitly assign `qkv_layout` when constructing an SdpaConfig,
  // so a default is unnecessary.
  qkv_layout_t qkv_layout;
  int iters;
  int warmup_iters;
  std::string modelName;
};

/**
 * @brief Converts a layout string to its enum value.
 *
 * Accepts (case-insensitive): "bhsd", "bshd". Throws std::invalid_argument on
 * any other input -- callers in the parsers translate this into a "skip this
 * config" diagnostic.
 */
qkv_layout_t strToQkvLayout(const std::string &str);

/**
 * @brief Inverse of `strToQkvLayout` for printing/logging.
 */
std::string qkvLayoutToStr(qkv_layout_t layout);

/**
 * @brief Returns true if the (qkv_dt, mask_ndims, mask_dt) triple is supported.
 *
 * Mirrors the validation rules in
 * `lowoha_flash_sdpa_utils.cpp::validate_flash_sdpa_inputs` so we can fail fast
 * before allocating tensors.  `reason` is populated with a human-readable
 * explanation when the combination is rejected.
 *
 * @param cfg Config to validate.
 * @param reason Out-parameter populated with the rejection reason on failure.
 * @return true if the combination is supported by sdpa_direct.
 */
bool isSupportedSdpaConfig(const SdpaConfig &cfg, std::string &reason);

/**
 * @brief Parses the FILE-mode input file into a vector of SdpaConfig.
 *
 * Format (CSV, one config per line; '#' comments and blank lines ignored):
 *   batch, num_heads, seq_len, kv_seq_len, head_dim, qkv_dt,
 *   is_causal, mask_ndims, mask_dt, iters
 *   [, warmup_iters, scale, num_threads, out_dt, qkv_layout]
 *
 * Optional trailing fields (in positional order; omit any field by leaving it
 * empty or by truncating the line at the previous field):
 *   - warmup_iters : int      (default: 0.2 * iters)
 *   - scale        : double   (default: 0.0  -> auto = 1 / sqrt(head_dim))
 *   - num_threads  : int      (default: 0    -> auto / OMP_NUM_THREADS)
 *   - out_dt       : string   (default: none -> use qkv_dt)
 *   - qkv_layout   : string   (default: bhsd -- "bhsd" or "bshd")
 *
 * Keep this list in sync with `benchdnn/doc/sdpa.md` (the user-facing docs)
 * and with the parsing logic in `inputFileParser()` in `sdpa_utils.cpp`.
 */
void inputFileParser(std::ifstream &infile, std::vector<SdpaConfig> &configs);

/**
 * @brief Builds a single SdpaConfig from CLI-provided global_options.
 */
void inputCommandLineParser(std::vector<SdpaConfig> &configs,
                            const global_options &options);

/**
 * @brief Parses a model-shapes file into SdpaConfig list using global_options
 *        for everything not specified in the file.
 *
 * Format per line (CSV, '#' comments and blank lines ignored):
 *   ModelName, batch, num_heads, seq_len, kv_seq_len, head_dim
 */
void inputModelFileParser(std::ifstream &infile,
                          std::vector<SdpaConfig> &configs,
                          const global_options &options);

/**
 * @brief Logs a detailed error message for a failed benchmark configuration.
 */
void log_benchmark_failure(const SdpaConfig &cfg);

/**
 * @brief Prints SDPA benchmark results in a formatted table to outfile.
 */
void print_results(std::vector<std::pair<SdpaConfig, TimingStats>>
                   &sdpa_results, std::ostream &outfile, const InputMode inputMode);

/**
 * @brief Logs SDPA benchmark results in CSV format to outfile.
 */
void log_results(std::vector<std::pair<SdpaConfig, TimingStats>>
                 &sdpa_results, std::ostream &outfile, const InputMode inputMode);

} // namespace sdpa
} // namespace benchdnn
} // namespace zendnnl

#endif
