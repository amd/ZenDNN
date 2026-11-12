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
#ifndef _GDN_UTILS_HPP_
#define _GDN_UTILS_HPP_

#include "benchdnn.hpp"
#include "lowoha_operators/gdn/lowoha_gdn_common.hpp"

namespace zendnnl {
namespace benchdnn {
namespace gdn {

using zendnnl::lowoha::gdn::gdn_kernel_t;
using zendnnl::lowoha::gdn::gdn_prefill_algo_t;
using zendnnl::lowoha::gdn::gdn_state_layout_t;

// Required CSV fields per line:
// model, num_k_heads, num_v_heads, head_k_dim, head_v_dim, conv_width,
// num_decodes, num_prefills, prefill_len
constexpr size_t GDN_REQUIRED_FIELD_COUNT = 9;

/**
 * @struct GdnConfig
 * @brief One GDN benchmark case: a layer shape and one vLLM-style batch of
 *        num_decodes single-token decodes followed by num_prefills prefills
 *        of prefill_len tokens each, run as one gdn_direct call.
 *
 * @var model             Label printed with the results (e.g. the model name).
 * @var num_k_heads       Key / query heads (linear_num_key_heads).
 * @var num_v_heads       Value heads (linear_num_value_heads).
 * @var head_k_dim        Key head dim (linear_key_head_dim).
 * @var head_v_dim        Value head dim (linear_value_head_dim).
 * @var conv_width        Causal conv1d width (linear_conv_kernel_dim).
 * @var num_decodes       Decode sequences, one token each, leading the batch.
 * @var num_prefills      Prefill sequences after the decodes.
 * @var prefill_len       Tokens per prefill sequence.
 * @var iters             Timed iterations.
 * @var warmup_iters      Untimed warmup iterations.
 * @var src_dt            Activations, conv weights and conv state: f32, bf16
 *                        or f16.
 * @var state_dt          SSM state pool: f32, bf16 or f16.
 * @var has_initial_state Prefills continue from their slot (true) or start
 *                        from zero states (false); decodes always continue.
 * @var state_layout      SSM state matrix layout: kv (vLLM CPU) or vk (FLA).
 * @var fused_norm        Fuse the gated RMSNorm (silu gate) on the output.
 * @var prefill_algo      Native prefill algorithm: automatic, recurrent or
 *                        chunked.
 * @var kernel            Kernel request: none (automatic), native or
 *                        reference.
 * @var num_threads       OpenMP threads of the call (0 = all).
 */
struct GdnConfig {
    std::string model;
    int64_t num_k_heads;
    int64_t num_v_heads;
    int64_t head_k_dim;
    int64_t head_v_dim;
    int64_t conv_width;
    int64_t num_decodes;
    int64_t num_prefills;
    int64_t prefill_len;
    int iters;
    int warmup_iters;
    data_type_t src_dt;
    data_type_t state_dt;
    bool has_initial_state;
    gdn_state_layout_t state_layout;
    bool fused_norm;
    gdn_prefill_algo_t prefill_algo;
    gdn_kernel_t kernel;
    int32_t num_threads;

    int64_t batch() const { return num_decodes + num_prefills; }
    int64_t num_tokens() const {
        return num_decodes + num_prefills * prefill_len;
    }
};

/**
 * @brief Returns true if @p cfg describes a call gdn_direct accepts; fills
 *        @p reason otherwise, so the driver fails fast before allocating.
 */
bool isSupportedGdnConfig(const GdnConfig &cfg, std::string &reason);

/**
 * @brief Parses the GDN input file into a vector of GdnConfig.
 *
 * Format (CSV, one config per line; '#' comments and blank lines ignored):
 *   model, num_k_heads, num_v_heads, head_k_dim, head_v_dim, conv_width,
 *   num_decodes, num_prefills, prefill_len
 *   [, iters, warmup_iters, src_dt, state_dt, has_initial_state,
 *      state_layout, fused_norm, prefill_algo, kernel, num_threads]
 *
 * Optional trailing fields (positional; leave a field empty or truncate the
 * line to take the default):
 *   - iters             : int    (default: --iters, 100)
 *   - warmup_iters      : int    (default: --warmup_iters, else iters / 5)
 *   - src_dt            : string (default: --src_dt, else bf16)
 *   - state_dt          : string (default: --state_dt, else f32)
 *   - has_initial_state : bool   (default: true)
 *   - state_layout      : kv | vk (default: kv)
 *   - fused_norm        : bool   (default: false)
 *   - prefill_algo      : auto | recurrent | chunked (default: auto)
 *   - kernel            : auto | native | reference (default: auto)
 *   - num_threads       : int    (default: --num_threads, 0 = all)
 *
 * Keep this list in sync with `benchdnn/doc/gdn.md`.
 */
void inputFileParser(std::ifstream &infile, std::vector<GdnConfig> &configs,
        const global_options &options);

/**
 * @brief Logs a detailed error message for a failed benchmark configuration.
 */
void log_benchmark_failure(const GdnConfig &cfg);

/**
 * @brief Prints GDN benchmark results in a formatted table to outfile.
 */
void print_results(std::vector<std::pair<GdnConfig, TimingStats>> &results,
        std::ostream &outfile);

/**
 * @brief Logs GDN benchmark results in CSV format to outfile.
 */
void log_results(std::vector<std::pair<GdnConfig, TimingStats>> &results,
        std::ostream &outfile);

} // namespace gdn
} // namespace benchdnn
} // namespace zendnnl

#endif
