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
#ifndef _GDN_LOWOHA_HPP_
#define _GDN_LOWOHA_HPP_

#include "benchdnn.hpp"
#include "gdn_tensor_factory.hpp"
#include "gdn_utils.hpp"

#include "lowoha_operators/gdn/lowoha_gdn.hpp"

namespace zendnnl {
namespace benchdnn {
namespace gdn {

// Required forward declaration: benchdnn.hpp includes this header while
// gdn_utils.hpp (included above) is still being processed when a translation
// unit starts from gdn_utils.hpp, so GdnConfig may not be defined yet here.
struct GdnConfig;

using zendnnl::lowoha::gdn::gdn_direct;
using zendnnl::lowoha::gdn::gdn_params;

/**
 * @brief Benchmarks the Low Overhead API GDN (gdn_direct).
 *
 * For each configuration in @p configs:
 *   1. Allocates the activations, weights and one conv / SSM state slot per
 *      sequence (shuffled slot ids), in vLLM's layouts.
 *   2. Builds the batch description as vLLM does: num_decodes single-token
 *      decodes first, then the prefills, as cu_seqlens with
 *      has_initial_state per sequence.
 *   3. Runs `warmup_iters` untimed calls and `iters` timed calls of
 *      gdn_direct, each state pool advancing in place.
 *   4. Appends a (cfg, TimingStats) entry to @p gdn_results.
 *
 * @param configs     Configurations to run.
 * @param gdn_results Output vector of (cfg, TimingStats) pairs.
 * @param options     Global options. With `CacheMode::COLD` (the GDN default
 *                    when --cache_mode is not given) the caches are flushed
 *                    before every timed call, so the state pools stream from
 *                    memory as they do between the layers of a model;
 *                    `CacheMode::HOT` runs the calls back to back.
 * @param cache_size  Cache size in bytes used by `flush_cache()`.
 * @return int OK (0) on success, NOT_OK (1) on fatal failure (per-config
 *             failures are logged and skipped).
 */
int gdn_lowoha_benchdnn(const std::vector<GdnConfig> &configs,
        std::vector<std::pair<GdnConfig, TimingStats>> &gdn_results,
        const global_options &options, size_t cache_size);

} // namespace gdn
} // namespace benchdnn
} // namespace zendnnl

#endif
