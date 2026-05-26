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
#ifndef _SDPA_LOWOHA_HPP_
#define _SDPA_LOWOHA_HPP_

#include "benchdnn.hpp"
#include "sdpa_utils.hpp"
#include "sdpa_tensor_factory.hpp"

#include "lowoha_operators/sdpa/lowoha_sdpa.hpp"
#include "lowoha_operators/sdpa/lowoha_sdpa_common.hpp"

namespace zendnnl {
namespace benchdnn {
namespace sdpa {

// Forward declaration is REQUIRED here, not optional.
//
// `benchdnn.hpp` (which is included transitively from `sdpa_utils.hpp`) drags
// in this header before `sdpa_utils.hpp` has finished defining `SdpaConfig`:
//
//   sdpa_utils.hpp  ->  benchdnn.hpp  ->  sdpa_benchdnn.hpp  ->  sdpa_lowoha.hpp
//                                                                    |
//                                                                    +-- needs SdpaConfig
//                                                                        in std::vector<>
//
// The `#include "sdpa_utils.hpp"` above is guarded out during that inner
// re-entry, so `SdpaConfig` would be unknown if we did not forward-declare it
// here. The full definition still arrives via `sdpa_utils.hpp` for any
// translation unit that compiles `sdpa_lowoha.cpp` directly.
struct SdpaConfig;

using zendnnl::lowoha::sdpa::sdpa_direct;
using zendnnl::lowoha::sdpa::sdpa_params;
using zendnnl::lowoha::sdpa::mask_type_t;

/**
 * @brief Benchmarks Low Overhead API SDPA (sdpa_direct).
 *
 * For each configuration in @p configs:
 *   1. Validates the (qkv_dt, mask_ndims, mask_dt) combination.
 *   2. Allocates Q/K/V/O tensors (and optional mask) in the requested QKV
 *      layout (BHSD or BSHD; see `qkv_layout_t`).
 *   3. Builds an `sdpa_params` struct with derived strides + mask metadata.
 *   4. Runs `warmup_iters` warmup calls and `iters` timed calls of
 *      `sdpa_direct`.
 *   5. Records total elapsed time in `TimingStats` and appends a (cfg, stats)
 *      entry to @p sdpa_results.
 *
 * @param configs Vector of configurations to run.
 * @param sdpa_results Output vector of (cfg, TimingStats) pairs.
 * @param options Global benchmarking options. The runner reads
 *                `options.cache_mode` to decide whether to flush the cache
 *                between timed iterations (`CacheMode::COLD`) or to leave it
 *                untouched (`CacheMode::HOT`). `CacheMode::WARM` is rejected
 *                by `main()` for non-matmul ops, so SDPA only ever sees
 *                `COLD` or `HOT`.
 * @param cache_size Cache size in bytes used by `flush_cache()` when
 *                   `options.cache_mode == CacheMode::COLD`.
 * @return int OK (0) on success, NOT_OK (1) on fatal failure (per-config
 *             failures are logged and skipped, not propagated).
 */
int sdpa_lowoha_benchdnn(
  std::vector<SdpaConfig> configs,
  std::vector<std::pair<SdpaConfig, TimingStats>> &sdpa_results,
  const global_options &options,
  size_t cache_size);

} // namespace sdpa
} // namespace benchdnn
} // namespace zendnnl

#endif
