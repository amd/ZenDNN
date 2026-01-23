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
#ifndef _EMBAG_LOWOHA_HPP_
#define _EMBAG_LOWOHA_HPP_

#include "benchdnn.hpp"
#include "embag_utils.hpp"
#include "embag_tensor_factory.hpp"

// Include LOWOHA embedding bag API
using zendnnl::lowoha::embag::embag_data_types_t;
using zendnnl::lowoha::embag::embag_params_t;
using zendnnl::lowoha::embag::embedding_bag_direct;

namespace zendnnl {
namespace benchdnn {
namespace embag {

/**
 * @brief Benchmarks Low Overhead API embedding bag (embag).
 *
 * For each configuration in the provided vector, performs warmup and measured iterations,
 * collects timing statistics, and stores the results in the provided vector.
 *
 * @param configs Vector of EmbagConfig objects specifying benchmark parameters for each run.
 * @param embag_results Vector to store pairs of configuration and timing statistics.
 * @param cache_size Cache size for cold cache flushing (if enabled).
 * @return int Returns OK (0) on success, NOT_OK (1) on failure.
 */
int embag_lowoha_benchdnn(std::vector<EmbagConfig> configs,
                          std::vector<std::pair<EmbagConfig, TimingStats>> &embag_results,
                          size_t cache_size);

} // namespace embag
} // namespace benchdnn
} // namespace zendnnl

#endif
