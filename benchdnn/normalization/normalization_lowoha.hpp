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
#ifndef _NORMALIZATION_LOWOHA_HPP_
#define _NORMALIZATION_LOWOHA_HPP_

#include "benchdnn.hpp"
#include "normalization_utils.hpp"
#include "normalization_tensor_factory.hpp"

using zendnnl::lowoha::normalization::norm_params;
using zendnnl::lowoha::normalization::normalization_direct;

namespace zendnnl {
namespace benchdnn {
namespace normalization {

/**
 * @brief Benchmarks Low Overhead API normalization.
 *
 * For each configuration in the provided vector, creates the necessary tensors,
 * performs warmup and measured iterations calling the LOWOHA normalization_direct API,
 * collects timing statistics, and stores the results in the provided vector.
 *
 * @param configs Vector of NormalizationConfig objects specifying benchmark parameters for each run.
 * @param normalization_results Vector to store pairs of configuration and timing statistics.
 * @param cache_size Cache size for cold cache flushing (if enabled).
 * @return int Returns OK (0) on success, NOT_OK (1) on failure.
 */
int normalization_lowoha_benchdnn(
  std::vector<NormalizationConfig> configs,
  std::vector<std::pair<NormalizationConfig, TimingStats>> &normalization_results,
  size_t cache_size);

} // namespace normalization
} // namespace benchdnn
} // namespace zendnnl

#endif
