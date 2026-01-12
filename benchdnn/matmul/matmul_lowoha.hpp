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
#ifndef _MATMUL_LOWOHA_HPP_
#define _MATMUL_LOWOHA_HPP_

#include "benchdnn.hpp"

using zendnnl::lowoha::matmul::matmul_data_types;
using zendnnl::lowoha::matmul::matmul_params;
using zendnnl::lowoha::matmul::matmul_batch_params_t;
using zendnnl::lowoha::matmul::matmul_direct;

namespace zendnnl {
namespace benchdnn {
namespace matmul {

/**
 * @brief Benchmarks Low Overhead API matrix multiplication (matmul).
 *
 * For each configuration in the provided vector, performs warmup and measured iterations,
 * collects timing statistics, and stores the results in the provided vector.
 *
 * @param configs Vector of MatmulConfig objects specifying benchmark parameters for each run.
 * @param matmul_results Vector to store pairs of configuration and timing statistics (std::pair<MatmulConfig, std::vector<TimingStats>>).
 *        Each entry corresponds to a unique configuration and its associated vector of timing results for all runs.
 * @param options Global benchmarking options and settings.
 * @return int Returns OK (0) on success, NOT_OK (1) on failure.
 */
int matmul_lowoha_benchdnn(std::vector<MatmulConfig> configs,
                           std::vector<std::pair<MatmulConfig, std::vector<TimingStats>>> &matmul_results,
                           const global_options &options,
                           size_t cache_size);

} // namespace matmul
} // namespace benchdnn
} // namespace zendnnl

#endif