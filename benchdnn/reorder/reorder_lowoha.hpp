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
#ifndef _REORDER_LOWOHA_HPP_
#define _REORDER_LOWOHA_HPP_

#include "benchdnn.hpp"
#include "reorder_utils.hpp"
#include "reorder_tensor_factory.hpp"

using zendnnl::lowoha::reorder::reorder_params_t;
using zendnnl::lowoha::reorder::reorder_quant_params_t;
using zendnnl::lowoha::reorder::reorder_algo_t;
using zendnnl::lowoha::reorder::reorder_direct;

namespace zendnnl {
namespace benchdnn {
namespace reorder {

/**
 * @brief Benchmarks Low Overhead API (LOWOHA) reorder operation.
 *
 * For each configuration, creates tensors via tensor_factory, extracts raw
 * pointers, builds reorder_params_t, and calls reorder_direct().
 *
 * @param configs Vector of ReorderConfig objects for LOWOHA benchmarking.
 * @param reorder_results Vector to store (ReorderConfig, TimingStats) pairs.
 * @param cache_size Cache size for cold cache flushing (if enabled).
 * @return int Returns OK (0) on success, NOT_OK (1) on failure.
 */
int reorder_lowoha_benchdnn(const std::vector<ReorderConfig> &configs,
                            std::vector<std::pair<ReorderConfig, TimingStats>> &reorder_results,
                            size_t cache_size);

} // namespace reorder
} // namespace benchdnn
} // namespace zendnnl

#endif
