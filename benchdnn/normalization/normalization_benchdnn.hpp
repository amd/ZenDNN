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
#ifndef _NORMALIZATION_BENCHDNN_HPP_
#define _NORMALIZATION_BENCHDNN_HPP_

#include "example_utils.hpp"
#include "benchdnn.hpp"
#include "normalization_utils.hpp"
#include "normalization_tensor_factory.hpp"
#include "normalization_lowoha.hpp"

namespace zendnnl {
namespace benchdnn {
namespace normalization {

using namespace zendnnl::interface;
using namespace zendnnl::examples;

/**
 * @brief Runs the full normalization benchmark suite from an input file and writes results to a CSV file.
 *
 * Reads benchmark configurations from the specified input file, executes normalization
 * benchmarks via the LOWOHA API, collects timing and performance statistics, and writes
 * the results to the specified output CSV file.
 *
 * @note Only the LOWOHA (Low Overhead API) path is supported. If isLOWOHA is false,
 *       the function returns NOT_OK with an error message.
 *
 * @param in_filename Path to the input file containing benchmark configurations.
 * @param out_filename Path to the output CSV file for writing results.
 * @param isLOWOHA If true, runs the LOWOHA (Low Overhead API) benchmark variant.
 * @param cache_size Cache size for cold cache flushing (if enabled).
 * @return int Returns OK (0) on success, NOT_OK (1) on failure.
 */
int bench(const std::string &in_filename, const std::string &out_filename,
          const bool isLOWOHA, size_t cache_size);

} // namespace normalization
} // namespace benchdnn
} // namespace zendnnl

#endif
