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
#ifndef _SDPA_BENCHDNN_HPP_
#define _SDPA_BENCHDNN_HPP_

#include "example_utils.hpp"
#include "benchdnn.hpp"
#include "sdpa_utils.hpp"
#include "sdpa_tensor_factory.hpp"
#include "sdpa_lowoha.hpp"

namespace zendnnl {
namespace benchdnn {
namespace sdpa {

using namespace zendnnl::interface;
using namespace zendnnl::examples;

/**
 * @brief Runs the SDPA benchmark suite.
 *
 * Reads benchmark configurations from @p in_filename (FILE / MODEL mode) or
 * builds a single configuration from @p options (COMMAND_LINE mode), executes
 * the SDPA benchmarks via the LOWOHA `sdpa_direct` API, prints a formatted
 * results table to stdout, and writes the same data to a CSV file at
 * @p out_filename.
 *
 * @note Only the LOWOHA path is supported. If isLOWOHA is false, the function
 *       returns NOT_OK with an error message (matches the normalization driver).
 *
 * @param in_filename   Path to the input file (ignored for COMMAND_LINE mode).
 * @param out_filename  Path to the output CSV file.
 * @param inputMode     FILE, MODEL, or COMMAND_LINE.
 * @param options       Global CLI options (provides defaults for MODEL mode and
 *                      the single-config values for COMMAND_LINE mode).
 * @param isLOWOHA      Must be true; non-LOWOHA path is intentionally unsupported.
 * @param cache_size    Cache size for cold cache flushing (if enabled).
 * @return int OK (0) on success, NOT_OK (1) on failure.
 */
int bench(const std::string &in_filename, const std::string &out_filename,
          const InputMode inputMode, const global_options &options,
          const bool isLOWOHA, size_t cache_size);

} // namespace sdpa
} // namespace benchdnn
} // namespace zendnnl

#endif
