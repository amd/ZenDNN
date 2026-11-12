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
#ifndef _GDN_BENCHDNN_HPP_
#define _GDN_BENCHDNN_HPP_

#include "benchdnn.hpp"
#include "gdn_lowoha.hpp"
#include "gdn_tensor_factory.hpp"
#include "gdn_utils.hpp"

namespace zendnnl {
namespace benchdnn {
namespace gdn {

/**
 * @brief Runs the GDN (Gated DeltaNet) benchmark suite.
 *
 * Reads benchmark configurations from @p in_filename (FILE or MODEL mode;
 * both take the format of `benchdnn/input/gdn/gdn_models.txt`), runs each as
 * one LOWOHA `gdn_direct` call over a batch of decodes followed by prefills,
 * prints a results table to stdout and writes the same data to a CSV file
 * at @p out_filename.
 *
 * @note Only the LOWOHA path is supported; COMMAND_LINE mode is rejected by
 *       main() for this operator.
 *
 * @param in_filename   Path to the input file.
 * @param out_filename  Path to the output CSV file.
 * @param inputMode     FILE or MODEL.
 * @param options       Global CLI options (defaults for iters, warmup_iters,
 *                      num_threads and the cache mode).
 * @param isLOWOHA      Must be true.
 * @param cache_size    Cache size for cold cache flushing.
 * @return int OK (0) on success, NOT_OK (1) on failure.
 */
int bench(const std::string &in_filename, const std::string &out_filename,
        const InputMode inputMode, const global_options &options,
        const bool isLOWOHA, size_t cache_size);

} // namespace gdn
} // namespace benchdnn
} // namespace zendnnl

#endif
