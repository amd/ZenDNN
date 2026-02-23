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

#ifndef _LOWOHA_NORMALIZATION_UTILS_HPP
#define _LOWOHA_NORMALIZATION_UTILS_HPP

#include <cstdint>
#include <string>
#include "common/zendnnl_global.hpp"
#include "common/logging.hpp"
#include "lowoha_operators/normalization/lowoha_normalization_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace normalization {

using namespace zendnnl::common;

/**
 * @brief Validate normalization inputs
 *
 * Checks that input, output, and optional parameter pointers are valid,
 * that data types are supported, and that dimensions are consistent
 * for the selected normalization type.
 *
 * For BatchNorm inference, running_mean and running_var are required.
 *
 * @param input         Input tensor pointer
 * @param output        Output tensor pointer
 * @param gamma         Gamma (scale) parameter pointer (may be nullptr if !use_scale)
 * @param beta          Beta (shift) parameter pointer (may be nullptr if !use_shift or RMSNorm)
 * @param running_mean  Pre-computed running mean (required for BatchNorm, nullptr otherwise)
 * @param running_var   Pre-computed running variance (required for BatchNorm, nullptr otherwise)
 * @param params        Normalization parameters
 * @return status_t::success if valid, status_t::failure otherwise
 */
status_t validate_normalization_inputs(
  const void *input,
  const void *output,
  const void *gamma,
  const void *beta,
  const void *running_mean,
  const void *running_var,
  const norm_params &params
);

/**
 * @brief Convert norm_type_t to a human-readable string
 *
 * @param type  The normalization type enum value
 * @return A string representation (e.g. "LayerNorm", "RMSNorm", "BatchNorm")
 */
std::string norm_type_to_str(norm_type_t type);

} // namespace normalization
} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_NORMALIZATION_UTILS_HPP

