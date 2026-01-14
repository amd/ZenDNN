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

#ifndef _LOWOHA_SOFTMAX_UTILS_HPP
#define _LOWOHA_SOFTMAX_UTILS_HPP

#include <cstdint>
#include <string>
#include "common/zendnnl_global.hpp"
#include "common/logging.hpp"
#include "lowoha_operators/softmax/lowoha_softmax_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace softmax {

using namespace zendnnl::common;

/**
 * @brief Validate Softmax inputs
 *
 * @param input        Input tensor pointer
 * @param output       Output tensor pointer
 * @param params       Softmax parameters
 * @return status_t::success if valid, status_t::failure otherwise
 */
status_t validate_softmax_inputs(
    const void *input,
    const void *output,
    softmax_params &params
);

/**
 * @brief Initialize softmax_params with N-dimensional tensor shape
 *
 * This function populates both the original shape information (shape[], ndims)
 * and the flattened parameters (batch, axis_dim) from an N-dimensional
 * tensor shape. This allows the OneDNN backend to use the full shape while the
 * reference implementation can still use the flattened representation.
 *
 * @param params       Softmax parameters to populate
 * @param shape        Array of tensor dimensions (e.g., [N, C, H, W, D] for 5D)
 * @param ndims        Number of dimensions
 * @param axis         Axis along which to compute softmax (-1 for last axis)
 * @return status_t::success if valid, status_t::failure otherwise
 */
status_t setup_softmax_shape(
    softmax_params &params,
    const uint64_t *shape,
    int ndims,
    int axis
);

} // namespace softmax
} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_SOFTMAX_UTILS_HPP
