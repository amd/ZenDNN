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

#include "lowoha_softmax_utils.hpp"
#include <sstream>

namespace zendnnl {
namespace lowoha {
namespace softmax {

status_t validate_softmax_inputs(
    const void *input,
    const void *output,
    softmax_params &params
) {
    // Check for null pointers
    if (input == nullptr) {
        log_error("Softmax: Input pointer is null");
        return status_t::failure;
    }

    if (output == nullptr) {
        log_error("Softmax: Output pointer is null");
        return status_t::failure;
    }

    // Validate dimensions
    if (params.axis_dim == 0) {
        log_error("Softmax: Axis dimension cannot be zero");
        return status_t::failure;
    }

    if (params.batch == 0) {
        log_error("Softmax: Batch size cannot be zero");
        return status_t::failure;
    }

    return status_t::success;
}

status_t setup_softmax_shape(
    softmax_params &params,
    const uint64_t *shape,
    int ndims,
    int axis
) {
    // Validate inputs
    if (shape == nullptr) {
        log_error("Softmax: Shape pointer is null");
        return status_t::failure;
    }

    if (ndims <= 0 || ndims > SOFTMAX_MAX_NDIMS) {
        log_error("Softmax: Invalid number of dimensions: ", ndims,
                  " (must be between 1 and ", SOFTMAX_MAX_NDIMS, " for 5D support)");
        return status_t::failure;
    }

    // Normalize axis to positive value
    int normalized_axis = axis >= 0 ? axis : ndims + axis;
    if (normalized_axis < 0 || normalized_axis >= ndims) {
        log_error("Softmax: Invalid axis: ", axis, " for ", ndims, "D tensor");
        return status_t::failure;
    }

    // Store original shape information
    params.ndims = ndims;
    params.axis = normalized_axis;
    for (int i = 0; i < ndims; ++i) {
        if (shape[i] == 0) {
            log_error("Softmax: Dimension ", i, " has zero size");
            return status_t::failure;
        }
        params.shape[i] = shape[i];
    }

    // Fill batch, axis_dim with original shape values (not flattened)
    // These will be recalculated by backends that need flattened representation
    params.batch = (normalized_axis > 0) ? shape[0] : 1;
    params.axis_dim = shape[normalized_axis];

    log_info("Softmax: Setup ", ndims, "D tensor with shape=[", shape[0]);
    for (int i = 1; i < ndims; ++i) {
        log_info(",", shape[i]);
    }
    log_info("], axis=", normalized_axis,
             " -> batch=", params.batch,
             ", axis_dim=", params.axis_dim);

    return status_t::success;
}

} // namespace softmax
} // namespace lowoha
} // namespace zendnnl
