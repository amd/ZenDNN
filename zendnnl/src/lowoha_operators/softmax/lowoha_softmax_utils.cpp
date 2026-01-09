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

status_t calculate_softmax_dims(
    const int64_t *tensor_shape,
    int ndims,
    softmax_params &params
) {
    if (tensor_shape == nullptr || ndims <= 0) {
        log_error("Softmax: Invalid tensor shape");
        return status_t::failure;
    }

    // Normalize axis to positive index
    int normalized_axis = params.axis;
    if (params.axis < 0) {
        normalized_axis = ndims + params.axis;
    }

    if (normalized_axis < 0 || normalized_axis >= ndims) {
        log_error("Softmax: Axis out of range");
        return status_t::failure;
    }

    // Calculate outer dimensions (batch)
    params.batch = 1;
    for (int i = 0; i < normalized_axis; ++i) {
        params.batch *= tensor_shape[i];
    }

    // Axis dimension
    params.axis_dim = tensor_shape[normalized_axis];

    // Calculate inner dimensions
    params.inner_size = 1;
    for (int i = normalized_axis + 1; i < ndims; ++i) {
        params.inner_size *= tensor_shape[i];
    }

    return status_t::success;
}

} // namespace softmax
} // namespace lowoha
} // namespace zendnnl
