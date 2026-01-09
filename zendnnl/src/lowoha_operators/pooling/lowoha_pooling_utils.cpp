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

#include "lowoha_pooling_utils.hpp"
#include <sstream>

namespace zendnnl {
namespace lowoha {
namespace pooling {

status_t validate_pooling_inputs(
    const void *input,
    const void *output,
    pool_params &params
) {
    // Check for null pointers
    if (input == nullptr) {
        log_error("Pooling: Input pointer is null");
        return status_t::failure;
    }
    
    if (output == nullptr) {
        log_error("Pooling: Output pointer is null");
        return status_t::failure;
    }
    
    // Validate dimensions
    if (params.dims.batch == 0 || params.dims.in_height == 0 || params.dims.in_width == 0 || params.dims.channels == 0) {
        log_error("Pooling: Invalid input dimensions");
        return status_t::failure;
    }
    
    if (params.dims.kernel_height == 0 || params.dims.kernel_width == 0) {
        log_error("Pooling: Invalid kernel dimensions");
        return status_t::failure;
    }
    
    if (params.stride_h == 0 || params.stride_w == 0) {
        log_error("Pooling: Invalid stride values");
        return status_t::failure;
    }
    
    if (params.dims.out_height == 0 || params.dims.out_width == 0) {
        log_error("Pooling: Invalid output dimensions");
        return status_t::failure;
    }
    
    return status_t::success;
}

} // namespace pooling
} // namespace lowoha
} // namespace zendnnl
