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

#include "lowoha_conv_utils.hpp"
#include <cmath>

namespace zendnnl {
namespace lowoha {
namespace conv {

status_t validate_conv_inputs(
    const void *input,
    const void *filter,
    const void *output,
    conv_params &params
) {
    const conv_dims_t &dims = params.dims;
    // Check for null pointers
    if (!input || !filter || !output) {
        log_error("Conv validation failed: null pointer(s) detected");
        return status_t::failure;
    }

    // Validate dimensions are non-zero
    if (dims.batch == 0 || dims.in_height == 0 ||
        dims.in_width == 0 || dims.in_channels == 0) {
        log_error("Conv validation failed: input dimensions contain zero");
        return status_t::failure;
    }

    if (dims.filter_height == 0 || dims.filter_width == 0 ||
        dims.out_channels == 0) {
        log_error("Conv validation failed: filter dimensions contain zero");
        return status_t::failure;
    }

    // Validate strides are non-zero
    if (params.stride_h == 0 || params.stride_w == 0) {
        log_error("Conv validation failed: strides cannot be zero");
        return status_t::failure;
    }

    // Validate dilations are non-zero
    if (params.dilation_h == 0 || params.dilation_w == 0) {
        log_error("Conv validation failed: dilations cannot be zero");
        return status_t::failure;
    }

    // Validate output dimensions match expected values
    // Formula: output_size = floor((input_size + pad_before + pad_after - effective_filter_size) / stride) + 1
    // where effective_filter_size = dilation * (filter_size - 1) + 1
    uint64_t effective_filter_height = params.dilation_h * (dims.filter_height - 1) + 1;
    uint64_t effective_filter_width = params.dilation_w * (dims.filter_width - 1) + 1;

    uint64_t expected_out_height = (dims.in_height + params.pad_top + params.pad_bottom -
                                    effective_filter_height + params.stride_h) /
                                   params.stride_h;
    uint64_t expected_out_width = (dims.in_width + params.pad_left + params.pad_right -
                                   effective_filter_width + params.stride_w) /
                                  params.stride_w;

    if (dims.out_height != expected_out_height ||
        dims.out_width != expected_out_width) {
        log_error("Conv validation failed: output dimensions mismatch. ",
                  "Expected: [", expected_out_height, ", ",
                  expected_out_width, "], Got: [", dims.out_height, ", ",
                  dims.out_width, "]. ",
                  "Input: [", dims.in_height, ", ", dims.in_width, "], ",
                  "Filter: [", dims.filter_height, ", ", dims.filter_width, "], ",
                  "Stride: [", params.stride_h, ", ", params.stride_w, "], ",
                  "Padding: [", params.pad_top, ", ", params.pad_bottom, ", ",
                  params.pad_left, ", ", params.pad_right, "], ",
                  "Dilation: [", params.dilation_h, ", ", params.dilation_w, "]");
        return status_t::failure;
    }

    log_info("Conv input validation successful");
    return status_t::success;
}

} // namespace conv
} // namespace lowoha
} // namespace zendnnl
