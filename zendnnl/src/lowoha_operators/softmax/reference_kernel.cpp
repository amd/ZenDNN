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

#include "reference_kernel.hpp"
#include "common/logging.hpp"
#include "common/bfloat16.hpp"
#include <cmath>
#include <algorithm>
#include <vector>
#include <limits>
#include <omp.h>

namespace zendnnl {
namespace lowoha {
namespace softmax {

// FP32 softmax implementation
void softmax_reference_fp32_impl(
    const float *input,
    float *output,
    const softmax_params &params,
    int num_threads
) {
    const uint64_t axis_size = params.axis_dim;
    const uint64_t outer_size = params.batch;

    #pragma omp parallel for num_threads(num_threads)
    for (uint64_t outer = 0; outer < outer_size; ++outer) {
        // Find max for numerical stability
        float max_val = -std::numeric_limits<float>::infinity();
        for (uint64_t inner = 0; inner < axis_size; ++inner) {
            uint64_t idx = outer * axis_size + inner;
            max_val = std::max(max_val, input[idx]);
        }

        // Compute exp and sum
        std::vector<float> exp_vals(axis_size);
        float sum_exp = 0.0f;
        for (uint64_t i = 0; i < axis_size; ++i) {
            uint64_t idx = outer * axis_size + i;
            exp_vals[i] = std::exp(input[idx] - max_val);
            sum_exp += exp_vals[i];
        }

        // Normalize
        if (params.log_softmax) {
            float log_sum_exp = std::log(sum_exp);
            for (uint64_t i = 0; i < axis_size; ++i) {
                uint64_t idx = outer * axis_size + i;
                output[idx] = input[idx] - max_val - log_sum_exp;
            }
        } else {
            for (uint64_t i = 0; i < axis_size; ++i) {
                uint64_t idx = outer * axis_size + i;
                output[idx] = exp_vals[i] / sum_exp;
            }
        }
    }
}

// BF16 softmax implementation: BF16 -> FP32 -> Softmax -> BF16
//
// For numerical stability, we perform softmax computation in FP32:
// - BF16 has limited precision (7-8 bits mantissa vs 23 bits in FP32)
// - Softmax involves exp() and division operations that accumulate errors
// - Max subtraction and sum operations benefit from FP32 precision
// - Only the final result is converted back to BF16
void softmax_reference_bf16_impl(
    const bfloat16_t *input,
    bfloat16_t *output,
    const softmax_params &params,
    int num_threads
) {
    const uint64_t total_size = params.batch * params.axis_dim;

    // Step 1: Convert BF16 input to FP32 for numerical stability
    std::vector<float> fp32_input(total_size);
    for (uint64_t i = 0; i < total_size; ++i) {
        fp32_input[i] = static_cast<float>(input[i]);
    }

    // Step 2: Run softmax in FP32 (maintains precision in exp/log/sum operations)
    std::vector<float> fp32_output(total_size);
    softmax_reference_fp32_impl(
        fp32_input.data(),
        fp32_output.data(),
        params,
        num_threads
    );

    // Step 3: Convert FP32 output back to BF16
    for (uint64_t i = 0; i < total_size; ++i) {
        output[i] = bfloat16_t(fp32_output[i]);
    }
}

status_t softmax_reference_wrapper(
    const void *input,
    void *output,
    softmax_params &params
) {
    // Calculate flattened parameters from shape
    if (params.ndims <= 0 || params.ndims > SOFTMAX_MAX_NDIMS) {
        log_error("Softmax Reference: Invalid ndims: ", params.ndims,
                  " (must be 1-", SOFTMAX_MAX_NDIMS, "). Use setup_softmax_shape() to populate params.");
        return status_t::failure;
    }

    int normalized_axis = params.axis >= 0 ? params.axis : params.ndims + params.axis;

    // Calculate batch as product of all dimensions except axis_dim
    params.batch = 1;
    for (int i = 0; i < params.ndims; ++i) {
        if (i != normalized_axis) {
            params.batch *= params.shape[i];
        }
    }
    params.axis_dim = params.shape[normalized_axis];

    log_info("Softmax Reference: ", params.ndims, "D tensor, flattened to batch=",
             params.batch, ", axis_dim=", params.axis_dim);

    const int num_threads = params.num_threads > 0 ? params.num_threads :
                            omp_get_max_threads();

    if (params.src_dt == data_type_t::f32) {
        // FP32: Direct computation
        softmax_reference_fp32_impl(
            static_cast<const float*>(input),
            static_cast<float*>(output),
            params,
            num_threads
        );
        return status_t::success;
    } else if (params.src_dt == data_type_t::bf16) {
        // BF16: Convert to FP32, compute, convert back to BF16
        softmax_reference_bf16_impl(
            static_cast<const bfloat16_t*>(input),
            static_cast<bfloat16_t*>(output),
            params,
            num_threads
        );
        return status_t::success;
    }

    log_error("Softmax Reference: Unsupported data type");
    return status_t::failure;
}

} // namespace softmax
} // namespace lowoha
} // namespace zendnnl
