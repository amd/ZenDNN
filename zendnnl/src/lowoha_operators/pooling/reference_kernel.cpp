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
#include <limits>
#include <omp.h>

namespace zendnnl {
namespace lowoha {
namespace pooling {

using namespace zendnnl::common;

template<typename T>
void max_pooling_reference_impl(
    const T *input,
    T *output,
    const pool_params &params,
    const int num_threads
) {
    const uint64_t batch = params.dims.batch;
    const uint64_t in_h = params.dims.in_height;
    const uint64_t in_w = params.dims.in_width;
    const uint64_t channels = params.dims.channels;
    const uint64_t out_h = params.dims.out_height;
    const uint64_t out_w = params.dims.out_width;
    const uint64_t kh = params.dims.kernel_height;
    const uint64_t kw = params.dims.kernel_width;
    const uint32_t stride_h = params.stride_h;
    const uint32_t stride_w = params.stride_w;
    const int32_t pad_t = params.pad_top;
    const int32_t pad_l = params.pad_left;

    // NHWC format: [batch, height, width, channels]
    // Parallelize over batch, output height, and output width dimensions
    #pragma omp parallel for collapse(3) num_threads(num_threads)
    for (uint64_t n = 0; n < batch; ++n) {
        for (uint64_t oh = 0; oh < out_h; ++oh) {
            for (uint64_t ow = 0; ow < out_w; ++ow) {
                for (uint64_t c = 0; c < channels; ++c) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    
                    // Calculate input region for this pooling window
                    int64_t h_start = static_cast<int64_t>(oh * stride_h) - pad_t;
                    int64_t w_start = static_cast<int64_t>(ow * stride_w) - pad_l;
                    
                    // Iterate over pooling window
                    for (uint64_t kh_idx = 0; kh_idx < kh; ++kh_idx) {
                        for (uint64_t kw_idx = 0; kw_idx < kw; ++kw_idx) {
                            int64_t ih = h_start + kh_idx;
                            int64_t iw = w_start + kw_idx;
                            
                            // Check bounds
                            if (ih >= 0 && ih < static_cast<int64_t>(in_h) &&
                                iw >= 0 && iw < static_cast<int64_t>(in_w)) {
                                // NHWC layout: index = n*H*W*C + h*W*C + w*C + c
                                uint64_t in_idx = n * in_h * in_w * channels +
                                                 ih * in_w * channels +
                                                 iw * channels +
                                                 c;
                                float val = static_cast<float>(input[in_idx]);
                                max_val = std::max(max_val, val);
                            }
                        }
                    }
                    
                    // Write output
                    uint64_t out_idx = n * out_h * out_w * channels +
                                      oh * out_w * channels +
                                      ow * channels +
                                      c;
                    output[out_idx] = static_cast<T>(max_val);
                }
            }
        }
    }
}

template<typename T>
void avg_pooling_reference_impl(
    const T *input,
    T *output,
    const pool_params &params,
    const int num_threads
) {
    const uint64_t batch = params.dims.batch;
    const uint64_t in_h = params.dims.in_height;
    const uint64_t in_w = params.dims.in_width;
    const uint64_t channels = params.dims.channels;
    const uint64_t out_h = params.dims.out_height;
    const uint64_t out_w = params.dims.out_width;
    const uint64_t kh = params.dims.kernel_height;
    const uint64_t kw = params.dims.kernel_width;
    const uint32_t stride_h = params.stride_h;
    const uint32_t stride_w = params.stride_w;
    const int32_t pad_t = params.pad_top;
    const int32_t pad_l = params.pad_left;
    const bool include_padding = (params.avg_mode == avg_pooling_mode_t::include_padding);

    // NHWC format: [batch, height, width, channels]
    // Parallelize over batch, output height, and output width dimensions
    #pragma omp parallel for collapse(3) num_threads(num_threads)   
    for (uint64_t n = 0; n < batch; ++n) {
        for (uint64_t oh = 0; oh < out_h; ++oh) {
            for (uint64_t ow = 0; ow < out_w; ++ow) {
                for (uint64_t c = 0; c < channels; ++c) {
                    float sum = 0.0f;
                    uint64_t valid_count = 0;
                    
                    // Calculate input region for this pooling window
                    int64_t h_start = static_cast<int64_t>(oh * stride_h) - pad_t;
                    int64_t w_start = static_cast<int64_t>(ow * stride_w) - pad_l;
                    
                    // Iterate over pooling window
                    for (uint64_t kh_idx = 0; kh_idx < kh; ++kh_idx) {
                        for (uint64_t kw_idx = 0; kw_idx < kw; ++kw_idx) {
                            int64_t ih = h_start + kh_idx;
                            int64_t iw = w_start + kw_idx;
                            
                            // Check bounds
                            if (ih >= 0 && ih < static_cast<int64_t>(in_h) &&
                                iw >= 0 && iw < static_cast<int64_t>(in_w)) {
                                // NHWC layout: index = n*H*W*C + h*W*C + w*C + c
                                uint64_t in_idx = n * in_h * in_w * channels +
                                                 ih * in_w * channels +
                                                 iw * channels +
                                                 c;
                                sum += static_cast<float>(input[in_idx]);
                                valid_count++;
                            }
                        }
                    }
                    
                    // Write output (average)
                    uint64_t out_idx = n * out_h * out_w * channels +
                                      oh * out_w * channels +
                                      ow * channels +
                                      c;
                    
                    // Calculate divisor based on padding mode
                    uint64_t divisor;
                    if (include_padding) {
                        // Include padding: divide by total kernel size
                        divisor = kh * kw;
                    } else {
                        // Exclude padding: divide by valid element count
                        divisor = valid_count;
                    }
                    
                    output[out_idx] = static_cast<T>(divisor > 0 ? sum / divisor : 0.0f);
                }
            }
        }
    }
}

status_t pooling_reference_wrapper(
    const void *input,
    void *output,
    pool_params &params
) {
    const int num_threads = params.num_threads > 0 ? params.num_threads :
                            omp_get_max_threads();

    if (params.dtypes.src == data_type_t::f32) {
        if (params.is_max_pooling) {
            max_pooling_reference_impl<float>(
                static_cast<const float*>(input),
                static_cast<float*>(output),
                params,
                num_threads
            );
        } else {
            avg_pooling_reference_impl<float>(
                static_cast<const float*>(input),
                static_cast<float*>(output),
                params,
                num_threads
            );
        }
        return status_t::success;
    } else if (params.dtypes.src == data_type_t::bf16) {
        if (params.is_max_pooling) {
            max_pooling_reference_impl<bfloat16_t>(
                static_cast<const bfloat16_t*>(input),
                static_cast<bfloat16_t*>(output),
                params,
                num_threads
            );
        } else {
            avg_pooling_reference_impl<bfloat16_t>(
                static_cast<const bfloat16_t*>(input),
                static_cast<bfloat16_t*>(output),
                params,
                num_threads
            );
        }
        log_info("Pooling Reference: BF16 execution completed");
        return status_t::success;
    }

    log_error("Pooling Reference: Unsupported data type");
    return status_t::failure;
}

} // namespace pooling
} // namespace lowoha
} // namespace zendnnl

