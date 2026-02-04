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
namespace conv {

using namespace zendnnl::common;

/**
 * @brief Apply post-operation to a value
 */
template<typename T>
inline float apply_postop(float val, const conv_postop &postop, 
                          const void *binary_buff, const int64_t idx) {
    using namespace zendnnl::ops;
    
    switch (postop.po_type) {
        case post_op_type_t::relu:
            return std::max(0.0f, val);
        
        case post_op_type_t::clip:
            // Clip can be used for relu6: clip(0, 6)
            return std::min(std::max(val, postop.alpha), postop.beta);
        
        case post_op_type_t::gelu_tanh: {
            // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            float x = val;
            float x3 = x * x * x;
            float inner = 0.7978845608f * (x + 0.044715f * x3);
            return 0.5f * x * (1.0f + std::tanh(inner));
        }
        
        case post_op_type_t::gelu_erf: {
            // GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
            return 0.5f * val * (1.0f + std::erf(val * 0.7071067812f));
        }
        
        case post_op_type_t::swish:
            // Swish/SiLU: x * sigmoid(x)
            return val / (1.0f + std::exp(-val));
        
        case post_op_type_t::sigmoid:
            return 1.0f / (1.0f + std::exp(-val));
        
        case post_op_type_t::tanh:
            return std::tanh(val);
        
        case post_op_type_t::binary_add:
            if (binary_buff) {
                const T *buff_typed = static_cast<const T*>(binary_buff);
                return val + static_cast<float>(buff_typed[idx]);
            }
            return val;
        
        case post_op_type_t::binary_mul:
            if (binary_buff) {
                const T *buff_typed = static_cast<const T*>(binary_buff);
                return val * static_cast<float>(buff_typed[idx]);
            }
            return val;
        
        default:
            return val;
    }
}

/**
 * @brief Reference implementation of 2D convolution
 * 
 * Supports standard, depthwise, and grouped convolutions with dilations
 */
template<typename T>
void conv2d_reference_impl(
    const T *input,
    const T *filter,
    const T *bias,
    T *output,
    const conv_params &params,
    const int num_threads
) {
    const uint64_t batch = params.dims.batch;
    const uint64_t in_h = params.dims.in_height;
    const uint64_t in_w = params.dims.in_width;
    const uint64_t in_c = params.dims.in_channels;
    const uint64_t out_h = params.dims.out_height;
    const uint64_t out_w = params.dims.out_width;
    const uint64_t out_c = params.dims.out_channels;
    const uint64_t kh = params.dims.filter_height;
    const uint64_t kw = params.dims.filter_width;
    
    const uint32_t stride_h = params.stride_h;
    const uint32_t stride_w = params.stride_w;
    const int32_t pad_t = params.pad_top;
    const int32_t pad_l = params.pad_left;
    const uint32_t dilation_h = params.dilation_h;
    const uint32_t dilation_w = params.dilation_w;
    
    const uint32_t groups = params.depthwise.groups;
    const bool is_depthwise = params.depthwise.is_depthwise;
    
    // Calculate channels per group
    const uint64_t in_c_per_group = in_c / groups;
    const uint64_t out_c_per_group = out_c / groups;
    
    log_info("Conv2D Reference: batch=", batch, " in=[", in_h, ",", in_w, ",", in_c, 
             "] out=[", out_h, ",", out_w, ",", out_c, "] kernel=[", kh, ",", kw, "]");
    log_info("Conv2D Reference: stride=[", stride_h, ",", stride_w, "] pad=[", pad_t, ",", pad_l, 
             "] dilation=[", dilation_h, ",", dilation_w, "] groups=", groups);
    
    // NHWC format: [batch, height, width, channels]
    // Parallelize over batch and output spatial dimensions
    #pragma omp parallel for collapse(3) num_threads(num_threads)
    for (uint64_t n = 0; n < batch; ++n) {
        for (uint64_t oh = 0; oh < out_h; ++oh) {
            for (uint64_t ow = 0; ow < out_w; ++ow) {
                // Process all output channels
                for (uint64_t oc = 0; oc < out_c; ++oc) {
                    float sum = 0.0f;
                    
                    // Determine which group this output channel belongs to
                    uint64_t group = oc / out_c_per_group;
                    uint64_t oc_in_group = oc % out_c_per_group;
                    
                    // Input channel range for this group
                    uint64_t ic_start = group * in_c_per_group;
                    uint64_t ic_end = ic_start + in_c_per_group;
                    
                    // Iterate over kernel spatial dimensions
                    for (uint64_t kh_idx = 0; kh_idx < kh; ++kh_idx) {
                        for (uint64_t kw_idx = 0; kw_idx < kw; ++kw_idx) {
                            // Calculate input position with dilation
                            int64_t ih = static_cast<int64_t>(oh * stride_h) + 
                                        static_cast<int64_t>(kh_idx * dilation_h) - pad_t;
                            int64_t iw = static_cast<int64_t>(ow * stride_w) + 
                                        static_cast<int64_t>(kw_idx * dilation_w) - pad_l;
                            
                            // Check bounds
                            if (ih >= 0 && ih < static_cast<int64_t>(in_h) &&
                                iw >= 0 && iw < static_cast<int64_t>(in_w)) {
                                
                                // Iterate over input channels in this group
                                for (uint64_t ic = ic_start; ic < ic_end; ++ic) {
                                    // NHWC layout: index = n*H*W*C + h*W*C + w*C + c
                                    uint64_t in_idx = n * in_h * in_w * in_c +
                                                     ih * in_w * in_c +
                                                     iw * in_c +
                                                     ic;
                                    
                                    // Filter layout: [KH, KW, C_in, C_out]
                                    // For grouped: each group has its own set of filters
                                    uint64_t ic_offset = ic - ic_start;
                                    uint64_t filter_idx = kh_idx * kw * in_c_per_group * out_c_per_group +
                                                         kw_idx * in_c_per_group * out_c_per_group +
                                                         ic_offset * out_c_per_group +
                                                         oc_in_group;
                                    
                                    // For depthwise: adjust filter indexing
                                    if (is_depthwise) {
                                        filter_idx = kh_idx * kw * in_c * params.depthwise.depth_multiplier +
                                                    kw_idx * in_c * params.depthwise.depth_multiplier +
                                                    ic * params.depthwise.depth_multiplier +
                                                    oc_in_group;
                                    } else {
                                        // Standard or grouped convolution
                                        filter_idx = group * kh * kw * in_c_per_group * out_c_per_group +
                                                    kh_idx * kw * in_c_per_group * out_c_per_group +
                                                    kw_idx * in_c_per_group * out_c_per_group +
                                                    ic_offset * out_c_per_group +
                                                    oc_in_group;
                                    }
                                    
                                    float in_val = static_cast<float>(input[in_idx]);
                                    float filter_val = static_cast<float>(filter[filter_idx]);
                                    sum += in_val * filter_val;
                                }
                            }
                        }
                    }
                    
                    // Add bias if provided
                    if (bias != nullptr) {
                        sum += static_cast<float>(bias[oc]);
                    }
                    
                    // Apply post-operations
                    uint64_t out_idx = n * out_h * out_w * out_c +
                                      oh * out_w * out_c +
                                      ow * out_c +
                                      oc;
                    
                    for (const auto &postop : params.postop_) {
                        sum = apply_postop<T>(sum, postop, postop.buff, out_idx);
                    }
                    
                    // Write output
                    output[out_idx] = static_cast<T>(sum);
                }
            }
        }
    }
}

status_t conv_reference_wrapper(
    const void *input,
    const void *filter,
    const void *bias,
    void *output,
    const bool is_weights_const,
    conv_params &params
) {
    // Get number of threads
    const int num_threads = omp_get_max_threads();
    
    log_info("Conv2D Reference: Starting reference kernel execution");
    
    // Dispatch based on data type
    if (params.dtypes.input == data_type_t::f32) {
        conv2d_reference_impl<float>(
            static_cast<const float*>(input),
            static_cast<const float*>(filter),
            static_cast<const float*>(bias),
            static_cast<float*>(output),
            params,
            num_threads
        );
        log_info("Conv2D Reference: FP32 execution completed");
        return status_t::success;
    } 
    else if (params.dtypes.input == data_type_t::bf16) {
        conv2d_reference_impl<bfloat16_t>(
            static_cast<const bfloat16_t*>(input),
            static_cast<const bfloat16_t*>(filter),
            static_cast<const bfloat16_t*>(bias),
            static_cast<bfloat16_t*>(output),
            params,
            num_threads
        );
        log_info("Conv2D Reference: BF16 execution completed");
        return status_t::success;
    }
    
    log_error("Conv2D Reference: Unsupported data type");
    return status_t::failure;
}

} // namespace conv
} // namespace lowoha
} // namespace zendnnl
