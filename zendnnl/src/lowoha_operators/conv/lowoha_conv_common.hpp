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

#ifndef _LOWOHA_CONV_COMMON_HPP
#define _LOWOHA_CONV_COMMON_HPP

#include <cstdint>
#include <string>
#include "common/zendnnl_global.hpp"
#include "common/logging.hpp"
#include "operators/common/post_op.hpp"

namespace zendnnl {
namespace lowoha {
namespace conv {

using namespace zendnnl::common;

/**
 * @brief Conv tensor dimensions
 *
 * Represents dimensions for convolution tensors in NHWC format:
 * - Input: [N, H, W, C]
 * - Filter: [KH, KW, C_in, C_out]
 * - Output: [N, H_out, W_out, C_out]
 */
struct conv_dims_t {

    uint64_t batch;                ///< Batch size (N)
    uint64_t in_height;            ///< Input height (H)
    uint64_t in_width;             ///< Input width (W)
    uint64_t in_channels;          ///< Input channels (C = C_in)

    uint64_t filter_height;        ///< Filter height (KH)
    uint64_t filter_width;         ///< Filter width (KW)
    uint64_t out_channels;         ///< Output channels (C_out)

    uint64_t out_height;           ///< Output height
    uint64_t out_width;            ///< Output width

    /**
     * @brief Default constructor
     */
    conv_dims_t() : batch(0), in_height(0), in_width(0),
                      in_channels(0), filter_height(0), filter_width(0),
                      out_channels(0), out_height(0), out_width(0) {}
};

/**
 * @brief Structure to hold data types for convolution operands
 */
 struct conv_data_types {
    data_type_t input = data_type_t::none;     ///< Input data type
    data_type_t filter = data_type_t::none;     ///< Filter data type
    data_type_t bias = data_type_t::none;       ///< Bias data type
    data_type_t output = data_type_t::none;     ///< Output data type
};

/**
 * @brief Structure for post-operation parameters
 */
 struct conv_postop {
    zendnnl::ops::post_op_type_t po_type;    ///< Type of post-operation
    void *buff;                              ///< Buffer for binary operations
    data_type_t dtype;                       ///< Data type of the buffer
    std::vector<int64_t> dims;               ///< Dimensions of the buffer
    float alpha;                             ///< Alpha parameter for post-operation
    float beta;                              ///< Beta parameter for post-operation

    /**
     * @brief Default constructor for postop
     */
    conv_postop() : po_type(zendnnl::ops::post_op_type_t::none), buff(nullptr),
      dtype(data_type_t::none), dims(), alpha(0.0f), beta(0.0f) {}
};

/**
 * @brief Depthwise convolution parameters
 */
struct depthwise_params {
    // Grouped/Depthwise convolution settings
    uint32_t groups;               ///< Number of groups for grouped convolution (1 = standard conv)
    bool is_depthwise;             ///< True if depthwise convolution (groups == in_channels)
    uint32_t depth_multiplier;     ///< Depth multiplier for depthwise (output_channels_per_group)

    /**
     * @brief Default constructor
     */
    depthwise_params() : groups(1), is_depthwise(false), depth_multiplier(1) {}
};

/**
 * @brief Convolution algorithm type
 */
 enum class conv_algo_t {
    none = -1,             /*!< No algorithm selected */
    dynamic_dispatch = 0,  /*!< Dynamic dispatch - Not implemented */
    aocl_dlp_blocked = 1,  /*!< Blocked AOCL - Not implemented */
    onednn_blocked = 2,    /*!< Blocked OneDNN */
    aocl_dlp  = 4,         /*!< AOCL - Not implemented */
    onednn = 5,            /*!< OneDNN */
    auto_tuner = 8,        /*!< Auto Tuner - Not implemented */
    reference = 9,         /*!< Reference - Not implemented */
};

/**
 * @brief Conv parameters (strides, padding, dilations, grouped/depthwise settings)
 *
 * Represents all parameters needed for convolution computation:
 * - Strides: Step size for filter movement
 * - Padding: Zero-padding around input
 * - Dilations: Spacing between filter elements
 * - Grouped/Depthwise: Settings for grouped and depthwise convolutions
 */
struct conv_params {

    // Strides [stride_h, stride_w]
    uint32_t stride_h;             ///< Stride along height dimension
    uint32_t stride_w;             ///< Stride along width dimension

    // Padding [top, left, bottom, right]
    uint32_t pad_top;              ///< Padding at top (height)
    uint32_t pad_left;             ///< Padding at left (width)
    uint32_t pad_bottom;           ///< Padding at bottom (height)
    uint32_t pad_right;            ///< Padding at right (width)

    // Dilations [dilation_h, dilation_w]
    uint32_t dilation_h;           ///< Dilation along height dimension
    uint32_t dilation_w;           ///< Dilation along width dimension

    depthwise_params depthwise;     ///< Depthwise convolution parameters

    // Data format (currently NHWC only)
    char data_format[8];           ///< Data format string ("NHWC")

    conv_dims_t dims;             ///< Convolution dimensions
    conv_algo_t algo;             ///< Convolution algorithm

    conv_data_types dtypes;           ///< Data types

    std::vector<conv_postop> postop_;

    /**
     * @brief Default constructor
     */
    conv_params() : stride_h(1), stride_w(1),
                        pad_top(0), pad_left(0),
                        pad_bottom(0), pad_right(0),
                        dilation_h(1), dilation_w(1),
                        depthwise(),
                        dims(), algo(conv_algo_t::none),
                        dtypes(), postop_() {
        std::strncpy(data_format, "NHWC", 8);
    }
};

} // namespace conv
} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_CONV_COMMON_HPP
