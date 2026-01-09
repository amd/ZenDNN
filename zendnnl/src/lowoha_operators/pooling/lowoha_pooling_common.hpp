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

#ifndef _LOWOHA_POOLING_COMMON_HPP
#define _LOWOHA_POOLING_COMMON_HPP

#include <cstdint>
#include <cstring>
#include "common/data_types.hpp"

namespace zendnnl {
namespace lowoha {
namespace pooling {

using namespace zendnnl::common;

/**
 * @brief Pooling algorithm type
 */
enum class pooling_algo_t {
    none = -1,            /*!< No algorithm selected */
    dynamic_dispatch = 0, /*!< Dynamic dispatch - Not implemented */
    onednn = 1,           /*!< OneDNN backend */
    reference = 2         /*!< Reference implementation */
};

/**
 * @brief Average pooling padding mode
 */
enum class avg_pooling_mode_t {
    include_padding = 0,  ///< Include padding in average calculation (SAME as TensorFlow)
    exclude_padding = 1   ///< Exclude padding from average calculation (VALID counting)
};

/**
 * @brief Pooling tensor dimensions
 *
 * Represents dimensions for pooling tensors in NHWC format:
 * - Input: [N, H, W, C]
 * - Output: [N, H_out, W_out, C]
 */
struct pooling_dims_t {
    // Input dimensions [N, H, W, C]
    uint64_t batch;                ///< Batch size (N)
    uint64_t in_height;            ///< Input height (H)
    uint64_t in_width;             ///< Input width (W)
    uint64_t channels;             ///< Number of channels (C)

    // Pooling window dimensions [KH, KW]
    uint64_t kernel_height;        ///< Kernel height (KH)
    uint64_t kernel_width;         ///< Kernel width (KW)

    // Output dimensions [N, H_out, W_out, C]
    uint64_t out_height;           ///< Output height
    uint64_t out_width;            ///< Output width

    /**
     * @brief Default constructor
     */
    pooling_dims_t() : batch(0), in_height(0), in_width(0), channels(0),
                       kernel_height(0), kernel_width(0),
                       out_height(0), out_width(0) {}
};

/**
 * @brief Structure to hold data types for pooling operands
 */
struct pooling_data_types {
    data_type_t src = data_type_t::none;     ///< Input tensor data type
    data_type_t dst = data_type_t::none;     ///< Output tensor data type

    /**
     * @brief Default constructor
     */
    pooling_data_types() : src(data_type_t::none), dst(data_type_t::none) {}
};

/**
 * @brief Main parameter structure for LOWOHA pooling operations
 *
 * This structure encapsulates all parameters needed for pooling operations,
 * following the same pattern as lowoha_params for matmul.
 */
struct pool_params {
    pooling_dims_t dims;             ///< Tensor dimensions (input/output)
    pooling_data_types dtypes;       ///< Data types for input/output)
    pooling_algo_t algo;             ///< Selected backend algorithm

    // Strides [stride_h, stride_w]
    uint32_t stride_h;             ///< Stride along height dimension
    uint32_t stride_w;             ///< Stride along width dimension

    // Padding [top, left, bottom, right]
    uint32_t pad_top;              ///< Padding at top (height)
    uint32_t pad_left;             ///< Padding at left (width)
    uint32_t pad_bottom;           ///< Padding at bottom (height)
    uint32_t pad_right;            ///< Padding at right (width)

    // Pooling type
    bool is_max_pooling;           ///< True for max pooling, false for average pooling

    // Average pooling mode (only used when is_max_pooling = false)
    avg_pooling_mode_t avg_mode;   ///< Include or exclude padding in average calculation

    // Data format (currently NHWC only)
    char data_format[8];           ///< Data format string ("NHWC")

    uint64_t num_threads;            ///< Number of threads (0 = auto)

    /**
     * @brief Default constructor
     */

    pool_params() 
        : dims(), dtypes(), algo(pooling_algo_t::none),
          stride_h(1), stride_w(1), 
          pad_top(0), pad_left(0), pad_bottom(0), pad_right(0),
          is_max_pooling(true), 
          avg_mode(avg_pooling_mode_t::exclude_padding),
          num_threads(0) {
          std::strncpy(data_format, "NHWC", 8);
          }
};

} // namespace pooling
} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_POOLING_COMMON_HPP
