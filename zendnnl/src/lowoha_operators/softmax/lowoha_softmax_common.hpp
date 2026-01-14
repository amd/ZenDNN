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

#ifndef _LOWOHA_SOFTMAX_COMMON_HPP
#define _LOWOHA_SOFTMAX_COMMON_HPP

#include <cstdint>
#include "memory/memory_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace softmax {

using namespace zendnnl::common;

/**
 * @brief Softmax algorithm type
 */
enum class softmax_algo_t {
    none = -1,             /*!< No algorithm selected */
    dynamic_dispatch = 0,  /*!< Dynamic dispatch - Not implemented */
    onednn = 1,            /*!< OneDNN backend */
    reference = 2          /*!< Reference implementation */
};

/**
 * @brief Maximum supported tensor dimensions
 */
constexpr int SOFTMAX_MAX_NDIMS = 5;

/**
 * @brief Parameter structure for LOWOHA softmax operations
 *
 * This structure contains all parameters specific to softmax
 * computation including dimensions, computation parameters,
 * data types, and algorithm selection.
 */
struct softmax_params {
    uint64_t batch;                 ///< Batch size (outer dimensions product)
    uint64_t axis_dim;              ///< Dimension size along softmax axis
    int axis;                       ///< Axis along which to compute softmax (-1 for last axis)
    bool log_softmax;               ///< If true, compute log(softmax(x)) instead of softmax(x)
    data_type_t src_dt;             ///< Source/input data type
    data_type_t dst_dt;             ///< Destination/output data type
    softmax_algo_t algorithm;       ///< Selected algorithm
    uint64_t num_threads;           ///< Number of threads

    // Original tensor shape information (for OneDNN backend)
    uint64_t shape[SOFTMAX_MAX_NDIMS];  ///< Original tensor dimensions
    int ndims;                          ///< Number of dimensions in original tensor

    /**
     * @brief Default constructor
     */
    softmax_params() : batch(1), axis_dim(0),
                       axis(-1),
                       log_softmax(false),
                       src_dt(data_type_t::none),
                       dst_dt(data_type_t::none),
                       algorithm(softmax_algo_t::none),
                       num_threads(0), ndims(0) {
        for (int i = 0; i < SOFTMAX_MAX_NDIMS; ++i) {
            shape[i] = 0;
        }
    }
};

} // namespace softmax
} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_SOFTMAX_COMMON_HPP
