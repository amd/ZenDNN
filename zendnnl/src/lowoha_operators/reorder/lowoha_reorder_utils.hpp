/*******************************************************************************
# * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef LOWOHA_REORDER_UTILS_HPP
#define LOWOHA_REORDER_UTILS_HPP

#include "lowoha_operators/reorder/lowoha_reorder_common.hpp"
#include "memory/memory_utils.hpp"

namespace zendnnl {
namespace lowoha {

using zendnnl::memory::status_t;

/**
 * @brief Validates input parameters for reorder operation.
 *
 * This function performs comprehensive validation of all input parameters to ensure
 * they are valid and compatible for reorder operation. It checks for null pointers,
 * valid nelems, supported data type combinations, and valid quantization parameters.
 *
 * @param src Pointer to the source data
 * @param dst Pointer to the destination data
 * @param nelems Number of elements to reorder
 * @param params Const reference to lowoha_reorder_params_t containing operation configuration
 * @return status_t::success if all validations pass, status_t::failure otherwise
 */
status_t validate_reorder_inputs(const void *src, void *dst, size_t nelems,
                                  const lowoha_reorder_params_t &params);

/**
 * @brief Convert data_type_t enum to string representation.
 *
 * @param dtype The data_type_t enum value to convert.
 * @return A const char* pointer to the string representation of the data type.
 */
const char *reorder_data_type_to_string(data_type_t dtype);

/**
 * @brief Convert reorder_algo_t enum to string representation.
 *
 * @param algo The reorder_algo_t enum value to convert.
 * @return A const char* pointer to the string representation of the algorithm.
 */
const char *reorder_algo_to_string(reorder_algo_t algo);

/**
 * @brief Select the optimal reorder algorithm based on parameters.
 *
 * @param params Reorder parameters
 * @param nelems Number of elements to process
 * @return Selected algorithm
 */
reorder_algo_t select_reorder_algo(const lowoha_reorder_params_t &params, size_t nelems);

/**
 * @brief Check if the given data type combination is supported for reorder.
 *
 * @param src_dtype Source data type
 * @param dst_dtype Destination data type
 * @return true if the combination is supported, false otherwise
 */
bool is_reorder_supported(data_type_t src_dtype, data_type_t dst_dtype);

} // namespace lowoha
} // namespace zendnnl

#endif // LOWOHA_REORDER_UTILS_HPP

