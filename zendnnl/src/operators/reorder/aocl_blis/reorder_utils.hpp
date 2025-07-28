/********************************************************************************
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

#ifndef _REORDER_UTILS_HPP_
#define _REORDER_UTILS_HPP_

#include "memory/tensor.hpp"
#include "operators/reorder/reorder_context.hpp"

namespace zendnnl {
namespace ops {

using namespace zendnnl::memory;

/** @class aocl_blis_reorder_utils_t
 *  @brief Utility class for AOCL BLIS reorder operations.
 *
 */
class aocl_blis_reorder_utils_t {
 public:

  /** @brief Computes size for AOCL BLIS reorder/unreorder operations
  *  @param context The reorder context containing configuration parameters.
  *  @param input_tensor The input tensor for which the reorder/unreorder size is computed.
  *  @return The size required for the AOCL BLIS reorder/unreorder operation.
  */
  static size_t get_aocl_reorder_size(const reorder_context_t &context,
                                      const tensor_t &input_tensor);

};

} // namespace ops
} // namespace zendnnl

#endif