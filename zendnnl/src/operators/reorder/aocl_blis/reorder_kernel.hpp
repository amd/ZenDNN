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
#ifndef _SAMPLE_FP32_AVX512_KERNEL_HPP_
#define _SAMPLE_FP32_AVX512_KERNEL_HPP_

#include <iostream>
#include <memory>
#include "common/zendnnl_global.hpp"
#include "operators/common/operator_kernel.hpp"
#include "operators/reorder/reorder_context.hpp"

namespace zendnnl {
namespace ops {
/** @class reorder_kernel_t
 *  @brief a reorder kernel for fp32/bf16/s8/s4 data type.
 *
 * It is invoked if input data type is @c data_type_t::f32 or
 * @c data_type_t::bf16 or @c data_type_t::s8 or @c data_type_t::s4.
 */
class reorder_kernel_t final : public op_kernel_t<reorder_context_t> {
 public:
  /** @brief Default destructor */
  ~reorder_kernel_t() = default;

  /** @brief Execute */
  status_t execute(const context_type &context_,
                   tensor_map_type &inputs_,
                   tensor_map_type &outputs_) override;

  /** @fn data_copy
  *
  * @brief
  * Templatized API, copies the data to actual buffer from the local buffer.
  *
  */
  template <typename T>
  void data_copy(void *output, void *reorder_weights, size_t reorder_size) {
    for (long long int idx = 0; idx < (long long int)(reorder_size/sizeof(T));
         idx++) {
      ((T *)output)[idx] = ((T *)reorder_weights)[idx];
    }
  }
};

} //namespace ops
} //namespace zendnnl

extern "C" {
  /** @fn get_reorder_aocl_kernel
   *
   * This is needed inside extern"C" scope to avoid name mangling. This arrangement is
   * made to enable dynamic loading. After loading the module using
   * @c operator_t::load_module(), this function is searched using
   * @c operator_t::load_symbol(), and executed to get kernel pointer.
   */
  zendnnl::ops::reorder_kernel_t *get_reorder_aocl_kernel();
}

#endif
