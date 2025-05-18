/********************************************************************************
# * Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "sample_context.hpp"

namespace zendnnl {
namespace ops {
/** @class sample_f32_avx512_kernel_t
 *  @brief a sample kernel for fp32 data type.
 *
 * It is invoked if input data type is @c data_type_t::f32.
 */
class sample_f32_avx512_kernel_t final : public op_kernel_t<sample_context_t> {
public:
  /** @brief Default destructor */
  ~sample_f32_avx512_kernel_t() = default;

  /** @brief Execute */
  status_t execute(const context_type& context_,
                   tensor_map_type& inputs_,
                   tensor_map_type& outputs_) override;
};

} //namespace ops
} //namespace zendnnl

extern "C" {
  /** @fn get_sample_f32_avx512_kernel
   *  @brief returns a shared pointer to sample_f32_kernel_t kernel
   *
   * This is needed inside extern"C" scope to avoid name mangling. This arrangement is
   * made to enable dynamic loading. After loading the module using
   * @c operator_t::load_module(), this function is searched using
   * @c operator_t::load_symbol(), and executed to get kernel pointer.
   */
  std::shared_ptr<zendnnl::ops::sample_f32_avx512_kernel_t> get_sample_f32_avx512_kernel();
}

#endif
