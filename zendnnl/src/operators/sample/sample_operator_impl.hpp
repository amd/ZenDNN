/********************************************************************************
# * Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef _SAMPLE_OPERATOR_IMPL_HPP_
#define _SAMPLE_OPERATOR_IMPL_HPP_

#include "common/zendnnl_global.hpp"
#include "operators/common/operator_impl.hpp"
#include "sample_context.hpp"

namespace zendnnl {
namespace ops {
/** @class sample_impl_t
 *  @brief A sample operator class for demonstration and starting point for new
 *  operators.
 *
 * @par Synopsys
 *
 * Invokes a fp32(bf16) kernel if input type is fp32(bf16). The kernel prints
 * its name.
 *
 * An new operator can be developed by taking this class as a boiler-plate code,
 * or a starting point.
 *
 * In order to elable chaining, the first parameter in @c operator_t template
 * should be the class itself, and the second parameter should be this operators
 * context, derived from @c operator_context_t.
 *
 * @par Parameters, Inputs, Outputs
 *
 * The operator has following parameters and input/outputs
 * - Parameter(s)
 *   1. (mandatory) sample_param  : An arbitrary tensor.
 * - Inputs
 *   1. (mandatory) sample_input  : An arbitrary tensor of type(f32,bf16).
 * - Output(s)
 *   1. (mandatory) sample_output : An arbitrary tensor.
 */
class sample_impl_t final : public operator_impl_t<sample_context_t> {
public:
  /** @brief Self type **/
  using self_type = sample_impl_t;
  /** @brief Parent type **/
  using parent_type = operator_impl_t<sample_context_t>;
  /** @brief context type **/
  using context_type = parent_type::context_type;
  /** @brief kernel type **/
  using   kernel_type =  parent_type::kernel_type;
  /** @brief Shared pointer to kernels */
  using   kernel_sptr_type =  parent_type::kernel_sptr_type;
  /** @brief A map type from strings to tensors */
  using   tensor_map_type = parent_type::tensor_map_type;
  /** @brief Kernel handle type */
  using   create_kernel_handle_type  = parent_type::create_kernel_handle_type;

protected:
  /** @brief Validate input/output
   *
   * Validates if all mandatory inputs and outputs are given.
   * @return @c status_t::success if successful.
   */
  status_t validate() override;

  /** @brief Print operator create information
   * @return @c std::string
   */
  std::string op_create_info() override;

  /** @brief Print operator execute information
   * @return @c std::string
   */
  std::string op_execute_info() override;

  /** @brief Select kernel based on input data type.
   * @return @c status_t::success if successful.
   */
  status_t kernel_factory() override;
};

} //namespace ops
} //namespace zendnnl
#endif
