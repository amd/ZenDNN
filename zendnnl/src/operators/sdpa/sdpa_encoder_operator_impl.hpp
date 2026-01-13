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
#ifndef _SDPA_ENCODER_OPERATOR_IMPL_HPP_
#define _SDPA_ENCODER_OPERATOR_IMPL_HPP_

#include "common/zendnnl_global.hpp"
#include "operators/common/operator_impl.hpp"
#include "sdpa_encoder_context.hpp"

namespace zendnnl {
namespace ops {
/** @class sdpa_encoder_impl_t
 *  @brief Implementation class for SDPA encoder operator.
 *
 * @par Synopsys
 *
 * Implements Scaled Dot-Product Attention (SDPA) for encoder architectures.
 * Invokes a fp32 kernel if input type is fp32.
 *
 * @par Parameters, Inputs, Outputs
 *
 * The operator has following parameters and input/outputs
 * - Parameter(s)
 *   1. (mandatory) query  : Query tensor [B, H, S, D].
 *   2. (mandatory) key    : Key tensor [B, H, S, D].
 *   3. (mandatory) value  : Value tensor [B, H, S, D].
 *   4. (optional)  mask   : Attention mask tensor.
 * - Output(s)
 *   1. (mandatory) sdpa_output : Output tensor.
 */
class sdpa_encoder_impl_t final : public operator_impl_t<sdpa_encoder_context_t> {
public:
  /** @brief Self type **/
  using self_type = sdpa_encoder_impl_t;
  /** @brief Parent type **/
  using parent_type = operator_impl_t<sdpa_encoder_context_t>;
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
