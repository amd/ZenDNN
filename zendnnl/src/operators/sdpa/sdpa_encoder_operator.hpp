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
#ifndef _SDPA_ENCODER_OPERATOR_HPP_
#define _SDPA_ENCODER_OPERATOR_HPP_

#include "common/zendnnl_global.hpp"
#include "operators/common/operator.hpp"
#include "operators/sdpa/sdpa_encoder_context.hpp"
#include "operators/sdpa/sdpa_encoder_operator_impl.hpp"

namespace zendnnl {
namespace ops {
/** @class sdpa_encoder_operator_t
 *  @brief A SDPA encoder operator class for Scaled Dot-Product Attention.
 *
 * @par Synopsys
 *
 * Invokes a fp32 kernel if input type is fp32. The kernel performs
 * Scaled Dot-Product Attention computation.
 *
 * In order to enable chaining, the first parameter in @c operator_t template
 * should be the class itself, and the second parameter should be this operators
 * context, derived from @c operator_context_t.
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
class sdpa_encoder_operator_t final : public operator_t<sdpa_encoder_operator_t,
                                                  sdpa_encoder_context_t,
                                                  sdpa_encoder_impl_t> {
public:
  /** @brief Self type **/
  using self_type = sdpa_encoder_operator_t;
  /** @brief Parent type **/
  using parent_type = operator_t<sdpa_encoder_operator_t, sdpa_encoder_context_t, sdpa_encoder_impl_t>;
  /** @brief context type **/
  using context_type = parent_type::context_type;
  /** @brief impl type **/
  using impl_type = parent_type::impl_type;
  /** @brief impl pointer type **/
  using impl_sptr_type = parent_type::impl_sptr_type;
};

} //namespace ops

namespace interface {
using sdpa_encoder_operator_t = zendnnl::ops::sdpa_encoder_operator_t;
} //export

} //namespace zendnnl
#endif