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
#ifndef _SDPA_ENCODER_CONTEXT_HPP_
#define _SDPA_ENCODER_CONTEXT_HPP_

#include "common/zendnnl_global.hpp"
#include "operators/common/operator_context.hpp"

namespace zendnnl {
namespace ops {

/** @class sdpa_encoder_context_t
 *  @brief SDPA encoder context for @c sdpa_encoder_operator_t.
 *
 * In order to enable chaining, the first parameter in @c op_context_t template
 * should be the class itself.
 * @sa sdpa_encoder_operator_t
 */
class sdpa_encoder_context_t final : public
  op_context_t<sdpa_encoder_context_t> {
 public:
  using parent_type = op_context_t<sdpa_encoder_context_t>;
  /** @brief constructor */
  sdpa_encoder_context_t();

  /** @brief Set scale parameter value.*/
  sdpa_encoder_context_t &set_scale(float scale_);

  /** @brief Get scale parameter value.*/
  float get_scale() const;

  /** @brief Set is_dropout parameter value.*/
  sdpa_encoder_context_t &set_is_dropout(bool is_dropout_);

  /** @brief Get is_dropout parameter value.*/
  bool get_is_dropout() const;

  /** @brief Set is_causal parameter value.*/
  sdpa_encoder_context_t &set_is_causal(bool is_causal_);

  /** @brief Get is_causal parameter value.*/
  bool get_is_causal() const;

  /** @brief Set has_mask parameter value.*/
  sdpa_encoder_context_t &set_has_mask(bool has_mask_);

  /** @brief Get has_mask parameter value.*/
  bool get_has_mask() const;


  /** @brief Returns SDPA encoder context information */
  std::string context_info() override;

 protected:
  /** @brief validate parameters */
  status_t validate() override;
 private:
  float _scale;      /**< scale parameter */
  bool _is_dropout;  /**< is_dropout parameter */
  bool _is_causal;   /**< is_causal parameter */
  bool _has_mask;    /**< has_mask parameter */
};

} //namespace ops

namespace interface {
using sdpa_encoder_context_t = zendnnl::ops::sdpa_encoder_context_t;
} //export

} //namespace zendnnl
#endif