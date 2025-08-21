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

#ifndef _ZENAI_OPERATOR_CONFIG_HPP_
#define _ZENAI_OPERATOR_CONFIG_HPP_

#include "nlohmann/json.hpp"
#include "common/error_status.hpp"

using json = nlohmann::json;
using namespace zendnnl::error_handling;

namespace zendnnl {
namespace ops {
/** @class op_config_t
 *  @brief A base class for operator config.
 *
 *  Given an operator, it will have specific runtime parameters that leads
 *  to optimal performance.
 *
 *  operator config can be inherited to different operators and runtime
 *  parameters can be updated accordingly.
 *
 */
class op_config_t {
 public:
  /** @brief Set default runtime variables.
  */
  virtual void set_default_config() = 0;

  /** @brief Set runtime variables from json.
  */
  virtual status_t set_user_config(json config_json) = 0;

  /** @brief Set runtime variables from environment.
  */
  virtual void set_env_config() = 0;

  /** @brief Virtual destructor
  *
  *  Virtual since this class acts as virtual base class.
  */
  virtual ~op_config_t() = default;
};

} // namespace ops
} // namespace zendnnl

#endif