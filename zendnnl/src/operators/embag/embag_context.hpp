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

#ifndef _EMBAG_CONTEXT_HPP_
#define _EMBAG_CONTEXT_HPP_

#include <cstdint>
#include <memory>

#include "common/zendnnl_global.hpp"
#include "operators/common/operator_context.hpp"

namespace zendnnl {
namespace ops {

using namespace zendnnl::memory;

enum class embag_algo_t : uint8_t {
  none = 0, sum = 1, mean = 2, max = 3
};

/** @class embag_context_t
 *  @brief context for @c embag_operator_t.
 *
 * In order to elable chaining, the first parameter in @c op_context_t template
 * should be the class itself.
 * @sa embag_operator_t
 */
class embag_context_t final : public op_context_t<embag_context_t> {
 public:
  /** @brief parent type */
  using parent_type = op_context_t<embag_context_t>;

  /** @brief constructor */
  embag_context_t();

  embag_context_t &set_algo(embag_algo_t algo_);
  embag_algo_t     get_algo() const;

  embag_context_t &set_padding_index(int64_t padding_index_);
  int64_t          get_padding_index() const;

  embag_context_t &set_include_last_offset(bool include_last_offset_);
  bool             get_include_last_offset() const;

  embag_context_t &set_is_weights(bool is_weights_);
  bool             get_is_weights() const;

  embag_context_t &set_fp16_scale_bias(bool fp16_scale_bias_);
  bool             get_fp16_scale_bias() const;

 protected:
  /** @brief validate parameters */
  status_t validate() override;

  /** @brief Returns embedding bag context information */
  std::string context_info() override;

 private:
  embag_algo_t  algo;
  int64_t       padding_index;
  bool          include_last_offset;
  bool          is_weights;
  bool          fp16_scale_bias;

};

} //namespace ops

namespace interface {
using embag_context_t = zendnnl::ops::embag_context_t;
using embag_algo_t    = zendnnl::ops::embag_algo_t;
} //interface

} //namespace zendnnl
#endif