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
#ifndef _COMPARE_CONTEXT_HPP_
#define _COMPARE_CONTEXT_HPP_

#include <vector>
#include <memory>
#include <optional>

#include "common/zendnnl_global.hpp"
#include "operators/common/operator_context.hpp"

namespace zendnnl {
namespace ops {

using namespace zendnnl::memory;

struct compare_stats_t{
  float match_percent;
  float max_deviation;
  float mean_deviation;
  float min_deviation;
};

/** @class compare_context_t
 *  @brief context for @c compare_operator_t.
 *
 * In order to elable chaining, the first parameter in @c op_context_t template
 * should be the class itself.
 * @sa compare_operator_t
 */
class compare_context_t final : public op_context_t<compare_context_t> {
public:
  /** @brief parent type */
  using parent_type = op_context_t<compare_context_t>;

  /** @brief constructor */
  compare_context_t();

  /** @brief set the tolerance */
  compare_context_t& set_tolerance(float tolerance_);

  /** @brief get the tolerance */
  float              get_tolerance() const;

  /** @brief get compare stats pointer */
  std::shared_ptr<compare_stats_t> get_compare_stats() const;

protected:
  /** @brief validate parameters */
  status_t validate() override;

  /** @brief Returns compare context information */
  std::string context_info() override;

private:
  float tolerance;
  std::shared_ptr<compare_stats_t> stats;
};

} //namespace ops

namespace interface {
using compare_context_t = zendnnl::ops::compare_context_t;
using compare_stats_t   = zendnnl::ops::compare_stats_t;
} //interface

} //namespace zendnnl
#endif
