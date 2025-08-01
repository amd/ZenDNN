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
#ifndef _REORDER_OPERATOR_HPP_
#define _REORDER_OPERATOR_HPP_

#include "aocl_blis/reorder_utils.hpp"
#include "common/zendnnl_global.hpp"
#include "operators/common/operator.hpp"
#include "reorder_context.hpp"

namespace zendnnl {
namespace ops {

class reorder_operator_t final : public
  operator_t<reorder_operator_t, reorder_context_t> {
 public:
  /** @brief Parent type **/
  using parent_type = operator_t<reorder_operator_t, reorder_context_t>;
  size_t get_reorder_size();
  size_t reorder_size;

 protected:
  status_t validate() override;
  std::string op_create_info() override;
  std::string op_execute_info() override;
  status_t kernel_factory() override;

};

} //namespace ops

namespace interface {
using reorder_operator_t = zendnnl::ops::reorder_operator_t;
} //export

} //namespace zendnnl
#endif
