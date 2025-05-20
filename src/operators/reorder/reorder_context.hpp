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
#ifndef _REORDER_CONTEXT_HPP_
#define _REORDER_CONTEXT_HPP_

#include "common/zendnnl_global.hpp"
#include "operators/common/operator_context.hpp"
#include "blis.h"

namespace zendnnl {
namespace ops {

/** @class reorder_context_t
 *  @brief reorder context for @c reorder_operator_t.
 *
 * In order to enable chaining, the first parameter in @c op_context_t template
 * should be the class itself.
 * @sa reorder_operator_t
 */
class reorder_context_t final : public op_context_t<reorder_context_t> {
 public:
  using parent_type = op_context_t<reorder_context_t>;

  /** @brief default constructor */
  reorder_context_t();

  /** @brief set backend algo */
  reorder_context_t &set_algo_format(std::string algo);

  /** @brief get backend algo */
  std::string get_algo_format() const;

  /** @brief set source data type */
  reorder_context_t &set_source_dtype(data_type_t dtype);

  /** @brief get source data type */
  data_type_t get_source_dtype() const;

 protected:
  /** @brief validate parameters */
  status_t validate() override;

  std::string algo_format;    /*!< Backend for reorder */
  data_type_t source_dtype;   /*!< Source Data type for u8/s8 input*/
};

} //namespace ops

namespace interface {
using reorder_context_t = zendnnl::ops::reorder_context_t;
} //export

} //namespace zendnnl
#endif
