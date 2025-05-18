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
#ifndef _EXAMPLE_UTILS_HPP_
#define _EXAMPLE_UTILS_HPP_

#include <vector>
#include <cstring>

#include "common/zendnnl_global.hpp"
#include "memory/tensor.hpp"


namespace zendnnl {
/** @namespace zendnnl::examples
 *  @brief A namespace that contains examples of how to use ZenDNNL.
 */
namespace examples {
using namespace zendnnl::memory;
using namespace zendnnl::error_handling;
using namespace zendnnl::common;

/** @class tensor_factory_t
 * @brief Quick generation of predefined tensors.
 */
class tensor_factory_t {
public:
  /** @brief Index type */
  using index_type = tensor_t::index_type;
  using data_type  = common::data_type_t;

  /** @brief zero tensor */
  tensor_t zero_tensor(const std::vector<index_type> size_, data_type dtype_);

  /** @brief uniform tensor */
  tensor_t uniform_tensor(const std::vector<index_type> size_, data_type dtype_,
                          float val_);
};

} //examples
} //zendnnl

#endif
