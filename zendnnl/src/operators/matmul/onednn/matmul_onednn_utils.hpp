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

#ifndef _MATMUL_ONEDNN_UTILS_HPP_
#define _MATMUL_ONEDNN_UTILS_HPP_

#include "memory/tensor.hpp"
#if ZENDNNL_DEPENDS_ONEDNN
  #include "dnnl.hpp"
#endif

namespace zendnnl {
namespace ops {

using namespace zendnnl::memory;
using namespace zendnnl::error_handling;
using namespace zendnnl::common;
#if ZENDNNL_DEPENDS_ONEDNN
  using namespace dnnl;
#endif

class onednn_utils_t {
 public:
  onednn_utils_t() = default;
  ~onednn_utils_t() = default;
#if ZENDNNL_DEPENDS_ONEDNN
  static dnnl::memory::dims      to_dnnl_dims(tensor_t &zendnnl_tensor);
  static dnnl::memory::format_tag to_dnnl_format(tensor_t &zendnnl_tensor);
  static dnnl::memory::data_type  to_dnnl_datatype(tensor_t &zendnnl_tensor);

  static dnnl::memory             to_dnnl_tensor(tensor_t &zendnnl_tensor,
      dnnl::engine eng);
#endif

};

} //namespace ops
} //namespace zendnnl

#endif