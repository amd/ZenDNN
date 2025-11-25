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
#include "operators/matmul/matmul_config.hpp"
#include <string>

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

  // Describes a single tensor's properties for a OneDNN operation.
  struct onednn_tensor_params {
    void                   *buffer = nullptr;
    std::vector<int64_t>    dims;
    std::vector<int64_t>    strides;
    data_type_t             dtype = data_type_t::none;
    std::string             format_tag = "any"; // e.g., "ab", "abc"
    bool                    is_transposed = false;
  };

  // Holds all parameters for the complete matmul operation.
  struct onednn_matmul_params {
    // Tensor descriptors
    onednn_tensor_params src;
    onednn_tensor_params weights;
    onednn_tensor_params dst;
    onednn_tensor_params bias;

    // Scaling factors
    float alpha = 1.0f;
    float beta = 0.0f;

    matmul_algo_t algo = matmul_algo_t::none;
    bool is_blocked = false;
  };

#if ZENDNNL_DEPENDS_ONEDNN
  static dnnl::memory::format_tag to_dnnl_format(std::string tag);
  static dnnl::memory::data_type  to_dnnl_datatype(data_type_t dtype);

  static dnnl::memory::desc       to_dnnl_tensor(const onednn_tensor_params
      &params,
      dnnl::engine eng);
#endif

};

} //namespace ops
} //namespace zendnnl

#endif