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
#ifndef _MATMUL_ONEDNN_KERNEL_HPP_
#define _MATMUL_ONEDNN_KERNEL_HPP_

#include "operators/common/operator_kernel.hpp"
#include "operators/matmul/matmul_context.hpp"

#include "matmul_onednn_utils.hpp"

namespace zendnnl {
namespace ops {

using namespace zendnnl::common;
using namespace zendnnl::memory;
using namespace zendnnl::error_handling;
#if ZENDNNL_DEPENDS_ONEDNN
  using namespace dnnl;
#endif

class matmul_onednn_kernel_t final : public op_kernel_t<matmul_context_t> {
 public:
  ~matmul_onednn_kernel_t();

  status_t execute(const context_type &context_,
                   tensor_map_type &inputs_,
                   tensor_map_type &outputs_) override;
#if ZENDNNL_DEPENDS_ONEDNN
  static void execute_matmul(const onednn_utils_t::onednn_matmul_params &params,
                             std::unordered_map<int, dnnl::memory> &matmul_args,
                             dnnl::primitive_attr &matmul_attr, dnnl::engine &eng);

 private:
  status_t preprocess(const context_type &context_, tensor_map_type &inputs_,
                      tensor_map_type &outputs_, onednn_utils_t::onednn_matmul_params &params,
                      std::unordered_map<int, dnnl::memory> &matmul_args,
                      dnnl::primitive_attr &matmul_attr, const dnnl::engine &eng);
#endif
};

} //namespace ops
} //namespace zendnnl

extern "C" {
  zendnnl::ops::matmul_onednn_kernel_t *get_matmul_onednn_kernel();
}

#endif
