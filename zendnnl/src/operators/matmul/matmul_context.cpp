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
#include "matmul_context.hpp"

namespace zendnnl {
namespace ops {

status_t matmul_context_t::validate() {
  LOG_DEBUG_INFO("Validating matmul_context_t");
  if (parent_type::validate() != status_t::success) {
    return status_t::failure;
  }

  auto weights = get_param("weights");
  auto bias    = get_param("bias");

  if (!weights) {
    return status_t::failure;
  }

  auto weights_size = weights->get_size();
  if (weights_size.size() != 2) {
    return status_t::failure;
  }

  if (bias) {
    auto bias_size = bias->get_size();
    if (weights_size.at(1) != bias_size.at(0)) {
      return status_t::failure;
    }
  }

  return status_t::success;
}

status_t matmul_context_t::preprocess() {
  LOG_DEBUG_INFO("Preprocessing matmul_context_t");
  //aocl context pointer
  aocl_blis_utils_ptr = std::make_shared<aocl_blis_utils_t>();
  return status_t::success;
}

std::string matmul_context_t::context_info() {
  std::stringstream ss;
  auto weights = get_param("weights").value();
  auto bias    = get_param("bias");

  auto post_op_count = get_post_op_count();

  ss <<weights.tensor_info()<<",";

  if (bias) {
    ss <<bias.value().tensor_info()<<",";
  }

  ss <<"post-op";

  for (uint32_t i = 0; i < post_op_count; ++i) {
    post_op_t zen_po = get_post_op(i);
    ss << ":" << zen_po.post_op_info(zen_po);
  }

  return ss.str();
}

aocl_post_op *matmul_context_t::get_aocl_blis_post_op_ptr_unsafe() const {
  LOG_DEBUG_INFO("Getting aocl_blis_post_op_ptr from matmul_context_t");
  return aocl_blis_utils_ptr->get_aocl_blis_post_op_ptr_unsafe();
}

void *matmul_context_t::get_aocl_blis_reordered_weights_ptr_unsafe() const {
  LOG_DEBUG_INFO("Getting aocl_blis_reordered_weights_ptr from matmul_context_t");
  return aocl_blis_utils_ptr->get_aocl_blis_reordered_weights_ptr_unsafe();
}

} //namespace ops
} //namespace zendnnl

