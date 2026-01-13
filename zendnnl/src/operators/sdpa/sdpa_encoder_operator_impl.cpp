/********************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "sdpa_encoder_operator_impl.hpp"
#include "sdpa_encoder_kernel_list.hpp"

namespace zendnnl {
namespace ops {

status_t sdpa_encoder_impl_t::validate() {
  LOG_DEBUG_INFO("Validating sdpa_encoder_impl_t");

  if (parent_type::validate() != status_t::success) {
    return status_t::failure;
  }

  // Validate mandatory output
  if (!get_output("sdpa_output")) {
    apilog_error("Missing mandatory sdpa_output");
    return status_t::failure;
  }

  // Validate context parameters (Q, K, V tensors)
  auto query = context.get_param("query");
  auto key = context.get_param("key");
  auto value = context.get_param("value");

  if (!query || !key || !value) {
    apilog_error("Missing mandatory Q, K, V parameters in context");
    return status_t::failure;
  }

  // Validate optional mask parameter
  auto mask = context.get_param("mask");
  if (mask) {
    LOG_DEBUG_INFO("Mask parameter found in context");
  }
  else {
    LOG_DEBUG_INFO("No mask parameter provided - proceeding without mask");
  }

  // Validate tensor dimensions compatibility
  auto q_size = query->get_size();
  auto k_size = key->get_size();
  auto v_size = value->get_size();

  if (q_size.size() != 4 || k_size.size() != 4 || v_size.size() != 4) {
    apilog_error("Q, K, V tensors must be 4D [B, H, S, D]");
    return status_t::failure;
  }

  // Check dimension compatibility: [B, H, S, D]
  if (q_size[0] != k_size[0] || q_size[0] != v_size[0] ||  // Batch
      q_size[1] != k_size[1] || q_size[1] != v_size[1] ||  // Heads
      q_size[2] != k_size[2] || q_size[2] != v_size[2] ||  // Sequence
      q_size[3] != k_size[3] || q_size[3] != v_size[3]) {  // Dimension
    apilog_error("Q, K, V tensor dimensions are incompatible");
    return status_t::failure;
  }

  // Validate mask dimensions if mask is present
  if (mask) {
    auto mask_size = mask->get_size();

    // Mask should be compatible with attention matrix dimensions
    // Common mask formats: [B, H, S, S] or [B, 1, S, S] or [1, 1, S, S]
    if (mask_size.size() != 4) {
      apilog_error("Mask tensor must be 4D [B, H, S, S] or compatible format");
      return status_t::failure;
    }

    // Check mask compatibility with Q, K, V dimensions
    if ((mask_size[0] != 1 && mask_size[0] != q_size[0]) ||  // Batch: 1 or B
        (mask_size[1] != 1 && mask_size[1] != q_size[1]) ||  // Heads: 1 or H
        mask_size[2] != q_size[2] ||                         // Sequence length
        mask_size[3] != k_size[2]) {                         // Key sequence length
      apilog_error("Mask tensor dimensions are incompatible with Q, K, V tensors");
      return status_t::failure;
    }
  }

  return status_t::success;
}

std::string sdpa_encoder_impl_t::op_create_info() {
  std::stringstream ss;

  ss << "SDPA Encoder operator create - ";
  if (!(get_name().empty())) {
    ss << get_name();
  }

  // Add context information
  ss << ", scale: " << context.get_scale();
  ss << ", is_dropout: " << (context.get_is_dropout() ? "true" : "false");
  ss << ", is_causal: " << (context.get_is_causal() ? "true" : "false");
  ss << ", has_mask: " << (context.get_has_mask() ? "true" : "false");

  return ss.str();
}

std::string sdpa_encoder_impl_t::op_execute_info() {
  std::stringstream ss;

  ss << "SDPA Encoder operator execute - ";
  if (!(get_name().empty())) {
    ss << get_name() << ",";
  }
  auto output = get_output("sdpa_output");

  ss << "Output: " << output.value().tensor_info();

  // Add Q, K, V tensor info
  auto query = context.get_param("query");
  auto key = context.get_param("key");
  auto value = context.get_param("value");
  auto mask = context.get_param("mask");

  if (query && key && value) {
    ss << ", Q: " << query.value().tensor_info()
       << ", K: " << key.value().tensor_info()
       << ", V: " << value.value().tensor_info();
  }

  // Add mask tensor info if present
  if (mask) {
    ss << ", Mask: " << mask.value().tensor_info();
  }

  return ss.str();
}

status_t sdpa_encoder_impl_t::kernel_factory() {
  LOG_DEBUG_INFO("Creating SDPA encoder kernel");

  auto input_dtype = context.get_param("query")->get_data_type();

  if (input_dtype == data_type_t::f32) {
    kernel = std::shared_ptr<sdpa_encoder_fp32_kernel_t>
             (get_sdpa_encoder_fp32_kernel());  // SDPA FP32 kernel
  }
  else {
    apilog_error("Unsupported data type for SDPA encoder: ",
                 static_cast<int>(input_dtype));
    return status_t::unimplemented;
  }

  kernel->create();
  if (!kernel->check()) {
    apilog_error("SDPA encoder kernel creation failed");
    return status_t::failure;
  }

  return status_t::success;
}

} //namespace ops
} //namespace zendnnl
