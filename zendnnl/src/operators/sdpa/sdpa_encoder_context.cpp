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

#include "sdpa_encoder_context.hpp"

namespace zendnnl {
namespace ops {

sdpa_encoder_context_t::sdpa_encoder_context_t() : op_context_t(), _scale(1.0f),
  _is_dropout(false), _is_causal(false), _has_mask(false) {
}

sdpa_encoder_context_t &sdpa_encoder_context_t::set_scale(float scale_) {
  LOG_DEBUG_INFO("Setting scale param in sdpa_encoder_context_t");
  _scale = scale_;
  return *this;
}

float sdpa_encoder_context_t::get_scale() const {
  LOG_DEBUG_INFO("Getting scale param from sdpa_encoder_context_t");
  return _scale;
}

sdpa_encoder_context_t &sdpa_encoder_context_t::set_is_dropout(
  bool is_dropout_) {
  LOG_DEBUG_INFO("Setting is_dropout param in sdpa_encoder_context_t");
  _is_dropout = is_dropout_;
  return *this;
}

bool sdpa_encoder_context_t::get_is_dropout() const {
  LOG_DEBUG_INFO("Getting is_dropout param from sdpa_encoder_context_t");
  return _is_dropout;
}

sdpa_encoder_context_t &sdpa_encoder_context_t::set_is_causal(bool is_causal_) {
  LOG_DEBUG_INFO("Setting is_causal param in sdpa_encoder_context_t");
  _is_causal = is_causal_;
  return *this;
}

bool sdpa_encoder_context_t::get_is_causal() const {
  LOG_DEBUG_INFO("Getting is_causal param from sdpa_encoder_context_t");
  return _is_causal;
}

sdpa_encoder_context_t &sdpa_encoder_context_t::set_has_mask(bool has_mask_) {
  LOG_DEBUG_INFO("Setting has_mask param in sdpa_encoder_context_t");
  _has_mask = has_mask_;
  return *this;
}

bool sdpa_encoder_context_t::get_has_mask() const {
  LOG_DEBUG_INFO("Getting has_mask param from sdpa_encoder_context_t");
  return _has_mask;
}

status_t sdpa_encoder_context_t::validate() {
  LOG_DEBUG_INFO("Validating sdpa_encoder_context_t");
  if (parent_type::validate() != status_t::success) {
    return status_t::failure;
  }

  // Validate required parameters for SDPA encoder
  auto query = get_param("query");
  auto key = get_param("key");
  auto value = get_param("value");

  if (!query) {
    apilog_error("Query parameter is null");
    return status_t::failure;
  }

  if (!key) {
    apilog_error("Key parameter is null");
    return status_t::failure;
  }

  if (!value) {
    apilog_error("Value parameter is null");
    return status_t::failure;
  }

  // Validate scale parameter
  if (_scale <= 0.0f) {
    apilog_error("Scale parameter must be positive, got: ", _scale);
    return status_t::failure;
  }

  return status_t::success;
}

std::string sdpa_encoder_context_t::context_info() {
  std::stringstream ss;
  auto query = get_param("query");
  auto key = get_param("key");
  auto value = get_param("value");

  ss << "SDPA Encoder context create";

  if (query) {
    ss << " - " << query.value().tensor_info();
  }
  if (key) {
    ss << ", " << key.value().tensor_info();
  }
  if (value) {
    ss << ", " << value.value().tensor_info();
  }

  ss << ", scale: " << get_scale();
  ss << ", is_dropout: " << (get_is_dropout() ? "true" : "false");
  ss << ", is_causal: " << (get_is_causal() ? "true" : "false");
  ss << ", has_mask: " << (get_has_mask() ? "true" : "false");

  return ss.str();
}


} //namespace ops
} //namespace zendnnl