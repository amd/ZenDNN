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

#include <cstdint>
#include "embag_context.hpp"

namespace zendnnl {
namespace ops {

embag_context_t::embag_context_t()
  : op_context_t(),
    algo{embag_algo_t::none},
    padding_index{-1},
    scatter_stride{-1},
    scatter_offset{0},
    include_last_offset{false},
    is_weights{false} {
}

embag_context_t &embag_context_t::set_algo(embag_algo_t algo_) {
  LOG_DEBUG_INFO("Setting algo for embag_context_t");
  algo = algo_;
  return *this;
}

embag_algo_t embag_context_t::get_algo() const {
  LOG_DEBUG_INFO("Getting algo for embag_context_t");
  return algo;
}

embag_context_t &embag_context_t::set_padding_index(int64_t padding_index_) {
  LOG_DEBUG_INFO("Setting padding index for embag_context_t");
  padding_index = padding_index_;
  return *this;
}

int64_t embag_context_t::get_padding_index() const {
  LOG_DEBUG_INFO("Getting padding index for embag_context_t");
  return padding_index;
}

embag_context_t &embag_context_t::set_scatter_stride(int64_t scatter_stride_) {
  LOG_DEBUG_INFO("Setting scatter_stride for embag_context_t");
  scatter_stride = scatter_stride_;
  return *this;
}

int64_t embag_context_t::get_scatter_stride() const {
  LOG_DEBUG_INFO("Getting scatter_stride for embag_context_t");
  return scatter_stride;
}

embag_context_t &embag_context_t::set_scatter_offset(int64_t scatter_offset_) {
  LOG_DEBUG_INFO("Setting scatter_offset for embag_context_t");
  scatter_offset = scatter_offset_;
  return *this;
}

int64_t embag_context_t::get_scatter_offset() const {
  LOG_DEBUG_INFO("Getting scatter_offset for embag_context_t");
  return scatter_offset;
}

embag_context_t &embag_context_t::set_include_last_offset(
  bool include_last_offset_) {
  LOG_DEBUG_INFO("Setting include_last_offset parameter for embag_context_t");
  include_last_offset = include_last_offset_;
  return *this;
}

bool embag_context_t::get_include_last_offset() const {
  LOG_DEBUG_INFO("Getting include_last_offset parameter for embag_context_t");
  return include_last_offset;
}

embag_context_t &embag_context_t::set_is_weights(bool is_weights_) {
  LOG_DEBUG_INFO("Setting is_weights parameter for embag_context_t");
  is_weights = is_weights_;
  return *this;
}

bool embag_context_t::get_is_weights() const {
  LOG_DEBUG_INFO("Getting is_weights parameter for embag_context_t");
  return is_weights;
}

status_t embag_context_t::validate() {
  LOG_DEBUG_INFO("Validating embag_context_t");
  if (parent_type::validate() != status_t::success) {
    return status_t::failure;
  }
  auto table = get_param("table");
  if (!table) {
    apilog_error("Table parameter is null");
    return status_t::failure;
  }

  return status_t::success;
}

std::string embag_context_t::context_info() {
  std::stringstream ss;
  auto table = get_param("table").value();

  if (algo == embag_algo_t::none) {
    ss << "Embedding context create - " << table.tensor_info();
  }
  else {
    ss << "Embedding bag context create - " << table.tensor_info();
    if (algo == embag_algo_t::sum) {
      ss << ",algo:sum" ;
    }
    else if (algo == embag_algo_t::mean) {
      ss << ",algo:mean" ;
    }
    else {
      ss << ",algo:max" ;
    }
    ss << ",include_last_offset:" << std::boolalpha
       << include_last_offset;
  }
  return ss.str();
}

} //namespace ops
} //namespace zendnnl
