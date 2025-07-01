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

#include "reorder_context.hpp"

namespace zendnnl {
namespace ops {

reorder_context_t::reorder_context_t() : algo_format{"aocl"}, source_dtype{} {}

status_t reorder_context_t::validate() {
  if (parent_type::validate() != status_t::success) {
    return status_t::failure;
  }

  // TODO: Add support for OneDNN backend
  if (!(get_algo_format() == "aocl")) {
    apilog_error("Unsupported algo format: " + get_algo_format());
    return status_t::failure;
  }

  return status_t::success;
}

reorder_context_t &reorder_context_t::set_algo_format(std::string algo) {
  algo_format = algo;
  return *this;
}

std::string reorder_context_t::get_algo_format() const {
  return algo_format;
}

reorder_context_t &reorder_context_t::set_source_dtype(data_type_t dtype) {
  source_dtype = dtype;
  return *this;
}

data_type_t reorder_context_t::get_source_dtype() const {
  return source_dtype;
}

std::string reorder_context_t::context_info() {
  std::stringstream ss;
  auto algo_format = get_algo_format();
  ss << "Reorder context create - ";
  if (algo_format.empty()) {
    ss << "";
  }
  else {
    ss << "algo_format:" << algo_format;
    auto src_type = dtype_info(get_source_dtype());
    if (src_type == "s8" || src_type =="u8") {
      ss << ",source_dtype:" << src_type;
    }
  }

  return ss.str();
}

} //namespace ops
} //namespace zendnnl
