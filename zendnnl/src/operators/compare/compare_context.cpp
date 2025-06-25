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
#include "compare_context.hpp"

namespace zendnnl {
namespace ops {

compare_context_t::compare_context_t()
  : op_context_t(),
    tolerance(0.0f),
    stats(std::make_shared<compare_stats_t>()) {
}

status_t compare_context_t::validate() {
  LOG_DEBUG_INFO("Validating compare context parameters");
  if (parent_type::validate() != status_t::success) {
    return status_t::failure;
  }

  if (get_tolerance() < 0) {
    apilog_error("Tolerance cannot be negative.");
    return status_t::failure;
  }
  return status_t::success;
}

compare_context_t &compare_context_t::set_tolerance(float tolerance_) {
  LOG_DEBUG_INFO("Setting tolerance");
  tolerance = tolerance_;
  return *this;
}

float compare_context_t::get_tolerance() const {
  LOG_DEBUG_INFO("Getting tolerance");
  return tolerance;
}

std::shared_ptr<compare_stats_t> compare_context_t::get_compare_stats() const {
  LOG_DEBUG_INFO("Getting compare statistics");
  return stats;
}

} //namespace ops
} //namespace zendnnl

