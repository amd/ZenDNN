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

#include "compare_ref_kernel.hpp"

namespace zendnnl {
namespace ops {
using namespace zendnnl::error_handling;

status_t compare_ref_kernel_t::execute(const context_type &context_,
                                       tensor_map_type &input_,
                                       tensor_map_type &output_) {
  LOG_DEBUG_INFO("Executing compare operator");

  auto     expec_tensor  = input_.at("expected_tensor");
  auto     test_tensor   = input_.at("test_tensor");
  auto     diff_tensor   = output_.at("diff_tensor");

  float   *expec_ptr  = (float *)expec_tensor.get_raw_handle_unsafe();
  float   *test_ptr   = (float *)test_tensor.get_raw_handle_unsafe();
  float   *diff_ptr   = (float *)diff_tensor.get_raw_handle_unsafe();

  auto stats        = context_.get_compare_stats();
  auto tolerance    = context_.get_tolerance();

  auto nelem        = expec_tensor.get_nelem();
  auto match_count  = 0;
  float sum_dev     = 0.0f;
  float max_dev     = std::numeric_limits<float>::lowest();
  float min_dev     = std::numeric_limits<float>::max();

  for (size_t i = 0; i<nelem; i++) {
    float diff = std::fabs(expec_ptr[i] - test_ptr[i]);
    diff_ptr[i] = diff;
    sum_dev += diff;

    if (diff > max_dev) {
      max_dev = diff;
    }

    if (diff < min_dev) {
      min_dev = diff;
    }

    // Check if the difference is within the tolerance range
    if (std::fabs(diff - tolerance) <= tolerance) {
      ++match_count;
    }
  }

  stats->match_percent  = static_cast<float>(match_count) / nelem * 100.0f;
  stats->max_deviation  = max_dev;
  stats->mean_deviation = sum_dev / nelem;
  stats->min_deviation  = min_dev;

  return status_t::success;
}

extern "C" {
  zendnnl::ops::compare_ref_kernel_t *get_compare_kernel() {
    return new zendnnl::ops::compare_ref_kernel_t();
  }
}
} //namespace ops
} //namespace zendnnl
