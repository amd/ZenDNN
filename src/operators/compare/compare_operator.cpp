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
#include "compare_operator.hpp"
#include "compare_kernel_list.hpp"

namespace zendnnl {
namespace ops {

status_t compare_operator_t::validate() {
  LOG_DEBUG_INFO("Validating compare op parameters");
  if (parent_type::validate() != status_t::success) {
    return status_t::failure;
  }

  auto expec_tensor   = get_input("expected_tensor");
  auto test_tensor    = get_input("test_tensor");
  auto diff_tensor    = get_output("diff_tensor");

  auto expec_dtype    = expec_tensor->get_data_type();
  auto test_dtype     = test_tensor->get_data_type();
  auto diff_dtype     = diff_tensor->get_data_type();

  auto expec_layout   = expec_tensor->get_layout();
  auto test_layout    = test_tensor->get_layout();
  auto diff_layout    = diff_tensor->get_layout();

  auto expec_size = expec_tensor->get_size();
  auto test_size  = test_tensor->get_size();
  auto diff_size  = diff_tensor->get_size();

  //Checking shape of the tensors
  if ((expec_size.size() != test_size.size()) ||
      (test_size.size()  != diff_size.size()) ||
      (expec_size.size() != diff_size.size())) {
        log_error("<", get_name(), "> tensors are of same shape.");
    return status_t::failure;
  }

  for (auto i=0; i<expec_size.size(); i++) {
    if ((expec_size.at(i) != test_size.at(i)) ||
        (test_size.at(i)  != diff_size.at(i)) ||
        (expec_size.at(i) != diff_size.at(i))) {
      return status_t::failure;
    }
  }

  //Checking data type of the tensors
  if (expec_dtype  != test_dtype) {
    return status_t::failure;
  }

  if ((expec_layout != tensor_layout_t::contiguous) ||
      (test_layout  != tensor_layout_t::contiguous) ||
      (diff_layout  != tensor_layout_t::contiguous)) {
    log_error("<", get_name(), "> compare kernel needs contiguous tensors.");
    return status_t::failure;
  }

  if (diff_dtype != data_type_t::f32) {
    return status_t::failure;
  }

  return status_t::success;
}

status_t compare_operator_t::kernel_factory() {
  LOG_DEBUG_INFO("Executing compare operator");

  kernel = get_compare_kernel();

  kernel->create();
  if (! kernel->check()) {
    return status_t::failure;
  }

  return status_t::success;
}

compare_stats_t compare_operator_t::get_compare_stats() {
  return *(get_context().get_compare_stats());
}

} //namespace ops
} //namespace zendnnl

