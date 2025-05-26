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
#ifndef _DLSAMPLE_OPERATOR_HPP_
#define _DLSAMPLE_OPERATOR_HPP_

#include "operator.hpp"
#include "dlsample_context.hpp"
#include "dlsample_kernel_list.hpp"
#include "error_handling.hpp"
#include "dynamic_module.hpp"

namespace zendnnl {
namespace ops {

using namespace zendnnl::common;
using namespace zendnnl::error_handling;

class dlsample_operator_t final : public operator_t<dlsample_operator_t, dlsample_context_t> {
public:
  using parent_type = operator_t<dlsample_operator_t, dlsample_context_t>;
  dlsample_operator_t& create() override;

protected:
  status_t validate()           override;
  status_t kernel_factory()     override;
  status_t preprocess();
};

status_t dlsample_operator_t::validate() {

  if (parent_type::validate() != status_t::success)
    return status_t::failure;

  if (!get_input("dlsample_input") || !get_output("dlsample_output"))
    return status_t::failure;

  return status_t::success;
}

status_t dlsample_operator_t::preprocess() {
  status_t return_status = status_t::failure;
  try {
    //dynamic module loading based on ISA
    std::string module_name;
    if(cpu_info.is_genoa())
      module_name = "dlsample_avx512_kernels";
    else
      module_name = "dlsample_avx2_kernels";

    return_status = load_module(module_name);
  } catch(const exception_t& ex) {
    EXCEPTION_WITH_LOC(ex.what());
  }

  return return_status;
}

dlsample_operator_t& dlsample_operator_t::create() {
  try {
    if (preprocess() != status_t::success) {
      log_error("<", name, "> failed to load module");
      return (*this);
    }
  } catch(const exception_t& ex) {
    EXCEPTION_WITH_LOC(ex.what());
  }

  parent_type::create();
  return (*this);
}

status_t dlsample_operator_t::kernel_factory() {
  try {
    auto input_dtype = get_input("dlsample_input")->get_data_type();

    std::string symbol;
    if (input_dtype == data_type_t::f32) {
      if(cpu_info.is_genoa())
        symbol = "get_dlsample_f32_avx512_kernel";
      else
        symbol = "get_dlsample_f32_avx2_kernel";
    }
    else if (input_dtype == data_type_t::bf16) {
      if(cpu_info.is_genoa())
        symbol = "get_dlsample_bf16_avx512_kernel";
      else
        symbol = "get_dlsample_bf16_avx2_kernel";
    }
    else {
      return status_t::unimplemented;
    }

    load_kernel(symbol);
    kernel->create();
    if (! kernel->check())
      return status_t::failure;

  } catch(const exception_t& ex) {
    EXCEPTION_WITH_LOC(ex.what());
  }

  return status_t::success;
}

} //namespace ops
} //namespace zendnnl
#endif
