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
#ifndef _EMBAG_OPERATOR_HPP_
#define _EMBAG_OPERATOR_HPP_

#include "operator.hpp"
#include "embag_context.hpp"
#include "embag_kernel_list.hpp"
#include "error_handling.hpp"
#include "dynamic_module.hpp"

namespace zendnnl {
namespace ops {

using namespace zendnnl::common;
using namespace zendnnl::error_handling;

class embag_operator_t final : public operator_t<embag_operator_t, embag_context_t> {
public:
  using parent_type = operator_t<embag_operator_t, embag_context_t>;

  embag_operator_t(embag_context_t& context_);
  embag_operator_t(std::string name_, embag_context_t& context_);

  embag_operator_t& pre_process() override;
protected:
  status_t context_sanity_check() override;
  status_t io_sanity_check() override;
  status_t kernel_factory() override;
};

embag_operator_t::embag_operator_t(embag_context_t& context_):
  operator_t{context_} {
}

embag_operator_t::embag_operator_t(std::string name_, embag_context_t& context_):
  operator_t{name_, context_} {
}

embag_operator_t& embag_operator_t::pre_process() {
  LOG_DEBUG_INFO("Preprocessing embag_operator_t");
  try {
    //sanity checks done in parent
    parent_type::pre_process();
    if (pre_process_status != status_t::success)
      return (*this);

    //dynamic module loading based on ISA
    std::string module_name;
    if(cpu_info.is_genoa())
      module_name = "embag_avx512_kernels";
    else
      module_name = "embag_avx2_kernels";

    status_t status = load_module(module_name);
    if (status != status_t::success) {
      pre_process_status = status;
      return (*this);
    }
  }
  catch(const exception_t& ex) {
    EXCEPTION_WITH_LOC(ex.what());
  }

  return (*this);
}

status_t embag_operator_t::context_sanity_check() {
  LOG_DEBUG_INFO("Context_sanity_check embag_operator_t");
  if (!context.get_param("table")) {
    log_error(name, " required parameters missing in the context.");
    return status_t::failure;
  }

  return status_t::success;
}

status_t embag_operator_t::io_sanity_check() {
  LOG_DEBUG_INFO("IO_sanity_check embag_operator_t");
  if (!get_input("indices") || !get_input("offsets") ||
      !get_output("output")) {
    log_error(name, " required input/output missing.");
    return status_t::failure;
  }

  //input output dimensions
  auto indices_sizes  = get_input("indices")->get_sizes();
  auto offsets_sizes  = get_input("offsets")->get_sizes();
  auto table_sizes    = context.get_param("table")->get_sizes();
  auto output_sizes   = get_output("output")->get_sizes();

  auto indices_data_type = get_input("indices")->get_data_type();
  auto offsets_data_type = get_input("offsets")->get_data_type();
  auto table_data_type   = context.get_param("table")->get_data_type();
  auto output_data_type  = get_output("output")->get_data_type();

  // if (output_sizes[0] != offsets_sizes[0])
  //   return status_t::failure;

  if ((output_sizes[1] != table_sizes[1]) || (table_data_type != output_data_type)) {
    log_error(name, ": size mismatch in input/output/params");
    return status_t::failure;
  }

  if ((indices_data_type != data_type_t::s32) || (offsets_data_type != data_type_t::s32)) {
    log_error(name, ": indices or offsets datatype is not int32");
    return status_t::failure;
  }

  return status_t::success;
}

status_t embag_operator_t::kernel_factory() {
  LOG_DEBUG_INFO("<", get_name(), "> Executing kernel_factory embag_operator_t");
  try {
    auto table_dtype   = context.get_param("table")->get_data_type();

    std::string symbol;
    if (table_dtype == data_type_t::f32) {
      if(cpu_info.is_genoa())
        symbol = "get_embag_f32_avx512_kernel";
      else
        symbol = "get_embag_f32_avx2_kernel";
    }
    else if (table_dtype == data_type_t::bf16) {
      if(cpu_info.is_genoa())
        symbol = "get_embag_bf16_avx512_kernel";
      else
        symbol = "get_embag_bf16_avx2_kernel";
    }
    else {
      return status_t::unimplemented;
    }

    load_kernel(symbol);

  } catch(const exception_t& ex) {
    EXCEPTION_WITH_LOC(ex.what());
  }

  return status_t::success;
}

} //namespace ops
} //namespace zendnnl
#endif
