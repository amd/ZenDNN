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
#include "reorder_operator_impl.hpp"
#include "reorder_kernel_list.hpp"
#include "aocl_dlp/reorder_utils.hpp"

#if ZENDNNL_DEPENDS_AOCLDLP
  #include "aocl_dlp.h"
#else
  #include "blis.h"
#endif

namespace zendnnl {
namespace ops {

status_t reorder_impl_t::validate() {
  if (parent_type::validate() != status_t::success) {
    return status_t::failure;
  }

  auto input        = get_input("reorder_input");
  auto output       = get_output("reorder_output");

  if (!input || !output) {
    apilog_error("Invalid input or output tensor.");
    return status_t::failure;
  }

  auto input_size  = input->get_size();
  auto output_size = output->get_size();

  bool memory_reorder         = ((!(input->get_layout() | uint16_t(
                                      tensor_layout_t::contiguous)) ||
                                  (input->get_layout() & uint16_t(tensor_layout_t::aligned))) &&
                                 (output->get_layout() & uint16_t(tensor_layout_t::blocked)));

  bool memory_unreorder       = ((input->get_layout() & uint16_t(
                                    tensor_layout_t::blocked)) &&
                                 !(output->get_layout() | uint16_t(tensor_layout_t::contiguous)));

  if (!(memory_reorder || memory_unreorder)) {
    apilog_error("Mismatch in layout is observed for conversion");
    return status_t::failure;
  }

  if (memory_reorder) {
    if (input->get_raw_handle_unsafe() == output->get_raw_handle_unsafe()) {
      size_t input_buffer_size = input->get_buffer_sz_bytes();

      if (reorder_size != input_buffer_size) {
        apilog_error("Reorder size mismatch, Inplace reorder doesn't work for given matrix: reorder_size=",
                     reorder_size, " input_buffer_size=", input_buffer_size);
        return status_t::failure;
      }
      else {
        apilog_info("Inplace reorder works for given matrix");
      }
    }
  }

  if ((input_size.size() != 2) || (output_size.size() != 2)) {
    apilog_error("Input or output size is not valid");
    return status_t::failure;
  }

  if (input_size.at(0) != output_size.at(0)) {
    apilog_error("Input and output size mismatch at dim - 0: input_size=",
                 input_size.at(0), " output_size=", output_size.at(0));
    return status_t::failure;
  }

  if (input_size.at(1) != output_size.at(1)) {
    apilog_error("Input and output size mismatch at dim - 1: input_size=",
                 input_size.at(1), " output_size=", output_size.at(1));
    return status_t::failure;
  }

  auto source_dtype  = context.get_source_dtype();
  if ((input->get_data_type() == data_type_t::s8) &&
      (!((source_dtype == data_type_t::s8) || (source_dtype == data_type_t::u8)))) {
    apilog_error("Source data type mismatch for s8: source_dtype=",
                 dtype_info(source_dtype));
    return status_t::failure;
  }

  return status_t::success;
}

std::string reorder_impl_t::op_create_info() {
  std::stringstream ss;

  ss << "Reorder operator create - ";
  if (!(get_name().empty())) {
    ss << get_name() << ",";
  }

  auto algo_format = context.get_algo_format();
  ss << "algo_format:" << algo_format;

  return ss.str();
}

std::string reorder_impl_t::op_execute_info() {
  std::stringstream ss;

  ss << "Reorder operator execute - ";
  if (!(get_name().empty())) {
    ss << get_name() << ",";
  }

  auto input       = get_input("reorder_input");
  auto output      = get_output("reorder_output");
  auto algo_format = context.get_algo_format();

  ss << input.value().tensor_info() << ","
     << output.value().tensor_info() << ","
     << "algo_format:" << algo_format;

  return ss.str();
}

status_t reorder_impl_t::kernel_factory() {
  auto algo_format = context.get_algo_format();

  if (algo_format == "aocl") {
    kernel = std::shared_ptr<reorder_kernel_t>(get_reorder_aocl_kernel());
  }
  else if (algo_format == "onednn") {
    apilog_error("onednn kernel is not supported");
    return status_t::unimplemented;
  }
  else {
    return status_t::unimplemented;
  }

  kernel->create();
  if (! kernel->check()) {
    return status_t::failure;
  }

  return status_t::success;
}

size_t reorder_impl_t::get_reorder_size() {
  auto algo_format   = context.get_algo_format();

  if (algo_format == "aocl") {
    auto input_tensor = get_input("reorder_input");
    reorder_size = aocl_dlp_reorder_utils_t::get_aocl_reorder_size(context,
                   *input_tensor);
  }
  else if (algo_format == "onednn") {
    apilog_error("onednn reorder is not supported");
  }
  else {
    apilog_error("Unsupported algorithm format for reorder");
  }
  return reorder_size;
}

} //namespace ops
} //namespace zendnnl

