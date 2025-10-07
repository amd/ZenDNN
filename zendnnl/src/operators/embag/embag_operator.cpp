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

#include "embag_operator.hpp"
#include "embag_kernel_list.hpp"
#include "common/platform_info.hpp"

namespace zendnnl {
namespace ops {

status_t embag_operator_t::preprocess() {
  LOG_DEBUG_INFO("Preprocessing embag_operator_t");

  return status_t::success;
}

status_t embag_operator_t::validate() {
  LOG_DEBUG_INFO("<", get_name(), "> Validating kernel input/output");
  if (!get_input("indices") || !get_output("output")) {
    apilog_error(name, " required input/output missing.");
    return status_t::failure;
  }

  //Get input-output dimensions
  auto table_sizes         = context.get_param("table")->get_size();
  auto is_weights          = context.get_is_weights();

  auto indices_sizes       = get_input("indices")->get_size();
  auto output_sizes        = get_output("output")->get_size();

  //Get input-output data type
  auto table_data_type   = context.get_param("table")->get_data_type();
  auto indices_data_type = get_input("indices")->get_data_type();
  auto output_data_type  = get_output("output")->get_data_type();

  if (output_sizes[1] != table_sizes[1]) {
    log_error(name, ": size mismatch in input/output/params. ", name,
              " output_size=", output_sizes[1], " table_size=", table_sizes[1]);
    return status_t::failure;
  }

  if ((table_data_type != data_type_t::f32 &&
       table_data_type != data_type_t::bf16) ||
      (output_data_type != data_type_t::f32 &&
       output_data_type != data_type_t::bf16)) {
    apilog_error(name, ": table and output datatype must be float32 or bfloat16");
    return status_t::failure;
  }

  if (indices_data_type != data_type_t::s64 &&
      indices_data_type != data_type_t::s32) {
    apilog_error(name, ": indices datatype must be int64 or int32");
    return status_t::failure;
  }

  // Validate optional offset tensor
  if (get_input("offsets")) {
    auto offsets_data_type = get_input("offsets")->get_data_type();
    if (offsets_data_type != data_type_t::s64 &&
        offsets_data_type != data_type_t::s32) {
      apilog_error(name, ": offsets datatype must be int64 or int32");
      return status_t::failure;
    }

    auto include_last_offset = context.get_include_last_offset();
    auto offsets_sizes       = get_input("offsets")->get_size();
    auto offsets_tensor      = get_input("offsets").value();
    [[maybe_unused]] const int32_t *offsets_data = static_cast<const int32_t *>
        (offsets_tensor.get_raw_handle_const());
    [[maybe_unused]] size_t batch_size = output_sizes[0];
    [[maybe_unused]] size_t num_indices = indices_sizes[0];

    if (include_last_offset) {
      assert(offsets_data[batch_size] <= static_cast<int32_t>(num_indices) &&
             "offsets[batch_size] must be <= indices_sizes");
    }
    else {
      assert(offsets_data[batch_size - 1] <= static_cast<int32_t>(num_indices) &&
             "offsets[batch_size-1] must be <= indices_sizes");
    }
  }

  if (get_input("weights")) {
    auto weights_data_type = get_input("weights")->get_data_type();
    if (weights_data_type != data_type_t::f32) {
      apilog_error(name, ": weights datatype must be float32");
      return status_t::failure;
    }
  }

  if ((is_weights && !get_input("weights")) ||
      (!is_weights && get_input("weights"))) {
    apilog_error(name, ": weights input is missing or is_weights is not enabled.");
    return status_t::failure;
  }

  return status_t::success;
}

status_t embag_operator_t::validate_forced_kernel() {
  LOG_DEBUG_INFO("<", get_name(), "> Validating forced kernel input/output");

  if (forced_kernel.empty()) {
    return status_t::success;
  }

  if (forced_kernel == "reference") {
    if (!get_input("indices") || !get_output("output")) {
      apilog_error(name, " required input/output missing.");
      return status_t::failure;
    }

    //Get input-output dimensions
    auto table_sizes         = context.get_param("table")->get_size();
    auto is_weights          = context.get_is_weights();

    auto indices_sizes       = get_input("indices")->get_size();
    auto output_sizes        = get_output("output")->get_size();

    //Get input-output data type
    auto table_data_type   = context.get_param("table")->get_data_type();
    auto indices_data_type = get_input("indices")->get_data_type();
    auto output_data_type  = get_output("output")->get_data_type();

    if (output_sizes[1] != table_sizes[1]) {
      log_error(name, ": size mismatch in input/output/params. ", name,
                " output_size=", output_sizes[1], " table_size=", table_sizes[1]);
      return status_t::failure;
    }

    if ((table_data_type != data_type_t::f32 &&
         table_data_type != data_type_t::bf16) ||
        (output_data_type != data_type_t::f32 &&
         output_data_type != data_type_t::bf16)) {
      apilog_error(name, ": table and output datatype must be float32 or bfloat16");
      return status_t::failure;
    }

    if (indices_data_type != data_type_t::s64 &&
        indices_data_type != data_type_t::s32) {
      apilog_error(name, ": indices datatype must be int64 or int32");
      return status_t::failure;
    }

    // Validate optional offset tensor
    if (get_input("offsets")) {
      auto offsets_data_type = get_input("offsets")->get_data_type();
      if (offsets_data_type != data_type_t::s64 &&
          offsets_data_type != data_type_t::s32) {
        apilog_error(name, ": offsets datatype must be int64 or int32");
        return status_t::failure;
      }

      auto include_last_offset = context.get_include_last_offset();
      auto offsets_sizes       = get_input("offsets")->get_size();
      auto offsets_tensor      = get_input("offsets").value();
      [[maybe_unused]] const int32_t *offsets_data = static_cast<const int32_t *>
          (offsets_tensor.get_raw_handle_const());
      [[maybe_unused]] size_t batch_size = output_sizes[0];
      [[maybe_unused]] size_t num_indices = indices_sizes[0];

      if (include_last_offset) {
        assert(offsets_data[batch_size] <= static_cast<int32_t>(num_indices) &&
               "offsets[batch_size] must be <= indices_sizes");
      }
      else {
        assert(offsets_data[batch_size - 1] <= static_cast<int32_t>(num_indices) &&
               "offsets[batch_size-1] must be <= indices_sizes");
      }
    }

    if (get_input("weights")) {
      auto weights_data_type = get_input("weights")->get_data_type();
      if (weights_data_type != data_type_t::f32) {
        apilog_error(name, ": weights datatype must be float32");
        return status_t::failure;
      }
    }

    if ((is_weights && !get_input("weights")) ||
        (!is_weights && get_input("weights"))) {
      apilog_error(name, ": weights input is missing or is_weights is not enabled.");
      return status_t::failure;
    }

    return status_t::success;
  }
  else {
    apilog_error("<", get_name(), "> ", forced_kernel,
                 " kernel can not be forced.");
    return status_t::failure;
  }
  return status_t::success;
}

std::string embag_operator_t::op_create_info() {
  std::stringstream ss;

  auto table = context.get_param("table").value();
  auto algo  = context.get_algo();

  if (algo == embag_algo_t::none) {
    ss << "Embedding operator create - ";
  }
  else {
    ss << "Embedding bag operator create - ";
  }
  if (!(get_name().empty())) {
    ss << get_name() << ",";
  }
  ss << table.tensor_info();
  if (algo == embag_algo_t::sum) {
    ss << ",algo:sum" ;
  }
  else if (algo == embag_algo_t::mean) {
    ss << ",algo:mean" ;
  }
  else if (algo == embag_algo_t::max) {
    ss << ",algo:max" ;
  }
  else {
    ss << "" ;
  }

  return ss.str();
}

std::string embag_operator_t::op_execute_info() {
  std::stringstream ss;

  auto indices  = get_input("indices");
  auto offsets  = get_input("offsets");
  auto output   = get_output("output");
  auto algo     = context.get_algo();

  if (!offsets && algo == embag_algo_t::none) {
    ss << "Embedding operator execute - ";
  }
  else {
    ss << "Embedding bag operator execute - ";
  }
  if (!(get_name().empty())) {
    ss << get_name() << ",";
  }

  if (forced_kernel.empty()) {
    ss << "kernel:native" << ",";
  }
  else {
    ss << "kernel:" << forced_kernel << ",";
  }

  ss << indices.value().tensor_info() << ",";
  if (offsets) {
    ss << offsets.value().tensor_info() << ",";
  }
  ss << output.value().tensor_info();

  if (algo == embag_algo_t::sum) {
    ss << ",algo:sum" ;
  }
  else if (algo == embag_algo_t::mean) {
    ss << ",algo:mean" ;
  }
  else if (algo == embag_algo_t::max) {
    ss << ",algo:max" ;
  }
  else {
    ss << "" ;
  }

  return ss.str();
}

status_t embag_operator_t::kernel_factory() {
  LOG_DEBUG_INFO("<", get_name(), "> Executing kernel_factory embag_operator_t");
  try {
    auto table_dtype   = context.get_param("table")->get_data_type();

    //get forced kernel if any
    if (forced_kernel.empty()) {

      if (table_dtype == data_type_t::f32) {
        if (platform_info.get_avx512f_status()) {
          kernel = std::shared_ptr<embag_f32_avx512_kernel_t>
                   (get_embag_f32_avx512_kernel());
        }
        else {
          kernel = std::shared_ptr<embag_f32_avx2_kernel_t>(get_embag_f32_avx2_kernel());
        }
      }
      else if (table_dtype == data_type_t::bf16) {
        if (platform_info.get_avx512f_status()) {
//TODO:To implement BF16 kernel for gcc<12
#if __GNUC__ >= 12
          kernel = std::shared_ptr<embag_bf16_avx512_kernel_t>
                   (get_embag_bf16_avx512_kernel());
#else
          kernel = std::shared_ptr<embag_bf16_avx2_kernel_t>
                   (get_embag_bf16_avx2_kernel());
#endif
        }
        else {
          kernel = std::shared_ptr<embag_bf16_avx2_kernel_t>
                   (get_embag_bf16_avx2_kernel());
        }
      }
      else {
        apilog_error("<", name, "> kernel unimplemented.");
        return status_t::unimplemented;
      }
    }
    else {
      if (forced_kernel == "reference") {
        kernel = std::shared_ptr<embag_ref_kernel_t>(get_embag_ref_kernel());
      }
      else {
        apilog_error("<", name, "> kernel unimplemented using forced kernel ",
                     forced_kernel);
        return status_t::unimplemented;
      }
    }

    //load_kernel(symbol);
    kernel->create();
    if (! kernel->check()) {
      return kernel->get_last_status();
    }

  }
  catch (const exception_t &ex) {
    EXCEPTION_WITH_LOC(ex.what());
  }

  return status_t::success;
}

} //namespace ops
} //namespace zendnnl
