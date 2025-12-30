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
#include "embag_avx2_kernels.hpp"
#include "embag_avx2_fp32_bf16_utils.hpp"

namespace zendnnl {
namespace ops {

using namespace zendnnl::memory;
using namespace zendnnl::error_handling;

template <
  typename InType,
  typename IndexType,
  typename OffsetType,
  typename OutType>
void embag_avx2_kernel(const InType *input, const float *weights,
                       const IndexType *indices, const OffsetType *offsets,
                       OutType *dst, int64_t width, int64_t indsz,
                       int64_t offsz, int64_t padidx, bool is_weights,
                       embag_algo_t reduction_type, int64_t dst_stride);

status_t embag_f32_avx2_kernel_t::execute(const context_type &context_,
    tensor_map_type &inputs_,
    tensor_map_type &outputs_) {
  LOG_DEBUG_INFO("Executing embag_f32_avx2_kernel_t");
  log_info("Executing embag_fp32_avx2 kernel");

  const auto table_param = context_.get_param("table");
  const auto &table_tensor = table_param.value();
  auto indices_iter = inputs_.find("indices");
  auto dst_iter = outputs_.find("output");
  auto offsets_iter   = inputs_.find("offsets");
  auto weights_iter   = inputs_.find("weights");

  if (indices_iter == inputs_.end()) {
    log_error("indices tensor not found");
    return status_t::failure;
  }
  if (dst_iter == outputs_.end()) {
    log_error("output tensor not found");
    return status_t::failure;
  }

  const auto &indices_tensor = indices_iter->second;
  const auto &dst_tensor = dst_iter->second;

  float const *input    = (const float *)table_tensor.get_raw_handle_const();
  float       *weights  = nullptr;

  const int64_t  width            = table_tensor.get_size(1);
  const int64_t  indsz            = indices_tensor.get_size(0);
  bool is_offsets                 = (offsets_iter != inputs_.end()) ? true :
                                    false;
  auto indices_data_type          = indices_tensor.get_data_type();
  auto offsets_data_type          = is_offsets ?
                                    offsets_iter->second.get_data_type() : data_type_t::none;
  auto output_data_type           = dst_tensor.get_data_type();
  const int64_t  padidx           = context_.get_padding_index();
  int64_t stride                  = dst_tensor.get_stride()[0];
  const embag_algo_t algo         = context_.get_algo();
  const bool include_last_offset  = context_.get_include_last_offset();
  const bool is_weights           = context_.get_is_weights();
  int64_t offsz                   = 0;

  // weights tensor is present
  if ((weights_iter != inputs_.end()) && is_weights) {
    const auto &weights_tensor = weights_iter->second;
    weights = (float *)weights_tensor.get_raw_handle_unsafe();
  }

  // Offsets tensor is optional - when not provided,
  // operates as simple embedding lookup rather than embedding bag aggregation
  if (is_offsets) {
    auto offsets_tensor = offsets_iter->second;
    offsz = offsets_tensor.get_size(0);
    if (include_last_offset==1) {
      offsz -= 1;
    }
  }

  if (output_data_type == data_type_t::f32) {
    float *dst = (float *)dst_tensor.get_raw_handle_unsafe();
    if (indices_data_type == data_type_t::s64) {
      int64_t *indices = (int64_t *)indices_tensor.get_raw_handle_unsafe();
      int64_t *offsets = nullptr;
      if (is_offsets && offsets_data_type == data_type_t::s64) {
        offsets = (int64_t *)offsets_iter->second.get_raw_handle_unsafe();
      }
      embag_avx2_kernel<float, int64_t, int64_t, float>(
        input, weights, indices, offsets, dst, width, indsz, offsz,
        padidx, is_weights, algo, stride, include_last_offset);
    }
    else if (indices_data_type == data_type_t::s32) {
      int32_t *indices = (int32_t *)indices_tensor.get_raw_handle_unsafe();
      int32_t *offsets = nullptr;
      if (is_offsets && offsets_data_type == data_type_t::s32) {
        offsets = (int32_t *)offsets_iter->second.get_raw_handle_unsafe();
      }
      embag_avx2_kernel<float, int32_t, int32_t, float>(
        input, weights, indices, offsets, dst, width, indsz, offsz,
        padidx, is_weights, algo, stride, include_last_offset);
    }
    else {
      apilog_error("Unsupported data type for indices and offsets");
    }
  }
  else if (output_data_type == data_type_t::bf16) {
    uint16_t *dst = (uint16_t *)dst_tensor.get_raw_handle_unsafe();
    if (indices_data_type == data_type_t::s64) {
      int64_t *indices = (int64_t *)indices_tensor.get_raw_handle_unsafe();
      int64_t *offsets = nullptr;
      if (is_offsets && offsets_data_type == data_type_t::s64) {
        offsets = (int64_t *)offsets_iter->second.get_raw_handle_unsafe();
      }
      embag_avx2_kernel<float, int64_t, int64_t, uint16_t>(
        input, weights, indices, offsets, dst, width, indsz, offsz,
        padidx, is_weights, algo, stride, include_last_offset);
    }
    else if (indices_data_type == data_type_t::s32) {
      int32_t *indices = (int32_t *)indices_tensor.get_raw_handle_unsafe();
      int32_t *offsets = nullptr;
      if (is_offsets && offsets_data_type == data_type_t::s32) {
        offsets = (int32_t *)offsets_iter->second.get_raw_handle_unsafe();
      }
      embag_avx2_kernel<float, int32_t, int32_t, uint16_t>(
        input, weights, indices, offsets, dst, width, indsz, offsz,
        padidx, is_weights, algo, stride, include_last_offset);
    }
    else {
      apilog_error("Unsupported data type for indices and offsets");
    }
  }
  else {
    apilog_error("kernel unimplemented.");
    return status_t::unimplemented;
  }
  return status_t::success;

}

status_t embag_bf16_avx2_kernel_t::execute(const context_type &context_,
    tensor_map_type &inputs_,
    tensor_map_type &outputs_) {
  LOG_DEBUG_INFO("Executing embag_bf16_avx2_kernel_t");
  log_info("Executing embag_bf16_avx2 kernel");

  const auto table_param = context_.get_param("table");
  const auto &table_tensor = table_param.value();
  auto indices_iter = inputs_.find("indices");
  auto dst_iter = outputs_.find("output");
  auto offsets_iter   = inputs_.find("offsets");
  auto weights_iter   = inputs_.find("weights");

  if (indices_iter == inputs_.end()) {
    log_error("indices tensor not found");
    return status_t::failure;
  }
  if (dst_iter == outputs_.end()) {
    log_error("output tensor not found");
    return status_t::failure;
  }

  const auto &indices_tensor = indices_iter->second;
  const auto &dst_tensor = dst_iter->second;

  uint16_t const *input   = (const uint16_t *)table_tensor.get_raw_handle_const();
  float          *weights = nullptr;

  const int64_t  width            = table_tensor.get_size(1);
  const int64_t  indsz            = indices_tensor.get_size(0);
  bool is_offsets                 = (offsets_iter != inputs_.end()) ? true :
                                    false;
  auto indices_data_type         = indices_tensor.get_data_type();
  auto offsets_data_type         = is_offsets ?
                                   offsets_iter->second.get_data_type() : data_type_t::none;
  auto output_data_type           = dst_tensor.get_data_type();
  const int64_t  padidx           = context_.get_padding_index();
  int64_t stride                  = dst_tensor.get_stride()[0];
  const embag_algo_t algo         = context_.get_algo();
  const bool include_last_offset  = context_.get_include_last_offset();
  const bool is_weights           = context_.get_is_weights();
  int64_t offsz                   = 0;

  // weights tensor is present
  if ((weights_iter != inputs_.end()) && is_weights) {
    const auto &weights_tensor = weights_iter->second;
    weights = (float *)weights_tensor.get_raw_handle_unsafe();
  }

  // Offsets tensor is optional - when not provided,
  // operates as simple embedding lookup rather than embedding bag aggregation
  if (is_offsets) {
    auto offsets_tensor = offsets_iter->second;
    offsz = offsets_tensor.get_size(0);
    if (include_last_offset==1) {
      offsz -= 1;
    }
  }

  if (output_data_type == data_type_t::f32) {
    float *dst = (float *)dst_tensor.get_raw_handle_unsafe();
    if (indices_data_type == data_type_t::s64) {
      int64_t *indices = (int64_t *)indices_tensor.get_raw_handle_unsafe();
      int64_t *offsets = nullptr;
      if (is_offsets && offsets_data_type == data_type_t::s64) {
        offsets = (int64_t *)offsets_iter->second.get_raw_handle_unsafe();
      }
      embag_avx2_kernel<uint16_t, int64_t, int64_t, float>(
        input, weights, indices, offsets, dst, width, indsz, offsz,
        padidx, is_weights, algo, stride, include_last_offset);
    }
    else if (indices_data_type == data_type_t::s32) {
      int32_t *indices = (int32_t *)indices_tensor.get_raw_handle_unsafe();
      int32_t *offsets = nullptr;
      if (is_offsets && offsets_data_type == data_type_t::s32) {
        offsets = (int32_t *)offsets_iter->second.get_raw_handle_unsafe();
      }
      embag_avx2_kernel<uint16_t, int32_t, int32_t, float>(
        input, weights, indices, offsets, dst, width, indsz, offsz,
        padidx, is_weights, algo, stride, include_last_offset);
    }
    else {
      apilog_error("Unsupported data type for indices and offsets");
    }
  }
  else if (output_data_type == data_type_t::bf16) {
    uint16_t *dst = (uint16_t *)dst_tensor.get_raw_handle_unsafe();
    if (indices_data_type == data_type_t::s64) {
      int64_t *indices = (int64_t *)indices_tensor.get_raw_handle_unsafe();
      int64_t *offsets = nullptr;
      if (is_offsets && offsets_data_type == data_type_t::s64) {
        offsets = (int64_t *)offsets_iter->second.get_raw_handle_unsafe();
      }
      embag_avx2_kernel<uint16_t, int64_t, int64_t, uint16_t>(
        input, weights, indices, offsets, dst, width, indsz, offsz,
        padidx, is_weights, algo, stride, include_last_offset);
    }
    else if (indices_data_type == data_type_t::s32) {
      int32_t *indices = (int32_t *)indices_tensor.get_raw_handle_unsafe();
      int32_t *offsets = nullptr;
      if (is_offsets && offsets_data_type == data_type_t::s32) {
        offsets = (int32_t *)offsets_iter->second.get_raw_handle_unsafe();
      }
      embag_avx2_kernel<uint16_t, int32_t, int32_t, uint16_t>(
        input, weights, indices, offsets, dst, width, indsz, offsz,
        padidx, is_weights, algo, stride, include_last_offset);
    }
    else {
      apilog_error("Unsupported data type for indices and offsets");
    }
  }
  else {
    apilog_error("kernel unimplemented.");
    return status_t::unimplemented;
  }

  return status_t::success;
}

// Template instantiations
template void embag_avx2_kernel<float, int64_t, int64_t, float>(
  const float *, const float *, const int64_t *, const int64_t *, float *,
  int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t, int64_t, bool);

template void embag_avx2_kernel<float, int32_t, int32_t, float>(
  const float *, const float *, const int32_t *, const int32_t *, float *,
  int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t, int64_t, bool);

template void embag_avx2_kernel<float, int64_t, int64_t, uint16_t>(
  const float *, const float *, const int64_t *, const int64_t *, uint16_t *,
  int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t, int64_t, bool);

template void embag_avx2_kernel<float, int32_t, int32_t, uint16_t>(
  const float *, const float *, const int32_t *, const int32_t *, uint16_t *,
  int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t, int64_t, bool);

template void embag_avx2_kernel<uint16_t, int64_t, int64_t, uint16_t>(
  const uint16_t *, const float *, const int64_t *, const int64_t *, uint16_t *,
  int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t, int64_t, bool);

template void embag_avx2_kernel<uint16_t, int32_t, int32_t, uint16_t>(
  const uint16_t *, const float *, const int32_t *, const int32_t *, uint16_t *,
  int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t, int64_t, bool);

template void embag_avx2_kernel<uint16_t, int64_t, int64_t, float>(
  const uint16_t *, const float *, const int64_t *, const int64_t *, float *,
  int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t, int64_t, bool);

template void embag_avx2_kernel<uint16_t, int32_t, int32_t, float>(
  const uint16_t *, const float *, const int32_t *, const int32_t *, float *,
  int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t, int64_t, bool);

extern "C" {
  embag_f32_avx2_kernel_t *get_embag_f32_avx2_kernel() {
    return new embag_f32_avx2_kernel_t();
  }
  embag_bf16_avx2_kernel_t *get_embag_bf16_avx2_kernel() {
    return new embag_bf16_avx2_kernel_t();
  }
}

} //namespace ops
} //namespace zendnnl

