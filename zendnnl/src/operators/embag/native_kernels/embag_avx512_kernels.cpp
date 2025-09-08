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
#include "embag_avx512_kernels.hpp"
#include "embag_avx512_fp32_bf16_utils.hpp"

namespace zendnnl {
namespace ops {

using namespace zendnnl::memory;
using namespace zendnnl::error_handling;

template <typename InType, typename OutType>
void embag_avx512_kernel(const InType *input, const float *weights,
                         const int32_t *indices, const int32_t *offsets,
                         OutType *dst, int64_t width, int32_t indsz,
                         int32_t offsz, int32_t padidx, bool is_weights,
                         embag_algo_t reduction_type, int64_t dst_stride);

status_t embag_f32_avx512_kernel_t::execute(const context_type &context_,
    tensor_map_type &inputs_,
    tensor_map_type &outputs_) {
  LOG_DEBUG_INFO("Executing embag_f32_avx512_kernel_t");
  log_info("Executing embag_fp32_avx512 kernel");

  auto table_tensor   = context_.get_param("table").value();
  auto indices_tensor = inputs_.find("indices")->second;
  auto dst_tensor     = outputs_.find("output")->second;
  auto offsets_iter   = inputs_.find("offsets");
  auto weights_iter   = inputs_.find("weights");

  float const *input    = (const float *)table_tensor.get_raw_handle_const();
  int32_t     *indices  = (int32_t *)indices_tensor.get_raw_handle_unsafe();
  int32_t     *offsets  = nullptr;
  float       *weights  = nullptr;

  const int64_t  width            = table_tensor.get_size(1);
  const int64_t  indsz            = indices_tensor.get_size(0);
  auto output_data_type           = dst_tensor.get_data_type();
  const int64_t  padidx           = context_.get_padding_index();
  int64_t stride                  = context_.get_scatter_stride();
  const embag_algo_t algo         = context_.get_algo();
  const bool include_last_offset  = context_.get_include_last_offset();
  const bool is_weights           = context_.get_is_weights();
  int64_t offsz                   = 0;

  // weights tensor is present
  if ((weights_iter != inputs_.end()) && is_weights) {
    auto weights_tensor = weights_iter->second;
    weights = (float *)weights_tensor.get_raw_handle_unsafe();
  }

  // Offsets tensor is optional - when not provided,
  // operates as simple embedding lookup rather than embedding bag aggregation
  if (offsets_iter != inputs_.end()) {
    auto offsets_tensor = offsets_iter->second;
    offsets = (int32_t *)offsets_tensor.get_raw_handle_unsafe();
    offsz = offsets_tensor.get_size(0);
    if (include_last_offset==1) {
      offsz -= 1;
    }
  }

  if (stride==-1) {
    stride=width;
  }

  if (output_data_type == data_type_t::f32) {
    float *dst     = (float *)dst_tensor.get_raw_handle_unsafe();

    embag_avx512_kernel<float, float>(
      input, weights, indices, offsets, dst, width, indsz, offsz,
      padidx, is_weights, algo, stride, include_last_offset);
  }
  else if (output_data_type == data_type_t::bf16) {
    uint16_t *dst     = (uint16_t *)dst_tensor.get_raw_handle_unsafe();

    embag_avx512_kernel<float, uint16_t>(
      input, weights, indices, offsets, dst, width, indsz, offsz,
      padidx, is_weights, algo, stride, include_last_offset);
  }
  else {
    apilog_error("kernel unimplemented.");
    return status_t::unimplemented;
  }
  return status_t::success;

}

status_t embag_bf16_avx512_kernel_t::execute(const context_type &context_,
    tensor_map_type &inputs_,
    tensor_map_type &outputs_) {
  LOG_DEBUG_INFO("Executing embag_bf16_avx512_kernel_t");
  log_info("Executing embag_bf16_avx512 kernel");

  auto table_tensor   = context_.get_param("table").value();
  auto indices_tensor = inputs_.find("indices")->second;
  auto dst_tensor     = outputs_.find("output")->second;
  auto offsets_iter   = inputs_.find("offsets");
  auto weights_iter   = inputs_.find("weights");

  uint16_t const *input   = (const uint16_t *)table_tensor.get_raw_handle_const();
  int32_t        *indices = (int32_t *)indices_tensor.get_raw_handle_unsafe();
  int32_t        *offsets = nullptr;
  float          *weights = nullptr;

  const int64_t  width            = table_tensor.get_size(1);
  const int64_t  indsz            = indices_tensor.get_size(0);
  auto output_data_type           = dst_tensor.get_data_type();
  const int64_t  padidx           = context_.get_padding_index();
  int64_t stride                  = context_.get_scatter_stride();
  const embag_algo_t algo         = context_.get_algo();
  const bool include_last_offset  = context_.get_include_last_offset();
  const bool is_weights           = context_.get_is_weights();
  int64_t offsz                   = 0;

  // weights tensor is present
  if ((weights_iter != inputs_.end()) && is_weights) {
    auto weights_tensor = weights_iter->second;
    weights = (float *)weights_tensor.get_raw_handle_unsafe();
  }

  // Offsets tensor is optional - when not provided,
  // operates as simple embedding lookup rather than embedding bag aggregation
  if (offsets_iter != inputs_.end()) {
    auto offsets_tensor = offsets_iter->second;
    offsets = (int32_t *)offsets_tensor.get_raw_handle_unsafe();
    offsz = offsets_tensor.get_size(0);
    if (include_last_offset==1) {
      offsz -= 1;
    }
  }

  if (stride==-1) {
    stride=width;
  }

  if (output_data_type == data_type_t::f32) {
    float *dst     = (float *)dst_tensor.get_raw_handle_unsafe();

    embag_avx512_kernel<uint16_t, float>(
      input, weights, indices, offsets, dst, width, indsz, offsz,
      padidx, is_weights, algo, stride, include_last_offset);
  }
  else if (output_data_type == data_type_t::bf16) {
    uint16_t *dst     = (uint16_t *)dst_tensor.get_raw_handle_unsafe();

    embag_avx512_kernel<uint16_t, uint16_t>(
      input, weights, indices, offsets, dst, width, indsz, offsz,
      padidx, is_weights, algo, stride, include_last_offset);
  }
  else {
    apilog_error("kernel unimplemented.");
    return status_t::unimplemented;
  }

  return status_t::success;
}

// Template instantiations
template void embag_avx512_kernel<float, float>(
  const float *, const float *, const int32_t *, const int32_t *, float *,
  int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t, int64_t, bool);

template void embag_avx512_kernel<float, uint16_t>(
  const float *, const float *, const int32_t *, const int32_t *, uint16_t *,
  int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t, int64_t, bool);

template void embag_avx512_kernel<uint16_t, uint16_t>(
  const uint16_t *, const float *, const int32_t *, const int32_t *, uint16_t *,
  int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t, int64_t, bool);

template void embag_avx512_kernel<uint16_t, float>(
  const uint16_t *, const float *, const int32_t *, const int32_t *, float *,
  int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t, int64_t, bool);

extern "C" {
  embag_f32_avx512_kernel_t *get_embag_f32_avx512_kernel() {
    return new embag_f32_avx512_kernel_t();
  }

  embag_bf16_avx512_kernel_t *get_embag_bf16_avx512_kernel() {
    return new embag_bf16_avx512_kernel_t();
  }
}

} //namespace ops
} //namespace zendnnl

