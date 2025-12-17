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

namespace zendnnl {
namespace ops {

using namespace zendnnl::memory;
using namespace zendnnl::error_handling;

template <
  typename InType,
  typename IndexType,
  typename OffsetType,
  typename OutType>
void embag_avx512_kernel(const InType *input, const float *weights,
                         const IndexType *indices, const OffsetType *offsets,
                         OutType *dst, int64_t width, int64_t indsz,
                         int64_t offsz, int64_t padidx, bool is_weights,
                         embag_algo_t reduction_type, int64_t dst_stride);

template <
  bool IsInt4,
  typename InType,
  typename IndexType,
  typename OffsetType,
  typename OutType>
void embag_avx512_int8_int4_kernel(const InType *input, const float *weights,
                                   const IndexType *indices, const OffsetType *offsets,
                                   OutType *dst, int64_t width, int64_t indsz,
                                   int64_t offsz, int64_t padidx, bool is_weights,
                                   embag_algo_t reduction_type, int64_t dst_stride,
                                   bool include_last_offset, data_type_t table_dtype, bool fp16_scale_bias);

status_t embag_f32_avx512_kernel_t::execute(const context_type &context_,
    tensor_map_type &inputs_,
    tensor_map_type &outputs_) {
  LOG_DEBUG_INFO("Executing embag_f32_avx512_kernel_t");
  log_info("Executing embag_fp32_avx512 kernel");

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
  int64_t stride                  = context_.get_scatter_stride();
  const embag_algo_t algo         = context_.get_algo();
  const bool include_last_offset  = context_.get_include_last_offset();
  const bool is_weights           = context_.get_is_weights();
  int64_t offsz                   = 0;

  // weights tensor is present
  if (is_weights) {
    if (weights_iter == inputs_.end()) {
      log_error("weights tensor not found but is_weights is true");
      return status_t::failure;
    }
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

  if (stride==-1) {
    stride=width;
  }

  if (output_data_type == data_type_t::f32) {
    float *dst = (float *)dst_tensor.get_raw_handle_unsafe();
    if (indices_data_type == data_type_t::s64) {
      int64_t *indices = (int64_t *)indices_tensor.get_raw_handle_unsafe();
      int64_t *offsets = nullptr;
      if (is_offsets && offsets_data_type == data_type_t::s64) {
        offsets = (int64_t *)offsets_iter->second.get_raw_handle_unsafe();
      }
      embag_avx512_kernel<float, int64_t, int64_t, float>(
        input, weights, indices, offsets, dst, width, indsz, offsz,
        padidx, is_weights, algo, stride, include_last_offset);
    }
    else if (indices_data_type == data_type_t::s32) {
      int32_t *indices = (int32_t *)indices_tensor.get_raw_handle_unsafe();
      int32_t *offsets = nullptr;
      if (is_offsets && offsets_data_type == data_type_t::s32) {
        offsets = (int32_t *)offsets_iter->second.get_raw_handle_unsafe();
      }
      embag_avx512_kernel<float, int32_t, int32_t, float>(
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
      embag_avx512_kernel<float, int64_t, int64_t, uint16_t>(
        input, weights, indices, offsets, dst, width, indsz, offsz,
        padidx, is_weights, algo, stride, include_last_offset);
    }
    else if (indices_data_type == data_type_t::s32) {
      int32_t *indices = (int32_t *)indices_tensor.get_raw_handle_unsafe();
      int32_t *offsets = nullptr;
      if (is_offsets && offsets_data_type == data_type_t::s32) {
        offsets = (int32_t *)offsets_iter->second.get_raw_handle_unsafe();
      }
      embag_avx512_kernel<float, int32_t, int32_t, uint16_t>(
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

status_t embag_bf16_avx512_kernel_t::execute(const context_type &context_,
    tensor_map_type &inputs_,
    tensor_map_type &outputs_) {
  LOG_DEBUG_INFO("Executing embag_bf16_avx512_kernel_t");
  log_info("Executing embag_bf16_avx512 kernel");

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
  auto indices_data_type          = indices_tensor.get_data_type();
  auto offsets_data_type          = is_offsets ?
                                    offsets_iter->second.get_data_type() : data_type_t::none;
  auto output_data_type           = dst_tensor.get_data_type();
  const int64_t  padidx           = context_.get_padding_index();
  int64_t stride                  = context_.get_scatter_stride();
  const embag_algo_t algo         = context_.get_algo();
  const bool include_last_offset  = context_.get_include_last_offset();
  const bool is_weights           = context_.get_is_weights();
  int64_t offsz                   = 0;

  // weights tensor is present
  if (is_weights) {
    if (weights_iter == inputs_.end()) {
      log_error("weights tensor not found but is_weights is true");
      return status_t::failure;
    }
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

  if (stride==-1) {
    stride=width;
  }

  if (output_data_type == data_type_t::f32) {
    float *dst = (float *)dst_tensor.get_raw_handle_unsafe();
    if (indices_data_type == data_type_t::s64) {
      int64_t *indices = (int64_t *)indices_tensor.get_raw_handle_unsafe();
      int64_t *offsets = nullptr;
      if (is_offsets && offsets_data_type == data_type_t::s64) {
        offsets = (int64_t *)offsets_iter->second.get_raw_handle_unsafe();
      }
      embag_avx512_kernel<uint16_t, int64_t, int64_t, float>(
        input, weights, indices, offsets, dst, width, indsz, offsz,
        padidx, is_weights, algo, stride, include_last_offset);
    }
    else if (indices_data_type == data_type_t::s32) {
      int32_t *indices = (int32_t *)indices_tensor.get_raw_handle_unsafe();
      int32_t *offsets = nullptr;
      if (is_offsets && offsets_data_type == data_type_t::s32) {
        offsets = (int32_t *)offsets_iter->second.get_raw_handle_unsafe();
      }
      embag_avx512_kernel<uint16_t, int32_t, int32_t, float>(
        input, weights, indices, offsets, dst, width, indsz, offsz,
        padidx, is_weights, algo, stride, include_last_offset);
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
      embag_avx512_kernel<uint16_t, int64_t, int64_t, uint16_t>(
        input, weights, indices, offsets, dst, width, indsz, offsz,
        padidx, is_weights, algo, stride, include_last_offset);
    }
    else if (indices_data_type == data_type_t::s32) {
      int32_t *indices = (int32_t *)indices_tensor.get_raw_handle_unsafe();
      int32_t *offsets = nullptr;
      if (is_offsets && offsets_data_type == data_type_t::s32) {
        offsets = (int32_t *)offsets_iter->second.get_raw_handle_unsafe();
      }
      embag_avx512_kernel<uint16_t, int32_t, int32_t, uint16_t>(
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

status_t embag_int8_int4_avx512_kernel_t::execute(const context_type &context_,
    tensor_map_type &inputs_,
    tensor_map_type &outputs_) {
  LOG_DEBUG_INFO("Executing embag_int8_int4_avx512_kernel_t");
  log_info("Executing embag_int8_int4_avx512 kernel");

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

  int8_t const *input    = (const int8_t *)table_tensor.get_raw_handle_const();
  void         *dst      = dst_tensor.get_raw_handle_unsafe();
  float        *weights  = nullptr;

  const int64_t  width            = table_tensor.get_size(1);
  const int64_t  indsz            = indices_tensor.get_size(0);
  bool is_offsets                 = (offsets_iter != inputs_.end()) ? true :
                                    false;
  auto table_dtype                = table_tensor.get_data_type();
  auto dst_dtype                  = dst_tensor.get_data_type();
  auto indices_data_type          = indices_tensor.get_data_type();
  auto offsets_data_type          = is_offsets ?
                                    offsets_iter->second.get_data_type() : data_type_t::none;
  const int64_t  padidx           = context_.get_padding_index();
  int64_t stride                  = context_.get_scatter_stride();
  const embag_algo_t algo         = context_.get_algo();
  const bool include_last_offset  = context_.get_include_last_offset();
  const bool is_weights           = context_.get_is_weights();
  int64_t offsz                   = 0;
  const bool fp16_scale_bias      = context_.get_fp16_scale_bias();

  // weights tensor is present
  if (is_weights) {
    if (weights_iter == inputs_.end()) {
      log_error("weights tensor not found but is_weights is true");
      return status_t::failure;
    }
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

  if (stride==-1) {
    stride=width;
  }

  if (table_dtype == data_type_t::s8 && dst_dtype == data_type_t::f32) {
    // INT8 input -> FP32 output
    const int8_t *input_s8 = reinterpret_cast<const int8_t *>(input);
    float *dst_f32 = reinterpret_cast<float *>(dst);
    if (indices_data_type == data_type_t::s64) {
      int64_t *indices = (int64_t *)indices_tensor.get_raw_handle_unsafe();
      int64_t *offsets = nullptr;
      if (is_offsets && offsets_data_type == data_type_t::s64) {
        offsets = (int64_t *)offsets_iter->second.get_raw_handle_unsafe();
      }
      embag_avx512_int8_int4_kernel<false, int8_t, int64_t, int64_t, float>(
        input_s8, weights, indices, offsets, dst_f32, width, indsz, offsz,
        padidx, is_weights, algo, stride, include_last_offset, table_dtype,
        fp16_scale_bias);
    }
    else if (indices_data_type == data_type_t::s32) {
      int32_t *indices = (int32_t *)indices_tensor.get_raw_handle_unsafe();
      int32_t *offsets = nullptr;
      if (is_offsets && offsets_data_type == data_type_t::s32) {
        offsets = (int32_t *)offsets_iter->second.get_raw_handle_unsafe();
      }
      embag_avx512_int8_int4_kernel<false, int8_t, int32_t, int32_t, float>(
        input_s8, weights, indices, offsets, dst_f32, width, indsz, offsz,
        padidx, is_weights, algo, stride, include_last_offset, table_dtype,
        fp16_scale_bias);
    }
    else {
      apilog_error("Unsupported data type for indices and offsets");
    }
  }
  else if (table_dtype == data_type_t::s8 && dst_dtype == data_type_t::bf16) {
    // INT8 input -> BF16 output
    const int8_t *input_s8 = reinterpret_cast<const int8_t *>(input);
    uint16_t *dst_bf16 = reinterpret_cast<uint16_t *>(dst);
    if (indices_data_type == data_type_t::s64) {
      int64_t *indices = (int64_t *)indices_tensor.get_raw_handle_unsafe();
      int64_t *offsets = nullptr;
      if (is_offsets && offsets_data_type == data_type_t::s64) {
        offsets = (int64_t *)offsets_iter->second.get_raw_handle_unsafe();
      }
      embag_avx512_int8_int4_kernel<false, int8_t, int64_t, int64_t, uint16_t>(
        input_s8, weights, indices, offsets, dst_bf16, width, indsz, offsz,
        padidx, is_weights, algo, stride, include_last_offset, table_dtype,
        fp16_scale_bias);
    }
    else if (indices_data_type == data_type_t::s32) {
      int32_t *indices = (int32_t *)indices_tensor.get_raw_handle_unsafe();
      int32_t *offsets = nullptr;
      if (is_offsets && offsets_data_type == data_type_t::s32) {
        offsets = (int32_t *)offsets_iter->second.get_raw_handle_unsafe();
      }
      embag_avx512_int8_int4_kernel<false, int8_t, int32_t, int32_t, uint16_t>(
        input_s8, weights, indices, offsets, dst_bf16, width, indsz, offsz,
        padidx, is_weights, algo, stride, include_last_offset, table_dtype,
        fp16_scale_bias);
    }
    else {
      apilog_error("Unsupported data type for indices and offsets");
    }
  }
  else if ((table_dtype == data_type_t::s4 || table_dtype == data_type_t::u4) &&
           dst_dtype == data_type_t::f32) {
    // INT4 input -> FP32 output
    const uint8_t *input_s4 = reinterpret_cast<const uint8_t *>(input);
    float *dst_f32 = reinterpret_cast<float *>(dst);
    if (indices_data_type == data_type_t::s64) {
      int64_t *indices = (int64_t *)indices_tensor.get_raw_handle_unsafe();
      int64_t *offsets = nullptr;
      if (is_offsets && offsets_data_type == data_type_t::s64) {
        offsets = (int64_t *)offsets_iter->second.get_raw_handle_unsafe();
      }
      embag_avx512_int8_int4_kernel<true, uint8_t, int64_t, int64_t, float>(
        input_s4, weights, indices, offsets, dst_f32, width, indsz, offsz,
        padidx, is_weights, algo, stride, include_last_offset, table_dtype,
        fp16_scale_bias);
    }
    else if (indices_data_type == data_type_t::s32) {
      int32_t *indices = (int32_t *)indices_tensor.get_raw_handle_unsafe();
      int32_t *offsets = nullptr;
      if (is_offsets && offsets_data_type == data_type_t::s32) {
        offsets = (int32_t *)offsets_iter->second.get_raw_handle_unsafe();
      }
      embag_avx512_int8_int4_kernel<true, uint8_t, int32_t, int32_t, float>(
        input_s4, weights, indices, offsets, dst_f32, width, indsz, offsz,
        padidx, is_weights, algo, stride, include_last_offset, table_dtype,
        fp16_scale_bias);
    }
    else {
      apilog_error("Unsupported data type for indices and offsets");
    }
  }
  else if ((table_dtype == data_type_t::s4 || table_dtype == data_type_t::u4) &&
           dst_dtype == data_type_t::bf16) {
    // INT4 input -> BF16 output
    const uint8_t *input_s4 = reinterpret_cast<const uint8_t *>(input);
    uint16_t *dst_bf16 = reinterpret_cast<uint16_t *>(dst);
    if (indices_data_type == data_type_t::s64) {
      int64_t *indices = (int64_t *)indices_tensor.get_raw_handle_unsafe();
      int64_t *offsets = nullptr;
      if (is_offsets && offsets_data_type == data_type_t::s64) {
        offsets = (int64_t *)offsets_iter->second.get_raw_handle_unsafe();
      }
      embag_avx512_int8_int4_kernel<true, uint8_t, int64_t, int64_t, uint16_t>(
        input_s4, weights, indices, offsets, dst_bf16, width, indsz, offsz,
        padidx, is_weights, algo, stride, include_last_offset, table_dtype,
        fp16_scale_bias);
    }
    else if (indices_data_type == data_type_t::s32) {
      int32_t *indices = (int32_t *)indices_tensor.get_raw_handle_unsafe();
      int32_t *offsets = nullptr;
      if (is_offsets && offsets_data_type == data_type_t::s32) {
        offsets = (int32_t *)offsets_iter->second.get_raw_handle_unsafe();
      }
      embag_avx512_int8_int4_kernel<true, uint8_t, int32_t, int32_t, uint16_t>(
        input_s4, weights, indices, offsets, dst_bf16, width, indsz, offsz,
        padidx, is_weights, algo, stride, include_last_offset, table_dtype,
        fp16_scale_bias);
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
template void embag_avx512_kernel<float, int64_t, int64_t, float>(
  const float *, const float *, const int64_t *, const int64_t *, float *,
  int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t, int64_t, bool);

template void embag_avx512_kernel<float, int32_t, int32_t, float>(
  const float *, const float *, const int32_t *, const int32_t *, float *,
  int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t, int64_t, bool);

template void embag_avx512_kernel<float, int64_t, int64_t, uint16_t>(
  const float *, const float *, const int64_t *, const int64_t *, uint16_t *,
  int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t, int64_t, bool);

template void embag_avx512_kernel<float, int32_t, int32_t, uint16_t>(
  const float *, const float *, const int32_t *, const int32_t *, uint16_t *,
  int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t, int64_t, bool);

template void embag_avx512_kernel<uint16_t, int64_t, int64_t, uint16_t>(
  const uint16_t *, const float *, const int64_t *, const int64_t *, uint16_t *,
  int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t, int64_t, bool);

template void embag_avx512_kernel<uint16_t, int32_t, int32_t, uint16_t>(
  const uint16_t *, const float *, const int32_t *, const int32_t *, uint16_t *,
  int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t, int64_t, bool);

template void embag_avx512_kernel<uint16_t, int64_t, int64_t, float>(
  const uint16_t *, const float *, const int64_t *, const int64_t *, float *,
  int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t, int64_t, bool);

template void embag_avx512_kernel<uint16_t, int32_t, int32_t, float>(
  const uint16_t *, const float *, const int32_t *, const int32_t *, float *,
  int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t, int64_t, bool);

template void
embag_avx512_int8_int4_kernel<true, uint8_t, int64_t, int64_t, float>(
  const uint8_t *, const float *, const int64_t *, const int64_t *, float *,
  int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t, int64_t, bool,
  data_type_t, bool);

template void
embag_avx512_int8_int4_kernel<true, uint8_t, int32_t, int32_t, float>(
  const uint8_t *, const float *, const int32_t *, const int32_t *, float *,
  int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t, int64_t, bool,
  data_type_t, bool);

template void
embag_avx512_int8_int4_kernel<true, uint8_t, int64_t, int64_t, uint16_t>(
  const uint8_t *, const float *, const int64_t *, const int64_t *, uint16_t *,
  int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t, int64_t, bool,
  data_type_t, bool);

template void
embag_avx512_int8_int4_kernel<true, uint8_t, int32_t, int32_t, uint16_t>(
  const uint8_t *, const float *, const int32_t *, const int32_t *, uint16_t *,
  int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t, int64_t, bool,
  data_type_t, bool);

template void
embag_avx512_int8_int4_kernel<false, int8_t, int64_t, int64_t, float>(
  const int8_t *, const float *, const int64_t *, const int64_t *, float *,
  int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t, int64_t, bool,
  data_type_t, bool);

template void
embag_avx512_int8_int4_kernel<false, int8_t, int32_t, int32_t, float>(
  const int8_t *, const float *, const int32_t *, const int32_t *, float *,
  int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t, int64_t, bool,
  data_type_t, bool);

template void
embag_avx512_int8_int4_kernel<false, int8_t, int64_t, int64_t, uint16_t>(
  const int8_t *, const float *, const int64_t *, const int64_t *, uint16_t *,
  int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t, int64_t, bool,
  data_type_t, bool);

template void
embag_avx512_int8_int4_kernel<false, int8_t, int32_t, int32_t, uint16_t>(
  const int8_t *, const float *, const int32_t *, const int32_t *, uint16_t *,
  int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t, int64_t, bool,
  data_type_t, bool);

extern "C" {
  embag_f32_avx512_kernel_t *get_embag_f32_avx512_kernel() {
    return new embag_f32_avx512_kernel_t();
  }

  embag_bf16_avx512_kernel_t *get_embag_bf16_avx512_kernel() {
    return new embag_bf16_avx512_kernel_t();
  }

  embag_int8_int4_avx512_kernel_t *get_embag_int8_int4_avx512_kernel() {
    return new embag_int8_int4_avx512_kernel_t();
  }
}

} //namespace ops
} //namespace zendnnl

