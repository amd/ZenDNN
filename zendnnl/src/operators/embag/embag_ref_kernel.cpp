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

#include "embag_ref_kernel.hpp"

namespace zendnnl {
namespace ops {

using namespace zendnnl::memory;
using namespace zendnnl::error_handling;
using namespace zendnnl::common;

template <
  typename InType,
  typename IndexType,
  typename OffsetType,
  typename OutType>
void embag_ref_kernel(
  const InType *input,           // [num_embeddings, width] - embedding table
  const float *weights,          // [indsz] or nullptr if is_weights == false
  const IndexType *indices,     // [indsz] - indices into embedding table
  const OffsetType *offsets,    // [offsz] - start positions of each bag
  OutType *dst,                  // [offsz, width] with stride dst_stride - output buffer
  int64_t width,                 // embedding dimension
  int64_t indsz,                 // number of indices
  int64_t offsz,                 // number of bags
  int64_t padidx,                // padding index to skip
  bool is_weights,               // whether weights are used
  embag_algo_t algo,             // REDUCE_SUM, REDUCE_MEAN, REDUCE_MAX
  int64_t dst_stride,            // stride between output rows
  bool include_last_offset       // whether to include the last offset
) {

  // Determine if we need type conversions
  constexpr bool input_is_bf16 = std::is_same_v<InType, uint16_t>;
  constexpr bool output_is_bf16 = std::is_same_v<OutType, uint16_t>;

  // Temporary buffers for type conversion
  std::vector<float> temp_input_row;
  std::vector<float> temp_output_row;

  if constexpr(input_is_bf16) {
    temp_input_row.resize(static_cast<size_t>(width));
  }

  if constexpr(output_is_bf16) {
    temp_output_row.resize(static_cast<size_t>(width));
  }
  bool is_embedding = (offsets == nullptr) ? true : false;
  int outer_loop = is_embedding ? indsz : offsz;

  // Iterate over the offsets
  for (int oi = 0; oi < outer_loop; ++oi) {
    int64_t start = is_embedding ? oi : offsets[oi];
    int64_t end = is_embedding ? oi + 1 : (include_last_offset ? offsets[oi + 1] :
                                           (oi < offsz - 1 ? offsets[oi + 1] : indsz));
    auto dst_offset = oi * dst_stride;
    float wt_sum = 0;
    bool first_valid_index = true;

    // Get output row pointer (use temp buffer for BF16 output)
    float *output_row;
    if constexpr(output_is_bf16) {
      output_row = temp_output_row.data();
      // Initialize temp output row to zero
      std::fill(output_row, output_row + width, 0.0f);
    }
    else {
      output_row = reinterpret_cast<float *>(&dst[dst_offset]);
      // Initialize output row to zero
      std::fill(output_row, output_row + width, 0.0f);
    }

    // Process all indices in the current bag
    for (auto i = start; i < end; ++i) {
      if (indices[i] != padidx) {
        auto input_offset = indices[i] * width;
        auto wt = is_weights ? weights[i] : 1.0f;

        // Get input row pointer (convert from BF16 if needed)
        const float *input_row;
        if constexpr(input_is_bf16) {
          const uint16_t *bf16_row = reinterpret_cast<const uint16_t *>
                                     (&input[input_offset]);
          bfloat16_t::bf16_to_f32_buf(bf16_row, temp_input_row.data(), width);
          input_row = temp_input_row.data();
        }
        else {
          input_row = reinterpret_cast<const float *>(&input[input_offset]);
        }

        if (is_embedding) {
          for (auto j = 0; j < width; ++j) {
            output_row[j] = input_row[j];
          }
        }
        else {
          if (first_valid_index) {
            wt_sum = wt;
            // Initialize with first valid embedding
            for (auto j = 0; j < width; ++j) {
              if (algo != embag_algo_t::max) {
                output_row[j] = wt * input_row[j];
              }
              else {
                output_row[j] = input_row[j];
              }
            }
            first_valid_index = false;
          }
          else {
            // Compute embedding bags as per the algorithm
            if (algo == embag_algo_t::max) {
              for (auto j = 0; j < width; ++j) {
                if (output_row[j] < input_row[j]) {
                  output_row[j] = input_row[j];
                }
              }
            }
            else {
              wt_sum += wt;
              for (auto j = 0; j < width; ++j) {
                output_row[j] += wt * input_row[j];
              }
            }
          }
        }
      }
    }

    if (!is_embedding) {
      // Apply mean normalization if required
      if (algo == embag_algo_t::mean && wt_sum > 0) {
        for (auto j = 0; j < width; ++j) {
          output_row[j] /= wt_sum;
        }
      }
    }

    // Convert output back to BF16 if needed
    if constexpr(output_is_bf16) {
      int16_t *bf16_dst = reinterpret_cast<int16_t *>(&dst[dst_offset]);
      bfloat16_t::f32_to_bf16(temp_output_row.data(), bf16_dst, width);
    }
  }

}

status_t embag_ref_kernel_t::execute(const context_type &context_,
                                     tensor_map_type &inputs_,
                                     tensor_map_type &outputs_) {
  LOG_DEBUG_INFO("Executing embag_ref kernel");
  log_info("Executing embag_ref kernel");

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
  void        *dst      = dst_tensor.get_raw_handle_unsafe();
  float       *weights  = nullptr;

  const int64_t  width            = table_tensor.get_size(1);
  const int64_t  indsz            = indices_tensor.get_size(0);
  const int64_t  padidx           = context_.get_padding_index();
  int64_t stride                  = context_.get_scatter_stride();
  const embag_algo_t algo         = context_.get_algo();
  const bool include_last_offset  = context_.get_include_last_offset();
  const bool is_weights           = context_.get_is_weights();
  bool is_offsets                 = (offsets_iter != inputs_.end()) ? true :
                                    false;
  int64_t offsz                   = 0;

  // Get data types
  auto table_dtype = table_tensor.get_data_type();
  auto dst_dtype   = dst_tensor.get_data_type();
  auto indices_data_type = indices_tensor.get_data_type();
  auto offsets_data_type = is_offsets ? offsets_iter->second.get_data_type() :
                           data_type_t::none;

  // weights tensor is present
  if (is_weights) {
    if (weights_iter == inputs_.end()) {
      log_error("weights tensor not found but is_weights is true");
      return status_t::failure;
    }
    const tensor_t &weights_tensor = weights_iter->second;
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

  // Dispatch to appropriate template instantiation based on data types
  if (table_dtype == data_type_t::f32 && dst_dtype == data_type_t::f32) {
    // FP32 input -> FP32 output
    const float *input_f32 = reinterpret_cast<const float *>(input);
    float *dst_f32 = reinterpret_cast<float *>(dst);
    if (indices_data_type == data_type_t::s64) {
      int64_t *indices = (int64_t *)indices_tensor.get_raw_handle_unsafe();
      int64_t *offsets = nullptr;
      if (is_offsets && offsets_data_type == data_type_t::s64) {
        offsets = (int64_t *)offsets_iter->second.get_raw_handle_unsafe();
      }
      embag_ref_kernel<float, int64_t, int64_t, float>(
        input_f32, weights, indices, offsets, dst_f32, width, indsz, offsz,
        padidx, is_weights, algo, stride, include_last_offset);
    }
    else if (indices_data_type == data_type_t::s32) {
      int32_t *indices = (int32_t *)indices_tensor.get_raw_handle_unsafe();
      int32_t *offsets = nullptr;
      if (is_offsets && offsets_data_type == data_type_t::s32) {
        offsets = (int32_t *)offsets_iter->second.get_raw_handle_unsafe();
      }
      embag_ref_kernel<float, int32_t, int32_t, float>(
        input_f32, weights, indices, offsets, dst_f32, width, indsz, offsz,
        padidx, is_weights, algo, stride, include_last_offset);
    }
    else {
      apilog_error("Unsupported data type for indices and offsets");
    }
  }
  else if (table_dtype == data_type_t::f32 && dst_dtype == data_type_t::bf16) {
    // FP32 input -> BF16 output
    const float *input_f32 = reinterpret_cast<const float *>(input);
    uint16_t *dst_bf16 = reinterpret_cast<uint16_t *>(dst);
    if (indices_data_type == data_type_t::s64) {
      int64_t *indices = (int64_t *)indices_tensor.get_raw_handle_unsafe();
      int64_t *offsets = nullptr;
      if (is_offsets && offsets_data_type == data_type_t::s64) {
        offsets = (int64_t *)offsets_iter->second.get_raw_handle_unsafe();
      }
      embag_ref_kernel<float, int64_t, int64_t, uint16_t>(
        input_f32, weights, indices, offsets, dst_bf16, width, indsz, offsz,
        padidx, is_weights, algo, stride, include_last_offset);
    }
    else if (indices_data_type == data_type_t::s32) {
      int32_t *indices = (int32_t *)indices_tensor.get_raw_handle_unsafe();
      int32_t *offsets = nullptr;
      if (is_offsets && offsets_data_type == data_type_t::s32) {
        offsets = (int32_t *)offsets_iter->second.get_raw_handle_unsafe();
      }
      embag_ref_kernel<float, int32_t, int32_t, uint16_t>(
        input_f32, weights, indices, offsets, dst_bf16, width, indsz, offsz,
        padidx, is_weights, algo, stride, include_last_offset);
    }
    else {
      apilog_error("Unsupported data type for indices and offsets");
    }
  }
  else if (table_dtype == data_type_t::bf16 && dst_dtype == data_type_t::f32) {
    // BF16 input -> FP32 output
    const uint16_t *input_bf16 = reinterpret_cast<const uint16_t *>(input);
    float *dst_f32 = reinterpret_cast<float *>(dst);
    if (indices_data_type == data_type_t::s64) {
      int64_t *indices = (int64_t *)indices_tensor.get_raw_handle_unsafe();
      int64_t *offsets = nullptr;
      if (is_offsets && offsets_data_type == data_type_t::s64) {
        offsets = (int64_t *)offsets_iter->second.get_raw_handle_unsafe();
      }
      embag_ref_kernel<uint16_t, int64_t, int64_t, float>(
        input_bf16, weights, indices, offsets, dst_f32, width, indsz, offsz,
        padidx, is_weights, algo, stride, include_last_offset);
    }
    else if (indices_data_type == data_type_t::s32) {
      int32_t *indices = (int32_t *)indices_tensor.get_raw_handle_unsafe();
      int32_t *offsets = nullptr;
      if (is_offsets && offsets_data_type == data_type_t::s32) {
        offsets = (int32_t *)offsets_iter->second.get_raw_handle_unsafe();
      }
      embag_ref_kernel<uint16_t, int32_t, int32_t, float>(
        input_bf16, weights, indices, offsets, dst_f32, width, indsz, offsz,
        padidx, is_weights, algo, stride, include_last_offset);
    }
    else {
      apilog_error("Unsupported data type for indices and offsets");
    }
  }
  else if (table_dtype == data_type_t::bf16 && dst_dtype == data_type_t::bf16) {
    // BF16 input -> BF16 output
    const uint16_t *input_bf16 = reinterpret_cast<const uint16_t *>(input);
    uint16_t *dst_bf16 = reinterpret_cast<uint16_t *>(dst);
    if (indices_data_type == data_type_t::s64) {
      int64_t *indices = (int64_t *)indices_tensor.get_raw_handle_unsafe();
      int64_t *offsets = nullptr;
      if (is_offsets && offsets_data_type == data_type_t::s64) {
        offsets = (int64_t *)offsets_iter->second.get_raw_handle_unsafe();
      }
      embag_ref_kernel<uint16_t, int64_t, int64_t, uint16_t>(
        input_bf16, weights, indices, offsets, dst_bf16, width, indsz, offsz,
        padidx, is_weights, algo, stride, include_last_offset);
    }
    else if (indices_data_type == data_type_t::s32) {
      int32_t *indices = (int32_t *)indices_tensor.get_raw_handle_unsafe();
      int32_t *offsets = nullptr;
      if (is_offsets && offsets_data_type == data_type_t::s32) {
        offsets = (int32_t *)offsets_iter->second.get_raw_handle_unsafe();
      }
      embag_ref_kernel<uint16_t, int32_t, int32_t, uint16_t>(
        input_bf16, weights, indices, offsets, dst_bf16, width, indsz, offsz,
        padidx, is_weights, algo, stride, include_last_offset);
    }
    else {
      apilog_error("Unsupported data type for indices and offsets");
    }
  }
  else {
    LOG_DEBUG_INFO("Unsupported data type combination");
    return status_t::unimplemented;
  }

  return status_t::success;

}

// Template instantiations
template void embag_ref_kernel<float, int64_t, int64_t, float>(
  const float *, const float *, const int64_t *, const int64_t *, float *,
  int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t, int64_t, bool);

template void embag_ref_kernel<float, int32_t, int32_t, float>(
  const float *, const float *, const int32_t *, const int32_t *, float *,
  int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t, int64_t, bool);

template void embag_ref_kernel<float, int64_t, int64_t, uint16_t>(
  const float *, const float *, const int64_t *, const int64_t *, uint16_t *,
  int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t, int64_t, bool);

template void embag_ref_kernel<float, int32_t, int32_t, uint16_t>(
  const float *, const float *, const int32_t *, const int32_t *, uint16_t *,
  int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t, int64_t, bool);

template void embag_ref_kernel<uint16_t, int64_t, int64_t, uint16_t>(
  const uint16_t *, const float *, const int64_t *, const int64_t *, uint16_t *,
  int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t, int64_t, bool);

template void embag_ref_kernel<uint16_t, int32_t, int32_t, uint16_t>(
  const uint16_t *, const float *, const int32_t *, const int32_t *, uint16_t *,
  int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t, int64_t, bool);

template void embag_ref_kernel<uint16_t, int64_t, int64_t, float>(
  const uint16_t *, const float *, const int64_t *, const int64_t *, float *,
  int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t, int64_t, bool);

template void embag_ref_kernel<uint16_t, int32_t, int32_t, float>(
  const uint16_t *, const float *, const int32_t *, const int32_t *, float *,
  int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t, int64_t, bool);

extern "C" {
  embag_ref_kernel_t *get_embag_ref_kernel() {
    return new embag_ref_kernel_t();
  }
}

} //namespace ops
} //namespace zendnnl
