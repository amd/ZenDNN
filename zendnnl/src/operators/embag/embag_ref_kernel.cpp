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

#include <immintrin.h>
#include "embag_ref_kernel.hpp"

namespace zendnnl {
namespace ops {

using namespace zendnnl::memory;
using namespace zendnnl::error_handling;

/**
 * @brief Convert BF16 value to float32 value using rounding to nearest-even.
 * @param bf16_val The BF16 value to be converted.
 * @return The converted float32 value.
 */
float embag_bf16_to_float(int16_t bf16_val) {
  int32_t inter_temp = *((int16_t *) &bf16_val);
  inter_temp = inter_temp << 16;
  float float_value = 0.0;
  memcpy(&float_value, &inter_temp, sizeof(int32_t));
  return float_value;
}

/**
 * @brief Convert 16 float32 values to 16 BF16 values using AVX512 instructions.
 * @param val The 16 float32 values packed in an AVX512 register.
 * @return The converted 16 BF16 values packed in an AVX512 register.
 */
__attribute__((target("avx512f")))
inline __m256i embag_float_to_bf16_avx512(__m512 val) {
  // Reinterpret float32 as int32 for bit manipulation
  __m512i int_val = _mm512_castps_si512(val);
  // Extract LSB of the BF16 part to determine rounding direction
  __m512i lsb = _mm512_and_si512(_mm512_srli_epi32(int_val, 16),
                                 _mm512_set1_epi32(1));
  // Add rounding bias (0x7FFF + lsb) for round-to-nearest-even
  __m512i rounding_bias = _mm512_add_epi32(_mm512_set1_epi32(0x7FFF), lsb);
  // Add bias to original bits
  __m512i rounded = _mm512_add_epi32(int_val, rounding_bias);
  // Shift right to extract upper 16 bits (BF16)
  __m512i bf16 = _mm512_srli_epi32(rounded, 16);
  // Narrow 32-bit integers to 16-bit integers
  return _mm512_cvtepi32_epi16(bf16);
}

/**
 * @brief Convert an array of float32 values to BF16 values with rounding.
 * @param input Pointer to the input array of float32 values.
 * @param output Pointer to the output array of BF16 values.
 * @param count Number of elements to convert.
 */
__attribute__((target("avx512f")))
void embag_float32_to_bf16(const float *input, int16_t *output, size_t count) {
  log_info("Validating the conversion");
  size_t i = 0;
  for (; i + 15 < count; i += 16) {
    // Load 16 float32 values
    __m512 val = _mm512_loadu_ps(input + i);
    // Convert to BF16 with rounding
    __m256i bf16 = embag_float_to_bf16_avx512(val);
    // Store 16 BF16 values
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(output + i), bf16);
  }
  // Handle remaining elements
  for (; i < count; ++i) {
    uint32_t bits;
    std::memcpy(&bits, &input[i], sizeof(float));
    uint32_t lsb = (bits >> 16) & 1;
    uint32_t rounding_bias = 0x7FFF + lsb;
    bits += rounding_bias;
    output[i] = static_cast<uint16_t>(bits >> 16);
  }
}

/**
 * @brief Convert a BF16 embedding row to float32 on-demand.
 * @param bf16_row Pointer to the BF16 embedding row.
 * @param f32_row Pointer to the output float32 row buffer.
 * @param width Width of the embedding row.
 */
void embag_convert_bf16_row_to_f32(const uint16_t *bf16_row, float *f32_row,
                                   int64_t width) {
  for (int64_t j = 0; j < width; ++j) {
    f32_row[j] = embag_bf16_to_float(static_cast<int16_t>(bf16_row[j]));
  }
}

template <typename InType, typename OutType>
void embag_ref_kernel(
  const InType *input,           // [num_embeddings, width] - embedding table
  const float *weights,          // [indsz] or nullptr if is_weights == false
  const int32_t *indices,        // [indsz] - indices into embedding table
  const int32_t *offsets,        // [offsz] - start positions of each bag
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

  // Iterate over the offsets
  for (int oi = 0; oi < offsz; ++oi) {
    int32_t start = offsets[oi];
    int32_t end = 0;
    if (include_last_offset==0) {
      end = oi < (offsz -1) ? offsets[oi+1] : indsz;
    }
    else {
      end = offsets[oi+1];
    }
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
          embag_convert_bf16_row_to_f32(bf16_row, temp_input_row.data(), width);
          input_row = temp_input_row.data();
        }
        else {
          input_row = reinterpret_cast<const float *>(&input[input_offset]);
        }

        if (first_valid_index) {
          wt_sum = wt;
          // Initialize with first valid embedding
          for (auto j = 0; j < width; ++j) {
            output_row[j] = wt * input_row[j];
          }
          first_valid_index = false;
        }
        else {
          // Compute embedding bags as per the algorithm
          if (algo == embag_algo_t::max) {
            for (auto j = 0; j < width; ++j) {
              float weighted_value = wt * input_row[j];
              if (output_row[j] < weighted_value) {
                output_row[j] = weighted_value;
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

    // Apply mean normalization if required
    if (algo == embag_algo_t::mean && wt_sum > 0) {
      for (auto j = 0; j < width; ++j) {
        output_row[j] /= wt_sum;
      }
    }

    // Convert output back to BF16 if needed
    if constexpr(output_is_bf16) {
      int16_t *bf16_dst = reinterpret_cast<int16_t *>(&dst[dst_offset]);
      embag_float32_to_bf16(temp_output_row.data(), bf16_dst, width);
    }
  }

}

status_t embag_ref_kernel_t::execute(const context_type &context_,
                                     tensor_map_type &inputs_,
                                     tensor_map_type &outputs_) {
  LOG_DEBUG_INFO("Executing embag_ref kernel");
  log_info("Executing embag_ref kernel");

  auto table_tensor   = context_.get_param("table").value();
  auto indices_tensor = inputs_.find("indices")->second;
  auto weights_iter   = inputs_.find("weights");
  auto offsets_tensor = inputs_.find("offsets")->second;
  auto dst_tensor     = outputs_.find("output")->second;

  void const  *input    = (const float *)table_tensor.get_raw_handle_const();
  float const *weights  = nullptr;
  int32_t     *indices  = (int32_t *)indices_tensor.get_raw_handle_unsafe();
  int32_t     *offsets  = (int32_t *)offsets_tensor.get_raw_handle_unsafe();
  void        *dst      = dst_tensor.get_raw_handle_unsafe();

  const int64_t  width             = table_tensor.get_size(1);
  const int64_t  indsz             = indices_tensor.get_size(0);
  int64_t        offsz             = offsets_tensor.get_size(0);
  const int64_t  padidx            = context_.get_padding_index();
  // const uint32_t scatter_offset = context_.get_scatter_offset();
  int64_t stride                   = context_.get_scatter_stride();
  const bool include_last_offset   = context_.get_include_last_offset();
  const embag_algo_t algo          = context_.get_algo();
  const bool is_weights            = context_.get_is_weights();

  // Get data types
  auto table_dtype = table_tensor.get_data_type();
  auto dst_dtype   = dst_tensor.get_data_type();

  // weights tensor is present
  if ((weights_iter != inputs_.end()) && is_weights) {
    auto weights_tensor = weights_iter->second;
    weights = (float *)weights_tensor.get_raw_handle_unsafe();
  }

  if (stride==-1) {
    stride=width;
  }

  if (include_last_offset==1) {
    offsz -= 1;
  }

  // Dispatch to appropriate template instantiation based on data types
  if (table_dtype == data_type_t::f32 && dst_dtype == data_type_t::f32) {
    // FP32 input -> FP32 output
    const float *input_f32 = reinterpret_cast<const float *>(input);
    float *dst_f32 = reinterpret_cast<float *>(dst);

    embag_ref_kernel<float, float>(
      input_f32, weights, indices, offsets, dst_f32, width, indsz, offsz,
      padidx, is_weights, algo, stride, include_last_offset);
  }
  else if (table_dtype == data_type_t::f32 && dst_dtype == data_type_t::bf16) {
    // FP32 input -> BF16 output
    const float *input_f32 = reinterpret_cast<const float *>(input);
    uint16_t *dst_bf16 = reinterpret_cast<uint16_t *>(dst);

    embag_ref_kernel<float, uint16_t>(
      input_f32, weights, indices, offsets, dst_bf16, width, indsz, offsz,
      padidx, is_weights, algo, stride, include_last_offset);
  }
  else if (table_dtype == data_type_t::bf16 && dst_dtype == data_type_t::f32) {
    // BF16 input -> FP32 output
    const uint16_t *input_bf16 = reinterpret_cast<const uint16_t *>(input);
    float *dst_f32 = reinterpret_cast<float *>(dst);

    embag_ref_kernel<uint16_t, float>(
      input_bf16, weights, indices, offsets, dst_f32, width, indsz, offsz,
      padidx, is_weights, algo, stride, include_last_offset);
  }
  else if (table_dtype == data_type_t::bf16 && dst_dtype == data_type_t::bf16) {
    // BF16 input -> BF16 output
    const uint16_t *input_bf16 = reinterpret_cast<const uint16_t *>(input);
    uint16_t *dst_bf16 = reinterpret_cast<uint16_t *>(dst);

    embag_ref_kernel<uint16_t, uint16_t>(
      input_bf16, weights, indices, offsets, dst_bf16, width, indsz, offsz,
      padidx, is_weights, algo, stride, include_last_offset);
  }
  else {
    LOG_DEBUG_INFO("Unsupported data type combination");
    return status_t::unimplemented;
  }

  return status_t::success;

}

// Template instantiations
template void embag_ref_kernel<float, float>(
  const float *, const float *, const int32_t *, const int32_t *, float *,
  int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t, int64_t, bool);

template void embag_ref_kernel<float, uint16_t>(
  const float *, const float *, const int32_t *, const int32_t *, uint16_t *,
  int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t, int64_t, bool);

template void embag_ref_kernel<uint16_t, uint16_t>(
  const uint16_t *, const float *, const int32_t *, const int32_t *, uint16_t *,
  int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t, int64_t, bool);

template void embag_ref_kernel<uint16_t, float>(
  const uint16_t *, const float *, const int32_t *, const int32_t *, float *,
  int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t, int64_t, bool);

extern "C" {
  std::shared_ptr<embag_ref_kernel_t> get_embag_ref_kernel() {
    return std::make_shared<embag_ref_kernel_t>();
  }
}

} //namespace ops
} //namespace zendnnl