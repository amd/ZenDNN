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

#include "matmul_ref_kernel.hpp"

namespace zendnnl {
namespace ops {
using namespace zendnnl::error_handling;

/**
 * @brief Convert BF16 value to float32 value using rounding to nearest-even.
 * @param bf16_val The BF16 value to be converted.
 * @return The converted float32 value.
 */
float bf16_to_float(int16_t bf16_val) {
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
inline __m256i float_to_bf16_avx512(__m512 val) {
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
void float32_to_bf16(const float *input, int16_t *output, size_t count) {
  log_info("Validating the conversion");
  size_t i = 0;
  for (; i + 15 < count; i += 16) {
    // Load 16 float32 values
    __m512 val = _mm512_loadu_ps(input + i);
    // Convert to BF16 with rounding
    __m256i bf16 = float_to_bf16_avx512(val);
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

status_t matmul_ref_kernel_t::execute(const context_type &context_,
                                      tensor_map_type &inputs_,
                                      tensor_map_type &outputs_) {
  LOG_DEBUG_INFO("Executing matmul_fp32_ref kernel");
  log_info("Executing matmul_fp32_ref kernel");

  auto  input_tensor           = inputs_.find("matmul_input")->second;
  auto  output_tensor          = outputs_.find("matmul_output")->second;
  auto  weight_tensor          = context_.get_param("weights").value();
  float alpha                  = context_.get_alpha();
  float beta                   = context_.get_beta();

  void *input                  = input_tensor.get_raw_handle_unsafe();
  void *output                 = output_tensor.get_raw_handle_unsafe();
  void *weights                = weight_tensor.get_raw_handle_unsafe();

  const int M                  = input_tensor.get_size(0);
  const int K                  = input_tensor.get_size(1);
  const int N                  = output_tensor.get_size(1);

  bool is_trans_src            = input_tensor.get_order() == "ba";
  bool is_trans_weights        = weight_tensor.get_order() == "ba";

  auto input_dtype             = input_tensor.get_data_type();
  auto weight_dtype            = weight_tensor.get_data_type();
  auto output_dtype            = output_tensor.get_data_type();

  const int   lda              = is_trans_src ? input_tensor.get_aligned_size(
                                   0) : input_tensor.get_aligned_size(1);
  const int   ldb              = is_trans_weights ? weight_tensor.get_aligned_size(
                                   0) : weight_tensor.get_aligned_size(1);
  const int   ldc              = output_tensor.get_aligned_size(1);

  // Interim accumaltion buffer with float type
  float *output_buff_f32       = (float *)aligned_alloc(64,
                                 M * N * sizeof(float));

  auto optional_bias_tensor        = context_.get_param("bias");
  [[maybe_unused]] void *bias      = nullptr;
  [[maybe_unused]] auto bias_dtype = data_type_t::f32;
  if (optional_bias_tensor) {
    auto bias_tensor           = context_.get_param("bias").value();
    bias                       = bias_tensor.get_raw_handle_unsafe();
    bias_dtype                 = bias_tensor.get_data_type();
  }

  for (auto i = 0; i < M; ++i) {
    for (auto j = 0; j < N; ++j) {
      float sum = 0.0f;
      size_t op_idx = i*ldc + j;
      for (auto k = 0; k < K; ++k) {
        size_t wt_idx = is_trans_weights ? (j*ldb + k) : (k*ldb + j);
        size_t ip_idx = is_trans_src ? (k*lda + i) : (i*lda + k);
        if (input_dtype == data_type_t::f32) {
          if (weight_dtype == data_type_t::f32) {
            sum += ((float *)input)[ip_idx] * ((float *)weights)[wt_idx];
          }
          else {
            sum += ((float *)input)[ip_idx] * bf16_to_float(((int16_t *)weights)[wt_idx]);
          }
        }
        else {
          if (weight_dtype == data_type_t::f32) {
            sum += bf16_to_float(((int16_t *)input)[ip_idx]) * ((float *)weights)[wt_idx];
          }
          else {
            sum += bf16_to_float(((int16_t *)input)[ip_idx]) * bf16_to_float(((
                     int16_t *)weights)[wt_idx]);
          }
        }
      }
      sum *= alpha;
      sum += output_dtype == data_type_t::bf16 ? bf16_to_float(((
               int16_t *)output)[op_idx]) * beta : ((float *)output)[op_idx] * beta;

      output_buff_f32[op_idx] = sum;
      if (optional_bias_tensor) {
        if (bias_dtype == data_type_t::f32) {
          output_buff_f32[op_idx] += ((float *)bias)[j];
        }
        else if (bias_dtype == data_type_t::bf16) {
          output_buff_f32[op_idx] += bf16_to_float(((int16_t *)bias)[j]);
        }
      }
    }
  }
  //Applying Post-op
  auto max_post_ops  = context_.get_post_op_count();
  int add_idx = 0;
  int mul_idx = 0;
  if (max_post_ops) {
    for (uint32_t i = 0; i < max_post_ops; ++ i) {
      post_op_t zen_po = context_.get_post_op(i);
      if (zen_po.type == post_op_type_t::binary_add) {
        std::string add_key = "binary_add_tensor_" + std::to_string(add_idx);
        auto binary_add_tensor = inputs_.find(add_key);
        if (binary_add_tensor == inputs_.end()) {
          return status_t::failure;
        }
        apply_post_op(output_tensor, binary_add_tensor->second, zen_po,
                      output_buff_f32);
        add_idx++;
      }
      else if (zen_po.type == post_op_type_t::binary_mul) {
        std::string mul_key = "binary_mul_tensor_" + std::to_string(mul_idx);
        auto binary_mul_tensor = inputs_.find(mul_key);
        if (binary_mul_tensor == inputs_.end()) {
          return status_t::failure;
        }
        apply_post_op(output_tensor, binary_mul_tensor->second, zen_po,
                      output_buff_f32);
        mul_idx++;
      }
      else {
        if (apply_post_op(output_tensor, zen_po,
                          output_buff_f32) != status_t::success) {
          return status_t::failure;
        }
      }
    }
  }
  if (output_dtype == data_type_t::bf16) {
    float32_to_bf16(output_buff_f32, (int16_t *)output, output_tensor.get_nelem());
  }
  else {
    for (uint64_t i = 0; i < output_tensor.get_nelem(); i++) {
      ((float *)output)[i] = output_buff_f32[i];
    }
  }
  if (output_buff_f32) {
    free(output_buff_f32);
  }
  return status_t::success;
}

status_t matmul_ref_kernel_t::apply_post_op(tensor_t &tensor_,
    tensor_t &buffer_tensor_, post_op_t zen_po_, float *output) {
  void *buffer       = buffer_tensor_.get_raw_handle_unsafe();
  auto  buff_data    = buffer_tensor_.get_data_type();

  const int size     = tensor_.get_nelem();
  const int buf_size = buffer_tensor_.get_nelem();

  if (zen_po_.type == post_op_type_t::binary_add) {
    float add_po_scale = zen_po_.binary_add_params.scale;
    for (int i = 0; i < size; ++i) {
      if (buff_data == data_type_t::bf16) {
        float temp = float(((bfloat16_t *)buffer)[i % buf_size]);
        output[i] = binary_add_fwd(output[i], temp, add_po_scale);
      }
      else {
        output[i] = binary_add_fwd(output[i], ((float *)buffer)[i % buf_size],
                                   add_po_scale);
      }
    }
  }
  else if (zen_po_.type == post_op_type_t::binary_mul) {
    float mul_po_scale = zen_po_.binary_mul_params.scale;
    for (int i = 0; i < size; ++i) {
      if (buff_data == data_type_t::bf16) {
        float temp = float(((bfloat16_t *)buffer)[i % buf_size]);
        output[i] = binary_mul_fwd(output[i], temp, mul_po_scale);
      }
      else {
        output[i] = binary_mul_fwd(output[i], ((float *)buffer)[i % buf_size],
                                   mul_po_scale);
      }
    }
  }
  return status_t::success;
}

status_t matmul_ref_kernel_t::apply_post_op(tensor_t &tensor_,
    post_op_t zen_po_, float *output) {
  switch (zen_po_.type) {
  case post_op_type_t::elu: {
    float alpha = zen_po_.elu_params.alpha;
    apply_eltwise_post_op(&matmul_ref_kernel_t::elu_fwd, tensor_, output,
                          alpha);
    break;
  }
  case post_op_type_t::relu:
    apply_eltwise_post_op(&matmul_ref_kernel_t::relu_fwd, tensor_, output);
    break;
  case post_op_type_t::leaky_relu: {
    float nslope = zen_po_.leaky_relu_params.nslope;
    apply_eltwise_post_op(&matmul_ref_kernel_t::leaky_relu_fwd, tensor_, output,
                          nslope);
    break;
  }
  case post_op_type_t::gelu_tanh:
    apply_eltwise_post_op(&matmul_ref_kernel_t::gelu_tanh_fwd, tensor_, output);
    break;
  case post_op_type_t::gelu_erf:
    apply_eltwise_post_op(&matmul_ref_kernel_t::gelu_erf_fwd, tensor_, output);
    break;
  case post_op_type_t::swish: {
    float scale = zen_po_.swish_params.scale;
    apply_eltwise_post_op(&matmul_ref_kernel_t::swish_fwd, tensor_, output,
                          scale);
    break;
  }
  case post_op_type_t::sigmoid: {
    apply_eltwise_post_op(&matmul_ref_kernel_t::sigmoid_fwd, tensor_, output);
    break;
  }
  case post_op_type_t::tanh:
    apply_eltwise_post_op(&matmul_ref_kernel_t::tanh_fwd, tensor_, output);
    break;
  case post_op_type_t::softmax:
    apply_softmax(tensor_, output);
    break;
  case post_op_type_t::square:
    apply_eltwise_post_op(&matmul_ref_kernel_t::square_fwd, tensor_, output);
    break;
  case post_op_type_t::abs:
    apply_eltwise_post_op(&matmul_ref_kernel_t::abs_fwd, tensor_, output);
    break;
  case post_op_type_t::sqrt:
    apply_eltwise_post_op(&matmul_ref_kernel_t::sqrt_fwd, tensor_, output);
    break;
  case post_op_type_t::exp:
    apply_eltwise_post_op(&matmul_ref_kernel_t::exp_fwd, tensor_, output);
    break;
  case post_op_type_t::log:
    apply_eltwise_post_op(&matmul_ref_kernel_t::log_fwd, tensor_, output);
    break;
  case post_op_type_t::clip: {
    float lower = zen_po_.clip_params.lower;
    float upper = zen_po_.clip_params.upper;
    apply_eltwise_post_op(&matmul_ref_kernel_t::clip_fwd, tensor_, output,
                          lower,
                          upper);
    break;
  }
  default:
    log_error("this postop is not supported in ref kernel");
    return status_t::unimplemented;
  }

  return status_t::success;
}

template<typename... Args>
status_t matmul_ref_kernel_t::apply_eltwise_post_op(float (
      matmul_ref_kernel_t::*post_op_func)(float, Args...),
    tensor_t &tensor_,
    float *output,
    Args... args) {
  uint64_t  size         = tensor_.get_nelem();

  for (uint64_t i = 0; i < size; ++i) {
    output[i] = (this->*post_op_func)(output[i], std::forward<Args>(args)...);
  }
  return status_t::success;
}

status_t matmul_ref_kernel_t::apply_softmax(tensor_t &tensor_,
    float *output) {
  const uint64_t rows    = tensor_.get_size(0);
  const uint64_t cols    = tensor_.get_size(1);

  for (uint64_t row = 0; row < rows; ++row) {
    // Compute exponentials for the current row
    double sumExp = 0.0;
    std::vector<double> expRow(cols);

    for (uint64_t col = 0; col < cols; ++col) {
      uint64_t index = row * cols + col;
      expRow[col] = expf((output[index]));
      sumExp += expRow[col];
    }

    // Normalize each exponential by the sum to get softmax probabilities
    for (uint64_t col = 0; col < cols; ++col) {
      size_t index = row * cols + col;
      output[index] = expRow[col] / sumExp;
    }
  }
  return status_t::success;
}

float matmul_ref_kernel_t::elu_fwd(float x, float alpha) {
  return x > 0 ? x : alpha * (expf(x) - 1.0);
}

float matmul_ref_kernel_t::relu_fwd(float x) {
  return x > 0 ? x : 0;
}

float matmul_ref_kernel_t::leaky_relu_fwd(float x, float nslope) {
  return x > 0 ? x : nslope * x;
}

float matmul_ref_kernel_t::gelu_tanh_fwd(float x) {
  float v = tanh_fwd(SQRT_2_OVER_PI * x * (1 + FITTING_CONST * x * x));
  return (0.5 * x * (1.0f + v));
}

float matmul_ref_kernel_t::gelu_erf_fwd(float x) {
  float v = x * SQRT_2_OVER_2;
  return (0.5f * x * (1.0f + erff(v)));
}

float matmul_ref_kernel_t::sigmoid_fwd(float x) {
  return (1.0 / (1.0 + expf(-x)));
}

float matmul_ref_kernel_t::swish_fwd(float x, float scale) {
  float scaled_x = x * scale;
  return x * (1.0 / (1.0 + expf(-scaled_x)));
}

float matmul_ref_kernel_t::tanh_fwd(float x) {
  const float e = tanhf(x);
  return e;
}

float matmul_ref_kernel_t::square_fwd(float x) {
  return x * x;
}

float matmul_ref_kernel_t::abs_fwd(float x) {
  return x > 0 ? x : -x;
}

float matmul_ref_kernel_t::sqrt_fwd(float x) {
  return sqrtf(x);
}

float matmul_ref_kernel_t::exp_fwd(float x) {
  return expf(x);
}

float matmul_ref_kernel_t::log_fwd(float x) {
  return logf(x);
}

float matmul_ref_kernel_t::clip_fwd(float x, float lower, float upper) {
  x = x > lower ? x : lower;
  return x > upper ? upper : x ;
}

float matmul_ref_kernel_t::binary_add_fwd(float x, float y, float scale) {
  return x + (y * scale) ;
}

float matmul_ref_kernel_t::binary_mul_fwd(float x, float y, float scale) {
  return x * (y * scale) ;
}

} //namespace ops
} //namespace zendnnl

extern "C" {
  std::shared_ptr<zendnnl::ops::matmul_ref_kernel_t>
  get_matmul_ref_kernel() {
    return std::make_shared<zendnnl::ops::matmul_ref_kernel_t>();
  }
}
