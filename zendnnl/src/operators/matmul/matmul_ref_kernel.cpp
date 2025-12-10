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

#include "matmul_ref_kernel.hpp"
#include <cstring>
#include <vector>

namespace zendnnl {
namespace ops {
using namespace zendnnl::error_handling;
using namespace zendnnl::common;

inline void read_quant_params(const tensor_t &tensor,
                              scale_and_zero_point_t::quant_t &scale, scale_and_zero_point_t::quant_t &zp) {
  LOG_DEBUG_INFO("Read quant param in matmul_ref_kernel_t");
  // Read scale parameters
  scale.buff = tensor.get_quant_scale_raw_handle_const();
  scale.size = compute_product(tensor.get_quant_scale_size());
  scale.dt = tensor.get_quant_scale_data_type();

  // Read zero-point parameters if the tensor is asymmetric
  if (tensor.get_quant_subtype() == quant_subtype_t::asymmetric) {
    zp.buff = tensor.get_quant_zero_raw_handle_const();
    zp.size = compute_product(tensor.get_quant_zero_size());
    zp.dt = tensor.get_quant_zero_data_type();
  }
  else {
    zp.buff = nullptr;
    zp.size = 0;
    zp.dt = data_type_t::none;
  }
}

// Extract nibble from packed S4 byte (low nibble if is_low_nibble=true, else high nibble)
inline int8_t extract_s4_nibble(int8_t packed_byte, bool is_low_nibble) {
  int8_t s4_value = is_low_nibble ? (packed_byte & 0x0F) : ((packed_byte >> 4) & 0x0F);
  // Sign extend from bit 3
  if (s4_value & 0x08) {
    s4_value |= 0xF0;
  }
  return s4_value;
}

void matmul_ref_kernel_t::compute_zero_point_compensation(int M, int N, int K,
    char *src, int src_s0, int src_s1, int8_t *wei, int wei_s0, int wei_s1,
    int32_t *&zp_comp, int32_t src_zero_point, int32_t wei_zero_point,
    int &zp_comp_size) {
  LOG_DEBUG_INFO("Calculating zero-point compensation in zero_point_compensation");

  if (!wei_zero_point && !src_zero_point) {
    return;
  }
  else if (!wei_zero_point) {
    // zp_comp is freed in main function
    size_t alignment = 64;
    size_t comp_size = (N*sizeof(int32_t) + alignment - 1) &
                       ~(alignment - 1);
    zp_comp = (int32_t *)aligned_alloc(64, comp_size);
    std::vector<int32_t> wei_comp(N,0);
    zp_comp_size = N;

    for (auto k = 0; k < K; ++k) {
      for (auto n = 0; n < N; ++n) {
        if (k == 0) {
          wei_comp[n] = int32_t(0);
        }
        wei_comp[n] += wei[wei_s0 * k + wei_s1 * n];
      }
    }
    for (auto n = 0; n < N; ++n) {
      zp_comp[n] = 0 - src_zero_point * wei_comp[n];
    }
  }
  else if (!src_zero_point) {
    std::vector<int32_t> src_comp(M,0);
    // zp_comp is freed in main function
    size_t alignment = 64;
    size_t comp_size = (M*N*sizeof(int32_t) + alignment - 1) &
                       ~(alignment - 1);
    zp_comp = (int32_t *)aligned_alloc(64, comp_size);
    zp_comp_size = M*N;

    for (auto m = 0; m < M; ++m) {
      for (auto k = 0; k < K; ++k) {
        if (k == 0) {
          src_comp[m] = int32_t(0);
        }
        src_comp[m] += src[src_s0 * m + src_s1 * k];
      }
    }

    for (auto m = 0; m < M; ++m) {
      for (auto n = 0; n < N; ++n) {
        zp_comp[m * N + n] = 0 - wei_zero_point * src_comp[m];
      }
    }
  }
  else {
    std::vector<int32_t> src_comp(M,0);
    std::vector<int32_t> wei_comp(N,0);
    // zp_comp is freed in main function
    size_t alignment = 64;
    size_t comp_size = (M*N*sizeof(int32_t) + alignment - 1) &
                       ~(alignment - 1);
    zp_comp = (int32_t *)aligned_alloc(64, comp_size);
    zp_comp_size = M*N;
    //Src comp
    for (auto m = 0; m < M; ++m) {
      for (auto k = 0; k < K; ++k) {
        if (k == 0) {
          src_comp[m] = int32_t(0);
        }
        src_comp[m] += src[src_s0 * m + src_s1 * k];
      }
    }

    for (auto k = 0; k < K; ++k) {
      for (auto n = 0; n < N; ++n) {
        if (k == 0) {
          wei_comp[n] = int32_t(0);
        }
        wei_comp[n] += wei[wei_s0 * k + wei_s1 * n];
      }
    }

    for (auto m = 0; m < M; ++m) {
      for (auto n = 0; n < N; ++n) {
        zp_comp[m * N + n] = 0 - src_zero_point * wei_comp[n]
                             - wei_zero_point * src_comp[m]
                             + src_zero_point * wei_zero_point * (int)K;
      }
    }
  }
}

void matmul_ref_kernel_t::store_output(uint64_t nelem, float *accum_buff_f32,
                                       void *output, data_type_t output_dtype) {
  LOG_DEBUG_INFO("Storing matmul_ref_kernel_t output");

  if (output_dtype == data_type_t::u8) {
    for (uint64_t i = 0; i < nelem; ++i) {
      ((uint8_t *)output)[i] = (uint8_t)std::nearbyint(
                                 clip_fwd(accum_buff_f32[i], 0.0, (float)UINT8_MAX));
    }
  }
  else if (output_dtype == data_type_t::s8) {
    for (uint64_t i = 0; i < nelem; ++i) {
      ((int8_t *)output)[i] = (int8_t)std::nearbyint(
                                clip_fwd(accum_buff_f32[i], (float)INT8_MIN, (float)INT8_MAX));
    }
  }
  else if (output_dtype == data_type_t::s32) {
    for (uint64_t i = 0; i < nelem; ++i) {
      ((int32_t *)output)[i] = (int32_t)std::nearbyint(
                                 clip_fwd(accum_buff_f32[i], (float)INT32_MIN, (float)INT32_MAX));
    }
  }
  else if (output_dtype == data_type_t::bf16) {
    bfloat16_t::f32_to_bf16(accum_buff_f32, (int16_t *)output, nelem);
  }
  else {
    for (uint64_t i = 0; i < nelem; ++i) {
      ((float *)output)[i] = accum_buff_f32[i];
    }
  }
}

status_t matmul_ref_kernel_t::execute(const context_type &context_,
                                      tensor_map_type &inputs_,
                                      tensor_map_type &outputs_) {
  LOG_DEBUG_INFO("Executing matmul_ref kernel");
  log_info("Executing matmul_ref kernel");

  auto input_iter = inputs_.find("matmul_input");
  auto output_iter = outputs_.find("matmul_output");

  if (input_iter == inputs_.end()) {
    log_error("matmul_input tensor not found");
    return status_t::failure;
  }
  if (output_iter == outputs_.end()) {
    log_error("matmul_output tensor not found");
    return status_t::failure;
  }

  const auto &input_tensor = input_iter->second;
  const auto &output_tensor = output_iter->second;

  const auto weight_param = context_.get_param("weights");
  const auto &weight_tensor = weight_param.value();
  float alpha                  = context_.get_alpha();
  float beta                   = context_.get_beta();

  void *input                  = input_tensor.get_raw_handle_unsafe();
  void *output                 = output_tensor.get_raw_handle_unsafe();
  void *weights                = weight_tensor.get_raw_handle_unsafe();

  auto input_dim               = input_tensor.get_dim();
  auto weight_dim              = weight_tensor.get_dim();
  auto output_dim              = output_tensor.get_dim();

  auto input_dtype             = input_tensor.get_data_type();
  auto weight_dtype            = weight_tensor.get_data_type();
  auto output_dtype            = output_tensor.get_data_type();

  bool is_transpose_src        = (input_dim == 2)  ? (input_tensor.get_order() ==
                                 "ba") : (input_tensor.get_order() == "acb");
  bool is_transpose_weights    = (weight_dim == 2) ? (weight_tensor.get_order() ==
                                 "ba") : (weight_tensor.get_order() == "acb");

  const int batch_size         = (output_dim==3) ? output_tensor.get_size(
                                   output_dim-3) : 1;
  const int M                  = output_tensor.get_size(output_dim-2);
  const int K                  = input_tensor.get_size(input_dim-1);
  const int N                  = output_tensor.get_size(output_dim-1);

  const int   lda              = is_transpose_src ?
                                 input_tensor.get_stride(input_dim-1) :
                                 input_tensor.get_stride(input_dim-2);
  const int   ldb              = is_transpose_weights ?
                                 weight_tensor.get_stride(weight_dim-1):
                                 weight_tensor.get_stride(weight_dim-2);
  const int   ldc              = output_tensor.get_stride(output_dim-2);

  unsigned int offset_src      = (input_dim == 3) ? input_tensor.get_stride(
                                   input_dim-3) : 0;
  unsigned int offset_wei      = (weight_dim == 3) ? weight_tensor.get_stride(
                                   weight_dim-3) : 0;
  unsigned int offset_out      = (output_dim == 3) ? M*N : 0;
  bool is_int8                 = (input_dtype == data_type_t::s8 ||
                                  input_dtype == data_type_t::u8) &&
                                 weight_dtype == data_type_t::s8;
  bool is_woq                  = input_dtype == data_type_t::bf16 &&
                                 weight_dtype == data_type_t::s4;
  // Interim buffer  size
  size_t output_size = batch_size*M*N;
  // Interim accumulation buffer with float type
  tensor_t accum_tensor        = tensor_t()
                                 .set_size({output_size})
                                 .set_data_type(data_type_t::f32)
                                 .set_storage()
                                 .create();
  float *accum_buff_f32        = (float *)accum_tensor.get_raw_handle_unsafe();
  if (accum_buff_f32 == nullptr) {
    log_error("accum_buff_f32 can not have align allocation");
    return status_t::unimplemented;
  }

  auto optional_bias_tensor    = context_.get_param("bias");
  [[maybe_unused]] void *bias      = nullptr;
  [[maybe_unused]] auto bias_dtype = data_type_t::f32;
  if (optional_bias_tensor) {
    const auto &bias_tensor    = optional_bias_tensor.value();
    bias                       = (void *)bias_tensor.get_raw_handle_unsafe();
    bias_dtype                 = bias_tensor.get_data_type();
  }

  if (is_int8) {
    scale_and_zero_point_t quant_param;
    if (input_tensor.is_quantized()) {
      read_quant_params(input_tensor, quant_param.src_scale, quant_param.src_zp);
    }
    if (weight_tensor.is_quantized()) {
      read_quant_params(weight_tensor, quant_param.wei_scale, quant_param.wei_zp);
    }
    compute_quantized_matmul(batch_size, M, N, K, lda,
                             ldb, ldc, offset_src,
                             offset_wei, offset_out, alpha, beta, is_transpose_src, is_transpose_weights,
                             input, weights, bias, output, accum_buff_f32, input_dtype, weight_dtype,
                             bias_dtype, output_dtype, quant_param);
  }
  else if (is_woq) {
    // WOQ: Weight-Only Quantization (BF16/F32 input, S4 weights)
    // Convert S4 weights to BF16 and use compute_matmul
    scale_and_zero_point_t quant_param;
    if (weight_tensor.is_quantized()) {
      read_quant_params(weight_tensor, quant_param.wei_scale, quant_param.wei_zp);
    }
    else {
      log_error("WOQ requires quantized weights with scale factors");
      return status_t::failure;
    }

    // Allocate tensor for dequantized BF16 weights
    size_t weight_nelem = batch_size * K * N;
    tensor_t bf16_weight_tensor = tensor_t()
                                  .set_size({weight_nelem})
                                  .set_data_type(data_type_t::bf16)
                                  .set_storage()
                                  .create();
    bfloat16_t *bf16_weights = static_cast<bfloat16_t *>(bf16_weight_tensor.get_raw_handle_unsafe());
    if (bf16_weights == nullptr) {
      log_error("Failed to allocate BF16 weight buffer for WOQ dequantization");
      return status_t::unimplemented;
    }

    // Dequantize S4 weights to BF16
    // S4 weights are packed: 2 values per byte
    // The tensor's physical layout depends on its order (ab vs ba)
    const int8_t *packed_weights = (const int8_t *)weights;
    auto wei_scale_size = quant_param.wei_scale.size;
    bool has_zp = (quant_param.wei_zp.buff != nullptr);
    
    // Determine quantization granularity:
    // - Per-tensor:  wei_scale_size == 1
    // - Per-channel: wei_scale_size == N
    // - Per-group:   wei_scale_size == G * N, where G = K / group_size
    int num_groups = 1;
    int group_size = K;
    if (wei_scale_size > 1 && wei_scale_size != static_cast<size_t>(N)) {
      // Per-group quantization
      num_groups = wei_scale_size / N;
      group_size = K / num_groups;
    }

    // Use tensor strides for correct S4 data access
    // For S4, ldb represents stride in elements (not bytes)
    #pragma omp parallel for collapse(3)
    for (int bs = 0; bs < batch_size; ++bs) {
      for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
          // Calculate S4 element index using tensor strides
          // For non-transposed (ab): element[k,n] at k*ldb + n
          // For transposed (ba): element[k,n] at n*ldb + k
          size_t unpacked_idx = is_transpose_weights ?
                                (bs * offset_wei + n * ldb + k) :
                                (bs * offset_wei + k * ldb + n);
          size_t packed_byte_idx = unpacked_idx / 2;
          bool is_low_nibble = (unpacked_idx % 2) == 0;

          // Extract S4 value
          int8_t packed_byte = packed_weights[packed_byte_idx];
          int8_t s4_value = extract_s4_nibble(packed_byte, is_low_nibble);

          // Get scale based on quantization granularity
          // - Per-tensor:  index = 0
          // - Per-channel: index = n
          // - Per-group:   index = group_idx * N + n, where group_idx = k / group_size
          size_t scale_idx;
          if (wei_scale_size == 1) {
            scale_idx = 0;  // Per-tensor
          } else if (wei_scale_size == static_cast<size_t>(N)) {
            scale_idx = n;  // Per-channel
          } else {
            // Per-group: scale[group_idx, n]
            int group_idx = k / group_size;
            scale_idx = group_idx * N + n;
          }
          float scale = read_and_cast<float>(quant_param.wei_scale.buff,
                                             quant_param.wei_scale.dt,
                                             scale_idx);

          // Get zero point if available (same indexing logic)
          float zp = 0.0f;
          if (has_zp) {
            size_t zp_idx;
            auto zp_size = quant_param.wei_zp.size;
            if (zp_size == 1) {
              zp_idx = 0;  // Per-tensor
            } else if (zp_size == static_cast<size_t>(N)) {
              zp_idx = n;  // Per-channel
            } else {
              // Per-group: zp[group_idx, n]
              int group_idx = k / group_size;
              zp_idx = group_idx * N + n;
            }
            zp = read_and_cast<float>(quant_param.wei_zp.buff,
                                      quant_param.wei_zp.dt,
                                      zp_idx);
          }

          // Dequantize: (s4_value - zero_point) * scale
          float dequant_value = (static_cast<float>(s4_value) - zp) * scale;

          // Store in standard K x N layout (untranspose if needed)
          size_t bf16_idx = bs * K * N + k * N + n;
          bf16_weights[bf16_idx] = bfloat16_t(dequant_value);
        }
      }
    }

    // Dequantized weights are now in standard K x N layout
    int ldb_bf16 = N;
    unsigned int offset_wei_bf16 = K * N;

    // Call compute_matmul with dequantized BF16 weights (no longer transposed)
    compute_matmul(batch_size, M, N, K, lda, ldb_bf16, ldc,
                   offset_src, offset_wei_bf16, offset_out, alpha, beta, is_transpose_src,
                   false, input, bf16_weights, bias, output, accum_buff_f32,
                   input_dtype, data_type_t::bf16, bias_dtype, output_dtype);
  }
  else {
    compute_matmul(batch_size, M, N, K, lda, ldb, ldc,
                   offset_src, offset_wei, offset_out, alpha, beta, is_transpose_src,
                   is_transpose_weights, input, weights, bias, output, accum_buff_f32,
                   input_dtype, weight_dtype, bias_dtype, output_dtype);
  }
  //Applying Post-op
  if (apply_post_op(const_cast<tensor_t &>(output_tensor), inputs_,
                    accum_buff_f32, context_) != status_t::success) {
    return status_t::failure;
  }
  // Apply dst scale and zp
  if (is_int8) {
    quantize_dst(const_cast<tensor_t &>(output_tensor), accum_buff_f32);
  }
  uint64_t nelem = output_tensor.get_nelem();
  store_output(nelem, accum_buff_f32, output, output_dtype);

  return status_t::success;
}

status_t matmul_ref_kernel_t::apply_post_op(tensor_t &output_tensor,
    tensor_map_type &inputs_,
    float *accum_buff_f32,
    const context_type &context_) {
  LOG_DEBUG_INFO("Apply post ops in matmul_ref kernel");

  auto max_post_ops = context_.get_post_op_count();
  int add_idx = 0;
  int mul_idx = 0;

  for (uint32_t i = 0; i < max_post_ops; ++i) {
    post_op_t zen_po = context_.get_post_op(i);
    if (zen_po.type == post_op_type_t::binary_add) {
      std::string add_key = "binary_add_tensor_" + std::to_string(add_idx);
      auto binary_add_tensor = inputs_.find(add_key);
      if (binary_add_tensor == inputs_.end()) {
        return status_t::failure;
      }
      apply_post_op(output_tensor, binary_add_tensor->second, zen_po,
                    accum_buff_f32);
      add_idx++;
    }
    else if (zen_po.type == post_op_type_t::binary_mul) {
      std::string mul_key = "binary_mul_tensor_" + std::to_string(mul_idx);
      auto binary_mul_tensor = inputs_.find(mul_key);
      if (binary_mul_tensor == inputs_.end()) {
        return status_t::failure;
      }
      apply_post_op(output_tensor, binary_mul_tensor->second, zen_po,
                    accum_buff_f32);
      mul_idx++;
    }
    else {
      if (apply_post_op(output_tensor, std::move(zen_po),
                        accum_buff_f32) != status_t::success) {
        return status_t::failure;
      }
    }
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
      float temp = read_and_cast<float>(buffer, buff_data, i % buf_size);
      output[i] = binary_add_fwd(output[i], temp, add_po_scale);
    }
  }
  else if (zen_po_.type == post_op_type_t::binary_mul) {
    float mul_po_scale = zen_po_.binary_mul_params.scale;
    for (int i = 0; i < size; ++i) {
      float temp = read_and_cast<float>(buffer, buff_data, i % buf_size);
      output[i] = binary_mul_fwd(output[i], temp, mul_po_scale);
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
    output[i] = (this->*post_op_func)(output[i], args...);
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

void matmul_ref_kernel_t::quantize_dst(tensor_t &output_tensor,
                                       float *accum_buff_f32) {
  LOG_DEBUG_INFO("Quantizing output matmul_ref");

  if (output_tensor.is_quantized()) {
    const void *dst_scale = output_tensor.get_quant_scale_raw_handle_const();
    int dst_scale_size = compute_product(output_tensor.get_quant_scale_size());
    data_type_t dst_scale_dt = output_tensor.get_quant_scale_data_type();

    [[maybe_unused]] const void *dst_zp = nullptr;
    [[maybe_unused]] int dst_zp_size = 0;
    [[maybe_unused]] data_type_t dst_zp_dt = data_type_t::none;
    if (output_tensor.get_quant_subtype() == quant_subtype_t::asymmetric) {
      dst_zp = output_tensor.get_quant_zero_raw_handle_const();
      dst_zp_size = compute_product(output_tensor.get_quant_zero_size());
      dst_zp_dt = output_tensor.get_quant_zero_data_type();
    }

    for (uint64_t i = 0; i < output_tensor.get_nelem(); ++i) {
      accum_buff_f32[i] *= read_and_cast<float>(dst_scale, dst_scale_dt,
                           i % dst_scale_size);
      if (dst_zp_size != 0) {
        accum_buff_f32[i] += read_and_cast<int32_t>(dst_zp, dst_zp_dt,
                             i % dst_zp_size);
      }
    }
  }
}

void matmul_ref_kernel_t::compute_matmul(int batch_size, int M, int N, int K,
    int lda, int ldb, int ldc, unsigned int offset_src, unsigned int offset_wei,
    unsigned int offset_out, float alpha, float beta, bool is_transpose_src,
    bool is_transpose_weights, const void *input, const void *weights,
    const void *bias, const void *output, float *accum_buff_f32,
    data_type_t input_dtype, data_type_t weight_dtype, data_type_t bias_dtype,
    data_type_t output_dtype) {
  LOG_DEBUG_INFO("Computing MatMul_ref");

  #pragma omp parallel for collapse(3)
  for (auto bs = 0; bs < batch_size; ++bs) {
    for (auto i = 0; i < M; ++i) {
      for (auto j = 0; j < N; ++j) {
        float sum = 0.0;
        size_t op_idx = bs * offset_out + i * ldc + j;
        for (auto k = 0; k < K; ++k) {
          size_t wt_idx = is_transpose_weights ? (bs * offset_wei + j * ldb + k) :
                          (bs * offset_wei + k * ldb + j);
          size_t ip_idx = is_transpose_src ? (bs * offset_src + k * lda + i) :
                          (bs * offset_src + i * lda + k);
          sum += read_and_cast<float>(input, input_dtype,
                                      ip_idx) * read_and_cast<float>(weights, weight_dtype, wt_idx);
        }

        if (alpha != 1.0f) {
          sum *= alpha;
        }
        if (beta) {
          sum += read_and_cast<float>(output, output_dtype, op_idx) * beta;
        }
        if (bias) {
          sum += read_and_cast<float>(bias, bias_dtype, j);
        }
        accum_buff_f32[op_idx] = sum;
      }
    }
  }
}

void matmul_ref_kernel_t::compute_quantized_matmul(int batch_size, int M, int N,
    int K, int lda, int ldb, int ldc, unsigned int offset_src,
    unsigned int offset_wei,
    unsigned int offset_out, float alpha, float beta, bool is_transpose_src,
    bool is_transpose_weights, const void *input, const void *weights,
    const void *bias, void *output, float *accum_buff_f32, data_type_t input_dtype,
    data_type_t weight_dtype, data_type_t bias_dtype, data_type_t output_dtype,
    scale_and_zero_point_t quant_param) {

  LOG_DEBUG_INFO("Computing quantized MatMul_ref");

  auto src_scale_size = quant_param.src_scale.size;
  auto wei_scale_size = quant_param.wei_scale.size;

  int32_t *zp_comp = nullptr;
  int zp_comp_size = 0;

  if (quant_param.src_zp.buff || quant_param.wei_zp.buff) {
    int32_t src_zero_point = quant_param.src_zp.buff != nullptr ?
                             read_and_cast<int32_t>(quant_param.src_zp.buff,
                                 quant_param.src_zp.dt) : 0;
    int32_t wei_zero_point = quant_param.wei_zp.buff != nullptr ?
                             read_and_cast<int32_t>(quant_param.wei_zp.buff,
                                 quant_param.wei_zp.dt) : 0;
    int src_0 = is_transpose_src ? 1 : lda;
    int src_1 = is_transpose_src ? lda : 1;
    int wei_0 = is_transpose_weights ? 1 : ldb;
    int wei_1 = is_transpose_weights ? ldb : 1;

    compute_zero_point_compensation(M, N, K, (char *)input, src_0, src_1,
                                    (int8_t *)weights, wei_0, wei_1,
                                    zp_comp, src_zero_point, wei_zero_point, zp_comp_size);
  }

  #pragma omp parallel for collapse(3)
  for (auto bs = 0; bs < batch_size; ++bs) {
    for (auto i = 0; i < M; ++i) {
      for (auto j = 0; j < N; ++j) {
        int32_t sum_s32 = 0;
        size_t op_idx = bs * offset_out + i * ldc + j;
        for (auto k = 0; k < K; ++k) {
          size_t wt_idx = is_transpose_weights ? (bs * offset_wei + j * ldb + k) :
                          (bs * offset_wei + k * ldb + j);
          size_t ip_idx = is_transpose_src ? (bs * offset_src + k * lda + i) :
                          (bs * offset_src + i * lda + k);
          sum_s32 += read_and_cast<int32_t>(input, input_dtype, ip_idx) *
                     read_and_cast<int32_t>(weights, weight_dtype, wt_idx);
        }
        float sum = static_cast<float>(sum_s32);
        if (alpha != 1.0f) {
          sum *= alpha;
        }
        if (beta) {
          sum += read_and_cast<float>(output, output_dtype, op_idx) * beta;
        }
        if (zp_comp) {
          sum += (float)(zp_comp[(i * N + j) % zp_comp_size]);
        }
        if (src_scale_size) {
          sum *= read_and_cast<float>(quant_param.src_scale.buff,
                                      quant_param.src_scale.dt, j % src_scale_size);
        }
        if (wei_scale_size) {
          sum *= read_and_cast<float>(quant_param.wei_scale.buff,
                                      quant_param.wei_scale.dt, j % wei_scale_size);
        }
        if (bias) {
          sum += read_and_cast<float>(bias, bias_dtype, j);
        }
        accum_buff_f32[op_idx] = sum;
      }
    }
  }

  if (zp_comp) {
    free(zp_comp);
  }
}

} //namespace ops
} //namespace zendnnl

extern "C" {
  zendnnl::ops::matmul_ref_kernel_t *get_matmul_ref_kernel() {
    return new zendnnl::ops::matmul_ref_kernel_t();
  }
}
