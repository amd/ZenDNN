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

#include "matmul_fp32_ref_kernel.hpp"

namespace zendnnl {
namespace ops {
using namespace zendnnl::error_handling;

status_t matmul_f32_ref_kernel_t::execute(const context_type &context_,
    tensor_map_type &inputs_,
    tensor_map_type &outputs_) {
  LOG_DEBUG_INFO("Executing matmul_fp32_ref kernel");
  log_info("Executing matmul_fp32_ref kernel");

  auto  input_tensor    = inputs_.find("matmul_input")->second;
  auto  output_tensor   = outputs_.find("matmul_output")->second;
  auto  weight_tensor   = context_.get_param("weights").value();

  float *input          = (float *)input_tensor.get_raw_handle_unsafe();
  float *output         = (float *)output_tensor.get_raw_handle_unsafe();
  float *weights        = (float *)weight_tensor.get_raw_handle_unsafe();

  const int M           = input_tensor.get_size(0);
  const int K           = input_tensor.get_size(1);
  const int N           = output_tensor.get_size(1);

  bool is_trans_src     = input_tensor.get_order() == "ba";
  bool is_trans_weights = weight_tensor.get_order() == "ba";

  const int   lda       = is_trans_src ? input_tensor.get_stride_size(
                            0) : input_tensor.get_stride_size(1);
  const int   ldb       = is_trans_weights ? weight_tensor.get_stride_size(
                            0) : weight_tensor.get_stride_size(1);
  const int   ldc       = output_tensor.get_stride_size(1);

  auto optional_bias_tensor = context_.get_param("bias");
  if (optional_bias_tensor) {
    auto bias_tensor   = context_.get_param("bias").value();
    float   *bias      = (float *)bias_tensor.get_raw_handle_unsafe();
    for (auto i = 0; i < M; ++i) {
      for (auto j = 0; j < N; ++j) {
        float sum = 0.0f;
        for (auto k = 0; k < K; ++k) {
          size_t wt_idx = is_trans_weights ? (j*ldb + k) : (k*ldb + j);
          size_t ip_idx = is_trans_src ? (k*lda + i) : (i*lda + k);
          sum += input[ip_idx] * weights[wt_idx];
        }
        output[i * ldc + j] = sum + bias[j];
      }
    }
  }
  else {
    for (auto i = 0; i < M; ++i) {
      for (auto j = 0; j < N; ++j) {
        float sum = 0.0f;
        for (auto k = 0; k < K; ++k) {
          size_t wt_idx = is_trans_weights ? (j*ldb + k) : (k*ldb + j);
          size_t ip_idx = is_trans_src ? (k*lda + i) : (i*lda + k);
          sum += input[ip_idx] * weights[wt_idx];
        }
        output[i * ldc + j] = sum;
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
        apply_post_op(output_tensor, binary_add_tensor->second, zen_po);
        add_idx++;
      }
      else if (zen_po.type == post_op_type_t::binary_mul) {
        std::string mul_key = "binary_mul_tensor_" + std::to_string(mul_idx);
        auto binary_mul_tensor = inputs_.find(mul_key);
        if (binary_mul_tensor == inputs_.end()) {
          return status_t::failure;
        }
        apply_post_op(output_tensor, binary_mul_tensor->second, zen_po);
        mul_idx++;
      }
      else {
        if (apply_post_op(output_tensor, zen_po) != status_t::success) {
          return status_t::failure;
        }
      }
    }
  }
  return status_t::success;
}

status_t matmul_f32_ref_kernel_t::apply_post_op(tensor_t &tensor_,
    tensor_t &buffer_tensor_, post_op_t zen_po_) {
  float *output      = (float *)tensor_.get_raw_handle_unsafe();
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

status_t matmul_f32_ref_kernel_t::apply_post_op(tensor_t &tensor_,
    post_op_t zen_po_) {
  switch (zen_po_.type) {
  case post_op_type_t::elu: {
    float alpha = zen_po_.elu_params.alpha;
    apply_eltwise_post_op(&matmul_f32_ref_kernel_t::elu_fwd, tensor_, alpha);
    break;
  }
  case post_op_type_t::relu:
    apply_eltwise_post_op(&matmul_f32_ref_kernel_t::relu_fwd, tensor_);
    break;
  case post_op_type_t::leaky_relu: {
    float nslope = zen_po_.leaky_relu_params.nslope;
    apply_eltwise_post_op(&matmul_f32_ref_kernel_t::leaky_relu_fwd, tensor_,
                          nslope);
    break;
  }
  case post_op_type_t::gelu_tanh:
    apply_eltwise_post_op(&matmul_f32_ref_kernel_t::gelu_tanh_fwd, tensor_);
    break;
  case post_op_type_t::gelu_erf:
    apply_eltwise_post_op(&matmul_f32_ref_kernel_t::gelu_erf_fwd, tensor_);
    break;
  case post_op_type_t::swish: {
    float scale = zen_po_.swish_params.scale;
    apply_eltwise_post_op(&matmul_f32_ref_kernel_t::swish_fwd, tensor_, scale);
    break;
  }
  case post_op_type_t::sigmoid: {
    apply_eltwise_post_op(&matmul_f32_ref_kernel_t::sigmoid_fwd, tensor_);
    break;
  }
  case post_op_type_t::tanh:
    apply_eltwise_post_op(&matmul_f32_ref_kernel_t::tanh_fwd, tensor_);
    break;
  case post_op_type_t::softmax:
    apply_softmax(tensor_);
    break;
  case post_op_type_t::square:
    apply_eltwise_post_op(&matmul_f32_ref_kernel_t::square_fwd, tensor_);
    break;
  case post_op_type_t::abs:
    apply_eltwise_post_op(&matmul_f32_ref_kernel_t::abs_fwd, tensor_);
    break;
  case post_op_type_t::sqrt:
    apply_eltwise_post_op(&matmul_f32_ref_kernel_t::sqrt_fwd, tensor_);
    break;
  case post_op_type_t::exp:
    apply_eltwise_post_op(&matmul_f32_ref_kernel_t::exp_fwd, tensor_);
    break;
  case post_op_type_t::log:
    apply_eltwise_post_op(&matmul_f32_ref_kernel_t::log_fwd, tensor_);
    break;
  case post_op_type_t::clip: {
    float lower = zen_po_.clip_params.lower;
    float upper = zen_po_.clip_params.upper;
    apply_eltwise_post_op(&matmul_f32_ref_kernel_t::clip_fwd, tensor_, lower,
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
status_t matmul_f32_ref_kernel_t::apply_eltwise_post_op(float (
      matmul_f32_ref_kernel_t::*post_op_func)(float, Args...),
    tensor_t &tensor_,
    Args... args) {
  float *output_t = (float *)tensor_.get_raw_handle_unsafe();
  uint64_t  size         = tensor_.get_nelem();

  for (uint64_t i = 0; i < size; ++i) {
    output_t[i] = (this->*post_op_func)(output_t[i], std::forward<Args>(args)...);
  }
  return status_t::success;
}

status_t matmul_f32_ref_kernel_t::apply_softmax(tensor_t &tensor_) {
  float   *output_t = (float *)tensor_.get_raw_handle_unsafe();
  const uint64_t rows    = tensor_.get_size(0);
  const uint64_t cols    = tensor_.get_size(1);

  for (uint64_t row = 0; row < rows; ++row) {
    // Compute exponentials for the current row
    double sumExp = 0.0;
    std::vector<double> expRow(cols);

    for (uint64_t col = 0; col < cols; ++col) {
      uint64_t index = row * cols + col;
      expRow[col] = expf((output_t[index]));
      sumExp += expRow[col];
    }

    // Normalize each exponential by the sum to get softmax probabilities
    for (uint64_t col = 0; col < cols; ++col) {
      size_t index = row * cols + col;
      output_t[index] = expRow[col] / sumExp;
    }
  }
  return status_t::success;
}

float matmul_f32_ref_kernel_t::elu_fwd(float x, float alpha) {
  return x > 0 ? x : alpha * (expf(x) - 1.0);
}

float matmul_f32_ref_kernel_t::relu_fwd(float x) {
  return x > 0 ? x : 0;
}

float matmul_f32_ref_kernel_t::leaky_relu_fwd(float x, float nslope) {
  return x > 0 ? x : nslope * x;
}

float matmul_f32_ref_kernel_t::gelu_tanh_fwd(float x) {
  float v = tanh_fwd(SQRT_2_OVER_PI * x * (1 + FITTING_CONST * x * x));
  return (0.5 * x * (1.0f + v));
}

float matmul_f32_ref_kernel_t::gelu_erf_fwd(float x) {
  float v = x * SQRT_2_OVER_2;
  return (0.5f * x * (1.0f + erff(v)));
}

float matmul_f32_ref_kernel_t::sigmoid_fwd(float x) {
  return (1.0 / (1.0 + expf(-x)));
}

float matmul_f32_ref_kernel_t::swish_fwd(float x, float scale) {
  float scaled_x = x * scale;
  return x * (1.0 / (1.0 + expf(-scaled_x)));
}

float matmul_f32_ref_kernel_t::tanh_fwd(float x) {
  const float e = tanhf(x);
  return e;
}

float matmul_f32_ref_kernel_t::square_fwd(float x) {
  return x * x;
}

float matmul_f32_ref_kernel_t::abs_fwd(float x) {
  return x > 0 ? x : -x;
}

float matmul_f32_ref_kernel_t::sqrt_fwd(float x) {
  return sqrtf(x);
}

float matmul_f32_ref_kernel_t::exp_fwd(float x) {
  return expf(x);
}

float matmul_f32_ref_kernel_t::log_fwd(float x) {
  return logf(x);
}

float matmul_f32_ref_kernel_t::clip_fwd(float x, float lower, float upper) {
  x = x > lower ? x : lower;
  return x > upper ? upper : x ;
}

float matmul_f32_ref_kernel_t::binary_add_fwd(float x, float y, float scale) {
  return x + (y * scale) ;
}

float matmul_f32_ref_kernel_t::binary_mul_fwd(float x, float y, float scale) {
  return x * (y * scale) ;
}

} //namespace ops
} //namespace zendnnl

extern "C" {
  std::shared_ptr<zendnnl::ops::matmul_f32_ref_kernel_t>
  get_matmul_f32_ref_kernel() {
    return std::make_shared<zendnnl::ops::matmul_f32_ref_kernel_t>();
  }
}
