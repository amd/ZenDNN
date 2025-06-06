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
#ifndef _MATMUL_REF_KERNEL_HPP_
#define _MATMUL_REF_KERNEL_HPP_

#include <vector>
#include <iostream>
#include <memory>
#include <cstring>
#include <cstdlib>
#include "common/zendnnl_global.hpp"
#include "operators/common/operator_kernel.hpp"
#include "matmul_context.hpp"

#define SQRT_2_OVER_PI 0.79788458347320556640625f
#define SQRT_2_OVER_2  0.707106769084930419921875f
#define FITTING_CONST  0.044715f

namespace zendnnl {
namespace ops {

using namespace zendnnl::error_handling;

/** @class matmul_ref_kernel_t
 *  @brief A class to implement a reference matmul kernel.
 *
 *  This kernel is a reference implementation of matmul layer.
 *  It demonstrates the matmul layer algorithm and does computation with highest accuracy.
 *  This however is not a performant kernel and generally used for comparision purposes.
 *
 *  The kernel is limited to work with FP32 contiguous tensors and supports only ReLU post-op.
 *
 *  @todo Generalize to strided tensor and support all other post-ops.
 *
 */

class matmul_ref_kernel_t final : public op_kernel_t<matmul_context_t> {
 public:

  /** @brief Overriden from parent class
  * @param context_ The context containing kernel parameters.
  * @param inputs_ The input tensors for the MatMul operation.
  * @param outputs_ The output tensors for the MatMul operation.
  * @return Status of the execution (success or failure).
  */
  status_t execute(const context_type &context_,
                   tensor_map_type &inputs_,
                   tensor_map_type &outputs_) override;
 private:

  /** @brief Apply post-ops to the output tensor.
  * @param tensor_ The output tensor.
  * @param zen_po_ The post-operation to apply.
  * @param output Pointer to the interim output buffer.
  * @return Status of the operation (success or failure).
  */
  status_t apply_post_op(tensor_t &tensor_, post_op_t zen_po_, float *output);

  /** @brief Apply buffer based post-ops
  * @param tensor_ The output tensor.
  * @param buffer_tensor_ The buffer tensor for binary operations.
  * @param zen_po_ The post-operation to apply.
  * @param output Pointer to the interim output buffer.
  * @return Status of the operation (success or failure).
  */
  status_t apply_post_op(tensor_t &tensor_, tensor_t &buffer_tensor_,
                         post_op_t zen_po_, float *output);


  /**
  * @brief Apply an element-wise post-operation using a function pointer.
  * @tparam Args Variadic template for additional arguments to the post-op function.
  * @param post_op_func Pointer to the post-op function.
  * @param tensor_ The output tensor.
  * @param output Pointer to the interim output buffer.
  * @param args Additional arguments for the post-op function.
  * @return Status of the operation (success or failure).
  */
  template<typename... Args>
  status_t apply_eltwise_post_op(float (matmul_ref_kernel_t::*post_op_func)
                                 (float, Args...), tensor_t &tensor_, float *output, Args... args);

  /**
  * @brief Apply the softmax operation to the output tensor.
  * @param tensor_ The output tensor.
  * @param output Pointer to the interim output buffer.
  * @return Status of the operation (success or failure).
  */
  status_t apply_softmax(tensor_t &tensor_, float *output);

  /**
  * @brief Forward implementation of the ELU activation function.
  * @param x Input value.
  * @param alpha ELU alpha parameter.
  * @return Output value after applying ELU.
  */
  float elu_fwd(float x, float alpha);

  /**
  * @brief Forward implementation of the ReLU activation function.
  * @param x Input value.
  * @return Output value after applying ReLU.
  */
  float relu_fwd(float x);

  /**
   * @brief Forward implementation of the Leaky ReLU activation function.
   * @param x Input value.
   * @param nslope Negative slope for Leaky ReLU.
   * @return Output value after applying Leaky ReLU.
   */
  float leaky_relu_fwd(float x, float nslope);

  /**
  * @brief Forward implementation of the GELU (tanh approximation) activation function.
  * @param x Input value.
  * @return Output value after applying GELU (tanh approximation).
  */
  float gelu_tanh_fwd(float x);

  /**
  * @brief Forward implementation of the GELU (erf approximation) activation function.
  * @param x Input value.
  * @return Output value after applying GELU (erf approximation).
  */
  float gelu_erf_fwd(float x);

  /**
  * @brief Forward implementation of the Swish activation function.
  * @param x Input value.
  * @param scale Scaling parameter for Swish.
  * @return Output value after applying Swish.
  */
  float swish_fwd(float x, float scale);

  /**
  * @brief Forward implementation of the Sigmoid activation function.
  * @param x Input value.
  * @return Output value after applying Sigmoid.
  */
  float sigmoid_fwd(float x);

  /**
  * @brief Forward implementation of the Tanh activation function.
  * @param x Input value.
  * @return Output value after applying Tanh.
  */
  float tanh_fwd(float x);

  /**
  * @brief Forward implementation of the Square operation.
  * @param x Input value.
  * @return Output value after squaring the input.
  */
  float square_fwd(float x);

  /**
  * @brief Forward implementation of the Absolute Value operation.
  * @param x Input value.
  * @return Absolute value of the input.
  */
  float abs_fwd(float x);

  /**
  * @brief Forward implementation of the Square Root operation.
  * @param x Input value.
  * @return Square root of the input.
  */
  float sqrt_fwd(float x);

  /**
  * @brief Forward implementation of the Exponential operation.
  * @param x Input value.
  * @return Exponential of the input.
  */
  float exp_fwd(float x);

  /**
  * @brief Forward implementation of the Logarithm operation.
  * @param x Input value.
  * @return Natural logarithm of the input.
  */
  float log_fwd(float x);

  /**
  * @brief Forward implementation of the Clip operation.
  * @param x Input value.
  * @param lower Lower bound for clipping.
  * @param upper Upper bound for clipping.
  * @return Clipped value.
  */
  float clip_fwd(float x, float lower, float upper);

  /**
  * @brief Forward implementation of the Binary Add operation.
  * @param x Input value.
  * @param y Second input value.
  * @param scale Scaling parameter for the second input.
  * @return Result of the binary add operation.
  */
  float binary_add_fwd(float x, float y, float scale);

  /**
  * @brief Forward implementation of the Binary Multiply operation.
  * @param x Input value.
  * @param y Second input value.
  * @param scale Scaling parameter for the second input.
  * @return Result of the binary multiply operation.
  */
  float binary_mul_fwd(float x, float y, float scale);
};

} //namespace ops
} //namespace zendnnl

extern "C" {
  std::shared_ptr<zendnnl::ops::matmul_ref_kernel_t>
  get_matmul_ref_kernel();
}

#endif
