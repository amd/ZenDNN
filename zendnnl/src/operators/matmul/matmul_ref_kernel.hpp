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
#include <omp.h>
#include "common/zendnnl_global.hpp"
#include "operators/common/operator_kernel.hpp"
#include "matmul_context.hpp"

#define SQRT_2_OVER_PI 0.79788458347320556640625f
#define SQRT_2_OVER_2  0.707106769084930419921875f
#define FITTING_CONST  0.044715f

namespace zendnnl {
namespace ops {

using namespace zendnnl::error_handling;

/**
 * @struct scale_and_zero_point_t
 * @brief A structure to encapsulate scale and zero-point information for quantized operations.
 *
 * This structure is used to store the scale and zero-point parameters for both the source
 * and weight tensors in quantized operations. It contains an inner structure `inner_t` to
 * represent individual scale or zero-point data, and the outer structure aggregates these
 * for the source and weight tensors.
 *
 * @details
 * The `scale_and_zero_point_t` structure is designed to handle the following:
 * - Scale values for the source and weight tensors.
 * - Zero-point values for the source and weight tensors.
 * - Data type and size information for each scale and zero-point.
 *
 * The structure is initialized with default values to ensure safe usage.
 */
struct scale_and_zero_point_t {
  /**
   * @struct quant_t
   * @brief A nested structure to represent individual scale or zero-point data.
   *
   * This inner structure contains a pointer to the data buffer, the data type,
   * and the size of the buffer. It is used to represent scale or zero-point
   * information for a single tensor.
   */
  struct quant_t {
    const void *buff;    /**< Pointer to the buffer holding scale or
                              zero-point data. */
    data_type_t dt;      /**< Data type of the buffer (e.g., float, int32_t). */
    size_t size;         /**< Size of the buffer in bytes. */

    /**
     * @brief Default constructor for `quant_t`.
     *
     * Initializes the buffer pointer to `nullptr`, the data type to `none`,
     * and the size to `0`.
     */
    quant_t() : buff(nullptr), dt(data_type_t::none), size(0) {}
  };

  quant_t src_scale;  /**< Scale information for the source tensor. */
  quant_t wei_scale;  /**< Scale information for the weight tensor. */
  quant_t src_zp;     /**< Zero-point information for the source tensor. */
  quant_t wei_zp;     /**< Zero-point information for the weight tensor. */

  /**
   * @brief Default constructor for `scale_and_zero_point_t`.
   *
   * Initializes all members (`src_scale`, `wei_scale`, `src_zp`, `wei_zp`)
   * using the default constructor of `quant_t`.
   */
  scale_and_zero_point_t() : src_scale(), wei_scale(), src_zp(), wei_zp() {}
};

/** @class matmul_ref_kernel_t
 *  @brief A class to implement a reference matmul kernel.
 *
 *  This kernel is a reference implementation of matmul layer.
 *  It demonstrates the matmul layer algorithm and does computation with highest accuracy.
 *  This however is not a performant kernel and generally used for comparison purposes.
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

  /**
   * @brief Computes the zero-point compensation matrix for quantized MatMul operations.
   *
   * This function calculates the compensation matrix required to adjust for the effects
   * of zero-points in quantized matrix multiplication. It modifies the accumulation buffer (`acc`)
   * to account for the zero-points of the source (`src_zero_point`) and weights (`wei_zero_point`),
   * ensuring accurate results in the presence of quantization offsets.
   *
   * @param M The number of rows in the source matrix.
   * @param N The number of columns in the weights matrix.
   * @param K The shared dimension between the source and weights matrices.
   * @param src Pointer to the source matrix data.
   * @param src_s0 Stride of the source matrix along the first dimension.
   * @param src_s1 Stride of the source matrix along the second dimension.
   * @param wei Pointer to the weights matrix data.
   * @param wei_s0 Stride of the weights matrix along the first dimension.
   * @param wei_s1 Stride of the weights matrix along the second dimension.
   * @param acc Reference to the pointer for the accumulation buffer, where the compensation
   *            matrix will be stored.
   * @param src_zero_point The zero-point value for the source matrix.
   * @param wei_zero_point The zero-point value for the weights matrix.
   * @param zp_comp_size Reference to an integer to store the size of the compensation matrix.
   */
  void compute_zero_point_compensation(int M, int N, int K, char *src, int src_s0,
                                       int src_s1,
                                       int8_t *wei, int wei_s0, int wei_s1,
                                       int32_t *&acc, int32_t src_zero_point,
                                       int32_t wei_zero_point, int &zp_comp_size);
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
   * @brief Apply post-ops to the output tensor using context and input tensors.
   * @param output_tensor The output tensor.
   * @param inputs_ The input tensors for the MatMul operation.
   * @param accum_buff_f32 Pointer to the interim output buffer.
   * @param context_ The context containing kernel parameters.
   * @return Status of the operation (success or failure).
   */
  status_t apply_post_op(tensor_t &output_tensor,
                         tensor_map_type &inputs_,
                         float *accum_buff_f32,
                         const context_type &context_);

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

  /**
   * @brief Quantize the destination tensor by applying scale and zero-point adjustments.
   * @param output_tensor The output tensor to be quantized.
   * @param accum_buff_f32 Pointer to the interim output buffer.
   */
  void quantize_dst(tensor_t &output_tensor, float *accum_buff_f32);

  /**
   * @brief Compute the matrix multiplication operation.
   * @param batch_size Batch size for the operation.
   * @param M Number of rows in the output matrix.
   * @param N Number of columns in the output matrix.
   * @param K Shared dimension between input and weights matrices.
   * @param lda Leading dimension of the input matrix.
   * @param ldb Leading dimension of the weights matrix.
   * @param ldc Leading dimension of the output matrix.
   * @param offset_src Offset for the source matrix.
   * @param offset_wei Offset for the weights matrix.
   * @param offset_out Offset for the output matrix.
   * @param alpha Scaling factor for the input matrix.
   * @param beta Scaling factor for the output matrix.
   * @param is_transpose_src Whether the source matrix is transposed.
   * @param is_transpose_weights Whether the weights matrix is transposed.
   * @param input Pointer to the input tensor.
   * @param weights Pointer to the weights tensor.
   * @param bias Pointer to the bias tensor.
   * @param output Pointer to the output tensor.
   * @param accum_buff_f32 Pointer to the output buffer.
   * @param input_dtype Data type of the input tensor.
   * @param weight_dtype Data type of the weights tensor.
   * @param bias_dtype Data type of the bias tensor.
   * @param output_dtype Data type of the output tensor.
   */
  void compute_matmul(int batch_size, int M, int N, int K, int lda,
                      int ldb, int ldc, unsigned int offset_src, unsigned int offset_wei,
                      unsigned int offset_out, float alpha, float beta, bool is_transpose_src,
                      bool is_transpose_weights, const void *input, const void *weights,
                      const void *bias, const void *output, float *accum_buff_f32,
                      data_type_t input_dtype, data_type_t weight_dtype, data_type_t bias_dtype,
                      data_type_t output_dtype);

  /**
   * @brief Compute the quantized matrix multiplication operation.
   * @param batch_size Batch size for the operation.
   * @param M Number of rows in the output matrix.
   * @param N Number of columns in the output matrix.
   * @param K Shared dimension between input and weights matrices.
   * @param lda Leading dimension of the input matrix.
   * @param ldb Leading dimension of the weights matrix.
   * @param ldc Leading dimension of the output matrix.
   * @param offset_src Offset for the source matrix.
   * @param offset_wei Offset for the weights matrix.
   * @param offset_out Offset for the output matrix.
   * @param alpha Scaling factor for the input matrix.
   * @param beta Scaling factor for the output matrix.
   * @param is_transpose_src Whether the source matrix is transposed.
   * @param is_transpose_weights Whether the weights matrix is transposed.
   * @param input Pointer to the input tensor.
   * @param weights Pointer to the weights tensor.
   * @param bias Pointer to the bias tensor.
   * @param output Pointer to the output tensor.
   * @param accum_buff_f32 Pointer to the interim output buffer.
   * @param input_dtype Data type of the input tensor.
   * @param weight_dtype Data type of the weights tensor.
   * @param bias_dtype Data type of the bias tensor.
   * @param output_dtype Data type of the output tensor.
   * @param quant_param Structure to input and weights quantization parameters.
   */
  void compute_quantized_matmul(int batch_size, int M, int N, int K, int lda,
                                int ldb, int ldc, unsigned int offset_src, unsigned int offset_wei,
                                unsigned int offset_out, float alpha, float beta, bool is_transpose_src,
                                bool is_transpose_weights, const void *input, const void *weights,
                                const void *bias, void *output, float *accum_buff_f32, data_type_t input_dtype,
                                data_type_t weight_dtype, data_type_t bias_dtype, data_type_t output_dtype,
                                scale_and_zero_point_t quant_param);

  /**
   * @brief Store the computed output tensor to the destination buffer.
   * @param nelem Number of elements in the output tensor.
   * @param accum_buff_f32 Pointer to the accumulation buffer containing the computed results.
   * @param output Pointer to the destination buffer where the output tensor will be stored.
   * @param output_dtype Data type of the output tensor.
   */
  void store_output(uint64_t nelem, float *accum_buff_f32, void *output,
                    data_type_t output_dtype);
};

} //namespace ops
} //namespace zendnnl

extern "C" {
  std::shared_ptr<zendnnl::ops::matmul_ref_kernel_t>
  get_matmul_ref_kernel();
}

#endif
