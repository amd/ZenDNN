/********************************************************************************
# * Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef _MATMUL_FP32_REF_KERNEL_HPP_
#define _MATMUL_FP32_REF_KERNEL_HPP_

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

/** @class matmul_f32_ref_kernel_t
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

class matmul_f32_ref_kernel_t final : public op_kernel_t<matmul_context_t> {
 public:

  /** @brief Overriden from parent class
   *  @returns status_t::success
   */
  status_t execute(const context_type &context_,
                   tensor_map_type &inputs_,
                   tensor_map_type &outputs_) override;
 private:

  /** @brief Apply post-ops
   * @returns status_t::success
   */
  status_t apply_post_op(tensor_t &tensor_, post_op_t zen_po_);


  /** @brief Apply element wise post-ops
   * @returns status_t::success
   */
  template<typename... Args>
  status_t apply_eltwise_post_op(float (matmul_f32_ref_kernel_t::*post_op_func)
                                 (float, Args...), tensor_t &tensor_, Args... args);

  /** @brief Implements softmax post-op
   * @returns status_t::success
   */
  status_t apply_softmax(tensor_t &tensor_);

  //Eltwise post-ops
  float elu_fwd(float x, float alpha);
  float relu_fwd(float x);
  float leaky_relu_fwd(float x, float nslope);
  float gelu_tanh_fwd(float x);
  float gelu_erf_fwd(float x);
  float swish_fwd(float x, float scale);
  float sigmoid_fwd(float x);
  float tanh_fwd(float x);
  float softmax_fwd(float x);
  float pooling_fwd(float x);
  float square_fwd(float x);
  float abs_fwd(float x);
  float sqrt_fwd(float x);
  float exp_fwd(float x);
  float log_fwd(float x);
  float clip_fwd(float x, float lower, float upper);
};

} //namespace ops
} //namespace zendnnl

extern "C" {
  std::shared_ptr<zendnnl::ops::matmul_f32_ref_kernel_t>
  get_matmul_f32_ref_kernel();
}

#endif
