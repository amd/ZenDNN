/********************************************************************************
# * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef _MATMUL_OPERATOR_IMPL_HPP_
#define _MATMUL_OPERATOR_IMPL_HPP_

#include "common/zendnnl_global.hpp"
#include "operators/common/operator_impl.hpp"
#include "matmul_config.hpp"
#include "matmul_context.hpp"

namespace zendnnl {
namespace ops {

/** @class matmul_impl_t
 *  @brief Implements matmul (matrix multiplication) operator.
 *
 * @par Synopsys
 *
 * Given a KxN weight matrix, (optional) 1xN bias vector, and MxK input, it computes
 * MxN output = weight*input + bias. Matrix multiplication is generally implemented
 * by using a BLAS library for GEMM. This implementation uses AMD AOCL DLP library
 * for GEMM.
 *
 * In order to elable chaining, the first parameter in @c operator_t template
 * should be the class itself, and the second parameter should be this operators
 * context, derived from @c operator_context_t.
 *
 * @par Quantization support
 *
 * The operator supports both fp32 and bf16 computations.
 *
 * @par Post_op support
 *
 * Supports all eltwise post_ops.
 *
 * @par Parameters, Inputs, Outputs
 *
 * The operator has following parameters and input/outputs
 * - Parameter(s)
 *   1. (mandatory) weights : A KxN 2D tensor.
 *   1. (optional) bias     : A 1xN 1D tensor.
 * - Inputs
 *   1. (mandatory) matmul_input  : A MxK 2D tensor.
 * - Output(s)
 *   1. (mandatory) matmul_output : A MxN 2D tensor.
 *
 */
class matmul_impl_t final : public operator_impl_t<matmul_context_t> {
 public:
  /** @brief Self type **/
  using self_type = matmul_impl_t;
  /** @brief Parent type **/
  using parent_type = operator_impl_t<matmul_context_t>;
  /** @brief context type **/
  using context_type = parent_type::context_type;
  /** @brief kernel type **/
  using   kernel_type =  parent_type::kernel_type;
  /** @brief Shared pointer to kernels */
  using   kernel_sptr_type =  parent_type::kernel_sptr_type;
  /** @brief A map type from strings to tensors */
  using   tensor_map_type = parent_type::tensor_map_type;
  /** @brief Kernel handle type */
  using   create_kernel_handle_type  = parent_type::create_kernel_handle_type;

 protected:
  /** @brief Validate input/output
   *
   * Validates if all mandatory inputs and outputs are given.
   * @return @c status_t::success if successful.
   */
  status_t validate() override;

  /** @brief Validate forced kernel
   *
   * Validates if forced kernel is valid.
   * @return @c status_t::success if successful.
   */
  status_t validate_forced_kernel() override;

  /** @brief Select kernel based on input data type.
   * @return @c status_t::success if successful.
   */
  status_t kernel_factory() override;

  /** @brief Print operator create information
   * @return @c std::string
   */
  std::string op_create_info() override;

  /** @brief Print operator execute information
   * @return @c std::string
   */
  std::string op_execute_info() override;

  /** @brief Preprocess operator
   * @return @c status_t::success if successful.
   */
  status_t preprocess();

  /** @brief Update matmul kernel
   * @return @c status_t::success if successful.
   */
  status_t update_matmul_kernel();

  /** @brief Validate buffer post-op
   * @return @c status_t::success if successful.
   */
  status_t validate_buffer_post_op(std::vector<uint64_t> &output_size,
                                   std::vector<post_op_t> &po,
                                   std::map<std::string,tensor_t> &inputs);
 private:
  bool is_bmm = false;
};

} //namespace ops
} //namespace zendnnl
#endif

