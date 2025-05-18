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
#ifndef _MATMUL_OPERATOR_HPP_
#define _MATMUL_OPERATOR_HPP_

#include "common/zendnnl_global.hpp"
#include "operators/common/operator.hpp"
#include "matmul_context.hpp"

namespace zendnnl {
namespace ops {

/** @class matmul_operator_t
 *  @brief Implements matmul (matrix multiplication) operator.
 *
 * @par Synopsys
 *
 * Given a KxN weight matrix, (optional) Nx1 bias vector, and MxK input, it computes
 * MxN output = weight*input + bias. Matrix multiplication is generally implemented
 * by using a BLAS library for GEMM. This implementation uses AMD AOCL BLIS library
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
 *   1. (optional) bias     : A Nx1 1D tensor.
 * - Inputs
 *   1. (mandatory) matmul_input  : A MxK 2D tensor.
 * - Output(s)
 *   1. (mandatory) matmul_output : A MxN 2D tensor.
 *
 */
class matmul_operator_t final : public operator_t<matmul_operator_t, matmul_context_t> {
public:
  using parent_type = operator_t<matmul_operator_t, matmul_context_t>;

protected:
  status_t validate() override;
  status_t validate_forced_kernel() override;
  status_t kernel_factory() override;
  status_t preprocess();
};
} //namespace ops

namespace interface {
using matmul_operator_t = zendnnl::ops::matmul_operator_t;
}

} //namespace zendnnl
#endif
