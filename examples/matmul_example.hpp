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
#ifndef _MATMUL_EXAMPLE_HPP_
#define _MATMUL_EXAMPLE_HPP_

#include "example_utils.hpp"
#include "memory/tensor.hpp"
#include "operators/matmul/matmul_context.hpp"
#include "operators/matmul/matmul_operator.hpp"

#define  OK          (0)
#define  NOT_OK      (1)

#define  MATMUL_M 10
#define  MATMUL_K 5
#define  MATMUL_N 4

namespace zendnnl {
namespace examples {

/** @fn matmul_relu_f32_kernel_example
 *  @brief Demonstrates matmul+relu fused operator on fp32 inputs.
 *
 *  matmul operator implements matrix multiplication on 2D tensors. This operator
 *  can be fused with various post_ops.
 *
 *  This example demonstrates matmul_relu fused operator creation and execution of
 *  one of its fp32 computation based kernel.
 */
int matmul_relu_f32_kernel_example();

/** @fn matmul_relu_bf16_kernel_example
 *  @brief Demonstrates matmul+relu fused operator on fp32 inputs.
 *
 *  matmul operator implements matrix multiplication on 2D tensors. This operator
 *  can be fused with various post_ops.
 *
 *  This example demonstrates matmul_relu fused operator creation and execution of
 *  one of its bf16 computation based kernel.
 */
int matmul_relu_bf16_kernel_example();

/** @fn matmul_relu_forced_ref_kernel_example
 *  @brief Demonstrates matmul+relu fused operator reference kernel enforced by user.
 *
 *  matmul operator implements matrix multiplication on 2D tensors. This operator
 *  can be fused with various post_ops.
 *
 *  This example demonstrates how an operator kernel can be forced by the user.
 */
int matmul_relu_forced_ref_kernel_example();

} //examples
} //zendnnl

#endif
