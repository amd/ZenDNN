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
#ifndef _BATCHMATMUL_EXAMPLE_HPP_
#define _BATCHMATMUL_EXAMPLE_HPP_

#include "zendnnl.hpp"
#include "example_utils.hpp"

#define  OK          (0)
#define  NOT_OK      (1)

#define  BATCH_SIZE 128
#define  BATCH_MATMUL_M 10
#define  BATCH_MATMUL_K 5
#define  BATCH_MATMUL_N 4

namespace zendnnl {
namespace examples {

/** @fn batch_matmul_relu_f32_kernel_example
 *  @brief Demonstrates batch_matmul+relu fused operator on fp32 inputs.
 *
 *  batch_matmul operator implements matrix multiplication on 3D tensors. This operator
 *  can be fused with various post_ops.
 *
 *  This example demonstrates batch_matmul_relu fused operator creation and execution of
 *  one of its fp32 computation based kernel.
 */
int batch_matmul_relu_f32_kernel_example();

/** @fn batch_matmul_wei2d_relu_f32_kernel_example
 *  @brief Demonstrates batch_matmul+relu fused operator on fp32 inputs.
 *
 *  batch_matmul operator implements matrix multiplication on 3D tensors. This operator
 *  can be fused with various post_ops.
 *
 *  This example demonstrates batch_matmul_relu fused operator creation and execution of
 *  one of its fp32 computation based kernel.
 */
int batch_matmul_wei2d_relu_f32_kernel_example();

/** @fn batch_matmul_inp2d_relu_f32_kernel_example
 *  @brief Demonstrates batch_matmul+relu fused operator on fp32 inputs.
 *
 *  batch_matmul operator implements matrix multiplication on 3D tensors. This operator
 *  can be fused with various post_ops.
 *
 *  This example demonstrates batch_matmul_relu fused operator creation and execution of
 *  one of its fp32 computation based kernel.
 */
int batch_matmul_inp2d_relu_f32_kernel_example();

/** @fn batch_matmul_relu_bf16_kernel_example
 *  @brief Demonstrates batch_matmul+relu fused operator on bf16 inputs.
 *
 *  batch_matmul operator implements matrix multiplication on 3D tensors. This operator
 *  can be fused with various post_ops.
 *
 *  This example demonstrates batch_matmul_relu fused operator creation and execution of
 *  one of its bf16 computation based kernel.
 */
int batch_matmul_relu_bf16_kernel_example();

/** @fn batch_matmul_mul_silu_mul_f32_kernel_example
 *  @brief Demonstrates batch_matmul+elt_mul+silu+elt_mul fused operator on f32 inputs.
 *
 *  batch_matmul operator implements matrix multiplication on 3D tensors. This operator
 *  can be fused with various post_ops.
 *
 *  This example demonstrates batch_matmul binary_mul silu binary_mul fused operator
 *  creation and execution of one of its f32 computation based kernel.
 */
int batch_matmul_mul_silu_mul_f32_kernel_example();

/** @fn batch_matmul_silu_mul_bf16_kernel_example
 *  @brief Demonstrates batch_matmul+silu+elt_mul fused operator on bf16 inputs.
 *
 *  batch_matmul operator implements matrix multiplication on 3D tensors. This operator
 *  can be fused with various post_ops.
 *
 *  This example demonstrates batch_matmul silu binary_mul fused operator creation and
 *  execution of one of its bf16 computation based kernel.
 */
int batch_matmul_silu_mul_bf16_kernel_example();

/** @fn batch_matmul_relu_forced_ref_kernel_example
 *  @brief Demonstrates batch_matmul+relu fused operator reference kernel enforced by user.
 *
 *  batch_matmul operator implements matrix multiplication on 3D tensors. This operator
 *  can be fused with various post_ops.
 *
 *  This example demonstrates how an operator kernel can be forced by the user.
 */
int batch_matmul_relu_forced_ref_kernel_example();

/** @fn batchmatmul_broadcast_example
 *  @brief Demonstrates batchmatmul+relu fused operator reference kernel enforced by user.
 *
 *  batchmatmul operator implements matrix multiplication on 3D tensors but weight is
 *  2D tensor and already is broadcasted . This operator can be fused with
 *  various post_ops.
 *
 *  This example demonstrates batchmatmul operator creation for user-broadcast tensor
 *  and execution of broadcast case batchmatmul.
 */
int batchmatmul_broadcast_example();

} //examples
} //zendnnl

#endif
