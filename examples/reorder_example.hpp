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
#ifndef _REORDER_EXAMPLE_HPP_
#define _REORDER_EXAMPLE_HPP_

#include "example_utils.hpp"
#include "memory/tensor.hpp"
#include "operators/reorder/reorder_context.hpp"
#include "operators/reorder/reorder_operator.hpp"
#include "operators/matmul/matmul_context.hpp"
#include "operators/matmul/matmul_operator.hpp"

#define  OK          (0)
#define  NOT_OK      (1)

#define  ROWS 50
#define  COLS 50
#define  MATMUL_M 10
#define  MATMUL_K 5
#define  MATMUL_N 4


namespace zendnnl {
namespace examples {

/** @fn reorder_f32_kernel_example
 *  @brief Demonstrates reorder operator on f32 input.
 *
 *  Reorder operator converts memory to blocked format based on different backend.
 *
 *  This example demonstrates reorder operator creation and execution of one
 *  of its f32 kernel.
 */
int reorder_f32_kernel_example();

/** @fn reorder_s8_kernel_example
 *  @brief Demonstrates reorder operator on s8 input.
 *
 *  Reorder operator converts memory to blocked format based on different backend.
 *
 *  This example demonstrates reorder operator creation and execution of one
 *  of its s8 kernel.
 */
int reorder_s8_kernel_example();

/** @fn reorder_matmul_relu_f32_kernel_example
 *  @brief Demonstrates matmul+relu operator with reordered weights on f32 weights.
 *
 *  matmul operator implements matrix multiplication on 2D tensors with reordered
 *  weights, OutofPlace reorder takes place.
 *  This operator can be fused with various post_ops.
 *
 *  This example demonstrates matmul and reorder operator creation and
 *  execution of one of its f32 kernel.
 */
int reorder_matmul_relu_f32_kernel_example();

/** @fn reorder_bf16_kernel_example
 *  @brief Demonstrates reorder operator on bf16 input.
 *
 *  Reorder operator converts memory to blocked format based on different
 *  backend. Reorder takesplace as Inplace
 *
 *  This example demonstrates reorder operator creation and execution of one
 *  of its bf16 kernel.
 */
int reorder_inplace_bf16_example();

/** @fn reorder_inplace_matmul_relu_bf16_kernel_example
*  @brief Demonstrates matmul+relu operator with reordered weights on bf16 weights.
*
*  matmul operator implements matrix multiplication on 2D tensors with reordered
*  weights, InPlace reorder takes place. This operator can be fused with various post_ops.
*
*  This example demonstrates matmul and reorder(Inplace) operator creation and
*  execution of one of its bf16 kernel.
*/
int reorder_inplace_matmul_relu_bf16_kernel_example();

} //examples
} //zendnnl

#endif
