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
#ifndef _COMPARE_EXAMPLE_HPP_
#define _COMPARE_EXAMPLE_HPP_

#include "zendnnl.hpp"
#include "example_utils.hpp"

#define  OK          (0)
#define  NOT_OK      (1)

#define  MATMUL_M 10
#define  MATMUL_K 5
#define  MATMUL_N 4

namespace zendnnl {
namespace examples {

/** @fn compare_operator_execute
 *  @brief Demonstrates compare operator execution with two inputs.
 *
 *  Compare operator implements element-wise comparision on 2 tensors.
 *
 *  This example demonstrates compare operator creation and execution of
 *  its reference kernel.
 */
int compare_operator_execute(tensor_t& a, tensor_t& b);

/** @fn compare_op_example
 *  @brief Demonstrates compare operator on two input tensors.
 */
int compare_op_example();

/** @fn compare_ref_and_aocl_matmul_kernel_example
 *  @brief Demonstrates compare operator on two matmul output.
 *
 *
 *  This example demonstrates use of compare operator to compare output
 *  from ref matmul kernel and aocl matmul kernel.
 */
int compare_ref_and_aocl_matmul_kernel_example();

} //examples
} //zendnnl

#endif
