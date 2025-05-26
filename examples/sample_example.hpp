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
#ifndef _SAMPLE_EXAMPLE_HPP_
#define _SAMPLE_EXAMPLE_HPP_

#include "zendnnl.hpp"
#include "example_utils.hpp"

#define  OK          (0)
#define  NOT_OK      (1)

#define  ROWS 50
#define  COLS 50


namespace zendnnl {
namespace examples {

/** @fn sample_f32_kernel_example
 *  @brief Demonstrates sample operator creation and its f32 kernel execution.
 *
 *  Sample operator is an operator that demonstrates how an operator can be
 *  implemented. Any new operator can be implemented by cloning the sample operator
 *  and implementing operator functionality in it.
 *
 *  This example demonstrates sample operator creation and execution of one of its
 *  kernel based on data type of the input.
 */
int sample_f32_kernel_example();

/** @fn sample_f32_kernel_example
 *  @brief Demonstrates sample operator creation and its f32 kernel execution.
 *
 *  Sample operator is an operator that demonstrates how an operator can be
 *  implemented. Any new operator can be implemented by cloning the sample operator
 *  and implementing operator functionality in it.
 *
 *  This example demonstrates sample operator creation and execution of one of its
 *  kernel based on data type of the input.
 */
int sample_bf16_kernel_example();

} //examples
} //zendnnl

#endif
