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

#ifndef _LOWOHA_REORDER_EXAMPLE_HPP_
#define _LOWOHA_REORDER_EXAMPLE_HPP_

#include "zendnnl.hpp"
#include <iostream>
#include <vector>

#define OK      (0)
#define NOT_OK  (1)

namespace zendnnl {
namespace examples {

using namespace zendnnl::lowoha::reorder;

/** @fn run_lowoha_reorder_bf16_to_int8_test
 *  @brief Demonstrates BF16 to INT8 quantization using LOWOHA reorder API.
 *
 *  This example demonstrates quantizing BF16 data to INT8 using the
 *  low-overhead LOWOHA reorder API with scale and zero-point parameters.
 *
 *  Formula: int8_val = clamp(round(bf16_val / scale) + zero_point, -128, 127)
 */
int run_lowoha_reorder_bf16_to_int8_test();

/** @fn run_lowoha_reorder_int8_to_bf16_test
 *  @brief Demonstrates INT8 to BF16 dequantization using LOWOHA reorder API.
 *
 *  This example demonstrates dequantizing INT8 data to BF16 using the
 *  low-overhead LOWOHA reorder API with scale and zero-point parameters.
 *
 *  Formula: bf16_val = (int8_val - zero_point) * scale
 */
int run_lowoha_reorder_int8_to_bf16_test();

/** @fn run_lowoha_reorder_bf16_to_uint8_test
 *  @brief Demonstrates BF16 to UINT8 quantization using LOWOHA reorder API.
 *
 *  This example demonstrates quantizing BF16 data to UINT8 using the
 *  low-overhead LOWOHA reorder API with scale and zero-point parameters.
 *
 *  Formula: uint8_val = clamp(round(bf16_val / scale) + zero_point, 0, 255)
 */
int run_lowoha_reorder_bf16_to_uint8_test();

/** @fn run_lowoha_reorder_uint8_to_bf16_test
 *  @brief Demonstrates UINT8 to BF16 dequantization using LOWOHA reorder API.
 *
 *  This example demonstrates dequantizing UINT8 data to BF16 using the
 *  low-overhead LOWOHA reorder API with scale and zero-point parameters.
 *
 *  Formula: bf16_val = (uint8_val - zero_point) * scale
 */
int run_lowoha_reorder_uint8_to_bf16_test();

} // namespace examples
} // namespace zendnnl

#endif // _LOWOHA_REORDER_EXAMPLE_HPP_

