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
#ifndef _EMBEDDING_BAG_EXAMPLE_HPP_
#define _EMBEDDING_BAG_EXAMPLE_HPP_

#include "zendnnl.hpp"
#include "example_utils.hpp"

#define  OK          (0)
#define  NOT_OK      (1)

#define  EMB_ROW 100
#define  EMB_DIM 16
#define  EMB_BATCH_SIZE 5
#define  INDICES_SIZE 10

namespace zendnnl {
namespace examples {

using namespace zendnnl::lowoha::embag;
using zendnnl::interface::testlog_info;
using zendnnl::interface::testlog_error;

/** @fn embedding_bag_f32_kernel_example
 *  @brief Demonstrates embedding bag operator on fp32 inputs.
 *
 * embedding bag operator performs efficient lookups and aggregations
 * such as sum or mean) over a set of embedding vectors selected by indices.
 *
 *  This example demonstrates embedding bag operator creation and execution of
 *  one of its fp32 computation based kernel.
 */
int embedding_bag_f32_kernel_example();

/** @fn embedding_bag_f32_forced_ref_kernel_example
 *  @brief Demonstrates embedding bag operator reference kernel enforced by user.
 *
 * embedding bag operator performs efficient lookups and aggregations
 * such as sum or mean) over a set of embedding vectors selected by indices.
 *
 *  This example demonstrates how an operator kernel can be forced by the user.
 */
int embedding_bag_f32_forced_ref_kernel_example();

/** @fn embedding_f32_kernel_example
 *  @brief Demonstrates embedding operator on fp32 inputs.
 *
 * This example demonstrates creating an embedding lookup operator for f32 data type
 * using the same underlying embedding bag operator infrastructure.
 * Unlike embedding bag operations, this performs direct index-to-embedding lookups
 * without offsets or reduction operations (no sum/mean/max aggregation).
 *
 *  This example demonstrates embedding operator creation and execution of
 *  one of its fp32 computation based kernel.
 */
int embedding_f32_kernel_example();

/** @fn embedding_bag_u4_ref_kernel_example
 *  @brief Demonstrates embedding bag operator reference kernel enforced by user.
 *
 * embedding bag operator performs efficient lookups and aggregations
 * such as sum or mean) over a set of embedding vectors selected by indices.
 *
 *  This example demonstrates how an operator kernel can be forced by the user.
 */
int embedding_bag_u4_ref_kernel_example();

/** @fn embedding_bag_u4_kernel_example
 *  @brief Demonstrates embedding bag operator on u4 inputs.
 *
 * embedding bag operator performs efficient lookups and aggregations
 * such as sum or mean) over a set of embedding vectors selected by indices.
 *
 *  This example demonstrates embedding bag operator creation and execution of
 *  one of its fp32 computation based kernel.
 */
int embedding_bag_u4_kernel_example();

/** @fn group_embedding_bag_direct_example
 *  @brief Demonstrates group embedding bag direct API for batched operations.
 *
 * Group embedding bag performs multiple embedding bag operations in a single call.
 * This is useful when you have multiple embedding tables and want to batch the
 * operations together for efficiency.
 *
 * This example demonstrates the direct LOWOHA API for group embedding bag
 * operations using fp32 data types with sum reduction.
 */
int group_embedding_bag_direct_example();

} // namespace examples
} // namespace zendnnl

#endif
