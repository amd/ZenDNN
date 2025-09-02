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

#ifndef _BENCHDNN_HPP_
#define _BENCHDNN_HPP_

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>

#define MEASURE_INDIVIDUAL_TIMINGS 0

#include "utils/benchdnn_utils.hpp"

#define  OK          (0)
#define  NOT_OK      (1)

using namespace zendnnl::interface;
using namespace zendnnl;

namespace zendnnl {
namespace benchdnn {

/**
 * @struct TimingStats
 * @brief Holds timing statistics for various stages of the benchmark.
 *
 * This structure records the time (in milliseconds) taken for different stages
 * of the benchmarking process, including context creation, operator creation,
 * operator execution, other operations, and total time.
 *
 * @var context_creation_ms Time taken for context creation (if enabled).
 * @var operator_creation_ms Time taken for operator creation (if enabled).
 * @var operator_execution_ms Time taken for operator execution (if enabled).
 * @var other_ms Time taken for other operations (e.g., buffer allocation, tensor setup).
 * @var total_time_ms Total time taken for the operation.
 */
struct TimingStats {
#if MEASURE_INDIVIDUAL_TIMINGS
  double context_creation_ms = 0.0; /**< Time taken for context creation */
  double operator_creation_ms = 0.0; /**< Time taken for operator creation */
  double operator_execution_ms = 0.0; /**< Time taken for operator execution */
  double other_ms = 0.0; /**< Time taken for other operations
                          (e.g., buffer allocation, tensor setup) */
#endif
  double total_time_ms = 0.0; /**< Total time taken for the operation */
};

/**
 * @struct global_options
 * @brief Holds global configuration options for benchmarking.
 *
 * This structure contains options that affect the overall benchmarking behavior,
 * such as the number of dimensions (ndims) for tensor creation and operator setup.
 *
 * @var ndims Number of dimensions for tensors (e.g., 2 for standard matmul, 3 for batched matmul).
 */
struct global_options {
  int ndims;
};

} // namespace benchdnn
} // namespace zendnnl

// Include the main matmul and reorder benchmarking interfaces
#include "matmul/matmul_benchdnn.hpp"
#include "reorder/reorder_benchdnn.hpp"

#endif // _BENCHDNN_HPP_