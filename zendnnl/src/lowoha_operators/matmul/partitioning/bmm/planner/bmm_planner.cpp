/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#include "lowoha_operators/matmul/partitioning/bmm/planner/bmm_planner.hpp"
#include <algorithm>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace bmm {

static constexpr float kMFlopsThreshold = 6.0f;

int calculate_optimal_m_block(const BmmConfig &config) {
  /*
   * Parallel partitioning strategy:
   * The total number of available threads is divided across batches to compute
   * threads_per_batch, ensuring that each batch gets a fair share of compute
   * resources. Within each batch, the M dimension (rows of the output matrix)
   * is further partitioned into blocks of size M_block, calculated to evenly
   * distribute the workload among the threads assigned to that batch.
   */
  int threads_per_batch = std::max(1, config.num_threads / config.batch_count);
  int M_block = std::max(1,
                         (config.M + threads_per_batch - 1) / threads_per_batch);

  // TODO: Further refine the tuning based on heuristics
  // involving batch_count, M, and num_threads.
  if ((config.batch_count >= 1024 && config.M <= 2048) ||
      (config.batch_count >= 512 && config.M <= 256) ||
      (config.batch_count > 128 && config.batch_count < 192 && config.M <= 512)) {
    if (config.kernel == matmul_algo_t::libxsmm) {
      M_block = std::min(128, config.M);
    }
    else {
      M_block = std::min(36, config.M);
    }
  }
  else if ((config.batch_count == 64 && config.M >= 512) ||
           (config.batch_count == 128 && config.M >= 512)) {
    M_block = std::min(192, config.M);
  }
  else {
    M_block = std::min(M_block, config.M);
  }

  return M_block;
}

bool should_use_zendnnl_parallel(int M, int N, int K) {
  double flops = (2.0 * M * K * N) / 1000000.0;
  return flops > kMFlopsThreshold;
}

BmmPlan plan_bmm(const BmmConfig &config) {
  BmmPlan plan;
  plan.M_block = calculate_optimal_m_block(config);
  plan.use_zendnnl_parallel = should_use_zendnnl_parallel(
                                config.M, config.N, config.K);
  return plan;
}

} // namespace bmm
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
