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

#ifndef MATMUL_BMM_PLANNER_HPP
#define MATMUL_BMM_PLANNER_HPP

#include <cstddef>
#include "lowoha_operators/matmul/lowoha_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace bmm {

// Input descriptor for BMM planning — captures the problem shape, batch
// geometry, thread budget, and selected kernel algorithm.
//
// Analogous to GemmDescriptor for the GEMM planner: purely declarative,
// no data pointers.
struct BmmConfig {
  int M;
  int N;
  int K;
  int batch_count;
  int num_threads;
  matmul_algo_t kernel;
  size_t src_batch_stride_bytes;
  size_t weight_batch_stride_bytes;
  size_t dst_batch_stride_bytes;

  BmmConfig()
    : M(0), N(0), K(0), batch_count(1), num_threads(1),
      kernel(matmul_algo_t::none),
      src_batch_stride_bytes(0),
      weight_batch_stride_bytes(0),
      dst_batch_stride_bytes(0) {}
};

// Output of the BMM planner — blocking and parallelization strategy.
//
// Analogous to BlockPlan / BrgemmPlan: the looper consumes this to
// set up its parallel region and tile iteration bounds.
struct BmmPlan {
  int M_block;
  bool use_zendnnl_parallel;

  BmmPlan() : M_block(0), use_zendnnl_parallel(false) {}
};

// Calculate the optimal M-dimension block size for parallel partitioning.
//
// Divides threads across batches, then subdivides M within each batch.
// Applies kernel-specific and shape-specific heuristics (e.g. smaller
// M_block for high batch counts with libxsmm to improve load balance).
int calculate_optimal_m_block(const BmmConfig &config);

// Determine whether to use zendnnl_parallel_for (chunked task-based)
// or OpenMP collapse(2) based on the per-batch MFLOPs.
//
// zendnnl_parallel_for has better load balance for large workloads
// (MFLOPs > 6.0); OMP collapse(2) avoids scheduling overhead for
// small tasks.
bool should_use_zendnnl_parallel(int M, int N, int K);

// Build a complete BMM plan from the config.
// Combines M_block calculation and parallelization strategy selection.
BmmPlan plan_bmm(const BmmConfig &config);

} // namespace bmm
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_BMM_PLANNER_HPP
