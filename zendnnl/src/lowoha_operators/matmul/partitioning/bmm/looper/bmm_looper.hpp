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

#ifndef MATMUL_BMM_LOOPER_HPP
#define MATMUL_BMM_LOOPER_HPP

#include "lowoha_operators/matmul/lowoha_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace bmm {

/// BMM entry point (restructured).
///
/// Owns: planning (via plan_bmm), parallelization strategy selection,
///       batch-stride computation, OMP / zendnnl_parallel_for regions,
///       batch × M-block tile loops, specialized dispatch to
///       batched_sgemm and oneDNN backends.
/// Calls: bmm_tile_execute from kernel/ for each work item.
///
/// This is a drop-in replacement for the original bmm_execute() that
/// was inlined in lowoha_matmul.cpp + bmm_partitioner.cpp.
void bmm_execute(const char layout, const bool transA, const bool transB,
                 const int M, const int N, const int K, const float alpha,
                 const void *src, const int lda,
                 const void *weight, const int ldb,
                 const void *bias, const float beta,
                 void *dst, const int ldc,
                 const bool is_weights_const,
                 matmul_batch_params_t &batch_params,
                 const size_t src_type_size, const size_t weight_type_size,
                 const size_t out_type_size,
                 const int num_threads,
                 matmul_algo_t &kernel, matmul_params &params);

} // namespace bmm
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_BMM_LOOPER_HPP
