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

#ifndef MATMUL_BMM_KERNEL_HPP
#define MATMUL_BMM_KERNEL_HPP

#include <cstddef>
#include <cstdint>
#include <functional>
#include "lowoha_operators/matmul/lowoha_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace bmm {

/// Callback signature for the parallel tile loops in the looper.
///
/// The looper computes per-batch source/weight/dst pointers and passes them
/// together with the (batch, m_start, m_len) coordinates.  The kernel module
/// provides the default implementation via bmm_tile_execute().
using bmm_tile_callback_t = std::function<void(
                              int batch_idx,
                              int m_start,
                              int m_len,
                              const uint8_t *src_ptr,
                              const uint8_t *weight_ptr,
                              uint8_t *dst_ptr)>;

/// Invariant context shared across all tiles within a single BMM invocation.
///
/// Filled once by the looper and passed by const-reference to every
/// bmm_tile_execute() call.  Analogous to the captured closure state of
/// the original process_tile lambda, but explicit so the kernel module
/// has no implicit dependency on the looper's stack frame.
struct BmmKernelContext {
  char layout;
  char trans_input;
  char trans_weight;
  bool transA;
  int N;
  int K;
  float alpha;
  float beta;
  int lda;
  int ldb;
  int ldc;
  size_t src_type_size;
  size_t out_type_size;
  matmul_algo_t kernel;
  const void *bias;
  bool is_weights_const;
};

/// Execute a single BMM tile: compute A/C sub-matrix pointers, apply
/// per-batch and per-row post-op offsets, then dispatch to the selected
/// backend (libxsmm, oneDNN, or AOCL DLP).
///
/// This is the BMM equivalent of a GEMM microkernel call: it processes
/// one (batch_idx, m_start, m_len) work item with no threading or
/// tiling decisions.
///
/// @param batch_idx  Batch index (used for 3-D post-op offset)
/// @param m_start    Starting row within the batch
/// @param m_len      Number of rows in this tile
/// @param src_ptr    Batch-offset source pointer
/// @param weight_ptr Batch-offset weight pointer
/// @param dst_ptr    Batch-offset destination pointer
/// @param ctx        Invariant context for the full BMM invocation
/// @param params     Matmul params (post-ops will be copied and offset)
/// @param batch_params Batch parameters (Batch_A, Batch_B)
void bmm_tile_execute(
  int batch_idx, int m_start, int m_len,
  const uint8_t *src_ptr, const uint8_t *weight_ptr, uint8_t *dst_ptr,
  const BmmKernelContext &ctx,
  matmul_params &params,
  matmul_batch_params_t &batch_params);

} // namespace bmm
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_BMM_KERNEL_HPP
