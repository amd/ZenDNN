/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#ifndef MATMUL_NATIVE_NATIVE_MATMUL_HPP
#define MATMUL_NATIVE_NATIVE_MATMUL_HPP

#include "lowoha_operators/matmul/lowoha_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

/// Public entry point for the Native matmul kernel (GEMM and BRGEMM).
///
/// Called from matmul_execute() when kernel == native_gemm or native_brgemm.
/// Returns true if the operation was handled, false if the caller should
/// fall back to an alternative kernel (e.g., DLP blocked).
/// Shape-aware heuristics may return false for shapes where the Native kernel
/// is known to be slower than the fallback.
bool native_matmul_execute(
  matmul_algo_t kernel,
  char layout, bool transA, bool transB,
  int M, int N, int K, float alpha,
  const void *src, int lda,
  const void *weight, int ldb,
  const void *bias, float beta,
  void *dst, int ldc,
  bool is_weights_const, int num_threads,
  matmul_params &params);

/// Best-algo selector for BF16 M≤8 (GEMV/decode) shapes.
/// Returns the optimal algo (native_brgemm or aocl_dlp_blocked)
/// based on empirical benchmarks on AMD Zen 5.
/// For M>1, always returns native_brgemm (BRGEMM dominates DLP for M=2-8).
/// For M=1 single-thread, applies shape-based DLP gates for small-N high-K.
matmul_algo_t bf16_gemv_best_algo(int M, int N, int K, int num_threads);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_NATIVE_MATMUL_HPP
