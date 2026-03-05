/*******************************************************************************
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lowoha_operators/matmul/matmul_ai/ai_matmul.hpp"
#include "lowoha_operators/matmul/matmul_ai/common/gemm_descriptor.hpp"
#include "lowoha_operators/matmul/matmul_ai/common/cost_model.hpp"
#include "lowoha_operators/matmul/matmul_ai/gemm/intrinsic/fp32/avx512_fp32_gemm.hpp"
#include "lowoha_operators/matmul/matmul_ai/gemm/intrinsic/bf16/avx512_bf16_gemm.hpp"
#include "lowoha_operators/matmul/matmul_ai/brgemm/intrinsic/fp32/avx512_fp32_brgemm.hpp"
#include "lowoha_operators/matmul/matmul_ai/brgemm/intrinsic/bf16/avx512_bf16_brgemm.hpp"
#include "common/data_types.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ai {

using zendnnl::common::size_of;

void ai_matmul_execute(
  matmul_algo_t kernel,
  char layout, bool transA, bool transB,
  int M, int N, int K, float alpha,
  const void *src, int lda,
  const void *weight, int ldb,
  const void *bias, float beta,
  void *dst, int ldc,
  bool is_weights_const, int num_threads,
  matmul_params &params) {

  // Early exit
  if (M <= 0 || N <= 0 || K <= 0) return;

  // ── 1. UArch detection (cached static, zero cost after first call) ──
  const UarchParams &uarch = detect_uarch();

  // ── 2. Build descriptor (stack only, no heap) ──
  GemmDescriptor desc;
  desc.M = M; desc.N = N; desc.K = K;
  desc.lda = lda; desc.ldb = ldb; desc.ldc = ldc;
  desc.transA = transA; desc.transB = transB;
  desc.alpha = alpha; desc.beta = beta;
  desc.src_dt = params.dtypes.src;
  desc.wei_dt = params.dtypes.wei;
  desc.dst_dt = params.dtypes.dst;
  desc.bias_dt = params.dtypes.bias;
  desc.src_elem_size = size_of(desc.src_dt);
  desc.wei_elem_size = size_of(desc.wei_dt);
  desc.dst_elem_size = size_of(desc.dst_dt);
  desc.bias = bias;
  desc.is_weights_const = is_weights_const;
  desc.num_threads = num_threads;

  // ── 3. Dispatch: each module owns its planner + B prepack + thread loop ──
  const bool is_bf16 = (desc.src_dt == data_type_t::bf16 &&
                        desc.wei_dt == data_type_t::bf16);

  if (is_bf16 && kernel == matmul_algo_t::ai_brgemm && uarch.avx512bf16) {
    bf16_brgemm_execute(desc, uarch, src, weight, dst, bias, params);
  } else if (is_bf16) {
    bf16_gemm_execute(desc, uarch, src, weight, dst, bias, params);
  } else if (kernel == matmul_algo_t::ai_brgemm && uarch.avx512f) {
    brgemm_execute(desc, uarch, src, weight, dst, bias, params);
  } else {
    gemm_execute(desc, uarch, src, weight, dst, bias, params);
  }
}

} // namespace ai
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
