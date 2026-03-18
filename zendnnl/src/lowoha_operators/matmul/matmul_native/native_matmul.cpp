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

#include "lowoha_operators/matmul/matmul_native/native_matmul.hpp"
#include "lowoha_operators/matmul/matmul_native/common/gemm_descriptor.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/bf16_gemv_direct.hpp"
#include "lowoha_operators/matmul/matmul_native/gemm/looper/bf16_gemm_looper.hpp"
#include "lowoha_operators/matmul/matmul_native/gemm/looper/fp32_gemm_looper.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/bf16_brgemm_looper.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/fp32_brgemm_looper.hpp"
#include "common/data_types.hpp"
#include "common/zendnnl_global.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::common::size_of;
using namespace zendnnl::error_handling;

// ════════════════════════════════════════════════════════════════════════
// Unified best-of-3 selector for BF16 GEMV (M=1, single-thread).
// Called by ALGO 0 (dynamic_dispatch) to route each shape to the
// empirically fastest kernel.
//
// Decision tree (AMD EPYC 9B45, Zen 5, L1d=48KB, L2=1MB, L3=32MB/CCD),
// validated against 172 BF16 GEMV shapes (K,N = 16..1024), 91.9% accuracy:
//
//   Rules are evaluated in order (first match wins):
//
//   ┌──────────────────────────────────┬──────────┬────────────────────────┐
//   │ Shape characteristic             │ Best     │ Reason                 │
//   ├──────────────────────────────────┼──────────┼────────────────────────┤
//   │ K≥256, N≤32                     │ DLP      │ Row-major streaming    │
//   │ NR-pad waste≥25%, K high, N≤256 │ DLP      │ KC zero-pad bandwidth  │
//   │ Non-64-aligned near-sq, B≥150KB │ DLP      │ Panel tail waste       │
//   │ K>N, K≥384, N>256, B≥500KB     │ DLP      │ K-dominant streaming   │
//   │ K<N, K≥256, N>256, B=400K-1.2M │ DLP      │ Wide-N L2 boundary     │
//   │ N≤256, packed B≤L2              │ BRGEMM   │ KC GEMV path (1.4-2.5x)│
//   │ K≤64, N>256                     │ BRGEMM   │ BRGEMM looper M=1 fast │
//   │ K≥2*N, N≥64                     │ BRGEMM   │ K-dominant BRGEMM      │
//   │ K≤128, N>256                    │ GEMM     │ Tiny-K wide-N, low OH  │
//   │ Near-square, B=256KB-1.2MB      │ GEMM     │ Panel loop, L2 reuse   │
//   │ Everything else                  │ BRGEMM   │ General BRGEMM looper  │
//   └──────────────────────────────────┴──────────┴────────────────────────┘
// ════════════════════════════════════════════════════════════════════════
static matmul_algo_t bf16_gemv_best_algo_impl(int N, int K, int num_threads) {
  if (num_threads != 1)
    return matmul_algo_t::native_brgemm;

  const size_t b_bytes = static_cast<size_t>(K) * N * sizeof(uint16_t);
  const int packed_N = ((N + 63) / 64) * 64;
  const int pad_waste = packed_N - N;
  const size_t packed_K = static_cast<size_t>((K + 1) & ~1);
  const size_t b_packed_bytes = packed_K * packed_N * sizeof(uint16_t);

  // ── DLP zone 1: high-K tiny-N ─────────────────────────────────
  if (N <= 32 && K >= 256)
    return matmul_algo_t::aocl_dlp_blocked;

  // ── DLP zone 2: high NR-padding waste with large K ────────────
  // When packed_N has ≥25% zero columns (e.g. N=48→64, N=96→128),
  // the KC kernel wastes that fraction of all loads. DLP reads B in
  // row-major without padding.  Threshold depends on panel count:
  //   1 panel  (N≤64):  K≥384 needed (low per-panel overhead)
  //   2 panels (N>64):  K≥2*N sufficient (more overhead to amortize)
  if (pad_waste * 4 >= packed_N && N > 32 && N <= 256) {
    if (N <= 64 && K >= 384)
      return matmul_algo_t::aocl_dlp_blocked;
    if (N > 64 && K >= 2 * N)
      return matmul_algo_t::aocl_dlp_blocked;
  }

  // ── DLP zone 3: non-64-aligned near-square, large B ───────────
  // When min(K,N) is not a multiple of 64, BRGEMM panels have
  // a partially-filled tail strip. DLP's flat streaming avoids
  // this overhead for near-square shapes with B ≥ 150KB.
  {
    const int mn = K < N ? K : N;
    const int mx = K > N ? K : N;
    if (mn % 64 != 0 && b_bytes >= 150 * 1024
        && mn >= 128 && (mx - mn) <= mn)
      return matmul_algo_t::aocl_dlp_blocked;
  }

  // ── DLP zone 4: K>N wide, large B ────────────────────────────
  if (K > N && K >= 384 && N > 256
      && b_bytes >= 500 * 1024)
    return matmul_algo_t::aocl_dlp_blocked;

  // ── DLP zone 5: K<N wide, B at L2 boundary ───────────────────
  if (K < N && K >= 256 && N > 256
      && b_bytes >= 400 * 1024 && b_bytes <= 1200 * 1024)
    return matmul_algo_t::aocl_dlp_blocked;

  // ── BRGEMM zone: N≤256 (KC path) ─────────────────────────────
  {
    static const size_t l2 = static_cast<size_t>(detect_uarch().l2_bytes);
    if (N <= 256 && b_packed_bytes <= l2)
      return matmul_algo_t::native_brgemm;
  }

  // ── BRGEMM zone: K≤64, wide N ────────────────────────────────
  if (K <= 64 && N > 256)
    return matmul_algo_t::native_brgemm;

  // ── BRGEMM zone: K-dominant (K ≥ 2*N) ────────────────────────
  if (K >= 2 * N && N >= 64)
    return matmul_algo_t::native_brgemm;

  // ── GEMM zone: tiny-K wide-N ─────────────────────────────────
  if (K <= 128 && N > 256)
    return matmul_algo_t::native_gemm;

  // ── GEMM zone: near-square, B=256KB-1.2MB ────────────────────
  {
    const int mn = K < N ? K : N;
    const int mx = K > N ? K : N;
    if (b_bytes >= 256 * 1024 && b_bytes <= 1200 * 1024
        && mn >= 256 && (mx - mn) <= mn / 2)
      return matmul_algo_t::native_gemm;
  }

  // ── BRGEMM (default) ─────────────────────────────────────────
  return matmul_algo_t::native_brgemm;
}

// Wrapper that logs the dispatch decision.
matmul_algo_t bf16_gemv_best_algo(int N, int K, int num_threads) {
  matmul_algo_t result = bf16_gemv_best_algo_impl(N, K, num_threads);
  static bool s_log = apilog_info_enabled();
  if (s_log) {
    const char *name = (result == matmul_algo_t::native_brgemm) ? "BRGEMM" :
                       (result == matmul_algo_t::native_gemm) ? "GEMM" :
                       (result == matmul_algo_t::aocl_dlp_blocked) ? "DLP" : "???";
    apilog_info("ALGO0 GEMV dispatch: M=1 K=", K, " N=", N,
                " threads=", num_threads,
                " B=", static_cast<size_t>(K)*N*2/1024, "KB"
                " → ", name);
  }
  return result;
}

// Common descriptor setup — shared by both ALGO 10 and ALGO 11.
static GemmDescriptor make_desc(
    bool transA, bool transB, int M, int N, int K,
    float alpha, float beta, int lda, int ldb, int ldc,
    bool is_weights_const, int num_threads,
    const matmul_params &params) {
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
  desc.bias = nullptr;
  desc.is_weights_const = is_weights_const;
  desc.num_threads = num_threads;
  return desc;
}

bool native_matmul_execute(
  matmul_algo_t kernel,
  char layout, bool transA, bool transB,
  int M, int N, int K, float alpha,
  const void *src, int lda,
  const void *weight, int ldb,
  const void *bias, float beta,
  void *dst, int ldc,
  bool is_weights_const, int num_threads,
  matmul_params &params) {

  (void)layout;
  if (M <= 0 || N <= 0 || K <= 0) return true;

  const bool is_bf16 = (params.dtypes.src == data_type_t::bf16 &&
                        params.dtypes.wei == data_type_t::bf16);

  // ════════════════════════════════════════════════════════════════════
  // ALGO 11 (Native BRGEMM): BRGEMM-based paths with GEMM fallback.
  //   BF16 + AVX512BF16 → bf16_brgemm_execute
  //   FP32 + AVX512F → brgemm_execute (FP32 BRGEMM)
  //   No AVX512 → gemm_execute (scalar/AVX2 fallback)
  // ════════════════════════════════════════════════════════════════════
  if (kernel == matmul_algo_t::native_brgemm) {
    const UarchParams &uarch = detect_uarch();
    GemmDescriptor desc = make_desc(transA, transB, M, N, K, alpha, beta,
                                    lda, ldb, ldc, is_weights_const,
                                    num_threads, params);
    desc.bias = bias;

    if (is_bf16 && uarch.avx512bf16) {
      // BF16 GEMV fast path: K-contiguous kernel for M=1 single-thread.
      // Eligible when N≤256 and B fits in L2. bf16_gemv_direct() has its
      // own gate and returns false for ineligible shapes.
      if (M == 1 && num_threads == 1) {
        if (bf16_gemv_direct(desc, uarch, src, weight, dst, bias, params))
          return true;
      }
      // General BF16 BRGEMM (Planner → Looper → Kernel)
      bf16_brgemm_execute(desc, uarch, src, weight, dst, bias, params);
    } else if (uarch.avx512f) {
      // FP32 BRGEMM
      brgemm_execute(desc, uarch, src, weight, dst, bias, params);
    } else {
      gemm_execute(desc, uarch, src, weight, dst, bias, params);
    }
    return true;
  }

  // ════════════════════════════════════════════════════════════════════
  // ALGO 10 (Native GEMM): GEMM-based paths only.
  //   All shapes → bf16_gemm_execute (Planner → Looper → GEMM kernel)
  //   No BRGEMM microkernel, no GEMV special path.
  //   Uses inner-product style, adaptive NR, simpler looper.
  // ════════════════════════════════════════════════════════════════════

  const UarchParams &uarch = detect_uarch();
  GemmDescriptor desc = make_desc(transA, transB, M, N, K, alpha, beta,
                                  lda, ldb, ldc, is_weights_const,
                                  num_threads, params);
  desc.bias = bias;

  if (is_bf16 && uarch.avx512bf16) {
    bf16_gemm_execute(desc, uarch, src, weight, dst, bias, params);
  } else {
    // FP32 GEMM (avx512f or scalar fallback — gemm_execute handles both)
    gemm_execute(desc, uarch, src, weight, dst, bias, params);
  }
  return true;
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
