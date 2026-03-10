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

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::common::size_of;

#if 0  // Per-kernel heuristics disabled — routing moved to bf16_gemv_best_algo
static bool bf16_gemv_prefer_native(int M, int N, int K, int num_threads) {
  if (M != 1 || num_threads != 1) return true;

  const size_t b_bytes = static_cast<size_t>(K) * N * sizeof(uint16_t);

  // Conservative approach: only use ALGO 11 where it demonstrably wins.
  // Default to DLP Blocked (ALGO 1) for everything else.
  //
  // ALGO 11 wins when:
  //  - N > K (wide shapes) AND B small enough for the fast path or
  //    medium-large for L3 streaming with K <= 128
  //  - K > N with N >= 64 and B in the L3 range with wide N (4MB+, N>=512)
  //
  // ALGO 1 wins when:
  //  - K >= N at the L2 sweet-spot (B = 64KB-1MB)
  //  - K >> N with N < 64 (DLP's row-major streaming)
  //  - Square shapes except very small (overhead) or very large (streaming)
  //  - Very large shapes (B > 8MB, any ratio)

  if (N > K) {
    // Wide shapes: BRGEMM excels with no-prefetch kernel for L2-resident B.
    if (b_bytes <= 128 * 1024) return true;   // L1: GEMV fast path
    if (K < 256 && b_bytes <= 4 * 1024 * 1024) return true; // Small K: BRGEMM streams well
    return false;  // L2 sweet-spot or very large: DLP wins
  }

  if (N == K) {
    // Square: only small B (overhead-dominated)
    return (b_bytes <= 16 * 1024);
  }

  // K > N
  if (N < 64) return (K <= 128);  // Small K+N: direct GEMV. Large K: DLP wins.
  if (b_bytes >= 4 * 1024 * 1024 && N >= 512 &&
      K >= 2048) return true;  // L3 streaming with wide N, large K
  return false;
}

static bool bf16_gemv_prefer_gemm(int M, int N, int K, int num_threads) {
  if (M != 1 || num_threads != 1) return true;

  const size_t b_bytes = static_cast<size_t>(K) * N * sizeof(uint16_t);

  // Big-N / small-K: NR-panel overhead dominates (many panels, few iters each).
  if (K <= 32 && N >= 512) return false;
  if (K <= 64 && N >= 1024) return false;
  if (K <= 128 && N >= 2048) return false;

  // K >> N with small N: DLP's row-major streaming of few short rows wins.
  // Check BEFORE small-B guard since these shapes lose even when B is small.
  if (N <= 64 && K >= 512) return false;

  // Small B (< 32KB): GEMM wins for remaining shapes (overhead-dominated).
  if (b_bytes <= 32 * 1024) return true;

  // L2 sweet-spot with K >> N and larger B.
  if (N <= 128 && K >= 512 && b_bytes >= 64 * 1024) return false;

  // Square L2-resident shapes (B = 512KB-1MB): DLP's row-major streaming
  // hits peak L2 BW (~74%) while GEMM's panel-based access gets ~65%.
  // Only defer when B >= 512KB (each NR=64 panel exceeds L1d=48KB).
  if (K >= 512 && N >= 512 && b_bytes >= 512 * 1024
      && b_bytes <= 1024 * 1024) return false;

  return true;
}
#endif  // Per-kernel heuristics disabled

// ════════════════════════════════════════════════════════════════════════
// Unified best-of-3 selector for BF16 GEMV (M=1, single-thread).
// Called by ALGO 0 (dynamic_dispatch) to route each shape to the
// empirically fastest kernel.
//
// Decision tree (AMD EPYC 9B45, Zen 5, L1d=48KB, L2=1MB, L3=32MB/CCD),
// validated against 88 Citadel BF16 GEMV shapes:
//
//   ┌──────────────────────────────────┬──────────┬────────────────────────┐
//   │ Shape characteristic             │ Best     │ Reason                 │
//   ├──────────────────────────────────┼──────────┼────────────────────────┤
//   │ K<64, N>K                        │ BRGEMM   │ GEMV fast path         │
//   │ K<256, N>K, B>256KB              │ BRGEMM   │ Small-K L2/L3 stream   │
//   │ K>=4096, N=256-512, B>=1MB       │ BRGEMM   │ L3 batch-reduce        │
//   │ K>=2*N, K>=2048, N>=512, 2-8MB   │ BRGEMM   │ L3 batch-reduce        │
//   │ N<=32, K>=256                    │ DLP      │ Row-major streaming    │
//   │ N<=64, K>=512                    │ DLP      │ Row-major streaming    │
//   │ N<=128, K=512-1024               │ DLP      │ L2 streaming           │
//   │ K>=4*N, N>=64, B>=128KB          │ DLP      │ Strongly K-dominant    │
//   │ B≈512KB (448-768KB), K,N>=256    │ DLP      │ L2 sweet-spot          │
//   │ Near-square, B=768KB-1.2MB      │ DLP      │ Upper L2 streaming     │
//   │ Everything else                  │ GEMM     │ NR-panel, low overhead │
//   └──────────────────────────────────┴──────────┴────────────────────────┘
// ════════════════════════════════════════════════════════════════════════
matmul_algo_t bf16_gemv_best_algo(int N, int K, int num_threads) {
  if (num_threads != 1)
    return matmul_algo_t::native_brgemm;

  const size_t b_bytes = static_cast<size_t>(K) * N * sizeof(uint16_t);

  // ── BRGEMM zone: wide shapes (N > K) with small K ────────────────
  // BRGEMM's outer-product GEMV fast path excels here: few K iterations
  // per batch, wide N coverage per call, minimal overhead.
  if (K < 64 && N > K)
    return matmul_algo_t::native_brgemm;
  if (K >= 64 && K < 256 && N > K && b_bytes > 256 * 1024)
    return matmul_algo_t::native_brgemm;

  // ── BRGEMM zone: L3 batch-reduce for very large K with moderate N ─
  // K>=4096 with N=256-512: BRGEMM's batch-reduce over K-panels gives
  // better L3 utilization than DLP's row-major streaming.
  // N=128 excluded: DLP is more consistently fast there (K:N=32).
  if (K >= 4096 && N >= 256 && N <= 512 && b_bytes >= 1024 * 1024)
    return matmul_algo_t::native_brgemm;
  // K>=2048 with N>=512, K>>N: similar L3 batch-reduce advantage.
  if (K >= 2048 && K >= 2 * N && N >= 512
      && b_bytes >= 2 * 1024 * 1024 && b_bytes < 8 * 1024 * 1024)
    return matmul_algo_t::native_brgemm;

  // ── DLP zone: K-dominant with small N ─────────────────────────────
  // DLP's row-major sequential access across K elements is ideal
  // when N is small (few output elements per row).
  if (N <= 32 && K >= 256)
    return matmul_algo_t::aocl_dlp_blocked;
  if (N <= 64 && K >= 512)
    return matmul_algo_t::aocl_dlp_blocked;
  // N=65-128 with moderate K: DLP wins for K=512-1024, but for
  // K>=2048 GEMM/BRGEMM take over (handled by rules above).
  if (N > 64 && N <= 128 && K >= 512 && K <= 1024)
    return matmul_algo_t::aocl_dlp_blocked;

  // ── DLP zone: strongly K-dominant (K >= 4*N) ─────────────────────
  // When K >> N and B is at least L2-resident, DLP's streaming pattern
  // hits peak L2 BW. Covers shapes like 1024x256, 2048x256, 4096x1024.
  if (K >= 4 * N && N >= 64 && b_bytes >= 128 * 1024)
    return matmul_algo_t::aocl_dlp_blocked;

  // ── DLP zone: L2 sweet-spot (B ≈ 512KB) ──────────────────────────
  // At B=448-768KB, DLP's streaming hits peak L2 fill BW (~74%).
  // Both K,N must be >= 256 to ensure non-trivial shapes.
  if (b_bytes >= 448 * 1024 && b_bytes < 768 * 1024
      && K >= 256 && N >= 256)
    return matmul_algo_t::aocl_dlp_blocked;

  // ── DLP zone: near-square upper L2 (B = 768KB-1.2MB) ─────────────
  // Near-square shapes (K ≈ N) in this range: DLP's row-major streaming
  // hits higher L2 BW than GEMM's panel access. Non-square shapes in
  // this range stay with GEMM (panel reuse is better when K != N).
  {
    const int mn = K < N ? K : N;
    const int mx = K > N ? K : N;
    if (b_bytes >= 768 * 1024 && b_bytes <= 1200 * 1024
        && mn >= 256 && (mx - mn) <= mn / 4)
      return matmul_algo_t::aocl_dlp_blocked;
  }

  // ── GEMM zone (default) ───────────────────────────────────────────
  // GEMM's NR-panel approach with adaptive NR and flat M=1 fast path
  // handles all remaining shapes: small B (overhead-dominated),
  // medium B (L2 reuse), and large B (L3 general path).
  return matmul_algo_t::native_gemm;
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
#if 0  // Per-kernel DLP fallback disabled — routing moved to ALGO 0 (DT)
    if (is_bf16 && M == 1 && !bf16_gemv_prefer_native(M, N, K, num_threads))
      return false;
#endif

    const UarchParams &uarch = detect_uarch();
    GemmDescriptor desc = make_desc(transA, transB, M, N, K, alpha, beta,
                                    lda, ldb, ldc, is_weights_const,
                                    num_threads, params);
    desc.bias = bias;

    if (is_bf16 && uarch.avx512bf16) {
      // BF16 GEMV fast path (M=1, single-thread, BRGEMM microkernel)
      if (M == 1 && num_threads == 1) {
        const size_t b_bytes = static_cast<size_t>(K) * N * sizeof(uint16_t);
        const bool small_n_ok = (N < 64 && N > 0 && K <= 128 && !transB);
        const bool fast_path_ok = (N >= 64 && N > K
                                   && b_bytes <= static_cast<size_t>(uarch.l1d_bytes)
                                   && is_weights_const);
        if (small_n_ok || fast_path_ok) {
          if (bf16_gemv_direct(desc, uarch, src, weight, dst, bias, params))
            return true;
        }
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

#if 0  // Per-kernel DLP fallback disabled — routing moved to ALGO 0 (DT)
  if (is_bf16 && M == 1 && !bf16_gemv_prefer_gemm(M, N, K, num_threads))
    return false;
#endif

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
