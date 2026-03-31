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
#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_utils.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/bf16_gemv_direct.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/int8_gemv_direct.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/int8_brgemm_looper.hpp"
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
// validated against 172 BF16 GEMV shapes (K,N = 16..1024) at 2.7 GHz fixed.
// BKC-GEMV uses block-aware packing (256-col blocks) for any N where B ≤ L2.
//
//   Rules are evaluated in order (first match wins):
//
//   ┌──────────────────────────────────┬──────────┬────────────────────────┐
//   │ Shape characteristic             │ Best     │ Reason                 │
//   ├──────────────────────────────────┼──────────┼────────────────────────┤
//   │ N≤32, K≥384                     │ DLP      │ 50-75% NR-pad waste    │
//   │ N∈(33,63), ≥25%pad, K≥12N,384 │ DLP      │ High zero-pad BW waste │
//   │ N∈(65,127), ≥25%pad, K≥5N,384 │ DLP      │ Double dispatch + waste │
//   │ Packed B ≤ L2                    │ BRGEMM   │ BKC GEMV: block packing│
//   │ Everything else                  │ BRGEMM   │ General BRGEMM looper  │
//   └──────────────────────────────────┴──────────┴────────────────────────┘
// ════════════════════════════════════════════════════════════════════════
static matmul_algo_t bf16_gemv_best_algo_impl(int N, int K, int num_threads) {
  if (num_threads != 1)
    return matmul_algo_t::native_brgemm;

  const int packed_N = ((N + BKC_NR_PAD - 1) / BKC_NR_PAD) * BKC_NR_PAD;
  const int pad_waste = packed_N - N;
  const size_t packed_K = static_cast<size_t>((K + 1) & ~1);
  const size_t b_packed_bytes = packed_K * packed_N * sizeof(uint16_t);

  // ── DLP zone 1: tiny-N with large K ─────────────────────────────
  // N≤32 pads to 64 → 50-75% of every ZMM load is zeros.
  // DLP's row-major streaming avoids this entirely.
  // Threshold K≥384: BKC-GEMV still wins at K=256 despite padding.
  if (N <= 32 && K >= 384)
    return matmul_algo_t::aocl_dlp_blocked;

  // ── DLP zone 2: NR-padding waste with K-dominant shapes ─────────
  // When packed_N has ≥25% zero columns AND K is large relative to N,
  // the BKC kernel wastes bandwidth on zero-padded loads.
  //   N ∈ (32,64): DLP wins for very K-dominant (K ≥ 12*N, K ≥ 384).
  //     Lower K thresholds cause false positives (N=48 K=512 BKC-GEMV wins 12%).
  //   N ∈ (64,128) with tail: double dispatch overhead + padding.
  //     DLP wins only for very K-dominant shapes (K ≥ 5*N, K ≥ 384).
  if (pad_waste * 4 >= packed_N && N > 32) {
    if (N < 64 && K >= 12 * N && K >= 384)
      return matmul_algo_t::aocl_dlp_blocked;
    if (N > 64 && N < 128 && K >= 5 * N && K >= 384)
      return matmul_algo_t::aocl_dlp_blocked;
  }

  // ── DLP zone 3: highly K-dominant shapes with large B ────────────
  // When B > ~600KB and K ≥ 3×N with N in the wide-block range,
  // DLP's row-major streaming outperforms BKC's block dispatch
  // (e.g. K=1024 N=320: DLP 82% vs BKC 71%).
  // K ≥ 2×N is too aggressive — K=896 N=384 BKC still wins by 13%.
  if (b_packed_bytes > 600 * 1024 && K >= 3 * N && N > 256)
    return matmul_algo_t::aocl_dlp_blocked;

  // ── BKC-GEMV zone: packed B fits in L2 ──────────────────────────
  // Block-aware packing eliminates stride gaps for any N.
  // Wide blocks (384 cols, NP=5/6) used for N ∈ (256, 512].
  {
    static const size_t l2 = static_cast<size_t>(detect_uarch().l2_bytes);
    if (b_packed_bytes <= l2)
      return matmul_algo_t::native_brgemm;
  }

  // ── BRGEMM (default for L3-resident shapes) ────────────────────
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
    int nt_exec = num_threads;
    if (M == 1 && is_bf16 && num_threads > 1)
      nt_exec = m1_gemv_cap_threads(N, num_threads);
    GemmDescriptor desc = make_desc(transA, transB, M, N, K, alpha, beta,
                                    lda, ldb, ldc, is_weights_const,
                                    nt_exec, params);
    desc.bias = bias;

    // INT8 paths: u8/s8 source × s8 weights → bf16/fp32.
    // M=1 single-thread: try BKC GEMV fast path first (L2-resident B).
    // All other shapes: general INT8 BRGEMM looper.
    const bool is_int8 = ((params.dtypes.src == data_type_t::u8 ||
                           params.dtypes.src == data_type_t::s8) &&
                          params.dtypes.wei == data_type_t::s8);
    if (is_int8) {
      if (!uarch.avx512vnni) return false;
      // INT8 BKC GEMV is single-thread only today; BF16 M=1 can use bf16_gemv_direct
      // with nt>1. Parallel INT8 M=1 goes straight to BRGEMM.
      if (M == 1 && num_threads == 1) {
        if (int8_gemv_direct(desc, uarch, src, weight, dst, bias, params))
          return true;
      }
      int8_brgemm_execute(desc, uarch, src, weight, dst, bias, params);
      return true;
    }

    if (is_bf16 && uarch.avx512bf16) {
      // M=1: try BKC GEMV first (single-thread: global/thread-local pack;
      // multi-thread: per-thread N-slices, each packs its columns). On false,
      // fall through to BRGEMM looper.
      if (M == 1) {
        if (bf16_gemv_direct(desc, uarch, src, weight, dst, bias, params))
          return true;
      }
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
