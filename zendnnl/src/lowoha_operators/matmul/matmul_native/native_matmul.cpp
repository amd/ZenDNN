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
// Best-algo selector for BF16 M≤8 (GEMV/decode) shapes.
// Called by ALGO 0 (dynamic_dispatch) to route M≤8 shapes to
// native_brgemm or aocl_dlp_blocked.
//
// M=5-8: always native_brgemm (BRGEMM looper has different cache
// access patterns; the L3/CCD heuristic isn't validated for these).
//
// M=1-4 multi-thread: native_brgemm unless BKC packed data overflows
// CCD-local L3 while raw B fits in aggregate L3. In that case DLP's
// unpacked streaming stays L3-resident. Exception: K > 5120 shapes
// with low NR-pad waste (≤25%) benefit from BKC K-contiguous format
// even with L3 overflow.
// Validated against 72 LLM shapes × {M=1,2,4,8} × {32,64,128}t.
// Catches 14 losing shapes (up to -195%) with 1 minor regression (+5%).
//
// M=1 single-thread: shape-based DLP gates for small-N high-K where
// BKC dispatch overhead exceeds DLP's thin-dispatcher advantage.
// Decision tree (AMD EPYC 9B45, Zen 5, L1d=48KB, L2=1MB, L3=32MB/CCD),
// validated against 188 BF16 GEMV shapes + 22 MoE shapes.
//
//   Rules are evaluated in order (first match wins):
//
//   ┌──────────────────────────────────┬──────────┬────────────────────────┐
//   │ Shape characteristic             │ Best     │ Reason                 │
//   ├──────────────────────────────────┼──────────┼────────────────────────┤
//   │ M≤4 MT: per-CCD BKC ≥ L3/CCD,  │ DLP      │ BKC overflows CCD L3; │
//   │   raw B < agg L3,               │          │ DLP stays L3-resident  │
//   │   K≤5120 or NR-waste>25%        │          │                        │
//   │ N≤32, K≥256                     │ DLP      │ 50-75% NR-pad waste    │
//   │ N=33-48, K≥384                 │ DLP      │ BKC dispatch overhead  │
//   │ N∈(49,63), ≥25%pad, K≥8N,384  │ DLP      │ High zero-pad BW waste │
//   │ N∈(65,127), ≥25%pad, K≥5N,384 │ DLP      │ Double dispatch + waste │
//   │ Packed B ≤ L2                    │ BRGEMM   │ BKC+flat GEMV kernel   │
//   │ Everything else                  │ BRGEMM   │ General BRGEMM looper  │
//   └──────────────────────────────────┴──────────┴────────────────────────┘
// ════════════════════════════════════════════════════════════════════════
static matmul_algo_t bf16_gemv_best_algo_impl(int M, int N, int K, int num_threads) {
  // M=5-8: always native BRGEMM. The L3/CCD overflow heuristic below is
  // validated for M=1-4 paths (BKC GEMV + decode flat); M=5-8 uses the
  // general BRGEMM looper which has different cache access patterns.
  if (M > 4)
    return matmul_algo_t::native_brgemm;

  // ── M=1-4 multi-thread: L3/CCD overflow gate ─────────────────────
  // BKC packing pads each thread's N-slice to a multiple of BKC_NR_PAD (64).
  // When the per-CCD packed footprint exceeds L3/CCD but raw B fits in
  // aggregate L3, DLP streams unpacked B from L3 while BKC spills to DRAM.
  // High-K shapes (K > 5120) with low NR-pad waste (≤25%) still benefit
  // from BKC's K-contiguous streaming pattern even with L3 overflow.
  if (num_threads > 1) {
    static const UarchParams &uarch = detect_uarch();
    const int nt = m1_gemv_cap_threads(N, num_threads);
    const int K_padded = (K + 1) & ~1;
    const int chunk = (N + nt - 1) / nt;
    const int n_pad = ((chunk + BKC_NR_PAD - 1) / BKC_NR_PAD) * BKC_NR_PAD;
    const int ccx_w = std::max(1, uarch.ccx_cores);
    const size_t per_ccd =
        static_cast<size_t>(K_padded) * n_pad * sizeof(uint16_t) * ccx_w;
    const size_t l3_ccd = static_cast<size_t>(uarch.l3_bytes_per_ccd);

    if (per_ccd >= l3_ccd) {
      const size_t raw_B = static_cast<size_t>(K) * N * sizeof(uint16_t);
      const int num_ccds = std::max(1, nt / ccx_w);
      const size_t agg_l3 = l3_ccd * num_ccds;
      const int nr_waste_pct = n_pad > 0 ? (n_pad - chunk) * 100 / n_pad : 0;

      if (raw_B < agg_l3 && (K <= 5120 || nr_waste_pct > 25))
        return matmul_algo_t::aocl_dlp_blocked;
    }
    return matmul_algo_t::native_brgemm;
  }

  // ── M=1 single-thread heuristics below ────────────────────────────
  if (num_threads != 1)
    return matmul_algo_t::native_brgemm;

  const int packed_N = ((N + BKC_NR_PAD - 1) / BKC_NR_PAD) * BKC_NR_PAD;
  const int pad_waste = packed_N - N;
  const size_t packed_K = static_cast<size_t>((K + 1) & ~1);
  const size_t b_packed_bytes = packed_K * packed_N * sizeof(uint16_t);

  // ── DLP zone 1: small-N with large K ────────────────────────────
  // N≤48: BKC dispatch overhead (pack + block iteration + tail-only
  // kernel) dominates for K-dominant shapes. DLP's thin dispatcher
  // and row-major streaming avoid this entirely.
  //   N≤32, K≥256: 50-75% NR-pad waste on every ZMM load.
  //   N=33-48, K≥384: no pad waste but BKC dispatch cost is high
  //     relative to the small per-row compute.
  if (N <= 32 && K >= 256)
    return matmul_algo_t::aocl_dlp_blocked;
  if (N > 32 && N <= 48 && K >= 384)
    return matmul_algo_t::aocl_dlp_blocked;

  // ── DLP zone 2: NR-padding waste with K-dominant shapes ─────────
  // When packed_N has ≥25% zero columns AND K is large relative to N,
  // the BKC kernel wastes bandwidth on zero-padded loads.
  //   N ∈ (48,64) with padding: DLP wins for K-dominant (K ≥ 8*N).
  //   N ∈ (64,128) with tail: double dispatch overhead + padding.
  //     DLP wins only for very K-dominant shapes (K ≥ 5*N, K ≥ 384).
  if (pad_waste * 4 >= packed_N && N > 48) {
    if (N < 64 && K >= 8 * N && K >= 384)
      return matmul_algo_t::aocl_dlp_blocked;
    if (N > 64 && N < 128 && K >= 5 * N && K >= 384)
      return matmul_algo_t::aocl_dlp_blocked;
  }

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
matmul_algo_t bf16_gemv_best_algo(int M, int N, int K, int num_threads) {
  matmul_algo_t result = bf16_gemv_best_algo_impl(M, N, K, num_threads);
  static bool s_log = apilog_info_enabled();
  if (s_log) {
    const char *name = (result == matmul_algo_t::native_brgemm) ? "BRGEMM" :
                       (result == matmul_algo_t::aocl_dlp_blocked) ? "DLP" : "???";
    apilog_info("ALGO0 decode dispatch: M=", M, " K=", K, " N=", N,
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

      // Native INT8 supports:
      //   src: per-tensor scale + per-tensor zero point only
      //   wei: per-tensor (wei_scale_count==1) or per-channel (==N)
      // Per-group, per-token, and other granularities (e.g. src_scale with
      // dims > 1, or wei_scale_count not in {0,1,N}) fall back to DLP.
      {
        auto qp = extract_int8_quant(params);
        if (qp.wei_scale_count != 0 && qp.wei_scale_count != 1
            && qp.wei_scale_count != N)
          return false;
        const auto &src_dims = params.quant_params.src_scale.dims;
        if (!src_dims.empty() && src_dims.back() > 1)
          return false;
        const auto &zp_dims = params.quant_params.src_zp.dims;
        if (!zp_dims.empty() && zp_dims.back() > 1)
          return false;
      }

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
