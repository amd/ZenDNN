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

#include "lowoha_operators/matmul/matmul_native/brgemm/planner/brgemm_planner.hpp"
#include "common/zendnnl_global.hpp"
#include <algorithm>
#include <cmath>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using namespace zendnnl::error_handling;

static inline int round_up(int x, int m) { return ((x + m - 1) / m) * m; }
static inline int round_down(int x, int m) { return (x / m) * m; }

BrgemmPlan plan_fp32_brgemm(const GemmDescriptor &desc, const UarchParams &uarch) {
  BrgemmPlan plan;
  const int M = desc.M, N = desc.N, K = desc.K;
  const int nt = desc.num_threads;
  const int elem = 4; // sizeof(float)
  plan.num_threads = nt;

  // ── Register tile: MR=6, NR=16 ──
  // Same as GEMM path for compiler-generated intrinsic kernels.
  // BRGEMM's advantage comes from keeping accumulators live across K-blocks,
  // not from wider register tiles (that requires JIT assembly).
  plan.MR = 6;
  plan.NR = 16;
  const int MR = plan.MR, NR = plan.NR;

  // ── BK: K-block size inside microkernel ──
  // A micro-panel (MR × BK) must fit in L1 so that A data stays hot
  // while the microkernel iterates over BK K-steps.
  int l1_budget = static_cast<int>(0.8 * uarch.l1d_bytes);
  int bk_max = l1_budget / (MR * elem);
  bk_max = std::max(bk_max, 64);
  // Distribute K evenly into BK-sized blocks
  if (bk_max >= K) {
    plan.BK = K;
  } else {
    int n_blocks = (K + bk_max - 1) / bk_max;
    plan.BK = round_up((K + n_blocks - 1) / n_blocks, 8);
  }

  // ── NB: N-blocking for L2 (B panel reuse) ──
  // B panel (BK × NB) should fit in 50% L2 for reuse across MC blocks.
  int l2_budget = uarch.l2_bytes / 2;
  int nb_from_l2 = l2_budget / (plan.BK * elem);

  // L3-aware for multi-thread
  int nb_from_l3 = nb_from_l2;
  if (nt > 1) {
    int cores_per_ccd = std::min(uarch.num_cores, 8);
    int threads_sharing_l3 = std::min(nt, cores_per_ccd);
    int l3_budget = static_cast<int>(0.5 * uarch.l3_bytes_per_ccd);
    nb_from_l3 = l3_budget / (threads_sharing_l3 * plan.BK * elem);
  }

  int nb_max = std::min(nb_from_l2, nb_from_l3);
  plan.NB = round_down(std::max(nb_max, NR), NR);
  plan.NB = std::min(plan.NB, N);

  // Even NB distribution
  if (plan.NB < N && plan.NB >= NR) {
    int n_jtiles = (N + plan.NB - 1) / plan.NB;
    int even_nb = round_up((N + n_jtiles - 1) / n_jtiles, NR);
    if (even_nb > 0 && even_nb <= plan.NB + NR)
      plan.NB = std::min(even_nb, N);
  }

  // ── MB: M-blocking for L2 ──
  // C tile (MB × NB) + A tile (MB × BK) fits in L2.
  int total_per_core = uarch.l1d_bytes + uarch.l2_bytes;
  plan.MB = round_down(
    std::max(total_per_core / ((plan.NB + plan.BK) * elem), MR), MR);
  plan.MB = std::min(plan.MB, M);

  // ── Thread load balancing ──
  if (nt > 1) {
    int ic_tiles = (M + plan.MB - 1) / plan.MB;
    int jc_tiles = (N + plan.NB - 1) / plan.NB;
    int target = nt;
    int ideal = 2 * nt;
    if (ic_tiles * jc_tiles >= ideal) target = ideal;

    while (true) {
      ic_tiles = (M + plan.MB - 1) / plan.MB;
      jc_tiles = (N + plan.NB - 1) / plan.NB;
      if (ic_tiles * jc_tiles >= target) break;
      if (plan.NB > NR) { plan.NB -= NR; continue; }
      if (plan.MB > MR && M > MR) { plan.MB -= MR; continue; }
      break;
    }
  }

  static bool s_log_fp32 = apilog_info_enabled();
  if (s_log_fp32) {
    apilog_info("Native FP32 BRGEMM plan: M=", M, " N=", N, " K=", K,
                " MB=", plan.MB, " NB=", plan.NB, " BK=", plan.BK,
                " MR=", plan.MR, " NR=", plan.NR,
                " threads=", plan.num_threads);
  }

  return plan;
}

// ============================================================================
// BF16 BRGEMM planner (NR=64, VNNI dpbf16ps microkernels)
// ============================================================================

BrgemmPlan plan_bf16_brgemm(const GemmDescriptor &desc,
                            const UarchParams &uarch) {
  BrgemmPlan plan;
  const int M = desc.M, N = desc.N, K = desc.K;
  const int K_padded = (K + 1) & ~1;
  const bool is_decode = (M <= 4);
  plan.num_threads = desc.num_threads;
  plan.NR = 64;

  // ── MR: adaptive for decode vs throughput ──
  if (is_decode) {
    plan.MR = M;
  } else if (M == 8) {
    plan.MR = 8;
  } else if (M % 6 == 0 || M >= 18) {
    plan.MR = 6;
  } else if (M % 4 == 0) {
    plan.MR = 4;
  } else if (M % 6 <= 3 && M > 12) {
    plan.MR = 4;
  } else {
    plan.MR = 6;
  }

  // ── BK: maximize to keep accumulators live longer ──
  // A panel (MR×BK×2B) in L1, B panel (NR_PACK×BK×2B) in L2/2.
  constexpr int NR_PACK = 64;
  {
    int kb_a = static_cast<int>(0.8 * uarch.l1d_bytes)
               / std::max(plan.MR * 2, 1);
    int kb_b = (uarch.l2_bytes / 2)
               / (NR_PACK * static_cast<int>(sizeof(uint16_t)));
    int bk_max = std::min(kb_a, kb_b);
    bk_max = std::max(bk_max, 64);
    bk_max = (bk_max + 1) & ~1;
    if (K_padded <= bk_max) {
      plan.BK = K_padded;
    } else {
      int n_blk = (K_padded + bk_max - 1) / bk_max;
      plan.BK = ((K_padded + n_blk - 1) / n_blk + 1) & ~1;
    }
  }

  // ── NB: L2-aware, aligned to NR=64 ──
  // Use sizeof(float) for NB sizing even though B is BF16, because the
  // microkernel accumulates into FP32 C tiles (4 bytes per element).
  // The effective L2 footprint per tile is dominated by C, not B.
  // Empirically: BK*4 produces better tile granularity than BK*2.
  {
    int l2_budget = uarch.l2_bytes / 2;
    int nb_from_l2 = l2_budget / (plan.BK * 4);
    int nb_from_l3 = nb_from_l2;
    if (plan.num_threads > 1) {
      int cores_per_ccd = std::min(uarch.num_cores, 8);
      int threads_sharing_l3 = std::min(plan.num_threads, cores_per_ccd);
      int l3_budget = static_cast<int>(0.5 * uarch.l3_bytes_per_ccd);
      nb_from_l3 = l3_budget / (threads_sharing_l3 * plan.BK * 4);
    }
    int nb_max = std::min(nb_from_l2, nb_from_l3);
    plan.NB = (std::max(nb_max, plan.NR) / plan.NR) * plan.NR;
    plan.NB = std::min(plan.NB, N);
  }

  // ── MB: cache-aware with load balance ──
  if (!is_decode) {
    int mb_budget = (uarch.l1d_bytes + uarch.l2_bytes)
                    / std::max(plan.NB * 4 + plan.BK * 2, 1);
    mb_budget = (mb_budget / plan.MR) * plan.MR;
    mb_budget = std::max(mb_budget, plan.MR);
    plan.MB = std::min(mb_budget, M);

    if (plan.num_threads > 1 && plan.MB < M) {
      int m_panels = (M + plan.MR - 1) / plan.MR;
      int jc_tiles = (N + plan.NB - 1) / plan.NB;
      int ic_tiles = (M + plan.MB - 1) / plan.MB;
      int last_ic = M - (ic_tiles - 1) * plan.MB;
      bool imbalanced = (last_ic < plan.MB / 2)
                        || (ic_tiles * jc_tiles < plan.num_threads);
      if (imbalanced) {
        int target = std::max(2 * plan.num_threads,
                              plan.num_threads + jc_tiles);
        int needed_ic = std::max((target + jc_tiles - 1) / jc_tiles, 2);
        int ppb = std::max(m_panels / needed_ic, 1);
        plan.MB = std::min(ppb * plan.MR, M);
      }
    }
  } else {
    plan.MB = M;
  }

  static bool s_log_bf16 = apilog_info_enabled();
  if (s_log_bf16) {
    int jt = (N + plan.NB - 1) / plan.NB;
    int it = (M + plan.MB - 1) / plan.MB;
    apilog_info("Native BF16 BRGEMM plan: M=", M, " N=", N, " K=", K,
                " MB=", plan.MB, " NB=", plan.NB, " BK=", plan.BK,
                " MR=", plan.MR, " NR=", plan.NR,
                " ic=", it, " jc=", jt, " tiles=", it * jt,
                " threads=", plan.num_threads);
  }

  return plan;
}

// ============================================================================
// INT8 BRGEMM planner (NR=64, INT8 VNNI vpdpbusd microkernels)
// ============================================================================

BrgemmPlan plan_int8_brgemm(const GemmDescriptor &desc,
                            const UarchParams &uarch) {
  BrgemmPlan plan;
  const int M = desc.M, N = desc.N, K = desc.K;
  const int K_padded = (K + 3) & ~3;
  const bool is_decode = (M <= 4);
  plan.num_threads = desc.num_threads;
  plan.NR = 64;

  // ── MR: same adaptive logic as BF16 ──
  if (is_decode) {
    plan.MR = M;
  } else if (M % 6 == 0 || M >= 18) {
    plan.MR = 6;
  } else if (M % 4 == 0) {
    plan.MR = 4;
  } else {
    plan.MR = 6;
  }

  // ── BK: INT8 elements are 1 byte (vs 2 for BF16) ──
  // A panel (MR×BK×1B) in L1, B panel (NR_PACK×BK×1B) in L2/2.
  // INT8 gets 2× larger BK than BF16 for the same cache budget.
  constexpr int NR_PACK = 64;
  {
    int kb_a = static_cast<int>(0.8 * uarch.l1d_bytes)
               / std::max(plan.MR * 1, 1);
    int kb_b = (uarch.l2_bytes / 2) / (NR_PACK * 1);
    int bk_max = std::min(kb_a, kb_b);
    bk_max = std::max(bk_max, 64);
    bk_max = (bk_max + 3) & ~3;  // align to VNNI group of 4
    if (K_padded <= bk_max) {
      plan.BK = K_padded;
    } else {
      int n_blk = (K_padded + bk_max - 1) / bk_max;
      plan.BK = ((K_padded + n_blk - 1) / n_blk + 3) & ~3;
    }
  }

  // ── NB: L2-aware, aligned to NR=64 ──
  // Accumulator is i32 (4 bytes), so effective footprint uses 4 bytes.
  {
    int l2_budget = uarch.l2_bytes / 2;
    int nb_from_l2 = l2_budget / (plan.BK * 4);
    int nb_from_l3 = nb_from_l2;
    if (plan.num_threads > 1) {
      int cores_per_ccd = std::min(uarch.num_cores, 8);
      int threads_sharing_l3 = std::min(plan.num_threads, cores_per_ccd);
      int l3_budget = static_cast<int>(0.5 * uarch.l3_bytes_per_ccd);
      nb_from_l3 = l3_budget / (threads_sharing_l3 * plan.BK * 4);
    }
    int nb_max = std::min(nb_from_l2, nb_from_l3);
    plan.NB = (std::max(nb_max, plan.NR) / plan.NR) * plan.NR;
    plan.NB = std::min(plan.NB, N);
  }

  // ── MB: cache-aware ──
  if (!is_decode) {
    int mb_budget = (uarch.l1d_bytes + uarch.l2_bytes)
                    / std::max(plan.NB * 4 + plan.BK * 1, 1);
    mb_budget = (mb_budget / plan.MR) * plan.MR;
    mb_budget = std::max(mb_budget, plan.MR);
    plan.MB = std::min(mb_budget, M);

    if (plan.num_threads > 1 && plan.MB < M) {
      int m_panels = (M + plan.MR - 1) / plan.MR;
      int jc_tiles = (N + plan.NB - 1) / plan.NB;
      int ic_tiles = (M + plan.MB - 1) / plan.MB;
      int last_ic = M - (ic_tiles - 1) * plan.MB;
      bool imbalanced = (last_ic < plan.MB / 2)
                        || (ic_tiles * jc_tiles < plan.num_threads);
      if (imbalanced) {
        int target = std::max(2 * plan.num_threads,
                              plan.num_threads + jc_tiles);
        int needed_ic = std::max((target + jc_tiles - 1) / jc_tiles, 2);
        int ppb = std::max(m_panels / needed_ic, 1);
        plan.MB = std::min(ppb * plan.MR, M);
      }
    }
  } else {
    plan.MB = M;
  }

  static bool s_log_int8 = apilog_info_enabled();
  if (s_log_int8) {
    int jt = (N + plan.NB - 1) / plan.NB;
    int it = (M + plan.MB - 1) / plan.MB;
    apilog_info("Native INT8 BRGEMM plan: M=", M, " N=", N, " K=", K,
                " MB=", plan.MB, " NB=", plan.NB, " BK=", plan.BK,
                " MR=", plan.MR, " NR=", plan.NR,
                " ic=", it, " jc=", jt, " tiles=", it * jt,
                " threads=", plan.num_threads);
  }

  return plan;
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
