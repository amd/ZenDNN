/*******************************************************************************
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lowoha_operators/matmul/matmul_ai/brgemm/planner/brgemm_planner.hpp"
#include "common/zendnnl_global.hpp"
#include <algorithm>
#include <cmath>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ai {

using namespace zendnnl::error_handling;

static inline int round_up(int x, int m) { return ((x + m - 1) / m) * m; }
static inline int round_down(int x, int m) { return (x / m) * m; }

BrgemmPlan plan_brgemm(const GemmDescriptor &desc, const UarchParams &uarch) {
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

  apilog_info("AI BRGEMM plan: M=", M, " N=", N, " K=", K,
              " MB=", plan.MB, " NB=", plan.NB, " BK=", plan.BK,
              " MR=", plan.MR, " NR=", plan.NR,
              " threads=", plan.num_threads);

  return plan;
}

} // namespace ai
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
