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

#include "lowoha_operators/matmul/matmul_ai/gemm/planner/gemm_planner.hpp"
#include <algorithm>
#include <cmath>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ai {

static inline int round_up(int x, int m) { return ((x + m - 1) / m) * m; }
static inline int round_down(int x, int m) { return (x / m) * m; }

static int choose_even_kb(int K, int kb_max) {
  if (kb_max >= K) return K;
  int n_blocks = (K + kb_max - 1) / kb_max;
  int even_kb = (K + n_blocks - 1) / n_blocks;
  return round_up(even_kb, 8);
}

BlockPlan plan_blocks(const GemmDescriptor &desc, const UarchParams &uarch) {
  BlockPlan plan;
  const int M = desc.M, N = desc.N, K = desc.K;
  // Use actual element size from descriptor (4 for FP32, 2 for BF16)
  const int elem = static_cast<int>(desc.wei_elem_size);
  const int nt = desc.num_threads;
  plan.num_threads = nt;
  plan.pack_a = true; plan.pack_b = true;

  // MR/NR defaults. Callers override based on dtype/postops/M-size.
  plan.MR = 6;
  plan.NR = 16;
  const int MR = plan.MR, NR = plan.NR;

  // ── KB: K-block size ──
  //
  // Zen5: L1D = 48KB per core.
  // Constraint: A micro-panel (MR × KB × elem) must fit in 80% of L1.
  //   FP32: 0.8 × 48KB / (6 × 4) = 1,600
  //   BF16: 0.8 × 48KB / (6 × 2) = 3,200
  //
  // Also: B micro-tile (NR × KB × elem) should fit in L1 for reuse.
  //   FP32: 0.8 × 48KB / (16 × 4) = 600
  //   BF16: 0.8 × 48KB / (16 × 2) = 1,200
  //
  // KB = min(A-fit, B-fit), distributed evenly across K.
  int l1_budget = static_cast<int>(0.8 * uarch.l1d_bytes);
  int l2_budget = uarch.l2_bytes / 2;

  int kb_max_l1 = l1_budget / (MR * elem);
  int kb_max_l2 = l2_budget / (NR * elem);
  int kb_max = std::min(kb_max_l1, kb_max_l2);
  kb_max = std::max(kb_max, 64);
  plan.KB = choose_even_kb(K, kb_max);

  // ── NB: N-block size ──
  //
  // B panel (KB × NB × elem) should fit in 50% of L2 for reuse across M-tiles.
  //
  // Zen5: L2 = 1MB per core.
  //   FP32 KB=512: 512KB / (512 × 4) = 250 columns
  //   BF16 KB=1024: 512KB / (1024 × 2) = 250 columns (same, elem halved but KB doubled)
  //
  // L3-aware: When multi-threaded, B panel is shared in L3.
  // Zen5: L3 = 32MB per CCD, 8 cores per CCD.
  //   Per-thread L3 budget = 0.5 × 32MB / min(nt, 8) = 2MB for 8 threads
  //   NB_l3 = 2MB / (KB × elem × threads_sharing_l3)
  int nb_from_l2 = l2_budget / (plan.KB * elem);

  int nb_from_l3 = nb_from_l2;
  if (nt > 1) {
    int cores_per_ccd = std::min(uarch.num_cores, 8);
    int threads_sharing_l3 = std::min(nt, cores_per_ccd);
    int l3_budget = static_cast<int>(0.5 * uarch.l3_bytes_per_ccd);
    nb_from_l3 = l3_budget / (threads_sharing_l3 * plan.KB * elem);
  }

  int nb_max = std::min(nb_from_l2, nb_from_l3);
  plan.NB = round_down(std::max(nb_max, NR), NR);
  if (plan.NB > N) plan.NB = N;

  // Distribute N evenly to avoid one small leftover tile
  if (plan.NB < N && plan.NB >= NR) {
    int n_jtiles = (N + plan.NB - 1) / plan.NB;
    int even_nb = round_up((N + n_jtiles - 1) / n_jtiles, NR);
    if (even_nb > 0 && even_nb <= plan.NB + NR)
      plan.NB = std::min(even_nb, N);
  }

  // ── MB: M-block size ──
  //
  // C tile (MB × NB × 4) + A tile (MB × KB × elem) should fit in L1+L2.
  // C is always FP32 (accumulation), so use 4 for C element size.
  //
  // Zen5: L1+L2 = 48KB + 1MB = ~1.05MB
  //   MB = floor((L1+L2) / ((NB × 4) + (KB × elem)), MR)
  int total_per_core = uarch.l1d_bytes + uarch.l2_bytes;
  int c_cost_per_row = plan.NB * 4;  // C is always FP32
  int a_cost_per_row = plan.KB * elem;
  plan.MB = round_down(
    std::max(total_per_core / (c_cost_per_row + a_cost_per_row), MR), MR);
  if (plan.MB > M) plan.MB = M;

  // ── Thread load balancing ──
  //
  // Ensure at least nt tiles (ideally 2×nt) for good work distribution.
  // Shrink NB/MB if too few tiles. Zen5 with 8+ threads needs enough
  // parallelism across both M and N dimensions.
  if (nt > 1) {
    int min_tiles = nt;
    int ideal_tiles = 2 * nt;
    int target = min_tiles;

    {
      int ic_t = (M + plan.MB - 1) / plan.MB;
      int jc_t = (N + plan.NB - 1) / plan.NB;
      if (ic_t * jc_t >= ideal_tiles) target = ideal_tiles;
    }

    while (true) {
      int ic_t = (M + plan.MB - 1) / plan.MB;
      int jc_t = (N + plan.NB - 1) / plan.NB;
      if (ic_t * jc_t >= target) break;
      if (plan.NB > NR) { plan.NB -= NR; continue; }
      if (plan.MB > MR && M > MR) { plan.MB -= MR; continue; }
      break;
    }
  }

  return plan;
}

} // namespace ai
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
