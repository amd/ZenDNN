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

#include "lowoha_operators/matmul/matmul_native/gemm/planner/gemm_planner.hpp"
#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"
#include "operators/matmul/matmul_config.hpp"
#include "common/zendnnl_global.hpp"
#include <algorithm>
#include <cmath>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::ops::post_op_type_t;
using namespace zendnnl::error_handling;

static inline int round_up(int x, int m) { return ((x + m - 1) / m) * m; }
static inline int round_down(int x, int m) { return (x / m) * m; }

static int choose_even_kb(int K, int kb_max) {
  if (kb_max >= K) return K;
  int n_blocks = (K + kb_max - 1) / kb_max;
  int even_kb = (K + n_blocks - 1) / n_blocks;
  return round_up(even_kb, 8);
}

// ============================================================================
// Base blocking plan (shared by BF16 and FP32)
// ============================================================================

BlockPlan plan_blocks(const GemmDescriptor &desc, const UarchParams &uarch) {
  BlockPlan plan;
  const int M = desc.M, N = desc.N, K = desc.K;
  const int elem = static_cast<int>(desc.wei_elem_size);
  const int nt = desc.num_threads;
  plan.num_threads = nt;

  plan.MR = 6;
  plan.NR = 16;
  const int MR = plan.MR, NR = plan.NR;

  // KB: A micro-panel (MR×KB×elem) in 80% L1, B tile (NR×KB×elem) in L2/2.
  int l1_budget = static_cast<int>(0.8 * uarch.l1d_bytes);
  int l2_budget = uarch.l2_bytes / 2;

  int kb_max_l1 = l1_budget / (MR * elem);
  int kb_max_l2 = l2_budget / (NR * elem);
  int kb_max = std::min(kb_max_l1, kb_max_l2);
  kb_max = std::max(kb_max, 64);
  plan.KB = choose_even_kb(K, kb_max);

  // NB: B panel (KB×NB×elem) in L2/2, L3-aware for multi-thread.
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

  if (plan.NB < N && plan.NB >= NR) {
    int n_jtiles = (N + plan.NB - 1) / plan.NB;
    int even_nb = round_up((N + n_jtiles - 1) / n_jtiles, NR);
    if (even_nb > 0 && even_nb <= plan.NB + NR)
      plan.NB = std::min(even_nb, N);
  }

  // MB: C tile (MB×NB×4) + A tile (MB×KB×elem) in L1+L2.
  int total_per_core = uarch.l1d_bytes + uarch.l2_bytes;
  int c_cost_per_row = plan.NB * 4;
  int a_cost_per_row = plan.KB * elem;
  plan.MB = round_down(
    std::max(total_per_core / (c_cost_per_row + a_cost_per_row), MR), MR);
  if (plan.MB > M) plan.MB = M;

  // Thread load balancing.
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

// ============================================================================
// FP32 GEMM plan
// ============================================================================

FP32GemmPlan plan_fp32_gemm(
    const GemmDescriptor &desc,
    const UarchParams &uarch,
    const matmul_params &params) {

    const int M = desc.M, N = desc.N, K = desc.K;

    [[maybe_unused]] bool has_activation = false;
    bool has_complex_activation = false;
    for (const auto &po : params.postop_) {
        auto pt = po.po_type;
        if (pt == post_op_type_t::relu || pt == post_op_type_t::leaky_relu) {
            has_activation = true;
        } else if (pt == post_op_type_t::gelu_tanh ||
                   pt == post_op_type_t::gelu_erf ||
                   pt == post_op_type_t::sigmoid ||
                   pt == post_op_type_t::tanh ||
                   pt == post_op_type_t::swish ||
                   pt == post_op_type_t::elu) {
            has_activation = true;
            has_complex_activation = true;
            break;
        }
    }

    static thread_local struct {
        int M, N, K, threads; bool transA, transB;
        BlockPlan plan;
    } s_plan_cache = {0, 0, 0, 0, false, false, {}};

    BlockPlan plan;
    if (s_plan_cache.M == M && s_plan_cache.N == N && s_plan_cache.K == K &&
        s_plan_cache.threads == desc.num_threads &&
        s_plan_cache.transA == desc.transA &&
        s_plan_cache.transB == desc.transB) {
        plan = s_plan_cache.plan;
    } else {
        plan = plan_blocks(desc, uarch);

        if (desc.num_threads <= 1) {
            int b_bytes = K * NR_PACK * static_cast<int>(sizeof(float));
            if (b_bytes <= uarch.l2_bytes / 2)
                plan.KB = K;
        } else {
            int kb_b = (uarch.l2_bytes / 2)
                       / (NR_PACK * static_cast<int>(sizeof(float)));
            int kb_a = (uarch.l2_bytes / 2)
                       / (6 * static_cast<int>(sizeof(float)));
            int kb_max_mt = std::min(kb_a, kb_b);
            if (K <= kb_max_mt) {
                plan.KB = K;
            } else if (kb_max_mt > plan.KB) {
                int n_blocks = (K + kb_max_mt - 1) / kb_max_mt;
                plan.KB = ((K + n_blocks - 1) / n_blocks + 7) & ~7;
            }
        }

        s_plan_cache = {M, N, K, desc.num_threads, desc.transA, desc.transB, plan};
    }

    plan.MR = 6;
    if (N >= 64 && !has_complex_activation) {
        plan.NR = 64;
    } else if (N >= 32) {
        plan.NR = 32;
    } else {
        plan.NR = 16;
    }

    plan.NB = std::max(plan.NB / plan.NR * plan.NR, plan.NR);
    plan.NB = std::min(plan.NB, N);

    {
        bool will_pack_a = desc.transA
                           || (desc.lda * static_cast<int>(sizeof(float)) > 4096);
        if (will_pack_a) {
            int b_lines_per_krow = ((plan.NR + 15) / 16) * 64;
            int b_accessed_bytes = plan.KB * b_lines_per_krow;
            int l2_for_a = std::max(uarch.l2_bytes - b_accessed_bytes, 0);
            plan.MB = std::max(l2_for_a / (plan.KB * 4), plan.MR);
        } else {
            plan.MB = std::max((uarch.l1d_bytes + uarch.l2_bytes)
                               / (plan.NB * 4 + plan.KB * 4), plan.MR);
        }
        plan.MB = plan.MB / plan.MR * plan.MR;
        plan.MB = std::min(plan.MB, M);
    }

    if (plan.num_threads > 1) {
        int jc_tiles = (N + plan.NB - 1) / plan.NB;
        int ic_tiles = (M + plan.MB - 1) / plan.MB;
        int needed_ic = (plan.num_threads + jc_tiles - 1) / jc_tiles;
        if (needed_ic > ic_tiles && needed_ic > 1) {
            int m_panels = (M + plan.MR - 1) / plan.MR;
            int panels_per_block = std::max(m_panels / needed_ic, 1);
            plan.MB = panels_per_block * plan.MR;
            plan.MB = std::min(plan.MB, M);
        }
    }

    static bool s_log_fp32 = apilog_info_enabled();
    if (s_log_fp32) {
        apilog_info("Native FP32 GEMM plan: M=", M, " N=", N, " K=", K,
                    " MB=", plan.MB, " NB=", plan.NB, " KB=", plan.KB,
                    " MR=", plan.MR, " NR=", plan.NR,
                    " threads=", plan.num_threads);
    }

    return FP32GemmPlan{plan};
}

// ============================================================================
// BF16 GEMM plan
// ============================================================================

BF16GemmPlan plan_bf16_gemm(
    const GemmDescriptor &desc,
    const UarchParams &uarch,
    const matmul_params &params) {

    const int M = desc.M, N = desc.N, K = desc.K;
    const int K_padded = (K + 1) & ~1;

    bool has_activation = false;
    bool has_complex_activation = false;
    for (const auto &po : params.postop_) {
        auto pt = po.po_type;
        if (pt == post_op_type_t::relu || pt == post_op_type_t::leaky_relu) {
            has_activation = true;
        } else if (pt == post_op_type_t::gelu_tanh ||
                   pt == post_op_type_t::gelu_erf ||
                   pt == post_op_type_t::sigmoid ||
                   pt == post_op_type_t::tanh ||
                   pt == post_op_type_t::swish ||
                   pt == post_op_type_t::elu) {
            has_activation = true;
            has_complex_activation = true;
            break;
        }
    }

    static thread_local struct {
        int M, N, K, threads; bool transA, transB;
        BlockPlan plan;
    } s_plan_cache = {0, 0, 0, 0, false, false, {}};

    BlockPlan plan;
    if (s_plan_cache.M == M && s_plan_cache.N == N && s_plan_cache.K == K &&
        s_plan_cache.threads == desc.num_threads &&
        s_plan_cache.transA == desc.transA &&
        s_plan_cache.transB == desc.transB) {
        plan = s_plan_cache.plan;
    } else {
        plan = plan_blocks(desc, uarch);
        plan.KB = (plan.KB + 1) & ~1;

        {
            int b_full_k_bytes = K_padded * NR_PACK
                                 * static_cast<int>(sizeof(uint16_t));
            if (desc.num_threads <= 1) {
                int a_panel_bytes = 6 * K_padded
                                    * static_cast<int>(sizeof(uint16_t));
                int l1_limit = static_cast<int>(0.8 * uarch.l1d_bytes);
                if (b_full_k_bytes <= uarch.l2_bytes / 2
                    && a_panel_bytes <= l1_limit) {
                    plan.KB = K_padded;
                }
            } else {
                int l2_for_a = uarch.l2_bytes / 2;
                int kb_a = l2_for_a / (6 * static_cast<int>(sizeof(uint16_t)));
                int kb_b = (uarch.l2_bytes / 2)
                           / (NR_PACK * static_cast<int>(sizeof(uint16_t)));
                int kb_max_mt = std::min(kb_a, kb_b);
                kb_max_mt = (kb_max_mt + 1) & ~1;
                if (K_padded <= kb_max_mt) {
                    plan.KB = K_padded;
                } else if (kb_max_mt > plan.KB) {
                    int n_blocks = (K_padded + kb_max_mt - 1) / kb_max_mt;
                    int even_kb = ((K_padded + n_blocks - 1) / n_blocks + 7) & ~7;
                    plan.KB = (even_kb + 1) & ~1;
                }
            }
        }

        s_plan_cache = {M, N, K, desc.num_threads, desc.transA, desc.transB, plan};
    }

    const bool is_decode = (M <= 4);
    const char *path_name = "gemm";

    if (is_decode) {
        plan.MR = M;
        plan.MB = M;
        if (N >= 64)       plan.NR = 64;
        else if (N >= 32)  plan.NR = 32;
        else               plan.NR = 16;
        path_name = "decode";
    } else if (N >= 64 && !has_complex_activation) {
        if (M % 6 == 0 || M >= 18) {
            plan.MR = 6;
        } else if (M % 4 == 0) {
            plan.MR = 4;
        } else if (M % 6 <= 3 && M > 12) {
            plan.MR = 4;
        } else {
            plan.MR = 6;
        }
        plan.NR = 64;
        path_name = "gemm-nr64";
    } else if (N >= 32) {
        plan.MR = (M % 6 == 0 || M >= 18) ? 6
                : (M % 4 == 0) ? 4 : 6;
        plan.NR = 32;
    } else {
        plan.MR = (M % 6 == 0 || M >= 18) ? 6
                : (M % 4 == 0) ? 4 : 6;
        plan.NR = 16;
    }

    plan.NB = std::max(plan.NB / plan.NR * plan.NR, plan.NR);
    plan.NB = std::min(plan.NB, N);

    if (!is_decode) {
        bool will_pack_a = (desc.lda * static_cast<int>(sizeof(uint16_t)) > 4096);
        if (will_pack_a) {
            int b_lines_per_krow = ((plan.NR + 15) / 16) * 64;
            int k_pairs_kb = (plan.KB + 1) / 2;
            int b_accessed_bytes = k_pairs_kb * b_lines_per_krow;
            int l2_for_a = std::max(uarch.l2_bytes - b_accessed_bytes, 0);
            plan.MB = std::max(l2_for_a / (plan.KB * 2), plan.MR);
        } else {
            plan.MB = std::max((uarch.l1d_bytes + uarch.l2_bytes)
                               / (plan.NB * 4 + plan.KB * 2), plan.MR);
        }
        plan.MB = plan.MB / plan.MR * plan.MR;
        plan.MB = std::min(plan.MB, M);
    }

    if (!is_decode && plan.num_threads > 1) {
        int jc_tiles = (N + plan.NB - 1) / plan.NB;
        int ic_tiles = (M + plan.MB - 1) / plan.MB;
        int needed_ic = (plan.num_threads + jc_tiles - 1) / jc_tiles;
        if (needed_ic > ic_tiles && needed_ic > 1) {
            int m_panels = (M + plan.MR - 1) / plan.MR;
            int panels_per_block = std::max(m_panels / needed_ic, 1);
            plan.MB = panels_per_block * plan.MR;
            plan.MB = std::min(plan.MB, M);
            ic_tiles = (M + plan.MB - 1) / plan.MB;
        }
        if (ic_tiles * jc_tiles < plan.num_threads && plan.NB > plan.NR) {
            int needed_jc = (plan.num_threads + ic_tiles - 1) / ic_tiles;
            int nb_target = (N + needed_jc - 1) / needed_jc;
            nb_target = std::max((nb_target / plan.NR) * plan.NR, plan.NR);
            plan.NB = std::min(nb_target, plan.NB);
        }
    }

    static bool s_log_bf16 = apilog_info_enabled();
    if (s_log_bf16) {
        int jt = (N + plan.NB - 1) / plan.NB;
        int it = (M + plan.MB - 1) / plan.MB;
        apilog_info("Native BF16 GEMM plan: M=", M, " N=", N, " K=", K,
                    " MB=", plan.MB, " NB=", plan.NB, " KB=", plan.KB,
                    " MR=", plan.MR, " NR=", plan.NR,
                    " path=", path_name,
                    " ic=", it, " jc=", jt, " tiles=", it * jt,
                    " threads=", plan.num_threads);
    }

    return BF16GemmPlan{plan, path_name, is_decode,
                        has_activation, has_complex_activation};
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
