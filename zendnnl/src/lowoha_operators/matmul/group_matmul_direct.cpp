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
 *******************************************************************************/

/// Group MatMul direct — public entry point and input validation.
///
/// Implementation details:
///   - group_matmul/group_matmul_parallel.cpp — parallel expert dispatch (OMP).
///   - group_matmul/group_matmul_moe_postop.cpp — optional MoE weighted-reduce post-op.

#include <sstream>
#include <vector>

#include "group_matmul/group_matmul_direct.hpp"
#include "lowoha_matmul_utils.hpp"
#include "lowoha_operators/common/omp_thread_control.hpp"
#include "lowoha_operators/common/operator_instrumentation.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

using namespace zendnnl::ops;
using zendnnl::common::size_of;
using zendnnl::common::op_instrumentation;

namespace {

status_t validate_group_matmul_nonempty_vectors(
    const std::vector<int> &M,
    const std::vector<int> &N,
    const std::vector<int> &K,
    const std::vector<matmul_params> &params,
    const std::vector<const void *> &src,
    const std::vector<const void *> &weight,
    const std::vector<void *> &dst,
    const std::vector<const void *> &bias,
    const std::vector<bool> &is_weights_const,
    const std::vector<int> &lda,
    const std::vector<int> &ldb,
    const std::vector<int> &ldc) {

  if (M.empty() || N.empty() || K.empty() || params.empty() || src.empty() ||
      weight.empty() || dst.empty() || bias.empty() || is_weights_const.empty() ||
      lda.empty() || ldb.empty() || ldc.empty()) {
    log_error("group_matmul_direct: empty input vectors");
    return status_t::failure;
  }
  return status_t::success;
}

status_t validate_parallel_gemm_inputs(
    const std::vector<char> &layout,
    const std::vector<bool> &transA,
    const std::vector<bool> &transB,
    const std::vector<int> &M,
    const std::vector<int> &N,
    const std::vector<int> &K,
    const std::vector<float> &alpha,
    const std::vector<const void *> &src,
    const std::vector<int> &lda,
    const std::vector<const void *> &weight,
    const std::vector<int> &ldb,
    const std::vector<const void *> &bias,
    const std::vector<float> &beta,
    const std::vector<void *> &dst,
    const std::vector<int> &ldc,
    const std::vector<bool> &is_weights_const,
    const std::vector<matmul_params> &params) {

  const size_t num_ops = M.size();

  if (num_ops == 0) {
    log_error("group_matmul parallel: num_ops is 0");
    return status_t::failure;
  }

  if (layout.size() != num_ops || transA.size() != num_ops ||
      transB.size() != num_ops || N.size() != num_ops ||
      K.size() != num_ops || alpha.size() != num_ops ||
      src.size() != num_ops || lda.size() != num_ops ||
      weight.size() != num_ops || ldb.size() != num_ops ||
      bias.size() != num_ops || beta.size() != num_ops ||
      dst.size() != num_ops || ldc.size() != num_ops ||
      is_weights_const.size() != num_ops || params.size() != num_ops) {
    log_error("group_matmul parallel: vector size mismatch, num_ops=", num_ops);
    return status_t::failure;
  }

  for (size_t i = 0; i < num_ops; ++i) {
    if (!src[i] || !weight[i] || !dst[i]) {
      log_error("group_matmul parallel: null pointer at operation ", i);
      return status_t::failure;
    }
    if (M[i] <= 0 || N[i] <= 0 || K[i] <= 0) {
      log_error("group_matmul parallel: invalid dimensions at operation ", i,
                ": M=", M[i], ", N=", N[i], ", K=", K[i]);
      return status_t::failure;
    }
  }

  return status_t::success;
}

status_t validate_sequential_gemm_inputs(
    const std::vector<char> &layout,
    const std::vector<bool> &transA,
    const std::vector<bool> &transB,
    const std::vector<int> &M,
    const std::vector<int> &N,
    const std::vector<int> &K,
    const std::vector<float> &alpha,
    const std::vector<const void *> &src,
    const std::vector<int> &lda,
    const std::vector<const void *> &weight,
    const std::vector<int> &ldb,
    const std::vector<const void *> &bias,
    const std::vector<float> &beta,
    const std::vector<void *> &dst,
    const std::vector<int> &ldc,
    const std::vector<bool> &is_weights_const,
    const std::vector<matmul_params> &params) {

  const size_t num_ops = M.size();

  if (num_ops == 0) {
    log_error("group_matmul sequential: num_ops is 0");
    return status_t::failure;
  }

  if (src.size() != 1) {
    log_error("group_matmul sequential: src.size() must be 1, got ", src.size());
    return status_t::failure;
  }

  if (layout.size() != num_ops || transA.size() != num_ops ||
      transB.size() != num_ops || N.size() != num_ops ||
      K.size() != num_ops || alpha.size() != num_ops ||
      lda.size() != num_ops || weight.size() != num_ops ||
      ldb.size() != num_ops || bias.size() != num_ops ||
      beta.size() != num_ops || dst.size() != num_ops ||
      ldc.size() != num_ops || is_weights_const.size() != num_ops ||
      params.size() != num_ops) {
    log_error("group_matmul sequential: vector size mismatch, num_ops=", num_ops);
    return status_t::failure;
  }

  if (!src[0]) {
    log_error("group_matmul sequential: null src pointer");
    return status_t::failure;
  }

  for (size_t i = 0; i < num_ops; ++i) {
    if (!weight[i] || !dst[i]) {
      log_error("group_matmul sequential: null pointer at operation ", i);
      return status_t::failure;
    }
    if (M[i] <= 0 || N[i] <= 0 || K[i] <= 0) {
      log_error("group_matmul sequential: invalid dimensions at operation ", i,
                ": M=", M[i], ", N=", N[i], ", K=", K[i]);
      return status_t::failure;
    }
  }

  for (size_t i = 1; i < num_ops; ++i) {
    if (M[i] != M[0]) {
      log_error("group_matmul sequential: M must be constant across layers, "
                "M[0]=", M[0], ", M[", i, "]=", M[i]);
      return status_t::failure;
    }
  }

  for (size_t i = 1; i < num_ops; ++i) {
    if (K[i] != N[i - 1]) {
      log_error("group_matmul sequential: dimension mismatch at layer ", i,
                ": K[", i, "]=", K[i], " != N[", i - 1, "]=", N[i - 1]);
      return status_t::failure;
    }
  }

  return status_t::success;
}

} // namespace

status_t group_matmul_direct(const std::vector<char> &layout,
                             const std::vector<bool> &transA,
                             const std::vector<bool> &transB,
                             const std::vector<int> &M,
                             const std::vector<int> &N,
                             const std::vector<int> &K,
                             const std::vector<float> &alpha,
                             const std::vector<const void *> &src,
                             const std::vector<int> &lda,
                             const std::vector<const void *> &weight,
                             const std::vector<int> &ldb,
                             const std::vector<const void *> &bias,
                             const std::vector<float> &beta,
                             const std::vector<void *> &dst,
                             const std::vector<int> &ldc,
                             const std::vector<bool> &is_weights_const,
                             std::vector<matmul_params> &params,
                             const group_matmul_moe_postop_params *moe_postop,
                             const grp_matmul_gated_act_params *gated_act,
                             const grp_matmul_fused_moe_params *fused_moe) {

  // Always-on guard: check all vectors before any indexing.
  // Enforces size consistency without per-element loops — single branch.
  if (M.empty() || params.empty() || src.empty())
    return status_t::failure;

  const size_t num_ops = M.size();

  if (N.size() != num_ops || K.size() != num_ops || weight.size() != num_ops ||
      dst.size() != num_ops || lda.size() != num_ops || ldb.size() != num_ops ||
      ldc.size() != num_ops || layout.size() != num_ops ||
      transA.size() != num_ops || transB.size() != num_ops ||
      alpha.size() != num_ops || beta.size() != num_ops ||
      bias.size() != num_ops || is_weights_const.size() != num_ops ||
      params.size() != num_ops) {
    log_error("group_matmul_direct: vector size mismatch");
    return status_t::failure;
  }
  if (src.size() != 1 && src.size() != num_ops) {
    log_error("group_matmul_direct: src.size() must be 1 or num_ops");
    return status_t::failure;
  }

  // MoE post-op and gated activation are parallel mode only.
  if (moe_postop != nullptr && src.size() == 1) {
    log_error("group_matmul_direct: moe_postop is only supported in parallel mode");
    return status_t::failure;
  }
  if (gated_act != nullptr
      && gated_act->act != grp_matmul_gated_act_t::none
      && src.size() == 1) {
    log_error("group_matmul_direct: gated_act is only supported in parallel mode");
    return status_t::failure;
  }
  if (fused_moe != nullptr && src.size() == 1) {
    log_error("group_matmul_direct: fused_moe is only supported in parallel mode");
    return status_t::failure;
  }
  if (fused_moe != nullptr && fused_moe->down_weight.empty()) {
    log_error("group_matmul_direct: fused_moe is non-null but down_weight "
              "is empty; pass nullptr to disable");
    return status_t::failure;
  }

  // Validate remaining inputs only when ZENDNNL_DIAGNOSTICS_ENABLE=1.
  status_t val = op_instrumentation::validate([&]() {
    if (validate_group_matmul_nonempty_vectors(M, N, K, params, src, weight, dst,
            bias, is_weights_const, lda, ldb, ldc) != status_t::success)
      return status_t::failure;
    // When MoE is enabled, all experts must have identical output dim
    // and dst dtype — the weighted-reduce reads all expert rows uniformly.
    if (moe_postop != nullptr) {
      for (size_t i = 1; i < num_ops; ++i) {
        if (N[i] != N[0]) {
          log_error("group_matmul_direct: moe_postop requires identical N across experts");
          return status_t::failure;
        }
        if (params[i].dtypes.dst != params[0].dtypes.dst) {
          log_error("group_matmul_direct: moe_postop requires identical dst dtype across experts");
          return status_t::failure;
        }
      }
      // When fused_moe is active, moe_postop operates on down_proj output
      // (D = N_down) not gate+up output (D = N). Validate uniform N_down.
      if (fused_moe != nullptr && fused_moe->N_down.size() >= num_ops) {
        for (size_t i = 1; i < num_ops; ++i) {
          if (fused_moe->N_down[i] != fused_moe->N_down[0]) {
            log_error("group_matmul_direct: fused_moe + moe_postop requires "
                      "uniform N_down across experts");
            return status_t::failure;
          }
        }
      }
    }
    // Validate moe_postop with the correct D: N_down[0] when fused, N[0] otherwise.
    const int moe_D = (fused_moe != nullptr && !fused_moe->N_down.empty())
                       ? fused_moe->N_down[0] : N[0];
    if (validate_group_matmul_moe_postop(moe_postop, moe_D,
                                          params[0].dtypes.dst) != status_t::success)
      return status_t::failure;
    return status_t::success;
  });
  if (val != status_t::success)
    return val;

  profiler_t profiler;
  bool is_profile = is_profile_enabled();
  if (is_profile)
    profiler.tbp_start();

  const char *gemm_mode = nullptr;

  const int32_t omp_mt = thread_guard::max_threads();
  const int32_t num_threads = resolve_num_threads(params[0].num_threads, omp_mt);
  thread_guard tg(num_threads, omp_mt);

  if (src.size() == 1) {
    val = op_instrumentation::validate([&]() {
      return validate_sequential_gemm_inputs(layout, transA, transB, M, N, K,
                 alpha, src, lda, weight, ldb, bias, beta, dst, ldc,
                 is_weights_const, params);
    });
    if (val != status_t::success)
      return val;

    static unsigned int auto_version = get_auto_tuner_ver();
    for (size_t i = 0; i < num_ops; ++i) {
      matmul_batch_params_t bp;
      bp.Batch_A = 1;
      bp.Batch_B = 1;
      const void *cur_src = (i == 0) ? src[i] : dst[i - 1];
      int cur_lda = (i == 0) ? lda[i] : ldc[i - 1];

      matmul_algo_t kernel = kernel_select(
          params[i], bp.Batch_A, bp.Batch_B,
          1, M[i], N[i], K[i], num_threads, bias[i], is_weights_const[i]);

      params[i].num_threads = num_threads;
      matmul_execute(
          layout[i], transA[i], transB[i],
          M[i], N[i], K[i], alpha[i],
          cur_src, cur_lda, weight[i], ldb[i],
          bias[i], beta[i], dst[i], ldc[i],
          is_weights_const[i],
          size_of(params[i].dtypes.src), size_of(params[i].dtypes.dst),
          num_threads, kernel, params[i], bp, auto_version);
    }
    gemm_mode = "sequential";
  } else {
    val = op_instrumentation::validate([&]() {
      return validate_parallel_gemm_inputs(layout, transA, transB, M, N, K,
                 alpha, src, lda, weight, ldb, bias, beta, dst, ldc,
                 is_weights_const, params);
    });
    if (val != status_t::success)
      return val;

    // Validate gated_act prerequisites before launching GEMMs (fast-fail).
    data_type_t act_dtype = data_type_t::none;
    const bool run_gated_act = (gated_act != nullptr
        && gated_act->act != grp_matmul_gated_act_t::none);
    if (run_gated_act) {
      act_dtype = params[0].dtypes.dst;
      if (act_dtype != data_type_t::f32 && act_dtype != data_type_t::bf16) {
        log_error("group_matmul_direct: gated_act requires f32 or bf16 dst");
        return status_t::failure;
      }
      for (size_t i = 1; i < num_ops; ++i) {
        if (params[i].dtypes.dst != act_dtype) {
          log_error("group_matmul_direct: gated_act requires uniform dst dtype");
          return status_t::failure;
        }
      }
      for (size_t i = 0; i < num_ops; ++i) {
        if (N[i] % 2 != 0) {
          log_error("group_matmul_direct: gated_act requires even N, N[",
                    i, "]=", N[i]);
          return status_t::failure;
        }
      }
    }

    // Gated activation and fused M-tile assume row-major layout.
    // apply_gated_act_inplace accesses dst as row-major (row = dst + m*ldc).
    const bool need_rowmajor = run_gated_act || (fused_moe != nullptr);
    if (need_rowmajor) {
      for (size_t i = 0; i < num_ops; ++i) {
        if (layout[i] != 'r' && layout[i] != 'R') {
          log_error("group_matmul_direct: gated_act/fused_moe requires "
                    "row-major layout, layout[", i, "]='", layout[i], "'");
          return status_t::failure;
        }
      }
    }

    if (fused_moe != nullptr) {
      // Note on act=none + fused_moe: Op2 will consume the raw first-half
      // columns of Op1's [M, 2*dim] output as its input.  This is *not*
      // MoE-semantically equivalent to a gated workflow and is a
      // potential footgun for framework integrators — the grp_matmul
      // gtests use this path deliberately as a two-call reference.
      // Frameworks that want "Op1 → activation → Op2" MUST also pass
      // gated_act with a non-none activation.
      // Fused down_proj uses K_down = N[i]/2 — N must be even.
      for (size_t i = 0; i < num_ops; ++i) {
        if (N[i] % 2 != 0) {
          log_error("group_matmul_direct: fused_moe requires even N, N[",
                    i, "]=", N[i]);
          return status_t::failure;
        }
      }
      if (fused_moe->down_weight.size() != num_ops ||
          fused_moe->N_down.size() != num_ops ||
          fused_moe->ldb_down.size() != num_ops ||
          fused_moe->bias_down.size() != num_ops ||
          fused_moe->dst_down.size() != num_ops ||
          fused_moe->ldc_down.size() != num_ops) {
        log_error("group_matmul_direct: fused_moe vector size mismatch");
        return status_t::failure;
      }
      for (size_t i = 0; i < num_ops; ++i) {
        if (fused_moe->down_weight[i] == nullptr) {
          log_error("group_matmul_direct: fused_moe down_weight[", i, "] is null");
          return status_t::failure;
        }
        if (fused_moe->dst_down[i] == nullptr) {
          log_error("group_matmul_direct: fused_moe dst_down[", i, "] is null");
          return status_t::failure;
        }
        if (fused_moe->N_down[i] <= 0) {
          log_error("group_matmul_direct: fused_moe N_down[", i, "]=",
                    fused_moe->N_down[i], " must be positive");
          return status_t::failure;
        }
        const int K_down_i = N[i] / 2;
        const int min_ldb = transB[i] ? K_down_i : fused_moe->N_down[i];
        if (fused_moe->ldb_down[i] < min_ldb) {
          log_error("group_matmul_direct: fused_moe ldb_down[", i, "]=",
                    fused_moe->ldb_down[i], " < required=", min_ldb);
          return status_t::failure;
        }
        if (fused_moe->ldc_down[i] < fused_moe->N_down[i]) {
          log_error("group_matmul_direct: fused_moe ldc_down[", i, "]=",
                    fused_moe->ldc_down[i], " < N_down=", fused_moe->N_down[i]);
          return status_t::failure;
        }
      }
      // Validate bias_dt_down is set when any bias_down is non-null.
      for (size_t i = 0; i < num_ops; ++i) {
        if (fused_moe->bias_down[i] != nullptr
            && fused_moe->bias_dt_down == data_type_t::none) {
          log_error("group_matmul_direct: fused_moe bias_down[", i,
                    "] is non-null but bias_dt_down is none");
          return status_t::failure;
        }
      }
      // When moe_postop is active, N_down must be uniform across experts
      // (weighted-reduce reads all expert outputs with the same D).
      if (moe_postop != nullptr) {
        for (size_t i = 1; i < num_ops; ++i) {
          if (fused_moe->N_down[i] != fused_moe->N_down[0]) {
            log_error("group_matmul_direct: fused_moe + moe_postop requires "
                      "uniform N_down, N_down[0]=", fused_moe->N_down[0],
                      " N_down[", i, "]=", fused_moe->N_down[i]);
            return status_t::failure;
          }
        }
      }
      status_t fused_st = group_matmul_fused_moe_execute(
          *fused_moe,
          run_gated_act ? gated_act->act : grp_matmul_gated_act_t::none,
          act_dtype,
          layout, transA, transB, M, N, K, alpha,
          src, lda, weight, ldb, bias, beta, dst, ldc,
          is_weights_const, params, num_threads, &gemm_mode);
      if (fused_st != status_t::success)
        return fused_st;

      // Weighted reduce after all experts complete Op2.
      if (moe_postop != nullptr) {
        const int D_down = fused_moe->N_down[0];
        status_t moe_val = validate_group_matmul_moe_postop(
            moe_postop, D_down, params[0].dtypes.dst);
        if (moe_val != status_t::success)
          return moe_val;
        status_t moe_st = group_matmul_moe_postop_execute(
            moe_postop, D_down, num_threads, params[0].dtypes.dst);
        if (moe_st != status_t::success)
          return moe_st;
      }
    } else {
      // Non-fused path: Op1 + Act (fused where possible), then separate Op2.
      const bool act_fused = group_matmul_run_parallel_dispatch(
          layout, transA, transB, M, N, K, alpha,
          src, lda, weight, ldb, bias, beta, dst, ldc,
          is_weights_const, params, num_threads, &gemm_mode,
          run_gated_act ? gated_act->act : grp_matmul_gated_act_t::none,
          act_dtype);

      if (run_gated_act && !act_fused) {
        status_t act_st = group_matmul_moe_act_execute(
            gated_act, dst, M, N, ldc, act_dtype, num_threads);
        if (act_st != status_t::success)
          return act_st;
      }

      if (moe_postop != nullptr) {
        status_t moe_val = validate_group_matmul_moe_postop(
            moe_postop, N[0], params[0].dtypes.dst);
        if (moe_val != status_t::success)
          return moe_val;
        status_t moe_st = group_matmul_moe_postop_execute(moe_postop, N[0],
            num_threads, params[0].dtypes.dst);
        if (moe_st != status_t::success)
          return moe_st;
      }
    }
  }

  if (is_profile)
    profiler.tbp_stop();

  if (apilog_info_enabled() || is_profile) {
    auto dt_str = [](data_type_t dt) -> const char * {
      switch (dt) {
      case data_type_t::f32:  return "f32";
      case data_type_t::bf16: return "bf16";
      case data_type_t::f16:  return "f16";
      case data_type_t::s8:   return "s8";
      case data_type_t::u8:   return "u8";
      default:                return "?";
      }
    };

    std::ostringstream ss;
    ss << "LOWOHA group_matmul_direct: "
       << "num_ops=" << num_ops
       << ", mode=" << (gemm_mode != nullptr ? gemm_mode : "null")
       << ", threads=" << num_threads
       << ", dtype=" << dt_str(params[0].dtypes.src)
       << ">" << dt_str(params[0].dtypes.dst);

    // M values (vary per expert) + sum
    int64_t m_sum = 0;
    ss << ", M=[";
    for (size_t i = 0; i < num_ops; ++i) {
      if (i > 0) ss << ",";
      ss << M[i];
      m_sum += M[i];
    }
    ss << "](sum=" << m_sum << ")";

    ss << ", N[0]=" << N[0] << ", K[0]=" << K[0];

    // Fusion flags: compact summary of what was enabled
    const bool has_act = (gated_act != nullptr
        && gated_act->act != grp_matmul_gated_act_t::none);
    const bool has_fused = (fused_moe != nullptr);
    const bool has_moe = (moe_postop != nullptr);

    if (has_act || has_fused || has_moe) {
      ss << ", fused=[";
      bool need_comma = false;
      if (has_act) {
        static const char *act_names[] = {
            "none", "silu_and_mul", "gelu_and_mul", "swiglu_oai_mul"};
        int ai = static_cast<int>(gated_act->act);
        ss << "act=" << ((ai >= 0 && ai <= 3) ? act_names[ai] : "?");
        need_comma = true;
      }
      if (has_fused) {
        if (need_comma) ss << ",";
        ss << "down_proj=N_down[0]=" << fused_moe->N_down[0];
        need_comma = true;
      }
      if (has_moe) {
        if (need_comma) ss << ",";
        ss << "moe_postop(tokens=" << moe_postop->num_tokens
           << ",topk=" << moe_postop->topk << ")";
      }
      ss << "]";
    }

    apilog_info(ss.str());
    if (is_profile)
      profilelog_verbose(ss.str(), ", time=", profiler.tbp_elapsedtime(),
                         profiler.get_res_str());
  }

  return status_t::success;
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
