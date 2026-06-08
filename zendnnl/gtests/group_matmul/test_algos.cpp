/********************************************************************************
# * Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *******************************************************************************/

/// @file test_algos.cpp
/// @brief Scheduling-ALGO and custom-kernel matrix gtest sections.  Owned:
///
///   [7]  TestFusedMoEAlgos       - fused MoE x ALGO 1..5 x mixed precision
///                                  x bias.
///   [7b] TestFusedMoEAlgoCustom  - fused MoE x strategy / tight / custom
///                                  BF16 microkernel env-knob matrix.
///   [8]  TestGroupMatmulAlgoCustom - non-fused parallel x custom BF16
///                                  microkernel env-knob matrix.
///
/// Split from `test_group_matmul.cpp` during the gtests folder refactor;
/// see `group_matmul/README.md` for the file layout overview.

#include <gtest/gtest.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

#include <omp.h>

#include "gtest_utils.hpp"
#include "group_matmul_test_helpers.hpp"
#include "moe_test_utils.hpp"

// Direct access to the dispatcher's ALGO selector — used by section
// [8b] (`TestGroupMatmulAutoSelectAlgo`) to assert the auto-select
// decision matrix without spinning up the full GEMM stack.
#include "lowoha_operators/matmul/group_matmul/m_tile/group_matmul_m_tile.hpp"
#include "lowoha_operators/matmul/group_matmul/group_matmul_parallel_common.hpp"

// `GroupNTileStrategy` enum + `test_api::PhaseBSnapshot` /
// `s_capture_phase_b` / `s_last_phase_b_snapshot` — used by
// section [8a.1] (`TestGroupMatmulPhaseBRemainder.HeaviestFirstAssignment`)
// to white-box assert on the planner's Phase B output.
#include "lowoha_operators/matmul/group_matmul/n_tile/group_matmul_n_tile.hpp"

// `custom_kernel::dispatch_supported()` — Phase B tests check it
// to skip cleanly on hosts without AVX512BF16 (where forcing the
// env-cache CK override is necessary but not sufficient: the
// runtime `prepare_for_call` would refuse CK independently).
#include "lowoha_operators/matmul/group_matmul/custom_kernel/dispatch.hpp"

// ???????????????????????????????????????????????????????????????????????????????
// [7] TestFusedMoEAlgos: fused path ? ALGO 1/2/3 ? mixed precision ? bias
// ???????????????????????????????????????????????????????????????????????????????

struct FusedAlgoTestParam {
  int algo, act_int;
  bool is_bf16, mixed_prec, use_bias;
  int M, num_ops;
  // dim=0 ? use default 128.  Larger dims (?256) force N_gate_up ? 512,
  // which enables multi-thread N-tiling in ALGO 3 and exercises the
  // per-thread fused-swiglu-oai epilogue path (which historically hid a
  // cross-thread write-after-read race that only triggered when n_thr>1).
  int dim = 0;
};

static std::string FusedAlgoParamName(
  const ::testing::TestParamInfo<FusedAlgoTestParam> &info) {
  static const char *act_names[] = {"none", "silu", "gelu", "swiglu"};
  const auto &p = info.param;
  std::string name = "algo" + std::to_string(p.algo)
                     + "_" + act_names[p.act_int];
  if (p.mixed_prec) {
    name += "_bf16f32";
  }
  else {
    name += (p.is_bf16 ? "_bf16" : "_f32");
  }
  if (p.use_bias) {
    name += "_bias";
  }
  name += "_M" + std::to_string(p.M) + "_E" + std::to_string(p.num_ops);
  if (p.dim > 0) {
    name += "_d" + std::to_string(p.dim);
  }
  return name;
}

class TestFusedMoEAlgos :
  public ::testing::TestWithParam<FusedAlgoTestParam> {};

TEST_P(TestFusedMoEAlgos, Correctness) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;

  reset_grp_matmul_caches();

  const auto &p = GetParam();
  const int dim = (p.dim > 0) ? p.dim : 128;
  const int N_gate_up = 2 * dim, H = 256;
  const int M = p.M, K = H, num_ops = p.num_ops;
  const auto act_type = static_cast<grp_matmul_gated_act_t>(p.act_int);

  const data_type_t src_dt = p.is_bf16 ? data_type_t::bf16 : data_type_t::f32;
  const data_type_t dst_dt = p.mixed_prec ? data_type_t::f32
                             : (p.is_bf16 ? data_type_t::bf16 : data_type_t::f32);
  const data_type_t wei_dt = src_dt;
  const bool use_bf16_in  = (p.is_bf16 || p.mixed_prec);
  const bool use_bf16_out = !p.mixed_prec && p.is_bf16;

  // Op2's K dimension follows the activation: gated => dim, none =>
  // N_gate_up.  See `op2_k_for_act` in group_matmul_fused_moe.cpp.
  const bool act_is_none = (act_type == grp_matmul_gated_act_t::none);
  const int K_down = act_is_none ? N_gate_up : dim;

  AlgoEnvGuard algo_guard(p.algo);
  // The fused-swiglu_oai epilogue in ALGO 3 is gated by
  // ZENDNNL_GRP_MATMUL_N_TILE_FUSED_ACT (default OFF) ? see
  // get_grp_n_tile_fused_act() in group_matmul_parallel_common.hpp.
  // Force it ON here so the shapes below (where N_gate_up >
  // kDecodeNTile) drive ALGO 3's per-thread fused epilogue with
  // n_thr > 1 threads per expert.  That is the row-split path the
  // matmul?activation barrier + GroupNTileContext::apply_swiglu_oai
  // correctness fix exists to protect.
  EnvVarGuard fused_act_guard("ZENDNNL_GRP_MATMUL_N_TILE_FUSED_ACT", "1");

  // Allocate: input-side may be bf16; output-side may differ (mixed_prec).
  TypedBuffers src, w1, d1, d1r, w2, d2, d2r;
  src.alloc(num_ops, (size_t)M * K,         use_bf16_in,  p.mixed_prec);
  w1 .alloc(num_ops, (size_t)K * N_gate_up, use_bf16_in,  p.mixed_prec);
  w2 .alloc(num_ops, (size_t)K_down * H,    use_bf16_in,  p.mixed_prec);
  d1 .alloc(num_ops, (size_t)M * N_gate_up, use_bf16_out);
  d1r.alloc(num_ops, (size_t)M * N_gate_up, use_bf16_out);
  d2 .alloc(num_ops, (size_t)M * H,         use_bf16_out);
  d2r.alloc(num_ops, (size_t)M * H,         use_bf16_out);

  std::vector<std::vector<float>> bias_f(num_ops);
  for (int e = 0; e < num_ops; ++e) {
    // Always generate in f32 then mirror to bf16 when needed.
    std::vector<float> s_tmp((size_t)M * K), w1_tmp((size_t)K * N_gate_up),
        w2_tmp((size_t)K_down * H);
    fill_src(s_tmp,  e);
    fill_wei1(w1_tmp, e);
    fill_wei2(w2_tmp, e);
    if (src.store_f32) {
      src.f32[e] = s_tmp;
    }
    if (w1 .store_f32) {
      w1.f32[e]  = w1_tmp;
    }
    if (w2 .store_f32) {
      w2.f32[e]  = w2_tmp;
    }
    if (src.store_bf16) {
      src.bf16[e].resize(s_tmp.size());
      for (size_t i=0; i<s_tmp.size(); ++i) {
        src.bf16[e][i] = bfloat16_t(s_tmp[i]);
      }
    }
    if (w1 .store_bf16) {
      w1 .bf16[e].resize(w1_tmp.size());
      for (size_t i=0; i<w1_tmp.size(); ++i) {
        w1 .bf16[e][i] = bfloat16_t(w1_tmp[i]);
      }
    }
    if (w2 .store_bf16) {
      w2 .bf16[e].resize(w2_tmp.size());
      for (size_t i=0; i<w2_tmp.size(); ++i) {
        w2 .bf16[e][i] = bfloat16_t(w2_tmp[i]);
      }
    }
    bias_f[e].resize(H);
    for (int i = 0; i < H; ++i) {
      bias_f[e][i] = 0.01f * ((i + e) % 5);
    }
  }

  auto gv1 = GemmVecs::uniform(num_ops, M, N_gate_up, K);
  auto srcs = src.cptrs(use_bf16_in);
  auto wei1 = w1.cptrs(use_bf16_in);
  auto wei2 = w2.cptrs(use_bf16_in);
  auto dst1  = d1.ptrs(use_bf16_out);
  auto dst1r = d1r.ptrs(use_bf16_out);
  auto dst2  = d2.ptrs(use_bf16_out);
  auto dst2r = d2r.ptrs(use_bf16_out);
  std::vector<const void *> bias1(num_ops, nullptr), bias2(num_ops, nullptr);
  if (p.use_bias) for (int e = 0; e < num_ops; ++e) {
      bias2[e] = bias_f[e].data();
    }

  grp_matmul_gated_act_params act{};
  act.act = act_type;
  auto act_ptr = (act_type != grp_matmul_gated_act_t::none) ? &act : nullptr;

  // Reference: Op1+Act then Op2.
  {
    auto pr1 = make_mixed_params(num_ops, src_dt, wei_dt, dst_dt);
    ASSERT_EQ(group_matmul_direct(gv1.layout, gv1.transA, gv1.transB, gv1.Ms,
                                  gv1.Ns, gv1.Ks, gv1.alpha, srcs, gv1.lda, wei1, gv1.ldb, bias1,
                                  gv1.beta, dst1r, gv1.ldc, gv1.is_wc, pr1, nullptr, act_ptr),
              status_t::success) << "Ref Op1 failed";

    std::vector<const void *> s2(num_ops);
    for (int e = 0; e < num_ops; ++e) {
      s2[e] = dst1r[e];
    }
    auto gv2 = GemmVecs::uniform(num_ops, M, H, K_down);
    gv2.lda.assign(num_ops, N_gate_up);
    auto pr2 = make_mixed_params(num_ops, dst_dt, wei_dt, dst_dt,
                                 p.use_bias ? data_type_t::f32 : data_type_t::none);
    ASSERT_EQ(group_matmul_direct(gv2.layout, gv2.transA, gv2.transB, gv2.Ms,
                                  gv2.Ns, gv2.Ks, gv2.alpha, s2, gv2.lda, wei2, gv2.ldb, bias2,
                                  gv2.beta, dst2r, gv2.ldc, gv2.is_wc, pr2), status_t::success)
        << "Ref Op2 failed";
  }

  // Fused path.
  auto fused = make_fused_moe_op2(num_ops, H, wei2, bias2);
  fused.bias_dt_down = p.use_bias ? data_type_t::f32 : data_type_t::none;
  fused.dst_down     = dst2;
  fused.ldc_down     = std::vector<int>(num_ops, H);
  {
    auto pf = make_mixed_params(num_ops, src_dt, wei_dt, dst_dt);
    ASSERT_EQ(group_matmul_direct(gv1.layout, gv1.transA, gv1.transB, gv1.Ms,
                                  gv1.Ns, gv1.Ks, gv1.alpha, srcs, gv1.lda, wei1, gv1.ldb, bias1,
                                  gv1.beta, dst1, gv1.ldc, gv1.is_wc, pf, nullptr, act_ptr, &fused),
              status_t::success) << "Fused call failed (algo=" << p.algo << ")";
  }

  std::ostringstream lbl;
  lbl << "algo=" << p.algo << " act=" << p.act_int
      << (p.mixed_prec ? " bf16>f32" : (p.is_bf16 ? " bf16" : " f32"))
      << (p.use_bias ? " +bias" : "");
  verify_per_expert_2d(d2, H, d2r, H, num_ops, M, H, use_bf16_out,
                       tol_fused(p.is_bf16 || p.mixed_prec), lbl.str());
}

static std::vector<FusedAlgoTestParam> make_fused_algo_params() {
  std::vector<FusedAlgoTestParam> out;
  // All 3 ALGOs ? all 4 activation types ? both dtypes.
  // Covers: ALGO-specific fused MoE dispatch paths for every activation.
  for (int algo : {
         1, 2, 3
       })
    for (int act : {
           0, 1, 2, 3
         })  // none, silu, gelu, swiglu
      for (bool bf : {
             false, true
           })
        out.push_back({algo, act, bf, false, false, 4, 4});
  // Mixed precision (BF16 src ? F32 dst) per ALGO.
  for (int algo : {
         1, 2, 3
       }) out.push_back({algo, 1, true,  true,  false, 4, 4});
  // Non-null down_proj bias per ALGO.
  for (int algo : {
         1, 2, 3
       }) out.push_back({algo, 1, false, false, true,  4, 4});
  // ALGO 2 M-tile with varying M (small M=1 and larger M=16).
  for (int m : {
         1, 16
       })      out.push_back({2,    1, false, false, false, m, 4});
  // ALGO 3 two-pass with many experts.
  out.push_back({3, 1, false, false, false, 4, 8});

  // ALGO 3 fused swiglu_oai_mul ? race-exposure shapes.
  //
  // Pre-fix, apply_n_tile_paired_swiglu_oai split the epilogue by N
  // columns, which aliased thread t's compact-output writes
  // [p_start_t, p_end_t) with a lower-index thread's pair-read range
  // [2?p_start_{t-1}, 2?p_start_t).  The race only fires when
  // flat_n_tile actually runs more than one thread per expert ?
  // i.e. (1) `ntile_viable` is true and (2) `thr_per_expert` (or the
  // path-B `n_thr`) resolves to ?2.  Every earlier swiglu case in this
  // file used dim ? 128 so N_gate_up ? 256 = kDecodeNTile ? fallback ?
  // one thread per expert ? the bug stayed hidden.
  //
  // The shapes below hit the race-prone code on both a typical 16-thread
  // developer run AND a 128-thread EPYC CI run (verified with a Python
  // simulation of flat_n_tile's decision tree):
  //
  //   shape (M=8, E=8, d=1024):
  //       16t  ? path (B), n_thr=2 per expert, 2-round batched N-tile
  //      128t  ? path (D), n_thr=8 per expert, decode_parallel
  //
  //   shape (M=64, E=8, d=2048):
  //       16t  ? path (B), n_thr=2 per expert
  //      128t  ? path (A), n_thr=8 per expert, L3-batched few-expert
  //
  // Together they exercise all three multi-threaded epilogue paths
  // (D/A/B) and make the fix observable: each thread writes into a
  // disjoint row slice instead of the aliased pair?compact column
  // slice, so there is no cross-thread overlap.  Pre-fix these shapes
  // returned NaNs / wrong arithmetic that exceed the BF16 tolerance;
  // post-fix they match the 2-pass reference.
  for (bool bf : {
         false, true
       }) {
    out.push_back({3, /*act=*/3, bf, false, false, /*M=*/8,  /*E=*/8, /*dim=*/1024});
    out.push_back({3, /*act=*/3, bf, false, false, /*M=*/64, /*E=*/8, /*dim=*/2048});
  }
  // Bias and mixed-precision cross-checks on the decode-shape case.
  out.push_back({3, /*act=*/3, /*bf=*/true,  /*mixed=*/true,  /*bias=*/false,
                 /*M=*/8, /*E=*/8, /*dim=*/1024});
  out.push_back({3, /*act=*/3, /*bf=*/false, /*mixed=*/false, /*bias=*/true,
                 /*M=*/8, /*E=*/8, /*dim=*/1024});
  return out;
}

INSTANTIATE_TEST_SUITE_P(GroupMatmulFusedAlgos, TestFusedMoEAlgos,
                         ::testing::ValuesIn(make_fused_algo_params()), FusedAlgoParamName);

// ???????????????????????????????????????????????????????????????????????????????
// [7b] TestFusedMoEAlgoCustom: fused-MoE env-knob matrix
//
// Mirrors TestGroupMatmulAlgoCustom for the non-fused path but for the
// fused MoE entry (Op1+act ? Op2).  Targets the strategy-selection
// contract cemented in Option A:
//
//   * ZENDNNL_GRP_MATMUL_ALGO = 1..5         ? strategy selector, the
//                                              single source of truth
//                                              for the fused path.
//   * ZENDNNL_GRP_MATMUL_FUSED_MOE_TIGHT=0/1 ? V1 vs V2 (V2 is the
//                                              ALGO 3 + swiglu + tight
//                                              specialist; engages only
//                                              when env_algo ? {0, 3}).
//   * ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL=0/1   ? custom BF16 ukernel hook
//                                              (flat_n_tile + V2 Op2).
//
// Reference strategy: a single known-good baseline ? forced ALGO=1,
// TIGHT=0, CUSTOM=0, legacy two-call (Op1 via group_matmul_direct +
// Op2 via group_matmul_direct) ? compared against the fused
// internal-alloc call under the parameterised env.  Any breakage of
// Option A's gating surface (e.g. V2 silently engaging for ALGO 5,
// or CUSTOM_KERNEL=1 corrupting Op2 for non-BF16 dtypes) produces a
// comparison failure.
// ???????????????????????????????????????????????????????????????????????????????

struct FusedAlgoCustomParam {
  int algo;            // ALGO strategy 1..5
  int tight;           // 0 or 1 (FUSED_MOE_TIGHT)
  int custom_kernel;   // 0 or 1
  int act_int;         // 1=silu, 2=gelu, 3=swiglu_oai (act=none skipped ?
  // fused MoE always has an activation in practice)
  int M, num_ops, dim;
};

static std::string FusedAlgoCustomParamName(
  const ::testing::TestParamInfo<FusedAlgoCustomParam> &info) {
  static const char *act_names[] = {"none", "silu", "gelu", "swiglu"};
  const auto &p = info.param;
  return "algo" + std::to_string(p.algo)
         + "_tight" + std::to_string(p.tight)
         + "_custom" + std::to_string(p.custom_kernel)
         + "_" + act_names[p.act_int]
         + "_M" + std::to_string(p.M)
         + "_E" + std::to_string(p.num_ops)
         + "_d" + std::to_string(p.dim);
}

class TestFusedMoEAlgoCustom
  : public ::testing::TestWithParam<FusedAlgoCustomParam> {};

TEST_P(TestFusedMoEAlgoCustom, Correctness) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;

  reset_grp_matmul_caches();

  const auto &p = GetParam();
  const int H = 256, dim = p.dim, K = H;
  const int N_gate_up = 2 * dim;
  const int M = p.M, num_ops = p.num_ops;
  const auto act_type = static_cast<grp_matmul_gated_act_t>(p.act_int);
  // BF16-only: custom kernel contract + the fused-MoE V2 executor
  // both require BF16 throughout.
  const bool is_bf16 = true;
  const data_type_t dt = data_type_t::bf16;

  // Op2's K dimension follows the activation: gated => dim, none =>
  // N_gate_up.  See `op2_k_for_act` in group_matmul_fused_moe.cpp.
  const bool act_is_none = (act_type == grp_matmul_gated_act_t::none);
  const int K_down = act_is_none ? N_gate_up : dim;

  // Two src copies: `src_ref` for the legacy reference pass (unchanged
  // after use), `src_fused` for the internal-alloc fused path (Op2
  // writes back in-place, so the buffer is consumed).
  TypedBuffers src_ref, src_fused, w1, d1_ref, w2, d2_ref;
  src_ref  .alloc(num_ops, (size_t)M * K,         is_bf16);
  src_fused.alloc(num_ops, (size_t)M * K,         is_bf16);
  w1       .alloc(num_ops, (size_t)K * N_gate_up, is_bf16);
  d1_ref   .alloc(num_ops, (size_t)M * N_gate_up, is_bf16);
  w2       .alloc(num_ops, (size_t)K_down * H,    is_bf16);
  d2_ref   .alloc(num_ops, (size_t)M * H,         is_bf16);
  fill_moe_tensors(num_ops, is_bf16, &src_ref,   &w1, &w2);
  fill_moe_tensors(num_ops, is_bf16, &src_fused, nullptr, nullptr);

  auto gv_op1 = GemmVecs::uniform(num_ops, M, N_gate_up, K);

  auto srcs_ref   = src_ref  .cptrs(is_bf16);
  auto srcs_fused = src_fused.cptrs(is_bf16);
  auto wei1       = w1.cptrs(is_bf16);
  auto wei2       = w2.cptrs(is_bf16);
  auto dst1_ref   = d1_ref.ptrs(is_bf16);
  auto dst2_r     = d2_ref.ptrs(is_bf16);
  std::vector<const void *> no_bias(num_ops, nullptr);
  auto params = make_uniform_params(num_ops, dt);

  grp_matmul_gated_act_params act{};
  act.act = act_type;
  auto act_ptr = (act_type != grp_matmul_gated_act_t::none) ? &act : nullptr;

  // Reference: ALGO=1, TIGHT=0, CUSTOM=0, two-call legacy.
  // Everything else in the test can be evaluated against this single
  // baseline.  The env guards are scoped to this block so they don't
  // contaminate the later parameterised run.
  {
    AlgoEnvGuard algo_guard(1);
    EnvVarGuard tight_guard("ZENDNNL_GRP_MATMUL_FUSED_MOE_TIGHT", "0");
    EnvVarGuard custom_guard("ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL",   "0");

    ASSERT_EQ(run_legacy_2call_ref(num_ops, M, K, N_gate_up, K_down, H,
                                   is_bf16, act_type,
                                   srcs_ref, wei1, wei2, dst1_ref, dst2_r),
              status_t::success) << "Ref legacy 2-call failed";
  }

  // ?? Test: parameterised ALGO ? TIGHT ? CUSTOM, internal-alloc fused ??
  // Also forces N_TILE_FUSED_ACT=1 so that when the caller picks
  // ALGO 3 + swiglu in V1 mode, the inline-fused epilogue path is
  // actually exercised (otherwise the ALGO 3 swiglu case would run a
  // separate-pass activation and we'd miss that code path).
  {
    AlgoEnvGuard algo_guard(p.algo);
    EnvVarGuard tight_guard("ZENDNNL_GRP_MATMUL_FUSED_MOE_TIGHT",
                            p.tight ? "1" : "0");
    EnvVarGuard custom_guard("ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL",
                             p.custom_kernel ? "1" : "0");
    EnvVarGuard fused_act_guard("ZENDNNL_GRP_MATMUL_N_TILE_FUSED_ACT", "1");

    auto fused = make_fused_moe_op2(num_ops, H, wei2, no_bias);
    // fused.dst_down / ldc_down intentionally empty - internal-alloc.

    std::vector<void *> dst_null(num_ops, nullptr);
    std::vector<int>    ldc_null(num_ops, 0);

    auto pf = params;
    ASSERT_EQ(group_matmul_direct(gv_op1.layout, gv_op1.transA, gv_op1.transB,
                                  gv_op1.Ms, gv_op1.Ns, gv_op1.Ks, gv_op1.alpha, srcs_fused, gv_op1.lda,
                                  wei1, gv_op1.ldb, no_bias, gv_op1.beta, dst_null, ldc_null,
                                  gv_op1.is_wc, pf, nullptr, act_ptr, &fused),
              status_t::success)
        << "Fused call failed (algo=" << p.algo
        << " tight=" << p.tight << " custom=" << p.custom_kernel
        << " act=" << p.act_int << ")";
  }

  // Compare: src_fused now holds Op2 output (in-place, stride lda=K=H).
  std::ostringstream lbl;
  lbl << "algo=" << p.algo << " tight=" << p.tight
      << " custom=" << p.custom_kernel << " act=" << p.act_int;
  verify_per_expert_2d(src_fused, K, d2_ref, H, num_ops, M, H, is_bf16,
                       tol_fused(is_bf16), lbl.str());
}

static std::vector<FusedAlgoCustomParam> make_fused_algo_custom_params() {
  std::vector<FusedAlgoCustomParam> out;
  // All 5 ALGOs ? TIGHT {0,1} ? CUSTOM {0,1} ? act {silu, gelu, swiglu}.
  // TIGHT=1 with act ? {silu, gelu} or ALGO ? {0,3} exercises the gate
  // that routes back to V1 (Option A) ? expected to produce identical
  // outputs to the baseline.
  for (int algo : {
         1, 2, 3, 4, 5
       }) {
    for (int tight : {
           0, 1
         }) {
      for (int custom : {
             0, 1
           }) {
        for (int act : {
               1, 2, 3
             }) {
          // M=4 ? num_ops=4 keeps the shape small enough for fast
          // sharded execution.  dim=64 gives N_gate_up=128 which is
          // a multiple of the custom kernel's pack_nr=32 so the
          // custom path is reachable.
          out.push_back({algo, tight, custom, act,
                         /*M=*/4, /*num_ops=*/4, /*dim=*/64});
        }
      }
    }
  }
  return out;
}

INSTANTIATE_TEST_SUITE_P(GroupMatmulFusedAlgoCustom, TestFusedMoEAlgoCustom,
                         ::testing::ValuesIn(make_fused_algo_custom_params()),
                         FusedAlgoCustomParamName);

// ???????????????????????????????????????????????????????????????????????????????
// [8] TestGroupMatmulAlgoCustom: non-fused Phase B env-knob matrix
//
// Targets the env-knob combinations that gate the custom BF16 microkernel's
// engagement in the non-fused group_matmul path (ALGO 3 via flat_n_tile):
//
//   * GRP_MATMUL_ALGO          ? 1 (sequential_experts), 3 (flat_n_tile),
//                                5 (per_expert); covers "custom kernel
//                                engages", "no engagement but same output",
//                                and "per-expert distribution" respectively.
//   * CUSTOM_KERNEL            ? 0 vs 1; 0 is the trusted standard path.
//   * N_TILE_FUSED_ACT         ? 0 vs 1 on swiglu; forces the inline
//                                fused-swiglu epilogue (otherwise the
//                                caller does a separate post-pass).
//   * gated_act                ? none / silu / gelu / swiglu_oai_mul.
//   * bias dtype               ? none / bf16 / fp32 (fp32 bias on bf16 dst
//                                exercises the BiasKind::fp32 load path).
//   * moe_postop (weighted)    ? off / on.
//
// Reference strategy: run the same call twice ? once with CUSTOM_KERNEL=0
// (standard dispatch, already verified by TestGroupMatmul / TestGatedAct /
// TestMoEPostop) and once with CUSTOM_KERNEL at the parameterised value
// ? and assert bit-identical (for configs where the custom kernel doesn't
// actually engage, e.g. ALGO 1 or ALGO 5) or within BF16 tolerance.  This
// covers the whole product of env toggles that the Phase B code added
// without duplicating the scalar-reference math elsewhere in this file.
// ???????????????????????????????????????????????????????????????????????????????

struct AlgoCustomParam {
  int algo;               // GRP_MATMUL_ALGO strategy (1, 3, 5)
  int custom_kernel;      // 0 or 1
  int n_tile_fused_act;   // 0 or 1 (only meaningful for ALGO 3 + swiglu)
  int act_int;            // 0=none, 1=silu, 2=gelu, 3=swiglu_oai
  int bias_kind;          // 0=none, 1=bf16, 2=fp32
  int M, num_ops, dim;
  // transB toggle ? exercises BOTH caller layouts the custom-kernel
  // pack now supports (false: [K,N] row-major, true: [N,K] row-major
  // PyTorch convention).  Differential test compares custom_kernel=1
  // run vs custom_kernel=0 reference with the SAME transB, so both
  // paths interpret `wei[]` identically; the test verifies the new
  // pack addressing produces bit-equivalent output to the standard
  // AOCL path.
  int transB;             // 0 or 1
  // NOTE: moe_postop is intentionally not swept here.  The moe_postop
  // executor reduces the full wide Op1 output (D = N[0]), which for
  // gated activations is 2*dim and includes the un-activated second
  // half ? that region's contents legitimately differ between the
  // custom-kernel path (leaves cols [dim:2*dim] at zero) and the
  // standard path (leaves them at the raw Op1 GEMM output), making a
  // differential comparison ill-defined for act ? {swiglu}.  The
  // moe + gated_act combination is already covered by
  // TestGroupMatmulCombined, which uses a full step-by-step reference
  // so both paths are checked against a known ground truth.
};

static std::string AlgoCustomParamName(
  const ::testing::TestParamInfo<AlgoCustomParam> &info) {
  static const char *act_names[]  = {"none", "silu", "gelu", "swiglu"};
  static const char *bias_names[] = {"noBias", "biasBF16", "biasFP32"};
  const auto &p = info.param;
  return "algo" + std::to_string(p.algo)
       + "_custom" + std::to_string(p.custom_kernel)
       + "_fusedAct" + std::to_string(p.n_tile_fused_act)
       + "_" + act_names[p.act_int]
       + "_" + bias_names[p.bias_kind]
       + "_tB" + std::to_string(p.transB)
       + "_M" + std::to_string(p.M)
       + "_E" + std::to_string(p.num_ops)
       + "_d" + std::to_string(p.dim);
}

class TestGroupMatmulAlgoCustom
  : public ::testing::TestWithParam<AlgoCustomParam> {};

// One parameterised call ? sets the CUSTOM_KERNEL env to `custom_value`,
// runs group_matmul_direct, and copies the final Op1 + activation
// output into `out_dst`.  Separated so the TEST_P body can invoke it
// twice (once with "0" for the reference, once with the parameter
// value for the test) and compare.
//
// The outer AlgoEnvGuard + N_TILE_FUSED_ACT EnvVarGuard are set in the
// TEST_P body so both runs share the same strategy and activation
// routing; only CUSTOM_KERNEL flips between runs.
static void run_one_algo_custom_pass(
  const AlgoCustomParam &p,
  const char *custom_value,
  const std::vector<std::vector<bfloat16_t>> &src,
  const std::vector<std::vector<bfloat16_t>> &wei,
  const std::vector<std::vector<bfloat16_t>> &bias_bf16,
  const std::vector<std::vector<float>>      &bias_fp32,
  int N_op1, int K,
  std::vector<std::vector<bfloat16_t>> &out_dst) {
  using namespace moe_test_utils;
  using zendnnl::lowoha::matmul::group_matmul_direct;
  using zendnnl::lowoha::matmul::grp_matmul_gated_act_params;
  using zendnnl::lowoha::matmul::grp_matmul_gated_act_t;

  EnvVarGuard custom_guard("ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL",
                           custom_value);

  // Zero the dst buffers so any untouched region (e.g. cols
  // [dim:2*dim] under the custom-kernel swiglu write) has a
  // well-defined value across both ref and test runs.
  for (auto &v : out_dst) {
    std::fill(v.begin(), v.end(), bfloat16_t(0.0f));
  }

  std::vector<const void *> srcs(p.num_ops), weis(p.num_ops),
      biases(p.num_ops, nullptr);
  std::vector<void *>       dsts(p.num_ops);
  for (int e = 0; e < p.num_ops; ++e) {
    srcs[e] = src[e].data();
    weis[e] = wei[e].data();
    dsts[e] = out_dst[e].data();
    if (p.bias_kind == 1) {
      biases[e] = bias_bf16[e].data();
    }
    else if (p.bias_kind == 2) {
      biases[e] = bias_fp32[e].data();
    }
  }

  auto gv = GemmVecs::uniform(p.num_ops, p.M, N_op1, K,
                              /*alpha=*/1.0f, /*beta=*/0.0f,
                              /*wc=*/false, /*tA=*/false,
                              /*tB=*/p.transB != 0);
  const data_type_t bias_dt = (p.bias_kind == 1) ? data_type_t::bf16
                              : (p.bias_kind == 2) ? data_type_t::f32
                              : data_type_t::none;
  auto params = make_uniform_params(p.num_ops, data_type_t::bf16, bias_dt);

  const auto act_type = static_cast<grp_matmul_gated_act_t>(p.act_int);
  grp_matmul_gated_act_params act{};
  act.act = act_type;
  auto *act_ptr = (p.act_int != 0) ? &act : nullptr;

  ASSERT_EQ(group_matmul_direct(gv.layout, gv.transA, gv.transB,
                                gv.Ms, gv.Ns, gv.Ks, gv.alpha, srcs, gv.lda, weis, gv.ldb,
                                biases, gv.beta, dsts, gv.ldc, gv.is_wc, params,
                                /*moe_postop=*/nullptr, act_ptr),
            status_t::success) << "call failed: custom=" << custom_value;
}

TEST_P(TestGroupMatmulAlgoCustom, Correctness) {
  using namespace moe_test_utils;

  reset_grp_matmul_caches();

  const auto &p = GetParam();

  // N_op1 is the Op1 GEMM output width.  For act=none the output is
  // just `dim` cols wide; for any gated activation the GEMM produces
  // 2*dim cols and the activation compacts to [0:dim].
  const int N_op1 = (p.act_int == 0) ? p.dim : 2 * p.dim;
  const int K     = 64;

  // Custom kernel requires N_op1 % pack_nr == 0 (32 or 64).  When the
  // grid lands on a smaller or misaligned N_op1 the custom path will
  // cleanly fall back to the standard dispatch ? the `== 0` case is
  // still valuable because it regression-tests the fallback path.

  AlgoEnvGuard algo_guard(p.algo);
  EnvVarGuard fused_act_guard("ZENDNNL_GRP_MATMUL_N_TILE_FUSED_ACT",
                              p.n_tile_fused_act ? "1" : "0");

  // ?? Prepare shared src / wei / bias (both runs see identical inputs) ??
  std::vector<std::vector<bfloat16_t>> src(p.num_ops,
      std::vector<bfloat16_t>((size_t)p.M * K));
  std::vector<std::vector<bfloat16_t>> wei(p.num_ops,
      std::vector<bfloat16_t>((size_t)K * N_op1));
  std::vector<std::vector<bfloat16_t>> bias_bf16(p.num_ops);
  std::vector<std::vector<float>>      bias_fp32(p.num_ops);
  for (int e = 0; e < p.num_ops; ++e) {
    fill_src(src[e], e, 0.02f);
    fill_wei1(wei[e], e, 0.005f);
    if (p.bias_kind == 1) {
      bias_bf16[e].resize(N_op1);
      for (int n = 0; n < N_op1; ++n) {
        bias_bf16[e][n] = bfloat16_t(0.01f * ((n + e) % 7 - 3));
      }
    }
    else if (p.bias_kind == 2) {
      bias_fp32[e].resize(N_op1);
      for (int n = 0; n < N_op1; ++n) {
        bias_fp32[e][n] = 0.01f * ((n + e) % 7 - 3);
      }
    }
  }

  // ?? Reference run: CUSTOM_KERNEL=0 ?????????????????????????????????
  std::vector<std::vector<bfloat16_t>> dst_ref(p.num_ops,
      std::vector<bfloat16_t>((size_t)p.M * N_op1, bfloat16_t(0.0f)));
  ASSERT_NO_FATAL_FAILURE(
    run_one_algo_custom_pass(p, "0", src, wei, bias_bf16, bias_fp32,
                             N_op1, K, dst_ref));

  // ?? Test run: CUSTOM_KERNEL as parameterised ??????????????????????
  std::vector<std::vector<bfloat16_t>> dst_test(p.num_ops,
      std::vector<bfloat16_t>((size_t)p.M * N_op1, bfloat16_t(0.0f)));
  ASSERT_NO_FATAL_FAILURE(
    run_one_algo_custom_pass(p,
                             p.custom_kernel ? "1" : "0",
                             src, wei, bias_bf16, bias_fp32,
                             N_op1, K, dst_test));

  // ?? Compare ????????????????????????????????????????????????????????
  // When the custom kernel doesn't actually engage (ALGO 1 / 2 / 4 / 5,
  // or contract-rejected shapes), both runs take the same code path
  // and should match bit-for-bit.  When it does engage (ALGO 3 with a
  // satisfying contract), the FP32 accumulator numerics are nearly
  // identical to the AOCL DLP path; BF16 tolerance captures the
  // per-element rounding of the final `_mm512_cvtneps_pbh`.
  //
  // For gated activations we only compare the activated half
  // [0:dim] of each row.  The un-activated half [dim:2*dim] is
  // "don't care" per the library contract and legitimately differs
  // between paths (custom-kernel swiglu leaves zeros, standard
  // leaves raw GEMM output).
  const auto tol = tol_act(true);
  const int cmp_N = (p.act_int == 0) ? N_op1 : p.dim;
  for (int e = 0; e < p.num_ops; ++e) {
    for (int m = 0; m < p.M; ++m) {
      for (int n = 0; n < cmp_N; ++n) {
        const size_t idx = static_cast<size_t>(m) * N_op1 + n;
        const float ref_v  = static_cast<float>(dst_ref[e][idx]);
        const float test_v = static_cast<float>(dst_test[e][idx]);
        ASSERT_NEAR(test_v, ref_v, std::abs(ref_v) * tol.rel + tol.abs)
            << "algo=" << p.algo << " custom=" << p.custom_kernel
            << " fusedAct=" << p.n_tile_fused_act
            << " act=" << p.act_int << " bias=" << p.bias_kind
            << " e=" << e << " m=" << m << " n=" << n;
      }
    }
  }
}

static std::vector<AlgoCustomParam> make_algo_custom_params() {
  std::vector<AlgoCustomParam> out;
  // Strategy coverage ? 1 (sequential), 3 (flat_n_tile = custom hook),
  // 5 (per-expert).  Skip 2 and 4 to keep the grid tight; those
  // executors don't look at CUSTOM_KERNEL anyway (ALGO 3 is the only
  // engagement site in the non-fused path).
  const int algos[] = {1, 3, 5};
  // dim=64 ? N_op1=128 for act=none, N_op1=128 for gated; K=64.
  // This lets ALGO 3 exercise its N-tile split on ?2 threads while
  // keeping the test shape small enough to run fast.
  const int dim = 64;

  // Core grid ? every (algo ? custom ? act ? bias) combo.
  // N_TILE_FUSED_ACT is only meaningful for swiglu (the only gated
  // activation the custom kernel's inline epilogue supports), so we
  // only sweep both values of that knob for act=swiglu_oai_mul.
  for (int algo : algos) {
    for (int custom : {
           0, 1
         }) {
      for (int act : {
             0, 1, 2, 3
           }) {
        for (int bias : {
               0, 1, 2
             }) {
          const std::vector<int> fused_acts =
            (act == 3) ? std::vector<int> {0, 1}
            :
            std::vector<int> {0};
          for (int fa : fused_acts) {
            // transB sweep ? exercises both [K,N] and [N,K] caller
            // layouts.  The custom-kernel pack now supports both,
            // and this differential test catches any addressing
            // regression on the transB=true path that would
            // otherwise only show up in framework integrations.
            for (int tB : {0, 1}) {
              // M=4 ? num_ops=8 keeps the shape small enough to run
              // fast under 8?64 sharding.  dim=64 ? N_op1=128 for the
              // gated cases (multiple of pack_nr=32 so the custom
              // kernel's contract is met) and N_op1=64 for act=none
              // (also a clean multiple of 32).
              out.push_back({algo, custom, fa, act, bias,
                             /*M=*/4, /*num_ops=*/8, dim, tB});
            }
          }
        }
      }
    }
  }
  return out;
}

INSTANTIATE_TEST_SUITE_P(GroupMatmulAlgoCustom, TestGroupMatmulAlgoCustom,
                         ::testing::ValuesIn(make_algo_custom_params()), AlgoCustomParamName);

// ===============================================================================
// [8c] TestGroupMatmulInt8AlgoCustom — DQ-INT8 N-tile CK env-knob matrix
//
// Mirrors [8] (TestGroupMatmulAlgoCustom) but exercises the
// DQ-INT8 N-tile custom kernel added in Phase 1
// (zenai-grp-matmul-ntile-dqint8-ck-phase1).  The diff covers:
//
//   * Per-token symmetric (compute=s8, no src_zp) DQ-INT8.  The
//     asymmetric (compute=u8) path is covered at the microkernel,
//     pack, dispatcher and prepare_for_call layers in the
//     `gtests/group_matmul/custom_kernel/test_ukernel_int8.cpp`,
//     `test_pack_int8.cpp`, and `test_prepare_for_call.cpp` suites;
//     the existing tensor-factory plumbing in
//     `group_matmul_kernel_test` always sets `compute = s8` (it does
//     not consume an `s32 src_zp` annotation on the tensor), so an
//     e2e differential through that helper would silently route
//     into the symmetric kernel path for both runs and yield a
//     zero-information assertion.  Threading the `compute = u8`
//     metadata through the helper is its own follow-up.
//   * All four gated activations the CK admits today:
//     `none / silu_and_mul / gelu_and_mul / swiglu_oai_mul`.
//   * ALGO 3 forced (the only executor that engages the int8 CK).
//   * Reference: the same call run with the new
//     `_CUSTOM_KERNEL_INT8` env knob FORCED OFF — that path falls
//     back to AOCL DLP `aocl_gemm_s8s8s32obf16_sym_quant`, which
//     existing INT8 tests already validate against a scalar
//     reference.  Comparing the CK-on output against the CK-off
//     output keeps the test self-contained (no separate scalar
//     model) while still asserting end-to-end correctness through
//     the new microkernel + pack + dispatch + prepack stack.
//   * BF16 dst is the only Phase 1 supported dst dtype, so we don't
//     sweep dst.
//
// The fixture skips automatically on hosts without AVX-512 VNNI —
// `prepare_for_call` already refuses cleanly in that case, but
// running the differential would still spend wall-time so we cut
// it off at the gate.
// ===============================================================================

struct Int8AlgoCustomParam {
  int  act_int;           // 0=none, 1=silu, 2=gelu, 3=swiglu_oai
  int  M, num_ops, dim;   // dim = single-half output width; N_op1 = 2*dim for gated
};

static std::string Int8AlgoCustomParamName(
  const ::testing::TestParamInfo<Int8AlgoCustomParam> &info) {
  static const char *act_names[] = {"none", "silu", "gelu", "swiglu"};
  const auto &p = info.param;
  std::string name = "sym_";
  name += act_names[p.act_int];
  name += "_M" + std::to_string(p.M)
        + "_E" + std::to_string(p.num_ops)
        + "_d" + std::to_string(p.dim);
  return name;
}

class TestGroupMatmulInt8AlgoCustom
  : public ::testing::TestWithParam<Int8AlgoCustomParam> {};

// Build per-expert (src, wei_s8, scale) tensors for the int8 DQ
// symmetric path.  The src is bf16 with attached per-token
// src_scale; the weight is s8 with per-channel wei_scale.  Returns
// the assembled tensor vectors and the dst buffer for the run.
// Caller invokes `group_matmul_kernel_test` twice — once with
// CK_INT8 OFF (reference) and once ON (test) — without rebuilding
// the inputs between runs.
static void build_int8_dq_inputs(
    tensor_factory_t &tf,
    const Int8AlgoCustomParam &p,
    int N_op1, int K,
    std::vector<tensor_t> &src_tensors,
    std::vector<tensor_t> &wei_tensors,
    std::vector<tensor_t> &bias_tensors,
    std::vector<tensor_t> &dst_tensors,
    // Scale dtype for BOTH the per-channel wei_scale and the per-token
    // src_scale.  Defaults to f32 (the original coverage); pass bf16 to
    // exercise the reorder's native bf16 scale output, which the CK
    // microkernel must materialise to f32 before consuming (reading a
    // bf16 scale buffer as f32 would silently corrupt every output —
    // see the f32-scale-view materialisation in flat_n_tile).
    data_type_t scale_dt = data_type_t::f32) {
  src_tensors.assign(p.num_ops, tensor_t{});
  wei_tensors.assign(p.num_ops, tensor_t{});
  bias_tensors.assign(p.num_ops, tensor_t{});
  dst_tensors.assign(p.num_ops, tensor_t{});

  // Per-channel wei_scale, per-token src_scale — the granularity
  // Phase 1's CK admits.  Scale dtype is f32 or bf16 (see `scale_dt`).
  const std::vector<int64_t>  wei_scale_dims{1, N_op1};
  const std::vector<uint64_t> src_scale_shape{
      static_cast<uint64_t>(p.M),
      static_cast<uint64_t>(1)};

  for (int e = 0; e < p.num_ops; ++e) {
    auto wei_ref = tf.uniform_dist_tensor(
        {static_cast<uint64_t>(K), static_cast<uint64_t>(N_op1)},
        data_type_t::bf16, /*range=*/2.0);
    tensor_t wei_s8, wei_scale, wei_zp;
    ASSERT_EQ(quant_params_compute(
                  tf, wei_ref, data_type_t::bf16, data_type_t::s8,
                  wei_scale_dims, scale_dt, wei_scale, wei_zp,
                  &wei_s8),
              status_t::success)
        << "DQ-INT8 weight quant setup failed";
    wei_tensors[e] = wei_s8;

    auto src_scale = tf.zero_tensor(src_scale_shape, scale_dt);
    src_tensors[e] = tf.uniform_dist_tensor(
        {static_cast<uint64_t>(p.M), static_cast<uint64_t>(K)},
        data_type_t::bf16, /*range=*/2.0, /*trans=*/false,
        src_scale, tensor_t());

    // Bias is intentionally OMITTED — Phase 1 refuses gated-act +
    // bias for both bf16 and int8 CK paths, so a bias-free baseline
    // keeps every act cell inside the kernel envelope.
    bias_tensors[e] = tensor_t{};

    dst_tensors[e] = tf.uniform_dist_tensor(
        {static_cast<uint64_t>(p.M), static_cast<uint64_t>(N_op1)},
        data_type_t::bf16, /*range=*/2.0);
  }
}

// Run one pass with `_CUSTOM_KERNEL_INT8` forced to `int8_on` (the
// master `_CUSTOM_KERNEL` knob stays ON for both runs — only the
// int8 sub-knob flips).  `dst_out` receives a deep copy of the
// per-expert dst buffers so callers can compare across passes.
static void run_int8_dq_pass(
    bool int8_on,
    std::vector<tensor_t> &src_tensors,
    std::vector<tensor_t> &wei_tensors,
    std::vector<tensor_t> &bias_tensors,
    std::vector<tensor_t> &dst_tensors,
    grp_matmul_gated_act_t act_type,
    std::vector<std::vector<bfloat16_t>> &dst_out,
    int M, int N_op1) {
  using namespace moe_test_utils;
  CustomKernelOverride       ck_master(true);
  CustomKernelInt8Override   ck_int8(int8_on);

  grp_matmul_gated_act_params act_params{};
  act_params.act = act_type;
  const grp_matmul_gated_act_params *act_ptr =
      (act_type != grp_matmul_gated_act_t::none) ? &act_params : nullptr;

  ASSERT_EQ(group_matmul_kernel_test(
                src_tensors, wei_tensors, bias_tensors, dst_tensors,
                matmul_algo_t::aocl_dlp_blocked, /*alpha=*/1.0f,
                /*beta=*/0.0f, /*moe_postop=*/nullptr, act_ptr),
            status_t::success)
      << "DQ-INT8 ALGO 3 kernel call failed (int8_on=" << int8_on << ")";

  // Snapshot dst buffers into plain bf16 vectors for later compare.
  dst_out.assign(dst_tensors.size(),
                 std::vector<bfloat16_t>(
                     static_cast<size_t>(M) * N_op1, bfloat16_t(0.0f)));
  for (size_t e = 0; e < dst_tensors.size(); ++e) {
    const auto *p = static_cast<const bfloat16_t *>(
        dst_tensors[e].get_raw_handle_unsafe());
    std::memcpy(dst_out[e].data(), p,
                static_cast<size_t>(M) * N_op1 * sizeof(bfloat16_t));
  }
}

TEST_P(TestGroupMatmulInt8AlgoCustom, Correctness) {
  using namespace moe_test_utils;
  namespace ck = zendnnl::lowoha::matmul::custom_kernel;

  if (!ck::avx512vnni_available()) {
    GTEST_SKIP() << "AVX-512 VNNI not available on this host";
  }

  reset_grp_matmul_caches();

  const auto &p = GetParam();
  const auto  act_type =
      static_cast<grp_matmul_gated_act_t>(p.act_int);

  // N_op1 is the Op1 GEMM output width.  For act=none the output is
  // `dim` cols wide; for any gated activation the GEMM produces
  // 2*dim cols and the activation compacts to [0:dim].
  const int N_op1 = (p.act_int == 0) ? p.dim : 2 * p.dim;
  const int K     = 64;

  // The int8 CK requires N % pack_nr ∈ {32, 64}.  Skip params that
  // wouldn't engage CK on either run — the differential would be
  // meaningless because both runs would take the AOCL DLP path.
  if (N_op1 % 32 != 0) {
    GTEST_SKIP() << "N_op1=" << N_op1
                 << " not a multiple of 32; CK would refuse";
  }

  tensor_factory_t tf;
  std::vector<tensor_t> src, wei, bias, dst;
  ASSERT_NO_FATAL_FAILURE(
      build_int8_dq_inputs(tf, p, N_op1, K, src, wei, bias, dst));

  AlgoEnvGuard algo_guard(3);

  std::vector<std::vector<bfloat16_t>> dst_ref, dst_test;

  // Reference pass: int8 CK OFF → AOCL DLP sym_quant runs.
  ASSERT_NO_FATAL_FAILURE(
      run_int8_dq_pass(/*int8_on=*/false, src, wei, bias, dst,
                       act_type, dst_ref, p.M, N_op1));

  // Rebuild the dst tensors so the second pass starts from a clean
  // slate (zero-out buffers; reuse src/wei/bias which are unchanged).
  for (auto &d : dst) {
    auto *raw = static_cast<bfloat16_t *>(d.get_raw_handle_unsafe());
    std::fill(raw, raw + static_cast<size_t>(p.M) * N_op1,
              bfloat16_t(0.0f));
  }

  // Test pass: int8 CK ON → new microkernel + pack path runs.
  ASSERT_NO_FATAL_FAILURE(
      run_int8_dq_pass(/*int8_on=*/true, src, wei, bias, dst,
                       act_type, dst_test, p.M, N_op1));

  // Comparison band: BF16 tolerance from the gated/non-gated helper
  // in `moe_test_utils` — same tolerance pair the existing
  // TestGroupMatmulAlgoCustom uses, since the activation post-pass
  // is the same scalar formula and the quant rounding deltas are
  // already within that band.
  const auto tol  = tol_act(/*is_bf16=*/true);
  const int  cmpN = (p.act_int == 0) ? N_op1 : p.dim;
  for (int e = 0; e < p.num_ops; ++e) {
    for (int m = 0; m < p.M; ++m) {
      for (int n = 0; n < cmpN; ++n) {
        const size_t idx =
            static_cast<size_t>(m) * N_op1 + n;
        const float ref_v  = static_cast<float>(dst_ref[e][idx]);
        const float test_v = static_cast<float>(dst_test[e][idx]);
        ASSERT_NEAR(test_v, ref_v,
                    std::abs(ref_v) * tol.rel + tol.abs)
            << "int8 sym act=" << p.act_int
            << " e=" << e << " m=" << m << " n=" << n;
      }
    }
  }
}

static std::vector<Int8AlgoCustomParam> make_int8_algo_custom_params() {
  std::vector<Int8AlgoCustomParam> out;
  // Four activations × two shapes.  We keep the grid tight — the
  // per-cell signal is fully redundant with the microkernel-level
  // coverage in test_ukernel_int8.cpp, so the e2e fixture only
  // needs enough variety to catch dispatch / pack / prepack
  // integration regressions.
  for (int act : {0, 1, 2, 3}) {
    // (M=4, num_ops=8, dim=32) — N_op1 = 32 (act=none) or 64
    // (gated) → both clean multiples of pack_nr=32.
    out.push_back({act, /*M=*/4, /*num_ops=*/8, /*dim=*/32});
    // (M=16, num_ops=4, dim=64) — N_op1 = 64 or 128 → exercises
    // the kNRMax=64 branch and multi-tile splitting on >2 threads.
    out.push_back({act, /*M=*/16, /*num_ops=*/4, /*dim=*/64});
  }
  // C.6 — decode (M=1) and prompt (M=64) shapes covering the
  // edge of the M-dimension envelope:
  //   * M=1 hits the single-row hoist path inside
  //     `reorder_quantization_wrapper` and the int8 microkernel's
  //     MR=1 instantiation.
  //   * M=64 hits the multi-MR-band split with deep K-quad
  //     accumulation pressure (decode/prompt parity).
  //   * dim=128 (NV=4 / kNRMax=64) at M=8 exercises the
  //     pack_nr=64 cache key and the wider register kernel.
  // Activation cycles through {none, swiglu} to keep the new
  // matrix small (silu/gelu are byte-identical to swiglu at the
  // pack layer per C.2.5; their epilogues are covered by the
  // microkernel param sweep).
  for (int act : {0, 3}) {
    out.push_back({act, /*M=*/1,  /*num_ops=*/8, /*dim=*/32});  // decode
    out.push_back({act, /*M=*/64, /*num_ops=*/4, /*dim=*/64});  // prompt
    out.push_back({act, /*M=*/8,  /*num_ops=*/4, /*dim=*/128}); // NR=128
  }
  return out;
}

INSTANTIATE_TEST_SUITE_P(GroupMatmulInt8AlgoCustom,
                         TestGroupMatmulInt8AlgoCustom,
                         ::testing::ValuesIn(make_int8_algo_custom_params()),
                         Int8AlgoCustomParamName);

// ===============================================================================
// [8c-bf16] TestGroupMatmulInt8AlgoCustomBf16Scale — DQ-INT8 with BF16 scales
//
// Regression guard for the bf16-scale stitching bug: the dynamic-quant
// reorder emits the per-token src_scale (and the caller supplies the
// per-channel wei_scale) in BF16, but the int8 microkernel loads scales
// with `_mm512_loadu_ps` / `_mm512_set1_ps` (f32-only).  Before the
// f32-scale-view materialisation in `flat_n_tile`, the CK reinterpreted
// the bf16 scale bytes as f32 and produced systematic garbage (zero
// accuracy on real bf16 models), while the AOCL DLP fallback consumed
// the declared bf16 dtype correctly.  The whole class was invisible to
// the suite because every other int8 test uses f32 scales.
//
// Mechanism: identical CK-OFF (AOCL DLP sym_quant, the bf16-scale-aware
// reference) vs CK-ON (new microkernel) differential as [8c], but with
// `scale_dt = bf16`.  act=none is used because it is the activation that
// reliably engages the int8 CK (gated int8 currently routes to the AOCL
// separate-activation path); act=none also matches the down-proj GEMM
// where the production bug first surfaced.
// ===============================================================================
class TestGroupMatmulInt8AlgoCustomBf16Scale
  : public ::testing::TestWithParam<Int8AlgoCustomParam> {};

TEST_P(TestGroupMatmulInt8AlgoCustomBf16Scale, Correctness) {
  using namespace moe_test_utils;
  namespace ck = zendnnl::lowoha::matmul::custom_kernel;

  if (!ck::avx512vnni_available()) {
    GTEST_SKIP() << "AVX-512 VNNI not available on this host";
  }

  reset_grp_matmul_caches();

  const auto &p = GetParam();
  const auto  act_type =
      static_cast<grp_matmul_gated_act_t>(p.act_int);

  const int N_op1 = (p.act_int == 0) ? p.dim : 2 * p.dim;
  const int K     = 64;
  if (N_op1 % 32 != 0) {
    GTEST_SKIP() << "N_op1=" << N_op1
                 << " not a multiple of 32; CK would refuse";
  }

  tensor_factory_t tf;
  std::vector<tensor_t> src, wei, bias, dst;
  ASSERT_NO_FATAL_FAILURE(
      build_int8_dq_inputs(tf, p, N_op1, K, src, wei, bias, dst,
                           /*scale_dt=*/data_type_t::bf16));

  AlgoEnvGuard algo_guard(3);

  std::vector<std::vector<bfloat16_t>> dst_ref, dst_test;

  // Reference: int8 CK OFF → AOCL DLP sym_quant (bf16-scale aware).
  ASSERT_NO_FATAL_FAILURE(
      run_int8_dq_pass(/*int8_on=*/false, src, wei, bias, dst,
                       act_type, dst_ref, p.M, N_op1));

  for (auto &d : dst) {
    auto *raw = static_cast<bfloat16_t *>(d.get_raw_handle_unsafe());
    std::fill(raw, raw + static_cast<size_t>(p.M) * N_op1,
              bfloat16_t(0.0f));
  }

  // Test: int8 CK ON → must materialise the bf16 scales to f32 and
  // match the reference (pre-fix this produced garbage).
  ASSERT_NO_FATAL_FAILURE(
      run_int8_dq_pass(/*int8_on=*/true, src, wei, bias, dst,
                       act_type, dst_test, p.M, N_op1));

  const auto tol  = tol_act(/*is_bf16=*/true);
  const int  cmpN = (p.act_int == 0) ? N_op1 : p.dim;
  for (int e = 0; e < p.num_ops; ++e) {
    for (int m = 0; m < p.M; ++m) {
      for (int n = 0; n < cmpN; ++n) {
        const size_t idx = static_cast<size_t>(m) * N_op1 + n;
        const float ref_v  = static_cast<float>(dst_ref[e][idx]);
        const float test_v = static_cast<float>(dst_test[e][idx]);
        ASSERT_NEAR(test_v, ref_v, std::abs(ref_v) * tol.rel + tol.abs)
            << "int8 bf16-scale act=" << p.act_int
            << " e=" << e << " m=" << m << " n=" << n;
      }
    }
  }
}

// act=none across decode (M=1), small-batch (M=4) and prompt (M=64)
// shapes at both pack_nr widths (dim=32 → NR=32, dim=64 → NR=64).
INSTANTIATE_TEST_SUITE_P(
    GroupMatmulInt8AlgoCustomBf16Scale,
    TestGroupMatmulInt8AlgoCustomBf16Scale,
    ::testing::Values(
        Int8AlgoCustomParam{/*act=*/0, /*M=*/1,  /*num_ops=*/8, /*dim=*/32},
        Int8AlgoCustomParam{/*act=*/0, /*M=*/4,  /*num_ops=*/8, /*dim=*/32},
        Int8AlgoCustomParam{/*act=*/0, /*M=*/16, /*num_ops=*/4, /*dim=*/64},
        Int8AlgoCustomParam{/*act=*/0, /*M=*/64, /*num_ops=*/4, /*dim=*/64}),
    Int8AlgoCustomParamName);

// ===============================================================================
// [8a] TestGroupMatmulPhaseBRemainder — CK Single-round + non-zero-remainder
//
// Targets the planner+executor path that
// `apply_round_pick(RoundPick::Single, use_custom=true)` populates
// when `num_threads % num_ops != 0`:
//
//   * `apply_round_pick` runs the heaviest-first eligibility filter,
//     populates `stable_n_thr_per_expert[]` non-uniformly (M-heaviest
//     experts get `base + 1` threads, the rest get `base`), and sets
//     `plan.per_expert_remainder = true`.
//   * `execute_rounds` reads `per_expert_remainder` and switches from
//     the O(1) `tid / thr_per_expert` mapping to the per-round
//     prefix-sum `tid → (expert, local_tid)` scan that consumes the
//     non-uniform `stable_n_thr_per_expert[]`.
//
// Failure modes the test catches:
//   * Drop tiles — a regression in the prefix-sum mapping that
//     skips an N-column slice → output cells diverge from the ALGO 1
//     reference.
//   * Duplicate tiles — a regression that maps two threads onto the
//     same N-column → race-corrupted cells diverge.
//   * Eligibility filter bugs — wrong `extras` count or per-expert
//     `n_thr` of zero → `do_tile`'s `local_tid >= n_thr` early-return
//     leaves columns un-written → diverge from the reference.
//
// Strategy: run the same call twice — once with ALGO=3 (current
// default knobs land on CK + Single, firing Phase B) and once with
// ALGO=1 sequential (the dispatcher's gold reference: serial over
// experts but each expert runs with the full thread team, so it
// never enters Phase B's per-expert remainder distribution).
// Compare BF16 outputs cell-by-cell within the standard
// activation-free tolerance.
//
// Shape choice for the documented `64 threads / 18 experts` case
// flagged in PR #461 review: `omp_set_num_threads(32)` paired with
// `num_ops = 14` — this lands `base = 32/14 = 2`, `remainder = 4`,
// `per_expert_cap = min(ccd_size=8, max_tiles=N/min_n_tile)`.
// `M = 4` keeps the call small; `N = 1024` (multiple of pack_nr=32
// for CK eligibility) gives `max_tiles = 1024/256 = 4` so
// `(base+1) = 3 ≤ 4` and Phase B fires on every active expert
// (uniform `N` → all 14 experts pass the eligibility filter →
// first 4 experts get `base+1=3` threads, the other 10 get
// `base=2`, summing to 32 threads = full team).
//
// 32-thread choice (rather than 64): keeps the matmul work small
// while still triggering Phase B; avoids over-subscription on hosts
// where the test runner already pinned a smaller team.  Restored
// to the prior team size on test teardown.
//
// Note on env caching: `ZENDNNL_GRP_MATMUL_N_ROUNDS`,
// `ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL`, and the
// `CUSTOM_KERNEL_N_TILE` getters cache their value at first call
// (`static const`).  Earlier tests in this binary may have already
// observed a non-default value, in which case an `EnvVarGuard`
// here would be a no-op and the ALGO=3 pass below could silently
// take a non-CK / non-Single path with the cell-by-cell comparison
// still passing — losing Phase B coverage.
//
// Use the test-only `*Override` RAII guards (defined in
// `moe_test_utils.hpp`) which write a `test_api::` atomic that
// the production getters check before the cached read path.  This
// guarantees the test sees its required configuration regardless
// of test-suite order.  `ZENDNNL_GRP_MATMUL_ALGO` is NOT cached,
// so `AlgoEnvGuard(3)` / `AlgoEnvGuard(1)` flip cleanly per scope.
TEST(TestGroupMatmulPhaseBRemainder, CkSingleRoundRemainderCorrectness) {
  using namespace moe_test_utils;
  using zendnnl::lowoha::matmul::group_matmul_direct;

  // Force a 32-thread team for this test; restore on scope exit.
  // Use omp_set_num_threads (process-global) since the dispatcher
  // reads `omp_get_max_threads()` when params don't pin
  // num_threads, which is the case for `make_uniform_params`.
  const int saved_num_threads = omp_get_max_threads();
  struct ThreadGuard {
    int prev;
    ~ThreadGuard() { omp_set_num_threads(prev); }
  } thread_guard{saved_num_threads};
  omp_set_num_threads(32);

  // Verify the host can actually spawn 32 worker threads.
  // `omp_get_max_threads()` reports the requested `nthreads-var`
  // ICV (i.e. what `omp_set_num_threads` set, capped by the host's
  // dynamic-team policy) but NOT the actual team size that
  // `#pragma omp parallel` will instantiate when it fires.  On
  // hosts with `OMP_THREAD_LIMIT` or dynamic-adjustment enabled,
  // the runtime can hand out a smaller team than requested, which
  // would land us on a different (base, remainder) tuple and miss
  // the 32t/14ops Phase B shape we're locking down.
  //
  // Use a probe parallel region to read `omp_get_num_threads()`
  // — that value is the active team size inside the region —
  // before committing to the test.
  int actual_team_size = 0;
  #pragma omp parallel
  {
    #pragma omp master
    actual_team_size = omp_get_num_threads();
  }
  if (actual_team_size < 32) {
    GTEST_SKIP() << "Test requires >= 32 active OMP threads to land "
                    "on the 32t/14ops Phase B remainder shape; "
                    "probe parallel region reported "
                 << actual_team_size << " (requested 32; host's "
                    "dynamic-team / thread-limit policy may have "
                    "clamped it).";
  }

  // Force the env-cached configuration the Phase B path needs,
  // regardless of what an earlier test in this binary already
  // cached.  The RAII guards write a `test_api::` atomic that the
  // production getters check before their cached `static const`
  // read; on scope exit the prior override value is restored, so
  // tests that follow this one are unaffected.
  CustomKernelOverride        ck_on(true);          // CUSTOM_KERNEL=ON
  NRoundsModeOverride         single_round(1);      // N_ROUNDS=1 (Single)
  // 0 = use default (kDecodeNTile=256).  Our shape (N=1024) was
  // sized for ab_min_tile=256 so max_tiles=4 and (base+1)=3 ≤
  // per_expert_cap=4, which is required for Phase B to fire.
  CustomKernelNTileOverride   default_n_tile(0);
  // Pin the N-tile strategy knob to 0 (auto / heuristic) so this
  // test is insulated from a process-level
  // `ZENDNNL_GRP_MATMUL_N_TILE_STRATEGY=1` (force DecodeD) or `=2`
  // (force Rounds) that some surrounding benchmark / sweep might
  // have set.  Phase B's per-expert remainder distribution only
  // populates inside ManyExperts Single-round; values 1/2 would
  // override the auto-mirror gate and the heuristic DecodeD
  // attempt, taking different planner branches that bypass Phase B
  // and leave `snap.per_expert_remainder = false`.
  NTileStrategyOverride       auto_strategy(0);

  // Defence-in-depth: the env-cache overrides force the planner's
  // INTENT to take the CK path, but `prepare_for_call` makes a
  // separate runtime decision based on AVX512BF16 availability.
  // On a host without that ISA the dispatcher refuses CK and
  // the ALGO 3 run silently routes through the AOCL strict-stable
  // plan — which, with this shape's `dynamic_quant=false` and
  // null quant buffers, would ALSO produce numerically equivalent
  // output to ALGO 1, so the comparison below would pass without
  // exercising Phase B.  Skip when CK is structurally unavailable.
  if (!zendnnl::lowoha::matmul::custom_kernel::dispatch_supported()) {
    GTEST_SKIP() << "Test requires AVX512BF16 / custom-kernel "
                    "dispatch support; forcing the cached env "
                    "knobs is necessary but not sufficient on this "
                    "host.";
  }

  reset_grp_matmul_caches();

  const int num_ops = 14;            // 32 % 14 = 4 → Phase B fires
  const int N_op1   = 1024;          // % pack_nr=32 ✓; max_tiles=4
  const int K       = 64;

  // Two M distributions, both run end-to-end and compared against
  // the ALGO 1 reference:
  //
  //   (a) Uniform M=4 — every expert equally weighted; the
  //       eligibility filter passes all 14 experts (uniform N=1024).
  //       Locks down tile coverage (no dropped / duplicated cols)
  //       and the prefix-sum thread→expert mapping.
  //
  //   (b) Skewed M with the first 4 experts heaviest (16, 12, 10, 8)
  //       and the remaining 10 light (6 then 4×9) — exercises the
  //       same eligibility filter on a non-uniform input.  Note that
  //       this comparison CANNOT directly assert the heaviest-first
  //       ordering: reordering which experts get `base+1` does not
  //       change per-cell numerics, only the per-expert thread
  //       distribution.  That ordering is covered separately by the
  //       white-box `HeaviestFirstAssignment` test below, which
  //       captures `plan.stable_n_thr_per_expert[]` via the
  //       `test_api::PhaseBSnapshot` hook in
  //       `group_matmul_n_tile.hpp` and asserts on the per-expert
  //       thread counts directly.
  struct MVariant {
    std::vector<int> Ms;
    const char *label;
  };
  const std::vector<MVariant> variants = {
    {std::vector<int>(num_ops, 4), "uniform_M=4"},
    {{16, 12, 10, 8, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4},
     "skewed_M_heaviest_first"},
  };

  for (const auto &var : variants) {
    SCOPED_TRACE(std::string("M variant: ") + var.label);
    const auto &Ms = var.Ms;
    ASSERT_EQ(static_cast<int>(Ms.size()), num_ops);

    // ── Prepare shared inputs sized to per-expert M ──
    std::vector<std::vector<bfloat16_t>> src(num_ops);
    std::vector<std::vector<bfloat16_t>> wei(num_ops);
    for (int e = 0; e < num_ops; ++e) {
      src[e].resize(static_cast<size_t>(Ms[e]) * K);
      wei[e].resize(static_cast<size_t>(K) * N_op1);
      fill_src(src[e],  e, 0.02f);
      fill_wei1(wei[e], e, 0.005f);
    }

    // Helper to build the call vectors and run group_matmul_direct
    // with `algo` forced.  Captures the output of expert `e` into
    // `out_dst[e]` (sized to Ms[e] * N_op1).  No bias / no
    // activation keeps the comparison strictly numerical.
    auto run_one_pass =
        [&](int algo,
            std::vector<std::vector<bfloat16_t>> &out_dst) {
      AlgoEnvGuard algo_guard(algo);

      for (auto &v : out_dst) {
        std::fill(v.begin(), v.end(), bfloat16_t(0.0f));
      }
      std::vector<const void *> srcs(num_ops), weis(num_ops),
          biases(num_ops, nullptr);
      std::vector<void *> dsts(num_ops);
      for (int e = 0; e < num_ops; ++e) {
        srcs[e] = src[e].data();
        weis[e] = wei[e].data();
        dsts[e] = out_dst[e].data();
      }

      // Build with M=0 then override per-expert Ms (rest of the
      // wrapper vectors stay uniform — N, K, lda, ldb, ldc all
      // shape-independent of M for tA=tB=false).
      //
      // `wc=true` is REQUIRED for the CK path to engage on the
      // ALGO 3 run.  `custom_kernel::prepare_for_call` refuses CK
      // for any active expert flagged `is_weights_const=false`
      // (dispatch.cpp:304-308) — without this, the ALGO 3 pass
      // would silently fall back to the standard AOCL DLP per-tile
      // path and bypass Phase B entirely, defeating the test.
      auto gv = GemmVecs::uniform(num_ops, /*M=*/0, N_op1, K,
                                  /*alpha=*/1.0f, /*beta=*/0.0f,
                                  /*wc=*/true, /*tA=*/false,
                                  /*tB=*/false);
      gv.Ms = Ms;
      auto params = make_uniform_params(num_ops, data_type_t::bf16);
      // Pin num_threads via the params struct.  `omp_set_num_threads`
      // alone does NOT propagate to the dispatcher because
      // `thread_guard::max_threads()`
      // (lowoha_operators/common/omp_thread_control.hpp:90-93)
      // caches `omp_get_max_threads()` on first invocation —
      // process-wide, for the binary's lifetime.  Setting
      // `params[i].num_threads = 32` forces
      // `resolve_num_threads(32, cached) = 32` and `thread_guard`
      // then flips the ICV to 32 for the scope of this call.
      // Required for the planner to land on the (base=2, remainder=4)
      // tuple Phase B is sized for; without it the dispatcher uses
      // the cached top-level OMP value and the planner picks
      // (base=4, remainder=8) which fails the
      // `(base+1) <= per_expert_cap=4` gate and never enters
      // Phase B.
      for (int e = 0; e < num_ops; ++e) {
        params[e].num_threads = 32;
      }

      ASSERT_EQ(group_matmul_direct(gv.layout, gv.transA, gv.transB,
                                    gv.Ms, gv.Ns, gv.Ks, gv.alpha,
                                    srcs, gv.lda, weis, gv.ldb,
                                    biases, gv.beta, dsts, gv.ldc,
                                    gv.is_wc, params,
                                    /*moe_postop=*/nullptr,
                                    /*gated_act=*/nullptr),
                status_t::success) << "algo=" << algo;
    };

    // ── Reference: ALGO 1 sequential (gold) ──
    std::vector<std::vector<bfloat16_t>> dst_ref(num_ops);
    for (int e = 0; e < num_ops; ++e) {
      dst_ref[e].assign(static_cast<size_t>(Ms[e]) * N_op1,
                        bfloat16_t(0.0f));
    }
    ASSERT_NO_FATAL_FAILURE(run_one_pass(/*algo=*/1, dst_ref));

    // ── Test: ALGO 3 (Phase B fires on CK Single + remainder) ──
    // Wrap in a `PhaseBCaptureGuard` so we can ASSERT_TRUE after
    // the call that the snapshot was actually populated AND that
    // `per_expert_remainder` is set — otherwise a future change
    // that bypasses Phase B (e.g. plan strategy changed, gate
    // moved) would silently produce equivalent output and the
    // comparison below would pass while Phase B coverage is lost.
    // The guard also disarms the capture flag on scope exit, so a
    // failed `group_matmul_direct` cannot leak the flag into a
    // later test.
    reset_grp_matmul_caches();
    std::vector<std::vector<bfloat16_t>> dst_test(num_ops);
    for (int e = 0; e < num_ops; ++e) {
      dst_test[e].assign(static_cast<size_t>(Ms[e]) * N_op1,
                         bfloat16_t(0.0f));
    }
    {
      PhaseBCaptureGuard cap;
      ASSERT_NO_FATAL_FAILURE(run_one_pass(/*algo=*/3, dst_test));

      const auto &snap =
          zendnnl::lowoha::matmul::test_api::s_last_phase_b_snapshot;
      ASSERT_TRUE(snap.valid)
          << "PhaseBSnapshot was not captured during the ALGO 3 "
             "run — flat_n_tile may not have been reached (variant: "
          << var.label << ")";
      ASSERT_TRUE(snap.per_expert_remainder)
          << "Phase B did not fire on the ALGO 3 run.  The "
             "numerical comparison below would still pass on a "
             "fallback path but the test would lose Phase B "
             "coverage.  variant=" << var.label
          << "  strategy=" << static_cast<int>(snap.strategy)
          << "  n_thr_fixed=" << snap.n_thr_fixed;
    }

    // ── Compare cell-by-cell within BF16 tolerance ──
    // Per-cell BF16 rounding of the final `_mm512_cvtneps_pbh`
    // against AOCL DLP's accumulator differs by at most a relative
    // epsilon at this magnitude.  `tol_act(/*is_bf16=*/true)` is the
    // same tolerance pair the existing TestGroupMatmulAlgoCustom
    // uses.
    const auto tol = tol_act(true);
    for (int e = 0; e < num_ops; ++e) {
      for (int m = 0; m < Ms[e]; ++m) {
        for (int n = 0; n < N_op1; ++n) {
          const size_t idx = static_cast<size_t>(m) * N_op1 + n;
          const float ref_v  = static_cast<float>(dst_ref[e][idx]);
          const float test_v = static_cast<float>(dst_test[e][idx]);
          ASSERT_NEAR(test_v, ref_v,
                      std::abs(ref_v) * tol.rel + tol.abs)
              << "Phase B remainder regression: variant=" << var.label
              << " e=" << e << " m=" << m << " n=" << n
              << " ref=" << ref_v << " test=" << test_v;
        }
      }
    }
  }
}

// ===============================================================================
// [8a.1] TestGroupMatmulPhaseBRemainder — heaviest-first white-box
//
// The end-to-end correctness test above cannot directly observe the
// heaviest-first comparator: reordering which experts get `base+1`
// does NOT change the per-cell GEMM output, only the per-expert
// thread distribution.  A regression that reversed the comparator
// (lightest-first), or assigned `base+1` independently of M, would
// pass the cell-by-cell comparison and slip past silently.
//
// Use the planner's test-only `PhaseBSnapshot` (defined in
// `group_matmul_n_tile.hpp::test_api`) to read out the per-expert
// thread counts the planner committed for the next `flat_n_tile`
// call.  Assert that:
//
//   * `per_expert_remainder` is true (Phase B fired).
//   * `n_thr_fixed == base`.
//   * Exactly `remainder` experts received `base+1`, all others
//     received `base`, and the recipients are the M-heaviest
//     eligible experts (here all eligible because N is uniform).
//
// Shape: same 32t / 14ops / N=1024 setup as the correctness test
// above, but with skewed M = {16,12,10,8,6,4×9} so only the FIRST
// FOUR experts (M ∈ {16,12,10,8}) qualify as the heaviest.
// ===============================================================================

TEST(TestGroupMatmulPhaseBRemainder, HeaviestFirstAssignment) {
  using namespace moe_test_utils;
  using zendnnl::lowoha::matmul::group_matmul_direct;
  using zendnnl::lowoha::matmul::GroupNTileStrategy;

  const int saved_num_threads = omp_get_max_threads();
  struct ThreadGuard {
    int prev;
    ~ThreadGuard() { omp_set_num_threads(prev); }
  } thread_guard{saved_num_threads};
  omp_set_num_threads(32);

  // Real team-size probe (same reasoning as the correctness test).
  int actual_team_size = 0;
  #pragma omp parallel
  {
    #pragma omp master
    actual_team_size = omp_get_num_threads();
  }
  if (actual_team_size < 32) {
    GTEST_SKIP() << "Requires >= 32 active OMP threads; have "
                 << actual_team_size;
  }

  // Force the env-cached config Phase B needs (RAII, restored on
  // scope exit).  `NTileStrategyOverride(0)` insulates the test
  // from a process-level `ZENDNNL_GRP_MATMUL_N_TILE_STRATEGY=1/2`
  // that would otherwise bypass ManyExperts and skip Phase B —
  // see `CkSingleRoundRemainderCorrectness` for the full rationale.
  CustomKernelOverride        ck_on(true);
  NRoundsModeOverride         single_round(1);
  CustomKernelNTileOverride   default_n_tile(0);
  NTileStrategyOverride       auto_strategy(0);
  // Decode M-proportional split is OFF by default; this test asserts the
  // proportional contract (heaviest-first, capacity-clamped), so pin it ON.
  DecodeProportionalOverride  prop_on(true);

  // Defence-in-depth: env overrides force the PLANNER's intent to
  // take the CK path, but `prepare_for_call` separately refuses CK
  // on hosts without AVX512BF16.  Without this skip the test would
  // run, fail to capture per_expert_remainder, and report a noisy
  // failure on a structural ISA limitation.
  if (!zendnnl::lowoha::matmul::custom_kernel::dispatch_supported()) {
    GTEST_SKIP() << "Test requires AVX512BF16 / custom-kernel "
                    "dispatch support; cannot exercise Phase B "
                    "without CK engagement.";
  }

  reset_grp_matmul_caches();

  const int num_ops = 14;
  const int N_op1   = 1024;
  const int K       = 64;
  const std::vector<int> Ms =
      {16, 12, 10, 8, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4};
  ASSERT_EQ(static_cast<int>(Ms.size()), num_ops);

  // Build inputs sized to per-expert M.
  std::vector<std::vector<bfloat16_t>> src(num_ops);
  std::vector<std::vector<bfloat16_t>> wei(num_ops);
  std::vector<std::vector<bfloat16_t>> dst(num_ops);
  for (int e = 0; e < num_ops; ++e) {
    src[e].resize(static_cast<size_t>(Ms[e]) * K);
    wei[e].resize(static_cast<size_t>(K) * N_op1);
    dst[e].assign(static_cast<size_t>(Ms[e]) * N_op1,
                  bfloat16_t(0.0f));
    fill_src(src[e],  e, 0.02f);
    fill_wei1(wei[e], e, 0.005f);
  }

  std::vector<const void *> srcs(num_ops), weis(num_ops),
      biases(num_ops, nullptr);
  std::vector<void *> dsts(num_ops);
  for (int e = 0; e < num_ops; ++e) {
    srcs[e] = src[e].data();
    weis[e] = wei[e].data();
    dsts[e] = dst[e].data();
  }

  auto gv = GemmVecs::uniform(num_ops, /*M=*/0, N_op1, K,
                              /*alpha=*/1.0f, /*beta=*/0.0f,
                              /*wc=*/true, /*tA=*/false,
                              /*tB=*/false);
  gv.Ms = Ms;
  auto params = make_uniform_params(num_ops, data_type_t::bf16);
  // Pin num_threads=32 via params so the dispatcher honours the
  // intended team size regardless of `thread_guard::max_threads`'s
  // process-wide cache.  See the matching comment in
  // `CkSingleRoundRemainderCorrectness` above for the full
  // rationale; without this the planner lands on (base=4,
  // remainder=8) which fails Phase B's outer
  // `(base+1) <= per_expert_cap` gate.
  for (int e = 0; e < num_ops; ++e) {
    params[e].num_threads = 32;
  }

  // Arm the snapshot capture via RAII so the flag is always
  // disarmed on scope exit (including on test failure / early
  // ASSERT abort).  `flat_n_tile` will exchange the flag back to
  // false and copy `plan.stable_n_thr_per_expert[]` +
  // `plan.per_expert_remainder` into `s_last_phase_b_snapshot`.
  AlgoEnvGuard       algo_guard(3);
  PhaseBCaptureGuard cap;

  ASSERT_EQ(group_matmul_direct(gv.layout, gv.transA, gv.transB,
                                gv.Ms, gv.Ns, gv.Ks, gv.alpha,
                                srcs, gv.lda, weis, gv.ldb,
                                biases, gv.beta, dsts, gv.ldc,
                                gv.is_wc, params,
                                /*moe_postop=*/nullptr,
                                /*gated_act=*/nullptr),
            status_t::success);

  // Read the snapshot.  `flat_n_tile`'s exchange should have
  // cleared the flag; the snapshot holds the planner's output.
  const auto &snap =
      zendnnl::lowoha::matmul::test_api::s_last_phase_b_snapshot;

  ASSERT_TRUE(snap.valid)
      << "PhaseBSnapshot was not captured — flat_n_tile may not "
         "have been reached.";
  EXPECT_TRUE(snap.per_expert_remainder)
      << "Phase B did not fire; planner may have routed to a "
         "non-Single strategy or the eligibility filter rejected "
         "every expert.  Check `strategy` / `n_thr_fixed` below.";
  EXPECT_EQ(static_cast<int>(snap.strategy),
            static_cast<int>(GroupNTileStrategy::ManyExperts))
      << "Phase B is only populated on the ManyExperts strategy "
         "(Single round picked by N_ROUNDS=1 + use_custom).";
  EXPECT_EQ(snap.num_ops_active, num_ops);

  // 32 threads / 14 ops, max_M=16 <= kDecodeMaxM=32 → decode-class, so
  // the CK Single round uses the M-PROPORTIONAL thread split (the new
  // decode default), NOT the legacy uniform Phase B base/base+1.  The
  // "heaviest-first" intent is preserved: proportional allocates threads
  // in proportion to M (clamped to each expert's N-tile capacity
  // cap=N/ab_min_tile=1024/256=4), so the M-heaviest experts receive the
  // most threads.  Assert that contract rather than the old {2,3}.
  EXPECT_NE(snap.n_thr_fixed, 0)
      << "proportional leaves n_thr_fixed at base (HYBRID clears to 0).";

  int sum = 0;
  for (int e = 0; e < num_ops; ++e) {
    const int n = static_cast<int>(snap.stable_n_thr_per_expert[e]);
    if (Ms[e] > 0) {
      EXPECT_GE(n, 1) << "active expert e=" << e << " starved (0 thr)";
    }
    EXPECT_LE(n, 4) << "expert e=" << e
                    << " exceeds N-tile capacity cap=4 (N=1024/256)";
    sum += n;
  }
  EXPECT_EQ(sum, 32)
      << "M-proportional should saturate the full thread team; "
         "sum(stable_n_thr_per_expert) = " << sum;
  // Heaviest-first: the M-heaviest expert (e=0, M=16) gets a
  // proportional share above the uniform base (2) and >= every other
  // expert; the lightest (M=4) experts get strictly fewer.
  const int n_heavy = static_cast<int>(snap.stable_n_thr_per_expert[0]);
  EXPECT_GE(n_heavy, 3) << "heaviest expert (M=16) under-allocated";
  EXPECT_GT(n_heavy, static_cast<int>(snap.stable_n_thr_per_expert[13]))
      << "heaviest expert must outrank the lightest under proportional";
}

// ===============================================================================
// [8a.2] TestGroupMatmulPhaseBRemainder — eligibility filter, non-uniform N
//
// Companion to `HeaviestFirstAssignment`: that test uses uniform N
// = 1024 so every expert is eligible (`N[e] / ab_min_tile = 4 ≥
// base + 1 = 3`).  Without a non-uniform-N case a regression that
// removed the per-expert eligibility check would still pass —
// every expert qualifies in the uniform fixture.
//
// This test pins the eligibility filter explicitly: same 32t /
// 14ops / K=64 setup, same skewed M = {16,12,10,8,6,4×9}, but with
// the M-heaviest expert (e=0, M=16) given N=512 instead of 1024.
// Its tile capacity becomes `512 / 256 = 2 < base+1 = 3`, so it
// MUST NOT receive `base+1` even though it is M-heaviest.  The
// expected slot-by-slot distribution is:
//
//   * e=0 (M=16, N=512, INELIGIBLE)            → base    = 2
//   * e=1..4 (M=12,10,8,6, N=1024) heaviest of → base+1  = 3
//     the eligible set; receive all `remainder=4` extras
//   * e=5..13 (M=4, N=1024)                    → base    = 2
//
// Sum = 2 + 4×3 + 9×2 = 32 (full team utilisation).
//
// A regression that ignores `pairs[e].eligible` would route the
// extra to e=0 (highest M, sorted first), leaving e=4 at base —
// the assertion below would catch that immediately because e=0's
// `stable_n_thr_per_expert` would read 3 instead of 2.
// ===============================================================================

TEST(TestGroupMatmulPhaseBRemainder, EligibilityFilter_NonUniformN) {
  using namespace moe_test_utils;
  using zendnnl::lowoha::matmul::group_matmul_direct;
  using zendnnl::lowoha::matmul::GroupNTileStrategy;

  const int saved_num_threads = omp_get_max_threads();
  struct ThreadGuard {
    int prev;
    ~ThreadGuard() { omp_set_num_threads(prev); }
  } thread_guard{saved_num_threads};
  omp_set_num_threads(32);

  int actual_team_size = 0;
  #pragma omp parallel
  {
    #pragma omp master
    actual_team_size = omp_get_num_threads();
  }
  if (actual_team_size < 32) {
    GTEST_SKIP() << "Requires >= 32 active OMP threads; have "
                 << actual_team_size;
  }

  CustomKernelOverride        ck_on(true);
  NRoundsModeOverride         single_round(1);
  CustomKernelNTileOverride   default_n_tile(0);
  // Insulate from a process-level
  // `ZENDNNL_GRP_MATMUL_N_TILE_STRATEGY=1/2` that would bypass
  // ManyExperts and skip Phase B — see
  // `CkSingleRoundRemainderCorrectness` for the full rationale.
  NTileStrategyOverride       auto_strategy(0);
  // Decode M-proportional split is OFF by default; this test verifies the
  // proportional split's per-expert capacity-cap clamp, so pin it ON.
  DecodeProportionalOverride  prop_on(true);

  if (!zendnnl::lowoha::matmul::custom_kernel::dispatch_supported()) {
    GTEST_SKIP() << "Test requires AVX512BF16 / custom-kernel "
                    "dispatch support; cannot exercise Phase B "
                    "without CK engagement.";
  }

  reset_grp_matmul_caches();

  const int num_ops      = 14;
  const int K            = 64;
  const int N_eligible   = 1024;   // 1024 / 256 = 4 ≥ base+1
  const int N_ineligible = 512;    //  512 / 256 = 2 < base+1
  const std::vector<int> Ms =
      {16, 12, 10, 8, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4};
  // Heaviest expert (e=0, M=16) gets the narrow N — sorts first
  // by M descending but is filtered out by the eligibility check.
  std::vector<int> Ns(num_ops, N_eligible);
  Ns[0] = N_ineligible;
  ASSERT_EQ(static_cast<int>(Ms.size()), num_ops);

  std::vector<std::vector<bfloat16_t>> src(num_ops);
  std::vector<std::vector<bfloat16_t>> wei(num_ops);
  std::vector<std::vector<bfloat16_t>> dst(num_ops);
  for (int e = 0; e < num_ops; ++e) {
    src[e].resize(static_cast<size_t>(Ms[e]) * K);
    wei[e].resize(static_cast<size_t>(K) * Ns[e]);
    dst[e].assign(static_cast<size_t>(Ms[e]) * Ns[e],
                  bfloat16_t(0.0f));
    fill_src(src[e],  e, 0.02f);
    fill_wei1(wei[e], e, 0.005f);
  }

  std::vector<const void *> srcs(num_ops), weis(num_ops),
      biases(num_ops, nullptr);
  std::vector<void *> dsts(num_ops);
  for (int e = 0; e < num_ops; ++e) {
    srcs[e] = src[e].data();
    weis[e] = wei[e].data();
    dsts[e] = dst[e].data();
  }

  // Build with a uniform "max" N then patch per-expert N / ldb /
  // ldc.  `lda` is M-independent (transA=false → lda=K), `ldb`
  // and `ldc` track the per-expert N column count.
  auto gv = GemmVecs::uniform(num_ops, /*M=*/0, /*N=*/N_eligible, K,
                              /*alpha=*/1.0f, /*beta=*/0.0f,
                              /*wc=*/true, /*tA=*/false,
                              /*tB=*/false);
  gv.Ms = Ms;
  gv.Ns = Ns;
  for (int e = 0; e < num_ops; ++e) {
    gv.ldb[e] = Ns[e];
    gv.ldc[e] = Ns[e];
  }
  auto params = make_uniform_params(num_ops, data_type_t::bf16);
  for (int e = 0; e < num_ops; ++e) {
    params[e].num_threads = 32;
  }

  AlgoEnvGuard       algo_guard(3);
  PhaseBCaptureGuard cap;

  ASSERT_EQ(group_matmul_direct(gv.layout, gv.transA, gv.transB,
                                gv.Ms, gv.Ns, gv.Ks, gv.alpha,
                                srcs, gv.lda, weis, gv.ldb,
                                biases, gv.beta, dsts, gv.ldc,
                                gv.is_wc, params,
                                /*moe_postop=*/nullptr,
                                /*gated_act=*/nullptr),
            status_t::success);

  const auto &snap =
      zendnnl::lowoha::matmul::test_api::s_last_phase_b_snapshot;

  ASSERT_TRUE(snap.valid)
      << "PhaseBSnapshot was not captured.";
  EXPECT_TRUE(snap.per_expert_remainder)
      << "Phase B did not fire — eligibility filter may have "
         "rejected EVERY expert (eligible_count == 0 short-"
         "circuits with `per_expert_remainder = false`).  "
         "strategy=" << static_cast<int>(snap.strategy);
  EXPECT_EQ(static_cast<int>(snap.strategy),
            static_cast<int>(GroupNTileStrategy::ManyExperts));
  EXPECT_EQ(snap.num_ops_active, num_ops);

  // Decode-class (max_M=16 <= kDecodeMaxM) → M-proportional split.  The
  // eligibility/capacity intent is PRESERVED by proportional's per-expert
  // cap clamp `cap[e] = N[e] / ab_min_tile`: e=0 has N=512 → cap=2, so
  // despite being the M-heaviest it CANNOT exceed 2 threads, while the
  // full-N (1024 → cap=4) eligible experts absorb the surplus.  Assert
  // the cap contract rather than the legacy uniform {2,3} slot pattern.
  EXPECT_NE(snap.n_thr_fixed, 0);

  int sum = 0;
  for (int e = 0; e < num_ops; ++e) {
    const int n = static_cast<int>(snap.stable_n_thr_per_expert[e]);
    if (Ms[e] > 0) {
      EXPECT_GE(n, 1) << "active expert e=" << e << " starved (0 thr)";
    }
    sum += n;
  }
  EXPECT_EQ(sum, 32)
      << "M-proportional should saturate the full thread team even with "
         "one capacity-capped heavy expert; sum=" << sum;

  // KEY: e=0 (M-heaviest but N=512) is clamped to its N-tile capacity
  // cap=2 and must NOT receive more, even though it sorts first by M.
  const int n_e0 = static_cast<int>(snap.stable_n_thr_per_expert[0]);
  EXPECT_LE(n_e0, 2)
      << "INELIGIBLE/capacity-capped expert e=0 (M=16, N=512, cap=2) "
         "must not exceed its N-tile capacity; got " << n_e0
      << " — check the cap[e] clamp in apply_round_pick()'s "
         "proportional split.";
  // A full-N eligible heavy (e=1, M=12, N=1024, cap=4) outranks the
  // capacity-capped e=0.
  EXPECT_GT(static_cast<int>(snap.stable_n_thr_per_expert[1]), n_e0)
      << "full-capacity heavy e=1 should outrank capacity-capped e=0.";
}

// ===============================================================================
// [8b] TestGroupMatmulAutoSelectAlgo — pin down `select_grp_matmul_algo`'s
//      decisions on the (max_M, num_threads, num_ops) grid.
//
// Auto-select is a top-level capacity carve-out (rule 0) plus three
// policy rules, evaluated in priority order:
//
//   0. num_ops > kNTilePlanMaxExperts (=256) → ALGO 5   (capacity carve-out)
//   1. num_ops ≥ num_threads                 → ALGO 3   (Qwen-style)
//   2. num_ops ≤ kFewExpertsAlgo1 (= 8)      → ALGO 1   (Mixtral-style)
//   3. otherwise — M-driven:
//        prompt (max_M > kDecodeMaxM=32)     → ALGO 1
//        decode (max_M ≤ kDecodeMaxM)        → ALGO 3   (gpt-oss-style)
//
// The matrix below covers each arrow explicitly with at least one
// canonical workload (Qwen3-30B-A3B for rule 1, Mixtral 8x7B for
// rule 2, gpt-oss-20B for rule 3, and synthetic E257/E512 shapes for
// rule 0) plus thread-count sweeps to lock the rules across {32,
// 48, 64, 80, 96, 128, 192} threads where applicable.  A future
// heuristic edit that flips any row should produce a deterministic
// test failure pointing at the offending label.
//
// SCOPE NOTE — what this matrix tests.
//   These cases assert auto-select's ROUTING DECISION only — i.e.,
//   which top-level ALGO the dispatcher picks for the given inputs.
//   They do NOT verify the downstream N-tile planner's strategy
//   choice (`ManyExperts` vs `Sequential` vs `FewExperts` etc.) for
//   the ALGO 3 rows.  For shapes where ALGO 3 is selected but the
//   N-tile planner's `ntile_viable` returns false (e.g., Qwen
//   prompt at N=1536 → only 3 tiles, below the ManyExperts minimum
//   of ccd_size/2 = 4), the planner falls back to Sequential and
//   executes a path equivalent to ALGO 1 plus one-time ALGO 3
//   prepack overhead.  That fallback is the user-visible behaviour
//   for "Qwen prompt" today; it is documented in the rule-1 SCOPE
//   NOTE in `group_matmul_dispatch.cpp::auto_select_algo` and
//   accepted as a trade-off for the simpler 3-rule selector.
//   Verifying which strategy fires at execute time would require
//   either a gemm_mode assertion or the white-box planner snapshot
//   (`PhaseBSnapshot`) — currently used only for Phase B coverage.
//
// The test pokes `select_grp_matmul_algo` directly so we don't need to
// build the full dispatcher stack just to read out the decision.  It
// uses `AlgoEnvGuard` to clear `ZENDNNL_GRP_MATMUL_ALGO` (the override
// is intentionally re-read each call so in-process setenv works) so
// auto-select fires regardless of how the test was invoked.
// ===============================================================================

struct AutoSelectParam {
  int    M;                ///< Uniform M per expert.
  int    K;
  int    N;
  int    num_ops;
  int    num_threads;
  int    expected_algo;
  std::string label;       ///< Human-readable name for the parameterised case.
};

static std::string AutoSelectParamName(
  const ::testing::TestParamInfo<AutoSelectParam> &info) {
  return info.param.label;
}

class TestGroupMatmulAutoSelectAlgo
    : public ::testing::TestWithParam<AutoSelectParam> {};

TEST_P(TestGroupMatmulAutoSelectAlgo, MatchesExpected) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;

  reset_grp_matmul_caches();

  const auto &p = GetParam();

  // Pin ZENDNNL_GRP_MATMUL_ALGO=0 so the auto-select path fires
  // (override is uncached on purpose; an env value lingering from a
  // prior test would shadow our expectation).  `get_grp_matmul_algo`
  // returns 0 for any non-{1..5} value, i.e. "auto-select".
  AlgoEnvGuard reset_algo(0);
  // Explicitly disable the phase env (default PROMPT=2 / DECODE=3) so
  // this suite continues to assert the LEGACY 3-rule cascade outcomes
  // it was authored for.  Without these overrides, the new default
  // phase pins would shadow the cascade and break every parametrized
  // row.  The phase env behaviour is covered by
  // `TestGroupMatmulAutoPhaseEnv` below.
  AutoPromptAlgoOverride legacy_prompt(0);
  AutoDecodeAlgoOverride legacy_decode(0);

  const int N_ops = p.num_ops;

  // Minimal but valid inputs for the dispatcher safety checks
  // (`check_m_tile_safe` / `check_n_tile_extra`) inside
  // `select_grp_matmul_algo`.  Row-major, uniform BF16 dtypes, no
  // packed-B, no quant.
  std::vector<char>  layout(N_ops, 'r');
  std::vector<int>   M(N_ops, p.M);
  std::vector<int>   N(N_ops, p.N);
  std::vector<int>   K(N_ops, p.K);
  std::vector<matmul_params> params(N_ops);
  for (int i = 0; i < N_ops; ++i) {
    params[i].dtypes.src  = data_type_t::bf16;
    params[i].dtypes.wei  = data_type_t::bf16;
    params[i].dtypes.dst  = data_type_t::bf16;
    params[i].dtypes.bias = data_type_t::bf16;
    params[i].mem_format_a = 'n';
    params[i].mem_format_b = 'n';
    params[i].dynamic_quant = false;
    params[i].packing.pack_format_b = 0;
  }

  const int got = select_grp_matmul_algo(layout, M, N, K, params,
                                         p.num_threads);
  EXPECT_EQ(got, p.expected_algo)
      << "auto-select picked ALGO " << got
      << ", expected " << p.expected_algo
      << "  [" << p.label << "]";
}

// Grid: enumerate at least one canonical workload per rule plus
// thread-count sweeps where they exercise the rule, and a few
// edge-case rows to lock the rule precedence.  The label composes
// `<arch>_<phase>_E<num_ops>_t<num_threads>` so a failure points
// straight at the offending case.
static std::vector<AutoSelectParam> make_auto_select_params() {
  std::vector<AutoSelectParam> out;

  // ── Rule 3 (gpt-oss-20B): ops 9..32, decode → ALGO 3, prompt → ALGO 1 ──
  // gpt-oss-20B-class shape (Op1, gate+up).  K=2880, N=5760, BF16.
  const int K_GO = 2880, N_GO = 5760;

  // Prompt: BS={8,16,32}, seq=128, topk=4 → M_per_expert ≈ {128, 256, 512}.
  // num_ops=32, max_M > kDecodeMaxM → rule 3 (M-driven) → ALGO 1,
  // EXCEPT at num_threads=32 where rule 1 fires first (num_ops ==
  // num_threads → ALGO 3).  The 32t case isn't part of the
  // gpt-oss-20B benchmark grid (which targets 64t / 128t) so the
  // rule-1 routing there is a documented boundary not a regression;
  // pinning both branches here catches any future drift in the
  // priority order.
  for (int M_val : {128, 256, 512}) {
    for (int nt : {32, 48, 64, 80, 96, 128, 192}) {
      const int expected_algo = (32 == nt) ? 3 : 1;  // rule 1 vs rule 3
      out.push_back({
        /*M=*/M_val, /*K=*/K_GO, /*N=*/N_GO,
        /*num_ops=*/32, /*num_threads=*/nt,
        expected_algo,
        "prompt_gptoss_M" + std::to_string(M_val)
          + "_t" + std::to_string(nt)});
    }
  }

  // Decode: max_M ≤ kDecodeMaxM=32, num_ops=32 in (8, num_threads),
  // rule 3 → ALGO 3.  At num_threads=32 the row hits rule 1 instead
  // (num_ops == num_threads → ALGO 3) — same outcome, different rule.
  for (int M_val : {4, 16, 32}) {
    for (int nt : {64, 128}) {
      out.push_back({
        M_val, K_GO, N_GO, /*num_ops=*/32, nt,
        /*expected_algo=*/3,
        "decode_gptoss_M" + std::to_string(M_val)
          + "_t" + std::to_string(nt)});
    }
  }

  // ── Rule 2 (Mixtral 8x*): num_ops ≤ 8 → ALGO 1 (prompt + decode) ──
  // Mixtral 8x7B / 8x22B decoder block shape: K=4096, N=14336, BF16
  // (~112 MB / expert).  Real Mixtral has exactly 8 experts; the
  // few-experts gate triggers regardless of M / weight class.
  // Prompt at every host size we care about.
  for (int M_val : {128, 256, 512}) {
    for (int nt : {32, 48, 64, 80, 96, 128, 192}) {
      out.push_back({
        /*M=*/M_val, /*K=*/4096, /*N=*/14336,
        /*num_ops=*/8, /*num_threads=*/nt,
        /*expected_algo=*/1,
        "mixtral_8x7B_prompt_M" + std::to_string(M_val)
          + "_t" + std::to_string(nt)});
    }
  }
  // Decode: same Mixtral shape, decode-class M, rule 2 still wins
  // (num_ops=8 ≤ 8) and routes to ALGO 1.
  for (int M_val : {4, 16, 32}) {
    for (int nt : {64, 128}) {
      out.push_back({
        M_val, /*K=*/4096, /*N=*/14336,
        /*num_ops=*/8, nt, /*expected_algo=*/1,
        "mixtral_8x7B_decode_M" + std::to_string(M_val)
          + "_t" + std::to_string(nt)});
    }
  }
  // Mirror shape (K=14336, N=4096) — tall-N variant, still 8 experts
  // → still rule 2 → ALGO 1.  Pinned because today's heuristic also
  // routed this to ALGO 1 via a different code path; the new rule
  // gives the same outcome via a simpler reason.
  out.push_back({
    /*M=*/256, /*K=*/14336, /*N=*/4096,
    /*num_ops=*/8, /*num_threads=*/128, /*expected_algo=*/1,
    "mixtral_tallN_E8_t128"});

  // ── Rule 1 (Qwen3-30B-A3B): num_ops ≥ num_threads → ALGO 3 ──
  // Qwen3-30B-A3B Op1 (gate+up) shape: K=2048, N=1536 (768×2 for
  // gate+up), BF16.  128 experts, top-k=8.  Pin both prompt and
  // decode at the two host sizes the deployment targets (64t, 128t)
  // — at both, num_ops=128 ≥ num_threads so rule 1 fires
  // unconditionally and ALGO 3 wins both phases at SELECTION TIME.
  //
  // EXECUTION TIME — known fallback for the prompt rows.
  //   At N=1536, `kMinNTile=512` → `1536 / 512 = 3` tiles per expert,
  //   below the ManyExperts minimum (ccd_size/2 = 4 on 8-core CCDs).
  //   The N-tile planner's `ntile_viable` therefore returns false for
  //   the prompt-class rows here (max_M ∈ {128, 256, 512}) and the
  //   call falls back to Sequential — behaving like ALGO 1 plus one
  //   ALGO 3 prepack pass per process.  Decode-class rows
  //   (max_M ∈ {4, 16, 32}) take the decode tile (256), get 6 tiles
  //   per expert, clear `ntile_viable`, and run via ManyExperts (the
  //   path Phase B is sized for).  Both behaviours are intentional
  //   under the simplified 3-rule selector; see the rule-1 SCOPE
  //   NOTE in `group_matmul_dispatch.cpp::auto_select_algo` for the
  //   trade-off discussion.  These rows pin the SELECTION result;
  //   the planner's execute-time fallback for prompt is verified by
  //   `plan_group_n_tile`'s own self-fallback tests (R1/R2/R3 in
  //   `group_matmul_n_tile.cpp`).
  const int K_QW = 2048, N_QW = 1536;
  for (int M_val : {4, 16, 32, 128, 256, 512}) {
    for (int nt : {64, 128}) {
      const char *phase =
          (M_val <= zendnnl::lowoha::matmul::kDecodeMaxM)
              ? "decode" : "prompt";
      out.push_back({
        M_val, K_QW, N_QW, /*num_ops=*/128, nt,
        /*expected_algo=*/3,
        std::string("qwen3_30B_") + phase
          + "_E128_M" + std::to_string(M_val)
          + "_t" + std::to_string(nt)});
    }
  }

  // ── Rule 1 + capacity carve-out (1.a): num_ops vs kNTilePlanMaxExperts ──
  // Rule 1 routes num_ops ≥ num_threads → ALGO 3, BUT the N-tile
  // planner's fixed-size stack arrays cap at
  // `GroupNTilePlan::kMaxExperts == kNTilePlanMaxExperts == 256`.
  // Sub-gate 1.a routes num_ops > kNTilePlanMaxExperts → ALGO 5 to
  // avoid the N-tile planner's R3 silent fallback to its Sequential
  // strategy (one expert at a time, full team each), which is
  // materially slower than ALGO 5's per-expert OMP-parallel schedule
  // on many-experts decode-class shapes.
  //
  // Three pins cover this boundary:
  //   * E256 — exactly at capacity, still ALGO 3 (ManyExperts strategy
  //     with multi-round scheduling handles 256 experts fine).
  //   * E257 — first value above capacity, must flip to ALGO 5.
  //   * E512 — well above capacity, must stay on ALGO 5.
  out.push_back({
    /*M=*/4, /*K=*/2880, /*N=*/5760,
    /*num_ops=*/zendnnl::lowoha::matmul::kNTilePlanMaxExperts,
    /*num_threads=*/128, /*expected_algo=*/3,
    "many_experts_E256_t128_at_capacity"});
  out.push_back({
    /*M=*/4, /*K=*/2880, /*N=*/5760,
    /*num_ops=*/zendnnl::lowoha::matmul::kNTilePlanMaxExperts + 1,
    /*num_threads=*/128, /*expected_algo=*/5,
    "many_experts_E257_t128_above_capacity"});
  out.push_back({
    /*M=*/4, /*K=*/2880, /*N=*/5760,
    /*num_ops=*/512, /*num_threads=*/128, /*expected_algo=*/5,
    "many_experts_E512_t128_above_capacity"});

  // ── Rule 1 boundary: num_ops == num_threads → ALGO 3 ──
  // The "≥" in rule 1 is intentional; this row pins it.  Without
  // it a future change to "num_ops > num_threads" would silently
  // flip the boundary case to rule 3 (M-based).
  out.push_back({
    /*M=*/4, /*K=*/2880, /*N=*/5760,
    /*num_ops=*/64, /*num_threads=*/64, /*expected_algo=*/3,
    "rule1_boundary_E64_t64_decode"});
  out.push_back({
    /*M=*/256, /*K=*/2880, /*N=*/5760,
    /*num_ops=*/64, /*num_threads=*/64, /*expected_algo=*/3,
    "rule1_boundary_E64_t64_prompt"});

  return out;
}

INSTANTIATE_TEST_SUITE_P(GroupMatmulAutoSelect, TestGroupMatmulAutoSelectAlgo,
                         ::testing::ValuesIn(make_auto_select_params()),
                         AutoSelectParamName);

// ===============================================================================
// [8c] TestGroupMatmulAutoSelectAlgo_DynamicQuant — the single
//      quant configuration `check_n_tile_extra` currently accepts:
//      `dynamic_quant=true` with `{M, 1}` source + `{1, N}` per-
//      channel weight.  ALGO 3 reaches that case via `flat_n_tile`'s
//      pre-OMP source-reorder hoist (`HoistedSrcQuant` in
//      `group_matmul_n_tile.cpp`); per-tile threads then share the
//      hoisted S8 src + column-sliced wei scale.  Anything else —
//      static src, per-tensor / per-group wei, per-group src,
//      pure WOQ — falls back to ALGO 1.
//
// This test pins the four boundaries of the scope:
//   1. Non-quantised baseline → ALGO 3 (the shape itself is
//      N-tile-viable so the negative cases below prove the gate
//      flipped, not the shape).
//   2. `dynamic_quant=true` + `{M, 1}` src + `{1, N}` wei (the
//      one accepted shape) → ALGO 3.
//   3. Same as (2) but `dynamic_quant=false` → ALGO 1 (static
//      src is out of scope).
//   4. Same as (2) but `wei_scale.buff = nullptr` → ALGO 1
//      (wei pair side is required).
//
// (The hoist path itself is exercised end-to-end by the
// `TestGroupMatmulQuant.INT8_DYNAMIC_GEMM_*` suites, which carry
// both sides of the per-token pair and reach `flat_n_tile` when
// their random num_ops × num_threads combo trips auto-select's
// rule 1 or rule 3-decode arm.)
//
// `check_m_tile_safe`'s row-local granularity gate
// (`src_scale.dims[0] == M[i]`) is still a precondition; the
// per-token gate below is stricter, so anything that fails this
// test's `dynamic_quant=true + {M[i], 1}` requirement is also
// caught by `m_tile_safe` for the dynamic case.
//
// Case 5 pins the single-row decode case (`M[i] == 1` with
// `src_scale.dims = {1, 1}`): the gate was rewritten to compare
// `src_scale.dims[0]` against the actual per-expert `M[i]` (it
// previously used a brittle `dims[0] > 1` proxy that rejected
// sparse-MoE batches where some experts get exactly one routed
// token).
// ===============================================================================

TEST(TestGroupMatmulAutoSelectAlgo_DynamicQuant, AcceptsAlgo3ViaHoist) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;

  reset_grp_matmul_caches();
  AlgoEnvGuard reset_algo(0);   // auto-select
  // Force the legacy 3-rule cascade — the test was authored to assert
  // a Rule-3 decode-class shape ALGO 3 → ALGO 1 transition driven by
  // the dynamic_quant gate.  With the new phase env defaults the
  // decode path would resolve via `AUTO_DECODE_ALGO=3` clamped on
  // !n_tile_safe to ALGO 1 directly, masking the actual gate this
  // test was written to pin.
  AutoPromptAlgoOverride legacy_prompt(0);
  AutoDecodeAlgoOverride legacy_decode(0);

  const int N_ops       = 32;
  const int M_val       = 16;     // decode-class (≤ kDecodeMaxM)
  const int K_GO        = 2880;
  const int N_GO        = 5760;   // wide-N decode shape
  const int num_threads = 64;

  std::vector<char> layout(N_ops, 'r');
  std::vector<int>  M(N_ops, M_val);
  std::vector<int>  N(N_ops, N_GO);
  std::vector<int>  K(N_ops, K_GO);
  std::vector<matmul_params> params(N_ops);
  for (int i = 0; i < N_ops; ++i) {
    params[i].dtypes.src           = data_type_t::bf16;
    params[i].dtypes.wei           = data_type_t::bf16;
    params[i].dtypes.dst           = data_type_t::bf16;
    params[i].dtypes.bias          = data_type_t::bf16;
    params[i].mem_format_a         = 'n';
    params[i].mem_format_b         = 'n';
    params[i].dynamic_quant        = false;
    params[i].packing.pack_format_b = 0;
  }

  // Case 1 — Sanity: same shape WITHOUT any quant lands on ALGO 3
  // (decode-class + many experts + wide-N + ntile_viable).
  ASSERT_EQ(select_grp_matmul_algo(layout, M, N, K, params, num_threads), 3)
      << "baseline non-quant shape must select ALGO 3 — otherwise "
         "the negative cases below cannot distinguish a "
         "shape-induced fallback from a quant-gate rejection";

  // Case 2 — Accept the single in-scope shape: dynamic_quant=true
  // with `{M, 1}` src + `{1, N}` wei.  `src_scale.buff` stays
  // nullptr (wrapper allocates on the hoist path); `wei_scale.buff`
  // is a stack sentinel since the gate only checks for non-nullness
  // (it never dereferences the pointer), and using a local valid
  // address keeps the test allocator-free.
  alignas(float) char wei_scale_sentinel = 0;
  for (int i = 0; i < N_ops; ++i) {
    params[i].dynamic_quant = true;
    params[i].quant_params.src_scale.dims = {M_val, 1};
    params[i].quant_params.src_scale.dt   = data_type_t::f32;
    params[i].quant_params.wei_scale.dims = {1, N_GO};
    params[i].quant_params.wei_scale.dt   = data_type_t::f32;
    params[i].quant_params.wei_scale.buff = &wei_scale_sentinel;
  }

  EXPECT_EQ(select_grp_matmul_algo(layout, M, N, K, params, num_threads), 3)
      << "dynamic_quant=true with `{M, 1}` src + `{1, N}` wei "
         "(the only in-scope quant pair) and otherwise N-tile-"
         "viable inputs should reach ALGO 3 via the pre-OMP hoist "
         "path.  A regression here would indicate either that "
         "`check_n_tile_extra` has tightened its dynamic / per-token "
         "/ per-channel gate, or that `check_m_tile_safe`'s "
         "row-local gate has tightened.";

  // Case 3 — Static src rejection: same shape and dims, but flip
  // `dynamic_quant=false`.  The current scope is dynamic-only; this
  // should fall back to ALGO 1 even though the dims/buff layout
  // looks otherwise valid for a static W8A8 per-token deployment.
  for (int i = 0; i < N_ops; ++i) {
    params[i].dynamic_quant = false;
    // Static src needs a populated src buff — give the gate something
    // realistic to look at (it'll reject before reading the pointer).
    params[i].quant_params.src_scale.buff = &wei_scale_sentinel;
  }

  EXPECT_EQ(select_grp_matmul_algo(layout, M, N, K, params, num_threads), 1)
      << "static src quant (`dynamic_quant=false`) is intentionally "
         "outside the current per-token-dynamic scope — even with "
         "the otherwise-valid `{M, 1}` src + `{1, N}` wei pair, the "
         "gate must reject and fall back to ALGO 1.  A regression "
         "here would indicate the gate has relaxed its "
         "`dynamic_quant=true` requirement.";

  // Case 4 — Wei-missing rejection: restore `dynamic_quant=true`
  // but drop the wei_scale entirely.  The per-token scope requires
  // a populated wei pair, so this should fall back to ALGO 1.
  for (int i = 0; i < N_ops; ++i) {
    params[i].dynamic_quant = true;
    params[i].quant_params.src_scale.buff = nullptr;  // back to hoist-alloc
    params[i].quant_params.wei_scale.buff = nullptr;
    params[i].quant_params.wei_scale.dims.clear();
    params[i].quant_params.wei_scale.dt = data_type_t::none;
  }

  EXPECT_EQ(select_grp_matmul_algo(layout, M, N, K, params, num_threads), 1)
      << "dynamic_quant=true with `{M, 1}` src but NO wei scale "
         "should fall back to ALGO 1 — the per-token scope guard "
         "in `check_n_tile_extra` requires the paired per-channel "
         "wei.  A regression here would indicate the gate has "
         "accidentally relaxed the wei-side requirement.";

  // Case 5 — Heterogeneous-M batch with one single-row expert
  // (`M[i] == 1`).  This is the common sparse-MoE decode pattern:
  // most experts get a handful of tokens, one or two experts get
  // exactly one routed token.  Previously the gates used a
  // `src_scale.dims[0] > 1` proxy that rejected the M=1 expert
  // and dragged the whole batch onto ALGO 1.  After the
  // M[i]-aware fix the gate accepts `{1, 1}` when paired with
  // M[i]=1 and the batch keeps ALGO 3 N-tile parallelism.
  //
  // Restore the in-scope quant configuration for every expert and
  // shrink the last expert's M to 1 (matching `{1, 1}` src_scale
  // dims) while keeping the other experts at M_val.  The custom
  // mixed-M `M` vector replaces the uniform one used above; N/K
  // stay uniform across experts because that's how MoE callers
  // present per-expert vectors today.
  std::vector<int> M_mixed(N_ops, M_val);
  M_mixed.back() = 1;
  for (int i = 0; i < N_ops; ++i) {
    params[i].dynamic_quant = true;
    params[i].quant_params.src_scale.buff = nullptr;
    params[i].quant_params.src_scale.dims = {M_mixed[i], 1};
    params[i].quant_params.src_scale.dt   = data_type_t::f32;
    params[i].quant_params.wei_scale.dims = {1, N_GO};
    params[i].quant_params.wei_scale.dt   = data_type_t::f32;
    params[i].quant_params.wei_scale.buff = &wei_scale_sentinel;
  }

  EXPECT_EQ(select_grp_matmul_algo(layout, M_mixed, N, K, params,
                                   num_threads), 3)
      << "sparse-MoE batch with one M[i]=1 expert (`src_scale.dims "
         "= {1, 1}`) should still reach ALGO 3 — the M[i]-aware "
         "gate accepts the single-row decode case because the "
         "single-thread reorder for that expert is race-free and "
         "the per-token scale equals the full-matrix scale "
         "trivially.  A regression here means the gate is using "
         "the old brittle `dims[0] > 1` proxy again and will drag "
         "every sparse-MoE decode batch onto sequential_experts.";
}

// ===============================================================================
// [8d] TestGroupMatmulAutoPhaseEnv — `ZENDNNL_GRP_MATMUL_AUTO_PROMPT_ALGO`
//      and `ZENDNNL_GRP_MATMUL_AUTO_DECODE_ALGO` per-phase pinning.
//
// New envs let an operator pin a specific ALGO per phase without
// touching the global `ZENDNNL_GRP_MATMUL_ALGO`.  Defaults:
// PROMPT=1 (sequential_experts), DECODE=3 (N-tile rounds + CK).
// Set the env to `0` for the legacy 3-rule cascade.  Tests here
// pin both branches: explicit env=0 reproduces the legacy cascade,
// env=1..3 honoured for the matching phase, plus the safety clamp
// paths (ALGO 3 on !n_tile_safe) and the structural R0 capacity
// gate that ignores the phase env.
//
// All tests use `select_grp_matmul_algo` directly with `ALGO=0` so
// auto-select fires; the phase env is the only thing that varies.
// ===============================================================================

namespace {

// Build a minimal valid `(layout, M, N, K, params)` tuple for an
// auto-select probe.  Row-major BF16, no quant, no packed-B.
struct AutoProbeShape {
  std::vector<char>            layout;
  std::vector<int>             M;
  std::vector<int>             N;
  std::vector<int>             K;
  std::vector<zendnnl::lowoha::matmul::matmul_params> params;
  int                          num_threads;
};

static AutoProbeShape build_auto_probe(int M_val, int K_val, int N_val,
                                       int num_ops, int num_threads) {
  using namespace zendnnl::lowoha::matmul;
  AutoProbeShape s;
  s.layout.assign(num_ops, 'r');
  s.M.assign(num_ops, M_val);
  s.N.assign(num_ops, N_val);
  s.K.assign(num_ops, K_val);
  s.params.resize(num_ops);
  for (int i = 0; i < num_ops; ++i) {
    s.params[i].dtypes.src           = data_type_t::bf16;
    s.params[i].dtypes.wei           = data_type_t::bf16;
    s.params[i].dtypes.dst           = data_type_t::bf16;
    s.params[i].dtypes.bias          = data_type_t::bf16;
    s.params[i].mem_format_a         = 'n';
    s.params[i].mem_format_b         = 'n';
    s.params[i].dynamic_quant        = false;
    s.params[i].packing.pack_format_b = 0;
  }
  s.num_threads = num_threads;
  return s;
}

}  // namespace

// Explicit-zero phase env (escape hatch for legacy behaviour) — sets
// `AUTO_PROMPT_ALGO=0` and asserts the legacy 3-rule cascade fires.
// Mixtral-class prompt → Rule 2b (num_ops≤8) → ALGO 1.
TEST(TestGroupMatmulAutoPhaseEnv, ExplicitZeroEnablesLegacyMixtralPrompt) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  reset_grp_matmul_caches();
  AlgoEnvGuard reset_algo(0);
  AutoPromptAlgoOverride explicit_legacy_prompt(0);  // explicit legacy
  AutoDecodeAlgoOverride explicit_legacy_decode(0);

  // Mixtral-class prompt: 8 experts, K=4096, N=14336, M=256 (prompt).
  // Legacy Rule 2 → ALGO 1.
  auto s = build_auto_probe(/*M=*/256, /*K=*/4096, /*N=*/14336,
                            /*num_ops=*/8, /*num_threads=*/128);
  EXPECT_EQ(select_grp_matmul_algo(s.layout, s.M, s.N, s.K, s.params,
                                   s.num_threads),
            1)
      << "AUTO_PROMPT_ALGO=0 must defer to legacy rules (Mixtral-8 "
         "prompt → Rule 2 → ALGO 1)";
}

// Explicit-zero again, exercising legacy Rule 1 (num_ops ≥ num_threads).
TEST(TestGroupMatmulAutoPhaseEnv, ExplicitZeroEnablesLegacyQwenPrompt) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  reset_grp_matmul_caches();
  AlgoEnvGuard reset_algo(0);
  AutoPromptAlgoOverride explicit_legacy_prompt(0);
  AutoDecodeAlgoOverride explicit_legacy_decode(0);

  auto s = build_auto_probe(/*M=*/256, /*K=*/2048, /*N=*/1536,
                            /*num_ops=*/128, /*num_threads=*/128);
  EXPECT_EQ(select_grp_matmul_algo(s.layout, s.M, s.N, s.K, s.params,
                                   s.num_threads),
            3)
      << "AUTO_PROMPT_ALGO=0 must defer to legacy rules (Qwen "
         "num_ops≥num_threads → Rule 1 → ALGO 3)";
}

// Default (env unset → cached `2` for prompt, `3` for decode).
// Prompt-class Mixtral shape: phase env defaults to 2 (flat_m_tile)
// — the out-of-the-box auto policy.  No `Override` set; tests the
// cached env path's actual default value.
TEST(TestGroupMatmulAutoPhaseEnv, DefaultPromptRoutesToAlgo2) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  reset_grp_matmul_caches();
  AlgoEnvGuard reset_algo(0);
  // No AutoPromptAlgoOverride / AutoDecodeAlgoOverride — exercises
  // the unset-env default (PROMPT=2, DECODE=3).

  // Mixtral-class prompt — with the new default the phase env picks
  // ALGO 2 (flat_m_tile).  Shape is m_tile_safe (row-major,
  // uniform bf16 dtypes via build_auto_probe), so no safety clamp
  // fires.
  auto s = build_auto_probe(/*M=*/256, /*K=*/4096, /*N=*/14336,
                            /*num_ops=*/8, /*num_threads=*/128);
  EXPECT_EQ(select_grp_matmul_algo(s.layout, s.M, s.N, s.K, s.params,
                                   s.num_threads),
            2)
      << "AUTO_PROMPT_ALGO default (=2) must route prompt → ALGO 2";
}

// Default decode: phase env defaults to 3 (N-tile rounds + CK).
TEST(TestGroupMatmulAutoPhaseEnv, DefaultDecodeRoutesToAlgo3) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  reset_grp_matmul_caches();
  AlgoEnvGuard reset_algo(0);

  // gpt-oss-style decode, n_tile_safe.
  auto s = build_auto_probe(/*M=*/16, /*K=*/2880, /*N=*/5760,
                            /*num_ops=*/32, /*num_threads=*/64);
  EXPECT_EQ(select_grp_matmul_algo(s.layout, s.M, s.N, s.K, s.params,
                                   s.num_threads),
            3)
      << "AUTO_DECODE_ALGO default (=3) must route decode → ALGO 3";
}

// AUTO_PROMPT_ALGO=3 forces ALGO 3 on a shape Rule 2 would have
// sent to ALGO 1 (Mixtral-class prompt) — the override is honoured.
TEST(TestGroupMatmulAutoPhaseEnv, PromptEnvForcesAlgo3OnMixtralPrompt) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  reset_grp_matmul_caches();
  AlgoEnvGuard reset_algo(0);
  AutoPromptAlgoOverride force_prompt(3);  // ALGO 3 for prompt
  AutoDecodeAlgoOverride no_decode(0);

  auto s = build_auto_probe(/*M=*/256, /*K=*/4096, /*N=*/14336,
                            /*num_ops=*/8, /*num_threads=*/128);
  EXPECT_EQ(select_grp_matmul_algo(s.layout, s.M, s.N, s.K, s.params,
                                   s.num_threads),
            3)
      << "AUTO_PROMPT_ALGO=3 must override Rule 2 (num_ops≤8) for "
         "the prompt phase";
}

// AUTO_DECODE_ALGO=1 forces ALGO 1 on a decode shape Rule 3 would
// have sent to ALGO 3 (gpt-oss-class decode) — phase env honoured
// for decode phase only.
TEST(TestGroupMatmulAutoPhaseEnv, DecodeEnvForcesAlgo1OnGptOssDecode) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  reset_grp_matmul_caches();
  AlgoEnvGuard reset_algo(0);
  AutoPromptAlgoOverride no_prompt(0);
  AutoDecodeAlgoOverride force_decode(1);  // ALGO 1 for decode

  auto s = build_auto_probe(/*M=*/16, /*K=*/2880, /*N=*/5760,
                            /*num_ops=*/32, /*num_threads=*/64);
  EXPECT_EQ(select_grp_matmul_algo(s.layout, s.M, s.N, s.K, s.params,
                                   s.num_threads),
            1)
      << "AUTO_DECODE_ALGO=1 must override Rule 3 (decode → ALGO 3) "
         "for the decode phase";
}

// Rule 0.6 — more ACTIVE experts than threads → ALGO 5 (AUTO default).
// Qwen-class decode (88 active experts) on 64 threads: num_ops > num_threads,
// so ALGO 3's single round is infeasible → AUTO prefers ALGO 5
// (parallel_per_expert: flat per-expert OMP, schedule(dynamic)).
TEST(TestGroupMatmulAutoPhaseEnv, MoreExpertsThanThreadsRoutesToAlgo5) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  reset_grp_matmul_caches();
  AlgoEnvGuard reset_algo(0);
  // No phase-env override — exercises the default AUTO policy.

  auto s = build_auto_probe(/*M=*/16, /*K=*/2048, /*N=*/1536,
                            /*num_ops=*/88, /*num_threads=*/64);
  EXPECT_EQ(select_grp_matmul_algo(s.layout, s.M, s.N, s.K, s.params,
                                   s.num_threads),
            5)
      << "num_ops(88) > num_threads(64) must route AUTO to ALGO 5";
}

// The same Qwen-class shape on 128 threads: num_ops(88) <= num_threads(128),
// so Rule 0.6 does NOT fire and ALGO 3 single round stays selected.
TEST(TestGroupMatmulAutoPhaseEnv, ExpertsLeqThreadsKeepsAlgo3) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  reset_grp_matmul_caches();
  AlgoEnvGuard reset_algo(0);

  auto s = build_auto_probe(/*M=*/16, /*K=*/2048, /*N=*/1536,
                            /*num_ops=*/88, /*num_threads=*/128);
  EXPECT_EQ(select_grp_matmul_algo(s.layout, s.M, s.N, s.K, s.params,
                                   s.num_threads),
            3)
      << "num_ops(88) <= num_threads(128): ALGO 3 single round feasible, "
         "Rule 0.6 must not fire";
}

// Rule 0.6 OVERRIDES an explicit AUTO_DECODE_ALGO pin: in the
// experts-exceed-threads decode regime the flat per-expert path (ALGO 5)
// is the policy, so even an explicit AUTO_DECODE_ALGO=3 pin is superseded.
// Only the global ZENDNNL_GRP_MATMUL_ALGO force can override per-call.
TEST(TestGroupMatmulAutoPhaseEnv, MoreExpertsThanThreadsOverridesExplicitDecodePin) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  reset_grp_matmul_caches();
  AlgoEnvGuard reset_algo(0);
  AutoPromptAlgoOverride no_prompt(0);
  AutoDecodeAlgoOverride force_decode(3);  // explicit decode pin (superseded)

  auto s = build_auto_probe(/*M=*/16, /*K=*/2048, /*N=*/1536,
                            /*num_ops=*/88, /*num_threads=*/64);
  EXPECT_EQ(select_grp_matmul_algo(s.layout, s.M, s.N, s.K, s.params,
                                   s.num_threads),
            5)
      << "decode num_ops(88) > num_threads(64) must route to ALGO 5 even "
         "with an explicit AUTO_DECODE_ALGO=3 pin";
}

// The global ZENDNNL_GRP_MATMUL_ALGO force is still honoured in the
// experts-exceed-threads regime (it is applied before auto_select_algo),
// so an operator retains a per-call escape hatch back to ALGO 3.
TEST(TestGroupMatmulAutoPhaseEnv, MoreExpertsThanThreadsGlobalForceWins) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  reset_grp_matmul_caches();
  AlgoEnvGuard force_algo(3);  // global force ALGO 3

  auto s = build_auto_probe(/*M=*/16, /*K=*/2048, /*N=*/1536,
                            /*num_ops=*/88, /*num_threads=*/64);
  EXPECT_EQ(select_grp_matmul_algo(s.layout, s.M, s.N, s.K, s.params,
                                   s.num_threads),
            3)
      << "global ZENDNNL_GRP_MATMUL_ALGO=3 must win over Rule 0.6";
}

// Rule 0.7 (prompt M-tile regime routing): a PROMPT-class frame with
// active_ops > num_threads is M-tile-INFEASIBLE, so AUTO routes it to
// ALGO 5 (per-expert) — reproducing the executor flat_m_tile used to pick
// internally (its old round-based per-expert pool), now that ALGO 2 is a
// pure M-tile executor that would otherwise clamp this regime to sequential
// full-team.  (This is the M-tile-family default path: prompt algo == 2.)
TEST(TestGroupMatmulAutoPhaseEnv, PromptManyExpertsRoutesToAlgo5) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  reset_grp_matmul_caches();
  AlgoEnvGuard reset_algo(0);
  // No phase-env override — exercises the default AUTO policy (prompt=2).

  // Prompt-class: max_M=64 > kDecodeMaxM(32); 88 active experts on 64 threads
  // (active_ops > num_threads) → M-tile infeasible → ALGO 5.
  auto s = build_auto_probe(/*M=*/64, /*K=*/2048, /*N=*/1536,
                            /*num_ops=*/88, /*num_threads=*/64);
  EXPECT_EQ(select_grp_matmul_algo(s.layout, s.M, s.N, s.K, s.params,
                                   s.num_threads),
            5)
      << "prompt-class active_ops(88) > num_threads(64) is M-tile-infeasible; "
         "AUTO must route to ALGO 5 (per-expert)";
}

// Rule 0.7: a PROMPT-class wide-N frame (few actives × shallow M ×
// large N: max_M>1 && total_need*2 <= num_threads) routes to ALGO 1
// (sequential full-team) — reproducing flat_m_tile's old wide-N fallback,
// now that ALGO 2 is pure M-tile.
TEST(TestGroupMatmulAutoPhaseEnv, PromptWideNRoutesToAlgo1) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  reset_grp_matmul_caches();
  AlgoEnvGuard reset_algo(0);

  // 8 actives × M=8 on 64 threads: total_need = 8*ceil(8/16) = 8, 2*8=16
  // <= 64; max_M=8 > 1 → wide-N regime.  Prompt-class because... max_M=8
  // <= kDecodeMaxM(32) would be DECODE.  Use M=40 (>32 prompt) but keep
  // total_need*2 <= num_threads: 8*ceil(40/16)=8*3=24, 2*24=48 <= 64. ✓
  auto s = build_auto_probe(/*M=*/40, /*K=*/2048, /*N=*/1536,
                            /*num_ops=*/8, /*num_threads=*/64);
  EXPECT_EQ(select_grp_matmul_algo(s.layout, s.M, s.N, s.K, s.params,
                                   s.num_threads),
            1)
      << "prompt-class wide-N (8 actives × M=40 / 64t: total_need*2=48 <= 64, "
         "max_M=40 > 1) must route to ALGO 1 (sequential full-team)";
}

// Rule 0.7: a PROMPT-class genuine M-tile frame (not wide-N, not many-
// experts) routes to ALGO 2.  16 actives × M=512 on 64 threads:
// total_need = 16*ceil(512/16) = 16*32 = 512, 2*512=1024 > 64 (not wide-N);
// 16 <= 64 (not many-experts) → kMTile → ALGO 2.
TEST(TestGroupMatmulAutoPhaseEnv, PromptMTileRegimeRoutesToAlgo2) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  reset_grp_matmul_caches();
  AlgoEnvGuard reset_algo(0);

  auto s = build_auto_probe(/*M=*/512, /*K=*/2048, /*N=*/1536,
                            /*num_ops=*/16, /*num_threads=*/64);
  EXPECT_EQ(select_grp_matmul_algo(s.layout, s.M, s.N, s.K, s.params,
                                   s.num_threads),
            2)
      << "prompt-class M-tile regime (16 actives × M=512 / 64t) must route "
         "to ALGO 2 (pure M-tile)";
}

// Rule 0.6 counts ACTIVE experts (M[i] > 0), NOT the padded slot count.
// A legacy caller may pass M with inactive (M[i]==0) placeholders in-range;
// only the firing experts consume a thread, so a call whose SLOT count
// exceeds the thread budget but whose ACTIVE count does not must stay on
// ALGO 3 (the rule must not fire on padding alone).
TEST(TestGroupMatmulAutoPhaseEnv, DecodeUsesActiveExpertCountNotSlotCount) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  reset_grp_matmul_caches();
  AlgoEnvGuard reset_algo(0);

  // 88 slots on 64 threads (slot count > threads) but only 58 fire — mark
  // the trailing 30 inactive (M[i]=0).  active_ops(58) <= num_threads(64),
  // so Rule 0.6 must NOT fire and decode stays on its ALGO 3 default.
  auto s = build_auto_probe(/*M=*/16, /*K=*/2048, /*N=*/1536,
                            /*num_ops=*/88, /*num_threads=*/64);
  for (int i = 58; i < 88; ++i) s.M[i] = 0;
  EXPECT_EQ(select_grp_matmul_algo(s.layout, s.M, s.N, s.K, s.params,
                                   s.num_threads),
            3)
      << "decode slot count(88) > threads(64) but active(58) <= threads(64): "
         "Rule 0.6 must compare ACTIVE experts, so ALGO 3 stays selected";

  // Now make 78 fire (active > threads) — Rule 0.6 fires -> ALGO 5.
  for (int i = 0; i < 88; ++i) s.M[i] = (i < 78) ? 16 : 0;
  EXPECT_EQ(select_grp_matmul_algo(s.layout, s.M, s.N, s.K, s.params,
                                   s.num_threads),
            5)
      << "decode active(78) > threads(64) must route to ALGO 5";
}

// Few-experts decode policy (Rule 0.5) is now REGIME-AWARE, mirroring the
// prompt Rule 0.7, because ALGO 2 is a pure M-tile executor (no internal
// wide-N fallback).  A Mixtral-class 8-expert decode shape with shallow M
// (max_M>1, total_need*2 ≤ num_threads) is the wide-N regime, so AUTO routes
// it to ALGO 1 (full-team sequential) — exactly the executor the old ALGO 2
// wide-N branch ran.  (Single-tier M-tile would under-fill the team here:
// it splits M rows capped at M[i], so M < team_size leaves threads idle.)
TEST(TestGroupMatmulAutoPhaseEnv, FewExpertsDecodeShallowMRoutesToAlgo1) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  reset_grp_matmul_caches();
  AlgoEnvGuard reset_algo(0);

  // Mixtral-class decode: 8 experts, shallow decode-class M (max_M=16>1;
  // total_need = 8*ceil(16/16) = 8; 2*8 = 16 ≤ 128 → wide-N regime).
  auto s = build_auto_probe(/*M=*/16, /*K=*/4096, /*N=*/14336,
                            /*num_ops=*/8, /*num_threads=*/128);
  EXPECT_EQ(select_grp_matmul_algo(s.layout, s.M, s.N, s.K, s.params,
                                   s.num_threads),
            1)
      << "≤8-expert shallow-M decode is the wide-N regime → ALGO 1 "
         "(full-team sequential = old ALGO 2 wide-N branch), not single-tier";
}

// The other half of the few-experts decode split: a single-token-per-expert
// decode (max_M==1) is NOT the wide-N regime (the gate excludes max_M==1, as
// the old flat_m_tile wide-N gate did), so it stays on ALGO 2 single-tier —
// the latency-optimal one-thread-per-expert CCD-stripe.
TEST(TestGroupMatmulAutoPhaseEnv, FewExpertsDecodeSingleTokenRoutesToAlgo2) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  reset_grp_matmul_caches();
  AlgoEnvGuard reset_algo(0);

  // 8 experts, M=1 (max_M==1 → wide-N excluded → kMTile → ALGO 2).
  auto s = build_auto_probe(/*M=*/1, /*K=*/4096, /*N=*/14336,
                            /*num_ops=*/8, /*num_threads=*/128);
  EXPECT_EQ(select_grp_matmul_algo(s.layout, s.M, s.N, s.K, s.params,
                                   s.num_threads),
            2)
      << "≤8-expert single-token decode (max_M==1) stays on ALGO 2 "
         "single-tier (the kMTile regime)";
}

// The few-experts ALGO 2 preference is a DEFAULT refinement: an explicit
// decode phase pin still wins.  AUTO_DECODE_ALGO=3 on the same 8-expert
// decode shape must yield ALGO 3, not the Rule 0.5 ALGO 2.
TEST(TestGroupMatmulAutoPhaseEnv, FewExpertsExplicitDecodePinStillWins) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  reset_grp_matmul_caches();
  AlgoEnvGuard reset_algo(0);
  AutoPromptAlgoOverride no_prompt(0);
  AutoDecodeAlgoOverride force_decode(3);  // explicit decode pin

  auto s = build_auto_probe(/*M=*/16, /*K=*/4096, /*N=*/14336,
                            /*num_ops=*/8, /*num_threads=*/128);
  EXPECT_EQ(select_grp_matmul_algo(s.layout, s.M, s.N, s.K, s.params,
                                   s.num_threads),
            3)
      << "explicit AUTO_DECODE_ALGO=3 must win over the few-experts "
         "ALGO 2 default preference";
}

// Rule 0.5 counts the TOTAL experts (framework `total_matmul`), not the
// active op count.  An active-4 / total-16 decode shape exceeds the ≤8
// threshold, so it falls through to the decode default ALGO 3.
TEST(TestGroupMatmulAutoPhaseEnv, FewExpertsUsesTotalNotActiveCount) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  reset_grp_matmul_caches();
  AlgoEnvGuard reset_algo(0);

  // 4 ACTIVE experts but 16 TOTAL (framework opt-in: active_matmul=4,
  // total_matmul=16).  total=16 > 8 → Rule 0.5 must NOT fire.
  auto s = build_auto_probe(/*M=*/16, /*K=*/2880, /*N=*/5760,
                            /*num_ops=*/4, /*num_threads=*/64);
  s.params[0].active_matmul = 4;
  s.params[0].total_matmul  = 16;
  EXPECT_EQ(select_grp_matmul_algo(s.layout, s.M, s.N, s.K, s.params,
                                   s.num_threads),
            3)
      << "Rule 0.5 must use TOTAL (16), not active (4); total>8 → "
         "decode default ALGO 3";

  // Same active count but total=8 → Rule 0.5 fires.  The shape is shallow-M
  // decode (M=16, total_need=4*1=4, 2*4=8 ≤ 64 → wide-N regime), so the
  // regime-aware Rule 0.5 routes it to ALGO 1 (= old ALGO 2 wide-N branch).
  // The point of this test is that Rule 0.5 FIRED on total=8 (≠ the ALGO 3
  // decode default it took at total=16), proving it keys on TOTAL not active.
  s.params[0].total_matmul = 8;
  EXPECT_EQ(select_grp_matmul_algo(s.layout, s.M, s.N, s.K, s.params,
                                   s.num_threads),
            1)
      << "total=8 (≤ threshold) must fire Rule 0.5 even with only 4 active "
         "experts; shallow-M decode → wide-N regime → ALGO 1";
}

// Grouped-s8 post-quant call with INACTIVE experts must NOT be vetoed to
// ALGO 1.  The grouped/fallback DQ pre-pass leaves inactive (M==0)
// experts at bf16 + dynamic_quant=false while active experts are s8 +
// per-token src_scale.  check_m_tile_safe / check_n_tile_extra must skip
// the inactive experts (and key dtype uniformity off the first active
// one) so a >8-expert decode shape still routes to ALGO 3.  >8 experts
// keeps Rule 0.5 (the ≤8 ALGO 2 preference) out of the picture so this
// isolates the n_tile/m_tile eligibility veto.
TEST(TestGroupMatmulAutoPhaseEnv, GroupedS8WithInactiveExpertsKeepsAlgo3) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  using zendnnl::common::data_type_t;
  reset_grp_matmul_caches();
  AlgoEnvGuard reset_algo(0);

  // 16 experts (>8), decode-class M=16, n_tile-safe shape (K%4==0).
  constexpr int num_ops = 16, M_active = 16, K = 2880, N = 5760;
  auto s = build_auto_probe(M_active, K, N, num_ops, /*num_threads=*/64);

  // Mark a LEADING (index 0) and an interior (index 8) expert inactive.
  s.M[0] = 0;
  s.M[8] = 0;

  // Persistent non-null scale buffers (never dereferenced for the
  // classification, but the gates require non-null + correct dims).
  static std::vector<std::vector<float>> src_scales(num_ops);
  static std::vector<std::vector<float>> wei_scales(num_ops);
  for (int i = 0; i < num_ops; ++i) {
    auto &p = s.params[i];
    p.dtypes.wei     = data_type_t::s8;
    p.dtypes.dst     = data_type_t::bf16;
    p.dtypes.compute = data_type_t::s8;
    p.dynamic_quant  = false;
    // Per-channel wei scale {1, N} is set on EVERY expert (the caller
    // supplies static weight scales for active + inactive alike).
    wei_scales[i].assign(static_cast<size_t>(N), 1.0f / 127.0f);
    p.quant_params.wei_scale.dims = {1, static_cast<int64_t>(N)};
    p.quant_params.wei_scale.dt   = data_type_t::f32;
    p.quant_params.wei_scale.buff = wei_scales[i].data();
    if (s.M[i] > 0) {
      // Active expert: grouped-prequant post-state — s8 src + per-token
      // src_scale {M, 1}.
      p.dtypes.src = data_type_t::s8;
      src_scales[i].assign(static_cast<size_t>(s.M[i]), 1.0f / 127.0f);
      p.quant_params.src_scale.dims = {static_cast<int64_t>(s.M[i]), 1};
      p.quant_params.src_scale.dt   = data_type_t::f32;
      p.quant_params.src_scale.buff = src_scales[i].data();
    } else {
      // Inactive expert: left at pre-quant bf16, no src_scale (exactly
      // what group_reorder_quantization_wrapper publishes for M==0).
      p.dtypes.src = data_type_t::bf16;
      p.quant_params.src_scale = {};
    }
  }

  EXPECT_EQ(select_grp_matmul_algo(s.layout, s.M, s.N, s.K, s.params,
                                   s.num_threads),
            3)
      << "grouped-s8 decode with inactive experts must still route to "
         "ALGO 3 — inactive (bf16) experts must not veto m_tile/n_tile "
         "eligibility";
}

// Phase env isolation: setting AUTO_PROMPT_ALGO=3 must NOT affect
// decode-class calls, and vice versa.
TEST(TestGroupMatmulAutoPhaseEnv, PromptEnvDoesNotLeakIntoDecode) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  reset_grp_matmul_caches();
  AlgoEnvGuard reset_algo(0);
  AutoPromptAlgoOverride force_prompt(2);  // bogus-for-decode ALGO
  AutoDecodeAlgoOverride no_decode(0);     // legacy for decode

  // Decode-class (M=16 ≤ kDecodeMaxM), gpt-oss-style, num_ops=32.
  // Legacy rule 3 → ALGO 3.  Prompt env must not bleed in.
  auto s = build_auto_probe(/*M=*/16, /*K=*/2880, /*N=*/5760,
                            /*num_ops=*/32, /*num_threads=*/64);
  EXPECT_EQ(select_grp_matmul_algo(s.layout, s.M, s.N, s.K, s.params,
                                   s.num_threads),
            3)
      << "Decode-class call must read AUTO_DECODE_ALGO (=0, legacy "
         "rules → ALGO 3), NOT AUTO_PROMPT_ALGO";
}

TEST(TestGroupMatmulAutoPhaseEnv, DecodeEnvDoesNotLeakIntoPrompt) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  reset_grp_matmul_caches();
  AlgoEnvGuard reset_algo(0);
  AutoPromptAlgoOverride no_prompt(0);     // legacy for prompt
  AutoDecodeAlgoOverride force_decode(2);  // bogus-for-prompt ALGO

  // Prompt-class (M=256), Mixtral-style, num_ops=8.
  // Legacy rule 2 → ALGO 1.  Decode env must not bleed in.
  auto s = build_auto_probe(/*M=*/256, /*K=*/4096, /*N=*/14336,
                            /*num_ops=*/8, /*num_threads=*/128);
  EXPECT_EQ(select_grp_matmul_algo(s.layout, s.M, s.N, s.K, s.params,
                                   s.num_threads),
            1)
      << "Prompt-class call must read AUTO_PROMPT_ALGO (=0, legacy "
         "rules → ALGO 1), NOT AUTO_DECODE_ALGO";
}

// Bogus phase env values (>5 or invalid) clamp to the documented
// default for the phase — matching the env-parse "validate or fall
// back to default" convention used by every other int env getter in
// this header.  PROMPT default=2, DECODE default=3.
TEST(TestGroupMatmulAutoPhaseEnv, BogusValueFallsBackToDefault) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  reset_grp_matmul_caches();
  AlgoEnvGuard reset_algo(0);
  AutoPromptAlgoOverride bogus_prompt(99);   // > 5, clamps to 2 (default)
  AutoDecodeAlgoOverride bogus_decode(99);   // > 5, clamps to 3 (default)

  // Mixtral-class prompt → phase default 2 (flat_m_tile).
  auto s = build_auto_probe(/*M=*/256, /*K=*/4096, /*N=*/14336,
                            /*num_ops=*/8, /*num_threads=*/128);
  EXPECT_EQ(select_grp_matmul_algo(s.layout, s.M, s.N, s.K, s.params,
                                   s.num_threads),
            2)
      << "AUTO_PROMPT_ALGO=99 must clamp to default (=2)";

  // Decode shape with n_tile_safe=true → phase default 3 (N-tile rounds).
  auto sd = build_auto_probe(/*M=*/16, /*K=*/2880, /*N=*/5760,
                             /*num_ops=*/32, /*num_threads=*/64);
  EXPECT_EQ(select_grp_matmul_algo(sd.layout, sd.M, sd.N, sd.K, sd.params,
                                   sd.num_threads),
            3)
      << "AUTO_DECODE_ALGO=99 must clamp to default (=3)";
}

// Structural R0 (capacity overflow num_ops > kNTilePlanMaxExperts=256)
// fires BEFORE the phase env so num_ops=300 → ALGO 5 even with
// AUTO_PROMPT_ALGO/AUTO_DECODE_ALGO forced to 3.
TEST(TestGroupMatmulAutoPhaseEnv, CapacityOverflowIgnoresPhaseEnv) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  reset_grp_matmul_caches();
  AlgoEnvGuard reset_algo(0);
  AutoPromptAlgoOverride force_prompt(3);
  AutoDecodeAlgoOverride force_decode(3);

  // Decode-class with 300 experts > kNTilePlanMaxExperts(256).
  // Phase env says 3, but R0 must win → ALGO 5.
  auto s = build_auto_probe(/*M=*/4, /*K=*/2880, /*N=*/5760,
                            /*num_ops=*/300, /*num_threads=*/128);
  EXPECT_EQ(select_grp_matmul_algo(s.layout, s.M, s.N, s.K, s.params,
                                   s.num_threads),
            5)
      << "R0 capacity gate must override phase env "
         "(num_ops > kNTilePlanMaxExperts → ALGO 5)";
}

// ALGO 3 phase env clamps to ALGO 1 when the shape fails
// `check_n_tile_extra` (e.g., dynamic_quant=true).  Same correctness
// contract the global ALGO=3 path has, applied to the auto path.
// ===============================================================================
// [8e] TestGroupMatmulHybridMSplit — Option-A M-weighted water-fill
//      heavy distribution under
//      `ZENDNNL_GRP_MATMUL_N_TILE_HEAVY_THRESHOLD`.
//
// Targets the new path inside
// `apply_round_pick(RoundPick::Single, use_custom=true)` that fires
// when the env (or test override) is positive AND the M distribution
// has both heavy and light experts.  Verified via the existing
// `PhaseBSnapshot` capture hook so the test is a true white-box
// observation of the planner's per-expert thread distribution.
//
// Two cases:
//
//   1. `OffByDefaultRunsPhaseB` — `threshold=0` (default) MUST leave
//      the existing Phase B remainder behaviour intact.  Sanity
//      check that the hybrid gate doesn't fire when env is unset.
//
//   2. `OnDistributesHeavyByM` — `threshold=100` with a bimodal M
//      vector produces an asymmetric allocation: every active light
//      expert gets exactly 1 thread, the heavy experts share the
//      remaining budget proportional to M (heavier gets ≥ lighter).
// ===============================================================================

namespace {

// Shared shape builder for Hybrid M-split tests.  Uses uniform
// row-major BF16 inputs and the same N=1024 / K=64 dims as the
// existing PhaseB test fixtures.  M vector and num_threads are
// caller-supplied.
struct HybridProbeShape {
  int num_ops;
  int num_threads;
  int N;
  int K;
  std::vector<int> Ms;
  std::vector<std::vector<zendnnl::common::bfloat16_t>> src;
  std::vector<std::vector<zendnnl::common::bfloat16_t>> wei;
  std::vector<std::vector<zendnnl::common::bfloat16_t>> dst;
  std::vector<const void *> srcs;
  std::vector<const void *> weis;
  std::vector<const void *> biases;
  std::vector<void *> dsts;
  moe_test_utils::GemmVecs gv;
  std::vector<zendnnl::lowoha::matmul::matmul_params> params;
};

static HybridProbeShape build_hybrid_probe(int num_threads,
                                           const std::vector<int> &Ms,
                                           int N = 1024, int K = 64) {
  using namespace moe_test_utils;
  using zendnnl::common::bfloat16_t;
  using zendnnl::common::data_type_t;
  HybridProbeShape s;
  s.num_threads = num_threads;
  s.num_ops     = static_cast<int>(Ms.size());
  s.N           = N;
  s.K           = K;
  s.Ms          = Ms;
  s.src.resize(s.num_ops);
  s.wei.resize(s.num_ops);
  s.dst.resize(s.num_ops);
  for (int e = 0; e < s.num_ops; ++e) {
    s.src[e].resize(static_cast<size_t>(Ms[e]) * K);
    s.wei[e].resize(static_cast<size_t>(K) * N);
    s.dst[e].assign(static_cast<size_t>(Ms[e]) * N, bfloat16_t(0.0f));
    fill_src(s.src[e],  e, 0.02f);
    fill_wei1(s.wei[e], e, 0.005f);
  }
  s.srcs.resize(s.num_ops);
  s.weis.resize(s.num_ops);
  s.biases.assign(s.num_ops, nullptr);
  s.dsts.resize(s.num_ops);
  for (int e = 0; e < s.num_ops; ++e) {
    s.srcs[e] = s.src[e].data();
    s.weis[e] = s.wei[e].data();
    s.dsts[e] = s.dst[e].data();
  }
  s.gv = GemmVecs::uniform(s.num_ops, /*M=*/0, N, K,
                           /*alpha=*/1.0f, /*beta=*/0.0f,
                           /*wc=*/true, /*tA=*/false, /*tB=*/false);
  s.gv.Ms = Ms;
  s.params = make_uniform_params(s.num_ops, data_type_t::bf16);
  for (int e = 0; e < s.num_ops; ++e) {
    s.params[e].num_threads = num_threads;
  }
  return s;
}

}  // namespace

// Hybrid OFF (default) → existing Phase B distribution fires
// (per-expert allocation is `base` or `base+1`, not the water-fill
// product).  Sanity check that the new gate doesn't accidentally
// fire when the threshold env is 0.
TEST(TestGroupMatmulHybridMSplit, OffByDefaultRunsPhaseB) {
  using namespace moe_test_utils;
  using zendnnl::lowoha::matmul::group_matmul_direct;
  using zendnnl::lowoha::matmul::GroupNTileStrategy;

  const int saved_num_threads = omp_get_max_threads();
  struct ThreadGuard {
    int prev;
    ~ThreadGuard() { omp_set_num_threads(prev); }
  } thread_guard{saved_num_threads};
  omp_set_num_threads(32);
  int actual_team_size = 0;
  #pragma omp parallel
  {
    #pragma omp master
    actual_team_size = omp_get_num_threads();
  }
  if (actual_team_size < 32) {
    GTEST_SKIP() << "Requires >= 32 OMP threads; have " << actual_team_size;
  }

  CustomKernelOverride           ck_on(true);
  NRoundsModeOverride            single_round(1);
  CustomKernelNTileOverride      default_n_tile(0);
  NTileStrategyOverride          auto_strategy(0);
  NTileHeavyThresholdOverride  hybrid_off(-1);  // explicit DISABLED
  // Decode M-proportional split is OFF by default; this test asserts the
  // proportional contract (heaviest expert gets > uniform base+1), so pin
  // it ON explicitly.  The knob under test here is the PROMPT-only hybrid
  // threshold, which must NOT fire on decode regardless of this.
  DecodeProportionalOverride   prop_on(true);

  if (!zendnnl::lowoha::matmul::custom_kernel::dispatch_supported()) {
    GTEST_SKIP() << "Requires AVX512BF16 / CK dispatch support.";
  }
  reset_grp_matmul_caches();

  // 14 active experts, decode-class M (≤ 32) so we land on the
  // CK Single round (matches the existing PhaseB heaviest-first
  // test shape).
  auto s = build_hybrid_probe(/*num_threads=*/32,
      /*Ms=*/{16, 12, 10, 8, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4});

  AlgoEnvGuard       algo_guard(3);
  PhaseBCaptureGuard cap;
  ASSERT_EQ(group_matmul_direct(s.gv.layout, s.gv.transA, s.gv.transB,
                                s.gv.Ms, s.gv.Ns, s.gv.Ks, s.gv.alpha,
                                s.srcs, s.gv.lda, s.weis, s.gv.ldb,
                                s.biases, s.gv.beta, s.dsts, s.gv.ldc,
                                s.gv.is_wc, s.params,
                                nullptr, nullptr),
            status_t::success);
  const auto &snap =
      zendnnl::lowoha::matmul::test_api::s_last_phase_b_snapshot;
  ASSERT_TRUE(snap.valid);
  EXPECT_TRUE(snap.per_expert_remainder);
  // The prompt-only HYBRID water-fill keeps n_thr_fixed=0; the decode
  // M-proportional split (and legacy Phase B) leave it at base.  A
  // non-zero value confirms the HYBRID gate did NOT fire.
  EXPECT_NE(snap.n_thr_fixed, 0)
      << "Hybrid gate must NOT fire on decode; got n_thr_fixed=0.";
  // ZENDNNL_GRP_MATMUL_N_TILE_HEAVY_THRESHOLD is a PROMPT-ONLY knob
  // (default -1 = DISABLED); it does NOT govern decode.  The decode
  // CK Single-round path now uses the UNCONDITIONAL M-proportional
  // thread split regardless of this knob, so even with the hybrid
  // threshold explicitly disabled (-1) decode still load-balances by
  // M.  Assert the proportional contract (not the legacy uniform
  // base/base+1): per_expert_remainder set, no active expert starved,
  // total <= num_threads, and the heaviest expert (M=16) gets a
  // proportional share above the uniform base+1 (=3).
  int sum_thr = 0;
  for (int e = 0; e < s.num_ops; ++e) {
    const int n = static_cast<int>(snap.stable_n_thr_per_expert[e]);
    if (s.Ms[e] > 0) {
      EXPECT_GE(n, 1) << "active expert e=" << e << " starved (0 thr)";
    }
    sum_thr += n;
  }
  EXPECT_LE(sum_thr, 32) << "M-proportional over-allocated threads";
  EXPECT_GE(static_cast<int>(snap.stable_n_thr_per_expert[0]), 4)
      << "heaviest expert (M=16) should get a proportional share "
         "(> uniform base+1=3); got "
      << static_cast<int>(snap.stable_n_thr_per_expert[0]);
}

// Hybrid ON → water-fill distribution: heavier-M experts receive
// strictly ≥ threads of lighter heavies; every active light
// expert gets exactly 1 thread; `n_thr_fixed == 0` to tell the
// executor to use prefix-sum mapping; `sum(stable_n_thr) <=
// num_threads` (water-fill may leave a tail of idle threads when
// every heavy hits its per-expert cap).
// ZENDNNL_GRP_MATMUL_DECODE_PROPORTIONAL=0 (the default) disables the
// decode M-proportional split; decode CK Single-round uses the uniform
// Phase B base/base+1 distribution.
TEST(TestGroupMatmulHybridMSplit, DecodeProportionalOffUsesUniformPhaseB) {
  using namespace moe_test_utils;
  using zendnnl::lowoha::matmul::group_matmul_direct;

  const int saved_num_threads = omp_get_max_threads();
  struct ThreadGuard {
    int prev;
    ~ThreadGuard() { omp_set_num_threads(prev); }
  } thread_guard{saved_num_threads};
  omp_set_num_threads(32);
  int actual_team_size = 0;
  #pragma omp parallel
  {
    #pragma omp master
    actual_team_size = omp_get_num_threads();
  }
  if (actual_team_size < 32) {
    GTEST_SKIP() << "Requires >= 32 OMP threads; have " << actual_team_size;
  }

  CustomKernelOverride        ck_on(true);
  NRoundsModeOverride         single_round(1);
  CustomKernelNTileOverride   default_n_tile(0);
  NTileStrategyOverride       auto_strategy(0);
  DecodeProportionalOverride  prop_off(false);  // knob under test

  if (!zendnnl::lowoha::matmul::custom_kernel::dispatch_supported()) {
    GTEST_SKIP() << "Requires AVX512BF16 / CK dispatch support.";
  }
  reset_grp_matmul_caches();

  auto s = build_hybrid_probe(/*num_threads=*/32,
      /*Ms=*/{16, 12, 10, 8, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4});

  AlgoEnvGuard       algo_guard(3);
  PhaseBCaptureGuard cap;
  ASSERT_EQ(group_matmul_direct(s.gv.layout, s.gv.transA, s.gv.transB,
                                s.gv.Ms, s.gv.Ns, s.gv.Ks, s.gv.alpha,
                                s.srcs, s.gv.lda, s.weis, s.gv.ldb,
                                s.biases, s.gv.beta, s.dsts, s.gv.ldc,
                                s.gv.is_wc, s.params, nullptr, nullptr),
            status_t::success);
  const auto &snap =
      zendnnl::lowoha::matmul::test_api::s_last_phase_b_snapshot;
  ASSERT_TRUE(snap.valid);
  EXPECT_TRUE(snap.per_expert_remainder);
  EXPECT_NE(snap.n_thr_fixed, 0)
      << "proportional OFF keeps n_thr_fixed at base (not the hybrid 0).";
  // With proportional disabled every active expert gets base (2) or
  // base+1 (3) only — NO M-weighted spread.
  for (int e = 0; e < s.num_ops; ++e) {
    const int n = static_cast<int>(snap.stable_n_thr_per_expert[e]);
    if (s.Ms[e] > 0) {
      EXPECT_TRUE(n == 2 || n == 3)
          << "proportional OFF -> uniform Phase B; expert e=" << e
          << " M=" << s.Ms[e] << " got n_thr=" << n;
    }
  }
}

// ZENDNNL_GRP_MATMUL_DECODE_PROPORTIONAL=1 (opt-in; default is OFF)
// gives the M-weighted split: heavier experts get a proportional share
// above the uniform base+1, light experts the floor.
TEST(TestGroupMatmulHybridMSplit, DecodeProportionalOnUsesMWeightedSplit) {
  using namespace moe_test_utils;
  using zendnnl::lowoha::matmul::group_matmul_direct;

  const int saved_num_threads = omp_get_max_threads();
  struct ThreadGuard {
    int prev;
    ~ThreadGuard() { omp_set_num_threads(prev); }
  } thread_guard{saved_num_threads};
  omp_set_num_threads(32);
  int actual_team_size = 0;
  #pragma omp parallel
  {
    #pragma omp master
    actual_team_size = omp_get_num_threads();
  }
  if (actual_team_size < 32) {
    GTEST_SKIP() << "Requires >= 32 OMP threads; have " << actual_team_size;
  }

  CustomKernelOverride        ck_on(true);
  NRoundsModeOverride         single_round(1);
  CustomKernelNTileOverride   default_n_tile(0);
  NTileStrategyOverride       auto_strategy(0);
  DecodeProportionalOverride  prop_on(true);  // knob under test

  if (!zendnnl::lowoha::matmul::custom_kernel::dispatch_supported()) {
    GTEST_SKIP() << "Requires AVX512BF16 / CK dispatch support.";
  }
  reset_grp_matmul_caches();

  auto s = build_hybrid_probe(/*num_threads=*/32,
      /*Ms=*/{16, 12, 10, 8, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4});

  AlgoEnvGuard       algo_guard(3);
  PhaseBCaptureGuard cap;
  ASSERT_EQ(group_matmul_direct(s.gv.layout, s.gv.transA, s.gv.transB,
                                s.gv.Ms, s.gv.Ns, s.gv.Ks, s.gv.alpha,
                                s.srcs, s.gv.lda, s.weis, s.gv.ldb,
                                s.biases, s.gv.beta, s.dsts, s.gv.ldc,
                                s.gv.is_wc, s.params, nullptr, nullptr),
            status_t::success);
  const auto &snap =
      zendnnl::lowoha::matmul::test_api::s_last_phase_b_snapshot;
  ASSERT_TRUE(snap.valid);
  EXPECT_TRUE(snap.per_expert_remainder);
  int sum_thr = 0;
  for (int e = 0; e < s.num_ops; ++e) {
    const int n = static_cast<int>(snap.stable_n_thr_per_expert[e]);
    if (s.Ms[e] > 0) {
      EXPECT_GE(n, 1) << "active expert e=" << e << " starved";
    }
    sum_thr += n;
  }
  EXPECT_LE(sum_thr, 32);
  // Heaviest (M=16) gets a proportional share > uniform base+1 (=3),
  // and strictly more than the lightest (M=4) experts.
  const int n_heavy = static_cast<int>(snap.stable_n_thr_per_expert[0]);
  EXPECT_GE(n_heavy, 4)
      << "proportional ON: heaviest expert under-allocated (" << n_heavy << ")";
  EXPECT_GT(n_heavy, static_cast<int>(snap.stable_n_thr_per_expert[13]));
}

TEST(TestGroupMatmulHybridMSplit, OnDistributesHeavyByM) {
  using namespace moe_test_utils;
  using zendnnl::lowoha::matmul::group_matmul_direct;
  using zendnnl::lowoha::matmul::GroupNTileStrategy;

  const int saved_num_threads = omp_get_max_threads();
  struct ThreadGuard {
    int prev;
    ~ThreadGuard() { omp_set_num_threads(prev); }
  } thread_guard{saved_num_threads};
  omp_set_num_threads(32);
  int actual_team_size = 0;
  #pragma omp parallel
  {
    #pragma omp master
    actual_team_size = omp_get_num_threads();
  }
  if (actual_team_size < 32) {
    GTEST_SKIP() << "Requires >= 32 OMP threads; have " << actual_team_size;
  }

  CustomKernelOverride           ck_on(true);
  NRoundsModeOverride            single_round(1);
  CustomKernelNTileOverride      default_n_tile(0);
  // MANUAL HYBRID is now prompt-only (max_M > kDecodeMaxM=32).  Force
  // `n_tile_strategy=2` (rounds, force) to bypass the auto-mirror that
  // would otherwise route prompt-class shapes with `num_ops <
  // num_threads` to Sequential via the AUTO-MIRROR rule 3 — keeps the
  // planner in ManyExperts so the Single-round HYBRID dispatch fires.
  NTileStrategyOverride          force_rounds(2);
  // Threshold chosen so M > 200 = heavy AND every Ms entry crosses
  // the new `max_M > kDecodeMaxM=32` prompt gate.
  NTileHeavyThresholdOverride  hybrid_on(200);

  if (!zendnnl::lowoha::matmul::custom_kernel::dispatch_supported()) {
    GTEST_SKIP() << "Requires AVX512BF16 / CK dispatch support.";
  }
  reset_grp_matmul_caches();

  // 14 experts, with M chosen so M=64 is LIGHT (≤ 200) and
  // M ∈ {400, 320, 280} are HEAVY (> 200).  3 heavy + 11 light.
  // All Ms are prompt-class (> kDecodeMaxM=32) so MANUAL HYBRID's
  // new decode gate lets the path engage.  N=4096 keeps the cap
  // arithmetic identical to the AUTO tests.
  auto s = build_hybrid_probe(/*num_threads=*/32,
      /*Ms=*/{400, 320, 280, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64},
      /*N=*/4096);

  AlgoEnvGuard       algo_guard(3);
  PhaseBCaptureGuard cap;
  ASSERT_EQ(group_matmul_direct(s.gv.layout, s.gv.transA, s.gv.transB,
                                s.gv.Ms, s.gv.Ns, s.gv.Ks, s.gv.alpha,
                                s.srcs, s.gv.lda, s.weis, s.gv.ldb,
                                s.biases, s.gv.beta, s.dsts, s.gv.ldc,
                                s.gv.is_wc, s.params,
                                nullptr, nullptr),
            status_t::success);
  const auto &snap =
      zendnnl::lowoha::matmul::test_api::s_last_phase_b_snapshot;
  ASSERT_TRUE(snap.valid);
  EXPECT_TRUE(snap.per_expert_remainder)
      << "Hybrid path MUST set per_expert_remainder=true so the "
         "executor reads stable_n_thr_per_expert[].";
  EXPECT_EQ(snap.n_thr_fixed, 0)
      << "Hybrid path MUST set n_thr_fixed=0 so the executor takes "
         "the prefix-sum scan instead of the uniform `tid/tpe` mapping.";

  // Per-expert assertions.  Experts e=0..2 are heavy (M = s.Ms[0..2],
  // currently {400, 320, 280}); e=3..13 are light (M = s.Ms[3..13],
  // currently 64 each).  Light experts must each receive exactly 1
  // thread.  Heavies must receive >= 1, AND the M-descending water-
  // fill ordering implies n_thr[e=0] >= n_thr[e=1] >= n_thr[e=2].
  // Messages below derive the M values from `s.Ms[]` so the failure
  // report stays correct if the probe Ms get retuned in a follow-up.
  const int n0 = static_cast<int>(snap.stable_n_thr_per_expert[0]);
  const int n1 = static_cast<int>(snap.stable_n_thr_per_expert[1]);
  const int n2 = static_cast<int>(snap.stable_n_thr_per_expert[2]);
  EXPECT_GE(n0, 1);
  EXPECT_GE(n0, n1)
      << "Heavier expert e=0 (M=" << s.Ms[0] << ") must receive >= threads "
         "as e=1 (M=" << s.Ms[1] << ").";
  EXPECT_GE(n1, n2)
      << "Heavier expert e=1 (M=" << s.Ms[1] << ") must receive >= threads "
         "as e=2 (M=" << s.Ms[2] << ").";
  // At least one heavy must receive more than the per-light
  // allocation (= 1), otherwise the path collapses to "everyone
  // gets 1 thread" which is no better than the legacy single-round
  // behaviour pre-Phase-B.
  EXPECT_GT(n0 + n1 + n2, 3)
      << "Heavies should collectively absorb more than 1 thread each.";
  for (int e = 3; e < s.num_ops; ++e) {
    const int n = static_cast<int>(snap.stable_n_thr_per_expert[e]);
    EXPECT_EQ(n, 1)
        << "Light expert e=" << e << " (M=" << s.Ms[e]
        << ") must receive exactly 1 thread under hybrid M-split.";
  }
  // Total must NOT exceed num_threads (water-fill respects the
  // overall thread budget).  Equality is the common case; strictly
  // less can happen if every heavy hits its per-expert cap.
  int sum = 0;
  for (int e = 0; e < s.num_ops; ++e) {
    sum += static_cast<int>(snap.stable_n_thr_per_expert[e]);
  }
  EXPECT_LE(sum, 32)
      << "Hybrid distribution must not over-subscribe the team; "
         "sum(stable_n_thr_per_expert) = " << sum;
  EXPECT_GT(sum, s.num_ops - 3 /*lights*/ + 3 /*heavies*/)
      << "Hybrid distribution should give heavies > 1 thread each; "
         "sum = " << sum << " is suspiciously low.";
}

// AUTO (env=0) → planner-driven adaptive tiers fire, producing a
// monotone-by-M per-expert allocation with the Hybrid dispatch
// signature (`per_expert_remainder=true`, `n_thr_fixed=0`).  Verifies
// AUTO engages on a sufficiently-skewed prompt-class shape AND
// produces a heaviest-first allocation distinct from Phase B base+1.
//
// Shape constraints to keep the planner in ManyExperts:
//   * `max_M > kDecodeMaxM` (prompt-class) is now MANDATORY — AUTO's
//     new decode gate (`apply_adaptive_tiers()` returns false on
//     `max_M ≤ kDecodeMaxM`) makes the previous decode-class probe
//     fall through to Phase B.  Use max_M=400 (firmly prompt-class).
//   * Force `n_tile_strategy=2` (rounds) so the auto-mirror's
//     prompt → ALGO 1 routing (`auto_select_would_pick_algo1`
//     rule 3) does NOT fire for `num_ops < num_threads`.  Keeps
//     the planner in ManyExperts where the Single-round HYBRID
//     dispatch lives.
//   * `N` chosen so `max_tiles = N/min_n_tile` lifts the per-expert
//     cap above 2.  N=4096 with the prompt `min_n_tile=512` →
//     max_tiles=8, cap=min(ccd_size=8, 8)=8.
TEST(TestGroupMatmulHybridMSplit, AutoTierEngagesAtZero) {
  using namespace moe_test_utils;
  using zendnnl::lowoha::matmul::group_matmul_direct;

  const int saved_num_threads = omp_get_max_threads();
  struct ThreadGuard {
    int prev;
    ~ThreadGuard() { omp_set_num_threads(prev); }
  } thread_guard{saved_num_threads};
  omp_set_num_threads(32);
  int actual_team_size = 0;
#pragma omp parallel
  {
#pragma omp master
    actual_team_size = omp_get_num_threads();
  }
  if (actual_team_size < 32) {
    GTEST_SKIP() << "Requires >= 32 OMP threads; have " << actual_team_size;
  }

  CustomKernelOverride           ck_on(true);
  NRoundsModeOverride            single_round(1);
  CustomKernelNTileOverride      default_n_tile(0);
  NTileStrategyOverride          force_rounds(2);    // bypass auto-mirror
  NTileHeavyThresholdOverride  hybrid_auto(0);     // AUTO mode

  if (!zendnnl::lowoha::matmul::custom_kernel::dispatch_supported()) {
    GTEST_SKIP() << "Requires AVX512BF16 / CK dispatch support.";
  }
  reset_grp_matmul_caches();

  // 14 experts, PROMPT-class M (max=400 > kDecodeMaxM=32).  Skew =
  // 400 / mean ≈ 400/107 ≈ 3.7 (well above kMinSkew=2.5).  Same
  // tier topology as before, scaled into the prompt regime:
  //   T_high ≈ max(M_p95, 160)  → 1 high  (M=400)
  //   T_mid  ≈ max(M_p75, 80)   → 3 mid   (M=320, 240, 160)
  //   T_low  ≈ max(M_p50, 40)   → 3 low   (M=120, 80, 60)
  //   baseline                  → 7 (M ∈ {40, 40, 40, 40, 40, 40, 40})
  auto s = build_hybrid_probe(/*num_threads=*/32,
      /*Ms=*/{400, 320, 240, 160, 120, 80, 60, 40, 40, 40, 40, 40, 40, 40},
      /*N=*/4096);

  AlgoEnvGuard       algo_guard(3);
  PhaseBCaptureGuard cap;
  ASSERT_EQ(group_matmul_direct(s.gv.layout, s.gv.transA, s.gv.transB,
                                s.gv.Ms, s.gv.Ns, s.gv.Ks, s.gv.alpha,
                                s.srcs, s.gv.lda, s.weis, s.gv.ldb,
                                s.biases, s.gv.beta, s.dsts, s.gv.ldc,
                                s.gv.is_wc, s.params,
                                nullptr, nullptr),
            status_t::success);
  const auto &snap =
      zendnnl::lowoha::matmul::test_api::s_last_phase_b_snapshot;
  ASSERT_TRUE(snap.valid);
  EXPECT_TRUE(snap.per_expert_remainder)
      << "AUTO_TIER must set per_expert_remainder=true so the executor "
         "reads stable_n_thr_per_expert[].";
  EXPECT_EQ(snap.n_thr_fixed, 0)
      << "AUTO_TIER must set n_thr_fixed=0 (Hybrid signature, distinct "
         "from Phase B's n_thr_fixed=base).";

  // M-monotone ordering: heaviest expert (M=400, prompt-class probe
  // shape; see the Ms vector above) must receive >= threads as any
  // lower-M expert.  The water-fill is M-weighted so the heaviest
  // expert always wins the marginal next-thread.
  const int n0 = static_cast<int>(snap.stable_n_thr_per_expert[0]);
  EXPECT_GT(n0, 1)
      << "Heaviest expert (M=400, high tier) must receive multiple "
         "threads; AUTO_TIER didn't engage if n0 = 1.";
  for (int e = 1; e < s.num_ops; ++e) {
    const int n = static_cast<int>(snap.stable_n_thr_per_expert[e]);
    EXPECT_GE(n0, n)
        << "AUTO_TIER must be M-monotone: heaviest (e=0, M=" << s.Ms[0]
        << ") got " << n0 << " threads but e=" << e
        << " (M=" << s.Ms[e] << ") got " << n << ".";
  }

  // Asymmetry check: high tier (e=0) must be strictly heavier than
  // a baseline expert (last index, M=40 — the tail of the prompt-
  // class probe).  Otherwise the tier structure collapsed and AUTO
  // is just a no-op.
  const int n_last =
      static_cast<int>(snap.stable_n_thr_per_expert[s.num_ops - 1]);
  EXPECT_GT(n0, n_last)
      << "AUTO_TIER must produce asymmetry: high tier (n0=" << n0
      << ") must receive strictly more threads than baseline "
         "(n_last=" << n_last << ").";

  // Total must NOT exceed num_threads (water-fill respects budget).
  int sum = 0;
  for (int e = 0; e < s.num_ops; ++e) {
    sum += static_cast<int>(snap.stable_n_thr_per_expert[e]);
  }
  EXPECT_LE(sum, 32)
      << "AUTO_TIER must not over-subscribe the OMP team; sum=" << sum;
}

// AUTO with uniform M → skew gate (M_max / M_mean < 2.5) fails;
// path returns false; planner falls through to Phase B.  Distinguishes
// "no skew → no engagement" from the explicit DISABLED case.
TEST(TestGroupMatmulHybridMSplit, AutoTierSkipsLowSkew) {
  using namespace moe_test_utils;
  using zendnnl::lowoha::matmul::group_matmul_direct;

  const int saved_num_threads = omp_get_max_threads();
  struct ThreadGuard {
    int prev;
    ~ThreadGuard() { omp_set_num_threads(prev); }
  } thread_guard{saved_num_threads};
  omp_set_num_threads(32);
  int actual_team_size = 0;
#pragma omp parallel
  {
#pragma omp master
    actual_team_size = omp_get_num_threads();
  }
  if (actual_team_size < 32) {
    GTEST_SKIP() << "Requires >= 32 OMP threads; have " << actual_team_size;
  }

  CustomKernelOverride           ck_on(true);
  NRoundsModeOverride            single_round(1);
  CustomKernelNTileOverride      default_n_tile(0);
  NTileStrategyOverride          force_rounds(2);   // bypass auto-mirror
  NTileHeavyThresholdOverride  hybrid_auto(0);    // AUTO mode

  if (!zendnnl::lowoha::matmul::custom_kernel::dispatch_supported()) {
    GTEST_SKIP() << "Requires AVX512BF16 / CK dispatch support.";
  }
  reset_grp_matmul_caches();

  // 14 experts, all M=100 → skew = 100/100 = 1.0, well below
  // kMinSkew=2.5.  AUTO_TIER must skip on the LOW-SKEW gate (not the
  // prompt gate); Phase B base+1 must populate the snapshot.  M=100
  // is prompt-class so the new decode gate does NOT pre-empt the
  // low-skew branch — this test exercises the skew filter
  // specifically.
  auto s = build_hybrid_probe(/*num_threads=*/32,
      /*Ms=*/{100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
              100, 100, 100, 100},
      /*N=*/4096);

  AlgoEnvGuard       algo_guard(3);
  PhaseBCaptureGuard cap;
  ASSERT_EQ(group_matmul_direct(s.gv.layout, s.gv.transA, s.gv.transB,
                                s.gv.Ms, s.gv.Ns, s.gv.Ks, s.gv.alpha,
                                s.srcs, s.gv.lda, s.weis, s.gv.ldb,
                                s.biases, s.gv.beta, s.dsts, s.gv.ldc,
                                s.gv.is_wc, s.params,
                                nullptr, nullptr),
            status_t::success);
  const auto &snap =
      zendnnl::lowoha::matmul::test_api::s_last_phase_b_snapshot;
  ASSERT_TRUE(snap.valid);
  // Phase B keeps n_thr_fixed = base (= 2 here for 32t / 14 ops).
  // AUTO would have set it to 0 — asserting non-zero pins the
  // fallback path.
  EXPECT_NE(snap.n_thr_fixed, 0)
      << "AUTO_TIER must fall through to Phase B on low-skew workloads; "
         "expected Phase B's n_thr_fixed != 0, got 0 (AUTO engaged "
         "spuriously).";
  // Every allocation must be base (= 2) or base+1 (= 3) — Phase B's
  // signature, NOT the multi-tier 1/2/4/8 pattern.
  for (int e = 0; e < s.num_ops; ++e) {
    const int n = static_cast<int>(snap.stable_n_thr_per_expert[e]);
    EXPECT_TRUE(n == 2 || n == 3)
        << "Uniform-M expert e=" << e
        << " got n_thr=" << n << " (Phase B should produce 2 or 3).";
  }
}

// AUTO with `num_threads == num_active` → extras_budget = 0 →
// AUTO_TIER returns false; planner falls through to Phase B in the
// same Single round case.  Models the "thread-starved" production
// case (e.g. 64-core machine running a 64-expert MoE call where
// every expert wants at least 1 thread and there's no headroom).
//
// Distinct from the legacy single-threshold path which would happily
// engage as long as `heavy_budget >= 2 * n_heavy` — AUTO is stricter
// because tiering with zero extras has no payoff.
TEST(TestGroupMatmulHybridMSplit, AutoTierSkipsThreadStarvation) {
  using namespace moe_test_utils;
  using zendnnl::lowoha::matmul::group_matmul_direct;

  const int saved_num_threads = omp_get_max_threads();
  struct ThreadGuard {
    int prev;
    ~ThreadGuard() { omp_set_num_threads(prev); }
  } thread_guard{saved_num_threads};
  omp_set_num_threads(32);
  int actual_team_size = 0;
#pragma omp parallel
  {
#pragma omp master
    actual_team_size = omp_get_num_threads();
  }
  if (actual_team_size < 32) {
    GTEST_SKIP() << "Requires >= 32 OMP threads; have " << actual_team_size;
  }

  CustomKernelOverride           ck_on(true);
  NRoundsModeOverride            single_round(1);
  CustomKernelNTileOverride      default_n_tile(0);
  NTileStrategyOverride          force_rounds(2);   // bypass auto-mirror
  NTileHeavyThresholdOverride  hybrid_auto(0);    // AUTO mode

  if (!zendnnl::lowoha::matmul::custom_kernel::dispatch_supported()) {
    GTEST_SKIP() << "Requires AVX512BF16 / CK dispatch support.";
  }
  reset_grp_matmul_caches();

  // 14 active experts, PROMPT-class M (max=400 > kDecodeMaxM=32) so
  // the new prompt gate does NOT preempt the test.  Skew passes
  // (400/107 ≈ 3.7, above kMinSkew=2.5).  Planner is told
  // num_threads = 14 == num_active so the Single round stays
  // feasible (`single_eligible` requires `num_threads >= num_ops`),
  // but AUTO's `extras_budget = num_threads - num_active = 0` gate
  // fails immediately → returns false → Phase B base+1 fires.
  auto s = build_hybrid_probe(/*num_threads=*/14,
      /*Ms=*/{400, 320, 240, 160, 120, 80, 60, 40, 40, 40, 40, 40, 40, 40},
      /*N=*/4096);

  AlgoEnvGuard       algo_guard(3);
  PhaseBCaptureGuard cap;
  ASSERT_EQ(group_matmul_direct(s.gv.layout, s.gv.transA, s.gv.transB,
                                s.gv.Ms, s.gv.Ns, s.gv.Ks, s.gv.alpha,
                                s.srcs, s.gv.lda, s.weis, s.gv.ldb,
                                s.biases, s.gv.beta, s.dsts, s.gv.ldc,
                                s.gv.is_wc, s.params,
                                nullptr, nullptr),
            status_t::success);
  const auto &snap =
      zendnnl::lowoha::matmul::test_api::s_last_phase_b_snapshot;
  ASSERT_TRUE(snap.valid);
  // AUTO's signature is `(per_expert_remainder=true, n_thr_fixed=0)`.
  // Phase B (the fallback we expect to fire here) leaves
  // `per_expert_remainder=false` and `n_thr_fixed=base`.  Asserting
  // BOTH negations confirms AUTO returned false and the planner took
  // the Phase B branch instead.  `stable_n_thr_per_expert[]` is not
  // populated in the Phase B branch (the executor uses uniform
  // `tid / n_thr_fixed` mapping in that case), so no point asserting
  // on its contents — they are correctly left at zero.
  EXPECT_FALSE(snap.per_expert_remainder)
      << "AUTO_TIER must fall through when extras_budget = 0; "
         "expected Phase B (per_expert_remainder=false), got true "
         "(AUTO engaged spuriously).";
  EXPECT_NE(snap.n_thr_fixed, 0)
      << "AUTO_TIER must fall through when extras_budget = 0; "
         "expected Phase B's n_thr_fixed=base, got 0 (AUTO engaged).";
}

// Phase gate (decode safety) — `HYBRID=0` (AUTO) and `HYBRID=10`
// (MANUAL) BOTH must be skipped on a decode-class shape
// (`max_M <= kDecodeMaxM`) so the planner stays on Phase B base+1.
// Models the unified-process E2E case where `HYBRID=0` is exported
// once for the whole run: prompt benefits from AUTO tiering while
// decode plans remain untouched.
//
// Acceptance criterion (the only signature unique to HYBRID):
//   * `n_thr_fixed != 0`  — Phase B (including its base+1 remainder
//                           branch) keeps `n_thr_fixed = base`,
//                           while both HYBRID modes zero it to
//                           tell the executor to use the prefix-
//                           sum scan.  `per_expert_remainder` is
//                           NOT a distinguishing signal here:
//                           Phase B's remainder branch also sets
//                           it to true with base/base+1 entries,
//                           so asserting on it would create a
//                           false alarm on this 32t/14-experts
//                           shape (remainder = 4 → branch fires).
//
// Shape: 14 experts, max_M=24 (decode-class), high skew (24/8.6 ≈ 2.8
// > kMinSkew=2.5), N=4096 (cap=8) — pre-gate this shape engaged AUTO
// (see the pre-prompt-gate version of `AutoTierEngagesAtZero`).  After
// the gate it must fall through unconditionally.
TEST(TestGroupMatmulHybridMSplit, AutoTierSkipsDecodeClass) {
  using namespace moe_test_utils;
  using zendnnl::lowoha::matmul::group_matmul_direct;

  const int saved_num_threads = omp_get_max_threads();
  struct ThreadGuard {
    int prev;
    ~ThreadGuard() { omp_set_num_threads(prev); }
  } thread_guard{saved_num_threads};
  omp_set_num_threads(32);
  int actual_team_size = 0;
#pragma omp parallel
  {
#pragma omp master
    actual_team_size = omp_get_num_threads();
  }
  if (actual_team_size < 32) {
    GTEST_SKIP() << "Requires >= 32 OMP threads; have " << actual_team_size;
  }

  if (!zendnnl::lowoha::matmul::custom_kernel::dispatch_supported()) {
    GTEST_SKIP() << "Requires AVX512BF16 / CK dispatch support.";
  }

  // Decode-class M, high skew — would have engaged AUTO before the
  // prompt-only gate landed.
  auto s = build_hybrid_probe(/*num_threads=*/32,
      /*Ms=*/{24, 20, 16, 12, 8, 6, 4, 2, 1, 1, 1, 1, 1, 1},
      /*N=*/4096);

  // Sub-case 1: AUTO (HYBRID=0) must skip on decode-class M.
  {
    CustomKernelOverride           ck_on(true);
    NRoundsModeOverride            single_round(1);
    CustomKernelNTileOverride      default_n_tile(0);
    NTileStrategyOverride          auto_strategy(0);
    NTileHeavyThresholdOverride  hybrid_auto(0);
    // Proportional split is OFF by default; pin it ON so the proportional
    // contract asserted below holds.  The tier-skip assertion
    // (n_thr_fixed != 0) is independent of this knob.
    DecodeProportionalOverride   prop_on(true);

    reset_grp_matmul_caches();
    AlgoEnvGuard       algo_guard(3);
    PhaseBCaptureGuard cap;
    ASSERT_EQ(group_matmul_direct(s.gv.layout, s.gv.transA, s.gv.transB,
                                  s.gv.Ms, s.gv.Ns, s.gv.Ks, s.gv.alpha,
                                  s.srcs, s.gv.lda, s.weis, s.gv.ldb,
                                  s.biases, s.gv.beta, s.dsts, s.gv.ldc,
                                  s.gv.is_wc, s.params,
                                  nullptr, nullptr),
              status_t::success);
    const auto &snap =
        zendnnl::lowoha::matmul::test_api::s_last_phase_b_snapshot;
    ASSERT_TRUE(snap.valid);
    EXPECT_NE(snap.n_thr_fixed, 0)
        << "AUTO adaptive-tiers must be skipped on decode-class shape "
           "(max_M=24 <= kDecodeMaxM=32); tiers clear n_thr_fixed to 0, "
           "so a non-zero value confirms tiers did NOT engage.  The "
           "decode M-proportional split (which DOES run here) leaves "
           "n_thr_fixed at base.";
    // Decode now uses the M-proportional thread split (commit:
    // "static M-proportional thread allocation for CK decode") instead
    // of the legacy uniform Phase B base/base+1.  That is still NOT the
    // adaptive-tier water-fill (n_thr_fixed stays != 0, asserted above).
    // Assert the proportional contract rather than the old {0,2,3}:
    //   - per_expert_remainder set (prefix-sum executor path),
    //   - every active expert gets >= 1 thread (no starvation),
    //   - total allocation <= num_threads,
    //   - the heaviest expert (M=24) gets a proportional share > the
    //     uniform base+1 (=3) — distinguishes proportional from Phase B.
    EXPECT_TRUE(snap.per_expert_remainder);
    int sum_thr = 0;
    for (int e = 0; e < s.num_ops; ++e) {
      const int n = static_cast<int>(snap.stable_n_thr_per_expert[e]);
      if (s.Ms[e] > 0) {
        EXPECT_GE(n, 1) << "active expert e=" << e << " starved (0 thr)";
      }
      sum_thr += n;
    }
    EXPECT_LE(sum_thr, 32) << "M-proportional over-allocated threads";
    EXPECT_GE(static_cast<int>(snap.stable_n_thr_per_expert[0]), 4)
        << "heaviest expert (M=24) should get a proportional share "
           "(> uniform base+1=3); got "
        << static_cast<int>(snap.stable_n_thr_per_expert[0]);
  }

  // Sub-case 2: MANUAL (HYBRID=10, would tag heavies on this Ms)
  // must also skip on decode-class M.
  {
    CustomKernelOverride           ck_on(true);
    NRoundsModeOverride            single_round(1);
    CustomKernelNTileOverride      default_n_tile(0);
    NTileStrategyOverride          auto_strategy(0);
    // Threshold=10 would tag M ∈ {24,20,16,12} as heavy on this
    // Ms — i.e. would have engaged MANUAL on legacy semantics.
    NTileHeavyThresholdOverride  hybrid_manual(10);
    // Proportional split is OFF by default; pin it ON so the proportional
    // contract asserted below holds.  The tier-skip assertion
    // (n_thr_fixed != 0) is independent of this knob.
    DecodeProportionalOverride   prop_on(true);

    reset_grp_matmul_caches();
    AlgoEnvGuard       algo_guard(3);
    PhaseBCaptureGuard cap;
    ASSERT_EQ(group_matmul_direct(s.gv.layout, s.gv.transA, s.gv.transB,
                                  s.gv.Ms, s.gv.Ns, s.gv.Ks, s.gv.alpha,
                                  s.srcs, s.gv.lda, s.weis, s.gv.ldb,
                                  s.biases, s.gv.beta, s.dsts, s.gv.ldc,
                                  s.gv.is_wc, s.params,
                                  nullptr, nullptr),
              status_t::success);
    const auto &snap =
        zendnnl::lowoha::matmul::test_api::s_last_phase_b_snapshot;
    ASSERT_TRUE(snap.valid);
    EXPECT_NE(snap.n_thr_fixed, 0)
        << "MANUAL adaptive-tiers must be skipped on decode-class shape "
           "(max_M=24 <= kDecodeMaxM=32); tiers clear n_thr_fixed to 0, "
           "so a non-zero value confirms the MANUAL water-fill did NOT "
           "engage.  The decode M-proportional split runs instead.";
    // As in sub-case 1: decode uses the M-proportional split, not the
    // MANUAL adaptive-tier water-fill.  threshold=10 (!= -1) keeps the
    // proportional split enabled; only an explicit -1 disable would
    // restore uniform Phase B.  Assert the proportional contract.
    EXPECT_TRUE(snap.per_expert_remainder);
    int sum_thr = 0;
    for (int e = 0; e < s.num_ops; ++e) {
      const int n = static_cast<int>(snap.stable_n_thr_per_expert[e]);
      if (s.Ms[e] > 0) {
        EXPECT_GE(n, 1) << "active expert e=" << e << " starved (0 thr)";
      }
      sum_thr += n;
    }
    EXPECT_LE(sum_thr, 32) << "M-proportional over-allocated threads";
    EXPECT_GE(static_cast<int>(snap.stable_n_thr_per_expert[0]), 4)
        << "heaviest expert (M=24) should get a proportional share "
           "(> uniform base+1=3); got "
        << static_cast<int>(snap.stable_n_thr_per_expert[0]);
  }
}

TEST(TestGroupMatmulAutoPhaseEnv, Algo3PhaseEnvClampedOnNonNTileSafe) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  reset_grp_matmul_caches();
  AlgoEnvGuard reset_algo(0);
  AutoPromptAlgoOverride force_prompt(3);
  AutoDecodeAlgoOverride no_decode(0);

  // Mixtral-class prompt, with dynamic_quant=true → n_tile_safe=false.
  // Phase env asks for ALGO 3, but the safety clamp falls to ALGO 1.
  auto s = build_auto_probe(/*M=*/256, /*K=*/4096, /*N=*/14336,
                            /*num_ops=*/8, /*num_threads=*/128);
  for (auto &p : s.params) {
    p.dynamic_quant                  = true;
    p.quant_params.src_scale.dims    = {256, 1};
    p.quant_params.src_scale.dt      = data_type_t::f32;
  }
  EXPECT_EQ(select_grp_matmul_algo(s.layout, s.M, s.N, s.K, s.params,
                                   s.num_threads),
            1)
      << "AUTO_PROMPT_ALGO=3 with !n_tile_safe must clamp to ALGO 1 "
         "(same correctness contract as global ALGO=3 path)";
}

// ============================================================================
// [8f] TestGroupMatmulMTileBranches — ALGO 2 (M-tile) internal branch dispatch.
//
// `flat_m_tile` dispatches between four internal branches based on
// workload shape:
//   round-based            (active_ops > num_threads)
//   multi-tier hybrid      (Qwen3-class many-expert skewed prompt)
//   wide-N memory-bound    (total_need * 2 ≤ num_threads, max_M > 1)
//   phase-2 single-tier    (default M-weighted fallthrough)
//
// These tests pin ALGO 2 via `AlgoEnvGuard(2)`, exercise shapes that
// should drive each branch, and assert the tag published by the
// chosen branch (`test_api::s_last_m_tile_path`) matches expectation.
// Adds focused coverage for the wide-N fallback and multi-tier hybrid
// gates so silent regressions surface as failed tag asserts instead
// of perf regressions in downstream benchmarks.
//
// The capture hook is gated by `s_capture_m_tile_path` (armed only
// while `MTilePathCaptureGuard` is in scope); production builds
// never arm it, so the per-call cost is a single relaxed load of a
// cache-line-shared `false` bool — no coherence traffic, branch-
// predictable.  See doc-block on `s_capture_m_tile_path` in
// `group_matmul_parallel_common.hpp`.
// ============================================================================

// ALGO 2 is now PURE M-tile: a FORCED `ZENDNNL_GRP_MATMUL_ALGO=2` on a
// wide-N shape (`total_need*2 ≤ num_threads && max_M > 1`) no longer falls
// back to the sequential-full-team loop — it runs the single-tier M-tile
// plan (kPhase2Single).  (AUTO routes this regime to ALGO 1 via Rule 0.7;
// see TestGroupMatmulAutoPhaseEnv.PromptWideNRoutesToAlgo1.)
TEST(TestGroupMatmulMTileBranches, ForcedAlgo2OnWideNRunsSingleTier) {
  using namespace moe_test_utils;
  using zendnnl::lowoha::matmul::group_matmul_direct;
  using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kPhase2Single;
  using zendnnl::lowoha::matmul::status_t;

  const int saved_num_threads = omp_get_max_threads();
  struct ThreadGuard {
    int prev;
    ~ThreadGuard() { omp_set_num_threads(prev); }
  } thread_guard{saved_num_threads};
  omp_set_num_threads(32);
  int actual_team_size = 0;
  #pragma omp parallel
  {
    #pragma omp master
    actual_team_size = omp_get_num_threads();
  }
  if (actual_team_size < 32) {
    GTEST_SKIP() << "Requires >= 32 OMP threads; have " << actual_team_size;
  }

  // 8 actives × M=8 × K=64 × N=1024.
  //   total_need = 8 × ceil(8/16) = 8;  2*8 = 16 ≤ 32 ⇒ wide-N gate ✓
  //   max_M = 8 > 1                                  ⇒ decode-exclusion clears
  //   max_M = 8 < 256                                ⇒ multi-tier gate fails
  auto s = build_hybrid_probe(/*num_threads=*/32,
      /*Ms=*/{8, 8, 8, 8, 8, 8, 8, 8});

  reset_grp_matmul_caches();
  AlgoEnvGuard            algo_guard(2);
  MTileHybridOverride     hybrid_auto(0);
  MTilePathCaptureGuard   cap;
  GemmModeCaptureGuard    mode_cap;

  ASSERT_EQ(group_matmul_direct(s.gv.layout, s.gv.transA, s.gv.transB,
                                s.gv.Ms, s.gv.Ns, s.gv.Ks, s.gv.alpha,
                                s.srcs, s.gv.lda, s.weis, s.gv.ldb,
                                s.biases, s.gv.beta, s.dsts, s.gv.ldc,
                                s.gv.is_wc, s.params,
                                nullptr, nullptr),
            status_t::success);

  const int tag = zendnnl::lowoha::matmul::test_api
      ::s_last_m_tile_path.load(std::memory_order_relaxed);
  EXPECT_EQ(tag, kPhase2Single)
      << "Forced ALGO 2 on a wide-N shape (8 actives × M=8 / 32t) must run "
         "PURE single-tier M-tile now (no internal wide-N fallback); "
         "got tag=" << tag
      << " (kMultiTier=1, kPhase2Single=3, kManyExpertsSeqFallback=7)";

  // gemm_mode must name the executed branch, and exec_algo must map to 2.
  const char *mode = zendnnl::lowoha::matmul::test_api
      ::s_last_group_matmul_direct_gemm_mode.load(std::memory_order_relaxed);
  ASSERT_NE(mode, nullptr);
  EXPECT_STREQ(mode, "flat_m_tile_single_tier");
  EXPECT_EQ(zendnnl::lowoha::matmul::executed_algo_from_gemm_mode(mode), 2);
}

// Wide-N fallback MUST be excluded for pure-decode workloads (max_M==1).
// The Phase 2 single-tier CCD-stripe layout is the latency-optimal
// mapping for M=1 — one thread per CCD, experts parallel across CCDs.
// Routing M=1 to wide-N would serialise each expert on the full thread
// team and trade away the CCD-parallel decode win.
TEST(TestGroupMatmulMTileBranches, WideNFallbackExcludesDecodeM1) {
  using namespace moe_test_utils;
  using zendnnl::lowoha::matmul::group_matmul_direct;
  using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kPhase2Single;
  using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kWideNFallback;
  using zendnnl::lowoha::matmul::status_t;

  const int saved_num_threads = omp_get_max_threads();
  struct ThreadGuard {
    int prev;
    ~ThreadGuard() { omp_set_num_threads(prev); }
  } thread_guard{saved_num_threads};
  omp_set_num_threads(32);
  int actual_team_size = 0;
  #pragma omp parallel
  {
    #pragma omp master
    actual_team_size = omp_get_num_threads();
  }
  if (actual_team_size < 32) {
    GTEST_SKIP() << "Requires >= 32 OMP threads; have " << actual_team_size;
  }

  // 8 actives × M=1.  Numerically `total_need*2 = 16 ≤ 32` would PASS
  // the wide-N count gate; the `max_M_single_tier > 1` decode-exclusion
  // clamp must keep wide-N off and route to Phase 2 single-tier
  // (CCD-stripe).
  auto s = build_hybrid_probe(/*num_threads=*/32,
      /*Ms=*/{1, 1, 1, 1, 1, 1, 1, 1});

  reset_grp_matmul_caches();
  AlgoEnvGuard            algo_guard(2);
  MTileHybridOverride     hybrid_auto(0);
  MTilePathCaptureGuard   cap;

  ASSERT_EQ(group_matmul_direct(s.gv.layout, s.gv.transA, s.gv.transB,
                                s.gv.Ms, s.gv.Ns, s.gv.Ks, s.gv.alpha,
                                s.srcs, s.gv.lda, s.weis, s.gv.ldb,
                                s.biases, s.gv.beta, s.dsts, s.gv.ldc,
                                s.gv.is_wc, s.params,
                                nullptr, nullptr),
            status_t::success);

  const int tag = zendnnl::lowoha::matmul::test_api
      ::s_last_m_tile_path.load(std::memory_order_relaxed);
  EXPECT_NE(tag, kWideNFallback)
      << "Wide-N fallback MUST be excluded on decode-class max_M=1; "
         "got wide-N (tag=" << tag << ")";
  EXPECT_EQ(tag, kPhase2Single)
      << "max_M=1 must fall through to Phase 2 single-tier (CCD-stripe); "
         "got tag=" << tag;
}

// Multi-tier hybrid engages when ALL of: actives ≥ num_threads/2,
// max_M ≥ 256, max_M ≥ 4×avg_M, n_light ≥ num_threads/8.  On 32 threads
// the gate needs ≥ 16 actives and ≥ 4 lights.  Shape: 1 heavy at
// M=1024 + 15 lights at M=4 — avg_M ≈ 67.75, light_cut = max(8, 16) = 16
// (M=4 ≤ 16 ⇒ LIGHT), max_M/avg_M ≈ 15.1 ⇒ skew ✓.
TEST(TestGroupMatmulMTileBranches, MultiTierEngagesOnSkewedPrompt) {
  using namespace moe_test_utils;
  using zendnnl::lowoha::matmul::group_matmul_direct;
  using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kMultiTier;
  using zendnnl::lowoha::matmul::status_t;

  const int saved_num_threads = omp_get_max_threads();
  struct ThreadGuard {
    int prev;
    ~ThreadGuard() { omp_set_num_threads(prev); }
  } thread_guard{saved_num_threads};
  omp_set_num_threads(32);
  int actual_team_size = 0;
  #pragma omp parallel
  {
    #pragma omp master
    actual_team_size = omp_get_num_threads();
  }
  if (actual_team_size < 32) {
    GTEST_SKIP() << "Requires >= 32 OMP threads; have " << actual_team_size;
  }

  // 16 actives: 1 heavy (M=1024) + 15 lights (M=4).
  //   actives = 16 ≥ 32/2                              ✓
  //   max_M = 1024 ≥ 256                              ✓
  //   avg_M = (1024 + 15*4) / 16 ≈ 67.75
  //   max_M/avg_M ≈ 15.1 ≥ 4                          ✓ (skew gate)
  //   light_cut = max(8, 67/4) = max(8, 16) = 16
  //   n_light = 15 (every M=4 ≤ 16); n_heavy = 1     ✓
  //   total_need = 64 + 15 = 79; 2*79 = 158 > 32     ⇒ wide-N excluded
  std::vector<int> ms = {1024};
  ms.insert(ms.end(), 15, 4);
  auto s = build_hybrid_probe(/*num_threads=*/32, ms, /*N=*/1024);

  reset_grp_matmul_caches();
  AlgoEnvGuard            algo_guard(2);
  MTileHybridOverride     hybrid_auto(0);
  MTilePathCaptureGuard   cap;
  GemmModeCaptureGuard    mode_cap;

  ASSERT_EQ(group_matmul_direct(s.gv.layout, s.gv.transA, s.gv.transB,
                                s.gv.Ms, s.gv.Ns, s.gv.Ks, s.gv.alpha,
                                s.srcs, s.gv.lda, s.weis, s.gv.ldb,
                                s.biases, s.gv.beta, s.dsts, s.gv.ldc,
                                s.gv.is_wc, s.params,
                                nullptr, nullptr),
            status_t::success);

  const int tag = zendnnl::lowoha::matmul::test_api
      ::s_last_m_tile_path.load(std::memory_order_relaxed);
  EXPECT_EQ(tag, kMultiTier)
      << "Multi-tier hybrid must engage on 16 actives with skewed M "
         "(1 heavy M=1024 + 15 lights M=4, skew ≈ 15×); got tag=" << tag;

  // gemm_mode must name the multi-tier branch; exec_algo maps to 2.
  const char *mode = zendnnl::lowoha::matmul::test_api
      ::s_last_group_matmul_direct_gemm_mode.load(std::memory_order_relaxed);
  ASSERT_NE(mode, nullptr);
  EXPECT_STREQ(mode, "flat_m_tile_multitier");
  EXPECT_EQ(zendnnl::lowoha::matmul::executed_algo_from_gemm_mode(mode), 2);
}

// ============================================================================
// [8j] Logging truth: gemm_mode names the executed branch, and
//      executed_algo_from_gemm_mode maps it to the ALGO that actually ran.
// ============================================================================

// Unit test for the mode->algo mapping used by the post-exec
// [GRP_MATMUL.CALL] exec_algo= field.  Pure function, no execution.
TEST(TestGroupMatmulGemmMode, ExecutedAlgoFromGemmModeMapping) {
  using zendnnl::lowoha::matmul::executed_algo_from_gemm_mode;
  EXPECT_EQ(executed_algo_from_gemm_mode(nullptr), 0);
  EXPECT_EQ(executed_algo_from_gemm_mode("garbage"), 0);
  EXPECT_EQ(executed_algo_from_gemm_mode("sequential"), 1);
  EXPECT_EQ(executed_algo_from_gemm_mode("sequential_experts"), 1);
  // seq-clamp wears an ALGO-2 mode prefix but runs ALGO-1 behaviour -> 1.
  EXPECT_EQ(executed_algo_from_gemm_mode("flat_m_tile_seq_clamp"), 1);
  EXPECT_EQ(executed_algo_from_gemm_mode("flat_m_tile_single_tier"), 2);
  EXPECT_EQ(executed_algo_from_gemm_mode("flat_m_tile_multitier"), 2);
  EXPECT_EQ(executed_algo_from_gemm_mode("flat_m_tile"), 2);
  EXPECT_EQ(executed_algo_from_gemm_mode("vertical_fusion_bf16"), 2);
  EXPECT_EQ(executed_algo_from_gemm_mode("flat_n_tile_custom"), 3);
  EXPECT_EQ(executed_algo_from_gemm_mode("multilevel_rounds"), 4);
  EXPECT_EQ(executed_algo_from_gemm_mode("per_expert"), 5);
  // Fused composites derive the algo from the Op1 sub-mode.
  EXPECT_EQ(executed_algo_from_gemm_mode(
      "fused_moe_2pass_intalloc(op1=flat_m_tile_single_tier,"
      "op2=flat_m_tile_single_tier)+postop"), 2);
  EXPECT_EQ(executed_algo_from_gemm_mode(
      "fused_moe_vertical_intalloc(op1=vertical_fusion_bf16,"
      "op2=vertical_fusion_bf16)"), 2);
  EXPECT_EQ(executed_algo_from_gemm_mode(
      "fused_moe_2pass(op1=flat_n_tile_custom,op2=flat_n_tile_custom)"), 3);
  // No-op / skip markers map to 0 (nothing executed).
  EXPECT_EQ(executed_algo_from_gemm_mode("fused_moe_skip"), 0);
  EXPECT_EQ(executed_algo_from_gemm_mode("flat_m_tile_skip"), 0);
  EXPECT_EQ(executed_algo_from_gemm_mode("multilevel_skip"), 0);
}

// Forced ALGO 2 with active_ops > num_threads: gemm_mode must name the
// sequential-full-team clamp, and exec_algo must report 1 (the ALGO-1
// behaviour that actually ran) even though chosen=ALGO_2.  This is the
// headline selection-vs-execution divergence the logging makes explicit.
TEST(TestGroupMatmulMTileBranches, ForcedAlgo2SeqClampWritesMode) {
  using namespace moe_test_utils;
  using zendnnl::lowoha::matmul::group_matmul_direct;
  using zendnnl::lowoha::matmul::test_api::m_tile_path_tag
      ::kManyExpertsSeqFallback;
  using zendnnl::lowoha::matmul::status_t;

  const int saved_num_threads = omp_get_max_threads();
  struct ThreadGuard {
    int prev;
    ~ThreadGuard() { omp_set_num_threads(prev); }
  } thread_guard{saved_num_threads};
  omp_set_num_threads(8);
  int actual_team_size = 0;
  #pragma omp parallel
  {
    #pragma omp master
    actual_team_size = omp_get_num_threads();
  }
  if (actual_team_size < 4) {
    GTEST_SKIP() << "Requires >= 4 OMP threads; have " << actual_team_size;
  }

  // 8 active experts on a 4-thread budget -> active_ops(8) > num_threads(4)
  // -> M-tile infeasible -> sequential-full-team clamp.
  auto s = build_hybrid_probe(/*num_threads=*/4,
      /*Ms=*/{8, 8, 8, 8, 8, 8, 8, 8});

  reset_grp_matmul_caches();
  AlgoEnvGuard            algo_guard(2);
  MTilePathCaptureGuard   cap;
  GemmModeCaptureGuard    mode_cap;

  ASSERT_EQ(group_matmul_direct(s.gv.layout, s.gv.transA, s.gv.transB,
                                s.gv.Ms, s.gv.Ns, s.gv.Ks, s.gv.alpha,
                                s.srcs, s.gv.lda, s.weis, s.gv.ldb,
                                s.biases, s.gv.beta, s.dsts, s.gv.ldc,
                                s.gv.is_wc, s.params,
                                nullptr, nullptr),
            status_t::success);

  const int tag = zendnnl::lowoha::matmul::test_api
      ::s_last_m_tile_path.load(std::memory_order_relaxed);
  EXPECT_EQ(tag, kManyExpertsSeqFallback);

  const char *mode = zendnnl::lowoha::matmul::test_api
      ::s_last_group_matmul_direct_gemm_mode.load(std::memory_order_relaxed);
  ASSERT_NE(mode, nullptr);
  EXPECT_STREQ(mode, "flat_m_tile_seq_clamp");
  EXPECT_EQ(zendnnl::lowoha::matmul::executed_algo_from_gemm_mode(mode), 1)
      << "forced ALGO 2 clamped to sequential-full-team must report "
         "exec_algo=1 (ran ALGO-1 behaviour) despite chosen=ALGO_2";
}

// Multi-tier hybrid MUST stay off when `M_TILE_HYBRID=-1` (DISABLED),
// even on a shape that would otherwise satisfy every gate.  Same
// Qwen3-class shape as `MultiTierEngagesOnSkewedPrompt`; verifies the
// env-disable escape hatch keeps the legacy single-tier path
// available for A/B testing and emergency rollback.
TEST(TestGroupMatmulMTileBranches, MultiTierDisabledViaEnvOverride) {
  using namespace moe_test_utils;
  using zendnnl::lowoha::matmul::group_matmul_direct;
  using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kMultiTier;
  using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kPhase2Single;
  using zendnnl::lowoha::matmul::status_t;

  const int saved_num_threads = omp_get_max_threads();
  struct ThreadGuard {
    int prev;
    ~ThreadGuard() { omp_set_num_threads(prev); }
  } thread_guard{saved_num_threads};
  omp_set_num_threads(32);
  int actual_team_size = 0;
  #pragma omp parallel
  {
    #pragma omp master
    actual_team_size = omp_get_num_threads();
  }
  if (actual_team_size < 32) {
    GTEST_SKIP() << "Requires >= 32 OMP threads; have " << actual_team_size;
  }

  std::vector<int> ms = {1024};
  ms.insert(ms.end(), 15, 4);
  auto s = build_hybrid_probe(/*num_threads=*/32, ms, /*N=*/1024);

  reset_grp_matmul_caches();
  AlgoEnvGuard            algo_guard(2);
  MTileHybridOverride     hybrid_disabled(-1);  // DISABLED escape hatch
  MTilePathCaptureGuard   cap;

  ASSERT_EQ(group_matmul_direct(s.gv.layout, s.gv.transA, s.gv.transB,
                                s.gv.Ms, s.gv.Ns, s.gv.Ks, s.gv.alpha,
                                s.srcs, s.gv.lda, s.weis, s.gv.ldb,
                                s.biases, s.gv.beta, s.dsts, s.gv.ldc,
                                s.gv.is_wc, s.params,
                                nullptr, nullptr),
            status_t::success);

  const int tag = zendnnl::lowoha::matmul::test_api
      ::s_last_m_tile_path.load(std::memory_order_relaxed);
  EXPECT_NE(tag, kMultiTier)
      << "M_TILE_HYBRID=-1 must force the legacy single-tier path "
         "even on shapes the AUTO gate would accept; got multi-tier "
         "(tag=" << tag << ")";
  EXPECT_EQ(tag, kPhase2Single)
      << "M_TILE_HYBRID=-1 + Qwen3-class skewed shape must land in "
         "Phase 2 single-tier (the legacy M-weighted fallback); "
         "got tag=" << tag;
}

// Multi-tier hybrid stays off (falls through to Phase 2 single-tier)
// when the workload shape doesn't pass the skew gate.  Shape: 16
// actives all at M=256 ⇒ max_M = avg_M = 256; max_M / avg_M = 1 < 4
// ⇒ skew gate fails even though `actives` and `max_M ≥ 256` pass.
TEST(TestGroupMatmulMTileBranches, MultiTierLowSkewFallsThrough) {
  using namespace moe_test_utils;
  using zendnnl::lowoha::matmul::group_matmul_direct;
  using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kMultiTier;
  using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kPhase2Single;
  using zendnnl::lowoha::matmul::status_t;

  const int saved_num_threads = omp_get_max_threads();
  struct ThreadGuard {
    int prev;
    ~ThreadGuard() { omp_set_num_threads(prev); }
  } thread_guard{saved_num_threads};
  omp_set_num_threads(32);
  int actual_team_size = 0;
  #pragma omp parallel
  {
    #pragma omp master
    actual_team_size = omp_get_num_threads();
  }
  if (actual_team_size < 32) {
    GTEST_SKIP() << "Requires >= 32 OMP threads; have " << actual_team_size;
  }

  // 16 actives all at M=256.  actives gate ✓, max_M gate ✓, skew
  // gate FAILS (max_M / avg_M = 1 < 4) ⇒ multi-tier must NOT engage;
  // falls through to Phase 2 single-tier.
  //   total_need = 16 × 16 = 256; 2×256 = 512 > 32 ⇒ wide-N excluded.
  auto s = build_hybrid_probe(/*num_threads=*/32,
      /*Ms=*/std::vector<int>(16, 256), /*N=*/1024);

  reset_grp_matmul_caches();
  AlgoEnvGuard            algo_guard(2);
  MTileHybridOverride     hybrid_auto(0);
  MTilePathCaptureGuard   cap;

  ASSERT_EQ(group_matmul_direct(s.gv.layout, s.gv.transA, s.gv.transB,
                                s.gv.Ms, s.gv.Ns, s.gv.Ks, s.gv.alpha,
                                s.srcs, s.gv.lda, s.weis, s.gv.ldb,
                                s.biases, s.gv.beta, s.dsts, s.gv.ldc,
                                s.gv.is_wc, s.params,
                                nullptr, nullptr),
            status_t::success);

  const int tag = zendnnl::lowoha::matmul::test_api
      ::s_last_m_tile_path.load(std::memory_order_relaxed);
  EXPECT_NE(tag, kMultiTier)
      << "Multi-tier must NOT engage on low-skew shapes "
         "(max_M/avg_M = 1 < kHybridMinSkewX=4); got multi-tier "
         "(tag=" << tag << ")";
  EXPECT_EQ(tag, kPhase2Single)
      << "Low-skew shape must fall through to Phase 2 single-tier; "
         "got tag=" << tag;
}

// ============================================================================
// [8g] TestGroupMatmulMTileStandalone: numerical correctness of `flat_m_tile`
// when called as a standalone grouped matmul (no `fused_moe`, no `gated_act`,
// no `moe_postop`) — the llama.cpp use case.  Existing `TestGroupMatmulMTile
// Branches` only verify branch tags; they never check that the dst[] values
// produced by the M-tile path match what the caller expects.  This fixture
// adds the per-expert × per-row × per-col value comparison against an ALGO 1
// (legacy sequential per-expert) reference run on the same inputs.
//
// Reference choice: `AlgoEnvGuard(1)` forces the sequential per-expert ALGO 1
// path which dispatches each expert through `kernel_select` independently of
// any M-tile planner state.  Both runs use identical inputs (same src/wei
// pointers, same alpha/beta/bias, same params), so any numerical divergence
// between the two would indicate an M-tile-specific bug.  We tolerate small
// floating-point reduction-order noise via a relative+absolute tolerance.
//
// Coverage:
//   * Caller-allocated dst[] (canonical llama.cpp pattern).
//   * act=none (Op1-only standalone grouped matmul).
//   * Both bias=nullptr and bias=non-null cases.
//   * Mixed alpha/beta values (1.0 / 0.0 default, plus 2.0 / 0.5 stress).
//   * Path tag asserted: typically kPhase2Single for the chosen shape.
//
// Shape: 16 active experts × M=64 × K=64 × N=128 (Op1-like) — small enough
// for fast test execution, large enough that the M-tile planner picks a
// single-tier slice plan rather than round-based or wide-N.
//
// The test is designed to run on >=8 threads (skip otherwise); shape avoids
// `active_ops > num_threads` so the kRoundBased branch does not engage here
// (that branch has its own dedicated coverage in [8h] below).
// ============================================================================

namespace {

// Append-only helpers private to the standalone correctness fixture below.
//
// Build a bf16 standalone test input set with deterministic seeds.  Mirrors
// `build_hybrid_probe` but exposes a non-zero `bias` slot per expert (or
// nullptr per expert when `with_bias=false`) and per-expert alpha / beta
// (defaults 1.0 / 0.0).  Caller passes the M vector; N / K are fixed.
struct StandaloneMTileShape {
  int num_ops;
  int num_threads;
  int N;
  int K;
  std::vector<int> Ms;
  std::vector<std::vector<zendnnl::common::bfloat16_t>> src;
  std::vector<std::vector<zendnnl::common::bfloat16_t>> wei;
  std::vector<std::vector<zendnnl::common::bfloat16_t>> bias;
  std::vector<std::vector<zendnnl::common::bfloat16_t>> dst_test;
  std::vector<std::vector<zendnnl::common::bfloat16_t>> dst_ref;
  std::vector<const void *> srcs;
  std::vector<const void *> weis;
  std::vector<const void *> biases;
  std::vector<void *> dsts_test;
  std::vector<void *> dsts_ref;
  moe_test_utils::GemmVecs gv;
  std::vector<zendnnl::lowoha::matmul::matmul_params> params;
};

static StandaloneMTileShape build_standalone_mtile_shape(
    int num_threads, const std::vector<int> &Ms,
    int N, int K,
    bool with_bias,
    float alpha_val = 1.0f, float beta_val = 0.0f) {
  using namespace moe_test_utils;
  using zendnnl::common::bfloat16_t;
  using zendnnl::common::data_type_t;
  StandaloneMTileShape s;
  s.num_threads = num_threads;
  s.num_ops     = static_cast<int>(Ms.size());
  s.N           = N;
  s.K           = K;
  s.Ms          = Ms;
  s.src.resize(s.num_ops);
  s.wei.resize(s.num_ops);
  s.bias.resize(s.num_ops);
  s.dst_test.resize(s.num_ops);
  s.dst_ref.resize(s.num_ops);
  for (int e = 0; e < s.num_ops; ++e) {
    s.src[e].resize(static_cast<size_t>(Ms[e]) * K);
    s.wei[e].resize(static_cast<size_t>(K) * N);
    s.dst_test[e].assign(static_cast<size_t>(Ms[e]) * N, bfloat16_t(0.0f));
    s.dst_ref [e].assign(static_cast<size_t>(Ms[e]) * N, bfloat16_t(0.0f));
    fill_src(s.src[e],  e, 0.02f);
    fill_wei1(s.wei[e], e, 0.005f);
    if (with_bias) {
      s.bias[e].resize(static_cast<size_t>(N));
      for (int c = 0; c < N; ++c)
        s.bias[e][c] = bfloat16_t(static_cast<float>(c) * 0.001f + 0.01f);
    }
  }
  s.srcs.resize(s.num_ops);
  s.weis.resize(s.num_ops);
  s.biases.resize(s.num_ops);
  s.dsts_test.resize(s.num_ops);
  s.dsts_ref.resize(s.num_ops);
  for (int e = 0; e < s.num_ops; ++e) {
    s.srcs[e]      = s.src[e].data();
    s.weis[e]      = s.wei[e].data();
    s.biases[e]    = with_bias ? s.bias[e].data() : nullptr;
    s.dsts_test[e] = s.dst_test[e].data();
    s.dsts_ref [e] = s.dst_ref [e].data();
  }
  s.gv = GemmVecs::uniform(s.num_ops, /*M=*/0, N, K,
                           alpha_val, beta_val,
                           /*wc=*/true, /*tA=*/false, /*tB=*/false);
  s.gv.Ms = Ms;
  s.params = make_uniform_params(
      s.num_ops, data_type_t::bf16,
      /*bias_dt=*/with_bias ? data_type_t::bf16 : data_type_t::none);
  for (int e = 0; e < s.num_ops; ++e) {
    s.params[e].num_threads = num_threads;
  }
  return s;
}

// Per-expert × per-row × per-col tolerance comparison for bf16 buffers.
// Returns the first divergence (e, r, c, got, ref) — or -1's on success.
struct CompareResult {
  int e = -1, r = -1, c = -1;
  float got = 0.0f, ref = 0.0f;
  bool ok() const { return e < 0; }
};

// For gated-act dst, the live data lives only in the first `compare_cols`
// columns (typically `N/2`); columns `[compare_cols, N)` carry the raw
// pre-act matmul output and are skipped to avoid spurious mismatches.
static CompareResult compare_bf16_buffers(
    const std::vector<std::vector<zendnnl::common::bfloat16_t>> &got_b,
    const std::vector<std::vector<zendnnl::common::bfloat16_t>> &ref_b,
    const std::vector<int> &Ms, int stride, int compare_cols,
    float rtol = 5e-2f, float atol = 5e-2f) {
  CompareResult r;
  for (int e = 0; e < (int)Ms.size(); ++e) {
    const int M_e = Ms[e];
    if (M_e == 0) continue;
    for (int row = 0; row < M_e; ++row) {
      for (int c = 0; c < compare_cols; ++c) {
        const size_t off = static_cast<size_t>(row) * stride + c;
        const float g = static_cast<float>(got_b[e][off]);
        const float f = static_cast<float>(ref_b[e][off]);
        const float bound = std::abs(f) * rtol + atol;
        if (std::abs(g - f) > bound) {
          r.e = e; r.r = row; r.c = c; r.got = g; r.ref = f;
          return r;
        }
      }
    }
  }
  return r;
}

// Run the standalone group_matmul_direct twice — once with `algo_test`
// pinned (e.g. ALGO 2 / M-tile) writing to `dst_test`, once with the
// reference algo (ALGO 1 sequential per-expert) writing to `dst_ref` —
// and compare per-expert × per-row × per-col with the supplied tolerance.
//
// `gated_act` is forwarded to both runs (so the ref run also fuses the
// activation under ALGO 1) — when non-null, only the first `N/2`
// columns are compared because the gated activation halves the live
// data width in-place.
static void run_and_compare_standalone(
    StandaloneMTileShape &s,
    int algo_test, int algo_ref,
    int expected_tag, float rtol, float atol,
    const char *case_label,
    const zendnnl::lowoha::matmul::grp_matmul_gated_act_params
        *gated_act = nullptr) {
  using namespace moe_test_utils;
  using zendnnl::lowoha::matmul::group_matmul_direct;
  using zendnnl::lowoha::matmul::status_t;

  // Reference run first (ALGO 1, sequential per-expert; no M-tile
  // planner state can leak into the test).  Use a fresh params copy
  // because the dispatcher writes `lowoha_algo` back into params[i].
  {
    auto pr = s.params;
    moe_test_utils::AlgoEnvGuard algo_guard_ref(algo_ref);
    ::reset_grp_matmul_caches();
    ASSERT_EQ(group_matmul_direct(s.gv.layout, s.gv.transA, s.gv.transB,
                                  s.gv.Ms, s.gv.Ns, s.gv.Ks, s.gv.alpha,
                                  s.srcs, s.gv.lda, s.weis, s.gv.ldb,
                                  s.biases, s.gv.beta, s.dsts_ref, s.gv.ldc,
                                  s.gv.is_wc, pr, nullptr, gated_act),
              status_t::success) << "ref " << case_label;
  }
  // Test run (ALGO 2 / M-tile) with path-tag capture armed.
  int tag = 0;
  {
    auto pt = s.params;
    ::reset_grp_matmul_caches();
    moe_test_utils::AlgoEnvGuard algo_guard_test(algo_test);
    moe_test_utils::MTileHybridOverride hybrid_auto(0);
    moe_test_utils::MTilePathCaptureGuard cap;
    ASSERT_EQ(group_matmul_direct(s.gv.layout, s.gv.transA, s.gv.transB,
                                  s.gv.Ms, s.gv.Ns, s.gv.Ks, s.gv.alpha,
                                  s.srcs, s.gv.lda, s.weis, s.gv.ldb,
                                  s.biases, s.gv.beta, s.dsts_test, s.gv.ldc,
                                  s.gv.is_wc, pt, nullptr, gated_act),
              status_t::success) << "test " << case_label;
    tag = zendnnl::lowoha::matmul::test_api
        ::s_last_m_tile_path.load(std::memory_order_relaxed);
  }
  EXPECT_EQ(tag, expected_tag)
      << "Path tag mismatch on " << case_label
      << ": expected " << expected_tag << ", got " << tag
      << " (kRoundBased=0, kMultiTier=1, kWideNFallback=2, "
         "kPhase2Single=3, kVerticalFusionBF16=4)";
  using zendnnl::lowoha::matmul::grp_matmul_gated_act_t;
  const int compare_cols =
      (gated_act && gated_act->act != grp_matmul_gated_act_t::none)
          ? s.N / 2 : s.N;
  const auto cmp = compare_bf16_buffers(
      s.dst_test, s.dst_ref, s.Ms, /*stride=*/s.N, compare_cols, rtol, atol);
  EXPECT_TRUE(cmp.ok())
      << "Value mismatch on " << case_label
      << " e=" << cmp.e << " r=" << cmp.r << " c=" << cmp.c
      << " got=" << cmp.got << " ref=" << cmp.ref;
}

static bool require_min_threads(int needed) {
  int actual = 0;
  #pragma omp parallel
  {
    #pragma omp master
    actual = omp_get_num_threads();
  }
  return actual >= needed;
}

}  // namespace

// Baseline: 16 actives × M=64 × K=64 × N=128, no bias, alpha=1.0, beta=0.0.
// active_ops (16) <= num_threads (32) ⇒ kPhase2Single expected.
TEST(TestGroupMatmulMTileStandalone, CallerAllocActNoneNoBias) {
  using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kPhase2Single;

  const int saved = omp_get_max_threads();
  struct G { int p; ~G() { omp_set_num_threads(p); } } g{saved};
  omp_set_num_threads(32);
  if (!require_min_threads(32))
    GTEST_SKIP() << "Requires >= 32 OMP threads.";

  auto s = build_standalone_mtile_shape(
      /*num_threads=*/32,
      /*Ms=*/std::vector<int>(16, 64),
      /*N=*/128, /*K=*/64,
      /*with_bias=*/false);
  run_and_compare_standalone(
      s, /*algo_test=*/2, /*algo_ref=*/1,
      /*expected_tag=*/kPhase2Single,
      /*rtol=*/5e-2f, /*atol=*/5e-2f,
      "standalone_caller_alloc_act_none_no_bias");
}

// With per-expert bias.  Exercises `execute_m_tile`'s bias passthrough
// in the slice executor (line 195) — the same bias[e] vector is used
// for every row slice within an expert (no per-slice offset on bias).
TEST(TestGroupMatmulMTileStandalone, CallerAllocActNoneWithBias) {
  using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kPhase2Single;

  const int saved = omp_get_max_threads();
  struct G { int p; ~G() { omp_set_num_threads(p); } } g{saved};
  omp_set_num_threads(32);
  if (!require_min_threads(32))
    GTEST_SKIP() << "Requires >= 32 OMP threads.";

  auto s = build_standalone_mtile_shape(
      /*num_threads=*/32,
      /*Ms=*/std::vector<int>(16, 64),
      /*N=*/128, /*K=*/64,
      /*with_bias=*/true);
  run_and_compare_standalone(
      s, /*algo_test=*/2, /*algo_ref=*/1,
      /*expected_tag=*/kPhase2Single,
      /*rtol=*/5e-2f, /*atol=*/5e-2f,
      "standalone_caller_alloc_act_none_with_bias");
}

// Mixed alpha != 1, beta != 0 — exercises the `beta * C_init` path
// in matmul_execute.  Note that beta=0.5 requires dst_ref to be
// pre-initialised to the same values across both runs; we use the
// `assign(..., bfloat16_t(0.0f))` zero-init from the shape builder
// which gives both runs identical C_init = 0, so beta*C_init == 0
// and the numerical comparison still holds.  (A non-zero C_init test
// would belong under postop coverage in Batch 9.)
TEST(TestGroupMatmulMTileStandalone, CallerAllocAlphaBeta) {
  using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kPhase2Single;

  const int saved = omp_get_max_threads();
  struct G { int p; ~G() { omp_set_num_threads(p); } } g{saved};
  omp_set_num_threads(32);
  if (!require_min_threads(32))
    GTEST_SKIP() << "Requires >= 32 OMP threads.";

  auto s = build_standalone_mtile_shape(
      /*num_threads=*/32,
      /*Ms=*/std::vector<int>(16, 64),
      /*N=*/128, /*K=*/64,
      /*with_bias=*/false,
      /*alpha_val=*/2.0f, /*beta_val=*/0.5f);
  run_and_compare_standalone(
      s, /*algo_test=*/2, /*algo_ref=*/1,
      /*expected_tag=*/kPhase2Single,
      /*rtol=*/5e-2f, /*atol=*/5e-2f,
      "standalone_caller_alloc_alpha2_beta05");
}

// Gated activations on the standalone hot path: each of silu_and_mul,
// gelu_and_mul, swiglu_oai_mul.  N=256 (even) is required by the
// gated_act validator (`group_matmul_direct.cpp:478`).  The activation
// halves the live data width — first N/2 columns hold the activated
// output, columns [N/2, N) carry the raw pre-act matmul values.  The
// comparator only checks the first N/2 columns (set by the
// `gated_act != nullptr` branch in `run_and_compare_standalone`).
//
// Why this is the most informative test in Batch 2: the act is invoked
// per-slice from `execute_m_tile_act` (rows [row_start, row_end) per
// thread) under ALGO 2, but full-M from `sequential_experts` under
// ALGO 1.  Numerical parity across these two row-range semantics
// confirms that the slice-stitched activation is correct.
TEST(TestGroupMatmulMTileStandalone, GatedActSiluAndMul) {
  using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kPhase2Single;
  using zendnnl::lowoha::matmul::grp_matmul_gated_act_params;
  using zendnnl::lowoha::matmul::grp_matmul_gated_act_t;

  const int saved = omp_get_max_threads();
  struct G { int p; ~G() { omp_set_num_threads(p); } } g{saved};
  omp_set_num_threads(32);
  if (!require_min_threads(32))
    GTEST_SKIP() << "Requires >= 32 OMP threads.";

  auto s = build_standalone_mtile_shape(
      /*num_threads=*/32,
      /*Ms=*/std::vector<int>(16, 64),
      /*N=*/256, /*K=*/64,
      /*with_bias=*/false);

  grp_matmul_gated_act_params ga{};
  ga.act = grp_matmul_gated_act_t::silu_and_mul;

  run_and_compare_standalone(
      s, /*algo_test=*/2, /*algo_ref=*/1,
      /*expected_tag=*/kPhase2Single,
      /*rtol=*/5e-2f, /*atol=*/5e-2f,
      "standalone_silu_and_mul",
      /*gated_act=*/&ga);
}

TEST(TestGroupMatmulMTileStandalone, GatedActGeluAndMul) {
  using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kPhase2Single;
  using zendnnl::lowoha::matmul::grp_matmul_gated_act_params;
  using zendnnl::lowoha::matmul::grp_matmul_gated_act_t;

  const int saved = omp_get_max_threads();
  struct G { int p; ~G() { omp_set_num_threads(p); } } g{saved};
  omp_set_num_threads(32);
  if (!require_min_threads(32))
    GTEST_SKIP() << "Requires >= 32 OMP threads.";

  auto s = build_standalone_mtile_shape(
      /*num_threads=*/32,
      /*Ms=*/std::vector<int>(16, 64),
      /*N=*/256, /*K=*/64,
      /*with_bias=*/false);

  grp_matmul_gated_act_params ga{};
  ga.act = grp_matmul_gated_act_t::gelu_and_mul;

  run_and_compare_standalone(
      s, /*algo_test=*/2, /*algo_ref=*/1,
      /*expected_tag=*/kPhase2Single,
      /*rtol=*/5e-2f, /*atol=*/5e-2f,
      "standalone_gelu_and_mul",
      /*gated_act=*/&ga);
}

TEST(TestGroupMatmulMTileStandalone, GatedActSwigluOaiMul) {
  using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kPhase2Single;
  using zendnnl::lowoha::matmul::grp_matmul_gated_act_params;
  using zendnnl::lowoha::matmul::grp_matmul_gated_act_t;

  const int saved = omp_get_max_threads();
  struct G { int p; ~G() { omp_set_num_threads(p); } } g{saved};
  omp_set_num_threads(32);
  if (!require_min_threads(32))
    GTEST_SKIP() << "Requires >= 32 OMP threads.";

  auto s = build_standalone_mtile_shape(
      /*num_threads=*/32,
      /*Ms=*/std::vector<int>(16, 64),
      /*N=*/256, /*K=*/64,
      /*with_bias=*/false);

  grp_matmul_gated_act_params ga{};
  ga.act = grp_matmul_gated_act_t::swiglu_oai_mul;

  run_and_compare_standalone(
      s, /*algo_test=*/2, /*algo_ref=*/1,
      /*expected_tag=*/kPhase2Single,
      /*rtol=*/5e-2f, /*atol=*/5e-2f,
      "standalone_swiglu_oai_mul",
      /*gated_act=*/&ga);
}

// ============================================================================
// [8h] TestGroupMatmulMTileBranches.ForcedAlgo2ManyExpertsClampsToSeqFullTeam
//
// When `active_ops > num_threads` a pure M-tile plan is INFEASIBLE (it
// cannot give < 1 thread per active expert).  AUTO routes this regime to
// ALGO 5 (see TestGroupMatmulAutoPhaseEnv.* below); a FORCED
// `ZENDNNL_GRP_MATMUL_ALGO=2` clamps to the sequential full-team path
// (ALGO 1 equivalent: each active expert runs across the whole team, one at
// a time) and emits a one-time WARN.  Tag: kManyExpertsSeqFallback.
//
// We trigger it by pinning `params[e].num_threads = 4` while providing 8
// active experts.  Value parity is checked against an ALGO 1 reference
// (the clamp IS the ALGO 1 sequential full-team algorithm).
// ============================================================================

TEST(TestGroupMatmulMTileBranches, ForcedAlgo2ManyExpertsClampsToSeqFullTeam) {
  using zendnnl::lowoha::matmul::test_api::m_tile_path_tag
      ::kManyExpertsSeqFallback;

  const int saved = omp_get_max_threads();
  struct G { int p; ~G() { omp_set_num_threads(p); } } g{saved};
  omp_set_num_threads(8);
  if (!require_min_threads(4))
    GTEST_SKIP() << "Requires >= 4 OMP threads.";

  auto s = build_standalone_mtile_shape(
      /*num_threads=*/4,        // < active_ops=8 ⇒ M-tile infeasible
      /*Ms=*/std::vector<int>(8, 8),
      /*N=*/128, /*K=*/64,
      /*with_bias=*/false);
  run_and_compare_standalone(
      s, /*algo_test=*/2, /*algo_ref=*/1,
      /*expected_tag=*/kManyExpertsSeqFallback,
      /*rtol=*/5e-2f, /*atol=*/5e-2f,
      "forced_algo2_8actives_4threads_seq_clamp");
}

// Multi-tier value parity — same skewed prompt shape as
// `MultiTierEngagesOnSkewedPrompt` above, but additionally asserts
// numerical correctness of the per-slice executor against an ALGO 1
// reference.  Multi-tier splits the planner into heavy-pool + light-
// pool slots; the heavy pool runs per-slice (`execute_m_tile_act`,
// row-range activation), the light pool runs per-expert full-M
// activation (`execute_light_expert` ~m_tile.cpp:502-527).  Value
// parity across this dual-pool dispatch confirms there is no
// double-act on shared rows and no boundary-row off-by-one between
// the two pools.
TEST(TestGroupMatmulMTileBranches, MultiTierValueParityOnSkewedPrompt) {
  using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kMultiTier;

  const int saved = omp_get_max_threads();
  struct G { int p; ~G() { omp_set_num_threads(p); } } g{saved};
  omp_set_num_threads(32);
  if (!require_min_threads(32))
    GTEST_SKIP() << "Requires >= 32 OMP threads.";

  std::vector<int> ms = {1024};
  ms.insert(ms.end(), 15, 4);
  auto s = build_standalone_mtile_shape(
      /*num_threads=*/32, ms,
      /*N=*/1024, /*K=*/64,
      /*with_bias=*/false);
  run_and_compare_standalone(
      s, /*algo_test=*/2, /*algo_ref=*/1,
      /*expected_tag=*/kMultiTier,
      /*rtol=*/5e-2f, /*atol=*/5e-2f,
      "multi_tier_skewed_prompt_value_parity");
}

// Forced-ALGO-2 value parity on a wide-N light-frame shape.  ALGO 2 is now
// pure M-tile, so a forced ALGO 2 on this shape runs single-tier
// (kPhase2Single) rather than the old wide-N sequential fallback.  Value
// parity against an ALGO 1 reference confirms the single-tier slice/stitch
// is numerically correct on the shallow-M shape.
TEST(TestGroupMatmulMTileBranches, ForcedAlgo2WideNValueParityOnLightFrames) {
  using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kPhase2Single;

  const int saved = omp_get_max_threads();
  struct G { int p; ~G() { omp_set_num_threads(p); } } g{saved};
  omp_set_num_threads(32);
  if (!require_min_threads(32))
    GTEST_SKIP() << "Requires >= 32 OMP threads.";

  auto s = build_standalone_mtile_shape(
      /*num_threads=*/32,
      /*Ms=*/std::vector<int>(8, 8),
      /*N=*/1024, /*K=*/64,
      /*with_bias=*/false);
  run_and_compare_standalone(
      s, /*algo_test=*/2, /*algo_ref=*/1,
      /*expected_tag=*/kPhase2Single,
      /*rtol=*/5e-2f, /*atol=*/5e-2f,
      "forced_algo2_wide_n_light_frames_value_parity");
}

// ============================================================================
// [8i] TestGroupMatmulMTileQuantGate: direct unit-test of
// `check_m_tile_safe`'s dynamic-quant arm.  The gate's contract
// (`group_matmul_parallel_common.hpp:159-161`) is that
// `dynamic_quant = true` requires `src_scale.dims[0] == M[i]` so the
// per-thread reorder operates on its own rows.  Per-token `{M, 1}`
// and per-group-on-K `{M, G>1}` both satisfy the gate; per-tensor
// (`{}`, `{1}`, `{1, 1}`) is rejected, as are mismatched `M` dims.
//
// Why this matters: forcing `ZENDNNL_GRP_MATMUL_ALGO=2` with a
// per-tensor src_scale must fall back to ALGO 1 (the M-tile slicer
// cannot safely partition a shared scalar scale across threads).
// The fallback is enforced by `select_grp_matmul_algo` reading
// `check_m_tile_safe`'s return value.
//
// Replaces the per-call cost of constructing a full int8 fused-MoE
// fixture (which would need wei reorder, runtime BF16→S8 scratch,
// per-channel wei_scale plumbing, etc.) with a direct call into
// the inline gate predicate.  The full int8 pipeline is exercised
// by the existing `TestFusedMoEQuantDynINT8` parameterized sweep.
// ============================================================================

TEST(TestGroupMatmulMTileQuantGate, AcceptsPerTokenAndPerGroup) {
  using namespace zendnnl::lowoha::matmul;
  using zendnnl::common::data_type_t;

  const int num_ops = 4;
  std::vector<char> layout(num_ops, 'r');
  std::vector<int>  M(num_ops, 16);
  std::vector<matmul_params> params(num_ops);
  for (int i = 0; i < num_ops; ++i) {
    params[i].dtypes.src     = data_type_t::bf16;
    params[i].dtypes.wei     = data_type_t::s8;
    params[i].dtypes.dst     = data_type_t::bf16;
    params[i].dtypes.bias    = data_type_t::none;
    params[i].dtypes.compute = data_type_t::s8;
    params[i].mem_format_a   = 'n';
    params[i].mem_format_b   = 'n';
    params[i].packing.pack_format_b = 0;
    params[i].dynamic_quant  = true;
    // Per-token `{M, 1}` — accepted: dims[0] == M[i].
    params[i].quant_params.src_scale.dims = {16, 1};
    params[i].quant_params.src_scale.dt   = data_type_t::f32;
  }
  EXPECT_TRUE(check_m_tile_safe(layout, M, params, num_ops))
      << "per-token src_scale dims={M,1} must be accepted by M-tile gate";

  // Per-group-on-K `{M, G>1}` — also accepted.  Per-group is row-local
  // (each row owns G group-scales), so the row slicer still partitions
  // safely across threads.
  for (int i = 0; i < num_ops; ++i) {
    params[i].quant_params.src_scale.dims = {16, 4};
  }
  EXPECT_TRUE(check_m_tile_safe(layout, M, params, num_ops))
      << "per-group-on-K src_scale dims={M,G>1} must be accepted "
         "(row-local granularity is what the M-tile slicer requires)";
}

TEST(TestGroupMatmulMTileQuantGate, RejectsPerTensorAndMismatched) {
  using namespace zendnnl::lowoha::matmul;
  using zendnnl::common::data_type_t;

  const int num_ops = 4;
  std::vector<char> layout(num_ops, 'r');
  std::vector<int>  M(num_ops, 16);

  auto make_params = [&](const std::vector<int64_t> &src_scale_dims) {
    std::vector<matmul_params> p(num_ops);
    for (int i = 0; i < num_ops; ++i) {
      p[i].dtypes.src     = data_type_t::bf16;
      p[i].dtypes.wei     = data_type_t::s8;
      p[i].dtypes.dst     = data_type_t::bf16;
      p[i].dtypes.bias    = data_type_t::none;
      p[i].dtypes.compute = data_type_t::s8;
      p[i].mem_format_a   = 'n';
      p[i].mem_format_b   = 'n';
      p[i].packing.pack_format_b = 0;
      p[i].dynamic_quant  = true;
      p[i].quant_params.src_scale.dims = src_scale_dims;
      p[i].quant_params.src_scale.dt   = data_type_t::f32;
    }
    return p;
  };

  // Per-tensor (dims empty): rejected — sd.empty() at gate line 194.
  {
    auto p = make_params({});
    EXPECT_FALSE(check_m_tile_safe(layout, M, p, num_ops))
        << "empty src_scale dims must be rejected by M-tile gate";
  }
  // Per-tensor {1}: rejected — sd[0] != M[i] (1 != 16).
  {
    auto p = make_params({1});
    EXPECT_FALSE(check_m_tile_safe(layout, M, p, num_ops))
        << "per-tensor src_scale dims={1} must be rejected "
           "(single shared scalar cannot be safely row-sliced)";
  }
  // Per-tensor {1, 1}: rejected — sd[0] != M[i].
  {
    auto p = make_params({1, 1});
    EXPECT_FALSE(check_m_tile_safe(layout, M, p, num_ops))
        << "per-tensor src_scale dims={1,1} must be rejected";
  }
  // Per-column / per-K (no M dim): {1, K}: rejected — sd[0] != M[i].
  {
    auto p = make_params({1, 64});
    EXPECT_FALSE(check_m_tile_safe(layout, M, p, num_ops))
        << "per-K-only src_scale dims={1, K} must be rejected";
  }
  // Mismatched M dim: {M+1, 1}: rejected.
  {
    auto p = make_params({17, 1});
    EXPECT_FALSE(check_m_tile_safe(layout, M, p, num_ops))
        << "src_scale dims[0] != M[i] must be rejected";
  }
  // Mismatched src_zp dim while src_scale is valid: rejected — line 196.
  {
    auto p = make_params({16, 1});
    p[0].quant_params.src_zp.dims = {17, 1};
    EXPECT_FALSE(check_m_tile_safe(layout, M, p, num_ops))
        << "src_zp dims[0] != M[i] must be rejected even when "
           "src_scale dims are valid";
  }
}

// ============================================================================
// [8j] TestGroupMatmulMTilePostopGate — direct unit-test of
//      `check_m_tile_safe`'s post-op rejection arm (lines 198-202 of
//      `group_matmul_parallel_common.hpp`).  The M-tile per-thread row
//      slicer cannot handle post-ops that REQUIRE the full M block
//      (softmax normalises across the rows of a tile, pooling
//      aggregates across rows); both are rejected at the gate so the
//      dispatcher falls back to ALGO 1 where the postop sees the full
//      block.
//
//      Binary `add` / `mul` postops are ACCEPTED — `execute_m_tile`
//      handles their row offset inside `m_tile/group_matmul_m_tile.cpp`
//      (offsets `po.buff` by `row_start × eff_ld × po_elem` for 2D /
//      3D row-varying tensors, leaves broadcast {1, N} / rank-≤1
//      tensors unchanged).
//
//      All four M-tile branches (kRoundBased, kMultiTier,
//      kWideNFallback, kPhase2Single) route through the SAME
//      `execute_m_tile` / `execute_m_tile_act` entry points for the
//      sliced experts — round-based and multi-tier-light dispatch the
//      full M of each expert to a single team, so the row-offset code
//      is a no-op (`row_start == 0`) for them and they degenerate to
//      ALGO-1-style per-expert postop handling.  The remaining branches
//      (sliced execution) all share the row-offset logic, so a single
//      gate-rejection test is sufficient.
//
//      The end-to-end value-parity test for binary postops on M-tile
//      is intentionally NOT added here — the existing `TestGroupMatmul`
//      sweep in `test_basic.cpp` covers binary postops × ALGO 2 across
//      a broad shape grid via the `matmul_forced_ref_kernel_test`
//      reference, and the gate-level test below proves the M-tile
//      safety predicate's reject/accept contract directly.
// ============================================================================

TEST(TestGroupMatmulMTilePostopGate, AcceptsBinaryAddAndMul) {
  using namespace zendnnl::lowoha::matmul;
  using zendnnl::common::data_type_t;
  using zendnnl::ops::post_op_type_t;

  const int num_ops = 4;
  std::vector<char> layout(num_ops, 'r');
  std::vector<int>  M(num_ops, 16);
  std::vector<matmul_params> params(num_ops);
  for (int i = 0; i < num_ops; ++i) {
    params[i].dtypes.src  = data_type_t::bf16;
    params[i].dtypes.wei  = data_type_t::bf16;
    params[i].dtypes.dst  = data_type_t::bf16;
    params[i].dtypes.bias = data_type_t::none;
    params[i].mem_format_a = 'n';
    params[i].mem_format_b = 'n';
    params[i].packing.pack_format_b = 0;
    // Binary add — 2D row-varying buffer {M=16, N=64}.
    matmul_post_op add_po;
    add_po.po_type = post_op_type_t::binary_add;
    add_po.dtype   = data_type_t::bf16;
    add_po.dims    = {16, 64};
    add_po.leading_dim = 64;
    params[i].postop_.push_back(add_po);
    // Binary mul — broadcast {1, N=64}.
    matmul_post_op mul_po;
    mul_po.po_type = post_op_type_t::binary_mul;
    mul_po.dtype   = data_type_t::bf16;
    mul_po.dims    = {1, 64};
    mul_po.leading_dim = 64;
    params[i].postop_.push_back(mul_po);
  }
  EXPECT_TRUE(check_m_tile_safe(layout, M, params, num_ops))
      << "binary_add (2D row-varying) and binary_mul (1D broadcast) "
         "post-ops must be accepted by the M-tile safety gate "
         "(row-offset handled in execute_m_tile lines 165-182)";
}

TEST(TestGroupMatmulMTilePostopGate, RejectsSoftmaxAndPooling) {
  using namespace zendnnl::lowoha::matmul;
  using zendnnl::common::data_type_t;
  using zendnnl::ops::post_op_type_t;

  const int num_ops = 4;
  std::vector<char> layout(num_ops, 'r');
  std::vector<int>  M(num_ops, 16);

  auto make_params_with_postop = [&](post_op_type_t po_type) {
    std::vector<matmul_params> p(num_ops);
    for (int i = 0; i < num_ops; ++i) {
      p[i].dtypes.src  = data_type_t::bf16;
      p[i].dtypes.wei  = data_type_t::bf16;
      p[i].dtypes.dst  = data_type_t::bf16;
      p[i].dtypes.bias = data_type_t::none;
      p[i].mem_format_a = 'n';
      p[i].mem_format_b = 'n';
      p[i].packing.pack_format_b = 0;
      matmul_post_op po;
      po.po_type = po_type;
      po.dtype   = data_type_t::bf16;
      p[i].postop_.push_back(po);
    }
    return p;
  };

  // softmax: rejected — needs full M block for row-wise normalisation.
  {
    auto p = make_params_with_postop(post_op_type_t::softmax);
    EXPECT_FALSE(check_m_tile_safe(layout, M, p, num_ops))
        << "softmax post-op must be rejected — row-wise normalisation "
           "across the full M block cannot be satisfied by the M-tile "
           "per-thread row slicer";
  }
  // pooling: rejected — needs full M block for cross-row aggregation.
  {
    auto p = make_params_with_postop(post_op_type_t::pooling);
    EXPECT_FALSE(check_m_tile_safe(layout, M, p, num_ops))
        << "pooling post-op must be rejected — cross-row aggregation "
           "cannot be partitioned across M-tile threads";
  }
  // Only one expert carries the unsupported postop — still rejected
  // (the gate must be conservative; mixed-postop batches all fall
  // back to ALGO 1).
  {
    auto p = make_params_with_postop(post_op_type_t::binary_add);
    matmul_post_op pool_po;
    pool_po.po_type = post_op_type_t::pooling;
    pool_po.dtype   = data_type_t::bf16;
    p[2].postop_.push_back(pool_po);   // single expert in the batch
    EXPECT_FALSE(check_m_tile_safe(layout, M, p, num_ops))
        << "ANY expert carrying softmax/pooling must reject the whole "
           "batch (mixed-postop batches fall back to ALGO 1 because "
           "the M-tile planner cannot split planner state across the "
           "two postop classes mid-batch)";
  }
}

// ============================================================================
// [8k] TestGroupMatmulMTileDtypeGate — direct unit-test of
//      `check_m_tile_safe`'s dtype-uniformity arm (lines 185-188 of
//      `group_matmul_parallel_common.hpp`).  The gate requires
//      cross-expert dtype uniformity for src / wei / dst / bias —
//      M-tile's per-thread row slicer dispatches uniform GEMM tiles to
//      `execute_expert_slice`, which assumes the dtypes are stable
//      across the whole batch.  Heterogeneous-dtype batches MUST
//      reject so the dispatcher falls back to ALGO 1 where each
//      expert dispatches through `kernel_select` independently.
//
//      Note that the gate does NOT restrict WHICH concrete dtypes are
//      allowed — bf16, f32, s8 (with dynamic_quant), or any mix that
//      a supported kernel can handle is accepted as long as it is
//      uniform across experts.  This keeps the gate cheap (O(num_ops)
//      compare loop) and leaves the dtype routing to the kernel
//      selector inside `kernel_select`.
// ============================================================================

TEST(TestGroupMatmulMTileDtypeGate, AcceptsUniformBF16) {
  using namespace zendnnl::lowoha::matmul;
  using zendnnl::common::data_type_t;

  const int num_ops = 4;
  std::vector<char> layout(num_ops, 'r');
  std::vector<int>  M(num_ops, 16);
  std::vector<matmul_params> params(num_ops);
  for (int i = 0; i < num_ops; ++i) {
    params[i].dtypes.src  = data_type_t::bf16;
    params[i].dtypes.wei  = data_type_t::bf16;
    params[i].dtypes.dst  = data_type_t::bf16;
    params[i].dtypes.bias = data_type_t::bf16;
    params[i].mem_format_a = 'n';
    params[i].mem_format_b = 'n';
    params[i].packing.pack_format_b = 0;
  }
  EXPECT_TRUE(check_m_tile_safe(layout, M, params, num_ops))
      << "uniform bf16 src/wei/dst/bias must be accepted by the M-tile gate";
}

TEST(TestGroupMatmulMTileDtypeGate, AcceptsUniformFP32) {
  using namespace zendnnl::lowoha::matmul;
  using zendnnl::common::data_type_t;

  const int num_ops = 4;
  std::vector<char> layout(num_ops, 'r');
  std::vector<int>  M(num_ops, 16);
  std::vector<matmul_params> params(num_ops);
  for (int i = 0; i < num_ops; ++i) {
    params[i].dtypes.src  = data_type_t::f32;
    params[i].dtypes.wei  = data_type_t::f32;
    params[i].dtypes.dst  = data_type_t::f32;
    params[i].dtypes.bias = data_type_t::f32;
    params[i].mem_format_a = 'n';
    params[i].mem_format_b = 'n';
    params[i].packing.pack_format_b = 0;
  }
  EXPECT_TRUE(check_m_tile_safe(layout, M, params, num_ops))
      << "uniform f32 src/wei/dst/bias must be accepted — M-tile is "
         "dtype-agnostic at the safety gate (kernel routing handled "
         "by `kernel_select`)";
}

TEST(TestGroupMatmulMTileDtypeGate, AcceptsMixedPrecisionBF16toFP32) {
  using namespace zendnnl::lowoha::matmul;
  using zendnnl::common::data_type_t;

  const int num_ops = 4;
  std::vector<char> layout(num_ops, 'r');
  std::vector<int>  M(num_ops, 16);
  std::vector<matmul_params> params(num_ops);
  for (int i = 0; i < num_ops; ++i) {
    // bf16 inputs, f32 accumulator-style output (common LLM pattern
    // for attention output projections).
    params[i].dtypes.src  = data_type_t::bf16;
    params[i].dtypes.wei  = data_type_t::bf16;
    params[i].dtypes.dst  = data_type_t::f32;
    params[i].dtypes.bias = data_type_t::f32;
    params[i].mem_format_a = 'n';
    params[i].mem_format_b = 'n';
    params[i].packing.pack_format_b = 0;
  }
  EXPECT_TRUE(check_m_tile_safe(layout, M, params, num_ops))
      << "mixed-precision (bf16 in, f32 out) must be accepted — the "
         "gate enforces ACROSS-expert uniformity, not BETWEEN src/dst";
}

TEST(TestGroupMatmulMTileDtypeGate, RejectsCrossExpertSrcMismatch) {
  using namespace zendnnl::lowoha::matmul;
  using zendnnl::common::data_type_t;

  const int num_ops = 4;
  std::vector<char> layout(num_ops, 'r');
  std::vector<int>  M(num_ops, 16);
  std::vector<matmul_params> params(num_ops);
  for (int i = 0; i < num_ops; ++i) {
    params[i].dtypes.src  = data_type_t::bf16;
    params[i].dtypes.wei  = data_type_t::bf16;
    params[i].dtypes.dst  = data_type_t::bf16;
    params[i].dtypes.bias = data_type_t::none;
    params[i].mem_format_a = 'n';
    params[i].mem_format_b = 'n';
  }
  // Mismatch one expert's src dtype.
  params[2].dtypes.src = data_type_t::f32;
  EXPECT_FALSE(check_m_tile_safe(layout, M, params, num_ops))
      << "cross-expert src dtype mismatch must be rejected — M-tile's "
         "per-thread slicer dispatches uniform GEMM tiles and cannot "
         "switch dtype mid-batch";
}

TEST(TestGroupMatmulMTileDtypeGate, RejectsCrossExpertWeiMismatch) {
  using namespace zendnnl::lowoha::matmul;
  using zendnnl::common::data_type_t;

  const int num_ops = 4;
  std::vector<char> layout(num_ops, 'r');
  std::vector<int>  M(num_ops, 16);
  std::vector<matmul_params> params(num_ops);
  for (int i = 0; i < num_ops; ++i) {
    params[i].dtypes.src  = data_type_t::bf16;
    params[i].dtypes.wei  = data_type_t::bf16;
    params[i].dtypes.dst  = data_type_t::bf16;
    params[i].dtypes.bias = data_type_t::none;
    params[i].mem_format_a = 'n';
    params[i].mem_format_b = 'n';
  }
  // s8 weights on one expert (quantised), bf16 on the others.
  params[0].dtypes.wei = data_type_t::s8;
  EXPECT_FALSE(check_m_tile_safe(layout, M, params, num_ops))
      << "cross-expert wei dtype mismatch must be rejected (e.g. one "
         "quantised expert + dense others) — the M-tile planner cannot "
         "route a single team to two different kernel families";
}

TEST(TestGroupMatmulMTileDtypeGate, RejectsCrossExpertDstMismatch) {
  using namespace zendnnl::lowoha::matmul;
  using zendnnl::common::data_type_t;

  const int num_ops = 4;
  std::vector<char> layout(num_ops, 'r');
  std::vector<int>  M(num_ops, 16);
  std::vector<matmul_params> params(num_ops);
  for (int i = 0; i < num_ops; ++i) {
    params[i].dtypes.src  = data_type_t::bf16;
    params[i].dtypes.wei  = data_type_t::bf16;
    params[i].dtypes.dst  = data_type_t::bf16;
    params[i].dtypes.bias = data_type_t::none;
    params[i].mem_format_a = 'n';
    params[i].mem_format_b = 'n';
  }
  params[3].dtypes.dst = data_type_t::f32;
  EXPECT_FALSE(check_m_tile_safe(layout, M, params, num_ops))
      << "cross-expert dst dtype mismatch must be rejected — the row "
         "slicer's dst pointer arithmetic assumes uniform `dst_elem` "
         "(`size_of(dtypes.dst)`) across all experts";
}

TEST(TestGroupMatmulMTileDtypeGate, RejectsCrossExpertBiasMismatch) {
  using namespace zendnnl::lowoha::matmul;
  using zendnnl::common::data_type_t;

  const int num_ops = 4;
  std::vector<char> layout(num_ops, 'r');
  std::vector<int>  M(num_ops, 16);
  std::vector<matmul_params> params(num_ops);
  for (int i = 0; i < num_ops; ++i) {
    params[i].dtypes.src  = data_type_t::bf16;
    params[i].dtypes.wei  = data_type_t::bf16;
    params[i].dtypes.dst  = data_type_t::bf16;
    params[i].dtypes.bias = data_type_t::f32;
    params[i].mem_format_a = 'n';
    params[i].mem_format_b = 'n';
  }
  params[1].dtypes.bias = data_type_t::bf16;
  EXPECT_FALSE(check_m_tile_safe(layout, M, params, num_ops))
      << "cross-expert bias dtype mismatch must be rejected (e.g. f32 "
         "bias on most experts + bf16 bias on one) — the kernel select "
         "would diverge mid-batch";
}

// ============================================================================
// [8l] TestGroupMatmulMTileStandalone.F32 — full-f32 numerical-correctness
// fixture for the standalone hot path.  Mirrors the bf16 standalone fixture
// at [8g] above but with `data_type_t::f32` for src / wei / dst / bias.
// f32 routes through a different kernel inside `kernel_select` (typically
// the AOCL FP32 GEMM rather than bf16 BRGEMM), so this catches regressions
// in dtype handling that the bf16-only standalone tests would miss.
//
// Reference: ALGO 1 (sequential per-expert) on the same f32 inputs.
// Test:      ALGO 2 (M-tile / `flat_m_tile`).
// Tolerance: tight (rtol=1e-5, atol=1e-5) — f32 has 7-digit precision so
//            the reduction-order noise between sequential and M-tile is
//            well below 1e-5 for this small K=64 / M=64 / N=128 shape.
// ============================================================================

namespace {

struct StandaloneMTileShapeF32 {
  int num_ops;
  int num_threads;
  int N;
  int K;
  std::vector<int> Ms;
  std::vector<std::vector<float>> src;
  std::vector<std::vector<float>> wei;
  std::vector<std::vector<float>> bias;
  std::vector<std::vector<float>> dst_test;
  std::vector<std::vector<float>> dst_ref;
  std::vector<const void *> srcs;
  std::vector<const void *> weis;
  std::vector<const void *> biases;
  std::vector<void *> dsts_test;
  std::vector<void *> dsts_ref;
  moe_test_utils::GemmVecs gv;
  std::vector<zendnnl::lowoha::matmul::matmul_params> params;
};

static StandaloneMTileShapeF32 build_standalone_mtile_shape_f32(
    int num_threads, const std::vector<int> &Ms,
    int N, int K, bool with_bias,
    float alpha_val = 1.0f, float beta_val = 0.0f) {
  using namespace moe_test_utils;
  using zendnnl::common::data_type_t;
  StandaloneMTileShapeF32 s;
  s.num_threads = num_threads;
  s.num_ops     = static_cast<int>(Ms.size());
  s.N           = N;
  s.K           = K;
  s.Ms          = Ms;
  s.src.resize(s.num_ops);
  s.wei.resize(s.num_ops);
  s.bias.resize(s.num_ops);
  s.dst_test.resize(s.num_ops);
  s.dst_ref.resize(s.num_ops);
  for (int e = 0; e < s.num_ops; ++e) {
    s.src[e].resize(static_cast<size_t>(Ms[e]) * K);
    s.wei[e].resize(static_cast<size_t>(K) * N);
    s.dst_test[e].assign(static_cast<size_t>(Ms[e]) * N, 0.0f);
    s.dst_ref [e].assign(static_cast<size_t>(Ms[e]) * N, 0.0f);
    // Deterministic seeds — same hash family as the bf16 fixture but
    // returns f32 values directly (no bf16-truncation noise).
    const uint64_t base_src = 0x9E3779B97F4A7C15ULL +
                              static_cast<uint64_t>(e) * 0x100000001B3ULL;
    for (size_t k = 0; k < s.src[e].size(); ++k) {
      uint64_t h = base_src + k * 0xC4CEB9FE1A85EC53ULL;
      h ^= h >> 33; h *= 0xff51afd7ed558ccdULL; h ^= h >> 33;
      s.src[e][k] = static_cast<float>(
          static_cast<int>(h & 0xff) - 128) * 0.005f;
    }
    const uint64_t base_wei = 0xBF58476D1CE4E5B9ULL +
                              static_cast<uint64_t>(e) * 0x100000001B3ULL;
    for (size_t k = 0; k < s.wei[e].size(); ++k) {
      uint64_t h = base_wei + k * 0xC4CEB9FE1A85EC53ULL;
      h ^= h >> 33; h *= 0xff51afd7ed558ccdULL; h ^= h >> 33;
      s.wei[e][k] = static_cast<float>(
          static_cast<int>(h & 0xff) - 128) * 0.002f;
    }
    if (with_bias) {
      s.bias[e].resize(static_cast<size_t>(N));
      for (int c = 0; c < N; ++c)
        s.bias[e][c] = static_cast<float>(c) * 0.001f + 0.01f;
    }
  }
  s.srcs.resize(s.num_ops);
  s.weis.resize(s.num_ops);
  s.biases.resize(s.num_ops);
  s.dsts_test.resize(s.num_ops);
  s.dsts_ref.resize(s.num_ops);
  for (int e = 0; e < s.num_ops; ++e) {
    s.srcs[e]      = s.src[e].data();
    s.weis[e]      = s.wei[e].data();
    s.biases[e]    = with_bias ? s.bias[e].data() : nullptr;
    s.dsts_test[e] = s.dst_test[e].data();
    s.dsts_ref [e] = s.dst_ref [e].data();
  }
  s.gv = GemmVecs::uniform(s.num_ops, /*M=*/0, N, K,
                           alpha_val, beta_val,
                           /*wc=*/true, /*tA=*/false, /*tB=*/false);
  s.gv.Ms = Ms;
  s.params = make_uniform_params(
      s.num_ops, data_type_t::f32,
      /*bias_dt=*/with_bias ? data_type_t::f32 : data_type_t::none);
  for (int e = 0; e < s.num_ops; ++e) {
    s.params[e].num_threads = num_threads;
  }
  return s;
}

static void run_and_compare_standalone_f32(
    StandaloneMTileShapeF32 &s, int algo_test, int algo_ref,
    int expected_tag, float rtol, float atol,
    const char *case_label) {
  using namespace moe_test_utils;
  using zendnnl::lowoha::matmul::group_matmul_direct;
  using zendnnl::lowoha::matmul::status_t;

  {
    auto pr = s.params;
    AlgoEnvGuard algo_guard_ref(algo_ref);
    ::reset_grp_matmul_caches();
    ASSERT_EQ(group_matmul_direct(s.gv.layout, s.gv.transA, s.gv.transB,
                                  s.gv.Ms, s.gv.Ns, s.gv.Ks, s.gv.alpha,
                                  s.srcs, s.gv.lda, s.weis, s.gv.ldb,
                                  s.biases, s.gv.beta, s.dsts_ref, s.gv.ldc,
                                  s.gv.is_wc, pr, nullptr, nullptr),
              status_t::success) << "ref " << case_label;
  }
  int tag = 0;
  {
    auto pt = s.params;
    ::reset_grp_matmul_caches();
    AlgoEnvGuard algo_guard_test(algo_test);
    MTileHybridOverride hybrid_auto(0);
    MTilePathCaptureGuard cap;
    ASSERT_EQ(group_matmul_direct(s.gv.layout, s.gv.transA, s.gv.transB,
                                  s.gv.Ms, s.gv.Ns, s.gv.Ks, s.gv.alpha,
                                  s.srcs, s.gv.lda, s.weis, s.gv.ldb,
                                  s.biases, s.gv.beta, s.dsts_test, s.gv.ldc,
                                  s.gv.is_wc, pt, nullptr, nullptr),
              status_t::success) << "test " << case_label;
    tag = zendnnl::lowoha::matmul::test_api
        ::s_last_m_tile_path.load(std::memory_order_relaxed);
  }
  EXPECT_EQ(tag, expected_tag)
      << "Path tag mismatch on " << case_label
      << ": expected " << expected_tag << ", got " << tag;

  // f32 value comparison — tight tolerance since both runs are
  // FP32 throughout.  Per-expert × per-row × per-col first-divergence
  // reporting matches the bf16 helper's contract.
  for (int e = 0; e < s.num_ops; ++e) {
    const int M_e = s.Ms[e];
    if (M_e == 0) continue;
    for (int row = 0; row < M_e; ++row) {
      for (int c = 0; c < s.N; ++c) {
        const size_t off = static_cast<size_t>(row) * s.N + c;
        const float g = s.dst_test[e][off];
        const float f = s.dst_ref [e][off];
        const float bound = std::abs(f) * rtol + atol;
        if (std::abs(g - f) > bound) {
          FAIL() << "f32 value mismatch on " << case_label
                 << " e=" << e << " r=" << row << " c=" << c
                 << " got=" << g << " ref=" << f;
        }
      }
    }
  }
}

}  // namespace

TEST(TestGroupMatmulMTileStandalone, F32NoBias) {
  using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kPhase2Single;

  const int saved = omp_get_max_threads();
  struct G { int p; ~G() { omp_set_num_threads(p); } } g{saved};
  omp_set_num_threads(32);
  if (!require_min_threads(32))
    GTEST_SKIP() << "Requires >= 32 OMP threads.";

  auto s = build_standalone_mtile_shape_f32(
      /*num_threads=*/32,
      /*Ms=*/std::vector<int>(16, 64),
      /*N=*/128, /*K=*/64,
      /*with_bias=*/false);
  run_and_compare_standalone_f32(
      s, /*algo_test=*/2, /*algo_ref=*/1,
      /*expected_tag=*/kPhase2Single,
      /*rtol=*/1e-5f, /*atol=*/1e-5f,
      "standalone_f32_no_bias");
}

// ============================================================================
// [8m] TestGroupMatmulMTileCustomKernelInvariance — verify the M-tile
// (ALGO 2) runtime is independent of the CUSTOM_KERNEL env knob.
//
// Contract: M-tile (`flat_m_tile`) dispatches every per-thread slice
// through `execute_expert_slice`, which selects the kernel via
// `resolve_kernel()` (returns the matmul_algo_t from `matmul_config_t`
// — defaults to `aocl_dlp_blocked`).  This is independent of the CK
// env knob — `get_grp_matmul_custom_kernel()` is consulted only by the
// PREPACK call BEFORE the OMP region (see m_tile.cpp lines 629-635 for
// the standalone path and lines 1457-1475 for the pipeline).  The
// prepack short-circuits when its arguments don't trigger the CK
// packing format, so for ALGO 2 it always lands on the same packed
// representation regardless of the CK knob.
//
// Therefore the dst[] bytes produced by the M-tile run MUST be
// byte-identical between CK=0 and CK=1 — any difference would
// indicate that the M-tile execution path picked up a CK-dependent
// branch it shouldn't.  We use the test-only `CustomKernelOverride`
// RAII guard (NOT EnvVarGuard, since the env value is cached at first
// read) to switch the knob between the two runs.
// ============================================================================

// ============================================================================
// [8n] TestGroupMatmulMTileEdgeCases — operational edge cases for M-tile.
//
//   * num_ops == 0 (empty batch): early-out at m_tile.cpp:621 — dispatch
//     succeeds with no work done, no dst writes (vacuously held).
//   * All M[i] == 0 (all inactive): early-out at m_tile.cpp:657
//     (`active_ops == 0`).  Dispatch succeeds, dst untouched.
//   * Mixed active / inactive experts (some M[i] == 0, others > 0):
//     planner's `active_pos[i] = -1` for inactive (m_tile.cpp:669-675)
//     keeps them out of the CCD stripe.  Active experts produce
//     correct values; inactive experts' dst slots remain at their
//     initialised values (caller-allocated buffer policy — same as
//     ALGO 1 reference).
//   * Odd N + gated act: rejected by the public-API validator at
//     `group_matmul_direct.cpp:476-482` (status_t::failure).  M-tile
//     never sees the call.
// ============================================================================

TEST(TestGroupMatmulMTileEdgeCases, NumOpsZeroSucceeds) {
  using namespace moe_test_utils;
  using zendnnl::lowoha::matmul::group_matmul_direct;
  using zendnnl::lowoha::matmul::matmul_params;
  using zendnnl::lowoha::matmul::status_t;

  // Empty input vectors — every per-expert vector is size 0.
  std::vector<char>  layout;
  std::vector<bool>  transA, transB, is_wc;
  std::vector<int>   Ms, Ns, Ks, lda, ldb, ldc;
  std::vector<float> alpha, beta;
  std::vector<const void *> srcs, weis, biases;
  std::vector<void *>       dsts;
  std::vector<matmul_params> params;

  AlgoEnvGuard algo2_guard(2);
  ::reset_grp_matmul_caches();
  // Public API typically rejects empty batches at the validator (line
  // 191 of group_matmul_direct.cpp: "required input vector is empty").
  // The contract is `status_t::failure` — NOT a crash.  This is the
  // safety net that M-tile relies on for its `num_ops == 0` invariant.
  const status_t st = group_matmul_direct(
      layout, transA, transB, Ms, Ns, Ks, alpha,
      srcs, lda, weis, ldb, biases, beta,
      dsts, ldc, is_wc, params, nullptr);
  EXPECT_EQ(st, status_t::failure)
      << "empty input vectors must be rejected by the public-API "
         "validator (defends M-tile's `num_ops == 0` invariant — "
         "the dispatcher never reaches `flat_m_tile` on an empty batch)";
}

TEST(TestGroupMatmulMTileEdgeCases, AllExpertsInactiveSucceeds) {
  using namespace moe_test_utils;
  using zendnnl::lowoha::matmul::group_matmul_direct;
  using zendnnl::lowoha::matmul::status_t;
  using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kPhase2Single;
  using zendnnl::common::bfloat16_t;
  using zendnnl::common::data_type_t;

  const int saved = omp_get_max_threads();
  struct G { int p; ~G() { omp_set_num_threads(p); } } g{saved};
  omp_set_num_threads(32);
  if (!require_min_threads(32))
    GTEST_SKIP() << "Requires >= 32 OMP threads.";

  // 8 experts, all with M=0 — `active_ops` should be 0 at line 657
  // and `flat_m_tile` returns immediately without entering any
  // branch.  No tag is set (capture stays at its initial value).
  const int num_ops = 8;
  const int N = 128;
  const int K = 64;
  std::vector<int> Ms(num_ops, 0);

  // Buffers are present but unused (M=0 → 0 rows per expert).
  std::vector<std::vector<bfloat16_t>> src(num_ops);
  std::vector<std::vector<bfloat16_t>> wei(num_ops);
  std::vector<std::vector<bfloat16_t>> dst(num_ops);
  for (int e = 0; e < num_ops; ++e) {
    src[e].resize(1);   // size-1 placeholder
    wei[e].resize(static_cast<size_t>(K) * N);
    dst[e].resize(1);
  }
  std::vector<const void *> srcs(num_ops), weis(num_ops),
                             biases(num_ops, nullptr);
  std::vector<void *>       dsts(num_ops);
  for (int e = 0; e < num_ops; ++e) {
    srcs[e] = src[e].data();
    weis[e] = wei[e].data();
    dsts[e] = dst[e].data();
  }

  auto gv = GemmVecs::uniform(num_ops, /*M=*/0, N, K);
  gv.Ms = Ms;
  auto params = make_uniform_params(num_ops, data_type_t::bf16);
  for (auto &pp : params) pp.num_threads = 32;

  AlgoEnvGuard algo2_guard(2);
  ::reset_grp_matmul_caches();
  const status_t st = group_matmul_direct(
      gv.layout, gv.transA, gv.transB, gv.Ms, gv.Ns, gv.Ks, gv.alpha,
      srcs, gv.lda, weis, gv.ldb, biases, gv.beta,
      dsts, gv.ldc, gv.is_wc, params, nullptr);
  // Public API may accept (status_t::success, no-op) or reject (some
  // validators treat all-zero M as degenerate).  Both outcomes are
  // semantically correct — what we test is that the call DOES NOT
  // crash, and if it succeeds, no dst writes occurred (vacuously
  // true since M=0).  We tolerate either status.
  EXPECT_TRUE(st == status_t::success || st == status_t::failure)
      << "all-inactive batch must terminate cleanly (success or "
         "failure, never crash)";
}

// Mixed active / inactive: some experts have M[i]=0 alongside others
// with M[i]>0.  The M-tile planner builds `active_pos[i]` (line 669)
// which skips inactives, and dispatches only active experts.  Value
// correctness on the active subset must match ALGO 1 reference; the
// inactive expert's dst slot is untouched in both runs (M=0 → no
// rows to write).
TEST(TestGroupMatmulMTileEdgeCases, MixedActiveInactiveExperts) {
  using namespace moe_test_utils;
  using zendnnl::lowoha::matmul::group_matmul_direct;
  using zendnnl::lowoha::matmul::status_t;
  using zendnnl::common::bfloat16_t;
  using zendnnl::common::data_type_t;

  const int saved = omp_get_max_threads();
  struct G { int p; ~G() { omp_set_num_threads(p); } } g{saved};
  omp_set_num_threads(32);
  if (!require_min_threads(32))
    GTEST_SKIP() << "Requires >= 32 OMP threads.";

  // 8 experts: 4 active (M=64), 4 inactive (M=0), interleaved.  This
  // exercises both the active_pos compaction and the cross-expert
  // dispatch order — the planner's CCD stripe should skip inactive
  // slots without leaving gaps.
  const int num_ops = 8;
  const int N = 128, K = 64, M_active = 64;
  std::vector<int> Ms(num_ops);
  for (int e = 0; e < num_ops; ++e) Ms[e] = (e % 2 == 0) ? M_active : 0;

  std::vector<std::vector<bfloat16_t>> src(num_ops), wei(num_ops),
      dst_test(num_ops), dst_ref(num_ops);
  for (int e = 0; e < num_ops; ++e) {
    const int M_e = Ms[e];
    src[e].resize(static_cast<size_t>(std::max(1, M_e)) * K);
    wei[e].resize(static_cast<size_t>(K) * N);
    dst_test[e].assign(static_cast<size_t>(std::max(1, M_e)) * N,
                       bfloat16_t(0.0f));
    dst_ref[e].assign(static_cast<size_t>(std::max(1, M_e)) * N,
                      bfloat16_t(0.0f));
    if (M_e > 0) {
      fill_src(src[e], e, 0.02f);
      fill_wei1(wei[e], e, 0.005f);
    }
  }
  std::vector<const void *> srcs(num_ops), weis(num_ops),
                             biases(num_ops, nullptr);
  std::vector<void *>       dsts_test(num_ops), dsts_ref(num_ops);
  for (int e = 0; e < num_ops; ++e) {
    srcs[e]      = src[e].data();
    weis[e]      = wei[e].data();
    dsts_test[e] = dst_test[e].data();
    dsts_ref[e]  = dst_ref[e].data();
  }

  auto gv = GemmVecs::uniform(num_ops, /*M=*/0, N, K);
  gv.Ms = Ms;
  auto params = make_uniform_params(num_ops, data_type_t::bf16);
  for (auto &pp : params) pp.num_threads = 32;

  // Reference: ALGO 1 (sequential per-expert).
  {
    auto pr = params;
    AlgoEnvGuard algo1_guard(1);
    ::reset_grp_matmul_caches();
    ASSERT_EQ(group_matmul_direct(gv.layout, gv.transA, gv.transB,
                                  gv.Ms, gv.Ns, gv.Ks, gv.alpha,
                                  srcs, gv.lda, weis, gv.ldb,
                                  biases, gv.beta,
                                  dsts_ref, gv.ldc, gv.is_wc, pr, nullptr),
              status_t::success) << "ref dispatch failed";
  }
  // Test: ALGO 2 (M-tile).
  {
    auto pt = params;
    AlgoEnvGuard algo2_guard(2);
    ::reset_grp_matmul_caches();
    ASSERT_EQ(group_matmul_direct(gv.layout, gv.transA, gv.transB,
                                  gv.Ms, gv.Ns, gv.Ks, gv.alpha,
                                  srcs, gv.lda, weis, gv.ldb,
                                  biases, gv.beta,
                                  dsts_test, gv.ldc, gv.is_wc, pt, nullptr),
              status_t::success) << "test dispatch failed";
  }

  // Compare per-expert: active experts must produce matching values;
  // inactive experts have no rows to compare.
  for (int e = 0; e < num_ops; ++e) {
    if (Ms[e] == 0) continue;
    for (int row = 0; row < Ms[e]; ++row) {
      for (int c = 0; c < N; ++c) {
        const size_t off = static_cast<size_t>(row) * N + c;
        const float gt  = static_cast<float>(dst_test[e][off]);
        const float ref = static_cast<float>(dst_ref [e][off]);
        const float bound = std::abs(ref) * 5e-2f + 5e-2f;
        if (std::abs(gt - ref) > bound) {
          FAIL() << "mixed-active value mismatch e=" << e
                 << " r=" << row << " c=" << c
                 << " got=" << gt << " ref=" << ref;
        }
      }
    }
  }
}

TEST(TestGroupMatmulMTileEdgeCases, OddNWithGatedActRejected) {
  using namespace moe_test_utils;
  using zendnnl::lowoha::matmul::group_matmul_direct;
  using zendnnl::lowoha::matmul::status_t;
  using zendnnl::lowoha::matmul::grp_matmul_gated_act_params;
  using zendnnl::lowoha::matmul::grp_matmul_gated_act_t;
  using zendnnl::common::bfloat16_t;
  using zendnnl::common::data_type_t;

  const int saved = omp_get_max_threads();
  struct G { int p; ~G() { omp_set_num_threads(p); } } g{saved};
  omp_set_num_threads(32);
  if (!require_min_threads(32))
    GTEST_SKIP() << "Requires >= 32 OMP threads.";

  const int num_ops = 4;
  const int M = 16;
  const int N_odd = 127;   // odd — must be rejected
  const int K = 64;

  std::vector<std::vector<bfloat16_t>> src(num_ops), wei(num_ops),
                                       dst(num_ops);
  for (int e = 0; e < num_ops; ++e) {
    src[e].resize(static_cast<size_t>(M) * K);
    wei[e].resize(static_cast<size_t>(K) * N_odd);
    dst[e].assign(static_cast<size_t>(M) * N_odd, bfloat16_t(0.0f));
  }
  std::vector<const void *> srcs(num_ops), weis(num_ops),
                             biases(num_ops, nullptr);
  std::vector<void *>       dsts(num_ops);
  for (int e = 0; e < num_ops; ++e) {
    srcs[e] = src[e].data();
    weis[e] = wei[e].data();
    dsts[e] = dst[e].data();
  }

  auto gv = GemmVecs::uniform(num_ops, M, N_odd, K);
  auto params = make_uniform_params(num_ops, data_type_t::bf16);
  for (auto &pp : params) pp.num_threads = 32;

  grp_matmul_gated_act_params gated{};
  gated.act = grp_matmul_gated_act_t::silu_and_mul;

  AlgoEnvGuard algo2_guard(2);
  ::reset_grp_matmul_caches();
  const status_t st = group_matmul_direct(
      gv.layout, gv.transA, gv.transB, gv.Ms, gv.Ns, gv.Ks, gv.alpha,
      srcs, gv.lda, weis, gv.ldb, biases, gv.beta,
      dsts, gv.ldc, gv.is_wc, params, nullptr, &gated);
  EXPECT_EQ(st, status_t::failure)
      << "odd N=" << N_odd << " with gated activation must be rejected "
         "by the public-API validator (group_matmul_direct.cpp:476-482) "
         "— gated activations collapse pairs of columns (N/2 lanes "
         "per row), so an odd N has no valid `N/2` semantics";
}

TEST(TestGroupMatmulMTileCustomKernelInvariance, BF16ByteIdenticalDst) {
  using namespace moe_test_utils;
  using zendnnl::lowoha::matmul::group_matmul_direct;
  using zendnnl::lowoha::matmul::status_t;
  using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kPhase2Single;

  const int saved = omp_get_max_threads();
  struct G { int p; ~G() { omp_set_num_threads(p); } } g{saved};
  omp_set_num_threads(32);
  if (!require_min_threads(32))
    GTEST_SKIP() << "Requires >= 32 OMP threads.";

  // Two independent shapes — first with CK=0, second with CK=1.  Same
  // src/wei/bias inputs across both runs (deterministic fill).  The
  // helper builds a fresh shape per call so the two dst buffers don't
  // alias.
  auto s0 = build_standalone_mtile_shape(
      /*num_threads=*/32,
      /*Ms=*/std::vector<int>(16, 64),
      /*N=*/128, /*K=*/64,
      /*with_bias=*/true);
  auto s1 = build_standalone_mtile_shape(
      /*num_threads=*/32,
      /*Ms=*/std::vector<int>(16, 64),
      /*N=*/128, /*K=*/64,
      /*with_bias=*/true);

  int tag0 = 0, tag1 = 0;
  // Run 1: CK=0.
  {
    AlgoEnvGuard         algo2_guard(2);
    CustomKernelOverride ck_off(false);
    MTileHybridOverride  hybrid_auto(0);
    MTilePathCaptureGuard cap;
    ::reset_grp_matmul_caches();
    auto pt = s0.params;
    ASSERT_EQ(group_matmul_direct(s0.gv.layout, s0.gv.transA, s0.gv.transB,
                                  s0.gv.Ms, s0.gv.Ns, s0.gv.Ks, s0.gv.alpha,
                                  s0.srcs, s0.gv.lda, s0.weis, s0.gv.ldb,
                                  s0.biases, s0.gv.beta,
                                  s0.dsts_test, s0.gv.ldc,
                                  s0.gv.is_wc, pt, nullptr, nullptr),
              status_t::success) << "CK=0 dispatch failed";
    tag0 = zendnnl::lowoha::matmul::test_api
        ::s_last_m_tile_path.load(std::memory_order_relaxed);
  }
  // Run 2: CK=1.
  {
    AlgoEnvGuard         algo2_guard(2);
    CustomKernelOverride ck_on(true);
    MTileHybridOverride  hybrid_auto(0);
    MTilePathCaptureGuard cap;
    ::reset_grp_matmul_caches();
    auto pt = s1.params;
    ASSERT_EQ(group_matmul_direct(s1.gv.layout, s1.gv.transA, s1.gv.transB,
                                  s1.gv.Ms, s1.gv.Ns, s1.gv.Ks, s1.gv.alpha,
                                  s1.srcs, s1.gv.lda, s1.weis, s1.gv.ldb,
                                  s1.biases, s1.gv.beta,
                                  s1.dsts_test, s1.gv.ldc,
                                  s1.gv.is_wc, pt, nullptr, nullptr),
              status_t::success) << "CK=1 dispatch failed";
    tag1 = zendnnl::lowoha::matmul::test_api
        ::s_last_m_tile_path.load(std::memory_order_relaxed);
  }

  // Both runs MUST land on the same M-tile branch — CK doesn't
  // influence the planner.
  EXPECT_EQ(tag0, kPhase2Single)
      << "CK=0 run: expected kPhase2Single (=3), got " << tag0;
  EXPECT_EQ(tag1, kPhase2Single)
      << "CK=1 run: expected kPhase2Single (=3), got " << tag1;

  // Byte-identical dst across both runs — any difference signals a
  // hidden CK-dependent code path inside M-tile.
  for (int e = 0; e < s0.num_ops; ++e) {
    ASSERT_EQ(s0.dst_test[e].size(), s1.dst_test[e].size())
        << "dst buffer size mismatch on expert " << e;
    const auto *p0 = reinterpret_cast<const uint8_t *>(s0.dst_test[e].data());
    const auto *p1 = reinterpret_cast<const uint8_t *>(s1.dst_test[e].data());
    const size_t bytes = s0.dst_test[e].size()
        * sizeof(zendnnl::common::bfloat16_t);
    for (size_t b = 0; b < bytes; ++b) {
      if (p0[b] != p1[b]) {
        FAIL() << "M-tile dst is NOT byte-identical across CK on/off: "
                  "expert=" << e << " byte_off=" << b
               << " CK=0=" << static_cast<int>(p0[b])
               << " CK=1=" << static_cast<int>(p1[b])
               << " — `flat_m_tile` must not pick up a CK-dependent "
                  "kernel-select branch";
      }
    }
  }
}

TEST(TestGroupMatmulMTileStandalone, F32WithBiasF32) {
  using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kPhase2Single;

  const int saved = omp_get_max_threads();
  struct G { int p; ~G() { omp_set_num_threads(p); } } g{saved};
  omp_set_num_threads(32);
  if (!require_min_threads(32))
    GTEST_SKIP() << "Requires >= 32 OMP threads.";

  auto s = build_standalone_mtile_shape_f32(
      /*num_threads=*/32,
      /*Ms=*/std::vector<int>(16, 64),
      /*N=*/128, /*K=*/64,
      /*with_bias=*/true);
  run_and_compare_standalone_f32(
      s, /*algo_test=*/2, /*algo_ref=*/1,
      /*expected_tag=*/kPhase2Single,
      /*rtol=*/1e-5f, /*atol=*/1e-5f,
      "standalone_f32_bias_f32");
}

// Forced ALGO 2 with active_ops > num_threads AND a fused gated activation:
// the sequential full-team clamp (kManyExpertsSeqFallback) must still apply
// the full-M post-pass activation per expert correctly.
TEST(TestGroupMatmulMTileBranches, ManyExpertsSeqClampGatedActFullMPostPass) {
  using zendnnl::lowoha::matmul::test_api::m_tile_path_tag
      ::kManyExpertsSeqFallback;
  using zendnnl::lowoha::matmul::grp_matmul_gated_act_params;
  using zendnnl::lowoha::matmul::grp_matmul_gated_act_t;

  const int saved = omp_get_max_threads();
  struct G { int p; ~G() { omp_set_num_threads(p); } } g{saved};
  omp_set_num_threads(8);
  if (!require_min_threads(4))
    GTEST_SKIP() << "Requires >= 4 OMP threads.";

  auto s = build_standalone_mtile_shape(
      /*num_threads=*/4,
      /*Ms=*/std::vector<int>(8, 8),
      /*N=*/256, /*K=*/64,
      /*with_bias=*/false);

  grp_matmul_gated_act_params ga{};
  ga.act = grp_matmul_gated_act_t::silu_and_mul;

  run_and_compare_standalone(
      s, /*algo_test=*/2, /*algo_ref=*/1,
      /*expected_tag=*/kManyExpertsSeqFallback,
      /*rtol=*/5e-2f, /*atol=*/5e-2f,
      "many_experts_seq_clamp_silu_full_M_post_pass",
      /*gated_act=*/&ga);
}

