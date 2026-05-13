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

#include "gtest_utils.hpp"
#include "group_matmul_test_helpers.hpp"
#include "moe_test_utils.hpp"

// Direct access to the dispatcher's ALGO selector — used by section
// [8b] (`TestGroupMatmulAutoSelectAlgo`) to assert the auto-select
// decision matrix without spinning up the full GEMM stack.
#include "lowoha_operators/matmul/group_matmul/group_matmul_parallel_common.hpp"

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
// [8b] TestGroupMatmulAutoSelectAlgo — pin down `select_grp_matmul_algo`'s
//      decisions on the (max_M, num_threads, weight_class) grid.
//
// Two invariants this section locks in:
//
//   1. Small / medium weight (≤ kMediumWeight = 64 MB / expert) +
//      prompt (max_M > kDecodeMaxM = 32) → ALGO 1, at every thread
//      count we care about {32, 48, 64, 80, 96, 128}.  Covers
//      gpt-oss-20B-class prompt at BS ∈ {8, 16, 32}, seq=128.
//
//   2. Large weight (> 64 MB) + prompt + wide-N / many-experts +
//      N-tile viable → ALGO 3 (Mixtral 8×-class carve-out preserved).
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

// Grid: enumerate the cases the user cares about plus key
// regression-pins.  `make_*` helpers compose the label so a future
// failure points straight at the offending row.
static std::vector<AutoSelectParam> make_auto_select_params() {
  std::vector<AutoSelectParam> out;

  // gpt-oss-20B-class shape (Op1, gate+up).  weight_per_expert =
  // 2880 × 5760 × 2 = ~31.6 MB → medium class.
  const int K_GO = 2880, N_GO = 5760;

  // ── Invariant 1 — small/medium prompt → ALGO 1 across threads ──
  // BS={8,16,32}, seq=128, topk=4 → M_per_expert ≈ {128, 256, 512}.
  // num_threads sweep covers the user's request (32, 64, 128) plus
  // in-between values (48, 80, 96) and the cap above 128.
  for (int M_val : {128, 256, 512}) {
    for (int nt : {32, 48, 64, 80, 96, 128, 192}) {
      out.push_back({
        /*M=*/M_val, /*K=*/K_GO, /*N=*/N_GO,
        /*num_ops=*/32, /*num_threads=*/nt,
        /*expected_algo=*/1,
        "prompt_gptoss_M" + std::to_string(M_val)
          + "_t" + std::to_string(nt)});
    }
  }

  // ── Decode regression-pin — gpt-oss-20B decode keeps ALGO 3 ──
  // max_M ≤ 32 and ≥4 experts and ntile_ok.
  for (int M_val : {4, 16, 32}) {
    for (int nt : {64, 128}) {
      out.push_back({
        M_val, K_GO, N_GO, /*num_ops=*/32, nt,
        /*expected_algo=*/3,
        "decode_gptoss_M" + std::to_string(M_val)
          + "_t" + std::to_string(nt)});
    }
  }

  // ── Invariant 2 — Mixtral large-weight wide-N prompt → ALGO 3 ──
  // Mixtral 8× decode-prompt shape: K=4096, N=14336, BF16 →
  // weight_per_expert = 4096 × 14336 × 2 = ~112 MB → large class.
  // wide-N (N > K), many-experts (16+).  PR carve-out preserved.
  for (int num_ops : {16, 32}) {
    for (int nt : {64, 128}) {
      out.push_back({
        /*M=*/256, /*K=*/4096, /*N=*/14336,
        num_ops, nt, /*expected_algo=*/3,
        "mixtral_wideN_E" + std::to_string(num_ops)
          + "_t" + std::to_string(nt)});
    }
  }

  // ── Invariant 2.b — Mixtral large-weight TALL-N prompt → ALGO 1 ──
  // Mirror shape (K=14336, N=4096): tall-N few-experts → ALGO 1
  // (existing behaviour, sanity-pinning so future heuristic edits
  // don't accidentally flip it to ALGO 3).
  out.push_back({
    /*M=*/256, /*K=*/14336, /*N=*/4096,
    /*num_ops=*/8, /*num_threads=*/128, /*expected_algo=*/1,
    "mixtral_tallN_E8_t128"});

  // ── Edge: many experts > num_threads → ALGO 5 (per-expert) ──
  out.push_back({
    /*M=*/4, /*K=*/2880, /*N=*/5760,
    /*num_ops=*/256, /*num_threads=*/128, /*expected_algo=*/5,
    "many_experts_E256_t128"});

  return out;
}

INSTANTIATE_TEST_SUITE_P(GroupMatmulAutoSelect, TestGroupMatmulAutoSelectAlgo,
                         ::testing::ValuesIn(make_auto_select_params()),
                         AutoSelectParamName);

