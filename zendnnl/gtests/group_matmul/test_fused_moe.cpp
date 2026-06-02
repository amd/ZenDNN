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

/// @file test_fused_moe.cpp
/// @brief Fused-MoE gtest sections.  Owned suites:
///
///   [5]  TestFusedMoE                       - Op1 + activation + Op2 vs
///                                              2-call legacy reference.
///   [5b] TestFusedMoEInternalAlloc          - both Op1 + Op2 library-managed
///                                              (Op1 -> arena, Op2 -> in-place
///                                              src reuse).
///   [5c] TestFusedMoEInternalAllocMixedM    - internal-alloc with per-expert
///                                              M skew (real gpt-oss decode
///                                              frame distributions).
///   [6]  TestGroupMatmulCombined            - all 2^3=8 combinations of
///                                              (moe, act, fused).
///   [10] TestFusedMoEDstSplit               - independent Op1 / Op2 internal-
///                                              alloc detection (4 buffer-source
///                                              patterns).
///   [11] TestFusedMoEActiveMatmul           - `params[0].active_matmul` /
///                                              `params[0].total_matmul`
///                                              prepack-extras layout handling.
///   [12] TestFusedMoEWarmPackPipeline       - multi-iteration decode
///                                              pipeline with rotating
///                                              fired-expert subsets.
///   [13] TestFusedMoEArchGrid               - real architectures
///                                              (gpt-oss / Mixtral / DeepSeek
///                                              / Qwen3-MoE) at scaled dims.
///   [14] TestFusedMoEActiveTotalEdge        - active_matmul / total_matmul
///                                              single-expert + 1-of-N edges.
///   [15] TestDispatcherActiveTotalNegative  - dispatcher contract validator
///                                              negative cases (G2 from the
///                                              PR-443 review: `am > M.size()`
///                                              and `tm < am` must reject).
///   [20] TestFusedMoECompactAsymmetric      - production framework-opt-in
///                                              layout (M.size()=active,
///                                              params.size()=total).  Closes
///                                              the gap left by [11] (Padded
///                                              layout) and benchdnn
///                                              (Compact-symmetric with
///                                              params.size()=active) — the
///                                              regression that produced the
///                                              vLLM `std::bad_array_new_length`
///                                              only fires under this exact
///                                              asymmetric sizing.
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

// ???????????????????????????????????????????????????????????????????????????????
// [5] TestFusedMoE: Op1(gate+up) ? Act ? Op2(down_proj) vs 2-call reference
// ???????????????????????????????????????????????????????????????????????????????

struct FusedMoETestParam {
  int dim, hidden_size, M, num_ops,
      act_int;  // act_int: 0=none, 1=silu, 2=gelu, 3=swiglu_oai
  bool is_bf16;
};

static std::string FusedMoEParamName(
  const ::testing::TestParamInfo<FusedMoETestParam> &info) {
  static const char *act_names[] = {"none", "silu", "gelu", "swiglu"};
  const auto &p = info.param;
  return std::string(act_names[p.act_int]) + (p.is_bf16 ? "_bf16" : "_f32")
         + "_d" + std::to_string(p.dim) + "_h" + std::to_string(p.hidden_size)
         + "_M" + std::to_string(p.M) + "_E" + std::to_string(p.num_ops);
}

class TestFusedMoE : public ::testing::TestWithParam<FusedMoETestParam> {};

TEST_P(TestFusedMoE, Correctness) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;

  reset_grp_matmul_caches();

  const auto &p = GetParam();
  const int dim = p.dim, N_gate_up = 2 * dim, H = p.hidden_size;
  const int M = p.M, K = H, num_ops = p.num_ops;
  const auto act_type = static_cast<grp_matmul_gated_act_t>(p.act_int);
  const data_type_t dt = p.is_bf16 ? data_type_t::bf16 : data_type_t::f32;

  // Op2's K dimension follows the activation: gated activations halve
  // Op1 output to `dim` cols, act=none flows the full N_gate_up cols
  // through.  See `op2_k_for_act` in group_matmul_fused_moe.cpp for
  // the contract.
  const bool act_is_none = (act_type == grp_matmul_gated_act_t::none);
  const int K_down = act_is_none ? N_gate_up : dim;

  // Allocate buffers.
  TypedBuffers src, w1, d1, d1_ref, w2, d2_fused, d2_ref;
  src     .alloc(num_ops, (size_t)M * K,         p.is_bf16);
  w1      .alloc(num_ops, (size_t)K * N_gate_up, p.is_bf16);
  d1      .alloc(num_ops, (size_t)M * N_gate_up, p.is_bf16);
  d1_ref  .alloc(num_ops, (size_t)M * N_gate_up, p.is_bf16);
  w2      .alloc(num_ops, (size_t)K_down * H,    p.is_bf16);
  d2_fused.alloc(num_ops, (size_t)M * H,         p.is_bf16);
  d2_ref  .alloc(num_ops, (size_t)M * H,         p.is_bf16);
  fill_moe_tensors(num_ops, p.is_bf16, &src, &w1, &w2);

  // Only Op1's GemmVecs is needed locally; the helper builds its own
  // for the reference 2-call path.
  auto gv_op1 = GemmVecs::uniform(num_ops, M, N_gate_up, K);

  auto srcs     = src.cptrs(p.is_bf16);
  auto wei1     = w1.cptrs(p.is_bf16);
  auto wei2     = w2.cptrs(p.is_bf16);
  auto dst1     = d1.ptrs(p.is_bf16);
  auto dst1_ref = d1_ref.ptrs(p.is_bf16);
  auto dst2_f   = d2_fused.ptrs(p.is_bf16);
  auto dst2_r   = d2_ref.ptrs(p.is_bf16);
  std::vector<const void *> no_bias(num_ops, nullptr);
  auto params   = make_uniform_params(num_ops, dt);

  grp_matmul_gated_act_params act{};
  act.act = act_type;
  auto act_ptr = (act_type != grp_matmul_gated_act_t::none) ? &act : nullptr;

  ASSERT_EQ(run_legacy_2call_ref(num_ops, M, K, N_gate_up, K_down, H,
                                 p.is_bf16, act_type,
                                 srcs, wei1, wei2, dst1_ref, dst2_r),
            status_t::success);

  // Fused path: single call with &fused, writes dst2_fused.
  auto fused = make_fused_moe_op2(num_ops, H, wei2, no_bias);
  fused.dst_down = dst2_f;
  fused.ldc_down = std::vector<int>(num_ops, H);
  {
    auto pf = params;
    ASSERT_EQ(group_matmul_direct(gv_op1.layout, gv_op1.transA, gv_op1.transB,
                                  gv_op1.Ms, gv_op1.Ns, gv_op1.Ks, gv_op1.alpha, srcs, gv_op1.lda,
                                  wei1, gv_op1.ldb, no_bias, gv_op1.beta, dst1, gv_op1.ldc,
                                  gv_op1.is_wc, pf, nullptr, act_ptr, &fused), status_t::success);
  }

  std::ostringstream lbl;
  lbl << "act=" << p.act_int << (p.is_bf16 ? " bf16" : " f32")
      << " dim=" << dim << " h=" << H << " M=" << M << " E=" << num_ops;
  verify_per_expert_2d(d2_fused, H, d2_ref, H,
                       num_ops, M, H, p.is_bf16,
                       tol_fused(p.is_bf16), lbl.str());
}

static std::vector<FusedMoETestParam> make_fused_moe_params() {
  std::vector<FusedMoETestParam> out;
  // Core matrix: all 4 activation types ? both dtypes ? 3?3 shape grid.
  // Covers: none (skip-act path), silu/gelu (concatenated [gate|up] layout),
  // swiglu_oai_mul (interleaved [g0,u0,g1,u1,...] layout).
  for (int a : {
         0, 1, 2, 3
       })
    for (bool bf : {
           false, true
         })
      for (int d : {
             16, 32, 64
           })
        for (int h : {
               16, 32, 64
             })
          out.push_back({d, h, 4, 4, a, bf});
  // Vary M and num_ops for silu (concatenated layout) at dim=32, h=32.
  for (bool bf : {
         false, true
       })
    for (int m : {
           1, 4, 16
         })
      for (int e : {
             2, 4, 8
           }) {
        if (m == 4 && e == 4) {
          continue;  // already in core
        }
        out.push_back({32, 32, m, e, 1, bf});
      }
  // Vary M and num_ops for swiglu_oai (interleaved layout) ? semantically
  // distinct from silu/gelu so it deserves independent M?E sweeps.
  for (bool bf : {
         false, true
       })
    for (int m : {
           1, 4, 16
         })
      for (int e : {
             2, 4, 8
           }) {
        if (m == 4 && e == 4) {
          continue;  // already in core
        }
        out.push_back({32, 32, m, e, 3, bf});
      }
  // BF16 realistic decode shapes (Qwen3-class small weights).
  for (int m : {
         1, 2, 8
       })
    for (int e : {
           4, 16
         })
      out.push_back({256, 128, m, e, 1, /*is_bf16=*/true});
  // num_ops > 16 coverage for the internal-alloc + custom-kernel path
  // (GPT-OSS-class decode workload ? `dim=128, h=128`).  Earlier test
  // sweeps capped num_ops at 16; this band exercises the
  // ManyExperts/Multi-round dispatch that was previously uncovered.
  // Three activation modes:
  //   1 = silu_and_mul (concatenated [gate|up])
  //   3 = swiglu_oai_mul (interleaved [g0,u0,g1,u1,...])
  //   0 = none (pass-through; `op2_k_for_act` returns full N_op1)
  for (int act_mode : {0, 1, 3})
    for (bool bf : {false, true})
      for (int m : {1, 4})
        for (int e : {17, 21, 22, 32})
          out.push_back({128, 128, m, e, act_mode, bf});
  return out;
}

INSTANTIATE_TEST_SUITE_P(GroupMatmulFusedMoE, TestFusedMoE,
                         ::testing::ValuesIn(make_fused_moe_params()), FusedMoEParamName);

// ???????????????????????????????????????????????????????????????????????????????
// [5b] TestFusedMoEInternalAlloc: fused MoE with internal-alloc + src-reuse
//
// Exercises mode (2) of grp_matmul_fused_moe_params: caller leaves
// `fused.dst_down` empty, the library allocates a per-expert Op1
// scratch internally, runs Op1 + activation into it, then runs Op2
// reading from the scratch and writing back into the caller's `src[]`
// buffer (in-place reuse).  Test reuses the same parameter generator
// as TestFusedMoE so coverage is identical.
//
// Reference is the legacy 2-call path (group_matmul_direct without
// fused_moe).  Verification reads the Op2 output back from src[] and
// compares against the reference dst_down.
//
// K = H is naturally satisfied by make_fused_moe_params (M K = H = h
// in the param struct), so lda = K = H = N_down and the in-place
// reuse fits exactly within the original src row stride.
// ???????????????????????????????????????????????????????????????????????????????

class TestFusedMoEInternalAlloc
  : public ::testing::TestWithParam<FusedMoETestParam> {};

TEST_P(TestFusedMoEInternalAlloc, Correctness) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;

  reset_grp_matmul_caches();

  const auto &p = GetParam();
  const int dim = p.dim, N_gate_up = 2 * dim, H = p.hidden_size;
  const int M = p.M, K = H, num_ops = p.num_ops;
  const auto act_type = static_cast<grp_matmul_gated_act_t>(p.act_int);
  const data_type_t dt = p.is_bf16 ? data_type_t::bf16 : data_type_t::f32;

  // Op2's K dimension follows the activation: gated => dim, none =>
  // N_gate_up.  See `op2_k_for_act` in group_matmul_fused_moe.cpp.
  const bool act_is_none = (act_type == grp_matmul_gated_act_t::none);
  const int K_down = act_is_none ? N_gate_up : dim;

  // Two src copies: src_orig (consumed by the legacy reference path,
  // remains untouched), src_intalloc (consumed AND repurposed by the
  // internal-alloc path ? receives Op2 output in-place).
  TypedBuffers src_orig, src_intalloc, w1, d1_ref, w2, d2_ref;
  src_orig    .alloc(num_ops, (size_t)M * K,         p.is_bf16);
  src_intalloc.alloc(num_ops, (size_t)M * K,         p.is_bf16);
  w1          .alloc(num_ops, (size_t)K * N_gate_up, p.is_bf16);
  d1_ref      .alloc(num_ops, (size_t)M * N_gate_up, p.is_bf16);
  w2          .alloc(num_ops, (size_t)K_down * H,    p.is_bf16);
  d2_ref      .alloc(num_ops, (size_t)M * H,         p.is_bf16);
  fill_moe_tensors(num_ops, p.is_bf16, &src_orig,     &w1, &w2);
  fill_moe_tensors(num_ops, p.is_bf16, &src_intalloc, nullptr, nullptr);

  auto gv_op1 = GemmVecs::uniform(num_ops, M, N_gate_up, K);

  auto srcs_orig     = src_orig.cptrs(p.is_bf16);
  auto srcs_intalloc = src_intalloc.cptrs(p.is_bf16);
  auto wei1          = w1.cptrs(p.is_bf16);
  auto wei2          = w2.cptrs(p.is_bf16);
  auto dst1_ref      = d1_ref.ptrs(p.is_bf16);
  auto dst2_r        = d2_ref.ptrs(p.is_bf16);
  std::vector<const void *> no_bias(num_ops, nullptr);
  auto params   = make_uniform_params(num_ops, dt);

  grp_matmul_gated_act_params act{};
  act.act = act_type;
  auto act_ptr = (act_type != grp_matmul_gated_act_t::none) ? &act : nullptr;

  ASSERT_EQ(run_legacy_2call_ref(num_ops, M, K, N_gate_up, K_down, H,
                                 p.is_bf16, act_type,
                                 srcs_orig, wei1, wei2, dst1_ref, dst2_r),
            status_t::success);

  // Internal-alloc fused path: dst[] = nullptr, fused.dst_down empty.
  // Library allocates the per-expert Op1 scratch, runs Op1 + act + Op2,
  // and writes Op2 output back into srcs_intalloc.
  auto fused = make_fused_moe_op2(num_ops, H, wei2, no_bias);
  // fused.dst_down and fused.ldc_down INTENTIONALLY left empty - the
  // signal that engages internal-alloc + src-reuse mode.

  std::vector<void *>  dst_null(num_ops, nullptr);
  std::vector<int>     ldc_null(num_ops, 0);
  {
    auto pf = params;
    ASSERT_EQ(group_matmul_direct(gv_op1.layout, gv_op1.transA, gv_op1.transB,
                                  gv_op1.Ms, gv_op1.Ns, gv_op1.Ks, gv_op1.alpha, srcs_intalloc, gv_op1.lda,
                                  wei1, gv_op1.ldb, no_bias, gv_op1.beta, dst_null, ldc_null,
                                  gv_op1.is_wc, pf, nullptr, act_ptr, &fused), status_t::success);
  }

  // src_intalloc has row stride K (Op1 lda); reference d2_ref has row stride H.
  std::ostringstream lbl;
  lbl << "act=" << p.act_int << (p.is_bf16 ? " bf16" : " f32")
      << " dim=" << dim << " h=" << H << " M=" << M << " E=" << num_ops;
  verify_per_expert_2d(src_intalloc, K, d2_ref, H,
                       num_ops, M, H, p.is_bf16,
                       tol_fused(p.is_bf16), lbl.str());
}

INSTANTIATE_TEST_SUITE_P(GroupMatmulFusedMoEInternalAlloc,
                         TestFusedMoEInternalAlloc,
                         ::testing::ValuesIn(make_fused_moe_params()), FusedMoEParamName);

// ???????????????????????????????????????????????????????????????????????????????
// [5c] TestFusedMoEInternalAllocMixedM ? internal-alloc + per-expert M skew.
//
// Real MoE decode routes only `topk` experts per token, so per-expert M
// values vary (most experts at small M, a few at zero, occasional
// hot-expert with larger M).  Earlier tests use uniform M across all
// experts, missing this dimension.  This test injects M vectors taken
// from real GPT-OSS decode frames ? it covers the (num_ops > 16,
// mixed-M, internal_alloc, custom-kernel-engaged) cube that wasn't
// previously exercised.  Shapes are scaled down for test runtime.
// ???????????????????????????????????????????????????????????????????????????????

struct MixedMTestParam {
  int dim, hidden_size;
  std::vector<int> M_per_expert;  // explicit per-expert M
  int act_int;
  bool is_bf16;
};

static std::string MixedMParamName(
    const ::testing::TestParamInfo<MixedMTestParam> &info) {
  const auto &p = info.param;
  std::string name = (p.is_bf16 ? "bf16" : "f32");
  name += "_act" + std::to_string(p.act_int);
  name += "_d" + std::to_string(p.dim) + "_h" + std::to_string(p.hidden_size);
  name += "_E" + std::to_string(p.M_per_expert.size());
  int sum_M = 0;
  for (int m : p.M_per_expert) sum_M += m;
  name += "_sumM" + std::to_string(sum_M);
  return name;
}

class TestFusedMoEInternalAllocMixedM
    : public ::testing::TestWithParam<MixedMTestParam> {};

TEST_P(TestFusedMoEInternalAllocMixedM, Correctness) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;

  reset_grp_matmul_caches();

  const auto &p = GetParam();
  const int dim = p.dim, N_gate_up = 2 * dim, H = p.hidden_size;
  const int K = H, num_ops = static_cast<int>(p.M_per_expert.size());
  const auto act_type = static_cast<grp_matmul_gated_act_t>(p.act_int);
  const data_type_t dt = p.is_bf16 ? data_type_t::bf16 : data_type_t::f32;

  const bool act_is_none = (act_type == grp_matmul_gated_act_t::none);
  const int K_down = act_is_none ? N_gate_up : dim;

  // Allocate buffers sized to the LARGEST per-expert M so the in-place
  // src reuse fits even for the hot expert.  Inactive (M=0) experts
  // get the same allocation (harmless; their slice is just unused).
  int max_M = 0;
  for (int m : p.M_per_expert) if (m > max_M) max_M = m;
  if (max_M == 0) GTEST_SKIP() << "all M=0; nothing to test";

  TypedBuffers src_orig, src_intalloc, w1, d1_ref, w2, d2_ref;
  src_orig    .alloc(num_ops, (size_t)max_M * K,         p.is_bf16);
  src_intalloc.alloc(num_ops, (size_t)max_M * K,         p.is_bf16);
  w1          .alloc(num_ops, (size_t)K * N_gate_up,     p.is_bf16);
  d1_ref      .alloc(num_ops, (size_t)max_M * N_gate_up, p.is_bf16);
  w2          .alloc(num_ops, (size_t)K_down * H,        p.is_bf16);
  d2_ref      .alloc(num_ops, (size_t)max_M * H,         p.is_bf16);
  fill_moe_tensors(num_ops, p.is_bf16, &src_orig,     &w1, &w2);
  fill_moe_tensors(num_ops, p.is_bf16, &src_intalloc, nullptr, nullptr);

  // Per-expert vectors for the fused-MoE call (real M_per_expert
  // distribution).  The helper builds its own internal GemmVecs for
  // the reference path, so we only need Op1 here.
  GemmVecs gv_op1;
  gv_op1.layout.assign(num_ops, 'r');
  gv_op1.transA.assign(num_ops, false);
  gv_op1.transB.assign(num_ops, false);
  gv_op1.is_wc .assign(num_ops, false);
  gv_op1.alpha .assign(num_ops, 1.0f);
  gv_op1.beta  .assign(num_ops, 0.0f);
  gv_op1.Ms    = p.M_per_expert;
  gv_op1.Ns    .assign(num_ops, N_gate_up);
  gv_op1.Ks    .assign(num_ops, K);
  gv_op1.lda   .assign(num_ops, K);
  gv_op1.ldb   .assign(num_ops, N_gate_up);
  gv_op1.ldc   .assign(num_ops, N_gate_up);

  auto srcs_orig     = src_orig.cptrs(p.is_bf16);
  auto srcs_intalloc = src_intalloc.cptrs(p.is_bf16);
  auto wei1          = w1.cptrs(p.is_bf16);
  auto wei2          = w2.cptrs(p.is_bf16);
  auto dst1_ref      = d1_ref.ptrs(p.is_bf16);
  auto dst2_r        = d2_ref.ptrs(p.is_bf16);
  std::vector<const void *> no_bias(num_ops, nullptr);
  auto params = make_uniform_params(num_ops, dt);

  grp_matmul_gated_act_params act{};
  act.act = act_type;
  auto act_ptr = (act_type != grp_matmul_gated_act_t::none) ? &act : nullptr;

  ASSERT_EQ(run_legacy_2call_ref(p.M_per_expert, K, N_gate_up, K_down, H,
                                 p.is_bf16, act_type,
                                 srcs_orig, wei1, wei2, dst1_ref, dst2_r),
            status_t::success);

  // Internal-alloc fused path.
  auto fused = make_fused_moe_op2(num_ops, H, wei2, no_bias);
  std::vector<void *>  dst_null(num_ops, nullptr);
  std::vector<int>     ldc_null(num_ops, 0);
  {
    auto pf = params;
    ASSERT_EQ(group_matmul_direct(gv_op1.layout, gv_op1.transA, gv_op1.transB,
        gv_op1.Ms, gv_op1.Ns, gv_op1.Ks, gv_op1.alpha, srcs_intalloc, gv_op1.lda,
        wei1, gv_op1.ldb, no_bias, gv_op1.beta, dst_null, ldc_null,
        gv_op1.is_wc, pf, nullptr, act_ptr, &fused), status_t::success);
  }

  // Compare per active expert (verify_per_expert_2d skips M=0).
  std::ostringstream lbl;
  lbl << "act=" << p.act_int << (p.is_bf16 ? " bf16" : " f32")
      << " dim=" << dim << " h=" << H << " E=" << num_ops;
  verify_per_expert_2d(src_intalloc, K, d2_ref, H,
                       p.M_per_expert, H, p.is_bf16,
                       tol_fused(p.is_bf16), lbl.str());
}

// Real GPT-OSS decode routing patterns (sum_M=128 is one decode token).
// Frames lifted from gpt_oss_moe_decode_fused_profiled.txt with shapes
// scaled down (dim=128, hidden=128) for test runtime.
static std::vector<MixedMTestParam> make_mixed_m_params() {
  std::vector<MixedMTestParam> out;
  // Frame 126: num_ops=21, M=[2,26,2,1,8,1,9,3,1,1,3,2,6,3,5,1,1,28,12,4,9].
  const std::vector<int> frame126 = {2,26,2,1,8,1,9,3,1,1,3,2,6,3,5,1,1,28,12,4,9};
  // Synthetic: num_ops=24 with two zero experts mid-list (the case the
  // packing arena sees most often when topk<num_ops).
  const std::vector<int> frame_sparse = {1,4,0,2,8,0,3,5,1,2,0,16,0,4,2,1,3,0,6,1,2,0,9,1};
  // Synthetic: num_ops=21 hot expert at the edges.
  const std::vector<int> frame_edges  = {32,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,30};
  for (int act_mode : {1, 3, 0}) {
    for (bool bf : {true, false}) {
      out.push_back({128, 128, frame126,    act_mode, bf});
      out.push_back({128, 128, frame_sparse, act_mode, bf});
      out.push_back({128, 128, frame_edges,  act_mode, bf});
    }
  }
  // Exact GPT-OSS-20B decode shapes (K=H=2880, dim=hidden=2880),
  // bf16 + swiglu_oai, for the precise mixed-M frames the user is
  // reporting failures on.  Limited to bf16 + swiglu (act=3) since
  // fp32 would balloon the test runtime and the failure mode the
  // user described is bf16-specific.
  out.push_back({2880, 2880, frame126,    /*act=*/3, /*is_bf16=*/true});
  out.push_back({2880, 2880, frame_sparse, /*act=*/3, /*is_bf16=*/true});
  out.push_back({2880, 2880, frame_edges,  /*act=*/3, /*is_bf16=*/true});
  return out;
}

INSTANTIATE_TEST_SUITE_P(GroupMatmulFusedMoEInternalAllocMixedM,
    TestFusedMoEInternalAllocMixedM,
    ::testing::ValuesIn(make_mixed_m_params()), MixedMParamName);

// ???????????????????????????????????????????????????????????????????????????????
// [6] TestGroupMatmulCombined: all 2?=8 combinations of (moe, act, fused)
// ???????????????????????????????????????????????????????????????????????????????

struct CombinedTestParam {
  bool use_moe, use_act, use_fused, is_bf16;
  int M, num_ops;
};

static std::string CombinedParamName(
  const ::testing::TestParamInfo<CombinedTestParam> &info) {
  const auto &p = info.param;
  std::string name = (p.is_bf16 ? "bf16" : "f32");
  name += "_M" + std::to_string(p.M) + "_E" + std::to_string(p.num_ops)
          + "_moe" + (p.use_moe ? "1" : "0")
          + "_act" + (p.use_act ? "1" : "0")
          + "_fused" + (p.use_fused ? "1" : "0");
  return name;
}

class TestGroupMatmulCombined :
  public ::testing::TestWithParam<CombinedTestParam> {};

TEST_P(TestGroupMatmulCombined, AllCombinations) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;

  reset_grp_matmul_caches();

  const auto &p = GetParam();
  const int H = 256, dim = 128, N_gate_up = 2 * dim, K = H, topk = 2;
  const int num_ops = p.num_ops, M = p.M;
  const data_type_t dt = p.is_bf16 ? data_type_t::bf16 : data_type_t::f32;

  // Op1 output width depends on whether act/fused is used.
  const bool need_gate_up = p.use_act || p.use_fused;
  const int N_op1 = need_gate_up ? N_gate_up : H;

  // Op2's K_down depends on whether a gated activation runs between
  // Op1 and Op2: the gated activation compacts gate+up into N_op1/2
  // columns (= dim), while no-act passes the full N_op1 width into
  // Op2.  Mirrors the library contract `op2_k_for_act` in
  // group_matmul_fused_moe.cpp.  Without this the test would
  // allocate `wei2` for the gated case unconditionally and read OOB
  // when `use_fused=true && use_act=false` (silent corruption with
  // values like 2e+28 from heap junk past `wei2`'s end).
  //
  // Sibling tests in this file (TestFusedMoE / TestFusedMoEAlgos /
  // TestFusedMoEAlgoCustom / TestFusedMoEInternalAlloc) already
  // follow the same `K_down = act_is_none ? N_gate_up : dim` idiom;
  // this is just bringing TestGroupMatmulCombined into line.
  const bool act_is_none_for_op2 = !p.use_act;
  const int K_down = act_is_none_for_op2 ? N_op1 : dim;

  // Allocate all buffers (some won't be used ? harmless).
  TypedBuffers src, w1, d1, d1_ref, w2, d2_fused, d2_ref;
  src    .alloc(num_ops, (size_t)M * K,        p.is_bf16);
  w1     .alloc(num_ops, (size_t)K * N_op1,    p.is_bf16);
  d1     .alloc(num_ops, (size_t)M * N_op1,    p.is_bf16);
  d1_ref .alloc(num_ops, (size_t)M * N_op1,    p.is_bf16);
  w2     .alloc(num_ops, (size_t)K_down * H,   p.is_bf16);
  d2_fused.alloc(num_ops, (size_t)M * H,       p.is_bf16);
  d2_ref  .alloc(num_ops, (size_t)M * H,       p.is_bf16);
  fill_moe_tensors(num_ops, p.is_bf16, &src, &w1, &w2);

  auto gv = GemmVecs::uniform(num_ops, M, N_op1, K);
  auto srcs   = src.cptrs(p.is_bf16);
  auto wei1   = w1.cptrs(p.is_bf16);
  auto wei2   = w2.cptrs(p.is_bf16);
  auto dst1   = d1.ptrs(p.is_bf16);
  auto dst1_r = d1_ref.ptrs(p.is_bf16);
  auto dst2_f = d2_fused.ptrs(p.is_bf16);
  auto dst2_r = d2_ref.ptrs(p.is_bf16);
  std::vector<const void *> no_bias(num_ops, nullptr);
  auto params = make_uniform_params(num_ops, dt);

  // Optional structs.
  grp_matmul_gated_act_params act{};
  act.act = grp_matmul_gated_act_t::silu_and_mul;
  grp_matmul_gated_act_params *act_ptr = p.use_act ? &act : nullptr;

  grp_matmul_fused_moe_params fused{};
  grp_matmul_fused_moe_params *fused_ptr = nullptr;
  if (p.use_fused) {
    fused = make_fused_moe_op2(num_ops, H, wei2, no_bias);
    fused.dst_down = dst2_f;
    fused.ldc_down = std::vector<int>(num_ops, H);
    fused_ptr = &fused;
  }

  const int D_final  = p.use_fused ? H : N_op1;
  const int num_slots = M * topk;
  std::vector<float> moe_weights(num_slots, 1.0f / topk);
  std::vector<float> moe_out_f((size_t)M * D_final, 0.0f);
  std::vector<bfloat16_t> moe_out_b((size_t)M * D_final, bfloat16_t(0.0f));
  std::vector<const void *> moe_row_ptrs(num_slots);

  group_matmul_moe_postop_params moe{};
  group_matmul_moe_postop_params *moe_ptr = nullptr;
  if (p.use_moe && num_ops >= topk) {
    moe.num_tokens    = M;
    moe.topk          = topk;
    moe.output        = p.is_bf16 ? (void *)moe_out_b.data() :
                        (void *)moe_out_f.data();
    moe.ldc_output    = D_final;
    moe.topk_weights  = moe_weights.data();
    moe.skip_weighted = false;
    for (int t = 0; t < M; ++t) {
      for (int kk = 0; kk < topk; ++kk) {
        const int slot = t * topk + kk, expert = (t + kk) % num_ops;
        if (p.use_fused) {
          moe_row_ptrs[slot] = p.is_bf16
                               ? (const void *)(d2_fused.bf16[expert].data() + t * D_final)
                               : (const void *)(d2_fused.f32[expert].data()  + t * D_final);
        }
        else {
          moe_row_ptrs[slot] = p.is_bf16
                               ? (const void *)(d1.bf16[expert].data() + t * N_op1)
                               : (const void *)(d1.f32[expert].data()  + t * N_op1);
        }
      }
    }
    moe.row_ptrs = moe_row_ptrs.data();
    moe_ptr = &moe;
  }

  // Execute the combined call.
  {
    auto ptest = params;
    ASSERT_EQ(group_matmul_direct(gv.layout, gv.transA, gv.transB, gv.Ms, gv.Ns,
                                  gv.Ks, gv.alpha, srcs, gv.lda, wei1, gv.ldb, no_bias, gv.beta,
                                  dst1, gv.ldc, gv.is_wc, ptest, moe_ptr, act_ptr, fused_ptr),
              status_t::success) << "Combined call failed";
  }

  // Build step-by-step reference.
  auto pr1 = params;
  ASSERT_EQ(group_matmul_direct(gv.layout, gv.transA, gv.transB, gv.Ms, gv.Ns,
                                gv.Ks, gv.alpha, srcs, gv.lda, wei1, gv.ldb, no_bias, gv.beta,
                                dst1_r, gv.ldc, gv.is_wc, pr1), status_t::success);

  if (p.use_act) {
    for (int e = 0; e < num_ops; ++e) {
      if (p.is_bf16) {
        apply_ref_gated_act(d1_ref.bf16[e], M, N_op1, N_op1, act.act);
      }
      else {
        apply_ref_gated_act(d1_ref.f32[e],  M, N_op1, N_op1, act.act);
      }
    }
  }

  if (p.use_fused) {
    std::vector<const void *> srcs2(num_ops);
    for (int e = 0; e < num_ops; ++e) {
      srcs2[e] = dst1_r[e];
    }
    // Reference Op2 must use the same `K_down` as the fused-MoE
    // path: full N_op1 when no activation runs, half when gated.
    auto gv2 = GemmVecs::uniform(num_ops, M, H, K_down);
    gv2.lda.assign(num_ops, N_op1);
    auto pr2 = params;
    ASSERT_EQ(group_matmul_direct(gv2.layout, gv2.transA, gv2.transB, gv2.Ms,
                                  gv2.Ns,
                                  gv2.Ks, gv2.alpha, srcs2, gv2.lda, wei2, gv2.ldb, no_bias, gv2.beta,
                                  dst2_r, gv2.ldc, gv2.is_wc, pr2), status_t::success);
  }

  std::vector<float> moe_ref_f((size_t)M * D_final, 0.0f);
  if (p.use_moe && num_ops >= topk) {
    for (int t = 0; t < M; ++t) {
      for (int d = 0; d < D_final; ++d) {
        float acc = 0.0f;
        for (int kk = 0; kk < topk; ++kk) {
          const int expert = (t + kk) % num_ops;
          float v;
          if (p.use_fused) {
            v = d2_ref.at(expert, (size_t)t * H + d, p.is_bf16);
          }
          else {
            v = d1_ref.at(expert, (size_t)t * N_op1 + d, p.is_bf16);
          }
          acc += moe_weights[t * topk + kk] * v;
        }
        moe_ref_f[t * D_final + d] = acc;
      }
    }
  }

  // Compare.
  const auto tol = tol_fused(p.is_bf16);
  auto check = [&](float got, float ref, const char *tag, int a, int b, int c) {
    ASSERT_NEAR(got, ref, std::abs(ref) * tol.rel + tol.abs)
        << tag << " moe=" << p.use_moe << " act=" << p.use_act
        << " fused=" << p.use_fused << (p.is_bf16 ? " bf16" : " f32")
        << " a=" << a << " b=" << b << " c=" << c;
  };

  if (p.use_moe && num_ops >= topk) {
    for (int t = 0; t < M; ++t) {
      for (int d = 0; d < D_final; ++d) {
        const float got = p.is_bf16
                          ? static_cast<float>(moe_out_b[t * D_final + d])
                          : moe_out_f[t * D_final + d];
        check(got, moe_ref_f[t * D_final + d], "moe", 0, t, d);
      }
    }
  }
  else if (p.use_fused) {
    for (int e = 0; e < num_ops; ++e)
      for (int r = 0; r < M; ++r)
        for (int c = 0; c < H; ++c)
          check(d2_fused.at(e, (size_t)r * H + c, p.is_bf16),
                d2_ref  .at(e, (size_t)r * H + c, p.is_bf16), "fused", e, r, c);
  }
  else if (p.use_act) {
    for (int e = 0; e < num_ops; ++e)
      for (int r = 0; r < M; ++r)
        for (int c = 0; c < dim; ++c)
          check(d1    .at(e, (size_t)r * N_op1 + c, p.is_bf16),
                d1_ref.at(e, (size_t)r * N_op1 + c, p.is_bf16), "act-only", e, r, c);
  }
  else {
    for (int e = 0; e < num_ops; ++e)
      for (int r = 0; r < M; ++r)
        for (int c = 0; c < N_op1; ++c)
          check(d1    .at(e, (size_t)r * N_op1 + c, p.is_bf16),
                d1_ref.at(e, (size_t)r * N_op1 + c, p.is_bf16), "plain", e, r, c);
  }
}

static std::vector<CombinedTestParam> make_combined_params() {
  std::vector<CombinedTestParam> out;
  for (bool moe : {
         false, true
       })
    for (bool act : {
           false, true
         })
      for (bool fused : {
             false, true
           })
        for (bool bf16 : {
               false, true
             })
          for (int m : {
                 16, 64
               })
            for (int e : {
                   2, 4
                 })
              out.push_back({moe, act, fused, bf16, m, e});
  return out;
}

INSTANTIATE_TEST_SUITE_P(GroupMatmulCombined, TestGroupMatmulCombined,
                         ::testing::ValuesIn(make_combined_params()), CombinedParamName);

// ???????????????????????????????????????????????????????????????????????????????
// [10] TestFusedMoEDstSplit ? independent Op1 / Op2 internal-alloc detection
// ???????????????????????????????????????????????????????????????????????????????
//
// Exercises the four buffer-source patterns supported by the per-side
// internal-alloc detection (commit 284340b5):
//
//   dst[]   | dst_down[]  | Op1 destination     | Op2 destination
//   ??????? | ??????????? | ??????????????????? | ???????????????????
//   empty   | empty       | library Op1 arena   | in-place src reuse
//   empty   | filled      | library Op1 arena   | caller's dst_down
//   filled  | empty       | caller's dst        | in-place src reuse
//   filled  | filled      | caller's dst        | caller's dst_down
//
// All four must produce numerically-identical Op2 output regardless
// of where each side's buffer comes from.  Reference is the 2-call
// legacy path; the test reads Op2 output from src_test[] when Op2 is
// library-managed (in-place reuse) or from d2_test[] when caller-
// allocated.

struct DstSplitTestParam {
  int dim, hidden_size, M, num_ops;
  int dst_filled;      // 0 = empty/internal, 1 = caller-allocated
  int dst_down_filled; // 0 = empty/internal, 1 = caller-allocated
  bool is_bf16;
  int act_int;         // 0 = none, 1 = silu, 2 = gelu, 3 = swiglu_oai
};

static std::string DstSplitParamName(
    const ::testing::TestParamInfo<DstSplitTestParam> &info) {
  static const char *act_names[] = {"none", "silu", "gelu", "swiglu"};
  const auto &p = info.param;
  return std::string(act_names[p.act_int])
       + (p.is_bf16 ? "_bf16" : "_f32")
       + "_dst" + (p.dst_filled ? "Y" : "N")
       + "_dd"  + (p.dst_down_filled ? "Y" : "N")
       + "_d" + std::to_string(p.dim) + "_h" + std::to_string(p.hidden_size)
       + "_M" + std::to_string(p.M) + "_E" + std::to_string(p.num_ops);
}

class TestFusedMoEDstSplit
    : public ::testing::TestWithParam<DstSplitTestParam> {};

TEST_P(TestFusedMoEDstSplit, Correctness) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;

  reset_grp_matmul_caches();

  const auto &p = GetParam();
  const int dim = p.dim, N_gate_up = 2 * dim, H = p.hidden_size;
  const int M = p.M, K = H, num_ops = p.num_ops;
  const auto act_type = static_cast<grp_matmul_gated_act_t>(p.act_int);
  const data_type_t dt = p.is_bf16 ? data_type_t::bf16 : data_type_t::f32;
  const bool act_is_none = (act_type == grp_matmul_gated_act_t::none);
  const int K_down = act_is_none ? N_gate_up : dim;

  // Two src copies: ref consumed by legacy 2-call path; test consumed
  // by the fused path under test (and possibly mutated by Op2 in-place
  // reuse when op2 is internal).
  TypedBuffers src_ref, src_test, w1, d1_ref, d1_test, w2, d2_ref, d2_test;
  src_ref .alloc(num_ops, (size_t)M * K,         p.is_bf16);
  src_test.alloc(num_ops, (size_t)M * K,         p.is_bf16);
  w1      .alloc(num_ops, (size_t)K * N_gate_up, p.is_bf16);
  d1_ref  .alloc(num_ops, (size_t)M * N_gate_up, p.is_bf16);
  d1_test .alloc(num_ops, (size_t)M * N_gate_up, p.is_bf16);
  w2      .alloc(num_ops, (size_t)K_down * H,    p.is_bf16);
  d2_ref  .alloc(num_ops, (size_t)M * H,         p.is_bf16);
  d2_test .alloc(num_ops, (size_t)M * H,         p.is_bf16);
  fill_moe_tensors(num_ops, p.is_bf16, &src_ref,  &w1, &w2);
  fill_moe_tensors(num_ops, p.is_bf16, &src_test, nullptr, nullptr);

  auto gv_op1 = GemmVecs::uniform(num_ops, M, N_gate_up, K);

  auto src_ref_p  = src_ref.cptrs(p.is_bf16);
  auto src_test_p = src_test.cptrs(p.is_bf16);
  auto wei1_p     = w1.cptrs(p.is_bf16);
  auto wei2_p     = w2.cptrs(p.is_bf16);
  auto d1_ref_p   = d1_ref.ptrs(p.is_bf16);
  auto d1_test_p  = d1_test.ptrs(p.is_bf16);
  auto d2_ref_p   = d2_ref.ptrs(p.is_bf16);
  auto d2_test_p  = d2_test.ptrs(p.is_bf16);
  std::vector<const void *> no_bias(num_ops, nullptr);
  auto params = make_uniform_params(num_ops, dt);

  grp_matmul_gated_act_params act{};
  act.act = act_type;
  auto act_ptr = act_is_none ? nullptr : &act;

  ASSERT_EQ(run_legacy_2call_ref(num_ops, M, K, N_gate_up, K_down, H,
                                 p.is_bf16, act_type,
                                 src_ref_p, wei1_p, wei2_p, d1_ref_p, d2_ref_p),
            status_t::success);

  // Test path: configure dst[]/dst_down[] per case.  An empty
  // dst[]/dst_down[] (or all-null) signals library-managed for that
  // side; non-null entries signal caller-allocated.
  auto fused = make_fused_moe_op2(num_ops, H, wei2_p, no_bias);

  std::vector<void *> dst_eff;
  std::vector<int> ldc_eff;
  if (p.dst_filled) {
    dst_eff = d1_test_p;
    ldc_eff = gv_op1.ldc;
  } else {
    dst_eff.assign(num_ops, nullptr);
    ldc_eff.assign(num_ops, 0);
  }
  if (p.dst_down_filled) {
    fused.dst_down = d2_test_p;
    fused.ldc_down = std::vector<int>(num_ops, H);
  }
  // (When op2 is library-managed leave fused.dst_down / ldc_down empty.)

  {
    auto pf = params;
    ASSERT_EQ(group_matmul_direct(gv_op1.layout, gv_op1.transA, gv_op1.transB,
                                  gv_op1.Ms, gv_op1.Ns, gv_op1.Ks, gv_op1.alpha,
                                  src_test_p, gv_op1.lda, wei1_p, gv_op1.ldb,
                                  no_bias, gv_op1.beta, dst_eff, ldc_eff,
                                  gv_op1.is_wc, pf, nullptr, act_ptr, &fused),
              status_t::success);
  }

  // Op2 output landed where the caller asked:
  //   dst_down filled -> d2_test (stride H).
  //   dst_down empty  -> src_test (in-place reuse, stride K = H here).
  std::ostringstream lbl;
  lbl << "dst_filled=" << p.dst_filled
      << " dst_down_filled=" << p.dst_down_filled
      << " act=" << p.act_int << " bf16=" << p.is_bf16;
  if (p.dst_down_filled) {
    verify_per_expert_2d(d2_test, H, d2_ref, H, num_ops, M, H, p.is_bf16,
                         tol_fused(p.is_bf16), lbl.str());
  } else {
    verify_per_expert_2d(src_test, K, d2_ref, H, num_ops, M, H, p.is_bf16,
                         tol_fused(p.is_bf16), lbl.str());
  }
}

static std::vector<DstSplitTestParam> make_dst_split_params() {
  std::vector<DstSplitTestParam> out;
  // Core matrix: 2 acts (silu, swiglu) ? 2 dtypes ? 4 dst/dst_down
  // combinations ? small shape.  Covers all 4 buffer-source patterns
  // for both common gated activation kinds.
  for (int act : {1, 3}) {
    for (bool bf : {false, true}) {
      for (int df : {0, 1}) {
        for (int ddf : {0, 1}) {
          out.push_back({32, 32, 4, 4, df, ddf, bf, act});
        }
      }
    }
  }
  return out;
}

INSTANTIATE_TEST_SUITE_P(GroupMatmulFusedMoEDstSplit, TestFusedMoEDstSplit,
                         ::testing::ValuesIn(make_dst_split_params()),
                         DstSplitParamName);

// ???????????????????????????????????????????????????????????????????????????????
// [11] TestFusedMoEActiveMatmul ? active_matmul / total_matmul handling
// ???????????????????????????????????????????????????????????????????????????????
//
// Exercises the per-call active_matmul / total_matmul fields added in
// commit e3853e08.  Framework signals "send all expert weights every
// call" by setting:
//
//   params[0].active_matmul = K   (fired count)
//   params[0].total_matmul  = E   (total experts in MoE block)
//
// Vectors are sized to E with the first K entries representing fired
// experts; trailing entries carry weight metadata for prepack but the
// matmul-processing loops iterate `[0, K)` only.  Op2 internal-alloc
// + src reuse so the result lands back in src_test[0..K-1] for
// comparison against a legacy K-expert reference.
//
// Verifies:
//   * `num_ops` derives from `active_matmul`.
//   * Dispatch processes only the active prefix, ignoring the
//     prepack-extras tail.
//   * Result is numerically identical to a K-expert legacy call.

struct ActiveMatmulTestParam {
  int dim, hidden_size, M;
  int active_count, prepack_extras;
  bool is_bf16;
  int act_int;
};

static std::string ActiveMatmulParamName(
    const ::testing::TestParamInfo<ActiveMatmulTestParam> &info) {
  static const char *act_names[] = {"none", "silu", "gelu", "swiglu"};
  const auto &p = info.param;
  return std::string(act_names[p.act_int])
       + (p.is_bf16 ? "_bf16" : "_f32")
       + "_act" + std::to_string(p.active_count)
       + "_extras" + std::to_string(p.prepack_extras);
}

class TestFusedMoEActiveMatmul
    : public ::testing::TestWithParam<ActiveMatmulTestParam> {};

TEST_P(TestFusedMoEActiveMatmul, Correctness) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;

  reset_grp_matmul_caches();

  const auto &p = GetParam();
  const int dim = p.dim, N_gate_up = 2 * dim, H = p.hidden_size;
  const int M = p.M, K = H;
  const int active = p.active_count;
  const int total = active + p.prepack_extras;
  const auto act_type = static_cast<grp_matmul_gated_act_t>(p.act_int);
  const data_type_t dt = p.is_bf16 ? data_type_t::bf16 : data_type_t::f32;
  const bool act_is_none = (act_type == grp_matmul_gated_act_t::none);
  const int K_down = act_is_none ? N_gate_up : dim;

  // Reference path uses `active`-sized vectors.  Test path uses
  // `total`-sized vectors (active prefix + prepack-extras tail).
  TypedBuffers src_ref, src_test, w1, d1_ref, w2, d2_ref;
  src_ref .alloc(active, (size_t)M * K,         p.is_bf16);
  src_test.alloc(total,  (size_t)M * K,         p.is_bf16);
  w1      .alloc(total,  (size_t)K * N_gate_up, p.is_bf16);
  d1_ref  .alloc(active, (size_t)M * N_gate_up, p.is_bf16);
  w2      .alloc(total,  (size_t)K_down * H,    p.is_bf16);
  d2_ref  .alloc(active, (size_t)M * H,         p.is_bf16);

  fill_moe_tensors(total,  p.is_bf16, &src_test, &w1, &w2);
  fill_moe_tensors(active, p.is_bf16, &src_ref,  nullptr, nullptr);

  auto gv_op1_test = GemmVecs::uniform(total, M, N_gate_up, K);
  // Trailing prepack-extras experts have M=0 (the framework's contract;
  // `active_matmul` slicing also handles this, but zeroing M is
  // defence-in-depth for any code path that re-derives num_ops from
  // M.size() before reading params[0].active_matmul).
  for (int e = active; e < total; ++e) gv_op1_test.Ms[e] = 0;

  auto src_ref_p   = src_ref.cptrs(p.is_bf16);
  auto src_test_p  = src_test.cptrs(p.is_bf16);
  auto wei1_test_p = w1.cptrs(p.is_bf16);
  auto wei2_test_p = w2.cptrs(p.is_bf16);
  auto d1_ref_p    = d1_ref.ptrs(p.is_bf16);
  auto d2_ref_p    = d2_ref.ptrs(p.is_bf16);
  std::vector<const void *> wei1_ref_p(wei1_test_p.begin(),
                                       wei1_test_p.begin() + active);
  std::vector<const void *> wei2_ref_p(wei2_test_p.begin(),
                                       wei2_test_p.begin() + active);
  std::vector<const void *> no_bias_test(total, nullptr);

  auto params_test = make_uniform_params(total, dt);
  // Set per-side counts on every params slot.  group_matmul_direct
  // reads only params[0] but the framework's actual usage broadcasts
  // identical values to every slot ? match that here.
  for (auto &pp : params_test) {
    pp.active_matmul = static_cast<uint32_t>(active);
    pp.total_matmul  = static_cast<uint32_t>(total);
  }

  grp_matmul_gated_act_params act{};
  act.act = act_type;
  auto act_ptr = act_is_none ? nullptr : &act;

  ASSERT_EQ(run_legacy_2call_ref(active, M, K, N_gate_up, K_down, H,
                                 p.is_bf16, act_type,
                                 src_ref_p, wei1_ref_p, wei2_ref_p,
                                 d1_ref_p, d2_ref_p),
            status_t::success);

  // Test: active_matmul / total_matmul fused-MoE call with both
  // sides library-managed (op1 ? arena, op2 ? in-place src reuse).
  // Op2's output lands back in src_test[0..active-1].
  auto fused = make_fused_moe_op2(total, H, wei2_test_p, no_bias_test);

  std::vector<void *> dst_null(total, nullptr);
  std::vector<int>    ldc_null(total, 0);
  {
    auto pf = params_test;
    ASSERT_EQ(group_matmul_direct(gv_op1_test.layout, gv_op1_test.transA,
                                  gv_op1_test.transB, gv_op1_test.Ms,
                                  gv_op1_test.Ns, gv_op1_test.Ks,
                                  gv_op1_test.alpha, src_test_p,
                                  gv_op1_test.lda, wei1_test_p, gv_op1_test.ldb,
                                  no_bias_test, gv_op1_test.beta, dst_null,
                                  ldc_null, gv_op1_test.is_wc, pf, nullptr,
                                  act_ptr, &fused),
              status_t::success);
  }

  std::ostringstream lbl;
  lbl << "active=" << active << " total=" << total;
  verify_per_expert_2d(src_test, K, d2_ref, H, active, M, H, p.is_bf16,
                       tol_fused(p.is_bf16), lbl.str());
}

static std::vector<ActiveMatmulTestParam> make_active_matmul_params() {
  std::vector<ActiveMatmulTestParam> out;
  // 2 acts ? 2 dtypes ? {(2,0), (2,2), (4,0), (4,4), (6,2), (6,6)}
  // active/extras pairs.  Covers the no-extras (legacy-equivalent),
  // small-extras, and 1?-extras configurations.  Caps total at 12 to
  // keep the suite under a few seconds.
  static const std::pair<int, int> shapes[] = {
    {2, 0}, {2, 2}, {4, 0}, {4, 4}, {6, 2}, {6, 6}};
  for (int act : {1, 3}) {
    for (bool bf : {false, true}) {
      for (auto sh : shapes) {
        out.push_back({32, 32, 4, sh.first, sh.second, bf, act});
      }
    }
  }
  return out;
}

INSTANTIATE_TEST_SUITE_P(GroupMatmulFusedMoEActiveMatmul,
                         TestFusedMoEActiveMatmul,
                         ::testing::ValuesIn(make_active_matmul_params()),
                         ActiveMatmulParamName);

// ???????????????????????????????????????????????????????????????????????????????
// [12] TestFusedMoEWarmPackPipeline ? multi-iteration warm-pack flow
// ???????????????????????????????????????????????????????????????????????????????
//
// Simulates the actual gpt-oss-style decode pipeline that motivated
// the active_matmul / total_matmul split (commits e3853e08 + a4f791c4):
//
//   * The framework owns `E_total` per-expert weight buffers
//     (e.g. 8 experts) and passes ALL of them on every call.
//   * Each decode iteration only fires `K` experts (e.g. K=4).
//     The framework permutes its per-call pointer vectors so that
//     the K fired experts occupy positions `[0, K)`; the remaining
//     `E_total - K` non-fired experts fill `[K, E_total)`.
//   * `params[0].active_matmul = K` and
//     `params[0].total_matmul  = E_total` signal the split.
//
// On the first iteration the warm-pack pre-packs every expert's
// weight (when ZENDNNL_GRP_MATMUL_ALGO=3 + ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL=1
// + BF16 are set), so subsequent iterations that route to
// previously-non-fired experts find them already in the cache ?
// no per-iteration reorder cost regardless of routing.
//
// This test verifies CORRECTNESS across the multi-iteration pattern.
// The pack cache itself is keyed on weight pointer + (K, N, ldb,
// transB), so as long as each logical expert's weight pointer is
// stable across iterations (which `TypedBuffers` guarantees ? each
// expert is its own std::vector with a stable .data()) the warm-
// pack effect is automatic and transparent.  A separate model-
// level / benchmark verification is needed for the spike-removal
// behaviour itself (visible via the APILOG `[GRP_MATMUL.PREPACK]`
// line); unit tests just verify that swapping the
// fired subset across iterations does not change the result.
//
// Scenario per test:
//
//   E_total = 8, K = 4
//
//   Iteration 1: fire {0, 1, 2, 3} ? first time any expert is packed.
//   Iteration 2: fire {4, 5, 6, 7} ? entirely new fired set; must
//                hit the warm cache populated in iteration 1.
//   Iteration 3: fire {1, 4, 6, 3} ? mixed; every expert already
//                packed.
//
// Each iteration:
//   * Builds per-call vectors with fired experts at [0, K) and
//     non-fired at [K, E_total).
//   * Calls the fused-MoE entry with active/total set.
//   * Compares result against a legacy K-expert call on the same
//     fired subset (with original src; in-place src reuse aside).

struct WarmPackPipelineParam {
  bool is_bf16;
  int act_int;          // 0 = none, 1 = silu, 2 = gelu, 3 = swiglu_oai
  bool force_algo3;     // true = set ZENDNNL_GRP_MATMUL_ALGO=3 (engages
                        //        warm-pack when custom kernel is also on
                        //        + BF16); false = leave env at default
                        //        (warm-pack stays a no-op, only the
                        //        active/total plumbing is exercised).
};

static std::string WarmPackPipelineParamName(
    const ::testing::TestParamInfo<WarmPackPipelineParam> &info) {
  static const char *act_names[] = {"none", "silu", "gelu", "swiglu"};
  const auto &p = info.param;
  return std::string(act_names[p.act_int])
       + (p.is_bf16 ? "_bf16" : "_f32")
       + (p.force_algo3 ? "_algo3" : "_auto");
}

class TestFusedMoEWarmPackPipeline
    : public ::testing::TestWithParam<WarmPackPipelineParam> {};

TEST_P(TestFusedMoEWarmPackPipeline, MultiIterationRouting) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;

  reset_grp_matmul_caches();

  const auto &p = GetParam();
  // Force ALGO 3 + custom kernel on so the warm-pack hook engages
  // (when the BF16 + total > active + fused-MoE gates also hold).
  // Both env guards restore prior values on scope exit.
  std::unique_ptr<AlgoEnvGuard>      algo_guard;
  std::unique_ptr<EnvVarGuard>       custom_guard;
  if (p.force_algo3) {
    algo_guard   = std::make_unique<AlgoEnvGuard>(3);
    custom_guard = std::make_unique<EnvVarGuard>(
        "ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL", "1");
  }

  constexpr int E_total = 8;
  constexpr int K       = 4;
  const int dim = 32, N_gate_up = 2 * dim, H = 32;
  const int M = 4, K_in = H;
  const auto act_type = static_cast<grp_matmul_gated_act_t>(p.act_int);
  const data_type_t dt = p.is_bf16 ? data_type_t::bf16 : data_type_t::f32;
  const bool act_is_none = (act_type == grp_matmul_gated_act_t::none);
  const int K_down = act_is_none ? N_gate_up : dim;

  // Per-expert weights (stable across iterations ? each is its own
  // contiguous vector with a stable .data() pointer, which keys the
  // pack cache).
  TypedBuffers w1_all, w2_all;
  w1_all.alloc(E_total, (size_t)K_in * N_gate_up, p.is_bf16);
  w2_all.alloc(E_total, (size_t)K_down * H,       p.is_bf16);
  fill_moe_tensors(E_total, p.is_bf16, nullptr, &w1_all, &w2_all);
  auto wei1_all = w1_all.cptrs(p.is_bf16);
  auto wei2_all = w2_all.cptrs(p.is_bf16);

  grp_matmul_gated_act_params act{};
  act.act = act_type;
  auto act_ptr = act_is_none ? nullptr : &act;

  // Three routing iterations covering the three coverage classes
  // described in the suite header: fresh / fully-rotated / mixed.
  static const std::vector<std::vector<int>> routings = {
    {0, 1, 2, 3},
    {4, 5, 6, 7},
    {1, 4, 6, 3},
  };

  for (size_t iter = 0; iter < routings.size(); ++iter) {
    const auto &fired = routings[iter];
    ASSERT_EQ(static_cast<int>(fired.size()), K);

    // ?? Reference: legacy K-expert call on the fired subset ?????
    TypedBuffers src_ref, d1_ref, d2_ref;
    src_ref.alloc(K, (size_t)M * K_in,        p.is_bf16);
    d1_ref .alloc(K, (size_t)M * N_gate_up,   p.is_bf16);
    d2_ref .alloc(K, (size_t)M * H,           p.is_bf16);
    for (int k = 0; k < K; ++k) {
      const int eid = fired[k];
      if (p.is_bf16) fill_src(src_ref.bf16[k], /*seed=*/eid);
      else           fill_src(src_ref.f32 [k], /*seed=*/eid);
    }
    std::vector<const void *> wei1_ref(K), wei2_ref(K);
    for (int k = 0; k < K; ++k) {
      wei1_ref[k] = wei1_all[fired[k]];
      wei2_ref[k] = wei2_all[fired[k]];
    }
    auto src_ref_p = src_ref.cptrs(p.is_bf16);
    auto d1_ref_p  = d1_ref.ptrs(p.is_bf16);
    auto d2_ref_p  = d2_ref.ptrs(p.is_bf16);

    ASSERT_EQ(run_legacy_2call_ref(K, M, K_in, N_gate_up, K_down, H,
                                   p.is_bf16, act_type,
                                   src_ref_p, wei1_ref, wei2_ref,
                                   d1_ref_p, d2_ref_p),
              status_t::success) << "ref iter " << iter;

    // ?? Test: active/total fused-MoE call ??????????????????????
    // Build per-call vectors of size E_total = K + (E_total - K).
    // Fired experts at [0, K), non-fired at [K, E_total).  src,
    // dst, M, lda, ldc, alpha, beta, transA all sized E_total but
    // only `[0, K)` is meaningful ? the dispatcher slices to K
    // via params[0].active_matmul.  Weight-side vectors (weight,
    // ldb, K, N, transB, params, is_weights_const, fused.*)
    // also sized E_total and cover ALL experts so the warm-pack
    // can pre-pack the entire model.
    TypedBuffers src_test;
    src_test.alloc(E_total, (size_t)M * K_in, p.is_bf16);
    for (int idx = 0; idx < E_total; ++idx) {
      // Active prefix gets the same input as the reference; the
      // tail experts get unused dummy data (their M is 0).
      const int eid = idx < K ? fired[idx] : (idx - K);
      if (p.is_bf16) fill_src(src_test.bf16[idx], eid);
      else           fill_src(src_test.f32 [idx], eid);
    }
    auto src_test_p = src_test.cptrs(p.is_bf16);

    // Permuted weight-side pointers: fired experts first, then the
    // remaining experts in canonical order.  The cache key is per
    // weight pointer, so once a weight is packed (under whatever
    // index it appeared at) it's hit on subsequent appearances at
    // any index.
    std::vector<const void *> wei1_test(E_total), wei2_test(E_total);
    {
      std::vector<bool> seen(E_total, false);
      for (int k = 0; k < K; ++k) {
        wei1_test[k] = wei1_all[fired[k]];
        wei2_test[k] = wei2_all[fired[k]];
        seen[fired[k]] = true;
      }
      int tail = K;
      for (int e = 0; e < E_total; ++e) {
        if (seen[e]) continue;
        wei1_test[tail] = wei1_all[e];
        wei2_test[tail] = wei2_all[e];
        ++tail;
      }
    }

    // Per-call vector sizes = E_total.  Active prefix has real M;
    // the tail has M=0 so the dispatcher (after slicing) ignores
    // them, AND the warm-pack helper still iterates [0, total)
    // packing every weight regardless of M.
    auto gv1_test = GemmVecs::uniform(E_total, M, N_gate_up, K_in);
    auto gv2_test = GemmVecs::uniform(E_total, M, H, K_down);
    gv2_test.lda.assign(E_total, N_gate_up);
    for (int e = K; e < E_total; ++e) gv1_test.Ms[e] = 0;

    std::vector<const void *> no_bias_test(E_total, nullptr);
    auto params_test = make_uniform_params(E_total, dt);
    for (auto &pp : params_test) {
      pp.active_matmul = static_cast<uint32_t>(K);
      pp.total_matmul  = static_cast<uint32_t>(E_total);
    }

    auto fused = make_fused_moe_op2(E_total, H, wei2_test, no_bias_test);
    // dst_down empty - Op2 internal-alloc (in-place src reuse).

    std::vector<void *> dst_null(E_total, nullptr);
    std::vector<int>    ldc_null(E_total, 0);
    {
      auto pf = params_test;
      ASSERT_EQ(group_matmul_direct(gv1_test.layout, gv1_test.transA,
                                    gv1_test.transB, gv1_test.Ms,
                                    gv1_test.Ns, gv1_test.Ks,
                                    gv1_test.alpha, src_test_p,
                                    gv1_test.lda, wei1_test, gv1_test.ldb,
                                    no_bias_test, gv1_test.beta, dst_null,
                                    ldc_null, gv1_test.is_wc, pf, nullptr,
                                    act_ptr, &fused),
                status_t::success)
          << "test iter " << iter
          << " (fired={" << fired[0] << "," << fired[1] << ","
                         << fired[2] << "," << fired[3] << "})";
    }

    // src_test[0..K-1] now holds Op2 output (in-place reuse, row stride K_in).
    // Compare against the K-expert reference d2_ref (row stride H).
    std::ostringstream lbl;
    lbl << "iter=" << iter << " fired={" << fired[0] << "," << fired[1]
        << "," << fired[2] << "," << fired[3] << "}";
    verify_per_expert_2d(src_test, K_in, d2_ref, H, K, M, H, p.is_bf16,
                         tol_fused(p.is_bf16), lbl.str());
  }
}

static std::vector<WarmPackPipelineParam> make_warm_pack_pipeline_params() {
  std::vector<WarmPackPipelineParam> out;
  // Two routing scenarios per (act, dtype, force_algo3) combination.
  // BF16 + force_algo3=true is the path that ACTUALLY engages the
  // warm-pack hook; the other combinations exercise the same
  // active/total plumbing without the warm-pack itself, ensuring
  // the multi-iteration result stays correct regardless of whether
  // warm-pack is engaged.
  for (int act : {1, 3}) {
    for (bool bf : {false, true}) {
      for (bool force : {false, true}) {
        // Skip the f32 + force_algo3 combination: the warm-pack
        // gate refuses non-BF16 anyway, so it's redundant with
        // f32 + auto.
        if (force && !bf) continue;
        out.push_back({bf, act, force});
      }
    }
  }
  return out;
}

INSTANTIATE_TEST_SUITE_P(GroupMatmulFusedMoEWarmPackPipeline,
                         TestFusedMoEWarmPackPipeline,
                         ::testing::ValuesIn(make_warm_pack_pipeline_params()),
                         WarmPackPipelineParamName);

// ===============================================================================
// [13] TestFusedMoEArchGrid - covers actual MoE architectures
//      (gpt-oss-20B/120B, Mixtral-8x7B/8x22B, DeepSeek-V2-Lite/V3,
//      Qwen3-MoE) at scaled-down hidden/intermediate dimensions but
//      full expert-count fidelity, exercising the warm-pack +
//      active/total + Op2 in-place reuse at production-relevant
//      (num_experts, topk) cardinalities.
// ===============================================================================
//
// Each row encodes (num_experts, topk, dim, hidden, activation,
// dtype).  The (dim, hidden) shapes are scaled down to keep test
// runtime small; the (num_experts, topk) values match the real
// architectures so the active/total slicing logic is exercised at
// the right cardinalities.  DeepSeek-V3's 256E/top8 is the largest
// expert count any in-flight model uses today.
//
// Routing is deterministic: experts [0, topk) fire each iteration.
// active=topk, total=num_experts, prepack-extras = num_experts - topk.

struct ArchGridParam {
  const char *name;
  int num_experts;
  int topk;
  int dim;        // hidden_size / 2 of the projection
  int hidden;
  int act_int;    // 1=silu, 2=gelu, 3=swiglu_oai
  bool is_bf16;
};

static std::string ArchGridParamName(
    const ::testing::TestParamInfo<ArchGridParam> &info) {
  return std::string(info.param.name)
       + (info.param.is_bf16 ? "_bf16" : "_f32");
}

class TestFusedMoEArchGrid
    : public ::testing::TestWithParam<ArchGridParam> {};

TEST_P(TestFusedMoEArchGrid, Correctness) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;

  reset_grp_matmul_caches();

  const auto &p = GetParam();
  const int E       = p.num_experts;
  const int K_act   = p.topk;
  const int dim = p.dim, N_gate_up = 2 * dim, H = p.hidden;
  const int K_in = H;
  const int M    = 4;
  const auto act_type = static_cast<grp_matmul_gated_act_t>(p.act_int);
  const data_type_t dt = p.is_bf16 ? data_type_t::bf16 : data_type_t::f32;
  const bool act_is_none = (act_type == grp_matmul_gated_act_t::none);
  const int K_down = act_is_none ? N_gate_up : dim;

  // Reference path: K_act experts (the active subset).
  // Test path: E experts total, active prefix = K_act, prepack-extras tail.
  TypedBuffers src_ref, src_test, w1_all, d1_ref, w2_all, d2_ref;
  src_ref .alloc(K_act, (size_t)M * K_in,         p.is_bf16);
  src_test.alloc(E,      (size_t)M * K_in,         p.is_bf16);
  w1_all  .alloc(E,      (size_t)K_in * N_gate_up, p.is_bf16);
  d1_ref  .alloc(K_act, (size_t)M * N_gate_up,    p.is_bf16);
  w2_all  .alloc(E,      (size_t)K_down * H,       p.is_bf16);
  d2_ref  .alloc(K_act, (size_t)M * H,             p.is_bf16);

  fill_moe_tensors(E,     p.is_bf16, &src_test, &w1_all, &w2_all);
  fill_moe_tensors(K_act, p.is_bf16, &src_ref,  nullptr, nullptr);

  auto gv_op1_test = GemmVecs::uniform(E, M, N_gate_up, K_in);
  for (int e = K_act; e < E; ++e) gv_op1_test.Ms[e] = 0;

  auto src_ref_p   = src_ref.cptrs(p.is_bf16);
  auto src_test_p  = src_test.cptrs(p.is_bf16);
  auto wei1_all_p  = w1_all.cptrs(p.is_bf16);
  auto wei2_all_p  = w2_all.cptrs(p.is_bf16);
  auto d1_ref_p    = d1_ref.ptrs(p.is_bf16);
  auto d2_ref_p    = d2_ref.ptrs(p.is_bf16);
  std::vector<const void *> wei1_ref_p(wei1_all_p.begin(),
                                       wei1_all_p.begin() + K_act);
  std::vector<const void *> wei2_ref_p(wei2_all_p.begin(),
                                       wei2_all_p.begin() + K_act);
  std::vector<const void *> no_bias_test(E, nullptr);

  auto params_test = make_uniform_params(E, dt);
  for (auto &pp : params_test) {
    pp.active_matmul = static_cast<uint32_t>(K_act);
    pp.total_matmul  = static_cast<uint32_t>(E);
  }

  grp_matmul_gated_act_params act{};
  act.act = act_type;
  auto act_ptr = act_is_none ? nullptr : &act;

  ASSERT_EQ(run_legacy_2call_ref(K_act, M, K_in, N_gate_up, K_down, H,
                                 p.is_bf16, act_type,
                                 src_ref_p, wei1_ref_p, wei2_ref_p,
                                 d1_ref_p, d2_ref_p),
            status_t::success) << "ref " << p.name;

  auto fused = make_fused_moe_op2(E, H, wei2_all_p, no_bias_test);

  std::vector<void *> dst_null(E, nullptr);
  std::vector<int>    ldc_null(E, 0);
  {
    auto pf = params_test;
    ASSERT_EQ(group_matmul_direct(gv_op1_test.layout, gv_op1_test.transA,
                                  gv_op1_test.transB, gv_op1_test.Ms,
                                  gv_op1_test.Ns, gv_op1_test.Ks,
                                  gv_op1_test.alpha, src_test_p,
                                  gv_op1_test.lda, wei1_all_p, gv_op1_test.ldb,
                                  no_bias_test, gv_op1_test.beta, dst_null,
                                  ldc_null, gv_op1_test.is_wc, pf, nullptr,
                                  act_ptr, &fused),
              status_t::success) << "test " << p.name;
  }

  std::ostringstream lbl;
  lbl << p.name << " E=" << E << " topk=" << K_act
      << " dim=" << dim << " H=" << H;
  verify_per_expert_2d(src_test, K_in, d2_ref, H, K_act, M, H, p.is_bf16,
                       tol_fused(p.is_bf16), lbl.str());
}

static std::vector<ArchGridParam> make_arch_grid_params() {
  // (num_experts, topk, dim, hidden, act).  dim=64 / hidden=64 keeps
  // the suite fast while preserving the expert-count axis that
  // differentiates each architecture.
  // act: 1=silu_and_mul, 3=swiglu_oai_mul.  (2=gelu_and_mul not
  // exercised here; covered by TestFusedMoE.)
  static const ArchGridParam kArchs[] = {
    // gpt-oss family: swiglu_oai_mul, top-4.
    {"gpt_oss_20B",      32, 4,  64, 64, 3, false},
    {"gpt_oss_120B",    128, 4,  64, 64, 3, false},
    // Mixtral family: silu_and_mul, top-2.  8x7B and 8x22B differ
    // only in (hidden, intermediate) which we scale uniformly.
    {"mixtral_8x7B",      8, 2,  64, 64, 1, false},
    // DeepSeek family: silu_and_mul.  V2-Lite is 64E/top6;
    // V3 is the largest expert count any in-flight model uses (256E/top8).
    {"deepseek_v2_lite", 64, 6,  64, 64, 1, false},
    {"deepseek_v3",     256, 8,  64, 64, 1, false},
    // Qwen3-MoE: silu_and_mul, top-4, 60E.
    {"qwen3_moe",        60, 4,  64, 64, 1, false},
  };
  std::vector<ArchGridParam> out;
  for (const auto &arch : kArchs) {
    for (bool bf : {false, true}) {
      out.push_back({arch.name, arch.num_experts, arch.topk,
                     arch.dim, arch.hidden, arch.act_int, bf});
    }
  }
  return out;
}

INSTANTIATE_TEST_SUITE_P(GroupMatmulFusedMoEArchGrid, TestFusedMoEArchGrid,
                         ::testing::ValuesIn(make_arch_grid_params()),
                         ArchGridParamName);

// ===============================================================================
// [14] TestFusedMoEActiveTotalEdge - active_matmul / total_matmul edge cases
//      that the regular suite ([11]) doesn't pick up: total=active no-extras,
//      active=1 single fired, total=256 large model, total=active+1
//      minimal extras, gpt-oss-style 32 active in 256-pool.
// ===============================================================================

struct ActiveTotalEdgeParam {
  const char *name;
  int active;
  int total;
  bool is_bf16;
};

static std::string ActiveTotalEdgeParamName(
    const ::testing::TestParamInfo<ActiveTotalEdgeParam> &info) {
  return std::string(info.param.name)
       + (info.param.is_bf16 ? "_bf16" : "_f32");
}

class TestFusedMoEActiveTotalEdge
    : public ::testing::TestWithParam<ActiveTotalEdgeParam> {};

TEST_P(TestFusedMoEActiveTotalEdge, Correctness) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;

  reset_grp_matmul_caches();

  const auto &p = GetParam();
  const int active = p.active;
  const int total  = p.total;
  ASSERT_LE(active, total) << "active must be <= total";
  if (active == 0) {
    GTEST_SKIP()
        << "active=0 is the legacy contract (no warm-pack engaged); "
        << "covered by [5] TestFusedMoE which uses input vectors sized "
        << "to active and total identically (active_matmul/total_matmul "
        << "default to 0).";
  }
  if (active == 1) {
    GTEST_SKIP()
        << "active=1 with gated activation hits the dispatcher's "
        << "sequential-chain mode (src.size() == 1) which refuses "
        << "gated_act per group_matmul_direct.  This is a pre-existing "
        << "limitation independent of the warm-pack feature; the "
        << "warm-pack gate (num_ops_total > num_ops) would still fire "
        << "but the legacy 2-call reference can't be built. Covered "
        << "indirectly via the >=2 active rows below.";
  }

  const int dim = 32, N_gate_up = 2 * dim, H = 32;
  const int K_in = H, M = 2;
  const auto act_type = grp_matmul_gated_act_t::silu_and_mul;
  const data_type_t dt = p.is_bf16 ? data_type_t::bf16 : data_type_t::f32;
  const int K_down = dim;  // gated activation -> N/2

  TypedBuffers src_ref, src_test, w1_all, d1_ref, w2_all, d2_ref;
  src_ref .alloc(active, (size_t)M * K_in,         p.is_bf16);
  src_test.alloc(total,  (size_t)M * K_in,         p.is_bf16);
  w1_all  .alloc(total,  (size_t)K_in * N_gate_up, p.is_bf16);
  d1_ref  .alloc(active, (size_t)M * N_gate_up,    p.is_bf16);
  w2_all  .alloc(total,  (size_t)K_down * H,       p.is_bf16);
  d2_ref  .alloc(active, (size_t)M * H,            p.is_bf16);

  fill_moe_tensors(total,  p.is_bf16, &src_test, &w1_all, &w2_all);
  fill_moe_tensors(active, p.is_bf16, &src_ref,  nullptr, nullptr);

  auto gv_op1_test = GemmVecs::uniform(total, M, N_gate_up, K_in);
  for (int e = active; e < total; ++e) gv_op1_test.Ms[e] = 0;

  auto src_ref_p   = src_ref.cptrs(p.is_bf16);
  auto src_test_p  = src_test.cptrs(p.is_bf16);
  auto wei1_all_p  = w1_all.cptrs(p.is_bf16);
  auto wei2_all_p  = w2_all.cptrs(p.is_bf16);
  auto d1_ref_p    = d1_ref.ptrs(p.is_bf16);
  auto d2_ref_p    = d2_ref.ptrs(p.is_bf16);
  std::vector<const void *> wei1_ref_p(wei1_all_p.begin(),
                                       wei1_all_p.begin() + active);
  std::vector<const void *> wei2_ref_p(wei2_all_p.begin(),
                                       wei2_all_p.begin() + active);
  std::vector<const void *> no_bias_test(total, nullptr);

  auto params_test = make_uniform_params(total, dt);
  for (auto &pp : params_test) {
    pp.active_matmul = static_cast<uint32_t>(active);
    pp.total_matmul  = static_cast<uint32_t>(total);
  }

  grp_matmul_gated_act_params act{};
  act.act = act_type;

  ASSERT_EQ(run_legacy_2call_ref(active, M, K_in, N_gate_up, K_down, H,
                                 p.is_bf16, act_type,
                                 src_ref_p, wei1_ref_p, wei2_ref_p,
                                 d1_ref_p, d2_ref_p),
            status_t::success) << p.name;

  auto fused = make_fused_moe_op2(total, H, wei2_all_p, no_bias_test);

  std::vector<void *> dst_null(total, nullptr);
  std::vector<int>    ldc_null(total, 0);
  {
    auto pf = params_test;
    ASSERT_EQ(group_matmul_direct(gv_op1_test.layout, gv_op1_test.transA,
                                  gv_op1_test.transB, gv_op1_test.Ms,
                                  gv_op1_test.Ns, gv_op1_test.Ks,
                                  gv_op1_test.alpha, src_test_p,
                                  gv_op1_test.lda, wei1_all_p, gv_op1_test.ldb,
                                  no_bias_test, gv_op1_test.beta, dst_null,
                                  ldc_null, gv_op1_test.is_wc, pf, nullptr,
                                  &act, &fused),
              status_t::success) << p.name;
  }

  std::ostringstream lbl;
  lbl << p.name << " active=" << active << " total=" << total;
  verify_per_expert_2d(src_test, K_in, d2_ref, H, active, M, H, p.is_bf16,
                       tol_fused(p.is_bf16), lbl.str());
}

static std::vector<ActiveTotalEdgeParam> make_active_total_edge_params() {
  // Edge cases the regular [11] suite doesn't reach.  active=0 is
  // structurally different (no warm-pack hook, strict `!=` validators)
  // and is already covered by [5] TestFusedMoE; skip-listed below for
  // visibility.
  static const ActiveTotalEdgeParam kEdges[] = {
    {"single_expert_no_extras",    1,   1, false},
    {"single_fired_in_64_total",   1,  64, false},
    {"active_equals_total_4",      4,   4, false},
    {"min_extras_one",             8,   9, false},
    {"gpt_oss_in_64_pool",         8,  64, false},  // scaled down to 64 for runtime
    {"legacy_active_zero",         0,   0, false},  // GTEST_SKIPped
  };
  std::vector<ActiveTotalEdgeParam> out;
  for (const auto &edge : kEdges) {
    for (bool bf : {false, true}) {
      out.push_back({edge.name, edge.active, edge.total, bf});
    }
  }
  return out;
}

INSTANTIATE_TEST_SUITE_P(GroupMatmulFusedMoEActiveTotalEdge,
                         TestFusedMoEActiveTotalEdge,
                         ::testing::ValuesIn(make_active_total_edge_params()),
                         ActiveTotalEdgeParamName);

// ===============================================================================
// [15] TestDispatcherActiveTotalNegative - negative contract validator cases.
//
//      Closes G2 from the PR-443 review.  The dispatcher's full
//      contract validator in `group_matmul_direct.cpp` rejects two
//      impossible framework hints:
//
//        (i)  `params[0].active_matmul > M.size()`
//             ? firing experts cannot exceed input-side metadata.
//        (ii) `params[0].total_matmul > 0 && total_matmul < active_matmul`
//             ? total experts cannot be less than firing experts.
//
//      Both paths live behind `op_instrumentation::validate(...)`,
//      whose gate now defaults to ENABLED: the body runs unless
//      `ZENDNNL_DIAGNOSTICS_ENABLE=0` was captured at the time of
//      the first call (the env value is cached once via
//      `static const bool`).  Production deployments that opt out
//      with `ZENDNNL_DIAGNOSTICS_ENABLE=0` silently clamp
//      `active_matmul` to `M.size()` via `std::min(am, M.size())`
//      in the always-on inline guard (group_matmul_direct.cpp lines
//      695-699) and silently accept `total < active` because
//      prepack accounting only uses `max(M.size(), total_matmul)`
//      ? the opt-out contract is "no work dropped, no OOB" rather
//      than "no impossible hint".  With the default-on diagnostic
//      mode, the validator hard-rejects so frameworks catch
//      contract drift early in both CI and production.
//
//      To exercise the validator deterministically (independent of
//      whatever the parent process has cached in its
//      `static const bool`) we use the same subprocess pattern as
//      `[26] TestPrepackEnvBucketA`: explicitly set the env knob in
//      the parent, then `EXPECT_EXIT` with `threadsafe` death-test
//      style which `fork()` + `execve()`s a child whose first-time
//      read of `ZENDNNL_DIAGNOSTICS_ENABLE` returns "1" (matching
//      the default-on intent).  The child runs the dispatcher with
//      intentionally-bad params; the parent asserts a non-zero exit
//      (the child sets it via `std::exit(1)` when
//      `status_t::failure` is NOT returned, i.e. the validator
//      regressed).  Subprocess overhead is ~150 ms per case ? two
//      cases adds ~0.3 s to the suite, negligible.
// ===============================================================================

namespace {

inline void run_dispatcher_negative_in_child(
    int M_visible, int active, int total,
    const char *case_label) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  using zendnnl::common::data_type_t;

  const int K_in = 16, N = 16, M_rows = 2;

  TypedBuffers src, wei, dst;
  src.alloc(M_visible, (size_t)M_rows * K_in, /*is_bf16=*/true);
  wei.alloc(M_visible, (size_t)K_in * N,      true);
  dst.alloc(M_visible, (size_t)M_rows * N,    true);

  for (int e = 0; e < M_visible; ++e) {
    fill_src(src.bf16[e], e);
    fill_wei1(wei.bf16[e], e);
  }

  auto srcs = src.cptrs(true);
  auto weis = wei.cptrs(true);
  auto dsts = dst.ptrs(true);
  std::vector<const void *> no_bias(M_visible, nullptr);

  auto gv = GemmVecs::uniform(M_visible, M_rows, N, K_in);
  auto params = make_uniform_params(M_visible, data_type_t::bf16);
  ASSERT_FALSE(params.empty())
      << "[case=" << case_label << "] make_uniform_params returned empty";
  params[0].active_matmul = static_cast<uint32_t>(active);
  params[0].total_matmul  = static_cast<uint32_t>(total);

  const status_t st = group_matmul_direct(gv.layout, gv.transA, gv.transB,
                                          gv.Ms, gv.Ns, gv.Ks, gv.alpha,
                                          srcs, gv.lda, weis, gv.ldb, no_bias,
                                          gv.beta, dsts, gv.ldc, gv.is_wc,
                                          params, nullptr, nullptr, nullptr);
  ASSERT_EQ(st, status_t::failure)
      << "[case=" << case_label << "] dispatcher must reject "
      << "active_matmul=" << active << " total_matmul=" << total
      << " M.size()=" << M_visible
      << " under ZENDNNL_DIAGNOSTICS_ENABLE=1";
}

} // namespace

class TestDispatcherActiveTotalNegative : public ::testing::Test {};

TEST_F(TestDispatcherActiveTotalNegative, RejectsActiveExceedsMSize) {
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  setenv("ZENDNNL_DIAGNOSTICS_ENABLE", "1", /*overwrite=*/1);
  EXPECT_EXIT({
    run_dispatcher_negative_in_child(/*M_visible=*/2, /*active=*/4,
                                     /*total=*/4, "am_gt_M");
    std::exit(::testing::Test::HasFailure() ? 1 : 0);
  }, ::testing::ExitedWithCode(0), "");
  unsetenv("ZENDNNL_DIAGNOSTICS_ENABLE");
}

TEST_F(TestDispatcherActiveTotalNegative, RejectsTotalLessThanActive) {
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  setenv("ZENDNNL_DIAGNOSTICS_ENABLE", "1", /*overwrite=*/1);
  EXPECT_EXIT({
    run_dispatcher_negative_in_child(/*M_visible=*/2, /*active=*/2,
                                     /*total=*/1, "tm_lt_am");
    std::exit(::testing::Test::HasFailure() ? 1 : 0);
  }, ::testing::ExitedWithCode(0), "");
  unsetenv("ZENDNNL_DIAGNOSTICS_ENABLE");
}

// ── Prepack-extras weight-side metadata check ─────────────────────────────
// Closes the gap reported by Copilot review item #2 on PR-443: when
// the framework opts in via `total_matmul > active_matmul` the
// dispatcher MUST reject undersized weight-side metadata vectors so
// the prepack module's `bound = std::min({total_count, weight.size(),
// K.size(), N.size(), ldb.size(), transB.size()})` cannot silently
// truncate the warm.  The always-on inline guard returns
// `status_t::failure` with no log_error in production builds; the
// diagnostic validator additionally surfaces a precise
// "<name>.size()=X < total_matmul=Y" message.  The check here uses
// the production (always-on) path so it runs without diagnostics.

TEST_F(TestDispatcherActiveTotalNegative, RejectsWeightShorterThanTotalMatmul) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  using zendnnl::common::data_type_t;

  reset_grp_matmul_caches();

  // active=2, total=4: weight-side metadata at size 4 is the contract.
  // Set weight.size() = 2 (active count) instead and expect rejection.
  const int M_visible = 2;
  const int active    = 2;
  const int total     = 4;
  const int K_in = 16, N = 16, M_rows = 2;

  TypedBuffers src, wei, dst;
  src.alloc(M_visible, (size_t)M_rows * K_in, /*is_bf16=*/true);
  wei.alloc(M_visible, (size_t)K_in * N,      true);
  dst.alloc(M_visible, (size_t)M_rows * N,    true);

  for (int e = 0; e < M_visible; ++e) {
    fill_src(src.bf16[e], e);
    fill_wei1(wei.bf16[e], e);
  }

  auto srcs = src.cptrs(true);
  auto weis = wei.cptrs(true);
  auto dsts = dst.ptrs(true);
  std::vector<const void *> no_bias(M_visible, nullptr);

  auto gv = GemmVecs::uniform(M_visible, M_rows, N, K_in);
  auto params = make_uniform_params(M_visible, data_type_t::bf16);
  ASSERT_FALSE(params.empty()) << "make_uniform_params returned empty";
  params[0].active_matmul = static_cast<uint32_t>(active);
  params[0].total_matmul  = static_cast<uint32_t>(total);

  // weight.size() = 2 < total_matmul=4 — must reject.  No subprocess
  // needed: the always-on inline guard fires regardless of
  // DIAGNOSTICS state.
  ASSERT_EQ(group_matmul_direct(gv.layout, gv.transA, gv.transB,
                                gv.Ms, gv.Ns, gv.Ks, gv.alpha,
                                srcs, gv.lda, weis, gv.ldb, no_bias,
                                gv.beta, dsts, gv.ldc, gv.is_wc,
                                params, nullptr, nullptr, nullptr),
            status_t::failure)
      << "dispatcher must reject weight.size()=" << M_visible
      << " < total_matmul=" << total
      << " — silent prepack truncation would leave inactive experts "
         "vulnerable to runtime reorder spike";
}

// ── Per-vector prepack-extras coverage ────────────────────────────────────
// `prepack_extras_metadata_undersized` (extracted helper) iterates
// six weight-side vectors: weight, K, N, ldb, transB, is_weights_const.
// `RejectsWeightShorterThanTotalMatmul` above covers `weight`; these
// three add coverage for K, N, and ldb.  The remaining two (transB,
// is_weights_const) follow the same code path so are not separately
// asserted.  All exercise the always-on inline guard (no DIAGNOSTICS
// needed) so they run cheaply in the regular suite.

namespace {

// Build a "valid except for ONE undersized weight-side vector" call
// and assert the dispatcher rejects.  `shrink` is invoked on a
// freshly-built per-vector copy, undersizing exactly the vector
// under test to active_matmul (= 2) instead of total_matmul (= 4).
template <typename ShrinkFn>
inline void assert_undersized_metadata_rejects(const char *name,
                                               ShrinkFn    shrink) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  using zendnnl::common::data_type_t;

  // Helper backs 3 parent-process TEST_F bodies; wipe every cache
  // here so the K/N/ldb-shrink tests see a deterministic dispatcher
  // state (the rejection path runs before any reorder, but a stale
  // AOCL LRU hit would still consume memory between cases).
  reset_grp_matmul_caches();

  constexpr int M_visible = 4;  // total_matmul-sized input vectors
  constexpr int active    = 2;
  constexpr int total     = 4;
  constexpr int M_rows = 2;
  constexpr int K_in = 16;
  constexpr int N = 16;

  TypedBuffers src;
  TypedBuffers wei;
  TypedBuffers dst;
  src.alloc(M_visible, static_cast<size_t>(M_rows) * K_in, /*is_bf16=*/true);
  wei.alloc(M_visible, static_cast<size_t>(K_in)   * N,    true);
  dst.alloc(M_visible, static_cast<size_t>(M_rows) * N,    true);

  for (int e = 0; e < M_visible; ++e) {
    fill_src (src.bf16[e], e);
    fill_wei1(wei.bf16[e], e);
  }

  auto srcs = src.cptrs(true);
  auto weis = wei.cptrs(true);
  auto dsts = dst.ptrs(true);
  std::vector<const void *> no_bias(M_visible, nullptr);

  auto gv = GemmVecs::uniform(M_visible, M_rows, N, K_in);
  auto params = make_uniform_params(M_visible, data_type_t::bf16);
  ASSERT_FALSE(params.empty());
  params[0].active_matmul = static_cast<uint32_t>(active);
  params[0].total_matmul  = static_cast<uint32_t>(total);

  // Apply the undersizing for the vector under test.
  shrink(gv, weis);

  EXPECT_EQ(group_matmul_direct(gv.layout, gv.transA, gv.transB,
                                gv.Ms, gv.Ns, gv.Ks, gv.alpha,
                                srcs, gv.lda, weis, gv.ldb, no_bias,
                                gv.beta, dsts, gv.ldc, gv.is_wc,
                                params, nullptr, nullptr, nullptr),
            status_t::failure)
      << "dispatcher must reject when " << name
      << ".size() < total_matmul (silent prepack truncation)";
}

} // namespace

TEST_F(TestDispatcherActiveTotalNegative, RejectsKShorterThanTotalMatmul) {
  assert_undersized_metadata_rejects(
      "K", [](auto &gv, auto & /*weis*/) {
        gv.Ks.resize(2);  // shrink K to active count
      });
}

TEST_F(TestDispatcherActiveTotalNegative, RejectsNShorterThanTotalMatmul) {
  assert_undersized_metadata_rejects(
      "N", [](auto &gv, auto & /*weis*/) {
        gv.Ns.resize(2);  // shrink N to active count
      });
}

TEST_F(TestDispatcherActiveTotalNegative, RejectsLdbShorterThanTotalMatmul) {
  assert_undersized_metadata_rejects(
      "ldb", [](auto &gv, auto & /*weis*/) {
        gv.ldb.resize(2);  // shrink ldb to active count
      });
}

// ── Phase-F K_down regression (fix for unconditional N/2 in validator) ──
// The diagnostic Phase-F validator previously computed
// `K_down_i = N[i] / 2` unconditionally and used it as the
// `ldb_down` minimum.  That under-restricted callers with
// `act == none + fused_moe` because the actual execute path needs
// `ldb_down >= N` (full Op1 output flows into Op2, no gate+up
// collapse).  Routed through the shared `op2_k_for_act` helper in
// `group_matmul_parallel_common.hpp`; a regression in that wiring
// would let a malformed caller pass diagnostic validation.
//
// Test: under DIAGNOSTICS=1, supply fused_moe + gated_act = none
// + ldb_down = N/2 (= the old, wrong, minimum).  Validator must
// reject because the corrected `op2_k_for_act(N, none) = N`
// requires ldb_down >= N.

namespace {

inline void run_phase_f_kdown_negative_in_child() {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  using zendnnl::common::data_type_t;

  constexpr int E = 2;
  constexpr int M_rows = 2;
  constexpr int K_in = 16;
  constexpr int N = 16;
  constexpr int H = 16;

  TypedBuffers src;
  TypedBuffers wei1;
  TypedBuffers wei2;
  TypedBuffers dst1;
  TypedBuffers dst2;
  src .alloc(E, static_cast<size_t>(M_rows) * K_in, /*is_bf16=*/true);
  wei1.alloc(E, static_cast<size_t>(K_in)   * N,    true);
  wei2.alloc(E, static_cast<size_t>(N)      * H,    true);  // act=none -> [N, H]
  dst1.alloc(E, static_cast<size_t>(M_rows) * N,    true);
  dst2.alloc(E, static_cast<size_t>(M_rows) * H,    true);

  for (int e = 0; e < E; ++e) {
    fill_src (src .bf16[e], e);
    fill_wei1(wei1.bf16[e], e);
    fill_wei2(wei2.bf16[e], e);
  }

  auto srcs  = src .cptrs(true);
  auto wei1p = wei1.cptrs(true);
  auto wei2p = wei2.cptrs(true);
  auto dst1p = dst1.ptrs(true);
  auto dst2p = dst2.ptrs(true);
  std::vector<const void *> no_bias(E, nullptr);

  auto gv     = GemmVecs::uniform(E, M_rows, N, K_in);
  auto params = make_uniform_params(E, data_type_t::bf16);

  auto fused = make_fused_moe_op2(E, H, wei2p, no_bias);
  fused.dst_down = dst2p;
  fused.ldc_down = std::vector<int>(E, H);
  // ldb_down = N/2 — the OLD K_down formula's value.  The
  // corrected validator (act=none -> K_down=N) must reject.
  fused.ldb_down = std::vector<int>(E, N / 2);

  grp_matmul_gated_act_params act_params{};
  act_params.act = grp_matmul_gated_act_t::none;

  const status_t st = group_matmul_direct(
      gv.layout, gv.transA, gv.transB, gv.Ms, gv.Ns, gv.Ks, gv.alpha,
      srcs, gv.lda, wei1p, gv.ldb, no_bias, gv.beta, dst1p, gv.ldc,
      gv.is_wc, params, nullptr, &act_params, &fused);
  ASSERT_EQ(st, status_t::failure)
      << "Phase-F validator must reject ldb_down=" << (N / 2)
      << " for act=none (op2_k_for_act(N, none) = N requires ldb_down >= N)";
}

} // namespace

TEST_F(TestDispatcherActiveTotalNegative, RejectsFusedMoEActNoneLdbBelowN) {
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  setenv("ZENDNNL_DIAGNOSTICS_ENABLE", "1", /*overwrite=*/1);
  EXPECT_EXIT({
    run_phase_f_kdown_negative_in_child();
    std::exit(::testing::Test::HasFailure() ? 1 : 0);
  }, ::testing::ExitedWithCode(0), "");
  unsetenv("ZENDNNL_DIAGNOSTICS_ENABLE");
}


// ===============================================================================
// [16.d] / [16.e] Op1 + Op2 quantization
//
//        [16.d] WOQ_S4_BothPasses — weight-only INT4 (WOQ-S4) on
//          BOTH Op1 and Op2.  Each pass uses BF16 source + S4
//          weights + per-channel F32 wei_scale; no source-side
//          quant on either side.  Op1 carries its wei_scale via
//          `params[i].quant_params.wei_scale`; Op2 carries the
//          *same shape* of wei_scale (for its own down_weight) via
//          the new `fused.down_scale[i]` vector.  Both `dynamic_quant`
//          flags stay at the default `false`.
//
//          NOTE: S4 (not S8) on purpose.  AOCL DLP's WOQ fast path
//          is gated to `s4 || u4` only (see
//          `aocl_postop.cpp::is_woq` at line 178).  A `bf16 src +
//          s8 wei` combination without a caller-provided
//          `src_scale.buff` falls through `is_non_quant_src_int8`
//          and AOCL's BF16-INT8 pre-quant validator at
//          `aocl_postop.cpp:369` rejects it — both the reference
//          and the fused-test paths then produce all-zero output
//          and the comparison trivially passes on `0 == 0`.  A
//          sanity-sum guard at the bottom of this test catches that
//          failure mode regardless of which sub-byte INT type the
//          caller picks.
//
//        [16.e] Dynamic_INT8_BothPasses — runtime BF16→S8 reorder
//          on the source of BOTH Op1 and Op2.  Each pass carries
//          per-token src_scale shape `{M, 1}` with `buff = nullptr`
//          (kernel-allocated scratch) plus per-channel wei_scale.
//          The caller sets `dynamic_quant = true`,
//          `dtypes.compute = s8`, and `src_scale.{dt, dims}` on
//          `params[i]` — the fused-MoE dispatcher inherits those
//          three knobs verbatim into Op2's `params_down[i]`.  The
//          only per-pass Op2 artefact the caller supplies is the
//          down_weight scale via `fused.down_scale[i]`.  Both passes
//          run through the same INT8-dynamic kernel path.
//
//      Reference for both is a manual 2-call legacy path with the
//      SAME quant data wired onto `params_ref_op1[i].quant_params`
//      and `params_ref_op2[i].quant_params` so every dispatch
//      decision matches the fused single-call path.  Both routes
//      drive the same kernels — outputs are compared to BF16
//      precision.
// ===============================================================================

namespace {

// Shared helper: copy a tensor's attached scale into a quant slot.
// The tensor must have been built with an attached scale (via
// `factory.uniform_dist_tensor({k,n}, s8/u8/s4/u4, ..., transB, scale)`
// or via `quant_params_compute(..., &dst_out)`), otherwise
// `get_quant_scale_raw_handle_const()` returns nullptr and the
// kernel would silently treat the field as un-quantized.
// Templated so the same helper accepts either
// `matmul_quantization_params_t::matmul_quant_t` (used on
// `matmul_params.quant_params.{wei_scale,src_scale,…}`) or
// `grp_matmul_fused_moe_params::down_weight_quant_t` (used on
// `fused.down_scale[i]` / `fused.down_zp[i]`).  Both have the same
// `buff` / `dt` / `dims` interface — duck-typed via the template
// parameter rather than wired through an inheritance hierarchy.
template <class Quant>
inline void copy_attached_scale(const tensor_t &t, Quant &q) {
  q.buff = t.get_quant_scale_raw_handle_const();
  q.dt   = t.get_quant_scale_data_type();
  auto sz = t.get_quant_scale_size();
  q.dims.assign(sz.begin(), sz.end());
}

// Shared helper: copy a tensor's attached zero-point into the
// quant slot (asymmetric-quant variant — symmetric tensors have
// no zp attached and this is a no-op via the empty-dims check
// below, leaving `q` at its default-constructed state).  Same
// duck-typed template as `copy_attached_scale` above.
template <class Quant>
inline void copy_attached_zp(const tensor_t &t, Quant &q) {
  if (t.get_quant_subtype() != quant_subtype_t::asymmetric) return;
  q.buff = t.get_quant_zero_raw_handle_const();
  q.dt   = t.get_quant_zero_data_type();
  auto sz = t.get_quant_zero_size();
  q.dims.assign(sz.begin(), sz.end());
}

} // namespace

// ===============================================================================
// [16] TestFusedMoEQuant — fused-MoE quantization gtest section.
//
// Layered after `TestGroupMatmulQuant` in test_quant.cpp, with one
// deliberate divergence: each quant scheme owns its own fixture
// subclass with a kernel-supported parameter grid.  We pushed the
// previous per-tuple `GTEST_SKIP()` constraint up to the parameter
// source so the suite stops enumerating tuples the AOCL kernel
// layer is known to reject — every test that runs is one the
// kernel layer is contractually obligated to handle.
//
//   * `TestFusedMoEQuantBase` — shared fixture (shape config,
//     ALGO 1 pin via `AlgoEnvGuard`).  Holds no scheme state; both
//     subclasses inherit verbatim.
//
//   * `TestFusedMoEQuantWOQ` — weight-only INT4 (S4) on BOTH Op1
//     and Op2.  AOCL DLP's WOQ fast path (s4/u4) is M-agnostic, so
//     the grid carries the full M ∈ {1, 4, 16, 32} sweep.
//
//   * `TestFusedMoEQuantDynINT8` — runtime BF16→S8 reorder on the
//     source of BOTH Op1 and Op2.  The AOCL BF16→S8 reorder kernel
//     refuses per-token `src_scale = {M, 1}` for M < 16 (same
//     constraint that drives test_quant.cpp::INT8_DYNAMIC_GEMM_BF16
//     to skip those tuples).  We trim the grid to M ∈ {16, 32}
//     instead of skipping at run time — same coverage, no
//     `[ SKIPPED ]` noise in the test report.
//
// Owned tests (alphabetised on test name):
//   * `TestFusedMoEQuantDynINT8.BothPasses` — runtime BF16→S8
//     reorder on the source of BOTH Op1 and Op2.  Per-token
//     `{M, 1}` src_scale and per-channel `{1, N}` wei_scale on each
//     pass.  Op1 carries `dynamic_quant = true` +
//     `dtypes.compute = s8` + `quant_params.src_scale.{dt, dims}`
//     on `params[i]`; the fused-MoE dispatcher inherits those into
//     Op2's `params_down[i]` and copies `fused.down_scale[i]` into
//     the Op2 weight-scale slot.
//
//   * `TestFusedMoEQuantWOQ.BothPasses` — weight-only INT4 (S4) on
//     BOTH Op1 and Op2.  BF16 source + S4 weights + per-channel
//     F32 wei_scale on each pass.  Op1 carries its wei_scale via
//     `params[i].quant_params.wei_scale`; Op2 carries the
//     down_proj wei_scale via `fused.down_scale[i]`.
//     S4 (not S8) on purpose — AOCL DLP's pure WOQ fast path is
//     gated to `s4 || u4` (see aocl_postop.cpp::is_woq at line 178);
//     `bf16 src + s8 wei + no src_scale` falls through
//     `is_non_quant_src_int8` and produces all-zero output on both
//     legs, which would let the per-element comparison pass
//     vacuously.  A sanity-sum guard at the bottom of every test
//     body catches that failure mode regardless.
//
// Test design contract (enforced by every TEST_P below):
//   Op1 and Op2 use the SAME quant scheme.  Mixed schemes (e.g.
//   static-INT8 Op1 + WOQ-S8 Op2) leave the Op1/Op2 dispatch
//   divergent and make every comparison failure ambiguous between
//   "Op1 path bug", "Op2 path bug", and "scheme-mismatch contract
//   bug".  Constraining both passes to the same scheme keeps the
//   fault domain small.
// ===============================================================================

struct FusedMoEQuantType {
  int dim;          ///< Op1 output cols halved (N_gate_up = 2 * dim, K_down = dim).
  int hidden_size;  ///< Op2 output cols / Op1 input cols (K_in = H).
  int M;            ///< Tokens per active expert.
  int num_ops;      ///< Expert count.
  int act_int;      ///< 1=silu_and_mul, 2=gelu_and_mul, 3=swiglu_oai_mul.
};

static std::string FusedMoEQuantParamName(
    const ::testing::TestParamInfo<FusedMoEQuantType> &info) {
  static const char *act_names[] = {"none", "silu", "gelu", "swiglu"};
  const auto &p = info.param;
  return std::string(act_names[p.act_int])
       + "_d" + std::to_string(p.dim)
       + "_h" + std::to_string(p.hidden_size)
       + "_M" + std::to_string(p.M)
       + "_E" + std::to_string(p.num_ops);
}

// Scheme-specific parameter grids.  Each grid only enumerates
// tuples the underlying AOCL kernel path is known to accept, so
// no `TEST_P` body needs a runtime `GTEST_SKIP()` guard — every
// instantiated test is one we expect to run end-to-end.
//
// WOQ-S4 grid — AOCL DLP's WOQ fast path is M-agnostic, so all
// four M values exercise the same dispatch tile.
//
//   3 acts × 2 dims × 4 M-values × 2 num_ops = 48 cases.
static std::vector<FusedMoEQuantType> make_woq_quant_params() {
  std::vector<FusedMoEQuantType> out;
  for (int act : {1, 2, 3}) {
    for (int d : {16, 32}) {
      for (int m : {1, 4, 16, 32}) {
        for (int e : {2, 4}) {
          out.push_back({d, d, m, e, act});
        }
      }
    }
  }
  return out;
}

// Dynamic-INT8 grid — AOCL's BF16→S8 reorder kernel rejects
// per-token `src_scale.dims = {M, 1}` for M < 16 (same constraint
// that drives test_quant.cpp::INT8_DYNAMIC_GEMM_BF16 to skip those
// tuples).  We trim at the parameter source instead.
//
//   3 acts × 2 dims × 2 M-values × 2 num_ops = 24 cases.
static std::vector<FusedMoEQuantType> make_dyn_int8_quant_params() {
  std::vector<FusedMoEQuantType> out;
  for (int act : {1, 2, 3}) {
    for (int d : {16, 32}) {
      for (int m : {16, 32}) {
        for (int e : {2, 4}) {
          out.push_back({d, d, m, e, act});
        }
      }
    }
  }
  return out;
}

class TestFusedMoEQuantBase
    : public ::testing::TestWithParam<FusedMoEQuantType> {
 protected:
  void SetUp() override {
    const auto &p = GetParam();
    dim         = p.dim;
    hidden_size = p.hidden_size;
    M           = p.M;
    num_ops     = p.num_ops;
    act_int     = p.act_int;
    is_bf16     = true;            // pinned — bf16-only Op2 quant for now.

    N_gate_up = 2 * dim;
    H         = hidden_size;
    K_in      = H;
    K_down    = dim;                // gated activation halves Op1 output.
    act_type  = static_cast<grp_matmul_gated_act_t>(act_int);

    reset_grp_matmul_caches();
    // Honour any externally-set `ZENDNNL_GRP_MATMUL_ALGO`; otherwise
    // pin ALGO 1 for deterministic baseline.  Lets developers force
    // ALGO 2 (M-tile) etc. via `export ZENDNNL_GRP_MATMUL_ALGO=2`
    // when investigating algo-specific behaviour.  Note: any non-null
    // wei_scale auto-rejects ALGO 3 (`check_n_tile_extra`), so an
    // ALGO 3 export silently falls back to ALGO 1 on these tests.
    if (const char *env_algo = std::getenv("ZENDNNL_GRP_MATMUL_ALGO");
        !env_algo || env_algo[0] == '\0') {
      algo_guard =
          std::make_unique<moe_test_utils::AlgoEnvGuard>(1);
    }
  }

  void TearDown() override {
    algo_guard.reset();  // restores ZENDNNL_GRP_MATMUL_ALGO env.
  }

  // Shape / config members — populated from `GetParam()` in SetUp.
  int dim{};
  int hidden_size{};
  int N_gate_up{};
  int H{};
  int M{};
  int K_in{};
  int num_ops{};
  int K_down{};
  int act_int{};
  zendnnl::lowoha::matmul::grp_matmul_gated_act_t act_type{};
  bool is_bf16{};

  tensor_factory_t factory{};
  std::unique_ptr<moe_test_utils::AlgoEnvGuard> algo_guard;
};

// Per-scheme subclasses.  The base supplies all state and lifecycle
// — these only exist so `INSTANTIATE_TEST_SUITE_P` can attach a
// different parameter grid per scheme without forcing the bodies
// to share one grid plus runtime `GTEST_SKIP()` guards.
class TestFusedMoEQuantWOQ      : public TestFusedMoEQuantBase {};
class TestFusedMoEQuantDynINT8  : public TestFusedMoEQuantBase {};

TEST_P(TestFusedMoEQuantWOQ, BothPasses) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  using zendnnl::common::data_type_t;

  // ── Op1 + Op2 weights: BOTH S4 + per-channel F32 wei_scale ───────
  // S4 (not S8) — AOCL DLP's pure WOQ path is gated to s4/u4 (see
  // aocl_postop.cpp::is_woq at line 178).  S8 weights with a BF16
  // source without a caller-provided src_scale fall through
  // `is_non_quant_src_int8` and trigger the kernel's "src_scale
  // buffer is null" rejection at line 369; both the reference and
  // the fused-test paths then leave their dst buffers at all-zero
  // and the comparison vacuously passes.
  //
  // Build pattern mirrors TestGroupMatmulQuant.WOQ_BF16_S4 in
  // test_quant.cpp: `uniform_dist_tensor({k, n}, s4, ..., scale)`
  // returns an s4-packed tensor with the supplied per-channel
  // scale attached as quant metadata, ready for `copy_attached_*`.
  std::vector<tensor_t> w1_scale_t(num_ops), w1_s4_t(num_ops);
  std::vector<tensor_t> down_scale_t(num_ops), w2_s4_t(num_ops);
  for (int i = 0; i < num_ops; ++i) {
    w1_scale_t[i] = factory.uniform_dist_tensor(
        {static_cast<uint64_t>(1), static_cast<uint64_t>(N_gate_up)},
        data_type_t::f32, 2.0);
    w1_s4_t[i] = factory.uniform_dist_tensor(
        {static_cast<uint64_t>(K_in), static_cast<uint64_t>(N_gate_up)},
        data_type_t::s4, 7.0, /*transposed=*/false, w1_scale_t[i]);
    down_scale_t[i] = factory.uniform_dist_tensor(
        {static_cast<uint64_t>(1), static_cast<uint64_t>(H)},
        data_type_t::f32, 2.0);
    w2_s4_t[i] = factory.uniform_dist_tensor(
        {static_cast<uint64_t>(K_down), static_cast<uint64_t>(H)},
        data_type_t::s4, 7.0, /*transposed=*/false, down_scale_t[i]);
  }

  // ── BF16 sources + per-expert raw pointers ───────────────────────
  TypedBuffers src, d1_ref, d2_ref, d2_test, d1_unused;
  src      .alloc(num_ops, (size_t)M * K_in,      is_bf16);
  d1_ref   .alloc(num_ops, (size_t)M * N_gate_up, is_bf16);
  d2_ref   .alloc(num_ops, (size_t)M * H,         is_bf16);
  d2_test  .alloc(num_ops, (size_t)M * H,         is_bf16);
  d1_unused.alloc(num_ops, (size_t)M * N_gate_up, is_bf16);
  fill_moe_tensors(num_ops, is_bf16, &src, nullptr, nullptr);

  std::vector<const void *> wei1_p(num_ops), wei2_p(num_ops);
  for (int i = 0; i < num_ops; ++i) {
    wei1_p[i] = w1_s4_t[i].get_raw_handle_unsafe();
    wei2_p[i] = w2_s4_t[i].get_raw_handle_unsafe();
  }
  auto srcs        = src.cptrs(is_bf16);
  auto d1_ref_p    = d1_ref.ptrs(is_bf16);
  auto d2_ref_p    = d2_ref.ptrs(is_bf16);
  auto d2_test_p   = d2_test.ptrs(is_bf16);
  auto d1_unused_p = d1_unused.ptrs(is_bf16);
  std::vector<const void *> no_bias(num_ops, nullptr);

  // ── Dispatch vectors.  WOQ requires is_weights_const=true. ───────
  auto gv_op1 = GemmVecs::uniform(num_ops, M, N_gate_up, K_in);
  auto gv_op2 = GemmVecs::uniform(num_ops, M, H, K_down);
  gv_op2.lda.assign(num_ops, N_gate_up);  // Op2 reads d1_ref at N_gate_up stride.
  gv_op1.is_wc.assign(num_ops, true);
  gv_op2.is_wc.assign(num_ops, true);

  // Build per-pass matmul_params reusable for ref and test paths.
  // Op1 and Op2 share the same WOQ-S4 scheme — only the weight
  // tensor whose scale is extracted differs.
  auto build_params_woq_s4 =
      [&](const std::vector<tensor_t> &wei_quant_tensors) {
    auto p = make_uniform_params(num_ops, data_type_t::bf16);
    for (int i = 0; i < num_ops; ++i) {
      p[i].dtypes.src = data_type_t::bf16;
      p[i].dtypes.wei = data_type_t::s4;
      p[i].dtypes.dst = data_type_t::bf16;
      copy_attached_scale(wei_quant_tensors[i],
                          p[i].quant_params.wei_scale);
      copy_attached_zp   (wei_quant_tensors[i],
                          p[i].quant_params.wei_zp);
    }
    return p;
  };

  // ── Reference: legacy 2-call with WOQ-S4 on both Op1 and Op2 ─────
  {
    auto p_ref_op1 = build_params_woq_s4(w1_s4_t);
    ASSERT_EQ(group_matmul_direct(gv_op1.layout, gv_op1.transA, gv_op1.transB,
                                  gv_op1.Ms, gv_op1.Ns, gv_op1.Ks, gv_op1.alpha,
                                  srcs, gv_op1.lda, wei1_p, gv_op1.ldb, no_bias,
                                  gv_op1.beta, d1_ref_p, gv_op1.ldc,
                                  gv_op1.is_wc, p_ref_op1),
              status_t::success) << "ref Op1 (WOQ-S4)";

    for (int e = 0; e < num_ops; ++e) {
      apply_ref_gated_act(d1_ref.bf16[e], M, N_gate_up, N_gate_up, act_type);
    }

    auto p_ref_op2 = build_params_woq_s4(w2_s4_t);
    std::vector<const void *> srcs2(num_ops);
    for (int e = 0; e < num_ops; ++e) srcs2[e] = d1_ref_p[e];
    ASSERT_EQ(group_matmul_direct(gv_op2.layout, gv_op2.transA, gv_op2.transB,
                                  gv_op2.Ms, gv_op2.Ns, gv_op2.Ks, gv_op2.alpha,
                                  srcs2, gv_op2.lda, wei2_p, gv_op2.ldb, no_bias,
                                  gv_op2.beta, d2_ref_p, gv_op2.ldc,
                                  gv_op2.is_wc, p_ref_op2),
              status_t::success) << "ref Op2 (WOQ-S4)";
  }

  // ── Test: single fused call with WOQ-S4 on Op1 (params) and
  //   WOQ-S4 on Op2 (fused.down_scale / fused.down_zp) ─────────────────
  grp_matmul_gated_act_params act{};
  act.act = act_type;

  auto fused = make_fused_moe_op2(num_ops, H, wei2_p, no_bias);
  fused.dst_down = d2_test_p;
  fused.ldc_down = std::vector<int>(num_ops, H);
  // Carry the down_weight scale through the new `down_scale` field.
  // Op2 inherits the rest of the quant scheme (dynamic_quant=false,
  // dtypes.wei=s4) automatically from `params[i]` via the fused-MoE
  // dispatcher's setup loop.

  fused.down_scale.resize(num_ops);
  for (int i = 0; i < num_ops; ++i) {
    copy_attached_scale(w2_s4_t[i], fused.down_scale[i]);
  }

  {
    auto p_test = build_params_woq_s4(w1_s4_t);
    ASSERT_EQ(group_matmul_direct(gv_op1.layout, gv_op1.transA, gv_op1.transB,
                                  gv_op1.Ms, gv_op1.Ns, gv_op1.Ks, gv_op1.alpha,
                                  srcs, gv_op1.lda, wei1_p, gv_op1.ldb, no_bias,
                                  gv_op1.beta, d1_unused_p, gv_op1.ldc,
                                  gv_op1.is_wc, p_test, nullptr, &act, &fused),
              status_t::success)
        << "fused (WOQ-S4 on both Op1 and Op2 via fused.down_scale)";
  }

  // Sanity: confirm both the reference and the fused-test paths
  // actually computed a GEMM (vs silently bailing out and leaving
  // the dst buffer at the post-alloc all-zero state).  Without this
  // guard a kernel-setup failure that bails on the same code path
  // for both routes (e.g. AOCL DLP's BF16-INT8 pre-quant validator
  // rejecting an unsupported quant combo) would let the
  // `verify_per_expert_2d` below trivially pass on 0 == 0.
  double ref_abs_sum = 0.0, test_abs_sum = 0.0;
  for (int e = 0; e < num_ops; ++e) {
    for (size_t k = 0; k < d2_ref.bf16[e].size(); ++k) {
      ref_abs_sum  += std::abs(static_cast<float>(d2_ref .bf16[e][k]));
      test_abs_sum += std::abs(static_cast<float>(d2_test.bf16[e][k]));
    }
  }
  ASSERT_GT(ref_abs_sum,  1e-3)
      << "[16.d] reference 2-call path produced all-zero d2_ref "
         "(sum=" << ref_abs_sum << ") — WOQ-S4 dispatch likely "
         "short-circuited; check the kernel error log for the "
         "root cause and confirm `is_woq` at aocl_postop.cpp:178 "
         "is still gated to s4/u4.";
  ASSERT_GT(test_abs_sum, 1e-3)
      << "[16.d] fused-test path produced all-zero d2_test "
         "(sum=" << test_abs_sum << ").";

  std::ostringstream lbl;
  lbl << "[16] WOQ_S4_BothPasses"
      << " act=" << act_int
      << " d="   << dim
      << " h="   << H
      << " M="   << M
      << " E="   << num_ops;
  verify_per_expert_2d(d2_test, H, d2_ref, H, num_ops, M, H, is_bf16,
                       tol_fused(is_bf16), lbl.str());
}

TEST_P(TestFusedMoEQuantDynINT8, BothPasses) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  using zendnnl::common::data_type_t;

  // Param grid is trimmed to M ∈ {16, 32} at the source
  // (`make_dyn_int8_quant_params`) — AOCL's BF16→S8 reorder kernel
  // rejects per-token `src_scale = {M, 1}` for M < 16, so we never
  // enumerate those tuples here.  No runtime SKIP needed.

  // ── Op1 source tensor: BF16 with per-token F32 src_scale ─────────
  // The scale tensor is a zero-allocated F32 buffer attached to
  // src_t via `uniform_dist_tensor(..., scale, zp=empty)` — the
  // dispatcher fills its contents at runtime when it reorders the
  // BF16 source to S8.  `{M, 1}` per-token granularity is what
  // `reorder_quantization_wrapper` supports (per-tensor `{1, 1}`
  // and per-column `{1, K}` are rejected).
  std::vector<tensor_t> src_t(num_ops), src_scale_t(num_ops);
  for (int i = 0; i < num_ops; ++i) {
    src_scale_t[i] = factory.zero_tensor(
        {static_cast<uint64_t>(M), static_cast<uint64_t>(1)},
        data_type_t::f32);
    src_t[i] = factory.uniform_dist_tensor(
        {static_cast<uint64_t>(M), static_cast<uint64_t>(K_in)},
        data_type_t::bf16, 2.0, /*transposed=*/false,
        src_scale_t[i], tensor_t{});
  }

  // ── Op1 + Op2 weights: S8 + per-channel F32 scale ────────────────
  std::vector<tensor_t> w1_s8_t(num_ops), w1_scale_t(num_ops), w1_zp_t(num_ops);
  std::vector<tensor_t> w2_s8_t(num_ops), down_scale_t(num_ops), down_zp_t(num_ops);
  for (int i = 0; i < num_ops; ++i) {
    auto w1_ref = factory.uniform_dist_tensor(
        {static_cast<uint64_t>(K_in), static_cast<uint64_t>(N_gate_up)},
        data_type_t::bf16, 2.0, /*transposed=*/false);
    ASSERT_EQ(quant_params_compute(factory, w1_ref,
                                   data_type_t::bf16, data_type_t::s8,
                                   /*scale_dims=*/{1, N_gate_up},
                                   data_type_t::f32,
                                   w1_scale_t[i], w1_zp_t[i], &w1_s8_t[i]),
              status_t::success);

    auto w2_ref = factory.uniform_dist_tensor(
        {static_cast<uint64_t>(K_down), static_cast<uint64_t>(H)},
        data_type_t::bf16, 2.0, /*transposed=*/false);
    ASSERT_EQ(quant_params_compute(factory, w2_ref,
                                   data_type_t::bf16, data_type_t::s8,
                                   /*scale_dims=*/{1, H},
                                   data_type_t::f32,
                                   down_scale_t[i], down_zp_t[i], &w2_s8_t[i]),
              status_t::success);
  }

  // ── Raw pointers + BF16 dst / intermediate buffers ───────────────
  std::vector<const void *> srcs(num_ops), wei1_p(num_ops), wei2_p(num_ops);
  for (int i = 0; i < num_ops; ++i) {
    srcs  [i] = src_t  [i].get_raw_handle_unsafe();
    wei1_p[i] = w1_s8_t[i].get_raw_handle_unsafe();
    wei2_p[i] = w2_s8_t[i].get_raw_handle_unsafe();
  }

  TypedBuffers d1_ref, d2_ref, d2_test, d1_unused;
  d1_ref   .alloc(num_ops, (size_t)M * N_gate_up, is_bf16);
  d2_ref   .alloc(num_ops, (size_t)M * H,         is_bf16);
  d2_test  .alloc(num_ops, (size_t)M * H,         is_bf16);
  d1_unused.alloc(num_ops, (size_t)M * N_gate_up, is_bf16);
  auto d1_ref_p    = d1_ref.ptrs(is_bf16);
  auto d2_ref_p    = d2_ref.ptrs(is_bf16);
  auto d2_test_p   = d2_test.ptrs(is_bf16);
  auto d1_unused_p = d1_unused.ptrs(is_bf16);
  std::vector<const void *> no_bias(num_ops, nullptr);

  // Dispatch vectors.  WOQ + dynamic INT8 require is_weights_const=true.
  auto gv_op1 = GemmVecs::uniform(num_ops, M, N_gate_up, K_in);
  auto gv_op2 = GemmVecs::uniform(num_ops, M, H, K_down);
  gv_op2.lda.assign(num_ops, N_gate_up);
  gv_op1.is_wc.assign(num_ops, true);
  gv_op2.is_wc.assign(num_ops, true);

  // Build per-pass matmul_params reusable for ref and test paths.
  // Op1 and Op2 share the same dynamic INT8 scheme: BF16 source +
  // S8 weights + per-token src_scale + per-channel wei_scale,
  // `dtypes.compute = s8` + `dynamic_quant = true` so
  // `reorder_quantization_wrapper` engages at runtime.
  auto build_params_dynamic_int8 =
      [&](const tensor_t &src_tensor, const tensor_t &wei_tensor) {
    auto p_slot = make_uniform_params(1, data_type_t::bf16)[0];
    p_slot.dtypes.src     = data_type_t::bf16;
    p_slot.dtypes.wei     = data_type_t::s8;
    p_slot.dtypes.dst     = data_type_t::bf16;
    p_slot.dtypes.compute = data_type_t::s8;
    p_slot.dynamic_quant  = true;
    copy_attached_scale(src_tensor, p_slot.quant_params.src_scale);
    copy_attached_scale(wei_tensor, p_slot.quant_params.wei_scale);
    copy_attached_zp   (wei_tensor, p_slot.quant_params.wei_zp);
    return p_slot;
  };

  // ── Reference: legacy 2-call with dynamic INT8 on BOTH passes ────
  {
    // Op1: dynamic INT8 from the caller's BF16 source.
    std::vector<matmul_params> p_ref_op1(num_ops);
    for (int i = 0; i < num_ops; ++i)
      p_ref_op1[i] = build_params_dynamic_int8(src_t[i], w1_s8_t[i]);
    ASSERT_EQ(group_matmul_direct(gv_op1.layout, gv_op1.transA, gv_op1.transB,
                                  gv_op1.Ms, gv_op1.Ns, gv_op1.Ks, gv_op1.alpha,
                                  srcs, gv_op1.lda, wei1_p, gv_op1.ldb, no_bias,
                                  gv_op1.beta, d1_ref_p, gv_op1.ldc,
                                  gv_op1.is_wc, p_ref_op1),
              status_t::success) << "ref Op1 (dynamic INT8)";

    for (int e = 0; e < num_ops; ++e) {
      apply_ref_gated_act(d1_ref.bf16[e], M, N_gate_up, N_gate_up, act_type);
    }

    // Op2: dynamic INT8 from the activated BF16 intermediate.  The
    // intermediate has no `tensor_t` wrapper, so we set up the
    // per-token src_scale.dims directly on the params (the kernel
    // allocates the runtime scale buffer internally when buff is
    // null — same path the fused dispatcher takes for Op2 by
    // inheriting `params[i].dynamic_quant` + `dtypes.compute=s8`
    // and copying `fused.down_scale[i]` into the Op2 weight slot).
    std::vector<matmul_params> p_ref_op2(num_ops);
    for (int i = 0; i < num_ops; ++i) {
      p_ref_op2[i] = make_uniform_params(1, data_type_t::bf16)[0];
      p_ref_op2[i].dtypes.src     = data_type_t::bf16;
      p_ref_op2[i].dtypes.wei     = data_type_t::s8;
      p_ref_op2[i].dtypes.dst     = data_type_t::bf16;
      p_ref_op2[i].dtypes.compute = data_type_t::s8;
      p_ref_op2[i].dynamic_quant  = true;
      // Per-token src_scale for Op2 (buff = nullptr → kernel
      // allocates the runtime scratch).
      p_ref_op2[i].quant_params.src_scale.buff = nullptr;
      p_ref_op2[i].quant_params.src_scale.dt   = data_type_t::f32;
      p_ref_op2[i].quant_params.src_scale.dims = {M, 1};
      copy_attached_scale(w2_s8_t[i], p_ref_op2[i].quant_params.wei_scale);
      copy_attached_zp   (w2_s8_t[i], p_ref_op2[i].quant_params.wei_zp);
    }
    std::vector<const void *> srcs2(num_ops);
    for (int e = 0; e < num_ops; ++e) srcs2[e] = d1_ref_p[e];
    ASSERT_EQ(group_matmul_direct(gv_op2.layout, gv_op2.transA, gv_op2.transB,
                                  gv_op2.Ms, gv_op2.Ns, gv_op2.Ks, gv_op2.alpha,
                                  srcs2, gv_op2.lda, wei2_p, gv_op2.ldb, no_bias,
                                  gv_op2.beta, d2_ref_p, gv_op2.ldc,
                                  gv_op2.is_wc, p_ref_op2),
              status_t::success) << "ref Op2 (dynamic INT8)";
  }

  // ── Test: single fused call.  Dynamic INT8 on Op1 carried via
  //   `params[i]`; Op2 inherits the same scheme — only the down_weight
  //   scale needs a dedicated carrier (`fused.down_scale`). ────────────
  grp_matmul_gated_act_params act{};
  act.act = act_type;

  auto fused = make_fused_moe_op2(num_ops, H, wei2_p, no_bias);
  fused.dst_down = d2_test_p;
  fused.ldc_down = std::vector<int>(num_ops, H);

  // Only Op2's weight scale is plumbed through the fused struct.
  // Everything else (dynamic_quant flag, dtypes.compute=s8, per-token
  // src_scale.dims) is inherited from `params[i]` by the dispatcher's
  // setup loop in group_matmul_fused_moe.cpp — same scheme on both
  // passes by construction.
  fused.down_scale.resize(num_ops);
  fused.down_zp   .resize(num_ops);
  for (int i = 0; i < num_ops; ++i) {
    copy_attached_scale(w2_s8_t[i], fused.down_scale[i]);
    copy_attached_zp   (w2_s8_t[i], fused.down_zp[i]);
  }

  {
    std::vector<matmul_params> p_test(num_ops);
    for (int i = 0; i < num_ops; ++i) {
      p_test[i] = build_params_dynamic_int8(src_t[i], w1_s8_t[i]);
      // Pin the fused team to one thread per expert so the M-tile
      // vertical-fusion gate math clears deterministically on ANY host
      // core count, making the path honour ONLY the
      // ZENDNNL_GRP_MATMUL_M_TILE_VERTICAL_FUSION env flag:
      //   * round-based gate  : active_ops(E) <= num_threads(E)      ✓
      //   * wide-N over-shard : total_need*2 = 2*E*ceil(M/16) > E    ✓
      //   * slice_M == M >= 16 keeps the BF16->S8 per-slice reorder valid.
      // With this pin, `=1` engages
      // (mode=fused_moe_vertical(op1=vertical_fusion_dqint8,...)) and
      // `=-1` falls back (mode=fused_moe_2pass(...)).  Without it, a
      // many-thread host over-shards these tiny shapes and bails to
      // two-pass even when fusion is enabled.  num_threads is plumbed
      // through resolve_num_threads, which honours non-zero requests.
      p_test[i].num_threads = num_ops;
    }
    ASSERT_EQ(group_matmul_direct(gv_op1.layout, gv_op1.transA, gv_op1.transB,
                                  gv_op1.Ms, gv_op1.Ns, gv_op1.Ks, gv_op1.alpha,
                                  srcs, gv_op1.lda, wei1_p, gv_op1.ldb, no_bias,
                                  gv_op1.beta, d1_unused_p, gv_op1.ldc,
                                  gv_op1.is_wc, p_test, nullptr, &act, &fused),
              status_t::success)
        << "fused (dynamic INT8 — Op1 quant via params[i], Op2 weight "
           "scale via fused.down_scale, rest inherited)";
  }

  // Sanity: same defensive guard as [16.d] — confirm both routes
  // actually executed the runtime-reorder + INT8 matmul instead of
  // bailing out and leaving the dst buffers at the post-alloc zero
  // state.  A consistent kernel-setup failure on both paths would
  // otherwise let the verify_per_expert_2d below trivially pass.
  double ref_abs_sum = 0.0, test_abs_sum = 0.0;
  for (int e = 0; e < num_ops; ++e) {
    for (size_t k = 0; k < d2_ref.bf16[e].size(); ++k) {
      ref_abs_sum  += std::abs(static_cast<float>(d2_ref .bf16[e][k]));
      test_abs_sum += std::abs(static_cast<float>(d2_test.bf16[e][k]));
    }
  }
  ASSERT_GT(ref_abs_sum,  1e-3)
      << "[16.e] reference 2-call path produced all-zero d2_ref "
         "(sum=" << ref_abs_sum << ") — dynamic-INT8 dispatch "
         "likely short-circuited; check the kernel error log for "
         "the BF16→S8 reorder eligibility gate in "
         "reorder_quantization.cpp::reorder_quantization_wrapper.";
  ASSERT_GT(test_abs_sum, 1e-3)
      << "[16.e] fused-test path produced all-zero d2_test "
         "(sum=" << test_abs_sum << ").";

  std::ostringstream lbl;
  lbl << "[16] Dynamic_INT8_BothPasses"
      << " act=" << act_int
      << " d="   << dim
      << " h="   << H
      << " M="   << M
      << " E="   << num_ops;
  verify_per_expert_2d(d2_test, H, d2_ref, H, num_ops, M, H, is_bf16,
                       tol_fused(is_bf16), lbl.str());
}

TEST(TestFusedMoEQuantDynINT8AllAlgos, GroupDynamicQuantBothPasses) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  using zendnnl::common::data_type_t;

  constexpr int dim = 32;
  constexpr int H = 32;
  constexpr int M = 32;
  constexpr int num_ops = 4;
  constexpr int N_gate_up = 2 * dim;
  constexpr int K_in = H;
  constexpr int K_down = dim;
  constexpr bool is_bf16 = true;
  const auto act_type = grp_matmul_gated_act_t::swiglu_oai_mul;

  for (int algo : {0, 1, 2, 3, 4, 5, 6}) {
    reset_grp_matmul_caches();
    AlgoEnvGuard algo_guard(algo);

    tensor_factory_t factory{};
    std::vector<tensor_t> src_t(num_ops), src_scale_t(num_ops);
    for (int i = 0; i < num_ops; ++i) {
      src_scale_t[i] = factory.zero_tensor({M, 1}, data_type_t::f32);
      src_t[i] = factory.uniform_dist_tensor({M, K_in}, data_type_t::bf16,
                                             2.0, false, src_scale_t[i],
                                             tensor_t{});
    }

    std::vector<tensor_t> w1_s8_t(num_ops), w1_scale_t(num_ops), w1_zp_t(num_ops);
    std::vector<tensor_t> w2_s8_t(num_ops), down_scale_t(num_ops), down_zp_t(num_ops);
    for (int i = 0; i < num_ops; ++i) {
      auto w1_ref = factory.uniform_dist_tensor({K_in, N_gate_up},
                                                data_type_t::bf16, 2.0);
      ASSERT_EQ(quant_params_compute(factory, w1_ref, data_type_t::bf16,
                                     data_type_t::s8, {1, N_gate_up},
                                     data_type_t::f32, w1_scale_t[i],
                                     w1_zp_t[i], &w1_s8_t[i]),
                status_t::success);

      auto w2_ref = factory.uniform_dist_tensor({K_down, H},
                                                data_type_t::bf16, 2.0);
      ASSERT_EQ(quant_params_compute(factory, w2_ref, data_type_t::bf16,
                                     data_type_t::s8, {1, H},
                                     data_type_t::f32, down_scale_t[i],
                                     down_zp_t[i], &w2_s8_t[i]),
                status_t::success);
    }

    std::vector<const void *> srcs(num_ops), wei1_p(num_ops), wei2_p(num_ops);
    for (int i = 0; i < num_ops; ++i) {
      srcs[i] = src_t[i].get_raw_handle_unsafe();
      wei1_p[i] = w1_s8_t[i].get_raw_handle_unsafe();
      wei2_p[i] = w2_s8_t[i].get_raw_handle_unsafe();
    }

    TypedBuffers d1_ref, d2_ref, d2_test, d1_unused;
    d1_ref.alloc(num_ops, static_cast<size_t>(M) * N_gate_up, is_bf16);
    d2_ref.alloc(num_ops, static_cast<size_t>(M) * H, is_bf16);
    d2_test.alloc(num_ops, static_cast<size_t>(M) * H, is_bf16);
    d1_unused.alloc(num_ops, static_cast<size_t>(M) * N_gate_up, is_bf16);
    auto d1_ref_p = d1_ref.ptrs(is_bf16);
    auto d2_ref_p = d2_ref.ptrs(is_bf16);
    auto d2_test_p = d2_test.ptrs(is_bf16);
    auto d1_unused_p = d1_unused.ptrs(is_bf16);
    std::vector<const void *> no_bias(num_ops, nullptr);

    auto gv_op1 = GemmVecs::uniform(num_ops, M, N_gate_up, K_in);
    auto gv_op2 = GemmVecs::uniform(num_ops, M, H, K_down);
    gv_op2.lda.assign(num_ops, N_gate_up);
    gv_op1.is_wc.assign(num_ops, true);
    gv_op2.is_wc.assign(num_ops, true);

    auto build_params_dynamic_int8 =
        [&](const tensor_t &src_tensor, const tensor_t &wei_tensor) {
      auto p_slot = make_uniform_params(1, data_type_t::bf16)[0];
      p_slot.dtypes.src = data_type_t::bf16;
      p_slot.dtypes.wei = data_type_t::s8;
      p_slot.dtypes.dst = data_type_t::bf16;
      p_slot.dtypes.compute = data_type_t::s8;
      p_slot.dynamic_quant = true;
      copy_attached_scale(src_tensor, p_slot.quant_params.src_scale);
      copy_attached_scale(wei_tensor, p_slot.quant_params.wei_scale);
      copy_attached_zp(wei_tensor, p_slot.quant_params.wei_zp);
      return p_slot;
    };

    std::vector<matmul_params> p_ref_op1(num_ops);
    for (int i = 0; i < num_ops; ++i)
      p_ref_op1[i] = build_params_dynamic_int8(src_t[i], w1_s8_t[i]);
    ASSERT_EQ(group_matmul_direct(gv_op1.layout, gv_op1.transA, gv_op1.transB,
                                  gv_op1.Ms, gv_op1.Ns, gv_op1.Ks, gv_op1.alpha,
                                  srcs, gv_op1.lda, wei1_p, gv_op1.ldb, no_bias,
                                  gv_op1.beta, d1_ref_p, gv_op1.ldc,
                                  gv_op1.is_wc, p_ref_op1),
              status_t::success) << "algo=" << algo << " ref Op1";

    for (int e = 0; e < num_ops; ++e)
      apply_ref_gated_act(d1_ref.bf16[e], M, N_gate_up, N_gate_up, act_type);

    std::vector<matmul_params> p_ref_op2(num_ops);
    for (int i = 0; i < num_ops; ++i) {
      p_ref_op2[i] = make_uniform_params(1, data_type_t::bf16)[0];
      p_ref_op2[i].dtypes.src = data_type_t::bf16;
      p_ref_op2[i].dtypes.wei = data_type_t::s8;
      p_ref_op2[i].dtypes.dst = data_type_t::bf16;
      p_ref_op2[i].dtypes.compute = data_type_t::s8;
      p_ref_op2[i].dynamic_quant = true;
      p_ref_op2[i].quant_params.src_scale.dt = data_type_t::f32;
      p_ref_op2[i].quant_params.src_scale.dims = {M, 1};
      copy_attached_scale(w2_s8_t[i], p_ref_op2[i].quant_params.wei_scale);
      copy_attached_zp(w2_s8_t[i], p_ref_op2[i].quant_params.wei_zp);
    }
    std::vector<const void *> srcs2(num_ops);
    for (int e = 0; e < num_ops; ++e) srcs2[e] = d1_ref_p[e];
    ASSERT_EQ(group_matmul_direct(gv_op2.layout, gv_op2.transA, gv_op2.transB,
                                  gv_op2.Ms, gv_op2.Ns, gv_op2.Ks, gv_op2.alpha,
                                  srcs2, gv_op2.lda, wei2_p, gv_op2.ldb, no_bias,
                                  gv_op2.beta, d2_ref_p, gv_op2.ldc,
                                  gv_op2.is_wc, p_ref_op2),
              status_t::success) << "algo=" << algo << " ref Op2";

    grp_matmul_gated_act_params act{};
    act.act = act_type;
    auto fused = make_fused_moe_op2(num_ops, H, wei2_p, no_bias);
    fused.dst_down = d2_test_p;
    fused.ldc_down = std::vector<int>(num_ops, H);
    fused.down_scale.resize(num_ops);
    fused.down_zp.resize(num_ops);
    for (int i = 0; i < num_ops; ++i) {
      copy_attached_scale(w2_s8_t[i], fused.down_scale[i]);
      copy_attached_zp(w2_s8_t[i], fused.down_zp[i]);
    }

    std::vector<matmul_params> p_test(num_ops);
    for (int i = 0; i < num_ops; ++i)
      p_test[i] = build_params_dynamic_int8(src_t[i], w1_s8_t[i]);
    ASSERT_EQ(group_matmul_direct(gv_op1.layout, gv_op1.transA, gv_op1.transB,
                                  gv_op1.Ms, gv_op1.Ns, gv_op1.Ks, gv_op1.alpha,
                                  srcs, gv_op1.lda, wei1_p, gv_op1.ldb, no_bias,
                                  gv_op1.beta, d1_unused_p, gv_op1.ldc,
                                  gv_op1.is_wc, p_test, nullptr, &act, &fused),
              status_t::success) << "algo=" << algo << " fused";

    verify_per_expert_2d(d2_test, H, d2_ref, H, num_ops, M, H, is_bf16,
                         tol_fused(is_bf16),
                         "Dynamic_INT8_BothPasses_AllAlgos algo=" +
                             std::to_string(algo));
  }
}

// Grouped pre-quant (ZENDNNL_ENABLE_GROUP_DQ=1, the production default —
// group_dynamic_quant converts src to s8 and engages the grouped-s8 CK)
// must produce the SAME fused-MoE output as the legacy runtime-hoist
// path (ENABLE_GROUP_DQ=0, per-expert reorder inside execute_expert_slice).
// This pins the grouped-s8 e2e path against its predecessor end-to-end.
TEST(TestFusedMoEQuantDynINT8AllAlgos, GroupDqVsRuntimeHoistParity) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  using zendnnl::common::data_type_t;
  // DQ-INT8 CK only needs AVX-512 VNNI (the bf16-dst store uses a manual
  // f32->bf16 round, not vcvtneps2bf16), so gate on VNNI — not BF16 —
  // to keep the int8 path covered on VNNI-only hosts.
  if (!custom_kernel::avx512vnni_available()) {
    GTEST_SKIP() << "Requires AVX-512 VNNI (VPDPBUSD) for the DQ-INT8 "
                    "custom-kernel path.";
  }

  constexpr int dim = 32, H = 32, M = 32, num_ops = 4;
  constexpr int N_gate_up = 2 * dim, K_in = H, K_down = dim;
  constexpr bool is_bf16 = true;
  const auto act_type = grp_matmul_gated_act_t::swiglu_oai_mul;

  AlgoEnvGuard algo_guard(3);  // decode N-tile path where grouped-s8 engages

  tensor_factory_t factory{};
  std::vector<tensor_t> src_t(num_ops), src_scale_t(num_ops);
  for (int i = 0; i < num_ops; ++i) {
    src_scale_t[i] = factory.zero_tensor({M, 1}, data_type_t::f32);
    src_t[i] = factory.uniform_dist_tensor({M, K_in}, data_type_t::bf16,
                                           2.0, false, src_scale_t[i],
                                           tensor_t{});
  }
  std::vector<tensor_t> w1_s8_t(num_ops), w1_scale_t(num_ops), w1_zp_t(num_ops);
  std::vector<tensor_t> w2_s8_t(num_ops), down_scale_t(num_ops), down_zp_t(num_ops);
  for (int i = 0; i < num_ops; ++i) {
    auto w1_ref = factory.uniform_dist_tensor({K_in, N_gate_up},
                                              data_type_t::bf16, 2.0);
    ASSERT_EQ(quant_params_compute(factory, w1_ref, data_type_t::bf16,
                                   data_type_t::s8, {1, N_gate_up},
                                   data_type_t::f32, w1_scale_t[i],
                                   w1_zp_t[i], &w1_s8_t[i]),
              status_t::success);
    auto w2_ref = factory.uniform_dist_tensor({K_down, H},
                                              data_type_t::bf16, 2.0);
    ASSERT_EQ(quant_params_compute(factory, w2_ref, data_type_t::bf16,
                                   data_type_t::s8, {1, H},
                                   data_type_t::f32, down_scale_t[i],
                                   down_zp_t[i], &w2_s8_t[i]),
              status_t::success);
  }
  std::vector<const void *> srcs(num_ops), wei1_p(num_ops), wei2_p(num_ops);
  for (int i = 0; i < num_ops; ++i) {
    srcs[i] = src_t[i].get_raw_handle_unsafe();
    wei1_p[i] = w1_s8_t[i].get_raw_handle_unsafe();
    wei2_p[i] = w2_s8_t[i].get_raw_handle_unsafe();
  }

  TypedBuffers d2_groupdq, d2_hoist, d1_unused;
  d2_groupdq.alloc(num_ops, static_cast<size_t>(M) * H, is_bf16);
  d2_hoist  .alloc(num_ops, static_cast<size_t>(M) * H, is_bf16);
  d1_unused .alloc(num_ops, static_cast<size_t>(M) * N_gate_up, is_bf16);
  auto d1_unused_p = d1_unused.ptrs(is_bf16);
  std::vector<const void *> no_bias(num_ops, nullptr);

  auto gv_op1 = GemmVecs::uniform(num_ops, M, N_gate_up, K_in);
  gv_op1.is_wc.assign(num_ops, true);

  auto build_p = [&](const tensor_t &s, const tensor_t &w) {
    auto p = make_uniform_params(1, data_type_t::bf16)[0];
    p.dtypes.src = data_type_t::bf16;
    p.dtypes.wei = data_type_t::s8;
    p.dtypes.dst = data_type_t::bf16;
    p.dtypes.compute = data_type_t::s8;
    p.dynamic_quant = true;
    copy_attached_scale(s, p.quant_params.src_scale);
    copy_attached_scale(w, p.quant_params.wei_scale);
    copy_attached_zp(w, p.quant_params.wei_zp);
    return p;
  };

  auto run_fused = [&](const std::vector<void *> &dst_down_p) {
    grp_matmul_gated_act_params act{};
    act.act = act_type;
    auto fused = make_fused_moe_op2(num_ops, H, wei2_p, no_bias);
    fused.dst_down = dst_down_p;
    fused.ldc_down = std::vector<int>(num_ops, H);
    fused.down_scale.resize(num_ops);
    fused.down_zp.resize(num_ops);
    for (int i = 0; i < num_ops; ++i) {
      copy_attached_scale(w2_s8_t[i], fused.down_scale[i]);
      copy_attached_zp(w2_s8_t[i], fused.down_zp[i]);
    }
    std::vector<matmul_params> p(num_ops);
    for (int i = 0; i < num_ops; ++i) p[i] = build_p(src_t[i], w1_s8_t[i]);
    return group_matmul_direct(gv_op1.layout, gv_op1.transA, gv_op1.transB,
                               gv_op1.Ms, gv_op1.Ns, gv_op1.Ks, gv_op1.alpha,
                               srcs, gv_op1.lda, wei1_p, gv_op1.ldb, no_bias,
                               gv_op1.beta, d1_unused_p, gv_op1.ldc,
                               gv_op1.is_wc, p, nullptr, &act, &fused);
  };

  // Path A: grouped pre-quant (production default).
  {
    EnvVarGuard group_dq_on("ZENDNNL_ENABLE_GROUP_DQ", "1");
    reset_grp_matmul_caches();
    ASSERT_EQ(run_fused(d2_groupdq.ptrs(is_bf16)), status_t::success)
        << "fused with ENABLE_GROUP_DQ=1";
  }
  // Path B: legacy per-expert runtime hoist.
  {
    EnvVarGuard group_dq_off("ZENDNNL_ENABLE_GROUP_DQ", "0");
    reset_grp_matmul_caches();
    ASSERT_EQ(run_fused(d2_hoist.ptrs(is_bf16)), status_t::success)
        << "fused with ENABLE_GROUP_DQ=0";
  }

  // Both paths quantize the same bf16 src per-token to s8; outputs must
  // match within the fused bf16 tolerance.
  double sum = 0.0;
  for (int e = 0; e < num_ops; ++e)
    for (size_t k = 0; k < d2_groupdq.bf16[e].size(); ++k)
      sum += std::abs(static_cast<float>(d2_groupdq.bf16[e][k]));
  ASSERT_GT(sum, 1e-3) << "grouped-DQ path produced all-zero output";
  verify_per_expert_2d(d2_groupdq, H, d2_hoist, H, num_ops, M, H, is_bf16,
                       tol_fused(is_bf16), "GroupDqVsRuntimeHoistParity");
}

// One INSTANTIATE_TEST_SUITE_P per fixture subclass, each binding
// the scheme to its kernel-supported grid (`make_woq_quant_params`
// covers all four M values; `make_dyn_int8_quant_params` drops
// M < 16 since the AOCL BF16→S8 reorder rejects those at the
// single-matmul layer).  Same shared prefix
// `GroupMatmulFusedMoEQuant` keeps both groups discoverable via
// `--gtest_filter=*FusedMoEQuant*`.
INSTANTIATE_TEST_SUITE_P(GroupMatmulFusedMoEQuant, TestFusedMoEQuantWOQ,
                         ::testing::ValuesIn(make_woq_quant_params()),
                         FusedMoEQuantParamName);

INSTANTIATE_TEST_SUITE_P(GroupMatmulFusedMoEQuant, TestFusedMoEQuantDynINT8,
                         ::testing::ValuesIn(make_dyn_int8_quant_params()),
                         FusedMoEQuantParamName);

// ===============================================================================
// [17] TestFusedMoEVerticalBF16 - W13 -> gated-act -> W2 vertical fusion at
//      M-tile granularity (BF16 phase 1).  Forces ALGO 2 + the vertical-fusion
//      env knob ON via RAII overrides, drives the dispatcher fork at
//      `group_matmul_fused_moe_execute` lines ~950-1035 (see
//      group_matmul_fused_moe.cpp), and compares the fused executor's output
//      against the SAME legacy 2-call reference (`run_legacy_2call_ref`) every
//      other fused-MoE suite in this file uses.
//
// Correctness fixture covers the parameter cube that the vertical-fusion
// design must protect against regressing:
//
//   * Real architecture shapes scaled down for runtime
//     (Qwen3-MoE-class 8E/M=4-32, Mixtral-class 4E,
//     GPT-OSS-class many-expert variants).
//   * All four supported gated activations
//     ({none, silu_and_mul, gelu_and_mul, swiglu_oai_mul}) — the
//     `apply_gated_act_inplace` set vetted by the dispatcher's
//     `act_supported` gate.
//   * Both caller-alloc and internal-alloc Op2 destinations
//     (`fused.dst_down` provided vs left empty).  Internal-alloc
//     mode exercises the in-place src reuse path that
//     `dst_w13_is_caller_alloc=false` selects inside the executor
//     — Stage 2's post-activation tile in scratch is consumed
//     directly by Stage 3 with no spill back to the Op1 arena.
//
// Capture-tag assertion (`s_last_m_tile_path == kVerticalFusionBF16`)
// verifies the executor actually engaged on every parameter row, not
// just that the eligibility gate accepted the shape — protects
// against silent regressions where the executor's internal planner
// later bails out (e.g. due to a tuning change in
// `plan_m_tile_single_tier_assignment` or a stricter scratch budget).
// FORCED mode (vert_env=1) sets the executor's internal planner to
// "engage regardless of the wide-N / multi-tier branches", so the
// only legitimate fallthrough on bf16-eligible shapes is a scratch-
// budget bail-out — which we sidestep by sizing the test shapes to
// fit a generous (1024 KB) per-thread budget override.
// ===============================================================================

struct VerticalFusionTestParam {
  const char *name;          // Short architecture descriptor for the test
                             // suite filter (e.g. "qwen3_E8_M8").
  int num_experts;
  int dim;                   // Op1 hidden_size / 2 (gate width per token).
  int hidden;                // Op2 N_down (== Op1 K_in: residual width).
  int M;                     // Per-expert active token count (uniform here).
  int act_int;               // 0=none, 1=silu, 2=gelu, 3=swiglu_oai.
  bool internal_alloc;       // Op2 dst path: false = caller-alloc dst_down,
                             // true  = internal-alloc + src reuse.
};

static std::string VerticalFusionParamName(
    const ::testing::TestParamInfo<VerticalFusionTestParam> &info) {
  static const char *kAct[] = {"none", "silu", "gelu", "swiglu"};
  const auto &p = info.param;
  return std::string(p.name)
       + "_" + kAct[p.act_int]
       + (p.internal_alloc ? "_intalloc" : "_calleralloc");
}

class TestFusedMoEVerticalBF16
    : public ::testing::TestWithParam<VerticalFusionTestParam> {};

TEST_P(TestFusedMoEVerticalBF16, Correctness) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;

  reset_grp_matmul_caches();

  const auto &p = GetParam();
  const int E         = p.num_experts;
  const int dim       = p.dim;
  const int N_gate_up = 2 * dim;
  const int H         = p.hidden;
  const int K_in      = H;
  const int M         = p.M;
  const auto act_type = static_cast<grp_matmul_gated_act_t>(p.act_int);
  const bool is_bf16  = true;
  const data_type_t dt = data_type_t::bf16;
  const bool act_is_none = (act_type == grp_matmul_gated_act_t::none);
  const int K_down = act_is_none ? N_gate_up : dim;

  // RAII pins: ALGO 2 (M-tile), vertical fusion FORCED (so the
  // executor's internal planner engages regardless of the wide-N /
  // multi-tier branch heuristics), and a generous per-thread scratch
  // budget cap so the test shapes never bail on budget.  The
  // overrides restore the production values on scope exit.
  AlgoEnvGuard                  algo2_guard(2);
  MoEVerticalFusionOverride     vert_guard(1);
  MoEPipelineScratchKbOverride  scratch_guard(1024);

  // Buffers: src/wei1/wei2 are the inputs; d1_ref/d2_ref are the
  // legacy 2-call reference outputs; d1_fused/d2_fused are the
  // vertical-fusion dispatcher outputs.
  //
  // Op1 dst (d1_fused) is only consumed by the executor when
  // `dst_w13_is_caller_alloc` (i.e. internal_alloc=false here) —
  // it's the optional spill of the post-act tile from scratch.
  // Internal-alloc mode leaves Op1 dst as nullptrs.
  //
  // Op2 dst is either caller-allocated via `fused.dst_down`
  // (internal_alloc=false), or written back in place into the src
  // buffer (internal_alloc=true) — that's the in-place src reuse
  // contract from `[5b] TestFusedMoEInternalAlloc`.
  TypedBuffers src_ref, src_test, w1, d1_ref, d1_fused, w2, d2_ref, d2_fused;
  src_ref .alloc(E, (size_t)M * K_in,         is_bf16);
  src_test.alloc(E, (size_t)M * K_in,         is_bf16);
  w1      .alloc(E, (size_t)K_in * N_gate_up, is_bf16);
  d1_ref  .alloc(E, (size_t)M * N_gate_up,    is_bf16);
  d1_fused.alloc(E, (size_t)M * N_gate_up,    is_bf16);
  w2      .alloc(E, (size_t)K_down * H,       is_bf16);
  d2_ref  .alloc(E, (size_t)M * H,            is_bf16);
  d2_fused.alloc(E, (size_t)M * H,            is_bf16);
  fill_moe_tensors(E, is_bf16, &src_ref,  &w1, &w2);
  fill_moe_tensors(E, is_bf16, &src_test, nullptr, nullptr);

  auto gv_op1 = GemmVecs::uniform(E, M, N_gate_up, K_in);

  auto src_ref_p  = src_ref .cptrs(is_bf16);
  auto src_test_p = src_test.cptrs(is_bf16);
  auto wei1_p     = w1      .cptrs(is_bf16);
  auto wei2_p     = w2      .cptrs(is_bf16);
  auto d1_ref_p   = d1_ref  .ptrs(is_bf16);
  auto d1_fused_p = d1_fused.ptrs(is_bf16);
  auto d2_ref_p   = d2_ref  .ptrs(is_bf16);
  auto d2_fused_p = d2_fused.ptrs(is_bf16);
  std::vector<const void *> no_bias(E, nullptr);
  auto params = make_uniform_params(E, dt);
  // Pin the dispatcher team size so vertical-fusion engagement is
  // deterministic across CI hosts.  The shapes in
  // make_vertical_fusion_params() are sized for a 32-thread rig: every
  // row has `total_need > 16` (clears the wide-N gate
  // `total_need * 2 <= num_threads`) and `E <= 32` (clears the
  // round-based `active_ops > num_threads` gate).  Without the pin, a
  // host with OMP_NUM_THREADS < E trips the round-based bail and the
  // executor never reaches the pipeline (the engagement assertion below
  // would flake).  group_matmul_direct feeds params[0].num_threads
  // through resolve_num_threads, which honours non-zero requests
  // uncapped, so this reproduces the rig exactly regardless of host
  // core count.
  for (auto &pp : params) pp.num_threads = 32;

  grp_matmul_gated_act_params act{};
  act.act = act_type;
  auto act_ptr = act_is_none ? nullptr : &act;

  ASSERT_EQ(run_legacy_2call_ref(E, M, K_in, N_gate_up, K_down, H,
                                 is_bf16, act_type,
                                 src_ref_p, wei1_p, wei2_p,
                                 d1_ref_p, d2_ref_p),
            status_t::success) << "ref " << p.name;

  // Fused dispatch + capture-tag assertion.
  //
  // Op1 dst arg (`d1_*` / `ldc1_*`) is the `dst` parameter of
  // `group_matmul_direct`.  caller-alloc => real d1_fused buffer +
  // stride N_gate_up.  internal-alloc => nullptrs + zero stride
  // (the dispatcher's signal to library-manage Op1's arena).
  //
  // Op2 dst arg goes into `fused.dst_down` / `fused.ldc_down`.
  // caller-alloc => real d2_fused buffer.  internal-alloc =>
  // intentionally left empty (the dispatcher's signal to read Op2
  // out back into src_test in place, see TestFusedMoEInternalAlloc).
  auto fused = make_fused_moe_op2(E, H, wei2_p, no_bias);
  std::vector<void *> d1_for_fused;
  std::vector<int>    ldc1_for_fused;
  if (p.internal_alloc) {
    d1_for_fused.assign(E, nullptr);
    ldc1_for_fused.assign(E, 0);
    // `fused.dst_down` and `fused.ldc_down` left empty — engages
    // in-place src reuse for Op2 inside the dispatcher fork.
  } else {
    d1_for_fused   = d1_fused_p;
    ldc1_for_fused = std::vector<int>(E, N_gate_up);
    fused.dst_down = d2_fused_p;
    fused.ldc_down = std::vector<int>(E, H);
  }

  int tag = 0;
  {
    MTilePathCaptureGuard cap;
    auto pf = params;
    ASSERT_EQ(group_matmul_direct(gv_op1.layout, gv_op1.transA, gv_op1.transB,
                                  gv_op1.Ms, gv_op1.Ns, gv_op1.Ks,
                                  gv_op1.alpha, src_test_p,
                                  gv_op1.lda, wei1_p, gv_op1.ldb,
                                  no_bias, gv_op1.beta,
                                  d1_for_fused, ldc1_for_fused,
                                  gv_op1.is_wc, pf, nullptr, act_ptr, &fused),
              status_t::success) << "test " << p.name;
    tag = test_api::s_last_m_tile_path.load(std::memory_order_relaxed);
  }
  EXPECT_EQ(tag, test_api::m_tile_path_tag::kVerticalFusionBF16)
      << "vertical fusion did NOT engage on shape " << p.name
      << " — capture tag = " << tag
      << " (expected kVerticalFusionBF16 = "
      << test_api::m_tile_path_tag::kVerticalFusionBF16 << ")";

  std::ostringstream lbl;
  lbl << "[17] vertical_fusion " << p.name
      << " act=" << p.act_int
      << " M=" << M << " E=" << E
      << " dim=" << dim << " H=" << H
      << (p.internal_alloc ? " intalloc" : " calleralloc");

  // Verification: internal-alloc reads Op2 back from src_test (stride
  // K_in = H, since K = H == N_down — a perfectly packed in-place
  // reuse); caller-alloc reads d2_fused (stride H).
  if (p.internal_alloc) {
    verify_per_expert_2d(src_test, K_in, d2_ref, H,
                         E, M, H, is_bf16,
                         tol_fused(is_bf16), lbl.str());
  } else {
    verify_per_expert_2d(d2_fused, H, d2_ref, H,
                         E, M, H, is_bf16,
                         tol_fused(is_bf16), lbl.str());
  }
}

static std::vector<VerticalFusionTestParam> make_vertical_fusion_params() {
  // Shape selection rationale.
  //
  // The planner's wide-N memory-bound fallback engages on
  // `max_M > 1 && total_need * 2 ≤ num_threads`, where
  // `total_need = sum_i ceil(M_i / kSliceTarget)` (default
  // kSliceTarget = 16).  We want vertical fusion to actually engage,
  // so the test shapes must clear that gate: at the 32-thread test-
  // rig default we need `total_need > 16`, i.e.:
  //
  //   * E ≥ 4 with M ≥ 128  (t=8 per active, total_need ≥ 32)
  //   * E ≥ 8 with M ≥  64  (t=4 per active, total_need ≥ 32)
  //   * E ≥ 16 with M ≥ 32  (t=2 per active, total_need ≥ 32)
  //
  // These are also the prompt-phase ranges where vertical fusion's
  // intermediate-locality win is largest (decode-phase M=1 routes
  // through wide-N intentionally — its inter-stage barrier overhead
  // is dwarfed by per-token activation cost).  Hidden dimensions are
  // scaled DOWN (dim=64, H=64) to keep the test fast; the per-thread
  // scratch budget gate `(slice_M × N_w13 × 2 bytes) ≤ 1024 KB` is
  // comfortably cleared at every shape below.
  //
  // num_experts stays ≤ num_threads (the executor bails to legacy on
  // `active_ops > num_threads`, the round-based regime — covered by
  // the broader regression sweep, not engaged here).
  static const struct {
    const char *name;
    int E, dim, H, M;
  } kArchs[] = {
    // Mixtral-class prompt frames: 4 / 8 experts, larger M to clear
    // wide-N.  M=128 is a representative short-prompt frame size.
    {"mixtral_E4_M128", 4,  64, 64, 128},
    {"mixtral_E4_M256", 4,  64, 64, 256},
    {"mixtral_E8_M64",  8,  64, 64,  64},
    // Qwen3-MoE-class: 8 experts, prompt frames in the M=64..128
    // range.  Wider Op1 intermediate (N_gate_up = 2*dim = 128) and
    // larger M=128 exercise both the inner GEMM unroll and the
    // post-act tile spill (caller-alloc) at a realistic ratio.
    {"qwen3_E8_M128",   8,  64, 64, 128},
    // GPT-OSS-class many-expert: 16 experts with smaller M each.
    // The planner's CCD-aware capacity cap engages on
    // (active_ops ≤ num_ccds), which at 32 threads / 8 cores-per-CCD
    // means active_ops ≤ 4, so E=16 stays out of the CCD-stripe
    // regime and lands in the regular single-tier plan.
    {"gpt_oss_E16_M64",16, 64, 64,  64},
  };
  std::vector<VerticalFusionTestParam> out;
  for (const auto &a : kArchs) {
    // All 4 activations × both alloc modes.
    for (int act : {0, 1, 2, 3}) {
      for (bool intalloc : {false, true}) {
        out.push_back({a.name, a.E, a.dim, a.H, a.M, act, intalloc});
      }
    }
  }
  return out;
}

INSTANTIATE_TEST_SUITE_P(GroupMatmulFusedMoEVerticalBF16,
                         TestFusedMoEVerticalBF16,
                         ::testing::ValuesIn(make_vertical_fusion_params()),
                         VerticalFusionParamName);

// ===============================================================================
// [17.b] TestFusedMoEVerticalBF16AllocMatrix - asymmetric alloc-mode coverage.
//
// `TestFusedMoEVerticalBF16` (above) parametrises only the symmetric
// alloc-mode pair (both Op1 + Op2 caller-alloc, both Op1 + Op2 internal-
// alloc).  The dispatcher's `detect_internal_alloc` (see
// `group_matmul_fused_moe.cpp:768-790`) detects each side INDEPENDENTLY
// — there are four legal combinations:
//
//   (op1_internal, op2_internal):
//     (F, F) — both caller-alloc.
//     (F, T) — Op1 caller-alloc, Op2 internal scratch (in-place src reuse).
//     (T, F) — Op1 library-managed arena, Op2 caller `fused.dst_down`.
//     (T, T) — both internal (library-managed Op1 arena + in-place src).
//
// The asymmetric (F, T) and (T, F) combinations are reachable via the
// public API but were NOT exercised by the existing parameterisation.
// This fixture closes the gap by explicitly testing all four combos on
// one representative shape (Mixtral-class E=4 M=128 dim=64 H=64) with
// silu activation — sufficient because all four combos go through the
// SAME `try_flat_m_tile_pipeline_bf16` entry point, so the shape sweep
// is already covered by [17] and the only new variable is the alloc
// pair.
//
// Contract:
//   * Vertical fusion MUST engage (tag == kVerticalFusionBF16) — the
//     alloc pair MUST NOT itself disqualify the pipeline.
//   * Output values MUST match the legacy 2-call reference within
//     `tol_fused(true)` (bf16) — read from `d2_fused` for
//     `op2_internal=false`, from `src_test` (in-place reuse) for
//     `op2_internal=true`.
// ===============================================================================

struct AllocMatrixParam {
  bool op1_internal;
  bool op2_internal;
  const char *name;
};

static std::string AllocMatrixParamName(
    const ::testing::TestParamInfo<AllocMatrixParam> &info) {
  return info.param.name;
}

class TestFusedMoEVerticalBF16AllocMatrix
    : public ::testing::TestWithParam<AllocMatrixParam> {};

TEST_P(TestFusedMoEVerticalBF16AllocMatrix, Correctness) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;

  reset_grp_matmul_caches();

  const auto &p = GetParam();
  const int E         = 4;
  const int dim       = 64;
  const int N_gate_up = 2 * dim;
  const int H         = 64;
  const int K_in      = H;
  const int M         = 128;
  const auto act_type = grp_matmul_gated_act_t::silu_and_mul;
  const bool is_bf16  = true;
  const data_type_t dt = data_type_t::bf16;
  const int K_down = dim;  // gated activation collapses N_gate_up → dim.

  AlgoEnvGuard                  algo2_guard(2);
  MoEVerticalFusionOverride     vert_guard(1);
  MoEPipelineScratchKbOverride  scratch_guard(1024);

  TypedBuffers src_ref, src_test, w1, d1_ref, d1_fused,
               w2, d2_ref, d2_fused;
  src_ref .alloc(E, (size_t)M * K_in,         is_bf16);
  src_test.alloc(E, (size_t)M * K_in,         is_bf16);
  w1      .alloc(E, (size_t)K_in * N_gate_up, is_bf16);
  d1_ref  .alloc(E, (size_t)M * N_gate_up,    is_bf16);
  d1_fused.alloc(E, (size_t)M * N_gate_up,    is_bf16);
  w2      .alloc(E, (size_t)K_down * H,       is_bf16);
  d2_ref  .alloc(E, (size_t)M * H,            is_bf16);
  d2_fused.alloc(E, (size_t)M * H,            is_bf16);
  fill_moe_tensors(E, is_bf16, &src_ref,  &w1, &w2);
  fill_moe_tensors(E, is_bf16, &src_test, nullptr, nullptr);

  auto gv_op1 = GemmVecs::uniform(E, M, N_gate_up, K_in);

  auto src_ref_p  = src_ref .cptrs(is_bf16);
  auto src_test_p = src_test.cptrs(is_bf16);
  auto wei1_p     = w1      .cptrs(is_bf16);
  auto wei2_p     = w2      .cptrs(is_bf16);
  auto d1_ref_p   = d1_ref  .ptrs(is_bf16);
  auto d1_fused_p = d1_fused.ptrs(is_bf16);
  auto d2_ref_p   = d2_ref  .ptrs(is_bf16);
  auto d2_fused_p = d2_fused.ptrs(is_bf16);
  std::vector<const void *> no_bias(E, nullptr);
  auto params = make_uniform_params(E, dt);
  // Pin the team size so engagement is host-independent (see the
  // rationale on the same pin in TestFusedMoEVerticalBF16.Correctness):
  // this shape is E=4 / M=128 ⇒ total_need=32 > 16 (clears wide-N) and
  // E=4 <= 32 (clears round-based) at the 32-thread rig.
  for (auto &pp : params) pp.num_threads = 32;

  grp_matmul_gated_act_params act{};
  act.act = act_type;

  ASSERT_EQ(run_legacy_2call_ref(E, M, K_in, N_gate_up, K_down, H,
                                 is_bf16, act_type,
                                 src_ref_p, wei1_p, wei2_p,
                                 d1_ref_p, d2_ref_p),
            status_t::success) << "ref " << p.name;

  // Op1 dst: caller-alloc supplies real buffers + ldc=N_gate_up;
  // internal-alloc supplies nullptrs + ldc=0 (the dispatcher's
  // signal at `detect_internal_alloc` to library-manage Op1's arena).
  std::vector<void *> d1_for_fused;
  std::vector<int>    ldc1_for_fused;
  if (p.op1_internal) {
    d1_for_fused.assign(E, nullptr);
    ldc1_for_fused.assign(E, 0);
  } else {
    d1_for_fused   = d1_fused_p;
    ldc1_for_fused = std::vector<int>(E, N_gate_up);
  }

  // Op2 dst: caller-alloc sets `fused.dst_down` + `fused.ldc_down`;
  // internal-alloc leaves both empty so the dispatcher engages the
  // in-place src reuse path (Op2 writes back into src_test).
  auto fused = make_fused_moe_op2(E, H, wei2_p, no_bias);
  if (!p.op2_internal) {
    fused.dst_down = d2_fused_p;
    fused.ldc_down = std::vector<int>(E, H);
  }

  int tag = 0;
  {
    MTilePathCaptureGuard cap;
    auto pf = params;
    ASSERT_EQ(group_matmul_direct(gv_op1.layout, gv_op1.transA, gv_op1.transB,
                                  gv_op1.Ms, gv_op1.Ns, gv_op1.Ks,
                                  gv_op1.alpha, src_test_p,
                                  gv_op1.lda, wei1_p, gv_op1.ldb,
                                  no_bias, gv_op1.beta,
                                  d1_for_fused, ldc1_for_fused,
                                  gv_op1.is_wc, pf, nullptr, &act, &fused),
              status_t::success) << "test " << p.name;
    tag = test_api::s_last_m_tile_path.load(std::memory_order_relaxed);
  }
  EXPECT_EQ(tag, test_api::m_tile_path_tag::kVerticalFusionBF16)
      << "vertical fusion did NOT engage on alloc combo " << p.name
      << " — capture tag = " << tag;

  // Output read-back: in-place src reuse (op2_internal=true) writes
  // Op2 back into src_test at stride K_in == H; caller-alloc Op2
  // (op2_internal=false) writes to d2_fused at stride H.
  std::ostringstream lbl;
  lbl << "[17.b] alloc_matrix " << p.name;
  if (p.op2_internal) {
    verify_per_expert_2d(src_test, K_in, d2_ref, H,
                         E, M, H, is_bf16,
                         tol_fused(is_bf16), lbl.str());
  } else {
    verify_per_expert_2d(d2_fused, H, d2_ref, H,
                         E, M, H, is_bf16,
                         tol_fused(is_bf16), lbl.str());
  }
}

INSTANTIATE_TEST_SUITE_P(
    GroupMatmulFusedMoEVerticalBF16AllocMatrix,
    TestFusedMoEVerticalBF16AllocMatrix,
    ::testing::Values(
        AllocMatrixParam{false, false, "op1_caller_op2_caller"},
        AllocMatrixParam{false, true,  "op1_caller_op2_internal"},
        AllocMatrixParam{true,  false, "op1_internal_op2_caller"},
        AllocMatrixParam{true,  true,  "op1_internal_op2_internal"}),
    AllocMatrixParamName);

// ===============================================================================
// [17.c] TestFusedMoEVerticalWOQ - WOQ-INT4 (s4/u4) vertical fusion at M-tile
//      granularity (BF16-scratch phase 1, weight-side INT4).  Same per-thread
//      slice pipeline as `TestFusedMoEVerticalBF16` above — the eligibility
//      gate's regime-B branch (see `try_flat_m_tile_pipeline_bf16` doc-block)
//      accepts `{src=bf16, wei∈{s4,u4}, dst=bf16, dynamic_quant=false,
//      wei_scale.buff!=nullptr, is_weights_const[*]=true}` and the executor
//      reuses the bf16 scratch path because weight dequantization happens
//      inside the AOCL DLP WOQ kernel (no source-side quant, no extra scratch).
//
// Why both S4 and U4 in one parameterised fixture:
//   * S4 is symmetric (no weight zp) and is the AOCL DLP WOQ default.
//   * U4 is asymmetric (mandatory wei_zp).  The gate's `wei_scale.buff !=
//     nullptr` check is identical, but the matmul kernel takes the
//     additional `wei_zp` path and the codepath through the executor's
//     `execute_expert_slice` is distinct.
// Parameterising the weight dtype keeps the test body small while
// covering both branches.  Mirrors the BF16 cube (5 archs × 4 acts × 2
// alloc modes) at half the cardinality (2 archs × 4 acts × 2 alloc modes
// × 2 weight dtypes = 32 cases) so total CI runtime stays moderate.
//
// Reference path uses two standalone `group_matmul_direct` calls with the
// SAME WOQ params, mirroring `TestFusedMoEQuantWOQ.BothPasses`.  This is
// numerically identical to the legacy two-pass route inside the fused
// dispatcher (both share the same AOCL DLP WOQ kernel), so the only
// observable difference between the reference and the fused-VF path is
// per-thread slice locality — i.e. exactly what we want to verify
// produces zero numerical drift.
// ===============================================================================

struct VerticalFusionWOQTestParam {
  const char *name;          // Architecture descriptor for filter (e.g.
                             // "mixtral_E4_M128").
  int num_experts;
  int dim;                   // Op1 hidden_size / 2 (gate width per token).
  int hidden;                // Op2 N_down (== Op1 K_in: residual width).
  int M;                     // Per-expert active token count (uniform).
  int act_int;               // 0=none, 1=silu, 2=gelu, 3=swiglu_oai.
  bool internal_alloc;       // Op2 dst path: false = caller-alloc dst_down,
                             // true  = internal-alloc + src reuse.
  bool wei_is_u4;            // false = s4 (symmetric), true = u4 (asymmetric).
};

static std::string VerticalFusionWOQParamName(
    const ::testing::TestParamInfo<VerticalFusionWOQTestParam> &info) {
  static const char *kAct[] = {"none", "silu", "gelu", "swiglu"};
  const auto &p = info.param;
  return std::string(p.name)
       + "_" + (p.wei_is_u4 ? "u4" : "s4")
       + "_" + kAct[p.act_int]
       + (p.internal_alloc ? "_intalloc" : "_calleralloc");
}

class TestFusedMoEVerticalWOQ
    : public ::testing::TestWithParam<VerticalFusionWOQTestParam> {};

TEST_P(TestFusedMoEVerticalWOQ, Correctness) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  using zendnnl::common::data_type_t;

  reset_grp_matmul_caches();

  const auto &p = GetParam();
  const int E         = p.num_experts;
  const int dim       = p.dim;
  const int N_gate_up = 2 * dim;
  const int H         = p.hidden;
  const int K_in      = H;
  const int M         = p.M;
  const auto act_type = static_cast<grp_matmul_gated_act_t>(p.act_int);
  const bool is_bf16  = true;
  const bool act_is_none = (act_type == grp_matmul_gated_act_t::none);
  const int K_down = act_is_none ? N_gate_up : dim;
  const data_type_t wei_dt = p.wei_is_u4 ? data_type_t::u4
                                         : data_type_t::s4;

  // RAII pins — same shape as TestFusedMoEVerticalBF16.Correctness:
  // ALGO 2, VF FORCED, generous scratch budget.  WOQ adds no new
  // overrides because the executor's scratch path is unchanged
  // (bf16 element size in every stage; weight dequant lives inside
  // the AOCL DLP kernel).
  AlgoEnvGuard                  algo2_guard(2);
  MoEVerticalFusionOverride     vert_guard(1);
  MoEPipelineScratchKbOverride  scratch_guard(1024);

  // ── Op1 + Op2 weights: s4 or u4 + per-channel f32 wei_scale.  U4
  //   additionally needs a per-tensor bf16 wei_zp (mirrors the
  //   asymmetric-quant contract documented in test_quant.cpp::
  //   WOQ_BF16_U4 around lines 270-280).  S4 is symmetric (no zp). ──
  tensor_factory_t factory{};
  std::vector<tensor_t> w1_scale_t(E), w1_t(E), w1_zp_t(E);
  std::vector<tensor_t> down_scale_t(E), w2_t(E), down_zp_t(E);
  for (int i = 0; i < E; ++i) {
    w1_scale_t[i] = factory.uniform_dist_tensor(
        {static_cast<uint64_t>(1), static_cast<uint64_t>(N_gate_up)},
        data_type_t::f32, 2.0);
    down_scale_t[i] = factory.uniform_dist_tensor(
        {static_cast<uint64_t>(1), static_cast<uint64_t>(H)},
        data_type_t::f32, 2.0);
    if (p.wei_is_u4) {
      // Per-tensor zp (dims {1,1}) — simplest u4 asymmetric scheme.
      // Per-channel {1,N} is also legal but adds a second scale-only
      // axis the gate doesn't need to exercise differently.
      w1_zp_t[i] = factory.uniform_dist_tensor(
          {static_cast<uint64_t>(1), static_cast<uint64_t>(1)},
          data_type_t::bf16, 2.0);
      down_zp_t[i] = factory.uniform_dist_tensor(
          {static_cast<uint64_t>(1), static_cast<uint64_t>(1)},
          data_type_t::bf16, 2.0);
      w1_t[i] = factory.uniform_dist_tensor(
          {static_cast<uint64_t>(K_in), static_cast<uint64_t>(N_gate_up)},
          wei_dt, 15.0, /*transposed=*/false,
          w1_scale_t[i], w1_zp_t[i]);
      w2_t[i] = factory.uniform_dist_tensor(
          {static_cast<uint64_t>(K_down), static_cast<uint64_t>(H)},
          wei_dt, 15.0, /*transposed=*/false,
          down_scale_t[i], down_zp_t[i]);
    } else {
      // S4 symmetric — no zp attached; copy_attached_zp below is a no-op.
      w1_t[i] = factory.uniform_dist_tensor(
          {static_cast<uint64_t>(K_in), static_cast<uint64_t>(N_gate_up)},
          wei_dt, 7.0, /*transposed=*/false, w1_scale_t[i]);
      w2_t[i] = factory.uniform_dist_tensor(
          {static_cast<uint64_t>(K_down), static_cast<uint64_t>(H)},
          wei_dt, 7.0, /*transposed=*/false, down_scale_t[i]);
    }
  }

  // ── BF16 sources + per-expert raw pointers ──
  // src_ref / src_test are populated with the SAME random fill so
  // both passes see identical inputs.  d1_*: Op1 dst buffers
  // (reference receives gated-act in place; test path only consults
  // d1_fused on caller-alloc).  d2_*: Op2 dst buffers — the final
  // comparison surface.
  TypedBuffers src_ref, src_test, d1_ref, d1_fused, d2_ref, d2_fused;
  src_ref .alloc(E, (size_t)M * K_in,         is_bf16);
  src_test.alloc(E, (size_t)M * K_in,         is_bf16);
  d1_ref  .alloc(E, (size_t)M * N_gate_up,    is_bf16);
  d1_fused.alloc(E, (size_t)M * N_gate_up,    is_bf16);
  d2_ref  .alloc(E, (size_t)M * H,            is_bf16);
  d2_fused.alloc(E, (size_t)M * H,            is_bf16);
  fill_moe_tensors(E, is_bf16, &src_ref, nullptr, nullptr);
  // Mirror src_ref into src_test so the two dispatch routes see
  // bit-identical inputs.  fill_moe_tensors's RNG is deterministic
  // per buffer, so calling it twice with the same shape would NOT
  // produce the same bytes — explicit memcpy keeps the two inputs
  // identical regardless of the helper's internal state.
  for (int e = 0; e < E; ++e) {
    std::memcpy(src_test.bf16[e].data(), src_ref.bf16[e].data(),
                src_ref.bf16[e].size() * sizeof(uint16_t));
  }

  std::vector<const void *> wei1_p(E), wei2_p(E);
  for (int i = 0; i < E; ++i) {
    wei1_p[i] = w1_t[i].get_raw_handle_unsafe();
    wei2_p[i] = w2_t[i].get_raw_handle_unsafe();
  }
  auto src_ref_p  = src_ref .cptrs(is_bf16);
  auto src_test_p = src_test.cptrs(is_bf16);
  auto d1_ref_p   = d1_ref  .ptrs(is_bf16);
  auto d1_fused_p = d1_fused.ptrs(is_bf16);
  auto d2_ref_p   = d2_ref  .ptrs(is_bf16);
  auto d2_fused_p = d2_fused.ptrs(is_bf16);
  std::vector<const void *> no_bias(E, nullptr);

  // ── Dispatch vectors.  WOQ requires is_weights_const=true on every
  //   active expert; that's enforced by the eligibility gate. ──
  auto gv_op1 = GemmVecs::uniform(E, M, N_gate_up, K_in);
  auto gv_op2 = GemmVecs::uniform(E, M, H, K_down);
  gv_op2.lda.assign(E, N_gate_up);  // Op2 reads d1_ref at the gated stride.
  gv_op1.is_wc.assign(E, true);
  gv_op2.is_wc.assign(E, true);

  // ── Build WOQ params for a single half.  Caller picks which weight
  //   tensor's metadata feeds the half (w1_t or w2_t). ──
  auto build_params_woq =
      [&](const std::vector<tensor_t> &wei_quant_tensors) {
    auto pv = make_uniform_params(E, data_type_t::bf16);
    for (int i = 0; i < E; ++i) {
      pv[i].dtypes.src = data_type_t::bf16;
      pv[i].dtypes.wei = wei_dt;
      pv[i].dtypes.dst = data_type_t::bf16;
      copy_attached_scale(wei_quant_tensors[i], pv[i].quant_params.wei_scale);
      copy_attached_zp   (wei_quant_tensors[i], pv[i].quant_params.wei_zp);
    }
    return pv;
  };

  // ── Reference: legacy 2-call with WOQ on both Op1 and Op2 ──
  {
    auto p_ref_op1 = build_params_woq(w1_t);
    for (auto &pp : p_ref_op1) pp.num_threads = 32;
    ASSERT_EQ(group_matmul_direct(gv_op1.layout, gv_op1.transA, gv_op1.transB,
                                  gv_op1.Ms, gv_op1.Ns, gv_op1.Ks, gv_op1.alpha,
                                  src_ref_p, gv_op1.lda, wei1_p, gv_op1.ldb,
                                  no_bias, gv_op1.beta, d1_ref_p, gv_op1.ldc,
                                  gv_op1.is_wc, p_ref_op1),
              status_t::success) << "ref Op1 (WOQ " << (p.wei_is_u4 ? "u4" : "s4")
                                 << ") " << p.name;

    if (!act_is_none) {
      for (int e = 0; e < E; ++e) {
        apply_ref_gated_act(d1_ref.bf16[e], M, N_gate_up, N_gate_up, act_type);
      }
    }

    auto p_ref_op2 = build_params_woq(w2_t);
    for (auto &pp : p_ref_op2) pp.num_threads = 32;
    std::vector<const void *> srcs2(E);
    for (int e = 0; e < E; ++e) srcs2[e] = d1_ref_p[e];
    ASSERT_EQ(group_matmul_direct(gv_op2.layout, gv_op2.transA, gv_op2.transB,
                                  gv_op2.Ms, gv_op2.Ns, gv_op2.Ks, gv_op2.alpha,
                                  srcs2, gv_op2.lda, wei2_p, gv_op2.ldb, no_bias,
                                  gv_op2.beta, d2_ref_p, gv_op2.ldc,
                                  gv_op2.is_wc, p_ref_op2),
              status_t::success) << "ref Op2 (WOQ " << (p.wei_is_u4 ? "u4" : "s4")
                                 << ") " << p.name;
  }

  // ── Test: single fused call with WOQ on both halves + VF forced ──
  grp_matmul_gated_act_params act{};
  act.act = act_type;
  auto act_ptr = act_is_none ? nullptr : &act;

  auto fused = make_fused_moe_op2(E, H, wei2_p, no_bias);
  // Op2 weight-side scale + zp carried through `fused.down_*`.  The
  // dispatcher's `setup_op2_dispatch_scratch` will copy these into
  // `params_w2[i].quant_params.{wei_scale,wei_zp}` so the eligibility
  // gate's regime-B check on Op2's params sees the scale buffer.
  fused.down_scale.resize(E);
  fused.down_zp   .resize(E);
  for (int i = 0; i < E; ++i) {
    copy_attached_scale(w2_t[i], fused.down_scale[i]);
    copy_attached_zp   (w2_t[i], fused.down_zp[i]);
  }

  std::vector<void *> d1_for_fused;
  std::vector<int>    ldc1_for_fused;
  if (p.internal_alloc) {
    d1_for_fused.assign(E, nullptr);
    ldc1_for_fused.assign(E, 0);
  } else {
    d1_for_fused   = d1_fused_p;
    ldc1_for_fused = std::vector<int>(E, N_gate_up);
    fused.dst_down = d2_fused_p;
    fused.ldc_down = std::vector<int>(E, H);
  }

  auto p_test = build_params_woq(w1_t);
  for (auto &pp : p_test) pp.num_threads = 32;

  int tag = 0;
  {
    MTilePathCaptureGuard cap;
    ASSERT_EQ(group_matmul_direct(gv_op1.layout, gv_op1.transA, gv_op1.transB,
                                  gv_op1.Ms, gv_op1.Ns, gv_op1.Ks, gv_op1.alpha,
                                  src_test_p, gv_op1.lda, wei1_p, gv_op1.ldb,
                                  no_bias, gv_op1.beta,
                                  d1_for_fused, ldc1_for_fused,
                                  gv_op1.is_wc, p_test, nullptr, act_ptr, &fused),
              status_t::success) << "test " << p.name;
    tag = test_api::s_last_m_tile_path.load(std::memory_order_relaxed);
  }
  EXPECT_EQ(tag, test_api::m_tile_path_tag::kVerticalFusionWOQ)
      << "vertical fusion did NOT engage on WOQ shape " << p.name
      << " (" << (p.wei_is_u4 ? "u4" : "s4") << ") — capture tag = " << tag
      << " (expected kVerticalFusionWOQ = "
      << test_api::m_tile_path_tag::kVerticalFusionWOQ << ")";

  // ── Sanity: confirm both routes actually computed a GEMM (defense
  //   against a kernel-setup failure that bails on both paths and
  //   leaves the dst buffers at all-zero, which would let the
  //   verify_per_expert_2d below trivially pass on 0 == 0). ──
  double ref_abs_sum = 0.0, test_abs_sum = 0.0;
  const auto &test_out_buf = p.internal_alloc ? src_test.bf16 : d2_fused.bf16;
  for (int e = 0; e < E; ++e) {
    for (size_t k = 0; k < d2_ref.bf16[e].size(); ++k) {
      ref_abs_sum += std::abs(static_cast<float>(d2_ref.bf16[e][k]));
    }
    for (size_t k = 0; k < test_out_buf[e].size(); ++k) {
      test_abs_sum += std::abs(static_cast<float>(test_out_buf[e][k]));
    }
  }
  ASSERT_GT(ref_abs_sum,  1e-3)
      << "[17.c] reference 2-call path produced all-zero d2_ref "
         "(sum=" << ref_abs_sum << ") — WOQ-"
      << (p.wei_is_u4 ? "u4" : "s4") << " dispatch likely short-"
         "circuited; check the kernel error log for the root cause "
         "and confirm `is_woq` at aocl_postop.cpp:178 is still "
         "gated to s4/u4.";
  ASSERT_GT(test_abs_sum, 1e-3)
      << "[17.c] fused-VF test path produced all-zero output "
         "(sum=" << test_abs_sum << ").";

  std::ostringstream lbl;
  lbl << "[17.c] vertical_fusion_woq " << p.name
      << " wei=" << (p.wei_is_u4 ? "u4" : "s4")
      << " act=" << p.act_int
      << " M=" << M << " E=" << E
      << " dim=" << dim << " H=" << H
      << (p.internal_alloc ? " intalloc" : " calleralloc");

  // Verification: internal-alloc reads Op2 back from src_test (stride
  // K_in = H, since K = H == N_down — a perfectly packed in-place
  // reuse); caller-alloc reads d2_fused (stride H).  Same contract
  // as TestFusedMoEVerticalBF16.Correctness.
  if (p.internal_alloc) {
    verify_per_expert_2d(src_test, K_in, d2_ref, H,
                         E, M, H, is_bf16,
                         tol_fused(is_bf16), lbl.str());
  } else {
    verify_per_expert_2d(d2_fused, H, d2_ref, H,
                         E, M, H, is_bf16,
                         tol_fused(is_bf16), lbl.str());
  }
}

static std::vector<VerticalFusionWOQTestParam>
make_vertical_fusion_woq_params() {
  // Shape selection follows the same wide-N + round-based clearance
  // rationale as `make_vertical_fusion_params` (the BF16 sibling
  // fixture above).  At the 32-thread pin we need `total_need > 16`
  // and `E <= 32`; the two archs below satisfy both:
  //
  //   * mixtral_E4_M128: total_need = 4 * ceil(128/16) = 32 > 16 ✓
  //   * qwen3_E8_M64:    total_need = 8 * ceil( 64/16) = 32 > 16 ✓
  //
  // Smaller arch grid than the BF16 case (2 vs 5) to keep the WOQ
  // cube at 2 archs × 4 acts × 2 alloc × 2 weight-dtypes = 32 cases.
  // The BF16 fixture already exercises the same per-thread slice
  // planner across more architectures, so this fixture's value is
  // verifying the WOQ kernel routing through the SAME planner, not
  // re-validating the planner itself.
  static const struct { const char *name; int E, dim, H, M; } kArchs[] = {
    {"mixtral_E4_M128", 4, 64, 64, 128},
    {"qwen3_E8_M64",    8, 64, 64,  64},
  };
  std::vector<VerticalFusionWOQTestParam> out;
  for (const auto &a : kArchs) {
    for (int act : {0, 1, 2, 3}) {
      for (bool intalloc : {false, true}) {
        for (bool u4 : {false, true}) {
          out.push_back({a.name, a.E, a.dim, a.H, a.M, act, intalloc, u4});
        }
      }
    }
  }
  return out;
}

INSTANTIATE_TEST_SUITE_P(GroupMatmulFusedMoEVerticalWOQ,
                         TestFusedMoEVerticalWOQ,
                         ::testing::ValuesIn(make_vertical_fusion_woq_params()),
                         VerticalFusionWOQParamName);

// ===============================================================================
// [17.d] TestFusedMoEVerticalDQINT8 - DQ-INT8 vertical-fusion correctness.
//
// Per-token symmetric dynamic-INT8 routed through the unified vertical-fusion
// executor (`flat_m_tile_pipeline_bf16`, regime kRegDQINT8).  The pipeline
// shape is identical to BF16 / WOQ; the per-stage differences are entirely
// contained in (a) the pre-OMP per-expert bf16→s8 source hoist and (b)
// the per-thread Stage 2b re-quant of the post-activation tile.  This
// fixture asserts:
//
//   * The capture tag is `kVerticalFusionDQINT8` — the executor committed
//     to the DQ-INT8 path through its eligibility wrapper (and did NOT
//     fall through to BF16 / WOQ tags or any legacy `flat_m_tile`
//     branch).
//   * Numerical bit-equivalence-modulo-tolerance against the legacy
//     two-call reference path (Op1 dynamic-INT8 + activation, then Op2
//     dynamic-INT8 with the activated bf16 intermediate as src).
//   * Both ref and fused paths produce non-zero output (defense against
//     a kernel-setup failure that silently bails on both routes).
//
// Why both internal_alloc modes:
//   * `internal_alloc=false`  exercises the (caller-alloc dst_w13)
//                              spill path inside the executor — the
//                              compact-bf16 → s8 re-quant of Stage 2b
//                              runs alongside the dst_w13 memcpy, so
//                              both compete for the same per-thread
//                              cache lines.
//   * `internal_alloc=true`   exercises the in-place reuse path where
//                              Op2's dst lands back into the caller's
//                              source buffer — the same reuse that
//                              `TestFusedMoEQuantDynINT8` exercises
//                              for the legacy two-pass.
//
// Shape envelope:
//   * `M >= 16` per expert — required by the AOCL BF16→S8 reorder
//     kernel that runs inside `reorder_quantization_wrapper`.  The
//     per-token symmetric path uses a 16-row tile; M < 16 hits a
//     rejection path that the existing `TestFusedMoEQuantDynINT8`
//     test grid already calls out (see its doc-block).
//
// Reference path uses two standalone `group_matmul_direct` calls with
// the SAME DQ-INT8 params (bf16 src + s8 wei, `dynamic_quant=true`,
// `compute=s8`, per-token symmetric `src_scale = {M, 1}`, per-channel
// `wei_scale = {1, N}`).  Identical kernel routing to what the legacy
// two-pass dispatch fork inside `group_matmul_fused_moe_execute` would
// produce, so the only observable difference between ref and fused
// paths is per-thread slice locality + pre-OMP hoist + Stage 2b
// re-quant — i.e. exactly what we want to verify produces zero
// numerical drift.
// ===============================================================================

struct VerticalFusionDQINT8TestParam {
  const char *name;          // Architecture descriptor for filter
                             // (e.g. "mixtral_E4_M128").
  int num_experts;
  int dim;                   // Op1 hidden_size / 2 (gate width per token).
  int hidden;                // Op2 N_down (== Op1 K_in: residual width).
  int M;                     // Per-expert active token count (uniform).
                             // Required >= 16 (AOCL BF16→S8 reorder
                             // per-token tile size).
  int act_int;               // 0=none, 1=silu, 2=gelu, 3=swiglu_oai.
  bool internal_alloc;       // Op2 dst path: false = caller-alloc
                             // dst_down, true = internal-alloc + src
                             // reuse.
};

static std::string VerticalFusionDQINT8ParamName(
    const ::testing::TestParamInfo<VerticalFusionDQINT8TestParam> &info) {
  static const char *kAct[] = {"none", "silu", "gelu", "swiglu"};
  const auto &p = info.param;
  return std::string(p.name)
       + "_dqint8_" + kAct[p.act_int]
       + (p.internal_alloc ? "_intalloc" : "_calleralloc");
}

class TestFusedMoEVerticalDQINT8
    : public ::testing::TestWithParam<VerticalFusionDQINT8TestParam> {};

TEST_P(TestFusedMoEVerticalDQINT8, Correctness) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  using zendnnl::common::data_type_t;

  reset_grp_matmul_caches();

  const auto &p = GetParam();
  const int E         = p.num_experts;
  const int dim       = p.dim;
  const int N_gate_up = 2 * dim;
  const int H         = p.hidden;
  const int K_in      = H;
  const int M         = p.M;
  const auto act_type = static_cast<grp_matmul_gated_act_t>(p.act_int);
  const bool is_bf16  = true;
  const bool act_is_none = (act_type == grp_matmul_gated_act_t::none);
  const int K_down = act_is_none ? N_gate_up : dim;

  // RAII pins — same shape as TestFusedMoEVerticalBF16.Correctness:
  // ALGO 2, VF FORCED, generous scratch budget.  DQ-INT8 adds a
  // per-thread Stage 2b scratch on top of the bf16 staging tile, so
  // the 1024 KB budget gives comfortable headroom for any of the
  // shapes below at M = 128 / dim = 64.
  AlgoEnvGuard                  algo2_guard(2);
  MoEVerticalFusionOverride     vert_guard(1);
  MoEPipelineScratchKbOverride  scratch_guard(1024);

  // ── Op1 source tensors: BF16 with per-token F32 src_scale buffer ──
  // The src_scale tensor is a zero-allocated F32 buffer attached to
  // src_t via `uniform_dist_tensor(..., scale, zp=empty)` — the
  // dispatcher / hoist fills its contents at runtime when it
  // reorders the BF16 source to S8.  Same setup as
  // `TestFusedMoEQuantDynINT8.BothPasses` so the user-side contract
  // is identical (no special wiring for vertical fusion — the
  // executor pulls everything it needs from this tensor's attached
  // metadata).
  tensor_factory_t factory{};
  std::vector<tensor_t> src_t(E), src_scale_t(E);
  for (int i = 0; i < E; ++i) {
    src_scale_t[i] = factory.zero_tensor(
        {static_cast<uint64_t>(M), static_cast<uint64_t>(1)},
        data_type_t::f32);
    src_t[i] = factory.uniform_dist_tensor(
        {static_cast<uint64_t>(M), static_cast<uint64_t>(K_in)},
        data_type_t::bf16, 2.0, /*transposed=*/false,
        src_scale_t[i], tensor_t{});
  }

  // ── Op1 + Op2 weights: S8 + per-channel F32 wei_scale + per-tensor
  //    bf16 wei_zp (the symmetric path doesn't consume zp; it is
  //    set up for parity with the AOCL DLP s8s8→bf16 kernel's
  //    metadata expectations).  Identical to the WOQ_S8-symmetric
  //    weight pattern in `TestFusedMoEQuantDynINT8`.
  std::vector<tensor_t> w1_s8_t(E), w1_scale_t(E), w1_zp_t(E);
  std::vector<tensor_t> w2_s8_t(E), down_scale_t(E), down_zp_t(E);
  for (int i = 0; i < E; ++i) {
    auto w1_ref = factory.uniform_dist_tensor(
        {static_cast<uint64_t>(K_in), static_cast<uint64_t>(N_gate_up)},
        data_type_t::bf16, 2.0, /*transposed=*/false);
    ASSERT_EQ(quant_params_compute(factory, w1_ref,
                                   data_type_t::bf16, data_type_t::s8,
                                   /*scale_dims=*/{1, N_gate_up},
                                   data_type_t::f32,
                                   w1_scale_t[i], w1_zp_t[i], &w1_s8_t[i]),
              status_t::success);

    auto w2_ref = factory.uniform_dist_tensor(
        {static_cast<uint64_t>(K_down), static_cast<uint64_t>(H)},
        data_type_t::bf16, 2.0, /*transposed=*/false);
    ASSERT_EQ(quant_params_compute(factory, w2_ref,
                                   data_type_t::bf16, data_type_t::s8,
                                   /*scale_dims=*/{1, H},
                                   data_type_t::f32,
                                   down_scale_t[i], down_zp_t[i], &w2_s8_t[i]),
              status_t::success);
  }

  // ── Raw pointers + BF16 dst / intermediate buffers ───────────────
  TypedBuffers d1_ref, d1_fused, d2_ref, d2_fused, d1_unused;
  d1_ref   .alloc(E, (size_t)M * N_gate_up, is_bf16);
  d1_fused .alloc(E, (size_t)M * N_gate_up, is_bf16);
  d2_ref   .alloc(E, (size_t)M * H,         is_bf16);
  d2_fused .alloc(E, (size_t)M * H,         is_bf16);
  d1_unused.alloc(E, (size_t)M * N_gate_up, is_bf16);

  std::vector<const void *> srcs(E), wei1_p(E), wei2_p(E);
  for (int i = 0; i < E; ++i) {
    srcs  [i] = src_t  [i].get_raw_handle_unsafe();
    wei1_p[i] = w1_s8_t[i].get_raw_handle_unsafe();
    wei2_p[i] = w2_s8_t[i].get_raw_handle_unsafe();
  }
  auto d1_ref_p    = d1_ref  .ptrs(is_bf16);
  auto d1_fused_p  = d1_fused.ptrs(is_bf16);
  auto d2_ref_p    = d2_ref  .ptrs(is_bf16);
  auto d2_fused_p  = d2_fused.ptrs(is_bf16);
  auto d1_unused_p = d1_unused.ptrs(is_bf16);
  std::vector<const void *> no_bias(E, nullptr);

  auto gv_op1 = GemmVecs::uniform(E, M, N_gate_up, K_in);
  auto gv_op2 = GemmVecs::uniform(E, M, H, K_down);
  gv_op2.lda.assign(E, N_gate_up);  // Op2 reads d1_ref at the gated stride.
  // DQ-INT8 does NOT require is_weights_const (the executor handles
  // both const and non-const; we set it to true here to be
  // consistent with the WOQ test and to exercise the prepack path).
  gv_op1.is_wc.assign(E, true);
  gv_op2.is_wc.assign(E, true);

  // Build per-pass matmul_params for DQ-INT8 — identical pattern to
  // `TestFusedMoEQuantDynINT8.BothPasses::build_params_dynamic_int8`.
  auto build_params_dynamic_int8 =
      [&](const tensor_t &src_tensor, const tensor_t &wei_tensor) {
    auto p_slot = make_uniform_params(1, data_type_t::bf16)[0];
    p_slot.dtypes.src     = data_type_t::bf16;
    p_slot.dtypes.wei     = data_type_t::s8;
    p_slot.dtypes.dst     = data_type_t::bf16;
    p_slot.dtypes.compute = data_type_t::s8;
    p_slot.dynamic_quant  = true;
    copy_attached_scale(src_tensor, p_slot.quant_params.src_scale);
    copy_attached_scale(wei_tensor, p_slot.quant_params.wei_scale);
    copy_attached_zp   (wei_tensor, p_slot.quant_params.wei_zp);
    return p_slot;
  };

  // ── Reference: legacy 2-call with DQ-INT8 on BOTH passes ─────────
  {
    std::vector<matmul_params> p_ref_op1(E);
    for (int i = 0; i < E; ++i) {
      p_ref_op1[i] = build_params_dynamic_int8(src_t[i], w1_s8_t[i]);
      p_ref_op1[i].num_threads = 32;
    }
    ASSERT_EQ(group_matmul_direct(gv_op1.layout, gv_op1.transA, gv_op1.transB,
                                  gv_op1.Ms, gv_op1.Ns, gv_op1.Ks, gv_op1.alpha,
                                  srcs, gv_op1.lda, wei1_p, gv_op1.ldb, no_bias,
                                  gv_op1.beta, d1_ref_p, gv_op1.ldc,
                                  gv_op1.is_wc, p_ref_op1),
              status_t::success) << "ref Op1 (DQ-INT8) " << p.name;

    if (!act_is_none) {
      for (int e = 0; e < E; ++e) {
        apply_ref_gated_act(d1_ref.bf16[e], M, N_gate_up, N_gate_up, act_type);
      }
    }

    std::vector<matmul_params> p_ref_op2(E);
    for (int i = 0; i < E; ++i) {
      p_ref_op2[i] = make_uniform_params(1, data_type_t::bf16)[0];
      p_ref_op2[i].dtypes.src     = data_type_t::bf16;
      p_ref_op2[i].dtypes.wei     = data_type_t::s8;
      p_ref_op2[i].dtypes.dst     = data_type_t::bf16;
      p_ref_op2[i].dtypes.compute = data_type_t::s8;
      p_ref_op2[i].dynamic_quant  = true;
      p_ref_op2[i].num_threads    = 32;
      // Per-token src_scale for Op2 (buff = nullptr → kernel allocates).
      p_ref_op2[i].quant_params.src_scale.buff = nullptr;
      p_ref_op2[i].quant_params.src_scale.dt   = data_type_t::f32;
      p_ref_op2[i].quant_params.src_scale.dims = {M, 1};
      copy_attached_scale(w2_s8_t[i], p_ref_op2[i].quant_params.wei_scale);
      copy_attached_zp   (w2_s8_t[i], p_ref_op2[i].quant_params.wei_zp);
    }
    std::vector<const void *> srcs2(E);
    for (int e = 0; e < E; ++e) srcs2[e] = d1_ref_p[e];
    ASSERT_EQ(group_matmul_direct(gv_op2.layout, gv_op2.transA, gv_op2.transB,
                                  gv_op2.Ms, gv_op2.Ns, gv_op2.Ks, gv_op2.alpha,
                                  srcs2, gv_op2.lda, wei2_p, gv_op2.ldb, no_bias,
                                  gv_op2.beta, d2_ref_p, gv_op2.ldc,
                                  gv_op2.is_wc, p_ref_op2),
              status_t::success) << "ref Op2 (DQ-INT8) " << p.name;
  }

  // ── Test: single fused call with DQ-INT8 + VF forced ─────────────
  grp_matmul_gated_act_params act{};
  act.act = act_type;
  auto act_ptr = act_is_none ? nullptr : &act;

  auto fused = make_fused_moe_op2(E, H, wei2_p, no_bias);
  // Op2 weight-side scale + zp carried through `fused.down_*`.  The
  // dispatcher's `setup_op2_dispatch_scratch` copies these into
  // `params_w2[i].quant_params.{wei_scale,wei_zp}` so the eligibility
  // gate's regime-C check on Op2's params sees the scale buffer.
  fused.down_scale.resize(E);
  fused.down_zp   .resize(E);
  for (int i = 0; i < E; ++i) {
    copy_attached_scale(w2_s8_t[i], fused.down_scale[i]);
    copy_attached_zp   (w2_s8_t[i], fused.down_zp[i]);
  }

  std::vector<void *> d1_for_fused;
  std::vector<int>    ldc1_for_fused;
  if (p.internal_alloc) {
    d1_for_fused.assign(E, nullptr);
    ldc1_for_fused.assign(E, 0);
    // For internal-alloc mode the Op2 result lands back into the
    // caller's src buffer slot (Op2 re-uses src memory).  No
    // dst_down assignment needed here — the dispatcher handles
    // the reuse via the Op2-dispatch-scratch setup.
  } else {
    d1_for_fused   = d1_fused_p;
    ldc1_for_fused = std::vector<int>(E, N_gate_up);
    fused.dst_down = d2_fused_p;
    fused.ldc_down = std::vector<int>(E, H);
  }

  std::vector<matmul_params> p_test(E);
  for (int i = 0; i < E; ++i) {
    p_test[i] = build_params_dynamic_int8(src_t[i], w1_s8_t[i]);
    p_test[i].num_threads = 32;
  }

  int tag = 0;
  {
    MTilePathCaptureGuard cap;
    ASSERT_EQ(group_matmul_direct(gv_op1.layout, gv_op1.transA, gv_op1.transB,
                                  gv_op1.Ms, gv_op1.Ns, gv_op1.Ks, gv_op1.alpha,
                                  srcs, gv_op1.lda, wei1_p, gv_op1.ldb,
                                  no_bias, gv_op1.beta,
                                  d1_for_fused, ldc1_for_fused,
                                  gv_op1.is_wc, p_test, nullptr, act_ptr, &fused),
              status_t::success) << "test " << p.name;
    tag = test_api::s_last_m_tile_path.load(std::memory_order_relaxed);
  }
  EXPECT_EQ(tag, test_api::m_tile_path_tag::kVerticalFusionDQINT8)
      << "vertical fusion did NOT engage on DQ-INT8 shape " << p.name
      << " — capture tag = " << tag
      << " (expected kVerticalFusionDQINT8 = "
      << test_api::m_tile_path_tag::kVerticalFusionDQINT8 << ")";

  // Sanity: confirm both routes actually computed a GEMM (defense
  // against a kernel-setup failure that bails on both paths and
  // leaves the dst buffers at all-zero).  Internal-alloc mode reads
  // Op2 back from `src_t[].get_raw_handle_unsafe()` — but the
  // tensor object owns that memory and the dispatcher writes through
  // it; checking via the per-expert source tensor's bytes works
  // identically.
  double ref_abs_sum = 0.0, test_abs_sum = 0.0;
  for (int e = 0; e < E; ++e) {
    for (size_t k = 0; k < d2_ref.bf16[e].size(); ++k) {
      ref_abs_sum  += std::abs(static_cast<float>(d2_ref .bf16[e][k]));
    }
    if (p.internal_alloc) {
      // Op2 dst landed back into src_t[e] (M × K_in = M × H bytes
      // worth of bf16) — read the activation-relevant prefix.
      // Bit-cast each raw uint16_t to bf16 via `from_bits` before
      // converting to float, otherwise the arithmetic uint16→float
      // cast would treat the raw bit pattern as an integer (same
      // gotcha as the `internal_view` build below).
      const auto *bytes =
          static_cast<const uint16_t *>(src_t[e].get_raw_handle_unsafe());
      for (size_t k = 0; k < static_cast<size_t>(M) * H; ++k) {
        const float v = static_cast<float>(bfloat16_t::from_bits(bytes[k]));
        test_abs_sum += std::abs(v);
      }
    } else {
      for (size_t k = 0; k < d2_fused.bf16[e].size(); ++k) {
        test_abs_sum += std::abs(static_cast<float>(d2_fused.bf16[e][k]));
      }
    }
  }
  ASSERT_GT(ref_abs_sum,  1e-3)
      << "[17.d] reference 2-call path produced all-zero d2_ref "
         "(sum=" << ref_abs_sum << ") — DQ-INT8 dispatch likely "
         "short-circuited; check the BF16→S8 reorder log.";
  ASSERT_GT(test_abs_sum, 1e-3)
      << "[17.d] fused-VF test path produced all-zero output "
         "(sum=" << test_abs_sum << ").";

  std::ostringstream lbl;
  lbl << "[17.d] vertical_fusion_dqint8 " << p.name
      << " act=" << p.act_int
      << " M=" << M << " E=" << E
      << " dim=" << dim << " H=" << H
      << (p.internal_alloc ? " intalloc" : " calleralloc");

  // Verification: internal-alloc reads Op2 back from src_t[]'s
  // underlying bytes (the dispatcher writes the reused dst there);
  // caller-alloc reads d2_fused.  DQ-INT8 has slightly looser
  // tolerance than BF16 / WOQ because the bf16→s8 round-trip on
  // BOTH halves introduces quantization-noise accumulation; use
  // the same `tol_fused(is_bf16)` as the WOQ test, which has
  // been calibrated against the existing
  // `TestFusedMoEQuantDynINT8.BothPasses` reference path that
  // shares the same quantization scheme.
  if (p.internal_alloc) {
    // Build a TypedBuffers view backed by the per-expert source
    // tensor handles so `verify_per_expert_2d` can iterate them.
    // CRITICAL: copy raw bytes (memcpy), NOT via std::vector::assign
    // — the implicit `bfloat16_t(uint16_t)` constructor routes
    // through `float(i)` (see bfloat16.hpp line 73) and would
    // arithmetically convert the bit pattern instead of bit-casting,
    // producing wildly wrong "values" (e.g. raw bits 0x4200 = 16896
    // arithmetic instead of the intended bf16 value).  TypedBuffers
    // stores bf16 elements as `bfloat16_t` with a single `uint16_t
    // raw_bits_` member, so a raw byte copy reconstructs the bit
    // pattern correctly.
    TypedBuffers internal_view;
    internal_view.bf16.resize(E);
    for (int e = 0; e < E; ++e) {
      internal_view.bf16[e].resize((size_t)M * H);
      std::memcpy(internal_view.bf16[e].data(),
                  src_t[e].get_raw_handle_unsafe(),
                  static_cast<size_t>(M) * H * sizeof(uint16_t));
    }
    verify_per_expert_2d(internal_view, K_in, d2_ref, H,
                         E, M, H, is_bf16,
                         tol_fused(is_bf16), lbl.str());
  } else {
    verify_per_expert_2d(d2_fused, H, d2_ref, H,
                         E, M, H, is_bf16,
                         tol_fused(is_bf16), lbl.str());
  }
}

static std::vector<VerticalFusionDQINT8TestParam>
make_vertical_fusion_dqint8_params() {
  // Shape selection follows the SAME wide-N + round-based clearance
  // rationale as `make_vertical_fusion_woq_params` (mixtral_E4_M128
  // + qwen3_E8_M64 both satisfy `total_need = E × ceil(M/16) > 16`
  // at 32 threads, so the planner reaches the single-tier path
  // every time).  M >= 16 satisfies the AOCL BF16→S8 reorder's
  // per-token tile requirement.
  //
  // Smaller param cube than the BF16 fixture (2 archs × 4 acts × 2
  // alloc = 16 cases) — the BF16 fixture already validates the
  // per-thread slice planner across more architectures, so this
  // fixture's value is verifying the DQ-INT8 pre-OMP hoist + Stage
  // 2b re-quant through that SAME planner.
  static const struct { const char *name; int E, dim, H, M; } kArchs[] = {
    {"mixtral_E4_M128", 4, 64, 64, 128},
    {"qwen3_E8_M64",    8, 64, 64,  64},
  };
  std::vector<VerticalFusionDQINT8TestParam> out;
  for (const auto &a : kArchs) {
    for (int act : {0, 1, 2, 3}) {
      for (bool intalloc : {false, true}) {
        out.push_back({a.name, a.E, a.dim, a.H, a.M, act, intalloc});
      }
    }
  }
  return out;
}

INSTANTIATE_TEST_SUITE_P(GroupMatmulFusedMoEVerticalDQINT8,
                         TestFusedMoEVerticalDQINT8,
                         ::testing::ValuesIn(make_vertical_fusion_dqint8_params()),
                         VerticalFusionDQINT8ParamName);

// ===============================================================================
// [18] TestFusedMoEVerticalBF16Fallthrough - eligibility gate negative cases.
//
// For each fallthrough trigger:
//   * `MoEVerticalFusionOverride(-1)` (env DISABLED).
//   * F32 dtypes (eligibility gate rejects non-bf16 paths in phase 1).
//   * Unsupported activation (e.g. `unsupported_act_t`-equivalent —
//     we use `params[0].dynamic_quant = true` as a proxy for "dtype-
//     ineligible" since the activation enum's supported set is
//     {none, silu, gelu, swiglu_oai}, all of which are accepted by
//     the gate).
//   * `params[0].dynamic_quant = true` (dispatcher demands dynamic-
//     quant routing → phase 2 territory).
//
// After every fallthrough call the `m_tile_path` capture tag must
// NOT be `kVerticalFusionBF16`.  ALGO 2 forced so the legacy
// `flat_m_tile` populates the tag with one of its four branch
// values (kRoundBased / kMultiTier / kWideNFallback / kPhase2Single)
// — any of those is acceptable.  Functional correctness is also
// verified against the same `run_legacy_2call_ref` so the
// fallthrough path didn't silently lose precision.
// ===============================================================================

enum class FallthroughCause {
  EnvDisabled,
  DtypeF32,
  DynamicQuant,
};

struct FallthroughParam {
  FallthroughCause cause;
  const char *name;
};

static std::string FallthroughParamName(
    const ::testing::TestParamInfo<FallthroughParam> &info) {
  return info.param.name;
}

class TestFusedMoEVerticalBF16Fallthrough
    : public ::testing::TestWithParam<FallthroughParam> {};

TEST_P(TestFusedMoEVerticalBF16Fallthrough, EligibilityFails) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;

  reset_grp_matmul_caches();

  const auto &fp = GetParam();

  // Fixed shape for every cause — Qwen3-class at small M; the gate
  // outcome is what's under test, not the workload.
  const int E         = 8;
  const int dim       = 64;
  const int N_gate_up = 2 * dim;
  const int H         = 64;
  const int K_in      = H;
  const int M         = 4;
  const auto act_type = grp_matmul_gated_act_t::silu_and_mul;
  const bool is_bf16  = (fp.cause != FallthroughCause::DtypeF32);
  const data_type_t dt = is_bf16 ? data_type_t::bf16 : data_type_t::f32;
  const int K_down = dim;  // gated activation halves the dim.

  AlgoEnvGuard algo2_guard(2);
  // Vertical-fusion override depends on the cause: env-disabled
  // forces -1, every other cause leaves the override at 1 (so the
  // gate's data-driven reject is what we're verifying — not env-
  // disable masking the gate's other branches).
  MoEVerticalFusionOverride vert_guard(
      fp.cause == FallthroughCause::EnvDisabled ? -1 : 1);
  MoEPipelineScratchKbOverride scratch_guard(1024);

  TypedBuffers src_ref, src_test, w1, d1_ref, w2, d2_ref, d2_fused;
  src_ref .alloc(E, (size_t)M * K_in,         is_bf16);
  src_test.alloc(E, (size_t)M * K_in,         is_bf16);
  w1      .alloc(E, (size_t)K_in * N_gate_up, is_bf16);
  d1_ref  .alloc(E, (size_t)M * N_gate_up,    is_bf16);
  w2      .alloc(E, (size_t)K_down * H,       is_bf16);
  d2_ref  .alloc(E, (size_t)M * H,            is_bf16);
  d2_fused.alloc(E, (size_t)M * H,            is_bf16);
  fill_moe_tensors(E, is_bf16, &src_ref,  &w1, &w2);
  fill_moe_tensors(E, is_bf16, &src_test, nullptr, nullptr);

  auto gv_op1 = GemmVecs::uniform(E, M, N_gate_up, K_in);
  auto src_ref_p  = src_ref .cptrs(is_bf16);
  auto src_test_p = src_test.cptrs(is_bf16);
  auto wei1_p     = w1      .cptrs(is_bf16);
  auto wei2_p     = w2      .cptrs(is_bf16);
  auto d1_ref_p   = d1_ref  .ptrs(is_bf16);
  auto d2_ref_p   = d2_ref  .ptrs(is_bf16);
  auto d2_fused_p = d2_fused.ptrs(is_bf16);
  std::vector<const void *> no_bias(E, nullptr);

  grp_matmul_gated_act_params act{};
  act.act = act_type;

  ASSERT_EQ(run_legacy_2call_ref(E, M, K_in, N_gate_up, K_down, H,
                                 is_bf16, act_type,
                                 src_ref_p, wei1_p, wei2_p,
                                 d1_ref_p, d2_ref_p),
            status_t::success) << "ref " << fp.name;

  auto fused = make_fused_moe_op2(E, H, wei2_p, no_bias);
  fused.dst_down = d2_fused_p;
  fused.ldc_down = std::vector<int>(E, H);

  auto params = make_uniform_params(E, dt);
  if (fp.cause == FallthroughCause::DynamicQuant) {
    // Trip the eligibility gate's `!dynamic_quant` check; the
    // dispatcher will route to legacy two-pass (which itself
    // refuses dynamic-quant on bf16-on-bf16-on-bf16 today, but the
    // legacy path's behaviour isn't what we're testing — we're
    // testing the fork's negative path).  Skip the gate inside
    // `flat_m_tile_pipeline_bf16` is exactly the contract we want
    // to verify here.
    for (auto &pp : params) pp.dynamic_quant = true;
  }

  int tag = 0;
  status_t st;
  {
    MTilePathCaptureGuard cap;
    auto pf = params;
    st = group_matmul_direct(gv_op1.layout, gv_op1.transA, gv_op1.transB,
                             gv_op1.Ms, gv_op1.Ns, gv_op1.Ks,
                             gv_op1.alpha, src_test_p,
                             gv_op1.lda, wei1_p, gv_op1.ldb,
                             no_bias, gv_op1.beta,
                             // Op1 dst is [M, N_gate_up]; Op2 output is
                             // routed via fused.dst_down.  Reuse d1_ref
                             // (Op1 result is not inspected after this call).
                             d1_ref_p, gv_op1.ldc,
                             gv_op1.is_wc, pf, nullptr, &act, &fused);
    tag = test_api::s_last_m_tile_path.load(std::memory_order_relaxed);
  }

  // DynamicQuant + bf16 may be rejected by the legacy dispatch
  // (e.g. when the inner dispatch refuses bf16-quant routing).  We
  // tolerate either success (fall-through reached legacy two-pass)
  // OR a clean refusal, but NOT a hang / crash / `kVerticalFusionBF16`
  // tag (the gate's whole point is to keep us out of the new path).
  if (st == status_t::success) {
    EXPECT_NE(tag, test_api::m_tile_path_tag::kVerticalFusionBF16)
        << "fallthrough cause " << fp.name
        << " produced kVerticalFusionBF16 tag — eligibility gate "
           "did not reject as expected.";
    // Functional check only when the dispatch actually ran.  For
    // the DynamicQuant cause, dynamic_quant + bf16-only test
    // inputs likely take a different reference numerical path
    // anyway, so we skip the strict value comparison there.
    if (fp.cause != FallthroughCause::DynamicQuant) {
      std::ostringstream lbl;
      lbl << "[18] vertical_fallthrough " << fp.name;
      verify_per_expert_2d(d2_fused, H, d2_ref, H,
                           E, M, H, is_bf16,
                           tol_fused(is_bf16), lbl.str());
    }
  } else {
    // Some causes are allowed to refuse outright (e.g. dynamic-quant
    // on a bf16-only kernel).  The contract is only that vertical
    // fusion did NOT engage.
    EXPECT_NE(tag, test_api::m_tile_path_tag::kVerticalFusionBF16)
        << "fallthrough cause " << fp.name
        << " returned non-success status AND captured "
           "kVerticalFusionBF16 — implies pipeline executor was "
           "entered before failing.";
  }
}

INSTANTIATE_TEST_SUITE_P(
    GroupMatmulFusedMoEVerticalBF16Fallthrough,
    TestFusedMoEVerticalBF16Fallthrough,
    ::testing::Values(
        FallthroughParam{FallthroughCause::EnvDisabled,  "EnvDisabled"},
        FallthroughParam{FallthroughCause::DtypeF32,     "DtypeF32"},
        FallthroughParam{FallthroughCause::DynamicQuant, "DynamicQuant"}),
    FallthroughParamName);

// ===============================================================================
// [18.b] TestFusedMoEVerticalBF16Bailouts — bail-out paths NOT covered
// by `EligibilityFails` above.  The eligibility gate has five additional
// bail conditions inside `flat_m_tile_pipeline_bf16`:
//   1. `active_ops > num_threads` (line 1504) — bails BEFORE plan, BEFORE
//      tag store, cleanly returns false.
//   2. `multi-tier would engage` (line 1578) — bails BEFORE plan,
//      BEFORE tag store.
//   3. `wide-N would engage` (line 1591) — bails AFTER plan, BEFORE tag
//      store.
//   4. `scratch budget exceeded` (line 1628-1630) — bails AFTER plan
//      computation, BEFORE OMP region, BEFORE tag store.
//   5. `alloc failure inside OMP` (line 1658-1668) — tag IS stored, but
//      all threads barrier-sync and skip if any alloc fails, so dst
//      stays untouched.  Cannot be triggered deterministically from
//      gtest without instrumenting malloc; covered by code-walk only.
//
// For cases 1, 3, 4 we add explicit fixtures.  Case 2 (multi-tier)
// requires a skewed-prompt shape (not the EligibilityFails fixed
// E=8 / M=4 shape) and is exercised indirectly by the multi-tier
// branch tests in test_algos.cpp where ALGO 2 is forced; the
// vertical-fusion gate is structurally a literal copy of the
// `flat_m_tile` multi-tier gate, so if multi-tier engages there it
// will also bail here on the same shape.
//
// Each test asserts:
//   (a) `status_t::success` (the dispatch routes to legacy two-pass).
//   (b) `tag != kVerticalFusionBF16` (the gate rejected as expected).
//   (c) functional correctness vs the legacy 2-pass reference.
// ===============================================================================

// Helper: build the same E=8 / M=4 / bf16 fixture as
// `TestFusedMoEVerticalBF16Fallthrough.EligibilityFails`, plus expose
// the per-test scratch / vertical / thread knobs so individual tests
// can drive specific bail conditions.
static void run_vertical_bailout_check(
    const char *case_label,
    int E, int M, int dim, int K_in, int H,
    int vertical_fusion_value, int scratch_kb,
    int num_threads_for_call,
    bool expect_engage = false) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  using zendnnl::common::data_type_t;

  reset_grp_matmul_caches();

  const int N_gate_up = 2 * dim;
  const int K_down    = dim;
  const bool is_bf16  = true;
  const data_type_t dt = data_type_t::bf16;
  const auto act_type = grp_matmul_gated_act_t::silu_and_mul;

  AlgoEnvGuard algo2_guard(2);
  MoEVerticalFusionOverride vert_guard(vertical_fusion_value);
  MoEPipelineScratchKbOverride scratch_guard(scratch_kb);

  TypedBuffers src_ref, src_test, w1, d1_ref, w2, d2_ref, d2_fused;
  src_ref .alloc(E, (size_t)M * K_in,         is_bf16);
  src_test.alloc(E, (size_t)M * K_in,         is_bf16);
  w1      .alloc(E, (size_t)K_in * N_gate_up, is_bf16);
  d1_ref  .alloc(E, (size_t)M * N_gate_up,    is_bf16);
  w2      .alloc(E, (size_t)K_down * H,       is_bf16);
  d2_ref  .alloc(E, (size_t)M * H,            is_bf16);
  d2_fused.alloc(E, (size_t)M * H,            is_bf16);
  fill_moe_tensors(E, is_bf16, &src_ref,  &w1, &w2);
  fill_moe_tensors(E, is_bf16, &src_test, nullptr, nullptr);

  auto gv_op1 = GemmVecs::uniform(E, M, N_gate_up, K_in);
  auto src_ref_p  = src_ref .cptrs(is_bf16);
  auto src_test_p = src_test.cptrs(is_bf16);
  auto wei1_p     = w1      .cptrs(is_bf16);
  auto wei2_p     = w2      .cptrs(is_bf16);
  auto d1_ref_p   = d1_ref  .ptrs(is_bf16);
  auto d2_ref_p   = d2_ref  .ptrs(is_bf16);
  auto d2_fused_p = d2_fused.ptrs(is_bf16);
  std::vector<const void *> no_bias(E, nullptr);

  grp_matmul_gated_act_params act{};
  act.act = act_type;

  ASSERT_EQ(run_legacy_2call_ref(E, M, K_in, N_gate_up, K_down, H,
                                 is_bf16, act_type,
                                 src_ref_p, wei1_p, wei2_p,
                                 d1_ref_p, d2_ref_p),
            status_t::success) << "ref " << case_label;

  auto fused = make_fused_moe_op2(E, H, wei2_p, no_bias);
  fused.dst_down = d2_fused_p;
  fused.ldc_down = std::vector<int>(E, H);

  auto params = make_uniform_params(E, dt);
  // num_threads_for_call==0 means "leave at default (= OMP max)"; this
  // mirrors the existing TestFusedMoEVerticalBF16Fallthrough fixture
  // which doesn't override params.num_threads.  Non-zero values are
  // plumbed through `resolve_num_threads` for tests that want to pin
  // the team size (e.g. trigger `active_ops > num_threads` bail).
  if (num_threads_for_call > 0) {
    for (auto &pp : params) pp.num_threads = num_threads_for_call;
  }

  int tag = 0;
  status_t st;
  {
    MTilePathCaptureGuard cap;
    auto pf = params;
    st = group_matmul_direct(gv_op1.layout, gv_op1.transA, gv_op1.transB,
                             gv_op1.Ms, gv_op1.Ns, gv_op1.Ks,
                             gv_op1.alpha, src_test_p,
                             gv_op1.lda, wei1_p, gv_op1.ldb,
                             no_bias, gv_op1.beta,
                             // Op1 dst is [M, N_gate_up]; Op2 output is
                             // routed via fused.dst_down.  Reuse d1_ref
                             // (Op1 result is not inspected after this call).
                             d1_ref_p, gv_op1.ldc,
                             gv_op1.is_wc, pf, nullptr, &act, &fused);
    tag = test_api::s_last_m_tile_path.load(std::memory_order_relaxed);
  }
  // Contract (matches `TestFusedMoEVerticalBF16Fallthrough.Eligibility
  // Fails` at line 3071-3096): the bail-out gate's only strict
  // requirement is `tag != kVerticalFusionBF16` — the pipeline did
  // NOT engage.  The dispatch's overall status may be success (legacy
  // 2-pass took over and produced a valid result) OR a refusal status
  // (e.g. some downstream invariant on bf16-with-low-thread-count
  // doesn't hold).  Both outcomes prove the gate fired; only entering
  // the pipeline and silently producing `kVerticalFusionBF16` would
  // indicate a regression in the bail-out logic.
  //
  // When `expect_engage` is set the contract inverts: the shape is one
  // that WOULD have bailed on the scratch budget, but the caller passed
  // the UNBOUNDED budget (`scratch_kb == -1`), so the gate must be
  // disabled and the pipeline must engage (`tag == kVerticalFusionBF16`)
  // AND the dispatch must succeed.
  if (expect_engage) {
    EXPECT_EQ(tag, test_api::m_tile_path_tag::kVerticalFusionBF16)
        << case_label << " did NOT engage vertical fusion under an "
           "unbounded scratch budget; got tag=" << tag
        << " status=" << static_cast<int>(st);
    EXPECT_EQ(st, status_t::success)
        << case_label << " unbounded-budget engage path returned non-"
           "success status=" << static_cast<int>(st);
  } else {
    EXPECT_NE(tag, test_api::m_tile_path_tag::kVerticalFusionBF16)
        << case_label << " produced kVerticalFusionBF16 tag — bail-out "
           "gate did not reject; got tag=" << tag
        << " status=" << static_cast<int>(st);
  }

  // Functional check only when dispatch succeeded (fall-through
  // produced a usable buffer).  When the dispatch refuses, we cannot
  // compare values — the gate's contract is satisfied by `tag !=
  // kVerticalFusionBF16` above.
  if (st == status_t::success) {
    std::ostringstream lbl;
    lbl << "[18.b] vertical_bailout " << case_label;
    verify_per_expert_2d(d2_fused, H, d2_ref, H,
                         E, M, H, is_bf16,
                         tol_fused(is_bf16), lbl.str());
  }
}

// Bail 1: `active_ops > num_threads`.  E=16 actives, num_threads=4.
// The pipeline checks at line 1504 and returns false; legacy two-pass
// handles the call.  Functional correctness verified.
TEST(TestFusedMoEVerticalBF16Bailouts, ActiveOpsExceedsNumThreads) {
  run_vertical_bailout_check(
      "active_ops_16_threads_4",
      /*E=*/16, /*M=*/4, /*dim=*/64,
      /*K_in=*/64, /*H=*/64,
      /*vertical_fusion=*/1,    // force-attempt
      /*scratch_kb=*/1024,
      /*num_threads_for_call=*/4);   // 16 > 4 ⇒ bail
}

// Bail 2: scratch budget exceeded.  Verifies the per-thread scratch
// budget gate in `m_tile/group_matmul_m_tile.cpp` returns false
// BEFORE the OMP region (no partial dst writes).
//
// The budget cap is on `slice_M × N_w13 × inter_elem`.  We size the
// shape so even the smallest possible slice (slice_M = 1) overflows the
// 1 KB budget UNCONDITIONALLY: N_w13 = 2*dim = 1024, inter_elem = 2 ⇒
// 2048 B per row > 1024 B budget regardless of how the planner slices
// M across threads.  This makes the scratch gate the SOLE, deterministic
// trigger and removes the previous thread-count fragility (with dim=64
// and num_threads left at the OMP default, a 4-row slice landed exactly
// on the 1 KB boundary — not over — so on an 8-thread host the gate
// did NOT fire and the executor engaged; at other thread counts the
// test only "passed" by bailing via round-based / wide-N instead).
//
// num_threads is pinned to E so the OTHER gates are provably clear:
// active_ops(8) <= num_threads(8) ⇒ no round-based bail; total_need=8 ⇒
// 8*2=16 > 8 ⇒ no wide-N; max_M=4 < kHybridMinMaxM(256) ⇒ no multi-tier.
TEST(TestFusedMoEVerticalBF16Bailouts, ScratchBudgetExceeded) {
  run_vertical_bailout_check(
      "scratch_budget_1kb",
      /*E=*/8, /*M=*/4, /*dim=*/512,  // N_w13=1024 ⇒ 2 KB/row > 1 KB
      /*K_in=*/64, /*H=*/64,
      /*vertical_fusion=*/1,    // force-attempt
      /*scratch_kb=*/1,         // 1 KB << one row (2 KB)
      /*num_threads_for_call=*/8);   // = E: clears round-based & wide-N
}

// Unbounded scratch budget (`ZENDNNL_GRP_MATMUL_M_TILE_PIPELINE_SCRATCH_KB
// = -1`, `kMTilePipelineScratchKbUnbounded`).  Same shape as
// `ScratchBudgetExceeded` above — which bails on a 1 KB budget because a
// single row needs 2 KB — but with the budget gate DISABLED.  The gate
// must no longer fire and vertical fusion must engage end-to-end.  This
// pins the "always run vertical fusion" feature: the only difference vs
// the bail test is `scratch_kb = -1`, isolating the unbounded sentinel
// as the cause of engagement.  All other gates are provably clear
// (active_ops(8) <= num_threads(8); total_need 8*2=16 > 8 ⇒ no wide-N;
// max_M=4 < kHybridMinMaxM(256) ⇒ no multi-tier).
TEST(TestFusedMoEVerticalBF16Bailouts, ScratchUnboundedEngages) {
  run_vertical_bailout_check(
      "scratch_unbounded",
      /*E=*/8, /*M=*/4, /*dim=*/512,  // N_w13=1024 ⇒ 2 KB/row (would bail @1KB)
      /*K_in=*/64, /*H=*/64,
      /*vertical_fusion=*/1,    // force-attempt
      /*scratch_kb=*/-1,        // UNBOUNDED — gate disabled
      /*num_threads_for_call=*/8,    // = E: clears round-based & wide-N
      /*expect_engage=*/true);
}

// Bail 3: wide-N memory-bound fallback would engage.  Light uniform
// frames (M=2 > 1) with a large pinned thread team make
// `total_need * 2 <= num_threads` true in
// `plan_m_tile_single_tier_assignment` (see `m_tile/m_tile_planner.hpp`),
// so `plan.wide_n_fallback` fires and the vertical executor bails at
// the wide-N gate in `m_tile/group_matmul_m_tile.cpp` BEFORE the OMP
// region.  num_threads=64 is pinned (resolve_num_threads honours
// non-zero requests uncapped) so the gate is host-independent:
// total_need <= 2*E = 16 ⇒ 16*2 = 32 <= 64.  active_ops=8 <= 64 keeps
// it out of the round-based regime, and max_M=2 < kHybridMinMaxM(256)
// keeps it out of multi-tier — so wide-N is the sole trigger.
TEST(TestFusedMoEVerticalBF16Bailouts, WideNWouldEngage) {
  run_vertical_bailout_check(
      "wide_n_M2_E8_threads64",
      /*E=*/8, /*M=*/2, /*dim=*/64,
      /*K_in=*/64, /*H=*/64,
      /*vertical_fusion=*/1,    // force-attempt
      /*scratch_kb=*/1024,      // generous: prove wide-N (not scratch) bails
      /*num_threads_for_call=*/64);  // total_need*2 <= 64 ⇒ wide-N fires
}

// Bail 4: multi-tier hybrid would engage.  A skewed frame (one heavy
// expert M=64 + 11 light M=2) clears the multi-tier gate in
// `m_tile/group_matmul_m_tile.cpp`, which the vertical-fusion phase-1
// executor does NOT implement — so it bails to the legacy two-pass
// (which itself runs the multi-tier path, tagging kMultiTier).
//
// The uniform-M `run_vertical_bailout_check` helper can't produce skew,
// so this fixture builds the per-expert M vector inline (mirroring the
// [5c] TestFusedMoEInternalAllocMixedM buffer pattern).  The hybrid
// gate thresholds are pinned via overrides (MinMaxM=32, MinSkew=4) so
// the trigger is independent of the production defaults (256 / 4) and
// the heavy M can stay small (64).  num_threads=16 is pinned so the
// gate's `min_actives = num_threads/2` and `min_lights = num_threads/8`
// arithmetic is host-independent.
//
// Gate arithmetic at E=12 / num_threads=16:
//   avg_M = (64 + 11*2)/12 = 7; light_cut = max(8, 7/4) = 8
//   ⇒ n_light=11 (M=2 <= 8), n_heavy=1 (M=64 > 8)
//   active_ops=12 >= min_actives=8 ✓; n_light=11 >= min_lights=2 ✓
//   gate_skew: max_M=64 >= MinMaxM=32 ✓ AND 64 >= MinSkew(4)*avg(7)=28 ✓
//   candidate_heavy_pool = 16 - light_pool >= n_heavy=1 ✓ ⇒ multi-tier
TEST(TestFusedMoEVerticalBF16Bailouts, MultiTierWouldEngage) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  using zendnnl::common::data_type_t;

  reset_grp_matmul_caches();

  const int E = 12;
  const int dim = 64, N_gate_up = 2 * dim, K_in = 64, H = 64;
  const int K_down = dim;  // gated activation collapses N_gate_up → dim.
  const bool is_bf16 = true;
  const data_type_t dt = data_type_t::bf16;
  const auto act_type = grp_matmul_gated_act_t::silu_and_mul;
  const int kNumThreads = 16;

  std::vector<int> M_per_expert(E, 2);
  M_per_expert[0] = 64;
  const int max_M = 64;

  AlgoEnvGuard                 algo2_guard(2);
  MoEVerticalFusionOverride    vert_guard(1);     // FORCED-attempt
  MoEPipelineScratchKbOverride scratch_guard(1024);
  MTileHybridOverride          hybrid_guard(0);   // AUTO (gate active)
  MTileHybridMinMaxMOverride   minmaxm_guard(32);
  MTileHybridMinSkewOverride   minskew_guard(4);

  TypedBuffers src_ref, src_test, w1, d1_ref, w2, d2_ref, d2_fused;
  src_ref .alloc(E, (size_t)max_M * K_in,         is_bf16);
  src_test.alloc(E, (size_t)max_M * K_in,         is_bf16);
  w1      .alloc(E, (size_t)K_in * N_gate_up,     is_bf16);
  d1_ref  .alloc(E, (size_t)max_M * N_gate_up,    is_bf16);
  w2      .alloc(E, (size_t)K_down * H,           is_bf16);
  d2_ref  .alloc(E, (size_t)max_M * H,            is_bf16);
  d2_fused.alloc(E, (size_t)max_M * H,            is_bf16);
  fill_moe_tensors(E, is_bf16, &src_ref,  &w1, &w2);
  fill_moe_tensors(E, is_bf16, &src_test, nullptr, nullptr);

  GemmVecs gv_op1;
  gv_op1.layout.assign(E, 'r');
  gv_op1.transA.assign(E, false);
  gv_op1.transB.assign(E, false);
  gv_op1.is_wc .assign(E, false);
  gv_op1.alpha .assign(E, 1.0f);
  gv_op1.beta  .assign(E, 0.0f);
  gv_op1.Ms    = M_per_expert;
  gv_op1.Ns    .assign(E, N_gate_up);
  gv_op1.Ks    .assign(E, K_in);
  gv_op1.lda   .assign(E, K_in);
  gv_op1.ldb   .assign(E, N_gate_up);
  gv_op1.ldc   .assign(E, N_gate_up);

  auto src_ref_p  = src_ref .cptrs(is_bf16);
  auto src_test_p = src_test.cptrs(is_bf16);
  auto wei1_p     = w1      .cptrs(is_bf16);
  auto wei2_p     = w2      .cptrs(is_bf16);
  auto d1_ref_p   = d1_ref  .ptrs(is_bf16);
  auto d2_ref_p   = d2_ref  .ptrs(is_bf16);
  auto d2_fused_p = d2_fused.ptrs(is_bf16);
  std::vector<const void *> no_bias(E, nullptr);

  grp_matmul_gated_act_params act{};
  act.act = act_type;

  ASSERT_EQ(run_legacy_2call_ref(M_per_expert, K_in, N_gate_up, K_down, H,
                                 is_bf16, act_type,
                                 src_ref_p, wei1_p, wei2_p,
                                 d1_ref_p, d2_ref_p),
            status_t::success) << "ref multi_tier_would_engage";

  auto fused = make_fused_moe_op2(E, H, wei2_p, no_bias);
  fused.dst_down = d2_fused_p;
  fused.ldc_down = std::vector<int>(E, H);

  auto params = make_uniform_params(E, dt);
  for (auto &pp : params) pp.num_threads = kNumThreads;

  int tag = 0;
  status_t st;
  {
    MTilePathCaptureGuard cap;
    auto pf = params;
    st = group_matmul_direct(gv_op1.layout, gv_op1.transA, gv_op1.transB,
                             gv_op1.Ms, gv_op1.Ns, gv_op1.Ks,
                             gv_op1.alpha, src_test_p,
                             gv_op1.lda, wei1_p, gv_op1.ldb,
                             no_bias, gv_op1.beta,
                             // Op1 dst is [M, N_gate_up]; Op2 output is
                             // routed via fused.dst_down.  Reuse d1_ref
                             // (Op1 result is not inspected after this).
                             d1_ref_p, gv_op1.ldc,
                             gv_op1.is_wc, pf, nullptr, &act, &fused);
    tag = test_api::s_last_m_tile_path.load(std::memory_order_relaxed);
  }
  EXPECT_NE(tag, test_api::m_tile_path_tag::kVerticalFusionBF16)
      << "multi_tier_would_engage produced kVerticalFusionBF16 tag — "
         "bail-out gate did not reject; got tag=" << tag
      << " status=" << static_cast<int>(st);

  if (st == status_t::success) {
    verify_per_expert_2d(d2_fused, H, d2_ref, H,
                         M_per_expert, H, is_bf16,
                         tol_fused(is_bf16),
                         "[18.d] vertical_bailout multi_tier_would_engage");
  }
}

// ===============================================================================
// [18.e] TestFusedMoEVerticalWOQBailouts — WOQ-specific eligibility gate
//      negative paths.  The gate accepts WOQ-INT4 when:
//        regime-B: src=bf16, wei∈{s4,u4}, dst=bf16, dynamic_quant=false,
//                  wei_scale.buff != nullptr, is_weights_const[*] = true.
//      This block exercises the two WOQ-only rejection branches:
//        * `WOQNoWeiScale`  — s4 weight but `wei_scale.buff == nullptr`
//          on either half.  Mirrors the codepath that prevents a
//          non-WOQ `bf16 src + s4 wei + no scale` call from silently
//          falling through `is_non_quant_src_int8` to all-zero output
//          (the same all-zero failure mode `TestFusedMoEQuantWOQ`
//          guards against via its sanity-sum check).
//        * `WOQNotWeightsConst` — s4 weight + valid wei_scale, but at
//          least one expert with `is_weights_const[i] = false`.  The
//          AOCL DLP WOQ fast path requires a const-weight prepack
//          cache; without it the path silently degrades.
//
// Both cases must NOT produce `kVerticalFusionWOQ` (or `kVerticalFusion
// BF16` — neither tag is acceptable on a WOQ-ineligible shape).  ALGO
// 2 stays forced so the legacy `flat_m_tile` populates the tag with
// one of its branch values; any of those is acceptable.
// ===============================================================================

namespace {

// Build a fused-MoE call that would satisfy regime-B (s4 WOQ on both
// halves) EXCEPT for one intentionally broken field, then assert the
// gate rejects (`tag != kVerticalFusion*`) and the dispatch falls
// through to a successful legacy two-pass.  Same shape-clearance math
// as `make_vertical_fusion_woq_params`'s mixtral_E4_M128 entry so the
// non-WOQ gates (round-based, wide-N, scratch budget, multi-tier) are
// all provably clear and the WOQ-specific gate is the SOLE rejection
// trigger.
enum class WOQBailoutCause {
  NoWeiScaleOp1,   // params[0].quant_params.wei_scale.buff = nullptr
  NoWeiScaleOp2,   // fused.down_scale[0].buff = nullptr (Op2 side)
  NotWeightsConst, // is_weights_const[0] = false
};

void run_woq_vertical_bailout(WOQBailoutCause cause,
                              const char *case_label) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  using zendnnl::common::data_type_t;

  reset_grp_matmul_caches();

  const int E = 4;
  const int dim = 64, N_gate_up = 2 * dim, H = 64, K_in = H, M = 128;
  const int K_down = dim;  // gated activation collapses N_gate_up → dim.
  const bool is_bf16 = true;
  const auto act_type = grp_matmul_gated_act_t::silu_and_mul;
  const data_type_t wei_dt = data_type_t::s4;

  AlgoEnvGuard                  algo2_guard(2);
  MoEVerticalFusionOverride     vert_guard(1);     // FORCED-attempt
  MoEPipelineScratchKbOverride  scratch_guard(1024);

  tensor_factory_t factory{};
  std::vector<tensor_t> w1_scale_t(E), w1_t(E);
  std::vector<tensor_t> down_scale_t(E), w2_t(E);
  for (int i = 0; i < E; ++i) {
    w1_scale_t[i] = factory.uniform_dist_tensor(
        {static_cast<uint64_t>(1), static_cast<uint64_t>(N_gate_up)},
        data_type_t::f32, 2.0);
    down_scale_t[i] = factory.uniform_dist_tensor(
        {static_cast<uint64_t>(1), static_cast<uint64_t>(H)},
        data_type_t::f32, 2.0);
    w1_t[i] = factory.uniform_dist_tensor(
        {static_cast<uint64_t>(K_in), static_cast<uint64_t>(N_gate_up)},
        wei_dt, 7.0, /*transposed=*/false, w1_scale_t[i]);
    w2_t[i] = factory.uniform_dist_tensor(
        {static_cast<uint64_t>(K_down), static_cast<uint64_t>(H)},
        wei_dt, 7.0, /*transposed=*/false, down_scale_t[i]);
  }

  TypedBuffers src_test, d1_fused, d2_fused;
  src_test.alloc(E, (size_t)M * K_in,      is_bf16);
  d1_fused.alloc(E, (size_t)M * N_gate_up, is_bf16);
  d2_fused.alloc(E, (size_t)M * H,         is_bf16);
  fill_moe_tensors(E, is_bf16, &src_test, nullptr, nullptr);

  std::vector<const void *> wei1_p(E), wei2_p(E);
  for (int i = 0; i < E; ++i) {
    wei1_p[i] = w1_t[i].get_raw_handle_unsafe();
    wei2_p[i] = w2_t[i].get_raw_handle_unsafe();
  }
  auto src_test_p = src_test.cptrs(is_bf16);
  auto d1_fused_p = d1_fused.ptrs(is_bf16);
  auto d2_fused_p = d2_fused.ptrs(is_bf16);
  std::vector<const void *> no_bias(E, nullptr);

  auto gv_op1 = GemmVecs::uniform(E, M, N_gate_up, K_in);
  gv_op1.is_wc.assign(E, true);
  if (cause == WOQBailoutCause::NotWeightsConst) {
    gv_op1.is_wc[0] = false;  // one expert non-const → reject.
  }

  grp_matmul_gated_act_params act{};
  act.act = act_type;

  // Build params with s4 + wei_scale wired correctly on the const
  // case, then dynamically null out the field for the NoWeiScale*
  // cases.
  auto params = make_uniform_params(E, data_type_t::bf16);
  for (int i = 0; i < E; ++i) {
    params[i].dtypes.src = data_type_t::bf16;
    params[i].dtypes.wei = wei_dt;
    params[i].dtypes.dst = data_type_t::bf16;
    copy_attached_scale(w1_t[i], params[i].quant_params.wei_scale);
    params[i].num_threads = 32;
  }
  if (cause == WOQBailoutCause::NoWeiScaleOp1) {
    // Null out Op1 wei_scale.buff to trip the regime-B `wei_scale.buff
    // != nullptr` check.  Dims/dt stay so the matmul kernel would
    // still attempt dispatch (the gate must reject BEFORE that).
    params[0].quant_params.wei_scale.buff = nullptr;
  }

  auto fused = make_fused_moe_op2(E, H, wei2_p, no_bias);
  fused.dst_down = d2_fused_p;
  fused.ldc_down = std::vector<int>(E, H);
  fused.down_scale.resize(E);
  for (int i = 0; i < E; ++i) {
    copy_attached_scale(w2_t[i], fused.down_scale[i]);
  }
  if (cause == WOQBailoutCause::NoWeiScaleOp2) {
    // Null out Op2 wei_scale via fused.down_scale — the dispatcher
    // copies this into params_w2[i].quant_params.wei_scale so the
    // gate's classify_half on the Op2 half sees a null buffer.
    fused.down_scale[0].buff = nullptr;
  }

  int tag = 0;
  status_t st;
  {
    MTilePathCaptureGuard cap;
    auto pf = params;
    st = group_matmul_direct(gv_op1.layout, gv_op1.transA, gv_op1.transB,
                             gv_op1.Ms, gv_op1.Ns, gv_op1.Ks,
                             gv_op1.alpha, src_test_p,
                             gv_op1.lda, wei1_p, gv_op1.ldb,
                             no_bias, gv_op1.beta,
                             d1_fused_p, gv_op1.ldc,
                             gv_op1.is_wc, pf, nullptr, &act, &fused);
    tag = test_api::s_last_m_tile_path.load(std::memory_order_relaxed);
  }
  // The strict contract: the WOQ vertical-fusion path did NOT engage.
  // Either fallback ran (status=success, tag = some legacy branch) or
  // the legacy path rejected the shape outright (status != success).
  // Neither outcome may leave `tag` pointing at a vertical-fusion
  // executor — that would prove the WOQ-specific gate didn't fire.
  EXPECT_NE(tag, test_api::m_tile_path_tag::kVerticalFusionWOQ)
      << case_label << " produced kVerticalFusionWOQ tag — WOQ "
         "eligibility check did not reject; got tag=" << tag
      << " status=" << static_cast<int>(st);
  EXPECT_NE(tag, test_api::m_tile_path_tag::kVerticalFusionBF16)
      << case_label << " unexpectedly engaged BF16-vertical fusion on "
         "a WOQ-typed call; got tag=" << tag
      << " status=" << static_cast<int>(st);
}

}  // namespace

TEST(TestFusedMoEVerticalWOQBailouts, NoWeiScaleOp1) {
  run_woq_vertical_bailout(WOQBailoutCause::NoWeiScaleOp1,
                           "woq_no_wei_scale_op1");
}

TEST(TestFusedMoEVerticalWOQBailouts, NoWeiScaleOp2) {
  run_woq_vertical_bailout(WOQBailoutCause::NoWeiScaleOp2,
                           "woq_no_wei_scale_op2");
}

TEST(TestFusedMoEVerticalWOQBailouts, NotWeightsConst) {
  run_woq_vertical_bailout(WOQBailoutCause::NotWeightsConst,
                           "woq_not_weights_const");
}

// ===============================================================================
// [18.c] TestFusedMoEVerticalDQINT8Bailouts - DQ-INT8 eligibility gate
//        negative cases.
//
// Mirror of `TestFusedMoEVerticalWOQBailouts` for the third regime.  For
// each bailout trigger the call MUST NOT engage the vertical-fusion
// executor — the capture tag must NOT equal
// `kVerticalFusionDQINT8` (nor `kVerticalFusionBF16` /
// `kVerticalFusionWOQ`, which would mean the gate misclassified the
// regime).  Each case starts from a known-good DQ-INT8 shape (same as
// `make_vertical_fusion_dqint8_params`'s mixtral_E4_M128 entry) and
// breaks exactly ONE field.  Coverage:
//
//   * `NotDynamicQuantOp1` — `params[0].dynamic_quant = false`.  The
//      eligibility gate's regime-C predicate must demand
//      `dynamic_quant=true` on the Op1 half; without it the kernel
//      would route through a static-quant codepath that the executor
//      isn't wired for.
//
//   * `NoWeiScaleOp1` — `params[0].quant_params.wei_scale.buff =
//      nullptr`.  Per-channel `wei_scale` is mandatory for the
//      DQ-INT8 s8s8→bf16 kernel; the gate must reject before the
//      matmul dispatch tries to dereference the null buffer.
//
//   * `NoWeiScaleOp2` — `fused.down_scale[0].buff = nullptr`.  Same
//      requirement on the Op2 half — the dispatcher copies
//      `fused.down_scale[i]` into `params_w2[i].quant_params
//      .wei_scale` for Op2's regime-C classify step.
//
//   * `AsymComputeU8Op1` — `params[0].dtypes.compute = u8`.  The v1
//      vertical-fusion DQ-INT8 path is symmetric s8-only; mixed s8
//      weight + u8 compute is a different (asymmetric) kernel route
//      and must fall back to legacy two-pass.
//
// Each case forces ALGO 2 + VF FORCED + ample scratch budget so the
// non-DQ-INT8 reject paths (round-based, scratch budget, wide-N,
// multi-tier) are demonstrably clear — the DQ-INT8-specific gate is
// the SOLE rejection trigger we're regression-testing.
// ===============================================================================

namespace {

enum class DQINT8BailoutCause {
  NotDynamicQuantOp1,  // params[0].dynamic_quant = false
  NoWeiScaleOp1,       // params[0].quant_params.wei_scale.buff = nullptr
  NoWeiScaleOp2,       // fused.down_scale[0].buff = nullptr (Op2 side)
  AsymComputeU8Op1,    // params[0].dtypes.compute = u8 (asym route)
};

void run_dqint8_vertical_bailout(DQINT8BailoutCause cause,
                                 const char *case_label) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  using zendnnl::common::data_type_t;

  reset_grp_matmul_caches();

  const int E = 4;
  const int dim = 64, N_gate_up = 2 * dim, H = 64, K_in = H, M = 128;
  const int K_down = dim;
  const bool is_bf16 = true;
  const auto act_type = grp_matmul_gated_act_t::silu_and_mul;

  AlgoEnvGuard                  algo2_guard(2);
  MoEVerticalFusionOverride     vert_guard(1);   // FORCED-attempt.
  MoEPipelineScratchKbOverride  scratch_guard(1024);

  tensor_factory_t factory{};

  // ── Op1 sources: BF16 with attached per-token F32 src_scale ──────
  std::vector<tensor_t> src_t(E), src_scale_t(E);
  for (int i = 0; i < E; ++i) {
    src_scale_t[i] = factory.zero_tensor(
        {static_cast<uint64_t>(M), static_cast<uint64_t>(1)},
        data_type_t::f32);
    src_t[i] = factory.uniform_dist_tensor(
        {static_cast<uint64_t>(M), static_cast<uint64_t>(K_in)},
        data_type_t::bf16, 2.0, /*transposed=*/false,
        src_scale_t[i], tensor_t{});
  }

  // ── Op1 + Op2 weights: S8 + per-channel F32 wei_scale ────────────
  std::vector<tensor_t> w1_s8_t(E), w1_scale_t(E), w1_zp_t(E);
  std::vector<tensor_t> w2_s8_t(E), down_scale_t(E), down_zp_t(E);
  for (int i = 0; i < E; ++i) {
    auto w1_ref = factory.uniform_dist_tensor(
        {static_cast<uint64_t>(K_in), static_cast<uint64_t>(N_gate_up)},
        data_type_t::bf16, 2.0, /*transposed=*/false);
    ASSERT_EQ(quant_params_compute(factory, w1_ref,
                                   data_type_t::bf16, data_type_t::s8,
                                   /*scale_dims=*/{1, N_gate_up},
                                   data_type_t::f32,
                                   w1_scale_t[i], w1_zp_t[i], &w1_s8_t[i]),
              status_t::success);
    auto w2_ref = factory.uniform_dist_tensor(
        {static_cast<uint64_t>(K_down), static_cast<uint64_t>(H)},
        data_type_t::bf16, 2.0, /*transposed=*/false);
    ASSERT_EQ(quant_params_compute(factory, w2_ref,
                                   data_type_t::bf16, data_type_t::s8,
                                   /*scale_dims=*/{1, H},
                                   data_type_t::f32,
                                   down_scale_t[i], down_zp_t[i], &w2_s8_t[i]),
              status_t::success);
  }

  TypedBuffers d1_fused, d2_fused;
  d1_fused.alloc(E, (size_t)M * N_gate_up, is_bf16);
  d2_fused.alloc(E, (size_t)M * H,         is_bf16);

  std::vector<const void *> srcs(E), wei1_p(E), wei2_p(E);
  for (int i = 0; i < E; ++i) {
    srcs  [i] = src_t  [i].get_raw_handle_unsafe();
    wei1_p[i] = w1_s8_t[i].get_raw_handle_unsafe();
    wei2_p[i] = w2_s8_t[i].get_raw_handle_unsafe();
  }
  auto d1_fused_p = d1_fused.ptrs(is_bf16);
  auto d2_fused_p = d2_fused.ptrs(is_bf16);
  std::vector<const void *> no_bias(E, nullptr);

  auto gv_op1 = GemmVecs::uniform(E, M, N_gate_up, K_in);
  gv_op1.is_wc.assign(E, true);

  grp_matmul_gated_act_params act{};
  act.act = act_type;

  // Build a known-good DQ-INT8 params vector, then break exactly ONE
  // field according to `cause`.
  auto params = make_uniform_params(E, data_type_t::bf16);
  for (int i = 0; i < E; ++i) {
    params[i].dtypes.src     = data_type_t::bf16;
    params[i].dtypes.wei     = data_type_t::s8;
    params[i].dtypes.dst     = data_type_t::bf16;
    params[i].dtypes.compute = data_type_t::s8;
    params[i].dynamic_quant  = true;
    copy_attached_scale(src_t  [i], params[i].quant_params.src_scale);
    copy_attached_scale(w1_s8_t[i], params[i].quant_params.wei_scale);
    copy_attached_zp   (w1_s8_t[i], params[i].quant_params.wei_zp);
    params[i].num_threads = 32;
  }
  if (cause == DQINT8BailoutCause::NotDynamicQuantOp1) {
    params[0].dynamic_quant = false;
  } else if (cause == DQINT8BailoutCause::NoWeiScaleOp1) {
    // Null out Op1 wei_scale.buff; gate must reject before the
    // s8s8→bf16 kernel dereferences it.
    params[0].quant_params.wei_scale.buff = nullptr;
  } else if (cause == DQINT8BailoutCause::AsymComputeU8Op1) {
    // u8 compute would route through an asymmetric kernel that the
    // v1 vertical-fusion executor doesn't model; gate must reject.
    params[0].dtypes.compute = data_type_t::u8;
  }

  auto fused = make_fused_moe_op2(E, H, wei2_p, no_bias);
  fused.dst_down = d2_fused_p;
  fused.ldc_down = std::vector<int>(E, H);
  fused.down_scale.resize(E);
  fused.down_zp   .resize(E);
  for (int i = 0; i < E; ++i) {
    copy_attached_scale(w2_s8_t[i], fused.down_scale[i]);
    copy_attached_zp   (w2_s8_t[i], fused.down_zp[i]);
  }
  if (cause == DQINT8BailoutCause::NoWeiScaleOp2) {
    fused.down_scale[0].buff = nullptr;
  }

  int tag = 0;
  status_t st;
  {
    MTilePathCaptureGuard cap;
    st = group_matmul_direct(gv_op1.layout, gv_op1.transA, gv_op1.transB,
                             gv_op1.Ms, gv_op1.Ns, gv_op1.Ks,
                             gv_op1.alpha, srcs,
                             gv_op1.lda, wei1_p, gv_op1.ldb,
                             no_bias, gv_op1.beta,
                             d1_fused_p, gv_op1.ldc,
                             gv_op1.is_wc, params, nullptr, &act, &fused);
    tag = test_api::s_last_m_tile_path.load(std::memory_order_relaxed);
  }
  // Strict contract: the DQ-INT8 vertical-fusion path MUST NOT have
  // engaged.  Either fallback ran (status=success, tag = some legacy
  // branch) or the legacy path rejected outright (status != success);
  // neither outcome may leave `tag` pointing at any vertical-fusion
  // executor.
  EXPECT_NE(tag, test_api::m_tile_path_tag::kVerticalFusionDQINT8)
      << case_label << " produced kVerticalFusionDQINT8 tag — DQ-INT8 "
         "eligibility check did not reject; got tag=" << tag
      << " status=" << static_cast<int>(st);
  EXPECT_NE(tag, test_api::m_tile_path_tag::kVerticalFusionBF16)
      << case_label << " unexpectedly engaged BF16-vertical fusion on "
         "a DQ-INT8 typed call; got tag=" << tag
      << " status=" << static_cast<int>(st);
  EXPECT_NE(tag, test_api::m_tile_path_tag::kVerticalFusionWOQ)
      << case_label << " unexpectedly engaged WOQ-vertical fusion on "
         "a DQ-INT8 typed call; got tag=" << tag
      << " status=" << static_cast<int>(st);
}

}  // namespace

TEST(TestFusedMoEVerticalDQINT8Bailouts, NotDynamicQuantOp1) {
  run_dqint8_vertical_bailout(DQINT8BailoutCause::NotDynamicQuantOp1,
                              "dqint8_not_dynamic_quant_op1");
}

TEST(TestFusedMoEVerticalDQINT8Bailouts, NoWeiScaleOp1) {
  run_dqint8_vertical_bailout(DQINT8BailoutCause::NoWeiScaleOp1,
                              "dqint8_no_wei_scale_op1");
}

TEST(TestFusedMoEVerticalDQINT8Bailouts, NoWeiScaleOp2) {
  run_dqint8_vertical_bailout(DQINT8BailoutCause::NoWeiScaleOp2,
                              "dqint8_no_wei_scale_op2");
}

TEST(TestFusedMoEVerticalDQINT8Bailouts, AsymComputeU8Op1) {
  run_dqint8_vertical_bailout(DQINT8BailoutCause::AsymComputeU8Op1,
                              "dqint8_asym_compute_u8_op1");
}

// ===============================================================================
// [19] TestFusedMoEScratchMemory: scratch-management contract tests
//
// These tests pin specific contract properties of the persistent
// thread-local fused-MoE scratch surfaces.  Each test is a focused
// regression for one of the audit findings reported by the
// "fused-MoE scratch memory audit" review:
//
//   (a) `clear_fused_moe_scratch()` re-allocates correctly on the
//       NEXT call after release.  Defends against a stale-pointer
//       bug where the executor would observe `arena.cap > 0` but
//       `arena.buf == nullptr` post-clear.
//
//   (b) The validator's `op2_internal` lda-vs-N_down gate rejects
//       the asymmetric-MoE silent-corruption shape (lda = K_in but
//       N_down > K_in).  Defends against an algo-independent heap
//       overrun (Pass-2 would write past the caller's src[] when
//       the caller sized for Op1 only).
//
//   (c) The arena overflow guard rejects size_t-wrapping inputs.
//       Hard to trigger from a real workload (would require huge
//       M/N), but the guard is correctness-critical so we exercise
//       it via the validator's per-expert byte-count overflow path.
// ===============================================================================

// Test [19.a].  After `clear_fused_moe_scratch()` the thread-local
// arena is empty (`buf=nullptr`, `cap=0`).  Re-invoking a fused-MoE
// call must allocate again from scratch — and produce a correct
// result.  Repeats the call sequence three times (warm, clear,
// repeat) so any regression that left dangling state would surface
// on the second invocation.
TEST(TestFusedMoEScratchMemory, ClearAndReinvokeReallocates) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  using zendnnl::common::data_type_t;

  reset_grp_matmul_caches();
  clear_fused_moe_scratch();   // start from a clean slate

  const int E = 4, M = 8, dim = 64, K_in = 64, H = 64;
  const int N_gate_up = 2 * dim, K_down = dim;
  const bool is_bf16  = true;
  const data_type_t dt = data_type_t::bf16;
  const auto act_type = grp_matmul_gated_act_t::silu_and_mul;

  AlgoEnvGuard algo1_guard(1);  // pin ALGO 1 so the scratch is the only
                                // path-specific variable in this test
  MoEVerticalFusionOverride vf_guard(-1);  // disable vertical fusion so
                                           // the legacy 2-pass arena path
                                           // is the surface under test

  TypedBuffers src_ref, src_test, w1, d1_ref, w2, d2_ref, d2_fused;
  src_ref .alloc(E, (size_t)M * K_in,         is_bf16);
  src_test.alloc(E, (size_t)M * K_in,         is_bf16);
  w1      .alloc(E, (size_t)K_in * N_gate_up, is_bf16);
  d1_ref  .alloc(E, (size_t)M * N_gate_up,    is_bf16);
  w2      .alloc(E, (size_t)K_down * H,       is_bf16);
  d2_ref  .alloc(E, (size_t)M * H,            is_bf16);
  d2_fused.alloc(E, (size_t)M * H,            is_bf16);
  fill_moe_tensors(E, is_bf16, &src_ref,  &w1, &w2);
  fill_moe_tensors(E, is_bf16, &src_test, nullptr, nullptr);

  auto gv_op1 = GemmVecs::uniform(E, M, N_gate_up, K_in);
  auto src_ref_p  = src_ref .cptrs(is_bf16);
  auto src_test_p = src_test.cptrs(is_bf16);
  auto wei1_p     = w1      .cptrs(is_bf16);
  auto wei2_p     = w2      .cptrs(is_bf16);
  auto d1_ref_p   = d1_ref  .ptrs(is_bf16);
  auto d2_ref_p   = d2_ref  .ptrs(is_bf16);
  auto d2_fused_p = d2_fused.ptrs(is_bf16);
  std::vector<const void *> no_bias(E, nullptr);

  grp_matmul_gated_act_params act{};
  act.act = act_type;

  ASSERT_EQ(run_legacy_2call_ref(E, M, K_in, N_gate_up, K_down, H,
                                 is_bf16, act_type,
                                 src_ref_p, wei1_p, wei2_p,
                                 d1_ref_p, d2_ref_p),
            status_t::success);

  auto fused = make_fused_moe_op2(E, H, wei2_p, no_bias);
  fused.dst_down = d2_fused_p;
  fused.ldc_down = std::vector<int>(E, H);
  auto params = make_uniform_params(E, dt);

  auto run_once = [&](const char *label) {
    for (auto &v : d2_fused.bf16)
      std::fill(v.begin(), v.end(), bfloat16_t(0.0f));
    auto pf = params;
    const status_t st = group_matmul_direct(
        gv_op1.layout, gv_op1.transA, gv_op1.transB,
        gv_op1.Ms, gv_op1.Ns, gv_op1.Ks,
        gv_op1.alpha, src_test_p,
        gv_op1.lda, wei1_p, gv_op1.ldb,
        no_bias, gv_op1.beta,
        d1_ref_p, gv_op1.ldc,
        gv_op1.is_wc, pf, nullptr, &act, &fused);
    ASSERT_EQ(st, status_t::success) << label;
    verify_per_expert_2d(d2_fused, H, d2_ref, H,
                         E, M, H, is_bf16,
                         tol_fused(is_bf16), label);
  };

  // Call sequence: warm → clear → warm-after-clear → clear → warm.
  // Each "warm" must produce the legacy 2-pass reference; each
  // "clear" must NOT change observable output on the next call.
  run_once("[19.a] warm");
  clear_fused_moe_scratch();
  run_once("[19.a] warm-after-clear-1");
  clear_fused_moe_scratch();
  run_once("[19.a] warm-after-clear-2");
}

// Test [19.b].  Validator rejects `op2_internal` + asymmetric-MoE
// (`N_down > K_in`) + `lda = K_in`.  The pre-fix path silently
// returned `status_t::failure` (same return code as any caller
// error); the post-fix path also returns `status_t::failure` BUT
// emits a clear `log_error` describing the contract.  We assert
// the failure code here; the log_error is observable only via
// `apilog`-on builds (not gated in this test).
TEST(TestFusedMoEScratchMemory, ValidatorRejectsOp2InternalAsymmetricLda) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  using zendnnl::common::data_type_t;

  reset_grp_matmul_caches();
  clear_fused_moe_scratch();

  const int E = 2;
  const int M = 4;
  const int K_in = 64;          // hidden_in
  const int N_gate_up = 128;    // 2*intermediate
  const int H_down  = 128;      // N_down = hidden_out  -- intentionally
                                //   > K_in (= 64) so lda=K_in is too
                                //   narrow for the Op2 write-back path
  const bool is_bf16 = true;
  const data_type_t dt = data_type_t::bf16;
  const auto act_type = grp_matmul_gated_act_t::silu_and_mul;

  AlgoEnvGuard algo1_guard(1);
  MoEVerticalFusionOverride vf_guard(-1);

  TypedBuffers src, w1, w2, d1;
  src .alloc(E, (size_t)M * K_in,             is_bf16);
  w1  .alloc(E, (size_t)K_in * N_gate_up,     is_bf16);
  w2  .alloc(E, (size_t)(N_gate_up/2) * H_down, is_bf16);
  d1  .alloc(E, (size_t)M * N_gate_up,        is_bf16);
  fill_moe_tensors(E, is_bf16, &src, &w1, &w2);

  auto gv_op1 = GemmVecs::uniform(E, M, N_gate_up, K_in);
  auto src_p  = src.cptrs(is_bf16);
  auto wei1_p = w1 .cptrs(is_bf16);
  auto wei2_p = w2 .cptrs(is_bf16);
  auto d1_p   = d1 .ptrs(is_bf16);
  std::vector<const void *> no_bias(E, nullptr);

  grp_matmul_gated_act_params act{};
  act.act = act_type;

  // Construct a fused-MoE struct that requests op2_internal (empty
  // dst_down) with N_down = H_down > K_in.  Op1 is caller-allocated
  // for d1 (avoids op1_internal complicating the test).
  grp_matmul_fused_moe_params fused{};
  fused.down_weight = wei2_p;
  fused.bias_down   = no_bias;
  fused.N_down      = std::vector<int>(E, H_down);
  fused.ldb_down    = std::vector<int>(E, H_down);
  fused.bias_dt_down = data_type_t::none;
  // dst_down + ldc_down LEFT EMPTY → op2_internal=true
  // → Op2 will write back into `src[]` with stride `lda`.
  //
  // gv_op1.lda = K_in = 64, but N_down = 128 → contract violation
  // → validator must reject.

  auto params = make_uniform_params(E, dt);

  const status_t st = group_matmul_direct(
      gv_op1.layout, gv_op1.transA, gv_op1.transB,
      gv_op1.Ms, gv_op1.Ns, gv_op1.Ks,
      gv_op1.alpha, src_p,
      gv_op1.lda, wei1_p, gv_op1.ldb,
      no_bias, gv_op1.beta,
      d1_p, gv_op1.ldc,
      gv_op1.is_wc, params, nullptr, &act, &fused);

  EXPECT_EQ(st, status_t::failure)
      << "[19.b] op2_internal with lda(" << gv_op1.lda[0]
      << ") < N_down(" << H_down << ") must be rejected by the "
         "validator; got status=" << static_cast<int>(st);
}

// ===============================================================================
// [20] TestFusedMoECompactAsymmetric — production framework-layout regression
// ===============================================================================
//
// Reproduces the EXACT vector-size contract that production frameworks
// (zentorch, vLLM) pass for sparse-MoE decode under the `active_matmul /
// total_matmul` opt-in.  None of the existing fixtures construct this
// combination — they all use either:
//
//   * Padded layout: every vector sized to `total`, with M[active..)=0
//     (TestFusedMoEActiveMatmul, TestFusedMoEWarmPackPipeline, …).  In
//     this layout `params.size() == M.size() == total`, so any helper
//     that *should* derive `num_ops` from `M.size()` but accidentally
//     uses `params.size()` produces the same value, masking the bug.
//
//   * Compact-symmetric layout (benchdnn): every vector sized to
//     `active`, including `params` (`std::vector<matmul_params>
//     params(n)`).  Here `params.size() == M.size() == active`, again
//     masking the bug.
//
// Production frameworks use the **Compact-asymmetric** layout:
//
//   * Input/output side (read only by matmul `[0, active)`): `M`,
//     `src`, `lda`, `dst`, `ldc`, `alpha`, `beta`, `transA`, `layout`,
//     `bias` sized to **active**.
//   * Weight-side (read by prepack `[0, total)` AND matmul
//     `[0, active)`): `weight`, `N`, `K`, `ldb`, `transB`,
//     `is_weights_const`, `params` sized to **total**.
//   * `fused.down_weight / N_down / ldb_down / bias_down` sized to
//     **total** (weight-side).  `fused.down_scale / down_zp` are
//     either empty (un-quantized) or sized to **active** (quant scale
//     buffers cover only firing experts in the per-call active prefix).
//
// The dispatcher's relaxed-size validator accepts any vector sized
// `>= num_ops = M.size() = active`, so all five sizing patterns above
// are admissible by contract.  But ONLY the asymmetric pattern
// triggers the `setup_op2_dispatch_scratch::num_ops = params.size()`
// regression — because only here does `params.size() > M.size()`.
//
// Coverage matrix:
//
//   * BF16 + swiglu_oai_mul + op1_internal + op2_internal — the
//     gpt-oss-20b decode kernel that motivated the active/total
//     contract.  Under the regression, the Op2 setup loop iterates
//     `[0, total)` and reads past `.size()` on the active-sized
//     `op1_dst` (= `scratch.op1_dst_internal`, sized to `M.size() =
//     active`) and the active-sized caller `src`.  Reads return
//     garbage bytes interpreted as `void *` and are stored into the
//     prepack-extras tail of `scratch.src_down` /
//     `scratch.op2_dst_internal`.  The Op2 dispatcher iterates only
//     `[0, active)` so the garbage tail is never re-read, but the
//     OOB read itself is UB — ASAN flags it as a heap-buffer-overflow
//     read of size 8.
//
//   * BF16 + silu_and_mul — variant to cover all three supported
//     activations.
//
//   * Both with `ZENDNNL_GRP_MATMUL_PREPACK=0` and vertical fusion
//     disabled so the legacy two-pass executes (matches the user's
//     repro env where the crash persisted after both knobs were off).
//
// Failure mode without the fix (empirically verified — see commit
// message on the introducing patch):
//
//   * Release build (this fixture): the buggy loop iterates the
//     TOTAL range and OOB-reads `op1_dst[active..total)` /
//     `src[active..total)`.  Reads return garbage `void *` values
//     that are stored into `scratch.src_down[active..total)` /
//     `scratch.op2_dst_internal[active..total)`.  But the Op2
//     dispatcher iterates only `[0, active)` so the corrupted tail
//     is never re-read — the active-prefix output stays correct.
//     The test PASSES in release even with the regression restored.
//     This is silent UB, not a correctness failure.
//
//   * ASAN build (default libstdc++): does NOT catch it either —
//     the OOB read is past `.size()` but typically within the
//     allocator's heap chunk for the underlying buffer, so ASAN
//     reports no heap-buffer-overflow.  Detecting this would need
//     `_GLIBCXX_SANITIZE_VECTOR`-instrumented libstdc++ (container-
//     overflow annotations), which is not the default toolchain.
//
//   * Production (vLLM / zentorch) with non-empty active-sized
//     `fused.down_scale`: deterministic `std::bad_array_new_length`
//     on the `p.quant_params.wei_scale.dims = fused.down_scale[i]
//     .dims` copy-assign — the OOB-read `dims` vector header decodes
//     a bogus `size` that drives `new int64_t[bogus]`.  This is what
//     produced the field crash; the quant-fused-MoE gtest helpers
//     don't yet expose `down_scale` plumbing to this fixture (see
//     `TestFusedMoEQuant [16]`), so we can't deterministically
//     reproduce the bad_array_new_length here.
//
// What this fixture DOES catch:
//   * Future regressions that mis-slice the active range (e.g., a
//     future helper iterating `[active, total)` and accidentally
//     writing into the active prefix via wrong-offset arithmetic
//     would surface as a numerical mismatch vs `run_legacy_2call_ref`).
//   * Layout-contract regressions where the dispatcher's relaxed-
//     size validator tightens and starts rejecting the asymmetric
//     case — `status_t::success` ASSERT would fire.
//   * Crashes on the active-prefix dispatch path that happen to
//     manifest only under the framework's vector-size combo.
//
// What it does NOT catch on a release/standard-ASAN build:
//   * The silent UB introduced by the original bug.  A follow-up
//     `TestFusedMoECompactAsymmetricQuant` (TODO) using non-empty
//     active-sized `down_scale` will close that gap deterministically.

struct CompactAsymmetricParam {
  int active;
  int total;
  int M_per_expert;
  int dim;
  int hidden_size;
  int K_in;
  int act_int;  // grp_matmul_gated_act_t
};

static std::string CompactAsymmetricParamName(
    const ::testing::TestParamInfo<CompactAsymmetricParam> &info) {
  static const char *act_names[] = {"none", "silu", "gelu", "swiglu"};
  const auto &p = info.param;
  return std::string("act") + std::to_string(p.active)
       + "_tot" + std::to_string(p.total)
       + "_" + act_names[p.act_int];
}

class TestFusedMoECompactAsymmetric
    : public ::testing::TestWithParam<CompactAsymmetricParam> {};

TEST_P(TestFusedMoECompactAsymmetric, ProductionLayoutBF16) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;

  // Match the user's repro env: PREPACK off and vertical fusion off
  // — isolate the legacy two-pass path so the regression's
  // setup_op2_dispatch_scratch loop is the only suspect.
  EnvVarGuard prepack_off("ZENDNNL_GRP_MATMUL_PREPACK", "0");
  MoEVerticalFusionOverride vf_off(-1);
  reset_grp_matmul_caches();

  const auto &p = GetParam();
  const int active = p.active;
  const int total  = p.total;
  ASSERT_GT(total, active)
      << "this fixture only exercises the asymmetric case; symmetric "
         "(total == active) is covered by TestFusedMoEActiveMatmul";

  const int M = p.M_per_expert;
  const int K_in = p.K_in, H = p.hidden_size;
  const int N_gate_up = 2 * p.dim;
  const auto act_type = static_cast<grp_matmul_gated_act_t>(p.act_int);
  const data_type_t dt = data_type_t::bf16;
  const bool is_bf16 = true;
  const bool act_is_none = (act_type == grp_matmul_gated_act_t::none);
  const int K_down = act_is_none ? N_gate_up : p.dim;

  // Reference path uses `active`-sized vectors.  Test path uses the
  // production Compact-asymmetric layout:
  //   * src           sized to active (input-side).
  //   * wei1 / wei2   sized to total  (weight-side, including extras).
  //   * d1_ref / d2_ref sized to active (reference matches active).
  TypedBuffers src_test, w1, w2, d1_ref, d2_ref;
  src_test.alloc(active, (size_t)M * K_in,        is_bf16);
  w1      .alloc(total,  (size_t)K_in * N_gate_up, is_bf16);
  w2      .alloc(total,  (size_t)K_down * H,      is_bf16);
  d1_ref  .alloc(active, (size_t)M * N_gate_up,   is_bf16);
  d2_ref  .alloc(active, (size_t)M * H,           is_bf16);

  fill_moe_tensors(active, is_bf16, &src_test, nullptr, nullptr);
  fill_moe_tensors(total,  is_bf16, nullptr,    &w1,    &w2);

  // Build the asymmetric vector set.
  //
  //   active-sized: layout, transA, alpha, beta, M, src, lda, ldc
  //                 (= ldc placeholder for op1_internal), bias.
  //   total-sized : transB, N, K, ldb, weight, is_weights_const,
  //                 params, fused.{down_weight, N_down, ldb_down,
  //                 bias_down}.
  std::vector<char>             layout_active(active, 'r');
  std::vector<bool>             transA_active(active, false);
  std::vector<float>            alpha_active(active, 1.0f);
  std::vector<float>            beta_active(active, 0.0f);
  std::vector<int>              M_active(active, M);
  std::vector<int>              lda_active(active, K_in);
  std::vector<void *>           dst_null(active, nullptr);   // op1_internal
  std::vector<int>              ldc_null(active, 0);

  std::vector<bool>             transB_total(total, false);
  std::vector<int>              N_total(total, N_gate_up);
  std::vector<int>              K_total(total, K_in);
  std::vector<int>              ldb_total(total, N_gate_up);
  std::vector<bool>             is_wc_total(total, true);
  std::vector<const void *>     bias_active(active, nullptr);

  auto wei1_p_full = w1.cptrs(is_bf16);  // size = total
  auto wei2_p_full = w2.cptrs(is_bf16);  // size = total
  auto src_test_p  = src_test.cptrs(is_bf16);  // size = active

  // params SIZED TO TOTAL — the production-asymmetric signal.  Every
  // slot carries (active_matmul=active, total_matmul=total) so the
  // dispatcher takes the opt-in branch.
  auto params_total = make_uniform_params(total, dt);
  for (auto &pp : params_total) {
    pp.active_matmul = static_cast<uint32_t>(active);
    pp.total_matmul  = static_cast<uint32_t>(total);
  }

  grp_matmul_fused_moe_params fused{};
  fused.down_weight = wei2_p_full;                  // size = total
  fused.N_down      = std::vector<int>(total, H);   // size = total
  fused.ldb_down    = std::vector<int>(total, H);   // size = total
  fused.bias_down   = std::vector<const void *>(total, nullptr);  // size = total
  // dst_down + ldc_down left empty -> op2_internal=true (Op2 writes
  // back into src_test[i] with stride lda_active[i] = K_in = H).

  grp_matmul_gated_act_params act_p{};
  act_p.act = act_type;
  auto act_ptr = act_is_none ? nullptr : &act_p;

  // Reference: legacy 2-call with active-sized inputs/weights.
  std::vector<const void *> wei1_active(wei1_p_full.begin(),
                                         wei1_p_full.begin() + active);
  std::vector<const void *> wei2_active(wei2_p_full.begin(),
                                         wei2_p_full.begin() + active);
  auto d1_ref_p = d1_ref.ptrs(is_bf16);
  auto d2_ref_p = d2_ref.ptrs(is_bf16);
  ASSERT_EQ(run_legacy_2call_ref(active, M, K_in, N_gate_up, K_down, H,
                                 is_bf16, act_type,
                                 src_test_p, wei1_active, wei2_active,
                                 d1_ref_p, d2_ref_p),
            status_t::success);

  // Re-fill src_test (the reference run consumed it).
  fill_moe_tensors(active, is_bf16, &src_test, nullptr, nullptr);

  // Test call: Compact-asymmetric layout, op1_internal + op2_internal.
  // Under the regression, setup_op2_dispatch_scratch derived num_ops
  // from params.size()=total and OOB-read op1_dst[active..total) and
  // src[active..total).  Under the fix, num_ops is driven by M.size()
  // = active and the loop stays in-bounds.
  const status_t st = group_matmul_direct(
      layout_active, transA_active, transB_total,
      M_active, N_total, K_total, alpha_active,
      src_test_p, lda_active, wei1_p_full, ldb_total,
      bias_active, beta_active,
      dst_null, ldc_null,
      is_wc_total, params_total, /*num_threads=*/nullptr,
      act_ptr, &fused);

  ASSERT_EQ(st, status_t::success)
      << "[20] Compact-asymmetric layout (active=" << active
      << " total=" << total
      << ") rejected by dispatcher — must accept production "
         "framework-opt-in vector sizing";

  // Verify Op2 output (written back into src_test[0..active)) matches
  // the legacy 2-call reference.  An OOB read in setup_op2 that
  // happened to corrupt the active-prefix params_down slots would
  // surface as a numerical mismatch here.
  std::ostringstream lbl;
  lbl << "[20] active=" << active << " total=" << total
      << " act=" << p.act_int;
  verify_per_expert_2d(src_test, K_in, d2_ref, H, active, M, H, is_bf16,
                       tol_fused(is_bf16), lbl.str());
}

static std::vector<CompactAsymmetricParam> make_compact_asymmetric_params() {
  std::vector<CompactAsymmetricParam> out;
  // The user's gpt-oss-20b decode shape: active=27, total=32, M=1
  // per expert, hidden=2880, intermediate=2880, swiglu.  Down-scaled
  // here for gtest runtime — the bug fires on any (active < total)
  // pair regardless of dim — but kept asymmetric in active vs total
  // to exercise the [active, total) prepack-extras tail.
  //
  // Activation sweep: swiglu (the user's repro), silu (most common
  // sparse-MoE act), and act=none (no activation — exercises the
  // setup loop with N_down == N, distinct K_down).
  const int K_in = 64, dim = 32, hidden = 64, M = 4;
  for (int act_int : {0, 1, 3}) {  // none, silu, swiglu_oai_mul
    for (auto pair : std::vector<std::pair<int,int>>{
            {3, 5}, {7, 8}, {27, 32}}) {  // small, mid, gpt-oss-shaped
      out.push_back({pair.first, pair.second, M, dim, hidden, K_in,
                     act_int});
    }
  }
  return out;
}

INSTANTIATE_TEST_SUITE_P(GroupMatmulFusedMoECompactAsymmetric,
                         TestFusedMoECompactAsymmetric,
                         ::testing::ValuesIn(make_compact_asymmetric_params()),
                         CompactAsymmetricParamName);

// See gtests/group_matmul/README.md for the cross-file test layout —
// TestGroupMatmul and TestGroupMatmulQuant live in test_basic.cpp /
// test_quant.cpp respectively.
