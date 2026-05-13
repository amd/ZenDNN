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
// behaviour itself (visible via the APILOG `[GRP_MATMUL Level3
// PACK_PROBE]` line); unit tests just verify that swapping the
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
//      which is a no-op unless `ZENDNNL_DIAGNOSTICS_ENABLE=1` was
//      set at the time of the first call (the env value is captured
//      once via `static const bool`).  Production runs WITHOUT
//      diagnostics silently clamp `active_matmul` to `M.size()` via
//      `std::min(am, M.size())` in the always-on inline guard
//      (group_matmul_direct.cpp lines 695-699) and silently accept
//      `total < active` because prepack accounting only uses
//      `max(M.size(), total_matmul)` ? the contract is "no work
//      dropped, no OOB" rather than "no impossible hint".  The
//      diagnostic mode upgrades to a hard reject so frameworks can
//      enable it in CI and catch contract drift early.
//
//      To exercise the validator we use the same subprocess pattern
//      as `[26] TestPrepackEnvBucketA`: set the env knob in the
//      parent, then `EXPECT_EXIT` with `threadsafe` death-test style
//      which `fork()` + `execve()`s a child whose first-time read of
//      `ZENDNNL_DIAGNOSTICS_ENABLE` returns "1".  The child runs the
//      dispatcher with intentionally-bad params; the parent asserts
//      a non-zero exit (the child sets it via `std::exit(1)` when
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


// See gtests/group_matmul/README.md for the cross-file test layout —
// TestGroupMatmul and TestGroupMatmulQuant live in test_basic.cpp /
// test_quant.cpp respectively.
