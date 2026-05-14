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

/// @file test_basic.cpp
/// @brief Basic group_matmul gtest sections.  Owned suites:
///
///   [2]  TestGroupMatmul              - F32_F32, BF16_F32, BF16_BF16 +
///                                       Stride variants on a random shape
///                                       grid (with optional MoE post-op).
///   [3]  TestGatedAct                 - gated-activation correctness
///                                       (silu/gelu/swiglu_oai).
///   [4]  TestMoEPostop                - weighted-reduce post-op correctness.
///   [31] TestGroupMatmulHelperParity  - parity tests for the shared
///                                       `detect_internal_alloc` helper
///                                       (extracted from 3 open-coded copies)
///                                       across the 4 v-states x 2 modes
///                                       cells, plus negative cases.
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

// Shared helper under test in [31].  Lives in the production tree so
// the parity test exercises the same instantiation production code
// uses (no test-only copy).
#include "lowoha_operators/matmul/group_matmul/detect_internal_alloc.hpp"

// ???????????????????????????????????????????????????????????????????????????????
// [2] TestGroupMatmul: F32_F32, BF16_F32, BF16_BF16 with optional MoE post-op
//
// Uses the framework's tensor_t / tensor_factory_t infrastructure and the
// reference matmul_forced_ref_kernel_test for correctness.  The three TEST_Ps
// share a common body via the templated run_basic_test helper.
// ???????????????????????????????????????????????????????????????????????????????

class TestGroupMatmul : public ::testing::TestWithParam<MatmulType> {
 protected:
  virtual void SetUp() {
    MatmulType params = GetParam();
    srand(static_cast<unsigned int>(seed));
    m            = params.matmul_m;
    k            = params.matmul_k;
    n            = params.matmul_n;
    transA       = params.transA;
    transB       = params.transB;
    alpha        = params.alpha;
    beta         = params.beta;
    algo         = params.algo;
    num_threads  = params.num_threads;
    source_dtype       = params.source_dtype;
    output_dtype       = params.output_dtype;
    weight_granularity = params.weight_granularity;
    omp_set_num_threads(num_threads);
    num_ops = 2 + (rand() % 4);
    log_info("GroupMatmul test: m=", m, " k=", k, " n=", n,
             " transA=", transA, " transB=", transB,
             " alpha=", alpha, " beta=", beta,
             " num_ops=", num_ops, " num_threads=", num_threads);
  }
  /** @brief TearDown is used to free resource used in test */
  virtual void TearDown() {
    clear_matmul_test_caches();
  }
  uint64_t m, k, n;
  bool transA, transB;
  tensor_factory_t tensor_factory{};
  float alpha, beta;
  matmul_algo_t algo;
  int32_t num_threads;
  size_t num_ops;
  // Quant-specific params forwarded from MatmulType (used by INT8/WOQ tests).
  data_type_t source_dtype{};
  data_type_t output_dtype{};
  quant_granularity_t weight_granularity{};

  // Shared body for the 6 parameterized tests:
  //  F32_F32, BF16_F32, BF16_BF16                         (use_stride=false, default)
  //  F32_F32_Stride, BF16_F32_Stride, BF16_BF16_Stride    (use_stride=true)
  // src_dt/wei_dt/dst_dt/bias_dt control the dtype configuration.
  // When use_stride is true we exercise the non-contiguous lda/ldb/ldc paths
  // (no MoE post-op); otherwise we drive the lower-level direct API and
  // optionally enable the top-2 MoE post-op.
  void run_basic_test(data_type_t src_dt, data_type_t wei_dt,
                      data_type_t dst_dt, data_type_t bias_dt,
                      float rtol_pref, float eps_pref,
                      bool use_stride = false) {
    if (use_stride) {
      // Random stride increments on input/weight/output so that each tensor's
      // leading dimension exceeds the minimum.  No MoE post-op in this mode.
      size_t stride_in_inc  = rand() % 50;
      size_t stride_wt_inc  = rand() % 50;
      size_t stride_dst_inc = rand() % 50;
      std::vector<size_t> stride_in  = {m, k};
      std::vector<size_t> stride_wt  = {k, n};
      std::vector<size_t> stride_dst = {m, n + stride_dst_inc};
      if (transB) {
        stride_wt[0] += stride_wt_inc;
      }
      else {
        stride_wt[1] += stride_wt_inc;
      }
      if (transA) {
        stride_in[0] += stride_in_inc;
      }
      else {
        stride_in[1] += stride_in_inc;
      }

      // Same bias-skip predicate as in the non-strided path / the kernel
      // helpers in gtest_utils: libxsmm doesn't accept bias for bf16 dst, and
      // aocl_dlp doesn't accept it for f16 dst.  Skip allocating bias here so
      // the kernel-under-test and reference both see an empty bias.
      const bool is_libxsmm_kernel = (algo == matmul_algo_t::libxsmm ||
                                      algo == matmul_algo_t::libxsmm_blocked);
      const bool is_aocl_kernel    = (algo == matmul_algo_t::aocl_dlp ||
                                      algo == matmul_algo_t::aocl_dlp_blocked);
      const bool skip_bias =
        (is_libxsmm_kernel && dst_dt == data_type_t::bf16) ||
        (is_aocl_kernel    && dst_dt == data_type_t::f16);

      std::vector<tensor_t> inp(num_ops), wt(num_ops), bias(num_ops),
          out(num_ops), out_ref(num_ops);
      for (size_t i = 0; i < num_ops; ++i) {
        inp[i]     = tensor_factory.uniform_dist_strided_tensor({m, k}, stride_in,
                     src_dt, 2.0, transA);
        wt[i]      = tensor_factory.uniform_dist_strided_tensor({k, n}, stride_wt,
                     wei_dt, 2.0, transB);
        bias[i]    = skip_bias
                     ? tensor_t{}
                     :
                     tensor_factory.uniform_dist_tensor({1, n}, bias_dt, 2.0);
        out[i]     = tensor_factory.uniform_dist_strided_tensor({m, n}, stride_dst,
                     dst_dt, 2.0);
        out_ref[i] = tensor_factory.uniform_dist_strided_tensor({m, n}, stride_dst,
                     dst_dt, 2.0);
      }

      status_t status = group_matmul_kernel_test(inp, wt, bias, out, algo,
                        alpha, beta);

      status_t ref_status = status_t::success;
      for (size_t i = 0; i < num_ops && ref_status == status_t::success; ++i) {
        std::vector<post_op_type_t> ref_po;
        std::vector<tensor_t> bin;
        ref_status = matmul_forced_ref_kernel_test(inp[i], wt[i], bias[i],
                     out_ref[i], ref_po, bin, false, algo, alpha, beta);
      }

      bool ok = (status == status_t::success && ref_status == status_t::success);
      // F32 relaxation only matters when accumulating into an f32 destination
      // on the libxsmm / native kernels; bf16 has its own bf16-tolerance path.
      const bool enable_f32_relaxation = (dst_dt == data_type_t::f32) &&
                                         (algo == matmul_algo_t::libxsmm    ||
                                          algo == matmul_algo_t::libxsmm_blocked ||
                                          algo == matmul_algo_t::native_gemm ||
                                          algo == matmul_algo_t::native_brgemm);
      if (ok) {
        for (size_t i = 0; i < num_ops && ok; ++i)
          compare_tensor_2D_matrix(out[i], out_ref[i], m, n, k,
                                   rtol_pref, eps_pref, ok, enable_f32_relaxation, alpha);
      }
      EXPECT_TRUE(ok);
      return;
    }

    const int D = static_cast<int>(n);
    const int num_tokens = static_cast<int>(m);
    const int topk = 2;
    const bool enable_moe = ((m + n + k + num_ops) % 2 == 1)
                            && (num_ops >= static_cast<size_t>(topk));

    std::vector<char> layouts(num_ops, 'r');
    std::vector<bool> transAs(num_ops, transA), transBs(num_ops, transB);
    std::vector<int> Ms(num_ops, (int)m), Ns(num_ops, (int)n), Ks(num_ops, (int)k);
    std::vector<float> alphas(num_ops, alpha), betas(num_ops, beta);
    std::vector<int> ldas(num_ops), ldbs(num_ops), ldcs(num_ops);
    std::vector<bool> is_wc(num_ops, false);

    std::vector<tensor_t> in_t(num_ops), wei_t(num_ops), bias_t(num_ops),
        out_t(num_ops), out_ref_t(num_ops);
    std::vector<const void *> srcs(num_ops), weis(num_ops), biases(num_ops);
    std::vector<void *> dsts(num_ops);
    std::vector<matmul_params> params(num_ops);

    for (size_t i = 0; i < num_ops; ++i) {
      in_t[i]      = tensor_factory.uniform_dist_tensor({m, k}, src_dt,  2.0, transA);
      wei_t[i]     = tensor_factory.uniform_dist_tensor({k, n}, wei_dt,  2.0, transB);
      bias_t[i]    = tensor_factory.uniform_dist_tensor({1, n}, bias_dt, 2.0);
      out_t[i]     = tensor_factory.uniform_dist_tensor({m, n}, dst_dt,  2.0);
      out_ref_t[i] = tensor_factory.uniform_dist_tensor({m, n}, dst_dt,  2.0);

      ldas[i] = transA ? (int)m : (int)k;
      ldbs[i] = transB ? (int)k : (int)n;
      ldcs[i] = (int)n;

      srcs[i]   = in_t[i].get_raw_handle_unsafe();
      weis[i]   = wei_t[i].get_raw_handle_unsafe();
      biases[i] = bias_t[i].get_raw_handle_unsafe();
      dsts[i]   = out_t[i].get_raw_handle_unsafe();

      params[i].dtypes.src  = src_dt;
      params[i].dtypes.wei  = wei_dt;
      params[i].dtypes.dst  = dst_dt;
      params[i].dtypes.bias = bias_dt;
      params[i].num_threads = num_threads;
    }

    // libxsmm doesn't accept bias for bf16 dst, and aocl_dlp doesn't accept it
    // for f16 dst.  matmul_forced_ref_kernel_test applies the same condition,
    // so we must drop bias here too to keep kernel-under-test and reference in
    // sync.
    {
      const bool is_libxsmm_kernel = (algo == matmul_algo_t::libxsmm ||
                                      algo == matmul_algo_t::libxsmm_blocked);
      const bool is_aocl_kernel    = (algo == matmul_algo_t::aocl_dlp ||
                                      algo == matmul_algo_t::aocl_dlp_blocked);
      const bool skip_bias =
        (is_libxsmm_kernel && dst_dt == data_type_t::bf16) ||
        (is_aocl_kernel    && dst_dt == data_type_t::f16);
      if (skip_bias) {
        for (auto &b : biases) {
          b = nullptr;
        }
      }
    }

    // Build MoE post-op with top-2 uniform-weight routing.
    const int num_slots = num_tokens * topk;
    std::vector<float> moe_weights(num_slots, 1.0f / topk);
    const size_t dst_elem_sz = zendnnl::common::size_of(dst_dt);
    std::vector<char> moe_output(num_tokens * D * dst_elem_sz, 0);
    std::vector<const void *> row_ptrs(num_slots);

    group_matmul_moe_postop_params moe{};
    group_matmul_moe_postop_params *moe_ptr = nullptr;
    if (enable_moe) {
      const bool skip_weighted = (src_dt == data_type_t::f32) &&
                                 ((m + k + num_ops) % 3 == 0);
      moe.num_tokens     = num_tokens;
      moe.topk           = topk;
      moe.output         = moe_output.data();
      moe.ldc_output     = D;
      moe.topk_weights   = skip_weighted ? nullptr : moe_weights.data();
      moe.skip_weighted  = skip_weighted;
      for (int t = 0; t < num_tokens; ++t) {
        for (int kk = 0; kk < topk; ++kk) {
          const size_t expert = (t + kk) % num_ops;
          const auto *base = static_cast<const char *>(dsts[expert]);
          row_ptrs[t * topk + kk] = base + (size_t)t * ldcs[expert] * dst_elem_sz;
        }
      }
      moe.row_ptrs = row_ptrs.data();
      moe_ptr = &moe;
    }

    status_t st = group_matmul_direct(layouts, transAs, transBs, Ms, Ns, Ks,
                                      alphas, srcs, ldas, weis, ldbs, biases, betas, dsts, ldcs,
                                      is_wc, params, moe_ptr);

    // Reference using the per-op forced reference kernel.
    status_t ref_st = status_t::success;
    for (size_t i = 0; i < num_ops && ref_st == status_t::success; ++i) {
      std::vector<post_op_type_t> ref_po;
      std::vector<tensor_t> dummy;
      ref_st = matmul_forced_ref_kernel_test(in_t[i], wei_t[i], bias_t[i],
                                             out_ref_t[i], ref_po, dummy, false, algo, alphas[i], betas[i]);
    }

    bool ok = (st == status_t::success && ref_st == status_t::success);

    if (ok && !enable_moe) {
      for (size_t i = 0; i < num_ops && ok; ++i)
        compare_tensor_2D_matrix(out_t[i], out_ref_t[i], m, n, k,
                                 rtol_pref, eps_pref, ok, false, alpha);
    }

    if (ok && enable_moe) {
      // MoE absolute-error budget per output: bf16-src GEMM noise
      // (∝ |alpha|*k), the `beta * C_init` rounding contribution,
      // and a topk factor for production-vs-reference reduction-order
      // divergence.  The relative `rtol * |acc|` term in the
      // comparator covers ULP-scale noise relative to the result.
      const float moe_abs_bound = (src_dt == data_type_t::f32)
                                  ? std::fabs(alpha) * ((20 + std::log2((float)k) / 4) * k + 15) * eps_pref
                                  : (std::fabs(alpha) * (float)k + std::fabs(beta)) * topk * eps_pref;
      // Relative tolerance also widens by `topk` for bf16-src to
      // absorb the worst-case reduction-order divergence between the
      // production weighted-reduce and the test's reference reduce.
      const float effective_rtol = (src_dt == data_type_t::f32)
                                   ? rtol_pref
                                   : rtol_pref * topk;
      const bool is_bf16_dst = (dst_dt == data_type_t::bf16);

      for (int t = 0; t < num_tokens && ok; ++t) {
        for (int d = 0; d < D && ok; ++d) {
          float acc = 0.0f;
          for (int kk = 0; kk < topk; ++kk) {
            const size_t expert = (t + kk) % num_ops;
            const auto *rb = static_cast<const char *>(
                               out_ref_t[expert].get_raw_handle_unsafe());
            float v;
            if (is_bf16_dst) {
              auto *rb_bf = reinterpret_cast<const uint16_t *>(rb);
              v = zendnnl::common::bfloat16_t::bf16_to_f32_val(
                    static_cast<int16_t>(rb_bf[(size_t)t * n + d]));
            }
            else {
              auto *rb_f = reinterpret_cast<const float *>(rb);
              v = rb_f[(size_t)t * n + d];
            }
            const float w = moe.skip_weighted ? 1.0f : moe_weights[t * topk + kk];
            acc += w * v;
          }
          float got;
          if (is_bf16_dst) {
            auto *op = reinterpret_cast<const uint16_t *>(moe_output.data());
            got = zendnnl::common::bfloat16_t::bf16_to_f32_val(
                    static_cast<int16_t>(op[(size_t)t * D + d]));
          }
          else {
            got = reinterpret_cast<const float *>(moe_output.data())[(size_t)t * D + d];
          }
          if (std::abs(acc - got) > moe_abs_bound + effective_rtol * std::abs(acc)) {
            log_error("MoE mismatch t=", t, " d=", d, " expected=", acc, " got=", got);
            ok = false;
          }
        }
      }
    }
    EXPECT_TRUE(ok);
  }
};

TEST_P(TestGroupMatmul, F32_F32) {
  run_basic_test(data_type_t::f32, data_type_t::f32, data_type_t::f32,
                 data_type_t::f32, rtol_f32, epsilon_f32);
}
TEST_P(TestGroupMatmul, BF16_F32) {
  run_basic_test(data_type_t::bf16, data_type_t::bf16, data_type_t::f32,
                 data_type_t::f32, rtol_f32, epsilon_f32);
}
TEST_P(TestGroupMatmul, BF16_BF16) {
  run_basic_test(data_type_t::bf16, data_type_t::bf16, data_type_t::bf16,
                 data_type_t::bf16, rtol_bf16, epsilon_bf16);
}

TEST_P(TestGroupMatmul, F32_F32_Stride) {
  run_basic_test(data_type_t::f32, data_type_t::f32, data_type_t::f32,
                 data_type_t::f32, rtol_f32, epsilon_f32, /*use_stride=*/true);
}

TEST_P(TestGroupMatmul, BF16_F32_Stride) {
  run_basic_test(data_type_t::bf16, data_type_t::bf16, data_type_t::f32,
                 data_type_t::f32, rtol_f32, epsilon_f32, /*use_stride=*/true);
}

TEST_P(TestGroupMatmul, BF16_BF16_Stride) {
  run_basic_test(data_type_t::bf16, data_type_t::bf16, data_type_t::bf16,
                 data_type_t::bf16, rtol_bf16, epsilon_bf16, /*use_stride=*/true);
}

INSTANTIATE_TEST_SUITE_P(GroupMatmul, TestGroupMatmul,
                         ::testing::ValuesIn(matmul_test));

// ???????????????????????????????????????????????????????????????????????????????
// [3] TestGatedAct: gated activation correctness (silu, gelu, swiglu)
// ???????????????????????????????????????????????????????????????????????????????

struct GatedActTestParam {
  int dim, M, num_ops, act_int;  // act_int: 0=none, 1=silu, 2=gelu, 3=swiglu_oai
  bool is_bf16;
};

static std::string GatedActParamName(
  const ::testing::TestParamInfo<GatedActTestParam> &info) {
  static const char *act_names[] = {"none", "silu", "gelu", "swiglu"};
  const auto &p = info.param;
  return std::string(act_names[p.act_int]) + (p.is_bf16 ? "_bf16" : "_f32")
         + "_d" + std::to_string(p.dim) + "_M" + std::to_string(p.M)
         + "_E" + std::to_string(p.num_ops);
}

class TestGatedAct : public ::testing::TestWithParam<GatedActTestParam> {};

TEST_P(TestGatedAct, Correctness) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;

  const auto &p = GetParam();
  const int dim = p.dim, N = 2 * dim, M = p.M, K = 32;
  const int num_ops = p.num_ops;
  const auto act_type = static_cast<grp_matmul_gated_act_t>(p.act_int);
  const data_type_t dt = p.is_bf16 ? data_type_t::bf16 : data_type_t::f32;

  // Allocate and fill buffers.
  TypedBuffers src, wei, dst_act, dst_ref;
  src.alloc(num_ops, (size_t)M * K, p.is_bf16);
  wei.alloc(num_ops, (size_t)K * N, p.is_bf16);
  dst_act.alloc(num_ops, (size_t)M * N, p.is_bf16);
  dst_ref.alloc(num_ops, (size_t)M * N, p.is_bf16);
  for (int e = 0; e < num_ops; ++e) {
    if (p.is_bf16) {
      fill_src(src.bf16[e], e, 0.05f);
      fill_wei1(wei.bf16[e], e, 0.01f);
    }
    else           {
      fill_src(src.f32[e],  e, 0.05f);
      fill_wei1(wei.f32[e],  e, 0.01f);
    }
  }

  auto gv = GemmVecs::uniform(num_ops, M, N, K, 1.0f, 0.0f, /*wc=*/true);
  auto srcs    = src.cptrs(p.is_bf16);
  auto weis    = wei.cptrs(p.is_bf16);
  auto dsts    = dst_act.ptrs(p.is_bf16);
  auto dsts_ref= dst_ref.ptrs(p.is_bf16);
  std::vector<const void *> biases(num_ops, nullptr);
  auto params  = make_uniform_params(num_ops, dt);

  // Run reference Op1 (no activation) ? dst_ref.
  {
    auto pr = params;
    ASSERT_EQ(group_matmul_direct(gv.layout, gv.transA, gv.transB, gv.Ms, gv.Ns,
                                  gv.Ks, gv.alpha, srcs, gv.lda, weis, gv.ldb, biases, gv.beta,
                                  dsts_ref, gv.ldc, gv.is_wc, pr, nullptr, nullptr), status_t::success);
  }

  // Run fused Op1+Act ? dst_act.
  grp_matmul_gated_act_params act{};
  act.act = act_type;
  {
    auto pa = params;
    ASSERT_EQ(group_matmul_direct(gv.layout, gv.transA, gv.transB, gv.Ms, gv.Ns,
                                  gv.Ks, gv.alpha, srcs, gv.lda, weis, gv.ldb, biases, gv.beta,
                                  dsts, gv.ldc, gv.is_wc, pa, nullptr,
                                  (act_type != grp_matmul_gated_act_t::none) ? &act : nullptr),
              status_t::success);
  }

  // For act=none: must be byte-identical to reference.
  if (act_type == grp_matmul_gated_act_t::none) {
    for (int e = 0; e < num_ops; ++e) {
      const size_t sz = (size_t)M * N;
      if (p.is_bf16)
        ASSERT_EQ(std::memcmp(dst_act.bf16[e].data(), dst_ref.bf16[e].data(),
                              sz * sizeof(bfloat16_t)), 0) << "expert=" << e;
      else
        ASSERT_EQ(std::memcmp(dst_act.f32[e].data(), dst_ref.f32[e].data(),
                              sz * sizeof(float)), 0) << "expert=" << e;
    }
    return;
  }

  // Compare activated output against scalar reference (applied to dst_ref).
  const auto tol = tol_act(p.is_bf16);
  for (int e = 0; e < num_ops; ++e) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < dim; ++n) {
        float g_val, u_val;
        if (act_type == grp_matmul_gated_act_t::swiglu_oai_mul) {
          g_val = dst_ref.at(e, (size_t)m * N + 2 * n, p.is_bf16);
          u_val = dst_ref.at(e, (size_t)m * N + 2 * n + 1, p.is_bf16);
        }
        else {
          g_val = dst_ref.at(e, (size_t)m * N + n, p.is_bf16);
          u_val = dst_ref.at(e, (size_t)m * N + dim + n, p.is_bf16);
        }
        const float expected = ref_gated_act(act_type, g_val, u_val);
        const float actual   = dst_act.at(e, (size_t)m * N + n, p.is_bf16);
        ASSERT_NEAR(actual, expected, std::abs(expected) * tol.rel + tol.abs)
            << "act=" << p.act_int << (p.is_bf16 ? " bf16" : " f32")
            << " dim=" << dim << " M=" << M << " e=" << e
            << " m=" << m << " n=" << n;
      }
    }
  }
}

static std::vector<GatedActTestParam> make_gated_act_params() {
  std::vector<GatedActTestParam> out;
  // Core: all act ? dtype ? dim at M=4, E=2.
  for (int a : {
         0, 1, 2, 3
       })
    for (bool bf : {
           false, true
         })
      for (int d : {
             1, 7, 15, 16, 17, 31, 32, 33, 64, 128, 255, 256, 512
           })
        out.push_back({d, 4, 2, a, bf});
  // Vary M and num_ops (skip duplicates of core M=4,E=2).
  for (int a : {
         1, 2, 3
       })
    for (bool bf : {
           false, true
         })
      for (int d : {
             1, 16, 33, 128
           })
        for (int m : {
               1, 4, 16, 64
             })
          for (int e : {
                 2, 4, 8, 16
               }) {
            if (m == 4 && e == 2) {
              continue;
            }
            out.push_back({d, m, e, a, bf});
          }
  // Large-expert MoE-realistic configs.
  for (bool bf : {
         false, true
       })
    for (int e : {
           32, 64
         })
      for (int m : {
             1, 2
           })
        out.push_back({16, m, e, 1, bf});
  return out;
}

INSTANTIATE_TEST_SUITE_P(GroupMatmulGatedAct, TestGatedAct,
                         ::testing::ValuesIn(make_gated_act_params()), GatedActParamName);

// ???????????????????????????????????????????????????????????????????????????????
// [4] TestMoEPostop: weighted-reduce post-op correctness
// ???????????????????????????????????????????????????????????????????????????????

struct MoEPostopTestParam {
  int num_ops, M, N, K, topk;
  bool skip_weighted, is_bf16;
};

static std::string MoEPostopParamName(
  const ::testing::TestParamInfo<MoEPostopTestParam> &info) {
  const auto &p = info.param;
  return (p.is_bf16 ? "bf16" : "f32")
         + std::string("_E") + std::to_string(p.num_ops)
         + "_M" + std::to_string(p.M) + "_N" + std::to_string(p.N)
         + "_K" + std::to_string(p.K) + "_topk" + std::to_string(p.topk)
         + (p.skip_weighted ? "_skip" : "");
}

class TestMoEPostop : public ::testing::TestWithParam<MoEPostopTestParam> {};

TEST_P(TestMoEPostop, Correctness) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;

  const auto &p = GetParam();
  const int total_M = p.num_ops * p.M;
  if (total_M % p.topk != 0) {
    return;
  }
  const int num_tokens = total_M / p.topk;
  const data_type_t dt = p.is_bf16 ? data_type_t::bf16 : data_type_t::f32;
  const size_t elem_sz = zendnnl::common::size_of(dt);

  TypedBuffers src, wei, dst, dst_ref;
  src.alloc(p.num_ops, (size_t)p.M * p.K, p.is_bf16);
  wei.alloc(p.num_ops, (size_t)p.K * p.N, p.is_bf16);
  dst.alloc(p.num_ops, (size_t)p.M * p.N, p.is_bf16);
  dst_ref.alloc(p.num_ops, (size_t)p.M * p.N, p.is_bf16);
  for (int e = 0; e < p.num_ops; ++e) {
    // Keep init in f32 domain first, then mirror to bf16 (matches legacy pattern).
    std::vector<float> sf((size_t)p.M * p.K), wf((size_t)p.K * p.N);
    fill_src(sf, e, 0.01f);
    fill_wei1(wf, e, 0.005f);
    if (p.is_bf16) {
      for (size_t i = 0; i < sf.size(); ++i) {
        src.bf16[e][i] = bfloat16_t(sf[i]);
      }
      for (size_t i = 0; i < wf.size(); ++i) {
        wei.bf16[e][i] = bfloat16_t(wf[i]);
      }
    }
    else {
      src.f32[e] = std::move(sf);
      wei.f32[e] = std::move(wf);
    }
  }

  auto gv   = GemmVecs::uniform(p.num_ops, p.M, p.N, p.K);
  auto srcs = src.cptrs(p.is_bf16);
  auto weis = wei.cptrs(p.is_bf16);
  auto dsts = dst.ptrs(p.is_bf16);
  auto dsts_ref = dst_ref.ptrs(p.is_bf16);
  std::vector<const void *> biases(p.num_ops, nullptr);
  auto params = make_uniform_params(p.num_ops, dt);

  // Build moe_postop: sequential row mapping (expert e, row j) ? token j+e*M.
  std::vector<float> moe_w(total_M, p.skip_weighted ? 0.0f : 1.0f / p.topk);
  std::vector<char> moe_out((size_t)num_tokens * p.N * elem_sz, 0);
  std::vector<const void *> row_ptrs(total_M);
  {
    int slot = 0;
    for (int e = 0; e < p.num_ops; ++e)
      for (int j = 0; j < p.M; ++j)
        row_ptrs[slot++] = static_cast<const char *>(dsts[e])
                           + (size_t)j * p.N * elem_sz;
  }
  group_matmul_moe_postop_params moe{};
  moe.num_tokens    = num_tokens;
  moe.topk          = p.topk;
  moe.output        = moe_out.data();
  moe.ldc_output    = p.N;
  moe.topk_weights  = p.skip_weighted ? nullptr : moe_w.data();
  moe.skip_weighted = p.skip_weighted;
  moe.row_ptrs      = row_ptrs.data();

  // Run GEMM + MoE reduce and a separate plain-GEMM reference in one shot.
  auto pa_run = params;
  ASSERT_EQ(group_matmul_direct(gv.layout, gv.transA, gv.transB, gv.Ms, gv.Ns,
                                gv.Ks, gv.alpha, srcs, gv.lda, weis, gv.ldb, biases, gv.beta,
                                dsts, gv.ldc, gv.is_wc, pa_run, &moe), status_t::success);

  auto pa_ref = params;
  ASSERT_EQ(group_matmul_direct(gv.layout, gv.transA, gv.transB, gv.Ms, gv.Ns,
                                gv.Ks, gv.alpha, srcs, gv.lda, weis, gv.ldb, biases, gv.beta,
                                dsts_ref, gv.ldc, gv.is_wc, pa_ref), status_t::success);

  // Manual weighted reduce over reference outputs.
  const auto tol = tol_moe(p.is_bf16);
  for (int t = 0; t < num_tokens; ++t) {
    for (int d = 0; d < p.N; ++d) {
      float acc = 0.0f;
      for (int kk = 0; kk < p.topk; ++kk) {
        const int si = t * p.topk + kk;
        const int expert = si / p.M, row = si % p.M;
        if (expert >= p.num_ops) {
          continue;
        }
        const float val = dst_ref.at(expert, (size_t)row * p.N + d, p.is_bf16);
        acc += (p.skip_weighted ? 1.0f : moe_w[si]) * val;
      }
      float got;
      if (p.is_bf16)
        got = static_cast<float>(reinterpret_cast<const bfloat16_t *>(
                                   moe_out.data())[(size_t)t * p.N + d]);
      else {
        got = reinterpret_cast<const float *>(moe_out.data())[(size_t)t * p.N + d];
      }
      ASSERT_NEAR(got, acc, std::abs(acc) * tol.rel + tol.abs)
          << (p.is_bf16 ? "bf16" : "f32") << " E=" << p.num_ops
          << " M=" << p.M << " N=" << p.N << " topk=" << p.topk
          << (p.skip_weighted ? " skip" : "") << " t=" << t << " d=" << d;
    }
  }
}

static std::vector<MoEPostopTestParam> make_moe_postop_params() {
  std::vector<MoEPostopTestParam> out;
  for (bool bf : {
         false, true
       })
    for (int e : {
           2, 4, 8
         })
      for (int m : {
             2, 4, 8, 16
           })
        for (int topk : {
               1, 2
             })
          if ((e * m) % topk == 0) out.push_back({e, m, 64, 32, topk, false, bf});
  for (bool bf : {
         false, true
       }) out.push_back({4, 4, 64, 32, 2, true, bf});
  for (int n : {
         16, 128, 256
       })
    for (int k : {
           16, 64
         }) out.push_back({4, 4, n, k, 2, false, false});
  out.push_back({16, 2, 64, 32, 2, false, false});
  out.push_back({32, 1, 64, 32, 2, false, false});
  return out;
}

INSTANTIATE_TEST_SUITE_P(GroupMatmulMoEPostop, TestMoEPostop,
                         ::testing::ValuesIn(make_moe_postop_params()), MoEPostopParamName);

// ===============================================================================
// [31] TestGroupMatmulHelperParity - parity tests for shared dispatcher
//      helpers extracted during the post-PR-443 maintenance refactor.
//
//      Currently covers:
//        * `detect_internal_alloc` (lifted from 3 open-coded copies in
//          group_matmul_direct.cpp:192 + 710-714 + group_matmul_fused_moe
//          .cpp:315 to a single header at group_matmul/detect_internal_
//          alloc.hpp).  The helper has 2 modes (`quick_o1`, `sweep_active`)
//          and the 4 v-states (empty / all-null / all-non-null / mixed
//          null+non-null) so we cover the full 8-cell matrix plus the
//          `fused_moe_present == false` short-circuit and the prepack-
//          extras tail (sweep stops at min(num_ops, v.size())).
//
//      Future helper extractions (size_check_ctx,
//      prepack_extras_metadata_undersized) will land additional sections
//      here as parity checks on their respective contracts.
// ===============================================================================

namespace {
using zendnnl::lowoha::matmul::group_matmul_internal::detect_internal_alloc;
using zendnnl::lowoha::matmul::group_matmul_internal::internal_alloc_mode;
using zendnnl::error_handling::status_t;

// Convenience builders for the 4 v-states.  `n` is the active count
// (= num_ops); `tail_kind` controls the prepack-extras tail used by
// the sweep-bound test below.
inline std::vector<void *> v_empty() { return {}; }
inline std::vector<void *> v_all_null(size_t n) {
  return std::vector<void *>(n, nullptr);
}
inline std::vector<void *> v_all_nonnull(size_t n) {
  // Use a sentinel non-null pointer that won't be dereferenced.  The
  // helper only inspects pointer-equality vs nullptr.
  static int sentinel = 0;
  return std::vector<void *>(n, &sentinel);
}
inline std::vector<void *> v_mixed_null_at(size_t n, size_t null_idx) {
  auto v = v_all_nonnull(n);
  v[null_idx] = nullptr;
  return v;
}

constexpr bool kIsInternalTrue  = true;
constexpr bool kIsInternalFalse = false;

} // namespace

class TestGroupMatmulHelperParity : public ::testing::Test {};

// ── Cell 1/8: quick_o1 + empty + fused_moe_present ────────────────────────
TEST_F(TestGroupMatmulHelperParity, QuickO1_EmptyFusedPresent_IsInternal) {
  bool out = false;
  const auto v = v_empty();
  ASSERT_EQ(detect_internal_alloc(v, /*num_ops=*/4, /*fused_moe_present=*/true,
                                  internal_alloc_mode::quick_o1, &out),
            status_t::success);
  EXPECT_EQ(out, kIsInternalTrue)
      << "empty vector + fused-MoE engaged = library-managed";
}

// ── Cell 2/8: quick_o1 + all-null + fused_moe_present ─────────────────────
TEST_F(TestGroupMatmulHelperParity, QuickO1_AllNullFusedPresent_IsInternal) {
  bool out = false;
  const auto v = v_all_null(4);
  ASSERT_EQ(detect_internal_alloc(v, /*num_ops=*/4, /*fused_moe_present=*/true,
                                  internal_alloc_mode::quick_o1, &out),
            status_t::success);
  EXPECT_EQ(out, kIsInternalTrue) << "v[0]=nullptr -> internal-alloc";
}

// ── Cell 3/8: quick_o1 + all-non-null + fused_moe_present ─────────────────
TEST_F(TestGroupMatmulHelperParity, QuickO1_AllNonNullFusedPresent_NotInternal) {
  bool out = true;
  const auto v = v_all_nonnull(4);
  ASSERT_EQ(detect_internal_alloc(v, /*num_ops=*/4, /*fused_moe_present=*/true,
                                  internal_alloc_mode::quick_o1, &out),
            status_t::success);
  EXPECT_EQ(out, kIsInternalFalse) << "v[0]!=nullptr -> caller-allocated";
}

// ── Cell 4/8: quick_o1 + mixed (null at [0]) + fused_moe_present ──────────
//   quick_o1 SAMPLES v[0] only.  When v[0]==nullptr it reports internal,
//   even if downstream slots are non-null — the diagnostic sweep
//   path is responsible for catching the mixed state.
TEST_F(TestGroupMatmulHelperParity, QuickO1_MixedAt0FusedPresent_NoMixedDetect) {
  bool out = false;
  auto v = v_all_nonnull(4);
  v[0] = nullptr;  // mixed: only [0] is null
  ASSERT_EQ(detect_internal_alloc(v, /*num_ops=*/4, /*fused_moe_present=*/true,
                                  internal_alloc_mode::quick_o1, &out),
            status_t::success)
      << "quick_o1 must NOT report failure on mixed-state — that's the "
         "sweep_active path's job; quick_o1 is a fast production check";
  EXPECT_EQ(out, kIsInternalTrue);
}

// ── Cell 5/8: sweep_active + empty + fused_moe_present ────────────────────
TEST_F(TestGroupMatmulHelperParity, Sweep_EmptyFusedPresent_IsInternal) {
  bool out = false;
  const auto v = v_empty();
  ASSERT_EQ(detect_internal_alloc(v, /*num_ops=*/4, /*fused_moe_present=*/true,
                                  internal_alloc_mode::sweep_active, &out),
            status_t::success);
  EXPECT_EQ(out, kIsInternalTrue);
}

// ── Cell 6/8: sweep_active + all-null + fused_moe_present ─────────────────
TEST_F(TestGroupMatmulHelperParity, Sweep_AllNullFusedPresent_IsInternal) {
  bool out = false;
  const auto v = v_all_null(4);
  ASSERT_EQ(detect_internal_alloc(v, /*num_ops=*/4, /*fused_moe_present=*/true,
                                  internal_alloc_mode::sweep_active, &out),
            status_t::success);
  EXPECT_EQ(out, kIsInternalTrue);
}

// ── Cell 7/8: sweep_active + all-non-null + fused_moe_present ─────────────
TEST_F(TestGroupMatmulHelperParity, Sweep_AllNonNullFusedPresent_NotInternal) {
  bool out = true;
  const auto v = v_all_nonnull(4);
  ASSERT_EQ(detect_internal_alloc(v, /*num_ops=*/4, /*fused_moe_present=*/true,
                                  internal_alloc_mode::sweep_active, &out),
            status_t::success);
  EXPECT_EQ(out, kIsInternalFalse);
}

// ── Cell 8/8: sweep_active + mixed null+non-null in active range ──────────
//   The hard contract break: sweep MUST report failure so the caller
//   can `log_error` and reject.
TEST_F(TestGroupMatmulHelperParity, Sweep_MixedFusedPresent_ReturnsFailure) {
  bool out = false;
  const auto v = v_mixed_null_at(/*n=*/4, /*null_idx=*/2);
  EXPECT_EQ(detect_internal_alloc(v, /*num_ops=*/4, /*fused_moe_present=*/true,
                                  internal_alloc_mode::sweep_active, &out),
            status_t::failure)
      << "sweep_active must report failure on mixed null/non-null active range";
}

// ── fused_moe_present=false short-circuit: never internal ─────────────────
TEST_F(TestGroupMatmulHelperParity, NotFused_AllPathsFalse) {
  bool out = true;
  // Even an all-null vector returns false when fused-MoE isn't engaged.
  const auto v_null = v_all_null(4);
  ASSERT_EQ(detect_internal_alloc(v_null, /*num_ops=*/4,
                                  /*fused_moe_present=*/false,
                                  internal_alloc_mode::quick_o1, &out),
            status_t::success);
  EXPECT_EQ(out, kIsInternalFalse);

  out = true;
  ASSERT_EQ(detect_internal_alloc(v_null, /*num_ops=*/4,
                                  /*fused_moe_present=*/false,
                                  internal_alloc_mode::sweep_active, &out),
            status_t::success);
  EXPECT_EQ(out, kIsInternalFalse);

  // Empty vector also short-circuits to false (no fused-MoE = no Op2).
  out = true;
  const auto v_e = v_empty();
  ASSERT_EQ(detect_internal_alloc(v_e, /*num_ops=*/4,
                                  /*fused_moe_present=*/false,
                                  internal_alloc_mode::sweep_active, &out),
            status_t::success);
  EXPECT_EQ(out, kIsInternalFalse);
}

// ── Prepack-extras tail: sweep bound is min(num_ops, v.size()) ────────────
//   When v is sized to total_matmul (= active + extras) but num_ops is
//   the active count, the sweep MUST stop at num_ops so a tail of all-
//   null placeholder slots [active..total) doesn't false-flag the
//   active range as mixed.
TEST_F(TestGroupMatmulHelperParity, Sweep_PrepackExtrasTail_NotMixed) {
  bool out = true;
  // 4 active slots all non-null, 4 trailing slots all null (extras).
  // v.size() = 8, num_ops = 4.
  std::vector<void *> v(8, nullptr);
  static int sentinel = 0;
  for (int i = 0; i < 4; ++i) v[i] = &sentinel;
  ASSERT_EQ(detect_internal_alloc(v, /*num_ops=*/4, /*fused_moe_present=*/true,
                                  internal_alloc_mode::sweep_active, &out),
            status_t::success)
      << "sweep must stop at num_ops=4; the [4..8) all-null tail is the "
         "prepack-extras tail and is legitimate";
  EXPECT_EQ(out, kIsInternalFalse) << "active range is all non-null";
}

// ── num_ops exceeding v.size(): sweep clamps to v.size() ──────────────────
//   The dispatcher's framework-opt-in path may call with num_ops set
//   to params[0].active_matmul which can exceed v.size() for the
//   inline guard.  The validator's other guards reject that case
//   upstream, but the helper must still behave (no OOB read).
TEST_F(TestGroupMatmulHelperParity, Sweep_NumOpsExceedsVSize_ClampsSafely) {
  bool out = true;
  const auto v = v_all_nonnull(2);
  ASSERT_EQ(detect_internal_alloc(v, /*num_ops=*/8, /*fused_moe_present=*/true,
                                  internal_alloc_mode::sweep_active, &out),
            status_t::success);
  EXPECT_EQ(out, kIsInternalFalse) << "min(8, 2) = 2 slots, all non-null";
}

// ── Production-parity: quick_o1 matches the inline guard's predicate ──────
//   The inline guard at group_matmul_direct.cpp:710-714 ran:
//     (fused_moe != nullptr) && (v.empty() || v[0] == nullptr)
//   The helper's quick_o1 path must produce identical results across
//   the 2x2 (fused_moe_present, v.empty()/non-empty/null/non-null) grid.
TEST_F(TestGroupMatmulHelperParity, ProductionParity_QuickO1_MatchesOldFormula) {
  auto run_old = [](const std::vector<void *> &v, bool fmp) -> bool {
    return fmp && (v.empty() || v[0] == nullptr);
  };
  auto run_new = [](const std::vector<void *> &v, bool fmp) -> bool {
    bool out = false;
    detect_internal_alloc(v, /*num_ops=*/4, fmp,
                          internal_alloc_mode::quick_o1, &out);
    return out;
  };
  for (bool fmp : {false, true}) {
    EXPECT_EQ(run_old(v_empty(),       fmp), run_new(v_empty(),       fmp));
    EXPECT_EQ(run_old(v_all_null(4),   fmp), run_new(v_all_null(4),   fmp));
    EXPECT_EQ(run_old(v_all_nonnull(4),fmp), run_new(v_all_nonnull(4),fmp));
  }
}
