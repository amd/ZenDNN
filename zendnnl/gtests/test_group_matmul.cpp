/********************************************************************************
# * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

/// @file test_group_matmul.cpp
/// @brief GTests for group_matmul_direct API, organized into sections:
///   [1] Shared helpers (moe_test_utils): buffers, init, params, reference math
///   [2] TestGroupMatmul         — F32/BF16 basic + moe_postop coverage
///   [3] TestGatedAct            — gated activation correctness (silu/gelu/swiglu)
///   [4] TestMoEPostop           — weighted-reduce post-op correctness
///   [5] TestFusedMoE            — fused Op1→Act→Op2 vs 2-call reference
///   [6] TestGroupMatmulCombined — all 2³=8 combinations of (moe,act,fused)
///   [7] TestFusedMoEAlgos       — fused path × ALGOs 1/2/3 × mixed precision
///   [8] TestFusedMoENegative    — validation fast-fail paths
///
/// Coverage matrix (separate suites for each optional feature):
///   1. GEMM + activation only        → TestGatedAct
///      • all 4 acts (none/silu/gelu/swiglu), both f32/bf16, many dim×M×E
///   2. GEMM + moe_postop only        → TestMoEPostop
///      • weighted-reduce + skip_weighted, both f32/bf16, many E×M×topk
///   3. GEMM + fused MoE (Op1→Act→Op2)→ TestFusedMoE, TestFusedMoEAlgos
///      • all 4 acts (none/silu/gelu/swiglu), both f32/bf16,
///        mixed precision (bf16→f32), with/without down_proj bias,
///        per-ALGO coverage (1/2/3), BF16 realistic-decode shapes.
///   4. All three combined            → TestGroupMatmulCombined
///      • 2³=8 combinations of (moe, act, fused) × both dtypes × key shapes.
/// BF16 is verified in every suite; it is the primary precision for MoE inference.

#include <gtest/gtest.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "gtest_utils.hpp"
#include "common/bfloat16.hpp"

namespace moe_test_utils {

using bfloat16_t = zendnnl::common::bfloat16_t;
using data_type_t = zendnnl::common::data_type_t;
using zendnnl::lowoha::matmul::matmul_params;
using zendnnl::lowoha::matmul::grp_matmul_gated_act_t;

// ═══════════════════════════════════════════════════════════════════════════════
// [1.a] Test data initialization
//
// Deterministic pattern: a[i] = scale * (int((i + e*seed_step) % mod) - shift).
// The explicit int cast prevents size_t underflow that would produce garbage
// like 1.7e+35 (see git history for the bug this fixes).
// ═══════════════════════════════════════════════════════════════════════════════

inline float make_value(size_t i, int e, int seed_step, int mod, int shift,
                        float scale) {
  return scale * static_cast<float>(
      static_cast<int>((i + e * seed_step) % mod) - shift);
}

template <typename T>
inline void fill_pattern(std::vector<T> &vec, int e, int seed_step, int mod,
                         int shift, float scale) {
  for (size_t i = 0; i < vec.size(); ++i) {
    const float v = make_value(i, e, seed_step, mod, shift, scale);
    if constexpr (std::is_same_v<T, bfloat16_t>)
      vec[i] = bfloat16_t(v);
    else
      vec[i] = static_cast<T>(v);
  }
}

// Standard fill presets used across tests (picked so activation doesn't overflow).
inline void fill_src (std::vector<float> &v, int e, float s = 0.02f)   { fill_pattern(v, e, 7, 11, 5, s); }
inline void fill_src (std::vector<bfloat16_t> &v, int e, float s = 0.02f) { fill_pattern(v, e, 7, 11, 5, s); }
inline void fill_wei1(std::vector<float> &v, int e, float s = 0.005f)  { fill_pattern(v, e, 3, 7, 3, s); }
inline void fill_wei1(std::vector<bfloat16_t> &v, int e, float s = 0.005f) { fill_pattern(v, e, 3, 7, 3, s); }
inline void fill_wei2(std::vector<float> &v, int e, float s = 0.008f)  { fill_pattern(v, e, 5, 9, 4, s); }
inline void fill_wei2(std::vector<bfloat16_t> &v, int e, float s = 0.008f) { fill_pattern(v, e, 5, 9, 4, s); }

// ═══════════════════════════════════════════════════════════════════════════════
// [1.b] TypedBuffers: holds both bf16 and f32 storage, exposes opaque pointers.
//
// Avoids the if/else maze around every buffer access.  Only one of the two
// vectors is actually resized based on `is_bf16`.  `is_mixed = true` resizes
// both (used by TestFusedMoEAlgos for bf16-src → f32-dst mixed precision).
// ═══════════════════════════════════════════════════════════════════════════════

struct TypedBuffers {
  std::vector<std::vector<float>> f32;
  std::vector<std::vector<bfloat16_t>> bf16;
  bool store_f32 = false;
  bool store_bf16 = false;

  void alloc(int num_ops, size_t elems, bool is_bf16, bool is_mixed = false) {
    // Mixed precision: bf16 inputs, f32 outputs.  Caller selects which half
    // to use for a given role by calling ptrs_f32/ptrs_bf16.
    store_bf16 = is_bf16 || is_mixed;
    store_f32  = !is_bf16 || is_mixed;
    if (store_f32)  { f32.resize(num_ops);  for (auto &v : f32)  v.assign(elems, 0.0f); }
    if (store_bf16) { bf16.resize(num_ops); for (auto &v : bf16) v.assign(elems, bfloat16_t(0.0f)); }
  }

  // Opaque pointer view selector: `bf16 = true` picks bf16 storage.
  std::vector<const void *> cptrs(bool pick_bf16) const {
    std::vector<const void *> out(pick_bf16 ? bf16.size() : f32.size());
    if (pick_bf16) for (size_t e = 0; e < bf16.size(); ++e) out[e] = bf16[e].data();
    else           for (size_t e = 0; e < f32.size();  ++e) out[e] = f32[e].data();
    return out;
  }
  std::vector<void *> ptrs(bool pick_bf16) {
    std::vector<void *> out(pick_bf16 ? bf16.size() : f32.size());
    if (pick_bf16) for (size_t e = 0; e < bf16.size(); ++e) out[e] = bf16[e].data();
    else           for (size_t e = 0; e < f32.size();  ++e) out[e] = f32[e].data();
    return out;
  }
  float at(int e, size_t i, bool is_bf16) const {
    return is_bf16 ? static_cast<float>(bf16[e][i]) : f32[e][i];
  }
};

// ═══════════════════════════════════════════════════════════════════════════════
// [1.c] matmul_params & GEMM wrapper vector builders
// ═══════════════════════════════════════════════════════════════════════════════

inline std::vector<matmul_params> make_uniform_params(int num_ops, data_type_t dt,
    data_type_t bias_dt = data_type_t::none) {
  std::vector<matmul_params> params(num_ops);
  for (auto &p : params) {
    p.dtypes.src  = dt;
    p.dtypes.wei  = dt;
    p.dtypes.dst  = dt;
    p.dtypes.bias = bias_dt;
  }
  return params;
}

inline std::vector<matmul_params> make_mixed_params(int num_ops,
    data_type_t src_dt, data_type_t wei_dt, data_type_t dst_dt,
    data_type_t bias_dt = data_type_t::none) {
  std::vector<matmul_params> params(num_ops);
  for (auto &p : params) {
    p.dtypes.src  = src_dt;
    p.dtypes.wei  = wei_dt;
    p.dtypes.dst  = dst_dt;
    p.dtypes.bias = bias_dt;
  }
  return params;
}

// Packs the uniform GEMM wrapper vectors (layouts, transposes, alphas, etc.)
// that every test initializes identically.
struct GemmVecs {
  std::vector<char> layout;
  std::vector<bool> transA, transB, is_wc;
  std::vector<float> alpha, beta;
  std::vector<int> Ms, Ns, Ks, lda, ldb, ldc;

  static GemmVecs uniform(int num_ops, int M, int N, int K,
                          float a = 1.0f, float b = 0.0f,
                          bool wc = false, bool tA = false, bool tB = false) {
    GemmVecs v;
    v.layout.assign(num_ops, 'r');
    v.transA.assign(num_ops, tA);
    v.transB.assign(num_ops, tB);
    v.is_wc.assign(num_ops, wc);
    v.alpha.assign(num_ops, a);
    v.beta.assign(num_ops, b);
    v.Ms.assign(num_ops, M);
    v.Ns.assign(num_ops, N);
    v.Ks.assign(num_ops, K);
    v.lda.assign(num_ops, tA ? M : K);
    v.ldb.assign(num_ops, tB ? K : N);
    v.ldc.assign(num_ops, N);
    return v;
  }
};

// ═══════════════════════════════════════════════════════════════════════════════
// [1.d] Scalar reference math for gated activations and weighted reduce
// ═══════════════════════════════════════════════════════════════════════════════

inline float ref_silu_mul(float g, float u) {
  return g * (1.0f / (1.0f + std::exp(-g))) * u;
}
inline float ref_gelu_mul(float g, float u) {
  return g * 0.5f * (1.0f + std::erf(g * 0.7071067811865476f)) * u;
}
inline float ref_swiglu_oai(float g, float u) {
  constexpr float alpha = 1.702f;
  g = std::max(-7.0f, std::min(g, 7.0f));
  u = std::max(-7.0f, std::min(u, 7.0f));
  const float sig = 1.0f / (1.0f + std::exp(-g * alpha));
  return (1.0f + u) * g * sig;
}

// Computes reference activation output for row `m` column `n` of expert `e`.
// The raw (pre-activation) Op1 output is read from `raw_dst` with stride N.
inline float ref_gated_act(grp_matmul_gated_act_t act, float g_or_even, float u_or_odd) {
  switch (act) {
  case grp_matmul_gated_act_t::silu_and_mul:   return ref_silu_mul(g_or_even, u_or_odd);
  case grp_matmul_gated_act_t::gelu_and_mul:   return ref_gelu_mul(g_or_even, u_or_odd);
  case grp_matmul_gated_act_t::swiglu_oai_mul: return ref_swiglu_oai(g_or_even, u_or_odd);
  default:                                     return 0.0f;
  }
}

// Apply activation in-place on dst[:, 0:dim].  dst has M rows with stride ldc.
// Matches the kernel semantics in group_matmul_moe_act.cpp.
template <typename T>
inline void apply_ref_gated_act(std::vector<T> &dst, int M, int N, int ldc,
                                grp_matmul_gated_act_t act) {
  if (act == grp_matmul_gated_act_t::none) return;
  const int dim = N / 2;
  const bool swiglu = (act == grp_matmul_gated_act_t::swiglu_oai_mul);
  for (int m = 0; m < M; ++m) {
    T *row = dst.data() + m * ldc;
    // Read raw values first (since we write to row[n] below, which may be read).
    std::vector<float> g(dim), u(dim);
    for (int n = 0; n < dim; ++n) {
      if (swiglu) {
        g[n] = static_cast<float>(row[2 * n]);
        u[n] = static_cast<float>(row[2 * n + 1]);
      } else {
        g[n] = static_cast<float>(row[n]);
        u[n] = static_cast<float>(row[dim + n]);
      }
    }
    for (int n = 0; n < dim; ++n) {
      float v = ref_gated_act(act, g[n], u[n]);
      if constexpr (std::is_same_v<T, bfloat16_t>) row[n] = bfloat16_t(v);
      else                                         row[n] = static_cast<T>(v);
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// [1.e] Tolerance presets (bf16 / f32 / mixed)
// ═══════════════════════════════════════════════════════════════════════════════

struct Tol { float rel, abs; };
inline Tol tol_fused(bool is_bf16) { return is_bf16 ? Tol{0.20f, 0.05f} : Tol{5e-4f, 1e-4f}; }
inline Tol tol_act(bool is_bf16)   { return is_bf16 ? Tol{0.15f, 0.02f} : Tol{2e-4f, 1e-5f}; }
inline Tol tol_moe(bool is_bf16)   { return is_bf16 ? Tol{0.15f, 0.02f} : Tol{5e-4f, 1e-5f}; }

// ═══════════════════════════════════════════════════════════════════════════════
// [1.f] ALGO env var RAII guard
// ═══════════════════════════════════════════════════════════════════════════════

struct AlgoEnvGuard {
  std::string prev_value;
  bool had_prev = false;
  explicit AlgoEnvGuard(int algo) {
    if (const char *p = std::getenv("ZENDNNL_GRP_MATMUL_ALGO")) {
      prev_value = p; had_prev = true;
    }
    std::string s = std::to_string(algo);
    setenv("ZENDNNL_GRP_MATMUL_ALGO", s.c_str(), 1);
  }
  ~AlgoEnvGuard() {
    if (had_prev) setenv("ZENDNNL_GRP_MATMUL_ALGO", prev_value.c_str(), 1);
    else          unsetenv("ZENDNNL_GRP_MATMUL_ALGO");
  }
};

// RAII override for ZENDNNL_GRP_N_TILE_FUSED_ACT.  The production default
// is OFF (opt-in), so tests that need to exercise the fused-swiglu_oai
// epilogue must flip it on explicitly.  The env var is read per-call
// (no process caching), so toggling here takes effect immediately.
struct FusedActEnvGuard {
  std::string prev_value;
  bool had_prev = false;
  explicit FusedActEnvGuard(bool enable) {
    if (const char *p = std::getenv("ZENDNNL_GRP_N_TILE_FUSED_ACT")) {
      prev_value = p; had_prev = true;
    }
    setenv("ZENDNNL_GRP_N_TILE_FUSED_ACT", enable ? "1" : "0", 1);
  }
  ~FusedActEnvGuard() {
    if (had_prev) setenv("ZENDNNL_GRP_N_TILE_FUSED_ACT",
                         prev_value.c_str(), 1);
    else          unsetenv("ZENDNNL_GRP_N_TILE_FUSED_ACT");
  }
};

} // namespace moe_test_utils


// ═══════════════════════════════════════════════════════════════════════════════
// [2] TestGroupMatmul: F32_F32, BF16_F32, BF16_BF16 with optional MoE post-op
//
// Uses the framework's tensor_t / tensor_factory_t infrastructure and the
// reference matmul_forced_ref_kernel_test for correctness.  The three TEST_Ps
// share a common body via the templated run_basic_test helper.
// ═══════════════════════════════════════════════════════════════════════════════

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
    omp_set_num_threads(num_threads);
    num_ops = 2 + (rand() % 4);
    log_info("GroupMatmul test: m=", m, " k=", k, " n=", n,
             " transA=", transA, " transB=", transB,
             " alpha=", alpha, " beta=", beta,
             " num_ops=", num_ops, " num_threads=", num_threads);
  }
  virtual void TearDown() {}

  uint64_t m, k, n;
  bool transA, transB;
  tensor_factory_t tensor_factory{};
  float alpha, beta;
  matmul_algo_t algo;
  int32_t num_threads;
  size_t num_ops;

  // Shared body for the 3 parameterized tests (F32_F32, BF16_F32, BF16_BF16).
  // src_dt/wei_dt/dst_dt/bias_dt control the dtype configuration.
  void run_basic_test(data_type_t src_dt, data_type_t wei_dt,
                      data_type_t dst_dt, data_type_t bias_dt,
                      float rtol_pref, float eps_pref) {
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

    // Some BLAS kernels don't accept bias for libxsmm — disable.
    if (algo == matmul_algo_t::libxsmm || algo == matmul_algo_t::libxsmm_blocked) {
      for (auto &b : biases) b = nullptr;
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
      const bool skip_weighted = (src_dt == data_type_t::f32) && ((m + k + num_ops) % 3 == 0);
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
      tensor_t dummy = tensor_factory.zero_tensor({1, 1}, dst_dt);
      ref_st = matmul_forced_ref_kernel_test(in_t[i], wei_t[i], bias_t[i],
          out_ref_t[i], post_op_type_t::none, dummy, false, algo, alphas[i], betas[i]);
    }

    bool ok = (st == status_t::success && ref_st == status_t::success);

    if (ok && !enable_moe) {
      for (size_t i = 0; i < num_ops && ok; ++i)
        compare_tensor_2D_matrix(out_t[i], out_ref_t[i], m, n, k,
            rtol_pref, eps_pref, ok, false, alpha);
    }

    if (ok && enable_moe) {
      const float moe_abs_bound = (src_dt == data_type_t::f32)
          ? std::fabs(alpha) * ((20 + std::log2((float)k) / 4) * k + 15) * eps_pref
          : std::fabs(alpha) * (float)k * eps_pref;
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
            } else {
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
          } else {
            got = reinterpret_cast<const float *>(moe_output.data())[(size_t)t * D + d];
          }
          if (std::abs(acc - got) > moe_abs_bound + rtol_pref * std::abs(acc)) {
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

// TODO: INT8, WOQ, and dynamic-quant group matmul tests need dedicated fixtures
// (no transpose, alpha=1, beta=0, aligned K, compatible algos) — follow-up PR.

INSTANTIATE_TEST_SUITE_P(GroupMatmul, TestGroupMatmul,
                         ::testing::ValuesIn(matmul_test));

// ═══════════════════════════════════════════════════════════════════════════════
// [3] TestGatedAct: gated activation correctness (silu, gelu, swiglu)
// ═══════════════════════════════════════════════════════════════════════════════

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
  src.alloc   (num_ops, (size_t)M * K, p.is_bf16);
  wei.alloc   (num_ops, (size_t)K * N, p.is_bf16);
  dst_act.alloc(num_ops, (size_t)M * N, p.is_bf16);
  dst_ref.alloc(num_ops, (size_t)M * N, p.is_bf16);
  for (int e = 0; e < num_ops; ++e) {
    if (p.is_bf16) { fill_src(src.bf16[e], e, 0.05f); fill_wei1(wei.bf16[e], e, 0.01f); }
    else           { fill_src(src.f32[e],  e, 0.05f); fill_wei1(wei.f32[e],  e, 0.01f); }
  }

  auto gv = GemmVecs::uniform(num_ops, M, N, K, 1.0f, 0.0f, /*wc=*/true);
  auto srcs    = src.cptrs(p.is_bf16);
  auto weis    = wei.cptrs(p.is_bf16);
  auto dsts    = dst_act.ptrs(p.is_bf16);
  auto dsts_ref= dst_ref.ptrs(p.is_bf16);
  std::vector<const void *> biases(num_ops, nullptr);
  auto params  = make_uniform_params(num_ops, dt);

  // Run reference Op1 (no activation) → dst_ref.
  {
    auto pr = params;
    ASSERT_EQ(group_matmul_direct(gv.layout, gv.transA, gv.transB, gv.Ms, gv.Ns,
        gv.Ks, gv.alpha, srcs, gv.lda, weis, gv.ldb, biases, gv.beta,
        dsts_ref, gv.ldc, gv.is_wc, pr, nullptr, nullptr), status_t::success);
  }

  // Run fused Op1+Act → dst_act.
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
        } else {
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
  // Core: all act × dtype × dim at M=4, E=2.
  for (int a : {0, 1, 2, 3})
    for (bool bf : {false, true})
      for (int d : {1, 7, 15, 16, 17, 31, 32, 33, 64, 128, 255, 256, 512})
        out.push_back({d, 4, 2, a, bf});
  // Vary M and num_ops (skip duplicates of core M=4,E=2).
  for (int a : {1, 2, 3})
    for (bool bf : {false, true})
      for (int d : {1, 16, 33, 128})
        for (int m : {1, 4, 16, 64})
          for (int e : {2, 4, 8, 16}) {
            if (m == 4 && e == 2) continue;
            out.push_back({d, m, e, a, bf});
          }
  // Large-expert MoE-realistic configs.
  for (bool bf : {false, true})
    for (int e : {32, 64})
      for (int m : {1, 2})
        out.push_back({16, m, e, 1, bf});
  return out;
}

INSTANTIATE_TEST_SUITE_P(GroupMatmulGatedAct, TestGatedAct,
    ::testing::ValuesIn(make_gated_act_params()), GatedActParamName);

// ═══════════════════════════════════════════════════════════════════════════════
// [4] TestMoEPostop: weighted-reduce post-op correctness
// ═══════════════════════════════════════════════════════════════════════════════

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
  if (total_M % p.topk != 0) return;
  const int num_tokens = total_M / p.topk;
  const data_type_t dt = p.is_bf16 ? data_type_t::bf16 : data_type_t::f32;
  const size_t elem_sz = zendnnl::common::size_of(dt);

  TypedBuffers src, wei, dst, dst_ref;
  src.alloc    (p.num_ops, (size_t)p.M * p.K, p.is_bf16);
  wei.alloc    (p.num_ops, (size_t)p.K * p.N, p.is_bf16);
  dst.alloc    (p.num_ops, (size_t)p.M * p.N, p.is_bf16);
  dst_ref.alloc(p.num_ops, (size_t)p.M * p.N, p.is_bf16);
  for (int e = 0; e < p.num_ops; ++e) {
    // Keep init in f32 domain first, then mirror to bf16 (matches legacy pattern).
    std::vector<float> sf((size_t)p.M * p.K), wf((size_t)p.K * p.N);
    fill_src (sf, e, 0.01f);
    fill_wei1(wf, e, 0.005f);
    if (p.is_bf16) {
      for (size_t i = 0; i < sf.size(); ++i) src.bf16[e][i] = bfloat16_t(sf[i]);
      for (size_t i = 0; i < wf.size(); ++i) wei.bf16[e][i] = bfloat16_t(wf[i]);
    } else {
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

  // Build moe_postop: sequential row mapping (expert e, row j) → token j+e*M.
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
        if (expert >= p.num_ops) continue;
        const float val = dst_ref.at(expert, (size_t)row * p.N + d, p.is_bf16);
        acc += (p.skip_weighted ? 1.0f : moe_w[si]) * val;
      }
      float got;
      if (p.is_bf16)
        got = static_cast<float>(reinterpret_cast<const bfloat16_t *>(
            moe_out.data())[(size_t)t * p.N + d]);
      else
        got = reinterpret_cast<const float *>(moe_out.data())[(size_t)t * p.N + d];
      ASSERT_NEAR(got, acc, std::abs(acc) * tol.rel + tol.abs)
          << (p.is_bf16 ? "bf16" : "f32") << " E=" << p.num_ops
          << " M=" << p.M << " N=" << p.N << " topk=" << p.topk
          << (p.skip_weighted ? " skip" : "") << " t=" << t << " d=" << d;
    }
  }
}

static std::vector<MoEPostopTestParam> make_moe_postop_params() {
  std::vector<MoEPostopTestParam> out;
  for (bool bf : {false, true})
    for (int e : {2, 4, 8})
      for (int m : {2, 4, 8, 16})
        for (int topk : {1, 2})
          if ((e * m) % topk == 0) out.push_back({e, m, 64, 32, topk, false, bf});
  for (bool bf : {false, true}) out.push_back({4, 4, 64, 32, 2, true, bf});
  for (int n : {16, 128, 256})
    for (int k : {16, 64}) out.push_back({4, 4, n, k, 2, false, false});
  out.push_back({16, 2, 64, 32, 2, false, false});
  out.push_back({32, 1, 64, 32, 2, false, false});
  return out;
}

INSTANTIATE_TEST_SUITE_P(GroupMatmulMoEPostop, TestMoEPostop,
    ::testing::ValuesIn(make_moe_postop_params()), MoEPostopParamName);

// ═══════════════════════════════════════════════════════════════════════════════
// [5] TestFusedMoE: Op1(gate+up) → Act → Op2(down_proj) vs 2-call reference
// ═══════════════════════════════════════════════════════════════════════════════

struct FusedMoETestParam {
  int dim, hidden_size, M, num_ops, act_int;  // act_int: 0=none, 1=silu, 2=gelu, 3=swiglu_oai
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

  const auto &p = GetParam();
  const int dim = p.dim, N_gate_up = 2 * dim, H = p.hidden_size;
  const int M = p.M, K = H, num_ops = p.num_ops;
  const auto act_type = static_cast<grp_matmul_gated_act_t>(p.act_int);
  const data_type_t dt = p.is_bf16 ? data_type_t::bf16 : data_type_t::f32;

  // Allocate buffers.
  TypedBuffers src, w1, d1, d1_ref, w2, d2_fused, d2_ref;
  src     .alloc(num_ops, (size_t)M * K,         p.is_bf16);
  w1      .alloc(num_ops, (size_t)K * N_gate_up, p.is_bf16);
  d1      .alloc(num_ops, (size_t)M * N_gate_up, p.is_bf16);
  d1_ref  .alloc(num_ops, (size_t)M * N_gate_up, p.is_bf16);
  w2      .alloc(num_ops, (size_t)dim * H,       p.is_bf16);
  d2_fused.alloc(num_ops, (size_t)M * H,         p.is_bf16);
  d2_ref  .alloc(num_ops, (size_t)M * H,         p.is_bf16);
  for (int e = 0; e < num_ops; ++e) {
    if (p.is_bf16) {
      fill_src (src.bf16[e], e); fill_wei1(w1.bf16[e], e); fill_wei2(w2.bf16[e], e);
    } else {
      fill_src (src.f32[e],  e); fill_wei1(w1.f32[e],  e); fill_wei2(w2.f32[e],  e);
    }
  }

  auto gv_op1 = GemmVecs::uniform(num_ops, M, N_gate_up, K);
  auto gv_op2 = GemmVecs::uniform(num_ops, M, H,         dim);
  gv_op2.lda.assign(num_ops, N_gate_up);  // Op2 reads dst1 with stride 2*dim.

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

  // Reference path: Op1+Act into dst1_ref, then Op2 reading dst1_ref with stride.
  {
    auto pr1 = params;
    ASSERT_EQ(group_matmul_direct(gv_op1.layout, gv_op1.transA, gv_op1.transB,
        gv_op1.Ms, gv_op1.Ns, gv_op1.Ks, gv_op1.alpha, srcs, gv_op1.lda,
        wei1, gv_op1.ldb, no_bias, gv_op1.beta, dst1_ref, gv_op1.ldc,
        gv_op1.is_wc, pr1, nullptr, act_ptr), status_t::success);
    std::vector<const void *> srcs2(num_ops);
    for (int e = 0; e < num_ops; ++e) srcs2[e] = dst1_ref[e];
    auto pr2 = params;
    ASSERT_EQ(group_matmul_direct(gv_op2.layout, gv_op2.transA, gv_op2.transB,
        gv_op2.Ms, gv_op2.Ns, gv_op2.Ks, gv_op2.alpha, srcs2, gv_op2.lda,
        wei2, gv_op2.ldb, no_bias, gv_op2.beta, dst2_r, gv_op2.ldc,
        gv_op2.is_wc, pr2), status_t::success);
  }

  // Fused path: single call with &fused, writes dst2_fused.
  grp_matmul_fused_moe_params fused{};
  fused.down_weight = wei2;
  fused.N_down      = std::vector<int>(num_ops, H);
  fused.ldb_down    = std::vector<int>(num_ops, H);
  fused.bias_down   = no_bias;
  fused.dst_down    = dst2_f;
  fused.ldc_down    = std::vector<int>(num_ops, H);
  {
    auto pf = params;
    ASSERT_EQ(group_matmul_direct(gv_op1.layout, gv_op1.transA, gv_op1.transB,
        gv_op1.Ms, gv_op1.Ns, gv_op1.Ks, gv_op1.alpha, srcs, gv_op1.lda,
        wei1, gv_op1.ldb, no_bias, gv_op1.beta, dst1, gv_op1.ldc,
        gv_op1.is_wc, pf, nullptr, act_ptr, &fused), status_t::success);
  }

  // Compare dst2_fused vs dst2_ref.
  const auto tol = tol_fused(p.is_bf16);
  for (int e = 0; e < num_ops; ++e) {
    for (int r = 0; r < M; ++r) {
      for (int c = 0; c < H; ++c) {
        const float got = d2_fused.at(e, (size_t)r * H + c, p.is_bf16);
        const float ref = d2_ref  .at(e, (size_t)r * H + c, p.is_bf16);
        ASSERT_NEAR(got, ref, std::abs(ref) * tol.rel + tol.abs)
            << "act=" << p.act_int << (p.is_bf16 ? " bf16" : " f32")
            << " dim=" << dim << " h=" << H << " M=" << M
            << " E=" << num_ops << " e=" << e << " r=" << r << " c=" << c;
      }
    }
  }
}

static std::vector<FusedMoETestParam> make_fused_moe_params() {
  std::vector<FusedMoETestParam> out;
  // Core matrix: all 4 activation types × both dtypes × 3×3 shape grid.
  // Covers: none (skip-act path), silu/gelu (concatenated [gate|up] layout),
  // swiglu_oai_mul (interleaved [g0,u0,g1,u1,...] layout).
  for (int a : {0, 1, 2, 3})
    for (bool bf : {false, true})
      for (int d : {16, 32, 64})
        for (int h : {16, 32, 64})
          out.push_back({d, h, 4, 4, a, bf});
  // Vary M and num_ops for silu (concatenated layout) at dim=32, h=32.
  for (bool bf : {false, true})
    for (int m : {1, 4, 16})
      for (int e : {2, 4, 8}) {
        if (m == 4 && e == 4) continue;  // already in core
        out.push_back({32, 32, m, e, 1, bf});
      }
  // Vary M and num_ops for swiglu_oai (interleaved layout) — semantically
  // distinct from silu/gelu so it deserves independent M×E sweeps.
  for (bool bf : {false, true})
    for (int m : {1, 4, 16})
      for (int e : {2, 4, 8}) {
        if (m == 4 && e == 4) continue;  // already in core
        out.push_back({32, 32, m, e, 3, bf});
      }
  // BF16 realistic decode shapes (Qwen3-class small weights).
  for (int m : {1, 2, 8})
    for (int e : {4, 16})
      out.push_back({256, 128, m, e, 1, /*is_bf16=*/true});
  return out;
}

INSTANTIATE_TEST_SUITE_P(GroupMatmulFusedMoE, TestFusedMoE,
    ::testing::ValuesIn(make_fused_moe_params()), FusedMoEParamName);

// ═══════════════════════════════════════════════════════════════════════════════
// [6] TestGroupMatmulCombined: all 2³=8 combinations of (moe, act, fused)
// ═══════════════════════════════════════════════════════════════════════════════

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

class TestGroupMatmulCombined : public ::testing::TestWithParam<CombinedTestParam> {};

TEST_P(TestGroupMatmulCombined, AllCombinations) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;

  const auto &p = GetParam();
  const int H = 256, dim = 128, N_gate_up = 2 * dim, K = H, topk = 2;
  const int num_ops = p.num_ops, M = p.M;
  const data_type_t dt = p.is_bf16 ? data_type_t::bf16 : data_type_t::f32;

  // Op1 output width depends on whether act/fused is used.
  const bool need_gate_up = p.use_act || p.use_fused;
  const int N_op1 = need_gate_up ? N_gate_up : H;

  // Allocate all buffers (some won't be used — harmless).
  TypedBuffers src, w1, d1, d1_ref, w2, d2_fused, d2_ref;
  src    .alloc(num_ops, (size_t)M * K,    p.is_bf16);
  w1     .alloc(num_ops, (size_t)K * N_op1,p.is_bf16);
  d1     .alloc(num_ops, (size_t)M * N_op1,p.is_bf16);
  d1_ref .alloc(num_ops, (size_t)M * N_op1,p.is_bf16);
  w2     .alloc(num_ops, (size_t)dim * H,  p.is_bf16);
  d2_fused.alloc(num_ops, (size_t)M * H,   p.is_bf16);
  d2_ref  .alloc(num_ops, (size_t)M * H,   p.is_bf16);
  for (int e = 0; e < num_ops; ++e) {
    if (p.is_bf16) {
      fill_src (src.bf16[e], e); fill_wei1(w1.bf16[e], e); fill_wei2(w2.bf16[e], e);
    } else {
      fill_src (src.f32[e],  e); fill_wei1(w1.f32[e],  e); fill_wei2(w2.f32[e],  e);
    }
  }

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
    fused.down_weight = wei2;
    fused.N_down      = std::vector<int>(num_ops, H);
    fused.ldb_down    = std::vector<int>(num_ops, H);
    fused.bias_down   = no_bias;
    fused.dst_down    = dst2_f;
    fused.ldc_down    = std::vector<int>(num_ops, H);
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
    moe.output        = p.is_bf16 ? (void *)moe_out_b.data() : (void *)moe_out_f.data();
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
        } else {
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
      if (p.is_bf16) apply_ref_gated_act(d1_ref.bf16[e], M, N_op1, N_op1, act.act);
      else           apply_ref_gated_act(d1_ref.f32[e],  M, N_op1, N_op1, act.act);
    }
  }

  if (p.use_fused) {
    std::vector<const void *> srcs2(num_ops);
    for (int e = 0; e < num_ops; ++e) srcs2[e] = dst1_r[e];
    auto gv2 = GemmVecs::uniform(num_ops, M, H, dim);
    gv2.lda.assign(num_ops, N_op1);
    auto pr2 = params;
    ASSERT_EQ(group_matmul_direct(gv2.layout, gv2.transA, gv2.transB, gv2.Ms, gv2.Ns,
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
          if (p.use_fused)
            v = d2_ref.at(expert, (size_t)t * H + d, p.is_bf16);
          else
            v = d1_ref.at(expert, (size_t)t * N_op1 + d, p.is_bf16);
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
  } else if (p.use_fused) {
    for (int e = 0; e < num_ops; ++e)
      for (int r = 0; r < M; ++r)
        for (int c = 0; c < H; ++c)
          check(d2_fused.at(e, (size_t)r * H + c, p.is_bf16),
                d2_ref  .at(e, (size_t)r * H + c, p.is_bf16), "fused", e, r, c);
  } else if (p.use_act) {
    for (int e = 0; e < num_ops; ++e)
      for (int r = 0; r < M; ++r)
        for (int c = 0; c < dim; ++c)
          check(d1    .at(e, (size_t)r * N_op1 + c, p.is_bf16),
                d1_ref.at(e, (size_t)r * N_op1 + c, p.is_bf16), "act-only", e, r, c);
  } else {
    for (int e = 0; e < num_ops; ++e)
      for (int r = 0; r < M; ++r)
        for (int c = 0; c < N_op1; ++c)
          check(d1    .at(e, (size_t)r * N_op1 + c, p.is_bf16),
                d1_ref.at(e, (size_t)r * N_op1 + c, p.is_bf16), "plain", e, r, c);
  }
}

static std::vector<CombinedTestParam> make_combined_params() {
  std::vector<CombinedTestParam> out;
  for (bool moe : {false, true})
    for (bool act : {false, true})
      for (bool fused : {false, true})
        for (bool bf16 : {false, true})
          for (int m : {16, 64})
            for (int e : {2, 4})
              out.push_back({moe, act, fused, bf16, m, e});
  return out;
}

INSTANTIATE_TEST_SUITE_P(GroupMatmulCombined, TestGroupMatmulCombined,
    ::testing::ValuesIn(make_combined_params()), CombinedParamName);

// ═══════════════════════════════════════════════════════════════════════════════
// [7] TestFusedMoEAlgos: fused path × ALGO 1/2/3 × mixed precision × bias
// ═══════════════════════════════════════════════════════════════════════════════

struct FusedAlgoTestParam {
  int algo, act_int;
  bool is_bf16, mixed_prec, use_bias;
  int M, num_ops;
  // dim=0 → use default 128.  Larger dims (≥256) force N_gate_up ≥ 512,
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
  if (p.mixed_prec) name += "_bf16f32";
  else              name += (p.is_bf16 ? "_bf16" : "_f32");
  if (p.use_bias)   name += "_bias";
  name += "_M" + std::to_string(p.M) + "_E" + std::to_string(p.num_ops);
  if (p.dim > 0)    name += "_d" + std::to_string(p.dim);
  return name;
}

class TestFusedMoEAlgos : public ::testing::TestWithParam<FusedAlgoTestParam> {};

TEST_P(TestFusedMoEAlgos, Correctness) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;

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

  AlgoEnvGuard algo_guard(p.algo);
  // The fused-swiglu_oai epilogue in ALGO 3 is opt-in (default OFF).
  // Force it on for every test case in this suite so the correctness
  // check actually covers the fused path — in particular the shapes
  // below where N_gate_up > kDecodeNTile let the epilogue run with
  // n_thr > 1 threads per expert, which is the path the row-split fix
  // in apply_n_tile_paired_swiglu_oai is there to protect.
  FusedActEnvGuard fused_act_guard(/*enable=*/true);

  // Allocate: input-side may be bf16; output-side may differ (mixed_prec).
  TypedBuffers src, w1, d1, d1r, w2, d2, d2r;
  src.alloc(num_ops, (size_t)M * K,         use_bf16_in,  p.mixed_prec);
  w1 .alloc(num_ops, (size_t)K * N_gate_up, use_bf16_in,  p.mixed_prec);
  w2 .alloc(num_ops, (size_t)dim * H,       use_bf16_in,  p.mixed_prec);
  d1 .alloc(num_ops, (size_t)M * N_gate_up, use_bf16_out);
  d1r.alloc(num_ops, (size_t)M * N_gate_up, use_bf16_out);
  d2 .alloc(num_ops, (size_t)M * H,         use_bf16_out);
  d2r.alloc(num_ops, (size_t)M * H,         use_bf16_out);

  std::vector<std::vector<float>> bias_f(num_ops);
  for (int e = 0; e < num_ops; ++e) {
    // Always generate in f32 then mirror to bf16 when needed.
    std::vector<float> s_tmp((size_t)M * K), w1_tmp((size_t)K * N_gate_up),
                       w2_tmp((size_t)dim * H);
    fill_src (s_tmp,  e);
    fill_wei1(w1_tmp, e);
    fill_wei2(w2_tmp, e);
    if (src.store_f32) src.f32[e] = s_tmp;
    if (w1 .store_f32) w1.f32[e]  = w1_tmp;
    if (w2 .store_f32) w2.f32[e]  = w2_tmp;
    if (src.store_bf16) { src.bf16[e].resize(s_tmp.size());  for (size_t i=0;i<s_tmp.size();++i)  src.bf16[e][i] = bfloat16_t(s_tmp[i]); }
    if (w1 .store_bf16) { w1 .bf16[e].resize(w1_tmp.size()); for (size_t i=0;i<w1_tmp.size();++i) w1 .bf16[e][i] = bfloat16_t(w1_tmp[i]); }
    if (w2 .store_bf16) { w2 .bf16[e].resize(w2_tmp.size()); for (size_t i=0;i<w2_tmp.size();++i) w2 .bf16[e][i] = bfloat16_t(w2_tmp[i]); }
    bias_f[e].resize(H);
    for (int i = 0; i < H; ++i) bias_f[e][i] = 0.01f * ((i + e) % 5);
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
  if (p.use_bias) for (int e = 0; e < num_ops; ++e) bias2[e] = bias_f[e].data();

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
    for (int e = 0; e < num_ops; ++e) s2[e] = dst1r[e];
    auto gv2 = GemmVecs::uniform(num_ops, M, H, dim);
    gv2.lda.assign(num_ops, N_gate_up);
    auto pr2 = make_mixed_params(num_ops, dst_dt, wei_dt, dst_dt,
        p.use_bias ? data_type_t::f32 : data_type_t::none);
    ASSERT_EQ(group_matmul_direct(gv2.layout, gv2.transA, gv2.transB, gv2.Ms,
        gv2.Ns, gv2.Ks, gv2.alpha, s2, gv2.lda, wei2, gv2.ldb, bias2,
        gv2.beta, dst2r, gv2.ldc, gv2.is_wc, pr2), status_t::success)
        << "Ref Op2 failed";
  }

  // Fused path.
  grp_matmul_fused_moe_params fused{};
  fused.down_weight   = wei2;
  fused.N_down        = std::vector<int>(num_ops, H);
  fused.ldb_down      = std::vector<int>(num_ops, H);
  fused.bias_down     = bias2;
  fused.bias_dt_down  = p.use_bias ? data_type_t::f32 : data_type_t::none;
  fused.dst_down      = dst2;
  fused.ldc_down      = std::vector<int>(num_ops, H);
  {
    auto pf = make_mixed_params(num_ops, src_dt, wei_dt, dst_dt);
    ASSERT_EQ(group_matmul_direct(gv1.layout, gv1.transA, gv1.transB, gv1.Ms,
        gv1.Ns, gv1.Ks, gv1.alpha, srcs, gv1.lda, wei1, gv1.ldb, bias1,
        gv1.beta, dst1, gv1.ldc, gv1.is_wc, pf, nullptr, act_ptr, &fused),
        status_t::success) << "Fused call failed (algo=" << p.algo << ")";
  }

  // Compare.
  const auto tol = tol_fused(p.is_bf16 || p.mixed_prec);
  for (int e = 0; e < num_ops; ++e) {
    for (int r = 0; r < M; ++r) {
      for (int c = 0; c < H; ++c) {
        const float got = d2 .at(e, (size_t)r * H + c, use_bf16_out);
        const float ref = d2r.at(e, (size_t)r * H + c, use_bf16_out);
        ASSERT_NEAR(got, ref, std::abs(ref) * tol.rel + tol.abs)
            << "algo=" << p.algo << " act=" << p.act_int
            << (p.mixed_prec ? " bf16>f32" : (p.is_bf16 ? " bf16" : " f32"))
            << (p.use_bias ? " +bias" : "")
            << " e=" << e << " r=" << r << " c=" << c;
      }
    }
  }
}

static std::vector<FusedAlgoTestParam> make_fused_algo_params() {
  std::vector<FusedAlgoTestParam> out;
  // All 3 ALGOs × all 4 activation types × both dtypes.
  // Covers: ALGO-specific fused MoE dispatch paths for every activation.
  for (int algo : {1, 2, 3})
    for (int act : {0, 1, 2, 3})  // none, silu, gelu, swiglu
      for (bool bf : {false, true})
        out.push_back({algo, act, bf, false, false, 4, 4});
  // Mixed precision (BF16 src → F32 dst) per ALGO.
  for (int algo : {1, 2, 3}) out.push_back({algo, 1, true,  true,  false, 4, 4});
  // Non-null down_proj bias per ALGO.
  for (int algo : {1, 2, 3}) out.push_back({algo, 1, false, false, true,  4, 4});
  // ALGO 2 M-tile with varying M (small M=1 and larger M=16).
  for (int m : {1, 16})      out.push_back({2,    1, false, false, false, m, 4});
  // ALGO 3 two-pass with many experts.
  out.push_back({3, 1, false, false, false, 4, 8});

  // ALGO 3 fused swiglu_oai_mul — race-exposure shapes.
  //
  // Pre-fix, apply_n_tile_paired_swiglu_oai split the epilogue by N
  // columns, which aliased thread t's compact-output writes
  // [p_start_t, p_end_t) with a lower-index thread's pair-read range
  // [2·p_start_{t-1}, 2·p_start_t).  The race only fires when
  // flat_n_tile actually runs more than one thread per expert —
  // i.e. (1) `ntile_viable` is true and (2) `thr_per_expert` (or the
  // path-B `n_thr`) resolves to ≥2.  Every earlier swiglu case in this
  // file used dim ≤ 128 so N_gate_up ≤ 256 = kDecodeNTile → fallback →
  // one thread per expert → the bug stayed hidden.
  //
  // The shapes below hit the race-prone code on both a typical 16-thread
  // developer run AND a 128-thread EPYC CI run (verified with a Python
  // simulation of flat_n_tile's decision tree):
  //
  //   shape (M=8, E=8, d=1024):
  //       16t  → path (B), n_thr=2 per expert, 2-round batched N-tile
  //      128t  → path (D), n_thr=8 per expert, decode_parallel
  //
  //   shape (M=64, E=8, d=2048):
  //       16t  → path (B), n_thr=2 per expert
  //      128t  → path (A), n_thr=8 per expert, L3-batched few-expert
  //
  // Together they exercise all three multi-threaded epilogue paths
  // (D/A/B) and make the fix observable: each thread writes into a
  // disjoint row slice instead of the aliased pair→compact column
  // slice, so there is no cross-thread overlap.  Pre-fix these shapes
  // returned NaNs / wrong arithmetic that exceed the BF16 tolerance;
  // post-fix they match the 2-pass reference.
  for (bool bf : {false, true}) {
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

// ═══════════════════════════════════════════════════════════════════════════════
// [8] TestFusedMoENegative: validation error paths (fast-fail)
// ═══════════════════════════════════════════════════════════════════════════════

namespace {

// Helper to build a minimal fused-MoE API invocation for negative tests.
// The caller patches specific fields on the returned fused struct or the
// GemmVecs to trigger the desired validation failure.
struct NegCtx {
  std::vector<float> s, w, d, wd, dd;
  std::vector<const void *> sp, wp, bp;
  std::vector<void *> dp;
  moe_test_utils::GemmVecs gv;
  std::vector<matmul_params> pa;
  grp_matmul_fused_moe_params fused{};

  NegCtx(int n, int M, int N, int K, int N_down, char layout = 'r') {
    s.assign((size_t)M * K * n, 1.0f);
    w.assign((size_t)K * N * n, 1.0f);
    d.assign((size_t)M * N * n, 0.0f);
    wd.assign((size_t)(N / 2) * N_down * n, 1.0f);
    dd.assign((size_t)M * N_down * n, 0.0f);
    sp.assign(n, s.data());
    wp.assign(n, w.data());
    bp.assign(n, nullptr);
    dp.assign(n, d.data());
    gv = moe_test_utils::GemmVecs::uniform(n, M, N, K);
    gv.layout.assign(n, layout);
    pa = moe_test_utils::make_uniform_params(n, data_type_t::f32);
    fused.down_weight.assign(n, wd.data());
    fused.N_down     .assign(n, N_down);
    fused.ldb_down   .assign(n, N_down);
    fused.bias_down  .assign(n, nullptr);
    fused.dst_down   .assign(n, dd.data());
    fused.ldc_down   .assign(n, N_down);
  }

  status_t call() {
    return group_matmul_direct(gv.layout, gv.transA, gv.transB, gv.Ms, gv.Ns,
        gv.Ks, gv.alpha, sp, gv.lda, wp, gv.ldb, bp, gv.beta, dp, gv.ldc,
        gv.is_wc, pa, nullptr, nullptr, &fused);
  }
};

} // namespace

class TestFusedMoENegative : public ::testing::Test {};

TEST_F(TestFusedMoENegative, OddN_Rejected) {
  NegCtx c(2, /*M=*/1, /*N=*/33, /*K=*/32, /*N_down=*/32);  // N=33 is odd
  EXPECT_EQ(c.call(), status_t::failure) << "Odd N should be rejected";
}

TEST_F(TestFusedMoENegative, NullDownWeight_Rejected) {
  NegCtx c(2, 1, 64, 32, 32);
  c.fused.down_weight[0] = nullptr;
  EXPECT_EQ(c.call(), status_t::failure) << "Null down_weight should be rejected";
}

TEST_F(TestFusedMoENegative, SizeMismatch_Rejected) {
  NegCtx c(2, 1, 64, 32, 32);
  c.fused.N_down.pop_back();  // size 1, should be 2
  EXPECT_EQ(c.call(), status_t::failure) << "Size mismatch should be rejected";
}

TEST_F(TestFusedMoENegative, ColumnMajor_Rejected) {
  NegCtx c(2, 1, 64, 32, 32, /*layout=*/'c');
  EXPECT_EQ(c.call(), status_t::failure)
      << "Column-major should be rejected for fused_moe";
}

TEST_F(TestFusedMoENegative, SequentialMode_Rejected) {
  NegCtx c(1, 1, 64, 32, 32);  // src.size() == 1 → sequential mode
  EXPECT_EQ(c.call(), status_t::failure)
      << "Sequential mode should reject fused_moe";
}

// ldb_down must be >= (transB ? K_down : N_down).  K_down = N/2 = 32,
// N_down = 32; with transB=false the minimum is 32, so 31 must fail.
TEST_F(TestFusedMoENegative, LdbDown_TooSmall_Rejected) {
  NegCtx c(2, 1, /*N=*/64, 32, /*N_down=*/32);
  c.fused.ldb_down[0] = 31;
  EXPECT_EQ(c.call(), status_t::failure)
      << "ldb_down below min should be rejected";
}

// ldc_down must be >= N_down.
TEST_F(TestFusedMoENegative, LdcDown_TooSmall_Rejected) {
  NegCtx c(2, 1, 64, 32, /*N_down=*/32);
  c.fused.ldc_down[0] = 31;
  EXPECT_EQ(c.call(), status_t::failure)
      << "ldc_down below N_down should be rejected";
}
