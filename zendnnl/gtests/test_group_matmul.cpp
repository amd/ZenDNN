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
///   [1]  Shared helpers (moe_test_utils): buffers, init, params, ref math
///   [2]  TestGroupMatmul           — F32/BF16 basic + moe_postop coverage
///   [3]  TestGatedAct              — gated activation correctness
///   [4]  TestMoEPostop             — weighted-reduce post-op correctness
///   [5]  TestFusedMoE              — fused Op1→Act→Op2 (legacy mode:
///                                    caller-allocated dst & dst_down)
///   [5b] TestFusedMoEInternalAlloc — same fused MoE, but internal-alloc
///                                    + src-reuse mode (caller passes
///                                    dst all-null and fused.dst_down
///                                    empty; library allocates Op1
///                                    scratch and writes Op2 back into
///                                    src in place)
///   [6]  TestGroupMatmulCombined   — all 2³=8 combinations (moe,act,fused)
///   [7]  TestFusedMoEAlgos         — fused path × ALGOs × mixed precision
///   [7b] TestFusedMoEAlgoCustom    — fused MoE × strategy/tight/custom
///                                    BF16 ukernel env-knob matrix
///   [8]  TestGroupMatmulAlgoCustom — non-fused custom BF16 ukernel matrix
///   [9]  TestFusedMoENegative      — validation fast-fail paths
///   [10] TestGroupMatmulQuant      — WOQ / INT8 / dynamic-quant suites,
///                                    parameterised by GroupQuantMatmulType
///                                    (constrained: alpha=1, beta=0, no
///                                    transpose, K aligned to 4) so that
///                                    activated comparison stays in a
///                                    well-bounded numerical regime.
///
/// Coverage matrix (separate suites for each optional feature):
///   1. GEMM + activation only        → TestGatedAct
///      • all 4 acts (none/silu/gelu/swiglu), both f32/bf16, many dim×M×E
///   2. GEMM + moe_postop only        → TestMoEPostop
///      • weighted-reduce + skip_weighted, both f32/bf16, many E×M×topk
///   3. GEMM + fused MoE (Op1→Act→Op2)→ TestFusedMoE, TestFusedMoEAlgos,
///                                       TestFusedMoEInternalAlloc
///      • all 4 acts (none/silu/gelu/swiglu), both f32/bf16,
///        mixed precision (bf16→f32), with/without down_proj bias,
///        per-ALGO coverage, both legacy and internal-alloc modes.
///   4. All three combined            → TestGroupMatmulCombined
///      • 2³=8 combinations of (moe, act, fused) × both dtypes × key shapes.
///   5. Quantized GEMM + random gated act → TestGroupMatmulQuant
///      • WOQ (S4/U4), symmetric INT8 (per-group / per-token / dynamic),
///        random act selection per-test (skipped on odd N), reference
///        applies apply_ref_gated_act_tensor on the un-activated reference
///        and compares only the activated [0:N/2) half via
///        compare_activated_2D's three-regime envelope.
/// BF16 is verified in every suite; it is the primary precision for MoE inference.

#include <gtest/gtest.h>
#include <cmath>
#include <cstdio>
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
    if constexpr(std::is_same_v<T, bfloat16_t>) {
      vec[i] = bfloat16_t(v);
    }
    else {
      vec[i] = static_cast<T>(v);
    }
  }
}

// Standard fill presets used across tests (picked so activation doesn't overflow).
inline void fill_src(std::vector<float> &v, int e, float s = 0.02f)   {
  fill_pattern(v, e, 7, 11, 5, s);
}
inline void fill_src(std::vector<bfloat16_t> &v, int e, float s = 0.02f) {
  fill_pattern(v, e, 7, 11, 5, s);
}
inline void fill_wei1(std::vector<float> &v, int e, float s = 0.005f)  {
  fill_pattern(v, e, 3, 7, 3, s);
}
inline void fill_wei1(std::vector<bfloat16_t> &v, int e, float s = 0.005f) {
  fill_pattern(v, e, 3, 7, 3, s);
}
inline void fill_wei2(std::vector<float> &v, int e, float s = 0.008f)  {
  fill_pattern(v, e, 5, 9, 4, s);
}
inline void fill_wei2(std::vector<bfloat16_t> &v, int e, float s = 0.008f) {
  fill_pattern(v, e, 5, 9, 4, s);
}

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
    if (store_f32)  {
      f32.resize(num_ops);
      for (auto &v : f32) {
        v.assign(elems, 0.0f);
      }
    }
    if (store_bf16) {
      bf16.resize(num_ops);
      for (auto &v : bf16) {
        v.assign(elems, bfloat16_t(0.0f));
      }
    }
  }

  // Opaque pointer view selector: `bf16 = true` picks bf16 storage.
  std::vector<const void *> cptrs(bool pick_bf16) const {
    std::vector<const void *> out(pick_bf16 ? bf16.size() : f32.size());
    if (pick_bf16) for (size_t e = 0; e < bf16.size(); ++e) {
        out[e] = bf16[e].data();
      }
    else           for (size_t e = 0; e < f32.size();  ++e) {
        out[e] = f32[e].data();
      }
    return out;
  }
  std::vector<void *> ptrs(bool pick_bf16) {
    std::vector<void *> out(pick_bf16 ? bf16.size() : f32.size());
    if (pick_bf16) for (size_t e = 0; e < bf16.size(); ++e) {
        out[e] = bf16[e].data();
      }
    else           for (size_t e = 0; e < f32.size();  ++e) {
        out[e] = f32[e].data();
      }
    return out;
  }
  float at(int e, size_t i, bool is_bf16) const {
    return is_bf16 ? static_cast<float>(bf16[e][i]) : f32[e][i];
  }
};

// ═══════════════════════════════════════════════════════════════════════════════
// [1.c] matmul_params & GEMM wrapper vector builders
// ═══════════════════════════════════════════════════════════════════════════════

inline std::vector<matmul_params> make_uniform_params(int num_ops,
    data_type_t dt,
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
inline float ref_gated_act(grp_matmul_gated_act_t act, float g_or_even,
                           float u_or_odd) {
  switch (act) {
  case grp_matmul_gated_act_t::silu_and_mul:
    return ref_silu_mul(g_or_even, u_or_odd);
  case grp_matmul_gated_act_t::gelu_and_mul:
    return ref_gelu_mul(g_or_even, u_or_odd);
  case grp_matmul_gated_act_t::swiglu_oai_mul:
    return ref_swiglu_oai(g_or_even, u_or_odd);
  default:
    return 0.0f;
  }
}

// Apply activation in-place on dst[:, 0:dim].  dst has M rows with stride ldc.
// Matches the kernel semantics in group_matmul_moe_act.cpp.
template <typename T>
inline void apply_ref_gated_act(T *dst, int M, int N, int ldc,
                                grp_matmul_gated_act_t act) {
  if (act == grp_matmul_gated_act_t::none) {
    return;
  }
  const int dim = N / 2;
  const bool swiglu = (act == grp_matmul_gated_act_t::swiglu_oai_mul);
  for (int m = 0; m < M; ++m) {
    T *row = dst + m * ldc;
    // Read raw values first (since we write to row[n] below, which may be read).
    std::vector<float> g(dim), u(dim);
    for (int n = 0; n < dim; ++n) {
      if (swiglu) {
        g[n] = static_cast<float>(row[2 * n]);
        u[n] = static_cast<float>(row[2 * n + 1]);
      }
      else {
        g[n] = static_cast<float>(row[n]);
        u[n] = static_cast<float>(row[dim + n]);
      }
    }
    for (int n = 0; n < dim; ++n) {
      float v = ref_gated_act(act, g[n], u[n]);
      if constexpr(std::is_same_v<T, bfloat16_t>) {
        row[n] = bfloat16_t(v);
      }
      else {
        row[n] = static_cast<T>(v);
      }
    }
  }
}

template <typename T>
inline void apply_ref_gated_act(std::vector<T> &dst, int M, int N, int ldc,
                                grp_matmul_gated_act_t act) {
  apply_ref_gated_act<T>(dst.data(), M, N, ldc, act);
}

// Tensor-aware reference activation: dispatches on dtype and operates on the
// underlying contiguous buffer.  The reference activation only writes columns
// [0, N/2); columns [N/2, N) keep the raw GEMM output, matching the kernel
// contract that those positions are "don't care".
inline void apply_ref_gated_act_tensor(tensor_t &t, int M, int N, int ldc,
                                       grp_matmul_gated_act_t act) {
  if (act == grp_matmul_gated_act_t::none) {
    return;
  }
  const data_type_t dtype = t.get_data_type();
  if (dtype == data_type_t::f32) {
    apply_ref_gated_act<float>(
      static_cast<float *>(t.get_raw_handle_unsafe()), M, N, ldc, act);
  }
  else if (dtype == data_type_t::bf16) {
    apply_ref_gated_act<bfloat16_t>(
      static_cast<bfloat16_t *>(t.get_raw_handle_unsafe()), M, N, ldc, act);
  }
  // Other dtypes intentionally unsupported — gated_act validation in
  // group_matmul_direct rejects non-f32/bf16 dst before reaching the kernel.
}

// Pick a random gated activation type compatible with the given output width.
// Returns `none` when N is odd (gated activations require even N), otherwise
// uniformly samples from {none, silu_and_mul, gelu_and_mul, swiglu_oai_mul}.
// Uses the supplied RNG so the choice is reproducible per test parameter set.
inline grp_matmul_gated_act_t pick_random_gated_act(uint64_t n,
                                                    std::mt19937 &rng) {
  if (n % 2 != 0 || n < 2) {
    return grp_matmul_gated_act_t::none;
  }
  return static_cast<grp_matmul_gated_act_t>(rng() % 4);
}

// Activation-aware comparison for the [0, N/2) columns of an activated
// kernel output against the post-`apply_ref_gated_act_tensor` reference.
// The gated activation `silu(g) * u` (or analogues) amplifies the per-row
// GEMM noise multiplicatively: an input pair (g, u) carrying error ~δ
// each produces output error roughly (|g| + |u|) · δ.  Both |g| and
// |u| can be O(alpha · k) — especially with quantization, where
// dequantized weights have a non-zero mean so the GEMM accumulator
// scales linearly in k rather than √k.  Combined with the alpha · k · ε
// per-element matmul noise, that yields an activated error budget on
// the order of alpha² · k² · ε for some configurations.  We use a
// generous mixed bound: 4 · alpha² · k · ε absolute (matches the
// loose-but-stable tolerance the gated MoE reference comparisons in
// this file already use) plus 30% relative.  These aren't tight bounds;
// they're "test does not regress on reasonable shapes" bounds.
// Three-regime tolerance envelope for the activated [0, N/2) comparison.
// Activation amplifies per-element kernel↔reference noise multiplicatively
// (silu(g) · u → output error ~2 · max(|g|, |u|) · δ where δ is the per-
// element GEMM noise).  A flat rel%+abs bound undershoots near-zero
// reference values (where the kernel sees milli-scale drift even when
// the reference rounds to exactly 0), so we max three regimes:
//
//   floor_abs   max(16·α²·k·ε, 0.5).  Hard 0.5 floor handles INT8 quant
//               outliers — `epsilon = ε_f32 ≈ 1e-7` would otherwise give
//               a sub-millis floor that doesn't bound INT8 noise near 0.
//   sqrt_term   16·α·k·ε · √|ref| — dominates the moderate-|ref| regime
//               where |g|, |u| ≈ √|ref|.
//   rel_term    30% · |ref| — takes over for very large |ref|.
//
// The bound is a "no order-of-magnitude regression" envelope, not a
// BF16-precise tracking bound.  When `ok` flips false the per-element
// fprintf prints the full breakdown (regime contributions) so the
// dominating term is obvious in the test log; only the FIRST miss is
// printed (subsequent ones short-circuit via `i < m && ok`).
inline void compare_activated_2D(const tensor_t &out, const tensor_t &out_ref,
                                 uint64_t m, uint64_t cmp_n, uint64_t k,
                                 float alpha, float epsilon,
                                 bool &ok) {
  if (!ok) {
    return;
  }
  const float kf = static_cast<float>(k);
  const float a_abs = std::fabs(alpha);
  const float floor_abs_formula = 16.0f * a_abs * a_abs * kf * epsilon;
  const float floor_abs   = std::max(floor_abs_formula, 0.5f);
  const float sqrt_factor = 16.0f * a_abs * kf * epsilon;
  const float rel_bound   = 0.30f;
  for (uint64_t i = 0; i < m && ok; ++i) {
    for (uint64_t j = 0; j < cmp_n && ok; ++j) {
      const float a = const_cast<tensor_t &>(out).at({i, j});
      const float r = const_cast<tensor_t &>(out_ref).at({i, j});
      const float r_abs     = std::fabs(r);
      const float sqrt_term = sqrt_factor * std::sqrt(r_abs);
      const float rel_term  = rel_bound * r_abs;
      float allowed = floor_abs;
      if (sqrt_term > allowed) {
        allowed = sqrt_term;
      }
      if (rel_term > allowed) {
        allowed = rel_term;
      }
      const float abs_err = std::fabs(a - r);
      if (abs_err > allowed) {
        ok = false;
      }
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// [1.e] Tolerance presets (bf16 / f32 / mixed)
// ═══════════════════════════════════════════════════════════════════════════════

struct Tol {
  float rel, abs;
};
inline Tol tol_fused(bool is_bf16) {
  return is_bf16 ? Tol{0.20f, 0.05f} :
         Tol{5e-4f, 1e-4f};
}
inline Tol tol_act(bool is_bf16)   {
  return is_bf16 ? Tol{0.15f, 0.02f} :
         Tol{2e-4f, 1e-5f};
}
inline Tol tol_moe(bool is_bf16)   {
  return is_bf16 ? Tol{0.15f, 0.02f} :
         Tol{5e-4f, 1e-5f};
}

// ═══════════════════════════════════════════════════════════════════════════════
// [1.f] ALGO env var RAII guard
// ═══════════════════════════════════════════════════════════════════════════════

struct AlgoEnvGuard {
  std::string prev_value;
  bool had_prev = false;
  explicit AlgoEnvGuard(int algo) {
    if (const char *p = std::getenv("ZENDNNL_GRP_MATMUL_ALGO")) {
      prev_value = p;
      had_prev = true;
    }
    std::string s = std::to_string(algo);
    setenv("ZENDNNL_GRP_MATMUL_ALGO", s.c_str(), 1);
  }
  ~AlgoEnvGuard() {
    if (had_prev) {
      setenv("ZENDNNL_GRP_MATMUL_ALGO", prev_value.c_str(), 1);
    }
    else {
      unsetenv("ZENDNNL_GRP_MATMUL_ALGO");
    }
  }
};

// Generic RAII setter for any single env var (saves & restores prior
// value).  Used to flip optional feature flags ON for tests that must
// exercise the gated code path (e.g. ZENDNNL_GRP_MATMUL_N_TILE_FUSED_ACT for
// the ALGO 3 fused-swiglu_oai epilogue).
struct EnvVarGuard {
  const char *name;
  std::string prev_value;
  bool had_prev = false;
  EnvVarGuard(const char *env_name, const char *new_value) : name(env_name) {
    if (const char *p = std::getenv(name)) {
      prev_value = p;
      had_prev = true;
    }
    setenv(name, new_value, 1);
  }
  ~EnvVarGuard() {
    if (had_prev) {
      setenv(name, prev_value.c_str(), 1);
    }
    else {
      unsetenv(name);
    }
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
  // Core: all act × dtype × dim at M=4, E=2.
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

// ═══════════════════════════════════════════════════════════════════════════════
// [5] TestFusedMoE: Op1(gate+up) → Act → Op2(down_proj) vs 2-call reference
// ═══════════════════════════════════════════════════════════════════════════════

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
  for (int e = 0; e < num_ops; ++e) {
    if (p.is_bf16) {
      fill_src(src.bf16[e], e);
      fill_wei1(w1.bf16[e], e);
      fill_wei2(w2.bf16[e], e);
    }
    else {
      fill_src(src.f32[e],  e);
      fill_wei1(w1.f32[e],  e);
      fill_wei2(w2.f32[e],  e);
    }
  }

  auto gv_op1 = GemmVecs::uniform(num_ops, M, N_gate_up, K);
  auto gv_op2 = GemmVecs::uniform(num_ops, M, H,         K_down);
  gv_op2.lda.assign(num_ops, N_gate_up);  // Op2 reads dst1 with stride N_gate_up.

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
    for (int e = 0; e < num_ops; ++e) {
      srcs2[e] = dst1_ref[e];
    }
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
  // Vary M and num_ops for swiglu_oai (interleaved layout) — semantically
  // distinct from silu/gelu so it deserves independent M×E sweeps.
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
  // (GPT-OSS-class decode workload — `dim=128, h=128`).  Earlier test
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

// ═══════════════════════════════════════════════════════════════════════════════
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
// ═══════════════════════════════════════════════════════════════════════════════

class TestFusedMoEInternalAlloc
  : public ::testing::TestWithParam<FusedMoETestParam> {};

TEST_P(TestFusedMoEInternalAlloc, Correctness) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;

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
  // internal-alloc path — receives Op2 output in-place).
  TypedBuffers src_orig, src_intalloc, w1, d1_ref, w2, d2_ref;
  src_orig    .alloc(num_ops, (size_t)M * K,         p.is_bf16);
  src_intalloc.alloc(num_ops, (size_t)M * K,         p.is_bf16);
  w1          .alloc(num_ops, (size_t)K * N_gate_up, p.is_bf16);
  d1_ref      .alloc(num_ops, (size_t)M * N_gate_up, p.is_bf16);
  w2          .alloc(num_ops, (size_t)K_down * H,    p.is_bf16);
  d2_ref      .alloc(num_ops, (size_t)M * H,         p.is_bf16);
  for (int e = 0; e < num_ops; ++e) {
    if (p.is_bf16) {
      fill_src(src_orig.bf16[e], e);
      fill_src(src_intalloc.bf16[e], e);
      fill_wei1(w1.bf16[e], e);
      fill_wei2(w2.bf16[e], e);
    }
    else {
      fill_src(src_orig.f32[e], e);
      fill_src(src_intalloc.f32[e], e);
      fill_wei1(w1.f32[e], e);
      fill_wei2(w2.f32[e], e);
    }
  }

  auto gv_op1 = GemmVecs::uniform(num_ops, M, N_gate_up, K);
  auto gv_op2 = GemmVecs::uniform(num_ops, M, H,         K_down);
  gv_op2.lda.assign(num_ops, N_gate_up);

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

  // Reference path: two-call legacy → dst2_r.
  {
    auto pr1 = params;
    ASSERT_EQ(group_matmul_direct(gv_op1.layout, gv_op1.transA, gv_op1.transB,
                                  gv_op1.Ms, gv_op1.Ns, gv_op1.Ks, gv_op1.alpha, srcs_orig, gv_op1.lda,
                                  wei1, gv_op1.ldb, no_bias, gv_op1.beta, dst1_ref, gv_op1.ldc,
                                  gv_op1.is_wc, pr1, nullptr, act_ptr), status_t::success);
    std::vector<const void *> srcs2(num_ops);
    for (int e = 0; e < num_ops; ++e) {
      srcs2[e] = dst1_ref[e];
    }
    auto pr2 = params;
    ASSERT_EQ(group_matmul_direct(gv_op2.layout, gv_op2.transA, gv_op2.transB,
                                  gv_op2.Ms, gv_op2.Ns, gv_op2.Ks, gv_op2.alpha, srcs2, gv_op2.lda,
                                  wei2, gv_op2.ldb, no_bias, gv_op2.beta, dst2_r, gv_op2.ldc,
                                  gv_op2.is_wc, pr2), status_t::success);
  }

  // Internal-alloc fused path: dst[] = nullptr, fused.dst_down empty.
  // Library allocates the per-expert Op1 scratch, runs Op1 + act + Op2,
  // and writes Op2 output back into srcs_intalloc.
  grp_matmul_fused_moe_params fused{};
  fused.down_weight = wei2;
  fused.N_down      = std::vector<int>(num_ops, H);
  fused.ldb_down    = std::vector<int>(num_ops, H);
  fused.bias_down   = no_bias;
  // fused.dst_down and fused.ldc_down INTENTIONALLY left empty — the
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

  // Compare srcs_intalloc (now contains Op2 output, row stride lda=K=H)
  // against the reference dst2_r (row stride H).
  const auto tol = tol_fused(p.is_bf16);
  for (int e = 0; e < num_ops; ++e) {
    for (int r = 0; r < M; ++r) {
      for (int c = 0; c < H; ++c) {
        const float got = src_intalloc.at(e, (size_t)r * K + c, p.is_bf16);
        const float ref = d2_ref      .at(e, (size_t)r * H + c, p.is_bf16);
        ASSERT_NEAR(got, ref, std::abs(ref) * tol.rel + tol.abs)
            << "act=" << p.act_int << (p.is_bf16 ? " bf16" : " f32")
            << " dim=" << dim << " h=" << H << " M=" << M
            << " E=" << num_ops << " e=" << e << " r=" << r << " c=" << c;
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(GroupMatmulFusedMoEInternalAlloc,
                         TestFusedMoEInternalAlloc,
                         ::testing::ValuesIn(make_fused_moe_params()), FusedMoEParamName);

// ═══════════════════════════════════════════════════════════════════════════════
// [5c] TestFusedMoEInternalAllocMixedM — internal-alloc + per-expert M skew.
//
// Real MoE decode routes only `topk` experts per token, so per-expert M
// values vary (most experts at small M, a few at zero, occasional
// hot-expert with larger M).  Earlier tests use uniform M across all
// experts, missing this dimension.  This test injects M vectors taken
// from real GPT-OSS decode frames — it covers the (num_ops > 16,
// mixed-M, internal_alloc, custom-kernel-engaged) cube that wasn't
// previously exercised.  Shapes are scaled down for test runtime.
// ═══════════════════════════════════════════════════════════════════════════════

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
  for (int e = 0; e < num_ops; ++e) {
    if (p.is_bf16) {
      fill_src(src_orig.bf16[e], e); fill_src(src_intalloc.bf16[e], e);
      fill_wei1(w1.bf16[e], e); fill_wei2(w2.bf16[e], e);
    } else {
      fill_src(src_orig.f32[e], e); fill_src(src_intalloc.f32[e], e);
      fill_wei1(w1.f32[e], e); fill_wei2(w2.f32[e], e);
    }
  }

  // Per-expert vectors (using the real M_per_expert distribution).
  GemmVecs gv_op1, gv_op2;
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

  gv_op2 = gv_op1;
  gv_op2.Ns .assign(num_ops, H);
  gv_op2.Ks .assign(num_ops, K_down);
  gv_op2.lda.assign(num_ops, N_gate_up);  // src for Op2 = Op1 dst at stride N_gate_up
  gv_op2.ldb.assign(num_ops, H);
  gv_op2.ldc.assign(num_ops, H);

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

  // Reference: legacy two-call path.
  {
    auto pr1 = params;
    ASSERT_EQ(group_matmul_direct(gv_op1.layout, gv_op1.transA, gv_op1.transB,
        gv_op1.Ms, gv_op1.Ns, gv_op1.Ks, gv_op1.alpha, srcs_orig, gv_op1.lda,
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

  // Internal-alloc fused path.
  grp_matmul_fused_moe_params fused{};
  fused.down_weight = wei2;
  fused.N_down      = std::vector<int>(num_ops, H);
  fused.ldb_down    = std::vector<int>(num_ops, H);
  fused.bias_down   = no_bias;
  std::vector<void *>  dst_null(num_ops, nullptr);
  std::vector<int>     ldc_null(num_ops, 0);
  {
    auto pf = params;
    ASSERT_EQ(group_matmul_direct(gv_op1.layout, gv_op1.transA, gv_op1.transB,
        gv_op1.Ms, gv_op1.Ns, gv_op1.Ks, gv_op1.alpha, srcs_intalloc, gv_op1.lda,
        wei1, gv_op1.ldb, no_bias, gv_op1.beta, dst_null, ldc_null,
        gv_op1.is_wc, pf, nullptr, act_ptr, &fused), status_t::success);
  }

  // Compare per active expert (skip M=0 — Op2 doesn't write that slot).
  const auto tol = tol_fused(p.is_bf16);
  for (int e = 0; e < num_ops; ++e) {
    const int M_e = p.M_per_expert[e];
    if (M_e == 0) continue;
    for (int r = 0; r < M_e; ++r) {
      for (int c = 0; c < H; ++c) {
        const float got = src_intalloc.at(e, (size_t)r * K + c, p.is_bf16);
        const float ref = d2_ref      .at(e, (size_t)r * H + c, p.is_bf16);
        ASSERT_NEAR(got, ref, std::abs(ref) * tol.rel + tol.abs)
            << "act=" << p.act_int << (p.is_bf16 ? " bf16" : " f32")
            << " dim=" << dim << " h=" << H
            << " E=" << num_ops << " e=" << e
            << " M_e=" << M_e << " r=" << r << " c=" << c;
      }
    }
  }
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

class TestGroupMatmulCombined :
  public ::testing::TestWithParam<CombinedTestParam> {};

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

  // Allocate all buffers (some won't be used — harmless).
  TypedBuffers src, w1, d1, d1_ref, w2, d2_fused, d2_ref;
  src    .alloc(num_ops, (size_t)M * K,        p.is_bf16);
  w1     .alloc(num_ops, (size_t)K * N_op1,    p.is_bf16);
  d1     .alloc(num_ops, (size_t)M * N_op1,    p.is_bf16);
  d1_ref .alloc(num_ops, (size_t)M * N_op1,    p.is_bf16);
  w2     .alloc(num_ops, (size_t)K_down * H,   p.is_bf16);
  d2_fused.alloc(num_ops, (size_t)M * H,       p.is_bf16);
  d2_ref  .alloc(num_ops, (size_t)M * H,       p.is_bf16);
  for (int e = 0; e < num_ops; ++e) {
    if (p.is_bf16) {
      fill_src(src.bf16[e], e);
      fill_wei1(w1.bf16[e], e);
      fill_wei2(w2.bf16[e], e);
    }
    else {
      fill_src(src.f32[e],  e);
      fill_wei1(w1.f32[e],  e);
      fill_wei2(w2.f32[e],  e);
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
  // ZENDNNL_GRP_MATMUL_N_TILE_FUSED_ACT (default OFF) — see
  // get_grp_n_tile_fused_act() in group_matmul_parallel_common.hpp.
  // Force it ON here so the shapes below (where N_gate_up >
  // kDecodeNTile) drive ALGO 3's per-thread fused epilogue with
  // n_thr > 1 threads per expert.  That is the row-split path the
  // matmul→activation barrier + GroupNTileContext::apply_swiglu_oai
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
  // Mixed precision (BF16 src → F32 dst) per ALGO.
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

// ═══════════════════════════════════════════════════════════════════════════════
// [7b] TestFusedMoEAlgoCustom: fused-MoE env-knob matrix
//
// Mirrors TestGroupMatmulAlgoCustom for the non-fused path but for the
// fused MoE entry (Op1+act → Op2).  Targets the strategy-selection
// contract cemented in Option A:
//
//   * ZENDNNL_GRP_MATMUL_ALGO = 1..5         — strategy selector, the
//                                              single source of truth
//                                              for the fused path.
//   * ZENDNNL_GRP_MATMUL_FUSED_MOE_TIGHT=0/1 — V1 vs V2 (V2 is the
//                                              ALGO 3 + swiglu + tight
//                                              specialist; engages only
//                                              when env_algo ∈ {0, 3}).
//   * ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL=0/1   — custom BF16 ukernel hook
//                                              (flat_n_tile + V2 Op2).
//
// Reference strategy: a single known-good baseline — forced ALGO=1,
// TIGHT=0, CUSTOM=0, legacy two-call (Op1 via group_matmul_direct +
// Op2 via group_matmul_direct) — compared against the fused
// internal-alloc call under the parameterised env.  Any breakage of
// Option A's gating surface (e.g. V2 silently engaging for ALGO 5,
// or CUSTOM_KERNEL=1 corrupting Op2 for non-BF16 dtypes) produces a
// comparison failure.
// ═══════════════════════════════════════════════════════════════════════════════

struct FusedAlgoCustomParam {
  int algo;            // ALGO strategy 1..5
  int tight;           // 0 or 1 (FUSED_MOE_TIGHT)
  int custom_kernel;   // 0 or 1
  int act_int;         // 1=silu, 2=gelu, 3=swiglu_oai (act=none skipped —
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
  for (int e = 0; e < num_ops; ++e) {
    fill_src(src_ref  .bf16[e], e);
    fill_src(src_fused.bf16[e], e);
    fill_wei1(w1.bf16[e], e);
    fill_wei2(w2.bf16[e], e);
  }

  auto gv_op1 = GemmVecs::uniform(num_ops, M, N_gate_up, K);
  auto gv_op2 = GemmVecs::uniform(num_ops, M, H, K_down);
  gv_op2.lda.assign(num_ops, N_gate_up);

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

  // ── Reference: ALGO=1, TIGHT=0, CUSTOM=0, two-call legacy ────────
  // Everything else in the test can be evaluated against this single
  // baseline.  The `AlgoEnvGuard(1)` + cleared TIGHT / CUSTOM guards
  // are scoped to this block so they don't contaminate the later
  // parameterised run.
  {
    AlgoEnvGuard algo_guard(1);
    EnvVarGuard tight_guard("ZENDNNL_GRP_MATMUL_FUSED_MOE_TIGHT", "0");
    EnvVarGuard custom_guard("ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL",   "0");

    auto pr1 = params;
    ASSERT_EQ(group_matmul_direct(gv_op1.layout, gv_op1.transA, gv_op1.transB,
                                  gv_op1.Ms, gv_op1.Ns, gv_op1.Ks, gv_op1.alpha, srcs_ref, gv_op1.lda,
                                  wei1, gv_op1.ldb, no_bias, gv_op1.beta, dst1_ref, gv_op1.ldc,
                                  gv_op1.is_wc, pr1, nullptr, act_ptr),
              status_t::success) << "Ref Op1 failed";

    std::vector<const void *> srcs_op2(num_ops);
    for (int e = 0; e < num_ops; ++e) {
      srcs_op2[e] = dst1_ref[e];
    }
    auto pr2 = params;
    ASSERT_EQ(group_matmul_direct(gv_op2.layout, gv_op2.transA, gv_op2.transB,
                                  gv_op2.Ms, gv_op2.Ns, gv_op2.Ks, gv_op2.alpha, srcs_op2, gv_op2.lda,
                                  wei2, gv_op2.ldb, no_bias, gv_op2.beta, dst2_r, gv_op2.ldc,
                                  gv_op2.is_wc, pr2), status_t::success) << "Ref Op2 failed";
  }

  // ── Test: parameterised ALGO × TIGHT × CUSTOM, internal-alloc fused ──
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

    grp_matmul_fused_moe_params fused{};
    fused.down_weight = wei2;
    fused.N_down      = std::vector<int>(num_ops, H);
    fused.ldb_down    = std::vector<int>(num_ops, H);
    fused.bias_down   = no_bias;
    // fused.dst_down / ldc_down intentionally empty → internal-alloc.

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

  // ── Compare: src_fused now holds Op2 output (in-place, stride lda=K=H) ──
  const auto tol = tol_fused(is_bf16);
  for (int e = 0; e < num_ops; ++e) {
    for (int r = 0; r < M; ++r) {
      for (int c = 0; c < H; ++c) {
        const float got = src_fused.at(e, (size_t)r * K + c, is_bf16);
        const float ref = d2_ref  .at(e, (size_t)r * H + c, is_bf16);
        ASSERT_NEAR(got, ref, std::abs(ref) * tol.rel + tol.abs)
            << "algo=" << p.algo << " tight=" << p.tight
            << " custom=" << p.custom_kernel
            << " act=" << p.act_int
            << " e=" << e << " r=" << r << " c=" << c;
      }
    }
  }
}

static std::vector<FusedAlgoCustomParam> make_fused_algo_custom_params() {
  std::vector<FusedAlgoCustomParam> out;
  // All 5 ALGOs × TIGHT {0,1} × CUSTOM {0,1} × act {silu, gelu, swiglu}.
  // TIGHT=1 with act ∈ {silu, gelu} or ALGO ∉ {0,3} exercises the gate
  // that routes back to V1 (Option A) — expected to produce identical
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
          // M=4 × num_ops=4 keeps the shape small enough for fast
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

// ═══════════════════════════════════════════════════════════════════════════════
// [8] TestGroupMatmulAlgoCustom: non-fused Phase B env-knob matrix
//
// Targets the env-knob combinations that gate the custom BF16 microkernel's
// engagement in the non-fused group_matmul path (ALGO 3 via flat_n_tile):
//
//   * GRP_MATMUL_ALGO          — 1 (sequential_experts), 3 (flat_n_tile),
//                                5 (per_expert); covers "custom kernel
//                                engages", "no engagement but same output",
//                                and "per-expert distribution" respectively.
//   * CUSTOM_KERNEL            — 0 vs 1; 0 is the trusted standard path.
//   * N_TILE_FUSED_ACT         — 0 vs 1 on swiglu; forces the inline
//                                fused-swiglu epilogue (otherwise the
//                                caller does a separate post-pass).
//   * gated_act                — none / silu / gelu / swiglu_oai_mul.
//   * bias dtype               — none / bf16 / fp32 (fp32 bias on bf16 dst
//                                exercises the BiasKind::fp32 load path).
//   * moe_postop (weighted)    — off / on.
//
// Reference strategy: run the same call twice — once with CUSTOM_KERNEL=0
// (standard dispatch, already verified by TestGroupMatmul / TestGatedAct /
// TestMoEPostop) and once with CUSTOM_KERNEL at the parameterised value
// — and assert bit-identical (for configs where the custom kernel doesn't
// actually engage, e.g. ALGO 1 or ALGO 5) or within BF16 tolerance.  This
// covers the whole product of env toggles that the Phase B code added
// without duplicating the scalar-reference math elsewhere in this file.
// ═══════════════════════════════════════════════════════════════════════════════

struct AlgoCustomParam {
  int algo;               // GRP_MATMUL_ALGO strategy (1, 3, 5)
  int custom_kernel;      // 0 or 1
  int n_tile_fused_act;   // 0 or 1 (only meaningful for ALGO 3 + swiglu)
  int act_int;            // 0=none, 1=silu, 2=gelu, 3=swiglu_oai
  int bias_kind;          // 0=none, 1=bf16, 2=fp32
  int M, num_ops, dim;
  // transB toggle — exercises BOTH caller layouts the custom-kernel
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
  // half — that region's contents legitimately differ between the
  // custom-kernel path (leaves cols [dim:2*dim] at zero) and the
  // standard path (leaves them at the raw Op1 GEMM output), making a
  // differential comparison ill-defined for act ∈ {swiglu}.  The
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

// One parameterised call — sets the CUSTOM_KERNEL env to `custom_value`,
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

  const auto &p = GetParam();

  // N_op1 is the Op1 GEMM output width.  For act=none the output is
  // just `dim` cols wide; for any gated activation the GEMM produces
  // 2*dim cols and the activation compacts to [0:dim].
  const int N_op1 = (p.act_int == 0) ? p.dim : 2 * p.dim;
  const int K     = 64;

  // Custom kernel requires N_op1 % pack_nr == 0 (32 or 64).  When the
  // grid lands on a smaller or misaligned N_op1 the custom path will
  // cleanly fall back to the standard dispatch — the `== 0` case is
  // still valuable because it regression-tests the fallback path.

  AlgoEnvGuard algo_guard(p.algo);
  EnvVarGuard fused_act_guard("ZENDNNL_GRP_MATMUL_N_TILE_FUSED_ACT",
                              p.n_tile_fused_act ? "1" : "0");

  // ── Prepare shared src / wei / bias (both runs see identical inputs) ──
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

  // ── Reference run: CUSTOM_KERNEL=0 ─────────────────────────────────
  std::vector<std::vector<bfloat16_t>> dst_ref(p.num_ops,
      std::vector<bfloat16_t>((size_t)p.M * N_op1, bfloat16_t(0.0f)));
  ASSERT_NO_FATAL_FAILURE(
    run_one_algo_custom_pass(p, "0", src, wei, bias_bf16, bias_fp32,
                             N_op1, K, dst_ref));

  // ── Test run: CUSTOM_KERNEL as parameterised ──────────────────────
  std::vector<std::vector<bfloat16_t>> dst_test(p.num_ops,
      std::vector<bfloat16_t>((size_t)p.M * N_op1, bfloat16_t(0.0f)));
  ASSERT_NO_FATAL_FAILURE(
    run_one_algo_custom_pass(p,
                             p.custom_kernel ? "1" : "0",
                             src, wei, bias_bf16, bias_fp32,
                             N_op1, K, dst_test));

  // ── Compare ────────────────────────────────────────────────────────
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
  // Strategy coverage — 1 (sequential), 3 (flat_n_tile = custom hook),
  // 5 (per-expert).  Skip 2 and 4 to keep the grid tight; those
  // executors don't look at CUSTOM_KERNEL anyway (ALGO 3 is the only
  // engagement site in the non-fused path).
  const int algos[] = {1, 3, 5};
  // dim=64 → N_op1=128 for act=none, N_op1=128 for gated; K=64.
  // This lets ALGO 3 exercise its N-tile split on ≥2 threads while
  // keeping the test shape small enough to run fast.
  const int dim = 64;

  // Core grid — every (algo × custom × act × bias) combo.
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
            // transB sweep — exercises both [K,N] and [N,K] caller
            // layouts.  The custom-kernel pack now supports both,
            // and this differential test catches any addressing
            // regression on the transB=true path that would
            // otherwise only show up in framework integrations.
            for (int tB : {0, 1}) {
              // M=4 × num_ops=8 keeps the shape small enough to run
              // fast under 8×64 sharding.  dim=64 → N_op1=128 for the
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

// ═══════════════════════════════════════════════════════════════════════════════
// [9] TestFusedMoENegative: validation error paths (fast-fail)
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

// ═══════════════════════════════════════════════════════════════════════════════
// [10] TestGroupMatmulQuant: dedicated fixture for quantized group-matmul
//
// The shared `MatmulType` random grid (alpha, beta ∈ U[0,10]; random
// transA/transB; arbitrary K) drives the quantized kernels into regimes
// where no useful tolerance can be derived for activated comparison —
// silu(g)·u-style outputs inherit an O(α²·k²·ε) noise term that swamps
// any rel/abs bound for large α·β.  Per the long-standing TODO at the
// bottom of this file, the quantized suites move to a dedicated fixture
// with hardcoded alpha=1, beta=0, no-transpose, K aligned to 4, and the
// compatible-algos algo selector baked into `GroupQuantMatmulType`.
//
// Within the fixture, alpha / beta / transA / transB are `static
// constexpr` members so the migrated test bodies use them as plain
// member names (no rename needed) and the un-activated comparison
// tolerances scale exactly as before for alpha=1.
// ═══════════════════════════════════════════════════════════════════════════════

class TestGroupMatmulQuant
    : public ::testing::TestWithParam<GroupQuantMatmulType> {
 protected:
  virtual void SetUp() {
    GroupQuantMatmulType params = GetParam();
    srand(static_cast<unsigned int>(seed));
    m            = params.matmul_m;
    k            = params.matmul_k;
    n            = params.matmul_n;
    algo         = params.algo;
    num_threads  = params.num_threads;
    source_dtype       = params.source_dtype;
    output_dtype       = params.output_dtype;
    weight_granularity = params.weight_granularity;
    omp_set_num_threads(num_threads);
    num_ops = 2 + (rand() % 4);
    log_info("GroupMatmulQuant test: m=", m, " k=", k, " n=", n,
             " (alpha=1 beta=0 no-transpose) num_ops=", num_ops,
             " num_threads=", num_threads);
  }
  /** @brief TearDown is used to free resource used in test */
  virtual void TearDown() {
    clear_matmul_test_caches();
  }

  uint64_t m, k, n;
  matmul_algo_t algo{};
  int32_t num_threads{};
  data_type_t source_dtype{};
  data_type_t output_dtype{};
  quant_granularity_t weight_granularity{};
  size_t num_ops{};
  tensor_factory_t tensor_factory{};

  // Pinned by construction — these are the constraints the TODO calls for.
  // Members rather than locals so the migrated test bodies (which read
  // `alpha`, `beta`, `transA`, `transB`) keep working unchanged.
  static constexpr float alpha = 1.0f;
  static constexpr float beta  = 0.0f;
  static constexpr bool  transA = false;
  static constexpr bool  transB = false;
};

/** @fn TEST_P
 *  @param TestGroupMatmulQuant parameterized test class
 *  @param WOQ_BF16_S4 user-defined name of test
 *  @brief Test to validate group_matmul_direct weight-only quantization with
 *         BF16 activation and signed 4-bit (S4) weights. Randomly selects
 *         per-tensor, per-channel, or per-group scale granularity and
 *         scale dtype (F32/BF16). Output dtype is randomly BF16 or F32.
 *         Skips oneDNN algo paths.
 */
TEST_P(TestGroupMatmulQuant, WOQ_BF16_S4) {
  if (algo == matmul_algo_t::onednn || algo == matmul_algo_t::onednn_blocked) {
    GTEST_SKIP() << "WOQ_BF16_S4 is not supported for oneDNN-based algorithms.";
  }

  int quant_combo = rand() % 3;

  std::vector<uint64_t> scale_size;
  uint64_t group_size = 0;
  uint64_t num_groups = 1;

  std::vector<uint64_t> valid_group_sizes;
  for (uint64_t gs = 2; gs <= k; gs += 2) {
    if (k % gs == 0) {
      valid_group_sizes.push_back(gs);
    }
  }
  bool has_valid_group_size = !valid_group_sizes.empty() || (k % 2 == 0);
  if (valid_group_sizes.empty() && k % 2 == 0) {
    valid_group_sizes.push_back(k);
  }
  if (!has_valid_group_size && quant_combo == 2) {
    quant_combo = 1;
  }

  switch (quant_combo) {
  case 0:
    scale_size = {1, 1};
    break;
  case 1:
    scale_size = {1, n};
    break;
  case 2: {
    uint64_t gs = valid_group_sizes[rand() % valid_group_sizes.size()];
    group_size = gs;
    num_groups = k / group_size;
    scale_size = {num_groups, n};
    break;
  }
  }

  auto scale_dtype = (rand() % 2 == 0) ? data_type_t::f32 : data_type_t::bf16;

  std::vector<tensor_t> inp(num_ops), wt(num_ops), bias(num_ops),
      out(num_ops), out_ref(num_ops), wei_scale(num_ops);

  auto out_dt = rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32;

  for (size_t i = 0; i < num_ops; ++i) {
    wei_scale[i] = tensor_factory.uniform_dist_tensor(scale_size, scale_dtype, 2.0);
    wt[i]      = tensor_factory.uniform_dist_tensor({k, n}, data_type_t::s4, 7.0,
                 transB, wei_scale[i]);
    inp[i]     = tensor_factory.uniform_dist_tensor({m, k}, data_type_t::bf16, 2.0,
                 transA);
    bias[i]    = tensor_factory.uniform_dist_tensor({1, n},
                 rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
    out[i]     = tensor_factory.uniform_dist_tensor({m, n}, out_dt, 2.0);
    out_ref[i] = tensor_factory.uniform_dist_tensor({m, n}, out_dt, 2.0);
  }

  // Random gated activation: paired with the WOQ kernel exercises
  // bf16-src + s4-wei + (bf16/f32)-dst dispatch through the gated
  // epilogue.  `pick_random_gated_act` returns `none` when N is odd
  // since gated activations require even N.  Activation must agree
  // across experts (the validator enforces uniform dst dtype, so we
  // pick once outside the loop).
  std::mt19937 act_rng((m << 1) ^ (k << 7) ^ (n << 13) ^ 0xAC54u);
  grp_matmul_gated_act_t act_type = moe_test_utils::pick_random_gated_act(
                                      static_cast<uint64_t>(n), act_rng);
  grp_matmul_gated_act_params act_params{};
  act_params.act = act_type;
  const grp_matmul_gated_act_params *act_ptr =
    (act_type != grp_matmul_gated_act_t::none) ? &act_params : nullptr;

  status_t status = group_matmul_kernel_test(inp, wt, bias, out, algo, alpha,
                    beta, /*moe_postop=*/nullptr, act_ptr);

  std::vector<post_op_type_t> ref_po;
  status_t ref_status = status_t::success;
  for (size_t i = 0; i < num_ops && ref_status == status_t::success; ++i) {
    std::vector<tensor_t> bin;
    ref_status = matmul_forced_ref_kernel_test(inp[i], wt[i], bias[i],
                 out_ref[i], ref_po, bin, true, algo, alpha, beta);
  }

  bool ok = (status == status_t::success && ref_status == status_t::success);
  if (ok) {
    // For activated runs apply the scalar gated-activation reference
    // to out_ref in-place (modifies cols [0:N/2]) and compare only the
    // activated half — cols [N/2:N] are "don't care" per the kernel
    // contract (TestGroupMatmulAlgoCustom comment).  Activated values
    // amplify noise multiplicatively, so use the looser
    // `compare_activated_2D` bound instead of `compare_tensor_2D_matrix`.
    if (act_type != grp_matmul_gated_act_t::none) {
      const float eps = (out_dt == data_type_t::bf16) ? epsilon_bf16
                        : epsilon_woq;
      for (size_t i = 0; i < num_ops; ++i) {
        moe_test_utils::apply_ref_gated_act_tensor(
          out_ref[i], static_cast<int>(m), static_cast<int>(n),
          static_cast<int>(n), act_type);
      }
      for (size_t i = 0; i < num_ops && ok; ++i)
        moe_test_utils::compare_activated_2D(out[i], out_ref[i], m, n / 2, k,
                                             alpha, eps, ok);
    }
    else {
      for (size_t i = 0; i < num_ops && ok; ++i)
        compare_tensor_2D_matrix(out[i], out_ref[i], m, n, k,
                                 out_dt == data_type_t::bf16 ? rtol_bf16 : rtol_woq,
                                 out_dt == data_type_t::bf16 ? epsilon_bf16 : epsilon_woq,
                                 ok, false, alpha, true);
    }
  }
  EXPECT_TRUE(ok);
}

/** @fn TEST_P
 *  @param TestGroupMatmulQuant parameterized test class
 *  @param WOQ_BF16_U4 user-defined name of test
 *  @brief Test to validate group_matmul_direct weight-only quantization with
 *         BF16 activation and unsigned 4-bit (U4) weights. Exercises
 *         asymmetric quantization with zero-point support. Randomly selects
 *         per-tensor, per-channel, or per-group scale/zp granularity, zp
 *         dtype (BF16/S8), scale dtype (F32/BF16), and output dtype
 *         (BF16/F32).
 */
TEST_P(TestGroupMatmulQuant, WOQ_BF16_U4) {
  int quant_combo = (rand() + k) % 4;
  bool random_zp_domain = (rand() + k) % 2 == 0;

  std::vector<uint64_t> scale_size, zp_size;
  uint64_t group_size = 0;
  uint64_t num_groups = 1;

  std::vector<uint64_t> valid_group_sizes;
  for (uint64_t gs = 2; gs <= k; gs += 2) {
    if (k % gs == 0) {
      valid_group_sizes.push_back(gs);
    }
  }
  bool has_valid_group_size = !valid_group_sizes.empty() || (k % 2 == 0);
  if (valid_group_sizes.empty() && k % 2 == 0) {
    valid_group_sizes.push_back(k);
  }
  if (!has_valid_group_size && (quant_combo == 2 || quant_combo == 3)) {
    quant_combo = 1;
  }

  switch (quant_combo) {
  case 0:
    scale_size = {1, 1};
    zp_size = {1, 1};
    break;
  case 1:
    scale_size = {1, n};
    zp_size = {1, 1};
    break;
  case 2:
    scale_size = {1, n};
    zp_size = {1, n};
    break;
  case 3: {
    uint64_t gs = valid_group_sizes[rand() % valid_group_sizes.size()];
    group_size = gs;
    num_groups = k / group_size;
    scale_size = {num_groups, n};
    zp_size    = {num_groups, n};
    break;
  }
  }

  auto scale_dtype = (rand() % 2 == 0) ? data_type_t::f32 : data_type_t::bf16;
  auto out_dt = rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32;

  std::vector<tensor_t> inp(num_ops), wt(num_ops), bias(num_ops),
      out(num_ops), out_ref(num_ops),
      w_scale(num_ops), w_zp(num_ops);

  for (size_t i = 0; i < num_ops; ++i) {
    w_scale[i] = tensor_factory.uniform_dist_tensor(scale_size, scale_dtype, 2.0);
    w_zp[i]    = tensor_factory.uniform_dist_tensor(zp_size,
                 random_zp_domain ? data_type_t::bf16 : data_type_t::s8,
                 random_zp_domain ? 2.0 : 25.0);
    wt[i]      = tensor_factory.uniform_dist_tensor({k, n}, data_type_t::u4,
                 15.0, transB, w_scale[i], w_zp[i]);
    inp[i]     = tensor_factory.uniform_dist_tensor({m, k}, data_type_t::bf16, 2.0,
                 transA);
    bias[i]    = tensor_factory.uniform_dist_tensor({1, n},
                 rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
    out[i]     = tensor_factory.uniform_dist_tensor({m, n}, out_dt, 2.0);
    out_ref[i] = tensor_factory.uniform_dist_tensor({m, n}, out_dt, 2.0);
  }

  // Random gated activation (asymmetric U4 WOQ + gated epilogue).
  std::mt19937 act_rng((m << 1) ^ (k << 7) ^ (n << 13) ^ 0xACE4u);
  grp_matmul_gated_act_t act_type = moe_test_utils::pick_random_gated_act(
                                      static_cast<uint64_t>(n), act_rng);
  grp_matmul_gated_act_params act_params{};
  act_params.act = act_type;
  const grp_matmul_gated_act_params *act_ptr =
    (act_type != grp_matmul_gated_act_t::none) ? &act_params : nullptr;

  status_t status = group_matmul_kernel_test(inp, wt, bias, out, algo, alpha,
                    beta, /*moe_postop=*/nullptr, act_ptr);

  std::vector<post_op_type_t> ref_po;
  status_t ref_status = status_t::success;
  for (size_t i = 0; i < num_ops && ref_status == status_t::success; ++i) {
    std::vector<tensor_t> bin;
    ref_status = matmul_forced_ref_kernel_test(inp[i], wt[i], bias[i],
                 out_ref[i], ref_po, bin, true, algo, alpha, beta);
  }

  bool ok = (status == status_t::success && ref_status == status_t::success);
  if (ok) {
    if (act_type != grp_matmul_gated_act_t::none) {
      const float eps = (out_dt == data_type_t::bf16) ? epsilon_bf16
                        : epsilon_woq;
      for (size_t i = 0; i < num_ops; ++i) {
        moe_test_utils::apply_ref_gated_act_tensor(
          out_ref[i], static_cast<int>(m), static_cast<int>(n),
          static_cast<int>(n), act_type);
      }
      for (size_t i = 0; i < num_ops && ok; ++i)
        moe_test_utils::compare_activated_2D(out[i], out_ref[i], m, n / 2, k,
                                             alpha, eps, ok);
    }
    else {
      for (size_t i = 0; i < num_ops && ok; ++i)
        compare_tensor_2D_matrix(out[i], out_ref[i], m, n, k,
                                 out_dt == data_type_t::bf16 ? rtol_bf16 : rtol_woq,
                                 out_dt == data_type_t::bf16 ? epsilon_bf16 : epsilon_woq,
                                 ok, false, alpha, true);
    }
  }
  EXPECT_TRUE(ok);
}

/** @fn TEST_P
 *  @param TestGroupMatmulQuant parameterized test class
 *  @param INT8 user-defined name of test
 *  @brief Test to validate group_matmul_direct INT8 (S8 weight) kernel support.
 *         Uses parameterized source_dtype/output_dtype and weight_granularity
 *         (per-tensor or per-channel) to exercise asymmetric dynamic
 *         quantization paths. Constrains alpha=1, beta=0.
 */
TEST_P(TestGroupMatmulQuant, INT8) {
  std::vector<uint64_t> wei_scale_size = (weight_granularity ==
                                          quant_granularity_t::tensor) ?
                                         std::vector<uint64_t> {1, 1} :
                                         std::vector<uint64_t> {1, n};
  data_type_t ref_dt = (output_dtype == data_type_t::f32) ? data_type_t::f32
                       : data_type_t::bf16;

  std::vector<tensor_t> inp(num_ops), wt(num_ops), bias(num_ops),
      out(num_ops), out_ref(num_ops);

  for (size_t i = 0; i < num_ops; ++i) {
    auto wei_ref = tensor_factory.uniform_dist_tensor({k, n}, ref_dt, 25.0, transB);
    tensor_t weight_tensor, wei_scale, wei_zp;
    if (quant_params_compute(tensor_factory, wei_ref, ref_dt, data_type_t::s8, {
    static_cast<int64_t>(wei_scale_size[0]),
      static_cast<int64_t>(wei_scale_size[1])
    },
    data_type_t::f32, wei_scale, wei_zp, &weight_tensor) != status_t::success) {
      FAIL() << "weight dynamic quantization failed";
    }

    auto src_ref = tensor_factory.uniform_dist_tensor({m, k}, ref_dt, 25.0, transA);
    tensor_t input_tensor, src_scale, src_zp;
    if (quant_params_compute(tensor_factory, src_ref, ref_dt, source_dtype,
  {1, 1}, data_type_t::f32, src_scale, src_zp,
  &input_tensor) != status_t::success) {
      FAIL() << "source dynamic quantization failed";
    }

    tensor_t dst_scale, dst_zp;
    if (output_dtype != data_type_t::f32 && output_dtype != data_type_t::bf16) {
      auto dst_ref = tensor_factory.uniform_dist_tensor({m, n}, ref_dt, 2.0);
      if (quant_params_compute(tensor_factory, dst_ref, ref_dt, output_dtype,
      {1, 1}, data_type_t::f32, dst_scale, dst_zp) != status_t::success) {
        FAIL() << "destination scale/zp computation failed";
      }
    }

    inp[i]     = input_tensor;
    wt[i]      = weight_tensor;
    bias[i]    = tensor_factory.uniform_dist_tensor({1, n},
                 rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
    out[i]     = tensor_factory.uniform_dist_tensor({m, n}, output_dtype, 2.0,
                 false, dst_scale, dst_zp);
    out_ref[i] = tensor_factory.uniform_dist_tensor({m, n}, output_dtype, 2.0,
                 false, dst_scale, dst_zp);
  }

  // Random gated activation — only when dst is f32/bf16.  Quantized
  // dst dtypes (s8/u8/...) are rejected by the gated_act validator.
  // INT8 GEMMs always run with alpha=1, beta=0 (see kernel call below),
  // so the bound here is informational; real bounding is per-suite.
  std::mt19937 act_rng((m << 1) ^ (k << 7) ^ (n << 13) ^ 0xACE8u);
  const bool act_dtype_ok = (output_dtype == data_type_t::f32
                             || output_dtype == data_type_t::bf16);
  grp_matmul_gated_act_t act_type = act_dtype_ok
                                    ? moe_test_utils::pick_random_gated_act(
                                        static_cast<uint64_t>(n), act_rng)
                                    : grp_matmul_gated_act_t::none;
  grp_matmul_gated_act_params act_params{};
  act_params.act = act_type;
  const grp_matmul_gated_act_params *act_ptr =
    (act_type != grp_matmul_gated_act_t::none) ? &act_params : nullptr;

  status_t status = group_matmul_kernel_test(inp, wt, bias, out, algo, 1.0f,
                    0.0f, /*moe_postop=*/nullptr, act_ptr);

  std::vector<post_op_type_t> ref_po;
  status_t ref_status = status_t::success;
  for (size_t i = 0; i < num_ops && ref_status == status_t::success; ++i) {
    std::vector<tensor_t> bin;
    ref_status = matmul_forced_ref_kernel_test(inp[i], wt[i], bias[i],
                 out_ref[i], ref_po, bin, true, algo, 1.0, 0.0);
  }

  bool ok = (status == status_t::success && ref_status == status_t::success);
  if (ok) {
    if (act_type != grp_matmul_gated_act_t::none) {
      for (size_t i = 0; i < num_ops; ++i) {
        moe_test_utils::apply_ref_gated_act_tensor(
          out_ref[i], static_cast<int>(m), static_cast<int>(n),
          static_cast<int>(n), act_type);
      }
      for (size_t i = 0; i < num_ops && ok; ++i)
        moe_test_utils::compare_activated_2D(out[i], out_ref[i], m, n / 2, k,
                                             1.0f, epsilon_bf16, ok);
    }
    else {
      for (size_t i = 0; i < num_ops && ok; ++i)
        compare_tensor_2D_matrix(out[i], out_ref[i], m, n, k,
                                 rtol_bf16, epsilon_bf16, ok, false, 1.0f, true);
    }
  }
  EXPECT_TRUE(ok);
}

/** @fn TEST_P
 *  @param TestGroupMatmulQuant parameterized test class
 *  @param INT8_SYM_QUANT_PER_GROUP_BF16 user-defined name of test
 *  @brief Test to validate group_matmul_direct symmetric INT8 quantization
 *         with per-group scaling and BF16 output. K is rounded to a multiple
 *         of 4 and group_size is randomly selected from valid divisors.
 *         Both source and weight are symmetrically quantized to S8.
 */
TEST_P(TestGroupMatmulQuant, INT8_SYM_QUANT_PER_GROUP_BF16) {
  uint64_t sym_k = (k / 4) * 4;
  if (sym_k == 0) {
    sym_k = 4;
  }

  std::vector<uint64_t> valid_gs;
  for (uint64_t gs = 4; gs <= sym_k; gs *= 2) {
    if (sym_k % gs == 0) {
      valid_gs.push_back(gs);
    }
  }
  if (valid_gs.empty()) {
    GTEST_SKIP() << "No valid group_size for K=" << sym_k;
  }
  std::mt19937 local_rng(m ^ k ^ n ^ 0xBF16);
  uint64_t group_size = valid_gs[local_rng() % valid_gs.size()];

  data_type_t ref_dt = data_type_t::bf16;
  data_type_t scale_dt = (local_rng() % 2 == 0) ? data_type_t::f32 :
                         data_type_t::bf16;
  uint64_t ngroups = sym_k / group_size;
  std::vector<int64_t> wei_sd = {static_cast<int64_t>(ngroups), static_cast<int64_t>(n)};
  std::vector<int64_t> src_sd = {static_cast<int64_t>(m), static_cast<int64_t>(ngroups)};

  std::vector<tensor_t> inp(num_ops), wt(num_ops), bias(num_ops),
      out(num_ops), out_ref(num_ops);

  for (size_t i = 0; i < num_ops; ++i) {
    auto wei_ref = tensor_factory.uniform_dist_tensor({sym_k, n}, ref_dt, 25.0,
                   transB);
    tensor_t weight_tensor, wei_scale, wei_zp;
    if (quant_params_compute(tensor_factory, wei_ref, ref_dt, data_type_t::s8,
                             wei_sd, scale_dt, wei_scale, wei_zp, &weight_tensor) != status_t::success) {
      FAIL() << "weight dynamic quantization failed";
    }
    auto src_ref = tensor_factory.uniform_dist_tensor({m, sym_k}, ref_dt, 25.0,
                   transA);
    tensor_t input_tensor, src_scale, src_zp;
    if (quant_params_compute(tensor_factory, src_ref, ref_dt, data_type_t::s8,
                             src_sd, scale_dt, src_scale, src_zp, &input_tensor) != status_t::success) {
      FAIL() << "source dynamic quantization failed";
    }

    inp[i]     = input_tensor;
    wt[i]      = weight_tensor;
    bias[i]    = tensor_factory.uniform_dist_tensor({1, n},
                 rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
    out[i]     = tensor_factory.uniform_dist_tensor({m, n}, data_type_t::bf16, 2.0);
    out_ref[i] = tensor_factory.uniform_dist_tensor({m, n}, data_type_t::bf16, 2.0);
  }

  // Random gated activation paired with the symmetric INT8 + per-group
  // GEMM (dst dtype is bf16, satisfying gated_act's f32/bf16 requirement).
  std::mt19937 act_rng((m << 1) ^ (k << 7) ^ (n << 13) ^ 0xACBEu);
  grp_matmul_gated_act_t act_type = moe_test_utils::pick_random_gated_act(
                                      static_cast<uint64_t>(n), act_rng);
  grp_matmul_gated_act_params act_params{};
  act_params.act = act_type;
  const grp_matmul_gated_act_params *act_ptr =
    (act_type != grp_matmul_gated_act_t::none) ? &act_params : nullptr;

  status_t status = group_matmul_kernel_test(inp, wt, bias, out, algo, 1.0f,
                    0.0f, /*moe_postop=*/nullptr, act_ptr);

  std::vector<post_op_type_t> ref_po;
  status_t ref_status = status_t::success;
  for (size_t i = 0; i < num_ops && ref_status == status_t::success; ++i) {
    std::vector<tensor_t> bin;
    ref_status = matmul_forced_ref_kernel_test(inp[i], wt[i], bias[i],
                 out_ref[i], ref_po, bin, true, algo, 1.0, 0.0);
  }

  bool ok = (status == status_t::success && ref_status == status_t::success);
  if (ok) {
    if (act_type != grp_matmul_gated_act_t::none) {
      for (size_t i = 0; i < num_ops; ++i) {
        moe_test_utils::apply_ref_gated_act_tensor(
          out_ref[i], static_cast<int>(m), static_cast<int>(n),
          static_cast<int>(n), act_type);
      }
      for (size_t i = 0; i < num_ops && ok; ++i)
        moe_test_utils::compare_activated_2D(out[i], out_ref[i], m, n / 2,
                                             sym_k, 1.0f, epsilon_bf16, ok);
    }
    else {
      for (size_t i = 0; i < num_ops && ok; ++i)
        compare_tensor_2D_matrix(out[i], out_ref[i], m, n, sym_k,
                                 rtol_bf16, epsilon_bf16, ok, false, 1.0f);
    }
  }
  EXPECT_TRUE(ok);
}

/** @fn TEST_P
 *  @param TestGroupMatmulQuant parameterized test class
 *  @param INT8_SYM_QUANT_PER_GROUP_F32 user-defined name of test
 *  @brief Test to validate group_matmul_direct symmetric INT8 quantization
 *         with per-group scaling and F32 output. Same quantization setup as
 *         INT8_SYM_QUANT_PER_GROUP_BF16 but accumulates into FP32.
 */
TEST_P(TestGroupMatmulQuant, INT8_SYM_QUANT_PER_GROUP_F32) {
  uint64_t sym_k = (k / 4) * 4;
  if (sym_k == 0) {
    sym_k = 4;
  }

  std::vector<uint64_t> valid_gs;
  for (uint64_t gs = 4; gs <= sym_k; gs *= 2) {
    if (sym_k % gs == 0) {
      valid_gs.push_back(gs);
    }
  }
  if (valid_gs.empty()) {
    GTEST_SKIP() << "No valid group_size for K=" << sym_k;
  }
  std::mt19937 local_rng(m ^ k ^ n ^ 0xF320);
  uint64_t group_size = valid_gs[local_rng() % valid_gs.size()];

  data_type_t ref_dt = data_type_t::f32;
  data_type_t scale_dt = (local_rng() % 2 == 0) ? data_type_t::f32 :
                         data_type_t::bf16;
  uint64_t ngroups = sym_k / group_size;
  std::vector<int64_t> wei_sd = {static_cast<int64_t>(ngroups), static_cast<int64_t>(n)};
  std::vector<int64_t> src_sd = {static_cast<int64_t>(m), static_cast<int64_t>(ngroups)};

  std::vector<tensor_t> inp(num_ops), wt(num_ops), bias(num_ops),
      out(num_ops), out_ref(num_ops);

  for (size_t i = 0; i < num_ops; ++i) {
    auto wei_ref = tensor_factory.uniform_dist_tensor({sym_k, n}, ref_dt, 2.0,
                   transB);
    tensor_t weight_tensor, wei_scale, wei_zp;
    if (quant_params_compute(tensor_factory, wei_ref, ref_dt, data_type_t::s8,
                             wei_sd, scale_dt, wei_scale, wei_zp, &weight_tensor) != status_t::success) {
      FAIL() << "weight dynamic quantization failed";
    }
    auto src_ref = tensor_factory.uniform_dist_tensor({m, sym_k}, ref_dt, 25.0,
                   transA);
    tensor_t input_tensor, src_scale, src_zp;
    if (quant_params_compute(tensor_factory, src_ref, ref_dt, data_type_t::s8,
                             src_sd, scale_dt, src_scale, src_zp, &input_tensor) != status_t::success) {
      FAIL() << "source dynamic quantization failed";
    }

    inp[i]     = input_tensor;
    wt[i]      = weight_tensor;
    bias[i]    = tensor_factory.uniform_dist_tensor({1, n},
                 rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
    out[i]     = tensor_factory.uniform_dist_tensor({m, n}, data_type_t::f32, 2.0);
    out_ref[i] = tensor_factory.uniform_dist_tensor({m, n}, data_type_t::f32, 2.0);
  }

  // Random gated activation paired with the symmetric INT8 + per-group
  // GEMM (dst dtype is f32, satisfying gated_act's f32/bf16 requirement).
  std::mt19937 act_rng((m << 1) ^ (k << 7) ^ (n << 13) ^ 0xACFEu);
  grp_matmul_gated_act_t act_type = moe_test_utils::pick_random_gated_act(
                                      static_cast<uint64_t>(n), act_rng);
  grp_matmul_gated_act_params act_params{};
  act_params.act = act_type;
  const grp_matmul_gated_act_params *act_ptr =
    (act_type != grp_matmul_gated_act_t::none) ? &act_params : nullptr;

  status_t status = group_matmul_kernel_test(inp, wt, bias, out, algo, 1.0f,
                    0.0f, /*moe_postop=*/nullptr, act_ptr);

  std::vector<post_op_type_t> ref_po;
  status_t ref_status = status_t::success;
  for (size_t i = 0; i < num_ops && ref_status == status_t::success; ++i) {
    std::vector<tensor_t> bin;
    ref_status = matmul_forced_ref_kernel_test(inp[i], wt[i], bias[i],
                 out_ref[i], ref_po, bin, true, algo, 1.0, 0.0);
  }

  bool ok = (status == status_t::success && ref_status == status_t::success);
  if (ok) {
    if (act_type != grp_matmul_gated_act_t::none) {
      for (size_t i = 0; i < num_ops; ++i) {
        moe_test_utils::apply_ref_gated_act_tensor(
          out_ref[i], static_cast<int>(m), static_cast<int>(n),
          static_cast<int>(n), act_type);
      }
      for (size_t i = 0; i < num_ops && ok; ++i)
        moe_test_utils::compare_activated_2D(out[i], out_ref[i], m, n / 2,
                                             sym_k, 1.0f, epsilon_f32, ok);
    }
    else {
      for (size_t i = 0; i < num_ops && ok; ++i)
        compare_tensor_2D_matrix(out[i], out_ref[i], m, n, sym_k,
                                 rtol_f32, epsilon_f32, ok, false, 1.0f);
    }
  }
  EXPECT_TRUE(ok);
}

/** @fn TEST_P
 *  @param TestGroupMatmulQuant parameterized test class
 *  @param INT8_SYM_QUANT_PER_TOKEN_BF16 user-defined name of test
 *  @brief Test to validate group_matmul_direct symmetric INT8 quantization
 *         with per-token source scaling and per-channel weight scaling,
 *         producing BF16 output. Exercises the token-wise quantization path
 *         used in inference-time activation quantization.
 */
TEST_P(TestGroupMatmulQuant, INT8_SYM_QUANT_PER_TOKEN_BF16) {
  uint64_t sym_k = (k / 4) * 4;
  if (sym_k == 0) {
    sym_k = 4;
  }

  data_type_t ref_dt = data_type_t::bf16;
  std::mt19937 local_rng(m ^ k ^ n ^ 0xBF17);
  data_type_t scale_dt = (local_rng() % 2 == 0) ? data_type_t::f32 :
                         data_type_t::bf16;
  std::vector<int64_t> wei_sd = {1, static_cast<int64_t>(n)};
  std::vector<int64_t> src_sd = {static_cast<int64_t>(m), 1};

  std::vector<tensor_t> inp(num_ops), wt(num_ops), bias(num_ops),
      out(num_ops), out_ref(num_ops);

  for (size_t i = 0; i < num_ops; ++i) {
    auto wei_ref = tensor_factory.uniform_dist_tensor({sym_k, n}, ref_dt, 25.0,
                   transB);
    tensor_t weight_tensor, wei_scale, wei_zp;
    if (quant_params_compute(tensor_factory, wei_ref, ref_dt, data_type_t::s8,
                             wei_sd, scale_dt, wei_scale, wei_zp, &weight_tensor) != status_t::success) {
      FAIL() << "weight dynamic quantization failed";
    }
    auto src_ref = tensor_factory.uniform_dist_tensor({m, sym_k}, ref_dt, 25.0,
                   transA);
    tensor_t input_tensor, src_scale, src_zp;
    if (quant_params_compute(tensor_factory, src_ref, ref_dt, data_type_t::s8,
                             src_sd, scale_dt, src_scale, src_zp, &input_tensor) != status_t::success) {
      FAIL() << "source dynamic quantization failed";
    }

    inp[i]     = input_tensor;
    wt[i]      = weight_tensor;
    bias[i]    = tensor_factory.uniform_dist_tensor({1, n},
                 rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
    out[i]     = tensor_factory.uniform_dist_tensor({m, n}, data_type_t::bf16, 2.0);
    out_ref[i] = tensor_factory.uniform_dist_tensor({m, n}, data_type_t::bf16, 2.0);
  }

  // Random gated activation paired with the symmetric INT8 + per-token
  // GEMM (dst dtype is bf16).
  std::mt19937 act_rng((m << 1) ^ (k << 7) ^ (n << 13) ^ 0xACBDu);
  grp_matmul_gated_act_t act_type = moe_test_utils::pick_random_gated_act(
                                      static_cast<uint64_t>(n), act_rng);
  grp_matmul_gated_act_params act_params{};
  act_params.act = act_type;
  const grp_matmul_gated_act_params *act_ptr =
    (act_type != grp_matmul_gated_act_t::none) ? &act_params : nullptr;

  status_t status = group_matmul_kernel_test(inp, wt, bias, out, algo, 1.0f,
                    0.0f, /*moe_postop=*/nullptr, act_ptr);

  std::vector<post_op_type_t> ref_po;
  status_t ref_status = status_t::success;
  for (size_t i = 0; i < num_ops && ref_status == status_t::success; ++i) {
    std::vector<tensor_t> bin;
    ref_status = matmul_forced_ref_kernel_test(inp[i], wt[i], bias[i],
                 out_ref[i], ref_po, bin, true, algo, 1.0, 0.0);
  }

  bool ok = (status == status_t::success && ref_status == status_t::success);
  if (ok) {
    if (act_type != grp_matmul_gated_act_t::none) {
      for (size_t i = 0; i < num_ops; ++i) {
        moe_test_utils::apply_ref_gated_act_tensor(
          out_ref[i], static_cast<int>(m), static_cast<int>(n),
          static_cast<int>(n), act_type);
      }
      for (size_t i = 0; i < num_ops && ok; ++i)
        moe_test_utils::compare_activated_2D(out[i], out_ref[i], m, n / 2,
                                             sym_k, 1.0f, epsilon_bf16, ok);
    }
    else {
      for (size_t i = 0; i < num_ops && ok; ++i)
        compare_tensor_2D_matrix(out[i], out_ref[i], m, n, sym_k,
                                 rtol_bf16, epsilon_bf16, ok, false, 1.0f, true);
    }
  }
  EXPECT_TRUE(ok);
}

/** @fn TEST_P
 *  @param TestGroupMatmulQuant parameterized test class
 *  @param INT8_SYM_QUANT_PER_TOKEN_F32 user-defined name of test
 *  @brief Test to validate group_matmul_direct symmetric INT8 quantization
 *         with per-token source scaling and per-channel weight scaling,
 *         producing F32 output. Same quantization setup as
 *         INT8_SYM_QUANT_PER_TOKEN_BF16 but accumulates into FP32.
 */
TEST_P(TestGroupMatmulQuant, INT8_SYM_QUANT_PER_TOKEN_F32) {
  uint64_t sym_k = (k / 4) * 4;
  if (sym_k == 0) {
    sym_k = 4;
  }

  data_type_t ref_dt = data_type_t::f32;
  std::mt19937 local_rng(m ^ k ^ n ^ 0xF321);
  data_type_t scale_dt = (local_rng() % 2 == 0) ? data_type_t::f32 :
                         data_type_t::bf16;
  std::vector<int64_t> wei_sd = {1, static_cast<int64_t>(n)};
  std::vector<int64_t> src_sd = {static_cast<int64_t>(m), 1};

  std::vector<tensor_t> inp(num_ops), wt(num_ops), bias(num_ops),
      out(num_ops), out_ref(num_ops);

  for (size_t i = 0; i < num_ops; ++i) {
    auto wei_ref = tensor_factory.uniform_dist_tensor({sym_k, n}, ref_dt, 25.0,
                   transB);
    tensor_t weight_tensor, wei_scale, wei_zp;
    if (quant_params_compute(tensor_factory, wei_ref, ref_dt, data_type_t::s8,
                             wei_sd, scale_dt, wei_scale, wei_zp, &weight_tensor) != status_t::success) {
      FAIL() << "weight dynamic quantization failed";
    }
    auto src_ref = tensor_factory.uniform_dist_tensor({m, sym_k}, ref_dt, 25.0,
                   transA);
    tensor_t input_tensor, src_scale, src_zp;
    if (quant_params_compute(tensor_factory, src_ref, ref_dt, data_type_t::s8,
                             src_sd, scale_dt, src_scale, src_zp, &input_tensor) != status_t::success) {
      FAIL() << "source dynamic quantization failed";
    }

    inp[i]     = input_tensor;
    wt[i]      = weight_tensor;
    bias[i]    = tensor_factory.uniform_dist_tensor({1, n},
                 rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
    out[i]     = tensor_factory.uniform_dist_tensor({m, n}, data_type_t::f32, 2.0);
    out_ref[i] = tensor_factory.uniform_dist_tensor({m, n}, data_type_t::f32, 2.0);
  }

  // Random gated activation paired with the symmetric INT8 + per-token
  // GEMM (dst dtype is f32).
  std::mt19937 act_rng((m << 1) ^ (k << 7) ^ (n << 13) ^ 0xACFDu);
  grp_matmul_gated_act_t act_type = moe_test_utils::pick_random_gated_act(
                                      static_cast<uint64_t>(n), act_rng);
  grp_matmul_gated_act_params act_params{};
  act_params.act = act_type;
  const grp_matmul_gated_act_params *act_ptr =
    (act_type != grp_matmul_gated_act_t::none) ? &act_params : nullptr;

  status_t status = group_matmul_kernel_test(inp, wt, bias, out, algo, 1.0f,
                    0.0f, /*moe_postop=*/nullptr, act_ptr);

  std::vector<post_op_type_t> ref_po;
  status_t ref_status = status_t::success;
  for (size_t i = 0; i < num_ops && ref_status == status_t::success; ++i) {
    std::vector<tensor_t> bin;
    ref_status = matmul_forced_ref_kernel_test(inp[i], wt[i], bias[i],
                 out_ref[i], ref_po, bin, true, algo, 1.0, 0.0);
  }

  bool ok = (status == status_t::success && ref_status == status_t::success);
  if (ok) {
    if (act_type != grp_matmul_gated_act_t::none) {
      for (size_t i = 0; i < num_ops; ++i) {
        moe_test_utils::apply_ref_gated_act_tensor(
          out_ref[i], static_cast<int>(m), static_cast<int>(n),
          static_cast<int>(n), act_type);
      }
      for (size_t i = 0; i < num_ops && ok; ++i)
        moe_test_utils::compare_activated_2D(out[i], out_ref[i], m, n / 2,
                                             sym_k, 1.0f, epsilon_f32, ok);
    }
    else {
      for (size_t i = 0; i < num_ops && ok; ++i)
        compare_tensor_2D_matrix(out[i], out_ref[i], m, n, sym_k,
                                 rtol_f32, epsilon_f32, ok, false, 1.0f, true);
    }
  }
  EXPECT_TRUE(ok);
}

/** @fn TEST_P
 *  @param TestGroupMatmulQuant parameterized test class
 *  @param INT8_DYNAMIC_GEMM_BF16 user-defined name of test
 *  @brief Test to validate group_matmul_direct dynamic quantization GEMM path
 *         with BF16 activation, S8 weights, and BF16 output. The source
 *         tensor carries pre-allocated scale buffers for runtime quantization.
 *         Randomly selects per-token or per-group source scaling.
 */
TEST_P(TestGroupMatmulQuant, INT8_DYNAMIC_GEMM_BF16) {
  uint64_t sym_k = (k / 4) * 4;
  if (sym_k == 0) {
    sym_k = 4;
  }

  data_type_t test_dt = data_type_t::bf16;
  std::mt19937 local_rng(m ^ k ^ n ^ 0xBF18);
  bool use_per_group = (local_rng() % 2 == 0);
  uint64_t group_size = 0, ngroups = 0;

  if (use_per_group) {
    std::vector<uint64_t> valid_gs;
    for (uint64_t gs = 4; gs <= sym_k; gs *= 2) {
      if (sym_k % gs == 0) {
        valid_gs.push_back(gs);
      }
    }
    if (valid_gs.empty()) {
      use_per_group = false;
    }
    else {
      group_size = valid_gs[local_rng() % valid_gs.size()];
      ngroups = sym_k / group_size;
    }
  }

  data_type_t scale_dt = (local_rng() % 2 == 0) ? data_type_t::f32 :
                         data_type_t::bf16;
  std::vector<int64_t> wei_scale_dims;
  std::vector<uint64_t> src_scale_shape;
  if (use_per_group) {
    wei_scale_dims = {static_cast<int64_t>(ngroups), static_cast<int64_t>(n)};
    src_scale_shape = {m, ngroups};
  }
  else {
    wei_scale_dims = {1, static_cast<int64_t>(n)};
    src_scale_shape = {m, 1};
  }

  std::vector<tensor_t> inp(num_ops), inp_ref(num_ops),
      wt_s8(num_ops), wt_ref(num_ops),
      bias(num_ops), out(num_ops), out_ref(num_ops);

  for (size_t i = 0; i < num_ops; ++i) {
    auto wt_ref_t = tensor_factory.uniform_dist_tensor({sym_k, n}, test_dt, 2.0);
    tensor_t wt_s8_t, wei_scale, wei_zp;
    if (quant_params_compute(tensor_factory, wt_ref_t, test_dt, data_type_t::s8,
                             wei_scale_dims, scale_dt, wei_scale, wei_zp, &wt_s8_t) != status_t::success) {
      FAIL() << "weight dynamic quantization failed";
    }
    auto src_scale = tensor_factory.zero_tensor(src_scale_shape, scale_dt);
    auto inp_t     = tensor_factory.uniform_dist_tensor({m, sym_k}, test_dt, 2.0,
                     transA, src_scale, tensor_t());
    auto inp_ref_t = tensor_factory.uniform_dist_tensor({m, sym_k}, test_dt, 2.0,
                     transA);

    inp[i]     = inp_t;
    inp_ref[i] = inp_ref_t;
    wt_s8[i]   = wt_s8_t;
    wt_ref[i]  = wt_ref_t;
    bias[i]    = tensor_factory.uniform_dist_tensor({1, n},
                 rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
    out[i]     = tensor_factory.uniform_dist_tensor({m, n}, test_dt, 2.0);
    out_ref[i] = tensor_factory.uniform_dist_tensor({m, n}, test_dt, 2.0);
  }

  // Random gated activation paired with the dynamic INT8 quant GEMM
  // (dst dtype is bf16).
  std::mt19937 act_rng((m << 1) ^ (k << 7) ^ (n << 13) ^ 0xADCBu);
  grp_matmul_gated_act_t act_type = moe_test_utils::pick_random_gated_act(
                                      static_cast<uint64_t>(n), act_rng);
  grp_matmul_gated_act_params act_params{};
  act_params.act = act_type;
  const grp_matmul_gated_act_params *act_ptr =
    (act_type != grp_matmul_gated_act_t::none) ? &act_params : nullptr;

  status_t status = group_matmul_kernel_test(inp, wt_s8, bias, out, algo, 1.0f,
                    0.0f, /*moe_postop=*/nullptr, act_ptr);

  std::vector<post_op_type_t> ref_po;
  status_t ref_status = status_t::success;
  for (size_t i = 0; i < num_ops && ref_status == status_t::success; ++i) {
    std::vector<tensor_t> bin;
    ref_status = matmul_forced_ref_kernel_test(inp_ref[i], wt_ref[i], bias[i],
                 out_ref[i], ref_po, bin, true, algo, 1.0, 0.0);
  }

  bool ok = (status == status_t::success && ref_status == status_t::success);
  if (ok) {
    if (act_type != grp_matmul_gated_act_t::none) {
      for (size_t i = 0; i < num_ops; ++i) {
        moe_test_utils::apply_ref_gated_act_tensor(
          out_ref[i], static_cast<int>(m), static_cast<int>(n),
          static_cast<int>(n), act_type);
      }
      for (size_t i = 0; i < num_ops && ok; ++i)
        moe_test_utils::compare_activated_2D(out[i], out_ref[i], m, n / 2,
                                             sym_k, 1.0f,
                                             16 * epsilon_bf16, ok);
    }
    else {
      for (size_t i = 0; i < num_ops && ok; ++i)
        compare_tensor_2D_matrix(out[i], out_ref[i], m, n, sym_k,
                                 rtol_bf16, 16 * epsilon_bf16, ok, false, 1.0f, true);
    }
  }
  EXPECT_TRUE(ok);
}

/** @fn TEST_P
 *  @param TestGroupMatmulQuant parameterized test class
 *  @param INT8_DYNAMIC_GEMM_F32 user-defined name of test
 *  @brief Test to validate group_matmul_direct dynamic quantization GEMM path
 *         with F32 activation, S8 weights, and F32 output. Same dynamic
 *         quantization setup as INT8_DYNAMIC_GEMM_BF16 but operates entirely
 *         in FP32 precision.
 */
TEST_P(TestGroupMatmulQuant, INT8_DYNAMIC_GEMM_F32) {
  uint64_t sym_k = (k / 4) * 4;
  if (sym_k == 0) {
    sym_k = 4;
  }

  data_type_t test_dt = data_type_t::f32;
  std::mt19937 local_rng(m ^ k ^ n ^ 0xF322);
  bool use_per_group = (local_rng() % 2 == 0);
  uint64_t group_size = 0, ngroups = 0;

  if (use_per_group) {
    std::vector<uint64_t> valid_gs;
    for (uint64_t gs = 4; gs <= sym_k; gs *= 2) {
      if (sym_k % gs == 0) {
        valid_gs.push_back(gs);
      }
    }
    if (valid_gs.empty()) {
      use_per_group = false;
    }
    else {
      group_size = valid_gs[local_rng() % valid_gs.size()];
      ngroups = sym_k / group_size;
    }
  }

  data_type_t scale_dt = (local_rng() % 2 == 0) ? data_type_t::f32 :
                         data_type_t::bf16;
  std::vector<int64_t> wei_scale_dims;
  std::vector<uint64_t> src_scale_shape;
  if (use_per_group) {
    wei_scale_dims = {static_cast<int64_t>(ngroups), static_cast<int64_t>(n)};
    src_scale_shape = {m, ngroups};
  }
  else {
    wei_scale_dims = {1, static_cast<int64_t>(n)};
    src_scale_shape = {m, 1};
  }

  std::vector<tensor_t> inp(num_ops), inp_ref(num_ops),
      wt_s8(num_ops), wt_ref(num_ops),
      bias(num_ops), out(num_ops), out_ref(num_ops);

  for (size_t i = 0; i < num_ops; ++i) {
    auto wt_ref_t = tensor_factory.uniform_dist_tensor({sym_k, n}, test_dt, 2.0);
    tensor_t wt_s8_t, wei_scale, wei_zp;
    if (quant_params_compute(tensor_factory, wt_ref_t, test_dt, data_type_t::s8,
                             wei_scale_dims, scale_dt, wei_scale, wei_zp, &wt_s8_t) != status_t::success) {
      FAIL() << "weight dynamic quantization failed";
    }
    auto src_scale = tensor_factory.zero_tensor(src_scale_shape, scale_dt);
    auto inp_t     = tensor_factory.uniform_dist_tensor({m, sym_k}, test_dt, 2.0,
                     transA, src_scale, tensor_t());
    auto inp_ref_t = tensor_factory.uniform_dist_tensor({m, sym_k}, test_dt, 2.0,
                     transA);

    inp[i]     = inp_t;
    inp_ref[i] = inp_ref_t;
    wt_s8[i]   = wt_s8_t;
    wt_ref[i]  = wt_ref_t;
    bias[i]    = tensor_factory.uniform_dist_tensor({1, n},
                 rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
    out[i]     = tensor_factory.uniform_dist_tensor({m, n}, test_dt, 2.0);
    out_ref[i] = tensor_factory.uniform_dist_tensor({m, n}, test_dt, 2.0);
  }

  // Random gated activation paired with the dynamic INT8 quant GEMM
  // (dst dtype is f32).
  std::mt19937 act_rng((m << 1) ^ (k << 7) ^ (n << 13) ^ 0xADCFu);
  grp_matmul_gated_act_t act_type = moe_test_utils::pick_random_gated_act(
                                      static_cast<uint64_t>(n), act_rng);
  grp_matmul_gated_act_params act_params{};
  act_params.act = act_type;
  const grp_matmul_gated_act_params *act_ptr =
    (act_type != grp_matmul_gated_act_t::none) ? &act_params : nullptr;

  status_t status = group_matmul_kernel_test(inp, wt_s8, bias, out, algo, 1.0f,
                    0.0f, /*moe_postop=*/nullptr, act_ptr);

  std::vector<post_op_type_t> ref_po;
  status_t ref_status = status_t::success;
  for (size_t i = 0; i < num_ops && ref_status == status_t::success; ++i) {
    std::vector<tensor_t> bin;
    ref_status = matmul_forced_ref_kernel_test(inp_ref[i], wt_ref[i], bias[i],
                 out_ref[i], ref_po, bin, true, algo, 1.0, 0.0);
  }

  bool ok = (status == status_t::success && ref_status == status_t::success);
  if (ok) {
    if (act_type != grp_matmul_gated_act_t::none) {
      for (size_t i = 0; i < num_ops; ++i) {
        moe_test_utils::apply_ref_gated_act_tensor(
          out_ref[i], static_cast<int>(m), static_cast<int>(n),
          static_cast<int>(n), act_type);
      }
      for (size_t i = 0; i < num_ops && ok; ++i)
        moe_test_utils::compare_activated_2D(out[i], out_ref[i], m, n / 2,
                                             sym_k, 1.0f,
                                             18 * epsilon_bf16, ok);
    }
    else {
      for (size_t i = 0; i < num_ops && ok; ++i)
        compare_tensor_2D_matrix(out[i], out_ref[i], m, n, sym_k,
                                 rtol_bf16, 18 * epsilon_bf16, ok, false, 1.0f, true);
    }
  }
  EXPECT_TRUE(ok);

// TODO: Add per-ALGO coverage tests for ZENDNNL_GRP_MATMUL_ALGO=1..5.
// get_grp_matmul_algo() reads the env var on every call (no caching),
// so setenv/putenv can switch algos within a single test process.
// Exercise all dispatch paths (1=sequential, 2=flat_ccd_m_tile,
// 3=flat_ccd_n_tile, 4=multilevel, 5=per_expert). Add in a follow-up PR.
}

INSTANTIATE_TEST_SUITE_P(GroupMatmulQuant, TestGroupMatmulQuant,
                         ::testing::ValuesIn(quant_matmul_test));

// Note:
//   * TestGroupMatmul (basic F32/BF16 suite) is instantiated near the top
//     of the file (search for `INSTANTIATE_TEST_SUITE_P(GroupMatmul,
//     TestGroupMatmul, ...)`).
//   * TestGroupMatmulQuant (WOQ / INT8 / dynamic-quant suite) is
//     instantiated immediately above this comment with the constrained
//     `quant_matmul_test` parameter list — see GroupQuantMatmulType in
//     gtest_utils.hpp for the rationale.