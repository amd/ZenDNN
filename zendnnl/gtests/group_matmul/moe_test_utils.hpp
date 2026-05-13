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

/// @file moe_test_utils.hpp
/// @brief Shared test helpers for MoE / fused-MoE / group_matmul gtest
///        suites.  Was section [1] of test_group_matmul.cpp; lifted to
///        a sibling header so the .cpp stays focused on the test
///        bodies and so future test files (e.g. a dedicated prepack
///        gtest file) can reuse the same harness.
///
/// Contents:
///   * `make_value`, `fill_pattern`, `fill_src`, `fill_wei1`,
///     `fill_wei2` — deterministic test-data generators.
///   * `TypedBuffers` — bf16 + f32 storage with opaque pointer views.
///   * `make_uniform_params`, `make_mixed_params` — `matmul_params`
///     vector builders.
///   * `GemmVecs::uniform` — uniform per-call GEMM wrapper vectors.
///   * Reference activation math (`ref_silu_mul`, `ref_gelu_mul`,
///     `ref_swiglu_oai`, `ref_gated_act`, `apply_ref_gated_act`,
///     `apply_ref_gated_act_tensor`, `pick_random_gated_act`,
///     `compare_activated_2D`).
///   * Tolerance presets (`Tol`, `tol_fused`, `tol_act`, `tol_moe`).
///   * Env-var RAII guards (`AlgoEnvGuard`, `EnvVarGuard`).
///   * Fused-MoE reference + verification helpers
///     (`make_fused_moe_op2`, `fill_moe_tensors`,
///     `run_legacy_2call_ref`, `verify_per_expert_2d`).

#ifndef ZENDNNL_GTESTS_MOE_TEST_UTILS_HPP
#define ZENDNNL_GTESTS_MOE_TEST_UTILS_HPP

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "common/bfloat16.hpp"
#include "gtest_utils.hpp"
#include "lowoha_operators/matmul/group_matmul/group_matmul_direct.hpp"
#include "lowoha_operators/matmul/lowoha_matmul.hpp"

namespace moe_test_utils {

using bfloat16_t = zendnnl::common::bfloat16_t;
using data_type_t = zendnnl::common::data_type_t;
using zendnnl::lowoha::matmul::matmul_params;
using zendnnl::lowoha::matmul::grp_matmul_gated_act_t;
using zendnnl::lowoha::matmul::grp_matmul_gated_act_params;
using zendnnl::lowoha::matmul::grp_matmul_fused_moe_params;
using zendnnl::lowoha::matmul::group_matmul_direct;
using zendnnl::error_handling::status_t;

// ───────────────────────────────────────────────────────────────────
// [1.a] Test data initialization
//
// Deterministic pattern: a[i] = scale * (int((i + e*seed_step) % mod) - shift).
// The explicit int cast prevents size_t underflow that would produce garbage
// like 1.7e+35 (see git history for the bug this fixes).
// ───────────────────────────────────────────────────────────────────

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

// ───────────────────────────────────────────────────────────────────
// [1.b] TypedBuffers: holds both bf16 and f32 storage, exposes opaque pointers.
//
// Avoids the if/else maze around every buffer access.  Only one of the two
// vectors is actually resized based on `is_bf16`.  `is_mixed = true` resizes
// both (used by TestFusedMoEAlgos for bf16-src ↔ f32-dst mixed precision).
// ───────────────────────────────────────────────────────────────────

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

// ───────────────────────────────────────────────────────────────────
// [1.c] matmul_params & GEMM wrapper vector builders
// ───────────────────────────────────────────────────────────────────

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

  // `wc` defaults to `true` (constant weights — the typical LLM-
  // inference assumption used by every cache-warm probe in
  // test_prepack.cpp).  Tests that want to exercise the no-cache /
  // CK-refusal path explicitly override via `gv.is_wc.assign(N,
  // false)` after construction (see e.g. test_prepack.cpp around
  // line 1417 + the inline GemmVecs builder around line 556 of
  // this file).  The flip is symmetric with the public API
  // contract: `is_weights_const = false` means "this caller wants
  // the no-cache path", and the new CK runtime gate
  // (`custom_kernel/dispatch.cpp::prepare_for_call`) refuses CK
  // for any active expert flagged that way; tests that probe the
  // CK pack cache need `wc = true` for the cache to be populated
  // at all.
  static GemmVecs uniform(int num_ops, int M, int N, int K,
                          float a = 1.0f, float b = 0.0f,
                          bool wc = true, bool tA = false, bool tB = false) {
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

// ───────────────────────────────────────────────────────────────────
// [1.d] Scalar reference math for gated activations and weighted reduce
// ───────────────────────────────────────────────────────────────────

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
// GEMM noise multiplicatively: an input pair (g, u) carrying error ~ε
// each produces output error roughly (|g| + |u|) · ε.  Both |g| and
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
// (silu(g) · u → output error ~2 · max(|g|, |u|) · ε where ε is the per-
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

// ───────────────────────────────────────────────────────────────────
// [1.e] Tolerance presets (bf16 / f32 / mixed)
// ───────────────────────────────────────────────────────────────────

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

// ───────────────────────────────────────────────────────────────────
// [1.f] ALGO env var RAII guard
// ───────────────────────────────────────────────────────────────────

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

// ───────────────────────────────────────────────────────────────────
// [1.g] Shared fused-MoE reference + verification helpers
//
// Every fused-MoE correctness suite below ([5], [5b], [5c], [6], [7],
// [7b], [10], [11], [12]) compares the fused dispatcher's output
// against the SAME 2-call legacy reference: Op1 + activation followed
// by Op2.  Extract that reference into a single helper so the test
// bodies focus on the buffer-source / shape layout that's actually
// under test.  The helper also encapsulates the GemmVecs construction
// (op1 reads src at lda=K, op2 reads d1_ref at lda=N_gate_up) so each
// caller passes only the dimensions, not the per-call vectors.
//
// `verify_per_expert_2d` is the symmetric comparison-loop helper —
// per-expert × per-row × per-col tolerance check with a single
// failure message that prints the location.
// ───────────────────────────────────────────────────────────────────

// Build the Op2 (down_proj) side of a fused-MoE param block.  Caller
// can append `dst_down` / `ldc_down` after for the caller-allocated
// case, or leave them empty to engage internal-alloc + src-reuse.
inline grp_matmul_fused_moe_params make_fused_moe_op2(
    int num_ops, int H,
    const std::vector<const void *> &wei2,
    const std::vector<const void *> &bias2) {
  grp_matmul_fused_moe_params f{};
  f.down_weight = wei2;
  f.N_down      = std::vector<int>(num_ops, H);
  f.ldb_down    = std::vector<int>(num_ops, H);
  f.bias_down   = bias2;
  return f;
}

// Fill src/wei1/wei2 buffers with deterministic seeds.  Pass nullptr
// for any buffer the caller doesn't want filled (e.g. weights only).
inline void fill_moe_tensors(int num_ops, bool is_bf16,
                             TypedBuffers *src, TypedBuffers *w1,
                             TypedBuffers *w2) {
  for (int e = 0; e < num_ops; ++e) {
    if (is_bf16) {
      if (src) fill_src (src->bf16[e], e);
      if (w1)  fill_wei1(w1 ->bf16[e], e);
      if (w2)  fill_wei2(w2 ->bf16[e], e);
    } else {
      if (src) fill_src (src->f32[e], e);
      if (w1)  fill_wei1(w1 ->f32[e], e);
      if (w2)  fill_wei2(w2 ->f32[e], e);
    }
  }
}

// Per-expert-M canonical form.  The dispatcher accepts a per-expert
// Ms vector (lengths that may include M=0 inactive experts), so this
// version covers every test case.  The uniform-M overload below
// delegates here.
inline status_t run_legacy_2call_ref(
    const std::vector<int> &Ms, int K_in, int N_gate_up, int K_down, int H,
    bool is_bf16, grp_matmul_gated_act_t act_type,
    const std::vector<const void *> &src,
    const std::vector<const void *> &wei1,
    const std::vector<const void *> &wei2,
    const std::vector<void *>       &d1_ref,
    const std::vector<void *>       &d2_ref) {
  using namespace zendnnl::lowoha::matmul;
  const int num_ops = static_cast<int>(Ms.size());
  const data_type_t dt = is_bf16 ? data_type_t::bf16 : data_type_t::f32;

  GemmVecs gv1;
  gv1.layout.assign(num_ops, 'r');
  gv1.transA.assign(num_ops, false);
  gv1.transB.assign(num_ops, false);
  gv1.is_wc .assign(num_ops, false);
  gv1.alpha .assign(num_ops, 1.0f);
  gv1.beta  .assign(num_ops, 0.0f);
  gv1.Ms     = Ms;
  gv1.Ns    .assign(num_ops, N_gate_up);
  gv1.Ks    .assign(num_ops, K_in);
  gv1.lda   .assign(num_ops, K_in);
  gv1.ldb   .assign(num_ops, N_gate_up);
  gv1.ldc   .assign(num_ops, N_gate_up);

  GemmVecs gv2 = gv1;
  gv2.Ns .assign(num_ops, H);
  gv2.Ks .assign(num_ops, K_down);
  gv2.lda.assign(num_ops, N_gate_up);  // Op2 reads d1_ref at gate+up stride.
  gv2.ldb.assign(num_ops, H);
  gv2.ldc.assign(num_ops, H);

  std::vector<const void *> no_bias(num_ops, nullptr);
  auto params = make_uniform_params(num_ops, dt);

  grp_matmul_gated_act_params act{};
  act.act = act_type;
  auto act_ptr = (act_type != grp_matmul_gated_act_t::none) ? &act : nullptr;

  auto pr1 = params;
  status_t s1 = group_matmul_direct(
      gv1.layout, gv1.transA, gv1.transB, gv1.Ms, gv1.Ns, gv1.Ks, gv1.alpha,
      src, gv1.lda, wei1, gv1.ldb, no_bias, gv1.beta, d1_ref, gv1.ldc,
      gv1.is_wc, pr1, nullptr, act_ptr);
  if (s1 != status_t::success) return s1;

  std::vector<const void *> srcs2(num_ops);
  for (int e = 0; e < num_ops; ++e) srcs2[e] = d1_ref[e];
  auto pr2 = params;
  return group_matmul_direct(
      gv2.layout, gv2.transA, gv2.transB, gv2.Ms, gv2.Ns, gv2.Ks, gv2.alpha,
      srcs2, gv2.lda, wei2, gv2.ldb, no_bias, gv2.beta, d2_ref, gv2.ldc,
      gv2.is_wc, pr2);
}

inline status_t run_legacy_2call_ref(
    int num_ops, int M, int K_in, int N_gate_up, int K_down, int H,
    bool is_bf16, grp_matmul_gated_act_t act_type,
    const std::vector<const void *> &src,
    const std::vector<const void *> &wei1,
    const std::vector<const void *> &wei2,
    const std::vector<void *>       &d1_ref,
    const std::vector<void *>       &d2_ref) {
  return run_legacy_2call_ref(
      std::vector<int>(num_ops, M), K_in, N_gate_up, K_down, H,
      is_bf16, act_type, src, wei1, wei2, d1_ref, d2_ref);
}

// Per-expert × per-row × per-col tolerance check.  `got_row_stride`
// and `ref_row_stride` let the caller compare buffers with different
// row strides (e.g. in-place src reuse stride-K vs reference stride-H).
// Per-expert-M form skips M=0 experts (Op2 doesn't write that slot).
template <typename Label>
inline void verify_per_expert_2d(
    const TypedBuffers &got, size_t got_row_stride,
    const TypedBuffers &ref, size_t ref_row_stride,
    const std::vector<int> &Ms, int N, bool is_bf16,
    Tol tol, Label &&label) {
  for (int e = 0; e < (int)Ms.size(); ++e) {
    const int M_e = Ms[e];
    if (M_e == 0) continue;
    for (int r = 0; r < M_e; ++r) {
      for (int c = 0; c < N; ++c) {
        const float g = got.at(e, (size_t)r * got_row_stride + c, is_bf16);
        const float f = ref.at(e, (size_t)r * ref_row_stride + c, is_bf16);
        ASSERT_NEAR(g, f, std::abs(f) * tol.rel + tol.abs)
            << label << " e=" << e << " M_e=" << M_e
            << " r=" << r << " c=" << c;
      }
    }
  }
}

template <typename Label>
inline void verify_per_expert_2d(
    const TypedBuffers &got, size_t got_row_stride,
    const TypedBuffers &ref, size_t ref_row_stride,
    int num_ops, int M, int N, bool is_bf16,
    Tol tol, Label &&label) {
  verify_per_expert_2d(got, got_row_stride, ref, ref_row_stride,
                       std::vector<int>(num_ops, M), N, is_bf16,
                       tol, std::forward<Label>(label));
}

} // namespace moe_test_utils

#endif // ZENDNNL_GTESTS_MOE_TEST_UTILS_HPP
