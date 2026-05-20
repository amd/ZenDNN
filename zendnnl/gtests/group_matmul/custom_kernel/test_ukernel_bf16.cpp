/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

/// BF16 microkernel end-to-end correctness — exercises both the
/// `bf16:bf16:bf16` and `bf16:bf16:f32` variants against a scalar
/// FP32 reference computed alongside.  The test goes through
/// `group_matmul_direct` with `ZENDNNL_GRP_MATMUL_ALGO=3` and the
/// custom-kernel gate forced on, so the ALGO 3 dispatcher takes the
/// CK path (when supported by the host) and the per-tile invocation
/// actually fires.
///
/// Coverage axes (parameterised):
///   * Shape grid: (M, K, N) ∈ a curated set covering common MoE
///     decode shapes and CK pack-NR boundaries.
///   * Activation: {none, swiglu_oai_mul, silu_and_mul, gelu_and_mul}.
///     - `swiglu_oai_mul` — fused in the per-tile epilogue, halved
///       output width (interleaved `[g0,u0,g1,u1,...]` layout).
///     - `silu_and_mul` / `gelu_and_mul` — split-halves
///       `[gate_cols | up_cols]` layout; CK runs the matmul-only
///       path and `flat_n_tile`'s post-pass applies the activation
///       to dst[:, 0:N/2).  cols [N/2, N) become garbage by contract.
///     - `none` — plain matmul, full N-wide output.
///   * Bias dtype: {none, bf16, f32}.
///   * Dst dtype: {bf16, f32} — but `(swiglu, f32)` is structurally
///     invalid (kernel refuses), tested by negative-gate tests in
///     test_prepare_for_call.cpp; we filter it here.  silu/gelu
///     accept both dst dtypes (the matmul-only path doesn't constrain
///     dst beyond the variant gate).
///
/// Reference: a scalar FP32 GEMM computed inline (no library call).
/// For swiglu the reference reads gate / up from interleaved cols
/// (2n+0, 2n+1).  For silu/gelu the reference reads gate from cols
/// [0, N/2) and up from cols [N/2, N), matching the split-halves
/// public-API contract (`grp_matmul_gated_act_t` doc-block in
/// `group_matmul_direct.hpp`).
/// Tolerance: BF16-mantissa epsilon (`tol_act(/*is_bf16=*/true)`).

#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "ck_test_helpers.hpp"
#include "moe_test_utils.hpp"

// For the CK-engagement assertion: `group_matmul_direct` publishes
// the resolved `gemm_mode` static literal to a test-only atomic in
// `test_api::s_last_group_matmul_direct_gemm_mode` (declared in
// `parallel_common.hpp`).  Reading that after the call is the
// reliable signal that the custom kernel actually ran (the LRU pack
// cache alone is not — `flat_n_tile` warms it via
// `prepack_for_algo_3` BEFORE the plan is built, so cache hits do
// not prove the plan engaged the per-tile dispatch).
#include "lowoha_operators/matmul/group_matmul/group_matmul_parallel_common.hpp"

namespace {

namespace mt = moe_test_utils;
using mt::bfloat16_t;
using mt::data_type_t;
using mt::grp_matmul_gated_act_t;
using mt::group_matmul_direct;
using mt::status_t;

// Convert a BF16 buffer to FP32 for reference comparisons.
inline float to_f32(bfloat16_t v) { return static_cast<float>(v); }

// Pin the dispatcher's per-call thread team to a moderate value
// regardless of the CI host's `OMP_NUM_THREADS`.  This suite is
// meant to validate the BF16 microkernel — not stress-test the
// planner at extreme thread counts — so we pass `kCkTestThreads`
// through `params[i].num_threads`, which `group_matmul_direct`
// feeds into `resolve_num_threads(params[0].num_threads, omp_mt)`
// (group_matmul_direct.cpp:992).  When the params field is non-
// zero, it overrides `thread_guard::max_threads()` for this call
// — even though the latter is process-cached on first use and
// otherwise unrecoverable from a test.  At very high host thread
// counts (e.g. 256) the planner would otherwise correctly bypass
// CK on most shapes in the matrix (per-thread N-tile slack drops
// below `ntile_viable`'s engagement floor), which would false-
// fail the engagement counter and the strict gate.  Pinning
// sidesteps that and makes engagement deterministic across hosts.
//
// Pinning value (`kCkTestThreads = 4`) chosen empirically to:
//   * maximise CK coverage of the parameterised matrix (43.6% of
//     cases engage at this pin, vs ~6% at pin=8 — the smaller
//     team gives `ntile_viable` more per-thread slack on the
//     small-K shapes), and
//   * still spawn an OpenMP team most CI hosts can satisfy
//     (4 is well below typical `OMP_THREAD_LIMIT`).
//
// The dominant matrix at this pin is still NR=32 default cases
// — by design — but every NR=64-pinned shape (`nr64_shapes`)
// engages 100% of the time, ensuring the NV=4/NR=64 microkernels
// and the new BF16:BF16:F32 NR=64 store epilogue are actually
// exercised end-to-end.
constexpr int kCkTestThreads = 4;

// ──────────────────────────────────────────────────────────────────
// Scalar FP32 reference GEMM with optional bias + swiglu_oai_mul
// activation.  Mirrors the kernel's contract:
//   * src is BF16; weights are BF16; bias is BF16 or FP32 or absent.
//   * Output is BF16 or FP32 per `dst_is_f32`.
//   * Activation is applied AFTER the matmul + bias.  swiglu_oai_mul
//     halves N — output cols are N/2.
// Returns the per-row, per-col FP32 reference value at (m, n).  The
// test caller fills the result tensor element-by-element using this
// helper (slow but bulletproof — 200 cases × ~500 elements = ~100k
// scalar ops, negligible).
// ──────────────────────────────────────────────────────────────────
// Helper: scalar FP32 matmul element at (m, n) with optional bias.
// Reused by every act-class branch below.  Weight contract:
// transB=false, so wei is [K, N] row-major and `ldb = N` is the
// stride between K-rows.
inline float ref_matmul_elem(int m, int n,
                             int K,
                             const bfloat16_t *src, int lda,
                             const bfloat16_t *wei, int ldb,
                             const void *bias, data_type_t bias_dt) {
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    acc += to_f32(src[m * lda + k]) * to_f32(wei[k * ldb + n]);
  }
  if (bias_dt == data_type_t::bf16) {
    acc += to_f32(static_cast<const bfloat16_t *>(bias)[n]);
  } else if (bias_dt == data_type_t::f32) {
    acc += static_cast<const float *>(bias)[n];
  }
  return acc;
}

inline float ref_gemm_act(int m, int n,
                          int K, int N,
                          const bfloat16_t *src, int lda,
                          const bfloat16_t *wei, int ldb,
                          const void *bias, data_type_t bias_dt,
                          grp_matmul_gated_act_t act) {
  // ── act = none — plain matmul element (M, n) ──────────────────
  if (act == grp_matmul_gated_act_t::none) {
    return ref_matmul_elem(m, n, K, src, lda, wei, ldb, bias, bias_dt);
  }

  // ── act = swiglu_oai_mul (interleaved gate/up; halved output) ─
  // gate is at even col `2n+0`, up at odd col `2n+1`.
  // gate, up clamped to [-7, 7]; sig = sigmoid(gate * 1.702f);
  // out[m, n] = (1 + up) * (gate * sig).  N here is post-activation
  // (caller already halved N before calling).
  if (act == grp_matmul_gated_act_t::swiglu_oai_mul) {
    float acc_g = ref_matmul_elem(m, 2 * n + 0, K, src, lda, wei, ldb,
                                   bias, bias_dt);
    float acc_u = ref_matmul_elem(m, 2 * n + 1, K, src, lda, wei, ldb,
                                   bias, bias_dt);
    acc_g = std::max(-7.0f, std::min(7.0f, acc_g));
    acc_u = std::max(-7.0f, std::min(7.0f, acc_u));
    const float sig = 1.0f / (1.0f + std::exp(-acc_g * 1.702f));
    return (1.0f + acc_u) * (acc_g * sig);
  }

  // ── act = silu_and_mul / gelu_and_mul (split-halves) ──────────
  // gate is at col `n` in the first half; up is at col `n + N/2` in
  // the second half.  CK runs the matmul-only path; the activation
  // is applied by `flat_n_tile`'s post-pass.  N here is the
  // post-activation width (caller already halved).
  const int gate_col = n;
  const int up_col   = n + N;     // N here is the half-width = original_N / 2
  const float gate = ref_matmul_elem(m, gate_col, K, src, lda, wei, ldb,
                                      bias, bias_dt);
  const float up   = ref_matmul_elem(m, up_col,   K, src, lda, wei, ldb,
                                      bias, bias_dt);
  if (act == grp_matmul_gated_act_t::silu_and_mul) {
    const float sigmoid_g = 1.0f / (1.0f + std::exp(-gate));
    return (gate * sigmoid_g) * up;
  }
  // gelu_and_mul (erf form, matches `moe_test_utils::ref_gelu_mul`).
  const float gelu_g = gate * 0.5f
      * (1.0f + std::erf(gate * 0.7071067811865476f));
  return gelu_g * up;
}

// ──────────────────────────────────────────────────────────────────
// Test-parameter struct — one per (shape, act, bias, dst) combo.
// `label` is the gtest case name (composed in the test factory).
// ──────────────────────────────────────────────────────────────────
struct UkernelCase {
  int                    M, K, N;          // matmul dims
  grp_matmul_gated_act_t act;
  data_type_t            bias_dt;
  data_type_t            dst_dt;
  // Optional `ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_NR` override for this
  // case.  0 = default (let `plan_pack_nr` pick — currently NR=32 for
  // every N divisible by 64), 64 = pin NR=64 to exercise the
  // `pack_n_for_algo_3<64>` + NV=4 microkernel + 64-col store
  // epilogue paths that the default never reaches.  The override is
  // applied via `mt::CustomKernelNROverride` for the duration of the
  // call inside the TEST_P body.
  int                    nr_override;
  std::string            label;
};

inline const char *ck_act_label(grp_matmul_gated_act_t act) {
  switch (act) {
    case grp_matmul_gated_act_t::none:           return "actNone";
    case grp_matmul_gated_act_t::swiglu_oai_mul: return "swiglu";
    case grp_matmul_gated_act_t::silu_and_mul:   return "silu";
    case grp_matmul_gated_act_t::gelu_and_mul:   return "gelu";
    default:                                     return "actUnk";
  }
}

inline std::string mk_label(int M, int K, int N,
                             grp_matmul_gated_act_t act,
                             data_type_t bias_dt,
                             data_type_t dst_dt,
                             int nr_override = 0) {
  std::string s;
  s += "M";  s += std::to_string(M);
  s += "_K"; s += std::to_string(K);
  s += "_N"; s += std::to_string(N);
  s += "_";  s += ck_act_label(act);
  s += "_bias_"; s += ck_test::dt_name(bias_dt);
  s += "_dst_";  s += ck_test::dt_name(dst_dt);
  if (nr_override == 64) s += "_nr64";  // suffix only when pinned
  else if (nr_override == 32) s += "_nr32";
  return s;
}

// Suite-level CK-engagement counters.
//
// Each parameterised case reads `s_last_group_matmul_direct_gemm_mode`
// after `group_matmul_direct` returns and increments the appropriate
// counter (engaged when the resolved gemm_mode carries the `_custom`
// suffix; bypassed otherwise).  `TearDownTestSuite` then asserts that
// the suite engaged the BF16 microkernel on a non-zero number of
// cases — the regression net that distinguishes a healthy run from a
// silent fall-through to AOCL DLP across every shape.
//
// We DO NOT assert engagement per-case because `plan_group_n_tile`
// legitimately routes some shapes to Sequential (small-M / narrow-N
// at high thread counts: with NR=32 and 64 threads, N=768 yields
// 24 NR-tiles, below the planner's per-thread slack threshold,
// so it correctly picks Sequential / AOCL DLP rather than over-
// splitting).  Per-case strict assertion would false-flag those.
// Numerical correctness is still validated on whatever path runs.
class CkUkernelCorrectness
    : public ::testing::TestWithParam<UkernelCase> {
 protected:
  static std::atomic<int> s_total_cases;
  static std::atomic<int> s_ck_engaged_cases;

  static void SetUpTestSuite() {
    s_total_cases.store(0, std::memory_order_relaxed);
    s_ck_engaged_cases.store(0, std::memory_order_relaxed);
  }

  static void TearDownTestSuite() {
    const int total   = s_total_cases.load(std::memory_order_relaxed);
    const int engaged = s_ck_engaged_cases.load(std::memory_order_relaxed);
    if (total == 0) return;  // suite skipped (no BF16 ISA, etc.).
    ASSERT_GT(engaged, 0)
        << "CkUkernelCorrectness: ran " << total
        << " parameterised cases but the BF16 microkernel never "
           "engaged on any (no '_custom' gemm_mode observed).  Either "
           "every case took the AOCL DLP path or the test instrumentation "
           "is broken — inspect "
           "`zendnnl::lowoha::matmul::test_api"
           "::s_last_group_matmul_direct_gemm_mode` to debug.  See "
           "`CkUkernelEngages.OnCanonicalShape` for the strict "
           "single-shape engagement gate.";
    // Visibility: print the engagement breakdown so CI logs make it
    // easy to spot a drop in CK coverage even when the suite passes.
    std::cout << "[CkUkernelCorrectness] BF16 microkernel engaged on "
              << engaged << " / " << total << " cases ("
              << (engaged * 100.0 / total) << "%)" << std::endl;
  }
};
std::atomic<int> CkUkernelCorrectness::s_total_cases{0};
std::atomic<int> CkUkernelCorrectness::s_ck_engaged_cases{0};

TEST_P(CkUkernelCorrectness, MatchesScalarRef) {
  CK_SKIP_IF_NO_BF16_ISA();

  const auto &c = GetParam();

  // Force the CK path: ALGO=3 selection + the test-only custom-kernel
  // override (atomic, beats the cached static-const env getter that
  // backs `get_grp_matmul_custom_kernel()`).  An `EnvVarGuard` won't
  // reliably override here because the getter snapshots its env value
  // on first use; if any earlier test in the same process already
  // hit the getter (or the process started with the env unset / =0),
  // an env-var swap mid-process is invisible and the test would
  // silently fall through to the AOCL DLP path.
  mt::AlgoEnvGuard            algo_guard(3);
  mt::CustomKernelOverride    ck_guard(true);
  // Optional NR override.  Built unconditionally so the lifetime
  // matches the call below; passing 0 leaves `plan_pack_nr` on its
  // default truth-table.  Pinning to 64 (the only other supported
  // value besides 32) routes the per-tile dispatch through the
  // NV=4 microkernel + 64-col pack/store epilogue, which the
  // default never reaches because `plan_pack_nr` prefers NR=32 for
  // every N divisible by 64.  Reset caches AFTER the override
  // takes effect so any subsequent prepack/microkernel pack uses
  // the requested NR.
  mt::CustomKernelNROverride  nr_guard(c.nr_override);
  ::reset_grp_matmul_caches();

  // Three buffer-shape regimes:
  //   * swiglu_oai_mul: dst is a half-width [M, N/2] arena; ldc = N/2.
  //     Kernel writes the activated cols directly.
  //   * silu_and_mul / gelu_and_mul: dst is the full [M, N] arena;
  //     ldc = N.  CK writes the matmul output (full width); the
  //     post-pass activation rewrites cols [0, N/2) in-place leaving
  //     cols [N/2, N) as garbage by the public-API contract.
  //   * none: dst is full [M, N]; ldc = N.
  const bool is_swiglu = (c.act == grp_matmul_gated_act_t::swiglu_oai_mul);
  const bool is_split_act =
      (c.act == grp_matmul_gated_act_t::silu_and_mul
       || c.act == grp_matmul_gated_act_t::gelu_and_mul);
  // Allocation width (bytes per row count) — half for swiglu (kernel
  // halves), full for everything else.
  const int N_alloc = is_swiglu ? c.N / 2 : c.N;
  // Comparison width — half for any gated activation (only the
  // first half holds activated values; for silu/gelu the second
  // half is "garbage" by contract); full for plain matmul.
  const int N_cmp = (is_swiglu || is_split_act) ? c.N / 2 : c.N;
  // Post-activation N passed to ref_gemm_act.  Always equals the
  // comparison width: ref reads the matmul wei via `ldb = c.N` and
  // applies act-specific column math (interleave for swiglu, half-
  // offset for split-halves) using this N as the "half" dimension.
  const int N_ref = N_cmp;
  const int N_eff = N_alloc;

  // Use FOUR experts so plan_group_n_tile keeps the call on the
  // ntile path (DecodeD / FewExperts / ManyExperts).  Combined with
  // `params.num_threads = kCkTestThreads = 4` set below, this gives
  // `num_ops == num_threads` → auto-select Rule 1 fires inside
  // `auto_select_would_pick_algo1`, the auto-mirror gate returns
  // false, and the planner builds a real ntile plan that exercises
  // `GroupNTileContext::do_tile` / `custom_kernel::dispatch_tile`.
  // (Without the thread pin, num_ops < num_threads on a typical CI
  // host would trip the auto-mirror Rule 2 gate — `num_ops <= 8`
  // → Sequential — bypassing the BF16 microkernel and silently
  // running the AOCL DLP fallback for every case in the matrix.)
  // Four experts also keep `group_matmul_direct` on the parallel-
  // mode path (`src.size() == num_ops`); sequential-chain mode
  // rejects gated_act and would refuse before reaching the kernel.
  // All experts share the same shape and (deterministic) data; we
  // verify expert 0's output element-wise.
  constexpr int kNumOps = 4;

  // Per-expert src + weight buffers.
  std::vector<std::vector<bfloat16_t>> src_bufs(kNumOps);
  std::vector<std::vector<bfloat16_t>> wei_bufs(kNumOps);
  for (int e = 0; e < kNumOps; ++e) {
    src_bufs[e].assign(static_cast<size_t>(c.M) * c.K, bfloat16_t(0.0f));
    wei_bufs[e].assign(static_cast<size_t>(c.K) * c.N, bfloat16_t(0.0f));
    mt::fill_src(src_bufs[e],  /*e=*/e, 0.02f);
    mt::fill_wei1(wei_bufs[e], /*e=*/e, 0.005f);
  }

  // Bias buffer per expert (when bias_dt != none).
  std::vector<std::vector<bfloat16_t>> bias_bf16_bufs(kNumOps);
  std::vector<std::vector<float>>      bias_f32_bufs(kNumOps);
  std::vector<const void *>            bias_ptrs(kNumOps, nullptr);
  for (int e = 0; e < kNumOps; ++e) {
    if (c.bias_dt == data_type_t::bf16) {
      bias_bf16_bufs[e].assign(c.N, bfloat16_t(0.0f));
      for (int n = 0; n < c.N; ++n)
        bias_bf16_bufs[e][n] = bfloat16_t(
            0.0005f * static_cast<float>((n + e * 7) % 13 - 6));
      bias_ptrs[e] = bias_bf16_bufs[e].data();
    } else if (c.bias_dt == data_type_t::f32) {
      bias_f32_bufs[e].assign(c.N, 0.0f);
      for (int n = 0; n < c.N; ++n)
        bias_f32_bufs[e][n] =
            0.0005f * static_cast<float>((n + e * 7) % 13 - 6);
      bias_ptrs[e] = bias_f32_bufs[e].data();
    }
  }

  // Per-expert output buffers (bf16 or f32 per dst_dt).
  std::vector<std::vector<bfloat16_t>> dst_bf16_bufs(kNumOps);
  std::vector<std::vector<float>>      dst_f32_bufs(kNumOps);
  std::vector<void *>                  dst_ptrs(kNumOps, nullptr);
  for (int e = 0; e < kNumOps; ++e) {
    if (c.dst_dt == data_type_t::bf16) {
      dst_bf16_bufs[e].assign(static_cast<size_t>(c.M) * N_eff,
                              bfloat16_t(0.0f));
      dst_ptrs[e] = dst_bf16_bufs[e].data();
    } else {
      dst_f32_bufs[e].assign(static_cast<size_t>(c.M) * N_eff, 0.0f);
      dst_ptrs[e] = dst_f32_bufs[e].data();
    }
  }

  // Wrapper vectors sized to kNumOps.
  std::vector<char>         layout(kNumOps, 'r');
  std::vector<bool>         transA(kNumOps, false), transB(kNumOps, false);
  std::vector<int>          Ms(kNumOps, c.M), Ns(kNumOps, c.N),
                            Ks(kNumOps, c.K);
  std::vector<float>        alpha(kNumOps, 1.0f), beta(kNumOps, 0.0f);
  std::vector<int>          lda(kNumOps, c.K), ldb(kNumOps, c.N),
                            ldc(kNumOps, N_eff);
  std::vector<const void *> src_ptrs(kNumOps), wei_ptrs(kNumOps);
  for (int e = 0; e < kNumOps; ++e) {
    src_ptrs[e] = src_bufs[e].data();
    wei_ptrs[e] = wei_bufs[e].data();
  }
  std::vector<bool>         is_wc(kNumOps, true);

  std::vector<mt::matmul_params> params(kNumOps);
  for (auto &p : params) {
    p.dtypes.src  = data_type_t::bf16;
    p.dtypes.wei  = data_type_t::bf16;
    p.dtypes.dst  = c.dst_dt;
    p.dtypes.bias = c.bias_dt;
    // Pin the dispatcher's per-call thread team (see
    // `kCkTestThreads` doc-block) so engagement is consistent
    // across CI hosts.  `group_matmul_direct` reads
    // `params[0].num_threads` via `resolve_num_threads` and uses
    // it instead of the process-cached `thread_guard::max_threads`.
    p.num_threads = kCkTestThreads;
  }

  // Activation params (when applicable).
  zendnnl::lowoha::matmul::grp_matmul_gated_act_params act_params{};
  act_params.act = c.act;

  // Arm the test-only "last gemm_mode" capture for the scope of this
  // call.  RAII guard resets the published atomic to nullptr at
  // construction (so a previous test's value can't leak in) and
  // arms `s_capture_gemm_mode` so the dispatcher's gated store
  // fires.  Disarmed on scope exit, so any subsequent
  // `group_matmul_direct` traffic (e.g. inside a later teardown)
  // doesn't pay the cache-line-Modified penalty.
  moe_test_utils::GemmModeCaptureGuard gemm_mode_guard;

  const auto status = group_matmul_direct(
      layout, transA, transB, Ms, Ns, Ks, alpha, src_ptrs, lda,
      wei_ptrs, ldb, bias_ptrs, beta, dst_ptrs, ldc, is_wc, params,
      /*moe_postop=*/nullptr,
      c.act == grp_matmul_gated_act_t::none ? nullptr : &act_params);
  ASSERT_EQ(status, status_t::success)
      << "group_matmul_direct refused the call — case=" << c.label;

  // ── [Property] track which executor path actually ran ─────────
  // `flat_n_tile`'s `gemm_mode_label` (group_matmul_n_tile.cpp)
  // writes one of seven static literals:
  //   "flat_n_tile_sequential"             — Sequential bypassed CK
  //   "flat_n_tile"                        — non-fused, AOCL DLP
  //   "flat_n_tile_custom"                 — non-fused, CK   ✓
  //   "flat_n_tile_fused_swiglu_oai"       — fused swiglu, AOCL
  //   "flat_n_tile_fused_swiglu_oai_custom"— fused swiglu, CK ✓
  //   "flat_n_tile_fused_swiglu_oai_tight" — tight wide, AOCL
  //   "flat_n_tile_fused_swiglu_oai_tight_custom" — tight, CK ✓
  // The "_custom" suffix is the BF16 microkernel signature.  A
  // pack-cache probe alone is NOT a reliable signal — `flat_n_tile`
  // calls `prepack_for_algo_3` BEFORE building the plan, and that
  // prepack path can populate the LRU even if the plan later routes
  // to Sequential (so a probe would see `cache_hits == kNumOps` for
  // a call that never reached `dispatch_tile`).  `gemm_mode_label`
  // runs AFTER plan + execute and therefore reflects what actually
  // ran.
  //
  // We do NOT assert engagement per-case — `plan_group_n_tile`
  // legitimately routes small-M / narrow-N shapes to Sequential at
  // high thread counts.  Instead, increment a suite-level counter;
  // `TearDownTestSuite` asserts the BF16 microkernel engaged on at
  // least one case across the whole matrix, and a separate strict
  // gate test (`CkUkernelEngages.OnCanonicalShape`) pins a shape
  // known to engage at any reasonable thread count.
  const char *mode = zendnnl::lowoha::matmul::test_api
      ::s_last_group_matmul_direct_gemm_mode
      .load(std::memory_order_relaxed);
  ASSERT_NE(mode, nullptr)
      << "case '" << c.label
      << "': group_matmul_direct did not publish a gemm_mode";
  s_total_cases.fetch_add(1, std::memory_order_relaxed);
  if (std::strstr(mode, "_custom") != nullptr) {
    s_ck_engaged_cases.fetch_add(1, std::memory_order_relaxed);
  } else {
    // Visibility: surface bypassed-path cases as gtest properties so
    // CI XML reports show which shape × act × bias × dst tuples the
    // planner routed away from CK.  Not a failure — Sequential is a
    // correct planner choice on small or narrow shapes.
    RecordProperty("ck_bypassed_via", mode);
  }

  // Compare expert 0's output element-wise against the FP32 scalar
  // reference.  (Expert 1 uses different deterministic data; the
  // numerics are validated identically by the kernel for both, so
  // checking one is enough — we're testing the kernel, not the
  // dispatch over experts.)
  //
  // For silu/gelu the buffer is wide (N_eff = full N) but only the
  // first half (N_cmp = N/2) holds activated values; cols [N/2, N)
  // are explicitly garbage per the public-API contract, so the
  // comparison loop iterates [0, N_cmp).  Buffer indexing uses
  // `m * N_eff + n` (the full row stride) regardless.
  const auto tol = mt::tol_act(/*is_bf16=*/c.dst_dt == data_type_t::bf16);
  for (int m = 0; m < c.M; ++m) {
    for (int n = 0; n < N_cmp; ++n) {
      const float ref = ref_gemm_act(m, n, c.K, N_ref,
                                      src_bufs[0].data(), c.K,
                                      wei_bufs[0].data(), c.N,
                                      bias_ptrs[0],
                                      c.bias_dt, c.act);
      const float got =
          (c.dst_dt == data_type_t::bf16)
              ? to_f32(dst_bf16_bufs[0][m * N_eff + n])
              : dst_f32_bufs[0][m * N_eff + n];
      const float bound = std::abs(ref) * tol.rel + tol.abs;
      ASSERT_NEAR(got, ref, bound)
          << "case=" << c.label << " m=" << m << " n=" << n
          << " ref=" << ref << " got=" << got;
    }
  }
}

// ──────────────────────────────────────────────────────────────────
// Build the parameter set — focused matrix that hits every distinct
// code path without exploding into 1000+ cases.
//
// Shape grid: kept CI-friendly.  Each shape is instantiated against
// an inline scalar FP32 reference for element-wise comparison; the
// reference cost scales linearly with M × N × K and is single-
// threaded per case, so large-K shapes crossed with the full
// (act × bias × dst) tuple would balloon the suite into seconds
// of scalar work per case on slower CI hosts.
//
// The grid is split into four groups (all NR=32 default unless
// otherwise noted):
//
//   * `small_shapes` — small-K, fast scalar reference.  Get the
//     FULL (act × bias × dst) cross-product (21 cases each) so the
//     epilogue / bias-load / activation paths are exhaustively
//     covered against the kernel's MR fan-out at the NR=32 default.
//
//   * `large_shapes` — moderate-K shapes (K=1024) with two N
//     widths (wide and narrow) that exercise the kernel's K-loop
//     in a realistic-ish steady-state regime (~512 K-pair
//     iterations per call).  These get a SMOKE subset (4 cases
//     each: act ∈ {none, swiglu_oai_mul} × dst ∈ {bf16}, plus
//     (act=none, dst=f32), plus (act=none, bias=bf16, dst=bf16)
//     for bias-load coverage).  The full epilogue × bias × dst
//     cross-product is already covered by the small-K shapes;
//     these rows just smoke-test that the kernel still produces
//     correct output at a longer K-loop.  Production-scale
//     shapes (K=2880+, N=5760+) belong in the benchmark harness,
//     not the correctness suite — the K-loop's correctness does
//     not depend on iteration count once K > ~64, and the
//     scalar reference cost scales linearly with M × N × K.
//
//   * `edge_shapes` — K parity + smallest valid N (narrow sweep:
//     act=none × bias ∈ {none, bf16} × dst ∈ {bf16, f32}).
//
//   * `nr64_shapes` — NR=64 forced sweep.  `plan_pack_nr` prefers
//     NR=32 for every shape, so without an explicit override the
//     NV=4 microkernel + 64-col store-epilogue path is never hit.
//     These cases pin `ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_NR=64` via
//     `CustomKernelNROverride(64)` for the duration of the call so
//     the NR=64 epilogue (including the new BF16:BF16:F32 NV=4
//     variant added in this PR) actually runs against the scalar
//     reference.  Tight tuple set (4 per shape × 2 shapes = 8) to
//     stay in the CI cost budget; the NR=32 default is still the
//     dominant matrix above.
// ──────────────────────────────────────────────────────────────────
static std::vector<UkernelCase> make_ukernel_cases() {
  struct Shape { int M, K, N; };
  const Shape small_shapes[] = {
      // (M, K, N) — small-K, full (act × bias × dst) cross-product.
      // All cases run with `plan_pack_nr`'s default truth-table
      // (NR=32 for every shape here).  NR=64 is not reachable
      // through these — `plan_pack_nr` prefers NR=32 even for shapes
      // where N % 64 == 0 — so the NR=64 path is exercised by the
      // separate `nr64_shapes` sweep at the end of this factory,
      // which holds `CustomKernelNROverride(64)` for the call.
      {1,    64,   256},   // tiny — exercise MR=1 single-row path
      {4,    64,   256},   // mini decode
      {16,   256,  512},   // mid decode (MR fan-out + N-tile splits)
      {16,    64,  512},   // N % 64 == 0 — runs as NR=32 by default;
                            // see nr64_shapes for the NR=64 variant.
      {8,    128,  256},   // multi-MR per-call partition
  };
  const Shape large_shapes[] = {
      // (M, K, N) — moderate-K, smoke subset only (4 cases each).
      // K=1024 → 512 K-pair iterations of the inner loop, plenty
      // for steady-state correctness.  N=2048 (wide) and N=768
      // (narrow) are both N % 32 == 0; together with the small-K
      // shapes' N % 64 == 0 entry these cover both pack_nr
      // regimes.
      {4,    1024, 2048},  // wide-N moderate-K (N=2*K)
      {4,    1024, 768 },  // narrow-N moderate-K (N=0.75*K)
  };
  // Edge-shape additions: K parity + N=2*pack_nr (smallest valid).
  const Shape edge_shapes[] = {
      {16,  63,  64 },    // odd K + smallest N
      {4,   65,  128},    // K = even+1, N = 4 * pack_nr / 2
      {16, 256,  64 },    // smallest N at pack_nr=32
  };

  std::vector<UkernelCase> cases;
  // 5 small_shapes × 21 (act × bias × dst, swiglu+f32 filtered) = 105
  // 2 large_shapes × 4 smoke cases each                          =   8
  // 3 edge_shapes × 4                                            =  12
  // 2 nr64_shapes × 4 NR=64-pinned tuples each                   =   8
  // Total                                                        ~133
  cases.reserve(5 * 21 + 2 * 4 + 3 * 4 + 2 * 4);

  for (const auto &s : small_shapes) {
    for (auto act : {grp_matmul_gated_act_t::none,
                     grp_matmul_gated_act_t::swiglu_oai_mul,
                     grp_matmul_gated_act_t::silu_and_mul,
                     grp_matmul_gated_act_t::gelu_and_mul}) {
      for (auto bias : {data_type_t::none, data_type_t::bf16,
                        data_type_t::f32}) {
        for (auto dst : {data_type_t::bf16, data_type_t::f32}) {
          // Filter the structurally invalid (swiglu, FP32-dst) tuple
          // — swiglu's pair-pack store helper writes BF16 only.
          if (act == grp_matmul_gated_act_t::swiglu_oai_mul
              && dst == data_type_t::f32) {
            continue;
          }
          // silu/gelu are split-halves; their post-activation valid
          // region is `c.N / 2`.  N must be even for the split to
          // work — the smallest shape (N=128) and others in the
          // grid are already even, so no shape filter is needed.
          cases.push_back(
              {s.M, s.K, s.N, act, bias, dst, /*nr_override=*/0,
               mk_label(s.M, s.K, s.N, act, bias, dst)});
        }
      }
    }
  }

  // Smoke subset for production-K shapes — bounded scalar-reference
  // cost while still exercising the realistic K-loop + pack-NR path.
  // Curated tuples (each one covers a distinct kernel branch):
  //   * (none,   none, bf16) — plain matmul, BF16 epilogue.
  //   * (none,   none, f32 ) — plain matmul, FP32 epilogue.
  //   * (none,   bf16, bf16) — bias-load path on top of plain matmul.
  //   * (swiglu, none, bf16) — fused-swiglu epilogue.
  struct SmokeTuple {
    grp_matmul_gated_act_t act;
    data_type_t bias;
    data_type_t dst;
  };
  const SmokeTuple smoke[] = {
      {grp_matmul_gated_act_t::none,           data_type_t::none, data_type_t::bf16},
      {grp_matmul_gated_act_t::none,           data_type_t::none, data_type_t::f32 },
      {grp_matmul_gated_act_t::none,           data_type_t::bf16, data_type_t::bf16},
      {grp_matmul_gated_act_t::swiglu_oai_mul, data_type_t::none, data_type_t::bf16},
  };
  for (const auto &s : large_shapes) {
    for (const auto &t : smoke) {
      cases.push_back(
          {s.M, s.K, s.N, t.act, t.bias, t.dst, /*nr_override=*/0,
           mk_label(s.M, s.K, s.N, t.act, t.bias, t.dst)});
    }
  }

  // K-parity / pack-NR-edge cases: act=none × bias ∈ {none, bf16} ×
  // dst ∈ {bf16, f32} = 4 cases per shape.  Activation paths
  // (swiglu / silu / gelu) are not exercised on these shapes — the
  // small-K full-cross-product matrix above already covers the
  // activation × dst space exhaustively, so the edge shapes only
  // need to confirm K-parity / pack-NR-edge correctness for plain
  // matmul plus the bias-load.
  for (const auto &s : edge_shapes) {
    for (auto bias : {data_type_t::none, data_type_t::bf16}) {
      for (auto dst : {data_type_t::bf16, data_type_t::f32}) {
        cases.push_back(
            {s.M, s.K, s.N, grp_matmul_gated_act_t::none,
             bias, dst, /*nr_override=*/0,
             mk_label(s.M, s.K, s.N, grp_matmul_gated_act_t::none,
                      bias, dst)});
      }
    }
  }

  // ── NR=64 forced sweep ──────────────────────────────────────────
  // `plan_pack_nr` (group_matmul_n_tile.cpp) currently prefers NR=32
  // for every N divisible by 64 — the default never reaches the
  // `pack_n_for_algo_3<64>` packer, the NV=4 microkernel, or its
  // 64-col store-epilogue path.  These shapes pin
  // `ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_NR=64` via the test override
  // so the NR=64 path is actually exercised end-to-end against the
  // scalar reference.  The shape and tuple set is intentionally
  // tight (N % 64 == 0 only — narrow-NR-32 shapes are obviously
  // ineligible — and a curated act × bias × dst handful) to keep
  // the suite under its CI cost budget; the NR=32 default is still
  // the dominant matrix above.  Cases:
  //   * (none, none, bf16) — plain matmul, BF16 epilogue.
  //   * (none, none, f32 ) — plain matmul, FP32 epilogue (new in
  //     this PR — one of the variants Copilot's review flagged as
  //     "newly instantiated NV=4/NR=64 microkernels … unexercised").
  //   * (none, bf16, bf16) — bias-load on BF16 dst.
  //   * (swiglu_oai_mul, none, bf16) — fused-swiglu epilogue.
  struct NrShape { int M, K, N; };
  const NrShape nr64_shapes[] = {
      {16,    64,  512 },   // small-K, N % 64 == 0 (8 NR=64 tiles)
      { 4,  1024, 2048 },   // moderate-K, wide-N (32 NR=64 tiles)
  };
  struct NrTuple {
    grp_matmul_gated_act_t act;
    data_type_t            bias;
    data_type_t            dst;
  };
  const NrTuple nr64_tuples[] = {
      {grp_matmul_gated_act_t::none,           data_type_t::none, data_type_t::bf16},
      {grp_matmul_gated_act_t::none,           data_type_t::none, data_type_t::f32 },
      {grp_matmul_gated_act_t::none,           data_type_t::bf16, data_type_t::bf16},
      {grp_matmul_gated_act_t::swiglu_oai_mul, data_type_t::none, data_type_t::bf16},
  };
  for (const auto &s : nr64_shapes) {
    for (const auto &t : nr64_tuples) {
      cases.push_back(
          {s.M, s.K, s.N, t.act, t.bias, t.dst, /*nr_override=*/64,
           mk_label(s.M, s.K, s.N, t.act, t.bias, t.dst,
                    /*nr_override=*/64)});
    }
  }

  return cases;
}

// gtest holds parameter sources for the lifetime of the test suite,
// and `::testing::ValuesIn(const Container&)` may store iterators
// into that container.  Passing a temporary `vector` would leave
// dangling iterators after the rvalue's destruction; wrap the
// builder in an immediately-invoked lambda whose function-local
// static gives the container static storage duration.
INSTANTIATE_TEST_SUITE_P(
    ShapeMatrix, CkUkernelCorrectness,
    ::testing::ValuesIn([]() -> const std::vector<UkernelCase>& {
      static const std::vector<UkernelCase> kCases = make_ukernel_cases();
      return kCases;
    }()),
    [](const ::testing::TestParamInfo<UkernelCase> &info) {
      return info.param.label;
    });

// ──────────────────────────────────────────────────────────────────
// Strict single-shape engagement gate — the regression net for
// "did CK actually run?"
//
// Distinct from `CkUkernelCorrectness` (which validates numerical
// correctness across a wide matrix on whatever path the planner
// picks): this test pins a shape known to engage `dispatch_tile`
// at any reasonable thread count and asserts the resolved gemm_mode
// carries the `_custom` suffix.  If it ever doesn't, either the
// planner regressed (now bypassing CK on production-MoE shapes) or
// the kernel gate refused (kctx mismatch, unsupported variant, etc.)
// — both are bugs worth failing the build on.
//
// Canonical shape: M=4 × K=1024 × N=2048, four experts, plain
// matmul (no act, no bias), BF16 dst.  N=2048 / NR=32 = 64 NR-tiles
// — enough slack across 64 threads × 4 experts that the planner
// stays in the per-tile dispatch path even at high thread counts.
// ──────────────────────────────────────────────────────────────────
TEST(CkUkernelEngages, OnCanonicalShape) {
  CK_SKIP_IF_NO_BF16_ISA();

  mt::AlgoEnvGuard         algo_guard(3);
  mt::CustomKernelOverride ck_guard(true);
  ::reset_grp_matmul_caches();

  constexpr int kNumOps = 4;
  constexpr int M = 4, K = 1024, N = 2048;

  std::vector<std::vector<bfloat16_t>> src_bufs(kNumOps);
  std::vector<std::vector<bfloat16_t>> wei_bufs(kNumOps);
  std::vector<std::vector<bfloat16_t>> dst_bufs(kNumOps);
  for (int e = 0; e < kNumOps; ++e) {
    src_bufs[e].assign(static_cast<size_t>(M) * K, bfloat16_t(0.0f));
    wei_bufs[e].assign(static_cast<size_t>(K) * N, bfloat16_t(0.0f));
    dst_bufs[e].assign(static_cast<size_t>(M) * N, bfloat16_t(0.0f));
    mt::fill_src(src_bufs[e],  /*e=*/e, 0.02f);
    mt::fill_wei1(wei_bufs[e], /*e=*/e, 0.005f);
  }

  std::vector<char>         layout(kNumOps, 'r');
  std::vector<bool>         transA(kNumOps, false), transB(kNumOps, false);
  std::vector<int>          Ms(kNumOps, M), Ns(kNumOps, N), Ks(kNumOps, K);
  std::vector<float>        alpha(kNumOps, 1.0f), beta(kNumOps, 0.0f);
  std::vector<int>          lda(kNumOps, K), ldb(kNumOps, N), ldc(kNumOps, N);
  std::vector<const void *> src_ptrs(kNumOps), wei_ptrs(kNumOps);
  std::vector<const void *> bias_ptrs(kNumOps, nullptr);
  std::vector<void *>       dst_ptrs(kNumOps);
  for (int e = 0; e < kNumOps; ++e) {
    src_ptrs[e] = src_bufs[e].data();
    wei_ptrs[e] = wei_bufs[e].data();
    dst_ptrs[e] = dst_bufs[e].data();
  }
  std::vector<bool> is_wc(kNumOps, true);

  std::vector<mt::matmul_params> params(kNumOps);
  for (auto &p : params) {
    p.dtypes.src  = data_type_t::bf16;
    p.dtypes.wei  = data_type_t::bf16;
    p.dtypes.dst  = data_type_t::bf16;
    p.dtypes.bias = data_type_t::none;
    // Pin the dispatcher's per-call thread team for deterministic
    // CK engagement across CI hosts (see `kCkTestThreads`
    // doc-block).  Without this, hosts with `OMP_NUM_THREADS`
    // very high (e.g. 256) would see the planner correctly bypass
    // CK on this shape because per-thread N-tile slack drops
    // below the engagement floor — which would false-fail the
    // gate even though the kernel itself is healthy.  Pinning
    // keeps this a kernel regression detector rather than a
    // planner stress test.
    p.num_threads = kCkTestThreads;
  }

  // Arm the gemm_mode capture for the scope of this call; see the
  // doc-block on `GemmModeCaptureGuard` in `moe_test_utils.hpp`.
  moe_test_utils::GemmModeCaptureGuard gemm_mode_guard;

  const auto status = group_matmul_direct(
      layout, transA, transB, Ms, Ns, Ks, alpha, src_ptrs, lda,
      wei_ptrs, ldb, bias_ptrs, beta, dst_ptrs, ldc, is_wc, params,
      /*moe_postop=*/nullptr, /*gated_act=*/nullptr);
  ASSERT_EQ(status, status_t::success);

  const char *mode = zendnnl::lowoha::matmul::test_api
      ::s_last_group_matmul_direct_gemm_mode
      .load(std::memory_order_relaxed);
  ASSERT_NE(mode, nullptr) << "group_matmul_direct did not publish a gemm_mode";
  ASSERT_NE(std::strstr(mode, "_custom"), nullptr)
      << "Canonical engagement shape (M=4 × K=1024 × N=2048, 4 experts) "
         "ran on '" << mode << "' instead of the BF16 microkernel.  "
         "Either the planner now refuses CK on production-MoE shapes, "
         "the dispatcher fell back to AOCL DLP, or the BF16 ISA gate / "
         "custom-kernel override regressed.  This is the strict CK "
         "engagement gate — investigate before merging.";
}

}  // namespace
