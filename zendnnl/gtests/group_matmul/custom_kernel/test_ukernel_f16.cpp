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

/// FP16 microkernel end-to-end correctness -- sibling of
/// test_ukernel_bf16.cpp.  Exercises both the `f16:f16:f16` and
/// `f16:f16:f32` variants of the native AVX-512-FP16 custom kernel
/// against a scalar FP32 reference computed alongside.  The test goes
/// through `group_matmul_direct` with `ZENDNNL_GRP_MATMUL_ALGO=3`, the
/// master custom-kernel gate forced on, AND the FP16 sub-toggle forced
/// on, so the ALGO 3 dispatcher takes the FP16 CK path (when supported
/// by the host + toolchain) and the per-tile invocation actually fires.
///
/// Coverage axes (parameterised):
///   * Shape grid: (M, K, N) in a curated set covering common MoE
///     decode shapes and CK pack-NR boundaries (shared with the bf16
///     sibling).
///   * Activation: {none, swiglu_oai_mul, silu_and_mul, gelu_and_mul}.
///     - `swiglu_oai_mul` -- fused in the per-tile epilogue, halved
///       output width (interleaved `[g0,u0,g1,u1,...]` layout).
///     - `silu_and_mul` / `gelu_and_mul` -- split-halves
///       `[gate_cols | up_cols]` layout; prepack re-interleaves so the
///       FP16 kernel sees the same physical layout as swiglu_oai_mul.
///     - `none` -- plain matmul, full N-wide output.
///   * Bias dtype: {none, bf16, f32, f16}.  A bf16/f32 bias is narrowed
///     to f16 at accumulator init; an f16 bias is loaded directly.
///   * Dst dtype: {f16, f32} -- but every gated kind is f16-dst only
///     (the pair-store helpers write f16), so `(gated, f32)` is
///     structurally invalid and filtered here, mirroring the bf16
///     sibling's `(swiglu, f32)` filter.
///
/// Reference: a scalar FP32 GEMM computed inline (no library call).
/// For swiglu the reference reads gate / up from interleaved cols
/// (2n+0, 2n+1).  For silu/gelu it reads gate from cols [0, N/2) and
/// up from cols [N/2, N), matching the split-halves public-API
/// contract.
///
/// PRECISION NOTE -- the FP16 microkernel accumulates the dot product
/// in NATIVE FP16 (`_mm512_fmadd_ph`), unlike the bf16 sibling which
/// accumulates in FP32.  So BOTH the f16 dst AND the f32 dst carry the
/// native-FP16-accumulate divergence vs the FP32 scalar reference --
/// the f32 dst is NOT FP32-accumulate-accurate here.  We therefore use
/// the f16 tolerance band (`tol_act(data_type_t::f16)`) for EVERY
/// comparison regardless of dst dtype, which is the documented contract
/// for this kernel (see the PRECISION NOTE in f16_microkernel.hpp).

#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "ck_test_helpers.hpp"
#include "moe_test_utils.hpp"

// For the CK-engagement assertion: `group_matmul_direct` publishes the
// resolved `gemm_mode` static literal to a test-only atomic in
// `test_api::s_last_group_matmul_direct_gemm_mode`.  Reading that after
// the call is the reliable signal that the custom kernel actually ran.
// The gemm_mode label does not encode the dtype family -- the FP16 CK
// path emits the same `_custom`-suffixed labels as the bf16 path
// (`flat_n_tile_custom`, `flat_n_tile_fused_silu_and_mul_tight_custom`,
// ...), so the `_custom` substring check is family-agnostic.
#include "lowoha_operators/matmul/group_matmul/group_matmul_parallel_common.hpp"

namespace {

namespace mt = moe_test_utils;
using mt::data_type_t;
using mt::float16_t;
using mt::group_matmul_direct;
using mt::grp_matmul_gated_act_t;
using mt::status_t;

// Convert an F16 element to FP32 for reference comparisons.
inline float to_f32(float16_t v) {
    return static_cast<float>(v);
}

// Pin the dispatcher's per-call thread team to a moderate value
// regardless of the CI host's `OMP_NUM_THREADS` (same rationale as the
// bf16 sibling's `kCkTestThreads`).  4 threads: small enough that
// `plan_group_n_tile` stays on the n-tile path (not the smaller-tile
// fallback) across all parameterised shapes here, so every case
// actually exercises the CK kernel — and it stays within typical CI
// thread limits.
constexpr int kCkTestThreads = 4;

// ------------------------------------------------------------------
// Helper: scalar FP32 matmul element at (m, n) with optional bias.
// Weight contract: transB=false, so wei is [K, N] row-major and
// `ldb = N` is the stride between K-rows.  The reference accumulates
// in FP32; the kernel accumulates in native FP16 -- the f16 tolerance
// band absorbs the difference (see the PRECISION NOTE up top).
// ------------------------------------------------------------------
inline float ref_matmul_elem(int m, int n, int K, const float16_t *src, int lda,
        const float16_t *wei, int ldb, const void *bias, data_type_t bias_dt) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += to_f32(src[m * lda + k]) * to_f32(wei[k * ldb + n]);
    }
    if (bias_dt == data_type_t::bf16) {
        acc += static_cast<float>(static_cast<const mt::bfloat16_t *>(bias)[n]);
    } else if (bias_dt == data_type_t::f32) {
        acc += static_cast<const float *>(bias)[n];
    } else if (bias_dt == data_type_t::f16) {
        acc += to_f32(static_cast<const float16_t *>(bias)[n]);
    }
    return acc;
}

// `N` param: its meaning depends on `act`.
//   * act = none      → the full output width (unused by the plain
//                        matmul element, passed for signature symmetry).
//   * act = swiglu     → unused (gate/up are the interleaved cols
//                        `2n` / `2n+1`, no N-derived offset).
//   * act = silu/gelu  → the POST-activation half-width; the up column
//                        for output col `n` is `n + N` (caller passes
//                        the already-halved width, i.e. `c.N / 2`).
inline float ref_gemm_act(int m, int n, int K, int N, const float16_t *src,
        int lda, const float16_t *wei, int ldb, const void *bias,
        data_type_t bias_dt, grp_matmul_gated_act_t act) {
    // act = none -- plain matmul element (m, n).
    if (act == grp_matmul_gated_act_t::none) {
        return ref_matmul_elem(m, n, K, src, lda, wei, ldb, bias, bias_dt);
    }

    // act = swiglu_oai_mul (interleaved gate/up; halved output).
    // gate at even col `2n+0`, up at odd col `2n+1`.  gate, up clamped
    // to [-7, 7]; sig = sigmoid(gate * 1.702f); out = (1 + up) * gate*sig.
    if (act == grp_matmul_gated_act_t::swiglu_oai_mul) {
        float acc_g = ref_matmul_elem(
                m, 2 * n + 0, K, src, lda, wei, ldb, bias, bias_dt);
        float acc_u = ref_matmul_elem(
                m, 2 * n + 1, K, src, lda, wei, ldb, bias, bias_dt);
        acc_g = std::max(-7.0f, std::min(7.0f, acc_g));
        acc_u = std::max(-7.0f, std::min(7.0f, acc_u));
        const float sig = 1.0f / (1.0f + std::exp(-acc_g * 1.702f));
        return (1.0f + acc_u) * (acc_g * sig);
    }

    // act = silu_and_mul / gelu_and_mul (split-halves).  gate at col `n`
    // in the first half; up at col `n + N/2` in the second half.  N here
    // is the post-activation half-width (caller already halved).
    const int gate_col = n;
    const int up_col = n + N;
    const float gate = ref_matmul_elem(
            m, gate_col, K, src, lda, wei, ldb, bias, bias_dt);
    const float up
            = ref_matmul_elem(m, up_col, K, src, lda, wei, ldb, bias, bias_dt);
    if (act == grp_matmul_gated_act_t::silu_and_mul) {
        const float sigmoid_g = 1.0f / (1.0f + std::exp(-gate));
        return (gate * sigmoid_g) * up;
    }
    // gelu_and_mul (erf form, matches `moe_test_utils::ref_gelu_mul`).
    const float gelu_g
            = gate * 0.5f * (1.0f + std::erf(gate * 0.7071067811865476f));
    return gelu_g * up;
}

// ------------------------------------------------------------------
// Test-parameter struct -- one per (shape, act, bias, dst) combo.
// ------------------------------------------------------------------
struct UkernelCase {
    int M, K, N;
    grp_matmul_gated_act_t act;
    data_type_t bias_dt;
    data_type_t dst_dt;
    // Optional NR override (0 = default truth-table NR=32, 64 = pin
    // NR=64 to exercise the NV=4 microkernel + 64-col store epilogue).
    int nr_override;
    std::string label;
};

inline const char *ck_act_label(grp_matmul_gated_act_t act) {
    switch (act) {
        case grp_matmul_gated_act_t::none: return "actNone";
        case grp_matmul_gated_act_t::swiglu_oai_mul: return "swiglu";
        case grp_matmul_gated_act_t::silu_and_mul: return "silu";
        case grp_matmul_gated_act_t::gelu_and_mul: return "gelu";
        default: return "actUnk";
    }
}

inline std::string mk_label(int M, int K, int N, grp_matmul_gated_act_t act,
        data_type_t bias_dt, data_type_t dst_dt, int nr_override = 0) {
    std::string s;
    s += "M";
    s += std::to_string(M);
    s += "_K";
    s += std::to_string(K);
    s += "_N";
    s += std::to_string(N);
    s += "_";
    s += ck_act_label(act);
    s += "_bias_";
    s += ck_test::dt_name(bias_dt);
    s += "_dst_";
    s += ck_test::dt_name(dst_dt);
    if (nr_override == 64)
        s += "_nr64";
    else if (nr_override == 32)
        s += "_nr32";
    return s;
}

// Suite-level CK-engagement counters.  Each parameterised case reads
// `s_last_group_matmul_direct_gemm_mode` after `group_matmul_direct`
// returns and increments the appropriate counter (engaged when the
// resolved gemm_mode carries the `_custom` suffix; bypassed
// otherwise).  `TearDownTestSuite` asserts the suite engaged the FP16
// microkernel on a non-zero number of cases -- the regression net that
// distinguishes a healthy run from a silent fall-through to AOCL DLP
// across every shape.  We do NOT assert engagement per-case because
// `plan_group_n_tile` legitimately routes some shapes to Sequential
// (small-M / narrow-N at high thread counts); numerical correctness is
// still validated on whatever path runs.
class CkF16UkernelCorrectness : public ::testing::TestWithParam<UkernelCase> {
protected:
    static std::atomic<int> s_total_cases;
    static std::atomic<int> s_ck_engaged_cases;

    static void SetUpTestSuite() {
        s_total_cases.store(0, std::memory_order_relaxed);
        s_ck_engaged_cases.store(0, std::memory_order_relaxed);
    }

    static void TearDownTestSuite() {
        const int total = s_total_cases.load(std::memory_order_relaxed);
        const int engaged = s_ck_engaged_cases.load(std::memory_order_relaxed);
        if (total == 0) return; // suite skipped (no FP16 ISA/toolchain).
        ASSERT_GT(engaged, 0)
                << "CkF16UkernelCorrectness: ran " << total
                << " parameterised cases but the FP16 microkernel never "
                   "engaged on any (no '_custom' gemm_mode observed).  Either "
                   "every case took the AOCL DLP path or the test "
                   "instrumentation "
                   "is broken -- inspect "
                   "`zendnnl::lowoha::matmul::test_api"
                   "::s_last_group_matmul_direct_gemm_mode` to debug.  See "
                   "`CkF16UkernelEngages.OnCanonicalShape` for the strict "
                   "single-shape engagement gate.";
        std::cout << "[CkF16UkernelCorrectness] FP16 microkernel engaged on "
                  << engaged << " / " << total << " cases ("
                  << (engaged * 100.0 / total) << "%)" << std::endl;
    }
};
std::atomic<int> CkF16UkernelCorrectness::s_total_cases {0};
std::atomic<int> CkF16UkernelCorrectness::s_ck_engaged_cases {0};

TEST_P(CkF16UkernelCorrectness, MatchesScalarRef) {
    CK_SKIP_IF_NO_F16_ISA();

    const auto &c = GetParam();

    // Force the FP16 CK path: ALGO=3 selection + the master custom-kernel
    // override + the FP16 sub-toggle override (all atomic, beating the
    // cached static-const env getters).  An `EnvVarGuard` would not
    // reliably override here because the getters snapshot their env value
    // on first use.
    mt::AlgoEnvGuard algo_guard(3);
    mt::CustomKernelOverride ck_guard(true);
    mt::CustomKernelF16Override ck_f16_guard(true);
    // Optional NR override.  Built unconditionally so the lifetime
    // matches the call below; passing 0 leaves `plan_pack_nr` on its
    // default truth-table.  Reset caches AFTER the override takes effect.
    mt::CustomKernelNROverride nr_guard(c.nr_override);
    ::reset_grp_matmul_caches();

    // Two buffer-shape regimes (identical to the bf16 sibling):
    //   * Any gated activation -- dst is a half-width [M, N/2] arena;
    //     ldc = N/2.  Kernel writes the activated cols directly.
    //   * `none` -- dst is full [M, N]; ldc = N.  Plain matmul.
    const bool is_gated_fused_ck
            = (c.act == grp_matmul_gated_act_t::swiglu_oai_mul)
            || (c.act == grp_matmul_gated_act_t::silu_and_mul)
            || (c.act == grp_matmul_gated_act_t::gelu_and_mul);
    const int N_alloc = is_gated_fused_ck ? c.N / 2 : c.N;
    const int N_cmp = is_gated_fused_ck ? c.N / 2 : c.N;
    const int N_ref = N_cmp;
    const int N_eff = N_alloc;

    // Four experts so plan_group_n_tile keeps the call on the ntile path
    // (same rationale as the bf16 sibling): the planner builds a real
    // ntile plan that exercises `do_tile` / `dispatch_tile`.
    constexpr int kNumOps = 4;

    // Per-expert src + weight buffers (FP16).
    std::vector<std::vector<float16_t>> src_bufs(kNumOps);
    std::vector<std::vector<float16_t>> wei_bufs(kNumOps);
    for (int e = 0; e < kNumOps; ++e) {
        src_bufs[e].assign(static_cast<size_t>(c.M) * c.K, float16_t(0.0f));
        wei_bufs[e].assign(static_cast<size_t>(c.K) * c.N, float16_t(0.0f));
        mt::fill_src(src_bufs[e], /*e=*/e, 0.02f);
        mt::fill_wei1(wei_bufs[e], /*e=*/e, 0.005f);
    }

    // Bias buffer per expert (when bias_dt != none).  The FP16 family
    // accepts bf16 / f32 (narrowed to f16 at accumulator init) as well as
    // an f16 bias (loaded directly).
    std::vector<std::vector<mt::bfloat16_t>> bias_bf16_bufs(kNumOps);
    std::vector<std::vector<float>> bias_f32_bufs(kNumOps);
    std::vector<std::vector<float16_t>> bias_f16_bufs(kNumOps);
    std::vector<const void *> bias_ptrs(kNumOps, nullptr);
    for (int e = 0; e < kNumOps; ++e) {
        if (c.bias_dt == data_type_t::bf16) {
            bias_bf16_bufs[e].assign(c.N, mt::bfloat16_t(0.0f));
            for (int n = 0; n < c.N; ++n)
                bias_bf16_bufs[e][n] = mt::bfloat16_t(
                        0.0005f * static_cast<float>((n + e * 7) % 13 - 6));
            bias_ptrs[e] = bias_bf16_bufs[e].data();
        } else if (c.bias_dt == data_type_t::f32) {
            bias_f32_bufs[e].assign(c.N, 0.0f);
            for (int n = 0; n < c.N; ++n)
                bias_f32_bufs[e][n]
                        = 0.0005f * static_cast<float>((n + e * 7) % 13 - 6);
            bias_ptrs[e] = bias_f32_bufs[e].data();
        } else if (c.bias_dt == data_type_t::f16) {
            bias_f16_bufs[e].assign(c.N, float16_t(0.0f));
            for (int n = 0; n < c.N; ++n)
                bias_f16_bufs[e][n] = float16_t(
                        0.0005f * static_cast<float>((n + e * 7) % 13 - 6));
            bias_ptrs[e] = bias_f16_bufs[e].data();
        }
    }

    // Per-expert output buffers (f16 or f32 per dst_dt).
    std::vector<std::vector<float16_t>> dst_f16_bufs(kNumOps);
    std::vector<std::vector<float>> dst_f32_bufs(kNumOps);
    std::vector<void *> dst_ptrs(kNumOps, nullptr);
    for (int e = 0; e < kNumOps; ++e) {
        if (c.dst_dt == data_type_t::f16) {
            dst_f16_bufs[e].assign(
                    static_cast<size_t>(c.M) * N_eff, float16_t(0.0f));
            dst_ptrs[e] = dst_f16_bufs[e].data();
        } else {
            dst_f32_bufs[e].assign(static_cast<size_t>(c.M) * N_eff, 0.0f);
            dst_ptrs[e] = dst_f32_bufs[e].data();
        }
    }

    // Wrapper vectors sized to kNumOps.
    std::vector<char> layout(kNumOps, 'r');
    std::vector<bool> transA(kNumOps, false), transB(kNumOps, false);
    std::vector<int> Ms(kNumOps, c.M), Ns(kNumOps, c.N), Ks(kNumOps, c.K);
    std::vector<float> alpha(kNumOps, 1.0f), beta(kNumOps, 0.0f);
    std::vector<int> lda(kNumOps, c.K), ldb(kNumOps, c.N), ldc(kNumOps, N_eff);
    std::vector<const void *> src_ptrs(kNumOps), wei_ptrs(kNumOps);
    for (int e = 0; e < kNumOps; ++e) {
        src_ptrs[e] = src_bufs[e].data();
        wei_ptrs[e] = wei_bufs[e].data();
    }
    std::vector<bool> is_wc(kNumOps, true);

    std::vector<mt::matmul_params> params(kNumOps);
    for (auto &p : params) {
        p.dtypes.src = data_type_t::f16;
        p.dtypes.wei = data_type_t::f16;
        p.dtypes.dst = c.dst_dt;
        p.dtypes.bias = c.bias_dt;
        p.num_threads = kCkTestThreads;
    }

    zendnnl::lowoha::matmul::grp_matmul_gated_act_params act_params {};
    act_params.act = c.act;

    moe_test_utils::GemmModeCaptureGuard gemm_mode_guard;

    const auto status = group_matmul_direct(layout, transA, transB, Ms, Ns, Ks,
            alpha, src_ptrs, lda, wei_ptrs, ldb, bias_ptrs, beta, dst_ptrs, ldc,
            is_wc, params,
            /*moe_postop=*/nullptr,
            c.act == grp_matmul_gated_act_t::none ? nullptr : &act_params);
    ASSERT_EQ(status, status_t::success)
            << "group_matmul_direct refused the call -- case=" << c.label;

    // Track which executor path actually ran.  The `_custom` suffix is
    // the FP16 microkernel signature (same labels as the bf16 path; the
    // dtype family is not encoded in the gemm_mode string).
    const char *mode = zendnnl::lowoha::matmul::test_api ::
                               s_last_group_matmul_direct_gemm_mode.load(
                                       std::memory_order_relaxed);
    ASSERT_NE(mode, nullptr)
            << "case '" << c.label
            << "': group_matmul_direct did not publish a gemm_mode";
    s_total_cases.fetch_add(1, std::memory_order_relaxed);
    if (std::strstr(mode, "_custom") != nullptr) {
        s_ck_engaged_cases.fetch_add(1, std::memory_order_relaxed);
    } else {
        RecordProperty("ck_bypassed_via", mode);
    }

    // Compare expert 0's output element-wise against the FP32 scalar
    // reference.  Native-FP16 accumulation means the f32 dst is NOT
    // FP32-accumulate-accurate, so we use the f16 tolerance band for
    // EVERY dst dtype (see the PRECISION NOTE at the top of this file).
    const auto tol = mt::tol_act(data_type_t::f16);
    for (int m = 0; m < c.M; ++m) {
        for (int n = 0; n < N_cmp; ++n) {
            const float ref = ref_gemm_act(m, n, c.K, N_ref, src_bufs[0].data(),
                    c.K, wei_bufs[0].data(), c.N, bias_ptrs[0], c.bias_dt,
                    c.act);
            const float got = (c.dst_dt == data_type_t::f16)
                    ? to_f32(dst_f16_bufs[0][m * N_eff + n])
                    : dst_f32_bufs[0][m * N_eff + n];
            const float bound = std::abs(ref) * tol.rel + tol.abs;
            ASSERT_NEAR(got, ref, bound)
                    << "case=" << c.label << " m=" << m << " n=" << n
                    << " ref=" << ref << " got=" << got;
        }
    }
}

// ------------------------------------------------------------------
// Build the parameter set -- mirrors the bf16 sibling's focused matrix
// (small-K full cross-product, moderate-K smoke subset, K-parity edge
// shapes, and an NR=64 forced sweep) with the dst axis swapped to the
// FP16 family's {f16, f32}.
// ------------------------------------------------------------------
static std::vector<UkernelCase> make_ukernel_cases() {
    struct Shape {
        int M, K, N;
    };
    const Shape small_shapes[] = {
            {1, 64, 256}, // tiny -- exercise MR=1 single-row path
            {4, 64, 256}, // mini decode
            {16, 256, 512}, // mid decode (MR fan-out + N-tile splits)
            {16, 64, 512}, // N % 64 == 0 -- runs as NR=32 by default
            {8, 128, 256}, // multi-MR per-call partition
    };
    const Shape large_shapes[] = {
            {4, 1024, 2048}, // wide-N moderate-K (N=2*K)
            {4, 1024, 768}, // narrow-N moderate-K (N=0.75*K)
    };
    const Shape edge_shapes[] = {
            {16, 63, 64}, // odd K + smallest N
            {4, 65, 128}, // K = even+1, N = 4 * pack_nr / 2
            {16, 256, 64}, // smallest N at pack_nr=32
    };

    std::vector<UkernelCase> cases;
    // 5 small_shapes × 14 (act × bias × dst over bias∈{none,bf16,f32,f16},
    //   (gated+f32) and (silu/gelu+bias) filtered: none=8, swiglu=4,
    //   silu=1, gelu=1)                                               = 70
    // 2 large_shapes × 5 smoke cases each                             = 10
    // 3 edge_shapes × 4                                               = 12
    // 2 nr64_shapes × 5 NR=64-pinned tuples each                      = 10
    // Total                                                           ~102
    cases.reserve(5 * 14 + 2 * 5 + 3 * 4 + 2 * 5);

    // small_shapes: full (act x bias x dst) cross-product, with the
    // structurally-invalid (gated, f32-dst) tuples filtered (every fused
    // gated kind writes f16 only) and the split-halves-with-bias tuples
    // filtered (silu/gelu fused path is bias-free).
    for (const auto &s : small_shapes) {
        for (auto act : {grp_matmul_gated_act_t::none,
                     grp_matmul_gated_act_t::swiglu_oai_mul,
                     grp_matmul_gated_act_t::silu_and_mul,
                     grp_matmul_gated_act_t::gelu_and_mul}) {
            for (auto bias : {data_type_t::none, data_type_t::bf16,
                         data_type_t::f32, data_type_t::f16}) {
                for (auto dst : {data_type_t::f16, data_type_t::f32}) {
                    const bool is_gated_act
                            = (act == grp_matmul_gated_act_t::swiglu_oai_mul)
                            || (act == grp_matmul_gated_act_t::silu_and_mul)
                            || (act == grp_matmul_gated_act_t::gelu_and_mul);
                    if (is_gated_act && dst == data_type_t::f32) { continue; }
                    const bool is_split_halves_fused
                            = (act == grp_matmul_gated_act_t::silu_and_mul)
                            || (act == grp_matmul_gated_act_t::gelu_and_mul);
                    if (is_split_halves_fused && bias != data_type_t::none) {
                        continue;
                    }
                    cases.push_back(
                            {s.M, s.K, s.N, act, bias, dst, /*nr_override=*/0,
                                    mk_label(s.M, s.K, s.N, act, bias, dst)});
                }
            }
        }
    }

    // Smoke subset for moderate-K shapes (bounded scalar-reference cost).
    struct SmokeTuple {
        grp_matmul_gated_act_t act;
        data_type_t bias;
        data_type_t dst;
    };
    const SmokeTuple smoke[] = {
            {grp_matmul_gated_act_t::none, data_type_t::none, data_type_t::f16},
            {grp_matmul_gated_act_t::none, data_type_t::none, data_type_t::f32},
            {grp_matmul_gated_act_t::none, data_type_t::bf16, data_type_t::f16},
            {grp_matmul_gated_act_t::none, data_type_t::f16, data_type_t::f16},
            {grp_matmul_gated_act_t::swiglu_oai_mul, data_type_t::none,
                    data_type_t::f16},
    };
    for (const auto &s : large_shapes) {
        for (const auto &t : smoke) {
            cases.push_back(
                    {s.M, s.K, s.N, t.act, t.bias, t.dst, /*nr_override=*/0,
                            mk_label(s.M, s.K, s.N, t.act, t.bias, t.dst)});
        }
    }

    // K-parity / pack-NR-edge cases: act=none x bias in {none, bf16} x
    // dst in {f16, f32}.
    for (const auto &s : edge_shapes) {
        for (auto bias : {data_type_t::none, data_type_t::bf16}) {
            for (auto dst : {data_type_t::f16, data_type_t::f32}) {
                cases.push_back({s.M, s.K, s.N, grp_matmul_gated_act_t::none,
                        bias, dst, /*nr_override=*/0,
                        mk_label(s.M, s.K, s.N, grp_matmul_gated_act_t::none,
                                bias, dst)});
            }
        }
    }

    // NR=64 forced sweep -- pins `ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_NR=64`
    // so the NV=4 microkernel + 64-col store epilogue (incl. the f32-dst
    // NV=4 variant) run against the scalar reference.
    struct NrShape {
        int M, K, N;
    };
    const NrShape nr64_shapes[] = {
            {16, 64, 512}, // small-K, N % 64 == 0 (8 NR=64 tiles)
            {4, 1024, 2048}, // moderate-K, wide-N (32 NR=64 tiles)
    };
    struct NrTuple {
        grp_matmul_gated_act_t act;
        data_type_t bias;
        data_type_t dst;
    };
    const NrTuple nr64_tuples[] = {
            {grp_matmul_gated_act_t::none, data_type_t::none, data_type_t::f16},
            {grp_matmul_gated_act_t::none, data_type_t::none, data_type_t::f32},
            {grp_matmul_gated_act_t::none, data_type_t::bf16, data_type_t::f16},
            {grp_matmul_gated_act_t::none, data_type_t::f16, data_type_t::f16},
            {grp_matmul_gated_act_t::swiglu_oai_mul, data_type_t::none,
                    data_type_t::f16},
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

INSTANTIATE_TEST_SUITE_P(ShapeMatrix, CkF16UkernelCorrectness,
        ::testing::ValuesIn([]() -> const std::vector<UkernelCase> & {
            static const std::vector<UkernelCase> kCases = make_ukernel_cases();
            return kCases;
        }()),
        [](const ::testing::TestParamInfo<UkernelCase> &info) {
            return info.param.label;
        });

// ------------------------------------------------------------------
// Strict single-shape engagement gate -- the regression net for "did
// the FP16 CK actually run?".  Pins a shape known to engage
// `dispatch_tile` at any reasonable thread count and asserts the
// resolved gemm_mode carries the `_custom` suffix.  Sibling of the
// bf16 `CkUkernelEngages.OnCanonicalShape`.
// ------------------------------------------------------------------
TEST(CkF16UkernelEngages, OnCanonicalShape) {
    CK_SKIP_IF_NO_F16_ISA();

    mt::AlgoEnvGuard algo_guard(3);
    mt::CustomKernelOverride ck_guard(true);
    mt::CustomKernelF16Override ck_f16_guard(true);
    ::reset_grp_matmul_caches();

    constexpr int kNumOps = 4;
    constexpr int M = 4, K = 1024, N = 2048;

    std::vector<std::vector<float16_t>> src_bufs(kNumOps);
    std::vector<std::vector<float16_t>> wei_bufs(kNumOps);
    std::vector<std::vector<float16_t>> dst_bufs(kNumOps);
    for (int e = 0; e < kNumOps; ++e) {
        src_bufs[e].assign(static_cast<size_t>(M) * K, float16_t(0.0f));
        wei_bufs[e].assign(static_cast<size_t>(K) * N, float16_t(0.0f));
        dst_bufs[e].assign(static_cast<size_t>(M) * N, float16_t(0.0f));
        mt::fill_src(src_bufs[e], /*e=*/e, 0.02f);
        mt::fill_wei1(wei_bufs[e], /*e=*/e, 0.005f);
    }

    std::vector<char> layout(kNumOps, 'r');
    std::vector<bool> transA(kNumOps, false), transB(kNumOps, false);
    std::vector<int> Ms(kNumOps, M), Ns(kNumOps, N), Ks(kNumOps, K);
    std::vector<float> alpha(kNumOps, 1.0f), beta(kNumOps, 0.0f);
    std::vector<int> lda(kNumOps, K), ldb(kNumOps, N), ldc(kNumOps, N);
    std::vector<const void *> src_ptrs(kNumOps), wei_ptrs(kNumOps);
    std::vector<const void *> bias_ptrs(kNumOps, nullptr);
    std::vector<void *> dst_ptrs(kNumOps);
    for (int e = 0; e < kNumOps; ++e) {
        src_ptrs[e] = src_bufs[e].data();
        wei_ptrs[e] = wei_bufs[e].data();
        dst_ptrs[e] = dst_bufs[e].data();
    }
    std::vector<bool> is_wc(kNumOps, true);

    std::vector<mt::matmul_params> params(kNumOps);
    for (auto &p : params) {
        p.dtypes.src = data_type_t::f16;
        p.dtypes.wei = data_type_t::f16;
        p.dtypes.dst = data_type_t::f16;
        p.dtypes.bias = data_type_t::none;
        p.num_threads = kCkTestThreads;
    }

    moe_test_utils::GemmModeCaptureGuard gemm_mode_guard;

    const auto status = group_matmul_direct(layout, transA, transB, Ms, Ns, Ks,
            alpha, src_ptrs, lda, wei_ptrs, ldb, bias_ptrs, beta, dst_ptrs, ldc,
            is_wc, params,
            /*moe_postop=*/nullptr, /*gated_act=*/nullptr);
    ASSERT_EQ(status, status_t::success);

    const char *mode = zendnnl::lowoha::matmul::test_api ::
                               s_last_group_matmul_direct_gemm_mode.load(
                                       std::memory_order_relaxed);
    ASSERT_NE(mode, nullptr)
            << "group_matmul_direct did not publish a gemm_mode";
    ASSERT_NE(std::strstr(mode, "_custom"), nullptr)
            << "Canonical engagement shape (M=4 x K=1024 x N=2048, 4 experts) "
               "ran on '"
            << mode
            << "' instead of the FP16 microkernel.  "
               "Either the planner now refuses CK on production-MoE shapes, "
               "the dispatcher fell back to AOCL DLP, or the FP16 ISA gate / "
               "custom-kernel override regressed.  This is the strict FP16 CK "
               "engagement gate -- investigate before merging.";
}

// ------------------------------------------------------------------
// Strict engagement gate -- silu_and_mul on the canonical fused-CK
// shape.  Sibling of the bf16 `OnCanonicalShapeSiluFused`.  Expected
// gemm_mode: `flat_n_tile_fused_silu_and_mul_tight_custom`.
// ------------------------------------------------------------------
TEST(CkF16UkernelEngages, OnCanonicalShapeSiluFused) {
    CK_SKIP_IF_NO_F16_ISA();

    mt::AlgoEnvGuard algo_guard(3);
    mt::CustomKernelOverride ck_guard(true);
    mt::CustomKernelF16Override ck_f16_guard(true);
    ::reset_grp_matmul_caches();

    constexpr int kNumOps = 4;
    constexpr int M = 4, K = 1024, N = 2048;
    constexpr int I = N / 2;

    std::vector<std::vector<float16_t>> src_bufs(kNumOps);
    std::vector<std::vector<float16_t>> wei_bufs(kNumOps);
    std::vector<std::vector<float16_t>> dst_bufs(kNumOps);
    for (int e = 0; e < kNumOps; ++e) {
        src_bufs[e].assign(static_cast<size_t>(M) * K, float16_t(0.0f));
        wei_bufs[e].assign(static_cast<size_t>(K) * N, float16_t(0.0f));
        dst_bufs[e].assign(static_cast<size_t>(M) * I, float16_t(0.0f));
        mt::fill_src(src_bufs[e], /*e=*/e, 0.02f);
        mt::fill_wei1(wei_bufs[e], /*e=*/e, 0.005f);
    }

    std::vector<char> layout(kNumOps, 'r');
    std::vector<bool> transA(kNumOps, false), transB(kNumOps, false);
    std::vector<int> Ms(kNumOps, M), Ns(kNumOps, N), Ks(kNumOps, K);
    std::vector<float> alpha(kNumOps, 1.0f), beta(kNumOps, 0.0f);
    std::vector<int> lda(kNumOps, K), ldb(kNumOps, N);
    std::vector<int> ldc(kNumOps, I); // tight -> fused tight epilogue
    std::vector<const void *> src_ptrs(kNumOps), wei_ptrs(kNumOps);
    std::vector<const void *> bias_ptrs(kNumOps, nullptr);
    std::vector<void *> dst_ptrs(kNumOps);
    for (int e = 0; e < kNumOps; ++e) {
        src_ptrs[e] = src_bufs[e].data();
        wei_ptrs[e] = wei_bufs[e].data();
        dst_ptrs[e] = dst_bufs[e].data();
    }
    std::vector<bool> is_wc(kNumOps, true);

    std::vector<mt::matmul_params> params(kNumOps);
    for (auto &p : params) {
        p.dtypes.src = data_type_t::f16;
        p.dtypes.wei = data_type_t::f16;
        p.dtypes.dst = data_type_t::f16;
        p.dtypes.bias = data_type_t::none;
        p.num_threads = kCkTestThreads;
    }

    zendnnl::lowoha::matmul::grp_matmul_gated_act_params act_params {};
    act_params.act = grp_matmul_gated_act_t::silu_and_mul;

    moe_test_utils::GemmModeCaptureGuard gemm_mode_guard;

    const auto status = group_matmul_direct(layout, transA, transB, Ms, Ns, Ks,
            alpha, src_ptrs, lda, wei_ptrs, ldb, bias_ptrs, beta, dst_ptrs, ldc,
            is_wc, params,
            /*moe_postop=*/nullptr, &act_params);
    ASSERT_EQ(status, status_t::success);

    const char *mode = zendnnl::lowoha::matmul::test_api ::
                               s_last_group_matmul_direct_gemm_mode.load(
                                       std::memory_order_relaxed);
    ASSERT_NE(mode, nullptr)
            << "group_matmul_direct did not publish a gemm_mode for the "
               "silu_and_mul fused-CK engagement check.";
    EXPECT_NE(std::strstr(mode, "_custom"), nullptr)
            << "silu_and_mul + tight dst ran on '" << mode
            << "' instead of the FP16 microkernel.  Either prepack rejected "
               "the interleave, the dispatcher's silu acceptance gate "
               "refused, the planner's a3_can_fuse_act stopped advertising "
               "silu fused, or the canonical shape moved out of the per-tile "
               "dispatch envelope.  Investigate.";
    EXPECT_NE(std::strstr(mode, "silu_and_mul"), nullptr)
            << "fused silu_and_mul ran but `gemm_mode` lacks the "
               "`silu_and_mul` label fragment (`"
            << mode << "`).";
}

// ------------------------------------------------------------------
// Strict engagement gate -- gelu_and_mul on the canonical fused-CK
// shape.  Sibling of the bf16 `OnCanonicalShapeGeluFused`.  Expected
// gemm_mode: `flat_n_tile_fused_gelu_and_mul_tight_custom`.
// ------------------------------------------------------------------
TEST(CkF16UkernelEngages, OnCanonicalShapeGeluFused) {
    CK_SKIP_IF_NO_F16_ISA();

    mt::AlgoEnvGuard algo_guard(3);
    mt::CustomKernelOverride ck_guard(true);
    mt::CustomKernelF16Override ck_f16_guard(true);
    ::reset_grp_matmul_caches();

    constexpr int kNumOps = 4;
    constexpr int M = 4, K = 1024, N = 2048;
    constexpr int I = N / 2;

    std::vector<std::vector<float16_t>> src_bufs(kNumOps);
    std::vector<std::vector<float16_t>> wei_bufs(kNumOps);
    std::vector<std::vector<float16_t>> dst_bufs(kNumOps);
    for (int e = 0; e < kNumOps; ++e) {
        src_bufs[e].assign(static_cast<size_t>(M) * K, float16_t(0.0f));
        wei_bufs[e].assign(static_cast<size_t>(K) * N, float16_t(0.0f));
        dst_bufs[e].assign(static_cast<size_t>(M) * I, float16_t(0.0f));
        mt::fill_src(src_bufs[e], /*e=*/e, 0.02f);
        mt::fill_wei1(wei_bufs[e], /*e=*/e, 0.005f);
    }

    std::vector<char> layout(kNumOps, 'r');
    std::vector<bool> transA(kNumOps, false), transB(kNumOps, false);
    std::vector<int> Ms(kNumOps, M), Ns(kNumOps, N), Ks(kNumOps, K);
    std::vector<float> alpha(kNumOps, 1.0f), beta(kNumOps, 0.0f);
    std::vector<int> lda(kNumOps, K), ldb(kNumOps, N);
    std::vector<int> ldc(kNumOps, I);
    std::vector<const void *> src_ptrs(kNumOps), wei_ptrs(kNumOps);
    std::vector<const void *> bias_ptrs(kNumOps, nullptr);
    std::vector<void *> dst_ptrs(kNumOps);
    for (int e = 0; e < kNumOps; ++e) {
        src_ptrs[e] = src_bufs[e].data();
        wei_ptrs[e] = wei_bufs[e].data();
        dst_ptrs[e] = dst_bufs[e].data();
    }
    std::vector<bool> is_wc(kNumOps, true);

    std::vector<mt::matmul_params> params(kNumOps);
    for (auto &p : params) {
        p.dtypes.src = data_type_t::f16;
        p.dtypes.wei = data_type_t::f16;
        p.dtypes.dst = data_type_t::f16;
        p.dtypes.bias = data_type_t::none;
        p.num_threads = kCkTestThreads;
    }

    zendnnl::lowoha::matmul::grp_matmul_gated_act_params act_params {};
    act_params.act = grp_matmul_gated_act_t::gelu_and_mul;

    moe_test_utils::GemmModeCaptureGuard gemm_mode_guard;

    const auto status = group_matmul_direct(layout, transA, transB, Ms, Ns, Ks,
            alpha, src_ptrs, lda, wei_ptrs, ldb, bias_ptrs, beta, dst_ptrs, ldc,
            is_wc, params,
            /*moe_postop=*/nullptr, &act_params);
    ASSERT_EQ(status, status_t::success);

    const char *mode = zendnnl::lowoha::matmul::test_api ::
                               s_last_group_matmul_direct_gemm_mode.load(
                                       std::memory_order_relaxed);
    ASSERT_NE(mode, nullptr)
            << "group_matmul_direct did not publish a gemm_mode for the "
               "gelu_and_mul fused-CK engagement check.";
    EXPECT_NE(std::strstr(mode, "_custom"), nullptr)
            << "gelu_and_mul + tight dst ran on '" << mode
            << "' instead of the FP16 microkernel.  Investigate (prepack "
               "interleave, gelu acceptance gate, a3_can_fuse_act, or the "
               "per-tile dispatch envelope).";
    EXPECT_NE(std::strstr(mode, "gelu_and_mul"), nullptr)
            << "fused gelu_and_mul ran but `gemm_mode` lacks the "
               "`gelu_and_mul` label fragment (`"
            << mode << "`).";
}

// ------------------------------------------------------------------
// CK-REFUSAL TIGHT FALLBACK regression -- silu_and_mul / gelu_and_mul.
// Sibling of the bf16 `CkUkernelFallback.Tight*WithBiasRoutesToSequential`.
// When a caller passes tight dst (ldc < N) with a gated split-halves
// activation AND a refusal trigger (+bias), `flat_n_tile` MUST reroute
// to the Sequential strategy (matmul wide -> apply_gated_act_inplace
// (f16) -> memcpy I cols into tight dst), NOT silently apply swiglu
// math to silu/gelu data.
// ------------------------------------------------------------------
void RunTightCkRefusalTest(grp_matmul_gated_act_t act, const char *case_label) {
    mt::AlgoEnvGuard algo_guard(3);
    mt::CustomKernelOverride ck_guard(true);
    mt::CustomKernelF16Override ck_f16_guard(true);
    ::reset_grp_matmul_caches();

    constexpr int kNumOps = 4;
    constexpr int M = 4, K = 1024, N = 2048;
    constexpr int I = N / 2;

    std::vector<std::vector<float16_t>> src_bufs(kNumOps);
    std::vector<std::vector<float16_t>> wei_bufs(kNumOps);
    std::vector<std::vector<mt::bfloat16_t>> bias_bufs(kNumOps);
    std::vector<std::vector<float16_t>> dst_bufs(kNumOps);
    for (int e = 0; e < kNumOps; ++e) {
        src_bufs[e].assign(static_cast<size_t>(M) * K, float16_t(0.0f));
        wei_bufs[e].assign(static_cast<size_t>(K) * N, float16_t(0.0f));
        bias_bufs[e].assign(N, mt::bfloat16_t(0.0f));
        dst_bufs[e].assign(static_cast<size_t>(M) * I, float16_t(0.0f));
        mt::fill_src(src_bufs[e], /*e=*/e, 0.02f);
        mt::fill_wei1(wei_bufs[e], /*e=*/e, 0.005f);
        for (int n = 0; n < N; ++n) {
            bias_bufs[e][n] = mt::bfloat16_t(
                    0.0005f * static_cast<float>((n + e * 7) % 13 - 6));
        }
    }

    std::vector<char> layout(kNumOps, 'r');
    std::vector<bool> transA(kNumOps, false), transB(kNumOps, false);
    std::vector<int> Ms(kNumOps, M), Ns(kNumOps, N), Ks(kNumOps, K);
    std::vector<float> alpha(kNumOps, 1.0f), beta(kNumOps, 0.0f);
    std::vector<int> lda(kNumOps, K), ldb(kNumOps, N);
    std::vector<int> ldc(kNumOps, I); // tight
    std::vector<const void *> src_ptrs(kNumOps), wei_ptrs(kNumOps),
            bias_ptrs(kNumOps);
    std::vector<void *> dst_ptrs(kNumOps);
    for (int e = 0; e < kNumOps; ++e) {
        src_ptrs[e] = src_bufs[e].data();
        wei_ptrs[e] = wei_bufs[e].data();
        bias_ptrs[e] = bias_bufs[e].data();
        dst_ptrs[e] = dst_bufs[e].data();
    }
    std::vector<bool> is_wc(kNumOps, true);

    std::vector<mt::matmul_params> params(kNumOps);
    for (auto &p : params) {
        p.dtypes.src = data_type_t::f16;
        p.dtypes.wei = data_type_t::f16;
        p.dtypes.dst = data_type_t::f16;
        p.dtypes.bias = data_type_t::bf16; // bf16 bias -> CK refuses fused
        p.num_threads = kCkTestThreads;
    }

    zendnnl::lowoha::matmul::grp_matmul_gated_act_params act_params {};
    act_params.act = act;

    moe_test_utils::GemmModeCaptureGuard gemm_mode_guard;

    const auto status = group_matmul_direct(layout, transA, transB, Ms, Ns, Ks,
            alpha, src_ptrs, lda, wei_ptrs, ldb, bias_ptrs, beta, dst_ptrs, ldc,
            is_wc, params,
            /*moe_postop=*/nullptr, &act_params);
    ASSERT_EQ(status, status_t::success)
            << case_label << ": tight + " << case_label
            << " + bias must succeed (graceful Sequential fallback, not "
               "a hard refusal).";

    const char *mode = zendnnl::lowoha::matmul::test_api ::
                               s_last_group_matmul_direct_gemm_mode.load(
                                       std::memory_order_relaxed);
    ASSERT_NE(mode, nullptr) << case_label
                             << ": group_matmul_direct did not publish a "
                                "gemm_mode.";
    EXPECT_NE(std::strstr(mode, "sequential"), nullptr)
            << case_label << ": tight + " << case_label << " + bias ran on '"
            << mode
            << "' instead of Sequential.  "
               "If this is `flat_n_tile_fused_*_tight_custom` the CK bias "
               "gate regressed; if it is `flat_n_tile_fused_swiglu_oai_*` "
               "the silent-wrong-activation bug is back.";
    EXPECT_EQ(std::strstr(mode, "swiglu_oai"), nullptr)
            << case_label << ": tight + " << case_label
            << " + bias selected the swiglu_oai code path (`" << mode
            << "`) -- this is the silent-wrong-activation regression.";

    // Per-element numerics -- match the scalar reference (bias inside the
    // matmul, then silu / gelu).  f16 tolerance band (native-FP16
    // accumulate divergence), expert 0 only.
    const auto tol = mt::tol_act(data_type_t::f16);
    const auto *dst_actual = dst_bufs[0].data();
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < I; ++n) {
            const float ref = ref_gemm_act(m, n, K, /*N_post=*/I,
                    src_bufs[0].data(), K, wei_bufs[0].data(), N,
                    bias_bufs[0].data(), data_type_t::bf16, act);
            const float got = to_f32(dst_actual[m * I + n]);
            const float bound = std::abs(ref) * tol.rel + tol.abs;
            EXPECT_NEAR(got, ref, bound)
                    << case_label << " m=" << m << " n=" << n << " ref=" << ref
                    << " got=" << got;
        }
    }
}

TEST(CkF16UkernelFallback, TightSiluWithBiasRoutesToSequential) {
    CK_SKIP_IF_NO_F16_ISA();
    RunTightCkRefusalTest(grp_matmul_gated_act_t::silu_and_mul, "silu");
}

TEST(CkF16UkernelFallback, TightGeluWithBiasRoutesToSequential) {
    CK_SKIP_IF_NO_F16_ISA();
    RunTightCkRefusalTest(grp_matmul_gated_act_t::gelu_and_mul, "gelu");
}

// ------------------------------------------------------------------
// Focused parity test for the vectorized `gelu_avx512` polynomial
// fallback on the FP16 dtype.  Sibling of the bf16
// `CkUkernelFallback.VectorizedGeluAvx512MatchesErfReference`.  Drives
// `apply_gated_act_inplace(gelu_and_mul, ..., data_type_t::f16)`
// directly on a deterministic F16 buffer -- exercising
// `gelu_and_mul_row_avx512_f16` (group_matmul_moe_act.cpp) -- and
// compares against a per-lane `std::erf`-based reference (the same
// `gelu_erf` math the shared `gelu_avx512` approximates).
//
// Pins:
//   * The polynomial form's max delta vs `gelu_erf` stays within the
//     `tol_act(data_type_t::f16)` band across the F16 input range.
//   * The fallback applies the activation in-place: gate cols [0, I)
//     are overwritten with `gelu(gate) * up`; up cols [I, N) are left
//     as-is (garbage per the public-API contract).
//
// This pins the non-CK / fallback side of the FP16 gelu path
// independently of the CK fused gelu path (which is covered by
// `OnCanonicalShapeGeluFused`).
// ------------------------------------------------------------------
TEST(CkF16UkernelFallback, VectorizedGeluAvx512MatchesErfReference) {
    CK_SKIP_IF_NO_F16_ISA();

    // Sweep inputs across the realistic gelu range, including the
    // steepest-curvature boundary values (|x| ~ 0.5..1.5) where the
    // polynomial-vs-erf delta matters most.
    constexpr int M = 4;
    constexpr int I = 256;
    constexpr int N = 2 * I;

    // Deterministic gate/up values: gate sweeps [-4, +4] across cols,
    // up sweeps [-1, +1] -- both well inside the F16 representable range.
    std::vector<float16_t> buf(static_cast<size_t>(M) * N, float16_t(0.0f));
    for (int m = 0; m < M; ++m) {
        for (int i = 0; i < I; ++i) {
            const float g = -4.0f
                    + 8.0f * static_cast<float>((m * I + i) % I)
                            / static_cast<float>(I - 1);
            const float u = -1.0f
                    + 2.0f * static_cast<float>((m * I + i + 17) % I)
                            / static_cast<float>(I - 1);
            buf[m * N + i] = float16_t(g);
            buf[m * N + i + I] = float16_t(u);
        }
    }

    // Snapshot input for the reference computation BEFORE
    // apply_gated_act_inplace overwrites the gate cols.
    std::vector<float> gate_ref(static_cast<size_t>(M) * I);
    std::vector<float> up_ref(static_cast<size_t>(M) * I);
    for (int m = 0; m < M; ++m) {
        for (int i = 0; i < I; ++i) {
            gate_ref[m * I + i] = to_f32(buf[m * N + i]);
            up_ref[m * I + i] = to_f32(buf[m * N + i + I]);
        }
    }

    // Drive the fallback path directly.  apply_gated_act_inplace's
    // AVX-512 branch dispatches to `gelu_and_mul_row_avx512_f16`, which
    // uses the vectorized polynomial `gelu_avx512`.
    zendnnl::lowoha::matmul::apply_gated_act_inplace(
            grp_matmul_gated_act_t::gelu_and_mul, buf.data(),
            /*row_start=*/0, /*row_end=*/M,
            /*N=*/N, /*ldc=*/N, data_type_t::f16);

    // Per-lane scalar `gelu_erf` reference -- any vectorisation must stay
    // within the f16 tolerance band of this (which also absorbs the
    // F16 round-trip on gate/up and the stored result).
    const float kSqrtHalf = 0.7071067811865476f;
    const auto tol = mt::tol_act(data_type_t::f16);
    for (int m = 0; m < M; ++m) {
        for (int i = 0; i < I; ++i) {
            const float g = gate_ref[m * I + i];
            const float u = up_ref[m * I + i];
            const float gelu_erf = g * 0.5f * (1.0f + std::erf(g * kSqrtHalf));
            const float ref = gelu_erf * u;
            const float got = to_f32(buf[m * N + i]);
            const float bound = std::abs(ref) * tol.rel + tol.abs;
            EXPECT_NEAR(got, ref, bound)
                    << "gelu polynomial drifted from gelu_erf reference: "
                    << " m=" << m << " i=" << i << " g=" << g << " u=" << u
                    << " ref=" << ref << " got=" << got;
        }
    }
}

} // namespace
