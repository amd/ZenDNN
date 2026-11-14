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

/// Gated-activation forms selected by ZENDNNL_GRP_MATMUL_GELU_TANH and
/// ZENDNNL_GRP_MATMUL_FAST_ACT: the shared AVX-512 math against double
/// references, the knobs' defaults, the separate-pass fallback, and the
/// custom kernel of every dtype, single and multi expert, against that
/// fallback under the same knobs.

#include <gtest/gtest.h>

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "common/platform_info.hpp"
#include "common/zendnnl_compat.hpp"
#include "gtest_utils.hpp"
#include "lowoha_operators/matmul/group_matmul/common/group_matmul_act_avx512.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/dispatch.hpp"
#include "lowoha_operators/matmul/group_matmul/group_matmul_direct.hpp"
#include "lowoha_operators/matmul/lowoha_matmul.hpp"
#include "moe_test_utils.hpp"

namespace {

namespace mm = zendnnl::lowoha::matmul;
namespace act = zendnnl::lowoha::matmul::group_matmul_act_avx512;
namespace mt = moe_test_utils;
using zendnnl::common::bfloat16_t;
using zendnnl::common::float16_t;
using zendnnl::memory::data_type_t;
using zendnnl::memory::status_t;

/// Pins both form knobs for a scope.
struct ActFormOverride {
    int prev_tanh;
    int prev_fast;
    ActFormOverride(int gelu_tanh, int fast) {
        prev_tanh = mm::test_api::s_grp_matmul_gelu_tanh_override.exchange(
                gelu_tanh);
        prev_fast = mm::test_api::s_grp_matmul_fast_act_override.exchange(fast);
    }
    ~ActFormOverride() {
        mm::test_api::s_grp_matmul_gelu_tanh_override.store(prev_tanh);
        mm::test_api::s_grp_matmul_fast_act_override.store(prev_fast);
    }
    ActFormOverride(const ActFormOverride &) = delete;
    ActFormOverride &operator=(const ActFormOverride &) = delete;
};

bool has_avx512() {
    return zendnnl::common::zendnnl_platform_info().get_avx512f_status();
}

#define SKIP_WITHOUT_AVX512() \
    if (!has_avx512()) GTEST_SKIP() << "needs AVX-512"

double gelu_erf_ref(double x) {
    return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
}
double gelu_tanh_ref(double x) {
    constexpr double kPi = 3.14159265358979323846;
    return 0.5 * x
            * (1.0
                    + std::tanh(
                            std::sqrt(2.0 / kPi) * (x + 0.044715 * x * x * x)));
}
double sigmoid_ref(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}
double swiglu_oai_ref(double g, double u) {
    g = std::clamp(g, -7.0, 7.0);
    u = std::clamp(u, -7.0, 7.0);
    return (1.0 + u) * g * sigmoid_ref(1.702 * g);
}

enum class Fn { kSigmoid, kSilu, kGelu };

ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,avx512dq,fma")
void eval_act(Fn fn, const float *x, float *y, int n, act::act_mode_t m) {
    for (int i = 0; i < n; i += 16) {
        const __m512 v = _mm512_loadu_ps(x + i);
        const __m512 r = fn == Fn::kSigmoid ? act::sigmoid_act_avx512(v, m)
                : fn == Fn::kSilu           ? act::silu_act_avx512(v, m)
                                            : act::gelu_act_avx512(v, m);
        _mm512_storeu_ps(y + i, r);
    }
}

ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,avx512dq,fma")
void eval_legacy(Fn fn, const float *x, float *y, int n) {
    for (int i = 0; i < n; i += 16) {
        const __m512 v = _mm512_loadu_ps(x + i);
        const __m512 r = fn == Fn::kSigmoid ? act::sigmoid_avx512(v)
                : fn == Fn::kSilu           ? act::silu_avx512(v)
                                            : act::gelu_avx512(v);
        _mm512_storeu_ps(y + i, r);
    }
}

ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,avx512dq,fma")
void eval_swiglu(const float *g, const float *u, float *y, int n,
        act::act_mode_t m, bool legacy) {
    for (int i = 0; i < n; i += 16) {
        const __m512 vg = _mm512_loadu_ps(g + i);
        const __m512 vu = _mm512_loadu_ps(u + i);
        _mm512_storeu_ps(y + i,
                legacy ? act::swiglu_oai_avx512(vg, vu)
                       : act::swiglu_oai_act_avx512(vg, vu, m));
    }
}

/// `n` (a multiple of 16) points over [lo, hi] plus zero.
std::vector<float> grid(float lo, float hi, int n) {
    std::vector<float> x(n);
    for (int i = 0; i < n; ++i)
        x[i] = lo + (hi - lo) * static_cast<float>(i) / (n - 1);
    x[n / 2] = 0.0f;
    return x;
}

/// Max of |y - ref| / (rel_floor + |ref|) over the points.
double max_scaled_err(const std::vector<float> &x, const std::vector<float> &y,
        double (*ref)(double), double floor) {
    double worst = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        const double r = ref(x[i]);
        worst = std::max(worst, std::fabs(y[i] - r) / (floor + std::fabs(r)));
    }
    return worst;
}

// ── Knobs ────────────────────────────────────────────────────────────

TEST(GatedActForms, KnobsDefaultOffAndOverridesApply) {
    if (std::getenv("ZENDNNL_GRP_MATMUL_GELU_TANH") == nullptr
            && std::getenv("ZENDNNL_GRP_MATMUL_FAST_ACT") == nullptr) {
        const act::act_mode_t m = act::act_mode();
        EXPECT_FALSE(m.gelu_tanh);
        EXPECT_FALSE(m.fast);
    }
    {
        ActFormOverride o(1, 0);
        EXPECT_TRUE(act::act_mode().gelu_tanh);
        EXPECT_FALSE(act::act_mode().fast);
    }
    {
        ActFormOverride o(0, 1);
        EXPECT_FALSE(act::act_mode().gelu_tanh);
        EXPECT_TRUE(act::act_mode().fast);
    }
}

// ── Shared math ──────────────────────────────────────────────────────

// With both knobs off every `*_act_avx512` form is the existing function,
// bit for bit, so default results do not move.
TEST(GatedActForms, DefaultFormsAreTheExistingOnes) {
    SKIP_WITHOUT_AVX512();
    const std::vector<float> x = grid(-30.0f, 30.0f, 4096);
    std::vector<float> a(x.size()), b(x.size());
    for (const Fn fn : {Fn::kSigmoid, Fn::kSilu, Fn::kGelu}) {
        eval_act(fn, x.data(), a.data(), static_cast<int>(x.size()), {});
        eval_legacy(fn, x.data(), b.data(), static_cast<int>(x.size()));
        EXPECT_EQ(0, std::memcmp(a.data(), b.data(), a.size() * sizeof(float)))
                << static_cast<int>(fn);
    }
    const std::vector<float> u = grid(-9.0f, 9.0f, 4096);
    eval_swiglu(x.data(), u.data(), a.data(), static_cast<int>(x.size()), {},
            false);
    eval_swiglu(
            x.data(), u.data(), b.data(), static_cast<int>(x.size()), {}, true);
    EXPECT_EQ(0, std::memcmp(a.data(), b.data(), a.size() * sizeof(float)));
}

// The fused-MoE references evaluate the activations in the library's own
// forms, so the accuracy of the default forms is pinned here.
TEST(GatedActForms, DefaultFormsTrackExactReferences) {
    SKIP_WITHOUT_AVX512();
    const act::act_mode_t def {};
    const std::vector<float> x = grid(-40.0f, 40.0f, 32000);
    std::vector<float> y(x.size());
    eval_act(Fn::kSigmoid, x.data(), y.data(), static_cast<int>(x.size()), def);
    EXPECT_LT(max_scaled_err(x, y, sigmoid_ref, 0.0), 1e-4);
    eval_act(Fn::kSilu, x.data(), y.data(), static_cast<int>(x.size()), def);
    EXPECT_LT(max_scaled_err(
                      x, y, [](double v) { return v * sigmoid_ref(v); }, 1e-6),
            1e-4);
    const std::vector<float> xg = grid(-12.0f, 12.0f, 24000);
    std::vector<float> yg(xg.size());
    eval_act(Fn::kGelu, xg.data(), yg.data(), static_cast<int>(xg.size()), def);
    EXPECT_LT(max_scaled_err(xg, yg, gelu_erf_ref, 1e-3), 2.5e-4);
    const std::vector<float> g = grid(-9.0f, 9.0f, 4096);
    const std::vector<float> u = grid(9.0f, -9.0f, 4096);
    std::vector<float> ys(g.size());
    eval_swiglu(g.data(), u.data(), ys.data(), static_cast<int>(g.size()), def,
            false);
    double worst = 0.0;
    for (size_t i = 0; i < g.size(); ++i) {
        const double r = swiglu_oai_ref(g[i], u[i]);
        worst = std::max(worst, std::fabs(ys[i] - r) / (1e-3 + std::fabs(r)));
    }
    EXPECT_LT(worst, 1e-4);
}

TEST(GatedActForms, GeluTanhMatchesPyTorchTanhGelu) {
    SKIP_WITHOUT_AVX512();
    const std::vector<float> x = grid(-12.0f, 12.0f, 24000);
    std::vector<float> y(x.size());
    eval_act(Fn::kGelu, x.data(), y.data(), static_cast<int>(x.size()),
            {/*gelu_tanh=*/true, /*fast=*/false});
    EXPECT_LT(max_scaled_err(x, y, gelu_tanh_ref, 1e-3), 1e-4);
    // At x = -3 the two GELUs differ by 10% of their value; the result is
    // the tanh one.
    const float x3[16] = {-3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f,
            -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f};
    float y3[16];
    eval_act(Fn::kGelu, x3, y3, 16, {true, false});
    EXPECT_NEAR(y3[0], gelu_tanh_ref(-3.0), 1e-6);
    EXPECT_GT(std::fabs(y3[0] - gelu_erf_ref(-3.0)), 3e-4);
    // Large and non-finite inputs: GELU tends to x and 0, NaN propagates.
    const float big[16] = {1e30f, -1e30f, 1e6f, -1e6f, 88.0f, -88.0f, 200.0f,
            -200.0f, INFINITY, NAN, 1.0f, -1.0f, 0.0f, -0.0f, 5.0f, -5.0f};
    float yb[16];
    eval_act(Fn::kGelu, big, yb, 16, {true, false});
    EXPECT_FLOAT_EQ(yb[0], 1e30f);
    EXPECT_NEAR(yb[1], 0.0f, 1e-6);
    EXPECT_FLOAT_EQ(yb[2], 1e6f);
    EXPECT_NEAR(yb[3], 0.0f, 1e-6);
    EXPECT_TRUE(std::isinf(yb[8]) && yb[8] > 0);
    EXPECT_TRUE(std::isnan(yb[9]));
}

TEST(GatedActForms, FastFormsStayWithinTheirBounds) {
    SKIP_WITHOUT_AVX512();
    const act::act_mode_t fast {/*gelu_tanh=*/false, /*fast=*/true};
    const std::vector<float> x = grid(-40.0f, 40.0f, 32000);
    std::vector<float> y(x.size());
    eval_act(
            Fn::kSigmoid, x.data(), y.data(), static_cast<int>(x.size()), fast);
    EXPECT_LT(max_scaled_err(x, y, sigmoid_ref, 0.0), 3e-4);
    eval_act(Fn::kSilu, x.data(), y.data(), static_cast<int>(x.size()), fast);
    EXPECT_LT(max_scaled_err(
                      x, y, [](double v) { return v * sigmoid_ref(v); }, 1e-6),
            3e-4);
    // The fast GELU is the tanh form, whichever GELU is selected.
    const std::vector<float> xg = grid(-12.0f, 12.0f, 24000);
    std::vector<float> yg(xg.size());
    for (const bool tanh_sel : {false, true}) {
        eval_act(Fn::kGelu, xg.data(), yg.data(), static_cast<int>(xg.size()),
                {tanh_sel, true});
        EXPECT_LT(max_scaled_err(xg, yg, gelu_tanh_ref, 1e-3), 3e-4);
    }
    const std::vector<float> g = grid(-9.0f, 9.0f, 4096);
    const std::vector<float> u = grid(9.0f, -9.0f, 4096);
    std::vector<float> ys(g.size());
    eval_swiglu(g.data(), u.data(), ys.data(), static_cast<int>(g.size()), fast,
            false);
    double worst = 0.0;
    for (size_t i = 0; i < g.size(); ++i) {
        const double r = swiglu_oai_ref(g[i], u[i]);
        worst = std::max(worst, std::fabs(ys[i] - r) / (1e-3 + std::fabs(r)));
    }
    EXPECT_LT(worst, 3e-4);
}

// Gates far below zero, where every activation underflows, give a
// negligible, finite value in every form (the tanh forms clamp their
// sigmoid argument and the fast exp is capped, so no reciprocal goes
// subnormal), and a NaN gate stays NaN through SiLU and GELU.
TEST(GatedActForms, LargeNegativeGatesAndNaN) {
    SKIP_WITHOUT_AVX512();
    std::vector<float> x = grid(-1.0e4f, -100.0f, 1024);
    x[x.size() / 2] = -100.0f; // grid() puts a zero there
    std::vector<float> y(x.size());
    const std::vector<float> nan(16, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> y_nan(16);
    for (const act::act_mode_t m :
            {act::act_mode_t {false, false}, act::act_mode_t {true, false},
                    act::act_mode_t {false, true}}) {
        SCOPED_TRACE("gelu_tanh=" + std::to_string(m.gelu_tanh)
                + " fast=" + std::to_string(m.fast));
        for (const Fn fn : {Fn::kSigmoid, Fn::kSilu, Fn::kGelu}) {
            eval_act(fn, x.data(), y.data(), static_cast<int>(x.size()), m);
            for (size_t i = 0; i < x.size(); ++i) {
                ASSERT_TRUE(std::isfinite(y[i])) << "x=" << x[i];
                ASSERT_LE(std::fabs(y[i]), 1e-30f) << "x=" << x[i];
            }
            if (fn == Fn::kSigmoid) continue;
            eval_act(fn, nan.data(), y_nan.data(), 16, m);
            for (const float v : y_nan)
                EXPECT_TRUE(std::isnan(v));
        }
    }
}

// ── Separate-pass fallback ───────────────────────────────────────────

TEST(GatedActForms, SeparatePassFollowsTheKnobs) {
    constexpr int M = 3, dim = 70, N = 2 * dim; // 70: a vector body and a tail
    std::vector<float> base(static_cast<size_t>(M) * N);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < dim; ++n) {
            base[m * N + n] = -6.0f + 12.0f * n / (dim - 1) + 0.37f * m;
            base[m * N + dim + n] = 0.5f + 0.01f * n;
        }
    struct Form {
        int tanh, fast;
        double (*ref)(double);
        double tol;
    };
    for (const Form f :
            {Form {0, 0, gelu_erf_ref, 2e-4}, Form {1, 0, gelu_tanh_ref, 2e-4},
                    Form {0, 1, gelu_tanh_ref, 6e-4}}) {
        ActFormOverride o(f.tanh, f.fast);
        std::vector<float> buf = base;
        mm::apply_gated_act_inplace(mm::grp_matmul_gated_act_t::gelu_and_mul,
                buf.data(), 0, M, N, N, data_type_t::f32);
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < dim; ++n) {
                const double want
                        = f.ref(base[m * N + n]) * base[m * N + dim + n];
                EXPECT_NEAR(
                        buf[m * N + n], want, f.tol * (1e-2 + std::fabs(want)))
                        << "tanh=" << f.tanh << " fast=" << f.fast << " m=" << m
                        << " n=" << n;
            }
    }
}

// ── Custom kernel against the fallback, every dtype ──────────────────

enum class Family { kBf16, kInt8, kW4a8, kF16 };

const char *family_name(Family f) {
    switch (f) {
        case Family::kBf16: return "bf16";
        case Family::kInt8: return "int8";
        case Family::kW4a8: return "w4a8";
        case Family::kF16: return "f16";
    }
    return "?";
}

struct CkCase {
    Family family;
    int num_ops;
    int M;
    mm::grp_matmul_gated_act_t act;
};

std::string ck_case_name(const ::testing::TestParamInfo<CkCase> &info) {
    const CkCase &c = info.param;
    return std::string(family_name(c.family)) + "_E" + std::to_string(c.num_ops)
            + "_M" + std::to_string(c.M)
            + (c.act == mm::grp_matmul_gated_act_t::gelu_and_mul ? "_gelu"
                                                                 : "_silu");
}

uint16_t to_bits(float v, bool f16) {
    return f16 ? float16_t::f32_to_f16_val(v)
               : static_cast<uint16_t>(bfloat16_t::f32_to_bf16_val(v));
}

float from_bits(uint16_t bits, bool f16) {
    return f16 ? float16_t::f16_to_f32_val(bits)
               : bfloat16_t::bf16_to_f32_val(static_cast<int16_t>(bits));
}

/// Inputs of one grouped call; experts share nothing.
struct Problem {
    static constexpr int K = 256;
    static constexpr int N = 256; // gate | up, dim 128
    static constexpr int G = K / 128; // W4A8 groups
    Family family;
    int num_ops, M;
    std::vector<std::vector<uint16_t>> src; // bf16 or f16 bits
    std::vector<std::vector<uint8_t>> wei; // bf16/f16 bits, s8 or packed s4
    std::vector<std::vector<float>> wei_scale;

    Problem(Family f, int e, int m) : family(f), num_ops(e), M(m) {
        std::mt19937 rng(0xACE5u + e * 131 + m);
        std::uniform_real_distribution<float> fd(-1.0f, 1.0f);
        std::uniform_int_distribution<int> wd(-127, 127);
        std::uniform_int_distribution<int> bd(0, 255);
        src.resize(e);
        wei.resize(e);
        wei_scale.resize(e);
        const bool f16 = f == Family::kF16;
        for (int x = 0; x < e; ++x) {
            src[x].resize(static_cast<size_t>(M) * K);
            for (auto &v : src[x])
                v = to_bits(fd(rng), f16);
            if (f == Family::kBf16 || f == Family::kF16) {
                wei[x].resize(static_cast<size_t>(N) * K * 2);
                auto *w = reinterpret_cast<uint16_t *>(wei[x].data());
                for (size_t i = 0; i < static_cast<size_t>(N) * K; ++i) {
                    // Spreads the gate halves over roughly [-4, 4], where
                    // tanh and erf GELU visibly differ.
                    w[i] = to_bits(0.25f * fd(rng), f16);
                }
            } else if (f == Family::kInt8) {
                wei[x].resize(static_cast<size_t>(N) * K);
                for (auto &v : wei[x])
                    v = static_cast<uint8_t>(static_cast<int8_t>(wd(rng)));
                wei_scale[x].assign(N, 0.25f / 127.0f);
            } else {
                wei[x].resize(static_cast<size_t>(N) * K / 2);
                for (auto &v : wei[x])
                    v = static_cast<uint8_t>(bd(rng));
                wei_scale[x].assign(static_cast<size_t>(G) * N, 0.25f / 7.0f);
            }
        }
    }

    /// Runs the call and returns each expert's activated [M, N/2] block as
    /// floats, with the path it took.
    std::vector<float> run(mm::grp_matmul_gated_act_t a, std::string &path) {
        const bool f16 = family == Family::kF16;
        const data_type_t act_dt = f16 ? data_type_t::f16 : data_type_t::bf16;
        std::vector<std::vector<uint16_t>> dst(
                num_ops, std::vector<uint16_t>(static_cast<size_t>(M) * N, 0));
        std::vector<std::vector<float>> src_scale(
                num_ops, std::vector<float>(M, 0.0f));
        std::vector<mm::matmul_params> params(num_ops);
        for (int x = 0; x < num_ops; ++x) {
            mm::matmul_params &p = params[x];
            p.dtypes.src = act_dt;
            p.dtypes.dst = act_dt;
            p.num_threads = 8;
            if (family == Family::kBf16 || family == Family::kF16) {
                p.dtypes.wei = act_dt;
                continue;
            }
            p.dtypes.wei = family == Family::kInt8 ? data_type_t::s8
                                                   : data_type_t::s4;
            p.dtypes.compute = data_type_t::s8;
            p.dynamic_quant = true;
            p.quant_params.src_scale.buff = src_scale[x].data();
            p.quant_params.src_scale.dt = data_type_t::f32;
            p.quant_params.src_scale.dims = {M, 1};
            p.quant_params.wei_scale.buff = wei_scale[x].data();
            p.quant_params.wei_scale.dt = data_type_t::f32;
            p.quant_params.wei_scale.dims = family == Family::kInt8
                    ? std::vector<int64_t> {1, N}
                    : std::vector<int64_t> {G, N};
        }
        std::vector<const void *> sp(num_ops), wp(num_ops),
                bp(num_ops, nullptr);
        std::vector<void *> dp(num_ops);
        for (int x = 0; x < num_ops; ++x) {
            sp[x] = src[x].data();
            wp[x] = wei[x].data();
            dp[x] = dst[x].data();
        }
        mm::grp_matmul_gated_act_params ga;
        ga.act = a;
        mt::ExecutionPathCaptureGuard capture;
        const status_t st = mm::group_matmul_direct(
                std::vector<char>(num_ops, 'r'),
                std::vector<bool>(num_ops, false),
                std::vector<bool>(num_ops, true), std::vector<int>(num_ops, M),
                std::vector<int>(num_ops, N), std::vector<int>(num_ops, K),
                std::vector<float>(num_ops, 1.0f), sp,
                std::vector<int>(num_ops, K), wp, std::vector<int>(num_ops, K),
                bp, std::vector<float>(num_ops, 0.0f), dp,
                std::vector<int>(num_ops, N), std::vector<bool>(num_ops, true),
                params, nullptr, &ga);
        EXPECT_EQ(st, status_t::success);
        const char *p
                = mm::test_api::s_last_group_matmul_direct_execution_path.load(
                        std::memory_order_relaxed);
        path = p != nullptr ? p : "";
        std::vector<float> out;
        out.reserve(static_cast<size_t>(num_ops) * M * N / 2);
        for (int x = 0; x < num_ops; ++x)
            for (int m = 0; m < M; ++m)
                for (int n = 0; n < N / 2; ++n) {
                    out.push_back(from_bits(
                            dst[x][static_cast<size_t>(m) * N + n], f16));
                }
        return out;
    }
};

bool family_supported(Family f) {
    namespace ck = zendnnl::lowoha::matmul::custom_kernel;
    if (!has_avx512()) return false;
    switch (f) {
        case Family::kBf16: return ck::bf16_isa_supported();
        case Family::kF16: return ck::avx512f16_available();
        default: return ck::avx512vnni_available();
    }
}

class GatedActFormsCk : public ::testing::TestWithParam<CkCase> {};

// Under each knob setting the custom kernel matches the separate-pass
// fallback, and flipping the knob changes the custom kernel's output: the
// knobs reach every dtype's epilogue, single and multi expert, decode and
// prompt.
TEST_P(GatedActFormsCk, MatchesTheFallbackUnderEachKnob) {
    const CkCase c = GetParam();
    if (!family_supported(c.family))
        GTEST_SKIP() << family_name(c.family) << " custom kernel needs ISA";
    reset_grp_matmul_caches();
    // Out of place: the custom-kernel run must leave the weights for the
    // fallback run.  ALGO 3 runs the custom kernel's fused epilogue for
    // every case, prompts of a single expert included.
    WeightCacheGuard out_of_place(1);
    mt::AlgoEnvGuard algo3(3);
    mt::SingleExpertCkPolicy single(1, 1, 1);
    Problem prob(c.family, c.num_ops, c.M);
    struct Knobs {
        int tanh, fast;
    };
    const bool gelu = c.act == mm::grp_matmul_gated_act_t::gelu_and_mul;
    const Knobs flipped = gelu ? Knobs {1, 0} : Knobs {0, 1};
    std::vector<float> ck_default;
    for (const Knobs k : {Knobs {0, 0}, flipped, Knobs {0, 1}}) {
        SCOPED_TRACE("gelu_tanh=" + std::to_string(k.tanh)
                + " fast=" + std::to_string(k.fast));
        ActFormOverride o(k.tanh, k.fast);
        std::string ck_path, ref_path;
        const std::vector<float> ck = prob.run(c.act, ck_path);
        std::vector<float> ref;
        {
            mt::CustomKernelOverride off(false);
            ref = prob.run(c.act, ref_path);
        }
        EXPECT_NE(ck_path.find("custom"), std::string::npos) << ck_path;
        EXPECT_EQ(ref_path.find("custom"), std::string::npos) << ref_path;
        ASSERT_EQ(ck.size(), ref.size());
        int bad = 0;
        for (size_t i = 0; i < ck.size(); ++i) {
            const float tol = 0.06f + 0.03f * std::fabs(ref[i]);
            if (std::fabs(ck[i] - ref[i]) > tol && bad++ < 5)
                ADD_FAILURE() << "i=" << i << " ck=" << ck[i]
                              << " fallback=" << ref[i];
        }
        EXPECT_EQ(bad, 0);
        if (k.tanh == 0 && k.fast == 0) {
            ck_default = ck;
        } else {
            size_t changed = 0;
            for (size_t i = 0; i < ck.size(); ++i)
                changed += ck[i] != ck_default[i];
            EXPECT_GT(changed, 0u) << "the knob did not reach the kernel";
        }
    }
    reset_grp_matmul_caches();
}

std::vector<CkCase> ck_cases() {
    std::vector<CkCase> v;
    for (const Family f :
            {Family::kBf16, Family::kInt8, Family::kW4a8, Family::kF16})
        for (const int e : {1, 3})
            for (const int m : {4, 96})
                for (const auto a : {mm::grp_matmul_gated_act_t::gelu_and_mul,
                             mm::grp_matmul_gated_act_t::silu_and_mul})
                    v.push_back({f, e, m, a});
    return v;
}

INSTANTIATE_TEST_SUITE_P(AllFamilies, GatedActFormsCk,
        ::testing::ValuesIn(ck_cases()), ck_case_name);

} // namespace
