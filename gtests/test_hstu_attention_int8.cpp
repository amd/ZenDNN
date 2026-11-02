/********************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

/// HSTU attention, dynamic INT8 (is_qk_quant / is_pv_quant): every tiled case
/// in each INT8 mode against the FP64 oracle with the kernel's quantization
/// and against the exact one, edge inputs, determinism, the diagonal term of
/// target rows against the BF16 kernel, scratch reuse with BF16 calls, the
/// flags where INT8 does not run, the qk_dim limit, the environment
/// overrides, and the dynamic quantizers code for code. The INT8 suites all
/// start with HstuAttentionInt8 (the cases run as Cases/HstuAttentionInt8),
/// so --gtest_filter='*HstuAttentionInt8*' runs them;
/// HstuAttentionFastSilu.DefaultPathUnchanged, which covers the BF16 kernel
/// too, runs with the fast silu test of test_hstu_attention.cpp.

#include "hstu_attention_test_utils.hpp"
#include "lowoha_operators/hstu_attention/common/lowoha_hstu_select.hpp"
#include "lowoha_operators/hstu_attention/hstu_quant/lowoha_hstu_quant.hpp"

#include <cfenv>
#include <cstdio>

namespace {

using namespace hstu_test;

/// The dynamic-INT8 modes of the tiled kernel.
struct int8_mode {
    const char *name;
    bool qk, pv;
};
const int8_mode kInt8Modes[]
        = {{"qk", true, false}, {"qk_pv", true, true}, {"pv", false, true}};

/// rel_error without the rounding of a BF16 output (2^-8 |want| per
/// element).
double rel_error_past_output(const std::vector<float> &got,
        const std::vector<double> &want, data_type_t out_dt) {
    const double u = out_dt == data_type_t::bf16 ? 1.0 / 256 : 0.0;
    double err = 0.0, mag = 1e-30;
    for (size_t i = 0; i < want.size(); ++i) {
        if (!std::isfinite(got[i]) || !std::isfinite(want[i]))
            return std::numeric_limits<double>::infinity();
        err = std::max(err, std::abs(got[i] - want[i]) - u * std::abs(want[i]));
        mag = std::max(mag, std::abs(want[i]));
    }
    return err / mag;
}

/// Against the oracle with the kernel's quantization, which catches scale
/// and indexing errors, and against the exact oracle, which bounds the
/// quantization error. Prints both errors. `fast_silu` as in
/// `hstu_problem::run`. The tight bound covers values the two round to
/// neighbouring codes: rounding ties of the quantized and BF16 values, and
/// with the fast silu, P within its error (1.2e-4) of a rounding boundary,
/// which on a 17-key window reaches 2.1e-3.
void check_int8(const hstu_problem &prob, const std::vector<double> &exact,
        const int8_mode &mode, const std::vector<int32_t> &thread_counts,
        int fast_silu = -1) {
    const std::vector<double> want = prob.oracle_int8(mode.qk, mode.pv);
    const bool fast = fast_silu < 0
            ? ha::parse_bool_env(
                      std::getenv("ZENDNNL_HSTU_ATTENTION_FAST_SILU"))
                    == 1
            : fast_silu == 1;
    const double kTight = fast ? 3e-3 : 2e-3, kLoose = 4e-2;
    std::vector<float> got;
    bool canary = false;
    double quant_err = 0.0, exact_err = 0.0;
    // With fast_silu >= 0, run() calls the tiled kernel whatever `kernel`.
    const std::vector<kernel_t> kernels = fast_silu < 0
            ? std::vector<kernel_t> {kernel_t::none, kernel_t::tiled}
            : std::vector<kernel_t> {kernel_t::tiled};
    for (int32_t threads : thread_counts)
        for (kernel_t kernel : kernels) {
            ASSERT_EQ(prob.run(kernel, got, threads, &canary, mode.qk, mode.pv,
                              fast_silu),
                    status_t::success);
            const double qe = rel_error_past_output(got, want, prob.cs.out_dt);
            quant_err = std::max(quant_err, qe);
            exact_err = std::max(exact_err, rel_error(got, exact));
            EXPECT_LT(qe, kTight) << mode.name << ", threads=" << threads;
            EXPECT_LT(rel_error(got, exact), kLoose)
                    << mode.name << ", threads=" << threads;
            EXPECT_TRUE(canary) << mode.name << " wrote outside its rows";
        }
    std::printf(
            "[   INT8   ] %-36s %-5s vs quantized oracle %.2e, vs exact "
            "%.2e\n",
            prob.cs.name.c_str(), mode.name, quant_err, exact_err);
}

class HstuAttentionInt8 : public ::testing::TestWithParam<hstu_case> {};

TEST_P(HstuAttentionInt8, MatchOracles) {
    if (env_overrides_kernel()) GTEST_SKIP();
    if (!has_int8()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    const hstu_problem prob(GetParam());
    const std::vector<double> exact = prob.oracle();
    for (const int8_mode &mode : kInt8Modes)
        check_int8(prob, exact, mode, {0, 1, 5, 64});
}

TEST_P(HstuAttentionInt8, FastSiluMatchesOracles) {
    if (!has_int8()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    const hstu_problem prob(GetParam());
    const std::vector<double> exact = prob.oracle();
    for (const int8_mode &mode : kInt8Modes)
        check_int8(prob, exact, mode, {0, 1, 5, 64}, 1);
}

/// The cases the tiled kernel runs (causal, BF16 Q/K/V; FlagsWithoutInt8
/// covers the others), plus scores large enough that exp(-x) saturates, the
/// regime of most recorded DLRM v3 scores.
INSTANTIATE_TEST_SUITE_P(Cases, HstuAttentionInt8,
        ::testing::ValuesIn(fast_silu_cases()),
        [](const ::testing::TestParamInfo<hstu_case> &info) {
            return info.param.name;
        });

/// Zero Q / K rows and V channels (zero scales, no NaN), and outliers.
TEST(HstuAttentionInt8Api, ZeroAndOutlierRows) {
    if (env_overrides_kernel() || !has_int8()) GTEST_SKIP();
    hstu_case c {"zero_outlier", {300, 129, 5}, 2, 32, 48, {7, 0, 2}};
    hstu_problem prob(c);
    for (int64_t row = 0; row < prob.offsets.back(); ++row) {
        float *x = &prob.values[row * prob.width];
        if (row % 7 == 0) std::fill_n(x + prob.q_off, c.heads * c.qk_dim, 0.f);
        if (row % 5 == 0 || row >= 300)
            std::fill_n(x + prob.k_off, c.heads * c.qk_dim, 0.f);
        x[prob.v_off + 3] = 0.f; // a zero V channel of head 0
        x[prob.v_off + c.v_dim + 1] *= 64.f; // an outlier channel of head 1
        if (row == 40) x[prob.k_off + 5] = 64.f; // an outlier key
    }
    const std::vector<double> exact = prob.oracle();
    for (const int8_mode &mode : kInt8Modes)
        check_int8(prob, exact, mode, {0, 3});
}

/// Rows are independent and every scale is per token or per key block, so
/// the INT8 output does not depend on the tile size or thread count either.
TEST(HstuAttentionInt8Api, DeterministicAcrossThreads) {
    if (env_overrides_kernel() || !has_int8()) GTEST_SKIP();
    for (const hstu_case &c : make_cases()) {
        if (!c.causal || c.in_dt != data_type_t::bf16) continue;
        const hstu_problem prob(c);
        for (const int8_mode &mode : kInt8Modes) {
            std::vector<float> want, got;
            ASSERT_EQ(prob.run(kernel_t::tiled, want, 1, nullptr, mode.qk,
                              mode.pv),
                    status_t::success);
            for (int32_t threads : {5, 64}) {
                ASSERT_EQ(prob.run(kernel_t::tiled, got, threads, nullptr,
                                  mode.qk, mode.pv),
                        status_t::success);
                EXPECT_TRUE(bitwise_equal(got, want))
                        << c.name << " " << mode.name
                        << ", threads=" << threads;
            }
        }
    }
}

/// Whether the rows past the first of every sequence are bit-identical.
bool rows_past_first_equal(const hstu_problem &prob,
        const std::vector<float> &got, const std::vector<float> &want) {
    if (got.size() != want.size()) return false;
    const int64_t w = prob.cs.heads * prob.cs.v_dim; // a row of `run`'s result
    for (size_t b = 0; b + 1 < prob.offsets.size(); ++b)
        for (int64_t row = prob.offsets[b] + 1; row < prob.offsets[b + 1];
                ++row)
            if (std::memcmp(&got[row * w], &want[row * w], w * sizeof(float)))
                return false;
    return true;
}

/// With every row a target (num_targets = the sequence length) there is no
/// history. The first row's band then ends at the row and takes its diagonal
/// in (row_band_t), but every later row has an empty band, so its output is
/// the diagonal term alone, which every policy adds the same way: bit for
/// bit the BF16 kernel's in each INT8 mode, with and without the fast silu
/// (it only changes the band), for BF16 and FP32 outputs and per-token
/// scales.
TEST(HstuAttentionInt8Api, TargetRowDiagonalMatchesBf16) {
    if (env_overrides_kernel() || !has_int8()) GTEST_SKIP();
    hstu_case bf16_out {
            "all_targets", {1, 17, 64, 130}, 2, 64, 96, {1, 17, 64, 130}};
    hstu_case f32_out = bf16_out;
    f32_out.name = "all_targets_f32_out_per_token_scale";
    f32_out.out_dt = data_type_t::f32;
    f32_out.per_token_scale = true;
    for (const hstu_case &c : {bf16_out, f32_out}) {
        const hstu_problem prob(c);
        for (int fast_silu : {0, 1}) {
            std::vector<float> want, got;
            ASSERT_EQ(prob.run(kernel_t::tiled, want, 0, nullptr, false, false,
                              fast_silu),
                    status_t::success);
            for (const int8_mode &mode : kInt8Modes) {
                ASSERT_EQ(prob.run(kernel_t::tiled, got, 0, nullptr, mode.qk,
                                  mode.pv, fast_silu),
                        status_t::success);
                EXPECT_TRUE(rows_past_first_equal(prob, got, want))
                        << c.name << " " << mode.name
                        << ", fast_silu=" << fast_silu;
            }
        }
    }
}

/// The default path is the explicit `fast_silu = false` one, bit for bit
/// (the golden dumps of the harness pin it across commits), in the BF16
/// kernel and every INT8 mode; the fast path is deterministic across thread
/// counts and differs from it.
TEST(HstuAttentionFastSilu, DefaultPathUnchanged) {
    if (env_overrides_kernel() || !has_bf16()) GTEST_SKIP();
    const char *env = std::getenv("ZENDNNL_HSTU_ATTENTION_FAST_SILU");
    const bool env_fast = ha::parse_bool_env(env) == 1;
    std::vector<int8_mode> modes {{"bf16", false, false}};
    if (has_int8())
        modes.insert(modes.end(), std::begin(kInt8Modes), std::end(kInt8Modes));
    for (const hstu_case &c : fast_silu_cases()) {
        const hstu_problem prob(c);
        for (const int8_mode &mode : modes) {
            const bool qk = mode.qk, pv = mode.pv;
            std::vector<float> slow, fast, got;
            ASSERT_EQ(prob.run(kernel_t::tiled, slow, 1, nullptr, qk, pv, 0),
                    status_t::success);
            ASSERT_EQ(prob.run(kernel_t::tiled, fast, 1, nullptr, qk, pv, 1),
                    status_t::success);
            for (int32_t threads : {0, 5, 64}) {
                ASSERT_EQ(
                        prob.run(kernel_t::none, got, threads, nullptr, qk, pv),
                        status_t::success);
                EXPECT_TRUE(bitwise_equal(got, env_fast ? fast : slow))
                        << c.name << " " << mode.name
                        << ", threads=" << threads;
                ASSERT_EQ(prob.run(kernel_t::tiled, got, threads, nullptr, qk,
                                  pv, 1),
                        status_t::success);
                EXPECT_TRUE(bitwise_equal(got, fast))
                        << c.name << " " << mode.name
                        << ", fast, threads=" << threads;
            }
            if (c.name == "dlrm_shape_fused_uvqk" || c.name == "large_scores") {
                EXPECT_FALSE(bitwise_equal(fast, slow))
                        << c.name << " " << mode.name;
            }
        }
    }
}

/// INT8 and BF16 calls share the scratch arenas, and the INT8 pack stages
/// rows in the tile scratch.
TEST(HstuAttentionInt8Api, RepeatedCallsReuseScratch) {
    if (env_overrides_kernel() || !has_int8()) GTEST_SKIP();
    const hstu_case big = find_case("dlrm_shape_fused_uvqk");
    const hstu_case odd = find_case("odd_dims");
    const hstu_problem big_prob(big), odd_prob(odd);
    std::vector<float> big_bf16;
    ASSERT_EQ(big_prob.run(kernel_t::tiled, big_bf16), status_t::success);
    for (const int8_mode &mode : kInt8Modes) {
        std::vector<float> big0, odd0, got;
        ASSERT_EQ(big_prob.run(
                          kernel_t::tiled, big0, 0, nullptr, mode.qk, mode.pv),
                status_t::success);
        ASSERT_EQ(odd_prob.run(
                          kernel_t::tiled, odd0, 0, nullptr, mode.qk, mode.pv),
                status_t::success);
        for (int rep = 0; rep < 2; ++rep) {
            // BF16 after INT8 packs staged K/V in the same tile scratch.
            ASSERT_EQ(big_prob.run(kernel_t::tiled, got), status_t::success);
            EXPECT_TRUE(bitwise_equal(got, big_bf16)) << mode.name;
            ASSERT_EQ(odd_prob.run(kernel_t::tiled, got, 0, nullptr, mode.qk,
                              mode.pv),
                    status_t::success);
            EXPECT_TRUE(bitwise_equal(got, odd0)) << mode.name;
            ASSERT_EQ(big_prob.run(kernel_t::tiled, got, 0, nullptr, mode.qk,
                              mode.pv),
                    status_t::success);
            EXPECT_TRUE(bitwise_equal(got, big0)) << mode.name;
        }
    }
}

/// The flags change nothing where INT8 does not run: f32 q/k/v (dropped),
/// non-causal calls (reference) and a forced reference kernel.
TEST(HstuAttentionInt8Api, FlagsWithoutInt8) {
    if (env_overrides_kernel()) GTEST_SKIP();
    for (const char *name : {"f32", "noncausal", "causal"}) {
        const hstu_case c = find_case(name);
        const hstu_problem prob(c);
        const kernel_t kernel = std::string(name) == "causal"
                ? kernel_t::reference
                : kernel_t::none;
        std::vector<float> want, got;
        ASSERT_EQ(prob.run(kernel, want), status_t::success);
        for (int8_mode mode : kInt8Modes) {
            ASSERT_EQ(prob.run(kernel, got, 0, nullptr, mode.qk, mode.pv),
                    status_t::success);
            EXPECT_TRUE(bitwise_equal(got, want)) << name << " " << mode.name;
        }
    }
}

std::vector<size_t> non_finite_at(const std::vector<float> &x) {
    std::vector<size_t> at;
    for (size_t i = 0; i < x.size(); ++i)
        if (!std::isfinite(x[i])) at.push_back(i);
    return at;
}

/// A NaN or an infinity in Q, K or V reaches, in every INT8 mode, exactly
/// the outputs it reaches on the BF16 path, except that the scale of its V
/// channel in its key block can spread a V one to more rows of that channel;
/// the other head is untouched.
TEST(HstuAttentionInt8Api, NonFiniteInputs) {
    if (env_overrides_kernel() || !has_int8()) GTEST_SKIP();
    const hstu_case c {"non_finite", {300}, 2, 32, 48, {7}};
    const int64_t H = c.heads, Dv = c.v_dim;
    const hstu_problem clean_prob(c);
    std::vector<std::vector<float>> clean;
    for (const int8_mode &mode : kInt8Modes) {
        clean.emplace_back();
        ASSERT_EQ(clean_prob.run(kernel_t::tiled, clean.back(), 0, nullptr,
                          mode.qk, mode.pv),
                status_t::success);
    }
    for (const char *operand : {"q", "k", "v"})
        for (float bad : {std::numeric_limits<float>::quiet_NaN(),
                     std::numeric_limits<float>::infinity()}) {
            hstu_problem prob(c);
            const int64_t off = std::string(operand) == "q" ? prob.q_off
                    : std::string(operand) == "k"           ? prob.k_off
                                                            : prob.v_off;
            prob.values[100 * prob.width + off + 3] = bad;
            std::vector<float> got;
            ASSERT_EQ(prob.run(kernel_t::tiled, got), status_t::success);
            const std::vector<size_t> want = non_finite_at(got);
            ASSERT_FALSE(want.empty()) << operand << " " << bad;
            const bool is_v = std::string(operand) == "v";
            for (size_t m = 0; m < clean.size(); ++m) {
                const int8_mode &mode = kInt8Modes[m];
                ASSERT_EQ(prob.run(kernel_t::tiled, got, 0, nullptr, mode.qk,
                                  mode.pv),
                        status_t::success);
                const std::vector<size_t> have = non_finite_at(got);
                if (is_v) {
                    EXPECT_TRUE(std::includes(
                            have.begin(), have.end(), want.begin(), want.end()))
                            << operand << " " << bad << " " << mode.name;
                    EXPECT_TRUE(std::all_of(have.begin(), have.end(),
                            [&](size_t i) {
                        return int64_t(i) / Dv % H == 0 && int64_t(i) % Dv == 3;
                    })) << operand
                        << " " << bad << " " << mode.name
                        << ": spread past head 0, channel 3";
                } else {
                    EXPECT_EQ(have, want)
                            << operand << " " << bad << " " << mode.name << ": "
                            << have.size() << " non-finite outputs, BF16 has "
                            << want.size();
                }
                bool other_head_clean = true;
                for (size_t i = 0; i < got.size(); ++i)
                    if (int64_t(i) / Dv % H == 1
                            && std::memcmp(&got[i], &clean[m][i], sizeof(float))
                                    != 0)
                        other_head_clean = false;
                EXPECT_TRUE(other_head_clean)
                        << operand << " " << bad << " " << mode.name;
            }
        }
}

/// INT8 Q*K^T needs qk_dim <= kInt8MaxQkDim (its int32 dot products reach
/// 127^2 qk_dim); past it is_qk_quant is dropped and the call runs BF16, and
/// the INT8 entry point refuses it. At the limit, all-ones Q and K rows
/// (K negated on odd tokens) take the dot products to 127^2 kInt8MaxQkDim =
/// 2,147,479,576, which still fits.
TEST(HstuAttentionInt8Api, QkDimLimit) {
    if (env_overrides_kernel() || !has_int8()) GTEST_SKIP();
    const hstu_case c {"qk_dim_limit", {40}, 1, ha::kInt8MaxQkDim + 1, 16};
    const hstu_problem prob(c);
    std::vector<float> want, got;
    ASSERT_EQ(prob.run(kernel_t::tiled, want, 1), status_t::success);
    for (const int8_mode &mode : kInt8Modes) {
        if (mode.pv) continue; // INT8 P*V still runs, so the output differs
        ASSERT_EQ(prob.run(kernel_t::tiled, got, 1, nullptr, mode.qk, mode.pv),
                status_t::success);
        EXPECT_TRUE(bitwise_equal(got, want)) << mode.name;
    }
    EXPECT_EQ(prob.run(kernel_t::tiled, got, 1, nullptr, true, false, 0),
            status_t::unimplemented);

    const hstu_case at {"qk_dim_at_limit", {40}, 1, ha::kInt8MaxQkDim, 16};
    hstu_problem limit(at);
    for (int64_t row = 0; row < limit.offsets.back(); ++row) {
        float *x = &limit.values[row * limit.width];
        std::fill_n(x + limit.q_off, at.qk_dim, 1.f);
        std::fill_n(x + limit.k_off, at.qk_dim, row % 2 ? -1.f : 1.f);
    }
    check_int8(limit, limit.oracle(), {"qk", true, false}, {1});
}

void set_env(const char *name, const char *value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

/// ZENDNNL_HSTU_ATTENTION_INT8_QK / _PV reach the kernel. They are read
/// once, so each setting runs in a fresh process: with the params flags off
/// the variables switch each INT8 mode on (the output is not the BF16 one
/// and matches the quantized oracle), and with them on, 0 switches both off.
TEST(HstuAttentionInt8DeathTest, EnvironmentOverrides) {
    if (env_overrides_kernel() || !has_int8()) GTEST_SKIP();
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    const hstu_case c = find_case("targets");
    const hstu_problem prob(c);
    // The BF16 baseline through the tiled kernel directly: hstu_attention_direct
    // before set_env would cache the unset variables in the child too.
    const int fast = ha::parse_bool_env(
                             std::getenv("ZENDNNL_HSTU_ATTENTION_FAST_SILU"))
            == 1;
    const double tight = fast ? 3e-3 : 2e-3;
    std::vector<float> bf16;
    ASSERT_EQ(prob.run(kernel_t::tiled, bf16, 0, nullptr, false, false, fast),
            status_t::success);
    for (const int8_mode &mode : kInt8Modes)
        EXPECT_EXIT(
                {
                    set_env("ZENDNNL_HSTU_ATTENTION_INT8_QK",
                            mode.qk ? "1" : "0");
                    set_env("ZENDNNL_HSTU_ATTENTION_INT8_PV",
                            mode.pv ? "1" : "0");
                    std::vector<float> got;
                    const bool ok
                            = prob.run(kernel_t::none, got) == status_t::success
                            && !bitwise_equal(got, bf16)
                            && rel_error_past_output(got,
                                       prob.oracle_int8(mode.qk, mode.pv),
                                       prob.cs.out_dt)
                                    < tight;
                    std::exit(ok ? 0 : 1);
                },
                ::testing::ExitedWithCode(0), "")
                << mode.name;
    EXPECT_EXIT(
            {
                set_env("ZENDNNL_HSTU_ATTENTION_INT8_QK", "0");
                set_env("ZENDNNL_HSTU_ATTENTION_INT8_PV", "off");
                std::vector<float> got;
                const bool ok
                        = prob.run(kernel_t::none, got, 0, nullptr, true, true)
                                == status_t::success
                        && bitwise_equal(got, bf16);
                std::exit(ok ? 0 : 1);
            },
            ::testing::ExitedWithCode(0), "")
            << "overrides off";
}

namespace hq = ha::quant;

/// The code of x for a group with inverse scale `inv`: RNE of the exact
/// product (exact in double), plus `offset`, as a byte.
uint8_t quant_code(float x, float inv, int offset) {
    return uint8_t(int(std::nearbyint(double(x) * double(inv))) + offset);
}

/// Each quantizer against the scalar definition, code for code.
TEST(HstuAttentionInt8Quant, Row) {
    if (!has_bf16()) GTEST_SKIP();
    std::mt19937 gen(11);
    std::normal_distribution<float> nd(0.f, 3.f);
    for (int64_t n : {1, 15, 16, 17, 100, 128})
        for (int offset : {0, 128})
            for (int kind = 0; kind < 4; ++kind) {
                std::vector<bfloat16_t> x(n);
                for (auto &v : x)
                    v = bfloat16_t(kind == 2 ? 0.f : nd(gen));
                if (kind == 1) x[n / 2] = bfloat16_t(-200.f); // an outlier
                if (kind == 3) x[n - 1] = bfloat16_t(200.f); // in the tail
                float amax = 0.f;
                for (auto v : x)
                    amax = std::max(amax, std::abs(float(v)));
                std::vector<uint8_t> y(n + 64, 0xAB);
                const float scale
                        = hq::quantize_row(x.data(), n, offset, y.data());
                EXPECT_EQ(scale, hq::quant_scale(amax)) << n;
                const float inv = hq::quant_inv(amax);
                for (int64_t i = 0; i < n; ++i)
                    ASSERT_EQ(y[i], quant_code(float(x[i]), inv, offset))
                            << "n=" << n << " offset=" << offset << " i=" << i;
                for (int64_t i = n; i < n + 64; ++i)
                    ASSERT_EQ(y[i], 0xAB) << "wrote past n=" << n;
            }
    // Exact ties round to even: max 127 gives inv = 1.
    const float tie[] = {127.f, 2.5f, 3.5f, -2.5f, -0.5f, 0.5f, 1.5f, -1.5f};
    const int want[] = {127, 2, 4, -2, 0, 0, 2, -2};
    std::vector<bfloat16_t> x(std::begin(tie), std::end(tie));
    std::vector<uint8_t> y(x.size());
    hq::quantize_row(x.data(), int64_t(x.size()), 0, y.data());
    for (size_t i = 0; i < x.size(); ++i)
        EXPECT_EQ(int(int8_t(y[i])), want[i]) << tie[i];
    // ... whatever the caller's rounding mode.
    for (int mode : {FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
        const int saved = std::fegetround();
        std::fesetround(mode);
        hq::quantize_row(x.data(), int64_t(x.size()), 0, y.data());
        std::fesetround(saved);
        for (size_t i = 0; i < x.size(); ++i)
            EXPECT_EQ(int(int8_t(y[i])), want[i])
                    << tie[i] << ", rounding mode " << mode;
    }
}

TEST(HstuAttentionInt8Quant, Columns) {
    if (!has_bf16()) GTEST_SKIP();
    std::mt19937 gen(12);
    std::normal_distribution<float> nd(0.f, 2.f);
    constexpr int64_t ldx = 40, ldy = 32;
    for (int64_t rows : {1, 7, 256})
        for (int64_t n : {1, 16, 17, 32}) {
            std::vector<bfloat16_t> x(rows * ldx);
            for (auto &v : x)
                v = bfloat16_t(nd(gen));
            for (int64_t r = 0; r < rows; ++r)
                x[r * ldx + n / 2] = bfloat16_t(0.f); // a zero column
            x[(rows / 2) * ldx] = bfloat16_t(150.f); // an outlier
            std::vector<int8_t> y(rows * ldy, 0x55);
            std::vector<float> scale(32, -1.f);
            hq::quantize_cols(
                    x.data(), ldx, rows, n, y.data(), ldy, scale.data());
            for (int64_t j = 0; j < 32; ++j) {
                if (j >= n) {
                    EXPECT_EQ(scale[j], 0.f) << j;
                    for (int64_t r = 0; r < rows; ++r)
                        ASSERT_EQ(y[r * ldy + j], 0) << "r=" << r << " j=" << j;
                    continue;
                }
                float amax = 0.f;
                for (int64_t r = 0; r < rows; ++r)
                    amax = std::max(amax, std::abs(float(x[r * ldx + j])));
                EXPECT_EQ(scale[j], hq::quant_scale(amax)) << j;
                const float inv = hq::quant_inv(amax);
                for (int64_t r = 0; r < rows; ++r)
                    ASSERT_EQ(uint8_t(y[r * ldy + j]),
                            quant_code(float(x[r * ldx + j]), inv, 0))
                            << "rows=" << rows << " n=" << n << " r=" << r
                            << " j=" << j;
            }
        }
}

TEST(HstuAttentionInt8Quant, RowsWithMax) {
    if (!has_bf16()) GTEST_SKIP();
    std::mt19937 gen(13);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.5f, 2.f);
    constexpr int64_t ld = 64, rows = 9, ra = 2, rb = 8;
    for (auto [lo, hi] :
            {std::pair<int64_t, int64_t> {0, 64}, {4, 37}, {8, 9}}) {
        std::vector<float> x(rows * ld), amax(16 * rows, 0.f), mul(rows);
        for (auto &v : x)
            v = nd(gen);
        for (int64_t r = 0; r < rows; ++r) {
            float m = 0.f;
            for (int64_t c = lo; c < hi; ++c)
                m = std::max(m, std::abs(x[r * ld + c]));
            // The max in one lane; row 5 keeps a looser bound than its data.
            amax[16 * r + r % 16] = r == 5 ? 4.f * m : m;
            mul[r] = r == 3 ? 0.f : ud(gen);
        }
        std::vector<uint8_t> y(rows * ld, 0xAB);
        std::vector<float> scale(rows, -1.f);
        hq::quantize_rows_amax(x.data(), ld, amax.data(), mul.data(), ra, rb,
                lo, hi, 128, y.data(), scale.data());
        for (int64_t r = 0; r < rows; ++r) {
            const bool in = r >= ra && r < rb;
            const float am = amax[16 * r + r % 16];
            EXPECT_EQ(scale[r], in ? hq::quant_scale(am) * mul[r] : -1.f) << r;
            const float inv = hq::quant_inv(am);
            for (int64_t c = 0; c < ld; ++c) {
                const bool w = in && c >= lo && c < hi;
                ASSERT_EQ(y[r * ld + c],
                        w ? quant_code(x[r * ld + c], inv, 128) : 0xAB)
                        << "lo=" << lo << " hi=" << hi << " r=" << r
                        << " c=" << c;
            }
        }
    }
}

} // namespace
