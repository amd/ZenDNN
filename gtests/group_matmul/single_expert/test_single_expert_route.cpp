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

/// Single-expert `matmul_direct` route tests: which narrow single
/// experts the scalar custom-kernel route takes under the single-expert
/// phase knobs and `..._FORCE_CK`, the prompt -> decode -> prompt phase
/// switch of a narrow quantized layer, and the env-string grammar end to
/// end in a child process.  Route code lives in
/// `lowoha_operators/matmul/group_matmul/single_expert/single_expert_route.hpp`.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <type_traits>

#include "../custom_kernel/ck_test_helpers.hpp"
#include "../custom_kernel/route_test_utils.hpp"
#include "common/op_config.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/pack.hpp"
#include "lowoha_operators/matmul/group_matmul/executors/n_tile/prompt_tile/prompt_tile_tunables.hpp"
#include "lowoha_operators/matmul/group_matmul/prepack/prepack_custom_kernel.hpp"
#include "lowoha_operators/matmul/group_matmul/single_expert/single_expert_policy.hpp"
#include "lowoha_operators/matmul/group_matmul/single_expert/single_expert_route.hpp"
#include "lowoha_operators/matmul/lowoha_matmul.hpp"
#include "moe_test_utils.hpp"

namespace {

namespace mt = moe_test_utils;
using mt::bfloat16_t;
using mt::data_type_t;
using mt::matmul_params;
using mt::status_t;
using route_test_utils::ScalarRouteConfigGuard;

// A narrow single expert (N below decode's N-tile count) routes only for a
// dtype `..._FORCE_CK` lists, and only in a phase a phase knob covers.
TEST(CkDispatchRouting, ScalarRouteOnlyForceCkLiftsNGate) {
    CK_SKIP_IF_NO_INT8_ISA();

    ScalarRouteConfigGuard guard;
    mt::CustomKernelOverride group_ck_on(true);
    mt::CustomKernelInt8Override int8_ck_on(true);
    mt::NTileStrategyOverride rounds(2);
    guard.config.set_custom_kernel_route(true);
    guard.config.set_algo(static_cast<int32_t>(
            zendnnl::common::matmul_algo_t::aocl_dlp_blocked));
    guard.config.set_weight_cache(2);
    omp_set_dynamic(0);

    constexpr int K = 1280;
    constexpr int N = 1280;
    constexpr int kThreads = 32;
    ASSERT_FALSE(zendnnl::lowoha::matmul::has_enough_single_expert_tiles(
            N, K, kThreads));
    float wei_scale = 1.0f;

    const auto routable = [&](int M, const matmul_params &p) {
        return zendnnl::lowoha::matmul::single_matmul_group_routable('r', M, N,
                K, kThreads, 1, false, true, 1.0f, 0.0f, nullptr, K, true, p);
    };
    const auto make = [&](data_type_t wei) {
        matmul_params p;
        p.dtypes.src = data_type_t::bf16;
        p.dtypes.wei = wei;
        p.dtypes.dst = data_type_t::bf16;
        p.dtypes.bias = data_type_t::none;
        if (wei != data_type_t::bf16) {
            p.dtypes.compute = data_type_t::s8;
            p.dynamic_quant = true;
            p.quant_params.src_scale.dt = data_type_t::f32;
            p.quant_params.src_scale.dims = {1, 1};
            p.quant_params.wei_scale.buff = &wei_scale;
            p.quant_params.wei_scale.dt = data_type_t::f32;
            p.quant_params.wei_scale.dims
                    = {wei == data_type_t::s4 ? K / 64 : 1, N};
        }
        return p;
    };

    for (const data_type_t wei :
            {data_type_t::bf16, data_type_t::s8, data_type_t::s4}) {
        if (wei == data_type_t::bf16
                && !zendnnl::lowoha::matmul::custom_kernel::
                           bf16_isa_supported())
            continue;
        SCOPED_TRACE(static_cast<int>(wei));
        for (const int M : {1, 512}) {
            SCOPED_TRACE(M);
            matmul_params p = make(wei);
            if (wei != data_type_t::bf16)
                p.quant_params.src_scale.dims = {M, 1};
            {
                mt::SingleExpertCkPolicy off(0, 0, 0);
                EXPECT_FALSE(routable(M, p))
                        << "policy off: the N gate applies";
            }
            for (const auto &[pk, dk] :
                    {std::pair {1, 0}, std::pair {0, 1}, std::pair {1, 1}}) {
                SCOPED_TRACE(::testing::Message()
                        << "prompt=" << pk << " decode=" << dk);
                mt::SingleExpertCkPolicy policy(pk, dk, 0);
                EXPECT_FALSE(routable(M, p)) << "the phase knobs keep the gate";
            }
            {
                mt::SingleExpertCkPolicy force_only(0, 0, 1);
                EXPECT_FALSE(routable(M, p)) << "FORCE_CK adds no phase";
            }
            // With a phase knob on, FORCE_CK lifts the gate; a raw weight
            // then routes in both phases.
            for (const auto &[pk, dk] : {std::pair {0, 1}, std::pair {1, 1}}) {
                mt::SingleExpertCkPolicy forced(pk, dk, 1);
                EXPECT_TRUE(routable(M, p));
            }
            {
                mt::SingleExpertCkPolicy force(1, 1, 1);
                EXPECT_TRUE(routable(M, p));
                mt::CustomKernelOverride ck_off(false);
                EXPECT_FALSE(routable(M, p))
                        << "the master CK knob still gates";
            }
        }
    }

    if (zendnnl::lowoha::matmul::custom_kernel::avx512f16_available()) {
        mt::CustomKernelF16Override f16_on(true);
        matmul_params p;
        p.dtypes.src = data_type_t::f16;
        p.dtypes.wei = data_type_t::f16;
        p.dtypes.dst = data_type_t::f16;
        p.dtypes.bias = data_type_t::none;
        for (const auto &[pk, dk] :
                {std::pair {0, 0}, std::pair {1, 0}, std::pair {1, 1}}) {
            mt::SingleExpertCkPolicy phase_only(pk, dk, 0);
            EXPECT_FALSE(routable(1, p));
            EXPECT_FALSE(routable(512, p));
        }
        mt::SingleExpertCkPolicy force(1, 1, 1);
        EXPECT_TRUE(routable(1, p));
        EXPECT_TRUE(routable(512, p));
    }

    // A dtype list covers only the dtypes it names.
    using zendnnl::lowoha::matmul::kSingleExpertAll;
    using zendnnl::lowoha::matmul::kSingleExpertW4a8;
    mt::SingleExpertCkMasks w4a8_only(
            kSingleExpertAll, kSingleExpertAll, kSingleExpertW4a8);
    for (const int M : {1, 512}) {
        SCOPED_TRACE(M);
        matmul_params w4 = make(data_type_t::s4);
        matmul_params i8 = make(data_type_t::s8);
        w4.quant_params.src_scale.dims = {M, 1};
        i8.quant_params.src_scale.dims = {M, 1};
        EXPECT_TRUE(routable(M, w4));
        EXPECT_FALSE(routable(M, i8));
    }
}

TEST(CkDispatchRouting, SingleExpertW4A8F32OutputIsRoutable) {
    CK_SKIP_IF_NO_INT8_ISA();

    ScalarRouteConfigGuard guard;
    mt::CustomKernelOverride group_ck_on(true);
    mt::CustomKernelInt8Override int8_ck_on(true);
    mt::SingleExpertCkPolicy force_w4a8(1, 1, 1);
    guard.config.set_custom_kernel_route(true);
    guard.config.set_algo(static_cast<int32_t>(
            zendnnl::common::matmul_algo_t::aocl_dlp_blocked));
    guard.config.set_weight_cache(2);

    constexpr int M = 1;
    constexpr int N = 1280;
    constexpr int K = 1280;
    constexpr int kThreads = 32;
    float wei_scale = 1.0f;

    matmul_params p;
    p.dtypes.src = data_type_t::bf16;
    p.dtypes.wei = data_type_t::s4;
    p.dtypes.dst = data_type_t::f32;
    p.dtypes.bias = data_type_t::none;
    p.dtypes.compute = data_type_t::s8;
    p.dynamic_quant = true;
    p.quant_params.src_scale.dt = data_type_t::f32;
    p.quant_params.src_scale.dims = {M, 1};
    p.quant_params.wei_scale.buff = &wei_scale;
    p.quant_params.wei_scale.dt = data_type_t::f32;
    p.quant_params.wei_scale.dims = {K / 64, N};

    EXPECT_TRUE(zendnnl::lowoha::matmul::single_matmul_group_routable('r', M, N,
            K, kThreads, 1, false, true, 1.0f, 0.0f, nullptr, K, true, p));
}

// A Whisper-shaped narrow quantized linear (K 1280, N 1280 on 32 threads is
// 40 NR32 tiles, below decode's 64) runs prompt -> decode -> prompt through
// `matmul_direct` under each single-expert CK knob setting, the split ones
// included.  A covered decode runs the custom kernel, a covered prompt the
// prompt tile executor, and every call matches the policy-off AOCL-DLP
// result.
void run_narrow_quant_phase_switch(bool w4, int weight_cache, int prompt_knob,
        int decode_knob, int force_knob, int kThreads = 32) {
    constexpr int N = 1280, K = 1280, kGroup = 64;
    constexpr int G = K / kGroup;
    const std::vector<int> rows {256, 1, 4, 64, 65, 1, 512};
    SCOPED_TRACE(::testing::Message()
            << (w4 ? "W4A8" : "DQ-INT8") << " wc=" << weight_cache
            << " prompt=" << prompt_knob << " decode=" << decode_knob
            << " force=" << force_knob << " threads=" << kThreads);

    const bool wide_n = zendnnl::lowoha::matmul::has_enough_single_expert_tiles(
            N, K, kThreads);
    std::vector<int8_t> weight(
            w4 ? static_cast<size_t>(N) * K / 2 : static_cast<size_t>(N) * K);
    std::mt19937 rng(w4 ? 0x5EED4U : 0x5EED8U);
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            const int v = static_cast<int>(rng() % 15U) - 7;
            const size_t i = static_cast<size_t>(n) * K + k;
            if (!w4) {
                weight[i] = static_cast<int8_t>(v);
                continue;
            }
            uint8_t byte = static_cast<uint8_t>(weight[i >> 1]);
            const uint8_t nib = static_cast<uint8_t>(v) & uint8_t {0x0F};
            byte = (i & 1U) ? static_cast<uint8_t>((byte & 0x0FU) | (nib << 4))
                            : static_cast<uint8_t>((byte & 0xF0U) | nib);
            weight[i >> 1] = static_cast<int8_t>(byte);
        }
    }
    const std::vector<int8_t> weight_copy = weight;
    std::vector<int8_t> weight_ref = weight;
    std::vector<float> wei_scale(static_cast<size_t>(w4 ? G : 1) * N);
    for (size_t i = 0; i < wei_scale.size(); ++i)
        wei_scale[i] = 0.02f + 0.001f * static_cast<float>(rng() % 9U);

    const auto make_params = [&](int M, std::vector<float> &src_scale) {
        matmul_params p;
        p.dtypes.src = data_type_t::bf16;
        p.dtypes.wei = w4 ? data_type_t::s4 : data_type_t::s8;
        p.dtypes.dst = data_type_t::bf16;
        p.dtypes.bias = data_type_t::none;
        p.dtypes.compute = data_type_t::s8;
        p.dynamic_quant = true;
        p.mem_format_a = 'n';
        p.mem_format_b = 'n';
        p.weight_cache_type = weight_cache;
        p.num_threads = kThreads;
        src_scale.assign(static_cast<size_t>(M), 0.0f);
        p.quant_params.src_scale.buff = w4 ? src_scale.data() : nullptr;
        p.quant_params.src_scale.dt = data_type_t::f32;
        p.quant_params.src_scale.dims = {M, 1};
        p.quant_params.wei_scale.buff = wei_scale.data();
        p.quant_params.wei_scale.dt = data_type_t::f32;
        p.quant_params.wei_scale.dims = {w4 ? G : 1, N};
        return p;
    };
    const auto make_src = [&](int M) {
        std::mt19937 src_rng(0xA5A5U + static_cast<unsigned>(M));
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<bfloat16_t> src(static_cast<size_t>(M) * K);
        for (auto &v : src)
            v = bfloat16_t(dist(src_rng));
        return src;
    };
    const auto run
            = [&](int M, std::vector<int8_t> &w, std::vector<bfloat16_t> &dst) {
        std::vector<float> src_scale;
        const matmul_params p = make_params(M, src_scale);
        const auto src = make_src(M);
        dst.assign(static_cast<size_t>(M) * N, bfloat16_t(-123.0f));
        zendnnl::lowoha::matmul::matmul_batch_params_t batch;
        return zendnnl::lowoha::matmul::matmul_direct('r', false, true, M, N, K,
                1.0f, src.data(), K, w.data(), K, /*bias=*/nullptr, 0.0f,
                dst.data(), N, /*is_weights_const=*/true, batch, p);
    };

    ScalarRouteConfigGuard guard;
    mt::AlgoEnvGuard group_auto(0);
    mt::CustomKernelOverride group_ck_on(true);
    mt::CustomKernelInt8Override int8_ck_on(true);
    mt::NTileStrategyOverride rounds(2);
    guard.config.set_algo(static_cast<int32_t>(
            zendnnl::common::matmul_algo_t::aocl_dlp_blocked));
    guard.config.set_weight_cache(weight_cache);
    omp_set_dynamic(0);

    std::vector<std::vector<bfloat16_t>> expected(rows.size());
    {
        mt::SingleExpertCkPolicy off(0, 0, 0);
        guard.config.set_custom_kernel_route(true);
        ::reset_grp_matmul_caches();
        for (size_t i = 0; i < rows.size(); ++i) {
            ASSERT_EQ(run(rows[i], weight_ref, expected[i]), status_t::success);
            std::vector<float> scale;
            if (!wide_n) {
                ASSERT_FALSE(
                        zendnnl::lowoha::matmul::single_matmul_group_routable(
                                'r', rows[i], N, K, kThreads, 1, false, true,
                                1.0f, 0.0f, nullptr, K, true,
                                make_params(rows[i], scale)))
                        << "policy off: the N gate keeps DLP";
            }
        }
    }

    mt::SingleExpertCkPolicy policy(prompt_knob, decode_knob, force_knob);
    // A phase knob covers its phase; below decode's N-tile count only for a
    // dtype FORCE_CK lists.  `-1` reads the env-configured mask.
    namespace mm = zendnnl::lowoha::matmul;
    const unsigned bit = w4 ? mm::kSingleExpertW4a8 : mm::kSingleExpertInt8;
    const auto on = [&](int knob, unsigned mask) {
        return knob < 0 ? (mask & bit) != 0 : knob != 0;
    };
    const bool n_ok = wide_n
            || on(force_knob,
                    mm::get_grp_matmul_single_expert_force_ck_dtypes());
    const bool prompt_ck = n_ok
            && on(prompt_knob,
                    mm::get_grp_matmul_single_expert_prompt_tile_dtypes());
    const bool decode_ck = n_ok
            && on(decode_knob,
                    mm::get_grp_matmul_single_expert_decode_tile_dtypes());
    ::reset_grp_matmul_caches();
    namespace pt = zendnnl::lowoha::matmul::prompt_tile;
    for (size_t i = 0; i < rows.size(); ++i) {
        const int M = rows[i];
        SCOPED_TRACE(M);
        std::vector<bfloat16_t> dst;
        const long served0 = pt::test_api::s_calls_served.load();
        std::string mode;
        {
            mt::ExecutionPathCaptureGuard capture;
            ASSERT_EQ(run(M, weight, dst), status_t::success);
            const char *m
                    = zendnnl::lowoha::matmul::test_api::
                              s_last_group_matmul_direct_execution_path.load(
                                      std::memory_order_relaxed);
            mode = m != nullptr ? m : "";
        }
        if (M <= zendnnl::lowoha::matmul::kDecodeMaxM) {
            // An uncovered decode still routes and may take Rule 0.45's CK.
            if (decode_ck) {
                EXPECT_NE(mode.find("custom"), std::string::npos) << mode;
            }
        } else {
            EXPECT_EQ(pt::test_api::s_calls_served.load() - served0,
                    prompt_ck ? 1 : 0)
                    << mode;
        }
        double sq = 0.0;
        for (const bfloat16_t v : expected[i])
            sq += static_cast<double>(v) * static_cast<double>(v);
        const float rms = static_cast<float>(
                std::sqrt(sq / static_cast<double>(expected[i].size())));
        ASSERT_GT(rms, 0.5f) << "the oracle must not be near zero";
        size_t mismatches = 0;
        float max_err = 0.0f;
        for (size_t j = 0; j < dst.size(); ++j) {
            const float a = static_cast<float>(dst[j]);
            const float e = static_cast<float>(expected[i][j]);
            const float err = std::fabs(a - e);
            max_err = std::max(max_err, err);
            if (!std::isfinite(a) || err > 0.02f * std::fabs(e) + 0.01f * rms)
                ++mismatches;
        }
        EXPECT_EQ(mismatches, 0u)
                << "max_abs_error=" << max_err << " rms=" << rms;
    }
    EXPECT_TRUE(weight == weight_copy)
            << "INT8 and W4A8 custom-kernel packs stay out of place";
    ::reset_grp_matmul_caches();
}

TEST(CkDispatchRouting, NarrowQuantLayerPromptDecodeUnderSingleExpertPolicies) {
    CK_SKIP_IF_NO_INT8_ISA();
    for (const bool w4 : {false, true})
        for (const int wc : {1, 2})
            // Below the N gate (32 threads) with and without FORCE_CK, and
            // above it (8 threads: 40 NR32 tiles), for each phase knob.
            for (const auto &[pk, dk, fk, threads] : {std::tuple {1, 1, 1, 32},
                         std::tuple {0, 1, 1, 32}, std::tuple {0, 0, 1, 32},
                         std::tuple {1, 1, 0, 8}, std::tuple {1, 0, 0, 8},
                         std::tuple {0, 1, 0, 8}, std::tuple {1, 1, 0, 32}})
                run_narrow_quant_phase_switch(w4, wc, pk, dk, fk, threads);
}

// The knobs are read once per process, so the real env strings run in a
// child.  `FORCE_CK=W4A8`, `PROMPT_TILE=off`, `DECODE_TILE=all` puts only
// W4A8 decode on the custom kernel below the N gate, end to end through
// `matmul_direct`.
void check_single_expert_env_w4a8_only() {
    namespace mm = zendnnl::lowoha::matmul;
    setenv("ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_FORCE_CK", "W4A8", 1);
    setenv("ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_PROMPT_TILE", "off", 1);
    setenv("ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_DECODE_TILE", "all\n", 1);
    mt::SingleExpertCkPolicy env(-1, -1, -1);
    EXPECT_EQ(mm::get_grp_matmul_single_expert_force_ck_dtypes(),
            mm::kSingleExpertW4a8);
    EXPECT_EQ(mm::get_grp_matmul_single_expert_prompt_tile_dtypes(), 0u);
    EXPECT_EQ(mm::get_grp_matmul_single_expert_decode_tile_dtypes(),
            mm::kSingleExpertAll);
    ScalarRouteConfigGuard guard;
    mt::CustomKernelOverride group_ck_on(true);
    mt::CustomKernelInt8Override int8_ck_on(true);
    mt::NTileStrategyOverride rounds(2);
    guard.config.set_custom_kernel_route(true);
    guard.config.set_algo(static_cast<int32_t>(
            zendnnl::common::matmul_algo_t::aocl_dlp_blocked));
    guard.config.set_weight_cache(2);
    omp_set_dynamic(0);
    constexpr int K = 1280, N = 1280;
    float wei_scale = 1.0f;
    for (const bool w4 : {true, false}) {
        for (const int M : {1, 512}) {
            matmul_params p;
            p.dtypes.src = data_type_t::bf16;
            p.dtypes.wei = w4 ? data_type_t::s4 : data_type_t::s8;
            p.dtypes.dst = data_type_t::bf16;
            p.dtypes.compute = data_type_t::s8;
            p.dynamic_quant = true;
            p.quant_params.src_scale.dt = data_type_t::f32;
            p.quant_params.src_scale.dims = {M, 1};
            p.quant_params.wei_scale.buff = &wei_scale;
            p.quant_params.wei_scale.dt = data_type_t::f32;
            p.quant_params.wei_scale.dims = {w4 ? K / 64 : 1, N};
            EXPECT_EQ(mm::single_matmul_group_routable('r', M, N, K, 32, 1,
                              false, true, 1.0f, 0.0f, nullptr, K, true, p),
                    w4)
                    << (w4 ? "W4A8" : "DQ-INT8") << " M=" << M;
        }
    }
    run_narrow_quant_phase_switch(
            /*w4=*/true, /*wc=*/2, -1, -1, -1);
}

// Unknown and empty values keep each knob's default.
void check_single_expert_env_junk_keeps_default() {
    namespace mm = zendnnl::lowoha::matmul;
    setenv("ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_FORCE_CK", "fp16", 1);
    setenv("ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_PROMPT_TILE", "", 1);
    unsetenv("ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_DECODE_TILE");
    mt::SingleExpertCkPolicy env(-1, -1, -1);
    EXPECT_EQ(mm::get_grp_matmul_single_expert_force_ck_dtypes(),
            mm::kSingleExpertAll)
            << "a value with no known token keeps the default";
    EXPECT_EQ(mm::get_grp_matmul_single_expert_prompt_tile_dtypes(),
            mm::kSingleExpertAll);
    EXPECT_EQ(mm::get_grp_matmul_single_expert_decode_tile_dtypes(),
            mm::kSingleExpertAll);
}

TEST(CkDispatchRouting, SingleExpertEnvStringsCascadeEndToEnd) {
    CK_SKIP_IF_NO_INT8_ISA();
    testing::FLAGS_gtest_death_test_style = "threadsafe";
    EXPECT_EXIT(
            {
                check_single_expert_env_w4a8_only();
                std::exit(::testing::Test::HasFailure() ? 1 : 0);
            },
            ::testing::ExitedWithCode(0), "");
    EXPECT_EXIT(
            {
                check_single_expert_env_junk_keeps_default();
                std::exit(::testing::Test::HasFailure() ? 1 : 0);
            },
            ::testing::ExitedWithCode(0), "");
}

} // namespace
