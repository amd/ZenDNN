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

/// Single-expert blocking plan and executor tests: the knob grammar, the plan
/// `se_block_plan` picks for each split, and that every plan writes the same
/// output as the N-tile executor's N-only split (exactly for DQ-INT8, to f32
/// rounding for BF16).  Code lives in
/// `lowoha_operators/matmul/group_matmul/single_expert/single_expert_plan.*`
/// and `single_expert_executor.*`.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "../custom_kernel/ck_test_helpers.hpp"
#include "../custom_kernel/route_test_utils.hpp"
#include "common/op_config.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/dispatch.hpp"
#include "lowoha_operators/matmul/group_matmul/single_expert/single_expert_plan.hpp"
#include "lowoha_operators/matmul/group_matmul/single_expert/single_expert_policy.hpp"
#include "lowoha_operators/matmul/lowoha_matmul.hpp"
#include "moe_test_utils.hpp"

namespace {

namespace mt = moe_test_utils;
namespace mm = zendnnl::lowoha::matmul;
using mt::bfloat16_t;
using mt::data_type_t;
using mt::matmul_params;
using mt::status_t;
using route_test_utils::ScalarRouteConfigGuard;

// Pins the plan knobs for the lifetime of a test.
class PlanOverride {
public:
    PlanOverride(const char *block, int ksplit) {
        if (block != nullptr) {
            std::string error;
            EXPECT_TRUE(mm::parse_single_expert_block(block, &knob_, &error))
                    << block << ": " << error;
        }
        prev_knob_
                = mm::test_api::s_single_expert_block_override.exchange(&knob_);
        prev_ksplit_ = mm::test_api::s_grp_matmul_single_expert_ksplit_override
                               .exchange(ksplit);
    }
    ~PlanOverride() {
        mm::test_api::s_single_expert_block_override.store(prev_knob_);
        mm::test_api::s_grp_matmul_single_expert_ksplit_override.store(
                prev_ksplit_);
    }

private:
    mm::se_block_knob_t knob_;
    const mm::se_block_knob_t *prev_knob_;
    int prev_ksplit_;
};

TEST(TestSingleExpertPlan, KnobGrammar) {
    mm::se_block_knob_t k;
    std::string error;
    ASSERT_TRUE(mm::parse_single_expert_block(
            "split=grid, MParts=2 kparts=4,kc=1024 mr=6 order=k", &k, &error));
    EXPECT_EQ(k.split, mm::se_split_t::grid);
    EXPECT_EQ(k.m_parts, 2);
    EXPECT_EQ(k.k_parts, 4);
    EXPECT_EQ(k.kc, 1024);
    EXPECT_EQ(k.mr, 6);
    EXPECT_EQ(k.order, mm::se_order_t::k_outer);
    ASSERT_TRUE(mm::parse_single_expert_block(
            "split=remainder kparts=auto", &k, &error));
    EXPECT_EQ(k.split, mm::se_split_t::remainder);
    EXPECT_EQ(k.k_parts, 0);
    ASSERT_TRUE(mm::parse_single_expert_block(" ", &k, &error));
    EXPECT_EQ(k.split, mm::se_split_t::legacy);
    EXPECT_FALSE(k.given);
    ASSERT_TRUE(mm::parse_single_expert_block("split=legacy", &k, &error));
    EXPECT_TRUE(k.given);
    for (const char *bad : {"split=tiles", "mparts=0", "mr=13", "order=m",
                 "kc=-4", "bogus=1", "kparts"}) {
        EXPECT_FALSE(mm::parse_single_expert_block(bad, &k, &error)) << bad;
    }
}

TEST(TestSingleExpertPlan, DefaultsKeepTheLegacySplit) {
    if (std::getenv("ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_BLOCK") != nullptr
            || std::getenv("ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_KSPLIT")
                    != nullptr)
        GTEST_SKIP() << "plan knobs set in the environment";
    EXPECT_FALSE(mm::se_block_plan_enabled());
    EXPECT_FALSE(mm::se_block_plan(1, 8, 2560, 4096, 32, 4, 32).active());
}

TEST(TestSingleExpertPlan, RemainderSplitsOnlyTheRemainderBlocks) {
    PlanOverride auto_ksplit(nullptr, 1);
    // 80 blocks on 32 threads: two whole blocks each, the last 16 in halves.
    auto p = mm::se_block_plan(1, 8, 2560, 4096, 32, 4, 32);
    EXPECT_EQ(p.split, mm::se_split_t::remainder);
    EXPECT_EQ(p.whole_blocks, 2);
    EXPECT_EQ(p.split_blocks, 16);
    EXPECT_EQ(p.k_parts, 2);
    // 72 blocks: 8 remainder blocks in four K ranges.
    p = mm::se_block_plan(1, 8, 2304, 4096, 32, 4, 32);
    EXPECT_EQ(p.split_blocks, 8);
    EXPECT_EQ(p.k_parts, 4);
    // Fewer blocks than threads.
    p = mm::se_block_plan(1, 8, 512, 4096, 32, 4, 32);
    EXPECT_EQ(p.whole_blocks, 0);
    EXPECT_EQ(p.k_parts, 2);
    // Even split, too many remainder blocks, shallow K, prompt, MoE: legacy.
    EXPECT_FALSE(mm::se_block_plan(1, 8, 3072, 4096, 32, 4, 32).active());
    EXPECT_FALSE(mm::se_block_plan(1, 8, 1792, 4096, 32, 4, 32).active());
    EXPECT_FALSE(mm::se_block_plan(1, 8, 2560, 256, 32, 4, 32).active());
    EXPECT_FALSE(mm::se_block_plan(1, 512, 2560, 4096, 32, 4, 32).active());
    EXPECT_FALSE(mm::se_block_plan(4, 8, 2560, 4096, 32, 4, 32).active());
}

TEST(TestSingleExpertPlan, AnExplicitLegacyKnobOverridesKsplit) {
    PlanOverride legacy("split=legacy", 1);
    EXPECT_FALSE(mm::se_block_plan_enabled());
    EXPECT_FALSE(mm::se_block_plan(1, 8, 2560, 4096, 32, 4, 32).active());
}

TEST(TestSingleExpertPlan, GridValidatesAgainstTheShape) {
    {
        PlanOverride grid("split=grid mparts=2 kparts=2 kc=1000", 0);
        const auto p = mm::se_block_plan(1, 8, 2560, 4096, 32, 4, 32);
        EXPECT_EQ(p.split, mm::se_split_t::grid);
        EXPECT_EQ(p.n_parts, 8);
        EXPECT_EQ(p.kc, 1000);
        // M parts are capped at M.
        EXPECT_EQ(mm::se_block_plan(1, 1, 2560, 4096, 32, 4, 32).m_parts, 1);
    }
    {
        // 3 x 1 does not divide 32 threads.
        PlanOverride grid("split=grid mparts=3", 0);
        EXPECT_FALSE(mm::se_block_plan(1, 8, 2560, 4096, 32, 4, 32).active());
    }
    {
        PlanOverride grid("split=grid kparts=2 kc=6", 0);
        EXPECT_EQ(mm::se_block_plan(1, 8, 2560, 4096, 32, 4, 32).kc, 4)
                << "kc rounds down to the K step";
    }
}

struct Case {
    const char *block;
    int ksplit;
};

// Every plan matches the N-only split, through `matmul_direct`.
void run_plans_match_legacy(bool int8, int N, int K, int threads) {
    SCOPED_TRACE(::testing::Message()
            << (int8 ? "DQ-INT8" : "BF16") << " N=" << N << " K=" << K
            << " threads=" << threads);
    std::mt19937 rng(0x5EEDU + static_cast<unsigned>(N + K + int8));
    std::vector<int8_t> w8;
    std::vector<bfloat16_t> w16;
    std::vector<float> wei_scale;
    if (int8) {
        w8.resize(static_cast<size_t>(N) * K);
        for (auto &w : w8)
            w = static_cast<int8_t>(static_cast<int>(rng() % 255U) - 127);
        wei_scale.resize(N);
        for (auto &s : wei_scale)
            s = 0.001f + 0.0001f * static_cast<float>(rng() % 17U);
    } else {
        std::uniform_real_distribution<float> wd(-0.05f, 0.05f);
        w16.resize(static_cast<size_t>(N) * K);
        for (auto &w : w16)
            w = bfloat16_t(wd(rng));
    }

    ScalarRouteConfigGuard guard;
    mt::AlgoEnvGuard group_auto(0);
    mt::CustomKernelOverride group_ck_on(true);
    mt::CustomKernelInt8Override int8_ck_on(true);
    mt::SingleExpertCkMasks masks(
            0u, mm::kSingleExpertAll, mm::kSingleExpertAll);
    guard.config.set_custom_kernel_route(true);
    guard.config.set_algo(static_cast<int32_t>(
            zendnnl::common::matmul_algo_t::aocl_dlp_blocked));
    guard.config.set_weight_cache(1);
    omp_set_dynamic(0);

    const auto run = [&](int M, std::vector<bfloat16_t> &dst) {
        matmul_params p;
        p.dtypes.src = data_type_t::bf16;
        p.dtypes.wei = int8 ? data_type_t::s8 : data_type_t::bf16;
        p.dtypes.dst = data_type_t::bf16;
        p.dtypes.bias = data_type_t::none;
        p.mem_format_a = 'n';
        p.mem_format_b = 'n';
        p.weight_cache_type = 1;
        p.num_threads = threads;
        if (int8) {
            p.dtypes.compute = data_type_t::s8;
            p.dynamic_quant = true;
            p.quant_params.src_scale.dt = data_type_t::f32;
            p.quant_params.src_scale.dims = {M, 1};
            p.quant_params.wei_scale.buff = wei_scale.data();
            p.quant_params.wei_scale.dt = data_type_t::f32;
            p.quant_params.wei_scale.dims = {1, N};
        }
        std::mt19937 src_rng(0xA5A5U + static_cast<unsigned>(M));
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<bfloat16_t> src(static_cast<size_t>(M) * K);
        for (auto &v : src)
            v = bfloat16_t(dist(src_rng));
        dst.assign(static_cast<size_t>(M) * N, bfloat16_t(-123.0f));
        mm::matmul_batch_params_t batch;
        const void *w = int8 ? static_cast<const void *>(w8.data())
                             : static_cast<const void *>(w16.data());
        return mm::matmul_direct('r', false, true, M, N, K, 1.0f, src.data(), K,
                w, K, /*bias=*/nullptr, 0.0f, dst.data(), N,
                /*is_weights_const=*/true, batch, p);
    };

    const Case cases[] = {
            {nullptr, 1},
            {"split=remainder kc=1024 order=k", 0},
            {"split=grid", 0},
            {"split=grid mparts=2 kparts=2", 0},
            {"split=grid kparts=4 kc=512", 0},
            {"split=grid mparts=2 mr=4", 0},
            {"split=grid kparts=2 kc=256 order=k mr=5", 0},
    };
    for (const int M : {1, 3, 8, 13, 32}) {
        SCOPED_TRACE(M);
        std::vector<bfloat16_t> legacy;
        {
            PlanOverride none("", 0);
            ASSERT_EQ(run(M, legacy), status_t::success);
        }
        for (const Case &c : cases) {
            SCOPED_TRACE(c.block != nullptr ? c.block : "ksplit=1");
            PlanOverride plan(c.block, c.ksplit);
            const int runs_before
                    = mm::test_api::s_single_expert_executor_runs.load();
            std::vector<bfloat16_t> out;
            ASSERT_EQ(run(M, out), status_t::success);
            const bool ran = mm::test_api::s_single_expert_executor_runs.load()
                    > runs_before;
            const bool expect_run = mm::se_block_plan(1, M, N, K,
                    mm::custom_kernel::plan_pack_nr(K, N), int8 ? 4 : 2,
                    threads)
                                            .active();
            EXPECT_EQ(ran, expect_run) << "executor engagement";
            if (int8) {
                ASSERT_EQ(std::memcmp(legacy.data(), out.data(),
                                  legacy.size() * sizeof(bfloat16_t)),
                        0);
            } else {
                for (size_t i = 0; i < legacy.size(); ++i) {
                    const float a = static_cast<float>(legacy[i]);
                    const float b = static_cast<float>(out[i]);
                    ASSERT_NEAR(a, b, 1e-2f * (1.0f + std::fabs(a)))
                            << "at " << i;
                }
            }
        }
    }
}

TEST(TestSingleExpertExecutor, Int8PlansMatchLegacy) {
    CK_SKIP_IF_NO_INT8_ISA();
    run_plans_match_legacy(true, 2560, 4096, 32);
    run_plans_match_legacy(true, 1536, 3072, 32);
    run_plans_match_legacy(true, 3072, 1024, 32);
    run_plans_match_legacy(true, 2560, 4096, 24);
}

TEST(TestSingleExpertExecutor, Bf16PlansMatchLegacy) {
    if (!mm::custom_kernel::dispatch_supported()) GTEST_SKIP();
    run_plans_match_legacy(false, 2560, 4096, 32);
    run_plans_match_legacy(false, 1536, 3072, 32);
}

} // namespace
