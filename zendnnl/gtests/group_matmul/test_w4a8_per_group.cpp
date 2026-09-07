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

/// @file test_w4a8_per_group.cpp
/// @brief Grouped MoE W4A8 per-group matmul tests (sparse expert routing).
/// ALGO 3 W4A8 is always simulated (s4→s8) + aocl_dlp_blocked s8s8_sym_quant.
/// Full-N ALGOs 1/2/4/5 follow w4a8_runtime_algo (blocked = native s4).

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "group_matmul_test_helpers.hpp"
#include "gtest_utils.hpp"
#include "lowoha_operators/common/omp_thread_control.hpp"
#include "lowoha_operators/matmul/group_matmul/group_matmul_parallel_common.hpp"
#include "lowoha_operators/matmul/group_matmul/prepack/prepack.hpp"
#include "moe_test_utils.hpp"

namespace {

/// Build every expert with per-group s4 weights and a bf16 source with
/// dynamic per-token quantization, then drive `group_matmul_direct` with the
/// supplied per-expert active row counts (`rows[e] == 0` => inactive expert /
/// no routed tokens).  Routed experts are compared against a single-expert
/// reference matmul.
void run_w4a8_per_group_scenario(const std::string &label,
        const std::vector<int> &rows, uint64_t K, uint64_t N,
        uint64_t group_size, float src_range = 2.0f,
        matmul_algo_t algo = matmul_algo_t::aocl_dlp_blocked) {
    ASSERT_EQ(K % group_size, 0u)
            << label << ": K must be a multiple of group_size";
    const uint64_t G = K / group_size;
    ASSERT_GE(G, 1u) << label << ": need >= 1 group for per-group scaling";

    const int E = static_cast<int>(rows.size());
    ASSERT_GT(E, 0) << label;

    reset_grp_matmul_caches();

    const data_type_t out_dt = data_type_t::bf16;
    const data_type_t scale_dt = data_type_t::bf16;

    tensor_factory_t tf;

    std::vector<tensor_t> inp(E), wt(E), bias(E), out(E), out_ref(E);
    std::vector<int> active(E);

    const int64_t saved_seed = seed;

    for (int e = 0; e < E; ++e) {
        active[e] = rows[e];
        seed = saved_seed + 1 + static_cast<int64_t>(e);

        const uint64_t Mbuf = static_cast<uint64_t>(rows[e] > 0 ? rows[e] : 1);

        // ── Per-group s4 weight [K, N] with per-group {G, N} scale ──
        auto wei_scale = tf.uniform_dist_tensor({G, N}, scale_dt, 2.0);
        wt[e] = tf.uniform_dist_tensor(
                {K, N}, data_type_t::s4, 7.0, false, wei_scale);

        // ── bf16 source [Mbuf, K] with per-token {Mbuf, 1} dynamic scale ──
        auto src_scale = tf.zero_tensor({Mbuf, 1u}, scale_dt);
        inp[e] = tf.uniform_dist_tensor({Mbuf, K}, data_type_t::bf16, src_range,
                false, src_scale, tensor_t());

        bias[e] = tf.uniform_dist_tensor({1u, N}, out_dt, 2.0);
        out[e] = tf.zero_tensor({Mbuf, N}, out_dt);
        out_ref[e] = tf.zero_tensor({Mbuf, N}, out_dt);
    }
    seed = saved_seed;

    // ── Drive the grouped W4A8 path ──
    status_t st = group_matmul_kernel_test(inp, wt, bias, out, algo, 1.0f, 0.0f,
            /*moe_postop=*/nullptr,
            /*gated_act=*/nullptr,
            /*pack_format_b=*/ {}, active);
    ASSERT_EQ(st, status_t::success) << label << ": group_matmul_direct failed";

    // ── Compare every routed expert against single-expert reference ──
    const std::vector<post_op_type_t> ref_po;
    for (int e = 0; e < E; ++e) {
        if (rows[e] == 0) { continue; }
        std::vector<tensor_t> bin;
        status_t rst = matmul_kernel_test(inp[e], wt[e], bias[e], out_ref[e],
                ref_po, bin,
                /*use_LOWOHA=*/true, algo, 1.0f, 0.0f, true);
        ASSERT_EQ(rst, status_t::success)
                << label << ": reference failed (expert " << e << ")";
        bool expert_ok = true;
        // W4A8 tolerance: the s4→s8 conversion + bf16 dynamic quant introduces
        // more noise than pure INT8 per-group, so use a generous 128x epsilon.
        compare_tensor_2D_matrix(out[e], out_ref[e],
                static_cast<uint64_t>(rows[e]), N, K, rtol_bf16,
                128.0f * epsilon_bf16, expert_ok,
                /*enable_f32_relaxation=*/false, 1.0f, true);
        EXPECT_TRUE(expert_ok) << label << ": output mismatch on expert " << e
                               << " (rows=" << rows[e] << ")";
    }
}

void run_w4a8_cross_algo_scenario(const std::string &label,
        const std::vector<int> &rows, uint64_t K, uint64_t N,
        uint64_t group_size) {
    ASSERT_EQ(K % group_size, 0u)
            << label << ": K must be a multiple of group_size";
    const uint64_t G = K / group_size;
    const int E = static_cast<int>(rows.size());
    ASSERT_GT(E, 0) << label;

    reset_grp_matmul_caches();

    const matmul_algo_t algo = matmul_algo_t::aocl_dlp_blocked;
    const data_type_t scale_dt = data_type_t::bf16;
    const data_type_t out_dt = data_type_t::bf16;

    tensor_factory_t tf;
    std::vector<tensor_t> inp(E), wt(E), bias(E), out_a0(E), out_a1(E),
            out_a2(E), out_a3(E), out_a4(E), out_ref(E);
    std::vector<int> active(E);

    const int64_t saved_seed = seed;
    for (int e = 0; e < E; ++e) {
        active[e] = rows[e];
        seed = saved_seed + 1 + static_cast<int64_t>(e);
        const uint64_t Mbuf = static_cast<uint64_t>(rows[e] > 0 ? rows[e] : 1);

        auto wei_scale = tf.uniform_dist_tensor({G, N}, scale_dt, 2.0);
        wt[e] = tf.uniform_dist_tensor(
                {K, N}, data_type_t::s4, 7.0, false, wei_scale);
        auto src_scale = tf.zero_tensor({Mbuf, 1u}, scale_dt);
        inp[e] = tf.uniform_dist_tensor({Mbuf, K}, data_type_t::bf16, 2.0,
                false, src_scale, tensor_t());
        bias[e] = tf.zero_tensor({1u, N}, out_dt);
        out_a0[e] = tf.zero_tensor({Mbuf, N}, out_dt);
        out_a1[e] = tf.zero_tensor({Mbuf, N}, out_dt);
        out_a2[e] = tf.zero_tensor({Mbuf, N}, out_dt);
        out_a3[e] = tf.zero_tensor({Mbuf, N}, out_dt);
        out_a4[e] = tf.zero_tensor({Mbuf, N}, out_dt);
        out_ref[e] = tf.zero_tensor({Mbuf, N}, out_dt);
    }
    seed = saved_seed;

    status_t st;
    {
        moe_test_utils::AlgoEnvGuard g(1);
        reset_grp_matmul_caches();
        st = group_matmul_kernel_test(inp, wt, bias, out_a1, algo, 1.0f, 0.0f,
                nullptr, nullptr, {}, active);
    }
    ASSERT_EQ(st, status_t::success) << label << ": ALGO 1 failed";

    {
        moe_test_utils::AlgoEnvGuard g(0);
        reset_grp_matmul_caches();
        st = group_matmul_kernel_test(inp, wt, bias, out_a0, algo, 1.0f, 0.0f,
                nullptr, nullptr, {}, active);
    }
    ASSERT_EQ(st, status_t::success) << label << ": ALGO 0 failed";

    {
        moe_test_utils::AlgoEnvGuard g(2);
        reset_grp_matmul_caches();
        st = group_matmul_kernel_test(inp, wt, bias, out_a2, algo, 1.0f, 0.0f,
                nullptr, nullptr, {}, active);
    }
    ASSERT_EQ(st, status_t::success) << label << ": ALGO 2 failed";

    {
        moe_test_utils::AlgoEnvGuard g(3);
        reset_grp_matmul_caches();
        st = group_matmul_kernel_test(inp, wt, bias, out_a3, algo, 1.0f, 0.0f,
                nullptr, nullptr, {}, active);
    }
    ASSERT_EQ(st, status_t::success) << label << ": ALGO 3 failed";

    {
        moe_test_utils::AlgoEnvGuard g(4);
        reset_grp_matmul_caches();
        st = group_matmul_kernel_test(inp, wt, bias, out_a4, algo, 1.0f, 0.0f,
                nullptr, nullptr, {}, active);
    }
    ASSERT_EQ(st, status_t::success) << label << ": ALGO 4 failed";

    {
        moe_test_utils::AlgoEnvGuard g(1);
        const std::vector<post_op_type_t> ref_po;
        for (int e = 0; e < E; ++e) {
            if (rows[e] == 0) { continue; }
            std::vector<tensor_t> bin;
            st = matmul_kernel_test(inp[e], wt[e], bias[e], out_ref[e], ref_po,
                    bin, true, algo, 1.0f, 0.0f, true);
            ASSERT_EQ(st, status_t::success) << label << ": ref failed e=" << e;
        }
    }

    const float abs_tol = 128.0f * epsilon_bf16;
    for (int e = 0; e < E; ++e) {
        if (rows[e] == 0) { continue; }
        const uint64_t M_e = static_cast<uint64_t>(rows[e]);
        bool ok = true;
        compare_tensor_2D_matrix(out_a1[e], out_ref[e], M_e, N, K, rtol_bf16,
                abs_tol, ok, false, 1.0f, true);
        EXPECT_TRUE(ok) << label << ": ALGO 1 vs ref mismatch (e=" << e << ")";

        ok = true;
        compare_tensor_2D_matrix(out_a0[e], out_a1[e], M_e, N, K, rtol_bf16,
                abs_tol, ok, false, 1.0f, true);
        EXPECT_TRUE(ok) << label << ": ALGO 0 vs 1 mismatch (e=" << e << ")";

        ok = true;
        compare_tensor_2D_matrix(out_a2[e], out_a1[e], M_e, N, K, rtol_bf16,
                abs_tol, ok, false, 1.0f, true);
        EXPECT_TRUE(ok) << label << ": ALGO 2 vs 1 mismatch (e=" << e << ")";

        ok = true;
        compare_tensor_2D_matrix(out_a3[e], out_a1[e], M_e, N, K, rtol_bf16,
                abs_tol, ok, false, 1.0f, true);
        EXPECT_TRUE(ok) << label << ": ALGO 3 vs 1 mismatch (e=" << e << ")";

        ok = true;
        compare_tensor_2D_matrix(out_a4[e], out_a1[e], M_e, N, K, rtol_bf16,
                abs_tol, ok, false, 1.0f, true);
        EXPECT_TRUE(ok) << label << ": ALGO 4 vs 1 mismatch (e=" << e << ")";
    }
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════
// MoE routing-pattern scenarios (15 experts, sparse active sets)
// ═══════════════════════════════════════════════════════════════════════

// Headline scenario: 15 experts, only 6 routed (interleaved) — one MoE decode
// iteration that fires 6 of 15 experts.
TEST(GroupMatmulW4A8PerGroup, FifteenExpertsSixActiveInterleavedBF16) {
    std::vector<int> rows(15, 0);
    for (int e : {1, 3, 5, 8, 11, 14})
        rows[e] = 32;
    run_w4a8_per_group_scenario("15/6 interleaved bf16", rows, /*K=*/128,
            /*N=*/64, /*group_size=*/32);
}

// First 6 experts routed.
TEST(GroupMatmulW4A8PerGroup, FifteenExpertsSixActiveContiguousFirstBF16) {
    std::vector<int> rows(15, 0);
    for (int e = 0; e < 6; ++e) {
        rows[e] = 24;
    }
    run_w4a8_per_group_scenario("15/6 first-6 bf16", rows, 128, 48, 32);
}

// Last 6 experts routed => expert 0 is inactive (M[0] == 0).  Stresses the
// prepack representative-expert selection (must skip M[i] <= 0).
TEST(GroupMatmulW4A8PerGroup, FifteenExpertsSixActiveContiguousLastBF16) {
    std::vector<int> rows(15, 0);
    for (int e = 9; e < 15; ++e) {
        rows[e] = 16;
    }
    run_w4a8_per_group_scenario("15/6 last-6 bf16", rows, 256, 32, 32);
}

// Non-uniform token counts across the routed experts (incl. M == 1) — stresses
// per-token dynamic quant over ragged M and the row-level grouped quant
// scheduler.
TEST(GroupMatmulW4A8PerGroup, FifteenExpertsVariedTokenCountsBF16) {
    std::vector<int> rows(15, 0);
    const int idx[6] = {0, 2, 4, 6, 9, 13};
    const int act[6] = {1, 2, 3, 5, 8, 13};
    for (int j = 0; j < 6; ++j) {
        rows[idx[j]] = act[j];
    }
    run_w4a8_per_group_scenario("15 varied tokens bf16", rows, 128, 64, 32);
}

// Single routed expert in the middle of the inactive set.
TEST(GroupMatmulW4A8PerGroup, FifteenExpertsSingleActiveBF16) {
    std::vector<int> rows(15, 0);
    rows[7] = 8;
    run_w4a8_per_group_scenario("15/1 single active bf16", rows, 96, 80, 32);
}

// Dense routing: every expert fires (no inactive experts).
TEST(GroupMatmulW4A8PerGroup, FifteenExpertsAllActiveBF16) {
    std::vector<int> rows(15, 12);
    run_w4a8_per_group_scenario("15/15 all active bf16", rows, 128, 64, 32);
}

// Degenerate routing: no expert fires (all M == 0).  The call must succeed
// and produce no output.
TEST(GroupMatmulW4A8PerGroup, FifteenExpertsNoneActiveBF16) {
    std::vector<int> rows(15, 0);
    run_w4a8_per_group_scenario("15/0 none active bf16", rows, 128, 64, 32);
}

// ═══════════════════════════════════════════════════════════════════════
// Group-size variations (W4A8 supports flexible group sizes, not just 32)
// ═══════════════════════════════════════════════════════════════════════

// group_size = 128 (Qwen3-style, K=4096)
TEST(GroupMatmulW4A8PerGroup, GroupSize128Qwen3BF16) {
    std::vector<int> rows(8, 0);
    for (int e : {0, 2, 4, 7})
        rows[e] = 7;
    run_w4a8_per_group_scenario("gs128 qwen3 bf16", rows, 4096, 4096, 128);
}

// group_size = 64
TEST(GroupMatmulW4A8PerGroup, GroupSize64BF16) {
    std::vector<int> rows(15, 0);
    for (int e : {1, 5, 10, 14})
        rows[e] = 16;
    run_w4a8_per_group_scenario("gs64 bf16", rows, 256, 128, 64);
}

// group_size=K => single weight group; M=1 decode boundary.
TEST(GroupMatmulW4A8PerGroup, GroupSizeEqualsKDecodeM1BF16) {
    std::vector<int> rows(8, 1);
    run_w4a8_per_group_scenario("gs_eq_k decode M1 bf16", rows, /*K=*/128,
            /*N=*/64, /*group_size=*/128);
}

TEST(GroupMatmulW4A8PerGroup, AoclDlpSixActiveInterleavedBF16) {
    std::vector<int> rows(15, 0);
    for (int e : {1, 3, 5, 8, 11, 14})
        rows[e] = 32;
    run_w4a8_per_group_scenario("aocl_dlp 15/6 interleaved bf16", rows,
            /*K=*/128, /*N=*/64, /*group_size=*/32, /*src_range=*/2.0f,
            matmul_algo_t::aocl_dlp);
}

// ═══════════════════════════════════════════════════════════════════════
// Cross-algo accuracy coverage
// ═══════════════════════════════════════════════════════════════════════

TEST(GroupMatmulW4A8PerGroup, CrossAlgoSmallDecodeBF16) {
    run_w4a8_cross_algo_scenario("small", std::vector<int>(4, 32), 128, 64, 32);
}

TEST(GroupMatmulW4A8PerGroup, CrossAlgoMidRangeBF16) {
    run_w4a8_cross_algo_scenario("mid", std::vector<int>(10, 128), 128, 64, 32);
}

TEST(GroupMatmulW4A8PerGroup, CrossAlgoSingleTokenBF16) {
    run_w4a8_cross_algo_scenario("M1", std::vector<int>(20, 1), 128, 64, 32);
}

TEST(GroupMatmulW4A8PerGroup, CrossAlgoSquareBF16) {
    run_w4a8_cross_algo_scenario(
            "square", std::vector<int>(8, 16), 128, 128, 32);
}

TEST(GroupMatmulW4A8PerGroup, CrossAlgoQwen3DecodeBF16) {
    run_w4a8_cross_algo_scenario(
            "qwen3", std::vector<int>(8, 7), 4096, 4096, 128);
}

TEST(GroupMatmulW4A8PerGroup, CrossAlgoQwen3ManyOpsBF16) {
    run_w4a8_cross_algo_scenario(
            "qwen16", std::vector<int>(16, 1), 4096, 4096, 128);
}

// ═══════════════════════════════════════════════════════════════════════
// ALGO 3 (N-tile) direct validation
// ═══════════════════════════════════════════════════════════════════════

// Cross-algo comparison: ALGO 1 and ALGO 3 must produce identical
// output for the same W4A8 input (ALGO 3 now runs natively, not fallback).
TEST(GroupMatmulW4A8PerGroup, Algo3MatchesAlgo1BF16) {
    const int E = 15;
    const uint64_t K = 128, N = 64, group_size = 32;
    const uint64_t G = K / group_size;
    const data_type_t scale_dt = data_type_t::bf16;
    const data_type_t out_dt = data_type_t::bf16;

    reset_grp_matmul_caches();

    tensor_factory_t tf;
    std::vector<tensor_t> inp(E), wt(E), bias(E), out_a1(E), out_a3(E);
    std::vector<int> active(E);

    std::vector<int> rows(E, 0);
    for (int e : {1, 3, 5, 8, 11, 14})
        rows[e] = 32;

    const int64_t saved_seed = seed;
    for (int e = 0; e < E; ++e) {
        active[e] = rows[e];
        seed = saved_seed + 1 + static_cast<int64_t>(e);
        const uint64_t Mbuf = static_cast<uint64_t>(rows[e] > 0 ? rows[e] : 1);

        auto ws = tf.uniform_dist_tensor({G, N}, scale_dt, 2.0);
        wt[e] = tf.uniform_dist_tensor({K, N}, data_type_t::s4, 7.0, false, ws);
        auto ss = tf.zero_tensor({Mbuf, 1u}, scale_dt);
        inp[e] = tf.uniform_dist_tensor(
                {Mbuf, K}, data_type_t::bf16, 2.0, false, ss, tensor_t());
        bias[e] = tf.zero_tensor({1u, N}, out_dt);
        out_a1[e] = tf.zero_tensor({Mbuf, N}, out_dt);
        out_a3[e] = tf.zero_tensor({Mbuf, N}, out_dt);
    }
    seed = saved_seed;

    const matmul_algo_t algo = matmul_algo_t::aocl_dlp_blocked;
    status_t s;

    {
        moe_test_utils::AlgoEnvGuard g(1);
        reset_grp_matmul_caches();
        s = group_matmul_kernel_test(inp, wt, bias, out_a1, algo, 1.0f, 0.0f,
                nullptr, nullptr, {}, active);
    }
    ASSERT_EQ(s, status_t::success) << "ALGO 1 failed";

    {
        moe_test_utils::AlgoEnvGuard g(3);
        reset_grp_matmul_caches();
        s = group_matmul_kernel_test(inp, wt, bias, out_a3, algo, 1.0f, 0.0f,
                nullptr, nullptr, {}, active);
    }
    ASSERT_EQ(s, status_t::success) << "ALGO 3 failed";

    const float abs_tol = 128.0f * epsilon_bf16;
    for (int e = 0; e < E; ++e) {
        if (rows[e] == 0) { continue; }
        bool ok = true;
        compare_tensor_2D_matrix(out_a3[e], out_a1[e],
                static_cast<uint64_t>(rows[e]), N, K, rtol_bf16, abs_tol, ok,
                false, 1.0f, true);
        EXPECT_TRUE(ok) << "ALGO 3 vs ALGO 1 mismatch on expert " << e;
    }
}

// ALGO 3 decode test: M=1 per expert (typical MoE decode, N-tile dominant).
TEST(GroupMatmulW4A8PerGroup, Algo3DecodeM1BF16) {
    moe_test_utils::AlgoEnvGuard algo3(3);
    std::vector<int> rows(8, 1);
    run_w4a8_per_group_scenario("algo3 decode M1", rows, 256, 128, 64);
}

// ALGO 3 with larger shapes (Qwen3-like).
TEST(GroupMatmulW4A8PerGroup, Algo3Qwen3DecodeBF16) {
    moe_test_utils::AlgoEnvGuard algo3(3);
    std::vector<int> rows(8, 0);
    for (int e : {0, 2, 4, 7})
        rows[e] = 7;
    run_w4a8_per_group_scenario("algo3 qwen3 decode", rows, 4096, 4096, 128);
}

// ═══════════════════════════════════════════════════════════════════════
// Prepack W4A8 cache warming validation
// ═══════════════════════════════════════════════════════════════════════

// Validate that prepack warms the W4A8 cache for ALL experts (including
// inactive ones) so a later decode iteration that routes to a currently-cold
// expert pays no first-fire reorder spike.
TEST(GroupMatmulW4A8PerGroup, PrepackWarmsAllExpertsBF16) {
    namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;
    moe_test_utils::LastInvocationCaptureGuard prepack_capture;
    prepack::clear_fingerprint_cache_for_test();
    prepack::test_api::clear_last_invocation_stats();

    std::vector<int> rows(15, 0);
    for (int e : {1, 3, 5, 8, 11, 14})
        rows[e] = 32; // 6 of 15 routed
    run_w4a8_per_group_scenario("15/6 prepack warm", rows, /*K=*/128,
            /*N=*/64, /*group_size=*/32);

    auto stats = prepack::test_api::get_last_invocation_stats();
    ASSERT_TRUE(stats.valid) << "prepack must fire for the W4A8 per-group call";
    if (moe_test_utils::k_grp_matmul_aocl_dlp_compiled) {
        // The prepack warmer should attempt all 15 experts (total_attempted >= 15),
        // not just the 6 routed ones.
        EXPECT_GE(stats.aocl.total_attempted, 15)
                << "prepack must warm all 15 experts' W4A8 weight cache (not "
                   "just "
                   "the "
                   "6 routed ones) to eliminate first-fire reorder spikes on "
                   "rotating-experts MoE patterns";
        EXPECT_GT(stats.aocl.total_attempted - stats.aocl.skipped_invalid, 0)
                << "with AOCL-DLP, at least one expert W4A8 cache entry must "
                   "be warmed";
    } else {
        // Without AOCL-DLP the inner kernel resolves to reference, so prepack
        // skips AOCL W4A8 warm (no cache to populate).  Correctness is
        // validated above by run_w4a8_per_group_scenario.
        EXPECT_EQ(stats.aocl.total_attempted, 0)
                << "without AOCL-DLP no AOCL W4A8 cache warm is attempted";
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Wide dynamic range (stress dynamic quantization clipping)
// ═══════════════════════════════════════════════════════════════════════

// Source range 25.0 — exercises the dynamic bf16→s8 quantization with large
// activations that stress scale computation and potential clipping.
TEST(GroupMatmulW4A8PerGroup, WideRangeSrcBF16) {
    std::vector<int> rows(8, 0);
    for (int e : {0, 2, 4, 7})
        rows[e] = 7;
    run_w4a8_per_group_scenario("wide src range", rows, 4096, 4096, 128,
            /*src_range=*/25.0f);
}

// Source range 20.0 — exercises the dynamic bf16→s8 quantization with larger
// activations that stress scale computation (typical of layer-norm outputs
// with outlier channels).
TEST(GroupMatmulW4A8PerGroup, ExtremeRangeSrcBF16) {
    std::vector<int> rows(15, 0);
    for (int e : {1, 3, 5, 8, 11, 14})
        rows[e] = 16;
    run_w4a8_per_group_scenario("extreme src range", rows, 256, 128, 64,
            /*src_range=*/20.0f);
}

// ═══════════════════════════════════════════════════════════════════════
// Fused MoE W4A8: ALGO 3 vs ALGO 1 accuracy on the full
//   Op1(gate+up) → silu_and_mul → Op2(down_proj) pipeline
// ═══════════════════════════════════════════════════════════════════════

// Reproduces the vLLM accuracy failure where ALGO 3's grouped DQ
// pre-pass stored Op2's per-token src_scale in the hoisted state
// rather than in params_down[e], leaving params_down[e].src_scale.buff
// null.  The test drives the raw group_matmul_direct API (not the
// tensor_factory_t wrapper) so it exercises the exact same fused-MoE
// code path as the vLLM/Zentorch production caller.
//
// Shape: 8 experts, M=128, K=2048, N_gate_up=1024, dim=512,
//        K_down=512, N_down=2048, group_size=128.
TEST(GroupMatmulW4A8PerGroup, FusedMoeAlgo3VsAlgo1BF16) {
    using namespace zendnnl::lowoha::matmul;
    using bfloat16_t = zendnnl::common::bfloat16_t;

    constexpr int E = 8;
    constexpr int M = 128;
    constexpr int K = 2048;
    constexpr int N_GATE_UP = 1024;
    constexpr int DIM = N_GATE_UP / 2; // 512
    constexpr int K_DOWN = DIM; // 512
    constexpr int N_DOWN = 2048;
    constexpr int GROUP_SIZE = 128;
    constexpr int NUM_GROUPS_W1 = K / GROUP_SIZE; // 16
    constexpr int NUM_GROUPS_W2 = K_DOWN / GROUP_SIZE; // 4

    reset_grp_matmul_caches();

    // ── Allocate per-expert buffers ──────────────────────────────────────
    std::vector<std::vector<uint16_t>> src_buf(E);
    std::vector<std::vector<int8_t>> wei_gu_packed(E);
    std::vector<std::vector<uint16_t>> wei_gu_scale(E);
    std::vector<std::vector<uint16_t>> src_scale_buf(E);

    std::vector<std::vector<int8_t>> wei_down_packed(E);
    std::vector<std::vector<uint16_t>> wei_down_scale(E);

    auto f32_to_bf16 = [](float v) -> uint16_t {
        return static_cast<uint16_t>(bfloat16_t::f32_to_bf16_val(v));
    };
    auto bf16_to_f32 = [](uint16_t bits) -> float {
        return static_cast<float>(bfloat16_t::from_bits(bits));
    };

    std::mt19937 rng(42);
    auto rand_bf16 = [&]() -> uint16_t {
        return f32_to_bf16(
                std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng));
    };
    auto rand_s4_byte = [&]() -> int8_t {
        int lo = std::uniform_int_distribution<int>(-7, 7)(rng);
        int hi = std::uniform_int_distribution<int>(-7, 7)(rng);
        return static_cast<int8_t>((hi & 0x0F) << 4 | (lo & 0x0F));
    };

    for (int i = 0; i < E; ++i) {
        // bf16 source [M, K]
        src_buf[i].resize(static_cast<size_t>(M) * K);
        for (auto &v : src_buf[i])
            v = rand_bf16();

        // s4 gate+up weight — packed [N_GATE_UP, K] (transB=T → stored [N,K])
        wei_gu_packed[i].resize(static_cast<size_t>((N_GATE_UP * K + 1) / 2));
        for (auto &v : wei_gu_packed[i])
            v = rand_s4_byte();

        // per-group weight scale {NUM_GROUPS_W1, N_GATE_UP}
        wei_gu_scale[i].resize(static_cast<size_t>(NUM_GROUPS_W1) * N_GATE_UP);
        for (auto &v : wei_gu_scale[i])
            v = f32_to_bf16(
                    std::uniform_real_distribution<float>(0.5f, 2.0f)(rng));

        // per-token src_scale {M, 1} — zero-initialized; runtime computes
        src_scale_buf[i].assign(static_cast<size_t>(M), 0u);

        // s4 down_proj weight — packed [N_DOWN, K_DOWN] (transB=T)
        wei_down_packed[i].resize(
                static_cast<size_t>((N_DOWN * K_DOWN + 1) / 2));
        for (auto &v : wei_down_packed[i])
            v = rand_s4_byte();

        // per-group down weight scale {NUM_GROUPS_W2, N_DOWN}
        wei_down_scale[i].resize(static_cast<size_t>(NUM_GROUPS_W2) * N_DOWN);
        for (auto &v : wei_down_scale[i])
            v = f32_to_bf16(
                    std::uniform_real_distribution<float>(0.5f, 2.0f)(rng));
    }

    // ── Build API vectors (common to both ALGO runs) ─────────────────────
    std::vector<char> layouts(E, 'r');
    std::vector<bool> transAs(E, false);
    std::vector<bool> transBs(E, true);
    std::vector<float> alphas(E, 1.f), betas(E, 0.f);
    std::vector<bool> wconst(E, true);

    std::vector<int> Ms(E, M), Ns(E, N_GATE_UP), Ks(E, K);
    std::vector<int> ldas(E, K);
    std::vector<int> ldbs(E, K); // transB=T → ldb = K

    std::vector<const void *> sp(E), wp(E);
    std::vector<const void *> bp(E, nullptr);
    for (int i = 0; i < E; ++i) {
        sp[i] = src_buf[i].data();
        wp[i] = wei_gu_packed[i].data();
    }

    // Internal-alloc: dst all nullptr, ldc zeros
    std::vector<void *> dp(E, nullptr);
    std::vector<int> ldcs(E, 0);

    // matmul_params: W4A8 dynamic quant
    auto build_params = [&]() {
        std::vector<matmul_params> params(E);
        for (int i = 0; i < E; ++i) {
            params[i].dtypes.src = data_type_t::bf16;
            params[i].dtypes.wei = data_type_t::s4;
            params[i].dtypes.dst = data_type_t::bf16;
            params[i].dtypes.compute = data_type_t::s8;
            params[i].dynamic_quant = true;

            params[i].quant_params.src_scale.buff = src_scale_buf[i].data();
            params[i].quant_params.src_scale.dt = data_type_t::bf16;
            params[i].quant_params.src_scale.dims = {M, 1};

            params[i].quant_params.wei_scale.buff = wei_gu_scale[i].data();
            params[i].quant_params.wei_scale.dt = data_type_t::bf16;
            params[i].quant_params.wei_scale.dims = {NUM_GROUPS_W1, N_GATE_UP};
        }
        return params;
    };

    // Gated activation: silu_and_mul
    grp_matmul_gated_act_params act {};
    act.act = grp_matmul_gated_act_t::silu_and_mul;

    // Fused MoE params (Op2 = down_proj)
    auto build_fused = [&](std::vector<void *> &dst_down_vec,
                               std::vector<int> &ldc_down_vec) {
        grp_matmul_fused_moe_params fused {};
        fused.N_down.resize(E, N_DOWN);
        fused.ldb_down.resize(E, K_DOWN); // transB=T → ldb_down = K_down
        fused.bias_down.resize(E, nullptr);
        fused.down_weight.resize(E);
        fused.down_scale.resize(E);
        for (int i = 0; i < E; ++i) {
            fused.down_weight[i] = wei_down_packed[i].data();

            grp_matmul_fused_moe_params::down_weight_quant_t ds;
            ds.buff = wei_down_scale[i].data();
            ds.dt = data_type_t::bf16;
            ds.dims = {NUM_GROUPS_W2, N_DOWN};
            fused.down_scale[i] = ds;
        }
        // Caller-allocated Op2 output so we can compare across algo runs
        fused.dst_down = dst_down_vec;
        fused.ldc_down = ldc_down_vec;
        return fused;
    };

    // ── Run ALGO 1 ───────────────────────────────────────────────────────
    std::vector<std::vector<uint16_t>> out_a1_buf(E);
    std::vector<void *> dst_a1(E);
    std::vector<int> ldc_a1(E, N_DOWN);
    for (int i = 0; i < E; ++i) {
        out_a1_buf[i].assign(static_cast<size_t>(M) * N_DOWN, 0u);
        dst_a1[i] = out_a1_buf[i].data();
    }

    {
        moe_test_utils::AlgoEnvGuard g(1);
        reset_grp_matmul_caches();
        auto pf = build_params();
        auto fused = build_fused(dst_a1, ldc_a1);
        status_t s = group_matmul_direct(layouts, transAs, transBs, Ms, Ns, Ks,
                alphas, sp, ldas, wp, ldbs, bp, betas, dp, ldcs, wconst, pf,
                nullptr, &act, &fused);
        ASSERT_EQ(s, status_t::success) << "Fused MoE W4A8 ALGO 1 failed";
    }

    // ── Run ALGO 3 ───────────────────────────────────────────────────────
    std::vector<std::vector<uint16_t>> out_a3_buf(E);
    std::vector<void *> dst_a3(E);
    std::vector<int> ldc_a3(E, N_DOWN);
    for (int i = 0; i < E; ++i) {
        out_a3_buf[i].assign(static_cast<size_t>(M) * N_DOWN, 0u);
        dst_a3[i] = out_a3_buf[i].data();
    }

    {
        moe_test_utils::AlgoEnvGuard g(3);
        reset_grp_matmul_caches();
        auto pf = build_params();
        auto fused = build_fused(dst_a3, ldc_a3);
        status_t s = group_matmul_direct(layouts, transAs, transBs, Ms, Ns, Ks,
                alphas, sp, ldas, wp, ldbs, bp, betas, dp, ldcs, wconst, pf,
                nullptr, &act, &fused);
        ASSERT_EQ(s, status_t::success) << "Fused MoE W4A8 ALGO 3 failed";
    }

    // ── Compare Op2 output: ALGO 3 vs ALGO 1 ────────────────────────────
    // Fused W4A8 tolerance: the full pipeline (Op1 W4A8 quant → silu_and_mul
    // nonlinear activation → Op2 W4A8 quant) amplifies per-element noise
    // multiplicatively.  ALGO 1 (sequential full-N) and ALGO 3 (N-tile) use
    // different AOCL blocking layouts, producing slightly different s32→bf16
    // rounding.  Use 50% relative + generous absolute floor — this is a
    // "same ballpark, no garbage" gate, not a precision-tracking bound.
    const float fused_rel = 0.50f;
    const float fused_abs = 65536.0f;
    for (int e = 0; e < E; ++e) {
        bool expert_ok = true;
        for (int r = 0; r < M && expert_ok; ++r) {
            for (int c = 0; c < N_DOWN && expert_ok; ++c) {
                const size_t idx = static_cast<size_t>(r) * N_DOWN + c;
                const float v1 = bf16_to_f32(out_a1_buf[e][idx]);
                const float v3 = bf16_to_f32(out_a3_buf[e][idx]);
                const float err = std::fabs(v3 - v1);
                const float tol = std::fabs(v1) * fused_rel + fused_abs;
                if (err > tol) {
                    EXPECT_LE(err, tol)
                            << "ALGO 3 vs ALGO 1 Op2 output mismatch on expert "
                            << e << " row=" << r << " col=" << c
                            << " (algo1=" << v1 << " algo3=" << v3
                            << " err=" << err << " tol=" << tol << ")";
                    expert_ok = false;
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Prepack OFF coverage (validates dispatch-level L1 path independently)
// ═══════════════════════════════════════════════════════════════════════

// ALGO 3 with prepack disabled — validates that the dispatch-level
// w4a8_populate_plain_s8_cache fills L1 and runtime per-tile L2 reorders
// work without any prepack pre-warming.
TEST(GroupMatmulW4A8PerGroup, Algo3PrepackOffBF16) {
    moe_test_utils::AlgoEnvGuard algo3(3);
    moe_test_utils::EnvVarGuard prepack_off("ZENDNNL_GRP_MATMUL_PREPACK", "0");
    std::vector<int> rows(8, 0);
    for (int e : {0, 2, 4, 7})
        rows[e] = 7;
    run_w4a8_per_group_scenario("algo3 prepack-off", rows, 4096, 4096, 128);
}

// ALGO 1 with prepack disabled — validates lazy L2 reorder inside
// w4a8ReorderAndCacheWeightsAocl without prepack pre-warming.
TEST(GroupMatmulW4A8PerGroup, Algo1PrepackOffBF16) {
    moe_test_utils::AlgoEnvGuard algo1(1);
    moe_test_utils::EnvVarGuard prepack_off("ZENDNNL_GRP_MATMUL_PREPACK", "0");
    std::vector<int> rows(8, 0);
    for (int e : {0, 2, 4, 7})
        rows[e] = 7;
    run_w4a8_per_group_scenario("algo1 prepack-off", rows, 4096, 4096, 128);
}

// ═══════════════════════════════════════════════════════════════════════
// vLLM shape reproduction (triggers AOCL illegal-value edge case)
// ═══════════════════════════════════════════════════════════════════════

// Reproduces the Qwen3-30B-A3B vLLM deployment shape that triggers AOCL
// "illegal value" on per-tile reorder.  K=2048, N=1024, group_size=128
// produces n_tile=172 with stable=6 threads — the exact parameters that
// hit the AOCL s8s8s32os32_sym_quant validation.
TEST(GroupMatmulW4A8PerGroup, Algo3VllmQwen3ShapeBF16) {
    moe_test_utils::AlgoEnvGuard algo3(3);
    std::vector<int> rows(8, 0);
    for (int e : {0, 1, 2, 3, 4, 5, 6, 7})
        rows[e] = 4096;
    run_w4a8_per_group_scenario(
            "vllm qwen3 K2048 N1024", rows, 2048, 1024, 128);
}

// Same shape with ALGO 1 as a reference — if ALGO 1 passes but ALGO 3 fails,
// the bug is in the per-tile N-tile path specifically.
TEST(GroupMatmulW4A8PerGroup, Algo1VllmQwen3ShapeBF16) {
    moe_test_utils::AlgoEnvGuard algo1(1);
    std::vector<int> rows(8, 0);
    for (int e : {0, 1, 2, 3, 4, 5, 6, 7})
        rows[e] = 4096;
    run_w4a8_per_group_scenario(
            "vllm qwen3 algo1 K2048 N1024", rows, 2048, 1024, 128);
}
