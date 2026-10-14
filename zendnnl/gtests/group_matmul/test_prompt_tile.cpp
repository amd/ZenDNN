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

/// @file test_prompt_tile.cpp
/// @brief ALGO 3 prompt tile executor: planner properties, topology map,
///        bitwise A/B against the round executor, and engagement gates.

#include <gtest/gtest.h>

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "lowoha_operators/matmul/group_matmul/custom_kernel/dispatch.hpp"
#include "lowoha_operators/matmul/group_matmul/prompt_tile/prompt_tile_executor.hpp"
#include "lowoha_operators/matmul/group_matmul/prompt_tile/prompt_tile_planner.hpp"
#include "lowoha_operators/matmul/group_matmul/prompt_tile/prompt_tile_topology.hpp"
#include "lowoha_operators/matmul/group_matmul/prompt_tile/prompt_tile_tunables.hpp"
#include "moe_test_utils.hpp"

// GTEST_SKIP returns from the enclosing function, so call these from TEST
// bodies.
#define SKIP_WITHOUT_BF16_CUSTOM_KERNEL() \
    do { \
        if (!zendnnl::lowoha::matmul::custom_kernel::dispatch_supported()) { \
            GTEST_SKIP() << "the BF16 custom kernel needs AVX-512 BF16"; \
        } \
    } while (0)

#define SKIP_WITHOUT_F16_CUSTOM_KERNEL() \
    do { \
        if (!zendnnl::lowoha::matmul::custom_kernel::avx512f16_available()) { \
            GTEST_SKIP() << "the FP16 custom kernel needs AVX-512 FP16"; \
        } \
    } while (0)

namespace {

namespace pt = zendnnl::lowoha::matmul::prompt_tile;
using moe_test_utils::bfloat16_t;
using moe_test_utils::data_type_t;
using moe_test_utils::float16_t;
using zendnnl::lowoha::matmul::grp_matmul_gated_act_t;

/// Sets one prompt-tile knob for the scope.
using PtOverride = pt::test_api::ScopedOverride;

// ── Planner ─────────────────────────────────────────────────────────

/// Every output cell of every active expert is covered by exactly one
/// tile, tiles honour the CK / bit-exactness contracts, and the team
/// ranges partition the tile list.
void check_plan(const std::vector<int> &M, const std::vector<int> &N,
        const std::vector<int> &K, const pt::PlannerConfig &cfg,
        const std::vector<int> &team_size, const pt::Plan &plan) {
    const int num_ops = static_cast<int>(M.size());
    std::vector<std::vector<int>> grid(num_ops);
    for (int e = 0; e < num_ops; ++e) {
        if (M[e] > 0)
            grid[e].assign(static_cast<size_t>(M[e]) * (N[e] / cfg.pack_nr), 0);
    }
    for (const auto &t : plan.tiles) {
        ASSERT_GE(t.e, 0);
        ASSERT_LT(t.e, num_ops);
        ASSERT_GT(M[t.e], 0) << "tile for an inactive expert";
        ASSERT_GE(t.m0, 0);
        ASSERT_GT(t.rows, 0);
        ASSERT_LE(t.m0 + t.rows, M[t.e]);
        ASSERT_EQ(t.n0 % cfg.pack_nr, 0);
        ASSERT_GT(t.ncols, 0);
        ASSERT_EQ(t.ncols % cfg.pack_nr, 0);
        ASSERT_LE(t.n0 + t.ncols, N[t.e]);
        ASSERT_TRUE(pt::min_call_rows(t.rows, cfg.max_mr) >= cfg.min_mr
                || t.rows == M[t.e])
                << "e=" << t.e << " m0=" << t.m0 << " rows=" << t.rows
                << " M=" << M[t.e] << " max_mr=" << cfg.max_mr;
        const int nb = N[t.e] / cfg.pack_nr;
        for (int r = t.m0; r < t.m0 + t.rows; ++r)
            for (int c = t.n0 / cfg.pack_nr; c < (t.n0 + t.ncols) / cfg.pack_nr;
                    ++c)
                ++grid[t.e][static_cast<size_t>(r) * nb + c];
    }
    for (int e = 0; e < num_ops; ++e)
        for (size_t i = 0; i < grid[e].size(); ++i)
            ASSERT_EQ(grid[e][i], 1) << "e=" << e << " cell=" << i;

    const int n_teams = static_cast<int>(team_size.size());
    ASSERT_EQ(plan.team_begin.size(), static_cast<size_t>(n_teams) + 1);
    EXPECT_EQ(plan.team_begin.front(), 0);
    EXPECT_EQ(plan.team_begin.back(), static_cast<int>(plan.tiles.size()));
    for (int t = 0; t < n_teams; ++t)
        EXPECT_LE(plan.team_begin[t], plan.team_begin[t + 1]);
}

std::vector<int> skewed_m(int num_ops, int max_m, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<int> M(num_ops);
    for (auto &m : M) {
        const double x = u(rng);
        m = static_cast<int>(max_m * x * x * x); // long tail, many tiny / zero
    }
    return M;
}

TEST(PromptTilePlanner, CoversEveryOutputCellOnce) {
    const std::vector<std::vector<int>> dists = {
            {0, 1, 2, 3, 4, 5, 7, 8, 9, 12, 13, 33, 64, 65, 100, 257, 1000},
            {4097},
            {0, 0, 0},
            skewed_m(64, 2000, 7),
            skewed_m(256, 800, 11),
    };
    struct Variant {
        int mb, ns, order, min_tiles, threads;
        bool l3sb;
        int64_t l3;
        std::vector<int> teams;
    };
    const std::vector<Variant> variants = {
            {0, 0, 0, 8, 32, true, int64_t(32) << 20, {8, 8, 8, 8}},
            {8, 0, 0, 8, 32, true, int64_t(32) << 20, {8, 8, 8, 8}},
            {1, 32, 1, 1, 7, true, int64_t(32) << 20, {3, 3, 1}},
            {100, 96, 0, 64, 1, true, int64_t(32) << 20, {1}},
            {16, 1000, 1, 8, 6, false, int64_t(32) << 20, {6}},
            {0, 0, 0, 8, 32, true, int64_t(64) << 10, {8, 8, 8, 8}},
            {0, 0, 1, 4, 32, true, int64_t(256) << 10, {8, 8, 8, 2, 6}},
            {0, 0, 2, 8, 32, true, int64_t(32) << 20, {8, 8, 8, 8}},
            {24, 64, 2, 2, 12, true, int64_t(1) << 20, {6, 6}},
    };
    // Kernel facts: BF16 at pack NR 32 and 64, FP16 at pack NR 32, INT8,
    // and W4A8 with groups of 32 and bf16 scales (K = 640).
    struct Kernel {
        int max_mr, row_align, min_mr, a_bytes, wei_bits;
        int64_t extra;
    };
    const std::vector<Kernel> kernels
            = {{8, 8, 4, 2, 16, 0}, {6, 8, 4, 2, 16, 0}, {8, 8, 1, 2, 16, 0},
                    {12, 12, 1, 1, 8, 4}, {6, 6, 1, 1, 4, 120}};
    for (const auto &M : dists) {
        const int num_ops = static_cast<int>(M.size());
        const std::vector<int> N(num_ops, 512), K(num_ops, 640);
        for (const auto &v : variants) {
            for (const auto &kn : kernels) {
                pt::PlannerConfig cfg;
                cfg.max_mr = kn.max_mr;
                cfg.row_align = kn.row_align;
                cfg.min_mr = kn.min_mr;
                cfg.a_bytes = kn.a_bytes;
                cfg.wei_bits = kn.wei_bits;
                cfg.wei_col_extra_bytes = kn.extra;
                cfg.tun[pt::kMb] = v.mb;
                cfg.tun[pt::kNs] = v.ns;
                cfg.tun[pt::kOrder] = v.order;
                cfg.tun[pt::kMinTiles] = v.min_tiles;
                cfg.tun[pt::kL3sb] = v.l3sb ? 1 : 0;
                pt::sanitize(cfg.tun, kn.row_align, kn.min_mr);
                cfg.num_threads = v.threads;
                cfg.l3_bytes_per_team = v.l3;
                pt::Plan plan;
                pt::plan_tiles(M.data(), N.data(), K.data(), num_ops, cfg,
                        v.teams.data(), static_cast<int>(v.teams.size()), plan);
                SCOPED_TRACE("num_ops=" + std::to_string(num_ops) + " mb="
                        + std::to_string(v.mb) + " ns=" + std::to_string(v.ns)
                        + " teams=" + std::to_string(v.teams.size())
                        + " max_mr=" + std::to_string(kn.max_mr)
                        + " min_mr=" + std::to_string(kn.min_mr));
                check_plan(M, N, K, cfg, v.teams, plan);
            }
        }
    }
}

TEST(PromptTilePlanner, TailBlocksKeepEveryMrCallAtFourRows) {
    const auto split = [](int M, int mb, int max_mr) {
        pt::PlannerConfig cfg;
        cfg.max_mr = max_mr;
        std::vector<int> s;
        pt::split_rows(M, mb, cfg, s);
        return s;
    };
    EXPECT_EQ(split(23, 16, 8), (std::vector<int> {0, 16, 23}));
    // A 7-row tail runs as MR 4 + 3.
    EXPECT_EQ(split(23, 16, 6), (std::vector<int> {0, 23}));
    EXPECT_EQ(split(18, 16, 8), (std::vector<int> {0, 18}));
    EXPECT_EQ(split(20, 16, 6), (std::vector<int> {0, 16, 20}));
    const std::vector<int> s = split(71, 8, 6);
    EXPECT_EQ(s.size(), 9u);
    EXPECT_EQ(s[7], 56);
    EXPECT_EQ(s.back(), 71);
}

// INT8 sums in int32, so a tail of any size keeps its bits; blocks follow
// the CK max MR.
TEST(PromptTilePlanner, QuantFamiliesKeepShortTails) {
    pt::PlannerConfig cfg;
    cfg.max_mr = 12;
    cfg.row_align = 12;
    cfg.min_mr = 1;
    std::vector<int> s;
    pt::split_rows(23, 16, cfg, s);
    EXPECT_EQ(s, (std::vector<int> {0, 12, 23}));
    pt::split_rows(25, 12, cfg, s);
    EXPECT_EQ(s, (std::vector<int> {0, 12, 24, 25}));
}

TEST(PromptTilePlanner, AutoSizesFollowL2) {
    pt::PlannerConfig cfg; // 1 MB L2, bf16, B slice 100%, A block 50%
    EXPECT_EQ(pt::auto_ns(2048, 1024, cfg), 256); // Qwen3.6 W13
    EXPECT_EQ(pt::auto_ns(512, 2048, cfg), 1024); // Qwen3.6 W2
    EXPECT_EQ(pt::auto_ns(2880, 5760, cfg), 160); // GPT-OSS W13 / W2
    EXPECT_EQ(pt::auto_ns(64, 96, cfg), 96); // capped by N
    EXPECT_EQ(pt::auto_mb(2048, cfg), 128);
    EXPECT_EQ(pt::auto_mb(512, cfg), 512);
    EXPECT_EQ(pt::auto_mb(2880, cfg), 88);
    EXPECT_EQ(pt::auto_mb(64, cfg), 512);
    EXPECT_EQ(pt::auto_mb(100000, cfg), 16);
    cfg.tun[pt::kMbMin] = 32;
    cfg.tun[pt::kMbMax] = 96;
    EXPECT_EQ(pt::auto_mb(64, cfg), 96);
    EXPECT_EQ(pt::auto_mb(100000, cfg), 32);
    cfg.tun[pt::kNsL2Pct] = 50;
    cfg.tun[pt::kMbL2Pct] = 25;
    EXPECT_EQ(pt::auto_ns(2048, 1024, cfg), 128);
    EXPECT_EQ(pt::auto_mb(2048, cfg), 64);
}

// Column bytes are K * wei_bits / 8 plus the packed extra; row bytes are
// K * a_bytes.
TEST(PromptTilePlanner, AutoSizesFollowTheFamilyByteModel) {
    pt::PlannerConfig cfg;
    cfg.a_bytes = 1;
    cfg.wei_bits = 8;
    cfg.wei_col_extra_bytes = 4;
    cfg.row_align = 12;
    EXPECT_EQ(pt::wei_col_bytes(2048, cfg), 2052);
    EXPECT_EQ(pt::auto_ns(2048, 4096, cfg), 480); // 1 MiB / 2052 = 511
    EXPECT_EQ(pt::auto_mb(2048, cfg), 252); // 512 KiB / 2048 = 256
    cfg.wei_bits = 4;
    cfg.wei_col_extra_bytes = 2048 / 32 * (4 + 2);
    EXPECT_EQ(pt::wei_col_bytes(2048, cfg), 1408);
    EXPECT_EQ(pt::auto_ns(2048, 4096, cfg), 736); // 1 MiB / 1408 = 744
}

TEST(PromptTilePlanner, ShrinksTilesForSmallCalls) {
    const std::vector<int> M = {40, 40}, N = {256, 256}, K = {2048, 2048};
    pt::PlannerConfig cfg;
    cfg.num_threads = 32;
    const std::vector<int> teams = {8, 8, 8, 8};
    pt::Plan plan;
    pt::plan_tiles(M.data(), N.data(), K.data(), 2, cfg, teams.data(), 4, plan);
    EXPECT_EQ(plan.mb, cfg.tun[pt::kMbMin]);
    EXPECT_EQ(plan.ns, cfg.pack_nr);
    check_plan(M, N, K, cfg, teams, plan);

    // A forced size is never shrunk.
    cfg.tun[pt::kMb] = 64;
    cfg.tun[pt::kNs] = 128;
    pt::plan_tiles(M.data(), N.data(), K.data(), 2, cfg, teams.data(), 4, plan);
    EXPECT_EQ(plan.mb, 64);
    EXPECT_EQ(plan.ns, 128);
}

// A single dense layer gets one run of equal M-blocks per team and one
// equal N-slice per thread, so every thread runs the same work; a forced
// size, or a call that is not one dense layer, keeps the shrink loop.
TEST(PromptTilePlanner, SingleDenseLayerSplitsEvenlyPerThread) {
    struct Case {
        int M, N, K, mb, blocks;
        bool even;
    };
    const std::vector<Case> cases = {
            {2048, 768, 768, 256, 8, true}, // embeddinggemma o_proj
            {2048, 2304, 768, 256, 8, true}, // embeddinggemma W13
            {1024, 1280, 768, 128, 8, true}, // embeddinggemma QKV
            {4096, 768, 768, 256, 16, true}, // two runs per team past L2
            {2048, 1024, 3072, 64, 32, true}, // harrier W2: A capped by L2
            {1500, 640, 2048, 96, 16, false}, // short last block
    };
    const std::vector<int> teams(8, 4);
    for (const auto &c : cases) {
        SCOPED_TRACE("M=" + std::to_string(c.M) + " N=" + std::to_string(c.N)
                + " K=" + std::to_string(c.K));
        pt::PlannerConfig cfg;
        cfg.num_threads = 32;
        cfg.single_dense = true;
        pt::Plan plan;
        pt::plan_tiles(&c.M, &c.N, &c.K, 1, cfg, teams.data(), 8, plan);
        check_plan({c.M}, {c.N}, {c.K}, cfg, teams, plan);
        EXPECT_EQ(plan.mb, c.mb);
        ASSERT_EQ(plan.tiles.size(), static_cast<size_t>(c.blocks) * 4);
        for (int t = 0; t < 8 && c.even; ++t)
            EXPECT_EQ(
                    plan.team_begin[t + 1] - plan.team_begin[t], c.blocks / 2);
        const int nblk = c.N / cfg.pack_nr;
        for (const auto &tile : plan.tiles)
            EXPECT_LE(std::abs(tile.ncols / cfg.pack_nr - nblk / 4), 1);
    }

    // A 262144-column LM head and a 100-row call keep the shrink loop.
    for (const auto &[m, n, k] :
            {std::tuple {256, 262144, 640}, std::tuple {100, 768, 768}}) {
        pt::PlannerConfig cfg;
        cfg.num_threads = 32;
        cfg.single_dense = true;
        pt::Plan dense, plain;
        pt::plan_tiles(&m, &n, &k, 1, cfg, teams.data(), 8, dense);
        cfg.single_dense = false;
        pt::plan_tiles(&m, &n, &k, 1, cfg, teams.data(), 8, plain);
        EXPECT_EQ(dense.tiles.size(), plain.tiles.size()) << "M=" << m;
        EXPECT_EQ(dense.mb, plain.mb) << "M=" << m;
        EXPECT_EQ(dense.ns, plain.ns) << "M=" << m;
    }

    const int M = 2048, N = 768, K = 768;
    pt::PlannerConfig cfg;
    cfg.num_threads = 32;
    pt::Plan plain;
    pt::plan_tiles(&M, &N, &K, 1, cfg, teams.data(), 8, plain);
    cfg.single_dense = true;
    cfg.tun[pt::kMb] = 64;
    pt::Plan forced;
    pt::plan_tiles(&M, &N, &K, 1, cfg, teams.data(), 8, forced);
    EXPECT_EQ(forced.mb, 64);
    EXPECT_GE(forced.tiles.size(), plain.tiles.size());
    EXPECT_GT(plain.tiles.size(), 32u);
}

TEST(PromptTilePlanner, HeaviestExpertFirstAndBalancedTeams) {
    const std::vector<int> M = skewed_m(256, 900, 3);
    const int num_ops = static_cast<int>(M.size());
    const std::vector<int> N(num_ops, 1024), K(num_ops, 2048);
    pt::PlannerConfig cfg;
    cfg.num_threads = 32;
    const std::vector<int> teams = {8, 8, 8, 8};
    pt::Plan plan;
    pt::plan_tiles(
            M.data(), N.data(), K.data(), num_ops, cfg, teams.data(), 4, plan);
    ASSERT_FALSE(plan.tiles.empty());
    const int heaviest = static_cast<int>(
            std::max_element(M.begin(), M.end()) - M.begin());
    EXPECT_EQ(plan.tiles.front().e, heaviest);
    EXPECT_LT(plan.est_imbalance, 1.05);
    check_plan(M, N, K, cfg, teams, plan);
}

TEST(PromptTilePlanner, AutoOrderWalksTallExpertsBlockMajor) {
    pt::PlannerConfig cfg; // kAuto, 32 MB L3 per team
    cfg.num_threads = 32;
    const std::vector<int> teams = {8, 8, 8, 8};
    pt::Plan plan;

    // Shared expert over a 4096-token chunk: A (16 MB) > B (4 MB).  Every
    // team must own whole M-blocks across all slices, i.e. a quarter of A.
    {
        const std::vector<int> M = {4096}, N = {1024}, K = {2048};
        pt::plan_tiles(
                M.data(), N.data(), K.data(), 1, cfg, teams.data(), 4, plan);
        check_plan(M, N, K, cfg, teams, plan);
        ASSERT_GE(plan.tiles.size(), size_t(2));
        EXPECT_EQ(plan.tiles[0].m0, plan.tiles[1].m0);
        EXPECT_LT(plan.tiles[0].n0, plan.tiles[1].n0);
        for (int t = 0; t < 4; ++t) {
            int lo = M[0], hi = 0;
            for (int i = plan.team_begin[t]; i < plan.team_begin[t + 1]; ++i) {
                lo = std::min(lo, plan.tiles[i].m0);
                hi = std::max(hi, plan.tiles[i].m0 + plan.tiles[i].rows);
            }
            EXPECT_LE(hi - lo, M[0] / 4 + plan.mb) << "team " << t;
        }
    }
    // A lone expert with M == N still spans every team: block major.
    {
        const std::vector<int> M = {1024}, N = {1024}, K = {2048};
        pt::plan_tiles(
                M.data(), N.data(), K.data(), 1, cfg, teams.data(), 4, plan);
        check_plan(M, N, K, cfg, teams, plan);
        ASSERT_GE(plan.tiles.size(), size_t(2));
        EXPECT_EQ(plan.tiles[0].m0, plan.tiles[1].m0);
        EXPECT_LT(plan.tiles[0].n0, plan.tiles[1].n0);
    }
    // Routed experts (M < N, each well under half a team's share): slice
    // major, consecutive tiles share n0.
    {
        const std::vector<int> M(16, 100), N(16, 1024), K(16, 2048);
        cfg.tun[pt::kMinTiles] = 1;
        cfg.tun[pt::kMb] = 64;
        pt::plan_tiles(
                M.data(), N.data(), K.data(), 16, cfg, teams.data(), 4, plan);
        check_plan(M, N, K, cfg, teams, plan);
        ASSERT_GE(plan.tiles.size(), size_t(2));
        EXPECT_EQ(plan.tiles[0].e, plan.tiles[1].e);
        EXPECT_EQ(plan.tiles[0].n0, plan.tiles[1].n0);
        EXPECT_LT(plan.tiles[0].m0, plan.tiles[1].m0);
    }
    // B no longer fits half the L3: slice major whatever A or the share.
    {
        EXPECT_EQ(pt::expert_order(16384, 4096, 4096, true, cfg),
                pt::TileOrder::kSliceMajor);
        EXPECT_EQ(pt::expert_order(4096, 1024, 2048, false, cfg),
                pt::TileOrder::kBlockMajor);
        EXPECT_EQ(pt::expert_order(1024, 1024, 2048, false, cfg),
                pt::TileOrder::kSliceMajor);
        EXPECT_EQ(pt::expert_order(1024, 1024, 2048, true, cfg),
                pt::TileOrder::kBlockMajor);
        // B (4 MB) over a 1% L3 share falls back to slice major.
        cfg.tun[pt::kBmL3Pct] = 1;
        EXPECT_EQ(pt::expert_order(4096, 1024, 2048, false, cfg),
                pt::TileOrder::kSliceMajor);
        cfg.tun[pt::kBmL3Pct] = 50;
        cfg.tun[pt::kOrder] = 0;
        EXPECT_EQ(pt::expert_order(4096, 1024, 2048, true, cfg),
                pt::TileOrder::kSliceMajor);
    }
}

TEST(PromptTilePlanner, UnevenTeamsGetProportionalCost) {
    const std::vector<int> M(16, 512), N(16, 512), K(16, 512);
    pt::PlannerConfig cfg;
    cfg.num_threads = 12;
    const std::vector<int> teams = {8, 4};
    pt::Plan plan;
    pt::plan_tiles(
            M.data(), N.data(), K.data(), 16, cfg, teams.data(), 2, plan);
    int64_t cost[2] = {0, 0};
    for (int t = 0; t < 2; ++t)
        for (int i = plan.team_begin[t]; i < plan.team_begin[t + 1]; ++i)
            cost[t] += pt::tile_cost(plan.tiles[i], 512, cfg.tun[pt::kRb]);
    ASSERT_GT(cost[1], 0);
    EXPECT_NEAR(static_cast<double>(cost[0]) / cost[1], 2.0, 0.1);
}

/// Long-tailed M in [0, max_m] from raw `std::mt19937` output, whose
/// sequence the standard fixes (the distributions vary by library).
std::vector<int> golden_m(int num_ops, int max_m, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<int> M(num_ops);
    for (auto &m : M) {
        const uint64_t x = rng() % 1001;
        m = static_cast<int>(
                static_cast<uint64_t>(max_m) * x * x * x / 1000000000u);
    }
    return M;
}

/// FNV-1a over a plan's integer fields.
uint64_t plan_hash(const pt::Plan &p) {
    uint64_t h = 0xcbf29ce484222325ull;
    const auto mix = [&h](int64_t v) {
        const auto u = static_cast<uint64_t>(v);
        for (int i = 0; i < 8; ++i) {
            h ^= (u >> (8 * i)) & 0xffu;
            h *= 0x100000001b3ull;
        }
    };
    mix(static_cast<int64_t>(p.tiles.size()));
    for (const auto &t : p.tiles) {
        mix(t.e);
        mix(t.m0);
        mix(t.rows);
        mix(t.n0);
        mix(t.ncols);
    }
    for (const int b : p.team_begin)
        mix(b);
    mix(p.mb);
    mix(p.ns);
    mix(p.total_cost);
    return h;
}

/// Every planner input, spelled out so that a default changing cannot
/// move the golden plans.
struct GoldenKnobs {
    int pack_nr, max_mr;
    int64_t l2, l3;
    int mb, ns, ns_l2_pct, mb_l2_pct, rb, min_tiles;
    bool l3sb;
    pt::TileOrder order;
    int threads;
};

pt::PlannerConfig golden_cfg(const GoldenKnobs &g) {
    static_assert(pt::kNumKnobs == 18, "set every knob below");
    pt::PlannerConfig cfg;
    cfg.tun = pt::Tunables {};
    cfg.tun[pt::kEnable] = 1;
    cfg.tun[pt::kTeam] = 0;
    cfg.tun[pt::kSubteam] = 4;
    cfg.tun[pt::kMb] = g.mb;
    cfg.tun[pt::kNs] = g.ns;
    cfg.tun[pt::kNsL2Pct] = g.ns_l2_pct;
    cfg.tun[pt::kMbL2Pct] = g.mb_l2_pct;
    cfg.tun[pt::kMbMin] = 16;
    cfg.tun[pt::kMbMax] = 512;
    cfg.tun[pt::kRb] = g.rb;
    cfg.tun[pt::kGrab] = 1;
    cfg.tun[pt::kOrder] = static_cast<int>(g.order);
    cfg.tun[pt::kSteal] = 1;
    cfg.tun[pt::kMinTiles] = g.min_tiles;
    cfg.tun[pt::kL3sb] = g.l3sb ? 1 : 0;
    cfg.tun[pt::kSbL3Pct] = 50;
    cfg.tun[pt::kBmL3Pct] = 50;
    cfg.tun[pt::kSpanPct] = 50;
    cfg.pack_nr = g.pack_nr;
    cfg.max_mr = g.max_mr;
    cfg.l2_bytes = g.l2;
    cfg.l3_bytes_per_team = g.l3;
    cfg.a_bytes = 2;
    cfg.wei_bits = 16;
    cfg.min_mr = 4;
    cfg.row_align = 8;
    cfg.wei_col_extra_bytes = 0;
    cfg.num_threads = g.threads;
    return cfg;
}

// Today's plans, pinned bit for bit: a planner refactor must keep every
// hash.  Shapes are Qwen3.6-35B-A3B (W13 N=1024 K=2048, W2 N=2048 K=512)
// and GPT-OSS-20B (W13 N=5760 K=2880, W2 N=2880 K=2880) prompt passes.
TEST(PromptTilePlanner, GoldenPlanHashes) {
    constexpr int64_t kMiB = int64_t(1) << 20;
    const auto kAuto = pt::TileOrder::kAuto;
    const auto kSlice = pt::TileOrder::kSliceMajor;
    const auto kBlock = pt::TileOrder::kBlockMajor;
    const auto uni = [](size_t n, int v) { return std::vector<int>(n, v); };
    const auto cat = [](std::vector<int> a, const std::vector<int> &b) {
        a.insert(a.end(), b.begin(), b.end());
        return a;
    };
    const auto alt = [](size_t n, int a, int b) {
        std::vector<int> v(n);
        for (size_t i = 0; i < n; ++i)
            v[i] = (i % 2) ? b : a;
        return v;
    };
    struct GoldenCase {
        std::vector<int> M, N, K;
        GoldenKnobs g;
        std::vector<int> teams;
        uint64_t hash;
    };
    // knobs: pack_nr, max_mr, l2, l3, mb, ns, ns_l2_pct, mb_l2_pct, rb,
    //        min_tiles, l3sb, order, threads
    const std::vector<GoldenCase> cases = {
            {golden_m(256, 160, 1), uni(256, 1024), uni(256, 2048),
                    {32, 8, kMiB, 32 * kMiB, 0, 0, 100, 50, 32, 8, true, kAuto,
                            128},
                    uni(32, 4), 0x03a944dcaa28ccaaull},
            {golden_m(256, 160, 1), uni(256, 2048), uni(256, 512),
                    {32, 8, kMiB, 32 * kMiB, 0, 0, 100, 50, 32, 8, true, kAuto,
                            128},
                    uni(32, 4), 0xc0311e7693ab1c40ull},
            {golden_m(256, 640, 2), uni(256, 1024), uni(256, 2048),
                    {32, 8, kMiB, 32 * kMiB, 0, 0, 100, 50, 32, 8, true, kAuto,
                            128},
                    uni(32, 4), 0xf80fd8baf6390fdaull},
            {golden_m(32, 1000, 3), uni(32, 5760), uni(32, 2880),
                    {32, 8, kMiB, 32 * kMiB, 0, 0, 100, 50, 32, 8, true, kAuto,
                            64},
                    uni(16, 4), 0x4e31b21659dcab4full},
            {golden_m(32, 4000, 4), uni(32, 2880), uni(32, 2880),
                    {32, 8, kMiB, 32 * kMiB, 0, 0, 100, 50, 32, 8, true, kAuto,
                            64},
                    uni(16, 4), 0x0b92b55f81ad19ebull},
            {{8192}, {1024}, {2048},
                    {32, 8, kMiB, 32 * kMiB, 0, 0, 100, 50, 32, 8, true, kAuto,
                            128},
                    uni(32, 4), 0x5c3256bb543873e2ull},
            {cat(golden_m(256, 160, 5), {4096}), uni(257, 1024), uni(257, 2048),
                    {32, 8, kMiB, 32 * kMiB, 0, 0, 100, 50, 32, 8, true, kAuto,
                            128},
                    uni(16, 8), 0x92256b6a01c10f85ull},
            {golden_m(64, 400, 6), uni(64, 1024), uni(64, 2048),
                    {64, 6, kMiB, 32 * kMiB, 0, 0, 100, 50, 32, 8, true, kAuto,
                            32},
                    uni(4, 8), 0x7e3fe0aa8776d9e2ull},
            {golden_m(64, 400, 7), uni(64, 1024), uni(64, 2048),
                    {32, 8, kMiB, 32 * kMiB, 24, 96, 100, 50, 32, 8, true,
                            kAuto, 32},
                    uni(4, 8), 0x5e335ecaaab3e41full},
            {golden_m(128, 800, 8), uni(128, 5760), uni(128, 2880),
                    {32, 8, kMiB, 32 * kMiB, 0, 0, 100, 50, 32, 8, false,
                            kSlice, 64},
                    uni(16, 4), 0x3db63e5fb8bf2c0eull},
            {golden_m(16, 3000, 9), uni(16, 1024), uni(16, 2048),
                    {32, 8, kMiB, 2 * kMiB, 0, 0, 100, 50, 32, 8, true, kBlock,
                            32},
                    uni(4, 8), 0xb1db292adc5e460full},
            {golden_m(128, 300, 10), uni(128, 2048), uni(128, 512),
                    {32, 8, 2 * kMiB, 32 * kMiB, 0, 0, 200, 25, 32, 8, true,
                            kAuto, 128},
                    uni(32, 4), 0x7307936185e8b94cull},
            {golden_m(40, 200, 11), uni(40, 1024), uni(40, 2048),
                    {32, 8, kMiB / 2, 32 * kMiB, 0, 0, 50, 25, 0, 1, true,
                            kAuto, 7},
                    {3, 3, 1}, 0xdfdbc361f446024cull},
            {golden_m(256, 160, 12), uni(256, 1024), uni(256, 2048),
                    {32, 8, kMiB, 32 * kMiB, 0, 0, 100, 50, 32, 64, true, kAuto,
                            192},
                    uni(24, 8), 0xbf15286b30244527ull},
            {golden_m(32, 600, 13), alt(32, 1024, 2048), alt(32, 2048, 512),
                    {32, 8, kMiB, 32 * kMiB, 0, 0, 100, 50, 48, 8, true, kAuto,
                            12},
                    {6, 6}, 0x78530a6dadba6894ull},
            {golden_m(8, 40, 14), uni(8, 1024), uni(8, 2048),
                    {32, 8, kMiB, 32 * kMiB, 0, 0, 100, 50, 32, 8, true, kAuto,
                            128},
                    uni(32, 4), 0x5aae5060f7e97041ull},
    };
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto &c = cases[i];
        const pt::PlannerConfig cfg = golden_cfg(c.g);
        pt::Plan plan;
        pt::plan_tiles(c.M.data(), c.N.data(), c.K.data(),
                static_cast<int>(c.M.size()), cfg, c.teams.data(),
                static_cast<int>(c.teams.size()), plan);
        SCOPED_TRACE("case " + std::to_string(i));
        check_plan(c.M, c.N, c.K, cfg, c.teams, plan);
        EXPECT_EQ(plan_hash(plan), c.hash)
                << "case " << i << " hash 0x" << std::hex << plan_hash(plan)
                << std::dec << " tiles=" << plan.tiles.size();
    }
}

// ── Tunables ────────────────────────────────────────────────────────

TEST(PromptTileTunables, EnvNamesAndValues) {
    EXPECT_EQ(pt::env_name(pt::kEnable), "ZENDNNL_GRP_MATMUL_PROMPT_TILE");
    EXPECT_EQ(pt::env_name(pt::kNsL2Pct),
            "ZENDNNL_GRP_MATMUL_PROMPT_TILE_NS_L2_PCT");
    EXPECT_EQ(pt::env_name(pt::kSpanPct),
            "ZENDNNL_GRP_MATMUL_PROMPT_TILE_SPAN_PCT");
    int v = -1;
    EXPECT_TRUE(pt::parse_knob_value(pt::kEnable, "15", v));
    EXPECT_EQ(v, 15);
    for (const char *bad : {"16", "-1", "1x", "", " ", "0x2"})
        EXPECT_FALSE(pt::parse_knob_value(pt::kEnable, bad, v)) << bad;
    EXPECT_FALSE(pt::parse_knob_value(pt::kMb, nullptr, v));
    EXPECT_TRUE(pt::parse_knob_value(pt::kNs, "1048576", v));
    EXPECT_FALSE(pt::parse_knob_value(pt::kNs, "1048577", v));
    EXPECT_EQ(v, 1048576);
}

TEST(PromptTileTunables, KnobTableIsConsistent) {
    for (int k = 0; k < pt::kNumKnobs; ++k) {
        const pt::KnobDef &d = pt::kKnobs[k];
        SCOPED_TRACE(d.name);
        EXPECT_GE(d.def, d.lo);
        EXPECT_LE(d.def, d.hi);
    }
    // The planner's default config is a BF16 call.
    const pt::PlannerConfig cfg;
    const pt::FamilyFacts &bf16 = pt::facts(pt::Family::kBf16);
    EXPECT_EQ(cfg.tun.v, pt::defaults().v);
    EXPECT_EQ(cfg.tun[pt::kEnable], 9);
    EXPECT_EQ(cfg.a_bytes, bf16.a_bytes);
    EXPECT_EQ(cfg.wei_bits, bf16.wei_bits);
    EXPECT_EQ(cfg.min_mr, bf16.min_mr);
    EXPECT_EQ(cfg.row_align, bf16.row_align);
    const pt::FamilyFacts &f16 = pt::facts(pt::Family::kF16);
    EXPECT_EQ(f16.a_bytes, 2);
    EXPECT_EQ(f16.wei_bits, 16);
    EXPECT_EQ(f16.min_mr, 1);
    EXPECT_EQ(f16.row_align, 0);
    EXPECT_FALSE(f16.kblock);
}

TEST(PromptTileTunables, ProfileGrammar) {
    pt::Profile p;
    std::string err;
    ASSERT_TRUE(pt::parse_profile(
            "fam=bf16,K=2880,N=5760,M=8192-:mb=64,ns=320; K=2048:rb=48", p,
            &err))
            << err;
    ASSERT_EQ(p.size(), 2u);
    EXPECT_EQ(p[0].fam, static_cast<int>(pt::Family::kBf16));
    EXPECT_EQ(p[0].lo[2], 8192);
    EXPECT_EQ(p[0].hi[2], pt::ProfileEntry::kAny);
    EXPECT_EQ(p[0].set, (1u << pt::kMb) | (1u << pt::kNs));
    EXPECT_EQ(p[0].val[pt::kNs], 320);
    EXPECT_EQ(p[1].fam, -1);
    EXPECT_EQ(p[1].lo[0], 2048);
    EXPECT_EQ(p[1].hi[0], 2048);
    EXPECT_EQ(p[1].lo[1], 0);
    EXPECT_EQ(p[1].hi[1], pt::ProfileEntry::kAny);

    ASSERT_TRUE(pt::parse_profile("fam=f16:mb=32", p, &err)) << err;
    ASSERT_EQ(p.size(), 1u);
    EXPECT_EQ(p[0].fam, static_cast<int>(pt::Family::kF16));

    // Spaces, empty entries, `-hi` and `lo-hi` ranges, and no keys at all.
    ASSERT_TRUE(pt::parse_profile(
            " Mmax = -64 , T=32-128 : order=1 ;; :grab=2 ;", p, &err))
            << err;
    ASSERT_EQ(p.size(), 2u);
    EXPECT_EQ(p[0].lo[3], 0);
    EXPECT_EQ(p[0].hi[3], 64);
    EXPECT_EQ(p[0].lo[4], 32);
    EXPECT_EQ(p[0].hi[4], 128);
    EXPECT_EQ(p[1].set, 1u << pt::kGrab);

    EXPECT_TRUE(pt::parse_profile("", p) && p.empty());

    // One bad item rejects the whole profile.
    for (const char *bad : {"K=2048:rb=48; K=1:bogus=1", "K=2048", "K=2048:",
                 "fam=fp8:mb=8", "Q=1:mb=8", "K=1,K=2:mb=8", "K=9-3:mb=8",
                 "K=-:mb=8", "K=a:mb=8", "K=-5-:mb=8", "M=1:mb=65537",
                 "M=1:mb=8,mb=16", "M=1:team=4", ":enable=0", "M=1:mb"}) {
        p.assign(1, pt::ProfileEntry {});
        err.clear();
        EXPECT_FALSE(pt::parse_profile(bad, p, &err)) << bad;
        EXPECT_TRUE(p.empty()) << bad;
        EXPECT_FALSE(err.empty()) << bad;
    }
    std::string many;
    for (int i = 0; i < 64; ++i)
        many += "T=" + std::to_string(i) + ":grab=2;";
    EXPECT_TRUE(pt::parse_profile(many, p) && p.size() == 64u);
    EXPECT_FALSE(pt::parse_profile(many + "T=64:grab=2", p));
}

TEST(PromptTileTunables, ProfileFromFile) {
    const std::string path = ::testing::TempDir() + "ptile_profile_"
            + std::to_string(
                    std::chrono::steady_clock::now().time_since_epoch().count())
            + ".txt";
    {
        std::ofstream f(path);
        f << "fam=bf16,K=2048:mb=24\nK=512:ns=64\n";
        ASSERT_TRUE(f.good()) << path;
    }
    pt::Profile p;
    std::string err;
    ASSERT_TRUE(pt::load_profile(("@" + path).c_str(), p, &err)) << err;
    ASSERT_EQ(p.size(), 2u);
    EXPECT_EQ(p[1].val[pt::kNs], 64);
    ASSERT_EQ(std::remove(path.c_str()), 0) << path;
    EXPECT_FALSE(pt::load_profile(("@" + path).c_str(), p, &err));
    EXPECT_TRUE(p.empty());
    EXPECT_TRUE(pt::load_profile("K=1:mb=8", p) && p.size() == 1u);
    EXPECT_TRUE(pt::load_profile(nullptr, p) && p.empty());
}

TEST(PromptTileTunables, FirstMatchingEntryWins) {
    pt::Profile p;
    ASSERT_TRUE(pt::parse_profile(
            "fam=int8:mb=8; K=2048,M=-1000:mb=24; K=2048:mb=32; :mb=40", p));
    pt::ShapeKey key;
    key.K = 2048;
    key.N = 1024;
    key.M = 4096;
    key.Mmax = 300;
    key.T = 64;
    EXPECT_EQ(pt::match_profile(p, key), 2);
    key.M = 1000;
    EXPECT_EQ(pt::match_profile(p, key), 1);
    key.K = 512;
    EXPECT_EQ(pt::match_profile(p, key), 3);
    key.fam = pt::Family::kInt8;
    EXPECT_EQ(pt::match_profile(p, key), 0);
    EXPECT_EQ(pt::match_profile(pt::Profile {}, key), -1);
}

TEST(PromptTileTunables, PrecedenceRunsDefaultsEnvProfile) {
    pt::Profile env_profile;
    ASSERT_TRUE(pt::parse_profile("K=2048:mb=40,rb=8", env_profile));
    pt::ProfileEntry env;
    env.set = (1u << pt::kRb) | (1u << pt::kSteal);
    env.val[pt::kRb] = 16;
    env.val[pt::kSteal] = 0;
    pt::ShapeKey key;
    key.K = 2048;
    pt::Resolved r = pt::resolve_with(key, env, env_profile);
    EXPECT_EQ(r.env_entry, 0);
    EXPECT_EQ(r.t[pt::kMb], 40); // env profile over default
    EXPECT_EQ(r.t[pt::kRb], 8); // env profile over env var
    EXPECT_EQ(r.t[pt::kSteal], 0); // env var over default
    EXPECT_EQ(r.t[pt::kGrab], 1);
    EXPECT_EQ(r.t[pt::kNs], 0);
    key.K = 512;
    r = pt::resolve_with(key, env, env_profile);
    EXPECT_EQ(r.env_entry, -1);
    EXPECT_EQ(r.t[pt::kMb], 0);
    EXPECT_EQ(r.t[pt::kRb], 16);
    EXPECT_EQ(r.t[pt::kGrab], 1);
}

TEST(PromptTileTunables, OverridesWinAndKeepTheFloors) {
    pt::ShapeKey key;
    key.K = 2048;
    {
        PtOverride mb(pt::kMb, 72), grab(pt::kGrab, 0), team(pt::kTeam, 3);
        const pt::Resolved r = pt::resolve(key);
        EXPECT_EQ(r.t[pt::kMb], 72);
        EXPECT_EQ(r.t[pt::kGrab], 1);
        EXPECT_EQ(r.t[pt::kTeam], 3);
        EXPECT_EQ(pt::global_value(pt::kTeam), 3);
        PtOverride mask(pt::kEnable, 2);
        EXPECT_FALSE(pt::enabled(pt::Family::kBf16));
        EXPECT_TRUE(pt::enabled(pt::Family::kInt8));
        EXPECT_FALSE(pt::enabled(pt::Family::kW4a8));
        EXPECT_FALSE(pt::enabled(pt::Family::kF16));
    }
    {
        PtOverride mask(pt::kEnable, 8);
        EXPECT_FALSE(pt::enabled(pt::Family::kBf16));
        EXPECT_FALSE(pt::enabled(pt::Family::kInt8));
        EXPECT_FALSE(pt::enabled(pt::Family::kW4a8));
        EXPECT_TRUE(pt::enabled(pt::Family::kF16));
    }
    for (int k = 0; k < pt::kNumKnobs; ++k)
        EXPECT_EQ(pt::test_api::s_override[k].load(), 0) << pt::kKnobs[k].name;
}

TEST(PromptTileTunables, SanitizeFitsMbBoundsToTheCall) {
    pt::Tunables t = pt::defaults();
    pt::sanitize(t, 8, 4);
    EXPECT_EQ(t[pt::kMbMin], 16);
    EXPECT_EQ(t[pt::kMbMax], 512);
    // INT8 at CK max MR 12.
    t = pt::defaults();
    pt::sanitize(t, 12, 1);
    EXPECT_EQ(t[pt::kMbMin], 24);
    EXPECT_EQ(t[pt::kMbMax], 504);
    t[pt::kMbMin] = 1;
    t[pt::kMbMax] = 5;
    pt::sanitize(t, 6, 1);
    EXPECT_EQ(t[pt::kMbMin], 6);
    EXPECT_EQ(t[pt::kMbMax], 6);
    t[pt::kMbMin] = 300;
    t[pt::kMbMax] = 100;
    pt::sanitize(t, 8, 4);
    EXPECT_EQ(t[pt::kMbMin], 304);
    EXPECT_EQ(t[pt::kMbMax], 304);
    t[pt::kMbMin] = 1;
    pt::sanitize(t, 1, 4);
    EXPECT_EQ(t[pt::kMbMin], 4);
}

// ── Topology ────────────────────────────────────────────────────────

/// Team sizes match `team_of`, and every team's victim list holds every
/// other team once, same-domain teams first.
void check_team_map(const pt::TeamMap &m, int T) {
    ASSERT_EQ(m.num_threads, T);
    ASSERT_EQ(static_cast<int>(m.team_of.size()), T);
    ASSERT_EQ(static_cast<int>(m.team_size.size()), m.n_teams);
    ASSERT_EQ(static_cast<int>(m.domain_of_team.size()), m.n_teams);
    std::vector<int> count(m.n_teams, 0);
    for (const int t : m.team_of) {
        ASSERT_GE(t, 0);
        ASSERT_LT(t, m.n_teams);
        ++count[t];
    }
    EXPECT_EQ(count, m.team_size);
    for (const int s : m.team_size)
        EXPECT_GE(s, 1);
    const int n = m.n_teams;
    ASSERT_EQ(m.steal_order.size(), static_cast<size_t>(n) * (n - 1));
    for (int t = 0; t < n; ++t) {
        std::vector<int> seen(n, 0);
        bool other_domain_seen = false;
        for (int i = 0; i < n - 1; ++i) {
            const int v = m.steal_order[static_cast<size_t>(t) * (n - 1) + i];
            ASSERT_NE(v, t);
            ++seen[v];
            const bool same = m.domain_of_team[v] == m.domain_of_team[t];
            EXPECT_FALSE(same && other_domain_seen) << "team " << t;
            other_domain_seen |= !same;
        }
        for (int v = 0; v < n; ++v)
            EXPECT_EQ(seen[v], v == t ? 0 : 1);
    }
}

TEST(PromptTileTopology, ContiguousMap) {
    const pt::TeamMap m = pt::make_contiguous_team_map(10, 4, 1 << 20, 1 << 25);
    EXPECT_EQ(m.n_teams, 3);
    EXPECT_EQ(m.team_size, (std::vector<int> {4, 4, 2}));
    for (int t = 0; t < 10; ++t)
        EXPECT_EQ(m.team_of[t], t / 4);
    EXPECT_EQ(m.domain_of_team, (std::vector<int> {0, 1, 2}));
    EXPECT_EQ(m.steal_order, (std::vector<int> {1, 2, 2, 0, 0, 1}));
    EXPECT_FALSE(m.from_l3);
    check_team_map(m, 10);
}

TEST(PromptTileTopology, DomainMapSplitsIntoSubTeams) {
    // Two 8-thread L3s, sub-teams of 4: siblings are stolen from first.
    {
        std::vector<int> dom(16);
        for (int t = 0; t < 16; ++t)
            dom[t] = 10 + t / 8;
        const pt::TeamMap m
                = pt::make_domain_team_map(dom, 4, 1 << 20, 1 << 25);
        EXPECT_EQ(m.team_size, (std::vector<int> {4, 4, 4, 4}));
        EXPECT_EQ(m.domain_of_team, (std::vector<int> {0, 0, 1, 1}));
        for (int t = 0; t < 16; ++t)
            EXPECT_EQ(m.team_of[t], t / 4);
        EXPECT_EQ(m.steal_order,
                (std::vector<int> {1, 2, 3, 0, 2, 3, 3, 0, 1, 2, 0, 1}));
        check_team_map(m, 16);
    }
    // Uneven domains (6, 6, 2) split into near-equal sub-teams.
    {
        const std::vector<int> dom = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2};
        const pt::TeamMap m
                = pt::make_domain_team_map(dom, 4, 1 << 20, 1 << 25);
        EXPECT_EQ(m.team_size, (std::vector<int> {3, 3, 3, 3, 2}));
        EXPECT_EQ(m.domain_of_team, (std::vector<int> {0, 0, 1, 1, 2}));
        check_team_map(m, 14);
        const pt::TeamMap odd = pt::make_domain_team_map(
                {0, 0, 0, 0, 0}, 4, 1 << 20, 1 << 25);
        EXPECT_EQ(odd.team_size, (std::vector<int> {3, 2}));
        check_team_map(odd, 5);
    }
    // Threads interleaved across domains (spread binding over a
    // non-contiguous CPU numbering): sub-teams follow the domains.
    {
        const std::vector<int> dom = {0, 1, 0, 1, 0, 1, 0, 1};
        const pt::TeamMap m
                = pt::make_domain_team_map(dom, 2, 1 << 20, 1 << 25);
        EXPECT_EQ(m.team_of, (std::vector<int> {0, 2, 0, 2, 1, 3, 1, 3}));
        EXPECT_EQ(m.domain_of_team, (std::vector<int> {0, 0, 1, 1}));
        check_team_map(m, 8);
    }
    // subteam <= 0 keeps whole domains.
    {
        const std::vector<int> dom = {3, 3, 3, 7, 7, 7};
        const pt::TeamMap m
                = pt::make_domain_team_map(dom, 0, 1 << 20, 1 << 25);
        EXPECT_EQ(m.team_size, (std::vector<int> {3, 3}));
        check_team_map(m, 6);
    }
}

TEST(PromptTileTopology, DetectedMapIsConsistent) {
    for (const int T : {1, 4, 8}) {
        for (const int sub : {0, 2, 4}) {
            const pt::TeamMap &m = pt::get_team_map(T, 0, sub);
            check_team_map(m, T);
            EXPECT_GT(m.l2_bytes, 0);
            EXPECT_GT(m.l3_bytes, 0);
            if (sub > 0) {
                for (const int s : m.team_size)
                    EXPECT_LE(s, sub);
            }
        }
    }
    const pt::TeamMap &forced = pt::get_team_map(4, 3, 4);
    EXPECT_EQ(forced.team_size, (std::vector<int> {3, 1}));
    check_team_map(forced, 4);
}

TEST(PromptTileTopology, LookupsFollowTheirKeys) {
    const pt::TeamMap *a = &pt::get_team_map(6, 2, 4);
    const pt::TeamMap *b = &pt::get_team_map(6, 3, 4);
    const pt::TeamMap *c = &pt::get_team_map(5, 2, 4);
    EXPECT_NE(a, b);
    EXPECT_NE(a, c);
    EXPECT_EQ(&pt::get_team_map(6, 2, 4), a);
    EXPECT_EQ(&pt::get_team_map(6, 2, 4), a);
    EXPECT_EQ(&pt::get_team_map(6, 3, 4), b);
    EXPECT_EQ(a->team_size, (std::vector<int> {2, 2, 2}));
    EXPECT_EQ(b->team_size, (std::vector<int> {3, 3}));
    EXPECT_EQ(c->team_size, (std::vector<int> {2, 2, 1}));
    // Inside a parallel region the same arguments name a different map.
    const pt::TeamMap *in_region = nullptr;
    bool active = false;
#pragma omp parallel num_threads(2)
    {
#pragma omp single
        {
            active = omp_in_parallel() != 0;
            in_region = &pt::get_team_map(6, 2, 4);
        }
    }
    if (active) { EXPECT_NE(in_region, a); }
    EXPECT_EQ(&pt::get_team_map(6, 2, 4), a);
}

// ── Executor A/B against the round executor ─────────────────────────

struct Case {
    std::vector<int> M;
    int N = 256;
    int K = 128;
    data_type_t src_dt = data_type_t::bf16;
    grp_matmul_gated_act_t act = grp_matmul_gated_act_t::none;
    data_type_t bias_dt = data_type_t::none;
    data_type_t dst_dt = data_type_t::bf16;
    bool tight = true; // gated acts only: ldc = N / 2
    int threads = 8;
    int nr = 0; // CK pack NR override; 0 keeps the default
    bool wide = false; // inputs spread over 2^-6..2^6 (see `wide_value`)
};

struct RunOut {
    std::vector<std::vector<uint8_t>> dst;
    std::string mode;
    long served = 0;
};

const char *act_label(grp_matmul_gated_act_t a) {
    switch (a) {
        case grp_matmul_gated_act_t::silu_and_mul: return "silu";
        case grp_matmul_gated_act_t::gelu_and_mul: return "gelu";
        case grp_matmul_gated_act_t::swiglu_oai_mul: return "swiglu";
        default: return "none";
    }
}

std::string describe(const Case &c) {
    std::string s = std::string("act=") + act_label(c.act)
            + " src=" + std::to_string(static_cast<int>(c.src_dt))
            + " T=" + std::to_string(c.threads) + " N=" + std::to_string(c.N)
            + " K=" + std::to_string(c.K) + " tight=" + (c.tight ? "1" : "0")
            + " bias=" + std::to_string(static_cast<int>(c.bias_dt))
            + " dst=" + std::to_string(static_cast<int>(c.dst_dt)) + " nr="
            + std::to_string(c.nr) + " wide=" + (c.wide ? "1" : "0") + " M=";
    for (const int m : c.M)
        s += std::to_string(m) + ",";
    return s;
}

/// Values whose exponents span 2^-6..2^6, so an fp32 sum of their products
/// depends on the order it is accumulated in (the default test data sums
/// exactly in any order).
float wide_value(uint32_t h) {
    h ^= h >> 16;
    h *= 0x7feb352dU;
    h ^= h >> 15;
    h *= 0x846ca68bU;
    h ^= h >> 16;
    const float mant
            = static_cast<float>(static_cast<int>(h & 0xFF) - 128) / 128.0f;
    return std::ldexp(mant, static_cast<int>((h >> 8) % 13) - 6);
}

/// Runs the call once through `algo` (ALGO 3 unless a test asks for AUTO)
/// with the CK on and the strategy pinned to rounds; `prompt_tile` toggles
/// the tile executor.
template <typename T>
RunOut run_case_typed(const Case &c, int prompt_tile, int algo) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    NTileStrategyOverride strat(2);
    CustomKernelOverride ck_on(true);
    CustomKernelF16Override ck_f16_on(true);
    CustomKernelNROverride nr(c.nr > 0 ? c.nr : -1);
    AlgoEnvGuard algo_guard(algo);
    PtOverride enable(pt::kEnable, prompt_tile);
    reset_grp_matmul_caches();

    const int num_ops = static_cast<int>(c.M.size());
    const bool gated = c.act != grp_matmul_gated_act_t::none;
    const int ldc = (gated && c.tight) ? c.N / 2 : c.N;
    const size_t dst_elem = (c.dst_dt == data_type_t::f32) ? 4 : 2;

    std::vector<std::vector<T>> src(num_ops), wei(num_ops),
            bias_native(num_ops);
    std::vector<std::vector<float>> bias_f(num_ops);
    std::vector<const void *> srcs(num_ops), weis(num_ops),
            biases(num_ops, nullptr);
    RunOut out;
    out.dst.resize(num_ops);
    std::vector<void *> dsts(num_ops);
    for (int e = 0; e < num_ops; ++e) {
        const int m = std::max(c.M[e], 1);
        src[e].resize(static_cast<size_t>(m) * c.K);
        wei[e].resize(static_cast<size_t>(c.K) * c.N);
        for (size_t i = 0; i < src[e].size(); ++i)
            src[e][i] = c.wide
                    ? T(wide_value(static_cast<uint32_t>(
                              i * 7 + static_cast<size_t>(e) * 1000003)))
                    : T(0.03f
                              * static_cast<float>(
                                      static_cast<int>((i * 7 + e * 3) % 11)
                                      - 5));
        for (size_t i = 0; i < wei[e].size(); ++i)
            wei[e][i] = c.wide
                    ? T(wide_value(static_cast<uint32_t>(
                              i * 5 + static_cast<size_t>(e) * 7919 + 0x9e37)))
                    : T(0.01f
                              * static_cast<float>(
                                      static_cast<int>((i * 5 + e) % 13) - 6));
        srcs[e] = src[e].data();
        weis[e] = wei[e].data();
        if (c.bias_dt == c.src_dt) {
            bias_native[e].resize(c.N);
            for (int j = 0; j < c.N; ++j)
                bias_native[e][j]
                        = T(0.1f * static_cast<float>((j + e) % 7 - 3));
            biases[e] = bias_native[e].data();
        } else if (c.bias_dt == data_type_t::f32) {
            bias_f[e].resize(c.N);
            for (int j = 0; j < c.N; ++j)
                bias_f[e][j] = 0.1f * static_cast<float>((j + e) % 7 - 3);
            biases[e] = bias_f[e].data();
        }
        out.dst[e].assign(static_cast<size_t>(m) * ldc * dst_elem, 0x5A);
        dsts[e] = out.dst[e].data();
    }
    auto gv = GemmVecs::uniform(num_ops, 1, c.N, c.K);
    gv.Ms = c.M;
    gv.ldc.assign(num_ops, ldc);
    auto params = make_mixed_params(
            num_ops, c.src_dt, c.src_dt, c.dst_dt, c.bias_dt);
    for (auto &p : params)
        p.num_threads = c.threads;

    const long served0 = pt::test_api::s_calls_served.load();
    const char *mode = nullptr;
    group_matmul_run_parallel_dispatch(gv.layout, gv.transA, gv.transB, gv.Ms,
            gv.Ns, gv.Ks, gv.alpha, srcs, gv.lda, weis, gv.ldb, biases, gv.beta,
            dsts, gv.ldc, gv.is_wc, params, c.threads, &mode, c.act, c.src_dt);
    out.mode = mode != nullptr ? mode : "";
    out.served = pt::test_api::s_calls_served.load() - served0;
    return out;
}

RunOut run_case(const Case &c, int prompt_tile, int algo = 3) {
    if (c.src_dt == data_type_t::f16)
        return run_case_typed<float16_t>(c, prompt_tile, algo);
    return run_case_typed<bfloat16_t>(c, prompt_tile, algo);
}

/// Tile executor output must be bitwise equal to the round executor's.
void expect_bitwise_equal(const Case &c) {
    SCOPED_TRACE(describe(c));
    const RunOut rounds = run_case(c, 0);
    const RunOut tiles = run_case(c, c.src_dt == data_type_t::f16 ? 8 : 1);
    ASSERT_EQ(rounds.mode.find("ptile"), std::string::npos) << rounds.mode;
    ASSERT_EQ(rounds.mode.find("sequential"), std::string::npos)
            << "reference did not run the round executor: " << rounds.mode;
    ASSERT_NE(tiles.mode.find("flat_n_tile_ptile_"), std::string::npos)
            << "prompt tile executor did not engage: " << tiles.mode;
    ASSERT_EQ(tiles.served, 1);
    for (size_t e = 0; e < rounds.dst.size(); ++e) {
        ASSERT_EQ(rounds.dst[e].size(), tiles.dst[e].size());
        if (c.M[e] > 0) {
            ASSERT_TRUE(std::any_of(rounds.dst[e].begin(), rounds.dst[e].end(),
                    [](uint8_t b) {
                return b != 0x5A;
            })) << "reference left expert "
                << e << " unwritten";
        } else {
            EXPECT_TRUE(std::all_of(rounds.dst[e].begin(), rounds.dst[e].end(),
                    [](uint8_t b) {
                return b == 0x5A;
            })) << "rounds wrote inactive expert "
                << e;
            EXPECT_TRUE(std::all_of(tiles.dst[e].begin(), tiles.dst[e].end(),
                    [](uint8_t b) {
                return b == 0x5A;
            })) << "prompt tiles wrote inactive expert "
                << e;
        }
        EXPECT_EQ(std::memcmp(rounds.dst[e].data(), tiles.dst[e].data(),
                          rounds.dst[e].size()),
                0)
                << "expert " << e << " differs";
    }
}

const std::vector<std::vector<int>> &ab_dists() {
    static const std::vector<std::vector<int>> d = {
            {1, 2, 3, 4, 5, 6, 7, 9, 13, 17, 33, 0, 64, 100, 257},
            {300, 41},
            skewed_m(48, 220, 5),
            {0, 700, 0, 450, 3, 0},
    };
    return d;
}

TEST(PromptTileExecutor, BitwiseMatchesRoundsAcrossActs) {
    SKIP_WITHOUT_BF16_CUSTOM_KERNEL();
    for (const auto act :
            {grp_matmul_gated_act_t::none, grp_matmul_gated_act_t::silu_and_mul,
                    grp_matmul_gated_act_t::gelu_and_mul,
                    grp_matmul_gated_act_t::swiglu_oai_mul}) {
        for (const auto &M : ab_dists()) {
            Case c;
            c.M = M;
            c.act = act;
            expect_bitwise_equal(c);
        }
    }
}

TEST(PromptTileExecutor, BitwiseMatchesRoundsAcrossThreadCounts) {
    SKIP_WITHOUT_BF16_CUSTOM_KERNEL();
    for (const int T : {1, 3, 6, 8, 32}) {
        for (const auto act : {grp_matmul_gated_act_t::none,
                     grp_matmul_gated_act_t::swiglu_oai_mul}) {
            for (const size_t d : {size_t(0), size_t(2)}) {
                Case c;
                c.M = ab_dists()[d];
                c.act = act;
                c.threads = T;
                expect_bitwise_equal(c);
            }
        }
    }
}

TEST(PromptTileExecutor, BitwiseMatchesRoundsAcrossKnobs) {
    SKIP_WITHOUT_BF16_CUSTOM_KERNEL();
    struct KnobValue {
        pt::Knob knob;
        int value;
        const char *name;
    };
    const std::vector<KnobValue> knobs = {
            {pt::kMb, 8, "mb=8"},
            {pt::kMb, 24, "mb=24"},
            {pt::kNs, 32, "ns=32"},
            {pt::kNs, 96, "ns=96"},
            {pt::kOrder, 0, "order=0"},
            {pt::kOrder, 1, "order=1"},
            {pt::kTeam, 3, "team=3"},
            {pt::kTeam, 1, "team=1"},
            {pt::kSubteam, 0, "subteam=0"},
            {pt::kSubteam, 1, "subteam=1"},
            {pt::kSteal, 0, "steal=0"},
            {pt::kGrab, 3, "grab=3"},
            {pt::kMinTiles, 64, "min_tiles=64"},
            {pt::kMbMax, 16, "mb_max=16"},
            {pt::kSbL3Pct, 1, "sb_l3_pct=1"},
            {pt::kSpanPct, 1, "span_pct=1"},
    };
    for (const auto &k : knobs) {
        SCOPED_TRACE(k.name);
        PtOverride o(k.knob, k.value);
        for (const size_t d : {size_t(0), size_t(3)}) {
            Case c;
            c.M = ab_dists()[d];
            c.act = grp_matmul_gated_act_t::silu_and_mul;
            c.threads = 6;
            expect_bitwise_equal(c);
        }
    }
}

TEST(PromptTileExecutor, BitwiseMatchesRoundsWithBiasAndF32Dst) {
    SKIP_WITHOUT_BF16_CUSTOM_KERNEL();
    const auto none = grp_matmul_gated_act_t::none;
    const auto swiglu = grp_matmul_gated_act_t::swiglu_oai_mul;
    struct V {
        grp_matmul_gated_act_t act;
        data_type_t bias, dst;
        int K;
    };
    for (const V v : {V {none, data_type_t::bf16, data_type_t::bf16, 128},
                 V {none, data_type_t::f32, data_type_t::bf16, 160},
                 V {swiglu, data_type_t::bf16, data_type_t::bf16, 128},
                 V {none, data_type_t::none, data_type_t::f32, 128},
                 V {none, data_type_t::f32, data_type_t::f32, 96}}) {
        Case c;
        c.M = ab_dists()[0];
        c.act = v.act;
        c.bias_dt = v.bias;
        c.dst_dt = v.dst;
        c.K = v.K;
        expect_bitwise_equal(c);
    }
}

// Pack NR 64 caps the BF16 MR at 6, so the CK splits a 7-row block into MR
// calls of 4 and 3 rows, and an MR <= 3 call accumulates on a second path.
// No tile may cut such a block out of a taller expert: forced blocks of 8
// and 16 rows leave 7-row tails on most of these experts.
TEST(PromptTileExecutor, BitwiseMatchesRoundsAtPackNr64) {
    SKIP_WITHOUT_BF16_CUSTOM_KERNEL();
    const auto none = grp_matmul_gated_act_t::none;
    const auto swiglu = grp_matmul_gated_act_t::swiglu_oai_mul;
    for (const int mb : {8, 16}) {
        SCOPED_TRACE("mb=" + std::to_string(mb));
        PtOverride o(pt::kMb, mb);
        for (const auto &v : {std::make_pair(none, data_type_t::bf16),
                     std::make_pair(none, data_type_t::f32),
                     std::make_pair(swiglu, data_type_t::bf16)}) {
            Case c;
            c.M = {23, 39, 15, 7, 71, 0, 100, 135, 47};
            c.act = v.first;
            c.dst_dt = v.second;
            c.nr = 64;
            c.wide = true;
            expect_bitwise_equal(c);
        }
    }
}

TEST(PromptTileExecutor, F16BitwiseMatchesRounds) {
    SKIP_WITHOUT_F16_CUSTOM_KERNEL();
    for (const auto act :
            {grp_matmul_gated_act_t::none, grp_matmul_gated_act_t::silu_and_mul,
                    grp_matmul_gated_act_t::gelu_and_mul,
                    grp_matmul_gated_act_t::swiglu_oai_mul}) {
        Case c;
        c.M = ab_dists()[0];
        c.src_dt = data_type_t::f16;
        c.dst_dt = data_type_t::f16;
        c.act = act;
        expect_bitwise_equal(c);
    }

    for (const auto bias : {data_type_t::f16, data_type_t::f32}) {
        Case c;
        c.M = {23, 39, 15, 7, 71, 0, 100};
        c.src_dt = data_type_t::f16;
        c.dst_dt = data_type_t::f32;
        c.bias_dt = bias;
        c.nr = 64;
        expect_bitwise_equal(c);
    }

    // NR64 caps max MR at 6. Forced 8-row blocks leave short tails while the
    // gated epilogue writes the tight half-width FP16 destination.
    PtOverride mb(pt::kMb, 8);
    for (const auto act : {grp_matmul_gated_act_t::silu_and_mul,
                 grp_matmul_gated_act_t::gelu_and_mul}) {
        Case c;
        c.M = {23, 39, 15, 7, 71, 0, 100, 135, 47};
        c.src_dt = data_type_t::f16;
        c.dst_dt = data_type_t::f16;
        c.act = act;
        c.nr = 64;
        expect_bitwise_equal(c);
    }
}

TEST(PromptTileExecutor, F16FamilyGate) {
    namespace ck = zendnnl::lowoha::matmul::custom_kernel;
    ck::CallContext kctx;
    kctx.enabled = true;
    kctx.variant = ck::KernelVariant::kF16_F16_F16;
    kctx.pack_nr = 32;
    const std::vector<int> M = {64, 96};
    const std::vector<int> N = {128, 128};

    {
        PtOverride f16_on(pt::kEnable, 8);
        EXPECT_TRUE(pt::should_run(kctx, /*use_custom=*/true,
                /*rounds_strategy=*/true, /*prompt_phase=*/true, M, N));
    }
    {
        PtOverride bf16_only(pt::kEnable, 1);
        EXPECT_FALSE(pt::should_run(kctx, /*use_custom=*/true,
                /*rounds_strategy=*/true, /*prompt_phase=*/true, M, N));
    }
}

TEST(PromptTileExecutor, F16EngagementGates) {
    SKIP_WITHOUT_F16_CUSTOM_KERNEL();
    Case c;
    c.M = ab_dists()[0];
    c.src_dt = data_type_t::f16;
    c.dst_dt = data_type_t::f16;

    const RunOut wrong_family = run_case(c, /*BF16 only=*/1);
    EXPECT_EQ(wrong_family.mode.find("ptile"), std::string::npos)
            << wrong_family.mode;
    EXPECT_EQ(wrong_family.served, 0);

    c.M = {0, 300, 0};
    const RunOut single = run_case(c, /*FP16=*/8);
    EXPECT_EQ(single.mode.find("ptile"), std::string::npos) << single.mode;
    EXPECT_EQ(single.served, 0);

    c.M = {4, 8, 16, 32};
    const RunOut decode = run_case(c, /*FP16=*/8);
    EXPECT_EQ(decode.mode.find("ptile"), std::string::npos) << decode.mode;
    EXPECT_EQ(decode.served, 0);
}

// ── Engagement gates ────────────────────────────────────────────────

TEST(PromptTileExecutor, DecodeCallsNeverEngage) {
    Case c;
    c.M = {4, 8, 2, 16, 0, 32, 1};
    for (const auto act : {grp_matmul_gated_act_t::none,
                 grp_matmul_gated_act_t::silu_and_mul}) {
        c.act = act;
        const RunOut r = run_case(c, 1);
        EXPECT_EQ(r.mode.find("ptile"), std::string::npos) << r.mode;
        EXPECT_EQ(r.served, 0);
    }
}

// A single-expert BF16 call (a dense layer) runs the tile executor, which
// splits its M across teams, bitwise equal to rounds.
TEST(PromptTileExecutor, SingleBf16ExpertMatchesRoundsBitwise) {
    SKIP_WITHOUT_BF16_CUSTOM_KERNEL();
    using namespace moe_test_utils;
    SingleExpertPromptTileOverride on(1);
    for (const auto &M : {std::vector<int> {300}, std::vector<int> {2048}}) {
        for (const auto act : {grp_matmul_gated_act_t::none,
                     grp_matmul_gated_act_t::gelu_and_mul,
                     grp_matmul_gated_act_t::silu_and_mul}) {
            for (const auto dst : {data_type_t::bf16, data_type_t::f32}) {
                if (act != grp_matmul_gated_act_t::none
                        && dst == data_type_t::f32)
                    continue;
                Case c;
                c.M = M;
                c.act = act;
                c.dst_dt = dst;
                expect_bitwise_equal(c);
            }
        }
    }
    // embeddinggemma FFN-like widths: K 768, N 2304 (gated) and 768.
    for (const auto &[N, K, act] :
            {std::tuple {2304, 768, grp_matmul_gated_act_t::gelu_and_mul},
                    std::tuple {768, 1152, grp_matmul_gated_act_t::none}}) {
        Case c;
        c.M = {1024};
        c.N = N;
        c.K = K;
        c.act = act;
        c.wide = true;
        expect_bitwise_equal(c);
    }
}

// With the knob on a single-expert BF16 prompt runs 12-row CK calls; its
// output must match the 8-row calls the round executor makes with the knob
// off bit for bit.
TEST(PromptTileExecutor, SingleBf16ExpertWideMrMatchesNarrowRounds) {
    SKIP_WITHOUT_BF16_CUSTOM_KERNEL();
    using namespace moe_test_utils;
    for (const auto &[M, N, K, act] :
            {std::tuple {300, 256, 128, grp_matmul_gated_act_t::none},
                    std::tuple {1024, 2304, 768,
                            grp_matmul_gated_act_t::gelu_and_mul},
                    std::tuple {2048, 768, 1152, grp_matmul_gated_act_t::none},
                    std::tuple {517, 512, 640,
                            grp_matmul_gated_act_t::silu_and_mul}}) {
        Case c;
        c.M = {M};
        c.N = N;
        c.K = K;
        c.act = act;
        c.wide = true;
        SCOPED_TRACE(describe(c));
        RunOut narrow, wide;
        {
            SingleExpertPromptTileOverride off(0);
            narrow = run_case(c, 0);
        }
        {
            SingleExpertPromptTileOverride on(1);
            wide = run_case(c, 1);
        }
        ASSERT_EQ(narrow.mode.find("ptile"), std::string::npos) << narrow.mode;
        ASSERT_NE(wide.mode.find("flat_n_tile_ptile_"), std::string::npos)
                << wide.mode;
        ASSERT_EQ(narrow.dst[0].size(), wide.dst[0].size());
        EXPECT_EQ(std::memcmp(narrow.dst[0].data(), wide.dst[0].data(),
                          narrow.dst[0].size()),
                0);
    }
}

// Under AUTO with the knob on, a single BF16 expert reaches the prompt tile
// whenever the CK takes its epilogue, a plain GEMM included.  A split-halves
// gated activation with bias, which the CK refuses, stays on ALGO 1 rather
// than N-tile rounds without the CK.
TEST(PromptTileExecutor, AutoSingleExpertFollowsCkEpilogue) {
    SKIP_WITHOUT_BF16_CUSTOM_KERNEL();
    using namespace moe_test_utils;
    SingleExpertPromptTileOverride on(1);
    {
        Case plain;
        plain.M = {300};
        SCOPED_TRACE(describe(plain));
        const RunOut r = run_case(plain, 1, /*algo=*/0);
        EXPECT_NE(r.mode.find("ptile"), std::string::npos) << r.mode;
        EXPECT_EQ(r.served, 1);
    }
    for (const auto act : {grp_matmul_gated_act_t::gelu_and_mul,
                 grp_matmul_gated_act_t::silu_and_mul}) {
        Case c;
        c.M = {300};
        c.act = act;
        SCOPED_TRACE(describe(c));
        const RunOut served = run_case(c, 1, /*algo=*/0);
        EXPECT_NE(served.mode.find("ptile"), std::string::npos) << served.mode;
        EXPECT_EQ(served.served, 1);

        c.bias_dt = data_type_t::bf16;
        SCOPED_TRACE(describe(c));
        const RunOut refused = run_case(c, 1, /*algo=*/0);
        EXPECT_EQ(refused.mode.find("flat_n_tile"), std::string::npos)
                << refused.mode;
        EXPECT_EQ(refused.served, 0);
    }
}

// A fused BF16 dense FFN (W13, gelu_and_mul, W2: what ZenTorch's fused FFN
// sends) runs both passes on the prompt tile under AUTO with the knob on, at
// any size, and matches the AOCL two-pass (knob off) to BF16 tolerance.
TEST(PromptTileExecutor, AutoSingleExpertFusedFfnBothPasses) {
    SKIP_WITHOUT_BF16_CUSTOM_KERNEL();
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    struct Ffn {
        int H, I;
        bool tiles;
    };
    constexpr int M = 512;
    for (const Ffn f : {Ffn {256, 384, true}, Ffn {1536, 256, true}}) {
        SCOPED_TRACE("H=" + std::to_string(f.H) + " I=" + std::to_string(f.I));
        const int N13 = 2 * f.I;
        std::vector<bfloat16_t> src(static_cast<size_t>(M) * f.H),
                w13(static_cast<size_t>(f.H) * N13),
                w2(static_cast<size_t>(f.I) * f.H);
        for (size_t i = 0; i < src.size(); ++i)
            src[i] = bfloat16_t(0.03f
                    * static_cast<float>(static_cast<int>(i * 7 % 11) - 5));
        for (size_t i = 0; i < w13.size(); ++i)
            w13[i] = bfloat16_t(0.01f
                    * static_cast<float>(static_cast<int>(i * 5 % 13) - 6));
        for (size_t i = 0; i < w2.size(); ++i)
            w2[i] = bfloat16_t(0.01f
                    * static_cast<float>(static_cast<int>(i * 3 % 7) - 3));
        const std::vector<bfloat16_t> w13_plain = w13, w2_plain = w2;

        // The in-place weight cache may pack a weight in place, so each leg
        // starts from the plain bytes.
        const auto run = [&](int knob, TypedBuffers &out, std::string &mode) {
            std::memcpy(w13.data(), w13_plain.data(),
                    w13.size() * sizeof(bfloat16_t));
            std::memcpy(
                    w2.data(), w2_plain.data(), w2.size() * sizeof(bfloat16_t));
            reset_grp_matmul_caches();
            AlgoEnvGuard algo(0);
            CustomKernelOverride ck_on(true);
            SingleExpertPromptTileOverride ovr(knob);
            const std::vector<const void *> srcs {src.data()},
                    wei1 {w13.data()}, wei2 {w2.data()}, no_bias {nullptr};
            auto gv = GemmVecs::uniform(1, M, N13, f.H);
            auto p = make_uniform_params(1, data_type_t::bf16);
            p[0].num_threads = 8;
            grp_matmul_gated_act_params act {};
            act.act = grp_matmul_gated_act_t::gelu_and_mul;
            auto fused = make_fused_moe_op2(1, f.H, wei2, no_bias);
            out.alloc(1, static_cast<size_t>(M) * f.H, /*is_bf16=*/true);
            fused.dst_down = out.ptrs(true);
            fused.ldc_down = {f.H};
            const std::vector<void *> dst_null {nullptr};
            const std::vector<int> ldc_null {0};
            GemmModeCaptureGuard cap;
            ASSERT_EQ(group_matmul_direct(gv.layout, gv.transA, gv.transB,
                              gv.Ms, gv.Ns, gv.Ks, gv.alpha, srcs, gv.lda, wei1,
                              gv.ldb, no_bias, gv.beta, dst_null, ldc_null,
                              gv.is_wc, p, nullptr, &act, &fused),
                    status_t::success);
            const char *m = test_api::s_last_group_matmul_direct_gemm_mode.load(
                    std::memory_order_relaxed);
            mode = m != nullptr ? m : "";
        };
        const auto pass = [](const std::string &mode, const char *tag) {
            const size_t b = mode.find(tag);
            if (b == std::string::npos) return std::string();
            return mode.substr(b, mode.find_first_of(",)", b) - b);
        };

        TypedBuffers aocl, tiles;
        std::string mode_aocl, mode_tiles;
        run(0, aocl, mode_aocl);
        run(1, tiles, mode_tiles);
        EXPECT_EQ(mode_aocl.find("ptile"), std::string::npos) << mode_aocl;
        for (const char *tag : {"op1=", "op2="}) {
            const std::string p = pass(mode_tiles, tag);
            ASSERT_FALSE(p.empty()) << mode_tiles;
            EXPECT_EQ(p.find("ptile") != std::string::npos, f.tiles)
                    << mode_tiles;
        }
        verify_per_expert_2d(tiles, f.H, aocl, f.H, std::vector<int> {M}, f.H,
                data_type_t::bf16, tol_fused(data_type_t::bf16),
                "single_expert_fused_ffn");
    }
}

// The single-expert scope is a call of one expert: one active expert among
// idle MoE slots keeps the round executor with the knob on, as does a single
// expert with the knob off.  FP16 keeps the multi-expert scope.
TEST(PromptTileExecutor, SingleActiveExpertOutsideScopeNeverEngages) {
    SKIP_WITHOUT_BF16_CUSTOM_KERNEL();
    using namespace moe_test_utils;
    for (const auto &[knob, M] : {std::pair {1, std::vector<int> {0, 300, 0}},
                 std::pair {0, std::vector<int> {300}},
                 std::pair {0, std::vector<int> {0, 300, 0}}}) {
        SingleExpertPromptTileOverride ovr(knob);
        Case c;
        c.M = M;
        SCOPED_TRACE(describe(c) + " knob=" + std::to_string(knob));
        const RunOut r = run_case(c, 1);
        EXPECT_NE(r.mode.find("flat_n_tile"), std::string::npos) << r.mode;
        EXPECT_EQ(r.mode.find("ptile"), std::string::npos) << r.mode;
        EXPECT_EQ(r.served, 0);
    }
    if (!zendnnl::lowoha::matmul::custom_kernel::avx512f16_available()) return;
    SingleExpertPromptTileOverride on(1);
    Case c;
    c.M = {300};
    c.src_dt = data_type_t::f16;
    c.dst_dt = data_type_t::f16;
    SCOPED_TRACE(describe(c));
    const RunOut r = run_case(c, 8);
    EXPECT_EQ(r.mode.find("ptile"), std::string::npos) << r.mode;
    EXPECT_EQ(r.served, 0);
}

TEST(PromptTileExecutor, KnobOffOrNonCustomNeverEngage) {
    Case c;
    c.M = ab_dists()[0];
    // A mask without the BF16 bit keeps BF16 calls on the rounds.
    for (const int mask : {0, 2, 4, 6}) {
        const RunOut off = run_case(c, mask);
        EXPECT_EQ(off.mode.find("ptile"), std::string::npos)
                << "enable=" << mask << " " << off.mode;
        EXPECT_NE(off.mode.find("flat_n_tile"), std::string::npos)
                << "enable=" << mask << " " << off.mode;
        EXPECT_EQ(off.served, 0) << "enable=" << mask;
    }

    // Wide fused swiglu (ldc == N) runs off the CK, so it keeps its executor.
    c.act = grp_matmul_gated_act_t::swiglu_oai_mul;
    c.tight = false;
    const RunOut wide = run_case(c, 1);
    EXPECT_EQ(wide.mode.find("ptile"), std::string::npos) << wide.mode;
    EXPECT_EQ(wide.served, 0);
}

TEST(PromptTileExecutor, CustomKernelOffNeverEngages) {
    using namespace moe_test_utils;
    using namespace zendnnl::lowoha::matmul;
    const std::vector<int> M = ab_dists()[0];
    const int num_ops = static_cast<int>(M.size()), N = 256, K = 128;
    std::vector<std::vector<bfloat16_t>> src(num_ops), wei(num_ops),
            dst(num_ops);
    std::vector<const void *> srcs(num_ops), weis(num_ops),
            biases(num_ops, nullptr);
    std::vector<void *> dsts(num_ops);
    for (int e = 0; e < num_ops; ++e) {
        const int m = std::max(M[e], 1);
        src[e].assign(static_cast<size_t>(m) * K, bfloat16_t(0.01f));
        wei[e].assign(static_cast<size_t>(K) * N, bfloat16_t(0.02f));
        dst[e].assign(static_cast<size_t>(m) * N, bfloat16_t(0.0f));
        srcs[e] = src[e].data();
        weis[e] = wei[e].data();
        dsts[e] = dst[e].data();
    }
    auto gv = GemmVecs::uniform(num_ops, 1, N, K);
    gv.Ms = M;
    auto params = make_uniform_params(num_ops, data_type_t::bf16);
    for (auto &p : params)
        p.num_threads = 8;
    NTileStrategyOverride strat(2);
    CustomKernelOverride ck_off(false);
    AlgoEnvGuard algo3(3);
    PtOverride enable(pt::kEnable, 1);
    reset_grp_matmul_caches();
    const long served0 = pt::test_api::s_calls_served.load();
    const char *mode = nullptr;
    group_matmul_run_parallel_dispatch(gv.layout, gv.transA, gv.transB, gv.Ms,
            gv.Ns, gv.Ks, gv.alpha, srcs, gv.lda, weis, gv.ldb, biases, gv.beta,
            dsts, gv.ldc, gv.is_wc, params, 8, &mode,
            grp_matmul_gated_act_t::none, data_type_t::bf16);
    const std::string m = mode != nullptr ? mode : "";
    EXPECT_EQ(m.find("ptile"), std::string::npos) << m;
    EXPECT_EQ(pt::test_api::s_calls_served.load() - served0, 0);
}

// ── AUTO prompt routing ─────────────────────────────────────────────

struct Probe {
    std::vector<char> layout;
    std::vector<int> M, N, K;
    std::vector<zendnnl::lowoha::matmul::matmul_params> params;
};

/// Row-major selector probe; `dt` is the src and weight type.
Probe auto_probe(int m, int n, int k, int num_ops,
        data_type_t dt = data_type_t::bf16,
        data_type_t dst = data_type_t::bf16) {
    Probe p;
    p.layout.assign(num_ops, 'r');
    p.M.assign(num_ops, m);
    p.N.assign(num_ops, n);
    p.K.assign(num_ops, k);
    p.params.resize(num_ops);
    for (auto &q : p.params) {
        q.dtypes.src = dt;
        q.dtypes.wei = dt;
        q.dtypes.dst = dst;
        q.mem_format_a = 'n';
        q.mem_format_b = 'n';
    }
    return p;
}

using Pick = std::pair<int, std::string>;
const Pick kAlgo3 {3, "auto_algo3"};
const Pick kShapeRulePrompt {1, "auto_rule07_prompt_seq"};

/// AUTO's ALGO for `p` and the rule that picked it.
Pick auto_pick(const Probe &p, int threads = 32) {
    namespace mm = zendnnl::lowoha::matmul;
    mm::auto_algo_trace trace;
    const int algo = mm::select_grp_matmul_algo(
            p.layout, p.M, p.N, p.K, p.params, threads, &trace);
    return {algo, trace.reason != nullptr ? trace.reason : "<none>"};
}

TEST(PromptTileAuto, AutoSendsBf16PromptsToAlgo3) {
    using namespace moe_test_utils;
    AlgoEnvGuard algo0(0);
    CustomKernelOverride ck_on(true);
    PtOverride enable(pt::kEnable, 1);
    reset_grp_matmul_caches();

    // Routed gate-up and down projections; the lone shared expert keeps the
    // shape rules, whose prompt runs ALGO 1.
    EXPECT_EQ(auto_pick(auto_probe(300, 1024, 2048, 64)), kAlgo3);
    EXPECT_EQ(auto_pick(auto_probe(300, 2048, 512, 64)), kAlgo3);
    EXPECT_EQ(auto_pick(auto_probe(4096, 1024, 2048, 1)), kShapeRulePrompt);
    EXPECT_EQ(auto_pick(auto_probe(
                      300, 1024, 2048, 8, data_type_t::bf16, data_type_t::f32)),
            kAlgo3);
    // Decode takes the same route.
    EXPECT_EQ(auto_pick(auto_probe(16, 1024, 2048, 128)), kAlgo3);
}

TEST(PromptTileAuto, AutoSendsF16PromptsToAlgo3) {
    namespace mm = zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    AlgoEnvGuard algo0(0);
    CustomKernelOverride ck_on(true);
    CustomKernelF16Override ck_f16_on(true);
    reset_grp_matmul_caches();

    EXPECT_EQ(auto_pick(auto_probe(
                      300, 1024, 2048, 64, data_type_t::f16, data_type_t::f16)),
            kAlgo3);
    EXPECT_EQ(auto_pick(auto_probe(
                      16, 1024, 2048, 128, data_type_t::f16, data_type_t::f32)),
            kAlgo3);
    EXPECT_EQ(auto_pick(auto_probe(
                      16, 1024, 2048, 300, data_type_t::f16, data_type_t::f16)),
            kAlgo3);
    EXPECT_EQ(auto_pick(auto_probe(
                      300, 1024, 2048, 1, data_type_t::f16, data_type_t::f16)),
            kShapeRulePrompt);

    {
        Probe mixed = auto_probe(
                300, 1024, 2048, 8, data_type_t::f16, data_type_t::f16);
        mixed.params[1].dtypes.src = data_type_t::bf16;
        mixed.params[1].dtypes.wei = data_type_t::bf16;
        EXPECT_FALSE(mm::grp_matmul_auto_algo3_call(mixed.M, mixed.params));
        EXPECT_EQ(auto_pick(mixed), kShapeRulePrompt);
    }
    {
        Probe mismatch = auto_probe(
                300, 1024, 2048, 8, data_type_t::f16, data_type_t::f16);
        for (auto &p : mismatch.params)
            p.dtypes.wei = data_type_t::bf16;
        EXPECT_FALSE(
                mm::grp_matmul_auto_algo3_call(mismatch.M, mismatch.params));
        EXPECT_EQ(auto_pick(mismatch), kShapeRulePrompt);
    }
    {
        Probe packed = auto_probe(
                300, 1024, 2048, 8, data_type_t::f16, data_type_t::f16);
        for (auto &p : packed.params) {
            p.mem_format_b = 'r';
            p.lowoha_algo = zendnnl::common::matmul_algo_t::aocl_dlp_blocked;
        }
        EXPECT_FALSE(mm::grp_matmul_auto_algo3_call(packed.M, packed.params));
        EXPECT_EQ(auto_pick(packed), kShapeRulePrompt);
    }
}

// AUTO does not ask whether the tile executor will run: every case here is
// one it would refuse, and each still runs ALGO 3 without it.  A call ALGO 3
// cannot serve keeps the shape rules.  Holds on any host.
TEST(PromptTileAuto, AutoRunsAlgo3WhenTheExecutorWouldNotRun) {
    using namespace moe_test_utils;
    AlgoEnvGuard algo0(0);
    CustomKernelOverride ck_on(true);
    PtOverride enable(pt::kEnable, 1);
    reset_grp_matmul_caches();

    for (const int mask : {0, 2, 4, 6}) {
        PtOverride off(pt::kEnable, mask);
        EXPECT_EQ(auto_pick(auto_probe(300, 1024, 2048, 64)), kAlgo3)
                << "enable=" << mask;
    }
    {
        CustomKernelOverride ck_off(false);
        EXPECT_EQ(auto_pick(auto_probe(300, 1024, 2048, 64)), kAlgo3);
    }
    {
        KblockOverride kblock_on(1);
        EXPECT_EQ(auto_pick(auto_probe(300, 1024, 2048, 64)), kAlgo3);
    }
    {
        NTileStrategyOverride decode_d(1);
        EXPECT_EQ(auto_pick(auto_probe(300, 1024, 2048, 64)), kAlgo3);
    }
    // No CK pack NR divides N.
    EXPECT_EQ(auto_pick(auto_probe(300, 1000, 2048, 64)), kAlgo3);
    // An f32 layer keeps the shape rules, whose prompt runs ALGO 1.
    EXPECT_EQ(auto_pick(auto_probe(
                      300, 1024, 2048, 64, data_type_t::f32, data_type_t::f32)),
            kShapeRulePrompt);
    // Not n_tile-safe.
    {
        Probe p = auto_probe(300, 1024, 2048, 64);
        p.layout.assign(p.layout.size(), 'c');
        EXPECT_EQ(auto_pick(p), kShapeRulePrompt);
    }
}

TEST(PromptTileAuto, ExplicitPromptPinOutranksTheDefault) {
    using namespace moe_test_utils;
    AlgoEnvGuard algo0(0);
    CustomKernelOverride ck_on(true);
    PtOverride enable(pt::kEnable, 1);
    reset_grp_matmul_caches();
    for (const int pin : {1, 2}) {
        AutoPromptAlgoOverride pin_prompt(pin);
        EXPECT_EQ(auto_pick(auto_probe(300, 1024, 2048, 64)),
                (Pick {pin, "auto_phase_env"}));
    }
}

TEST(PromptTileAuto, AutoPromptRunsTheTileExecutor) {
    SKIP_WITHOUT_BF16_CUSTOM_KERNEL();
    for (const auto act :
            {grp_matmul_gated_act_t::none, grp_matmul_gated_act_t::silu_and_mul,
                    grp_matmul_gated_act_t::swiglu_oai_mul}) {
        Case c;
        c.M = ab_dists()[0];
        c.act = act;
        SCOPED_TRACE(describe(c));
        const RunOut pinned = run_case(c, 1);
        const RunOut autod = run_case(c, 1, /*algo=*/0);
        ASSERT_NE(autod.mode.find("flat_n_tile_ptile_"), std::string::npos)
                << autod.mode;
        ASSERT_EQ(autod.served, 1);
        for (size_t e = 0; e < pinned.dst.size(); ++e) {
            EXPECT_EQ(std::memcmp(pinned.dst[e].data(), autod.dst[e].data(),
                              pinned.dst[e].size()),
                    0)
                    << "expert " << e << " differs";
        }
    }
    // With the executor off, AUTO still runs the prompt on ALGO 3.
    Case c;
    c.M = ab_dists()[0];
    const RunOut off = run_case(c, 0, /*algo=*/0);
    EXPECT_NE(off.mode.find("flat_n_tile"), std::string::npos) << off.mode;
    EXPECT_EQ(off.mode.find("ptile"), std::string::npos) << off.mode;
    EXPECT_EQ(off.served, 0);
}

TEST(PromptTileAuto, AutoF16PromptRunsTheTileExecutor) {
    SKIP_WITHOUT_F16_CUSTOM_KERNEL();
    Case c;
    c.M = ab_dists()[0];
    c.src_dt = data_type_t::f16;
    c.dst_dt = data_type_t::f16;
    c.act = grp_matmul_gated_act_t::silu_and_mul;

    const RunOut pinned = run_case(c, /*FP16=*/8);
    const RunOut autod = run_case(c, /*FP16=*/8, /*algo=*/0);
    ASSERT_NE(autod.mode.find("flat_n_tile_ptile_"), std::string::npos)
            << autod.mode;
    ASSERT_EQ(autod.served, 1);
    for (size_t e = 0; e < pinned.dst.size(); ++e) {
        EXPECT_EQ(std::memcmp(pinned.dst[e].data(), autod.dst[e].data(),
                          pinned.dst[e].size()),
                0)
                << "expert " << e << " differs";
    }
}

struct DirectOut {
    zendnnl::lowoha::matmul::group_matmul_prepack::test_api::LastInvocationStats
            stats;
    std::vector<std::vector<bfloat16_t>> dst;
    bool weights_mutated = false;
};

/// One row-major BF16 `group_matmul_direct` call (N = K = 256) under the
/// caller's settings, with the prepack stats it left.
DirectOut run_direct(int m, int num_ops, int threads) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    constexpr int N = 256, K = 256;
    std::vector<std::vector<bfloat16_t>> src(num_ops), wei(num_ops);
    std::vector<const void *> srcs(num_ops), weis(num_ops),
            biases(num_ops, nullptr);
    std::vector<void *> dsts(num_ops);
    DirectOut out;
    out.dst.resize(num_ops);
    for (int e = 0; e < num_ops; ++e) {
        src[e].resize(static_cast<size_t>(m) * K);
        for (size_t i = 0; i < src[e].size(); ++i)
            src[e][i] = bfloat16_t(0.02f
                    * static_cast<float>(static_cast<int>((i + e) % 5) - 2));
        wei[e].resize(static_cast<size_t>(K) * N);
        for (size_t i = 0; i < wei[e].size(); ++i)
            wei[e][i] = bfloat16_t(0.01f
                    * static_cast<float>(static_cast<int>((i + e) % 7) - 3));
        out.dst[e].assign(static_cast<size_t>(m) * N, bfloat16_t(0.0f));
        srcs[e] = src[e].data();
        weis[e] = wei[e].data();
        dsts[e] = out.dst[e].data();
    }
    const auto wei_orig = wei;
    auto gv = GemmVecs::uniform(num_ops, m, N, K);
    auto params = make_uniform_params(num_ops, data_type_t::bf16);
    for (auto &p : params)
        p.num_threads = threads;
    LastInvocationCaptureGuard capture;
    EXPECT_EQ(group_matmul_direct(gv.layout, gv.transA, gv.transB, gv.Ms, gv.Ns,
                      gv.Ks, gv.alpha, srcs, gv.lda, weis, gv.ldb, biases,
                      gv.beta, dsts, gv.ldc, gv.is_wc, params,
                      /*moe_postop=*/nullptr, /*gated_act=*/nullptr),
            zendnnl::error_handling::status_t::success);
    out.stats = group_matmul_prepack::test_api::get_last_invocation_stats();
    for (int e = 0; e < num_ops; ++e) {
        if (std::memcmp(wei[e].data(), wei_orig[e].data(),
                    wei[e].size() * sizeof(bfloat16_t))
                != 0)
            out.weights_mutated = true;
    }
    return out;
}

// A decode prepack under AUTO cross-warms the full-weight layout only for a
// prompt pinned off ALGO 3, whether or not the tile executor is on.
TEST(PromptTileAuto, Algo3CrossWarmSkipsTheFullWeightUnlessPromptIsPinned) {
    SKIP_GRP_MATMUL_TESTS_WITHOUT_AOCL_DLP();
    SKIP_WITHOUT_BF16_CUSTOM_KERNEL();
    using namespace moe_test_utils;
    namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;
    AlgoEnvGuard algo0(0);
    CustomKernelOverride ck_on(true);
    WeightCacheGuard wc(1);
    for (const int pt_on : {1, 0}) {
        for (const int prompt_pin : {-1, 1}) {
            SCOPED_TRACE(std::string(pt_on != 0 ? "prompt tile on"
                                                : "prompt tile off")
                    + " prompt_pin=" + std::to_string(prompt_pin));
            PtOverride enable(pt::kEnable, pt_on);
            AutoPromptAlgoOverride pin(prompt_pin);
            reset_grp_matmul_caches();
            const DirectOut d
                    = run_direct(/*m=*/8, /*num_ops=*/8, /*threads=*/32);
            ASSERT_TRUE(d.stats.valid);
            EXPECT_EQ(d.stats.scheduling_algo, 3);
            if (prompt_pin < 0) {
                EXPECT_EQ(d.stats.cross_warm_regime,
                        prepack::CrossWarmRegime::none);
                EXPECT_EQ(d.stats.aocl.total_attempted, 0);
            } else {
                EXPECT_EQ(d.stats.cross_warm_regime,
                        prepack::CrossWarmRegime::aocl_full_weight);
                EXPECT_EQ(d.stats.aocl.total_attempted, 8);
            }
        }
    }
}

// Under WC=2 both AUTO phases run ALGO 3, so the CK pack is the weights' only
// layout: the BF16 prompt on the tile executor packs them in place, nothing
// is cross-warmed, and the output matches WC=1 bit for bit.
TEST(PromptTileAuto, AutoBf16PromptPacksWeightsInPlaceUnderWc2) {
    SKIP_GRP_MATMUL_TESTS_WITHOUT_AOCL_DLP();
    SKIP_WITHOUT_BF16_CUSTOM_KERNEL();
    using namespace moe_test_utils;
    namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;
    AlgoEnvGuard algo0(0);
    CustomKernelOverride ck_on(true);
    PtOverride enable(pt::kEnable, 1);
    DirectOut ref, in_place;
    {
        WeightCacheGuard wc(1);
        reset_grp_matmul_caches();
        ref = run_direct(/*m=*/64, /*num_ops=*/8, /*threads=*/32);
    }
    {
        WeightCacheGuard wc(2);
        reset_grp_matmul_caches();
        in_place = run_direct(/*m=*/64, /*num_ops=*/8, /*threads=*/32);
        auto &cfg = zendnnl::common::matmul_config_t::instance();
        EXPECT_EQ(cfg.get_weight_cache(), 2);
        EXPECT_TRUE(cfg.get_grp_auto_mixed_inplace());
    }
    ASSERT_TRUE(in_place.stats.valid);
    EXPECT_EQ(in_place.stats.scheduling_algo, 3);
    EXPECT_EQ(in_place.stats.cross_warm_regime, prepack::CrossWarmRegime::none);
    EXPECT_FALSE(ref.weights_mutated);
    EXPECT_TRUE(in_place.weights_mutated);
    for (size_t e = 0; e < ref.dst.size(); ++e) {
        EXPECT_EQ(std::memcmp(ref.dst[e].data(), in_place.dst[e].data(),
                          ref.dst[e].size() * sizeof(bfloat16_t)),
                0)
                << "expert " << e << " differs";
    }
}

// ── Micro-benchmark ─────────────────────────────────────────────────

// Per-call cost of what `execute` does before its OMP region: resolve the
// knobs, look up the team map and plan.  Run with
// --gtest_also_run_disabled_tests --gtest_filter='*ResolveAndPlan*'.
TEST(PromptTileBench, DISABLED_ResolveAndPlan) {
    namespace ck = zendnnl::lowoha::matmul::custom_kernel;
    struct Shape {
        const char *name;
        std::vector<int> M;
        int N, K, T;
    };
    const std::vector<Shape> shapes = {
            {"qwen3.6_w13_t1024", golden_m(256, 160, 1), 1024, 2048, 128},
            {"qwen3.6_w2_t4096", golden_m(256, 640, 2), 2048, 512, 128},
            {"gptoss_w13_t4096", golden_m(32, 4000, 4), 5760, 2880, 64},
    };
    ck::CallContext kctx;
    kctx.variant = ck::KernelVariant::kBF16_BF16_BF16;
    kctx.pack_nr = 32;
    kctx.max_mr = 8;
    for (const auto &s : shapes) {
        const size_t n = s.M.size();
        const std::vector<int> N(n, s.N), K(n, s.K);
        const std::vector<const void *> src(n, nullptr), bias(n, nullptr);
        const std::vector<void *> dst(n, nullptr);
        const pt::CallView call {s.M, N, K, src, K, bias, dst, N};
        size_t tiles = 0;
        for (int i = 0; i < 200; ++i)
            tiles = pt::test_api::plan_only(kctx, call, s.T).tiles.size();
        constexpr int kIters = 2000;
        double best = 1e300;
        for (int rep = 0; rep < 7; ++rep) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < kIters; ++i)
                pt::test_api::plan_only(kctx, call, s.T);
            const std::chrono::duration<double, std::nano> d
                    = std::chrono::steady_clock::now() - t0;
            best = std::min(best, d.count() / kIters);
        }
        std::printf("[PTILE_BENCH] %s tiles=%zu ns_per_call=%.0f\n", s.name,
                tiles, best);
    }
}

} // namespace
