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
#include "lowoha_operators/matmul/group_matmul/executors/n_tile/prompt_tile/prompt_tile_executor.hpp"
#include "lowoha_operators/matmul/group_matmul/executors/n_tile/prompt_tile/prompt_tile_planner.hpp"
#include "lowoha_operators/matmul/group_matmul/executors/n_tile/prompt_tile/prompt_tile_topology.hpp"
#include "lowoha_operators/matmul/group_matmul/executors/n_tile/prompt_tile/prompt_tile_tunables.hpp"
#include "moe_test_utils.hpp"

// GTEST_SKIP returns from the enclosing function, so call these from TEST
// bodies.
#define SKIP_WITHOUT_BF16_CUSTOM_KERNEL() \
    do { \
        if (!zendnnl::lowoha::matmul::custom_kernel::bf16_isa_supported()) { \
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
using zendnnl::lowoha::matmul::executed_algo_from_execution_path;
using zendnnl::lowoha::matmul::grp_matmul_gated_act_t;

/// Sets one prompt-tile knob for the scope.
using PtOverride = pt::test_api::ScopedOverride;

/// Shallowest even K whose B for one prompt call (`blocks` NR32 column
/// blocks) overflows this host's custom-kernel L2 budget, so the default
/// K-chunk gate engages whatever the L2 size.
int prompt_kc_min_K(int blocks) {
    const int64_t k
            = zendnnl::lowoha::matmul::custom_kernel::ck_l2_budget_bytes()
                    / (int64_t {32} * blocks * 2)
            + 1;
    return static_cast<int>((k + 1) & ~int64_t {1});
}

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

    // A 262144-column LM head, a 100-row call, and an N with fewer pack-width
    // blocks than a team has threads keep the shrink loop.
    for (const auto &[m, n, k] : {std::tuple {256, 262144, 640},
                 std::tuple {100, 768, 768}, std::tuple {512, 32, 1024},
                 std::tuple {512, 64, 1024}, std::tuple {512, 96, 1024}}) {
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
        EXPECT_GE(dense.tiles.size(), 32u) << "every thread gets a tile";
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

/// Every thread of a per-thread plan owns one slot, the slots of a team tile
/// its range in rank order, and together they cover the tile list once.
void check_owner_slots(
        const std::vector<int> &team_size, const pt::Plan &plan) {
    int threads = 0;
    for (const int s : team_size)
        threads += s;
    ASSERT_EQ(plan.owner_begin.size(), static_cast<size_t>(threads) + 1);
    EXPECT_EQ(plan.owner_begin.front(), 0);
    EXPECT_EQ(plan.owner_begin.back(), static_cast<int>(plan.tiles.size()));
    int slot = 0;
    for (size_t t = 0; t < team_size.size(); ++t) {
        EXPECT_EQ(plan.owner_begin[slot], plan.team_begin[t]) << "team " << t;
        for (int r = 0; r < team_size[t]; ++r, ++slot)
            EXPECT_LE(plan.owner_begin[slot], plan.owner_begin[slot + 1]);
        EXPECT_EQ(plan.owner_begin[slot], plan.team_begin[t + 1])
                << "team " << t;
    }
}

// The per-thread plan keeps the busiest thread near M*N/T at thread counts
// the sub-team size does not divide (Turin 8-core and Venice 32-core L3
// domains), for every family's row unit and at pack NR 64; each thread owns
// one slot inside its team's range, its busiest thread costs no more than
// the family's own plan, and at T = 32 it costs the same as the dense split.
TEST(PromptTilePlanner, BalancedPlanBoundsTheBusiestThread) {
    struct Family {
        const char *name;
        int a_bytes, wei_bits, min_mr, row_align, max_mr, pack_nr;
    };
    const Family families[] = {{"bf16", 2, 16, 4, 8, 12, 32},
            {"bf16_nr64", 2, 16, 4, 8, 6, 64}, {"f16", 2, 16, 1, 12, 12, 32},
            {"int8", 1, 8, 1, 12, 12, 32}, {"w4a8", 1, 4, 1, 6, 6, 32}};
    struct Shape {
        int M, N, K;
    };
    const Shape shapes[] = {{2048, 1280, 768}, {2048, 768, 768},
            {2048, 2304, 768}, {2048, 768, 1152}, {2048, 1024, 640},
            {2048, 2048, 1024}, {2048, 1024, 3072}, {4096, 512, 256},
            {4096, 256, 1024}, {512, 1280, 768}};
    for (const int l3 : {8, 32}) {
        for (const int T : {14, 15, 16, 30, 31, 32, 33, 62, 63, 64}) {
            std::vector<int> dom(T);
            for (int t = 0; t < T; ++t)
                dom[t] = t / l3;
            const pt::TeamMap tm
                    = pt::make_domain_team_map(dom, 4, 1 << 20, 32 << 20);
            for (int t = 0; t < tm.n_teams; ++t) {
                std::vector<int> seen;
                for (int i = 0; i < T; ++i)
                    if (tm.team_of[i] == t) seen.push_back(tm.rank_in_team[i]);
                ASSERT_EQ(static_cast<int>(seen.size()), tm.team_size[t]);
                for (int r = 0; r < tm.team_size[t]; ++r)
                    EXPECT_EQ(seen[r], r) << "team " << t;
            }
            for (const Family &f : families) {
                for (const Shape &s : shapes) {
                    SCOPED_TRACE(std::string(f.name) + " l3="
                            + std::to_string(l3) + " T=" + std::to_string(T)
                            + " M=" + std::to_string(s.M)
                            + " N=" + std::to_string(s.N)
                            + " K=" + std::to_string(s.K));
                    pt::PlannerConfig cfg;
                    cfg.num_threads = T;
                    cfg.a_bytes = f.a_bytes;
                    cfg.wei_bits = f.wei_bits;
                    cfg.min_mr = f.min_mr;
                    cfg.row_align = f.row_align;
                    cfg.max_mr = f.max_mr;
                    cfg.pack_nr = f.pack_nr;
                    cfg.balanced = true;
                    pt::Plan plan;
                    pt::plan_tiles(&s.M, &s.N, &s.K, 1, cfg,
                            tm.team_size.data(), tm.n_teams, plan);
                    check_plan({s.M}, {s.N}, {s.K}, cfg, tm.team_size, plan);
                    ASSERT_NO_FATAL_FAILURE(
                            check_owner_slots(tm.team_size, plan));
                    const int rb = cfg.tun[pt::kRb];
                    const double ms = pt::plan_makespan(
                            plan, &s.K, rb, tm.team_size.data(), tm.n_teams);
                    // Worst over these shapes: 1.0625, a 12-row unit's
                    // rounding at 30 and 64 threads.
                    EXPECT_LE(ms, 1.065);
                    // The plan's own estimate, recounted from its tiles.
                    EXPECT_DOUBLE_EQ(ms, plan.est_imbalance);
                    // Busiest thread's absolute cost against the family's
                    // own plan (dense for BF16, shrink otherwise).
                    cfg.balanced = false;
                    cfg.single_dense = f.wei_bits == 16 && f.min_mr == 4;
                    pt::Plan legacy;
                    pt::plan_tiles(&s.M, &s.N, &s.K, 1, cfg,
                            tm.team_size.data(), tm.n_teams, legacy);
                    const double busiest = ms * plan.total_cost;
                    const double legacy_busiest
                            = pt::plan_makespan(legacy, &s.K, rb,
                                      tm.team_size.data(), tm.n_teams)
                            * legacy.total_cost;
                    EXPECT_LE(busiest, legacy_busiest * (1.0 + 1e-9));
                }
            }
        }
    }

    // The dense split runs a second whole tile on one thread at T = 31; the
    // per-thread plan does not, and at T = 32 both cost the same.
    const int M = 2048, N = 1280, K = 768;
    for (const int T : {31, 32}) {
        std::vector<int> dom(T);
        for (int t = 0; t < T; ++t)
            dom[t] = t / 8;
        const pt::TeamMap tm
                = pt::make_domain_team_map(dom, 4, 1 << 20, 32 << 20);
        pt::PlannerConfig cfg;
        cfg.num_threads = T;
        cfg.single_dense = true;
        pt::Plan dense, balanced;
        pt::plan_tiles(
                &M, &N, &K, 1, cfg, tm.team_size.data(), tm.n_teams, dense);
        cfg.balanced = true;
        pt::plan_tiles(
                &M, &N, &K, 1, cfg, tm.team_size.data(), tm.n_teams, balanced);
        const double d = pt::plan_makespan(
                dense, &K, cfg.tun[pt::kRb], tm.team_size.data(), tm.n_teams);
        const double b = pt::plan_makespan(balanced, &K, cfg.tun[pt::kRb],
                tm.team_size.data(), tm.n_teams);
        EXPECT_TRUE(dense.owner_begin.empty());
        if (T == 31) {
            EXPECT_GT(d, 1.8);
            EXPECT_LT(b, 1.05);
        } else {
            EXPECT_DOUBLE_EQ(d, 1.0);
            EXPECT_DOUBLE_EQ(b, 1.0);
            EXPECT_EQ(balanced.tiles.size(), dense.tiles.size());
        }
    }

    // A layer the dense split does not take keeps the shrink loop.
    for (const auto &[m, n, k] : {std::tuple {256, 262144, 640},
                 std::tuple {100, 768, 768}, std::tuple {512, 64, 1024}}) {
        const std::vector<int> teams(8, 4);
        pt::PlannerConfig cfg;
        cfg.num_threads = 32;
        pt::Plan plain, balanced;
        pt::plan_tiles(&m, &n, &k, 1, cfg, teams.data(), 8, plain);
        cfg.balanced = true;
        pt::plan_tiles(&m, &n, &k, 1, cfg, teams.data(), 8, balanced);
        EXPECT_TRUE(balanced.owner_begin.empty()) << "M=" << m;
        EXPECT_EQ(balanced.tiles.size(), plain.tiles.size()) << "M=" << m;
    }
}

// At pack NR 64 (MR 6) the rows a whole unit leaves never give a team a
// 7-row tile (an MR 3 call), nor does a deep K whose L2 run is shorter
// than one row block; a call no team can take them on keeps the dense
// split.
TEST(PromptTilePlanner, BalancedPlanKeepsMinMrAtPackNr64) {
    pt::PlannerConfig cfg;
    cfg.pack_nr = 64;
    cfg.max_mr = 6;
    cfg.balanced = true;
    cfg.single_dense = true;
    for (const int T : {15, 26, 31, 35, 63}) {
        std::vector<int> dom(T);
        for (int t = 0; t < T; ++t)
            dom[t] = t / 8;
        const pt::TeamMap tm
                = pt::make_domain_team_map(dom, 4, 1 << 20, 32 << 20);
        cfg.num_threads = T;
        for (int M = 101; M <= 411; M += 2) {
            for (const auto &[N, K] : {std::pair {256, 128},
                         std::pair {1280, 768}, std::pair {4096, 2048}}) {
                SCOPED_TRACE("T=" + std::to_string(T) + " M="
                        + std::to_string(M) + " N=" + std::to_string(N));
                pt::Plan plan;
                pt::plan_tiles(&M, &N, &K, 1, cfg, tm.team_size.data(),
                        tm.n_teams, plan);
                check_plan({M}, {N}, {K}, cfg, tm.team_size, plan);
            }
        }
    }
    // Four teams of 4, 4, 4 and 3 threads, 139 rows: the per-thread plan.
    const std::vector<int> teams {4, 4, 4, 3};
    const int M = 139, N = 256, K = 128;
    cfg.num_threads = 15;
    pt::Plan plan;
    pt::plan_tiles(&M, &N, &K, 1, cfg, teams.data(), 4, plan);
    check_plan({M}, {N}, {K}, cfg, teams, plan);
    ASSERT_NO_FATAL_FAILURE(check_owner_slots(teams, plan));
    // A 27251-deep K and a 512 KiB L2: runs of at most 5 rows.
    const std::vector<int> deep(9, 3);
    const int Md = 51, Nd = 512, Kd = 27251;
    pt::PlannerConfig dcfg = cfg;
    dcfg.num_threads = 27;
    dcfg.l2_bytes = 512 << 10;
    dcfg.l3_bytes_per_team = int64_t(1) << 32;
    dcfg.tun[pt::kMbMin] = 4;
    pt::Plan dplan;
    pt::plan_tiles(&Md, &Nd, &Kd, 1, dcfg, deep.data(), 9, dplan);
    check_plan({Md}, {Nd}, {Kd}, dcfg, deep, dplan);
    // 11 rows over teams of 4 and 3 threads: either team would run the 3
    // rows past two 4-row units as a 7-row tile, so the per-thread plan
    // declines and the dense split runs one 11-row tile.
    const std::vector<int> two {4, 3};
    const int Ms = 11, Ns = 320, Ks = 9184;
    pt::PlannerConfig scfg = dcfg;
    scfg.num_threads = 7;
    scfg.tun[pt::kMbMin] = 2;
    pt::Plan splan;
    pt::plan_tiles(&Ms, &Ns, &Ks, 1, scfg, two.data(), 2, splan);
    EXPECT_TRUE(splan.owner_begin.empty());
    check_plan({Ms}, {Ns}, {Ks}, scfg, two, splan);
}

// A team whose thread count does not divide N's pack blocks gives each
// thread `nblk / s` whole blocks and a piece of the rows of the blocks left
// over where that leaves its busiest thread less work than whole slices:
// 5 blocks over 4 threads cost 1.25 blocks a thread, not 2.
TEST(PromptTilePlanner, BalancedPlanSplitsLeftoverBlocksByRows) {
    const std::vector<int> teams(4, 4);
    const int M = 2048, N = 160, K = 2048;
    pt::PlannerConfig cfg;
    cfg.num_threads = 16;
    cfg.balanced = true;
    cfg.single_dense = true;
    pt::Plan plan;
    pt::plan_tiles(&M, &N, &K, 1, cfg, teams.data(), 4, plan);
    check_plan({M}, {N}, {K}, cfg, teams, plan);
    ASSERT_NO_FATAL_FAILURE(check_owner_slots(teams, plan));
    for (int slot = 0; slot < 16; ++slot) {
        int blocks = 0, piece_rows = 0;
        for (int i = plan.owner_begin[slot]; i < plan.owner_begin[slot + 1];
                ++i) {
            const pt::Tile &t = plan.tiles[i];
            if (t.n0 == 4 * cfg.pack_nr)
                piece_rows += t.rows;
            else
                blocks = std::max(blocks, t.ncols / cfg.pack_nr);
        }
        EXPECT_EQ(blocks, 1) << "slot " << slot;
        EXPECT_EQ(piece_rows, M / 16) << "slot " << slot;
    }
    const double ms
            = pt::plan_makespan(plan, &K, cfg.tun[pt::kRb], teams.data(), 4);
    EXPECT_LT(ms, 1.05);
}

// The per-thread plan replaces the team plan only where its busiest thread
// carries no more: N 640 at pack NR 64 is 10 blocks a 4-thread team drains
// in interleaved slices of 3, 3, 2 and 2 blocks, while fixed slices leave
// one thread 3 blocks of every row.
TEST(PromptTilePlanner, BalancedPlanNeverCostsMoreThanTheTeamPlan) {
    std::vector<int64_t> busy;
    const auto busiest
            = [&](const pt::Plan &p, const int *K, const pt::PlannerConfig &c,
                      const std::vector<int> &teams) {
        return pt::plan_busiest(p, K, c.tun[pt::kRb], teams.data(),
                static_cast<int>(teams.size()), busy);
    };
    const std::vector<int> sixteen(16, 4);
    const int M = 2048, N = 640, K = 3072;
    pt::PlannerConfig cfg;
    cfg.num_threads = 64;
    cfg.pack_nr = 64;
    cfg.max_mr = 6;
    cfg.single_dense = true;
    pt::Plan team_plan, chosen;
    pt::plan_tiles(&M, &N, &K, 1, cfg, sixteen.data(), 16, team_plan);
    cfg.balanced = true;
    pt::plan_tiles(&M, &N, &K, 1, cfg, sixteen.data(), 16, chosen);
    check_plan({M}, {N}, {K}, cfg, sixteen, chosen);
    EXPECT_TRUE(chosen.owner_begin.empty())
            << "kept the dearer per-thread plan";
    EXPECT_EQ(busiest(chosen, &K, cfg, sixteen),
            busiest(team_plan, &K, cfg, sixteen));

    // And over narrow-N layers at uneven thread counts.
    for (const int T : {15, 16, 31, 32, 63}) {
        std::vector<int> dom(T);
        for (int t = 0; t < T; ++t)
            dom[t] = t / 8;
        const pt::TeamMap tm
                = pt::make_domain_team_map(dom, 4, 1 << 20, 32 << 20);
        for (const int n : {128, 160, 192, 224, 320, 640})
            for (const int m : {256, 1024, 4096})
                for (const int k : {256, 2048}) {
                    SCOPED_TRACE("T=" + std::to_string(T) + " M="
                            + std::to_string(m) + " N=" + std::to_string(n)
                            + " K=" + std::to_string(k));
                    pt::PlannerConfig c;
                    c.num_threads = T;
                    c.single_dense = true;
                    pt::Plan base, plan;
                    pt::plan_tiles(&m, &n, &k, 1, c, tm.team_size.data(),
                            tm.n_teams, base);
                    c.balanced = true;
                    pt::plan_tiles(&m, &n, &k, 1, c, tm.team_size.data(),
                            tm.n_teams, plan);
                    check_plan({m}, {n}, {k}, c, tm.team_size, plan);
                    EXPECT_LE(busiest(plan, &k, c, tm.team_size),
                            busiest(base, &k, c, tm.team_size));
                }
    }
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
    static_assert(pt::kNumKnobs == 19, "set every knob below");
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
    cfg.tun[pt::kKc] = 0;
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
    EXPECT_EQ(pt::env_name(pt::kKc), "ZENDNNL_GRP_MATMUL_PROMPT_TILE_KC");
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
    const pt::kernel_traits_t &bf16 = pt::facts(pt::kernel_type_t::kBf16);
    EXPECT_EQ(cfg.tun.v, pt::defaults().v);
    EXPECT_EQ(cfg.tun[pt::kEnable], 15);
    EXPECT_EQ(cfg.a_bytes, bf16.a_bytes);
    EXPECT_EQ(cfg.wei_bits, bf16.wei_bits);
    EXPECT_EQ(cfg.min_mr, bf16.min_mr);
    EXPECT_EQ(cfg.row_align, bf16.row_align);
    const pt::kernel_traits_t &f16 = pt::facts(pt::kernel_type_t::kF16);
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
    EXPECT_EQ(p[0].fam, static_cast<int>(pt::kernel_type_t::kBf16));
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
    EXPECT_EQ(p[0].fam, static_cast<int>(pt::kernel_type_t::kF16));

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
    key.fam = pt::kernel_type_t::kInt8;
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

TEST(PromptTileTunables, FamilyDefaultsSitUnderEnvAndProfile) {
    for (const pt::kernel_type_t f :
            {pt::kernel_type_t::kBf16, pt::kernel_type_t::kInt8,
                    pt::kernel_type_t::kW4a8, pt::kernel_type_t::kF16}) {
        const pt::Tunables d = pt::defaults(f);
        for (int k = 0; k < pt::kNumKnobs; ++k) {
            if (f == pt::kernel_type_t::kW4a8 && k == pt::kMinTiles) continue;
            EXPECT_EQ(d.v[k], pt::kKnobs[k].def) << pt::kKnobs[k].name;
        }
    }
    EXPECT_EQ(pt::defaults(pt::kernel_type_t::kW4a8)[pt::kMinTiles], 32);

    pt::ShapeKey key;
    key.fam = pt::kernel_type_t::kW4a8;
    key.K = 2048;
    const pt::ProfileEntry none;
    EXPECT_EQ(pt::resolve_with(key, none, {}).t[pt::kMinTiles], 32);
    key.fam = pt::kernel_type_t::kInt8;
    EXPECT_EQ(pt::resolve_with(key, none, {}).t[pt::kMinTiles], 8);

    key.fam = pt::kernel_type_t::kW4a8;
    pt::ProfileEntry env;
    env.set = 1u << pt::kMinTiles;
    env.val[pt::kMinTiles] = 4;
    EXPECT_EQ(pt::resolve_with(key, env, {}).t[pt::kMinTiles], 4);
    pt::Profile profile;
    ASSERT_TRUE(pt::parse_profile("fam=w4a8:min_tiles=16", profile));
    EXPECT_EQ(pt::resolve_with(key, none, profile).t[pt::kMinTiles], 16);
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
        EXPECT_FALSE(pt::enabled(pt::kernel_type_t::kBf16));
        EXPECT_TRUE(pt::enabled(pt::kernel_type_t::kInt8));
        EXPECT_FALSE(pt::enabled(pt::kernel_type_t::kW4a8));
        EXPECT_FALSE(pt::enabled(pt::kernel_type_t::kF16));
    }
    {
        PtOverride mask(pt::kEnable, 8);
        EXPECT_FALSE(pt::enabled(pt::kernel_type_t::kBf16));
        EXPECT_FALSE(pt::enabled(pt::kernel_type_t::kInt8));
        EXPECT_FALSE(pt::enabled(pt::kernel_type_t::kW4a8));
        EXPECT_TRUE(pt::enabled(pt::kernel_type_t::kF16));
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
    int strategy = 2; // ZENDNNL_GRP_MATMUL_N_TILE_STRATEGY
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
            + " dst=" + std::to_string(static_cast<int>(c.dst_dt))
            + " nr=" + std::to_string(c.nr) + " wide=" + (c.wide ? "1" : "0")
            + " strategy=" + std::to_string(c.strategy) + " M=";
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
/// with the CK on and the strategy pinned (rounds unless the case sets it);
/// `prompt_tile` toggles the tile executor.
template <typename T>
RunOut run_case_typed(const Case &c, int prompt_tile, int algo) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    NTileStrategyOverride strat(c.strategy);
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
    group_matmul_execute(gv.layout, gv.transA, gv.transB, gv.Ms, gv.Ns, gv.Ks,
            gv.alpha, srcs, gv.lda, weis, gv.ldb, biases, gv.beta, dsts, gv.ldc,
            gv.is_wc, params, c.threads, &mode, c.act, c.src_dt);
    out.mode = mode != nullptr ? mode : "";
    out.served = pt::test_api::s_calls_served.load() - served0;
    // The weights die with this call; a pack they leave registered (in place
    // under WEIGHT_CACHE=2) would alias a later test's buffer at their address.
    reset_grp_matmul_caches();
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
        EXPECT_TRUE(pt::should_execute_prompt_tile(kctx, /*use_custom=*/true,
                /*rounds_strategy=*/true, /*prompt_phase=*/true, M, N,
                /*single_expert_prompt=*/false));
    }
    {
        PtOverride bf16_only(pt::kEnable, 1);
        EXPECT_FALSE(pt::should_execute_prompt_tile(kctx, /*use_custom=*/true,
                /*rounds_strategy=*/true, /*prompt_phase=*/true, M, N,
                /*single_expert_prompt=*/false));
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

// ── Single-expert policy (see single_expert/ for the policy tests) ──

// A single-expert BF16 call (a dense layer) runs the tile executor, which
// splits its M across teams, bitwise equal to rounds.  These layers are
// narrower than decode's N-tile count, so only `..._FORCE_CK` covers them.
TEST(PromptTileExecutor, SingleBf16ExpertMatchesRoundsBitwise) {
    SKIP_WITHOUT_BF16_CUSTOM_KERNEL();
    using namespace moe_test_utils;
    SingleExpertCkPolicy on(1, 1, 1);
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
    // embeddinggemma FFN-like widths: K 768, N 2304 (gated) and 768, also at
    // thread counts the sub-team size does not divide (per-thread plan),
    // under the detected team map and under teams of 4 (4, 4, 4, 3 at 15
    // threads), which a host's pinning cannot make even.  N 160 (5 blocks)
    // splits its leftover block by rows, and M 1001 leaves rows past whole
    // units.  The BF16 mask is set, whatever the environment says.
    namespace ck = zendnnl::lowoha::matmul::custom_kernel;
    SingleExpertBalanceMasks balance(
            static_cast<int>(zendnnl::lowoha::matmul::kSingleExpertBf16), -1);
    const auto owner_slots = [](int M, int N, int K, int T) {
        const std::vector<int> Ms {M}, Ns {N}, Ks {K};
        const std::vector<const void *> src(1, nullptr), bias(1, nullptr);
        const std::vector<void *> dst(1, nullptr);
        const pt::CallView call {Ms, Ns, Ks, src, Ks, bias, dst, Ns};
        ck::CallContext kctx;
        kctx.variant = ck::KernelVariant::kBF16_BF16_BF16;
        kctx.pack_nr = 32;
        kctx.max_mr = 12;
        return pt::test_api::plan_only(kctx, call, T).owner_begin.size();
    };
    for (const int team : {0, 4}) {
        PtOverride teams(pt::kTeam, team);
        for (const int T : {0, 15, 31}) {
            for (const auto &[M, N, K, act] :
                    {std::tuple {1024, 2304, 768,
                             grp_matmul_gated_act_t::gelu_and_mul},
                            std::tuple {1024, 768, 1152,
                                    grp_matmul_gated_act_t::none},
                            std::tuple {1001, 160, 512,
                                    grp_matmul_gated_act_t::none}}) {
                Case c;
                c.M.assign(1, M);
                c.N = N;
                c.K = K;
                c.act = act;
                c.wide = true;
                if (T > 0) c.threads = T;
                SCOPED_TRACE("team=" + std::to_string(team));
                if (team == 4 && T > 0 && N != 160) {
                    EXPECT_EQ(
                            owner_slots(M, N, K, T), static_cast<size_t>(T) + 1)
                            << "per-thread plan not taken";
                }
                expect_bitwise_equal(c);
            }
        }
    }
}

// `..._SINGLE_EXPERT_PROMPT_BALANCE` lists the dtypes whose single-expert
// prompt gets the per-thread plan (one owner slot per thread): BF16 by
// default, FP16 only when listed, and none at `0`.
TEST(PromptTileExecutor, SingleExpertPromptBalanceFollowsItsDtypeMask) {
    namespace ck = zendnnl::lowoha::matmul::custom_kernel;
    namespace mm = zendnnl::lowoha::matmul;
    PtOverride teams(pt::kTeam, 4);
    const std::vector<int> M {1024}, N {768}, K {768};
    const std::vector<const void *> src(1, nullptr), bias(1, nullptr);
    const std::vector<void *> dst(1, nullptr);
    const pt::CallView call {M, N, K, src, K, bias, dst, N};
    const auto slots = [&](ck::KernelVariant v, int mask) {
        moe_test_utils::SingleExpertBalanceMasks masks(mask, -1);
        ck::CallContext kctx;
        kctx.variant = v;
        kctx.pack_nr = 32;
        kctx.max_mr = 8;
        return pt::test_api::plan_only(kctx, call, 15).owner_begin.size();
    };
    const int bf16 = static_cast<int>(mm::kSingleExpertBf16);
    const int all = static_cast<int>(mm::kSingleExpertAll);
    EXPECT_EQ(slots(ck::KernelVariant::kBF16_BF16_BF16, bf16), 16u);
    EXPECT_EQ(slots(ck::KernelVariant::kBF16_BF16_BF16, 0), 0u);
    EXPECT_EQ(slots(ck::KernelVariant::kF16_F16_F16, bf16), 0u);
    EXPECT_EQ(slots(ck::KernelVariant::kF16_F16_F16, all), 16u);
    // A call of several experts never takes it.
    const std::vector<int> M2 {512, 512}, N2 {768, 768}, K2 {768, 768};
    const std::vector<const void *> src2(2, nullptr), bias2(2, nullptr);
    const std::vector<void *> dst2(2, nullptr);
    const std::vector<int> ld2 {768, 768};
    const pt::CallView moe {M2, N2, K2, src2, ld2, bias2, dst2, ld2};
    moe_test_utils::SingleExpertBalanceMasks masks(all, -1);
    ck::CallContext kctx;
    kctx.variant = ck::KernelVariant::kBF16_BF16_BF16;
    kctx.pack_nr = 32;
    kctx.max_mr = 8;
    EXPECT_TRUE(pt::test_api::plan_only(kctx, moe, 15).owner_begin.empty());
}

struct DecodeRun {
    RunOut out;
    zendnnl::lowoha::matmul::test_api::PhaseBSnapshot plan;
};

/// One single-expert decode with `..._SINGLE_EXPERT_DECODE_BALANCE` at
/// `mask`, and the N-tile plan it ran.
DecodeRun run_decode(const Case &c, int mask) {
    using namespace moe_test_utils;
    SingleExpertBalanceMasks masks(-1, mask);
    PhaseBCaptureGuard capture;
    DecodeRun r;
    r.out = run_case(c, 0);
    r.plan = zendnnl::lowoha::matmul::test_api::s_last_phase_b_snapshot;
    return r;
}

/// Both runs planned on the N-tile path, ran without error (on the custom
/// kernel when `custom`) and wrote the destination, with the same bits.
void expect_same_decode(
        const DecodeRun &a, const DecodeRun &b, bool custom = true) {
    for (const DecodeRun *r : {&a, &b}) {
        ASSERT_TRUE(r->plan.valid) << "no N-tile plan: " << r->out.mode;
        ASSERT_EQ(r->out.mode.rfind("error_", 0), std::string::npos)
                << r->out.mode;
        if (custom) {
            ASSERT_NE(r->out.mode.find("_custom"), std::string::npos)
                    << r->out.mode;
        }
        ASSERT_TRUE(std::any_of(r->out.dst[0].begin(), r->out.dst[0].end(),
                [](uint8_t v) {
            return v != 0x5A;
        })) << "left unwritten: "
            << r->out.mode;
    }
    ASSERT_EQ(a.out.dst[0].size(), b.out.dst[0].size());
    EXPECT_EQ(std::memcmp(a.out.dst[0].data(), b.out.dst[0].data(),
                      a.out.dst[0].size()),
            0);
}

// Decode's per-thread N split changes which thread computes a column, never
// a column's sum: a single-expert decode is bitwise equal with the balanced
// split on and off.  It engages only where it leaves the busiest thread
// fewer pack blocks than the exact-divisor tile: N 2048 at 15 threads
// (8 x 256 columns: 8 blocks, balanced 5) and 31 (16 x 128: 4, balanced 3),
// N 1280 at 15 (10 x 128: 4, balanced 3); not N 1280 at 31 (20 x 64: 2,
// as over 31 threads).
TEST(PromptTileExecutor, SingleExpertDecodeBalanceKeepsBits) {
    SKIP_WITHOUT_BF16_CUSTOM_KERNEL();
    using namespace moe_test_utils;
    SingleExpertCkPolicy on(1, 1, 1);
    struct Shape {
        int T, N, K;
        bool engages;
    };
    for (const Shape &s : {Shape {15, 2048, 1024, true},
                 Shape {31, 2048, 1024, true}, Shape {15, 1280, 768, true},
                 Shape {31, 1280, 768, false}}) {
        Case c;
        c.M.assign(1, 8);
        c.N = s.N;
        c.K = s.K;
        c.wide = true;
        c.threads = s.T;
        SCOPED_TRACE(describe(c));
        const DecodeRun legacy = run_decode(c, 0);
        const DecodeRun balanced = run_decode(c,
                static_cast<int>(zendnnl::lowoha::matmul::kSingleExpertBf16));
        expect_same_decode(legacy, balanced);
        EXPECT_FALSE(legacy.plan.se_balanced);
        EXPECT_EQ(balanced.plan.se_balanced, s.engages);
        if (s.engages) { EXPECT_EQ(balanced.plan.min_n_tile, 32); }
    }
}

// An explicit tile, adaptive sizing off, `..._N_TILE_STRATEGY=0` (no
// single-expert knob applies) or `=1` (StaticExpertTeams) keeps the decode
// tile they select (N 2048 at 31 threads, which balances without them), and
// the balanced split keeps at least `..._N_TILE_FLOOR` columns a thread in
// whole pack blocks: a 128 floor at 31 threads leaves 16 threads as busy as
// the exact 128-column tile, and at 15 it balances 4-5 blocks a thread (at
// least 128 columns, not 8 blocks).
TEST(PromptTileExecutor, SingleExpertDecodeBalanceHonoursTileKnobs) {
    SKIP_WITHOUT_BF16_CUSTOM_KERNEL();
    using namespace moe_test_utils;
    SingleExpertCkPolicy on(1, 1, 1);
    const int all = static_cast<int>(zendnnl::lowoha::matmul::kSingleExpertAll);
    Case c;
    c.M.assign(1, 8);
    c.N = 2048;
    c.K = 1024;
    c.wide = true;
    c.threads = 31;
    SCOPED_TRACE(describe(c));
    const auto expect_kept = [&](const Case &p, const std::string &what) {
        SCOPED_TRACE(what);
        const DecodeRun base = run_decode(p, 0);
        const DecodeRun r = run_decode(p, all);
        expect_same_decode(base, r, /*custom=*/p.strategy == 2);
        EXPECT_FALSE(r.plan.se_balanced);
        EXPECT_EQ(r.plan.min_n_tile, base.plan.min_n_tile);
    };
    {
        CustomKernelNTileOverride tile(256);
        expect_kept(c, "CUSTOM_KERNEL_N_TILE=256");
    }
    {
        NTileOversubOverride oversub(0);
        expect_kept(c, "N_TILE_OVERSUB=0");
    }
    for (const int strategy : {0, 1}) {
        Case p = c;
        p.strategy = strategy;
        expect_kept(p, "N_TILE_STRATEGY=" + std::to_string(strategy));
    }
    NTileFloorOverride floor(128);
    expect_kept(c, "N_TILE_FLOOR=128");
    Case p = c;
    p.threads = 15;
    const DecodeRun base = run_decode(p, 0);
    const DecodeRun b = run_decode(p, all);
    expect_same_decode(base, b);
    EXPECT_TRUE(b.plan.se_balanced);
    EXPECT_EQ(b.plan.min_n_tile, 128);
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
            SingleExpertCkPolicy off(0, 0, 0);
            narrow = run_case(c, 0);
        }
        {
            SingleExpertCkPolicy on(1, 1, 1);
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

// A deep-K single-expert BF16 prompt accumulates in `..._PROMPT_KC` chunks.
// Each chunk after the first reloads the f32 partial sums and continues in K
// order, so the output matches the full-K tile bit for bit.
TEST(PromptTileExecutor, SingleBf16ExpertKChunksMatchFullK) {
    SKIP_WITHOUT_BF16_CUSTOM_KERNEL();
    using namespace moe_test_utils;
    SingleExpertCkPolicy on(1, 1, 1);
    // The second depth ends in a partial chunk.
    const int deep_K = prompt_kc_min_K(1);
    for (const auto &[M, N, K] : {std::tuple {300, 256, deep_K},
                 std::tuple {517, 640, deep_K + 1000}}) {
        for (const auto dst : {data_type_t::bf16, data_type_t::f32}) {
            Case c;
            c.M = {M};
            c.N = N;
            c.K = K;
            c.dst_dt = dst;
            c.wide = true;
            SCOPED_TRACE(describe(c));
            RunOut full, chunked;
            {
                SingleExpertPromptKcOverride kc(0);
                full = run_case(c, 1);
            }
            {
                SingleExpertPromptKcOverride kc(512);
                KblockCaptureGuard capture;
                chunked = run_case(c, 1);
                EXPECT_TRUE(KblockCaptureGuard::used());
            }
            ASSERT_NE(
                    chunked.mode.find("flat_n_tile_ptile_"), std::string::npos)
                    << chunked.mode;
            ASSERT_EQ(full.dst[0].size(), chunked.dst[0].size());
            EXPECT_EQ(std::memcmp(full.dst[0].data(), chunked.dst[0].data(),
                              full.dst[0].size()),
                    0);
        }
    }
}

// A single FP16 expert's prompt runs the two-o-block prompt microkernel,
// K-chunked or not.  It sums K in the decode microkernel's blocks and order,
// so it is bitwise equal to the rounds run of a mask without FP16, across
// odd o-block counts, MR tails, bias and every epilogue.
TEST(PromptTileExecutor, SingleF16ExpertPromptKernelMatchesDecodeKernel) {
    SKIP_WITHOUT_F16_CUSTOM_KERNEL();
    using namespace moe_test_utils;
    struct Shape {
        int M, N, K, kc;
    };
    const int deep_K = prompt_kc_min_K(
            zendnnl::lowoha::matmul::custom_kernel::kF16PromptNb);
    for (const auto &s : {Shape {300, 256, 128, 0}, Shape {517, 288, 192, 0},
                 Shape {301, 288, deep_K, 1000}}) {
        for (const auto act : {grp_matmul_gated_act_t::none,
                     grp_matmul_gated_act_t::silu_and_mul,
                     grp_matmul_gated_act_t::gelu_and_mul,
                     grp_matmul_gated_act_t::swiglu_oai_mul}) {
            const bool gated = act != grp_matmul_gated_act_t::none;
            for (const auto dst : {data_type_t::f16, data_type_t::f32}) {
                for (const auto bias : {data_type_t::none, data_type_t::f16}) {
                    const bool split_halves
                            = act == grp_matmul_gated_act_t::silu_and_mul
                            || act == grp_matmul_gated_act_t::gelu_and_mul;
                    if (gated && (dst == data_type_t::f32 || s.kc > 0))
                        continue;
                    // The CK refuses a split-halves activation with bias.
                    if (split_halves && bias != data_type_t::none) continue;
                    Case c;
                    c.M.push_back(s.M);
                    c.N = s.N;
                    c.K = s.K;
                    c.src_dt = data_type_t::f16;
                    c.dst_dt = dst;
                    c.act = act;
                    c.bias_dt = bias;
                    c.wide = true;
                    SCOPED_TRACE(describe(c));
                    RunOut ref, got;
                    {
                        SingleExpertCkPolicy no_f16(
                                zendnnl::lowoha::matmul::kSingleExpertInt8, 1,
                                1);
                        ref = run_case(c, /*FP16=*/8);
                    }
                    {
                        SingleExpertCkPolicy on(1, 1, 1);
                        SingleExpertPromptKcOverride kc(s.kc);
                        KblockCaptureGuard capture;
                        got = run_case(c, /*FP16=*/8);
                        EXPECT_EQ(
                                KblockCaptureGuard::used(), s.kc > 0 && !gated);
                    }
                    ASSERT_EQ(ref.mode.find("ptile"), std::string::npos)
                            << ref.mode;
                    ASSERT_NE(got.mode.find("flat_n_tile_ptile_"),
                            std::string::npos)
                            << got.mode;
                    EXPECT_TRUE(ref.dst[0] == got.dst[0]);
                }
            }
        }
    }
}

// A single-expert prompt the prompt tile executor does not take (pinned
// ALGO 3 with `ZENDNNL_GRP_MATMUL_PROMPT_TILE=0`) runs the multi-expert
// kernels: no K chunks, though the call's B overflows L2.
TEST(PromptTileExecutor, SingleExpertPromptOffTheTileRunsMultiExpertKernels) {
    SKIP_WITHOUT_BF16_CUSTOM_KERNEL();
    using namespace moe_test_utils;
    SingleExpertCkPolicy on(1, 1, 1);
    SingleExpertPromptKcOverride kc(512);
    Case c;
    c.M = {300};
    c.K = prompt_kc_min_K(1);
    c.wide = true;
    SCOPED_TRACE(describe(c));
    KblockCaptureGuard capture;
    const RunOut r = run_case(c, /*prompt tile off=*/0);
    EXPECT_EQ(r.mode.find("ptile"), std::string::npos) << r.mode;
    EXPECT_FALSE(KblockCaptureGuard::used());
}

// The prompt tile's `kc` knob sets the K chunk of a tile that can chunk,
// even where the call's B fits L2; a depth of K or more, or a gated
// epilogue, keeps the full K.  Chunked output matches the full K bit for
// bit.
TEST(PromptTileExecutor, KcKnobSetsTheChunkOfEligibleTiles) {
    SKIP_WITHOUT_BF16_CUSTOM_KERNEL();
    using namespace moe_test_utils;
    SingleExpertCkPolicy on(1, 1, 1);
    Case c;
    c.M = {300};
    c.K = 4096;
    c.wide = true;
    SCOPED_TRACE(describe(c));
    RunOut full;
    {
        KblockCaptureGuard capture;
        full = run_case(c, 1);
        EXPECT_FALSE(KblockCaptureGuard::used());
    }
    ASSERT_NE(full.mode.find("flat_n_tile_ptile_"), std::string::npos)
            << full.mode;
    {
        PtOverride knob(pt::kKc, 1024);
        KblockCaptureGuard capture;
        const RunOut chunked = run_case(c, 1);
        EXPECT_TRUE(KblockCaptureGuard::used());
        EXPECT_TRUE(full.dst[0] == chunked.dst[0]);
    }
    {
        PtOverride knob(pt::kKc, c.K);
        KblockCaptureGuard capture;
        run_case(c, 1);
        EXPECT_FALSE(KblockCaptureGuard::used());
    }
    c.act = grp_matmul_gated_act_t::silu_and_mul;
    {
        PtOverride knob(pt::kKc, 1024);
        KblockCaptureGuard capture;
        run_case(c, 1);
        EXPECT_FALSE(KblockCaptureGuard::used());
    }
}

// Under AUTO with the knob on, a single BF16 expert reaches the prompt tile
// whenever the CK takes its epilogue, a plain GEMM included.  A split-halves
// gated activation with bias, which the CK refuses, stays on ALGO 1 rather
// than N-tile rounds without the CK.
TEST(PromptTileExecutor, AutoSingleExpertFollowsCkEpilogue) {
    SKIP_WITHOUT_BF16_CUSTOM_KERNEL();
    using namespace moe_test_utils;
    SingleExpertCkPolicy on(1, 1, 1);
    {
        Case plain;
        plain.M.assign(1, 300);
        SCOPED_TRACE(describe(plain));
        const RunOut r = run_case(plain, 1, /*algo=*/0);
        EXPECT_NE(r.mode.find("ptile"), std::string::npos) << r.mode;
        EXPECT_EQ(r.served, 1);
    }
    for (const auto act : {grp_matmul_gated_act_t::gelu_and_mul,
                 grp_matmul_gated_act_t::silu_and_mul}) {
        Case c;
        c.M.assign(1, 300);
        c.act = act;
        SCOPED_TRACE(describe(c));
        const RunOut served = run_case(c, 1, /*algo=*/0);
        EXPECT_NE(served.mode.find("ptile"), std::string::npos) << served.mode;
        EXPECT_EQ(served.served, 1);

        c.bias_dt = data_type_t::bf16;
        SCOPED_TRACE(describe(c));
        const RunOut refused = run_case(c, 1, /*algo=*/0);
        EXPECT_NE(executed_algo_from_execution_path(refused.mode.c_str()), 3)
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
            SingleExpertCkPolicy ovr(knob, knob, knob);
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
            ExecutionPathCaptureGuard cap;
            ASSERT_EQ(group_matmul_direct(gv.layout, gv.transA, gv.transB,
                              gv.Ms, gv.Ns, gv.Ks, gv.alpha, srcs, gv.lda, wei1,
                              gv.ldb, no_bias, gv.beta, dst_null, ldc_null,
                              gv.is_wc, p, nullptr, &act, &fused),
                    status_t::success);
            const char *m
                    = test_api::s_last_group_matmul_direct_execution_path.load(
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
// expert with the knob off.  An FP16 single expert follows the same knobs.
TEST(PromptTileExecutor, SingleActiveExpertOutsideScopeNeverEngages) {
    SKIP_WITHOUT_BF16_CUSTOM_KERNEL();
    using namespace moe_test_utils;
    for (const auto &[knob, M] : {std::pair {1, std::vector<int> {0, 300, 0}},
                 std::pair {0, std::vector<int> {300}},
                 std::pair {0, std::vector<int> {0, 300, 0}}}) {
        SingleExpertCkPolicy ovr(knob, knob, 0);
        Case c;
        c.M = M;
        SCOPED_TRACE(describe(c) + " knob=" + std::to_string(knob));
        const RunOut r = run_case(c, 1);
        EXPECT_EQ(executed_algo_from_execution_path(r.mode.c_str()), 3)
                << r.mode;
        EXPECT_EQ(r.mode.find("ptile"), std::string::npos) << r.mode;
        EXPECT_EQ(r.served, 0);
    }
    if (!zendnnl::lowoha::matmul::custom_kernel::avx512f16_available()) return;
    Case c;
    c.M = {300};
    c.src_dt = data_type_t::f16;
    c.dst_dt = data_type_t::f16;
    SCOPED_TRACE(describe(c));
    for (const int knob : {0, 1}) {
        SingleExpertCkPolicy ovr(knob, knob, knob);
        const RunOut r = run_case(c, 8);
        EXPECT_EQ(r.mode.find("ptile") != std::string::npos, knob == 1)
                << r.mode;
        EXPECT_EQ(r.served, knob);
    }
}

TEST(PromptTileExecutor, KnobOffOrNonCustomNeverEngage) {
    Case c;
    c.M = ab_dists()[0];
    // A mask without the BF16 bit keeps BF16 calls on the rounds.
    for (const int mask : {0, 2, 4, 6}) {
        const RunOut off = run_case(c, mask);
        EXPECT_EQ(off.mode.find("ptile"), std::string::npos)
                << "enable=" << mask << " " << off.mode;
        EXPECT_EQ(executed_algo_from_execution_path(off.mode.c_str()), 3)
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
    group_matmul_execute(gv.layout, gv.transA, gv.transB, gv.Ms, gv.Ns, gv.Ks,
            gv.alpha, srcs, gv.lda, weis, gv.ldb, biases, gv.beta, dsts, gv.ldc,
            gv.is_wc, params, 8, &mode, grp_matmul_gated_act_t::none,
            data_type_t::bf16);
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
    mm::GroupMatmulAlgoTrace trace;
    const int algo = mm::group_matmul_algo_select(
            p.layout, p.M, p.N, p.K, p.params, threads, &trace);
    return {algo, trace.reason != nullptr ? trace.reason : "<none>"};
}

TEST(PromptTileAuto, AutoSendsBf16PromptsToAlgo3) {
    using namespace moe_test_utils;
    AlgoEnvGuard algo0(0);
    CustomKernelOverride ck_on(true);
    PtOverride enable(pt::kEnable, 1);
    SingleExpertCkPolicy single_expert_off(0, 0, 0);
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
    SingleExpertCkPolicy single_expert_off(0, 0, 0);
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
        EXPECT_FALSE(mm::grp_matmul_auto_algo3_call(
                mixed.M, mixed.N, mixed.K, mixed.params));
        EXPECT_EQ(auto_pick(mixed), kShapeRulePrompt);
    }
    {
        Probe mismatch = auto_probe(
                300, 1024, 2048, 8, data_type_t::f16, data_type_t::f16);
        for (auto &p : mismatch.params)
            p.dtypes.wei = data_type_t::bf16;
        EXPECT_FALSE(mm::grp_matmul_auto_algo3_call(
                mismatch.M, mismatch.N, mismatch.K, mismatch.params));
        EXPECT_EQ(auto_pick(mismatch), kShapeRulePrompt);
    }
    {
        Probe packed = auto_probe(
                300, 1024, 2048, 8, data_type_t::f16, data_type_t::f16);
        for (auto &p : packed.params) {
            p.mem_format_b = 'r';
            p.lowoha_algo = zendnnl::common::matmul_algo_t::aocl_dlp_blocked;
        }
        EXPECT_FALSE(mm::grp_matmul_auto_algo3_call(
                packed.M, packed.N, packed.K, packed.params));
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
    EXPECT_EQ(executed_algo_from_execution_path(off.mode.c_str()), 3)
            << off.mode;
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
            EXPECT_EQ(d.stats.group_schedule, 3);
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
    EXPECT_EQ(in_place.stats.group_schedule, 3);
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
