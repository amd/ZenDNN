/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

// The cost model of the native BRGEMM plan (native_cost_model.hpp) and the
// decisions it takes over in the built-in plan, on two cache hierarchies of
// 48 KiB L1d and 1 MiB L2: L3 domains of 8 cores and 32 MiB, and of 32
// cores and 128 MiB (a 32-thread team in one domain, a 64-thread team over
// two).  A BF16 tile of 32 columns takes 8 rows per call where 12 rows leave
// a short last call (on the 8-thread domains while the workers' weight
// streams are short), and above decode_m where taller calls cost more (the
// 32-thread domains); two row groups of half the team take a single row
// block where the team's threads share one L3 domain, never where a column
// group's threads span domains, and only where they balance a slice of
// whole tiles, split the rows evenly, and a column tile's slab over K fits
// L2 and the call's weights fit L3.  Where one core streams slower than its
// share of its domain's link (the 32-thread domains; not the 16-thread
// domains of 32 MiB, where the two rates tie), a long weight stream decides
// how a single row block takes K: a block whose worker computes longer than
// it streams takes the long K pass, one that streams longer than it
// computes M blocks of 6 rows.  A single per-channel INT8 row
// block takes two row groups under the same bounds, and with cm_int8_l1_pct
// 6 rows per call and a shorter L1 pass for three calls.  FP32, per-group
// INT8 and tiles of fewer than 32 columns keep the rules' plan; pins and the
// switch keep their meaning; every plan stays legal.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/int8/native_int4_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_config.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_cost_model.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_planner.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_tuning_file.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "lowoha_operators/matmul/matmul_native/native_layout_policy.hpp"
#include "matmul_native/native_brgemm_test_utils.hpp"

namespace {

namespace nat = zendnnl::lowoha::matmul::native;
using zendnnl::memory::data_type_t;

/// 48 KiB of L1d, 1 MiB of L2 and l3_mib MiB of L3 per domain of domain
/// cores.
nat::UarchParams caches(int l3_mib, int domain) {
    nat::UarchParams u = nat::detect_uarch();
    u.l1d_bytes = 48 << 10;
    u.l2_bytes = 1 << 20;
    u.l3_domain_bytes = l3_mib << 20;
    u.family_l3_cores = 1;
    u.l3_domain_cores = domain;
    u.fma_ports = 2;
    return u;
}

const nat::UarchParams &domains8() {
    static const nat::UarchParams u = caches(32, 8);
    return u;
}

const nat::UarchParams &domains32() {
    static const nat::UarchParams u = caches(128, 32);
    return u;
}

const nat::UarchParams &domains16() {
    static const nat::UarchParams u = caches(32, 16);
    return u;
}

/// The model takes a variant only as a legal plan, and a plan is legal only
/// with this CPU's kernels for it: the tests of BF16 choices need
/// AVX512-BF16 and those of INT8 and s4 choices AVX512-VNNI.  The prices
/// themselves need neither.
bool bf16_kernels() {
    return nat::detect_uarch().avx512bf16;
}

bool int8_kernels() {
    return nat::detect_uarch().avx512vnni;
}

/// A dense call of the framework's linear layers: a bf16 source, bf16
/// destination, no bias; INT8: per-channel weights, a bf16 source quantized
/// in the call.
nat::native_problem_t problem(int M, int N, int K, int threads, bool bf16) {
    nat::native_problem_t p;
    p.M = M;
    p.N = N;
    p.K = K;
    p.pack_nr = nat::kNativeLayoutPackNr;
    p.threads = threads;
    p.src_dt = data_type_t::bf16;
    p.wei_dt = bf16 ? data_type_t::bf16 : data_type_t::s8;
    p.dst_dt = data_type_t::bf16;
    p.dynamic_quant = !bf16;
    return p;
}

/// The problem of an s4 weight in groups of gs with scales of scale_bytes
/// over a bf16 source quantized in the call.
nat::native_problem_t int4_problem(
        int M, int N, int K, int threads, int gs, int scale_bytes = 2) {
    nat::native_problem_t p = problem(M, N, K, threads, false);
    p.wei_dt = data_type_t::s4;
    p.int8_scheme = nat::native_int8_scheme_t::per_group;
    p.wei_group_size = gs;
    p.wei_scale_bytes = scale_bytes;
    return p;
}

nat::heuristic_params_t params(const std::string &text) {
    nat::native_config_t cfg;
    std::string why;
    EXPECT_TRUE(nat::parse_native_config(text, cfg, why))
            << text << ": " << why;
    return nat::apply_native_config(nat::heuristic_params_t {}, cfg);
}

/// The built-in plan, and the rules' plan without the cost model.
nat::native_plan_t plan(
        const nat::UarchParams &u, int M, int N, int K, int threads) {
    return nat::heuristic_plan(
            problem(M, N, K, threads, true), u, nat::heuristic_params_t {});
}

nat::native_plan_t rules(
        const nat::UarchParams &u, int M, int N, int K, int threads) {
    return nat::heuristic_plan(
            problem(M, N, K, threads, true), u, params("cost_model=0"));
}

/// The same for a per-channel INT8 call.
nat::native_plan_t plan8(
        const nat::UarchParams &u, int M, int N, int K, int threads) {
    return nat::heuristic_plan(
            problem(M, N, K, threads, false), u, nat::heuristic_params_t {});
}

nat::native_plan_t rules8(
        const nat::UarchParams &u, int M, int N, int K, int threads) {
    return nat::heuristic_plan(
            problem(M, N, K, threads, false), u, params("cost_model=0"));
}

bool same(const nat::native_plan_t &a, const nat::native_plan_t &b) {
    return a.nr == b.nr && a.mr == b.mr && a.mb == b.mb && a.kc == b.kc
            && a.threads == b.threads && a.col_groups == b.col_groups
            && a.unit_tiles == b.unit_tiles && a.row_unit == b.row_unit
            && a.traversal == b.traversal
            && a.scratch_floats == b.scratch_floats
            && a.int8_rows_nv1 == b.int8_rows_nv1
            && a.int8_rows_nv2 == b.int8_rows_nv2
            && a.int8_pass_quads == b.int8_pass_quads;
}

std::string text(const nat::native_plan_t &q) {
    return "nr=" + std::to_string(q.nr) + " mr=" + std::to_string(q.mr)
            + " mb=" + std::to_string(q.mb) + " kc=" + std::to_string(q.kc)
            + " threads=" + std::to_string(q.threads)
            + " col_groups=" + std::to_string(q.col_groups)
            + " int8_rows=" + std::to_string(q.int8_rows_nv2)
            + " int8_pass=" + std::to_string(q.int8_pass_quads);
}

const nat::UarchParams *const kBoth[] = {&domains8(), &domains32()};

// A tile of 16 rows takes two calls of 8 rows and the L1 pass of two full
// calls instead of a call of 12 rows and one of 4, on both hierarchies and on
// a team of two domains; a block of 12 rows keeps its one call of 12, and 32
// rows (12, 12 and 8) keep 12 rows per call.
TEST(NativeCostModel, RowsPerCallWithTheRemainder) {
    if (!bf16_kernels()) GTEST_SKIP() << "needs AVX512-BF16";
    for (const nat::UarchParams *u : kBoth)
        for (const int threads : {32, 64}) {
            SCOPED_TRACE("domain " + std::to_string(u->l3_domain_cores)
                    + " threads " + std::to_string(threads));
            const nat::native_plan_t before
                    = rules(*u, 16, 4096, 4096, threads);
            EXPECT_EQ(before.mr, 12);
            EXPECT_EQ(before.kc, 4096);
            const nat::native_plan_t q = plan(*u, 16, 4096, 4096, threads);
            EXPECT_EQ(q.mr, 8) << text(q);
            EXPECT_EQ(q.mb, 16);
            EXPECT_LT(q.kc, 4096) << text(q);
            EXPECT_EQ(q.kc % 2, 0);
            EXPECT_EQ(plan(*u, 12, 4096, 4096, threads).mr, 12);
            if (u == &domains8() || threads == 64) {
                EXPECT_EQ(plan(*u, 32, 4096, 4096, threads).mr, 12);
            }
        }
}

// The price of a call of more than 8 rows defaults to 12.5 percent per
// doubling of the threads of an L3 domain past 8: none on the 8-thread
// domains, 25 percent on the 32-thread ones.  Above decode_m a block of 192
// rows then takes calls of 8 rows on the 32-thread domain, whether or not K
// makes the source rows a multiple of 4 KiB apart, and keeps the rules' plan
// on the 8-thread domains.
TEST(NativeCostModel, ShortRowsAboveDecodeM) {
    const nat::heuristic_params_t hp;
    EXPECT_DOUBLE_EQ(nat::native_plan_tall_pct(domains8(), hp), 0.0);
    EXPECT_DOUBLE_EQ(nat::native_plan_tall_pct(domains32(), hp), 25.0);
    EXPECT_DOUBLE_EQ(nat::native_plan_tall_pct(caches(512, 256), hp), 50.0);
    EXPECT_DOUBLE_EQ(
            nat::native_plan_tall_pct(domains8(), params("cm_tall_pct=3")),
            3.0);
    if (!bf16_kernels()) GTEST_SKIP() << "needs AVX512-BF16";
    for (const int K : {1536, 2560, 4096, 5120}) {
        SCOPED_TRACE("K " + std::to_string(K));
        const nat::native_plan_t q = plan(domains32(), 1024, 4096, K, 32);
        EXPECT_EQ(q.mr, 8) << text(q);
        EXPECT_EQ(q.mb, rules(domains32(), 1024, 4096, K, 32).mb);
        EXPECT_TRUE(same(plan(domains8(), 1024, 4096, K, 32),
                rules(domains8(), 1024, 4096, K, 32)));
        // The 8-thread domains with the wide domain's price take 8 rows, and
        // the wide domain without a price keeps the rules' plan.
        EXPECT_EQ(nat::heuristic_plan(problem(1024, 4096, K, 32, true),
                          domains8(), params("cm_tall_pct=25"))
                          .mr,
                8);
        EXPECT_TRUE(same(nat::heuristic_plan(problem(1024, 4096, K, 32, true),
                                 domains32(), params("cm_tall_pct=0")),
                rules(domains32(), 1024, 4096, K, 32)));
    }
}

// Two row groups of 16 threads take M = 32 and 55 of a slice of 2.5 column
// tiles a thread (N 2560) where the 32 threads share one L3 domain; over
// 8-thread domains a column group's two threads sit in different domains,
// and on a 64-thread team over two 32-thread domains in both: each would
// stream every weight tile from memory once per domain, and the plan keeps
// the slice.
TEST(NativeCostModel, RowGroupsInsideOneL3Domain) {
    if (!bf16_kernels()) GTEST_SKIP() << "needs AVX512-BF16";
    for (const int M : {32, 55}) {
        SCOPED_TRACE("M " + std::to_string(M));
        const nat::native_plan_t wide = plan(domains32(), M, 2560, 4096, 32);
        EXPECT_EQ(wide.col_groups, 16) << text(wide);
        EXPECT_EQ(plan(domains8(), M, 2560, 4096, 32).col_groups, 0);
        EXPECT_EQ(plan(domains32(), M, 2560, 4096, 64).col_groups, 0);
        // A pinned domain of 8 threads on the wide hierarchy is the narrow
        // one.
        EXPECT_EQ(nat::heuristic_plan(problem(M, 2560, 4096, 32, true),
                          domains32(), params("domain_threads=8"))
                          .col_groups,
                0);
    }
    // Row groups of 8-row calls split the rows by those calls: 16 + 16 at
    // M = 32, 24 + 31 at M = 55.  A larger group of more than
    // cm_group_rows_pct (115) percent of half the rows (12 + 20 rows of
    // 12-row calls at M = 32) is not weighed.
    const nat::native_plan_t m32 = plan(domains32(), 32, 2560, 4096, 32);
    EXPECT_EQ(m32.mr, 8) << text(m32);
    EXPECT_EQ(m32.row_unit, 8);
    EXPECT_EQ(plan(domains32(), 55, 2560, 4096, 32).mr, 8);
    EXPECT_EQ(nat::heuristic_plan(problem(32, 2560, 4096, 32, true),
                      domains32(), params("mr=12"))
                      .col_groups,
            0);
    // Each group takes whole row units of 12 rows and more than
    // cm_short_rows rows: M = 16 (12 + 4) and 20 (12 + 8) keep the slice,
    // M = 21 (12 + 9) takes the groups, and so does M = 20 with a lower
    // cm_short_rows and a larger group of up to twice half the rows.
    EXPECT_EQ(plan(domains32(), 16, 2560, 4096, 32).col_groups, 0);
    EXPECT_EQ(plan(domains32(), 20, 2560, 4096, 32).col_groups, 0);
    EXPECT_EQ(plan(domains32(), 21, 2560, 4096, 32).col_groups, 16);
    EXPECT_EQ(
            nat::heuristic_plan(problem(20, 2560, 4096, 32, true), domains32(),
                    params("cm_short_rows=3,cm_group_rows_pct=200"))
                    .col_groups,
            16);
}

// A row unit other than the built-in one pins the units of the row groups,
// also of groups of cm_short_rows rows per call: at M 55, N 2560, K 4096 on
// the 32-thread domain row_unit=1 takes 27 + 28 rows of 8-row calls and
// row_unit=4 28 + 27, where the built-in unit gives such groups units of 8
// rows (24 + 31).
TEST(NativeCostModel, RowGroupsKeepAPinnedRowUnit) {
    if (!bf16_kernels()) GTEST_SKIP() << "needs AVX512-BF16";
    const nat::native_problem_t p = problem(55, 2560, 4096, 32, true);
    const nat::native_plan_t built_in = plan(domains32(), 55, 2560, 4096, 32);
    ASSERT_EQ(built_in.col_groups, 16) << text(built_in);
    EXPECT_EQ(built_in.mr, 8);
    EXPECT_EQ(built_in.row_unit, 8);
    for (const int unit : {1, 4}) {
        const nat::native_plan_t q = nat::heuristic_plan(
                p, domains32(), params("row_unit=" + std::to_string(unit)));
        EXPECT_EQ(q.col_groups, 16) << "row_unit " << unit << " " << text(q);
        EXPECT_EQ(q.mr, 8) << "row_unit " << unit;
        EXPECT_EQ(q.row_unit, unit);
        EXPECT_EQ(nat::validate_native_plan(q, p), nullptr);
    }
}

// Two row groups balance a slice whose busiest worker takes at least
// cm_group_skew_pct (110) percent of the mean column tiles a thread: at
// M 32 on 32 threads N 1536 (1.5 tiles a thread), 2560 (2.5) and 4608 (4.5)
// take them, N 2048, 4096 and 5120 (whole tiles a thread) keep the slice,
// which a skew bound of 100 percent gives up.
TEST(NativeCostModel, RowGroupsBalanceTheSlice) {
    const nat::heuristic_params_t hp;
    EXPECT_TRUE(nat::native_plan_slice_skewed(
            rules(domains32(), 32, 2560, 4096, 32), hp));
    EXPECT_FALSE(nat::native_plan_slice_skewed(
            rules(domains32(), 32, 4096, 4096, 32), hp));
    if (!bf16_kernels()) GTEST_SKIP() << "needs AVX512-BF16";
    for (const int N : {1536, 2560, 4608})
        EXPECT_EQ(plan(domains32(), 32, N, 4096, 32).col_groups, 16)
                << "N " << N;
    for (const int N : {2048, 4096, 5120}) {
        EXPECT_EQ(plan(domains32(), 32, N, 4096, 32).col_groups, 0)
                << "N " << N;
        EXPECT_EQ(nat::heuristic_plan(problem(32, N, 4096, 32, true),
                          domains32(), params("cm_group_skew_pct=100"))
                          .col_groups,
                16)
                << "N " << N;
    }
}

// The weight bytes of row groups by L3 domain: over 8-thread domains the two
// row groups stream each tile from memory twice, so their memory time is
// twice the slice's; inside one domain the second read hits L3 and the
// memory time stays below the slice's while the domain's weights fit its
// reuse window.
TEST(NativeCostModel, WeightReadsPerL3Domain) {
    const nat::heuristic_params_t hp;
    for (const nat::UarchParams *u : kBoth) {
        const nat::native_problem_t p = problem(32, 4096, 4096, 32, true);
        nat::native_plan_t slice = rules(*u, 32, 4096, 4096, 32);
        ASSERT_EQ(slice.col_groups, 0);
        nat::native_plan_t groups = nat::heuristic_plan(
                p, *u, params("cost_model=0,col_groups=16"));
        ASSERT_EQ(groups.col_groups, 16);
        const nat::native_plan_cost_t s
                = nat::native_plan_cost(slice, p, *u, hp);
        const nat::native_plan_cost_t g
                = nat::native_plan_cost(groups, p, *u, hp);
        ASSERT_GT(s.memory, 0);
        // Each worker has 4 column tiles in the slice and 8 in a row group;
        // the slice's later calls of a pass leave its stream idle longer.
        if (u == &domains8()) {
            EXPECT_GT(g.memory, 1.5 * s.memory)
                    << "slice " << s.memory << " groups " << g.memory;
        } else {
            EXPECT_LT(g.memory, s.memory)
                    << "slice " << s.memory << " groups " << g.memory;
            EXPECT_LT(g.cycles, s.cycles);
        }
        // The modeled time is the larger part plus cm_overlap_pct of the
        // smaller.
        for (const nat::native_plan_cost_t &c : {s, g})
            EXPECT_DOUBLE_EQ(c.cycles,
                    std::max(c.core, c.memory)
                            + hp.cm_overlap_pct / 100.0
                                    * std::min(c.core, c.memory));
    }
    // Weights beyond the reuse window: the second reads stream from memory.
    const nat::native_problem_t lm = problem(32, 262144, 4096, 32, true);
    const nat::native_plan_t groups = nat::heuristic_plan(
            lm, domains32(), params("cost_model=0,col_groups=16"));
    const nat::native_plan_t slice = rules(domains32(), 32, 262144, 4096, 32);
    EXPECT_GT(nat::native_plan_cost(groups, lm, domains32(), {}).memory,
            1.5 * nat::native_plan_cost(slice, lm, domains32(), {}).memory);
    if (!bf16_kernels()) GTEST_SKIP() << "needs AVX512-BF16";
    EXPECT_EQ(plan(domains32(), 32, 262144, 4096, 32).col_groups, 0);
}

// A domain reads a weight tile a second time, from L3, only where more
// than one of its row groups with rows reads it.  At M 32, N 4096, K 4096
// each worker streams its domain's share of the weights: 1024 KiB on the
// 32-thread domains, both for 32 column groups on 64 threads (one row group
// in each domain) and for 16 (two in each), and 2048 KiB for 16 column
// groups over the 8-thread domains (8 threads of one row group in each).
// Only the domains of two row groups add the second reads of the worker's
// tiles at cm_l3_cycles per KiB, as a 32-thread team in one domain does.
TEST(NativeCostModel, SecondReadsOnlyWhereADomainRereads) {
    const nat::heuristic_params_t hp;
    const auto cost = [&](const nat::UarchParams &u, int threads, int g) {
        const nat::native_problem_t p = problem(32, 4096, 4096, threads, true);
        const nat::native_plan_t q = nat::heuristic_plan(
                p, u, params("cost_model=0,col_groups=" + std::to_string(g)));
        EXPECT_EQ(q.col_groups, g);
        return nat::native_plan_cost(q, p, u, hp);
    };
    const double mem32 = nat::native_plan_mem_cycles(domains32(), hp);
    const double mem8 = nat::native_plan_mem_cycles(domains8(), hp);
    // 8 column tiles of 32 x 4096 bf16 weights a worker under 16 groups.
    const double rereads = 8 * 256.0 * hp.cm_l3_cycles;
    const nat::native_plan_cost_t own = cost(domains32(), 64, 32);
    EXPECT_DOUBLE_EQ(own.stream_kib, 1024.0);
    EXPECT_DOUBLE_EQ(own.stream, 1024.0 * mem32);
    const nat::native_plan_cost_t two = cost(domains32(), 64, 16);
    EXPECT_DOUBLE_EQ(two.stream_kib, 1024.0);
    EXPECT_DOUBLE_EQ(two.stream, 1024.0 * mem32 + rereads);
    EXPECT_DOUBLE_EQ(
            cost(domains32(), 32, 16).stream, 1024.0 * mem32 + rereads);
    const nat::native_plan_cost_t spans = cost(domains8(), 32, 16);
    EXPECT_DOUBLE_EQ(spans.stream_kib, 2048.0);
    EXPECT_DOUBLE_EQ(spans.stream, 2048.0 * mem8);
}

// A thread's memory rate defaults to 2400 cycles per KiB over the square
// root of the threads of an L3 domain, and cm_mem_cycles replaces it; the
// slower stream of the 8-thread domains covers more of a tile's later calls,
// so a decode tile of 16 rows on a long weight stream keeps 12 + 4 rows
// there and takes 8 + 8 on the 32-thread domain.
TEST(NativeCostModel, MemoryRateFollowsTheL3Domain) {
    const nat::heuristic_params_t hp;
    EXPECT_DOUBLE_EQ(nat::native_plan_mem_cycles(domains8(), hp),
            2400.0 / std::sqrt(8.0));
    EXPECT_DOUBLE_EQ(nat::native_plan_mem_cycles(domains32(), hp),
            2400.0 / std::sqrt(32.0));
    EXPECT_DOUBLE_EQ(nat::native_plan_mem_cycles(
                             domains32(), params("domain_threads=8")),
            2400.0 / std::sqrt(8.0));
    EXPECT_DOUBLE_EQ(nat::native_plan_mem_cycles(
                             domains8(), params("cm_mem_cycles=300")),
            300.0);
    if (!bf16_kernels()) GTEST_SKIP() << "needs AVX512-BF16";
    EXPECT_EQ(plan(domains8(), 16, 151936, 2048, 32).mr, 12);
    EXPECT_EQ(plan(domains32(), 16, 151936, 2048, 32).mr, 8);
    // The same machine with the wide domain's rate pinned takes 8 rows.
    EXPECT_EQ(nat::heuristic_plan(problem(16, 151936, 2048, 32, true),
                      domains8(), params("cm_mem_cycles=424"))
                      .mr,
            8);
}

// Where a core streams faster than its share of the domain's link
// (cm_stream_cycles below cm_mem_cycles: the 8-thread domains), the other
// streams of the domain fill the link while one idles, past the first
// cm_sync_tiles tiles of each worker.  A decode tile of 16 rows takes 8 + 8
// rows where the workers have few column tiles (2.5 at N = 2560) and keeps
// 12 + 4 on a long weight stream (242.5 at N = 248320); on the 32-thread
// domain every tile's idle counts and both take 8 + 8.
TEST(NativeCostModel, LinkFillsTheIdleOfLongStreams) {
    if (!bf16_kernels()) GTEST_SKIP() << "needs AVX512-BF16";
    for (const int threads : {32, 64}) {
        SCOPED_TRACE("threads " + std::to_string(threads));
        EXPECT_EQ(plan(domains8(), 16, 2560, 9216, threads).mr, 8);
        EXPECT_TRUE(same(plan(domains8(), 16, 248320, 4096, threads),
                rules(domains8(), 16, 248320, 4096, threads)));
        EXPECT_EQ(plan(domains32(), 16, 248320, 4096, threads).mr, 8);
    }
    // Every tile's idle counts where a core streams no faster than its
    // share, or where the workers stay in step.
    for (const char *input :
            {"cm_stream_cycles=1048576", "cm_sync_tiles=1048576"})
        EXPECT_EQ(nat::heuristic_plan(problem(16, 248320, 4096, 32, true),
                          domains8(), params(input))
                          .mr,
                8)
                << input;
}

// The FMA latency prices a call of few rows: a call of 4 rows of 32 columns
// issues 4 cycles of multiply-adds per K pair and waits cm_fma_lat (6) for
// its accumulators, so with the price of tall calls, the call cost and the
// L2 rate of the later calls left out, a decode tile of 4 rows costs
// cm_fma_lat cycles per K pair and 12 + 4 rows cost more core cycles than
// 8 + 8 over the same K and passes; without the latency the two cost the
// same.  The built-in plans of N 1536, K 4096 follow the latency: M 17 on
// the 8-thread domains keeps 12 + 5 rows (8 + 8 + 1 would wait on its 1-row
// call), and M 38 on the 32-thread domain takes 8-row calls (12 + 12 + 12 +
// 2 would wait on the 2-row call).
TEST(NativeCostModel, ShortCallsWaitForTheAccumulators) {
    const char *const only = "cm_tall_pct=0,cm_call_cycles=0,cm_l2_bytes=4096";
    const nat::native_problem_t decode = problem(4, 4096, 4096, 32, true);
    const nat::native_plan_t d = rules(domains8(), 4, 4096, 4096, 32);
    ASSERT_EQ(d.n_tiles, 128);
    ASSERT_EQ(d.threads, 32);
    // Each worker takes 4 column tiles of one call over K / 2 pairs.
    for (const int lat : {1, 6, 8}) {
        const nat::heuristic_params_t hp = params(
                std::string(only) + ",cm_fma_lat=" + std::to_string(lat));
        EXPECT_DOUBLE_EQ(nat::native_plan_cost(d, decode, domains8(), hp).core,
                4 * 2048.0 * std::max(lat, 4))
                << "cm_fma_lat " << lat;
    }
    const nat::native_problem_t p = problem(16, 4096, 4096, 32, true);
    nat::native_plan_t q = rules(domains8(), 16, 4096, 4096, 32);
    nat::native_plan_t r = q;
    r.mr = 8;
    q.kc = r.kc = 4096;
    q.scratch_floats = nat::native_plan_scratch_floats(q, p);
    r.scratch_floats = nat::native_plan_scratch_floats(r, p);
    const nat::heuristic_params_t lat = params(only);
    EXPECT_DOUBLE_EQ(nat::native_plan_cost(q, p, domains8(), lat).core,
            4 * 2048.0 * (12 + 6));
    EXPECT_DOUBLE_EQ(nat::native_plan_cost(r, p, domains8(), lat).core,
            4 * 2048.0 * (8 + 8));
    const nat::heuristic_params_t hp;
    EXPECT_GT(nat::native_plan_cost(q, p, domains8(), hp).core,
            nat::native_plan_cost(r, p, domains8(), hp).core);
    const nat::heuristic_params_t flat
            = params(std::string(only) + ",cm_fma_lat=1");
    EXPECT_DOUBLE_EQ(nat::native_plan_cost(q, p, domains8(), flat).core,
            nat::native_plan_cost(r, p, domains8(), flat).core);
    if (!bf16_kernels()) GTEST_SKIP() << "needs AVX512-BF16";
    const nat::native_plan_t m17 = plan(domains8(), 17, 1536, 4096, 32);
    EXPECT_EQ(m17.mr, 12) << text(m17);
    const nat::native_plan_t m38 = plan(domains32(), 38, 1536, 4096, 32);
    EXPECT_EQ(m38.mr, 8) << text(m38);
}

// A single row block of four calls or more of the rules' 12 rows whose
// busiest worker computes longer than it streams, over more than
// cm_drift_kib KiB of weights, takes the K pass of cm_pass_kib KiB of slab
// (2048 at 32 columns) and 12 rows per call, keeping the model's schedule,
// where one core streams slower than its share of the domain's link: at
// M 55 the slices of N 65536, K 5120 and of N 24576, K 4096.
// A shorter stream (N 4096, K 4096: 1 MiB a worker) keeps the L1 pass, and
// so do the 8-thread domains and cm_pass_kib=0.
TEST(NativeCostModel, LongPassForLongComputeBoundStreams) {
    EXPECT_TRUE(nat::native_plan_core_streams(domains32(), {}));
    EXPECT_FALSE(nat::native_plan_core_streams(domains8(), {}));
    if (!bf16_kernels()) GTEST_SKIP() << "needs AVX512-BF16";
    const nat::native_plan_t q = plan(domains32(), 55, 65536, 5120, 32);
    EXPECT_EQ(q.kc, 2048) << text(q);
    EXPECT_EQ(q.mr, 12);
    EXPECT_EQ(q.col_groups, 0);
    const nat::native_plan_t g = plan(domains32(), 55, 24576, 4096, 32);
    EXPECT_EQ(g.kc, 2048) << text(g);
    EXPECT_EQ(g.mr, 12);
    EXPECT_EQ(g.col_groups, 0);
    EXPECT_LT(plan(domains32(), 55, 4096, 4096, 32).kc, 1024);
    const nat::native_problem_t p = problem(55, 65536, 5120, 32, true);
    const nat::native_plan_t off
            = nat::heuristic_plan(p, domains32(), params("cm_pass_kib=0"));
    EXPECT_LT(off.kc, 1024) << text(off);
    // The rule's inputs: the busiest worker's core and weight-stream cycles
    // and the KiB of its share of the domain's weights.
    const nat::native_plan_cost_t c
            = nat::native_plan_cost(off, p, domains32(), {});
    EXPECT_GT(c.core, c.stream);
    EXPECT_GT(c.stream_kib, nat::heuristic_params_t {}.cm_drift_kib);
    EXPECT_GE(c.memory, c.stream);
    for (const int threads : {32, 64})
        EXPECT_TRUE(same(plan(domains8(), 55, 65536, 5120, threads),
                nat::heuristic_plan(problem(55, 65536, 5120, threads, true),
                        domains8(), params("cm_pass_kib=0"))));
    EXPECT_EQ(
            nat::heuristic_plan(p, domains32(), params("cm_drift_kib=1048576"))
                    .kc,
            off.kc);
}

// The long K pass takes the rules' rows per call and row unit under the
// model's row groups only where that unit splits the rows within
// cm_group_rows_pct.  The 32-thread domain takes two row groups of 8-row
// calls at M 37 and 55 of N 4608, K 8192; with cm_drift_kib 2048 the pass of
// 2048 takes M 55 in 12-row calls of 12-row units (24 + 31 rows), while at
// M 37 those units would split 24 + 13 rows and the groups keep their plan.
TEST(NativeCostModel, LongPassKeepsTheRowGroupBounds) {
    if (!bf16_kernels()) GTEST_SKIP() << "needs AVX512-BF16";
    const nat::heuristic_params_t drift = params("cm_drift_kib=2048");
    const nat::native_problem_t p37 = problem(37, 4608, 8192, 32, true);
    const nat::native_plan_t g37 = plan(domains32(), 37, 4608, 8192, 32);
    ASSERT_EQ(g37.col_groups, 16) << text(g37);
    ASSERT_EQ(g37.row_unit, 8);
    ASSERT_EQ(g37.mr, 8);
    ASSERT_LT(g37.kc, 2048);
    const nat::native_plan_t q37 = nat::heuristic_plan(p37, domains32(), drift);
    EXPECT_TRUE(same(q37, g37)) << text(q37);
    const nat::native_problem_t p55 = problem(55, 4608, 8192, 32, true);
    const nat::native_plan_t g55 = plan(domains32(), 55, 4608, 8192, 32);
    ASSERT_EQ(g55.col_groups, 16) << text(g55);
    ASSERT_EQ(g55.row_unit, 8);
    const nat::native_plan_t q55 = nat::heuristic_plan(p55, domains32(), drift);
    EXPECT_EQ(q55.kc, 2048) << text(q55);
    EXPECT_EQ(q55.col_groups, 16);
    EXPECT_EQ(q55.row_unit, 12);
    EXPECT_EQ(q55.mr, 12);
    EXPECT_EQ(nat::validate_native_plan(q55, p55), nullptr);
}

// The slice of a single row block of more than two calls whose worker
// streams longer than it computes, over more than cm_block_kib KiB of
// weights, takes M blocks of tile_rows rows over all of K where the block's
// source rows and slab over K fit the bf16_l2_pct share of L2, on either
// hierarchy: M 32 of an lm_head of N 248320, K 4096.  At K 6656 they do
// not fit and the block stays; a shorter stream and a higher cm_block_kib
// keep 32 rows.
TEST(NativeCostModel, ShortBlocksForLongMemoryBoundStreams) {
    if (!bf16_kernels()) GTEST_SKIP() << "needs AVX512-BF16";
    for (const nat::UarchParams *u : kBoth) {
        SCOPED_TRACE("domain " + std::to_string(u->l3_domain_cores));
        const nat::native_plan_t q = plan(*u, 32, 248320, 4096, 32);
        EXPECT_EQ(q.mb, 6) << text(q);
        EXPECT_EQ(q.kc, 4096);
        EXPECT_EQ(q.col_groups, 0);
        EXPECT_EQ(plan(*u, 32, 202048, 6656, 32).mb, 32);
        EXPECT_EQ(plan(*u, 32, 24576, 4096, 32).mb, 32);
    }
    EXPECT_EQ(nat::heuristic_plan(problem(32, 248320, 4096, 32, true),
                      domains32(), params("cm_block_kib=1073741824"))
                      .mb,
            32);
    // A pinned K pass stays.
    EXPECT_EQ(nat::heuristic_plan(problem(32, 248320, 4096, 32, true),
                      domains32(), params("kc=512"))
                      .mb,
            32);
}

// Where one core streams slower than its share of the link, the M blocks
// of a BF16 team that fills each L3 domain it spans take only a K above
// cm_block_k: on the 32-thread domains the M 32 lm_heads of N 262144,
// K 1536 and of N 151936, K 2048 keep their K passes at 32 threads, which
// the 8- and 16-thread domains replace by the blocks, and K 2560 takes
// them.  cm_block_k=0 and a slower core's limit lifted give the blocks
// back.  Teams of 8 to 28 and of 48 threads, and f16 plans, take the plan
// of no bound: the blocks at 16 threads for N 128256, K 2048.
TEST(NativeCostModel, ShortBlocksNeedALongerKWhereOneCoreStreamsSlower) {
    if (!bf16_kernels()) GTEST_SKIP() << "needs AVX512-BF16";
    const nat::heuristic_params_t unbounded = params("cm_block_k=0");
    const int heads[][2] = {{262144, 1536}, {151936, 2048}, {128256, 2048}};
    for (const auto &s : heads)
        for (const int threads : {8, 16, 24, 28, 48}) {
            SCOPED_TRACE("N " + std::to_string(s[0]) + " K "
                    + std::to_string(s[1]) + " threads "
                    + std::to_string(threads));
            const nat::native_problem_t p
                    = problem(32, s[0], s[1], threads, true);
            EXPECT_TRUE(same(nat::heuristic_plan(p, domains32(),
                                     nat::heuristic_params_t {}),
                    nat::heuristic_plan(p, domains32(), unbounded)));
        }
    const nat::native_plan_t t16 = plan(domains32(), 32, 128256, 2048, 16);
    EXPECT_EQ(t16.mb, 6) << text(t16);
    EXPECT_EQ(t16.kc, 2048);
    for (const auto &s : heads) {
        SCOPED_TRACE(
                "f16 N " + std::to_string(s[0]) + " K " + std::to_string(s[1]));
        nat::native_problem_t p = problem(32, s[0], s[1], 32, true);
        p.src_dt = p.wei_dt = p.dst_dt = data_type_t::f16;
        const nat::native_plan_t f = nat::heuristic_plan(
                p, domains32(), nat::heuristic_params_t {});
        EXPECT_TRUE(same(f, nat::heuristic_plan(p, domains32(), unbounded)));
        if (f.kind == nat::native_kernel_kind_t::f16_ph && s[0] != 128256) {
            EXPECT_EQ(f.mb, 6) << text(f);
            EXPECT_EQ(f.kc, s[1]);
        }
    }
    const int shapes[][2] = {{262144, 1536}, {151936, 2048}};
    for (const auto &s : shapes) {
        SCOPED_TRACE(
                "N " + std::to_string(s[0]) + " K " + std::to_string(s[1]));
        const nat::native_problem_t p = problem(32, s[0], s[1], 32, true);
        const nat::native_plan_t q = plan(domains32(), 32, s[0], s[1], 32);
        EXPECT_EQ(q.mb, 32) << text(q);
        EXPECT_LT(q.kc, s[1]);
        for (const nat::UarchParams *u : {&domains8(), &domains16()}) {
            const nat::native_plan_t b = plan(*u, 32, s[0], s[1], 32);
            EXPECT_EQ(b.mb, 6) << text(b);
            EXPECT_EQ(b.kc, s[1]);
        }
        for (const std::string lift :
                {"cm_block_k=0", "cm_stream_cycles=424"}) {
            const nat::native_plan_t b
                    = nat::heuristic_plan(p, domains32(), params(lift));
            EXPECT_EQ(b.mb, 6) << lift << " " << text(b);
            EXPECT_EQ(b.kc, s[1]) << lift;
            EXPECT_EQ(nat::validate_native_plan(b, p), nullptr) << lift;
        }
    }
    const nat::native_plan_t r = plan(domains32(), 32, 262144, 2560, 32);
    EXPECT_EQ(r.mb, 6) << text(r);
    EXPECT_EQ(r.kc, 2560);
    EXPECT_EQ(nat::heuristic_plan(problem(32, 262144, 2560, 32, true),
                      domains32(), params("cm_block_k=2560"))
                      .mb,
            32);
}

// The rules for long weight streams keep a blocks schedule and its unit
// tiles: under schedule=blocks,mr=12 the M 55 block of N 65536, K 5120 on
// the 32-thread domain takes the long K pass (2048) in the blocks of the
// rules' plan, with one tile a unit or with unit_tiles=4, and under
// schedule=blocks the M 32 lm_head of N 248320, K 4096 keeps the rules'
// plan: M blocks of tile_rows rows are a rule of the slice.
TEST(NativeCostModel, LongStreamRulesKeepTheBlocksSchedule) {
    if (!bf16_kernels()) GTEST_SKIP() << "needs AVX512-BF16";
    const nat::native_problem_t p = problem(55, 65536, 5120, 32, true);
    for (const std::string pin :
            {"schedule=blocks,mr=12", "schedule=blocks,mr=12,unit_tiles=4"}) {
        const nat::native_plan_t r = nat::heuristic_plan(
                p, domains32(), params(pin + ",cost_model=0"));
        ASSERT_GT(r.unit_tiles, 0) << pin << " " << text(r);
        const nat::native_plan_t q
                = nat::heuristic_plan(p, domains32(), params(pin));
        EXPECT_EQ(q.kc, 2048) << pin << " " << text(q);
        EXPECT_EQ(q.mr, 12) << pin;
        EXPECT_EQ(q.col_groups, 0) << pin;
        EXPECT_EQ(q.unit_tiles, r.unit_tiles) << pin;
        EXPECT_EQ(q.domain_threads, 32) << pin;
        EXPECT_EQ(nat::validate_native_plan(q, p), nullptr) << pin;
    }
    const nat::native_problem_t lm = problem(32, 248320, 4096, 32, true);
    const nat::native_plan_t q
            = nat::heuristic_plan(lm, domains32(), params("schedule=blocks"));
    EXPECT_TRUE(same(q,
            nat::heuristic_plan(
                    lm, domains32(), params("schedule=blocks,cost_model=0"))))
            << text(q);
    EXPECT_GT(q.unit_tiles, 0);
}

// The long K pass needs one core to stream slower than its share of the
// domain's link, as on the 32-thread domains (424 cycles per KiB against
// cm_stream_cycles 600).  The derived rate of 16 threads to a domain ties
// it (2400 / sqrt(16) = 600): there the M 55 slices of the long pass keep
// the plan the model gives without the long-stream rules, on one domain
// and on two, while every tile's idle still counts (8 + 8 rows at M 16 of
// N 248320, which the 8-thread domains keep at 12 + 4), and the M 32
// lm_head takes the short blocks, which need no slower core.  A higher
// cm_stream_cycles gives them the long pass, and the tie holds on any
// hierarchy.
TEST(NativeCostModel, LongStreamRulesNeedASlowerCore) {
    const nat::heuristic_params_t hp;
    const nat::heuristic_params_t faster = params("cm_stream_cycles=601");
    const nat::heuristic_params_t none
            = params("cm_pass_kib=0,cm_block_kib=1073741824");
    EXPECT_DOUBLE_EQ(nat::native_plan_mem_cycles(domains16(), hp), 600.0);
    EXPECT_FALSE(nat::native_plan_core_streams(domains16(), hp));
    EXPECT_TRUE(nat::native_plan_core_streams(domains16(), faster));
    EXPECT_FALSE(nat::native_plan_core_streams(
            domains32(), params("cm_mem_cycles=600")));
    if (!bf16_kernels()) GTEST_SKIP() << "needs AVX512-BF16";
    const int shapes[][3] = {{55, 65536, 5120}, {55, 24576, 4096}};
    for (const int threads : {32, 64})
        for (const auto &s : shapes) {
            const nat::native_problem_t p
                    = problem(s[0], s[1], s[2], threads, true);
            EXPECT_TRUE(same(nat::heuristic_plan(p, domains16(), hp),
                    nat::heuristic_plan(p, domains16(), none)))
                    << "M " << s[0] << " N " << s[1] << " K " << s[2]
                    << " threads " << threads << " "
                    << text(nat::heuristic_plan(p, domains16(), hp));
        }
    EXPECT_EQ(nat::heuristic_plan(
                      problem(55, 65536, 5120, 32, true), domains16(), faster)
                      .kc,
            2048);
    for (const nat::heuristic_params_t *h : {&hp, &faster})
        EXPECT_EQ(nat::heuristic_plan(
                          problem(32, 248320, 4096, 32, true), domains16(), *h)
                          .mb,
                6);
    EXPECT_EQ(plan(domains16(), 16, 248320, 4096, 32).mr, 8);
    EXPECT_EQ(plan(domains8(), 16, 248320, 4096, 32).mr, 12);
}

// Two row groups are weighed only while a column tile's slab over all of K
// takes at most cm_group_slab_pct (75) percent of L2: at M 32, N 2560 the
// groups take K 4096 (256 KiB) and the slice keeps K 17920 (1120 KiB).
TEST(NativeCostModel, RowGroupsNeedTheSlabInL2) {
    if (!bf16_kernels()) GTEST_SKIP() << "needs AVX512-BF16";
    EXPECT_EQ(plan(domains32(), 32, 2560, 4096, 32).col_groups, 16);
    EXPECT_EQ(plan(domains32(), 32, 2560, 17920, 32).col_groups, 0);
    EXPECT_EQ(nat::heuristic_plan(problem(32, 2560, 17920, 32, true),
                      domains32(), params("cm_group_slab_pct=200"))
                      .col_groups,
            16);
}

// Two row groups are weighed only while the call's weights take at most
// cm_group_l3_pct (75) percent of the L3 of the domains the team spans:
// at M 32 the groups take N 4608, K 4096 (36 MiB of 128) and the slice
// keeps N 4608, K 12288 (108 MiB); a higher cm_group_l3_pct takes the
// groups there too, and the INT8 weights of one byte fit.
TEST(NativeCostModel, RowGroupsNeedTheWeightsInL3) {
    const nat::heuristic_params_t hp;
    const nat::native_problem_t within = problem(32, 4608, 4096, 32, true);
    const nat::native_problem_t beyond = problem(32, 4608, 12288, 32, true);
    const nat::native_plan_t q = rules(domains32(), 32, 4608, 4096, 32);
    EXPECT_TRUE(nat::native_plan_groups_fit_l3(within, q, domains32(), hp));
    EXPECT_FALSE(nat::native_plan_groups_fit_l3(beyond, q, domains32(), hp));
    EXPECT_TRUE(nat::native_plan_groups_fit_l3(
            problem(32, 4608, 12288, 32, false), q, domains32(), hp));
    if (!bf16_kernels()) GTEST_SKIP() << "needs AVX512-BF16";
    EXPECT_EQ(plan(domains32(), 32, 4608, 4096, 32).col_groups, 16);
    EXPECT_EQ(plan(domains32(), 32, 4608, 12288, 32).col_groups, 0);
    EXPECT_EQ(nat::heuristic_plan(
                      beyond, domains32(), params("cm_group_l3_pct=200"))
                      .col_groups,
            16);
}

// The model prices per-channel INT8 tiles: K in quads at the plan's call
// width and rows per call, one byte per weight, the K pass the INT8 tile
// takes for the tile's rows; it prices no per-group INT8 tile.
TEST(NativeCostModel, PricesPerChannelInt8Tiles) {
    const nat::heuristic_params_t hp;
    const nat::native_problem_t p = problem(16, 4096, 4096, 32, false);
    const nat::native_plan_t q = rules8(domains32(), 16, 4096, 4096, 32);
    const nat::native_plan_cost_t c
            = nat::native_plan_cost(q, p, domains32(), hp);
    EXPECT_GT(c.core, 0);
    EXPECT_GT(c.stream, 0);
    // The weight stream of a worker: its 4 column tiles of 32 x 4096 bytes.
    EXPECT_DOUBLE_EQ(c.stream_kib, 4 * 32 * 4096 / 1024.0);
    nat::native_problem_t g = p;
    g.int8_scheme = nat::native_int8_scheme_t::per_group;
    EXPECT_FALSE(nat::native_plan_cost_int8(g));
    EXPECT_DOUBLE_EQ(nat::native_plan_cost(q, g, domains32(), hp).cycles, 0.0);
    // Half the bytes of a BF16 weight: half the stream.
    const nat::native_problem_t b = problem(16, 4096, 4096, 32, true);
    EXPECT_DOUBLE_EQ(
            nat::native_plan_cost(
                    rules(domains32(), 16, 4096, 4096, 32), b, domains32(), hp)
                    .stream_kib,
            2 * c.stream_kib);
}

// A single per-channel INT8 row block takes two row groups where they
// balance the slice on the 32-thread domain (M 55, N 2560: 2.5 column tiles
// a thread), not a balanced slice (N 65536) nor with a higher
// cm_group_skew_pct, and not 12 + 20 rows (M 32); M 16 keeps 12 + 4 rows,
// and the 8-thread domains and pins keep the rules' plan.
TEST(NativeCostModel, Int8RowGroups) {
    if (!int8_kernels()) GTEST_SKIP() << "needs AVX512-VNNI";
    EXPECT_TRUE(same(plan8(domains32(), 16, 4096, 4096, 32),
            rules8(domains32(), 16, 4096, 4096, 32)));
    EXPECT_EQ(plan8(domains32(), 55, 2560, 4096, 32).col_groups, 16);
    EXPECT_EQ(plan8(domains32(), 55, 65536, 5120, 32).col_groups, 0);
    EXPECT_EQ(plan8(domains32(), 32, 2560, 4096, 32).col_groups, 0);
    EXPECT_EQ(nat::heuristic_plan(problem(55, 2560, 4096, 32, false),
                      domains32(), params("cm_group_skew_pct=200"))
                      .col_groups,
            0);
    for (const int M : {16, 32, 55})
        for (const int N : {2560, 24576})
            EXPECT_TRUE(same(plan8(domains8(), M, N, 4096, 32),
                    rules8(domains8(), M, N, 4096, 32)))
                    << "M " << M << " N " << N;
    for (const char *pin : {"int8_mr=10", "col_groups=0", "schedule=slice"}) {
        const nat::native_problem_t p = problem(55, 2560, 4096, 32, false);
        EXPECT_TRUE(same(nat::heuristic_plan(p, domains32(), params(pin)),
                nat::heuristic_plan(p, domains32(),
                        params(std::string(pin) + ",cost_model=0"))))
                << pin;
    }
}

// With cm_int8_l1_pct (off by default), where one core streams no faster
// than its share of the domain's link, a single INT8 row block of three
// calls (M 32) takes 6 rows per call and the L1 pass at that percent of L1
// (40: 128 quads at K 4096) while its slab and rows over K fit the
// int8_l2_pct share of L2.  Long K, K of fewer quads than the pass, the
// 8-thread domains and pins keep the rules' plan.
TEST(NativeCostModel, Int8BlocksOfThreeCallsTakeTileRows) {
    if (!int8_kernels()) GTEST_SKIP() << "needs AVX512-VNNI";
    const nat::heuristic_params_t on = params("cm_int8_l1_pct=40");
    const auto with = [&](const nat::UarchParams &u, int M, int N, int K) {
        return nat::heuristic_plan(problem(M, N, K, 32, false), u, on);
    };
    EXPECT_TRUE(same(plan8(domains32(), 32, 24576, 4096, 32),
            rules8(domains32(), 32, 24576, 4096, 32)));
    const nat::native_plan_t q32 = with(domains32(), 32, 24576, 4096);
    EXPECT_EQ(q32.int8_rows_nv2, 6) << text(q32);
    EXPECT_EQ(q32.int8_pass_quads, 128);
    EXPECT_EQ(q32.col_groups, 0);
    EXPECT_TRUE(same(with(domains32(), 32, 5120, 17920),
            rules8(domains32(), 32, 5120, 17920, 32)));
    EXPECT_TRUE(same(with(domains32(), 32, 4096, 122),
            rules8(domains32(), 32, 4096, 122, 32)));
    EXPECT_TRUE(same(with(domains8(), 32, 24576, 4096),
            rules8(domains8(), 32, 24576, 4096, 32)));
    const nat::native_problem_t p32 = problem(32, 24576, 4096, 32, false);
    for (const char *pin : {"int8_mr=10", "int8_kc=1024", "mb=16"})
        EXPECT_TRUE(
                same(nat::heuristic_plan(p32, domains32(),
                             params(std::string(pin) + ",cm_int8_l1_pct=40")),
                        nat::heuristic_plan(p32, domains32(),
                                params(std::string(pin) + ",cost_model=0"))))
                << pin;
    EXPECT_EQ(
            nat::heuristic_plan(p32, domains32(), params("cm_int8_l1_pct=100"))
                    .int8_pass_quads,
            256);
}

// An s4 plan is priced on its per-group tile: weights of half a byte an
// element plus the group's scale (bf16 scales at groups of 128: 0.515625
// bytes), so, with the stream idle and the L3 reads priced out, its memory
// term over the same grid is that of a BF16 weight times 0.515625 / 2; each
// group adds one FMA per row and output vector to the core term; below
// cm_int4_min_m rows the model leaves the s4 plan to the rules.
TEST(NativeCostModel, PricesInt4Tiles) {
    const nat::native_problem_t g128 = int4_problem(6, 4096, 4096, 32, 128);
    EXPECT_DOUBLE_EQ(nat::native_int4_weight_bytes(g128), 0.515625);
    EXPECT_DOUBLE_EQ(nat::native_int4_weight_bytes(
                             int4_problem(6, 4096, 4096, 32, 128, 4)),
            0.53125);
    EXPECT_DOUBLE_EQ(nat::native_int4_weight_bytes(
                             int4_problem(6, 4096, 4096, 32, 4096)),
            0.5 + 2.0 / 4096);
    const nat::heuristic_params_t hp
            = params("cm_prefetch_kib=1048576,cm_l3_cycles=0,cm_call_cycles=0");
    for (const nat::UarchParams *u : kBoth) {
        SCOPED_TRACE("domain " + std::to_string(u->l3_domain_cores));
        const nat::native_plan_t q = nat::heuristic_plan(g128, *u, hp);
        ASSERT_EQ(q.nr, 32);
        ASSERT_EQ(q.mb, 6);
        ASSERT_EQ(q.col_groups, 0);
        ASSERT_EQ(q.int8_rows_nv2, 6);
        const nat::native_plan_cost_t c
                = nat::native_plan_cost(q, g128, *u, hp);
        const nat::native_plan_cost_t b = nat::native_plan_cost(
                q, problem(6, 4096, 4096, 32, true), *u, hp);
        ASSERT_GT(c.memory, 0.0);
        EXPECT_NEAR(c.memory / b.memory, 0.515625 / 2, 1e-12);
        // 4 tiles per worker, one call of 6 rows over 2 output vectors and 2
        // pipes: 6 cycles more per group.
        const nat::native_plan_cost_t c32 = nat::native_plan_cost(
                q, int4_problem(6, 4096, 4096, 32, 32), *u, hp);
        EXPECT_DOUBLE_EQ(c32.core - c.core, 4 * (128 - 32) * 6.0);
        EXPECT_GT(nat::native_plan_cost(
                          q, int4_problem(6, 4096, 4096, 32, 128, 4), *u, hp)
                          .memory,
                c.memory);
        // A per-group s8 weight is not priced.
        nat::native_problem_t g8 = problem(6, 4096, 4096, 32, false);
        g8.int8_scheme = nat::native_int8_scheme_t::per_group;
        EXPECT_EQ(nat::native_plan_cost(q, g8, *u, hp).cycles, 0.0);
    }
}

// At the largest K an s4 weight takes, groups of 4 (536870910 of them) price
// their epilogue as many groups: 6 cycles more per group for each of the
// worker's 4 tiles of one 6-row call than groups of 8, whose count times
// the rows fits an int.
TEST(NativeCostModel, Int4GroupsOfTheLargestK) {
    const int K = nat::kMaxNativeInt4K;
    const nat::heuristic_params_t hp
            = params("cm_prefetch_kib=1048576,cm_l3_cycles=0,cm_call_cycles=0");
    for (const nat::UarchParams *u : kBoth) {
        SCOPED_TRACE("domain " + std::to_string(u->l3_domain_cores));
        const nat::native_plan_t q = nat::heuristic_plan(
                int4_problem(6, 4096, 4096, 32, 128), *u, hp);
        ASSERT_EQ(q.nr, 32);
        ASSERT_EQ(q.mb, 6);
        ASSERT_EQ(q.int8_rows_nv2, 6);
        const nat::native_plan_cost_t g4 = nat::native_plan_cost(
                q, int4_problem(6, 4096, K, 32, 4), *u, hp);
        const nat::native_plan_cost_t g8 = nat::native_plan_cost(
                q, int4_problem(6, 4096, K, 32, 8), *u, hp);
        ASSERT_GT(g8.core, 0.0);
        EXPECT_DOUBLE_EQ(g4.core - g8.core, 4 * (536870910.0 - 268435455) * 6);
        EXPECT_GT(g4.cycles, g8.cycles);
    }
}

// A single s4 row block of cm_int4_min_m rows or more whose slice does not
// divide its column tiles over the threads takes the M blocks the model
// rates fastest, on both hierarchies: at M 128 and 32 threads, N 6656 (208
// tiles, 6.5 a thread) two blocks of 64 rows (13 tiles a thread), N 3840
// (120 tiles) four blocks of 32 rows (15 a thread).  Slices that divide (N
// 4096 and 5120), fewer rows than cm_int4_min_m, pins and the switches
// keep the rules' plan; the row blocks keep the plan legal.
TEST(NativeCostModel, Int4RowBlocks) {
    const nat::heuristic_params_t built_in;
    const nat::heuristic_params_t off = params("cost_model=0");
    ASSERT_EQ(built_in.cm_int4_min_m, 48);
    if (!int8_kernels()) GTEST_SKIP() << "needs AVX512-VNNI";
    for (const nat::UarchParams *u : kBoth) {
        SCOPED_TRACE("domain " + std::to_string(u->l3_domain_cores));
        const struct {
            int N, K, mb;
        } taken[] = {{6656, 19968, 64}, {3840, 15360, 32}};
        for (const auto &t : taken) {
            const nat::native_problem_t p
                    = int4_problem(128, t.N, t.K, 32, 128);
            const nat::native_plan_t r = nat::heuristic_plan(p, *u, off);
            ASSERT_EQ(r.m_blocks, 1) << t.N;
            ASSERT_NE(r.n_tiles % r.threads, 0) << t.N;
            const nat::native_plan_t q = nat::heuristic_plan(p, *u, built_in);
            EXPECT_EQ(q.mb, t.mb) << t.N << " " << text(q);
            EXPECT_EQ(q.nr, r.nr) << t.N;
            EXPECT_EQ(q.threads, r.threads) << t.N;
            EXPECT_EQ(q.col_groups, 0) << t.N;
            EXPECT_EQ(nat::validate_native_plan(q, p), nullptr) << t.N;
            EXPECT_LT(nat::native_plan_cost(q, p, *u, built_in).cycles,
                    0.95 * nat::native_plan_cost(r, p, *u, built_in).cycles)
                    << t.N;
            for (const char *pin : {"mb=128", "col_groups=0", "schedule=slice",
                         "cm_int4_min_m=0", "cm_int4_min_m=129",
                         "cm_gain_pct=100"})
                EXPECT_TRUE(same(nat::heuristic_plan(p, *u, params(pin)),
                        nat::heuristic_plan(p, *u,
                                params(std::string(pin) + ",cost_model=0"))))
                        << t.N << " " << pin;
        }
        for (const int N : {4096, 5120}) {
            const nat::native_problem_t p
                    = int4_problem(128, N, 14336, 32, 128);
            EXPECT_TRUE(same(nat::heuristic_plan(p, *u, built_in),
                    nat::heuristic_plan(p, *u, off)))
                    << N;
        }
        const nat::native_problem_t below = int4_problem(
                built_in.cm_int4_min_m - 1, 6656, 19968, 32, 128);
        EXPECT_TRUE(same(nat::heuristic_plan(below, *u, built_in),
                nat::heuristic_plan(below, *u, off)));
        EXPECT_GT(nat::heuristic_plan(below, *u,
                          params("cm_int4_min_m=" + std::to_string(below.M)))
                          .m_blocks,
                1);
    }
}

// The s4 row blocks of a weight of one group over K are the ones the model
// rates at the INT4 group kernel's rows per call (the plan at int8_mr=6),
// on both hierarchies, and the plan keeps the 12 rows per call of the
// kernels of one group inside them: among them the slices that 12-row
// prices would block otherwise (M 55 of N 7680, K 5120 and of N 6656, K
// 4096; M 64-128 of N 4608, K 6656), and blocks every price takes.
TEST(NativeCostModel, Int4OneGroupRowBlocksAtTheGroupKernelRows) {
    const nat::heuristic_params_t built_in;
    const nat::heuristic_params_t six = params("int8_mr=6");
    if (!int8_kernels()) GTEST_SKIP() << "needs AVX512-VNNI";
    size_t blocked = 0;
    for (const nat::UarchParams *u : kBoth) {
        SCOPED_TRACE("domain " + std::to_string(u->l3_domain_cores));
        const struct {
            int N, K;
        } shapes[] = {{7680, 5120}, {6656, 4096}, {4608, 6656}, {3840, 15360},
                {6656, 19968}, {128256, 4096}};
        for (const auto &s : shapes)
            for (const int M : {48, 55, 64, 80, 96, 112, 128}) {
                const nat::native_problem_t p
                        = int4_problem(M, s.N, s.K, 32, s.K);
                const nat::native_plan_t q
                        = nat::heuristic_plan(p, *u, built_in);
                const nat::native_plan_t g = nat::heuristic_plan(p, *u, six);
                const std::string at = std::to_string(M) + " "
                        + std::to_string(s.N) + " " + std::to_string(s.K);
                EXPECT_EQ(q.int8_rows_nv2, 12) << at;
                EXPECT_EQ(g.int8_rows_nv2, 6) << at;
                EXPECT_EQ(q.mb, g.mb) << at << " " << text(q);
                EXPECT_EQ(q.m_blocks, g.m_blocks) << at;
                EXPECT_EQ(nat::validate_native_plan(q, p), nullptr) << at;
                blocked += q.m_blocks > 1;
            }
    }
    EXPECT_GT(blocked, 0u);
}

// Under the cost model one group over K takes the int8_l2_pct share of L2,
// on both hierarchies, whatever cm_pass_calls: one pass where the group's
// slab fits it (M 32 of N 4096, K 14336, which the rules' plan splits into
// passes of the int4_l1_pct share of L1; an lm_head of N 128256, K 4096 at
// M 128; M 16), passes of that share for K 32768, whose slab exceeds it.
// Groups of 128 keep the L1 share.
TEST(NativeCostModel, Int4OneGroupTakesTheL2Share) {
    const nat::heuristic_params_t built_in;
    if (!int8_kernels()) GTEST_SKIP() << "needs AVX512-VNNI";
    const auto passes = [](const nat::native_plan_t &q, int K) {
        const int quads = K / 4;
        return quads
                / nat::native_int4_pass_quads(
                        q, quads, 32, std::min(q.mb, 32), q.int8_rows_nv2);
    };
    for (const nat::UarchParams *u : kBoth) {
        SCOPED_TRACE("domain " + std::to_string(u->l3_domain_cores));
        const int64_t l2 = static_cast<int64_t>(u->l2_bytes)
                * built_in.int8_l2_pct / 100;
        const nat::native_problem_t dn
                = int4_problem(32, 4096, 14336, 32, 14336);
        const nat::native_plan_t r
                = nat::heuristic_plan(dn, *u, params("cost_model=0"));
        ASSERT_LT(r.int8_pass_budget, l2);
        EXPECT_GT(passes(r, 14336), 1);
        for (const char *input : {"", "cm_pass_calls=1", "cm_pass_calls=3"}) {
            SCOPED_TRACE(input);
            const nat::native_plan_t q = nat::heuristic_plan(
                    dn, *u, *input ? params(input) : built_in);
            EXPECT_EQ(q.int8_pass_budget, l2) << text(q);
            EXPECT_EQ(passes(q, 14336), 1);
        }
        const nat::native_problem_t lm
                = int4_problem(128, 128256, 4096, 32, 4096);
        const nat::native_plan_t l = nat::heuristic_plan(lm, *u, built_in);
        EXPECT_EQ(l.int8_pass_budget, l2) << text(l);
        EXPECT_EQ(passes(l, 4096), 1);
        const nat::native_problem_t dn16
                = int4_problem(16, 4096, 14336, 32, 14336);
        EXPECT_EQ(nat::heuristic_plan(dn16, *u, built_in).int8_pass_budget, l2);
        const nat::native_problem_t wide
                = int4_problem(32, 5120, 32768, 32, 32768);
        const nat::native_plan_t w = nat::heuristic_plan(wide, *u, built_in);
        EXPECT_EQ(w.int8_pass_budget, l2) << text(w);
        EXPECT_GT(passes(w, 32768), 1);
        EXPECT_LT(nat::heuristic_plan(int4_problem(128, 128256, 4096, 32, 128),
                          *u, built_in)
                          .int8_pass_budget,
                l2);
    }
}

// FP32 plans, per-group INT8 plans, INT4 plans below cm_int4_min_m rows or
// of more than one row block, INT8 plans of more than one row block or of
// at most 8 rows, BF16 tiles of fewer than 32 columns and single-call
// decode tiles keep the rules' plan.
TEST(NativeCostModel, LeavesOtherPlansToTheRules) {
    const nat::heuristic_params_t built_in;
    const nat::heuristic_params_t off = params("cost_model=0");
    size_t checked = 0;
    for (const nat::UarchParams *u : kBoth)
        for (const int threads : {8, 32, 64})
            for (const int M : {1, 4, 8, 16, 32, 55, 1024, 4096})
                for (const int N : {256, 1536, 4096, 28672})
                    for (const int K : {1536, 4096, 14336}) {
                        nat::native_problem_t s8
                                = problem(M, N, K, threads, false);
                        if (M <= 16 || M > 64) {
                            EXPECT_TRUE(
                                    same(nat::heuristic_plan(s8, *u, built_in),
                                            nat::heuristic_plan(s8, *u, off)))
                                    << "s8 M " << M << " N " << N << " K " << K;
                        }
                        nat::native_problem_t group = s8;
                        group.int8_scheme
                                = nat::native_int8_scheme_t::per_group;
                        EXPECT_TRUE(
                                same(nat::heuristic_plan(group, *u, built_in),
                                        nat::heuristic_plan(group, *u, off)))
                                << "per-group s8 M " << M << " N " << N << " K "
                                << K;
                        const nat::native_problem_t s4
                                = int4_problem(M, N, K, threads, 128);
                        EXPECT_TRUE(same(nat::heuristic_plan(s4, *u, built_in),
                                nat::heuristic_plan(s4, *u, off)))
                                << "s4 M " << M << " N " << N << " K " << K;
                        nat::native_problem_t f32 = s8;
                        f32.src_dt = f32.wei_dt = f32.dst_dt = data_type_t::f32;
                        f32.dynamic_quant = false;
                        EXPECT_TRUE(same(nat::heuristic_plan(f32, *u, built_in),
                                nat::heuristic_plan(f32, *u, off)))
                                << "f32 M " << M << " N " << N << " K " << K;
                        const nat::native_problem_t bf16
                                = problem(M, N, K, threads, true);
                        const nat::native_plan_t r
                                = nat::heuristic_plan(bf16, *u, off);
                        if (r.nr < 32 || M <= 8) {
                            EXPECT_TRUE(same(
                                    nat::heuristic_plan(bf16, *u, built_in), r))
                                    << "bf16 M " << M << " N " << N << " K "
                                    << K << " " << text(r);
                        }
                        ++checked;
                    }
    EXPECT_GT(checked, 0u);
}

// What a source pins stays pinned: rows per call, the K pass, the schedule
// and the column groups; tall_rows bounds the rows per call the model
// weighs; cost_model=0 is the rules' plan and cm_gain_pct=100 keeps it.
TEST(NativeCostModel, PinsAndSwitches) {
    for (const nat::UarchParams *u : kBoth) {
        const nat::native_problem_t m16 = problem(16, 4096, 4096, 32, true);
        const nat::native_problem_t m32 = problem(32, 4096, 4096, 32, true);
        EXPECT_EQ(nat::heuristic_plan(m16, *u, params("mr=12")).mr, 12);
        EXPECT_EQ(nat::heuristic_plan(m16, *u, params("kc=4096")).kc, 4096);
        EXPECT_EQ(nat::heuristic_plan(m16, *u, params("tall_rows=6")).mr, 6);
        for (const char *slice : {"schedule=slice", "col_groups=0"})
            EXPECT_EQ(nat::heuristic_plan(m32, *u, params(slice)).col_groups, 0)
                    << slice;
        EXPECT_EQ(
                nat::heuristic_plan(m32, *u, params("col_groups=8")).col_groups,
                8);
        EXPECT_TRUE(
                same(nat::heuristic_plan(m16, *u, params("cm_gain_pct=100")),
                        rules(*u, 16, 4096, 4096, 32)));
        EXPECT_TRUE(same(nat::heuristic_plan(m16, *u, params("cost_model=0")),
                rules(*u, 16, 4096, 4096, 32)));
        // cm_short_rows names the shorter call: calls of 4 rows wait for
        // the accumulators, and 12 + 4 rows stay.
        EXPECT_EQ(
                nat::heuristic_plan(m16, *u, params("cm_short_rows=4")).mr, 12);
    }
}

// The model's plans are legal for every input of its keys' ranges, on both
// hierarchies, over teams of 1 to 96 threads and shapes at the edges of the
// tile grid, and a planner without sources returns them.
TEST(NativeCostModel, PlansAreLegalOverTheInputs) {
    if (!bf16_kernels() || !int8_kernels())
        GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    const char *const inputs[] = {"", "cm_gain_pct=0", "cm_short_rows=1",
            "cm_short_rows=11", "cm_fma_pipes=1", "cm_fma_pipes=16",
            "cm_fma_lat=1", "cm_fma_lat=64", "cm_call_cycles=0",
            "cm_call_cycles=1048576", "cm_tall_pct=0", "cm_tall_pct=100",
            "cm_l2_bytes=1", "cm_l2_bytes=4096", "cm_l3_cycles=1048576",
            "cm_mem_cycles=1", "cm_mem_cycles=1048576", "cm_prefetch_kib=0",
            "cm_prefetch_kib=1048576", "cm_overlap_pct=0", "cm_overlap_pct=100",
            "cm_reuse_pct=1", "cm_reuse_pct=1000", "cm_stream_cycles=0",
            "cm_stream_cycles=1048576", "cm_sync_tiles=0",
            "cm_sync_tiles=1048576", "cm_group_slab_pct=1",
            "cm_group_slab_pct=100000", "cm_group_l3_pct=1",
            "cm_group_l3_pct=100000", "cm_drift_kib=0",
            "cm_drift_kib=1073741824", "cm_pass_kib=0", "cm_pass_kib=1",
            "cm_pass_kib=1048576", "cm_pass_calls=1", "cm_pass_calls=4096",
            "cm_block_kib=0", "cm_block_kib=1073741824", "tile_rows=1",
            "cm_group_skew_pct=100", "cm_group_skew_pct=100000",
            "cm_group_rows_pct=100", "cm_group_rows_pct=200",
            "cm_int8_l1_pct=0", "cm_int8_l1_pct=1", "cm_int8_l1_pct=1000",
            "cm_int4_min_m=0", "cm_int4_min_m=1", "cm_int4_min_m=1073741824",
            "tall_rows=8", "row_unit=1", "min_tiles_per_thread=4"};
    size_t checked = 0, failed = 0;
    for (const char *input : inputs) {
        const nat::heuristic_params_t hp = params(input);
        for (const nat::UarchParams *u : kBoth) {
            const nat::native_planner_t planner(*u,
                    *input ? nat::make_native_override_source(input) : nullptr,
                    nullptr, nullptr);
            for (const int threads : {1, 2, 7, 16, 32, 33, 64, 96})
                for (const int M : {1, 9, 13, 16, 17, 24, 32, 40, 55, 64, 65,
                             129, 1024, 1031})
                    for (const int N : {32, 64, 96, 1536, 2560, 4096, 151936})
                        for (const int K : {2, 64, 1000, 4096, 14336})
                            for (const int dt : {0, 1, 2}) {
                                const bool bf16 = dt == 0;
                                const nat::native_problem_t p = dt == 2
                                        ? int4_problem(M, N, K, threads, K)
                                        : problem(M, N, K, threads, bf16);
                                const nat::native_plan_t q
                                        = nat::heuristic_plan(p, *u, hp);
                                const char *why
                                        = nat::validate_native_plan(q, p);
                                ++checked;
                                if (why != nullptr && failed++ < 10)
                                    ADD_FAILURE() << input
                                                  << (bf16 ? " bf16"
                                                                     : dt == 2
                                                                     ? " s4"
                                                                     : " s8")
                                                  << " M " << M << " N " << N
                                                  << " K " << K << " threads "
                                                  << threads << ": " << why;
                                if (!*input && !same(planner.plan(p), q)
                                        && failed++ < 10)
                                    ADD_FAILURE()
                                            << "the planner differs at M " << M
                                            << " N " << N << " K " << K;
                            }
        }
    }
    EXPECT_EQ(failed, 0u) << "of " << checked;
}

} // namespace
