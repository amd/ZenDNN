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

// Cache sizes CPUID does not report: the detected parameters of a CPU
// without an L3, and the native BRGEMM plan on such a CPU, whose L3 rules
// stay off.

#include <string>

#include <gtest/gtest.h>

#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_config.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_planner.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_hw.hpp"
#include "lowoha_operators/matmul/matmul_native/native_layout_policy.hpp"
#include "native_brgemm_test_utils.hpp"

namespace {

namespace nat = zendnnl::lowoha::matmul::native;
using zendnnl::memory::data_type_t;

// CPUID's sizes are kept; an L1d or L2 it does not report takes the
// constructor's size, an L3 it does not report the L2 size.
TEST(NativeCacheFallback, UnreportedL3IsTheL2) {
    nat::UarchParams p;
    nat::set_cache_sizes(p, 48 << 10, 2 << 20, 0);
    EXPECT_EQ(p.l1d_bytes, 48 << 10);
    EXPECT_EQ(p.l2_bytes, 2 << 20);
    EXPECT_EQ(p.l3_domain_bytes, 2 << 20);
    EXPECT_TRUE(p.l1d_detected);
    EXPECT_TRUE(p.l2_detected);
    EXPECT_FALSE(p.l3_detected);

    nat::set_cache_sizes(p, 0, 0, 0);
    const nat::UarchParams d;
    EXPECT_EQ(p.l1d_bytes, d.l1d_bytes);
    EXPECT_EQ(p.l2_bytes, d.l2_bytes);
    EXPECT_EQ(p.l3_domain_bytes, d.l2_bytes);
    EXPECT_FALSE(p.l1d_detected || p.l2_detected || p.l3_detected);

    nat::set_cache_sizes(p, 32 << 10, 1 << 20, 32 << 20);
    EXPECT_EQ(p.l1d_bytes, 32 << 10);
    EXPECT_EQ(p.l2_bytes, 1 << 20);
    EXPECT_EQ(p.l3_domain_bytes, 32 << 20);
    EXPECT_TRUE(p.l1d_detected && p.l2_detected && p.l3_detected);
}

/// A CPU of 32 cores with 48 KiB of L1d and 1 MiB of L2 per core, 8 cores
/// per L3 domain, and l3 bytes of L3 per domain; reported as no L3 when l3
/// is 0, with the L3 size detection then gives.
nat::native_hw_t cpu(int l3) {
    nat::native_hw_t hw = nat::native_hw();
    hw.uarch.l1d_bytes = 48 << 10;
    hw.uarch.l2_bytes = 1 << 20;
    hw.uarch.l3_domain_bytes = l3 > 0 ? l3 : 32 << 20;
    hw.uarch.l3_detected = l3 > 0;
    hw.uarch.family_l3_cores = 1;
    hw.smt = 1;
    hw.l3_domain_threads = 8;
    return hw;
}

nat::native_problem_t problem(int M, int N, int K, bool bf16) {
    nat::native_problem_t p;
    p.M = M;
    p.N = N;
    p.K = K;
    p.pack_nr = nat::kNativeLayoutPackNr;
    p.threads = 32;
    p.src_dt = data_type_t::bf16;
    p.wei_dt = bf16 ? data_type_t::bf16 : data_type_t::s8;
    p.dst_dt = data_type_t::bf16;
    p.dynamic_quant = !bf16;
    return p;
}

nat::heuristic_params_t params(const char *config) {
    nat::native_config_t cfg;
    std::string why;
    EXPECT_TRUE(nat::parse_native_config(config, cfg, why)) << why;
    return nat::apply_native_config(nat::heuristic_params_t {}, cfg);
}

// The plan of a CPU whose CPUID reports no L3 has none: its L3 size is 0.
TEST(NativeCacheFallback, PlanHasNoL3WithoutOne) {
    EXPECT_EQ(nat::native_plan_uarch(cpu(0)).l3_domain_bytes, 0);
    EXPECT_EQ(nat::native_plan_uarch(cpu(32 << 20)).l3_domain_bytes, 32 << 20);
}

// Without an L3 the plan takes neither the L3 row-group test, the traffic
// rule nor the blocks schedule, which the same problems take with 32 MiB
// of L3 per domain.
TEST(NativeCacheFallback, L3RulesStayOffWithoutAnL3) {
    const nat::UarchParams with_l3 = nat::native_plan_uarch(cpu(32 << 20));
    const nat::UarchParams no_l3 = nat::native_plan_uarch(cpu(0));
    // The L3 row-group test (traffic_groups=0): the source and a domain's B
    // slabs over 32 MiB, 24 column tiles for 32 threads.
    const nat::heuristic_params_t row_test
            = params(native_brgemm_test::kNativeRulesOff);
    const nat::native_problem_t p = problem(4096, 768, 4096, true);
    EXPECT_EQ(nat::heuristic_plan(p, with_l3, row_test).col_groups, 8);
    EXPECT_EQ(nat::heuristic_plan(p, no_l3, row_test).col_groups, 0);

    // Only the traffic rule, then only the blocks schedule, over problems
    // of both data types.
    const nat::heuristic_params_t traffic = params(
            "makespan=0,l1_pass=0,traffic_groups=1,bf16_alias=0,"
            "bf16_narrow_n=0,blocks=0");
    const nat::heuristic_params_t blocks = params(
            "makespan=0,l1_pass=0,traffic_groups=0,bf16_alias=0,"
            "bf16_narrow_n=0,blocks=1");
    int traffic_with_l3 = 0, blocks_with_l3 = 0;
    for (const bool bf16 : {true, false})
        for (const int M : {256, 1024, 4096})
            for (const int N : {768, 4096})
                for (const int K : {2048, 8192}) {
                    const nat::native_problem_t q = problem(M, N, K, bf16);
                    SCOPED_TRACE(std::string(bf16 ? "bf16" : "int8") + " M="
                            + std::to_string(M) + " N=" + std::to_string(N)
                            + " K=" + std::to_string(K));
                    traffic_with_l3 += nat::heuristic_plan(q, with_l3, traffic)
                                               .col_groups
                            > 0;
                    blocks_with_l3 += nat::heuristic_plan(q, with_l3, blocks)
                                              .unit_tiles
                            > 0;
                    EXPECT_EQ(nat::heuristic_plan(q, no_l3, traffic).col_groups,
                            0);
                    EXPECT_EQ(nat::heuristic_plan(q, no_l3, blocks).unit_tiles,
                            0);
                }
    EXPECT_GT(traffic_with_l3, 0);
    EXPECT_GT(blocks_with_l3, 0);
}

} // namespace
