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

/// Group matmul's machine view and N-tile flat parallel's config: what each
/// derives, and how the config's clauses resolve.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "lowoha_operators/matmul/group_matmul/common/grp_matmul_machine.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/nr32_pack.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/nr32_tile.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/ntile_flat_parallel_config.hpp"

namespace {

namespace mm = zendnnl::lowoha::matmul;
namespace gm = zendnnl::lowoha::matmul::grp_machine;
namespace nfp = zendnnl::lowoha::matmul::ntile_flat_parallel;
using mm::grp_matmul_phase;
using zendnnl::common::data_type_t;

mm::native::native_hw_t topology(const int smt, const int l3_threads) {
    mm::native::native_hw_t hw;
    hw.smt = smt;
    hw.l3_domain_threads = l3_threads;
    return hw;
}

/// Rates as a core measures them, {ipc, latency} on each path.
std::array<gm::dot_rate, gm::kDotPaths> rates(const gm::dot_rate fp32,
        const gm::dot_rate bf16, const gm::dot_rate int8,
        const gm::dot_rate fp16 = {}) {
    return {fp32, bf16, int8, fp16};
}

/// `topology(2, 16)` of an AMD CPU of `family` and `model`.
mm::native::native_hw_t amd(const int family, const int model) {
    mm::native::native_hw_t hw = topology(2, 16);
    hw.vendor = "AuthenticAMD";
    hw.family = family;
    hw.model = model;
    return hw;
}

// The u8-source kernels pay where INT8 issues more than 1.5 times the faster
// of BF16's and FP32's measured rates: Zen 6's 3.75 against 2, not Zen 5's 2
// against 2, Zen 4's 1 against 1, a CPU whose BF16 is slow (2 against FP32's
// 2) or one with VNNI and no BF16, whatever the CPU's model.  Without measured
// rates they pay on AMD family 1Ah models 50h-5Fh only.
TEST(GrpMatmulMachine, TheRatesDecideTheU8SourceKernels) {
    const gm::dot_rates_t zen6 = gm::make_dot_rates(amd(0x1A, 0x50),
            rates({2.0, 4}, {2.0, 6}, {3.75, 4}, {2.0, 4}), true);
    EXPECT_TRUE(zen6.u8_src_pays);
    EXPECT_EQ(zen6.chains(gm::dot_path::int8), 15);
    EXPECT_EQ(zen6.chains(gm::dot_path::bf16), 12);
    const auto zen5 = rates({2.0, 4}, {2.0, 6}, {2.0, 4});
    EXPECT_FALSE(gm::make_dot_rates(amd(0x1A, 0x44), zen5, true).u8_src_pays);
    EXPECT_FALSE(gm::make_dot_rates(amd(0x1A, 0x50), zen5, true).u8_src_pays);
    const gm::dot_rates_t zen4 = gm::make_dot_rates(
            amd(0x19, 0x11), rates({1.0, 5}, {0.75, 6}, {1.0, 4}), true);
    EXPECT_FALSE(zen4.u8_src_pays);
    EXPECT_EQ(zen4.chains(gm::dot_path::int8), 4);
    EXPECT_FALSE(gm::make_dot_rates(
            topology(2, 16), rates({1.0, 4}, {}, {1.0, 5}), true)
                         .u8_src_pays);
    EXPECT_FALSE(gm::make_dot_rates(
            topology(2, 16), rates({2.0, 4}, {0.5, 9}, {2.0, 5}), true)
                         .u8_src_pays);
    EXPECT_FALSE(
            gm::make_dot_rates(topology(2, 16), rates({}, {}, {1.0, 5}), true)
                    .u8_src_pays);
    const gm::dot_rates_t unmeasured
            = gm::make_dot_rates(amd(0x1A, 0x50), {}, false);
    EXPECT_TRUE(unmeasured.u8_src_pays);
    EXPECT_EQ(unmeasured.chains(gm::dot_path::int8), 1);
    EXPECT_TRUE(gm::make_dot_rates(amd(0x1A, 0x5F), {}, false).u8_src_pays);
    EXPECT_FALSE(gm::make_dot_rates(amd(0x1A, 0x44), {}, false).u8_src_pays);
    EXPECT_FALSE(gm::make_dot_rates(amd(0x19, 0x11), {}, false).u8_src_pays);
    EXPECT_FALSE(gm::make_dot_rates(topology(2, 16), {}, false).u8_src_pays);
}

// Each path this CPU has runs at a quarter to eight instructions a cycle, in
// quarters, at a latency of 1 to 32 cycles; a path it lacks is not measured.
TEST(GrpMatmulMachine, MeasuresEachPathThisCpuHas) {
    const mm::native::UarchParams &u = mm::native::native_hw().uarch;
    std::array<gm::dot_rate, gm::kDotPaths> dot;
    if (!gm::measure_dot_rates(u, dot)) {
        for (const gm::dot_rate &d : dot)
            EXPECT_EQ(d.ipc, 0.0);
        GTEST_SKIP() << "this build or CPU measures no path";
    }
    const bool have[gm::kDotPaths]
            = {u.avx512f, u.avx512bf16, u.avx512vnni, u.avx512fp16};
    for (int p = 0; p < gm::kDotPaths; ++p) {
        SCOPED_TRACE(gm::kDotPathNames[p]);
        const gm::dot_rate &d = dot[p];
        // A build without the FP16 loop leaves that path unmeasured.
        if (!have[p]
                || (p == static_cast<int>(gm::dot_path::fp16) && d.ipc == 0)) {
            EXPECT_EQ(d.ipc, 0.0);
            continue;
        }
        EXPECT_GE(d.ipc, 0.25);
        EXPECT_LE(d.ipc, 8.0);
        EXPECT_EQ(d.ipc * 4, std::round(d.ipc * 4));
        EXPECT_GE(d.latency, 1);
        EXPECT_LE(d.latency, 32);
    }
}

// Plans size for the L2, or for 256 KiB below 64 KiB, and budget 80% of it;
// the L3 domain counts physical cores, and is 0 when the OS doesn't say.
TEST(GrpMatmulMachine, CachesAndTopology) {
    mm::native::native_hw_t hw = topology(2, 16);
    hw.uarch.l2_bytes = 1 << 20;
    EXPECT_EQ(gm::make_machine(hw).l2_plan_bytes, 1 << 20);
    EXPECT_EQ(gm::make_machine(hw).l2_budget_bytes, (1 << 20) * 4 / 5);
    hw.uarch.l2_bytes = 32 * 1024;
    EXPECT_EQ(gm::make_machine(hw).l2_plan_bytes, 256 * 1024);
    EXPECT_EQ(gm::make_machine(hw).l2_budget_bytes, (256 * 1024) * 4 / 5);
    EXPECT_EQ(gm::make_machine(hw).l3_domain_cores, 8);
    hw.l3_domain_threads = 0;
    EXPECT_EQ(gm::make_machine(hw).l3_domain_cores, 0);
    hw.smt = 0;
    hw.l3_domain_threads = 8;
    EXPECT_EQ(gm::make_machine(hw).smt, 1);
    EXPECT_EQ(gm::make_machine(hw).l3_domain_cores, 8);
}

// The process's view holds this CPU's caches and its rates: the u8-source
// kernels pay on Zen 6 (family 1Ah models 50h-5Fh), and not on Zen 4 or Zen 5
// (family 19h, and 1Ah models below 50h), measured or not.
TEST(GrpMatmulMachine, TheProcessViewMatchesItsHardware) {
    const mm::native::native_hw_t &hw = mm::native::native_hw();
    const gm::machine_t &m = gm::machine();
    const gm::dot_rates_t &d = gm::dot_rates();
    EXPECT_EQ(m.l2_budget_bytes, gm::make_machine(hw).l2_budget_bytes);
    EXPECT_FALSE(gm::describe(m, d).empty());
    if (hw.vendor != "AuthenticAMD") return;
    if (hw.family == 0x1A && hw.model >= 0x50 && hw.model <= 0x5F) {
        EXPECT_TRUE(d.u8_src_pays);
    }
    if (hw.family == 0x19 || (hw.family == 0x1A && hw.model < 0x50)) {
        EXPECT_FALSE(d.u8_src_pays);
    }
}

/// Restores the environment's config when a test ends.
class NTileFlatParallelConfig : public ::testing::Test {
protected:
    ~NTileFlatParallelConfig() override {
        mm::test_api::set_ntile_flat_parallel_config(nullptr);
    }
};

// With no config, every setting has the default the docs give.
TEST_F(NTileFlatParallelConfig, TheDefaults) {
    mm::test_api::set_ntile_flat_parallel_config("");
    EXPECT_TRUE(mm::test_api::ntile_flat_parallel_config_warnings().empty());
    for (const data_type_t wei : {data_type_t::bf16, data_type_t::f16,
                 data_type_t::s8, data_type_t::s4}) {
        for (const nfp::api_kind api :
                {nfp::api_kind::routed, nfp::api_kind::grouped}) {
            for (const grp_matmul_phase phase :
                    {grp_matmul_phase::decode, grp_matmul_phase::prompt}) {
                const nfp::call_knobs &k = nfp::call_knobs_for(phase, wei, api);
                EXPECT_EQ(k.units, 2);
                EXPECT_EQ(k.prompt_units, 6);
                EXPECT_EQ(k.sweep_rows, mm::kDecodeMaxM);
                EXPECT_EQ(k.order, nfp::unit_order::automatic);
                EXPECT_TRUE(k.heavy_light);
            }
            const nfp::plan_knobs &p = nfp::plan_knobs_for(wei, api);
            EXPECT_EQ(p.subtile_rows, 16);
            EXPECT_EQ(nfp::l2_budget_bytes(wei, api),
                    gm::machine().l2_budget_bytes);
        }
    }
    EXPECT_EQ(nfp::u8_src(data_type_t::s8, nfp::api_kind::routed),
            gm::dot_rates().u8_src_pays);
    EXPECT_EQ(nfp::u8_src(data_type_t::s8, nfp::api_kind::grouped),
            gm::dot_rates().u8_src_pays);
    const nfp::limits_t &l = nfp::limits();
    EXPECT_EQ(l.gather_rows, 16);
    EXPECT_EQ(l.wait_spins, 4096);
    EXPECT_EQ(l.scratch_keep_bytes, size_t {1} << 30);
    EXPECT_EQ(l.call_sites, 1024u);
    EXPECT_EQ(l.warmed_pools, 4096u);
}

// For each setting, the first clause whose filters hold and that sets it.
TEST_F(NTileFlatParallelConfig, TheFirstMatchingClauseThatSetsASettingWins) {
    mm::test_api::set_ntile_flat_parallel_config(
            "phase=prompt,dtype=s8:prompt_units=12;api=grouped:units=5;"
            "units=3,prompt_units=4,heavy_light=off;phase=decode:units=9");
    EXPECT_TRUE(mm::test_api::ntile_flat_parallel_config_warnings().empty());
    const auto knobs = [](grp_matmul_phase p, data_type_t w, nfp::api_kind a) {
        return nfp::call_knobs_for(p, w, a);
    };
    const nfp::call_knobs s8_prompt = knobs(
            grp_matmul_phase::prompt, data_type_t::s8, nfp::api_kind::routed);
    EXPECT_EQ(s8_prompt.prompt_units, 12);
    EXPECT_EQ(s8_prompt.units, 3);
    EXPECT_FALSE(s8_prompt.heavy_light);
    const nfp::call_knobs bf16_prompt = knobs(
            grp_matmul_phase::prompt, data_type_t::bf16, nfp::api_kind::routed);
    EXPECT_EQ(bf16_prompt.prompt_units, 4);
    EXPECT_EQ(knobs(grp_matmul_phase::decode, data_type_t::s4,
                      nfp::api_kind::grouped)
                      .units,
            5);
    EXPECT_EQ(knobs(grp_matmul_phase::decode, data_type_t::s4,
                      nfp::api_kind::routed)
                      .units,
            3);
    EXPECT_EQ(knobs(grp_matmul_phase::decode, data_type_t::s4,
                      nfp::api_kind::routed)
                      .sweep_rows,
            mm::kDecodeMaxM);
}

// Plan settings follow the weight type and API; process limits take only a
// clause without filters.
TEST_F(NTileFlatParallelConfig, PlanSettingsAndLimits) {
    mm::test_api::set_ntile_flat_parallel_config(
            "dtype=s8:l2_frac=0.5,u8_src=on,subtile_rows=8;"
            "api=grouped:u8_src=off;gather_rows=8,call_sites=7,"
            "scratch_keep_mib=64");
    EXPECT_TRUE(mm::test_api::ntile_flat_parallel_config_warnings().empty());
    const gm::machine_t &m = gm::machine();
    EXPECT_EQ(nfp::l2_budget_bytes(data_type_t::s8, nfp::api_kind::routed),
            static_cast<int64_t>(m.l2_plan_bytes * 0.5));
    EXPECT_EQ(nfp::l2_budget_bytes(data_type_t::bf16, nfp::api_kind::routed),
            m.l2_budget_bytes);
    EXPECT_EQ(nfp::plan_knobs_for(data_type_t::s8, nfp::api_kind::grouped)
                      .subtile_rows,
            8);
    EXPECT_TRUE(nfp::u8_src(data_type_t::s8, nfp::api_kind::routed));
    // The dtype clause comes first and sets u8_src for grouped calls too.
    EXPECT_TRUE(nfp::u8_src(data_type_t::s8, nfp::api_kind::grouped));
    mm::test_api::set_ntile_flat_parallel_config(
            "api=grouped:u8_src=off;u8_src=on");
    EXPECT_FALSE(nfp::u8_src(data_type_t::s8, nfp::api_kind::grouped));
    EXPECT_TRUE(nfp::u8_src(data_type_t::s8, nfp::api_kind::routed));
    // Only s8 has u8-source kernels.
    EXPECT_FALSE(nfp::u8_src(data_type_t::s4, nfp::api_kind::routed));
    EXPECT_FALSE(nfp::u8_src(data_type_t::bf16, nfp::api_kind::routed));
    mm::test_api::set_ntile_flat_parallel_config(
            "gather_rows=8,call_sites=7,scratch_keep_mib=64");
    EXPECT_EQ(nfp::limits().gather_rows, 8);
    EXPECT_EQ(nfp::limits().call_sites, 7u);
    EXPECT_EQ(nfp::limits().scratch_keep_bytes, size_t {64} << 20);
}

// A bad clause is dropped with one warning; the good ones still apply.
TEST_F(NTileFlatParallelConfig, BadClausesAreDroppedWithOneWarningEach) {
    const std::vector<std::string> bad = {
            "units=0",
            "unitz=2",
            "units",
            "units=2.5",
            "units=inf",
            "units=1e100",
            "l2_frac=nan",
            "units=2,units=3",
            "u8_src=maybe",
            "phase=lunch:units=2",
            "dtype=s8,dtype=bf16:units=2",
            "mode=fast:units=2",
            "phase=decode:l2_frac=0.5",
            "dtype=s8:gather_rows=8",
            "dtype=s4:u8_src=on",
            "dtype=s8:",
    };
    std::string text;
    for (const std::string &b : bad)
        text += b + ";";
    text += "units=7";
    mm::test_api::set_ntile_flat_parallel_config(text.c_str());
    EXPECT_EQ(mm::test_api::ntile_flat_parallel_config_warnings().size(),
            bad.size());
    EXPECT_EQ(nfp::call_knobs_for(grp_matmul_phase::decode, data_type_t::bf16,
                      nfp::api_kind::routed)
                      .units,
            7);
    EXPECT_EQ(nfp::limits().gather_rows, 16);
}

// The L2 budget and subtile rows size a plan's subtiles.
TEST(NTileFlatParallelPlan, TheBudgetSizesTheSubtiles) {
    nfp::nr32::weight_desc d;
    d.wei = data_type_t::bf16;
    d.K = 4096;
    d.N = 4096;
    d.trans = true;
    d.ldb = d.K;
    const nfp::nr32::pack_layout layout = nfp::nr32::layout_of(d);
    nfp::nr32::tile_plan wide, narrow;
    const int64_t l2 = 2 << 20;
    if (!nfp::nr32::make_plan(layout, nfp::nr32::epilogue::plain, false, wide,
                false, {l2, 16})) {
        GTEST_SKIP() << "no bf16 kernel on this CPU";
    }
    ASSERT_TRUE(nfp::nr32::make_plan(layout, nfp::nr32::epilogue::plain, false,
            narrow, false, {l2 / 4, 16}));
    EXPECT_GT(wide.subtile, narrow.subtile);
    nfp::nr32::tile_plan rows;
    ASSERT_TRUE(nfp::nr32::make_plan(
            layout, nfp::nr32::epilogue::plain, false, rows, false, {l2, 64}));
    EXPECT_LT(rows.subtile, wide.subtile);
}

} // namespace
