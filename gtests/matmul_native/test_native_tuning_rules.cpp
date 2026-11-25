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

// The rules of the native BRGEMM plan behind their switches: the makespan
// grid with its row-group fallback (never narrower than the narrowing's
// tile), the L1-sized K pass, col_groups by memory traffic, fewer BF16 rows
// per call for source rows a multiple of 4 KiB apart, BF16 tiles of several
// 32-column slabs for few column tiles against the team's L3 domains, and the
// blocks schedule where one L3 domain's share of the source fits its L3.  A
// planner over an override that sets every switch on must return the
// built-in plan.  Every statically legal pin must give a legal plan through
// the planner, and a pin the plan with every rule switched off keeps must
// stay legal in heuristic_plan.  Each rule sits behind its own switch, one
// switch per int32 key of range 0 .. 1: switched on alone it changes only the
// fields it decides.  Worked examples show one plan of each rule on two cache
// hierarchies, and the properties each rule keeps.

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/int8/native_int4_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_config.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_planner.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_tuning_file.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_tuning_keys.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_hw.hpp"
#include "lowoha_operators/matmul/matmul_native/native_layout_policy.hpp"
#include "matmul_native/native_brgemm_test_utils.hpp"

namespace {

namespace nat = zendnnl::lowoha::matmul::native;
using zendnnl::memory::data_type_t;

constexpr data_type_t kBf16 = data_type_t::bf16;
constexpr data_type_t kS8 = data_type_t::s8;

nat::native_problem_t problem(
        int M, int N, int K, int pack_nr, int threads, bool bf16) {
    nat::native_problem_t p;
    p.M = M;
    p.N = N;
    p.K = K;
    p.pack_nr = pack_nr;
    p.threads = threads;
    p.src_dt = kBf16;
    p.wei_dt = bf16 ? kBf16 : kS8;
    p.dst_dt = kBf16;
    p.dynamic_quant = !bf16;
    return p;
}

/// The problem of an s4 weight in groups of gs: an s8 source the call
/// quantizes, per-group scales.
nat::native_problem_t int4_problem(int M, int N, int K, int threads, int gs) {
    nat::native_problem_t p
            = problem(M, N, K, nat::kNativeLayoutPackNr, threads, false);
    p.wei_dt = data_type_t::s4;
    p.int8_scheme = nat::native_int8_scheme_t::per_group;
    p.wei_group_size = gs;
    return p;
}

std::string describe(const nat::native_problem_t &p) {
    std::ostringstream os;
    os << (p.bf16() ? "bf16" : (p.int4() ? "int4" : "int8")) << " M=" << p.M
       << " N=" << p.N << " K=" << p.K << " pack_nr=" << p.pack_nr
       << " threads=" << p.threads;
    if (p.int4()) os << " gs=" << p.wei_group_size;
    return os.str();
}

nat::heuristic_params_t params(const std::string &text) {
    nat::native_config_t cfg;
    std::string why;
    EXPECT_TRUE(nat::parse_native_config(text, cfg, why))
            << text << ": " << why;
    return nat::apply_native_config(nat::heuristic_params_t {}, cfg);
}

/// The parameters of base with the keys overlay sets replacing its own.
nat::heuristic_params_t over(
        const std::string &base, const std::string &overlay) {
    nat::native_config_t a, b;
    std::string why;
    EXPECT_TRUE(nat::parse_native_config(base, a, why)) << base << ": " << why;
    EXPECT_TRUE(nat::parse_native_config(overlay, b, why))
            << overlay << ": " << why;
    nat::merge_native_config(a, b);
    return nat::apply_native_config(nat::heuristic_params_t {}, a);
}

/// Every rule switch at value.
std::string every_switch(const char *value) {
    std::string text;
    for (const std::string &s : native_brgemm_test::native_rule_switches())
        text += (text.empty() ? "" : ",") + s + "=" + value;
    return text;
}

/// Every rule switch off but the switch of rule.
std::string only(const std::string &rule) {
    std::string text;
    for (const std::string &s : native_brgemm_test::native_rule_switches())
        text += (text.empty() ? "" : ",") + s + (s == rule ? "=1" : "=0");
    return text;
}

bool has_isa() {
    const nat::UarchParams &u = nat::detect_uarch();
    return u.avx512bf16 && u.avx512vnni;
}

/// A cache hierarchy of 48 KiB L1, 1 MiB L2 and l3_mib MiB of L3 shared by
/// domain cores.
nat::UarchParams caches(int l3_mib, int domain) {
    nat::UarchParams u = nat::detect_uarch();
    u.l1d_bytes = 48 << 10;
    u.l2_bytes = 1 << 20;
    u.l3_domain_bytes = l3_mib << 20;
    u.family_l3_cores = 1;
    u.l3_domain_cores = domain;
    return u;
}

/// Counts checks and lists the first failures in full.
class report_t {
public:
    template <typename Describe>
    void check(bool ok, Describe &&describe) {
        ++checked_;
        if (ok) return;
        if (failed_++ < kListed) ADD_FAILURE() << describe();
    }
    void finish() const {
        EXPECT_EQ(failed_, 0u) << "of " << checked_ << " checks (at most "
                               << kListed << " listed above)";
    }

private:
    static constexpr size_t kListed = 20;
    size_t checked_ = 0;
    size_t failed_ = 0;
};

/// The plan fields a rule may decide, as a mask.
enum field_t : unsigned {
    kGrid = 1, ///< nr, mr, mb, m_blocks, n_tiles, threads
    kPass = 2, ///< kc, int8_pass_quads
    kGroups = 4, ///< col_groups, unit_tiles, domain_threads
    kScratch = 8, ///< scratch_floats
    kRowUnit = 16, ///< row_unit
    kBudget = 32, ///< int8_pass_budget
};

/// The fields in which a and b differ, as a mask; other fields that differ
/// set every bit above kRowUnit.
unsigned plan_fields(const nat::native_plan_t &a, const nat::native_plan_t &b) {
    unsigned m = 0;
    if (a.nr != b.nr || a.mr != b.mr || a.mb != b.mb || a.m_blocks != b.m_blocks
            || a.n_tiles != b.n_tiles || a.threads != b.threads)
        m |= kGrid;
    if (a.kc != b.kc || a.int8_pass_quads != b.int8_pass_quads) m |= kPass;
    if (a.col_groups != b.col_groups || a.unit_tiles != b.unit_tiles
            || a.domain_threads != b.domain_threads)
        m |= kGroups;
    if (a.scratch_floats != b.scratch_floats) m |= kScratch;
    if (a.row_unit != b.row_unit) m |= kRowUnit;
    if (a.int8_pass_budget != b.int8_pass_budget) m |= kBudget;
    if (a.traversal != b.traversal || a.int8_rows_nv1 != b.int8_rows_nv1
            || a.int8_rows_nv2 != b.int8_rows_nv2
            || a.int8_cw_mask != b.int8_cw_mask
            || a.int8_direct_store != b.int8_direct_store
            || a.int8_bias_in_dequant != b.int8_bias_in_dequant
            || a.copy_block != b.copy_block || a.copy_par_min != b.copy_par_min
            || a.sources != b.sources)
        m |= ~0u << 6;
    return m;
}

bool same_plan(const nat::native_plan_t &a, const nat::native_plan_t &b) {
    return plan_fields(a, b) == 0;
}

struct shape_t {
    int N, K;
};

const shape_t kShapes[]
        = {{32, 64}, {96, 4097}, {256, 1536}, {1024, 1000}, {1536, 4096},
                {3072, 8192}, {5120, 32768}, {14336, 4096}, {28672, 4096}};
const int kMs[]
        = {1, 7, 8, 12, 24, 32, 64, 65, 128, 193, 1031, 3077, 4096, 8192};

TEST(NativeTuningRules, PlannerReturnsTheRulesPlan) {
    if (!has_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    const nat::UarchParams &u = nat::native_plan_uarch();
    const nat::heuristic_params_t rules = nat::heuristic_params_t {};
    const nat::native_planner_t planner(u,
            nat::make_native_override_source(every_switch("1")), nullptr,
            nullptr);
    report_t report;
    for (const int threads : {1, 8, 24, 32, 64, 128})
        for (const shape_t &s : kShapes)
            for (const int M : kMs)
                for (const bool bf16 : {true, false}) {
                    const nat::native_problem_t p = problem(M, s.N, s.K,
                            nat::kNativeLayoutPackNr, threads, bf16);
                    nat::native_plan_t want = nat::heuristic_plan(p, u, rules);
                    const nat::native_plan_t got = planner.plan(p);
                    want.sources = got.sources;
                    report.check(same_plan(got, want)
                                    && nat::native_plan_source_names(got)
                                            == "override",
                            [&] {
                        return describe(p)
                                    + ": the planner's plan is not the rules' "
                                      "plan";
                    });
                }
    report.finish();
}

/// Every statically legal value of every key as "key=value",
/// sampled for wide integer ranges (both ends, a middle value and the powers
/// of two).
std::vector<std::string> legal_settings() {
    std::vector<std::string> out;
    size_t count = 0;
    const nat::native_config_key_t *keys = nat::native_config_keys(count);
    for (size_t i = 0; i < count; ++i) {
        const nat::native_config_key_t &k = keys[i];
        const std::string name = k.name;
        switch (k.type) {
            case nat::native_key_type_t::boolean:
                out.push_back(name + "=0");
                out.push_back(name + "=1");
                break;
            case nat::native_key_type_t::cw_list:
                for (const char *v : {"16", "32/16"})
                    out.push_back(name + "=" + v);
                break;
            case nat::native_key_type_t::schedule:
            case nat::native_key_type_t::traversal:
            case nat::native_key_type_t::kc_split:
                for (const std::string &v :
                        native_brgemm_test::native_enum_values(k.type))
                    out.push_back(name + "=" + v);
                break;
            default: {
                std::set<int64_t> values;
                const auto add = [&](int64_t v) {
                    v -= v % k.step;
                    if (v >= k.min && v <= k.max
                            && (!k.pow2 || (v & (v - 1)) == 0))
                        values.insert(v);
                };
                for (int64_t v = 1; v <= k.max && v > 0; v *= 2)
                    add(v);
                for (int64_t v : {k.min, k.min + k.step, (k.min + k.max) / 2,
                             k.max, int64_t {3}, int64_t {5}, int64_t {7},
                             int64_t {12}, int64_t {13}})
                    add(v);
                for (const int64_t v : values)
                    out.push_back(name + "=" + std::to_string(v));
            }
        }
    }
    return out;
}

/// Shapes at the edges of the tile grid, and teams around a round of tiles.
std::vector<nat::native_problem_t> edge_problems() {
    std::vector<nat::native_problem_t> v;
    for (bool bf16 : {true, false})
        for (int N : {32, 64, 96, 128, 1536, 4096})
            for (int K : {1, 64, 4097})
                for (int M : {1, 5, 6, 7, 11, 12, 13, 24, 25, 64, 100, 191, 192,
                             193, 1031})
                    for (int threads : {1, 8, 32, 33, 256})
                        v.push_back(problem(M, N, K, nat::kNativeLayoutPackNr,
                                threads, bf16));
    return v;
}

TEST(NativeTuningRules, TotalOverLegalPins) {
    if (!has_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    const nat::UarchParams &uarch = nat::native_plan_uarch();
    const std::vector<nat::native_problem_t> problems = edge_problems();
    report_t report;
    for (const std::string &setting : legal_settings()) {
        const nat::heuristic_params_t none
                = over(native_brgemm_test::kNativeRulesOff, setting);
        const std::string &text = setting;
        const nat::heuristic_params_t rules = params(text);
        const std::unique_ptr<nat::native_planner_t> planner
                = nat::make_native_planner(text.c_str(), nullptr);
        for (const nat::native_problem_t &p : problems) {
            const nat::native_plan_t q = nat::heuristic_plan(p, uarch, rules);
            const char *planned
                    = nat::validate_native_plan(planner->plan(p), p);
            report.check(planned == nullptr, [&] {
                return text + " " + describe(p)
                        + ": the planner returns an illegal plan: " + planned;
            });
            if (nat::validate_native_plan(
                        nat::heuristic_plan(p, uarch, none), p)
                    != nullptr)
                continue;
            const char *why = nat::validate_native_plan(q, p);
            report.check(why == nullptr, [&] {
                return text + " " + describe(p)
                        + ": legal with every rule off, illegal under the "
                          "rules: "
                        + why;
            });
        }
    }
    report.finish();
}

TEST(NativeTuningRules, EachRuleChangesOnlyItsFields) {
    const struct {
        std::string setting;
        unsigned fields;
    } rules[] = {
            {only("makespan"), kGrid | kPass | kGroups | kScratch},
            {only("l1_pass"), kPass | kScratch},
            {only("traffic_groups"), kGroups},
            {only("bf16_alias"), kGrid | kPass | kScratch},
            {only("bf16_narrow_n"), kGrid | kPass | kGroups | kScratch},
            {only("blocks"), kGroups},
            {only("cost_model"), kGrid | kPass | kGroups | kScratch | kRowUnit},
    };
    // Every switch of the key table has its entry, and kNativeRulesOff sets
    // each of them, and nothing else, to 0.
    for (const std::string &s : native_brgemm_test::native_rule_switches()) {
        size_t entries = 0;
        for (const auto &r : rules)
            entries += r.setting == only(s);
        EXPECT_EQ(entries, 1u) << s;
    }
    nat::native_config_t off;
    std::string why;
    ASSERT_TRUE(nat::parse_native_config(
            native_brgemm_test::kNativeRulesOff, off, why))
            << why;
    EXPECT_EQ(nat::format_native_config(off), every_switch("0"));
    const nat::heuristic_params_t none
            = params(native_brgemm_test::kNativeRulesOff);
    report_t report;
    size_t changed[std::size(rules)] = {};
    for (const nat::UarchParams &u : {caches(32, 8), caches(128, 32)})
        for (const int threads : {8, 32, 64})
            for (const shape_t &s : kShapes)
                for (const int M : kMs)
                    for (const bool bf16 : {true, false}) {
                        const nat::native_problem_t p = problem(M, s.N, s.K,
                                nat::kNativeLayoutPackNr, threads, bf16);
                        const nat::native_plan_t base
                                = nat::heuristic_plan(p, u, none);
                        for (size_t r = 0; r < std::size(rules); ++r) {
                            const unsigned diff = plan_fields(
                                    nat::heuristic_plan(
                                            p, u, params(rules[r].setting)),
                                    base);
                            changed[r] += diff != 0;
                            report.check((diff & ~rules[r].fields) == 0, [&] {
                                return rules[r].setting + " " + describe(p)
                                        + " changes a field of another rule";
                            });
                        }
                    }
    report.finish();
    // The narrow-N rule's wide plan and the cost model's variants validate
    // only where the registry offers BF16 kernels; the other rules change
    // INT8 plans on any CPU.
    const bool bf16_kernels = nat::detect_uarch().avx512bf16;
    for (size_t r = 0; r < std::size(rules); ++r)
        if (bf16_kernels
                || (rules[r].setting != only("bf16_narrow_n")
                        && rules[r].setting != only("cost_model"))) {
            EXPECT_GT(changed[r], 0u) << rules[r].setting << " changed no plan";
        }
}

// The blocks rule on 8 cores per 32 MiB of L3, 32 threads: a BF16 call whose
// B slab over all of K exceeds half of L2, or an INT8 call the slice serves,
// takes the blocks schedule of one column tile per unit once a domain's
// quarter of the source fits its L3; not with one domain, a quarter over the
// L3 share, a slab within half of L2 (BF16), row groups (INT8), fewer rows
// than the rule's bound, or the switch off.  A team that is not a multiple
// of the domain weighs a full domain's share.  The plans are checked legal
// where the registry offers the dtype's kernels.
TEST(NativeTuningRules, BlocksRuleExamples) {
    const nat::UarchParams domain8 = caches(32, 8), domain32 = caches(128, 32);
    const nat::UarchParams &host = nat::detect_uarch();
    const auto plan = [](const nat::UarchParams &u, const char *settings, int M,
                              int N, int K, bool bf16) {
        return nat::heuristic_plan(
                problem(M, N, K, nat::kNativeLayoutPackNr, 32, bf16), u,
                params(settings));
    };
    const auto blocks = [](const nat::native_plan_t &q) {
        return q.unit_tiles == 1 && q.col_groups == 0 && q.domain_threads == 8;
    };
    // BF16, K = 14336: a quarter of the source is 28 MiB, the B slab 896 KiB.
    const nat::native_plan_t b = plan(domain8, "", 4096, 4096, 14336, true);
    EXPECT_TRUE(blocks(b));
    EXPECT_LT(b.kc, 14336);
    if (host.avx512bf16) {
        EXPECT_EQ(nat::validate_native_plan(b,
                          problem(4096, 4096, 14336, nat::kNativeLayoutPackNr,
                                  32, true)),
                nullptr);
    }
    EXPECT_FALSE(blocks(plan(domain8, "", 4096, 5120, 32768, true)));
    EXPECT_FALSE(blocks(plan(domain8, "", 4096, 4096, 4096, true)));
    EXPECT_FALSE(blocks(plan(domain8, "", 512, 4096, 14336, true)));
    EXPECT_TRUE(blocks(plan(domain8, "", 1024, 4096, 14336, true)));
    EXPECT_FALSE(blocks(
            plan(domain8, "bf16_blocks_min_m=2048", 1024, 4096, 14336, true)));
    EXPECT_EQ(plan(domain32, "", 4096, 4096, 14336, true).unit_tiles, 0);
    EXPECT_EQ(plan(domain8, native_brgemm_test::kNativeRulesOff, 4096, 4096,
                      14336, true)
                      .unit_tiles,
            0);
    // The rule needs no other rule.
    EXPECT_TRUE(blocks(
            plan(domain8, only("blocks").c_str(), 4096, 4096, 14336, true)));
    EXPECT_EQ(plan(domain8, "blocks=0", 4096, 4096, 14336, true).unit_tiles, 0);
    // M=2048: 16 threads give each domain half the rows, 28 MiB of source;
    // 12 threads deal the first domain 8 / 12 of the row blocks, over the
    // L3 share.
    const auto team = [&](int threads) {
        return nat::heuristic_plan(
                problem(2048, 4096, 14336, nat::kNativeLayoutPackNr, threads,
                        true),
                domain8, params(""));
    };
    EXPECT_GT(team(16).unit_tiles, 0);
    EXPECT_EQ(team(12).unit_tiles, 0);
    // INT8: the slice of K = 4096 takes it from int8_blocks_min_m rows.
    const nat::native_plan_t i = plan(domain8, "", 4096, 4096, 4096, false);
    EXPECT_TRUE(blocks(i));
    if (host.avx512vnni) {
        EXPECT_EQ(nat::validate_native_plan(i,
                          problem(4096, 4096, 4096, nat::kNativeLayoutPackNr,
                                  32, false)),
                nullptr);
    }
    EXPECT_TRUE(blocks(plan(domain8, "", 2048, 4096, 4096, false)));
    // A tile whose B slab over all of K is below L1 (32 x 1024 bytes) takes it
    // from half as many rows more.
    EXPECT_FALSE(blocks(plan(domain8, "", 2048, 4096, 1024, false)));
    EXPECT_TRUE(blocks(plan(domain8, "", 3072, 4096, 1024, false)));
    // An L3 domain of more than 16 cores takes it from half as many rows more.
    const auto plan64 = [&](int M, const char *text) {
        return nat::heuristic_plan(
                problem(M, 4096, 4096, nat::kNativeLayoutPackNr, 64, false),
                domain32, params(text));
    };
    EXPECT_EQ(plan64(2048, "").unit_tiles, 0);
    EXPECT_EQ(plan64(3072, "").unit_tiles, 1);
    EXPECT_EQ(plan64(3072, "").domain_threads, 32);
    EXPECT_FALSE(blocks(plan(domain8, "", 1024, 4096, 4096, false)));
    // A set int8_blocks_min_m takes it from that many rows for every tile.
    EXPECT_EQ(nat::heuristic_params_t {}.int8_blocks_min_m, -1);
    EXPECT_TRUE(blocks(
            plan(domain8, "int8_blocks_min_m=1024", 1024, 4096, 4096, false)));
    EXPECT_TRUE(blocks(
            plan(domain8, "int8_blocks_min_m=2048", 2048, 4096, 1024, false)));
    EXPECT_FALSE(blocks(
            plan(domain8, "int8_blocks_min_m=3072", 2048, 4096, 4096, false)));
    EXPECT_TRUE(blocks(
            plan(domain8, "int8_blocks_min_m=3072", 3072, 4096, 1024, false)));
    EXPECT_EQ(plan64(2048, "int8_blocks_min_m=2048").unit_tiles, 1);
    EXPECT_FALSE(blocks(plan(domain8, "", 4096, 4096, 14336, false)));
    // A pinned schedule or col_groups keeps its own plan.
    EXPECT_EQ(
            plan(domain8, "schedule=slice", 4096, 4096, 14336, true).unit_tiles,
            0);
    EXPECT_EQ(plan(domain8, "col_groups=8", 4096, 4096, 14336, true).col_groups,
            8);
}

// The blocks rule on 8 cores per 32 MiB of L3, 32 threads, for a BF16 slice
// of 32-column tiles with at most two column tiles per domain (N up to 256):
// it takes the blocks schedule of one tile per unit from bf16_blocks_min_m
// rows, keeping the slice's grid; not where bf16_narrow_n widens the tiles
// (N 64), past two tiles per domain, under row groups, below the rule's rows,
// on one domain, with the switch off, or for INT8.  16 threads are two
// domains: four column tiles at most.
TEST(NativeTuningRules, BlocksRuleNarrowN) {
    const nat::UarchParams domain8 = caches(32, 8), domain32 = caches(128, 32);
    const nat::UarchParams &host = nat::detect_uarch();
    const auto plan
            = [](const nat::UarchParams &u, const char *settings, int M, int N,
                      int K, int threads = 32, bool bf16 = true) {
        return nat::heuristic_plan(
                problem(M, N, K, nat::kNativeLayoutPackNr, threads, bf16), u,
                params(settings));
    };
    const auto blocks = [](const nat::native_plan_t &q) {
        return q.unit_tiles == 1 && q.col_groups == 0 && q.domain_threads == 8
                && q.nr == 32;
    };
    for (const int M : {1024, 4096})
        for (const int N : {96, 256}) {
            SCOPED_TRACE("M " + std::to_string(M) + " N " + std::to_string(N));
            const nat::native_plan_t slice
                    = plan(domain8, "blocks=0", M, N, 1536);
            const nat::native_plan_t q = plan(domain8, "", M, N, 1536);
            ASSERT_EQ(slice.col_groups, 0);
            EXPECT_TRUE(blocks(q));
            EXPECT_EQ(q.mb, slice.mb);
            EXPECT_EQ(q.mr, slice.mr);
            EXPECT_EQ(q.kc, slice.kc);
            EXPECT_EQ(q.threads, slice.threads);
            EXPECT_EQ(q.scratch_floats, slice.scratch_floats);
            if (host.avx512bf16) {
                EXPECT_EQ(nat::validate_native_plan(q,
                                  problem(M, N, 1536, nat::kNativeLayoutPackNr,
                                          32, true)),
                        nullptr);
            }
        }
    const nat::native_plan_t n64 = plan(domain8, "", 4096, 64, 2560);
    EXPECT_EQ(n64.unit_tiles, 0);
    EXPECT_EQ(n64.nr, 64);
    EXPECT_EQ(plan(domain8, "", 4096, 288, 1536).unit_tiles, 0);
    EXPECT_EQ(plan(domain8, "col_groups=1", 4096, 256, 1536).unit_tiles, 0);
    EXPECT_EQ(plan(domain8, "", 512, 256, 1536).unit_tiles, 0);
    EXPECT_FALSE(
            blocks(plan(domain8, "bf16_blocks_min_m=2048", 1024, 256, 1536)));
    EXPECT_EQ(plan(domain32, "", 4096, 256, 1536).unit_tiles, 0);
    EXPECT_EQ(plan(domain8, "", 1024, 256, 1536, 32, false).unit_tiles, 0);
    EXPECT_TRUE(blocks(plan(domain8, "", 4096, 96, 1536, 16)));
    EXPECT_EQ(plan(domain8, "", 4096, 256, 1536, 16).unit_tiles, 0);
    EXPECT_EQ(plan(domain8, "", 4096, 256, 1536, 8).unit_tiles, 0);
}

// An s4 weight is half a byte an element: up to decode_m the makespan
// weighs a source row as two weight columns, the traffic rule weighs the
// weight at half the bytes of an s8 weight, and the tile's group batches
// take the int4_l1_pct share of L1.  Next to an s8 weight of per-group
// scales its plans differ only in the fields those rules (and the cost
// model's row blocks) decide, and under each of the makespan and traffic
// rules somewhere; the rules' plans do not depend on the group size (the
// cost model's may, by the price of the groups), and the plans are legal
// over every shape, M and team where the registry offers the s4 kernels.
TEST(NativeTuningRules, Int4Weights) {
    const bool kernels = nat::detect_uarch().avx512vnni;
    const struct {
        std::string setting;
        unsigned fields;
    } rules[] = {{every_switch("1"), kGrid | kGroups | kScratch | kBudget},
            {only("makespan"), kGrid | kGroups | kScratch | kBudget},
            {only("traffic_groups"), kGroups | kBudget}};
    report_t report;
    size_t changed[std::size(rules)] = {};
    for (size_t r = 0; r < std::size(rules); ++r) {
        const nat::heuristic_params_t hp = params(rules[r].setting);
        for (const nat::UarchParams &u : {caches(32, 8), caches(128, 32)})
            for (const int threads : {8, 24, 32, 64})
                for (const shape_t &s : kShapes)
                    for (const int M : kMs) {
                        nat::native_problem_t s8 = problem(M, s.N, s.K,
                                nat::kNativeLayoutPackNr, threads, false);
                        s8.int8_scheme = nat::native_int8_scheme_t::per_group;
                        // At least two groups over K.
                        const int gs = std::min(128, s.K / 2);
                        const nat::native_problem_t s4
                                = int4_problem(M, s.N, s.K, threads, gs);
                        const nat::native_plan_t q
                                = nat::heuristic_plan(s4, u, hp);
                        const unsigned diff = plan_fields(
                                q, nat::heuristic_plan(s8, u, hp));
                        changed[r] += diff != 0;
                        report.check((diff & ~rules[r].fields) == 0, [&] {
                            return rules[r].setting + " " + describe(s4)
                                    + " changes a field of another rule";
                        });
                        const nat::native_problem_t quarter
                                = int4_problem(M, s.N, s.K, threads, gs / 4);
                        nat::heuristic_params_t rules_only = hp;
                        rules_only.cost_model = 0;
                        const nat::native_plan_t r4
                                = nat::heuristic_plan(s4, u, rules_only);
                        report.check(same_plan(r4,
                                             nat::heuristic_plan(
                                                     quarter, u, rules_only)),
                                [&] {
                            return rules[r].setting + " " + describe(s4)
                                    + ": the plan depends on the group size";
                        });
                        const char *why = kernels
                                ? nat::validate_native_plan(q, s4)
                                : nullptr;
                        report.check(why == nullptr, [&] {
                            return rules[r].setting + " " + describe(s4)
                                    + ": illegal plan: " + why;
                        });
                        // One group over K: the rules' plan of several
                        // groups at the rows of the kernels of one group.
                        const nat::native_problem_t one
                                = int4_problem(M, s.N, s.K, threads, s.K);
                        nat::native_plan_t want = r4;
                        want.int8_rows_nv1 = std::min(rules_only.int8_mr,
                                nat::native_int4_one_group_ukernel_max_mr(1));
                        want.int8_rows_nv2 = std::min(rules_only.int8_mr,
                                nat::native_int4_one_group_ukernel_max_mr(2));
                        report.check(same_plan(nat::heuristic_plan(
                                                       one, u, rules_only),
                                             want),
                                [&] {
                            return rules[r].setting + " " + describe(one)
                                    + ": the plan of one group differs in "
                                      "more than its rows per call";
                        });
                        const char *why_one = kernels
                                ? nat::validate_native_plan(
                                          nat::heuristic_plan(one, u, hp), one)
                                : nullptr;
                        report.check(why_one == nullptr, [&] {
                            return rules[r].setting + " " + describe(one)
                                    + ": illegal plan: " + why_one;
                        });
                    }
    }
    report.finish();
    for (size_t r = 0; r < std::size(rules); ++r)
        EXPECT_GT(changed[r], 0u) << rules[r].setting << " changed no plan";
}

// The group batches of an s4 tile in a plan of at most decode_m rows take
// the int4_l1_pct share of L1 (its later row calls read each batch again):
// by default half of L1 where the weight fits the L2 of the plan's threads
// and the source rows of its row block fit one L2 (both hold here up to
// M 128), else 125 percent of it; above decode_m rows and at int4_l1_pct=0 the
// int8_l2_pct share of L2, as a per-group s8 tile does at any M; a slab of
// K 32768 then takes batches that fit that share.  A set int4_l1_pct holds
// whatever the weight.
TEST(NativeTuningRules, Int4GroupBatchesInL1) {
    const nat::heuristic_params_t built_in;
    ASSERT_EQ(built_in.int4_l1_pct, -1);
    ASSERT_EQ(built_in.decode_m, 128);
    // 32 threads hold 32 MiB of L2: a weight of 4096 x 4096 s4 (8 MiB) fits
    // it, one of 5120 x 32768 (80 MiB) does not, nor does 4096 x 4096 the
    // 4 MiB of 4 threads.
    const struct {
        int N, K, threads, pct;
    } cases[] = {
            {4096, 4096, 32, 50}, {5120, 32768, 32, 125}, {4096, 4096, 4, 125}};
    for (const nat::UarchParams &u : {caches(32, 8), caches(128, 32)})
        for (const auto &c : cases) {
            const int64_t l1 = static_cast<int64_t>(u.l1d_bytes) * c.pct / 100;
            const int64_t pinned = static_cast<int64_t>(u.l1d_bytes) * 75 / 100;
            const int64_t l2 = static_cast<int64_t>(u.l2_bytes) * 50 / 100;
            for (const int M : {1, 32, 128, 129, 1024}) {
                const std::string what = std::to_string(c.N) + "x"
                        + std::to_string(c.K) + " threads "
                        + std::to_string(c.threads) + " M " + std::to_string(M);
                const nat::native_problem_t s4
                        = int4_problem(M, c.N, c.K, c.threads, 128);
                nat::native_problem_t s8 = problem(M, c.N, c.K,
                        nat::kNativeLayoutPackNr, c.threads, false);
                s8.int8_scheme = nat::native_int8_scheme_t::per_group;
                const nat::native_plan_t q
                        = nat::heuristic_plan(s4, u, built_in);
                ASSERT_EQ(q.threads, c.threads) << what;
                EXPECT_EQ(q.int8_pass_budget, M <= 128 ? l1 : l2) << what;
                EXPECT_EQ(nat::heuristic_plan(s8, u, built_in).int8_pass_budget,
                        l2)
                        << what;
                EXPECT_EQ(nat::heuristic_plan(s4, u, params("int4_l1_pct=0"))
                                  .int8_pass_budget,
                        l2)
                        << what;
                EXPECT_EQ(nat::heuristic_plan(s4, u, params("int4_l1_pct=75"))
                                  .int8_pass_budget,
                        M <= 128 ? pinned : l2)
                        << what;
            }
        }
}

// The bounds of the built-in share.  4096 x 14336 s4 (28 MiB) fits the
// 32 MiB of L2 of 32 threads; its batches take half of L1 while the row
// block's M x 14336 bytes of source rows fit one 1 MiB L2 (M at most 73),
// 125 percent from M 74.  The L2 caches are those of the threads' cores:
// 64 threads of 32 physical cores (SMT siblings) hold 32 MiB, which
// 28672 x 4096 (56 MiB) exceeds; 32 threads of 16 cores hold 16 MiB, which
// 4096 x 4096 (8 MiB) fits and 4096 x 14336 does not; unknown cores count
// one L2 a thread.
TEST(NativeTuningRules, Int4GroupBatchesBoundedBySourceRowsAndCores) {
    const nat::heuristic_params_t built_in;
    const auto budget = [&](nat::UarchParams u, int cores, int M, int N, int K,
                                int threads) {
        u.physical_cores = cores;
        const nat::native_plan_t q = nat::heuristic_plan(
                int4_problem(M, N, K, threads, 128), u, built_in);
        EXPECT_EQ(q.threads, threads) << M << " " << N << "x" << K;
        EXPECT_EQ(q.mb, M) << M << " " << N << "x" << K;
        return q.int8_pass_budget;
    };
    for (const nat::UarchParams &u : {caches(32, 8), caches(128, 32)}) {
        const int64_t half = static_cast<int64_t>(u.l1d_bytes) * 50 / 100;
        const int64_t stream = static_cast<int64_t>(u.l1d_bytes) * 125 / 100;
        for (const int M : {7, 32, 64, 73})
            EXPECT_EQ(budget(u, 0, M, 4096, 14336, 32), half) << M;
        for (const int M : {74, 96, 128})
            EXPECT_EQ(budget(u, 0, M, 4096, 14336, 32), stream) << M;
        for (const int M : {7, 128}) {
            EXPECT_EQ(budget(u, 0, M, 4096, 4096, 32), half) << M;
            EXPECT_EQ(budget(u, 0, M, 28672, 4096, 64), half) << M;
            EXPECT_EQ(budget(u, 64, M, 28672, 4096, 64), half) << M;
            EXPECT_EQ(budget(u, 32, M, 28672, 4096, 64), stream) << M;
            EXPECT_EQ(budget(u, 16, M, 4096, 4096, 32), half) << M;
            EXPECT_EQ(budget(u, 16, M, 4096, 14336, 32), stream) << M;
            EXPECT_EQ(
                    budget(u, 32, M, 4096, 14336, 32), M <= 73 ? half : stream)
                    << M;
        }
    }
}

TEST(NativeTuningRules, WorkedExamples) {
    // The makespan rule weighs BF16 widths by the kernels of this CPU.
    if (!has_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    // 8 cores per 32 MiB of L3, and 32 cores per 128 MiB.
    const nat::UarchParams domain8 = caches(32, 8), domain32 = caches(128, 32);
    const nat::heuristic_params_t none
            = params(native_brgemm_test::kNativeRulesOff);
    const nat::heuristic_params_t rules = nat::heuristic_params_t {};
    const auto plan
            = [](const nat::UarchParams &u, const nat::heuristic_params_t &hp,
                      int M, int N, int K, bool bf16) {
        return nat::heuristic_plan(
                problem(M, N, K, nat::kNativeLayoutPackNr, 32, bf16), u, hp);
    };
    // L1 pass: M=32 takes three BF16 row calls per pass of a 32 KiB slab,
    // eight equal passes over K=4096 (every rule off: one pass).  INT8 takes
    // equal passes in whole quads.
    EXPECT_EQ(plan(domain8, none, 32, 4096, 4096, true).kc, 4096);
    EXPECT_EQ(plan(domain8, rules, 32, 4096, 4096, true).kc, 512);
    EXPECT_EQ(
            plan(domain8, params("l1_pass=0"), 32, 4096, 4096, true).kc, 4096);
    EXPECT_EQ(plan(domain8, none, 32, 4096, 4096, false).int8_pass_quads, 0);
    const int quads
            = plan(domain8, rules, 32, 4096, 4096, false).int8_pass_quads;
    ASSERT_GT(quads, 0);
    EXPECT_LT(quads, 1024);
    const int passes = (1024 + quads - 1) / quads;
    EXPECT_EQ(quads, (1024 + passes - 1) / passes) << "equal passes";
    // One call's rows per tile: no pass.
    EXPECT_EQ(plan(domain8, rules, 8, 4096, 4096, true).kc, 4096);
    EXPECT_EQ(plan(domain8, rules, 8, 4096, 4096, false).int8_pass_quads, 0);
    // Makespan: M=128 at N=1536 takes two row blocks that fill three rounds
    // of the team instead of one block of 128 rows.
    EXPECT_EQ(plan(domain8, none, 128, 1536, 4096, true).mb, 128);
    EXPECT_EQ(plan(domain8, rules, 128, 1536, 4096, true).mb, 72);
    // It is never narrower than the narrowing's width, and its plans are
    // legal.
    struct shape_t {
        int M, N, K;
    };
    for (const shape_t &s : {shape_t {1, 1536, 4096}, shape_t {128, 256, 1536},
                 shape_t {1031, 96, 5120}, shape_t {1031, 256, 1536}})
        for (const bool bf16 : {true, false}) {
            const nat::native_plan_t l
                    = plan(domain8, none, s.M, s.N, s.K, bf16);
            const nat::native_plan_t q
                    = plan(domain8, rules, s.M, s.N, s.K, bf16);
            const nat::native_problem_t p = problem(
                    s.M, s.N, s.K, nat::kNativeLayoutPackNr, 32, bf16);
            EXPECT_GE(q.nr, l.nr) << describe(p);
            EXPECT_EQ(nat::validate_native_plan(q, p), nullptr) << describe(p);
        }
    // Traffic: the source of M=4096, K=4096 overflows 32 MiB, so the threads
    // of each 8-core domain take the same rows; at M=3077 it fits, and a
    // 128 MiB domain of 32 cores holds it.
    EXPECT_EQ(plan(domain8, none, 4096, 28672, 4096, true).col_groups, 0);
    EXPECT_EQ(plan(domain8, rules, 4096, 28672, 4096, true).col_groups, 8);
    EXPECT_EQ(plan(domain8, rules, 3077, 28672, 4096, true).col_groups, 0);
    EXPECT_EQ(plan(domain32, rules, 4096, 28672, 4096, true).col_groups, 0);
    EXPECT_EQ(plan(domain8, params("traffic_groups=0"), 4096, 28672, 4096, true)
                      .col_groups,
            0);
    // Aliasing rows: above decode_m, BF16 source rows of K a multiple of 2048
    // take 8 rows per call, those of a transposed source too (its copy has
    // rows K apart); other K, M up to decode_m and INT8 keep their rows per
    // call.
    EXPECT_EQ(plan(domain8, none, 4096, 4096, 4096, true).mr, 12);
    EXPECT_EQ(plan(domain8, rules, 4096, 4096, 4096, true).mr, 8);
    EXPECT_EQ(plan(domain8, rules, 1031, 4096, 14336, true).mr, 8);
    EXPECT_EQ(plan(domain8, rules, 4096, 4096, 6656, true).mr, 12);
    EXPECT_EQ(plan(domain8, rules, 128, 4096, 4096, true).mr, 12);
    // The cost model takes 8 rows per call above decode_m at any K where it
    // prices taller calls (32-thread L3 domains); cost_model=0 keeps 12.
    EXPECT_EQ(plan(domain32, rules, 4096, 4096, 6656, true).mr, 8);
    EXPECT_EQ(plan(domain32, params("cost_model=0"), 4096, 4096, 6656, true).mr,
            12);
    EXPECT_EQ(plan(domain8, rules, 4096, 4096, 4096, false).mr,
            plan(domain8, none, 4096, 4096, 4096, false).mr);
    EXPECT_EQ(plan(domain8, params("bf16_alias=0"), 4096, 4096, 4096, true).mr,
            12);
    EXPECT_EQ(
            plan(domain8, params("bf16_alias_mr=6"), 4096, 4096, 4096, true).mr,
            6);
    // The rule tests the block's rows: a pinned block of 12 or 9 rows takes
    // two calls of at most 8 rows, one of 8 rows keeps its call.
    EXPECT_EQ(plan(domain8, params("mb=12"), 4096, 4096, 4096, true).mr, 8);
    EXPECT_EQ(plan(domain8, params("mb=9"), 4096, 4096, 4096, true).mr, 8);
    EXPECT_EQ(plan(domain8, params("mb=8"), 4096, 4096, 4096, true).mr,
            plan(domain8, params("mb=8,bf16_alias=0"), 4096, 4096, 4096, true)
                    .mr);
    nat::native_problem_t ta
            = problem(4096, 4096, 4096, nat::kNativeLayoutPackNr, 32, true);
    ta.trans_a = true;
    EXPECT_EQ(nat::heuristic_plan(ta, domain8, rules).mr, 8);
    EXPECT_EQ(nat::validate_native_plan(
                      nat::heuristic_plan(ta, domain8, rules), ta),
            nullptr);
    // Narrow N: 32 threads over 8-core domains are 4 domains.  From
    // bf16_narrow_n_min_m rows a BF16 slice of fewer than 8 32-column tiles
    // takes the widest tiles of several slabs that are fewer than 4 and whose
    // plan, by these rules at that width, gives the busiest worker at most
    // min_gain_pct percent more work: 64 columns at N=64 and N=192, 128 at
    // N=128, and 64 at N=128, M=2048, where 128 columns leave threads idle.
    EXPECT_EQ(plan(domain8, none, 4096, 64, 2560, true).nr, 32);
    const nat::native_plan_t n64 = plan(domain8, rules, 4096, 64, 2560, true);
    EXPECT_EQ(n64.nr, 64);
    EXPECT_EQ(n64.n_tiles, 1);
    EXPECT_TRUE(same_plan(
            n64, plan(domain8, params("nr=64"), 4096, 64, 2560, true)));
    EXPECT_EQ(plan(domain8, rules, 4096, 192, 2560, true).nr, 64);
    // The 128-column slab exceeds the bf16_l2_pct share of L2: the wide plan
    // takes K passes and the blocks schedule, and is weighed as its slice.
    const nat::native_plan_t n128 = plan(domain8, rules, 4096, 128, 2560, true);
    EXPECT_EQ(n128.nr, 128);
    EXPECT_EQ(n128.unit_tiles, 1);
    EXPECT_LT(n128.kc, 2560);
    EXPECT_EQ(plan(domain8, rules, 2048, 128, 4096, true).nr, 64);
    // A busier worker at every wider width (at M=1031 they leave threads
    // idle), two 32-column tiles per domain (N=256), wider tiles no fewer
    // than the domains, fewer rows, a 32-column plan of row groups (the
    // source of M=4096, K=4096 overflows the domain's L3), one domain for
    // the team, the switch off and INT8 keep 32 columns.
    EXPECT_EQ(plan(domain8, rules, 1031, 128, 4096, true).nr, 32);
    EXPECT_EQ(plan(domain8, rules, 2048, 256, 4096, true).nr, 32);
    EXPECT_EQ(plan(domain8, rules, 4096, 512, 2560, true).nr, 32);
    EXPECT_EQ(plan(domain8, rules, 768, 128, 4096, true).nr, 32);
    EXPECT_EQ(plan(domain8, rules, 4096, 64, 4096, true).nr, 32);
    EXPECT_EQ(plan(domain8, rules, 4096, 64, 4096, true).col_groups, 2);
    EXPECT_EQ(plan(domain32, rules, 4096, 64, 2560, true).nr, 32);
    EXPECT_EQ(plan(domain8, params("bf16_narrow_n=0"), 4096, 64, 2560, true).nr,
            32);
    EXPECT_EQ(plan(domain8, params("bf16_narrow_n_min_m=4097"), 4096, 64, 2560,
                      true)
                      .nr,
            32);
    EXPECT_EQ(plan(domain8, rules, 4096, 64, 2560, false).nr, 32);
    // The domains are those of the team: 16 threads are 2, 64 threads 8.
    const auto team = [&](int M, int N, int K, int threads) {
        return nat::heuristic_plan(
                problem(M, N, K, nat::kNativeLayoutPackNr, threads, true),
                domain8, rules);
    };
    EXPECT_EQ(team(4096, 64, 2560, 16).nr, 64);
    EXPECT_EQ(team(4096, 128, 2560, 16).nr, 32);
    EXPECT_EQ(team(4096, 256, 2560, 64).nr, 128);
    EXPECT_EQ(team(4096, 512, 2560, 64).nr, 32);
    EXPECT_EQ(team(4096, 64, 2560, 8).nr, 32);
    // Those of the planned team: of 16 threads requested, a
    // min_tiles_per_thread pin plans 2 at M=1024, one domain, which keeps its
    // 32-column tiles (counting the requested team's two domains, the rule
    // would widen them to 64).
    const nat::native_plan_t pinned = nat::heuristic_plan(
            problem(1024, 64, 2560, nat::kNativeLayoutPackNr, 16, true),
            domain8, params("min_tiles_per_thread=6"));
    EXPECT_LE(pinned.threads, 8);
    EXPECT_EQ(pinned.nr, 32);
    for (const shape_t &s :
            {shape_t {4096, 64, 2560}, shape_t {4096, 192, 2560},
                    shape_t {4096, 128, 2560}, shape_t {2048, 128, 4096}}) {
        const nat::native_problem_t p
                = problem(s.M, s.N, s.K, nat::kNativeLayoutPackNr, 32, true);
        EXPECT_EQ(nat::validate_native_plan(
                          nat::heuristic_plan(p, domain8, rules), p),
                nullptr)
                << describe(p);
    }
}

// A single row block of more than l1_pass_max_mb rows, up to decode_m, takes
// K passes by the weight's fit; the 32 threads span four of the test
// machine's L3 domains.  A weight within three quarters of a domain takes the
// L1 pass of the l1_pct share (BF16 at M 128: that share of L1d over the 32
// columns and 12 rows of a call gives eight equal passes of K 4096); one
// within the team's L3 passes whose block of source rows takes a quarter of
// L2 (1024 K steps for 128 rows of BF16); a larger one passes of a 32-column
// slab of an eighth of L2 (2048 steps of BF16, 4096 of s8).  INT8 passes are
// in quads.  Several row blocks, more rows than decode_m and l1_pass=0 keep
// all of K; with l1_pass_max_mb=128 the block takes the l1_pct share whatever
// the weight.  Two thirds of that L1d hold fewer than 192 words of a
// 32-column tile's pass, of BF16 (pairs) and of s8 (quads): the block keeps
// all of K, while a weight past three quarters of a domain keeps its passes.
// So does a K whose equal passes come below 192 words (s8 K 1152: two of 144
// quads; BF16 K 576: two of 144 words); equal passes of 192 words stay.
TEST(NativeTuningRules, SingleRowBlockPasses) {
    if (!has_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    const nat::UarchParams domain8 = caches(32, 8);
    const nat::heuristic_params_t rules {};
    ASSERT_EQ(rules.l1_pass_max_mb, 64);
    ASSERT_EQ(rules.decode_m, 128);
    const auto plan = [&](const nat::heuristic_params_t &hp, int M, int N,
                              int K, bool bf16) {
        return nat::heuristic_plan(
                problem(M, N, K, nat::kNativeLayoutPackNr, 32, bf16), domain8,
                hp);
    };
    EXPECT_EQ(plan(rules, 128, 2880, 4096, true).kc, 512);
    EXPECT_EQ(plan(rules, 128, 4096, 4096, true).kc, 1024);
    EXPECT_EQ(plan(rules, 128, 5120, 32768, true).kc, 2048);
    EXPECT_EQ(plan(rules, 128, 16384, 5120, true).kc, 1708);
    EXPECT_EQ(plan(rules, 96, 4096, 4096, true).kc, 1024);
    EXPECT_EQ(plan(rules, 128, 4096, 4096, false).int8_pass_quads, 256);
    EXPECT_EQ(plan(rules, 128, 10240, 4096, false).int8_pass_quads, 512);
    EXPECT_EQ(plan(rules, 128, 5120, 32768, false).int8_pass_quads, 1024);
    const nat::native_plan_t two = plan(rules, 128, 1536, 6144, true);
    EXPECT_EQ(two.m_blocks, 2);
    EXPECT_EQ(two.kc, 6144);
    EXPECT_EQ(plan(rules, 160, 4096, 4096, true).kc, 4096);
    EXPECT_EQ(plan(rules, 160, 4096, 4096, false).int8_pass_quads, 0);
    EXPECT_EQ(plan(params("l1_pass=0"), 128, 4096, 4096, true).kc, 4096);
    EXPECT_EQ(
            plan(params("l1_pass_max_mb=128"), 128, 4096, 4096, true).kc, 512);
    EXPECT_EQ(plan(params("l1_pass_max_mb=128"), 128, 5120, 32768, true).kc,
            plan(params("l1_pass_max_mb=128"), 128, 2880, 32768, true).kc);
    nat::UarchParams l1_small = domain8;
    l1_small.l1d_bytes = domain8.l1d_bytes * 2 / 3;
    const auto small_l1 = [&](int M, int N, int K, bool bf16) {
        return nat::heuristic_plan(
                problem(M, N, K, nat::kNativeLayoutPackNr, 32, bf16), l1_small,
                rules);
    };
    EXPECT_EQ(small_l1(128, 2880, 4096, true).kc, 4096);
    EXPECT_EQ(small_l1(128, 4096, 4096, false).int8_pass_quads, 0);
    EXPECT_EQ(small_l1(128, 4096, 4096, true).kc, 1024);
    EXPECT_EQ(plan(rules, 128, 4096, 1152, false).int8_pass_quads, 0);
    EXPECT_EQ(plan(rules, 128, 4096, 1536, false).int8_pass_quads, 192);
    EXPECT_EQ(plan(rules, 128, 4096, 576, true).kc, 576);
    EXPECT_EQ(plan(rules, 128, 4096, 768, true).kc, 384);
    // An INT8 tile takes these passes only where an L3 domain holds at most
    // 16 cores: in one domain of four of domain8's, s8 keeps all of K (its
    // decode block keeps its pass) while BF16 takes the L1 pass.
    nat::UarchParams wide = domain8;
    wide.l3_domain_bytes *= 4;
    wide.l3_domain_cores *= 4;
    const auto on_wide = [&](int M, int N, int K, bool bf16) {
        return nat::heuristic_plan(
                problem(M, N, K, nat::kNativeLayoutPackNr, 32, bf16), wide,
                rules);
    };
    EXPECT_EQ(on_wide(128, 4096, 4096, false).int8_pass_quads, 0);
    EXPECT_EQ(on_wide(96, 2560, 1536, false).int8_pass_quads, 0);
    EXPECT_EQ(on_wide(64, 4096, 4096, false).int8_pass_quads, 205);
    EXPECT_EQ(on_wide(128, 4096, 4096, true).kc, 586);
    for (const int M : {65, 96, 128})
        for (const bool bf16 : {true, false})
            for (const shape_t &s : {shape_t {2880, 4096}, shape_t {4096, 4096},
                         shape_t {5120, 32768}}) {
                const nat::native_problem_t p = problem(
                        M, s.N, s.K, nat::kNativeLayoutPackNr, 32, bf16);
                EXPECT_EQ(nat::validate_native_plan(
                                  nat::heuristic_plan(p, domain8, rules), p),
                        nullptr)
                        << describe(p);
            }
}

// A decode block (a single row block of more than two calls of the tallest
// kernels, up to l1_pass_max_mb rows) takes the L1 pass's share of L1 by the
// weight against the L2 of the plan's threads (the L2s of its 32 threads,
// one a physical core) at l1_pct -1: 75 percent where an s8 weight fits it
// or a BF16 one fits an eighth of it and the tile's B slab over all of K
// exceeds L1 (BF16 1024 x 1536 at M 32: four equal passes of 384); 150
// percent where a BF16 weight does not fit it in a block of more than four
// calls, in an L3 domain of at most 16 cores (6144 x 4096 at M 64: five
// passes of 820).  100 percent otherwise (6144 x 4096 at M 32), in a block
// of two calls, for an F16 or FP32 weight (one K value a word), and
// wherever l1_pct is set.
TEST(NativeTuningRules, DecodePassShares) {
    if (!has_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    const nat::UarchParams domain8 = caches(32, 8);
    // One L3 domain of four of domain8's: more than 16 cores.
    nat::UarchParams domain32 = domain8;
    domain32.l3_domain_bytes *= 4;
    domain32.l3_domain_cores *= 4;
    const nat::heuristic_params_t rules {};
    ASSERT_EQ(rules.l1_pct, -1);
    const auto plan_on
            = [&](const nat::UarchParams &u, const nat::heuristic_params_t &hp,
                      int M, int N, int K, bool bf16) {
        return nat::heuristic_plan(
                problem(M, N, K, nat::kNativeLayoutPackNr, 32, bf16), u, hp);
    };
    const auto plan = [&](const nat::heuristic_params_t &hp, int M, int N,
                              int K, bool bf16) {
        return plan_on(domain8, hp, M, N, K, bf16);
    };
    EXPECT_EQ(plan(rules, 32, 1024, 1536, true).kc, 384);
    EXPECT_EQ(plan(rules, 32, 4096, 4096, true).kc, 512);
    EXPECT_EQ(plan(rules, 32, 2560, 1536, true).kc, 512);
    // 40 column tiles on 32 threads: the busiest thread's two tiles exceed
    // an eighth of L2, the average share does not.
    EXPECT_EQ(plan(rules, 32, 1280, 1536, true).kc, 512);
    EXPECT_EQ(plan(params("l1_pct=75"), 32, 1280, 1536, true).kc, 384);
    // SMT siblings share their core's L2: 32 threads of 16 physical cores
    // weigh the weight against 16 L2s, as 16 threads do; of 32 cores, or of
    // an unknown count, against 32.
    nat::UarchParams smt = domain8;
    smt.physical_cores = 16;
    EXPECT_EQ(plan_on(smt, rules, 64, 4096, 4096, true).kc, 820);
    EXPECT_EQ(plan_on(smt, rules, 32, 6144, 4096, false).int8_pass_quads, 256);
    smt.physical_cores = 32;
    EXPECT_EQ(plan_on(smt, rules, 64, 4096, 4096, true).kc, 512);
    EXPECT_EQ(plan_on(smt, rules, 32, 6144, 4096, false).int8_pass_quads, 205);
    EXPECT_EQ(plan(rules, 64, 4096, 4096, true).kc, 512);
    EXPECT_EQ(plan_on(domain32, rules, 64, 6144, 4096, true).kc,
            plan_on(domain32, params("l1_pct=100"), 64, 6144, 4096, true).kc);
    EXPECT_NE(plan_on(domain32, params("l1_pct=150"), 64, 6144, 4096, true).kc,
            plan_on(domain32, params("l1_pct=100"), 64, 6144, 4096, true).kc);
    EXPECT_EQ(plan(rules, 32, 6144, 4096, true).kc, 512);
    EXPECT_EQ(plan(rules, 64, 6144, 4096, true).kc, 820);
    EXPECT_EQ(plan(rules, 64, 28672, 4096, true).kc, 820);
    EXPECT_EQ(plan(rules, 48, 28672, 4096, true).kc, 512);
    EXPECT_EQ(plan(rules, 32, 4096, 4096, false).int8_pass_quads, 205);
    EXPECT_EQ(plan(rules, 32, 10240, 4096, false).int8_pass_quads, 256);
    EXPECT_EQ(plan(rules, 32, 1024, 1024, false).int8_pass_quads, 0);
    EXPECT_EQ(plan(rules, 24, 1024, 1536, true).kc, 512);
    EXPECT_EQ(plan(params("l1_pct=100"), 32, 1024, 1536, true).kc, 512);
    EXPECT_EQ(plan(params("l1_pct=100"), 64, 6144, 4096, true).kc, 512);
    EXPECT_EQ(plan(params("l1_pct=100"), 32, 4096, 4096, false).int8_pass_quads,
            256);
    for (const data_type_t dt : {data_type_t::f16, data_type_t::f32})
        for (const shape_t &s : {shape_t {1024, 1536}, shape_t {6144, 4096}})
            for (const int M : {32, 64}) {
                nat::native_problem_t p = problem(
                        M, s.N, s.K, nat::kNativeLayoutPackNr, 32, true);
                p.src_dt = p.wei_dt = p.dst_dt = dt;
                EXPECT_EQ(nat::heuristic_plan(p, domain8, rules).kc,
                        nat::heuristic_plan(p, domain8, params("l1_pct=100"))
                                .kc)
                        << describe(p);
            }
    for (const int M : {25, 32, 48, 64})
        for (const bool bf16 : {true, false})
            for (const shape_t &s : {shape_t {1024, 1536}, shape_t {6144, 4096},
                         shape_t {5120, 32768}}) {
                const nat::native_problem_t p = problem(
                        M, s.N, s.K, nat::kNativeLayoutPackNr, 32, bf16);
                EXPECT_EQ(nat::validate_native_plan(
                                  nat::heuristic_plan(p, domain8, rules), p),
                        nullptr)
                        << describe(p);
            }
}

// The rules bounded by an L3 domain of at most 16 cores (the decode block's
// 150 percent share, an INT8 single row block's K passes by the weight's fit,
// the INT8 blocks schedule from 2048 rows) read the cores of the machine's L3
// domain, whatever domain_threads sets; domain_threads deals the work.  8 or
// 16 threads to a domain on one domain of four of domain8's keep that
// domain's plans: BF16 6144 x 4096 at M 64 the share of l1_pct=100, s8 at M
// 96 and 128 all of K, s8 4096 x 4096 on 64 threads the slice at M 2048 and
// blocks of the set domain_threads at M 3072.  32 threads to a domain on
// domain8 keep domain8's: the share of l1_pct=150, s8 passes at M 128, and
// blocks at M 2048.
TEST(NativeTuningRules, DomainBoundsReadTheMachineCores) {
    if (!has_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    const nat::UarchParams domain8 = caches(32, 8);
    nat::UarchParams domain32 = domain8;
    domain32.l3_domain_bytes *= 4;
    domain32.l3_domain_cores *= 4;
    const auto plan = [&](const nat::UarchParams &u, const std::string &text,
                              int M, int N, int K, int threads, bool bf16) {
        return nat::heuristic_plan(
                problem(M, N, K, nat::kNativeLayoutPackNr, threads, bf16), u,
                params(text));
    };
    for (const int dt : {8, 16}) {
        const std::string pin = "domain_threads=" + std::to_string(dt);
        SCOPED_TRACE(pin);
        EXPECT_EQ(plan(domain32, pin, 64, 6144, 4096, 32, true).kc,
                plan(domain32, pin + ",l1_pct=100", 64, 6144, 4096, 32, true)
                        .kc);
        EXPECT_EQ(
                plan(domain32, pin, 128, 4096, 4096, 32, false).int8_pass_quads,
                0);
        EXPECT_EQ(
                plan(domain32, pin, 96, 2560, 1536, 32, false).int8_pass_quads,
                0);
        EXPECT_EQ(
                plan(domain32, pin, 2048, 4096, 4096, 64, false).unit_tiles, 0);
        const nat::native_plan_t b
                = plan(domain32, pin, 3072, 4096, 4096, 64, false);
        EXPECT_EQ(b.unit_tiles, 1);
        EXPECT_EQ(b.domain_threads, dt);
    }
    const std::string wide = "domain_threads=32";
    EXPECT_EQ(plan(domain8, wide, 64, 6144, 4096, 32, true).kc,
            plan(domain8, wide + ",l1_pct=150", 64, 6144, 4096, 32, true).kc);
    EXPECT_EQ(plan(domain8, wide, 128, 4096, 4096, 32, false).int8_pass_quads,
            256);
    const nat::native_plan_t b
            = plan(domain8, wide, 2048, 4096, 4096, 64, false);
    EXPECT_EQ(b.unit_tiles, 1);
    EXPECT_EQ(b.domain_threads, 32);
}

} // namespace
