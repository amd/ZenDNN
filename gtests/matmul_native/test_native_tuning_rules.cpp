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

std::string describe(const nat::native_problem_t &p) {
    std::ostringstream os;
    os << (p.bf16() ? "bf16" : "int8") << " M=" << p.M << " N=" << p.N
       << " K=" << p.K << " pack_nr=" << p.pack_nr << " threads=" << p.threads;
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
    u.l3_bytes_per_ccd = l3_mib << 20;
    u.ccx_cores = 1;
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
};

/// The fields in which a and b differ, as a mask; other fields that differ
/// set every bit above kScratch.
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
    if (a.row_unit != b.row_unit || a.traversal != b.traversal
            || a.int8_rows_nv1 != b.int8_rows_nv1
            || a.int8_rows_nv2 != b.int8_rows_nv2
            || a.int8_cw_mask != b.int8_cw_mask
            || a.int8_pass_budget != b.int8_pass_budget
            || a.int8_direct_store != b.int8_direct_store
            || a.int8_bias_in_dequant != b.int8_bias_in_dequant
            || a.copy_block != b.copy_block || a.copy_par_min != b.copy_par_min
            || a.sources != b.sources)
        m |= ~0u << 4;
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
    // The narrow-N rule's wide plan validates only where the registry offers
    // BF16 kernels; the other rules change INT8 plans on any CPU.
    const bool bf16_kernels = nat::detect_uarch().avx512bf16;
    for (size_t r = 0; r < std::size(rules); ++r)
        if (bf16_kernels || rules[r].setting != only("bf16_narrow_n")) {
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
    EXPECT_FALSE(blocks(plan(domain8, "", 1024, 4096, 4096, false)));
    EXPECT_TRUE(blocks(
            plan(domain8, "int8_blocks_min_m=1024", 1024, 4096, 4096, false)));
    EXPECT_FALSE(blocks(plan(domain8, "", 4096, 4096, 14336, false)));
    // A pinned schedule or col_groups keeps its own plan.
    EXPECT_EQ(
            plan(domain8, "schedule=slice", 4096, 4096, 14336, true).unit_tiles,
            0);
    EXPECT_EQ(plan(domain8, "col_groups=8", 4096, 4096, 14336, true).col_groups,
            8);
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

} // namespace
