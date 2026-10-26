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

// The v2 rules of the native BRGEMM plan (preset=v2): the makespan grid with
// its row-group fallback (never narrower than the legacy tile), the L1-sized
// K pass and col_groups by memory traffic.  Every v2 plan must pass
// validate_native_plan over dense linear and edge shapes, M from 1 to 8192,
// 1 to 256 threads, both dtypes and L3 domains of 1 to 64 cores, and a
// planner over the override preset=v2 must return it.  Under preset=v2 every
// statically legal pin must give a legal plan through the planner, and a pin
// the legacy preset keeps must stay legal in heuristic_plan.  Each rule sits
// behind its own switch: switched on alone under the legacy preset it changes
// only the fields it decides, and preset=v2 is the three together.  Worked
// examples show one plan of each rule on two cache hierarchies, and the
// properties each rule keeps.

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
    kGroups = 4, ///< col_groups
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
    if (a.col_groups != b.col_groups) m |= kGroups;
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

TEST(NativeTuningPlanV2, LegalForEveryDomainAndTeam) {
    if (!has_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    const nat::heuristic_params_t v2 = params("preset=v2");
    report_t report;
    for (int domain = 1; domain <= 64; domain += domain < 8 ? 1 : 7)
        for (const int l3_mib : {8, 32, 128}) {
            const nat::UarchParams u = caches(l3_mib, domain);
            for (int threads = 1; threads <= 256;
                    threads += threads < 40 ? 1 : 23)
                for (const shape_t &s : kShapes)
                    for (const int M : kMs)
                        for (const bool bf16 : {true, false}) {
                            const nat::native_problem_t p = problem(M, s.N, s.K,
                                    nat::kNativeLayoutPackNr, threads, bf16);
                            const nat::native_plan_t q
                                    = nat::heuristic_plan(p, u, v2);
                            const char *why = nat::validate_native_plan(q, p);
                            report.check(why == nullptr, [&] {
                                return "domain " + std::to_string(domain)
                                        + " L3 " + std::to_string(l3_mib)
                                        + " MiB " + describe(p)
                                        + ": validate_native_plan rejects it: "
                                        + why;
                            });
                        }
        }
    report.finish();
}

TEST(NativeTuningPlanV2, PlannerReturnsTheV2Plan) {
    if (!has_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    const nat::UarchParams &u = nat::native_plan_uarch();
    const nat::heuristic_params_t v2 = params("preset=v2");
    const nat::native_planner_t planner(
            u, nat::make_native_override_source("preset=v2"), nullptr, nullptr);
    report_t report;
    for (const int pack_nr : {32, 64})
        for (const int threads : {1, 8, 24, 32, 64, 128})
            for (const shape_t &s : kShapes)
                for (const int M : kMs)
                    for (const bool bf16 : {true, false}) {
                        if (s.N % pack_nr != 0) continue;
                        const nat::native_problem_t p
                                = problem(M, s.N, s.K, pack_nr, threads, bf16);
                        nat::native_plan_t want = nat::heuristic_plan(p, u, v2);
                        const nat::native_plan_t got = planner.plan(p);
                        want.sources = got.sources;
                        report.check(same_plan(got, want)
                                        && nat::native_plan_source_names(got)
                                                == "override",
                                [&] {
                            return describe(p)
                                    + ": the planner's plan is not the v2 plan";
                        });
                    }
    report.finish();
}

/// Every statically legal value of every key but the preset as "key=value",
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
            case nat::native_key_type_t::preset: break;
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
        for (int pack_nr : {32, 64})
            for (int N : {32, 64, 96, 128, 1536, 4096})
                for (int K : {1, 64, 4097})
                    for (int M : {1, 5, 6, 7, 11, 12, 13, 24, 25, 64, 100, 191,
                                 192, 193, 1031})
                        for (int threads : {1, 8, 32, 33, 256}) {
                            if (N < pack_nr || N % pack_nr != 0) continue;
                            v.push_back(
                                    problem(M, N, K, pack_nr, threads, bf16));
                        }
    return v;
}

TEST(NativeTuningPlanV2, TotalOverLegalPins) {
    if (!has_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    const nat::UarchParams &uarch = nat::native_plan_uarch();
    const std::vector<nat::native_problem_t> problems = edge_problems();
    report_t report;
    for (const std::string &setting : legal_settings()) {
        const std::string v2_text = "preset=v2," + setting;
        const nat::heuristic_params_t legacy
                = params("preset=legacy," + setting);
        const nat::heuristic_params_t v2 = params(v2_text);
        const std::unique_ptr<nat::native_planner_t> planner
                = nat::make_native_planner(v2_text.c_str(), nullptr);
        for (const nat::native_problem_t &p : problems) {
            const nat::native_plan_t q = nat::heuristic_plan(p, uarch, v2);
            const char *planned
                    = nat::validate_native_plan(planner->plan(p), p);
            report.check(planned == nullptr, [&] {
                return v2_text + " " + describe(p)
                        + ": the planner returns an illegal plan: " + planned;
            });
            if (nat::validate_native_plan(
                        nat::heuristic_plan(p, uarch, legacy), p)
                    != nullptr)
                continue;
            const char *why = nat::validate_native_plan(q, p);
            report.check(why == nullptr, [&] {
                return v2_text + " " + describe(p)
                        + ": legal under legacy, illegal under v2: " + why;
            });
        }
    }
    report.finish();
}

TEST(NativeTuningPlanV2, EachRuleChangesOnlyItsFields) {
    const struct {
        const char *setting;
        unsigned fields;
    } rules[] = {
            {"preset=legacy,makespan=1", kGrid | kPass | kGroups | kScratch},
            {"preset=legacy,l1_pass=1", kPass | kScratch},
            {"preset=legacy,traffic_groups=1", kGroups},
    };
    const nat::heuristic_params_t legacy = params("preset=legacy");
    const nat::heuristic_params_t v2 = params("preset=v2");
    const nat::heuristic_params_t all
            = params("makespan=1,l1_pass=1,traffic_groups=1");
    const nat::heuristic_params_t off
            = params("preset=v2,makespan=0,l1_pass=0,traffic_groups=0");
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
                                = nat::heuristic_plan(p, u, legacy);
                        for (size_t r = 0; r < std::size(rules); ++r) {
                            const unsigned diff = plan_fields(
                                    nat::heuristic_plan(
                                            p, u, params(rules[r].setting)),
                                    base);
                            changed[r] += diff != 0;
                            report.check((diff & ~rules[r].fields) == 0, [&] {
                                return std::string(rules[r].setting) + " "
                                        + describe(p)
                                        + " changes a field of another rule";
                            });
                        }
                        report.check(same_plan(nat::heuristic_plan(p, u, v2),
                                             nat::heuristic_plan(p, u, all)),
                                [&] {
                            return describe(p)
                                    + ": preset=v2 is not its three rules";
                        });
                        report.check(
                                same_plan(nat::heuristic_plan(p, u, off), base),
                                [&] {
                            return describe(p)
                                    + ": preset=v2 with its rules off is not "
                                      "the legacy plan";
                        });
                    }
    report.finish();
    for (size_t r = 0; r < std::size(rules); ++r)
        EXPECT_GT(changed[r], 0u) << rules[r].setting << " changed no plan";
}

TEST(NativeTuningPlanV2, WorkedExamples) {
    // The makespan rule weighs BF16 widths by the kernels of this CPU.
    if (!has_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    // 8 cores per 32 MiB of L3, and 32 cores per 128 MiB.
    const nat::UarchParams domain8 = caches(32, 8), domain32 = caches(128, 32);
    const nat::heuristic_params_t legacy = params("preset=legacy");
    const nat::heuristic_params_t v2 = params("preset=v2");
    const auto plan
            = [](const nat::UarchParams &u, const nat::heuristic_params_t &hp,
                      int M, int N, int K, bool bf16) {
        return nat::heuristic_plan(
                problem(M, N, K, nat::kNativeLayoutPackNr, 32, bf16), u, hp);
    };
    // L1 pass: M=32 takes three BF16 row calls per pass of a 32 KiB slab,
    // eight equal passes over K=4096 (legacy: one pass).  INT8 takes equal
    // passes in whole quads.
    EXPECT_EQ(plan(domain8, legacy, 32, 4096, 4096, true).kc, 4096);
    EXPECT_EQ(plan(domain8, v2, 32, 4096, 4096, true).kc, 512);
    EXPECT_EQ(plan(domain8, params("preset=v2,l1_pass=0"), 32, 4096, 4096, true)
                      .kc,
            4096);
    EXPECT_EQ(plan(domain8, legacy, 32, 4096, 4096, false).int8_pass_quads, 0);
    const int quads = plan(domain8, v2, 32, 4096, 4096, false).int8_pass_quads;
    ASSERT_GT(quads, 0);
    EXPECT_LT(quads, 1024);
    const int passes = (1024 + quads - 1) / quads;
    EXPECT_EQ(quads, (1024 + passes - 1) / passes) << "equal passes";
    // One call's rows per tile: no pass.
    EXPECT_EQ(plan(domain8, v2, 8, 4096, 4096, true).kc, 4096);
    EXPECT_EQ(plan(domain8, v2, 8, 4096, 4096, false).int8_pass_quads, 0);
    // Makespan: M=128 at N=1536 takes two row blocks that fill three rounds
    // of the team instead of one block of 128 rows.
    EXPECT_EQ(plan(domain8, legacy, 128, 1536, 4096, true).mb, 128);
    EXPECT_EQ(plan(domain8, v2, 128, 1536, 4096, true).mb, 72);
    // It is never narrower than the legacy width, and its plans are legal.
    struct shape_t {
        int M, N, K;
    };
    for (const shape_t &s : {shape_t {1, 1536, 4096}, shape_t {128, 256, 1536},
                 shape_t {1031, 96, 5120}, shape_t {1031, 256, 1536}})
        for (const bool bf16 : {true, false}) {
            const nat::native_plan_t l
                    = plan(domain8, legacy, s.M, s.N, s.K, bf16);
            const nat::native_plan_t q = plan(domain8, v2, s.M, s.N, s.K, bf16);
            const nat::native_problem_t p = problem(
                    s.M, s.N, s.K, nat::kNativeLayoutPackNr, 32, bf16);
            EXPECT_GE(q.nr, l.nr) << describe(p);
            EXPECT_EQ(nat::validate_native_plan(q, p), nullptr) << describe(p);
        }
    // Traffic: the source of M=4096, K=4096 overflows 32 MiB, so the threads
    // of each 8-core domain take the same rows; at M=3077 it fits, and a
    // 128 MiB domain of 32 cores holds it.
    EXPECT_EQ(plan(domain8, legacy, 4096, 28672, 4096, true).col_groups, 0);
    EXPECT_EQ(plan(domain8, v2, 4096, 28672, 4096, true).col_groups, 8);
    EXPECT_EQ(plan(domain8, v2, 3077, 28672, 4096, true).col_groups, 0);
    EXPECT_EQ(plan(domain32, v2, 4096, 28672, 4096, true).col_groups, 0);
    EXPECT_EQ(plan(domain8, params("preset=v2,traffic_groups=0"), 4096, 28672,
                      4096, true)
                      .col_groups,
            0);
}

} // namespace
