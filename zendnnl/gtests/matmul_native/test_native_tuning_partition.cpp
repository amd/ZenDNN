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

// Partition of the native BRGEMM.  For every strategy (slice in both tile
// orders, row groups), column range and team size, every tile of the range
// must go to exactly one worker, in whole tiles of at most mb rows; slice,
// and row groups over the full range, must run the units in the order of a
// reference copy of the schedules; a team other than the plan's must fall
// back to slice.  The CPU team must run every unit once with scratch of its
// own per thread, inline for a team of 1.

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <omp.h>
#include <gtest/gtest.h>

#include "lowoha_operators/common/omp_thread_control.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_partition.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_config.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_planner.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "lowoha_operators/matmul/matmul_native/native_layout_policy.hpp"

namespace {

namespace nat = zendnnl::lowoha::matmul::native;
using unit_t = nat::native_work_unit_t;
using zendnnl::error_handling::status_t;
using zendnnl::memory::data_type_t;

constexpr data_type_t kBf16 = data_type_t::bf16;

int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

std::string text(const unit_t &u) {
    std::ostringstream os;
    os << "{m0 " << u.m0 << " rows " << u.rows << " n0 " << u.n0 << "}";
    return os.str();
}

std::string text(const nat::native_plan_t &p) {
    std::ostringstream os;
    os << "nr " << p.nr << " mb " << p.mb << " m_blocks " << p.m_blocks
       << " n_tiles " << p.n_tiles << " threads " << p.threads << " col_groups "
       << p.col_groups << " row_unit " << p.row_unit
       << (p.traversal == nat::native_traversal_t::m_inner ? " m_inner"
                                                           : " n_inner");
    return os.str();
}

bool same_units(const std::vector<unit_t> &a, const std::vector<unit_t> &b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i].m0 != b[i].m0 || a[i].rows != b[i].rows || a[i].n0 != b[i].n0)
            return false;
    return true;
}

/// Rows [0, M) of column tiles c0 .. c1, each in exactly one unit; every
/// unit one whole tile inside them, of at most mb rows (the scratch holds mb
/// rows of one tile).
::testing::AssertionResult covers_once(const std::vector<unit_t> &units,
        const nat::native_plan_t &p, int M, int c0, int c1) {
    std::vector<uint8_t> hits(static_cast<size_t>(M) * (c1 - c0), 0);
    for (const unit_t &u : units) {
        if (u.rows < 1 || u.m0 < 0 || u.m0 + u.rows > M || u.rows > p.mb
                || u.n0 % p.nr != 0 || u.n0 < c0 * p.nr
                || u.n0 + p.nr > c1 * p.nr)
            return ::testing::AssertionFailure()
                    << "unit " << text(u) << " outside rows [0, " << M
                    << ") x tiles [" << c0 << ", " << c1 << ") of nr " << p.nr
                    << ", mb " << p.mb;
        const int t = u.n0 / p.nr;
        for (int r = u.m0; r < u.m0 + u.rows; ++r)
            if (++hits[static_cast<size_t>(t - c0) * M + r] > 1)
                return ::testing::AssertionFailure()
                        << "row " << r << " of tile " << t
                        << " in two units, the second " << text(u);
    }
    for (size_t i = 0; i < hits.size(); ++i)
        if (hits[i] == 0)
            return ::testing::AssertionFailure()
                    << "row " << i % M << " of tile " << c0 + i / M
                    << " in no unit";
    return ::testing::AssertionSuccess();
}

/// The units worker w of a team of workers runs over column tiles c0 .. c1,
/// by a reference copy of the schedules.  Slice: the worker's share of the
/// tile order, in which consecutive tiles walk the row blocks of one column
/// tile (m_inner) or the column tiles of one row block (n_inner).  Row groups
/// (full column range): col_groups column ranges times workers / col_groups
/// row ranges of whole row_unit units, adjacent workers in one row range.
std::vector<unit_t> reference_units(const nat::native_plan_t &p, int M, int c0,
        int c1, int w, int workers, bool row_groups) {
    std::vector<unit_t> v;
    auto add = [&](int m0, int rows, int tile) {
        unit_t u;
        u.m0 = m0;
        u.rows = rows;
        u.n0 = tile * p.nr;
        v.push_back(u);
    };
    if (!row_groups) {
        const bool m_inner = p.traversal == nat::native_traversal_t::m_inner;
        const int cols = c1 - c0;
        const int64_t tiles = static_cast<int64_t>(p.m_blocks) * cols;
        for (int64_t t = tiles * w / workers; t < tiles * (w + 1) / workers;
                ++t) {
            const int block
                    = static_cast<int>(m_inner ? t % p.m_blocks : t / cols);
            const int tile = c0
                    + static_cast<int>(m_inner ? t / p.m_blocks : t % cols);
            const int m0 = block * p.mb;
            add(m0, std::min(p.mb, M - m0), tile);
        }
        return v;
    }
    const int row_ranges = workers / p.col_groups;
    const int rg = w / p.col_groups;
    const int64_t units = ceil_div(M, p.row_unit);
    const int r0 = static_cast<int>(
            std::min<int64_t>(M, units * rg / row_ranges * p.row_unit));
    const int r1 = static_cast<int>(
            std::min<int64_t>(M, units * (rg + 1) / row_ranges * p.row_unit));
    const int cols = p.n_tiles / p.col_groups;
    const int first = (w % p.col_groups) * cols;
    for (int c = first; c < first + cols; ++c)
        for (int m0 = r0; m0 < r1; m0 += p.mb)
            add(m0, std::min(p.mb, r1 - m0), c);
    return v;
}

/// The per-problem rules the partition relies on (grid, team, column
/// groups, row unit).
bool partition_legal(
        const nat::native_plan_t &p, const nat::native_problem_t &pr) {
    return p.nr >= 16 && p.n_tiles == pr.N / p.nr && p.mb >= 1 && p.mb <= pr.M
            && p.m_blocks == ceil_div(pr.M, p.mb) && p.threads >= 1
            && p.threads <= pr.threads
            && (p.col_groups == 0
                    || (p.threads % p.col_groups == 0
                            && p.n_tiles % p.col_groups == 0))
            && p.row_unit >= 1;
}

nat::native_problem_t bf16_problem(int M, int N, int K, int threads) {
    nat::native_problem_t p;
    p.M = M;
    p.N = N;
    p.K = K;
    p.pack_nr = nat::kNativeLayoutPackNr;
    p.threads = threads;
    p.src_dt = kBf16;
    p.wei_dt = kBf16;
    p.dst_dt = kBf16;
    return p;
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

/// Grids and schedules the partition must serve, as config settings ("" is
/// the built-in plan).
const char *const kPlanVariants[] = {"", "traversal=n_inner", "schedule=slice",
        "schedule=row_groups,domain_threads=8",
        "schedule=row_groups,domain_threads=16,traversal=n_inner",
        "col_groups=1", "col_groups=2,row_unit=5",
        "col_groups=4,mb=7,row_unit=1", "mb=50,traversal=n_inner",
        "nr=16,schedule=row_groups,domain_threads=8,row_unit=24",
        "threads_cap=6,schedule=row_groups,domain_threads=6",
        "min_tiles_per_thread=4,col_groups=2"};

nat::heuristic_params_t params_of(const char *settings) {
    nat::native_config_t cfg;
    std::string why;
    EXPECT_TRUE(nat::parse_native_config(settings, cfg, why))
            << settings << ": " << why;
    return nat::apply_native_config(nat::heuristic_params_t {}, cfg);
}

std::vector<std::pair<int, int>> column_ranges(int n) {
    std::vector<std::pair<int, int>> v = {{0, n}, {0, 0}, {n, n}};
    for (int s : {1, n / 4, n / 2, n - 1}) {
        if (s <= 0 || s >= n) continue;
        v.push_back({0, n - s});
        v.push_back({n - s, n});
    }
    if (n > 2) v.push_back({1, n - 1});
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}

std::vector<int> team_sizes(int plan_threads, int64_t units) {
    std::vector<int> v = {1, 2, 3, plan_threads - 1, plan_threads,
            plan_threads + 1, 2 * plan_threads,
            static_cast<int>(std::min<int64_t>(units + 5, 4096))};
    v.erase(std::remove_if(v.begin(), v.end(), [](int t) { return t < 1; }),
            v.end());
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}

/// The partition of column tiles c0 .. c1 of p for a team of workers: its
/// kind, every tile once, the reference order (slice, and row groups over
/// the full range) and the unit count of slice.  Returns the strategy it
/// exercised, "" for an empty range.
std::string check_partition(const nat::native_plan_t &p, int M, int c0, int c1,
        int team, const std::string &where, report_t &report) {
    const nat::native_partition_t part(p, M, c0, c1);
    const bool full = c0 == 0 && c1 == p.n_tiles;
    const bool row_groups = p.col_groups > 0 && team == p.threads
            && (c1 - c0) % p.col_groups == 0;
    const nat::native_partition_kind_t want = row_groups
            ? nat::native_partition_kind_t::row_groups
            : nat::native_partition_kind_t::slice;
    const nat::native_partition_kind_t kind = part.kind(team);
    report.check(kind == want, [&] {
        return where + ": kind " + std::to_string(static_cast<int>(kind));
    });
    std::vector<unit_t> all;
    int out_of_order = -1;
    for (int w = 0; w < team; ++w) {
        std::vector<unit_t> mine;
        part.for_each(w, team, [&](const unit_t &u) { mine.push_back(u); });
        const bool ordered = (row_groups && !full)
                || same_units(mine,
                        reference_units(p, M, c0, c1, w, team, row_groups));
        if (!ordered && out_of_order < 0) out_of_order = w;
        all.insert(all.end(), mine.begin(), mine.end());
    }
    const ::testing::AssertionResult once = covers_once(all, p, M, c0, c1);
    report.check(static_cast<bool>(once),
            [&] { return where + ": " + once.message(); });
    report.check(out_of_order < 0, [&] {
        return where + ": worker " + std::to_string(out_of_order)
                + " does not run the reference schedule";
    });
    if (!row_groups) {
        const int64_t tiles = static_cast<int64_t>(p.m_blocks) * (c1 - c0);
        const int64_t visited = static_cast<int64_t>(all.size());
        report.check(part.units() == tiles && visited == tiles, [&] {
            return where + ": units() " + std::to_string(part.units())
                    + ", visited " + std::to_string(visited) + ", tiles "
                    + std::to_string(tiles);
        });
    }
    if (c1 == c0) return "";
    if (row_groups)
        return full ? "row groups, full range" : "row groups, part range";
    if (p.col_groups > 0) return "row-group plan, other team";
    return p.traversal == nat::native_traversal_t::m_inner ? "slice m_inner"
                                                           : "slice n_inner";
}

TEST(NativeTuningPartition, EveryTileOnceForEveryStrategyRangeAndTeam) {
    constexpr int kK = 64;
    const int shapes[][2] = {{1, 32}, {7, 96}, {12, 512}, {25, 1024},
            {192, 256}, {300, 4096}, {1031, 2048}};
    const nat::UarchParams &uarch = nat::detect_uarch();
    report_t report;
    std::set<std::string> exercised;
    for (const auto &s : shapes)
        for (int threads : {1, 3, 8, 32})
            for (const char *settings : kPlanVariants) {
                const nat::native_problem_t pr
                        = bf16_problem(s[0], s[1], kK, threads);
                const nat::native_plan_t p
                        = nat::heuristic_plan(pr, uarch, params_of(settings));
                if (!partition_legal(p, pr)) continue;
                for (const auto &range : column_ranges(p.n_tiles)) {
                    const int64_t tiles = static_cast<int64_t>(p.m_blocks)
                            * (range.second - range.first);
                    for (int team : team_sizes(p.threads, tiles)) {
                        std::ostringstream where;
                        where << "'" << settings << "' M " << s[0] << " N "
                              << s[1] << " threads " << threads << ", plan "
                              << text(p) << ", tiles [" << range.first << ", "
                              << range.second << "), team " << team;
                        exercised.insert(check_partition(p, s[0], range.first,
                                range.second, team, where.str(), report));
                    }
                }
            }
    report.finish();
    for (const char *kind :
            {"slice m_inner", "slice n_inner", "row groups, full range",
                    "row groups, part range", "row-group plan, other team"})
        EXPECT_EQ(exercised.count(kind), 1u) << kind << " never exercised";
}

/// Records the units a CPU team runs and the scratch each thread gets.
class team_log_t {
public:
    void add(const unit_t &u, float *scratch) {
        std::lock_guard<std::mutex> lock(mu_);
        units_.push_back(u);
        scratch_[std::this_thread::get_id()].insert(scratch);
        inline_ = inline_ && std::this_thread::get_id() == caller_
                && !omp_in_parallel();
    }
    const std::vector<unit_t> &units() const { return units_; }
    bool all_inline() const { return inline_; }
    /// No scratch buffer used by two threads.
    bool scratch_private() const {
        std::set<float *> seen;
        for (const auto &t : scratch_)
            for (float *s : t.second)
                if (!seen.insert(s).second) return false;
        return true;
    }

private:
    std::mutex mu_;
    std::vector<unit_t> units_;
    std::map<std::thread::id, std::set<float *>> scratch_;
    const std::thread::id caller_ = std::this_thread::get_id();
    bool inline_ = true;
};

TEST(NativeTuningPartition, CpuTeamRunsEveryUnitOnce) {
    constexpr int kK = 64;
    const nat::UarchParams &uarch = nat::detect_uarch();
    for (const char *settings : kPlanVariants) {
        const nat::native_problem_t pr = bf16_problem(300, 4096, kK, 8);
        const nat::native_plan_t p
                = nat::heuristic_plan(pr, uarch, params_of(settings));
        if (!partition_legal(p, pr)) continue;
        const nat::native_partition_t part(p, pr.M, 0, p.n_tiles);
        for (int team : {1, 2, p.threads, p.threads + 1}) {
            SCOPED_TRACE(std::string("'") + settings + "', plan " + text(p)
                    + ", team " + std::to_string(team));
            team_log_t log;
            const size_t floats = p.scratch_floats;
            // The whole scratch is the thread's own.
            auto tile = [&](const unit_t &u, float *scratch) {
                ASSERT_NE(scratch, nullptr);
                std::fill(scratch, scratch + floats, 1.0f);
                log.add(u, scratch);
            };
            EXPECT_EQ(nat::native_run_cpu_team(part, team, team, floats, tile),
                    status_t::success);
            EXPECT_TRUE(covers_once(log.units(), p, pr.M, 0, p.n_tiles));
            EXPECT_TRUE(log.scratch_private());
            if (team == 1) {
                EXPECT_TRUE(log.all_inline())
                        << "a team of 1 runs on the calling thread";
            }
        }
    }
}

TEST(NativeTuningPartition, CpuTeamInsideParallelRegionRunsEveryUnitOnce) {
    constexpr int kK = 64;
    nat::heuristic_params_t hp;
    hp.col_groups = 4;
    const nat::native_problem_t pr = bf16_problem(300, 1024, kK, 8);
    const nat::native_plan_t p
            = nat::heuristic_plan(pr, nat::detect_uarch(), hp);
    ASSERT_TRUE(partition_legal(p, pr)) << text(p);
    ASSERT_GT(p.col_groups, 0);
    // An inner region of one thread: the row-group plan meets another team.
    team_log_t log;
    std::atomic<int> failures {0};
    {
        const zendnnl::lowoha::scoped_active_levels one_level(1);
#pragma omp parallel num_threads(2)
        {
            const int h = omp_get_thread_num(), n = omp_get_num_threads();
            const nat::native_partition_t part(
                    p, pr.M, p.n_tiles * h / n, p.n_tiles * (h + 1) / n);
            const status_t st = nat::native_run_cpu_team(part, p.threads,
                    p.threads, p.scratch_floats,
                    [&](const unit_t &u, float *scratch) {
                log.add(u, scratch);
            });
            if (st != status_t::success) ++failures;
        }
    }
    EXPECT_EQ(failures.load(), 0);
    EXPECT_TRUE(covers_once(log.units(), p, pr.M, 0, p.n_tiles));
}

} // namespace
