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

// The built-in plan of the native BRGEMM: heuristic_params_t's defaults are
// the built-in constants, a planner without sources returns
// heuristic_plan(heuristic_params_t{}) over dense linear shapes plus small N,
// M from 1 to 8192, 1 to 256 threads and both dtypes, and that plan and the
// plan with every rule switched off pass validate_native_plan on the detected
// and on synthetic L2 / L3 / L3-domain sizes.  The INT8 tile's call widths,
// rows per call and K passes follow their definitions.  The L3 domain of the
// plan is the topology's physical cores per L3, else family_l3_cores; it must
// decide the row-group schedule and keep the plan legal for every domain and
// team size.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/int8/native_int8_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_config.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_planner.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_tuning_file.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_hw.hpp"
#include "lowoha_operators/matmul/matmul_native/native_layout_policy.hpp"
#include "matmul_native/native_brgemm_test_utils.hpp"

namespace {

namespace nat = zendnnl::lowoha::matmul::native;
using zendnnl::memory::data_type_t;
using epilogue_t = nat::native_epilogue_class_t;

constexpr data_type_t kF32 = data_type_t::f32;
constexpr data_type_t kBf16 = data_type_t::bf16;
constexpr data_type_t kS8 = data_type_t::s8;
constexpr data_type_t kU8 = data_type_t::u8;
constexpr nat::native_int8_scheme_t kPerGroup
        = nat::native_int8_scheme_t::per_group;

struct shape_t {
    int N, K;
};

/// (N, K) of dense transformer linear layers, output projections included.
const shape_t kModelShapes[] = {{64, 2560}, {64, 4096}, {96, 5120}, {256, 1536},
        {256, 2560}, {768, 768}, {768, 3072}, {1024, 1024}, {1024, 4096},
        {1536, 256}, {1536, 768}, {1536, 1536}, {1536, 2048}, {1536, 4096},
        {1536, 6144}, {1536, 12288}, {2048, 1536}, {2048, 2048}, {2048, 2560},
        {2048, 6144}, {2560, 256}, {2560, 768}, {2560, 1536}, {2560, 2048},
        {2560, 4096}, {2560, 9216}, {2560, 10240}, {3072, 768}, {3072, 1024},
        {3072, 2560}, {3072, 3072}, {3072, 8192}, {3840, 3840}, {3840, 4096},
        {3840, 6912}, {3840, 8192}, {3840, 15360}, {4096, 1536}, {4096, 2048},
        {4096, 2560}, {4096, 4096}, {4096, 6656}, {4096, 12288}, {4096, 14336},
        {4608, 1536}, {4608, 6656}, {5120, 1024}, {5120, 1536}, {5120, 3072},
        {5120, 4096}, {5120, 5120}, {5120, 6144}, {5120, 17408}, {5120, 17920},
        {5120, 32768}, {6144, 2560}, {6144, 4096}, {6144, 5120}, {6656, 4096},
        {6656, 19968}, {7680, 5120}, {8192, 1024}, {8192, 3840}, {8960, 1536},
        {9216, 3840}, {10240, 2560}, {10240, 4096}, {10752, 2560},
        {12288, 1536}, {12288, 2048}, {12288, 2560}, {12288, 4096},
        {14336, 5120}, {16384, 3072}, {16384, 5120}, {18432, 2560},
        {20480, 2560}, {24576, 1536}, {24576, 4096}, {28672, 4096},
        {30720, 3840}, {34816, 5120}, {35840, 5120}, {39936, 6656},
        {65536, 5120}, {100352, 5120}, {128256, 4096}, {131072, 5120},
        {151936, 2048}, {151936, 4096}, {200064, 3072}, {202048, 6656},
        {248320, 2560}, {248320, 4096}, {248320, 5120}, {262144, 1536},
        {262144, 2560}, {262144, 3840}};

/// The model shapes plus narrow outputs at short, odd and long K.
std::vector<shape_t> grid_shapes() {
    std::vector<shape_t> v(std::begin(kModelShapes), std::end(kModelShapes));
    for (int N : {32, 64, 96, 128, 160, 192, 224, 256, 384, 512})
        for (int K : {1, 7, 64, 1000, 4097})
            v.push_back({N, K});
    return v;
}

std::vector<int> grid_ms() {
    std::vector<int> v;
    for (int m = 1; m <= 16; ++m)
        v.push_back(m);
    for (int m : {24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 1024, 2048, 4096,
                 8192})
        v.push_back(m);
    return v;
}

const int kThreads[] = {1, 8, 24, 32, 64, 96, 128, 192, 256};

/// Call properties the built-in plan does not read, cycled over the grid.
struct variant_t {
    data_type_t src, dst;
    bool trans_a, dynamic, has_bias;
    epilogue_t epilogue;
    nat::native_int8_scheme_t scheme = nat::native_int8_scheme_t::per_channel;
};

const variant_t kBf16Variants[] = {
        {kBf16, kBf16, false, false, false, epilogue_t::store},
        {kBf16, kF32, false, false, true, epilogue_t::bias},
        {kBf16, kBf16, true, false, false, epilogue_t::postops},
        {kBf16, kBf16, false, false, true, epilogue_t::gated},
};

const variant_t kInt8Variants[] = {
        {kBf16, kBf16, false, true, false, epilogue_t::store},
        {kU8, kF32, false, false, true, epilogue_t::store},
        {kS8, kBf16, true, false, false, epilogue_t::postops},
        {kF32, kBf16, false, true, true, epilogue_t::gated},
        {kS8, kS8, false, false, false, epilogue_t::postops},
        {kU8, kBf16, false, false, true, epilogue_t::bias},
        {kBf16, kBf16, false, true, false, epilogue_t::store, kPerGroup},
        {kS8, kF32, false, false, false, epilogue_t::store, kPerGroup},
};

nat::native_problem_t make_problem(int M, int N, int K, int pack_nr,
        int threads, bool bf16, size_t variant) {
    const variant_t &v = bf16
            ? kBf16Variants[variant % std::size(kBf16Variants)]
            : kInt8Variants[variant % std::size(kInt8Variants)];
    nat::native_problem_t p;
    p.M = M;
    p.N = N;
    p.K = K;
    p.pack_nr = pack_nr;
    p.threads = threads;
    p.src_dt = v.src;
    p.wei_dt = bf16 ? kBf16 : kS8;
    p.dst_dt = v.dst;
    p.trans_a = v.trans_a;
    p.dynamic_quant = v.dynamic;
    p.has_bias = v.has_bias;
    p.epilogue = v.epilogue;
    p.int8_scheme = v.scheme;
    return p;
}

std::string describe(const nat::native_problem_t &p) {
    std::ostringstream os;
    os << (p.bf16() ? "bf16" : "int8") << " M=" << p.M << " N=" << p.N
       << " K=" << p.K << " pack_nr=" << p.pack_nr << " threads=" << p.threads
       << " src=" << static_cast<int>(p.src_dt)
       << " dst=" << static_cast<int>(p.dst_dt) << " trans_a=" << p.trans_a
       << " epilogue=" << static_cast<int>(p.epilogue);
    return os.str();
}

/// The first field in which two plans differ, or nullptr.
const char *plan_diff(
        const nat::native_plan_t &a, const nat::native_plan_t &b) {
    if (a.nr != b.nr) return "nr";
    if (a.mr != b.mr) return "mr";
    if (a.mb != b.mb) return "mb";
    if (a.m_blocks != b.m_blocks) return "m_blocks";
    if (a.n_tiles != b.n_tiles) return "n_tiles";
    if (a.kc != b.kc) return "kc";
    if (a.threads != b.threads) return "threads";
    if (a.col_groups != b.col_groups) return "col_groups";
    if (a.unit_tiles != b.unit_tiles) return "unit_tiles";
    if (a.domain_threads != b.domain_threads) return "domain_threads";
    if (a.row_unit != b.row_unit) return "row_unit";
    if (a.traversal != b.traversal) return "traversal";
    if (a.scratch_floats != b.scratch_floats) return "scratch_floats";
    if (a.int8_rows_nv1 != b.int8_rows_nv1) return "int8_rows_nv1";
    if (a.int8_rows_nv2 != b.int8_rows_nv2) return "int8_rows_nv2";
    if (a.int8_cw_mask != b.int8_cw_mask) return "int8_cw_mask";
    if (a.int8_pass_budget != b.int8_pass_budget) return "int8_pass_budget";
    if (a.int8_pass_quads != b.int8_pass_quads) return "int8_pass_quads";
    if (a.int8_direct_store != b.int8_direct_store) return "int8_direct_store";
    if (a.int8_bias_in_dequant != b.int8_bias_in_dequant)
        return "int8_bias_in_dequant";
    if (a.copy_block != b.copy_block) return "copy_block";
    if (a.copy_par_min != b.copy_par_min) return "copy_par_min";
    if (a.sources != b.sources) return "sources";
    if (a.rules_entry != b.rules_entry) return "rules_entry";
    if (a.table_entry != b.table_entry) return "table_entry";
    if (a.override_entry != b.override_entry) return "override_entry";
    return nullptr;
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

/// The parameters with every rule switch off.
nat::heuristic_params_t rules_off() {
    nat::native_config_t cfg;
    std::string why;
    EXPECT_TRUE(nat::parse_native_config(
            native_brgemm_test::kNativeRulesOff, cfg, why))
            << why;
    return nat::apply_native_config(nat::heuristic_params_t {}, cfg);
}

// The plans with every rule switch off, field by field, on 48 KiB of L1,
// 1 MiB of L2 and 32 MiB of L3 per 8 cores at 32 threads: the narrowing to
// 16 columns of a decode slice, the row split of one block, the BF16 L2 K
// pass, the INT8 pass budget and the L3 row-group test.
TEST(NativeTuningPlan, RulesOffWorkedExamples) {
    nat::UarchParams u = nat::detect_uarch();
    u.l1d_bytes = 48 << 10;
    u.l2_bytes = 1 << 20;
    u.l3_domain_bytes = 32 << 20;
    u.family_l3_cores = 1;
    u.l3_domain_cores = 8;
    const nat::heuristic_params_t none = rules_off();
    const auto plan = [&](int M, int N, int K, bool bf16) {
        return nat::heuristic_plan(
                make_problem(M, N, K, nat::kNativeLayoutPackNr, 32, bf16, 0), u,
                none);
    };
    const struct {
        int M, N, K;
        bool bf16;
        int nr, mb, kc, threads, col_groups;
    } examples[] = {
            // A decode slice of 8 tiles narrows to 16 columns.
            {1, 256, 4096, true, 16, 1, 4096, 16, 0},
            // One block of 192 rows splits into two for 32 threads.
            {192, 256, 4096, true, 16, 96, 4096, 32, 0},
            // A 32-column B slab over K = 14336 passes half of L2.
            {1024, 4096, 14336, true, 32, 192, 8192, 32, 0},
            // The source and a domain's B slabs over 32 MiB, 24 column tiles
            // for 32 threads: row groups of the 8 threads of a domain.
            {4096, 768, 4096, true, 32, 192, 4096, 32, 8},
            // INT8: no K pass of its own; the INT8 source fits the share.
            {1024, 4096, 14336, false, 32, 192, 14336, 32, 0},
    };
    for (const auto &e : examples) {
        const nat::native_plan_t q = plan(e.M, e.N, e.K, e.bf16);
        SCOPED_TRACE(std::string(e.bf16 ? "bf16" : "int8")
                + " M=" + std::to_string(e.M) + " N=" + std::to_string(e.N)
                + " K=" + std::to_string(e.K));
        EXPECT_EQ(q.nr, e.nr);
        EXPECT_EQ(q.mb, e.mb);
        EXPECT_EQ(q.kc, e.kc);
        EXPECT_EQ(q.threads, e.threads);
        EXPECT_EQ(q.col_groups, e.col_groups);
        EXPECT_EQ(q.unit_tiles, 0);
    }
    // INT8 passes come from the tile's budget, half of L2: a tile of 192
    // rows at 32 columns and 12 rows per call takes two passes of K = 14336.
    const nat::native_plan_t i = plan(1024, 4096, 14336, false);
    EXPECT_EQ(i.int8_pass_quads, 0);
    EXPECT_EQ(i.int8_pass_budget, 1 << 19);
    EXPECT_EQ(nat::native_int8_pass_quads(i, 3584, 32, 192, 12), 1792);
}

// nb sizes the unit of the blocks schedule in columns: nb / nr column
// tiles, at least one, so a unit of 256 columns keeps its width at tiles of
// 16, 32 and 64 columns.  It replaces unit_tiles under the same bounds (at
// most n_tiles; under BF16 K passes the tiles whose accumulators fit the
// bf16_l2_pct share of L2) and takes no call to the blocks schedule; where
// the blocks rule takes one, the unit is nb wide.  1 MiB of L2 and 32 MiB
// of L3 per 8 cores, at 32 threads.
TEST(NativeTuningPlan, NbSizesTheBlocksUnit) {
    nat::UarchParams u = nat::detect_uarch();
    u.l1d_bytes = 48 << 10;
    u.l2_bytes = 1 << 20;
    u.l3_domain_bytes = 32 << 20;
    u.family_l3_cores = 1;
    u.l3_domain_cores = 8;
    const nat::UarchParams &host = nat::detect_uarch();
    for (const bool bf16 : {true, false}) {
        SCOPED_TRACE(bf16 ? "bf16" : "int8");
        const nat::native_problem_t p = make_problem(
                1024, 4096, 4096, nat::kNativeLayoutPackNr, 32, bf16, 0);
        const bool isa = bf16 ? host.avx512bf16 : host.avx512vnni;
        nat::heuristic_params_t hp = rules_off();
        hp.schedule = nat::native_schedule_t::blocks;
        hp.kc = p.K;
        for (const int nr : {16, 32, 64}) {
            if (!bf16 && nr == 64) continue;
            hp.nr = nr;
            for (const int nb : {16, 48, 256, 1 << 20}) {
                SCOPED_TRACE("nr " + std::to_string(nr) + " nb "
                        + std::to_string(nb));
                hp.nb = nb;
                hp.unit_tiles = 0;
                const nat::native_plan_t q = nat::heuristic_plan(p, u, hp);
                EXPECT_EQ(q.unit_tiles,
                        std::min(std::max(1, nb / nr), q.n_tiles));
                if (isa) {
                    EXPECT_EQ(nat::validate_native_plan(q, p), nullptr);
                }
                hp.unit_tiles = 3;
                EXPECT_EQ(plan_diff(nat::heuristic_plan(p, u, hp), q), nullptr);
            }
            hp.nb = 256;
            hp.unit_tiles = 0;
            EXPECT_EQ(nr * nat::heuristic_plan(p, u, hp).unit_tiles, 256) << nr;
        }
        // A unit of every column: under BF16 K passes of 1024, the tiles
        // whose accumulators fit half of L2.
        hp.nr = 32;
        hp.nb = p.N;
        hp.kc = bf16 ? 1024 : 0;
        const nat::native_plan_t k = nat::heuristic_plan(p, u, hp);
        const int64_t fit = static_cast<int64_t>(u.l2_bytes) / 2
                / (static_cast<int64_t>(k.mb) * k.nr * sizeof(float));
        EXPECT_EQ(k.unit_tiles,
                bf16 ? std::clamp<int64_t>(fit, 1, k.n_tiles) : k.n_tiles);
        if (bf16) { EXPECT_LT(k.unit_tiles, k.n_tiles); }
        const nat::heuristic_params_t none = rules_off();
        nat::heuristic_params_t wide = none;
        wide.nb = 256;
        EXPECT_EQ(plan_diff(nat::heuristic_plan(p, u, wide),
                          nat::heuristic_plan(p, u, none)),
                nullptr);
    }
    // A BF16 B slab over K = 14336 exceeds half of L2 and a domain's share
    // of the source fits its L3: the blocks rule takes the call.
    nat::heuristic_params_t rule = rules_off();
    rule.blocks = 1;
    const nat::native_problem_t r = make_problem(
            4096, 4096, 14336, nat::kNativeLayoutPackNr, 32, true, 0);
    ASSERT_EQ(nat::heuristic_plan(r, u, rule).unit_tiles, 1);
    rule.nb = 128;
    const nat::native_plan_t b = nat::heuristic_plan(r, u, rule);
    EXPECT_EQ(b.unit_tiles, 128 / b.nr);
}

TEST(NativeTuningPlan, DefaultParamsAreTheBuiltInConstants) {
    const nat::heuristic_params_t hp;
    EXPECT_EQ(hp.nr, 0);
    EXPECT_EQ(hp.mr, 0);
    EXPECT_EQ(hp.mb, 0);
    EXPECT_EQ(hp.nb, 0);
    EXPECT_EQ(hp.kc, 0);
    EXPECT_EQ(hp.col_groups, -1);
    EXPECT_EQ(hp.schedule, nat::native_schedule_t::automatic);
    EXPECT_EQ(hp.traversal, nat::native_traversal_t::m_inner);
    EXPECT_EQ(hp.threads_cap, 0);
    EXPECT_EQ(hp.min_tiles_per_thread, 1);
    EXPECT_EQ(hp.row_unit, 12);
    EXPECT_EQ(hp.tile_rows, 6);
    EXPECT_EQ(hp.tall_rows, 12);
    EXPECT_EQ(hp.block_rows, 192);
    EXPECT_EQ(hp.min_tile_cols, 16);
    EXPECT_TRUE(hp.row_split);
    EXPECT_EQ(hp.min_pass_k, 256);
    EXPECT_EQ(hp.bf16_l2_pct, 50);
    EXPECT_EQ(hp.kc_split, nat::native_kc_split_t::budget);
    EXPECT_EQ(hp.l3_pct, 100);
    EXPECT_EQ(hp.domain_threads, 0);
    EXPECT_EQ(hp.int8_mr, 12);
    EXPECT_EQ(hp.int8_cw_mask, nat::kNativeInt8CwAll);
    EXPECT_EQ(hp.int8_l2_pct, 50);
    EXPECT_EQ(hp.int8_kc, 0);
    EXPECT_EQ(hp.int4_l1_pct, 125);
    EXPECT_TRUE(hp.direct_store);
    EXPECT_TRUE(hp.bias_in_dequant);
    EXPECT_EQ(hp.copy_block, 64);
    EXPECT_EQ(hp.bf16_copy_par_min, 0);
    EXPECT_EQ(hp.int8_copy_par_min, 65536);
    // Every rule is on.
    EXPECT_EQ(hp.makespan, 1);
    EXPECT_EQ(hp.min_gain_pct, 5);
    EXPECT_EQ(hp.idle_pct, 10);
    EXPECT_EQ(hp.decode_m, 128);
    EXPECT_EQ(hp.l1_pass, 1);
    EXPECT_EQ(hp.l1_pct, 100);
    EXPECT_EQ(hp.l1_pass_max_mb, 64);
    EXPECT_EQ(hp.traffic_groups, 1);
    EXPECT_EQ(hp.bf16_alias, 1);
    EXPECT_EQ(hp.bf16_narrow_n, 1);
    EXPECT_EQ(hp.blocks, 1);
    EXPECT_EQ(hp.cost_model, 1);
    // The cost model's inputs.
    EXPECT_EQ(hp.cm_gain_pct, 5);
    EXPECT_EQ(hp.cm_short_rows, 8);
    EXPECT_EQ(hp.cm_fma_pipes, 0);
    EXPECT_EQ(hp.cm_fma_lat, 6);
    EXPECT_EQ(hp.cm_call_cycles, 800);
    EXPECT_EQ(hp.cm_tall_pct, -1);
    EXPECT_EQ(hp.cm_l2_bytes, 16);
    EXPECT_EQ(hp.cm_l3_cycles, 10);
    EXPECT_EQ(hp.cm_mem_cycles, 0);
    EXPECT_EQ(hp.cm_prefetch_kib, 4);
    EXPECT_EQ(hp.cm_overlap_pct, 20);
    EXPECT_EQ(hp.cm_reuse_pct, 50);
    EXPECT_EQ(hp.cm_stream_cycles, 600);
    EXPECT_EQ(hp.cm_sync_tiles, 5);
    EXPECT_EQ(hp.cm_group_slab_pct, 75);
    EXPECT_EQ(hp.cm_group_l3_pct, 75);
    EXPECT_EQ(hp.cm_group_skew_pct, 110);
    EXPECT_EQ(hp.cm_group_rows_pct, 115);
    EXPECT_EQ(hp.cm_drift_kib, 4096);
    EXPECT_EQ(hp.cm_pass_kib, 128);
    EXPECT_EQ(hp.cm_pass_calls, 4);
    EXPECT_EQ(hp.cm_block_kib, 16384);
    EXPECT_EQ(hp.cm_block_k, 2048);
    EXPECT_EQ(hp.cm_int8_l1_pct, 0);
    EXPECT_EQ(hp.cm_int4_min_m, 48);
}

/// Quads per K pass of an INT8 tile cw columns wide and rows tall, max_mr
/// rows per call, under budget bytes: all of K unless the tile takes more
/// than one row call and the B slab plus one call's A rows exceed the
/// budget; then the fewest equal passes that fit.
int pass_quads_by_definition(
        int k_quads, int cw, int rows, int max_mr, int64_t budget) {
    const int64_t quad_bytes = static_cast<int64_t>(4) * (cw + max_mr);
    if (rows <= max_mr || k_quads * quad_bytes <= budget) return k_quads;
    const int64_t per = std::max<int64_t>(1, budget / quad_bytes);
    const int64_t passes = (k_quads + per - 1) / per;
    return static_cast<int>((k_quads + passes - 1) / passes);
}

TEST(NativeTuningPlan, Int8TileHelpers) {
    report_t report;
    for (int l2_kib : {256, 512, 1000, 1024, 2048}) {
        nat::UarchParams uarch = nat::detect_uarch();
        uarch.l2_bytes = l2_kib * 1024;
        // Without the L1 pass rule the plan sets no pass of its own.
        const nat::native_problem_t p = make_problem(
                64, 256, 4096, nat::kNativeLayoutPackNr, 1, false, 0);
        const nat::native_plan_t q = nat::heuristic_plan(p, uarch, rules_off());
        ASSERT_EQ(q.int8_pass_quads, 0);
        EXPECT_EQ(q.int8_pass_budget, uarch.l2_bytes / 2);
        // A call is 32 columns wide when the mask allows it and 32 columns
        // are left, else 16.
        for (const uint8_t mask : {nat::kNativeInt8CwAll, nat::kNativeInt8Cw16,
                     nat::kNativeInt8Cw32}) {
            nat::native_plan_t m = q;
            m.int8_cw_mask = mask;
            for (int left = 16; left <= 256; left += 16) {
                const int cw = nat::native_int8_col_width(m, left);
                const int want
                        = (mask & nat::kNativeInt8Cw32) != 0 && left >= 32 ? 32
                                                                           : 16;
                report.check(cw == want, [&] {
                    return "mask " + std::to_string(mask) + ", "
                            + std::to_string(left)
                            + " columns left: call width " + std::to_string(cw);
                });
            }
        }
        // The rows per call of a per-channel call: int8_mr, at most the
        // kernel's rows at each width; of a per-group call, at most the
        // group kernel's.
        nat::native_problem_t pg = p;
        pg.int8_scheme = kPerGroup;
        const nat::native_plan_t g
                = nat::heuristic_plan(pg, uarch, rules_off());
        for (int nv : {1, 2}) {
            const int rows = nat::native_int8_call_rows(q, nv);
            const int want = std::min(12, nat::native_int8_ukernel_max_mr(nv));
            report.check(rows == want, [&] {
                return "nv " + std::to_string(nv) + ": rows per call "
                        + std::to_string(rows);
            });
            const int group_rows = nat::native_int8_call_rows(g, nv);
            const int group_want
                    = std::min(12, nat::native_int8_group_ukernel_max_mr(nv));
            report.check(group_rows == group_want, [&] {
                return "nv " + std::to_string(nv) + ": per-group rows per call "
                        + std::to_string(group_rows);
            });
        }
        // K quads up to the int32-exact bound of the INT8 layout.
        for (int k_quads : {1, 2, 3, 63, 64, 255, 256, 1000, 1024, 1025, 1724,
                     1725, 2048, 4096, 8192, 16449})
            for (int cw : {16, 32})
                for (int max_mr : {6, 12})
                    for (int rows = 1; rows <= 192;
                            rows += rows < 26 ? 1 : 11) {
                        const int got = nat::native_int8_pass_quads(
                                q, k_quads, cw, rows, max_mr);
                        const int want = pass_quads_by_definition(
                                k_quads, cw, rows, max_mr, q.int8_pass_budget);
                        report.check(got == want, [&] {
                            std::ostringstream os;
                            os << "K pass quads, L2 " << l2_kib
                               << " KiB k_quads " << k_quads << " cw " << cw
                               << " rows " << rows << " max_mr " << max_mr
                               << ": " << got << ", by definition " << want;
                            return os.str();
                        });
                    }
        // A pinned pass is kept, at most all of K.
        nat::native_plan_t pinned = q;
        for (int pin : {1, 7, 256, 1024, 20000}) {
            pinned.int8_pass_quads = pin;
            for (int k_quads : {1, 255, 1024, 16449}) {
                const int got = nat::native_int8_pass_quads(
                        pinned, k_quads, 32, 192, 12);
                report.check(got == std::min(pin, k_quads), [&] {
                    return "pinned " + std::to_string(pin) + " of "
                            + std::to_string(k_quads)
                            + " quads: " + std::to_string(got);
                });
            }
        }
    }
    report.finish();
}

TEST(NativeTuningPlan, PlannerWithoutSettingsIsTheBuiltInPlan) {
    for (const char *name : {"ZENDNNL_MATMUL_NATIVE_CONFIG",
                 "ZENDNNL_MATMUL_NATIVE_TUNING_FILE"}) {
        const char *value = std::getenv(name);
        if (value != nullptr && *value != '\0')
            GTEST_SKIP() << name << " is set in this process";
    }
    const std::unique_ptr<nat::native_planner_t> unset
            = nat::make_native_planner(nullptr, nullptr);
    const std::unique_ptr<nat::native_planner_t> empty
            = nat::make_native_planner("", "");
    ASSERT_NE(unset, nullptr);
    ASSERT_NE(empty, nullptr);
    const nat::native_planner_t *planners[]
            = {&nat::native_planner(), unset.get(), empty.get()};
    const char *names[] = {"native_planner()", "make_native_planner(nullptr)",
            "make_native_planner(\"\")"};
    for (size_t i = 0; i < std::size(planners); ++i)
        EXPECT_FALSE(planners[i]->has_sources()) << names[i];

    const nat::UarchParams &uarch = nat::native_plan_uarch();
    const std::vector<shape_t> shapes = grid_shapes();
    const std::vector<int> ms = grid_ms();
    report_t report;
    for (size_t si = 0; si < shapes.size(); ++si)
        for (size_t mi = 0; mi < ms.size(); ++mi)
            for (size_t ti = 0; ti < std::size(kThreads); ++ti)
                for (bool bf16 : {true, false}) {
                    const nat::native_problem_t p
                            = make_problem(ms[mi], shapes[si].N, shapes[si].K,
                                    nat::kNativeLayoutPackNr, kThreads[ti],
                                    bf16, si * 5 + mi * 3 + ti);
                    const nat::native_plan_t want = nat::heuristic_plan(
                            p, uarch, nat::heuristic_params_t {});
                    for (size_t i = 0; i < std::size(planners); ++i) {
                        const char *field
                                = plan_diff(planners[i]->plan(p), want);
                        report.check(field == nullptr, [&] {
                            return describe(p) + ": " + names[i]
                                    + " differs in " + field;
                        });
                    }
                }
    report.finish();
}

struct uarch_case_t {
    std::string name;
    int l2_kib = 0; ///< 0: the micro-architecture the process plans with
    int l3_mib = 0;
    int family_l3_cores = 0;
    int l3_domain_cores = 0;
};

void PrintTo(const uarch_case_t &c, std::ostream *os) {
    *os << c.name;
}

std::vector<uarch_case_t> uarch_cases() {
    std::vector<uarch_case_t> v(1);
    v[0].name = "detected";
    // L3 domain unknown: family_l3_cores is the domain.
    for (int l2 : {512, 1024, 2048})
        for (int l3 : {16, 32, 96})
            for (int cores : {1, 8, 16}) {
                uarch_case_t c;
                c.name = "l2_" + std::to_string(l2) + "k_l3_"
                        + std::to_string(l3) + "m_fam" + std::to_string(cores);
                c.l2_kib = l2;
                c.l3_mib = l3;
                c.family_l3_cores = cores;
                v.push_back(c);
            }
    // L3 domain from the topology, over a family_l3_cores of 1.
    for (int l2 : {1024, 2048})
        for (int l3 : {32, 128, 256})
            for (int domain : {8, 32, 64}) {
                uarch_case_t c;
                c.name = "l2_" + std::to_string(l2) + "k_l3_"
                        + std::to_string(l3) + "m_dom" + std::to_string(domain);
                c.l2_kib = l2;
                c.l3_mib = l3;
                c.family_l3_cores = 1;
                c.l3_domain_cores = domain;
                v.push_back(c);
            }
    return v;
}

nat::UarchParams uarch_of(const uarch_case_t &c) {
    nat::UarchParams u = nat::native_plan_uarch();
    if (c.l2_kib == 0) return u;
    u.l2_bytes = c.l2_kib * 1024;
    u.l3_domain_bytes = c.l3_mib * 1024 * 1024;
    u.family_l3_cores = c.family_l3_cores;
    u.l3_domain_cores = c.l3_domain_cores;
    return u;
}

class NativeTuningPlanHierarchy
    : public ::testing::TestWithParam<uarch_case_t> {};

// The built-in plan and the plan with every rule switched off are legal on
// each hierarchy, and a planner without sources returns the built-in plan.
TEST_P(NativeTuningPlanHierarchy, PlansAreLegal) {
    const nat::UarchParams &host = nat::detect_uarch();
    const nat::UarchParams uarch = uarch_of(GetParam());
    const nat::native_planner_t planner(uarch, nullptr, nullptr, nullptr);
    EXPECT_FALSE(planner.has_sources());
    const nat::heuristic_params_t none = rules_off();
    const std::vector<shape_t> shapes = grid_shapes();
    const std::vector<int> ms = grid_ms();
    report_t report;
    for (size_t si = 0; si < shapes.size(); ++si)
        for (size_t mi = 0; mi < ms.size(); ++mi)
            for (size_t ti = 0; ti < std::size(kThreads); ++ti)
                for (bool bf16 : {true, false}) {
                    const nat::native_problem_t p
                            = make_problem(ms[mi], shapes[si].N, shapes[si].K,
                                    nat::kNativeLayoutPackNr, kThreads[ti],
                                    bf16, si * 5 + mi * 3 + ti);
                    const nat::native_plan_t q = nat::heuristic_plan(
                            p, uarch, nat::heuristic_params_t {});
                    // The kernel registry offers a dtype only on a CPU with
                    // its ISA.
                    if (p.bf16() ? host.avx512bf16 : host.avx512vnni) {
                        const nat::native_plan_t plans[]
                                = {q, nat::heuristic_plan(p, uarch, none)};
                        const char *const names[]
                                = {"the built-in plan", "every rule off"};
                        for (size_t i = 0; i < std::size(plans); ++i) {
                            const char *why
                                    = nat::validate_native_plan(plans[i], p);
                            report.check(why == nullptr, [&] {
                                return describe(p) + ": " + names[i]
                                        + ": validate_native_plan rejects it: "
                                        + why;
                            });
                        }
                    }
                    const char *field = plan_diff(planner.plan(p), q);
                    report.check(field == nullptr, [&] {
                        return describe(p)
                                + ": the planner without sources differs in "
                                + field;
                    });
                }
    report.finish();
}

INSTANTIATE_TEST_SUITE_P(NativeTuning, NativeTuningPlanHierarchy,
        ::testing::ValuesIn(uarch_cases()),
        [](const ::testing::TestParamInfo<uarch_case_t> &info) {
            return info.param.name;
        });

/// A machine with l3_threads logical CPUs on each L3 of l3_mib MiB, smt per
/// core (0: unknown), whose CPU family gives family_l3_cores.
nat::native_hw_t topology(
        int l3_threads, int smt, int l3_mib, int family_l3_cores) {
    nat::native_hw_t hw;
    hw.uarch = nat::detect_uarch();
    hw.uarch.l2_bytes = 1 << 20;
    hw.uarch.l3_domain_bytes = l3_mib << 20;
    hw.uarch.family_l3_cores = family_l3_cores;
    hw.logical_cpus = 512;
    hw.smt = smt;
    hw.physical_cores = smt > 0 ? hw.logical_cpus / smt : 0;
    hw.l3_domain_threads = l3_threads;
    return hw;
}

TEST(NativeTuningPlanDomain, CoresPerL3FromTheTopology) {
    const struct {
        int l3_threads, smt, family_l3_cores, cores;
    } cases[] = {{64, 2, 8, 32}, {64, 2, 1, 32}, {32, 1, 8, 32}, {16, 2, 8, 8},
            {16, 2, 1, 8}, {16, 0, 8, 16}, {1, 2, 8, 1}, {0, 2, 8, 0},
            {0, 0, 1, 0}};
    for (const auto &c : cases) {
        SCOPED_TRACE("l3_threads " + std::to_string(c.l3_threads) + " smt "
                + std::to_string(c.smt) + " family_l3_cores "
                + std::to_string(c.family_l3_cores));
        const nat::native_hw_t hw
                = topology(c.l3_threads, c.smt, 128, c.family_l3_cores);
        const nat::UarchParams u = nat::native_plan_uarch(hw);
        EXPECT_EQ(u.l3_domain_cores, c.cores);
        EXPECT_EQ(u.family_l3_cores, c.family_l3_cores);
        EXPECT_EQ(u.l1d_bytes, hw.uarch.l1d_bytes);
        EXPECT_EQ(u.l2_bytes, hw.uarch.l2_bytes);
        EXPECT_EQ(u.l3_domain_bytes, hw.uarch.l3_domain_bytes);
        EXPECT_EQ(u.num_cores, hw.uarch.num_cores);
        EXPECT_EQ(u.fma_ports, hw.uarch.fma_ports);
        EXPECT_EQ(u.vec_width_bits, hw.uarch.vec_width_bits);
        EXPECT_EQ(u.avx512f, hw.uarch.avx512f);
        EXPECT_EQ(u.avx512bf16, hw.uarch.avx512bf16);
        EXPECT_EQ(u.avx512vnni, hw.uarch.avx512vnni);
    }
    // The process plans over its own topology; detect_uarch() has no domain.
    const nat::UarchParams &host = nat::detect_uarch();
    const nat::UarchParams &plan = nat::native_plan_uarch();
    EXPECT_EQ(host.l3_domain_cores, 0);
    EXPECT_EQ(plan.l3_domain_cores,
            nat::native_plan_uarch(nat::native_hw()).l3_domain_cores);
    EXPECT_EQ(plan.family_l3_cores, host.family_l3_cores);
    EXPECT_EQ(plan.l2_bytes, host.l2_bytes);
    EXPECT_EQ(plan.l3_domain_bytes, host.l3_domain_bytes);
}

TEST(NativeTuningPlanDomain, RowGroupsFollowTheDomain) {
    const nat::UarchParams wide
            = nat::native_plan_uarch(topology(64, 2, 128, 8));
    const nat::UarchParams wide_no_family
            = nat::native_plan_uarch(topology(64, 2, 128, 1));
    const nat::UarchParams narrow
            = nat::native_plan_uarch(topology(16, 2, 128, 8));
    const nat::UarchParams unknown
            = nat::native_plan_uarch(topology(0, 2, 128, 8));
    ASSERT_EQ(wide.l3_domain_cores, 32);
    ASSERT_EQ(wide_no_family.l3_domain_cores, 32);
    ASSERT_EQ(narrow.l3_domain_cores, 8);
    ASSERT_EQ(unknown.l3_domain_cores, 0);
    const nat::UarchParams &host = nat::detect_uarch();
    // The row-group rule of the plan with every rule switched off.
    const nat::heuristic_params_t none = rules_off();
    nat::heuristic_params_t pin8 = none, pin32 = none;
    pin8.domain_threads = 8;
    pin32.domain_threads = 32;
    const struct {
        const char *what;
        int M, N, K;
        bool bf16;
        int threads;
        int wide_groups, narrow_groups;
    } cases[] = {
            {"source and slabs over the L3 for both domains", 8192, 3072, 8192,
                    true, 128, 32, 8},
            {"over the L3 with the slabs of the wide domain only", 7680, 3072,
                    8192, true, 128, 32, 0},
            {"s8 over the L3 with the slabs of the wide domain only", 16000,
                    3072, 8192, false, 128, 32, 0},
            {"whole columns per thread", 8192, 4096, 8192, true, 128, 0, 0},
            {"team smaller than the wide domain", 8192, 3328, 8192, true, 24, 8,
                    8},
    };
    for (const auto &c : cases) {
        SCOPED_TRACE(c.what);
        const nat::native_problem_t p = make_problem(
                c.M, c.N, c.K, nat::kNativeLayoutPackNr, c.threads, c.bf16, 0);
        const nat::native_plan_t w = nat::heuristic_plan(p, wide, none);
        const nat::native_plan_t n = nat::heuristic_plan(p, narrow, none);
        EXPECT_EQ(w.col_groups, c.wide_groups) << describe(p);
        EXPECT_EQ(n.col_groups, c.narrow_groups) << describe(p);
        if (p.bf16() ? host.avx512bf16 : host.avx512vnni) {
            EXPECT_EQ(nat::validate_native_plan(w, p), nullptr) << describe(p);
            EXPECT_EQ(nat::validate_native_plan(n, p), nullptr) << describe(p);
        }
        // family_l3_cores counts only when the domain is unknown.
        const char *field
                = plan_diff(nat::heuristic_plan(p, wide_no_family, none), w);
        EXPECT_EQ(field, nullptr) << "family_l3_cores 1 under a known domain";
        field = plan_diff(nat::heuristic_plan(p, unknown, none), n);
        EXPECT_EQ(field, nullptr) << "unknown domain, family_l3_cores 8";
        // domain_threads replaces the detected domain.
        field = plan_diff(nat::heuristic_plan(p, wide, pin8), n);
        EXPECT_EQ(field, nullptr) << "domain_threads=8 on the wide domain";
        field = plan_diff(nat::heuristic_plan(p, narrow, pin32), w);
        EXPECT_EQ(field, nullptr) << "domain_threads=32 on the narrow domain";
        const nat::native_planner_t planner(wide, nullptr, nullptr, nullptr);
        field = plan_diff(planner.plan(p),
                nat::heuristic_plan(p, wide, nat::heuristic_params_t {}));
        EXPECT_EQ(field, nullptr) << "the planner over the wide domain";
        const nat::native_planner_t overridden(wide,
                nat::make_native_override_source(
                        std::string(native_brgemm_test::kNativeRulesOff)
                        + ",domain_threads=8"),
                nullptr, nullptr);
        const nat::native_plan_t o = overridden.plan(p);
        EXPECT_EQ(o.col_groups, c.narrow_groups) << "the override key";
        EXPECT_EQ(nat::native_plan_source_names(o), "override");
    }
}

TEST(NativeTuningPlanDomain, LegalForEveryDomainAndTeam) {
    const shape_t shapes[] = {{32, 64}, {96, 4097}, {1024, 1000}, {3072, 8192},
            {5120, 32768}, {14336, 4096}};
    const int ms[] = {1, 7, 12, 193, 1031, 4096, 8192};
    const nat::UarchParams &host = nat::detect_uarch();
    report_t report;
    for (int domain = 1; domain <= 64; ++domain) {
        const nat::UarchParams u = nat::native_plan_uarch(
                topology(2 * domain, 2, 4 * domain, 1));
        ASSERT_EQ(u.l3_domain_cores, domain);
        const nat::native_planner_t planner(u, nullptr, nullptr, nullptr);
        for (int threads = 1; threads <= 256; ++threads)
            for (size_t si = 0; si < std::size(shapes); ++si)
                for (size_t mi = 0; mi < std::size(ms); ++mi)
                    for (bool bf16 : {true, false}) {
                        const nat::native_problem_t p = make_problem(ms[mi],
                                shapes[si].N, shapes[si].K,
                                nat::kNativeLayoutPackNr, threads, bf16,
                                si * 5 + mi * 3 + threads);
                        const nat::native_plan_t q = nat::heuristic_plan(
                                p, u, nat::heuristic_params_t {});
                        const auto where = [&] {
                            return "domain " + std::to_string(domain) + " "
                                    + describe(p);
                        };
                        if (p.bf16() ? host.avx512bf16 : host.avx512vnni) {
                            const char *why = nat::validate_native_plan(q, p);
                            report.check(why == nullptr, [&] {
                                return where()
                                        + ": validate_native_plan rejects it: "
                                        + why;
                            });
                        }
                        report.check(q.col_groups >= 0
                                        && (q.col_groups == 0
                                                || domain % q.col_groups == 0),
                                [&] {
                            return where() + ": col_groups "
                                    + std::to_string(q.col_groups);
                        });
                        const char *field = plan_diff(planner.plan(p), q);
                        report.check(field == nullptr, [&] {
                            return where() + ": the planner differs in "
                                    + field;
                        });
                    }
    }
    report.finish();
}

} // namespace
