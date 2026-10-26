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

// The legacy preset of the native BRGEMM plan against a reference copy of
// its planning rules.  heuristic_plan with preset=legacy must give, field by
// field, the reference copy's tile grid, K pass, schedule constants, scratch
// size and the INT8 tile's call widths, rows per call and K passes, over
// dense linear shapes plus small N, M from 1 to 8192, 1 to 256 threads, both
// dtypes, the detected micro-architecture and synthetic L2 / L3 / L3-domain
// sizes; so must preset=v2 with every v2 rule switched off.  Every such plan
// must pass validate_native_plan, and a planner without sources must return
// the built-in plan (heuristic_params_t{}) unchanged.  The
// L3 domain of the plan is the topology's physical cores per L3, else
// ccx_cores; it must decide the row-group schedule and keep the plan legal
// for every domain and team size.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <numeric>
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

namespace {

namespace nat = zendnnl::lowoha::matmul::native;
using zendnnl::memory::data_type_t;
using epilogue_t = nat::native_epilogue_class_t;

constexpr data_type_t kF32 = data_type_t::f32;
constexpr data_type_t kBf16 = data_type_t::bf16;
constexpr data_type_t kS8 = data_type_t::s8;
constexpr data_type_t kU8 = data_type_t::u8;

/// Reference copy of the built-in planning rules, with the micro-architecture
/// as a parameter.  The library's built-in plan must reproduce it exactly, so
/// it must not be edited to follow the library.
namespace frozen {

constexpr int kTileRows = 6;
constexpr int kTallTileRows = 12;
constexpr int kBlockRows = 16 * kTallTileRows;
constexpr int kMinTileCols = 16;
constexpr int kMinPassK = 256;
constexpr int kMaxTileMr = 12;
/// Rows and columns of a source copy block (BF16 transpose, INT8 copy).
constexpr int kCopyBlock = 64;
/// M * K from which the INT8 source copy runs in parallel.
constexpr int64_t kInt8CopyParMin = int64_t {1} << 16;

int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

struct native_tile_plan_t {
    int nr = 0;
    int mr = kTileRows;
    int mb = 0;
    int m_blocks = 0;
    int n_tiles = 0;
    int kc = 0;
    int threads = 1;
    int col_groups = 0;

    int tiles() const { return m_blocks * n_tiles; }
};

native_tile_plan_t plan_native_tiles(int M, int N, int K, int pack_nr,
        int threads, bool bf16, const nat::UarchParams &uarch) {
    native_tile_plan_t p;
    p.nr = pack_nr;
    p.mb = std::min(M, kBlockRows);
    auto count = [&p, M, N]() {
        p.m_blocks = ceil_div(M, p.mb);
        p.n_tiles = N / p.nr;
        return p.tiles();
    };
    while (count() < threads && p.m_blocks == 1 && p.nr > kMinTileCols)
        p.nr /= 2;
    if (bf16 && p.nr == 32) p.mr = kTallTileRows;
    if (count() < threads && p.mb > p.mr) {
        const int blocks = ceil_div(threads, p.n_tiles);
        p.mb = std::max(p.mr, ceil_div(ceil_div(M, blocks), p.mr) * p.mr);
        count();
    }
    p.threads = std::max(1, std::min(threads, p.tiles()));

    p.kc = K;
    if (bf16 && p.mb > p.mr) {
        const int kc = (uarch.l2_bytes / 2)
                / (p.nr * static_cast<int>(sizeof(uint16_t)));
        if (kc < K) p.kc = std::max(kMinPassK, kc & ~1);
    }

    const int64_t domain_bytes
            = (static_cast<int64_t>(M)
                      + static_cast<int64_t>(uarch.ccx_cores) * p.nr)
            * K * (bf16 ? 2 : 1);
    if (p.m_blocks > 1 && p.n_tiles % p.threads != 0 && uarch.ccx_cores > 1
            && domain_bytes > uarch.l3_bytes_per_ccd)
        p.col_groups
                = std::gcd(std::gcd(p.n_tiles, p.threads), uarch.ccx_cores);
    return p;
}

/// Accumulator floats per thread.
size_t scratch_floats(const native_tile_plan_t &p, int K, bool bf16) {
    return static_cast<size_t>(bf16 && p.kc >= K ? p.mr : p.mb) * p.nr;
}

/// INT8 call width with left pack columns of the tile still to compute.
int int8_col_width(int left) {
    return left >= 32 ? 32 : 16;
}

/// Rows per INT8 call at NR = 16 * nv.
int int8_max_mr(int nv) {
    return std::min(kMaxTileMr, nat::native_int8_ukernel_max_mr(nv));
}

int k_pass_quads(int k_quads, int cw, int mr, int max_mr,
        const nat::UarchParams &uarch) {
    const int64_t quad_bytes = static_cast<int64_t>(4) * (cw + max_mr);
    const int64_t budget = uarch.l2_bytes / 2;
    if (mr <= max_mr || k_quads * quad_bytes <= budget) return k_quads;
    const int64_t per = std::max<int64_t>(1, budget / quad_bytes);
    const int64_t passes = (k_quads + per - 1) / per;
    return static_cast<int>((k_quads + passes - 1) / passes);
}

} // namespace frozen

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

/// The first plan value that differs from the value it must have.
class first_mismatch_t {
public:
    void check(const char *field, int64_t got, int64_t want) {
        if (field_ != nullptr || got == want) return;
        field_ = field;
        got_ = got;
        want_ = want;
    }
    bool ok() const { return field_ == nullptr; }
    std::string text() const {
        return std::string(field_) + " " + std::to_string(got_)
                + ", reference copy " + std::to_string(want_);
    }

private:
    const char *field_ = nullptr;
    int64_t got_ = 0;
    int64_t want_ = 0;
};

/// What the reference copy reads for uarch: its ccx_cores is the L3 domain,
/// which is l3_domain_cores when known.
nat::UarchParams frozen_input(const nat::UarchParams &uarch) {
    nat::UarchParams u = uarch;
    if (uarch.l3_domain_cores > 0) u.ccx_cores = uarch.l3_domain_cores;
    return u;
}

/// The parameters of the legacy preset.
nat::heuristic_params_t legacy_params() {
    nat::heuristic_params_t hp;
    hp.preset = nat::native_preset_t::legacy;
    return hp;
}

/// Preset v2 with every v2 rule switched off.
nat::heuristic_params_t v2_rules_off() {
    nat::heuristic_params_t hp;
    hp.preset = nat::native_preset_t::v2;
    hp.makespan = 0;
    hp.l1_pass = 0;
    hp.traffic_groups = 0;
    return hp;
}

/// The legacy plan q of p against the reference copy's choices.
first_mismatch_t compare_with_frozen(const nat::native_plan_t &q,
        const nat::native_problem_t &p, const nat::UarchParams &plan_uarch) {
    const bool bf16 = p.bf16();
    const nat::UarchParams uarch = frozen_input(plan_uarch);
    const frozen::native_tile_plan_t f = frozen::plan_native_tiles(
            p.M, p.N, p.K, p.pack_nr, p.threads, bf16, uarch);
    constexpr unsigned kHeuristicBit = 1u
            << static_cast<unsigned>(nat::native_plan_source_t::heuristic);
    first_mismatch_t m;
    m.check("nr", q.nr, f.nr);
    m.check("mr", q.mr, f.mr);
    m.check("mb", q.mb, f.mb);
    m.check("m_blocks", q.m_blocks, f.m_blocks);
    m.check("n_tiles", q.n_tiles, f.n_tiles);
    m.check("kc", q.kc, f.kc);
    m.check("threads", q.threads, f.threads);
    m.check("col_groups", q.col_groups, f.col_groups);
    m.check("row_unit", q.row_unit, frozen::kTallTileRows);
    m.check("traversal", static_cast<int>(q.traversal),
            static_cast<int>(nat::native_traversal_t::m_inner));
    m.check("scratch_floats", static_cast<int64_t>(q.scratch_floats),
            static_cast<int64_t>(frozen::scratch_floats(f, p.K, bf16)));
    m.check("int8_rows_nv1", q.int8_rows_nv1, frozen::int8_max_mr(1));
    m.check("int8_rows_nv2", q.int8_rows_nv2, frozen::int8_max_mr(2));
    m.check("int8_cw_mask", q.int8_cw_mask, nat::kNativeInt8CwAll);
    m.check("int8_pass_budget", q.int8_pass_budget, uarch.l2_bytes / 2);
    m.check("int8_pass_quads", q.int8_pass_quads, 0);
    m.check("int8_direct_store", q.int8_direct_store, 1);
    m.check("int8_bias_in_dequant", q.int8_bias_in_dequant, 1);
    m.check("copy_block", q.copy_block, frozen::kCopyBlock);
    m.check("copy_par_min", q.copy_par_min, bf16 ? 0 : frozen::kInt8CopyParMin);
    m.check("sources other than the heuristic", q.sources & ~kHeuristicBit, 0);
    m.check("entry pointers",
            q.rules_entry != nullptr || q.table_entry != nullptr
                    || q.override_entry != nullptr,
            0);
    if (bf16) return m;
    // Every call width of a tile, and row counts on both sides of the
    // one-call limit of each width.
    const int k_quads = (p.K + 3) / 4;
    const int last_rows = p.M - (q.m_blocks - 1) * q.mb;
    for (int c = 0; c < q.nr;) {
        const int cw = frozen::int8_col_width(q.nr - c);
        m.check("int8 call width", nat::native_int8_col_width(q, q.nr - c), cw);
        const int nv = cw / 16;
        const int max_mr = frozen::int8_max_mr(nv);
        m.check("int8 rows per call", nat::native_int8_call_rows(q, nv),
                max_mr);
        for (int rows : {1, max_mr, max_mr + 1, last_rows, q.mb}) {
            if (rows < 1 || rows > q.mb) continue;
            m.check("int8 K pass quads",
                    nat::native_int8_pass_quads(q, k_quads, cw, rows, max_mr),
                    frozen::k_pass_quads(k_quads, cw, rows, max_mr, uarch));
        }
        c += cw;
    }
    return m;
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

/// The legacy plan of p against the reference copy, the per-problem rules,
/// preset v2 with its rules off, and a planner without sources against the
/// built-in plan.
void check_built_in(const nat::native_problem_t &p,
        const nat::UarchParams &uarch, const nat::native_planner_t &planner,
        report_t &report) {
    const nat::native_plan_t q = nat::heuristic_plan(p, uarch, legacy_params());
    const first_mismatch_t m = compare_with_frozen(q, p, uarch);
    report.check(m.ok(), [&] { return describe(p) + ": " + m.text(); });
    // The kernel registry offers a dtype only on a CPU with its ISA.
    const nat::UarchParams &host = nat::detect_uarch();
    if (p.bf16() ? host.avx512bf16 : host.avx512vnni) {
        const char *why = nat::validate_native_plan(q, p);
        report.check(why == nullptr, [&] {
            return describe(p) + ": validate_native_plan rejects it: " + why;
        });
    }
    const char *off
            = plan_diff(nat::heuristic_plan(p, uarch, v2_rules_off()), q);
    report.check(off == nullptr, [&] {
        return describe(p) + ": preset v2 with its rules off differs in " + off;
    });
    const char *field = plan_diff(planner.plan(p),
            nat::heuristic_plan(p, uarch, nat::heuristic_params_t {}));
    report.check(field == nullptr, [&] {
        return describe(p) + ": the planner without sources differs in "
                + field;
    });
}

/// check_built_in over the grid at one pack width.
void check_grid(int pack_nr, const nat::UarchParams &uarch,
        const nat::native_planner_t &planner, report_t &report) {
    const std::vector<shape_t> shapes = grid_shapes();
    const std::vector<int> ms = grid_ms();
    for (size_t si = 0; si < shapes.size(); ++si) {
        if (shapes[si].N % pack_nr != 0) continue;
        for (size_t mi = 0; mi < ms.size(); ++mi)
            for (size_t ti = 0; ti < std::size(kThreads); ++ti)
                for (bool bf16 : {true, false})
                    check_built_in(make_problem(ms[mi], shapes[si].N,
                                           shapes[si].K, pack_nr, kThreads[ti],
                                           bf16, si * 5 + mi * 3 + ti),
                            uarch, planner, report);
    }
}

struct uarch_case_t {
    std::string name;
    int l2_kib = 0; ///< 0: the micro-architecture the process plans with
    int l3_mib = 0;
    int ccx_cores = 0;
    int l3_domain_cores = 0;
};

void PrintTo(const uarch_case_t &c, std::ostream *os) {
    *os << c.name;
}

std::vector<uarch_case_t> uarch_cases() {
    std::vector<uarch_case_t> v(1);
    v[0].name = "detected";
    // L3 domain unknown: ccx_cores is the domain.
    for (int l2 : {512, 1024, 2048})
        for (int l3 : {16, 32, 96})
            for (int ccx : {1, 8, 16}) {
                uarch_case_t c;
                c.name = "l2_" + std::to_string(l2) + "k_l3_"
                        + std::to_string(l3) + "m_ccx" + std::to_string(ccx);
                c.l2_kib = l2;
                c.l3_mib = l3;
                c.ccx_cores = ccx;
                v.push_back(c);
            }
    // L3 domain from the topology, over a ccx_cores of 1.
    for (int l2 : {1024, 2048})
        for (int l3 : {32, 128, 256})
            for (int domain : {8, 32, 64}) {
                uarch_case_t c;
                c.name = "l2_" + std::to_string(l2) + "k_l3_"
                        + std::to_string(l3) + "m_dom" + std::to_string(domain);
                c.l2_kib = l2;
                c.l3_mib = l3;
                c.ccx_cores = 1;
                c.l3_domain_cores = domain;
                v.push_back(c);
            }
    return v;
}

nat::UarchParams uarch_of(const uarch_case_t &c) {
    nat::UarchParams u = nat::native_plan_uarch();
    if (c.l2_kib == 0) return u;
    u.l2_bytes = c.l2_kib * 1024;
    u.l3_bytes_per_ccd = c.l3_mib * 1024 * 1024;
    u.ccx_cores = c.ccx_cores;
    u.l3_domain_cores = c.l3_domain_cores;
    return u;
}

class NativeTuningPlanFrozen : public ::testing::TestWithParam<uarch_case_t> {};

TEST_P(NativeTuningPlanFrozen, BuiltInPlanMatchesFrozenCopy) {
    const nat::UarchParams uarch = uarch_of(GetParam());
    const nat::native_planner_t planner(uarch, nullptr, nullptr, nullptr);
    report_t report;
    check_grid(nat::kNativeLayoutPackNr, uarch, planner, report);
    report.finish();
    EXPECT_FALSE(planner.has_sources());
}

INSTANTIATE_TEST_SUITE_P(NativeTuning, NativeTuningPlanFrozen,
        ::testing::ValuesIn(uarch_cases()),
        [](const ::testing::TestParamInfo<uarch_case_t> &info) {
            return info.param.name;
        });

TEST(NativeTuningPlan, OtherPackWidthsMatchFrozenCopy) {
    const nat::UarchParams &uarch = nat::native_plan_uarch();
    const nat::native_planner_t planner(uarch, nullptr, nullptr, nullptr);
    report_t report;
    check_grid(16, uarch, planner, report);
    report.finish();
}

TEST(NativeTuningPlan, DefaultParamsAreTheBuiltInConstants) {
    const nat::heuristic_params_t hp;
    EXPECT_EQ(hp.nr, 0);
    EXPECT_EQ(hp.mr, 0);
    EXPECT_EQ(hp.mb, 0);
    EXPECT_EQ(hp.kc, 0);
    EXPECT_EQ(hp.col_groups, -1);
    EXPECT_EQ(hp.schedule, nat::native_schedule_t::automatic);
    EXPECT_EQ(hp.traversal, nat::native_traversal_t::m_inner);
    EXPECT_EQ(hp.threads_cap, 0);
    EXPECT_EQ(hp.min_tiles_per_thread, 1);
    EXPECT_EQ(hp.row_unit, frozen::kTallTileRows);
    EXPECT_EQ(hp.tile_rows, frozen::kTileRows);
    EXPECT_EQ(hp.tall_rows, frozen::kTallTileRows);
    EXPECT_EQ(hp.block_rows, frozen::kBlockRows);
    EXPECT_EQ(hp.min_tile_cols, frozen::kMinTileCols);
    EXPECT_TRUE(hp.row_split);
    EXPECT_EQ(hp.min_pass_k, frozen::kMinPassK);
    EXPECT_EQ(hp.bf16_l2_pct, 50);
    EXPECT_EQ(hp.kc_split, nat::native_kc_split_t::budget);
    EXPECT_EQ(hp.l3_pct, 100);
    EXPECT_EQ(hp.domain_threads, 0);
    EXPECT_EQ(hp.int8_mr, frozen::kMaxTileMr);
    EXPECT_EQ(hp.int8_cw_mask, nat::kNativeInt8CwAll);
    EXPECT_EQ(hp.int8_l2_pct, 50);
    EXPECT_EQ(hp.int8_kc, 0);
    EXPECT_TRUE(hp.direct_store);
    EXPECT_TRUE(hp.bias_in_dequant);
    EXPECT_EQ(hp.copy_block, frozen::kCopyBlock);
    EXPECT_EQ(hp.bf16_copy_par_min, 0);
    EXPECT_EQ(hp.int8_copy_par_min, frozen::kInt8CopyParMin);
    // The built-in plan is the v2 preset; its rules follow it.
    EXPECT_EQ(hp.preset, nat::native_preset_t::v2);
    EXPECT_EQ(hp.makespan, -1);
    EXPECT_EQ(hp.min_gain_pct, 5);
    EXPECT_EQ(hp.idle_pct, 10);
    EXPECT_EQ(hp.decode_m, 128);
    EXPECT_EQ(hp.l1_pass, -1);
    EXPECT_EQ(hp.l1_pct, 100);
    EXPECT_EQ(hp.l1_pass_max_mb, 64);
    EXPECT_EQ(hp.traffic_groups, -1);
}

TEST(NativeTuningPlan, Int8HelpersMatchFrozenRules) {
    // The rows per call the built-in plan reproduces.
    EXPECT_EQ(frozen::int8_max_mr(1), 12);
    EXPECT_EQ(frozen::int8_max_mr(2), 12);
    report_t report;
    for (int l2_kib : {256, 512, 1000, 1024, 2048}) {
        nat::UarchParams uarch = nat::detect_uarch();
        uarch.l2_bytes = l2_kib * 1024;
        const nat::native_problem_t p = make_problem(
                64, 256, 4096, nat::kNativeLayoutPackNr, 1, false, 0);
        const nat::native_plan_t q
                = nat::heuristic_plan(p, uarch, legacy_params());
        for (int left = 16; left <= 256; left += 16) {
            const int cw = nat::native_int8_col_width(q, left);
            report.check(cw == frozen::int8_col_width(left), [&] {
                return std::to_string(left) + " columns left: call width "
                        + std::to_string(cw);
            });
        }
        for (int nv : {1, 2}) {
            const int rows = nat::native_int8_call_rows(q, nv);
            report.check(rows == frozen::int8_max_mr(nv), [&] {
                return "nv " + std::to_string(nv) + ": rows per call "
                        + std::to_string(rows);
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
                        const int want = frozen::k_pass_quads(
                                k_quads, cw, rows, max_mr, uarch);
                        report.check(got == want, [&] {
                            std::ostringstream os;
                            os << "K pass quads, L2 " << l2_kib
                               << " KiB k_quads " << k_quads << " cw " << cw
                               << " rows " << rows << " max_mr " << max_mr
                               << ": " << got << ", reference copy " << want;
                            return os.str();
                        });
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

/// A machine with l3_threads logical CPUs on each L3 of l3_mib MiB, smt per
/// core (0: unknown), whose CPU family gives ccx_cores.
nat::native_hw_t topology(int l3_threads, int smt, int l3_mib, int ccx_cores) {
    nat::native_hw_t hw;
    hw.uarch = nat::detect_uarch();
    hw.uarch.l2_bytes = 1 << 20;
    hw.uarch.l3_bytes_per_ccd = l3_mib << 20;
    hw.uarch.ccx_cores = ccx_cores;
    hw.logical_cpus = 512;
    hw.smt = smt;
    hw.physical_cores = smt > 0 ? hw.logical_cpus / smt : 0;
    hw.l3_domain_threads = l3_threads;
    return hw;
}

TEST(NativeTuningPlanDomain, CoresPerL3FromTheTopology) {
    const struct {
        int l3_threads, smt, ccx_cores, cores;
    } cases[] = {{64, 2, 8, 32}, {64, 2, 1, 32}, {32, 1, 8, 32}, {16, 2, 8, 8},
            {16, 2, 1, 8}, {16, 0, 8, 16}, {1, 2, 8, 1}, {0, 2, 8, 0},
            {0, 0, 1, 0}};
    for (const auto &c : cases) {
        SCOPED_TRACE("l3_threads " + std::to_string(c.l3_threads) + " smt "
                + std::to_string(c.smt) + " ccx_cores "
                + std::to_string(c.ccx_cores));
        const nat::native_hw_t hw
                = topology(c.l3_threads, c.smt, 128, c.ccx_cores);
        const nat::UarchParams u = nat::native_plan_uarch(hw);
        EXPECT_EQ(u.l3_domain_cores, c.cores);
        EXPECT_EQ(u.ccx_cores, c.ccx_cores);
        EXPECT_EQ(u.l1d_bytes, hw.uarch.l1d_bytes);
        EXPECT_EQ(u.l2_bytes, hw.uarch.l2_bytes);
        EXPECT_EQ(u.l3_bytes_per_ccd, hw.uarch.l3_bytes_per_ccd);
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
    EXPECT_EQ(plan.ccx_cores, host.ccx_cores);
    EXPECT_EQ(plan.l2_bytes, host.l2_bytes);
    EXPECT_EQ(plan.l3_bytes_per_ccd, host.l3_bytes_per_ccd);
}

TEST(NativeTuningPlanDomain, RowGroupsFollowTheDomain) {
    const nat::UarchParams wide
            = nat::native_plan_uarch(topology(64, 2, 128, 8));
    const nat::UarchParams wide_no_ccx
            = nat::native_plan_uarch(topology(64, 2, 128, 1));
    const nat::UarchParams narrow
            = nat::native_plan_uarch(topology(16, 2, 128, 8));
    const nat::UarchParams unknown
            = nat::native_plan_uarch(topology(0, 2, 128, 8));
    ASSERT_EQ(wide.l3_domain_cores, 32);
    ASSERT_EQ(wide_no_ccx.l3_domain_cores, 32);
    ASSERT_EQ(narrow.l3_domain_cores, 8);
    ASSERT_EQ(unknown.l3_domain_cores, 0);
    const nat::UarchParams &host = nat::detect_uarch();
    const nat::heuristic_params_t legacy = legacy_params();
    nat::heuristic_params_t pin8 = legacy, pin32 = legacy;
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
        const nat::native_plan_t w = nat::heuristic_plan(p, wide, legacy);
        const nat::native_plan_t n = nat::heuristic_plan(p, narrow, legacy);
        EXPECT_EQ(w.col_groups, c.wide_groups) << describe(p);
        EXPECT_EQ(n.col_groups, c.narrow_groups) << describe(p);
        if (p.bf16() ? host.avx512bf16 : host.avx512vnni) {
            EXPECT_EQ(nat::validate_native_plan(w, p), nullptr) << describe(p);
            EXPECT_EQ(nat::validate_native_plan(n, p), nullptr) << describe(p);
        }
        // The family's ccx_cores counts only when the domain is unknown.
        const char *field
                = plan_diff(nat::heuristic_plan(p, wide_no_ccx, legacy), w);
        EXPECT_EQ(field, nullptr) << "ccx_cores 1 under a known domain";
        field = plan_diff(nat::heuristic_plan(p, unknown, legacy), n);
        EXPECT_EQ(field, nullptr) << "unknown domain, ccx_cores 8";
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
                        "preset=legacy,domain_threads=8"),
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
                        const nat::native_plan_t q
                                = nat::heuristic_plan(p, u, legacy_params());
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
                        const first_mismatch_t m = compare_with_frozen(q, p, u);
                        report.check(m.ok(),
                                [&] { return where() + ": " + m.text(); });
                        const char *field = plan_diff(planner.plan(p),
                                nat::heuristic_plan(
                                        p, u, nat::heuristic_params_t {}));
                        report.check(field == nullptr, [&] {
                            return where() + ": the planner differs in "
                                    + field;
                        });
                    }
    }
    report.finish();
}

} // namespace
