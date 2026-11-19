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

// Edge cases of the native BRGEMM that must neither crash nor return wrong
// results: plans for every statically legal pin, partitions of unvalidated
// plans, malformed tuning files, weight buffers rewritten in place that are
// later overwritten, read through overlapping views or reached by calls
// selecting other algorithms, INT8 zero points and dynamic quantization
// settings at the edges of what the INT8 tile computes exactly, the rounding of
// the epilogue and special values on the BF16 path, the OpenMP team of a call,
// weight packs that must not hold up the lookups of other weights, and a bias
// read in its own type at the tile.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <omp.h>
#include <xmmintrin.h>

#if defined(__linux__)
#include <unistd.h>
#include <sys/mman.h>
#endif

#include "gtest_utils.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/bf16/bf16_brgemm_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_brgemm_looper.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_int8_tile.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_partition.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_config.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_planner.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_tuning_file.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_tuning_keys.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_epilogue.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_hw.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_pack.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_weight_cache.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_weight_view.hpp"
#include "lowoha_operators/matmul/matmul_native/native_layout_policy.hpp"
#include "native_brgemm_test_utils.hpp"

namespace {

namespace nat = zendnnl::lowoha::matmul::native;
namespace nt = native_brgemm_test;
using zendnnl::error_handling::status_t;
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

class NativeBrgemmHardeningPlan : public ::testing::Test {
protected:
    void SetUp() override {
        const nat::UarchParams &u = nat::detect_uarch();
        if (!u.avx512bf16 || !u.avx512vnni)
            GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    }
};

// A row-group plan whose col_groups does not divide the team runs as a
// slice partition and still covers every tile once.
TEST(NativeBrgemmHardeningPartition, ColGroupsNotDividingWorkersRunsSlice) {
    nat::native_plan_t plan;
    plan.nr = 32;
    plan.mr = 12;
    plan.mb = 24;
    plan.m_blocks = 5;
    plan.n_tiles = 8;
    plan.threads = 2;
    plan.col_groups = 4;
    plan.row_unit = 12;
    const int M = 100;
    const nat::native_partition_t part(plan, M, 0, plan.n_tiles);
    EXPECT_EQ(part.kind(2), nat::native_partition_kind_t::slice);
    std::vector<int> cover(static_cast<size_t>(M) * plan.n_tiles, 0);
    for (int w = 0; w < 2; ++w)
        part.for_each(w, 2, [&](const nat::native_work_unit_t &u) {
            for (int r = u.m0; r < u.m0 + u.rows; ++r)
                ++cover[static_cast<size_t>(r) * plan.n_tiles + u.n0 / plan.nr];
        });
    EXPECT_TRUE(std::all_of(
            cover.begin(), cover.end(), [](int c) { return c == 1; }));
}

const char kTuningHead[]
        = R"({"format": "zendnnl-native-brgemm-tuning", "version": 2, )";

/// The sources of a tuning file of the given text, on this CPU.
nat::native_tuning_sources_t load_text(const std::string &text) {
    const nt::temp_file_t file("hardening.json", text);
    return nat::load_native_tuning_file(file.path(), nat::native_hw());
}

// Nesting far past the limit, anywhere in the file, makes the whole file
// ignored before it is parsed.
TEST(NativeBrgemmHardeningTuningFile, DeeplyNestedFileIsIgnored) {
    const std::string deep
            = std::string(100000, '[') + std::string(100000, ']');
    const std::string head = kTuningHead;
    const std::string files[] = {
            head + R"("entries": [{"match": {"m": )" + deep + "}}]}",
            head + R"("entries": [{"config": {"nr": )" + deep + "}}]}",
            head + R"("hardware": {"isa": )" + deep + "}}",
            head + R"("rules": {"kind": "ordered", "list": [{"when": )" + deep
                    + R"(, "config": {"nr": 32}}]}})",
    };
    for (size_t i = 0; i < std::size(files); ++i) {
        const nat::native_tuning_sources_t s = load_text(files[i]);
        EXPECT_EQ(s.table, nullptr) << "file " << i;
        EXPECT_EQ(s.rules, nullptr) << "file " << i;
    }
    // Brackets inside a string do not nest.
    const std::string in_string = head + R"("entries": [{"id": ")"
            + std::string(1000, '[')
            + R"(", "match": {"n": 64}, "config": {"nr": 32}}]})";
    const nat::native_tuning_sources_t s = load_text(in_string);
    EXPECT_NE(s.table, nullptr);
}

// A file with more values than the loader keeps a parsed document of is
// ignored before it is parsed, whatever its size in bytes.
TEST(NativeBrgemmHardeningTuningFile, FileWithTooManyValuesIsIgnored) {
    std::string many;
    many.reserve(3 << 21);
    for (size_t i = 0; i < (size_t {1} << 21) + 8; ++i)
        many += "{},";
    many += "{}";
    const nat::native_tuning_sources_t s = load_text(
            std::string(kTuningHead) + R"("entries": [)" + many + "]}");
    EXPECT_EQ(s.table, nullptr);
    EXPECT_EQ(s.rules, nullptr);
}

// A nested value an entry cannot use drops that entry only, and the warning
// names its size instead of printing it.
TEST(NativeBrgemmHardeningTuningFile, NestedBadValueDropsOnlyItsEntry) {
    const std::string nested = std::string(100, '[') + std::string(100, ']');
    const std::string text = std::string(kTuningHead) + R"("entries": [)"
            + R"({"match": {"m": )" + nested + R"(}, "config": {"nr": 32}},)"
            + R"({"config": {"mb": )" + nested + R"(}},)"
            + R"({"match": {"n": 64}, "config": {"nr": 32}}]})";
    ::testing::internal::CaptureStdout();
    const nat::native_tuning_sources_t s = load_text(text);
    const std::string log = ::testing::internal::GetCapturedStdout();
    ASSERT_NE(s.table, nullptr);
    nat::native_problem_t p = problem(8, 64, 64, 32, 8, true);
    const nat::native_config_entry_t *e = s.table->lookup(p);
    ASSERT_NE(e, nullptr);
    EXPECT_NE(e->label.find(" entry 2"), std::string::npos) << e->label;
    if (zendnnl::error_handling::apilog_warning_enabled()) {
        EXPECT_NE(log.find("<array of 1>"), std::string::npos) << log;
        EXPECT_EQ(log.find("[[[["), std::string::npos) << log;
    }
}

/// The table source of a tuning file with the given entries and rules.
nat::native_tuning_sources_t tuning_sources(
        const std::string &entries, const std::string &rules = "") {
    std::string text
            = std::string(kTuningHead) + R"("entries": [)" + entries + "]";
    if (!rules.empty()) text += R"(, "rules": )" + rules;
    return load_text(text + "}");
}

// Both layers are legal alone and illegal together: the higher layer stays
// and the lower one is dropped.
TEST_F(NativeBrgemmHardeningPlan, HigherLayerWinsWhenBothLegalAlone) {
    const nat::UarchParams &u = nat::detect_uarch();
    const nat::native_problem_t p = problem(256, 4096, 4096, 32, 8, true);
    {
        nat::native_tuning_sources_t t
                = tuning_sources(R"({"config": {"nr": 16}})");
        ASSERT_NE(t.table, nullptr);
        const nat::native_planner_t planner(u,
                nat::make_native_override_source("mr=12"), std::move(t.table),
                nullptr);
        const nat::native_plan_t q = planner.plan(p);
        EXPECT_EQ(nat::validate_native_plan(q, p), nullptr);
        EXPECT_EQ(nat::native_plan_source_names(q), "override");
        EXPECT_EQ(q.mr, 12);
        EXPECT_EQ(q.nr, 32);
        EXPECT_EQ(q.table_entry, nullptr);
        EXPECT_EQ(nat::native_plan_config_text(q), "mr=12");
    }
    {
        nat::native_tuning_sources_t t = tuning_sources(
                R"({"config": {"mr": 12}})",
                R"({"kind": "ordered", "list": [{"config": {"nr": 16}}]})");
        ASSERT_NE(t.table, nullptr);
        ASSERT_NE(t.rules, nullptr);
        const nat::native_planner_t planner(
                u, nullptr, std::move(t.table), std::move(t.rules));
        const nat::native_plan_t q = planner.plan(p);
        EXPECT_EQ(nat::native_plan_source_names(q), "table");
        EXPECT_EQ(q.mr, 12);
        EXPECT_EQ(q.rules_entry, nullptr);
    }
}

// schedule and col_groups merge as one key, and the merged config text of a
// plan parses back.
TEST_F(NativeBrgemmHardeningPlan, ScheduleAndColGroupsMergeAsOneKey) {
    const nat::UarchParams &u = nat::detect_uarch();
    const nat::native_problem_t p = problem(256, 4096, 4096, 32, 8, true);
    struct case_t {
        const char *override_text;
        const char *table_config;
        int col_groups;
    };
    for (const case_t &c : {
                 case_t {"schedule=slice", R"({"col_groups": 2})", 0},
                 case_t {"col_groups=2", R"({"schedule": "slice"})", 2},
                 case_t {"schedule=row_groups", R"({"col_groups": 0})", -1}}) {
        SCOPED_TRACE(c.override_text);
        nat::native_tuning_sources_t t = tuning_sources(
                std::string(R"({"config": )") + c.table_config + "}");
        ASSERT_NE(t.table, nullptr);
        const nat::native_planner_t planner(u,
                nat::make_native_override_source(c.override_text),
                std::move(t.table), nullptr);
        const nat::native_plan_t q = planner.plan(p);
        EXPECT_EQ(nat::validate_native_plan(q, p), nullptr);
        if (c.col_groups >= 0)
            EXPECT_EQ(q.col_groups, c.col_groups);
        else
            EXPECT_GT(q.col_groups, 0);
        const std::string text = nat::native_plan_config_text(q);
        nat::native_config_t back;
        std::string why;
        EXPECT_TRUE(nat::parse_native_config(text, back, why))
                << text << ": " << why;
    }
}

// A pinned BF16 mr above the narrow kernels' rows keeps the tile width those
// rows have kernels at, instead of dropping the layer.
TEST_F(NativeBrgemmHardeningPlan, PinnedTallMrStopsNarrowing) {
    const nat::UarchParams &u = nat::detect_uarch();
    const nat::native_problem_t p = problem(8, 256, 512, 32, 32, true);
    EXPECT_EQ(nat::heuristic_plan(p, u, nat::heuristic_params_t {}).nr, 16);
    const nat::native_planner_t planner(
            u, nat::make_native_override_source("mr=12"), nullptr, nullptr);
    const nat::native_plan_t q = planner.plan(p);
    EXPECT_EQ(nat::validate_native_plan(q, p), nullptr);
    EXPECT_EQ(nat::native_plan_source_names(q), "override");
    EXPECT_EQ(q.mr, 12);
    EXPECT_EQ(q.nr, 32);
}

// ── Weight buffers rewritten in place ────────────────────────────────────

/// A bf16 [K, N] (or [N, K] when trans_b) weight call into an f32 dst.
struct bf16_dense_t {
    int M, N, K;
    bool trans_b = false;
    nt::buffer_t src;
    std::vector<float> dst;

    bf16_dense_t(int m, int n, int k, bool tb) : M(m), N(n), K(k), trans_b(tb) {
        src = nt::random_buffer(kBf16, static_cast<size_t>(M) * K, 11, -1, 1);
        dst.assign(static_cast<size_t>(M) * N, -777.0f);
    }
    nt::call_t call(const void *wei, int32_t wc, matmul_algo_t algo) {
        nt::call_t c;
        c.M = M;
        c.N = N;
        c.K = K;
        c.transB = trans_b;
        c.src = src.get();
        c.wei = wei;
        c.dst = dst.data();
        c.params.dtypes.src = kBf16;
        c.params.dtypes.wei = kBf16;
        c.params.dtypes.dst = data_type_t::f32;
        c.params.num_threads = 8;
        c.params.weight_cache_type = wc;
        c.params.lowoha_algo = algo;
        return c;
    }
    /// dst against x * w computed here from the raw weight w.
    ::testing::AssertionResult matches(const nt::buffer_t &w) const {
        size_t bad = 0;
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) {
                double ref = 0;
                for (int k = 0; k < K; ++k) {
                    const size_t wi = trans_b ? static_cast<size_t>(n) * K + k
                                              : static_cast<size_t>(k) * N + n;
                    ref += static_cast<double>(nt::load_f32(src.get(), kBf16,
                                   static_cast<size_t>(m) * K + k))
                            * nt::load_f32(w.get(), kBf16, wi);
                }
                const float got = dst[static_cast<size_t>(m) * N + n];
                if (!(std::fabs(got - ref) <= 1e-2 + 1e-2 * std::fabs(ref)))
                    ++bad;
            }
        if (bad == 0) return ::testing::AssertionSuccess();
        return ::testing::AssertionFailure()
                << bad << " of " << M * N << " outputs differ from x * w";
    }
};

using zendnnl::common::matmul_algo_t;

class NativeBrgemmHardeningInPlace : public ::testing::Test {
protected:
    void SetUp() override {
        if (!nat::detect_uarch().avx512bf16)
            GTEST_SKIP() << "needs AVX512-BF16";
        ::reset_grp_matmul_caches();
        cache_ = std::make_unique<WeightCacheGuard>(2);
    }
    void TearDown() override {
        algo_.reset();
        cache_.reset();
        ::reset_grp_matmul_caches();
    }
    void pin(matmul_algo_t algo) {
        algo_.reset();
        algo_ = std::make_unique<nt::algo_pin_t>(algo);
    }

private:
    std::unique_ptr<WeightCacheGuard> cache_;
    std::unique_ptr<nt::algo_pin_t> algo_;
};

nt::buffer_t raw_bf16(size_t n, uint32_t seed) {
    return nt::random_buffer(kBf16, n, seed, -1.0f, 1.0f);
}

nt::buffer_t copy_buffer(const nt::buffer_t &b) {
    nt::buffer_t c = nt::make_buffer(b.bytes);
    std::memcpy(c.get(), b.get(), b.bytes);
    return c;
}

// A buffer rewritten in place and then overwritten with another raw weight
// (a reload, or the memory freed and reused) is read as that raw weight.
TEST_F(NativeBrgemmHardeningInPlace, OverwrittenImageIsReadAsRaw) {
    pin(matmul_algo_t::native_brgemm);
    for (bool trans_b : {false, true}) {
        SCOPED_TRACE(trans_b ? "transB" : "no transB");
        bf16_dense_t d(4, 64, 64, trans_b);
        const nt::buffer_t a = raw_bf16(64 * 64, 21);
        const nt::buffer_t b = raw_bf16(64 * 64, 22);
        nt::buffer_t w = copy_buffer(a);
        ASSERT_EQ(nt::run_matmul(d.call(w.get(), 2, matmul_algo_t::none)),
                status_t::success);
        EXPECT_TRUE(d.matches(a));
        ASSERT_TRUE(nat::native_weight_packed_in_place(w.get(), w.bytes));
        std::memcpy(w.get(), b.get(), b.bytes);
        EXPECT_FALSE(nat::native_weight_packed_in_place(w.get(), w.bytes));
        ASSERT_EQ(nt::run_matmul(d.call(w.get(), 2, matmul_algo_t::none)),
                status_t::success);
        EXPECT_EQ(nt::native_layouts(w.get()), 1u) << "the image of b";
        EXPECT_TRUE(d.matches(b));
        // Rewritten again, for b.
        EXPECT_TRUE(nat::native_weight_packed_in_place(w.get(), w.bytes));
        ASSERT_EQ(nt::run_matmul(d.call(w.get(), 2, matmul_algo_t::none)),
                status_t::success);
        EXPECT_TRUE(d.matches(b));
    }
}

// A request that starts below an image and reads into it fails instead of
// reading the image bytes as raw weight.
TEST_F(NativeBrgemmHardeningInPlace, ViewReadingIntoImageFails) {
    pin(matmul_algo_t::native_brgemm);
    const int K = 64;
    const nt::buffer_t raw = raw_bf16(2 * 64 * K, 31);
    nt::buffer_t w = copy_buffer(raw);
    bf16_dense_t upper(4, 64, K, true);
    const void *upper_w = w.as<uint16_t>() + 64 * K;
    ASSERT_EQ(nt::run_matmul(upper.call(upper_w, 2, matmul_algo_t::none)),
            status_t::success);
    ASSERT_TRUE(nat::native_weight_packed_in_place(
            upper_w, w.bytes - 64 * K * sizeof(uint16_t)));
    bf16_dense_t whole(4, 128, K, true);
    EXPECT_NE(nt::run_matmul(whole.call(w.get(), 2, matmul_algo_t::none)),
            status_t::success);
    EXPECT_TRUE(std::all_of(whole.dst.begin(), whole.dst.end(),
            [](float v) { return v == -777.0f; }));
}

// Selecting the native BRGEMM per call never rewrites the buffer: other
// calls on it may take other routes.
TEST_F(NativeBrgemmHardeningInPlace, PerCallSelectionNeverRewritesInPlace) {
    pin(matmul_algo_t::none);
    bf16_dense_t d(4, 64, 64, false);
    const nt::buffer_t a = raw_bf16(64 * 64, 41);
    nt::buffer_t w = copy_buffer(a);
    ASSERT_EQ(nt::run_matmul(d.call(w.get(), 2, matmul_algo_t::native_brgemm)),
            status_t::success);
    EXPECT_EQ(nt::native_layouts(w.get()), 1u) << "one layout, cached";
    EXPECT_TRUE(d.matches(a));
    EXPECT_FALSE(nat::native_weight_packed_in_place(w.get(), w.bytes));
    EXPECT_EQ(std::memcmp(w.get(), a.get(), a.bytes), 0);
}

// Under ALGO 10 a call that selects another algorithm on an image is served
// from the image.
TEST_F(NativeBrgemmHardeningInPlace, OtherAlgorithmOnImageIsServedFromIt) {
    pin(matmul_algo_t::native_brgemm);
    bf16_dense_t d(4, 64, 64, false);
    const nt::buffer_t a = raw_bf16(64 * 64, 51);
    nt::buffer_t w = copy_buffer(a);
    ASSERT_EQ(nt::run_matmul(d.call(w.get(), 2, matmul_algo_t::none)),
            status_t::success);
    ASSERT_TRUE(nat::native_weight_packed_in_place(w.get(), w.bytes));
    for (matmul_algo_t other :
            {matmul_algo_t::aocl_dlp_blocked, matmul_algo_t::onednn_blocked}) {
        SCOPED_TRACE(static_cast<int>(other));
        std::fill(d.dst.begin(), d.dst.end(), -777.0f);
        ASSERT_EQ(nt::run_matmul(d.call(w.get(), 2, other)), status_t::success);
        // Another route would read the image as a raw weight.
        EXPECT_TRUE(d.matches(a));
        EXPECT_TRUE(nat::native_weight_packed_in_place(w.get(), w.bytes));
    }
}

// ── INT8 zero points and dynamic quantization ────────────────────────────

using quant_t
        = zendnnl::lowoha::matmul::matmul_quantization_params_t::matmul_quant_t;

void set_quant(quant_t &q, const void *buff, data_type_t dt,
        std::vector<int64_t> dims) {
    q.buff = buff;
    q.dt = dt;
    q.dims = std::move(dims);
}

nat::GemmDescriptor int8_desc(int M, int N, int K, data_type_t src) {
    nat::GemmDescriptor d;
    d.M = M;
    d.N = N;
    d.K = K;
    d.lda = K;
    d.ldb = N;
    d.ldc = N;
    d.src_dt = src;
    d.wei_dt = kS8;
    d.dst_dt = data_type_t::f32;
    d.num_threads = 4;
    return d;
}

zendnnl::lowoha::matmul::matmul_params int8_params(data_type_t src) {
    zendnnl::lowoha::matmul::matmul_params p;
    p.dtypes.src = src;
    p.dtypes.wei = kS8;
    p.dtypes.dst = data_type_t::f32;
    p.num_threads = 4;
    return p;
}

class NativeBrgemmHardeningInt8 : public ::testing::Test {
protected:
    void SetUp() override {
        if (!nat::detect_uarch().avx512vnni)
            GTEST_SKIP() << "needs AVX512-VNNI";
    }
};

/// x * w through native_brgemm_execute over a raw s8 [K, N] weight, packed
/// for the call (a cached layout is keyed by the weight's address, which a
/// later buffer can reuse).
status_t run_int8(const nat::GemmDescriptor &desc, const void *src,
        const void *wei, float *dst,
        zendnnl::lowoha::matmul::matmul_params &params) {
    nat::native_weight_ref_t ref;
    ref.weight = wei;
    ref.pack_nr = 32;
    ref.weight_cache_mode = 0;
    return nat::native_brgemm_execute(desc, ref, src, dst, nullptr, params);
}

// Zero points outside the source type's range, at depths where the int32
// zero-point term would wrap: the result is sum over K of (a - zp) * w,
// rounded once.
TEST_F(NativeBrgemmHardeningInt8, LargeZeroPointsAreExact) {
    struct case_t {
        data_type_t src, zp_dt;
        int32_t zp;
        int K;
    };
    const case_t cases[] = {
            {data_type_t::u8, data_type_t::s32, 300, 256},
            {data_type_t::u8, data_type_t::s32, 100000, 4096},
            {data_type_t::u8, data_type_t::s32, -5, 8192},
            {data_type_t::u8, kS8, -128, 8192},
            {kS8, data_type_t::s32, 128, 8192},
            {kS8, data_type_t::s32, -70000, 4096},
            {kS8, data_type_t::u8, 255, 8192},
    };
    const int M = 2, N = 32;
    for (const case_t &c : cases) {
        SCOPED_TRACE("src " + std::to_string(static_cast<int>(c.src)) + " zp "
                + std::to_string(c.zp) + " K " + std::to_string(c.K));
        const bool sgn = c.src == kS8;
        const nt::buffer_t src
                = nt::random_buffer(c.src, static_cast<size_t>(M) * c.K, 71,
                        sgn ? -128 : 0, sgn ? 127 : 255);
        const nt::buffer_t wei = nt::random_buffer(
                kS8, static_cast<size_t>(c.K) * N, 72, -127, 127);
        const int8_t zp8 = static_cast<int8_t>(c.zp);
        const uint8_t zpu8 = static_cast<uint8_t>(c.zp);
        const void *zp = c.zp_dt == data_type_t::s32
                ? static_cast<const void *>(&c.zp)
                : c.zp_dt == kS8 ? static_cast<const void *>(&zp8)
                                 : static_cast<const void *>(&zpu8);
        auto params = int8_params(c.src);
        set_quant(params.quant_params.src_zp, zp, c.zp_dt, {1, 1});
        std::vector<float> dst(static_cast<size_t>(M) * N, -1.0f);
        ASSERT_EQ(run_int8(int8_desc(M, N, c.K, c.src), src.get(), wei.get(),
                          dst.data(), params),
                status_t::success);
        size_t bad = 0;
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) {
                int64_t ref = 0;
                for (int k = 0; k < c.K; ++k) {
                    const size_t idx = static_cast<size_t>(m) * c.K + k;
                    const int64_t a = sgn ? src.as<int8_t>()[idx]
                                          : src.as<uint8_t>()[idx];
                    ref += (a - c.zp)
                            * wei.as<int8_t>()[static_cast<size_t>(k) * N + n];
                }
                bad += dst[static_cast<size_t>(m) * N + n]
                        != static_cast<float>(ref);
            }
        EXPECT_EQ(bad, 0u);
    }
}

// Asymmetric dynamic quantization of rows whose values exclude 0 puts the
// zero point far outside the byte range; the result stays exact.
TEST_F(NativeBrgemmHardeningInt8, AsymmetricRowsOutsideTheByteRangeAreExact) {
    const int M = 2, N = 32, K = 8192;
    std::vector<uint16_t> src(static_cast<size_t>(M) * K);
    nt::store_f32(src.data(), kBf16, 0, 10.0f);
    std::fill(src.begin(), src.end(), src[0]);
    const std::vector<int8_t> wei(static_cast<size_t>(K) * N, 127);
    std::vector<float> scale(M);
    std::vector<int32_t> zp(M);
    auto params = int8_params(kBf16);
    params.dynamic_quant = true;
    params.dtypes.compute = data_type_t::u8;
    set_quant(params.quant_params.src_scale, scale.data(), data_type_t::f32,
            {M, 1});
    set_quant(params.quant_params.src_zp, zp.data(), data_type_t::s32, {M, 1});
    std::vector<float> dst(static_cast<size_t>(M) * N, -1.0f);
    ASSERT_EQ(run_int8(int8_desc(M, N, K, kBf16), src.data(), wei.data(),
                      dst.data(), params),
            status_t::success);
    EXPECT_LT(zp[0], -255) << "the zero point the quantizer gave";
    const double want = 10.0 * 127 * K;
    for (const float v : dst)
        EXPECT_NEAR(v, want, want * 1e-5);
}

// Dynamic settings the shared quantizer does not take are turned away
// before any work instead of failing the call.
TEST_F(NativeBrgemmHardeningInt8, DynamicSettingsTheQuantizerRejectsDecline) {
    const int M = 4;
    std::vector<float> scale(M);
    std::vector<int32_t> zp32(M);
    std::vector<int8_t> zp8(M);
    struct case_t {
        const char *name;
        data_type_t compute, scale_dt, zp_dt;
        std::vector<int64_t> dims;
        bool served;
    };
    const case_t cases[] = {
            {"per-token {M, 1}", kS8, data_type_t::f32, data_type_t::none,
                    {M, 1}, true},
            {"per-tensor", kS8, data_type_t::f32, data_type_t::none, {1, 1},
                    true},
            {"1-D {M}", kS8, data_type_t::f32, data_type_t::none, {M}, false},
            {"s32 scale", kS8, data_type_t::s32, data_type_t::none, {M, 1},
                    false},
            {"u8 with s32 zero point", data_type_t::u8, data_type_t::f32,
                    data_type_t::s32, {M, 1}, true},
            {"u8 with s8 zero point", data_type_t::u8, data_type_t::f32, kS8,
                    {M, 1}, false},
    };
    for (const case_t &c : cases) {
        SCOPED_TRACE(c.name);
        auto params = int8_params(kBf16);
        params.dynamic_quant = true;
        params.dtypes.compute = c.compute;
        set_quant(params.quant_params.src_scale, scale.data(), c.scale_dt,
                c.dims);
        if (c.zp_dt != data_type_t::none)
            set_quant(params.quant_params.src_zp,
                    c.zp_dt == data_type_t::s32
                            ? static_cast<const void *>(zp32.data())
                            : static_cast<const void *>(zp8.data()),
                    c.zp_dt, c.dims);
        const nat::GemmDescriptor desc = int8_desc(M, 64, 128, kBf16);
        EXPECT_EQ(nat::check_int8_call(desc, params,
                          nat::decide_int8_call(desc, params, nullptr)),
                c.served ? status_t::success : status_t::unimplemented);
    }
}

// A source that is already u8 keeps its zero point when dynamic_quant is
// set: the flag only concerns bf16 / f32 sources.  Both runs give the sum
// over K of (a - zp) * w, exact in f32 at this depth.
TEST_F(NativeBrgemmHardeningInt8, QuantizedSourceKeepsItsZeroPoint) {
    const int M = 5, N = 64, K = 128;
    const int32_t zp = 7;
    const nt::buffer_t src = nt::random_buffer(
            data_type_t::u8, static_cast<size_t>(M) * K, 81, 0, 255);
    const nt::buffer_t wei
            = nt::random_buffer(kS8, static_cast<size_t>(K) * N, 82, -127, 127);
    std::vector<float> plain(static_cast<size_t>(M) * N, -1.0f);
    std::vector<float> flagged(plain.size(), -2.0f);
    for (bool dynamic : {false, true}) {
        auto params = int8_params(data_type_t::u8);
        params.dynamic_quant = dynamic;
        set_quant(params.quant_params.src_zp, &zp, data_type_t::s32, {1, 1});
        ASSERT_EQ(run_int8(int8_desc(M, N, K, data_type_t::u8), src.get(),
                          wei.get(), dynamic ? flagged.data() : plain.data(),
                          params),
                status_t::success);
    }
    EXPECT_EQ(std::memcmp(plain.data(), flagged.data(),
                      plain.size() * sizeof(float)),
            0);
    size_t bad = 0;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            int64_t ref = 0;
            for (int k = 0; k < K; ++k)
                ref += (src.as<uint8_t>()[static_cast<size_t>(m) * K + k] - zp)
                        * wei.as<int8_t>()[static_cast<size_t>(k) * N + n];
            bad += flagged[static_cast<size_t>(m) * N + n]
                    != static_cast<float>(ref);
        }
    EXPECT_EQ(bad, 0u) << "elements other than the integer reference";
}

// ── Epilogue rounding and special values ─────────────────────────────────

float from_bits(uint32_t b) {
    float f;
    std::memcpy(&f, &b, sizeof(f));
    return f;
}

uint32_t bits_of(float f) {
    uint32_t b;
    std::memcpy(&b, &f, sizeof(b));
    return b;
}

bool bf16_is_nan(uint16_t b) {
    return (b & 0x7F80u) == 0x7F80u && (b & 0x007Fu) != 0;
}

class NativeBrgemmHardeningEpilogue : public ::testing::Test {
protected:
    void SetUp() override {
        if (!nat::detect_uarch().avx512bf16)
            GTEST_SKIP() << "needs AVX512-BF16";
    }
};

/// One row of 16 columns of a BF16 call's epilogue into dst.
status_t epilogue_row(float alpha, float beta, data_type_t dst_dt, void *dst,
        float *acc, nat::native_epilogue_t &ep) {
    nat::GemmDescriptor d;
    d.M = 1;
    d.N = 16;
    d.K = 16;
    d.ldc = 16;
    d.alpha = alpha;
    d.beta = beta;
    d.src_dt = kBf16;
    d.wei_dt = kBf16;
    d.dst_dt = dst_dt;
    zendnnl::lowoha::matmul::matmul_params params;
    const status_t st
            = nat::prepare_epilogue(d, params, nullptr, dst, false, ep);
    if (st == status_t::success)
        nat::apply_epilogue_tile(ep, acc, 16, 0, 1, 0, 16);
    return st;
}

// beta * dst is rounded before it is added, as the reference does: an input
// pair whose fused and unfused results differ in the last bit.
TEST_F(NativeBrgemmHardeningEpilogue, BetaRoundsLikeTheReference) {
    const float acc0 = -0x1.9700fcp-1f, dst0 = -0x1.e03e74p+0f, beta = 1.5f;
    volatile float product = dst0 * beta;
    volatile float want = acc0 + product;
    ASSERT_EQ(bits_of(want), bits_of(-0x1.cdef18p+1f));
    std::vector<float> acc(16, acc0), dst(16, dst0);
    nat::native_epilogue_t ep;
    ASSERT_EQ(epilogue_row(
                      1.0f, beta, data_type_t::f32, dst.data(), acc.data(), ep),
            status_t::success);
    for (const float v : dst)
        EXPECT_EQ(bits_of(v), bits_of(want));
}

// The bf16 store keeps every NaN a NaN of its sign, infinities infinite and
// rounds the other values to nearest even.
TEST_F(NativeBrgemmHardeningEpilogue, Bf16StoreKeepsNan) {
    const uint32_t in[16] = {0x7F800001u, 0x7F807FFFu, 0x7FFFFFFFu, 0xFFFFFFFFu,
            0x7FC00000u, 0xFFC00001u, 0x7F800000u, 0xFF800000u, 0x00000001u,
            0x3F800000u, 0x3F808000u, 0x3F818000u, 0xBF800001u, 0x7F7FFFFFu,
            0x00800000u, 0x80000000u};
    std::vector<float> acc(16);
    for (int i = 0; i < 16; ++i)
        acc[i] = from_bits(in[i]);
    std::vector<uint16_t> dst(16, 0x1234);
    nat::native_epilogue_t ep;
    ASSERT_EQ(epilogue_row(1.0f, 0.0f, kBf16, dst.data(), acc.data(), ep),
            status_t::success);
    for (int i = 0; i < 6; ++i) {
        EXPECT_TRUE(bf16_is_nan(dst[i])) << i << ": " << dst[i];
        EXPECT_EQ(dst[i] & 0x8000u, (in[i] >> 16) & 0x8000u) << i;
    }
    const uint16_t want[10] = {0x7F80, 0xFF80, 0x0000, 0x3F80, 0x3F80, 0x3F82,
            0xBF80, 0x7F80, 0x0080, 0x8000};
    for (int i = 6; i < 16; ++i)
        EXPECT_EQ(dst[i], want[i - 6]) << i;
}

/// x * w + bias on the native BRGEMM over a raw bf16 [K, N] weight packed for
/// the call, f32 dst.
status_t run_bf16(int M, int N, int K, const void *src, const void *wei,
        const void *bias, float beta, float *dst, int threads) {
    nat::GemmDescriptor d;
    d.M = M;
    d.N = N;
    d.K = K;
    d.lda = K;
    d.ldb = N;
    d.ldc = N;
    d.beta = beta;
    d.src_dt = kBf16;
    d.wei_dt = kBf16;
    d.dst_dt = data_type_t::f32;
    d.num_threads = threads;
    zendnnl::lowoha::matmul::matmul_params params;
    params.dtypes.src = kBf16;
    params.dtypes.wei = kBf16;
    params.dtypes.dst = data_type_t::f32;
    params.dtypes.bias = data_type_t::f32;
    params.num_threads = threads;
    nat::native_weight_ref_t ref;
    ref.weight = wei;
    ref.pack_nr = 32;
    ref.weight_cache_mode = 0;
    return nat::native_brgemm_execute(d, ref, src, dst, bias, params);
}

// NaN and Inf in the source, the weight, the bias and the beta-read dst
// reach the output as the reference arithmetic gives them.
TEST_F(NativeBrgemmHardeningEpilogue, SpecialValuesPropagate) {
    const int M = 4, N = 32, K = 8;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    std::vector<uint16_t> src(static_cast<size_t>(M) * K);
    std::vector<uint16_t> wei(static_cast<size_t>(K) * N);
    for (size_t i = 0; i < src.size(); ++i)
        nt::store_f32(src.data(), kBf16, i, 0.5f + 0.25f * (i % 3));
    for (size_t i = 0; i < wei.size(); ++i)
        nt::store_f32(wei.data(), kBf16, i, 0.125f * (1 + i % 5));
    nt::store_f32(src.data(), kBf16, 1, nan); // row 0
    nt::store_f32(wei.data(), kBf16, 3, inf); // k 0, column 3
    nt::store_f32(wei.data(), kBf16, N + 4, -inf); // k 1, column 4
    nt::store_f32(wei.data(), kBf16, 2 * N + 4, inf); // k 2, column 4
    std::vector<float> bias(N, 0.5f);
    bias[5] = nan;
    bias[6] = -inf;
    std::vector<float> dst(static_cast<size_t>(M) * N, 1.0f);
    dst[1 * N + 7] = nan;
    ASSERT_EQ(run_bf16(M, N, K, src.data(), wei.data(), bias.data(), 1.0f,
                      dst.data(), 4),
            status_t::success);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            const float v = dst[static_cast<size_t>(m) * N + n];
            const bool want_nan
                    = m == 0 || n == 4 || n == 5 || (m == 1 && n == 7);
            if (want_nan) {
                EXPECT_TRUE(std::isnan(v)) << m << "," << n << ": " << v;
            } else if (n == 3) {
                EXPECT_EQ(v, inf) << m << "," << n;
            } else if (n == 6) {
                EXPECT_EQ(v, -inf) << m << "," << n;
            } else {
                EXPECT_TRUE(std::isfinite(v)) << m << "," << n << ": " << v;
            }
        }
}

// FTZ and DAZ in the MXCSR do not change the output of normal values (the
// products ignore the MXCSR; nothing else sees a subnormal).
TEST_F(NativeBrgemmHardeningEpilogue, FtzDazKeepsNormalOutputs) {
    const int M = 5, N = 64, K = 96;
    const nt::buffer_t src
            = nt::random_buffer(kBf16, static_cast<size_t>(M) * K, 101, -1, 1);
    const nt::buffer_t wei
            = nt::random_buffer(kBf16, static_cast<size_t>(K) * N, 102, -1, 1);
    std::vector<float> plain(static_cast<size_t>(M) * N, 0.25f);
    std::vector<float> ftz(plain);
    ASSERT_EQ(run_bf16(M, N, K, src.get(), wei.get(), nullptr, 0.5f,
                      plain.data(), 1),
            status_t::success);
    const unsigned int csr = _mm_getcsr();
    _mm_setcsr(csr | 0x8040);
    const status_t st = run_bf16(
            M, N, K, src.get(), wei.get(), nullptr, 0.5f, ftz.data(), 1);
    _mm_setcsr(csr);
    ASSERT_EQ(st, status_t::success);
    EXPECT_EQ(std::memcmp(plain.data(), ftz.data(), plain.size() * 4), 0);
}

// ── OpenMP team ──────────────────────────────────────────────────────────

// A plan with fewer workers than the caller's team runs in a region of the
// caller's team: the first workers run every unit once, the others none.  One
// worker runs inline.
TEST(NativeBrgemmHardeningTeam, FewerWorkersKeepTheCallersTeam) {
    nat::native_plan_t plan;
    plan.nr = 32;
    plan.mb = 12;
    const int M = 48;
    plan.m_blocks = 4;
    plan.n_tiles = 2;
    plan.threads = 4;
    plan.row_unit = 12;
    const nat::native_partition_t part(plan, M, 0, plan.n_tiles);
    for (int workers : {4, 1}) {
        SCOPED_TRACE("workers " + std::to_string(workers));
        std::mutex mu;
        std::set<int> tids;
        int region = 0;
        std::vector<int> cover(static_cast<size_t>(M) * plan.n_tiles, 0);
        int team_used = 0;
        const status_t st = nat::native_run_cpu_team(part, workers, 8, 64,
                [&](const nat::native_work_unit_t &u, float *) {
            std::lock_guard<std::mutex> lock(mu);
            tids.insert(omp_get_thread_num());
            region = omp_get_num_threads();
            for (int r = u.m0; r < u.m0 + u.rows; ++r)
                ++cover[static_cast<size_t>(r) * plan.n_tiles + u.n0 / plan.nr];
        }, &team_used);
        ASSERT_EQ(st, status_t::success);
        EXPECT_TRUE(std::all_of(
                cover.begin(), cover.end(), [](int c) { return c == 1; }));
        EXPECT_EQ(team_used, region);
        if (workers == 1) {
            EXPECT_EQ(team_used, 1);
        } else {
            EXPECT_EQ(team_used, 8) << "the caller's team";
            EXPECT_LE(*tids.rbegin(), workers - 1);
        }
    }
}

// An M = 1 call on a layout of at most a quarter of the L2 runs its units on
// the calling thread, whatever its plan's threads: its plan line says
// team=1.  The same call on a larger layout, and on the small one at M = 2,
// opens a region of the call's team.  bf16 x bf16 and u8 x s8 calls on 8
// threads, constant weights at weight cache mode 1.
TEST(NativeBrgemmHardeningTeam,
        OneRowCallsOfASmallLayoutRunOnTheCallingThread) {
    const nat::UarchParams &u = nat::detect_uarch();
    if (!u.avx512bf16 || !u.avx512vnni || !nat::native_base_isa())
        GTEST_SKIP() << "needs AVX512-BF16, AVX512-VNNI and AVX-512";
    nt::expect_in_child({{"ZENDNNL_API_LOG_LEVEL", "4"}}, [] {
        const size_t quarter
                = static_cast<size_t>(nat::detect_uarch().l2_bytes) / 4;
        const int N = 64;
        int bad = 0;
        uint32_t seed = 4700;
        for (const data_type_t wei : {data_type_t::bf16, data_type_t::s8}) {
            const bool int8 = wei == data_type_t::s8;
            const data_type_t src_dt = int8 ? data_type_t::u8 : wei;
            // K of a layout of a quarter of the L2, and of four quarters.
            const int small_k
                    = static_cast<int>(quarter
                              / (static_cast<size_t>(N) * nt::dt_bytes(wei)))
                    - 64;
            const int big_k = 4 * (small_k + 64);
            for (const auto &shape :
                    {std::make_pair(1, small_k), std::make_pair(2, small_k),
                            std::make_pair(1, big_k)}) {
                const int M = shape.first, K = shape.second;
                const bool alone = M == 1 && K == small_k;
                if ((K == small_k)
                        != (nat::native_packed_bytes(wei, K, N, 32)
                                <= quarter)) {
                    std::fprintf(stderr, "K %d: layout of %zu bytes\n", K,
                            nat::native_packed_bytes(wei, K, N, 32));
                    ++bad;
                    continue;
                }
                const nt::buffer_t src = nt::random_buffer(src_dt,
                        static_cast<size_t>(M) * K, seed++, int8 ? 0.0f : -1.0f,
                        int8 ? 15.0f : 1.0f);
                const nt::buffer_t w = nt::kept_until_exit(nt::random_buffer(
                        wei, static_cast<size_t>(K) * N, seed++,
                        int8 ? -8.0f : -1.0f, int8 ? 8.0f : 1.0f));
                const nt::buffer_t dst
                        = nt::make_buffer(static_cast<size_t>(M) * N * 4);
                float src_scale = 0.5f, wei_scale = 0.25f;
                nt::call_t c;
                c.M = M;
                c.N = N;
                c.K = K;
                c.src = src.get();
                c.wei = w.get();
                c.dst = dst.get();
                c.params.dtypes.src = src_dt;
                c.params.dtypes.wei = wei;
                c.params.dtypes.dst = data_type_t::f32;
                c.params.num_threads = 8;
                c.params.weight_cache_type = 1;
                c.params.lowoha_algo
                        = zendnnl::common::matmul_algo_t::native_brgemm;
                if (int8) {
                    auto &q = c.params.quant_params;
                    q.src_scale.buff = &src_scale;
                    q.src_scale.dt = data_type_t::f32;
                    q.src_scale.dims = {1, 1};
                    q.wei_scale.buff = &wei_scale;
                    q.wei_scale.dt = data_type_t::f32;
                    q.wei_scale.dims = {1, 1};
                }
                ::testing::internal::CaptureStdout();
                const status_t st = nt::run_matmul(c);
                const std::string log
                        = ::testing::internal::GetCapturedStdout();
                const size_t at = log.find("[MATMUL.NATIVE_BRGEMM] M=");
                const size_t team = log.find(" team=", at);
                const int region
                        = at == std::string::npos || team == std::string::npos
                        ? -1
                        : std::atoi(log.c_str() + team + 6);
                if (st != status_t::success || region != (alone ? 1 : 8)) {
                    std::fprintf(stderr,
                            "%s M %d K %d: status %d, team %d (%d expected)\n",
                            int8 ? "u8 x s8" : "bf16", M, K,
                            static_cast<int>(st), region, alone ? 1 : 8);
                    ++bad;
                }
            }
        }
        return bad;
    }, "ZENDNNL_API_LOG_LEVEL=4");
}

// ── Bias read at the tile ────────────────────────────────────────────────

// A bf16, f16, s8 or s32 bias read in its own type at the tile gives the bits
// of the same values given as f32, for a plain call and for a gated call
// over the same layout.
TEST_F(NativeBrgemmHardeningEpilogue, BiasInItsOwnTypeKeepsTheBits) {
    const int M = 5, N = 96, K = 64;
    const nt::buffer_t src
            = nt::random_buffer(kBf16, static_cast<size_t>(M) * K, 121, -1, 1);
    const nt::buffer_t wei
            = nt::random_buffer(kBf16, static_cast<size_t>(K) * N, 122, -1, 1);
    // Small integers: exact in every bias type.
    std::vector<float> bias_f32(N);
    for (int n = 0; n < N; ++n)
        bias_f32[n] = static_cast<float>((n * 7) % 23 - 11);
    using gated_t = zendnnl::lowoha::matmul::matmul_gated_act_t;
    for (gated_t gated : {gated_t::none, gated_t::silu_and_mul}) {
        const int n_out = gated == gated_t::none ? N : N / 2;
        auto run = [&](data_type_t bdt, const void *bias, float *dst) {
            nat::GemmDescriptor d;
            d.M = M;
            d.N = N;
            d.K = K;
            d.lda = K;
            d.ldb = N;
            d.ldc = n_out;
            d.src_dt = kBf16;
            d.wei_dt = kBf16;
            d.dst_dt = data_type_t::f32;
            d.num_threads = 4;
            d.gated_act = gated;
            zendnnl::lowoha::matmul::matmul_params params;
            params.dtypes.src = kBf16;
            params.dtypes.wei = kBf16;
            params.dtypes.dst = data_type_t::f32;
            params.dtypes.bias = bdt;
            params.num_threads = 4;
            nat::native_weight_ref_t ref;
            ref.weight = wei.get();
            ref.pack_nr = 32;
            ref.weight_cache_mode = 0;
            return nat::native_brgemm_execute(
                    d, ref, src.get(), dst, bias, params);
        };
        std::vector<float> want(static_cast<size_t>(M) * n_out, -1.0f);
        ASSERT_EQ(run(data_type_t::f32, bias_f32.data(), want.data()),
                status_t::success);
        for (data_type_t bdt :
                {kBf16, data_type_t::f16, kS8, data_type_t::s32}) {
            SCOPED_TRACE("gated " + std::to_string(static_cast<int>(gated))
                    + " bias " + std::to_string(static_cast<int>(bdt)));
            const nt::buffer_t bias = nt::make_buffer(N * 4);
            for (int n = 0; n < N; ++n)
                nt::store_f32(bias.get(), bdt, n, bias_f32[n]);
            std::vector<float> got(want.size(), -2.0f);
            ASSERT_EQ(run(bdt, bias.get(), got.data()), status_t::success);
            EXPECT_EQ(std::memcmp(got.data(), want.data(), want.size() * 4), 0);
        }
    }
}

// ── Staged gated rows ────────────────────────────────────────────────────

/// A bf16 call of M rows over a [N, K] weight (transB) into dst (bf16 for
/// silu_and_mul, else the f32 pre-activation), at weight cache mode 0.
nt::call_t staged_call(int M, int N, int K, const nt::buffer_t &src,
        const nt::buffer_t &wei, void *dst, bool gated) {
    nt::call_t c;
    c.M = M;
    c.N = N;
    c.K = K;
    c.transB = true;
    c.src = src.get();
    c.wei = wei.get();
    c.dst = dst;
    c.gated = gated ? zendnnl::lowoha::matmul::matmul_gated_act_t::silu_and_mul
                    : zendnnl::lowoha::matmul::matmul_gated_act_t::none;
    c.params.dtypes.src = kBf16;
    c.params.dtypes.wei = kBf16;
    c.params.dtypes.dst = gated ? kBf16 : data_type_t::f32;
    c.params.weight_cache_type = 0;
    return c;
}

// The staged rows of a gated call over halves that do not pair (N 4100):
// a check of the call allocates none; the calling thread keeps rows of up
// to kNativeThreadStagedFloats floats across its calls, grown to at least
// twice their size; a call that stages more grows them to its rows, and the
// thread's next call that fits the bound frees those.  Each call gives the
// activation of the same call's f32 pre-activation.
TEST_F(NativeBrgemmHardeningEpilogue,
        StagedRowsAboveTheBoundLastUntilACallFits) {
    const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
    const int N = 4100, K = 64;
    ASSERT_EQ(nat::native_paired_width(N, 32, 1, 32), 0);
    const nt::buffer_t wei = nt::random_buffer(
            kBf16, static_cast<size_t>(N) * K, 9100, -0.5f, 0.5f);
    const auto matches = [&](int M) {
        const nt::buffer_t src = nt::random_buffer(
                kBf16, static_cast<size_t>(M) * K, 9101 + M, -1.0f, 1.0f);
        std::vector<float> pre(static_cast<size_t>(M) * N);
        EXPECT_EQ(nt::run_matmul(
                          staged_call(M, N, K, src, wei, pre.data(), false)),
                status_t::success);
        std::vector<float> want(static_cast<size_t>(M) * N / 2);
        nt::gated_act_ref(pre.data(), M, N,
                zendnnl::lowoha::matmul::matmul_gated_act_t::silu_and_mul,
                want.data());
        nt::buffer_t dst = nt::make_buffer(want.size() * 2);
        const nt::call_t c = staged_call(M, N, K, src, wei, dst.get(), true);
        if (nt::refusal_of(c) != status_t::success
                || nt::run_matmul(c) != status_t::success)
            return false;
        for (size_t i = 0; i < want.size(); ++i)
            if (std::fabs(nt::load_f32(dst.get(), kBf16, i) - want[i])
                    > 1e-3f + std::fabs(want[i]) * (1.0f / 128.0f))
                return false;
        return true;
    };
    const size_t bound = nat::kNativeThreadStagedFloats;
    std::thread([&] {
        const nt::buffer_t src = nt::random_buffer(
                kBf16, static_cast<size_t>(4) * K, 9100, -1.0f, 1.0f);
        nt::buffer_t dst = nt::make_buffer(static_cast<size_t>(4) * N);
        EXPECT_EQ(
                nt::refusal_of(staged_call(4, N, K, src, wei, dst.get(), true)),
                status_t::success);
        EXPECT_EQ(nat::native_thread_staged_floats(), 0u)
                << "a check allocated staged rows";
        EXPECT_TRUE(matches(4));
        EXPECT_EQ(nat::native_thread_staged_floats(), size_t {4} * N);
        EXPECT_TRUE(matches(5));
        EXPECT_EQ(nat::native_thread_staged_floats(), size_t {8} * N);
        EXPECT_TRUE(matches(128));
        EXPECT_EQ(nat::native_thread_staged_floats(), size_t {128} * N);
        EXPECT_TRUE(matches(129));
        EXPECT_EQ(nat::native_thread_staged_floats(), bound);
        ASSERT_GT(size_t {256} * N, bound);
        EXPECT_TRUE(matches(256));
        EXPECT_EQ(nat::native_thread_staged_floats(), size_t {256} * N);
        EXPECT_TRUE(matches(4));
        EXPECT_EQ(nat::native_thread_staged_floats(), size_t {4} * N)
                << "a call that fits the bound kept the rows above it";
    }).join();
}

// The weight cache clears release the staged rows: the clearing thread's at
// once, another thread's at its next staged call, which takes rows of its
// own size where it would have kept larger ones.
TEST_F(NativeBrgemmHardeningEpilogue, TheClearsReleaseStagedRows) {
    const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
    const int N = 4100, K = 64;
    const nt::buffer_t wei = nt::random_buffer(
            kBf16, static_cast<size_t>(N) * K, 9200, -0.5f, 0.5f);
    const auto staged = [&](int M) {
        const nt::buffer_t src = nt::random_buffer(
                kBf16, static_cast<size_t>(M) * K, 9201 + M, -1.0f, 1.0f);
        nt::buffer_t dst = nt::make_buffer(static_cast<size_t>(M) * N);
        return nt::run_matmul(staged_call(M, N, K, src, wei, dst.get(), true));
    };
    for (bool all : {false, true}) {
        SCOPED_TRACE(all ? "clear_matmul_weight_caches"
                         : "clear_matmul_native_weight_caches");
        const auto clear = [&] {
            if (all)
                zendnnl::lowoha::matmul::clear_matmul_weight_caches();
            else
                zendnnl::lowoha::matmul::clear_matmul_native_weight_caches();
        };
        std::thread([&] {
            EXPECT_EQ(staged(1024), status_t::success);
            EXPECT_EQ(nat::native_thread_staged_floats(), size_t {1024} * N);
            clear();
            EXPECT_EQ(nat::native_thread_staged_floats(), 0u)
                    << "the clearing thread's rows";
        }).join();
        std::atomic<int> step {0};
        std::thread other([&] {
            EXPECT_EQ(staged(1024), status_t::success);
            step.store(1);
            while (step.load() != 2)
                std::this_thread::yield();
            EXPECT_EQ(staged(512), status_t::success);
            EXPECT_EQ(nat::native_thread_staged_floats(), size_t {512} * N)
                    << "rows taken before the clear were kept";
        });
        while (step.load() != 1)
            std::this_thread::yield();
        clear();
        step.store(2);
        other.join();
    }
}

// ── BF16 kernel addresses ────────────────────────────────────────────────

#if defined(__linux__)
/// A read-only mapping of bytes bytes that reads zero (MAP_NORESERVE) until
/// a page of it is made writable.
class sparse_map_t {
public:
    explicit sparse_map_t(size_t bytes) : bytes_(bytes) {
        void *p = mmap(nullptr, bytes, PROT_READ,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (p != MAP_FAILED) map_ = static_cast<uint8_t *>(p);
    }
    ~sparse_map_t() {
        if (map_ != nullptr) munmap(map_, bytes_);
    }
    sparse_map_t(const sparse_map_t &) = delete;
    sparse_map_t &operator=(const sparse_map_t &) = delete;

    uint8_t *get() const { return map_; }
    /// Makes the pages that hold [at, at + n) writable.
    bool writable(const void *at, size_t n) const {
        const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
        const auto lo = reinterpret_cast<std::uintptr_t>(at) / page * page;
        const auto hi = reinterpret_cast<std::uintptr_t>(at) + n;
        return mprotect(reinterpret_cast<void *>(lo), hi - lo,
                       PROT_READ | PROT_WRITE)
                == 0;
    }

private:
    uint8_t *map_ = nullptr;
    size_t bytes_ = 0;
};

uint16_t bf16_of(float f) {
    return static_cast<uint16_t>(bits_of(f) >> 16);
}

// The bf16 kernels offset the K pairs of a weight and the rows of a source
// in an int.  The layout rule takes bf16 weights of K up to kMaxNativeBf16K
// (2^26), whose last K pair lies 2^31 - 64 elements after the first, and
// declines K 67108866, whose pair 33554432 would lie 2^31 after it.  A call
// of more rows whose source rows (12 a kernel call) reach past the int range
// is turned away before any work, a call of one row is not.  Over a sparse
// mapping that reads zero elsewhere, a kernel call of K kMaxNativeBf16K sums
// its last K pair where it lies.
TEST(NativeBrgemmHardeningBf16Kernel, KAndSourceRowsWithinTheKernelsInt) {
    if (!nat::detect_uarch().avx512bf16 || !nat::native_base_isa())
        GTEST_SKIP() << "needs AVX512-BF16 and AVX-512 F / BW / VL / DQ";
    const auto layout = [](int K) {
        nat::native_layout_query_t q;
        q.wei_dt = kBf16;
        q.src_dt = kBf16;
        q.K = K;
        q.N = 32;
        q.ldb = 32;
        return nat::decide_native_layout(q);
    };
    EXPECT_EQ(nat::kMaxNativeBf16K, 1 << 26);
    EXPECT_TRUE(layout(nat::kMaxNativeBf16K).native);
    const nat::native_layout_decision_t past = layout(67108866);
    EXPECT_FALSE(past.native);
    EXPECT_STREQ(past.reason,
            "a bf16 weight of K past the bf16 kernels' int offsets");

    const int lda = 200000000;
    alignas(64) uint16_t buf[64] = {};
    for (const int M : {12, 1}) {
        SCOPED_TRACE(::testing::Message() << "M " << M);
        nt::call_t c;
        c.M = M;
        c.N = 32;
        c.K = 64;
        c.lda = lda;
        c.src = buf;
        c.wei = buf;
        c.dst = buf;
        c.params.dtypes.src = kBf16;
        c.params.dtypes.wei = kBf16;
        c.params.dtypes.dst = kBf16;
        c.params.lowoha_algo = matmul_algo_t::native_brgemm;
        std::string reason;
        const status_t st = nt::refusal_of(c, &reason);
        if (M == 1) {
            EXPECT_EQ(st, status_t::success) << reason;
        } else {
            EXPECT_EQ(st, status_t::unimplemented);
            EXPECT_EQ(reason,
                    "a bf16 source of rows past the bf16 kernels' int "
                    "offsets");
        }
    }

    const int K = nat::kMaxNativeBf16K, b_stride = 64;
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const size_t last = static_cast<size_t>(K / 2 - 1) * b_stride;
    const sparse_map_t wmap((last + b_stride) * 2 + page);
    const sparse_map_t amap(static_cast<size_t>(K) * 2 + page);
    if (wmap.get() == nullptr || amap.get() == nullptr)
        GTEST_SKIP() << "no address space for the weight";
    uint16_t *const b = reinterpret_cast<uint16_t *>(wmap.get());
    uint16_t *const a = reinterpret_cast<uint16_t *>(amap.get());
    ASSERT_TRUE(wmap.writable(b + last, b_stride * 2));
    ASSERT_TRUE(amap.writable(a + K - 2, 4));
    a[K - 2] = bf16_of(2.0f);
    for (int j = 0; j < 16; ++j) {
        b[last + 2 * j] = bf16_of(static_cast<float>(j + 1));
        b[last + 32 + 2 * j] = bf16_of(-static_cast<float>(j + 1));
    }
    std::vector<float> c(32, -1.0f);
    nat::select_bf16_brgemm_kernel(1, 32)(a, K, b, b_stride, c.data(), 32, K, K,
            0.0f, nullptr, nat::fused_postop_t::none, nullptr, 0);
    int bad = 0;
    for (int j = 0; j < 16; ++j) {
        bad += c[j] != 2.0f * (j + 1);
        bad += c[16 + j] != -2.0f * (j + 1);
    }
    EXPECT_EQ(bad, 0);
}
#endif

// ── Instruction sets ─────────────────────────────────────────────────────

// The native route needs AVX-512 F, BW, VL and DQ and FMA besides its dot
// product instructions; every CPU with AVX512-BF16 or AVX512-VNNI has them.
TEST(NativeBrgemmHardeningIsa, BaseIsaComesWithTheDotProducts) {
    const nat::UarchParams &u = nat::detect_uarch();
    if (!u.avx512bf16 && !u.avx512vnni)
        GTEST_SKIP() << "needs AVX512-BF16 or AVX512-VNNI";
    EXPECT_TRUE(nat::native_base_isa());
}

} // namespace
