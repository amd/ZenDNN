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

/// Native BRGEMM over native-layout weights: the weight view and the pack
/// against the custom-kernel pack bytes; BF16 native_brgemm_execute against a
/// double-precision reference for raw weights at weight cache 0 / 1 / 2; the
/// layout cache and in-place images; calls made inside a parallel region;
/// INT8 calls through the looper; and the row-group schedule against a
/// single-thread call.

#include <gtest/gtest.h>

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <initializer_list>

#include "common/bfloat16.hpp"
#include "common/float16.hpp"
#include "common/op_config.hpp"
#include "common/zendnnl_compat.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/pack.hpp"
#include "lowoha_operators/matmul/lowoha_matmul.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_brgemm_looper.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_int8_call.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_partition.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_planner.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_pack.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_weight_cache.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_weight_view.hpp"
#include "matmul_native/native_brgemm_test_utils.hpp"

namespace {

namespace ck = zendnnl::lowoha::matmul::custom_kernel;
namespace nv = zendnnl::lowoha::matmul::native;
using zendnnl::common::bfloat16_t;
using zendnnl::common::data_type_t;
using zendnnl::common::float16_t;
using zendnnl::error_handling::status_t;
using zendnnl::lowoha::matmul::matmul_params;

/// Whether the buffer of v holds an in-place image, checked within it.
template <typename V>
bool packed_in_place(const V &v) {
    return nv::native_weight_packed_in_place(
            v.data(), v.size() * sizeof(*v.data()));
}

constexpr double kF32Unit = 1.0 / (1 << 24);
constexpr double kBf16Unit = 1.0 / (1 << 8);
constexpr float kSentinel = 4096.0f;

struct aligned_free_t {
    void operator()(void *p) const { zendnnl_aligned_free(p); }
};
using buffer_t = std::unique_ptr<uint8_t, aligned_free_t>;

buffer_t alloc_buffer(size_t bytes) {
    const size_t rounded = (bytes + 63) & ~size_t {63};
    buffer_t b(static_cast<uint8_t *>(zendnnl_aligned_alloc(64, rounded)));
    if (b) std::memset(b.get(), 0, rounded);
    return b;
}

int all_threads() {
    return omp_get_max_threads();
}

/// Values k / 64 for k in [-64, 64]: exact in bf16, f16 and f32.
std::vector<float> grid_values(size_t n, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(-64, 64);
    std::vector<float> v(n);
    for (auto &x : v)
        x = static_cast<float>(dist(gen)) / 64.0f;
    return v;
}

std::vector<bfloat16_t> random_bf16(size_t n, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<bfloat16_t> v(n);
    for (auto &x : v)
        x = bfloat16_t(dist(gen));
    return v;
}

/// One BF16 problem: source, raw weight, bias / old-dst values, and the
/// double-precision products with their magnitudes.
struct problem_t {
    int M = 0, N = 0, K = 0;
    int lda = 0, ldb = 0, ldc = 0;
    bool transA = false, transB = true;
    std::vector<bfloat16_t> a;
    std::vector<bfloat16_t> w;
    std::vector<float> bias;
    std::vector<float> old;
    std::vector<double> dot;
    std::vector<double> mag;

    float a_at(int m, int k) const {
        return static_cast<float>(transA ? a[static_cast<size_t>(k) * lda + m]
                                         : a[static_cast<size_t>(m) * lda + k]);
    }
    float w_at(int k, int n) const {
        return static_cast<float>(transB ? w[static_cast<size_t>(n) * ldb + k]
                                         : w[static_cast<size_t>(k) * ldb + n]);
    }
};

problem_t make_problem(int M, int N, int K, bool transA, bool transB,
        int lda_pad, int ldb_pad, int ldc_pad, uint32_t seed) {
    problem_t pb;
    pb.M = M;
    pb.N = N;
    pb.K = K;
    pb.transA = transA;
    pb.transB = transB;
    pb.lda = (transA ? M : K) + lda_pad;
    pb.ldb = (transB ? K : N) + ldb_pad;
    pb.ldc = N + ldc_pad;
    // Padding holds large values so a read of it shows in the result.
    pb.a = random_bf16(static_cast<size_t>(transA ? K : M) * pb.lda, seed);
    pb.w = random_bf16(static_cast<size_t>(transB ? N : K) * pb.ldb, seed + 1);
    if (lda_pad > 0)
        for (int r = 0; r < (transA ? K : M); ++r)
            for (int c = transA ? M : K; c < pb.lda; ++c)
                pb.a[static_cast<size_t>(r) * pb.lda + c]
                        = bfloat16_t(kSentinel);
    if (ldb_pad > 0)
        for (int r = 0; r < (transB ? N : K); ++r)
            for (int c = transB ? K : N; c < pb.ldb; ++c)
                pb.w[static_cast<size_t>(r) * pb.ldb + c]
                        = bfloat16_t(kSentinel);
    pb.bias = grid_values(N, seed + 2);
    pb.old = grid_values(static_cast<size_t>(M) * N, seed + 3);

    std::vector<double> wk(static_cast<size_t>(K) * N);
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n)
            wk[static_cast<size_t>(k) * N + n] = pb.w_at(k, n);
    pb.dot.assign(static_cast<size_t>(M) * N, 0.0);
    pb.mag.assign(static_cast<size_t>(M) * N, 0.0);
    for (int m = 0; m < M; ++m) {
        double *dot = pb.dot.data() + static_cast<size_t>(m) * N;
        double *mag = pb.mag.data() + static_cast<size_t>(m) * N;
        for (int k = 0; k < K; ++k) {
            const double av = pb.a_at(m, k);
            const double aa = std::fabs(av);
            const double *row = wk.data() + static_cast<size_t>(k) * N;
            for (int n = 0; n < N; ++n) {
                dot[n] += av * row[n];
                mag[n] += aa * std::fabs(row[n]);
            }
        }
    }
    return pb;
}

/// One execution of a problem: the epilogue knobs and the output buffer.
struct run_t {
    data_type_t dst_dt = data_type_t::bf16;
    data_type_t bias_dt = data_type_t::none;
    float alpha = 1.0f;
    float beta = 0.0f;
    int threads = 1;
    std::string tag;
};

const char *dt_name(data_type_t dt) {
    switch (dt) {
        case data_type_t::none: return "none";
        case data_type_t::f32: return "f32";
        case data_type_t::bf16: return "bf16";
        case data_type_t::f16: return "f16";
        case data_type_t::s8: return "s8";
        case data_type_t::u8: return "u8";
        default: return "other";
    }
}

std::string describe(const problem_t &pb, const run_t &r) {
    std::ostringstream os;
    os << r.tag << " M=" << pb.M << " N=" << pb.N << " K=" << pb.K
       << " transA=" << pb.transA << " transB=" << pb.transB
       << " lda=" << pb.lda << " ldb=" << pb.ldb << " ldc=" << pb.ldc
       << " dst=" << dt_name(r.dst_dt) << " bias=" << dt_name(r.bias_dt)
       << " alpha=" << r.alpha << " beta=" << r.beta
       << " threads=" << r.threads;
    return os.str();
}

/// Bias and destination storage for one run, the destination padded with
/// sentinels and pre-filled with the old values the beta term reads.
struct io_t {
    buffer_t bias;
    buffer_t dst;
};

io_t make_io(const problem_t &pb, const run_t &r) {
    io_t io;
    if (r.bias_dt != data_type_t::none) {
        io.bias = alloc_buffer(static_cast<size_t>(pb.N) * 4);
        for (int n = 0; n < pb.N; ++n) {
            if (r.bias_dt == data_type_t::f32)
                reinterpret_cast<float *>(io.bias.get())[n] = pb.bias[n];
            else if (r.bias_dt == data_type_t::bf16)
                reinterpret_cast<bfloat16_t *>(io.bias.get())[n]
                        = bfloat16_t(pb.bias[n]);
            else
                reinterpret_cast<float16_t *>(io.bias.get())[n]
                        = float16_t(pb.bias[n]);
        }
    }
    const size_t elems = static_cast<size_t>(pb.M) * pb.ldc;
    io.dst = alloc_buffer(elems * 4);
    for (int m = 0; m < pb.M; ++m)
        for (int n = 0; n < pb.ldc; ++n) {
            const size_t at = static_cast<size_t>(m) * pb.ldc + n;
            const float v = n < pb.N ? pb.old[static_cast<size_t>(m) * pb.N + n]
                                     : kSentinel;
            if (r.dst_dt == data_type_t::f32)
                reinterpret_cast<float *>(io.dst.get())[at] = v;
            else
                reinterpret_cast<bfloat16_t *>(io.dst.get())[at]
                        = bfloat16_t(v);
        }
    return io;
}

status_t execute(const problem_t &pb, const run_t &r,
        const nv::native_weight_ref_t &ref, const io_t &io,
        const void *src = nullptr) {
    nv::GemmDescriptor desc;
    desc.M = pb.M;
    desc.N = pb.N;
    desc.K = pb.K;
    desc.lda = pb.lda;
    desc.ldb = pb.ldb;
    desc.ldc = pb.ldc;
    desc.transA = pb.transA;
    desc.transB = pb.transB;
    desc.alpha = r.alpha;
    desc.beta = r.beta;
    desc.src_dt = data_type_t::bf16;
    desc.wei_dt = data_type_t::bf16;
    desc.dst_dt = r.dst_dt;
    desc.bias_dt = r.bias_dt;
    desc.bias = io.bias.get();
    desc.is_weights_const = true;
    desc.num_threads = r.threads;
    matmul_params params;
    params.dtypes.src = data_type_t::bf16;
    params.dtypes.wei = data_type_t::bf16;
    params.dtypes.dst = r.dst_dt;
    params.dtypes.bias = r.bias_dt;
    return nv::native_brgemm_execute(desc, ref, src ? src : pb.a.data(),
            io.dst.get(), io.bias.get(), params);
}

/// Every output within the f32 accumulation bound of the reference (plus one
/// bf16 rounding for a bf16 destination), and the ldc padding untouched.
void expect_matches(const problem_t &pb, const run_t &r, const io_t &io) {
    const bool f32 = r.dst_dt == data_type_t::f32;
    const bool has_bias = r.bias_dt != data_type_t::none;
    const double acc_bound = 2.0 * (pb.K + 1) * kF32Unit;
    int bad = 0;
    std::string first;
    for (int m = 0; m < pb.M; ++m)
        for (int n = 0; n < pb.ldc; ++n) {
            const size_t at = static_cast<size_t>(m) * pb.ldc + n;
            const double got = f32
                    ? reinterpret_cast<const float *>(io.dst.get())[at]
                    : static_cast<float>(reinterpret_cast<const bfloat16_t *>(
                              io.dst.get())[at]);
            double want = kSentinel, tol = 0.0;
            if (n < pb.N) {
                const size_t mn = static_cast<size_t>(m) * pb.N + n;
                const double prod = r.alpha * pb.dot[mn];
                const double old = static_cast<double>(r.beta) * pb.old[mn];
                const double bias = has_bias ? pb.bias[n] : 0.0;
                want = prod + old + bias;
                tol = std::fabs(r.alpha) * acc_bound * pb.mag[mn]
                        + 4.0 * kF32Unit
                                * (std::fabs(prod) + std::fabs(old)
                                        + std::fabs(bias));
                if (!f32) tol += kBf16Unit * (std::fabs(want) + tol);
            }
            if (!(std::fabs(got - want) <= tol)) {
                if (bad++ == 0) {
                    std::ostringstream os;
                    os << "first mismatch at (" << m << ", " << n << "): got "
                       << got << " want " << want << " tol " << tol;
                    first = os.str();
                }
            }
        }
    EXPECT_EQ(bad, 0) << describe(pb, r) << ": " << first;
}

void run_and_expect(const problem_t &pb, const run_t &r,
        const nv::native_weight_ref_t &ref) {
    io_t io = make_io(pb, r);
    ASSERT_TRUE(io.dst);
    const status_t st = execute(pb, r, ref, io);
    ASSERT_EQ(st, status_t::success) << describe(pb, r);
    expect_matches(pb, r, io);
}

buffer_t prepack_bf16(const problem_t &pb, int pack_nr, bool interleave) {
    const size_t bytes = ck::packed_weight_size_bf16(pb.K, pb.N, pack_nr);
    buffer_t b = alloc_buffer(bytes);
    if (!b
            || ck::prepack_weight_into_bf16(pb.w.data(), pb.K, pb.N, pb.ldb,
                       pack_nr, pb.transB, interleave, b.get())
                    != status_t::success)
        return nullptr;
    return b;
}

nv::native_weight_ref_t raw_ref(const void *w, int pack_nr, int32_t wc) {
    nv::native_weight_ref_t ref;
    ref.weight = w;
    ref.pack_nr = pack_nr;
    ref.weight_cache_mode = wc;
    return ref;
}

class NativeBrgemmLayoutTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!nv::detect_uarch().avx512bf16) GTEST_SKIP() << "no AVX512-BF16";
        zendnnl::lowoha::matmul::clear_matmul_weight_caches();
    }
    void TearDown() override {
        zendnnl::lowoha::matmul::clear_matmul_weight_caches();
    }

    /// Runs a raw weight at weight cache wc on its own copy, then drops the
    /// pack caches before the copy is freed so no later allocation at the
    /// same address meets a stale entry.
    void run_raw(const problem_t &pb, const run_t &r, int pack_nr, int32_t wc) {
        std::vector<bfloat16_t> copy = pb.w;
        run_and_expect(pb, r, raw_ref(copy.data(), pack_nr, wc));
        zendnnl::lowoha::matmul::clear_matmul_weight_caches();
    }

    /// Every acquisition path of one problem; threads, destination, bias and
    /// alpha / beta rotate with seed so a sweep covers their combinations.
    void sweep_sources(const problem_t &pb, uint32_t seed) {
        static const int kThreads[] = {1, 4, 0};
        static const data_type_t kBias[] = {data_type_t::none, data_type_t::f32,
                data_type_t::bf16, data_type_t::f16};
        static const float kAlphaBeta[][2] = {{1.0f, 0.0f}, {0.5f, 0.0f},
                {1.0f, 1.0f}, {-2.0f, 0.25f}, {0.0f, 0.5f}};
        for (int v = 0; v < 5; ++v) {
            const uint32_t s = seed + v;
            run_t r;
            r.threads = kThreads[s % 3] ? kThreads[s % 3] : all_threads();
            r.dst_dt = (s % 2) ? data_type_t::f32 : data_type_t::bf16;
            r.bias_dt = kBias[(seed + 2 * v) % 4];
            r.alpha = kAlphaBeta[s % 5][0];
            r.beta = kAlphaBeta[s % 5][1];
            const int32_t wc = v < 2 ? 0 : v - 2;
            r.tag = "raw_wc" + std::to_string(wc);
            run_raw(pb, r, 32, wc);
        }
    }
};

// ── Weight view against the custom-kernel pack bytes ────────────────────

struct view_case_t {
    int K, N, pack_nr;
    bool transB, interleave;

    std::string str() const {
        return "K=" + std::to_string(K) + " N=" + std::to_string(N)
                + " pack_nr=" + std::to_string(pack_nr)
                + " transB=" + std::to_string(transB)
                + " interleave=" + std::to_string(interleave);
    }
};

std::vector<view_case_t> view_cases(
        std::initializer_list<int> Ks, std::initializer_list<int> Ns) {
    std::vector<view_case_t> v;
    for (int K : Ks)
        for (int N : Ns)
            for (bool transB : {false, true})
                for (bool interleave : {false, true})
                    v.push_back({K, N, 32, transB, interleave});
    return v;
}

/// The view the native BRGEMM reads a buffer in the layout through.
nv::native_weight_view_t view_of(
        const void *buf, const view_case_t &c, data_type_t dt) {
    nv::native_weight_view_t v;
    v.base = static_cast<const uint8_t *>(buf);
    v.wei_dt = dt;
    v.K = c.K;
    v.N = c.N;
    v.pack_nr = c.pack_nr;
    v.interleaved = c.interleave;
    v.k_groups = nv::native_k_groups(dt, c.K);
    v.block_bytes = nv::native_block_bytes(dt, c.K, c.pack_nr);
    return v;
}

std::vector<int8_t> random_s8(size_t n, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(-128, 127);
    std::vector<int8_t> v(n);
    for (auto &x : v)
        x = static_cast<int8_t>(dist(gen));
    return v;
}

TEST_F(NativeBrgemmLayoutTest, Bf16ViewMatchesPack) {
    for (const view_case_t &c :
            view_cases({1, 2, 3, 31, 64, 100}, {32, 64, 96, 128})) {
        problem_t pb = make_problem(
                1, c.N, c.K, false, c.transB, 0, 0, 0, c.K * 7 + c.N);
        buffer_t buf = prepack_bf16(pb, c.pack_nr, c.interleave);
        ASSERT_TRUE(buf) << c.str();
        const nv::native_weight_view_t v
                = view_of(buf.get(), c, data_type_t::bf16);
        EXPECT_EQ(v.base, buf.get()) << c.str();
        EXPECT_EQ(v.k_groups, (c.K + 1) / 2) << c.str();
        EXPECT_EQ(v.group_stride(), c.pack_nr * 2) << c.str();
        EXPECT_EQ((v.block_bytes * (c.N / c.pack_nr) + 63) & ~size_t {63},
                ck::packed_weight_size_bf16(c.K, c.N, c.pack_nr))
                << c.str();
        int bad = 0;
        for (int p = 0; p < c.N; ++p) {
            const auto *blk = reinterpret_cast<const uint16_t *>(
                    v.block(p / c.pack_nr));
            for (int k = 0; k < 2 * v.k_groups; ++k) {
                const uint16_t got = blk[(k / 2) * v.group_stride()
                        + (p % c.pack_nr) * 2 + k % 2];
                const float want
                        = k < c.K ? pb.w_at(k, v.logical_col(p)) : 0.0f;
                bad += static_cast<float>(bfloat16_t::from_bits(got)) != want;
            }
        }
        EXPECT_EQ(bad, 0) << c.str();
    }
}

TEST_F(NativeBrgemmLayoutTest, Bf16RawAcquisitionMatchesPrepack) {
    for (int K : {3, 64})
        for (bool interleave : {false, true})
            for (int32_t wc : {0, 1, 2}) {
                problem_t pb = make_problem(1, 64, K, false, true, 0, 0, 0, 11);
                buffer_t want = prepack_bf16(pb, 32, interleave);
                ASSERT_TRUE(want);
                std::vector<bfloat16_t> copy = pb.w;
                nv::native_weight_ref_t ref = raw_ref(copy.data(), 32, wc);
                ref.interleave_split_halves = interleave;
                nv::native_weight_view_t v;
                {
                    nv::native_weight_lease_t lease;
                    ASSERT_EQ(nv::acquire_native_weight(ref, data_type_t::bf16,
                                      K, pb.N, pb.ldb, true, v, lease),
                            status_t::success);
                    const size_t bytes = v.block_bytes * (pb.N / 32);
                    EXPECT_EQ(std::memcmp(v.base, want.get(), bytes), 0)
                            << "K=" << K << " wc=" << wc;
                    const bool in_place = wc == 2 && K % 2 == 0;
                    EXPECT_EQ(v.base
                                    == reinterpret_cast<const uint8_t *>(
                                            copy.data()),
                            in_place)
                            << "K=" << K << " wc=" << wc;
                    EXPECT_EQ(packed_in_place(copy), in_place)
                            << "K=" << K << " wc=" << wc;
                }
                zendnnl::lowoha::matmul::clear_matmul_weight_caches();
            }
}

TEST_F(NativeBrgemmLayoutTest, Int8ViewMatchesPack) {
    for (const view_case_t &c :
            view_cases({1, 3, 4, 5, 64, 100}, {32, 64, 96})) {
        const int ldb = c.transB ? c.K : c.N;
        const std::vector<int8_t> w
                = random_s8(static_cast<size_t>(c.K) * c.N, c.K * 13 + c.N);
        auto w_at = [&](int k, int n) {
            return static_cast<int>(
                    w[c.transB ? static_cast<size_t>(n) * ldb + k
                               : static_cast<size_t>(k) * ldb + n]);
        };
        const size_t size = ck::packed_weight_size_int8(c.K, c.N, c.pack_nr);
        buffer_t buf = alloc_buffer(size);
        ASSERT_EQ(ck::prepack_weight_into_int8(w.data(), c.K, c.N, ldb,
                          c.pack_nr, c.transB, c.interleave, buf.get()),
                status_t::success)
                << c.str();
        const nv::native_weight_view_t v
                = view_of(buf.get(), c, data_type_t::s8);
        EXPECT_EQ(v.k_groups, (c.K + 3) / 4) << c.str();
        EXPECT_EQ(v.group_stride(), c.pack_nr * 4) << c.str();
        EXPECT_EQ((v.block_bytes * (c.N / c.pack_nr) + 63) & ~size_t {63}, size)
                << c.str();
        int bad = 0;
        for (int p = 0; p < c.N; ++p) {
            const int nb = p / c.pack_nr;
            const auto *blk = reinterpret_cast<const int8_t *>(v.block(nb));
            bad += reinterpret_cast<const uint8_t *>(v.col_sums(nb))
                    != v.block(nb)
                            + static_cast<size_t>(v.k_groups) * c.pack_nr * 4;
            int sum = 0;
            for (int k = 0; k < 4 * v.k_groups; ++k) {
                const int want = k < c.K ? w_at(k, v.logical_col(p)) : 0;
                sum += want;
                bad += blk[(k / 4) * v.group_stride() + (p % c.pack_nr) * 4
                               + k % 4]
                        != want;
            }
            bad += v.col_sums(nb)[p % c.pack_nr] != sum;
        }
        EXPECT_EQ(bad, 0) << c.str();

        for (int32_t wc : {0, 1}) {
            std::vector<int8_t> copy = w;
            nv::native_weight_ref_t raw = raw_ref(copy.data(), c.pack_nr, wc);
            raw.interleave_split_halves = c.interleave;
            nv::native_weight_view_t rv;
            nv::native_weight_lease_t rl;
            ASSERT_EQ(nv::acquire_native_weight(raw, data_type_t::s8, c.K, c.N,
                              ldb, c.transB, rv, rl),
                    status_t::success)
                    << c.str();
            EXPECT_EQ(std::memcmp(rv.base, buf.get(),
                              v.block_bytes * (c.N / c.pack_nr)),
                    0)
                    << c.str() << " wc=" << wc;
            zendnnl::lowoha::matmul::clear_matmul_weight_caches();
        }
    }
}

TEST_F(NativeBrgemmLayoutTest, Int8RawWeightPacksOutOfPlace) {
    const int K = 64, N = 64;
    const std::vector<int8_t> w = random_s8(static_cast<size_t>(K) * N, 29);
    const size_t size = ck::packed_weight_size_int8(K, N, 32);
    buffer_t want = alloc_buffer(size);
    ASSERT_EQ(ck::prepack_weight_into_int8(
                      w.data(), K, N, K, 32, true, false, want.get()),
            status_t::success);
    for (size_t capacity : {size_t {0}, size})
        for (int32_t wc : {1, 2}) {
            buffer_t raw = alloc_buffer(size);
            std::memcpy(raw.get(), w.data(), w.size());
            nv::native_weight_ref_t ref = raw_ref(raw.get(), 32, wc);
            ref.capacity_bytes = capacity;
            nv::native_weight_view_t v;
            nv::native_weight_lease_t lease;
            ASSERT_EQ(nv::acquire_native_weight(
                              ref, data_type_t::s8, K, N, K, true, v, lease),
                    status_t::success);
            EXPECT_NE(v.base, raw.get()) << capacity << " wc=" << wc;
            EXPECT_EQ(std::memcmp(v.base, want.get(), v.block_bytes * (N / 32)),
                    0)
                    << capacity << " wc=" << wc;
            EXPECT_EQ(std::memcmp(raw.get(), w.data(), w.size()), 0)
                    << capacity << " wc=" << wc;
            zendnnl::lowoha::matmul::clear_matmul_weight_caches();
        }
}

// ── Native pack against the custom-kernel pack bytes ────────────────────

struct pack_case_t {
    int K, N, pack_nr, pad;
    bool transB, interleave;

    std::string str() const {
        return "K=" + std::to_string(K) + " N=" + std::to_string(N)
                + " pack_nr=" + std::to_string(pack_nr) + " pad="
                + std::to_string(pad) + " transB=" + std::to_string(transB)
                + " interleave=" + std::to_string(interleave);
    }
};

std::vector<pack_case_t> pack_cases() {
    std::vector<pack_case_t> v;
    for (int K : {1, 2, 3, 4, 5, 31, 63, 64, 100, 1000})
        for (int N : {32, 64, 96, 128})
            for (int pad : {0, 3})
                for (bool transB : {false, true})
                    for (bool interleave : {false, true})
                        v.push_back({K, N, 32, pad, transB, interleave});
    for (bool transB : {false, true})
        for (bool interleave : {false, true}) {
            v.push_back({1000, 4096, 32, 0, transB, interleave});
            v.push_back({4097, 1024, 32, 0, transB, interleave});
        }
    return v;
}

TEST_F(NativeBrgemmLayoutTest, PackMatchesCustomKernelPrepack) {
    int i = 0;
    for (const pack_case_t &c : pack_cases()) {
        const int ldb = (c.transB ? c.K : c.N) + c.pad;
        const size_t raw = static_cast<size_t>(c.transB ? c.N : c.K) * ldb;
        const int threads = i % 3 == 0 ? 1 : (i % 3 == 1 ? 4 : all_threads());
        ++i;
        const std::vector<bfloat16_t> wb = random_bf16(raw, c.K * 3 + c.N);
        const std::vector<int8_t> ws = random_s8(raw, c.K * 5 + c.N);
        for (data_type_t dt : {data_type_t::bf16, data_type_t::s8}) {
            const bool s8 = dt == data_type_t::s8;
            const std::string what = c.str() + " wei=" + dt_name(dt)
                    + " threads=" + std::to_string(threads);
            const size_t bytes
                    = nv::native_packed_bytes(dt, c.K, c.N, c.pack_nr);
            ASSERT_EQ(bytes,
                    s8 ? ck::packed_weight_size_int8(c.K, c.N, c.pack_nr)
                       : ck::packed_weight_size_bf16(c.K, c.N, c.pack_nr))
                    << what;
            buffer_t want = alloc_buffer(bytes);
            buffer_t got = alloc_buffer(bytes);
            ASSERT_TRUE(want && got) << what;
            std::memset(got.get(), 0xA5, bytes);
            const status_t ck_st = s8
                    ? ck::prepack_weight_into_int8(ws.data(), c.K, c.N, ldb,
                              c.pack_nr, c.transB, c.interleave, want.get())
                    : ck::prepack_weight_into_bf16(wb.data(), c.K, c.N, ldb,
                              c.pack_nr, c.transB, c.interleave, want.get());
            ASSERT_EQ(ck_st, status_t::success) << what;
            const void *w = s8 ? static_cast<const void *>(ws.data())
                               : static_cast<const void *>(wb.data());
            ASSERT_EQ(nv::native_pack_weight(dt, w, c.K, c.N, ldb, c.pack_nr,
                              c.transB, c.interleave, got.get(), threads),
                    status_t::success)
                    << what;
            const size_t used = nv::native_block_bytes(dt, c.K, c.pack_nr)
                    * static_cast<size_t>(c.N / c.pack_nr);
            EXPECT_EQ(std::memcmp(got.get(), want.get(), used), 0) << what;
        }
    }
}

TEST_F(NativeBrgemmLayoutTest, PackRejects) {
    const int K = 64, N = 64;
    const std::vector<bfloat16_t> w
            = random_bf16(static_cast<size_t>(K) * N, 7);
    ASSERT_GT(nv::native_packed_bytes(data_type_t::bf16, K, N, 32), 0u);
    std::vector<uint8_t> dst(
            nv::native_packed_bytes(data_type_t::s8, K, 96, 32) + 4096, 0xCD);
    auto untouched = [&dst]() {
        return std::count(dst.begin(), dst.end(), uint8_t {0xCD})
                == static_cast<std::ptrdiff_t>(dst.size());
    };
    // Shapes off the layout never reach the pack: the layout rule declines
    // them (NativeLayoutRule).
    const struct {
        const char *what;
        data_type_t dt;
        int K, N, ldb;
        bool transB;
    } cases[] = {
            {"ldb below N", data_type_t::bf16, K, N, N - 1, false},
            {"ldb below K", data_type_t::s8, K, N, K - 1, true},
    };
    for (const auto &b : cases) {
        EXPECT_NE(nv::native_pack_weight(b.dt, w.data(), b.K, b.N, b.ldb, 32,
                          b.transB, false, dst.data(), 4),
                status_t::success)
                << b.what;
        EXPECT_TRUE(untouched()) << b.what;
    }
    EXPECT_NE(nv::native_pack_weight(data_type_t::bf16, nullptr, K, N, N, 32,
                      false, false, dst.data(), 1),
            status_t::success);
    EXPECT_NE(nv::native_pack_weight(data_type_t::bf16, w.data(), K, N, N, 32,
                      false, false, nullptr, 1),
            status_t::success);
    EXPECT_TRUE(untouched());
}

TEST_F(NativeBrgemmLayoutTest, AcquisitionRejects) {
    problem_t pb = make_problem(1, 64, 64, false, true, 0, 0, 0, 5);
    std::vector<bfloat16_t> copy = pb.w;
    auto acquire = [&](nv::native_weight_ref_t ref, data_type_t dt, int K,
                           int N, int ldb) {
        nv::native_weight_view_t v;
        nv::native_weight_lease_t lease;
        return nv::acquire_native_weight(ref, dt, K, N, ldb, true, v, lease);
    };
    // Shapes off the layout never reach the acquisition: the layout rule
    // declines them (NativeLayoutRule).
    const nv::native_weight_ref_t ok = raw_ref(copy.data(), 32, 0);
    EXPECT_EQ(acquire(ok, data_type_t::bf16, 64, 64, 64), status_t::success);
    {
        nv::native_weight_ref_t r = ok;
        r.weight = nullptr;
        EXPECT_NE(acquire(r, data_type_t::bf16, 64, 64, 64), status_t::success);
    }
    for (int32_t wc : {-1, 3}) {
        EXPECT_NE(acquire(raw_ref(copy.data(), 32, wc), data_type_t::bf16, 64,
                          64, 64),
                status_t::success)
                << wc;
    }
    EXPECT_NE(
            acquire(raw_ref(copy.data(), 32, 1), data_type_t::bf16, 64, 64, 63),
            status_t::success);
    zendnnl::lowoha::matmul::clear_matmul_weight_caches();
}

TEST_F(NativeBrgemmLayoutTest, ExecuteRejectsAndCounts) {
    problem_t pb = make_problem(4, 64, 64, false, true, 0, 0, 0, 9);
    std::vector<bfloat16_t> copy = pb.w;
    const nv::native_weight_ref_t ref = raw_ref(copy.data(), 32, 0);
    run_t r;
    r.dst_dt = data_type_t::f32;
    r.threads = 1;
    io_t io = make_io(pb, r);

    problem_t bad = pb;
    bad.lda = pb.K - 1;
    EXPECT_NE(execute(bad, r, ref, io), status_t::success);
    bad = pb;
    bad.ldc = pb.N - 1;
    EXPECT_NE(execute(bad, r, ref, io), status_t::success);
    {
        nv::GemmDescriptor desc;
        desc.M = 4;
        desc.N = 64;
        desc.K = 64;
        desc.lda = 64;
        desc.ldb = 64;
        desc.ldc = 64;
        desc.src_dt = data_type_t::f32;
        desc.wei_dt = data_type_t::bf16;
        desc.dst_dt = data_type_t::f32;
        matmul_params params;
        EXPECT_NE(nv::native_brgemm_execute(desc, ref, pb.a.data(),
                          io.dst.get(), nullptr, params),
                status_t::success);
    }

    // M == 0 succeeds without touching dst.
    problem_t empty = pb;
    empty.M = 0;
    std::vector<uint8_t> snapshot(io.dst.get(),
            io.dst.get() + static_cast<size_t>(pb.M) * pb.ldc * 4);
    EXPECT_EQ(execute(empty, r, ref, io), status_t::success);
    EXPECT_EQ(std::memcmp(snapshot.data(), io.dst.get(), snapshot.size()), 0);

    run_and_expect(pb, r, ref);
}

// ── BF16 shape sweep ─────────────────────────────────────────────────────

struct shape_t {
    int M, K;
};

class NativeBrgemmLayoutShapes : public NativeBrgemmLayoutTest,
                                 public ::testing::WithParamInterface<shape_t> {
};

TEST_P(NativeBrgemmLayoutShapes, AllSources) {
    const shape_t s = GetParam();
    for (int N : {32, 64, 96}) {
        const uint32_t seed = static_cast<uint32_t>(s.M * 31 + s.K * 7 + N);
        problem_t pb = make_problem(s.M, N, s.K, false, true, 0, 0, 0, seed);
        sweep_sources(pb, seed);
    }
}

std::vector<shape_t> sweep_shapes() {
    std::vector<shape_t> v;
    for (int M : {1, 2, 3, 4, 7, 8, 16, 33, 64, 65, 128, 512, 1500})
        for (int K : {1, 3, 31, 63, 64, 100, 1000, 4096})
            v.push_back({M, K});
    return v;
}

INSTANTIATE_TEST_SUITE_P(Bf16, NativeBrgemmLayoutShapes,
        ::testing::ValuesIn(sweep_shapes()),
        [](const ::testing::TestParamInfo<shape_t> &info) {
            return "M" + std::to_string(info.param.M) + "_K"
                    + std::to_string(info.param.K);
        });

// ── Wide N ───────────────────────────────────────────────────────────────

struct wide_t {
    int M, N, K;
};

class NativeBrgemmLayoutWide : public NativeBrgemmLayoutTest,
                               public ::testing::WithParamInterface<wide_t> {};

TEST_P(NativeBrgemmLayoutWide, AllSources) {
    const wide_t s = GetParam();
    const uint32_t seed = static_cast<uint32_t>(s.M * 3 + s.K + s.N);
    problem_t pb = make_problem(s.M, s.N, s.K, false, true, 0, 0, 0, seed);
    sweep_sources(pb, seed);
}

std::vector<wide_t> wide_shapes() {
    std::vector<wide_t> v;
    for (int M : {1, 2, 4, 8, 16, 33})
        for (int K : {64, 1000, 4096})
            v.push_back({M, 4096, K});
    for (int M : {64, 128, 512})
        v.push_back({M, 4096, 63});
    v.push_back({1500, 4096, 31});
    v.push_back({7, 4096, 4096});
    return v;
}

INSTANTIATE_TEST_SUITE_P(Bf16, NativeBrgemmLayoutWide,
        ::testing::ValuesIn(wide_shapes()),
        [](const ::testing::TestParamInfo<wide_t> &info) {
            return "M" + std::to_string(info.param.M) + "_N"
                    + std::to_string(info.param.N) + "_K"
                    + std::to_string(info.param.K);
        });

// ── K split into passes when row tiles share a large B slab ──────────────

TEST_F(NativeBrgemmLayoutTest, LongK) {
    for (int M : {7, 64})
        for (int N : {32, 64})
            for (int K : {9001, 20001}) {
                problem_t pb = make_problem(
                        M, N, K, false, true, 0, 0, 0, M + N + K);
                for (int threads : {1, all_threads()}) {
                    run_t r;
                    r.dst_dt = data_type_t::f32;
                    r.bias_dt = data_type_t::f32;
                    r.alpha = 0.5f;
                    r.beta = 1.0f;
                    r.threads = threads;
                    r.tag = "raw_wc0";
                    run_raw(pb, r, 32, 0);
                }
            }
}

// ── Every row count at every tile width ─────────────────────────────────

TEST_F(NativeBrgemmLayoutTest, EveryRowTail) {
    // N 96 keeps 32-column tiles, and N 32 on two threads narrows to
    // 16-column tiles.
    const struct {
        int N, threads;
    } tilings[] = {{96, 1}, {32, 2}};
    for (const auto &t : tilings)
        for (int M = 1; M <= 25; ++M) {
            problem_t pb = make_problem(
                    M, t.N, 63, false, true, 0, 0, 0, M * 11 + t.N);
            run_t r;
            r.dst_dt = data_type_t::f32;
            r.bias_dt = data_type_t::f32;
            r.threads = t.threads;
            r.tag = "raw_wc0";
            run_raw(pb, r, 32, 0);
        }
}

// ── transA, leading dimensions, bias, destination, alpha / beta ─────────

struct feature_t {
    int M, N, K;
    bool transA, transB, padded;
};

class NativeBrgemmLayoutFeatures
    : public NativeBrgemmLayoutTest,
      public ::testing::WithParamInterface<feature_t> {};

TEST_P(NativeBrgemmLayoutFeatures, Combinations) {
    const feature_t f = GetParam();
    const uint32_t seed = static_cast<uint32_t>(f.M + 3 * f.N + 5 * f.K
            + 7 * f.transA + 11 * f.transB + 13 * f.padded);
    problem_t pb = make_problem(f.M, f.N, f.K, f.transA, f.transB,
            f.padded ? 5 : 0, f.padded ? 3 : 0, f.padded ? 16 : 0, seed);
    std::vector<bfloat16_t> copy = pb.w;
    const nv::native_weight_ref_t ref = raw_ref(copy.data(), 32, 0);
    for (data_type_t bias_dt : {data_type_t::none, data_type_t::f32,
                 data_type_t::bf16, data_type_t::f16})
        for (data_type_t dst_dt : {data_type_t::f32, data_type_t::bf16})
            for (int ab = 0; ab < 3; ++ab) {
                static const float kAlphaBeta[][2]
                        = {{1.0f, 0.0f}, {0.75f, -1.0f}, {0.0f, 2.0f}};
                run_t r;
                r.bias_dt = bias_dt;
                r.dst_dt = dst_dt;
                r.alpha = kAlphaBeta[ab][0];
                r.beta = kAlphaBeta[ab][1];
                r.threads = (ab == 0) ? 1 : (ab == 1 ? 4 : all_threads());
                r.tag = "raw_wc0";
                run_and_expect(pb, r, ref);
            }
    // The raw weight, including a padded ldb, reaches the same layout.
    run_t r;
    r.bias_dt = data_type_t::bf16;
    r.threads = all_threads();
    for (int32_t wc : {0, 1, 2}) {
        r.tag = "raw_wc" + std::to_string(wc);
        run_raw(pb, r, 32, wc);
    }
}

std::vector<feature_t> feature_cases() {
    std::vector<feature_t> v;
    const wide_t shapes[] = {{7, 96, 63}, {65, 64, 100}, {33, 4096, 31},
            {200, 96, 1000}, {1, 32, 1}};
    for (const auto &s : shapes)
        for (bool transA : {false, true})
            for (bool transB : {false, true})
                for (bool padded : {false, true})
                    v.push_back({s.M, s.N, s.K, transA, transB, padded});
    return v;
}

INSTANTIATE_TEST_SUITE_P(Bf16, NativeBrgemmLayoutFeatures,
        ::testing::ValuesIn(feature_cases()),
        [](const ::testing::TestParamInfo<feature_t> &info) {
            const feature_t &f = info.param;
            return "M" + std::to_string(f.M) + "_N" + std::to_string(f.N) + "_K"
                    + std::to_string(f.K) + (f.transA ? "_tA" : "")
                    + (f.transB ? "_tB" : "") + (f.padded ? "_ld" : "");
        });

// ── Weight cache 2: in-place image reused across calls ──────────────────

TEST_F(NativeBrgemmLayoutTest, InPlaceImageReusedAcrossCalls) {
    for (bool transB : {true, false}) {
        problem_t pb = make_problem(16, 64, 100, false, transB, 0, 0, 0, 21);
        buffer_t image32 = prepack_bf16(pb, 32, false);
        ASSERT_TRUE(image32);
        std::vector<bfloat16_t> w = pb.w;
        run_t r;
        r.dst_dt = data_type_t::f32;
        r.threads = 4;

        r.tag = "wc2_first";
        run_and_expect(pb, r, raw_ref(w.data(), 32, 2));
        ASSERT_TRUE(packed_in_place(w));
        EXPECT_EQ(std::memcmp(w.data(), image32.get(),
                          static_cast<size_t>(pb.K) * pb.N * 2),
                0);
        r.tag = "wc2_again";
        run_and_expect(pb, r, raw_ref(w.data(), 32, 2));
        r.tag = "wc1_after_image";
        run_and_expect(pb, r, raw_ref(w.data(), 32, 1));
        r.tag = "wc0_after_image";
        run_and_expect(pb, r, raw_ref(w.data(), 32, 0));
        // The buffer keeps the first image.
        EXPECT_EQ(std::memcmp(w.data(), image32.get(),
                          static_cast<size_t>(pb.K) * pb.N * 2),
                0);
        zendnnl::lowoha::matmul::clear_matmul_weight_caches();
    }
}

TEST_F(NativeBrgemmLayoutTest, InPlaceOnlyWhenTheImageFits) {
    // Odd K pads the image past the buffer; a padded ldb is not contiguous.
    const struct {
        int K, ldb_pad;
    } cases[] = {{63, 0}, {100, 4}};
    for (const auto &c : cases) {
        problem_t pb
                = make_problem(9, 64, c.K, false, true, 0, c.ldb_pad, 0, 3);
        std::vector<bfloat16_t> w = pb.w;
        run_t r;
        r.threads = 4;
        r.tag = "wc2_out_of_place";
        run_and_expect(pb, r, raw_ref(w.data(), 32, 2));
        EXPECT_FALSE(packed_in_place(w));
        EXPECT_EQ(std::memcmp(w.data(), pb.w.data(), w.size() * 2), 0);
        run_and_expect(pb, r, raw_ref(w.data(), 32, 2));
        zendnnl::lowoha::matmul::clear_matmul_weight_caches();
    }
}

// ── Native layout cache ─────────────────────────────────────────────────

nv::native_raw_weight_t raw_bf16(
        const problem_t &pb, const void *w, int pack_nr, bool interleave) {
    nv::native_raw_weight_t r;
    r.wei_dt = data_type_t::bf16;
    r.weight = w;
    r.K = pb.K;
    r.N = pb.N;
    r.ldb = pb.ldb;
    r.transB = pb.transB;
    r.pack_nr = pack_nr;
    r.interleave_split_halves = interleave;
    return r;
}

size_t layout_bytes(const problem_t &pb, int pack_nr) {
    return nv::native_block_bytes(data_type_t::bf16, pb.K, pack_nr)
            * static_cast<size_t>(pb.N / pack_nr);
}

TEST_F(NativeBrgemmLayoutTest, CacheKeepsOneLayoutPerRequest) {
    problem_t pb = make_problem(1, 128, 64, false, true, 0, 0, 0, 31);
    std::vector<bfloat16_t> w = pb.w;
    auto layout_of = [&](int pack_nr, bool interleave, int32_t wc) {
        nv::native_layout_t l;
        EXPECT_EQ(nv::get_native_layout(
                          raw_bf16(pb, w.data(), pack_nr, interleave), wc, 0, 4,
                          l),
                status_t::success);
        return l;
    };
    const nv::native_layout_t a = layout_of(32, false, 1);
    const nv::native_layout_t b = layout_of(32, false, 2);
    EXPECT_EQ(a.owned, nullptr);
    EXPECT_EQ(b.data, a.data) << "a cached layout serves weight cache 1 and 2";
    const nv::native_layout_t c = layout_of(32, true, 1);
    EXPECT_NE(c.data, a.data);
    nv::native_layout_t e = layout_of(32, false, 0);
    EXPECT_NE(e.owned, nullptr);
    EXPECT_EQ(e.data, e.owned);
    for (const auto &x : {std::make_pair(a, false), std::make_pair(c, true),
                 std::make_pair(e, false)}) {
        buffer_t want = prepack_bf16(pb, 32, x.second);
        EXPECT_EQ(
                std::memcmp(x.first.data, want.get(), layout_bytes(pb, 32)), 0);
    }
    nv::native_free_layout(e.owned);
    EXPECT_EQ(std::memcmp(w.data(), pb.w.data(), w.size() * 2), 0)
            << "a cached layout leaves the weight raw";
    EXPECT_FALSE(packed_in_place(w));
}

TEST_F(NativeBrgemmLayoutTest, CacheCapacityPacksBeyondItForTheCall) {
    const native_brgemm_test::lru_capacity_pin_t cap(2);
    // A padded ldb keeps weight cache 2 out of place, so both modes cache.
    problem_t pb = make_problem(1, 64, 64, false, true, 0, 3, 0, 41);
    buffer_t want = prepack_bf16(pb, 32, false);
    for (int32_t wc : {1, 2}) {
        std::vector<std::vector<bfloat16_t>> ws(4, pb.w);
        for (size_t i = 0; i < ws.size(); ++i) {
            nv::native_layout_t l;
            ASSERT_EQ(
                    nv::get_native_layout(
                            raw_bf16(pb, ws[i].data(), 32, false), wc, 0, 1, l),
                    status_t::success);
            EXPECT_EQ(l.owned != nullptr, i >= 2) << "wc=" << wc << " " << i;
            EXPECT_NE(l.data, ws[i].data());
            EXPECT_EQ(std::memcmp(l.data, want.get(), layout_bytes(pb, 32)), 0)
                    << "wc=" << wc << " " << i;
            nv::native_free_layout(l.owned);
        }
        nv::native_layout_t again;
        ASSERT_EQ(nv::get_native_layout(raw_bf16(pb, ws[0].data(), 32, false),
                          wc, 0, 1, again),
                status_t::success);
        EXPECT_EQ(again.owned, nullptr) << "entries within the capacity stay";
        zendnnl::lowoha::matmul::clear_matmul_native_weight_caches();
    }
}

// A weight of N not a multiple of 32 has a layout of padded blocks, larger
// than the weight: weight cache mode 2 packs it out of place into the cache,
// as mode 1 does, even when the caller states a capacity the layout fits;
// the cached layout serves both modes and the buffer is never written.
TEST_F(NativeBrgemmLayoutTest, PaddedLayoutIsCachedOutOfPlace) {
    for (int N : {1, 40, 72})
        for (bool transB : {true, false}) {
            problem_t pb = make_problem(1, N, 64, false, transB, 0, 0, 0, 71);
            const size_t padded
                    = nv::native_packed_bytes(data_type_t::bf16, pb.K, N, 32);
            std::vector<uint8_t> want(padded, 0);
            ASSERT_EQ(
                    nv::native_pack_weight(data_type_t::bf16, pb.w.data(), pb.K,
                            N, pb.ldb, 32, transB, false, want.data(), 1),
                    status_t::success);
            for (size_t capacity : {size_t {0}, padded + 4096}) {
                SCOPED_TRACE("N=" + std::to_string(N)
                        + " transB=" + std::to_string(transB)
                        + " capacity=" + std::to_string(capacity));
                std::vector<bfloat16_t> buf(
                        std::max(pb.w.size(), capacity / 2 + 1),
                        bfloat16_t(3.0f));
                std::copy(pb.w.begin(), pb.w.end(), buf.begin());
                const std::vector<bfloat16_t> before = buf;
                for (int32_t wc : {2, 1, 2}) {
                    nv::native_layout_t l;
                    ASSERT_EQ(nv::get_native_layout(
                                      raw_bf16(pb, buf.data(), 32, false), wc,
                                      capacity, 2, l),
                            status_t::success);
                    EXPECT_NE(l.data, static_cast<const void *>(buf.data()));
                    EXPECT_EQ(l.owned, nullptr);
                    EXPECT_EQ(std::memcmp(l.data, want.data(), padded), 0);
                }
                EXPECT_EQ(nv::native_weight_layouts(buf.data()), 1u);
                EXPECT_FALSE(packed_in_place(buf));
                EXPECT_EQ(std::memcmp(buf.data(), before.data(),
                                  buf.size() * sizeof(bfloat16_t)),
                        0)
                        << "the weight buffer is not written";
                zendnnl::lowoha::matmul::clear_matmul_native_weight_caches();
            }
        }
}

TEST_F(NativeBrgemmLayoutTest, InPlaceImageServesOrRebuildsEveryRequest) {
    problem_t pb = make_problem(1, 128, 64, false, true, 0, 0, 0, 51);
    std::vector<bfloat16_t> w = pb.w;
    nv::native_layout_t img;
    ASSERT_EQ(nv::get_native_layout(
                      raw_bf16(pb, w.data(), 32, false), 2, 0, 4, img),
            status_t::success);
    ASSERT_EQ(img.data, w.data());
    ASSERT_TRUE(packed_in_place(w));
    EXPECT_TRUE(nv::native_weight_packed_in_place(
            w.data() + 7, (w.size() - 7) * sizeof(bfloat16_t)));
    EXPECT_FALSE(nv::native_weight_packed_in_place(w.data() + w.size(), 0));
    buffer_t image = prepack_bf16(pb, 32, false);
    for (int32_t wc : {0, 1, 2})
        for (bool interleave : {false, true}) {
            SCOPED_TRACE("wc=" + std::to_string(wc)
                    + " interleave=" + std::to_string(interleave));
            nv::native_layout_t l;
            ASSERT_EQ(nv::get_native_layout(
                              raw_bf16(pb, w.data(), 32, interleave), wc, 0, 4,
                              l),
                    status_t::success);
            buffer_t want = prepack_bf16(pb, 32, interleave);
            EXPECT_EQ(std::memcmp(l.data, want.get(), layout_bytes(pb, 32)), 0);
            EXPECT_EQ(l.data == w.data(), !interleave);
            nv::native_free_layout(l.owned);
        }
    EXPECT_EQ(std::memcmp(w.data(), image.get(), w.size() * 2), 0)
            << "an image is never rewritten";

    // The first 64 weight rows lie inside the image: rebuilt from it when the
    // request's capacity covers the whole image, which the rebuild reads.
    problem_t half = pb;
    half.N = 64;
    nv::native_layout_t l;
    EXPECT_EQ(nv::get_native_layout(
                      raw_bf16(half, w.data(), 32, false), 1, 0, 1, l),
            status_t::failure);
    ASSERT_EQ(nv::get_native_layout(raw_bf16(half, w.data(), 32, false), 1,
                      w.size() * sizeof(bfloat16_t), 1, l),
            status_t::success);
    buffer_t want_half = alloc_buffer(layout_bytes(half, 32));
    ASSERT_EQ(ck::prepack_weight_into_bf16(pb.w.data(), half.K, half.N,
                      half.ldb, 32, true, false, want_half.get()),
            status_t::success);
    EXPECT_EQ(std::memcmp(l.data, want_half.get(), layout_bytes(half, 32)), 0);

    // A request starting inside the image, reading past it, or of another
    // weight type fails.
    nv::native_raw_weight_t inside = raw_bf16(half, w.data() + 64, 32, false);
    EXPECT_NE(nv::get_native_layout(inside, 1, 0, 1, l), status_t::success);
    problem_t wide = pb;
    wide.N = 256;
    EXPECT_NE(nv::get_native_layout(
                      raw_bf16(wide, w.data(), 32, false), 1, 0, 1, l),
            status_t::success);
    nv::native_raw_weight_t as_s8 = raw_bf16(pb, w.data(), 32, false);
    as_s8.wei_dt = data_type_t::s8;
    EXPECT_NE(nv::get_native_layout(as_s8, 0, 0, 1, l), status_t::success);

    zendnnl::lowoha::matmul::clear_matmul_native_weight_caches();
    EXPECT_FALSE(packed_in_place(w));
}

TEST_F(NativeBrgemmLayoutTest, ConcurrentAcquisitionsShareOneLayout) {
    problem_t pb = make_problem(1, 256, 128, false, true, 0, 0, 0, 61);
    buffer_t want = prepack_bf16(pb, 32, false);
    constexpr int kCallers = 8;
    for (int32_t wc : {1, 2}) {
        std::vector<bfloat16_t> w = pb.w;
        const void *seen[kCallers] = {};
        int callers = 0, bad = 0;
#pragma omp parallel num_threads(kCallers) reduction(+ : bad)
        {
            nv::native_layout_t l;
            if (nv::get_native_layout(raw_bf16(pb, w.data(), 32, false), wc, 0,
                        2,
                        l) != status_t::success
                    || std::memcmp(l.data, want.get(), layout_bytes(pb, 32))
                            != 0)
                ++bad;
            seen[omp_get_thread_num()] = l.data;
            if (omp_get_thread_num() == 0) callers = omp_get_num_threads();
            nv::native_free_layout(l.owned);
        }
        EXPECT_EQ(bad, 0) << "wc=" << wc;
        for (int i = 1; i < callers; ++i)
            EXPECT_EQ(seen[i], seen[0]) << "wc=" << wc << " caller " << i;
        EXPECT_EQ(packed_in_place(w), wc == 2);
        zendnnl::lowoha::matmul::clear_matmul_native_weight_caches();
    }
}

// ── Callers already inside a parallel region ─────────────────────────────

// Calls made inside a parallel region, which plan one worker each.
TEST_F(NativeBrgemmLayoutTest, NestedCallersMatchTheReference) {
    problem_t pb = make_problem(33, 96, 100, false, true, 0, 0, 0, 17);
    std::vector<bfloat16_t> copy = pb.w;
    const nv::native_weight_ref_t ref = raw_ref(copy.data(), 32, 0);
    run_t r;
    r.dst_dt = data_type_t::f32;
    r.bias_dt = data_type_t::f32;
    r.threads = 8;
    constexpr int kCallers = 3;
    std::vector<io_t> io;
    for (int i = 0; i < kCallers; ++i)
        io.push_back(make_io(pb, r));
    status_t st[kCallers] = {};
#pragma omp parallel num_threads(kCallers)
    {
        for (int i = omp_get_thread_num(); i < kCallers;
                i += omp_get_num_threads())
            st[i] = execute(pb, r, ref, io[i]);
    }
    for (int i = 0; i < kCallers; ++i) {
        EXPECT_EQ(st[i], status_t::success) << i;
        expect_matches(pb, r, io[i]);
    }
}

// ── INT8 through the looper ──────────────────────────────────────────────

/// A static INT8 problem: a u8 source with a per-tensor zero point and
/// scale, or an s8 source with per-token scales; an s8 [N, K] weight with
/// per-channel scales.  Scales are small integers over powers of two, so the
/// dequantized products are exact in double.
struct int8_problem_t {
    int M = 0, N = 0, K = 0, lda = 0;
    bool src_signed = false;
    std::vector<uint8_t> a;
    std::vector<int8_t> w;
    int32_t zp = 0;
    std::vector<float> src_scale;
    std::vector<float> wei_scale;
    std::vector<float> bias;
    std::vector<double> dq;
};

int8_problem_t make_int8_problem(
        int M, int N, int K, bool src_signed, uint32_t seed) {
    int8_problem_t pb;
    pb.M = M;
    pb.N = N;
    pb.K = K;
    pb.lda = K + 5;
    pb.src_signed = src_signed;
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> byte(0, 255);
    pb.a.resize(static_cast<size_t>(M) * pb.lda);
    for (auto &x : pb.a)
        x = static_cast<uint8_t>(byte(gen));
    pb.w = random_s8(static_cast<size_t>(N) * K, seed + 1);
    pb.zp = src_signed ? 0 : 100 + static_cast<int32_t>(seed % 50);
    pb.src_scale.resize(src_signed ? M : 1);
    for (size_t i = 0; i < pb.src_scale.size(); ++i)
        pb.src_scale[i] = static_cast<float>(1 + (i + seed) % 4) / 128.0f;
    pb.wei_scale.resize(N);
    for (int n = 0; n < N; ++n)
        pb.wei_scale[n] = static_cast<float>(1 + n % 7) / 256.0f;
    pb.bias = grid_values(N, seed + 2);
    pb.dq.assign(static_cast<size_t>(M) * N, 0.0);
    for (int m = 0; m < M; ++m) {
        const uint8_t *row = pb.a.data() + static_cast<size_t>(m) * pb.lda;
        const double s = pb.src_scale[src_signed ? m : 0];
        for (int n = 0; n < N; ++n) {
            const int8_t *col = pb.w.data() + static_cast<size_t>(n) * K;
            int64_t acc = 0;
            for (int k = 0; k < K; ++k) {
                const int av = src_signed ? static_cast<int8_t>(row[k])
                                          : static_cast<int>(row[k]);
                acc += static_cast<int64_t>(av - pb.zp) * col[k];
            }
            pb.dq[static_cast<size_t>(m) * N + n]
                    = static_cast<double>(acc) * s * pb.wei_scale[n];
        }
    }
    return pb;
}

void run_int8_and_expect(const int8_problem_t &pb, const run_t &r,
        const nv::native_weight_ref_t &ref) {
    const data_type_t src_dt
            = pb.src_signed ? data_type_t::s8 : data_type_t::u8;
    const int ldc = pb.N + 3;
    buffer_t bias;
    if (r.bias_dt != data_type_t::none) {
        bias = alloc_buffer(static_cast<size_t>(pb.N) * 4);
        for (int n = 0; n < pb.N; ++n) {
            if (r.bias_dt == data_type_t::f32)
                reinterpret_cast<float *>(bias.get())[n] = pb.bias[n];
            else
                reinterpret_cast<bfloat16_t *>(bias.get())[n]
                        = bfloat16_t(pb.bias[n]);
        }
    }
    const bool f32 = r.dst_dt == data_type_t::f32;
    buffer_t dst = alloc_buffer(static_cast<size_t>(pb.M) * ldc * 4);
    for (size_t i = 0; i < static_cast<size_t>(pb.M) * ldc; ++i) {
        if (f32)
            reinterpret_cast<float *>(dst.get())[i] = kSentinel;
        else
            reinterpret_cast<bfloat16_t *>(dst.get())[i]
                    = bfloat16_t(kSentinel);
    }

    nv::GemmDescriptor desc;
    desc.M = pb.M;
    desc.N = pb.N;
    desc.K = pb.K;
    desc.lda = pb.lda;
    desc.ldb = pb.K;
    desc.ldc = ldc;
    desc.transB = true;
    desc.src_dt = src_dt;
    desc.wei_dt = data_type_t::s8;
    desc.dst_dt = r.dst_dt;
    desc.bias_dt = r.bias_dt;
    desc.bias = bias.get();
    desc.is_weights_const = true;
    desc.num_threads = r.threads;
    matmul_params params;
    params.dtypes.src = src_dt;
    params.dtypes.wei = data_type_t::s8;
    params.dtypes.dst = r.dst_dt;
    params.dtypes.bias = r.bias_dt;
    auto &qp = params.quant_params;
    qp.src_scale.buff = pb.src_scale.data();
    qp.src_scale.dt = data_type_t::f32;
    qp.src_scale.dims = pb.src_signed
            ? std::vector<int64_t> {static_cast<int64_t>(pb.M), 1}
            : std::vector<int64_t> {1};
    if (!pb.src_signed) {
        qp.src_zp.buff = &pb.zp;
        qp.src_zp.dt = data_type_t::s32;
        qp.src_zp.dims = {1};
    }
    qp.wei_scale.buff = pb.wei_scale.data();
    qp.wei_scale.dt = data_type_t::f32;
    qp.wei_scale.dims = {1, static_cast<int64_t>(pb.N)};

    std::ostringstream what;
    what << r.tag << " M=" << pb.M << " N=" << pb.N << " K=" << pb.K
         << " src=" << dt_name(src_dt) << " dst=" << dt_name(r.dst_dt)
         << " bias=" << dt_name(r.bias_dt) << " threads=" << r.threads;
    ASSERT_EQ(nv::native_brgemm_execute(
                      desc, ref, pb.a.data(), dst.get(), bias.get(), params),
            status_t::success)
            << what.str();

    int bad = 0;
    std::string first;
    for (int m = 0; m < pb.M; ++m)
        for (int n = 0; n < ldc; ++n) {
            const size_t at = static_cast<size_t>(m) * ldc + n;
            const double got = f32
                    ? reinterpret_cast<const float *>(dst.get())[at]
                    : static_cast<float>(reinterpret_cast<const bfloat16_t *>(
                              dst.get())[at]);
            double want = kSentinel, tol = 0.0;
            if (n < pb.N) {
                const double b
                        = r.bias_dt != data_type_t::none ? pb.bias[n] : 0;
                const double dq = pb.dq[static_cast<size_t>(m) * pb.N + n];
                want = dq + b;
                tol = 4.0 * kF32Unit * (std::fabs(dq) + std::fabs(b));
                if (!f32) tol += kBf16Unit * (std::fabs(want) + tol);
            }
            if (!(std::fabs(got - want) <= tol) && bad++ == 0) {
                std::ostringstream os;
                os << "first mismatch at (" << m << ", " << n << "): got "
                   << got << " want " << want << " tol " << tol;
                first = os.str();
            }
        }
    EXPECT_EQ(bad, 0) << what.str() << ": " << first;
}

struct int8_shape_t {
    int M, N, K;
};

class NativeBrgemmLayoutInt8
    : public NativeBrgemmLayoutTest,
      public ::testing::WithParamInterface<int8_shape_t> {
protected:
    void SetUp() override {
        if (!nv::detect_uarch().avx512vnni) GTEST_SKIP() << "no AVX512-VNNI";
        zendnnl::lowoha::matmul::clear_matmul_weight_caches();
    }
};

TEST_P(NativeBrgemmLayoutInt8, AllSources) {
    const int8_shape_t s = GetParam();
    const uint32_t seed = static_cast<uint32_t>(s.M * 5 + s.N * 3 + s.K);
    for (bool src_signed : {false, true}) {
        const int8_problem_t pb
                = make_int8_problem(s.M, s.N, s.K, src_signed, seed);
        const size_t size = ck::packed_weight_size_int8(s.K, s.N, 32);
        static const int kThreads[] = {1, 4, 0};
        for (int v = 0; v < 5; ++v) {
            const uint32_t t = seed + v + (src_signed ? 1 : 0);
            run_t r;
            r.threads = kThreads[t % 3] ? kThreads[t % 3] : all_threads();
            r.dst_dt = (t % 2) ? data_type_t::f32 : data_type_t::bf16;
            r.bias_dt = (t % 3 == 0) ? data_type_t::none
                    : (t % 3 == 1)   ? data_type_t::f32
                                     : data_type_t::bf16;
            {
                const int32_t wc = v < 2 ? 0 : v - 2;
                std::vector<int8_t> copy = pb.w;
                nv::native_weight_ref_t ref = raw_ref(copy.data(), 32, wc);
                ref.capacity_bytes = size;
                r.tag = "raw_wc" + std::to_string(wc);
                run_int8_and_expect(pb, r, ref);
                EXPECT_EQ(std::memcmp(copy.data(), pb.w.data(), copy.size()), 0)
                        << r.tag;
                zendnnl::lowoha::matmul::clear_matmul_weight_caches();
            }
        }
    }
}

std::vector<int8_shape_t> int8_shapes() {
    std::vector<int8_shape_t> v;
    for (int M : {1, 7, 64, 200, 1500})
        for (int K : {4, 63, 100, 1000})
            for (int N : {32, 96})
                v.push_back({M, N, K});
    for (int M : {1, 64})
        for (int K : {100, 1000})
            v.push_back({M, 4096, K});
    return v;
}

INSTANTIATE_TEST_SUITE_P(Static, NativeBrgemmLayoutInt8,
        ::testing::ValuesIn(int8_shapes()),
        [](const ::testing::TestParamInfo<int8_shape_t> &info) {
            return "M" + std::to_string(info.param.M) + "_N"
                    + std::to_string(info.param.N) + "_K"
                    + std::to_string(info.param.K);
        });

// ── Row-group schedule ──────────────────────────────────────────────────

/// The call over a raw [N, K] weight with `threads` threads into an ldc N
/// destination: bf16 x bf16 into f32, or a u8 source (zero point,
/// per-tensor scale) times s8 with per-column scales into bf16.
void schedule_call(bool int8, int M, int N, int K, int threads,
        const float *wei_scale, nv::GemmDescriptor &desc,
        matmul_params &params) {
    static const float kSrcScale = 1.0f / 64;
    static const int32_t kSrcZp = 3;
    desc.M = M;
    desc.N = N;
    desc.K = K;
    desc.lda = K;
    desc.ldb = K;
    desc.ldc = N;
    desc.transB = true;
    desc.src_dt = int8 ? data_type_t::u8 : data_type_t::bf16;
    desc.wei_dt = int8 ? data_type_t::s8 : data_type_t::bf16;
    desc.dst_dt = int8 ? data_type_t::bf16 : data_type_t::f32;
    desc.is_weights_const = true;
    desc.num_threads = threads;
    params.dtypes.src = desc.src_dt;
    params.dtypes.wei = desc.wei_dt;
    params.dtypes.dst = desc.dst_dt;
    if (int8) {
        auto &qp = params.quant_params;
        qp.src_scale.buff = &kSrcScale;
        qp.src_scale.dt = data_type_t::f32;
        qp.src_scale.dims = {1};
        qp.src_zp.buff = &kSrcZp;
        qp.src_zp.dt = data_type_t::s32;
        qp.src_zp.dims = {1};
        qp.wei_scale.buff = wei_scale;
        qp.wei_scale.dt = data_type_t::f32;
        qp.wei_scale.dims = {1, static_cast<int64_t>(N)};
    }
}

/// The call of schedule_call over the raw weight w from a into dst.
status_t run_schedule_case(bool int8, int M, int N, int K, int threads,
        const void *a, const void *w, const float *wei_scale, void *dst) {
    nv::GemmDescriptor desc;
    matmul_params params;
    schedule_call(int8, M, N, K, threads, wei_scale, desc, params);
    return nv::native_brgemm_execute(
            desc, raw_ref(w, 32, 0), a, dst, nullptr, params);
}

// The row-group schedule, pinned in a child process whose
// ZENDNNL_MATMUL_NATIVE_CONFIG the process planner reads, takes column
// blocks that do not divide evenly over the team; every tile is the same
// computation, so the result keeps the bits of a single-thread call.
TEST_F(NativeBrgemmLayoutTest, RowGroupScheduleMatchesSingleThread) {
    constexpr int kM = 300, kK = 512, kMaxN = 384;
    // Column blocks 3 on 4 threads, and 12 on 8 threads in 4 column groups.
    const struct {
        int N, threads;
    } teams[] = {{96, 4}, {kMaxN, 8}};
    std::mt19937 gen(kM);
    std::uniform_int_distribution<int> val(-64, 64);
    buffer_t a = alloc_buffer(static_cast<size_t>(kM) * kK * 2);
    buffer_t w = alloc_buffer(static_cast<size_t>(kMaxN) * kK * 2);
    ASSERT_TRUE(a && w);
    std::vector<int> a_vals(static_cast<size_t>(kM) * kK);
    std::vector<int> w_vals(static_cast<size_t>(kMaxN) * kK);
    for (int &v : a_vals)
        v = val(gen);
    for (int &v : w_vals)
        v = val(gen);
    std::vector<float> wei_scale(kMaxN);
    for (int n = 0; n < kMaxN; ++n)
        wei_scale[n] = static_cast<float>(1 + n % 7) / 4096.0f;
    native_brgemm_test::expect_in_child(
            {{"ZENDNNL_MATMUL_NATIVE_CONFIG",
                     "schedule=row_groups,domain_threads=4"},
                    {"ZENDNNL_MATMUL_NATIVE_TUNING_FILE", nullptr}},
            [&] {
        int bad = 0;
        for (const bool int8 : {false, true}) {
            if (int8 && !nv::detect_uarch().avx512vnni) continue;
            // The operands in the source and weight types.
            for (size_t i = 0; i < a_vals.size(); ++i) {
                if (int8)
                    a.get()[i] = static_cast<uint8_t>(a_vals[i] + 64);
                else
                    reinterpret_cast<bfloat16_t *>(a.get())[i]
                            = bfloat16_t(a_vals[i] / 64.0f);
            }
            for (size_t i = 0; i < w_vals.size(); ++i) {
                if (int8)
                    reinterpret_cast<int8_t *>(w.get())[i]
                            = static_cast<int8_t>(w_vals[i]);
                else
                    reinterpret_cast<bfloat16_t *>(w.get())[i]
                            = bfloat16_t(w_vals[i] / 64.0f);
            }
            for (const auto &t : teams) {
                const std::string what = std::string(int8 ? "s8" : "bf16")
                        + " N=" + std::to_string(t.N)
                        + " threads=" + std::to_string(t.threads);
                nv::GemmDescriptor desc;
                matmul_params params;
                schedule_call(int8, kM, t.N, kK, t.threads, wei_scale.data(),
                        desc, params);
                const nv::native_plan_t plan = nv::native_planner().plan(
                        nv::make_native_problem(desc, params, nullptr, 32,
                                nv::decide_int8_call(desc, params, nullptr)));
                const int workers = static_cast<int>(std::min<int64_t>(
                        plan.threads,
                        static_cast<int64_t>(plan.m_blocks) * plan.n_tiles));
                if (plan.col_groups <= 0
                        || nv::native_partition_t(plan, kM, 0, plan.n_tiles)
                                        .kind(workers)
                                != nv::native_partition_kind_t::row_groups) {
                    std::fprintf(stderr,
                            "%s: col_groups %d of %d workers is not "
                            "the row-group schedule\n",
                            what.c_str(), plan.col_groups, workers);
                    ++bad;
                }
                const size_t bytes
                        = static_cast<size_t>(kM) * t.N * (int8 ? 2 : 4);
                buffer_t one = alloc_buffer(bytes);
                buffer_t team = alloc_buffer(bytes);
                const bool ran
                        = run_schedule_case(int8, kM, t.N, kK, 1, a.get(),
                                  w.get(), wei_scale.data(), one.get())
                                == status_t::success
                        && run_schedule_case(int8, kM, t.N, kK, t.threads,
                                   a.get(), w.get(), wei_scale.data(),
                                   team.get())
                                == status_t::success;
                if (!ran || std::memcmp(one.get(), team.get(), bytes) != 0) {
                    std::fprintf(stderr,
                            "%s: the team differs from one thread\n",
                            what.c_str());
                    ++bad;
                }
            }
        }
        return bad;
    }, "the row-group schedule");
}

} // namespace
