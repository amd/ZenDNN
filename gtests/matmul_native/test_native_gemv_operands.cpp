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

// M = 1 calls of matmul_direct, on the GEMV kernels of the weight's one
// layout, read the operands of each call: calls on one weight that carry
// other values than earlier calls (in other buffers or in the same ones) get
// the result of their own operands, at every weight cache mode.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "gtest_utils.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_brgemm_looper.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_weight_cache.hpp"
#include "lowoha_operators/matmul/matmul_native/native_layout_policy.hpp"
#include "native_brgemm_test_utils.hpp"

namespace {

namespace mm = zendnnl::lowoha::matmul;
namespace nat = zendnnl::lowoha::matmul::native;
namespace nt = native_brgemm_test;
using zendnnl::common::matmul_algo_t;
using zendnnl::common::matmul_config_t;
using zendnnl::common::post_op_type_t;
using zendnnl::memory::data_type_t;
using zendnnl::memory::status_t;

constexpr data_type_t kBf16 = data_type_t::bf16;
constexpr data_type_t kF32 = data_type_t::f32;
constexpr data_type_t kS8 = data_type_t::s8;
constexpr data_type_t kU8 = data_type_t::u8;

bool gemv_isa() {
    const nat::UarchParams &u = nat::detect_uarch();
    return u.avx512bf16 && u.avx512vnni && u.avx512f;
}

/// An M = 1 call on one thread on a constant [K, N] weight (or [N, K] with
/// transB), at weight cache mode 1, through matmul_direct with lowoha_algo
/// native_brgemm.  The weight stays allocated until the process exits, so
/// that no later weight takes its address in the pointer-keyed layout
/// cache.  INT8: a per-tensor f32 source scale (and an s32 zero point for a
/// u8 source) and f32 weight scales per column.
struct gemv_call_t {
    nt::call_t c;
    nt::buffer_t src, wei, dst, ref;
    float src_scale = 0.5f;
    int32_t src_zp = 0;
    std::vector<float> wei_scales;

    gemv_call_t(data_type_t src_dt, data_type_t wei_dt, data_type_t dst_dt,
            int N, int K, uint32_t seed, bool transB = false) {
        const bool int8 = wei_dt == kS8;
        const bool s8_src = src_dt == kS8;
        src = nt::random_buffer(src_dt, static_cast<size_t>(K), seed,
                int8 ? (s8_src ? -8.0f : 0.0f) : -1.0f, int8 ? 15.0f : 1.0f);
        wei = nt::kept_until_exit(
                nt::random_buffer(wei_dt, static_cast<size_t>(K) * N, seed + 1,
                        int8 ? -8.0f : -1.0f, int8 ? 8.0f : 1.0f));
        dst = nt::make_buffer(static_cast<size_t>(N) * nt::dt_bytes(dst_dt));
        ref = nt::make_buffer(static_cast<size_t>(N) * nt::dt_bytes(dst_dt));
        c.M = 1;
        c.N = N;
        c.K = K;
        c.transB = transB;
        c.src = src.get();
        c.wei = wei.get();
        c.dst = dst.get();
        c.is_weights_const = true;
        c.params.dtypes.src = src_dt;
        c.params.dtypes.wei = wei_dt;
        c.params.dtypes.dst = dst_dt;
        c.params.num_threads = 1;
        c.params.lowoha_algo = matmul_algo_t::native_brgemm;
        c.params.weight_cache_type = 1;
        if (int8) {
            wei_scales.resize(N);
            for (int n = 0; n < N; ++n)
                wei_scales[n] = 0.125f + 0.002f * static_cast<float>(n % 17);
            auto &q = c.params.quant_params;
            q.src_scale.buff = &src_scale;
            q.src_scale.dt = kF32;
            q.src_scale.dims = {1, 1};
            if (!s8_src) {
                q.src_zp.buff = &src_zp;
                q.src_zp.dt = data_type_t::s32;
                q.src_zp.dims = {1, 1};
            }
            q.wei_scale.buff = wei_scales.data();
            q.wei_scale.dt = kF32;
            q.wei_scale.dims = {1, N};
        }
    }

    void set_bias(const void *bias, data_type_t dt) {
        c.bias = bias;
        c.params.dtypes.bias = bias == nullptr ? data_type_t::none : dt;
    }

    /// The call through matmul_direct.
    status_t run() const { return nt::run_matmul(c); }

    /// The call, then its result against the reference algorithm on the
    /// weight as it is now, within the band of the call's data types.
    ::testing::AssertionResult runs_to_reference() const {
        const status_t st = run();
        if (st != status_t::success)
            return ::testing::AssertionFailure()
                    << "matmul_direct status " << static_cast<int>(st);
        if (nt::run_reference(c, c.wei, ref.get()) != status_t::success)
            return ::testing::AssertionFailure() << "no reference result";
        const bool int8 = c.params.dtypes.wei == kS8;
        return nt::within_band(dst.get(), c.N, ref.get(), c.N,
                c.params.dtypes.dst, 1, c.N, rtol_bf16,
                nt::comparison_abs_bound(
                        c.K, epsilon_bf16, 1.0f, 0.0f, true, int8, false));
    }
};

/// What the library writes to stdout (its log) while f runs.
template <typename F>
std::string stdout_of(F &&f) {
    ::testing::internal::CaptureStdout();
    f();
    return ::testing::internal::GetCapturedStdout();
}

size_t count_of(const std::string &text, const std::string &what) {
    size_t n = 0;
    for (size_t at = text.find(what); at != std::string::npos;
            at = text.find(what, at + what.size()))
        ++n;
    return n;
}

// An INT8 GEMV call reads the bias and the weight scales of every call: on
// one weight, a bias in another buffer, no bias, other weight scales, and
// new values in the same bias and weight-scale buffers each give the result
// of the call's own operands.
TEST(NativeGemvOperands, Int8GemvReadsTheBiasAndScalesOfEveryCall) {
    if (!gemv_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    uint32_t seed = 7100;
    for (const data_type_t src_dt : {kU8, kS8})
        for (const data_type_t dst_dt : {kF32, kBf16}) {
            SCOPED_TRACE(std::string(src_dt == kU8 ? "u8" : "s8") + " source, "
                    + (dst_dt == kF32 ? "f32" : "bf16") + " dst");
            gemv_call_t g(src_dt, kS8, dst_dt, 64, 128, seed += 4);
            g.src_zp = src_dt == kU8 ? 3 : 0;
            const nt::buffer_t bias_a
                    = nt::random_buffer(kF32, 64, seed + 2, -4.0f, 4.0f);
            const nt::buffer_t bias_b
                    = nt::random_buffer(kF32, 64, seed + 3, 5.0f, 10.0f);
            g.set_bias(bias_a.get(), kF32);
            EXPECT_TRUE(g.runs_to_reference()) << "first call";
            g.set_bias(bias_b.get(), kF32);
            EXPECT_TRUE(g.runs_to_reference()) << "a bias in another buffer";
            g.set_bias(nullptr, kF32);
            EXPECT_TRUE(g.runs_to_reference()) << "no bias";
            std::vector<float> other_scales(g.wei_scales.size());
            for (size_t n = 0; n < other_scales.size(); ++n)
                other_scales[n] = 2.0f * g.wei_scales[n];
            g.c.params.quant_params.wei_scale.buff = other_scales.data();
            g.set_bias(bias_a.get(), kF32);
            EXPECT_TRUE(g.runs_to_reference())
                    << "weight scales in another buffer";
            for (int n = 0; n < 64; ++n)
                bias_a.as<float>()[n] += 20.0f;
            EXPECT_TRUE(g.runs_to_reference())
                    << "new values in the same bias buffer";
            for (float &s : other_scales)
                s *= 0.5f;
            EXPECT_TRUE(g.runs_to_reference())
                    << "new values in the same weight-scale buffer";
        }
}

// An INT8 GEMV call whose source scale and zero point change from call to
// call (a source quantized per call) reads the one layout of its weight:
// the weight keeps at most one layout over the calls (none at weight cache
// mode 0), and every result follows its own scale and zero point.
TEST(NativeGemvOperands, Int8GemvKeepsItsLayoutWhenTheSourceScaleChanges) {
    if (!gemv_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    gemv_call_t g(kU8, kS8, kF32, 64, 1024, 7300);
    const struct {
        float scale;
        int32_t zp;
    } calls[] = {{0.5f, 0}, {0.25f, 0}, {0.75f, 4}, {0.0625f, 9}};
    for (const auto &q : calls) {
        SCOPED_TRACE("source scale " + std::to_string(q.scale) + ", zero point "
                + std::to_string(q.zp));
        g.src_scale = q.scale;
        g.src_zp = q.zp;
        EXPECT_TRUE(g.runs_to_reference());
        EXPECT_LE(nt::native_layouts(g.c.wei), 1u);
    }
}

// A GEMV call converts a bf16 bias on every call: a bias buffer that keeps
// its address and gets new values between two calls (a reused temporary)
// gives the second call its new values, on bf16 calls of 64 and of 3
// columns and on INT8 calls.
TEST(NativeGemvOperands, GemvCallsConvertABf16BiasOnEveryCall) {
    if (!gemv_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    const struct {
        data_type_t src, wei, dst;
        int N, K;
        const char *name;
    } cases[] = {
            {kBf16, kBf16, kF32, 64, 128, "bf16, N 64"},
            {kBf16, kBf16, kBf16, 3, 128, "bf16, N 3"},
            {kU8, kS8, kF32, 64, 128, "u8 x s8, N 64"},
    };
    uint32_t seed = 7500;
    for (const auto &cs : cases) {
        SCOPED_TRACE(cs.name);
        gemv_call_t call(cs.src, cs.wei, cs.dst, cs.N, cs.K, seed += 4);
        const nt::buffer_t bias
                = nt::random_buffer(kBf16, cs.N, seed + 2, -1.0f, 1.0f);
        call.set_bias(bias.get(), kBf16);
        EXPECT_TRUE(call.runs_to_reference()) << "first values";
        const nt::buffer_t next
                = nt::random_buffer(kBf16, cs.N, seed + 3, 8.0f, 12.0f);
        std::memcpy(bias.get(), next.get(), bias.bytes);
        EXPECT_TRUE(call.runs_to_reference())
                << "new values in the same bias buffer";
    }
}

// One square weight read as [K, N], then as [N, K] (ldb K), then as [K, N]
// again on the same thread: each call gets the layout of its own reading of
// the buffer, bf16 and INT8.
TEST(NativeGemvOperands, GemvLayoutsFollowHowTheCallReadsTheBuffer) {
    if (!gemv_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    const struct {
        data_type_t src, wei;
        const char *name;
    } cases[] = {{kBf16, kBf16, "bf16"}, {kU8, kS8, "u8 x s8"}};
    uint32_t seed = 7700;
    for (const auto &cs : cases) {
        SCOPED_TRACE(cs.name);
        gemv_call_t call(cs.src, cs.wei, kF32, 64, 64, seed += 4);
        for (const bool transB : {false, true, false}) {
            call.c.transB = transB;
            EXPECT_TRUE(call.runs_to_reference())
                    << (transB ? "read as [N, K]" : "read as [K, N]");
        }
    }
}

/// Restores the process-wide weight cache mode and LRU capacity.
struct config_restore_t {
    matmul_config_t &cfg = matmul_config_t::instance();
    const int32_t weight_cache = cfg.get_weight_cache();
    const uint32_t capacity = cfg.get_lru_cache_capacity();
    ~config_restore_t() {
        cfg.set_weight_cache(weight_cache);
        cfg.set_lru_cache_capacity(capacity);
    }
};

// At weight cache mode 0, per call or process-wide, a GEMV call reads the
// weight its buffer holds: a buffer refilled with a new weight between two
// calls gives the second call the new weight's result, bf16 and INT8.
TEST(NativeGemvOperands, GemvCallsAtWeightCacheModeZeroReadTheirBuffer) {
    if (!gemv_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    const config_restore_t restore;
    const struct {
        data_type_t src, wei;
        const char *name;
    } cases[] = {{kBf16, kBf16, "bf16"}, {kU8, kS8, "u8 x s8"}};
    uint32_t seed = 7900;
    for (const bool process_wide : {false, true})
        for (const auto &cs : cases) {
            SCOPED_TRACE(std::string(cs.name)
                    + (process_wide ? ", ZENDNNL_MATMUL_WEIGHT_CACHE=0"
                                    : ", weight_cache_type 0 on the call"));
            restore.cfg.set_weight_cache(process_wide ? 0 : 2);
            gemv_call_t call(cs.src, cs.wei, kF32, 64, 128, seed += 4);
            if (!process_wide) call.c.params.weight_cache_type = 0;
            EXPECT_TRUE(call.runs_to_reference()) << "first weight";
            const bool int8 = cs.wei == kS8;
            const nt::buffer_t next = nt::random_buffer(cs.wei, 64 * 128,
                    seed + 5, int8 ? -8.0f : -1.0f, int8 ? 8.0f : 1.0f);
            std::memcpy(call.wei.get(), next.get(), call.wei.bytes);
            EXPECT_TRUE(call.runs_to_reference())
                    << "a new weight in the same buffer";
        }
}

// At weight cache mode 1 with a capacity of one layout, the M = 1 calls of
// the first weight cache its layout, and those of two more weights pack
// theirs for the call and cache none; all are served, bf16 and INT8.
TEST(NativeGemvOperands, GemvCallsPastTheCacheCapacityPackForTheCall) {
    if (!gemv_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    const config_restore_t restore;
    const struct {
        data_type_t src, wei;
        const char *name;
    } cases[] = {{kBf16, kBf16, "bf16"}, {kU8, kS8, "u8 x s8"}};
    uint32_t seed = 8100;
    for (const auto &cs : cases) {
        SCOPED_TRACE(cs.name);
        restore.cfg.set_weight_cache(1);
        nat::clear_native_weight_caches();
        restore.cfg.set_lru_cache_capacity(1);
        for (int w = 0; w < 3; ++w) {
            gemv_call_t call(cs.src, cs.wei, kF32, 64, 128, seed += 4);
            EXPECT_TRUE(call.runs_to_reference()) << "weight " << w;
            EXPECT_TRUE(call.runs_to_reference()) << "weight " << w << " again";
            EXPECT_EQ(nt::native_layouts(call.c.wei), w == 0 ? 1u : 0u)
                    << "weight " << w;
        }
        restore.cfg.set_lru_cache_capacity(restore.capacity);
    }
}

// A constant weight whose buffer takes a new weight between calls, at
// weight cache mode 1 per call: the M = 1 calls, bf16 and INT8 with u8 and
// s8 sources, get the new weight's result, on the calling thread and on
// another one, as the calls of M > 1 do (with lowoha_algo native_brgemm and
// under ZENDNNL_MATMUL_ALGO=10).
TEST(NativeGemvOperands, GemvLayoutsFollowNewBytesOfTheirWeight) {
    if (!gemv_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    const config_restore_t restore;
    const struct {
        data_type_t src, wei;
        const char *name;
    } cases[] = {{kBf16, kBf16, "bf16"}, {kU8, kS8, "u8 x s8"},
            {kS8, kS8, "s8 x s8"}};
    const int N = 64, K = 128, rows = 4;
    uint32_t seed = 8200;
    for (const bool algo10 : {false, true})
        for (const auto &cs : cases) {
            SCOPED_TRACE(std::string(cs.name)
                    + (algo10 ? ", ZENDNNL_MATMUL_ALGO=10"
                              : ", lowoha_algo native_brgemm"));
            restore.cfg.set_weight_cache(2);
            std::unique_ptr<nt::algo_pin_t> pin;
            gemv_call_t call(cs.src, cs.wei, kF32, N, K, seed += 8);
            if (algo10) {
                call.c.params.lowoha_algo = matmul_algo_t::none;
                pin.reset(new nt::algo_pin_t(matmul_algo_t::native_brgemm));
            }
            const bool int8 = cs.wei == kS8;
            const auto refill = [&](uint32_t s) {
                const nt::buffer_t next
                        = nt::random_buffer(cs.wei, static_cast<size_t>(K) * N,
                                s, int8 ? -8.0f : -1.0f, int8 ? 8.0f : 1.0f);
                std::memcpy(call.wei.get(), next.get(), call.wei.bytes);
            };
            EXPECT_TRUE(call.runs_to_reference()) << "first weight";
            refill(seed + 4);
            EXPECT_TRUE(call.runs_to_reference())
                    << "a new weight in the same buffer";

            // The native tiles of the same weight: M > 1 on its bytes now.
            nt::call_t tiles = call.c;
            tiles.M = rows;
            const nt::buffer_t src = nt::random_buffer(cs.src,
                    static_cast<size_t>(rows) * K, seed + 5,
                    int8 ? (cs.src == kS8 ? -8.0f : 0.0f) : -1.0f,
                    int8 ? 15.0f : 1.0f);
            const nt::buffer_t dst = nt::make_buffer(rows * N * sizeof(float));
            const nt::buffer_t ref = nt::make_buffer(rows * N * sizeof(float));
            const nt::buffer_t raw = nt::make_buffer(call.wei.bytes);
            std::memcpy(raw.get(), call.wei.get(), call.wei.bytes);
            tiles.src = src.get();
            tiles.dst = dst.get();
            ASSERT_EQ(nt::run_matmul(tiles), status_t::success);
            ASSERT_EQ(nt::run_reference(tiles, raw.get(), ref.get()),
                    status_t::success);
            EXPECT_TRUE(nt::within_band(dst.get(), N, ref.get(), N, kF32, rows,
                    N, rtol_bf16,
                    nt::comparison_abs_bound(
                            K, epsilon_bf16, 1.0f, 0.0f, true, int8, false)))
                    << "M = " << rows << " on the new weight";

            refill(seed + 6);
            std::thread other([&] {
                EXPECT_TRUE(call.runs_to_reference())
                        << "a new weight, on another thread";
            });
            other.join();
            EXPECT_TRUE(call.runs_to_reference())
                    << "the same weight, on this thread again";
            ::reset_grp_matmul_caches();
        }
}

/// A call of rows rows on the weight of g, with its own source (of g's
/// source type and range) and an f32 dst.
struct rows_call_t {
    nt::call_t c;
    nt::buffer_t src, dst, ref;
    rows_call_t(const gemv_call_t &g, int rows, int threads, uint32_t seed) {
        c = g.c;
        c.M = rows;
        c.params.num_threads = threads;
        c.params.dtypes.dst = kF32;
        const data_type_t src_dt = g.c.params.dtypes.src;
        const bool int8 = g.c.params.dtypes.wei == kS8;
        src = nt::random_buffer(src_dt, static_cast<size_t>(rows) * g.c.K, seed,
                int8 ? (src_dt == kS8 ? -8.0f : 0.0f) : -1.0f,
                int8 ? 15.0f : 1.0f);
        dst = nt::make_buffer(static_cast<size_t>(rows) * g.c.N * 4);
        ref = nt::make_buffer(static_cast<size_t>(rows) * g.c.N * 4);
        c.src = src.get();
        c.dst = dst.get();
    }
    /// The result against the reference algorithm on raw_wei.
    ::testing::AssertionResult matches(const void *raw_wei) const {
        if (nt::run_reference(c, raw_wei, ref.get()) != status_t::success)
            return ::testing::AssertionFailure() << "no reference result";
        return nt::within_band(dst.get(), c.N, ref.get(), c.N, kF32, c.M, c.N,
                rtol_bf16,
                nt::comparison_abs_bound(c.K, epsilon_bf16, 1.0f, 0.0f, true,
                        c.params.dtypes.wei == kS8, false));
    }
};

/// Under ZENDNNL_MATMUL_ALGO=10 at weight cache mode 2 a first call on a
/// bf16 weight rewrites its buffer in place.
struct in_place_mode_t {
    const config_restore_t restore;
    const nt::algo_pin_t pin {matmul_algo_t::native_brgemm};
    in_place_mode_t() { restore.cfg.set_weight_cache(2); }
};

// M = 1 calls at weight cache mode 1 racing the first call of M = 8 on their
// weight at mode 2, which rewrites a bf16 buffer in place (an s8 weight's
// layout, with its column sums, is cached out of place, and its buffer keeps
// its bytes): on fresh weights, the M = 1 call on one thread and the M = 8
// call on eight start together, and each gets the weight's result whatever
// the order; bf16, and INT8 with u8 and s8 sources.
TEST(NativeGemvOperands, GemvCallsRacingAnInPlaceRewriteGetTheWeightsResult) {
    if (!gemv_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    const in_place_mode_t mode;
    const struct {
        data_type_t src, wei;
        int N, K;
        const char *name;
    } cases[] = {{kBf16, kBf16, 1024, 2048, "bf16"},
            {kU8, kS8, 256, 1024, "u8 x s8"}, {kS8, kS8, 256, 1024, "s8 x s8"}};
    const int kRuns = 12;
    for (int i = 0; i < 3 * kRuns; ++i) {
        const auto &cs = cases[i / kRuns];
        const int run = i % kRuns, N = cs.N, K = cs.K;
        SCOPED_TRACE(std::string(cs.name) + ", run " + std::to_string(run));
        gemv_call_t one(cs.src, cs.wei, kF32, N, K, 8500 + 8 * run);
        one.c.params.lowoha_algo = matmul_algo_t::none;
        const nt::buffer_t raw = nt::make_buffer(one.wei.bytes);
        std::memcpy(raw.get(), one.wei.get(), one.wei.bytes);
        rows_call_t eight(one, 8, 8, 8502 + 8 * run);
        eight.c.params.weight_cache_type = 2;
        std::atomic<int> ready {0};
        status_t st_one = status_t::failure, st_eight = status_t::failure;
        {
            std::thread a([&] {
                ready.fetch_add(1);
                while (ready.load() < 2) {}
                st_one = one.run();
            });
            std::thread b([&] {
                ready.fetch_add(1);
                while (ready.load() < 2) {}
                st_eight = nt::run_matmul(eight.c);
            });
            a.join();
            b.join();
        }
        ASSERT_EQ(st_eight, status_t::success);
        EXPECT_TRUE(eight.matches(raw.get())) << "the M = 8 call";
        ASSERT_EQ(st_one, status_t::success);
        ASSERT_EQ(nt::run_reference(one.c, raw.get(), one.ref.get()),
                status_t::success);
        EXPECT_TRUE(nt::within_band(one.dst.get(), N, one.ref.get(), N, kF32, 1,
                N, rtol_bf16,
                nt::comparison_abs_bound(K, epsilon_bf16, 1.0f, 0.0f, true,
                        cs.wei == kS8, false)))
                << "the M = 1 call";
        if (cs.wei == kS8) {
            EXPECT_EQ(std::memcmp(one.wei.get(), raw.get(), raw.bytes), 0)
                    << "the buffer of an s8 weight was rewritten";
        }
    }
}

/// The f64 model of the dst of c (M x N, row-major) on the [K, N] weight
/// raw: bf16 operands, or a u8 / s8 source on an s8 weight with c's source
/// scale and zero point and per-tensor or per-column weight scales; then
/// c's bias and its unary post-op.
std::vector<double> model_of(const nt::call_t &c, const void *raw) {
    const auto &q = c.params.quant_params;
    const data_type_t src_dt = c.params.dtypes.src,
                      wei_dt = c.params.dtypes.wei;
    const bool int8 = wei_dt == kS8;
    const double src_scale = int8 && q.src_scale.buff != nullptr
            ? *static_cast<const float *>(q.src_scale.buff)
            : 1.0;
    const double zp = int8 && q.src_zp.buff != nullptr
            ? *static_cast<const int32_t *>(q.src_zp.buff)
            : 0.0;
    const bool per_column = int8 && q.wei_scale.buff != nullptr
            && q.wei_scale.dims.size() == 2 && q.wei_scale.dims[1] == c.N;
    std::vector<double> out(static_cast<size_t>(c.M) * c.N);
    for (int m = 0; m < c.M; ++m)
        for (int n = 0; n < c.N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < c.K; ++k)
                acc += (nt::load_f32(
                                c.src, src_dt, static_cast<size_t>(m) * c.K + k)
                               - zp)
                        * nt::load_f32(
                                raw, wei_dt, static_cast<size_t>(k) * c.N + n);
            if (int8 && q.wei_scale.buff != nullptr)
                acc *= src_scale
                        * static_cast<const float *>(
                                q.wei_scale.buff)[per_column ? n : 0];
            else if (int8)
                acc *= src_scale;
            if (c.bias != nullptr)
                acc += nt::load_f32(c.bias, c.params.dtypes.bias, n);
            for (const mm::matmul_post_op &po : c.params.postop_) {
                if (po.po_type == post_op_type_t::relu)
                    acc = std::max(acc, 0.0);
                else if (po.po_type == post_op_type_t::tanh)
                    acc = std::tanh(acc);
                else if (po.po_type == post_op_type_t::gelu_erf)
                    acc = 0.5 * acc * (1.0 + std::erf(acc / std::sqrt(2.0)));
            }
            out[static_cast<size_t>(m) * c.N + n] = acc;
        }
    return out;
}

/// c's dst against model_of(c, raw): within the rounding of the dst (bf16
/// or f32) of the model's value, plus f32 accumulation over the call's K
/// terms and the kernels' approximation of the post-op.
::testing::AssertionResult matches_model(const nt::call_t &c, const void *raw) {
    const std::vector<double> want = model_of(c, raw);
    const data_type_t dst_dt = c.params.dtypes.dst;
    const double rel = dst_dt == kBf16 ? 1.0 / 128 : 1e-4;
    double terms = 0.0;
    for (int k = 0; k < c.K; ++k)
        terms = std::max(terms,
                static_cast<double>(std::fabs(
                        nt::load_f32(c.src, c.params.dtypes.src, k))));
    for (size_t i = 0; i < want.size(); ++i) {
        const double got = nt::load_f32(c.dst, dst_dt, i);
        const double tol
                = rel * std::fabs(want[i]) + 1e-4 + 1e-5 * c.K * (terms + 1.0);
        if (!(std::fabs(got - want[i]) <= tol))
            return ::testing::AssertionFailure()
                    << "dst[" << i << "] = " << got << ", the model " << want[i]
                    << " (M " << c.M << ")";
    }
    return ::testing::AssertionSuccess();
}

/// A kind of M = 1 call and what the first call on its weight does at
/// weight cache mode 2 under ZENDNNL_MATMUL_ALGO=10.
struct mode_two_case_t {
    data_type_t src;
    int N;
    bool in_place; ///< it rewrites the buffer (else caches out of place)
    const char *name;
};

const mode_two_case_t kModeTwoCases[] = {{kBf16, 64, true, "bf16, N 64"},
        {kBf16, 4, false, "bf16, N 4"}, {kU8, 64, false, "u8 x s8, N 64"},
        {kS8, 64, false, "s8 x s8, N 64"}};

/// A call of g's kind, at weight cache mode wc, with lowoha_algo
/// native_brgemm or (algo10) without lowoha_algo, for a process at
/// ZENDNNL_MATMUL_ALGO=10 then.
void set_route(gemv_call_t &g, int32_t wc, bool algo10) {
    g.c.params.weight_cache_type = wc;
    if (algo10) g.c.params.lowoha_algo = matmul_algo_t::none;
}

// M = 1 calls of the kinds of kModeTwoCases at weight cache modes 0, 1 and
// 2, with lowoha_algo native_brgemm and under ZENDNNL_MATMUL_ALGO=10, read
// the weight's one layout.  The first and the next call get the f64 model's
// result; the weight has one native layout (none at mode 0), an image in
// place only for a bf16 weight of whole column blocks at mode 2 under
// ZENDNNL_MATMUL_ALGO=10, and its buffer keeps its bytes otherwise.
TEST(NativeGemvOperands, GemvCallsReadOneLayoutAtEveryMode) {
    if (!gemv_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    const config_restore_t restore;
    restore.cfg.set_weight_cache(2);
    const int K = 256;
    uint32_t seed = 9500;
    for (const bool algo10 : {false, true})
        for (const int32_t wc : {0, 1, 2})
            for (const mode_two_case_t &cs : kModeTwoCases) {
                SCOPED_TRACE(std::string(cs.name)
                        + (algo10 ? ", ZENDNNL_MATMUL_ALGO=10"
                                  : ", lowoha_algo native_brgemm")
                        + ", weight cache mode " + std::to_string(wc));
                const nt::algo_pin_t pin(algo10 ? matmul_algo_t::native_brgemm
                                                : matmul_algo_t::none);
                gemv_call_t call(cs.src, cs.src == kBf16 ? kBf16 : kS8, kF32,
                        cs.N, K, seed += 4);
                set_route(call, wc, algo10);
                const nt::buffer_t raw = nt::make_buffer(call.wei.bytes);
                std::memcpy(raw.get(), call.wei.get(), call.wei.bytes);
                for (const char *which : {"the first call", "the next call"}) {
                    ASSERT_EQ(nt::run_matmul(call.c), status_t::success)
                            << which;
                    EXPECT_TRUE(matches_model(call.c, raw.get())) << which;
                }
                const bool in_place = cs.in_place && algo10 && wc == 2;
                EXPECT_EQ(
                        std::memcmp(call.wei.get(), raw.get(), raw.bytes) != 0,
                        in_place);
                EXPECT_EQ(nat::native_weight_packed_in_place(
                                  call.wei.get(), call.wei.bytes),
                        in_place);
                EXPECT_EQ(
                        nt::native_layouts(call.wei.get()), wc == 0 ? 0u : 1u);
            }
}

// The calls of GemvCallsReadOneLayoutAtEveryMode each log matmul_direct's
// line of the native BRGEMM on the native layout at the info level.
TEST(NativeGemvOperands, GemvCallsLogTheNativeLine) {
    if (!gemv_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    nt::expect_in_child({{"ZENDNNL_API_LOG_LEVEL", "3"}}, [] {
        int bad = 0;
        size_t calls = 0;
        const config_restore_t restore;
        restore.cfg.set_weight_cache(2);
        uint32_t seed = 9600;
        const std::string log = stdout_of([&] {
            for (const bool algo10 : {false, true})
                for (const int32_t wc : {0, 1, 2})
                    for (const mode_two_case_t &cs : kModeTwoCases) {
                        const nt::algo_pin_t pin(algo10
                                        ? matmul_algo_t::native_brgemm
                                        : matmul_algo_t::none);
                        gemv_call_t call(cs.src, cs.src == kBf16 ? kBf16 : kS8,
                                kF32, cs.N, 256, seed += 4);
                        set_route(call, wc, algo10);
                        for (int i = 0; i < 2; ++i, ++calls)
                            if (nt::run_matmul(call.c) != status_t::success)
                                ++bad;
                    }
        });
        const size_t native_lines
                = count_of(log, "kernel=native_brgemm, layout=native_nr");
        if (native_lines != calls) {
            std::fprintf(stderr, "%zu native lines (%zu expected)\n",
                    native_lines, calls);
            ++bad;
        }
        return bad;
    }, "ZENDNNL_API_LOG_LEVEL=3");
}

// At weight cache mode 2, M = 1 calls run on the tiles' layout whichever
// call on the weight comes first: on fresh weights of the kinds of
// kModeTwoCases, an M = 1 call, then an M = 8 call, then the M = 1 call
// again, and the M = 8 call first, then the M = 1 call twice, with a bias
// (f32 or bf16) or none, per-tensor or per-column weight scales (INT8), a
// unary post-op or none, and an f32 or bf16 dst, each get the f64 model's
// result.
TEST(NativeGemvOperands, AtWeightCacheModeTwoM1CallsGetTheModelsResult) {
    if (!gemv_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    const in_place_mode_t mode;
    const int K = 256, rows = 8;
    const post_op_type_t post_ops[]
            = {post_op_type_t::none, post_op_type_t::gelu_erf,
                    post_op_type_t::relu, post_op_type_t::tanh};
    uint32_t seed = 9700;
    int variant = 0;
    for (const mode_two_case_t &cs : kModeTwoCases)
        for (const bool m1_first : {true, false})
            for (const bool per_column : {true, false}) {
                if (cs.src == kBf16 && !per_column) continue;
                for (const post_op_type_t po : post_ops) {
                    ++variant;
                    const bool int8 = cs.src != kBf16;
                    const data_type_t dst_dt = variant % 2 ? kBf16 : kF32;
                    const data_type_t bias_dt = variant % 3 == 1 ? kBf16 : kF32;
                    const bool bias = variant % 3 != 0;
                    SCOPED_TRACE(std::string(cs.name)
                            + (m1_first ? ", M = 1 first" : ", M = 8 first")
                            + (int8 && !per_column ? ", per-tensor scale" : "")
                            + ", variant " + std::to_string(variant));
                    gemv_call_t one(cs.src, int8 ? kS8 : kBf16, dst_dt, cs.N, K,
                            seed += 8);
                    one.c.params.lowoha_algo = matmul_algo_t::none;
                    one.c.params.weight_cache_type = 2;
                    if (cs.src == kU8) one.src_zp = 3;
                    float scale = 0.0625f;
                    if (int8 && !per_column) {
                        auto &ws = one.c.params.quant_params.wei_scale;
                        ws.buff = &scale;
                        ws.dims = {1, 1};
                    }
                    const nt::buffer_t bias_buf = nt::random_buffer(bias_dt,
                            static_cast<size_t>(cs.N), seed + 1,
                            int8 ? -16.0f : -1.0f, int8 ? 16.0f : 1.0f);
                    if (bias) one.set_bias(bias_buf.get(), bias_dt);
                    if (po != post_op_type_t::none) {
                        mm::matmul_post_op m;
                        m.po_type = po;
                        if (po == post_op_type_t::relu) m.alpha = 0.0f;
                        one.c.params.postop_.push_back(m);
                    }
                    const nt::buffer_t raw = nt::make_buffer(one.wei.bytes);
                    std::memcpy(raw.get(), one.wei.get(), one.wei.bytes);
                    rows_call_t eight(one, rows, rows, seed + 2);
                    eight.c.params.dtypes.dst = dst_dt;
                    const nt::buffer_t eight_dst
                            = nt::make_buffer(static_cast<size_t>(rows) * cs.N
                                    * nt::dt_bytes(dst_dt));
                    eight.c.dst = eight_dst.get();
                    const auto run_one = [&](const char *which) {
                        ASSERT_EQ(nt::run_matmul(one.c), status_t::success)
                                << which;
                        EXPECT_TRUE(matches_model(one.c, raw.get())) << which;
                    };
                    const auto run_eight = [&] {
                        ASSERT_EQ(nt::run_matmul(eight.c), status_t::success)
                                << "M = 8";
                        EXPECT_TRUE(matches_model(eight.c, raw.get()))
                                << "M = 8";
                    };
                    if (m1_first) {
                        run_one("M = 1 first");
                        run_eight();
                    } else {
                        run_eight();
                        run_one("M = 1 after M = 8");
                    }
                    run_one("M = 1 again");
                }
            }
}

// An M = 1 INT8 call with a bias of any type the native BRGEMM takes (f32,
// bf16, f16, s32, s8 or u8) gives the f64 model's result.
TEST(NativeGemvOperands, Int8GemvCallsTakeEveryBiasType) {
    if (!gemv_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    const struct {
        data_type_t dt;
        const char *name;
    } biases[] = {{kF32, "f32"}, {kBf16, "bf16"}, {data_type_t::f16, "f16"},
            {data_type_t::s32, "s32"}, {kS8, "s8"}, {kU8, "u8"}};
    const int N = 64, K = 96;
    uint32_t seed = 9900;
    for (const auto &b : biases) {
        SCOPED_TRACE(std::string("a bias of ") + b.name);
        gemv_call_t call(kU8, kS8, kF32, N, K, seed += 4);
        call.src_zp = 3;
        const nt::buffer_t bias
                = nt::random_buffer(b.dt, static_cast<size_t>(N), seed + 2,
                        b.dt == kU8 ? 0.0f : -16.0f, 16.0f);
        call.set_bias(bias.get(), b.dt);
        const status_t st = call.run();
        EXPECT_EQ(st, status_t::success);
        if (st == status_t::success) {
            EXPECT_TRUE(matches_model(call.c, call.wei.get()));
        }
    }
}

// An M = 1 call whose weight rows are ldb apart with ldb below the row
// length (N - 1 for a [K, N] weight, K - 1 for [N, K]) is turned away as
// the same call at M = 2 is: native_brgemm_refusal and matmul_direct return
// unimplemented and leave dst untouched.
TEST(NativeGemvOperands, OneRowCallsTurnAwayAnLdbBelowTheWeightRow) {
    if (!gemv_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    const struct {
        data_type_t src, wei;
        int N;
        bool transB;
        const char *name;
    } cases[] = {{kBf16, kBf16, 64, false, "bf16, N 64"},
            {kBf16, kBf16, 64, true, "bf16, N 64, transB"},
            {kBf16, kBf16, 4, false, "bf16, N 4"},
            {kU8, kS8, 64, false, "u8 x s8, N 64"},
            {kS8, kS8, 64, true, "s8 x s8, N 64, transB"}};
    const int K = 128;
    uint32_t seed = 9100;
    for (const auto &cs : cases) {
        SCOPED_TRACE(cs.name);
        gemv_call_t call(cs.src, cs.wei, kF32, cs.N, K, seed += 4, cs.transB);
        nt::call_t c = call.c;
        c.ldb = (cs.transB ? K : cs.N) - 1;
        const auto untouched = [](const nt::buffer_t &out,
                                       const std::vector<uint8_t> &prior) {
            return std::memcmp(out.get(), prior.data(), prior.size()) == 0;
        };

        std::memset(call.dst.get(), 0x5a, call.dst.bytes);
        const std::vector<uint8_t> prior(call.dst.as<uint8_t>(),
                call.dst.as<uint8_t>() + call.dst.bytes);
        EXPECT_EQ(nt::run_matmul(c), status_t::unimplemented) << "M = 1";
        EXPECT_TRUE(untouched(call.dst, prior));
        EXPECT_TRUE(nt::refused_with(c, status_t::unimplemented)) << "M = 1";

        // The same call at M = 2, on the tiles.
        nt::call_t tiles = c;
        tiles.M = 2;
        const nt::buffer_t src = nt::random_buffer(cs.src,
                static_cast<size_t>(tiles.M) * K, seed + 1,
                cs.src == kBf16 ? -1.0f : 0.0f, cs.src == kBf16 ? 1.0f : 15.0f);
        const nt::buffer_t dst
                = nt::make_buffer(static_cast<size_t>(tiles.M) * cs.N * 4);
        std::memset(dst.get(), 0x5a, dst.bytes);
        const std::vector<uint8_t> tiles_prior(
                dst.as<uint8_t>(), dst.as<uint8_t>() + dst.bytes);
        tiles.src = src.get();
        tiles.dst = dst.get();
        EXPECT_EQ(nt::run_matmul(tiles), status_t::unimplemented) << "M = 2";
        EXPECT_TRUE(untouched(dst, tiles_prior));
        EXPECT_TRUE(nt::refused_with(tiles, status_t::unimplemented))
                << "M = 2";
        EXPECT_EQ(nt::native_layouts(c.wei), 0u);
    }
}

} // namespace
