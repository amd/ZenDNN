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

// The GEMV plan choices of the native BRGEMM read the operands of each call:
// calls at M = 1 on one weight that carry other values than earlier calls
// (in other buffers or in the same ones) get the result of their own
// operands, as on the native tiles.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#if defined(__GLIBC__) \
        && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 33))
#include <malloc.h>
#define NATIVE_GEMV_TEST_HEAP_IN_USE 1
#endif

#include <gtest/gtest.h>

#include "gtest_utils.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_brgemm_looper.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_gemv.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"
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

/// An M = 1 call on one thread that asks for the native BRGEMM
/// (lowoha_algo native_brgemm) on a constant [K, N] weight (or [N, K] with
/// transB), at weight cache mode 1: a GEMV kernel serves a call only below
/// mode 2.  The weight stays allocated until the process exits, so that no
/// later weight takes its address in the pointer-keyed caches of the GEMV
/// packs.  INT8: a per-tensor f32 source scale (and an s32 zero point for a
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

    /// The GEMV kernel matmul_direct runs the call on.
    nat::native_gemv_t kernel() const { return nt::gemv_kernel(c); }

    /// matmul_direct, then its result against the reference algorithm on the
    /// weight as it is now, within the band of the call's data types.
    ::testing::AssertionResult runs_to_reference() const {
        const status_t st = nt::run_matmul(c);
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
            ASSERT_EQ(static_cast<int>(g.kernel()),
                    static_cast<int>(nat::native_gemv_t::int8_bkc));
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
// call (a source quantized per call) reuses the pack of its weight: no call
// after the first packs the weight again (the API log at the info level
// names where each call's pack came from), and every result follows its own
// scale and zero point.
TEST(NativeGemvOperands, Int8GemvKeepsItsPackWhenTheSourceScaleChanges) {
    if (!gemv_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    nt::expect_in_child({{"ZENDNNL_API_LOG_LEVEL", "3"}}, [] {
        int bad = 0;
        gemv_call_t g(kU8, kS8, kF32, 64, 1024, 7300);
        if (g.kernel() != nat::native_gemv_t::int8_bkc) {
            std::fprintf(stderr, "the call does not take int8_bkc\n");
            return 1;
        }
        const struct {
            float scale;
            int32_t zp;
        } calls[] = {{0.5f, 0}, {0.25f, 0}, {0.75f, 4}, {0.0625f, 9}};
        const std::string log = stdout_of([&] {
            for (const auto &q : calls) {
                g.src_scale = q.scale;
                g.src_zp = q.zp;
                const ::testing::AssertionResult r = g.runs_to_reference();
                if (!r) {
                    std::fprintf(stderr, "scale %g zp %d: %s\n", q.scale, q.zp,
                            r.message());
                    ++bad;
                }
            }
        });
        const size_t gemv_lines = count_of(log, "Native INT8 BKC-GEMV");
        const size_t repacks = count_of(log, "pack=thread_local");
        if (gemv_lines != 4 || repacks != 0) {
            std::fprintf(stderr,
                    "%zu INT8 GEMV lines (4 expected), %zu calls packed the "
                    "weight for themselves (0 expected)\n",
                    gemv_lines, repacks);
            ++bad;
        }
        return bad;
    }, "ZENDNNL_API_LOG_LEVEL=3");
}

// A GEMV call converts a bf16 bias on every call: a bias buffer that keeps
// its address and gets new values between two calls (a reused temporary)
// gives the second call its new values, on the BKC and narrow kernels of
// bf16 calls and on the INT8 kernel.
TEST(NativeGemvOperands, GemvCallsConvertABf16BiasOnEveryCall) {
    if (!gemv_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    using g = nat::native_gemv_t;
    const struct {
        data_type_t src, wei, dst;
        int N, K;
        g kernel;
    } cases[] = {
            {kBf16, kBf16, kF32, 64, 128, g::bf16_bkc},
            {kBf16, kBf16, kBf16, 3, 128, g::bf16_narrow},
            {kU8, kS8, kF32, 64, 128, g::int8_bkc},
    };
    uint32_t seed = 7500;
    for (const auto &cs : cases) {
        SCOPED_TRACE(nat::native_gemv_name(cs.kernel));
        gemv_call_t call(cs.src, cs.wei, cs.dst, cs.N, cs.K, seed += 4);
        ASSERT_EQ(static_cast<int>(call.kernel()), static_cast<int>(cs.kernel));
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
// again on the same thread: each call gets the pack of its own reading of
// the buffer, on the BF16 and INT8 BKC kernels.
TEST(NativeGemvOperands, GemvPacksFollowHowTheCallReadsTheBuffer) {
    if (!gemv_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    using g = nat::native_gemv_t;
    const struct {
        data_type_t src, wei;
        g kernel;
    } cases[] = {{kBf16, kBf16, g::bf16_bkc}, {kU8, kS8, g::int8_bkc}};
    uint32_t seed = 7700;
    for (const auto &cs : cases) {
        SCOPED_TRACE(nat::native_gemv_name(cs.kernel));
        gemv_call_t call(cs.src, cs.wei, kF32, 64, 64, seed += 4);
        for (const bool transB : {false, true, false}) {
            call.c.transB = transB;
            ASSERT_EQ(static_cast<int>(call.kernel()),
                    static_cast<int>(cs.kernel));
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
// calls gives the second call the new weight's result, on the BF16 and INT8
// BKC kernels.
TEST(NativeGemvOperands, GemvCallsAtWeightCacheModeZeroReadTheirBuffer) {
    if (!gemv_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    const config_restore_t restore;
    using g = nat::native_gemv_t;
    const struct {
        data_type_t src, wei;
        g kernel;
    } cases[] = {{kBf16, kBf16, g::bf16_bkc}, {kU8, kS8, g::int8_bkc}};
    uint32_t seed = 7900;
    for (const bool process_wide : {false, true})
        for (const auto &cs : cases) {
            SCOPED_TRACE(std::string(nat::native_gemv_name(cs.kernel))
                    + (process_wide ? ", ZENDNNL_MATMUL_WEIGHT_CACHE=0"
                                    : ", weight_cache_type 0 on the call"));
            restore.cfg.set_weight_cache(process_wide ? 0 : 2);
            gemv_call_t call(cs.src, cs.wei, kF32, 64, 128, seed += 4);
            if (!process_wide) call.c.params.weight_cache_type = 0;
            ASSERT_EQ(static_cast<int>(call.kernel()),
                    static_cast<int>(cs.kernel));
            EXPECT_TRUE(call.runs_to_reference()) << "first weight";
            const bool int8 = cs.wei == kS8;
            const nt::buffer_t next = nt::random_buffer(cs.wei, 64 * 128,
                    seed + 5, int8 ? -8.0f : -1.0f, int8 ? 8.0f : 1.0f);
            std::memcpy(call.wei.get(), next.get(), call.wei.bytes);
            EXPECT_TRUE(call.runs_to_reference())
                    << "a new weight in the same buffer";
        }
}

// With ZENDNNL_LRU_CACHE_CAPACITY entries cached, a GEMV call on another
// weight packs for itself, as the native tiles do, and is served, on the
// BF16 and INT8 BKC kernels.
TEST(NativeGemvOperands, GemvCallsPastTheCacheCapacityPackForTheCall) {
    if (!gemv_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    const config_restore_t restore;
    using g = nat::native_gemv_t;
    const struct {
        data_type_t src, wei;
        g kernel;
    } cases[] = {{kBf16, kBf16, g::bf16_bkc}, {kU8, kS8, g::int8_bkc}};
    uint32_t seed = 8100;
    for (const auto &cs : cases) {
        SCOPED_TRACE(nat::native_gemv_name(cs.kernel));
        restore.cfg.set_lru_cache_capacity(1);
        for (int w = 0; w < 3; ++w) {
            gemv_call_t call(cs.src, cs.wei, kF32, 64, 128, seed += 4);
            ASSERT_EQ(static_cast<int>(call.kernel()),
                    static_cast<int>(cs.kernel));
            EXPECT_TRUE(call.runs_to_reference()) << "weight " << w;
            EXPECT_TRUE(call.runs_to_reference()) << "weight " << w << " again";
        }
        restore.cfg.set_lru_cache_capacity(restore.capacity);
    }
}

// A constant weight whose buffer takes a new weight between calls, at
// weight cache mode 1 per call, with lowoha_algo native_brgemm and under
// ZENDNNL_MATMUL_ALGO=10 (a GEMV choice is made only below mode 2): the
// M = 1 calls on the BF16 and INT8 BKC kernels get the new weight's result,
// on the calling thread and on another one, as the calls of M > 1 on the
// native tiles do.
TEST(NativeGemvOperands, GemvPacksFollowNewBytesOfTheirWeight) {
    if (!gemv_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    const config_restore_t restore;
    using g = nat::native_gemv_t;
    const struct {
        data_type_t src, wei;
        g kernel;
    } cases[] = {{kBf16, kBf16, g::bf16_bkc}, {kU8, kS8, g::int8_bkc},
            {kS8, kS8, g::int8_bkc}};
    const int N = 64, K = 128, rows = 4;
    uint32_t seed = 8200;
    for (const bool algo10 : {false, true})
        for (const auto &cs : cases) {
            SCOPED_TRACE(std::string(nat::native_gemv_name(cs.kernel)) + ", "
                    + (cs.src == kS8 ? "s8" : "") + (cs.src == kU8 ? "u8" : "")
                    + (algo10 ? " ZENDNNL_MATMUL_ALGO=10"
                              : " lowoha_algo native_brgemm"));
            restore.cfg.set_weight_cache(2);
            std::unique_ptr<nt::algo_pin_t> pin;
            gemv_call_t call(cs.src, cs.wei, kF32, N, K, seed += 8);
            if (algo10) {
                call.c.params.lowoha_algo = matmul_algo_t::none;
                pin.reset(new nt::algo_pin_t(matmul_algo_t::native_brgemm));
            }
            ASSERT_EQ(static_cast<int>(call.kernel()),
                    static_cast<int>(cs.kernel));
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

// One thread keeps giving a constant weight new bytes in its first 64, which
// the zero first source element multiplies, and runs an M = 1 call after
// each change, which packs the weight again and replaces its cached pack,
// while other threads keep running M = 1 calls on the weight: on the BF16
// and INT8 BKC kernels, at weight cache mode 1 per call, with lowoha_algo
// native_brgemm and under ZENDNNL_MATMUL_ALGO=10.  No call reads a pack
// after it is freed (AddressSanitizer reports such a read), and every call
// gets the bits of the first call, which match the reference algorithm.
// The caches are cleared between rounds, when no call runs.
TEST(NativeGemvOperands, ReplacedGemvPacksOutliveTheCallsReadingThem) {
    if (!gemv_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    const config_restore_t restore;
    using g = nat::native_gemv_t;
    const struct {
        data_type_t src, wei;
        g kernel;
    } cases[] = {{kBf16, kBf16, g::bf16_bkc}, {kU8, kS8, g::int8_bkc}};
    constexpr size_t kHead = 64;
    const int N = 256, K = 512, kReaders = 4, kRounds = 4, kChanges = 48;
    uint32_t seed = 8700;
    for (const bool algo10 : {false, true})
        for (const auto &cs : cases) {
            SCOPED_TRACE(std::string(nat::native_gemv_name(cs.kernel))
                    + (algo10 ? ", ZENDNNL_MATMUL_ALGO=10"
                              : ", lowoha_algo native_brgemm"));
            restore.cfg.set_weight_cache(2);
            std::unique_ptr<nt::algo_pin_t> pin;
            gemv_call_t call(cs.src, cs.wei, kF32, N, K, seed += 8);
            if (algo10) {
                call.c.params.lowoha_algo = matmul_algo_t::none;
                pin.reset(new nt::algo_pin_t(matmul_algo_t::native_brgemm));
            }
            ASSERT_EQ(static_cast<int>(call.kernel()),
                    static_cast<int>(cs.kernel));
            nt::store_f32(call.src.get(), cs.src, 0, 0.0f);
            const bool int8 = cs.wei == kS8;
            const nt::buffer_t other
                    = nt::random_buffer(cs.wei, kHead / nt::dt_bytes(cs.wei),
                            seed + 4, int8 ? -8.0f : -1.0f, int8 ? 8.0f : 1.0f);
            uint8_t heads[2][kHead];
            std::memcpy(heads[0], call.wei.get(), kHead);
            std::memcpy(heads[1], other.get(), kHead);
            ASSERT_TRUE(call.runs_to_reference());
            const std::vector<uint8_t> want(call.dst.as<uint8_t>(),
                    call.dst.as<uint8_t>() + call.dst.bytes);

            for (int round = 0; round < kRounds; ++round) {
                SCOPED_TRACE("round " + std::to_string(round));
                std::atomic<bool> stop {false};
                std::atomic<int> started {0}, bad {0};
                // One M = 1 call on the weight into a dst of its own.
                const auto run_one = [&](const nt::buffer_t &dst) {
                    nt::call_t c = call.c;
                    c.dst = dst.get();
                    if (nt::run_matmul(c) != status_t::success
                            || std::memcmp(dst.get(), want.data(), want.size())
                                    != 0)
                        bad.fetch_add(1);
                };
                std::vector<std::thread> readers;
                for (int r = 0; r < kReaders; ++r)
                    readers.emplace_back([&] {
                        const nt::buffer_t dst = nt::make_buffer(want.size());
                        run_one(dst);
                        started.fetch_add(1);
                        while (!stop.load())
                            run_one(dst);
                    });
                while (started.load() < kReaders)
                    std::this_thread::yield();
                const nt::buffer_t dst = nt::make_buffer(want.size());
                for (int i = 1; i <= kChanges; ++i) {
                    std::memcpy(call.wei.get(), heads[i % 2], kHead);
                    run_one(dst);
                }
                stop.store(true);
                for (std::thread &t : readers)
                    t.join();
                std::memcpy(call.wei.get(), heads[0], kHead);
                EXPECT_EQ(bad.load(), 0) << "calls with another result";
                ::reset_grp_matmul_caches();
            }
        }
}

/// The descriptor native_layout_matmul builds for c.
nat::GemmDescriptor descriptor_of(const nt::call_t &c) {
    nat::GemmDescriptor d;
    d.M = c.M;
    d.N = c.N;
    d.K = c.K;
    d.lda = c.lda_or_tight();
    d.ldb = c.ldb_or_tight();
    d.ldc = c.ldc_or_tight();
    d.transA = c.transA;
    d.transB = c.transB;
    d.alpha = c.alpha;
    d.beta = c.beta;
    d.src_dt = c.params.dtypes.src;
    d.wei_dt = c.params.dtypes.wei;
    d.dst_dt = c.params.dtypes.dst;
    d.bias_dt = c.params.dtypes.bias;
    d.bias = c.bias;
    d.is_weights_const = c.is_weights_const;
    d.num_threads = c.params.num_threads;
    d.gated_act = c.gated;
    return d;
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

/// Under ZENDNNL_MATMUL_ALGO=10 at weight cache mode 2 a first call of the
/// tiles rewrites a bf16 buffer in place.
struct in_place_mode_t {
    const config_restore_t restore;
    const nt::algo_pin_t pin {matmul_algo_t::native_brgemm};
    in_place_mode_t() { restore.cfg.set_weight_cache(2); }
};

// The worst order of a bf16_bkc call (at weight cache mode 1: a GEMV choice
// is made only below mode 2) and the first call of the tiles on its weight
// at mode 2: the GEMV call finds the raw weight, then the tiles' first call
// rewrites the buffer in place, then the GEMV call runs.  It is served from
// the image, with the weight's result, and the buffer keeps the image.
TEST(NativeGemvOperands, GemvCallAfterAnInPlaceRewriteRunsOnTheImage) {
    if (!gemv_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    const in_place_mode_t mode;
    gemv_call_t call(kBf16, kBf16, kF32, 64, 256, 8300);
    call.c.params.lowoha_algo = matmul_algo_t::none;
    // What native_layout_matmul hands native_brgemm_execute when it finds
    // the raw weight.
    const nat::GemmDescriptor desc = descriptor_of(call.c);
    const nat::native_gemv_plan_t gemv
            = nat::decide_native_gemv(desc, call.c.params);
    ASSERT_EQ(static_cast<int>(gemv.kernel),
            static_cast<int>(nat::native_gemv_t::bf16_bkc));
    nat::native_weight_ref_t ref;
    ref.weight = call.c.wei;
    ref.weight_cache_mode = 1;
    const nt::buffer_t raw = nt::make_buffer(call.wei.bytes);
    std::memcpy(raw.get(), call.wei.get(), call.wei.bytes);

    rows_call_t first(call, 16, 4, 8301);
    first.c.params.weight_cache_type = 2;
    ASSERT_EQ(nt::run_matmul(first.c), status_t::success);
    ASSERT_NE(std::memcmp(call.wei.get(), raw.get(), raw.bytes), 0)
            << "the first call of the tiles did not rewrite the buffer";
    EXPECT_TRUE(first.matches(raw.get()));

    ASSERT_EQ(nat::native_brgemm_execute(desc, ref, call.c.src, call.dst.get(),
                      nullptr, call.c.params, gemv),
            status_t::success);
    ASSERT_EQ(nt::run_reference(call.c, raw.get(), call.ref.get()),
            status_t::success);
    EXPECT_TRUE(nt::within_band(call.dst.get(), 64, call.ref.get(), 64, kF32, 1,
            64, rtol_bf16,
            nt::comparison_abs_bound(
                    256, epsilon_bf16, 1.0f, 0.0f, true, false, false)));
    EXPECT_NE(std::memcmp(call.wei.get(), raw.get(), raw.bytes), 0)
            << "the buffer no longer holds the image";
}

// GEMV calls at weight cache mode 1 racing the first call of the tiles on
// their weight at mode 2, which rewrites a bf16 buffer in place (an s8
// weight's layout, with its column sums, is cached out of place, and its
// buffer keeps its bytes): on fresh weights, an M = 1 call on one thread
// and an M = 8 call on eight start together, and each gets the weight's
// result whatever the order, on bf16_bkc and on int8_bkc with u8 and s8
// sources.
TEST(NativeGemvOperands, GemvCallsRacingAnInPlaceRewriteGetTheWeightsResult) {
    if (!gemv_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    const in_place_mode_t mode;
    using g = nat::native_gemv_t;
    const struct {
        data_type_t src, wei;
        int N, K;
        g kernel;
    } cases[] = {{kBf16, kBf16, 1024, 2048, g::bf16_bkc},
            {kU8, kS8, 256, 1024, g::int8_bkc},
            {kS8, kS8, 256, 1024, g::int8_bkc}};
    const int kRuns = 12;
    for (int i = 0; i < 3 * kRuns; ++i) {
        const auto &cs = cases[i / kRuns];
        const int run = i % kRuns, N = cs.N, K = cs.K;
        SCOPED_TRACE(std::string(nat::native_gemv_name(cs.kernel))
                + (cs.src == kS8 ? ", s8 source" : "") + ", run "
                + std::to_string(run));
        gemv_call_t one(cs.src, cs.wei, kF32, N, K, 8500 + 8 * run);
        one.c.params.lowoha_algo = matmul_algo_t::none;
        ASSERT_EQ(static_cast<int>(one.kernel()), static_cast<int>(cs.kernel));
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
                st_one = nt::run_matmul(one.c);
            });
            std::thread b([&] {
                ready.fetch_add(1);
                while (ready.load() < 2) {}
                st_eight = nt::run_matmul(eight.c);
            });
            a.join();
            b.join();
        }
        ASSERT_EQ(st_one, status_t::success);
        ASSERT_EQ(st_eight, status_t::success);
        ASSERT_EQ(nt::run_reference(one.c, raw.get(), one.ref.get()),
                status_t::success);
        EXPECT_TRUE(nt::within_band(one.dst.get(), N, one.ref.get(), N, kF32, 1,
                N, rtol_bf16,
                nt::comparison_abs_bound(K, epsilon_bf16, 1.0f, 0.0f, true,
                        cs.wei == kS8, false)))
                << "the M = 1 call";
        EXPECT_TRUE(eight.matches(raw.get())) << "the M = 8 call";
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

/// A GEMV kind at weight cache mode 1 and what the first call of the tiles
/// on its weight does at mode 2.
struct mode_two_case_t {
    data_type_t src;
    int N;
    nat::native_gemv_t kernel; ///< at weight cache mode 1
    bool in_place; ///< the tiles rewrite the buffer (else cache out of place)
};

const mode_two_case_t kModeTwoCases[]
        = {{kBf16, 64, nat::native_gemv_t::bf16_bkc, true},
                {kBf16, 4, nat::native_gemv_t::bf16_narrow, false},
                {kU8, 64, nat::native_gemv_t::int8_bkc, false},
                {kS8, 64, nat::native_gemv_t::int8_bkc, false}};

std::string mode_two_name(const mode_two_case_t &cs) {
    return std::string(nat::native_gemv_name(cs.kernel))
            + (cs.src == kS8 ? ", s8 source" : "")
            + (cs.src == kU8 ? ", u8 source" : "");
}

// At weight cache mode 2 (a constant weight under ZENDNNL_MATMUL_ALGO=10) a
// weight has one layout, the tiles', which serves its M = 1 calls: an M = 1
// call of a kind that takes a GEMV kernel at mode 1 runs on the tiles; the
// first one rewrites a bf16 weight of whole column blocks in place (and
// caches the layout of the other weights out of place); no GEMV pack of the
// weight exists afterwards; and the first and the next call get the f64
// model's result.
TEST(NativeGemvOperands, AtWeightCacheModeTwoAnM1CallMakesTheTilesLayout) {
    if (!gemv_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    const in_place_mode_t mode;
    const int K = 256;
    uint32_t seed = 9500;
    for (const mode_two_case_t &cs : kModeTwoCases) {
        SCOPED_TRACE(mode_two_name(cs));
        gemv_call_t call(cs.src, cs.src == kBf16 ? kBf16 : kS8, kF32, cs.N, K,
                seed += 4);
        call.c.params.lowoha_algo = matmul_algo_t::none;
        ASSERT_EQ(static_cast<int>(call.kernel()), static_cast<int>(cs.kernel))
                << "at weight cache mode 1";
        call.c.params.weight_cache_type = 2;
        EXPECT_EQ(static_cast<int>(call.kernel()),
                static_cast<int>(nat::native_gemv_t::none));
        const nt::buffer_t raw = nt::make_buffer(call.wei.bytes);
        std::memcpy(raw.get(), call.wei.get(), call.wei.bytes);
        for (const char *which : {"the first call", "the next call"}) {
            ASSERT_EQ(nt::run_matmul(call.c), status_t::success) << which;
            EXPECT_TRUE(matches_model(call.c, raw.get())) << which;
        }
        EXPECT_EQ(std::memcmp(call.wei.get(), raw.get(), raw.bytes) != 0,
                cs.in_place);
        EXPECT_EQ(nat::native_weight_packed_in_place(
                          call.wei.get(), call.wei.bytes),
                cs.in_place);
        EXPECT_EQ(nt::native_layouts(call.wei.get()), 1u);
        const nat::PrepackedWeightKey key {
                call.wei.get(), K, cs.N, cs.N, false};
        EXPECT_EQ(nat::BF16BKCWeightCache::instance().find(key), nullptr);
        EXPECT_EQ(
                nat::INT8KContiguousWeightCache::instance().find(key), nullptr);
    }
}

// At weight cache mode 2, M = 1 calls run on the tiles' layout whichever
// call on the weight comes first: on fresh weights of the GEMV kinds, an
// M = 1 call, then an M = 8 call, then the M = 1 call again, and the M = 8
// call first, then the M = 1 call twice, with a bias (f32 or bf16) or none,
// per-tensor or per-column weight scales (INT8), a unary post-op the GEMV
// kernels fuse or none, and an f32 or bf16 dst, each get the f64 model's
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
                    SCOPED_TRACE(mode_two_name(cs)
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

// An M = 1 call on a GEMV choice with a null src, weight or dst, or a bf16
// source of lda below K, is turned away as the native tiles turn away the
// same call at M = 2: native_brgemm_execute returns failure with the reason
// of their check, and leaves dst untouched.  matmul_direct passes such calls
// on to it with ZENDNNL_DIAGNOSTICS_ENABLE=0.
TEST(NativeGemvOperands, GemvChoicesTurnAwayWhatTheTilesTurnAway) {
    if (!gemv_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    using g = nat::native_gemv_t;
    const struct {
        data_type_t src, wei;
        int N;
        g kernel;
    } cases[] = {{kBf16, kBf16, 64, g::bf16_bkc},
            {kBf16, kBf16, 4, g::bf16_narrow}, {kU8, kS8, 64, g::int8_bkc}};
    const struct {
        const char *name, *reason;
    } faults[] = {{"a null src", "a null src, weight or dst"},
            {"a null weight", "a null src, weight or dst"},
            {"a null dst", "a null src, weight or dst"},
            {"lda K - 1", "a bad shape or leading dimension"}};
    const int K = 128, rows = 2;
    uint32_t seed = 8900;
    for (const auto &cs : cases)
        for (int f = 0; f < 4; ++f) {
            // The INT8 check takes any lda.
            if (f == 3 && cs.src != kBf16) continue;
            SCOPED_TRACE(std::string(nat::native_gemv_name(cs.kernel)) + ", "
                    + faults[f].name);
            gemv_call_t call(cs.src, cs.wei, kF32, cs.N, K, seed += 4);
            nt::call_t c = call.c;
            if (f == 0) c.src = nullptr;
            if (f == 1) c.wei = nullptr;
            if (f == 2) c.dst = nullptr;
            if (f == 3) c.lda = K - 1;
            const nat::GemmDescriptor desc = descriptor_of(c);
            const nat::native_gemv_plan_t gemv
                    = nat::decide_native_gemv(desc, c.params);
            ASSERT_EQ(
                    static_cast<int>(gemv.kernel), static_cast<int>(cs.kernel));
            nat::native_weight_ref_t ref;
            ref.weight = c.wei;
            ref.weight_cache_mode = 1;
            const auto expect_turned_away
                    = [&](const nt::call_t &x, const nt::buffer_t &out,
                              const nat::native_gemv_plan_t &p, int pack_nr) {
                std::memset(out.get(), 0x5a, out.bytes);
                const std::vector<uint8_t> prior(
                        out.as<uint8_t>(), out.as<uint8_t>() + out.bytes);
                nat::native_weight_ref_t r = ref;
                r.pack_nr = pack_nr;
                const char *why = nullptr;
                EXPECT_EQ(nat::native_brgemm_execute(descriptor_of(x), r, x.src,
                                  x.dst, nullptr, x.params, p, &why),
                        status_t::failure);
                EXPECT_STREQ(why, faults[f].reason);
                EXPECT_EQ(
                        std::memcmp(out.get(), prior.data(), prior.size()), 0);
            };
            expect_turned_away(c, call.dst, gemv, 0);

            // The same call at M = 2, on the tiles.
            nt::call_t tiles = c;
            tiles.M = rows;
            const nt::buffer_t src = nt::random_buffer(cs.src,
                    static_cast<size_t>(rows) * K, seed + 1,
                    cs.src == kU8 ? 0.0f : -1.0f, cs.src == kU8 ? 15.0f : 1.0f);
            const nt::buffer_t dst
                    = nt::make_buffer(static_cast<size_t>(rows) * cs.N * 4);
            if (tiles.src != nullptr) tiles.src = src.get();
            if (tiles.dst != nullptr) tiles.dst = dst.get();
            ASSERT_EQ(static_cast<int>(nt::gemv_kernel(tiles)),
                    static_cast<int>(g::none));
            SCOPED_TRACE("M = " + std::to_string(rows));
            expect_turned_away(tiles, dst, nat::native_gemv_plan_t {},
                    nat::kNativeLayoutPackNr);
        }
}

// An M = 1 call on a GEMV choice whose weight rows are ldb apart with ldb
// below the row length (N - 1 for a [K, N] weight, K - 1 for [N, K]) is
// turned away as the same call at M = 2 is: native_brgemm_refusal and
// matmul_direct return unimplemented and leave dst untouched, and
// native_brgemm_execute, given the GEMV choice, returns unimplemented with
// the layout rule's reason.
TEST(NativeGemvOperands, GemvChoicesTurnAwayAnLdbBelowTheWeightRow) {
    if (!gemv_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    using g = nat::native_gemv_t;
    const struct {
        data_type_t src, wei;
        int N;
        bool transB;
        g kernel;
    } cases[] = {{kBf16, kBf16, 64, false, g::bf16_bkc},
            {kBf16, kBf16, 64, true, g::bf16_bkc},
            {kBf16, kBf16, 4, false, g::bf16_narrow},
            {kU8, kS8, 64, false, g::int8_bkc},
            {kS8, kS8, 64, true, g::int8_bkc}};
    const int K = 128;
    uint32_t seed = 9100;
    for (const auto &cs : cases) {
        SCOPED_TRACE(std::string(nat::native_gemv_name(cs.kernel))
                + (cs.transB ? ", transB" : ""));
        gemv_call_t call(cs.src, cs.wei, kF32, cs.N, K, seed += 4, cs.transB);
        nt::call_t c = call.c;
        c.ldb = (cs.transB ? K : cs.N) - 1;
        const nat::GemmDescriptor desc = descriptor_of(c);
        const nat::native_gemv_plan_t gemv
                = nat::decide_native_gemv(desc, c.params);
        ASSERT_EQ(static_cast<int>(gemv.kernel), static_cast<int>(cs.kernel));
        const auto untouched = [](const nt::buffer_t &out,
                                       const std::vector<uint8_t> &prior) {
            return std::memcmp(out.get(), prior.data(), prior.size()) == 0;
        };

        std::memset(call.dst.get(), 0x5a, call.dst.bytes);
        const std::vector<uint8_t> prior(call.dst.as<uint8_t>(),
                call.dst.as<uint8_t>() + call.dst.bytes);
        nat::native_weight_ref_t ref;
        ref.weight = c.wei;
        ref.weight_cache_mode = 1;
        const char *why = nullptr;
        EXPECT_EQ(nat::native_brgemm_execute(desc, ref, c.src, c.dst, nullptr,
                          c.params, gemv, &why),
                status_t::unimplemented);
        EXPECT_STREQ(why, "ldb below the weight row length");
        EXPECT_TRUE(untouched(call.dst, prior));
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

// The buffers the GEMV kernels keep for their thread go with it: threads
// that grow every one of them (a bf16 bias converted to f32, the pack of a
// weight at weight cache mode 0 with its column sums, an s8 source made u8)
// and exit leave the heap in use as it was, within half of the smallest of
// them (the u8 copy of the source) per thread, so each one counts.
TEST(NativeGemvOperands, GemvBuffersGoWithTheirThread) {
#ifdef NATIVE_GEMV_TEST_HEAP_IN_USE
    if (!gemv_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    using g = nat::native_gemv_t;
    const int bf16_n = 2048, int8_n = 1024;
    gemv_call_t bf16(kBf16, kBf16, kF32, bf16_n, bf16_n, 9300);
    gemv_call_t int8(kS8, kS8, kF32, int8_n, int8_n, 9310);
    const nt::buffer_t bf16_bias = nt::random_buffer(
            kBf16, static_cast<size_t>(bf16_n), 9301, -1.0f, 1.0f);
    const nt::buffer_t int8_bias = nt::random_buffer(
            kBf16, static_cast<size_t>(int8_n), 9311, -1.0f, 1.0f);
    bf16.set_bias(bf16_bias.get(), kBf16);
    int8.set_bias(int8_bias.get(), kBf16);
    bf16.c.params.weight_cache_type = 0;
    int8.c.params.weight_cache_type = 0;
    ASSERT_EQ(static_cast<int>(bf16.kernel()), static_cast<int>(g::bf16_bkc));
    ASSERT_EQ(static_cast<int>(int8.kernel()), static_cast<int>(g::int8_bkc));
    const auto run_both = [&] {
        return nt::run_matmul(bf16.c) == status_t::success
                && nt::run_matmul(int8.c) == status_t::success;
    };
    ASSERT_TRUE(run_both());
    const auto in_use = [] {
        const struct mallinfo2 m = mallinfo2();
        return m.uordblks + m.hblkhd;
    };
    // mallinfo2 counts glibc's heap only: under another allocator (a
    // sanitizer build, a preloaded one) the heap in use does not move.
    const size_t probe_bytes = static_cast<size_t>(bf16_n) * bf16_n * 2;
    const size_t at = in_use();
    void *probe = zendnnl_aligned_alloc(64, probe_bytes);
    const bool heap_counted = probe != nullptr && in_use() >= at + probe_bytes;
    zendnnl_aligned_free(probe);
    const size_t smallest = static_cast<size_t>(int8_n);
    const size_t before = in_use();
    constexpr int kThreads = 32;
    int failed = 0;
    for (int t = 0; t < kThreads; ++t)
        std::thread([&] {
            if (!run_both()) ++failed;
        }).join();
    const size_t after = in_use();
    EXPECT_EQ(failed, 0);
    if (!heap_counted) GTEST_SKIP() << "the heap in use is not glibc's";
    EXPECT_LT(after, before + kThreads * smallest / 2)
            << "the threads that exited left " << (after - before)
            << " bytes in use";
#else
    GTEST_SKIP() << "counts the heap in use with glibc's mallinfo2";
#endif
}

} // namespace
