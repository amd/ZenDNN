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

// The lifetime of what the native BRGEMM keeps about a constant weight
// buffer: a cached layout whose buffer now holds another weight is packed
// again unless the new bytes equal the old ones at both ends of what the
// pack read, an in-place rewrite keeps the layouts cached from the bytes it
// overwrites, the native weight cache clear forgets the cached layouts and
// the in-place images, lookups of other buffers go on while a buffer packs,
// a rewrite in place waits for the packs that read the bytes it writes, and
// no check of what is kept reads past the buffer of the call that makes it.
// A call's check for an image is handed to its layout lookup, and what
// a thread's calls were served stays right for its later calls across
// concurrent first calls, images, stale layouts, clears and exited threads.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>
#endif
#if defined(__GLIBC__)
#include <malloc.h>
#endif

#include <gtest/gtest.h>

#include "gtest_utils.hpp"
#include "lowoha_operators/matmul/lowoha_matmul.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_hw.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_pack.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_weight_cache.hpp"
#include "lowoha_operators/matmul/matmul_native/native_layout_policy.hpp"
#include "native_brgemm_test_utils.hpp"

namespace {

namespace mm = zendnnl::lowoha::matmul;
namespace nat = zendnnl::lowoha::matmul::native;
namespace nt = native_brgemm_test;
using zendnnl::common::matmul_algo_t;
using zendnnl::error_handling::status_t;
using zendnnl::memory::data_type_t;

constexpr data_type_t kBf16 = data_type_t::bf16;
constexpr data_type_t kS8 = data_type_t::s8;
constexpr data_type_t kS4 = data_type_t::s4;
constexpr size_t kEndBytes = 64;

// AddressSanitizer replaces the process heap and keeps freed blocks in its
// quarantine.
#if defined(__SANITIZE_ADDRESS__)
constexpr bool kSanitizedHeap = true;
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
constexpr bool kSanitizedHeap = true;
#else
constexpr bool kSanitizedHeap = false;
#endif
#else
constexpr bool kSanitizedHeap = false;
#endif

/// A bf16 or s8 weight in a buffer of rows() x ldb elements: [K, N]
/// row-major, or [N, K] when trans_b.  Copies share the buffer.
struct weight_t {
    data_type_t dt = kBf16;
    int K = 0, N = 0, ldb = 0;
    bool trans_b = false;
    nt::buffer_t buf;

    int rows() const { return trans_b ? N : K; }
    int cols() const { return trans_b ? K : N; }
    size_t elem() const { return nt::dt_bytes(dt); }
    /// Bytes from the first element read to past the last one.
    size_t span() const {
        return (static_cast<size_t>(rows() - 1) * ldb + cols()) * elem();
    }
    float at(int k, int n) const {
        const size_t i = trans_b ? static_cast<size_t>(n) * ldb + k
                                 : static_cast<size_t>(k) * ldb + n;
        return nt::load_f32(buf.get(), dt, i);
    }
    const void *ptr() const { return buf.get(); }
};

/// Writes the weight drawn from seed into w (bf16 in [-1, 1], s8 in
/// [-128, 127]); the elements past cols() in a row are left as they are.
void fill(const weight_t &w, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> real(-1.0f, 1.0f);
    std::uniform_int_distribution<int> whole(-128, 127);
    for (int r = 0; r < w.rows(); ++r)
        for (int c = 0; c < w.cols(); ++c)
            nt::store_f32(w.buf.get(), w.dt, static_cast<size_t>(r) * w.ldb + c,
                    w.dt == kBf16 ? real(gen) : static_cast<float>(whole(gen)));
}

weight_t make_weight(
        data_type_t dt, int K, int N, int ldb, bool trans_b, uint32_t seed) {
    weight_t w;
    w.dt = dt;
    w.K = K;
    w.N = N;
    w.ldb = ldb;
    w.trans_b = trans_b;
    w.buf = nt::make_buffer(static_cast<size_t>(w.rows()) * ldb * w.elem());
    fill(w, seed);
    return w;
}

/// A copy of w with a buffer of its own.
weight_t clone(const weight_t &w) {
    weight_t c = w;
    c.buf = nt::make_buffer(w.buf.bytes);
    std::memcpy(c.buf.get(), w.buf.get(), w.buf.bytes);
    return c;
}

/// A [K, N] weight of type dt, rows ldb apart, at offset bytes into w's
/// buffer, which it shares.
weight_t part_of(const weight_t &w, size_t offset, int K, int N, int ldb,
        data_type_t dt = kBf16) {
    weight_t p;
    p.dt = dt;
    p.K = K;
    p.N = N;
    p.ldb = ldb;
    p.buf.mem
            = std::shared_ptr<uint8_t>(w.buf.mem, w.buf.as<uint8_t>() + offset);
    p.buf.bytes = p.span();
    return p;
}

/// Gives w the bytes of from at both ends of its span.
void keep_ends(const weight_t &w, const void *from) {
    const size_t last = w.span() - kEndBytes;
    std::memcpy(w.buf.get(), from, kEndBytes);
    std::memcpy(w.buf.as<uint8_t>() + last,
            static_cast<const uint8_t *>(from) + last, kEndBytes);
}

/// Whether the buffer of w holds an in-place image, checked within it.
bool in_place(const weight_t &w) {
    return nat::native_weight_packed_in_place(w.ptr(), w.buf.bytes);
}

/// Calls of M rows on a weight that select the native BRGEMM: bf16 x bf16,
/// or u8 x s8, into f32.
struct dense_t {
    static constexpr int M = 4;
    data_type_t wei_dt;
    int K;
    nt::buffer_t src;
    std::vector<float> dst;

    dense_t(data_type_t wdt, int k) : wei_dt(wdt), K(k) {
        src = wdt == kBf16
                ? nt::random_buffer(kBf16, M * k, 11, -1, 1)
                : nt::random_buffer(data_type_t::u8, M * k, 12, 0, 255);
    }
    data_type_t src_dt() const {
        return wei_dt == kBf16 ? kBf16 : data_type_t::u8;
    }
    /// capacity: wei_buffer_capacity_bytes of the call; algo its
    /// lowoha_algo.
    status_t run(const weight_t &w, int32_t wc, size_t capacity = 0,
            matmul_algo_t algo = matmul_algo_t::native_brgemm,
            bool is_const = true) {
        dst.assign(static_cast<size_t>(M) * w.N, kUnwritten);
        nt::call_t c;
        c.M = M;
        c.N = w.N;
        c.K = w.K;
        c.transB = w.trans_b;
        c.ldb = w.ldb;
        c.src = src.get();
        c.wei = w.ptr();
        c.dst = dst.data();
        c.params.dtypes.src = src_dt();
        c.params.dtypes.wei = wei_dt;
        c.params.dtypes.dst = data_type_t::f32;
        c.params.num_threads = 8;
        c.params.weight_cache_type = wc;
        c.params.wei_buffer_capacity_bytes = capacity;
        c.params.lowoha_algo = algo;
        c.is_weights_const = is_const;
        return nt::run_matmul(c);
    }
    /// No element of dst was written since the last run.
    bool untouched() const {
        return std::all_of(dst.begin(), dst.end(),
                [](float v) { return v == kUnwritten; });
    }
    static constexpr float kUnwritten = -777.0f;
    /// dst against x * w computed here from the elements of w: the same
    /// integers for an s8 weight, within the bf16 band otherwise.
    ::testing::AssertionResult matches(const weight_t &w) const {
        size_t bad = 0;
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < w.N; ++n) {
                double ref = 0;
                for (int k = 0; k < K; ++k)
                    ref += static_cast<double>(nt::load_f32(src.get(), src_dt(),
                                   static_cast<size_t>(m) * K + k))
                            * w.at(k, n);
                const float got = dst[static_cast<size_t>(m) * w.N + n];
                const bool ok = wei_dt == kS8
                        ? got == static_cast<float>(ref)
                        : std::fabs(got - ref) <= 1e-2 + 1e-2 * std::fabs(ref);
                if (!ok) ++bad;
            }
        if (bad == 0) return ::testing::AssertionSuccess();
        return ::testing::AssertionFailure()
                << bad << " of " << M * w.N << " outputs differ from x * w";
    }
};

/// A bf16 [K, N] weight at w, rows N apart, in the native layout.
nat::native_raw_weight_t raw_weight(const void *w, int K, int N) {
    nat::native_raw_weight_t r;
    r.wei_dt = kBf16;
    r.weight = w;
    r.K = K;
    r.N = N;
    r.ldb = N;
    r.pack_nr = 32;
    return r;
}

class NativeWeightLifetime : public ::testing::Test {
protected:
    void SetUp() override {
        const nat::UarchParams &u = nat::detect_uarch();
        if (!u.avx512bf16 || !u.avx512vnni)
            GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
        ::reset_grp_matmul_caches();
        cache_ = std::make_unique<WeightCacheGuard>(2);
    }
    void TearDown() override {
        algo_.reset();
        cache_.reset();
        ::reset_grp_matmul_caches();
    }
    /// The process-wide ALGO 10, under which weight cache mode 2 rewrites a
    /// bf16 buffer in place, until the test ends (the first pin restores
    /// the algorithm the test began with).
    void pin_native() {
        if (!algo_)
            algo_ = std::make_unique<nt::algo_pin_t>(
                    matmul_algo_t::native_brgemm);
    }

private:
    std::unique_ptr<WeightCacheGuard> cache_;
    std::unique_ptr<nt::algo_pin_t> algo_;
};

struct case_t {
    data_type_t dt;
    int32_t wc;
    bool trans_b;
    int ldb_pad;
};

std::string describe(const case_t &c) {
    return std::string(c.dt == kBf16 ? "bf16" : "s8")
            + " wc=" + std::to_string(c.wc) + (c.trans_b ? " transB" : "")
            + " ldb+" + std::to_string(c.ldb_pad);
}

// A cached layout whose buffer now holds another weight, different at the
// ends of the span the pack read, is packed again from the buffer; so is an
// s8 layout weight cache mode 2 caches out of place.
TEST_F(NativeWeightLifetime, RewriteWithOtherEndsIsPackedAgain) {
    const int K = 64, N = 64;
    const case_t cases[]
            = {{kBf16, 1, false, 0}, {kBf16, 1, true, 16}, {kS8, 1, false, 32},
                    {kS8, 1, true, 0}, {kS8, 2, false, 0}, {kS8, 2, true, 64}};
    uint32_t seed = 100;
    for (const case_t &c : cases) {
        SCOPED_TRACE(describe(c));
        ::reset_grp_matmul_caches();
        if (c.wc == 2) pin_native();
        const weight_t w = make_weight(
                c.dt, K, N, (c.trans_b ? K : N) + c.ldb_pad, c.trans_b, ++seed);
        const weight_t w1 = clone(w);
        dense_t d(c.dt, K);
        ASSERT_EQ(d.run(w, c.wc), status_t::success);
        EXPECT_TRUE(d.matches(w1));
        fill(w, ++seed);
        ASSERT_EQ(d.run(w, c.wc), status_t::success);
        EXPECT_TRUE(d.matches(w));
        EXPECT_EQ(nt::native_layouts(w.ptr()), 1u)
                << "one layout, packed again from the new bytes";
        EXPECT_FALSE(in_place(w));
    }
}

// A weight of fewer than 8 bytes is checked at each of its bytes: a layout
// cached from it whose buffer now holds other values is packed again from
// them.
TEST_F(NativeWeightLifetime, RewriteOfAWeightUnderEightBytesIsPackedAgain) {
    const struct {
        data_type_t dt;
        int K, N;
    } cases[] = {{kBf16, 1, 2}, {kBf16, 1, 3}, {kS8, 1, 7}, {kS8, 2, 2}};
    uint32_t seed = 150;
    for (const auto &c : cases) {
        SCOPED_TRACE(std::string(c.dt == kBf16 ? "bf16" : "s8")
                + " K=" + std::to_string(c.K) + " N=" + std::to_string(c.N));
        ::reset_grp_matmul_caches();
        const weight_t w = make_weight(c.dt, c.K, c.N, c.N, false, ++seed);
        const weight_t w1 = clone(w);
        dense_t d(c.dt, c.K);
        ASSERT_EQ(d.run(w, 1), status_t::success);
        EXPECT_TRUE(d.matches(w1));
        fill(w, ++seed);
        ASSERT_NE(std::memcmp(w.ptr(), w1.ptr(), w.span()), 0);
        ASSERT_EQ(d.run(w, 1), status_t::success);
        EXPECT_TRUE(d.matches(w)) << "packed again from the new bytes";
        EXPECT_EQ(nt::native_layouts(w.ptr()), 1u);
    }
}

// A buffer rewritten with bytes equal to the old ones at both ends of the
// span is still served the old layout, until the native weight caches are
// cleared.
TEST_F(NativeWeightLifetime, RewriteWithTheSameEndsIsStaleUntilCleared) {
    const int K = 64, N = 64;
    const case_t cases[] = {{kBf16, 1, false, 0}, {kBf16, 1, true, 16},
            {kS8, 1, true, 16}, {kS8, 1, false, 0}};
    uint32_t seed = 200;
    for (const case_t &c : cases) {
        SCOPED_TRACE(describe(c));
        ::reset_grp_matmul_caches();
        const weight_t w = make_weight(
                c.dt, K, N, (c.trans_b ? K : N) + c.ldb_pad, c.trans_b, ++seed);
        const weight_t w1 = clone(w);
        dense_t d(c.dt, K);
        ASSERT_EQ(d.run(w, c.wc), status_t::success);
        EXPECT_TRUE(d.matches(w1));
        fill(w, ++seed);
        keep_ends(w, w1.ptr());
        ASSERT_EQ(d.run(w, c.wc), status_t::success);
        EXPECT_TRUE(d.matches(w1)) << "served the cached layout of w1";
        EXPECT_FALSE(d.matches(w));
        EXPECT_EQ(nt::native_layouts(w.ptr()), 1u);
        mm::clear_matmul_native_weight_caches();
        ASSERT_EQ(d.run(w, c.wc), status_t::success);
        EXPECT_TRUE(d.matches(w));
    }
}

// Once the native weight caches are cleared, an in-place image keeps its
// bytes and its record stays as a tombstone, which finds them; the raw
// weight the caller then writes into the buffer is read as raw (and
// rewritten in place again).
TEST_F(NativeWeightLifetime, ClearedImageRewrittenIsReadAsRaw) {
    pin_native();
    const int K = 64, N = 64;
    for (bool trans_b : {false, true}) {
        SCOPED_TRACE(trans_b ? "transB" : "no transB");
        ::reset_grp_matmul_caches();
        const weight_t w = make_weight(
                kBf16, K, N, trans_b ? K : N, trans_b, trans_b ? 301 : 302);
        const weight_t w1 = clone(w);
        dense_t d(kBf16, K);
        ASSERT_EQ(d.run(w, 2), status_t::success);
        EXPECT_TRUE(d.matches(w1));
        ASSERT_TRUE(in_place(w));
        const weight_t image = clone(w);
        mm::clear_matmul_native_weight_caches();
        EXPECT_TRUE(in_place(w)) << "the tombstone finds the image";
        EXPECT_EQ(nat::native_weight_tombstones(w.ptr()), 1u);
        EXPECT_EQ(std::memcmp(w.ptr(), image.ptr(), w.buf.bytes), 0)
                << "a clear does not write the buffer";
        fill(w, trans_b ? 303 : 304);
        const weight_t w2 = clone(w);
        ASSERT_EQ(d.run(w, 2), status_t::success);
        EXPECT_EQ(nt::native_layouts(w.ptr()), 1u) << "the new image";
        EXPECT_EQ(nat::native_weight_tombstones(w.ptr()), 0u);
        EXPECT_TRUE(d.matches(w2));
        EXPECT_TRUE(in_place(w));
    }
}

// A clear while a weight rewritten in place stays loaded: the next call on
// the untouched buffer, after either clear and at weight cache mode 2 or 1,
// fails with dst untouched and the buffer as it was, never success with the
// image read as a raw weight; so do calls on it that select the native
// BRGEMM by ZENDNNL_MATMUL_ALGO=10 alone or another algorithm, and a second
// clear keeps it so.  Once the raw weight is back, the call is served.
TEST_F(NativeWeightLifetime, ClearedLoadedImageFailsUntilItsRawWeightIsBack) {
    pin_native();
    const int K = 64, N = 64;
    for (bool all : {false, true})
        for (int32_t wc : {2, 1}) {
            SCOPED_TRACE(std::string(all ? "clear_matmul_weight_caches"
                                         : "clear_matmul_native_weight_caches")
                    + ", then wc=" + std::to_string(wc));
            ::reset_grp_matmul_caches();
            const weight_t w = make_weight(kBf16, K, N, N, false, 1001 + wc);
            const weight_t w1 = clone(w);
            dense_t d(kBf16, K);
            ASSERT_EQ(d.run(w, 2), status_t::success);
            ASSERT_TRUE(in_place(w));
            const weight_t image = clone(w);
            if (all)
                mm::clear_matmul_weight_caches();
            else
                mm::clear_matmul_native_weight_caches();
            EXPECT_EQ(d.run(w, wc), status_t::failure);
            EXPECT_TRUE(d.untouched());
            for (matmul_algo_t algo :
                    {matmul_algo_t::none, matmul_algo_t::aocl_dlp_blocked}) {
                EXPECT_EQ(d.run(w, wc, 0, algo), status_t::failure)
                        << "lowoha_algo " << static_cast<int>(algo);
                EXPECT_TRUE(d.untouched());
            }
            mm::clear_matmul_native_weight_caches();
            EXPECT_EQ(d.run(w, wc), status_t::failure)
                    << "after a second clear";
            EXPECT_EQ(std::memcmp(w.ptr(), image.ptr(), w.buf.bytes), 0)
                    << "the image was written again";
            std::memcpy(w.buf.get(), w1.ptr(), w.buf.bytes);
            ASSERT_EQ(d.run(w, wc), status_t::success);
            EXPECT_TRUE(d.matches(w1));
            EXPECT_EQ(nat::native_weight_tombstones(w.ptr()), 0u);
        }
}

// A clear that comes while a first call at weight cache mode 2 rewrites its
// buffer in place (outside the quiescence the clears ask for) leaves a
// tombstone of the image the call writes: the call is served, the next call
// on the buffer fails with dst untouched, never success with the image read
// as a raw weight, and once the raw weight is back it is served.
TEST_F(NativeWeightLifetime, AClearDuringAFirstRewriteLeavesATombstone) {
    pin_native();
    const int K = 4096, N = 4096;
    int raced = 0;
    for (uint32_t round = 0; round < 3; ++round) {
        SCOPED_TRACE("round " + std::to_string(round));
        ::reset_grp_matmul_caches();
        const weight_t w = make_weight(kBf16, K, N, N, false, 1191 + round);
        const weight_t raw = clone(w);
        dense_t d(kBf16, K);
        std::atomic<bool> returned {false};
        bool cleared_first = false;
        status_t first = status_t::failure;
        std::thread call([&] {
            first = d.run(w, 2);
            returned.store(true);
        });
        std::thread clear([&] {
            while (!in_place(w) && !returned.load())
                std::this_thread::yield();
            cleared_first = !returned.load();
            mm::clear_matmul_native_weight_caches();
        });
        call.join();
        clear.join();
        if (cleared_first) ++raced;
        ASSERT_EQ(first, status_t::success);
        EXPECT_TRUE(d.matches(raw));
        EXPECT_EQ(d.run(w, 2), status_t::failure)
                << "the image read as a raw weight";
        EXPECT_TRUE(d.untouched());
        std::memcpy(w.buf.get(), raw.ptr(), w.buf.bytes);
        ASSERT_EQ(d.run(w, 1), status_t::success);
        EXPECT_TRUE(d.matches(raw));
    }
    EXPECT_GT(raced, 0) << "no clear came before the rewriting call returned";
}

// Of two models rewritten in place, a clear for the one that unloads, whose
// buffer then holds another weight, serves that weight, while the model
// that stays loaded fails its next call (never success with another
// result) until its raw weight is reloaded, and is served right then.
TEST_F(NativeWeightLifetime, ClearForOneModelFailsTheOtherUntilItsReload) {
    pin_native();
    const int K = 512, N = 256;
    const weight_t a = make_weight(kBf16, K, N, N, false, 1011);
    const weight_t b = make_weight(kBf16, K, N, N, false, 1012);
    const weight_t a1 = clone(a), b1 = clone(b);
    dense_t d(kBf16, K);
    ASSERT_EQ(d.run(a, 2), status_t::success);
    ASSERT_TRUE(d.matches(a1));
    ASSERT_EQ(d.run(b, 2), status_t::success);
    ASSERT_TRUE(d.matches(b1));
    ASSERT_TRUE(in_place(a));
    ASSERT_TRUE(in_place(b));
    mm::clear_matmul_native_weight_caches();
    fill(a, 1013);
    const weight_t a2 = clone(a);
    ASSERT_EQ(d.run(a, 2), status_t::success);
    EXPECT_TRUE(d.matches(a2)) << "the weight loaded where model a was";
    EXPECT_EQ(d.run(b, 2), status_t::failure);
    EXPECT_TRUE(d.untouched());
    std::memcpy(b.buf.get(), b1.ptr(), b.buf.bytes);
    ASSERT_EQ(d.run(b, 2), status_t::success);
    EXPECT_TRUE(d.matches(b1));
}

// After a clear, a request on bytes of an image that keeps them is checked at
// each of those bytes, however few: a bf16 K=1, N=2 request (4 bytes, ldb
// 32) at the start of a K=2, N=32 image; 32 bytes inside an 8 KiB image,
// which its record keeps whole; and a request from below
// that reads 32 bytes of an image, fewer than its head end.  Each fails at
// weight cache modes 2, 1 and 0 with dst untouched, the tombstone kept; once
// the raw weight is back, each is served from it and forgets the tombstone
// over its bytes, and the call on the whole weight forgets the rest.
TEST_F(NativeWeightLifetime, ClearedImageFailsAShortRequestOnItsBytes) {
    pin_native();
    for (int32_t wc : {2, 1, 0}) {
        SCOPED_TRACE("then wc=" + std::to_string(wc));
        for (int where = 0; where < 3; ++where) {
            SCOPED_TRACE(where == 0      ? "at the start of a small image"
                            : where == 1 ? "inside an image"
                                         : "from below an image");
            ::reset_grp_matmul_caches();
            // An image of 128 bytes at the start of its buffer, or of 8 KiB
            // past 1 KiB of raw weight.
            const weight_t whole = where == 0
                    ? make_weight(kBf16, 2, 32, 32, false, 1041 + wc)
                    : make_weight(
                              kBf16, 72, 64, 64, false, 1051 + 10 * where + wc);
            const weight_t w
                    = where == 0 ? whole : part_of(whole, 1024, 64, 64, 64);
            const weight_t raw = clone(w);
            dense_t d(kBf16, w.K);
            ASSERT_EQ(d.run(w, 2), status_t::success);
            ASSERT_TRUE(d.matches(raw));
            ASSERT_TRUE(in_place(w));
            const weight_t image = clone(w);
            mm::clear_matmul_native_weight_caches();
            const weight_t part = where == 0 ? part_of(whole, 0, 1, 2, 32)
                    : where == 1             ? part_of(whole, 2064, 1, 16, 16)
                                             : part_of(whole, 0, 33, 16, 16);
            dense_t dp(kBf16, part.K);
            EXPECT_EQ(dp.run(part, wc), status_t::failure);
            EXPECT_TRUE(dp.untouched());
            EXPECT_EQ(nat::native_weight_tombstones(w.ptr()), 1u);
            EXPECT_EQ(std::memcmp(w.ptr(), image.ptr(), w.buf.bytes), 0);
            std::memcpy(w.buf.get(), raw.ptr(), w.buf.bytes);
            ASSERT_EQ(dp.run(part, wc), status_t::success);
            EXPECT_TRUE(dp.matches(part));
            ASSERT_EQ(d.run(w, wc), status_t::success);
            EXPECT_TRUE(d.matches(raw));
            EXPECT_EQ(nat::native_weight_tombstones(w.ptr()), 0u);
        }
    }
}

// A refused call on bytes of an image prints one error line, which names
// what helps and the image's start and bytes: a weight whose bytes still
// hold a live image, inside it or reading into it from below, is told to
// write the raw weight back into the image's buffer, clear the weight
// caches and reload every weight a mode 2 call rewrote, as writing it back
// alone can leave the bytes a call checks as they were; one that holds none
// of the image's checked bytes (32 bytes between two sample lines of a
// 1 MiB image) is told that it cannot be told from the image and that, if
// the image's buffer was freed, two clears help; one inside a tombstone is
// told to reload the raw weight of the image's whole buffer, and not to
// clear again.
TEST_F(NativeWeightLifetime, RefusalInsideAnImageNamesWhatHelps) {
    nt::expect_in_child({{"ZENDNNL_API_LOG_LEVEL", nullptr},
                                {"ZENDNNL_COMMON_LOG_LEVEL", nullptr}},
            [this] {
        pin_native();
        // Images of 16 KiB, one past 1 KiB of raw weight, and one of 1 MiB,
        // whose sample lines lie about 1 KiB apart.
        const weight_t whole = make_weight(kBf16, 136, 64, 64, false, 1071);
        const weight_t w = part_of(whole, 1024, 128, 64, 64);
        const weight_t large = make_weight(kBf16, 8192, 64, 64, false, 1072);
        const weight_t other = make_weight(kBf16, 128, 64, 64, false, 1073);
        dense_t d(kBf16, 128), dl(kBf16, 8192);
        if (d.run(w, 2) != status_t::success || !in_place(w)) return 1;
        if (dl.run(large, 2) != status_t::success || !in_place(large)) return 1;
        if (d.run(other, 2) != status_t::success || !in_place(other)) return 1;
        int bad = 0;
        const auto refusal = [&](const weight_t &v) {
            dense_t dv(kBf16, v.K);
            ::testing::internal::CaptureStdout();
            if (dv.run(v, 1) != status_t::failure || !dv.untouched()) ++bad;
            const std::string log = ::testing::internal::GetCapturedStdout();
            size_t lines = 0;
            for (size_t at = log.find("][error"); at != std::string::npos;
                    at = log.find("][error", at + 1))
                ++lines;
            if (lines != 1
                    || log.find("the image recorded at ") == std::string::npos)
                ++bad;
            return log;
        };
        const std::string held[] = {refusal(part_of(whole, 5120, 8, 64, 64)),
                refusal(part_of(whole, 0, 16, 64, 64))};
        const std::string unchecked = refusal(part_of(large, 2560, 1, 16, 16));
        mm::clear_matmul_native_weight_caches();
        const std::string tomb = refusal(part_of(other, 4096, 8, 64, 64));
        for (const std::string &log : held)
            if (log.find("write the raw weight back into the image's buffer, "
                         "clear the weight caches and reload every weight a "
                         "mode 2 call rewrote")
                            == std::string::npos
                    || log.find(", 16384 bytes") == std::string::npos)
                ++bad;
        if (unchecked.find("cannot be told from the image") == std::string::npos
                || unchecked.find("clear the weight caches twice while no call "
                                  "runs, then reload every other weight a "
                                  "mode 2 call rewrote")
                        == std::string::npos)
            ++bad;
        if (tomb.find("reload the raw weight of the image's whole buffer")
                        == std::string::npos
                || tomb.find("clear the weight caches") != std::string::npos
                || tomb.find(", 16384 bytes") == std::string::npos)
            ++bad;
        if (bad != 0)
            for (const std::string *log :
                    {&held[0], &held[1], &unchecked, &tomb})
                std::fprintf(stderr, "%s\n", log->c_str());
        return bad;
    }, "the error lines of a weight inside an image");
}

// A constant weight of a type the native layout does not take (u4) whose
// buffer starts inside a live image is refused as a weight inside the
// image, dst untouched, under ZENDNNL_MATMUL_ALGO=10 with lowoha_algo
// native_brgemm and with none, at weight cache modes 1 and 2: no layout is
// asked of the cache for it.
TEST_F(NativeWeightLifetime, AWeightOfAnotherTypeInsideAnImageIsRefused) {
    pin_native();
    const weight_t whole = make_weight(kBf16, 128, 64, 64, false, 1211);
    dense_t d(kBf16, 128);
    ASSERT_EQ(d.run(whole, 2), status_t::success);
    ASSERT_TRUE(in_place(whole));
    const int M = 4, N = 64, K = 64;
    const nt::buffer_t src = nt::random_buffer(kBf16, M * K, 1212, -1, 1);
    std::vector<float> scales(N, 0.25f);
    std::vector<int8_t> zero_points(N, 8);
    for (const matmul_algo_t algo :
            {matmul_algo_t::native_brgemm, matmul_algo_t::none})
        for (const int32_t wc : {1, 2}) {
            SCOPED_TRACE("lowoha_algo " + std::to_string(static_cast<int>(algo))
                    + ", wc=" + std::to_string(wc));
            std::vector<float> dst(static_cast<size_t>(M) * N, -777.0f);
            nt::call_t c;
            c.M = M;
            c.N = N;
            c.K = K;
            c.ldb = N;
            c.src = src.get();
            c.wei = whole.buf.as<uint8_t>() + 4096;
            c.dst = dst.data();
            c.params.dtypes.src = kBf16;
            c.params.dtypes.wei = data_type_t::u4;
            c.params.dtypes.dst = data_type_t::f32;
            c.params.quant_params.wei_scale.buff = scales.data();
            c.params.quant_params.wei_scale.dt = data_type_t::f32;
            c.params.quant_params.wei_scale.dims = {1, N};
            c.params.quant_params.wei_zp.buff = zero_points.data();
            c.params.quant_params.wei_zp.dt = kS8;
            c.params.quant_params.wei_zp.dims = {1, N};
            c.params.weight_cache_type = wc;
            c.params.lowoha_algo = algo;
            c.is_weights_const = true;
            EXPECT_EQ(nt::run_matmul(c), status_t::failure);
            EXPECT_TRUE(std::all_of(dst.begin(), dst.end(),
                    [](float v) { return v == -777.0f; }));
        }
}

// A call whose check holds no byte of a tombstone (a u4 weight with no
// capacity stated, whose buffer the check takes as 0 bytes) at the start of
// a cleared image forgets none of it: under ZENDNNL_MATMUL_ALGO=10 with
// lowoha_algo native_brgemm and with none it is not served, dst untouched,
// the tombstone stays, and the next bf16 call on the untouched image fails
// with dst untouched, never success with the image read as a raw weight.
TEST_F(NativeWeightLifetime, ACallThatHoldsNoByteOfATombstoneForgetsNothing) {
    pin_native();
    const int M = 4, N = 64, K = 64;
    const nt::buffer_t src = nt::random_buffer(kBf16, M * K, 1232, -1, 1);
    std::vector<float> scales(N, 0.25f);
    std::vector<int8_t> zero_points(N, 8);
    for (const matmul_algo_t algo :
            {matmul_algo_t::native_brgemm, matmul_algo_t::none}) {
        SCOPED_TRACE("lowoha_algo " + std::to_string(static_cast<int>(algo)));
        ::reset_grp_matmul_caches();
        const weight_t whole = make_weight(kBf16, 128, 64, 64, false, 1231);
        dense_t d(kBf16, 128);
        ASSERT_EQ(d.run(whole, 2), status_t::success);
        ASSERT_TRUE(in_place(whole));
        mm::clear_matmul_native_weight_caches();
        std::vector<float> dst(static_cast<size_t>(M) * N, -777.0f);
        nt::call_t c;
        c.M = M;
        c.N = N;
        c.K = K;
        c.ldb = N;
        c.src = src.get();
        c.wei = whole.ptr();
        c.dst = dst.data();
        c.params.dtypes.src = kBf16;
        c.params.dtypes.wei = data_type_t::u4;
        c.params.dtypes.dst = data_type_t::f32;
        c.params.quant_params.wei_scale.buff = scales.data();
        c.params.quant_params.wei_scale.dt = data_type_t::f32;
        c.params.quant_params.wei_scale.dims = {1, N};
        c.params.quant_params.wei_zp.buff = zero_points.data();
        c.params.quant_params.wei_zp.dt = kS8;
        c.params.quant_params.wei_zp.dims = {1, N};
        c.params.weight_cache_type = 1;
        c.params.lowoha_algo = algo;
        c.is_weights_const = true;
        EXPECT_NE(nt::run_matmul(c), status_t::success);
        EXPECT_TRUE(std::all_of(
                dst.begin(), dst.end(), [](float v) { return v == -777.0f; }));
        EXPECT_EQ(nat::native_weight_tombstones(whole.ptr()), 1u);
        EXPECT_EQ(d.run(whole, 1), status_t::failure)
                << "the image read as a raw weight";
        EXPECT_TRUE(d.untouched());
    }
}

// A new weight smaller than a sample line that lands in the memory of a
// freed image after a clear is read as the raw weight it is, at its start,
// inside it and from below into its head end: its bytes differ from the
// image's, which the record kept, and the tombstone is forgotten over them
// only: the part of it before a weight inside it stays.
TEST_F(NativeWeightLifetime, ANewWeightUnderOneLineInAFreedImageIsServed) {
    pin_native();
    const int offsets[] = {1024, 1024 + 2064, 1024 - 32};
    for (int at : offsets) {
        SCOPED_TRACE("at byte " + std::to_string(at));
        ::reset_grp_matmul_caches();
        const weight_t whole = make_weight(kBf16, 72, 64, 64, false, 1081);
        const weight_t w = part_of(whole, 1024, 64, 64, 64);
        dense_t d(kBf16, w.K);
        ASSERT_EQ(d.run(w, 2), status_t::success);
        ASSERT_TRUE(in_place(w));
        mm::clear_matmul_native_weight_caches();
        const weight_t fresh = part_of(whole, at, 1, 37, 37, kS8);
        fill(fresh, 1082);
        dense_t ds(kS8, 1);
        ASSERT_EQ(ds.run(fresh, 1), status_t::success);
        EXPECT_TRUE(ds.matches(fresh));
        EXPECT_EQ(nat::native_weight_tombstones(w.ptr()),
                at == offsets[1] ? 1u : 0u);
    }
}

// A weight allocated in the memory of an image of over 64 KiB where the
// image holds zeros (the image of zero rows) and zeros of its own but in
// its first row is told from the image: no sample line of the record lies
// where the image repeats one word.  It holds none of the record's checked
// bytes and is refused inside the live image and after a clear, as it
// cannot be told from the image; after a second clear it is read as the
// raw weight it is, at weight cache modes 1 and 0, also once its raw bytes
// are written again, and the rest of the tombstone stays.
TEST_F(NativeWeightLifetime,
        AWeightOverZerosOfAFreedImageIsServedAfterTwoClears) {
    pin_native();
    const int K = 8192, N = 64;
    const weight_t whole = make_weight(kBf16, K, N, N, false, 1181);
    for (int k = 2048; k < 6144; ++k)
        for (int n = 0; n < N; ++n)
            nt::store_f32(whole.buf.get(), kBf16,
                    static_cast<size_t>(k) * N + n, 0.0f);
    dense_t d(kBf16, K);
    ASSERT_EQ(d.run(whole, 2), status_t::success);
    ASSERT_TRUE(in_place(whole));
    // The first 32 columns of rows 2048 to 6143 lie from 128 KiB on in the
    // image.
    const weight_t fresh = part_of(whole, 200 * 1024, 16, N, N);
    std::memset(fresh.buf.get(), 0, fresh.buf.bytes);
    for (int n = 0; n < N; ++n)
        nt::store_f32(fresh.buf.get(), kBf16, static_cast<size_t>(n),
                0.25f * static_cast<float>(n % 7) - 0.75f);
    const weight_t raw = clone(fresh);
    dense_t df(kBf16, fresh.K);
    EXPECT_EQ(df.run(fresh, 1), status_t::failure);
    EXPECT_TRUE(df.untouched());
    mm::clear_matmul_native_weight_caches();
    EXPECT_EQ(df.run(fresh, 1), status_t::failure);
    EXPECT_TRUE(df.untouched());
    mm::clear_matmul_native_weight_caches();
    for (const int32_t wc : {1, 0}) {
        ASSERT_EQ(df.run(fresh, wc), status_t::success) << "wc=" << wc;
        EXPECT_TRUE(df.matches(raw)) << "wc=" << wc;
    }
    std::memcpy(fresh.buf.get(), raw.ptr(), raw.buf.bytes);
    mm::clear_matmul_native_weight_caches();
    ASSERT_EQ(df.run(fresh, 0), status_t::success);
    EXPECT_TRUE(df.matches(raw));
    EXPECT_EQ(nat::native_weight_tombstones(whole.ptr()), 1u);
}

// An in-place rewrite that finds no memory for what its record keeps to
// check the image (its bytes, the digests of its sample lines) writes
// nothing: the call is served from a layout cached out of place, the buffer
// keeps the raw weight and holds no image, and after a clear a 4-byte
// request at its start reads the raw weight.  For a 128-byte image and a
// 128 KiB one.
TEST_F(NativeWeightLifetime, ARewriteWithNoMemoryForItsChecksWritesNothing) {
    pin_native();
    for (const int K : {2, 1024}) {
        SCOPED_TRACE("K=" + std::to_string(K));
        ::reset_grp_matmul_caches();
        const int N = K == 2 ? 32 : 64;
        const weight_t w = make_weight(kBf16, K, N, N, false, 1091 + K);
        const weight_t raw = clone(w);
        dense_t d(kBf16, K);
        nat::native_fail_next_image_checks(1);
        const status_t st = d.run(w, 2);
        nat::native_fail_next_image_checks(0);
        ASSERT_EQ(st, status_t::success);
        EXPECT_TRUE(d.matches(raw));
        EXPECT_FALSE(in_place(w));
        EXPECT_EQ(std::memcmp(w.ptr(), raw.ptr(), w.buf.bytes), 0)
                << "the buffer was written";
        EXPECT_EQ(nt::native_layouts(w.ptr()), 1u) << "cached out of place";
        mm::clear_matmul_native_weight_caches();
        const weight_t head = part_of(w, 0, 1, 2, N);
        dense_t dh(kBf16, 1);
        ASSERT_EQ(dh.run(head, 2), status_t::success);
        EXPECT_TRUE(dh.matches(head));
    }
}

// A bf16 rewrite that finds no memory for its record's checks, whose call
// is then served out of place, logs a warning line that says so, and no
// error line.
TEST_F(NativeWeightLifetime, ARewriteWithNoMemoryForItsChecksLogsAWarning) {
    nt::expect_in_child({{"ZENDNNL_API_LOG_LEVEL", nullptr},
                                {"ZENDNNL_COMMON_LOG_LEVEL", nullptr}},
            [this] {
        pin_native();
        const weight_t w = make_weight(kBf16, 64, 64, 64, false, 1251);
        const weight_t raw = clone(w);
        dense_t d(kBf16, 64);
        nat::native_fail_next_image_checks(1);
        ::testing::internal::CaptureStdout();
        const status_t st = d.run(w, 2);
        const std::string log = ::testing::internal::GetCapturedStdout();
        nat::native_fail_next_image_checks(0);
        if (st == status_t::success && d.matches(raw) && !in_place(w)
                && log.find("][error") == std::string::npos
                && log.find("][warning") != std::string::npos
                && log.find("no memory for the checks of an in-place image")
                        != std::string::npos)
            return 0;
        std::fprintf(stderr, "%s\n", log.c_str());
        return 1;
    }, "the line of a rewrite with no memory for its checks");
}

// A bf16 weight whose layout equals its own bytes wherever an image record
// would check them is not rewritten in place: an all-zero weight, a weight
// of one value, and an all-zero weight of odd K whose buffer holds other
// bytes past it within the stated capacity (the layout's zero row would
// land there, past what a reload writes).  Its layout is cached out of place
// and the buffer keeps every byte; after a clear and a reload of the raw
// weight, the next call is served right.
TEST_F(NativeWeightLifetime, AWeightWhoseLayoutIsItsBytesIsNotRewritten) {
    pin_native();
    for (int which = 0; which < 3; ++which) {
        SCOPED_TRACE(which == 0      ? "all zero"
                        : which == 1 ? "one value"
                                     : "all zero, odd K, bytes past it");
        ::reset_grp_matmul_caches();
        const weight_t whole = make_weight(kBf16, 64, 64, 64, false, 1101);
        const int K = which == 2 ? 63 : 64;
        const weight_t w = part_of(whole, 0, K, 64, 64);
        for (int k = 0; k < K; ++k)
            for (int n = 0; n < 64; ++n)
                nt::store_f32(w.buf.get(), kBf16,
                        static_cast<size_t>(k) * 64 + n,
                        which == 1 ? 1.0f : 0.0f);
        const weight_t before = clone(whole);
        dense_t d(kBf16, K);
        ASSERT_EQ(d.run(w, 2, whole.buf.bytes), status_t::success);
        EXPECT_TRUE(d.matches(w));
        EXPECT_FALSE(in_place(w));
        EXPECT_EQ(std::memcmp(whole.ptr(), before.ptr(), whole.buf.bytes), 0)
                << "the buffer was written";
        EXPECT_EQ(nt::native_layouts(w.ptr()), 1u) << "cached out of place";
        mm::clear_matmul_native_weight_caches();
        std::memcpy(w.buf.get(), before.ptr(), w.span());
        ASSERT_EQ(d.run(w, 2, whole.buf.bytes), status_t::success);
        EXPECT_TRUE(d.matches(w));
    }
}

// The first in-place rewrite of a process logs one warning line, visible at
// the default log level, that from that call the buffer holds the native
// layout; later rewrites and layouts cached out of place log none.
TEST_F(NativeWeightLifetime, FirstInPlaceRewriteLogsOneWarning) {
    nt::expect_in_child({{"ZENDNNL_API_LOG_LEVEL", nullptr}}, [this] {
        pin_native();
        const int K = 64, N = 64;
        const weight_t a = make_weight(kBf16, K, N, N, false, 1101);
        const weight_t b = make_weight(kBf16, K, N, N, false, 1102);
        const weight_t c = make_weight(kS8, K, N, N, false, 1103);
        dense_t d(kBf16, K), e(kS8, K);
        int bad = 0;
        ::testing::internal::CaptureStdout();
        if (d.run(a, 2) != status_t::success || !in_place(a)) ++bad;
        if (d.run(b, 2) != status_t::success || !in_place(b)) ++bad;
        if (e.run(c, 2) != status_t::success || in_place(c)) ++bad;
        if (d.run(a, 2) != status_t::success) ++bad;
        const std::string log = ::testing::internal::GetCapturedStdout();
        size_t lines = 0;
        for (size_t at = log.find("rewrote the weight buffer at ");
                at != std::string::npos;
                at = log.find("rewrote the weight buffer at ", at + 1))
            ++lines;
        if (lines != 1
                || log.find("weight_cache_type 1") == std::string::npos) {
            std::fprintf(stderr, "%zu lines (1 expected):\n%s\n", lines,
                    log.c_str());
            ++bad;
        }
        return bad;
    }, "ZENDNNL_API_LOG_LEVEL unset");
}

// A full native layout cache logs one warning line per process, visible at
// the default log level, when a call first packs a layout for itself for
// want of room, and none while the cache has room: at
// ZENDNNL_LRU_CACHE_CAPACITY 1 the calls on a second weight log it once,
// at capacity 0 the first call that would cache a layout does.
TEST_F(NativeWeightLifetime, AFullLayoutCacheLogsOneWarning) {
    for (const uint32_t cap : {1u, 0u}) {
        SCOPED_TRACE("capacity " + std::to_string(cap));
        nt::expect_in_child({{"ZENDNNL_API_LOG_LEVEL", nullptr}}, [cap] {
            const nt::lru_capacity_pin_t capacity(cap);
            const weight_t a = make_weight(kS8, 64, 64, 64, false, 1231);
            const weight_t b = make_weight(kS8, 64, 64, 64, false, 1232);
            dense_t d(kS8, 64);
            int bad = 0;
            const auto lines = [](const std::string &log) {
                size_t n = 0;
                for (size_t at = log.find("native layout cache is at its");
                        at != std::string::npos;
                        at = log.find("native layout cache is at its", at + 1))
                    ++n;
                return n;
            };
            ::testing::internal::CaptureStdout();
            if (d.run(a, 1) != status_t::success || !d.matches(a)) ++bad;
            const std::string first = ::testing::internal::GetCapturedStdout();
            ::testing::internal::CaptureStdout();
            for (int i = 0; i < 3; ++i)
                if (d.run(b, 1) != status_t::success || !d.matches(b)) ++bad;
            const std::string later = ::testing::internal::GetCapturedStdout();
            const std::string &full = cap == 0 ? first : later;
            if (lines(first) != (cap == 0 ? 1u : 0u)
                    || lines(later) != (cap == 0 ? 0u : 1u)
                    || full.find("capacity of " + std::to_string(cap)
                               + " layouts")
                            == std::string::npos
                    || full.find("weights that do not fit are packed for each "
                                 "of their calls until a clear makes room")
                            == std::string::npos) {
                std::fprintf(stderr, "first calls:\n%s\nlater calls:\n%s\n",
                        first.c_str(), later.c_str());
                ++bad;
            }
            return bad;
        }, "ZENDNNL_API_LOG_LEVEL unset");
    }
}

// A weight the framework also reads, such as an embedding table tied to the
// lm_head, keeps its bytes under ZENDNNL_MATMUL_ALGO=10 when its calls pass
// weight_cache_type 1 or a weight that is not constant, where weight cache
// mode 2 rewrites the buffer.
TEST_F(NativeWeightLifetime, SharedWeightKeepsItsBytesAtModeOneOrNotConstant) {
    pin_native();
    const int V = 1024, H = 256;
    for (int way = 0; way < 3; ++way) {
        SCOPED_TRACE(way == 0      ? "weight_cache_type 1"
                        : way == 1 ? "not constant"
                                   : "weight_cache_type 2");
        const weight_t table = make_weight(kBf16, H, V, H, true, 1111 + way);
        const weight_t raw = clone(table);
        dense_t d(kBf16, H);
        ASSERT_EQ(d.run(table, way == 0 ? 1 : 2, 0,
                          matmul_algo_t::native_brgemm, way != 1),
                status_t::success);
        EXPECT_TRUE(d.matches(raw));
        EXPECT_EQ(std::memcmp(table.ptr(), raw.ptr(), table.buf.bytes) == 0,
                way != 2);
    }
}

// A raw weight written over an image with the image's bytes at both ends is
// still read as the image, until the native weight caches are cleared; then
// it fails with dst untouched, as its ends hold the image among other
// bytes, until they hold other bytes too.
TEST_F(NativeWeightLifetime, RawWeightWithTheImageEndsIsStaleUntilCleared) {
    pin_native();
    const int K = 64, N = 64;
    const weight_t w = make_weight(kBf16, K, N, N, false, 311);
    dense_t d(kBf16, K);
    ASSERT_EQ(d.run(w, 2), status_t::success);
    ASSERT_TRUE(in_place(w));
    const weight_t image = clone(w);
    fill(w, 312);
    keep_ends(w, image.ptr());
    const weight_t w2 = clone(w);
    EXPECT_TRUE(in_place(w));
    ASSERT_EQ(d.run(w, 2), status_t::success);
    EXPECT_FALSE(d.matches(w2)) << "read as the image";
    EXPECT_EQ(nt::native_layouts(w.ptr()), 1u);
    mm::clear_matmul_native_weight_caches();
    EXPECT_EQ(d.run(w, 2), status_t::failure);
    EXPECT_TRUE(d.untouched());
    fill(w, 313);
    const weight_t w3 = clone(w);
    ASSERT_EQ(d.run(w, 2), status_t::success);
    EXPECT_TRUE(d.matches(w3));
}

// A layout packed from an in-place image, cached for a narrower view of the
// buffer, follows the buffer when a raw weight is written over the image.
// The rebuild reads the whole image, so the view states the buffer as its
// capacity; a view that does not cover the image fails instead.
TEST_F(NativeWeightLifetime, LayoutPackedFromAnImageFollowsTheBuffer) {
    pin_native();
    const int K = 64, N = 64;
    const weight_t w = make_weight(kBf16, K, N, N, false, 321);
    weight_t view = w;
    view.N = N / 2;
    const weight_t w1 = clone(w);
    weight_t view1 = w1;
    view1.N = N / 2;
    dense_t d(kBf16, K);
    ASSERT_EQ(d.run(w, 2), status_t::success);
    ASSERT_TRUE(in_place(w));
    EXPECT_EQ(d.run(view, 2), status_t::failure)
            << "a rebuild past the bytes the view reads";
    ASSERT_EQ(d.run(view, 2, w.buf.bytes), status_t::success);
    EXPECT_TRUE(d.matches(view1));
    fill(w, 322);
    const weight_t w2 = clone(w);
    weight_t view2 = w2;
    view2.N = N / 2;
    ASSERT_EQ(d.run(view, 2, w.buf.bytes), status_t::success);
    EXPECT_TRUE(d.matches(view2));
    EXPECT_FALSE(in_place(w));
}

// A layout cached from a buffer before an in-place rewrite of it stays
// cached, and is not packed again: the image holds the same weight.
TEST_F(NativeWeightLifetime, LayoutCachedBeforeAnInPlaceRewriteStays) {
    pin_native();
    const int K = 64, N = 64;
    const weight_t w = make_weight(kBf16, K, N, N, false, 331);
    weight_t view = w;
    view.N = N / 2;
    const weight_t w1 = clone(w);
    weight_t view1 = w1;
    view1.N = N / 2;
    dense_t d(kBf16, K);
    ASSERT_EQ(d.run(view, 2), status_t::success);
    EXPECT_TRUE(d.matches(view1));
    EXPECT_FALSE(in_place(w));
    const auto view_layout = [&] {
        nat::native_raw_weight_t r = raw_weight(w.ptr(), K, N / 2);
        r.ldb = N;
        nat::native_layout_t out;
        EXPECT_EQ(nat::get_native_layout(r, 1, 0, 1, out), status_t::success);
        EXPECT_EQ(out.owned, nullptr);
        return out.data;
    };
    const void *cached = view_layout();
    ASSERT_EQ(d.run(w, 2), status_t::success);
    EXPECT_TRUE(d.matches(w1));
    ASSERT_TRUE(in_place(w));
    ASSERT_EQ(d.run(view, 2), status_t::success);
    EXPECT_TRUE(d.matches(view1));
    EXPECT_EQ(view_layout(), cached) << "the cached layout was packed again";
    EXPECT_EQ(nt::native_layouts(w.ptr()), 2u)
            << "the cached layout and the image";
}

// The check of a call for an in-place image tells the image's layout when
// the image starts at the weight, and the layout lookup serves that layout
// from it without a check of its own when the call's check read the bytes
// the lookup checks; a pointer inside the image is in place but not served,
// and another layout of the buffer is packed from the image.
TEST_F(NativeWeightLifetime, InPlaceCheckIsHandedToTheLookup) {
    pin_native();
    const int K = 64, N = 64;
    const weight_t w = make_weight(kBf16, K, N, N, false, 341);
    const weight_t raw = make_weight(kBf16, K, N, N, false, 342);
    dense_t d(kBf16, K);
    ASSERT_EQ(d.run(w, 2), status_t::success);
    ASSERT_TRUE(in_place(w));

    nat::native_image_check_t at_start;
    ASSERT_TRUE(nat::native_weight_find_image(w.ptr(), w.buf.bytes, at_start));
    ASSERT_TRUE(at_start.ready);
    EXPECT_EQ(at_start.layout.weight, w.ptr());
    EXPECT_EQ(at_start.layout.K, K);
    EXPECT_EQ(at_start.layout.N, N);
    EXPECT_EQ(at_start.layout.pack_nr, nat::kNativeLayoutPackNr);
    EXPECT_EQ(at_start.checked_bytes, w.buf.bytes);

    // A check the lookup cannot find again, on a buffer without an image:
    // served as handed when it covers the bytes the lookup checks, looked up
    // (and packed) when it covers fewer.
    nat::native_image_check_t handed;
    handed.ready = true;
    handed.layout = raw_weight(raw.ptr(), K, N);
    handed.checked_bytes = raw.buf.bytes;
    for (int32_t wc : {0, 1, 2}) {
        nat::native_layout_t out;
        ASSERT_EQ(nat::get_native_layout(
                          raw_weight(raw.ptr(), K, N), wc, 0, 1, out, &handed),
                status_t::success)
                << "wc=" << wc;
        EXPECT_EQ(out.data, raw.ptr()) << "wc=" << wc;
        EXPECT_EQ(out.owned, nullptr) << "wc=" << wc;
    }
    handed.checked_bytes = 0;
    for (int32_t wc : {0, 1}) {
        nat::native_layout_t out;
        ASSERT_EQ(nat::get_native_layout(
                          raw_weight(raw.ptr(), K, N), wc, 0, 1, out, &handed),
                status_t::success)
                << "wc=" << wc;
        EXPECT_NE(out.data, raw.ptr()) << "wc=" << wc;
        EXPECT_EQ(out.owned != nullptr, wc == 0) << "wc=" << wc;
        nat::native_free_layout(out.owned);
    }
    EXPECT_FALSE(in_place(raw));

    nat::native_image_check_t inside;
    const auto *mid = w.buf.as<uint8_t>() + 256;
    EXPECT_TRUE(nat::native_weight_find_image(mid, w.buf.bytes - 256, inside));
    EXPECT_FALSE(inside.ready) << "an image is served from its start only";

    nat::native_image_check_t none;
    EXPECT_FALSE(nat::native_weight_find_image(raw.ptr(), raw.buf.bytes, none));
    EXPECT_FALSE(none.ready);

    for (int32_t wc : {0, 1, 2}) {
        nat::native_layout_t out;
        ASSERT_EQ(nat::get_native_layout(
                          raw_weight(w.ptr(), K, N), wc, 0, 1, out, &at_start),
                status_t::success)
                << "wc=" << wc;
        EXPECT_EQ(out.data, w.ptr()) << "wc=" << wc;
        EXPECT_EQ(out.owned, nullptr) << "wc=" << wc;
    }
    // A narrower view of the buffer reads another layout: rebuilt from the
    // image, which the view's capacity covers.
    nat::native_layout_t view;
    nat::native_raw_weight_t half = raw_weight(w.ptr(), K, N / 2);
    half.ldb = N;
    ASSERT_EQ(nat::get_native_layout(half, 1, w.buf.bytes, 1, view, &at_start),
            status_t::success);
    EXPECT_NE(view.data, w.ptr());
    nat::native_free_layout(view.owned);
    EXPECT_EQ(nt::native_layouts(w.ptr()), 2u)
            << "the image and the layout of the view";
}

/// Joins the threads it holds as it goes, so a check that fails while they
/// run leaves no joinable thread behind.
struct joiner_t {
    std::vector<std::thread> threads;
    ~joiner_t() {
        for (std::thread &t : threads)
            if (t.joinable()) t.join();
    }
};

/// Runs of a race whose overlap the scheduler decides: one run that
/// overlapped is enough, and what every run must hold holds in any order.
constexpr int kRaces = 5;

// Lookups of other buffers complete while one buffer packs, out of place at
// weight cache mode 1 or rewritten in place at mode 2, and the concurrent
// first calls on the packing buffer share one layout.
TEST_F(NativeWeightLifetime, OtherBuffersGoOnWhileABufferPacks) {
    const int K = 4096, N = 8192;
    const weight_t other = make_weight(kBf16, 64, 64, 64, false, 502);
    for (int32_t wc : {1, 2}) {
        SCOPED_TRACE("wc=" + std::to_string(wc));
        int overlapped = 0;
        for (int race = 0; race < kRaces && overlapped == 0; ++race) {
            const weight_t big = make_weight(kBf16, K, N, N, false, 501 + race);
            std::atomic<int> state {0}; // 1: the big pack started, 2: returned
            std::vector<const void *> layouts(4, nullptr);
            std::vector<status_t> packed(4, status_t::failure);
            bool ok = true;
            int during = 0;
            {
                joiner_t packers;
                for (int t = 0; t < 4; ++t)
                    packers.threads.emplace_back([&, t] {
                        nat::native_layout_t out;
                        if (t == 0) state.store(1);
                        packed[t] = nat::get_native_layout(
                                raw_weight(big.ptr(), K, N), wc, 0, 1, out);
                        layouts[t] = out.data;
                        if (t == 0) state.store(2);
                    });
                while (state.load() == 0)
                    std::this_thread::yield();
                while (ok && state.load() == 1) {
                    nat::native_layout_t out;
                    ok = nat::get_native_layout(
                                 raw_weight(other.ptr(), 64, 64), 1, 0, 1, out)
                            == status_t::success;
                    if (ok && state.load() == 1) ++during;
                }
            }
            EXPECT_TRUE(ok) << "a lookup of another buffer failed";
            for (int t = 0; t < 4; ++t) {
                EXPECT_EQ(packed[t], status_t::success) << "caller " << t;
                EXPECT_EQ(layouts[t], layouts[0]) << "one layout per weight";
            }
            EXPECT_EQ(layouts[0] == big.ptr(), wc == 2);
            EXPECT_EQ(in_place(big), wc == 2);
            EXPECT_EQ(nt::native_layouts(big.ptr()), 1u)
                    << "the layout or the image record";
            mm::clear_matmul_native_weight_caches();
            if (during > 0) ++overlapped;
        }
        EXPECT_GT(overlapped, 0)
                << "in " << kRaces << " races no call on another buffer "
                << "completed during the pack";
    }
}

/// Calls of M rows on a contiguous constant [N, K] weight, which weight
/// cache mode 2 rewrites in place: bf16 x bf16, or a bf16 source quantized
/// per token times an s4 weight in groups of 128 with bf16 scales, into f32.
struct nk_call_t {
    data_type_t wei_dt;
    int M, K, N;
    nt::buffer_t src, wei_scale;

    nk_call_t(data_type_t wdt, int m, int k, int n)
        : wei_dt(wdt), M(m), K(k), N(n) {
        src = nt::random_buffer(kBf16, static_cast<size_t>(m) * k, 21, -1, 1);
        if (wdt == kS4)
            wei_scale = nt::random_buffer(
                    kBf16, static_cast<size_t>(k / 128) * n, 22, 0.002f, 0.01f);
    }
    nt::buffer_t make_weight(uint32_t seed) const {
        if (wei_dt == kBf16)
            return nt::random_buffer(
                    kBf16, static_cast<size_t>(K) * N, seed, -1, 1);
        return nt::random_buffer(data_type_t::u8,
                nat::native_raw_weight_bytes(kS4, K, N, K, true), seed, 0, 255);
    }
    /// The call into dst, whose source scales (written by the quantization
    /// of the call) go to scales.
    nt::call_t call(const void *wei, int32_t wc, int threads, float *dst,
            float *scales) const {
        nt::call_t c;
        c.M = M;
        c.N = N;
        c.K = K;
        c.transB = true;
        c.src = src.get();
        c.wei = wei;
        c.dst = dst;
        c.params.dtypes.src = kBf16;
        c.params.dtypes.wei = wei_dt;
        c.params.dtypes.dst = data_type_t::f32;
        c.params.num_threads = threads;
        c.params.weight_cache_type = wc;
        c.params.lowoha_algo = matmul_algo_t::native_brgemm;
        if (wei_dt == kS4) {
            c.params.dtypes.compute = kS8;
            c.params.dynamic_quant = true;
            auto &qp = c.params.quant_params;
            qp.wei_scale.buff = wei_scale.get();
            qp.wei_scale.dt = kBf16;
            qp.wei_scale.dims = {K / 128, N};
            qp.src_scale.buff = scales;
            qp.src_scale.dt = data_type_t::f32;
            qp.src_scale.dims = {M, 1};
        }
        return c;
    }
    /// The output of the call; status_t::failure in st and no output when it
    /// is not served.
    std::vector<float> run(
            const void *wei, int32_t wc, int threads, status_t &st) const {
        std::vector<float> dst(static_cast<size_t>(M) * N, -777.0f);
        std::vector<float> scales(M);
        st = nt::run_matmul(call(wei, wc, threads, dst.data(), scales.data()));
        return dst;
    }
};

bool same_bits(const std::vector<float> &a, const std::vector<float> &b) {
    return a.size() == b.size()
            && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

nt::buffer_t copy_of(const nt::buffer_t &b) {
    nt::buffer_t c = nt::make_buffer(b.bytes);
    std::memcpy(c.get(), b.get(), b.bytes);
    return c;
}

// A call at weight cache mode 0 packs the raw weight of a constant buffer
// for itself while a first call on the buffer at mode 2 rewrites it in
// place: the rewrite waits until the packs in flight have read the bytes it
// writes, and a mode 0 call that comes after it reads the image.  So every
// mode 0 call, on a bf16 or an s4 weight, of four rows or of one, gives the
// bits of the same call on the raw weight or on the image, never of a
// weight read in part from each; the mode 2 call gives its bits on the raw
// weight.
TEST_F(NativeWeightLifetime, ModeZeroPacksAndARewriteDoNotOverlap) {
    pin_native();
    // Weights of 32 MiB, which one thread packs for longer than the rewrite
    // takes to start writing.
    const struct {
        data_type_t dt;
        int M, K, N;
        const char *name;
    } readers[] = {{kBf16, 4, 4096, 4096, "bf16 tiles"},
            {kBf16, 1, 4096, 4096, "bf16 one row"},
            {kS4, 4, 8192, 8192, "s4 tiles"}};
    using clock_t = std::chrono::steady_clock;
    for (const auto &rd : readers) {
        SCOPED_TRACE(rd.name);
        const nk_call_t reader(rd.dt, rd.M, rd.K, rd.N),
                writer(rd.dt, 4, rd.K, rd.N);
        int overlapped = 0;
        for (int race = 0; race < kRaces; ++race) {
            SCOPED_TRACE("race " + std::to_string(race));
            const nt::buffer_t w = reader.make_weight(801 + race);
            status_t st = status_t::failure;
            const nt::buffer_t raw = copy_of(w);
            const std::vector<float> want_raw = reader.run(raw.get(), 0, 1, st);
            ASSERT_EQ(st, status_t::success);
            const nt::buffer_t image = copy_of(w);
            const std::vector<float> want_written
                    = writer.run(image.get(), 2, 16, st);
            ASSERT_EQ(st, status_t::success);
            ASSERT_TRUE(nat::native_weight_packed_in_place(
                    image.get(), image.bytes));
            const std::vector<float> want_image
                    = reader.run(image.get(), 0, 1, st);
            ASSERT_EQ(st, status_t::success);

            // 1: the first mode 0 call starts; 2: the mode 2 call returned.
            std::atomic<int> phase {0};
            std::vector<std::vector<float>> got;
            std::vector<status_t> got_status;
            std::vector<std::pair<clock_t::time_point, clock_t::time_point>>
                    spans;
            std::vector<float> written;
            status_t written_status = status_t::failure;
            clock_t::time_point write_start, write_end;
            {
                joiner_t calls;
                calls.threads.emplace_back([&] {
                    for (bool last = false; !last;) {
                        last = phase.load() == 2;
                        const clock_t::time_point start = clock_t::now();
                        if (phase.load() == 0) phase.store(1);
                        status_t s = status_t::failure;
                        got.push_back(reader.run(w.get(), 0, 1, s));
                        got_status.push_back(s);
                        spans.emplace_back(start, clock_t::now());
                    }
                });
                while (phase.load() == 0)
                    std::this_thread::yield();
                write_start = clock_t::now();
                written = writer.run(w.get(), 2, 16, written_status);
                write_end = clock_t::now();
                phase.store(2);
            }
            EXPECT_EQ(written_status, status_t::success);
            EXPECT_TRUE(same_bits(written, want_written))
                    << "the mode 2 call read part of its own image";
            for (size_t i = 0; i < got.size(); ++i) {
                EXPECT_EQ(got_status[i], status_t::success) << "call " << i;
                EXPECT_TRUE(same_bits(got[i], want_raw)
                        || same_bits(got[i], want_image))
                        << "mode 0 call " << i << " of " << got.size()
                        << " read a weight in part raw, in part rewritten";
            }
            EXPECT_TRUE(nat::native_weight_packed_in_place(w.get(), w.bytes));
            for (const auto &s : spans)
                if (s.first < write_end && write_start < s.second) {
                    ++overlapped;
                    break;
                }
            mm::clear_matmul_native_weight_caches();
        }
        EXPECT_GT(overlapped, 0) << "in " << kRaces
                                 << " races no mode 0 call ran during the "
                                    "mode 2 call";
    }
}

// A layout of a view inside a buffer whose pack at weight cache mode 1 is
// in flight when a first call at mode 2 starts to rewrite the whole buffer
// in place is cached while the rewrite waits for the pack's read to end.
// It stays cached across the rewrite as a layout cached before the rewrite
// does: every lookup of the view, during the rewrite and after it, on the
// thread that packed it and on another, is served that layout, and none
// fails on the view now lying inside the image.
TEST_F(NativeWeightLifetime, LayoutCachedWhileARewriteWaitsStays) {
    // A view of 32 MiB, which one thread packs for longer than the rewrite
    // of the buffer takes to reserve its record.
    const int K = 4096, N = 8192;
    using clock_t = std::chrono::steady_clock;
    int overlapped = 0;
    for (int race = 0; race < kRaces; ++race) {
        SCOPED_TRACE("race " + std::to_string(race));
        const weight_t w = make_weight(kBf16, K, N, N, false, 1301 + race);
        // The last K / 2 rows of the buffer, as a weight of their own.
        const nat::native_raw_weight_t view = raw_weight(
                w.buf.as<uint16_t>() + static_cast<size_t>(K / 2) * N, K / 2,
                N);
        // 1: the view's first lookup starts; 2: the rewrite returned.
        std::atomic<int> phase {0};
        status_t first = status_t::failure, rewrite = status_t::failure;
        const void *layout = nullptr;
        int during = 0, failed = 0, other = 0;
        clock_t::time_point first_end, rewrite_start;
        {
            joiner_t calls;
            calls.threads.emplace_back([&] {
                nat::native_layout_t out;
                phase.store(1);
                first = nat::get_native_layout(view, 1, 0, 1, out);
                first_end = clock_t::now();
                layout = out.data;
                if (first != status_t::success) return;
                for (bool last = false; !last;) {
                    last = phase.load() == 2;
                    nat::native_layout_t again;
                    const status_t st
                            = nat::get_native_layout(view, 1, 0, 1, again);
                    if (st != status_t::success)
                        ++failed;
                    else if (again.data != layout)
                        ++other;
                    if (!last) ++during;
                }
            });
            while (phase.load() == 0)
                std::this_thread::yield();
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            rewrite_start = clock_t::now();
            nat::native_layout_t out;
            rewrite = nat::get_native_layout(
                    raw_weight(w.ptr(), K, N), 2, 0, 16, out);
            phase.store(2);
        }
        ASSERT_EQ(rewrite, status_t::success);
        ASSERT_TRUE(in_place(w));
        // A first lookup that came after the rewrite reserved its record
        // points inside the image and is refused; it packed nothing.
        if (first != status_t::success) {
            EXPECT_EQ(nt::native_layouts(view.weight), 0u);
            mm::clear_matmul_native_weight_caches();
            continue;
        }
        if (first_end > rewrite_start) ++overlapped;
        EXPECT_EQ(failed, 0) << "lookups of the view failed, of " << during
                             << " during the rewrite and one after it";
        EXPECT_EQ(other, 0) << "lookups of the view were served another "
                               "layout";
        nat::native_layout_t again;
        EXPECT_EQ(nat::get_native_layout(view, 1, 0, 1, again),
                status_t::success);
        EXPECT_EQ(again.data, layout) << "the cached layout was packed again";
        EXPECT_EQ(nt::native_layouts(view.weight), 1u);
        mm::clear_matmul_native_weight_caches();
    }
    EXPECT_GT(overlapped, 0)
            << "in " << kRaces << " races no view's pack ended after the "
            << "rewrite started";
}

/// A contiguous [K, N] raw weight of dt at w, in the native layout.
nat::native_raw_weight_t raw_weight(
        data_type_t dt, const void *w, int K, int N) {
    nat::native_raw_weight_t r = raw_weight(w, K, N);
    r.wei_dt = dt;
    return r;
}

/// A buffer holding a contiguous [K, N] weight of dt (bf16 or s4).
nt::buffer_t contiguous_weight(data_type_t dt, int K, int N, uint32_t seed) {
    if (dt == kBf16)
        return nt::random_buffer(
                kBf16, static_cast<size_t>(K) * N, seed, -1, 1);
    return nt::random_buffer(data_type_t::u8,
            nat::native_raw_weight_bytes(kS4, K, N, N, false), seed, 0, 255);
}

// A view of a buffer whose layout a thread cached at weight cache mode 1 and
// is served from its memo, while a first call at mode 2 rewrites the whole
// buffer in place: the view's lookups on that thread, during the rewrite and
// after it, are served the cached layout, for bf16 and s4 weights and a view
// of the first columns (at the buffer's start) or of the last rows (inside
// it).  A memo check reads no byte of the buffer while the rewrite runs.
TEST_F(NativeWeightLifetime, MemoLookupsOfAViewDuringTheFirstRewriteStay) {
    // The first two rewrite in microseconds; the others take longer than
    // the view's thread needs to look the view up several times.
    const struct {
        data_type_t dt;
        int K, N;
        bool columns;
    } shapes[] = {{kS4, 128, 64, true}, {kBf16, 128, 64, true},
            {kS4, 4096, 8192, false}, {kBf16, 2048, 4096, false}};
    for (const auto &s : shapes) {
        SCOPED_TRACE(std::string(s.dt == kS4 ? "s4" : "bf16")
                + " K=" + std::to_string(s.K) + " N=" + std::to_string(s.N)
                + (s.columns ? " first columns" : " last rows"));
        int overlapped = 0;
        for (int race = 0; race < kRaces; ++race) {
            SCOPED_TRACE("race " + std::to_string(race));
            const nt::buffer_t buf
                    = contiguous_weight(s.dt, s.K, s.N, 1401 + race);
            const nat::native_raw_weight_t whole
                    = raw_weight(s.dt, buf.get(), s.K, s.N);
            nat::native_raw_weight_t view = whole;
            if (s.columns) {
                view.N = s.N / 2;
            } else {
                view.weight = buf.as<uint8_t>() + buf.bytes / 2;
                view.K = s.K / 2;
            }
            // 1: the view is in its thread's memo; 2: the rewrite returned.
            std::atomic<int> phase {0};
            std::atomic<bool> rewriting {false};
            status_t first = status_t::failure, rewrite = status_t::failure;
            const void *layout = nullptr, *rewritten = nullptr;
            size_t memo_entries = 0;
            int failed = 0, other = 0, during = 0;
            {
                joiner_t calls;
                calls.threads.emplace_back([&] {
                    nat::native_layout_t out;
                    first = nat::get_native_layout(view, 1, 0, 1, out);
                    layout = out.data;
                    memo_entries = nat::native_weight_memo_entries(view.weight);
                    phase.store(1);
                    if (first != status_t::success) return;
                    for (bool last = false; !last;) {
                        last = phase.load() == 2;
                        const bool in_rewrite = rewriting.load();
                        nat::native_layout_t again;
                        const status_t st
                                = nat::get_native_layout(view, 1, 0, 1, again);
                        if (st != status_t::success)
                            ++failed;
                        else if (again.data != layout)
                            ++other;
                        if (in_rewrite && !last) ++during;
                    }
                });
                while (phase.load() == 0)
                    std::this_thread::yield();
                rewriting.store(true);
                nat::native_layout_t out;
                rewrite = nat::get_native_layout(whole, 2, 0, 1, out);
                rewritten = out.data;
                rewriting.store(false);
                phase.store(2);
            }
            ASSERT_EQ(first, status_t::success);
            EXPECT_EQ(memo_entries, 1u) << "the view's thread has no memo";
            ASSERT_EQ(rewrite, status_t::success);
            EXPECT_EQ(rewritten, buf.get()) << "not rewritten in place";
            EXPECT_TRUE(
                    nat::native_weight_packed_in_place(buf.get(), buf.bytes));
            EXPECT_EQ(failed, 0) << "lookups of the view failed";
            EXPECT_EQ(other, 0) << "lookups of the view were served another "
                                   "layout";
            nat::native_layout_t again;
            EXPECT_EQ(nat::get_native_layout(view, 1, 0, 1, again),
                    status_t::success);
            EXPECT_EQ(again.data, layout)
                    << "the cached layout was packed again";
            EXPECT_EQ(nt::native_layouts(view.weight), s.columns ? 2u : 1u)
                    << "the view's layout, and the image at its start";
            mm::clear_matmul_native_weight_caches();
            if (during > 0) ++overlapped;
        }
        if (!s.columns) {
            EXPECT_GT(overlapped, 0) << "in " << kRaces << " races no lookup "
                                     << "of the view ran during the rewrite";
        }
    }
}

// A layout cached from a view inside a buffer whose bytes then changed is
// not marked by a first call at weight cache mode 2 that rewrites the whole
// buffer in place: checking it reads bytes the rewrite writes, so a lookup
// of the view waits until the rewrite ends, and is then refused, as the view
// lies inside the image.  A hold of the buffer's first bytes
// (native_weight_raw_hold_t) keeps the rewrite from writing meanwhile.
TEST_F(NativeWeightLifetime, LookupOfAnUnmarkedLayoutWaitsForTheRewrite) {
    const int K = 256, N = 128;
    for (const data_type_t dt : {kBf16, kS4}) {
        SCOPED_TRACE(dt == kS4 ? "s4" : "bf16");
        const nt::buffer_t buf = contiguous_weight(dt, K, N, 1451);
        const nat::native_raw_weight_t whole = raw_weight(dt, buf.get(), K, N);
        nat::native_raw_weight_t view = whole;
        view.weight = buf.as<uint8_t>() + buf.bytes / 2;
        view.K = K / 2;
        nat::native_layout_t out;
        ASSERT_EQ(
                nat::get_native_layout(view, 1, 0, 1, out), status_t::success);
        buf.as<uint8_t>()[buf.bytes - 1] ^= 0x5a;
        std::atomic<bool> held {false}, release {false}, looked {false};
        status_t rewrite = status_t::failure, lookup = status_t::success;
        nat::native_raw_read_t read = nat::native_raw_read_t::failed;
        {
            joiner_t calls;
            calls.threads.emplace_back([&] {
                nat::native_weight_raw_hold_t hold;
                read = hold.hold(buf.get(), kEndBytes, buf.bytes);
                held.store(true);
                while (!release.load())
                    std::this_thread::yield();
            });
            while (!held.load())
                std::this_thread::yield();
            calls.threads.emplace_back([&] {
                nat::native_layout_t o;
                rewrite = nat::get_native_layout(whole, 2, 0, 1, o);
            });
            // The rewrite reserved its record and waits for the read.
            while (!nat::native_weight_packed_in_place(buf.get(), buf.bytes))
                std::this_thread::yield();
            calls.threads.emplace_back([&] {
                nat::native_layout_t o;
                lookup = nat::get_native_layout(view, 1, 0, 1, o);
                looked.store(true);
            });
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            EXPECT_FALSE(looked.load())
                    << "the lookup did not wait for the rewrite";
            release.store(true);
        }
        EXPECT_EQ(read, nat::native_raw_read_t::done);
        EXPECT_EQ(rewrite, status_t::success);
        EXPECT_TRUE(nat::native_weight_packed_in_place(buf.get(), buf.bytes));
        EXPECT_EQ(lookup, status_t::failure)
                << "a view inside the image is refused";
        mm::clear_matmul_native_weight_caches();
    }
}

// A view of a buffer whose layout a call cached at weight cache mode 1
// before a mode 2 call rewrote the whole buffer in place is served that
// layout by matmul_direct at modes 1 and 2, with the results of its raw
// weight: the last rows of a bf16 weight, its last columns, the last rows
// of a transposed one and its first rows (at the buffer's start), under
// ALGO 10 with the call's lowoha_algo native_brgemm or none (whose refusal
// would fall back to the default route), and the last rows of a weight of
// 128 KiB, whose record checks the view at its sample lines; then the last
// rows of a transposed s4 weight in groups of 128 with bf16 scales and a source
// quantized per token, with the bits of the call on a copy of its raw
// weight.  A mode 0 call on a view inside the image is refused, dst
// untouched: the image holds no raw weight for it.  With the algorithm
// unset nothing is rewritten and every call is served.
TEST_F(NativeWeightLifetime,
        ViewInsideARewrittenBufferIsServedItsCachedLayout) {
    const int K = 128, N = 64;
    const struct {
        matmul_algo_t process, call;
    } routes[] = {{matmul_algo_t::native_brgemm, matmul_algo_t::native_brgemm},
            {matmul_algo_t::native_brgemm, matmul_algo_t::none},
            {matmul_algo_t::none, matmul_algo_t::native_brgemm}};
    const char *const views[] = {
            "last rows", "last columns", "transposed, last rows", "first rows"};
    for (const auto &route : routes) {
        const bool rewrites = route.process == matmul_algo_t::native_brgemm;
        const nt::algo_pin_t pin(route.process);
        for (int which = 0; which < 4; ++which) {
            SCOPED_TRACE(std::string(rewrites ? "ALGO 10" : "algorithm unset")
                    + (route.call == matmul_algo_t::none ? ", call algo none"
                                                         : "")
                    + ", " + views[which]);
            ::reset_grp_matmul_caches();
            const bool trans = which == 2;
            const weight_t whole = make_weight(
                    kBf16, K, N, trans ? K : N, trans, 1121 + which);
            weight_t view = which == 0
                    ? part_of(whole, K / 2 * N * 2, K / 2, N, N)
                    : which == 1 ? part_of(whole, N / 2 * 2, K, N / 2, N)
                    : which == 2 ? part_of(whole, N / 2 * K * 2, K, N / 2, K)
                                 : part_of(whole, 0, K / 2, N, N);
            view.trans_b = trans;
            view.buf.bytes = view.span();
            const weight_t raw = clone(view);
            dense_t dv(kBf16, view.K), dw(kBf16, K);
            ASSERT_EQ(dv.run(view, 1, 0, route.call), status_t::success);
            EXPECT_TRUE(dv.matches(raw));
            ASSERT_EQ(dw.run(whole, 2, 0, route.call), status_t::success);
            ASSERT_EQ(in_place(whole), rewrites);
            for (const int32_t wc : {1, 2}) {
                ASSERT_EQ(dv.run(view, wc, 0, route.call), status_t::success)
                        << "weight cache mode " << wc;
                EXPECT_TRUE(dv.matches(raw)) << "weight cache mode " << wc;
            }
            if (rewrites && which != 3) {
                EXPECT_EQ(dv.run(view, 0, 0, route.call), status_t::failure);
                EXPECT_TRUE(dv.untouched());
            }
        }
    }
    {
        SCOPED_TRACE("ALGO 10, last rows of a weight of 128 KiB");
        const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
        ::reset_grp_matmul_caches();
        const int KL = 1024;
        const weight_t whole = make_weight(kBf16, KL, N, N, false, 1125);
        const weight_t view = part_of(whole, KL / 2 * N * 2, KL / 2, N, N);
        const weight_t raw = clone(view);
        dense_t dv(kBf16, view.K), dw(kBf16, KL);
        ASSERT_EQ(dv.run(view, 1), status_t::success);
        EXPECT_TRUE(dv.matches(raw));
        ASSERT_EQ(dw.run(whole, 2), status_t::success);
        ASSERT_TRUE(in_place(whole));
        for (const int32_t wc : {1, 2}) {
            ASSERT_EQ(dv.run(view, wc), status_t::success)
                    << "weight cache mode " << wc;
            EXPECT_TRUE(dv.matches(raw)) << "weight cache mode " << wc;
        }
        EXPECT_EQ(dv.run(view, 0), status_t::failure);
        EXPECT_TRUE(dv.untouched());
    }
    SCOPED_TRACE("s4, transposed, last rows");
    pin_native();
    ::reset_grp_matmul_caches();
    const nk_call_t whole_call(kS4, 4, 256, 128), view_call(kS4, 4, 256, 64);
    const nt::buffer_t whole = whole_call.make_weight(1131);
    const uint8_t *view = whole.as<uint8_t>() + whole.bytes / 2;
    const nt::buffer_t raw = nt::make_buffer(whole.bytes / 2);
    std::memcpy(raw.get(), view, raw.bytes);
    status_t st = status_t::failure;
    const std::vector<float> want = view_call.run(raw.get(), 0, 4, st);
    ASSERT_EQ(st, status_t::success);
    EXPECT_TRUE(same_bits(view_call.run(view, 1, 4, st), want));
    ASSERT_EQ(st, status_t::success);
    whole_call.run(whole.get(), 2, 4, st);
    ASSERT_EQ(st, status_t::success);
    ASSERT_TRUE(nat::native_weight_packed_in_place(whole.get(), whole.bytes));
    for (const int32_t wc : {1, 2}) {
        EXPECT_TRUE(same_bits(view_call.run(view, wc, 4, st), want))
                << "weight cache mode " << wc;
        EXPECT_EQ(st, status_t::success) << "weight cache mode " << wc;
    }
    const std::vector<float> refused = view_call.run(view, 0, 4, st);
    EXPECT_EQ(st, status_t::failure);
    EXPECT_TRUE(std::all_of(refused.begin(), refused.end(),
            [](float v) { return v == -777.0f; }));
}

// The line a call logs names the memory it read: weight_cache=inplace for
// a call served from an in-place image, the weight cache mode for a view
// inside it served the layout its rewrite carried.
TEST_F(NativeWeightLifetime, AViewServedInsideAnImageLogsItsMode) {
    nt::expect_in_child({{"ZENDNNL_API_LOG_LEVEL", "4"}}, [this] {
        pin_native();
        const int K = 128, N = 64;
        const weight_t whole = make_weight(kBf16, K, N, N, false, 1241);
        const weight_t view = part_of(whole, K * N, K / 2, N, N);
        dense_t dv(kBf16, view.K), dw(kBf16, K);
        if (dv.run(view, 1) != status_t::success) return 1;
        if (dw.run(whole, 2) != status_t::success || !in_place(whole)) return 1;
        int bad = 0;
        for (const int32_t wc : {1, 2}) {
            ::testing::internal::CaptureStdout();
            const bool served = dv.run(view, wc) == status_t::success;
            const std::string log = ::testing::internal::GetCapturedStdout();
            if (!served
                    || log.find("weight_cache=" + std::to_string(wc))
                            == std::string::npos
                    || log.find("weight_cache=inplace") != std::string::npos) {
                std::fprintf(
                        stderr, "the view at mode %d:\n%s\n", wc, log.c_str());
                ++bad;
            }
        }
        ::testing::internal::CaptureStdout();
        const bool served = dw.run(whole, 2) == status_t::success;
        const std::string log = ::testing::internal::GetCapturedStdout();
        if (!served || log.find("weight_cache=inplace") == std::string::npos) {
            std::fprintf(stderr, "the image:\n%s\n", log.c_str());
            ++bad;
        }
        return bad;
    }, "ZENDNNL_API_LOG_LEVEL=4");
}

// A cached layout serves a call whose weight an in-place image holds only
// when it was packed from that image or carried across its rewrite.  A view
// inside the image whose bytes changed after its layout was cached and
// before the buffer was rewritten is refused, dst untouched; so is one whose
// buffer was rewritten again for another weight with no clear (a model
// loaded where another was freed), whose image holds the same bytes over
// the view as the first: the new weight differs in one element of the
// view's rows whose column the view's bytes in the image do not hold.  The
// same view at the buffer's start, stating the buffer as its capacity, is
// packed again from the new image, with the new weight's results.
TEST_F(NativeWeightLifetime, AViewIsServedOnlyALayoutCarriedAcrossItsImage) {
    pin_native();
    const int K = 128, N = 64;
    const char *const cases[] = {"inside, its bytes changed before the rewrite",
            "at the start, the buffer rewritten again for another weight",
            "inside, the buffer rewritten again for another weight"};
    for (int which = 0; which < 3; ++which) {
        SCOPED_TRACE(cases[which]);
        ::reset_grp_matmul_caches();
        const bool start = which == 1, reloaded = which != 0;
        const size_t offset = start ? 0 : K / 2 * N * 2;
        const weight_t whole = make_weight(kBf16, K, N, N, false, 1141);
        const weight_t view = part_of(whole, offset, K / 2, N, N);
        const size_t capacity = start ? whole.buf.bytes : 0;
        dense_t dv(kBf16, view.K), dw(kBf16, K);
        ASSERT_EQ(dv.run(view, 1, capacity), status_t::success);
        if (!reloaded) fill(view, 1142);
        const weight_t raw = clone(whole);
        ASSERT_EQ(dw.run(whole, 2), status_t::success);
        ASSERT_TRUE(in_place(whole));
        if (reloaded) {
            ASSERT_EQ(dv.run(view, 1, capacity), status_t::success);
            EXPECT_TRUE(dv.matches(part_of(raw, offset, K / 2, N, N)));
            const weight_t image = clone(whole);
            // A row of the view; the image holds the column in a block of
            // 32 columns outside the view's bytes.
            nt::store_f32(raw.buf.get(), kBf16,
                    start ? 10 * N + 40 : 100 * N + 10, 3.0f);
            std::memcpy(whole.buf.get(), raw.ptr(), whole.buf.bytes);
            ASSERT_EQ(dw.run(whole, 2), status_t::success);
            EXPECT_TRUE(dw.matches(raw));
            ASSERT_TRUE(in_place(whole));
            ASSERT_EQ(std::memcmp(view.ptr(), image.buf.as<uint8_t>() + offset,
                              view.buf.bytes),
                    0)
                    << "the new image differs over the view";
        }
        if (start) {
            ASSERT_EQ(dv.run(view, 1, capacity), status_t::success);
            EXPECT_TRUE(dv.matches(part_of(raw, 0, K / 2, N, N)));
        } else {
            EXPECT_EQ(dv.run(view, 1), status_t::failure);
            EXPECT_TRUE(dv.untouched());
        }
    }
}

// A view of a weight rewritten in place, refused as it lies inside the
// image, whose raw elements the caller then writes back, after a clear or
// with none: the last rows of the weight, called at weight cache modes 1
// and 2, are read as that raw weight, and the record forgets the image
// over the view's buffer only; its last columns, whose rows lie ldb apart
// with bytes of the image between them, still fail with dst untouched.  The
// next call on the whole weight, whose other bytes still hold the image,
// fails with dst untouched, never success with the image read as a raw
// weight, until the whole raw weight is back, and the view is served then.
TEST_F(NativeWeightLifetime, AViewWrittenBackLeavesTheRestOfItsImageRecorded) {
    pin_native();
    const int K = 128, N = 64;
    uint32_t seed = 1151;
    for (const bool cleared : {true, false})
        for (const bool columns : {false, true})
            for (const int32_t wc : {1, 2}) {
                SCOPED_TRACE(std::string(cleared ? "after a clear" : "no clear")
                        + (columns ? ", last columns" : ", last rows")
                        + ", then wc=" + std::to_string(wc));
                ::reset_grp_matmul_caches();
                const weight_t whole
                        = make_weight(kBf16, K, N, N, false, ++seed);
                const weight_t raw = clone(whole);
                const size_t offset = columns ? N : static_cast<size_t>(K) * N;
                const weight_t view = columns
                        ? part_of(whole, offset, K, N / 2, N)
                        : part_of(whole, offset, K / 2, N, N);
                const weight_t raw_view
                        = part_of(raw, offset, view.K, view.N, view.ldb);
                dense_t dv(kBf16, view.K), dw(kBf16, K);
                ASSERT_EQ(dw.run(whole, 2), status_t::success);
                ASSERT_TRUE(in_place(whole));
                if (cleared) mm::clear_matmul_native_weight_caches();
                EXPECT_EQ(dv.run(view, wc), status_t::failure);
                EXPECT_TRUE(dv.untouched());
                for (int r = 0; r < view.rows(); ++r)
                    std::memcpy(view.buf.as<uint8_t>() + 2 * r * view.ldb,
                            raw_view.buf.as<uint8_t>() + 2 * r * view.ldb,
                            2 * static_cast<size_t>(view.cols()));
                if (columns) {
                    EXPECT_EQ(dv.run(view, wc), status_t::failure);
                    EXPECT_TRUE(dv.untouched());
                } else {
                    ASSERT_EQ(dv.run(view, wc), status_t::success);
                    EXPECT_TRUE(dv.matches(raw_view));
                }
                EXPECT_EQ(dw.run(whole, 1), status_t::failure)
                        << "the image read as a raw weight";
                EXPECT_TRUE(dw.untouched());
                std::memcpy(whole.buf.get(), raw.ptr(), whole.buf.bytes);
                ASSERT_EQ(dw.run(whole, 1), status_t::success);
                EXPECT_TRUE(dw.matches(raw));
                ASSERT_EQ(dv.run(view, wc), status_t::success);
                EXPECT_TRUE(dv.matches(raw_view));
            }
}

// A view whose rows lie ldb apart, the last 32 columns of a 64 x 64 bf16
// weight rewritten in place, whose raw elements the caller writes back
// fails with dst untouched, after a clear or with none: the first columns
// between its rows still hold the image, and the record keeps them.  So
// does a call on a weight in those bytes, the first 32 columns of row 1,
// never success with the image read as a raw weight, and the call on the
// whole weight.  So does a view of the first 16 columns once the columns
// from 16 on are written back, every line of its span holding written
// bytes: its own rows still hold the image.  Once the whole raw weight is
// back, each is served.
TEST_F(NativeWeightLifetime,
        AStridedViewWrittenBackLeavesTheBytesBetweenItsRows) {
    pin_native();
    const int K = 64, N = 64;
    uint32_t seed = 1241;
    for (const bool cleared : {true, false}) {
        SCOPED_TRACE(cleared ? "after a clear" : "no clear");
        ::reset_grp_matmul_caches();
        const weight_t whole = make_weight(kBf16, K, N, N, false, ++seed);
        const weight_t raw = clone(whole);
        const weight_t view = part_of(whole, N, K, N / 2, N);
        const weight_t raw_view = part_of(raw, N, K, N / 2, N);
        const weight_t gap = part_of(whole, 2 * N, 1, N / 2, N / 2);
        const weight_t raw_gap = part_of(raw, 2 * N, 1, N / 2, N / 2);
        dense_t dv(kBf16, K), dg(kBf16, 1), dw(kBf16, K);
        ASSERT_EQ(dw.run(whole, 2), status_t::success);
        ASSERT_TRUE(in_place(whole));
        if (cleared) mm::clear_matmul_native_weight_caches();
        for (int r = 0; r < K; ++r)
            std::memcpy(view.buf.as<uint8_t>() + 2 * r * N,
                    raw_view.buf.as<uint8_t>() + 2 * r * N, N);
        EXPECT_EQ(dv.run(view, 1), status_t::failure);
        EXPECT_TRUE(dv.untouched());
        EXPECT_EQ(dg.run(gap, 1), status_t::failure)
                << "the image read as a raw weight";
        EXPECT_TRUE(dg.untouched());
        EXPECT_EQ(dw.run(whole, 1), status_t::failure);
        EXPECT_TRUE(dw.untouched());
        std::memcpy(whole.buf.get(), raw.ptr(), whole.buf.bytes);
        ASSERT_EQ(dg.run(gap, 1), status_t::success);
        EXPECT_TRUE(dg.matches(raw_gap));
        ASSERT_EQ(dv.run(view, 1), status_t::success);
        EXPECT_TRUE(dv.matches(raw_view));
        ASSERT_EQ(dw.run(whole, 1), status_t::success);
        EXPECT_TRUE(dw.matches(raw));
    }
    for (const bool cleared : {true, false}) {
        SCOPED_TRACE(std::string(cleared ? "after a clear" : "no clear")
                + ", the first 16 columns");
        ::reset_grp_matmul_caches();
        const weight_t whole = make_weight(kBf16, K, N, N, false, ++seed);
        const weight_t raw = clone(whole);
        const weight_t view = part_of(whole, 0, K, 16, N);
        const weight_t raw_view = part_of(raw, 0, K, 16, N);
        dense_t dv(kBf16, K), dw(kBf16, K);
        ASSERT_EQ(dw.run(whole, 2), status_t::success);
        ASSERT_TRUE(in_place(whole));
        if (cleared) mm::clear_matmul_native_weight_caches();
        for (int r = 0; r < K; ++r)
            std::memcpy(whole.buf.as<uint8_t>() + 2 * r * N + 32,
                    raw.buf.as<uint8_t>() + 2 * r * N + 32, 2 * N - 32);
        EXPECT_EQ(dv.run(view, 1), status_t::failure)
                << "the image read as a raw weight";
        EXPECT_TRUE(dv.untouched());
        std::memcpy(whole.buf.get(), raw.ptr(), whole.buf.bytes);
        ASSERT_EQ(dv.run(view, 1), status_t::success);
        EXPECT_TRUE(dv.matches(raw_view));
        ASSERT_EQ(dw.run(whole, 1), status_t::success);
        EXPECT_TRUE(dw.matches(raw));
    }
}

// A buffer whose bytes still hold its image is refused after one, two and
// three clears, never read as a raw weight: 32 bytes inside a line of a
// 64 x 64 bf16 image.  It is served once the whole raw weight is back.
TEST_F(NativeWeightLifetime, AFreedTombstoneRefusesBytesThatHoldItsImage) {
    pin_native();
    const weight_t whole = make_weight(kBf16, 64, 64, 64, false, 1252);
    const weight_t raw = clone(whole);
    const weight_t slice = part_of(whole, 2064, 1, 16, 16);
    dense_t dw(kBf16, 64), ds(kBf16, 1);
    ASSERT_EQ(dw.run(whole, 2), status_t::success);
    ASSERT_TRUE(in_place(whole));
    for (int clears = 1; clears <= 3; ++clears) {
        mm::clear_matmul_native_weight_caches();
        EXPECT_EQ(ds.run(slice, 1), status_t::failure)
                << clears << " clears: the image read as a raw weight";
        EXPECT_TRUE(ds.untouched());
    }
    std::memcpy(whole.buf.get(), raw.ptr(), raw.buf.bytes);
    ASSERT_EQ(dw.run(whole, 1), status_t::success);
    EXPECT_TRUE(dw.matches(raw));
    ASSERT_EQ(ds.run(slice, 1), status_t::success);
    EXPECT_TRUE(ds.matches(part_of(raw, 2064, 1, 16, 16)));
}

// A bf16 K = 2, N = 32 weight of a row of ones and a row of twos, whose
// image would repeat one word in each line, so that no word of it could
// tell the image from the weight written back around it, is not rewritten
// in place at weight cache mode 2: its layout is cached out of place and
// the buffer keeps every byte.  After a clear and its first raw row
// written back, a mode 1 call with a source of ones gives 3 in every
// column, and so it does after a second and a third clear.
TEST_F(NativeWeightLifetime, AWeightWhoseImageNoWordTellsIsNotRewritten) {
    pin_native();
    const weight_t w = make_weight(kBf16, 2, 32, 32, false, 1251);
    for (size_t n = 0; n < 32; ++n) {
        nt::store_f32(w.buf.get(), kBf16, n, 1.0f);
        nt::store_f32(w.buf.get(), kBf16, 32 + n, 2.0f);
    }
    const weight_t raw = clone(w);
    dense_t d(kBf16, 2);
    for (size_t i = 0; i < 2 * dense_t::M; ++i)
        nt::store_f32(d.src.get(), kBf16, i, 1.0f);
    ASSERT_EQ(d.run(w, 2), status_t::success);
    EXPECT_TRUE(d.matches(raw));
    EXPECT_FALSE(in_place(w));
    EXPECT_EQ(std::memcmp(w.ptr(), raw.ptr(), w.buf.bytes), 0)
            << "the buffer was written";
    EXPECT_EQ(nt::native_layouts(w.ptr()), 1u) << "cached out of place";
    for (int clears = 1; clears <= 3; ++clears) {
        mm::clear_matmul_native_weight_caches();
        std::memcpy(w.buf.get(), raw.ptr(), 64);
        ASSERT_EQ(d.run(w, 1), status_t::success) << clears << " clears";
        EXPECT_TRUE(std::all_of(d.dst.begin(), d.dst.end(),
                [](float v) {
            return v == 3.0f;
        })) << clears
            << " clears: not 3 in every column";
    }
}

// The raw elements of the first 32 columns of a 64 x 64 bf16 image written
// back, after a clear and with none: a call on them as a view whose stated
// capacity is the whole 8 KiB buffer fails with dst untouched, and so do
// later calls on the untouched columns and on the whole weight, never
// success with the image read as a raw weight; once the whole raw weight
// is back, each is served.
TEST_F(NativeWeightLifetime, AFullCapacityViewOfColumnsWrittenBackIsRefused) {
    pin_native();
    const int K = 64, N = 64;
    uint32_t seed = 1261;
    for (const bool cleared : {true, false}) {
        SCOPED_TRACE(cleared ? "after a clear" : "no clear");
        ::reset_grp_matmul_caches();
        const weight_t whole = make_weight(kBf16, K, N, N, false, ++seed);
        const weight_t raw = clone(whole);
        const weight_t first = part_of(whole, 0, K, N / 2, N);
        const weight_t rest = part_of(whole, N, K, N / 2, N);
        dense_t dv(kBf16, K), dw(kBf16, K);
        ASSERT_EQ(dw.run(whole, 2), status_t::success);
        ASSERT_TRUE(in_place(whole));
        if (cleared) mm::clear_matmul_native_weight_caches();
        for (int r = 0; r < K; ++r)
            std::memcpy(whole.buf.as<uint8_t>() + 2 * r * N,
                    raw.buf.as<uint8_t>() + 2 * r * N, N);
        EXPECT_EQ(dv.run(first, 1, whole.buf.bytes), status_t::failure);
        EXPECT_TRUE(dv.untouched());
        EXPECT_EQ(dv.run(rest, 1), status_t::failure)
                << "the image read as a raw weight";
        EXPECT_TRUE(dv.untouched());
        EXPECT_EQ(dw.run(whole, 1), status_t::failure);
        EXPECT_TRUE(dw.untouched());
        std::memcpy(whole.buf.get(), raw.ptr(), whole.buf.bytes);
        ASSERT_EQ(dw.run(whole, 1), status_t::success);
        EXPECT_TRUE(dw.matches(raw));
        ASSERT_EQ(dv.run(first, 1, whole.buf.bytes), status_t::success);
        EXPECT_TRUE(dv.matches(part_of(raw, 0, K, N / 2, N)));
        ASSERT_EQ(dv.run(rest, 1), status_t::success);
        EXPECT_TRUE(dv.matches(part_of(raw, N, K, N / 2, N)));
    }
}

// Part of the raw weight of a 64 x 64 bf16 image written back, after a
// clear and with none, leaves bytes of the image that a call finds among
// the raw ones: once columns 8 to 55 are back, their view (rows 64 apart)
// fails with dst untouched, and so do the untouched last columns and the
// whole weight; once the first 32 rows are back, the whole weight fails,
// while those rows are read as the raw weight they hold.  Never success
// with the image read as a raw weight; once the whole raw weight is back,
// each is served.
TEST_F(NativeWeightLifetime, APartialWriteBackIsRefusedWhereItsImageShows) {
    pin_native();
    const int K = 64, N = 64;
    uint32_t seed = 1271;
    for (const bool cleared : {true, false}) {
        SCOPED_TRACE(std::string(cleared ? "after a clear" : "no clear")
                + ", columns 8 to 55");
        ::reset_grp_matmul_caches();
        const weight_t whole = make_weight(kBf16, K, N, N, false, ++seed);
        const weight_t raw = clone(whole);
        const weight_t middle = part_of(whole, 16, K, 48, N);
        const weight_t last = part_of(whole, 112, K - 1, 8, N);
        dense_t dm(kBf16, K), dl(kBf16, K - 1), dw(kBf16, K);
        ASSERT_EQ(dw.run(whole, 2), status_t::success);
        ASSERT_TRUE(in_place(whole));
        if (cleared) mm::clear_matmul_native_weight_caches();
        for (int r = 0; r < K; ++r)
            std::memcpy(whole.buf.as<uint8_t>() + 2 * r * N + 16,
                    raw.buf.as<uint8_t>() + 2 * r * N + 16, 96);
        EXPECT_EQ(dm.run(middle, 1), status_t::failure);
        EXPECT_TRUE(dm.untouched());
        EXPECT_EQ(dl.run(last, 1), status_t::failure)
                << "the image read as a raw weight";
        EXPECT_TRUE(dl.untouched());
        EXPECT_EQ(dw.run(whole, 1), status_t::failure);
        EXPECT_TRUE(dw.untouched());
        std::memcpy(whole.buf.get(), raw.ptr(), whole.buf.bytes);
        ASSERT_EQ(dw.run(whole, 1), status_t::success);
        EXPECT_TRUE(dw.matches(raw));
        ASSERT_EQ(dm.run(middle, 1), status_t::success);
        EXPECT_TRUE(dm.matches(part_of(raw, 16, K, 48, N)));
        ASSERT_EQ(dl.run(last, 1), status_t::success);
        EXPECT_TRUE(dl.matches(part_of(raw, 112, K - 1, 8, N)));
    }
    for (const bool cleared : {true, false}) {
        SCOPED_TRACE(std::string(cleared ? "after a clear" : "no clear")
                + ", the first 32 rows");
        ::reset_grp_matmul_caches();
        const weight_t whole = make_weight(kBf16, K, N, N, false, ++seed);
        const weight_t raw = clone(whole);
        const weight_t top = part_of(whole, 0, K / 2, N, N);
        dense_t dt(kBf16, K / 2), dw(kBf16, K);
        ASSERT_EQ(dw.run(whole, 2), status_t::success);
        ASSERT_TRUE(in_place(whole));
        if (cleared) mm::clear_matmul_native_weight_caches();
        std::memcpy(whole.buf.get(), raw.ptr(), whole.buf.bytes / 2);
        EXPECT_EQ(dw.run(whole, 1), status_t::failure)
                << "the image read as a raw weight";
        EXPECT_TRUE(dw.untouched());
        ASSERT_EQ(dt.run(top, 1), status_t::success);
        EXPECT_TRUE(dt.matches(part_of(raw, 0, K / 2, N, N)));
        EXPECT_EQ(dw.run(whole, 1), status_t::failure);
        EXPECT_TRUE(dw.untouched());
        std::memcpy(whole.buf.get(), raw.ptr(), whole.buf.bytes);
        ASSERT_EQ(dw.run(whole, 1), status_t::success);
        EXPECT_TRUE(dw.matches(raw));
    }
}

// A view of five rows of a cleared 64 x 64 bf16 image, whose raw rows were
// written back, called at weight cache mode 2 with the rest of the buffer as
// its capacity, is read as the raw weight it holds and rewritten in place
// into six rows (its layout pads K to an even count). The rows of the image
// past the bytes it wrote and past its capacity's checked bytes, which still
// hold the image, stay recorded: a call on the last two rows fails with dst
// untouched, never success with the image read as a raw weight, and is
// served once the raw weight of the whole buffer is back.
TEST_F(NativeWeightLifetime, APaddedRewriteKeepsTheImagePastItsBytes) {
    pin_native();
    const int K = 64, N = 64;
    const size_t at = 55 * 2 * N, tail_at = 62 * 2 * N;
    const weight_t whole = make_weight(kBf16, K, N, N, false, 1291);
    const weight_t raw = clone(whole);
    dense_t dw(kBf16, K), dv(kBf16, 5), dt(kBf16, 2);
    ASSERT_EQ(dw.run(whole, 2), status_t::success);
    ASSERT_TRUE(in_place(whole));
    mm::clear_matmul_native_weight_caches();
    std::memcpy(whole.buf.as<uint8_t>() + at, raw.buf.as<uint8_t>() + at,
            5 * 2 * N);
    const weight_t view = part_of(whole, at, 5, N, N);
    ASSERT_EQ(dv.run(view, 2, whole.buf.bytes - at), status_t::success);
    EXPECT_TRUE(dv.matches(part_of(raw, at, 5, N, N)));
    const weight_t tail = part_of(whole, tail_at, 2, N, N);
    EXPECT_EQ(dt.run(tail, 1), status_t::failure)
            << "the image read as a raw weight";
    EXPECT_TRUE(dt.untouched());
    std::memcpy(whole.buf.get(), raw.ptr(), whole.buf.bytes);
    ASSERT_EQ(dt.run(tail, 1), status_t::success);
    EXPECT_TRUE(dt.matches(part_of(raw, tail_at, 2, N, N)));
}

// The first five rows of a cleared 64 x 64 bf16 image, zeroed or given
// their raw rows back, called at weight cache mode 2 with the whole 8 KiB
// buffer as their capacity: the rewrite into the six rows of their layout
// (K padded to an even count) writes nothing, as the zero weight's image
// equals its bytes, or as no memory is left for the checks of the image,
// and the call is served out of place.  The image's bytes past the five
// rows, which the rewrite would have written, stay recorded: a call on the
// first six rows fails with dst untouched, never success with the image
// read as a raw weight, and is served once the whole raw weight is back.
TEST_F(NativeWeightLifetime, ARewriteThatWritesNothingLosesNoRecord) {
    pin_native();
    const int K = 64, N = 64;
    const size_t five_rows = 5 * 2 * N;
    uint32_t seed = 1301;
    for (const bool zeros : {true, false}) {
        SCOPED_TRACE(zeros ? "five zeroed rows" : "no memory for the checks");
        ::reset_grp_matmul_caches();
        const weight_t whole = make_weight(kBf16, K, N, N, false, ++seed);
        const weight_t raw = clone(whole);
        const weight_t five = part_of(whole, 0, 5, N, N);
        const weight_t six = part_of(whole, 0, 6, N, N);
        dense_t dw(kBf16, K), d5(kBf16, 5), d6(kBf16, 6);
        ASSERT_EQ(dw.run(whole, 2), status_t::success);
        ASSERT_TRUE(in_place(whole));
        mm::clear_matmul_native_weight_caches();
        if (zeros)
            std::memset(whole.buf.get(), 0, five_rows);
        else
            std::memcpy(whole.buf.get(), raw.ptr(), five_rows);
        const weight_t before = clone(whole);
        if (!zeros) nat::native_fail_next_image_checks(1);
        const status_t st = d5.run(five, 2, whole.buf.bytes);
        nat::native_fail_next_image_checks(0);
        ASSERT_EQ(st, status_t::success);
        EXPECT_TRUE(d5.matches(five));
        EXPECT_EQ(std::memcmp(whole.ptr(), before.ptr(), whole.buf.bytes), 0)
                << "the buffer was written";
        EXPECT_EQ(d6.run(six, 1), status_t::failure)
                << "the image read as a raw weight";
        EXPECT_TRUE(d6.untouched());
        std::memcpy(whole.buf.get(), raw.ptr(), whole.buf.bytes);
        ASSERT_EQ(d6.run(six, 1), status_t::success);
        EXPECT_TRUE(d6.matches(part_of(raw, 0, 6, N, N)));
        ASSERT_EQ(dw.run(whole, 1), status_t::success);
        EXPECT_TRUE(dw.matches(raw));
    }
}

// The raw weight written back over its in-place image with no clear, as a
// model unloaded and loaded again at the same address, is read as that raw
// weight by the next call (and rewritten in place again): bf16 weights,
// transposed or not, and transposed s4 weights in groups of 128, of images
// of up to 64 KiB and larger.
TEST_F(NativeWeightLifetime, ARawWeightWrittenBackOverItsImageIsReadAsRaw) {
    pin_native();
    uint32_t seed = 1161;
    for (const int K : {64, 1024})
        for (const bool trans_b : {false, true}) {
            SCOPED_TRACE(
                    "bf16 K=" + std::to_string(K) + (trans_b ? " transB" : ""));
            ::reset_grp_matmul_caches();
            const weight_t w = make_weight(
                    kBf16, K, 64, trans_b ? K : 64, trans_b, ++seed);
            const weight_t raw = clone(w);
            dense_t d(kBf16, K);
            for (int call = 0; call < 2; ++call) {
                ASSERT_EQ(d.run(w, 2), status_t::success);
                EXPECT_TRUE(d.matches(raw)) << "call " << call;
                ASSERT_TRUE(in_place(w));
                std::memcpy(w.buf.get(), raw.ptr(), w.buf.bytes);
            }
        }
    for (const int K : {256, 4096}) {
        SCOPED_TRACE("s4 K=" + std::to_string(K));
        ::reset_grp_matmul_caches();
        const nk_call_t call(kS4, 4, K, 64);
        const nt::buffer_t w = call.make_weight(++seed);
        const nt::buffer_t raw = copy_of(w);
        status_t st = status_t::failure;
        const std::vector<float> want = call.run(raw.get(), 0, 4, st);
        ASSERT_EQ(st, status_t::success);
        for (int i = 0; i < 2; ++i) {
            EXPECT_TRUE(same_bits(call.run(w.get(), 2, 4, st), want))
                    << "call " << i;
            EXPECT_EQ(st, status_t::success);
            ASSERT_TRUE(nat::native_weight_packed_in_place(w.get(), w.bytes));
            std::memcpy(w.get(), raw.get(), raw.bytes);
        }
    }
}

// ── The per-thread layout memo ───────────────────────────────────────────

// Concurrent first calls on one weight share one layout, and every caller's
// later calls, served from its memo, are served that layout.
TEST_F(NativeWeightLifetime, ConcurrentFirstCallsShareOneLayoutAfterward) {
    const int K = 256, N = 512, kCallers = 6, kCalls = 4;
    for (int32_t wc : {1, 2}) {
        SCOPED_TRACE("wc=" + std::to_string(wc));
        ::reset_grp_matmul_caches();
        if (wc == 2) pin_native();
        const weight_t w = make_weight(kBf16, K, N, N, false, 601 + wc);
        const weight_t w1 = clone(w);
        std::vector<std::vector<const void *>> seen(
                kCallers, std::vector<const void *>(kCalls, nullptr));
        std::vector<int> wrong(kCallers, 0);
        std::vector<size_t> memo(kCallers, 0);
        std::atomic<int> ready {0};
        {
            joiner_t callers;
            for (int t = 0; t < kCallers; ++t)
                callers.threads.emplace_back([&, t] {
                    ready.fetch_add(1);
                    while (ready.load() < kCallers)
                        std::this_thread::yield();
                    for (int i = 0; i < kCalls; ++i) {
                        nat::native_layout_t out;
                        if (nat::get_native_layout(
                                    raw_weight(w.ptr(), K, N), wc, 0, 1, out)
                                != status_t::success)
                            ++wrong[t];
                        seen[t][i] = out.data;
                    }
                    dense_t d(kBf16, K);
                    if (d.run(w, wc) != status_t::success || !d.matches(w1))
                        ++wrong[t];
                    memo[t] = nat::native_weight_memo_entries(w.ptr());
                });
        }
        for (int t = 0; t < kCallers; ++t) {
            EXPECT_EQ(wrong[t], 0) << "caller " << t;
            EXPECT_GE(memo[t], 1u) << "caller " << t << " kept no memo entry";
            for (int i = 0; i < kCalls; ++i)
                EXPECT_EQ(seen[t][i], seen[0][0])
                        << "caller " << t << " call " << i;
        }
        EXPECT_EQ(seen[0][0] == w.ptr(), wc == 2);
        EXPECT_EQ(nt::native_layouts(w.ptr()), 1u);
        mm::clear_matmul_native_weight_caches();
    }
}

// A thread's later calls on an in-place image follow the buffer: other bytes
// at its ends are read as the raw weight they are, the image's own bytes at
// both ends are served the image until the native weight caches are
// cleared, and refused after the clear, until the ends hold other bytes.
TEST_F(NativeWeightLifetime, RepeatedCallsOnAnImageFollowTheBuffer) {
    pin_native();
    const int K = 64, N = 64;
    const weight_t w = make_weight(kBf16, K, N, N, false, 611);
    const weight_t w1 = clone(w);
    dense_t d(kBf16, K);
    for (int i = 0; i < 3; ++i) {
        ASSERT_EQ(d.run(w, 2), status_t::success);
        EXPECT_TRUE(d.matches(w1)) << "call " << i;
    }
    ASSERT_TRUE(in_place(w));
    const weight_t image = clone(w);

    fill(w, 612);
    const weight_t w2 = clone(w);
    for (int i = 0; i < 3; ++i) {
        ASSERT_EQ(d.run(w, 2), status_t::success);
        EXPECT_TRUE(d.matches(w2)) << "call " << i << " on the new weight";
    }
    ASSERT_TRUE(in_place(w)) << "the new weight, rewritten in place";

    const weight_t image2 = clone(w);
    fill(w, 613);
    keep_ends(w, image2.ptr());
    const weight_t w3 = clone(w);
    for (int i = 0; i < 2; ++i) {
        ASSERT_EQ(d.run(w, 2), status_t::success);
        EXPECT_FALSE(d.matches(w3)) << "call " << i << ": read as the image";
    }
    mm::clear_matmul_native_weight_caches();
    EXPECT_EQ(d.run(w, 2), status_t::failure) << "the image's ends";
    EXPECT_TRUE(d.untouched());
    fill(w, 614);
    const weight_t w4 = clone(w);
    ASSERT_EQ(d.run(w, 2), status_t::success);
    EXPECT_TRUE(d.matches(w4));
    EXPECT_NE(std::memcmp(image.ptr(), image2.ptr(), image.buf.bytes), 0);
}

// After either clear, a thread's next call on a buffer it was served from
// is not served what it was: a new weight with the old bytes at both ends
// is packed again for a cached layout, and refused for an image, whose
// ends then hold it among other bytes, until a new weight there is
// rewritten again.
TEST_F(NativeWeightLifetime, ClearsEndWhatThreadsWereServed) {
    pin_native();
    const int K = 64, N = 64;
    for (bool all : {false, true}) {
        for (int32_t wc : {1, 2}) {
            SCOPED_TRACE(std::string(all ? "clear_matmul_weight_caches"
                                         : "clear_matmul_native_weight_caches")
                    + " wc=" + std::to_string(wc));
            ::reset_grp_matmul_caches();
            const weight_t w = make_weight(kBf16, K, N, N, false, 621 + wc);
            const weight_t w1 = clone(w);
            dense_t d(kBf16, K);
            for (int i = 0; i < 3; ++i) {
                ASSERT_EQ(d.run(w, wc), status_t::success);
                EXPECT_TRUE(d.matches(w1));
            }
            EXPECT_EQ(in_place(w), wc == 2);
            EXPECT_GE(nat::native_weight_memo_entries(w.ptr()), 1u);
            const weight_t served = clone(w);
            if (all)
                mm::clear_matmul_weight_caches();
            else
                mm::clear_matmul_native_weight_caches();
            EXPECT_EQ(nat::native_weight_memo_entries(w.ptr()), 0u);
            fill(w, 631 + wc);
            keep_ends(w, served.ptr());
            if (wc == 2) {
                // The image's bytes at both ends among other bytes.
                EXPECT_EQ(d.run(w, wc), status_t::failure);
                EXPECT_TRUE(d.untouched());
                fill(w, 641 + wc);
            }
            const weight_t w2 = clone(w);
            for (int i = 0; i < 2; ++i) {
                ASSERT_EQ(d.run(w, wc), status_t::success);
                EXPECT_TRUE(d.matches(w2)) << "call " << i;
            }
            EXPECT_EQ(nt::native_layouts(w.ptr()), 1u);
        }
    }
}

// A layout one thread finds stale and packs again is not served to another
// thread that was served the old layout, even when the buffer holds the old
// bytes again.
TEST_F(NativeWeightLifetime, StaleLayoutGoesForEveryThread) {
    const int K = 64, N = 64;
    for (data_type_t dt : {kBf16, kS8}) {
        SCOPED_TRACE(dt == kBf16 ? "bf16" : "s8");
        ::reset_grp_matmul_caches();
        const weight_t w = make_weight(dt, K, N, N, false, 641);
        const weight_t w1 = clone(w);
        std::atomic<int> step {0};
        int wrong = 0;
        size_t memo_before = 0, memo_after = 1;
        {
            joiner_t other;
            other.threads.emplace_back([&] {
                dense_t d(dt, K);
                for (int i = 0; i < 2; ++i)
                    if (d.run(w, 1) != status_t::success || !d.matches(w1))
                        ++wrong;
                memo_before = nat::native_weight_memo_entries(w.ptr());
                step.store(1);
                while (step.load() != 2)
                    std::this_thread::yield();
                memo_after = nat::native_weight_memo_entries(w.ptr());
                if (d.run(w, 1) != status_t::success || !d.matches(w1)) ++wrong;
            });
            while (step.load() != 1)
                std::this_thread::yield();
            fill(w, 642);
            const weight_t w2 = clone(w);
            dense_t d(dt, K);
            EXPECT_EQ(d.run(w, 1), status_t::success);
            EXPECT_TRUE(d.matches(w2)) << "packed again from the new bytes";
            std::memcpy(w.buf.get(), w1.ptr(), w.buf.bytes);
            step.store(2);
        }
        EXPECT_EQ(wrong, 0) << "the other thread was served a stale layout";
        EXPECT_GE(memo_before, 1u) << "the other thread kept no memo entry";
        EXPECT_EQ(memo_after, 0u)
                << "the other thread's entry outlived the stale layout";
        mm::clear_matmul_native_weight_caches();
    }
}

// One thread keeps giving a constant weight new bytes in its first
// kEndBytes, which the zero first source column multiplies, and runs a call
// after each change, which packs the weight again and drops its cached
// layout, while other threads keep running calls on the weight at weight
// cache mode 1.  No call reads a dropped layout after it is freed
// (AddressSanitizer reports such a read), and every call gets the bits of
// the first call.  The caches are cleared between rounds, when no call runs.
TEST_F(NativeWeightLifetime, DroppedLayoutsOutliveTheCallsReadingThem) {
    const int K = 512, N = 256;
    constexpr int kReaders = 4, kRounds = 4, kChanges = 48;
    for (data_type_t dt : {kBf16, kS8}) {
        SCOPED_TRACE(dt == kBf16 ? "bf16" : "s8");
        ::reset_grp_matmul_caches();
        const weight_t w = make_weight(dt, K, N, N, false, 671);
        const weight_t other = make_weight(dt, K, N, N, false, 672);
        uint8_t heads[2][kEndBytes];
        std::memcpy(heads[0], w.ptr(), kEndBytes);
        std::memcpy(heads[1], other.ptr(), kEndBytes);
        dense_t first(dt, K);
        for (int m = 0; m < dense_t::M; ++m)
            nt::store_f32(first.src.get(), first.src_dt(),
                    static_cast<size_t>(m) * K, 0.0f);
        ASSERT_EQ(first.run(w, 1), status_t::success);
        ASSERT_TRUE(first.matches(w));
        const std::vector<float> want = first.dst;

        for (int round = 0; round < kRounds; ++round) {
            SCOPED_TRACE("round " + std::to_string(round));
            std::atomic<bool> stop {false};
            std::atomic<int> started {0}, bad {0};
            const auto run_one = [&](dense_t &d) {
                if (d.run(w, 1) != status_t::success || d.dst != want)
                    bad.fetch_add(1);
            };
            {
                joiner_t readers;
                for (int r = 0; r < kReaders; ++r)
                    readers.threads.emplace_back([&] {
                        dense_t d = first;
                        run_one(d);
                        started.fetch_add(1);
                        while (!stop.load())
                            run_one(d);
                    });
                while (started.load() < kReaders)
                    std::this_thread::yield();
                dense_t d = first;
                for (int i = 1; i <= kChanges; ++i) {
                    std::memcpy(w.buf.get(), heads[i % 2], kEndBytes);
                    run_one(d);
                }
                stop.store(true);
            }
            std::memcpy(w.buf.get(), heads[0], kEndBytes);
            EXPECT_EQ(bad.load(), 0) << "calls with another result";
            mm::clear_matmul_native_weight_caches();
        }
    }
}

/// A thread that runs one call on a weight, then makes no call until
/// resume(): a layout dropped meanwhile may still be read by its call, as
/// far as the cache can tell.
class idle_caller_t {
public:
    idle_caller_t(const weight_t &w, data_type_t dt) {
        thread_ = std::thread([this, &w, dt] {
            dense_t d(dt, w.K);
            ok_ = d.run(w, 1) == status_t::success && d.matches(w);
            step_.store(1);
            while (step_.load() != 2)
                std::this_thread::yield();
            if (call_again_) ok_ = ok_ && d.run(w, 1) == status_t::success;
        });
        while (step_.load() != 1)
            std::this_thread::yield();
    }
    /// Lets the thread run one more call on its weight (again), or exit.
    void resume(bool again) {
        call_again_ = again;
        step_.store(2);
        thread_.join();
    }
    ~idle_caller_t() {
        if (thread_.joinable()) resume(false);
    }
    bool ok() const { return ok_; }

private:
    std::thread thread_;
    std::atomic<int> step_ {0};
    bool call_again_ = false;
    bool ok_ = false;
};

// The layouts a weight's new bytes drop stay held while a call may still
// read them, with the cached ones: at ZENDNNL_LRU_CACHE_CAPACITY 2, while
// another thread that read the first layout makes no call, the third layout
// of a weight whose bytes changed twice is packed for its call.
TEST_F(NativeWeightLifetime, DroppedLayoutsCountTowardTheCapacity) {
    const nt::lru_capacity_pin_t capacity(2);
    const int K = 64, N = 64;
    for (data_type_t dt : {kBf16, kS8}) {
        SCOPED_TRACE(dt == kBf16 ? "bf16" : "s8");
        ::reset_grp_matmul_caches();
        const weight_t w = make_weight(dt, K, N, N, false, 681);
        idle_caller_t idle(w, dt);
        EXPECT_TRUE(idle.ok());
        EXPECT_EQ(nt::native_layouts(w.ptr()), 1u);
        dense_t d(dt, K);
        fill(w, 682);
        ASSERT_EQ(d.run(w, 1), status_t::success);
        EXPECT_TRUE(d.matches(w));
        EXPECT_EQ(nt::native_layouts(w.ptr()), 1u)
                << "the new layout cached next to the dropped one";
        fill(w, 683);
        ASSERT_EQ(d.run(w, 1), status_t::success);
        EXPECT_TRUE(d.matches(w));
        EXPECT_EQ(nt::native_layouts(w.ptr()), 0u)
                << "two layouts held: packed for the call";
        idle.resume(false);
        mm::clear_matmul_native_weight_caches();
        ASSERT_EQ(d.run(w, 1), status_t::success);
        EXPECT_TRUE(d.matches(w));
        EXPECT_EQ(nt::native_layouts(w.ptr()), 1u) << "cached after the clear";
    }
}

// At ZENDNNL_LRU_CACHE_CAPACITY 1, a weight whose bytes changed while
// another thread that read its old layout makes no call has its new layout
// packed for each call; once that thread calls again or exits, the next
// call frees the dropped layout and caches the new one.  In a child process,
// whose only calling threads are the test's (as below).
TEST_F(NativeWeightLifetime,
        ACacheFullOfDroppedLayoutsCachesOnceTheyAreUnread) {
    nt::expect_in_child({}, [] {
        const nt::lru_capacity_pin_t capacity(1);
        const int K = 64, N = 64;
        int bad = 0;
        const auto expect = [&](bool ok, const char *what, bool exits) {
            if (ok) return;
            std::fprintf(stderr, "%s (%s)\n", what,
                    exits ? "the reader exits" : "the reader calls again");
            ++bad;
        };
        for (bool exits : {false, true}) {
            const weight_t w = make_weight(kBf16, K, N, N, false, 701);
            idle_caller_t idle(w, kBf16);
            dense_t d(kBf16, K);
            fill(w, 702);
            expect(d.run(w, 1) == status_t::success && d.matches(w),
                    "the call after the change", exits);
            expect(nt::native_layouts(w.ptr()) == 0
                            && nat::native_retired_layouts(w.ptr()) == 1,
                    "the dropped layout fills the cache", exits);
            idle.resume(!exits);
            expect(idle.ok(), "the reader's calls", exits);
            for (int i = 0; i < 2; ++i)
                expect(d.run(w, 1) == status_t::success && d.matches(w),
                        "a call once the reader moved on", exits);
            expect(nt::native_layouts(w.ptr()) == 1
                            && nat::native_retired_layouts(w.ptr()) == 0,
                    "the new layout cached", exits);
            mm::clear_matmul_native_weight_caches();
        }
        return bad;
    }, "a cache full of dropped layouts");
}

// The reader a thread leaves at its exit counts from then on the calls of
// the next thread that takes it: a layout dropped while the first thread
// could read it is freed once that thread exited, also when its reader went
// to a thread whose only call an image served at its start (no lookup
// counted it) and which then idles.  At ZENDNNL_LRU_CACHE_CAPACITY 1, the
// next call on the changed weight caches its layout.  In a child process,
// whose only calling threads are the test's.
TEST_F(NativeWeightLifetime, AReaderTakenByAnImageCallerLetsDroppedLayoutsGo) {
    nt::expect_in_child({}, [this] {
        pin_native();
        const nt::lru_capacity_pin_t capacity(1);
        const int K = 64, N = 64;
        int bad = 0;
        const auto expect = [&](bool ok, const char *what) {
            if (ok) return;
            std::fprintf(stderr, "%s\n", what);
            ++bad;
        };
        const weight_t image = make_weight(kBf16, K, N, N, false, 1221);
        const weight_t image_raw = clone(image);
        dense_t d(kBf16, K);
        expect(d.run(image, 2) == status_t::success && in_place(image),
                "the rewrite");
        const weight_t w = make_weight(kBf16, K, N, N, false, 1222);
        {
            idle_caller_t first(w, kBf16);
            expect(first.ok(), "the first thread's call");
            fill(w, 1223);
            expect(d.run(w, 1) == status_t::success && d.matches(w),
                    "the call after the change");
            expect(nt::native_layouts(w.ptr()) == 0
                            && nat::native_retired_layouts(w.ptr()) == 1,
                    "the dropped layout fills the cache");
        }
        std::atomic<int> step {0};
        bool served = false;
        std::thread second([&] {
            dense_t di(kBf16, K);
            served = di.run(image, 2) == status_t::success
                    && di.matches(image_raw);
            step.store(1);
            while (step.load() != 2)
                std::this_thread::yield();
        });
        while (step.load() != 1)
            std::this_thread::yield();
        expect(served, "the image's call on the second thread");
        expect(d.run(w, 1) == status_t::success && d.matches(w),
                "the call once the first thread exited");
        expect(nt::native_layouts(w.ptr()) == 1
                        && nat::native_retired_layouts(w.ptr()) == 0,
                "the dropped layout kept for the second thread");
        step.store(2);
        second.join();
        return bad;
    }, "a reader taken by an image caller");
}

// A thread that cannot list a reader obtains layouts that no drop can
// count, so no dropped layout is freed while it may read one; a clear, when
// no call runs, frees every layout and lets dropped layouts be freed again.
// In a child process, whose only calling threads are the test's.
TEST_F(NativeWeightLifetime, AClearLetsDroppedLayoutsGoAfterAnUncountedThread) {
    nt::expect_in_child({}, [] {
        const int K = 64, N = 64;
        int bad = 0;
        const auto expect = [&](bool ok, const char *what) {
            if (ok) return;
            std::fprintf(stderr, "%s\n", what);
            ++bad;
        };
        const weight_t v = make_weight(kBf16, K, N, N, false, 1501);
        nat::native_fail_reader_listings(true);
        std::thread([&] {
            dense_t d(kBf16, K);
            expect(d.run(v, 1) == status_t::success && d.matches(v),
                    "the call of the thread with no reader");
        }).join();
        nat::native_fail_reader_listings(false);
        const weight_t w = make_weight(kBf16, K, N, N, false, 1502);
        dense_t d(kBf16, K);
        for (uint32_t i = 0; i < 3; ++i) {
            fill(w, 1503 + i);
            expect(d.run(w, 1) == status_t::success && d.matches(w),
                    "a call before the clear");
        }
        expect(nat::native_retired_layouts(w.ptr()) == 2,
                "a dropped layout freed while an uncounted thread may read it");
        mm::clear_matmul_native_weight_caches();
        for (uint32_t i = 0; i < 4; ++i) {
            fill(w, 1511 + i);
            expect(d.run(w, 1) == status_t::success && d.matches(w),
                    "a call after the clear");
        }
        const size_t kept = nat::native_retired_layouts(w.ptr());
        if (kept > 1) {
            std::fprintf(
                    stderr, "%zu dropped layouts held after the clear\n", kept);
            ++bad;
        }
        return bad;
    }, "dropped layouts freed again after a clear");
}

// A layout dropped when its weight's bytes change is freed once no call can
// read it: with one calling thread at most one dropped layout is held; a
// thread that read an earlier layout and makes no call keeps the layouts
// dropped since, until it makes a call or exits, and they are freed by the
// next call that drops a layout, not by that thread's call, which is
// served the cached layout, nor by its exit.  In a child process, whose
// only calling threads are the test's: a thread that obtained layouts and
// stays idle, such as a pool thread that once made calls, would keep them.
TEST_F(NativeWeightLifetime, DroppedLayoutsAreFreedOnceNoCallReadsThem) {
    nt::expect_in_child({}, [] {
        const int K = 64, N = 64;
        int bad = 0;
        const auto expect = [&](bool ok, const char *what, uint32_t i) {
            if (ok) return;
            std::fprintf(stderr, "%s (%u)\n", what, i);
            ++bad;
        };
        for (bool exits : {false, true}) {
            const weight_t w = make_weight(kBf16, K, N, N, false, 691);
            dense_t d(kBf16, K);
            for (uint32_t i = 0; i < 4; ++i) {
                fill(w, 692 + i);
                expect(d.run(w, 1) == status_t::success && d.matches(w),
                        "a call after a change", i);
                expect(nat::native_retired_layouts(w.ptr()) <= 1,
                        "more than one dropped layout held", i);
            }
            idle_caller_t idle(w, kBf16);
            expect(idle.ok(), "the idle thread's call", 0);
            for (uint32_t i = 0; i < 3; ++i) {
                fill(w, 702 + i);
                expect(d.run(w, 1) == status_t::success && d.matches(w),
                        "a call while a thread idles", i);
            }
            expect(nat::native_retired_layouts(w.ptr()) >= 3,
                    "a layout freed while the idle thread may read it", 0);
            idle.resume(!exits);
            expect(idle.ok(), "the idle thread's second call", 0);
            expect(nat::native_retired_layouts(w.ptr()) >= 3,
                    "layouts freed by the idle thread's call or exit",
                    exits ? 1 : 0);
            fill(w, 712);
            expect(d.run(w, 1) == status_t::success && d.matches(w),
                    "the call after the idle thread's", 0);
            expect(nat::native_retired_layouts(w.ptr()) <= 1,
                    "layouts kept after the idle thread's call or exit",
                    exits ? 1 : 0);
            mm::clear_matmul_native_weight_caches();
        }
        return bad;
    }, "dropped layouts freed once unread");
}

#if defined(__linux__)
/// Resident bytes of the process.
size_t resident_bytes() {
    long pages = 0, resident = 0;
    FILE *f = std::fopen("/proc/self/statm", "r");
    if (f == nullptr) return 0;
    if (std::fscanf(f, "%ld %ld", &pages, &resident) != 2) resident = 0;
    std::fclose(f);
    return static_cast<size_t>(resident) * sysconf(_SC_PAGESIZE);
}
#endif

// A constant weight whose bytes a framework changes in place at weight
// cache mode 1, 16 times, with a call after each: the process keeps at most
// one layout of the weight besides the current one, not one per change.  In
// a child process, as above.
TEST_F(NativeWeightLifetime, EachChangeOfAWeightKeepsNoMoreThanOneOldLayout) {
#if !defined(__linux__)
    GTEST_SKIP() << "reads /proc/self/statm";
#else
    nt::expect_in_child({}, [] {
        const int K = 2048, N = 2048;
        const weight_t w = make_weight(kBf16, K, N, N, false, 721);
        dense_t d(kBf16, K);
        if (d.run(w, 1) != status_t::success) return 1;
        const size_t before = resident_bytes();
        for (int i = 1; i <= 16; ++i) {
            for (int j = 0; j < 32; ++j)
                nt::store_f32(w.buf.get(), kBf16, static_cast<size_t>(j),
                        0.001f * static_cast<float>(i + j));
            if (d.run(w, 1) != status_t::success) return 1;
        }
        const double grown_mib
                = (static_cast<double>(resident_bytes()) - before) / (1 << 20);
        const size_t kept = nat::native_retired_layouts(w.ptr());
        std::fprintf(stderr,
                "16 changes of an 8 MiB weight: %.1f MiB more resident, %zu "
                "dropped layouts held\n",
                grown_mib, kept);
        return kept <= 1 && (kSanitizedHeap || grown_mib < 32.0) ? 0 : 1;
    }, "16 changes of an 8 MiB weight");
#endif
}

// A thread whose calls are all packed for the call (weight cache mode 0 on
// a raw weight) keeps no memo entry.
TEST_F(NativeWeightLifetime, CallsPackedForTheCallKeepNoMemo) {
    const int K = 64, N = 64;
    for (data_type_t dt : {kBf16, kS8}) {
        SCOPED_TRACE(dt == kBf16 ? "bf16" : "s8");
        const weight_t w = make_weight(dt, K, N, N, false, 661);
        const weight_t w1 = clone(w);
        int wrong = 0;
        size_t memo = 1;
        {
            joiner_t caller;
            caller.threads.emplace_back([&] {
                dense_t d(dt, K);
                for (int i = 0; i < 3; ++i)
                    if (d.run(w, 0) != status_t::success || !d.matches(w1))
                        ++wrong;
                memo = nat::native_weight_memo_entries(w.ptr());
            });
        }
        EXPECT_EQ(wrong, 0);
        EXPECT_EQ(memo, 0u);
        EXPECT_EQ(nt::native_layouts(w.ptr()), 0u);
    }
}

// Threads that were served layouts and exited leave nothing a later thread
// or a clear meets: the next threads' calls, before and after a clear and
// new weights (one with the old ends of a cached layout), are served right.
TEST_F(NativeWeightLifetime, ThreadsThatExitedLeaveNothingBehind) {
    pin_native();
    const int K = 64, N = 64;
    const weight_t a = make_weight(kBf16, K, N, N, false, 651);
    const weight_t b = make_weight(kS8, K, N, N, false, 652);
    const weight_t a1 = clone(a), b1 = clone(b);
    const auto served_right = [&](const weight_t &wa, const weight_t &wb) {
        int wrong = 0;
        for (int round = 0; round < 4; ++round) {
            joiner_t t;
            t.threads.emplace_back([&] {
                dense_t da(kBf16, K), db(kS8, K);
                for (int i = 0; i < 2; ++i) {
                    if (da.run(a, 2) != status_t::success || !da.matches(wa))
                        ++wrong;
                    if (db.run(b, 1) != status_t::success || !db.matches(wb))
                        ++wrong;
                }
            });
        }
        return wrong;
    };
    EXPECT_EQ(served_right(a1, b1), 0);
    const weight_t image = clone(a);
    mm::clear_matmul_native_weight_caches();
    fill(a, 653);
    EXPECT_NE(std::memcmp(a.ptr(), image.ptr(), kEndBytes), 0);
    fill(b, 654);
    keep_ends(b, b1.ptr());
    const weight_t a2 = clone(a), b2 = clone(b);
    EXPECT_EQ(served_right(a2, b2), 0);
    dense_t d(kBf16, K);
    ASSERT_EQ(d.run(a, 2), status_t::success);
    EXPECT_TRUE(d.matches(a2));
}

/// A bf16 [K, 64] weight at p, in memory it does not own, drawn from seed.
weight_t weight_at(uint8_t *p, int K, uint32_t seed) {
    weight_t w;
    w.dt = kBf16;
    w.N = 64;
    w.ldb = 64;
    w.K = K;
    w.buf.mem = std::shared_ptr<uint8_t>(p, [](uint8_t *) {});
    w.buf.bytes = static_cast<size_t>(K) * 64 * sizeof(uint16_t);
    fill(w, seed);
    return w;
}

/// Rewrites w in place (weight cache mode 2) and checks the call.
void rewrite_in_place(const weight_t &w) {
    const weight_t raw = clone(w);
    dense_t d(kBf16, w.K);
    ASSERT_EQ(d.run(w, 2), status_t::success);
    EXPECT_TRUE(d.matches(raw));
    ASSERT_TRUE(in_place(w));
}

/// Runs a weight cache mode wc call on w, whose raw weight is raw.
void expect_served(const weight_t &w, const weight_t &raw, int32_t wc) {
    dense_t d(kBf16, w.K);
    ASSERT_EQ(d.run(w, wc), status_t::success);
    EXPECT_TRUE(d.matches(raw));
}

#ifndef _WIN32
/// Pages whose access can be turned off, as the memory of a freed buffer may
/// be unmapped.
class pages_t {
public:
    explicit pages_t(size_t count) : bytes_(count * page_bytes()) {
        void *p = mmap(nullptr, bytes_, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        base_ = p == MAP_FAILED ? nullptr : static_cast<uint8_t *>(p);
    }
    ~pages_t() {
        mm::clear_matmul_native_weight_caches();
        if (base_ != nullptr) munmap(base_, bytes_);
    }
    pages_t(const pages_t &) = delete;
    pages_t &operator=(const pages_t &) = delete;
    static size_t page_bytes() {
        return static_cast<size_t>(sysconf(_SC_PAGESIZE));
    }
    bool ok() const { return base_ != nullptr; }
    uint8_t *at(size_t page) const { return base_ + page * page_bytes(); }
    bool no_access(size_t page) const {
        return mprotect(at(page), page_bytes(), PROT_NONE) == 0;
    }

private:
    size_t bytes_;
    uint8_t *base_ = nullptr;
};

/// A bf16 [K, 64] weight filling pages of memory at p, drawn from seed.
weight_t weight_in_pages(uint8_t *p, size_t pages, uint32_t seed) {
    return weight_at(p,
            static_cast<int>(
                    pages * pages_t::page_bytes() / (64 * sizeof(uint16_t))),
            seed);
}
#endif

// The record of an image whose memory now holds other weights is checked
// only within the buffer of the call on it, while the rest of the image is
// no longer readable: a smaller weight at its start, a weight that starts
// inside it and one that starts below it are read as the raw weights they
// are, and a weight inside it that sees neither of its ends is refused as
// part of the image.
TEST_F(NativeWeightLifetime, StaleImageIsCheckedWithinTheCallersBuffer) {
#ifdef _WIN32
    GTEST_SKIP() << "needs mprotect";
#else
    pin_native();
    {
        SCOPED_TRACE("a smaller weight at the start of the image");
        pages_t pages(2);
        ASSERT_TRUE(pages.ok());
        ASSERT_NO_FATAL_FAILURE(
                rewrite_in_place(weight_in_pages(pages.at(0), 2, 701)));
        const weight_t b = weight_in_pages(pages.at(0), 1, 702);
        const weight_t raw = clone(b);
        ASSERT_TRUE(pages.no_access(1));
        ASSERT_NO_FATAL_FAILURE(expect_served(b, raw, 2));
    }
    {
        SCOPED_TRACE("a weight that starts inside the image");
        pages_t pages(3);
        ASSERT_TRUE(pages.ok());
        ASSERT_NO_FATAL_FAILURE(
                rewrite_in_place(weight_in_pages(pages.at(0), 2, 703)));
        const weight_t c = weight_in_pages(pages.at(1), 2, 704);
        const weight_t raw = clone(c);
        ASSERT_TRUE(pages.no_access(0));
        ASSERT_NO_FATAL_FAILURE(expect_served(c, raw, 2));
    }
    {
        SCOPED_TRACE("a weight that starts below the image");
        pages_t pages(3);
        ASSERT_TRUE(pages.ok());
        ASSERT_NO_FATAL_FAILURE(
                rewrite_in_place(weight_in_pages(pages.at(1), 2, 705)));
        const weight_t d = weight_in_pages(pages.at(0), 2, 706);
        const weight_t raw = clone(d);
        ASSERT_TRUE(pages.no_access(2));
        ASSERT_NO_FATAL_FAILURE(expect_served(d, raw, 2));
    }
    {
        SCOPED_TRACE("a weight inside the image that sees neither end");
        pages_t pages(3);
        ASSERT_TRUE(pages.ok());
        ASSERT_NO_FATAL_FAILURE(
                rewrite_in_place(weight_in_pages(pages.at(0), 3, 707)));
        ASSERT_TRUE(pages.no_access(0));
        ASSERT_TRUE(pages.no_access(2));
        EXPECT_TRUE(nat::native_weight_packed_in_place(
                pages.at(1), pages_t::page_bytes()));
    }
#endif
}

// A weight that lies inside a freed in-place image of over 64 KiB, whose
// record keeps the digests of sample lines between its two ends, and sees
// neither end is checked at the sample lines in its buffer: while those
// hold the image's bytes it is refused; with other bytes there it is read
// as the raw weight it is, at every weight cache mode, as a non-constant
// weight and through a call that names another algorithm, whether the
// image's record is live or a clear made it a tombstone.  The image fills
// pages of the system's size, its ends on the first and the last, which
// are made unreadable where pages can be protected, as the memory of a
// freed buffer may be unmapped.
TEST_F(NativeWeightLifetime, WeightInsideAFreedImageIsCheckedAtItsSamples) {
    pin_native();
#ifdef _WIN32
    const size_t page = 4096;
#else
    const size_t page = pages_t::page_bytes();
#endif
    // Past the 64 KiB a record keeps whole, with a page for each end.
    const size_t count = 64 * 1024 / page + 3;
    const int rows = static_cast<int>(page / (64 * sizeof(uint16_t)));
    for (bool cleared : {false, true}) {
        SCOPED_TRACE(cleared ? "a tombstone" : "an image");
#ifdef _WIN32
        const nt::buffer_t mem = nt::make_buffer(count * page);
        uint8_t *const base = mem.as<uint8_t>();
#else
        pages_t pages(count);
        ASSERT_TRUE(pages.ok());
        uint8_t *const base = pages.at(0);
#endif
        ASSERT_NO_FATAL_FAILURE(rewrite_in_place(
                weight_at(base, static_cast<int>(count) * rows, 1021)));
        if (cleared) mm::clear_matmul_native_weight_caches();
#ifndef _WIN32
        ASSERT_TRUE(pages.no_access(0));
        ASSERT_TRUE(pages.no_access(count - 1));
#endif
        uint8_t *const at = base + page;
        const size_t bytes = (count - 2) * page;
        nat::native_image_check_t found;
        EXPECT_TRUE(nat::native_weight_find_image(at, bytes, found))
                << "the image's sample lines";
        EXPECT_TRUE(found.interior);
        EXPECT_FALSE(found.holds_no_checked_byte)
                << "no sample line in the buffer";
        const weight_t inner
                = weight_at(at, static_cast<int>(count - 2) * rows, 1022);
        const weight_t raw = clone(inner);
        dense_t d(kBf16, inner.K);
        if (cleared) {
            ASSERT_EQ(d.run(inner, 1, 0, matmul_algo_t::aocl_dlp_blocked),
                    status_t::success);
            EXPECT_TRUE(d.matches(raw))
                    << "a call that names another algorithm";
        } else {
            ASSERT_EQ(d.run(inner, 0, 0, matmul_algo_t::native_brgemm, false),
                    status_t::success);
            EXPECT_TRUE(d.matches(raw)) << "a non-constant weight";
        }
        for (int32_t wc : {2, 1, 0})
            ASSERT_NO_FATAL_FAILURE(expect_served(inner, raw, wc));
    }
}

// A weight inside an image that holds none of the image's checked bytes
// (it lies between its head end and its second sample line) cannot be told
// from the image.  A new weight there is refused inside a live image and
// inside a tombstone, whose refusal says that a second clear helps if the
// image's buffer was freed; once a clear finds the record a tombstone
// already, it is read as the raw weight it is, and the tombstone is
// forgotten over its buffer only: the part before it stays.  A view of the
// image there whose bytes still hold it, after a clear with no reload, is
// refused, never success with the image read as a raw weight; once the
// whole raw weight is back and called, the view is served.  The image is
// of 4 MiB, with a sample line about every 4 KiB, in a buffer of the test.
TEST_F(NativeWeightLifetime, WeightThatHoldsNoCheckedByteOfARecord) {
    pin_native();
    const size_t bytes = 4 << 20;
    const int K = static_cast<int>(bytes / (64 * sizeof(uint16_t)));
    for (int clears = 0; clears <= 2; ++clears) {
        SCOPED_TRACE(std::to_string(clears) + " clears");
        const nt::buffer_t mem = nt::make_buffer(bytes);
        uint8_t *const base = mem.as<uint8_t>();
        ASSERT_NO_FATAL_FAILURE(rewrite_in_place(weight_at(base, K, 1031)));
        for (int i = 0; i < clears; ++i)
            mm::clear_matmul_native_weight_caches();
        const weight_t between = weight_at(base + 1024, 8, 1032);
        const weight_t raw = clone(between);
        dense_t d(kBf16, between.K);
        if (clears == 2) {
            ASSERT_NO_FATAL_FAILURE(expect_served(between, raw, 1));
            EXPECT_EQ(nat::native_weight_tombstones(base), 1u);
        } else {
            EXPECT_EQ(d.run(between, 1), status_t::failure);
            EXPECT_TRUE(d.untouched());
            nat::native_image_check_t found;
            ASSERT_TRUE(nat::native_weight_find_image(
                    between.ptr(), between.buf.bytes, found));
            EXPECT_TRUE(found.holds_no_checked_byte);
            if (clears == 1) {
                EXPECT_NE(std::string(nat::native_image_refusal(found))
                                  .find("clear the weight caches once more"),
                        std::string::npos);
            }
        }
        mm::clear_matmul_native_weight_caches();
    }
    {
        SCOPED_TRACE("a view of the image, after a clear");
        const nt::buffer_t mem = nt::make_buffer(bytes);
        uint8_t *const base = mem.as<uint8_t>();
        const weight_t whole = weight_at(base, K, 1033);
        const weight_t raw = clone(whole);
        ASSERT_NO_FATAL_FAILURE(rewrite_in_place(whole));
        mm::clear_matmul_native_weight_caches();
        const weight_t view = part_of(whole, 1024, 8, 64, 64);
        dense_t d(kBf16, view.K);
        EXPECT_EQ(d.run(view, 1), status_t::failure)
                << "the image read as a raw weight";
        EXPECT_TRUE(d.untouched());
        std::memcpy(whole.buf.get(), raw.ptr(), bytes);
        ASSERT_NO_FATAL_FAILURE(expect_served(whole, raw, 1));
        ASSERT_NO_FATAL_FAILURE(
                expect_served(view, part_of(raw, 1024, 8, 64, 64), 1));
        mm::clear_matmul_native_weight_caches();
    }
}

// A weight at the start of the part of a tombstone that a split left (the
// raw first 33 rows of a cleared 4 MiB bf16 image written back and called),
// which holds none of the record's checked bytes, is refused with the line
// that a second clear helps if the image's buffer was freed; after that
// clear it is read as the raw weight it is.
TEST_F(NativeWeightLifetime, AWeightAtAPartOfATombstoneNamesTheSecondClear) {
    pin_native();
    const size_t bytes = 4 << 20;
    const int K = static_cast<int>(bytes / (64 * sizeof(uint16_t)));
    const nt::buffer_t mem = nt::make_buffer(bytes);
    uint8_t *const base = mem.as<uint8_t>();
    const weight_t whole = weight_at(base, K, 1281);
    const weight_t raw = clone(whole);
    ASSERT_NO_FATAL_FAILURE(rewrite_in_place(whole));
    mm::clear_matmul_native_weight_caches();
    // The second sample line ends at byte 4224, the third starts at 8256.
    const weight_t top = part_of(whole, 0, 33, 64, 64);
    std::memcpy(base, raw.ptr(), top.buf.bytes);
    ASSERT_NO_FATAL_FAILURE(expect_served(top, part_of(raw, 0, 33, 64, 64), 1));
    const weight_t fresh = weight_at(base + top.buf.bytes, 8, 1282);
    const weight_t fresh_raw = clone(fresh);
    nat::native_image_check_t found;
    ASSERT_TRUE(
            nat::native_weight_find_image(fresh.ptr(), fresh.buf.bytes, found));
    EXPECT_TRUE(found.tomb);
    EXPECT_TRUE(found.holds_no_checked_byte);
    const std::string advice = nat::native_image_refusal(found);
    EXPECT_NE(advice.find("clear the weight caches once more"),
            std::string::npos);
    EXPECT_NE(advice.find("reload every other weight a mode 2 call rewrote"),
            std::string::npos);
    dense_t d(kBf16, fresh.K);
    EXPECT_EQ(d.run(fresh, 1), status_t::failure);
    EXPECT_TRUE(d.untouched());
    mm::clear_matmul_native_weight_caches();
    ASSERT_NO_FATAL_FAILURE(expect_served(fresh, fresh_raw, 1));
    mm::clear_matmul_native_weight_caches();
}

// A view inside an image whose layout its thread was served, carried across
// the rewrite, is served it again from the thread's memo on the digests of
// its ends, as any cached layout, without reading the image's bytes in its
// buffer: with a page in the middle of the view made unreadable, its calls
// at weight cache modes 1 and 2 are served with its raw weight's results.
// In a child process, as a read of that page faults.
TEST_F(NativeWeightLifetime, AViewServedFromTheMemoReadsOnlyItsEnds) {
#ifdef _WIN32
    GTEST_SKIP() << "needs mprotect";
#else
    nt::expect_in_child({}, [this] {
        pin_native();
        // An image of 64 KiB, kept whole by its record, and a view of its
        // last half.
        const size_t page = pages_t::page_bytes();
        const size_t bytes = 64 * 1024;
        if (bytes % page != 0 || bytes / page < 8) return 0;
        pages_t pages(bytes / page);
        if (!pages.ok()) return 1;
        const weight_t whole = weight_at(pages.at(0), 512, 1201);
        weight_t view = part_of(whole, bytes / 2, 256, 64, 64);
        const weight_t raw = clone(view);
        dense_t dv(kBf16, view.K), dw(kBf16, whole.K);
        int bad = 0;
        const auto served = [&](int32_t wc, const char *what) {
            if (dv.run(view, wc) == status_t::success && dv.matches(raw))
                return;
            std::fprintf(stderr, "%s, weight cache mode %d\n", what, wc);
            ++bad;
        };
        served(1, "before the rewrite");
        if (dw.run(whole, 2) != status_t::success || !in_place(whole)) return 1;
        served(1, "after the rewrite");
        if (!pages.no_access(bytes / page * 3 / 4)) return 1;
        for (const int32_t wc : {1, 2, 1})
            served(wc, "with a page of the view unreadable");
        return bad;
    }, "a view served from the memo");
#endif
}

// A worker forked after the caches filled, while no call ran, is served
// right from what it inherits: an image written in place at weight cache
// mode 2 and a layout cached at mode 1, also after a clear and a pack again
// in the child.  A thread of the parent that obtained a layout and idles
// across the fork is gone in the child and keeps none of the layouts the
// child drops: as the child refreshes a weight four times, it holds at most
// one dropped layout.  The child runs its calls on one thread.
TEST_F(NativeWeightLifetime, AForkedWorkerIsServedFromWhatItInherits) {
#ifdef _WIN32
    GTEST_SKIP() << "needs fork";
#else
    pin_native();
    const int K = 128, N = 256;
    const weight_t image = make_weight(kBf16, K, N, N, false, 1311);
    const weight_t image_raw = clone(image);
    const weight_t layout = make_weight(kS8, K, N, N, false, 1312);
    dense_t di(kBf16, K), dl(kS8, K);
    ASSERT_EQ(di.run(image, 2), status_t::success);
    ASSERT_TRUE(in_place(image));
    ASSERT_EQ(dl.run(layout, 1), status_t::success);
    ASSERT_EQ(nt::native_layouts(layout.ptr()), 1u);
    const weight_t other = make_weight(kBf16, K, N, N, false, 1313);
    idle_caller_t idle(other, kBf16);
    ASSERT_TRUE(idle.ok());
    std::fflush(stdout);
    std::fflush(stderr);
    const pid_t pid = fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        alarm(60);
        const auto one_thread = [](dense_t &d, const weight_t &w, int32_t wc) {
            nt::call_t c;
            c.M = dense_t::M;
            c.N = w.N;
            c.K = w.K;
            c.src = d.src.get();
            c.wei = w.ptr();
            d.dst.assign(static_cast<size_t>(dense_t::M) * w.N, -1.0f);
            c.dst = d.dst.data();
            c.params.dtypes.src = d.src_dt();
            c.params.dtypes.wei = d.wei_dt;
            c.params.dtypes.dst = data_type_t::f32;
            c.params.num_threads = 1;
            c.params.weight_cache_type = wc;
            c.params.lowoha_algo = matmul_algo_t::native_brgemm;
            return nt::run_matmul(c) == status_t::success;
        };
        int bad = 0;
        for (int i = 0; i < 2; ++i) {
            if (!one_thread(di, image, 2) || !di.matches(image_raw)) bad |= 1;
            if (!one_thread(dl, layout, 1) || !dl.matches(layout)) bad |= 2;
        }
        for (uint32_t i = 0; i < 4; ++i) {
            fill(layout, 1321 + i);
            if (!one_thread(dl, layout, 1) || !dl.matches(layout)) bad |= 8;
        }
        const size_t kept = nat::native_retired_layouts(layout.ptr());
        if (kept > 1) {
            std::fprintf(stderr, "the child holds %zu dropped layouts\n", kept);
            bad |= 16;
        }
        mm::clear_matmul_native_weight_caches();
        if (!one_thread(dl, layout, 1) || !dl.matches(layout)) bad |= 4;
        _exit(bad);
    }
    idle.resume(false);
    int status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0)
            << (WIFEXITED(status) ? "exit code " : "signal ")
            << (WIFEXITED(status) ? WEXITSTATUS(status) : WTERMSIG(status));
#endif
}

// An in-place rewrite does not read the source of a layout cached from
// bytes that reach past the rewritten buffer.
TEST_F(NativeWeightLifetime, RewriteDoesNotReadLayoutsPastItsBuffer) {
#ifdef _WIN32
    GTEST_SKIP() << "needs mprotect";
#else
    pin_native();
    pages_t pages(3);
    ASSERT_TRUE(pages.ok());
    const weight_t e = weight_in_pages(pages.at(1), 2, 711);
    ASSERT_NO_FATAL_FAILURE(expect_served(e, clone(e), 1));
    const weight_t f = weight_in_pages(pages.at(0), 2, 712);
    const weight_t raw = clone(f);
    ASSERT_TRUE(pages.no_access(2));
    ASSERT_NO_FATAL_FAILURE(expect_served(f, raw, 2));
    EXPECT_TRUE(in_place(f));
#endif
}

// ── A randomized sweep of write-backs, clears, views and new weights ─────

/// A weight in the arena of a sweep: [K, N] row-major, or [N, K] when trans,
/// of dt, rows ldb elements apart, offset bytes in; k0 and n0 are where it
/// starts in the arena's weight, for the scales of an s4 view.
struct sw_weight_t {
    data_type_t dt = kBf16;
    int K = 0, N = 0, ldb = 0;
    bool trans = false;
    size_t offset = 0;
    int k0 = 0, n0 = 0;

    int rows() const { return trans ? N : K; }
    int cols() const { return trans ? K : N; }
    size_t bytes_of(size_t elements) const {
        return dt == kS4 ? (elements + 1) / 2 : elements * nt::dt_bytes(dt);
    }
    size_t span() const {
        return bytes_of(static_cast<size_t>(rows() - 1) * ldb + cols());
    }
    /// Element (k, n) of the weight in the arena bytes at p.
    float at(const uint8_t *p, int k, int n) const {
        const size_t i = trans ? static_cast<size_t>(n) * ldb + k
                               : static_cast<size_t>(k) * ldb + n;
        const uint8_t *b = p + offset;
        if (dt != kS4) return nt::load_f32(b, dt, i);
        const int v = (b[i / 2] >> (4 * (i % 2))) & 15;
        return static_cast<float>(v >= 8 ? v - 16 : v);
    }
    bool same_request(const sw_weight_t &o) const {
        return dt == o.dt && K == o.K && N == o.N && ldb == o.ldb
                && trans == o.trans && offset == o.offset;
    }
};

/// One sweep: a weight of dt rewritten in place at weight cache mode 2 in an
/// arena of its bytes, then random steps (clears, the raw weight written
/// back over all of it, rows, columns or one row, new weights placed in its
/// memory, calls on it, its views and the new weights at modes 0 to 2,
/// with no capacity, the exact one or the rest of the arena).  After every
/// call: it failed with dst untouched, or it gave the output of a weight it
/// may be served, within the band of its rounding: the raw bytes it reads
/// when none of them is an image byte that differs from the byte it
/// replaced; the weight an image holds (the raw bytes it replaced, where
/// the call reads image bytes), as an image at its start, a layout carried
/// across its rewrite or rebuilt from it serve; or a layout cached for the
/// same request since the last clear, whose ends still hold the bytes it
/// was packed from.  Not checked: a call at the start of an image while no
/// clear came since its rewrite and both lines its served calls check hold
/// its bytes, which is served the image, or a layout rebuilt from it,
/// whatever its other bytes hold (two lines are what such a call checks);
/// and a call whose image bytes that differ from the bytes they replaced
/// are all zero (zero words do not tell an image from new zeros).  After the
/// whole raw weight is written back, a call on it is served; so is a new
/// weight after at most two clears.
class weight_sweep_t {
public:
    weight_sweep_t(data_type_t dt, int K, int N, bool trans, uint32_t seed)
        : rng_(seed) {
        w_.dt = dt;
        w_.K = K;
        w_.N = N;
        w_.trans = trans;
        w_.ldb = trans ? K : N;
        const size_t bytes = w_.span();
        arena_ = nt::make_buffer(bytes);
        std::uniform_real_distribution<float> real(-1.0f, 1.0f);
        std::uniform_int_distribution<int> byte(0, 255);
        for (size_t i = 0; i < bytes; ++i)
            arena_.as<uint8_t>()[i] = static_cast<uint8_t>(byte(rng_));
        if (dt != kS4)
            for (size_t i = 0; i < bytes / nt::dt_bytes(dt); ++i)
                nt::store_f32(arena_.get(), dt, i, real(rng_));
        raw_.assign(arena_.as<uint8_t>(), arena_.as<uint8_t>() + bytes);
        tag_.assign(bytes, -1);
        rimg_.assign(bytes, 0);
        const int kmax = std::max(K, 64);
        src_ = nt::make_buffer(static_cast<size_t>(kM) * kmax * 2);
        for (size_t i = 0; i < static_cast<size_t>(kM) * kmax; ++i)
            nt::store_f32(src_.get(), dt == data_type_t::f16 ? dt : kBf16, i,
                    real(rng_));
        if (dt == kS4) {
            std::uniform_real_distribution<float> scale(0.002f, 0.01f);
            scales_.resize(static_cast<size_t>(K / 128) * N);
            for (float &s : scales_)
                s = bfloat16_round(scale(rng_));
        }
    }

    /// Runs steps; the violations found, with what each was.
    int run(int steps, std::string &first) {
        const auto record = [&](const std::string &what) {
            if (violations_++ == 0) first = what;
        };
        sw_weight_t whole = w_;
        if (!call(whole, 2, 0, record) || tag_[0] < 0) {
            record("the first call did not rewrite the weight in place");
            return violations_;
        }
        for (int step = 0; step < steps && violations_ < 8; ++step) {
            const int pick = uniform(0, 99);
            if (pick < 6) {
                clear();
            } else if (pick < 28) {
                write_back(record);
            } else if (pick < 34) {
                place_new(record);
            } else {
                call(random_target(), uniform(0, 2), -1, record);
            }
        }
        return violations_;
    }

    long calls() const { return calls_; }
    long refused() const { return refused_; }
    long unchecked() const { return unchecked_; }

private:
    static constexpr int kM = 2;

    static float bfloat16_round(float v) {
        uint16_t h = 0;
        nt::store_f32(&h, kBf16, 0, v);
        return nt::load_f32(&h, kBf16, 0);
    }
    int uniform(int lo, int hi) {
        return std::uniform_int_distribution<int>(lo, hi)(rng_);
    }
    size_t arena_bytes() const { return raw_.size(); }

    void clear() {
        mm::clear_matmul_native_weight_caches();
        ++clears_;
        served_.clear();
    }

    /// The byte ranges [first, second) of the arena the rows of v read.
    std::vector<std::pair<size_t, size_t>> rows_of(const sw_weight_t &v) const {
        std::vector<std::pair<size_t, size_t>> r;
        for (int i = 0; i < v.rows(); ++i) {
            const size_t e = static_cast<size_t>(i) * v.ldb;
            r.emplace_back(v.offset + v.bytes_of(e),
                    v.offset + v.bytes_of(e + v.cols()));
        }
        return r;
    }

    void write_back(const std::function<void(const std::string &)> &record) {
        // On 8-byte boundaries of the arena: columns in steps of 8 bytes.
        const int cstep = w_.dt == kS4 ? 16 : 4;
        int r0 = 0, r1 = w_.rows(), c0 = 0, c1 = w_.cols();
        const int kind = uniform(0, 99);
        if (kind < 15) {
        } else if (kind < 50) {
            r0 = uniform(0, w_.rows() - 1);
            r1 = uniform(r0 + 1, w_.rows());
        } else if (kind < 85) {
            c0 = cstep * uniform(0, w_.cols() / cstep - 1);
            c1 = cstep * uniform(c0 / cstep + 1, w_.cols() / cstep);
        } else {
            r0 = uniform(0, w_.rows() - 1);
            r1 = r0 + 1;
        }
        for (int r = r0; r < r1; ++r) {
            const size_t e = static_cast<size_t>(r) * w_.ldb;
            const size_t a = w_.bytes_of(e + c0), b = w_.bytes_of(e + c1);
            std::memcpy(arena_.as<uint8_t>() + a, raw_.data() + a, b - a);
            std::fill(tag_.begin() + a, tag_.begin() + b, -1);
        }
        if (kind < 15) {
            // The whole raw weight back: a call on it is served.
            sw_weight_t whole = w_;
            if (!call(whole, uniform(1, 2), 0, record))
                record("the whole raw weight written back was refused");
        }
    }

    void place_new(const std::function<void(const std::string &)> &record) {
        const size_t lines = arena_bytes() / 128;
        if (lines == 0) return;
        const size_t at = 128
                * static_cast<size_t>(uniform(0, static_cast<int>(lines) - 1));
        const int K = uniform(
                1, static_cast<int>(std::min<size_t>(16, lines - at / 128)));
        sw_weight_t f;
        f.dt = kBf16;
        f.K = K;
        f.N = 64;
        f.ldb = 64;
        f.offset = at;
        std::uniform_real_distribution<float> real(-1.0f, 1.0f);
        for (size_t i = 0; i < static_cast<size_t>(K) * 64; ++i)
            nt::store_f32(arena_.as<uint8_t>() + at, kBf16, i, real(rng_));
        std::fill(tag_.begin() + at, tag_.begin() + at + f.span(), -1);
        fresh_.push_back(f);
        if (fresh_.size() > 3) fresh_.erase(fresh_.begin());
        // A new weight is served after at most two clears.
        for (int clears = 0; clears <= 2; ++clears) {
            if (call(f, 1, 0, record)) return;
            clear();
        }
        record("a new weight placed in the arena was refused after two "
               "clears");
    }

    sw_weight_t random_target() {
        const int pick = uniform(0, 99);
        if (pick < 20 && !fresh_.empty()) {
            sw_weight_t f = fresh_[static_cast<size_t>(
                    uniform(0, static_cast<int>(fresh_.size()) - 1))];
            const int r0 = uniform(0, f.K - 1);
            const int r1 = uniform(r0 + 1, f.K);
            f.offset += static_cast<size_t>(r0) * 128;
            f.K = r1 - r0;
            return f;
        }
        sw_weight_t v = w_;
        if (pick < 45) return v;
        // Rows and columns of the weight: s4 columns (K) by groups of 128.
        const int cstep = w_.dt == kS4 ? 128 : 1;
        const int r0 = uniform(0, w_.rows() - 1);
        const int r1 = uniform(r0 + 1, w_.rows());
        int c0 = 0, c1 = w_.cols();
        if (uniform(0, 1) == 0) {
            c0 = cstep * uniform(0, w_.cols() / cstep - 1);
            c1 = cstep * uniform(c0 / cstep + 1, w_.cols() / cstep);
        }
        v.offset = w_.bytes_of(static_cast<size_t>(r0) * w_.ldb + c0);
        if (w_.trans) {
            v.N = r1 - r0;
            v.K = c1 - c0;
            v.n0 = r0;
            v.k0 = c0;
        } else {
            v.K = r1 - r0;
            v.N = c1 - c0;
            v.k0 = r0;
            v.n0 = c0;
        }
        return v;
    }

    /// The output x * v for the elements of v in the bytes at p, as the
    /// call computes it (an s4 weight times the source quantized per row).
    std::vector<double> model(const sw_weight_t &v, const uint8_t *p) const {
        std::vector<double> out(static_cast<size_t>(kM) * v.N, 0.0);
        const data_type_t sdt = v.dt == data_type_t::f16 ? v.dt : kBf16;
        for (int m = 0; m < kM; ++m) {
            std::vector<double> x(static_cast<size_t>(v.K));
            double amax = 0;
            for (int k = 0; k < v.K; ++k) {
                x[static_cast<size_t>(k)] = nt::load_f32(
                        src_.get(), sdt, static_cast<size_t>(m) * v.K + k);
                amax = std::max(amax, std::fabs(x[static_cast<size_t>(k)]));
            }
            const double s = amax / 127.0;
            for (int n = 0; n < v.N; ++n) {
                double acc = 0;
                for (int k = 0; k < v.K; ++k) {
                    const double wk = v.at(p, k, n);
                    if (v.dt != kS4) {
                        acc += x[static_cast<size_t>(k)] * wk;
                    } else {
                        const double q = s == 0
                                ? 0
                                : std::nearbyint(x[static_cast<size_t>(k)] / s);
                        acc += q * wk * s
                                * scales_[static_cast<size_t>((v.k0 + k) / 128)
                                                * w_.N
                                        + v.n0 + n];
                    }
                }
                out[static_cast<size_t>(m) * v.N + n] = acc;
            }
        }
        return out;
    }

    /// Whether every image byte that differs from the byte it replaced
    /// among those v reads is zero: zero words do not tell an image from
    /// its weight, as new zeros there would match them.
    bool reads_only_zero_image(const sw_weight_t &v, const uint8_t *p) const {
        bool any = false;
        for (const auto &r : rows_of(v))
            for (size_t b = r.first; b < r.second; ++b) {
                if (tag_[b] < 0 || p[b] == rimg_[b]) continue;
                if (p[b] != 0) return false;
                any = true;
            }
        return any;
    }

    /// Bytes of another type that hold no weight of v's: an element that
    /// is not finite or out of the range of any weight here.
    static bool wild(const sw_weight_t &v, const uint8_t *p) {
        for (int k = 0; k < v.K; ++k)
            for (int n = 0; n < v.N; ++n) {
                const float x = v.at(p, k, n);
                if (!std::isfinite(x) || std::fabs(x) > 1e3f) return true;
            }
        return false;
    }

    static bool close_to(
            const std::vector<float> &got, const std::vector<double> &want) {
        double rms = 0;
        for (double v : want)
            rms += v * v;
        rms = std::sqrt(rms / std::max<size_t>(1, want.size()));
        for (size_t i = 0; i < got.size(); ++i)
            if (std::fabs(got[i] - want[i])
                    > 2e-2 * (std::fabs(want[i]) + rms) + 1e-5)
                return false;
        return true;
    }

    /// The lines [first, first + 64) a served call on an image checks.
    static std::pair<size_t, size_t> check_lines(const uint8_t *image,
            const uint8_t *replaced, size_t used, size_t extent) {
        const size_t n = std::min(used, extent);
        if (n < kEndBytes) return {0, 0};
        const auto repeats = [&](size_t at) {
            for (size_t j = 8; j < kEndBytes; j += 8)
                if (std::memcmp(image + at, image + at + j, 8) != 0)
                    return false;
            return true;
        };
        const auto first = [&](bool tail) {
            size_t found = n;
            for (size_t j = 0; j < n / kEndBytes; ++j) {
                const size_t at
                        = tail ? n - (j + 1) * kEndBytes : j * kEndBytes;
                if (std::memcmp(image + at, replaced + at, kEndBytes) == 0)
                    continue;
                if (!repeats(at)) return at;
                if (found == n) found = at;
            }
            return found;
        };
        const size_t head = first(false), tail = first(true);
        const size_t last = head == n ? tail
                : tail == n           ? head
                                      : std::max(head, tail);
        return {std::min(head, tail), last};
    }

    /// Runs a call on v at mode wc with capacity cap (-1: a random one) and
    /// checks it; true when it was served.
    bool call(const sw_weight_t &v, int32_t wc, long cap,
            const std::function<void(const std::string &)> &record) {
        ++calls_;
        const size_t bytes = arena_bytes();
        if (cap < 0) {
            const int pick = uniform(0, 99);
            cap = pick < 60     ? 0
                    : pick < 85 ? static_cast<long>(bytes - v.offset)
                                : static_cast<long>(v.span());
        }
        const std::vector<uint8_t> before(
                arena_.as<uint8_t>(), arena_.as<uint8_t>() + bytes);
        std::vector<float> dst(static_cast<size_t>(kM) * v.N, -777.0f);
        nt::buffer_t src = nt::make_buffer(static_cast<size_t>(kM) * v.K * 2);
        const data_type_t sdt = v.dt == data_type_t::f16 ? v.dt : kBf16;
        for (size_t i = 0; i < static_cast<size_t>(kM) * v.K; ++i)
            nt::store_f32(src.get(), sdt, i, nt::load_f32(src_.get(), sdt, i));
        std::vector<uint16_t> scales;
        std::vector<float> src_scales(kM);
        nt::call_t c;
        c.M = kM;
        c.N = v.N;
        c.K = v.K;
        c.transB = v.trans;
        c.ldb = v.ldb;
        c.src = src.get();
        c.wei = arena_.as<uint8_t>() + v.offset;
        c.dst = dst.data();
        c.params.dtypes.src = sdt;
        c.params.dtypes.wei = v.dt;
        c.params.dtypes.dst = data_type_t::f32;
        c.params.num_threads = 4;
        c.params.weight_cache_type = wc;
        c.params.wei_buffer_capacity_bytes = static_cast<size_t>(cap);
        c.params.lowoha_algo = matmul_algo_t::native_brgemm;
        c.is_weights_const = true;
        if (v.dt == kS4) {
            scales.resize(static_cast<size_t>(v.K / 128) * v.N);
            for (int g = 0; g < v.K / 128; ++g)
                for (int n = 0; n < v.N; ++n)
                    nt::store_f32(scales.data(), kBf16,
                            static_cast<size_t>(g) * v.N + n,
                            scales_[static_cast<size_t>(v.k0 / 128 + g) * w_.N
                                    + v.n0 + n]);
            c.params.dtypes.compute = kS8;
            c.params.dynamic_quant = true;
            auto &qp = c.params.quant_params;
            qp.wei_scale.buff = scales.data();
            qp.wei_scale.dt = kBf16;
            qp.wei_scale.dims = {v.K / 128, v.N};
            qp.src_scale.buff = src_scales.data();
            qp.src_scale.dt = data_type_t::f32;
            qp.src_scale.dims = {kM, 1};
        }
        const status_t st = nt::run_matmul(c);
        const uint8_t *now = arena_.as<uint8_t>();
        const std::string what = describe(v, wc, cap);
        size_t lo = bytes, hi = 0;
        for (size_t i = 0; i < bytes; ++i)
            if (now[i] != before[i]) {
                lo = std::min(lo, i);
                hi = i + 1;
            }
        if (st != status_t::success) {
            ++refused_;
            if (!std::all_of(dst.begin(), dst.end(),
                        [](float x) { return x == -777.0f; }))
                record("refused with dst written: " + what);
            if (lo < hi) record("refused and the arena written: " + what);
            return false;
        }
        // The candidates the call may be served.
        bool entitled = true;
        for (const auto &r : rows_of(v))
            for (size_t b = r.first; b < r.second && entitled; ++b)
                entitled = tag_[b] < 0 || before[b] == rimg_[b];
        std::vector<uint8_t> held = before;
        for (size_t b = 0; b < bytes; ++b)
            if (tag_[b] >= 0) held[b] = rimg_[b];
        bool ok = (entitled && close_to(dst, model(v, before.data())))
                || close_to(dst, model(v, held.data()));
        for (const served_t &s : served_)
            if (!ok && s.v.same_request(v) && ends_hold(s, before.data())
                    && close_to(dst,
                            std::vector<double>(s.out.begin(), s.out.end())))
                ok = true;
        if (!ok && (wild(v, before.data()) || wild(v, held.data()))) {
            ++unchecked_;
            ok = true;
        }
        if (!ok && reads_only_zero_image(v, before.data())) {
            ++unchecked_;
            ok = true;
        }
        if (!ok && two_line_limit(v, before.data())) {
            ++unchecked_;
            ok = true;
        }
        if (!ok) record("served a wrong output: " + what);
        if (lo < hi) {
            // A rewrite in place: a new image over the call's buffer.
            const size_t used = nat::native_packed_bytes(v.dt, v.K, v.N, 32);
            if (wc != 2 || lo < v.offset || hi > v.offset + used)
                record("the arena written outside a rewrite: " + what);
            else
                add_image(v, used, before, cap);
        }
        if (wc != 0) {
            served_t s;
            s.v = v;
            s.out = dst;
            ends_of(s, before.data());
            served_.push_back(std::move(s));
            if (served_.size() > 64) served_.erase(served_.begin());
        }
        return true;
    }

    struct served_t {
        sw_weight_t v;
        std::vector<float> out;
        std::vector<uint8_t> head, tail;
    };

    void ends_of(served_t &s, const uint8_t *p) const {
        const size_t span = s.v.span(), end = std::min(span, kEndBytes);
        s.head.assign(p + s.v.offset, p + s.v.offset + end);
        s.tail.assign(p + s.v.offset + span - end, p + s.v.offset + span);
    }
    bool ends_hold(const served_t &s, const uint8_t *p) const {
        served_t t;
        t.v = s.v;
        ends_of(t, p);
        return t.head == s.head && t.tail == s.tail;
    }

    struct region_t {
        size_t at = 0, used = 0;
        sw_weight_t layout;
        std::vector<uint8_t> image;
        std::pair<size_t, size_t> lines;
        int clears = 0;
    };

    void add_image(const sw_weight_t &v, size_t used,
            const std::vector<uint8_t> &before, long cap) {
        const uint8_t *now = arena_.as<uint8_t>();
        region_t r;
        r.at = v.offset;
        r.used = used;
        r.layout = v;
        r.image.assign(now + v.offset, now + v.offset + used);
        r.lines = check_lines(
                r.image.data(), before.data() + v.offset, used, v.span());
        r.clears = clears_;
        const int id = static_cast<int>(regions_.size());
        regions_.push_back(std::move(r));
        // Past the weight's own bytes the layout's padding fills bytes of
        // the capacity the call gave it: they hold no weight anyone is
        // owed.
        const size_t own = std::min(used, v.span());
        for (size_t b = v.offset; b < v.offset + used; ++b) {
            tag_[b] = b < v.offset + own ? id : -1;
            rimg_[b] = before[b];
        }
        // The layouts cached from bytes the rewrite overwrote that still
        // held them take the image's ends.
        const size_t owned
                = std::max<size_t>(v.span(), static_cast<size_t>(cap));
        for (served_t &s : served_)
            if (s.v.offset >= v.offset
                    && s.v.offset + s.v.span() <= v.offset + owned
                    && s.v.offset < v.offset + used
                    && ends_hold(s, before.data()))
                ends_of(s, now);
    }

    bool two_line_limit(const sw_weight_t &v, const uint8_t *p) const {
        for (const region_t &r : regions_) {
            if (r.at != v.offset || r.clears != clears_) continue;
            const size_t n = std::min(kEndBytes, r.used);
            if (std::memcmp(p + r.at + r.lines.first,
                        r.image.data() + r.lines.first,
                        n) == 0
                    && std::memcmp(p + r.at + r.lines.second,
                               r.image.data() + r.lines.second, n)
                            == 0)
                return true;
        }
        return false;
    }

    std::string describe(const sw_weight_t &v, int32_t wc, long cap) const {
        return std::string(v.dt == kS4         ? "s4"
                               : v.dt == kBf16 ? "bf16"
                                               : "f16")
                + " K=" + std::to_string(v.K) + " N=" + std::to_string(v.N)
                + " ldb=" + std::to_string(v.ldb) + (v.trans ? " transB" : "")
                + " at byte " + std::to_string(v.offset) + " wc="
                + std::to_string(wc) + " capacity=" + std::to_string(cap)
                + " after " + std::to_string(clears_) + " clears, call "
                + std::to_string(calls_);
    }

    std::mt19937 rng_;
    sw_weight_t w_;
    nt::buffer_t arena_, src_;
    std::vector<uint8_t> raw_, rimg_;
    std::vector<int> tag_;
    std::vector<float> scales_;
    std::vector<sw_weight_t> fresh_;
    std::vector<served_t> served_;
    std::vector<region_t> regions_;
    int clears_ = 0, violations_ = 0;
    long calls_ = 0, refused_ = 0, unchecked_ = 0;
};

/// The weights of the sweep: bf16, f16 and s4, transposed or not (s4
/// transposed, in groups of 128), of images up to 64 KiB, which a record
/// checks at every byte (a larger one is checked at its sample lines).
struct sweep_case_t {
    data_type_t dt;
    int K, N;
    bool trans;
};

const sweep_case_t kSweepCases[] = {{kBf16, 64, 64, false},
        {kBf16, 96, 64, true}, {data_type_t::f16, 64, 128, false},
        {kS4, 256, 64, true}, {kS4, 2048, 64, true}};

// A deterministic part of the randomized sweep (weight_sweep_t): each of
// its weights, two seeds of 150 steps, at ZENDNNL_MATMUL_ALGO=10 under
// weight cache modes 0 to 2: no call is served an output it is not owed,
// and the whole raw weight written back and new weights are served.
TEST_F(NativeWeightLifetime, ARandomizedSweepServesNoWrongOutput) {
    pin_native();
    for (const sweep_case_t &c : kSweepCases)
        for (const uint32_t seed : {1301u, 1302u}) {
            ::reset_grp_matmul_caches();
            std::string first;
            {
                weight_sweep_t sweep(c.dt, c.K, c.N, c.trans,
                        seed * 31 + static_cast<uint32_t>(c.K + c.N));
                EXPECT_EQ(sweep.run(150, first), 0)
                        << first << " (seed " << seed << ")";
            }
            // Its records go as freed memory for the tests after it.
            mm::clear_matmul_native_weight_caches();
            mm::clear_matmul_native_weight_caches();
        }
}

bool native_bf16_isa() {
    const nat::UarchParams &u = nat::detect_uarch();
    return u.avx512bf16 && u.avx512vnni;
}

void clear_native_caches_at_exit() {
    mm::clear_matmul_native_weight_caches();
}

// A clear of the native weight caches in an exit hook that the process
// registered before its first native call runs after the static objects
// made since are destroyed: the process still exits with code 0, the
// clear having found an image and a cached layout, after a call the native
// BRGEMM handed to the default route under ZENDNNL_MATMUL_ALGO=10.
TEST(NativeWeightLifetimeExit, ClearInAnExitHookRegisteredFirst) {
    if (!native_bf16_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    nt::expect_in_child({}, [] {
        if (std::atexit(clear_native_caches_at_exit) != 0) return 1;
        // Kept to the exit, as the hook runs then; volatile, so leak checks
        // find them through these pointers.
        static const nt::algo_pin_t *volatile const pin
                = new nt::algo_pin_t(matmul_algo_t::native_brgemm);
        static const WeightCacheGuard *volatile const mode
                = new WeightCacheGuard(2);
        (void)pin;
        (void)mode;
        int bad = 0;
        const weight_t image = make_weight(kBf16, 64, 64, 64, false, 801);
        dense_t d(kBf16, 64);
        if (d.run(image, 2) != status_t::success || !in_place(image)) ++bad;
        const weight_t cached = make_weight(kS8, 64, 64, 64, false, 802);
        dense_t e(kS8, 64);
        if (e.run(cached, 1) != status_t::success
                || nt::native_layouts(cached.ptr()) != 1)
            ++bad;
        // W4A16 with per-channel scales, which the native BRGEMM hands
        // back.
        const int M = 4, N = 64, K = 64;
        const nt::buffer_t wei
                = nt::random_buffer(data_type_t::u8, K * N / 2, 803, 0, 255);
        const nt::buffer_t src
                = nt::random_buffer(kBf16, M * K, 804, -1.0f, 1.0f);
        nt::buffer_t dst = nt::make_buffer(M * N * 2);
        std::vector<float> scales(N, 0.25f);
        nt::call_t c;
        c.M = M;
        c.N = N;
        c.K = K;
        c.src = src.get();
        c.wei = wei.get();
        c.dst = dst.get();
        c.params.dtypes.src = kBf16;
        c.params.dtypes.wei = kS4;
        c.params.dtypes.dst = kBf16;
        c.params.quant_params.wei_scale.buff = scales.data();
        c.params.quant_params.wei_scale.dt = data_type_t::f32;
        c.params.quant_params.wei_scale.dims = {1, N};
        c.params.weight_cache_type = 1;
        nt::run_matmul(c);
        return bad;
    }, "a clear in an exit hook registered before the first call");
}

// A process that exits while a call of another thread runs on a cached
// native layout exits with code 0: no static object destroyed at the exit
// holds what the call reads.
TEST(NativeWeightLifetimeExit, ExitWhileACallRuns) {
    if (!native_bf16_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    for (const int delay_us : {0, 1000, 3000}) {
        SCOPED_TRACE("exit " + std::to_string(delay_us) + " us after the "
                + "second call");
        nt::expect_in_child({}, [delay_us] {
            const int M = 64, N = 4096, K = 4096;
            // Kept to the exit, as the call goes on then; volatile, so leak
            // checks find it through this pointer.
            static const WeightCacheGuard *volatile const mode
                    = new WeightCacheGuard(1);
            (void)mode;
            static const nt::buffer_t *const wei
                    = new nt::buffer_t(nt::random_buffer(kBf16,
                            static_cast<size_t>(N) * K, 811, -1.0f, 1.0f));
            static const nt::buffer_t *const src
                    = new nt::buffer_t(nt::random_buffer(kBf16,
                            static_cast<size_t>(M) * K, 812, -1.0f, 1.0f));
            static const nt::buffer_t *const dst = new nt::buffer_t(
                    nt::make_buffer(static_cast<size_t>(M) * N * 4));
            static std::atomic<int> calls {0};
            std::thread([=] {
                for (;;) {
                    mm::matmul_batch_params_t batch;
                    mm::matmul_params p;
                    p.dtypes.src = kBf16;
                    p.dtypes.wei = kBf16;
                    p.dtypes.dst = data_type_t::f32;
                    p.lowoha_algo = matmul_algo_t::native_brgemm;
                    p.num_threads = 1;
                    p.weight_cache_type = 1;
                    if (mm::matmul_direct('r', false, false, M, N, K, 1.0f,
                                src->get(), K, wei->get(), N, nullptr, 0.0f,
                                dst->get(), N, true, batch, p)
                            != status_t::success)
                        std::_Exit(3);
                    calls.fetch_add(1);
                }
            }).detach();
            while (calls.load() < 2)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
            return nt::native_layouts(wei->get()) == 1 ? 0 : 1;
        }, "an exit while a native call runs");
    }
}

// ZENDNNL_MATMUL_WEIGHT_CACHE is read once, when the process starts using
// the library: setting it after the first call changes no later call.  In
// child processes, which begin at weight cache mode 2 or 0.
TEST(NativeWeightLifetimeConfig, WeightCacheVariableSetLaterChangesNothing) {
    if (!native_bf16_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    for (const char *first : {"2", "0"}) {
        SCOPED_TRACE(std::string("ZENDNNL_MATMUL_WEIGHT_CACHE=") + first);
        nt::expect_in_child({{"ZENDNNL_MATMUL_WEIGHT_CACHE", first}}, [first] {
            const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
            const bool mode2 = first[0] == '2';
            const weight_t a = make_weight(kBf16, 64, 64, 64, false, 1301);
            const weight_t b = make_weight(kBf16, 64, 64, 64, false, 1302);
            dense_t d(kBf16, 64);
            if (d.run(a, 2) != status_t::success || in_place(a) != mode2)
                return 1;
            const nt::env_pin_t later(
                    "ZENDNNL_MATMUL_WEIGHT_CACHE", mode2 ? "0" : "2");
            if (d.run(b, 2) != status_t::success || in_place(b) != mode2) {
                std::fprintf(stderr, "the later value took effect\n");
                return 1;
            }
            return 0;
        }, "ZENDNNL_MATMUL_WEIGHT_CACHE set after the first call");
    }
}

#if defined(__GLIBC__)
// A model swap on the process heap: the eight 1 MiB weights of one model
// are rewritten in place and freed, then the next model takes 48 weights of
// 192 KiB from the same heap, some inside the freed images where the C
// library places them there (how many depends on its version and settings;
// WeightInsideAFreedImageIsCheckedAtItsSamples places them).  Every call on
// the new weights is served right, with no clear between the models and
// with one.
TEST(NativeWeightLifetimeHeap, ModelSwapServesEveryNewWeight) {
    if (!native_bf16_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    if (kSanitizedHeap) GTEST_SKIP() << "needs the C library's heap";
    nt::expect_in_child({}, [] {
        // Freed weights stay on the heap, for the next model's weights.
        mallopt(M_MMAP_THRESHOLD, 256 << 20);
        mallopt(M_TRIM_THRESHOLD, 512 << 20);
        // Never destroyed; volatile, so leak checks find them through these
        // pointers.
        static const nt::algo_pin_t *volatile const pin
                = new nt::algo_pin_t(matmul_algo_t::native_brgemm);
        static const WeightCacheGuard *volatile const mode
                = new WeightCacheGuard(2);
        (void)pin;
        (void)mode;
        const int old_k = 8192, new_k = 1536, old_n = 8, new_n = 48;
        const size_t old_bytes = static_cast<size_t>(old_k) * 64 * 2;
        const size_t new_bytes = static_cast<size_t>(new_k) * 64 * 2;
        int failures = 0;
        for (bool cleared : {false, true}) {
            int bad = 0;
            std::vector<void *> old_w(old_n, nullptr);
            dense_t d_old(kBf16, old_k);
            for (void *&p : old_w) {
                if (posix_memalign(&p, 64, old_bytes) != 0) return 1;
                const weight_t w = weight_at(static_cast<uint8_t *>(p), old_k,
                        1041 + static_cast<uint32_t>(&p - old_w.data()));
                if (d_old.run(w, 2) != status_t::success || !in_place(w)) ++bad;
            }
            for (void *p : old_w)
                std::free(p);
            if (cleared) mm::clear_matmul_native_weight_caches();
            int inside = 0;
            std::vector<void *> new_w(new_n, nullptr);
            dense_t d_new(kBf16, new_k);
            for (void *&p : new_w) {
                if (posix_memalign(&p, 64, new_bytes) != 0) return 1;
                const weight_t w = weight_at(static_cast<uint8_t *>(p), new_k,
                        1061 + static_cast<uint32_t>(&p - new_w.data()));
                const weight_t raw = clone(w);
                const auto at = reinterpret_cast<std::uintptr_t>(p);
                for (void *o : old_w) {
                    const auto lo = reinterpret_cast<std::uintptr_t>(o);
                    inside += at > lo && at + new_bytes < lo + old_bytes;
                }
                if (d_new.run(w, 1) != status_t::success || !d_new.matches(raw))
                    ++bad;
            }
            // The C library may place no new weight inside a freed image:
            // the calls are then checked all the same.
            std::fprintf(stderr,
                    "%s: %d of %d new weights inside a freed image, %d bad\n",
                    cleared ? "with a clear" : "without a clear", inside, new_n,
                    bad);
            for (void *p : new_w)
                std::free(p);
            mm::clear_matmul_native_weight_caches();
            failures += bad;
        }
        return failures;
    }, "a model swap on the process heap");
}
#endif

#if defined(__GLIBC__)
// A cached layout that another thread finds stale while no memory is left to
// keep it as a dropped layout stays the entry's: the thread whose call holds
// it reads it on, the call that found it stale fails, and the next call
// drops it and packs the weight again, the dropped layout still readable.
// In a child process whose 8 MiB layouts go back to the system when freed,
// so that a read of a freed one faults.
TEST(NativeWeightLifetimeHeap, ALayoutWithNoMemoryToDropStaysReadable) {
    if (!native_bf16_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    nt::expect_in_child({}, [] {
        mallopt(M_MMAP_THRESHOLD, 1 << 20);
        const int K = 2048, N = 2048, pack_nr = 32;
        const weight_t w = make_weight(kBf16, K, N, N, false, 1401);
        nat::native_raw_weight_t r;
        r.wei_dt = kBf16;
        r.weight = w.ptr();
        r.K = K;
        r.N = N;
        r.ldb = N;
        r.pack_nr = pack_nr;
        nat::native_layout_t held;
        if (nat::get_native_layout(r, 1, 0, 1, held) != status_t::success
                || held.owned != nullptr)
            return 1;
        const size_t bytes = nat::native_block_bytes(kBf16, K, pack_nr)
                * static_cast<size_t>(N / pack_nr);
        const auto *layout = static_cast<const uint8_t *>(held.data);
        const std::vector<uint8_t> copy(layout, layout + bytes);
        status_t stale = status_t::success, again = status_t::failure;
        std::thread([&] {
            nt::store_f32(w.buf.get(), kBf16, 0, 3.0f);
            nat::native_fail_next_retirements(1);
            nat::native_layout_t l;
            stale = nat::get_native_layout(r, 1, 0, 1, l);
            nat::native_fail_next_retirements(0);
        }).join();
        if (stale != status_t::failure) return 2;
        if (std::memcmp(layout, copy.data(), bytes) != 0) return 3;
        std::thread([&] {
            nat::native_layout_t l;
            again = nat::get_native_layout(r, 1, 0, 1, l);
        }).join();
        if (again != status_t::success
                || nat::native_retired_layouts(w.ptr()) != 1)
            return 4;
        return std::memcmp(layout, copy.data(), bytes) != 0 ? 5 : 0;
    }, "a layout with no memory to drop it");
}
#endif

} // namespace
