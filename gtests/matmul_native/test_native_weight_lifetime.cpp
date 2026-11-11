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
// a rewrite in place waits for the packs and the GEMV kernels that read the
// bytes it writes, and
// no check of what is kept reads past the buffer of the call that makes it.
// A call's check for an image is handed to its layout lookup, and what
// a thread's calls were served stays right for its later calls across
// concurrent first calls, images, stale layouts, clears and exited threads.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#include <sys/mman.h>
#endif

#include <gtest/gtest.h>

#include "gtest_utils.hpp"
#include "lowoha_operators/matmul/lowoha_matmul.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_gemv.hpp"
#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_hw.hpp"
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
    /// capacity: wei_buffer_capacity_bytes of the call.
    status_t run(const weight_t &w, int32_t wc, size_t capacity = 0) {
        dst.assign(static_cast<size_t>(M) * w.N, -777.0f);
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
        c.params.lowoha_algo = matmul_algo_t::native_brgemm;
        return nt::run_matmul(c);
    }
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
    /// bf16 buffer in place.
    void pin_native() {
        algo_ = std::make_unique<nt::algo_pin_t>(matmul_algo_t::native_brgemm);
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
// bytes; the raw weight the caller then writes into the buffer is read as
// raw (and rewritten in place again).
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
        EXPECT_FALSE(in_place(w));
        EXPECT_EQ(std::memcmp(w.ptr(), image.ptr(), w.buf.bytes), 0)
                << "a clear does not write the buffer";
        fill(w, trans_b ? 303 : 304);
        const weight_t w2 = clone(w);
        ASSERT_EQ(d.run(w, 2), status_t::success);
        EXPECT_EQ(nt::native_layouts(w.ptr()), 1u) << "the new image";
        EXPECT_TRUE(d.matches(w2));
        EXPECT_TRUE(in_place(w));
    }
}

// A raw weight written over an image with the image's bytes at both ends is
// still read as the image, until the native weight caches are cleared.
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
    ASSERT_EQ(d.run(w, 2), status_t::success);
    EXPECT_TRUE(d.matches(w2));
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
// mode 0 call, on a bf16 or an s4 weight, through the tiles or through the
// GEMV kernel of a one-row call (which the tiles replace once the buffer
// holds an image), gives the bits of the same call on the raw weight or on
// the image, never of a weight read in part from each; the mode 2 call
// gives its bits on the raw weight.
TEST_F(NativeWeightLifetime, ModeZeroPacksAndARewriteDoNotOverlap) {
    pin_native();
    // Weights of 32 MiB, which one thread packs for longer than the rewrite
    // takes to start writing.
    const struct {
        data_type_t dt;
        int M, K, N;
        const char *name;
    } readers[] = {{kBf16, 4, 4096, 4096, "bf16 tiles"},
            {kBf16, 1, 4096, 4096, "bf16 GEMV"},
            {kS4, 4, 8192, 8192, "s4 tiles"}};
    using clock_t = std::chrono::steady_clock;
    for (const auto &rd : readers) {
        SCOPED_TRACE(rd.name);
        const nk_call_t reader(rd.dt, rd.M, rd.K, rd.N),
                writer(rd.dt, 4, rd.K, rd.N);
        {
            float dst[1], scale[4];
            const nt::buffer_t probe = reader.make_weight(800);
            EXPECT_EQ(
                    nt::gemv_kernel(reader.call(probe.get(), 0, 1, dst, scale))
                            != nat::native_gemv_t::none,
                    rd.M == 1);
        }
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
// lies inside the image.  A raw read of the buffer's first bytes, held
// through native_weight_read_raw, keeps the rewrite from writing meanwhile.
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
                read = nat::native_weight_read_raw(
                        buf.get(), kEndBytes, buf.bytes, [&] {
                    held.store(true);
                    while (!release.load())
                        std::this_thread::yield();
                });
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

/// The descriptor native_layout_matmul builds for c, a call of one thread.
nat::GemmDescriptor gemv_descriptor(const nt::call_t &c) {
    nat::GemmDescriptor d;
    d.M = c.M;
    d.N = c.N;
    d.K = c.K;
    d.lda = c.lda_or_tight();
    d.ldb = c.ldb_or_tight();
    d.ldc = c.ldc_or_tight();
    d.alpha = c.alpha;
    d.beta = c.beta;
    d.src_dt = c.params.dtypes.src;
    d.wei_dt = c.params.dtypes.wei;
    d.dst_dt = c.params.dtypes.dst;
    d.bias_dt = c.params.dtypes.bias;
    d.is_weights_const = c.is_weights_const;
    d.num_threads = 1;
    return d;
}

// A one-row bf16 call on its GEMV kernel that found its weight raw before a
// first call at weight cache mode 2 reserved the rewrite of the buffer in
// place: the BKC kernel's thread-local hit check and the narrow kernel (the
// first 4 columns, ldb N) read the raw weight without holding it, so they
// wait until the rewrite ends, and the call is then left to the tiles,
// which serve it from the image.  A raw read of the buffer's first bytes,
// held through native_weight_read_raw, keeps the rewrite from writing
// meanwhile.
TEST_F(NativeWeightLifetime, GemvReadsOfAWeightWaitForItsRewrite) {
    const int K = 256, N = 128;
    for (const int n : {N, 4}) {
        const nat::native_gemv_t kernel = n == N
                ? nat::native_gemv_t::bf16_bkc
                : nat::native_gemv_t::bf16_narrow;
        SCOPED_TRACE(nat::native_gemv_name(kernel));
        const nt::buffer_t buf = contiguous_weight(kBf16, K, N, 1461);
        const nt::buffer_t src = nt::random_buffer(kBf16, K, 1462, -1, 1);
        std::vector<float> warm(n, -777.0f), got(n, -777.0f);
        nt::call_t c;
        c.M = 1;
        c.N = n;
        c.K = K;
        c.ldb = N;
        c.src = src.get();
        c.wei = buf.get();
        c.dst = warm.data();
        c.params.dtypes.src = kBf16;
        c.params.dtypes.wei = kBf16;
        c.params.dtypes.dst = data_type_t::f32;
        c.params.num_threads = 1;
        c.params.weight_cache_type = 1;
        c.params.lowoha_algo = matmul_algo_t::native_brgemm;
        ASSERT_EQ(nt::gemv_kernel(c), kernel);
        const nat::GemmDescriptor desc = gemv_descriptor(c);
        const nat::native_gemv_plan_t plan
                = nat::decide_native_gemv(desc, c.params);
        ASSERT_EQ(plan.kernel, kernel);
        std::atomic<bool> found {false}, held {false}, release {false},
                go {false}, returned {false};
        status_t warmed = status_t::failure, again = status_t::failure,
                 gemv = status_t::success, rewrite = status_t::failure;
        nat::native_raw_read_t read = nat::native_raw_read_t::failed;
        {
            joiner_t calls;
            calls.threads.emplace_back([&] {
                warmed = nt::run_matmul(c);
                again = nat::run_native_gemv(plan, desc, src.get(), buf.get(),
                        got.data(), nullptr, c.params, 1);
                found.store(true);
                while (!go.load())
                    std::this_thread::yield();
                std::fill(got.begin(), got.end(), -777.0f);
                gemv = nat::run_native_gemv(plan, desc, src.get(), buf.get(),
                        got.data(), nullptr, c.params, 1);
                returned.store(true);
            });
            // The GEMV's call found the weight raw before the rewrite starts.
            while (!found.load())
                std::this_thread::yield();
            calls.threads.emplace_back([&] {
                read = nat::native_weight_read_raw(
                        buf.get(), kEndBytes, buf.bytes, [&] {
                    held.store(true);
                    while (!release.load())
                        std::this_thread::yield();
                });
            });
            while (!held.load())
                std::this_thread::yield();
            calls.threads.emplace_back([&] {
                nat::native_layout_t o;
                rewrite = nat::get_native_layout(
                        raw_weight(kBf16, buf.get(), K, N), 2, 0, 1, o);
            });
            // The rewrite reserved its record and waits for the read.
            while (!nat::native_weight_packed_in_place(buf.get(), buf.bytes))
                std::this_thread::yield();
            go.store(true);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            EXPECT_FALSE(returned.load())
                    << "the GEMV read the weight while its rewrite ran";
            release.store(true);
        }
        ASSERT_EQ(warmed, status_t::success);
        EXPECT_EQ(again, status_t::success);
        EXPECT_EQ(read, nat::native_raw_read_t::done);
        EXPECT_EQ(rewrite, status_t::success);
        EXPECT_TRUE(nat::native_weight_packed_in_place(buf.get(), buf.bytes));
        EXPECT_EQ(gemv, status_t::unimplemented)
                << "the buffer holds the image, which the tiles serve";
        EXPECT_TRUE(std::all_of(got.begin(), got.end(), [](float v) {
            return v == -777.0f;
        })) << "the GEMV wrote dst";
        mm::clear_matmul_native_weight_caches();
    }
}

// A rewrite in place waits for the checks under way before it writes, and
// holds no lock while it waits: while the narrow GEMV of a long weight runs
// inside its check, a first call at weight cache mode 2 rewriting another
// buffer waits for it, and a new thread's first cached layout, which lists
// the thread's reader under the cache lock, is cached meanwhile.  Were the
// lock held, a check that waits for the cache lock (as its image test did)
// would wait for itself through the two.  The GEMV reads its weight and its
// source from untouched pages of a sparse mapping, so it runs for longer
// than the other calls take.
TEST_F(NativeWeightLifetime, FirstLayoutsGoOnWhileARewriteWaitsForAGemv) {
#if defined(__linux__)
    const int K = nat::kMaxNativeBf16K, n = 4;
    const size_t wei_bytes = static_cast<size_t>(K) * n * sizeof(uint16_t);
    const size_t src_bytes = static_cast<size_t>(K) * sizeof(uint16_t);
    struct mapping_t {
        void *at = MAP_FAILED;
        size_t bytes = 0;
        ~mapping_t() {
            if (at != MAP_FAILED) munmap(at, bytes);
        }
    } map;
    map.bytes = wei_bytes + src_bytes;
    map.at = mmap(nullptr, map.bytes, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (map.at == MAP_FAILED)
        GTEST_SKIP() << "no address space for the sparse weight";
    const void *wei = map.at;
    const void *src = static_cast<const char *>(map.at) + wei_bytes;
    nt::call_t c;
    c.M = 1;
    c.N = n;
    c.K = K;
    c.src = src;
    c.wei = wei;
    c.params.dtypes.src = kBf16;
    c.params.dtypes.wei = kBf16;
    c.params.dtypes.dst = data_type_t::f32;
    c.params.num_threads = 1;
    c.params.weight_cache_type = 1;
    c.params.lowoha_algo = matmul_algo_t::native_brgemm;
    const nat::GemmDescriptor desc = gemv_descriptor(c);
    const nat::native_gemv_plan_t plan
            = nat::decide_native_gemv(desc, c.params);
    ASSERT_EQ(plan.kernel, nat::native_gemv_t::bf16_narrow);
    const nt::buffer_t rewritten = contiguous_weight(kBf16, 256, 128, 1471);
    const nt::buffer_t first = contiguous_weight(kBf16, 64, 64, 1472);
    const auto deadline
            = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    const auto waiting = [&] {
        std::this_thread::yield();
        return std::chrono::steady_clock::now() < deadline;
    };
    std::vector<float> got(n, -777.0f);
    std::atomic<bool> gemv_returned {false}, rewrite_returned {false};
    std::atomic<bool> cached_during_gemv {false}, cached_during_rewrite {false};
    status_t gemv = status_t::failure, rewrite = status_t::failure,
             cached = status_t::failure;
    {
        joiner_t calls;
        calls.threads.reserve(3);
        calls.threads.emplace_back([&] {
            gemv = nat::run_native_gemv(
                    plan, desc, src, wei, got.data(), nullptr, c.params, 1);
            gemv_returned.store(true);
        });
        // The GEMV's kernel read the weight's first page: its check began.
        unsigned char resident = 0;
        while (mincore(map.at, 1, &resident) == 0 && !(resident & 1)
                && waiting()) {}
        ASSERT_TRUE(resident & 1) << "the GEMV did not start";
        calls.threads.emplace_back([&] {
            nat::native_layout_t o;
            rewrite = nat::get_native_layout(
                    raw_weight(kBf16, rewritten.get(), 256, 128), 2, 0, 1, o);
            rewrite_returned.store(true);
        });
        // The rewrite reserved its record, and then waits for the check.
        while (!nat::native_weight_packed_in_place(
                       rewritten.get(), rewritten.bytes)
                && waiting()) {}
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        calls.threads.emplace_back([&] {
            nat::native_layout_t o;
            cached = nat::get_native_layout(
                    raw_weight(first.get(), 64, 64), 1, 0, 1, o);
            cached_during_gemv.store(!gemv_returned.load());
            cached_during_rewrite.store(!rewrite_returned.load());
        });
    }
    EXPECT_EQ(gemv, status_t::success);
    EXPECT_EQ(rewrite, status_t::success);
    EXPECT_EQ(cached, status_t::success);
    EXPECT_TRUE(nat::native_weight_packed_in_place(
            rewritten.get(), rewritten.bytes));
    EXPECT_TRUE(cached_during_gemv.load())
            << "the first layout was cached only once the GEMV's check had "
               "ended: the rewrite waited for the check holding a lock";
    EXPECT_TRUE(cached_during_rewrite.load())
            << "the rewrite did not wait for the GEMV's check";
    EXPECT_TRUE(std::all_of(got.begin(), got.end(), [](float v) {
        return v == 0.0f;
    })) << "the GEMV of a zero weight";
    mm::clear_matmul_native_weight_caches();
#else
    GTEST_SKIP() << "needs a sparse anonymous mapping";
#endif
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
// cleared.
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
    ASSERT_EQ(d.run(w, 2), status_t::success);
    EXPECT_TRUE(d.matches(w3));
    EXPECT_NE(std::memcmp(image.ptr(), image2.ptr(), image.buf.bytes), 0);
}

// After either clear, a thread's next call on a buffer it was served from
// is not served what it was: a new weight with the old bytes at both ends
// is packed (or rewritten) again, for a cached layout and for an image.
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

// The layouts a weight's new bytes drop stay held until the caches are
// cleared, with the cached ones: at ZENDNNL_LRU_CACHE_CAPACITY 2, the third
// layout of a weight whose bytes changed twice is packed for its call.
TEST_F(NativeWeightLifetime, DroppedLayoutsCountTowardTheCapacity) {
    const nt::lru_capacity_pin_t capacity(2);
    const int K = 64, N = 64;
    for (data_type_t dt : {kBf16, kS8}) {
        SCOPED_TRACE(dt == kBf16 ? "bf16" : "s8");
        ::reset_grp_matmul_caches();
        const weight_t w = make_weight(dt, K, N, N, false, 681);
        dense_t d(dt, K);
        ASSERT_EQ(d.run(w, 1), status_t::success);
        EXPECT_TRUE(d.matches(w));
        EXPECT_EQ(nt::native_layouts(w.ptr()), 1u);
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
        mm::clear_matmul_native_weight_caches();
        ASSERT_EQ(d.run(w, 1), status_t::success);
        EXPECT_TRUE(d.matches(w));
        EXPECT_EQ(nt::native_layouts(w.ptr()), 1u) << "cached after the clear";
    }
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
// or a clear meets: the next threads' calls, before and after a clear and a
// new weight with the old ends, are served right.
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
    keep_ends(a, image.ptr());
    fill(b, 654);
    keep_ends(b, b1.ptr());
    const weight_t a2 = clone(a), b2 = clone(b);
    EXPECT_EQ(served_right(a2, b2), 0);
    dense_t d(kBf16, K);
    ASSERT_EQ(d.run(a, 2), status_t::success);
    EXPECT_TRUE(d.matches(a2));
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
    weight_t w;
    w.dt = kBf16;
    w.N = 64;
    w.ldb = 64;
    w.K = static_cast<int>(
            pages * pages_t::page_bytes() / (64 * sizeof(uint16_t)));
    w.buf.mem = std::shared_ptr<uint8_t>(p, [](uint8_t *) {});
    w.buf.bytes = pages * pages_t::page_bytes();
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

} // namespace
