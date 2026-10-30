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
// and no check of what is kept reads past the buffer of the call that makes
// it.  A call's check for an image is handed to its layout lookup, and what
// a thread's calls were served stays right for its later calls across
// concurrent first calls, images, stale layouts, clears and exited threads.

#include <algorithm>
#include <atomic>
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
    /// The process-wide ALGO 11, under which weight cache mode 2 rewrites a
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
