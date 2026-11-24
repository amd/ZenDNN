/********************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *******************************************************************************/

// An omitted source scale is 1, as in oneDNN. A static W4A8 call reads it as
// {M, 1} ones of the weight scale's type, on raw and on prepacked weights. A
// classic int8 call's omitted per-tensor source scale is read as 1 as well.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#ifndef _WIN32
#include <sys/resource.h>
#endif

#include <gtest/gtest.h>

#include "lowoha_operators/matmul/lowoha_matmul.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"
#include "lowoha_operators/reorder/lowoha_reorder.hpp"
#include "lowoha_operators/reorder/prepack/lowoha_prepack.hpp"

namespace {

namespace mm = zendnnl::lowoha::matmul;
using zendnnl::common::data_type_t;
using zendnnl::common::matmul_algo_t;
using zendnnl::common::size_of;
using zendnnl::error_handling::status_t;
using dims_t = std::vector<int64_t>;
using quant_t = mm::matmul_quantization_params_t::matmul_quant_t;

std::vector<int8_t> random_s8(size_t n, int lo, int hi, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> d(lo, hi);
    std::vector<int8_t> v(n);
    for (auto &x : v)
        x = static_cast<int8_t>(d(gen));
    return v;
}

std::vector<float> random_f32(size_t n, float lo, float hi, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (auto &x : v)
        x = d(gen);
    return v;
}

uint16_t to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return static_cast<uint16_t>(u >> 16);
}

float from_bf16(uint16_t b) {
    const uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

/// Element i of a f32 or bf16 quant buffer, as f32.
float value_at(const quant_t &q, size_t i) {
    if (q.dt == data_type_t::f32) return static_cast<const float *>(q.buff)[i];
    return from_bf16(static_cast<const uint16_t *>(q.buff)[i]);
}

void set_quant(quant_t &q, const void *buff, data_type_t dt, dims_t dims) {
    q.buff = buff;
    q.dt = dt;
    q.dims = std::move(dims);
}

/// n ones of dt (f32 or bf16), as bytes.
std::vector<uint8_t> ones_of(data_type_t dt, size_t n) {
    std::vector<uint8_t> out(n * size_of(dt));
    for (size_t i = 0; i < n; ++i) {
        if (dt == data_type_t::f32) {
            const float one = 1.0f;
            std::memcpy(out.data() + i * 4, &one, 4);
        } else {
            const uint16_t one = 0x3F80;
            std::memcpy(out.data() + i * 2, &one, 2);
        }
    }
    return out;
}

/// A static W4A8 call: s8 source, s4 weights, s8 compute, f32 output.
mm::matmul_params w4a8_params(bool packed) {
    mm::matmul_params p;
    p.dtypes.src = data_type_t::s8;
    p.dtypes.wei = data_type_t::s4;
    p.dtypes.dst = data_type_t::f32;
    p.dtypes.compute = data_type_t::s8;
    p.weight_cache_type = 1;
    if (packed) {
        p.mem_format_b = 'r';
        p.lowoha_algo = matmul_algo_t::aocl_dlp_blocked;
    }
    return p;
}

/// Prepacks a K x N weight for AOCL-DLP. Empty when the pack is unavailable.
std::vector<uint8_t> prepack(const void *raw, data_type_t wei, data_type_t src,
        int K, int N, bool transposed, int group) {
    zendnnl::lowoha::reorder::reorder_params_t rp;
    rp.is_prepack = true;
    rp.prepack.algo = matmul_algo_t::aocl_dlp_blocked;
    rp.prepack.wei_dtype = wei;
    rp.prepack.src_dtype = src;
    rp.prepack.K = K;
    rp.prepack.N = N;
    rp.prepack.ldb = transposed ? K : N;
    rp.prepack.transposed = transposed;
    if (group > 0) rp.prepack.sym_group_size = group;
    std::vector<uint8_t> out(zendnnl::lowoha::reorder::weight_prepack_size(rp));
    if (out.empty()
            || zendnnl::lowoha::reorder::reorder_direct(raw, out.data(), rp)
                    != status_t::success)
        out.clear();
    return out;
}

struct run_t {
    status_t status = status_t::failure;
    std::vector<float> dst; ///< empty when the pack was unavailable
};

/// One matmul_direct call into an f32, bf16 or s32 destination, read back as
/// f32. The destination starts at -7, so a call that writes nothing shows.
run_t run(const mm::matmul_params &p, int M, int N, int K, const void *src,
        int lda, const void *wei, int ldb, bool transB, data_type_t dst_dt,
        mm::matmul_batch_params_t batch = {}) {
    mm::clear_matmul_weight_caches();
    const size_t n
            = static_cast<size_t>(std::max({batch.Batch_A, batch.Batch_B, 1}))
            * M * N;
    std::vector<float> f(n, -7.0f);
    std::vector<uint16_t> b(n, to_bf16(-7.0f));
    std::vector<int32_t> i(n, -7);
    void *dst = dst_dt == data_type_t::f32 ? static_cast<void *>(f.data())
            : dst_dt == data_type_t::bf16  ? static_cast<void *>(b.data())
                                           : static_cast<void *>(i.data());
    run_t r;
    r.status = mm::matmul_direct('r', false, transB, M, N, K, 1.0f, src, lda,
            wei, ldb, nullptr, 0.0f, dst, N, true, batch, p);
    r.dst.resize(n);
    for (size_t k = 0; k < n; ++k)
        r.dst[k] = dst_dt == data_type_t::f32 ? f[k]
                : dst_dt == data_type_t::bf16 ? from_bf16(b[k])
                                              : static_cast<float>(i[k]);
    return r;
}

// ── apply_quant_defaults ───────────────────────────────────────────────

TEST(QuantDefaults, W4A8OmittedSourceScaleIsOnesOfTheWeightScaleType) {
    constexpr int M = 3, N = 8, K = 64;
    const std::vector<float> wf(N, 0.25f);
    const std::vector<uint16_t> wb(N, to_bf16(0.25f));
    for (const data_type_t dt : {data_type_t::f32, data_type_t::bf16}) {
        SCOPED_TRACE(dt == data_type_t::f32 ? "f32" : "bf16");
        auto p = w4a8_params(false);
        set_quant(p.quant_params.wei_scale,
                dt == data_type_t::f32 ? static_cast<const void *>(wf.data())
                                       : static_cast<const void *>(wb.data()),
                dt, {1, N});
        std::vector<uint8_t> hold;
        ASSERT_TRUE(mm::apply_quant_defaults(p, M, N, K, hold));
        const auto &s = p.quant_params.src_scale;
        EXPECT_EQ(s.dt, dt);
        EXPECT_EQ(s.dims, (dims_t {M, 1}));
        EXPECT_EQ(mm::validate_w4a8_inputs(p, M, N, K), status_t::success);
        mm::materialize_quant_defaults(p, hold);
        ASSERT_EQ(hold.size(), static_cast<size_t>(M) * size_of(dt));
        for (int i = 0; i < M; ++i)
            EXPECT_EQ(value_at(s, i), 1.0f);
    }
}

TEST(QuantDefaults, UnacceptedWeightScaleBuildsNoSourceScale) {
    // M is about 2^31: a source scale of M elements would be gigabytes.
    constexpr int M = 0x7fffffff, N = 8, K = 64;
    const std::vector<float> wf(N, 0.25f);
    const std::vector<uint16_t> f16(N, 0x3C00);
    const std::vector<dims_t> bad
            = {{0, N}, {-1, N}, {3, N}, {1, N + 1}, {N}, {}};
    for (const auto &dims : bad) {
        auto p = w4a8_params(false);
        set_quant(p.quant_params.wei_scale, wf.data(), data_type_t::f32, dims);
        std::vector<uint8_t> hold;
        EXPECT_FALSE(mm::apply_quant_defaults(p, M, N, K, hold));
        EXPECT_TRUE(hold.empty());
        EXPECT_EQ(p.quant_params.src_scale.buff, nullptr);
    }
    // f16 is accepted only on the native BRGEMM.
    auto p = w4a8_params(false);
    set_quant(p.quant_params.wei_scale, f16.data(), data_type_t::f16, {1, N});
    std::vector<uint8_t> hold;
    EXPECT_FALSE(mm::apply_quant_defaults(p, M, N, K, hold));
    EXPECT_TRUE(hold.empty());
    ASSERT_TRUE(mm::apply_quant_defaults(p, 3, N, K, hold, /*native=*/true));
    EXPECT_EQ(p.quant_params.src_scale.dt, data_type_t::f16);
}

TEST(QuantDefaults, OtherCallsAreLeftAsGiven) {
    constexpr int M = 3, N = 8, K = 64;
    const std::vector<float> wf(N, 0.25f);
    const float given = 0.5f;
    const int8_t zp = 0;
    const auto base = [&] {
        auto p = w4a8_params(false);
        set_quant(
                p.quant_params.wei_scale, wf.data(), data_type_t::f32, {1, N});
        return p;
    };
    const auto untouched = [&](mm::matmul_params p, const char *what) {
        SCOPED_TRACE(what);
        std::vector<uint8_t> hold;
        EXPECT_FALSE(mm::apply_quant_defaults(p, M, N, K, hold));
        EXPECT_TRUE(hold.empty());
        EXPECT_EQ(p.quant_params.src_scale.buff, nullptr);
    };
    auto given_scale = base();
    set_quant(
            given_scale.quant_params.src_scale, &given, data_type_t::f32, {1});
    std::vector<uint8_t> hold;
    EXPECT_FALSE(mm::apply_quant_defaults(given_scale, M, N, K, hold));
    EXPECT_EQ(given_scale.quant_params.src_scale.buff, &given);

    auto dyn = base();
    dyn.dynamic_quant = true;
    untouched(dyn, "dynamic quantization");
    auto ggml = base();
    ggml.packing.pack_format_b = 1;
    untouched(ggml, "GGML pack");
    auto with_zp = base();
    set_quant(with_zp.quant_params.src_zp, &zp, data_type_t::s8, {1});
    untouched(with_zp, "source zero-point");
    auto to_s32 = base();
    to_s32.dtypes.dst = data_type_t::s32;
    untouched(to_s32, "s32 destination");
}

TEST(QuantDefaults, InvalidBatchBuildsNoSourceScale) {
    // M is about 2^31: building the scale first would allocate gigabytes.
    constexpr int M = 0x7fffffff, N = 8, K = 64;
    const std::vector<int8_t> src(K), wei(K * N / 2);
    const std::vector<float> wf(N, 0.25f);
    float dst[N] = {};
    auto p = w4a8_params(false);
    set_quant(p.quant_params.wei_scale, wf.data(), data_type_t::f32, {1, N});
    for (const int batch_a : {0, -1}) {
        mm::matmul_batch_params_t batch;
        batch.Batch_A = batch_a;
        batch.Batch_B = 1;
        EXPECT_NE(mm::matmul_direct('r', false, false, M, N, K, 1.0f,
                          src.data(), K, wei.data(), N, nullptr, 0.0f, dst, N,
                          true, batch, p),
                status_t::success);
    }
}

TEST(QuantDefaults, RejectedCallBuildsNoSourceScale) {
    // M is about 2^31: building the ones first would allocate gigabytes.
    constexpr int M = 0x7fffffff, N = 8, K = 64;
    const std::vector<int8_t> src(K), wei(K * N / 2);
    const std::vector<float> wf(N, 0.25f);
    float dst[N] = {};
    auto p = w4a8_params(false);
    set_quant(p.quant_params.wei_scale, wf.data(), data_type_t::f32, {1, N});
    mm::matmul_post_op clip;
    clip.po_type = zendnnl::common::post_op_type_t::clip;
    clip.alpha = 0.0f;
    clip.beta = std::numeric_limits<float>::infinity();
    p.postop_.push_back(clip);
    mm::matmul_batch_params_t batch;
    batch.Batch_A = 1;
    batch.Batch_B = 1;
    EXPECT_NE(mm::matmul_direct('r', false, false, M, N, K, 1.0f, src.data(), K,
                      wei.data(), N, nullptr, 0.0f, dst, N, true, batch, p),
            status_t::success);

    // Defaulting alone holds one placeholder value, not M ones.
    std::vector<uint8_t> hold;
    auto q = w4a8_params(false);
    set_quant(q.quant_params.wei_scale, wf.data(), data_type_t::f32, {1, N});
    ASSERT_TRUE(mm::apply_quant_defaults(q, M, N, K, hold));
    EXPECT_EQ(hold.size(), sizeof(float));
}

/// The default batch stride of several packed S4 weight matrices is out of
/// bounds, so W4A8 with Batch_B > 1 is rejected, with or without a source
/// scale, unless batch_stride_wei is 0 (one shared weight).
TEST(QuantDefaults, W4A8SeveralWeightMatricesAreRejected) {
    constexpr int M = 4, N = 16, K = 64;
    const auto src = random_s8(2 * M * K, -20, 20, 71);
    const auto wei = random_s8(2 * K * N / 2, -128, 127, 72);
    const std::vector<float> wf(N, 0.05f);
    const std::vector<float> sf(M, 0.1f);
    for (const bool give_src : {false, true})
        for (const int batch_a : {1, 2}) {
            SCOPED_TRACE(::testing::Message()
                    << "source scale " << give_src << " Batch_A " << batch_a);
            auto p = w4a8_params(false);
            set_quant(p.quant_params.wei_scale, wf.data(), data_type_t::f32,
                    {1, N});
            if (give_src)
                set_quant(p.quant_params.src_scale, sf.data(), data_type_t::f32,
                        {M, 1});
            mm::matmul_batch_params_t batch;
            batch.Batch_A = batch_a;
            batch.Batch_B = 2;
            const auto r = run(p, M, N, K, src.data(), K, wei.data(), N, false,
                    data_type_t::f32, batch);
            EXPECT_EQ(r.status, status_t::failure);
            EXPECT_EQ(r.dst.front(), -7.0f);
            EXPECT_EQ(r.dst.back(), -7.0f);
        }
}

/// The rejection comes before the default source scale is built: a call with a
/// huge M allocates none of it. The peak size is a process-wide figure, so
/// the check holds when this test runs alone. It is read with getrusage, so
/// native Windows checks the status and the untouched output only.
TEST(QuantDefaults, RejectedWeightBatchBuildsNoSourceScale) {
    constexpr int M = 1 << 30, N = 16, K = 64;
    const std::vector<float> wf(N, 0.05f);
    const std::vector<int8_t> src(K, 1), wei(2 * K * N / 2, 1);
    std::vector<float> dst(N, -7.0f);
    auto p = w4a8_params(false);
    set_quant(p.quant_params.wei_scale, wf.data(), data_type_t::f32, {1, N});
    mm::matmul_batch_params_t batch;
    batch.Batch_A = 2;
    batch.Batch_B = 2;
#ifndef _WIN32
    rusage before {}, after {};
    getrusage(RUSAGE_SELF, &before);
#endif
    const status_t st = mm::matmul_direct('r', false, false, M, N, K, 1.0f,
            src.data(), K, wei.data(), N, nullptr, 0.0f, dst.data(), N, true,
            batch, p);
#ifndef _WIN32
    getrusage(RUSAGE_SELF, &after);
#endif
    EXPECT_EQ(st, status_t::failure);
    EXPECT_EQ(dst.front(), -7.0f);
#ifndef _WIN32
    EXPECT_LT(after.ru_maxrss - before.ru_maxrss, 256 * 1024); // KiB
#endif
}

/// Weights shared by every batch (batch_stride_wei = 0) are served: each batch
/// matches a call on its own source rows, with the source scale omitted or
/// given.
TEST(QuantDefaults, W4A8SharedWeightBatchIsServed) {
    constexpr int M = 8, N = 24, K = 128, B = 2;
    const auto src = random_s8(B * M * K, -20, 20, 91);
    const auto wei = random_s8(K * N / 2, -128, 127, 92);
    const auto wf = random_f32(N, 0.01f, 0.1f, 93);
    const auto ones = ones_of(data_type_t::f32, M);
    auto whole = w4a8_params(false);
    set_quant(
            whole.quant_params.wei_scale, wf.data(), data_type_t::f32, {1, N});
    set_quant(whole.quant_params.src_scale, ones.data(), data_type_t::f32,
            {M, 1});
    for (const bool give_src : {false, true})
        for (const int threads : {1, 8}) {
            SCOPED_TRACE(::testing::Message()
                    << "source scale " << give_src << " threads " << threads);
            auto p = w4a8_params(false);
            p.num_threads = threads;
            set_quant(p.quant_params.wei_scale, wf.data(), data_type_t::f32,
                    {1, N});
            if (give_src)
                set_quant(p.quant_params.src_scale, ones.data(),
                        data_type_t::f32, {M, 1});
            mm::matmul_batch_params_t batch;
            batch.Batch_A = B;
            batch.Batch_B = B;
            batch.batch_stride_wei = 0;
            const auto got = run(p, M, N, K, src.data(), K, wei.data(), N,
                    false, data_type_t::f32, batch);
            ASSERT_EQ(got.status, status_t::success);
            for (int b = 0; b < B; ++b) {
                const auto want = run(whole, M, N, K,
                        src.data() + static_cast<size_t>(b) * M * K, K,
                        wei.data(), N, false, data_type_t::f32);
                ASSERT_EQ(want.status, status_t::success);
                for (size_t i = 0; i < want.dst.size(); ++i)
                    ASSERT_NEAR(got.dst[static_cast<size_t>(b) * M * N + i],
                            want.dst[i], 1e-3f * (1.0f + std::abs(want.dst[i])))
                            << "batch " << b << " at " << i;
            }
        }
}

/// The reference kernel converts the shared weight matrix once, so it has to
/// keep the zero batch stride: every batch reads that one matrix, not memory
/// after it. Each batch matches a reference call on its own source rows.
TEST(QuantDefaults, W4A8SharedWeightBatchOnReferenceKernel) {
    constexpr int M = 8, N = 24, K = 128, B = 3;
    const auto src = random_s8(B * M * K, -20, 20, 101);
    const auto wei = random_s8(K * N / 2, -128, 127, 102);
    const auto wf = random_f32(N, 0.01f, 0.1f, 103);
    const auto sf = random_f32(M, 0.01f, 0.1f, 104);
    auto whole = w4a8_params(false);
    whole.lowoha_algo = matmul_algo_t::reference;
    set_quant(
            whole.quant_params.wei_scale, wf.data(), data_type_t::f32, {1, N});
    set_quant(
            whole.quant_params.src_scale, sf.data(), data_type_t::f32, {M, 1});
    for (const bool give_src : {false, true})
        for (const bool trans_b : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "source scale " << give_src
                                              << " transposed " << trans_b);
            const auto ones = ones_of(data_type_t::f32, M);
            auto p = w4a8_params(false);
            p.lowoha_algo = matmul_algo_t::reference;
            set_quant(p.quant_params.wei_scale, wf.data(), data_type_t::f32,
                    {1, N});
            if (give_src)
                set_quant(p.quant_params.src_scale, sf.data(), data_type_t::f32,
                        {M, 1});
            mm::matmul_batch_params_t batch;
            batch.Batch_A = B;
            batch.Batch_B = B;
            batch.batch_stride_wei = 0;
            const int ldb = trans_b ? K : N;
            const auto got = run(p, M, N, K, src.data(), K, wei.data(), ldb,
                    trans_b, data_type_t::f32, batch);
            ASSERT_EQ(got.status, status_t::success);
            auto each = whole;
            if (!give_src)
                set_quant(each.quant_params.src_scale, ones.data(),
                        data_type_t::f32, {M, 1});
            for (int b = 0; b < B; ++b) {
                const auto want = run(each, M, N, K,
                        src.data() + static_cast<size_t>(b) * M * K, K,
                        wei.data(), ldb, trans_b, data_type_t::f32);
                ASSERT_EQ(want.status, status_t::success);
                for (size_t i = 0; i < want.dst.size(); ++i)
                    ASSERT_NEAR(got.dst[static_cast<size_t>(b) * M * N + i],
                            want.dst[i], 1e-3f * (1.0f + std::abs(want.dst[i])))
                            << "batch " << b << " at " << i;
            }
        }
}

// ── validator: prepacked sym-quant weights ─────────────────────────────

TEST(QuantDefaultsPrepacked, SymQuantPrepackNeedsItsWeightScale) {
    constexpr int N = 16, K = 64;
    const float one = 1.0f;
    const std::vector<float> wei_scale(2 * N, 1.0f);
    const std::vector<int8_t> src(K, 1), wei(K * N, 1);
    std::vector<float> dst(N, 0.0f);
    const auto check
            = [&](int M, data_type_t src_dt, dims_t src_dims,
                      bool with_wei_scale, status_t want, const char *what) {
        SCOPED_TRACE(what);
        mm::matmul_params p;
        p.dtypes.src = src_dt;
        p.dtypes.wei = data_type_t::s8;
        p.dtypes.dst = data_type_t::f32;
        p.mem_format_b = 'r';
        p.lowoha_algo = matmul_algo_t::aocl_dlp_blocked;
        set_quant(p.quant_params.src_scale, &one, data_type_t::f32,
                std::move(src_dims));
        if (with_wei_scale)
            set_quant(p.quant_params.wei_scale, wei_scale.data(),
                    data_type_t::f32, {2, N});
        EXPECT_EQ(mm::validate_matmul_direct_inputs(src.data(), wei.data(),
                          dst.data(), M, N, K, 1, 1, p, true),
                want);
    };
    check(1, data_type_t::s8, {1, 1}, false, status_t::success,
            "{1, 1} at M = 1, no weight scale: a classic prepack");
    check(1, data_type_t::s8, {1, 1}, true, status_t::success,
            "per-token {1, 1} at M = 1, with the weight scale");
    check(1, data_type_t::s8, {1}, false, status_t::success, "per-tensor {1}");
    check(1, data_type_t::s8, {}, false, status_t::success, "per-tensor {}");
    check(1, data_type_t::u8, {1, 1}, false, status_t::success, "u8 source");
    check(2, data_type_t::s8, {2, 1}, false, status_t::failure,
            "per-token {2, 1}, no weight scale");
}

// ── end to end: static W4A8 ────────────────────────────────────────────

/// A W4A8 call on N = 32, K = 128 with a {G, N} weight scale. The source
/// scale is omitted or given as ones; the weights are raw or prepacked.
run_t run_w4a8(bool packed, bool give_src, bool give_wei, int M, int G,
        data_type_t scale_dt, data_type_t dst_dt, unsigned seed) {
    constexpr int N = 32, K = 128;
    const auto src = random_s8(static_cast<size_t>(M) * K, -20, 20, seed);
    const auto raw
            = random_s8(static_cast<size_t>(N) * K / 2, -128, 127, seed + 1);
    const auto wf
            = random_f32(static_cast<size_t>(G) * N, 0.01f, 0.1f, seed + 2);
    std::vector<uint16_t> wb;
    for (const float x : wf)
        wb.push_back(to_bf16(x));
    std::vector<uint8_t> pk;
    if (packed) {
        pk = prepack(raw.data(), data_type_t::s4, data_type_t::s8, K, N,
                /*transposed=*/true, K / G);
        if (pk.empty()) return {};
    }
    auto p = w4a8_params(packed);
    p.dtypes.dst = dst_dt;
    if (give_wei)
        set_quant(p.quant_params.wei_scale,
                scale_dt == data_type_t::bf16
                        ? static_cast<const void *>(wb.data())
                        : static_cast<const void *>(wf.data()),
                scale_dt, {G, N});
    const auto ones = ones_of(scale_dt, M);
    if (give_src)
        set_quant(p.quant_params.src_scale, ones.data(), scale_dt, {M, 1});
    const void *w = packed ? static_cast<const void *>(pk.data())
                           : static_cast<const void *>(raw.data());
    return run(p, M, N, K, src.data(), K, w, K, /*transB=*/true, dst_dt);
}

TEST(QuantDefaultsPrepacked, OmittedSourceScaleComputesWhatOnesCompute) {
    for (const int M : {1, 7})
        for (const int G : {1, 4})
            for (const data_type_t sdt : {data_type_t::f32, data_type_t::bf16})
                for (const data_type_t ddt :
                        {data_type_t::f32, data_type_t::bf16}) {
                    SCOPED_TRACE(::testing::Message()
                            << "M " << M << " G " << G << " scale "
                            << (sdt == data_type_t::f32 ? "f32" : "bf16")
                            << " dst "
                            << (ddt == data_type_t::f32 ? "f32" : "bf16"));
                    const auto want
                            = run_w4a8(false, true, true, M, G, sdt, ddt, 5);
                    ASSERT_EQ(want.status, status_t::success) << "raw, ones";
                    const auto raw
                            = run_w4a8(false, false, true, M, G, sdt, ddt, 5);
                    ASSERT_EQ(raw.status, status_t::success) << "raw, omitted";
                    const auto pk_ones
                            = run_w4a8(true, true, true, M, G, sdt, ddt, 5);
                    if (pk_ones.dst.empty())
                        GTEST_SKIP() << "S4 AOCL prepack is unavailable";
                    const auto pk
                            = run_w4a8(true, false, true, M, G, sdt, ddt, 5);
                    EXPECT_EQ(pk_ones.status, status_t::success);
                    EXPECT_EQ(pk.status, status_t::success)
                            << "packed, omitted";
                    for (size_t i = 0; i < want.dst.size(); ++i) {
                        const float tol
                                = 1e-2f * (1.0f + std::abs(want.dst[i]));
                        EXPECT_NEAR(raw.dst[i], want.dst[i], tol) << i;
                        EXPECT_NEAR(pk_ones.dst[i], want.dst[i], tol) << i;
                        EXPECT_NEAR(pk.dst[i], want.dst[i], tol) << i;
                    }
                }
}

TEST(QuantDefaultsPrepacked, MissingWeightScaleIsRejected) {
    for (const int M : {1, 7})
        for (const bool packed : {false, true})
            for (const bool src : {false, true}) {
                SCOPED_TRACE(::testing::Message()
                        << "M " << M << (packed ? " packed" : " raw")
                        << (src ? " with source scale" : ""));
                const auto r = run_w4a8(packed, src, false, M, 1,
                        data_type_t::f32, data_type_t::f32, 9);
                if (r.dst.empty())
                    GTEST_SKIP() << "S4 AOCL prepack is unavailable";
                EXPECT_EQ(r.status, status_t::failure);
                EXPECT_EQ(r.dst.front(), -7.0f);
            }
}

/// A batched call split into M tiles across threads reads each tile's rows of
/// the default {M, 1} source scale, so it matches one call per batch.
TEST(QuantDefaultsBatched, OmittedSourceScaleFollowsEachTile) {
    constexpr int M = 8, N = 24, K = 128, B = 2;
    const auto src = random_s8(B * M * K, -20, 20, 81);
    const auto wei = random_s8(K * N / 2, -128, 127, 82);
    const auto wf = random_f32(N, 0.01f, 0.1f, 84);
    const auto ones = ones_of(data_type_t::f32, M);
    for (const int threads : {1, 8}) {
        SCOPED_TRACE(threads);
        auto p = w4a8_params(false);
        p.num_threads = threads;
        set_quant(
                p.quant_params.wei_scale, wf.data(), data_type_t::f32, {1, N});
        mm::matmul_batch_params_t batch;
        batch.Batch_A = B;
        batch.Batch_B = 1;
        const auto got = run(p, M, N, K, src.data(), K, wei.data(), N, false,
                data_type_t::f32, batch);
        ASSERT_EQ(got.status, status_t::success);

        auto whole = w4a8_params(false);
        set_quant(whole.quant_params.wei_scale, wf.data(), data_type_t::f32,
                {1, N});
        set_quant(whole.quant_params.src_scale, ones.data(), data_type_t::f32,
                {M, 1});
        for (int b = 0; b < B; ++b) {
            const auto want = run(whole, M, N, K,
                    src.data() + static_cast<size_t>(b) * M * K, K, wei.data(),
                    N, false, data_type_t::f32);
            ASSERT_EQ(want.status, status_t::success);
            for (size_t i = 0; i < want.dst.size(); ++i)
                ASSERT_NEAR(got.dst[static_cast<size_t>(b) * M * N + i],
                        want.dst[i], 1e-3f * (1.0f + std::abs(want.dst[i])))
                        << "batch " << b << " at " << i;
        }
    }
}

/// Explicit, nonuniform source scales of several forms, in batched calls split
/// into M tiles (with tails and one-row tiles), match one call per batch.
TEST(QuantDefaultsBatched, SourceScaleRowsFollowEachTile) {
    constexpr int N = 24, K = 128, B = 2, G = 4;
    struct form_t {
        const char *name;
        data_type_t wei;
        int cols;
        dims_t wei_dims;
    };
    const std::vector<form_t> forms = {
            {"W4A8 {M, 1}", data_type_t::s4, 1, {1, N}},
            {"W4A8 {M, G}", data_type_t::s4, G, {G, N}},
            {"s8 {M, 1}, {1, N}", data_type_t::s8, 1, {1, N}},
            {"s8 {M, 1}, {N}", data_type_t::s8, 1, {N}},
            {"s8 {M, G}, {G, N}", data_type_t::s8, G, {G, N}},
    };
    for (const int M : {4, 7})
        for (const auto &f : forms) {
            SCOPED_TRACE(::testing::Message() << f.name << " M " << M);
            const bool s4 = f.wei == data_type_t::s4;
            const auto src
                    = random_s8(static_cast<size_t>(B) * M * K, -20, 20, 91);
            const auto wei = s4 ? random_s8(K * N / 2, -128, 127, 92)
                                : random_s8(K * N, -20, 20, 92);
            const auto src_scale = random_f32(
                    static_cast<size_t>(M) * f.cols, 0.01f, 0.1f, 93);
            const auto wei_scale
                    = random_f32(static_cast<size_t>(f.cols == G ? G : 1) * N,
                            0.01f, 0.1f, 94);
            auto p = w4a8_params(false);
            if (!s4) {
                p.dtypes.wei = data_type_t::s8;
                p.dtypes.compute = data_type_t::none;
            }
            set_quant(p.quant_params.src_scale, src_scale.data(),
                    data_type_t::f32, {M, f.cols});
            set_quant(p.quant_params.wei_scale, wei_scale.data(),
                    data_type_t::f32, f.wei_dims);
            auto whole = p;
            p.num_threads = 8;
            mm::matmul_batch_params_t batch;
            batch.Batch_A = B;
            batch.Batch_B = 1;
            const auto got = run(p, M, N, K, src.data(), K, wei.data(), N,
                    false, data_type_t::f32, batch);
            ASSERT_EQ(got.status, status_t::success);
            whole.num_threads = 1;
            for (int b = 0; b < B; ++b) {
                const auto want = run(whole, M, N, K,
                        src.data() + static_cast<size_t>(b) * M * K, K,
                        wei.data(), N, false, data_type_t::f32);
                ASSERT_EQ(want.status, status_t::success);
                for (size_t i = 0; i < want.dst.size(); ++i)
                    ASSERT_NEAR(got.dst[static_cast<size_t>(b) * M * N + i],
                            want.dst[i], 1e-3f * (1.0f + std::abs(want.dst[i])))
                            << "batch " << b << " at " << i;
            }
        }
}

/// A batched dynamic quantization writes one source scale per row, {1, M, 1},
/// shared by the batch; each M tile reads its own rows. An {M, 1} scale is not
/// covered: main's batched reorder does not quantize that form. Rows differ in
/// magnitude, so another row's scale is far off the quantization step.
TEST(QuantDefaultsBatched, DynamicSourceScaleRowsFollowEachTile) {
    constexpr int N = 24, K = 128, B = 2;
    for (const data_type_t src_dt : {data_type_t::f32, data_type_t::bf16})
        for (const int M : {4, 7})
            for (int form = 0; form < 2; ++form) {
                const dims_t wei_dims = form ? dims_t {N} : dims_t {1, N};
                const dims_t src_dims = {1, M, 1};
                SCOPED_TRACE(::testing::Message()
                        << (src_dt == data_type_t::f32 ? "f32" : "bf16")
                        << " M " << M << " wei dims " << wei_dims.size()
                        << " src dims " << src_dims.size());
                auto src_f32 = random_f32(
                        static_cast<size_t>(B) * M * K, -1.0f, 1.0f, 95);
                std::vector<uint16_t> src_bf16;
                for (size_t i = 0; i < src_f32.size(); ++i) {
                    src_f32[i] *= static_cast<float>(1 + (i / K) % M * 3);
                    src_bf16.push_back(to_bf16(src_f32[i]));
                    if (src_dt == data_type_t::bf16)
                        src_f32[i] = from_bf16(src_bf16.back());
                }
                const auto wei = random_s8(K * N, -20, 20, 96);
                const auto wei_scale = random_f32(N, 0.01f, 0.1f, 97);
                std::vector<float> src_scale(M);
                mm::matmul_params p;
                p.dtypes.src = src_dt;
                p.dtypes.wei = data_type_t::s8;
                p.dtypes.dst = data_type_t::f32;
                p.dtypes.compute = data_type_t::s8;
                p.dynamic_quant = true;
                p.num_threads = 8;
                set_quant(p.quant_params.src_scale, src_scale.data(),
                        data_type_t::f32, src_dims);
                set_quant(p.quant_params.wei_scale, wei_scale.data(),
                        data_type_t::f32, wei_dims);
                mm::matmul_batch_params_t batch;
                batch.Batch_A = B;
                batch.Batch_B = 1;
                const auto got = run(p, M, N, K,
                        src_dt == data_type_t::f32
                                ? static_cast<const void *>(src_f32.data())
                                : static_cast<const void *>(src_bf16.data()),
                        K, wei.data(), N, false, data_type_t::f32, batch);
                ASSERT_EQ(got.status, status_t::success);
                for (int m = 0; m < M; ++m) {
                    float row_max = 0.0f;
                    for (int b = 0; b < B; ++b)
                        for (int k = 0; k < K; ++k)
                            row_max = std::max(row_max,
                                    std::abs(src_f32[(static_cast<size_t>(b) * M
                                                             + m)
                                                    * K
                                            + k]));
                    for (int b = 0; b < B; ++b)
                        for (int n = 0; n < N; ++n) {
                            double want = 0.0, wei_abs = 0.0;
                            for (int k = 0; k < K; ++k) {
                                const int w
                                        = wei[static_cast<size_t>(k) * N + n];
                                want += static_cast<double>(
                                                src_f32[(static_cast<size_t>(b)
                                                                        * M
                                                                + m)
                                                                * K
                                                        + k])
                                        * w;
                                wei_abs += std::abs(w);
                            }
                            want *= wei_scale[n];
                            const double step
                                    = row_max / 127.0 * wei_abs * wei_scale[n];
                            ASSERT_NEAR(
                                    got.dst[(static_cast<size_t>(b) * M + m) * N
                                            + n],
                                    want, step + 1e-3)
                                    << "batch " << b << " row " << m << " col "
                                    << n;
                        }
                }
            }
}

// ── end to end: classic int8 ───────────────────────────────────────────

/// An omitted per-tensor source scale of a classic int8 call is read as 1: it
/// computes what a source scale of 1 computes, on raw and prepacked weights,
/// for u8 and s8 sources, into f32, bf16 and s32, with per-tensor or
/// per-channel weight scales.
TEST(QuantDefaultsInt8, OmittedSourceScaleIsOne) {
    constexpr int N = 32, K = 128;
    const auto wei = random_s8(static_cast<size_t>(K) * N, -8, 7, 22);
    const auto wf = random_f32(N, 0.01f, 0.1f, 23);
    const float wone = 0.05f, one = 1.0f;
    for (const bool packed : {false, true})
        for (const data_type_t sdt : {data_type_t::s8, data_type_t::u8})
            for (const int M : {1, 7})
                for (const data_type_t ddt :
                        {data_type_t::f32, data_type_t::bf16, data_type_t::s32})
                    for (const bool per_channel : {false, true}) {
                        SCOPED_TRACE(::testing::Message()
                                << (packed ? "packed " : "raw ")
                                << (sdt == data_type_t::s8 ? "s8" : "u8")
                                << " M " << M << " dst "
                                << static_cast<int>(ddt)
                                << (per_channel ? " per-channel"
                                                : " per-tensor"));
                        const auto src8 = random_s8(static_cast<size_t>(M) * K,
                                sdt == data_type_t::u8 ? 0 : -8, 7, 21);
                        std::vector<uint8_t> src_u8(src8.size());
                        for (size_t k = 0; k < src8.size(); ++k)
                            src_u8[k] = static_cast<uint8_t>(src8[k]);
                        const void *src = sdt == data_type_t::u8
                                ? static_cast<const void *>(src_u8.data())
                                : static_cast<const void *>(src8.data());
                        const auto pk = packed
                                ? prepack(wei.data(), data_type_t::s8, sdt, K,
                                          N, /*transposed=*/false, 0)
                                : std::vector<uint8_t>();
                        if (packed && pk.empty())
                            GTEST_SKIP() << "int8 AOCL prepack is unavailable";
                        const void *w = packed
                                ? static_cast<const void *>(pk.data())
                                : static_cast<const void *>(wei.data());

                        auto p = mm::matmul_params();
                        p.dtypes.src = sdt;
                        p.dtypes.wei = data_type_t::s8;
                        p.dtypes.dst = ddt;
                        p.weight_cache_type = 1;
                        if (packed) {
                            p.mem_format_b = 'r';
                            p.lowoha_algo = matmul_algo_t::aocl_dlp_blocked;
                        }
                        if (per_channel)
                            set_quant(p.quant_params.wei_scale, wf.data(),
                                    data_type_t::f32, {1, N});
                        else
                            set_quant(p.quant_params.wei_scale, &wone,
                                    data_type_t::f32, {1, 1});
                        // On raw weights the reference's kernel makes the
                        // layout it reads.
                        auto ones_form = p;
                        ones_form.mem_format_b = 'n';
                        set_quant(ones_form.quant_params.src_scale, &one,
                                data_type_t::f32, {1, 1});
                        const auto want = run(ones_form, M, N, K, src, K,
                                wei.data(), N, false, ddt);
                        const auto got
                                = run(p, M, N, K, src, K, w, N, false, ddt);
                        ASSERT_EQ(want.status, status_t::success) << "ones";
                        ASSERT_EQ(got.status, status_t::success) << "omitted";
                        for (size_t k = 0; k < want.dst.size(); ++k)
                            EXPECT_NEAR(got.dst[k], want.dst[k],
                                    1e-3f * (1.0f + std::abs(want.dst[k])))
                                    << "at " << k;
                    }
}

/// A per-tensor source scale other than 1 ({}, {1}, {1, 1}) on classic int8
/// weights, raw or prepacked, at one row and at several, with a per-tensor,
/// per-channel or no weight scale, computes src_scale * wei_scale * (A x B).
TEST(QuantDefaultsInt8, PerTensorSourceScaleOnClassicPrepack) {
    constexpr int N = 32, K = 128;
    const auto wei = random_s8(static_cast<size_t>(K) * N, -8, 7, 32);
    const auto wf = random_f32(N, 0.01f, 0.1f, 33);
    const float s = 0.5f, wone = 0.05f;
    for (const data_type_t sdt : {data_type_t::s8, data_type_t::u8}) {
        const auto pk = prepack(wei.data(), data_type_t::s8, sdt, K, N,
                /*transposed=*/false, 0);
        if (pk.empty()) GTEST_SKIP() << "int8 AOCL prepack is unavailable";
        for (const int M : {1, 7}) {
            const auto src8 = random_s8(static_cast<size_t>(M) * K,
                    sdt == data_type_t::u8 ? 0 : -8, 7, 31);
            std::vector<uint8_t> src_u8(src8.size());
            for (size_t k = 0; k < src8.size(); ++k)
                src_u8[k] = static_cast<uint8_t>(src8[k]);
            const void *src = sdt == data_type_t::u8
                    ? static_cast<const void *>(src_u8.data())
                    : static_cast<const void *>(src8.data());
            for (const dims_t &sd : {dims_t {}, dims_t {1}, dims_t {1, 1}})
                for (const int wmode :
                        {0, 1, 2}) // none, per-tensor, per-channel
                    for (const bool packed : {false, true}) {
                        SCOPED_TRACE(::testing::Message()
                                << (sdt == data_type_t::s8 ? "s8" : "u8")
                                << " M " << M << " src dims " << sd.size()
                                << " wei mode " << wmode
                                << (packed ? " packed" : " raw"));
                        mm::matmul_params p;
                        p.dtypes.src = sdt;
                        p.dtypes.wei = data_type_t::s8;
                        p.dtypes.dst = data_type_t::f32;
                        p.weight_cache_type = 1;
                        if (packed) {
                            p.mem_format_b = 'r';
                            p.lowoha_algo = matmul_algo_t::aocl_dlp_blocked;
                        }
                        set_quant(p.quant_params.src_scale, &s,
                                data_type_t::f32, sd);
                        if (wmode == 1)
                            set_quant(p.quant_params.wei_scale, &wone,
                                    data_type_t::f32, {1, 1});
                        // {1, N} at one row selects the sym-quant kernel, whose
                        // pack is not the classic one: a classic prepack takes {N}.
                        if (wmode == 2)
                            set_quant(p.quant_params.wei_scale, wf.data(),
                                    data_type_t::f32,
                                    packed ? dims_t {N} : dims_t {1, N});
                        const auto got = run(p, M, N, K, src, K,
                                packed ? static_cast<const void *>(pk.data())
                                       : static_cast<const void *>(wei.data()),
                                N, false, data_type_t::f32);
                        ASSERT_EQ(got.status, status_t::success);
                        for (int m = 0; m < M; ++m)
                            for (int n = 0; n < N; ++n) {
                                double acc = 0;
                                for (int k = 0; k < K; ++k)
                                    acc += (sdt == data_type_t::u8
                                                           ? src_u8[m * K + k]
                                                           : src8[m * K + k])
                                            * wei[k * N + n];
                                const double w = wmode == 0 ? 1.0
                                        : wmode == 1        ? wone
                                                            : wf[n];
                                const float want
                                        = static_cast<float>(acc * s * w);
                                ASSERT_NEAR(got.dst[m * N + n], want,
                                        1e-3f * (1.0f + std::abs(want)))
                                        << "at " << m << ", " << n;
                            }
                    }
        }
    }
}

/// At one row, a per-tensor source scale with a {1, N} weight scale selects
/// the sym-quant kernel, so its prepack is the sym-quant one (one group).
TEST(QuantDefaultsInt8, PerTensorSourceScaleOnSymQuantPrepackAtOneRow) {
    constexpr int N = 32, K = 128;
    const auto wei = random_s8(static_cast<size_t>(K) * N, -8, 7, 42);
    const auto wf = random_f32(N, 0.01f, 0.1f, 43);
    const auto src = random_s8(K, -8, 7, 41);
    const float s = 0.5f;
    const auto pk = prepack(wei.data(), data_type_t::s8, data_type_t::s8, K, N,
            /*transposed=*/false, K);
    if (pk.empty()) GTEST_SKIP() << "sym-quant AOCL prepack is unavailable";
    for (const dims_t &sd : {dims_t {1}, dims_t {1, 1}}) {
        mm::matmul_params p;
        p.dtypes.src = data_type_t::s8;
        p.dtypes.wei = data_type_t::s8;
        p.dtypes.dst = data_type_t::f32;
        p.weight_cache_type = 1;
        p.mem_format_b = 'r';
        p.lowoha_algo = matmul_algo_t::aocl_dlp_blocked;
        set_quant(p.quant_params.src_scale, &s, data_type_t::f32, sd);
        set_quant(
                p.quant_params.wei_scale, wf.data(), data_type_t::f32, {1, N});
        const auto got = run(p, 1, N, K, src.data(), K, pk.data(), N, false,
                data_type_t::f32);
        ASSERT_EQ(got.status, status_t::success);
        for (int n = 0; n < N; ++n) {
            double acc = 0;
            for (int k = 0; k < K; ++k)
                acc += src[k] * wei[k * N + n];
            const float want = static_cast<float>(acc * s * wf[n]);
            ASSERT_NEAR(got.dst[n], want, 1e-3f * (1.0f + std::abs(want)))
                    << "src dims " << sd.size() << " at " << n;
        }
    }
}

} // namespace
