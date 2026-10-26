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

/// The native layout rule, the native pack bytes and the ALGO 11 route of
/// matmul_direct.  A routed call is compared with a direct
/// native_brgemm_execute call on the weight reference the route must build: the
/// same status, the same bytes and a layout the native BRGEMM keeps for the
/// weight when it is served, and no other backend writing dst when it is
/// not.

#include <omp.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <type_traits>

#include "gtest_utils.hpp"
#include "lowoha_operators/matmul/aocl_unreorder_cache.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/dispatch.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/pack.hpp"
#include "lowoha_operators/matmul/group_matmul/single_expert/single_expert_route.hpp"
#include "lowoha_operators/matmul/lowoha_matmul.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/int8/native_int8_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_brgemm_looper.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_aocl_unreorder.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_pack.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_weight_cache.hpp"
#include "lowoha_operators/matmul/matmul_native/native_layout_policy.hpp"
#include "lowoha_operators/matmul/matmul_native/native_matmul.hpp"
#include "lowoha_operators/reorder/lowoha_reorder.hpp"
#include "moe_test_utils.hpp"

namespace {

namespace mm = zendnnl::lowoha::matmul;
namespace native = zendnnl::lowoha::matmul::native;
namespace ck = zendnnl::lowoha::matmul::custom_kernel;
namespace rdr = zendnnl::lowoha::reorder;
namespace mt = moe_test_utils;
using mm::matmul_gated_act_t;
using mm::matmul_params;
using zendnnl::common::bfloat16_t;
using zendnnl::common::data_type_t;
using zendnnl::common::matmul_algo_t;
using zendnnl::common::matmul_config_t;
using zendnnl::common::post_op_type_t;
using zendnnl::common::size_of;
using zendnnl::error_handling::status_t;

/// Whether the buffer of v holds an in-place image, checked within it.
template <typename V>
bool packed_in_place(const V &v) {
    return native::native_weight_packed_in_place(
            v.data(), v.size() * sizeof(*v.data()));
}

constexpr int kNr = native::kNativeLayoutPackNr;
constexpr int kThreads = 4;
constexpr int32_t kAlgoNone = static_cast<int32_t>(matmul_algo_t::none);
constexpr int32_t kAlgoAocl
        = static_cast<int32_t>(matmul_algo_t::aocl_dlp_blocked);
constexpr int32_t kAlgoNative
        = static_cast<int32_t>(matmul_algo_t::native_brgemm);

bool has_bf16() {
    return native::detect_uarch().avx512bf16;
}
bool has_vnni() {
    return native::detect_uarch().avx512vnni;
}

native::native_layout_query_t query(data_type_t wei, data_type_t src, int K,
        int N, int ldb = 0, bool transB = false, int group = 0) {
    native::native_layout_query_t q;
    q.wei_dt = wei;
    q.src_dt = src;
    q.K = K;
    q.N = N;
    q.ldb = ldb > 0 ? ldb : (transB ? K : N);
    q.transB = transB;
    q.wei_group_size = group;
    return q;
}

bool isa_for(data_type_t wei) {
    return wei == data_type_t::bf16 ? has_bf16() : has_vnni();
}

void expect_native(const native::native_layout_query_t &q) {
    const native::native_layout_decision_t d = native::decide_native_layout(q);
    if (!isa_for(q.wei_dt)) {
        EXPECT_FALSE(d.native) << "no ISA for this weight type";
        return;
    }
    EXPECT_TRUE(d.native) << d.reason << " (K=" << q.K << " N=" << q.N
                          << " ldb=" << q.ldb << ")";
    EXPECT_EQ(d.pack_nr, kNr);
    EXPECT_STREQ(d.reason, "");
}

void expect_aocl(const native::native_layout_query_t &q) {
    const native::native_layout_decision_t d = native::decide_native_layout(q);
    EXPECT_FALSE(d.native) << "K=" << q.K << " N=" << q.N << " ldb=" << q.ldb
                           << " group=" << q.wei_group_size;
    EXPECT_EQ(d.pack_nr, 0);
    EXPECT_STRNE(d.reason, "");
}

// ── Layout rule ─────────────────────────────────────────────────────

TEST(NativeLayoutRule, Bf16WeightNeedsBf16Source) {
    expect_native(query(data_type_t::bf16, data_type_t::bf16, 64, 64));
    for (data_type_t src : {data_type_t::f32, data_type_t::f16, data_type_t::s8,
                 data_type_t::u8, data_type_t::none}) {
        expect_aocl(query(data_type_t::bf16, src, 64, 64));
    }
}

TEST(NativeLayoutRule, S8WeightNeedsU8OrS8Source) {
    expect_native(query(data_type_t::s8, data_type_t::u8, 64, 64));
    expect_native(query(data_type_t::s8, data_type_t::s8, 64, 64));
    for (data_type_t src : {data_type_t::bf16, data_type_t::f32,
                 data_type_t::f16, data_type_t::s4, data_type_t::none}) {
        expect_aocl(query(data_type_t::s8, src, 64, 64));
    }
}

TEST(NativeLayoutRule, OtherWeightTypesStayOnAocl) {
    expect_aocl(query(data_type_t::f32, data_type_t::f32, 64, 64));
    expect_aocl(query(data_type_t::f16, data_type_t::f16, 64, 64));
    expect_aocl(query(data_type_t::s4, data_type_t::s8, 64, 64));
    expect_aocl(query(data_type_t::s4, data_type_t::bf16, 64, 64));
    expect_aocl(query(data_type_t::u4, data_type_t::bf16, 64, 64));
    expect_aocl(query(data_type_t::none, data_type_t::bf16, 64, 64));
}

TEST(NativeLayoutRule, NIsAMultipleOf32) {
    for (data_type_t wei : {data_type_t::bf16, data_type_t::s8}) {
        const data_type_t src = wei == data_type_t::bf16 ? data_type_t::bf16
                                                         : data_type_t::u8;
        for (int N : {32, 64, 96, 4096, 11008})
            expect_native(query(wei, src, 64, N));
        for (int N : {-32, 0, 1, 16, 31, 33, 48, 100, 4095})
            expect_aocl(query(wei, src, 64, N, 4096));
    }
}

TEST(NativeLayoutRule, KIsPositiveAndS8KIsBounded) {
    for (data_type_t wei : {data_type_t::bf16, data_type_t::s8}) {
        const data_type_t src = wei == data_type_t::bf16 ? data_type_t::bf16
                                                         : data_type_t::s8;
        for (int K : {1, 2, 3, 31, 63, 100, 4096})
            expect_native(query(wei, src, K, 64));
        for (int K : {0, -1})
            expect_aocl(query(wei, src, K, 64));
    }
    constexpr int kBound = native::kMaxSafeInt8AccumK;
    expect_native(query(data_type_t::s8, data_type_t::u8, kBound, 64));
    expect_aocl(query(data_type_t::s8, data_type_t::u8, kBound + 1, 64));
    expect_native(query(data_type_t::bf16, data_type_t::bf16, kBound + 1, 64));
}

TEST(NativeLayoutRule, S8PerGroupWeightScalesNeedWholeQuadGroups) {
    constexpr int K = 192;
    expect_native(query(data_type_t::s8, data_type_t::u8, K, 64, 0, false, 0));
    expect_native(query(data_type_t::s8, data_type_t::u8, K, 64, 0, false, K));
    for (int group : {4, 12, 32, 64, 96})
        expect_native(query(
                data_type_t::s8, data_type_t::s8, K, 64, 0, false, group));
    for (int group : {-1, 1, 2, 6, 10, 36, 128})
        expect_aocl(query(
                data_type_t::s8, data_type_t::s8, K, 64, 0, false, group));
    expect_native(
            query(data_type_t::bf16, data_type_t::bf16, K, 64, 0, false, 6));
}

TEST(NativeLayoutRule, LdbCoversTheWeightRow) {
    constexpr int K = 64, N = 96;
    for (data_type_t wei : {data_type_t::bf16, data_type_t::s8}) {
        const data_type_t src = wei == data_type_t::bf16 ? data_type_t::bf16
                                                         : data_type_t::u8;
        expect_native(query(wei, src, K, N, N, false));
        expect_native(query(wei, src, K, N, N + 5, false));
        expect_aocl(query(wei, src, K, N, N - 1, false));
        expect_native(query(wei, src, K, N, K, true));
        expect_native(query(wei, src, K, N, K + 3, true));
        expect_aocl(query(wei, src, K, N, K - 1, true));
    }
}

// ── Native pack bytes ───────────────────────────────────────────────

rdr::reorder_params_t prepack_request(matmul_algo_t algo, data_type_t wei,
        data_type_t src, int K, int N, int ldb, bool transB, bool interleave,
        int pack_nr = 0, int group = 0) {
    rdr::reorder_params_t rp;
    rp.is_prepack = true;
    rp.prepack.algo = algo;
    rp.prepack.wei_dtype = wei;
    rp.prepack.src_dtype = src;
    rp.prepack.K = K;
    rp.prepack.N = N;
    rp.prepack.ldb = ldb;
    rp.prepack.transposed = transB;
    rp.prepack.interleave_split_halves = interleave;
    rp.prepack.pack_nr = pack_nr;
    rp.prepack.sym_group_size = group;
    return rp;
}

template <typename T>
std::vector<T> raw_weight(int K, int N, int ldb, bool transB) {
    std::vector<T> w(static_cast<size_t>(transB ? N : K) * ldb);
    for (size_t i = 0; i < w.size(); ++i) {
        const int v = static_cast<int>((i * 7 + 3) % 15) - 7;
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            w[i] = bfloat16_t(static_cast<float>(v) * 0.125f);
        } else {
            w[i] = static_cast<T>(v);
        }
    }
    return w;
}

/// native_pack_weight into a buffer of native_packed_bytes.
std::vector<uint8_t> native_image(const void *w, data_type_t wei, int K, int N,
        int ldb, bool transB, bool interleave = false) {
    std::vector<uint8_t> out(native::native_packed_bytes(wei, K, N, kNr), 0);
    EXPECT_EQ(native::native_pack_weight(wei, w, K, N, ldb, kNr, transB,
                      interleave, out.data(), 1),
            status_t::success);
    return out;
}

/// Bytes of the native image of a K x N weight.
size_t native_image_bytes(data_type_t wei, int K, int N) {
    return native::native_block_bytes(wei, K, kNr)
            * static_cast<size_t>(N / kNr);
}

TEST(NativeLayoutRulePack, Bf16BytesEqualCustomKernelPackAt32) {
    if (!has_bf16()) GTEST_SKIP() << "the bf16 native layout needs AVX512-BF16";
    for (int K : {1, 2, 3, 63, 64, 100})
        for (int N : {32, 96})
            for (bool transB : {false, true})
                for (bool interleave : {false, true})
                    for (int pad : {0, 5}) {
                        SCOPED_TRACE("K=" + std::to_string(K)
                                + " N=" + std::to_string(N)
                                + " transB=" + std::to_string(transB)
                                + " interleave=" + std::to_string(interleave)
                                + " pad=" + std::to_string(pad));
                        const int ldb = (transB ? K : N) + pad;
                        const std::vector<bfloat16_t> w
                                = raw_weight<bfloat16_t>(K, N, ldb, transB);
                        const size_t bytes
                                = native_image_bytes(data_type_t::bf16, K, N);
                        const std::vector<uint8_t> got
                                = native_image(w.data(), data_type_t::bf16, K,
                                        N, ldb, transB, interleave);
                        std::vector<uint8_t> want(
                                ck::packed_weight_size_bf16(K, N, kNr), 0);
                        ASSERT_GE(got.size(), bytes);
                        ASSERT_GE(want.size(), bytes);
                        ASSERT_EQ(ck::prepack_weight_into_bf16(w.data(), K, N,
                                          ldb, kNr, transB, interleave,
                                          want.data()),
                                status_t::success);
                        EXPECT_EQ(
                                std::memcmp(got.data(), want.data(), bytes), 0);
                    }
}

TEST(NativeLayoutRulePack, Int8BytesEqualCustomKernelPackAt32) {
    if (!has_vnni()) GTEST_SKIP() << "the s8 native layout needs AVX512-VNNI";
    for (int K : {1, 3, 4, 63, 64, 100})
        for (int N : {32, 96})
            for (bool transB : {false, true})
                for (bool interleave : {false, true}) {
                    SCOPED_TRACE("K=" + std::to_string(K)
                            + " N=" + std::to_string(N)
                            + " transB=" + std::to_string(transB)
                            + " interleave=" + std::to_string(interleave));
                    const int ldb = (transB ? K : N) + 3;
                    const std::vector<int8_t> w
                            = raw_weight<int8_t>(K, N, ldb, transB);
                    const size_t bytes
                            = native_image_bytes(data_type_t::s8, K, N);
                    const std::vector<uint8_t> got = native_image(w.data(),
                            data_type_t::s8, K, N, ldb, transB, interleave);
                    std::vector<uint8_t> want(
                            ck::packed_weight_size_int8(K, N, kNr), 0);
                    ASSERT_GE(got.size(), bytes);
                    ASSERT_GE(want.size(), bytes);
                    ASSERT_EQ(ck::prepack_weight_into_int8(w.data(), K, N, ldb,
                                      kNr, transB, interleave, want.data()),
                            status_t::success);
                    EXPECT_EQ(std::memcmp(got.data(), want.data(), bytes), 0);
                }
}

// ── ALGO 11 route ───────────────────────────────────────────────────

struct MatmulConfigGuard {
    matmul_config_t &cfg = matmul_config_t::instance();
    const int32_t algo = cfg.get_algo();
    const int32_t weight_cache = cfg.get_weight_cache();
    const bool ck_route = cfg.get_custom_kernel_route();
    ~MatmulConfigGuard() {
        cfg.set_algo(algo);
        cfg.set_weight_cache(weight_cache);
        cfg.set_custom_kernel_route(ck_route);
    }
};

class NativeLayoutRuleRoute : public ::testing::Test {
protected:
    void SetUp() override {
        reset_grp_matmul_caches();
        // resolve_kernel() latches the algorithm it first sees for the whole
        // process: let it see the process setting, not a test's.
        (void)mm::resolve_kernel();
        config_.cfg.set_algo(kAlgoNone);
        config_.cfg.set_weight_cache(2);
    }
    void TearDown() override { reset_grp_matmul_caches(); }
    MatmulConfigGuard config_;
};

TEST_F(NativeLayoutRuleRoute, CallQueryUsesTheQuantizedSourceType) {
    constexpr int K = 64, N = 96;
    matmul_params p;
    for (data_type_t src : {data_type_t::bf16, data_type_t::f32}) {
        SCOPED_TRACE(src == data_type_t::f32 ? "src=f32" : "src=bf16");
        p = matmul_params {};
        p.dtypes.src = src;
        p.dtypes.wei = data_type_t::s8;
        p.dtypes.dst = data_type_t::bf16;
        p.dynamic_quant = true;
        p.dtypes.compute = data_type_t::s8;
        EXPECT_EQ(native::native_layout_query_for(p, K, N, N, false).src_dt,
                data_type_t::s8);
        expect_native(native::native_layout_query_for(p, K, N, N, false));
        p.dtypes.compute = data_type_t::u8;
        EXPECT_EQ(native::native_layout_query_for(p, K, N, N, false).src_dt,
                data_type_t::u8);
        p.dtypes.compute = data_type_t::none;
        EXPECT_EQ(
                native::native_layout_query_for(p, K, N, N, false).src_dt, src);
        p.dynamic_quant = false;
        p.dtypes.compute = data_type_t::s8;
        EXPECT_EQ(
                native::native_layout_query_for(p, K, N, N, false).src_dt, src);
        expect_aocl(native::native_layout_query_for(p, K, N, N, false));
    }

    p.dtypes.src = data_type_t::u8;
    std::vector<float> scales(static_cast<size_t>(4) * N, 0.5f);
    p.quant_params.wei_scale.buff = scales.data();
    p.quant_params.wei_scale.dt = data_type_t::f32;
    p.quant_params.wei_scale.dims = {1, N};
    native::native_layout_query_t q
            = native::native_layout_query_for(p, K, N, N + 4, true);
    EXPECT_EQ(q.src_dt, data_type_t::u8);
    EXPECT_EQ(q.wei_group_size, 0);
    EXPECT_EQ(q.K, K);
    EXPECT_EQ(q.N, N);
    EXPECT_EQ(q.ldb, N + 4);
    EXPECT_TRUE(q.transB);
    p.quant_params.wei_scale.dims = {N};
    EXPECT_EQ(native::native_layout_query_for(p, K, N, N, false).wei_group_size,
            0);
    p.quant_params.wei_scale.dims = {4, N};
    EXPECT_EQ(native::native_layout_query_for(p, K, N, N, false).wei_group_size,
            K / 4);
    expect_native(native::native_layout_query_for(p, K, N, N, false));
    p.quant_params.wei_scale.dims = {32, N};
    EXPECT_EQ(native::native_layout_query_for(p, K, N, N, false).wei_group_size,
            2);
    expect_aocl(native::native_layout_query_for(p, K, N, N, false));
    p.quant_params.wei_scale.dims = {3, N};
    EXPECT_EQ(native::native_layout_query_for(p, K, N, N, false).wei_group_size,
            -1);
    expect_aocl(native::native_layout_query_for(p, K, N, N, false));
}

constexpr float kCanary = -1234.5f;

/// BF16 problem; weight values are exact in bf16 and sums stay exact in f32.
struct Bf16Case {
    int M, N, K, ldb;
    bool transB;
    std::vector<bfloat16_t> src, wei;
    Bf16Case(int m, int n, int k, bool tb = false)
        : M(m)
        , N(n)
        , K(k)
        , ldb(tb ? k : n)
        , transB(tb)
        , src(static_cast<size_t>(m) * k)
        , wei(raw_weight<bfloat16_t>(k, n, tb ? k : n, tb)) {
        for (size_t i = 0; i < src.size(); ++i) {
            src[i] = bfloat16_t(
                    static_cast<float>(static_cast<int>((i * 5 + 1) % 17) - 8)
                    * 0.125f);
        }
    }
    float w(int k, int n) const {
        return static_cast<float>(
                wei[transB ? static_cast<size_t>(n) * ldb + k
                           : static_cast<size_t>(k) * ldb + n]);
    }
    double ref(int m, int n) const {
        double s = 0.0;
        for (int k = 0; k < K; ++k)
            s += static_cast<double>(static_cast<float>(
                         src[static_cast<size_t>(m) * K + k]))
                    * w(k, n);
        return s;
    }
    matmul_params params(data_type_t dst = data_type_t::f32) const {
        matmul_params p;
        p.dtypes.src = data_type_t::bf16;
        p.dtypes.wei = data_type_t::bf16;
        p.dtypes.dst = dst;
        p.num_threads = kThreads;
        return p;
    }
};

/// Static u8 x s8 -> f32 with per-tensor scales and source zero point.
struct Int8Case {
    int M, N, K, ldb;
    std::vector<uint8_t> src;
    std::vector<int8_t> wei;
    float src_scale = 0.03125f, wei_scale = 0.0625f;
    int32_t src_zp = 7;
    Int8Case(int m, int n, int k)
        : M(m)
        , N(n)
        , K(k)
        , ldb(n)
        , src(static_cast<size_t>(m) * k)
        , wei(raw_weight<int8_t>(k, n, n, false)) {
        for (size_t i = 0; i < src.size(); ++i)
            src[i] = static_cast<uint8_t>((i * 11 + 5) % 23);
    }
    double ref(int m, int n) const {
        double s = 0.0;
        for (int k = 0; k < K; ++k)
            s += (static_cast<double>(src[static_cast<size_t>(m) * K + k])
                         - src_zp)
                    * wei[static_cast<size_t>(k) * ldb + n];
        return s * src_scale * wei_scale;
    }
    matmul_params params() const {
        matmul_params p;
        p.dtypes.src = data_type_t::u8;
        p.dtypes.wei = data_type_t::s8;
        p.dtypes.dst = data_type_t::f32;
        p.num_threads = kThreads;
        p.quant_params.src_scale.buff = &src_scale;
        p.quant_params.src_scale.dt = data_type_t::f32;
        p.quant_params.src_scale.dims = {1};
        p.quant_params.src_zp.buff = &src_zp;
        p.quant_params.src_zp.dt = data_type_t::s32;
        p.quant_params.src_zp.dims = {1};
        p.quant_params.wei_scale.buff = &wei_scale;
        p.quant_params.wei_scale.dt = data_type_t::f32;
        p.quant_params.wei_scale.dims = {1};
        return p;
    }
};

/// bf16 or f32 source quantized per token to s8, x s8 -> f32 with per-channel
/// weight scales.  Every source row reaches |127|: the per-token scale is 1
/// and the quantized source equals the original one.
struct DynQuantCase {
    static constexpr int M = 3, N = 64, K = 64;
    data_type_t src_dt;
    std::vector<float> src_f32;
    std::vector<bfloat16_t> src_bf16;
    std::vector<int8_t> wei;
    std::vector<float> wei_scale;
    explicit DynQuantCase(data_type_t dt)
        : src_dt(dt)
        , src_f32(static_cast<size_t>(M) * K)
        , src_bf16(src_f32.size())
        , wei(raw_weight<int8_t>(K, N, N, false))
        , wei_scale(N) {
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < K; ++k)
                src_f32[static_cast<size_t>(m) * K + k] = static_cast<float>(
                        k == m ? 127
                               : static_cast<int>((m * 31 + k * 7) % 41) - 20);
        for (size_t i = 0; i < src_f32.size(); ++i)
            src_bf16[i] = bfloat16_t(src_f32[i]);
        for (int n = 0; n < N; ++n)
            wei_scale[n] = 0.0078125f * static_cast<float>(1 + n % 3);
    }
    const void *src() const {
        if (src_dt == data_type_t::f32) return src_f32.data();
        return src_bf16.data();
    }
    double ref(int m, int n) const {
        double s = 0.0;
        for (int k = 0; k < K; ++k)
            s += static_cast<double>(src_f32[static_cast<size_t>(m) * K + k])
                    * wei[static_cast<size_t>(k) * N + n];
        return s * wei_scale[n];
    }
    matmul_params params() const {
        matmul_params p;
        p.dtypes.src = src_dt;
        p.dtypes.wei = data_type_t::s8;
        p.dtypes.dst = data_type_t::f32;
        p.dtypes.compute = data_type_t::s8;
        p.dynamic_quant = true;
        p.num_threads = kThreads;
        p.quant_params.src_scale.dt = data_type_t::f32;
        p.quant_params.src_scale.dims = {M, 1};
        p.quant_params.wei_scale.buff = wei_scale.data();
        p.quant_params.wei_scale.dt = data_type_t::f32;
        p.quant_params.wei_scale.dims = {1, N};
        return p;
    }
};

/// The descriptor native_matmul_execute builds for a row-major call.
native::GemmDescriptor descriptor(int M, int N, int K, int lda, int ldb,
        int ldc, bool transB, const matmul_params &p) {
    native::GemmDescriptor d;
    d.M = M;
    d.N = N;
    d.K = K;
    d.lda = lda;
    d.ldb = ldb;
    d.ldc = ldc;
    d.transA = false;
    d.transB = transB;
    d.alpha = 1.0f;
    d.beta = 0.0f;
    d.src_dt = p.dtypes.src;
    d.wei_dt = p.dtypes.wei;
    d.dst_dt = p.dtypes.dst;
    d.bias_dt = p.dtypes.bias;
    d.src_elem_size = size_of(d.src_dt);
    d.wei_elem_size = size_of(d.wei_dt);
    d.dst_elem_size = size_of(d.dst_dt);
    d.bias = nullptr;
    d.is_weights_const = true;
    d.num_threads = p.num_threads;
    return d;
}

native::native_weight_ref_t weight_ref(
        const void *w, int pack_nr, int32_t wc = 1) {
    native::native_weight_ref_t r;
    r.weight = w;
    r.pack_nr = pack_nr;
    r.weight_cache_mode = wc;
    return r;
}

status_t direct_call(int M, int N, int K, const void *src, int lda,
        const native::native_weight_ref_t &ref, int ldb, bool transB, void *dst,
        int ldc, const matmul_params &p, const void *bias = nullptr,
        matmul_gated_act_t gated = matmul_gated_act_t::none) {
    matmul_params copy = p;
    native::GemmDescriptor desc = descriptor(M, N, K, lda, ldb, ldc, transB, p);
    desc.bias = bias;
    desc.gated_act = gated;
    return native::native_brgemm_execute(desc, ref, src, dst, bias, copy);
}

status_t routed_call(int M, int N, int K, const void *src, int lda,
        const void *wei, int ldb, bool transB, void *dst, int ldc,
        const matmul_params &p, int batch_a = 1, bool weights_const = true,
        const void *bias = nullptr,
        matmul_gated_act_t gated = matmul_gated_act_t::none) {
    mm::matmul_batch_params_t bp;
    bp.Batch_A = batch_a;
    mm::matmul_gated_act_params act;
    act.act = gated;
    return mm::matmul_direct('r', false, transB, M, N, K, 1.0f, src, lda, wei,
            ldb, bias, 0.0f, dst, ldc, weights_const, bp, p,
            gated == matmul_gated_act_t::none ? nullptr : &act);
}

/// The routed call went to native_brgemm_execute with the expected weight
/// reference: same status as the direct call, and on success the same bytes
/// and, at weight cache mode wc 1 or 2, a layout of routed_wei kept by the
/// native BRGEMM; on failure dst is untouched.
template <typename T>
void expect_routed_like_direct(status_t routed, const void *routed_wei,
        int32_t wc, const std::vector<T> &routed_dst, status_t direct,
        const std::vector<T> &direct_dst, T canary) {
    EXPECT_EQ(static_cast<int>(routed), static_cast<int>(direct));
    if (direct == status_t::success) {
        if (wc != 0) {
            EXPECT_GE(native::native_weight_layouts(routed_wei), 1u)
                    << "not served by the native BRGEMM";
        }
        EXPECT_EQ(std::memcmp(routed_dst.data(), direct_dst.data(),
                          routed_dst.size() * sizeof(T)),
                0);
    } else {
        size_t written = 0;
        for (const T &v : routed_dst)
            written += std::memcmp(&v, &canary, sizeof(T)) != 0;
        EXPECT_EQ(written, 0u) << "a failed native call must not fall back";
    }
}

template <typename Case>
void expect_matches_reference(const Case &c, const std::vector<float> &dst,
        int ldc, double rel = 1e-4) {
    size_t bad = 0;
    for (int m = 0; m < c.M; ++m)
        for (int n = 0; n < c.N; ++n) {
            const double want = c.ref(m, n);
            const double got = dst[static_cast<size_t>(m) * ldc + n];
            if (std::fabs(got - want) > rel * std::fabs(want) + 1e-3) ++bad;
        }
    EXPECT_EQ(bad, 0u) << "native result differs from the reference";
}

TEST_F(NativeLayoutRuleRoute, CustomKernelRouteOffKeepsAlgo11Native) {
    config_.cfg.set_custom_kernel_route(false);
    config_.cfg.set_algo(kAlgoNative);
    for (int32_t wc : {0, 1, 2}) {
        SCOPED_TRACE("wc=" + std::to_string(wc));
        reset_grp_matmul_caches();
        if (has_bf16()) {
            const Bf16Case c(8, 2048, 64, /*transB=*/true);
            std::vector<bfloat16_t> w = c.wei;
            matmul_params p = c.params();
            p.weight_cache_type = wc;
            std::vector<float> dst(static_cast<size_t>(c.M) * c.N, kCanary);
            ASSERT_EQ(routed_call(c.M, c.N, c.K, c.src.data(), c.K, w.data(),
                              c.ldb, c.transB, dst.data(), c.N, p),
                    status_t::success);
            if (wc != 0) {
                EXPECT_GE(native::native_weight_layouts(w.data()), 1u);
            }
            expect_matches_reference(c, dst, c.N);
        }
        if (has_vnni()) {
            const DynQuantCase c(data_type_t::bf16);
            std::vector<int8_t> w = c.wei;
            matmul_params p = c.params();
            p.weight_cache_type = wc;
            std::vector<float> dst(static_cast<size_t>(c.M) * c.N, kCanary);
            ASSERT_EQ(routed_call(c.M, c.N, c.K, c.src(), c.K, w.data(), c.N,
                              false, dst.data(), c.N, p),
                    status_t::success);
            if (wc != 0) {
                EXPECT_GE(native::native_weight_layouts(w.data()), 1u);
            }
            expect_matches_reference(c, dst, c.N);
        }
    }
}

TEST_F(NativeLayoutRuleRoute, RawEligibleWeightReachesNativeAndStaysOffAocl) {
    if (!has_bf16()) GTEST_SKIP() << "needs AVX512-BF16";
    for (bool per_call : {true, false})
        for (int32_t wc : {0, 1, 2}) {
            SCOPED_TRACE(std::string(per_call ? "lowoha_algo=native_brgemm"
                                              : "ZENDNNL_MATMUL_ALGO=11")
                    + " wc=" + std::to_string(wc));
            reset_grp_matmul_caches();
            config_.cfg.set_algo(per_call ? kAlgoNone : kAlgoNative);
            const Bf16Case c(4, 64, 64);
            std::vector<bfloat16_t> routed_w = c.wei, direct_w = c.wei;
            matmul_params p = c.params();
            p.weight_cache_type = wc;
            if (per_call) p.lowoha_algo = matmul_algo_t::native_brgemm;

            std::vector<float> direct(static_cast<size_t>(c.M) * c.N, kCanary);
            const status_t direct_status = direct_call(c.M, c.N, c.K,
                    c.src.data(), c.K, weight_ref(direct_w.data(), kNr, wc),
                    c.ldb, false, direct.data(), c.N, p);
            std::vector<float> dst(direct.size(), kCanary);
            const status_t st = routed_call(c.M, c.N, c.K, c.src.data(), c.K,
                    routed_w.data(), c.ldb, false, dst.data(), c.N, p);
            expect_routed_like_direct(st, routed_w.data(), wc, dst,
                    direct_status, direct, kCanary);
            if (direct_status == status_t::success)
                expect_matches_reference(c, dst, c.N);

            const bool raw = std::memcmp(routed_w.data(), c.wei.data(),
                                     c.wei.size() * sizeof(bfloat16_t))
                    == 0;
            EXPECT_EQ(raw, wc != 2 || per_call)
                    << "only the process-wide ALGO 11 rewrites in place";
            EXPECT_EQ(packed_in_place(routed_w), !raw);
        }
}

TEST_F(NativeLayoutRuleRoute, RawEligibleInt8WeightReachesNative) {
    if (!has_vnni()) GTEST_SKIP() << "needs AVX512-VNNI";
    for (int32_t wc : {0, 1, 2}) {
        SCOPED_TRACE("wc=" + std::to_string(wc));
        reset_grp_matmul_caches();
        const Int8Case c(5, 96, 64);
        std::vector<int8_t> routed_w = c.wei, direct_w = c.wei;
        matmul_params p = c.params();
        p.weight_cache_type = wc;
        p.lowoha_algo = matmul_algo_t::native_brgemm;
        std::vector<float> direct(static_cast<size_t>(c.M) * c.N, kCanary);
        const status_t direct_status = direct_call(c.M, c.N, c.K, c.src.data(),
                c.K, weight_ref(direct_w.data(), kNr, wc), c.ldb, false,
                direct.data(), c.N, p);
        std::vector<float> dst(direct.size(), kCanary);
        const status_t st = routed_call(c.M, c.N, c.K, c.src.data(), c.K,
                routed_w.data(), c.ldb, false, dst.data(), c.N, p);
        expect_routed_like_direct(
                st, routed_w.data(), wc, dst, direct_status, direct, kCanary);
        if (direct_status == status_t::success)
            expect_matches_reference(c, dst, c.N);
        EXPECT_EQ(std::memcmp(routed_w.data(), c.wei.data(), c.wei.size()), 0)
                << "an undeclared-capacity s8 weight is never rewritten";
    }
}

TEST_F(NativeLayoutRuleRoute, RawEligibleDynamicQuantWeightReachesNative) {
    if (!has_vnni()) GTEST_SKIP() << "needs AVX512-VNNI";
    for (data_type_t src_dt : {data_type_t::bf16, data_type_t::f32})
        for (bool per_call : {true, false})
            for (int32_t wc : {0, 1, 2}) {
                SCOPED_TRACE(
                        std::string(src_dt == data_type_t::f32 ? "src=f32 "
                                                               : "src=bf16 ")
                        + (per_call ? "lowoha_algo=native_brgemm"
                                    : "ZENDNNL_MATMUL_ALGO=11")
                        + " wc=" + std::to_string(wc));
                reset_grp_matmul_caches();
                config_.cfg.set_algo(per_call ? kAlgoNone : kAlgoNative);
                const DynQuantCase c(src_dt);
                std::vector<int8_t> routed_w = c.wei, direct_w = c.wei;
                matmul_params p = c.params();
                p.weight_cache_type = wc;
                if (per_call) p.lowoha_algo = matmul_algo_t::native_brgemm;
                std::vector<float> direct(
                        static_cast<size_t>(c.M) * c.N, kCanary);
                const status_t direct_status = direct_call(c.M, c.N, c.K,
                        c.src(), c.K, weight_ref(direct_w.data(), kNr, wc), c.N,
                        false, direct.data(), c.N, p);
                EXPECT_EQ(direct_status, status_t::success);
                std::vector<float> dst(direct.size(), kCanary);
                const status_t st = routed_call(c.M, c.N, c.K, c.src(), c.K,
                        routed_w.data(), c.N, false, dst.data(), c.N, p);
                expect_routed_like_direct(st, routed_w.data(), wc, dst,
                        direct_status, direct, kCanary);
                if (direct_status == status_t::success)
                    expect_matches_reference(c, dst, c.N);
                EXPECT_EQ(std::memcmp(
                                  routed_w.data(), c.wei.data(), c.wei.size()),
                        0)
                        << "an undeclared-capacity s8 weight is never "
                           "rewritten";
            }
}

TEST_F(NativeLayoutRuleRoute, NonConstantRawWeightPacksForTheCallOnly) {
    if (!has_bf16()) GTEST_SKIP() << "needs AVX512-BF16";
    const Bf16Case c(2, 64, 32);
    std::vector<bfloat16_t> routed_w = c.wei, direct_w = c.wei;
    matmul_params p = c.params();
    p.lowoha_algo = matmul_algo_t::native_brgemm;
    std::vector<float> direct(static_cast<size_t>(c.M) * c.N, kCanary);
    const status_t direct_status = direct_call(c.M, c.N, c.K, c.src.data(), c.K,
            weight_ref(direct_w.data(), kNr, 0), c.ldb, false, direct.data(),
            c.N, p);
    std::vector<float> dst(direct.size(), kCanary);
    const status_t st = routed_call(c.M, c.N, c.K, c.src.data(), c.K,
            routed_w.data(), c.ldb, false, dst.data(), c.N, p, 1,
            /*weights_const=*/false);
    expect_routed_like_direct(
            st, routed_w.data(), 0, dst, direct_status, direct, kCanary);
    EXPECT_EQ(std::memcmp(routed_w.data(), c.wei.data(),
                      c.wei.size() * sizeof(bfloat16_t)),
            0);
    EXPECT_FALSE(packed_in_place(routed_w));
}

TEST_F(NativeLayoutRuleRoute, InPlaceImageIsServedByNativeUnderAlgo11) {
    if (!has_bf16()) GTEST_SKIP() << "needs AVX512-BF16";
    for (bool per_call : {true, false}) {
        SCOPED_TRACE(per_call ? "lowoha_algo=native_brgemm"
                              : "ZENDNNL_MATMUL_ALGO=11");
        reset_grp_matmul_caches();
        const Bf16Case c(4, 64, 64);
        std::vector<bfloat16_t> routed_w = c.wei, direct_w = c.wei;
        for (std::vector<bfloat16_t> *w : {&routed_w, &direct_w}) {
            native::native_raw_weight_t raw;
            raw.wei_dt = data_type_t::bf16;
            raw.weight = w->data();
            raw.K = c.K;
            raw.N = c.N;
            raw.ldb = c.ldb;
            raw.pack_nr = kNr;
            native::native_layout_t image;
            ASSERT_EQ(native::get_native_layout(raw, 2, 0, 1, image),
                    status_t::success);
            ASSERT_EQ(image.data, w->data());
            ASSERT_TRUE(packed_in_place(*w));
        }
        config_.cfg.set_algo(per_call ? kAlgoNone : kAlgoNative);
        matmul_params p = c.params();
        if (per_call) p.lowoha_algo = matmul_algo_t::native_brgemm;
        std::vector<float> direct(static_cast<size_t>(c.M) * c.N, kCanary);
        const status_t direct_status = direct_call(c.M, c.N, c.K, c.src.data(),
                c.K, weight_ref(direct_w.data(), kNr, 2), c.ldb, false,
                direct.data(), c.N, p);
        std::vector<float> dst(direct.size(), kCanary);
        const status_t st = routed_call(c.M, c.N, c.K, c.src.data(), c.K,
                routed_w.data(), c.ldb, false, dst.data(), c.N, p);
        expect_routed_like_direct(
                st, routed_w.data(), 2, dst, direct_status, direct, kCanary);
        if (direct_status == status_t::success)
            expect_matches_reference(c, dst, c.N);
        EXPECT_TRUE(packed_in_place(routed_w));
    }
}

// The gated result fills the first N / 2 columns of a full-width (ldc = N)
// or a half-width dst.
TEST_F(NativeLayoutRuleRoute, GatedActivationOnRawWeightReachesNative) {
    if (!has_bf16()) GTEST_SKIP() << "needs AVX512-BF16";
    const Bf16Case c(3, 128, 64);
    for (bool per_call : {true, false})
        for (matmul_gated_act_t act : {matmul_gated_act_t::silu_and_mul,
                     matmul_gated_act_t::gelu_and_mul,
                     matmul_gated_act_t::swiglu_oai_mul})
            for (int ldc : {c.N, c.N / 2}) {
                SCOPED_TRACE(std::string(per_call ? "lowoha_algo=native_brgemm"
                                                  : "ZENDNNL_MATMUL_ALGO=11")
                        + " gated_act " + std::to_string(static_cast<int>(act))
                        + " ldc=" + std::to_string(ldc));
                reset_grp_matmul_caches();
                config_.cfg.set_algo(per_call ? kAlgoNone : kAlgoNative);
                matmul_params p = c.params(data_type_t::bf16);
                p.weight_cache_type = 1;
                if (per_call) p.lowoha_algo = matmul_algo_t::native_brgemm;
                native::native_weight_ref_t ref
                        = weight_ref(c.wei.data(), kNr, 0);
                ref.interleave_split_halves
                        = act != matmul_gated_act_t::swiglu_oai_mul;
                const bfloat16_t canary(kCanary);
                std::vector<bfloat16_t> direct(
                        static_cast<size_t>(c.M) * ldc, canary);
                const status_t direct_status = direct_call(c.M, c.N, c.K,
                        c.src.data(), c.K, ref, c.ldb, false, direct.data(),
                        ldc, p, nullptr, act);
                ASSERT_EQ(direct_status, status_t::success);
                std::vector<bfloat16_t> w = c.wei;
                std::vector<bfloat16_t> dst(direct.size(), canary);
                const status_t st = routed_call(c.M, c.N, c.K, c.src.data(),
                        c.K, w.data(), c.ldb, false, dst.data(), ldc, p, 1,
                        true, nullptr, act);
                expect_routed_like_direct(
                        st, w.data(), 1, dst, direct_status, direct, canary);
            }
}

// A weight the rule keeps off the layout, and a gated call with a post-op:
// the one-op group route serves both under ALGO 11 as with ALGO unset.
TEST_F(NativeLayoutRuleRoute, GatedCallNativeDoesNotTakeKeepsTheGroupRoute) {
    const Bf16Case off_layout(4, 48, 64);
    const Bf16Case on_layout(4, 64, 64);
    for (bool post_op : {false, true})
        for (matmul_gated_act_t act : {matmul_gated_act_t::silu_and_mul,
                     matmul_gated_act_t::gelu_and_mul,
                     matmul_gated_act_t::swiglu_oai_mul}) {
            SCOPED_TRACE(std::string(post_op ? "post-op" : "N=48")
                    + " gated_act " + std::to_string(static_cast<int>(act)));
            const Bf16Case &c = post_op ? on_layout : off_layout;
            matmul_params p = c.params();
            p.weight_cache_type = 1;
            if (post_op) {
                mm::matmul_post_op relu;
                relu.po_type = post_op_type_t::relu;
                p.postop_.push_back(relu);
            }
            std::vector<bfloat16_t> w_unset = c.wei, w_native = c.wei;
            std::vector<float> want(static_cast<size_t>(c.M) * c.N, kCanary);
            std::vector<float> dst(want.size(), kCanary);
            reset_grp_matmul_caches();
            config_.cfg.set_algo(kAlgoNone);
            const status_t want_status = routed_call(c.M, c.N, c.K,
                    c.src.data(), c.K, w_unset.data(), c.ldb, false,
                    want.data(), c.N, p, 1, true, nullptr, act);
            reset_grp_matmul_caches();
            config_.cfg.set_algo(kAlgoNative);
            const status_t st = routed_call(c.M, c.N, c.K, c.src.data(), c.K,
                    w_native.data(), c.ldb, false, dst.data(), c.N, p, 1, true,
                    nullptr, act);
            EXPECT_EQ(native::native_weight_layouts(w_native.data()), 0u);
            EXPECT_EQ(static_cast<int>(st), static_cast<int>(want_status));
            if (st != status_t::success || want_status != status_t::success)
                continue;
            size_t bad = 0;
            for (int m = 0; m < c.M; ++m)
                for (int i = 0; i < c.N / 2; ++i) {
                    const float g = dst[static_cast<size_t>(m) * c.N + i];
                    const float r = want[static_cast<size_t>(m) * c.N + i];
                    if (std::fabs(g - r) > 1e-2f * std::fabs(r) + 1e-3f) ++bad;
                }
            EXPECT_EQ(bad, 0u);
        }
}

TEST_F(NativeLayoutRuleRoute, IneligibleRawWeightKeepsItsRoute) {
    if (!has_bf16()) GTEST_SKIP() << "needs AVX512-BF16";
    const Bf16Case c(4, 48, 64);
    for (bool per_call : {true, false}) {
        SCOPED_TRACE(per_call ? "lowoha_algo=native_brgemm"
                              : "ZENDNNL_MATMUL_ALGO=11");
        config_.cfg.set_algo(per_call ? kAlgoNone : kAlgoNative);
        matmul_params p = c.params();
        if (per_call) p.lowoha_algo = matmul_algo_t::native_brgemm;
        std::vector<float> dst(static_cast<size_t>(c.M) * c.N, kCanary);
        ASSERT_EQ(routed_call(c.M, c.N, c.K, c.src.data(), c.K, c.wei.data(),
                          c.ldb, false, dst.data(), c.N, p),
                status_t::success);
        EXPECT_EQ(native::native_weight_layouts(c.wei.data()), 0u);
        expect_matches_reference(c, dst, c.N, 1e-2);
    }
}

/// A matmul_direct call of the case on wei with the call shape and options
/// a refusal variant changes.
struct variant_t {
    const char *name;
    char layout = 'r';
    char mem_format_a = 'n';
    int batch_a = 1;
    int N = 0; ///< 0: the case's N
    data_type_t src_dt = data_type_t::bf16;
    matmul_gated_act_t gated = matmul_gated_act_t::none;
    bool relu = false;
};

status_t variant_call(const Bf16Case &c, const variant_t &v, const void *src,
        const void *wei, std::vector<float> &dst) {
    const int N = v.N != 0 ? v.N : c.N;
    matmul_params p = c.params();
    p.dtypes.src = v.src_dt;
    p.mem_format_a = v.mem_format_a;
    if (v.relu) {
        mm::matmul_post_op po;
        po.po_type = post_op_type_t::relu;
        p.postop_.push_back(po);
    }
    mm::matmul_batch_params_t bp;
    bp.Batch_A = v.batch_a;
    mm::matmul_gated_act_params act;
    act.act = v.gated;
    const int lda = v.layout == 'r' ? c.K : c.M;
    const int ldb = v.layout == 'r' ? c.ldb : c.K;
    const int ldc = v.layout == 'r' ? N : c.M;
    return mm::matmul_direct(v.layout, false, false, c.M, N, c.K, 1.0f, src,
            lda, wei, ldb, nullptr, 0.0f, dst.data(), ldc, true, bp, p,
            v.gated == matmul_gated_act_t::none ? nullptr : &act);
}

/// The calls the native BRGEMM refuses on an otherwise qualifying bf16
/// weight: a batched call, a column-major call, a reordered source, a gated
/// call with a post-op, a source of another type, and a view whose N the
/// layout rule refuses.
std::vector<variant_t> refused_variants() {
    std::vector<variant_t> v;
    v.push_back({"batched"});
    v.back().batch_a = 2;
    v.push_back({"column-major"});
    v.back().layout = 'c';
    v.push_back({"reordered source"});
    v.back().mem_format_a = 'r';
    v.push_back({"gated with a post-op"});
    v.back().gated = matmul_gated_act_t::silu_and_mul;
    v.back().relu = true;
    v.push_back({"f32 source"});
    v.back().src_dt = data_type_t::f32;
    v.push_back({"N off the layout"});
    v.back().N = 48;
    return v;
}

// A buffer rewritten in place has the native BRGEMM as its only reader: a
// call on it that the native BRGEMM refuses fails, leaves dst as it was and
// the buffer holding the image.
TEST_F(NativeLayoutRuleRoute, RefusedCallOnAnInPlaceImageFails) {
    if (!has_bf16()) GTEST_SKIP() << "needs AVX512-BF16";
    config_.cfg.set_algo(kAlgoNative);
    const Bf16Case c(4, 64, 64);
    std::vector<bfloat16_t> w = c.wei;
    std::vector<float> dst(static_cast<size_t>(c.M) * c.N, kCanary);
    ASSERT_EQ(routed_call(c.M, c.N, c.K, c.src.data(), c.K, w.data(), c.ldb,
                      false, dst.data(), c.N, c.params()),
            status_t::success);
    ASSERT_TRUE(packed_in_place(w));
    const std::vector<bfloat16_t> image = w;
    // Two batches of the source, f32 for the f32 variant.
    std::vector<bfloat16_t> src(2 * c.src.size());
    std::copy(c.src.begin(), c.src.end(), src.begin());
    std::copy(c.src.begin(), c.src.end(), src.begin() + c.src.size());
    std::vector<float> src_f32(src.size());
    for (size_t i = 0; i < src.size(); ++i)
        src_f32[i] = static_cast<float>(src[i]);
    for (const variant_t &v : refused_variants()) {
        SCOPED_TRACE(v.name);
        std::vector<float> out(2 * static_cast<size_t>(c.M) * c.N, kCanary);
        const void *a = v.src_dt == data_type_t::f32
                ? static_cast<const void *>(src_f32.data())
                : static_cast<const void *>(src.data());
        EXPECT_NE(variant_call(c, v, a, w.data(), out), status_t::success);
        EXPECT_TRUE(std::all_of(out.begin(), out.end(), [](float x) {
            return x == kCanary;
        })) << "another route wrote dst";
        EXPECT_TRUE(packed_in_place(w));
        EXPECT_EQ(std::memcmp(w.data(), image.data(),
                          w.size() * sizeof(bfloat16_t)),
                0)
                << "another route rewrote the image";
    }
}

// The same calls on a raw weight are declined: the native BRGEMM keeps
// nothing for the weight, the weight stays raw, and the routes that take
// them write the product.
TEST_F(NativeLayoutRuleRoute, RefusedCallOnARawWeightRunsElsewhere) {
    if (!has_bf16()) GTEST_SKIP() << "needs AVX512-BF16";
    for (bool per_call : {true, false})
        for (const variant_t &v : refused_variants()) {
            SCOPED_TRACE(std::string(v.name) + " "
                    + (per_call ? "lowoha_algo=native_brgemm"
                                : "ZENDNNL_MATMUL_ALGO=11"));
            reset_grp_matmul_caches();
            config_.cfg.set_algo(per_call ? kAlgoNone : kAlgoNative);
            const Bf16Case c(4, v.N != 0 ? v.N : 64, 64);
            std::vector<bfloat16_t> w = c.wei;
            std::vector<bfloat16_t> src(2 * c.src.size());
            std::copy(c.src.begin(), c.src.end(), src.begin());
            std::copy(c.src.begin(), c.src.end(), src.begin() + c.src.size());
            std::vector<float> src_f32(src.size());
            for (size_t i = 0; i < src.size(); ++i)
                src_f32[i] = static_cast<float>(src[i]);
            variant_t call = v;
            call.N = 0;
            std::vector<float> out(2 * static_cast<size_t>(c.M) * c.N, kCanary);
            const void *a = v.src_dt == data_type_t::f32
                    ? static_cast<const void *>(src_f32.data())
                    : static_cast<const void *>(src.data());
            matmul_params p = c.params();
            if (per_call) p.lowoha_algo = matmul_algo_t::native_brgemm;
            const status_t st = [&] {
                matmul_params q = p;
                q.dtypes.src = call.src_dt;
                q.mem_format_a = call.mem_format_a;
                if (call.relu) {
                    mm::matmul_post_op po;
                    po.po_type = post_op_type_t::relu;
                    q.postop_.push_back(po);
                }
                mm::matmul_batch_params_t bp;
                bp.Batch_A = call.batch_a;
                mm::matmul_gated_act_params act;
                act.act = call.gated;
                const bool col = call.layout != 'r';
                return mm::matmul_direct(call.layout, false, false, c.M, c.N,
                        c.K, 1.0f, a, col ? c.M : c.K, w.data(),
                        col ? c.K : c.ldb, nullptr, 0.0f, out.data(),
                        col ? c.M : c.N, true, bp, q,
                        call.gated == matmul_gated_act_t::none ? nullptr
                                                               : &act);
            }();
            EXPECT_EQ(native::native_weight_layouts(w.data()), 0u);
            EXPECT_FALSE(packed_in_place(w));
            if (st != status_t::success) continue;
            // A plain row-major bf16 product is checked against the
            // reference.
            if (call.layout == 'r' && call.mem_format_a == 'n'
                    && call.src_dt == data_type_t::bf16
                    && call.gated == matmul_gated_act_t::none)
                expect_matches_reference(c, out, c.N, 1e-2);
        }
}

// A GGML-packed weight is never read in the native layout: the native
// BRGEMM declines the call and leaves dst untouched.
TEST_F(NativeLayoutRuleRoute, GgmlPackedWeightIsDeclined) {
    if (!has_vnni()) GTEST_SKIP() << "needs AVX512-VNNI";
    config_.cfg.set_algo(kAlgoNative);
    const Int8Case c(4, 64, 64);
    matmul_params p = c.params();
    p.packing.pack_format_b = 1;
    std::vector<int8_t> w = c.wei;
    std::vector<float> dst(static_cast<size_t>(c.M) * c.N, kCanary);
    status_t st = status_t::success;
    EXPECT_FALSE(native::native_layout_matmul('r', false, false, c.M, c.N, c.K,
            1.0f, c.src.data(), c.K, w.data(), c.N, nullptr, 0.0f, dst.data(),
            c.N, true, 1, kThreads, matmul_gated_act_t::none, p, nullptr, st));
    EXPECT_EQ(native::native_weight_layouts(w.data()), 0u);
    EXPECT_TRUE(std::all_of(
            dst.begin(), dst.end(), [](float x) { return x == kCanary; }));
    EXPECT_EQ(w, c.wei);
}

// A buffer the custom kernel rewrote in place (a grouped call at weight
// cache mode 2) is read by the custom kernel's routes only: under ALGO 11 a
// dense call on it is not taken by the native BRGEMM, which keeps nothing
// for it and leaves the image as it is.
TEST_F(NativeLayoutRuleRoute, CustomKernelImageIsNotTaken) {
    if (!ck::dispatch_supported()) GTEST_SKIP() << "needs AVX512-BF16";
    constexpr int kExperts = 4, M = 8, N = 256, K = 256;
    std::vector<std::vector<bfloat16_t>> wei, src;
    std::vector<std::vector<bfloat16_t>> out(
            kExperts, std::vector<bfloat16_t>(static_cast<size_t>(M) * N));
    for (int e = 0; e < kExperts; ++e) {
        wei.push_back(raw_weight<bfloat16_t>(K, N, N, false));
        src.push_back(std::vector<bfloat16_t>(static_cast<size_t>(M) * K,
                bfloat16_t(0.25f * static_cast<float>(e + 1))));
    }
    {
        mt::AlgoEnvGuard grouped_auto(0);
        mt::CustomKernelOverride ck_on(true);
        std::vector<const void *> a, b;
        std::vector<void *> d;
        for (int e = 0; e < kExperts; ++e) {
            a.push_back(src[e].data());
            b.push_back(wei[e].data());
            d.push_back(out[e].data());
        }
        matmul_params p;
        p.dtypes.src = p.dtypes.wei = p.dtypes.dst = data_type_t::bf16;
        ASSERT_EQ(mm::group_matmul_direct(std::vector<char>(kExperts, 'r'),
                          std::vector<bool>(kExperts, false),
                          std::vector<bool>(kExperts, false),
                          std::vector<int>(kExperts, M),
                          std::vector<int>(kExperts, N),
                          std::vector<int>(kExperts, K),
                          std::vector<float>(kExperts, 1.0f), a,
                          std::vector<int>(kExperts, K), b,
                          std::vector<int>(kExperts, N),
                          std::vector<const void *>(kExperts, nullptr),
                          std::vector<float>(kExperts, 0.0f), d,
                          std::vector<int>(kExperts, N),
                          std::vector<bool>(kExperts, true),
                          std::vector<matmul_params>(kExperts, p)),
                status_t::success);
    }
    if (!ck::weight_packed_in_place(wei[0].data()))
        GTEST_SKIP() << "the grouped call did not pack in place";
    const std::vector<bfloat16_t> image = wei[0];
    config_.cfg.set_algo(kAlgoNative);
    std::vector<float> dst(static_cast<size_t>(M) * N, kCanary);
    matmul_params p;
    p.dtypes.src = p.dtypes.wei = data_type_t::bf16;
    p.dtypes.dst = data_type_t::f32;
    p.num_threads = kThreads;
    (void)routed_call(M, N, K, src[0].data(), K, wei[0].data(), N, false,
            dst.data(), N, p);
    EXPECT_EQ(native::native_weight_layouts(wei[0].data()), 0u);
    EXPECT_FALSE(packed_in_place(wei[0]));
    EXPECT_EQ(std::memcmp(wei[0].data(), image.data(),
                      image.size() * sizeof(bfloat16_t)),
            0)
            << "the image was rewritten";
}

// A bf16 call with an f16 dst, run end to end on the native BRGEMM, holds the
// reference product.
TEST_F(NativeLayoutRuleRoute, F16DstIsServedEndToEnd) {
    if (!has_bf16()) GTEST_SKIP() << "needs AVX512-BF16";
    if (!zendnnl::common::zendnnl_platform_info().get_avx512_f16_status())
        GTEST_SKIP() << "matmul_direct takes an f16 dst only with AVX512-FP16";
    for (bool per_call : {true, false}) {
        SCOPED_TRACE(per_call ? "lowoha_algo=native_brgemm"
                              : "ZENDNNL_MATMUL_ALGO=11");
        reset_grp_matmul_caches();
        config_.cfg.set_algo(per_call ? kAlgoNone : kAlgoNative);
        const Bf16Case c(5, 64, 96);
        std::vector<bfloat16_t> w = c.wei;
        matmul_params p = c.params(data_type_t::f16);
        p.weight_cache_type = 1;
        if (per_call) p.lowoha_algo = matmul_algo_t::native_brgemm;
        std::vector<zendnnl::common::float16_t> dst(
                static_cast<size_t>(c.M) * c.N);
        ASSERT_EQ(routed_call(c.M, c.N, c.K, c.src.data(), c.K, w.data(), c.ldb,
                          false, dst.data(), c.N, p),
                status_t::success);
        EXPECT_GE(native::native_weight_layouts(w.data()), 1u);
        std::vector<float> got(dst.size());
        for (size_t i = 0; i < dst.size(); ++i)
            got[i] = static_cast<float>(dst[i]);
        expect_matches_reference(c, got, c.N, 2e-3);
    }
}

TEST_F(NativeLayoutRuleRoute, BatchedCallTakesItsOtherRoute) {
    if (!has_bf16()) GTEST_SKIP() << "needs AVX512-BF16";
    const Bf16Case c(4, 64, 64);
    std::vector<bfloat16_t> w = c.wei;
    std::vector<bfloat16_t> src(2 * c.src.size());
    std::copy(c.src.begin(), c.src.end(), src.begin());
    std::copy(c.src.begin(), c.src.end(), src.begin() + c.src.size());
    matmul_params p = c.params();
    p.lowoha_algo = matmul_algo_t::native_brgemm;
    p.weight_cache_type = 2;
    std::vector<float> dst(2 * static_cast<size_t>(c.M) * c.N, kCanary);
    ASSERT_EQ(routed_call(c.M, c.N, c.K, src.data(), c.K, w.data(), c.ldb,
                      false, dst.data(), c.N, p, /*batch_a=*/2),
            status_t::success);
    EXPECT_EQ(native::native_weight_layouts(w.data()), 0u);
    EXPECT_FALSE(packed_in_place(w));
    const std::ptrdiff_t batch_size = static_cast<std::ptrdiff_t>(c.M) * c.N;
    for (int b = 0; b < 2; ++b) {
        const std::vector<float> slice(dst.begin() + b * batch_size,
                dst.begin() + (b + 1) * batch_size);
        expect_matches_reference(c, slice, c.N, 1e-2);
    }
}

TEST_F(NativeLayoutRuleRoute, DefaultRouteEngagesCustomKernel) {
    if (!ck::dispatch_supported()) GTEST_SKIP() << "needs AVX512-BF16";
    if (mm::resolve_kernel() != matmul_algo_t::aocl_dlp_blocked)
        GTEST_SKIP() << "the process-wide algorithm is not the default";
    config_.cfg.set_algo(kAlgoAocl);
    config_.cfg.set_custom_kernel_route(true);
    config_.cfg.set_weight_cache(1);
    mt::AlgoEnvGuard auto_algo(0);
    mt::CustomKernelOverride ck_on(true);
    mt::NTileStrategyOverride rounds(2);
    mm::clear_aocl_ck_weight_cache();
    const int old_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
    const Bf16Case c(8, 2048, 64, /*transB=*/true);
    matmul_params p = c.params(data_type_t::bf16);
    p.num_threads = 8;
    p.weight_cache_type = 1;
    std::vector<bfloat16_t> dst(static_cast<size_t>(c.M) * c.N);
    EXPECT_TRUE(mm::custom_kernel_routable('r', c.M, c.N, c.K, p.num_threads, 1,
            false, true, 1.0f, 0.0f, nullptr, c.ldb, true, p));
    std::string mode;
    {
        mt::GemmModeCaptureGuard capture;
        const status_t st = routed_call(c.M, c.N, c.K, c.src.data(), c.K,
                c.wei.data(), c.ldb, true, dst.data(), c.N, p);
        const char *captured
                = mm::test_api::s_last_group_matmul_direct_gemm_mode.load(
                        std::memory_order_relaxed);
        omp_set_dynamic(old_dynamic);
        ASSERT_EQ(st, status_t::success);
        ASSERT_NE(captured, nullptr);
        mode = captured;
    }
    EXPECT_NE(mode.find("custom"), std::string::npos) << mode;
    EXPECT_EQ(native::native_weight_layouts(c.wei.data()), 0u);
}

// ── AOCL-DLP prepacks under process ALGO 11 ─────────────────────────

/// An aocl_dlp_blocked prepack of w, in a buffer of the size the prepack
/// asks for.
std::vector<uint8_t> aocl_prepack(const void *w, data_type_t wei,
        data_type_t src, int K, int N, int ldb, bool transB, int group = 0) {
    rdr::reorder_params_t rp = prepack_request(matmul_algo_t::aocl_dlp_blocked,
            wei, src, K, N, ldb, transB, false, 0, group);
    rp.num_threads = kThreads;
    const size_t bytes = rdr::weight_prepack_size(rp);
    std::vector<uint8_t> out(bytes, 0);
    EXPECT_GT(bytes, 0u);
    EXPECT_EQ(rdr::reorder_direct(w, out.data(), rp), status_t::success);
    return out;
}

/// The request an ALGO 11 call on an AOCL prepack makes of the native weight
/// cache.
native::native_raw_weight_t aocl_request(const void *buffer, data_type_t wei,
        int K, int N, bool transB, bool interleave = false) {
    native::native_raw_weight_t r;
    r.wei_dt = wei;
    r.weight = buffer;
    r.K = K;
    r.N = N;
    r.ldb = transB ? K : N;
    r.transB = transB;
    r.pack_nr = kNr;
    r.interleave_split_halves = interleave;
    r.aocl_blocked = true;
    return r;
}

matmul_params on_aocl_prepack(matmul_params p, matmul_algo_t tag, int32_t wc) {
    p.mem_format_b = 'r';
    p.lowoha_algo = tag;
    p.weight_cache_type = wc;
    return p;
}

bool aocl_inverse_available(const matmul_params &p) {
    return native::aocl_prepack_unreorderable(p);
}

/// The reference algorithm on c's raw weight, within rel of got.
template <typename Case>
void expect_reference_algorithm(const Case &c, bool transB,
        const std::vector<float> &got, double rel = 1e-4) {
    matmul_params p = c.params();
    p.lowoha_algo = matmul_algo_t::reference;
    std::vector<float> want(got.size(), kCanary);
    ASSERT_EQ(routed_call(c.M, c.N, c.K, c.src.data(), c.K, c.wei.data(), c.ldb,
                      transB, want.data(), c.N, p),
            status_t::success);
    size_t bad = 0;
    for (size_t i = 0; i < got.size(); ++i)
        if (std::fabs(got[i] - want[i]) > rel * std::fabs(want[i]) + 1e-3)
            ++bad;
    EXPECT_EQ(bad, 0u) << "native result differs from the reference algorithm";
}

/// Zeroes buf but its first and last 64 bytes, which the fingerprint of a
/// layout cached from it reads: a call served from that layout gives the
/// same result, a call that reads the buffer again a zero weight.
void zero_between_the_ends(std::vector<uint8_t> &buf) {
    if (buf.size() > 128)
        std::fill(buf.begin() + 64, buf.end() - 64, uint8_t {0});
}

/// Two calls of p on `packed`, the AOCL prepack of a weight whose native
/// image is `image`: each is one native call writing `want`.  After the first
/// the buffer holds that image and serves the layout (weight cache 2) or
/// still holds the AOCL prepack (0, 1); at weight cache 1 the second call
/// reads only the ends of the buffer.
void expect_aocl_prepack_served_natively(std::vector<uint8_t> &packed,
        data_type_t wei, int M, int N, int K, const void *src, bool transB,
        int ldb, const matmul_params &p, const std::vector<float> &want,
        const std::vector<uint8_t> &image) {
    const int32_t wc = p.weight_cache_type;
    const std::vector<uint8_t> aocl_bytes = packed;
    for (int call = 0; call < 2; ++call) {
        SCOPED_TRACE("call " + std::to_string(call));
        std::vector<float> dst(want.size(), kCanary);
        ASSERT_EQ(routed_call(M, N, K, src, K, packed.data(), ldb, transB,
                          dst.data(), N, p),
                status_t::success);
        if (wc != 0) {
            EXPECT_GE(native::native_weight_layouts(packed.data()), 1u);
        }
        EXPECT_EQ(std::memcmp(
                          dst.data(), want.data(), dst.size() * sizeof(float)),
                0);
        if (call == 1) break;
        if (wc == 2) {
            ASSERT_TRUE(packed_in_place(packed));
            EXPECT_EQ(std::memcmp(packed.data(), image.data(),
                              native_image_bytes(wei, K, N)),
                    0);
            native::native_layout_t l;
            ASSERT_EQ(native::get_native_layout(
                              aocl_request(packed.data(), wei, K, N, transB), 1,
                              0, 1, l),
                    status_t::success);
            EXPECT_EQ(l.data, static_cast<const void *>(packed.data()))
                    << "the image itself serves the layout";
            EXPECT_EQ(l.owned, nullptr);
        } else {
            EXPECT_FALSE(packed_in_place(packed));
            EXPECT_EQ(packed, aocl_bytes);
            if (wc == 1) zero_between_the_ends(packed);
        }
    }
}

TEST_F(NativeLayoutRuleRoute, AoclPrepackedBf16IsUnreorderedOnceUnderAlgo11) {
    if (!has_bf16()) GTEST_SKIP() << "needs AVX512-BF16";
    if (!aocl_inverse_available(Bf16Case(1, 32, 32).params()))
        GTEST_SKIP() << "no AOCL inverse in this build";
    for (int K : {63, 64})
        for (bool transB : {false, true})
            for (matmul_algo_t tag :
                    {matmul_algo_t::aocl_dlp_blocked, matmul_algo_t::none})
                for (int32_t wc : {0, 1, 2}) {
                    SCOPED_TRACE("K=" + std::to_string(K)
                            + " transB=" + std::to_string(transB)
                            + " tag=" + mm::kernel_to_string(tag)
                            + " wc=" + std::to_string(wc));
                    reset_grp_matmul_caches();
                    config_.cfg.set_algo(kAlgoNative);
                    const Bf16Case c(5, 96, K, transB);
                    std::vector<uint8_t> packed
                            = aocl_prepack(c.wei.data(), data_type_t::bf16,
                                    data_type_t::bf16, c.K, c.N, c.ldb, transB);
                    const std::vector<uint8_t> image
                            = native_image(c.wei.data(), data_type_t::bf16, c.K,
                                    c.N, c.ldb, transB);
                    std::vector<float> want(
                            static_cast<size_t>(c.M) * c.N, kCanary);
                    ASSERT_EQ(direct_call(c.M, c.N, c.K, c.src.data(), c.K,
                                      weight_ref(c.wei.data(), kNr, 0), c.ldb,
                                      transB, want.data(), c.N, c.params()),
                            status_t::success);
                    expect_matches_reference(c, want, c.N);
                    expect_reference_algorithm(c, transB, want);
                    expect_aocl_prepack_served_natively(packed,
                            data_type_t::bf16, c.M, c.N, c.K, c.src.data(),
                            transB, c.ldb, on_aocl_prepack(c.params(), tag, wc),
                            want, image);
                }
}

TEST_F(NativeLayoutRuleRoute, AoclPrepackedInt8IsUnreorderedOnceUnderAlgo11) {
    if (!has_vnni()) GTEST_SKIP() << "needs AVX512-VNNI";
    if (!aocl_inverse_available(Int8Case(1, 32, 32).params()))
        GTEST_SKIP() << "no AOCL inverse in this build";
    for (int K : {64, 66})
        for (matmul_algo_t tag :
                {matmul_algo_t::aocl_dlp_blocked, matmul_algo_t::none})
            for (int32_t wc : {0, 1, 2}) {
                SCOPED_TRACE("K=" + std::to_string(K)
                        + " tag=" + mm::kernel_to_string(tag)
                        + " wc=" + std::to_string(wc));
                reset_grp_matmul_caches();
                config_.cfg.set_algo(kAlgoNative);
                const Int8Case c(5, 96, K);
                std::vector<uint8_t> packed
                        = aocl_prepack(c.wei.data(), data_type_t::s8,
                                data_type_t::u8, c.K, c.N, c.ldb, false);
                const std::vector<uint8_t> image = native_image(
                        c.wei.data(), data_type_t::s8, c.K, c.N, c.ldb, false);
                std::vector<float> want(
                        static_cast<size_t>(c.M) * c.N, kCanary);
                ASSERT_EQ(direct_call(c.M, c.N, c.K, c.src.data(), c.K,
                                  weight_ref(c.wei.data(), kNr, 0), c.ldb,
                                  false, want.data(), c.N, c.params()),
                        status_t::success);
                expect_matches_reference(c, want, c.N);
                expect_reference_algorithm(c, false, want);
                expect_aocl_prepack_served_natively(packed, data_type_t::s8,
                        c.M, c.N, c.K, c.src.data(), false, c.ldb,
                        on_aocl_prepack(c.params(), tag, wc), want, image);
            }
}

TEST_F(NativeLayoutRuleRoute, AoclPrepackedWeightKeepsAoclOnTheDefaultRoute) {
    for (matmul_algo_t tag :
            {matmul_algo_t::aocl_dlp_blocked, matmul_algo_t::none})
        for (int32_t wc : {1, 2}) {
            SCOPED_TRACE(std::string("tag=") + mm::kernel_to_string(tag)
                    + " wc=" + std::to_string(wc));
            reset_grp_matmul_caches();
            config_.cfg.set_algo(kAlgoNone);
            {
                const Bf16Case c(4, 96, 64);
                std::vector<uint8_t> packed
                        = aocl_prepack(c.wei.data(), data_type_t::bf16,
                                data_type_t::bf16, c.K, c.N, c.ldb, false);
                const std::vector<uint8_t> aocl_bytes = packed;
                std::vector<float> dst(static_cast<size_t>(c.M) * c.N, kCanary);
                ASSERT_EQ(routed_call(c.M, c.N, c.K, c.src.data(), c.K,
                                  packed.data(), c.ldb, false, dst.data(), c.N,
                                  on_aocl_prepack(c.params(), tag, wc)),
                        status_t::success);
                EXPECT_EQ(native::native_weight_layouts(packed.data()), 0u);
                EXPECT_EQ(packed, aocl_bytes);
                EXPECT_FALSE(packed_in_place(packed));
                expect_matches_reference(c, dst, c.N);
            }
            {
                const Int8Case c(5, 96, 64);
                std::vector<uint8_t> packed
                        = aocl_prepack(c.wei.data(), data_type_t::s8,
                                data_type_t::u8, c.K, c.N, c.ldb, false);
                const std::vector<uint8_t> aocl_bytes = packed;
                std::vector<float> dst(static_cast<size_t>(c.M) * c.N, kCanary);
                ASSERT_EQ(routed_call(c.M, c.N, c.K, c.src.data(), c.K,
                                  packed.data(), c.ldb, false, dst.data(), c.N,
                                  on_aocl_prepack(c.params(), tag, wc)),
                        status_t::success);
                EXPECT_EQ(native::native_weight_layouts(packed.data()), 0u);
                EXPECT_EQ(packed, aocl_bytes);
                EXPECT_FALSE(packed_in_place(packed));
                expect_matches_reference(c, dst, c.N);
            }
        }
}

TEST_F(NativeLayoutRuleRoute, AoclSymQuantPrepackStaysOnAoclUnderAlgo11) {
    for (int32_t wc : {1, 2}) {
        SCOPED_TRACE("wc=" + std::to_string(wc));
        reset_grp_matmul_caches();
        const DynQuantCase c(data_type_t::bf16);
        std::vector<uint8_t> packed
                = aocl_prepack(c.wei.data(), data_type_t::s8, data_type_t::s8,
                        c.K, c.N, c.N, false, /*group=*/c.K);
        const std::vector<uint8_t> aocl_bytes = packed;
        const matmul_params p = on_aocl_prepack(
                c.params(), matmul_algo_t::aocl_dlp_blocked, wc);
        std::vector<float> want(static_cast<size_t>(c.M) * c.N, kCanary);
        std::vector<float> dst(want.size(), kCanary);
        config_.cfg.set_algo(kAlgoNone);
        ASSERT_EQ(routed_call(c.M, c.N, c.K, c.src(), c.K, packed.data(), c.N,
                          false, want.data(), c.N, p),
                status_t::success);
        config_.cfg.set_algo(kAlgoNative);
        ASSERT_EQ(routed_call(c.M, c.N, c.K, c.src(), c.K, packed.data(), c.N,
                          false, dst.data(), c.N, p),
                status_t::success);
        EXPECT_EQ(native::native_weight_layouts(packed.data()), 0u);
        EXPECT_EQ(std::memcmp(
                          dst.data(), want.data(), dst.size() * sizeof(float)),
                0);
        EXPECT_EQ(packed, aocl_bytes);
        EXPECT_FALSE(packed_in_place(packed));
        expect_matches_reference(c, dst, c.N);
    }
}

TEST_F(NativeLayoutRuleRoute, AoclPrepackedCallNativeDoesNotServeStaysOnAocl) {
    if (!has_vnni()) GTEST_SKIP() << "needs AVX512-VNNI";
    const Int8Case c(5, 96, 64);
    if (!aocl_inverse_available(c.params()))
        GTEST_SKIP() << "no AOCL inverse in this build";
    int32_t wei_zp = 2;
    for (int32_t wc : {1, 2}) {
        SCOPED_TRACE("wc=" + std::to_string(wc));
        reset_grp_matmul_caches();
        std::vector<uint8_t> packed = aocl_prepack(c.wei.data(),
                data_type_t::s8, data_type_t::u8, c.K, c.N, c.ldb, false);
        const std::vector<uint8_t> aocl_bytes = packed;
        // The native BRGEMM has no weight zero point.
        matmul_params p = on_aocl_prepack(
                c.params(), matmul_algo_t::aocl_dlp_blocked, wc);
        p.quant_params.wei_zp.buff = &wei_zp;
        p.quant_params.wei_zp.dt = data_type_t::s32;
        p.quant_params.wei_zp.dims = {1};
        std::vector<float> want(static_cast<size_t>(c.M) * c.N, kCanary);
        std::vector<float> dst(want.size(), kCanary);
        config_.cfg.set_algo(kAlgoNone);
        ASSERT_EQ(routed_call(c.M, c.N, c.K, c.src.data(), c.K, packed.data(),
                          c.ldb, false, want.data(), c.N, p),
                status_t::success);
        config_.cfg.set_algo(kAlgoNative);
        ASSERT_EQ(routed_call(c.M, c.N, c.K, c.src.data(), c.K, packed.data(),
                          c.ldb, false, dst.data(), c.N, p),
                status_t::success);
        EXPECT_EQ(native::native_weight_layouts(packed.data()), 0u);
        EXPECT_EQ(std::memcmp(
                          dst.data(), want.data(), dst.size() * sizeof(float)),
                0);
        EXPECT_EQ(packed, aocl_bytes);
        EXPECT_FALSE(packed_in_place(packed));
        size_t bad = 0;
        for (int m = 0; m < c.M; ++m)
            for (int n = 0; n < c.N; ++n) {
                double s = 0.0;
                for (int k = 0; k < c.K; ++k)
                    s += (static_cast<double>(
                                  c.src[static_cast<size_t>(m) * c.K + k])
                                 - c.src_zp)
                            * (c.wei[static_cast<size_t>(k) * c.ldb + n]
                                    - wei_zp);
                const double ref = s * c.src_scale * c.wei_scale;
                if (std::fabs(dst[static_cast<size_t>(m) * c.N + n] - ref)
                        > 1e-4 * std::fabs(ref) + 1e-3)
                    ++bad;
            }
        EXPECT_EQ(bad, 0u);
    }
}

TEST_F(NativeLayoutRuleRoute, AoclPrepackedGatedActivationUnderAlgo11) {
    if (!has_bf16()) GTEST_SKIP() << "needs AVX512-BF16";
    const Bf16Case c(3, 128, 64);
    if (!aocl_inverse_available(c.params()))
        GTEST_SKIP() << "no AOCL inverse in this build";
    config_.cfg.set_algo(kAlgoNative);
    for (matmul_gated_act_t act :
            {matmul_gated_act_t::silu_and_mul, matmul_gated_act_t::gelu_and_mul,
                    matmul_gated_act_t::swiglu_oai_mul})
        for (int32_t wc : {1, 2}) {
            SCOPED_TRACE("gated_act " + std::to_string(static_cast<int>(act))
                    + " wc=" + std::to_string(wc));
            reset_grp_matmul_caches();
            const bool interleave = act != matmul_gated_act_t::swiglu_oai_mul;
            std::vector<uint8_t> packed
                    = aocl_prepack(c.wei.data(), data_type_t::bf16,
                            data_type_t::bf16, c.K, c.N, c.ldb, false);
            const std::vector<uint8_t> image = native_image(c.wei.data(),
                    data_type_t::bf16, c.K, c.N, c.ldb, false, interleave);
            matmul_params p = c.params(data_type_t::bf16);
            native::native_weight_ref_t ref = weight_ref(c.wei.data(), kNr, 0);
            ref.interleave_split_halves = interleave;
            const bfloat16_t canary(kCanary);
            const int ldc = c.N / 2;
            std::vector<bfloat16_t> want(
                    static_cast<size_t>(c.M) * ldc, canary);
            ASSERT_EQ(direct_call(c.M, c.N, c.K, c.src.data(), c.K, ref, c.ldb,
                              false, want.data(), ldc, p, nullptr, act),
                    status_t::success);
            for (int call = 0; call < 2; ++call) {
                SCOPED_TRACE("call " + std::to_string(call));
                std::vector<bfloat16_t> dst(want.size(), canary);
                ASSERT_EQ(routed_call(c.M, c.N, c.K, c.src.data(), c.K,
                                  packed.data(), c.ldb, false, dst.data(), ldc,
                                  on_aocl_prepack(p,
                                          matmul_algo_t::aocl_dlp_blocked, wc),
                                  1, true, nullptr, act),
                        status_t::success);
                EXPECT_GE(native::native_weight_layouts(packed.data()), 1u);
                EXPECT_EQ(std::memcmp(dst.data(), want.data(),
                                  dst.size() * sizeof(bfloat16_t)),
                        0);
            }
            EXPECT_EQ(packed_in_place(packed), wc == 2);
            if (wc == 2) {
                EXPECT_EQ(std::memcmp(packed.data(), image.data(),
                                  native_image_bytes(
                                          data_type_t::bf16, c.K, c.N)),
                        0);
            }
        }
}

TEST_F(NativeLayoutRuleRoute, AoclPrepackedImageThatDoesNotFitPacksOutOfPlace) {
    if (!has_bf16()) GTEST_SKIP() << "needs AVX512-BF16";
    const Bf16Case c(4, 96, 64);
    if (!aocl_inverse_available(c.params()))
        GTEST_SKIP() << "no AOCL inverse in this build";
    config_.cfg.set_algo(kAlgoNative);
    std::vector<uint8_t> packed = aocl_prepack(c.wei.data(), data_type_t::bf16,
            data_type_t::bf16, c.K, c.N, c.ldb, false);
    const std::vector<uint8_t> aocl_bytes = packed;
    const std::vector<uint8_t> image = native_image(
            c.wei.data(), data_type_t::bf16, c.K, c.N, c.ldb, false);
    std::vector<float> want(static_cast<size_t>(c.M) * c.N, kCanary);
    ASSERT_EQ(direct_call(c.M, c.N, c.K, c.src.data(), c.K,
                      weight_ref(c.wei.data(), kNr, 0), c.ldb, false,
                      want.data(), c.N, c.params()),
            status_t::success);
    matmul_params p = on_aocl_prepack(
            c.params(), matmul_algo_t::aocl_dlp_blocked, /*wc=*/2);
    p.wei_buffer_capacity_bytes
            = native_image_bytes(data_type_t::bf16, c.K, c.N) - 1;
    for (int call = 0; call < 2; ++call) {
        SCOPED_TRACE("call " + std::to_string(call));
        std::vector<float> dst(want.size(), kCanary);
        ASSERT_EQ(routed_call(c.M, c.N, c.K, c.src.data(), c.K, packed.data(),
                          c.ldb, false, dst.data(), c.N, p),
                status_t::success);
        EXPECT_GE(native::native_weight_layouts(packed.data()), 1u);
        EXPECT_EQ(std::memcmp(
                          dst.data(), want.data(), dst.size() * sizeof(float)),
                0);
        EXPECT_FALSE(packed_in_place(packed));
        if (call == 0) {
            EXPECT_EQ(packed, aocl_bytes);
            // The cached layout serves the next call.
            zero_between_the_ends(packed);
        }
    }
}

TEST_F(NativeLayoutRuleRoute, AoclPrepackedConcurrentFirstCallsPackOnce) {
    if (!has_bf16()) GTEST_SKIP() << "needs AVX512-BF16";
    const Bf16Case c(4, 256, 128);
    if (!aocl_inverse_available(c.params()))
        GTEST_SKIP() << "no AOCL inverse in this build";
    config_.cfg.set_algo(kAlgoNative);
    const std::vector<uint8_t> image = native_image(
            c.wei.data(), data_type_t::bf16, c.K, c.N, c.ldb, false);
    std::vector<float> want(static_cast<size_t>(c.M) * c.N, kCanary);
    ASSERT_EQ(direct_call(c.M, c.N, c.K, c.src.data(), c.K,
                      weight_ref(c.wei.data(), kNr, 0), c.ldb, false,
                      want.data(), c.N, c.params()),
            status_t::success);
    constexpr int kCallers = 8;
    for (int32_t wc : {1, 2}) {
        SCOPED_TRACE("wc=" + std::to_string(wc));
        reset_grp_matmul_caches();
        std::vector<uint8_t> packed = aocl_prepack(c.wei.data(),
                data_type_t::bf16, data_type_t::bf16, c.K, c.N, c.ldb, false);
        std::vector<uint8_t> layouts = packed;
        const void *seen[kCallers] = {};
        int callers = 0, bad = 0;
#pragma omp parallel num_threads(kCallers) reduction(+ : bad)
        {
            native::native_layout_t l;
            if (native::get_native_layout(
                        aocl_request(layouts.data(), data_type_t::bf16, c.K,
                                c.N, false),
                        wc, 0, 2,
                        l) != status_t::success
                    || std::memcmp(l.data, image.data(),
                               native_image_bytes(data_type_t::bf16, c.K, c.N))
                            != 0)
                ++bad;
            seen[omp_get_thread_num()] = l.data;
            if (omp_get_thread_num() == 0) callers = omp_get_num_threads();
            native::native_free_layout(l.owned);
        }
        EXPECT_EQ(bad, 0);
        for (int i = 1; i < callers; ++i)
            EXPECT_EQ(seen[i], seen[0]) << "caller " << i;
        EXPECT_EQ(packed_in_place(layouts), wc == 2);

        matmul_params p = on_aocl_prepack(
                c.params(), matmul_algo_t::aocl_dlp_blocked, wc);
        p.num_threads = 1;
        std::vector<std::vector<float>> dst(
                kCallers, std::vector<float>(want.size(), kCanary));
        bad = 0;
#pragma omp parallel num_threads(kCallers) reduction(+ : bad)
        {
            std::vector<float> &out = dst[omp_get_thread_num()];
            if (routed_call(c.M, c.N, c.K, c.src.data(), c.K, packed.data(),
                        c.ldb, false, out.data(), c.N,
                        p) != status_t::success
                    || std::memcmp(out.data(), want.data(),
                               out.size() * sizeof(float))
                            != 0)
                ++bad;
            if (omp_get_thread_num() == 0) callers = omp_get_num_threads();
        }
        EXPECT_EQ(bad, 0);
        if (wc != 0) {
            EXPECT_GE(native::native_weight_layouts(packed.data()), 1u);
        }
        EXPECT_EQ(packed_in_place(packed), wc == 2);
        if (wc == 2) {
            EXPECT_EQ(std::memcmp(packed.data(), image.data(),
                              native_image_bytes(data_type_t::bf16, c.K, c.N)),
                    0);
        }
    }
}

TEST_F(NativeLayoutRuleRoute, ClearingNativeCachesDropsAoclPrepackLayouts) {
    if (!has_bf16()) GTEST_SKIP() << "needs AVX512-BF16";
    const Bf16Case c(4, 96, 64);
    if (!aocl_inverse_available(c.params()))
        GTEST_SKIP() << "no AOCL inverse in this build";
    config_.cfg.set_algo(kAlgoNative);
    std::vector<uint8_t> packed = aocl_prepack(c.wei.data(), data_type_t::bf16,
            data_type_t::bf16, c.K, c.N, c.ldb, false);
    const matmul_params p1 = on_aocl_prepack(
            c.params(), matmul_algo_t::aocl_dlp_blocked, /*wc=*/1);
    std::vector<float> dst(static_cast<size_t>(c.M) * c.N, kCanary);
    ASSERT_EQ(routed_call(c.M, c.N, c.K, c.src.data(), c.K, packed.data(),
                      c.ldb, false, dst.data(), c.N, p1),
            status_t::success);
    expect_matches_reference(c, dst, c.N);

    // Another weight prepacked into the same buffer is read once the cached
    // layout is gone.
    std::vector<bfloat16_t> negated(c.wei.size());
    for (size_t i = 0; i < negated.size(); ++i)
        negated[i] = bfloat16_t(-static_cast<float>(c.wei[i]));
    rdr::reorder_params_t rp = prepack_request(matmul_algo_t::aocl_dlp_blocked,
            data_type_t::bf16, data_type_t::bf16, c.K, c.N, c.ldb, false,
            false);
    rp.num_threads = kThreads;
    ASSERT_EQ(rdr::weight_prepack_size(rp), packed.size());
    ASSERT_EQ(rdr::reorder_direct(negated.data(), packed.data(), rp),
            status_t::success);
    native::clear_all_weight_caches();
    std::fill(dst.begin(), dst.end(), kCanary);
    ASSERT_EQ(routed_call(c.M, c.N, c.K, c.src.data(), c.K, packed.data(),
                      c.ldb, false, dst.data(), c.N, p1),
            status_t::success);
    size_t bad = 0;
    for (int m = 0; m < c.M; ++m)
        for (int n = 0; n < c.N; ++n) {
            const double want = -c.ref(m, n);
            if (std::fabs(dst[static_cast<size_t>(m) * c.N + n] - want)
                    > 1e-4 * std::fabs(want) + 1e-3)
                ++bad;
        }
    EXPECT_EQ(bad, 0u);

    std::vector<uint8_t> in_place = aocl_prepack(c.wei.data(),
            data_type_t::bf16, data_type_t::bf16, c.K, c.N, c.ldb, false);
    ASSERT_EQ(routed_call(c.M, c.N, c.K, c.src.data(), c.K, in_place.data(),
                      c.ldb, false, dst.data(), c.N,
                      on_aocl_prepack(c.params(),
                              matmul_algo_t::aocl_dlp_blocked, /*wc=*/2)),
            status_t::success);
    ASSERT_TRUE(packed_in_place(in_place));
    native::clear_all_weight_caches();
    EXPECT_FALSE(packed_in_place(in_place));
}

TEST_F(NativeLayoutRuleRoute, AoclPrepackMadeWithAnMHintStaysOnAocl) {
    const Bf16Case c(4, 96, 64);
    struct MHintGuard {
        matmul_config_t &cfg = matmul_config_t::instance();
        const int64_t old = cfg.get_dlp_m_hint();
        ~MHintGuard() { cfg.set_dlp_m_hint(old); }
    } m_hint;
    m_hint.cfg.set_dlp_m_hint(c.M);
    EXPECT_FALSE(aocl_inverse_available(c.params()));
    config_.cfg.set_algo(kAlgoNative);
    std::vector<uint8_t> packed = aocl_prepack(c.wei.data(), data_type_t::bf16,
            data_type_t::bf16, c.K, c.N, c.ldb, false);
    const std::vector<uint8_t> aocl_bytes = packed;
    std::vector<float> dst(static_cast<size_t>(c.M) * c.N, kCanary);
    ASSERT_EQ(routed_call(c.M, c.N, c.K, c.src.data(), c.K, packed.data(),
                      c.ldb, false, dst.data(), c.N,
                      on_aocl_prepack(c.params(),
                              matmul_algo_t::aocl_dlp_blocked, /*wc=*/2)),
            status_t::success);
    EXPECT_EQ(native::native_weight_layouts(packed.data()), 0u);
    EXPECT_EQ(packed, aocl_bytes);
    EXPECT_FALSE(packed_in_place(packed));
    expect_matches_reference(c, dst, c.N);
}

} // namespace
