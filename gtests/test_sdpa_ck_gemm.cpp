/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

/// Flash SDPA BF16 GEMM on the custom kernel against AOCL-DLP `zendnn_gemm`,
/// and flash SDPA with the custom kernel on and off.

#include "common/zendnnl_global.hpp"
#include "lowoha_operators/sdpa/flash_sdpa/lowoha_sdpa_flash_cpu_common.hpp"
#include "lowoha_operators/sdpa/lowoha_sdpa.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <ostream>
#include <random>
#include <vector>

namespace {

using zendnnl::common::bfloat16_t;
using zendnnl::common::data_type_t;
using zendnnl::common::status_t;
namespace sd = zendnnl::lowoha::sdpa;
namespace ck = zendnnl::lowoha::sdpa::ck;
namespace cg = zendnnl::lowoha::sdpa::ck::bf16;

/// Sets the BF16 custom-kernel mode (0 off, 1 GEMM only) and counts the GEMM
/// calls CK takes; restores the environment's mode on exit.
struct ck_guard {
    explicit ck_guard(int value) {
        ck::test_api::set_bf16_mode(value);
        ck::test_api::count_calls(true);
    }
    ~ck_guard() {
        ck::test_api::count_calls(false);
        ck::test_api::set_bf16_mode(-1);
    }
    uint64_t calls() const { return ck::test_api::bf16_gemm_calls(); }
};

/// The ISA `ck::bf16_mode()` requires.
bool has_bf16() {
    const auto &pi = zendnnl::common::zendnnl_platform_info();
    return pi.get_avx512_bf16_status() && pi.get_avx512_bw_vl_status();
}

std::vector<bfloat16_t> random_bf16(size_t n, std::mt19937 &rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<bfloat16_t> v(n);
    for (auto &x : v)
        x = bfloat16_t(nd(rng));
    return v;
}

struct gemm_case {
    int64_t m, n, k;
    bool trans_b;
    float beta;
    int64_t pad; ///< extra elements per row of A, B and C
};

void PrintTo(const gemm_case &g, std::ostream *os) {
    *os << "M" << g.m << " N" << g.n << " K" << g.k
        << (g.trans_b ? " B^T" : " B") << " beta" << g.beta << " pad" << g.pad;
}

class SdpaCkGemm : public ::testing::TestWithParam<gemm_case> {
protected:
    void SetUp() override {
        if (!has_bf16()) GTEST_SKIP() << "needs AVX512-BF16/BW/VL";
    }
};

TEST_P(SdpaCkGemm, MatchesAocl) {
    const gemm_case g = GetParam();
    ck_guard guard(0);
    std::mt19937 rng(20261017);
    const int64_t lda = g.k + g.pad;
    const int64_t ldb = (g.trans_b ? g.k : g.n) + g.pad;
    const int64_t ldc = g.n + g.pad;
    const auto a = random_bf16(g.m * lda, rng);
    const auto b = random_bf16((g.trans_b ? g.n : g.k) * ldb, rng);
    std::vector<float> c_ref(g.m * ldc);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto &x : c_ref)
        x = nd(rng);
    std::vector<float> c_fly = c_ref, c_pre = c_ref;

    sd::zendnn_gemm<bfloat16_t>(g.m, g.n, g.k, 1.f, a.data(), lda, b.data(),
            ldb, g.beta, c_ref.data(), ldc, false, g.trans_b);
    ASSERT_TRUE(cg::gemm(g.m, g.n, g.k, 1.f, a.data(), lda, b.data(), ldb,
            g.beta, c_fly.data(), ldc, false, g.trans_b));
    std::vector<bfloat16_t> packed(cg::packed_size(g.k, g.n));
    cg::pack_b(g.k, g.n, b.data(), ldb, g.trans_b, packed.data());
    cg::gemm_packed(g.m, g.n, g.k, a.data(), lda, packed.data(), g.beta == 1.f,
            c_pre.data(), ldc);

    for (int64_t i = 0; i < g.m; ++i)
        for (int64_t j = 0; j < ldc; ++j) {
            const size_t x = i * ldc + j;
            // Both accumulate BF16 products in FP32; only the order differs.
            const float tol = 1e-4f * (1.f + std::fabs(c_ref[x])) * g.k;
            EXPECT_NEAR(c_fly[x], c_ref[x], tol) << "row " << i << " col " << j;
            EXPECT_EQ(c_pre[x], c_fly[x]) << "row " << i << " col " << j;
        }
}

INSTANTIATE_TEST_SUITE_P(Shapes, SdpaCkGemm,
        ::testing::Values(gemm_case {3, 1, 1, true, 0.f, 0},
                gemm_case {7, 33, 5, true, 0.f, 3},
                gemm_case {7, 33, 5, false, 1.f, 3},
                gemm_case {8, 32, 64, true, 0.f, 0},
                gemm_case {13, 50, 127, true, 1.f, 1},
                gemm_case {13, 50, 127, false, 1.f, 1},
                gemm_case {64, 476, 64, true, 0.f, 0},
                gemm_case {256, 128, 512, false, 1.f, 0},
                gemm_case {33, 64, 476, false, 0.f, 0},
                gemm_case {256, 512, 128, true, 0.f, 0}));

TEST(SdpaCkGemmApi, RejectsUncoveredCalls) {
    if (!has_bf16()) GTEST_SKIP() << "needs AVX512-BF16/BW/VL";
    std::vector<bfloat16_t> a(16, bfloat16_t(1.f)), b(16, bfloat16_t(1.f));
    std::vector<float> c(16, 7.f);
    EXPECT_FALSE(cg::gemm(4, 4, 4, 2.f, a.data(), 4, b.data(), 4, 0.f, c.data(),
            4, false, true));
    EXPECT_FALSE(cg::gemm(4, 4, 4, 1.f, a.data(), 4, b.data(), 4, 0.5f,
            c.data(), 4, false, true));
    EXPECT_FALSE(cg::gemm(4, 4, 4, 1.f, a.data(), 4, b.data(), 4, 0.f, c.data(),
            4, true, true));
    EXPECT_FALSE(cg::gemm(1, 4, 4, 1.f, a.data(), 4, b.data(), 4, 0.f, c.data(),
            4, false, true));
    EXPECT_FALSE(cg::gemm(2, 4, 4, 1.f, a.data(), 4, b.data(), 4, 1.f, c.data(),
            4, false, false));
    for (float x : c)
        EXPECT_EQ(x, 7.f);
}

// zendnn_gemm runs covered calls on CK and leaves the others to AOCL-DLP.
TEST(SdpaCkGemmApi, ZendnnGemmRoutes) {
    if (!has_bf16()) GTEST_SKIP() << "needs AVX512-BF16/BW/VL";
    std::mt19937 rng(20261017);
    const int64_t m = 16, n = 40, k = 24;
    const auto a = random_bf16(m * k, rng);
    const auto b = random_bf16(n * k, rng);
    struct call {
        int64_t m;
        float alpha;
        uint64_t ck_calls;
    };
    for (const call &x : {call {m, 1.f, 1}, call {m, 0.5f, 0}, call {2, 1.f, 0},
                 call {1, 1.f, 0}}) {
        std::vector<float> ref(m * n, 0.f), got(m * n, 0.f);
        {
            ck_guard guard(0);
            sd::zendnn_gemm<bfloat16_t>(x.m, n, k, x.alpha, a.data(), k,
                    b.data(), k, 0.f, ref.data(), n, false, true);
            EXPECT_EQ(guard.calls(), 0u);
        }
        {
            ck_guard guard(1);
            sd::zendnn_gemm<bfloat16_t>(x.m, n, k, x.alpha, a.data(), k,
                    b.data(), k, 0.f, got.data(), n, false, true);
            EXPECT_EQ(guard.calls(), x.ck_calls)
                    << "m " << x.m << " alpha " << x.alpha;
        }
        for (size_t i = 0; i < ref.size(); ++i)
            EXPECT_NEAR(got[i], ref[i], 1e-4f * (1.f + std::fabs(ref[i])) * k)
                    << "m " << x.m << " alpha " << x.alpha << " index " << i;
    }
}

struct sdpa_case {
    int64_t batch, heads, kv_heads, seq, kv_seq, head_dim;
    bool causal;
    int64_t window; ///< 0 = off
    bool bshd;
    bool int8_qk = false; ///< INT8 Q*K^T, BF16 P*V
    int mask = 0; ///< additive [S, KV] mask: 0 none, 1 F32, 2 BF16
};

void PrintTo(const sdpa_case &c, std::ostream *os) {
    *os << "B" << c.batch << " H" << c.heads << " KVH" << c.kv_heads << " S"
        << c.seq << " KV" << c.kv_seq << " D" << c.head_dim
        << (c.causal ? " causal" : "") << " W" << c.window
        << (c.bshd ? " BSHD" : " BHSD") << (c.int8_qk ? " int8-qk" : "")
        << (c.mask == 1                  ? " f32-mask"
                           : c.mask == 2 ? " bf16-mask"
                                         : "");
}

std::vector<bfloat16_t> run_sdpa(const sdpa_case &c,
        const std::vector<bfloat16_t> &q, const std::vector<bfloat16_t> &k,
        const std::vector<bfloat16_t> &v) {
    auto strides = [&](int64_t h, int64_t s, int64_t d, int64_t(&out)[4]) {
        out[0] = h * s * d;
        out[1] = c.bshd ? d : s * d;
        out[2] = c.bshd ? h * d : d;
        out[3] = 1;
    };
    int64_t qs[4], ks[4];
    strides(c.heads, c.seq, c.head_dim, qs);
    strides(c.kv_heads, c.kv_seq, c.head_dim, ks);
    sd::sdpa_params p;
    p.batch = c.batch;
    p.num_heads = c.heads;
    p.kv_num_heads = c.kv_heads;
    p.seq_len = c.seq;
    p.kv_seq_len = c.kv_seq;
    p.head_dim = c.head_dim;
    p.q_stride_b = p.o_stride_b = qs[0];
    p.q_stride_h = p.o_stride_h = qs[1];
    p.q_stride_s = p.o_stride_s = qs[2];
    p.q_stride_d = p.o_stride_d = qs[3];
    p.k_stride_b = p.v_stride_b = ks[0];
    p.k_stride_h = p.v_stride_h = ks[1];
    p.k_stride_s = p.v_stride_s = ks[2];
    p.k_stride_d = p.v_stride_d = ks[3];
    p.qkv_dt = data_type_t::bf16;
    p.out_dt = data_type_t::bf16;
    p.scale = 1.0 / std::sqrt(static_cast<double>(c.head_dim));
    p.is_causal = c.causal;
    p.sliding_window = c.window > 0;
    p.sliding_window_size = c.window;
    p.is_qk_quant = c.int8_qk;
    p.kernel = sd::sdpa_kernel_t::flash;
    // Additive mask in [-2, 0], the same for every call of a case.
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> ud(-2.f, 0.f);
    std::vector<float> mask_f32(c.mask ? c.seq * c.kv_seq : 0);
    for (auto &x : mask_f32)
        x = ud(rng);
    std::vector<bfloat16_t> mask_bf16(mask_f32.begin(), mask_f32.end());
    const void *mask = nullptr;
    if (c.mask) {
        p.mask_ndims = 2;
        p.mask_sizes[0] = c.seq;
        p.mask_sizes[1] = c.kv_seq;
        p.mask_strides[0] = c.kv_seq;
        p.mask_strides[1] = 1;
        p.mask_dt = c.mask == 1 ? data_type_t::f32 : data_type_t::bf16;
        mask = c.mask == 1 ? static_cast<const void *>(mask_f32.data())
                           : static_cast<const void *>(mask_bf16.data());
    }
    std::vector<bfloat16_t> out(q.size(), bfloat16_t(0.f));
    EXPECT_EQ(
            sd::sdpa_direct(q.data(), k.data(), v.data(), mask, out.data(), p),
            status_t::success);
    return out;
}

class SdpaCkFlash : public ::testing::TestWithParam<sdpa_case> {
protected:
    void SetUp() override {
        if (!has_bf16()) GTEST_SKIP() << "needs AVX512-BF16/BW/VL";
    }
};

TEST_P(SdpaCkFlash, MatchesAoclFlash) {
    const sdpa_case c = GetParam();
    if (c.int8_qk
            && !zendnnl::common::zendnnl_platform_info()
                        .get_avx512_vnni_status())
        GTEST_SKIP() << "needs AVX512-VNNI";
    std::mt19937 rng(20261017);
    const auto q = random_bf16(c.batch * c.heads * c.seq * c.head_dim, rng);
    const auto k
            = random_bf16(c.batch * c.kv_heads * c.kv_seq * c.head_dim, rng);
    const auto v = random_bf16(k.size(), rng);
    std::vector<bfloat16_t> ref, got;
    {
        ck_guard guard(0);
        ref = run_sdpa(c, q, k, v);
        EXPECT_EQ(guard.calls(), 0u);
    }
    {
        ck_guard guard(1);
        got = run_sdpa(c, q, k, v);
        // Q tiles of fewer than 3 rows (decode) stay on AOCL-DLP.
        if (c.seq >= 3)
            EXPECT_GT(guard.calls(), 0u);
        else
            EXPECT_EQ(guard.calls(), 0u);
    }
    float max_diff = 0.f;
    for (size_t x = 0; x < ref.size(); ++x) {
        const float y = static_cast<float>(got[x]);
        ASSERT_TRUE(std::isfinite(y)) << "index " << x;
        max_diff
                = std::max(max_diff, std::fabs(static_cast<float>(ref[x]) - y));
    }
    // Same FP32 math; the BF16 output may differ by one rounding step.
    EXPECT_LE(max_diff, 2e-2f);
}

INSTANTIATE_TEST_SUITE_P(Shapes, SdpaCkFlash,
        ::testing::Values(
                // Whisper-style encoder, D 64.
                sdpa_case {1, 4, 4, 600, 600, 64, false, 0, false},
                // Causal GQA, ragged KV tail over two KV blocks.
                sdpa_case {2, 8, 2, 700, 700, 128, true, 0, true},
                // Gemma-style GQA with a sliding window, D 256.
                sdpa_case {2, 3, 1, 600, 600, 256, false, 128, true},
                // Cross-attention, D 96 and odd lengths over two KV blocks.
                sdpa_case {3, 6, 3, 201, 701, 96, false, 0, false},
                // Decode: one-row GEMMs stay on AOCL-DLP.
                sdpa_case {2, 8, 2, 1, 1000, 128, false, 0, true},
                // F32 mask, causal GQA; BF16 mask with a window.
                sdpa_case {1, 4, 2, 300, 600, 64, true, 0, true, false, 1},
                sdpa_case {2, 4, 4, 200, 700, 128, false, 64, false, false, 2},
                // INT8 Q*K^T, BF16 P*V on CK; also masked with a window.
                sdpa_case {1, 4, 4, 300, 600, 64, true, 0, false, true},
                sdpa_case {1, 4, 2, 300, 600, 64, false, 128, true, true, 1}));

} // namespace
