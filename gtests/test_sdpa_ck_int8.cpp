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

/// Dynamic-INT8 flash SDPA matmuls on the custom kernel against an exact
/// integer reference and AOCL-DLP, and INT8 flash SDPA with the custom kernel
/// on and off.

#include "common/zendnnl_global.hpp"
#include "lowoha_operators/sdpa/flash_sdpa/lowoha_sdpa_int8_utils.hpp"
#include "lowoha_operators/sdpa/lowoha_sdpa.hpp"
#include "lowoha_operators/sdpa/sdpa_gemm_ck/lowoha_sdpa_ck.hpp"

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

/// The ISA `ck::int8_mode()` requires.
bool has_ck_isa() {
    const auto &pi = zendnnl::common::zendnnl_platform_info();
    return pi.get_avx512_vnni_status() && pi.get_avx512_bw_vl_status();
}

/// Sets the INT8 custom-kernel mode (0 off, 1 GEMM) and counts the CK calls;
/// restores the environment's mode on exit.
struct ck_guard {
    explicit ck_guard(int value) {
        ck::test_api::set_int8_mode(value);
        ck::test_api::count_calls(true);
    }
    ~ck_guard() {
        ck::test_api::count_calls(false);
        ck::test_api::set_int8_mode(-1);
    }
    uint64_t calls() const { return ck::test_api::int8_gemm_calls(); }
};

struct gemm_case {
    int64_t m, n, k;
};

void PrintTo(const gemm_case &g, std::ostream *os) {
    *os << "M" << g.m << " N" << g.n << " K" << g.k;
}

template <typename T>
std::vector<T> random_int(size_t n, int lo, int hi, std::mt19937 &rng) {
    std::uniform_int_distribution<int> ud(lo, hi);
    std::vector<T> v(n);
    for (auto &x : v)
        x = static_cast<T>(ud(rng));
    return v;
}

std::vector<float> random_scales(size_t n, std::mt19937 &rng) {
    std::uniform_real_distribution<float> ud(0.001f, 0.05f);
    std::vector<float> v(n);
    for (auto &x : v)
        x = ud(rng);
    return v;
}

class SdpaCkInt8Gemm : public ::testing::TestWithParam<gemm_case> {
protected:
    void SetUp() override {
        if (!has_ck_isa()) GTEST_SKIP() << "needs AVX512-VNNI/BW/VL";
    }
};

// The int32 sums are exact; only the scaling may round differently.
float tolerance(double ref) {
    return float(1e-5 * std::fabs(ref) + 1e-6);
}

TEST_P(SdpaCkInt8Gemm, QkMatchesReferenceAndAocl) {
    const gemm_case g = GetParam();
    std::mt19937 rng(20261018);
    const auto q = random_int<int8_t>(g.m * g.k, -127, 127, rng);
    const auto k = random_int<int8_t>(g.n * g.k, -127, 127, rng);
    const auto qs = random_scales(g.m, rng);
    const auto ks = random_scales(g.n, rng);
    std::vector<float> aocl(g.m * g.n, 7.f), got(g.m * g.n, 7.f);
    {
        ck_guard guard(0);
        ASSERT_EQ(sd::sdpa_qk_int8_matmul(q.data(), k.data(), aocl.data(),
                          int(g.m), int(g.n), int(g.k), qs.data(), ks.data()),
                status_t::success);
        EXPECT_EQ(guard.calls(), 0u);
    }
    {
        ck_guard guard(1);
        ASSERT_EQ(sd::sdpa_qk_int8_matmul(q.data(), k.data(), got.data(),
                          int(g.m), int(g.n), int(g.k), qs.data(), ks.data()),
                status_t::success);
        EXPECT_EQ(guard.calls(), 1u);
    }
    for (int64_t i = 0; i < g.m; ++i)
        for (int64_t j = 0; j < g.n; ++j) {
            int64_t sum = 0;
            for (int64_t x = 0; x < g.k; ++x)
                sum += int64_t(q[i * g.k + x]) * k[j * g.k + x];
            const double ref = double(sum) * qs[i] * ks[j];
            const size_t at = size_t(i * g.n + j);
            EXPECT_NEAR(got[at], ref, tolerance(ref))
                    << "row " << i << " col " << j;
            EXPECT_NEAR(got[at], aocl[at], tolerance(ref))
                    << "row " << i << " col " << j;
        }
}

TEST_P(SdpaCkInt8Gemm, PvMatchesReferenceAndAocl) {
    const gemm_case g = GetParam();
    std::mt19937 rng(20261019);
    const auto p = random_int<uint8_t>(g.m * g.k, 0, 255, rng);
    const auto v = random_int<int8_t>(g.k * g.n, -127, 127, rng);
    const auto vs = random_scales(g.n, rng);
    const float ps = 1.f / 255.f;
    std::vector<float> aocl(g.m * g.n, 7.f), got(g.m * g.n, 7.f);
    {
        ck_guard guard(0);
        ASSERT_EQ(sd::sdpa_pv_int8_matmul(p.data(), v.data(), aocl.data(),
                          int(g.m), int(g.n), int(g.k), &ps, vs.data()),
                status_t::success);
    }
    {
        ck_guard guard(1);
        ASSERT_EQ(sd::sdpa_pv_int8_matmul(p.data(), v.data(), got.data(),
                          int(g.m), int(g.n), int(g.k), &ps, vs.data()),
                status_t::success);
        EXPECT_EQ(guard.calls(), 1u);
    }
    for (int64_t i = 0; i < g.m; ++i)
        for (int64_t j = 0; j < g.n; ++j) {
            int64_t sum = 0;
            for (int64_t x = 0; x < g.k; ++x)
                sum += int64_t(p[i * g.k + x]) * v[x * g.n + j];
            const double ref = double(sum) * ps * vs[j];
            const size_t at = size_t(i * g.n + j);
            EXPECT_NEAR(got[at], ref, tolerance(ref))
                    << "row " << i << " col " << j;
            EXPECT_NEAR(got[at], aocl[at], tolerance(ref))
                    << "row " << i << " col " << j;
        }
}

INSTANTIATE_TEST_SUITE_P(Shapes, SdpaCkInt8Gemm,
        ::testing::Values(gemm_case {2, 1, 1}, gemm_case {3, 5, 2},
                gemm_case {7, 33, 5}, gemm_case {8, 32, 64},
                gemm_case {13, 50, 127}, gemm_case {9, 17, 3},
                gemm_case {32, 512, 64}, gemm_case {64, 476, 128},
                gemm_case {33, 512, 256}, gemm_case {256, 128, 512},
                gemm_case {64, 96, 509}));

// One-row P*V stays on AOCL-DLP, which leaves it correct; one-row Q*K^T
// (decode) runs on CK.
TEST(SdpaCkInt8GemmApi, RoutesOneRowPvToAocl) {
    if (!has_ck_isa()) GTEST_SKIP() << "needs AVX512-VNNI/BW/VL";
    std::vector<int8_t> a(64, 3), b(64 * 64, -2);
    std::vector<uint8_t> pa(64, 200);
    std::vector<float> s(64, 0.01f), out(64, 7.f);
    const float ps = 0.5f;
    EXPECT_FALSE(ck::int8::pv(
            pa.data(), b.data(), out.data(), 1, 64, 64, ps, s.data()));
    for (float x : out)
        EXPECT_EQ(x, 7.f);
    {
        ck_guard guard(1);
        ASSERT_EQ(sd::sdpa_pv_int8_matmul(pa.data(), b.data(), out.data(), 1,
                          64, 64, &ps, s.data()),
                status_t::success);
        EXPECT_EQ(guard.calls(), 0u);
        for (float x : out)
            EXPECT_NEAR(x, 200 * -2 * 64 * ps * 0.01f, 1e-4f);
    }
    {
        ck_guard guard(1);
        ASSERT_EQ(sd::sdpa_qk_int8_matmul(a.data(), b.data(), out.data(), 1, 64,
                          64, s.data(), s.data()),
                status_t::success);
        EXPECT_EQ(guard.calls(), 1u);
        for (float x : out)
            EXPECT_NEAR(x, 3 * -2 * 64 * 0.01f * 0.01f, 1e-6f);
    }
}

struct sdpa_case {
    int64_t batch, heads, kv_heads, seq, kv_seq, head_dim;
    bool causal;
    int64_t window; ///< 0 = off
    bool qk_quant, pv_quant;
};

void PrintTo(const sdpa_case &c, std::ostream *os) {
    *os << "B" << c.batch << " H" << c.heads << " KVH" << c.kv_heads << " S"
        << c.seq << " KV" << c.kv_seq << " D" << c.head_dim
        << (c.causal ? " causal" : "") << " W" << c.window
        << (c.qk_quant ? " int8-qk" : "") << (c.pv_quant ? " int8-pv" : "");
}

std::vector<bfloat16_t> random_bf16(size_t n, std::mt19937 &rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<bfloat16_t> v(n);
    for (auto &x : v)
        x = bfloat16_t(nd(rng));
    return v;
}

std::vector<bfloat16_t> run_sdpa(const sdpa_case &c,
        const std::vector<bfloat16_t> &q, const std::vector<bfloat16_t> &k,
        const std::vector<bfloat16_t> &v) {
    sd::sdpa_params p;
    p.batch = c.batch;
    p.num_heads = c.heads;
    p.kv_num_heads = c.kv_heads;
    p.seq_len = c.seq;
    p.kv_seq_len = c.kv_seq;
    p.head_dim = c.head_dim;
    p.q_stride_b = p.o_stride_b = c.heads * c.seq * c.head_dim;
    p.q_stride_h = p.o_stride_h = c.seq * c.head_dim;
    p.q_stride_s = p.o_stride_s = c.head_dim;
    p.q_stride_d = p.o_stride_d = 1;
    p.k_stride_b = p.v_stride_b = c.kv_heads * c.kv_seq * c.head_dim;
    p.k_stride_h = p.v_stride_h = c.kv_seq * c.head_dim;
    p.k_stride_s = p.v_stride_s = c.head_dim;
    p.k_stride_d = p.v_stride_d = 1;
    p.qkv_dt = data_type_t::bf16;
    p.out_dt = data_type_t::bf16;
    p.scale = 1.0 / std::sqrt(static_cast<double>(c.head_dim));
    p.is_causal = c.causal;
    p.sliding_window = c.window > 0;
    p.sliding_window_size = c.window;
    p.is_qk_quant = c.qk_quant;
    p.is_pv_quant = c.pv_quant;
    p.kernel = sd::sdpa_kernel_t::flash;
    std::vector<bfloat16_t> out(q.size(), bfloat16_t(0.f));
    EXPECT_EQ(sd::sdpa_direct(
                      q.data(), k.data(), v.data(), nullptr, out.data(), p),
            status_t::success);
    return out;
}

class SdpaCkInt8Flash : public ::testing::TestWithParam<sdpa_case> {
protected:
    void SetUp() override {
        if (!has_ck_isa()) GTEST_SKIP() << "needs AVX512-VNNI/BW/VL";
    }
};

TEST_P(SdpaCkInt8Flash, MatchesAoclInt8Flash) {
    const sdpa_case c = GetParam();
    std::mt19937 rng(20261018);
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
        EXPECT_GT(guard.calls(), 0u);
    }
    float max_diff = 0.f;
    for (size_t x = 0; x < ref.size(); ++x) {
        const float y = static_cast<float>(got[x]);
        ASSERT_TRUE(std::isfinite(y)) << "index " << x;
        max_diff
                = std::max(max_diff, std::fabs(static_cast<float>(ref[x]) - y));
    }
    // Same integer sums; the scaling and BF16 output may round differently.
    EXPECT_LE(max_diff, 2e-2f);
}

INSTANTIATE_TEST_SUITE_P(Shapes, SdpaCkInt8Flash,
        ::testing::Values(
                // Both INT8 matmuls: encoder, causal GQA over two KV
                // blocks, a window with D 256, cross-attention odd lengths.
                sdpa_case {1, 4, 4, 600, 600, 64, false, 0, true, true},
                sdpa_case {2, 8, 2, 700, 700, 128, true, 0, true, true},
                sdpa_case {2, 3, 1, 600, 600, 256, false, 128, true, true},
                sdpa_case {3, 6, 3, 201, 701, 96, false, 0, true, true},
                // INT8 Q*K^T only (BF16 P*V); INT8 P*V only (BF16 Q*K^T).
                sdpa_case {1, 4, 2, 300, 600, 64, true, 0, true, false},
                sdpa_case {1, 4, 2, 300, 600, 64, true, 0, false, true},
                // Decode: one-row Q*K^T on CK, P*V on AOCL-DLP.
                sdpa_case {2, 8, 2, 1, 1000, 128, false, 0, true, true}));

} // namespace
