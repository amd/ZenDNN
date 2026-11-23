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

/// CK ukernel module — W4A8 (s4) sibling of `test_ukernel_int8.cpp`.
///
/// Drives `select_w4a8_ukernel` against an EXACT scalar reference, so a
/// failure here is the kernel's arithmetic, not an integration or
/// tolerance artefact:
///
///   facc[m][v] = sum_g ( sum_{k in g} A_s8[m,k] * W_s4[k,v] )
///                * wei_scale[g][v]
///   C[m][v]    = facc[m][v] * src_scale[m] + bias[v]
///
/// Sweeps every MR x NV, group sizes 8 through K, both scale dtypes,
/// every BiasKind, the gated acts, and the selector bounds.
///
/// Tolerance is 1% relative: the BF16 store puts the floor near 0.4%.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "common/bfloat16.hpp"
#include "common/zendnnl_compat.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/pack.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/ukernel/int8_microkernel.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/ukernel/s4_microkernel.hpp"

namespace ck = ::zendnnl::lowoha::matmul::custom_kernel;
using zendnnl::common::bfloat16_t;
using zendnnl::error_handling::status_t;

namespace {

int s4_true(uint8_t raw) {
    return (raw & 0x08u) ? (static_cast<int>(raw) - 16) : static_cast<int>(raw);
}
uint8_t nib_at(int k, int n) {
    uint32_t h = static_cast<uint32_t>(k) * 2246822519u
            + static_cast<uint32_t>(n) * 3266489917u;
    return static_cast<uint8_t>((h >> 11) & 0x0Fu);
}
int8_t a_at(int m, int k) {
    uint32_t h = static_cast<uint32_t>(m) * 374761393u
            + static_cast<uint32_t>(k) * 668265263u;
    return static_cast<int8_t>((h >> 9) & 0xFFu);
}
float sscale_at(int m) {
    return 0.01f + 0.003f * static_cast<float>(m % 7);
}
float sscale_at(int m, int g) {
    return 0.01f + 0.002f * static_cast<float>((m * 5 + g * 3) % 9);
}
float wscale_at(int g, int n) {
    return 0.02f + 0.001f * static_cast<float>((g * 13 + n) % 11);
}
float bias_at(int n) {
    return 0.5f - 0.05f * static_cast<float>(n % 9);
}

float silu(float x) {
    return x / (1.0f + std::exp(-x));
}
float gelu_erf(float x) {
    return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752f));
}
// Kept out of the gated sweep below: re-deriving `swiglu_oai_avx512`
// here would check the test's math, not the library's.

struct Slab {
    void *p = nullptr;
    explicit Slab(size_t b) : p(zendnnl_aligned_alloc(64, b)) {
        if (p) std::memset(p, 0, b);
    }
    ~Slab() { zendnnl_aligned_free(p); }
    Slab(const Slab &) = delete;
    Slab &operator=(const Slab &) = delete;
};

std::vector<int8_t> build_packed(int K, int N, int ldb, bool transB) {
    std::vector<int8_t> buf((static_cast<size_t>(N) * K + 1) / 2, 0);
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n) {
            const uint8_t raw = nib_at(k, n);
            const size_t idx = transB ? static_cast<size_t>(n) * ldb + k
                                      : static_cast<size_t>(k) * ldb + n;
            auto &byte = reinterpret_cast<uint8_t &>(buf[idx >> 1]);
            if ((idx & 1u) == 0u)
                byte = static_cast<uint8_t>((byte & 0xF0u) | raw);
            else
                byte = static_cast<uint8_t>(
                        (byte & 0x0Fu) | static_cast<uint8_t>(raw << 4));
        }
    return buf;
}

// Exact per-(m, col) dequantised dot product, canonical column order.
// `bf16_scales` must match what the caller handed the kernel, or the
// comparison measures the test's rounding rather than the kernel's.
double ref_dot(int m, int col, int K, int group_size, bool bf16_scales,
        bool src_per_group = false) {
    const auto rnd = [bf16_scales](float x) -> double {
        return bf16_scales
                ? static_cast<double>(static_cast<float>(bfloat16_t(x)))
                : static_cast<double>(x);
    };
    const int G = K / group_size;
    double acc = 0.0;
    for (int g = 0; g < G; ++g) {
        long s32 = 0;
        for (int k = g * group_size; k < (g + 1) * group_size; ++k)
            s32 += static_cast<long>(a_at(m, k)) * s4_true(nib_at(k, col));
        const double src_scale = src_per_group ? rnd(sscale_at(m, g)) : 1.0;
        acc += static_cast<double>(s32) * rnd(wscale_at(g, col)) * src_scale;
    }
    return src_per_group ? acc : acc * rnd(sscale_at(m));
}

size_t oblock_bytes(int K, int pack_nr, int group_size) {
    return static_cast<size_t>(K / ck::kW4Octet) * pack_nr
            * ck::kW4BytesPerOctetCol
            + static_cast<size_t>(K / group_size) * pack_nr * sizeof(int32_t);
}

} // namespace

// act = none: every MR x NV x group_size, both scale dtypes, bias on
// and off, both caller layouts.
class CkUkernelS4None : public ::testing::TestWithParam<
                                std::tuple<int, int, int, bool, bool, bool>> {};

TEST_P(CkUkernelS4None, MatchesScalarReference) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    const auto [K, group_size, pack_nr, scale_bf16, with_bias, src_per_group]
            = GetParam();
    const int NV = pack_nr / 16;
    const int N = pack_nr * 2; // two o-blocks
    const int ldb = K; // transB = true
    const int max_mr = ck::max_mr_for_nv_s4(NV);

    const std::vector<int8_t> wsrc = build_packed(K, N, ldb, /*transB=*/true);
    Slab slab(ck::packed_weight_size_w4(K, N, pack_nr, group_size));
    ASSERT_NE(slab.p, nullptr);
    ASSERT_EQ(ck::prepack_weight_into_w4(wsrc.data(), K, N, ldb, pack_nr, true,
                      false, group_size, slab.p),
            status_t::success);

    const int G = K / group_size;
    for (int MR = 1; MR <= max_mr; ++MR) {
        auto fn = ck::select_w4a8_ukernel(MR, NV, ck::ActivationKind::none,
                ck::S4SourceKind::kS8, ck::S4OutputKind::kBf16,
                src_per_group ? ck::S4SrcScaleGranularity::kPerGroup
                              : ck::S4SrcScaleGranularity::kPerToken);
        ASSERT_NE(fn, nullptr) << "MR=" << MR << " NV=" << NV;

        std::vector<uint8_t> A(static_cast<size_t>(MR) * K);
        for (int m = 0; m < MR; ++m)
            for (int k = 0; k < K; ++k)
                A[static_cast<size_t>(m) * K + k]
                        = static_cast<uint8_t>(a_at(m, k));

        const int src_scale_groups = src_per_group ? G : 1;
        std::vector<float> ss_f(static_cast<size_t>(MR) * src_scale_groups),
                ws_f(static_cast<size_t>(G) * N), bias_f(N);
        std::vector<bfloat16_t> ss_b(
                static_cast<size_t>(MR) * src_scale_groups),
                ws_b(static_cast<size_t>(G) * N);
        for (int m = 0; m < MR; ++m)
            for (int g = 0; g < src_scale_groups; ++g) {
                const size_t idx
                        = static_cast<size_t>(m) * src_scale_groups + g;
                ss_f[idx] = src_per_group ? sscale_at(m, g) : sscale_at(m);
                ss_b[idx] = bfloat16_t(ss_f[idx]);
            }
        for (int g = 0; g < G; ++g)
            for (int n = 0; n < N; ++n) {
                ws_f[static_cast<size_t>(g) * N + n] = wscale_at(g, n);
                ws_b[static_cast<size_t>(g) * N + n]
                        = bfloat16_t(wscale_at(g, n));
            }
        for (int n = 0; n < N; ++n)
            bias_f[n] = bias_at(n);

        std::vector<bfloat16_t> C(
                static_cast<size_t>(MR) * N, bfloat16_t(0.0f));
        for (int ob = 0; ob < N / pack_nr; ++ob) {
            const int8_t *bp = static_cast<const int8_t *>(slab.p)
                    + static_cast<size_t>(ob)
                            * oblock_bytes(K, pack_nr, group_size);
            fn(A.data(), K, bp,
                    scale_bf16 ? static_cast<const void *>(ss_b.data())
                               : static_cast<const void *>(ss_f.data()),
                    scale_bf16 ? static_cast<const void *>(
                                         ws_b.data() + ob * pack_nr)
                               : static_cast<const void *>(
                                         ws_f.data() + ob * pack_nr),
                    scale_bf16 ? ck::ScaleKind::kBf16 : ck::ScaleKind::kF32,
                    with_bias ? static_cast<const void *>(
                                        bias_f.data() + ob * pack_nr)
                              : nullptr,
                    with_bias ? ck::BiasKind::fp32 : ck::BiasKind::none,
                    C.data() + ob * pack_nr, N, nullptr, 0, K, group_size, N,
                    nullptr, nullptr);
        }

        for (int m = 0; m < MR; ++m)
            for (int n = 0; n < N; ++n) {
                double ref = ref_dot(
                        m, n, K, group_size, scale_bf16, src_per_group);
                if (with_bias) ref += bias_at(n);
                const double got
                        = static_cast<float>(C[static_cast<size_t>(m) * N + n]);
                const double rel
                        = std::fabs(got - ref) / std::max(1.0, std::fabs(ref));
                ASSERT_LT(rel, 0.01) << "MR=" << MR << " m=" << m << " n=" << n
                                     << " ref=" << ref << " got=" << got;
            }
    }
}

INSTANTIATE_TEST_SUITE_P(Shapes, CkUkernelS4None,
        ::testing::Combine(::testing::Values(128, 256), // K
                ::testing::Values(8, 32, 128), // group_size
                ::testing::Values(32, 64), // pack_nr
                ::testing::Bool(), // bf16 scales
                ::testing::Bool(), // bias
                ::testing::Bool())); // per-token / per-group source scale

TEST(CkUkernelS4F32, StoresDequantizedF32) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";

    constexpr int K = 128;
    constexpr int group_size = 32;
    constexpr int pack_nr = 32;
    constexpr int N = pack_nr;
    constexpr int MR = 3;
    const int G = K / group_size;

    const std::vector<int8_t> wsrc = build_packed(K, N, K, true);
    Slab slab(ck::packed_weight_size_w4(K, N, pack_nr, group_size));
    ASSERT_NE(slab.p, nullptr);
    ASSERT_EQ(ck::prepack_weight_into_w4(wsrc.data(), K, N, K, pack_nr, true,
                      false, group_size, slab.p),
            status_t::success);

    auto fn = ck::select_w4a8_ukernel(MR, N / 16, ck::ActivationKind::none,
            ck::S4SourceKind::kS8, ck::S4OutputKind::kF32,
            ck::S4SrcScaleGranularity::kPerGroup);
    ASSERT_NE(fn, nullptr);

    std::vector<uint8_t> A(static_cast<size_t>(MR) * K);
    for (int m = 0; m < MR; ++m)
        for (int k = 0; k < K; ++k)
            A[static_cast<size_t>(m) * K + k]
                    = static_cast<uint8_t>(a_at(m, k));

    std::vector<float> src_scale(static_cast<size_t>(MR) * G);
    std::vector<float> wei_scale(static_cast<size_t>(G) * N);
    for (int m = 0; m < MR; ++m)
        for (int g = 0; g < G; ++g)
            src_scale[static_cast<size_t>(m) * G + g] = sscale_at(m, g);
    for (int g = 0; g < G; ++g)
        for (int n = 0; n < N; ++n)
            wei_scale[static_cast<size_t>(g) * N + n] = wscale_at(g, n);

    std::vector<float> C(static_cast<size_t>(MR) * N, 0.0f);
    fn(A.data(), K, static_cast<const int8_t *>(slab.p), src_scale.data(),
            wei_scale.data(), ck::ScaleKind::kF32, nullptr, ck::BiasKind::none,
            C.data(), N, nullptr, 0, K, group_size, N, nullptr, nullptr);

    for (int m = 0; m < MR; ++m)
        for (int n = 0; n < N; ++n) {
            const double ref = ref_dot(
                    m, n, K, group_size, false, /*src_per_group=*/true);
            const double got = C[static_cast<size_t>(m) * N + n];
            const double rel
                    = std::fabs(got - ref) / std::max(1.0, std::fabs(ref));
            ASSERT_LT(rel, 1e-5) << "m=" << m << " n=" << n << " ref=" << ref
                                 << " got=" << got;
        }
}

TEST(CkUkernelS4, PerGroupDequantizationAvoidsIntermediateOverflow) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";

    constexpr int K = 64;
    constexpr int group_size = 32;
    constexpr int N = 32;
    constexpr int G = K / group_size;
    constexpr int MR = 1;
    constexpr int NV = N / 16;

    // Every packed nibble is S4 +1 and every activation is quantized to 127.
    // Keep the first output channel at zero to exercise a zero dot with an
    // overflowing source-scale * weight-scale product.
    std::vector<int8_t> wsrc(
            static_cast<size_t>(N) * K / 2, static_cast<int8_t>(0x11));
    std::fill_n(wsrc.begin(), K / 2, static_cast<int8_t>(0));
    Slab slab(ck::packed_weight_size_w4(K, N, N, group_size));
    ASSERT_NE(slab.p, nullptr);
    ASSERT_EQ(ck::prepack_weight_into_w4(wsrc.data(), K, N, K, N, true,
                      /*interleave_split_halves=*/false, group_size, slab.p),
            status_t::success);

    std::vector<uint8_t> src(K, 127);
    const float activation = std::ldexp(1.0f, 125);
    const float src_scale_value = activation / 127.0f;
    const float weight_scale_value = std::ldexp(1.0f, -10);
    std::vector<float> src_scale(G, src_scale_value);
    std::vector<float> weight_scale(
            static_cast<size_t>(G) * N, weight_scale_value);
    for (int g = 0; g < G; ++g)
        weight_scale[static_cast<size_t>(g) * N] = std::ldexp(1.0f, 11);

    // Two groups each contribute (32 * 127) * (2^125 / 127) * 2^-10.
    const float expected = static_cast<float>(static_cast<double>(G)
            * group_size * 127.0 * static_cast<double>(src_scale_value)
            * static_cast<double>(weight_scale_value));
    ASSERT_TRUE(std::isfinite(expected));

    std::vector<float> f32_out(N, 0.0f);
    auto f32_fn = ck::select_w4a8_ukernel(MR, NV, ck::ActivationKind::none,
            ck::S4SourceKind::kS8, ck::S4OutputKind::kF32,
            ck::S4SrcScaleGranularity::kPerGroup);
    ASSERT_NE(f32_fn, nullptr);
    f32_fn(src.data(), K, static_cast<const int8_t *>(slab.p), src_scale.data(),
            weight_scale.data(), ck::ScaleKind::kF32, nullptr,
            ck::BiasKind::none, f32_out.data(), N, nullptr, 0, K, group_size, N,
            nullptr, nullptr);

    std::vector<bfloat16_t> bf16_out(N, bfloat16_t(0.0f));
    auto bf16_fn = ck::select_w4a8_ukernel(MR, NV, ck::ActivationKind::none,
            ck::S4SourceKind::kS8, ck::S4OutputKind::kBf16,
            ck::S4SrcScaleGranularity::kPerGroup);
    ASSERT_NE(bf16_fn, nullptr);
    bf16_fn(src.data(), K, static_cast<const int8_t *>(slab.p),
            src_scale.data(), weight_scale.data(), ck::ScaleKind::kF32, nullptr,
            ck::BiasKind::none, bf16_out.data(), N, nullptr, 0, K, group_size,
            N, nullptr, nullptr);

    const auto check = [expected](const char *output_kind, float zero_dot,
                               float nonzero_dot) {
        EXPECT_EQ(zero_dot, 0.0f) << "output=" << output_kind
                                  << " zero dot produced " << zero_dot;
        EXPECT_TRUE(std::isfinite(nonzero_dot))
                << "output=" << output_kind << " expected=" << expected
                << " observed=" << nonzero_dot;
        if (std::isfinite(nonzero_dot)) {
            EXPECT_NEAR(nonzero_dot, expected, expected * 0.01f)
                    << "output=" << output_kind << " expected=" << expected
                    << " observed=" << nonzero_dot;
        }
    };
    check("F32", f32_out[0], f32_out[1]);
    check("BF16", static_cast<float>(bf16_out[0]),
            static_cast<float>(bf16_out[1]));
}

// q=0, S=1, M'= -8 makes every weight zero, but S*dot and M'*sum are both
// huge.  Scaling either one by 2^125/127 before subtracting overflows.
TEST(CkUkernelS4, OffsetCancelsBeforeSourceScale) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    constexpr int K = 32;
    constexpr int group_size = 32;
    constexpr int N = 32;
    constexpr int MR = 1;
    constexpr int NV = N / 16;
    // Stored nibble 8 is signed -8.  Both nibbles of every byte.
    std::vector<int8_t> wsrc(
            static_cast<size_t>(N) * K / 2, static_cast<int8_t>(0x88));
    Slab slab(ck::packed_weight_size_w4(K, N, N, group_size));
    ASSERT_NE(slab.p, nullptr);
    ASSERT_EQ(ck::prepack_weight_into_w4(
                      wsrc.data(), K, N, K, N, true, false, group_size, slab.p),
            status_t::success);
    std::vector<uint8_t> src(K, 127);
    const float src_scale_value = std::ldexp(1.0f, 125) / 127.0f;
    std::vector<float> src_scale(1, src_scale_value);
    std::vector<float> weight_scale(N, 1.0f);
    std::vector<float> offset(N, -8.0f);
    std::vector<float> out(N, 1.0f);
    auto fn = ck::select_w4a8_ukernel(MR, NV, ck::ActivationKind::none,
            ck::S4SourceKind::kS8, ck::S4OutputKind::kF32,
            ck::S4SrcScaleGranularity::kPerGroup, /*has_offset=*/true);
    ASSERT_NE(fn, nullptr);
    fn(src.data(), K, static_cast<const int8_t *>(slab.p), src_scale.data(),
            weight_scale.data(), ck::ScaleKind::kF32, nullptr,
            ck::BiasKind::none, out.data(), N, nullptr, 0, K, group_size, N,
            offset.data(), nullptr);
    for (int n = 0; n < N; ++n) {
        EXPECT_EQ(out[n], 0.0f) << "n=" << n;
    }
}

// Sixteen +127 and sixteen -127.  The activation sum is zero, S is zero,
// and M' is large.  scale * M' overflows; multiplying that by the zero
// sum used to produce NaN.
TEST(CkUkernelS4, ZeroActivationSumSkipsOffset) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    constexpr int K = 32;
    constexpr int group_size = 32;
    constexpr int N = 32;
    constexpr int MR = 1;
    constexpr int NV = N / 16;
    std::vector<int8_t> wsrc(
            static_cast<size_t>(N) * K / 2, static_cast<int8_t>(0x88));
    Slab slab(ck::packed_weight_size_w4(K, N, N, group_size));
    ASSERT_NE(slab.p, nullptr);
    ASSERT_EQ(ck::prepack_weight_into_w4(
                      wsrc.data(), K, N, K, N, true, false, group_size, slab.p),
            status_t::success);
    std::vector<uint8_t> src(K);
    for (int k = 0; k < K; ++k) {
        src[k] = (k < 16) ? static_cast<uint8_t>(127)
                          : static_cast<uint8_t>(static_cast<int8_t>(-127));
    }
    const float src_scale_value = std::ldexp(1.0f, 125) / 127.0f;
    std::vector<float> src_scale(1, src_scale_value);
    std::vector<float> weight_scale(N, 0.0f);
    std::vector<float> offset(N, 2048.0f);
    std::vector<float> out(N, 1.0f);
    auto fn = ck::select_w4a8_ukernel(MR, NV, ck::ActivationKind::none,
            ck::S4SourceKind::kS8, ck::S4OutputKind::kF32,
            ck::S4SrcScaleGranularity::kPerGroup, /*has_offset=*/true);
    ASSERT_NE(fn, nullptr);
    fn(src.data(), K, static_cast<const int8_t *>(slab.p), src_scale.data(),
            weight_scale.data(), ck::ScaleKind::kF32, nullptr,
            ck::BiasKind::none, out.data(), N, nullptr, 0, K, group_size, N,
            offset.data(), nullptr);
    for (int n = 0; n < N; ++n) {
        EXPECT_EQ(out[n], 0.0f) << "n=" << n;
    }
}

// One shape, every scale / source / output / activation combination the
// offset epilogue has.  The offset varies by group and column, and the
// gated case permutes it with the gate/up pack.
TEST(CkUkernelS4, OffsetMatchesScalarAcrossVariants) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    constexpr int K = 64;
    constexpr int group_size = 32;
    constexpr int N = 32;
    constexpr int G = K / group_size;
    constexpr int MR = 1;
    constexpr int NV = 2;
    const auto wsrc = build_packed(K, N, K, /*transB=*/true);
    Slab plain(ck::packed_weight_size_w4(K, N, N, group_size));
    Slab gated(ck::packed_weight_size_w4(K, N, N, group_size));
    ASSERT_NE(plain.p, nullptr);
    ASSERT_NE(gated.p, nullptr);
    ASSERT_EQ(ck::prepack_weight_into_w4(wsrc.data(), K, N, K, N, true, false,
                      group_size, plain.p),
            status_t::success);
    ASSERT_EQ(ck::prepack_weight_into_w4(
                      wsrc.data(), K, N, K, N, true, true, group_size, gated.p),
            status_t::success);

    auto off_at = [](int g, int n) {
        return 0.03f * static_cast<float>((g * 5 + n * 2) % 9) - 0.08f;
    };
    std::vector<float> off_f(static_cast<size_t>(G) * N);
    std::vector<bfloat16_t> off_b(off_f.size());
    std::vector<float> ws_f(off_f.size());
    std::vector<bfloat16_t> ws_b(off_f.size());
    std::vector<float> off_perm(off_f.size());
    std::vector<float> ws_perm(off_f.size());
    constexpr int half = N / 2;
    for (int g = 0; g < G; ++g) {
        for (int n = 0; n < N; ++n) {
            const size_t i = static_cast<size_t>(g) * N + n;
            off_f[i] = off_at(g, n);
            off_b[i] = bfloat16_t(off_f[i]);
            ws_f[i] = wscale_at(g, n);
            ws_b[i] = bfloat16_t(ws_f[i]);
            const int canon = (n & 1) ? (half + (n >> 1)) : (n >> 1);
            off_perm[i] = off_at(g, canon);
            ws_perm[i] = wscale_at(g, canon);
        }
    }
    std::vector<uint8_t> s8(K);
    std::vector<uint8_t> u8(K);
    for (int k = 0; k < K; ++k) {
        s8[static_cast<size_t>(k)] = static_cast<uint8_t>(a_at(0, k));
        u8[static_cast<size_t>(k)] = static_cast<uint8_t>(a_at(0, k) ^ 0x80);
    }

    auto dot_at = [&](int col, bool per_group, bool bf16_scales) {
        const auto rnd = [&](float x) -> double {
            return bf16_scales
                    ? static_cast<double>(static_cast<float>(bfloat16_t(x)))
                    : static_cast<double>(x);
        };
        double acc = 0.0;
        for (int g = 0; g < G; ++g) {
            long dot = 0;
            long asum = 0;
            for (int k = g * group_size; k < (g + 1) * group_size; ++k) {
                const int a = a_at(0, k);
                dot += static_cast<long>(a) * s4_true(nib_at(k, col));
                asum += a;
            }
            double term = rnd(wscale_at(g, col)) * static_cast<double>(dot)
                    - rnd(off_at(g, col)) * static_cast<double>(asum);
            if (per_group) term *= rnd(sscale_at(0, g));
            acc += term;
        }
        if (!per_group) acc *= rnd(sscale_at(0));
        return acc;
    };

    auto run_plain = [&](ck::S4SourceKind source, ck::S4OutputKind output,
                             ck::S4SrcScaleGranularity gran, bool bf16_scales) {
        auto fn = ck::select_w4a8_ukernel(MR, NV, ck::ActivationKind::none,
                source, output, gran,
                /*has_offset=*/true);
        if (fn == nullptr) {
            ADD_FAILURE() << "missing ukernel";
            return;
        }
        const bool per_group = gran == ck::S4SrcScaleGranularity::kPerGroup;
        const int sg = per_group ? G : 1;
        std::vector<float> ss_f(static_cast<size_t>(sg));
        std::vector<bfloat16_t> ss_b(static_cast<size_t>(sg));
        for (int g = 0; g < sg; ++g) {
            ss_f[static_cast<size_t>(g)]
                    = per_group ? sscale_at(0, g) : sscale_at(0);
            ss_b[static_cast<size_t>(g)] = bfloat16_t(ss_f[g]);
        }
        std::vector<float> f32_out(N, 1.0f);
        std::vector<bfloat16_t> bf_out(N, bfloat16_t(1.0f));
        const uint8_t *src
                = source == ck::S4SourceKind::kU8Zp128 ? u8.data() : s8.data();
        fn(src, K, static_cast<const int8_t *>(plain.p),
                bf16_scales ? static_cast<const void *>(ss_b.data())
                            : static_cast<const void *>(ss_f.data()),
                bf16_scales ? static_cast<const void *>(ws_b.data())
                            : static_cast<const void *>(ws_f.data()),
                bf16_scales ? ck::ScaleKind::kBf16 : ck::ScaleKind::kF32,
                nullptr, ck::BiasKind::none,
                output == ck::S4OutputKind::kF32
                        ? static_cast<void *>(f32_out.data())
                        : static_cast<void *>(bf_out.data()),
                N, nullptr, 0, K, group_size, N,
                bf16_scales ? static_cast<const void *>(off_b.data())
                            : static_cast<const void *>(off_f.data()),
                nullptr);
        for (int n = 0; n < N; ++n) {
            const double got = output == ck::S4OutputKind::kF32
                    ? static_cast<double>(f32_out[static_cast<size_t>(n)])
                    : static_cast<double>(static_cast<float>(
                              bf_out[static_cast<size_t>(n)]));
            const double ref = dot_at(n, per_group, bf16_scales);
            const double rel
                    = std::fabs(got - ref) / std::max(1.0, std::fabs(ref));
            EXPECT_LT(rel, 0.02)
                    << "n=" << n << " ref=" << ref << " got=" << got;
        }
    };

    run_plain(ck::S4SourceKind::kS8, ck::S4OutputKind::kF32,
            ck::S4SrcScaleGranularity::kPerToken, false);
    run_plain(ck::S4SourceKind::kS8, ck::S4OutputKind::kF32,
            ck::S4SrcScaleGranularity::kPerGroup, false);
    run_plain(ck::S4SourceKind::kS8, ck::S4OutputKind::kBf16,
            ck::S4SrcScaleGranularity::kPerToken, true);
    run_plain(ck::S4SourceKind::kU8Zp128, ck::S4OutputKind::kF32,
            ck::S4SrcScaleGranularity::kPerToken, false);

    auto silu_fn = ck::select_w4a8_ukernel(MR, NV,
            ck::ActivationKind::silu_and_mul, ck::S4SourceKind::kS8,
            ck::S4OutputKind::kBf16, ck::S4SrcScaleGranularity::kPerToken,
            /*has_offset=*/true);
    ASSERT_NE(silu_fn, nullptr);
    std::vector<float> ss(1, sscale_at(0));
    std::vector<bfloat16_t> gated_out(half, bfloat16_t(1.0f));
    silu_fn(s8.data(), K, static_cast<const int8_t *>(gated.p), ss.data(),
            ws_perm.data(), ck::ScaleKind::kF32, nullptr, ck::BiasKind::none,
            nullptr, 0, gated_out.data(), half, K, group_size, N,
            off_perm.data(), nullptr);
    for (int i = 0; i < half; ++i) {
        const double gate = dot_at(i, false, false);
        const double up = dot_at(half + i, false, false);
        const double ref = silu(static_cast<float>(gate)) * up;
        const double got
                = static_cast<float>(gated_out[static_cast<size_t>(i)]);
        const double rel = std::fabs(got - ref) / std::max(1.0, std::fabs(ref));
        ASSERT_LT(rel, 0.02) << "i=" << i << " ref=" << ref << " got=" << got;
    }
}

// Gated acts against a CANONICAL-order reference: validates the
// split-halves pack permute, the per-group scale permute the N-tile
// hoist applies, and the pair-store epilogue as one contract.
class CkUkernelS4Gated
    : public ::testing::TestWithParam<std::tuple<int, int, int, int, bool>> {};

TEST_P(CkUkernelS4Gated, MatchesCanonicalReference) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    const auto [K, group_size, pack_nr, act_sel, src_per_group] = GetParam();
    const int NV = pack_nr / 16;
    const int N = pack_nr * 2;
    const int half = N / 2, ldb = K, G = K / group_size;
    const ck::ActivationKind act = (act_sel == 0)
            ? ck::ActivationKind::silu_and_mul
            : ck::ActivationKind::gelu_and_mul;

    const std::vector<int8_t> wsrc = build_packed(K, N, ldb, true);
    Slab slab(ck::packed_weight_size_w4(K, N, pack_nr, group_size));
    ASSERT_NE(slab.p, nullptr);
    // The pack permutes canonical [gate | up] into [g0, u0, g1, ...].
    ASSERT_EQ(ck::prepack_weight_into_w4(wsrc.data(), K, N, ldb, pack_nr, true,
                      /*interleave_split_halves=*/true, group_size, slab.p),
            status_t::success);

    const int max_mr = ck::max_mr_for_nv_s4(NV);
    for (int MR = 1; MR <= max_mr; ++MR) {
        auto fn = ck::select_w4a8_ukernel(MR, NV, act, ck::S4SourceKind::kS8,
                ck::S4OutputKind::kBf16,
                src_per_group ? ck::S4SrcScaleGranularity::kPerGroup
                              : ck::S4SrcScaleGranularity::kPerToken);
        ASSERT_NE(fn, nullptr);

        std::vector<uint8_t> A(static_cast<size_t>(MR) * K);
        for (int m = 0; m < MR; ++m)
            for (int k = 0; k < K; ++k)
                A[static_cast<size_t>(m) * K + k]
                        = static_cast<uint8_t>(a_at(m, k));
        const int src_scale_groups = src_per_group ? G : 1;
        std::vector<float> ss(static_cast<size_t>(MR) * src_scale_groups);
        for (int m = 0; m < MR; ++m)
            for (int g = 0; g < src_scale_groups; ++g)
                ss[static_cast<size_t>(m) * src_scale_groups + g]
                        = src_per_group ? sscale_at(m, g) : sscale_at(m);

        // Mirrors `materialise_f32_wei_scale(n_groups = G)`.
        std::vector<float> wperm(static_cast<size_t>(G) * N);
        for (int g = 0; g < G; ++g)
            for (int c = 0; c < N; ++c) {
                const int canon = (c & 1) ? (half + (c >> 1)) : (c >> 1);
                wperm[static_cast<size_t>(g) * N + c] = wscale_at(g, canon);
            }

        std::vector<bfloat16_t> C(
                static_cast<size_t>(MR) * half, bfloat16_t(0.0f));
        for (int ob = 0; ob < N / pack_nr; ++ob) {
            const int8_t *bp = static_cast<const int8_t *>(slab.p)
                    + static_cast<size_t>(ob)
                            * oblock_bytes(K, pack_nr, group_size);
            fn(A.data(), K, bp, ss.data(), wperm.data() + ob * pack_nr,
                    ck::ScaleKind::kF32, nullptr, ck::BiasKind::none, nullptr,
                    0, C.data() + ob * (pack_nr / 2), half, K, group_size, N,
                    nullptr, nullptr);
        }

        for (int m = 0; m < MR; ++m)
            for (int i = 0; i < half; ++i) {
                const double gate = ref_dot(m, i, K, group_size,
                        /*bf16_scales=*/false, src_per_group);
                const double up = ref_dot(m, half + i, K, group_size,
                        /*bf16_scales=*/false, src_per_group);
                const double ref = (act == ck::ActivationKind::silu_and_mul)
                        ? silu(static_cast<float>(gate)) * up
                        : gelu_erf(static_cast<float>(gate)) * up;
                const double got = static_cast<float>(
                        C[static_cast<size_t>(m) * half + i]);
                const double rel
                        = std::fabs(got - ref) / std::max(1.0, std::fabs(ref));
                ASSERT_LT(rel, 0.02) << "MR=" << MR << " m=" << m << " i=" << i
                                     << " ref=" << ref << " got=" << got;
            }
    }
}

INSTANTIATE_TEST_SUITE_P(Shapes, CkUkernelS4Gated,
        ::testing::Combine(::testing::Values(128, 256), // K
                ::testing::Values(32, 128), // group_size
                ::testing::Values(32, 64), // pack_nr
                ::testing::Values(0, 1), // silu / gelu
                ::testing::Bool())); // per-token / per-group source scale

TEST(CkUkernelS4, RoutedU8ModeMatchesS8AndStoresF32) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    constexpr int K = 128;
    constexpr int N = 32;
    constexpr int group_size = 32;
    constexpr int G = K / group_size;
    constexpr int NV = 2;
    const auto wsrc = build_packed(K, N, K, /*transB=*/true);
    Slab slab(ck::packed_weight_size_w4(K, N, N, group_size));
    ASSERT_NE(slab.p, nullptr);
    ASSERT_EQ(ck::prepack_weight_into_w4(wsrc.data(), K, N, K, N, true,
                      /*interleave_split_halves=*/false, group_size, slab.p),
            status_t::success);

    std::vector<float> scales(static_cast<size_t>(G) * N);
    for (int g = 0; g < G; ++g)
        for (int n = 0; n < N; ++n)
            scales[static_cast<size_t>(g) * N + n] = wscale_at(g, n);

    for (int MR = 1; MR <= ck::max_mr_for_nv_s4(NV); ++MR) {
        std::vector<uint8_t> s8(static_cast<size_t>(MR) * K);
        std::vector<uint8_t> u8(s8.size());
        std::vector<float> src_scales(static_cast<size_t>(MR));
        for (int m = 0; m < MR; ++m) {
            src_scales[static_cast<size_t>(m)] = sscale_at(m);
            for (int k = 0; k < K; ++k) {
                const uint8_t q = static_cast<uint8_t>(a_at(m, k));
                s8[static_cast<size_t>(m) * K + k] = q;
                u8[static_cast<size_t>(m) * K + k]
                        = static_cast<uint8_t>(q ^ 0x80u);
            }
        }

        const auto s8_fn
                = ck::select_w4a8_ukernel(MR, NV, ck::ActivationKind::none);
        const auto u8_fn
                = ck::select_w4a8_ukernel(MR, NV, ck::ActivationKind::none,
                        ck::S4SourceKind::kU8Zp128, ck::S4OutputKind::kBf16);
        const auto f32_fn
                = ck::select_w4a8_ukernel(MR, NV, ck::ActivationKind::none,
                        ck::S4SourceKind::kU8Zp128, ck::S4OutputKind::kF32);
        ASSERT_NE(s8_fn, nullptr);
        ASSERT_NE(u8_fn, nullptr);
        ASSERT_NE(f32_fn, nullptr);

        std::vector<bfloat16_t> from_s8(static_cast<size_t>(MR) * N);
        std::vector<bfloat16_t> from_u8(static_cast<size_t>(MR) * N);
        std::vector<float> from_u8_f32(static_cast<size_t>(MR) * N);
        const auto invoke
                = [&](ck::w4a8_ukernel_fn_t fn, const uint8_t *a, void *out) {
            fn(a, K, static_cast<const int8_t *>(slab.p), src_scales.data(),
                    scales.data(), ck::ScaleKind::kF32, nullptr,
                    ck::BiasKind::none, out, N, nullptr, 0, K, group_size, N,
                    nullptr, nullptr);
        };
        invoke(s8_fn, s8.data(), from_s8.data());
        invoke(u8_fn, u8.data(), from_u8.data());
        invoke(f32_fn, u8.data(), from_u8_f32.data());
        EXPECT_EQ(std::memcmp(from_u8.data(), from_s8.data(),
                          from_u8.size() * sizeof(bfloat16_t)),
                0)
                << "MR=" << MR;

        for (int m = 0; m < MR; ++m)
            for (int n = 0; n < N; ++n) {
                const double want
                        = ref_dot(m, n, K, group_size, /*bf16_scales=*/false);
                EXPECT_NEAR(from_u8_f32[static_cast<size_t>(m) * N + n], want,
                        2e-5 * std::max(1.0, std::fabs(want)))
                        << "MR=" << MR << " m=" << m << " n=" << n;
            }
    }

    EXPECT_EQ(ck::select_w4a8_ukernel(1, NV, ck::ActivationKind::silu_and_mul,
                      ck::S4SourceKind::kU8Zp128, ck::S4OutputKind::kF32),
            nullptr);
}

// An unpacked NR=32 block summed on the DQ-INT8 accumulators gives the bits
// of the in-register kernels for every row count up to 12, epilogue, output,
// source-scale layout and group size, from s8 or pre-biased u8 source bytes,
// whether its groups come in one unpacked chunk or one per chunk.
TEST(CkUkernelS4, UnpackedBlockMatchesInRegisterKernels) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    constexpr int K = 256;
    constexpr int N = 32;
    constexpr int NV = 2;
    constexpr int kMaxRows = 12;
    const auto wsrc = build_packed(K, N, K, /*transB=*/true);
    std::vector<float> bias(N);
    for (int n = 0; n < N; ++n)
        bias[static_cast<size_t>(n)] = bias_at(n);
    alignas(64) int32_t cs32[kMaxRows * N];

    for (const int group_size : {32, 128}) {
        const int G = K / group_size;
        Slab slab(ck::packed_weight_size_w4(K, N, N, group_size));
        ASSERT_NE(slab.p, nullptr);
        ASSERT_EQ(
                ck::prepack_weight_into_w4(wsrc.data(), K, N, K, N, true,
                        /*interleave_split_halves=*/false, group_size, slab.p),
                status_t::success);
        const auto *blk = static_cast<const int8_t *>(slab.p);
        Slab unpacked(static_cast<size_t>(K) * N);
        ASSERT_NE(unpacked.p, nullptr);
        const size_t group_packed_bytes
                = static_cast<size_t>(group_size / ck::kW4Octet) * N
                * ck::kW4BytesPerOctetCol;
        const auto *comp = reinterpret_cast<const int32_t *>(blk
                + static_cast<size_t>(K / ck::kW4Octet) * N
                        * ck::kW4BytesPerOctetCol);
        std::vector<float> wscales(static_cast<size_t>(G) * N);
        for (int g = 0; g < G; ++g)
            for (int n = 0; n < N; ++n)
                wscales[static_cast<size_t>(g) * N + n] = wscale_at(g, n);
        const std::vector<bfloat16_t> wscales_bf(
                wscales.begin(), wscales.end());

        for (const bool per_group : {false, true}) {
            const auto gran = per_group ? ck::S4SrcScaleGranularity::kPerGroup
                                        : ck::S4SrcScaleGranularity::kPerToken;
            const int sgroups = per_group ? G : 1;
            std::vector<float> sscales(static_cast<size_t>(kMaxRows) * sgroups);
            for (int m = 0; m < kMaxRows; ++m)
                for (int g = 0; g < sgroups; ++g)
                    sscales[static_cast<size_t>(m) * sgroups + g]
                            = per_group ? sscale_at(m, g) : sscale_at(m);
            const std::vector<bfloat16_t> sscales_bf(
                    sscales.begin(), sscales.end());
            std::vector<uint8_t> s8(static_cast<size_t>(kMaxRows) * K);
            std::vector<uint8_t> u8(s8.size());
            for (int m = 0; m < kMaxRows; ++m)
                for (int k = 0; k < K; ++k) {
                    const auto q = static_cast<uint8_t>(a_at(m, k));
                    s8[static_cast<size_t>(m) * K + k] = q;
                    u8[static_cast<size_t>(m) * K + k]
                            = static_cast<uint8_t>(q ^ 0x80u);
                }

            for (const bool bf16_scales : {false, true}) {
                const ck::ScaleKind sk = bf16_scales ? ck::ScaleKind::kBf16
                                                     : ck::ScaleKind::kF32;
                const void *ws = bf16_scales
                        ? static_cast<const void *>(wscales_bf.data())
                        : static_cast<const void *>(wscales.data());
                const auto ss = [&](int m0) -> const void * {
                    const size_t i = static_cast<size_t>(m0) * sgroups;
                    return bf16_scales
                            ? static_cast<const void *>(sscales_bf.data() + i)
                            : static_cast<const void *>(sscales.data() + i);
                };
                for (const auto act : {ck::ActivationKind::none,
                             ck::ActivationKind::swiglu_oai_mul,
                             ck::ActivationKind::silu_and_mul,
                             ck::ActivationKind::gelu_and_mul}) {
                    for (const auto out :
                            {ck::S4OutputKind::kBf16, ck::S4OutputKind::kF32}) {
                        const bool gated = act != ck::ActivationKind::none;
                        if (gated && out == ck::S4OutputKind::kF32) continue;
                        const auto gfn = ck::select_w4a8_unpacked_groups(gran);
                        const auto sfn = ck::select_w4a8_unpacked_store(
                                act, out, gran);
                        ASSERT_NE(sfn, nullptr);
                        const int cols = gated ? N / 2 : N;
                        const size_t elem
                                = out == ck::S4OutputKind::kF32 ? 4 : 2;
                        for (int rows = 1; rows <= kMaxRows; ++rows) {
                            SCOPED_TRACE("gs=" + std::to_string(group_size)
                                    + " per_group=" + std::to_string(per_group)
                                    + " act="
                                    + std::to_string(static_cast<int>(act))
                                    + " f32=" + std::to_string(elem == 4)
                                    + " bf16_scales="
                                    + std::to_string(bf16_scales)
                                    + " rows=" + std::to_string(rows));
                            const size_t bytes
                                    = static_cast<size_t>(rows) * cols * elem;
                            std::vector<unsigned char> want(bytes, 0);
                            const auto out_at
                                    = [&](std::vector<unsigned char> &v,
                                              int m) -> void * {
                                return v.data()
                                        + static_cast<size_t>(m) * cols * elem;
                            };
                            for (int m0 = 0; m0 < rows;
                                    m0 += ck::max_mr_for_nv_s4(NV)) {
                                const int mr = std::min(
                                        ck::max_mr_for_nv_s4(NV), rows - m0);
                                const auto kfn = ck::select_w4a8_ukernel(mr, NV,
                                        act, ck::S4SourceKind::kS8, out, gran);
                                ASSERT_NE(kfn, nullptr);
                                kfn(s8.data() + static_cast<size_t>(m0) * K, K,
                                        blk, ss(m0), ws, sk, bias.data(),
                                        ck::BiasKind::fp32,
                                        gated ? nullptr : out_at(want, m0),
                                        gated ? 0 : cols,
                                        gated ? out_at(want, m0) : nullptr,
                                        gated ? cols : 0, K, group_size, N,
                                        nullptr, nullptr);
                            }
                            for (const bool from_u8 : {false, true}) {
                                for (const int chunk : {1, G}) {
                                    std::vector<unsigned char> got(bytes, 0);
                                    alignas(64) float facc[kMaxRows * N] = {};
                                    for (int g0 = 0; g0 < G; g0 += chunk) {
                                        const int ng = std::min(chunk, G - g0);
                                        auto *dst = static_cast<int8_t *>(
                                                unpacked.p);
                                        ck::unpack_s4_block_nr32(
                                                blk + g0 * group_packed_bytes,
                                                ng * group_size, dst);
                                        gfn(rows,
                                                from_u8 ? u8.data() : s8.data(),
                                                K, dst, comp,
                                                ck::select_int8_accum_ukernel(
                                                        rows, from_u8),
                                                ss(0), ws, sk, g0, ng, G,
                                                group_size, N, facc, cs32);
                                    }
                                    sfn(rows, facc, ss(0), sk, bias.data(),
                                            ck::BiasKind::fp32,
                                            gated ? nullptr : got.data(),
                                            gated ? 0 : cols,
                                            gated ? got.data() : nullptr,
                                            gated ? cols : 0);
                                    EXPECT_EQ(got, want) << "u8=" << from_u8
                                                         << " chunk=" << chunk;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// Selector bounds — the s4 MR ceiling is tighter than the other
// families'; an over-budget specialisation would spill.
TEST(CkUkernelS4, SelectorRefusesOutOfRange) {
    for (const int NV : {2, 4}) {
        const int max_mr = ck::max_mr_for_nv_s4(NV);
        EXPECT_LE(max_mr, ck::kMaxMR);
        for (int mr = 1; mr <= max_mr; ++mr) {
            const auto legacy
                    = ck::select_w4a8_ukernel(mr, NV, ck::ActivationKind::none);
            EXPECT_NE(legacy, nullptr) << "NV=" << NV << " MR=" << mr;
            EXPECT_EQ(legacy,
                    ck::select_w4a8_ukernel(mr, NV, ck::ActivationKind::none,
                            ck::S4SourceKind::kS8, ck::S4OutputKind::kBf16,
                            ck::S4SrcScaleGranularity::kPerToken))
                    << "the default selector must preserve the existing "
                       "per-token specialization";
            EXPECT_NE(ck::select_w4a8_ukernel(mr, NV, ck::ActivationKind::none,
                              ck::S4SourceKind::kS8, ck::S4OutputKind::kBf16,
                              ck::S4SrcScaleGranularity::kPerGroup),
                    nullptr)
                    << "NV=" << NV << " MR=" << mr;
        }
        EXPECT_EQ(ck::select_w4a8_ukernel(
                          max_mr + 1, NV, ck::ActivationKind::none),
                nullptr)
                << "NV=" << NV << " must refuse MR>" << max_mr;
        EXPECT_EQ(ck::select_w4a8_ukernel(0, NV, ck::ActivationKind::none),
                nullptr);
    }
    // NV outside {2, 4} has no instantiation.
    EXPECT_EQ(ck::select_w4a8_ukernel(1, 3, ck::ActivationKind::none), nullptr);
    EXPECT_EQ(ck::select_w4a8_ukernel(1, 1, ck::ActivationKind::none), nullptr);
    EXPECT_NE(ck::select_w4a8_ukernel(1, 2, ck::ActivationKind::none,
                      ck::S4SourceKind::kU8Zp128, ck::S4OutputKind::kBf16,
                      ck::S4SrcScaleGranularity::kPerGroup),
            nullptr);
    EXPECT_NE(ck::select_w4a8_ukernel(1, 2, ck::ActivationKind::none,
                      ck::S4SourceKind::kU8Zp128, ck::S4OutputKind::kF32,
                      ck::S4SrcScaleGranularity::kPerGroup),
            nullptr);
}
