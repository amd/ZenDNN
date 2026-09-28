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
/// Drives `select_s4_ukernel` against an EXACT scalar reference, so a
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
#include <vector>

#include "common/bfloat16.hpp"
#include "common/zendnnl_compat.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/pack.hpp"
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
double ref_dot(int m, int col, int K, int group_size, bool bf16_scales) {
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
        acc += static_cast<double>(s32) * rnd(wscale_at(g, col));
    }
    return acc * rnd(sscale_at(m));
}

size_t oblock_bytes(int K, int pack_nr, int group_size) {
    return static_cast<size_t>(K / ck::kS4Octet) * pack_nr
            * ck::kS4BytesPerOctetCol
            + static_cast<size_t>(K / group_size) * pack_nr * sizeof(int32_t);
}

} // namespace

// act = none: every MR x NV x group_size, both scale dtypes, bias on
// and off, both caller layouts.
class CkUkernelS4None
    : public ::testing::TestWithParam<std::tuple<int, int, int, bool, bool>> {};

TEST_P(CkUkernelS4None, MatchesScalarReference) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    const auto [K, group_size, pack_nr, scale_bf16, with_bias] = GetParam();
    const int NV = pack_nr / 16;
    const int N = pack_nr * 2; // two o-blocks
    const int ldb = K; // transB = true
    const int max_mr = ck::max_mr_for_nv_s4(NV);

    const std::vector<int8_t> wsrc = build_packed(K, N, ldb, /*transB=*/true);
    Slab slab(ck::packed_weight_size_s4(K, N, pack_nr, group_size));
    ASSERT_NE(slab.p, nullptr);
    ASSERT_EQ(ck::prepack_weight_into_s4(wsrc.data(), K, N, ldb, pack_nr, true,
                      false, group_size, slab.p),
            status_t::success);

    const int G = K / group_size;
    for (int MR = 1; MR <= max_mr; ++MR) {
        auto fn = ck::select_s4_ukernel(MR, NV, ck::ActKind::none);
        ASSERT_NE(fn, nullptr) << "MR=" << MR << " NV=" << NV;

        std::vector<uint8_t> A(static_cast<size_t>(MR) * K);
        for (int m = 0; m < MR; ++m)
            for (int k = 0; k < K; ++k)
                A[static_cast<size_t>(m) * K + k]
                        = static_cast<uint8_t>(a_at(m, k));

        std::vector<float> ss_f(MR), ws_f(static_cast<size_t>(G) * N),
                bias_f(N);
        std::vector<bfloat16_t> ss_b(MR), ws_b(static_cast<size_t>(G) * N);
        for (int m = 0; m < MR; ++m) {
            ss_f[m] = sscale_at(m);
            ss_b[m] = bfloat16_t(ss_f[m]);
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
                    C.data() + ob * pack_nr, N, nullptr, 0, K, group_size, N);
        }

        for (int m = 0; m < MR; ++m)
            for (int n = 0; n < N; ++n) {
                double ref = ref_dot(m, n, K, group_size, scale_bf16);
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
                ::testing::Bool())); // bias

// Gated acts against a CANONICAL-order reference: validates the
// split-halves pack permute, the per-group scale permute the N-tile
// hoist applies, and the pair-store epilogue as one contract.
class CkUkernelS4Gated
    : public ::testing::TestWithParam<std::tuple<int, int, int, int>> {};

TEST_P(CkUkernelS4Gated, MatchesCanonicalReference) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    const auto [K, group_size, pack_nr, act_sel] = GetParam();
    const int NV = pack_nr / 16;
    const int N = pack_nr * 2;
    const int half = N / 2, ldb = K, G = K / group_size;
    const ck::ActKind act = (act_sel == 0) ? ck::ActKind::silu_and_mul
                                           : ck::ActKind::gelu_and_mul;

    const std::vector<int8_t> wsrc = build_packed(K, N, ldb, true);
    Slab slab(ck::packed_weight_size_s4(K, N, pack_nr, group_size));
    ASSERT_NE(slab.p, nullptr);
    // The pack permutes canonical [gate | up] into [g0, u0, g1, ...].
    ASSERT_EQ(ck::prepack_weight_into_s4(wsrc.data(), K, N, ldb, pack_nr, true,
                      /*interleave_split_halves=*/true, group_size, slab.p),
            status_t::success);

    const int max_mr = ck::max_mr_for_nv_s4(NV);
    for (int MR = 1; MR <= max_mr; ++MR) {
        auto fn = ck::select_s4_ukernel(MR, NV, act);
        ASSERT_NE(fn, nullptr);

        std::vector<uint8_t> A(static_cast<size_t>(MR) * K);
        for (int m = 0; m < MR; ++m)
            for (int k = 0; k < K; ++k)
                A[static_cast<size_t>(m) * K + k]
                        = static_cast<uint8_t>(a_at(m, k));
        std::vector<float> ss(MR);
        for (int m = 0; m < MR; ++m)
            ss[m] = sscale_at(m);

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
                    0, C.data() + ob * (pack_nr / 2), half, K, group_size, N);
        }

        for (int m = 0; m < MR; ++m)
            for (int i = 0; i < half; ++i) {
                const double gate
                        = ref_dot(m, i, K, group_size, /*bf16_scales=*/false);
                const double up = ref_dot(
                        m, half + i, K, group_size, /*bf16_scales=*/false);
                const double ref = (act == ck::ActKind::silu_and_mul)
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
                ::testing::Values(0, 1))); // silu / gelu

// Selector bounds — the s4 MR ceiling is tighter than the other
// families'; an over-budget specialisation would spill.
TEST(CkUkernelS4, SelectorRefusesOutOfRange) {
    for (const int NV : {2, 4}) {
        const int max_mr = ck::max_mr_for_nv_s4(NV);
        EXPECT_LE(max_mr, ck::kMaxMR);
        for (int mr = 1; mr <= max_mr; ++mr) {
            EXPECT_NE(ck::select_s4_ukernel(mr, NV, ck::ActKind::none), nullptr)
                    << "NV=" << NV << " MR=" << mr;
        }
        EXPECT_EQ(ck::select_s4_ukernel(max_mr + 1, NV, ck::ActKind::none),
                nullptr)
                << "NV=" << NV << " must refuse MR>" << max_mr;
        EXPECT_EQ(ck::select_s4_ukernel(0, NV, ck::ActKind::none), nullptr);
    }
    // NV outside {2, 4} has no instantiation.
    EXPECT_EQ(ck::select_s4_ukernel(1, 3, ck::ActKind::none), nullptr);
    EXPECT_EQ(ck::select_s4_ukernel(1, 1, ck::ActKind::none), nullptr);
}
