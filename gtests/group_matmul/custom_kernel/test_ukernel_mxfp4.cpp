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

/// CK ukernel module — MXFP4 sibling of `test_ukernel_s4.cpp`.
///
/// Drives `select_mxfp4_ukernel` against an EXACT scalar reference:
///
///   facc[m][v] = sum_g ( sum_{k in g} A_s8[m,k] * 2·W_e2m1[k,v] )
///                * 2^(e[g][v] - 128)
///   C[m][v]    = facc[m][v] * src_scale[m] + bias[v]
///
/// with per-block source scales moving `src_scale[m][g]` inside the block
/// sum.  Sweeps every MR x NV, group sizes 8 through K, both source-scale
/// dtypes and layouts, bias on and off, both outputs and the gated acts, at
/// 1% relative tolerance for a BF16 store (its floor is near 0.4%) and 1e-5
/// for an F32 one.  A bit-exact model of the kernel's arithmetic covers the
/// gpt-oss-20b shapes in every mode, the gated epilogues follow the
/// activation-form knobs bit for bit, and the selector bounds close the file.

#include <gtest/gtest.h>

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "common/bfloat16.hpp"
#include "common/data_types.hpp"
#include "common/zendnnl_compat.hpp"
#include "lowoha_operators/matmul/group_matmul/common/group_matmul_act_avx512.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/pack.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/ukernel/mxfp4_microkernel.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/ukernel/s8_group_sums.hpp"

namespace ck = ::zendnnl::lowoha::matmul::custom_kernel;
using zendnnl::common::bfloat16_t;
using ::zendnnl::common::kE2M1ToInt8x2;
using zendnnl::error_handling::status_t;

namespace {

constexpr auto kMx = ck::kW4FormatMxfp4;

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
// The f8_e8m0 scale byte, 2^-10..2^-2.
uint8_t e8m0_at(int g, int n) {
    return static_cast<uint8_t>(127 + ((g * 7 + n) % 9) - 10);
}
// The f32 scale the kernel applies per group: half the E8M0 value,
// 2^(e - 128), because it multiplies the doubled E2M1 code.
float wscale_at(int g, int n) {
    return std::ldexp(1.0f, e8m0_at(g, n) - 128);
}
std::vector<uint8_t> e8m0s(int G, int N) {
    std::vector<uint8_t> s(static_cast<size_t>(G) * N);
    for (int g = 0; g < G; ++g)
        for (int n = 0; n < N; ++n)
            s[static_cast<size_t>(g) * N + n] = e8m0_at(g, n);
    return s;
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

// Exact per-(m, col) dequantised dot product, canonical column order, with
// one source scale per row or per row and block.  `bf16_src_scale` must
// match what the caller handed the kernel, or the comparison measures the
// test's rounding rather than the kernel's.
double ref_dot(int m, int col, int K, int group_size, bool bf16_src_scale,
        bool per_block = false) {
    const auto rnd = [bf16_src_scale](float ss) {
        return bf16_src_scale
                ? static_cast<double>(static_cast<float>(bfloat16_t(ss)))
                : static_cast<double>(ss);
    };
    const int G = K / group_size;
    double acc = 0.0;
    for (int g = 0; g < G; ++g) {
        long s32 = 0;
        for (int k = g * group_size; k < (g + 1) * group_size; ++k)
            s32 += static_cast<long>(a_at(m, k))
                    * kE2M1ToInt8x2[nib_at(k, col)];
        acc += static_cast<double>(s32) * wscale_at(g, col)
                * (per_block ? rnd(sscale_at(m, g)) : 1.0);
    }
    return per_block ? acc : acc * rnd(sscale_at(m));
}

} // namespace

// act = none: every MR x NV x group_size, both source-scale dtypes and
// layouts, bias on and off, both outputs.
class CkUkernelMxfp4None
    : public ::testing::TestWithParam<
              std::tuple<int, int, int, bool, bool, bool, bool>> {};

TEST_P(CkUkernelMxfp4None, MatchesScalarReference) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    const auto [K, group_size, pack_nr, scale_bf16, with_bias, per_block,
            f32_out]
            = GetParam();
    const int NV = pack_nr / 16;
    const int N = pack_nr * 2; // two o-blocks
    const int ldb = K; // transB = true
    const int max_mr = ck::max_mr_for_nv_mxfp4(NV);

    const int G = K / group_size;
    const std::vector<int8_t> wsrc = build_packed(K, N, ldb, /*transB=*/true);
    const std::vector<uint8_t> e8 = e8m0s(G, N);
    Slab slab(ck::packed_weight_size_w4(K, N, pack_nr, group_size, kMx));
    ASSERT_NE(slab.p, nullptr);
    ASSERT_EQ(ck::prepack_weight_into_w4(wsrc.data(), K, N, ldb, pack_nr, true,
                      false, group_size, slab.p, kMx, e8.data(), N),
            status_t::success);

    for (int MR = 1; MR <= max_mr; ++MR) {
        auto fn = ck::select_mxfp4_ukernel(
                MR, NV, ck::ActivationKind::none, per_block);
        ASSERT_NE(fn, nullptr) << "MR=" << MR << " NV=" << NV;

        std::vector<uint8_t> A(static_cast<size_t>(MR) * K);
        for (int m = 0; m < MR; ++m)
            for (int k = 0; k < K; ++k)
                A[static_cast<size_t>(m) * K + k]
                        = static_cast<uint8_t>(a_at(m, k));

        const int src_scale_groups = per_block ? G : 1;
        const size_t n_ss = static_cast<size_t>(MR) * src_scale_groups;
        std::vector<float> ss_f(n_ss), bias_f(N);
        std::vector<bfloat16_t> ss_b(n_ss);
        for (int m = 0; m < MR; ++m)
            for (int g = 0; g < src_scale_groups; ++g) {
                const size_t i = static_cast<size_t>(m) * src_scale_groups + g;
                ss_f[i] = per_block ? sscale_at(m, g) : sscale_at(m);
                ss_b[i] = bfloat16_t(ss_f[i]);
            }
        for (int n = 0; n < N; ++n)
            bias_f[n] = bias_at(n);
        std::vector<int32_t> sums(static_cast<size_t>(MR) * G);
        ck::s8_group_sums(A.data(), K, MR, K, group_size, sums.data());

        const size_t n_out = static_cast<size_t>(MR) * N;
        std::vector<bfloat16_t> C(n_out, bfloat16_t(0.0f));
        std::vector<float> C_f32(n_out, 0.0f);
        for (int ob = 0; ob < N / pack_nr; ++ob) {
            const int8_t *bp = static_cast<const int8_t *>(slab.p)
                    + static_cast<size_t>(ob)
                            * ck::w4_oblock_bytes(K, pack_nr, group_size, kMx);
            void *out = f32_out
                    ? static_cast<void *>(C_f32.data() + ob * pack_nr)
                    : static_cast<void *>(C.data() + ob * pack_nr);
            fn(A.data(), K, bp,
                    scale_bf16 ? static_cast<const void *>(ss_b.data())
                               : static_cast<const void *>(ss_f.data()),
                    scale_bf16 ? ck::ScaleKind::kBf16 : ck::ScaleKind::kF32,
                    with_bias ? static_cast<const void *>(
                                        bias_f.data() + ob * pack_nr)
                              : nullptr,
                    with_bias ? ck::BiasKind::fp32 : ck::BiasKind::none, out,
                    f32_out ? ck::OutputType::kF32 : ck::OutputType::kBf16, N,
                    nullptr, 0, K, group_size, sums.data());
        }

        for (int m = 0; m < MR; ++m)
            for (int n = 0; n < N; ++n) {
                double ref
                        = ref_dot(m, n, K, group_size, scale_bf16, per_block);
                if (with_bias) ref += bias_at(n);
                const size_t i = static_cast<size_t>(m) * N + n;
                const double got
                        = f32_out ? C_f32[i] : static_cast<float>(C[i]);
                const double rel
                        = std::fabs(got - ref) / std::max(1.0, std::fabs(ref));
                ASSERT_LT(rel, f32_out ? 1e-5 : 0.01)
                        << "MR=" << MR << " m=" << m << " n=" << n
                        << " ref=" << ref << " got=" << got;
            }
    }
}

INSTANTIATE_TEST_SUITE_P(Shapes, CkUkernelMxfp4None,
        ::testing::Combine(::testing::Values(128, 256), // K
                ::testing::Values(8, 32, 128), // group_size
                ::testing::Values(32, 64), // pack_nr
                ::testing::Bool(), // bf16 source scale
                ::testing::Bool(), // bias
                ::testing::Bool(), // per-token / per-block source scale
                ::testing::Bool())); // bf16 / f32 output

namespace {

// One rounding per operation, as the kernel's final pass does; written as
// one expression, `a * b + c` may be contracted into an FMA by the compiler.
float rounded_mul(float a, float b) {
    volatile float r = a * b;
    return r;
}
float rounded_add(float a, float b) {
    volatile float r = a + b;
    return r;
}

// The library's swiglu_oai_mul store on 16 model values, so the gated
// comparison checks the kernel's arithmetic up to the activation rather
// than a re-derived activation.
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,fma")
void swiglu_oai_store16(const float *gate, const float *up, bfloat16_t *out) {
    namespace act = ::zendnnl::lowoha::matmul::group_matmul_act_avx512;
    const __m256i r = act::f32_to_bf16x16(
            act::swiglu_oai_avx512(_mm512_loadu_ps(gate), _mm512_loadu_ps(up)));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out), r);
}

// The pack halves an E8M0 byte (`e - 1`, flushing e <= 1) and lowers an
// o-block whose largest halved exponent exceeds `e8m0_in_pack_max_exp`.
int halved_exp(uint8_t e) {
    return e <= 1 ? 0 : e - 1;
}

uint16_t bf16_bits(bfloat16_t v) {
    uint16_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    return bits;
}
uint32_t f32_bits(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    return bits;
}

} // namespace

// The gpt-oss-20b MoE shapes, bit-exact against the kernel's own
// arithmetic: per-group int32 dots folded by one FMA each in group order, the
// source scale (per token after the fold, or per block inside it), the shift
// back for an o-block the pack lowered, then the bias, rounded on its own,
// and the bf16 or f32 store or the swiglu_oai_mul epilogue.
// Doubled E2M1 is integral, so a tolerance test would pass on a wrong LUT
// entry; this does not.  For one row per column the f32 bias cancels that
// row's scaled sum exactly, so a bias add contracted into an FMA, at any row
// or column of the tile, leaves the FMA's rounding error where the kernel
// must store zero.
TEST(CkUkernelMxfp4, BitExactVsKernelModel) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    constexpr int group_size = 32;
    struct Case {
        int M, K, N, NV;
        ck::ActivationKind act;
        bool lowered_oblock; // o-block 0's scales overflow f32 unlowered
    };
    const auto kNone = ck::ActivationKind::none;
    const auto kSwiglu = ck::ActivationKind::swiglu_oai_mul;
    // w13 then w2, at decode and at small prompt tiles.
    const Case cases[] = {{1, 2880, 5760, 2, kNone, false},
            {4, 2880, 2880, 4, kNone, false}, {8, 2880, 2880, 2, kNone, false},
            {2, 2880, 5760, 4, kNone, false}, {6, 2880, 640, 2, kSwiglu, false},
            {3, 2880, 640, 4, kSwiglu, false}, {6, 256, 128, 2, kNone, true},
            {3, 256, 256, 4, kSwiglu, true}};
    for (const Case &c : cases) {
        const bool gated = c.act != kNone;
        const int pack_nr = c.NV * 16, G = c.K / group_size, half = c.N / 2;
        const std::string label = "M=" + std::to_string(c.M)
                + " K=" + std::to_string(c.K) + " N=" + std::to_string(c.N)
                + " NV=" + std::to_string(c.NV) + (gated ? " swiglu" : "")
                + (c.lowered_oblock ? " lowered" : "");
        ASSERT_EQ(c.N % pack_nr, 0) << label;
        // A gated pack interleaves canonical [gate | up] as g0, u0, g1, ...
        const auto packed_col = [&](int col) {
            return !gated ? col : col < half ? 2 * col : 2 * (col - half) + 1;
        };
        const auto canon_col = [&](int pc) {
            return !gated ? pc : (pc & 1) ? half + (pc >> 1) : (pc >> 1);
        };

        // E8M0 bytes near 1, or near the top of the range in o-block 0 of a
        // lowered case.
        std::vector<uint8_t> e8(static_cast<size_t>(G) * c.N);
        std::vector<int> ob_max(c.N / pack_nr, 0);
        for (int g = 0; g < G; ++g)
            for (int n = 0; n < c.N; ++n) {
                const bool high = c.lowered_oblock && packed_col(n) < pack_nr;
                const uint8_t e = high
                        ? static_cast<uint8_t>(245 + (g * 7 + n) % 5)
                        : static_cast<uint8_t>(124 + (g * 7 + n) % 9);
                e8[static_cast<size_t>(g) * c.N + n] = e;
                int &mx = ob_max[packed_col(n) / pack_nr];
                mx = std::max(mx, halved_exp(e));
            }
        std::vector<int> shift(c.N);
        for (int n = 0; n < c.N; ++n) {
            const int mx = ob_max[packed_col(n) / pack_nr];
            const int cap = ck::e8m0_in_pack_max_exp(c.K / group_size);
            shift[n] = mx > cap ? mx - cap : 0;
        }
        if (c.lowered_oblock) { ASSERT_GT(shift[canon_col(0)], 0) << label; }

        const std::vector<int8_t> wsrc
                = build_packed(c.K, c.N, /*ldb=*/c.K, /*transB=*/true);
        Slab slab(
                ck::packed_weight_size_w4(c.K, c.N, pack_nr, group_size, kMx));
        ASSERT_NE(slab.p, nullptr);
        ASSERT_EQ(
                ck::prepack_weight_into_w4(wsrc.data(), c.K, c.N, c.K, pack_nr,
                        true, gated, group_size, slab.p, kMx, e8.data(), c.N),
                status_t::success)
                << label;

        std::vector<uint8_t> A(static_cast<size_t>(c.M) * c.K);
        for (int m = 0; m < c.M; ++m)
            for (int k = 0; k < c.K; ++k)
                A[static_cast<size_t>(m) * c.K + k]
                        = static_cast<uint8_t>(a_at(m, k));
        std::vector<float> ss(c.M);
        for (int m = 0; m < c.M; ++m)
            ss[m] = std::ldexp(sscale_at(m), c.lowered_oblock ? -110 : 0);
        std::vector<int32_t> sums(static_cast<size_t>(c.M) * G);
        ck::s8_group_sums(A.data(), c.K, c.M, c.K, group_size, sums.data());

        // Per-block source scales, one per row and block.
        std::vector<float> ss_pb(static_cast<size_t>(c.M) * G);
        for (int m = 0; m < c.M; ++m)
            for (int g = 0; g < G; ++g)
                ss_pb[static_cast<size_t>(m) * G + g] = std::ldexp(
                        sscale_at(m, g), c.lowered_oblock ? -110 : 0);

        for (const bool per_block : {false, true})
            for (const bool f32_out : {false, true}) {
                if (gated && f32_out) continue;
                const std::string mlabel = label
                        + (per_block ? ", per-block scales" : "")
                        + (f32_out ? ", f32 out" : "");
                // The scaled sum before the bias, per row and canonical
                // column.  Per block, each block's factor is the rounded
                // product of its source and weight scales, zero for a zero
                // dot; the shift back follows the last block.
                std::vector<float> scaled(static_cast<size_t>(c.M) * c.N);
                for (int m = 0; m < c.M; ++m)
                    for (int n = 0; n < c.N; ++n) {
                        float acc = 0.0f;
                        for (int g = 0; g < G; ++g) {
                            int32_t s32 = 0;
                            for (int k = g * group_size;
                                    k < (g + 1) * group_size; ++k)
                                s32 += static_cast<int32_t>(a_at(m, k))
                                        * kE2M1ToInt8x2[nib_at(k, n)];
                            const int h = halved_exp(
                                    e8[static_cast<size_t>(g) * c.N + n]);
                            const int stored = h > shift[n] ? h - shift[n] : 0;
                            const float ws = stored == 0
                                    ? 0.0f
                                    : std::ldexp(1.0f, stored - 127);
                            const float factor = !per_block ? ws
                                    : s32 == 0
                                    ? 0.0f
                                    : rounded_mul(
                                              ss_pb[static_cast<size_t>(m) * G
                                                      + g],
                                              ws);
                            acc = std::fma(
                                    static_cast<float>(s32), factor, acc);
                        }
                        scaled[static_cast<size_t>(m) * c.N + n] = per_block
                                ? std::ldexp(acc, shift[n])
                                : std::ldexp(rounded_mul(acc, ss[m]), shift[n]);
                    }

                // f32 bias: column n cancels row n % M exactly (the gate
                // columns only when gated, where a cancelled up would not
                // reach the store).  bf16 bias: ordinary values.
                std::vector<float> bias_canon(c.N);
                for (int n = 0; n < c.N; ++n) {
                    const bool cancel = !gated || n < half;
                    bias_canon[n] = cancel
                            ? -scaled[static_cast<size_t>(n % c.M) * c.N + n]
                            : bias_at(n);
                }
                std::vector<float> bias_f32(c.N);
                std::vector<bfloat16_t> bias_bf16(c.N);
                for (int pc = 0; pc < c.N; ++pc) {
                    bias_f32[pc] = bias_canon[canon_col(pc)];
                    bias_bf16[pc] = bfloat16_t(bias_at(canon_col(pc)));
                }

                const int out_cols = gated ? half : c.N;
                const int max_mr = ck::max_mr_for_nv_mxfp4(c.NV);
                for (const ck::BiasKind bk : {ck::BiasKind::none,
                             ck::BiasKind::fp32, ck::BiasKind::bf16}) {
                    const std::string blabel = mlabel
                            + (bk == ck::BiasKind::none ? ", no bias"
                                            : bk == ck::BiasKind::fp32
                                            ? ", f32 bias"
                                            : ", bf16 bias");
                    const auto pre_act = [&](int m, int n) {
                        const float v
                                = scaled[static_cast<size_t>(m) * c.N + n];
                        if (bk == ck::BiasKind::fp32)
                            return rounded_add(v, bias_canon[n]);
                        if (bk == ck::BiasKind::bf16)
                            return rounded_add(v,
                                    static_cast<float>(bfloat16_t(bias_at(n))));
                        return v;
                    };
                    const size_t n_out = static_cast<size_t>(c.M) * out_cols;
                    std::vector<bfloat16_t> want(n_out, bfloat16_t(0.0f));
                    std::vector<float> want_f32(n_out, 0.0f);
                    for (int m = 0; m < c.M; ++m) {
                        const size_t r = static_cast<size_t>(m) * out_cols;
                        if (!gated) {
                            for (int n = 0; n < c.N; ++n) {
                                want_f32[r + n] = pre_act(m, n);
                                want[r + n] = bfloat16_t(want_f32[r + n]);
                            }
                            continue;
                        }
                        for (int i0 = 0; i0 < half; i0 += 16) {
                            float gate[16], up[16];
                            for (int j = 0; j < 16; ++j) {
                                gate[j] = pre_act(m, i0 + j);
                                up[j] = pre_act(m, half + i0 + j);
                            }
                            swiglu_oai_store16(gate, up, want.data() + r + i0);
                        }
                    }

                    const void *bias = bk == ck::BiasKind::fp32
                            ? static_cast<const void *>(bias_f32.data())
                            : bk == ck::BiasKind::bf16
                            ? static_cast<const void *>(bias_bf16.data())
                            : nullptr;
                    const size_t bias_elem = bk == ck::BiasKind::fp32
                            ? sizeof(float)
                            : sizeof(bfloat16_t);
                    const size_t out_elem
                            = f32_out ? sizeof(float) : sizeof(bfloat16_t);
                    std::vector<bfloat16_t> C(n_out, bfloat16_t(0.0f));
                    std::vector<float> C_f32(n_out, 0.0f);
                    char *const C_base = f32_out
                            ? reinterpret_cast<char *>(C_f32.data())
                            : reinterpret_cast<char *>(C.data());
                    for (int ob = 0; ob < c.N / pack_nr; ++ob) {
                        const int8_t *bp = static_cast<const int8_t *>(slab.p)
                                + static_cast<size_t>(ob)
                                        * ck::w4_oblock_bytes(
                                                c.K, pack_nr, group_size, kMx);
                        const int n0 = ob * pack_nr;
                        const void *ob_bias = bias == nullptr
                                ? nullptr
                                : static_cast<const char *>(bias)
                                        + n0 * bias_elem;
                        for (int m0 = 0; m0 < c.M; m0 += max_mr) {
                            const int mr = std::min(max_mr, c.M - m0);
                            auto fn = ck::select_mxfp4_ukernel(
                                    mr, c.NV, c.act, per_block);
                            ASSERT_NE(fn, nullptr) << blabel << " MR=" << mr;
                            char *out = C_base
                                    + (static_cast<size_t>(m0) * out_cols
                                              + (gated ? n0 / 2 : n0))
                                            * out_elem;
                            const float *src_scale = per_block
                                    ? ss_pb.data() + static_cast<size_t>(m0) * G
                                    : ss.data() + m0;
                            fn(A.data() + static_cast<size_t>(m0) * c.K, c.K,
                                    bp, src_scale, ck::ScaleKind::kF32, ob_bias,
                                    bk, gated ? nullptr : out,
                                    f32_out ? ck::OutputType::kF32
                                            : ck::OutputType::kBf16,
                                    gated ? 0 : c.N, gated ? out : nullptr,
                                    gated ? half : 0, c.K, group_size,
                                    sums.data() + static_cast<size_t>(m0) * G);
                        }
                    }

                    for (int m = 0; m < c.M; ++m)
                        for (int n = 0; n < out_cols; ++n) {
                            const size_t i
                                    = static_cast<size_t>(m) * out_cols + n;
                            if (f32_out) {
                                ASSERT_EQ(f32_bits(C_f32[i]),
                                        f32_bits(want_f32[i]))
                                        << blabel << " m=" << m << " n=" << n
                                        << ": got " << C_f32[i] << ", want "
                                        << want_f32[i];
                            } else {
                                ASSERT_EQ(bf16_bits(C[i]), bf16_bits(want[i]))
                                        << blabel << " m=" << m << " n=" << n
                                        << ": got " << static_cast<float>(C[i])
                                        << ", want "
                                        << static_cast<float>(want[i]);
                            }
                        }
                }
            }
    }
}

// An E8M0 scale near the top of its range overflows the f32 sums before the
// source scale brings the result back, unless the pack lowers the o-block's
// exponents and the kernel restores them after the source scale:
// 32 * (1 * 6.0) * 2^123 * 2^-120 = 1536 exactly, in every column.
TEST(CkUkernelMxfp4, ExtremeScalesStayFinite) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    constexpr int K = 32, N = 64, pack_nr = 32, group_size = 32, NV = 2;
    // E2M1 code 7 is 6.0; the second o-block keeps an ordinary scale.
    const std::vector<int8_t> w(static_cast<size_t>(K) * N / 2, 0x77);
    std::vector<uint8_t> e8(N, 250);
    std::fill(e8.begin() + pack_nr, e8.end(), 117); // 2^-10
    Slab slab(ck::packed_weight_size_w4(K, N, pack_nr, group_size, kMx));
    ASSERT_NE(slab.p, nullptr);
    ASSERT_EQ(ck::prepack_weight_into_w4(w.data(), K, N, /*ldb=*/K, pack_nr,
                      true, false, group_size, slab.p, kMx, e8.data(), N),
            status_t::success);
    const std::vector<uint8_t> A(K, 1);
    const float src_scale = std::ldexp(1.0f, -120);
    std::vector<int32_t> sums(1);
    ck::s8_group_sums(A.data(), K, 1, K, group_size, sums.data());
    auto fn = ck::select_mxfp4_ukernel(1, NV, ck::ActivationKind::none);
    ASSERT_NE(fn, nullptr);
    std::vector<bfloat16_t> C(N, bfloat16_t(0.0f));
    for (int ob = 0; ob < N / pack_nr; ++ob) {
        const int8_t *bp = static_cast<const int8_t *>(slab.p)
                + static_cast<size_t>(ob)
                        * ck::w4_oblock_bytes(K, pack_nr, group_size, kMx);
        fn(A.data(), K, bp, &src_scale, ck::ScaleKind::kF32, nullptr,
                ck::BiasKind::none, C.data() + ob * pack_nr,
                ck::OutputType::kBf16, N, nullptr, 0, K, group_size,
                sums.data());
    }
    const float ordinary = 32.0f * 6.0f * std::ldexp(1.0f, -130);
    for (int n = 0; n < N; ++n)
        EXPECT_EQ(static_cast<float>(C[n]), n < pack_nr ? 1536.0f : ordinary)
                << "n=" << n;
}

// Per-block source scales on a lowered o-block: block 0 gives
// 32 * 6.0 * 2^123 * 2^-120 = 1536 in every column, exactly.  Block 1's
// source is zero under a source scale whose product with the weight scale
// overflows f32; its zero dot must add zero, not NaN.
TEST(CkUkernelMxfp4, PerBlockZeroDotIgnoresOverflowingScale) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    constexpr int K = 64, N = 32, pack_nr = 32, group_size = 32, NV = 2;
    constexpr int G = K / group_size;
    const std::vector<int8_t> w(static_cast<size_t>(K) * N / 2, 0x77);
    const std::vector<uint8_t> e8(static_cast<size_t>(G) * N, 250);
    Slab slab(ck::packed_weight_size_w4(K, N, pack_nr, group_size, kMx));
    ASSERT_NE(slab.p, nullptr);
    ASSERT_EQ(ck::prepack_weight_into_w4(w.data(), K, N, /*ldb=*/K, pack_nr,
                      true, false, group_size, slab.p, kMx, e8.data(), N),
            status_t::success);
    std::vector<uint8_t> A(K, 0);
    std::fill(A.begin(), A.begin() + group_size, 1);
    const float src_scale[G] = {std::ldexp(1.0f, -120), std::ldexp(1.0f, 120)};
    std::vector<int32_t> sums(G);
    ck::s8_group_sums(A.data(), K, 1, K, group_size, sums.data());
    auto fn = ck::select_mxfp4_ukernel(
            1, NV, ck::ActivationKind::none, /*per_block_src_scale=*/true);
    ASSERT_NE(fn, nullptr);
    for (const ck::OutputType out :
            {ck::OutputType::kBf16, ck::OutputType::kF32}) {
        std::vector<bfloat16_t> C(N, bfloat16_t(0.0f));
        std::vector<float> C_f32(N, 0.0f);
        fn(A.data(), K, static_cast<const int8_t *>(slab.p), src_scale,
                ck::ScaleKind::kF32, nullptr, ck::BiasKind::none,
                out == ck::OutputType::kF32 ? static_cast<void *>(C_f32.data())
                                            : static_cast<void *>(C.data()),
                out, N, nullptr, 0, K, group_size, sums.data());
        for (int n = 0; n < N; ++n)
            EXPECT_EQ(out == ck::OutputType::kF32 ? C_f32[n]
                                                  : static_cast<float>(C[n]),
                    1536.0f)
                    << "n=" << n;
    }
}

// With 2^13 groups of 32 * 127 * 6.0 under the largest E8M0 scale, the f32
// sums pass 2^128 unless the pack lowers the o-block by K as well:
// 2^18 * 127 * 6.0 * 2^127 * 2^-120 = 25568477184 exactly, in every column.
TEST(CkUkernelMxfp4, LargeKStaysFinite) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    constexpr int K = 1 << 18, N = 32, pack_nr = 32, group_size = 32, NV = 2;
    constexpr int G = K / group_size;
    const std::vector<int8_t> w(static_cast<size_t>(K) * N / 2, 0x77);
    const std::vector<uint8_t> e8(static_cast<size_t>(G) * N, 254);
    Slab slab(ck::packed_weight_size_w4(K, N, pack_nr, group_size, kMx));
    ASSERT_NE(slab.p, nullptr);
    ASSERT_EQ(ck::prepack_weight_into_w4(w.data(), K, N, /*ldb=*/K, pack_nr,
                      true, false, group_size, slab.p, kMx, e8.data(), N),
            status_t::success);
    const std::vector<uint8_t> A(K, 127);
    const float src_scale = std::ldexp(1.0f, -120);
    std::vector<int32_t> sums(G);
    ck::s8_group_sums(A.data(), K, 1, K, group_size, sums.data());
    auto fn = ck::select_mxfp4_ukernel(1, NV, ck::ActivationKind::none);
    ASSERT_NE(fn, nullptr);
    std::vector<float> C(N, 0.0f);
    fn(A.data(), K, static_cast<const int8_t *>(slab.p), &src_scale,
            ck::ScaleKind::kF32, nullptr, ck::BiasKind::none, C.data(),
            ck::OutputType::kF32, N, nullptr, 0, K, group_size, sums.data());
    for (int n = 0; n < N; ++n)
        EXPECT_EQ(C[n], 25568477184.0f) << "n=" << n;
}

// Column 0's E8M0 byte 254 lowers the o-block by 26.  Column 1, under byte
// 129, dots to 2 and takes a source scale of 2^-126, so it is
// 2 * 2^(129 - 128) * 2^-126 = 2^-124 exactly: the restore must not pass
// through a product below f32's range, per token or per block, in either
// output.
TEST(CkUkernelMxfp4, LoweredOblockKeepsSmallResults) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    constexpr int K = 64, N = 32, pack_nr = 32, group_size = 32, NV = 2;
    constexpr int G = K / group_size;
    std::vector<int8_t> w(static_cast<size_t>(K) * N / 2, 0);
    w[(1 * K + 0) / 2] = 0x02; // column 1, k = 0: E2M1 code 2 (1.0)
    std::vector<uint8_t> e8(static_cast<size_t>(G) * N, 127);
    for (int g = 0; g < G; ++g) {
        e8[static_cast<size_t>(g) * N + 0] = 254;
        e8[static_cast<size_t>(g) * N + 1] = 129;
    }
    Slab slab(ck::packed_weight_size_w4(K, N, pack_nr, group_size, kMx));
    ASSERT_NE(slab.p, nullptr);
    ASSERT_EQ(ck::prepack_weight_into_w4(w.data(), K, N, /*ldb=*/K, pack_nr,
                      true, false, group_size, slab.p, kMx, e8.data(), N),
            status_t::success);
    std::vector<uint8_t> A(K, 0);
    A[0] = 1;
    std::vector<int32_t> sums(G);
    ck::s8_group_sums(A.data(), K, 1, K, group_size, sums.data());
    const float tiny = std::ldexp(1.0f, -126);
    const float src_scale[G] = {tiny, tiny};
    for (const bool per_block : {false, true}) {
        auto fn = ck::select_mxfp4_ukernel(
                1, NV, ck::ActivationKind::none, per_block);
        ASSERT_NE(fn, nullptr);
        for (const ck::OutputType out :
                {ck::OutputType::kBf16, ck::OutputType::kF32}) {
            std::vector<bfloat16_t> C(N, bfloat16_t(0.0f));
            std::vector<float> C_f32(N, 0.0f);
            fn(A.data(), K, static_cast<const int8_t *>(slab.p), src_scale,
                    ck::ScaleKind::kF32, nullptr, ck::BiasKind::none,
                    out == ck::OutputType::kF32
                            ? static_cast<void *>(C_f32.data())
                            : static_cast<void *>(C.data()),
                    out, N, nullptr, 0, K, group_size, sums.data());
            for (int n = 0; n < N; ++n) {
                const float got = out == ck::OutputType::kF32
                        ? C_f32[n]
                        : static_cast<float>(C[n]);
                EXPECT_EQ(got, n == 1 ? std::ldexp(1.0f, -124) : 0.0f)
                        << (per_block ? "per block" : "per token")
                        << (out == ck::OutputType::kF32 ? ", f32" : ", bf16")
                        << " n=" << n;
            }
        }
    }
}

// Gated acts against a CANONICAL-order reference: validates the
// split-halves pack permute, which carries the in-pack scales with their
// columns, and the pair-store epilogue as one contract.
class CkUkernelMxfp4Gated
    : public ::testing::TestWithParam<std::tuple<int, int, int, int>> {};

TEST_P(CkUkernelMxfp4Gated, MatchesCanonicalReference) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    const auto [K, group_size, pack_nr, act_sel] = GetParam();
    const int NV = pack_nr / 16;
    const int N = pack_nr * 2;
    const int half = N / 2, ldb = K, G = K / group_size;
    const ck::ActivationKind act = (act_sel == 0)
            ? ck::ActivationKind::silu_and_mul
            : ck::ActivationKind::gelu_and_mul;

    const std::vector<int8_t> wsrc = build_packed(K, N, ldb, true);
    const std::vector<uint8_t> e8 = e8m0s(G, N);
    Slab slab(ck::packed_weight_size_w4(K, N, pack_nr, group_size, kMx));
    ASSERT_NE(slab.p, nullptr);
    // The pack permutes canonical [gate | up] into [g0, u0, g1, ...].
    ASSERT_EQ(ck::prepack_weight_into_w4(wsrc.data(), K, N, ldb, pack_nr, true,
                      /*interleave_split_halves=*/true, group_size, slab.p, kMx,
                      e8.data(), N),
            status_t::success);

    const int max_mr = ck::max_mr_for_nv_mxfp4(NV);
    for (int MR = 1; MR <= max_mr; ++MR) {
        auto fn = ck::select_mxfp4_ukernel(MR, NV, act);
        ASSERT_NE(fn, nullptr);

        std::vector<uint8_t> A(static_cast<size_t>(MR) * K);
        for (int m = 0; m < MR; ++m)
            for (int k = 0; k < K; ++k)
                A[static_cast<size_t>(m) * K + k]
                        = static_cast<uint8_t>(a_at(m, k));
        std::vector<float> ss(MR);
        for (int m = 0; m < MR; ++m)
            ss[m] = sscale_at(m);
        std::vector<int32_t> sums(static_cast<size_t>(MR) * G);
        ck::s8_group_sums(A.data(), K, MR, K, group_size, sums.data());

        std::vector<bfloat16_t> C(
                static_cast<size_t>(MR) * half, bfloat16_t(0.0f));
        for (int ob = 0; ob < N / pack_nr; ++ob) {
            const int8_t *bp = static_cast<const int8_t *>(slab.p)
                    + static_cast<size_t>(ob)
                            * ck::w4_oblock_bytes(K, pack_nr, group_size, kMx);
            fn(A.data(), K, bp, ss.data(), ck::ScaleKind::kF32, nullptr,
                    ck::BiasKind::none, nullptr, ck::OutputType::kBf16, 0,
                    C.data() + ob * (pack_nr / 2), half, K, group_size,
                    sums.data());
        }

        for (int m = 0; m < MR; ++m)
            for (int i = 0; i < half; ++i) {
                const double gate = ref_dot(
                        m, i, K, group_size, /*bf16_src_scale=*/false);
                const double up = ref_dot(
                        m, half + i, K, group_size, /*bf16_src_scale=*/false);
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

INSTANTIATE_TEST_SUITE_P(Shapes, CkUkernelMxfp4Gated,
        ::testing::Combine(::testing::Values(128, 256), // K
                ::testing::Values(32, 128), // group_size
                ::testing::Values(32, 64), // pack_nr
                ::testing::Values(0, 1))); // silu / gelu

namespace {

namespace act = ::zendnnl::lowoha::matmul::group_matmul_act_avx512;
namespace mm_test = ::zendnnl::lowoha::matmul::test_api;

/// Pins both activation-form knobs for a scope.
struct ActFormOverride {
    int prev_tanh, prev_fast;
    ActFormOverride(int gelu_tanh, int fast)
        : prev_tanh(
                  mm_test::s_grp_matmul_gelu_tanh_override.exchange(gelu_tanh))
        , prev_fast(mm_test::s_grp_matmul_fast_act_override.exchange(fast)) {}
    ~ActFormOverride() {
        mm_test::s_grp_matmul_gelu_tanh_override.store(prev_tanh);
        mm_test::s_grp_matmul_fast_act_override.store(prev_fast);
    }
    ActFormOverride(const ActFormOverride &) = delete;
    ActFormOverride &operator=(const ActFormOverride &) = delete;
};

// The library's gated store of `kind` on 16 gate/up pairs, in `mode`.
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,avx512dq,fma")
void gated_store16(ck::ActivationKind kind, const float *gate, const float *up,
        act::act_mode_t mode, bfloat16_t *out) {
    const __m512 g = _mm512_loadu_ps(gate);
    const __m512 u = _mm512_loadu_ps(up);
    const __m512 y = kind == ck::ActivationKind::swiglu_oai_mul
            ? act::swiglu_oai_act_avx512(g, u, mode)
            : kind == ck::ActivationKind::silu_and_mul
            ? _mm512_mul_ps(act::silu_act_avx512(g, mode), u)
            : _mm512_mul_ps(act::gelu_act_avx512(g, mode), u);
    _mm256_storeu_si256(
            reinterpret_cast<__m256i *>(out), act::f32_to_bf16x16(y));
}

} // namespace

// Every gated epilogue evaluates its activation in the form
// ZENDNNL_GRP_MATMUL_GELU_TANH and ZENDNNL_GRP_MATMUL_FAST_ACT select: bit for
// bit the library's own form on the kernel's f32 pre-activations, which an
// act=none f32 store of the same split-halves pack holds as [g0, u0, g1, ...].
// A fast form moves about 0.6% of bf16 outputs, hence 6,144 of them.
TEST(CkUkernelMxfp4, GatedEpilogueFollowsTheActivationKnobs) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    constexpr int K = 256, N = 256, pack_nr = 32, NV = 2, group_size = 32;
    constexpr int half = N / 2, G = K / group_size;
    const std::vector<int8_t> wsrc = build_packed(K, N, K, true);
    const std::vector<uint8_t> e8 = e8m0s(G, N);
    Slab slab(ck::packed_weight_size_w4(K, N, pack_nr, group_size, kMx));
    ASSERT_NE(slab.p, nullptr);
    ASSERT_EQ(ck::prepack_weight_into_w4(wsrc.data(), K, N, K, pack_nr, true,
                      /*interleave_split_halves=*/true, group_size, slab.p, kMx,
                      e8.data(), N),
            status_t::success);
    const auto oblock = [&](int ob) {
        return static_cast<const int8_t *>(slab.p)
                + static_cast<size_t>(ob)
                * ck::w4_oblock_bytes(K, pack_nr, group_size, kMx);
    };
    const int MR = ck::max_mr_for_nv_mxfp4(NV);
    const int M = 4 * MR;
    std::vector<uint8_t> A(static_cast<size_t>(M) * K);
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k)
            A[static_cast<size_t>(m) * K + k]
                    = static_cast<uint8_t>(a_at(m, k));
    std::vector<float> ss(M);
    for (int m = 0; m < M; ++m)
        ss[m] = sscale_at(m);
    std::vector<int32_t> sums(static_cast<size_t>(M) * G);
    ck::s8_group_sums(A.data(), K, M, K, group_size, sums.data());

    std::vector<float> pre(static_cast<size_t>(M) * N);
    const auto none
            = ck::select_mxfp4_ukernel(MR, NV, ck::ActivationKind::none);
    ASSERT_NE(none, nullptr);
    for (int m0 = 0; m0 < M; m0 += MR)
        for (int ob = 0; ob < N / pack_nr; ++ob)
            none(A.data() + static_cast<size_t>(m0) * K, K, oblock(ob),
                    ss.data() + m0, ck::ScaleKind::kF32, nullptr,
                    ck::BiasKind::none,
                    pre.data() + static_cast<size_t>(m0) * N + ob * pack_nr,
                    ck::OutputType::kF32, N, nullptr, 0, K, group_size,
                    sums.data() + static_cast<size_t>(m0) * G);

    struct Knobs {
        int tanh, fast;
    };
    for (const ck::ActivationKind kind : {ck::ActivationKind::swiglu_oai_mul,
                 ck::ActivationKind::silu_and_mul,
                 ck::ActivationKind::gelu_and_mul}) {
        const auto fn = ck::select_mxfp4_ukernel(MR, NV, kind);
        ASSERT_NE(fn, nullptr);
        std::vector<uint16_t> by_default;
        for (const Knobs k : {Knobs {0, 0}, Knobs {1, 0}, Knobs {0, 1}}) {
            const std::string label
                    = "act=" + std::to_string(static_cast<int>(kind))
                    + " gelu_tanh=" + std::to_string(k.tanh)
                    + " fast=" + std::to_string(k.fast);
            ActFormOverride o(k.tanh, k.fast);
            const act::act_mode_t mode = act::act_mode();
            std::vector<bfloat16_t> C(
                    static_cast<size_t>(M) * half, bfloat16_t(0.0f));
            for (int m0 = 0; m0 < M; m0 += MR)
                for (int ob = 0; ob < N / pack_nr; ++ob)
                    fn(A.data() + static_cast<size_t>(m0) * K, K, oblock(ob),
                            ss.data() + m0, ck::ScaleKind::kF32, nullptr,
                            ck::BiasKind::none, nullptr, ck::OutputType::kBf16,
                            0,
                            C.data() + static_cast<size_t>(m0) * half
                                    + ob * (pack_nr / 2),
                            half, K, group_size,
                            sums.data() + static_cast<size_t>(m0) * G);
            std::vector<uint16_t> got(C.size());
            for (size_t i = 0; i < C.size(); ++i)
                got[i] = bf16_bits(C[i]);
            for (int m = 0; m < M; ++m)
                for (int ob = 0; ob < N / pack_nr; ++ob) {
                    const float *p = pre.data() + static_cast<size_t>(m) * N
                            + ob * pack_nr;
                    float gate[16], up[16];
                    for (int j = 0; j < 16; ++j) {
                        gate[j] = p[2 * j];
                        up[j] = p[2 * j + 1];
                    }
                    bfloat16_t want[16];
                    gated_store16(kind, gate, up, mode, want);
                    for (int j = 0; j < 16; ++j)
                        ASSERT_EQ(got[static_cast<size_t>(m) * half + ob * 16
                                          + j],
                                bf16_bits(want[j]))
                                << label << " m=" << m << " i=" << ob * 16 + j;
                }
            if (k.tanh == 0 && k.fast == 0) {
                by_default = got;
                continue;
            }
            // GELU_TANH changes GELU alone; FAST_ACT changes every form.
            const bool changes
                    = k.fast == 1 || kind == ck::ActivationKind::gelu_and_mul;
            EXPECT_EQ(got != by_default, changes) << label;
        }
    }
}

// Selector bounds: every (MR, NV, act, source-scale layout) the
// dispatcher's table asks for is instantiated, and nothing past the register
// budget.
TEST(CkUkernelMxfp4, SelectorRefusesOutOfRange) {
    for (const int NV : {2, 4}) {
        const int max_mr = ck::max_mr_for_nv_mxfp4(NV);
        EXPECT_LE(max_mr, ck::kMaxMR);
        for (const auto act :
                {ck::ActivationKind::none, ck::ActivationKind::swiglu_oai_mul,
                        ck::ActivationKind::silu_and_mul,
                        ck::ActivationKind::gelu_and_mul}) {
            for (const bool per_block : {false, true}) {
                for (int mr = 1; mr <= max_mr; ++mr) {
                    EXPECT_NE(ck::select_mxfp4_ukernel(mr, NV, act, per_block),
                            nullptr)
                            << "NV=" << NV << " MR=" << mr;
                }
                EXPECT_EQ(ck::select_mxfp4_ukernel(
                                  max_mr + 1, NV, act, per_block),
                        nullptr)
                        << "NV=" << NV << " must refuse MR>" << max_mr;
                EXPECT_EQ(ck::select_mxfp4_ukernel(0, NV, act, per_block),
                        nullptr);
            }
        }
    }
    // NV outside {2, 4} has no instantiation.
    EXPECT_EQ(
            ck::select_mxfp4_ukernel(1, 3, ck::ActivationKind::none), nullptr);
    EXPECT_EQ(
            ck::select_mxfp4_ukernel(1, 1, ck::ActivationKind::none), nullptr);
}
