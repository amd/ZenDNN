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

/// GDN's custom-kernel GEMMs (gdn_gemm_ck) for every operand type (bf16,
/// f32) against a naive reference on the operand values: gemm on row-major
/// and transposed A, gemm_rows, to_a, pack_b and pack_b_transposed, at edge
/// sizes.

#include "common/zendnnl_global.hpp"
#include "lowoha_operators/gdn/gdn_gemm_ck/lowoha_gdn_ck.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <ostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace ck = zendnnl::lowoha::gdn::ck;
using zendnnl::common::data_type_t;

/// The ISA the GDN CK GEMMs are compiled for.
bool has_isa() {
    const auto &pi = zendnnl::common::zendnnl_platform_info();
    return pi.get_avx512f_status() && pi.get_avx512_bw_vl_status()
            && pi.get_avx512_bf16_status();
}

#define GDN_CK_SKIP_UNLESS_ISA() \
    if (!has_isa()) { GTEST_SKIP() << "needs AVX512-F/BW/VL/BF16"; }

const char *dt_name(data_type_t dt) {
    return dt == data_type_t::bf16 ? "bf16" : "f32";
}

float bf16_to_f32(uint16_t h) {
    const uint32_t bits = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// Round to nearest even; NaN stays a quiet NaN.
uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    if ((bits & 0x7fffffffu) > 0x7f800000u) {
        return static_cast<uint16_t>((bits >> 16) | 0x40u);
    }
    return static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

/// Storage of operand type DT in the tests: element encoding and decoding of
/// A and B operands.
template <data_type_t DT>
struct operand_ref;

template <>
struct operand_ref<data_type_t::bf16> {
    static uint16_t encode(float x) { return f32_to_bf16(x); }
    static float decode(uint16_t h) { return bf16_to_f32(h); }
};
template <>
struct operand_ref<data_type_t::f32> {
    static float encode(float x) { return x; }
    static float decode(float x) { return x; }
};

template <data_type_t DT>
float round_to(float x) {
    return operand_ref<DT>::decode(operand_ref<DT>::encode(x));
}

std::vector<float> random_f32(size_t n, std::mt19937 &rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> v(n);
    for (auto &x : v) {
        x = nd(rng);
    }
    return v;
}

// B operand [R / kPack][N][kPack] of FP32 [R][N] (rows ld apart), encoded
// element by element.
template <data_type_t DT>
std::vector<ck::b_elem_t<DT>> ref_pack_b(
        const std::vector<float> &src, int64_t r, int64_t n, int64_t ld) {
    constexpr int64_t P = ck::operand<DT>::kPack;
    std::vector<ck::b_elem_t<DT>> out(static_cast<size_t>(r * n));
    for (int64_t g = 0; g < r / P; ++g) {
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t e = 0; e < P; ++e) {
                out[(g * n + j) * P + e]
                        = operand_ref<DT>::encode(src[(g * P + e) * ld + j]);
            }
        }
    }
    return out;
}

template <data_type_t DT>
std::vector<ck::a_elem_t<DT>> ref_to_a(const std::vector<float> &src) {
    std::vector<ck::a_elem_t<DT>> out(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        out[i] = operand_ref<DT>::encode(src[i]);
    }
    return out;
}

// C [M][N] (+)= A [M][K] B [K][N] in double; mag gets sum_k |A B| per
// element, which bounds the FP32 accumulation error.
void ref_gemm(int64_t m, int64_t n, int64_t k,
        const std::vector<float> &a, // [M][K]
        const std::vector<float> &b, // [K][N]
        std::vector<float> &c, int64_t ldc, bool accumulate,
        std::vector<float> &mag) {
    mag.assign(c.size(), 0.0f);
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            double acc = accumulate ? c[i * ldc + j] : 0.0;
            double abs_sum = accumulate ? std::fabs(c[i * ldc + j]) : 0.0;
            for (int64_t p = 0; p < k; ++p) {
                const double x
                        = static_cast<double>(a[i * k + p]) * b[p * n + j];
                acc += x;
                abs_sum += std::fabs(x);
            }
            c[i * ldc + j] = static_cast<float>(acc);
            mag[i * ldc + j] = static_cast<float>(abs_sum);
        }
    }
}

// Every multiply-add step rounds to FP32: bound k steps by 2 k u sum |A B|.
void expect_gemm_close(const std::vector<float> &got,
        const std::vector<float> &ref, const std::vector<float> &mag, int64_t m,
        int64_t n, int64_t k, int64_t ldc) {
    const float u = std::ldexp(1.0f, -24);
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < ldc; ++j) {
            const size_t x = i * ldc + j;
            if (j >= n) {
                // Columns past N keep their canary.
                EXPECT_EQ(got[x], ref[x]) << "row " << i << " col " << j;
                continue;
            }
            const float tol
                    = 2.0f * static_cast<float>(k + 1) * u * mag[x] + 1e-6f;
            EXPECT_NEAR(got[x], ref[x], tol) << "row " << i << " col " << j;
        }
    }
}

// Operands of every type built from the same FP32 values.
const data_type_t kTypes[] = {data_type_t::bf16, data_type_t::f32};

#define GDN_CK_DISPATCH(DT, FN, ...) \
    switch (DT) { \
        case data_type_t::bf16: FN<data_type_t::bf16>(__VA_ARGS__); break; \
        default: FN<data_type_t::f32>(__VA_ARGS__); break; \
    }

// to_a rounds to nearest even (ties, carries across a binade, signed zeros
// and infinities), and f32 copies.
template <data_type_t DT>
void check_to_a() {
    std::mt19937 rng(11);
    std::vector<float> src = random_f32(64, rng);
    const uint32_t bits[] = {0x3f808000u, 0x3f818000u, 0xbf808000u, 0xbf818000u,
            0x3fffc000u, 0x7f7fffffu, 0x00800000u, 0x40490fdbu};
    for (size_t i = 0; i < sizeof(bits) / sizeof(bits[0]); ++i) {
        std::memcpy(&src[i], &bits[i], sizeof(float));
    }
    src[30] = 0.0f;
    src[31] = -0.0f;
    src[32] = std::numeric_limits<float>::infinity();
    src[33] = -std::numeric_limits<float>::infinity();
    src[34] = 1e-30f;
    const std::vector<ck::a_elem_t<DT>> ref = ref_to_a<DT>(src);
    std::vector<ck::a_elem_t<DT>> got(src.size());
    ck::to_a<DT>(src.data(), static_cast<int64_t>(src.size()), got.data());
    for (size_t i = 0; i < src.size(); ++i) {
        const float g = operand_ref<DT>::decode(got[i]);
        const float r = operand_ref<DT>::decode(ref[i]);
        EXPECT_EQ(std::memcmp(&g, &r, sizeof(float)), 0)
                << "index " << i << ": " << g << " vs " << r;
    }
}

class GdnCkToA : public testing::TestWithParam<data_type_t> {};

TEST_P(GdnCkToA, RoundsToNearestEven) {
    GDN_CK_SKIP_UNLESS_ISA();
    GDN_CK_DISPATCH(GetParam(), check_to_a, )
}

std::string type_name(const testing::TestParamInfo<data_type_t> &info) {
    return dt_name(info.param);
}

INSTANTIATE_TEST_SUITE_P(Types, GdnCkToA, testing::ValuesIn(kTypes), type_name);

struct pack_case {
    data_type_t dt;
    int64_t r, n, ld;
};

std::string pack_name(const testing::TestParamInfo<pack_case> &info) {
    const pack_case &c = info.param;
    return std::string(dt_name(c.dt)) + "_R" + std::to_string(c.r) + "_N"
            + std::to_string(c.n) + "_ld" + std::to_string(c.ld);
}

template <data_type_t DT>
void check_pack_b(const pack_case &c) {
    std::mt19937 rng(13);
    const std::vector<float> src = random_f32(c.r * c.ld, rng);
    std::vector<ck::b_elem_t<DT>> got(c.r * c.n);
    ck::pack_b<DT>(src.data(), c.r, c.n, c.ld, got.data());
    EXPECT_EQ(got, ref_pack_b<DT>(src, c.r, c.n, c.ld));
}

class GdnCkPackB : public testing::TestWithParam<pack_case> {};

TEST_P(GdnCkPackB, MatchesReference) {
    GDN_CK_SKIP_UNLESS_ISA();
    GDN_CK_DISPATCH(GetParam().dt, check_pack_b, GetParam())
}

template <typename T>
std::vector<T> for_types(std::vector<T> cases) {
    std::vector<T> out;
    for (data_type_t dt : kTypes) {
        for (T c : cases) {
            c.dt = dt;
            out.push_back(c);
        }
    }
    return out;
}

INSTANTIATE_TEST_SUITE_P(Shapes, GdnCkPackB,
        testing::ValuesIn(for_types<pack_case>({{{}, 2, 16, 16},
                {{}, 2, 16, 19}, {{}, 64, 32, 32}, {{}, 64, 48, 64},
                {{}, 6, 256, 300}, {{}, 128, 128, 128}})),
        pack_name);

struct transpose_case {
    data_type_t dt;
    int64_t r, k;
};

std::string transpose_name(const testing::TestParamInfo<transpose_case> &info) {
    const transpose_case &c = info.param;
    return std::string(dt_name(c.dt)) + "_R" + std::to_string(c.r) + "_K"
            + std::to_string(c.k);
}

// dst step (k, r) is src step (r, k): [K / kPack][R][kPack] from A [R][K].
template <data_type_t DT>
void check_pack_b_transposed(const transpose_case &c) {
    constexpr int64_t P = ck::operand<DT>::kPack;
    std::mt19937 rng(17);
    const std::vector<ck::a_elem_t<DT>> src
            = ref_to_a<DT>(random_f32(c.r * c.k, rng));
    std::vector<ck::b_elem_t<DT>> got(c.r * c.k);
    ck::pack_b_transposed<DT>(src.data(), c.r, c.k, got.data());
    for (int64_t ks = 0; ks < c.k / P; ++ks) {
        for (int64_t r = 0; r < c.r; ++r) {
            for (int64_t e = 0; e < P; ++e) {
                ASSERT_EQ(operand_ref<DT>::decode(got[(ks * c.r + r) * P + e]),
                        operand_ref<DT>::decode(src[r * c.k + ks * P + e]))
                        << "step " << ks << " r " << r << " e " << e;
            }
        }
    }
}

class GdnCkPackBTransposed : public testing::TestWithParam<transpose_case> {};

TEST_P(GdnCkPackBTransposed, MatchesReference) {
    GDN_CK_SKIP_UNLESS_ISA();
    GDN_CK_DISPATCH(GetParam().dt, check_pack_b_transposed, GetParam())
}

INSTANTIATE_TEST_SUITE_P(Shapes, GdnCkPackBTransposed,
        testing::ValuesIn(for_types<transpose_case>({{{}, 16, 2}, {{}, 16, 30},
                {{}, 64, 64}, {{}, 64, 128}, {{}, 48, 256}, {{}, 128, 32}})),
        transpose_name);

struct gemm_case {
    data_type_t dt;
    int64_t m, n, k;
    bool trans_a; ///< A read as the transpose of a B operand [K][M]
    bool accumulate;
    int64_t pad; ///< extra FP32 columns per row of C
};

std::string gemm_name(const testing::TestParamInfo<gemm_case> &info) {
    const gemm_case &g = info.param;
    return std::string(dt_name(g.dt)) + "_M" + std::to_string(g.m) + "_N"
            + std::to_string(g.n) + "_K" + std::to_string(g.k)
            + (g.trans_a ? "_At" : "_A") + (g.accumulate ? "_acc" : "_set")
            + "_pad" + std::to_string(g.pad);
}

template <data_type_t DT>
void check_gemm(const gemm_case &g) {
    constexpr int64_t P = ck::operand<DT>::kPack;
    std::mt19937 rng(19);
    const int64_t ldc = g.n + g.pad;

    // B [K][N] as a B operand.
    std::vector<float> bf = random_f32(g.k * g.n, rng);
    for (auto &x : bf) {
        x = round_to<DT>(x);
    }
    const std::vector<ck::b_elem_t<DT>> b = ref_pack_b<DT>(bf, g.k, g.n, g.n);

    // A [M][K] as operand values, stored as a row-major A operand or, for
    // trans_a, as the B operand [K / kPack][M][kPack] of A^T [K][M].
    std::vector<float> af = random_f32(g.m * g.k, rng);
    for (auto &x : af) {
        x = round_to<DT>(x);
    }
    std::vector<float> c0 = random_f32(g.m * ldc, rng);
    std::vector<float> ref = c0, got = c0, mag;
    ref_gemm(g.m, g.n, g.k, af, bf, ref, ldc, g.accumulate, mag);
    if (g.trans_a) {
        std::vector<float> at(g.k * g.m);
        for (int64_t i = 0; i < g.m; ++i) {
            for (int64_t p = 0; p < g.k; ++p) {
                at[p * g.m + i] = af[i * g.k + p];
            }
        }
        const std::vector<ck::b_elem_t<DT>> a
                = ref_pack_b<DT>(at, g.k, g.m, g.m);
        ck::gemm<DT>(g.m, g.n, g.k / P,
                reinterpret_cast<const ck::a_step_t<DT> *>(a.data()), 1, g.m,
                b.data(), got.data(), ldc, g.accumulate);
    } else {
        const std::vector<ck::a_elem_t<DT>> a = ref_to_a<DT>(af);
        ck::gemm_rows<DT>(g.m, g.n, g.k, a.data(), b.data(), got.data(), ldc,
                g.accumulate);
    }
    expect_gemm_close(got, ref, mag, g.m, g.n, g.k, ldc);
}

class GdnCkGemm : public testing::TestWithParam<gemm_case> {};

TEST_P(GdnCkGemm, MatchesReference) {
    GDN_CK_SKIP_UNLESS_ISA();
    GDN_CK_DISPATCH(GetParam().dt, check_gemm, GetParam())
}

INSTANTIATE_TEST_SUITE_P(Shapes, GdnCkGemm,
        testing::ValuesIn(for_types<gemm_case>({
                // Row tails of the 6-row tile: 1 to 7 rows.
                {{}, 1, 32, 2, false, false, 0},
                {{}, 2, 32, 4, false, true, 3},
                {{}, 3, 64, 30, false, false, 0},
                {{}, 4, 32, 64, false, true, 0},
                {{}, 5, 96, 64, false, false, 16},
                {{}, 6, 32, 128, false, true, 0},
                {{}, 7, 64, 128, false, false, 1},
                {{}, 13, 256, 256, false, true, 0},
                // The chunk's shapes: 64 rows against head dims and the chunk.
                {{}, 64, 64, 128, false, false, 0},
                {{}, 64, 128, 64, false, true, 0},
                {{}, 64, 256, 256, false, false, 0},
                // A as the transpose of a B operand (the state update
                // K^T Delta).
                {{}, 16, 32, 2, true, false, 0},
                {{}, 64, 32, 64, true, true, 0},
                {{}, 128, 128, 64, true, true, 5},
                {{}, 256, 256, 64, true, false, 0},
        })),
        gemm_name);

// The chunk's K K^T: gemm_rows on K's A operand against
// pack_b_transposed(K).
template <data_type_t DT>
void check_kkt() {
    for (int64_t kdim : {32, 128, 256}) {
        SCOPED_TRACE(testing::Message() << "K " << kdim);
        const int64_t c = 64;
        std::mt19937 rng(23);
        const std::vector<ck::a_elem_t<DT>> ka
                = ref_to_a<DT>(random_f32(c * kdim, rng));
        std::vector<float> kf(ka.size()), kt(ka.size());
        for (int64_t i = 0; i < c; ++i) {
            for (int64_t p = 0; p < kdim; ++p) {
                kf[i * kdim + p] = operand_ref<DT>::decode(ka[i * kdim + p]);
                kt[p * c + i] = kf[i * kdim + p];
            }
        }
        std::vector<ck::b_elem_t<DT>> ktv(c * kdim);
        ck::pack_b_transposed<DT>(ka.data(), c, kdim, ktv.data());
        std::vector<float> got(c * c, 0.0f), ref(c * c, 0.0f), mag;
        ck::gemm_rows<DT>(
                c, c, kdim, ka.data(), ktv.data(), got.data(), c, false);
        ref_gemm(c, c, kdim, kf, kt, ref, c, false, mag);
        expect_gemm_close(got, ref, mag, c, c, kdim, c);
    }
}

class GdnCkGemmChain : public testing::TestWithParam<data_type_t> {};

TEST_P(GdnCkGemmChain, KKtMatchesReference) {
    GDN_CK_SKIP_UNLESS_ISA();
    GDN_CK_DISPATCH(GetParam(), check_kkt, )
}

INSTANTIATE_TEST_SUITE_P(
        Types, GdnCkGemmChain, testing::ValuesIn(kTypes), type_name);

// The chunk's triangular products: lower_a on a lower-triangular A and
// lower_c give, bit for bit, what the full product gives on the entries
// they compute; lower_c leaves the row tiles above each column block as
// they were.
template <data_type_t DT>
void check_triangles() {
    constexpr int64_t P = ck::operand<DT>::kPack;
    for (const auto &[m, n] : {std::pair<int64_t, int64_t> {64, 128}, {61, 128},
                 {64, 96}, {61, 32}}) {
        SCOPED_TRACE(testing::Message() << "M " << m << " N " << n);
        const int64_t k = m + m % 2, ldc = n + 3;
        std::mt19937 rng(29);
        std::vector<float> af = random_f32(m * k, rng);
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t p = i + 1; p < k; ++p) {
                af[i * k + p] = 0.0f;
            }
        }
        const std::vector<ck::a_elem_t<DT>> a = ref_to_a<DT>(af);
        const std::vector<ck::b_elem_t<DT>> b
                = ref_pack_b<DT>(random_f32(k * n, rng), k, n, n);
        const std::vector<float> c0 = random_f32(m * ldc, rng);
        for (bool acc : {false, true}) {
            std::vector<float> full = c0, low = c0;
            ck::gemm_rows<DT>(
                    m, n, k, a.data(), b.data(), full.data(), ldc, acc);
            ck::gemm_rows<DT>(m, n, k, a.data(), b.data(), low.data(), ldc, acc,
                    ck::tri_t::lower_a);
            EXPECT_EQ(std::memcmp(full.data(), low.data(),
                              full.size() * sizeof(float)),
                    0)
                    << "lower_a, accumulate " << acc;
        }
        std::vector<float> full = c0, low = c0;
        ck::gemm<DT>(m, n, k / P,
                reinterpret_cast<const ck::a_step_t<DT> *>(a.data()), k / P, 1,
                b.data(), full.data(), ldc, false);
        ck::gemm<DT>(m, n, k / P,
                reinterpret_cast<const ck::a_step_t<DT> *>(a.data()), k / P, 1,
                b.data(), low.data(), ldc, false, ck::tri_t::lower_c);
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t j = 0; j < ldc; ++j) {
                const size_t x = i * ldc + j;
                // Rows from the first 4-row tile holding row n0 of column j's
                // block: blocks of 64 columns, the last of 32 when n is an odd
                // multiple of 32.
                const int64_t n64 = n / 64 * 64;
                const int64_t n0 = j < n64 ? j / 64 * 64 : n64;
                const bool computed = j < n && i >= n0 / 4 * 4;
                ASSERT_EQ(low[x], computed ? full[x] : c0[x])
                        << "lower_c row " << i << " col " << j;
            }
        }
    }
}

class GdnCkGemmTriangles : public testing::TestWithParam<data_type_t> {};

TEST_P(GdnCkGemmTriangles, MatchFullProduct) {
    GDN_CK_SKIP_UNLESS_ISA();
    GDN_CK_DISPATCH(GetParam(), check_triangles, )
}

INSTANTIATE_TEST_SUITE_P(
        Types, GdnCkGemmTriangles, testing::ValuesIn(kTypes), type_name);

} // namespace
