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

/// FUSED_GATED_LAYER_NORM: y = act(gate) * LayerNorm(x), with optional copies
/// of act(gate) and x, on dense rows and on the strided slices of a
/// concatenated [act(gate), x, y] output (the HSTU norm-mul of DLRM v3).
/// Both algorithms are checked against an FP64 oracle.

#include "common/zendnnl_global.hpp"
#include "lowoha_operators/normalization/lowoha_normalization.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <ostream>
#include <random>
#include <sstream>
#include <vector>

namespace {

using zendnnl::common::bfloat16_t;
using zendnnl::common::data_type_t;
using zendnnl::common::float16_t;
using zendnnl::common::status_t;
namespace nz = zendnnl::lowoha::normalization;

constexpr uint16_t kCanaryBits = 0x7FC1; // a NaN in both bf16 and f32 halves

bool has_avx512() {
    const auto &pi = zendnnl::common::zendnnl_platform_info();
    return pi.get_avx512f_status() && pi.get_avx512_bf16_status()
            && pi.get_avx512_bw_vl_status();
}

/// One typed buffer of `rows` rows with `stride` elements per row.
struct buf_t {
    data_type_t dt;
    uint64_t stride;
    std::vector<uint16_t> raw; // two uint16 per f32 element

    buf_t(data_type_t d, uint64_t rows, uint64_t s)
        : dt(d)
        , stride(s)
        , raw(rows * s * (d == data_type_t::f32 ? 2 : 1), kCanaryBits) {}

    void *ptr(uint64_t col = 0) {
        return raw.data() + col * (dt == data_type_t::f32 ? 2 : 1);
    }
    float get(uint64_t r, uint64_t c) const {
        const uint64_t i = r * stride + c;
        if (dt == data_type_t::bf16)
            return bfloat16_t::bf16_to_f32_val(static_cast<int16_t>(raw[i]));
        if (dt == data_type_t::f16) return float16_t::f16_to_f32_val(raw[i]);
        float v;
        std::memcpy(&v, &raw[2 * i], sizeof(v));
        return v;
    }
    void set(uint64_t r, uint64_t c, float v) {
        const uint64_t i = r * stride + c;
        if (dt == data_type_t::bf16) {
            raw[i] = static_cast<uint16_t>(bfloat16_t::f32_to_bf16_val(v));
        } else if (dt == data_type_t::f16) {
            raw[i] = float16_t::f32_to_f16_val(v);
        } else {
            std::memcpy(&raw[2 * i], &v, sizeof(v));
        }
    }
    bool canary(uint64_t r, uint64_t c) const {
        const uint64_t i = r * stride + c;
        if (dt != data_type_t::f32) return raw[i] == kCanaryBits;
        return raw[2 * i] == kCanaryBits && raw[2 * i + 1] == kCanaryBits;
    }
};

struct gated_case_t {
    uint64_t rows, n;
    data_type_t src_dt, dst_dt, gate_dt;
    bool silu, scale, shift;
    bool concat; ///< [act(gate) | x | y] slices of one strided output
    bool gate_copy = true; ///< with concat: write act(gate)
    bool src_copy = true; ///< with concat: write x
};

std::ostream &operator<<(std::ostream &os, const gated_case_t &c) {
    auto s = [](data_type_t d) {
        return d == data_type_t::bf16   ? "bf16"
                : d == data_type_t::f16 ? "f16"
                                        : "f32";
    };
    return os << "rows" << c.rows << "_n" << c.n << "_" << s(c.src_dt) << "_"
              << s(c.dst_dt) << "_gate" << s(c.gate_dt)
              << (c.silu ? "_silu" : "_id") << (c.scale ? "_g" : "")
              << (c.shift ? "_b" : "") << (c.concat ? "_concat" : "")
              << (c.concat && !c.gate_copy ? "_nogate" : "")
              << (c.concat && !c.src_copy ? "_nosrc" : "");
}

struct gated_problem_t {
    gated_case_t c;
    buf_t x, gate, out;
    std::vector<float> gamma, beta;
    uint64_t gate_col = 0, src_col = 0, y_col = 0;

    explicit gated_problem_t(const gated_case_t &cs)
        : c(cs)
        , x(cs.src_dt, cs.rows, cs.n + 3)
        , gate(cs.gate_dt, cs.rows, 2 * cs.n + 5)
        , out(cs.dst_dt, cs.rows, cs.concat ? 3 * cs.n + 7 : cs.n)
        , gamma(cs.n)
        , beta(cs.n) {
        std::mt19937 rng(1234 + cs.n * 7 + cs.rows);
        std::normal_distribution<float> xd(0.5f, 2.f), gd(0.f, 3.f),
                pd(1.f, 0.5f);
        for (uint64_t r = 0; r < c.rows; ++r)
            for (uint64_t i = 0; i < c.n; ++i) {
                x.set(r, i, xd(rng));
                gate.set(r, c.n + i, gd(rng)); // gate starts at column n
            }
        for (uint64_t i = 0; i < c.n; ++i) {
            gamma[i] = pd(rng);
            beta[i] = pd(rng) - 1.f;
        }
        if (c.concat) {
            src_col = c.n;
            y_col = 2 * c.n;
        }
    }

    nz::norm_params params(nz::norm_algo_t algo, int threads) {
        nz::norm_params p;
        p.norm_type = nz::norm_type_t::FUSED_GATED_LAYER_NORM;
        p.batch = c.rows;
        p.norm_size = c.n;
        p.epsilon = 1e-6f;
        p.use_scale = c.scale;
        p.use_shift = c.shift;
        p.src_dt = c.src_dt;
        p.dst_dt = c.dst_dt;
        p.gamma_dt = data_type_t::f32;
        p.beta_dt = data_type_t::f32;
        p.algorithm = algo;
        p.num_threads = threads;
        p.src_stride = x.stride;
        p.dst_stride = out.stride;
        p.gated.gate = gate.ptr(c.n);
        p.gated.gate_dt = c.gate_dt;
        p.gated.gate_stride = gate.stride;
        p.gated.gate_silu = c.silu;
        if (c.concat && c.gate_copy) {
            p.gated.gate_out = out.ptr(gate_col);
            p.gated.gate_out_stride = out.stride;
        }
        if (c.concat && c.src_copy) {
            p.gated.src_out = out.ptr(src_col);
            p.gated.src_out_stride = out.stride;
        }
        return p;
    }

    status_t run(nz::norm_algo_t algo, int threads = 0) {
        nz::norm_params p = params(algo, threads);
        return nz::normalization_direct(x.ptr(), out.ptr(y_col), gamma.data(),
                beta.data(), nullptr, nullptr, nullptr, p);
    }

    /// FP64 oracle on the stored (rounded) inputs.
    void check() const {
        const double tol = c.dst_dt == data_type_t::bf16 ? 1.0 / 128
                : c.dst_dt == data_type_t::f16           ? 2e-3
                                                         : 2e-5;
        for (uint64_t r = 0; r < c.rows; ++r) {
            double mean = 0, var = 0;
            for (uint64_t i = 0; i < c.n; ++i)
                mean += x.get(r, i);
            mean /= c.n;
            for (uint64_t i = 0; i < c.n; ++i) {
                const double d = x.get(r, i) - mean;
                var += d * d;
            }
            const double inv_std = 1.0 / std::sqrt(var / c.n + 1e-6);
            for (uint64_t i = 0; i < c.n; ++i) {
                double n = (x.get(r, i) - mean) * inv_std;
                if (c.scale) n *= gamma[i];
                if (c.shift) n += beta[i];
                double g = gate.get(r, c.n + i);
                if (c.silu) g = g / (1.0 + std::exp(-g));
                const double y = g * n;
                ASSERT_NEAR(out.get(r, y_col + i), y, tol * (1 + std::fabs(y)))
                        << "y row " << r << " col " << i;
                if (c.concat && c.gate_copy) {
                    ASSERT_NEAR(out.get(r, gate_col + i), g,
                            tol * (1 + std::fabs(g)))
                            << "act(gate) row " << r << " col " << i;
                } else if (c.concat) {
                    ASSERT_TRUE(out.canary(r, gate_col + i))
                            << "act(gate) written row " << r;
                }
                if (c.concat && c.src_copy) {
                    buf_t xc(c.dst_dt, 1, 1);
                    xc.set(0, 0, x.get(r, i));
                    ASSERT_EQ(out.get(r, src_col + i), xc.get(0, 0))
                            << "x copy row " << r << " col " << i;
                } else if (c.concat) {
                    ASSERT_TRUE(out.canary(r, src_col + i))
                            << "x copy written row " << r;
                }
            }
            const uint64_t used = c.concat ? 3 * c.n : c.n;
            for (uint64_t i = used; i < out.stride; ++i)
                ASSERT_TRUE(out.canary(r, i)) << "row pad overwritten " << r;
        }
    }
};

class GatedLayerNorm : public ::testing::TestWithParam<gated_case_t> {};

TEST_P(GatedLayerNorm, MatchesOracle) {
    if (!has_avx512()) GTEST_SKIP() << "needs AVX-512 BF16";
    for (auto algo :
            {nz::norm_algo_t::dynamic_dispatch, nz::norm_algo_t::reference}) {
        gated_problem_t pb(GetParam());
        ASSERT_EQ(pb.run(algo), status_t::success);
        pb.check();
    }
}

TEST_P(GatedLayerNorm, ThreadInvariant) {
    if (!has_avx512()) GTEST_SKIP() << "needs AVX-512 BF16";
    gated_problem_t a(GetParam()), b(GetParam());
    ASSERT_EQ(a.run(nz::norm_algo_t::dynamic_dispatch, 1), status_t::success);
    ASSERT_EQ(b.run(nz::norm_algo_t::dynamic_dispatch, 4), status_t::success);
    EXPECT_EQ(a.out.raw, b.out.raw);
}

std::vector<gated_case_t> gated_cases() {
    const auto f32 = data_type_t::f32, bf16 = data_type_t::bf16;
    std::vector<gated_case_t> v;
    // DLRM v3 STU: bf16 attention output, bf16 u, silu, affine, concat.
    v.push_back({37, 512, bf16, bf16, bf16, true, true, true, true});
    v.push_back({37, 512, bf16, bf16, bf16, true, true, true, false});
    // Tails (16-lane chunks plus masked remainder) and tiny rows.
    for (uint64_t n : {1, 15, 16, 17, 100, 129})
        v.push_back({9, n, bf16, bf16, bf16, true, true, true, true});
    // Dtype mixes and the identity activation / no affine.
    v.push_back({11, 96, f32, f32, f32, true, true, true, true});
    v.push_back({11, 96, bf16, f32, bf16, true, true, false, false});
    v.push_back({11, 96, f32, bf16, f32, false, false, true, true});
    v.push_back({11, 96, bf16, bf16, f32, false, false, false, false});
    v.push_back({1, 2048, bf16, bf16, bf16, true, true, true, true});
    // Only one of the copies.
    v.push_back(
            {7, 100, bf16, bf16, bf16, true, true, true, true, false, true});
    v.push_back(
            {7, 100, bf16, bf16, bf16, true, true, true, true, true, false});
    // f16 needs only AVX-512 (F16C conversions), not AVX512-FP16.
    const auto f16 = data_type_t::f16;
    v.push_back({13, 130, f16, f16, f16, true, true, true, true});
    v.push_back({13, 130, f16, f32, bf16, true, true, false, false});
    v.push_back({13, 130, bf16, bf16, f16, false, true, true, true});
    return v;
}

INSTANTIATE_TEST_SUITE_P(Normalization, GatedLayerNorm,
        ::testing::ValuesIn(gated_cases()),
        [](const ::testing::TestParamInfo<gated_case_t> &i) {
            std::ostringstream os;
            os << i.param;
            return os.str();
        });

TEST(GatedLayerNormGuards, RejectsInvalidParameters) {
    gated_problem_t pb({4, 64, data_type_t::bf16, data_type_t::bf16,
            data_type_t::bf16, true, true, true, true});
    auto run = [&](nz::norm_params p) {
        return nz::normalization_direct(pb.x.ptr(), pb.out.ptr(pb.y_col),
                pb.gamma.data(), pb.beta.data(), nullptr, nullptr, nullptr, p);
    };
    const auto ok = pb.params(nz::norm_algo_t::dynamic_dispatch, 0);
    EXPECT_EQ(run(ok), status_t::success);

    auto p = ok;
    p.gated.gate = nullptr;
    EXPECT_EQ(run(p), status_t::failure);
    p = ok;
    p.gated.gate_dt = data_type_t::s8;
    EXPECT_EQ(run(p), status_t::failure);
    p = ok;
    p.gated.gate_stride = 63;
    EXPECT_EQ(run(p), status_t::failure);
    p = ok;
    p.dst_stride = 10;
    EXPECT_EQ(run(p), status_t::failure);
    // Strides and a gate are rejected on the other norm types rather than
    // silently ignored.
    p = ok;
    p.norm_type = nz::norm_type_t::LAYER_NORM;
    EXPECT_EQ(run(p), status_t::failure);
    // Every gated field is rejected on the other types, not only the gate.
    nz::norm_params ln = ok;
    ln.norm_type = nz::norm_type_t::LAYER_NORM;
    ln.src_stride = ln.dst_stride = 0;
    ln.gated = nz::gated_norm_params();
    EXPECT_EQ(run(ln), status_t::success);
    p = ln;
    p.gated.gate_out = pb.out.ptr();
    EXPECT_EQ(run(p), status_t::failure);
    p = ln;
    p.gated.src_out_stride = 64;
    EXPECT_EQ(run(p), status_t::failure);
}

} // namespace
