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

// Per-channel INT8 calls of the native BRGEMM on a bf16 source quantized per
// token in the call: f32, bf16 and f16 per-channel weight scales, f32, bf16
// and f16 source-scale buffers, given by the caller or allocated by the
// quantization, bf16 and f32 destinations.  Where the scales are equal in
// both types the outputs must be bit identical, and a given source-scale
// buffer must hold the per-token scales in its type.  Where the per-token
// scales round differently in bf16, the weight-scale type alone must still
// not change the bits.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "common/bfloat16.hpp"
#include "common/float16.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "matmul_native/native_brgemm_test_utils.hpp"

namespace {

namespace mm = zendnnl::lowoha::matmul;
namespace nat = zendnnl::lowoha::matmul::native;
namespace nt = native_brgemm_test;
using zendnnl::common::bfloat16_t;
using zendnnl::common::float16_t;
using zendnnl::common::matmul_algo_t;
using zendnnl::memory::data_type_t;
using zendnnl::memory::status_t;

constexpr data_type_t kBf16 = data_type_t::bf16;
constexpr data_type_t kF16 = data_type_t::f16;
constexpr data_type_t kF32 = data_type_t::f32;

const char *dt_name(data_type_t dt) {
    return dt == kF32 ? "f32" : dt == kBf16 ? "bf16" : "f16";
}

struct scales_case_t {
    int M, N, K;
    data_type_t dst;
    /// Every per-token scale a power of two, so equal in bf16 and f32.
    bool exact;
};

std::string case_name(const ::testing::TestParamInfo<scales_case_t> &info) {
    const scales_case_t &c = info.param;
    return "M" + std::to_string(c.M) + "_N" + std::to_string(c.N) + "_K"
            + std::to_string(c.K) + (c.dst == kBf16 ? "_bf16" : "_f32")
            + (c.exact ? "_exact" : "_rounded");
}

std::vector<scales_case_t> cases() {
    std::vector<scales_case_t> v;
    for (const bool exact : {true, false})
        for (const data_type_t dst : {kBf16, kF32}) {
            v.push_back({1, 96, 256, dst, exact});
            v.push_back({4, 512, 1024, dst, exact});
            v.push_back({33, 320, 1024, dst, exact});
            v.push_back({131, 1024, 512, dst, exact});
        }
    return v;
}

/// The operands of one case: an s8 [N, K] weight, its per-channel scales in
/// the three types with equal values (bf16 values, exact in f16 at these
/// magnitudes), and the bf16 [M, K] source.  An exact
/// source row m holds k / 64 * 2^(m % 4) for integers k in -127 .. 127 with
/// |k| = 127 at its first element, so its per-token scale is 2^(m % 4) / 64.
struct operands_t {
    nt::buffer_t wei, scale_f32, scale_bf16, scale_f16, src;
    const void *scale(data_type_t dt) const {
        return dt == kF32     ? scale_f32.get()
                : dt == kBf16 ? scale_bf16.get()
                              : scale_f16.get();
    }
};

operands_t make_operands(const scales_case_t &c) {
    std::mt19937 rng(static_cast<uint32_t>(c.M * 131 + c.N * 7 + c.K));
    operands_t o;
    o.wei = nt::random_buffer(data_type_t::s8, static_cast<size_t>(c.N) * c.K,
            static_cast<uint32_t>(c.N + c.K), -127.0f, 127.0f);
    o.scale_f32 = nt::make_buffer(static_cast<size_t>(c.N) * sizeof(float));
    o.scale_bf16 = nt::make_buffer(static_cast<size_t>(c.N) * sizeof(uint16_t));
    o.scale_f16 = nt::make_buffer(static_cast<size_t>(c.N) * sizeof(uint16_t));
    std::uniform_real_distribution<float> sd(0.001f, 0.05f);
    for (int n = 0; n < c.N; ++n) {
        const bfloat16_t b(sd(rng));
        o.scale_bf16.as<bfloat16_t>()[n] = b;
        o.scale_f32.as<float>()[n] = static_cast<float>(b);
        o.scale_f16.as<float16_t>()[n] = float16_t(static_cast<float>(b));
    }
    o.src = nt::make_buffer(static_cast<size_t>(c.M) * c.K * sizeof(uint16_t));
    bfloat16_t *s = o.src.as<bfloat16_t>();
    std::uniform_int_distribution<int> kd(-127, 127);
    std::uniform_real_distribution<float> fd(-3.0f, 5.0f);
    for (int m = 0; m < c.M; ++m) {
        const float unit = std::ldexp(1.0f, m % 4) / 64.0f;
        for (int k = 0; k < c.K; ++k) {
            const size_t i = static_cast<size_t>(m) * c.K + k;
            if (!c.exact)
                s[i] = bfloat16_t(fd(rng));
            else
                s[i] = bfloat16_t(
                        (k == 0 ? (m % 2 ? -127 : 127) : kd(rng)) * unit);
        }
    }
    return o;
}

/// One call under ALGO 11 with weight scales of wei_dt and source scales of
/// src_dt, in scale (given) or allocated by the quantization (scale empty).
nt::buffer_t run(const scales_case_t &c, const operands_t &o,
        data_type_t wei_dt, data_type_t src_dt, const nt::buffer_t &scale) {
    nt::call_t call;
    call.M = c.M;
    call.N = c.N;
    call.K = c.K;
    call.transB = true;
    call.src = o.src.get();
    call.wei = o.wei.get();
    nt::buffer_t dst = nt::make_buffer(
            static_cast<size_t>(c.M) * c.N * nt::dt_bytes(c.dst));
    call.dst = dst.get();
    auto &p = call.params;
    p.dtypes.src = kBf16;
    p.dtypes.wei = data_type_t::s8;
    p.dtypes.dst = c.dst;
    p.dtypes.compute = data_type_t::s8;
    p.dynamic_quant = true;
    p.weight_cache_type = 1;
    p.quant_params.src_scale.buff = scale.get();
    p.quant_params.src_scale.dt = src_dt;
    p.quant_params.src_scale.dims = {c.M, 1};
    p.quant_params.wei_scale.buff = o.scale(wei_dt);
    p.quant_params.wei_scale.dt = wei_dt;
    p.quant_params.wei_scale.dims = {1, c.N};
    const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
    // Every call must leave a layout of its own.
    mm::clear_matmul_weight_caches();
    EXPECT_EQ(nt::run_matmul(call), status_t::success);
    EXPECT_GE(nt::native_layouts(o.wei.get()), 1u)
            << "the native BRGEMM did not serve the call";
    return dst;
}

bool same_bytes(const nt::buffer_t &a, const nt::buffer_t &b, size_t bytes) {
    return std::memcmp(a.get(), b.get(), bytes) == 0;
}

// A served call is told by the layouts cached for its weight pointer, which
// a later allocation may reuse: each case starts and ends with empty caches,
// and each call starts with them (run).
class NativeBrgemmInt8Scales : public ::testing::TestWithParam<scales_case_t> {
protected:
    void SetUp() override { mm::clear_matmul_weight_caches(); }
    void TearDown() override { mm::clear_matmul_weight_caches(); }
};

TEST_P(NativeBrgemmInt8Scales, EveryScaleTypeGivesTheSameBits) {
    if (!nat::detect_uarch().avx512vnni) GTEST_SKIP() << "needs AVX512-VNNI";
    const scales_case_t c = GetParam();
    const operands_t o = make_operands(c);
    const size_t dst_bytes
            = static_cast<size_t>(c.M) * c.N * nt::dt_bytes(c.dst);
    const data_type_t types[] = {kF32, kBf16, kF16};
    nt::buffer_t want[3];
    for (int si = 0; si < 3; ++si) {
        const data_type_t src_dt = types[si];
        const size_t scale_bytes
                = static_cast<size_t>(c.M) * nt::dt_bytes(src_dt);
        nt::buffer_t first;
        for (const data_type_t wei_dt : types)
            for (const bool given : {true, false}) {
                SCOPED_TRACE(std::string("weight scales ") + dt_name(wei_dt)
                        + ", source scales " + dt_name(src_dt)
                        + (given ? " given" : " allocated"));
                nt::buffer_t scale;
                if (given) {
                    scale = nt::make_buffer(scale_bytes);
                    std::memset(scale.get(), 0xff, scale_bytes);
                }
                const nt::buffer_t dst = run(c, o, wei_dt, src_dt, scale);
                if (given) {
                    for (int m = 0; m < c.M; ++m) {
                        const float s = nt::load_f32(scale.get(), src_dt, m);
                        if (c.exact)
                            EXPECT_EQ(s, std::ldexp(1.0f, m % 4) / 64.0f)
                                    << "row " << m;
                        else
                            EXPECT_GT(s, 0.0f) << "row " << m;
                    }
                }
                if (first.get() == nullptr) {
                    first = dst;
                    continue;
                }
                EXPECT_TRUE(same_bytes(dst, first, dst_bytes))
                        << "differs from f32 weight scales, given buffer";
            }
        want[si] = first;
    }
    if (c.exact) {
        EXPECT_TRUE(same_bytes(want[0], want[1], dst_bytes))
                << "bf16 source scales equal to the f32 ones change the bits";
        EXPECT_TRUE(same_bytes(want[0], want[2], dst_bytes))
                << "f16 source scales equal to the f32 ones change the bits";
    }
}

INSTANTIATE_TEST_SUITE_P(NativeBrgemmInt8, NativeBrgemmInt8Scales,
        ::testing::ValuesIn(cases()), case_name);

} // namespace
