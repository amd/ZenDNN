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

// Calls that matmul_direct serves on the AOCL-DLP route: the post-op metadata
// of each scale wiring, checked against the metadata builder and against an
// exact model of the call.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include "common/zendnnl_global.hpp"
#include "lowoha_operators/matmul/backends/aocl/aocl_postop.hpp"
#include "lowoha_operators/matmul/lowoha_matmul.hpp"
#include "matmul_native/native_brgemm_test_utils.hpp"

namespace {

namespace mm = zendnnl::lowoha::matmul;
using zendnnl::common::data_type_t;
using zendnnl::common::matmul_algo_t;
using zendnnl::common::post_op_type_t;
using zendnnl::error_handling::status_t;
using dims_t = std::vector<int64_t>;

// Deterministic small integers in [lo, hi].
struct values_t {
    uint32_t state;
    explicit values_t(uint32_t seed) : state(seed) {}
    int next(int lo, int hi) {
        state = state * 1664525u + 1013904223u;
        return lo
                + static_cast<int>(
                        (state >> 8) % static_cast<uint32_t>(hi - lo + 1));
    }
};

mm::matmul_quantization_params_t::matmul_quant_t quant(
        const void *buff, data_type_t dt, const dims_t &dims) {
    mm::matmul_quantization_params_t::matmul_quant_t q;
    q.buff = buff;
    q.dt = dt;
    q.dims = dims;
    return q;
}

} // namespace

// ── Post-op metadata cache: one source scale value, two wirings ─────────

namespace {

class AoclDlpPostOpCache : public ::testing::Test {
protected:
    void SetUp() override {
        if (!zendnnl::common::is_postop_cache_enabled())
            GTEST_SKIP() << "the post-op metadata cache is disabled";
        mm::clear_aocl_postop_metadata_cache();
    }
    void TearDown() override { mm::clear_aocl_postop_metadata_cache(); }
};

// The variants of an s8 x s8 call at M 1 with one source scale value.
enum class variant_t { plain, bias_and_mul, bf16_scales, weight_zero_point };

struct one_value_call_t {
    static constexpr int M = 1, N = 32, K = 64;
    std::vector<float> scales = std::vector<float>(64 * N, 0.25f);
    std::vector<uint16_t> bf16_scales = std::vector<uint16_t>(64 * N, 0x3e80);
    std::vector<float> bias = std::vector<float>(N, 0.5f);
    std::vector<float> operand = std::vector<float>(N, 1.5f);
    int32_t zero_point = 1;
    int8_t weight[K * N] = {};

    mm::matmul_params params(data_type_t dst, variant_t v, const dims_t &wei) {
        mm::matmul_params p;
        p.dtypes.src = data_type_t::s8;
        p.dtypes.wei = data_type_t::s8;
        p.dtypes.dst = dst;
        const bool bf16 = v == variant_t::bf16_scales;
        const void *s = bf16 ? static_cast<const void *>(bf16_scales.data())
                             : static_cast<const void *>(scales.data());
        const data_type_t sdt = bf16 ? data_type_t::bf16 : data_type_t::f32;
        p.quant_params.src_scale = quant(s, sdt, {});
        p.quant_params.wei_scale = quant(s, sdt, wei);
        if (v == variant_t::weight_zero_point)
            p.quant_params.wei_zp = quant(&zero_point, data_type_t::s32, {1});
        if (v == variant_t::bias_and_mul) {
            p.dtypes.bias = data_type_t::f32;
            mm::matmul_post_op po;
            po.po_type = post_op_type_t::binary_mul;
            po.buff = operand.data();
            po.dtype = data_type_t::f32;
            po.dims = {1, N};
            p.postop_.push_back(po);
        }
        return p;
    }

    dlp_metadata_t *create(const mm::matmul_params &p, variant_t v) {
        return mm::create_dlp_post_op(p,
                v == variant_t::bias_and_mul ? bias.data() : nullptr, p.dtypes,
                N, K, M, /*zp_comp_acc=*/nullptr, /*zp_comp_ndim=*/0,
                matmul_algo_t::aocl_dlp_blocked, weight,
                /*is_weights_const=*/true);
    }
};

// {G, N}: the sym-quant kernel's per-token source and per-group weight scales.
void expect_sym_quant_wiring(
        const dlp_metadata_t *md, const mm::matmul_params &p, int G) {
    const int M = one_value_call_t::M, N = one_value_call_t::N,
              K = one_value_call_t::K;
    ASSERT_NE(md, nullptr);
    ASSERT_NE(md->a_quant_op, nullptr);
    ASSERT_NE(md->b_quant_op, nullptr);
    EXPECT_EQ(md->a_quant_op->dequant_scale_factors->outer_dim,
            DLP_PARAM_DIM_PER_TOKEN);
    EXPECT_EQ(md->b_quant_op->dequant_scale_factors->len, G * N);
    EXPECT_EQ(md->b_quant_op->group_size,
            mm::sym_quant_group_size(p.quant_params, M, N, K));
    for (md_t i = 0; i < md->seq_length; ++i)
        EXPECT_NE(md->seq_vector[i], SCALE) << "scales are not post-ops here";
}

// {G, 1}: the source and weight scales as the first two SCALE post-ops.
void expect_scale_post_ops(const dlp_metadata_t *md, int G) {
    ASSERT_NE(md, nullptr);
    EXPECT_EQ(md->a_quant_op, nullptr);
    EXPECT_EQ(md->b_quant_op, nullptr);
    ASSERT_GE(md->seq_length, 2);
    ASSERT_NE(md->scale, nullptr);
    EXPECT_EQ(md->seq_vector[0], SCALE);
    EXPECT_EQ(md->scale[0].sf->scale_factor_len, 1);
    EXPECT_EQ(md->seq_vector[1], SCALE);
    EXPECT_EQ(md->scale[1].sf->scale_factor_len, G);
}

} // namespace

TEST_F(AoclDlpPostOpCache, OneSourceValueWiringsKeepTheirOwnHolders) {
    one_value_call_t c;
    const int N = one_value_call_t::N;
    for (const int G : {2, 3, 4, 16, 64})
        for (const data_type_t dst : {data_type_t::f32, data_type_t::bf16})
            for (const variant_t v : {variant_t::plain, variant_t::bias_and_mul,
                         variant_t::bf16_scales, variant_t::weight_zero_point})
                for (const bool sym_first : {true, false}) {
                    SCOPED_TRACE(::testing::Message()
                            << "G " << G << " dst "
                            << (dst == data_type_t::f32 ? "f32" : "bf16")
                            << " variant " << static_cast<int>(v)
                            << (sym_first ? " {G, N} first" : " {G, 1} first"));
                    mm::clear_aocl_postop_metadata_cache();
                    const mm::matmul_params group = c.params(dst, v, {G, N});
                    const mm::matmul_params column = c.params(dst, v, {G, 1});
                    dlp_metadata_t *first
                            = c.create(sym_first ? group : column, v);
                    dlp_metadata_t *second
                            = c.create(sym_first ? column : group, v);
                    EXPECT_NE(first, second);
                    expect_sym_quant_wiring(
                            sym_first ? first : second, group, G);
                    expect_scale_post_ops(sym_first ? second : first, G);
                    // Hits keep the holder of their own wiring.
                    EXPECT_EQ(c.create(group, v), sym_first ? first : second);
                    EXPECT_EQ(c.create(column, v), sym_first ? second : first);
                }
}

TEST_F(AoclDlpPostOpCache, ScalePostOpsShareTheirHolderAcrossRows) {
    // One source scale value with {1, N} weight scales takes the sym-quant
    // wiring at M 1 only for an s8 source without a zero point into f32 or
    // bf16; the other calls take the SCALE post-ops at every M, under one
    // holder.
    const int N = 32, K = 64;
    std::vector<float> scales(N, 0.25f);
    std::vector<int8_t> weight(static_cast<size_t>(K) * N, 1);
    int32_t zero_point = 3;
    struct case_t {
        data_type_t src;
        bool zp;
        data_type_t dst;
    };
    const case_t cases[] = {{data_type_t::u8, false, data_type_t::f32},
            {data_type_t::s8, true, data_type_t::f32},
            {data_type_t::bf16, false, data_type_t::bf16},
            {data_type_t::f32, false, data_type_t::f32},
            {data_type_t::s8, false, data_type_t::s32},
            {data_type_t::s8, false, data_type_t::s8},
            {data_type_t::s8, false, data_type_t::f32}};
    for (const case_t &k : cases) {
        SCOPED_TRACE(::testing::Message()
                << "src " << static_cast<int>(k.src) << " zp " << k.zp
                << " dst " << static_cast<int>(k.dst));
        mm::clear_aocl_postop_metadata_cache();
        mm::matmul_params p;
        p.dtypes.src = k.src;
        p.dtypes.wei = data_type_t::s8;
        p.dtypes.dst = k.dst;
        p.quant_params.src_scale = quant(scales.data(), data_type_t::f32, {});
        p.quant_params.wei_scale
                = quant(scales.data(), data_type_t::f32, {1, N});
        if (k.zp)
            p.quant_params.src_zp = quant(&zero_point, data_type_t::s32, {1});
        const auto create = [&](int M) {
            return mm::create_dlp_post_op(p, nullptr, p.dtypes, N, K, M,
                    /*zp_comp_acc=*/nullptr, /*zp_comp_ndim=*/0,
                    matmul_algo_t::aocl_dlp_blocked, weight.data(),
                    /*is_weights_const=*/true);
        };
        const dlp_metadata_t *one = create(1);
        const dlp_metadata_t *four = create(4);
        const bool sym_quant = k.src == data_type_t::s8 && !k.zp
                && k.dst == data_type_t::f32;
        EXPECT_EQ(one == four, !sym_quant);
        EXPECT_EQ(
                mm::get_aocl_postop_metadata_cache_size(), sym_quant ? 2u : 1u);
    }
}

namespace {

// matmul_direct, s8 x s8 -> f32 with one source scale value and per-channel
// weight scales on one weight: at M 1, {1, N} takes the sym-quant wiring and
// {N} the SCALE post-ops; over several rows both take the SCALE post-ops.
// The two wirings reorder the weight into different images, so the weight
// is cached out of place.
struct per_channel_call_t {
    static constexpr int max_M = 8, N = 64, K = 128;
    std::vector<int8_t> src = std::vector<int8_t>(max_M * K);
    std::vector<int8_t> weight = std::vector<int8_t>(K * N);
    std::vector<float> wei_scale = std::vector<float>(N);
    float src_scale = 0.5f;

    per_channel_call_t() {
        values_t r(11);
        for (auto &v : src)
            v = static_cast<int8_t>(r.next(-9, 9));
        for (auto &v : weight)
            v = static_cast<int8_t>(r.next(-7, 7));
        for (int n = 0; n < N; ++n)
            wei_scale[n] = 0.01f * static_cast<float>(1 + n % 5);
    }

    status_t call(
            int M, const dims_t &wei_dims, std::vector<float> &dst) const {
        mm::matmul_params p;
        p.dtypes.src = data_type_t::s8;
        p.dtypes.wei = data_type_t::s8;
        p.dtypes.dst = data_type_t::f32;
        p.quant_params.src_scale = quant(&src_scale, data_type_t::f32, {1});
        p.quant_params.wei_scale
                = quant(wei_scale.data(), data_type_t::f32, wei_dims);
        p.weight_cache_type = 1;
        dst.assign(static_cast<size_t>(M) * N, -1.0e30f);
        mm::matmul_batch_params_t batch;
        return mm::matmul_direct('r', false, false, M, N, K, 1.0f, src.data(),
                K, weight.data(), N, nullptr, 0.0f, dst.data(), N, true, batch,
                p);
    }

    void expect_model(int M, const std::vector<float> &dst) const {
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) {
                int32_t acc = 0;
                for (int k = 0; k < K; ++k)
                    acc += src[static_cast<size_t>(m) * K + k]
                            * weight[static_cast<size_t>(k) * N + n];
                const float ref
                        = static_cast<float>(acc) * src_scale * wei_scale[n];
                EXPECT_NEAR(dst[static_cast<size_t>(m) * N + n], ref,
                        1e-4f * (1.0f + std::fabs(ref)))
                        << m << ", " << n;
            }
    }
};

} // namespace

TEST_F(AoclDlpPostOpCache, PerChannelScaleDimsOnOneWeight) {
    const int N = per_channel_call_t::N;
    for (const bool sym_first : {true, false}) {
        SCOPED_TRACE(sym_first ? "{1, N} first" : "{N} first");
        mm::clear_matmul_weight_caches();
        per_channel_call_t c;
        const dims_t first = sym_first ? dims_t {1, N} : dims_t {N};
        const dims_t second = sym_first ? dims_t {N} : dims_t {1, N};
        std::vector<float> dst;
        ASSERT_EQ(c.call(1, first, dst), status_t::success);
        c.expect_model(1, dst);
        ASSERT_EQ(c.call(1, second, dst), status_t::success);
        c.expect_model(1, dst);
        ASSERT_EQ(c.call(1, first, dst), status_t::success);
        c.expect_model(1, dst);
    }
}

TEST_F(AoclDlpPostOpCache, OneLayoutAtOneAndSeveralRows) {
    const int N = per_channel_call_t::N;
    for (const bool one_row_first : {true, false}) {
        SCOPED_TRACE(one_row_first ? "M 1 first" : "M 8 first");
        mm::clear_matmul_weight_caches();
        per_channel_call_t c;
        std::vector<float> dst;
        for (const int M : one_row_first ? std::vector<int> {1, 8, 1}
                                         : std::vector<int> {8, 1, 8}) {
            ASSERT_EQ(c.call(M, {1, N}, dst), status_t::success);
            c.expect_model(M, dst);
        }
    }
}

// ── Calls and their exact models ────────────────────────────────────────

namespace {

uint16_t to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}

float from_bf16(uint16_t b) {
    const uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

size_t element_size(data_type_t t) {
    switch (t) {
        case data_type_t::f32:
        case data_type_t::s32: return 4;
        case data_type_t::bf16: return 2;
        default: return 1;
    }
}

float load(const std::vector<uint8_t> &buf, data_type_t t, size_t i) {
    const uint8_t *p = buf.data() + i * element_size(t);
    switch (t) {
        case data_type_t::f32: {
            float v;
            std::memcpy(&v, p, sizeof(v));
            return v;
        }
        case data_type_t::s32: {
            int32_t v;
            std::memcpy(&v, p, sizeof(v));
            return static_cast<float>(v);
        }
        case data_type_t::bf16: {
            uint16_t v;
            std::memcpy(&v, p, sizeof(v));
            return from_bf16(v);
        }
        case data_type_t::s8: return static_cast<int8_t>(*p);
        case data_type_t::u8: return *p;
        default: return 0.0f;
    }
}

void store(std::vector<uint8_t> &buf, data_type_t t, size_t i, float v) {
    uint8_t *p = buf.data() + i * element_size(t);
    switch (t) {
        case data_type_t::f32: std::memcpy(p, &v, sizeof(v)); break;
        case data_type_t::bf16: {
            const uint16_t b = to_bf16(v);
            std::memcpy(p, &b, sizeof(b));
            break;
        }
        case data_type_t::s32: {
            const int32_t s = static_cast<int32_t>(v);
            std::memcpy(p, &s, sizeof(s));
            break;
        }
        case data_type_t::s8:
            *p = static_cast<uint8_t>(static_cast<int8_t>(v));
            break;
        case data_type_t::u8: *p = static_cast<uint8_t>(v); break;
        default: break;
    }
}

// An f32 result as the destination holds it: rounded to nearest even, and
// saturated for an integer type.
double in_destination(double v, data_type_t t) {
    switch (t) {
        case data_type_t::bf16:
            return from_bf16(to_bf16(static_cast<float>(v)));
        case data_type_t::s8:
            return std::nearbyint(std::fmin(127.0, std::fmax(-128.0, v)));
        case data_type_t::u8:
            return std::nearbyint(std::fmin(255.0, std::fmax(0.0, v)));
        case data_type_t::s32: return std::nearbyint(v);
        default: return static_cast<float>(v);
    }
}

double eltwise(const mm::matmul_post_op &po, double x) {
    const double alpha = po.alpha_or_default();
    switch (po.po_type) {
        case post_op_type_t::relu: return x > 0 ? x : 0;
        case post_op_type_t::leaky_relu: return x > 0 ? x : alpha * x;
        case post_op_type_t::elu: return x > 0 ? x : alpha * std::expm1(x);
        case post_op_type_t::gelu_tanh:
            return 0.5 * x
                    * (1
                            + std::tanh(0.7978845608028654
                                    * (x + 0.044715 * x * x * x)));
        case post_op_type_t::gelu_erf:
            return 0.5 * x * (1 + std::erf(x / std::sqrt(2.0)));
        case post_op_type_t::sigmoid: return 1 / (1 + std::exp(-x));
        case post_op_type_t::swish: return x / (1 + std::exp(-alpha * x));
        case post_op_type_t::tanh: return std::tanh(x);
        case post_op_type_t::clip:
            return std::fmin(std::fmax(x, alpha), po.beta);
        case post_op_type_t::mish:
            return x * std::tanh(std::log1p(std::exp(x)));
        case post_op_type_t::square: return x * x;
        case post_op_type_t::abs: return std::fabs(x);
        case post_op_type_t::sqrt: return std::sqrt(x);
        case post_op_type_t::exp: return std::exp(x);
        case post_op_type_t::log: return std::log(x);
        default: return x;
    }
}

// The chain over output row m; a binary operand of dims {1, N} is one value
// per column, any other one rows of leading_dim values.
void apply_chain(const std::vector<mm::matmul_post_op> &chain, int m, int N,
        std::vector<double> &row) {
    for (const auto &po : chain) {
        if (po.po_type == post_op_type_t::softmax) {
            double mx = row[0], sum = 0;
            for (int j = 1; j < N; ++j)
                mx = std::fmax(mx, row[j]);
            for (int j = 0; j < N; ++j)
                sum += std::exp(row[j] - mx);
            for (int j = 0; j < N; ++j)
                row[j] = std::exp(row[j] - mx) / sum;
        } else if (po.po_type == post_op_type_t::binary_add
                || po.po_type == post_op_type_t::binary_mul) {
            const bool column = po.dims.size() == 2 && po.dims[0] == 1;
            const int ld = po.leading_dim < 0 ? N : po.leading_dim;
            for (int j = 0; j < N; ++j) {
                const size_t i = column ? j : static_cast<size_t>(m) * ld + j;
                const double y = po.dtype == data_type_t::bf16
                        ? from_bf16(static_cast<const uint16_t *>(po.buff)[i])
                        : static_cast<const float *>(po.buff)[i];
                row[j] = po.po_type == post_op_type_t::binary_add ? row[j] + y
                                                                  : row[j] * y;
            }
        } else {
            for (int j = 0; j < N; ++j)
                row[j] = eltwise(po, row[j]);
        }
    }
}

mm::matmul_post_op eltwise_op(
        post_op_type_t t, float alpha = std::nanf(""), float beta = 0.0f) {
    mm::matmul_post_op po;
    po.po_type = t;
    po.alpha = alpha;
    po.beta = beta;
    return po;
}

// matmul_direct with s8 weights, alpha 1 and beta 0.  Every source value is
// q * s for small integers q and powers of two s, and the weight scales are
// small multiples of powers of two, so the per-group dequantized sums are
// exact in f32 and the model rounds once, into the destination.
struct int8_call_t {
    int M = 4, N = 64, K = 256;
    data_type_t src_dt = data_type_t::s8;
    data_type_t dst_dt = data_type_t::f32;
    dims_t src_scale_dims {1};
    dims_t wei_scale_dims; // {4, N} when empty
    bool src_zp = false; // a per-tensor source zero point
    bool dst_q = false; // a per-tensor destination scale and zero point
    bool dynamic = false; // the source quantized in the call to compute
    data_type_t compute = data_type_t::none;
    data_type_t src_scale_dt = data_type_t::f32; // f32 or bf16
    data_type_t wei_scale_dt = data_type_t::f32; // f32 or bf16
    matmul_algo_t algo = matmul_algo_t::none;
    bool weights_const = true;
    int num_threads = 0; // the call's, 0 for all
    std::vector<mm::matmul_post_op> postops;
    uint32_t seed = 1;

    std::vector<int32_t> q; // the source as the kernel multiplies it
    std::vector<uint8_t> src, dst;
    std::vector<int8_t> weight;
    std::vector<float> src_scale, wei_scale;
    std::vector<uint16_t> src_scale_bf16, wei_scale_bf16;
    int32_t zero_point = 0;
    float dst_scale = 0.5f;
    int32_t dst_zero_point = 3;

    dims_t wei_dims() const {
        return wei_scale_dims.empty() ? dims_t {4, N} : wei_scale_dims;
    }
    int64_t wei_count() const {
        int64_t c = 1;
        for (const int64_t d : wei_dims())
            c *= d;
        return c;
    }
    // Weight groups along K: count / N for a whole number of rows of N.
    int wei_groups() const {
        const int64_t c = wei_count();
        return c % N == 0 && c > N ? static_cast<int>(c / N) : 1;
    }
    int src_groups() const {
        int64_t c = 1;
        for (const int64_t d : src_scale_dims)
            c *= d;
        return c > M ? static_cast<int>(c / M) : 1;
    }
    bool src_per_token() const {
        int64_t c = 1;
        for (const int64_t d : src_scale_dims)
            c *= d;
        return c == M && M > 1;
    }

    // The source scale of row m and K group g of src_groups().
    float s_src(int m, int g) const {
        if (dynamic) return 1.0f / 64.0f;
        if (src_groups() > 1) return std::ldexp(1.0f, -(1 + (m + g) % 3));
        if (src_per_token()) return std::ldexp(1.0f, -(1 + m % 3));
        return 0.5f;
    }
    float s_wei(int g, int n) const {
        if (wei_count() == 1) return 0.125f;
        if (wei_groups() == 1)
            return std::ldexp(static_cast<float>(1 + n % 3), -5);
        return std::ldexp(static_cast<float>(1 + n % 3), -(4 + g % 3));
    }

    // A new weight may reuse the address of a freed one, which keys the
    // weight and post-op caches.
    void prepare_weight() {
        mm::clear_matmul_weight_caches();
        values_t r(seed * 7919u + 1u);
        weight.resize(static_cast<size_t>(K) * N);
        for (auto &v : weight)
            v = static_cast<int8_t>(r.next(-7, 7));
        const int G = wei_groups();
        wei_scale.resize(static_cast<size_t>(wei_count()));
        wei_scale_bf16.resize(wei_scale.size());
        for (size_t i = 0; i < wei_scale.size(); ++i) {
            const int n = static_cast<int>(i % N);
            wei_scale[i] = s_wei(G > 1 ? static_cast<int>(i / N) : 0, n);
            wei_scale_bf16[i] = to_bf16(wei_scale[i]);
        }
    }

    void prepare_source() {
        values_t r(seed);
        const int Gs = src_groups(), gs = K / Gs;
        q.resize(static_cast<size_t>(M) * K);
        for (auto &v : q)
            v = dynamic ? r.next(-120, 120) : r.next(-9, 9);
        if (dynamic) {
            // max |q| = 127 within every scale group, so the scale the call
            // computes is 1 / 64 and the quantized source is q.
            const bool per_row = src_per_token() || Gs > 1;
            for (int m = 0; m < (per_row ? M : 1); ++m)
                for (int g = 0; g < Gs; ++g)
                    q[static_cast<size_t>(m) * K + static_cast<size_t>(g) * gs]
                            = 127;
        }
        zero_point = src_zp ? (src_dt == data_type_t::u8 ? 10 : 3) : 0;
        src.assign(static_cast<size_t>(M) * K * element_size(src_dt), 0);
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < K; ++k) {
                const size_t i = static_cast<size_t>(m) * K + k;
                const int g = k / gs;
                if (src_dt == data_type_t::s8)
                    src[i] = static_cast<uint8_t>(
                            static_cast<int8_t>(q[i] + zero_point));
                else if (src_dt == data_type_t::u8)
                    src[i] = static_cast<uint8_t>(q[i] + 9 + zero_point);
                else
                    store(src, src_dt, i,
                            static_cast<float>(q[i]) * s_src(m, g));
            }
        if (src_dt == data_type_t::u8)
            for (size_t i = 0; i < q.size(); ++i)
                q[i] += 9;
        int64_t count = 1;
        for (const int64_t d : src_scale_dims)
            count *= d;
        src_scale.assign(static_cast<size_t>(count), 0.0f);
        src_scale_bf16.resize(src_scale.size());
        for (int64_t i = 0; i < count; ++i) {
            src_scale[i] = Gs > 1
                    ? s_src(static_cast<int>(i / Gs), static_cast<int>(i % Gs))
                    : s_src(static_cast<int>(i), 0);
            src_scale_bf16[i] = to_bf16(src_scale[i]);
        }
        dst.assign(static_cast<size_t>(M) * N * element_size(dst_dt), 0x5a);
    }

    void prepare() {
        prepare_weight();
        prepare_source();
    }

    mm::matmul_params params() const {
        mm::matmul_params p;
        p.dtypes.src = src_dt;
        p.dtypes.wei = data_type_t::s8;
        p.dtypes.dst = dst_dt;
        p.dtypes.compute = compute;
        p.dynamic_quant = dynamic;
        p.lowoha_algo = algo;
        p.quant_params.src_scale = src_scale_dt == data_type_t::bf16
                ? quant(src_scale_bf16.data(), data_type_t::bf16,
                          src_scale_dims)
                : quant(src_scale.data(), data_type_t::f32, src_scale_dims);
        p.quant_params.wei_scale = wei_scale_dt == data_type_t::bf16
                ? quant(wei_scale_bf16.data(), data_type_t::bf16, wei_dims())
                : quant(wei_scale.data(), data_type_t::f32, wei_dims());
        if (src_zp)
            p.quant_params.src_zp = quant(&zero_point, data_type_t::s32, {1});
        if (dst_q) {
            p.quant_params.dst_scale = quant(&dst_scale, data_type_t::f32, {1});
            p.quant_params.dst_zp
                    = quant(&dst_zero_point, data_type_t::s32, {1});
        }
        p.postop_ = postops;
        p.num_threads = num_threads;
        return p;
    }

    status_t run() {
        const mm::matmul_params p = params();
        mm::matmul_batch_params_t batch;
        return mm::matmul_direct('r', false, false, M, N, K, 1.0f, src.data(),
                K, weight.data(), N, nullptr, 0.0f, dst.data(), N,
                weights_const, batch, p);
    }

    // Per-group dequantization (one group when the weight scale has none),
    // the chain, then the destination scale and zero point.
    std::vector<double> model() const {
        const int G = wei_groups(), Gs = src_groups();
        const int groups = std::max(G, Gs), gs = K / groups;
        std::vector<double> out(static_cast<size_t>(M) * N);
        std::vector<double> row(N);
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                double acc = 0;
                for (int g = 0; g < groups; ++g) {
                    int64_t part = 0;
                    for (int k = g * gs; k < (g + 1) * gs; ++k)
                        part += static_cast<int64_t>(
                                        q[static_cast<size_t>(m) * K + k])
                                * weight[static_cast<size_t>(k) * N + n];
                    acc += static_cast<double>(part) * s_src(m, Gs > 1 ? g : 0)
                            * s_wei(G > 1 ? g : 0, n);
                }
                row[n] = acc;
            }
            apply_chain(postops, m, N, row);
            for (int n = 0; n < N; ++n) {
                double v = row[n];
                if (dst_q) v = v * dst_scale + dst_zero_point;
                out[static_cast<size_t>(m) * N + n] = in_destination(v, dst_dt);
            }
        }
        return out;
    }

    void expect_model() const {
        const std::vector<double> ref = model();
        const bool integer = dst_dt == data_type_t::s8
                || dst_dt == data_type_t::u8 || dst_dt == data_type_t::s32;
        int bad = 0;
        for (size_t i = 0; i < ref.size() && bad < 8; ++i) {
            const double got = load(dst, dst_dt, i);
            const double tol = integer ? 1.0
                    : dst_dt == data_type_t::bf16
                    ? std::ldexp(std::fabs(ref[i]), -7) + 1e-6
                    : 1e-4 * (1 + std::fabs(ref[i]));
            if (!(std::fabs(got - ref[i]) <= tol)) {
                ADD_FAILURE() << "element " << i << ": " << got << ", model "
                              << ref[i];
                ++bad;
            }
        }
    }

    bool dst_untouched() const {
        for (const uint8_t b : dst)
            if (b != 0x5a) return false;
        return true;
    }
};

} // namespace

// ── Per-group weight scales on the AOCL-DLP route ───────────────────────

TEST(AoclDlpPerGroupWeights, PerTensorSourceIsComputed) {
    for (const int G : {2, 4, 8})
        for (const int M : {1, 2, 5, 16, 33})
            for (const data_type_t dst : {data_type_t::f32, data_type_t::bf16})
                for (const dims_t &s : {dims_t {}, dims_t {1}, dims_t {1, 1}})
                    for (const matmul_algo_t algo :
                            {matmul_algo_t::none, matmul_algo_t::aocl_dlp,
                                    matmul_algo_t::aocl_dlp_blocked})
                        for (const bool weights_const : {true, false}) {
                            SCOPED_TRACE(::testing::Message()
                                    << "G " << G << " M " << M << " dst "
                                    << static_cast<int>(dst) << " src dims "
                                    << s.size() << " algo "
                                    << static_cast<int>(algo) << " const "
                                    << weights_const);
                            int8_call_t c;
                            c.M = M;
                            c.dst_dt = dst;
                            c.src_scale_dims = s;
                            c.wei_scale_dims = {G, c.N};
                            c.algo = algo;
                            c.weights_const = weights_const;
                            c.seed = static_cast<uint32_t>(G * 100 + M);
                            c.prepare();
                            ASSERT_EQ(c.run(), status_t::success);
                            c.expect_model();
                        }
}

TEST(AoclDlpPerGroupWeights, PerTensorSourceWithPostOps) {
    for (const int M : {3, 16})
        for (const auto &chain : {std::vector<mm::matmul_post_op> {
                                          eltwise_op(post_op_type_t::relu)},
                     std::vector<mm::matmul_post_op> {
                             eltwise_op(post_op_type_t::gelu_tanh),
                             eltwise_op(post_op_type_t::clip, -4.0f, 4.0f)},
                     std::vector<mm::matmul_post_op> {
                             eltwise_op(post_op_type_t::mish)}}) {
            int8_call_t c;
            c.M = M;
            c.dst_dt = data_type_t::bf16;
            c.wei_scale_dims = {4, c.N};
            c.postops = chain;
            c.prepare();
            ASSERT_EQ(c.run(), status_t::success);
            c.expect_model();
        }
}

TEST(AoclDlpPerGroupWeights, OneWeightAtSeveralRowCounts) {
    mm::clear_matmul_weight_caches();
    int8_call_t c;
    c.wei_scale_dims = {4, c.N};
    c.prepare_weight();
    for (const int M : {1, 8, 1, 33, 8, 2}) {
        SCOPED_TRACE(M);
        c.M = M;
        c.seed = static_cast<uint32_t>(M + 3);
        c.prepare_source();
        ASSERT_EQ(c.run(), status_t::success);
        c.expect_model();
    }
}

TEST(AoclDlpPerGroupWeights, DynamicPerTensorSourceIsComputed) {
    for (const data_type_t src : {data_type_t::bf16, data_type_t::f32})
        for (const int M : {4, 17}) {
            int8_call_t c;
            c.M = M;
            c.src_dt = src;
            c.dst_dt = src;
            c.dynamic = true;
            c.compute = data_type_t::s8;
            c.src_scale_dims = {1, 1};
            c.wei_scale_dims = {4, c.N};
            c.prepare();
            ASSERT_EQ(c.run(), status_t::success);
            c.expect_model();
        }
}

TEST(AoclDlpPerGroupWeights, SourceScaleGroupingsKeepTheirResults) {
    for (const int M : {1, 6}) {
        for (const dims_t &s : {dims_t {M, 1}, dims_t {M, 4}}) {
            int8_call_t c;
            c.M = M;
            c.src_scale_dims = s;
            c.wei_scale_dims = {4, c.N};
            c.prepare();
            ASSERT_EQ(c.run(), status_t::success);
            c.expect_model();
        }
    }
}

TEST(AoclDlpPerGroupWeights, UnservedCallsFail) {
    struct case_t {
        const char *name;
        data_type_t src, dst;
        dims_t src_dims; // a 0 row count stands for M
        bool src_zp, dst_q, dynamic;
        data_type_t compute;
        int K, G; // G 0: a flat {4 * N} weight scale
    };
    const data_type_t none = data_type_t::none;
    const case_t cases[] = {
            {"u8 source", data_type_t::u8, data_type_t::f32, {1}, false, false,
                    false, none, 256, 4},
            {"u8 source with zero point", data_type_t::u8, data_type_t::bf16,
                    {1}, true, false, false, none, 256, 4},
            {"u8 source into u8", data_type_t::u8, data_type_t::u8, {1}, true,
                    true, false, none, 256, 4},
            {"s8 source with zero point", data_type_t::s8, data_type_t::f32,
                    {1}, true, false, false, none, 256, 4},
            {"s8 into s8", data_type_t::s8, data_type_t::s8, {1}, false, true,
                    false, none, 256, 4},
            {"s8 into s32", data_type_t::s8, data_type_t::s32, {1}, false,
                    false, false, none, 256, 4},
            {"bf16 source", data_type_t::bf16, data_type_t::bf16, {1}, false,
                    false, false, none, 256, 4},
            {"f32 source", data_type_t::f32, data_type_t::f32, {1}, false,
                    false, false, none, 256, 4},
            {"bf16 source, per-token scale", data_type_t::bf16,
                    data_type_t::bf16, {0, 1}, false, false, false, none, 256,
                    4},
            {"bf16 source, per-group scale", data_type_t::bf16,
                    data_type_t::bf16, {0, 4}, false, false, false, none, 256,
                    4},
            {"bf16 source quantized to u8", data_type_t::bf16,
                    data_type_t::bf16, {1, 1}, true, false, true,
                    data_type_t::u8, 256, 4},
            {"flat weight scale", data_type_t::s8, data_type_t::f32, {1}, false,
                    false, false, none, 256, 0},
            {"groups of 2", data_type_t::s8, data_type_t::f32, {0, 1}, false,
                    false, false, none, 256, 128},
            {"groups of 6", data_type_t::s8, data_type_t::bf16, {0, 1}, false,
                    false, false, none, 96, 16},
            {"groups of 2, per-tensor source", data_type_t::s8,
                    data_type_t::f32, {1}, false, false, false, none, 64, 32},
            {"three groups do not split K", data_type_t::s8, data_type_t::f32,
                    {0, 1}, false, false, false, none, 256, 3},
    };
    for (const case_t &k : cases)
        for (const int M : {1, 4})
            for (const matmul_algo_t algo :
                    {matmul_algo_t::none, matmul_algo_t::aocl_dlp}) {
                SCOPED_TRACE(::testing::Message()
                        << k.name << " M " << M << " algo "
                        << static_cast<int>(algo));
                int8_call_t c;
                c.M = M;
                c.K = k.K;
                c.src_dt = k.src;
                c.dst_dt = k.dst;
                c.src_scale_dims = k.src_dims;
                if (!c.src_scale_dims.empty() && c.src_scale_dims[0] == 0)
                    c.src_scale_dims[0] = M;
                c.src_zp = k.src_zp;
                c.dst_q = k.dst_q;
                c.dynamic = k.dynamic;
                c.compute = k.compute;
                c.algo = algo;
                c.wei_scale_dims
                        = k.G == 0 ? dims_t {4 * c.N} : dims_t {k.G, c.N};
                c.prepare();
                // The default route, whatever algorithm the process runs.
                const native_brgemm_test::algo_pin_t unset(matmul_algo_t::none);
                EXPECT_EQ(c.run(), status_t::unimplemented);
                EXPECT_TRUE(c.dst_untouched());
            }
}

// bf16 weight scales: with f32 source scales, the calls the sym-quant kernel
// would take fail with dst untouched; with bf16 source scales that kernel
// serves them; the SCALE post-ops apply them; the reference kernel, which
// runs the calls with a post-op the AOCL-DLP chain does not take, reads
// them.
TEST(AoclDlpPerGroupWeights, Bf16WeightScales) {
    using P = post_op_type_t;
    struct case_t {
        const char *name;
        data_type_t src, dst;
        dims_t src_dims, wei_dims; // a 0 row count stands for M
        bool dst_q;
        int M;
        bool served;
        std::vector<P> ops;
        data_type_t src_scale_dt = data_type_t::f32;
    };
    const int N = 64;
    const data_type_t s8 = data_type_t::s8, u8 = data_type_t::u8,
                      f32 = data_type_t::f32, bf16 = data_type_t::bf16;
    const case_t cases[] = {
            {"one source value at one row", s8, f32, {1}, {1, N}, false, 1,
                    false, {}},
            {"one source value at one row into bf16", s8, bf16, {1}, {1, N},
                    false, 1, false, {}},
            {"per-token source", s8, f32, {0, 1}, {1, N}, false, 4, false, {}},
            {"per-token source at one row", s8, bf16, {0, 1}, {1, N}, false, 1,
                    false, {}},
            {"per-group weights", s8, f32, {1}, {4, N}, false, 4, false, {}},
            {"per-group weights, per-token source", s8, bf16, {0, 1}, {4, N},
                    false, 33, false, {}},
            {"elu", s8, f32, {1}, {1, N}, false, 1, true, {P::elu}},
            {"relu and elu into bf16", s8, bf16, {0, 1}, {1, N}, false, 4, true,
                    {P::relu, P::elu}},
            {"several rows", s8, f32, {1}, {1, N}, false, 4, true, {}},
            {"u8 source", u8, f32, {1}, {1, N}, false, 1, true, {}},
            {"s8 into s8", s8, s8, {1}, {1, N}, true, 1, true, {}},
            {"s8 into s32, abs", s8, data_type_t::s32, {1}, {1, N}, false, 1,
                    true, {P::abs}},
            {"s8 into s8, elu", s8, s8, {1}, {1, N}, true, 1, true, {P::elu}},
            {"s8 into u8, per-token source at one row", s8, u8, {0, 1}, {1, N},
                    true, 1, true, {P::relu, P::sqrt}},
            {"bf16 source scale at one row", s8, f32, {1}, {1, N}, false, 1,
                    true, {}, bf16},
            {"bf16 per-token source scale", s8, bf16, {0, 1}, {1, N}, false, 4,
                    true, {P::elu}, bf16},
            {"bf16 source scale, per-group weights", s8, f32, {0, 4}, {4, N},
                    false, 4, true, {}, bf16},
    };
    for (const case_t &k : cases)
        for (const matmul_algo_t algo :
                {matmul_algo_t::none, matmul_algo_t::aocl_dlp,
                        matmul_algo_t::aocl_dlp_blocked}) {
            SCOPED_TRACE(::testing::Message()
                    << k.name << " algo " << static_cast<int>(algo));
            int8_call_t c;
            c.M = k.M;
            c.src_dt = k.src;
            c.dst_dt = k.dst;
            c.src_scale_dims = k.src_dims;
            if (!c.src_scale_dims.empty() && c.src_scale_dims[0] == 0)
                c.src_scale_dims[0] = k.M;
            c.wei_scale_dims = k.wei_dims;
            c.src_scale_dt = k.src_scale_dt;
            c.wei_scale_dt = bf16;
            c.dst_q = k.dst_q;
            c.algo = algo;
            for (const P t : k.ops)
                c.postops.push_back(eltwise_op(t));
            c.prepare();
            const native_brgemm_test::algo_pin_t unset(matmul_algo_t::none);
            if (k.served) {
                ASSERT_EQ(c.run(), status_t::success);
                c.expect_model();
            } else {
                EXPECT_EQ(c.run(), status_t::unimplemented);
                EXPECT_TRUE(c.dst_untouched());
            }
        }
}

// ── Post-ops the AOCL-DLP chain does not take: the reference kernel ─────

namespace {

// The default route, aocl_dlp and aocl_dlp_blocked: a call with a post-op
// the AOCL-DLP chain does not take runs on the reference kernel on each.
const matmul_algo_t kAoclDlpRoutes[] = {matmul_algo_t::none,
        matmul_algo_t::aocl_dlp, matmul_algo_t::aocl_dlp_blocked};

// matmul_direct with bf16 or f32 operands and destination; every value is a
// small multiple of a power of two, so the product is exact in f32.
struct float_call_t {
    int M = 16, N = 64, K = 96;
    char layout = 'r';
    data_type_t dt = data_type_t::f32;
    float alpha = 1.0f, beta = 0.0f;
    bool bias = false;
    int num_threads = 0; // the call's, 0 for all
    matmul_algo_t algo = matmul_algo_t::none;
    std::vector<mm::matmul_post_op> postops;
    std::vector<std::vector<uint8_t>> operands;
    std::vector<float> a, w, c0, bias_values;
    std::vector<uint8_t> src, weight, dst;

    // A binary operand of dims {M, N} or {1, N}, of dtype f32 or bf16.
    void add_binary(post_op_type_t t, bool column, data_type_t odt) {
        values_t r(static_cast<uint32_t>(operands.size() + 17));
        const size_t n = column ? N : static_cast<size_t>(M) * N;
        operands.emplace_back(n * element_size(odt));
        for (size_t i = 0; i < n; ++i)
            store(operands.back(), odt, i,
                    static_cast<float>(r.next(-8, 8)) / 8.0f
                            + (t == post_op_type_t::binary_mul ? 1.5f : 0.0f));
        mm::matmul_post_op po;
        po.po_type = t;
        po.buff = operands.back().data();
        po.dtype = odt;
        po.dims = column ? dims_t {1, N} : dims_t {M, N};
        postops.push_back(po);
    }

    void prepare() {
        mm::clear_matmul_weight_caches();
        values_t r(static_cast<uint32_t>(M * 31 + N));
        a.resize(static_cast<size_t>(M) * K);
        w.resize(static_cast<size_t>(K) * N);
        c0.resize(static_cast<size_t>(M) * N);
        bias_values.resize(N);
        for (auto &v : a)
            v = static_cast<float>(r.next(-8, 8)) / 16.0f;
        for (auto &v : w)
            v = static_cast<float>(r.next(-8, 8)) / 32.0f;
        for (auto &v : c0)
            v = static_cast<float>(r.next(-8, 8)) / 16.0f;
        for (auto &v : bias_values)
            v = static_cast<float>(r.next(-8, 8)) / 16.0f;
        src.assign(a.size() * element_size(dt), 0);
        weight.assign(w.size() * element_size(dt), 0);
        dst.assign(c0.size() * element_size(dt), 0);
        for (size_t i = 0; i < a.size(); ++i)
            store(src, dt, i, a[i]);
        for (size_t i = 0; i < w.size(); ++i)
            store(weight, dt, i, w[i]);
        for (size_t i = 0; i < c0.size(); ++i)
            store(dst, dt, i, c0[i]);
    }

    // On the default route whatever algorithm the process runs, or on algo.
    status_t run() {
        mm::matmul_params p;
        p.dtypes.src = dt;
        p.dtypes.wei = dt;
        p.dtypes.dst = dt;
        p.dtypes.bias = data_type_t::f32;
        p.lowoha_algo = algo;
        p.postop_ = postops;
        p.num_threads = num_threads;
        mm::matmul_batch_params_t batch;
        const native_brgemm_test::algo_pin_t unset(matmul_algo_t::none);
        return mm::matmul_direct(layout, false, false, M, N, K, alpha,
                src.data(), K, weight.data(), N,
                bias ? bias_values.data() : nullptr, beta, dst.data(), N, true,
                batch, p);
    }

    // dst holds the bytes the same call writes with lowoha_algo reference.
    void expect_reference_bytes() const {
        float_call_t r = *this;
        r.algo = matmul_algo_t::reference;
        r.prepare();
        ASSERT_EQ(r.run(), status_t::success);
        EXPECT_EQ(dst, r.dst) << "not the reference kernel's bytes";
    }

    void expect_model() const {
        std::vector<double> row(N);
        int bad = 0;
        for (int m = 0; m < M && bad < 8; ++m) {
            for (int n = 0; n < N; ++n) {
                double acc = 0;
                for (int k = 0; k < K; ++k)
                    acc += static_cast<double>(
                                   a[static_cast<size_t>(m) * K + k])
                            * w[static_cast<size_t>(k) * N + n];
                acc = acc * alpha
                        + static_cast<double>(beta)
                                * c0[static_cast<size_t>(m) * N + n];
                if (bias) acc += bias_values[n];
                row[n] = acc;
            }
            apply_chain(postops, m, N, row);
            for (int n = 0; n < N && bad < 8; ++n) {
                const double ref = in_destination(row[n], dt);
                const double got
                        = load(dst, dt, static_cast<size_t>(m) * N + n);
                const double tol = dt == data_type_t::bf16
                        ? std::ldexp(std::fabs(ref), -7) + 1e-6
                        : 1e-3 * std::fabs(ref) + 1e-5;
                if (!(std::fabs(got - ref) <= tol)) {
                    ADD_FAILURE() << "row " << m << " column " << n << ": "
                                  << got << ", model " << ref;
                    ++bad;
                }
            }
        }
    }
};

// Runs a call on each route and with lowoha_algo reference: success, and the
// reference kernel's bytes on each route.  run(algo, dst) runs the call into
// dst, which holds the call's initial destination.
template <typename Run>
void expect_reference_bytes_on_routes(
        const std::vector<uint8_t> &initial, const Run &run) {
    std::vector<uint8_t> want = initial;
    ASSERT_EQ(run(matmul_algo_t::reference, want), status_t::success);
    for (const matmul_algo_t algo : kAoclDlpRoutes) {
        SCOPED_TRACE(::testing::Message() << "algo " << static_cast<int>(algo));
        std::vector<uint8_t> got = initial;
        mm::clear_matmul_weight_caches();
        ASSERT_EQ(run(algo, got), status_t::success);
        EXPECT_EQ(got, want) << "not the reference kernel's bytes";
    }
}

} // namespace

TEST(AoclDlpUnmappedPostOps, FloatChainsRunOnTheReferenceKernel) {
    using P = post_op_type_t;
    struct chain_t {
        const char *name;
        std::vector<P> ops;
        int binary; // 0 none, 1 {M, N} f32 add last, 2 {1, N} bf16 mul third
    };
    const chain_t chains[] = {{"elu", {P::elu}, 0}, {"square", {P::square}, 0},
            {"abs", {P::abs}, 0}, {"relu sqrt", {P::relu, P::sqrt}, 0},
            {"exp", {P::exp}, 0}, {"exp log", {P::exp, P::log}, 0},
            {"softmax", {P::softmax}, 0},
            {"square relu", {P::square, P::relu}, 0},
            {"relu square add", {P::relu, P::square}, 1},
            {"gelu_erf abs mul tanh", {P::gelu_erf, P::abs, P::tanh}, 2},
            {"softmax add", {P::softmax}, 1}};
    for (const matmul_algo_t algo : kAoclDlpRoutes)
        for (const data_type_t dt : {data_type_t::f32, data_type_t::bf16})
            for (const int M : {1, 5, 16})
                for (const chain_t &ch : chains)
                    for (const bool with_beta : {false, true}) {
                        SCOPED_TRACE(::testing::Message()
                                << ch.name << " algo " << static_cast<int>(algo)
                                << " dt " << static_cast<int>(dt) << " M " << M
                                << " beta " << with_beta);
                        float_call_t c;
                        c.M = M;
                        c.dt = dt;
                        c.algo = algo;
                        c.bias = with_beta;
                        c.beta = with_beta ? 0.5f : 0.0f;
                        c.alpha = with_beta ? 0.5f : 1.0f;
                        for (size_t i = 0; i < ch.ops.size(); ++i) {
                            if (ch.binary == 2 && i == 2)
                                c.add_binary(
                                        P::binary_mul, true, data_type_t::bf16);
                            c.postops.push_back(eltwise_op(ch.ops[i]));
                        }
                        if (ch.binary == 1)
                            c.add_binary(
                                    P::binary_add, false, data_type_t::f32);
                        c.prepare();
                        ASSERT_EQ(c.run(), status_t::success);
                        c.expect_reference_bytes();
                        c.expect_model();
                    }
}

TEST(AoclDlpUnmappedPostOps, WideAndParallelChains) {
    // N 1100 and M x N past the reference kernel's parallel bound, on all
    // threads and on one.
    using P = post_op_type_t;
    struct chain_t {
        const char *name;
        std::vector<P> ops;
        int binary; // 0 none, 1 an {M, N} f32 add last, 2 a {1, N} f32 mul last
    };
    const chain_t chains[] = {{"abs add", {P::abs}, 1},
            {"softmax", {P::softmax}, 0}, {"square mul", {P::square}, 2},
            {"exp log", {P::exp, P::log}, 0}};
    for (const data_type_t dt : {data_type_t::f32, data_type_t::bf16})
        for (const int M : {3, 64})
            for (const int threads : {0, 1})
                for (const chain_t &ch : chains) {
                    SCOPED_TRACE(::testing::Message()
                            << ch.name << " dt " << static_cast<int>(dt)
                            << " M " << M << " threads " << threads);
                    float_call_t c;
                    c.M = M;
                    c.N = 1100;
                    c.dt = dt;
                    c.alpha = 0.5f;
                    c.beta = 0.5f;
                    c.num_threads = threads;
                    for (const P t : ch.ops)
                        c.postops.push_back(eltwise_op(t));
                    if (ch.binary == 1)
                        c.add_binary(P::binary_add, false, data_type_t::f32);
                    if (ch.binary == 2)
                        c.add_binary(P::binary_mul, true, data_type_t::f32);
                    c.prepare();
                    ASSERT_EQ(c.run(), status_t::success);
                    c.expect_reference_bytes();
                    c.expect_model();
                }
}

TEST(AoclDlpUnmappedPostOps, ParameterisedOpsAfterAnUnmappedOne) {
    using P = post_op_type_t;
    for (const data_type_t dt : {data_type_t::f32, data_type_t::bf16}) {
        float_call_t c;
        c.dt = dt;
        c.postops = {eltwise_op(P::abs), eltwise_op(P::clip, -0.25f, 0.375f),
                eltwise_op(P::elu, 0.5f), eltwise_op(P::swish, 1.5f),
                eltwise_op(P::leaky_relu, 0.125f)};
        c.prepare();
        ASSERT_EQ(c.run(), status_t::success);
        c.expect_reference_bytes();
        c.expect_model();
    }
}

TEST(AoclDlpUnmappedPostOps, Int8ChainsRunOnTheReferenceKernel) {
    using P = post_op_type_t;
    struct case_t {
        const char *name;
        data_type_t src, dst;
        dims_t src_dims, wei_dims;
        bool src_zp, dst_q;
        std::vector<P> ops;
    };
    const int N = 64;
    const case_t cases[] = {
            {"per-token s8 into f32", data_type_t::s8, data_type_t::f32, {0, 1},
                    {N}, false, false, {P::square}},
            {"per-token s8 into f32 with dst scale", data_type_t::s8,
                    data_type_t::f32, {0, 1}, {1, N}, false, true,
                    {P::relu, P::sqrt}},
            {"per-tensor s8 into s8", data_type_t::s8, data_type_t::s8, {1},
                    {N}, false, true, {P::abs, P::relu}},
            {"u8 into u8", data_type_t::u8, data_type_t::u8, {1}, {N}, true,
                    true, {P::relu, P::elu}},
            {"u8 into s32", data_type_t::u8, data_type_t::s32, {1}, {}, true,
                    false, {P::square}},
            {"per-group s8 into bf16", data_type_t::s8, data_type_t::bf16,
                    {0, 4}, {4, N}, false, false, {P::abs, P::sqrt}},
    };
    for (const case_t &k : cases)
        for (const int M : {1, 6}) {
            SCOPED_TRACE(::testing::Message() << k.name << " M " << M);
            int8_call_t c;
            c.M = M;
            c.src_dt = k.src;
            c.dst_dt = k.dst;
            c.src_scale_dims = k.src_dims;
            if (!c.src_scale_dims.empty() && c.src_scale_dims[0] == 0)
                c.src_scale_dims[0] = M;
            c.wei_scale_dims = k.wei_dims.empty() ? dims_t {1} : k.wei_dims;
            c.src_zp = k.src_zp;
            c.dst_q = k.dst_q;
            for (const P t : k.ops)
                c.postops.push_back(eltwise_op(t));
            c.prepare();
            const std::vector<uint8_t> initial = c.dst;
            expect_reference_bytes_on_routes(
                    initial, [&c](matmul_algo_t a, std::vector<uint8_t> &d) {
                c.algo = a;
                c.dst = d;
                const native_brgemm_test::algo_pin_t unset(matmul_algo_t::none);
                const status_t st = c.run();
                d = c.dst;
                return st;
            });
            c.algo = matmul_algo_t::none;
            c.dst = initial;
            const native_brgemm_test::algo_pin_t unset(matmul_algo_t::none);
            ASSERT_EQ(c.run(), status_t::success);
            c.expect_model();
        }
}

// Calls of other kinds the reference kernel serves: the bytes of the same
// call with lowoha_algo reference on the default route, aocl_dlp and
// aocl_dlp_blocked.
TEST(AoclDlpUnmappedPostOps, ReferenceKernelServesOtherCalls) {
    using P = post_op_type_t;
    const native_brgemm_test::algo_pin_t unset(matmul_algo_t::none);
    const int M = 4, N = 64, K = 128;
    values_t r(5);
    {
        SCOPED_TRACE("W4A8: an s8 source, packed s4 weights, bf16 dst");
        const int G = 4;
        std::vector<int8_t> src(static_cast<size_t>(M) * K);
        for (auto &v : src)
            v = static_cast<int8_t>(r.next(-9, 9));
        std::vector<uint8_t> weight(static_cast<size_t>(K) * N / 2);
        for (auto &v : weight)
            v = static_cast<uint8_t>(r.next(0, 255));
        std::vector<float> src_scale(M, 0.5f), wei_scale(G * N, 0.25f);
        const std::vector<uint8_t> initial(static_cast<size_t>(M) * N * 2, 0);
        expect_reference_bytes_on_routes(
                initial, [&](matmul_algo_t a, std::vector<uint8_t> &d) {
            mm::matmul_params p;
            p.dtypes.src = data_type_t::s8;
            p.dtypes.wei = data_type_t::s4;
            p.dtypes.dst = data_type_t::bf16;
            p.dtypes.compute = data_type_t::s8;
            p.lowoha_algo = a;
            p.quant_params.src_scale
                    = quant(src_scale.data(), data_type_t::f32, {M, 1});
            p.quant_params.wei_scale
                    = quant(wei_scale.data(), data_type_t::f32, {G, N});
            p.postop_ = {eltwise_op(P::elu)};
            mm::matmul_batch_params_t batch;
            return mm::matmul_direct('r', false, false, M, N, K, 1.0f,
                    src.data(), K, weight.data(), N, nullptr, 0.0f, d.data(), N,
                    true, batch, p);
        });
    }
    for (const matmul_algo_t batch_algo :
            {matmul_algo_t::none, matmul_algo_t::batched_sgemm}) {
        SCOPED_TRACE(::testing::Message() << "two f32 batches, batch algo "
                                          << static_cast<int>(batch_algo));
        const int B = 2;
        std::vector<float> a(static_cast<size_t>(B) * M * K);
        std::vector<float> b(static_cast<size_t>(B) * K * N);
        for (auto &v : a)
            v = static_cast<float>(r.next(-8, 8)) / 16.0f;
        for (auto &v : b)
            v = static_cast<float>(r.next(-8, 8)) / 32.0f;
        const std::vector<uint8_t> initial(
                static_cast<size_t>(B) * M * N * sizeof(float), 0);
        expect_reference_bytes_on_routes(
                initial, [&](matmul_algo_t a_algo, std::vector<uint8_t> &d) {
            mm::matmul_params p;
            p.dtypes.src = data_type_t::f32;
            p.dtypes.wei = data_type_t::f32;
            p.dtypes.dst = data_type_t::f32;
            p.lowoha_algo = a_algo == matmul_algo_t::none ? batch_algo : a_algo;
            p.postop_ = {eltwise_op(P::softmax), eltwise_op(P::elu)};
            mm::matmul_batch_params_t batches;
            batches.Batch_A = B;
            batches.Batch_B = B;
            return mm::matmul_direct('r', false, false, M, N, K, 1.0f, a.data(),
                    K, b.data(), N, nullptr, 0.0f, d.data(), N, false, batches,
                    p);
        });
    }
    {
        SCOPED_TRACE("bf16 on an [N, K] weight");
        std::vector<uint16_t> a(static_cast<size_t>(M) * K);
        std::vector<uint16_t> b(static_cast<size_t>(N) * K);
        for (auto &v : a)
            v = to_bf16(static_cast<float>(r.next(-8, 8)) / 16.0f);
        for (auto &v : b)
            v = to_bf16(static_cast<float>(r.next(-8, 8)) / 32.0f);
        const std::vector<uint8_t> initial(static_cast<size_t>(M) * N * 2, 0);
        expect_reference_bytes_on_routes(
                initial, [&](matmul_algo_t algo, std::vector<uint8_t> &d) {
            mm::matmul_params p;
            p.dtypes.src = data_type_t::bf16;
            p.dtypes.wei = data_type_t::bf16;
            p.dtypes.dst = data_type_t::bf16;
            p.lowoha_algo = algo;
            p.postop_ = {eltwise_op(P::square)};
            mm::matmul_batch_params_t batch;
            return mm::matmul_direct('r', false, true, M, N, K, 1.0f, a.data(),
                    K, b.data(), K, nullptr, 0.0f, d.data(), N, true, batch, p);
        });
    }
    {
        SCOPED_TRACE("bf16 quantized in the call per token, into f32");
        int8_call_t c;
        c.M = M;
        c.K = K;
        c.src_dt = data_type_t::bf16;
        c.dynamic = true;
        c.compute = data_type_t::s8;
        c.src_scale_dims = {M, 1};
        c.wei_scale_dims = {1, N};
        c.postops = {eltwise_op(P::exp)};
        c.prepare();
        expect_reference_bytes_on_routes(
                c.dst, [&c](matmul_algo_t a, std::vector<uint8_t> &d) {
            c.prepare_source();
            c.algo = a;
            c.dst = d;
            const status_t st = c.run();
            d = c.dst;
            return st;
        });
    }
    for (const data_type_t dst : {data_type_t::f32, data_type_t::bf16}) {
        // bf16 x s4 with {2, N} weight scales at K 64, which AOCL-DLP's
        // pre-op translation declines.
        SCOPED_TRACE(::testing::Message()
                << "bf16 x s4, {2, N} weight scales, dst "
                << static_cast<int>(dst));
        const int k = 64, G = 2;
        std::vector<uint16_t> a(static_cast<size_t>(M) * k);
        for (auto &v : a)
            v = to_bf16(static_cast<float>(r.next(-4, 4)) * 0.25f);
        std::vector<uint8_t> weight(static_cast<size_t>(k) * N / 2);
        for (auto &v : weight)
            v = static_cast<uint8_t>(r.next(0, 255));
        std::vector<float> scale(static_cast<size_t>(G) * N);
        for (size_t i = 0; i < scale.size(); ++i)
            scale[i] = std::ldexp(static_cast<float>(1 + i % 3), -5);
        const std::vector<uint8_t> initial(
                static_cast<size_t>(M) * N * element_size(dst), 0xc0);
        for (const P t : {P::elu, P::softmax})
            expect_reference_bytes_on_routes(
                    initial, [&](matmul_algo_t algo, std::vector<uint8_t> &d) {
                mm::matmul_params p;
                p.dtypes.src = data_type_t::bf16;
                p.dtypes.wei = data_type_t::s4;
                p.dtypes.dst = dst;
                p.lowoha_algo = algo;
                p.quant_params.wei_scale
                        = quant(scale.data(), data_type_t::f32, {G, N});
                p.postop_ = {eltwise_op(t)};
                mm::matmul_batch_params_t batch;
                return mm::matmul_direct('r', false, false, M, N, k, 1.0f,
                        a.data(), k, weight.data(), N, nullptr, 0.0f, d.data(),
                        N, true, batch, p);
            });
    }
}

TEST(AoclDlpUnmappedPostOps, CallsTheReferenceKernelDoesNotServeFail) {
    using P = post_op_type_t;
    for (const data_type_t dt : {data_type_t::f32, data_type_t::bf16})
        for (const matmul_algo_t algo : kAoclDlpRoutes)
            for (const bool pooling : {true, false}) {
                SCOPED_TRACE(::testing::Message()
                        << (pooling ? "pooling" : "column-major elu")
                        << " algo " << static_cast<int>(algo));
                float_call_t c;
                c.dt = dt;
                c.algo = algo;
                c.layout = pooling ? 'r' : 'c';
                c.postops = {eltwise_op(P::relu),
                        eltwise_op(pooling ? P::pooling : P::elu)};
                c.prepare();
                const std::vector<uint8_t> before = c.dst;
                EXPECT_EQ(c.run(), status_t::unimplemented);
                EXPECT_EQ(c.dst, before);
            }
    {
        SCOPED_TRACE("a binary operand of rows N + 8 values apart");
        float_call_t c;
        c.postops = {eltwise_op(P::abs)};
        c.add_binary(P::binary_add, false, data_type_t::f32);
        c.operands.back().resize(
                static_cast<size_t>(c.M) * (c.N + 8) * sizeof(float));
        c.postops.back().buff = c.operands.back().data();
        c.postops.back().leading_dim = c.N + 8;
        c.prepare();
        const std::vector<uint8_t> before = c.dst;
        EXPECT_EQ(c.run(), status_t::unimplemented);
        EXPECT_EQ(c.dst, before);
    }
    const int N = 64;
    struct int8_case_t {
        const char *name;
        data_type_t src;
        dims_t src_dims, wei_dims; // a 0 row count stands for M
    };
    // Per-group weight scales the reference kernel applies per group only
    // with a source scale of the same groups; a bf16 source with s8 weights
    // needs a source scale.
    const int8_case_t int8_cases[] = {
            {"per-group weights, per-tensor source", data_type_t::s8, {1},
                    {4, N}},
            {"per-group weights, per-token source", data_type_t::s8, {0, 1},
                    {4, N}},
            {"bf16 source without a source scale", data_type_t::bf16, {},
                    {1, N}},
    };
    for (const int8_case_t &k : int8_cases)
        for (const matmul_algo_t algo : kAoclDlpRoutes) {
            SCOPED_TRACE(::testing::Message()
                    << k.name << " algo " << static_cast<int>(algo));
            int8_call_t c;
            c.src_dt = k.src;
            c.src_scale_dims = k.src_dims;
            if (!c.src_scale_dims.empty() && c.src_scale_dims[0] == 0)
                c.src_scale_dims[0] = c.M;
            c.wei_scale_dims = k.wei_dims;
            c.algo = algo;
            c.postops = {eltwise_op(P::elu)};
            c.prepare();
            mm::matmul_params p = c.params();
            if (k.src_dims.empty()) p.quant_params.src_scale = {};
            mm::matmul_batch_params_t batch;
            const native_brgemm_test::algo_pin_t unset(matmul_algo_t::none);
            EXPECT_EQ(mm::matmul_direct('r', false, false, c.M, c.N, c.K, 1.0f,
                              c.src.data(), c.K, c.weight.data(), c.N, nullptr,
                              0.0f, c.dst.data(), c.N, true, batch, p),
                    status_t::unimplemented);
            EXPECT_TRUE(c.dst_untouched());
        }
    {
        SCOPED_TRACE("AOCL-prepacked weights");
        int8_call_t c;
        c.wei_scale_dims = {1, N};
        c.postops = {eltwise_op(P::square)};
        c.prepare();
        mm::matmul_params p = c.params();
        p.mem_format_b = 'r';
        mm::matmul_batch_params_t batch;
        const native_brgemm_test::algo_pin_t unset(matmul_algo_t::none);
        EXPECT_EQ(mm::matmul_direct('r', false, false, c.M, c.N, c.K, 1.0f,
                          c.src.data(), c.K, c.weight.data(), c.N, nullptr,
                          0.0f, c.dst.data(), c.N, true, batch, p),
                status_t::unimplemented);
        EXPECT_TRUE(c.dst_untouched());
    }
    {
        SCOPED_TRACE("GGML packed weights");
        const int M = 4, K = 64;
        std::vector<int8_t> src(static_cast<size_t>(M) * K, 1);
        // Q8_0: per 32 values of a row of the [N, K] weight, an f16 scale
        // and 32 s8 values.
        std::vector<uint8_t> weight(static_cast<size_t>(N) * (K / 32) * 34, 0);
        std::vector<float> src_scale(static_cast<size_t>(M) * (K / 32), 0.5f);
        std::vector<float> dst(static_cast<size_t>(M) * N, 7.0f);
        const std::vector<float> before = dst;
        mm::matmul_params p;
        p.dtypes.src = data_type_t::s8;
        p.dtypes.wei = data_type_t::s8;
        p.dtypes.dst = data_type_t::f32;
        p.packing.pack_format_b = 1;
        p.quant_params.src_scale
                = quant(src_scale.data(), data_type_t::f32, {M, K / 32});
        p.postop_ = {eltwise_op(P::elu)};
        mm::matmul_batch_params_t batch;
        const native_brgemm_test::algo_pin_t unset(matmul_algo_t::none);
        EXPECT_EQ(mm::matmul_direct('r', false, true, M, N, K, 1.0f, src.data(),
                          K, weight.data(), K, nullptr, 0.0f, dst.data(), N,
                          true, batch, p),
                status_t::unimplemented);
        EXPECT_EQ(dst, before);
        mm::clear_matmul_weight_caches();
    }
    // A gated activation: pooling and the post-ops the AOCL-DLP chain does
    // not take have no defined place against the activation; relu, which it
    // applies, keeps the call served.
    const int GM = 4, GN = 128, GK = 64;
    const std::vector<uint16_t> ga(
            static_cast<size_t>(GM) * GK, to_bf16(0.25f));
    const std::vector<uint16_t> gw(
            static_cast<size_t>(GK) * GN, to_bf16(0.125f));
    for (const post_op_type_t t :
            {P::pooling, P::elu, P::abs, P::softmax, P::relu}) {
        SCOPED_TRACE(::testing::Message()
                << "gated, post-op " << static_cast<int>(t));
        std::vector<uint16_t> gd(static_cast<size_t>(GM) * GN, 0x3f80);
        const std::vector<uint16_t> gd_before = gd;
        mm::matmul_params gp;
        gp.dtypes.src = data_type_t::bf16;
        gp.dtypes.wei = data_type_t::bf16;
        gp.dtypes.dst = data_type_t::bf16;
        gp.postop_ = {eltwise_op(t)};
        mm::matmul_gated_act_params gated;
        gated.act = mm::matmul_gated_act_t::silu_and_mul;
        mm::matmul_batch_params_t one;
        const native_brgemm_test::algo_pin_t unset(matmul_algo_t::none);
        const status_t st = mm::matmul_direct('r', false, false, GM, GN, GK,
                1.0f, ga.data(), GK, gw.data(), GN, nullptr, 0.0f, gd.data(),
                GN, true, one, gp, &gated);
        if (t == P::relu) {
            EXPECT_EQ(st, status_t::success);
        } else {
            EXPECT_EQ(st, status_t::unimplemented);
            EXPECT_EQ(gd, gd_before);
        }
    }
}

// A softmax over rows of u8 x s8 sums in the thousands, whose exponentials
// overflow f32: the reference kernel shifts such a row by its maximum, on the
// AOCL-DLP route and with lowoha_algo reference.
TEST(AoclDlpUnmappedPostOps, SoftmaxOfLargeRowsIsFinite) {
    for (const data_type_t dst : {data_type_t::f32, data_type_t::bf16})
        for (const int M : {1, 4})
            for (const matmul_algo_t algo :
                    {matmul_algo_t::none, matmul_algo_t::aocl_dlp,
                            matmul_algo_t::aocl_dlp_blocked,
                            matmul_algo_t::reference}) {
                SCOPED_TRACE(::testing::Message()
                        << "dst " << static_cast<int>(dst) << " M " << M
                        << " algo " << static_cast<int>(algo));
                int8_call_t c;
                c.M = M;
                c.src_dt = data_type_t::u8;
                c.dst_dt = dst;
                c.wei_scale_dims = {1};
                c.algo = algo;
                c.postops = {eltwise_op(post_op_type_t::softmax)};
                c.prepare();
                const native_brgemm_test::algo_pin_t unset(matmul_algo_t::none);
                ASSERT_EQ(c.run(), status_t::success);
                c.expect_model();
            }
}

TEST(AoclDlpUnmappedPostOps, IntegerStoresSaturate) {
    // u8 x s8 with every source value 30 and every weight 7 or -7, unit
    // scales: the integer sum is +-53760, its square passes the s32 range,
    // the square times -1 passes it below, and the log of -53760 is NaN.
    // The store clamps to the destination's range, as the native BRGEMM
    // does (the largest float below 2^31 into s32), and stores a NaN as the
    // lowest value.
    using P = post_op_type_t;
    const int M = 2, N = 16, K = 256;
    const std::vector<float> minus_one(N, -1.0f);
    struct case_t {
        data_type_t dst;
        int weight;
        std::vector<P> ops;
        bool negate;
        int64_t want;
    };
    const case_t cases[] = {
            {data_type_t::s32, 7, {P::square}, false, 2147483520},
            {data_type_t::s32, 7, {P::exp}, false, 2147483520},
            {data_type_t::s32, 7, {P::square}, true, INT32_MIN},
            {data_type_t::s32, -7, {P::log}, false, INT32_MIN},
            {data_type_t::s8, 7, {P::square}, false, 127},
            {data_type_t::s8, 7, {P::square}, true, -128},
            {data_type_t::s8, -7, {P::log}, false, -128},
            {data_type_t::u8, 7, {P::square}, false, 255},
            {data_type_t::u8, 7, {P::square}, true, 0},
            {data_type_t::u8, -7, {P::log}, false, 0},
    };
    const float one = 1.0f;
    for (const case_t &k : cases)
        for (const matmul_algo_t algo : {matmul_algo_t::none,
                     matmul_algo_t::aocl_dlp, matmul_algo_t::aocl_dlp_blocked,
                     matmul_algo_t::reference}) {
            SCOPED_TRACE(::testing::Message()
                    << "dst " << static_cast<int>(k.dst) << " weight "
                    << k.weight << " op " << static_cast<int>(k.ops[0])
                    << " negate " << k.negate << " algo "
                    << static_cast<int>(algo));
            mm::clear_matmul_weight_caches();
            const std::vector<uint8_t> src(static_cast<size_t>(M) * K, 30);
            const std::vector<int8_t> weight(
                    static_cast<size_t>(K) * N, static_cast<int8_t>(k.weight));
            std::vector<uint8_t> dst(
                    static_cast<size_t>(M) * N * element_size(k.dst), 0x5a);
            mm::matmul_params p;
            p.dtypes.src = data_type_t::u8;
            p.dtypes.wei = data_type_t::s8;
            p.dtypes.dst = k.dst;
            p.lowoha_algo = algo;
            p.quant_params.src_scale = quant(&one, data_type_t::f32, {1});
            p.quant_params.wei_scale = quant(&one, data_type_t::f32, {1});
            for (const P t : k.ops)
                p.postop_.push_back(eltwise_op(t));
            if (k.negate) {
                mm::matmul_post_op po;
                po.po_type = P::binary_mul;
                po.buff = const_cast<float *>(minus_one.data());
                po.dtype = data_type_t::f32;
                po.dims = {1, N};
                p.postop_.push_back(po);
            }
            mm::matmul_batch_params_t batch;
            const native_brgemm_test::algo_pin_t unset(matmul_algo_t::none);
            ASSERT_EQ(mm::matmul_direct('r', false, false, M, N, K, 1.0f,
                              src.data(), K, weight.data(), N, nullptr, 0.0f,
                              dst.data(), N, true, batch, p),
                    status_t::success);
            for (int i = 0; i < M * N; ++i)
                ASSERT_EQ(static_cast<int64_t>(load(dst, k.dst, i)), k.want)
                        << "element " << i;
        }
}

namespace {

// Two bf16 operations of a group call, the second with post-op t; the status
// and whether both destinations kept their bytes, under the algorithm the
// process runs.
std::pair<status_t, bool> run_group_with_post_op(post_op_type_t t) {
    const int M = 4, N = 64, K = 64;
    const std::vector<uint16_t> a(static_cast<size_t>(M) * K, to_bf16(0.25f));
    const std::vector<uint16_t> w0(static_cast<size_t>(K) * N, to_bf16(0.125f));
    const std::vector<uint16_t> w1(static_cast<size_t>(K) * N, to_bf16(0.5f));
    std::vector<uint16_t> d0(static_cast<size_t>(M) * N, 0x3f80);
    std::vector<uint16_t> d1(static_cast<size_t>(M) * N, 0x3f80);
    const std::vector<uint16_t> before = d0;
    std::vector<mm::matmul_params> params(2);
    for (auto &p : params) {
        p.dtypes.src = data_type_t::bf16;
        p.dtypes.wei = data_type_t::bf16;
        p.dtypes.dst = data_type_t::bf16;
    }
    params[1].postop_ = {eltwise_op(t)};
    const std::vector<char> layout(2, 'r');
    const std::vector<bool> trans(2, false), consts(2, true);
    const std::vector<int> ms(2, M), ns(2, N), ks(2, K), lda(2, K), ldb(2, N);
    const std::vector<float> alpha(2, 1.0f), beta(2, 0.0f);
    const std::vector<const void *> src {a.data(), a.data()};
    const std::vector<const void *> weight {w0.data(), w1.data()};
    const std::vector<const void *> bias(2, nullptr);
    const std::vector<void *> dst {d0.data(), d1.data()};
    const status_t st = mm::group_matmul_direct(layout, trans, trans, ms, ns,
            ks, alpha, src, lda, weight, ldb, bias, beta, dst, ldb, consts,
            params);
    mm::clear_matmul_weight_caches();
    return {st, d0 == before && d1 == before};
}

} // namespace

// group_matmul_direct passes its operations to AOCL-DLP (but on oneDNN): an
// operation with a post-op the AOCL-DLP chain does not take fails the call
// before any operation writes; relu, which that chain applies, is served.
TEST(AoclDlpUnmappedPostOps, GroupOperationsOnAoclDlpFail) {
    using P = post_op_type_t;
    for (const P t : {P::elu, P::softmax, P::pooling, P::relu}) {
        SCOPED_TRACE(::testing::Message() << "post-op " << static_cast<int>(t));
        const native_brgemm_test::algo_pin_t unset(matmul_algo_t::none);
        const auto r = run_group_with_post_op(t);
        if (t == P::relu) {
            EXPECT_EQ(r.first, status_t::success);
        } else {
            EXPECT_EQ(r.first, status_t::unimplemented);
            EXPECT_TRUE(r.second) << "a destination was written";
        }
    }
#if ZENDNNL_DEPENDS_ONEDNN
    native_brgemm_test::expect_in_child({{"ZENDNNL_MATMUL_ALGO", "2"}}, [] {
        const auto r = run_group_with_post_op(post_op_type_t::elu);
        if (r.first == status_t::success) return 0;
        std::fprintf(stderr, "elu on oneDNN: status %d\n",
                static_cast<int>(r.first));
        return 1;
    }, "ZENDNNL_MATMUL_ALGO=2");
#endif
}

// ── Alpha and beta on the AOCL-DLP route ────────────────────────────────

namespace {

// An INT8 call with alpha and beta against matmul_direct's order: into an
// f32 or bf16 destination (alpha * S + Z) * src_scale * wei_scale + beta * C
// + bias, into an integer one ((alpha * S + beta * C + Z) * src_scale) *
// wei_scale + bias, S the integer sum of the source and weight values and
// Z = -zero_point * column sum; then the elementwise post-ops, the
// destination scale and zero point.  Every value is a small multiple of a
// power of two, so the sums are exact in f32.
struct alpha_beta_call_t {
    int M = 4, N = 64, K = 128;
    data_type_t src_dt = data_type_t::s8;
    data_type_t dst_dt = data_type_t::f32;
    bool src_zp = false; // a per-tensor source zero point
    bool per_token = false; // an {M, 1} source scale
    bool dynamic = false; // a bf16 source quantized in the call to s8
    float alpha = 1.0f, beta = 0.0f;
    matmul_algo_t algo = matmul_algo_t::none;
    std::vector<mm::matmul_post_op> postops;

    std::vector<int32_t> q; // the source values the kernel multiplies
    std::vector<int8_t> weight;
    std::vector<float> src_scale, wei_scale, bias;
    std::vector<double> c0; // the destination before the call
    std::vector<uint8_t> src, dst;
    int32_t zero_point = 0;
    float dst_scale = 0.5f;
    int32_t dst_zero_point = 3;

    bool integer_dst() const {
        return dst_dt == data_type_t::s32 || dst_dt == data_type_t::s8
                || dst_dt == data_type_t::u8;
    }
    bool quantized_dst() const {
        return dst_dt == data_type_t::s8 || dst_dt == data_type_t::u8;
    }
    // A bf16 / f32 source holds q * s_src, which its quantization, in the
    // call or in AOCL-DLP, turns into q.
    bool float_src() const {
        return src_dt == data_type_t::bf16 || src_dt == data_type_t::f32;
    }
    float s_src(int m) const {
        if (dynamic) return 1.0f / 64.0f;
        return per_token ? std::ldexp(1.0f, -(1 + m % 3)) : 0.5f;
    }

    // A new weight may reuse the address of a freed one, which keys the
    // weight and post-op caches.
    void prepare() {
        mm::clear_matmul_weight_caches();
        values_t r(static_cast<uint32_t>(M * 131 + N));
        weight.resize(static_cast<size_t>(K) * N);
        for (auto &v : weight)
            v = static_cast<int8_t>(r.next(-7, 7));
        wei_scale.resize(N);
        bias.resize(N);
        for (int n = 0; n < N; ++n) {
            wei_scale[n] = std::ldexp(static_cast<float>(1 + n % 3), -5);
            bias[n] = static_cast<float>(r.next(-8, 8)) / 16.0f;
        }
        zero_point = src_zp ? (src_dt == data_type_t::u8 ? 10 : 3) : 0;
        q.resize(static_cast<size_t>(M) * K);
        src.assign(q.size() * element_size(src_dt), 0);
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < K; ++k) {
                const size_t i = static_cast<size_t>(m) * K + k;
                // A quantized row holds 127 once, so the call's scale of
                // each row is 1 / 64.
                if (dynamic)
                    q[i] = k == 0 ? 127 : r.next(-100, 100);
                else
                    q[i] = src_dt == data_type_t::u8 ? r.next(0, 30)
                                                     : r.next(-9, 9);
                store(src, src_dt, i,
                        float_src() ? static_cast<float>(q[i]) * s_src(m)
                                    : static_cast<float>(q[i]));
            }
        src_scale.resize(per_token ? M : 1);
        for (size_t m = 0; m < src_scale.size(); ++m)
            src_scale[m] = s_src(static_cast<int>(m));
        c0.resize(static_cast<size_t>(M) * N);
        dst.assign(c0.size() * element_size(dst_dt), 0);
        for (size_t i = 0; i < c0.size(); ++i) {
            // A C of floats holds values between the integers.
            const int v = dst_dt == data_type_t::u8 ? r.next(0, 40)
                                                    : r.next(-40, 40);
            store(dst, dst_dt, i,
                    integer_dst() ? static_cast<float>(v)
                                  : static_cast<float>(v) / 16.0f + 0.03125f);
            c0[i] = load(dst, dst_dt, i);
        }
    }

    // Under the algorithm the process runs.
    status_t run_call() {
        mm::matmul_params p;
        p.dtypes.src = src_dt;
        p.dtypes.wei = data_type_t::s8;
        p.dtypes.dst = dst_dt;
        p.dtypes.bias = data_type_t::f32;
        p.lowoha_algo = algo;
        if (dynamic) {
            p.dynamic_quant = true;
            p.dtypes.compute = data_type_t::s8;
        }
        p.quant_params.src_scale = quant(src_scale.data(), data_type_t::f32,
                per_token ? dims_t {M, 1} : dims_t {1});
        p.quant_params.wei_scale
                = quant(wei_scale.data(), data_type_t::f32, {1, N});
        if (src_zp)
            p.quant_params.src_zp = quant(&zero_point, data_type_t::s32, {1});
        if (quantized_dst()) {
            p.quant_params.dst_scale = quant(&dst_scale, data_type_t::f32, {1});
            p.quant_params.dst_zp
                    = quant(&dst_zero_point, data_type_t::s32, {1});
        }
        p.postop_ = postops;
        mm::matmul_batch_params_t batch;
        return mm::matmul_direct('r', false, false, M, N, K, alpha, src.data(),
                K, weight.data(), N, bias.data(), beta, dst.data(), N, true,
                batch, p);
    }

    // The default route, whatever algorithm the process runs.
    status_t run() {
        const native_brgemm_test::algo_pin_t unset(matmul_algo_t::none);
        return run_call();
    }

    void expect_model() const {
        int bad = 0;
        for (int m = 0; m < M && bad < 8; ++m)
            for (int n = 0; n < N && bad < 8; ++n) {
                int64_t s = 0, column = 0;
                for (int k = 0; k < K; ++k) {
                    const int w = weight[static_cast<size_t>(k) * N + n];
                    s += static_cast<int64_t>(q[static_cast<size_t>(m) * K + k])
                            * w;
                    column += w;
                }
                const size_t i = static_cast<size_t>(m) * N + n;
                const double scale
                        = static_cast<double>(s_src(m)) * wei_scale[n];
                const double z = -static_cast<double>(zero_point) * column;
                double v = integer_dst()
                        ? (alpha * static_cast<double>(s) + beta * c0[i] + z)
                                        * scale
                                + bias[n]
                        : (alpha * static_cast<double>(s) + z) * scale
                                + beta * c0[i] + bias[n];
                for (const auto &po : postops)
                    v = eltwise(po, v);
                if (quantized_dst()) v = v * dst_scale + dst_zero_point;
                const double ref = in_destination(v, dst_dt);
                const double got = load(dst, dst_dt, i);
                const double tol = integer_dst() ? 1.0
                        : dst_dt == data_type_t::bf16
                        ? std::ldexp(std::fabs(ref), -7) + 1e-6
                        : 1e-5 * (1 + std::fabs(ref));
                if (!(std::fabs(got - ref) <= tol)) {
                    ADD_FAILURE() << "row " << m << " column " << n << ": "
                                  << got << ", model " << ref;
                    ++bad;
                }
            }
    }
};

const matmul_algo_t kAoclDlpAlgos[] = {matmul_algo_t::none,
        matmul_algo_t::aocl_dlp, matmul_algo_t::aocl_dlp_blocked};

// The source and destination of a bf16 / f32 source with s8 weights: an f32
// source writes f32 only.
const std::pair<data_type_t, data_type_t> kFloatSourceCalls[]
        = {{data_type_t::bf16, data_type_t::f32},
                {data_type_t::bf16, data_type_t::bf16},
                {data_type_t::f32, data_type_t::f32}};

// A call AOCL-DLP cannot apply fails and leaves dst as it was; the reference
// kernel serves it.
void expect_refused_on_aocl_dlp(alpha_beta_call_t c) {
    for (const matmul_algo_t algo : kAoclDlpAlgos) {
        SCOPED_TRACE(::testing::Message() << "algo " << static_cast<int>(algo));
        c.algo = algo;
        c.prepare();
        const std::vector<uint8_t> before = c.dst;
        EXPECT_EQ(c.run(), status_t::unimplemented);
        EXPECT_EQ(c.dst, before);
    }
    SCOPED_TRACE("reference");
    c.algo = matmul_algo_t::reference;
    c.prepare();
    ASSERT_EQ(c.run(), status_t::success);
    c.expect_model();
}

} // namespace

TEST(AoclDlpAlphaBeta, IntegerAlphaAndBetaAreApplied) {
    const data_type_t floats[] = {data_type_t::f32, data_type_t::bf16};
    const data_type_t integers[]
            = {data_type_t::s32, data_type_t::s8, data_type_t::u8};
    for (const matmul_algo_t algo : kAoclDlpAlgos)
        for (const data_type_t src : {data_type_t::s8, data_type_t::u8})
            for (const bool zp : {false, true})
                for (const int M : {1, 4, 33})
                    for (const float alpha : {1.0f, 2.0f, -3.0f}) {
                        alpha_beta_call_t c;
                        c.algo = algo;
                        c.src_dt = src;
                        c.src_zp = zp;
                        c.M = M;
                        c.alpha = alpha;
                        for (const data_type_t dst : floats) {
                            SCOPED_TRACE(::testing::Message()
                                    << "algo " << static_cast<int>(algo)
                                    << " src " << static_cast<int>(src)
                                    << " zp " << zp << " M " << M << " alpha "
                                    << alpha << " dst "
                                    << static_cast<int>(dst));
                            c.dst_dt = dst;
                            c.beta = 0.0f;
                            c.prepare();
                            ASSERT_EQ(c.run(), status_t::success);
                            c.expect_model();
                        }
                        for (const data_type_t dst : integers)
                            for (const float beta : {0.0f, 1.0f, 2.0f}) {
                                SCOPED_TRACE(::testing::Message()
                                        << "algo " << static_cast<int>(algo)
                                        << " src " << static_cast<int>(src)
                                        << " zp " << zp << " M " << M
                                        << " alpha " << alpha << " beta "
                                        << beta << " dst "
                                        << static_cast<int>(dst));
                                c.dst_dt = dst;
                                c.beta = beta;
                                c.prepare();
                                ASSERT_EQ(c.run(), status_t::success);
                                c.expect_model();
                            }
                    }
    // The sym-quant kernel, which adds beta * C after the scales: a per-token
    // s8 source, and a bf16 source the call quantizes per token, which
    // reaches AOCL-DLP as s8.
    for (const bool dynamic : {false, true})
        for (const data_type_t dst : {data_type_t::f32, data_type_t::bf16})
            for (const int M : {1, 4, 33})
                for (const float beta : {0.0f, 1.0f, 2.0f}) {
                    SCOPED_TRACE(::testing::Message()
                            << "dynamic " << dynamic << " dst "
                            << static_cast<int>(dst) << " M " << M << " beta "
                            << beta);
                    alpha_beta_call_t c;
                    c.src_dt = dynamic ? data_type_t::bf16 : data_type_t::s8;
                    c.dynamic = dynamic;
                    c.per_token = true;
                    c.dst_dt = dst;
                    c.M = M;
                    c.alpha = 2.0f;
                    c.beta = beta;
                    c.prepare();
                    ASSERT_EQ(c.run(), status_t::success);
                    c.expect_model();
                }
    // A bf16 / f32 source the call does not quantize, with beta 0.
    for (const auto &sd : kFloatSourceCalls)
        for (const bool per_token : {false, true})
            for (const int M : {1, 4}) {
                SCOPED_TRACE(::testing::Message()
                        << "src " << static_cast<int>(sd.first) << " dst "
                        << static_cast<int>(sd.second) << " per token "
                        << per_token << " M " << M);
                alpha_beta_call_t c;
                c.src_dt = sd.first;
                c.dst_dt = sd.second;
                c.per_token = per_token;
                c.M = M;
                c.prepare();
                ASSERT_EQ(c.run(), status_t::success);
                c.expect_model();
            }
}

TEST(AoclDlpAlphaBeta, OtherAlphaAndBetaFail) {
    // An s8 x s8 call into f32 at M 1: alpha 0.5 and beta 0.5 left only the
    // bias, alpha 2 and beta 1 rounded C to an integer.
    for (const auto &ab : {std::pair<float, float> {0.5f, 0.5f},
                 std::pair<float, float> {2.0f, 1.0f}}) {
        SCOPED_TRACE(::testing::Message()
                << "alpha " << ab.first << " beta " << ab.second);
        alpha_beta_call_t c;
        c.M = 1;
        c.alpha = ab.first;
        c.beta = ab.second;
        expect_refused_on_aocl_dlp(c);
    }
    const data_type_t dsts[] = {data_type_t::f32, data_type_t::bf16,
            data_type_t::s32, data_type_t::s8, data_type_t::u8};
    for (const data_type_t src : {data_type_t::s8, data_type_t::u8})
        for (const data_type_t dst : dsts)
            for (const int M : {1, 4, 33}) {
                alpha_beta_call_t c;
                c.src_dt = src;
                c.src_zp = src == data_type_t::u8;
                c.dst_dt = dst;
                c.M = M;
                for (const auto &ab : {std::pair<float, float> {0.5f, 0.0f},
                             std::pair<float, float> {-1.5f, 0.0f},
                             std::pair<float, float> {1.0f, 0.5f},
                             std::pair<float, float> {1.0f, 1.0f}}) {
                    // An integer beta applies to an integer C.
                    if (ab.second == 1.0f && c.integer_dst()) continue;
                    SCOPED_TRACE(::testing::Message()
                            << "src " << static_cast<int>(src) << " dst "
                            << static_cast<int>(dst) << " M " << M << " alpha "
                            << ab.first << " beta " << ab.second);
                    c.alpha = ab.first;
                    c.beta = ab.second;
                    expect_refused_on_aocl_dlp(c);
                }
            }
    // The sym-quant kernel takes integers only, also for a bf16 source the
    // call quantizes per token, which reaches AOCL-DLP as s8.  Quantized with
    // one scale value, that source takes the s8 rules: C rounded to an
    // integer at M 4, beta after the scales on the sym-quant kernel at M 1.
    for (const bool dynamic : {false, true})
        for (const auto &ab : {std::pair<float, float> {0.5f, 0.0f},
                     std::pair<float, float> {1.0f, 0.5f}}) {
            SCOPED_TRACE(::testing::Message()
                    << "dynamic " << dynamic << " alpha " << ab.first
                    << " beta " << ab.second);
            alpha_beta_call_t c;
            c.src_dt = dynamic ? data_type_t::bf16 : data_type_t::s8;
            c.dynamic = dynamic;
            c.per_token = true;
            c.alpha = ab.first;
            c.beta = ab.second;
            expect_refused_on_aocl_dlp(c);
        }
    for (const int M : {1, 4}) {
        SCOPED_TRACE(::testing::Message() << "one source scale value, M " << M);
        alpha_beta_call_t c;
        c.src_dt = data_type_t::bf16;
        c.dynamic = true;
        c.M = M;
        c.beta = 1.0f;
        expect_refused_on_aocl_dlp(c);
    }
    // A bf16 / f32 source the call does not quantize: the GEMMs of a float
    // source and s8 weights take no beta, with one source scale value or one
    // per row ({1, 1} at M 1).
    for (const auto &sd : kFloatSourceCalls)
        for (const bool per_token : {false, true})
            for (const int M : {1, 4}) {
                SCOPED_TRACE(::testing::Message()
                        << "src " << static_cast<int>(sd.first) << " dst "
                        << static_cast<int>(sd.second) << " per token "
                        << per_token << " M " << M);
                alpha_beta_call_t c;
                c.src_dt = sd.first;
                c.dst_dt = sd.second;
                c.per_token = per_token;
                c.M = M;
                c.beta = 1.0f;
                expect_refused_on_aocl_dlp(c);
            }
}

// ── Alpha and beta with post-ops after the kernel ───────────────────────

// Under ZENDNNL_MATMUL_ALGO=8 the AOCL-DLP rules apply to the calls that run
// on AOCL-DLP: with oneDNN candidates only, every call runs on oneDNN with
// the bits of oneDNN blocked; with AOCL-DLP among them, a call it cannot
// apply runs on the oneDNN candidate; with AOCL-DLP alone it fails with dst
// untouched.
TEST(AoclDlpAlphaBeta, AutoTunerRunsElsewhereWhatAoclDlpRefuses) {
#if !ZENDNNL_DEPENDS_ONEDNN
    GTEST_SKIP() << "the oneDNN candidates run on oneDNN";
#endif
    for (const char *candidates : {"2", "1,2", "1"}) {
        native_brgemm_test::expect_in_child(
                {{"ZENDNNL_MATMUL_ALGO", "8"},
                        {"ZENDNNL_MATMUL_AUTO_ALGO_CANDIDATES", candidates}},
                [candidates] {
            const bool onednn_only = std::strcmp(candidates, "2") == 0;
            const bool aocl_dlp_only = std::strcmp(candidates, "1") == 0;
            struct case_t {
                data_type_t src;
                float alpha, beta;
                data_type_t dst;
                bool refused;
            };
            const data_type_t u8 = data_type_t::u8, f32 = data_type_t::f32;
            const case_t cases[] = {{u8, 1.0f, 1.0f, f32, true},
                    {data_type_t::s8, 1.0f, 0.5f, f32, true},
                    {u8, 0.5f, 0.0f, f32, true},
                    {u8, 1.0f, 1.0f, data_type_t::bf16, true},
                    {u8, 1.0f, 0.0f, f32, false}};
            int failures = 0;
            for (const case_t &k : cases) {
                if (!k.refused && !onednn_only) continue;
                alpha_beta_call_t c;
                c.src_dt = k.src;
                c.dst_dt = k.dst;
                c.alpha = k.alpha;
                c.beta = k.beta;
                c.prepare();
                const std::vector<uint8_t> before = c.dst;
                const status_t st = c.run_call();
                if (aocl_dlp_only) {
                    if (st != status_t::unimplemented || c.dst != before) {
                        std::fprintf(stderr, "alpha %g beta %g: served\n",
                                k.alpha, k.beta);
                        ++failures;
                    }
                    continue;
                }
                alpha_beta_call_t ref = c;
                ref.algo = matmul_algo_t::onednn_blocked;
                ref.prepare();
                if (st != status_t::success
                        || ref.run_call() != status_t::success
                        || c.dst != ref.dst) {
                    std::fprintf(stderr,
                            "alpha %g beta %g dst %d: status %d, not oneDNN "
                            "blocked's bits\n",
                            k.alpha, k.beta, static_cast<int>(k.dst),
                            static_cast<int>(st));
                    ++failures;
                }
            }
            return failures;
        }, std::string("ZENDNNL_MATMUL_AUTO_ALGO_CANDIDATES=") + candidates);
    }
}

TEST(AoclDlpUnmappedPostOps, Int8BetaIntoIntegerDestinations) {
    // A call into an integer destination with a post-op AOCL-DLP does not map
    // runs on the reference kernel, which adds beta * C to the integer sum:
    // also at M 1 with one source scale value and {1, N} weight scales, the
    // scales of the sym-quant kernel into f32.
    using P = post_op_type_t;
    const std::vector<P> chains[] = {{P::abs}, {P::relu, P::sqrt}, {P::elu}};
    for (const matmul_algo_t algo : kAoclDlpAlgos)
        for (const data_type_t src : {data_type_t::s8, data_type_t::u8})
            for (const data_type_t dst :
                    {data_type_t::s32, data_type_t::s8, data_type_t::u8})
                for (const int M : {1, 4})
                    for (const float beta : {1.0f, 2.0f})
                        for (const auto &chain : chains) {
                            SCOPED_TRACE(::testing::Message()
                                    << "algo " << static_cast<int>(algo)
                                    << " src " << static_cast<int>(src)
                                    << " dst " << static_cast<int>(dst) << " M "
                                    << M << " beta " << beta << " ops "
                                    << chain.size() << " first "
                                    << static_cast<int>(chain[0]));
                            alpha_beta_call_t c;
                            c.algo = algo;
                            c.src_dt = src;
                            c.src_zp = src == data_type_t::u8;
                            c.dst_dt = dst;
                            c.M = M;
                            c.alpha = 2.0f;
                            c.beta = beta;
                            for (const P t : chain)
                                c.postops.push_back(eltwise_op(t));
                            c.prepare();
                            ASSERT_EQ(c.run(), status_t::success);
                            c.expect_model();
                        }
}
