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

// Direct calls of the native-layout epilogue on synthetic accumulators
// against a scalar model of the reference algorithm (reference_kernel.cpp)
// and of the gated activations (group_matmul_moe_act.cpp).

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <omp.h>

#include "common/bfloat16.hpp"
#include "common/data_types.hpp"
#include "common/float16.hpp"
#include "common/zendnnl_compat.hpp"
#include "common/zendnnl_global.hpp"
#include "lowoha_operators/matmul/lowoha_matmul.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_brgemm_looper.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_epilogue.hpp"
#include "lowoha_operators/matmul/matmul_native/native_matmul.hpp"
#include "matmul_native/native_brgemm_test_utils.hpp"
#include "memory/memory_utils.hpp"

namespace {

namespace mm = zendnnl::lowoha::matmul;
namespace nat = zendnnl::lowoha::matmul::native;
namespace nt = native_brgemm_test;
using zendnnl::common::bfloat16_t;
using zendnnl::common::data_type_t;
using zendnnl::common::float16_t;
using zendnnl::common::post_op_type_t;
using zendnnl::common::size_of;
using zendnnl::error_handling::status_t;
using zendnnl::memory::read_and_cast;
using gated_t = mm::matmul_gated_act_t;

constexpr float kNan = std::numeric_limits<float>::quiet_NaN();

bool cpu_ok() {
    static const bool ok = [] {
        auto &info = zendnnl::common::zendnnl_platform_info();
        return info.get_avx512_bw_vl_status() && info.get_avx512_vnni_status();
    }();
    return ok;
}

#define SKIP_WITHOUT_AVX512() \
    if (!cpu_ok()) GTEST_SKIP() << "needs AVX512-BW/VL/VNNI"

/// A typed buffer of count elements.
struct buf_t {
    data_type_t dt = data_type_t::f32;
    std::vector<uint8_t> bytes;

    void init(data_type_t d, int64_t n) {
        dt = d;
        bytes.assign(
                static_cast<size_t>(std::max<int64_t>(n, 1)) * size_of(d), 0);
    }
    void *data() { return bytes.data(); }
    const void *data() const { return bytes.data(); }
    float get(int64_t i) const {
        return read_and_cast<float>(bytes.data(), dt, i);
    }
    void set(int64_t i, float v) {
        uint8_t *p = bytes.data() + i * size_of(dt);
        switch (dt) {
            case data_type_t::f32: std::memcpy(p, &v, 4); break;
            case data_type_t::bf16: {
                const bfloat16_t h(v);
                std::memcpy(p, &h, 2);
                break;
            }
            case data_type_t::f16: {
                const float16_t h(v);
                std::memcpy(p, &h, 2);
                break;
            }
            case data_type_t::s8:
                *reinterpret_cast<int8_t *>(p) = static_cast<int8_t>(
                        std::lrint(std::min(127.0f, std::max(-128.0f, v))));
                break;
            case data_type_t::u8:
                *p = static_cast<uint8_t>(
                        std::lrint(std::min(255.0f, std::max(0.0f, v))));
                break;
            case data_type_t::s32: {
                const int32_t q = static_cast<int32_t>(std::lrint(v));
                std::memcpy(p, &q, 4);
                break;
            }
            default: break;
        }
    }
    void fill(std::mt19937 &rng, float lo, float hi) {
        std::uniform_real_distribution<float> d(lo, hi);
        const int64_t n = static_cast<int64_t>(bytes.size() / size_of(dt));
        for (int64_t i = 0; i < n; ++i)
            set(i, d(rng));
    }
};

/// One post-op of a test problem; alpha NaN keeps the post-op default.
struct po_spec_t {
    post_op_type_t type = post_op_type_t::none;
    float alpha = kNan;
    float beta = 0.0f;
    data_type_t dt = data_type_t::f32;
    std::vector<int64_t> dims;
    int ld = -1;
};

po_spec_t eltwise(post_op_type_t t, float alpha = kNan, float beta = 0.0f) {
    po_spec_t s;
    s.type = t;
    s.alpha = alpha;
    s.beta = beta;
    return s;
}

po_spec_t binary(post_op_type_t t, data_type_t dt, std::vector<int64_t> dims,
        int ld = -1) {
    po_spec_t s;
    s.type = t;
    s.dt = dt;
    s.dims = std::move(dims);
    s.ld = ld;
    return s;
}

struct problem_t {
    int M = 7;
    int N = 64; ///< weight (pack) columns
    gated_t gated = gated_t::none;
    bool interleaved = false;
    float alpha = 1.0f;
    float beta = 0.0f;
    data_type_t bias_dt = data_type_t::none;
    data_type_t dst_dt = data_type_t::f32;
    bool int8 = false;
    int ldc_pad = 0;
    std::vector<po_spec_t> po;
    data_type_t dst_scale_dt = data_type_t::none;
    int64_t dst_scale_count = 1;
    data_type_t dst_zp_dt = data_type_t::none;
    int64_t dst_zp_count = 1;
    int tile_m = 6;
    int tile_n = 32;
    int acc_pad = 0;
    bool threaded = false;
    float acc_lo = -3.0f;
    float acc_hi = 3.0f;
    uint32_t seed = 1;

    int n_out() const { return gated == gated_t::none ? N : N / 2; }
    int ldc() const { return n_out() + ldc_pad; }
    int logical(int p) const {
        return !interleaved ? p : (p & 1) ? N / 2 + p / 2 : p / 2;
    }
    std::string str() const {
        std::ostringstream os;
        os << "M=" << M << " N=" << N << " gated=" << static_cast<int>(gated)
           << " interleaved=" << interleaved << " alpha=" << alpha
           << " beta=" << beta
           << " bias=" << zendnnl::common::dtype_info(bias_dt)
           << " dst=" << zendnnl::common::dtype_info(dst_dt) << " int8=" << int8
           << " tile=" << tile_m << "x" << tile_n << " po=";
        for (const auto &s : po)
            os << static_cast<int>(s.type) << ",";
        return os.str();
    }
};

// ---- scalar model of the reference algorithm ------------------------------

constexpr float kSqrt2OverPi = 0.79788458347320556640625f;
constexpr float kSqrt2Over2 = 0.707106769084930419921875f;
constexpr float kFittingConst = 0.044715f;

float ref_eltwise(post_op_type_t t, float x, float alpha, float beta) {
    switch (t) {
        case post_op_type_t::elu: return x > 0 ? x : alpha * (expf(x) - 1.0f);
        case post_op_type_t::relu: return x > 0 ? x : 0.0f;
        case post_op_type_t::leaky_relu: return x > 0 ? x : alpha * x;
        case post_op_type_t::gelu_tanh: {
            const float v
                    = tanhf(kSqrt2OverPi * x * (1.0f + kFittingConst * x * x));
            return 0.5 * x * (1.0f + v);
        }
        case post_op_type_t::gelu_erf:
            return 0.5f * x * (1.0f + erff(x * kSqrt2Over2));
        case post_op_type_t::sigmoid: return 1.0 / (1.0 + expf(-x));
        case post_op_type_t::swish: return x * (1.0 / (1.0 + expf(-x * alpha)));
        case post_op_type_t::tanh: return tanhf(x);
        case post_op_type_t::square: return x * x;
        case post_op_type_t::abs: return x > 0 ? x : -x;
        case post_op_type_t::sqrt: return sqrtf(x);
        case post_op_type_t::exp: return expf(x);
        case post_op_type_t::log: return logf(x);
        case post_op_type_t::clip:
            x = x > alpha ? x : alpha;
            return x > beta ? beta : x;
        case post_op_type_t::mish: {
            const float sp
                    = std::fmax(x, 0.0f) + std::log1p(std::exp(-std::fabs(x)));
            return x * std::tanh(sp);
        }
        default: return x;
    }
}

float ref_gated(gated_t g, float gate, float up) {
    switch (g) {
        case gated_t::silu_and_mul:
            return gate * (1.0f / (1.0f + std::exp(-gate))) * up;
        case gated_t::gelu_and_mul:
            return gate * 0.5f * (1.0f + std::erf(gate * 0.7071067811865476f))
                    * up;
        default: {
            const float gc = std::max(-7.0f, std::min(gate, 7.0f));
            const float uc = std::max(-7.0f, std::min(up, 7.0f));
            return (1.0f + uc) * gc * (1.0f / (1.0f + std::exp(-gc * 1.702f)));
        }
    }
}

int64_t count_of(const std::vector<int64_t> &dims) {
    int64_t n = 1;
    for (int64_t d : dims)
        n *= d;
    return n;
}

/// Materializes the post-ops of a [M, n_out] output, binary buffers included.
void make_postops(const std::vector<po_spec_t> &specs, int M, int n_out,
        std::mt19937 &rng, std::vector<buf_t> &bin,
        std::vector<mm::matmul_post_op> &postops) {
    bin.resize(specs.size());
    for (size_t i = 0; i < specs.size(); ++i) {
        const po_spec_t &s = specs[i];
        mm::matmul_post_op po;
        po.po_type = s.type;
        po.alpha = s.alpha;
        po.beta = s.beta;
        if (s.type == post_op_type_t::binary_add
                || s.type == post_op_type_t::binary_mul) {
            const int64_t count = count_of(s.dims);
            const int64_t ld = s.ld > 0 ? s.ld : n_out;
            const int64_t extent = count == static_cast<int64_t>(M) * n_out
                    ? static_cast<int64_t>(M - 1) * ld + n_out
                    : count;
            bin[i].init(s.dt, std::max(count, extent));
            bin[i].fill(rng, -2.0f, 2.0f);
            po.buff = bin[i].data();
            po.dtype = s.dt;
            po.dims = s.dims;
            po.leading_dim = s.ld;
        }
        postops.push_back(po);
    }
}

/// The reference algorithm's post-ops over out [M][n_out], in order.
void ref_postops(const std::vector<po_spec_t> &specs,
        const std::vector<mm::matmul_post_op> &postops,
        const std::vector<buf_t> &bin, int M, int no, std::vector<float> &out) {
    for (size_t i = 0; i < specs.size(); ++i) {
        const po_spec_t &s = specs[i];
        if (s.type == post_op_type_t::none) continue;
        if (s.type == post_op_type_t::softmax) {
            for (int m = 0; m < M; ++m) {
                float *row = out.data() + static_cast<size_t>(m) * no;
                double sum = 0.0;
                for (int o = 0; o < no; ++o)
                    sum += expf(row[o]);
                for (int o = 0; o < no; ++o)
                    row[o] = expf(row[o]) / sum;
            }
            continue;
        }
        const bool binary = s.type == post_op_type_t::binary_add
                || s.type == post_op_type_t::binary_mul;
        const int64_t count = count_of(s.dims);
        const int64_t ld = s.ld > 0 ? s.ld : no;
        for (int m = 0; m < M; ++m)
            for (int o = 0; o < no; ++o) {
                float &v = out[static_cast<size_t>(m) * no + o];
                if (!binary) {
                    v = ref_eltwise(
                            s.type, v, postops[i].alpha_or_default(), s.beta);
                    continue;
                }
                const int64_t at = count == static_cast<int64_t>(M) * no
                        ? m * ld + o
                        : (static_cast<int64_t>(m) * no + o) % count;
                const float b = bin[i].get(at);
                v = s.type == post_op_type_t::binary_add ? v + b : v * b;
            }
    }
}

/// The materialized inputs of a problem.
struct inputs_t {
    std::vector<float> acc; ///< [M][N], pack column order
    buf_t bias; ///< [N], weight column order
    buf_t dst0; ///< [M][ldc], dst before the call
    std::vector<buf_t> bin;
    buf_t dst_scale, dst_zp;
    std::vector<mm::matmul_post_op> postops;
    mm::matmul_quantization_params_t quant;
};

inputs_t make_inputs(const problem_t &pb) {
    std::mt19937 rng(pb.seed);
    inputs_t in;
    in.acc.resize(static_cast<size_t>(pb.M) * pb.N);
    std::uniform_real_distribution<float> da(pb.acc_lo, pb.acc_hi);
    for (float &v : in.acc)
        v = da(rng);
    if (pb.bias_dt != data_type_t::none) {
        in.bias.init(pb.bias_dt, pb.N);
        in.bias.fill(rng, -1.0f, 1.0f);
    }
    in.dst0.init(pb.dst_dt, static_cast<int64_t>(pb.M) * pb.ldc());
    in.dst0.fill(rng, -2.0f, 2.0f);
    // Past the columns the call reads: beta of a gated call reads N.
    const int read
            = pb.gated != gated_t::none && pb.beta != 0.0f ? pb.N : pb.n_out();
    for (int m = 0; m < pb.M; ++m)
        for (int j = read; j < pb.ldc(); ++j)
            in.dst0.set(static_cast<int64_t>(m) * pb.ldc() + j, 99.0f);

    make_postops(pb.po, pb.M, pb.gated == gated_t::none ? pb.n_out() : pb.N,
            rng, in.bin, in.postops);
    if (pb.dst_scale_dt != data_type_t::none) {
        in.dst_scale.init(pb.dst_scale_dt, pb.dst_scale_count);
        in.dst_scale.fill(rng, 4.0f, 16.0f);
        in.quant.dst_scale.buff = in.dst_scale.data();
        in.quant.dst_scale.dt = pb.dst_scale_dt;
        in.quant.dst_scale.dims = {pb.dst_scale_count};
    }
    if (pb.dst_zp_dt != data_type_t::none) {
        in.dst_zp.init(pb.dst_zp_dt, pb.dst_zp_count);
        in.dst_zp.fill(rng, 0.0f, 30.0f);
        in.quant.dst_zp.buff = in.dst_zp.data();
        in.quant.dst_zp.dt = pb.dst_zp_dt;
        in.quant.dst_zp.dims = {pb.dst_zp_count};
    }
    return in;
}

/// The f32 result before the store, [M][n_out].  A gated call takes beta,
/// the bias and the post-ops on its [M][N] pre-activation, in weight column
/// order, before the activation.
std::vector<float> reference(const problem_t &pb, const inputs_t &in) {
    const int M = pb.M, N = pb.N, no = pb.n_out();
    const bool has_bias = pb.bias_dt != data_type_t::none;
    std::vector<float> out(static_cast<size_t>(M) * no);
    // Pack column p of row m, before the post-ops.
    auto pre = [&](int m, int p) {
        float v = in.acc[static_cast<size_t>(m) * N + p];
        if (pb.alpha != 1.0f) v *= pb.alpha;
        if (pb.beta != 0.0f)
            v += in.dst0.get(static_cast<int64_t>(m) * pb.ldc() + pb.logical(p))
                    * pb.beta;
        if (has_bias) v += in.bias.get(pb.logical(p));
        return v;
    };
    if (pb.gated == gated_t::none) {
        for (int m = 0; m < M; ++m)
            for (int o = 0; o < no; ++o)
                out[static_cast<size_t>(m) * no + o] = pre(m, o);
        ref_postops(pb.po, in.postops, in.bin, M, no, out);
    } else {
        std::vector<float> wide(static_cast<size_t>(M) * N);
        for (int m = 0; m < M; ++m)
            for (int p = 0; p < N; ++p)
                wide[static_cast<size_t>(m) * N + pb.logical(p)] = pre(m, p);
        ref_postops(pb.po, in.postops, in.bin, M, N, wide);
        for (int m = 0; m < M; ++m)
            for (int o = 0; o < no; ++o) {
                const float *row = wide.data() + static_cast<size_t>(m) * N;
                out[static_cast<size_t>(m) * no + o] = pb.interleaved
                        ? ref_gated(pb.gated, row[o], row[N / 2 + o])
                        : ref_gated(pb.gated, row[2 * o], row[2 * o + 1]);
            }
    }
    if (pb.int8 && pb.dst_scale_dt != data_type_t::none) {
        for (size_t i = 0; i < out.size(); ++i) {
            out[i] *= in.dst_scale.get(i % pb.dst_scale_count);
            if (pb.dst_zp_dt != data_type_t::none)
                out[i] += static_cast<float>(read_and_cast<int32_t>(
                        in.dst_zp.data(), pb.dst_zp_dt, i % pb.dst_zp_count));
        }
    }
    return out;
}

float stored(data_type_t dt, float v) {
    switch (dt) {
        case data_type_t::u8:
            return std::nearbyint(std::min(255.0f, std::max(0.0f, v)));
        case data_type_t::s8:
            return std::nearbyint(std::min(127.0f, std::max(-128.0f, v)));
        case data_type_t::s32:
            return std::nearbyint(
                    std::min(2147483520.0f, std::max(-2147483648.0f, v)));
        default: return v;
    }
}

bool within_tol(data_type_t dt, float out, float ref, float tol) {
    if (out == ref) return true;
    if (std::isnan(ref)) return std::isnan(out);
    const float err = std::fabs(out - ref);
    switch (dt) {
        case data_type_t::f32: return err <= tol + tol * std::fabs(ref);
        case data_type_t::bf16:
            return err <= tol + std::fabs(ref) * (1.0f / 128.0f);
        case data_type_t::f16:
            return err <= tol + std::fabs(ref) * (1.0f / 1024.0f);
        default: return std::fabs(out - stored(dt, ref)) <= 1.0f;
    }
}

mm::matmul_params make_params(const problem_t &pb, const inputs_t &in) {
    mm::matmul_params p;
    p.dtypes.src = pb.int8 ? data_type_t::u8 : data_type_t::bf16;
    p.dtypes.wei = pb.int8 ? data_type_t::s8 : data_type_t::bf16;
    p.dtypes.dst = pb.dst_dt;
    p.dtypes.bias = pb.bias_dt;
    p.postop_ = in.postops;
    p.quant_params = in.quant;
    return p;
}

nat::GemmDescriptor make_desc(const problem_t &pb) {
    nat::GemmDescriptor d;
    d.M = pb.M;
    d.N = pb.N;
    d.K = 64;
    d.lda = 64;
    d.ldb = pb.N;
    d.ldc = pb.ldc();
    d.alpha = pb.alpha;
    d.beta = pb.beta;
    d.src_dt = pb.int8 ? data_type_t::u8 : data_type_t::bf16;
    d.wei_dt = pb.int8 ? data_type_t::s8 : data_type_t::bf16;
    d.dst_dt = pb.dst_dt;
    d.bias_dt = pb.bias_dt;
    d.gated_act = pb.gated;
    return d;
}

/// Applies the prepared epilogue to every tile of the problem, in a shuffled
/// order (or in parallel), from accumulators copied into tile scratch.
void run_tiles(const problem_t &pb, const nat::native_epilogue_t &ep,
        const std::vector<float> &acc) {
    struct tile_t {
        int m0, mr, n0;
    };
    std::vector<tile_t> tiles;
    for (int m0 = 0; m0 < pb.M; m0 += pb.tile_m)
        for (int n0 = 0; n0 < pb.N; n0 += pb.tile_n)
            tiles.push_back({m0, std::min(pb.tile_m, pb.M - m0), n0});
    std::mt19937 rng(pb.seed + 7);
    std::shuffle(tiles.begin(), tiles.end(), rng);
    const int ldc_acc = pb.tile_n + pb.acc_pad;
    auto one = [&](const tile_t &t, std::vector<float> &c) {
        c.assign(static_cast<size_t>(t.mr) * ldc_acc, -1234.0f);
        for (int i = 0; i < t.mr; ++i)
            std::memcpy(c.data() + static_cast<size_t>(i) * ldc_acc,
                    acc.data() + static_cast<size_t>(t.m0 + i) * pb.N + t.n0,
                    pb.tile_n * sizeof(float));
        nat::apply_epilogue_tile(
                ep, c.data(), ldc_acc, t.m0, t.mr, t.n0, pb.tile_n);
    };
    if (pb.threaded) {
#pragma omp parallel num_threads(4)
        {
            std::vector<float> c;
#pragma omp for schedule(dynamic, 1)
            for (int64_t t = 0; t < static_cast<int64_t>(tiles.size()); ++t)
                one(tiles[t], c);
        }
    } else {
        std::vector<float> c;
        for (const tile_t &t : tiles)
            one(t, c);
    }
}

/// Runs the problem through the epilogue and checks it against the model;
/// tol scales the f32 bound.
void check(const problem_t &pb, float tol = 3e-4f) {
    SCOPED_TRACE(pb.str());
    inputs_t in = make_inputs(pb);
    const std::vector<float> ref = reference(pb, in);
    buf_t dst = in.dst0;
    const mm::matmul_params params = make_params(pb, in);
    nat::native_epilogue_t ep;
    const status_t st = nat::prepare_epilogue(make_desc(pb), params,
            pb.bias_dt == data_type_t::none ? nullptr : in.bias.data(),
            dst.data(), pb.interleaved, ep);
    ASSERT_EQ(st, status_t::success);
    // The tiles of a swiglu call store over pre-activation columns other
    // tiles read for beta: beta reads a copy, as in native_brgemm_execute.
    const buf_t beta_copy = in.dst0;
    if (pb.gated == gated_t::swiglu_oai_mul && pb.beta != 0.0f)
        ep.beta_src = beta_copy.data();
    run_tiles(pb, ep, in.acc);

    int bad = 0;
    for (int m = 0; m < pb.M; ++m) {
        for (int o = 0; o < pb.ldc(); ++o) {
            const int64_t at = static_cast<int64_t>(m) * pb.ldc() + o;
            const float out = dst.get(at);
            if (o >= pb.n_out()) {
                if (out != in.dst0.get(at) && bad++ < 4)
                    ADD_FAILURE() << "padding (" << m << "," << o
                                  << ") written: " << out;
                continue;
            }
            const float r = ref[static_cast<size_t>(m) * pb.n_out() + o];
            if (!within_tol(pb.dst_dt, out, r, tol) && bad++ < 8)
                ADD_FAILURE() << "(" << m << "," << o << ") out " << out
                              << " ref " << r;
        }
    }
    EXPECT_EQ(bad, 0);
}

const std::vector<post_op_type_t> kEltwise = {post_op_type_t::relu,
        post_op_type_t::leaky_relu, post_op_type_t::elu,
        post_op_type_t::gelu_tanh, post_op_type_t::gelu_erf,
        post_op_type_t::sigmoid, post_op_type_t::swish, post_op_type_t::tanh,
        post_op_type_t::square, post_op_type_t::abs, post_op_type_t::sqrt,
        post_op_type_t::exp, post_op_type_t::log, post_op_type_t::clip,
        post_op_type_t::mish};

bool positive_domain(post_op_type_t t) {
    return t == post_op_type_t::sqrt || t == post_op_type_t::log;
}

// ---- tests -----------------------------------------------------------------

TEST(NativeBrgemmEpilogue, EveryEltwiseDefaultAlpha) {
    SKIP_WITHOUT_AVX512();
    for (post_op_type_t t : kEltwise)
        for (data_type_t dst : {data_type_t::f32, data_type_t::bf16}) {
            problem_t pb;
            pb.dst_dt = dst;
            pb.bias_dt = data_type_t::f32;
            if (positive_domain(t)) {
                pb.acc_lo = 0.05f;
                pb.bias_dt = data_type_t::none;
            }
            pb.po = {t == post_op_type_t::clip ? eltwise(t, -0.5f, 0.5f)
                                               : eltwise(t)};
            check(pb);
        }
}

TEST(NativeBrgemmEpilogue, EveryEltwiseExplicitAlpha) {
    SKIP_WITHOUT_AVX512();
    for (post_op_type_t t : kEltwise)
        for (float a : {0.0f, 0.25f, 1.5f, -0.75f}) {
            problem_t pb;
            pb.M = 5;
            pb.alpha = 1.25f;
            pb.bias_dt = data_type_t::bf16;
            if (positive_domain(t)) {
                pb.acc_lo = 0.05f;
                pb.bias_dt = data_type_t::none;
            }
            pb.po = {t == post_op_type_t::clip ? eltwise(t, -std::fabs(a), 1.0f)
                                               : eltwise(t, a, 0.5f)};
            pb.seed = 11;
            check(pb);
        }
}

TEST(NativeBrgemmEpilogue, PostOpChains) {
    SKIP_WITHOUT_AVX512();
    const int M = 9, N = 64;
    const std::vector<std::vector<po_spec_t>> chains = {
            {eltwise(post_op_type_t::gelu_erf),
                    binary(post_op_type_t::binary_add, data_type_t::f32,
                            {M, N})},
            {binary(post_op_type_t::binary_mul, data_type_t::bf16, {1, N}),
                    eltwise(post_op_type_t::relu),
                    eltwise(post_op_type_t::clip, -0.25f, 0.75f)},
            {eltwise(post_op_type_t::tanh),
                    eltwise(post_op_type_t::swish, 0.5f),
                    binary(post_op_type_t::binary_add, data_type_t::f32, {1})},
            {eltwise(post_op_type_t::abs), eltwise(post_op_type_t::sqrt),
                    eltwise(post_op_type_t::log)},
            {eltwise(post_op_type_t::mish), eltwise(post_op_type_t::square),
                    eltwise(post_op_type_t::exp)},
            {binary(post_op_type_t::binary_mul, data_type_t::f32, {M, N}),
                    binary(post_op_type_t::binary_add, data_type_t::f32,
                            {M, N}),
                    eltwise(post_op_type_t::sigmoid)},
            {eltwise(post_op_type_t::leaky_relu, 0.1f),
                    eltwise(post_op_type_t::elu, 0.7f),
                    eltwise(post_op_type_t::gelu_tanh)},
            {eltwise(post_op_type_t::none), eltwise(post_op_type_t::relu),
                    eltwise(post_op_type_t::none)},
    };
    uint32_t seed = 3;
    for (const auto &chain : chains)
        for (data_type_t dst : {data_type_t::f32, data_type_t::bf16}) {
            problem_t pb;
            pb.M = M;
            pb.N = N;
            pb.dst_dt = dst;
            pb.bias_dt = data_type_t::f32;
            pb.po = chain;
            pb.seed = seed++;
            check(pb, 1e-3f);
        }
}

TEST(NativeBrgemmEpilogue, BinaryDtypesShapesLeadingDim) {
    SKIP_WITHOUT_AVX512();
    const int M = 6, N = 96;
    const std::vector<std::pair<std::vector<int64_t>, int>> shapes
            = {{{M, N}, -1}, {{M, N}, N + 24}, {{M * N}, -1}, {{1, N}, -1},
                    {{N}, -1}, {{1}, -1}, {{}, -1}, {{1, 1}, -1}, {{M, 1}, -1},
                    {{2, 3}, -1}};
    uint32_t seed = 40;
    for (data_type_t bdt :
            {data_type_t::f32, data_type_t::bf16, data_type_t::f16,
                    data_type_t::s8, data_type_t::u8, data_type_t::s32})
        for (const auto &shape : shapes)
            for (post_op_type_t t :
                    {post_op_type_t::binary_add, post_op_type_t::binary_mul}) {
                problem_t pb;
                pb.M = M;
                pb.N = N;
                pb.ldc_pad = 5;
                pb.po = {binary(t, bdt, shape.first, shape.second)};
                pb.seed = seed++;
                check(pb);
            }
}

TEST(NativeBrgemmEpilogue, GatedActsBiasLayouts) {
    SKIP_WITHOUT_AVX512();
    uint32_t seed = 100;
    for (gated_t g : {gated_t::silu_and_mul, gated_t::gelu_and_mul,
                 gated_t::swiglu_oai_mul})
        for (data_type_t bdt : {data_type_t::none, data_type_t::f32,
                     data_type_t::bf16, data_type_t::f16})
            for (data_type_t dst : {data_type_t::f32, data_type_t::bf16})
                for (int tn : {16, 32, 64}) {
                    problem_t pb;
                    pb.M = 11;
                    pb.N = 128;
                    pb.gated = g;
                    pb.interleaved = g != gated_t::swiglu_oai_mul;
                    pb.bias_dt = bdt;
                    pb.dst_dt = dst;
                    pb.tile_n = tn;
                    pb.tile_m = 4;
                    pb.ldc_pad = 3;
                    pb.alpha = seed % 2 ? 1.0f : 0.5f;
                    pb.acc_lo = -6.0f;
                    pb.acc_hi = 6.0f;
                    pb.seed = seed++;
                    check(pb);
                }
}

TEST(NativeBrgemmEpilogue, GatedWithPostOpsAndInt8Store) {
    SKIP_WITHOUT_AVX512();
    problem_t pb;
    pb.M = 13;
    pb.N = 192;
    pb.gated = gated_t::silu_and_mul;
    pb.interleaved = true;
    pb.bias_dt = data_type_t::bf16;
    pb.po = {
            binary(post_op_type_t::binary_mul, data_type_t::bf16, {pb.M, pb.N}),
            eltwise(post_op_type_t::clip, -1.0f, 1.0f),
            binary(post_op_type_t::binary_add, data_type_t::f32, {1, pb.N})};
    pb.tile_n = 64;
    check(pb, 1e-3f);

    pb.gated = gated_t::swiglu_oai_mul;
    pb.interleaved = false;
    pb.int8 = true;
    pb.dst_dt = data_type_t::u8;
    pb.dst_scale_dt = data_type_t::f32;
    pb.dst_zp_dt = data_type_t::s32;
    pb.tile_n = 16;
    pb.seed = 9;
    check(pb);
}

// Gated calls with beta, post-ops and other destinations, as matmul_direct
// defines them: beta reads the dst's N pre-activation columns (the gate and
// up columns of an interleaved pack's pairs), the bias and the post-ops act
// on the pre-activation, the activation and the store (requantized for an
// INT8 u8 / s8 dst) follow; the columns past N / 2 stay as they were.
TEST(NativeBrgemmEpilogue, GatedBetaPostOpsDestinations) {
    SKIP_WITHOUT_AVX512();
    const int M = 9, N = 128;
    const std::vector<std::vector<po_spec_t>> chains = {{},
            {eltwise(post_op_type_t::elu, 0.5f)},
            {binary(post_op_type_t::binary_add, data_type_t::bf16, {M, N}),
                    eltwise(post_op_type_t::square)},
            {binary(post_op_type_t::binary_mul, data_type_t::f32, {1, N}),
                    eltwise(post_op_type_t::abs),
                    eltwise(post_op_type_t::clip, -1.5f, 1.5f)},
            {eltwise(post_op_type_t::gelu_tanh),
                    binary(post_op_type_t::binary_add, data_type_t::f32, {M, N},
                            N + 8)}};
    uint32_t seed = 500;
    int variant = 0;
    for (gated_t g : {gated_t::silu_and_mul, gated_t::gelu_and_mul,
                 gated_t::swiglu_oai_mul})
        for (data_type_t dst : {data_type_t::f32, data_type_t::bf16,
                     data_type_t::f16, data_type_t::u8, data_type_t::s8})
            for (size_t c = 0; c < chains.size(); ++c) {
                const int v = variant++;
                problem_t pb;
                pb.M = M;
                pb.N = N;
                pb.gated = g;
                pb.interleaved = g != gated_t::swiglu_oai_mul;
                pb.dst_dt = dst;
                // An INT8 call's alpha and beta are its tile's.
                pb.int8 = (dst == data_type_t::u8 || dst == data_type_t::s8)
                        && v % 2 == 0;
                if (pb.int8) {
                    pb.dst_scale_dt = data_type_t::f32;
                    pb.dst_zp_dt = data_type_t::s32;
                } else {
                    pb.beta = v % 3 == 0 ? 0.5f : (v % 3 == 1 ? -1.0f : 0.0f);
                    pb.alpha = v % 4 == 3 ? 1.5f : 1.0f;
                }
                pb.ldc_pad = N / 2 + 3;
                pb.bias_dt = v % 2 ? data_type_t::f32 : data_type_t::bf16;
                pb.po = chains[c];
                pb.tile_n = v % 3 == 0 ? 16 : (v % 3 == 1 ? 32 : 64);
                pb.tile_m = 4;
                pb.threaded = v % 2 == 1;
                pb.acc_lo = -4.0f;
                pb.acc_hi = 4.0f;
                pb.seed = seed++;
                check(pb, 1e-3f);
            }
}

// A gated call with a softmax, or with beta over a dst of fewer than N
// columns, is not the epilogue's.
TEST(NativeBrgemmEpilogue, GatedSoftmaxAndNarrowBetaUnimplemented) {
    SKIP_WITHOUT_AVX512();
    problem_t pb;
    pb.N = 128;
    pb.gated = gated_t::silu_and_mul;
    pb.interleaved = true;
    pb.po = {eltwise(post_op_type_t::softmax)};
    {
        inputs_t in = make_inputs(pb);
        buf_t dst = in.dst0;
        nat::native_epilogue_t ep;
        EXPECT_EQ(nat::prepare_epilogue(make_desc(pb), make_params(pb, in),
                          nullptr, dst.data(), pb.interleaved, ep),
                status_t::unimplemented);
    }
    pb.po.clear();
    pb.beta = 1.0f;
    for (int pad : {0, pb.N / 2 - 1, pb.N / 2}) {
        pb.ldc_pad = pad;
        inputs_t in = make_inputs(pb);
        buf_t dst = in.dst0;
        nat::native_epilogue_t ep;
        EXPECT_EQ(nat::prepare_epilogue(make_desc(pb), make_params(pb, in),
                          nullptr, dst.data(), pb.interleaved, ep),
                pad < pb.N / 2 ? status_t::unimplemented : status_t::success)
                << "ldc " << pb.ldc();
    }
}

TEST(NativeBrgemmEpilogue, AlphaBetaReferenceOrder) {
    SKIP_WITHOUT_AVX512();
    uint32_t seed = 200;
    for (float a : {0.0f, 1.0f, 2.5f, -1.0f})
        for (float b : {0.0f, 0.5f, 1.0f, -2.0f})
            for (data_type_t dst :
                    {data_type_t::f32, data_type_t::bf16, data_type_t::u8}) {
                problem_t pb;
                pb.alpha = a;
                pb.beta = b;
                pb.dst_dt = dst;
                pb.bias_dt = data_type_t::f32;
                pb.ldc_pad = 7;
                pb.seed = seed++;
                check(pb);
            }
}

// Calls give a destination scale and zero point per tensor; those per column
// or per element unit-test the operand reader, which binary post-ops of the
// same shapes share.
TEST(NativeBrgemmEpilogue, Int8StoresAndRequant) {
    SKIP_WITHOUT_AVX512();
    struct q_t {
        data_type_t scale_dt;
        int64_t scale_count;
        data_type_t zp_dt;
        int64_t zp_count;
    };
    const int M = 5, N = 64;
    const std::vector<q_t> qs = {{data_type_t::none, 1, data_type_t::none, 1},
            {data_type_t::f32, 1, data_type_t::none, 1},
            {data_type_t::f32, 1, data_type_t::s32, 1},
            {data_type_t::bf16, 1, data_type_t::u8, 1},
            {data_type_t::f32, N, data_type_t::s32, N},
            {data_type_t::f32, M * N, data_type_t::s8, 1},
            {data_type_t::f32, 1, data_type_t::f32, 1}};
    uint32_t seed = 300;
    for (data_type_t dst : {data_type_t::u8, data_type_t::s8, data_type_t::s32,
                 data_type_t::f32, data_type_t::bf16, data_type_t::f16})
        for (const q_t &q : qs) {
            problem_t pb;
            pb.M = M;
            pb.N = N;
            pb.int8 = true;
            pb.dst_dt = dst;
            pb.bias_dt = data_type_t::f32;
            pb.dst_scale_dt = q.scale_dt;
            pb.dst_scale_count = q.scale_count;
            pb.dst_zp_dt = q.zp_dt;
            pb.dst_zp_count = q.zp_count;
            pb.po = {eltwise(post_op_type_t::relu)};
            pb.seed = seed++;
            check(pb);
        }
}

TEST(NativeBrgemmEpilogue, RequantRoundsHalfToEven) {
    SKIP_WITHOUT_AVX512();
    // v * dst_scale lands on .5: 1, 3, 5, 7 -> 0.5 .. 3.5, then + 10.
    const int N = 32;
    std::vector<float> acc(N);
    for (int j = 0; j < N; ++j)
        acc[j] = static_cast<float>(2 * (j % 4) + 1);
    const float scale = 0.5f;
    const int32_t zp = 10;
    std::vector<uint8_t> dst(N, 0);
    mm::matmul_params p;
    p.dtypes.dst = data_type_t::u8;
    p.quant_params.dst_scale.buff = &scale;
    p.quant_params.dst_scale.dt = data_type_t::f32;
    p.quant_params.dst_zp.buff = &zp;
    p.quant_params.dst_zp.dt = data_type_t::s32;
    nat::GemmDescriptor d;
    d.M = 1;
    d.N = N;
    d.ldc = N;
    d.wei_dt = data_type_t::s8;
    d.src_dt = data_type_t::u8;
    d.dst_dt = data_type_t::u8;
    nat::native_epilogue_t ep;
    ASSERT_EQ(nat::prepare_epilogue(d, p, nullptr, dst.data(), false, ep),
            status_t::success);
    nat::apply_epilogue_tile(ep, acc.data(), N, 0, 1, 0, N);
    const uint8_t want[4] = {10, 12, 12, 14};
    for (int j = 0; j < N; ++j)
        EXPECT_EQ(dst[j], want[j % 4]) << j;
}

TEST(NativeBrgemmEpilogue, StoresRoundHalfToEven) {
    SKIP_WITHOUT_AVX512();
    struct case_t {
        data_type_t dt;
        std::vector<float> in;
        std::vector<uint32_t> bits; ///< stored element bits
    };
    const float b = 1.0f / 256.0f, h = 1.0f / 2048.0f;
    const std::vector<case_t> cases = {
            {data_type_t::bf16,
                    {1 + b, 1 + 3 * b, -(1 + b), 1 + b + 1.0f / (1 << 20)},
                    {0x3F80, 0x3F82, 0xBF80, 0x3F81}},
            {data_type_t::f16, {1 + h, 1 + 3 * h, -(1 + h)},
                    {0x3C00, 0x3C02, 0xBC00}},
            {data_type_t::s8, {2.5f, -2.5f, 3.5f, 200.0f, -300.0f},
                    {2, 0xFE, 4, 127, 0x80}},
            {data_type_t::s32, {2.5f, -3.5f, 1.5f}, {2, 0xFFFFFFFC, 2}},
            {data_type_t::u8, {0.5f, 1.5f, -4.0f, 300.0f}, {0, 2, 0, 255}}};
    for (const case_t &c : cases) {
        SCOPED_TRACE(zendnnl::common::dtype_info(c.dt));
        const int N = 32;
        std::vector<float> acc(N, 0.0f);
        std::copy(c.in.begin(), c.in.end(), acc.begin());
        buf_t dst;
        dst.init(c.dt, N);
        mm::matmul_params p;
        p.dtypes.dst = c.dt;
        nat::GemmDescriptor d;
        d.M = 1;
        d.N = N;
        d.ldc = N;
        d.dst_dt = c.dt;
        nat::native_epilogue_t ep;
        ASSERT_EQ(nat::prepare_epilogue(d, p, nullptr, dst.data(), false, ep),
                status_t::success);
        nat::apply_epilogue_tile(ep, acc.data(), N, 0, 1, 0, N);
        for (size_t i = 0; i < c.bits.size(); ++i) {
            uint32_t got = 0;
            std::memcpy(
                    &got, dst.bytes.data() + i * size_of(c.dt), size_of(c.dt));
            EXPECT_EQ(got, c.bits[i]) << "element " << i;
        }
    }
}

// An epilogue with only the store (an f32 or bf16 dst, alpha 1, beta 0,
// nothing else to apply) stores what the general path stores, byte for byte:
// NaNs, infinities, signed zeros, subnormals and rounding ties included, at
// every width and tail and scratch stride, and nothing outside the tile.
TEST(NativeBrgemmEpilogue, StoreOnlyStoresTheGeneralPathsBytes) {
    SKIP_WITHOUT_AVX512();
    const float b = 1.0f / 256.0f;
    const uint32_t nans[]
            = {0x7FC00000u, 0xFFC00001u, 0x7F800001u, 0xFF812345u};
    std::vector<float> specials = {0.0f, -0.0f, 1.0f, -1.0f, 1 + b, 1 + 3 * b,
            -(1 + b), 1 + b + 1.0f / (1 << 20),
            std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::denorm_min(),
            -std::numeric_limits<float>::denorm_min(), 1e-40f, -3e-39f,
            std::numeric_limits<float>::min(),
            std::numeric_limits<float>::max(),
            -std::numeric_limits<float>::max()};
    for (const uint32_t bits : nans) {
        float v;
        std::memcpy(&v, &bits, sizeof(v));
        specials.push_back(v);
    }
    std::mt19937 rng(20261028);
    std::uniform_real_distribution<float> any(-4.0f, 4.0f);
    for (const data_type_t dt : {data_type_t::bf16, data_type_t::f32})
        for (const int nr : {1, 15, 16, 17, 31, 32, 48, 64})
            for (const int mr : {1, 3}) {
                SCOPED_TRACE(::testing::Message()
                        << zendnnl::common::dtype_info(dt) << " nr " << nr
                        << " mr " << mr);
                constexpr int M = 5, N = 80, m0 = 1, n0 = 8;
                const int ld = nr + 3;
                std::vector<float> acc(static_cast<size_t>(mr) * ld);
                for (size_t i = 0; i < acc.size(); ++i)
                    acc[i] = i % 3 == 0 ? specials[(i / 3) % specials.size()]
                                        : any(rng);
                std::vector<float> acc2(acc);
                const size_t bytes = static_cast<size_t>(M) * N * size_of(dt);
                std::vector<uint8_t> fast(bytes, 0xA5), slow(bytes, 0xA5);
                mm::matmul_params p;
                p.dtypes.dst = dt;
                nat::GemmDescriptor d;
                d.M = M;
                d.N = N;
                d.ldc = N;
                d.dst_dt = dt;
                nat::native_epilogue_t e1, e2;
                ASSERT_EQ(nat::prepare_epilogue(
                                  d, p, nullptr, fast.data(), false, e1),
                        status_t::success);
                ASSERT_EQ(nat::prepare_epilogue(
                                  d, p, nullptr, slow.data(), false, e2),
                        status_t::success);
                ASSERT_TRUE(e1.store_only);
                e2.store_only = false;
                nat::apply_epilogue_tile(e1, acc.data(), ld, m0, mr, n0, nr);
                nat::apply_epilogue_tile(e2, acc2.data(), ld, m0, mr, n0, nr);
                EXPECT_TRUE(fast == slow);
                size_t written = 0;
                for (size_t i = 0; i < bytes; ++i)
                    written += fast[i] != 0xA5 ? 1 : 0;
                EXPECT_LE(written, static_cast<size_t>(mr) * nr * size_of(dt));
            }
    // Anything to apply, or another dst type, takes the general path.
    const auto store_only = [](nat::GemmDescriptor d, mm::matmul_params p,
                                    const void *bias, bool interleaved) {
        std::vector<uint8_t> dst(static_cast<size_t>(d.M) * d.ldc * 4);
        nat::native_epilogue_t e;
        EXPECT_EQ(nat::prepare_epilogue(d, p, bias, dst.data(), interleaved, e),
                status_t::success);
        return e.store_only;
    };
    nat::GemmDescriptor d;
    d.M = 2;
    d.N = 32;
    d.ldc = 32;
    d.dst_dt = data_type_t::bf16;
    d.wei_dt = data_type_t::bf16;
    mm::matmul_params p;
    p.dtypes.dst = data_type_t::bf16;
    EXPECT_TRUE(store_only(d, p, nullptr, false));
    const std::vector<float> bias(32, 0.0f);
    p.dtypes.bias = data_type_t::f32;
    EXPECT_FALSE(store_only(d, p, bias.data(), false));
    nat::GemmDescriptor a = d;
    a.alpha = 2.0f;
    EXPECT_FALSE(store_only(a, p, nullptr, false));
    a = d;
    a.beta = 1.0f;
    EXPECT_FALSE(store_only(a, p, nullptr, false));
    mm::matmul_params relu = p;
    mm::matmul_post_op po;
    po.po_type = post_op_type_t::relu;
    relu.postop_ = {po};
    EXPECT_FALSE(store_only(d, relu, nullptr, false));
    a = d;
    a.dst_dt = data_type_t::f16;
    mm::matmul_params f16 = p;
    f16.dtypes.dst = data_type_t::f16;
    EXPECT_FALSE(store_only(a, f16, nullptr, false));
    // A gated activation, of split or interleaved halves.
    a = d;
    a.gated_act = gated_t::swiglu_oai_mul;
    EXPECT_FALSE(store_only(a, p, nullptr, false));
    a.gated_act = gated_t::silu_and_mul;
    EXPECT_FALSE(store_only(a, p, nullptr, true));
    // An INT8 call's dst scale on an f32 dst, with nothing else to apply.
    nat::GemmDescriptor q = d;
    q.wei_dt = data_type_t::s8;
    q.src_dt = data_type_t::u8;
    q.dst_dt = data_type_t::f32;
    mm::matmul_params requant;
    requant.dtypes.dst = data_type_t::f32;
    EXPECT_TRUE(store_only(q, requant, nullptr, false));
    const float dst_scale = 0.5f;
    requant.quant_params.dst_scale.buff = &dst_scale;
    requant.quant_params.dst_scale.dt = data_type_t::f32;
    EXPECT_FALSE(store_only(q, requant, nullptr, false));
}

// Where tanh(x) rounds to x, the tanh post-op returns x: tiny and subnormal
// inputs keep their value, signed zeros their sign.  That is within one ulp
// of the reference's tanhf, which rounds some of them away from x.
TEST(NativeBrgemmEpilogue, TanhKeepsTinyInputsAndTheSignOfZero) {
    SKIP_WITHOUT_AVX512();
    const float sub = std::numeric_limits<float>::denorm_min();
    const float below = std::nextafter(std::ldexp(1.0f, -12), 0.0f);
    const std::vector<float> in = {0.0f, -0.0f, sub, -sub, 1e-40f, -1e-38f,
            1e-30f, -1e-20f, 1e-10f, -1e-7f, 1e-6f, -3e-5f,
            std::ldexp(1.0f + std::ldexp(1.0f, -23), -25), 1.2e-4f, below,
            -below};
    const int N = static_cast<int>(in.size());
    std::vector<float> acc(in), dst(N, 7.0f);
    mm::matmul_params p;
    p.dtypes.dst = data_type_t::f32;
    mm::matmul_post_op po;
    po.po_type = post_op_type_t::tanh;
    p.postop_ = {po};
    nat::GemmDescriptor d;
    d.M = 1;
    d.N = N;
    d.ldc = N;
    d.dst_dt = data_type_t::f32;
    nat::native_epilogue_t ep;
    ASSERT_EQ(nat::prepare_epilogue(d, p, nullptr, dst.data(), false, ep),
            status_t::success);
    nat::apply_epilogue_tile(ep, acc.data(), N, 0, 1, 0, N);
    for (int i = 0; i < N; ++i) {
        SCOPED_TRACE(::testing::Message() << "x = " << in[i]);
        EXPECT_EQ(std::memcmp(&dst[i], &in[i], sizeof(float)), 0)
                << "tanh(x) " << dst[i];
        const float ref = tanhf(in[i]);
        EXPECT_TRUE(dst[i] == ref || dst[i] == std::nextafter(ref, -1.0f)
                || dst[i] == std::nextafter(ref, 1.0f))
                << "reference " << ref;
        EXPECT_EQ(std::signbit(dst[i]), std::signbit(ref));
    }
}

TEST(NativeBrgemmEpilogue, TileOffsetsWidthsAndScratchStride) {
    SKIP_WITHOUT_AVX512();
    uint32_t seed = 400;
    for (int tm : {1, 5, 6, 7, 192})
        for (int tn : {16, 32, 64})
            for (int pad : {0, 3}) {
                problem_t pb;
                pb.M = 13;
                pb.N = 192;
                pb.tile_m = tm;
                pb.tile_n = tn;
                pb.acc_pad = pad;
                pb.bias_dt = data_type_t::f16;
                pb.dst_dt = seed % 2 ? data_type_t::bf16 : data_type_t::f32;
                pb.po = {binary(post_op_type_t::binary_add, data_type_t::f32,
                                 {pb.M, pb.N}, pb.N + 16),
                        eltwise(post_op_type_t::gelu_tanh)};
                pb.seed = seed++;
                check(pb);
            }
}

// A gated call does not take a softmax (GatedSoftmaxAndNarrowBetaUnimplemented).
TEST(NativeBrgemmEpilogue, SoftmaxWholeRows) {
    SKIP_WITHOUT_AVX512();
    uint32_t seed = 500;
    for (bool threaded : {false, true})
        for (int n : {256, 128}) {
            problem_t pb;
            pb.M = 17;
            pb.N = n;
            pb.tile_m = 6;
            pb.tile_n = 32;
            pb.threaded = threaded;
            pb.bias_dt = data_type_t::f32;
            pb.dst_dt = data_type_t::f32;
            pb.acc_lo = -1.0f;
            pb.acc_hi = 1.0f;
            pb.po = {eltwise(post_op_type_t::relu),
                    eltwise(post_op_type_t::softmax),
                    binary(post_op_type_t::binary_mul, data_type_t::f32,
                            {pb.M, pb.n_out()}),
                    eltwise(post_op_type_t::softmax)};
            pb.seed = seed++;
            check(pb, 1e-4f);
        }
}

TEST(NativeBrgemmEpilogue, ThreadedTilesMatchModel) {
    SKIP_WITHOUT_AVX512();
    problem_t pb;
    pb.M = 70;
    pb.N = 512;
    pb.threaded = true;
    pb.tile_m = 6;
    pb.tile_n = 64;
    pb.bias_dt = data_type_t::bf16;
    pb.dst_dt = data_type_t::bf16;
    pb.po = {eltwise(post_op_type_t::swish, 1.0f),
            binary(post_op_type_t::binary_add, data_type_t::bf16,
                    {pb.M, pb.N})};
    check(pb);
    pb.gated = gated_t::gelu_and_mul;
    pb.interleaved = true;
    pb.po = {eltwise(post_op_type_t::mish)};
    check(pb);
}

TEST(NativeBrgemmEpilogue, RefusesWhatItCannotFinish) {
    SKIP_WITHOUT_AVX512();
    const int M = 4, N = 64;
    std::vector<float> dst(M * N), bias(N);
    auto prep = [&](nat::GemmDescriptor d, const mm::matmul_params &p,
                        bool inter, const void *b = nullptr) {
        nat::native_epilogue_t ep;
        return nat::prepare_epilogue(d, p, b, dst.data(), inter, ep);
    };
    nat::GemmDescriptor d;
    d.M = M;
    d.N = N;
    d.ldc = N;
    d.dst_dt = data_type_t::f32;
    d.wei_dt = data_type_t::bf16;
    mm::matmul_params p;
    EXPECT_EQ(prep(d, p, false), status_t::success);
    EXPECT_EQ(prep(d, p, true), status_t::unimplemented);

    mm::matmul_post_op pool;
    pool.po_type = post_op_type_t::pooling;
    p.postop_ = {pool};
    EXPECT_EQ(prep(d, p, false), status_t::unimplemented);
    mm::matmul_post_op add;
    add.po_type = post_op_type_t::binary_add;
    add.dtype = data_type_t::f32;
    add.dims = {M, N};
    p.postop_ = {add};
    EXPECT_EQ(prep(d, p, false), status_t::failure);
    add.buff = bias.data();
    add.dtype = data_type_t::s4;
    p.postop_ = {add};
    EXPECT_EQ(prep(d, p, false), status_t::unimplemented);
    add.dtype = data_type_t::f32;
    add.leading_dim = N - 1;
    p.postop_ = {add};
    EXPECT_EQ(prep(d, p, false), status_t::failure);
    p.postop_.clear();

    p.dtypes.bias = data_type_t::none;
    EXPECT_EQ(prep(d, p, false, bias.data()), status_t::unimplemented);
    p.dtypes.bias = data_type_t::f32;

    nat::GemmDescriptor bad = d;
    bad.dst_dt = data_type_t::s4;
    EXPECT_EQ(prep(bad, p, false), status_t::unimplemented);
    bad = d;
    bad.ldc = N - 1;
    EXPECT_EQ(prep(bad, p, false), status_t::failure);

    // INT8 alpha / beta belong to the INT8 tile, except with per-group
    // weight scales, where the epilogue applies them.
    bad = d;
    bad.wei_dt = data_type_t::s8;
    bad.src_dt = data_type_t::u8;
    EXPECT_EQ(prep(bad, p, false), status_t::success);
    bad.alpha = 0.5f;
    EXPECT_EQ(prep(bad, p, false), status_t::unimplemented);
    bad.alpha = 1.0f;
    bad.beta = 1.0f;
    EXPECT_EQ(prep(bad, p, false), status_t::unimplemented);
    std::vector<float> group_scales(2 * N, 0.25f);
    mm::matmul_params grouped = p;
    grouped.quant_params.wei_scale.buff = group_scales.data();
    grouped.quant_params.wei_scale.dt = data_type_t::f32;
    grouped.quant_params.wei_scale.dims = {2, N};
    bad.alpha = 0.5f;
    EXPECT_EQ(prep(bad, grouped, false), status_t::success);

    d.gated_act = gated_t::silu_and_mul;
    EXPECT_EQ(prep(d, p, true), status_t::success);
    EXPECT_EQ(prep(d, p, false), status_t::failure);
    bad = d;
    bad.ldc = N / 2;
    EXPECT_EQ(prep(bad, p, true), status_t::success);
    bad.beta = 1.0f;
    EXPECT_EQ(prep(bad, p, true), status_t::unimplemented);
    d.gated_act = gated_t::swiglu_oai_mul;
    EXPECT_EQ(prep(d, p, false), status_t::success);
    EXPECT_EQ(prep(d, p, true), status_t::failure);
    d.gated_act = static_cast<gated_t>(7);
    EXPECT_EQ(prep(d, p, false), status_t::unimplemented);
}

// ---- end to end through native_brgemm_execute (BF16) ------------------------

bool bf16_cpu_ok() {
    return cpu_ok() && nat::detect_uarch().avx512bf16;
}

#define SKIP_WITHOUT_AVX512_BF16() \
    if (!bf16_cpu_ok()) GTEST_SKIP() << "needs AVX512-BF16"

struct e2e_t {
    int M = 8;
    int N = 64; ///< weight columns
    int K = 64;
    bool transA = false;
    bool transB = false;
    int lda_pad = 0;
    int ldc_pad = 0;
    gated_t gated = gated_t::none;
    float alpha = 1.0f;
    float beta = 0.0f;
    data_type_t bias_dt = data_type_t::none;
    data_type_t dst_dt = data_type_t::bf16;
    std::vector<po_spec_t> po;
    int threads = 1;
    uint32_t seed = 1;

    int n_out() const { return gated == gated_t::none ? N : N / 2; }
    int lda() const { return (transA ? M : K) + lda_pad; }
    int ldb() const { return transB ? K : N; }
    int ldc() const { return n_out() + ldc_pad; }
    bool interleave() const {
        return gated == gated_t::silu_and_mul || gated == gated_t::gelu_and_mul;
    }
    std::string str() const {
        std::ostringstream os;
        os << "M=" << M << " N=" << N << " K=" << K << " transA=" << transA
           << " transB=" << transB << " gated=" << static_cast<int>(gated)
           << " alpha=" << alpha << " beta=" << beta
           << " bias=" << zendnnl::common::dtype_info(bias_dt)
           << " dst=" << zendnnl::common::dtype_info(dst_dt)
           << " threads=" << threads << " po=";
        for (const auto &s : po)
            os << static_cast<int>(s.type) << ",";
        return os.str();
    }
};

struct e2e_data_t {
    std::vector<bfloat16_t> a, w;
    buf_t bias, dst0;
    std::vector<buf_t> bin;
    std::vector<mm::matmul_post_op> postops;

    const void *bias_ptr(const e2e_t &c) const {
        return c.bias_dt == data_type_t::none ? nullptr : bias.data();
    }
};

e2e_data_t make_e2e(const e2e_t &c) {
    std::mt19937 rng(c.seed);
    std::uniform_real_distribution<float> du(-1.0f, 1.0f);
    e2e_data_t d;
    const float a_scale = 1.0f / std::sqrt(static_cast<float>(c.K));
    d.a.resize(static_cast<size_t>(c.transA ? c.K : c.M) * c.lda());
    for (auto &v : d.a)
        v = bfloat16_t(du(rng) * a_scale);
    d.w.resize(static_cast<size_t>(c.K) * c.N);
    for (auto &v : d.w)
        v = bfloat16_t(du(rng));
    if (c.bias_dt != data_type_t::none) {
        d.bias.init(c.bias_dt, c.N);
        d.bias.fill(rng, -0.5f, 0.5f);
    }
    d.dst0.init(c.dst_dt, static_cast<int64_t>(c.M) * c.ldc());
    d.dst0.fill(rng, -1.0f, 1.0f);
    // Past the columns the call reads: beta of a gated call reads N.
    const int read = c.gated != gated_t::none && c.beta != 0.0f
            ? std::min(c.N, c.ldc())
            : c.n_out();
    for (int m = 0; m < c.M; ++m)
        for (int j = read; j < c.ldc(); ++j)
            d.dst0.set(static_cast<int64_t>(m) * c.ldc() + j, 99.0f);
    make_postops(c.po, c.M, c.gated == gated_t::none ? c.n_out() : c.N, rng,
            d.bin, d.postops);
    return d;
}

mm::matmul_params bf16_params(data_type_t dst_dt, data_type_t bias_dt) {
    mm::matmul_params p;
    p.dtypes.src = data_type_t::bf16;
    p.dtypes.wei = data_type_t::bf16;
    p.dtypes.dst = dst_dt;
    p.dtypes.bias = bias_dt;
    return p;
}

status_t run_native(const e2e_t &c, const e2e_data_t &d, buf_t &dst) {
    nat::GemmDescriptor desc;
    desc.M = c.M;
    desc.N = c.N;
    desc.K = c.K;
    desc.lda = c.lda();
    desc.ldb = c.ldb();
    desc.ldc = c.ldc();
    desc.transA = c.transA;
    desc.transB = c.transB;
    desc.alpha = c.alpha;
    desc.beta = c.beta;
    desc.src_dt = data_type_t::bf16;
    desc.wei_dt = data_type_t::bf16;
    desc.dst_dt = c.dst_dt;
    desc.bias_dt = c.bias_dt;
    desc.src_elem_size = 2;
    desc.wei_elem_size = 2;
    desc.dst_elem_size = size_of(c.dst_dt);
    desc.bias = d.bias_ptr(c);
    desc.is_weights_const = true;
    desc.num_threads = c.threads;
    desc.gated_act = c.gated;
    mm::matmul_params p = bf16_params(c.dst_dt, c.bias_dt);
    p.postop_ = d.postops;
    nat::native_weight_ref_t ref;
    ref.weight = d.w.data();
    ref.pack_nr = 32;
    ref.interleave_split_halves = c.interleave();
    ref.weight_cache_mode = 0;
    return nat::native_brgemm_execute(
            desc, ref, d.a.data(), dst.data(), d.bias_ptr(c), p);
}

/// The reference algorithm through matmul_direct; plain drops the gated act
/// and post-ops and writes the f32 [M, N] pre-activation instead.
status_t run_reference(
        const e2e_t &c, const e2e_data_t &d, bool plain, buf_t &dst) {
    mm::matmul_params p = bf16_params(dst.dt, c.bias_dt);
    p.lowoha_algo = mm::matmul_algo_t::reference;
    if (!plain) p.postop_ = d.postops;
    mm::matmul_batch_params_t bp;
    return mm::matmul_direct('r', c.transA, c.transB, c.M, c.N, c.K, c.alpha,
            d.a.data(), c.lda(), d.w.data(), c.ldb(), d.bias_ptr(c),
            plain ? 0.0f : c.beta, dst.data(), plain ? c.N : c.ldc(), false, bp,
            p);
}

/// matmul_direct with lowoha_algo native_brgemm on the raw weight, packed for
/// the call only.
status_t run_routed(const e2e_t &c, const e2e_data_t &d, buf_t &dst) {
    mm::matmul_params p = bf16_params(c.dst_dt, c.bias_dt);
    p.postop_ = d.postops;
    p.num_threads = c.threads;
    p.lowoha_algo = mm::matmul_algo_t::native_brgemm;
    p.weight_cache_type = 0;
    mm::matmul_batch_params_t bp;
    mm::matmul_gated_act_params act;
    act.act = c.gated;
    return mm::matmul_direct('r', c.transA, c.transB, c.M, c.N, c.K, c.alpha,
            d.a.data(), c.lda(), d.w.data(), c.ldb(), d.bias_ptr(c), c.beta,
            dst.data(), c.ldc(), true, bp, p,
            c.gated == gated_t::none ? nullptr : &act);
}

/// The one-op group call with the gated act, bf16 dst [M, N].
status_t run_group(const e2e_t &c, const e2e_data_t &d, buf_t &dst) {
    mm::matmul_params p = bf16_params(data_type_t::bf16, c.bias_dt);
    p.num_threads = c.threads;
    mm::grp_matmul_gated_act_params act;
    act.act = static_cast<mm::grp_matmul_gated_act_t>(
            static_cast<int>(c.gated));
    return mm::group_matmul_direct({'r'}, {c.transA}, {c.transB}, {c.M}, {c.N},
            {c.K}, {c.alpha}, {static_cast<const void *>(d.a.data())},
            {c.lda()}, {static_cast<const void *>(d.w.data())}, {c.ldb()},
            {d.bias_ptr(c)}, {0.0f}, {dst.data()}, {c.N}, {false}, {p}, nullptr,
            &act, nullptr);
}

/// Native dst against want(m, o) over [M, n_out] within f32 accumulation and
/// approximation error (plus one bf16 rounding), padding untouched.
template <typename want_t>
void expect_native(const e2e_t &c, const e2e_data_t &d, const buf_t &got,
        want_t want, float rel, float abs) {
    int bad = 0;
    for (int m = 0; m < c.M; ++m)
        for (int o = 0; o < c.ldc(); ++o) {
            const int64_t at = static_cast<int64_t>(m) * c.ldc() + o;
            const float g = got.get(at);
            if (o >= c.n_out()) {
                if (g != d.dst0.get(at) && bad++ < 4)
                    ADD_FAILURE() << "padding (" << m << "," << o
                                  << ") written: " << g;
                continue;
            }
            const float w = want(m, o);
            float tol = abs + rel * std::fabs(w);
            if (c.dst_dt == data_type_t::bf16) tol += std::fabs(w) / 128.0f;
            if (!(std::fabs(g - w) <= tol) && bad++ < 8)
                ADD_FAILURE() << "(" << m << "," << o << ") native " << g
                              << " want " << w;
        }
    EXPECT_EQ(bad, 0);
}

TEST(NativeBrgemmEpilogueE2E, PostOpsVsReference) {
    SKIP_WITHOUT_AVX512_BF16();
    struct shape_t {
        int M, N, K;
    };
    const std::vector<shape_t> shapes = {{1, 64, 64}, {7, 96, 31},
            {33, 256, 300}, {130, 128, 64}, {6, 64, 1}};
    uint32_t seed = 1000;
    int variant = 0;
    for (const shape_t &s : shapes) {
        const std::vector<std::vector<po_spec_t>> chains = {{},
                {eltwise(post_op_type_t::gelu_erf),
                        binary(post_op_type_t::binary_add, data_type_t::bf16,
                                {s.M, s.N})},
                {eltwise(post_op_type_t::mish)},
                {eltwise(post_op_type_t::swish, 0.5f),
                        eltwise(post_op_type_t::clip, -0.2f, 0.3f)},
                {binary(post_op_type_t::binary_mul, data_type_t::f32, {1, s.N}),
                        eltwise(post_op_type_t::sigmoid)},
                {eltwise(post_op_type_t::softmax)},
                {eltwise(post_op_type_t::tanh),
                        binary(post_op_type_t::binary_add, data_type_t::f32,
                                {1})}};
        for (const auto &chain : chains) {
            const int v = variant++;
            e2e_t c;
            c.M = s.M;
            c.N = s.N;
            c.K = s.K;
            c.po = chain;
            c.dst_dt = v % 2 ? data_type_t::f32 : data_type_t::bf16;
            c.bias_dt = v % 3 == 0 ? data_type_t::none
                    : v % 3 == 1   ? data_type_t::f32
                                   : data_type_t::bf16;
            c.transA = v % 5 == 1;
            c.transB = v % 4 == 2;
            c.lda_pad = v % 3 == 2 ? 5 : 0;
            c.ldc_pad = v % 2 ? 3 : 0;
            if (v % 6 == 4) {
                c.alpha = 0.5f;
                c.beta = 0.75f;
            }
            c.threads = v % 2 ? 8 : 1;
            c.seed = seed++;
            SCOPED_TRACE(c.str());
            e2e_data_t d = make_e2e(c);
            buf_t native = d.dst0, ref = d.dst0;
            ASSERT_EQ(run_native(c, d, native), status_t::success);
            ASSERT_EQ(run_reference(c, d, false, ref), status_t::success);
            const float abs = 1e-4f + 2e-7f * c.K;
            expect_native(c, d, native, [&](int m, int o) {
                return ref.get(static_cast<int64_t>(m) * c.ldc() + o);
            }, 1e-4f, abs);
        }
    }
}

TEST(NativeBrgemmEpilogueE2E, GatedVsReferenceAndGroupCall) {
    SKIP_WITHOUT_AVX512_BF16();
    // The one-op group call runs on the default route, which a process-wide
    // algorithm replaces for the whole process.
    const bool group_route = nt::default_route_reachable();
    if (!group_route)
        RecordProperty("group", "not_run (process-wide algorithm)");
    struct shape_t {
        int M, N, K;
    };
    const std::vector<shape_t> shapes
            = {{1, 64, 64}, {5, 256, 257}, {64, 128, 128}};
    uint32_t seed = 2000;
    int variant = 0;
    for (gated_t g : {gated_t::silu_and_mul, gated_t::gelu_and_mul,
                 gated_t::swiglu_oai_mul})
        for (const shape_t &s : shapes)
            for (data_type_t dst : {data_type_t::bf16, data_type_t::f32}) {
                const int v = variant++;
                e2e_t c;
                c.M = s.M;
                c.N = s.N;
                c.K = s.K;
                c.gated = g;
                c.dst_dt = dst;
                c.bias_dt = v % 3 == 0 ? data_type_t::none
                        : v % 3 == 1   ? data_type_t::bf16
                                       : data_type_t::f32;
                c.alpha = v % 4 == 3 ? 0.5f : 1.0f;
                c.transB = v % 2 == 1;
                c.ldc_pad = v % 3 == 1 ? 3 : 0;
                c.threads = v % 2 ? 8 : 1;
                if (v % 5 == 2)
                    c.po = {binary(post_op_type_t::binary_mul,
                                    data_type_t::bf16, {s.M, s.N}),
                            eltwise(post_op_type_t::clip, -0.5f, 0.5f)};
                c.seed = seed++;
                SCOPED_TRACE(c.str());
                e2e_data_t d = make_e2e(c);
                buf_t native = d.dst0;
                ASSERT_EQ(run_native(c, d, native), status_t::success);

                buf_t pre;
                pre.init(data_type_t::f32, static_cast<int64_t>(c.M) * c.N);
                ASSERT_EQ(run_reference(c, d, true, pre), status_t::success);
                // The post-ops act on the pre-activation.
                std::vector<float> wide(static_cast<size_t>(c.M) * c.N);
                for (size_t i = 0; i < wide.size(); ++i)
                    wide[i] = pre.get(static_cast<int64_t>(i));
                ref_postops(c.po, d.postops, d.bin, c.M, c.N, wide);
                const int I = c.N / 2;
                std::vector<float> model(static_cast<size_t>(c.M) * I);
                for (int m = 0; m < c.M; ++m)
                    for (int i = 0; i < I; ++i) {
                        const int gate = c.interleave() ? i : 2 * i;
                        const int up = c.interleave() ? I + i : 2 * i + 1;
                        model[static_cast<size_t>(m) * I + i] = ref_gated(g,
                                wide[static_cast<size_t>(m) * c.N + gate],
                                wide[static_cast<size_t>(m) * c.N + up]);
                    }
                const float abs = 1e-4f + 2e-7f * c.K;
                expect_native(c, d, native, [&](int m, int o) {
                    return model[static_cast<size_t>(m) * I + o];
                }, 1e-4f, abs);

                if (dst != data_type_t::bf16 || !c.po.empty() || !group_route)
                    continue;
                buf_t group;
                group.init(data_type_t::bf16, static_cast<int64_t>(c.M) * c.N);
                ASSERT_EQ(run_group(c, d, group), status_t::success);
                expect_native(c, d, native, [&](int m, int o) {
                    return group.get(static_cast<int64_t>(m) * c.N + o);
                }, 0.05f, 0.02f);
            }
}

// Through matmul_direct the gated result fills the first N / 2 columns of a
// full-width (ldc = N) or a half-width dst, the columns past it untouched.
TEST(NativeBrgemmEpilogueE2E, RoutedGatedOutputLayouts) {
    SKIP_WITHOUT_AVX512_BF16();
    uint32_t seed = 3000;
    for (gated_t g : {gated_t::silu_and_mul, gated_t::gelu_and_mul,
                 gated_t::swiglu_oai_mul})
        for (bool full_width : {true, false}) {
            e2e_t c;
            c.M = 5;
            c.N = 128;
            c.K = 96;
            c.gated = g;
            c.dst_dt = full_width ? data_type_t::bf16 : data_type_t::f32;
            c.bias_dt = data_type_t::f32;
            c.ldc_pad = full_width ? c.N / 2 : 0;
            c.threads = 4;
            c.seed = seed++;
            SCOPED_TRACE(c.str());
            e2e_data_t d = make_e2e(c);
            buf_t native = d.dst0;
            ASSERT_EQ(run_routed(c, d, native), status_t::success);
            buf_t direct = d.dst0;
            ASSERT_EQ(run_native(c, d, direct), status_t::success);
            EXPECT_EQ(native.bytes, direct.bytes)
                    << "not served by the native BRGEMM";

            buf_t pre;
            pre.init(data_type_t::f32, static_cast<int64_t>(c.M) * c.N);
            ASSERT_EQ(run_reference(c, d, true, pre), status_t::success);
            const int I = c.N / 2;
            std::vector<float> model(static_cast<size_t>(c.M) * I);
            for (int m = 0; m < c.M; ++m)
                for (int i = 0; i < I; ++i) {
                    const int gate = c.interleave() ? i : 2 * i;
                    const int up = c.interleave() ? I + i : 2 * i + 1;
                    model[static_cast<size_t>(m) * I + i] = ref_gated(g,
                            pre.get(static_cast<int64_t>(m) * c.N + gate),
                            pre.get(static_cast<int64_t>(m) * c.N + up));
                }
            expect_native(c, d, native, [&](int m, int o) {
                return model[static_cast<size_t>(m) * I + o];
            }, 1e-4f, 1e-4f + 2e-7f * c.K);
        }
}

/// The reference algorithm on the call without its gated act, into pre
/// (f32 [M, N], ldc N) holding the dst's first N columns: alpha, beta, the
/// bias and the post-ops of the pre-activation.
status_t run_reference_pre(const e2e_t &c, const e2e_data_t &d, buf_t &pre) {
    pre.init(data_type_t::f32, static_cast<int64_t>(c.M) * c.N);
    for (int m = 0; m < c.M; ++m)
        for (int n = 0; n < c.N; ++n)
            pre.set(static_cast<int64_t>(m) * c.N + n,
                    d.dst0.get(static_cast<int64_t>(m) * c.ldc() + n));
    mm::matmul_params p = bf16_params(data_type_t::f32, c.bias_dt);
    p.lowoha_algo = mm::matmul_algo_t::reference;
    p.postop_ = d.postops;
    mm::matmul_batch_params_t bp;
    return mm::matmul_direct('r', c.transA, c.transB, c.M, c.N, c.K, c.alpha,
            d.a.data(), c.lda(), d.w.data(), c.ldb(), d.bias_ptr(c), c.beta,
            pre.data(), c.N, false, bp, p);
}

/// The one-op group call with the call's gated act, beta and post-ops, into
/// dst (bf16 [M, N], ldc N) holding the dst's first N columns.
status_t run_group_with(const e2e_t &c, const e2e_data_t &d, buf_t &dst) {
    dst.init(data_type_t::bf16, static_cast<int64_t>(c.M) * c.N);
    for (int m = 0; m < c.M; ++m)
        for (int n = 0; n < c.N; ++n)
            dst.set(static_cast<int64_t>(m) * c.N + n,
                    d.dst0.get(static_cast<int64_t>(m) * c.ldc() + n));
    mm::matmul_params p = bf16_params(data_type_t::bf16, c.bias_dt);
    p.num_threads = c.threads;
    p.postop_ = d.postops;
    mm::grp_matmul_gated_act_params act;
    act.act = static_cast<mm::grp_matmul_gated_act_t>(
            static_cast<int>(c.gated));
    return mm::group_matmul_direct({'r'}, {c.transA}, {c.transB}, {c.M}, {c.N},
            {c.K}, {c.alpha}, {static_cast<const void *>(d.a.data())},
            {c.lda()}, {static_cast<const void *>(d.w.data())}, {c.ldb()},
            {d.bias_ptr(c)}, {c.beta}, {dst.data()}, {c.N}, {false}, {p},
            nullptr, &act, nullptr);
}

// Gated calls with beta and post-ops through matmul_direct, as it defines
// them: alpha, beta over the dst's N pre-activation columns, the bias and
// the post-ops of the pre-activation, then the activation into the first
// N / 2 columns.  The native BRGEMM serves them (the bits of the call on
// its own); against the reference algorithm's pre-activation and the
// activations of group_matmul_moe_act.cpp within their band, and against
// the one-op group route, which rounds the pre-activation to the bf16 dst
// before its activation pass, within that rounding (post-ops it also
// applies), on the default route, which a process-wide algorithm replaces
// for the whole process.
TEST(NativeBrgemmEpilogueE2E, RoutedGatedBetaAndPostOps) {
    SKIP_WITHOUT_AVX512_BF16();
    const bool group_route = nt::default_route_reachable();
    if (!group_route)
        RecordProperty("group", "not_run (process-wide algorithm)");
    uint32_t seed = 3200;
    int variant = 0;
    for (gated_t g : {gated_t::silu_and_mul, gated_t::gelu_and_mul,
                 gated_t::swiglu_oai_mul})
        for (int k = 0; k < 6; ++k) {
            const int v = variant++;
            e2e_t c;
            c.M = k == 5 ? 64 : 5;
            c.N = 128;
            c.K = k == 5 ? 300 : 96;
            c.gated = g;
            c.dst_dt = k % 2 ? data_type_t::f32 : data_type_t::bf16;
            c.bias_dt = k % 3 == 0 ? data_type_t::f32 : data_type_t::bf16;
            c.ldc_pad = c.N / 2 + (k == 1 ? 5 : 0);
            c.threads = v % 2 ? 8 : 1;
            c.transB = k == 1;
            bool group_applies = true;
            switch (k) {
                case 0: c.beta = 0.5f; break;
                case 1:
                    c.beta = -1.0f;
                    c.alpha = 0.5f;
                    break;
                case 2:
                    c.po = {eltwise(post_op_type_t::gelu_erf),
                            binary(post_op_type_t::binary_add,
                                    data_type_t::bf16, {c.M, c.N})};
                    break;
                case 3:
                    c.beta = 0.25f;
                    c.po = {binary(post_op_type_t::binary_mul, data_type_t::f32,
                                    {1, c.N}),
                            eltwise(post_op_type_t::relu)};
                    break;
                case 4:
                    c.po = {eltwise(post_op_type_t::elu, 0.5f)};
                    group_applies = false;
                    break;
                default:
                    c.beta = 1.0f;
                    c.po = {eltwise(post_op_type_t::leaky_relu, 0.1f)};
                    break;
            }
            c.seed = seed++;
            SCOPED_TRACE(c.str());
            e2e_data_t d = make_e2e(c);
            buf_t native = d.dst0;
            ASSERT_EQ(run_routed(c, d, native), status_t::success);
            buf_t direct = d.dst0;
            ASSERT_EQ(run_native(c, d, direct), status_t::success);
            EXPECT_EQ(native.bytes, direct.bytes)
                    << "not served by the native BRGEMM";

            buf_t pre;
            ASSERT_EQ(run_reference_pre(c, d, pre), status_t::success);
            const int I = c.N / 2;
            std::vector<float> model(static_cast<size_t>(c.M) * I);
            for (int m = 0; m < c.M; ++m)
                for (int i = 0; i < I; ++i) {
                    const int gate = c.interleave() ? i : 2 * i;
                    const int up = c.interleave() ? I + i : 2 * i + 1;
                    model[static_cast<size_t>(m) * I + i] = ref_gated(g,
                            pre.get(static_cast<int64_t>(m) * c.N + gate),
                            pre.get(static_cast<int64_t>(m) * c.N + up));
                }
            expect_native(c, d, native, [&](int m, int o) {
                return model[static_cast<size_t>(m) * I + o];
            }, 1e-4f, 1e-4f + 2e-7f * c.K);

            if (c.dst_dt != data_type_t::bf16 || !group_applies || !group_route)
                continue;
            buf_t group;
            ASSERT_EQ(run_group_with(c, d, group), status_t::success);
            expect_native(c, d, native, [&](int m, int o) {
                return group.get(static_cast<int64_t>(m) * c.N + o);
            }, 0.05f, 0.02f);
        }
}

// A gated call with post-ops, beta over a dst of N columns or more, or an
// f16 dst is the native BRGEMM's, with the bits of the call on its own; with
// beta over a narrower dst (which the one-op group route refuses too) it
// returns unimplemented, and dst untouched.
TEST(NativeBrgemmEpilogueE2E, GatedCallsTheNativeBrgemmServes) {
    SKIP_WITHOUT_AVX512_BF16();
    for (int variant = 0; variant < 4; ++variant) {
        e2e_t c;
        c.M = 5;
        c.N = 128;
        c.K = 96;
        c.gated = gated_t::silu_and_mul;
        c.dst_dt = variant == 2 ? data_type_t::f16 : data_type_t::bf16;
        if (variant == 0) c.po = {eltwise(post_op_type_t::relu)};
        if (variant == 1 || variant == 3) c.beta = 0.5f;
        if (variant == 1) c.ldc_pad = c.N / 2;
        c.seed = 3100 + variant;
        SCOPED_TRACE(c.str());
        e2e_data_t d = make_e2e(c);
        buf_t dst = d.dst0;
        mm::matmul_params p = bf16_params(c.dst_dt, c.bias_dt);
        p.postop_ = d.postops;
        p.lowoha_algo = mm::matmul_algo_t::native_brgemm;
        p.weight_cache_type = 0;
        status_t st = status_t::success;
        const bool served = nat::native_layout_matmul('r', c.transA, c.transB,
                c.M, c.N, c.K, c.alpha, d.a.data(), c.lda(), d.w.data(),
                c.ldb(), d.bias_ptr(c), c.beta, dst.data(), c.ldc(), true, 1,
                c.threads, c.gated, p, nullptr, st);
        EXPECT_TRUE(served);
        if (variant == 3) {
            EXPECT_EQ(st, status_t::unimplemented);
            for (int64_t i = 0; i < static_cast<int64_t>(c.M) * c.ldc(); ++i)
                ASSERT_EQ(dst.get(i), d.dst0.get(i)) << "element " << i;
            continue;
        }
        EXPECT_EQ(st, status_t::success);
        buf_t direct = d.dst0;
        ASSERT_EQ(run_native(c, d, direct), status_t::success);
        EXPECT_EQ(dst.bytes, direct.bytes);
    }
}

// ---- end to end through native_brgemm_execute (static INT8) -----------------

struct int8_case_t {
    int M = 8;
    int N = 64;
    int K = 64;
    data_type_t src_dt = data_type_t::u8;
    data_type_t dst_dt = data_type_t::u8;
    data_type_t bias_dt = data_type_t::none;
    bool per_channel = true;
    std::vector<po_spec_t> po;
    int threads = 1;
    uint32_t seed = 1;

    bool requant() const {
        return dst_dt == data_type_t::u8 || dst_dt == data_type_t::s8;
    }
    std::string str() const {
        std::ostringstream os;
        os << "M=" << M << " N=" << N << " K=" << K
           << " src=" << zendnnl::common::dtype_info(src_dt)
           << " dst=" << zendnnl::common::dtype_info(dst_dt)
           << " bias=" << zendnnl::common::dtype_info(bias_dt)
           << " per_channel=" << per_channel << " threads=" << threads
           << " po=";
        for (const auto &s : po)
            os << static_cast<int>(s.type) << ",";
        return os.str();
    }
};

/// Static INT8 through native_brgemm_execute and through the reference
/// algorithm on the same quantized operands; both outputs land in native / ref.
void run_int8_pair(const int8_case_t &c, buf_t &native, buf_t &ref) {
    std::mt19937 rng(c.seed);
    buf_t src, wei, wscale, bias;
    src.init(c.src_dt, static_cast<int64_t>(c.M) * c.K);
    src.fill(rng, c.src_dt == data_type_t::u8 ? 0.0f : -128.0f,
            c.src_dt == data_type_t::u8 ? 255.0f : 127.0f);
    wei.init(data_type_t::s8, static_cast<int64_t>(c.K) * c.N);
    wei.fill(rng, -127.0f, 127.0f);
    wscale.init(data_type_t::f32, c.per_channel ? c.N : 1);
    wscale.fill(rng, 0.005f, 0.015f);
    if (c.bias_dt != data_type_t::none) {
        bias.init(c.bias_dt, c.N);
        bias.fill(rng, -1.0f, 1.0f);
    }
    const float src_scale = 0.01f, dst_scale = 12.0f;
    const int32_t src_zp = 120, dst_zp = c.dst_dt == data_type_t::u8 ? 128 : 0;
    std::vector<buf_t> bin;
    std::vector<mm::matmul_post_op> postops;
    make_postops(c.po, c.M, c.N, rng, bin, postops);

    mm::matmul_params p;
    p.dtypes.src = c.src_dt;
    p.dtypes.wei = data_type_t::s8;
    p.dtypes.dst = c.dst_dt;
    p.dtypes.bias = c.bias_dt;
    p.postop_ = postops;
    auto &q = p.quant_params;
    q.src_scale.buff = &src_scale;
    q.src_scale.dt = data_type_t::f32;
    q.src_scale.dims = {1, 1};
    if (c.src_dt == data_type_t::u8) {
        q.src_zp.buff = &src_zp;
        q.src_zp.dt = data_type_t::s32;
        q.src_zp.dims = {1, 1};
    }
    q.wei_scale.buff = wscale.data();
    q.wei_scale.dt = data_type_t::f32;
    q.wei_scale.dims = {1, c.per_channel ? c.N : 1};
    if (c.requant()) {
        q.dst_scale.buff = &dst_scale;
        q.dst_scale.dt = data_type_t::f32;
        q.dst_scale.dims = {1, 1};
        q.dst_zp.buff = &dst_zp;
        q.dst_zp.dt = data_type_t::s32;
        q.dst_zp.dims = {1, 1};
    }
    const void *b = c.bias_dt == data_type_t::none ? nullptr : bias.data();
    native.init(c.dst_dt, static_cast<int64_t>(c.M) * c.N);
    ref.init(c.dst_dt, static_cast<int64_t>(c.M) * c.N);

    nat::GemmDescriptor desc;
    desc.M = c.M;
    desc.N = c.N;
    desc.K = c.K;
    desc.lda = c.K;
    desc.ldb = c.N;
    desc.ldc = c.N;
    desc.src_dt = c.src_dt;
    desc.wei_dt = data_type_t::s8;
    desc.dst_dt = c.dst_dt;
    desc.bias_dt = c.bias_dt;
    desc.src_elem_size = 1;
    desc.wei_elem_size = 1;
    desc.dst_elem_size = size_of(c.dst_dt);
    desc.bias = b;
    desc.is_weights_const = true;
    desc.num_threads = c.threads;
    nat::native_weight_ref_t wref;
    wref.weight = wei.data();
    wref.pack_nr = 32;
    wref.weight_cache_mode = 0;
    mm::matmul_params pn = p;
    ASSERT_EQ(nat::native_brgemm_execute(
                      desc, wref, src.data(), native.data(), b, pn),
            status_t::success);

    mm::matmul_params pr = p;
    pr.lowoha_algo = mm::matmul_algo_t::reference;
    mm::matmul_batch_params_t bp;
    ASSERT_EQ(mm::matmul_direct('r', false, false, c.M, c.N, c.K, 1.0f,
                      src.data(), c.K, wei.data(), c.N, b, 0.0f, ref.data(),
                      c.N, false, bp, pr),
            status_t::success);
}

TEST(NativeBrgemmEpilogueE2E, Int8RequantPostOpsVsReference) {
    SKIP_WITHOUT_AVX512();
    struct shape_t {
        int M, N, K;
    };
    const std::vector<shape_t> shapes
            = {{1, 64, 64}, {9, 96, 100}, {70, 128, 37}};
    const std::vector<data_type_t> dsts = {data_type_t::u8, data_type_t::bf16,
            data_type_t::s8, data_type_t::f32};
    uint32_t seed = 3000;
    int variant = 0;
    for (const shape_t &s : shapes) {
        const std::vector<std::vector<po_spec_t>> chains
                = {{}, {eltwise(post_op_type_t::relu)},
                        {eltwise(post_op_type_t::sigmoid)},
                        {binary(post_op_type_t::binary_mul, data_type_t::f32,
                                 {s.M, s.N}),
                                binary(post_op_type_t::binary_add,
                                        data_type_t::f32, {s.M, s.N})},
                        {eltwise(post_op_type_t::gelu_tanh),
                                eltwise(post_op_type_t::clip, -1.0f, 1.0f)},
                        {eltwise(post_op_type_t::swish, 0.5f),
                                binary(post_op_type_t::binary_add,
                                        data_type_t::bf16, {1, s.N})},
                        {eltwise(post_op_type_t::softmax)}};
        for (const auto &chain : chains) {
            const int v = variant++;
            int8_case_t c;
            c.M = s.M;
            c.N = s.N;
            c.K = s.K;
            c.po = chain;
            c.dst_dt = dsts[v % dsts.size()];
            c.src_dt = v % 3 == 2 ? data_type_t::s8 : data_type_t::u8;
            c.bias_dt = v % 3 == 0 ? data_type_t::none
                    : v % 3 == 1   ? data_type_t::f32
                                   : data_type_t::bf16;
            c.per_channel = v % 5 != 4;
            c.threads = v % 2 ? 8 : 1;
            c.seed = seed++;
            SCOPED_TRACE(c.str());
            buf_t native, ref;
            run_int8_pair(c, native, ref);
            int bad = 0;
            for (int64_t i = 0; i < static_cast<int64_t>(c.M) * c.N; ++i) {
                const float g = native.get(i), r = ref.get(i);
                float tol = 1e-4f * (1.0f + std::fabs(r));
                if (c.dst_dt == data_type_t::bf16) tol += std::fabs(r) / 128.0f;
                if (c.requant()) tol = 1.0f;
                if (!(std::fabs(g - r) <= tol) && bad++ < 8)
                    ADD_FAILURE() << "element " << i << " native " << g
                                  << " ref " << r;
            }
            EXPECT_EQ(bad, 0);
        }
    }
}

// ---- activations of extreme inputs (BF16 and INT8 tiles) --------------------

/// native_brgemm_execute over a zero [64, N] weight, bf16 or s8 (u8 source
/// with a zero point), with an f32 bias of N values and an f32 dst: the
/// activation sees every bias value exactly.
status_t run_zero_weight(bool int8, gated_t gated,
        const std::vector<mm::matmul_post_op> &postops,
        const std::vector<float> &bias, int M, std::vector<float> &dst) {
    constexpr int K = 64;
    const int N = static_cast<int>(bias.size());
    const int n_out = gated == gated_t::none ? N : N / 2;
    std::vector<bfloat16_t> a_bf16(static_cast<size_t>(M) * K);
    std::vector<uint8_t> a_u8(a_bf16.size());
    for (size_t i = 0; i < a_bf16.size(); ++i) {
        a_bf16[i] = bfloat16_t(static_cast<float>(i % 7) - 3.0f);
        a_u8[i] = static_cast<uint8_t>(i % 251);
    }
    const std::vector<bfloat16_t> w_bf16(
            static_cast<size_t>(K) * N, bfloat16_t(0.0f));
    const std::vector<int8_t> w_s8(static_cast<size_t>(K) * N, 0);
    const float src_scale = 0.5f;
    const int32_t src_zp = 3;
    const std::vector<float> wei_scale(static_cast<size_t>(N), 0.25f);

    mm::matmul_params p;
    p.dtypes.src = int8 ? data_type_t::u8 : data_type_t::bf16;
    p.dtypes.wei = int8 ? data_type_t::s8 : data_type_t::bf16;
    p.dtypes.dst = data_type_t::f32;
    p.dtypes.bias = data_type_t::f32;
    p.postop_ = postops;
    if (int8) {
        auto &q = p.quant_params;
        q.src_scale.buff = &src_scale;
        q.src_scale.dt = data_type_t::f32;
        q.src_scale.dims = {1, 1};
        q.src_zp.buff = &src_zp;
        q.src_zp.dt = data_type_t::s32;
        q.src_zp.dims = {1, 1};
        q.wei_scale.buff = wei_scale.data();
        q.wei_scale.dt = data_type_t::f32;
        q.wei_scale.dims = {1, N};
    }
    nat::GemmDescriptor desc;
    desc.M = M;
    desc.N = N;
    desc.K = K;
    desc.lda = K;
    desc.ldb = N;
    desc.ldc = n_out;
    desc.src_dt = p.dtypes.src;
    desc.wei_dt = p.dtypes.wei;
    desc.dst_dt = data_type_t::f32;
    desc.bias_dt = data_type_t::f32;
    desc.src_elem_size = int8 ? 1 : 2;
    desc.wei_elem_size = int8 ? 1 : 2;
    desc.dst_elem_size = 4;
    desc.bias = bias.data();
    desc.is_weights_const = true;
    desc.num_threads = 4;
    desc.gated_act = gated;
    nat::native_weight_ref_t ref;
    ref.weight = int8 ? static_cast<const void *>(w_s8.data())
                      : static_cast<const void *>(w_bf16.data());
    ref.pack_nr = 32;
    ref.interleave_split_halves
            = gated == gated_t::silu_and_mul || gated == gated_t::gelu_and_mul;
    ref.weight_cache_mode = 0;
    dst.assign(static_cast<size_t>(M) * n_out, -777.0f);
    const void *src = int8 ? static_cast<const void *>(a_u8.data())
                           : static_cast<const void *>(a_bf16.data());
    return nat::native_brgemm_execute(
            desc, ref, src, dst.data(), bias.data(), p);
}

/// got against the reference want: the same bits where want is a zero or
/// an infinity, NaN where it is NaN, within the f32 bound elsewhere.
bool same_or_near(float got, float want) {
    if (std::isnan(want)) return std::isnan(got);
    if (want == 0.0f || std::isinf(want)) {
        uint32_t g = 0, w = 0;
        std::memcpy(&g, &got, sizeof(g));
        std::memcpy(&w, &want, sizeof(w));
        return g == w;
    }
    return std::fabs(got - want) <= 3e-4f + 3e-4f * std::fabs(want);
}

// The activations of the epilogue that go through an exponential the vector
// code clamps, of extreme inputs given as an f32 bias over a zero GEMM sum,
// through the BF16 and the INT8 tile: NaN, infinities and values past both
// clamps.  The clamps must not leave x times a floor (mish, swish, silu),
// saturate the exp post-op, turn a NaN into a number (sigmoid, tanh, elu,
// exp, swish of a NaN alpha * x), or send a NaN swiglu gate or up to the
// other bound.  gelu does not multiply x by the clamped exponential.  Where
// the reference gives a zero, an infinity or NaN the output is the same,
// sign included; elsewhere it is within the f32 bound.
TEST(NativeBrgemmEpilogueE2E, ExtremeInputsOfExponentialActivations) {
    SKIP_WITHOUT_AVX512();
    const float inf = std::numeric_limits<float>::infinity();
    const std::vector<float> inputs = {-1e38f, -1e30f, -104.0f, -100.0f, -88.8f,
            -88.5f, -88.2f, -87.5f, -87.2f, -44.5f, -20.0f, -3.0f, -0.5f, 0.0f,
            0.5f, 3.0f, 20.0f, 44.5f, 87.2f, 87.5f, 88.2f, 88.5f, 88.8f, 100.0f,
            1e30f, 1e38f, -inf, inf, kNan, 7.5f, -7.5f, 1e-30f};
    const std::vector<float> ups = {2.0f, -0.75f, 1.0f, 0.25f};
    // The swiglu gates of the outputs whose up is an input: the NaN meets the
    // NaN input.
    const std::vector<float> gates = {kNan, -3.0f, 0.5f, 6.5f};
    struct act_t {
        const char *name;
        gated_t gated;
        post_op_type_t type;
        float alpha;
    };
    const std::vector<act_t> acts = {
            {"mish", gated_t::none, post_op_type_t::mish, kNan},
            {"swish", gated_t::none, post_op_type_t::swish, kNan},
            {"swish 2.5", gated_t::none, post_op_type_t::swish, 2.5f},
            {"swish -1.5", gated_t::none, post_op_type_t::swish, -1.5f},
            {"swish 0", gated_t::none, post_op_type_t::swish, 0.0f},
            {"sigmoid", gated_t::none, post_op_type_t::sigmoid, kNan},
            {"tanh", gated_t::none, post_op_type_t::tanh, kNan},
            {"elu", gated_t::none, post_op_type_t::elu, kNan},
            {"elu 0.5", gated_t::none, post_op_type_t::elu, 0.5f},
            {"exp", gated_t::none, post_op_type_t::exp, kNan},
            {"silu_and_mul", gated_t::silu_and_mul, post_op_type_t::none, kNan},
            {"gelu_and_mul", gated_t::gelu_and_mul, post_op_type_t::none, kNan},
            {"swiglu_oai_mul", gated_t::swiglu_oai_mul, post_op_type_t::none,
                    kNan}};
    constexpr int M = 3;
    const int count = static_cast<int>(inputs.size());
    for (bool int8 : {false, true}) {
        if (!int8 && !bf16_cpu_ok()) continue;
        for (const act_t &a : acts) {
            SCOPED_TRACE(std::string(int8 ? "int8 " : "bf16 ") + a.name);
            const bool gated = a.gated != gated_t::none;
            const bool swiglu = a.gated == gated_t::swiglu_oai_mul;
            // Every input as x or as a gate; for swiglu also as an up.
            const int n_out = gated && !swiglu ? count : 2 * count;
            std::vector<float> gate(n_out), up(n_out);
            for (int o = 0; o < n_out; ++o) {
                const bool up_input = swiglu && o >= count;
                gate[o] = up_input ? gates[o % gates.size()]
                                   : inputs[o % count];
                up[o] = up_input ? inputs[o - count] : ups[o % ups.size()];
            }
            // Split halves for silu and gelu, interleaved for swiglu.
            const int N = gated ? 2 * n_out : n_out;
            std::vector<float> bias(N);
            for (int o = 0; o < n_out; ++o) {
                if (!gated) {
                    bias[o] = gate[o];
                    continue;
                }
                bias[swiglu ? 2 * o : o] = gate[o];
                bias[swiglu ? 2 * o + 1 : n_out + o] = up[o];
            }
            std::vector<mm::matmul_post_op> postops;
            if (!gated) {
                mm::matmul_post_op po;
                po.po_type = a.type;
                po.alpha = a.alpha;
                postops.push_back(po);
            }
            std::vector<float> dst;
            ASSERT_EQ(run_zero_weight(int8, a.gated, postops, bias, M, dst),
                    status_t::success);
            int bad = 0;
            for (int m = 0; m < M; ++m)
                for (int o = 0; o < n_out; ++o) {
                    const float want = gated
                            ? ref_gated(a.gated, gate[o], up[o])
                            : ref_eltwise(a.type, gate[o],
                                      postops[0].alpha_or_default(), 0.0f);
                    const float got = dst[static_cast<size_t>(m) * n_out + o];
                    if (!same_or_near(got, want) && bad++ < 8)
                        ADD_FAILURE()
                                << "input " << gate[o] << (gated ? ", up " : "")
                                << (gated ? std::to_string(up[o]) : "") << ": "
                                << got << ", the reference " << want;
                }
            EXPECT_EQ(bad, 0);
        }
    }
}

} // namespace
