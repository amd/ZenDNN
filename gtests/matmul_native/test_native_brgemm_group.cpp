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

/// ALGO 10 selects the native BRGEMM of matmul_direct only.
/// group_matmul_direct (each scheduling algorithm with the custom kernel off
/// and on, the per-expert slice, the sequential chain, fused MoE) writes
/// under a pinned ZENDNNL_MATMUL_ALGO=10, and with lowoha_algo =
/// native_brgemm on its ops, the bits it writes with the algorithm unset,
/// and the native BRGEMM serves none of its calls.  INT8 experts (u8 / s8
/// sources, per-token and per-tensor source scales in f32 and bf16, zero
/// points, per-channel weight scales, bias, a binary post-op) are also
/// checked against an integer reference, BF16 experts against a double one.
///
/// A matmul_direct call that asks for the native BRGEMM and that it does not
/// serve returns its error, with dst untouched and its weight not rewritten
/// in place; a call on a weight whose N is not a multiple of 32 runs on it
/// over a padded layout packed out of place.  The same calls with the
/// algorithm unset run on AOCL DLP.

#include <gtest/gtest.h>

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "group_matmul/moe_test_utils.hpp"
#include "gtest_native_refusal.hpp"
#include "lowoha_operators/matmul/group_matmul/common/grp_matmul_expert_slice.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "lowoha_operators/matmul/matmul_native/native_matmul.hpp"
#include "matmul_native/native_brgemm_test_utils.hpp"

namespace {

namespace mm = zendnnl::lowoha::matmul;
namespace nat = zendnnl::lowoha::matmul::native;
namespace mt = moe_test_utils;
namespace nbt = native_brgemm_test;
using zendnnl::common::data_type_t;
using zendnnl::common::dtype_info;
using zendnnl::common::matmul_algo_t;
using zendnnl::common::matmul_config_t;
using zendnnl::common::post_op_type_t;
using zendnnl::error_handling::status_t;

bool cpu_ok() {
    const nat::UarchParams &u = nat::detect_uarch();
    return u.avx512vnni && u.avx512bf16 && nat::native_base_isa();
}

#define SKIP_WITHOUT_NATIVE_ISA() \
    if (!cpu_ok()) GTEST_SKIP() << "needs AVX512-VNNI/BF16/BW/VL/DQ and FMA"

int all_threads() {
    return std::max(1, omp_get_max_threads());
}

/// The kernel group matmul resolves with ZENDNNL_MATMUL_ALGO unset.
matmul_algo_t unset_kernel() {
#if ZENDNNL_DEPENDS_AOCLDLP
    return matmul_algo_t::aocl_dlp_blocked;
#else
    return matmul_algo_t::reference;
#endif
}

/// How a run selects the algorithm: unset, a pinned ZENDNNL_MATMUL_ALGO=10,
/// or lowoha_algo = native_brgemm on every op with the algorithm unset.
enum class algo_mode_t { unset, algo10, per_call };

constexpr algo_mode_t kModes[]
        = {algo_mode_t::unset, algo_mode_t::algo10, algo_mode_t::per_call};

const char *mode_name(algo_mode_t m) {
    return m == algo_mode_t::algo10      ? "ZENDNNL_MATMUL_ALGO=10"
            : m == algo_mode_t::per_call ? "lowoha_algo=native_brgemm"
                                         : "unset";
}

matmul_algo_t pinned_algo(algo_mode_t m) {
    return m == algo_mode_t::algo10 ? matmul_algo_t::native_brgemm
                                    : matmul_algo_t::none;
}

/// Absent, one per tensor, or one per source row.
enum class grain_t { none, tensor, token };

const char *grain_name(grain_t g) {
    return g == grain_t::token     ? "token"
            : g == grain_t::tensor ? "tensor"
                                   : "none";
}

/// One expert: u8 / s8 x s8 (static INT8, or a bf16 source quantized in the
/// call), bf16 x bf16 or f32 x f32.
struct case_t {
    int M = 8;
    int N = 64;
    int K = 96;
    bool transB = false;
    data_type_t src_dt = data_type_t::u8;
    data_type_t wei_dt = data_type_t::s8;
    /// A bf16 source quantized to s8 per token in the call.
    bool dynamic = false;
    grain_t scale = grain_t::token;
    data_type_t scale_dt = data_type_t::f32;
    grain_t zp = grain_t::none;
    data_type_t zp_dt = data_type_t::s32;
    /// none: no bias.
    data_type_t bias_dt = data_type_t::none;
    /// none, binary_add or binary_mul over an f32 operand.
    post_op_type_t binary = post_op_type_t::none;
    /// The operand is one row ({1, N}) broadcast over the rows, else [M, N].
    bool binary_row = false;
    data_type_t dst_dt = data_type_t::f32;
    uint32_t seed = 1;
    /// wei_buffer_capacity_bytes of the call, and at least the bytes of the
    /// weight buffer.
    size_t capacity = 0;

    bool int8() const { return wei_dt == data_type_t::s8; }

    std::string str() const {
        std::ostringstream os;
        os << "M=" << M << " N=" << N << " K=" << K << " transB=" << transB
           << " src=" << dtype_info(src_dt) << " wei=" << dtype_info(wei_dt)
           << (dynamic ? " dynamic" : "");
        if (int8())
            os << " scale=" << grain_name(scale) << "/" << dtype_info(scale_dt)
               << " zp=" << grain_name(zp) << "/" << dtype_info(zp_dt);
        os << " bias=" << dtype_info(bias_dt) << " binary="
           << (binary == post_op_type_t::binary_add                  ? "add"
                              : binary == post_op_type_t::binary_mul ? "mul"
                                                                     : "none")
           << (binary_row ? "/row" : "") << " dst=" << dtype_info(dst_dt)
           << " seed=" << seed;
        return os.str();
    }
};

/// An expert's operands, the matmul params describing them, the reference
/// of every output and its bound.
struct expert_t {
    case_t c;
    nbt::buffer_t src, wei, src_scale, src_zp, wei_scale, bias, binary, dst;
    std::vector<uint8_t> wei_bytes;
    std::vector<double> ref, bound;
    mm::matmul_params params;

    int lda() const { return c.K; }
    int ldb() const { return c.transB ? c.K : c.N; }
    size_t outputs() const { return static_cast<size_t>(c.M) * c.N; }
    size_t dst_bytes() const { return outputs() * nbt::dt_bytes(c.dst_dt); }
};

/// The experts of a test, kept allocated to its end: a buffer freed and
/// allocated again at the same address would meet what the caches of the
/// first one left.
using alive_t = std::vector<expert_t>;

constexpr float kUnwritten = 4096.0f;

void fill_dst(expert_t &e) {
    const data_type_t dt = e.c.dst_dt;
    if (dt == data_type_t::u8 || dt == data_type_t::s8) {
        std::memset(e.dst.get(), 0x5a, e.outputs());
        return;
    }
    for (size_t i = 0; i < e.outputs(); ++i)
        nbt::store_f32(e.dst.get(), dt, i, kUnwritten);
}

/// n values of dt drawn from [lo, hi], with room for n f32 values: a reader
/// that takes them for f32 stays inside the allocation.
nbt::buffer_t values(
        data_type_t dt, int64_t n, uint32_t seed, float lo, float hi) {
    const nbt::buffer_t v = nbt::random_buffer(dt, n, seed, lo, hi);
    nbt::buffer_t b = nbt::make_buffer(static_cast<size_t>(n) * 4);
    std::memcpy(b.get(), v.get(), static_cast<size_t>(n) * nbt::dt_bytes(dt));
    return b;
}

void set_quant(mm::matmul_quantization_params_t::matmul_quant_t &q,
        const nbt::buffer_t &b, data_type_t dt, std::vector<int64_t> dims) {
    q.buff = b.get();
    q.dt = dt;
    q.dims = std::move(dims);
}

expert_t make_expert(const case_t &c) {
    expert_t e;
    e.c = c;
    const size_t mk = static_cast<size_t>(c.M) * c.K;
    const size_t kn = static_cast<size_t>(c.K) * c.N;
    const size_t held = std::max(kn, c.capacity / nbt::dt_bytes(c.wei_dt));
    if (c.src_dt == data_type_t::u8)
        e.src = nbt::random_buffer(c.src_dt, mk, c.seed, 0.0f, 255.0f);
    else if (c.src_dt == data_type_t::s8)
        e.src = nbt::random_buffer(c.src_dt, mk, c.seed, -128.0f, 127.0f);
    else
        e.src = nbt::random_buffer(c.src_dt, mk, c.seed, -1.0f, 1.0f);
    if (c.int8())
        e.wei = nbt::kept_until_exit(nbt::random_buffer(
                c.wei_dt, held, c.seed + 1, -128.0f, 127.0f));
    else
        e.wei = nbt::kept_until_exit(
                nbt::random_buffer(c.wei_dt, held, c.seed + 1, -0.5f, 0.5f));
    e.wei_bytes.assign(e.wei.as<uint8_t>(),
            e.wei.as<uint8_t>() + kn * nbt::dt_bytes(c.wei_dt));

    mm::matmul_params &p = e.params;
    p.wei_buffer_capacity_bytes = c.capacity;
    p.dtypes.src = c.src_dt;
    p.dtypes.wei = c.wei_dt;
    p.dtypes.dst = c.dst_dt;
    p.dtypes.bias = c.bias_dt;
    auto &qp = p.quant_params;
    if (c.int8()) {
        if (c.dynamic) {
            // The quantization writes the per-token scales here.
            p.dynamic_quant = true;
            p.dtypes.compute = data_type_t::s8;
            e.src_scale = nbt::make_buffer(static_cast<size_t>(c.M) * 4);
            set_quant(qp.src_scale, e.src_scale, c.scale_dt, {c.M, 1});
        } else if (c.scale != grain_t::none) {
            const bool token = c.scale == grain_t::token;
            e.src_scale = values(
                    c.scale_dt, token ? c.M : 1, c.seed + 2, 0.002f, 0.02f);
            set_quant(qp.src_scale, e.src_scale, c.scale_dt,
                    token ? std::vector<int64_t> {c.M, 1}
                          : std::vector<int64_t> {1});
        }
        e.wei_scale = values(c.scale_dt, c.N, c.seed + 3, 0.001f, 0.01f);
        set_quant(qp.wei_scale, e.wei_scale, c.scale_dt, {1, c.N});
        if (c.zp != grain_t::none) {
            const bool token = c.zp == grain_t::token;
            const bool u8 = c.src_dt == data_type_t::u8;
            e.src_zp = values(c.zp_dt, token ? c.M : 1, c.seed + 4,
                    u8 ? 96.0f : -24.0f, u8 ? 160.0f : 24.0f);
            set_quant(qp.src_zp, e.src_zp, c.zp_dt,
                    token ? std::vector<int64_t> {c.M, 1}
                          : std::vector<int64_t> {1});
        }
    }
    if (c.bias_dt != data_type_t::none)
        e.bias = values(c.bias_dt, c.N, c.seed + 5, -2.0f, 2.0f);
    if (c.binary != post_op_type_t::none) {
        const int rows = c.binary_row ? 1 : c.M;
        const size_t count = static_cast<size_t>(rows) * c.N;
        const bool mul = c.binary == post_op_type_t::binary_mul;
        e.binary = nbt::random_buffer(data_type_t::f32, count, c.seed + 6,
                mul ? 0.5f : -1.0f, mul ? 1.5f : 1.0f);
        mm::matmul_post_op po;
        po.po_type = c.binary;
        po.buff = e.binary.get();
        po.dtype = data_type_t::f32;
        po.dims = {rows, c.N};
        po.leading_dim = c.N;
        p.postop_.push_back(po);
    }
    e.dst = nbt::make_buffer(e.dst_bytes());
    fill_dst(e);
    return e;
}

/// The reference of e over the source at src (rows lda apart), and its
/// bound.  INT8: the int64 sum of (a - zero point) * w, then the scales;
/// otherwise the double sum.  Then the bias and the binary post-op.  The
/// bound covers the f32 rounding of each step and of the sum, and the
/// rounding of a bf16 dst.  The weight is read from e.wei_bytes: a call at
/// weight cache mode 2 may have rewritten its buffer in place.
void compute_reference(expert_t &e, const void *src, int lda) {
    const case_t &c = e.c;
    e.ref.assign(e.outputs(), 0.0);
    e.bound.assign(e.outputs(), 0.0);
    for (int m = 0; m < c.M; ++m) {
        int64_t zp = 0;
        double ss = 1.0;
        if (c.int8() && c.zp != grain_t::none)
            zp = static_cast<int64_t>(nbt::load_f32(
                    e.src_zp.get(), c.zp_dt, c.zp == grain_t::token ? m : 0));
        if (c.int8() && c.scale != grain_t::none)
            ss = nbt::load_f32(e.src_scale.get(), c.scale_dt,
                    c.scale == grain_t::token ? m : 0);
        for (int n = 0; n < c.N; ++n) {
            double v = 0.0;
            double mag = 0.0;
            if (c.int8()) {
                int64_t acc = 0;
                for (int k = 0; k < c.K; ++k) {
                    const size_t wi = c.transB
                            ? static_cast<size_t>(n) * c.K + k
                            : static_cast<size_t>(k) * c.N + n;
                    const int64_t a = static_cast<int64_t>(nbt::load_f32(
                            src, c.src_dt, static_cast<size_t>(m) * lda + k));
                    acc += (a - zp)
                            * static_cast<int64_t>(nbt::load_f32(
                                    e.wei_bytes.data(), c.wei_dt, wi));
                }
                v = static_cast<double>(acc) * ss
                        * nbt::load_f32(e.wei_scale.get(), c.scale_dt, n);
                mag = 1e-5 * std::fabs(v);
            } else {
                double sum_abs = 0.0;
                for (int k = 0; k < c.K; ++k) {
                    const size_t wi = c.transB
                            ? static_cast<size_t>(n) * c.K + k
                            : static_cast<size_t>(k) * c.N + n;
                    const double t
                            = static_cast<double>(nbt::load_f32(src, c.src_dt,
                                      static_cast<size_t>(m) * lda + k))
                            * nbt::load_f32(e.wei_bytes.data(), c.wei_dt, wi);
                    v += t;
                    sum_abs += std::fabs(t);
                }
                mag = 2e-5 * sum_abs;
            }
            if (c.bias_dt != data_type_t::none) {
                const double b = nbt::load_f32(e.bias.get(), c.bias_dt, n);
                v += b;
                mag += 1e-6 * std::fabs(b);
            }
            if (c.binary != post_op_type_t::none) {
                const double x = nbt::load_f32(e.binary.get(), data_type_t::f32,
                        static_cast<size_t>(c.binary_row ? 0 : m) * c.N + n);
                if (c.binary == post_op_type_t::binary_add) {
                    v += x;
                    mag += 1e-6 * std::fabs(x);
                } else {
                    v *= x;
                    mag = (mag + 1e-6 * std::fabs(v)) * std::fabs(x);
                }
            }
            const size_t i = static_cast<size_t>(m) * c.N + n;
            e.ref[i] = v;
            e.bound[i] = mag + 1e-6
                    + (c.dst_dt == data_type_t::bf16
                                    ? std::ldexp(std::fabs(v), -8)
                                    : 0.0);
        }
    }
}

::testing::AssertionResult matches(const expert_t &e) {
    size_t bad = 0;
    size_t first = 0;
    double first_got = 0.0;
    for (size_t i = 0; i < e.outputs(); ++i) {
        const double got = nbt::load_f32(e.dst.get(), e.c.dst_dt, i);
        if (!(std::fabs(got - e.ref[i]) <= e.bound[i]) && bad++ == 0) {
            first = i;
            first_got = got;
        }
    }
    if (bad == 0) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure()
            << bad << " of " << e.outputs()
            << " outputs off the reference; first at m=" << first / e.c.N
            << " n=" << first % e.c.N << ": " << first_got << " vs "
            << e.ref[first] << " +- " << e.bound[first];
}

::testing::AssertionResult same_bits(const expert_t &a, const expert_t &b) {
    const size_t es = nbt::dt_bytes(a.c.dst_dt);
    for (size_t i = 0; i < a.outputs(); ++i) {
        if (std::memcmp(a.dst.as<uint8_t>() + i * es,
                    b.dst.as<uint8_t>() + i * es, es)
                != 0)
            return ::testing::AssertionFailure()
                    << "first different output at m=" << i / a.c.N
                    << " n=" << i % a.c.N << ": "
                    << nbt::load_f32(a.dst.get(), a.c.dst_dt, i) << " vs "
                    << nbt::load_f32(b.dst.get(), b.c.dst_dt, i);
    }
    return ::testing::AssertionSuccess();
}

bool weight_kept(const expert_t &e) {
    return std::memcmp(e.wei.get(), e.wei_bytes.data(), e.wei_bytes.size())
            == 0;
}

/// The native BRGEMM keeps no layout of any expert's weight.
bool no_native_layout(const std::vector<expert_t> &experts) {
    for (const expert_t &e : experts)
        if (nbt::native_layouts(e.wei.get()) != 0) return false;
    return true;
}

case_t int8_case(int M, int N, int K, data_type_t src, grain_t scale,
        data_type_t scale_dt, grain_t zp, data_type_t zp_dt,
        data_type_t bias_dt, post_op_type_t binary, bool binary_row,
        data_type_t dst, uint32_t seed) {
    case_t c;
    c.M = M;
    c.N = N;
    c.K = K;
    c.src_dt = src;
    c.scale = scale;
    c.scale_dt = scale_dt;
    c.zp = zp;
    c.zp_dt = zp_dt;
    c.bias_dt = bias_dt;
    c.binary = binary;
    c.binary_row = binary_row;
    c.dst_dt = dst;
    c.seed = seed;
    return c;
}

case_t float_case(int M, int N, int K, data_type_t dt, data_type_t bias_dt,
        post_op_type_t binary, data_type_t dst, uint32_t seed) {
    case_t c;
    c.M = M;
    c.N = N;
    c.K = K;
    c.src_dt = dt;
    c.wei_dt = dt;
    c.scale = grain_t::none;
    c.bias_dt = bias_dt;
    c.binary = binary;
    c.dst_dt = dst;
    c.seed = seed;
    return c;
}

case_t dynamic_case(int M, int N, int K, data_type_t scale_dt,
        data_type_t bias_dt, data_type_t dst, uint32_t seed) {
    case_t c;
    c.M = M;
    c.N = N;
    c.K = K;
    c.src_dt = data_type_t::bf16;
    c.dynamic = true;
    c.scale = grain_t::token;
    c.scale_dt = scale_dt;
    c.bias_dt = bias_dt;
    c.dst_dt = dst;
    c.seed = seed;
    return c;
}

constexpr data_type_t u8 = data_type_t::u8;
constexpr data_type_t s8 = data_type_t::s8;
constexpr data_type_t s32 = data_type_t::s32;
constexpr data_type_t f32 = data_type_t::f32;
constexpr data_type_t bf16 = data_type_t::bf16;
constexpr data_type_t no_dt = data_type_t::none;
constexpr post_op_type_t add_po = post_op_type_t::binary_add;
constexpr post_op_type_t mul_po = post_op_type_t::binary_mul;
constexpr post_op_type_t no_po = post_op_type_t::none;

/// Experts the group calls run; M is set per expert.  The reference holds
/// for every one except a dynamically quantized source.
std::vector<case_t> group_cases() {
    std::vector<case_t> v;
    v.push_back(int8_case(1, 64, 96, u8, grain_t::token, f32, grain_t::tensor,
            s32, f32, add_po, false, f32, 61));
    v.push_back(int8_case(1, 96, 64, s8, grain_t::token, bf16, grain_t::none,
            s32, bf16, no_po, false, bf16, 62));
    v.push_back(int8_case(1, 64, 128, u8, grain_t::tensor, bf16,
            grain_t::tensor, u8, no_dt, mul_po, true, bf16, 63));
    v.push_back(int8_case(1, 64, 96, s8, grain_t::tensor, f32, grain_t::none,
            s32, f32, add_po, true, f32, 64));
    v.back().transB = true;
    v.push_back(dynamic_case(1, 64, 128, f32, bf16, bf16, 65));
    v.push_back(dynamic_case(1, 96, 64, bf16, no_dt, f32, 66));
    v.push_back(float_case(1, 64, 96, bf16, bf16, no_po, bf16, 67));
    v.push_back(float_case(1, 96, 64, bf16, f32, add_po, f32, 68));
    v.back().transB = true;
    v.push_back(float_case(1, 48, 64, bf16, no_dt, no_po, bf16, 69));
    return v;
}

/// The experts of base at rows, an independent copy for each seed offset.
std::vector<expert_t> make_experts(
        const case_t &base, const std::vector<int> &rows) {
    std::vector<expert_t> experts;
    for (size_t x = 0; x < rows.size(); ++x) {
        case_t c = base;
        c.M = rows[x];
        c.seed = base.seed + 31 * static_cast<uint32_t>(x);
        experts.push_back(make_expert(c));
        compute_reference(experts.back(), experts.back().src.get(), c.K);
    }
    return experts;
}

/// group_matmul_direct over the experts, the algorithm selected as mode
/// says.
status_t run_group(std::vector<expert_t> &experts, algo_mode_t mode,
        const mm::grp_matmul_gated_act_params *act = nullptr,
        const mm::grp_matmul_fused_moe_params *fused = nullptr) {
    const nbt::algo_pin_t pin(pinned_algo(mode));
    std::vector<char> layout;
    std::vector<bool> trans_a, trans_b, wconst;
    std::vector<int> M, N, K, lda, ldb, ldc;
    std::vector<float> alpha, beta;
    std::vector<const void *> src, wei, bias;
    std::vector<void *> dst;
    std::vector<mm::matmul_params> params;
    for (const expert_t &e : experts) {
        layout.push_back('r');
        trans_a.push_back(false);
        trans_b.push_back(e.c.transB);
        wconst.push_back(true);
        M.push_back(e.c.M);
        N.push_back(e.c.N);
        K.push_back(e.c.K);
        lda.push_back(e.lda());
        ldb.push_back(e.ldb());
        ldc.push_back(e.c.N);
        alpha.push_back(1.0f);
        beta.push_back(0.0f);
        src.push_back(e.src.get());
        wei.push_back(e.wei.get());
        bias.push_back(e.bias.get());
        dst.push_back(e.dst.get());
        params.push_back(e.params);
        if (mode == algo_mode_t::per_call)
            params.back().lowoha_algo = matmul_algo_t::native_brgemm;
    }
    return mm::group_matmul_direct(layout, trans_a, trans_b, M, N, K, alpha,
            src, lda, wei, ldb, bias, beta, dst, ldc, wconst, params, nullptr,
            act, fused);
}

// The inner kernel of the experts is the one an unset algorithm resolves to
// whenever the process starts with ZENDNNL_MATMUL_ALGO=10 or unset, and the
// per-expert slice on it writes the same bits under each mode.
TEST(NativeBrgemmGroupScope, ExpertSliceRunsTheUnsetKernel) {
    SKIP_WITHOUT_NATIVE_ISA();
    const char *env = std::getenv("ZENDNNL_MATMUL_ALGO");
    if (env != nullptr && env[0] != '\0' && std::strcmp(env, "10") != 0)
        GTEST_SKIP() << "the process starts with ZENDNNL_MATMUL_ALGO=" << env;
    EXPECT_EQ(static_cast<int>(mm::resolve_matmul_backend()),
            static_cast<int>(unset_kernel()));
    mm::clear_matmul_weight_caches();
    alive_t alive;
    for (const case_t &base : group_cases()) {
        case_t c = base;
        c.M = 7;
        SCOPED_TRACE(c.str());
        std::vector<expert_t> runs;
        for (algo_mode_t mode : kModes) {
            expert_t e = make_expert(c);
            compute_reference(e, e.src.get(), e.lda());
            const nbt::algo_pin_t pin(pinned_algo(mode));
            mm::matmul_params p = e.params;
            if (mode == algo_mode_t::per_call)
                p.lowoha_algo = matmul_algo_t::native_brgemm;
            for (auto &po : p.postop_)
                po.apply_alpha_default();
            ASSERT_EQ(mm::execute_expert_slice_checked('r', false, c.transB,
                              c.M, c.N, c.K, 1.0f, e.src.get(), e.lda(),
                              e.wei.get(), e.ldb(), e.bias.get(), 0.0f,
                              e.dst.get(), c.N, true, all_threads(), p,
                              mm::resolve_expert_backend(
                                      5, mm::resolve_matmul_backend(), p)),
                    status_t::success);
            EXPECT_EQ(nbt::native_layouts(e.wei.get()), 0u) << mode_name(mode);
            runs.push_back(std::move(e));
        }
        if (!c.dynamic) { EXPECT_TRUE(matches(runs[0])); }
        for (size_t r = 1; r < runs.size(); ++r)
            EXPECT_TRUE(same_bits(runs[0], runs[r])) << mode_name(kModes[r]);
        for (expert_t &e : runs)
            alive.push_back(std::move(e));
    }
    mm::clear_matmul_weight_caches();
}

struct exec_case_t {
    int algo;
    bool ck;
};

class NativeBrgemmGroupScopeExecutor
    : public ::testing::TestWithParam<exec_case_t> {};

// group_matmul_direct over four experts on one scheduling algorithm, with the
// custom kernel off or on: the same bits under each mode, and no native
// BRGEMM call.
TEST_P(NativeBrgemmGroupScopeExecutor, MatchesAlgoUnset) {
    SKIP_WITHOUT_NATIVE_ISA();
    const exec_case_t ec = GetParam();
    const mt::AlgoEnvGuard grp_algo(ec.algo);
    const mt::CustomKernelOverride ck(ec.ck);
    ::reset_grp_matmul_caches();
    const std::vector<int> rows = {9, 1, 20, 4};
    alive_t alive;
    for (const case_t &base : group_cases()) {
        SCOPED_TRACE(base.str() + " algo=" + std::to_string(ec.algo)
                + " ck=" + std::to_string(ec.ck ? 1 : 0));
        std::vector<std::vector<expert_t>> runs;
        for (algo_mode_t mode : kModes) {
            runs.push_back(make_experts(base, rows));
            ASSERT_EQ(run_group(runs.back(), mode), status_t::success)
                    << mode_name(mode);
            EXPECT_TRUE(no_native_layout(runs.back())) << mode_name(mode);
        }
        for (size_t x = 0; x < rows.size(); ++x) {
            if (!ec.ck && !base.dynamic) {
                EXPECT_TRUE(matches(runs[0][x])) << "expert " << x;
            }
            for (size_t r = 1; r < runs.size(); ++r)
                EXPECT_TRUE(same_bits(runs[0][x], runs[r][x]))
                        << mode_name(kModes[r]) << ", expert " << x;
        }
        for (auto &run : runs)
            for (expert_t &e : run)
                alive.push_back(std::move(e));
    }
    ::reset_grp_matmul_caches();
}

std::vector<exec_case_t> exec_cases() {
    std::vector<exec_case_t> v;
    for (bool ck : {false, true})
        for (int algo : {1, 2, 3, 5, 6})
            v.push_back({algo, ck});
    return v;
}

INSTANTIATE_TEST_SUITE_P(SchedulingAlgo, NativeBrgemmGroupScopeExecutor,
        ::testing::ValuesIn(exec_cases()),
        [](const ::testing::TestParamInfo<exec_case_t> &info) {
            return "algo" + std::to_string(info.param.algo) + "_ck"
                    + std::to_string(info.param.ck ? 1 : 0);
        });

// A sequential chain of group_matmul_direct (one source, two ops), INT8 (op
// 0 requantizes to u8 for op 1) and BF16: the same bits under each mode, and
// no native BRGEMM call.  The INT8 op 0 is checked within one quantization
// step, op 1 against the reference over the bytes op 0 wrote.
TEST(NativeBrgemmGroupScope, SequentialChainMatchesAlgoUnset) {
    SKIP_WITHOUT_NATIVE_ISA();
    mm::clear_matmul_weight_caches();
    const float dst_scale = 8.0f;
    const int32_t dst_zp = 128;
    const nbt::buffer_t ds = nbt::make_buffer(sizeof(float));
    const nbt::buffer_t dz = nbt::make_buffer(sizeof(int32_t));
    *ds.as<float>() = dst_scale;
    *dz.as<int32_t>() = dst_zp;
    alive_t alive;
    for (int variant = 0; variant < 5; ++variant) {
        const bool int8 = variant < 4;
        const grain_t grain
                = variant % 2 == 0 ? grain_t::token : grain_t::tensor;
        const case_t c0 = int8
                ? int8_case(12, 64, 96, u8, grain, f32, grain_t::tensor, s32,
                          variant < 2 ? f32 : no_dt,
                          variant == 2 ? add_po : no_po, false, u8,
                          41 + variant)
                : float_case(12, 64, 96, bf16, bf16, no_po, bf16, 46);
        const case_t c1 = int8
                ? int8_case(12, 32, 64, u8, grain_t::tensor, f32,
                          grain_t::tensor, s32, bf16,
                          variant == 3 ? mul_po : no_po, true,
                          variant % 2 == 0 ? f32 : bf16, 51 + variant)
                : float_case(12, 32, 64, bf16, f32, add_po, f32, 56);
        SCOPED_TRACE(c0.str() + " | " + c1.str());
        std::vector<expert_t> op0, op1;
        for (algo_mode_t mode : kModes) {
            expert_t e0 = make_expert(c0);
            expert_t e1 = make_expert(c1);
            if (int8) {
                set_quant(e0.params.quant_params.dst_scale, ds, f32, {1});
                set_quant(e0.params.quant_params.dst_zp, dz, s32, {1});
                *e1.src_scale.as<float>() = 1.0f / dst_scale;
                *e1.src_zp.as<int32_t>() = dst_zp;
            }
            compute_reference(e0, e0.src.get(), e0.lda());
            const nbt::algo_pin_t pin(pinned_algo(mode));
            std::vector<mm::matmul_params> params = {e0.params, e1.params};
            if (mode == algo_mode_t::per_call)
                for (auto &p : params)
                    p.lowoha_algo = matmul_algo_t::native_brgemm;
            ASSERT_EQ(mm::group_matmul_direct({'r', 'r'}, {false, false},
                              {false, false}, {c0.M, c1.M}, {c0.N, c1.N},
                              {c0.K, c1.K}, {1.0f, 1.0f}, {e0.src.get()},
                              {c0.K, c1.K}, {e0.wei.get(), e1.wei.get()},
                              {c0.N, c1.N}, {e0.bias.get(), e1.bias.get()},
                              {0.0f, 0.0f}, {e0.dst.get(), e1.dst.get()},
                              {c0.N, c1.N}, {true, true}, params),
                    status_t::success)
                    << mode_name(mode);
            EXPECT_EQ(nbt::native_layouts(e0.wei.get())
                            + nbt::native_layouts(e1.wei.get()),
                    0u)
                    << mode_name(mode);
            compute_reference(e1, e0.dst.get(), c0.N);
            op0.push_back(std::move(e0));
            op1.push_back(std::move(e1));
        }
        if (int8) {
            int off = 0;
            for (size_t i = 0; i < op0[0].outputs(); ++i) {
                const double q = std::min(255.0,
                        std::max(0.0,
                                std::nearbyint(op0[0].ref[i] * dst_scale)
                                        + dst_zp));
                if (std::fabs(op0[0].dst.as<uint8_t>()[i] - q) > 1.0) ++off;
            }
            EXPECT_EQ(off, 0) << "op 0 outputs off by more than one step";
        } else {
            EXPECT_TRUE(matches(op0[0])) << "op 0";
        }
        EXPECT_TRUE(matches(op1[0])) << "op 1";
        for (size_t r = 1; r < op0.size(); ++r) {
            EXPECT_TRUE(same_bits(op0[0], op0[r])) << mode_name(kModes[r]);
            EXPECT_TRUE(same_bits(op1[0], op1[r])) << mode_name(kModes[r]);
        }
        for (size_t r = 0; r < op0.size(); ++r) {
            alive.push_back(std::move(op0[r]));
            alive.push_back(std::move(op1[r]));
        }
    }
    mm::clear_matmul_weight_caches();
}

/// A BF16 fused MoE block over experts: the gate / up weight [H, N1], the
/// down weight [K2, H] and their bf16 biases, dst [M, H].
struct fused_t {
    std::vector<expert_t> up;
    std::vector<nbt::buffer_t> down, down_bias, out;
    int H = 0;
    int K2 = 0;
};

fused_t make_fused(
        const std::vector<int> &rows, int H, int I, bool gated, uint32_t seed) {
    fused_t f;
    f.H = H;
    f.K2 = I;
    const case_t up = float_case(
            1, gated ? 2 * I : I, H, bf16, bf16, no_po, bf16, seed);
    f.up = make_experts(up, rows);
    for (size_t x = 0; x < rows.size(); ++x) {
        const uint32_t s = seed + 1000 + 31 * static_cast<uint32_t>(x);
        f.down.push_back(nbt::kept_until_exit(nbt::random_buffer(
                bf16, static_cast<size_t>(I) * H, s, -0.3f, 0.3f)));
        f.down_bias.push_back(values(bf16, H, s + 1, -0.3f, 0.3f));
        f.out.push_back(nbt::make_buffer(
                static_cast<size_t>(rows[x]) * H * sizeof(uint16_t)));
    }
    return f;
}

status_t run_fused(fused_t &f, algo_mode_t mode, bool gated) {
    mm::grp_matmul_fused_moe_params fused;
    for (size_t x = 0; x < f.up.size(); ++x) {
        fused.down_weight.push_back(f.down[x].get());
        fused.N_down.push_back(f.H);
        fused.ldb_down.push_back(f.H);
        fused.bias_down.push_back(f.down_bias[x].get());
        fused.dst_down.push_back(f.out[x].get());
        fused.ldc_down.push_back(f.H);
    }
    fused.bias_dt_down = bf16;
    mm::grp_matmul_gated_act_params act;
    act.act = gated ? mm::grp_matmul_gated_act_t::silu_and_mul
                    : mm::grp_matmul_gated_act_t::none;
    return run_group(f.up, mode, gated ? &act : nullptr, &fused);
}

// Fused MoE (gate / up, activation, down, both biases) over three experts and
// over one: the same bits under each mode, and no native BRGEMM call.
TEST(NativeBrgemmGroupScope, FusedMoeMatchesAlgoUnset) {
    SKIP_WITHOUT_NATIVE_ISA();
    ::reset_grp_matmul_caches();
    std::vector<fused_t> alive;
    for (const std::vector<int> &rows :
            {std::vector<int> {5, 1, 9}, std::vector<int> {7}})
        for (bool gated : {true, false}) {
            SCOPED_TRACE(std::string(gated ? "silu_and_mul" : "none")
                    + " experts=" + std::to_string(rows.size()));
            std::vector<fused_t> runs;
            for (algo_mode_t mode : kModes) {
                runs.push_back(make_fused(rows, 64, 64, gated, 81));
                ASSERT_EQ(
                        run_fused(runs.back(), mode, gated), status_t::success)
                        << mode_name(mode);
                EXPECT_TRUE(no_native_layout(runs.back().up))
                        << mode_name(mode);
            }
            for (size_t r = 1; r < runs.size(); ++r)
                for (size_t x = 0; x < rows.size(); ++x)
                    EXPECT_EQ(std::memcmp(runs[0].out[x].get(),
                                      runs[r].out[x].get(),
                                      static_cast<size_t>(rows[x]) * 64
                                              * sizeof(uint16_t)),
                            0)
                            << mode_name(kModes[r]) << ", expert " << x;
            for (fused_t &f : runs)
                alive.push_back(std::move(f));
        }
    ::reset_grp_matmul_caches();
}

/// matmul_direct on e with the algorithm selected as mode says, the process
/// algorithm process unless that is none.
status_t run_direct(expert_t &e, algo_mode_t mode,
        matmul_algo_t process = matmul_algo_t::none) {
    const nbt::algo_pin_t pin(
            process != matmul_algo_t::none ? process : pinned_algo(mode));
    mm::matmul_params p = e.params;
    if (mode == algo_mode_t::per_call)
        p.lowoha_algo = matmul_algo_t::native_brgemm;
    mm::matmul_batch_params_t batch;
    return mm::matmul_direct('r', false, e.c.transB, e.c.M, e.c.N, e.c.K, 1.0f,
            e.src.get(), e.lda(), e.wei.get(), e.ldb(), e.bias.get(), 0.0f,
            e.dst.get(), e.c.N, true, batch, p);
}

/// Calls of weights whose N is not a multiple of the native layout's
/// blocks, which the native BRGEMM serves over a padded layout.  The widest
/// f32 call has a weight the AOCL route may reorder in place at weight cache
/// mode 2.
std::vector<case_t> padded_cases() {
    std::vector<case_t> v;
    v.push_back(float_case(8, 48, 96, bf16, bf16, no_po, bf16, 91));
    v.push_back(float_case(5, 40, 64, bf16, f32, add_po, f32, 92));
    v.back().transB = true;
    v.push_back(float_case(1, 48, 512, bf16, bf16, no_po, f32, 86));
    v.push_back(float_case(8, 48, 96, f32, f32, no_po, f32, 93));
    v.push_back(float_case(6, 40, 64, f32, no_dt, add_po, f32, 94));
    v.push_back(float_case(1, 80, 96, f32, f32, no_po, f32, 87));
    v.push_back(float_case(4, 48, 2048, f32, f32, no_po, f32, 88));
    v.back().transB = true;
    v.push_back(float_case(8, 112, 256, f32, no_dt, no_po, f32, 85));
    v.push_back(int8_case(8, 48, 96, u8, grain_t::tensor, f32, grain_t::tensor,
            s32, f32, add_po, false, f32, 95));
    v.push_back(int8_case(4, 40, 64, s8, grain_t::tensor, f32, grain_t::none,
            s32, bf16, mul_po, true, bf16, 96));
    return v;
}

/// A call of such a weight that the AOCL route rewrites in place at weight
/// cache mode 2 when the scalar custom-kernel route is off: an s8 source
/// with per-token scales, which the kernel selection itself turns to
/// AOCL-DLP blocked, with room for the blocked image.
case_t in_place_padded_case() {
    case_t c = int8_case(4, 48, 128, s8, grain_t::token, f32, grain_t::none,
            s32, no_dt, no_po, false, f32, 98);
    c.capacity = 4 * 48 * 128 + 4096;
    return c;
}

/// Scoped ZENDNNL_MATMUL_CUSTOM_KERNEL route.
class ck_route_t {
public:
    explicit ck_route_t(bool on)
        : prev_(matmul_config_t::instance().get_custom_kernel_route()) {
        matmul_config_t::instance().set_custom_kernel_route(on);
    }
    ~ck_route_t() {
        matmul_config_t::instance().set_custom_kernel_route(prev_);
    }
    ck_route_t(const ck_route_t &) = delete;
    ck_route_t &operator=(const ck_route_t &) = delete;

private:
    bool prev_;
};

// A matmul_direct call that asks for the native BRGEMM with a weight whose
// N is not a multiple of 32 runs on the native BRGEMM over a padded layout
// packed out of place: within the reference band, with a native layout and
// the weight not rewritten in place, also when the kernel selection would
// turn the call to AOCL-DLP blocked by itself and under dynamic dispatch (a
// lowoha_algo = native_brgemm); with the algorithm unset the call runs on
// AOCL DLP, whose run of the in-place case rewrites its weight.  The calls
// on the layout's blocks (BF16 and FP32) run on the native BRGEMM too.
TEST(NativeBrgemmPaddedRoute, CallsRunOnTheNativeBrgemm) {
    SKIP_WITHOUT_NATIVE_ISA();
    mm::clear_matmul_weight_caches();
    WeightCacheGuard cache(2);
    alive_t alive;
    std::vector<case_t> cases = padded_cases();
    cases.push_back(in_place_padded_case());
    const ck_route_t route(false);
    for (const case_t &c : cases) {
        SCOPED_TRACE(c.str());
        {
            expert_t e = make_expert(c);
            compute_reference(e, e.src.get(), e.lda());
            ASSERT_EQ(run_direct(e, algo_mode_t::unset), status_t::success);
            EXPECT_TRUE(matches(e));
            alive.push_back(std::move(e));
        }
        for (algo_mode_t mode : {algo_mode_t::algo10, algo_mode_t::per_call}) {
            expert_t e = make_expert(c);
            compute_reference(e, e.src.get(), e.lda());
            EXPECT_EQ(run_direct(e, mode), status_t::success)
                    << mode_name(mode);
            EXPECT_TRUE(matches(e)) << mode_name(mode);
            EXPECT_EQ(nbt::native_layouts(e.wei.get()), 1u) << mode_name(mode);
            EXPECT_TRUE(weight_kept(e)) << mode_name(mode);
            alive.push_back(std::move(e));
        }
        {
            expert_t e = make_expert(c);
            compute_reference(e, e.src.get(), e.lda());
            EXPECT_EQ(run_direct(e, algo_mode_t::per_call,
                              matmul_algo_t::dynamic_dispatch),
                    status_t::success);
            EXPECT_TRUE(matches(e)) << "under dynamic dispatch";
            EXPECT_TRUE(weight_kept(e)) << "under dynamic dispatch";
            alive.push_back(std::move(e));
        }
    }
    for (const case_t &c : {float_case(8, 64, 96, bf16, bf16, no_po, bf16, 97),
                 float_case(6, 64, 64, f32, no_dt, add_po, f32, 99),
                 float_case(8, 128, 256, f32, no_dt, no_po, f32, 100)}) {
        SCOPED_TRACE("served " + c.str());
        expert_t e = make_expert(c);
        compute_reference(e, e.src.get(), e.lda());
        ASSERT_EQ(run_direct(e, algo_mode_t::algo10), status_t::success);
        EXPECT_GE(nbt::native_layouts(e.wei.get()), 1u);
        EXPECT_TRUE(matches(e));
        alive.push_back(std::move(e));
    }
    mm::clear_matmul_weight_caches();
}

// A batched call: the native BRGEMM serves only single calls.  A batched
// call that names it (lowoha_algo native_brgemm) returns unimplemented,
// with dst untouched; one with no lowoha_algo takes its kernel from
// ZENDNNL_BMM_ALGO, so under ZENDNNL_MATMUL_ALGO=10, which selects the
// kernel of single calls, it runs on its batched route with the bits it has
// with the algorithm unset, and the weight stays raw.
TEST(NativeBrgemmDeclinedRoute, BatchedCallsAskForTheNativeBrgemmOnlyByName) {
    SKIP_WITHOUT_NATIVE_ISA();
    mm::clear_matmul_weight_caches();
    const int batch = 3, M = 6, N = 64, K = 96;
    for (data_type_t dt : {bf16, f32}) {
        SCOPED_TRACE(dtype_info(dt));
        const size_t es = nbt::dt_bytes(dt);
        const size_t dst_bytes = static_cast<size_t>(batch) * M * N * es;
        const nbt::buffer_t src = nbt::random_buffer(
                dt, static_cast<size_t>(batch) * M * K, 101, -1.0f, 1.0f);
        const nbt::buffer_t raw = nbt::random_buffer(
                dt, static_cast<size_t>(batch) * K * N, 102, -0.5f, 0.5f);
        std::vector<nbt::buffer_t> wei, dst;
        for (algo_mode_t mode : kModes) {
            SCOPED_TRACE(mode_name(mode));
            wei.push_back(nbt::kept_until_exit(nbt::make_buffer(raw.bytes)));
            std::memcpy(wei.back().get(), raw.get(), raw.bytes);
            dst.push_back(nbt::make_buffer(dst_bytes));
            const nbt::algo_pin_t pin(pinned_algo(mode));
            mm::matmul_params p;
            p.dtypes.src = p.dtypes.wei = p.dtypes.dst = dt;
            if (mode == algo_mode_t::per_call)
                p.lowoha_algo = matmul_algo_t::native_brgemm;
            mm::matmul_batch_params_t bp;
            bp.Batch_A = batch;
            bp.Batch_B = batch;
            const bool named = mode == algo_mode_t::per_call;
            EXPECT_EQ(mm::native_brgemm_refusal('r', false, false, M, N, K,
                              1.0f, src.get(), K, wei.back().get(), N, nullptr,
                              0.0f, dst.back().get(), N, true, bp, p),
                    named ? status_t::unimplemented : status_t::success);
            EXPECT_EQ(run_matmul_direct('r', false, false, M, N, K, 1.0f,
                              src.get(), K, wei.back().get(), N, nullptr, 0.0f,
                              dst.back().get(), N, true, bp, p),
                    named ? status_t::unimplemented : status_t::success);
            EXPECT_EQ(native_refusal_checked(), named);
            EXPECT_EQ(nbt::native_layouts(wei.back().get()), 0u);
            EXPECT_EQ(std::memcmp(wei.back().get(), raw.get(), raw.bytes), 0)
                    << "the weight was rewritten";
        }
        EXPECT_EQ(std::memcmp(dst[0].get(), dst[1].get(), dst_bytes), 0)
                << "ZENDNNL_MATMUL_ALGO=10 changed the batched call's bits";
    }
    mm::clear_matmul_weight_caches();
}

} // namespace
