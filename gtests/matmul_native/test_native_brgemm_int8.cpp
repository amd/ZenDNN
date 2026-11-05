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

/// INT8 operands and tiles of the native BRGEMM over native-layout weights
/// (decide_int8_call, prepare_int8_call, compute_int8_tile), called
/// directly on weights packed by custom_kernel::prepack_weight_into_int8:
///   * against an integer reference (exact int32, then the custom kernel's
///     f32 dequantization), bit for bit;
///   * against the custom-kernel INT8 microkernel on the same operands;
///   * against the custom-kernel route (one-op group_matmul_direct, ALGO 3)
///     on a bf16 source quantized per token: same output and same scales /
///     zero points written back.

#include <gtest/gtest.h>

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "common/zendnnl_compat.hpp"
#include "group_matmul/moe_test_utils.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/pack.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/ukernel/int8_microkernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/int8/native_int8_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_brgemm_looper.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_int8_tile.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_epilogue.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_hw.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_pack.hpp"
#include "matmul_native/native_brgemm_test_utils.hpp"

namespace {

namespace ck = zendnnl::lowoha::matmul::custom_kernel;
namespace mm = zendnnl::lowoha::matmul;
namespace nat = zendnnl::lowoha::matmul::native;
namespace mt = moe_test_utils;
using mm::matmul_params;
using zendnnl::common::bfloat16_t;
using zendnnl::common::data_type_t;
using zendnnl::common::float16_t;
using zendnnl::error_handling::status_t;

#define NATIVE_BRGEMM_INT8_SKIP_IF_NO_VNNI() \
    do { \
        if (!ck::avx512vnni_available()) \
            GTEST_SKIP() << "AVX-512 VNNI not available on this host"; \
    } while (0)

int all_threads() {
    return std::max(1, omp_get_max_threads());
}

/// 64-byte aligned bytes: the custom-kernel microkernel reads its pack with
/// aligned loads.
struct AlignedBytes {
    struct free_t {
        void operator()(uint8_t *p) const { zendnnl_aligned_free(p); }
    };
    std::unique_ptr<uint8_t, free_t> ptr;
    void resize(size_t n) {
        const size_t bytes = (n + 63) / 64 * 64;
        ptr.reset(static_cast<uint8_t *>(zendnnl_aligned_alloc(64, bytes)));
        std::memset(ptr.get(), 0, bytes);
    }
    uint8_t *data() const { return ptr.get(); }
};

/// A raw s8 weight, its custom-kernel pack and the view over the pack.
struct CkWeight {
    int K = 0;
    int N = 0;
    int ldb = 0;
    bool transB = false;
    std::vector<int8_t> raw;
    AlignedBytes packed;
    nat::native_weight_view_t view;
    /// Logical weight [N][K].
    std::vector<int8_t> nk;
};

CkWeight make_weight(int K, int N, int pack_nr, bool transB, bool interleave,
        std::mt19937 &rng, int ld_pad = 0) {
    CkWeight w;
    w.K = K;
    w.N = N;
    w.transB = transB;
    w.ldb = (transB ? K : N) + ld_pad;
    const int rows = transB ? N : K;
    w.raw.resize(static_cast<size_t>(rows) * w.ldb);
    std::uniform_int_distribution<int> dist(-128, 127);
    for (auto &v : w.raw)
        v = static_cast<int8_t>(dist(rng));
    w.nk.resize(static_cast<size_t>(N) * K);
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            w.nk[static_cast<size_t>(n) * K + k] = transB
                    ? w.raw[static_cast<size_t>(n) * w.ldb + k]
                    : w.raw[static_cast<size_t>(k) * w.ldb + n];
    const size_t bytes = ck::packed_weight_size_int8(K, N, pack_nr);
    EXPECT_GT(bytes, 0u);
    w.packed.resize(bytes);
    EXPECT_EQ(ck::prepack_weight_into_int8(w.raw.data(), K, N, w.ldb, pack_nr,
                      transB, interleave, w.packed.data()),
            status_t::success);
    auto &v = w.view;
    v.base = w.packed.data();
    v.wei_dt = data_type_t::s8;
    v.K = K;
    v.N = N;
    v.pack_nr = pack_nr;
    v.k_groups = (K + 3) / 4;
    v.block_bytes = static_cast<size_t>(v.k_groups) * pack_nr * 4
            + static_cast<size_t>(pack_nr) * sizeof(int32_t);
    v.interleaved = interleave;
    EXPECT_LE(v.block_bytes * (N / pack_nr), bytes);
    return w;
}

/// f32 values (logical order) the reference reads; the stored copies keep
/// the dtype the call passes.
struct QuantValues {
    std::vector<float> f32;
    std::vector<bfloat16_t> bf16;
    std::vector<float16_t> f16;
    std::vector<float> rounded;

    const void *store(const std::vector<float> &values, data_type_t dt) {
        rounded = values;
        if (dt == data_type_t::bf16) {
            bf16.assign(values.begin(), values.end());
            for (size_t i = 0; i < values.size(); ++i)
                rounded[i] = static_cast<float>(bf16[i]);
            return bf16.data();
        }
        if (dt == data_type_t::f16) {
            f16.assign(values.begin(), values.end());
            for (size_t i = 0; i < values.size(); ++i)
                rounded[i] = static_cast<float>(f16[i]);
            return f16.data();
        }
        f32 = values;
        return f32.data();
    }
};

struct ZpValues {
    std::vector<int32_t> s32;
    std::vector<int8_t> s8;
    std::vector<uint8_t> u8;
    std::vector<int32_t> logical;

    const void *store(const std::vector<int32_t> &values, data_type_t dt) {
        logical = values;
        if (dt == data_type_t::s8) {
            s8.assign(values.begin(), values.end());
            return s8.data();
        }
        if (dt == data_type_t::u8) {
            u8.assign(values.begin(), values.end());
            return u8.data();
        }
        s32 = values;
        return s32.data();
    }
};

/// Dequantization inputs of the reference, logical order; stride 0 is per
/// tensor.
struct Dequant {
    const int32_t *zp = nullptr;
    int zp_stride = 0;
    const float *src_scale = nullptr;
    int src_scale_stride = 0;
    const float *wei_scale = nullptr;
    int wei_scale_stride = 0;
    const float *bias = nullptr;
};

/// The custom kernel's result for every (row, pack column): exact int32
/// accumulation and compensation (wrapping like the kernel), conversion,
/// then the scale product and one FMA with the bias.  a is [M][K] logical
/// source values.
std::vector<float> reference(const std::vector<int> &a, int M, int K,
        const CkWeight &w, bool src_signed, const Dequant &d) {
    const int N = w.N;
    std::vector<int32_t> col_sum(N, 0);
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            col_sum[n] += w.nk[static_cast<size_t>(n) * K + k];
    const int shift = src_signed ? 128 : 0;
    std::vector<float> out(static_cast<size_t>(M) * N);
#pragma omp parallel for schedule(dynamic)
    for (int m = 0; m < M; ++m) {
        const int *row = a.data() + static_cast<size_t>(m) * K;
        for (int p = 0; p < N; ++p) {
            const int col = w.view.logical_col(p);
            const int8_t *wc = w.nk.data() + static_cast<size_t>(col) * K;
            int64_t acc = 0;
            for (int k = 0; k < K; ++k)
                acc += static_cast<int64_t>(row[k] + shift) * wc[k];
            const uint32_t zp
                    = static_cast<uint32_t>(d.zp ? d.zp[m * d.zp_stride] : 0)
                    + shift;
            const int32_t v = static_cast<int32_t>(static_cast<uint32_t>(acc)
                    - zp * static_cast<uint32_t>(col_sum[col]));
            const float f = static_cast<float>(v);
            const float s = d.src_scale[m * d.src_scale_stride]
                    * d.wei_scale[col * d.wei_scale_stride];
            out[static_cast<size_t>(m) * N + p]
                    = d.bias ? std::fma(f, s, d.bias[col]) : f * s;
        }
    }
    return out;
}

/// The built-in plan of a one-row INT8 call: its call widths, rows per call
/// and copy sizes hold for any shape, and its K passes are the L2 rule's.
const nat::native_plan_t &int8_plan() {
    static const nat::native_plan_t plan = [] {
        nat::native_problem_t p;
        p.M = 1;
        p.N = nat::kNativeLayoutPackNr;
        p.K = 4;
        p.pack_nr = nat::kNativeLayoutPackNr;
        p.src_dt = data_type_t::u8;
        p.wei_dt = data_type_t::s8;
        p.dst_dt = data_type_t::f32;
        return nat::heuristic_plan(
                p, nat::native_plan_uarch(), nat::heuristic_params_t {});
    }();
    return plan;
}

/// Runs compute_int8_tile over all of M x N in tiles of mr x nr the way a
/// looper does, output in pack column order.
std::vector<float> run_tiles(const nat::native_int8_call_t &call,
        const nat::native_weight_view_t &view, int M, int mr, int nr,
        int threads) {
    const int N = view.N;
    std::vector<float> out(static_cast<size_t>(M) * N, -1.0f);
    const int m_tiles = (M + mr - 1) / mr;
    const int n_tiles = N / nr;
#pragma omp parallel num_threads(threads)
    {
        std::vector<float> scratch(static_cast<size_t>(mr) * nr);
#pragma omp for collapse(2) schedule(dynamic)
        for (int i = 0; i < m_tiles; ++i) {
            for (int j = 0; j < n_tiles; ++j) {
                const int m0 = i * mr;
                const int rows = std::min(mr, M - m0);
                const int n0 = j * nr;
                nat::compute_int8_tile(int8_plan(), call, view, m0, rows, n0,
                        nr, scratch.data(), nr);
                for (int r = 0; r < rows; ++r)
                    std::memcpy(
                            out.data() + static_cast<size_t>(m0 + r) * N + n0,
                            scratch.data() + static_cast<size_t>(r) * nr,
                            sizeof(float) * nr);
            }
        }
    }
    return out;
}

/// Logical source values of a prepared call (A without its padding).
std::vector<int> call_values(
        const nat::native_int8_call_t &call, int M, int K) {
    std::vector<int> a(static_cast<size_t>(M) * K);
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k) {
            const uint8_t b = call.A[static_cast<size_t>(m) * call.lda + k];
            a[static_cast<size_t>(m) * K + k] = call.src_signed
                    ? static_cast<int>(static_cast<int8_t>(b))
                    : static_cast<int>(b);
        }
    return a;
}

::testing::AssertionResult same_bits(
        const std::vector<float> &got, const std::vector<float> &want, int N) {
    if (got.size() != want.size())
        return ::testing::AssertionFailure()
                << "size " << got.size() << " != " << want.size();
    size_t bad = 0, first = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::memcmp(&got[i], &want[i], sizeof(float)) != 0) {
            if (bad++ == 0) first = i;
        }
    }
    if (bad == 0) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure()
            << bad << " of " << got.size() << " values differ; first at ("
            << first / N << ", " << first % N << "): got " << got[first]
            << ", want " << want[first];
}

nat::GemmDescriptor make_desc(int M, int N, int K, int lda, bool transA,
        data_type_t src_dt, data_type_t dst_dt, int threads) {
    nat::GemmDescriptor desc;
    desc.M = M;
    desc.N = N;
    desc.K = K;
    desc.lda = lda;
    desc.ldb = K;
    desc.ldc = N;
    desc.transA = transA;
    desc.transB = true;
    desc.alpha = 1.0f;
    desc.beta = 0.0f;
    desc.src_dt = src_dt;
    desc.wei_dt = data_type_t::s8;
    desc.dst_dt = dst_dt;
    desc.num_threads = threads;
    return desc;
}

/// check_int8_call, then prepare_int8_call with the copy sizes of
/// int8_plan(), on decide_int8_call's decision, as the looper does.
status_t prepare(const nat::GemmDescriptor &desc, const matmul_params &params,
        const void *src, nat::native_int8_call_t &call) {
    const nat::native_int8_decision_t d
            = nat::decide_int8_call(desc, params, nullptr);
    const status_t st = nat::check_int8_call(desc, params, d);
    if (st != status_t::success) return st;
    return nat::prepare_int8_call(desc, params, d, src, int8_plan().copy_block,
            int8_plan().copy_par_min, call);
}

/// Gives call the bias (of type bias_dt) the way native_brgemm_execute does
/// under the built-in plan: when decide_int8_call moves it into the
/// dequantization.  Returns whether it did.
bool take_bias(const nat::GemmDescriptor &desc, matmul_params params,
        const void *bias, data_type_t bias_dt, nat::native_int8_call_t &call) {
    params.dtypes.bias = bias_dt;
    call.bias = nullptr;
    if (!nat::decide_int8_call(desc, params, bias).bias_in_dequant)
        return false;
    call.bias = bias;
    call.bias_dt = bias_dt;
    return true;
}

std::vector<float> random_floats(
        size_t n, float lo, float hi, std::mt19937 &rng) {
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n);
    for (auto &x : v)
        x = dist(rng);
    return v;
}

// ── Static u8 / s8 sources against the integer reference ──────────────────

enum class Gran { none, tensor, row };

struct StaticCase {
    int M, N, K, pack_nr;
    bool transB, interleave, src_signed, transA;
    int lda_pad;
    Gran scale;
    data_type_t scale_dt;
    Gran zp;
    data_type_t zp_dt;
    Gran wei;
    data_type_t wei_dt;
    data_type_t bias_dt;
    int mr, nr, threads; ///< threads 0: all
};

std::string static_name(const ::testing::TestParamInfo<StaticCase> &info) {
    const auto &c = info.param;
    return "c" + std::to_string(info.index) + "_M" + std::to_string(c.M) + "_N"
            + std::to_string(c.N) + "_K" + std::to_string(c.K) + "_nr"
            + std::to_string(c.pack_nr) + (c.src_signed ? "_s8" : "_u8")
            + (c.transA ? "_tA" : "") + (c.interleave ? "_il" : "");
}

// The base case is what an asymmetric quantization in the call gives the
// tile: a u8 source with per-row scales and zero points.  The other sweeps
// keep to the source operands calls give: a u8 source with a per-tensor
// scale and zero point, an s8 source with no zero point and a per-tensor or
// per-row scale, or with a per-tensor scale and zero point.
std::vector<StaticCase> static_cases() {
    const auto f32 = data_type_t::f32, bf16 = data_type_t::bf16,
               f16 = data_type_t::f16, s32 = data_type_t::s32,
               s8 = data_type_t::s8, u8 = data_type_t::u8,
               none = data_type_t::none;
    const StaticCase base {16, 64, 64, 32, true, false, false, false, 0,
            Gran::row, f32, Gran::row, s32, Gran::row, f32, none, 12, 32, 4};
    std::vector<StaticCase> v;
    v.push_back(base);
    for (int M : {1, 2, 4, 7, 8, 13, 32, 63, 64, 65, 128, 512}) {
        auto c = base;
        c.M = M;
        v.push_back(c);
    }
    for (int K : {1, 3, 4, 31, 63, 100, 1000, 4096}) {
        auto c = base;
        c.K = K;
        v.push_back(c);
    }
    for (int N : {32, 96, 256}) {
        auto c = base;
        c.N = N;
        v.push_back(c);
    }
    {
        auto c = base;
        c.M = 1500;
        c.K = 100;
        c.N = 64;
        c.mr = 64;
        c.threads = 0;
        v.push_back(c);
    }
    for (int pack_nr : {32, 64}) {
        for (bool il : {false, true}) {
            for (bool tb : {false, true}) {
                for (int nr : {16, 32, 64}) {
                    if (nr > pack_nr) continue;
                    auto c = base;
                    c.N = 128;
                    c.pack_nr = pack_nr;
                    c.interleave = il;
                    c.transB = tb;
                    c.nr = nr;
                    v.push_back(c);
                }
            }
        }
    }
    for (bool sgn : {false, true}) {
        for (Gran zp : {Gran::none, Gran::tensor}) {
            for (data_type_t zdt : {s32, s8, u8}) {
                if (zp == Gran::none && zdt != s32) continue;
                auto c = base;
                c.src_signed = sgn;
                c.zp = zp;
                c.zp_dt = zdt;
                // Per-row scales only for an s8 source without a zero point.
                if (!sgn || zp != Gran::none) c.scale = Gran::tensor;
                v.push_back(c);
            }
        }
    }
    for (Gran s : {Gran::none, Gran::tensor, Gran::row}) {
        for (data_type_t sdt : {f32, bf16, f16}) {
            if (s == Gran::none && sdt != f32) continue;
            // The source scales of an s8 source without a zero point.
            auto c = base;
            c.src_signed = true;
            c.zp = Gran::none;
            c.scale = s;
            c.scale_dt = sdt;
            v.push_back(c);
            // The weight scales; per-row source scales take them per column.
            c = base;
            if (s != Gran::row) {
                c.scale = Gran::tensor;
                c.zp = Gran::tensor;
            }
            c.wei = s;
            c.wei_dt = sdt;
            v.push_back(c);
        }
    }
    for (data_type_t b : {f32, bf16, f16}) {
        for (bool il : {false, true}) {
            auto c = base;
            c.bias_dt = b;
            c.interleave = il;
            v.push_back(c);
        }
    }
    for (bool sgn : {false, true}) {
        for (int K : {64, 3, 31, 100}) {
            for (int pad : {0, 5}) {
                // A transposed u8 or s8 source as a call gives it, unquantized.
                auto c = base;
                c.src_signed = sgn;
                c.scale = sgn ? Gran::row : Gran::tensor;
                c.zp = sgn ? Gran::none : Gran::tensor;
                c.transA = true;
                c.K = K;
                c.lda_pad = pad;
                v.push_back(c);
                c.transA = false;
                v.push_back(c);
            }
        }
    }
    for (int mr : {1, 5, 12, 13, 64}) {
        for (int threads : {1, 4, 0}) {
            auto c = base;
            c.M = 70;
            c.mr = mr;
            c.threads = threads;
            v.push_back(c);
        }
    }
    {
        auto c = base;
        c.M = 64;
        c.N = 4096;
        c.K = 4096;
        c.threads = 0;
        c.bias_dt = bf16;
        c.wei_dt = bf16;
        v.push_back(c);
    }
    // K in passes: tiles of three row calls over a slab above half of an L2
    // of up to 2 MiB.
    for (const bool sgn : {false, true}) {
        auto c = base;
        c.M = 30;
        c.K = 40000;
        c.mr = 30;
        c.src_signed = sgn;
        c.bias_dt = bf16;
        c.wei_dt = bf16;
        v.push_back(c);
    }
    return v;
}

class NativeBrgemmInt8Static : public ::testing::TestWithParam<StaticCase> {};

TEST_P(NativeBrgemmInt8Static, MatchesIntegerReference) {
    NATIVE_BRGEMM_INT8_SKIP_IF_NO_VNNI();
    const StaticCase c = GetParam();
    std::mt19937 rng(static_cast<uint32_t>(
            c.M * 131 + c.N * 7 + c.K * 3 + c.pack_nr + c.lda_pad));
    CkWeight w = make_weight(c.K, c.N, c.pack_nr, c.transB, c.interleave, rng);

    const int lda = (c.transA ? c.M : c.K) + c.lda_pad;
    std::vector<uint8_t> src(
            static_cast<size_t>(c.transA ? c.K : c.M) * lda, 0xA5);
    std::vector<int> a(static_cast<size_t>(c.M) * c.K);
    std::uniform_int_distribution<int> byte(0, 255);
    for (int m = 0; m < c.M; ++m) {
        for (int k = 0; k < c.K; ++k) {
            const uint8_t b = static_cast<uint8_t>(byte(rng));
            src[c.transA ? static_cast<size_t>(k) * lda + m
                         : static_cast<size_t>(m) * lda + k]
                    = b;
            a[static_cast<size_t>(m) * c.K + k] = c.src_signed
                    ? static_cast<int>(static_cast<int8_t>(b))
                    : static_cast<int>(b);
        }
    }

    matmul_params params;
    params.dtypes.src = c.src_signed ? data_type_t::s8 : data_type_t::u8;
    params.dtypes.wei = data_type_t::s8;
    params.dtypes.dst = data_type_t::f32;
    params.dtypes.bias = c.bias_dt;
    auto &qp = params.quant_params;
    Dequant d;
    const std::vector<float> ones(1, 1.0f);

    QuantValues src_scale;
    if (c.scale != Gran::none) {
        const int n = c.scale == Gran::row ? c.M : 1;
        qp.src_scale.buff = src_scale.store(
                random_floats(n, 0.001f, 0.05f, rng), c.scale_dt);
        qp.src_scale.dt = c.scale_dt;
        qp.src_scale.dims = {n, 1};
        d.src_scale = src_scale.rounded.data();
        d.src_scale_stride = c.scale == Gran::row ? 1 : 0;
    } else {
        d.src_scale = ones.data();
    }
    ZpValues zp;
    if (c.zp != Gran::none) {
        const int n = c.zp == Gran::row ? c.M : 1;
        const int lo = c.zp_dt == data_type_t::u8 ? 0
                : c.zp_dt == data_type_t::s8      ? -128
                                                  : -300;
        const int hi = c.zp_dt == data_type_t::u8 ? 255
                : c.zp_dt == data_type_t::s8      ? 127
                                                  : 300;
        std::uniform_int_distribution<int> zd(lo, hi);
        std::vector<int32_t> values(n);
        for (auto &z : values)
            z = zd(rng);
        qp.src_zp.buff = zp.store(values, c.zp_dt);
        qp.src_zp.dt = c.zp_dt;
        qp.src_zp.dims = {n, 1};
        d.zp = zp.logical.data();
        d.zp_stride = c.zp == Gran::row ? 1 : 0;
    }
    QuantValues wei_scale;
    if (c.wei != Gran::none) {
        const int n = c.wei == Gran::row ? c.N : 1;
        qp.wei_scale.buff = wei_scale.store(
                random_floats(n, 0.001f, 0.05f, rng), c.wei_dt);
        qp.wei_scale.dt = c.wei_dt;
        qp.wei_scale.dims = {1, n};
        d.wei_scale = wei_scale.rounded.data();
        d.wei_scale_stride = c.wei == Gran::row ? 1 : 0;
    } else {
        d.wei_scale = ones.data();
    }
    QuantValues bias;
    const void *bias_ptr = nullptr;
    if (c.bias_dt != data_type_t::none) {
        bias_ptr = bias.store(random_floats(c.N, -2.0f, 2.0f, rng), c.bias_dt);
        d.bias = bias.rounded.data();
    }

    const int threads = c.threads == 0 ? all_threads() : c.threads;
    const auto desc = make_desc(c.M, c.N, c.K, lda, c.transA, params.dtypes.src,
            data_type_t::f32, threads);
    nat::native_int8_call_t call;
    ASSERT_EQ(prepare(desc, params, src.data(), call), status_t::success);
    EXPECT_EQ(take_bias(desc, params, bias_ptr, c.bias_dt, call),
            bias_ptr != nullptr);
    EXPECT_EQ(call.src_signed, c.src_signed);
    const bool in_place = !c.transA && c.K % 4 == 0;
    EXPECT_EQ(call.A == src.data(), in_place);
    EXPECT_EQ(call.lda, in_place ? lda : (c.K + 3) / 4 * 4);

    const auto got = run_tiles(call, w.view, c.M, c.mr, c.nr, threads);
    const auto want = reference(a, c.M, c.K, w, c.src_signed, d);
    EXPECT_TRUE(same_bits(got, want, c.N));
}

INSTANTIATE_TEST_SUITE_P(NativeBrgemmInt8, NativeBrgemmInt8Static,
        ::testing::ValuesIn(static_cases()), static_name);

// The int32 accumulation itself: unit scales and no zero point, values small
// enough for every sum to convert to f32 exactly.
TEST(NativeBrgemmInt8Tile, RawAccumulatorsAreExact) {
    NATIVE_BRGEMM_INT8_SKIP_IF_NO_VNNI();
    for (const bool sgn : {false, true}) {
        for (const int pack_nr : {32, 64}) {
            const int M = 29, N = 128, K = 1000;
            std::mt19937 rng(pack_nr + (sgn ? 1 : 0));
            CkWeight w = make_weight(K, N, pack_nr, true, false, rng);
            std::vector<uint8_t> src(static_cast<size_t>(M) * K);
            std::uniform_int_distribution<int> byte(0, 255);
            for (auto &b : src)
                b = static_cast<uint8_t>(byte(rng));
            matmul_params params;
            params.dtypes.src = sgn ? data_type_t::s8 : data_type_t::u8;
            params.dtypes.wei = data_type_t::s8;
            params.dtypes.dst = data_type_t::f32;
            nat::native_int8_call_t call;
            const auto desc = make_desc(
                    M, N, K, K, false, params.dtypes.src, data_type_t::f32, 4);
            ASSERT_EQ(
                    prepare(desc, params, src.data(), call), status_t::success);
            const auto got = run_tiles(call, w.view, M, 12, pack_nr, 4);
            for (int m = 0; m < M; ++m) {
                for (int n = 0; n < N; ++n) {
                    int64_t acc = 0;
                    for (int k = 0; k < K; ++k) {
                        const uint8_t b = src[static_cast<size_t>(m) * K + k];
                        const int av = sgn ? static_cast<int8_t>(b) : b;
                        acc += static_cast<int64_t>(av)
                                * w.nk[static_cast<size_t>(n) * K + k];
                    }
                    ASSERT_LT(std::llabs(acc), int64_t {1} << 24);
                    ASSERT_EQ(got[static_cast<size_t>(m) * N + n],
                            static_cast<float>(acc))
                            << "m=" << m << " n=" << n << " signed=" << sgn;
                }
            }
        }
    }
}

// A destination given to the tile: the bytes of the f32 tile stored by the
// epilogue, for a bf16 and an f32 destination, over all of K and in passes.
TEST(NativeBrgemmInt8Tile, DirectStoreMatchesEpilogueStore) {
    NATIVE_BRGEMM_INT8_SKIP_IF_NO_VNNI();
    for (const int K : {1000, 40000}) {
        for (const data_type_t dst_dt : {data_type_t::bf16, data_type_t::f32}) {
            const int M = 30, N = 64, ldc = N + 16;
            const bool f32 = dst_dt == data_type_t::f32;
            std::mt19937 rng(static_cast<uint32_t>(K + (f32 ? 1 : 0)));
            CkWeight w = make_weight(K, N, 32, true, false, rng);
            std::vector<uint8_t> src(static_cast<size_t>(M) * K);
            std::uniform_int_distribution<int> byte(0, 255);
            for (auto &b : src)
                b = static_cast<uint8_t>(byte(rng));
            std::vector<float> ss = random_floats(M, 0.001f, 0.05f, rng);
            QuantValues ws, bias;
            matmul_params params;
            params.dtypes.src = data_type_t::s8;
            params.dtypes.wei = data_type_t::s8;
            params.dtypes.dst = dst_dt;
            params.dtypes.bias = data_type_t::bf16;
            auto &qp = params.quant_params;
            qp.src_scale.buff = ss.data();
            qp.src_scale.dt = data_type_t::f32;
            qp.src_scale.dims = {M, 1};
            qp.wei_scale.buff = ws.store(
                    random_floats(N, 0.001f, 0.05f, rng), data_type_t::bf16);
            qp.wei_scale.dt = data_type_t::bf16;
            qp.wei_scale.dims = {1, N};
            const void *bias_ptr = bias.store(
                    random_floats(N, -2.0f, 2.0f, rng), data_type_t::bf16);
            auto desc
                    = make_desc(M, N, K, K, false, data_type_t::s8, dst_dt, 1);
            desc.ldc = ldc;
            nat::native_int8_call_t call;
            ASSERT_EQ(
                    prepare(desc, params, src.data(), call), status_t::success);
            ASSERT_TRUE(
                    take_bias(desc, params, bias_ptr, data_type_t::bf16, call));
            const size_t bytes = static_cast<size_t>(M) * ldc * (f32 ? 4 : 2);
            std::vector<uint8_t> stored(bytes, 0x5A), direct(bytes, 0x5A);
            nat::native_epilogue_t ep;
            ASSERT_EQ(nat::prepare_epilogue(
                              desc, params, nullptr, stored.data(), false, ep),
                    status_t::success);
            nat::native_int8_out_t out;
            out.dst = direct.data();
            out.bf16 = !f32;
            out.ldc = ldc;
            std::vector<float> scratch(static_cast<size_t>(M) * 32);
            for (int n0 = 0; n0 < N; n0 += 32) {
                nat::compute_int8_tile(int8_plan(), call, w.view, 0, M, n0, 32,
                        scratch.data(), 32);
                nat::apply_epilogue_tile(ep, scratch.data(), 32, 0, M, n0, 32);
                nat::compute_int8_tile(int8_plan(), call, w.view, 0, M, n0, 32,
                        scratch.data(), 32, &out);
            }
            EXPECT_EQ(stored, direct)
                    << "K=" << K << " dst=" << (f32 ? "f32" : "bf16");
        }
    }
}

// alpha and beta in the reference algorithm's order: ((alpha * S + beta * C
// + Z) * src_scale) * wei_scale, one rounding per step, C read as float the
// way the reference reads the destination.  Calls give per-row scales only
// with per-row zero points or none, and an f32 / bf16 destination; the other
// variants unit-test the finish, which reads every granularity and
// destination type the same way.
TEST(NativeBrgemmInt8Tile, AlphaBetaFollowReferenceOrder) {
    NATIVE_BRGEMM_INT8_SKIP_IF_NO_VNNI();
    struct AlphaBeta {
        float alpha, beta;
    };
    const AlphaBeta coeffs[] = {{0.5f, 0.0f}, {1.0f, 0.75f}, {2.5f, -1.5f},
            {0.0f, 1.0f}, {-1.25f, 0.3f}};
    int variant = 0;
    for (const bool sgn : {false, true}) {
        for (const Gran zp_gran : {Gran::none, Gran::tensor, Gran::row}) {
            for (const data_type_t dst_dt : {data_type_t::f32,
                         data_type_t::bf16, data_type_t::u8, data_type_t::s8,
                         data_type_t::f16, data_type_t::s32}) {
                const int K = (variant % 2) ? 63 : 100;
                const int pack_nr = (variant % 3) ? 32 : 64;
                // An interleaved pack (a gated activation) reads C at the
                // columns its pack columns hold.
                const bool interleave = (variant % 4) == 1;
                const AlphaBeta ab = coeffs[variant % 5];
                ++variant;
                const int M = 13, N = 128, ldc = N + 3;
                std::mt19937 rng(static_cast<uint32_t>(variant));
                CkWeight w = make_weight(K, N, pack_nr, true, interleave, rng);
                std::vector<uint8_t> src(static_cast<size_t>(M) * K);
                std::uniform_int_distribution<int> byte(0, 255);
                for (auto &b : src)
                    b = static_cast<uint8_t>(byte(rng));
                std::vector<float> ss = random_floats(M, 0.001f, 0.05f, rng);
                QuantValues ws;
                const void *ws_ptr
                        = ws.store(random_floats(N, 0.001f, 0.05f, rng),
                                data_type_t::bf16);
                std::vector<int32_t> zp(zp_gran == Gran::row ? M : 1);
                for (auto &z : zp)
                    z = sgn ? byte(rng) - 128 : byte(rng);
                const size_t dst_elems = static_cast<size_t>(M) * ldc;
                std::vector<uint8_t> dst(dst_elems * 4);
                const auto c_old = random_floats(dst_elems, -60.0f, 60.0f, rng);
                for (size_t i = 0; i < dst_elems; ++i) {
                    if (dst_dt == data_type_t::f32)
                        std::memcpy(dst.data() + 4 * i, &c_old[i], 4);
                    else if (dst_dt == data_type_t::bf16) {
                        const bfloat16_t v(c_old[i]);
                        std::memcpy(dst.data() + 2 * i, &v, 2);
                    } else if (dst_dt == data_type_t::f16) {
                        const float16_t v(c_old[i]);
                        std::memcpy(dst.data() + 2 * i, &v, 2);
                    } else if (dst_dt == data_type_t::s32) {
                        // Past 2^24, so that reading it as float rounds.
                        const int32_t v
                                = static_cast<int32_t>(c_old[i] * 16777216.0f);
                        std::memcpy(dst.data() + 4 * i, &v, 4);
                    } else {
                        dst[i] = static_cast<uint8_t>(byte(rng));
                    }
                }

                matmul_params params;
                params.dtypes.src = sgn ? data_type_t::s8 : data_type_t::u8;
                params.dtypes.wei = data_type_t::s8;
                params.dtypes.dst = dst_dt;
                auto &qp = params.quant_params;
                qp.src_scale.buff = ss.data();
                qp.src_scale.dt = data_type_t::f32;
                qp.src_scale.dims = {M, 1};
                if (zp_gran != Gran::none) {
                    qp.src_zp.buff = zp.data();
                    qp.src_zp.dt = data_type_t::s32;
                    qp.src_zp.dims = {static_cast<int64_t>(zp.size()), 1};
                }
                qp.wei_scale.buff = ws_ptr;
                qp.wei_scale.dt = data_type_t::bf16;
                qp.wei_scale.dims = {1, N};
                auto desc = make_desc(
                        M, N, K, K, false, params.dtypes.src, dst_dt, 4);
                desc.alpha = ab.alpha;
                desc.beta = ab.beta;
                nat::native_int8_call_t call;
                ASSERT_EQ(prepare(desc, params, src.data(), call),
                        status_t::success);
                const float bias_f32 = 1.0f;
                EXPECT_FALSE(take_bias(
                        desc, params, &bias_f32, data_type_t::f32, call));
                call.dst = dst.data();
                call.dst_dt = dst_dt;
                call.ldc = ldc;
                const auto got = run_tiles(call, w.view, M, 12, 32, 4);

                std::vector<float> want(static_cast<size_t>(M) * N);
                for (int m = 0; m < M; ++m) {
                    const int32_t z = zp_gran == Gran::none
                            ? 0
                            : zp[zp_gran == Gran::row ? m : 0];
                    for (int p = 0; p < N; ++p) {
                        const int col = w.view.logical_col(p);
                        int64_t s = 0;
                        int32_t col_sum = 0;
                        for (int k = 0; k < K; ++k) {
                            const uint8_t b
                                    = src[static_cast<size_t>(m) * K + k];
                            const int a = sgn ? static_cast<int8_t>(b) : b;
                            const int8_t wv
                                    = w.nk[static_cast<size_t>(col) * K + k];
                            s += static_cast<int64_t>(a) * wv;
                            col_sum += wv;
                        }
                        float v = static_cast<float>(static_cast<int32_t>(s));
                        if (ab.alpha != 1.0f) v *= ab.alpha;
                        if (ab.beta != 0.0f) {
                            const size_t at
                                    = static_cast<size_t>(m) * ldc + col;
                            float c = 0.0f;
                            if (dst_dt == data_type_t::f32)
                                std::memcpy(&c, dst.data() + 4 * at, 4);
                            else if (dst_dt == data_type_t::bf16) {
                                bfloat16_t h;
                                std::memcpy(&h, dst.data() + 2 * at, 2);
                                c = static_cast<float>(h);
                            } else if (dst_dt == data_type_t::f16) {
                                float16_t h;
                                std::memcpy(&h, dst.data() + 2 * at, 2);
                                c = static_cast<float>(h);
                            } else if (dst_dt == data_type_t::s32) {
                                int32_t x;
                                std::memcpy(&x, dst.data() + 4 * at, 4);
                                c = static_cast<float>(x);
                            } else if (dst_dt == data_type_t::s8)
                                c = static_cast<int8_t>(dst[at]);
                            else
                                c = dst[at];
                            v += c * ab.beta;
                        }
                        if (z != 0) v += static_cast<float>(-z * col_sum);
                        v *= ss[m];
                        v *= ws.rounded[col];
                        want[static_cast<size_t>(m) * N + p] = v;
                    }
                }
                EXPECT_TRUE(same_bits(got, want, N))
                        << "signed=" << sgn
                        << " zp=" << static_cast<int>(zp_gran)
                        << " dst=" << static_cast<int>(dst_dt)
                        << " alpha=" << ab.alpha << " beta=" << ab.beta
                        << " K=" << K << " pack_nr=" << pack_nr
                        << " interleave=" << interleave;
            }
        }
    }
}

TEST(NativeBrgemmInt8Tile, KernelTable) {
    for (int nv : {1, 2}) {
        const int max_mr = nat::native_int8_ukernel_max_mr(nv);
        EXPECT_GE(max_mr, 6);
        for (bool sgn : {false, true}) {
            EXPECT_EQ(nat::select_native_int8_ukernel(0, nv, sgn), nullptr);
            EXPECT_EQ(nat::select_native_int8_ukernel(max_mr + 1, nv, sgn),
                    nullptr);
            for (int mr = 1; mr <= max_mr; ++mr)
                EXPECT_NE(
                        nat::select_native_int8_ukernel(mr, nv, sgn), nullptr);
        }
    }
    for (int nv : {3, 4}) {
        EXPECT_EQ(nat::native_int8_ukernel_max_mr(nv), 0);
        EXPECT_EQ(nat::select_native_int8_ukernel(1, nv, false), nullptr);
    }
}

// ── Against the custom-kernel INT8 microkernel on the same operands ───────

enum class CkCompute { sym, asym, static_u8 };

struct CkKernelCase {
    int M, N, K, pack_nr;
    bool interleave;
    CkCompute compute;
    data_type_t bias_dt;
    bool wei_per_tensor;
};

std::string ck_kernel_name(const ::testing::TestParamInfo<CkKernelCase> &info) {
    const auto &c = info.param;
    const char *cmp = c.compute == CkCompute::sym ? "sym"
            : c.compute == CkCompute::asym        ? "asym"
                                                  : "static";
    return "c" + std::to_string(info.index) + "_" + cmp + "_M"
            + std::to_string(c.M) + "_K" + std::to_string(c.K) + "_nr"
            + std::to_string(c.pack_nr) + (c.interleave ? "_il" : "");
}

std::vector<CkKernelCase> ck_kernel_cases() {
    std::vector<CkKernelCase> v;
    const data_type_t biases[] = {data_type_t::none, data_type_t::f32,
            data_type_t::bf16, data_type_t::f16};
    int i = 0;
    for (CkCompute cmp :
            {CkCompute::sym, CkCompute::asym, CkCompute::static_u8})
        for (int pack_nr : {32, 64})
            for (int M : {1, 3, 8, 13, 64})
                for (int K : {4, 64, 1000})
                    v.push_back({M, 128, K, pack_nr, (i % 3) == 1, cmp,
                            biases[i++ % 4],
                            cmp == CkCompute::static_u8 && (i % 2) == 0});
    return v;
}

class NativeBrgemmInt8VsCkKernel
    : public ::testing::TestWithParam<CkKernelCase> {};

TEST_P(NativeBrgemmInt8VsCkKernel, BitIdentical) {
    NATIVE_BRGEMM_INT8_SKIP_IF_NO_VNNI();
    const CkKernelCase c = GetParam();
    std::mt19937 rng(static_cast<uint32_t>(c.M * 17 + c.K * 5 + c.pack_nr));
    CkWeight w = make_weight(c.K, c.N, c.pack_nr, true, c.interleave, rng);

    const bool sym = c.compute == CkCompute::sym;
    const bool per_tensor = c.compute == CkCompute::static_u8;
    std::vector<uint8_t> src(static_cast<size_t>(c.M) * c.K);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto &b : src)
        b = static_cast<uint8_t>(byte(rng));
    const int n_src = per_tensor ? 1 : c.M;
    std::vector<float> src_scale = random_floats(n_src, 0.001f, 0.05f, rng);
    std::vector<int32_t> src_zp(n_src);
    for (auto &z : src_zp)
        z = byte(rng);
    const int n_wei = c.wei_per_tensor ? 1 : c.N;
    std::vector<float> wei_scale = random_floats(n_wei, 0.001f, 0.05f, rng);
    QuantValues bias;
    const void *bias_ptr = nullptr;
    if (c.bias_dt != data_type_t::none)
        bias_ptr = bias.store(random_floats(c.N, -2.0f, 2.0f, rng), c.bias_dt);

    // The custom kernel reads scales and bias in pack column order.
    std::vector<float> wei_pack(c.N), bias_pack(c.N);
    std::vector<bfloat16_t> bias_pack_bf16(c.N);
    std::vector<float16_t> bias_pack_f16(c.N);
    for (int p = 0; p < c.N; ++p) {
        const int col = w.view.logical_col(p);
        wei_pack[p] = wei_scale[c.wei_per_tensor ? 0 : col];
        if (bias_ptr) {
            bias_pack[p] = bias.rounded[col];
            bias_pack_bf16[p]
                    = bias.bf16.empty() ? bfloat16_t(0.0f) : bias.bf16[col];
            bias_pack_f16[p]
                    = bias.f16.empty() ? float16_t(0.0f) : bias.f16[col];
        }
    }
    ck::BiasKind bias_kind = ck::BiasKind::none;
    const void *ck_bias = nullptr;
    if (c.bias_dt == data_type_t::f32) {
        bias_kind = ck::BiasKind::fp32;
        ck_bias = bias_pack.data();
    } else if (c.bias_dt == data_type_t::bf16) {
        bias_kind = ck::BiasKind::bf16;
        ck_bias = bias_pack_bf16.data();
    } else if (c.bias_dt == data_type_t::f16) {
        bias_kind = ck::BiasKind::f16;
        ck_bias = bias_pack_f16.data();
    }
    const size_t bias_elem = c.bias_dt == data_type_t::f32 ? 4 : 2;

    const int NV = c.pack_nr / 16;
    const auto compute
            = sym ? ck::IntCompute::kS8_Sym : ck::IntCompute::kU8_Asym;
    const int max_mr = ck::max_mr_for_nv_int8(NV, compute);
    ck::Int8EpilogueConfig epi;
    epi.src_quant_per_tensor = per_tensor;
    epi.wei_scale_per_tensor = c.wei_per_tensor;
    std::vector<float> ck_out(static_cast<size_t>(c.M) * c.N, -1.0f);
    for (int ob = 0; ob < c.N / c.pack_nr; ++ob) {
        const int8_t *blk = reinterpret_cast<const int8_t *>(w.view.block(ob));
        for (int r0 = 0; r0 < c.M; r0 += max_mr) {
            const int rows = std::min(max_mr, c.M - r0);
            auto fn = ck::select_int8_ukernel(rows, NV, compute,
                    ck::ActivationKind::none, ck::OutputType::kF32);
            ASSERT_NE(fn, nullptr);
            const int s_off = per_tensor ? 0 : r0;
            fn(src.data() + static_cast<size_t>(r0) * c.K, c.K, blk,
                    src_scale.data() + s_off,
                    sym ? nullptr : src_zp.data() + s_off,
                    c.wei_per_tensor ? wei_pack.data()
                                     : wei_pack.data() + ob * c.pack_nr,
                    ck::ScaleKind::kF32, epi,
                    ck_bias ? static_cast<const char *>(ck_bias)
                                    + static_cast<size_t>(ob) * c.pack_nr
                                            * bias_elem
                            : nullptr,
                    bias_kind,
                    ck_out.data() + static_cast<size_t>(r0) * c.N
                            + ob * c.pack_nr,
                    c.N, nullptr, 0, c.K);
        }
    }

    matmul_params params;
    params.dtypes.src = sym ? data_type_t::s8 : data_type_t::u8;
    params.dtypes.wei = data_type_t::s8;
    params.dtypes.dst = data_type_t::f32;
    params.dtypes.bias = c.bias_dt;
    auto &qp = params.quant_params;
    qp.src_scale = {};
    qp.src_scale.buff = src_scale.data();
    qp.src_scale.dt = data_type_t::f32;
    qp.src_scale.dims = {n_src, 1};
    if (!sym) {
        qp.src_zp.buff = src_zp.data();
        qp.src_zp.dt = data_type_t::s32;
        qp.src_zp.dims = {n_src, 1};
    }
    qp.wei_scale.buff = wei_scale.data();
    qp.wei_scale.dt = data_type_t::f32;
    qp.wei_scale.dims = {1, n_wei};
    const auto desc = make_desc(
            c.M, c.N, c.K, c.K, false, params.dtypes.src, data_type_t::f32, 4);
    nat::native_int8_call_t call;
    ASSERT_EQ(prepare(desc, params, src.data(), call), status_t::success);
    EXPECT_EQ(take_bias(desc, params, bias_ptr, c.bias_dt, call),
            bias_ptr != nullptr);
    for (int nr : {16, c.pack_nr}) {
        const auto got = run_tiles(call, w.view, c.M, 12, nr, 4);
        EXPECT_TRUE(same_bits(got, ck_out, c.N)) << "tile width " << nr;
    }
}

INSTANTIATE_TEST_SUITE_P(NativeBrgemmInt8, NativeBrgemmInt8VsCkKernel,
        ::testing::ValuesIn(ck_kernel_cases()), ck_kernel_name);

// ── Against the custom-kernel route on a bf16 source quantized per token ──

struct RouteCase {
    int M, N, K;
    bool asym;
    data_type_t scale_dt;
    data_type_t dst_dt;
    data_type_t bias_dt;
    bool group_dq;
};

std::string route_name(const ::testing::TestParamInfo<RouteCase> &info) {
    const auto &c = info.param;
    return "c" + std::to_string(info.index) + (c.asym ? "_asym" : "_sym") + "_M"
            + std::to_string(c.M) + "_K" + std::to_string(c.K)
            + (c.dst_dt == data_type_t::f32 ? "_f32" : "_bf16")
            + (c.group_dq ? "_gdq" : "_hoist");
}

std::vector<RouteCase> route_cases() {
    std::vector<RouteCase> v;
    const data_type_t biases[]
            = {data_type_t::none, data_type_t::f32, data_type_t::bf16};
    int i = 0;
    for (bool asym : {false, true})
        for (bool gdq : {true, false})
            for (data_type_t dst : {data_type_t::bf16, data_type_t::f32})
                for (int M : {1, 4, 7, 16, 64, 65, 200})
                    v.push_back({M, 256, (i % 2) ? 1024 : 256, asym,
                            (i % 3) ? data_type_t::f32 : data_type_t::bf16, dst,
                            biases[i++ % 3], gdq});
    return v;
}

class NativeBrgemmInt8VsCkRoute : public ::testing::TestWithParam<RouteCase> {};

TEST_P(NativeBrgemmInt8VsCkRoute, DynamicQuantBitIdentical) {
    NATIVE_BRGEMM_INT8_SKIP_IF_NO_VNNI();
    const RouteCase c = GetParam();
    // The group matmul caches start and end empty on every exit.
    struct caches_reset_t {
        caches_reset_t() { ::reset_grp_matmul_caches(); }
        ~caches_reset_t() { ::reset_grp_matmul_caches(); }
    } const caches_reset;
    mt::AlgoEnvGuard algo(3);
    mt::CustomKernelOverride ck_on(true);
    mt::CustomKernelInt8Override ck_int8_on(true);
    mt::NTileStrategyOverride rounds(2);
    mt::GroupDqOverride group_dq(c.group_dq ? 1 : 0);

    std::mt19937 rng(static_cast<uint32_t>(c.M * 29 + c.K + (c.asym ? 7 : 0)));
    std::vector<int8_t> weight(static_cast<size_t>(c.N) * c.K);
    std::uniform_int_distribution<int> wd(-128, 127);
    for (auto &x : weight)
        x = static_cast<int8_t>(wd(rng));
    std::vector<bfloat16_t> src(static_cast<size_t>(c.M) * c.K);
    {
        const auto f = random_floats(src.size(), -4.0f, 4.0f, rng);
        for (size_t i = 0; i < f.size(); ++i)
            src[i] = bfloat16_t(f[i]);
    }
    QuantValues wei_scale;
    const void *wei_scale_ptr = wei_scale.store(
            random_floats(c.N, 0.001f, 0.05f, rng), c.scale_dt);
    QuantValues bias;
    const void *bias_ptr = nullptr;
    if (c.bias_dt != data_type_t::none)
        bias_ptr = bias.store(random_floats(c.N, -2.0f, 2.0f, rng), c.bias_dt);

    const size_t scale_bytes = static_cast<size_t>(c.M)
            * (c.scale_dt == data_type_t::f32 ? 4 : 2);
    std::vector<uint8_t> ck_scale(scale_bytes, 0), nat_scale(scale_bytes, 0);
    std::vector<int32_t> ck_zp(c.M, -1), nat_zp(c.M, -2);
    const auto make_params = [&](void *scale, int32_t *zp) {
        matmul_params p;
        p.dtypes.src = data_type_t::bf16;
        p.dtypes.wei = data_type_t::s8;
        p.dtypes.dst = c.dst_dt;
        p.dtypes.bias = c.bias_dt;
        p.dtypes.compute = c.asym ? data_type_t::u8 : data_type_t::s8;
        p.dynamic_quant = true;
        p.weight_cache_type = 1;
        p.num_threads = 4;
        p.quant_params.src_scale.buff = scale;
        p.quant_params.src_scale.dt = c.scale_dt;
        p.quant_params.src_scale.dims = {c.M, 1};
        if (c.asym) {
            p.quant_params.src_zp.buff = zp;
            p.quant_params.src_zp.dt = data_type_t::s32;
            p.quant_params.src_zp.dims = {c.M, 1};
        }
        p.quant_params.wei_scale.buff = wei_scale_ptr;
        p.quant_params.wei_scale.dt = c.scale_dt;
        p.quant_params.wei_scale.dims = {1, c.N};
        return p;
    };

    // Which calls the custom kernel serves is that route's rule: a call it
    // does not serve is checked against the integer reference below.
    bool ck_serves = false;
    const size_t dst_elems = static_cast<size_t>(c.M) * c.N;
    std::vector<float> ck_f32(dst_elems, -7.0f);
    std::vector<bfloat16_t> ck_bf16(dst_elems, bfloat16_t(-7.0f));
    void *ck_dst = c.dst_dt == data_type_t::f32
            ? static_cast<void *>(ck_f32.data())
            : static_cast<void *>(ck_bf16.data());
    {
        // No dynamic team sizes while the custom-kernel route runs.
        struct dynamic_off_t {
            const int was = omp_get_dynamic();
            dynamic_off_t() { omp_set_dynamic(0); }
            ~dynamic_off_t() { omp_set_dynamic(was); }
        } const dynamic_off;
        mt::ExecutionPathCaptureGuard mode;
        const status_t st = mm::group_matmul_direct({'r'}, {false}, {true},
                {c.M}, {c.N}, {c.K}, {1.0f}, {src.data()}, {c.K},
                {weight.data()}, {c.K}, {bias_ptr}, {0.0f}, {ck_dst}, {c.N},
                {true}, {make_params(ck_scale.data(), ck_zp.data())});
        ASSERT_EQ(st, status_t::success);
        const char *execution_path
                = mm::test_api::s_last_group_matmul_direct_execution_path.load(
                        std::memory_order_relaxed);
        ASSERT_NE(execution_path, nullptr);
        ck_serves = std::string(execution_path).find("custom")
                != std::string::npos;
        RecordProperty("ck_execution_path", execution_path);
    }

    CkWeight w;
    w.K = c.K;
    w.N = c.N;
    const size_t bytes = ck::packed_weight_size_int8(c.K, c.N, 32);
    w.packed.resize(bytes);
    ASSERT_EQ(ck::prepack_weight_into_int8(weight.data(), c.K, c.N, c.K, 32,
                      true, false, w.packed.data()),
            status_t::success);
    w.view.base = w.packed.data();
    w.view.wei_dt = data_type_t::s8;
    w.view.K = c.K;
    w.view.N = c.N;
    w.view.pack_nr = 32;
    w.view.k_groups = (c.K + 3) / 4;
    w.view.block_bytes = static_cast<size_t>(w.view.k_groups) * 32 * 4 + 32 * 4;

    const auto params = make_params(nat_scale.data(), nat_zp.data());
    const auto desc = make_desc(
            c.M, c.N, c.K, c.K, false, data_type_t::bf16, c.dst_dt, 4);
    nat::native_int8_call_t call;
    ASSERT_EQ(prepare(desc, params, src.data(), call), status_t::success);
    EXPECT_EQ(take_bias(desc, params, bias_ptr, c.bias_dt, call),
            bias_ptr != nullptr);
    EXPECT_EQ(call.src_signed, !c.asym);
    const auto got = run_tiles(call, w.view, c.M, 12, 32, 4);

    EXPECT_EQ(std::memcmp(ck_scale.data(), nat_scale.data(), scale_bytes), 0)
            << "per-token scales written back differ";
    if (c.asym) {
        EXPECT_EQ(ck_zp, nat_zp) << "zero points written back differ";
    }
    if (!ck_serves) {
        // That fallback applies the first row's zero point to every row, so
        // the same quantized operands are checked against the reference.
        w.nk = weight;
        std::vector<float> scale(c.M);
        for (int m = 0; m < c.M; ++m) {
            if (c.scale_dt == data_type_t::f32) {
                std::memcpy(&scale[m], nat_scale.data() + 4 * m, 4);
            } else {
                bfloat16_t s;
                std::memcpy(&s, nat_scale.data() + 2 * m, 2);
                scale[m] = static_cast<float>(s);
            }
        }
        Dequant d;
        d.zp = c.asym ? nat_zp.data() : nullptr;
        d.zp_stride = 1;
        d.src_scale = scale.data();
        d.src_scale_stride = 1;
        d.wei_scale = wei_scale.rounded.data();
        d.wei_scale_stride = 1;
        d.bias = bias_ptr ? bias.rounded.data() : nullptr;
        const auto want = reference(
                call_values(call, c.M, c.K), c.M, c.K, w, !c.asym, d);
        EXPECT_TRUE(same_bits(got, want, c.N));
    } else if (c.dst_dt == data_type_t::f32) {
        EXPECT_TRUE(same_bits(got, ck_f32, c.N));
    } else {
        static_assert(sizeof(bfloat16_t) == sizeof(uint16_t), "bf16 bits");
        size_t bad = 0;
        for (size_t i = 0; i < dst_elems; ++i) {
            uint16_t want = 0;
            std::memcpy(&want, &ck_bf16[i], sizeof(want));
            const uint16_t have = static_cast<uint16_t>(
                    bfloat16_t::f32_to_bf16_val(got[i]));
            if (want != have && bad++ < 3)
                ADD_FAILURE() << "bf16 differs at (" << i / c.N << ", "
                              << i % c.N << "): " << got[i] << " vs "
                              << static_cast<float>(ck_bf16[i]);
        }
        EXPECT_EQ(bad, 0u);
    }
}

INSTANTIATE_TEST_SUITE_P(NativeBrgemmInt8, NativeBrgemmInt8VsCkRoute,
        ::testing::ValuesIn(route_cases()), route_name);

// ── Dynamic quantization on paths the custom kernel does not take ─────────

struct DynamicSetup {
    int M, K;
    bool asym;
    std::vector<bfloat16_t> src; ///< [M][K]
    std::vector<float> scale;
    std::vector<int32_t> zp;
    std::vector<float> wei_scale;

    matmul_params params(int N, data_type_t src_dt = data_type_t::bf16) {
        matmul_params p;
        p.dtypes.src = src_dt;
        p.dtypes.wei = data_type_t::s8;
        p.dtypes.dst = data_type_t::f32;
        p.dtypes.compute = asym ? data_type_t::u8 : data_type_t::s8;
        p.dynamic_quant = true;
        p.quant_params.src_scale.buff = scale.data();
        p.quant_params.src_scale.dt = data_type_t::f32;
        p.quant_params.src_scale.dims = {M, 1};
        if (asym) {
            p.quant_params.src_zp.buff = zp.data();
            p.quant_params.src_zp.dt = data_type_t::s32;
            p.quant_params.src_zp.dims = {M, 1};
        }
        p.quant_params.wei_scale.buff = wei_scale.data();
        p.quant_params.wei_scale.dt = data_type_t::f32;
        p.quant_params.wei_scale.dims = {1, N};
        return p;
    }
};

DynamicSetup make_dynamic(int M, int K, int N, bool asym, std::mt19937 &rng) {
    DynamicSetup s;
    s.M = M;
    s.K = K;
    s.asym = asym;
    const auto f = random_floats(static_cast<size_t>(M) * K, -3.0f, 5.0f, rng);
    s.src.assign(f.begin(), f.end());
    s.scale.assign(M, 0.0f);
    s.zp.assign(M, 0);
    s.wei_scale = random_floats(N, 0.001f, 0.05f, rng);
    return s;
}

TEST(NativeBrgemmInt8Dynamic, TransposedSourceQuantizesLikeRowMajor) {
    NATIVE_BRGEMM_INT8_SKIP_IF_NO_VNNI();
    for (const bool asym : {false, true}) {
        for (const bool group_dq : {true, false}) {
            mt::GroupDqOverride gdq(group_dq ? 1 : 0);
            const int M = 21, K = 100, N = 64;
            std::mt19937 rng(asym ? 3 : 4);
            CkWeight w = make_weight(K, N, 32, true, false, rng);
            DynamicSetup row = make_dynamic(M, K, N, asym, rng);
            DynamicSetup col = row;
            const int lda_t = M + 3;
            std::vector<bfloat16_t> src_t(static_cast<size_t>(K) * lda_t);
            for (int m = 0; m < M; ++m)
                for (int k = 0; k < K; ++k)
                    src_t[static_cast<size_t>(k) * lda_t + m]
                            = row.src[static_cast<size_t>(m) * K + k];

            nat::native_int8_call_t call_row, call_col;
            const auto p_row = row.params(N);
            const auto p_col = col.params(N);
            ASSERT_EQ(prepare(make_desc(M, N, K, K, false, data_type_t::bf16,
                                      data_type_t::f32, 4),
                              p_row, row.src.data(), call_row),
                    status_t::success);
            ASSERT_EQ(prepare(make_desc(M, N, K, lda_t, true, data_type_t::bf16,
                                      data_type_t::f32, 4),
                              p_col, src_t.data(), call_col),
                    status_t::success);
            EXPECT_EQ(call_values(call_row, M, K), call_values(call_col, M, K));
            EXPECT_EQ(row.scale, col.scale);
            if (asym) { EXPECT_EQ(row.zp, col.zp); }
            const auto a = run_tiles(call_row, w.view, M, 12, 32, 4);
            const auto b = run_tiles(call_col, w.view, M, 5, 16, 3);
            EXPECT_TRUE(same_bits(a, b, N));
        }
    }
}

TEST(NativeBrgemmInt8Dynamic, UnalignedKPadsQuantizedRows) {
    NATIVE_BRGEMM_INT8_SKIP_IF_NO_VNNI();
    for (const bool asym : {false, true}) {
        for (const int K : {1, 2, 3, 5, 31, 63, 101}) {
            const int M = 9, N = 64;
            std::mt19937 rng(K * 2 + (asym ? 1 : 0));
            CkWeight w = make_weight(K, N, 64, false, true, rng);
            DynamicSetup s = make_dynamic(M, K, N, asym, rng);
            const auto params = s.params(N);
            nat::native_int8_call_t call;
            ASSERT_EQ(prepare(make_desc(M, N, K, K, false, data_type_t::bf16,
                                      data_type_t::f32, 2),
                              params, s.src.data(), call),
                    status_t::success);
            const int k_pad = (K + 3) / 4 * 4;
            ASSERT_EQ(call.lda, k_pad);
            for (int m = 0; m < M; ++m)
                for (int k = K; k < k_pad; ++k)
                    EXPECT_EQ(call.A[static_cast<size_t>(m) * k_pad + k], 0);
            const auto a = call_values(call, M, K);
            for (int m = 0; m < M; ++m) {
                float absmax = 0.0f, lo = 0.0f, hi = 0.0f;
                for (int k = 0; k < K; ++k) {
                    const float x
                            = static_cast<float>(s.src[size_t(m) * K + k]);
                    absmax = std::max(absmax, std::fabs(x));
                    lo = k ? std::min(lo, x) : x;
                    hi = k ? std::max(hi, x) : x;
                }
                ASSERT_GT(s.scale[m], 0.0f);
                for (int k = 0; k < K; ++k) {
                    const float x
                            = static_cast<float>(s.src[size_t(m) * K + k]);
                    const int q = a[static_cast<size_t>(m) * K + k];
                    if (!asym) {
                        const float scale = std::max(absmax, 1e-10f) / 127.0f;
                        EXPECT_EQ(s.scale[m], scale);
                        EXPECT_EQ(
                                q, static_cast<int>(std::nearbyint(x / scale)));
                    } else {
                        EXPECT_GE(q, 0);
                        EXPECT_LE(q, 255);
                        // A half tie at the row maximum can clamp to 255.
                        EXPECT_NEAR(
                                static_cast<double>(q - s.zp[m]) * s.scale[m],
                                x, 1.01f * s.scale[m] + 1e-6f);
                    }
                }
            }
            Dequant d;
            d.zp = asym ? s.zp.data() : nullptr;
            d.zp_stride = 1;
            d.src_scale = s.scale.data();
            d.src_scale_stride = 1;
            d.wei_scale = s.wei_scale.data();
            d.wei_scale_stride = 1;
            const auto got = run_tiles(call, w.view, M, 12, 64, 2);
            const auto want = reference(a, M, K, w, !asym, d);
            EXPECT_TRUE(same_bits(got, want, N)) << "K=" << K;
        }
    }
}

TEST(NativeBrgemmInt8Dynamic, F32SourcePerTensorAndOwnedScales) {
    NATIVE_BRGEMM_INT8_SKIP_IF_NO_VNNI();
    const int M = 6, K = 64, N = 64;
    std::mt19937 rng(11);
    CkWeight w = make_weight(K, N, 32, true, false, rng);
    const auto f = random_floats(static_cast<size_t>(M) * K, -2.0f, 2.0f, rng);
    std::vector<float> wei_scale = random_floats(N, 0.001f, 0.05f, rng);
    for (const bool per_tensor : {false, true}) {
        for (const bool owned : {false, true}) {
            std::vector<float> scale(per_tensor ? 1 : M, 0.0f);
            matmul_params p;
            p.dtypes.src = data_type_t::f32;
            p.dtypes.wei = data_type_t::s8;
            p.dtypes.dst = data_type_t::f32;
            p.dtypes.compute = data_type_t::s8;
            p.dynamic_quant = true;
            p.quant_params.src_scale.buff = owned ? nullptr : scale.data();
            p.quant_params.src_scale.dt = data_type_t::f32;
            p.quant_params.src_scale.dims = {per_tensor ? 1 : M, 1};
            p.quant_params.wei_scale.buff = wei_scale.data();
            p.quant_params.wei_scale.dt = data_type_t::f32;
            p.quant_params.wei_scale.dims = {1, N};
            nat::native_int8_call_t call;
            ASSERT_EQ(prepare(make_desc(M, N, K, K, false, data_type_t::f32,
                                      data_type_t::f32, 4),
                              p, f.data(), call),
                    status_t::success);
            EXPECT_EQ(call.src_scale_stride, per_tensor ? 0 : 1);
            if (!owned) { EXPECT_EQ(call.src_scale, scale.data()); }
            float absmax = 0.0f;
            for (int i = 0; i < (per_tensor ? M * K : K); ++i)
                absmax = std::max(absmax, std::fabs(f[i]));
            EXPECT_EQ(call.src_scale[0], absmax / 127.0f);
            Dequant d;
            d.src_scale = call.src_scale;
            d.src_scale_stride = call.src_scale_stride;
            d.wei_scale = wei_scale.data();
            d.wei_scale_stride = 1;
            const auto got = run_tiles(call, w.view, M, 12, 32, 4);
            const auto want
                    = reference(call_values(call, M, K), M, K, w, true, d);
            EXPECT_TRUE(same_bits(got, want, N));
        }
    }
}

// ── End to end through native_brgemm_execute ─────────────────────────────────

namespace nbt = native_brgemm_test;

enum class E2ESrc { dyn_sym, dyn_asym, static_u8, static_s8_row_zp };

struct E2ECase {
    int M, N, K;
    E2ESrc src;
    data_type_t dst_dt;
    data_type_t bias_dt;
    /// Pack for the call only (weight cache 0) rather than through the
    /// native cache (weight cache 1).
    bool per_call_pack;
    bool transA = false;
    float alpha = 1.0f, beta = 0.0f;
    bool relu = false;
    /// A dynamically quantized source in f32 rather than bf16.
    bool f32_src = false;
};

const char *dst_name(data_type_t dt) {
    switch (dt) {
        case data_type_t::f32: return "f32";
        case data_type_t::bf16: return "bf16";
        case data_type_t::u8: return "u8";
        case data_type_t::s8: return "s8";
        case data_type_t::s32: return "s32";
        default: return "other";
    }
}

std::string e2e_name(const ::testing::TestParamInfo<E2ECase> &info) {
    const auto &c = info.param;
    const char *src = c.src == E2ESrc::dyn_sym ? "dynsym"
            : c.src == E2ESrc::dyn_asym        ? "dynasym"
            : c.src == E2ESrc::static_u8       ? "u8"
                                               : "s8rowzp";
    return "c" + std::to_string(info.index) + "_" + src + "_M"
            + std::to_string(c.M) + "_K" + std::to_string(c.K) + "_"
            + dst_name(c.dst_dt) + (c.transA ? "_tA" : "")
            + (c.per_call_pack ? "_wc0" : "_wc1")
            + (c.alpha != 1.0f || c.beta != 0.0f ? "_ab" : "")
            + (c.f32_src ? "_f32src" : "");
}

/// Operands of one INT8 call and the call itself (raw weight, transB).
struct E2EProblem {
    nbt::buffer_t src, wei, bias, wei_scale, src_scale, src_zp, dst_scale,
            dst_zp;
    nbt::call_t call;
    int weight_cache_mode = 1;
};

E2EProblem make_e2e(const E2ECase &c, uint32_t seed) {
    E2EProblem pb;
    const bool dynamic = c.src == E2ESrc::dyn_sym || c.src == E2ESrc::dyn_asym;
    const data_type_t src_dt = dynamic
            ? (c.f32_src ? data_type_t::f32 : data_type_t::bf16)
            : c.src == E2ESrc::static_u8 ? data_type_t::u8
                                         : data_type_t::s8;
    const size_t a_elems = static_cast<size_t>(c.M) * c.K;
    pb.src = dynamic ? nbt::random_buffer(src_dt, a_elems, seed, -3.0f, 5.0f)
                     : nbt::random_buffer(src_dt, a_elems, seed,
                               src_dt == data_type_t::u8 ? 0.0f : -128.0f,
                               src_dt == data_type_t::u8 ? 255.0f : 127.0f);
    pb.wei = nbt::random_buffer(data_type_t::s8, static_cast<size_t>(c.N) * c.K,
            seed + 1, -128.0f, 127.0f);
    if (c.bias_dt != data_type_t::none)
        pb.bias = nbt::random_buffer(c.bias_dt, c.N, seed + 2, -2.0f, 2.0f);
    pb.wei_scale = nbt::random_buffer(
            data_type_t::f32, c.N, seed + 3, 0.001f, 0.05f);
    auto &call = pb.call;
    call.M = c.M;
    call.N = c.N;
    call.K = c.K;
    call.transA = c.transA;
    call.transB = true;
    call.alpha = c.alpha;
    call.beta = c.beta;
    call.src = pb.src.get();
    call.wei = pb.wei.get();
    call.bias = c.bias_dt != data_type_t::none ? pb.bias.get() : nullptr;
    auto &p = call.params;
    p.dtypes.src = src_dt;
    p.dtypes.wei = data_type_t::s8;
    p.dtypes.dst = c.dst_dt;
    p.dtypes.bias = c.bias_dt;
    p.num_threads = 4;
    auto &qp = p.quant_params;
    const int rows = c.src == E2ESrc::static_u8 ? 1 : c.M;
    pb.src_scale = nbt::random_buffer(
            data_type_t::f32, rows, seed + 4, 0.002f, 0.04f);
    qp.src_scale.buff = pb.src_scale.get();
    qp.src_scale.dt = data_type_t::f32;
    qp.src_scale.dims = {rows, 1};
    if (dynamic) {
        p.dynamic_quant = true;
        p.dtypes.compute
                = c.src == E2ESrc::dyn_asym ? data_type_t::u8 : data_type_t::s8;
    }
    if (c.src != E2ESrc::dyn_sym) {
        pb.src_zp = nbt::random_buffer(data_type_t::s32, rows, seed + 5,
                c.src == E2ESrc::static_s8_row_zp ? -100.0f : 0.0f,
                c.src == E2ESrc::static_s8_row_zp ? 100.0f : 255.0f);
        qp.src_zp.buff = pb.src_zp.get();
        qp.src_zp.dt = data_type_t::s32;
        qp.src_zp.dims = {rows, 1};
    }
    qp.wei_scale.buff = pb.wei_scale.get();
    qp.wei_scale.dt = data_type_t::f32;
    qp.wei_scale.dims = {1, c.N};
    if (c.dst_dt == data_type_t::u8 || c.dst_dt == data_type_t::s8) {
        pb.dst_scale
                = nbt::random_buffer(data_type_t::f32, 1, seed + 6, 0.5f, 2.0f);
        pb.dst_zp = nbt::random_buffer(data_type_t::s32, 1, seed + 7,
                c.dst_dt == data_type_t::u8 ? 100.0f : -10.0f,
                c.dst_dt == data_type_t::u8 ? 130.0f : 10.0f);
        qp.dst_scale.buff = pb.dst_scale.get();
        qp.dst_scale.dt = data_type_t::f32;
        qp.dst_scale.dims = {1, 1};
        qp.dst_zp.buff = pb.dst_zp.get();
        qp.dst_zp.dt = data_type_t::s32;
        qp.dst_zp.dims = {1, 1};
    }
    if (c.relu) {
        mm::matmul_post_op relu;
        relu.po_type = zendnnl::common::post_op_type_t::relu;
        p.postop_.push_back(relu);
    }
    pb.weight_cache_mode = c.per_call_pack ? 0 : 1;
    return pb;
}

/// The call through native_brgemm_execute on its raw weight, packed for the
/// call or through the native cache.
status_t run_native(const E2EProblem &pb, void *dst) {
    const auto &c = pb.call;
    nat::GemmDescriptor d;
    d.M = c.M;
    d.N = c.N;
    d.K = c.K;
    d.lda = c.lda_or_tight();
    d.ldb = c.ldb_or_tight();
    d.ldc = c.ldc_or_tight();
    d.transA = c.transA;
    d.transB = c.transB;
    d.alpha = c.alpha;
    d.beta = c.beta;
    d.src_dt = c.params.dtypes.src;
    d.wei_dt = c.params.dtypes.wei;
    d.dst_dt = c.params.dtypes.dst;
    d.bias_dt = c.params.dtypes.bias;
    d.bias = c.bias;
    d.is_weights_const = true;
    d.num_threads = c.params.num_threads;
    nat::native_weight_ref_t ref;
    ref.pack_nr = 32;
    ref.weight = c.wei;
    ref.weight_cache_mode = pb.weight_cache_mode;
    matmul_params params = c.params;
    return nat::native_brgemm_execute(d, ref, c.src, dst, c.bias, params);
}

/// The two destinations start from the same values, which beta reads.
std::vector<nbt::buffer_t> e2e_dsts(const E2ECase &c, uint32_t seed, int n) {
    const size_t elems = static_cast<size_t>(c.M) * c.N;
    const nbt::buffer_t init
            = nbt::random_buffer(c.dst_dt, elems, seed + 9, -20.0f, 20.0f);
    std::vector<nbt::buffer_t> v;
    for (int i = 0; i < n; ++i) {
        v.push_back(nbt::make_buffer(elems * nbt::dt_bytes(c.dst_dt)));
        std::memcpy(v.back().get(), init.get(), init.bytes);
    }
    return v;
}

class NativeBrgemmInt8E2E : public ::testing::TestWithParam<E2ECase> {
protected:
    void SetUp() override {
        if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
        mm::clear_matmul_weight_caches();
    }
    void TearDown() override { mm::clear_matmul_weight_caches(); }
};

// Calls the custom-kernel route serves: the same bits (u8 within one step).
class NativeBrgemmInt8E2EVsCk : public NativeBrgemmInt8E2E {};

TEST_P(NativeBrgemmInt8E2EVsCk, BitIdentical) {
    if (!nbt::default_route_reachable())
        GTEST_SKIP() << "ZENDNNL_MATMUL_ALGO pins the process to another "
                        "backend; the custom-kernel route cannot serve";
    const E2ECase c = GetParam();
    const uint32_t seed = static_cast<uint32_t>(c.M * 7 + c.K * 3 + c.N);
    E2EProblem pb = make_e2e(c, seed);
    auto dsts = e2e_dsts(c, seed, 2);
    {
        nbt::ck_route_pin_t pin;
        nbt::ck_engagement_t engagement;
        nbt::call_t ck_call = pb.call;
        ck_call.dst = dsts[0].get();
        ASSERT_EQ(nbt::run_matmul(ck_call), status_t::success);
        ASSERT_TRUE(engagement.engaged()) << "the custom kernel did not serve";
    }
    ASSERT_EQ(run_native(pb, dsts[1].get()), status_t::success);
    EXPECT_TRUE(nbt::within_steps(dsts[1].get(), c.N, dsts[0].get(), c.N,
            c.dst_dt, c.M, c.N, c.dst_dt == data_type_t::u8 ? 1 : 0));
}

std::vector<E2ECase> e2e_ck_cases() {
    const auto bf16 = data_type_t::bf16, f32 = data_type_t::f32,
               u8 = data_type_t::u8, none = data_type_t::none;
    std::vector<E2ECase> v;
    int i = 0;
    for (int M : {1, 4, 16, 64})
        for (int K : {256, 4096})
            for (data_type_t dst : {bf16, f32}) {
                const data_type_t bias = (i % 3 == 0) ? none
                        : (i % 3 == 1)                ? f32
                                                      : bf16;
                v.push_back(
                        {M, 256, K, E2ESrc::dyn_sym, dst, bias, i % 2 == 1});
                ++i;
            }
    for (data_type_t dst : {bf16, f32})
        v.push_back({1, 512, 1024, E2ESrc::dyn_asym, dst, f32, false});
    // A deep K, which native and the custom kernel split into passes.
    v.push_back({16, 256, 32768, E2ESrc::dyn_sym, bf16, none, true});
    v.push_back({64, 256, 32768, E2ESrc::dyn_sym, f32, none, false});
    for (int M : {1, 8, 64})
        for (data_type_t dst : {bf16, f32, u8})
            v.push_back({M, 256, 512, E2ESrc::static_u8, dst,
                    M == 8 ? bf16 : none, M == 64, false, 1.0f, 0.0f, M == 1});
    return v;
}

INSTANTIATE_TEST_SUITE_P(NativeBrgemmInt8, NativeBrgemmInt8E2EVsCk,
        ::testing::ValuesIn(e2e_ck_cases()), e2e_name);

// Calls the custom-kernel route does not serve, against the reference
// algorithm (one source zero point per row runs row by row there).  The
// static s8 source with a zero point per row is not a call matmul_direct
// passes on: it unit-tests the zero-point term of an s8 source, which a
// per-tensor zero point takes through the same code.
class NativeBrgemmInt8E2EVsRef : public NativeBrgemmInt8E2E {};

TEST_P(NativeBrgemmInt8E2EVsRef, MatchesReference) {
    const E2ECase c = GetParam();
    const uint32_t seed = static_cast<uint32_t>(c.M * 5 + c.K * 11 + c.N);
    E2EProblem pb = make_e2e(c, seed);
    auto dsts = e2e_dsts(c, seed, 2);
    ASSERT_EQ(nbt::run_reference(pb.call, pb.wei.get(), dsts[0].get()),
            status_t::success);
    ASSERT_EQ(run_native(pb, dsts[1].get()), status_t::success);
    // An integer destination takes the reference's roundings.
    if (c.dst_dt == data_type_t::u8 || c.dst_dt == data_type_t::s8
            || c.dst_dt == data_type_t::s32) {
        EXPECT_TRUE(nbt::within_steps(
                dsts[1].get(), c.N, dsts[0].get(), c.N, c.dst_dt, c.M, c.N, 0));
        return;
    }
    // The reference multiplies the scales one at a time; a transposed bf16
    // source may also round a quantized value the other way.
    float max_abs = 0.0f;
    for (size_t i = 0; i < static_cast<size_t>(c.M) * c.N; ++i)
        max_abs = std::max(
                max_abs, std::fabs(nbt::load_f32(dsts[0].get(), c.dst_dt, i)));
    const bool dyn_trans = c.transA
            && (c.src == E2ESrc::dyn_sym || c.src == E2ESrc::dyn_asym);
    const float rtol = c.dst_dt == data_type_t::f32 ? 2e-6f : 1.0f / 128;
    const float abs_bound = (dyn_trans ? 2e-2f : 1e-5f) * max_abs;
    EXPECT_TRUE(nbt::within_band(dsts[1].get(), c.N, dsts[0].get(), c.N,
            c.dst_dt, c.M, c.N, rtol, abs_bound));
}

std::vector<E2ECase> e2e_ref_cases() {
    const auto bf16 = data_type_t::bf16, f32 = data_type_t::f32,
               u8 = data_type_t::u8, s8 = data_type_t::s8,
               s32 = data_type_t::s32, none = data_type_t::none;
    std::vector<E2ECase> v;
    // Integer destinations with a bias, unit alpha and zero beta.
    for (data_type_t dst : {u8, s8, s32})
        for (data_type_t bias : {f32, bf16})
            v.push_back({508, 320, 8, E2ESrc::static_u8, dst, bias,
                    bias == bf16, true, 1.0f, 0.0f, dst != s32});
    // Prompt sizes, K not whole quads, transposed sources.
    for (int M : {65, 200})
        v.push_back({M, 256, 1024, E2ESrc::dyn_sym, bf16, f32, M == 200});
    for (int K : {63, 101})
        for (E2ESrc s :
                {E2ESrc::dyn_sym, E2ESrc::static_u8, E2ESrc::static_s8_row_zp})
            v.push_back({7, 96, K, s, f32, bf16, false});
    for (E2ESrc s : {E2ESrc::dyn_sym, E2ESrc::dyn_asym, E2ESrc::static_u8,
                 E2ESrc::static_s8_row_zp})
        v.push_back({13, 128, 200, s, f32, none, s == E2ESrc::static_u8, true});
    // Per-row zero points, which the route's fallback reads as one.
    for (int M : {4, 64})
        v.push_back({M, 256, 512, E2ESrc::dyn_asym, bf16, f32, false});
    v.push_back({16, 64, 300, E2ESrc::static_s8_row_zp, bf16, none, true});
    // Requantized destinations (a per-token source scale cannot have one).
    for (data_type_t dst : {u8, s8})
        for (bool packed : {false, true})
            v.push_back({9, 128, 256, E2ESrc::static_u8, dst, f32, packed,
                    false, 1.0f, 0.0f, true});
    // alpha / beta in the reference algorithm's order.
    v.push_back({5, 64, 128, E2ESrc::static_u8, f32, none, false, false, 0.5f,
            0.0f});
    v.push_back({5, 64, 128, E2ESrc::static_s8_row_zp, f32, f32, false, false,
            2.0f, 0.75f});
    v.push_back(
            {8, 96, 64, E2ESrc::dyn_sym, bf16, bf16, true, false, 1.0f, -1.0f});
    v.push_back({3, 64, 32, E2ESrc::static_u8, u8, none, false, false, 1.5f,
            0.25f});
    // Rows past the first M block, per-row zero points of a transposed
    // source, an s8 destination read back by beta.
    v.push_back({200, 128, 256, E2ESrc::static_s8_row_zp, f32, f32, true, false,
            0.75f, 1.25f});
    v.push_back({33, 96, 100, E2ESrc::dyn_asym, bf16, bf16, false, true, 1.0f,
            0.5f});
    v.push_back(
            {6, 64, 64, E2ESrc::static_u8, s8, f32, true, false, 0.5f, 2.0f});
    // alpha / beta after the last of several K passes.
    v.push_back({40, 64, 40000, E2ESrc::static_s8_row_zp, f32, f32, false,
            false, 0.75f, 1.25f});
    // f32 sources quantized dynamically (the reference takes them with an
    // f32 destination only).
    for (E2ESrc s : {E2ESrc::dyn_sym, E2ESrc::dyn_asym}) {
        E2ECase f {33, 96, 100, s, f32, bf16, s == E2ESrc::dyn_sym,
                s == E2ESrc::dyn_asym};
        f.f32_src = true;
        v.push_back(f);
    }
    // Weights of N not a multiple of 32, whose last column tile reads and
    // stores only the columns below N: integer destinations with the
    // reference's roundings, the dequantization's bias, alpha / beta in the
    // reference order, and dynamically quantized sources.
    for (int N : {1, 17, 40, 100}) {
        for (data_type_t dst : {u8, s8, s32})
            v.push_back({7, N, 100, E2ESrc::static_u8, dst, f32, N % 2 == 0,
                    false, 1.0f, 0.0f, dst != s32});
        v.push_back({5, N, 64, E2ESrc::static_s8_row_zp, s8, f32, true, false,
                0.5f, 1.5f});
        v.push_back({33, N, 200, E2ESrc::dyn_sym, f32, bf16, false});
        v.push_back({1, N, 64, E2ESrc::dyn_asym, bf16, f32, true});
        v.push_back({13, N, 128, E2ESrc::static_u8, f32, none, false, true,
                0.75f, 1.25f});
    }
    return v;
}

// Gated INT8 calls with per-tensor source and per-channel weight scales
// through matmul_direct under ALGO 10, as it defines them: alpha, beta (over
// the dst's N pre-activation columns), the bias and the post-ops act on the
// pre-activation in the reference algorithm's order, then the activation
// and the store, requantized into a u8 / s8 dst.  The native BRGEMM serves
// them, with the bits of the call on its own; against the reference
// algorithm's pre-activation and the activations of group_matmul_moe_act.cpp
// within their band (one step for an integer dst).
TEST(NativeBrgemmInt8Gated, PerChannelMatchesTheReference) {
    NATIVE_BRGEMM_INT8_SKIP_IF_NO_VNNI();
    mm::clear_matmul_weight_caches();
    const auto bf16 = data_type_t::bf16, f32 = data_type_t::f32,
               u8 = data_type_t::u8, s8 = data_type_t::s8,
               none = data_type_t::none;
    const std::vector<E2ECase> cases = {{16, 256, 512, E2ESrc::dyn_sym, bf16,
                                                f32, false, false, 1.0f, 0.5f},
            {7, 128, 256, E2ESrc::static_u8, u8, bf16, false, false, 1.0f, 0.0f,
                    true},
            {9, 128, 200, E2ESrc::dyn_sym, f32, none, true, false, 0.5f, -1.0f},
            {5, 128, 128, E2ESrc::static_u8, s8, f32, false, false, 1.0f, 1.0f},
            {33, 256, 300, E2ESrc::static_u8, bf16, f32, false, true, 2.0f,
                    0.25f, true}};
    uint32_t seed = 4100;
    for (mm::matmul_gated_act_t act : {mm::matmul_gated_act_t::silu_and_mul,
                 mm::matmul_gated_act_t::gelu_and_mul,
                 mm::matmul_gated_act_t::swiglu_oai_mul})
        for (const E2ECase &c : cases) {
            SCOPED_TRACE(::testing::Message()
                    << "act " << static_cast<int>(act) << " M " << c.M
                    << " dst " << dst_name(c.dst_dt) << " beta " << c.beta);
            E2EProblem pb = make_e2e(c, seed++);
            nbt::call_t &call = pb.call;
            call.gated = act;
            call.ldc = c.N;
            const int I = c.N / 2;
            auto dsts = e2e_dsts(c, seed, 2);

            // The call through matmul_direct, then on its own.
            {
                const nbt::algo_pin_t pin(matmul_algo_t::native_brgemm);
                nbt::call_t routed = call;
                routed.dst = dsts[0].get();
                ASSERT_EQ(nbt::run_matmul(routed), status_t::success);
                EXPECT_GE(nbt::native_layouts(pb.wei.get()), 1u)
                        << "not served by the native BRGEMM";
            }
            {
                nat::GemmDescriptor d;
                d.M = c.M;
                d.N = c.N;
                d.K = c.K;
                d.lda = call.lda_or_tight();
                d.ldb = call.ldb_or_tight();
                d.ldc = c.N;
                d.transA = call.transA;
                d.transB = call.transB;
                d.alpha = c.alpha;
                d.beta = c.beta;
                d.src_dt = call.params.dtypes.src;
                d.wei_dt = data_type_t::s8;
                d.dst_dt = c.dst_dt;
                d.bias_dt = c.bias_dt;
                d.bias = call.bias;
                d.is_weights_const = true;
                d.num_threads = call.params.num_threads;
                d.gated_act = act;
                nat::native_weight_ref_t ref;
                ref.pack_nr = 32;
                ref.weight = call.wei;
                ref.weight_cache_mode = 0;
                ref.interleave_split_halves
                        = act != mm::matmul_gated_act_t::swiglu_oai_mul;
                ASSERT_EQ(nat::native_brgemm_execute(d, ref, call.src,
                                  dsts[1].get(), call.bias, call.params),
                        status_t::success);
            }
            EXPECT_TRUE(nbt::within_steps(dsts[0].get(), c.N, dsts[1].get(),
                    c.N, c.dst_dt, c.M, c.N, 0))
                    << "vs the call on its own";

            // The reference algorithm's pre-activation, f32 [M, N] holding
            // the dst's values for beta, without the dst scale and zero
            // point.
            const size_t elems = static_cast<size_t>(c.M) * c.N;
            const nbt::buffer_t init = nbt::random_buffer(
                    c.dst_dt, elems, seed + 9, -20.0f, 20.0f);
            nbt::buffer_t pre = nbt::make_buffer(elems * 4);
            for (size_t i = 0; i < elems; ++i)
                nbt::store_f32(pre.get(), f32, i,
                        nbt::load_f32(init.get(), c.dst_dt, i));
            nbt::call_t r = call;
            r.gated = mm::matmul_gated_act_t::none;
            r.params.dtypes.dst = f32;
            r.params.quant_params.dst_scale = {};
            r.params.quant_params.dst_zp = {};
            ASSERT_EQ(nbt::run_reference(r, pb.wei.get(), pre.get()),
                    status_t::success);
            std::vector<float> out(static_cast<size_t>(c.M) * I);
            nbt::gated_act_ref(pre.as<float>(), c.M, c.N, act, out.data());
            const auto &qp = call.params.quant_params;
            std::vector<float> want(static_cast<size_t>(c.M) * I);
            for (size_t i = 0; i < want.size(); ++i) {
                float v = out[i];
                if (qp.dst_scale.buff != nullptr) {
                    v = v
                            * nbt::load_f32(
                                    qp.dst_scale.buff, qp.dst_scale.dt, 0);
                    v = v + nbt::load_f32(qp.dst_zp.buff, qp.dst_zp.dt, 0);
                }
                want[i] = v;
            }
            size_t bad = 0;
            for (int m = 0; m < c.M; ++m)
                for (int i = 0; i < I; ++i) {
                    const float g = nbt::load_f32(dsts[0].get(), c.dst_dt,
                            static_cast<size_t>(m) * c.N + i);
                    const float w = want[static_cast<size_t>(m) * I + i];
                    float tol = 3e-4f + 3e-4f * std::fabs(w);
                    if (c.dst_dt == bf16) tol += std::fabs(w) / 256;
                    if (c.dst_dt == u8 || c.dst_dt == s8) {
                        const float lo = c.dst_dt == u8 ? 0.0f : -128.0f;
                        const float hi = c.dst_dt == u8 ? 255.0f : 127.0f;
                        if (std::fabs(g
                                    - std::nearbyint(
                                            std::min(std::max(w, lo), hi)))
                                        > 1.0f
                                && bad++ == 0)
                            ADD_FAILURE() << "(" << m << ", " << i << ") got "
                                          << g << " want " << w;
                        continue;
                    }
                    if (!(std::fabs(g - w) <= tol) && bad++ == 0)
                        ADD_FAILURE() << "(" << m << ", " << i << ") got " << g
                                      << " want " << w;
                }
            EXPECT_EQ(bad, 0u);
            // The columns past N / 2 keep the dst's values.
            for (int m = 0; m < c.M; ++m)
                for (int n = I; n < c.N; ++n) {
                    const size_t at = static_cast<size_t>(m) * c.N + n;
                    ASSERT_EQ(nbt::load_f32(dsts[0].get(), c.dst_dt, at),
                            nbt::load_f32(init.get(), c.dst_dt, at))
                            << "(" << m << ", " << n << ")";
                }
            mm::clear_matmul_weight_caches();
        }
}

INSTANTIATE_TEST_SUITE_P(NativeBrgemmInt8, NativeBrgemmInt8E2EVsRef,
        ::testing::ValuesIn(e2e_ref_cases()), e2e_name);

// A weight scale whose product with the integer sum rounds down, and a bias
// that brings the rounded product to exactly a half: one rounding per
// operation lands on the half, which rounds to even, where one rounding of
// the product and the bias together lands above it.
TEST(NativeBrgemmInt8E2EOrder, IntegerDestinationTakesReferenceRoundings) {
    NATIVE_BRGEMM_INT8_SKIP_IF_NO_VNNI();
    mm::clear_matmul_weight_caches();
    const int M = 4, N = 32, K = 4;
    const float wei_scale = 1.5f + std::ldexp(3.0f, -23);
    const float bias = -(4.0f + std::ldexp(1.0f, -20));
    for (const data_type_t dst_dt :
            {data_type_t::u8, data_type_t::s8, data_type_t::s32}) {
        E2EProblem pb;
        pb.src = nbt::make_buffer(static_cast<size_t>(M) * K);
        pb.wei = nbt::make_buffer(static_cast<size_t>(N) * K);
        pb.bias = nbt::make_buffer(N * sizeof(float));
        pb.wei_scale = nbt::make_buffer(N * sizeof(float));
        pb.src_scale = nbt::make_buffer(sizeof(float));
        auto *a = static_cast<uint8_t *>(pb.src.get());
        auto *w = static_cast<int8_t *>(pb.wei.get());
        std::memset(a, 0, static_cast<size_t>(M) * K);
        std::memset(w, 0, static_cast<size_t>(N) * K);
        for (int m = 0; m < M; ++m)
            a[static_cast<size_t>(m) * K] = 3;
        for (int n = 0; n < N; ++n) {
            w[static_cast<size_t>(n) * K] = 1;
            static_cast<float *>(pb.bias.get())[n] = bias;
            static_cast<float *>(pb.wei_scale.get())[n] = wei_scale;
        }
        *static_cast<float *>(pb.src_scale.get()) = 1.0f;
        auto &call = pb.call;
        call.M = M;
        call.N = N;
        call.K = K;
        call.transB = true;
        call.src = pb.src.get();
        call.wei = pb.wei.get();
        call.bias = pb.bias.get();
        auto &p = call.params;
        p.dtypes.src = data_type_t::u8;
        p.dtypes.wei = data_type_t::s8;
        p.dtypes.dst = dst_dt;
        p.dtypes.bias = data_type_t::f32;
        p.num_threads = 1;
        auto &qp = p.quant_params;
        qp.src_scale.buff = pb.src_scale.get();
        qp.src_scale.dt = data_type_t::f32;
        qp.src_scale.dims = {1, 1};
        qp.wei_scale.buff = pb.wei_scale.get();
        qp.wei_scale.dt = data_type_t::f32;
        qp.wei_scale.dims = {1, N};
        pb.weight_cache_mode = 0;

        const size_t bytes = static_cast<size_t>(M) * N * nbt::dt_bytes(dst_dt);
        nbt::buffer_t ref = nbt::make_buffer(bytes);
        nbt::buffer_t got = nbt::make_buffer(bytes);
        std::memset(ref.get(), 0, bytes);
        std::memset(got.get(), 0, bytes);
        ASSERT_EQ(nbt::run_reference(call, pb.wei.get(), ref.get()),
                status_t::success);
        ASSERT_EQ(run_native(pb, got.get()), status_t::success);
        EXPECT_EQ(nbt::load_f32(ref.get(), dst_dt, 0), 0.0f)
                << dst_name(dst_dt);
        EXPECT_TRUE(
                nbt::within_steps(got.get(), N, ref.get(), N, dst_dt, M, N, 0))
                << dst_name(dst_dt);
    }
    mm::clear_matmul_weight_caches();
}

// ── Status contract ───────────────────────────────────────────────────────

TEST(NativeBrgemmInt8Status, UnservedCallsReportUnimplemented) {
    NATIVE_BRGEMM_INT8_SKIP_IF_NO_VNNI();
    const int M = 4, K = 64, N = 64;
    std::vector<uint8_t> src(static_cast<size_t>(M) * K, 1);
    std::vector<float> scale(2 * M, 0.1f);
    int32_t wei_zp = 3;
    const auto good = [&]() {
        matmul_params p;
        p.dtypes.src = data_type_t::u8;
        p.dtypes.wei = data_type_t::s8;
        p.dtypes.dst = data_type_t::f32;
        return p;
    };
    const auto desc = make_desc(
            M, N, K, K, false, data_type_t::u8, data_type_t::f32, 1);
    nat::native_int8_call_t call;
    ASSERT_EQ(prepare(desc, good(), src.data(), call), status_t::success);

    auto p = good();
    p.quant_params.wei_zp.buff = &wei_zp;
    p.quant_params.wei_zp.dt = data_type_t::s32;
    p.quant_params.wei_zp.dims = {1, 1};
    EXPECT_EQ(prepare(desc, p, src.data(), call), status_t::unimplemented);

    p = good();
    p.quant_params.src_scale.buff = scale.data();
    p.quant_params.src_scale.dt = data_type_t::f32;
    p.quant_params.src_scale.dims = {M, 2};
    EXPECT_EQ(prepare(desc, p, src.data(), call), status_t::unimplemented);

    p = good();
    p.quant_params.wei_scale.buff = scale.data();
    p.quant_params.wei_scale.dt = data_type_t::f32;
    p.quant_params.wei_scale.dims = {2, N};
    EXPECT_EQ(prepare(desc, p, src.data(), call), status_t::success)
            << "a u8 source with per-group weight scales";
    p.quant_params.src_scale.buff = scale.data();
    p.quant_params.src_scale.dt = data_type_t::f32;
    p.quant_params.src_scale.dims = {M, 1};
    EXPECT_EQ(prepare(desc, p, src.data(), call), status_t::unimplemented)
            << "a u8 source per token with per-group weight scales";

    p = good();
    p.dtypes.wei = data_type_t::s4;
    EXPECT_EQ(prepare(desc, p, src.data(), call), status_t::unimplemented);

    p = good();
    p.dtypes.src = data_type_t::bf16;
    EXPECT_EQ(prepare(desc, p, src.data(), call), status_t::unimplemented)
            << "a bf16 source without dynamic quantization";

    auto deep = desc;
    deep.K = nat::kMaxSafeInt8AccumK + 1;
    static_assert(nat::kMaxSafeInt8AccumK == ck::kMaxSafeInt8AccumK,
            "native and custom-kernel INT8 K bounds differ");
    deep.lda = deep.K;
    EXPECT_EQ(prepare(deep, good(), src.data(), call), status_t::unimplemented);

    auto gated = desc;
    gated.gated_act = mm::matmul_gated_act_t::silu_and_mul;
    auto with_beta = desc;
    with_beta.beta = 0.5f;
    auto gated_with_beta = gated;
    gated_with_beta.beta = 0.5f;
    EXPECT_EQ(prepare(gated_with_beta, good(), src.data(), call),
            status_t::success)
            << "a gated activation with beta != 0";
    EXPECT_EQ(nat::check_int8_call(gated_with_beta, good(),
                      nat::decide_int8_call(gated_with_beta, good(), nullptr)),
            status_t::success);
    EXPECT_EQ(prepare(gated, good(), src.data(), call), status_t::success);
    EXPECT_EQ(prepare(with_beta, good(), src.data(), call), status_t::success);

    auto short_lda = desc;
    short_lda.lda = K - 1;
    EXPECT_EQ(prepare(short_lda, good(), src.data(), call), status_t::failure);
    EXPECT_EQ(prepare(desc, good(), nullptr, call), status_t::failure);
}

// decide_int8_call against the rules it encodes, for both schemes, alpha and
// beta, every destination and bias type; and what its readers make of it:
// the planner's epilogue class and prepare_int8_call.
TEST(NativeBrgemmInt8Decision, MatchesTheRules) {
    using scheme_t = nat::native_int8_scheme_t;
    using order_t = nat::native_int8_order_t;
    const auto f32 = data_type_t::f32, bf16 = data_type_t::bf16,
               f16 = data_type_t::f16, u8 = data_type_t::u8,
               s8 = data_type_t::s8, s32 = data_type_t::s32,
               none = data_type_t::none;
    constexpr int M = 4, N = 64, K = 128, G = 4;
    const std::vector<float> wei_scales(static_cast<size_t>(G) * N, 0.01f);
    const std::vector<float> src_scales(static_cast<size_t>(M) * G, 0.02f);
    const std::vector<int8_t> src(static_cast<size_t>(M) * K, 1);
    const std::vector<float> bias(N, 0.5f);
    enum class wei_t { none, tensor, channel, group };
    for (const wei_t wei :
            {wei_t::none, wei_t::tensor, wei_t::channel, wei_t::group})
        for (const float alpha : {1.0f, 0.5f})
            for (const float beta : {0.0f, 1.0f})
                for (const data_type_t dst : {f32, bf16, f16, u8, s8, s32})
                    for (const data_type_t bias_dt :
                            {none, f32, bf16, f16, s8, s32, u8}) {
                        SCOPED_TRACE(::testing::Message()
                                << "wei " << static_cast<int>(wei) << " alpha "
                                << alpha << " beta " << beta << " dst "
                                << static_cast<int>(dst) << " bias "
                                << static_cast<int>(bias_dt));
                        const bool group = wei == wei_t::group;
                        matmul_params p;
                        p.dtypes.src = s8;
                        p.dtypes.wei = s8;
                        p.dtypes.dst = dst;
                        p.dtypes.bias = bias_dt;
                        auto &qp = p.quant_params;
                        if (wei != wei_t::none) {
                            qp.wei_scale.buff = wei_scales.data();
                            qp.wei_scale.dt = f32;
                            qp.wei_scale.dims = wei == wei_t::tensor
                                    ? std::vector<int64_t> {1}
                                    : std::vector<int64_t> {group ? G : 1, N};
                        }
                        qp.src_scale.buff = src_scales.data();
                        qp.src_scale.dt = f32;
                        qp.src_scale.dims = group ? std::vector<int64_t> {M, G}
                                                  : std::vector<int64_t> {1, 1};
                        auto desc = make_desc(M, N, K, K, false, s8, dst, 1);
                        desc.alpha = alpha;
                        desc.beta = beta;
                        desc.bias_dt = bias_dt;
                        const void *b = bias_dt == none ? nullptr : bias.data();

                        const bool integer
                                = dst == u8 || dst == s8 || dst == s32;
                        const bool reference = group || alpha != 1.0f
                                || beta != 0.0f || integer;
                        const bool fused = !group && !reference;
                        const bool bias_moves = fused && b != nullptr
                                && (bias_dt == f32 || bias_dt == bf16
                                        || bias_dt == f16);
                        const nat::native_int8_decision_t d
                                = nat::decide_int8_call(desc, p, b);
                        EXPECT_EQ(d.scheme,
                                group ? scheme_t::per_group
                                      : scheme_t::per_channel);
                        EXPECT_EQ(d.groups, group ? G : 0);
                        EXPECT_EQ(d.order,
                                reference ? order_t::reference
                                          : order_t::fused);
                        EXPECT_EQ(d.bias_in_dequant, bias_moves);
                        const bool plain = fused
                                || (group && alpha == 1.0f && beta == 0.0f);
                        EXPECT_EQ(d.direct_store,
                                plain && (dst == f32 || dst == bf16));

                        using class_t = nat::native_epilogue_class_t;
                        const class_t epilogue = alpha != 1.0f || beta != 0.0f
                                ? class_t::postops
                                : (b != nullptr && !bias_moves
                                                  ? class_t::bias
                                                  : class_t::store);
                        EXPECT_EQ(nat::make_native_problem(desc, p, b, 32,
                                          nat::decide_int8_call(desc, p, b))
                                          .epilogue,
                                epilogue);

                        nat::native_int8_call_t call;
                        const status_t st = prepare(desc, p, src.data(), call);
                        if (group && dst == f16) {
                            EXPECT_EQ(st, status_t::unimplemented);
                            continue;
                        }
                        ASSERT_EQ(st, status_t::success);
                        EXPECT_EQ(call.scheme, d.scheme);
                        EXPECT_EQ(call.order, d.order);
                        EXPECT_EQ(call.groups, d.groups);
                    }
}

} // namespace
