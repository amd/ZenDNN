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

/// @file test_e2m1_group_matmul.cpp
/// @brief End-to-end `group_matmul_direct` and `matmul_direct` with MXFP4
/// weights (f4_e2m1 codes, f8_e8m0 scales): the custom kernel against an f32
/// dequantized reference over the contract's source, source-scale, bias,
/// layout and destination choices, the fused-MoE down projection, the
/// prepack warm of idle experts, the prompt tile under the u8-source switch,
/// and every way a call the custom kernel cannot serve must fail rather than
/// reach another backend.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "common/float16.hpp"
#include "group_matmul_test_helpers.hpp"
#include "gtest_utils.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/dispatch.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/pack.hpp"
#include "lowoha_operators/matmul/group_matmul/executors/n_tile/n_tile_knobs.hpp"
#include "lowoha_operators/matmul/routed_moe/routed_moe.hpp"
#include "lowoha_operators/reorder/lowoha_reorder.hpp"
#include "lowoha_operators/reorder/prepack/lowoha_prepack.hpp"
#include "moe_test_utils.hpp"

namespace {

using namespace zendnnl::lowoha::matmul;
using zendnnl::common::bfloat16_t;
using zendnnl::common::float16_t;
namespace ck = zendnnl::lowoha::matmul::custom_kernel;

constexpr int kGroup = kMxfp4BlockSize;
constexpr uint16_t kCanary = 0x7FC1;
constexpr float kCanaryF32 = -1.0e30f;

// Written out rather than taken from the library, so a wrong decode table or
// a broken scale-halving contract shows up here.
constexpr float kE2M1Value[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f,
        6.0f, -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

float bf16_to_f32(uint16_t bits) {
    return static_cast<float>(bfloat16_t::from_bits(bits));
}

uint16_t f32_to_bf16(float v) {
    return static_cast<uint16_t>(bfloat16_t::f32_to_bf16_val(v));
}

bool all_canary(const std::vector<uint16_t> &v) {
    return std::all_of(
            v.begin(), v.end(), [](uint16_t x) { return x == kCanary; });
}

/// One expert's MXFP4 weight: f4_e2m1 codes two per byte, the even linear
/// index in the low nibble ([N, K/2] for transB, else [K, N/2]), its f8_e8m0
/// `{K/32, N}` scales exactly as a checkpoint stores them, and the
/// dequantized [N, K] weight the reference multiplies by.
struct Mxfp4Weight {
    std::vector<uint8_t> codes;
    std::vector<uint8_t> e8m0;
    std::vector<float> dequant;
};

Mxfp4Weight make_weight(std::mt19937 &rng, int N, int K, bool transB = true) {
    const int G = K / kGroup;
    Mxfp4Weight w;
    w.codes.assign(static_cast<size_t>(N) * K / 2, 0);
    w.e8m0.resize(static_cast<size_t>(G) * N);
    w.dequant.resize(static_cast<size_t>(N) * K);
    std::uniform_int_distribution<int> nib(0, 15);
    std::uniform_int_distribution<int> exp2(-3, 2);
    for (auto &e : w.e8m0) {
        e = static_cast<uint8_t>(127 + exp2(rng));
    }
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            const int code = nib(rng);
            const size_t idx = transB ? static_cast<size_t>(n) * K + k
                                      : static_cast<size_t>(k) * N + n;
            w.codes[idx / 2]
                    |= static_cast<uint8_t>(idx % 2 == 0 ? code : code << 4);
            const int e = w.e8m0[static_cast<size_t>(k / kGroup) * N + n];
            w.dequant[static_cast<size_t>(n) * K + k]
                    = kE2M1Value[code] * std::ldexp(1.0f, e - 127);
        }
    }
    return w;
}

/// out[m, n] = sum_k src[m, k] * dequant[n, k] + bias[n], in f32.
std::vector<float> reference_matmul(const std::vector<float> &src, int M, int K,
        const std::vector<float> &dequant, int N,
        const std::vector<float> &bias = {}) {
    std::vector<float> out(static_cast<size_t>(M) * N, 0.0f);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double acc = bias.empty() ? 0.0 : bias[n];
            for (int k = 0; k < K; ++k) {
                acc += static_cast<double>(src[static_cast<size_t>(m) * K + k])
                        * dequant[static_cast<size_t>(n) * K + k];
            }
            out[static_cast<size_t>(m) * N + n] = static_cast<float>(acc);
        }
    }
    return out;
}

/// Frobenius-norm relative error; the per-token int8 activation quant keeps
/// it well under 2% at these sizes.
double relative_error(const std::function<double(size_t)> &got,
        const std::vector<float> &want) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < want.size(); ++i) {
        const double d = got(i) - want[i];
        num += d * d;
        den += static_cast<double>(want[i]) * want[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}
double relative_error(
        const std::vector<uint16_t> &got, const std::vector<float> &want) {
    return relative_error([&](size_t i) { return bf16_to_f32(got[i]); }, want);
}
double relative_error(
        const std::vector<float> &got, const std::vector<float> &want) {
    return relative_error([&](size_t i) { return got[i]; }, want);
}

/// The MXFP4 contract: f4_e2m1 weights and f8_e8m0 `{K/32, N}` scales, with
/// bf16 activations quantized per token.
matmul_params mxfp4_params(const uint8_t *e8m0, int K, int N,
        std::vector<float> &src_scale, int M) {
    matmul_params p;
    p.dtypes.src = data_type_t::bf16;
    p.dtypes.wei = data_type_t::f4_e2m1;
    p.dtypes.dst = data_type_t::bf16;
    p.dtypes.compute = data_type_t::s8;
    p.dynamic_quant = true;
    src_scale.assign(static_cast<size_t>(std::max(M, 1)), 0.0f);
    p.quant_params.src_scale.buff = src_scale.data();
    p.quant_params.src_scale.dt = data_type_t::f32;
    p.quant_params.src_scale.dims = {std::max(M, 1), 1};
    p.quant_params.wei_scale.buff = e8m0;
    p.quant_params.wei_scale.dt = data_type_t::f8_e8m0;
    p.quant_params.wei_scale.dims = {K / kGroup, N};
    return p;
}

/// The contract's choices besides the weight: the source entry (bf16 or f32
/// with dynamic quantization, or pre-quantized s8), its scale dtype and
/// layout (per token, or per row and block), the bias dtype, the weight
/// layout and the destination dtype.
struct Mxfp4Variant {
    bool s8_src = false;
    data_type_t src_scale_dt = data_type_t::f32;
    data_type_t bias_dt = data_type_t::none;
    bool transB = true;
    bool f32_src = false;
    bool per_block = false;
    bool f32_dst = false;

    std::string name() const {
        return std::string(s8_src        ? "s8"
                               : f32_src ? "f32"
                                         : "bf16")
                + " src, " + dtype_info(src_scale_dt)
                + (per_block ? " per-block" : "") + " src scale, bias "
                + dtype_info(bias_dt) + (transB ? ", transB" : ", K-major")
                + (f32_dst ? ", f32 dst" : "");
    }
};

std::vector<Mxfp4Variant> contract_variants() {
    std::vector<Mxfp4Variant> out;
    for (const bool s8 : {false, true})
        for (const data_type_t sdt : {data_type_t::f32, data_type_t::bf16})
            for (const data_type_t bdt : {data_type_t::none, data_type_t::f32,
                         data_type_t::bf16, data_type_t::f16})
                for (const bool transB : {true, false})
                    out.push_back({s8, sdt, bdt, transB});
    return out;
}

/// A grouped MXFP4 call over experts with `rows[e]` tokens and `Ns[e]`
/// output columns each.
struct Mxfp4Call {
    int K;
    std::vector<int> rows, Ns;
    Mxfp4Variant v;
    std::vector<Mxfp4Weight> weights;
    std::vector<std::vector<uint16_t>> src_bf16, dst;
    std::vector<std::vector<float>> dst_f32;
    std::vector<std::vector<int8_t>> src_s8;
    // What the reference multiplies: the bf16 source, or the s8 source
    // dequantized with the scale the kernel reads.
    std::vector<std::vector<float>> src_f32;
    std::vector<std::vector<float>> src_scale;
    std::vector<std::vector<uint16_t>> src_scale_bf16;
    std::vector<std::vector<uint8_t>> bias;
    std::vector<std::vector<float>> bias_f32;

    Mxfp4Call(const std::vector<int> &rows_, int K_, int N_, uint32_t seed,
            Mxfp4Variant v_ = {})
        : Mxfp4Call(rows_, K_, std::vector<int>(rows_.size(), N_), seed, v_) {}

    Mxfp4Call(std::vector<int> rows_, int K_, std::vector<int> Ns_,
            uint32_t seed, Mxfp4Variant v_ = {})
        : K(K_), rows(std::move(rows_)), Ns(std::move(Ns_)), v(v_) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> x(-1.0f, 1.0f);
        const size_t E = rows.size();
        weights.reserve(E);
        src_bf16.resize(E);
        src_s8.resize(E);
        src_f32.resize(E);
        dst.resize(E);
        dst_f32.resize(E);
        src_scale.resize(E);
        src_scale_bf16.resize(E);
        bias.resize(E);
        bias_f32.resize(E);
        for (size_t e = 0; e < E; ++e) {
            weights.push_back(make_weight(rng, Ns[e], K, v.transB));
            const int m_e = std::max(rows[e], 1);
            const size_t mk = static_cast<size_t>(m_e) * K;
            src_bf16[e].resize(mk);
            src_f32[e].resize(mk);
            for (size_t i = 0; i < mk; ++i) {
                src_bf16[e][i] = f32_to_bf16(x(rng));
                src_f32[e][i] = bf16_to_f32(src_bf16[e][i]);
            }
            const size_t n_scales
                    = static_cast<size_t>(m_e) * (v.per_block ? K / kGroup : 1);
            src_scale[e].assign(n_scales, 0.0f);
            src_scale_bf16[e].assign(n_scales, 0);
            if (v.s8_src) quantize_source(e, m_e);
            dst[e].assign(static_cast<size_t>(m_e) * Ns[e], 0);
            if (v.f32_dst)
                dst_f32[e].assign(static_cast<size_t>(m_e) * Ns[e], 0.0f);
            if (v.bias_dt != data_type_t::none) make_bias(e, rng);
        }
    }

    // Symmetric s8 per token or per block, with the scale rounded to the
    // dtype the kernel reads, so the reference sees exactly the kernel's
    // source.
    void quantize_source(size_t e, int m_e) {
        src_s8[e].resize(src_f32[e].size());
        const int span = v.per_block ? kGroup : K;
        const int spans = K / span;
        for (int m = 0; m < m_e; ++m) {
            for (int g = 0; g < spans; ++g) {
                const size_t at = static_cast<size_t>(m) * K
                        + static_cast<size_t>(g) * span;
                float *seg = src_f32[e].data() + at;
                float amax = 0.0f;
                for (int k = 0; k < span; ++k)
                    amax = std::max(amax, std::fabs(seg[k]));
                float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
                const size_t si = static_cast<size_t>(m) * spans + g;
                if (v.src_scale_dt == data_type_t::bf16) {
                    src_scale_bf16[e][si] = f32_to_bf16(scale);
                    scale = bf16_to_f32(src_scale_bf16[e][si]);
                } else {
                    src_scale[e][si] = scale;
                }
                for (int k = 0; k < span; ++k) {
                    const float q = std::max(-127.0f,
                            std::min(127.0f, std::nearbyint(seg[k] / scale)));
                    src_s8[e][at + k] = static_cast<int8_t>(q);
                    seg[k] = q * scale;
                }
            }
        }
    }

    void make_bias(size_t e, std::mt19937 &rng) {
        std::uniform_real_distribution<float> b(-0.5f, 0.5f);
        const size_t elem = v.bias_dt == data_type_t::f32 ? sizeof(float)
                                                          : sizeof(uint16_t);
        bias[e].resize(static_cast<size_t>(Ns[e]) * elem);
        bias_f32[e].resize(static_cast<size_t>(Ns[e]));
        for (int n = 0; n < Ns[e]; ++n) {
            float val = b(rng);
            uint8_t *slot = bias[e].data() + static_cast<size_t>(n) * elem;
            if (v.bias_dt == data_type_t::bf16) {
                const uint16_t bits = f32_to_bf16(val);
                std::memcpy(slot, &bits, elem);
                val = bf16_to_f32(bits);
            } else if (v.bias_dt == data_type_t::f16) {
                const float16_t h(val);
                const uint16_t bits = h.raw();
                std::memcpy(slot, &bits, elem);
                val = static_cast<float>(h);
            } else {
                std::memcpy(slot, &val, elem);
            }
            bias_f32[e][n] = val;
        }
    }

    const void *src_of(size_t e) const {
        return v.s8_src     ? static_cast<const void *>(src_s8[e].data())
                : v.f32_src ? static_cast<const void *>(src_f32[e].data())
                            : static_cast<const void *>(src_bf16[e].data());
    }
    void *dst_of(size_t e) {
        return v.f32_dst ? static_cast<void *>(dst_f32[e].data())
                         : static_cast<void *>(dst[e].data());
    }
    const void *bias_of(size_t e) const {
        return v.bias_dt == data_type_t::none ? nullptr : bias[e].data();
    }
    int ldb_of(size_t e) const { return v.transB ? K : Ns[e]; }

    matmul_params expert_params(
            size_t e, matmul_algo_t inner = matmul_algo_t::none) {
        const int m_e = std::max(rows[e], 1);
        matmul_params p;
        p.dtypes.src = v.s8_src ? data_type_t::s8
                : v.f32_src     ? data_type_t::f32
                                : data_type_t::bf16;
        p.dtypes.wei = data_type_t::f4_e2m1;
        p.dtypes.dst = v.f32_dst ? data_type_t::f32 : data_type_t::bf16;
        p.dtypes.bias = v.bias_dt;
        p.dtypes.compute = data_type_t::s8;
        p.dynamic_quant = !v.s8_src;
        p.quant_params.src_scale.buff = v.src_scale_dt == data_type_t::bf16
                ? static_cast<const void *>(src_scale_bf16[e].data())
                : static_cast<const void *>(src_scale[e].data());
        p.quant_params.src_scale.dt = v.src_scale_dt;
        p.quant_params.src_scale.dims = {m_e, v.per_block ? K / kGroup : 1};
        p.quant_params.wei_scale.buff = weights[e].e8m0.data();
        p.quant_params.wei_scale.dt = data_type_t::f8_e8m0;
        p.quant_params.wei_scale.dims = {K / kGroup, Ns[e]};
        p.lowoha_algo = inner;
        return p;
    }

    std::vector<matmul_params> params(
            matmul_algo_t inner = matmul_algo_t::none) {
        std::vector<matmul_params> p;
        for (size_t e = 0; e < rows.size(); ++e)
            p.push_back(expert_params(e, inner));
        return p;
    }

    status_t run(const std::vector<matmul_params> &p) {
        const int E = static_cast<int>(rows.size());
        std::vector<const void *> src(E), wei(E), bias_p(E);
        std::vector<void *> out(E);
        std::vector<int> ldb(E);
        for (int e = 0; e < E; ++e) {
            src[e] = src_of(e);
            wei[e] = weights[e].codes.data();
            bias_p[e] = bias_of(e);
            out[e] = dst_of(e);
            ldb[e] = ldb_of(e);
        }
        return group_matmul_direct(std::vector<char>(E, 'r'),
                std::vector<bool>(E, false), std::vector<bool>(E, v.transB),
                rows, Ns, std::vector<int>(E, K), std::vector<float>(E, 1.0f),
                src, std::vector<int>(E, K), wei, ldb, bias_p,
                std::vector<float>(E, 0.0f), out, Ns,
                std::vector<bool>(E, true), p);
    }

    /// Runs the call; `inner` pins the inner kernel as a caller would.
    status_t run(matmul_algo_t inner = matmul_algo_t::none) {
        return run(params(inner));
    }

    /// Runs expert 0 as one `matmul_direct` call, the dense-linear entry.
    status_t run_direct(matmul_algo_t algo = matmul_algo_t::none,
            char layout = 'r', int batch = 1,
            const matmul_gated_act_params *gated_act = nullptr) {
        const matmul_params p = expert_params(0, algo);
        matmul_batch_params_t bp;
        bp.Batch_A = batch;
        bp.Batch_B = batch;
        return matmul_direct(layout, false, v.transB, rows[0], Ns[0], K, 1.0f,
                src_of(0), K, weights[0].codes.data(), ldb_of(0), bias_of(0),
                0.0f, dst_of(0), Ns[0], true, bp, p, gated_act);
    }

    void fill_canary() {
        for (auto &d : dst)
            std::fill(d.begin(), d.end(), kCanary);
        for (auto &d : dst_f32)
            std::fill(d.begin(), d.end(), kCanaryF32);
    }

    bool dst_untouched() const {
        return std::all_of(dst.begin(), dst.end(), all_canary)
                && std::all_of(dst_f32.begin(), dst_f32.end(),
                        [](const std::vector<float> &d) {
            return std::all_of(d.begin(), d.end(),
                    [](float x) { return x == kCanaryF32; });
        });
    }

    void expect_matches_reference(const std::string &label) const {
        for (size_t e = 0; e < rows.size(); ++e) {
            if (rows[e] <= 0) { continue; }
            const auto want = reference_matmul(src_f32[e], rows[e], K,
                    weights[e].dequant, Ns[e], bias_f32[e]);
            const double err = v.f32_dst ? relative_error(dst_f32[e], want)
                                         : relative_error(dst[e], want);
            EXPECT_LT(err, 2e-2)
                    << label << ": expert " << e << " (rows=" << rows[e] << ")";
        }
    }
};

/// A fused MoE call's hidden size K, intermediate size I (gate/up N = 2I,
/// down K = I) and down N.
struct Mxfp4FusedShape {
    int K = 256, I = 128, N2 = 128;
};

/// A fused MoE call: MXFP4 gate/up with silu_and_mul ([gate | up] halves)
/// and an MXFP4 down projection, `M[e]` tokens per expert.  With `active`
/// set, experts from `active` on are the prepack-extras tail.
struct Mxfp4FusedCall {
    using Shape = Mxfp4FusedShape;
    int K, N13, I, N2;
    Mxfp4Call op1;
    std::vector<Mxfp4Weight> w2;
    std::vector<std::vector<uint16_t>> out;
    std::vector<std::vector<float>> out_f32; // with an f32 destination
    std::vector<matmul_params> params;
    grp_matmul_gated_act_params act {};
    grp_matmul_fused_moe_params fused {};

    Mxfp4FusedCall(const std::vector<int> &M, uint32_t seed, int active = 0,
            Shape s = {}, Mxfp4Variant v = {})
        : K(s.K)
        , N13(2 * s.I)
        , I(s.I)
        , N2(s.N2)
        , op1(M, s.K, 2 * s.I, seed, v) {
        const size_t E = M.size();
        std::mt19937 rng(seed + 4);
        for (size_t e = 0; e < E; ++e) {
            w2.push_back(make_weight(rng, N2, I));
            out.emplace_back(
                    static_cast<size_t>(std::max(M[e], 1)) * N2, kCanary);
            if (v.f32_dst)
                out_f32.emplace_back(
                        static_cast<size_t>(std::max(M[e], 1)) * N2, 0.0f);
        }
        params = op1.params();
        if (active > 0) {
            for (auto &p : params) {
                p.active_matmul = static_cast<uint32_t>(active);
                p.total_matmul = static_cast<uint32_t>(E);
            }
        }
        act.act = grp_matmul_gated_act_t::silu_and_mul;
        fused.N_down.assign(E, N2);
        fused.ldb_down.assign(E, I);
        fused.bias_down.assign(E, nullptr);
        fused.down_weight.resize(E);
        fused.down_scale.resize(E);
        fused.dst_down.resize(E);
        fused.ldc_down.assign(E, N2);
        for (size_t e = 0; e < E; ++e) {
            fused.down_weight[e] = w2[e].codes.data();
            fused.down_scale[e].buff = w2[e].e8m0.data();
            fused.down_scale[e].dt = data_type_t::f8_e8m0;
            fused.down_scale[e].dims = {I / kGroup, N2};
            fused.dst_down[e] = v.f32_dst
                    ? static_cast<void *>(out_f32[e].data())
                    : static_cast<void *>(out[e].data());
        }
    }

    status_t run() {
        const int E = static_cast<int>(op1.rows.size());
        std::vector<const void *> src(E), wei(E), bias(E, nullptr);
        for (int e = 0; e < E; ++e) {
            src[e] = op1.src_of(e);
            wei[e] = op1.weights[e].codes.data();
            bias[e] = op1.bias_of(e);
        }
        return group_matmul_direct(std::vector<char>(E, 'r'),
                std::vector<bool>(E, false), std::vector<bool>(E, true),
                op1.rows, std::vector<int>(E, N13), std::vector<int>(E, K),
                std::vector<float>(E, 1.0f), src, std::vector<int>(E, K), wei,
                std::vector<int>(E, K), bias, std::vector<float>(E, 0.0f),
                std::vector<void *>(E, nullptr), std::vector<int>(E, 0),
                std::vector<bool>(E, true), params, nullptr, &act, &fused);
    }

    void expect_matches_reference() const {
        for (size_t e = 0; e < op1.rows.size(); ++e) {
            const int M = op1.rows[e];
            if (M <= 0) continue;
            const auto h13 = reference_matmul(op1.src_f32[e], M, K,
                    op1.weights[e].dequant, N13, op1.bias_f32[e]);
            std::vector<float> h(static_cast<size_t>(M) * I);
            for (int m = 0; m < M; ++m) {
                for (int i = 0; i < I; ++i) {
                    const float g = h13[static_cast<size_t>(m) * N13 + i];
                    const float u = h13[static_cast<size_t>(m) * N13 + I + i];
                    h[static_cast<size_t>(m) * I + i]
                            = g / (1.0f + std::exp(-g)) * u;
                }
            }
            const auto want = reference_matmul(h, M, I, w2[e].dequant, N2);
            // Two quantized GEMMs and a bf16 activation in between.
            EXPECT_LT(op1.v.f32_dst ? relative_error(out_f32[e], want)
                                    : relative_error(out[e], want),
                    5e-2)
                    << "expert " << e;
        }
    }

    bool outputs_untouched() const {
        return std::all_of(out.begin(), out.end(), all_canary);
    }
};

const char *last_execution_path() {
    return test_api::s_last_group_matmul_direct_execution_path.load(
            std::memory_order_relaxed);
}

void expect_custom_kernel_path(const std::string &label) {
    const char *path = last_execution_path();
    ASSERT_NE(path, nullptr) << label << ": grouped route not taken";
    EXPECT_NE(std::strstr(path, "custom"), nullptr)
            << label << ": the custom kernel must serve the call; "
            << "execution_path=" << path;
}

void expect_custom_kernel_serves(
        const std::vector<int> &rows, int K, int N, const std::string &label) {
    moe_test_utils::AlgoEnvGuard auto_algo(0);
    moe_test_utils::CustomKernelOverride ck_on(true);
    reset_grp_matmul_caches();
    Mxfp4Call call(rows, K, N, /*seed=*/17);
    moe_test_utils::ExecutionPathCaptureGuard capture;
    ASSERT_EQ(call.run(), status_t::success) << label;
    expect_custom_kernel_path(label);
    call.expect_matches_reference(label);
}

/// Whether the custom kernel's cache already holds this weight's MXFP4 pack,
/// under the key the runtime looks it up with.  A miss packs it.
bool mxfp4_pack_cached(const uint8_t *codes, const uint8_t *e8m0, int K, int N,
        bool interleave) {
    const int8_t *packed = nullptr;
    bool hit = false;
    const status_t s
            = ck::get_or_pack_weight_w4(reinterpret_cast<const int8_t *>(codes),
                    K, N, /*ldb=*/K, ck::plan_pack_nr(K, N), /*transB=*/true,
                    interleave, kGroup, &packed, &hit, /*disable_cache=*/false,
                    ck::kW4FormatMxfp4, e8m0, N);
    return s == status_t::success && hit;
}

bool mxfp4_pack_cached(const Mxfp4Weight &w, int K, int N, bool interleave) {
    return mxfp4_pack_cached(w.codes.data(), w.e8m0.data(), K, N, interleave);
}

/// Experts stacked as the routed MoE API takes them: codes [E, N, K/2] and
/// E8M0 scales [E, K/32, N].
struct StackedMxfp4 {
    int K, N;
    std::vector<Mxfp4Weight> w;
    std::vector<uint8_t> codes, scales;

    StackedMxfp4(std::mt19937 &rng, int E, int N_, int K_) : K(K_), N(N_) {
        for (int e = 0; e < E; ++e) {
            w.push_back(make_weight(rng, N, K));
            codes.insert(codes.end(), w[e].codes.begin(), w[e].codes.end());
            scales.insert(scales.end(), w[e].e8m0.begin(), w[e].e8m0.end());
        }
    }

    bool cached(int e, bool interleave) const {
        return mxfp4_pack_cached(codes.data() + w[e].codes.size() * e,
                scales.data() + w[e].e8m0.size() * e, K, N, interleave);
    }

    group_matmul_projection_params projection(
            std::vector<float> &src_scale, int T) const {
        group_matmul_projection_params proj;
        proj.output_size = N;
        proj.input_size = K;
        proj.trans_weight = true;
        proj.weight = codes.data();
        proj.ldb = K;
        proj.params = mxfp4_params(scales.data(), K, N, src_scale, T);
        proj.params.quant_params.wei_scale.dims
                = {static_cast<int64_t>(w.size()), K / kGroup, N};
        return proj;
    }
};

const char *gated_act_name(grp_matmul_gated_act_t act) {
    return act == grp_matmul_gated_act_t::swiglu_oai_mul  ? "swiglu_oai_mul"
            : act == grp_matmul_gated_act_t::silu_and_mul ? "silu_and_mul"
                                                          : "gelu_and_mul";
}

/// The activation of one gate/up pair, in double.
double gated_ref(grp_matmul_gated_act_t act, double g, double u) {
    switch (act) {
        case grp_matmul_gated_act_t::swiglu_oai_mul:
            g = std::clamp(g, -7.0, 7.0);
            u = std::clamp(u, -7.0, 7.0);
            return (1.0 + u) * g / (1.0 + std::exp(-1.702 * g));
        case grp_matmul_gated_act_t::silu_and_mul:
            return g / (1.0 + std::exp(-g)) * u;
        default: return 0.5 * g * (1.0 + std::erf(g * 0.70710678118654752)) * u;
    }
}

/// Forces the custom kernel's u8-source switch for a scope: 0 off, 1 on.
struct U8SourceOverride {
    int prev;
    explicit U8SourceOverride(int on)
        : prev(test_api::s_ck_u8_src_override.exchange(on)) {}
    ~U8SourceOverride() { test_api::s_ck_u8_src_override.store(prev); }
    U8SourceOverride(const U8SourceOverride &) = delete;
    U8SourceOverride &operator=(const U8SourceOverride &) = delete;
};

} // namespace

TEST(GroupMatmulMxfp4, DecodeMatchesDequantReference) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    expect_custom_kernel_serves({1, 3, 0, 2}, /*K=*/256, /*N=*/128, "decode");
}

TEST(GroupMatmulMxfp4, PromptMatchesDequantReference) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    expect_custom_kernel_serves(
            {64, 0, 48, 96}, /*K=*/256, /*N=*/128, "prompt");
}

namespace {

int group_sum_hoists() {
    return test_api::s_src_group_sum_hoists.load(std::memory_order_relaxed);
}

} // namespace

// A call with `kSrcGroupSumsHoistMinBytes` of source has the N-tile hoist
// compute the group sums once; smaller calls leave them to each tile.  A
// prompt runs on the prompt-tile executor.
TEST(GroupMatmulMxfp4, LargeCallHoistsTheGroupSums) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    moe_test_utils::AlgoEnvGuard auto_algo(0);
    moe_test_utils::CustomKernelOverride ck_on(true);
    const std::vector<int> rows {384, 0, 256};
    constexpr int K = 512;
    ASSERT_GE((384 + 256) * K, static_cast<int>(kSrcGroupSumsHoistMinBytes))
            << "grow this call with the threshold";
    Mxfp4Variant v;
    v.bias_dt = data_type_t::f32;
    for (const bool s8_src : {false, true}) {
        v.s8_src = s8_src;
        reset_grp_matmul_caches();
        Mxfp4Call call(rows, K, /*N=*/128, 107, v);
        moe_test_utils::ExecutionPathCaptureGuard capture;
        const int hoists = group_sum_hoists();
        ASSERT_EQ(call.run(), status_t::success) << v.name();
        expect_custom_kernel_path(v.name());
        EXPECT_EQ(group_sum_hoists() - hoists, 1) << v.name();
        call.expect_matches_reference(v.name());
    }
}

// Decode reaches the threshold too, from 23 gpt-oss-20b tokens per step,
// and runs on the flat N-tile executors, which take the hoisted sums with
// each expert's tile rows.  K is sized from the threshold so both projections
// of the fused call, and the plain call, hoist.
TEST(GroupMatmulMxfp4, DecodeCallHoistsTheGroupSums) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    moe_test_utils::AlgoEnvGuard auto_algo(0);
    moe_test_utils::CustomKernelOverride ck_on(true);
    std::vector<int> rows(32);
    size_t total_rows = 0;
    for (int e = 0; e < 32; ++e) {
        rows[e] = e == 31 ? 0 : e % 3 == 2 ? 17 : 32;
        total_rows += static_cast<size_t>(rows[e]);
    }
    const size_t k_min
            = (kSrcGroupSumsHoistMinBytes + total_rows - 1) / total_rows;
    const int K = static_cast<int>((k_min + 63) / 64 * 64);

    const auto expect_flat_path = [](const std::string &label) {
        expect_custom_kernel_path(label);
        const char *path = last_execution_path();
        ASSERT_NE(path, nullptr) << label;
        EXPECT_EQ(std::strstr(path, "ptile"), nullptr)
                << label << ": decode takes a flat N-tile executor; "
                << "execution_path=" << path;
    };

    Mxfp4Variant v;
    v.bias_dt = data_type_t::f32;
    reset_grp_matmul_caches();
    Mxfp4Call call(rows, K, /*N=*/128, 109, v);
    {
        moe_test_utils::ExecutionPathCaptureGuard capture;
        const int hoists = group_sum_hoists();
        ASSERT_EQ(call.run(), status_t::success);
        expect_flat_path("grouped decode, K=" + std::to_string(K));
        EXPECT_EQ(group_sum_hoists() - hoists, 1);
    }
    call.expect_matches_reference("grouped decode");

    reset_grp_matmul_caches();
    Mxfp4FusedCall fused(rows, 113, /*active=*/0, {K, /*I=*/K, /*N2=*/128});
    {
        moe_test_utils::ExecutionPathCaptureGuard capture;
        const int hoists = group_sum_hoists();
        ASSERT_EQ(fused.run(), status_t::success);
        expect_flat_path("fused decode, K=" + std::to_string(K));
        EXPECT_EQ(group_sum_hoists() - hoists, 2)
                << "gate/up and down projections";
    }
    fused.expect_matches_reference();
}

// Each source entry, source-scale dtype, bias dtype and weight layout the
// contract offers, through both entries, at decode and prompt sizes.
TEST(GroupMatmulMxfp4, ContractVariantsMatchDequantReference) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    moe_test_utils::AlgoEnvGuard auto_algo(0);
    moe_test_utils::CustomKernelOverride ck_on(true);
    uint32_t seed = 101;
    for (const Mxfp4Variant &v : contract_variants()) {
        for (const std::vector<int> &rows :
                {std::vector<int> {1, 3, 0, 2}, std::vector<int> {64, 0, 48}}) {
            const std::string label
                    = v.name() + ", grouped M[0]=" + std::to_string(rows[0]);
            reset_grp_matmul_caches();
            Mxfp4Call call(rows, /*K=*/256, /*N=*/128, ++seed, v);
            moe_test_utils::ExecutionPathCaptureGuard capture;
            ASSERT_EQ(call.run(), status_t::success) << label;
            expect_custom_kernel_path(label);
            call.expect_matches_reference(label);
        }
        for (const int M : {4, 128}) {
            const std::string label
                    = v.name() + ", matmul_direct M=" + std::to_string(M);
            reset_grp_matmul_caches();
            Mxfp4Call call({M}, /*K=*/256, /*N=*/128, ++seed, v);
            moe_test_utils::ExecutionPathCaptureGuard capture;
            ASSERT_EQ(call.run_direct(), status_t::success) << label;
            expect_custom_kernel_path(label);
            call.expect_matches_reference(label);
        }
    }
}

// The entries MXFP4 shares with W4A8: an f32 source under dynamic
// quantization, one source scale per row and block ({M, K/32}), and an f32
// destination, alone and together, through both entries at decode and
// prompt sizes.
TEST(GroupMatmulMxfp4, SourceScaleAndDestinationVariantsMatchDequantReference) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    moe_test_utils::AlgoEnvGuard auto_algo(0);
    moe_test_utils::CustomKernelOverride ck_on(true);
    uint32_t seed = 301;
    for (const int src : {0, 1, 2}) // bf16, f32, s8
        for (const bool per_block : {false, true})
            for (const bool f32_dst : {false, true}) {
                if (src == 0 && !per_block && !f32_dst) continue;
                Mxfp4Variant v;
                v.s8_src = src == 2;
                v.f32_src = src == 1;
                v.per_block = per_block;
                v.f32_dst = f32_dst;
                v.bias_dt = data_type_t::f32;
                v.src_scale_dt
                        = per_block ? data_type_t::bf16 : data_type_t::f32;
                for (const std::vector<int> &rows :
                        {std::vector<int> {1, 3, 0, 2},
                                std::vector<int> {64, 0, 48}}) {
                    const std::string label = v.name()
                            + ", grouped M[0]=" + std::to_string(rows[0]);
                    reset_grp_matmul_caches();
                    Mxfp4Call call(rows, /*K=*/256, /*N=*/128, ++seed, v);
                    moe_test_utils::ExecutionPathCaptureGuard capture;
                    ASSERT_EQ(call.run(), status_t::success) << label;
                    expect_custom_kernel_path(label);
                    call.expect_matches_reference(label);
                }
                for (const int M : {4, 128}) {
                    const std::string label = v.name()
                            + ", matmul_direct M=" + std::to_string(M);
                    reset_grp_matmul_caches();
                    Mxfp4Call call({M}, /*K=*/256, /*N=*/128, ++seed, v);
                    moe_test_utils::ExecutionPathCaptureGuard capture;
                    ASSERT_EQ(call.run_direct(), status_t::success) << label;
                    expect_custom_kernel_path(label);
                    call.expect_matches_reference(label);
                }
            }
}

// A fused MoE whose gate/up source has one scale per row and block
// quantizes its down projection's source the same way, over the down
// weight's K/32 blocks.
TEST(GroupMatmulMxfp4, FusedMoeTakesPerBlockSourceScales) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    moe_test_utils::AlgoEnvGuard auto_algo(0);
    moe_test_utils::CustomKernelOverride ck_on(true);
    Mxfp4Variant v;
    v.per_block = true;
    uint32_t seed = 331;
    for (const std::vector<int> &rows :
            {std::vector<int> {1, 3, 0, 2}, std::vector<int> {64, 0, 48}}) {
        const std::string label
                = "fused, per-block, M[0]=" + std::to_string(rows[0]);
        reset_grp_matmul_caches();
        Mxfp4FusedCall fused(rows, ++seed, /*active=*/0, {}, v);
        moe_test_utils::ExecutionPathCaptureGuard capture;
        ASSERT_EQ(fused.run(), status_t::success) << label;
        expect_custom_kernel_path(label);
        fused.expect_matches_reference();
    }
}

// A per-block source scale has one value per E8M0 block (K/32) and row; any
// other group count is outside the contract, and MXFP4 has no backend to
// take it, so the call fails with the destination untouched.
TEST(GroupMatmulMxfp4, SourceScaleGroupsOtherThanTheBlocksAreRefused) {
    moe_test_utils::AlgoEnvGuard auto_algo(0);
    moe_test_utils::CustomKernelOverride ck_on(true);
    for (const bool s8_src : {false, true}) {
        Mxfp4Variant v;
        v.s8_src = s8_src;
        v.per_block = true;
        reset_grp_matmul_caches();
        Mxfp4Call call({2, 2}, /*K=*/256, /*N=*/128, 211 + s8_src, v);
        std::vector<matmul_params> p = call.params();
        for (auto &q : p)
            q.quant_params.src_scale.dims = {2, 256 / 64};
        call.fill_canary();
        EXPECT_NE(call.run(p), status_t::success) << v.name();
        EXPECT_TRUE(all_canary(call.dst[0]) && all_canary(call.dst[1]))
                << v.name();
        EXPECT_NE(matmul_direct('r', false, true, 2, 128, 256, 1.0f,
                          call.src_of(0), 256, call.weights[0].codes.data(),
                          256, nullptr, 0.0f, call.dst[0].data(), 128, true,
                          matmul_batch_params_t {}, p[0]),
                status_t::success)
                << v.name() << ", matmul_direct";
        EXPECT_TRUE(all_canary(call.dst[0])) << v.name() << ", matmul_direct";
    }
}

// The kernel's fused gated store is bf16 only; with an f32 destination the
// plain kernel writes f32 and the activation runs in place after it,
// dst[:, 0:N/2] = silu(gate) * up.
TEST(GroupMatmulMxfp4, F32DestinationTakesAGatedActivation) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    moe_test_utils::AlgoEnvGuard auto_algo(0);
    moe_test_utils::CustomKernelOverride ck_on(true);
    constexpr int M = 4, K = 256, N = 128, half = N / 2;
    Mxfp4Variant v;
    v.f32_dst = true;
    reset_grp_matmul_caches();
    Mxfp4Call call({M}, K, N, 223, v);
    matmul_gated_act_params act;
    act.act = matmul_gated_act_t::silu_and_mul;
    moe_test_utils::ExecutionPathCaptureGuard capture;
    ASSERT_EQ(call.run_direct(matmul_algo_t::none, 'r', 1, &act),
            status_t::success);
    expect_custom_kernel_path("f32 dst, silu_and_mul");
    const auto h = reference_matmul(
            call.src_f32[0], M, K, call.weights[0].dequant, N);
    std::vector<float> want(static_cast<size_t>(M) * half),
            got(static_cast<size_t>(M) * half);
    for (int m = 0; m < M; ++m)
        for (int i = 0; i < half; ++i) {
            const float g = h[static_cast<size_t>(m) * N + i];
            const float u = h[static_cast<size_t>(m) * N + half + i];
            want[static_cast<size_t>(m) * half + i]
                    = g / (1.0f + std::exp(-g)) * u;
            got[static_cast<size_t>(m) * half + i]
                    = call.dst_f32[0][static_cast<size_t>(m) * N + i];
        }
    EXPECT_LT(relative_error(got, want), 2e-2);
}

// A single-expert prompt on the prompt tile executor, past the per-thread
// work floor of its u8-source copy, whose kernels read INT4 nibbles: MXFP4
// keeps its own kernels with the switch forced on, bit for bit.
TEST(GroupMatmulMxfp4, PromptTileKeepsItsKernelsUnderTheU8Source) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    moe_test_utils::AlgoEnvGuard auto_algo(0);
    moe_test_utils::CustomKernelOverride ck_on(true);
    moe_test_utils::SingleExpertCkPolicy single(1, 1, 1);
    for (const bool f32_dst : {false, true}) {
        Mxfp4Variant v;
        v.f32_dst = f32_dst;
        std::vector<uint16_t> bf16_out[2];
        std::vector<float> f32_out[2];
        for (const int u8_src : {0, 1}) {
            const std::string label
                    = v.name() + ", u8 source " + std::to_string(u8_src);
            U8SourceOverride o(u8_src);
            reset_grp_matmul_caches();
            Mxfp4Call call({200}, /*K=*/384, /*N=*/512, 421, v);
            std::vector<matmul_params> p = call.params();
            p[0].num_threads = 2;
            moe_test_utils::ExecutionPathCaptureGuard capture;
            ASSERT_EQ(call.run(p), status_t::success) << label;
            const char *path = last_execution_path();
            ASSERT_NE(path, nullptr) << label;
            EXPECT_NE(std::strstr(path, "ptile"), nullptr)
                    << label << ": " << path;
            call.expect_matches_reference(label);
            bf16_out[u8_src] = call.dst[0];
            f32_out[u8_src] = call.dst_f32[0];
        }
        EXPECT_EQ(bf16_out[0], bf16_out[1]) << v.name();
        EXPECT_EQ(0,
                std::memcmp(f32_out[0].data(), f32_out[1].data(),
                        f32_out[0].size() * sizeof(float)))
                << v.name();
    }
    reset_grp_matmul_caches();
}

namespace {

/// Runs `call` as one grouped call with `act` into rows of stride `ldc`.
status_t run_gated(Mxfp4Call &call, grp_matmul_gated_act_t act, int ldc) {
    const int E = static_cast<int>(call.rows.size());
    std::vector<const void *> src(E), wei(E), bias(E);
    std::vector<void *> dst(E);
    for (int e = 0; e < E; ++e) {
        src[e] = call.src_of(e);
        wei[e] = call.weights[e].codes.data();
        bias[e] = call.bias_of(e);
        dst[e] = call.dst_of(e);
    }
    grp_matmul_gated_act_params ga;
    ga.act = act;
    return group_matmul_direct(std::vector<char>(E, 'r'),
            std::vector<bool>(E, false), std::vector<bool>(E, true), call.rows,
            call.Ns, std::vector<int>(E, call.K), std::vector<float>(E, 1.0f),
            src, std::vector<int>(E, call.K), wei, std::vector<int>(E, call.K),
            bias, std::vector<float>(E, 0.0f), dst, std::vector<int>(E, ldc),
            std::vector<bool>(E, true), call.params(), nullptr, &ga);
}

/// Runs expert 0 of `call` as one `matmul_direct` call with `act` into rows
/// of stride `ldc`.
status_t run_gated_direct(
        Mxfp4Call &call, grp_matmul_gated_act_t act, int ldc) {
    matmul_gated_act_params ga;
    ga.act = act == grp_matmul_gated_act_t::swiglu_oai_mul
            ? matmul_gated_act_t::swiglu_oai_mul
            : act == grp_matmul_gated_act_t::silu_and_mul
            ? matmul_gated_act_t::silu_and_mul
            : matmul_gated_act_t::gelu_and_mul;
    return matmul_direct('r', false, true, call.rows[0], call.Ns[0], call.K,
            1.0f, call.src_of(0), call.K, call.weights[0].codes.data(), call.K,
            call.bias_of(0), 0.0f, call.dst_of(0), ldc, true,
            matmul_batch_params_t {}, call.params()[0], &ga);
}

/// Each of the first `experts` experts' output, read at row stride `ldc`,
/// against the activation of its dequantized product and bias.
void expect_gated_rows(const Mxfp4Call &call, grp_matmul_gated_act_t act,
        int ldc, const std::string &label, size_t experts = SIZE_MAX) {
    const bool interleaved = act == grp_matmul_gated_act_t::swiglu_oai_mul;
    for (size_t e = 0; e < call.rows.size() && e < experts; ++e) {
        const int M = call.rows[e], N = call.Ns[e], half = N / 2;
        const auto h = reference_matmul(call.src_f32[e], M, call.K,
                call.weights[e].dequant, N, call.bias_f32[e]);
        std::vector<float> want(static_cast<size_t>(M) * half),
                got(want.size());
        for (int m = 0; m < M; ++m)
            for (int i = 0; i < half; ++i) {
                const size_t r = static_cast<size_t>(m) * N;
                const double g = interleaved ? h[r + 2 * i] : h[r + i];
                const double u
                        = interleaved ? h[r + 2 * i + 1] : h[r + half + i];
                const size_t o = static_cast<size_t>(m) * half + i;
                const size_t at = static_cast<size_t>(m) * ldc + i;
                want[o] = static_cast<float>(gated_ref(act, g, u));
                got[o] = call.v.f32_dst ? call.dst_f32[e][at]
                                        : bf16_to_f32(call.dst[e][at]);
            }
        // swiglu_oai's clamps at +-7 amplify the int8 source error of the
        // values near them.
        EXPECT_LT(relative_error(got, want), 5e-2) << label << ", expert " << e;
    }
}

constexpr grp_matmul_gated_act_t kGatedActs[]
        = {grp_matmul_gated_act_t::swiglu_oai_mul,
                grp_matmul_gated_act_t::silu_and_mul,
                grp_matmul_gated_act_t::gelu_and_mul};

} // namespace

// A gated activation into a destination as wide as the projection
// (ldc = N) runs the plain kernel and the activation as a separate pass: the
// wide fused epilogue is the standard backend's, which cannot read MXFP4.
TEST(GroupMatmulMxfp4, GatedActivationsIntoAWideDestination) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    moe_test_utils::AlgoEnvGuard auto_algo(0);
    moe_test_utils::CustomKernelOverride ck_on(true);
    constexpr int K = 256, N = 128;
    uint32_t seed = 501;
    for (const auto act : kGatedActs)
        for (const bool f32_dst : {false, true})
            for (const std::vector<int> &rows :
                    {std::vector<int> {4, 3}, std::vector<int> {64, 48}}) {
                Mxfp4Variant v;
                v.f32_dst = f32_dst;
                const std::string label = std::string(gated_act_name(act))
                        + (f32_dst ? ", f32" : ", bf16")
                        + ", M[0]=" + std::to_string(rows[0]);
                reset_grp_matmul_caches();
                Mxfp4Call call(rows, K, N, ++seed, v);
                moe_test_utils::ExecutionPathCaptureGuard capture;
                ASSERT_EQ(run_gated(call, act, N), status_t::success) << label;
                expect_custom_kernel_path(label);
                expect_gated_rows(call, act, N, label);
            }
}

// Into a tight destination (ldc = N/2) the kernel fuses the activation into
// its store: bf16, and a bias only under swiglu_oai_mul.  Grouped and dense
// calls alike; the stores it cannot fuse are refused before any work.
TEST(GroupMatmulMxfp4, GatedActivationsIntoATightDestination) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    moe_test_utils::AlgoEnvGuard auto_algo(0);
    moe_test_utils::CustomKernelOverride ck_on(true);
    constexpr int K = 256, N = 128, half = N / 2;
    uint32_t seed = 601;
    for (const auto act : kGatedActs)
        for (const data_type_t bias : {data_type_t::none, data_type_t::bf16}) {
            if (bias != data_type_t::none
                    && act != grp_matmul_gated_act_t::swiglu_oai_mul)
                continue;
            for (const std::vector<int> &rows :
                    {std::vector<int> {4, 3}, std::vector<int> {64, 48}}) {
                Mxfp4Variant v;
                v.bias_dt = bias;
                const std::string label = std::string(gated_act_name(act))
                        + (bias == data_type_t::none ? "" : ", bias")
                        + ", M[0]=" + std::to_string(rows[0]);
                reset_grp_matmul_caches();
                Mxfp4Call call(rows, K, N, ++seed, v);
                moe_test_utils::ExecutionPathCaptureGuard capture;
                ASSERT_EQ(run_gated(call, act, half), status_t::success)
                        << label;
                expect_custom_kernel_path(label);
                expect_gated_rows(call, act, half, label);
                ASSERT_EQ(run_gated_direct(call, act, half), status_t::success)
                        << label << ", matmul_direct";
                expect_gated_rows(
                        call, act, half, label + ", matmul_direct", 1);
            }
        }
}

// A fused MoE whose gated store the kernel cannot fuse, a split-halves
// activation with a bias or an f32 intermediate, takes the wide arena: the
// plain kernel, then the activation as a separate pass.
TEST(GroupMatmulMxfp4, FusedMoeRunsUnfusableStoresAsASeparatePass) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    moe_test_utils::AlgoEnvGuard auto_algo(0);
    moe_test_utils::CustomKernelOverride ck_on(true);
    uint32_t seed = 611;
    for (const bool f32_dst : {false, true})
        for (const data_type_t bias : {data_type_t::none, data_type_t::bf16}) {
            if (!f32_dst && bias == data_type_t::none) continue;
            Mxfp4Variant v;
            v.f32_src = f32_dst;
            v.f32_dst = f32_dst;
            v.bias_dt = bias;
            for (const std::vector<int> &rows :
                    {std::vector<int> {4, 3}, std::vector<int> {64, 48}}) {
                const std::string label = std::string(f32_dst ? "f32" : "bf16")
                        + (bias == data_type_t::none ? "" : ", op1 bias")
                        + ", M[0]=" + std::to_string(rows[0]);
                reset_grp_matmul_caches();
                Mxfp4FusedCall fused(rows, ++seed, /*active=*/0, {}, v);
                moe_test_utils::ExecutionPathCaptureGuard capture;
                ASSERT_EQ(fused.run(), status_t::success) << label;
                expect_custom_kernel_path(label);
                fused.expect_matches_reference();
            }
        }
}

// A fused MoE whose Op1 destination the library allocates takes every
// source and destination pair the contract admits: a pre-quantized s8
// source, an f32 source into bf16, and a bf16 source into f32.
TEST(GroupMatmulMxfp4, FusedMoeTakesEverySourceAndDestinationPair) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    moe_test_utils::AlgoEnvGuard auto_algo(0);
    moe_test_utils::CustomKernelOverride ck_on(true);
    uint32_t seed = 641;
    for (const int form : {0, 1, 2}) {
        Mxfp4Variant v;
        v.s8_src = form == 0;
        v.f32_src = form == 1;
        v.f32_dst = form == 2;
        for (const std::vector<int> &rows :
                {std::vector<int> {4, 3}, std::vector<int> {64, 48}}) {
            const std::string label
                    = v.name() + ", M[0]=" + std::to_string(rows[0]);
            reset_grp_matmul_caches();
            Mxfp4FusedCall fused(rows, ++seed, /*active=*/0, {}, v);
            moe_test_utils::ExecutionPathCaptureGuard capture;
            ASSERT_EQ(fused.run(), status_t::success) << label;
            expect_custom_kernel_path(label);
            fused.expect_matches_reference();
        }
    }
}

// Under per-token source scales the experts of one call may differ in K:
// each pack holds its own E8M0 blocks of 32.  Per-block source scales give
// the call one group count, so a mix of K fails with the destination
// untouched.
TEST(GroupMatmulMxfp4, ExpertsMayDifferInKPerToken) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    moe_test_utils::AlgoEnvGuard auto_algo(0);
    moe_test_utils::CustomKernelOverride ck_on(true);
    uint32_t seed = 661;
    for (const bool per_block : {false, true})
        for (const std::vector<int> &rows :
                {std::vector<int> {4, 3}, std::vector<int> {64, 48}}) {
            Mxfp4Variant v;
            v.per_block = per_block;
            const std::string label
                    = std::string(per_block ? "per block" : "per token")
                    + ", M[0]=" + std::to_string(rows[0]);
            reset_grp_matmul_caches();
            Mxfp4Call a({rows[0]}, /*K=*/128, /*N=*/128, ++seed, v);
            Mxfp4Call b({rows[1]}, /*K=*/256, /*N=*/128, ++seed, v);
            a.fill_canary();
            b.fill_canary();
            moe_test_utils::ExecutionPathCaptureGuard capture;
            const status_t st = group_matmul_direct({'r', 'r'}, {false, false},
                    {true, true}, rows, {128, 128}, {128, 256}, {1.0f, 1.0f},
                    {a.src_of(0), b.src_of(0)}, {128, 256},
                    {a.weights[0].codes.data(), b.weights[0].codes.data()},
                    {128, 256}, {nullptr, nullptr}, {0.0f, 0.0f},
                    {a.dst_of(0), b.dst_of(0)}, {128, 128}, {true, true},
                    {a.expert_params(0), b.expert_params(0)});
            if (per_block) {
                EXPECT_NE(st, status_t::success) << label;
                EXPECT_TRUE(all_canary(a.dst[0]) && all_canary(b.dst[0]))
                        << label;
                continue;
            }
            ASSERT_EQ(st, status_t::success) << label;
            expect_custom_kernel_path(label);
            a.expect_matches_reference(label + ", K=128");
            b.expect_matches_reference(label + ", K=256");
        }
}

// More slots than one custom-kernel call holds run as consecutive slices,
// per token or per block, while the active experts fit one call.
TEST(GroupMatmulMxfp4, ManySlotsRunAsSlices) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    moe_test_utils::AlgoEnvGuard auto_algo(0);
    moe_test_utils::CustomKernelOverride ck_on(true);
    for (const bool per_block : {false, true}) {
        Mxfp4Variant v;
        v.per_block = per_block;
        std::vector<int> rows(300, 0);
        rows[3] = 4;
        rows[280] = 2;
        const std::string label = per_block ? "per block" : "per token";
        reset_grp_matmul_caches();
        Mxfp4Call call(rows, /*K=*/256, /*N=*/128, 681 + per_block, v);
        moe_test_utils::ExecutionPathCaptureGuard capture;
        ASSERT_EQ(call.run(), status_t::success) << label;
        expect_custom_kernel_path(label);
        call.expect_matches_reference(label);
    }
}

// The inner kernel only ever chooses among non-custom tile paths, none of
// which reads f4_e2m1, so pinning it leaves MXFP4 on the custom kernel.
TEST(GroupMatmulMxfp4, InnerKernelPinStaysOnTheCustomKernel) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    moe_test_utils::AlgoEnvGuard auto_algo(0);
    moe_test_utils::CustomKernelOverride ck_on(true);
    reset_grp_matmul_caches();
    Mxfp4Call call({1, 3, 0, 2}, /*K=*/256, /*N=*/128, 19);
    moe_test_utils::ExecutionPathCaptureGuard capture;
    ASSERT_EQ(call.run(matmul_algo_t::aocl_dlp_blocked), status_t::success);
    expect_custom_kernel_path("inner aocl_dlp_blocked");
    call.expect_matches_reference("inner aocl_dlp_blocked");
}

// N = 80 is not a multiple of the custom kernel's pack width.  MXFP4 has no
// other backend, so the call fails whatever the scheduler setting.
TEST(GroupMatmulMxfp4, ShapeTheKernelRefusesFailsTheCall) {
    moe_test_utils::CustomKernelOverride ck_on(true);
    for (const int algo : {0, 3}) {
        moe_test_utils::AlgoEnvGuard pin(algo);
        reset_grp_matmul_caches();
        Mxfp4Call call({1, 2, 1, 3}, /*K=*/256, /*N=*/80, 23 + algo);
        EXPECT_NE(call.run(), status_t::success) << "ALGO " << algo;
    }
}

TEST(GroupMatmulMxfp4, CustomKernelOffFailsTheCall) {
    moe_test_utils::AlgoEnvGuard auto_algo(0);
    moe_test_utils::CustomKernelOverride ck_off(false);
    reset_grp_matmul_caches();
    Mxfp4Call call({1, 2, 1, 3}, /*K=*/256, /*N=*/128, 31);
    EXPECT_NE(call.run(), status_t::success);
}

// Every scheduler but ALGO 3 runs AOCL tiles, so a pin to one of them is a
// request MXFP4 cannot honour.
TEST(GroupMatmulMxfp4, SchedulerPinOtherThanAlgo3FailsTheCall) {
    moe_test_utils::CustomKernelOverride ck_on(true);
    for (const int algo : {1, 2, 5, 6}) {
        moe_test_utils::AlgoEnvGuard pin(algo);
        reset_grp_matmul_caches();
        Mxfp4Call call({1, 3, 0, 2}, /*K=*/256, /*N=*/128, 37 + algo);
        EXPECT_NE(call.run(), status_t::success) << "ALGO " << algo;
    }
}

// ALGO 4 runs calls of several experts itself and has no MXFP4 kernels: such
// a call fails under it.
TEST(GroupMatmulMxfp4, Algo4FailsTheCall) {
    moe_test_utils::CustomKernelOverride ck_on(true);
    moe_test_utils::AlgoEnvGuard algo4(4);
    // 7 leaves both phases unset, whatever the environment sets.
    moe_test_utils::AutoDecodeAlgoOverride decode_unset(7);
    moe_test_utils::AutoPromptAlgoOverride prompt_unset(7);
    reset_grp_matmul_caches();
    Mxfp4Call call({1, 3, 0, 2}, /*K=*/256, /*N=*/128, 41);
    EXPECT_EQ(call.run(), status_t::unimplemented);
}

namespace {

// Turns expert `e` of an MXFP4 call into a well-formed W4A8 s4 expert, so
// only the mix of weight types is wrong.
void make_s4_expert(std::vector<matmul_params> &p, size_t e,
        std::vector<float> &s4_scale, int K, int N) {
    s4_scale.assign(static_cast<size_t>(K / kGroup) * N, 0.01f);
    p[e].dtypes.wei = data_type_t::s4;
    p[e].quant_params.wei_scale.buff = s4_scale.data();
    p[e].quant_params.wei_scale.dt = data_type_t::f32;
}

} // namespace

TEST(GroupMatmulMxfp4, MixedWeightTypesAreRefused) {
    moe_test_utils::AlgoEnvGuard auto_algo(0);
    moe_test_utils::CustomKernelOverride ck_on(true);
    for (const size_t other : {0, 1}) {
        reset_grp_matmul_caches();
        Mxfp4Call call({2, 2}, /*K=*/256, /*N=*/128, 41);
        std::vector<matmul_params> p = call.params();
        std::vector<float> s4_scale;
        make_s4_expert(p, other, s4_scale, 256, 128);
        EXPECT_NE(call.run(p), status_t::success) << "s4 expert " << other;
    }
}

// f8_e8m0 has one unrepresentable byte, NaN; the pack refuses it, and with
// it the only backend, so the call fails.
TEST(GroupMatmulMxfp4, NaNScaleFailsTheCall) {
    moe_test_utils::AlgoEnvGuard auto_algo(0);
    moe_test_utils::CustomKernelOverride ck_on(true);
    reset_grp_matmul_caches();
    Mxfp4Call call({1, 2}, /*K=*/256, /*N=*/128, 43);
    call.weights[1].e8m0[5] = zendnnl::common::kE8M0NaN;
    EXPECT_NE(call.run(), status_t::success);
}

// Each expert's scales are packed with its own row stride, its N.
TEST(GroupMatmulMxfp4, MixedNExpertsMatchDequantReference) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    moe_test_utils::AlgoEnvGuard auto_algo(0);
    moe_test_utils::CustomKernelOverride ck_on(true);
    for (const std::vector<int> &Ns :
            {std::vector<int> {128, 64}, std::vector<int> {64, 128}}) {
        const std::string label = "N = {" + std::to_string(Ns[0]) + ", "
                + std::to_string(Ns[1]) + "}";
        reset_grp_matmul_caches();
        Mxfp4Call call({2, 2}, /*K=*/256, Ns, 47);
        moe_test_utils::ExecutionPathCaptureGuard capture;
        ASSERT_EQ(call.run(), status_t::success) << label;
        expect_custom_kernel_path(label);
        call.expect_matches_reference(label);
    }
}

// Large E8M0 scales against a small source scale: the f32 sums would
// overflow before the source scale brings the result back into range.
// Each output is K * (1 * 6.0) * 2^123 * 2^-120 = 12288 at K = 256.
TEST(GroupMatmulMxfp4, ExtremeScalesStayFinite) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    moe_test_utils::AlgoEnvGuard auto_algo(0);
    moe_test_utils::CustomKernelOverride ck_on(true);
    constexpr int K = 256, N = 128;
    const float want = static_cast<float>(K) * 6.0f * 8.0f;
    Mxfp4Variant s8_src;
    s8_src.s8_src = true;
    for (const bool direct : {false, true}) {
        reset_grp_matmul_caches();
        Mxfp4Call call({2}, K, N, 89, s8_src);
        std::fill(call.weights[0].codes.begin(), call.weights[0].codes.end(),
                0x77); // every code is 6.0
        std::fill(call.weights[0].e8m0.begin(), call.weights[0].e8m0.end(),
                250); // 2^123
        std::fill(call.src_s8[0].begin(), call.src_s8[0].end(), 1);
        std::fill(call.src_scale[0].begin(), call.src_scale[0].end(),
                std::ldexp(1.0f, -120));
        ASSERT_EQ(direct ? call.run_direct() : call.run(), status_t::success);
        for (const uint16_t got : call.dst[0])
            ASSERT_EQ(bf16_to_f32(got), want)
                    << (direct ? "matmul_direct" : "group_matmul_direct");
    }
}

// The routed MoE API slices stacked [E, N, K/2] codes per expert as
// four-bit bytes and hands each expert to the custom kernel.
TEST(GroupMatmulMxfp4, RoutedMoeServesEveryExpert) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    constexpr int E = 4, T = 4, K = 256, N = 128;
    moe_test_utils::AlgoEnvGuard auto_algo(0);
    moe_test_utils::CustomKernelOverride ck_on(true);
    reset_grp_matmul_caches();
    std::mt19937 rng(13);
    std::uniform_real_distribution<float> x(-1.0f, 1.0f);
    const StackedMxfp4 stacked(rng, E, N, K);
    std::vector<uint16_t> src(static_cast<size_t>(T) * K);
    std::vector<float> src_f32(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = f32_to_bf16(x(rng));
        src_f32[i] = bf16_to_f32(src[i]);
    }
    const std::vector<int32_t> ids {1, 2, 3, 0};
    const std::vector<float> topk_weights(T, 1.0f);
    std::vector<uint16_t> out(static_cast<size_t>(T) * N, 0);
    std::vector<float> src_scale;
    const group_matmul_projection_params proj
            = stacked.projection(src_scale, T);
    group_matmul_routing_params routing;
    routing.topk_ids = ids.data();
    routing.topk_weights = topk_weights.data();
    moe_test_utils::ExecutionPathCaptureGuard capture;
    ASSERT_EQ(routed_fused_moe_direct('r', false, src.data(), K, T, E,
                      /*topk=*/1, out.data(), N, proj, routing),
            status_t::success);
    expect_custom_kernel_path("routed");
    for (int t = 0; t < T; ++t) {
        const std::vector<float> row(
                src_f32.begin() + t * K, src_f32.begin() + (t + 1) * K);
        const std::vector<uint16_t> got(
                out.begin() + t * N, out.begin() + (t + 1) * N);
        const auto want
                = reference_matmul(row, 1, K, stacked.w[ids[t]].dequant, N);
        EXPECT_LT(relative_error(got, want), 2e-2)
                << "token " << t << ", expert " << ids[t];
    }
}

// An MXFP4 pack embeds its scales, so the prepack warm can pack an idle
// expert only from its prepack-extras tail entry: after a call with two of
// four experts active, all four packs are already cached, on both
// projections of a fused call.
TEST(GroupMatmulMxfp4, IdleTailExpertsArePackedAhead) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    moe_test_utils::AlgoEnvGuard auto_algo(0);
    moe_test_utils::CustomKernelOverride ck_on(true);
    WeightCacheGuard wc(1);
    constexpr int K = 256, N = 128;

    reset_grp_matmul_caches();
    Mxfp4Call call({2, 3, 0, 0}, K, N, 73);
    std::vector<matmul_params> p = call.params();
    for (auto &pp : p) {
        pp.active_matmul = 2;
        pp.total_matmul = 4;
    }
    ASSERT_EQ(call.run(p), status_t::success);
    call.expect_matches_reference("two of four active");
    for (int e = 0; e < 4; ++e)
        EXPECT_TRUE(mxfp4_pack_cached(call.weights[e], K, N, false))
                << "expert " << e;

    reset_grp_matmul_caches();
    Mxfp4FusedCall fused({8, 8, 0, 0}, 77, /*active=*/2);
    ASSERT_EQ(fused.run(), status_t::success);
    fused.expect_matches_reference();
    for (int e = 0; e < 4; ++e) {
        EXPECT_TRUE(mxfp4_pack_cached(
                fused.op1.weights[e], fused.K, fused.N13, /*interleave=*/true))
                << "fused gate/up, expert " << e;
        EXPECT_TRUE(mxfp4_pack_cached(fused.w2[e], fused.I, fused.N2, false))
                << "fused down, expert " << e;
    }
}

// An idle expert's scale descriptor gets the same check as an active one's:
// the warm skips a tail entry that is not f8_e8m0 {K/32, N} rather than
// read K/32 rows from it.
TEST(GroupMatmulMxfp4, MalformedIdleTailScalesAreNotWarmed) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    moe_test_utils::AlgoEnvGuard auto_algo(0);
    moe_test_utils::CustomKernelOverride ck_on(true);
    WeightCacheGuard wc(1);
    constexpr int K = 256, N = 128;

    reset_grp_matmul_caches();
    Mxfp4Call call({2, 3, 0, 0}, K, N, 79);
    std::vector<matmul_params> p = call.params();
    for (auto &pp : p) {
        pp.active_matmul = 2;
        pp.total_matmul = 4;
    }
    p[3].quant_params.wei_scale.dims = {1, N};
    ASSERT_EQ(call.run(p), status_t::success);
    EXPECT_TRUE(mxfp4_pack_cached(call.weights[2], K, N, false));
    EXPECT_FALSE(mxfp4_pack_cached(call.weights[3], K, N, false))
            << "a {1, N} descriptor must not be warmed as {K/32, N}";

    reset_grp_matmul_caches();
    Mxfp4FusedCall fused({8, 8, 0, 0}, 81, /*active=*/2);
    fused.fused.down_scale[3].dims = {1, fused.N2};
    ASSERT_EQ(fused.run(), status_t::success);
    EXPECT_TRUE(mxfp4_pack_cached(fused.w2[2], fused.I, fused.N2, false));
    EXPECT_FALSE(mxfp4_pack_cached(fused.w2[3], fused.I, fused.N2, false));
}

namespace {

/// Experts 0 and 1 run and 2 and 3 are the prepack-extras tail.
std::vector<matmul_params> two_of_four_active(Mxfp4Call &call) {
    std::vector<matmul_params> p = call.params();
    for (auto &pp : p) {
        pp.active_matmul = 2;
        pp.total_matmul = 4;
    }
    return p;
}

void expect_idle_packed(const Mxfp4Call &call, const std::string &what) {
    for (int e = 2; e < 4; ++e)
        EXPECT_TRUE(mxfp4_pack_cached(call.weights[e], call.K, call.Ns[e],
                /*interleave=*/false))
                << what << ", expert " << e;
}

} // namespace

// The warm runs once per weight pool, and an MXFP4 pool includes its
// scales: idle experts' scales that arrive on a later call are warmed.
TEST(GroupMatmulMxfp4, IdleScalesThatArriveLaterAreWarmed) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    moe_test_utils::AlgoEnvGuard auto_algo(0);
    moe_test_utils::CustomKernelOverride ck_on(true);
    WeightCacheGuard wc(1);
    reset_grp_matmul_caches();
    Mxfp4Call call({2, 3, 0, 0}, 256, 128, 85);
    const std::vector<matmul_params> p = two_of_four_active(call);
    std::vector<matmul_params> no_idle_scales = p;
    for (int e = 2; e < 4; ++e)
        no_idle_scales[e].quant_params.wei_scale.buff = nullptr;
    ASSERT_EQ(call.run(no_idle_scales), status_t::success);
    ASSERT_FALSE(mxfp4_pack_cached(call.weights[2], 256, 128, false));
    ASSERT_EQ(call.run(p), status_t::success);
    call.expect_matches_reference("idle scales on the second call");
    expect_idle_packed(call, "idle scales on the second call");
}

// A dynamic f32 source warms the idle experts as a bf16 one does, with the
// grouped source quantization on or off.
TEST(GroupMatmulMxfp4, F32SourcesPackIdleExpertsAhead) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    moe_test_utils::AlgoEnvGuard auto_algo(0);
    moe_test_utils::CustomKernelOverride ck_on(true);
    WeightCacheGuard wc(1);
    for (const int dq : {0, 1}) {
        moe_test_utils::GroupDqOverride group_dq(dq);
        reset_grp_matmul_caches();
        Mxfp4Variant v;
        v.f32_src = true;
        Mxfp4Call call({2, 3, 0, 0}, 256, 128, 87 + dq, v);
        const std::string what = "group DQ " + std::to_string(dq);
        ASSERT_EQ(call.run(two_of_four_active(call)), status_t::success);
        call.expect_matches_reference(what);
        expect_idle_packed(call, what);
    }
}

// One MXFP4 block per column (K = 32) is still a per-block layout.
TEST(GroupMatmulMxfp4, SingleBlockLayersPackIdleExpertsAhead) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    moe_test_utils::AlgoEnvGuard auto_algo(0);
    moe_test_utils::CustomKernelOverride ck_on(true);
    WeightCacheGuard wc(1);
    reset_grp_matmul_caches();
    Mxfp4Call call({2, 3, 0, 0}, kGroup, 128, 89);
    ASSERT_EQ(call.run(two_of_four_active(call)), status_t::success);
    call.expect_matches_reference("K = 32");
    expect_idle_packed(call, "K = 32");
}

// The down projection's tail takes its format from the first active
// expert: an inactive slot's weight dtype may be unset.  The gated
// activation still asks every slot for the same destination dtype.
TEST(GroupMatmulMxfp4, FusedDownTailIgnoresAnUnsetInactiveSlot) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    moe_test_utils::AlgoEnvGuard auto_algo(0);
    moe_test_utils::CustomKernelOverride ck_on(true);
    WeightCacheGuard wc(1);
    reset_grp_matmul_caches();
    Mxfp4FusedCall fused({0, 8, 0, 0}, 91, /*active=*/2);
    fused.params[0] = matmul_params {};
    fused.params[0].dtypes.dst = fused.params[1].dtypes.dst;
    fused.params[0].active_matmul = 2;
    fused.params[0].total_matmul = 4;
    ASSERT_EQ(fused.run(), status_t::success);
    fused.expect_matches_reference();
    for (int e = 2; e < 4; ++e)
        EXPECT_TRUE(mxfp4_pack_cached(fused.w2[e], fused.I, fused.N2, false))
                << "down, expert " << e;
}

// The routed adapter gives each projection the idle experts' scales in
// the order of their weight pointers: one call with a single expert
// active warms every expert, on one projection or on a fused pair.
TEST(GroupMatmulMxfp4, RoutedCallsPackIdleExpertsAhead) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    constexpr int E = 4, T = 4, K = 256, I = 64;
    moe_test_utils::AlgoEnvGuard auto_algo(0);
    moe_test_utils::CustomKernelOverride ck_on(true);
    WeightCacheGuard wc(1);
    std::mt19937 rng(17);
    std::uniform_real_distribution<float> x(-1.0f, 1.0f);
    const StackedMxfp4 up(rng, E, 2 * I, K), down(rng, E, K, I);
    std::vector<uint16_t> src(static_cast<size_t>(T) * K);
    for (auto &v : src)
        v = f32_to_bf16(x(rng));
    const std::vector<int32_t> ids(T, 2);
    const std::vector<float> topk_weights(T, 1.0f);
    group_matmul_routing_params routing;
    routing.topk_ids = ids.data();
    routing.topk_weights = topk_weights.data();
    std::vector<float> up_scale, down_scale;
    const group_matmul_projection_params first = up.projection(up_scale, T);
    const group_matmul_projection_params second
            = down.projection(down_scale, T);
    grp_matmul_gated_act_params act;
    act.act = grp_matmul_gated_act_t::silu_and_mul;

    reset_grp_matmul_caches();
    std::vector<uint16_t> out(static_cast<size_t>(T) * 2 * I, 0);
    ASSERT_EQ(routed_fused_moe_direct('r', false, src.data(), K, T, E,
                      /*topk=*/1, out.data(), 2 * I, first, routing),
            status_t::success);
    for (int e = 0; e < E; ++e)
        EXPECT_TRUE(up.cached(e, /*interleave=*/false))
                << "one projection, expert " << e;

    reset_grp_matmul_caches();
    out.assign(static_cast<size_t>(T) * K, 0);
    ASSERT_EQ(routed_fused_moe_direct('r', false, src.data(), K, T, E,
                      /*topk=*/1, out.data(), K, first, routing, &second, &act),
            status_t::success);
    for (int e = 0; e < E; ++e) {
        EXPECT_TRUE(up.cached(e, /*interleave=*/true))
                << "fused gate/up, expert " << e;
        EXPECT_TRUE(down.cached(e, /*interleave=*/false))
                << "fused down, expert " << e;
    }
}

namespace {

// The down projection's scales come from `fused.down_scale`, which Op1's
// admission never sees; every active expert's descriptor is checked before
// either projection is packed.
void expect_malformed_down_scales_refused() {
    moe_test_utils::AlgoEnvGuard auto_algo(0);
    moe_test_utils::CustomKernelOverride ck_on(true);
    using F = Mxfp4FusedCall;
    for (const int one : {0, 1}) {
        reset_grp_matmul_caches();
        F call({8, 8, 8, 8}, 83 + one);
        // One expert's {1, N}, or every expert's block count wrong.
        for (int e = 0; e < 4; ++e) {
            if (one && e != 1) continue;
            call.fused.down_scale[e].dims = one
                    ? std::vector<int64_t> {1, call.N2}
                    : std::vector<int64_t> {call.I / 16, call.N2};
        }
        EXPECT_NE(call.run(), status_t::success)
                << (one ? "expert 1 {1, N}" : "every expert {K/16, N}");
        EXPECT_TRUE(call.outputs_untouched());
    }
    reset_grp_matmul_caches();
    F call({8, 8}, 85);
    std::vector<float> f32_scale(call.w2[1].e8m0.size(), 1.0f);
    call.fused.down_scale[1].buff = f32_scale.data();
    call.fused.down_scale[1].dt = data_type_t::f32;
    EXPECT_NE(call.run(), status_t::success) << "f32 down scales";
    EXPECT_TRUE(call.outputs_untouched());
}

// The fused down projection takes its bias, weight scale and weight zero
// point from `fused`, and its weight type from Op1: an MX type in any of the
// three, or f8_e8m0 scales on a W4A8 down weight, fails the call before
// either projection runs.  Each case edits a well-formed W4A8 fused MoE, the
// traditional type whose backends would read the MX bytes as another type.
void expect_fused_down_mx_roles_refused() {
    moe_test_utils::AlgoEnvGuard auto_algo(0);
    using F = Mxfp4FusedCall;
    struct W4a8Fused {
        F call;
        std::vector<std::vector<float>> op1_scale, down_scale;
        W4a8Fused() : call({8, 8}, 87) {
            const size_t E = call.params.size();
            op1_scale.resize(E);
            down_scale.assign(E,
                    std::vector<float>(
                            static_cast<size_t>(call.I / kGroup) * call.N2,
                            0.01f));
            for (size_t e = 0; e < E; ++e) {
                make_s4_expert(call.params, e, op1_scale[e], call.K, call.N13);
                call.fused.down_scale[e].buff = down_scale[e].data();
                call.fused.down_scale[e].dt = data_type_t::f32;
            }
        }
    };
    std::vector<uint8_t> e8(
            static_cast<size_t>(F::Shape {}.I / kGroup) * F::Shape {}.N2, 127);
    std::vector<float> mx_buf(static_cast<size_t>(F::Shape {}.I) * 4, 0.0f);
    const std::vector<std::pair<const char *, std::function<void(F &)>>> cases {
            {"f8_e8m0 down scales on a W4A8 down weight",
                    [&](F &c) {
        for (auto &s : c.fused.down_scale) {
            s.buff = e8.data();
            s.dt = data_type_t::f8_e8m0;
        }
    }},
            {"an f8_e8m0 down bias",
                    [&](F &c) {
        c.fused.bias_dt_down = data_type_t::f8_e8m0;
        for (auto &b : c.fused.bias_down)
            b = mx_buf.data();
    }},
            {"an f8_e8m0 down zero point",
                    [&](F &c) {
        c.fused.down_zp.resize(c.params.size());
        for (auto &zp : c.fused.down_zp) {
            zp.buff = mx_buf.data();
            zp.dt = data_type_t::f8_e8m0;
            zp.dims = {c.I / kGroup, c.N2};
        }
    }},
            {"f4_e2m1 down scales",
                    [](F &c) {
        for (auto &s : c.fused.down_scale)
            s.dt = data_type_t::f4_e2m1;
    }},
    };
    for (const bool ck_enabled : {true, false}) {
        moe_test_utils::CustomKernelOverride ck_override(ck_enabled);
        const std::string kernel
                = ck_enabled ? ", custom kernel on" : ", custom kernel off";
        if (ck::avx512vnni_available()) {
            reset_grp_matmul_caches();
            W4a8Fused control;
            EXPECT_EQ(control.call.run(), status_t::success)
                    << "the unedited W4A8 fused call" << kernel;
        }
        for (const auto &[name, edit] : cases) {
            reset_grp_matmul_caches();
            W4a8Fused w;
            edit(w.call);
            EXPECT_NE(w.call.run(), status_t::success) << name << kernel;
            EXPECT_TRUE(w.call.outputs_untouched()) << name << kernel;
        }
    }
}

// f4_e2m1 is only ever a weight, and f8_e8m0 only the scale of an f4_e2m1
// weight; any other use is refused on both entries before a backend that
// would read the type as another one sees it.  Each case is a well-formed
// MXFP4 call with one role changed.
void expect_mx_roles_refused() {
    constexpr int M = 4, K = 256, N = 128;
    const std::vector<uint16_t> src(
            static_cast<size_t>(M) * K, f32_to_bf16(0.5f));
    const std::vector<int8_t> w8(static_cast<size_t>(K) * N, 1);
    const std::vector<uint8_t> w4(static_cast<size_t>(K) * N / 2, 0x11);
    const std::vector<uint16_t> w16(
            static_cast<size_t>(K) * N, f32_to_bf16(0.01f));
    std::vector<uint8_t> e8(static_cast<size_t>(K / kGroup) * N, 127);
    std::vector<uint8_t> mx_buf(static_cast<size_t>(M) * N * 4, 127);
    std::vector<float> src_scale;
    std::vector<float> s4_scale(static_cast<size_t>(K / kGroup) * N, 0.01f);
    std::vector<uint16_t> dst(static_cast<size_t>(M) * N);
    const auto mx_quant = [&](auto &q, std::vector<int64_t> dims) {
        q.buff = mx_buf.data();
        q.dt = data_type_t::f8_e8m0;
        q.dims = std::move(dims);
    };
    // Calls of the traditional types, which reach AOCL or oneDNN unless the
    // role check refuses them.
    const auto bf16_call = [](matmul_params &p) {
        p = matmul_params {};
        p.dtypes.src = data_type_t::bf16;
        p.dtypes.wei = data_type_t::bf16;
        p.dtypes.dst = data_type_t::bf16;
    };
    const auto w4a8_call = [&](matmul_params &p) {
        p.dtypes.wei = data_type_t::s4;
        p.quant_params.wei_scale.buff = s4_scale.data();
        p.quant_params.wei_scale.dt = data_type_t::f32;
    };
    const auto binary_add = [&](matmul_params &p) {
        matmul_post_op po;
        po.po_type = post_op_type_t::binary_add;
        po.buff = mx_buf.data();
        po.dtype = data_type_t::f8_e8m0;
        po.dims = {M, N};
        p.postop_.push_back(po);
    };
    struct Case {
        const char *name;
        std::function<void(matmul_params &)> edit;
        const void *weight;
        const void *bias;
    };
    const std::vector<Case> cases {
            {"dynamic INT8 with f8_e8m0 {1, N} scales",
                    [&](matmul_params &p) {
        p.dtypes.wei = data_type_t::s8;
        p.quant_params.wei_scale.dims = {1, N};
    }, w8.data(), nullptr},
            {"WOQ s4 with f8_e8m0 {K/32, N} scales",
                    [](matmul_params &p) {
        p.dtypes.wei = data_type_t::s4;
        p.dtypes.compute = data_type_t::none;
        p.dynamic_quant = false;
    }, w4.data(), nullptr},
            {"W4A8 s4 with f8_e8m0 {K/32, N} scales",
                    [](matmul_params &p) { p.dtypes.wei = data_type_t::s4; },
                    w4.data(), nullptr},
            {"an f8_e8m0 weight",
                    [](matmul_params &p) {
        p.dtypes.wei = data_type_t::f8_e8m0;
        p.dtypes.compute = data_type_t::none;
        p.dynamic_quant = false;
    }, e8.data(), nullptr},
            {"an f4_e2m1 source",
                    [](matmul_params &p) {
        p.dtypes.src = data_type_t::f4_e2m1;
        p.dtypes.compute = data_type_t::none;
        p.dynamic_quant = false;
    }, w4.data(), nullptr},
            {"an f8_e8m0 bias",
                    [](matmul_params &p) {
        p.dtypes.bias = data_type_t::f8_e8m0;
    }, w4.data(), mx_buf.data()},
            {"an f4_e2m1 destination",
                    [](matmul_params &p) {
        p.dtypes.dst = data_type_t::f4_e2m1;
    }, w4.data(), nullptr},
            {"an f4_e2m1 compute type",
                    [](matmul_params &p) {
        p.dtypes.compute = data_type_t::f4_e2m1;
    }, w4.data(), nullptr},
            {"f8_e8m0 source scales",
                    [&](matmul_params &p) {
        mx_quant(p.quant_params.src_scale, {M, 1});
    }, w4.data(), nullptr},
            {"an f8_e8m0 source zero point",
                    [&](matmul_params &p) {
        mx_quant(p.quant_params.src_zp, {M, 1});
    }, w4.data(), nullptr},
            {"an f8_e8m0 weight zero point",
                    [&](matmul_params &p) {
        mx_quant(p.quant_params.wei_zp, {K / kGroup, N});
    }, w4.data(), nullptr},
            {"f8_e8m0 destination scales",
                    [&](matmul_params &p) {
        mx_quant(p.quant_params.dst_scale, {1});
    }, w4.data(), nullptr},
            {"an f8_e8m0 destination zero point",
                    [&](matmul_params &p) {
        mx_quant(p.quant_params.dst_zp, {1});
    }, w4.data(), nullptr},
            {"f4_e2m1 weight scales",
                    [](matmul_params &p) {
        p.quant_params.wei_scale.dt = data_type_t::f4_e2m1;
    }, w4.data(), nullptr},
            {"an f8_e8m0 binary-add post-op", binary_add, w4.data(), nullptr},
            {"a bf16 call with an f8_e8m0 bias",
                    [&](matmul_params &p) {
        bf16_call(p);
        p.dtypes.bias = data_type_t::f8_e8m0;
    }, w16.data(), mx_buf.data()},
            {"a bf16 call with an f4_e2m1 destination",
                    [&](matmul_params &p) {
        bf16_call(p);
        p.dtypes.dst = data_type_t::f4_e2m1;
    }, w16.data(), nullptr},
            {"a bf16 call with an f8_e8m0 binary-add post-op",
                    [&](matmul_params &p) {
        bf16_call(p);
        binary_add(p);
    }, w16.data(), nullptr},
            {"a W4A8 call with an f8_e8m0 bias",
                    [&](matmul_params &p) {
        w4a8_call(p);
        p.dtypes.bias = data_type_t::f8_e8m0;
    }, w4.data(), mx_buf.data()},
            {"a W4A8 call with an f8_e8m0 weight zero point",
                    [&](matmul_params &p) {
        w4a8_call(p);
        mx_quant(p.quant_params.wei_zp, {K / kGroup, N});
    }, w4.data(), nullptr},
    };
    for (const Case &c : cases) {
        matmul_params p = mxfp4_params(e8.data(), K, N, src_scale, M);
        c.edit(p);
        const int ldb = K;

        std::fill(dst.begin(), dst.end(), kCanary);
        EXPECT_NE(matmul_direct('r', false, true, M, N, K, 1.0f, src.data(), K,
                          c.weight, ldb, c.bias, 0.0f, dst.data(), N, true,
                          matmul_batch_params_t {}, p),
                status_t::success)
                << "matmul_direct, " << c.name;
        EXPECT_TRUE(all_canary(dst)) << "matmul_direct, " << c.name;

        std::fill(dst.begin(), dst.end(), kCanary);
        EXPECT_NE(
                group_matmul_direct({'r', 'r'}, {false, false}, {true, true},
                        {M, M}, {N, N}, {K, K}, {1.0f, 1.0f},
                        {src.data(), src.data()}, {K, K}, {c.weight, c.weight},
                        {ldb, ldb}, {c.bias, c.bias}, {0.0f, 0.0f},
                        {dst.data(), dst.data()}, {N, N}, {true, true}, {p, p}),
                status_t::success)
                << "group_matmul_direct, " << c.name;
        EXPECT_TRUE(all_canary(dst)) << "group_matmul_direct, " << c.name;
    }
    expect_fused_down_mx_roles_refused();
}

// The custom kernel's prepack API packs bf16, f16 and s8; f4_e2m1, whose
// only backend is that kernel, is refused rather than sized and packed as
// bf16.
void expect_ck_prepack_refuses_e2m1() {
    constexpr int K = 256, N = 128;
    reorder_params_t rp;
    rp.is_prepack = true;
    rp.prepack.algo = matmul_algo_t::moe_custom_kernel;
    rp.prepack.wei_dtype = data_type_t::f4_e2m1;
    rp.prepack.src_dtype = data_type_t::bf16;
    rp.prepack.K = K;
    rp.prepack.N = N;
    rp.prepack.ldb = K;
    rp.prepack.transposed = true;
    EXPECT_EQ(weight_prepack_size(rp), 0u);
    const std::vector<uint8_t> codes(static_cast<size_t>(K) * N / 2, 0x11);
    std::vector<uint8_t> packed(static_cast<size_t>(K) * N * 2, 0xA5);
    EXPECT_NE(
            reorder_direct(codes.data(), packed.data(), rp), status_t::success);
    EXPECT_TRUE(std::all_of(
            packed.begin(), packed.end(), [](uint8_t b) { return b == 0xA5; }));
}

// reorder_direct has no conversion or quantization kernel for the MX types,
// and its dynamic quantization reads f32, bf16 or f16 sources only; each
// call fails with its destination and scales untouched, rather than reading
// the source as f16 or reporting a conversion it never wrote.
void expect_reorder_refuses_unreadable_types() {
    constexpr int M = 4, K = 1024;
    const std::vector<uint8_t> codes(static_cast<size_t>(M) * K / 2, 0x11);
    const std::vector<int8_t> src_s8(static_cast<size_t>(M) * K, 1);
    const std::vector<uint16_t> src_bf16(
            static_cast<size_t>(M) * K, f32_to_bf16(0.5f));
    struct Case {
        const char *name;
        data_type_t src, dst, scale;
        bool dynamic;
        bool compute_only;
    };
    const std::vector<Case> cases {
            {"dynamic quant f4_e2m1 -> s8", data_type_t::f4_e2m1,
                    data_type_t::s8, data_type_t::f32, true, false},
            {"dynamic quant f4_e2m1, scales only", data_type_t::f4_e2m1,
                    data_type_t::s8, data_type_t::f32, true, true},
            {"dynamic quant bf16 -> f4_e2m1", data_type_t::bf16,
                    data_type_t::f4_e2m1, data_type_t::f32, true, false},
            {"f4_e2m1 -> bf16", data_type_t::f4_e2m1, data_type_t::bf16,
                    data_type_t::f32, false, false},
            {"bf16 -> f4_e2m1", data_type_t::bf16, data_type_t::f4_e2m1,
                    data_type_t::f32, false, false},
            {"f8_e8m0 -> f32", data_type_t::f8_e8m0, data_type_t::f32,
                    data_type_t::f32, false, false},
            {"bf16 -> s8 with f8_e8m0 scales", data_type_t::bf16,
                    data_type_t::s8, data_type_t::f8_e8m0, false, false},
            {"dynamic quant of an s8 source", data_type_t::s8, data_type_t::s8,
                    data_type_t::f32, true, false},
    };
    for (const Case &c : cases) {
        reorder_params_t rp;
        rp.src_dtype = c.src;
        rp.dst_dtype = c.dst;
        rp.src_shape = {M, K};
        rp.dst_shape = {M, K};
        rp.dynamic_quant = c.dynamic;
        std::vector<float> scale(M, 0.25f);
        rp.quant_params.scale.buff = scale.data();
        rp.quant_params.scale.dt = c.scale;
        rp.quant_params.scale.dims = {M, 1};
        std::vector<uint16_t> dst(static_cast<size_t>(M) * K * 2, kCanary);
        const void *src = c.src == data_type_t::bf16
                ? static_cast<const void *>(src_bf16.data())
                : c.src == data_type_t::s8
                ? static_cast<const void *>(src_s8.data())
                : static_cast<const void *>(codes.data());
        EXPECT_NE(
                reorder_direct(src, c.compute_only ? nullptr : dst.data(), rp),
                status_t::success)
                << c.name;
        EXPECT_TRUE(all_canary(dst)) << c.name;
        EXPECT_TRUE(std::all_of(scale.begin(), scale.end(), [](float s) {
            return s == 0.25f;
        })) << c.name;
    }
}

// The checks on matmul_direct's MXFP4 gate that act even when validation
// is off: no batching, no layout but row-major, no per-call algorithm but
// the custom kernel's.  Each leaves the destination untouched.
void expect_matmul_direct_gate_refusals(int M) {
    reset_grp_matmul_caches();
    Mxfp4Call batched({M}, /*K=*/256, /*N=*/128, 91 + M);
    batched.fill_canary();
    EXPECT_NE(batched.run_direct(matmul_algo_t::none, 'r', /*batch=*/2),
            status_t::success)
            << "batch 2, M=" << M;
    EXPECT_TRUE(all_canary(batched.dst[0])) << "batch 2, M=" << M;

    reset_grp_matmul_caches();
    Mxfp4Call column({M}, /*K=*/256, /*N=*/128, 97 + M);
    column.fill_canary();
    EXPECT_NE(column.run_direct(matmul_algo_t::none, 'c'), status_t::success)
            << "layout 'c', M=" << M;
    EXPECT_TRUE(all_canary(column.dst[0])) << "layout 'c', M=" << M;

    reset_grp_matmul_caches();
    Mxfp4Call pinned({M}, /*K=*/256, /*N=*/128, 61 + M);
    pinned.fill_canary();
    EXPECT_NE(pinned.run_direct(matmul_algo_t::aocl_dlp_blocked),
            status_t::success)
            << "per-call aocl_dlp_blocked, M=" << M;
    EXPECT_TRUE(all_canary(pinned.dst[0]))
            << "per-call aocl_dlp_blocked, M=" << M;
}

// The rest of the contract, refused before the source quantization writes
// the caller's scales: destination quantization, which the kernel would
// drop; a source scale other than f32 or bf16, which the quantization would
// size by its type; a bias the kernel cannot add; and on a fused down
// projection, a zero point or such a bias, which Op2 would refuse only after
// Op1 has run.
void expect_contract_refusals() {
    moe_test_utils::AlgoEnvGuard auto_algo(0);
    moe_test_utils::CustomKernelOverride ck_on(true);
    float dst_scale = 0.5f;
    int8_t dst_zp = 3;
    const std::pair<const char *, std::function<void(matmul_params &)>> edits[]
            = {
                    {"dst_scale",
                            [&](matmul_params &q) {
        q.quant_params.dst_scale.buff = &dst_scale;
        q.quant_params.dst_scale.dt = data_type_t::f32;
        q.quant_params.dst_scale.dims = {1, 1};
    }},
                    {"dst_zp",
                            [&](matmul_params &q) {
        q.quant_params.dst_zp.buff = &dst_zp;
        q.quant_params.dst_zp.dt = data_type_t::s8;
        q.quant_params.dst_zp.dims = {1, 1};
    }},
                    {"s8 src_scale",
                            [](matmul_params &q) {
        q.quant_params.src_scale.dt = data_type_t::s8;
    }},
                    {"s8 bias",
                            [](matmul_params &q) {
        q.dtypes.bias = data_type_t::s8;
    }},
            };
    for (const auto &[name, edit] : edits) {
        reset_grp_matmul_caches();
        Mxfp4Call call({2, 2}, /*K=*/256, /*N=*/128, 401);
        std::vector<matmul_params> p = call.params();
        for (auto &q : p)
            edit(q);
        call.fill_canary();
        const std::vector<float> scales = call.src_scale[0];
        EXPECT_NE(call.run(p), status_t::success) << name << ", grouped";
        EXPECT_TRUE(all_canary(call.dst[0]) && all_canary(call.dst[1]))
                << name << ", grouped";
        EXPECT_EQ(call.src_scale[0], scales) << name << ", grouped";

        reset_grp_matmul_caches();
        Mxfp4Call direct({4}, /*K=*/256, /*N=*/128, 402);
        matmul_params q = direct.expert_params(0);
        edit(q);
        direct.fill_canary();
        EXPECT_NE(matmul_direct('r', false, true, 4, 128, 256, 1.0f,
                          direct.src_of(0), 256, direct.weights[0].codes.data(),
                          256, nullptr, 0.0f, direct.dst_of(0), 128, true,
                          matmul_batch_params_t {}, q),
                status_t::success)
                << name << ", matmul_direct";
        EXPECT_TRUE(all_canary(direct.dst[0])) << name << ", matmul_direct";
    }
    for (const bool zero_point : {true, false}) {
        const char *name = zero_point ? "s8 down zero point" : "s8 down bias";
        reset_grp_matmul_caches();
        Mxfp4FusedCall fused({4, 3}, 403 + zero_point);
        std::vector<int8_t> bytes(fused.N2, 1);
        if (zero_point) {
            fused.fused.down_zp.resize(2);
            for (auto &zp : fused.fused.down_zp) {
                zp.buff = bytes.data();
                zp.dt = data_type_t::s8;
                zp.dims = {1, fused.N2};
            }
        } else {
            fused.fused.bias_dt_down = data_type_t::s8;
            fused.fused.bias_down.assign(2, bytes.data());
        }
        const std::vector<float> scales = fused.op1.src_scale[0];
        EXPECT_NE(fused.run(), status_t::success) << name;
        EXPECT_TRUE(fused.outputs_untouched()) << name;
        EXPECT_EQ(fused.op1.src_scale[0], scales) << name << ": Op1 ran";
    }
    // A gated activation into a tight destination (ldc = N/2) whose store
    // the kernel cannot fuse: into f32, or silu or gelu with a bias.
    for (const auto act : kGatedActs)
        for (const bool f32_dst : {true, false}) {
            if (!f32_dst && act == grp_matmul_gated_act_t::swiglu_oai_mul)
                continue;
            Mxfp4Variant v;
            v.f32_dst = f32_dst;
            v.bias_dt = f32_dst ? data_type_t::none : data_type_t::bf16;
            const std::string name = std::string("tight ") + gated_act_name(act)
                    + (f32_dst ? " into f32" : " with a bias");
            reset_grp_matmul_caches();
            Mxfp4Call call({2, 2}, /*K=*/256, /*N=*/128, 405, v);
            call.fill_canary();
            const std::vector<float> scales = call.src_scale[0];
            EXPECT_NE(run_gated(call, act, 64), status_t::success)
                    << name << ", grouped";
            EXPECT_TRUE(call.dst_untouched()) << name << ", grouped";
            EXPECT_EQ(call.src_scale[0], scales) << name << ", grouped";
            EXPECT_NE(run_gated_direct(call, act, 64), status_t::success)
                    << name << ", matmul_direct";
            EXPECT_TRUE(call.dst_untouched()) << name << ", matmul_direct";
            EXPECT_EQ(call.src_scale[0], scales) << name << ", matmul_direct";
        }
}

// The refusals that keep MXFP4 off other backends, run in a process that
// starts with ZENDNNL_DIAGNOSTICS_ENABLE=0 (the gate is read once).
void expect_refusals_without_diagnostics() {
    moe_test_utils::AlgoEnvGuard auto_algo(0);
    moe_test_utils::CustomKernelOverride ck_on(true);

    // A sequential chain (one source, two ops) writes nothing.
    reset_grp_matmul_caches();
    constexpr int M = 4, K0 = 128, N0 = 128, N1 = 64;
    std::mt19937 rng(9);
    std::uniform_real_distribution<float> x(-1.0f, 1.0f);
    const Mxfp4Weight w0 = make_weight(rng, N0, K0);
    const Mxfp4Weight w1 = make_weight(rng, N1, N0);
    std::vector<uint16_t> src(static_cast<size_t>(M) * K0);
    for (auto &v : src) {
        v = f32_to_bf16(x(rng));
    }
    std::vector<uint16_t> d0(static_cast<size_t>(M) * N0, kCanary);
    std::vector<uint16_t> d1(static_cast<size_t>(M) * N1, kCanary);
    std::vector<float> ss0, ss1;
    const std::vector<matmul_params> chain {
            mxfp4_params(w0.e8m0.data(), K0, N0, ss0, M),
            mxfp4_params(w1.e8m0.data(), N0, N1, ss1, M)};
    EXPECT_NE(group_matmul_direct({'r', 'r'}, {false, false}, {true, true},
                      {M, M}, {N0, N1}, {K0, N0}, {1.0f, 1.0f}, {src.data()},
                      {K0, N0}, {w0.codes.data(), w1.codes.data()}, {K0, N0},
                      {nullptr, nullptr}, {0.0f, 0.0f}, {d0.data(), d1.data()},
                      {N0, N1}, {true, true}, chain),
            status_t::success)
            << "sequential chain";
    EXPECT_TRUE(all_canary(d0));
    EXPECT_TRUE(all_canary(d1));

    // Mixed weight types, with either expert first.
    for (const size_t other : {0, 1}) {
        reset_grp_matmul_caches();
        Mxfp4Call call({2, 2}, /*K=*/256, /*N=*/128, 53 + other);
        std::vector<matmul_params> p = call.params();
        std::vector<float> s4_scale;
        make_s4_expert(p, other, s4_scale, 256, 128);
        call.fill_canary();
        EXPECT_NE(call.run(p), status_t::success) << "s4 expert " << other;
        EXPECT_TRUE(all_canary(call.dst[1 - other])) << "s4 expert " << other;
    }

    // A scale descriptor other than f8_e8m0 {K/32, N}.
    reset_grp_matmul_caches();
    Mxfp4Call call({2, 2}, /*K=*/256, /*N=*/128, 59);
    std::vector<matmul_params> p = call.params();
    p[1].quant_params.wei_scale.dims = {256 / 16, 128};
    EXPECT_NE(call.run(p), status_t::success) << "scale dims {K/16, N}";

    for (const int rows : {1, 64})
        expect_matmul_direct_gate_refusals(rows);
    expect_contract_refusals();
    expect_malformed_down_scales_refused();
    expect_mx_roles_refused();
    expect_ck_prepack_refuses_e2m1();
    expect_reorder_refuses_unreadable_types();
}

} // namespace

TEST(GroupMatmulMxfp4, ContractRefusalsPrecedeAnyWork) {
    expect_contract_refusals();
}

TEST(GroupMatmulMxfp4, RefusalsHoldWithDiagnosticsOff) {
    testing::FLAGS_gtest_death_test_style = "threadsafe";
    setenv("ZENDNNL_DIAGNOSTICS_ENABLE", "0", /*overwrite=*/1);
    EXPECT_EXIT(
            {
                expect_refusals_without_diagnostics();
                std::exit(::testing::Test::HasFailure() ? 1 : 0);
            },
            ::testing::ExitedWithCode(0), "");
    unsetenv("ZENDNNL_DIAGNOSTICS_ENABLE");
}

// A one-op call takes the single-expert policy, which serves a dense
// prompt on the prompt-tile executor.
TEST(GroupMatmulMxfp4, MatmulDirectRoutesToTheCustomKernel) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    moe_test_utils::AlgoEnvGuard auto_algo(0);
    moe_test_utils::CustomKernelOverride ck_on(true);
    for (const int M : {1, 4, 64, 128, 512}) {
        const std::string label = "matmul_direct M=" + std::to_string(M);
        reset_grp_matmul_caches();
        Mxfp4Call call({M}, /*K=*/256, /*N=*/128, 53 + M);
        moe_test_utils::ExecutionPathCaptureGuard capture;
        ASSERT_EQ(call.run_direct(), status_t::success) << label;
        expect_custom_kernel_path(label);
        if (M >= 128) {
            const char *path = last_execution_path();
            EXPECT_NE(std::strstr(path, "ptile"), nullptr)
                    << label << ": a prompt takes the prompt-tile executor; "
                    << "execution_path=" << path;
        }
        call.expect_matches_reference(label);
    }
}

// A dense MXFP4 layer has no scalar fallback: batching, a column-major
// layout, a per-call algorithm naming a scalar backend, or a shape the
// kernel refuses, fails the call.
TEST(GroupMatmulMxfp4, MatmulDirectNeverLeavesTheCustomKernel) {
    moe_test_utils::AlgoEnvGuard auto_algo(0);
    moe_test_utils::CustomKernelOverride ck_on(true);
    for (const int M : {1, 64}) {
        expect_matmul_direct_gate_refusals(M);
        reset_grp_matmul_caches();
        Mxfp4Call refused({M}, /*K=*/256, /*N=*/80, 67 + M);
        EXPECT_NE(refused.run_direct(), status_t::success) << "N=80 M=" << M;
    }
}

// f4_e2m1 is MXFP4 only, with its f8_e8m0 block scales; f32 weight scales
// are not part of the contract.
TEST(GroupMatmulMxfp4, NonE8M0ScalesAreRefused) {
    moe_test_utils::AlgoEnvGuard auto_algo(0);
    reset_grp_matmul_caches();
    Mxfp4Call call({4}, /*K=*/256, /*N=*/128, 71);
    std::vector<float> f32_scale(call.weights[0].e8m0.size(), 1.0f);
    std::vector<matmul_params> p = call.params();
    p[0].quant_params.wei_scale.buff = f32_scale.data();
    p[0].quant_params.wei_scale.dt = data_type_t::f32;
    EXPECT_NE(call.run(p), status_t::success);
    EXPECT_NE(
            matmul_direct('r', false, true, 4, 128, 256, 1.0f,
                    call.src_bf16[0].data(), 256, call.weights[0].codes.data(),
                    256, nullptr, 0.0f, call.dst[0].data(), 128, true,
                    matmul_batch_params_t {}, p[0]),
            status_t::success);
}

TEST(GroupMatmulMxfp4, MxTypesOutsideTheirRolesAreRefused) {
    expect_mx_roles_refused();
}

TEST(GroupMatmulMxfp4, CustomKernelPrepackApiRefusesE2M1) {
    expect_ck_prepack_refuses_e2m1();
}

TEST(GroupMatmulMxfp4, ReorderDirectRefusesTypesItCannotRead) {
    expect_reorder_refuses_unreadable_types();
}

// The down projection is built inside the fused MoE call; it must read its
// own f4_e2m1 weight and f8_e8m0 scales.
TEST(GroupMatmulMxfp4, FusedMoeServesTheDownProjection) {
    if (!ck::avx512vnni_available()) GTEST_SKIP() << "no AVX-512 VNNI";
    moe_test_utils::AlgoEnvGuard auto_algo(0);
    moe_test_utils::CustomKernelOverride ck_on(true);
    reset_grp_matmul_caches();
    Mxfp4FusedCall call({8, 8, 8, 8}, 43);
    ASSERT_EQ(call.run(), status_t::success);
    call.expect_matches_reference();
}

TEST(GroupMatmulMxfp4, MalformedDownScalesFailTheFusedCall) {
    expect_malformed_down_scales_refused();
}

// The legacy operator API has no MXFP4 kernel: an f4_e2m1 tensor is refused
// whatever kernel is forced, rather than read as f32.
TEST(GroupMatmulMxfp4, LegacyOperatorsRefuseE2M1Tensors) {
    constexpr uint64_t M = 4, K = 256, N = 128;
    const auto make
            = [](const char *name, std::vector<uint64_t> size, data_type_t dt) {
        tensor_t t = tensor_t()
                             .set_name(name)
                             .set_size(size)
                             .set_data_type(dt)
                             .set_storage();
        t.create();
        return t;
    };
    tensor_t input = make("matmul_input", {M, K}, data_type_t::f32);
    tensor_t weights = make("weights", {K, N}, data_type_t::f4_e2m1);
    tensor_t output = make("matmul_output", {M, N}, data_type_t::f32);
    for (const std::string kernel : {"", "onednn", "reference", "aocl_dlp"}) {
        matmul_context_t context
                = matmul_context_t().set_param("weights", weights).create();
        matmul_operator_t op = matmul_operator_t()
                                       .set_name("mxfp4_legacy_matmul")
                                       .set_context(context)
                                       .create();
        if (op.is_bad_object()) continue;
        op.set_input("matmul_input", input).set_output("matmul_output", output);
        if (!kernel.empty()) op.set_forced_kernel(kernel);
        EXPECT_NE(op.execute(), status_t::success)
                << "kernel '" << kernel << "'";
    }

    reorder_context_t reorder_context
            = reorder_context_t().set_algo_format("aocl");
    reorder_context.create();
    tensor_t reordered = make("reorder_output", {K, N}, data_type_t::f4_e2m1);
    reorder_operator_t reorder = reorder_operator_t()
                                         .set_name("mxfp4_legacy_reorder")
                                         .set_context(reorder_context)
                                         .create()
                                         .set_input("reorder_input", weights);
    if (!reorder.is_bad_object()) {
        reorder.set_output("reorder_output", reordered);
        EXPECT_NE(reorder.execute(), status_t::success) << "reorder";
    }
}
